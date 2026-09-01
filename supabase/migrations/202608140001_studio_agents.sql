create table if not exists public.studio_agents (
  id text primary key,
  owner_id uuid not null references auth.users(id) on delete cascade,
  create_idempotency_key text not null,
  template_id text check (template_id is null or template_id in ('concierge', 'researcher', 'developer')),
  status text not null check (status in ('draft', 'published', 'archived')),
  operational_state text not null check (operational_state in ('draft', 'validating', 'waiting_for_capacity', 'activating_model', 'ready', 'serving', 'degraded', 'unavailable', 'revoked')),
  draft_version integer not null check (draft_version > 0),
  configuration jsonb not null,
  published_revision_id text,
  created_at timestamptz not null,
  updated_at timestamptz not null,
  archived_at timestamptz,
  unique(owner_id, create_idempotency_key),
  unique(id, owner_id)
);

create index if not exists studio_agents_owner_updated on public.studio_agents(owner_id, status, updated_at desc);

create table if not exists public.studio_agent_revisions (
  id text primary key,
  agent_id text not null,
  owner_id uuid not null,
  revision integer not null check (revision > 0),
  configuration jsonb not null,
  configuration_digest text not null,
  created_at timestamptz not null,
  foreign key(agent_id, owner_id) references public.studio_agents(id, owner_id) on delete cascade,
  unique(agent_id, revision),
  unique(agent_id, configuration_digest),
  unique(id, agent_id, owner_id)
);

alter table public.studio_agents drop constraint if exists studio_agents_published_revision_fk;
alter table public.studio_agents add constraint studio_agents_published_revision_fk
  foreign key(published_revision_id, id, owner_id)
  references public.studio_agent_revisions(id, agent_id, owner_id)
  deferrable initially deferred;

create table if not exists public.studio_channel_deployments (
  id text primary key,
  agent_id text not null,
  revision_id text not null,
  owner_id uuid not null,
  channel text not null check(channel in ('web', 'telegram', 'api')),
  state text not null check(state in ('waiting_for_capacity', 'ready', 'degraded', 'revoked')),
  public_id text not null unique,
  publish_idempotency_key text not null,
  created_at timestamptz not null,
  updated_at timestamptz not null,
  revoked_at timestamptz,
  foreign key(revision_id, agent_id, owner_id) references public.studio_agent_revisions(id, agent_id, owner_id),
  unique(owner_id, publish_idempotency_key, channel)
);

create unique index if not exists studio_deployments_active_channel
  on public.studio_channel_deployments(agent_id, channel) where revoked_at is null;

create table if not exists public.studio_agent_events (
  event_digest text primary key,
  sequence bigint not null,
  agent_id text not null,
  owner_id uuid not null,
  event_type text not null,
  details jsonb not null,
  previous_event_digest text,
  occurred_at timestamptz not null,
  foreign key(agent_id, owner_id) references public.studio_agents(id, owner_id) on delete cascade
);

create or replace function public.reject_studio_immutable_mutation() returns trigger
language plpgsql as $$ begin raise exception 'immutable_studio_record'; end $$;
drop trigger if exists studio_revision_update_guard on public.studio_agent_revisions;
create trigger studio_revision_update_guard before update or delete on public.studio_agent_revisions
for each row execute function public.reject_studio_immutable_mutation();

alter table public.studio_agents enable row level security;
alter table public.studio_agent_revisions enable row level security;
alter table public.studio_channel_deployments enable row level security;
alter table public.studio_agent_events enable row level security;

create policy "studio_agents_owner" on public.studio_agents for select to authenticated using(owner_id = auth.uid());
create policy "studio_revisions_owner" on public.studio_agent_revisions for select to authenticated using(owner_id = auth.uid());
create policy "studio_deployments_owner" on public.studio_channel_deployments for select to authenticated using(owner_id = auth.uid());
create policy "studio_events_owner" on public.studio_agent_events for select to authenticated using(owner_id = auth.uid());

grant select on public.studio_agents, public.studio_agent_revisions, public.studio_channel_deployments, public.studio_agent_events to authenticated;
grant all on public.studio_agents, public.studio_agent_revisions, public.studio_channel_deployments, public.studio_agent_events to service_role;

create table if not exists public.studio_knowledge_sources (
  id text primary key, agent_id text not null, owner_id uuid not null, name text not null,
  media_type text not null check(media_type in ('text/plain', 'text/markdown')),
  content_sha256 text not null, size_bytes integer not null check(size_bytes between 0 and 1000000),
  state text not null check(state in ('ready', 'failed', 'deleted')),
  created_at timestamptz not null, updated_at timestamptz not null, deleted_at timestamptz,
  foreign key(agent_id, owner_id) references public.studio_agents(id, owner_id) on delete cascade,
  unique(agent_id, content_sha256)
);
create table if not exists public.studio_knowledge_chunks (
  id text primary key, source_id text not null references public.studio_knowledge_sources(id) on delete cascade,
  agent_id text not null, owner_id uuid not null, ordinal integer not null check(ordinal >= 0),
  content text not null, content_sha256 text not null, created_at timestamptz not null,
  unique(source_id, ordinal)
);
create table if not exists public.studio_memory_facts (
  id text primary key, agent_id text not null, owner_id uuid not null, subject_id text not null,
  fact text not null, status text not null check(status in ('proposed','approved','rejected','expired','deleted')),
  origin text not null, confidence double precision not null check(confidence between 0 and 1),
  expires_at timestamptz, created_at timestamptz not null, updated_at timestamptz not null, deleted_at timestamptz,
  foreign key(agent_id, owner_id) references public.studio_agents(id, owner_id) on delete cascade
);
create table if not exists public.studio_tool_audit (
  id text primary key, agent_id text not null, owner_id uuid not null, tool_id text not null,
  input_digest text not null, outcome text not null, output jsonb, error_code text,
  duration_ms integer not null, created_at timestamptz not null,
  foreign key(agent_id, owner_id) references public.studio_agents(id, owner_id) on delete cascade
);
create table if not exists public.studio_invocations (
  id text primary key, deployment_id text not null references public.studio_channel_deployments(id),
  agent_id text not null, owner_id uuid not null, idempotency_key text not null, request_digest text not null,
  status text not null check(status in ('pending','completed','failed')), response jsonb, error_code text, usage jsonb,
  created_at timestamptz not null, updated_at timestamptz not null,
  foreign key(agent_id, owner_id) references public.studio_agents(id, owner_id) on delete cascade,
  unique(deployment_id, idempotency_key)
);
create table if not exists public.studio_telegram_policies (
  deployment_id text primary key references public.studio_channel_deployments(id) on delete cascade,
  secret_digest text not null, allowed_chats jsonb not null, updated_at timestamptz not null
);
create table if not exists public.studio_telegram_updates (
  id text primary key, deployment_id text not null references public.studio_channel_deployments(id) on delete cascade,
  update_id text not null, chat_id text not null, request jsonb not null, state text not null,
  response jsonb, created_at timestamptz not null, updated_at timestamptz not null,
  unique(deployment_id, update_id)
);

alter table public.studio_knowledge_sources enable row level security;
alter table public.studio_knowledge_chunks enable row level security;
alter table public.studio_memory_facts enable row level security;
alter table public.studio_tool_audit enable row level security;
alter table public.studio_invocations enable row level security;
alter table public.studio_telegram_policies enable row level security;
alter table public.studio_telegram_updates enable row level security;
create policy "studio_sources_owner" on public.studio_knowledge_sources for select to authenticated using(owner_id = auth.uid());
create policy "studio_chunks_owner" on public.studio_knowledge_chunks for select to authenticated using(owner_id = auth.uid());
create policy "studio_memory_owner" on public.studio_memory_facts for select to authenticated using(owner_id = auth.uid());
create policy "studio_tool_audit_owner" on public.studio_tool_audit for select to authenticated using(owner_id = auth.uid());
create policy "studio_invocations_owner" on public.studio_invocations for select to authenticated using(owner_id = auth.uid());
grant select on public.studio_knowledge_sources, public.studio_knowledge_chunks, public.studio_memory_facts, public.studio_tool_audit, public.studio_invocations to authenticated;
grant all on public.studio_knowledge_sources, public.studio_knowledge_chunks, public.studio_memory_facts, public.studio_tool_audit, public.studio_invocations, public.studio_telegram_policies, public.studio_telegram_updates to service_role;
