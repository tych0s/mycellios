begin;

create extension if not exists pgcrypto;

create table if not exists public.networks (
  id uuid primary key default gen_random_uuid(),
  slug text not null unique,
  name text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

insert into public.networks (id, slug, name)
values ('00000000-0000-0000-0000-000000000001', 'public', 'Mycellios public network')
on conflict (id) do update set slug = excluded.slug, name = excluded.name;

create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  avatar_url text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.network_members (
  network_id uuid not null references public.networks(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  role text not null check (role in ('owner', 'admin', 'operator', 'viewer')),
  created_at timestamptz not null default now(),
  primary key (network_id, user_id)
);

create table if not exists public.workers (
  id text primary key,
  network_id uuid not null default '00000000-0000-0000-0000-000000000001'
    references public.networks(id) on delete cascade,
  status text not null,
  capabilities_json jsonb not null,
  reliability double precision not null default 0.95,
  jobs_completed bigint not null default 0,
  last_seen_at bigint not null,
  created_at bigint not null,
  updated_at bigint not null,
  deregistered boolean not null default false,
  identity_kind text,
  identity_id text,
  unique (network_id, identity_kind, identity_id)
);

create index if not exists workers_network_status_seen
  on public.workers(network_id, status, last_seen_at desc);

create table if not exists public.requested_models (
  id text primary key,
  network_id uuid not null default '00000000-0000-0000-0000-000000000001'
    references public.networks(id) on delete cascade,
  source text not null,
  revision text,
  context_tokens integer not null,
  minimum_nodes integer not null,
  auto_activate boolean not null default true,
  profile_json jsonb,
  profile_error text,
  activation_requested_at bigint,
  activation_error text,
  created_at bigint not null,
  updated_at bigint not null
);

create table if not exists public.jobs (
  id text primary key,
  network_id uuid not null default '00000000-0000-0000-0000-000000000001'
    references public.networks(id) on delete cascade,
  session_id text not null,
  model text not null,
  workload_class text not null,
  status text not null,
  worker_id text references public.workers(id) on delete set null,
  deployment_id text,
  model_digest text,
  lease_id text,
  input_tokens integer not null default 0,
  output_tokens integer not null default 0,
  failure_code text,
  deadline_at bigint not null,
  created_at bigint not null,
  updated_at bigint not null
);

create index if not exists jobs_network_status_created
  on public.jobs(network_id, status, created_at desc);
create index if not exists jobs_worker_status
  on public.jobs(worker_id, status);

create table if not exists public.idempotency_keys (
  idempotency_key text primary key,
  network_id uuid not null default '00000000-0000-0000-0000-000000000001'
    references public.networks(id) on delete cascade,
  request_hash text not null,
  job_id text not null references public.jobs(id) on delete cascade,
  created_at bigint not null
);

create table if not exists public.sessions (
  id text primary key,
  network_id uuid not null default '00000000-0000-0000-0000-000000000001'
    references public.networks(id) on delete cascade,
  model text not null,
  route_json jsonb not null,
  expires_at bigint not null,
  last_used_at bigint not null,
  version integer not null default 1
);

create index if not exists sessions_network_expiry
  on public.sessions(network_id, expires_at);

create table if not exists public.worker_events (
  id uuid primary key default gen_random_uuid(),
  network_id uuid not null default '00000000-0000-0000-0000-000000000001'
    references public.networks(id) on delete cascade,
  worker_id text references public.workers(id) on delete set null,
  kind text not null,
  payload jsonb not null default '{}'::jsonb,
  occurred_at bigint not null,
  created_at timestamptz not null default now()
);

create index if not exists worker_events_worker_occurred
  on public.worker_events(worker_id, occurred_at desc);

create table if not exists public.activation_events (
  id text primary key,
  network_id uuid not null default '00000000-0000-0000-0000-000000000001'
    references public.networks(id) on delete cascade,
  model_id text not null,
  phase text not null,
  state text not null,
  message text not null,
  node_id text,
  process_id text,
  device text,
  details jsonb,
  occurred_at bigint not null,
  created_at timestamptz not null default now()
);

create index if not exists activation_events_model_occurred
  on public.activation_events(model_id, occurred_at desc);

create table if not exists public.benchmark_runs (
  run_id text primary key,
  network_id uuid not null default '00000000-0000-0000-0000-000000000001'
    references public.networks(id) on delete cascade,
  version text not null,
  label text not null,
  status text not null,
  trigger text,
  trigger_model_id text,
  started_at timestamptz not null,
  finished_at timestamptz not null,
  document jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists benchmark_runs_network_finished
  on public.benchmark_runs(network_id, finished_at desc);

create table if not exists public.inference_conversations (
  id text primary key,
  network_id uuid not null default '00000000-0000-0000-0000-000000000001'
    references public.networks(id) on delete cascade,
  user_id uuid references auth.users(id) on delete set null,
  session_id text not null,
  model text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (network_id, session_id)
);

create table if not exists public.inference_messages (
  id text primary key,
  conversation_id text not null references public.inference_conversations(id) on delete cascade,
  job_id text references public.jobs(id) on delete set null,
  role text not null check (role in ('system', 'developer', 'user', 'assistant', 'tool')),
  content text not null,
  status text not null default 'completed',
  input_tokens integer,
  output_tokens integer,
  route_class text,
  latency_ms integer,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists inference_messages_conversation_created
  on public.inference_messages(conversation_id, created_at);

create table if not exists public.artifacts (
  id text primary key,
  network_id uuid not null default '00000000-0000-0000-0000-000000000001'
    references public.networks(id) on delete cascade,
  kind text not null,
  storage_bucket text not null,
  storage_path text not null,
  sha256 text not null,
  size_bytes bigint not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.audit_log (
  id uuid primary key default gen_random_uuid(),
  network_id uuid not null default '00000000-0000-0000-0000-000000000001'
    references public.networks(id) on delete cascade,
  actor_user_id uuid references auth.users(id) on delete set null,
  actor_kind text not null,
  action text not null,
  target_type text,
  target_id text,
  payload jsonb not null default '{}'::jsonb,
  occurred_at timestamptz not null default now()
);

create index if not exists audit_log_network_occurred
  on public.audit_log(network_id, occurred_at desc);

create or replace function public.is_network_member(target_network_id uuid)
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select exists (
    select 1
    from public.network_members
    where network_id = target_network_id
      and user_id = auth.uid()
  );
$$;

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  insert into public.profiles (id, display_name)
  values (new.id, coalesce(new.raw_user_meta_data ->> 'display_name', split_part(new.email, '@', 1)))
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();

alter table public.networks enable row level security;
alter table public.profiles enable row level security;
alter table public.network_members enable row level security;
alter table public.workers enable row level security;
alter table public.requested_models enable row level security;
alter table public.jobs enable row level security;
alter table public.idempotency_keys enable row level security;
alter table public.sessions enable row level security;
alter table public.worker_events enable row level security;
alter table public.activation_events enable row level security;
alter table public.benchmark_runs enable row level security;
alter table public.inference_conversations enable row level security;
alter table public.inference_messages enable row level security;
alter table public.artifacts enable row level security;
alter table public.audit_log enable row level security;

create policy "profiles_read_self" on public.profiles
  for select to authenticated using (id = auth.uid());
create policy "profiles_update_self" on public.profiles
  for update to authenticated using (id = auth.uid()) with check (id = auth.uid());
create policy "members_read_own" on public.network_members
  for select to authenticated using (user_id = auth.uid());
create policy "networks_read_member" on public.networks
  for select to authenticated using (public.is_network_member(id));
create policy "workers_read_member" on public.workers
  for select to authenticated using (public.is_network_member(network_id));
create policy "models_read_member" on public.requested_models
  for select to authenticated using (public.is_network_member(network_id));
create policy "jobs_read_member" on public.jobs
  for select to authenticated using (public.is_network_member(network_id));
create policy "sessions_read_member" on public.sessions
  for select to authenticated using (public.is_network_member(network_id));
create policy "worker_events_read_member" on public.worker_events
  for select to authenticated using (public.is_network_member(network_id));
create policy "activation_events_read_member" on public.activation_events
  for select to authenticated using (public.is_network_member(network_id));
create policy "benchmarks_read_member" on public.benchmark_runs
  for select to authenticated using (public.is_network_member(network_id));
create policy "conversations_read_member" on public.inference_conversations
  for select to authenticated using (
    public.is_network_member(network_id)
    and (user_id is null or user_id = auth.uid())
  );
create policy "messages_read_member" on public.inference_messages
  for select to authenticated using (
    exists (
      select 1 from public.inference_conversations c
      where c.id = conversation_id
        and public.is_network_member(c.network_id)
        and (c.user_id is null or c.user_id = auth.uid())
    )
  );
create policy "artifacts_read_member" on public.artifacts
  for select to authenticated using (public.is_network_member(network_id));
create policy "audit_read_admin" on public.audit_log
  for select to authenticated using (
    exists (
      select 1 from public.network_members m
      where m.network_id = audit_log.network_id
        and m.user_id = auth.uid()
        and m.role in ('owner', 'admin')
    )
  );

insert into storage.buckets (id, name, public, file_size_limit)
values ('mycellios-artifacts', 'mycellios-artifacts', false, 5368709120)
on conflict (id) do update
set public = excluded.public, file_size_limit = excluded.file_size_limit;

grant usage on schema public to authenticated, service_role;
grant select on public.networks, public.profiles, public.network_members,
  public.workers, public.requested_models, public.jobs, public.sessions,
  public.worker_events, public.activation_events, public.benchmark_runs,
  public.inference_conversations, public.inference_messages, public.artifacts,
  public.audit_log
to authenticated;
grant update on public.profiles to authenticated;
grant all privileges on all tables in schema public to service_role;
grant all privileges on all sequences in schema public to service_role;
grant execute on function public.is_network_member(uuid) to authenticated, service_role;

commit;
