begin;

create table if not exists public.deployment_states (
  model_id text primary key references public.requested_models(id) on delete cascade,
  network_id uuid not null default '00000000-0000-0000-0000-000000000001'
    references public.networks(id) on delete cascade,
  desired_state text not null check (desired_state in ('active', 'inactive')),
  observed_state text not null check (observed_state in (
    'inactive', 'waiting_capacity', 'preparing', 'canary', 'active',
    'degraded', 'failed', 'stopping'
  )),
  generation integer not null default 1,
  observed_generation integer not null default 0,
  retry_count integer not null default 0,
  next_retry_at bigint,
  last_error text,
  active_operation_id text,
  controller_owner text,
  controller_lease_until bigint,
  created_at bigint not null,
  updated_at bigint not null
);

create index if not exists deployment_states_network_reconcile
  on public.deployment_states(
    network_id, desired_state, observed_state, next_retry_at, controller_lease_until
  );

create table if not exists public.deployment_operations (
  id text primary key,
  network_id uuid not null default '00000000-0000-0000-0000-000000000001'
    references public.networks(id) on delete cascade,
  model_id text not null references public.requested_models(id) on delete cascade,
  generation integer not null,
  kind text not null check (kind in ('activate', 'deactivate', 'repair', 'replan')),
  status text not null check (status in (
    'pending', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted'
  )),
  attempt integer not null default 1,
  idempotency_key text not null unique,
  error_code text,
  error_message text,
  metadata jsonb not null default '{}'::jsonb,
  started_at bigint not null,
  updated_at bigint not null,
  finished_at bigint
);

create index if not exists deployment_operations_network_model_started
  on public.deployment_operations(network_id, model_id, started_at desc);

create table if not exists public.route_reservations (
  id text primary key,
  network_id uuid not null default '00000000-0000-0000-0000-000000000001'
    references public.networks(id) on delete cascade,
  model_id text not null references public.requested_models(id) on delete cascade,
  operation_id text not null references public.deployment_operations(id) on delete cascade,
  generation integer not null,
  status text not null check (status in (
    'prepared', 'committed', 'released', 'expired', 'failed'
  )),
  route_digest text not null,
  stages jsonb not null,
  canary jsonb,
  expires_at bigint not null,
  committed_at bigint,
  released_at bigint,
  error text,
  created_at bigint not null,
  updated_at bigint not null
);

create index if not exists route_reservations_network_model_status
  on public.route_reservations(network_id, model_id, status, expires_at);

create table if not exists public.deployment_stage_leases (
  id text primary key,
  network_id uuid not null default '00000000-0000-0000-0000-000000000001'
    references public.networks(id) on delete cascade,
  reservation_id text not null references public.route_reservations(id) on delete cascade,
  model_id text not null references public.requested_models(id) on delete cascade,
  node_id text not null,
  stage_index integer not null,
  memory_mib integer not null,
  status text not null check (status in ('prepared', 'committed', 'released', 'expired')),
  expires_at bigint not null,
  created_at bigint not null,
  updated_at bigint not null,
  unique (reservation_id, node_id, stage_index)
);

create index if not exists deployment_stage_leases_network_node_status
  on public.deployment_stage_leases(network_id, node_id, status, expires_at);

alter table public.deployment_states enable row level security;
alter table public.deployment_operations enable row level security;
alter table public.route_reservations enable row level security;
alter table public.deployment_stage_leases enable row level security;

create policy "deployment_states_read_member" on public.deployment_states
  for select to authenticated using (public.is_network_member(network_id));
create policy "deployment_operations_read_member" on public.deployment_operations
  for select to authenticated using (public.is_network_member(network_id));
create policy "route_reservations_read_member" on public.route_reservations
  for select to authenticated using (public.is_network_member(network_id));
create policy "deployment_stage_leases_read_member" on public.deployment_stage_leases
  for select to authenticated using (public.is_network_member(network_id));

grant select on public.deployment_states, public.deployment_operations,
  public.route_reservations, public.deployment_stage_leases
to authenticated;
grant all privileges on public.deployment_states, public.deployment_operations,
  public.route_reservations, public.deployment_stage_leases
to service_role;

commit;
