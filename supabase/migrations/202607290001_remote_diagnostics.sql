begin;

create table if not exists public.diagnostic_events (
  id text primary key,
  network_id uuid not null default '00000000-0000-0000-0000-000000000001'
    references public.networks(id) on delete cascade,
  source_id uuid not null,
  app_version text not null,
  platform text not null,
  arch text not null,
  level text not null check (level in ('info', 'warning', 'error')),
  source text not null,
  event text not null,
  message text not null,
  details text,
  occurred_at bigint not null,
  received_at bigint not null
);

create index if not exists diagnostic_events_network_occurred
  on public.diagnostic_events(network_id, occurred_at desc);

create index if not exists diagnostic_events_source_occurred
  on public.diagnostic_events(source_id, occurred_at desc);

alter table public.diagnostic_events enable row level security;

create policy "diagnostics_read_admin" on public.diagnostic_events
  for select to authenticated using (
    exists (
      select 1
      from public.network_members m
      where m.network_id = diagnostic_events.network_id
        and m.user_id = auth.uid()
        and m.role in ('owner', 'admin')
    )
  );

grant select on public.diagnostic_events to authenticated;
grant all privileges on public.diagnostic_events to service_role;

commit;
