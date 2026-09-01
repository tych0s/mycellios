begin;

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
