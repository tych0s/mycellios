begin;

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

  insert into public.network_members (network_id, user_id, role)
  values ('00000000-0000-0000-0000-000000000001', new.id, 'viewer')
  on conflict (network_id, user_id) do nothing;

  return new;
end;
$$;

commit;
