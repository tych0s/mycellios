-- Cierra la fuga de conversaciones entre cuentas (25-07-2026).
--
-- El fallo era la composición de tres piezas correctas por separado:
--
--   1. `inference_conversations` en el SQLite local NO tiene columna `user_id`
--      (ver `src/storage/database.ts`), así que la sincronización sólo puede
--      empujar filas con `user_id = null`.
--   2. `handle_new_user` (202607240003) inscribe a CADA alta como `viewer` de
--      la red pública `00000000-0000-0000-0000-000000000001`.
--   3. Las políticas de lectura de 202607240001 aceptaban `user_id is null`.
--
-- Resultado: cualquier persona que se registrase podía leer TODAS las
-- conversaciones y mensajes de la red, porque en la práctica todas llegan sin
-- dueño. La cláusula estaba pensada como "conversación compartida de la red",
-- pero como nada rellena `user_id`, abarcaba el conjunto entero.
--
-- Arreglo: una fila sin dueño deja de ser legible por `authenticated`. El
-- coordinador escribe con la service role, que no pasa por RLS, así que la
-- ingesta no se toca. Atribuir conversaciones a su dueño exige propagar el
-- `user_id` desde el runtime — es un cambio de esquema aguas arriba y va
-- aparte; hasta entonces, fail-closed.

begin;

drop policy if exists "conversations_read_member" on public.inference_conversations;
create policy "conversations_read_member" on public.inference_conversations
  for select to authenticated using (
    public.is_network_member(network_id)
    and user_id = auth.uid()
  );

drop policy if exists "messages_read_member" on public.inference_messages;
create policy "messages_read_member" on public.inference_messages
  for select to authenticated using (
    exists (
      select 1 from public.inference_conversations c
      where c.id = conversation_id
        and public.is_network_member(c.network_id)
        and c.user_id = auth.uid()
    )
  );

comment on policy "conversations_read_member" on public.inference_conversations is
  'Solo el dueno. Una conversacion con user_id nulo no es de nadie y no se '
  'expone a `authenticated` (antes lo era para toda la red).';

commit;
