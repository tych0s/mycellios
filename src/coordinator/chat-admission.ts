import { createHash } from "node:crypto";
import type { ServerResponse } from "node:http";
import { setTimeout as delay } from "node:timers/promises";
import type { ChatCompletionRequest } from "../contracts/types.js";
import { inputHashForRequest } from "../core/request.js";
import { MeshServiceError, type MeshService } from "./mesh-service.js";

export async function waitForChatCapacity(
  service: MeshService,
  request: ChatCompletionRequest,
  sessionId: string | undefined,
  timeoutMs: number,
  response?: ServerResponse,
): Promise<void> {
  const deadline = Date.now() + Math.max(0, timeoutMs);
  const controller = new AbortController();
  const disconnected = () => controller.abort(new MeshServiceError(
    "client_disconnected", "The client disconnected before inference was admitted", 499,
  ));
  response?.once("close", disconnected);
  if (response?.destroyed) disconnected();
  try {
    controller.signal.throwIfAborted();
    while (!service.hasCapacity(request, sessionId) && Date.now() < deadline) {
      await delay(Math.min(250, Math.max(1, deadline - Date.now())), undefined, { signal: controller.signal });
    }
    controller.signal.throwIfAborted();
  } catch (error) {
    if (controller.signal.aborted) throw controller.signal.reason;
    throw error;
  } finally {
    response?.off("close", disconnected);
  }
}

type ChatPrincipal = { kind: "system" } | { kind: "user" | "api_key"; userId: string };

export function resolveChatIdempotencyKey(
  service: MeshService,
  request: ChatCompletionRequest,
  received: string | string[] | undefined,
  principal: ChatPrincipal,
): string | undefined {
  const key = parseIdempotencyKey(received, principal);
  let existing = key ? service.store.getIdempotentJob(key) : null;
  if (!existing && key && principal.kind !== "system") {
    const legacy = service.store.getIdempotentJob(parseIdempotencyKey(received)!);
    // Before account scoping, keys were stored verbatim. Preserve retry
    // protection only for accounts already associated with that legacy job.
    // Session ownership is checked before this compatibility lookup.
    if (legacy && service.store.database.raw.prepare(
      "SELECT 1 FROM api_usage WHERE job_id = ? AND user_id = ? LIMIT 1",
    ).get(legacy.jobId, principal.userId)) existing = legacy;
  }
  if (!existing) return key;
  throw new MeshServiceError(
    existing.requestHash === inputHashForRequest(request) ? "idempotency_replayed" : "idempotency_key_reused",
    `Idempotency key is already bound to job ${existing.jobId}`,
    409,
  );
}

export function parseIdempotencyKey(
  received: string | string[] | undefined,
  principal?: ChatPrincipal,
): string | undefined {
  const value = Array.isArray(received) ? received[0] : received;
  if (value === undefined) return undefined;
  if (!/^[\x21-\x7E]{1,128}$/.test(value)) {
    throw new MeshServiceError(
      "invalid_idempotency_key",
      "Idempotency-Key must contain 1-128 printable ASCII characters",
      400,
    );
  }
  if (!principal || principal.kind === "system") return value;
  // 136 characters cannot collide with any legacy/external key (at most 128).
  return `account-${createHash("sha512").update(JSON.stringify([principal.userId, value])).digest("hex")}`;
}
