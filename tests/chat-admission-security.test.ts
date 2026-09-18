import { createHash } from "node:crypto";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ChatCompletionRequest } from "../src/contracts/types.js";
import { chatCompletionRequestSchema } from "../src/contracts/schemas.js";
import { parseIdempotencyKey } from "../src/coordinator/chat-admission.js";
import { createCoordinator, type CoordinatorRuntime } from "../src/coordinator/server.js";
import { inputHashForRequest } from "../src/core/request.js";
import { addWorker } from "./helpers.js";

const runtimes: CoordinatorRuntime[] = [];
afterEach(async () => {
  vi.restoreAllMocks();
  await Promise.all(runtimes.splice(0).map((runtime) => runtime.close()));
});

const prompt: ChatCompletionRequest = {
  model: "private-model",
  messages: [{ role: "user", content: "A private account prompt" }],
  max_tokens: 4,
  stream: false,
  session_id: "known-session",
};

async function coordinator(requestTimeoutMs = 1_000) {
  const runtime = await createCoordinator({
    host: "127.0.0.1", port: 0, databasePath: ":memory:", requestTimeoutMs,
    apiAccessEnabled: true, apiStarterTokens: 10_000,
  }, { logger: false });
  runtimes.push(runtime);
  return runtime;
}

function ownSession(runtime: CoordinatorRuntime, userId: string, sessionId: string) {
  const reservation = runtime.apiAccess.beginUsage(userId, null, prompt);
  runtime.apiAccess.attachJob(reservation.id, `job-${userId}`, sessionId);
  runtime.apiAccess.completeUsage(reservation.id, 1, 1);
  runtime.store.startInferenceConversation(sessionId, prompt.model, prompt.messages);
}

function legacyIdempotency(runtime: CoordinatorRuntime, userId: string, rawKey = "legacy-request") {
  const request = chatCompletionRequestSchema.parse(prompt) as ChatCompletionRequest;
  runtime.store.createJob({ id: "legacy-job", sessionId: "known-session", model: request.model,
    workloadClass: "interactive", deadlineAt: Date.now() + 1_000 });
  runtime.store.bindIdempotencyKey(rawKey, inputHashForRequest(request), "legacy-job");
  const usage = runtime.apiAccess.beginUsage(userId, null, request);
  runtime.apiAccess.attachJob(usage.id, "legacy-job", "known-session");
  runtime.apiAccess.completeUsage(usage.id, 1, 1);
}

describe("authenticated chat admission", () => {
  it("rejects another account's session before cancelling or reserving usage", async () => {
    const runtime = await coordinator();
    ownSession(runtime, "victim", "known-session");
    const key = runtime.apiAccess.createKey("attacker", "Isolation regression");
    const cancel = vi.spyOn(runtime.service, "cancelMatchingActiveSession");
    const submit = vi.spyOn(runtime.service, "submit");
    const response = await runtime.app.inject({
      method: "POST", url: "/v1/chat/completions", payload: prompt,
      headers: { authorization: `Bearer ${key.secret}` },
    });
    expect(response.statusCode).toBe(404);
    expect(response.json()).toMatchObject({ error: { code: "session_not_found" } });
    expect(cancel).not.toHaveBeenCalled();
    expect(submit).not.toHaveBeenCalled();
    expect(runtime.apiAccess.listUsage("attacker")).toHaveLength(0);
    expect(runtime.apiAccess.userOwnsSession("attacker", "known-session")).toBe(false);
    const history = await runtime.app.inject({
      method: "GET", url: "/v1/conversations/known-session/messages",
      headers: { authorization: `Bearer ${key.secret}` },
    });
    expect(history.statusCode).toBe(404);
  });

  it("checks session ownership again after waiting for capacity", async () => {
    const runtime = await coordinator();
    const key = runtime.apiAccess.createKey("attacker", "Admission race regression");
    vi.spyOn(runtime.service, "hasCapacity").mockReturnValueOnce(false).mockImplementation(() => {
      ownSession(runtime, "victim", "known-session");
      return true;
    });
    const submit = vi.spyOn(runtime.service, "submit");
    const response = await runtime.app.inject({
      method: "POST", url: "/v1/chat/completions", payload: { ...prompt, stream: true },
      headers: { authorization: `Bearer ${key.secret}` },
    });
    expect(response.statusCode).toBe(404);
    expect(submit).not.toHaveBeenCalled();
    expect(runtime.apiAccess.listUsage("attacker")).toHaveLength(0);
  });

  it("preserves admission for the owner of an existing conversation", async () => {
    const runtime = await coordinator();
    ownSession(runtime, "owner", "known-session");
    const key = runtime.apiAccess.createKey("owner", "Owner regression");
    const response = await runtime.app.inject({
      method: "POST", url: "/v1/chat/completions", payload: prompt,
      headers: { authorization: `Bearer ${key.secret}` },
    });
    expect(response.statusCode).toBe(503);
    expect(response.json()).toMatchObject({ error: { code: "no_capacity" } });
  });

  it("does not let an account claim an unowned system conversation", async () => {
    const runtime = await coordinator();
    runtime.store.startInferenceConversation("known-session", prompt.model, prompt.messages);
    const key = runtime.apiAccess.createKey("attacker", "System conversation regression");
    const response = await runtime.app.inject({
      method: "POST", url: "/v1/chat/completions", payload: prompt,
      headers: { authorization: `Bearer ${key.secret}` },
    });
    expect(response.statusCode).toBe(404);
  });

  it("scopes idempotency keys to accounts while retaining same-account replay protection", async () => {
    const runtime = await coordinator();
    const first = runtime.apiAccess.createKey("first", "First account");
    const second = runtime.apiAccess.createKey("second", "Second account");
    const send = (secret: string) => runtime.app.inject({
      method: "POST", url: "/v1/chat/completions",
      payload: { ...prompt, session_id: secret === first.secret ? "first-session" : "second-session" },
      headers: { authorization: `Bearer ${secret}`, "idempotency-key": "request-1" },
    });
    expect((await send(first.secret)).statusCode).toBe(503);
    expect((await send(second.secret)).statusCode).toBe(503);
    const replay = await send(first.secret);
    expect(replay.statusCode).toBe(409);
    expect(replay.json()).toMatchObject({ error: { code: "idempotency_replayed" } });
  });

  it("returns the public error envelope for malformed chat input", async () => {
    const runtime = await coordinator();
    const key = runtime.apiAccess.createKey("caller", "Validation regression");
    const response = await runtime.app.inject({
      method: "POST", url: "/v1/chat/completions", payload: { model: "private-model" },
      headers: { authorization: `Bearer ${key.secret}` },
    });
    expect(response.statusCode).toBe(400);
    expect(response.json()).toMatchObject({ error: { code: "invalid_request" } });
  });

  it.each([false, true])("preserves legacy account replay protection after an upgrade (changed prompt: %s)", async (changed) => {
    const runtime = await coordinator();
    legacyIdempotency(runtime, "owner");
    const key = runtime.apiAccess.createKey("owner", "New key for the existing account");
    const submit = vi.spyOn(runtime.service, "submit");
    const response = await runtime.app.inject({
      method: "POST", url: "/v1/chat/completions",
      payload: { ...prompt, session_id: "fresh-session",
        ...(changed ? { messages: [{ role: "user", content: "Different prompt" }] } : {}) },
      headers: { authorization: `Bearer ${key.secret}`, "idempotency-key": "legacy-request" },
    });
    expect(response.statusCode).toBe(409);
    expect(response.json()).toMatchObject({ error: { code: changed ? "idempotency_key_reused" : "idempotency_replayed" } });
    expect(submit).not.toHaveBeenCalled();
    expect(runtime.apiAccess.listUsage("owner")).toHaveLength(1);
  });

  it("does not let a legacy key belonging to another account block a new request", async () => {
    const runtime = await coordinator();
    legacyIdempotency(runtime, "original-owner");
    const key = runtime.apiAccess.createKey("different-owner", "Independent account");
    const response = await runtime.app.inject({
      method: "POST", url: "/v1/chat/completions", payload: { ...prompt, session_id: "fresh-session" },
      headers: { authorization: `Bearer ${key.secret}`, "idempotency-key": "legacy-request" },
    });
    expect(response.statusCode).toBe(503);
    expect(response.json()).toMatchObject({ error: { code: "no_capacity" } });
    expect(response.body).not.toContain("legacy-job");
    expect(runtime.apiAccess.listUsage("different-owner")).toHaveLength(1);
  });

  it("keeps legacy retry compatibility behind mixed session ownership checks", async () => {
    const runtime = await coordinator();
    legacyIdempotency(runtime, "first-owner");
    ownSession(runtime, "second-owner", "known-session");
    const submit = vi.spyOn(runtime.service, "submit");
    for (const userId of ["first-owner", "second-owner"]) {
      const key = runtime.apiAccess.createKey(userId, "Legacy mixed ownership");
      const response = await runtime.app.inject({
        method: "POST", url: "/v1/chat/completions", payload: prompt,
        headers: { authorization: `Bearer ${key.secret}`, "idempotency-key": "legacy-request" },
      });
      expect(response.statusCode).toBe(404);
      expect(response.json()).toMatchObject({ error: { code: "session_not_found" } });
      expect(runtime.apiAccess.listUsage(userId)).toHaveLength(1);
    }
    expect(submit).not.toHaveBeenCalled();
  });

  it.each(["current-owner", "other-owner"])("cannot collide with a legacy raw key from %s that resembles the account namespace", async (legacyOwner) => {
    const runtime = await coordinator();
    const legacyKey = `account-${createHash("sha256").update(JSON.stringify(["current-owner", "foo"])).digest("hex")}`;
    legacyIdempotency(runtime, legacyOwner, legacyKey);
    const key = runtime.apiAccess.createKey("current-owner", "Namespace collision regression");
    const scoped = parseIdempotencyKey("foo", { kind: "api_key", userId: "current-owner" });
    expect(scoped).toHaveLength(136);
    expect(() => parseIdempotencyKey(scoped)).toThrow("Idempotency-Key must contain 1-128 printable ASCII characters");
    const response = await runtime.app.inject({
      method: "POST", url: "/v1/chat/completions", payload: { ...prompt, session_id: "fresh-session" },
      headers: { authorization: `Bearer ${key.secret}`, "idempotency-key": "foo" },
    });
    expect(response.statusCode).toBe(503);
    expect(response.json()).toMatchObject({ error: { code: "no_capacity" } });
    expect(response.body).not.toContain("legacy-job");
    expect(runtime.store.getIdempotentJob(scoped!)?.jobId).not.toBe("legacy-job");
    expect(runtime.store.getIdempotentJob(scoped!)).not.toBeNull();
  });

  it("does not cancel an in-flight request when its idempotency key is retried", async () => {
    const runtime = await coordinator(10_000);
    const key = runtime.apiAccess.createKey("owner", "Active idempotency regression");
    const worker = addWorker(runtime.store, { id: "active", model: prompt.model });
    vi.spyOn(runtime.hub, "connectedWorkerIds").mockReturnValue(new Set([worker.id]));
    vi.spyOn(runtime.hub, "send").mockReturnValue(true);
    const parsed = chatCompletionRequestSchema.parse(prompt) as ChatCompletionRequest;
    const active = runtime.service.submit(parsed, parsed.session_id,
      parseIdempotencyKey("active-request", { kind: "api_key", userId: "owner" }));
    ownSession(runtime, "owner", active.sessionId);
    const cancel = vi.spyOn(runtime.service, "cancelMatchingActiveSession");
    const capacity = vi.spyOn(runtime.service, "hasCapacity").mockReturnValue(false);
    try {
      const response = await runtime.app.inject({
        method: "POST", url: "/v1/chat/completions", payload: { ...prompt, stream: true },
        headers: { authorization: `Bearer ${key.secret}`, "idempotency-key": "active-request" },
      });
      expect(response.statusCode).toBe(409);
      expect(response.json()).toMatchObject({ error: { code: "idempotency_replayed" } });
      expect(cancel).not.toHaveBeenCalled();
      expect(capacity).not.toHaveBeenCalled();
      expect(runtime.store.getJob(active.jobId)?.status).not.toBe("cancelled");
    } finally {
      runtime.service.cancel(active.jobId);
    }
  });

  it("fails closed for a conversation with conflicting historical account reservations", async () => {
    const runtime = await coordinator();
    ownSession(runtime, "victim", "known-session");
    const key = runtime.apiAccess.createKey("attacker", "Legacy reservation regression");
    // Reproduce persisted state from before admission checked session ownership.
    const reservation = runtime.apiAccess.beginUsage("attacker", key.id, { ...prompt, session_id: "unrelated" });
    runtime.database.raw.prepare("UPDATE api_usage SET session_id = ? WHERE id = ?")
      .run("known-session", reservation.id);
    runtime.apiAccess.failUsage(reservation.id, "submission_failed");
    const response = await runtime.app.inject({
      method: "GET", url: "/v1/conversations/known-session/messages",
      headers: { authorization: `Bearer ${key.secret}` },
    });
    expect(response.statusCode).toBe(404);
    expect(runtime.apiAccess.userOwnsSession("attacker", "known-session")).toBe(false);
    expect(runtime.apiAccess.userOwnsSession("victim", "known-session")).toBe(false);
  });
});
