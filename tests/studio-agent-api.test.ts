import { afterEach, describe, expect, it, vi } from "vitest";
import { createCoordinator, type CoordinatorRuntime } from "../src/coordinator/server.js";
import type { SupabaseAuthService } from "../src/coordinator/supabase-auth.js";

const runtimes: CoordinatorRuntime[] = [];
afterEach(async () => Promise.all(runtimes.splice(0).map((runtime) => runtime.close())));

const configuration = {
  name: "Mara", role: "Product guide",
  instructions: "Explain Mycellios accurately and state uncertainty whenever evidence is unavailable.",
  memoryMode: "approved", knowledgeSourceIds: [], tools: ["documents"],
  modelPolicy: { preferredModel: "Qwen/Qwen3-0.6B", fallbackModel: null, privacy: "trusted-only", maxOutputTokens: 512, deadlineMs: 120_000 },
};

describe("Studio agent management API", () => {
  it("requires an account session and preserves owner isolation", async () => {
    const auth = {
      authenticate: vi.fn(async (token: string) => token === "owner-a-token"
        ? { id: "owner-a", email: null, role: null }
        : token === "owner-b-token" ? { id: "owner-b", email: null, role: null } : null),
    } as unknown as SupabaseAuthService;
    const runtime = await createCoordinator({ host: "127.0.0.1", port: 0, databasePath: ":memory:", requestTimeoutMs: 1_000, apiAccessEnabled: true }, { logger: false, supabaseAuthService: auth });
    runtimes.push(runtime);

    expect((await runtime.app.inject({ method: "GET", url: "/v1/studio/agents" })).statusCode).toBe(401);
    const created = await runtime.app.inject({
      method: "POST", url: "/v1/studio/agents", headers: { authorization: "Bearer owner-a-token" },
      payload: { idempotencyKey: "create-1", templateId: "concierge", configuration },
    });
    expect(created.statusCode).toBe(201);
    const agent = created.json<{ id: string; draftVersion: number }>();
    const hidden = await runtime.app.inject({ method: "GET", url: `/v1/studio/agents/${agent.id}`, headers: { authorization: "Bearer owner-b-token" } });
    expect(hidden.statusCode).toBe(404);
    expect((await runtime.app.inject({ method: "GET", url: "/v1/studio/agents", headers: { authorization: "Bearer owner-b-token" } })).json()).toEqual({ object: "list", data: [] });
  });

  it("publishes honestly as waiting when no node exists", async () => {
    const auth = { authenticate: vi.fn(async () => ({ id: "owner-a", email: null, role: null })) } as unknown as SupabaseAuthService;
    const runtime = await createCoordinator({ host: "127.0.0.1", port: 0, databasePath: ":memory:", requestTimeoutMs: 1_000, apiAccessEnabled: true }, { logger: false, supabaseAuthService: auth });
    runtimes.push(runtime);
    const created = await runtime.app.inject({ method: "POST", url: "/v1/studio/agents", headers: { authorization: "Bearer token" }, payload: { idempotencyKey: "create-1", templateId: null, configuration } });
    const agent = created.json<{ id: string }>();
    const published = await runtime.app.inject({ method: "POST", url: `/v1/studio/agents/${agent.id}/publish`, headers: { authorization: "Bearer token" }, payload: { idempotencyKey: "publish-1", expectedVersion: 1, channels: ["web", "api"] } });
    expect(published.statusCode).toBe(200);
    expect(published.json()).toMatchObject({
      agent: { status: "published", operationalState: "waiting_for_capacity" },
      deployments: [{ state: "waiting_for_capacity" }, { state: "waiting_for_capacity" }],
    });
    const web = published.json<{ deployments: Array<{ channel: string; publicId: string }> }>().deployments.find((item) => item.channel === "web")!;
    const unavailable = await runtime.app.inject({ method: "POST", url: `/public/v1/studio/web/${web.publicId}/invoke`, payload: { idempotencyKey: "no-capacity", subjectId: "visitor", message: "Hello" } });
    expect(unavailable.statusCode).toBe(503);
    expect(unavailable.json()).toMatchObject({ error: { code: "studio_waiting_for_capacity" } });
    expect(runtime.database.raw.prepare("SELECT COUNT(*) AS count FROM studio_invocations").get()).toEqual({ count: 0 });
    expect(runtime.database.raw.prepare("SELECT COUNT(*) AS count FROM api_usage").get()).toEqual({ count: 0 });
  });

  it("manages bounded context and exposes an idempotent Web invocation after certified activation", async () => {
    const auth = { authenticate: vi.fn(async () => ({ id: "owner-a", email: null, role: null })) } as unknown as SupabaseAuthService;
    const infer = vi.fn(async (request) => {
      request.onToken?.("verified ");
      request.onToken?.("reply");
      return { text: "verified reply", inputTokens: 3, outputTokens: 2, receiptId: "receipt-1" };
    });
    const runtime = await createCoordinator({ host: "127.0.0.1", port: 0, databasePath: ":memory:", requestTimeoutMs: 1_000, apiAccessEnabled: true }, { logger: false, supabaseAuthService: auth, studioInference: infer });
    runtimes.push(runtime);
    const headers = { authorization: "Bearer token" };
    const created = await runtime.app.inject({ method: "POST", url: "/v1/studio/agents", headers, payload: { idempotencyKey: "create-context", templateId: null, configuration } });
    const agent = created.json<{ id: string }>();
    const sourceResponse = await runtime.app.inject({ method: "POST", url: `/v1/studio/agents/${agent.id}/knowledge`, headers, payload: { name: "Guide", mediaType: "text/plain", content: "Mycellios Studio creates persistent identities." } });
    expect(sourceResponse.statusCode).toBe(201);
    const source = sourceResponse.json<{ id: string }>();
    const updated = await runtime.app.inject({ method: "PATCH", url: `/v1/studio/agents/${agent.id}/draft`, headers, payload: { expectedVersion: 1, configuration: { ...configuration, knowledgeSourceIds: [source.id] } } });
    expect(updated.statusCode).toBe(200);
    const publishedResponse = await runtime.app.inject({ method: "POST", url: `/v1/studio/agents/${agent.id}/publish`, headers, payload: { idempotencyKey: "publish-context", expectedVersion: 2, channels: ["web", "telegram", "api"] } });
    const deployments = publishedResponse.json<{ deployments: Array<{ id: string; publicId: string; channel: string }> }>().deployments;
    const deployment = deployments.find((item) => item.channel === "web")!;
    runtime.studioAgents.activateCompatibleWaiting("Qwen/Qwen3-0.6B");
    const payload = { idempotencyKey: "invoke-1", subjectId: "visitor", message: "What is Studio?" };
    const first = await runtime.app.inject({ method: "POST", url: `/public/v1/studio/web/${deployment.publicId}/invoke`, payload });
    const replay = await runtime.app.inject({ method: "POST", url: `/public/v1/studio/web/${deployment.publicId}/invoke`, payload });
    expect(first.json()).toMatchObject({ text: "verified reply", replayed: false, receiptId: "receipt-1" });
    expect(replay.json()).toMatchObject({ text: "verified reply", replayed: true });
    expect(infer).toHaveBeenCalledOnce();

    const streamed = await runtime.app.inject({ method: "POST", url: `/public/v1/studio/web/${deployment.publicId}/invoke`, payload: { ...payload, idempotencyKey: "invoke-stream", stream: true } });
    expect(streamed.statusCode).toBe(200);
    expect(streamed.headers["content-type"]).toContain("text/event-stream");
    expect(streamed.body).toContain('"type":"token","delta":"verified "');
    expect(streamed.body).toContain('"type":"completed"');
    expect(streamed.body).toContain("data: [DONE]");
    expect(infer).toHaveBeenCalledTimes(2);

    const api = deployments.find((item) => item.channel === "api")!;
    const apiInvocation = await runtime.app.inject({ method: "POST", url: `/v1/studio/api/${api.publicId}/invoke`, headers, payload: { ...payload, idempotencyKey: "invoke-api" } });
    expect(apiInvocation.json()).toMatchObject({ text: "verified reply", replayed: false, receiptId: "receipt-1" });
    expect(infer).toHaveBeenCalledTimes(3);

    const telegram = deployments.find((item) => item.channel === "telegram")!;
    expect((await runtime.app.inject({ method: "POST", url: `/v1/studio/deployments/${telegram.id}/telegram`, headers, payload: { secret: "telegram-webhook-secret", allowedChats: ["123"] } })).statusCode).toBe(204);
    const webhookPayload = { update_id: 99, message: { chat: { id: 123 }, text: "Hello" } };
    const webhookHeaders = { "x-telegram-bot-api-secret-token": "telegram-webhook-secret" };
    const delivered = await runtime.app.inject({ method: "POST", url: `/public/v1/studio/telegram/${telegram.publicId}`, headers: webhookHeaders, payload: webhookPayload });
    const telegramReplay = await runtime.app.inject({ method: "POST", url: `/public/v1/studio/telegram/${telegram.publicId}`, headers: webhookHeaders, payload: webhookPayload });
    expect(delivered.json()).toEqual({ method: "sendMessage", chat_id: 123, text: "verified reply" });
    expect(telegramReplay.json()).toEqual({ ok: true, replayed: true });
    expect(infer).toHaveBeenCalledTimes(4);
  });
});
