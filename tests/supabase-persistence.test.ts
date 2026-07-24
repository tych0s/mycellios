import { afterEach, describe, expect, it } from "vitest";
import { MeshDatabase } from "../src/storage/database.js";
import { MeshStore } from "../src/storage/store.js";
import { SupabasePersistence } from "../src/storage/supabase-sync.js";
import { addWorker } from "./helpers.js";

describe("Supabase durable persistence", () => {
  const databases: MeshDatabase[] = [];

  afterEach(() => {
    for (const database of databases.splice(0)) database.close();
  });

  it("flushes transactional coordinator state and history to Supabase", async () => {
    const database = new MeshDatabase(":memory:");
    databases.push(database);
    const store = new MeshStore(database);
    const worker = addWorker(store, { id: "persisted-worker" });
    store.upsertRequestedModel({
      id: "qwen-test",
      source: "Qwen/Qwen-test",
      revision: null,
      contextTokens: 4096,
      minimumNodes: 1,
      autoActivate: true,
    });
    store.createJob({
      id: "job-persisted",
      sessionId: "session-persisted",
      model: "qwen-test",
      workloadClass: "interactive",
      deadlineAt: Date.now() + 60_000,
    });
    const conversationId = store.startInferenceConversation(
      "session-persisted",
      "qwen-test",
      [{ role: "user", content: "hello" }],
    );
    store.appendInferenceMessage({
      conversationId,
      jobId: "job-persisted",
      role: "assistant",
      content: "hello back",
      status: "completed",
      outputTokens: 2,
    });
    store.appendActivationEvent("qwen-test", {
      phase: "active",
      state: "completed",
      message: "Model is active.",
      at: new Date().toISOString(),
      nodeId: worker.id,
    });

    const postedTables: string[] = [];
    const sync = new SupabasePersistence(store, {
      url: "https://supabase.example.test",
      serviceRoleKey: "test-service-role",
      flushIntervalMs: 60_000,
      fetchImpl: async (input, init) => {
        const url = new URL(String(input));
        const table = url.pathname.split("/").at(-1)!;
        if ((init?.method ?? "GET") === "GET") {
          return Response.json([]);
        }
        postedTables.push(table);
        return new Response(null, { status: 201 });
      },
    });
    await sync.initialize();

    expect(database.pendingRemoteChangeCount()).toBe(0);
    expect(new Set(postedTables)).toEqual(new Set([
      "workers",
      "requested_models",
      "jobs",
      "inference_conversations",
      "inference_messages",
      "activation_events",
    ]));
    expect(sync.status()).toMatchObject({
      connected: true,
      pendingChanges: 0,
      lastError: null,
    });
    await sync.close();
  });

  it("keeps unsent changes in the local outbox while Supabase is unavailable", async () => {
    const database = new MeshDatabase(":memory:");
    databases.push(database);
    const store = new MeshStore(database);
    addWorker(store, { id: "offline-sync-worker" });
    const sync = new SupabasePersistence(store, {
      url: "https://supabase.example.test",
      serviceRoleKey: "test-service-role",
      fetchImpl: async () => new Response("temporarily unavailable", { status: 503 }),
    });

    await sync.initialize();

    expect(database.pendingRemoteChangeCount()).toBeGreaterThan(0);
    expect(sync.status()).toMatchObject({
      connected: false,
      required: false,
    });
    await sync.close();
  });

  it("restores workers from Supabase before the coordinator starts", async () => {
    const database = new MeshDatabase(":memory:");
    databases.push(database);
    const store = new MeshStore(database);
    const remoteWorker = {
      id: "wrk-remote",
      status: "offline",
      capabilities_json: {
        region: "eu",
        agentVersion: "remote",
        gpus: [],
        limits: { maxConcurrency: 1, pauseWhenForeground: true },
        deployments: [],
        network: { coordinatorRttMs: 1, uplinkMbps: 1, downlinkMbps: 1 },
      },
      reliability: 0.99,
      jobs_completed: 12,
      last_seen_at: 100,
      created_at: 50,
      updated_at: 200,
      deregistered: false,
      identity_kind: "device",
      identity_id: "remote-device",
    };
    const sync = new SupabasePersistence(store, {
      url: "https://supabase.example.test",
      serviceRoleKey: "test-service-role",
      flushIntervalMs: 60_000,
      fetchImpl: async (input, init) => {
        const url = new URL(String(input));
        if ((init?.method ?? "GET") === "GET") {
          return Response.json(url.pathname.endsWith("/workers") ? [remoteWorker] : []);
        }
        return new Response(null, { status: 201 });
      },
    });

    await sync.initialize();

    expect(store.getWorker("wrk-remote")).toMatchObject({
      id: "wrk-remote",
      reliability: 0.99,
      jobsCompleted: 12,
      identityId: "remote-device",
    });
    await sync.close();
  });
});
