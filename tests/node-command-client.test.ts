import { randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { runNodeCommandClient } from "../src/node/command-client.js";
import { NodeCommandExecutor } from "../src/node/command-executor.js";
import { NodeReconciliationStateStore } from "../src/node/reconciliation-state.js";
import { NODE_EVENT_ORIGIN_CURSOR } from "../src/contracts/node-control.js";

const cleanup: string[] = [];
afterEach(async () => Promise.all(cleanup.splice(0).map((path) => rm(path, { recursive: true, force: true }))));

describe("node command client", () => {
  it("reposts a durable result without applying the redelivered command twice", async () => {
    const directory = await mkdtemp(join(tmpdir(), "mycellios-command-client-"));
    cleanup.push(directory);
    const executor = new NodeCommandExecutor(join(directory, "journal.json"), "node-1", 3, clock);
    const controller = new AbortController();
    const input = command();
    let pulls = 0;
    let posts = 0;
    let applies = 0;
    const fakeFetch = async (urlInput: string | URL | Request, init?: RequestInit): Promise<Response> => {
      const url = new URL(String(urlInput));
      expect(init?.headers).toMatchObject({ authorization: "Bearer session-token" });
      if (init?.method === "GET") {
        pulls += 1;
        expect(url.searchParams.get("generation")).toBe("3");
        return Response.json({ object: "list", data: [{ command: input, state: "delivered" }] });
      }
      posts += 1;
      if (posts === 1) return Response.json({ error: { code: "temporary" } }, { status: 503 });
      const result = JSON.parse(String(init?.body));
      controller.abort();
      return Response.json(result);
    };

    await runNodeCommandClient({
      coordinatorUrl: "https://coordinator.example.test",
      nodeId: "node-1",
      session: () => ({ token: "session-token", generation: 3 }),
      executor,
      apply: async () => { applies += 1; return { contributionEnabled: false }; },
      signal: controller.signal,
      fetch: fakeFetch as typeof fetch,
      pollIntervalMs: 1,
      retryIntervalMs: 1,
    });

    expect({ pulls, posts, applies }).toEqual({ pulls: 2, posts: 2, applies: 1 });
  });

  it("does not poll until a control session exists", async () => {
    const directory = await mkdtemp(join(tmpdir(), "mycellios-command-client-"));
    cleanup.push(directory);
    const controller = new AbortController();
    let requests = 0;
    setTimeout(() => controller.abort(), 10);
    await runNodeCommandClient({
      coordinatorUrl: "https://coordinator.example.test",
      nodeId: "node-1",
      session: () => null,
      executor: new NodeCommandExecutor(join(directory, "journal.json"), "node-1", 3, clock),
      apply: async () => null,
      signal: controller.signal,
      fetch: (async () => { requests += 1; return Response.json({}); }) as typeof fetch,
      pollIntervalMs: 1,
    });
    expect(requests).toBe(0);
  });

  it("publishes a snapshot and advances the durable cursor only after valid reconciliation", async () => {
    const directory = await mkdtemp(join(tmpdir(), "mycellios-command-client-")); cleanup.push(directory);
    const state = new NodeReconciliationStateStore(join(directory, "reconciliation.json"), "node-1");
    const controller = new AbortController();
    const latestCursor = "evt_2_aaaaaaaaaaaaaaaa";
    let postedCursor = "";
    let reconciled = false;
    let acknowledgedCursor = "";
    const fakeFetch = async (urlInput: string | URL | Request, init?: RequestInit): Promise<Response> => {
      const url = new URL(String(urlInput));
      if (init?.method === "GET") return Response.json({ object: "list", data: [] });
      expect(url.pathname).toBe("/internal/v1/nodes/node-1/snapshot");
      const posted = JSON.parse(String(init?.body));
      postedCursor = posted.cursor;
      controller.abort();
      return Response.json({
        snapshot: posted,
        desiredState: {
          schema: "mycellios-node-desired-state/1", nodeId: "node-1", generation: 3,
          contributionEnabled: true, drain: false, updateChannel: "stable", limits: posted.limits,
          updatedAt: posted.observedAt, updatedBy: { kind: "node", id: "node-1", scopes: ["node:read"] },
        },
        events: [], latestCursor,
      });
    };
    await runNodeCommandClient({
      coordinatorUrl: "https://coordinator.example.test", nodeId: "node-1",
      session: () => ({ token: "session-token", generation: 3 }),
      executor: new NodeCommandExecutor(join(directory, "journal.json"), "node-1", 3, clock),
      apply: async () => null, signal: controller.signal, fetch: fakeFetch as typeof fetch,
      pollIntervalMs: 1,
      reconciliation: {
        state,
        onReconciled: async () => { reconciled = true; },
        onSnapshotAcknowledged: async () => { acknowledgedCursor = (await state.load(3)).cursor; },
        snapshot: (cursor) => ({
          schema: "mycellios-node-snapshot/1", nodeId: "node-1", generation: 3, cursor,
          observedAt: new Date(clock()).toISOString(), state: "ready", contributionEnabled: true, draining: false,
          updateChannel: "stable", limits: { maxConcurrency: 2, maxCpuPercent: 80, maxRamMiB: 8192, maxVramMiB: 6144, maxDiskMiB: 16384, maxTemperatureC: 85 },
          activeCommandIds: [], build: { version: "1.0.0", sourceRevision: "a".repeat(40) },
          runtime: { ready: true, abi: "mycellios-distribution-runtime/4", backend: "cuda" },
        }),
      },
    });
    expect(postedCursor).toBe(NODE_EVENT_ORIGIN_CURSOR);
    expect(reconciled).toBe(true);
    expect(acknowledgedCursor).toBe(latestCursor);
    expect((await state.load(3)).cursor).toBe(latestCursor);
  });

  it("does not advance the cursor when applying desired state fails", async () => {
    const directory = await mkdtemp(join(tmpdir(), "mycellios-command-client-")); cleanup.push(directory);
    const state = new NodeReconciliationStateStore(join(directory, "reconciliation.json"), "node-1");
    const controller = new AbortController();
    let attempts = 0;
    const snapshot = {
      schema: "mycellios-node-snapshot/1" as const, nodeId: "node-1", generation: 3, cursor: NODE_EVENT_ORIGIN_CURSOR,
      observedAt: new Date(clock()).toISOString(), state: "ready" as const, contributionEnabled: true, draining: false,
      updateChannel: "stable" as const, limits: { maxConcurrency: 2, maxCpuPercent: 80, maxRamMiB: 8192, maxVramMiB: 6144, maxDiskMiB: 16384, maxTemperatureC: 85 },
      activeCommandIds: [], build: { version: "1.0.0", sourceRevision: "a".repeat(40) },
      runtime: { ready: true, abi: "mycellios-distribution-runtime/4", backend: "cuda" },
    };
    const fakeFetch = async (_url: string | URL | Request, init?: RequestInit): Promise<Response> => {
      if (init?.method === "GET") return Response.json({ object: "list", data: [] });
      attempts += 1;
      controller.abort();
      return Response.json({ snapshot, desiredState: {
        schema: "mycellios-node-desired-state/1", nodeId: "node-1", generation: 3, contributionEnabled: false,
        drain: false, updateChannel: "stable", limits: snapshot.limits, updatedAt: snapshot.observedAt,
        updatedBy: { kind: "node", id: "node-1", scopes: ["node:read"] },
      }, events: [], latestCursor: "evt_1_bbbbbbbbbbbbbbbb" });
    };
    await runNodeCommandClient({ coordinatorUrl: "https://coordinator.example.test", nodeId: "node-1",
      session: () => ({ token: "session-token", generation: 3 }), executor: new NodeCommandExecutor(join(directory, "journal.json"), "node-1", 3, clock),
      apply: async () => null, signal: controller.signal, fetch: fakeFetch as typeof fetch, retryIntervalMs: 1,
      reconciliation: { state, snapshot: () => snapshot, onReconciled: async () => { throw new Error("disk_full"); } },
    });
    expect(attempts).toBe(1);
    expect((await state.load(3)).cursor).toBe(NODE_EVENT_ORIGIN_CURSOR);
  });
});

function command() {
  return {
    schema: "mycellios-node-command/1",
    id: randomUUID(),
    nodeId: "node-1",
    actor: { kind: "account", id: "account-1", scopes: ["node:control"] },
    generation: 3,
    issuedAt: "2026-08-10T12:00:00.000Z",
    expiresAt: "2026-08-10T12:05:00.000Z",
    nonce: `nonce_${randomUUID().replaceAll("-", "")}`,
    type: "pause",
    payload: { version: 1 },
  };
}

const clock = () => Date.parse("2026-08-10T12:00:01.000Z");
