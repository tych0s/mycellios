import { afterEach, describe, expect, it } from "vitest";
import { resolve } from "node:path";
import { workerConfigSchema } from "../src/contracts/schemas.js";
import type { CoordinatorRuntime } from "../src/coordinator/server.js";
import { createCoordinator } from "../src/coordinator/server.js";
import { WorkerAgent } from "../src/worker/agent.js";

describe("inference-only coordinator and worker", () => {
  const cleanup: Array<() => Promise<void> | void> = [];

  afterEach(async () => {
    for (const operation of cleanup.splice(0).reverse()) await operation();
  });

  it("executes an OpenAI-compatible chat end to end", async () => {
    const { runtime, address, agent, run } = await startNetwork();
    cleanup.push(async () => {
      await agent.stop();
      await run;
      await runtime.close();
    });

    const response = await fetch(new URL("v1/chat/completions", `${address}/`), {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "idempotency-key": "e2e-request-1",
      },
      body: JSON.stringify({
        model: "distributed-small",
        messages: [{ role: "user", content: "prueba completa" }],
        max_tokens: 80,
        session_id: "session-e2e",
      }),
    });
    const body = (await response.json()) as {
      id: string;
      choices: Array<{ message: { content: string } }>;
      usage: { prompt_tokens: number; completion_tokens: number };
      x_network: { ttft_ms: number; active_ms: number };
    };
    expect(response.status).toBe(200);
    expect(body.choices[0]?.message.content).toContain("prueba completa");
    expect(body.usage.prompt_tokens).toBeGreaterThan(0);
    expect(body.usage.completion_tokens).toBeGreaterThan(0);
    expect(body.x_network.active_ms).toBeGreaterThan(0);
    expect(runtime.store.listJobs()).toHaveLength(1);
    expect(runtime.store.listJobs()[0]?.status).toBe("completed");

    const replay = await fetch(new URL("v1/chat/completions", `${address}/`), {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "idempotency-key": "e2e-request-1",
      },
      body: JSON.stringify({
        model: "distributed-small",
        messages: [{ role: "user", content: "prueba completa" }],
        max_tokens: 80,
        session_id: "session-e2e",
      }),
    });
    expect(replay.status).toBe(409);
    expect(runtime.store.listJobs()).toHaveLength(1);
  });

  it("returns OpenAI SSE chunks and a DONE marker", async () => {
    const { runtime, address, agent, run } = await startNetwork();
    cleanup.push(async () => {
      await agent.stop();
      await run;
      await runtime.close();
    });
    const response = await fetch(new URL("v1/chat/completions", `${address}/`), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "distributed-small",
        messages: [{ role: "user", content: "stream" }],
        stream: true,
        max_tokens: 40,
      }),
    });
    const stream = await response.text();
    expect(response.headers.get("content-type")).toContain("text/event-stream");
    expect(stream).toContain("chat.completion.chunk");
    expect(stream).toContain("data: [DONE]");
  });

  it("carries llmfit from worker startup into the application model catalog", async () => {
    const { runtime, address, agent, run } = await startNetwork({ llmfit: true });
    cleanup.push(async () => {
      await agent.stop();
      await run;
      await runtime.close();
    });
    const response = await fetch(new URL("v1/models", `${address}/`));
    const body = (await response.json()) as {
      data: Array<{
        id: string;
        x_llmfit?: {
          advised_replicas: number;
          best_fit: string;
          quantizations: string[];
          max_estimated_tokens_per_second: number;
          max_measured_tokens_per_second: number;
          min_memory_required_mb: number;
        };
      }>;
    };
    expect(body.data[0]).toMatchObject({
      id: "distributed-small",
      x_llmfit: {
        advised_replicas: 1,
        best_fit: "Perfect",
        quantizations: ["Q8_0"],
        max_estimated_tokens_per_second: 42,
        max_measured_tokens_per_second: 36,
        min_memory_required_mb: 3_072,
      },
    });
    expect(runtime.store.listWorkers()[0]?.capabilities.llmfit?.model?.deploymentId).toMatch(
      /^dep-/,
    );
  });

  it("removes a worker from the visible inventory after a voluntary stop", async () => {
    const { runtime, agent, run } = await startNetwork();
    cleanup.push(async () => {
      await agent.stop();
      await run;
      await runtime.close();
    });
    expect(runtime.store.listWorkers()).toHaveLength(1);

    await agent.stop();
    await run;
    await waitUntil(() => runtime.store.listWorkers().length === 0, 2_000);

    expect(runtime.store.listWorkers()).toHaveLength(0);
    expect(runtime.hub.connectedWorkerIds().size).toBe(0);
    const retained = runtime.database.raw
      .prepare("SELECT status, deregistered FROM workers")
      .get() as { status: string; deregistered: number };
    expect(retained).toEqual({ status: "offline", deregistered: 1 });
  });

  it("protects network and worker routes when a network token is configured", async () => {
    const networkToken = "test-network-token";
    const { runtime, address, agent, run } = await startNetwork({ networkToken });
    cleanup.push(async () => {
      await agent.stop();
      await run;
      await runtime.close();
    });

    expect((await fetch(new URL("health", `${address}/`))).status).toBe(200);
    expect((await fetch(new URL("public/v1/snapshot", `${address}/`))).status).toBe(200);
    expect((await fetch(new URL("network", `${address}/`))).status).toBe(200);
    expect((await fetch(new URL("v1/models", `${address}/`))).status).toBe(401);
    expect(
      (
        await fetch(new URL("v1/models", `${address}/`), {
          headers: { authorization: `Bearer ${networkToken}` },
        })
      ).status,
    ).toBe(200);
  });
});

async function startNetwork(options: { llmfit?: boolean; networkToken?: string } = {}): Promise<{
  runtime: CoordinatorRuntime;
  address: string;
  agent: WorkerAgent;
  run: Promise<void>;
}> {
  const runtime = await createCoordinator({
    host: "127.0.0.1",
    port: 0,
    databasePath: ":memory:",
    requestTimeoutMs: 10_000,
    mobileAssetsPath: resolve("tests/fixtures/mobile-assets"),
    landingAssetsPath: resolve("tests/fixtures/landing-assets"),
    ...(options.networkToken
      ? { networkToken: options.networkToken, mobileJoinToken: options.networkToken }
      : {}),
  });
  const address = await runtime.app.listen({ host: "127.0.0.1", port: 0 });
  const agent = new WorkerAgent(
    workerConfigSchema.parse({
      region: "es-mad",
      offeredVramMb: 4_096,
      limits: { maxConcurrency: 2, pauseWhenForeground: true },
      adapter: {
        kind: "mock",
        model: "distributed-small",
        tokensPerSecond: 500,
        ttftMs: 1,
        failureRate: 0,
      },
      deployment: {
        modelDigest: "sha256:distributed-small-test",
        peakVramMb: 3_000,
        contextLimit: 8_192,
        tokensPerSecond: 500,
        ttftMs: 1,
      },
      ...(options.llmfit
        ? {
            llmfit: {
              enabled: true,
              required: true,
              executable: process.execPath,
              arguments: [resolve("tests/fixtures/fake-llmfit.cjs")],
              timeoutMs: 5_000,
              model: "fixture/distributed-small",
              applyPerformanceEstimate: false,
            },
          }
        : {}),
    }),
    {
      coordinatorUrl: address,
      heartbeatIntervalMs: 100,
      reconnect: false,
      logger: { info() {}, warn() {}, error() {} },
      ...(options.networkToken ? { networkToken: options.networkToken } : {}),
    },
  );
  const run = agent.start();
  await waitUntil(() => runtime.hub.connectedWorkerIds().size === 1, 5_000);
  return { runtime, address, agent, run };
}

async function waitUntil(predicate: () => boolean, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error("Timed out waiting for test worker");
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}
