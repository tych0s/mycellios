import { createServer, type RequestListener, type Server } from "node:http";
import { afterEach, describe, expect, it } from "vitest";
import type { InferenceAdapter } from "../src/adapters/base.js";
import { local model runtimeAdapter } from "../src/adapters/local-model-runtime.js";
import { OpenAICompatibleAdapter } from "../src/adapters/openai-compatible.js";
import { workerConfigSchema } from "../src/contracts/schemas.js";
import type {
  JobPayload,
  WorkerCapabilities,
  WorkerEnvelope,
} from "../src/contracts/types.js";
import {
  parseServerMessage,
  validateCoordinatorUrl,
  WorkerAgent,
} from "../src/worker/agent.js";

describe("worker boundary hardening", () => {
  const servers: Server[] = [];

  afterEach(async () => {
    await Promise.all(
      servers
        .splice(0)
        .map((server) => new Promise<void>((resolve) => server.close(() => resolve()))),
    );
  });

  it("requires encrypted transport for a remote coordinator", () => {
    expect(() => validateCoordinatorUrl("http://coordinator.example")).toThrow(/HTTPS\/WSS/);
    expect(() => validateCoordinatorUrl("ws://coordinator.example")).toThrow(/HTTPS\/WSS/);
    expect(validateCoordinatorUrl("http://127.0.0.1:8080").protocol).toBe("http:");
    expect(validateCoordinatorUrl("https://coordinator.example").protocol).toBe("https:");
  });

  it("rejects malformed or oversized logical coordinator messages", () => {
    expect(() => parseServerMessage(null)).toThrow(/Invalid coordinator message/);
    expect(() =>
      parseServerMessage({
        v: 1,
        type: "lease.offer",
        payload: { jobId: "job-1" },
      }),
    ).toThrow(/Invalid coordinator message/);
    expect(() =>
      parseServerMessage({
        v: 1,
        type: "server.ready",
        payload: { workerId: "x".repeat(257) },
      }),
    ).toThrow(/Invalid coordinator message/);
  });

  it("strictly validates coordinator-issued direct grants and candidates", () => {
    const grant = {
      protocol: "mycellios-direct/1",
      connectionId: "direct-stream",
      sourceNodeId: "node-a",
      destinationNodeId: "node-b",
      targetPort: 9_850,
      expiresAt: Date.now() + 10_000,
      secret: Buffer.alloc(32, 9).toString("base64url"),
    };
    expect(parseServerMessage({
      v: 1,
      type: "runtime.direct.connect",
      payload: {
        streamId: "direct-stream",
        destinationNodeId: "node-b",
        grant,
        candidates: [{ host: "127.0.0.1", port: 50_001, scope: "configured" }],
        timeoutMs: 1_000,
      },
    }).type).toBe("runtime.direct.connect");
    expect(() => parseServerMessage({
      v: 1,
      type: "runtime.direct.connect",
      payload: {
        streamId: "direct-stream",
        destinationNodeId: "node-b",
        grant: { ...grant, debugSecret: grant.secret },
        candidates: [{ host: "127.0.0.1", port: 50_001, scope: "configured" }],
        timeoutMs: 1_000,
      },
    })).toThrow(/Invalid coordinator message/);
  });

  it("requires HTTPS whenever an OpenAI-compatible API key is present", () => {
    expect(
      () =>
        new OpenAICompatibleAdapter({
          baseUrl: "http://127.0.0.1:8080",
          model: "test",
          apiKey: "secret",
        }),
    ).toThrow(/must use HTTPS/);
  });

  it("does not follow backend redirects", async () => {
    let followed = false;
    const baseUrl = await listen(servers, (request, response) => {
      if (request.url === "/v1/models") {
        response.statusCode = 302;
        response.setHeader("location", "/redirect-target");
        response.end();
        return;
      }
      followed = true;
      response.end("unexpected");
    });
    const adapter = new OpenAICompatibleAdapter({
      baseUrl,
      model: "test",
      kind: "externalggufruntime",
    });
    await expect(adapter.probe()).rejects.toThrow(/HTTP 302/);
    expect(followed).toBe(false);
  });

  it("represents a sidecar cell as aggregate capacity without claiming one huge GPU", async () => {
    const baseUrl = await listen(servers, (_request, response) => {
      response.setHeader("content-type", "application/json");
      response.end('{"object":"list","data":[]}');
    });
    const config = workerConfigSchema.parse({
      region: "test-cell",
      capacityScope: "cell",
      offeredVramMb: 98_304,
      limits: { maxConcurrency: 8, pauseWhenForeground: false },
      adapter: {
        kind: "openai-compatible",
        model: "regional-large",
        baseUrl,
        allowedHosts: [],
      },
      deployment: {
        modelDigest: "sha256:pinned-cell-manifest",
        peakVramMb: 72_000,
        contextLimit: 32_768,
      },
    });
    const agent = new WorkerAgent(config, {
      coordinatorUrl: "http://127.0.0.1:9999",
      reconnect: false,
      logger: { info() {}, warn() {}, error() {} },
    });
    const capabilities = await (
      agent as unknown as Pick<AgentHarness, "buildCapabilities">
    ).buildCapabilities();
    expect(capabilities.gpus[0]).toMatchObject({
      id: "cell-aggregate",
      vendor: "sidecar-cell",
      physicalVramMb: 0,
      offeredVramMb: 98_304,
    });
    expect(capabilities.deployments[0]?.peakVramMb).toBe(72_000);

    expect(
      workerConfigSchema.safeParse({
        ...baseConfig(),
        capacityScope: "cell",
      }).success,
    ).toBe(false);
  });

  it("caps incomplete SSE and NDJSON buffers at 1 MiB", async () => {
    const sseBaseUrl = await listen(servers, (request, response) => {
      if (request.url === "/v1/chat/completions") {
        response.setHeader("content-type", "text/event-stream");
        response.end(`data: ${"x".repeat(1024 * 1024 + 1)}`);
      }
    });
    const sseAdapter = new OpenAICompatibleAdapter({
      baseUrl: sseBaseUrl,
      model: "test",
      kind: "externalggufruntime",
    });
    await expect(collect(sseAdapter)).rejects.toThrow(/buffer exceeded 1 MiB/);

    const ndjsonBaseUrl = await listen(servers, (request, response) => {
      if (request.url === "/api/chat") {
        response.setHeader("content-type", "application/x-ndjson");
        response.end("x".repeat(1024 * 1024 + 1));
      }
    });
    const ndjsonAdapter = new local model runtimeAdapter({ baseUrl: ndjsonBaseUrl, model: "test" });
    await expect(collect(ndjsonAdapter)).rejects.toThrow(/buffer exceeded 1 MiB/);
  });

  it("rejects a mismatched digest, duplicate job and oversized adapter chunk", async () => {
    const agent = new WorkerAgent(baseConfig(), {
      coordinatorUrl: "http://127.0.0.1:9999",
      reconnect: false,
      logger: { info() {}, warn() {}, error() {} },
    });
    const sent: WorkerEnvelope[] = [];
    const harness = agent as unknown as AgentHarness;
    harness.registeredWorkerId = "worker-test";
    harness.capabilities = capabilities();
    harness.socket = {
      readyState: 1,
      send(serialized) {
        sent.push(JSON.parse(serialized) as WorkerEnvelope);
      },
    };

    await harness.execute(payload({ modelDigest: "sha256:wrong", jobId: "wrong" }));
    expect(messagePayload(sent, "lease.reject")).toMatchObject({
      reason: "model_digest_mismatch",
    });

    sent.length = 0;
    harness.adapter = oversizedAdapter();
    const accepted = payload({ jobId: "duplicate", modelDigest: "sha256:test" });
    await harness.execute(accepted);
    expect(messagePayload(sent, "task.fail")).toMatchObject({
      code: "output_limit_exceeded",
    });
    expect(sent.some((message) => message.type === "task.token")).toBe(false);

    sent.length = 0;
    await harness.execute({ ...accepted, leaseId: "lease-second" });
    expect(messagePayload(sent, "lease.reject")).toMatchObject({ reason: "duplicate_job" });
  });
});

interface AgentHarness {
  registeredWorkerId?: string;
  capabilities: WorkerCapabilities;
  socket: { readyState: number; send(serialized: string): void };
  adapter: InferenceAdapter;
  execute(payload: JobPayload): Promise<void>;
  buildCapabilities(): Promise<WorkerCapabilities>;
}

function baseConfig() {
  return workerConfigSchema.parse({
    region: "test",
    offeredVramMb: 4_096,
    limits: { maxConcurrency: 1, pauseWhenForeground: false },
    adapter: {
      kind: "mock",
      model: "test-model",
      tokensPerSecond: 1_000,
      ttftMs: 0,
      failureRate: 0,
    },
    deployment: { modelDigest: "sha256:test", contextLimit: 8_192 },
  });
}

function capabilities(): WorkerCapabilities {
  return {
    region: "test",
    agentVersion: "test",
    gpus: [
      {
        id: "gpu",
        vendor: "test",
        model: "test",
        physicalVramMb: 4_096,
        offeredVramMb: 4_096,
        freeOfferedVramMb: 4_096,
      },
    ],
    limits: { maxConcurrency: 1, pauseWhenForeground: false },
    deployments: [
      {
        deploymentId: "deployment-test",
        model: "test-model",
        modelDigest: "sha256:test",
        mode: "replica",
        adapter: "mock",
        peakVramMb: 2_048,
        contextLimit: 8_192,
        maxConcurrency: 1,
        freeSlots: 1,
        tokensPerSecond: 10,
        ttftMs: 10,
        dataLocality: "local",
      },
    ],
    network: { coordinatorRttMs: 1, uplinkMbps: 10, downlinkMbps: 10 },
  };
}

function payload(overrides: Partial<JobPayload>): JobPayload {
  return {
    jobId: "job-test",
    leaseId: "lease-test",
    modelDigest: "sha256:test",
    deadlineAt: Date.now() + 60_000,
    request: {
      model: "test-model",
      messages: [{ role: "user", content: "hello" }],
      max_tokens: 32_768,
    },
    ...overrides,
  };
}

function oversizedAdapter(): InferenceAdapter {
  return {
    kind: "mock",
    async probe() {
      return {
        kind: "mock",
        models: ["test-model"],
        streaming: true,
        embeddings: false,
        reranking: false,
      };
    },
    async warm() {},
    async *generate() {
      yield { index: 0, text: "x".repeat(64 * 1024 + 1) };
    },
    async cancel() {},
    async metrics() {
      return { ready: true, activeJobs: 0, loadedModels: ["test-model"] };
    },
  };
}

function messagePayload(messages: WorkerEnvelope[], type: string): Record<string, unknown> {
  const message = messages.find((candidate) => candidate.type === type);
  expect(message, `Expected a ${type} message`).toBeDefined();
  return message!.payload as Record<string, unknown>;
}

async function collect(adapter: InferenceAdapter): Promise<string> {
  let output = "";
  for await (const chunk of adapter.generate(
    {
      jobId: "job",
      request: { model: "test", messages: [{ role: "user", content: "hello" }] },
    },
    new AbortController().signal,
  )) {
    output += chunk.text;
  }
  return output;
}

async function listen(servers: Server[], handler: RequestListener): Promise<string> {
  const server = createServer(handler);
  servers.push(server);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("Expected a TCP address");
  return `http://127.0.0.1:${address.port}`;
}
