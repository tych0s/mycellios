import { createServer, type RequestListener, type Server } from "node:http";
import { afterEach, describe, expect, it } from "vitest";
import type { InferenceAdapter } from "../src/adapters/base.js";
import { MycelliosPipelineAdapter } from "../src/adapters/mycellios-pipeline.js";
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
import { sha256Text } from "../src/core/json.js";

const CELL_MODEL_DIGEST = `sha256:${"b".repeat(64)}`;
const CELL_ACTIVATION_ID = "pipeline-activation-7";

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

  it("accepts only an exact coordinator evidence challenge contract", () => {
    const challenge = {
      schema: "mycellios-evidence-challenge/1",
      kind: "deployment-canary",
      challengeId: "challenge-1",
      nonce: Buffer.alloc(32, 2).toString("base64url"),
      sessionId: "session-1",
      workerId: "worker-1",
      issuedAt: "2026-07-25T10:00:00.000Z",
      expiresAt: "2026-07-25T10:10:00.000Z",
      deploymentId: "deployment-1",
      model: "model",
      modelDigest: `sha256:${"a".repeat(64)}`,
      activationId: "activation-1",
      prompt: "measure",
      promptDigest: `sha256:${"b".repeat(64)}`,
      maxOutputTokens: 64,
      warmupSamples: 1,
      samples: 3,
    } as const;
    expect(parseServerMessage({
      v: 1,
      type: "evidence.challenge",
      payload: challenge,
    }).type).toBe("evidence.challenge");
    expect(() => parseServerMessage({
      v: 1,
      type: "evidence.challenge",
      payload: { ...challenge, workerPerformanceOverride: 1_000_000 },
    })).toThrow(/Invalid coordinator message/);
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

  it("does not expose a remote host or credential escape hatch", () => {
    expect(
      () =>
        new MycelliosPipelineAdapter({
          baseUrl: "https://inference.example",
          model: "test",
          modelDigest: "sha256:test",
          activationId: "test-activation",
        }),
    ).toThrow(/loopback/);
  });

  it("does not follow native pipeline redirects", async () => {
    let followed = false;
    const baseUrl = await listen(servers, (request, response) => {
      if (request.url === "/health") {
        response.statusCode = 302;
        response.setHeader("location", "/redirect-target");
        response.end();
        return;
      }
      followed = true;
      response.end("unexpected");
    });
    const adapter = new MycelliosPipelineAdapter({
      baseUrl,
      model: "test",
      modelDigest: "sha256:test",
      activationId: "test-activation",
    });
    await expect(adapter.probe()).rejects.toThrow(/HTTP 302/);
    expect(followed).toBe(false);
  });

  it("represents a native pipeline as aggregate capacity without claiming one huge GPU", async () => {
    const evidenceSessionIds: string[] = [];
    const baseUrl = await listen(servers, (request, response) => {
      if (request.url === "/health") {
        response.setHeader("content-type", "application/json");
        response.end(JSON.stringify({
          status: "ready",
          model: "regional-large",
          artifact_identity: CELL_MODEL_DIGEST,
          pipeline_snapshot_identity: CELL_ACTIVATION_ID,
          stages: 2,
          boundaries: [0, 12, 24],
        }));
        return;
      }
      const evidenceSessionId = request.headers["x-session-id"];
      if (typeof evidenceSessionId === "string") evidenceSessionIds.push(evidenceSessionId);
      response.setHeader("content-type", "text/event-stream");
      response.write('data: {"choices":[{"delta":{"content":"ok"}}]}\n\n');
      response.write('data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":1,"completion_tokens":1},"distribution_metrics":{"ttft_ms":1,"pipeline_ms":1}}\n\n');
      response.end("data: [DONE]\n\n");
    });
    const config = workerConfigSchema.parse({
      region: "test-cell",
      capacityScope: "cell",
      offeredVramMb: 98_304,
      limits: { maxConcurrency: 8, pauseWhenForeground: false },
      adapter: {
        kind: "mycellios-pipeline",
        model: "regional-large",
        baseUrl,
      },
      deployment: {
        modelDigest: CELL_MODEL_DIGEST,
        activationId: CELL_ACTIVATION_ID,
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
      vendor: "mycellios",
      physicalVramMb: 0,
      offeredVramMb: 98_304,
    });
    expect(capabilities.deployments[0]?.peakVramMb).toBe(72_000);
    expect(capabilities.deployments[0]).toMatchObject({
      deploymentId: `dep-${sha256Text([
        "regional-large",
        "mycellios-pipeline",
        CELL_MODEL_DIGEST,
        CELL_ACTIVATION_ID,
      ].join(":")).slice(-12)}`,
      activationId: CELL_ACTIVATION_ID,
      tokensPerSecond: 1,
      throughputSource: "default",
      ttftMs: 60_000,
      verificationState: "pending",
    });

    expect(workerConfigSchema.safeParse({
      ...config,
      deployment: {
        ...config.deployment,
        canaryEvidence: { workerDeclared: true },
        tokensPerSecond: 99_999,
        ttftMs: 0,
      },
    }).success).toBe(false);

    const sent: WorkerEnvelope[] = [];
    const harness = agent as unknown as AgentHarness;
    harness.registeredWorkerId = "worker-cell";
    harness.capabilities = capabilities;
    harness.socket = {
      readyState: 1,
      send(serialized) {
        sent.push(JSON.parse(serialized) as WorkerEnvelope);
      },
    };
    const challengeNow = Date.now();
    await harness.handleEvidenceChallenge({
      schema: "mycellios-evidence-challenge/1",
      kind: "deployment-canary",
      challengeId: "challenge-session-isolation",
      nonce: Buffer.alloc(32, 7).toString("base64url"),
      sessionId: "coordinator-session",
      workerId: "worker-cell",
      issuedAt: new Date(challengeNow - 1_000).toISOString(),
      expiresAt: new Date(challengeNow + 60_000).toISOString(),
      deploymentId: capabilities.deployments[0]!.deploymentId,
      model: "regional-large",
      modelDigest: CELL_MODEL_DIGEST,
      activationId: CELL_ACTIVATION_ID,
      prompt: "measure",
      promptDigest: `sha256:${"c".repeat(64)}`,
      maxOutputTokens: 8,
      warmupSamples: 1,
      samples: 3,
    });
    expect(evidenceSessionIds).toHaveLength(4);
    expect(new Set(evidenceSessionIds).size).toBe(4);
    expect(evidenceSessionIds[0]).toContain("-warmup-0");
    expect(evidenceSessionIds.slice(1)).toEqual([
      expect.stringContaining("-sample-0"),
      expect.stringContaining("-sample-1"),
      expect.stringContaining("-sample-2"),
    ]);
    sent.length = 0;
    await harness.execute(payload({
      jobId: "cell-job",
      modelDigest: CELL_MODEL_DIGEST,
      request: {
        model: "regional-large",
        messages: [{ role: "user", content: "test" }],
        max_tokens: 8,
      },
    }));
    expect(messagePayload(sent, "task.complete")).toBeDefined();
    expect(harness.capabilities.deployments[0]).toMatchObject({
      tokensPerSecond: 1,
      throughputSource: "default",
      ttftMs: 60_000,
      verificationState: "pending",
    });

    expect(
      workerConfigSchema.safeParse({
        ...baseConfig(),
        capacityScope: "cell",
      }).success,
    ).toBe(false);
  });

  it("caps incomplete native SSE buffers at 1 MiB", async () => {
    const sseBaseUrl = await listen(servers, (request, response) => {
      if (request.url === "/health") {
        response.setHeader("content-type", "application/json");
        response.end(JSON.stringify({
          status: "ready",
          model: "test",
          artifact_identity: "sha256:test",
          pipeline_snapshot_identity: "test-activation",
          stages: 1,
          boundaries: [0, 1],
        }));
        return;
      }
      if (request.url === "/v1/chat/completions") {
        response.setHeader("content-type", "text/event-stream");
        response.end(`data: ${"x".repeat(1024 * 1024 + 1)}`);
      }
    });
    const sseAdapter = new MycelliosPipelineAdapter({
      baseUrl: sseBaseUrl,
      model: "test",
      modelDigest: "sha256:test",
      activationId: "test-activation",
    });
    await sseAdapter.probe();
    await expect(collect(sseAdapter)).rejects.toThrow(/buffer exceeded 1 MiB/);
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
  handleEvidenceChallenge(challenge: unknown): Promise<void>;
}

function baseConfig() {
  return workerConfigSchema.parse({
    region: "test",
    offeredVramMb: 4_096,
    limits: { maxConcurrency: 1, pauseWhenForeground: false },
    adapter: {
      kind: "mock",
      developmentOnly: true,
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
