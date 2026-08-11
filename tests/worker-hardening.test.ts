import { createServer, type RequestListener, type Server } from "node:http";
import { createConnection } from "node:net";
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
import type {
  LaunchAgent,
  LaunchProcessHandle,
} from "../src/distribution/launch-supervisor.js";
import { normalizeExecutorIsolationPolicy } from "../src/distribution/process-environment.js";
import type { PythonLaunchProcess } from "../src/distribution/python-launcher.js";

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

  it("cancels registration before a stopped worker can connect or retain its direct listener", async () => {
    let releaseRegistration!: () => void;
    const registrationRelease = new Promise<void>((resolve) => {
      releaseRegistration = resolve;
    });
    let registrationReceived!: () => void;
    const sawRegistration = new Promise<void>((resolve) => {
      registrationReceived = resolve;
    });
    let advertisedPort: number | undefined;
    let websocketUpgrades = 0;
    const coordinatorUrl = await listen(servers, (request, response) => {
      if (request.url !== "/internal/v1/workers/register") {
        response.statusCode = 404;
        response.end();
        return;
      }
      void (async () => {
        const chunks: Buffer[] = [];
        for await (const chunk of request) chunks.push(Buffer.from(chunk));
        const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as {
          capabilities?: {
            distributedExecutor?: {
              directTransport?: { candidates?: Array<{ port?: number }> };
            };
          };
        };
        advertisedPort = body.capabilities?.distributedExecutor?.directTransport
          ?.candidates?.[0]?.port;
        registrationReceived();
        await registrationRelease;
        if (response.destroyed) return;
        response.statusCode = 201;
        response.setHeader("content-type", "application/json");
        response.end(JSON.stringify({
          workerId: "worker-start-stop-race",
          protocolVersion: 1,
        }));
      })().catch((error: unknown) => {
        if (!response.destroyed) response.destroy(error as Error);
      });
    });
    servers.at(-1)!.on("upgrade", (_request, socket) => {
      websocketUpgrades += 1;
      socket.destroy();
    });
    const launchAgent: LaunchAgent = {
      id: "unused-start-stop-race-agent",
      async start() {
        throw new Error("The registration race must not launch a runtime");
      },
    };
    const agent = new WorkerAgent(baseConfig(), {
      coordinatorUrl,
      reconnect: false,
      advertiseDeployment: false,
      hardwareProbe: async () => ({
        hostname: "start-stop-race-worker",
        platform: process.platform,
        ramMb: 8_192,
        gpus: [],
      }),
      distributedExecutor: {
        nodeId: "start-stop-race-node",
        stageHost: "start-stop-race-node.relay",
        stagePort: 9_850,
        launchAgent,
        computeMode: "cpu-only",
        cpuEligible: true,
        directTransport: {
          enabled: true,
          listenHost: "127.0.0.1",
          candidateHosts: ["127.0.0.1"],
          publicPortMapping: false,
        },
      },
      logger: { info() {}, warn() {}, error() {} },
    });

    const run = agent.start();
    await sawRegistration;
    expect(advertisedPort).toBeTypeOf("number");
    expect(await portAcceptsConnections(advertisedPort!)).toBe(true);

    const stopped = agent.stop();
    releaseRegistration();
    await stopped;
    await run;

    expect(websocketUpgrades).toBe(0);
    expect(agent.workerId).toBeUndefined();
    expect(agent.isConnected).toBe(false);
    expect(agent.isReady).toBe(false);
    await expectPortClosed(advertisedPort!);
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

    sent.length = 0;
    await harness.execute(payload({ jobId: "expired", deadlineAt: Date.now() - 1 }));
    expect(messagePayload(sent, "lease.reject")).toMatchObject({
      jobId: "expired",
      reason: "deadline_exceeded",
    });
  });

  it("enforces local schedule and model policy before accepting a lease", async () => {
    const agent = new WorkerAgent(baseConfig(), {
      coordinatorUrl: "http://127.0.0.1:9999",
      reconnect: false,
      workAdmissionPolicy: (model) => model === null ? null : model === "test-model" ? "node_schedule_closed" : "node_model_not_allowed",
      logger: { info() {}, warn() {}, error() {} },
    });
    const sent: WorkerEnvelope[] = [];
    const harness = agent as unknown as AgentHarness;
    harness.registeredWorkerId = "worker-policy";
    harness.capabilities = capabilities();
    harness.socket = { readyState: 1, send(serialized) { sent.push(JSON.parse(serialized) as WorkerEnvelope); } };
    await harness.execute(payload({ jobId: "policy-denied", modelDigest: "sha256:test" }));
    expect(messagePayload(sent, "lease.reject")).toMatchObject({ jobId: "policy-denied", reason: "node_schedule_closed" });
    expect(sent.some((message) => message.type === "lease.accept")).toBe(false);
  });

  it("replays an identical runtime start idempotently and tears down the original handle", async () => {
    let starts = 0;
    let stops = 0;
    let resolveExit!: (exit: { code: number | null; signal: NodeJS.Signals | null }) => void;
    const exited = new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve) => {
      resolveExit = resolve;
    });
    const handle: LaunchProcessHandle = {
      ready: Promise.resolve(),
      exited,
      async stop() {
        stops += 1;
        resolveExit({ code: 0, signal: null });
      },
      output: () => ({
        stdout: "ready\n",
        stderr: "",
        stdoutTruncated: false,
        stderrTruncated: false,
      }),
    };
    const launchAgent: LaunchAgent = {
      id: "runtime-idempotency",
      async start() {
        starts += 1;
        return handle;
      },
    };
    const agent = new WorkerAgent(baseConfig(), {
      coordinatorUrl: "http://127.0.0.1:9999",
      reconnect: false,
      advertiseDeployment: false,
      distributedExecutor: {
        nodeId: "node-a",
        stageHost: "node-a.relay",
        stagePort: 9_850,
        launchAgent,
        computeMode: "cpu-only",
        cpuEligible: true,
      },
      logger: { info() {}, warn() {}, error() {} },
    });
    const sent: WorkerEnvelope[] = [];
    const closed: Array<{ code: number | undefined; reason: string | undefined }> = [];
    const harness = agent as unknown as AgentHarness;
    harness.registeredWorkerId = "worker-runtime";
    harness.socket = {
      readyState: 1,
      send(serialized) { sent.push(JSON.parse(serialized) as WorkerEnvelope); },
      close(code, reason) { closed.push({ code, reason }); },
    };
    const process = {
      kind: "root-engine",
      processId: "root-a",
      anchor: { memberId: "node-a", endpoint: { host: "node-a.relay", port: 9_850 } },
      command: { executable: "python", args: ["-m", "runtime"] },
      isolation: normalizeExecutorIsolationPolicy(),
    } as PythonLaunchProcess;
    const request = {
      launchId: "launch-a",
      pipelineId: "pipeline-a",
      nodeId: "node-a",
      process,
    };
    harness.authorizedRuntimeProcesses.set(process.processId, JSON.stringify(process));
    harness.preparedRuntimeProcesses.set(process.processId, process);

    await harness.startDistributedRuntime("request-a", request);
    await Promise.resolve();
    await harness.startDistributedRuntime("request-a", structuredClone(request));

    expect(starts).toBe(1);
    expect(sent.filter((message) => message.type === "runtime.ready")).toHaveLength(2);
    expect(harness.runtimeProcesses.size).toBe(1);

    await harness.startDistributedRuntime("request-a", {
      ...request,
      launchId: "conflicting-launch",
    });
    expect(starts).toBe(1);
    expect(harness.runtimeProcesses.size).toBe(1);
    expect(closed).toEqual([{ code: 4400, reason: "runtime start identity conflict" }]);

    await harness.resetDistributedRuntime("test_teardown");
    expect(stops).toBe(1);
    expect(harness.runtimeProcesses.size).toBe(0);
    expect(harness.runtimeStartRequests.size).toBe(0);
    expect(harness.readyRuntimeOutputs.size).toBe(0);
    await harness.runtimeTunnel?.close();
  });
});

interface AgentHarness {
  registeredWorkerId?: string;
  capabilities: WorkerCapabilities;
  socket: {
    readyState: number;
    send(serialized: string): void;
    close?(code?: number, reason?: string): void;
  };
  adapter: InferenceAdapter;
  execute(payload: JobPayload): Promise<void>;
  buildCapabilities(): Promise<WorkerCapabilities>;
  handleEvidenceChallenge(challenge: unknown): Promise<void>;
  authorizedRuntimeProcesses: Map<string, string>;
  preparedRuntimeProcesses: Map<string, PythonLaunchProcess>;
  runtimeProcesses: Map<string, LaunchProcessHandle>;
  runtimeStartRequests: Map<string, string>;
  readyRuntimeOutputs: Map<string, unknown>;
  runtimeTunnel: { close(): Promise<void> } | null;
  startDistributedRuntime(requestId: string, input: unknown): Promise<void>;
  resetDistributedRuntime(reason: string): Promise<void>;
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

async function expectPortClosed(port: number): Promise<void> {
  const deadline = Date.now() + 1_000;
  while (await portAcceptsConnections(port)) {
    if (Date.now() >= deadline) {
      throw new Error(`Expected direct listener ${port} to be closed`);
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

function portAcceptsConnections(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = createConnection({ host: "127.0.0.1", port });
    let settled = false;
    const finish = (connected: boolean) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      socket.destroy();
      resolve(connected);
    };
    const timeout = setTimeout(() => finish(false), 250);
    socket.once("connect", () => finish(true));
    socket.once("error", () => finish(false));
  });
}
