import { afterEach, describe, expect, it } from "vitest";
import { createServer, type Server } from "node:http";
import {
  LaunchCancelledError,
  PythonLaunchSupervisor,
  type LaunchAgent,
  type LaunchAgentStartRequest,
  type LaunchCapturedOutput,
  type LaunchProcessExit,
  type LaunchProcessHandle,
} from "../src/distribution/launch-supervisor.js";
import {
  HttpLaunchAgent,
  LaunchAgentRpcHttpError,
  LaunchAgentRpcServer,
  LaunchAgentRpcTimeoutError,
  launchAgentRpcHandleId,
  type LaunchAgentRpcServerAddress,
} from "../src/distribution/launch-agent-rpc.js";
import {
  compilePythonLaunchDescription,
  type PythonLaunchCompilerOptions,
  type PythonPipelineLaunchDescription,
} from "../src/distribution/python-launcher.js";
import {
  buildRuntimePipelineManifest,
  type RuntimePlanRequest,
} from "../src/distribution/runtime-manifest.js";
import type {
  DistributedModelProfile,
  DistributionPlan,
  DistributionWorkload,
} from "../src/distribution/types.js";

const MIB = 1024 * 1024;
const servers: LaunchAgentRpcServer[] = [];
const rawServers: Server[] = [];

afterEach(async () => {
  await Promise.all(servers.splice(0).map((server) => server.close("test_cleanup")));
  await Promise.all(rawServers.splice(0).map(closeRawServer));
});

describe("HTTP LaunchAgent RPC", () => {
  it("derives a valid stable agent id for an IPv6 endpoint", () => {
    const first = new HttpLaunchAgent({ endpoint: "http://[::1]:9750" });
    const second = new HttpLaunchAgent({ endpoint: "http://[::1]:9750" });
    expect(first.id).toMatch(/^http-launch-agent:[a-f0-9]{24}$/);
    expect(second.id).toBe(first.id);
  });

  it("materializes an entire gdlp-python-launch/2 route through loopback agents", async () => {
    const description = fixtureDescription();
    const fakeAgents = new Map<string, FakeAgent>();
    const clients = new Map<string, HttpLaunchAgent>();
    for (const nodeId of new Set(description.launchOrder.map((entry) => entry.anchor.memberId))) {
      const fake = new FakeAgent(`fake:${nodeId}`, { autoReady: true });
      const { address } = await serve(fake, nodeId);
      fakeAgents.set(nodeId, fake);
      clients.set(
        nodeId,
        new HttpLaunchAgent({
          endpoint: address.url,
          id: `rpc:${nodeId}`,
          pollIntervalMs: 2,
          requestTimeoutMs: 1_000,
          pollRequestTimeoutMs: 1_000,
        }),
      );
    }
    const supervisor = new PythonLaunchSupervisor(description, {
      resolveAgent: (nodeId) => clients.get(nodeId),
      readinessTimeoutMs: 2_000,
    });

    const running = await supervisor.start();
    expect(running.state).toBe("running");
    expect(
      [...fakeAgents.values()].flatMap((agent) => agent.starts).map((entry) => entry.process.processId),
    ).toHaveLength(description.launchOrder.length);
    expect(running.processes.every((process) => process.state === "ready")).toBe(true);

    const stopped = await supervisor.stop("rpc_e2e_complete");
    expect(stopped.state).toBe("stopped");
    expect([...fakeAgents.values()].reduce((total, agent) => total + agent.stopCount, 0)).toBe(
      description.launchOrder.length,
    );
  });

  it("deduplicates concurrent starts, latches ready across exit and bounds output", async () => {
    const request = fixtureRequest();
    const fake = new FakeAgent("fake:dedupe", {
      stdout: "€".repeat(512),
      stderr: "😀".repeat(512),
    });
    const { address } = await serve(fake, request.nodeId, {
      maxOutputBytesPerStream: 64,
    });
    const client = rpcClient(address, { maxOutputBytesPerStream: 64 });
    const signal = new AbortController().signal;

    const [first, second] = await Promise.all([
      client.start(request, signal),
      client.start(structuredClone(request), signal),
    ]);
    expect(fake.starts).toHaveLength(1);
    const local = fake.handles.get(request.process.processId)!;
    local.markReady();
    local.exit({ code: 23, signal: null });

    await expect(first.ready).resolves.toBeUndefined();
    await expect(second.ready).resolves.toBeUndefined();
    await expect(first.exited).resolves.toEqual({ code: 23, signal: null });
    await expect(second.exited).resolves.toEqual({ code: 23, signal: null });
    expect(Buffer.byteLength(first.output!().stdout, "utf8")).toBeLessThanOrEqual(64);
    expect(Buffer.byteLength(first.output!().stderr, "utf8")).toBeLessThanOrEqual(64);
    expect(first.output!().stdoutTruncated).toBe(true);
    expect(first.output!().stderrTruncated).toBe(true);
  });

  it("makes stop idempotent on both client and daemon", async () => {
    const request = fixtureRequest();
    const fake = new FakeAgent("fake:stop");
    const { address } = await serve(fake, request.nodeId);
    const handle = await rpcClient(address).start(request, new AbortController().signal);
    fake.handles.get(request.process.processId)!.markReady();
    await handle.ready;

    await Promise.all([handle.stop("operator_stop"), handle.stop("operator_stop")]);
    await expect(handle.exited).resolves.toEqual({ code: 0, signal: "SIGTERM" });
    expect(fake.stopCount).toBe(1);
  });

  it("allows a stop retry after a transient remote stop failure", async () => {
    const request = fixtureRequest();
    const fake = new FakeAgent("fake:stop-retry", { failStopAttempts: 1 });
    const { address } = await serve(fake, request.nodeId);
    const handle = await rpcClient(address).start(request, new AbortController().signal);
    fake.handles.get(request.process.processId)!.markReady();
    await handle.ready;

    await expect(handle.stop("first_stop")).rejects.toThrow("fake_stop_failed");
    await expect(handle.stop("retry_stop")).resolves.toBeUndefined();
    await expect(handle.exited).resolves.toEqual({ code: 0, signal: "SIGTERM" });
    expect(fake.stopAttempts).toBe(2);
    expect(fake.stopCount).toBe(1);
  });

  it("cleans up the remote allocation when lifecycle polling becomes unusable", async () => {
    const request = fixtureRequest();
    const fake = new FakeAgent("fake:poll-failure");
    const { address } = await serve(fake, request.nodeId);
    const client = rpcClient(address, { maxConsecutivePollErrors: 1 });
    client.poll = async () => {
      throw new Error("poll_transport_failed");
    };
    const handle = await client.start(request, new AbortController().signal);
    const readyFailure = handle.ready.catch((error: unknown) => error);
    const exitFailure = handle.exited.catch((error: unknown) => error);

    await waitFor(() => fake.stopCount === 1);
    await expect(readyFailure).resolves.toMatchObject({ message: "poll_transport_failed" });
    await expect(exitFailure).resolves.toMatchObject({ message: "poll_transport_failed" });
  });

  it("rejects readiness but preserves the exact exit when a process dies first", async () => {
    const request = fixtureRequest();
    const fake = new FakeAgent("fake:early-exit");
    const { address } = await serve(fake, request.nodeId);
    const handle = await rpcClient(address).start(request, new AbortController().signal);
    const ready = expect(handle.ready).rejects.toMatchObject({
      processId: request.process.processId,
    });
    const exited = expect(handle.exited).resolves.toEqual({ code: 7, signal: null });
    fake.handles.get(request.process.processId)!.exit({ code: 7, signal: null });
    await Promise.all([ready, exited]);
  });

  it("rejects a conflicting retry without spawning another process", async () => {
    const request = fixtureRequest();
    const fake = new FakeAgent("fake:conflict");
    const { address } = await serve(fake, request.nodeId);
    const client = rpcClient(address);
    const handle = await client.start(request, new AbortController().signal);
    fake.handles.get(request.process.processId)!.markReady();
    await handle.ready;
    const conflict = structuredClone(request);
    conflict.process.command.args.push("--different");

    await expect(client.start(conflict, new AbortController().signal)).rejects.toMatchObject({
      status: 409,
    } satisfies Partial<LaunchAgentRpcHttpError>);
    expect(fake.starts).toHaveLength(1);
    await handle.stop("test_cleanup");
  });

  it("accepts unequal tensor-parallel rank weights at the RPC boundary", async () => {
    const request = fixtureRequest();
    if (request.process.kind !== "remote-stage") {
      throw new Error("fixture_first_process_is_not_remote_stage");
    }
    const anchor = request.process.members[0]!;
    const peer = structuredClone(anchor);
    peer.nodeId = "weighted-peer";
    peer.endpoint = { host: "weighted-peer.internal", port: 25_001 };
    request.process.members.push(peer);
    request.process.cell = {
      mode: "tensor-parallel-cell",
      engine: "python-torch",
      collectiveBackend: "gloo",
      computeDtype: "float32",
      fixture: {
        schema: "gdlp-llama-cell-stage/2",
        location: "anchor-local",
        path: "fixtures/weighted-cell",
        layerCount: 2,
        manifestSha256: "a".repeat(64),
        shardSha256: ["b".repeat(64), "c".repeat(64)],
        rankMemory: [
          { fixedBytes: 3_000, kvBytesPerToken: 30, requiredBytes: 6_000 },
          { fixedBytes: 1_000, kvBytesPerToken: 10, requiredBytes: 2_000 },
        ],
      },
      worldSize: 2,
      rankMemberIds: [anchor.nodeId, peer.nodeId],
      rankWeights: [3, 1],
      rankDevices: ["cpu", "cpu"],
      operationTimeoutSeconds: 30,
    };
    const fake = new FakeAgent("fake:weighted-cell");
    const { address } = await serve(fake, request.nodeId);

    const handle = await rpcClient(address).start(request, new AbortController().signal);
    expect(fake.starts[0]!.process).toMatchObject({
      kind: "remote-stage",
      cell: { rankWeights: [3, 1] },
    });
    fake.handles.get(request.process.processId)!.markReady();
    await handle.ready;
    await handle.stop("weighted_complete");
  });

  it("reads the worst-case JSON expansion of bounded UTF-8 output", async () => {
    const request = fixtureRequest();
    const outputLimit = 32 * 1024;
    const fake = new FakeAgent("fake:escaped-output", {
      stdout: "\0".repeat(outputLimit),
      stderr: "\0".repeat(outputLimit),
    });
    const { address } = await serve(fake, request.nodeId, {
      maxOutputBytesPerStream: outputLimit,
    });
    const handle = await rpcClient(address, {
      maxOutputBytesPerStream: outputLimit,
    }).start(request, new AbortController().signal);

    expect(Buffer.byteLength(handle.output!().stdout, "utf8")).toBe(outputLimit);
    expect(Buffer.byteLength(handle.output!().stderr, "utf8")).toBe(outputLimit);
    fake.handles.get(request.process.processId)!.markReady();
    await handle.ready;
    await handle.stop("escaped_output_complete");
  });

  it("evicts an old completed tombstone before declaring process capacity exhausted", async () => {
    const firstRequest = fixtureRequest();
    const fake = new FakeAgent("fake:bounded-history");
    const { address } = await serve(fake, firstRequest.nodeId, { maxProcesses: 1 });
    const client = rpcClient(address);
    const first = await client.start(firstRequest, new AbortController().signal);
    const firstReady = first.ready.catch((error: unknown) => error);
    fake.handles.get(firstRequest.process.processId)!.exit({ code: 0, signal: null });
    await first.exited;
    await expect(firstReady).resolves.toBeInstanceOf(Error);

    const secondRequest = structuredClone(firstRequest);
    secondRequest.launchId = "rpc-second-launch";
    secondRequest.process.processId = "rpc-second-process";
    const second = await client.start(secondRequest, new AbortController().signal);
    expect(fake.starts).toHaveLength(2);
    fake.handles.get(secondRequest.process.processId)!.markReady();
    await second.ready;
    await second.stop("bounded_history_complete");
  });

  it("closes malformed envelopes and requests for another node before injection", async () => {
    const request = fixtureRequest();
    const fake = new FakeAgent("fake:validation");
    const { address } = await serve(fake, request.nodeId);
    const malformed = await fetch(`${address.url}/v1/processes`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        schema: "gdlp-launch-agent-start/1",
        request: { ...request, unexpected: true },
      }),
    });
    expect(malformed.status).toBe(400);

    const otherNode = structuredClone(request);
    otherNode.nodeId = "other-node";
    otherNode.process.anchor.memberId = "other-node";
    const wrongNode = await fetch(`${address.url}/v1/processes`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ schema: "gdlp-launch-agent-start/1", request: otherNode }),
    });
    expect(wrongNode.status).toBe(422);
    expect(fake.starts).toHaveLength(0);
  });

  it("times out a non-responsive injected start and aborts its allocation signal", async () => {
    const request = fixtureRequest();
    const fake = new FakeAgent("fake:timeout", { hangStart: true });
    const { address } = await serve(fake, request.nodeId, { startTimeoutMs: 20 });
    const client = rpcClient(address, { requestTimeoutMs: 1_000 });

    await expect(client.start(request, new AbortController().signal)).rejects.toMatchObject({
      status: 504,
    } satisfies Partial<LaunchAgentRpcHttpError>);
    await waitFor(() => fake.startSignals[0]?.aborted === true);
    expect(fake.starts).toHaveLength(1);
  });

  it("cancels an in-flight remote start without leaving the fake allocation alive", async () => {
    const request = fixtureRequest();
    const fake = new FakeAgent("fake:cancel", { hangStart: true });
    const { address } = await serve(fake, request.nodeId, { startTimeoutMs: 2_000 });
    const client = rpcClient(address, { requestTimeoutMs: 2_000 });
    const controller = new AbortController();
    const starting = client.start(request, controller.signal);
    await waitFor(() => fake.starts.length === 1);
    controller.abort(new LaunchCancelledError("test_cancel"));

    await expect(starting).rejects.toThrow("test_cancel");
    await waitFor(() => fake.startSignals[0]?.aborted === true);
  });

  it("stops a late handle returned by an agent that ignores its aborted start signal", async () => {
    const request = fixtureRequest();
    const fake = new FakeAgent("fake:late", { startDelayMs: 40 });
    const { address } = await serve(fake, request.nodeId, { startTimeoutMs: 10 });
    const client = rpcClient(address, { requestTimeoutMs: 1_000 });

    await expect(client.start(request, new AbortController().signal)).rejects.toMatchObject({
      status: 504,
    } satisfies Partial<LaunchAgentRpcHttpError>);
    await waitFor(() => fake.stopCount === 1);
    expect(fake.startSignals[0]!.aborted).toBe(true);
  });

  it("best-effort stops a malformed handle returned by an injected agent", async () => {
    const request = fixtureRequest();
    let stopCount = 0;
    const invalidAgent: LaunchAgent = {
      id: "fake:invalid-handle",
      async start() {
        return {
          ready: Promise.resolve(),
          exited: {} as Promise<LaunchProcessExit>,
          async stop() {
            stopCount += 1;
          },
        };
      },
    };
    const { address } = await serve(invalidAgent, request.nodeId);

    await expect(
      rpcClient(address).start(request, new AbortController().signal),
    ).rejects.toMatchObject({ status: 500 } satisfies Partial<LaunchAgentRpcHttpError>);
    await waitFor(() => stopCount >= 1);
  });

  it("preserves a timeout raised while reading a stalled response body and cleans up", async () => {
    const request = fixtureRequest();
    const handleId = launchAgentRpcHandleId(request);
    let stopRequests = 0;
    const server = createServer((incoming, response) => {
      if (incoming.url === "/v1/processes" && incoming.method === "POST") {
        response.writeHead(200, { "content-type": "application/json" });
        response.flushHeaders();
        const timer = setTimeout(() => {
          if (!response.destroyed) response.end(JSON.stringify(rpcSnapshot(request, handleId)));
        }, 200);
        response.once("close", () => clearTimeout(timer));
        return;
      }
      if (incoming.url === `/v1/processes/${handleId}/stop` && incoming.method === "POST") {
        stopRequests += 1;
        response.writeHead(200, { "content-type": "application/json" });
        response.end(
          JSON.stringify(
            rpcSnapshot(request, handleId, { code: 0, signal: "SIGTERM" }),
          ),
        );
        return;
      }
      response.writeHead(404).end();
    });
    rawServers.push(server);
    const address = await listenRawServer(server);
    const client = new HttpLaunchAgent({
      endpoint: address,
      requestTimeoutMs: 20,
      pollRequestTimeoutMs: 20,
      pollIntervalMs: 2,
    });

    await expect(client.start(request, new AbortController().signal)).rejects.toBeInstanceOf(
      LaunchAgentRpcTimeoutError,
    );
    expect(stopRequests).toBe(1);
  });

  it("stops the deterministic allocation when a start response has the wrong identity", async () => {
    const request = fixtureRequest();
    const handleId = launchAgentRpcHandleId(request);
    let stopRequests = 0;
    const server = createServer((incoming, response) => {
      response.setHeader("content-type", "application/json");
      if (incoming.url === "/v1/processes" && incoming.method === "POST") {
        response.end(
          JSON.stringify({
            ...rpcSnapshot(request, handleId),
            launchId: "wrong-launch",
          }),
        );
        return;
      }
      if (incoming.url === `/v1/processes/${handleId}/stop` && incoming.method === "POST") {
        stopRequests += 1;
        response.end(
          JSON.stringify(rpcSnapshot(request, handleId, { code: 0, signal: "SIGTERM" })),
        );
        return;
      }
      response.writeHead(404).end();
    });
    rawServers.push(server);
    const address = await listenRawServer(server);
    const client = new HttpLaunchAgent({ endpoint: address, requestTimeoutMs: 1_000 });

    await expect(client.start(request, new AbortController().signal)).rejects.toThrow(
      "launch_agent_rpc_start_returned_wrong_identity",
    );
    expect(stopRequests).toBe(1);
  });
});

interface FakeAgentOptions {
  autoReady?: boolean;
  hangStart?: boolean;
  startDelayMs?: number;
  stdout?: string;
  stderr?: string;
  failStopAttempts?: number;
}

class FakeHandle implements LaunchProcessHandle {
  readonly ready: Promise<void>;
  readonly exited: Promise<LaunchProcessExit>;
  private readonly readyDeferred = deferred<void>();
  private readonly exitDeferred = deferred<LaunchProcessExit>();
  private stopped = false;

  constructor(
    private readonly owner: FakeAgent,
    private readonly stdout: string,
    private readonly stderr: string,
  ) {
    this.ready = this.readyDeferred.promise;
    this.exited = this.exitDeferred.promise;
  }

  markReady(): void {
    this.readyDeferred.resolve();
  }

  exit(exit: LaunchProcessExit): void {
    this.exitDeferred.resolve(exit);
  }

  async stop(_reason: string): Promise<void> {
    if (this.stopped) return;
    this.owner.stopAttempts += 1;
    if (this.owner.stopAttempts <= (this.owner.failStopAttempts ?? 0)) {
      throw new Error("fake_stop_failed");
    }
    this.stopped = true;
    this.owner.stopCount += 1;
    this.exit({ code: 0, signal: "SIGTERM" });
  }

  output(): LaunchCapturedOutput {
    return {
      stdout: this.stdout,
      stderr: this.stderr,
      stdoutTruncated: false,
      stderrTruncated: false,
    };
  }
}

class FakeAgent implements LaunchAgent {
  readonly starts: LaunchAgentStartRequest[] = [];
  readonly startSignals: AbortSignal[] = [];
  readonly handles = new Map<string, FakeHandle>();
  stopCount = 0;
  stopAttempts = 0;

  constructor(
    readonly id: string,
    private readonly options: FakeAgentOptions = {},
  ) {}

  get failStopAttempts(): number | undefined {
    return this.options.failStopAttempts;
  }

  async start(
    request: LaunchAgentStartRequest,
    signal: AbortSignal,
  ): Promise<LaunchProcessHandle> {
    this.starts.push(structuredClone(request));
    this.startSignals.push(signal);
    if (this.options.hangStart) {
      return new Promise((_resolve, reject) => {
        const abort = () => reject(signal.reason ?? new Error("fake_aborted"));
        signal.addEventListener("abort", abort, { once: true });
        if (signal.aborted) abort();
      });
    }
    if (this.options.startDelayMs !== undefined) {
      await new Promise((resolve) => setTimeout(resolve, this.options.startDelayMs));
    }
    const handle = new FakeHandle(
      this,
      this.options.stdout ?? `stdout:${request.process.processId}`,
      this.options.stderr ?? `stderr:${request.process.processId}`,
    );
    this.handles.set(request.process.processId, handle);
    if (this.options.autoReady) queueMicrotask(() => handle.markReady());
    return handle;
  }
}

async function serve(
  agent: LaunchAgent,
  nodeId: string,
  options: Omit<ConstructorParameters<typeof LaunchAgentRpcServer>[0], "agent" | "nodeId"> = {},
): Promise<{ server: LaunchAgentRpcServer; address: LaunchAgentRpcServerAddress }> {
  const server = new LaunchAgentRpcServer({ agent, nodeId, ...options });
  servers.push(server);
  return { server, address: await server.listen(0, "127.0.0.1") };
}

function rpcClient(
  address: LaunchAgentRpcServerAddress,
  options: Partial<ConstructorParameters<typeof HttpLaunchAgent>[0]> = {},
): HttpLaunchAgent {
  return new HttpLaunchAgent({
    endpoint: address.url,
    pollIntervalMs: 2,
    requestTimeoutMs: 1_000,
    pollRequestTimeoutMs: 1_000,
    ...options,
  });
}

function fixtureRequest(): LaunchAgentStartRequest {
  const description = fixtureDescription();
  const process = description.launchOrder[0]!;
  return {
    launchId: description.launchId,
    pipelineId: description.pipelineId,
    nodeId: process.anchor.memberId,
    process: structuredClone(process),
  };
}

function fixtureDescription(): PythonPipelineLaunchDescription {
  const model: DistributedModelProfile = {
    id: "rpc-model",
    layers: Array.from({ length: 6 }, (_, index) => ({
      index,
      weightBytes: 20 * MIB,
      activationElements: 256,
      kvBytesPerToken: 64,
      decodeMsAtUnit: 1,
      prefillMsPerTokenAtUnit: 0.1,
    })),
    embeddingBytes: MIB,
    lmHeadBytes: MIB,
    runtimeOverheadBytesPerStage: 2 * MIB,
    embeddingDecodeMsAtUnit: 0.1,
    lmHeadDecodeMsAtUnit: 0.1,
    embeddingPrefillMsPerTokenAtUnit: 0.02,
    lmHeadPrefillMsPerTokenAtUnit: 0.02,
  };
  const workload: DistributionWorkload = {
    promptTokens: 16,
    outputTokens: 8,
    contextTokens: 32,
    concurrentSequences: 2,
    maxStages: 3,
    maxQualityLoss: 0,
    minRouteAvailability: 0.8,
    batchWindowMs: 1,
    p95: false,
  };
  const plan: DistributionPlan = {
    algorithm: "rpc-fixture",
    codec: "fp16",
    microBatchSize: 2,
    prefillChunkTokens: 8,
    stages: Array.from({ length: 3 }, (_, index) => ({
      nodeId: `node-${index}`,
      layerStart: index * 2,
      layerEnd: (index + 1) * 2,
    })),
  };
  const nodes = Array.from({ length: 3 }, (_, index) => ({
    id: `node-${index}`,
    region: `region-${index}`,
    memoryBytes: 64 * MIB,
    reserveBytes: 4 * MIB,
    decodeScale: 1,
    prefillScale: 1,
    codecScale: 1,
    batchGain: 0.2,
    maxBatchSpeedup: 1.5,
    powerWatts: 60,
    availability: 0.999,
    endpoint: { host: `node-${index}.internal`, port: 23_000 + index },
  }));
  const links = [];
  for (const from of nodes) {
    for (const to of nodes) {
      if (from.id === to.id) continue;
      links.push({
        from: from.id,
        to: to.id,
        oneWayLatencyMs: 2,
        jitterP95Ms: 0,
        bandwidthMbps: 1_000,
        lossRate: 0,
        availability: 0.999,
      });
    }
  }
  const request: RuntimePlanRequest = {
    model,
    modelRevision: "sha256:rpc-r1",
    topology: { nodes, links },
    workload,
    phasePlans: { prefill: plan, decode: plan },
  };
  const options: PythonLaunchCompilerOptions = {
    apiEndpoint: { host: "0.0.0.0", port: 8_081 },
    returnEndpoint: { host: "root.internal", port: 30_000 },
    returnBindHost: "0.0.0.0",
  };
  return compilePythonLaunchDescription(buildRuntimePipelineManifest(request), options);
}

async function waitFor(predicate: () => boolean, timeoutMs = 1_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error("condition_not_reached");
    await new Promise((resolve) => setTimeout(resolve, 2));
  }
}

function rpcSnapshot(
  request: LaunchAgentStartRequest,
  handleId: string,
  exit: LaunchProcessExit | null = null,
): object {
  return {
    schema: "gdlp-launch-agent-process/1",
    handleId,
    launchId: request.launchId,
    processId: request.process.processId,
    state: exit === null ? "starting" : "exited",
    ready: false,
    exit,
    output: {
      stdout: "",
      stderr: "",
      stdoutTruncated: false,
      stderrTruncated: false,
    },
  };
}

async function listenRawServer(server: Server): Promise<string> {
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.removeListener("error", reject);
      resolve();
    });
  });
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("raw_server_has_no_address");
  return `http://127.0.0.1:${address.port}`;
}

async function closeRawServer(server: Server): Promise<void> {
  if (!server.listening) return;
  await new Promise<void>((resolve, reject) => {
    server.close((error) => (error ? reject(error) : resolve()));
    server.closeAllConnections();
  });
}

function deferred<T>(): {
  promise: Promise<T>;
  resolve(value: T): void;
} {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}
