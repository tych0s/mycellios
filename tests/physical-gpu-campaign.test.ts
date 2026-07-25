import { describe, expect, it, vi } from "vitest";
import {
  runPhysicalGpuCampaign,
  type PhysicalGpuCampaignAgentBinding,
  type PhysicalGpuCampaignCanary,
  type PhysicalGpuCampaignDependencies,
  type PhysicalGpuCampaignInput,
  type PhysicalGpuCampaignSupervisor,
} from "../src/distribution/physical-gpu-campaign.js";
import type {
  LaunchAgent,
  LaunchAgentStartRequest,
  LaunchCapturedOutput,
  LaunchProcessExit,
  LaunchProcessHandle,
  LaunchSupervisorSnapshot,
} from "../src/distribution/launch-supervisor.js";
import {
  HttpLaunchAgent,
  LaunchAgentRpcServer,
} from "../src/distribution/launch-agent-rpc.js";
import {
  compilePythonLaunchDescription,
  type PythonPipelineLaunchDescription,
} from "../src/distribution/python-launcher.js";
import {
  buildRuntimePipelineManifest,
  type RuntimePlanRequest,
} from "../src/distribution/runtime-manifest.js";

const MIB = 1024 * 1024;
const SNAPSHOT_IDENTITY = "9223372036854775931";

describe("physical GPU campaign collector", () => {
  it("runs the exact lifecycle and derives throughput only from real usage tokens", async () => {
    const fixture = campaignFixture();
    const log: string[] = [];
    const supervisor = fakeSupervisor(fixture.launch, log);
    const healthCalls = new Map<string, number>();
    const dependencies: PhysicalGpuCampaignDependencies = {
      now: incrementalClock(),
      supervisor: () => supervisor,
      health: async (binding) => {
        const call = (healthCalls.get(binding.nodeId) ?? 0) + 1;
        healthCalls.set(binding.nodeId, call);
        log.push(`health:${binding.nodeId}:${call}`);
        return agentHealth(binding, 0);
      },
      fetch: apiFetch(fixture, log),
    };

    const observation = await runPhysicalGpuCampaign(fixture.input, dependencies);

    expect(observation.failures).toEqual([]);
    expect(observation.passed).toBe(true);
    expect(observation.lifecycle.cleanupPassed).toBe(true);
    expect(observation.lifecycle.supervisorStarted?.state).toBe("running");
    expect(observation.lifecycle.supervisorStopped?.state).toBe("stopped");
    expect(observation.canaries.pre).toHaveLength(3);
    expect(observation.canaries.post).toHaveLength(3);
    expect(observation.canaries.pre.every((item) => item.passed)).toBe(true);
    expect(observation.canaries.post.every((item) => item.passed)).toBe(true);
    expect(observation.samples).toHaveLength((fixture.input.warmups + fixture.input.iterations) * 3);
    expect(observation.summary.measuredRequests).toBe(15);
    // The request asks for four tokens but every deterministic stop emits two.
    // A nominal max_tokens calculation would report 60 instead of the real 30.
    expect(observation.summary.actualCompletionTokens).toBe(30);
    expect(observation.summary.aggregateOutputTokensPerSecondIncludingTtft).toBeGreaterThan(0);
    expect(observation.summary.byConcurrency.map((row) => row.concurrency)).toEqual([1, 2]);
    const measured = observation.samples.find((sample) => sample.phase === "measure")!;
    expect(measured.completionTokens).toBe(2);
    expect(measured.serverTtftMs).toBe(2);
    expect(measured.serverTpotMs).toBe(3);
    expect(measured.clientFirstContentMs).not.toBe(measured.serverTtftMs);
    expect(measured.clientResponseMs).toBeGreaterThan(measured.clientFirstContentMs!);
    expect(log.indexOf("supervisor:start")).toBeGreaterThan(log.indexOf("health:node-a:1"));
    expect(log.indexOf("supervisor:stop")).toBeLessThan(log.indexOf("health:node-a:2"));
  });

  it("rejects an underspecified campaign before health, fetch, or supervisor side effects", async () => {
    const fixture = campaignFixture();
    const supervisorFactory = vi.fn();
    const health = vi.fn();
    const fetchImpl = vi.fn();

    await expect(
      runPhysicalGpuCampaign(
        { ...fixture.input, iterations: 4 },
        { supervisor: supervisorFactory, health, fetch: fetchImpl },
      ),
    ).rejects.toThrow("physical_gpu_campaign_iterations_must_be_an_integer");
    expect(supervisorFactory).not.toHaveBeenCalled();
    expect(health).not.toHaveBeenCalled();
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("does not launch when an agent is dirty and still performs the final empty-agent audit", async () => {
    const fixture = campaignFixture();
    const supervisorFactory = vi.fn();
    const fetchImpl = vi.fn();
    const calls = new Map<string, number>();

    const observation = await runPhysicalGpuCampaign(fixture.input, {
      now: incrementalClock(),
      supervisor: supervisorFactory,
      fetch: fetchImpl,
      health: async (binding) => {
        const count = (calls.get(binding.nodeId) ?? 0) + 1;
        calls.set(binding.nodeId, count);
        return agentHealth(binding, count === 1 && binding.nodeId === "node-a" ? 1 : 0);
      },
    });

    expect(observation.passed).toBe(false);
    expect(supervisorFactory).not.toHaveBeenCalled();
    expect(fetchImpl).not.toHaveBeenCalled();
    expect(observation.lifecycle.agentHealthBefore.some((item) => !item.passed)).toBe(true);
    expect(observation.lifecycle.agentHealthAfter.every((item) => item.passed)).toBe(true);
  });

  it("fails a bad SSE hash but still runs post canaries, stops, and verifies zero residuals", async () => {
    const fixture = campaignFixture();
    const log: string[] = [];
    const supervisor = fakeSupervisor(fixture.launch, log);
    let corrupted = false;
    const fetchImpl = apiFetch(fixture, log, (body, response) => {
      if (body.stream === true && !corrupted) {
        corrupted = true;
        return response.replace(
          fixture.canaries[0]!.expectedOutputTokenIdsSha256,
          `sha256:${"f".repeat(64)}`,
        );
      }
      return response;
    });

    const observation = await runPhysicalGpuCampaign(fixture.input, {
      now: incrementalClock(),
      supervisor: () => supervisor,
      health: async (binding) => agentHealth(binding, 0),
      fetch: fetchImpl,
    });

    expect(observation.passed).toBe(false);
    expect(observation.samples.some((sample) => !sample.passed)).toBe(true);
    expect(observation.failures.some((failure) => failure.phase === "benchmark")).toBe(true);
    expect(observation.canaries.post).toHaveLength(3);
    expect(observation.canaries.post.every((item) => item.passed)).toBe(true);
    expect(observation.lifecycle.cleanupPassed).toBe(true);
    expect(log).toContain("supervisor:stop");
  });

  it("never passes when the post-stop agent audit reports a residual process", async () => {
    const fixture = campaignFixture();
    const supervisor = fakeSupervisor(fixture.launch, []);
    const calls = new Map<string, number>();
    const observation = await runPhysicalGpuCampaign(fixture.input, {
      now: incrementalClock(),
      supervisor: () => supervisor,
      fetch: apiFetch(fixture, []),
      health: async (binding) => {
        const count = (calls.get(binding.nodeId) ?? 0) + 1;
        calls.set(binding.nodeId, count);
        return agentHealth(binding, count === 2 && binding.nodeId === "node-b" ? 1 : 0);
      },
    });

    expect(observation.passed).toBe(false);
    expect(observation.lifecycle.cleanupPassed).toBe(false);
    expect(observation.lifecycle.agentHealthAfter.find((item) => item.expectedNodeId === "node-b")?.passed)
      .toBe(false);
  });

  it("collects every launch-agent health observation concurrently", async () => {
    const fixture = campaignFixture();
    let inFlight = 0;
    let maximumInFlight = 0;
    const observation = await runPhysicalGpuCampaign(fixture.input, {
      now: incrementalClock(),
      supervisor: () => fakeSupervisor(fixture.launch, []),
      fetch: apiFetch(fixture, []),
      health: async (binding) => {
        inFlight += 1;
        maximumInFlight = Math.max(maximumInFlight, inFlight);
        await new Promise((resolve) => setTimeout(resolve, 5));
        inFlight -= 1;
        return agentHealth(binding, 0);
      },
    });

    expect(observation.passed).toBe(true);
    expect(maximumInFlight).toBe(fixture.input.agents.length);
  });

  it("times out a hung API health request and still stops the supervisor", async () => {
    const fixture = campaignFixture();
    const log: string[] = [];
    let requestWasAborted = false;
    const observation = await runPhysicalGpuCampaign(fixture.input, {
      now: incrementalClock(),
      supervisor: () => fakeSupervisor(fixture.launch, log),
      health: async (binding) => agentHealth(binding, 0),
      timeouts: { apiHealthMs: 20 },
      fetch: async (_input, init) =>
        new Promise<Response>((_resolve, reject) => {
          const signal = init?.signal;
          const abort = () => {
            requestWasAborted = true;
            reject(signal?.reason ?? new Error("test_api_health_aborted"));
          };
          signal?.addEventListener("abort", abort, { once: true });
          if (signal?.aborted) abort();
        }),
    });

    expect(observation.passed).toBe(false);
    expect(requestWasAborted).toBe(true);
    expect(observation.failures.some((failure) =>
      failure.message.includes("physical_gpu_campaign_timeout:api_health:20"),
    )).toBe(true);
    expect(log).toContain("supervisor:stop");
    expect(observation.lifecycle.cleanupPassed).toBe(true);
  });

  it("bounds SSE inactivity, cancels the stream and still performs cleanup", async () => {
    const fixture = campaignFixture();
    const log: string[] = [];
    const fallback = apiFetch(fixture, log);
    let stalled = false;
    let streamCancelled = false;
    const encoder = new TextEncoder();
    const observation = await runPhysicalGpuCampaign(fixture.input, {
      now: incrementalClock(),
      supervisor: () => fakeSupervisor(fixture.launch, log),
      health: async (binding) => agentHealth(binding, 0),
      timeouts: { apiRequestMs: 500, sseIdleMs: 20 },
      fetch: async (input, init) => {
        const body = init?.body === undefined
          ? null
          : JSON.parse(String(init.body)) as { stream?: boolean };
        if (!stalled && body?.stream === true) {
          stalled = true;
          const firstChunk = `data: ${JSON.stringify({
            id: "chatcmpl-stalled",
            object: "chat.completion.chunk",
            model: fixture.launch.configuration.publicModelName,
            choices: [{ index: 0, delta: { content: "x" }, finish_reason: null }],
          })}\n\n`;
          const stream = new ReadableStream<Uint8Array>({
            start(controller) {
              controller.enqueue(encoder.encode(firstChunk));
            },
            cancel() {
              streamCancelled = true;
            },
          });
          return new Response(stream, {
            status: 200,
            headers: { "content-type": "text/event-stream" },
          });
        }
        return fallback(input, init);
      },
    });

    expect(observation.passed).toBe(false);
    expect(streamCancelled).toBe(true);
    expect(observation.samples.some((sample) =>
      sample.error?.includes("physical_gpu_campaign_timeout:sse_idle:20"),
    )).toBe(true);
    expect(log).toContain("supervisor:stop");
    expect(observation.lifecycle.cleanupPassed).toBe(true);
  });

  it("bounds a hung supervisor stop and still completes the final agent audit", async () => {
    const fixture = campaignFixture();
    const base = fakeSupervisor(fixture.launch, []);
    let healthCalls = 0;
    const observation = await runPhysicalGpuCampaign(fixture.input, {
      now: incrementalClock(),
      supervisor: () => ({
        ...base,
        stop: async () => new Promise<never>(() => undefined),
      }),
      fetch: apiFetch(fixture, []),
      health: async (binding) => {
        healthCalls += 1;
        return agentHealth(binding, 0);
      },
      timeouts: { cleanupStopMs: 20 },
    });

    expect(observation.passed).toBe(false);
    expect(observation.lifecycle.cleanupAttempted).toBe(true);
    expect(observation.lifecycle.cleanupPassed).toBe(false);
    expect(observation.failures.some((failure) =>
      failure.phase === "cleanup_stop" &&
      failure.message.includes("physical_gpu_campaign_timeout:cleanup_stop:20"),
    )).toBe(true);
    expect(observation.lifecycle.agentHealthAfter.every((item) => item.passed)).toBe(true);
    expect(healthCalls).toBe(fixture.input.agents.length * 2);
  });

  it("passes cleanup through real RPC agents while retaining idempotency tombstones", async () => {
    const fixture = campaignFixture();
    const servers: LaunchAgentRpcServer[] = [];
    try {
      const bindings: PhysicalGpuCampaignAgentBinding[] = [];
      for (const nodeId of new Set(
        fixture.launch.launchOrder.map((process) => process.anchor.memberId),
      )) {
        const backing = new ImmediateRpcCampaignAgent(`local-process:${nodeId}`);
        const server = new LaunchAgentRpcServer({
          agent: backing,
          nodeId,
          buildIdentity: {
            schema: "mycellios-native-build-provenance/1",
            version: "0.2.38",
            sourceId: `sha256:${"1".repeat(64)}`,
          },
        });
        servers.push(server);
        const address = await server.listen(0, "127.0.0.1");
        bindings.push({
          nodeId,
          agent: new HttpLaunchAgent({
            endpoint: address.url,
            id: backing.id,
            pollIntervalMs: 1,
            requestTimeoutMs: 1_000,
          }),
        });
      }

      const observation = await runPhysicalGpuCampaign(
        { ...fixture.input, agents: bindings },
        { now: incrementalClock(), fetch: apiFetch(fixture, []) },
      );

      expect(observation.passed).toBe(true);
      expect(observation.lifecycle.agentHealthAfter.every((item) =>
        item.health !== null &&
        item.health.activeProcesses === 0 &&
        item.health.retainedTombstones > 0,
      )).toBe(true);
    } finally {
      await Promise.all(servers.map((server) => server.close("campaign_rpc_test_complete")));
    }
  });

  it("rejects a physical campaign before launch when agents do not share one exact build", async () => {
    const fixture = campaignFixture();
    const observation = await runPhysicalGpuCampaign(fixture.input, {
      now: incrementalClock(),
      fetch: apiFetch(fixture, []),
      health: async (binding) => ({
        ...agentHealth(binding, 0),
        buildIdentity: {
          schema: "mycellios-native-build-provenance/1",
          version: "0.2.38",
          sourceId: binding.nodeId.endsWith("a")
            ? `sha256:${"1".repeat(64)}`
            : `sha256:${"2".repeat(64)}`,
        },
      }),
    });

    expect(observation.passed).toBe(false);
    expect(observation.lifecycle.supervisorStarted).toBeNull();
    expect(observation.failures).toEqual(expect.arrayContaining([
      expect.objectContaining({
        phase: "campaign",
        message: "physical_gpu_campaign_agent_build_cohort_is_inconsistent",
      }),
    ]));
  });
});

interface CampaignFixture {
  launch: PythonPipelineLaunchDescription;
  input: PhysicalGpuCampaignInput;
  canaries: PhysicalGpuCampaignCanary[];
}

function campaignFixture(): CampaignFixture {
  const launch = compilePythonLaunchDescription(buildRuntimePipelineManifest(runtimeRequest()), {
    apiEndpoint: { host: "0.0.0.0", port: 18_081 },
    returnEndpoint: { host: "node-a.internal", port: 30_000 },
    returnBindHost: "0.0.0.0",
    publicModelName: "distributed-test",
    runtimeModel: {
      source: "fixture/model",
      revision: "0123456789abcdef",
      snapshotIdentity: SNAPSHOT_IDENTITY,
    },
  });
  const agents = [...new Set(launch.launchOrder.map((process) => process.anchor.memberId))].map(
    (nodeId, index): PhysicalGpuCampaignAgentBinding => ({
      nodeId,
      agent: inertAgent(`agent-${index}`),
    }),
  );
  const canaries = ["a", "b", "c"].map(
    (id, index): PhysicalGpuCampaignCanary => ({
      id: `canary-${id}`,
      messages: [{ role: "user", content: `case-${id}` }],
      maxTokens: 4,
      expectedOutputTokenIdsSha256: `sha256:${String(index + 1).repeat(64)}`,
      expectedCompletionTokens: 2,
      expectedFinishReason: "stop",
    }),
  );
  return {
    launch,
    canaries,
    input: {
      launch,
      agents,
      apiBaseUrl: "http://root.internal:18081",
      canaries,
      warmups: 1,
      iterations: 5,
      concurrencies: [1, 2],
    },
  };
}

function runtimeRequest(): RuntimePlanRequest {
  const nodes = ["node-a", "node-b"].map((id, index) => ({
    id,
    region: "test-lan",
    memoryBytes: 256 * MIB,
    reserveBytes: 16 * MIB,
    decodeScale: 1,
    prefillScale: 1,
    codecScale: 1,
    batchGain: 0.2,
    maxBatchSpeedup: 1.5,
    powerWatts: 75,
    availability: 0.999,
    endpoint: { host: `${id}.internal`, port: 22_000 + index },
    backend: {
      engine: "python-transformers" as const,
      version: "1",
      modelFormats: ["safetensors"],
      executionModes: ["layer-range"],
    },
    capabilities: {
      deviceKinds: ["gpu" as const],
      computeApis: ["cuda" as const],
      weightDtypes: ["fp16" as const],
      activationCodecs: ["fp16" as const],
      features: ["layer-range" as const, "kv-reuse" as const, "kv-transfer" as const],
    },
  }));
  const stages = nodes.map((node, index) => ({
    nodeId: node.id,
    layerStart: index,
    layerEnd: index + 1,
  }));
  return {
    model: {
      id: "fixture-model",
      layers: [0, 1].map((index) => ({
        index,
        weightBytes: 32 * MIB,
        activationElements: 512,
        kvBytesPerToken: 128,
        decodeMsAtUnit: 1,
        prefillMsPerTokenAtUnit: 0.1,
      })),
      embeddingBytes: MIB,
      lmHeadBytes: MIB,
      runtimeOverheadBytesPerStage: 4 * MIB,
      embeddingDecodeMsAtUnit: 0.2,
      lmHeadDecodeMsAtUnit: 0.2,
      embeddingPrefillMsPerTokenAtUnit: 0.05,
      lmHeadPrefillMsPerTokenAtUnit: 0.05,
    },
    modelRevision: "sha256:fixture-model-r1",
    tokenizerId: "fixture-tokenizer-r1",
    topology: {
      nodes,
      links: [
        networkLink("node-a", "node-b"),
        networkLink("node-b", "node-a"),
      ],
    },
    workload: {
      promptTokens: 16,
      outputTokens: 8,
      contextTokens: 32,
      concurrentSequences: 2,
      maxStages: 2,
      maxQualityLoss: 0,
      minRouteAvailability: 0.8,
      batchWindowMs: 1,
      p95: true,
    },
    phasePlans: {
      prefill: {
        algorithm: "campaign-fixture",
        codec: "fp16",
        microBatchSize: 2,
        prefillChunkTokens: 8,
        stages,
      },
      decode: {
        algorithm: "campaign-fixture",
        codec: "fp16",
        microBatchSize: 2,
        prefillChunkTokens: 8,
        stages,
      },
    },
  };
}

function networkLink(from: string, to: string) {
  return {
    from,
    to,
    oneWayLatencyMs: 1,
    jitterP95Ms: 0.1,
    bandwidthMbps: 1_000,
    lossRate: 0,
    availability: 0.999,
  };
}

function inertAgent(id: string): LaunchAgent {
  return {
    id,
    async start() {
      throw new Error("injected supervisor must own process launch");
    },
  };
}

function agentHealth(binding: PhysicalGpuCampaignAgentBinding, processes: number) {
  return {
    schema: "gdlp-launch-agent-health/3",
    agentId: binding.agent.id,
    nodeId: binding.nodeId,
    buildIdentity: {
      schema: "mycellios-native-build-provenance/1",
      version: "0.2.38",
      sourceId: `sha256:${"1".repeat(64)}`,
    },
    activeProcesses: processes,
    retainedTombstones: 0,
  };
}

function fakeSupervisor(
  launch: PythonPipelineLaunchDescription,
  log: string[],
): PhysicalGpuCampaignSupervisor {
  let current = supervisorSnapshot(launch, "idle", "pending");
  return {
    async start() {
      log.push("supervisor:start");
      current = supervisorSnapshot(launch, "running", "ready");
      return current;
    },
    async stop() {
      log.push("supervisor:stop");
      current = supervisorSnapshot(launch, "stopped", "stopped");
      return current;
    },
    snapshot() {
      return structuredClone(current);
    },
  };
}

function supervisorSnapshot(
  launch: PythonPipelineLaunchDescription,
  state: LaunchSupervisorSnapshot["state"],
  processState: LaunchSupervisorSnapshot["processes"][number]["state"],
): LaunchSupervisorSnapshot {
  return {
    launchId: launch.launchId,
    pipelineId: launch.pipelineId,
    state,
    failure: null,
    processes: launch.launchOrder.map((process) => ({
      processId: process.processId,
      nodeId: process.anchor.memberId,
      kind: process.kind,
      stageIndex: process.stageIndex,
      state: processState,
    })),
    telemetry: [],
    telemetryDropped: 0,
  };
}

function apiFetch(
  fixture: CampaignFixture,
  log: string[],
  mutateSse?: (body: Record<string, unknown>, response: string) => string,
): NonNullable<PhysicalGpuCampaignDependencies["fetch"]> {
  const canaryByText = new Map(
    fixture.canaries.map((canary) => [canary.messages[0]!.content, canary]),
  );
  return async (input, init) => {
    const url = String(input);
    if (url.endsWith("/health")) {
      log.push("api:health");
      return jsonResponse({
        status: "ready",
        error: null,
        model: fixture.launch.configuration.publicModelName,
        artifact_identity: fixture.launch.runtimeModel.artifactIdentity,
        canonical_model_source: fixture.launch.runtimeModel.canonicalSource,
        canonical_model_revision: fixture.launch.runtimeModel.canonicalRevision,
        pipeline_snapshot_identity: fixture.launch.runtimeModel.snapshotIdentity,
        stages: 2,
        boundaries: [0, 1, 2],
        codec: fixture.launch.route.codec,
      });
    }
    const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
    const messages = body.messages as Array<{ content: string }>;
    const canary = canaryByText.get(messages[0]!.content)!;
    const final = completionDocument(fixture, canary, body.stream === true);
    if (body.stream !== true) {
      log.push(`canary:${canary.id}`);
      return jsonResponse(final);
    }
    log.push(`sample:${canary.id}`);
    let response = [
      `data: ${JSON.stringify({
        id: "chatcmpl-test",
        object: "chat.completion.chunk",
        model: fixture.launch.configuration.publicModelName,
        choices: [{ index: 0, delta: { content: "ok" }, finish_reason: null }],
      })}`,
      "",
      `data: ${JSON.stringify(final)}`,
      "",
      "data: [DONE]",
      "",
      "",
    ].join("\n");
    response = mutateSse?.(body, response) ?? response;
    return new Response(response, { status: 200, headers: { "content-type": "text/event-stream" } });
  };
}

function completionDocument(
  fixture: CampaignFixture,
  canary: PhysicalGpuCampaignCanary,
  stream: boolean,
) {
  return {
    id: "chatcmpl-test",
    object: stream ? "chat.completion.chunk" : "chat.completion",
    model: fixture.launch.configuration.publicModelName,
    choices: [
      stream
        ? { index: 0, delta: {}, finish_reason: canary.expectedFinishReason }
        : {
            index: 0,
            message: { role: "assistant", content: "ok" },
            finish_reason: canary.expectedFinishReason,
          },
    ],
    usage: { prompt_tokens: 7, completion_tokens: 2, total_tokens: 9 },
    distribution_metrics: {
      ttft_ms: 2,
      tpot_ms: 3,
      pipeline_ms: 5,
      output_token_ids_sha256: canary.expectedOutputTokenIdsSha256,
      output_token_ids_hash_scheme: "gdlp-output-token-ids-v1",
    },
  };
}

function jsonResponse(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

class ImmediateRpcCampaignAgent implements LaunchAgent {
  constructor(readonly id: string) {}

  async start(
    _request: LaunchAgentStartRequest,
    signal: AbortSignal,
  ): Promise<LaunchProcessHandle> {
    if (signal.aborted) throw signal.reason;
    return new ImmediateRpcCampaignHandle();
  }
}

class ImmediateRpcCampaignHandle implements LaunchProcessHandle {
  readonly ready = Promise.resolve();
  readonly exited: Promise<LaunchProcessExit>;
  private readonly resolveExit: (exit: LaunchProcessExit) => void;
  private stopped = false;

  constructor() {
    let resolveExit!: (exit: LaunchProcessExit) => void;
    this.exited = new Promise((resolve) => {
      resolveExit = resolve;
    });
    this.resolveExit = resolveExit;
  }

  async stop(): Promise<void> {
    if (this.stopped) return;
    this.stopped = true;
    this.resolveExit({ code: 0, signal: "SIGTERM" });
  }

  output(): LaunchCapturedOutput {
    return {
      stdout: "",
      stderr: "",
      stdoutTruncated: false,
      stderrTruncated: false,
    };
  }
}

function incrementalClock(): () => number {
  let value = 0;
  return () => ++value;
}
