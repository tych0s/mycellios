import { describe, expect, it } from "vitest";
import { sha256CanonicalEvidence } from "../src/core/json.js";
import {
  executePhysicalGpuConveyorAbCli,
  parsePhysicalGpuConveyorAbCliArguments,
} from "../src/distribution/physical-gpu-conveyor-ab-cli.js";
import {
  buildPhysicalGpuConveyorSchedule,
  runPhysicalGpuConveyorAb,
  validatePhysicalGpuConveyorLaunchPair,
  type PhysicalGpuConveyorAbDependencies,
  type PhysicalGpuConveyorArm,
} from "../src/distribution/physical-gpu-conveyor-ab.js";
import {
  compilePythonLaunchDescription,
  type PythonLaunchCompilerOptions,
  type PythonPipelineLaunchDescription,
} from "../src/distribution/python-launcher.js";
import {
  buildRuntimePipelineManifest,
  type RuntimePlanRequest,
  type RuntimeNodeProfile,
} from "../src/distribution/runtime-manifest.js";
import type {
  DistributedModelProfile,
  DistributionPlan,
} from "../src/distribution/types.js";
import type {
  LaunchAgent,
  LaunchSupervisorSnapshot,
} from "../src/distribution/launch-supervisor.js";
import type {
  PhysicalGpuCampaignInput,
  PhysicalGpuCampaignObservation,
} from "../src/distribution/physical-gpu-campaign.js";
import {
  OUTPUT_TOKEN_IDS_HASH_SCHEME,
  buildPhysicalTwoHostGpuGateReport,
  outputTokenIdsSha256,
} from "../src/distribution/physical-gpu-gate-report.js";

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const REFERENCE_TOKENS = Array.from({ length: 16 }, (_, index) => index + 101);
const OTHER_TOKENS = Array.from({ length: 16 }, (_, index) => index + 301);
const OUTPUT_HASH = outputTokenIdsSha256(REFERENCE_TOKENS);
const OTHER_OUTPUT_HASH = outputTokenIdsSha256(OTHER_TOKENS);
const SOURCE_ID = `sha256:${"1".repeat(64)}` as const;
const MEASURED_AT = Date.now();

describe("physical GPU conveyor paired campaign", () => {
  it("uses deterministic ABBA/BAAB rounds, serial cleanup, physical gates and a sealed report", async () => {
    expect(buildPhysicalGpuConveyorSchedule(2)).toEqual([
      "w1",
      "conveyor",
      "conveyor",
      "w1",
      "conveyor",
      "w1",
      "w1",
      "conveyor",
    ]);
    const pair = launchPair();
    const order: PhysicalGpuConveyorArm[] = [];
    let active = 0;
    const dependencies = successfulDependencies(order, {
      supervisor: (launch) => {
        const arm = launch.configuration.speculativeInflightWaves === undefined
          ? "w1"
          : "conveyor";
        let state: "idle" | "running" | "stopped" = "idle";
        return {
          async start() {
            expect(active).toBe(0);
            active += 1;
            state = "running";
            order.push(arm);
            return supervisorSnapshot(launch, "running");
          },
          async stop() {
            expect(state).toBe("running");
            state = "stopped";
            active -= 1;
            return supervisorSnapshot(launch, "stopped");
          },
          snapshot() {
            return supervisorSnapshot(
              launch,
              state === "running" ? "running" : "stopped",
            );
          },
        };
      },
    });
    const report = await runPhysicalGpuConveyorAb(
      abInput(pair.baseline, pair.conveyor, 2),
      dependencies,
    );

    expect(order).toEqual(buildPhysicalGpuConveyorSchedule(2));
    expect(active).toBe(0);
    expect(report.passed).toBe(true);
    expect(report.evidence).toBe("physical");
    expect(report.runs).toHaveLength(8);
    expect(report.runs.every((run) => run.physicalGate.passed)).toBe(true);
    expect(report.runs.every((run) => run.physicalGate.report !== null)).toBe(true);
    expect(report.summary.comparison.paired.pairCount).toBe(4);
    expect(report.summary.baseline.highWaterWaves).toBe(1);
    expect(report.summary.conveyor.highWaterWaves).toBe(3);
    expect(report.summary.conveyor.acceptanceRate).toBe(0.75);
    expect(report.summary.conveyor.resources.observedPowerWatts.count).toBe(0);
    expect(report.parity.length).toBeGreaterThan(0);
    expect(report.parity.every((check) => check.passed)).toBe(true);
    const { seal, ...body } = report;
    expect(seal.digest).toBe(sha256CanonicalEvidence(body));
  });

  it("rejects canonical launches that differ outside W/bytes and derived launch id", () => {
    const pair = launchPair();
    const different = compileLaunch({
      publicModelName: "different-public-model",
      speculativeInflightWaves: 3,
      speculativeInflightBytes: 64 * MIB,
    });
    expect(() =>
      validatePhysicalGpuConveyorLaunchPair(pair.baseline, different),
    ).toThrow("physical_gpu_conveyor_ab_launches_differ_outside_window_flags");
  });

  it("fails closed when a sample hash diverges between arms", async () => {
    const pair = launchPair();
    const dependencies = successfulDependencies([], {
      hashForArm: (arm) => arm === "w1" ? OUTPUT_HASH : OTHER_OUTPUT_HASH,
    });
    const report = await runPhysicalGpuConveyorAb(
      abInput(pair.baseline, pair.conveyor),
      dependencies,
    );
    expect(report.passed).toBe(false);
    expect(report.failures.some((failure) =>
      failure.startsWith("output_hash_mismatch:sample:"),
    )).toBe(true);
    expect(report.parity.some((check) => !check.passed)).toBe(true);
  });

  it("stops after a cleanup failure and seals the partial failure", async () => {
    const pair = launchPair();
    let campaignIndex = 0;
    const dependencies = successfulDependencies([], {
      cleanupForRun: () => campaignIndex++ !== 1,
    });
    const report = await runPhysicalGpuConveyorAb(
      abInput(pair.baseline, pair.conveyor),
      dependencies,
    );
    expect(report.passed).toBe(false);
    expect(report.runs).toHaveLength(2);
    expect(report.failures).toContain("run_1:cleanup_did_not_pass");
    expect(report.failures).toContain("schedule_incomplete");
    const { seal, ...body } = report;
    expect(seal.digest).toBe(sha256CanonicalEvidence(body));
  });

  it("fails closed when the conveyor never has more than one wave or leaves credits", async () => {
    const pair = launchPair();
    const dependencies = successfulDependencies([], {
      healthOverride: (arm, phase) => {
        const health = healthBody(arm, phase);
        if (arm === "conveyor" && phase === "after") {
          health.speculative_window.high_water_waves = 1;
          health.speculative_window.current_bytes = 128;
        }
        return health;
      },
    });
    const report = await runPhysicalGpuConveyorAb(
      abInput(pair.baseline, pair.conveyor),
      dependencies,
    );
    expect(report.passed).toBe(false);
    expect(report.failures).toContain("run_1:speculative_window_not_drained");
    expect(report.failures).toContain(
      "run_1:conveyor_high_water_did_not_exceed_one",
    );
  });

  it("fails closed when observed waves or bytes exceed the sealed arm caps", async () => {
    const pair = launchPair();
    const dependencies = successfulDependencies([], {
      healthOverride: (arm, phase) => {
        const health = healthBody(arm, phase);
        if (arm === "conveyor" && phase === "after") {
          health.speculative_window.high_water_waves = 4;
          health.speculative_window.max_request_waves = 4;
          health.speculative_window.high_water_reserved_bytes = 64 * MIB + 1;
        }
        return health;
      },
    });
    const report = await runPhysicalGpuConveyorAb(
      abInput(pair.baseline, pair.conveyor),
      dependencies,
    );
    expect(report.passed).toBe(false);
    expect(report.failures).toContain(
      "run_1:speculative_window_wave_cap_exceeded",
    );
    expect(report.failures).toContain(
      "run_1:speculative_window_byte_cap_exceeded",
    );
  });

  it("rejects reused counters and impossible acceptance telemetry", async () => {
    const pair = launchPair();
    const reused = await runPhysicalGpuConveyorAb(
      abInput(pair.baseline, pair.conveyor),
      successfulDependencies([], {
        healthOverride: (arm, phase) => {
          const health = healthBody(arm, phase);
          if (phase === "before") {
            health.speculative_window.high_water_waves = 1;
          }
          return health;
        },
      }),
    );
    expect(reused.passed).toBe(false);
    expect(reused.failures).toContain(
      "run_0:speculative_window_before_is_not_zero",
    );

    const impossible = await runPhysicalGpuConveyorAb(
      abInput(pair.baseline, pair.conveyor),
      successfulDependencies([], {
        healthOverride: (arm, phase) => {
          const health = healthBody(arm, phase);
          if (phase === "after") {
            health.speculation.proposed_tokens = 10;
            health.speculation.accepted_tokens = 11;
            health.speculation.acceptance_rate = 1.1;
          }
          return health;
        },
      }),
    );
    expect(impossible.passed).toBe(false);
    expect(impossible.failures.some((failure) =>
      failure.includes("physical_gpu_conveyor_ab_speculation_accounting_is_invalid"),
    )).toBe(true);
  });

  it("rejects a physical gate whose embedded report does not match its seal", async () => {
    const pair = launchPair();
    const dependencies = successfulDependencies([], {
      physicalGateOverride: async (_arm, launch, campaign) => {
        const valid = physicalGateResult(launch, campaign);
        return {
          ...valid,
          report: { ...valid.report!, unexpected: true },
        };
      },
    });
    const report = await runPhysicalGpuConveyorAb(
      abInput(pair.baseline, pair.conveyor),
      dependencies,
    );
    expect(report.passed).toBe(false);
    expect(report.failures.some((failure) =>
      failure.includes("physical_gpu_gate_"),
    )).toBe(true);
  });

  it("rejects a valid physical report from a different launch and campaign", async () => {
    const pair = launchPair();
    const foreignCampaign = successfulCampaign(
      pair.conveyor,
      OUTPUT_HASH,
      true,
    );
    foreignCampaign.lifecycle.supervisorStarted = supervisorSnapshot(
      pair.conveyor,
      "running",
    );
    foreignCampaign.lifecycle.supervisorBeforeStop = supervisorSnapshot(
      pair.conveyor,
      "running",
    );
    foreignCampaign.lifecycle.supervisorStopped = supervisorSnapshot(
      pair.conveyor,
      "stopped",
    );
    const foreign = physicalGateResult(pair.conveyor, foreignCampaign);
    const report = await runPhysicalGpuConveyorAb(
      abInput(pair.baseline, pair.conveyor),
      successfulDependencies([], {
        physicalGateOverride: async () => structuredClone(foreign),
      }),
    );

    expect(report.passed).toBe(false);
    expect(report.failures.some((failure) =>
      failure.includes("physical_gpu_conveyor_ab_physical_gate_launch_mismatch"),
    )).toBe(true);
    expect(report.evidence).toBe("unverified");
  });
});

describe("physical GPU conveyor CLI", () => {
  it("accepts two launches plus common config/output and never overwrites", async () => {
    expect(parsePhysicalGpuConveyorAbCliArguments([
      "--baseline-launch",
      "w1.json",
      "--conveyor-launch",
      "w4.json",
      "--config",
      "physical.json",
      "--output",
      "ab.json",
      "--rounds",
      "2",
    ])).toEqual({
      baselineLaunchPath: "w1.json",
      conveyorLaunchPath: "w4.json",
      configPath: "physical.json",
      outputPath: "ab.json",
      rounds: 2,
    });
    let read = false;
    const errors: string[] = [];
    const code = await executePhysicalGpuConveyorAbCli(
      [
        "--baseline-launch",
        "w1.json",
        "--conveyor-launch",
        "w4.json",
        "--config",
        "physical.json",
        "--output",
        "existing.json",
      ],
      {
        cwd: "C:\\campaign",
        pathExists: async () => true,
        readText: async () => {
          read = true;
          return "";
        },
        writeStderr: (message) => errors.push(message),
      },
    );
    expect(code).toBe(1);
    expect(read).toBe(false);
    expect(errors.join("")).toContain("output_already_exists");
  });
});

function successfulDependencies(
  order: PhysicalGpuConveyorArm[],
  options: {
    supervisor?: NonNullable<PhysicalGpuConveyorAbDependencies["supervisor"]>;
    hashForArm?: (arm: PhysicalGpuConveyorArm) => string;
    cleanupForRun?: () => boolean;
    healthOverride?: (
      arm: PhysicalGpuConveyorArm,
      phase: "before" | "after",
    ) => ReturnType<typeof healthBody>;
    physicalGateOverride?: PhysicalGpuConveyorAbDependencies["physicalGate"];
  } = {},
): PhysicalGpuConveyorAbDependencies {
  const supervisor =
    options.supervisor
    ?? ((launch) => {
      const arm = launch.configuration.speculativeInflightWaves === undefined
        ? "w1"
        : "conveyor";
      let running = false;
      return {
        async start() {
          running = true;
          order.push(arm);
          return supervisorSnapshot(launch, "running");
        },
        async stop() {
          running = false;
          return supervisorSnapshot(launch, "stopped");
        },
        snapshot() {
          return supervisorSnapshot(launch, running ? "running" : "stopped");
        },
      };
    });
  return {
    supervisor,
    apiHealth: async (_url, phase, arm) =>
      options.healthOverride?.(arm, phase) ?? healthBody(arm, phase),
    runCampaign: async (inputValue, dependencies = {}) => {
      const input = inputValue as PhysicalGpuCampaignInput;
      const factory = dependencies.supervisor;
      if (factory === undefined) throw new Error("test_supervisor_missing");
      const current = factory(input.launch, {
        resolveAgent: () => undefined,
      });
      const started = await current.start();
      const arm = input.launch.configuration.speculativeInflightWaves === undefined
        ? "w1"
        : "conveyor";
      const cleanupPassed = options.cleanupForRun?.() ?? true;
      const campaign = successfulCampaign(
        input.launch,
        options.hashForArm?.(arm) ?? OUTPUT_HASH,
        cleanupPassed,
      );
      campaign.lifecycle.supervisorStarted = started;
      campaign.lifecycle.supervisorBeforeStop = current.snapshot();
      campaign.lifecycle.supervisorStopped = await current.stop("test_complete");
      return campaign;
    },
    physicalGate:
      options.physicalGateOverride
      ?? (async (_arm, launch, campaign) =>
        physicalGateResult(launch, campaign)),
    now: () => new Date("2026-07-26T12:00:00.000Z"),
  };
}

function abInput(
  baselineLaunch: PythonPipelineLaunchDescription,
  conveyorLaunch: PythonPipelineLaunchDescription,
  rounds = 1,
) {
  const nodeIds = [
    ...new Set(baselineLaunch.launchOrder.map((process) => process.anchor.memberId)),
  ];
  const agents = nodeIds.map((nodeId) => ({
    nodeId,
    agent: fakeAgent(nodeId),
  }));
  return {
    baselineLaunch,
    conveyorLaunch,
    campaign: {
      agents,
      apiBaseUrl: "http://10.20.0.11:8081",
      canaries: ["canary-a", "canary-b", "canary-c"].map((id) => ({
        id,
        messages: [{ role: "user" as const, content: id }],
        maxTokens: 16,
        expectedOutputTokenIdsSha256: OUTPUT_HASH,
        expectedCompletionTokens: 16,
        expectedFinishReason: "length" as const,
      })),
      warmups: 1,
      iterations: 5,
      concurrencies: [1],
    },
    rounds,
  };
}

function fakeAgent(nodeId: string): LaunchAgent {
  return {
    id: `local-process:${nodeId}`,
    async start() {
      return {
        ready: Promise.resolve(),
        exited: new Promise(() => {}),
        async stop() {},
      };
    },
  };
}

function successfulCampaign(
  launch: PythonPipelineLaunchDescription,
  hash: string,
  cleanupPassed: boolean,
): PhysicalGpuCampaignObservation {
  const processSnapshots = launch.launchOrder.map((process) => ({
    processId: process.processId,
    nodeId: process.anchor.memberId,
    kind: process.kind,
    stageIndex: process.stageIndex,
    state: "stopped" as const,
  }));
  const sourceHealth = [
    ...new Set(launch.launchOrder.map((process) => process.anchor.memberId)),
  ].map((nodeId) => ({
    phase: "before" as const,
    expectedAgentId: `local-process:${nodeId}`,
    expectedNodeId: nodeId,
    health: {
      schema: "gdlp-launch-agent-health/3" as const,
      agentId: `local-process:${nodeId}`,
      nodeId,
      buildIdentity: {
        schema: "mycellios-native-build-provenance/1" as const,
        version: "0.2.40",
        sourceId: SOURCE_ID,
      },
      activeProcesses: 0,
      retainedTombstones: 0,
    },
    passed: true,
    error: null,
  }));
  const metric = {
    count: 1,
    mean: 10,
    p50: 10,
    p95: 10,
    min: 10,
    max: 10,
  };
  const canaries = (phase: "pre" | "post") =>
    ["canary-a", "canary-b", "canary-c"].map((canaryId) => ({
      phase,
      canaryId,
      passed: true,
      clientResponseMs: 20,
      evidence: {
        promptTokens: 8,
        completionTokens: 16,
        finishReason: "length" as const,
        outputTokenIdsSha256: hash,
        outputTokenIdsHashScheme: "gdlp-output-token-ids-v1" as const,
        serverTtftMs: 10,
        serverTpotMs: 2,
        serverPipelineMs: 20,
      },
      error: null,
    }));
  const sample = (phase: "warmup" | "measure", iteration: number) => ({
    sampleId: `${phase}-c1-i${iteration}-r0`,
    phase,
    canaryId: "canary-a",
    concurrency: 1,
    iteration,
    requestIndex: 0,
    passed: true,
    clientFirstContentMs: 10,
    clientResponseMs: 20,
    promptTokens: 8,
    completionTokens: 16,
    finishReason: "length" as const,
    outputTokenIdsSha256: hash,
    serverTtftMs: 10,
    serverTpotMs: 2,
    serverPipelineMs: 40,
    perUserOutputTokensPerSecondIncludingTtft: 800,
    error: null,
  });
  const samples = [
    sample("warmup", 0),
    ...Array.from({ length: 5 }, (_, iteration) =>
      sample("measure", iteration)),
  ];
  const root = launch.launchOrder.find((process) => process.kind === "root-engine");
  if (root?.kind !== "root-engine") throw new Error("test_root_missing");
  return {
    schema: "gdlp-physical-gpu-campaign-observation/1",
    launchId: launch.launchId,
    pipelineId: launch.pipelineId,
    apiBaseUrl: "http://10.20.0.11:8081",
    passed: cleanupPassed,
    lifecycle: {
      events: [],
      agentHealthBefore: sourceHealth,
      agentHealthAfter: sourceHealth.map((item) => ({ ...item, phase: "after" as const })),
      supervisorStarted: null,
      supervisorBeforeStop: null,
      supervisorStopped: {
        launchId: launch.launchId,
        pipelineId: launch.pipelineId,
        state: "stopped",
        failure: null,
        processes: processSnapshots,
        telemetry: [],
        telemetryDropped: 0,
      },
      cleanupAttempted: true,
      cleanupPassed,
    },
    apiHealth: {
      status: "ready",
      error: null,
      model: launch.configuration.publicModelName,
      artifactIdentity: launch.runtimeModel.artifactIdentity!,
      canonicalModelSource: launch.runtimeModel.canonicalSource!,
      canonicalModelRevision: launch.runtimeModel.canonicalRevision ?? null,
      pipelineSnapshotIdentity: launch.runtimeModel.snapshotIdentity!,
      stages: launch.route.logicalStageIds.length,
      boundaries: [...root.boundaries],
      codec: launch.route.codec,
    },
    canaries: { pre: canaries("pre"), post: canaries("post") },
    samples,
    batches: [],
    summary: {
      measuredRequests: samples.filter((item) => item.phase === "measure").length,
      actualCompletionTokens: 80,
      measuredBatchWallMs: 100,
      aggregateOutputTokensPerSecondIncludingTtft: 800,
      clientFirstContentMs: metric,
      clientResponseMs: metric,
      serverTtftMs: metric,
      serverTpotMs: metric,
      serverPipelineMs: metric,
      byConcurrency: [],
    },
    failures: cleanupPassed ? [] : [{ phase: "cleanup_stop", message: "test_failure" }],
  };
}

function physicalGateResult(
  launch: PythonPipelineLaunchDescription,
  campaign: PhysicalGpuCampaignObservation,
) {
  const stage = launch.launchOrder.find(
    (process) => process.kind === "remote-stage" && process.cell?.external !== undefined,
  );
  if (stage?.kind !== "remote-stage" || stage.cell?.external === undefined) {
    throw new Error("test_external_cell_missing");
  }
  const [rankZeroNode, rankOneNode] = stage.cell.rankMemberIds;
  const [rankZeroDevice, rankOneDevice] = stage.cell.rankDevices;
  if (!rankZeroNode || !rankOneNode || !rankZeroDevice || !rankOneDevice) {
    throw new Error("test_external_cell_rank_missing");
  }
  const started = campaign.lifecycle.supervisorStarted;
  const stopped = campaign.lifecycle.supervisorStopped;
  const health = campaign.apiHealth;
  if (started === null || stopped === null || health === null) {
    throw new Error("test_physical_campaign_is_incomplete");
  }
  const outputHash = campaign.samples[0]?.outputTokenIdsSha256;
  const outputTokens =
    outputHash === OUTPUT_HASH
      ? REFERENCE_TOKENS
      : outputHash === OTHER_OUTPUT_HASH
        ? OTHER_TOKENS
        : null;
  if (outputTokens === null) throw new Error("test_output_tokens_are_unknown");
  const buildIdentity = {
    schema: "mycellios-native-build-provenance/1" as const,
    version: "0.2.40",
    sourceId: SOURCE_ID,
  };
  const hosts = [
    {
      hostId: "host-a",
      hostFingerprintSha256: `sha256:${"2".repeat(64)}`,
      agentId: `local-process:${rankZeroNode}`,
      agentEndpoint: "http://10.20.0.11:9750",
      rankNodeId: rankZeroNode,
      buildIdentity,
      gpu: {
        deviceFingerprintSha256: `sha256:${"a".repeat(64)}`,
        device: rankZeroDevice,
        vendor: "nvidia",
        model: "GPU A",
        physicalVramBytes: 4 * GIB,
        offeredVramBytes: 3 * GIB,
        computeApi: "cuda" as const,
        runtimeAvailable: true,
        collectiveAvailable: true,
      },
    },
    {
      hostId: "host-b",
      hostFingerprintSha256: `sha256:${"3".repeat(64)}`,
      agentId: `local-process:${rankOneNode}`,
      agentEndpoint: "http://10.20.0.12:9750",
      rankNodeId: rankOneNode,
      buildIdentity,
      gpu: {
        deviceFingerprintSha256: `sha256:${"b".repeat(64)}`,
        device: rankOneDevice,
        vendor: "amd",
        model: "GPU B",
        physicalVramBytes: 6 * GIB,
        offeredVramBytes: 3 * GIB,
        computeApi: "rocm" as const,
        runtimeAvailable: true,
        collectiveAvailable: true,
      },
    },
  ];
  const report = buildPhysicalTwoHostGpuGateReport({
    capturedAt: "2026-07-26T12:00:00.000Z",
    provenance: {
      level: "hardware-physical",
      source: "measurement",
      networkScope: "lan",
      loopback: false,
      emulated: false,
      attestation: "self-reported",
    },
    launch: {
      description: structuredClone(launch),
      canonicalSha256: sha256CanonicalEvidence(launch),
    },
    hosts,
    networkLinks: [
      {
        fromHostId: "host-a",
        toHostId: "host-b",
        rttMs: [1.1, 1.2, 1.15],
        goodputMbps: [900, 910, 905],
      },
      {
        fromHostId: "host-b",
        toHostId: "host-a",
        rttMs: [1.2, 1.25, 1.18],
        goodputMbps: [880, 890, 885],
      },
    ],
    lifecycle: {
      readyProcessIds: started.processes.map((process) => process.processId),
      stoppedProcessIds: stopped.processes.map((process) => process.processId),
      residualProcessIds: [],
      health: {
        status: health.status,
        model: health.model,
        stages: health.stages,
        boundaries: [...health.boundaries],
        codec: health.codec,
      },
    },
    reference: {
      mode: "monolithic-greedy",
      modelId: launch.modelIdentity.id,
      modelRevision: launch.modelIdentity.revision,
      tokenizerId: launch.modelIdentity.tokenizerId,
      promptTokenIdsSha256: sha256CanonicalEvidence([7, 8, 9]),
      outputTokenIds: [...outputTokens],
      outputTokenIdsHashScheme: OUTPUT_TOKEN_IDS_HASH_SCHEME,
      outputTokenIdsSha256: outputTokenIdsSha256(outputTokens),
    },
    rankWork: [
      {
        rank: 0,
        hostId: "host-a",
        nodeId: rankZeroNode,
        device: rankZeroDevice,
        forwardCalls: 96,
        collectiveCalls: 192,
        tokensProcessed: 96,
        bytesSent: 2 * MIB,
        bytesReceived: 2 * MIB,
        peakAllocatedBytes: GIB,
      },
      {
        rank: 1,
        hostId: "host-b",
        nodeId: rankOneNode,
        device: rankOneDevice,
        forwardCalls: 96,
        collectiveCalls: 192,
        tokensProcessed: 96,
        bytesSent: 2 * MIB,
        bytesReceived: 2 * MIB,
        peakAllocatedBytes: GIB,
      },
    ],
    samples: campaign.samples.map((sample) => {
      if (
        sample.promptTokens === null
        || sample.completionTokens === null
        || sample.outputTokenIdsSha256 === null
        || sample.clientFirstContentMs === null
        || sample.clientResponseMs === null
        || sample.serverTtftMs === null
        || sample.serverTpotMs === null
        || sample.serverPipelineMs === null
      ) {
        throw new Error("test_sample_is_incomplete");
      }
      return {
        sampleId: sample.sampleId,
        phase: sample.phase,
        concurrency: sample.concurrency,
        iteration: sample.iteration,
        promptTokens: sample.promptTokens,
        completionTokens: sample.completionTokens,
        outputTokenIdsHashScheme: OUTPUT_TOKEN_IDS_HASH_SCHEME,
        outputTokenIdsSha256: sample.outputTokenIdsSha256,
        ttftMs: sample.clientFirstContentMs,
        tpotMs: sample.serverTpotMs,
        responseMs: sample.clientResponseMs,
        pipelineMs: sample.serverPipelineMs,
      };
    }),
  });
  return {
    passed: report.gate.passed,
    reportSealSha256: report.seal.digest,
    report,
  };
}

function supervisorSnapshot(
  launch: PythonPipelineLaunchDescription,
  state: "running" | "stopped",
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
      state: state === "running" ? "ready" : "stopped",
    })),
    telemetry: [],
    telemetryDropped: 0,
  };
}

function healthBody(
  arm: PhysicalGpuConveyorArm,
  phase: "before" | "after",
) {
  const conveyor = arm === "conveyor";
  return {
    status: "ready",
    speculative_window: {
      configured: conveyor,
      configured_waves_per_request: conveyor ? 3 : 1,
      configured_bytes_per_request: conveyor ? 64 * MIB : 0,
      current_waves: 0,
      current_bytes: 0,
      current_reserved_bytes: 0,
      high_water_waves: phase === "after" ? (conveyor ? 3 : 1) : 0,
      high_water_bytes: phase === "after" && conveyor ? 1_024 : 0,
      high_water_reserved_bytes: phase === "after" && conveyor ? 2_048 : 0,
      max_request_waves: phase === "after" ? (conveyor ? 3 : 1) : 0,
      max_request_bytes: phase === "after" && conveyor ? 1_024 : 0,
      max_request_reserved_bytes: phase === "after" && conveyor ? 2_048 : 0,
      dispatched_waves: phase === "after" ? 10 : 0,
      completed_waves: phase === "after" ? 10 : 0,
      committed_waves: phase === "after" ? 8 : 0,
      condemned_waves: phase === "after" ? 2 : 0,
      drained_waves: phase === "after" ? 2 : 0,
      rejection_collapses: phase === "after" ? 1 : 0,
      rejected_proposed_tokens: phase === "after" ? 2 : 0,
      condemned_proposed_tokens: phase === "after" ? 1 : 0,
      discarded_proposed_tokens: phase === "after" ? 1 : 0,
      rejected_wave_bytes: phase === "after" ? 64 : 0,
      tombstone_bytes: phase === "after" ? 32 : 0,
      discarded_bytes: phase === "after" ? 16 : 0,
    },
    speculation: {
      configured: true,
      proposed_tokens: phase === "after" ? 100 : 0,
      accepted_tokens: phase === "after" ? 75 : 0,
      acceptance_rate: phase === "after" ? 0.75 : null,
      verification_bytes: phase === "after" ? 4_096 : 0,
    },
  };
}

function launchPair() {
  return {
    baseline: compileLaunch(),
    conveyor: compileLaunch({
      speculativeInflightWaves: 3,
      speculativeInflightBytes: 64 * MIB,
    }),
  };
}

function compileLaunch(
  overrides: Partial<PythonLaunchCompilerOptions> = {},
): PythonPipelineLaunchDescription {
  return compilePythonLaunchDescription(
    buildRuntimePipelineManifest(runtimeRequest()),
    {
      apiEndpoint: { host: "0.0.0.0", port: 8_081 },
      returnEndpoint: { host: "10.20.0.11", port: 30_000 },
      returnBindHost: "0.0.0.0",
      runtimeModel: {
        source: "hf://example/conveyor-model",
        revision: `sha256:${"d".repeat(64)}`,
        snapshotIdentity: "123456789",
      },
      publicModelName: "conveyor-model",
      maxOutputTokens: 32,
      ...overrides,
    },
  );
}

function runtimeRequest(): RuntimePlanRequest {
  const measuredAt = MEASURED_AT;
  const root = runtimeNode("root-node", "10.20.0.11", 22_000);
  const rankZero = cellRuntimeNode("middle-node", "10.20.0.12", 22_100);
  const final = runtimeNode("final-node", "10.20.0.13", 22_200);
  const rankOne = cellRuntimeNode("cell-rank-1", "10.20.0.14", 22_300);
  const nodes = [root, rankZero, final, rankOne];
  const plan: DistributionPlan = {
    algorithm: "conveyor-ab-test",
    codec: "fp16",
    microBatchSize: 1,
    prefillChunkTokens: 8,
    stages: [
      { nodeId: root.id, layerStart: 0, layerEnd: 2 },
      { nodeId: rankZero.id, layerStart: 2, layerEnd: 4 },
      { nodeId: final.id, layerStart: 4, layerEnd: 6 },
    ],
  };
  return {
    model: modelProfile(),
    modelRevision: `sha256:${"d".repeat(64)}`,
    tokenizerId: "conveyor-tokenizer",
    topology: {
      nodes,
      links: nodes.flatMap((from) =>
        nodes
          .filter((to) => to.id !== from.id)
          .map((to) => ({
            from: from.id,
            to: to.id,
            oneWayLatencyMs: 0.2,
            jitterP95Ms: 0.1,
            bandwidthMbps: 1_000,
            lossRate: 0,
            availability: 0.999,
            evidence: {
              source: "runtime-probe" as const,
              measuredAt,
              validUntil: measuredAt + 60_000,
              successfulSamples: 8,
              failedSamples: 0,
            },
          })),
      ),
    },
    workload: {
      promptTokens: 12,
      outputTokens: 16,
      contextTokens: 64,
      concurrentSequences: 1,
      maxStages: 3,
      maxQualityLoss: 0,
      minRouteAvailability: 0.9,
      batchWindowMs: 0,
      p95: false,
    },
    phasePlans: { prefill: plan, decode: plan },
    tensorParallelCells: [
      {
        stageIndex: 1,
        memberNodeIds: [rankZero.id, rankOne.id],
        execution: {
          mode: "tensor-parallel-cell",
          engine: "python-torch",
          collectiveBackend: "nccl",
          computeDtype: "float16",
          fixture: {
            schema: "gdlp-llama-cell-stage/2",
            location: "member-local",
            path: "C:/gdlp/conveyor-cell-rank-0",
            layerCount: 2,
            manifestSha256: "1".repeat(64),
            shardSha256: ["2".repeat(64), "3".repeat(64)],
            rankMemory: [
              {
                fixedBytes: 8 * MIB,
                kvBytesPerToken: 1_024,
                requiredBytes: 8 * MIB + 1_024 * 64,
              },
              {
                fixedBytes: 8 * MIB,
                kvBytesPerToken: 1_024,
                requiredBytes: 8 * MIB + 1_024 * 64,
              },
            ],
          },
          worldSize: 2,
          rankMemberIds: [rankZero.id, rankOne.id],
          rankWeights: [1, 1],
          rankDevices: ["cuda:0", "cuda:0"],
          operationTimeoutSeconds: 60,
          external: {
            rankFixturePaths: [
              "C:/gdlp/conveyor-cell-rank-0",
              "C:/gdlp/conveyor-cell-rank-1",
            ],
            controlBindHost: "0.0.0.0",
            controlAdvertiseHost: "10.20.0.12",
            controlPort: 29_100,
            distributedAdvertiseHost: "10.20.0.12",
            distributedPort: 29_101,
            startupTimeoutSeconds: 60,
          },
        },
      },
    ],
    speculation: {
      mode: "adaptive",
      controller: "acceptance-adaptive",
      defaultStrategyId: "ngram",
      fallbackStrategyId: "autoregressive",
      acceptanceWindowTokens: 64,
      strategies: [
        {
          id: "ngram",
          kind: "ngram",
          maxDraftTokens: 2,
          minAcceptanceRate: 0.5,
          maxWasteRatio: 0.4,
          priority: 10,
        },
        {
          id: "autoregressive",
          kind: "autoregressive",
          maxDraftTokens: 1,
          minAcceptanceRate: 1,
          maxWasteRatio: 0,
          priority: 0,
        },
      ],
    },
  };
}

function modelProfile(): DistributedModelProfile {
  return {
    id: "conveyor-model",
    layers: Array.from({ length: 6 }, (_, index) => ({
      index,
      weightBytes: 8 * MIB,
      activationElements: 512,
      kvBytesPerToken: 128,
      decodeMsAtUnit: 1,
      prefillMsPerTokenAtUnit: 0.1,
    })),
    embeddingBytes: MIB,
    lmHeadBytes: MIB,
    runtimeOverheadBytesPerStage: MIB,
    embeddingDecodeMsAtUnit: 0.1,
    lmHeadDecodeMsAtUnit: 0.1,
    embeddingPrefillMsPerTokenAtUnit: 0.01,
    lmHeadPrefillMsPerTokenAtUnit: 0.01,
  };
}

function runtimeNode(
  id: string,
  host: string,
  port: number,
): RuntimeNodeProfile {
  return {
    id,
    region: "conveyor-test",
    memoryBytes: 128 * MIB,
    reserveBytes: 8 * MIB,
    decodeScale: 1,
    prefillScale: 1,
    codecScale: 1,
    batchGain: 0,
    maxBatchSpeedup: 1,
    powerWatts: 75,
    availability: 0.999,
    endpoint: { host, port },
    backend: {
      engine: "python-transformers",
      version: "1",
      modelFormats: ["safetensors"],
      executionModes: ["layer-range"],
    },
    capabilities: {
      deviceKinds: ["gpu"],
      computeApis: ["cuda"],
      weightDtypes: ["fp16"],
      activationCodecs: ["fp16"],
      features: ["layer-range", "kv-reuse"],
    },
  };
}

function cellRuntimeNode(
  id: string,
  host: string,
  port: number,
): RuntimeNodeProfile {
  return {
    ...runtimeNode(id, host, port),
    backend: {
      engine: "python-torch",
      version: "2",
      modelFormats: ["safetensors"],
      executionModes: ["tensor-parallel-cell"],
    },
    capabilities: {
      deviceKinds: ["gpu"],
      computeApis: ["nccl", "cuda"],
      weightDtypes: ["fp16"],
      activationCodecs: ["fp16"],
      features: ["rank-local-kv"],
    },
  };
}
