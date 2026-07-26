import { describe, expect, it } from "vitest";
import { importPhysicalCampaign } from "../src/benchlab/physical-import.js";
import type { RunIdentity } from "../src/benchlab/history.js";
import {
  OUTPUT_TOKEN_HASH_SCHEME,
  PHYSICAL_GPU_CAMPAIGN_SCHEMA,
  type PhysicalGpuCampaignObservation,
  type PhysicalGpuCampaignRequestSample,
  type PhysicalGpuCampaignSummary,
} from "../src/distribution/physical-gpu-campaign.js";
import {
  buildPhysicalGpuCampaignGateReport,
  type PhysicalGpuCampaignReportInput,
} from "../src/distribution/physical-gpu-campaign-report.js";
import { outputTokenIdsSha256 } from "../src/distribution/physical-gpu-gate-report.js";
import {
  compilePythonLaunchDescription,
  type PythonPipelineLaunchDescription,
} from "../src/distribution/python-launcher.js";
import {
  buildRuntimePipelineManifest,
  type RuntimeNodeProfile,
  type RuntimePlanRequest,
} from "../src/distribution/runtime-manifest.js";
import type { PhysicalProbeV1 } from "../src/distribution/physical-probe.js";
import type { DistributedModelProfile, DistributionPlan } from "../src/distribution/types.js";

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const REFERENCE_TOKENS = Array.from({ length: 16 }, (_, index) => 200 + index);
const OTHER_TOKENS = Array.from({ length: 7 }, (_, index) => 900 + index);

describe("physical GPU campaign report adapter", () => {
  it("produces a passing sealed report from one explicitly selected canary", () => {
    const input = fixture();
    const report = buildPhysicalGpuCampaignGateReport(input);

    expect(report.gate.passed).toBe(true);
    expect(report.samples).toHaveLength(6);
    expect(report.samples.every((sample) => sample.completionTokens === 16)).toBe(true);
    expect(report.samples[0]).not.toHaveProperty("outputTokenIds");
    expect(report.samples[0]).toMatchObject({
      ttftMs: 40,
      tpotMs: 10,
      responseMs: 250,
      pipelineMs: 170,
      outputTokenIdsSha256: outputTokenIdsSha256(REFERENCE_TOKENS),
      outputTokenIdsHashScheme: "gdlp-output-token-ids-v1",
    });
    expect(report.reference.outputTokenIds).toEqual(REFERENCE_TOKENS);
    expect(report.reference.tokenizerId).toBe(input.reference.tokenizerId);
    // The campaign deliberately contains another canary with another digest;
    // explicit referenceCanaryId filtering keeps it out of this gate report.
    expect(report.samples.some((sample) => sample.sampleId === "other-measure")).toBe(false);
  });

  it("rejects a campaign that did not pass or whose samples were truncated", () => {
    const failed = fixture();
    failed.campaign.passed = false;
    expect(() => buildPhysicalGpuCampaignGateReport(failed)).toThrow(
      "physical_gpu_campaign_report_campaign_did_not_pass",
    );

    const truncated = fixture();
    truncated.samplesTruncated = true;
    expect(() => buildPhysicalGpuCampaignGateReport(truncated)).toThrow(
      "physical_gpu_campaign_report_samples_are_truncated",
    );
  });

  it.each([
    [
      "nonce mismatch",
      "physical_gpu_campaign_report_probe_is_invalid",
      (input: PhysicalGpuCampaignReportInput) => {
        input.hosts[1]!.probe.nonce = "probe-host-b-wrong";
      },
    ],
    [
      "duplicate nonce",
      "physical_gpu_campaign_report_probe_nonces_are_not_unique",
      (input: PhysicalGpuCampaignReportInput) => {
        input.hosts[1]!.expectedProbeNonce = input.hosts[0]!.expectedProbeNonce;
        input.hosts[1]!.probe.nonce = input.hosts[0]!.expectedProbeNonce;
      },
    ],
    [
      "duplicate host",
      "physical_gpu_campaign_report_probe_hosts_are_not_unique",
      (input: PhysicalGpuCampaignReportInput) => {
        input.hosts[1]!.probe.host.fingerprintSha256 =
          input.hosts[0]!.probe.host.fingerprintSha256;
      },
    ],
    [
      "duplicate GPU",
      "physical_gpu_campaign_report_probe_gpus_are_not_unique",
      (input: PhysicalGpuCampaignReportInput) => {
        input.hosts[1]!.probe.devices[0]!.fingerprintSha256 =
          input.hosts[0]!.probe.devices[0]!.fingerprintSha256;
      },
    ],
  ])("rejects inconsistent physical probes: %s", (_label, message, mutate) => {
    const input = fixture();
    mutate(input);
    expect(() => buildPhysicalGpuCampaignGateReport(input)).toThrow(message);
  });

  it("rejects a reference whose canary hash is not the monolithic token hash", () => {
    const input = fixture();
    input.reference.outputTokenIds[0] = 999;
    expect(() => buildPhysicalGpuCampaignGateReport(input)).toThrow(
      "physical_gpu_campaign_report_pre_reference_canary_mismatch",
    );
  });

  it.each([
    [
      "artifact identity",
      "physical_gpu_campaign_report_reference_artifact_identity_mismatch",
      (input: PhysicalGpuCampaignReportInput) => {
        input.reference.artifactIdentity = `sha256:${"0".repeat(64)}`;
      },
    ],
    [
      "canonical source",
      "physical_gpu_campaign_report_reference_canonical_source_mismatch",
      (input: PhysicalGpuCampaignReportInput) => {
        input.reference.canonicalSource = "hf://another/model";
      },
    ],
    [
      "canonical revision",
      "physical_gpu_campaign_report_reference_canonical_revision_mismatch",
      (input: PhysicalGpuCampaignReportInput) => {
        input.reference.canonicalRevision = "another-revision";
      },
    ],
    [
      "tokenizer",
      "physical_gpu_campaign_report_reference_tokenizer_mismatch",
      (input: PhysicalGpuCampaignReportInput) => {
        input.reference.tokenizerId = "another-tokenizer";
      },
    ],
  ])("rejects monolithic reference identity mismatch: %s", (_label, message, mutate) => {
    const input = fixture();
    mutate(input);
    expect(() => buildPhysicalGpuCampaignGateReport(input)).toThrow(message);
  });

  it("rejects a weak snapshot:uint64 launch against a strong canary artifact", () => {
    const input = fixture(16, launchDescription("weak"));
    const strongIdentity = `sha256:${"a".repeat(64)}`;
    input.reference.artifactIdentity = strongIdentity;
    input.reference.canonicalSource = `content-addressed://${strongIdentity}`;
    input.reference.canonicalRevision = strongIdentity;
    expect(input.launch.runtimeModel.artifactIdentity).toMatch(/^snapshot:uint64:/);
    expect(() => buildPhysicalGpuCampaignGateReport(input)).toThrow(
      "physical_gpu_campaign_report_reference_artifact_identity_mismatch",
    );
  });

  it("rejects invalid, insufficient and shorter-than-policy selected samples", () => {
    const wrongDigest = fixture();
    const selected = wrongDigest.campaign.samples.find(
      (sample) => sample.canaryId === "reference-canary" && sample.phase === "measure",
    )!;
    selected.outputTokenIdsSha256 = outputTokenIdsSha256(OTHER_TOKENS);
    expect(() => buildPhysicalGpuCampaignGateReport(wrongDigest)).toThrow(
      /physical_gpu_campaign_report_sample_is_invalid/,
    );

    const insufficient = fixture();
    let remainingMeasured = 4;
    insufficient.campaign.samples = insufficient.campaign.samples.filter((sample) => {
      if (sample.canaryId !== "reference-canary" || sample.phase !== "measure") return true;
      if (remainingMeasured === 0) return false;
      remainingMeasured -= 1;
      return true;
    });
    insufficient.campaign.summary = summaryFor(insufficient.campaign.samples);
    expect(() => buildPhysicalGpuCampaignGateReport(insufficient)).toThrow(
      "physical_gpu_campaign_report_requires_five_reference_measurements",
    );

    const short = fixture(15);
    expect(() => buildPhysicalGpuCampaignGateReport(short)).toThrow(
      "physical_gpu_campaign_report_reference_measurement_is_too_short",
    );
  });

  it("rejects a forged clean campaign with a residual or mismatched agent binding", () => {
    const residual = fixture();
    residual.campaign.lifecycle.agentHealthAfter[0]!.health!.activeProcesses = 1;
    expect(() => buildPhysicalGpuCampaignGateReport(residual)).toThrow(
      "physical_gpu_campaign_report_agent_health_mismatch",
    );

    const wrongAgent = fixture();
    wrongAgent.hosts[1]!.agentId = "another-agent";
    expect(() => buildPhysicalGpuCampaignGateReport(wrongAgent)).toThrow(
      "physical_gpu_campaign_report_host_agent_binding_mismatch",
    );
  });

  it("imports benchmark evidence only from the sealed passing gate", () => {
    const report = buildPhysicalGpuCampaignGateReport(fixture());
    const identity: RunIdentity = {
      runId: "physical-import-test",
      version: "0.2.38",
      label: "physical import",
      gitCommit: null,
      gitBranch: null,
      gitDirty: null,
      build: {
        release: "0.2.38",
        releaseSource: "override",
        revision: null,
        revisionSource: "unknown",
        sourceId: null,
        sourceIdSource: "unknown",
        participantSourceIds: [],
      },
    };
    const run = importPhysicalCampaign(identity, report);
    expect(run.suite).toBe("physical-import");
    expect(run.trigger).toBe("physical-import");
    expect(run.measurements[0]).toMatchObject({
      evidence: "physical",
      inventory: { connectedDevices: 2, selectedDevices: 2 },
      metrics: { exactnessRate: 1, requestSuccessRate: 1 },
    });
    expect(run.build.participantSourceIds).toEqual([
      `sha256:${"1".repeat(64)}`,
    ]);

    const forged = structuredClone(report);
    forged.samples[1]!.responseMs += 1;
    expect(() => importPhysicalCampaign(identity, forged)).toThrow(
      "physical_gpu_gate_seal_mismatch",
    );
  });
});

function fixture(
  referenceLength = 16,
  launch = launchDescription(),
): PhysicalGpuCampaignReportInput {
  const cell = launch.launchOrder.find(
    (process) => process.kind === "remote-stage" && process.cell !== null,
  );
  if (!cell || cell.kind !== "remote-stage" || cell.cell === null) {
    throw new Error("test_cell_missing");
  }
  const [rankZeroNode, rankOneNode] = cell.cell.rankMemberIds;
  if (!rankZeroNode || !rankOneNode) throw new Error("test_rank_missing");
  const tokens = REFERENCE_TOKENS.slice(0, referenceLength);
  const digest = outputTokenIdsSha256(tokens);
  const otherDigest = outputTokenIdsSha256(OTHER_TOKENS);
  const campaign = campaignObservation(launch, digest, tokens.length, otherDigest);
  const hostA = hostBinding(
    "host-a",
    rankZeroNode,
    "10.20.0.11",
    "probe-host-a-0001",
    "1",
    "a",
  );
  const hostB = hostBinding(
    "host-b",
    rankOneNode,
    "10.20.0.12",
    "probe-host-b-0002",
    "2",
    "b",
  );
  return {
    capturedAt: "2026-07-21T14:00:00.000Z",
    networkScope: "lan",
    samplesTruncated: false,
    campaign,
    launch,
    hosts: [hostA, hostB],
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
    reference: {
      referenceCanaryId: "reference-canary",
      artifactIdentity: launch.runtimeModel.artifactIdentity!,
      canonicalSource: launch.runtimeModel.canonicalSource!,
      canonicalRevision: launch.runtimeModel.canonicalRevision ?? null,
      tokenizerId: launch.modelIdentity.tokenizerId,
      promptTokenIdsSha256: `sha256:${"f".repeat(64)}`,
      outputTokenIds: tokens,
    },
    rankWork: [
      {
        rank: 0,
        hostId: "host-a",
        nodeId: rankZeroNode,
        device: "cuda:0",
        forwardCalls: 96,
        collectiveCalls: 192,
        tokensProcessed: 96,
        bytesSent: MIB,
        bytesReceived: MIB,
        peakAllocatedBytes: GIB,
      },
      {
        rank: 1,
        hostId: "host-b",
        nodeId: rankOneNode,
        device: "cuda:0",
        forwardCalls: 96,
        collectiveCalls: 192,
        tokensProcessed: 96,
        bytesSent: MIB,
        bytesReceived: MIB,
        peakAllocatedBytes: GIB,
      },
    ],
  };
}

function campaignObservation(
  launch: PythonPipelineLaunchDescription,
  referenceDigest: string,
  referenceCount: number,
  otherDigest: string,
): PhysicalGpuCampaignObservation {
  const expectedNodes = [...new Set(launch.launchOrder.map((process) => process.anchor.memberId))];
  const processSnapshots = launch.launchOrder.map((process) => ({
    processId: process.processId,
    nodeId: process.anchor.memberId,
    kind: process.kind,
    stageIndex: process.stageIndex,
  }));
  const supervisor = (state: "running" | "stopped", processState: "ready" | "stopped") => ({
    launchId: launch.launchId,
    pipelineId: launch.pipelineId,
    state,
    failure: null,
    processes: processSnapshots.map((process) => ({ ...process, state: processState })),
    telemetry: [],
    telemetryDropped: 0,
  });
  const agentHealth = (phase: "before" | "after") =>
    expectedNodes.map((nodeId) => ({
      phase,
      expectedAgentId: `agent-${nodeId}`,
      expectedNodeId: nodeId,
      health: {
        schema: "gdlp-launch-agent-health/3" as const,
        agentId: `agent-${nodeId}`,
        nodeId,
        buildIdentity: {
          schema: "mycellios-native-build-provenance/1" as const,
          version: "0.2.38",
          sourceId: `sha256:${"1".repeat(64)}` as const,
        },
        activeProcesses: 0,
        retainedTombstones: 0,
      },
      passed: true,
      error: null,
    }));
  const canary = (phase: "pre" | "post", id: string, digest: string, count: number) => ({
    phase,
    canaryId: id,
    passed: true,
    clientResponseMs: 200,
    evidence: {
      promptTokens: 12,
      completionTokens: count,
      finishReason: "length" as const,
      outputTokenIdsSha256: digest,
      outputTokenIdsHashScheme: OUTPUT_TOKEN_HASH_SCHEME,
      serverTtftMs: 20,
      serverTpotMs: 10,
      serverPipelineMs: 20 + Math.max(0, count - 1) * 10,
    },
    error: null,
  });
  const samples: PhysicalGpuCampaignRequestSample[] = [
    sample("ref-warmup", "warmup", "reference-canary", 0, referenceCount, referenceDigest),
    ...Array.from({ length: 5 }, (_, index) =>
      sample(
        `ref-measure-${index}`,
        "measure",
        "reference-canary",
        index,
        referenceCount,
        referenceDigest,
      ),
    ),
    sample("other-measure", "measure", "other-canary", 0, OTHER_TOKENS.length, otherDigest),
  ];
  return {
    schema: PHYSICAL_GPU_CAMPAIGN_SCHEMA,
    launchId: launch.launchId,
    pipelineId: launch.pipelineId,
    apiBaseUrl: "http://10.20.0.11:8081",
    passed: true,
    lifecycle: {
      events: [],
      agentHealthBefore: agentHealth("before"),
      agentHealthAfter: agentHealth("after"),
      supervisorStarted: supervisor("running", "ready"),
      supervisorBeforeStop: supervisor("running", "ready"),
      supervisorStopped: supervisor("stopped", "stopped"),
      cleanupAttempted: true,
      cleanupPassed: true,
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
      boundaries: [0, 2, 4, 6],
      codec: launch.route.codec,
    },
    canaries: {
      pre: [
        canary("pre", "reference-canary", referenceDigest, referenceCount),
        canary("pre", "other-canary", otherDigest, OTHER_TOKENS.length),
      ],
      post: [
        canary("post", "reference-canary", referenceDigest, referenceCount),
        canary("post", "other-canary", otherDigest, OTHER_TOKENS.length),
      ],
    },
    samples,
    batches: samples.map((entry) => ({
      batchId: `batch-${entry.sampleId}`,
      phase: entry.phase,
      concurrency: entry.concurrency,
      iteration: entry.iteration,
      passed: true,
      clientWallMs: entry.clientResponseMs!,
      actualCompletionTokens: entry.completionTokens!,
      aggregateOutputTokensPerSecondIncludingTtft:
        entry.completionTokens! / (entry.clientResponseMs! / 1_000),
    })),
    summary: summaryFor(samples),
    failures: [],
  };
}

function sample(
  sampleId: string,
  phase: "warmup" | "measure",
  canaryId: string,
  iteration: number,
  completionTokens: number,
  digest: string,
): PhysicalGpuCampaignRequestSample {
  const clientResponseMs = 250;
  const serverTtftMs = 20;
  const serverTpotMs = 10;
  return {
    sampleId,
    phase,
    canaryId,
    concurrency: 1,
    iteration,
    requestIndex: 0,
    passed: true,
    clientFirstContentMs: 40,
    clientResponseMs,
    promptTokens: 12,
    completionTokens,
    finishReason: "length",
    outputTokenIdsSha256: digest,
    serverTtftMs,
    serverTpotMs,
    serverPipelineMs: serverTtftMs + Math.max(0, completionTokens - 1) * serverTpotMs,
    perUserOutputTokensPerSecondIncludingTtft:
      completionTokens / (clientResponseMs / 1_000),
    error: null,
  };
}

function summaryFor(samples: PhysicalGpuCampaignRequestSample[]): PhysicalGpuCampaignSummary {
  const stats = { count: 0, mean: null, p50: null, p95: null, min: null, max: null };
  const measured = samples.filter((sample) => sample.phase === "measure");
  return {
    measuredRequests: measured.length,
    actualCompletionTokens: measured.reduce(
      (total, sample) => total + (sample.completionTokens ?? 0),
      0,
    ),
    measuredBatchWallMs: 0,
    aggregateOutputTokensPerSecondIncludingTtft: null,
    clientFirstContentMs: { ...stats },
    clientResponseMs: { ...stats },
    serverTtftMs: { ...stats },
    serverTpotMs: { ...stats },
    serverPipelineMs: { ...stats },
    byConcurrency: [],
  };
}

function hostBinding(
  hostId: string,
  rankNodeId: string,
  ip: string,
  nonce: string,
  hostDigit: string,
  gpuDigit: string,
) {
  return {
    hostId,
    agentId: `agent-${rankNodeId}`,
    agentEndpoint: `http://${ip}:9750`,
    rankNodeId,
    device: "cuda:0",
    offeredVramBytes: 3 * GIB,
    vendor: gpuDigit === "a" ? "nvidia" : "amd",
    expectedProbeNonce: nonce,
    probe: physicalProbe(nonce, hostDigit, gpuDigit),
  };
}

function physicalProbe(nonce: string, hostDigit: string, gpuDigit: string): PhysicalProbeV1 {
  return {
    schema: "gdlp-physical-probe/1",
    nonce,
    host: {
      fingerprintSha256: `sha256:${hostDigit.repeat(64)}`,
      fingerprintSource: "machine-id",
      platform: "linux",
      architecture: "x64",
      kernelRelease: "6.8",
      pythonVersion: "3.12",
    },
    runtime: {
      torchVersion: "2.8",
      cudaVersion: gpuDigit === "a" ? "12.8" : null,
      rocmVersion: gpuDigit === "a" ? null : "6.4",
      cudaApiAvailable: true,
      distributedAvailable: true,
      ncclAvailable: true,
      ncclVersion: "2.27",
    },
    devices: [
      {
        index: 0,
        name: `GPU ${gpuDigit}`,
        totalMemoryBytes: 4 * GIB,
        freeMemoryBytes: 3 * GIB,
        runtimeTotalMemoryBytes: 4 * GIB,
        capability: [8, 9],
        uuidSha256: `sha256:${gpuDigit.repeat(64)}`,
        fingerprintSha256: `sha256:${gpuDigit.repeat(64)}`,
      },
    ],
  };
}

function launchDescription(
  identityMode: "strong" | "weak" = "strong",
): PythonPipelineLaunchDescription {
  const root = pipelineNode("root-node", "10.20.0.11", 22_000);
  const rankZero = cellNode("cell-rank-0", "10.20.0.11", 22_100);
  const final = pipelineNode("final-node", "10.20.0.11", 22_200);
  const rankOne = cellNode("cell-rank-1", "10.20.0.12", 22_300);
  const nodes = [root, rankZero, final, rankOne];
  const plan: DistributionPlan = {
    algorithm: "campaign-report-test",
    codec: "fp16",
    microBatchSize: 1,
    prefillChunkTokens: 8,
    stages: [
      { nodeId: root.id, layerStart: 0, layerEnd: 2 },
      { nodeId: rankZero.id, layerStart: 2, layerEnd: 4 },
      { nodeId: final.id, layerStart: 4, layerEnd: 6 },
    ],
  };
  const request: RuntimePlanRequest = {
    model: modelProfile(),
    modelRevision: `sha256:${"d".repeat(64)}`,
    tokenizerId: "campaign-report-tokenizer",
    topology: { nodes, links: measuredCollectiveLinks(nodes) },
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
            path: "C:/gdlp/cell-rank-0",
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
            rankFixturePaths: ["C:/gdlp/cell-rank-0", "C:/gdlp/cell-rank-1"],
            controlBindHost: "0.0.0.0",
            controlAdvertiseHost: "10.20.0.11",
            controlPort: 29_100,
            distributedAdvertiseHost: "10.20.0.11",
            distributedPort: 29_101,
            startupTimeoutSeconds: 60,
          },
        },
      },
    ],
  };
  return compilePythonLaunchDescription(buildRuntimePipelineManifest(request), {
    apiEndpoint: { host: "0.0.0.0", port: 8_081 },
    returnEndpoint: { host: "10.20.0.11", port: 30_000 },
    returnBindHost: "0.0.0.0",
    runtimeModel:
      identityMode === "strong"
        ? {
            source: "example/campaign-report-model",
            revision: "d".repeat(40),
          }
        : {
            source: "hf://example/campaign-report-model",
            revision: `sha256:${"d".repeat(64)}`,
            snapshotIdentity: "123456789",
          },
    publicModelName: "campaign-report-model",
    pythonExecutable: "python",
    threadsPerStage: 1,
    connectTimeoutSeconds: 60,
    batchWindowMs: 0,
    maxPendingRequests: 8,
    maxOutputTokens: 32,
  });
}

function modelProfile(): DistributedModelProfile {
  return {
    id: "campaign-report-model",
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

function pipelineNode(id: string, host: string, port: number): RuntimeNodeProfile {
  return {
    id,
    region: "campaign-report-test",
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
      deviceKinds: ["cpu"],
      computeApis: ["torch-cpu"],
      weightDtypes: ["fp32"],
      activationCodecs: ["fp16"],
      features: ["layer-range", "kv-reuse"],
    },
  };
}

function cellNode(id: string, host: string, port: number): RuntimeNodeProfile {
  return {
    ...pipelineNode(id, host, port),
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

function measuredCollectiveLinks(nodes: RuntimeNodeProfile[]) {
  const measuredAt = Date.now();
  return nodes.flatMap((from) =>
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
  );
}
