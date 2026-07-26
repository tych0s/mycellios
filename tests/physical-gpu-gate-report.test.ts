import { describe, expect, it } from "vitest";
import {
  canonicalEvidenceJson,
  sha256CanonicalEvidence,
} from "../src/core/json.js";
import {
  compilePythonLaunchDescription,
  type PythonPipelineLaunchDescription,
} from "../src/distribution/python-launcher.js";
import {
  buildPhysicalTwoHostGpuGateReport,
  OUTPUT_TOKEN_IDS_HASH_SCHEME,
  outputTokenIdsSha256,
  readPhysicalTwoHostGpuGateReport,
  requirePassingPhysicalTwoHostGpuGate,
  validatePhysicalTwoHostGpuGateReport,
  type PhysicalGateCheckId,
  type PhysicalTwoHostGpuGateEvidenceV1,
} from "../src/distribution/physical-gpu-gate-report.js";
import {
  buildRuntimePipelineManifest,
  type RuntimeNodeProfile,
  type RuntimePlanRequest,
} from "../src/distribution/runtime-manifest.js";
import type {
  DistributedModelProfile,
  DistributionPlan,
} from "../src/distribution/types.js";

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const TOKENS = Array.from({ length: 16 }, (_, index) => 100 + index);

describe("strict canonical evidence JSON", () => {
  it("uses ordinal keys and hashes equivalent JSON identically", () => {
    const left = { z: [1, true, null], a: { y: "dos", x: "uno" } };
    const right = { a: { x: "uno", y: "dos" }, z: [1, true, null] };
    expect(canonicalEvidenceJson(left)).toBe(
      '{"a":{"x":"uno","y":"dos"},"z":[1,true,null]}',
    );
    expect(sha256CanonicalEvidence(left)).toBe(sha256CanonicalEvidence(right));
  });

  it.each([
    ["NaN", { value: Number.NaN }],
    ["Infinity", { value: Number.POSITIVE_INFINITY }],
    ["undefined", { value: undefined }],
    ["Date", { value: new Date("2026-07-21T00:00:00.000Z") }],
  ])("rejects non-JSON evidence: %s", (_label, value) => {
    expect(() => canonicalEvidenceJson(value)).toThrow(/canonical_evidence_/);
  });

  it("rejects sparse arrays and cycles", () => {
    const sparse = new Array(2);
    sparse[1] = 1;
    expect(() => canonicalEvidenceJson(sparse)).toThrow(/sparse_array/);
    const cyclic: { self?: unknown } = {};
    cyclic.self = cyclic;
    expect(() => canonicalEvidenceJson(cyclic)).toThrow(/cycle/);
  });

  it("matches the runtime output-token digest wire format", () => {
    expect(outputTokenIdsSha256([10, 11, 99])).toBe(
      "sha256:75f9dc144ab4bc0c4a03e57a78e590d6d970c8624f16b98753e3c745673df4f6",
    );
    expect(() => outputTokenIdsSha256([-1])).toThrow("output_token_id_must_be_uint32");
    expect(() => outputTokenIdsSha256([2 ** 32])).toThrow(
      "output_token_id_must_be_uint32",
    );
  });
});

describe("gdlp-physical-two-host-gpu-gate/1", () => {
  it("builds a deterministic sealed passing report and reads it back", () => {
    const evidence = passingEvidence();
    const first = buildPhysicalTwoHostGpuGateReport(evidence);
    const second = buildPhysicalTwoHostGpuGateReport(structuredClone(evidence));

    expect(first).toEqual(second);
    expect(first.schema).toBe("gdlp-physical-two-host-gpu-gate/1");
    expect(first.gate.passed).toBe(true);
    expect(first.gate.checks.every((check) => check.passed)).toBe(true);
    expect(first.summary).toMatchObject({
      warmupSamples: 1,
      measuredSamples: 5,
      measuredCompletionTokens: 80,
    });
    expect(first.samples[0]).not.toHaveProperty("outputTokenIds");
    expect(first.samples[0]).toMatchObject({
      outputTokenIdsHashScheme: "gdlp-output-token-ids-v1",
      outputTokenIdsSha256: outputTokenIdsSha256(TOKENS),
    });
    expect(readPhysicalTwoHostGpuGateReport(JSON.stringify(first))).toEqual(first);
    expect(requirePassingPhysicalTwoHostGpuGate(first)).toEqual(first);
  });

  it("detects seal mutation, additional keys and a forged fixed policy", () => {
    const report = buildPhysicalTwoHostGpuGateReport(passingEvidence());
    const changed = structuredClone(report);
    changed.samples[1]!.ttftMs += 1;
    expect(() => validatePhysicalTwoHostGpuGateReport(changed)).toThrow(
      "physical_gpu_gate_seal_mismatch",
    );

    const extended = structuredClone(report) as unknown as Record<string, unknown>;
    extended.unsealedExtension = true;
    expect(() => validatePhysicalTwoHostGpuGateReport(extended)).toThrow(
      "physical_gpu_gate_report_has_invalid_keys",
    );

    const lowered = structuredClone(report);
    lowered.policy.minimumMeasuredSamples = 1;
    expect(() => validatePhysicalTwoHostGpuGateReport(lowered)).toThrow(
      "physical_gpu_gate_policy_mismatch",
    );
  });

  it("rejects non-finite and undefined evidence before sealing", () => {
    const nonFinite = passingEvidence();
    nonFinite.samples[0]!.ttftMs = Number.NaN;
    expect(() => buildPhysicalTwoHostGpuGateReport(nonFinite)).toThrow(
      /canonical_evidence_non_finite_number/,
    );

    const withUndefined = passingEvidence() as PhysicalTwoHostGpuGateEvidenceV1 & {
      hidden?: unknown;
    };
    withUndefined.hidden = undefined;
    expect(() => buildPhysicalTwoHostGpuGateReport(withUndefined)).toThrow(
      /canonical_evidence_unsupported_value/,
    );

    const withExtra = passingEvidence() as PhysicalTwoHostGpuGateEvidenceV1 & {
      hidden?: unknown;
    };
    withExtra.hidden = true;
    expect(() => buildPhysicalTwoHostGpuGateReport(withExtra)).toThrow(
      "physical_gpu_gate_evidence_has_invalid_keys",
    );
  });

  it.each([
    [
      "projection",
      "physical_measurement" as const,
      (evidence: PhysicalTwoHostGpuGateEvidenceV1) => {
        evidence.provenance.source = "projection";
      },
    ],
    [
      "emulation",
      "physical_measurement" as const,
      (evidence: PhysicalTwoHostGpuGateEvidenceV1) => {
        evidence.provenance.emulated = true;
      },
    ],
    [
      "declared loopback",
      "physical_measurement" as const,
      (evidence: PhysicalTwoHostGpuGateEvidenceV1) => {
        evidence.provenance.networkScope = "loopback";
        evidence.provenance.loopback = true;
      },
    ],
    [
      "loopback agent route",
      "non_loopback_route" as const,
      (evidence: PhysicalTwoHostGpuGateEvidenceV1) => {
        evidence.hosts[1]!.agentEndpoint = "http://127.0.0.1:9750";
      },
    ],
    [
      "duplicate host",
      "two_distinct_hosts" as const,
      (evidence: PhysicalTwoHostGpuGateEvidenceV1) => {
        evidence.hosts[1]!.hostFingerprintSha256 =
          evidence.hosts[0]!.hostFingerprintSha256;
      },
    ],
    [
      "different declared native build",
      "single_build_cohort" as const,
      (evidence: PhysicalTwoHostGpuGateEvidenceV1) => {
        evidence.hosts[1]!.buildIdentity.sourceId =
          `sha256:${"8".repeat(64)}`;
      },
    ],
    [
      "duplicate GPU",
      "two_distinct_gpus" as const,
      (evidence: PhysicalTwoHostGpuGateEvidenceV1) => {
        evidence.hosts[1]!.gpu.deviceFingerprintSha256 =
          evidence.hosts[0]!.gpu.deviceFingerprintSha256;
      },
    ],
    [
      "wrong launch hash",
      "sealed_launch" as const,
      (evidence: PhysicalTwoHostGpuGateEvidenceV1) => {
        evidence.launch.canonicalSha256 = `sha256:${"0".repeat(64)}`;
      },
    ],
    [
      "rank without work",
      "both_ranks_executed" as const,
      (evidence: PhysicalTwoHostGpuGateEvidenceV1) => {
        evidence.rankWork[1]!.collectiveCalls = 0;
      },
    ],
    [
      "token mismatch",
      "exact_token_parity" as const,
      (evidence: PhysicalTwoHostGpuGateEvidenceV1) => {
        const sample = evidence.samples.find((entry) => entry.phase === "measure")!;
        sample.outputTokenIdsSha256 = outputTokenIdsSha256([999, ...TOKENS.slice(1)]);
      },
    ],
    [
      "too few samples",
      "sufficient_samples" as const,
      (evidence: PhysicalTwoHostGpuGateEvidenceV1) => {
        evidence.samples = evidence.samples.slice(0, 3);
      },
    ],
    [
      "residual process",
      "healthy_clean_lifecycle" as const,
      (evidence: PhysicalTwoHostGpuGateEvidenceV1) => {
        evidence.lifecycle.residualProcessIds.push(
          evidence.lifecycle.stoppedProcessIds[0]!,
        );
      },
    ],
  ])("fails closed for %s", (_label, checkId, mutate) => {
    const evidence = passingEvidence();
    mutate(evidence);
    const report = buildPhysicalTwoHostGpuGateReport(evidence);
    expect(failedCheck(report, checkId)).toBe(true);
    expect(report.gate.passed).toBe(false);
    expect(() => requirePassingPhysicalTwoHostGpuGate(report)).toThrow(
      new RegExp(checkId),
    );
  });

  it("fails the physical cell check for a valid CPU/Gloo launch", () => {
    const evidence = passingEvidence("cpu");
    const report = buildPhysicalTwoHostGpuGateReport(evidence);
    expect(failedCheck(report, "two_rank_gpu_collective")).toBe(true);
    expect(report.gate.passed).toBe(false);
  });

  it("recomputes the decision instead of trusting stored gate fields", () => {
    const report = buildPhysicalTwoHostGpuGateReport(passingEvidence());
    const forged = structuredClone(report);
    forged.gate.checks[0]!.passed = false;
    forged.gate.passed = false;
    const { seal: _oldSeal, ...body } = forged;
    forged.seal.digest = sha256CanonicalEvidence(body);
    expect(() => validatePhysicalTwoHostGpuGateReport(forged)).toThrow(
      "physical_gpu_gate_decision_mismatch",
    );
  });
});

function passingEvidence(mode: "gpu" | "cpu" = "gpu"): PhysicalTwoHostGpuGateEvidenceV1 {
  const launch = launchDescription(mode);
  const cell = launch.launchOrder.find(
    (process) => process.kind === "remote-stage" && process.cell !== null,
  );
  if (!cell || cell.kind !== "remote-stage" || cell.cell === null) {
    throw new Error("test_cell_launch_missing");
  }
  const [rankZeroNode, rankOneNode] = cell.cell.rankMemberIds;
  const [rankZeroDevice, rankOneDevice] = cell.cell.rankDevices;
  if (!rankZeroNode || !rankOneNode || !rankZeroDevice || !rankOneDevice) {
    throw new Error("test_cell_rank_missing");
  }
  const processIds = launch.launchOrder.map((process) => process.processId);
  const tokenHash = outputTokenIdsSha256(TOKENS);
  const gpu = mode === "gpu";
  const hosts = [
    {
      hostId: "host-a",
      hostFingerprintSha256: `sha256:${"1".repeat(64)}`,
      agentId: "agent-a",
      agentEndpoint: "http://10.20.0.11:9750",
      rankNodeId: rankZeroNode,
      buildIdentity: {
        schema: "mycellios-native-build-provenance/1" as const,
        version: "0.2.19",
        sourceId: `sha256:${"9".repeat(64)}` as const,
      },
      gpu: {
        deviceFingerprintSha256: `sha256:${"a".repeat(64)}`,
        device: rankZeroDevice,
        vendor: gpu ? "nvidia" : "cpu-test",
        model: gpu ? "GPU A" : "CPU A",
        physicalVramBytes: gpu ? 4 * GIB : 0,
        offeredVramBytes: gpu ? 3 * GIB : 0,
        computeApi: gpu ? ("cuda" as const) : ("cpu" as const),
        runtimeAvailable: gpu,
        collectiveAvailable: gpu,
      },
    },
    {
      hostId: "host-b",
      hostFingerprintSha256: `sha256:${"2".repeat(64)}`,
      agentId: "agent-b",
      agentEndpoint: "http://10.20.0.12:9750",
      rankNodeId: rankOneNode,
      buildIdentity: {
        schema: "mycellios-native-build-provenance/1" as const,
        version: "0.2.19",
        sourceId: `sha256:${"9".repeat(64)}` as const,
      },
      gpu: {
        deviceFingerprintSha256: `sha256:${"b".repeat(64)}`,
        device: rankOneDevice,
        vendor: gpu ? "amd" : "cpu-test",
        model: gpu ? "GPU B" : "CPU B",
        physicalVramBytes: gpu ? 6 * GIB : 0,
        offeredVramBytes: gpu ? 3 * GIB : 0,
        computeApi: gpu ? ("rocm" as const) : ("cpu" as const),
        runtimeAvailable: gpu,
        collectiveAvailable: gpu,
      },
    },
  ];
  const samples = Array.from({ length: 6 }, (_, index) => ({
    sampleId: index === 0 ? "warmup-0" : `measure-${index - 1}`,
    phase: index === 0 ? ("warmup" as const) : ("measure" as const),
    concurrency: 1,
    iteration: index === 0 ? 0 : index - 1,
    promptTokens: 12,
    completionTokens: TOKENS.length,
    outputTokenIdsHashScheme: OUTPUT_TOKEN_IDS_HASH_SCHEME,
    outputTokenIdsSha256: tokenHash,
    ttftMs: 35 + index,
    tpotMs: 14 + index / 10,
    responseMs: 250 + index,
    pipelineMs: 230 + index,
  }));
  return {
    capturedAt: "2026-07-21T12:00:00.000Z",
    provenance: {
      level: "hardware-physical",
      source: "measurement",
      networkScope: "lan",
      loopback: false,
      emulated: false,
      attestation: "self-reported",
    },
    launch: {
      description: launch,
      canonicalSha256: sha256CanonicalEvidence(launch),
    },
    hosts,
    networkLinks: [
      {
        fromHostId: "host-a",
        toHostId: "host-b",
        rttMs: [1.1, 1.2, 1.15],
        goodputMbps: [925, 930, 920],
      },
      {
        fromHostId: "host-b",
        toHostId: "host-a",
        rttMs: [1.2, 1.25, 1.18],
        goodputMbps: [910, 915, 905],
      },
    ],
    lifecycle: {
      readyProcessIds: [...processIds],
      stoppedProcessIds: [...processIds],
      residualProcessIds: [],
      health: {
        status: "ready",
        model: launch.configuration.publicModelName,
        stages: launch.route.logicalStageIds.length,
        boundaries: [0, 2, 4, 6],
        codec: launch.route.codec,
      },
    },
    reference: {
      mode: "monolithic-greedy",
      modelId: launch.modelIdentity.id,
      modelRevision: launch.modelIdentity.revision,
      tokenizerId: launch.modelIdentity.tokenizerId,
      promptTokenIdsSha256: sha256CanonicalEvidence([7, 8, 9]),
      outputTokenIds: [...TOKENS],
      outputTokenIdsHashScheme: OUTPUT_TOKEN_IDS_HASH_SCHEME,
      outputTokenIdsSha256: tokenHash,
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
        peakAllocatedBytes: gpu ? GIB : 0,
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
        peakAllocatedBytes: gpu ? GIB : 0,
      },
    ],
    samples,
  };
}

function launchDescription(mode: "gpu" | "cpu"): PythonPipelineLaunchDescription {
  const model = modelProfile();
  const root = pipelineNode("root-node", "10.20.0.11", 22_000);
  const rankZero = cellNode("cell-rank-0", "10.20.0.11", 22_100, mode);
  const final = pipelineNode("final-node", "10.20.0.11", 22_200);
  const rankOne = cellNode("cell-rank-1", "10.20.0.12", 22_300, mode);
  const nodes = [root, rankZero, final, rankOne];
  const plan: DistributionPlan = {
    algorithm: "physical-gate-test",
    codec: "fp16",
    microBatchSize: 1,
    prefillChunkTokens: 8,
    stages: [
      { nodeId: root.id, layerStart: 0, layerEnd: 2 },
      { nodeId: rankZero.id, layerStart: 2, layerEnd: 4 },
      { nodeId: final.id, layerStart: 4, layerEnd: 6 },
    ],
  };
  const gpu = mode === "gpu";
  const request: RuntimePlanRequest = {
    model,
    modelRevision: `sha256:${"d".repeat(64)}`,
    tokenizerId: "physical-gate-tokenizer",
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
          collectiveBackend: gpu ? "nccl" : "gloo",
          computeDtype: gpu ? "float16" : "float32",
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
          rankDevices: gpu ? ["cuda:0", "cuda:0"] : ["cpu", "cpu"],
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
    runtimeModel: {
      source: "hf://example/physical-gate-model",
      revision: `sha256:${"d".repeat(64)}`,
      snapshotIdentity: "123456789",
    },
    publicModelName: "physical-gate-model",
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
    id: "physical-gate-model",
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
    region: "physical-gate-test",
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

function cellNode(
  id: string,
  host: string,
  port: number,
  mode: "gpu" | "cpu",
): RuntimeNodeProfile {
  const gpu = mode === "gpu";
  return {
    ...pipelineNode(id, host, port),
    backend: {
      engine: "python-torch",
      version: "2",
      modelFormats: ["safetensors"],
      executionModes: ["tensor-parallel-cell"],
    },
    capabilities: {
      deviceKinds: [gpu ? "gpu" : "cpu"],
      computeApis: gpu ? ["nccl", "cuda"] : ["gloo"],
      weightDtypes: [gpu ? "fp16" : "fp32"],
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

function failedCheck(
  report: ReturnType<typeof buildPhysicalTwoHostGpuGateReport>,
  id: PhysicalGateCheckId,
): boolean {
  return report.gate.checks.find((check) => check.id === id)?.passed === false;
}
