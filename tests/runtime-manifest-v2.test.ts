import { describe, expect, it } from "vitest";
import { compilePythonLaunchDescription } from "../src/distribution/python-launcher.js";
import {
  buildRuntimePipelineManifest,
  evaluateRuntimeKvGate,
  readRuntimePipelineManifest,
  validateRuntimePipelineManifest,
  type RuntimeKvGateProbe,
  type RuntimePipelineManifestV1,
  type RuntimePipelineManifestV2,
  type RuntimePlanRequest,
  type RuntimeSpeculationPolicy,
} from "../src/distribution/runtime-manifest.js";
import type {
  DistributedModelProfile,
  DistributionPlan,
  DistributionWorkload,
} from "../src/distribution/types.js";

const MIB = 1024 * 1024;

function tinyModel(): DistributedModelProfile {
  return {
    id: "tiny-universal",
    layers: Array.from({ length: 4 }, (_, index) => ({
      index,
      weightBytes: 8 * MIB,
      activationElements: 256,
      kvBytesPerToken: 128,
      decodeMsAtUnit: 0.8,
      prefillMsPerTokenAtUnit: 0.12,
    })),
    embeddingBytes: 2 * MIB,
    lmHeadBytes: 2 * MIB,
    tiedEmbeddingAndHead: false,
    runtimeOverheadBytesPerStage: 4 * MIB,
    embeddingDecodeMsAtUnit: 0.2,
    lmHeadDecodeMsAtUnit: 0.2,
    embeddingPrefillMsPerTokenAtUnit: 0.04,
    lmHeadPrefillMsPerTokenAtUnit: 0.04,
  };
}

function tinyWorkload(): DistributionWorkload {
  return {
    promptTokens: 16,
    outputTokens: 8,
    contextTokens: 64,
    concurrentSequences: 2,
    maxStages: 2,
    maxQualityLoss: 1,
    minRouteAvailability: 0.8,
    batchWindowMs: 1,
    p95: true,
  };
}

function runtimeRequest(nodeCount = 2): RuntimePlanRequest {
  const nodes = Array.from({ length: nodeCount }, (_, index) => ({
    id: `node-${index}`,
    region: index === 0 ? "mad" : "bcn",
    memoryBytes: 256 * MIB,
    reserveBytes: 16 * MIB,
    decodeScale: 1 + index * 0.2,
    prefillScale: 1 + index * 0.15,
    codecScale: 1,
    batchGain: 0.2,
    maxBatchSpeedup: 1.5,
    powerWatts: 80,
    availability: 0.999,
    endpoint: { host: `10.0.0.${index + 1}`, port: 21_000 + index },
    backend: {
      engine: index === 0 ? "external GGUF runtime" : "mlx",
      version: "1.0",
      modelFormats: ["gguf"],
      executionModes: ["layer-range", "expert-range"],
    },
    capabilities: {
      deviceKinds: [index === 0 ? "nvidia-gpu" : "apple-gpu"],
      computeApis: [index === 0 ? "cuda" : "metal"],
      weightDtypes: ["fp16", "q4_k_m"],
      activationCodecs: ["fp16" as const, "int8" as const],
      features: ["layer-range", "kv-reuse", "kv-transfer"],
    },
  }));
  const links = [];
  for (const from of nodes) {
    for (const to of nodes) {
      if (from.id === to.id) continue;
      links.push({
        from: from.id,
        to: to.id,
        oneWayLatencyMs: 5,
        jitterP95Ms: 1,
        bandwidthMbps: 500,
        lossRate: 0,
        availability: 0.999,
      });
    }
  }
  return {
    model: tinyModel(),
    modelRevision: "sha256:model-r1",
    tokenizerId: "tiny-tokenizer-r1",
    topology: { nodes, links },
    workload: tinyWorkload(),
  };
}

function cellAwareRequest(
  options: {
    insufficient?: boolean;
    withCell?: boolean;
    gpu?: {
      computeDtype: "float32" | "float16" | "bfloat16";
      computeApi?: "cuda" | "rocm";
      rankDevices?: string[];
    };
  } = {},
): RuntimePlanRequest {
  const current = runtimeRequest(4);
  const computeApi = options.gpu?.computeApi ?? "cuda";
  const normalizedDtype =
    options.gpu?.computeDtype === "float16"
      ? "fp16"
      : options.gpu?.computeDtype === "bfloat16"
        ? "bf16"
        : "fp32";
  current.workload.maxStages = 3;
  const [root, anchor, peer, tail] = current.topology.nodes;
  root!.memoryBytes = 128 * MIB;
  root!.reserveBytes = 8 * MIB;
  tail!.memoryBytes = 128 * MIB;
  tail!.reserveBytes = 8 * MIB;
  for (const [index, member] of [anchor!, peer!].entries()) {
    member.memoryBytes = options.insufficient ? 10 * MIB : 14 * MIB;
    member.reserveBytes = 2 * MIB;
    member.endpoint = { host: "cell-host.internal", port: 22_100 + index };
    member.backend = {
      engine: "python-torch",
      version: "2",
      modelFormats: ["safetensors"],
      executionModes: ["tensor-parallel-cell"],
    };
    member.capabilities = {
      deviceKinds: [options.gpu ? "gpu" : "cpu"],
      computeApis: options.gpu ? ["nccl", computeApi] : ["gloo"],
      weightDtypes: [normalizedDtype],
      activationCodecs: ["fp16"],
      features: ["rank-local-kv"],
    };
  }
  const physicalPlan: DistributionPlan = {
    algorithm: "explicit-cell-route",
    codec: "fp16",
    microBatchSize: 1,
    prefillChunkTokens: 8,
    stages: [
      { nodeId: root!.id, layerStart: 0, layerEnd: 1 },
      { nodeId: anchor!.id, layerStart: 1, layerEnd: 3 },
      { nodeId: tail!.id, layerStart: 3, layerEnd: 4 },
    ],
  };
  current.phasePlans = { prefill: physicalPlan, decode: physicalPlan };
  if (options.withCell) {
    current.tensorParallelCells = [
      {
        stageIndex: 1,
        memberNodeIds: [anchor!.id, peer!.id],
        execution: {
          mode: "tensor-parallel-cell",
          engine: "python-torch",
          collectiveBackend: options.gpu ? "nccl" : "gloo",
          computeDtype: options.gpu?.computeDtype ?? "float32",
          fixture: {
            schema: "gdlp-llama-cell-stage/2",
            location: "anchor-local",
            path: "C:/gdlp/cells/tiny/stage-1",
            layerCount: 2,
            manifestSha256: "a".repeat(64),
            shardSha256: ["b".repeat(64), "c".repeat(64)],
            rankMemory: [0, 1].map(() => ({
              fixedBytes: 10 * MIB,
              kvBytesPerToken: 128,
              requiredBytes: 10 * MIB + 16_384,
            })),
          },
          worldSize: 2,
          rankMemberIds: [anchor!.id, peer!.id],
          rankWeights: [1, 1],
          rankDevices: options.gpu?.rankDevices ?? ["cpu", "cpu"],
          operationTimeoutSeconds: 15,
        },
      },
    ];
  }
  return current;
}

function attachMeasuredTensorParallelLinks(request: RuntimePlanRequest): void {
  const cell =
    request.tensorParallelCells?.[0] ??
    request.phaseTensorParallelCells?.prefill?.[0] ??
    request.phaseTensorParallelCells?.decode?.[0];
  if (!cell) throw new Error("test TP cell is missing");
  const members = new Set(cell.memberNodeIds);
  const now = Date.now();
  for (const link of request.topology.links) {
    if (
      link.from !== link.to &&
      members.has(link.from) &&
      members.has(link.to)
    ) {
      link.oneWayLatencyMs = 0.05;
      link.availability = 0.999;
      link.evidence = {
        source: "runtime-probe",
        measuredAt: now - 1_000,
        validUntil: now + 60_000,
        successfulSamples: 8,
        failedSamples: 0,
      };
    }
  }
}

function singleNodePlan(nodeId: string, overrides: Partial<DistributionPlan> = {}): DistributionPlan {
  return {
    algorithm: "test-manual",
    codec: "fp16",
    microBatchSize: 1,
    prefillChunkTokens: 8,
    stages: [{ nodeId, layerStart: 0, layerEnd: 4 }],
    ...overrides,
  };
}

function kvProbe(
  manifest: RuntimePipelineManifestV2,
  overrides: Partial<RuntimeKvGateProbe> = {},
): RuntimeKvGateProbe {
  return {
    prefillPlanId: manifest.plans.prefill.planId,
    decodePlanId: manifest.plans.decode.planId,
    modelRevision: manifest.modelRevision,
    tokenizerId: manifest.tokenizerId,
    layerLayoutHash: manifest.kvTransition.gate.layerLayoutHash,
    formatId: manifest.kvTransition.format.id,
    sourceContextIdentity: "conversation-1:turn-4",
    targetContextIdentity: "conversation-1:turn-4",
    ...overrides,
  };
}

function asLegacyManifest(current: RuntimePipelineManifestV2): RuntimePipelineManifestV1 {
  const decode = current.plans.decode;
  return {
    protocol: "gdlp/1",
    pipelineId: `legacy-${current.pipelineId}`,
    modelId: current.modelId,
    modelRevision: current.modelRevision,
    tokenizerId: current.tokenizerId,
    totalLayers: current.totalLayers,
    hiddenSize: current.hiddenSize,
    activationCodec:
      decode.activationCodec === "fp16" ? "fp16" : "int8",
    transport: "persistent-tcp",
    prefillChunkTokens: current.plans.prefill.chunkTokens,
    microBatchSize: decode.microBatchSize,
    directTokenReturnStage: 0,
    stages: decode.stages.map((stage) => {
      const anchor = stage.members.find(
        (member) => member.nodeId === stage.anchor.memberId,
      )!;
      return {
        index: stage.index,
        nodeId: anchor.nodeId,
        endpoint: anchor.endpoint,
        layerStart: stage.layerStart,
        layerEnd: stage.layerEnd,
        first: stage.first,
        last: stage.last,
        memoryBytes: stage.memoryBytes,
        memoryLimitBytes: stage.memoryLimitBytes,
      };
    }),
    predicted: decode.predicted,
  };
}

function splitStageMemoryAcrossTwoMembers(stage: RuntimePipelineManifestV2["plans"]["decode"]["stages"][number]): void {
  const original = stage.members[0]!;
  const helperAssigned = Math.floor(original.assignedMemoryBytes / 2);
  const helperLimit = Math.floor(original.memoryLimitBytes / 2);
  original.assignedMemoryBytes -= helperAssigned;
  original.memoryLimitBytes -= helperLimit;
  stage.members.push({
    ...structuredClone(original),
    nodeId: `${original.nodeId}-helper`,
    endpoint: { host: "10.20.30.40", port: original.endpoint.port + 100 },
    assignedMemoryBytes: helperAssigned,
    memoryLimitBytes: helperLimit,
  });
}

describe("GDLP/2 runtime manifest", () => {
  it("seals distinct resident rank-weight profiles for prefill and decode", () => {
    const request = cellAwareRequest({ withCell: true });
    const baseCell = request.tensorParallelCells![0]!;
    const prefillCell = structuredClone(baseCell);
    prefillCell.execution.rankWeights = [3, 1];
    prefillCell.execution.fixture.path += "/prefill";
    prefillCell.execution.fixture.manifestSha256 = "d".repeat(64);
    prefillCell.execution.fixture.shardSha256 = ["e".repeat(64), "f".repeat(64)];
    const decodeCell = structuredClone(baseCell);
    decodeCell.execution.rankWeights = [1, 3];
    decodeCell.execution.fixture.path += "/decode";
    decodeCell.execution.fixture.manifestSha256 = "1".repeat(64);
    decodeCell.execution.fixture.shardSha256 = ["2".repeat(64), "3".repeat(64)];
    delete request.tensorParallelCells;
    request.phaseTensorParallelCells = {
      prefill: [prefillCell],
      decode: [decodeCell],
    };
    attachMeasuredTensorParallelLinks(request);

    const first = buildRuntimePipelineManifest(request);
    const second = buildRuntimePipelineManifest(structuredClone(request));
    expect(second).toEqual(first);
    const prefillExecution = first.plans.prefill.stages[1]!.execution!;
    const decodeExecution = first.plans.decode.stages[1]!.execution!;
    expect(prefillExecution.rankWeights).toEqual([3, 1]);
    expect(decodeExecution.rankWeights).toEqual([1, 3]);
    expect(prefillExecution.fixture.path).toMatch(/\/prefill$/);
    expect(decodeExecution.fixture.path).toMatch(/\/decode$/);
    expect(first.plans.prefill.stages[1]!.stageId).not.toBe(
      first.plans.decode.stages[1]!.stageId,
    );
    expect(first.kvTransition.mode).toBe("recompute");
    expect(first.plans.prefill.predicted.calibrationReasons).toContain(
      "phase_specific_tensor_parallel_profile",
    );
    expect(first.plans.decode.predicted.calibrationReasons).toContain(
      "phase_specific_tensor_parallel_profile",
    );
    expect(() => validateRuntimePipelineManifest(first)).not.toThrow();

    const tampered = structuredClone(first);
    tampered.plans.decode.stages[1]!.execution!.rankWeights = [2, 2];
    expect(() => validateRuntimePipelineManifest(tampered)).toThrow(
      "runtime_stage_id_mismatch:decode:1",
    );
    expect(() =>
      compilePythonLaunchDescription(first, {
        apiEndpoint: { host: "0.0.0.0", port: 8_081 },
        returnEndpoint: { host: "root.internal", port: 30_000 },
        returnBindHost: "0.0.0.0",
      }),
    ).toThrow("python_server_does_not_execute_phase_specific_cell_profiles");
  });

  it("requires an explicit non-live transition for differing phase cell profiles", () => {
    const mixed = cellAwareRequest({ withCell: true });
    const commonCells = structuredClone(mixed.tensorParallelCells!);
    mixed.phaseTensorParallelCells = {
      prefill: structuredClone(commonCells),
      decode: structuredClone(commonCells),
    };
    expect(() => buildRuntimePipelineManifest(mixed)).toThrow(
      "runtime_common_and_phase_cells_are_mutually_exclusive",
    );

    const shared = cellAwareRequest({ withCell: true });
    const sharedCells = structuredClone(shared.tensorParallelCells!);
    delete shared.tensorParallelCells;
    shared.phaseTensorParallelCells = {
      prefill: structuredClone(sharedCells),
      decode: structuredClone(sharedCells),
    };
    attachMeasuredTensorParallelLinks(shared);
    const compatible = buildRuntimePipelineManifest(shared);
    expect(compatible.kvTransition.mode).toBe("in-place");
    expect(compatible.plans.prefill.stages[1]!.stageId).toBe(
      compatible.plans.decode.stages[1]!.stageId,
    );
    expect(() =>
      compilePythonLaunchDescription(compatible, {
        apiEndpoint: { host: "0.0.0.0", port: 8_081 },
        returnEndpoint: { host: "root.internal", port: 30_000 },
        returnBindHost: "0.0.0.0",
      }),
    ).not.toThrow();

    delete mixed.tensorParallelCells;
    mixed.phaseTensorParallelCells!.decode![0]!.execution.rankWeights = [3, 1];
    mixed.phaseTensorParallelCells!.decode![0]!.execution.fixture.path += "/decode";
    mixed.phaseTensorParallelCells!.decode![0]!.execution.fixture.manifestSha256 =
      "9".repeat(64);
    mixed.kv = { mode: "in-place" };
    expect(() => buildRuntimePipelineManifest(mixed)).toThrow(
      "runtime_kv_in_place_requires_identical_routes",
    );
  });

  it("plans an otherwise infeasible intermediate stage as an aggregate TP cell", () => {
    expect(() => buildRuntimePipelineManifest(cellAwareRequest())).toThrow(
      "planned_prefill_pipeline_is_infeasible:memory_exceeded:node-1",
    );

    const first = buildRuntimePipelineManifest(cellAwareRequest({ withCell: true }));
    const second = buildRuntimePipelineManifest(cellAwareRequest({ withCell: true }));
    expect(second).toEqual(first);
    for (const phase of [first.plans.prefill, first.plans.decode]) {
      const stage = phase.stages[1]!;
      expect(stage.execution?.mode).toBe("tensor-parallel-cell");
      expect(stage.members.map((member) => member.nodeId)).toEqual(["node-1", "node-2"]);
      expect(stage.memoryLimitBytes).toBe(24 * MIB);
      expect(stage.members.reduce((sum, member) => sum + member.assignedMemoryBytes, 0)).toBe(
        stage.memoryBytes,
      );
      expect(phase.predicted.stageMetrics[1]!.memoryLimitBytes).toBe(24 * MIB);
      expect(phase.predicted.calibrationRequired).toBe(true);
      expect(phase.predicted.calibrationReasons).toContain(
        "tensor_parallel_collective_cost_unprofiled",
      );
    }
    expect(() => validateRuntimePipelineManifest(first)).not.toThrow();
  });

  it("omits empty calibration reasons after measured TP evidence clears the gate", () => {
    const request = cellAwareRequest({ withCell: true });
    attachMeasuredTensorParallelLinks(request);

    const manifest = buildRuntimePipelineManifest(request);

    for (const phase of [manifest.plans.prefill, manifest.plans.decode]) {
      expect(phase.predicted.calibrationRequired).toBe(false);
      expect(phase.predicted).not.toHaveProperty("calibrationReasons");
    }
    expect(() => validateRuntimePipelineManifest(manifest)).not.toThrow();
  });

  it("validates NCCL devices, normalized dtypes and seals the GPU contract", () => {
    const manifests = ([
      ["float32", "fp32"],
      ["float16", "fp16"],
      ["bfloat16", "bf16"],
    ] as const).map(([computeDtype, normalizedDtype]) => {
      const request = cellAwareRequest({
        withCell: true,
        gpu: { computeDtype, rankDevices: ["cuda:0", "cuda:1"] },
      });
      const manifest = buildRuntimePipelineManifest(request);
      const execution = manifest.plans.decode.stages[1]!.execution!;
      expect(execution).toMatchObject({
        collectiveBackend: "nccl",
        computeDtype,
        rankWeights: [1, 1],
        rankDevices: ["cuda:0", "cuda:1"],
      });
      expect(
        manifest.plans.decode.stages[1]!.members.every(
          (member) =>
            member.capabilities.computeApis.includes("nccl") &&
            member.capabilities.computeApis.includes("cuda") &&
            member.capabilities.weightDtypes.includes(normalizedDtype),
        ),
      ).toBe(true);
      expect(() => validateRuntimePipelineManifest(manifest)).not.toThrow();
      return manifest;
    });

    expect(manifests[0]!.plans.decode.stages[1]!.stageId).not.toBe(
      manifests[1]!.plans.decode.stages[1]!.stageId,
    );
    const otherDevices = buildRuntimePipelineManifest(
      cellAwareRequest({
        withCell: true,
        gpu: { computeDtype: "float32", rankDevices: ["cuda:2", "cuda:3"] },
      }),
    );
    expect(otherDevices.plans.decode.stages[1]!.stageId).not.toBe(
      manifests[0]!.plans.decode.stages[1]!.stageId,
    );
    const unequalRequest = cellAwareRequest({
      withCell: true,
      gpu: { computeDtype: "float32", rankDevices: ["cuda:0", "cuda:1"] },
    });
    unequalRequest.tensorParallelCells![0]!.execution.rankWeights = [3, 1];
    const unequal = buildRuntimePipelineManifest(unequalRequest);
    expect(unequal.plans.decode.stages[1]!.stageId).not.toBe(
      manifests[0]!.plans.decode.stages[1]!.stageId,
    );
  });

  it("fails closed for illegal collective, dtype, device and capability combinations", () => {
    const glooDtype = cellAwareRequest({ withCell: true });
    glooDtype.tensorParallelCells![0]!.execution.computeDtype = "float16";
    expect(() => buildRuntimePipelineManifest(glooDtype)).toThrow(
      "runtime_gloo_cell_requires_cpu_float32",
    );

    const glooDevice = cellAwareRequest({ withCell: true });
    glooDevice.tensorParallelCells![0]!.execution.rankDevices[1] = "cuda:0";
    expect(() => buildRuntimePipelineManifest(glooDevice)).toThrow(
      "runtime_gloo_cell_requires_cpu_float32",
    );

    const ncclDevice = cellAwareRequest({
      withCell: true,
      gpu: { computeDtype: "float16", rankDevices: ["cuda:0", "rocm:0"] },
    });
    expect(() => buildRuntimePipelineManifest(ncclDevice)).toThrow(
      "runtime_nccl_cell_requires_cuda_devices",
    );

    const missingBackend = cellAwareRequest({
      withCell: true,
      gpu: { computeDtype: "bfloat16", rankDevices: ["cuda:0", "cuda:1"] },
    });
    missingBackend.topology.nodes[2]!.capabilities!.computeApis = ["cuda"];
    expect(() => buildRuntimePipelineManifest(missingBackend)).toThrow(
      "runtime_cell_member_capabilities_are_not_executable",
    );

    const missingDtype = cellAwareRequest({
      withCell: true,
      gpu: { computeDtype: "bfloat16", rankDevices: ["cuda:0", "cuda:1"] },
    });
    missingDtype.topology.nodes[2]!.capabilities!.weightDtypes = ["fp16"];
    expect(() => buildRuntimePipelineManifest(missingDtype)).toThrow(
      "runtime_cell_member_capabilities_are_not_executable",
    );

    const missingDeviceApi = cellAwareRequest({
      withCell: true,
      gpu: { computeDtype: "float16", rankDevices: ["cuda:0", "cuda:1"] },
    });
    missingDeviceApi.topology.nodes[2]!.capabilities!.computeApis = ["nccl"];
    expect(() => buildRuntimePipelineManifest(missingDeviceApi)).toThrow(
      "runtime_cell_member_capabilities_are_not_executable",
    );

    const wrongCount = cellAwareRequest({
      withCell: true,
      gpu: { computeDtype: "float16", rankDevices: ["cuda:0"] },
    });
    expect(() => buildRuntimePipelineManifest(wrongCount)).toThrow(
      "runtime_cell_rank_devices_are_invalid",
    );

    const wrongWeights = cellAwareRequest({ withCell: true });
    wrongWeights.tensorParallelCells![0]!.execution.rankWeights = [1, 0];
    expect(() => buildRuntimePipelineManifest(wrongWeights)).toThrow(
      "runtime_cell_rank_weights_are_invalid",
    );
  });

  it("rejects a TP cell whose aggregate physical capacity is still insufficient", () => {
    expect(() =>
      buildRuntimePipelineManifest(
        cellAwareRequest({ insufficient: true, withCell: true }),
      ),
    ).toThrow("planned_prefill_pipeline_is_infeasible:memory_exceeded:node-1");
  });

  it("rejects an unbalanced fixture rank even when aggregate capacity is sufficient", () => {
    const request = cellAwareRequest({ withCell: true });
    const rankMemory = request.tensorParallelCells![0]!.execution.fixture.rankMemory;
    rankMemory[0] = {
      fixedBytes: 7 * MIB,
      kvBytesPerToken: 0,
      requiredBytes: 7 * MIB,
    };
    rankMemory[1] = {
      fixedBytes: 13 * MIB,
      kvBytesPerToken: 256,
      requiredBytes: 13 * MIB + 32_768,
    };
    expect(() => buildRuntimePipelineManifest(request)).toThrow(
      "runtime_cell_plan_rank_memory_exceeded:node-2",
    );
  });

  it("requires bidirectional internal links for a member-local cell", () => {
    const request = cellAwareRequest({ withCell: true });
    const cell = request.tensorParallelCells![0]!;
    cell.execution.fixture.location = "member-local";
    cell.execution.external = {
      rankFixturePaths: [cell.execution.fixture.path, "D:/gdlp/cells/tiny/stage-1"],
      controlBindHost: "0.0.0.0",
      controlAdvertiseHost: "10.0.0.2",
      controlPort: 29_100,
      distributedAdvertiseHost: "10.0.0.2",
      distributedPort: 29_101,
      startupTimeoutSeconds: 60,
    };
    request.topology.nodes[2]!.endpoint.host = "rank-1.internal";
    request.topology.links = request.topology.links.filter(
      (link) => !(link.from === "node-1" && link.to === "node-2"),
    );
    expect(() => buildRuntimePipelineManifest(request)).toThrow(
      "runtime_cell_plan_missing_internal_link:node-1<->node-2",
    );
  });

  it("seals planned cell membership into the stage, phase and pipeline identities", () => {
    const manifest = buildRuntimePipelineManifest(cellAwareRequest({ withCell: true }));
    manifest.plans.decode.stages[1]!.members[1]!.memoryLimitBytes += 1;
    expect(() => validateRuntimePipelineManifest(manifest)).toThrow();
  });

  it("rejects invalid numeric topology before producing misleading metrics", () => {
    const request = runtimeRequest();
    request.topology.nodes[0]!.availability = Number.NaN;
    expect(() => buildRuntimePipelineManifest(request)).toThrow(
      "runtime_node_availability_is_invalid",
    );
  });

  it("builds deterministic, separately addressable prefill and decode plans", () => {
    const first = buildRuntimePipelineManifest(runtimeRequest());
    const second = buildRuntimePipelineManifest(runtimeRequest());

    expect(second).toEqual(first);
    expect(first.protocol).toBe("gdlp/2");
    expect(first.plans.prefill.phase).toBe("prefill");
    expect(first.plans.decode.phase).toBe("decode");
    expect(first.plans.prefill.planId).not.toBe(first.plans.decode.planId);
    expect(first.plans.prefill.stages.map((stage) => stage.stageId)).toEqual(
      first.plans.decode.stages.map((stage) => stage.stageId),
    );
    expect(first.kvTransition.mode).toBe("in-place");
    expect(() => validateRuntimePipelineManifest(first)).not.toThrow();
  });

  it("materializes singleton virtual stages with an anchored, capable backend member", () => {
    const manifest = buildRuntimePipelineManifest(runtimeRequest());
    for (const phase of [manifest.plans.prefill, manifest.plans.decode]) {
      for (const stage of phase.stages) {
        expect(stage.members).toHaveLength(1);
        const member = stage.members[0]!;
        expect(stage.anchor).toEqual({ memberId: member.nodeId, endpoint: member.endpoint });
        expect(member.backend.engine).toMatch(/llama\.cpp|mlx/);
        expect(member.backend.modelFormats).toEqual(["gguf"]);
        expect(member.capabilities.activationCodecs).toContain(phase.activationCodec);
        expect(member.assignedMemoryBytes).toBe(stage.memoryBytes);
        expect(member.memoryLimitBytes).toBe(stage.memoryLimitBytes);
      }
    }
  });

  it("declares grouped and Hadamard INT8 only behind the lossy opt-in", () => {
    for (const codec of ["int8-grouped", "int8-hadamard"] as const) {
      const denied = runtimeRequest(1);
      denied.phasePlans = {
        prefill: singleNodePlan("node-0", { codec }),
        decode: singleNodePlan("node-0", { codec }),
      };
      expect(() => buildRuntimePipelineManifest(denied)).toThrow(
        "lossy_runtime_codec_requires_opt_in",
      );

      const allowed = runtimeRequest(1);
      allowed.allowLossyActivation = true;
      allowed.topology.nodes[0]!.capabilities!.activationCodecs!.push(codec);
      allowed.phasePlans = {
        prefill: singleNodePlan("node-0", { codec }),
        decode: singleNodePlan("node-0", { codec }),
      };
      const manifest = buildRuntimePipelineManifest(allowed);
      expect(manifest.plans.prefill.activationCodec).toBe(codec);
      expect(manifest.plans.decode.activationCodec).toBe(codec);
      expect(() => validateRuntimePipelineManifest(manifest)).not.toThrow();
    }
  });

  it("rejects a multi-member mutation whose deterministic IDs were not resealed", () => {
    const manifest = buildRuntimePipelineManifest(runtimeRequest(1));
    splitStageMemoryAcrossTwoMembers(manifest.plans.prefill.stages[0]!);
    splitStageMemoryAcrossTwoMembers(manifest.plans.decode.stages[0]!);

    expect(manifest.plans.prefill.stages[0]!.members).toHaveLength(2);
    expect(() => validateRuntimePipelineManifest(manifest)).toThrow(
      "runtime_stage_id_mismatch:prefill:0",
    );
  });

  it("rejects virtual stages whose anchor is absent or points at another endpoint", () => {
    const absent = structuredClone(buildRuntimePipelineManifest(runtimeRequest(1)));
    absent.plans.prefill.stages[0]!.anchor.memberId = "not-a-member";
    expect(() => validateRuntimePipelineManifest(absent)).toThrow(
      "runtime_stage_anchor_is_not_a_member",
    );

    const mismatch = structuredClone(buildRuntimePipelineManifest(runtimeRequest(1)));
    mismatch.plans.decode.stages[0]!.anchor.endpoint.port += 1;
    expect(() => validateRuntimePipelineManifest(mismatch)).toThrow(
      "runtime_stage_anchor_endpoint_mismatch",
    );
  });

  it("rejects duplicate members and dishonest aggregate memory declarations", () => {
    const duplicate = structuredClone(buildRuntimePipelineManifest(runtimeRequest(1)));
    const stage = duplicate.plans.decode.stages[0]!;
    stage.members.push(structuredClone(stage.members[0]!));
    stage.memoryBytes *= 2;
    stage.memoryLimitBytes *= 2;
    duplicate.kvTransition.mode = "recompute";
    expect(() => validateRuntimePipelineManifest(duplicate)).toThrow(
      "runtime_node_reused_in_phase",
    );

    const dishonest = structuredClone(buildRuntimePipelineManifest(runtimeRequest(1)));
    dishonest.plans.prefill.stages[0]!.memoryBytes += 1;
    expect(() => validateRuntimePipelineManifest(dishonest)).toThrow(
      "runtime_stage_memory_sum_mismatch",
    );
  });

  it("uses independent supplied routes and gates their KV transition to recompute", () => {
    const request = runtimeRequest();
    request.phasePlans = {
      prefill: singleNodePlan("node-0", { prefillChunkTokens: 16 }),
      decode: singleNodePlan("node-1", { microBatchSize: 2 }),
    };
    const manifest = buildRuntimePipelineManifest(request);

    expect(manifest.plans.prefill.stages[0]!.anchor.memberId).toBe("node-0");
    expect(manifest.plans.decode.stages[0]!.anchor.memberId).toBe("node-1");
    expect(manifest.kvTransition.mode).toBe("recompute");
    expect(evaluateRuntimeKvGate(manifest, kvProbe(manifest))).toEqual({
      decision: "recompute",
      allowed: true,
      reasons: [],
    });
  });

  it("does not permit an in-place KV declaration over divergent routes", () => {
    const request = runtimeRequest();
    request.phasePlans = {
      prefill: singleNodePlan("node-0"),
      decode: singleNodePlan("node-1"),
    };
    request.kv = { mode: "in-place" };
    expect(() => buildRuntimePipelineManifest(request)).toThrow(
      "runtime_kv_in_place_requires_identical_routes",
    );
  });

  it("accepts an explicit transfer contract over divergent routes", () => {
    const request = runtimeRequest();
    request.phasePlans = {
      prefill: singleNodePlan("node-0"),
      decode: singleNodePlan("node-1"),
    };
    request.kv = {
      mode: "transfer",
      format: { id: "paged-kv-v3", version: 3, dtype: "fp16", layout: "paged" },
      onMismatch: "reject",
    };
    const manifest = buildRuntimePipelineManifest(request);
    expect(evaluateRuntimeKvGate(manifest, kvProbe(manifest))).toEqual({
      decision: "transfer",
      allowed: true,
      reasons: [],
    });
  });

  it("falls back to recomputation when the strict KV gate detects stale state", () => {
    const manifest = buildRuntimePipelineManifest(runtimeRequest(1));
    const decision = evaluateRuntimeKvGate(
      manifest,
      kvProbe(manifest, {
        modelRevision: "sha256:stale",
        targetContextIdentity: "another-turn",
      }),
    );
    expect(decision).toEqual({
      decision: "recompute",
      allowed: true,
      reasons: ["model_revision_mismatch", "context_identity_mismatch"],
    });
  });

  it("rejects stale KV when the declared mismatch policy is strict rejection", () => {
    const request = runtimeRequest(1);
    request.kv = { onMismatch: "reject" };
    const manifest = buildRuntimePipelineManifest(request);
    const decision = evaluateRuntimeKvGate(
      manifest,
      kvProbe(manifest, { formatId: "wrong-format" }),
    );
    expect(decision).toEqual({
      decision: "reject",
      allowed: false,
      reasons: ["kv_format_mismatch"],
    });
  });

  it("requires matching, present context identities when that gate is enabled", () => {
    const request = runtimeRequest(1);
    request.kv = { onMismatch: "reject" };
    const manifest = buildRuntimePipelineManifest(request);
    const probe = kvProbe(manifest);
    delete probe.sourceContextIdentity;
    delete probe.targetContextIdentity;
    expect(evaluateRuntimeKvGate(manifest, probe)).toEqual({
      decision: "reject",
      allowed: false,
      reasons: ["context_identity_missing"],
    });
  });

  it("carries and validates an adaptive declarative speculation portfolio", () => {
    const speculation: RuntimeSpeculationPolicy = {
      mode: "adaptive",
      controller: "acceptance-adaptive",
      defaultStrategyId: "local-ngram",
      fallbackStrategyId: "autoregressive",
      acceptanceWindowTokens: 64,
      strategies: [
        {
          id: "local-ngram",
          kind: "ngram",
          maxDraftTokens: 4,
          minAcceptanceRate: 0.55,
          maxWasteRatio: 0.35,
          priority: 10,
        },
        {
          id: "stage-head",
          kind: "intermediate-head",
          maxDraftTokens: 3,
          minAcceptanceRate: 0.65,
          maxWasteRatio: 0.25,
          priority: 20,
          artifactId: `sha256:${"a".repeat(64)}`,
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
    };
    const request = runtimeRequest(1);
    request.speculation = speculation;
    const manifest = buildRuntimePipelineManifest(request);

    expect(manifest.plans.decode.speculation).toEqual(speculation);
    expect(() => validateRuntimePipelineManifest(manifest)).not.toThrow();

    const missingArtifact = structuredClone(speculation);
    delete missingArtifact.strategies[1]!.artifactId;
    const missingRequest = runtimeRequest(1);
    missingRequest.speculation = missingArtifact;
    expect(() => buildRuntimePipelineManifest(missingRequest)).toThrow(
      "runtime_speculation_artifact_is_missing",
    );

    const weakArtifact = structuredClone(speculation);
    weakArtifact.strategies[1]!.artifactId = "stage-head-r1";
    const weakRequest = runtimeRequest(1);
    weakRequest.speculation = weakArtifact;
    expect(() => buildRuntimePipelineManifest(weakRequest)).toThrow(
      "runtime_speculation_artifact_is_invalid",
    );
  });

  it("rejects unknown speculation references and non-autoregressive fallback", () => {
    const unknown = buildRuntimePipelineManifest(runtimeRequest(1));
    unknown.plans.decode.speculation.defaultStrategyId = "missing";
    expect(() => validateRuntimePipelineManifest(unknown)).toThrow(
      "runtime_speculation_default_is_unknown",
    );

    const fallback = buildRuntimePipelineManifest(runtimeRequest(1));
    fallback.plans.decode.speculation.mode = "adaptive";
    fallback.plans.decode.speculation.controller = "acceptance-adaptive";
    fallback.plans.decode.speculation.strategies.push({
      id: "ngram",
      kind: "ngram",
      maxDraftTokens: 2,
      minAcceptanceRate: 0.5,
      maxWasteRatio: 0.5,
      priority: 1,
    });
    fallback.plans.decode.speculation.fallbackStrategyId = "ngram";
    expect(() => validateRuntimePipelineManifest(fallback)).toThrow(
      "runtime_speculation_fallback_must_be_autoregressive",
    );
  });

  it("keeps lossy activation and unimplemented Q4 plans behind hard gates", () => {
    const lossy = runtimeRequest(1);
    lossy.phasePlans = {
      prefill: singleNodePlan("node-0", { codec: "int8" }),
      decode: singleNodePlan("node-0"),
    };
    expect(() => buildRuntimePipelineManifest(lossy)).toThrow(
      "lossy_runtime_codec_requires_opt_in:prefill",
    );

    const q4 = runtimeRequest(1);
    q4.phasePlans = {
      prefill: singleNodePlan("node-0"),
      decode: singleNodePlan("node-0", { codec: "q4" }),
    };
    expect(() => buildRuntimePipelineManifest(q4)).toThrow(
      "runtime_codec_not_implemented:q4",
    );
  });

  it("rejects a member whose declared capabilities cannot carry the plan codec", () => {
    const request = runtimeRequest(1);
    request.topology.nodes[0]!.capabilities!.activationCodecs = ["int8"];
    expect(() => buildRuntimePipelineManifest(request)).toThrow(
      "runtime_member_does_not_support_activation_codec",
    );
  });

  it("reads and validates GDLP/1 without silently rewriting it", () => {
    const current = buildRuntimePipelineManifest(runtimeRequest(1));
    const legacy = asLegacyManifest(current);
    const parsed = readRuntimePipelineManifest(JSON.stringify(legacy));

    expect(parsed).toEqual(legacy);
    expect(parsed.protocol).toBe("gdlp/1");
    expect(() => validateRuntimePipelineManifest(parsed)).not.toThrow();
  });

  it("applies the legacy coverage and node-reuse invariants", () => {
    const current = buildRuntimePipelineManifest(runtimeRequest());
    const legacy = asLegacyManifest(current);
    const duplicate = structuredClone(legacy);
    const first = duplicate.stages[0]!;
    duplicate.stages = [
      { ...first, index: 0, layerStart: 0, layerEnd: 2, first: true, last: false },
      { ...first, index: 1, layerStart: 2, layerEnd: 4, first: false, last: true },
    ];
    expect(() => validateRuntimePipelineManifest(duplicate)).toThrow("runtime_node_reused");

    const gap = structuredClone(legacy);
    gap.stages[0]!.layerStart = 1;
    expect(() => validateRuntimePipelineManifest(gap)).toThrow(
      "runtime_layers_are_not_contiguous",
    );
  });

  it("fails closed for malformed JSON, unknown protocols and incomplete objects", () => {
    expect(() => readRuntimePipelineManifest("{broken")).toThrow(
      "invalid_runtime_manifest_json",
    );
    expect(() => readRuntimePipelineManifest({ protocol: "gdlp/99" })).toThrow(
      "unsupported_runtime_protocol",
    );
    expect(() => readRuntimePipelineManifest({ protocol: "gdlp/2" })).toThrow(
      "runtime_pipeline_id_cannot_be_empty",
    );
    expect(() => readRuntimePipelineManifest(null)).toThrow(
      "runtime_manifest_must_be_an_object",
    );
  });
});
