import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import {
  compilePythonLaunchDescription,
  pythonPrefillFrameByteReservation,
  validatePythonLaunchDescription,
  type PythonLaunchCompilerOptions,
  type PythonNativeGgufStageInput,
  type PythonPagedKvStageInput,
  type PythonCellMemberLaunch,
  type PythonPipelineLaunchDescription,
  type PythonRemoteStageLaunch,
  type PythonRootEngineLaunch,
} from "../src/distribution/python-launcher.js";
import {
  buildRuntimePipelineManifest,
  certifyRuntimeTensorParallelCollectives,
  materializeRuntimeTensorParallelCell,
  validateRuntimePipelineManifest,
  type RuntimePipelineManifestV1,
  type RuntimePipelineManifestV2,
  type RuntimePlanRequest,
  type RuntimeSpeculationPolicy,
  type RuntimeTopology,
  type RuntimeVirtualStageManifest,
} from "../src/distribution/runtime-manifest.js";
import type {
  DistributedModelProfile,
  DistributionPlan,
  DistributionWorkload,
} from "../src/distribution/types.js";

const MIB = 1024 * 1024;

function model(): DistributedModelProfile {
  return {
    id: "launcher-model",
    layers: Array.from({ length: 6 }, (_, index) => ({
      index,
      weightBytes: 30 * MIB,
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
  };
}

function workload(maxStages = 3): DistributionWorkload {
  return {
    promptTokens: 16,
    outputTokens: 8,
    contextTokens: 32,
    concurrentSequences: 2,
    maxStages,
    maxQualityLoss: 1,
    minRouteAvailability: 0.8,
    batchWindowMs: 1,
    p95: true,
  };
}

function route(nodeOffset: number, codec: DistributionPlan["codec"] = "fp16"): DistributionPlan {
  return {
    algorithm: "launcher-fixture",
    codec,
    microBatchSize: 2,
    prefillChunkTokens: 8,
    stages: Array.from({ length: 3 }, (_, index) => ({
      nodeId: `node-${nodeOffset + index}`,
      layerStart: index * 2,
      layerEnd: (index + 1) * 2,
    })),
  };
}

function request(nodeCount = 6): RuntimePlanRequest {
  const nodes = Array.from({ length: nodeCount }, (_, index) => ({
    id: `node-${index}`,
    region: `region-${Math.floor(index / 3)}`,
    memoryBytes: 96 * MIB,
    reserveBytes: 8 * MIB,
    decodeScale: 1 + index * 0.02,
    prefillScale: 1 + index * 0.02,
    codecScale: 1,
    batchGain: 0.2,
    maxBatchSpeedup: 1.5,
    powerWatts: 75,
    availability: 0.999,
    endpoint: { host: `stage-${index}.internal`, port: 22_000 + index },
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
      activationCodecs: [
        "fp16" as const,
        "int8" as const,
        "int8-grouped" as const,
        "int8-hadamard" as const,
      ],
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
        oneWayLatencyMs: 2,
        jitterP95Ms: 0.5,
        bandwidthMbps: 1_000,
        lossRate: 0,
        availability: 0.999,
      });
    }
  }
  return {
    model: model(),
    modelRevision: "sha256:launcher-model-r1",
    tokenizerId: "launcher-tokenizer-r1",
    topology: { nodes, links },
    workload: workload(),
    phasePlans: { prefill: route(0), decode: route(0) },
  };
}

function manifest(): RuntimePipelineManifestV2 {
  return buildRuntimePipelineManifest(request());
}

function speculativeConveyorManifest(
  microBatchSize = 1,
): RuntimePipelineManifestV2 {
  const input = request();
  if (!input.phasePlans?.prefill || !input.phasePlans.decode) {
    throw new Error("speculative_conveyor_fixture_requires_both_phase_plans");
  }
  input.phasePlans.prefill.microBatchSize = microBatchSize;
  input.phasePlans.decode.microBatchSize = microBatchSize;
  input.speculation = {
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
  };
  return buildRuntimePipelineManifest(input);
}

function options(overrides: Partial<PythonLaunchCompilerOptions> = {}): PythonLaunchCompilerOptions {
  return {
    apiEndpoint: { host: "0.0.0.0", port: 8_081 },
    returnEndpoint: { host: "root.internal", port: 30_000 },
    returnBindHost: "0.0.0.0",
    pythonExecutable: "C:\\runtime\\python.exe",
    threadsPerStage: 3,
    connectTimeoutSeconds: 45.5,
    ...overrides,
  };
}

function strongArtifactIdentity(snapshotIdentity: string): string {
  const prefix = BigInt(snapshotIdentity).toString(16).padStart(16, "0");
  return `sha256:${prefix}${"a".repeat(48)}`;
}

function compile(
  current: RuntimePipelineManifestV2 = manifest(),
  overrides: Partial<PythonLaunchCompilerOptions> = {},
): PythonPipelineLaunchDescription {
  return compilePythonLaunchDescription(current, options(overrides));
}

function root(description: PythonPipelineLaunchDescription): PythonRootEngineLaunch {
  const result = description.launchOrder.at(-1)!;
  expect(result.kind).toBe("root-engine");
  return result as PythonRootEngineLaunch;
}

function remotes(description: PythonPipelineLaunchDescription): PythonRemoteStageLaunch[] {
  return description.launchOrder.filter(
    (entry): entry is PythonRemoteStageLaunch => entry.kind === "remote-stage",
  );
}

function legacyNativeStageBindings(
  description: PythonPipelineLaunchDescription,
): Record<string, LegacyNativeStageStageInput> {
  return (
    description.configuration as unknown as {
      native_stageStages: Record<string, LegacyNativeStageStageInput>;
    }
  ).native_stageStages;
}

function legacyNativeStageLaunch(
  launch: PythonRemoteStageLaunch,
): LegacyNativeStageStageInput | null {
  return (
    launch as unknown as {
      native_stage: LegacyNativeStageStageInput | null;
    }
  ).native_stage;
}

function cellMembers(description: PythonPipelineLaunchDescription): PythonCellMemberLaunch[] {
  return description.launchOrder.filter(
    (entry): entry is PythonCellMemberLaunch => entry.kind === "cell-member",
  );
}

function argumentValue(args: string[], flag: string): string | undefined {
  const index = args.indexOf(flag);
  return index < 0 ? undefined : args[index + 1];
}

function argumentValues(args: string[], flag: string): string[] {
  return args.flatMap((value, index) =>
    value === flag && args[index + 1] !== undefined ? [args[index + 1]!] : [],
  );
}

function moduleName(args: string[]): string | undefined {
  return argumentValue(args, "-m");
}

const NATIVE_STAGE_TEST_COMMIT = "12fd25f77366fa6b3b4b768ec3050bf629380bac";
const NATIVE_GGUF_SNAPSHOT_ID = "12345";
const NATIVE_GGUF_MODEL_IDENTITY = strongArtifactIdentity(NATIVE_GGUF_SNAPSHOT_ID);
const NATIVE_GGUF_MODEL_SOURCE = "mycellios://models/launcher-native-gguf";
const NATIVE_GGUF_MODEL_REVISION = "native-gguf-r1";

/** Archived research shape; it is deliberately absent from product exports. */
interface LegacyNativeStageStageInput {
  packagePath: string;
  packageId: string;
  manifestSha256: string;
  modelSource: string;
  modelRevision: string | null;
  layerStart: number;
  layerEnd: number;
  totalLayers: number;
  daemonExecutable: string;
  pipelineId: string;
  contextTokens: number;
  gpuLayers: number;
  computeApi: "cpu" | "cuda" | "rocm" | "metal" | "vulkan";
  startupTimeoutSeconds: number;
  callTimeoutSeconds: number;
  closeTimeoutSeconds: number;
  modelIdentity?: string;
}

type LegacyNativeStageOptions = Partial<PythonLaunchCompilerOptions> & {
  native_stageStages: Record<string, LegacyNativeStageStageInput>;
};

function native_stagePipelineId(commit = NATIVE_STAGE_TEST_COMMIT): string {
  const bytes = createHash("sha256")
    .update("gdlp-hub-snapshot-v1\0")
    .update(commit, "ascii")
    .digest()
    .subarray(0, 8);
  let result = 0n;
  for (const byte of bytes) result = (result << 8n) | BigInt(byte);
  return result.toString();
}

function native_stageStage(
  current: RuntimePipelineManifestV2,
  stageIndex: number,
  overrides: Partial<LegacyNativeStageStageInput> = {},
): LegacyNativeStageStageInput {
  const stage = current.plans.prefill.stages[stageIndex]!;
  return {
    packagePath: `D:\\packages\\${stage.stageId}`,
    packageId: "a".repeat(64),
    manifestSha256: "b".repeat(64),
    modelSource: "hf://HuggingFaceTB/SmolLM2-135M-Instruct",
    modelRevision: NATIVE_STAGE_TEST_COMMIT,
    layerStart: stage.layerStart,
    layerEnd: stage.layerEnd,
    totalLayers: current.totalLayers,
    daemonExecutable: "D:\\bin\\llama-native_stage-worker.exe",
    pipelineId: native_stagePipelineId(),
    contextTokens: 4_096,
    gpuLayers: 99,
    computeApi: "cuda",
    startupTimeoutSeconds: 90,
    callTimeoutSeconds: 30.5,
    closeTimeoutSeconds: 5,
    ...overrides,
  };
}

function native_stageOptions(
  current: RuntimePipelineManifestV2,
  stageIndexes: number[] = [2],
): LegacyNativeStageOptions {
  const snapshot = `C:\\cache\\models--HuggingFaceTB--SmolLM2-135M-Instruct\\snapshots\\${NATIVE_STAGE_TEST_COMMIT}`;
  return {
    runtimeModel: { source: snapshot, revision: null },
    native_stageStages: Object.fromEntries(
      stageIndexes.map((index) => {
        const stage = current.plans.prefill.stages[index]!;
        return [stage.stageId, native_stageStage(current, index)];
      }),
    ),
  };
}

function nativeGgufStage(
  current: RuntimePipelineManifestV2,
  stageIndex: number,
  overrides: Partial<PythonNativeGgufStageInput> = {},
): PythonNativeGgufStageInput {
  const stage = current.plans.prefill.stages[stageIndex]!;
  return {
    packagePath: `D:\\native-gguf\\${stage.stageId}`,
    packageId: "c".repeat(64),
    modelIdentity: NATIVE_GGUF_MODEL_IDENTITY,
    modelSource: NATIVE_GGUF_MODEL_SOURCE,
    modelRevision: NATIVE_GGUF_MODEL_REVISION,
    layerStart: stage.layerStart,
    layerEnd: stage.layerEnd,
    totalLayers: current.totalLayers,
    ...overrides,
  };
}

function nativeGgufOptions(
  current: RuntimePipelineManifestV2,
  stageIndexes: number[] = [2],
): Partial<PythonLaunchCompilerOptions> {
  return {
    runtimeModel: {
      source: "D:\\models\\launcher-native-root",
      revision: null,
      snapshotIdentity: NATIVE_GGUF_SNAPSHOT_ID,
      artifactIdentity: NATIVE_GGUF_MODEL_IDENTITY,
      canonicalSource: NATIVE_GGUF_MODEL_SOURCE,
      canonicalRevision: NATIVE_GGUF_MODEL_REVISION,
    },
    nativeGgufStages: Object.fromEntries(
      stageIndexes.map((index) => {
        const stage = current.plans.prefill.stages[index]!;
        return [stage.stageId, nativeGgufStage(current, index)];
      }),
    ),
  };
}

function pagedKvStage(
  overrides: Partial<PythonPagedKvStageInput> = {},
): PythonPagedKvStageInput {
  return {
    schema: "mycellios-hf-paged-stage/2",
    device: "cuda:0",
    attentionBackend: "eager",
    blockSize: 16,
    numBlocks: 4_096,
    maxBatchTokens: 256,
    maxActiveRequests: 16,
    maxSequenceTokens: 4_096,
    cpuSpillBytes: 64 * MIB,
    ...overrides,
  };
}

function pagedKvOptions(
  current: RuntimePipelineManifestV2,
  overrides: Partial<PythonPagedKvStageInput> = {},
): Partial<PythonLaunchCompilerOptions> {
  return {
    pagedKvStages: Object.fromEntries(
      current.plans.prefill.stages.map((stage) => [
        stage.stageId,
        pagedKvStage(overrides),
      ]),
    ),
  };
}

function pagedRuntimeManifest(
  overrides: {
    engine?: string;
    modelFormats?: string[];
    executionModes?: string[];
    deviceKinds?: string[];
    computeApis?: string[];
  } = {},
): RuntimePipelineManifestV2 {
  const input = request();
  for (const node of input.topology.nodes) {
    node.backend = {
      ...node.backend,
      engine: overrides.engine ?? "python-transformers",
      modelFormats: overrides.modelFormats ?? ["safetensors"],
      executionModes: overrides.executionModes ?? ["layer-range"],
    };
    node.capabilities = {
      ...node.capabilities,
      deviceKinds: overrides.deviceKinds ?? ["gpu"],
      computeApis: overrides.computeApis ?? ["cuda"],
    };
  }
  return buildRuntimePipelineManifest(input);
}

function addCooperativeMember(stage: RuntimeVirtualStageManifest): void {
  const original = stage.members[0]!;
  const assigned = Math.floor(original.assignedMemoryBytes / 2);
  const limit = Math.floor(original.memoryLimitBytes / 2);
  original.assignedMemoryBytes -= assigned;
  original.memoryLimitBytes -= limit;
  stage.members.push({
    ...structuredClone(original),
    nodeId: `${original.nodeId}-peer`,
    endpoint: { host: `${original.nodeId}-peer.internal`, port: original.endpoint.port + 100 },
    assignedMemoryBytes: assigned,
    memoryLimitBytes: limit,
  });
}

function materializeTensorParallelCell(
  current: RuntimePipelineManifestV2,
  stageIndex = 1,
  extraCellLimitBytes = 0,
  gpu?: {
    computeDtype: "float32" | "float16" | "bfloat16";
    computeApi?: "cuda" | "rocm";
    rankDevices?: string[];
  },
): RuntimePipelineManifestV2 {
  const stage = current.plans.prefill.stages[stageIndex]!;
  const anchor = structuredClone(stage.members[0]!);
  const peerAssigned = Math.floor(anchor.assignedMemoryBytes / 2);
  const peerLimit = Math.floor(anchor.memoryLimitBytes / 2);
  anchor.assignedMemoryBytes -= peerAssigned;
  anchor.memoryLimitBytes -= peerLimit;
  const peer = {
    ...structuredClone(anchor),
    nodeId: `${anchor.nodeId}-peer`,
    endpoint: {
      host: stage.anchor.endpoint.host,
      port: anchor.endpoint.port + 100,
    },
    assignedMemoryBytes: peerAssigned,
    memoryLimitBytes: peerLimit + extraCellLimitBytes,
  };
  const activationCodecs = [
    ...new Set([
      current.plans.prefill.activationCodec,
      current.plans.decode.activationCodec,
    ]),
  ];
  const computeApi = gpu?.computeApi ?? "cuda";
  const normalizedDtype =
    gpu?.computeDtype === "float16"
      ? "fp16"
      : gpu?.computeDtype === "bfloat16"
        ? "bf16"
        : "fp32";
  const members = [anchor, peer].map((member) => ({
    ...member,
    endpoint: { ...member.endpoint, host: stage.anchor.endpoint.host },
    backend: {
      engine: "python-torch",
      version: "2",
      modelFormats: ["safetensors"],
      executionModes: ["tensor-parallel-cell"],
    },
    capabilities: {
      deviceKinds: [gpu ? "gpu" : "cpu"],
      computeApis: gpu ? ["nccl", computeApi] : ["gloo"],
      weightDtypes: [normalizedDtype],
      activationCodecs,
      features: ["rank-local-kv"],
    },
  }));
  return materializeRuntimeTensorParallelCell(current, {
    stageIndex,
    members,
    execution: {
      mode: "tensor-parallel-cell",
      engine: "python-torch",
      collectiveBackend: gpu ? "nccl" : "gloo",
      computeDtype: gpu?.computeDtype ?? "float32",
      fixture: {
        schema: "gdlp-llama-cell-stage/2",
        location: "anchor-local",
        path: `/srv/gdlp/cells/${stage.anchor.memberId}/stage-${stage.index}`,
        layerCount: stage.layerEnd - stage.layerStart,
        manifestSha256: "a".repeat(64),
        shardSha256: ["b".repeat(64), "c".repeat(64)],
        rankMemory: members.map((member) => ({
          fixedBytes: member.assignedMemoryBytes,
          kvBytesPerToken: 0,
          requiredBytes: member.assignedMemoryBytes,
        })),
      },
      worldSize: members.length,
      rankMemberIds: members.map((member) => member.nodeId),
      rankWeights: members.map(() => 1),
      rankDevices: gpu?.rankDevices ?? members.map(() => "cpu"),
      operationTimeoutSeconds: 12.5,
    },
  });
}

function materializeExternalTensorParallelCell(
  current: RuntimePipelineManifestV2,
  gpu?: {
    computeDtype: "float32" | "float16" | "bfloat16";
    computeApi?: "cuda" | "rocm";
    rankDevices?: string[];
  },
): RuntimePipelineManifestV2 {
  const local = materializeTensorParallelCell(current, 1, 0, gpu);
  const stage = local.plans.prefill.stages[1]!;
  const members = structuredClone(stage.members);
  members[1]!.endpoint = { host: "rank-1.internal", port: 24_101 };
  const execution = structuredClone(stage.execution!);
  execution.fixture.location = "member-local";
  execution.external = {
    rankFixturePaths: [execution.fixture.path, "D:/gdlp/cells/stage-1"],
    controlBindHost: "0.0.0.0",
    controlAdvertiseHost: "10.20.0.10",
    controlPort: 29_100,
    distributedAdvertiseHost: "10.20.0.10",
    distributedPort: 29_101,
    startupTimeoutSeconds: 90,
  };
  return materializeRuntimeTensorParallelCell(local, {
    stageIndex: 1,
    members,
    execution,
  });
}

function certifyTensorParallelCell(
  current: RuntimePipelineManifestV2,
): RuntimePipelineManifestV2 {
  const now = Date.now();
  return certifyRuntimeTensorParallelCollectives(
    current,
    measuredTensorParallelTopology(current, now),
    now,
  );
}

function measuredTensorParallelTopology(
  current: RuntimePipelineManifestV2,
  now: number,
): RuntimeTopology {
  const nodes = new Map<
    string,
    RuntimeTopology["nodes"][number]
  >();
  const cellMemberIds: string[][] = [];
  for (const stage of current.plans.prefill.stages) {
    for (const member of stage.members) {
      nodes.set(member.nodeId, {
        id: member.nodeId,
        region: "physical-tp-fixture",
        memoryBytes: member.memoryLimitBytes,
        reserveBytes: 0,
        decodeScale: 1,
        prefillScale: 1,
        codecScale: 1,
        batchGain: 0,
        maxBatchSpeedup: 1,
        powerWatts: 75,
        availability: 0.999,
        endpoint: { ...member.endpoint },
        backend: structuredClone(member.backend),
        capabilities: structuredClone(member.capabilities),
      });
    }
    if (stage.execution?.mode === "tensor-parallel-cell") {
      cellMemberIds.push([...stage.execution.rankMemberIds]);
    }
  }
  const links = new Map<string, RuntimeTopology["links"][number]>();
  for (const memberIds of cellMemberIds) {
    for (const from of memberIds) {
      for (const to of memberIds) {
        if (from === to) continue;
        links.set(`${from}\0${to}`, {
          from,
          to,
          oneWayLatencyMs: 0.2,
          jitterP95Ms: 0.05,
          bandwidthMbps: 10_000,
          lossRate: 0,
          availability: 0.999,
          evidence: {
            source: "runtime-probe",
            measuredAt: now - 1_000,
            validUntil: now + 60_000,
            successfulSamples: 8,
            failedSamples: 0,
          },
        });
      }
    }
  }
  return {
    nodes: [...nodes.values()],
    links: [...links.values()],
  };
}

function legacy(current: RuntimePipelineManifestV2): RuntimePipelineManifestV1 {
  const plan = current.plans.decode;
  return {
    protocol: "gdlp/1",
    pipelineId: `legacy-${current.pipelineId}`,
    modelId: current.modelId,
    modelRevision: current.modelRevision,
    tokenizerId: current.tokenizerId,
    totalLayers: current.totalLayers,
    hiddenSize: current.hiddenSize,
    activationCodec: "fp16",
    transport: "persistent-tcp",
    prefillChunkTokens: current.plans.prefill.chunkTokens,
    microBatchSize: plan.microBatchSize,
    directTokenReturnStage: 0,
    stages: plan.stages.map((stage) => ({
      index: stage.index,
      nodeId: stage.anchor.memberId,
      endpoint: stage.anchor.endpoint,
      layerStart: stage.layerStart,
      layerEnd: stage.layerEnd,
      first: stage.first,
      last: stage.last,
      memoryBytes: stage.memoryBytes,
      memoryLimitBytes: stage.memoryLimitBytes,
    })),
    predicted: plan.predicted,
  };
}

function oneStageManifest(): RuntimePipelineManifestV2 {
  const input = request(1);
  input.topology.nodes[0]!.memoryBytes = 512 * MIB;
  input.workload = workload(1);
  const only: DistributionPlan = {
    algorithm: "single",
    codec: "fp16",
    microBatchSize: 1,
    prefillChunkTokens: 8,
    stages: [{ nodeId: "node-0", layerStart: 0, layerEnd: 6 }],
  };
  input.phasePlans = { prefill: only, decode: only };
  return buildRuntimePipelineManifest(input);
}

describe("GDLP/2 Python launch compiler", () => {
  it("is reproducible and validates every derived process and argv", () => {
    const first = compile();
    const second = compile();
    expect(second).toEqual(first);
    expect(JSON.stringify(second)).toBe(JSON.stringify(first));
    expect(first.schema).toBe("gdlp-python-launch/2");
    expect(first.launchId).toMatch(/^[a-f0-9]{24}$/);
    expect(() => validatePythonLaunchDescription(first)).not.toThrow();
  });

  it("seals the executor isolation policy into every process and launch identity", () => {
    const defaultLaunch = compile();
    const stricterDiagnostics = compile(manifest(), {
      executorIsolation: {
        maxOutputBytesPerStream: 32 * 1024,
        stopGraceMs: 4_000,
      },
    });

    expect(defaultLaunch.configuration.executorIsolation).toMatchObject({
      schema: "gdlp-executor-isolation/4",
      workspacePolicy: "private-temp-shared-runtime",
      processTreePolicy: "best-effort-process-tree",
      resourceLimitPolicy: "workspace-watchdog-only",
    });
    for (const process of defaultLaunch.launchOrder) {
      expect(process.isolation).toEqual(
        defaultLaunch.configuration.executorIsolation,
      );
    }
    expect(stricterDiagnostics.launchId).not.toBe(defaultLaunch.launchId);
    expect(stricterDiagnostics.configuration.executorIsolation).toMatchObject({
      maxOutputBytesPerStream: 32 * 1024,
      stopGraceMs: 4_000,
    });
    expect(() => validatePythonLaunchDescription(stricterDiagnostics)).not.toThrow();

    const tampered = structuredClone(defaultLaunch);
    tampered.launchOrder[0]!.isolation.resourceLimitPolicy =
      "os-enforced" as "workspace-watchdog-only";
    expect(() => validatePythonLaunchDescription(tampered)).toThrow(
      "python_launch_description_mismatch",
    );
  });

  it("launches stages N-1..1 downstream-first and the root engine last", () => {
    const description = compile();
    expect(description.launchOrder.map((entry) => entry.stageIndex)).toEqual([2, 1, 0]);
    expect(description.launchOrder.map((entry) => entry.kind)).toEqual([
      "remote-stage",
      "remote-stage",
      "root-engine",
    ]);
    expect(description.launchOrder.map((entry) => entry.launchIndex)).toEqual([0, 1, 2]);
    expect(description.route.downstreamFirstProcessIds).toEqual(
      description.launchOrder.map((entry) => entry.processId),
    );
    expect(description.route.rootProcessId).toBe(root(description).processId);
  });

  it("seals a complete native paged-KV contract into every stage process", () => {
    const current = manifest();
    const description = compile(current, pagedKvOptions(current));
    const stageIds = current.plans.prefill.stages.map((stage) => stage.stageId).sort();
    expect(Object.keys(description.configuration.pagedKvStages)).toEqual(stageIds);
    for (const process of description.launchOrder) {
      expect(process.command.args).toContain("--paged-kv");
      expect(argumentValue(process.command.args, "--paged-device")).toBe("cuda:0");
      expect(argumentValue(process.command.args, "--paged-attention-backend")).toBe(
        "eager",
      );
      expect(argumentValue(process.command.args, "--paged-block-size")).toBe("16");
      expect(argumentValue(process.command.args, "--paged-num-blocks")).toBe("4096");
      expect(
        argumentValue(process.command.args, "--paged-max-batch-tokens"),
      ).toBe("256");
      expect(
        argumentValue(process.command.args, "--paged-max-active-requests"),
      ).toBe("16");
      expect(
        argumentValue(process.command.args, "--paged-max-sequence-tokens"),
      ).toBe("4096");
      expect(argumentValue(process.command.args, "--paged-cpu-spill-bytes")).toBe(
        String(64 * MIB),
      );
    }
    expect(() => validatePythonLaunchDescription(description)).not.toThrow();

    const changed = compile(
      current,
      pagedKvOptions(current, { cpuSpillBytes: 32 * MIB }),
    );
    expect(changed.route.routeId).not.toBe(description.route.routeId);
    expect(changed.launchId).not.toBe(description.launchId);
  });

  it("fails closed for partial, unsafe or conflicting paged-KV contracts", () => {
    const current = manifest();
    const firstStage = current.plans.prefill.stages[0]!;
    expect(() =>
      compile(current, {
        pagedKvStages: { [firstStage.stageId]: pagedKvStage() },
      }),
    ).toThrow("python_paged_kv_stage_binding_is_missing");

    expect(() =>
      compile(current, pagedKvOptions(current, { device: "cpu" })),
    ).toThrow("python_paged_kv_cpu_spill_requires_gpu");
    expect(() =>
      compile(
        current,
        pagedKvOptions(current, {
          numBlocks: 8,
          maxSequenceTokens: 4_096,
        }),
      ),
    ).toThrow("python_paged_kv_sequence_exceeds_pool");
    expect(() =>
      compile(current, pagedKvOptions(current, { maxActiveRequests: 5 })),
    ).toThrow("python_paged_kv_active_request_capacity_is_too_small");
    expect(() =>
      compile(current, pagedKvOptions(current, { maxBatchTokens: 7 })),
    ).toThrow("python_paged_kv_batch_token_capacity_is_too_small");

    expect(() =>
      compile(current, {
        ...nativeGgufOptions(current),
        ...pagedKvOptions(current),
      }),
    ).toThrow("python_paged_kv_stage_backend_is_not_exclusive");

    const cell = certifyTensorParallelCell(
      materializeTensorParallelCell(current),
    );
    expect(() => compile(cell, pagedKvOptions(cell))).toThrow(
      "python_paged_kv_stage_has_conflicting_execution",
    );
  });

  it("binds paged-KV only to a compatible, verified native member", () => {
    const gpu = pagedRuntimeManifest();
    expect(() => compile(gpu, pagedKvOptions(gpu))).not.toThrow();
    expect(() =>
      compile(gpu, pagedKvOptions(gpu, { device: "cuda:1" })),
    ).toThrow("python_paged_kv_device_index_is_not_verified");
    expect(() =>
      compile(
        gpu,
        pagedKvOptions(gpu, { device: "cpu", cpuSpillBytes: 0 }),
      ),
    ).toThrow("python_paged_kv_stage_lacks_cpu_capability");

    const cpu = pagedRuntimeManifest({
      deviceKinds: ["cpu"],
      computeApis: ["torch"],
    });
    expect(() =>
      compile(
        cpu,
        pagedKvOptions(cpu, { device: "cpu", cpuSpillBytes: 0 }),
      ),
    ).not.toThrow();
    expect(() => compile(cpu, pagedKvOptions(cpu))).toThrow(
      "python_paged_kv_stage_lacks_gpu_capability",
    );

    const rocm = pagedRuntimeManifest({ computeApis: ["rocm"] });
    expect(() => compile(rocm, pagedKvOptions(rocm))).not.toThrow();
    const noGpuRuntime = pagedRuntimeManifest({ computeApis: ["torch"] });
    expect(() => compile(noGpuRuntime, pagedKvOptions(noGpuRuntime))).toThrow(
      "python_paged_kv_stage_lacks_gpu_capability",
    );
  });

  it("rejects paged-KV workers with an incompatible engine or artifact contract", () => {
    const foreignEngine = pagedRuntimeManifest({ engine: "foreign-runtime" });
    expect(() =>
      compile(foreignEngine, pagedKvOptions(foreignEngine)),
    ).toThrow("python_paged_kv_stage_engine_is_not_supported");

    const wrongFormat = pagedRuntimeManifest({ modelFormats: ["gguf"] });
    expect(() => compile(wrongFormat, pagedKvOptions(wrongFormat))).toThrow(
      "python_paged_kv_stage_model_format_is_not_supported",
    );

    const wrongMode = pagedRuntimeManifest({
      executionModes: ["tensor-parallel-cell"],
    });
    expect(() => compile(wrongMode, pagedKvOptions(wrongMode))).toThrow(
      "python_paged_kv_stage_execution_mode_is_not_supported",
    );
  });

  it("includes the sealed speculative wave in paged-KV query capacity", () => {
    const input = request();
    input.speculation = {
      mode: "adaptive",
      controller: "acceptance-adaptive",
      defaultStrategyId: "ngram",
      fallbackStrategyId: "autoregressive",
      acceptanceWindowTokens: 64,
      strategies: [
        {
          id: "ngram",
          kind: "ngram",
          maxDraftTokens: 9,
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
    };
    const current = buildRuntimePipelineManifest(input);
    expect(() =>
      compile(current, {
        ...pagedKvOptions(current, { maxBatchTokens: 9 }),
        maxRetainedSessions: 0,
      }),
    ).toThrow("python_paged_kv_batch_token_capacity_is_too_small");
    expect(() =>
      compile(current, {
        ...pagedKvOptions(current, { maxBatchTokens: 10 }),
        maxRetainedSessions: 0,
      }),
    ).not.toThrow();
  });

  it("proves logical stage zero never invokes stage_cli", () => {
    const description = compile();
    const rootLaunch = root(description);
    expect(rootLaunch.stageIndex).toBe(0);
    expect(moduleName(rootLaunch.command.args)).toBe("distributed_runtime.server");
    expect(rootLaunch.command.args[0]).toBe("-u");
    expect(rootLaunch.command.args).not.toContain("--layer-start");
    for (const remote of remotes(description)) {
      expect(remote.stageIndex).toBeGreaterThan(0);
      expect(moduleName(remote.command.args)).toBe("distributed_runtime.stage_cli");
      expect(remote.command.args[0]).toBe("-u");
    }
  });

  it("configures the root server with boundaries and the first remote stage", () => {
    const description = compile();
    const rootLaunch = root(description);
    const firstRemote = description.sourceManifest.plans.decode.stages[1]!;
    expect(rootLaunch.boundaries).toEqual([0, 2, 4, 6]);
    expect(argumentValue(rootLaunch.command.args, "--boundaries")).toBe("0,2,4,6");
    expect(argumentValue(rootLaunch.command.args, "--first-stage-host")).toBe(
      firstRemote.anchor.endpoint.host,
    );
    expect(argumentValue(rootLaunch.command.args, "--first-stage-port")).toBe(
      String(firstRemote.anchor.endpoint.port),
    );
    expect(argumentValue(rootLaunch.command.args, "--host")).toBe("0.0.0.0");
    expect(argumentValue(rootLaunch.command.args, "--port")).toBe("8081");
    expect(argumentValue(rootLaunch.command.args, "--return-bind-host")).toBe("0.0.0.0");
    expect(argumentValue(rootLaunch.command.args, "--return-advertise-host")).toBe(
      "root.internal",
    );
    expect(argumentValue(rootLaunch.command.args, "--return-port")).toBe("30000");
  });

  it("seals remote standby recovery contracts into the root CLI", () => {
    const current = manifest();
    const stageExecutorIds = current.plans.decode.stages.map((_stage, index) =>
      String(index + 1).repeat(32),
    );
    const description = compile(current, {
      recovery: {
        maxRetries: 2,
        stageExecutorIds,
        standbyRoutes: [
          {
            firstStage: { host: "standby-a.internal", port: 40_001 },
            stageExecutorIds,
          },
          {
            firstStage: { host: "standby-b.internal", port: 40_002 },
            stageExecutorIds,
          },
        ],
      },
    });
    const recovery = description.configuration.recovery!;
    const args = root(description).command.args;

    expect(recovery.maxRetries).toBe(2);
    expect(recovery.standbyRoutes).toHaveLength(2);
    expect(recovery.standbyRoutes[0]!.schema).toBe(
      "gdlp-recovery-standby-route/1",
    );
    expect(recovery.standbyRoutes[0]!.routeId).toMatch(
      /^standby-[0-9a-f]{24}$/,
    );
    expect(argumentValue(args, "--recovery-max-retries")).toBe("2");
    expect(argumentValues(args, "--stage-executor-id")).toEqual(
      stageExecutorIds,
    );
    expect(
      argumentValues(args, "--recovery-standby-route").map((value) =>
        JSON.parse(value),
      ),
    ).toEqual(recovery.standbyRoutes);
    expect(() => validatePythonLaunchDescription(description)).not.toThrow();

    const withoutRecovery = compile(current);
    expect(withoutRecovery.configuration.recovery).toBeNull();
    expect(root(withoutRecovery).command.args).not.toContain(
      "--recovery-max-retries",
    );
    expect(withoutRecovery.route.routeId).not.toBe(description.route.routeId);
  });

  it("rejects incomplete, mismatched or non-independent recovery routes", () => {
    const current = manifest();
    const primary = current.plans.decode.stages[1]!.anchor.endpoint;
    const stageExecutorIds = current.plans.decode.stages.map((_stage, index) =>
      String(index + 1).repeat(32),
    );
    const standby = {
      firstStage: { host: "standby.internal", port: 40_001 },
      stageExecutorIds,
    };

    expect(() =>
      compile(current, {
        recovery: { maxRetries: 1, stageExecutorIds, standbyRoutes: [] },
      }),
    ).toThrow("python_recovery_requires_at_least_one_standby_route");
    expect(() =>
      compile(current, {
        recovery: {
          maxRetries: 1,
          stageExecutorIds: stageExecutorIds.slice(1),
          standbyRoutes: [standby],
        },
      }),
    ).toThrow("python_recovery_primary_executor_contract_is_invalid");
    expect(() =>
      compile(current, {
        recovery: {
          maxRetries: 1,
          stageExecutorIds,
          standbyRoutes: [
            {
              ...standby,
              stageExecutorIds: [
                stageExecutorIds[0]!,
                "f".repeat(32),
                stageExecutorIds[2]!,
              ],
            },
          ],
        },
      }),
    ).toThrow("python_recovery_standby_executor_contract_mismatch");
    expect(() =>
      compile(current, {
        recovery: {
          maxRetries: 1,
          stageExecutorIds,
          standbyRoutes: [
            { firstStage: primary, stageExecutorIds },
          ],
        },
      }),
    ).toThrow("python_recovery_standby_reuses_primary_endpoint");
    expect(() =>
      compile(current, {
        recovery: {
          maxRetries: 1,
          stageExecutorIds,
          standbyRoutes: [standby, structuredClone(standby)],
        },
      }),
    ).toThrow("python_recovery_standby_endpoints_must_be_unique");
  });

  it("places prefill, batching and disabled speculation on the root server argv", () => {
    const description = compile();
    const args = root(description).command.args;
    expect(argumentValue(args, "--prefill-chunk-tokens")).toBe("8");
    expect(argumentValue(args, "--prefill-inflight-chunks")).toBe("3");
    expect(argumentValue(args, "--prefill-inflight-bytes")).toBe(
      String(64 * 1024 * 1024),
    );
    expect(argumentValue(args, "--sealed-wave-tokens")).toBe("1");
    expect(argumentValue(args, "--max-prefill-chunk-tokens")).toBe("8");
    expect(argumentValue(args, "--max-batch-size")).toBe("2");
    expect(argumentValue(args, "--max-active-sequences")).toBe("2");
    expect(argumentValue(args, "--speculation")).toBe("off");
    expect(argumentValue(args, "--speculative-max-draft-tokens")).toBe("1");
    expect(args).not.toContain("--speculative-inflight-waves");
    expect(args).not.toContain("--speculative-inflight-bytes");
    expect(
      Object.prototype.hasOwnProperty.call(
        description.configuration,
        "speculativeInflightWaves",
      ),
    ).toBe(false);
    expect(
      Object.prototype.hasOwnProperty.call(
        description.configuration,
        "speculativeInflightBytes",
      ),
    ).toBe(false);
    const explicitDefaults = compile(manifest(), {
      speculativeInflightWaves: 1,
      speculativeInflightBytes: 0,
    });
    expect(explicitDefaults.launchId).toBe(description.launchId);
    expect(root(explicitDefaults).command.args).not.toContain(
      "--speculative-inflight-waves",
    );
  });

  it("seals explicit bounded prefill pipeline credits into configuration and argv", () => {
    const description = compile(manifest(), {
      prefillInflightChunks: 5,
      prefillInflightBytes: 96 * 1024 * 1024,
    });
    const args = root(description).command.args;
    expect(description.configuration.prefillInflightChunks).toBe(5);
    expect(description.configuration.prefillInflightBytes).toBe(96 * 1024 * 1024);
    expect(argumentValue(args, "--prefill-inflight-chunks")).toBe("5");
    expect(argumentValue(args, "--prefill-inflight-bytes")).toBe(
      String(96 * 1024 * 1024),
    );
    expect(() => validatePythonLaunchDescription(description)).not.toThrow();
  });

  it("adds both bounded VERIFY conveyor credits only on explicit linear opt-in", () => {
    const current = speculativeConveyorManifest();
    const historical = compile(current);
    const description = compile(current, {
      speculativeInflightWaves: 3,
      speculativeInflightBytes: 64 * 1024 * 1024,
    });
    const args = root(description).command.args;

    expect(description.schema).toBe("gdlp-python-launch/2");
    expect(description.launchId).not.toBe(historical.launchId);
    expect(description.configuration.speculativeInflightWaves).toBe(3);
    expect(description.configuration.speculativeInflightBytes).toBe(
      64 * 1024 * 1024,
    );
    expect(argumentValue(args, "--speculative-inflight-waves")).toBe("3");
    expect(argumentValue(args, "--speculative-inflight-bytes")).toBe(
      String(64 * 1024 * 1024),
    );
    for (const stage of remotes(description)) {
      expect(stage.command.args).not.toContain("--speculative-inflight-waves");
      expect(stage.command.args).not.toContain("--speculative-inflight-bytes");
    }
    expect(() => validatePythonLaunchDescription(description)).not.toThrow();
  });

  it("rejects incomplete, unsafe or incompatible VERIFY conveyor credits", () => {
    const current = speculativeConveyorManifest();
    expect(() =>
      compile(current, { speculativeInflightWaves: 0 }),
    ).toThrow("python_speculative_inflight_waves_is_invalid");
    expect(() =>
      compile(current, { speculativeInflightWaves: 17 }),
    ).toThrow("python_speculative_inflight_waves_is_invalid");
    expect(() =>
      compile(current, {
        speculativeInflightWaves: 2,
        speculativeInflightBytes: 1024 * 1024 * 1024 + 1,
      }),
    ).toThrow("python_speculative_inflight_bytes_is_invalid");
    expect(() =>
      compile(current, { speculativeInflightWaves: 2 }),
    ).toThrow("python_speculative_conveyor_limits_must_be_disabled_or_complete");
    expect(() =>
      compile(current, { speculativeInflightBytes: 4096 }),
    ).toThrow("python_speculative_conveyor_limits_must_be_disabled_or_complete");
    expect(() =>
      compile(manifest(), {
        speculativeInflightWaves: 2,
        speculativeInflightBytes: 4096,
      }),
    ).toThrow("python_speculative_conveyor_requires_linear_speculation");
    expect(() =>
      compile(speculativeConveyorManifest(2), {
        speculativeInflightWaves: 2,
        speculativeInflightBytes: 4096,
      }),
    ).toThrow("python_speculative_conveyor_requires_single_active_sequence");
    expect(() =>
      compile(current, {
        speculativeInflightWaves: 2,
        speculativeInflightBytes: 1,
      }),
    ).toThrow("python_speculative_inflight_bytes_below_frame_reservation");
    expect(() =>
      compile(current, {
        speculativeInflightWaves: 2,
        speculativeInflightBytes: 4096,
        maxSpeculativeBranches: 2,
        maxSpeculativeBranchTokens: 128,
        maxSpeculativeKvBytes: 4096,
      }),
    ).toThrow("python_speculative_conveyor_cannot_use_physical_tree_limits");
  });

  it("seals disabled or complete physical tree limits into every pipeline process", () => {
    const disabled = compile();
    expect(disabled.configuration.maxSpeculativeBranches).toBe(0);
    expect(disabled.configuration.maxSpeculativeBranchTokens).toBe(0);
    expect(disabled.configuration.maxSpeculativeKvBytes).toBe(0);
    for (const process of [root(disabled), ...remotes(disabled)]) {
      expect(argumentValue(process.command.args, "--max-speculative-branches")).toBe("0");
      expect(
        argumentValue(process.command.args, "--max-speculative-branch-tokens"),
      ).toBe("0");
      expect(argumentValue(process.command.args, "--max-speculative-kv-bytes")).toBe("0");
    }

    const enabled = compile(manifest(), {
      maxSpeculativeBranches: 8,
      maxSpeculativeBranchTokens: 32_768,
      maxSpeculativeKvBytes: 512 * 1024 * 1024,
    });
    expect(enabled.configuration.maxSpeculativeBranches).toBe(8);
    expect(enabled.configuration.maxSpeculativeBranchTokens).toBe(32_768);
    expect(enabled.configuration.maxSpeculativeKvBytes).toBe(512 * 1024 * 1024);
    for (const process of [root(enabled), ...remotes(enabled)]) {
      expect(argumentValue(process.command.args, "--max-speculative-branches")).toBe("8");
      expect(
        argumentValue(process.command.args, "--max-speculative-branch-tokens"),
      ).toBe("32768");
      expect(argumentValue(process.command.args, "--max-speculative-kv-bytes")).toBe(
        String(512 * 1024 * 1024),
      );
    }
    expect(() => validatePythonLaunchDescription(enabled)).not.toThrow();
  });

  it("rejects partial or unsafe physical tree limits", () => {
    expect(() => compile(manifest(), { maxSpeculativeBranches: 1 })).toThrow(
      "python_speculative_tree_limits_must_be_disabled_or_complete",
    );
    expect(() =>
      compile(manifest(), {
        maxSpeculativeBranches: 65,
        maxSpeculativeBranchTokens: 1,
        maxSpeculativeKvBytes: 1,
      }),
    ).toThrow("python_max_speculative_branches_is_invalid");
    expect(() =>
      compile(manifest(), {
        maxSpeculativeBranches: 1,
        maxSpeculativeBranchTokens: 1_048_577,
        maxSpeculativeKvBytes: 1,
      }),
    ).toThrow("python_max_speculative_branch_tokens_is_invalid");
    expect(() =>
      compile(manifest(), {
        maxSpeculativeBranches: 1,
        maxSpeculativeBranchTokens: 1,
        maxSpeculativeKvBytes: 2 ** 40 + 1,
      }),
    ).toThrow("python_max_speculative_kv_bytes_is_invalid");
  });

  it("rejects unsafe prefill pipeline credit bounds", () => {
    expect(() => compile(manifest(), { prefillInflightChunks: 0 })).toThrow(
      "python_prefill_inflight_chunks_is_invalid",
    );
    expect(() => compile(manifest(), { prefillInflightChunks: 65 })).toThrow(
      "python_prefill_inflight_chunks_is_invalid",
    );
    expect(() => compile(manifest(), { prefillInflightBytes: 0 })).toThrow(
      "python_prefill_inflight_bytes_is_invalid",
    );
    expect(() =>
      compile(manifest(), { prefillInflightBytes: 1024 * 1024 * 1024 + 1 }),
    ).toThrow("python_prefill_inflight_bytes_is_invalid");
  });

  it("fails closed unless one maximum prefill frame fits the byte credit", () => {
    const required = 32 + 8 * 512 * 2;
    expect(
      compile(manifest(), {
        prefillInflightChunks: 4,
        prefillInflightBytes: required,
      }).configuration.prefillInflightBytes,
    ).toBe(required);
    expect(() =>
      compile(manifest(), {
        prefillInflightChunks: 4,
        prefillInflightBytes: required - 1,
      }),
    ).toThrow(`python_prefill_inflight_bytes_below_frame_reservation:${required}`);
  });

  it("uses the sealed manifest codec and shape for the launch-time capacity check", () => {
    for (const codec of [
      "fp16",
      "int8",
      "int8-grouped",
      "int8-hadamard",
    ] as const) {
      const input = request();
      input.model.layers = input.model.layers.map((layer) => ({
        ...layer,
        activationElements: 511,
      }));
      input.allowLossyActivation = codec !== "fp16";
      input.phasePlans = {
        prefill: route(0, codec),
        decode: route(0, codec),
      };
      const current = buildRuntimePipelineManifest(input);
      const required = Number(
        pythonPrefillFrameByteReservation(
          codec,
          current.plans.prefill.chunkTokens,
          current.hiddenSize,
        ),
      );

      expect(
        compile(current, { prefillInflightBytes: required }).configuration
          .prefillInflightBytes,
      ).toBe(required);
      expect(() =>
        compile(current, { prefillInflightBytes: required - 1 }),
      ).toThrow(`python_prefill_inflight_bytes_below_frame_reservation:${required}`);
    }
  });

  it("matches the Python reservation formula for every runtime tensor codec", () => {
    const tokens = 8;
    const hidden = 511;
    const header = 32n;
    const elements = BigInt(tokens * hidden);
    const groupedBlocks = 8n;
    const hadamardBlocks = 13n;
    const groupedPayload = elements + BigInt(tokens) * groupedBlocks * 4n;
    const hadamardPayload = elements + BigInt(tokens) * hadamardBlocks * 4n;
    const deflateBound = (bytes: bigint): bigint =>
      bytes +
      (bytes >> 12n) +
      (bytes >> 14n) +
      (bytes >> 25n) +
      19n;

    expect(pythonPrefillFrameByteReservation("fp32", tokens, hidden)).toBe(
      header + elements * 4n,
    );
    expect(pythonPrefillFrameByteReservation("fp16", tokens, hidden)).toBe(
      header + elements * 2n,
    );
    expect(pythonPrefillFrameByteReservation("int8", tokens, hidden)).toBe(
      header + elements + 4n,
    );
    expect(
      pythonPrefillFrameByteReservation("int8-grouped", tokens, hidden),
    ).toBe(header + groupedPayload);
    expect(
      pythonPrefillFrameByteReservation("int8-hadamard", tokens, hidden),
    ).toBe(header + hadamardPayload);
    expect(
      pythonPrefillFrameByteReservation("int8-grouped-deflate", tokens, hidden),
    ).toBe(header + deflateBound(groupedPayload));
    expect(
      pythonPrefillFrameByteReservation("int8-hadamard-deflate", tokens, hidden),
    ).toBe(header + deflateBound(hadamardPayload));
  });

  it("connects remote stages only to their next remote anchor", () => {
    const stages = remotes(compile());
    expect(stages[0]!.stageIndex).toBe(2);
    expect(stages[0]!.downstream).toBeNull();
    expect(stages[0]!.command.args).not.toContain("--next-host");
    expect(stages[1]!.stageIndex).toBe(1);
    expect(stages[1]!.downstream?.stageIndex).toBe(2);
    expect(argumentValue(stages[1]!.command.args, "--next-host")).toBe(
      stages[0]!.anchor.endpoint.host,
    );
    expect(argumentValue(stages[1]!.command.args, "--next-layer-end")).toBe("6");
  });

  it("separates immutable model identity from the runtime model source", () => {
    const description = compile();
    expect(description.modelIdentity).toEqual({
      id: "launcher-model",
      revision: "sha256:launcher-model-r1",
      tokenizerId: "launcher-tokenizer-r1",
    });
    expect(description.runtimeModel).toEqual({ source: "launcher-model", revision: null });
    for (const process of description.launchOrder) {
      expect(argumentValue(process.command.args, "--model")).toBe("launcher-model");
      expect(process.command.args).not.toContain("--revision");
      expect(process.command.args).not.toContain("sha256:launcher-model-r1");
    }
  });

  it("accepts a local snapshot with null revision and an explicit HF revision", () => {
    const local = compile(manifest(), {
      runtimeModel: { source: "D:\\models\\snapshot", revision: null },
    });
    for (const process of local.launchOrder) {
      expect(argumentValue(process.command.args, "--model")).toBe("D:\\models\\snapshot");
      expect(process.command.args).not.toContain("--revision");
    }

    const hf = compile(manifest(), {
      runtimeModel: { source: "org/model", revision: "refs/pr/7" },
    });
    for (const process of hf.launchOrder) {
      expect(argumentValue(process.command.args, "--revision")).toBe("refs/pr/7");
    }
  });

  it("propagates one path-independent artifact identity to every Python model process", () => {
    const description = compile(manifest(), {
      runtimeModel: {
        source: "D:\\different-host-cache\\Qwen3",
        revision: null,
        snapshotIdentity: "12345",
      },
    });
    expect(description.runtimeModel).toEqual({
      source: "D:\\different-host-cache\\Qwen3",
      revision: null,
      snapshotIdentity: "12345",
      artifactIdentity: "snapshot:uint64:0000000000003039",
      canonicalSource:
        "content-addressed://snapshot:uint64:0000000000003039",
      canonicalRevision: null,
    });
    for (const process of description.launchOrder) {
      if (process.kind === "cell-member") continue;
      expect(argumentValue(process.command.args, "--model-artifact-identity")).toBe(
        "snapshot:uint64:0000000000003039",
      );
      expect(argumentValue(process.command.args, "--model-canonical-source")).toBe(
        "content-addressed://snapshot:uint64:0000000000003039",
      );
      expect(argumentValue(process.command.args, "--pipeline-snapshot-identity")).toBe(
        "12345",
      );
    }
  });

  it("accepts and propagates a full content identity for a local snapshot", () => {
    const snapshotIdentity = "12345";
    const artifactIdentity = strongArtifactIdentity(snapshotIdentity);
    const canonicalSource = `content-addressed://${artifactIdentity}`;
    const description = compile(manifest(), {
      runtimeModel: {
        source: "D:\\different-host-cache\\Qwen3",
        revision: null,
        snapshotIdentity,
        artifactIdentity,
        canonicalSource,
        canonicalRevision: artifactIdentity,
      },
    });

    expect(description.runtimeModel).toEqual({
      source: "D:\\different-host-cache\\Qwen3",
      revision: null,
      snapshotIdentity,
      artifactIdentity,
      canonicalSource,
      canonicalRevision: artifactIdentity,
    });
    for (const process of description.launchOrder) {
      if (process.kind === "cell-member") continue;
      expect(argumentValue(process.command.args, "--model-artifact-identity")).toBe(
        artifactIdentity,
      );
      expect(argumentValue(process.command.args, "--model-canonical-source")).toBe(
        canonicalSource,
      );
      expect(argumentValue(process.command.args, "--model-canonical-revision")).toBe(
        artifactIdentity,
      );
      expect(argumentValue(process.command.args, "--pipeline-snapshot-identity")).toBe(
        snapshotIdentity,
      );
    }
    expect(() =>
      validatePythonLaunchDescription(JSON.parse(JSON.stringify(description))),
    ).not.toThrow();
  });

  it("requires a complete strong coordinate set bound to its snapshot uint64", () => {
    const snapshotIdentity = "12345";
    const artifactIdentity = strongArtifactIdentity(snapshotIdentity);
    const base = {
      source: "D:\\models\\Qwen3",
      revision: null,
      snapshotIdentity,
      artifactIdentity,
      canonicalSource: `content-addressed://${artifactIdentity}`,
      canonicalRevision: artifactIdentity,
    };

    const missingRevision = { ...base };
    delete (missingRevision as Partial<typeof base>).canonicalRevision;
    expect(() => compile(manifest(), { runtimeModel: missingRevision })).toThrow(
      "python_runtime_model_artifact_coordinates_are_incomplete",
    );

    const missingSnapshot = { ...base };
    delete (missingSnapshot as Partial<typeof base>).snapshotIdentity;
    expect(() => compile(manifest(), { runtimeModel: missingSnapshot })).toThrow(
      "python_runtime_model_artifact_coordinates_are_incomplete",
    );

    expect(() =>
      compile(manifest(), {
        runtimeModel: { ...base, artifactIdentity: "snapshot:uint64:0000000000003039" },
      }),
    ).toThrow("python_runtime_model_artifact_identity_is_invalid");

    expect(() =>
      compile(manifest(), {
        runtimeModel: {
          ...base,
          artifactIdentity: strongArtifactIdentity("12346"),
        },
      }),
    ).toThrow("python_runtime_model_artifact_snapshot_identity_mismatch");
  });

  it("canonicalizes a Hub cache snapshot to repository and commit coordinates", () => {
    const commit = "c1899de289a04d12100db370d81485cdf75e47ca";
    const description = compile(manifest(), {
      runtimeModel: {
        source: `C:\\cache\\models--Qwen--Qwen3-0.6B\\snapshots\\${commit}`,
        revision: null,
      },
    });
    expect(description.runtimeModel.canonicalSource).toBe("hf://Qwen/Qwen3-0.6B");
    expect(description.runtimeModel.canonicalRevision).toBe(commit);
    expect(description.runtimeModel.artifactIdentity).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(description.runtimeModel.snapshotIdentity).toMatch(/^\d+$/);
  });

  it("accepts matching explicit Hub coordinates and rejects every contradiction", () => {
    const commit = "c1899de289a04d12100db370d81485cdf75e47ca";
    const source = `C:\\cache\\models--Qwen--Qwen3-0.6B\\snapshots\\${commit}`;
    const implicit = compile(manifest(), {
      runtimeModel: { source, revision: null },
    });
    const expected = implicit.runtimeModel;
    const explicitRuntimeModel = {
      source,
      revision: null,
      snapshotIdentity: expected.snapshotIdentity!,
      artifactIdentity: expected.artifactIdentity!,
      canonicalSource: expected.canonicalSource!,
      canonicalRevision: expected.canonicalRevision!,
    };

    expect(
      compile(manifest(), { runtimeModel: explicitRuntimeModel }),
    ).toEqual(implicit);

    const wrongArtifactIdentity = `${expected.artifactIdentity!.slice(0, -1)}${
      expected.artifactIdentity!.endsWith("0") ? "1" : "0"
    }`;
    expect(() =>
      compile(manifest(), {
        runtimeModel: {
          ...explicitRuntimeModel,
          artifactIdentity: wrongArtifactIdentity,
        },
      }),
    ).toThrow("python_runtime_model_artifact_identity_mismatch");
    expect(() =>
      compile(manifest(), {
        runtimeModel: {
          ...explicitRuntimeModel,
          canonicalSource: "hf://another/model",
        },
      }),
    ).toThrow("python_runtime_model_canonical_source_mismatch");
    expect(() =>
      compile(manifest(), {
        runtimeModel: {
          ...explicitRuntimeModel,
          canonicalRevision: "f".repeat(40),
        },
      }),
    ).toThrow("python_runtime_model_canonical_revision_mismatch");
  });

  it("seals a native Mycellios GGUF range with global model and pipeline identity", () => {
    const current = manifest();
    const target = current.plans.prefill.stages[2]!;
    const description = compile(current, nativeGgufOptions(current));
    const launch = remotes(description).find((stage) => stage.stageId === target.stageId)!;
    const binding = description.configuration.nativeGgufStages[target.stageId]!;

    expect(launch.nativeGguf).toEqual(binding);
    expect(Object.hasOwn(launch, "native_stage")).toBe(false);
    expect(launch.cell).toBeNull();
    expect(binding).toEqual(nativeGgufStage(current, 2));
    expect(argumentValue(launch.command.args, "--model")).toBe(
      description.runtimeModel.source,
    );
    expect(argumentValue(launch.command.args, "--model-artifact-identity")).toBe(
      binding.modelIdentity,
    );
    expect(argumentValue(launch.command.args, "--model-canonical-source")).toBe(
      binding.modelSource,
    );
    expect(argumentValue(launch.command.args, "--model-canonical-revision")).toBe(
      binding.modelRevision,
    );
    expect(argumentValue(launch.command.args, "--pipeline-snapshot-identity")).toBe(
      NATIVE_GGUF_SNAPSHOT_ID,
    );
    expect(argumentValue(launch.command.args, "--stage-package-identity")).toBe(
      `sha256:${binding.packageId}`,
    );
    expect(argumentValue(launch.command.args, "--native-gguf-package")).toBe(
      binding.packagePath,
    );
    expect(argumentValue(launch.command.args, "--native-gguf-package-id")).toBe(
      binding.packageId,
    );
    expect(launch.command.args).toContain("--device");
    expect(launch.command.args.some((argument) => argument.includes("native_stage"))).toBe(false);
    expect(launch.command.args.some((argument) => argument.includes("local-model-runtime"))).toBe(false);
    expect(() =>
      validatePythonLaunchDescription(JSON.parse(JSON.stringify(description))),
    ).not.toThrow();
  });

  it("launches a fully native GGUF fleet across root and stages [0,1,2]", () => {
    const current = manifest();
    const description = compile(current, nativeGgufOptions(current, [0, 1, 2]));
    const stages = current.plans.prefill.stages;
    const rootLaunch = root(description);
    const rootBinding =
      description.configuration.nativeGgufStages[stages[0]!.stageId]!;

    expect(Object.keys(description.configuration.nativeGgufStages).sort()).toEqual(
      stages.map((stage) => stage.stageId).sort(),
    );
    expect(rootLaunch.nativeGguf).toEqual(rootBinding);
    expect(remotes(description).map((stage) => stage.nativeGguf?.packageId)).toEqual([
      "c".repeat(64),
      "c".repeat(64),
    ]);
    expect(argumentValue(rootLaunch.command.args, "--model-artifact-identity")).toBe(
      rootBinding.modelIdentity,
    );
    expect(argumentValue(rootLaunch.command.args, "--model-canonical-source")).toBe(
      rootBinding.modelSource,
    );
    expect(argumentValue(rootLaunch.command.args, "--model-canonical-revision")).toBe(
      rootBinding.modelRevision,
    );
    expect(
      argumentValue(rootLaunch.command.args, "--pipeline-snapshot-identity"),
    ).toBe(NATIVE_GGUF_SNAPSHOT_ID);
    expect(argumentValue(rootLaunch.command.args, "--stage-package-identity")).toBe(
      `sha256:${rootBinding.packageId}`,
    );
    expect(argumentValue(rootLaunch.command.args, "--native-gguf-package")).toBe(
      rootBinding.packagePath,
    );
    expect(argumentValue(rootLaunch.command.args, "--native-gguf-package-id")).toBe(
      rootBinding.packageId,
    );
    expect(rootLaunch.command.args.some((argument) => argument.includes("native_stage"))).toBe(
      false,
    );
    expect(() =>
      validatePythonLaunchDescription(JSON.parse(JSON.stringify(description))),
    ).not.toThrow();

    const metadataTampering = compile(
      current,
      nativeGgufOptions(current, [0, 1, 2]),
    );
    root(metadataTampering).nativeGguf!.packagePath += "-tampered";
    expect(() => validatePythonLaunchDescription(metadataTampering)).toThrow(
      "python_launch_description_mismatch",
    );

    const argvTampering = compile(current, nativeGgufOptions(current, [0, 1, 2]));
    const rootArgs = root(argvTampering).command.args;
    rootArgs[rootArgs.indexOf("--native-gguf-package-id") + 1] = "e".repeat(64);
    expect(() => validatePythonLaunchDescription(argvTampering)).toThrow(
      "python_launch_description_mismatch",
    );
  });

  it("sorts and authenticates native GGUF bindings in every derived launch identity", () => {
    const current = manifest();
    const forward = nativeGgufOptions(current, [1, 2]);
    const reversed = {
      ...forward,
      nativeGgufStages: Object.fromEntries(
        Object.entries(forward.nativeGgufStages!).reverse(),
      ),
    };
    expect(compile(current, reversed)).toEqual(compile(current, forward));

    const configuration = compile(current, forward);
    const stageId = current.plans.prefill.stages[2]!.stageId;
    configuration.configuration.nativeGgufStages[stageId]!.packagePath += "-tampered";
    expect(() => validatePythonLaunchDescription(configuration)).toThrow(
      "python_launch_description_mismatch",
    );

    const argv = compile(current, forward);
    const args = remotes(argv).find((stage) => stage.stageId === stageId)!.command.args;
    args[args.indexOf("--native-gguf-package-id") + 1] = "d".repeat(64);
    expect(() => validatePythonLaunchDescription(argv)).toThrow(
      "python_launch_description_mismatch",
    );
  });

  it("rejects unbound, conflicting or incorrectly ranged native GGUF stages", () => {
    const current = manifest();
    const target = current.plans.prefill.stages[2]!;

    const noPipeline = nativeGgufOptions(current);
    noPipeline.runtimeModel = {
      source: noPipeline.runtimeModel!.source,
      revision: null,
    };
    expect(() => compile(current, noPipeline)).toThrow(
      "python_native_gguf_requires_pipeline_snapshot_identity",
    );

    const wrongIdentity = nativeGgufOptions(current);
    wrongIdentity.nativeGgufStages![target.stageId]!.modelIdentity =
      `sha256:${"f".repeat(64)}`;
    expect(() => compile(current, wrongIdentity)).toThrow(
      `python_native_gguf_model_identity_mismatch:${target.stageId}`,
    );

    const wrongSource = nativeGgufOptions(current);
    wrongSource.nativeGgufStages![target.stageId]!.modelSource += "-other";
    expect(() => compile(current, wrongSource)).toThrow(
      `python_native_gguf_model_source_mismatch:${target.stageId}`,
    );

    const wrongRange = nativeGgufOptions(current);
    wrongRange.nativeGgufStages![target.stageId]!.layerStart -= 1;
    expect(() => compile(current, wrongRange)).toThrow(
      `python_native_gguf_stage_range_mismatch:${target.stageId}`,
    );

    const conflicting: LegacyNativeStageOptions = {
      ...nativeGgufOptions(current),
      native_stageStages: {
        [target.stageId]: native_stageStage(current, 2, {
          modelSource: nativeGgufOptions(current).runtimeModel!.source,
          modelRevision: null,
          pipelineId: NATIVE_GGUF_SNAPSHOT_ID,
        }),
      },
    };
    expect(() => compile(current, conflicting)).toThrow(
      "python_launch_options_have_unknown_or_missing_fields",
    );

    const cell = certifyTensorParallelCell(
      materializeTensorParallelCell(manifest()),
    );
    const cellStageId = cell.plans.prefill.stages[1]!.stageId;
    expect(() =>
      compile(cell, {
        ...nativeGgufOptions(cell, [1]),
      })).toThrow(
      `python_native_gguf_stage_has_conflicting_execution:${cellStageId}`,
    );
  });

  it("rejects NativeStage before a production launch description can be built", () => {
    const current = manifest();
    expect(() => compile(current, native_stageOptions(current))).toThrow(
      "python_launch_options_have_unknown_or_missing_fields",
    );
  });

  it.skip("research fixture: binds one sealed NativeStage package to an exact non-root stage", () => {
    const current = manifest();
    const target = current.plans.prefill.stages[2]!;
    const description = compile(current, native_stageOptions(current));
    const launch = remotes(description).find((stage) => stage.stageId === target.stageId)!;
    const normalized = legacyNativeStageBindings(description)[target.stageId]!;

    expect(legacyNativeStageLaunch(launch)).toEqual(normalized);
    expect(launch.cell).toBeNull();
    expect(normalized.modelIdentity).toBe(description.runtimeModel.artifactIdentity);
    expect(argumentValue(launch.command.args, "--model")).toBe(
      "hf://HuggingFaceTB/SmolLM2-135M-Instruct",
    );
    expect(argumentValue(launch.command.args, "--revision")).toBe(NATIVE_STAGE_TEST_COMMIT);
    expect(launch.command.args).not.toContain("--model-artifact-identity");
    expect(launch.command.args).not.toContain("--pipeline-snapshot-identity");
    expect(argumentValue(launch.command.args, "--native_stage-package")).toBe(
      normalized.packagePath,
    );
    expect(argumentValue(launch.command.args, "--native_stage-package-id")).toBe(
      normalized.packageId,
    );
    expect(argumentValue(launch.command.args, "--native_stage-manifest-sha256")).toBe(
      normalized.manifestSha256,
    );
    expect(argumentValue(launch.command.args, "--native_stage-daemon-bin")).toBe(
      normalized.daemonExecutable,
    );
    expect(argumentValue(launch.command.args, "--native_stage-pipeline-id")).toBe(
      normalized.pipelineId,
    );
    expect(argumentValue(launch.command.args, "--native_stage-context-tokens")).toBe("4096");
    expect(argumentValue(launch.command.args, "--native_stage-gpu-layers")).toBe("99");
    expect(argumentValue(launch.command.args, "--native_stage-compute-api")).toBe("cuda");
    expect(argumentValue(launch.command.args, "--native_stage-startup-timeout-seconds")).toBe(
      "90",
    );
    expect(argumentValue(launch.command.args, "--native_stage-call-timeout-seconds")).toBe(
      "30.5",
    );
    expect(argumentValue(launch.command.args, "--native_stage-close-timeout-seconds")).toBe(
      "5",
    );
    expect(argumentValue(root(description).command.args, "--model")).toContain(
      "models--HuggingFaceTB--SmolLM2-135M-Instruct",
    );
    expect(() =>
      validatePythonLaunchDescription(JSON.parse(JSON.stringify(description))),
    ).not.toThrow();
  });

  it.skip("research fixture: binds a local non-Hub NativeStage snapshot", () => {
    const current = manifest();
    const target = current.plans.prefill.stages[2]!;
    const localSource = "D:\\models\\SmolLM2-local-snapshot";
    const snapshotIdentity = "12345";
    const options = native_stageOptions(current);
    options.runtimeModel = {
      source: localSource,
      revision: null,
      snapshotIdentity,
    };
    options.native_stageStages![target.stageId] = native_stageStage(current, 2, {
      modelSource: localSource,
      modelRevision: null,
      pipelineId: snapshotIdentity,
    });

    const description = compile(current, options);
    const normalized = legacyNativeStageBindings(description)[target.stageId]!;
    const launch = remotes(description).find((stage) => stage.stageId === target.stageId)!;

    expect(description.runtimeModel.artifactIdentity).toBe(
      "snapshot:uint64:0000000000003039",
    );
    expect(normalized.modelIdentity).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(normalized.modelIdentity).not.toBe(description.runtimeModel.artifactIdentity);
    expect(normalized.pipelineId).toBe(snapshotIdentity);
    expect(argumentValue(launch.command.args, "--model")).toBe(localSource);
    expect(launch.command.args).not.toContain("--revision");
    expect(() =>
      validatePythonLaunchDescription(JSON.parse(JSON.stringify(description))),
    ).not.toThrow();

    const strongArtifact = strongArtifactIdentity(snapshotIdentity);
    const strongOptions = structuredClone(options);
    strongOptions.runtimeModel = {
      ...strongOptions.runtimeModel!,
      artifactIdentity: strongArtifact,
      canonicalSource: `content-addressed://${strongArtifact}`,
      canonicalRevision: strongArtifact,
    };
    const strongDescription = compile(current, strongOptions);
    expect(strongDescription.runtimeModel.artifactIdentity).toBe(strongArtifact);
    expect(() => validatePythonLaunchDescription(strongDescription)).not.toThrow();

    const wrongContent = structuredClone(options);
    wrongContent.native_stageStages![target.stageId]!.pipelineId = "12346";
    expect(() => compile(current, wrongContent)).toThrow(
      `python_native_stage_pipeline_identity_mismatch:${target.stageId}`,
    );

    const wrongCoordinates = structuredClone(options);
    wrongCoordinates.native_stageStages![target.stageId]!.modelIdentity =
      `sha256:${"f".repeat(64)}`;
    expect(() => compile(current, wrongCoordinates)).toThrow(
      `python_native_stage_model_identity_mismatch:${target.stageId}`,
    );
  });

  it.skip("research fixture: sorts multiple NativeStage stage bindings", () => {
    const current = manifest();
    const forward = native_stageOptions(current, [1, 2]);
    const reversedEntries = Object.entries(forward.native_stageStages!).reverse();
    const reversed = {
      ...forward,
      native_stageStages: Object.fromEntries(reversedEntries),
    };
    expect(compile(current, reversed)).toEqual(compile(current, forward));
  });

  it.skip("research fixture: detects NativeStage metadata tampering", () => {
    const current = manifest();
    const stageId = current.plans.prefill.stages[2]!.stageId;

    const configuration = compile(current, native_stageOptions(current));
    legacyNativeStageBindings(configuration)[stageId]!.packagePath += "-tampered";
    expect(() => validatePythonLaunchDescription(configuration)).toThrow(
      "python_launch_description_mismatch",
    );

    const process = compile(current, native_stageOptions(current));
    legacyNativeStageLaunch(
      remotes(process).find((stage) => stage.stageId === stageId)!,
    )!.packageId = "c".repeat(64);
    expect(() => validatePythonLaunchDescription(process)).toThrow(
      "python_launch_description_mismatch",
    );

    const argv = compile(current, native_stageOptions(current));
    const args = remotes(argv).find((stage) => stage.stageId === stageId)!.command.args;
    args[args.indexOf("--native_stage-context-tokens") + 1] = "8192";
    expect(() => validatePythonLaunchDescription(argv)).toThrow(
      "python_launch_description_mismatch",
    );
  });

  it.skip("research fixture: validates legacy NativeStage bindings", () => {
    const current = manifest();
    const rootId = current.plans.prefill.stages[0]!.stageId;
    expect(() => compile(current, native_stageOptions(current, [0]))).toThrow(
      `python_native_stage_stage_cannot_be_root:${rootId}`,
    );

    const unknown = native_stageOptions(current);
    unknown.native_stageStages = {
      "stage-does-not-exist": native_stageStage(current, 2),
    };
    expect(() => compile(current, unknown)).toThrow(
      "python_native_stage_stage_is_not_in_both_phases:stage-does-not-exist",
    );

    const stageId = current.plans.prefill.stages[2]!.stageId;
    const range = native_stageOptions(current);
    range.native_stageStages![stageId]!.layerStart -= 1;
    expect(() => compile(current, range)).toThrow(
      `python_native_stage_stage_range_mismatch:${stageId}`,
    );

    const pipeline = native_stageOptions(current);
    pipeline.native_stageStages![stageId]!.pipelineId = "1";
    expect(() => compile(current, pipeline)).toThrow(
      `python_native_stage_pipeline_identity_mismatch:${stageId}`,
    );

    const compute = native_stageOptions(current);
    compute.native_stageStages![stageId]!.computeApi = "cpu";
    expect(() => compile(current, compute)).toThrow(
      `python_native_stage_compute_api_gpu_layers_mismatch:${stageId}`,
    );

    const missingIdentity = native_stageOptions(current);
    delete (missingIdentity.native_stageStages![stageId] as Partial<LegacyNativeStageStageInput>)
      .packageId;
    expect(() => compile(current, missingIdentity)).toThrow(
      `python_native_stage_stage_configuration_keys_are_invalid:${stageId}`,
    );

    const cell = certifyTensorParallelCell(
      materializeTensorParallelCell(manifest()),
    );
    const cellStageId = cell.plans.prefill.stages[1]!.stageId;
    expect(() =>
      compile(cell, {
        ...native_stageOptions(cell, [1]),
      })).toThrow(
      `python_native_stage_stage_cannot_use_cell_execution:${cellStageId}`,
    );
  });

  it("maps adaptive ngram speculation to the Python server", () => {
    const policy: RuntimeSpeculationPolicy = {
      mode: "adaptive",
      controller: "acceptance-adaptive",
      defaultStrategyId: "ngram",
      fallbackStrategyId: "autoregressive",
      acceptanceWindowTokens: 64,
      strategies: [
        {
          id: "ngram",
          kind: "ngram",
          maxDraftTokens: 5,
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
    };
    const input = request();
    input.speculation = policy;
    const current = buildRuntimePipelineManifest(input);
    const args = root(compile(current)).command.args;
    expect(argumentValue(args, "--speculation")).toBe("ngram");
    expect(argumentValue(args, "--speculative-max-draft-tokens")).toBe("5");
    expect(argumentValue(args, "--sealed-wave-tokens")).toBe("6");
    expect(argumentValue(args, "--max-prefill-chunk-tokens")).toBe("8");
    for (const stage of remotes(compile(current))) {
      expect(argumentValue(stage.command.args, "--sealed-wave-tokens")).toBe("6");
      expect(argumentValue(stage.command.args, "--max-prefill-chunk-tokens")).toBe("8");
    }
  });

  it("maps a sealed sibling drafter only to the native root process", () => {
    const snapshotIdentity = "54321";
    const artifactIdentity = strongArtifactIdentity(snapshotIdentity);
    const input = request();
    input.speculation = {
      mode: "adaptive",
      controller: "acceptance-adaptive",
      defaultStrategyId: "local-sibling",
      fallbackStrategyId: "autoregressive",
      acceptanceWindowTokens: 64,
      strategies: [
        {
          id: "local-sibling",
          kind: "draft-model",
          maxDraftTokens: 4,
          minAcceptanceRate: 0.6,
          maxWasteRatio: 0.3,
          priority: 10,
          artifactId: artifactIdentity,
          parameterBytes: 16,
          memoryReservationBytes: 16,
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
    const current = buildRuntimePipelineManifest(input);
    const draftModel = {
      schema: "mycellios-local-draft-model/1" as const,
      source: "D:\\models\\local-sibling",
      revision: null,
      snapshotIdentity,
      artifactIdentity,
      canonicalSource: `content-addressed://${artifactIdentity}`,
      canonicalRevision: null,
      device: "cpu" as const,
      dtype: "float32" as const,
    };
    const description = compile(current, { draftModel });
    const args = root(description).command.args;

    expect(argumentValue(args, "--speculation")).toBe("draft-model");
    expect(argumentValue(args, "--speculative-max-draft-tokens")).toBe("4");
    expect(argumentValue(args, "--draft-model-source")).toBe(draftModel.source);
    expect(argumentValue(args, "--draft-model-artifact-identity")).toBe(
      artifactIdentity,
    );
    expect(argumentValue(args, "--draft-model-canonical-source")).toBe(
      draftModel.canonicalSource,
    );
    expect(argumentValue(args, "--draft-model-device")).toBe("cpu");
    expect(argumentValue(args, "--draft-model-dtype")).toBe("float32");
    for (const stage of remotes(description)) {
      expect(stage.command.args).not.toContain("--draft-model-source");
    }
    expect(description.configuration.draftModel).toEqual({
      ...draftModel,
      parameterBytes: 16,
      memoryReservationBytes: 16,
    });
    expect(() => validatePythonLaunchDescription(description)).not.toThrow();

    const otherPlacement = compile(current, {
      draftModel: { ...draftModel, device: "cuda:0", dtype: "float16" },
    });
    expect(root(otherPlacement).routeId).toBe(root(description).routeId);
    expect(root(otherPlacement).processId).not.toBe(root(description).processId);
    expect(remotes(otherPlacement).map((stage) => stage.processId)).toEqual(
      remotes(description).map((stage) => stage.processId),
    );
  });

  it("fails closed when a draft-model strategy is missing or mismatches its model", () => {
    const snapshotIdentity = "54321";
    const artifactIdentity = strongArtifactIdentity(snapshotIdentity);
    const input = request();
    input.speculation = {
      mode: "adaptive",
      controller: "acceptance-adaptive",
      defaultStrategyId: "local-sibling",
      fallbackStrategyId: "autoregressive",
      acceptanceWindowTokens: 64,
      strategies: [
        {
          id: "local-sibling",
          kind: "draft-model",
          maxDraftTokens: 4,
          minAcceptanceRate: 0.6,
          maxWasteRatio: 0.3,
          priority: 10,
          artifactId: artifactIdentity,
          parameterBytes: 16,
          memoryReservationBytes: 16,
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
    const current = buildRuntimePipelineManifest(input);
    expect(() => compile(current)).toThrow(
      "python_draft_model_configuration_is_missing",
    );
    expect(() =>
      compile(current, {
        draftModel: {
          schema: "mycellios-local-draft-model/1",
          source: "D:\\models\\local-sibling",
          revision: null,
          snapshotIdentity,
          artifactIdentity: strongArtifactIdentity("54322"),
          canonicalSource: `content-addressed://${strongArtifactIdentity("54322")}`,
          canonicalRevision: null,
          device: "cpu",
          dtype: "float32",
        },
      }),
    ).toThrow();

    const ngram = manifest();
    expect(() =>
      compile(ngram, {
        draftModel: {
          schema: "mycellios-local-draft-model/1",
          source: "D:\\models\\unexpected",
          revision: null,
          snapshotIdentity,
          artifactIdentity,
          canonicalSource: `content-addressed://${artifactIdentity}`,
          canonicalRevision: null,
          device: "cpu",
          dtype: "float32",
        },
      }),
    ).toThrow("python_draft_model_requires_draft_model_strategy");
  });

  it("maps only manifest-sealed draft-tree limits to every native Python process", () => {
    const input = request();
    input.speculation = {
      mode: "adaptive",
      controller: "acceptance-adaptive",
      defaultStrategyId: "native-tree",
      fallbackStrategyId: "autoregressive",
      acceptanceWindowTokens: 64,
      strategies: [
        {
          id: "native-tree",
          kind: "draft-tree",
          maxDraftTokens: 4,
          maxBranches: 6,
          maxBranchTokens: 32_768,
          maxKvBytes: 384 * MIB,
          maxWaveTokens: 5,
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
    };
    const current = buildRuntimePipelineManifest(input);
    const description = compile(current);
    expect(description.configuration.maxSpeculativeBranches).toBe(6);
    expect(description.configuration.maxSpeculativeBranchTokens).toBe(32_768);
    expect(description.configuration.maxSpeculativeKvBytes).toBe(384 * MIB);
    for (const process of [root(description), ...remotes(description)]) {
      expect(argumentValue(process.command.args, "--max-speculative-branches")).toBe("6");
      expect(
        argumentValue(process.command.args, "--max-speculative-branch-tokens"),
      ).toBe("32768");
      expect(argumentValue(process.command.args, "--max-speculative-kv-bytes")).toBe(
        String(384 * MIB),
      );
      expect(argumentValue(process.command.args, "--sealed-wave-tokens")).toBe("5");
    }
    expect(argumentValue(root(description).command.args, "--speculation")).toBe(
      "draft-tree",
    );
    expect(
      argumentValue(
        root(description).command.args,
        "--speculative-max-draft-tokens",
      ),
    ).toBe("4");
    expect(() => validatePythonLaunchDescription(description)).not.toThrow();

    expect(() =>
      compile(current, {
        maxSpeculativeBranches: 5,
        maxSpeculativeBranchTokens: 32_768,
        maxSpeculativeKvBytes: 384 * MIB,
      }),
    ).toThrow("python_draft_tree_launch_limits_do_not_match_manifest");

    const tampered = structuredClone(description);
    tampered.sourceManifest.plans.decode.speculation.strategies[0]!.maxBranches = 5;
    expect(() => validatePythonLaunchDescription(tampered)).toThrow();
  });

  it("rejects draft-tree when any physical limit is absent or inconsistent", () => {
    const input = request();
    input.speculation = {
      mode: "adaptive",
      controller: "acceptance-adaptive",
      defaultStrategyId: "native-tree",
      fallbackStrategyId: "autoregressive",
      acceptanceWindowTokens: 64,
      strategies: [
        {
          id: "native-tree",
          kind: "draft-tree",
          maxDraftTokens: 4,
          maxBranches: 4,
          maxBranchTokens: 8_192,
          maxKvBytes: 64 * MIB,
          maxWaveTokens: 5,
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
    };
    const missing = structuredClone(input);
    delete missing.speculation!.strategies[0]!.maxKvBytes;
    expect(() => buildRuntimePipelineManifest(missing)).toThrow(
      "runtime_draft_tree_limits_are_missing",
    );

    const inconsistent = structuredClone(input);
    inconsistent.speculation!.strategies[0]!.maxWaveTokens = 6;
    expect(() => buildRuntimePipelineManifest(inconsistent)).toThrow(
      "runtime_draft_tree_wave_does_not_match_draft_depth",
    );
  });

  it("rejects a speculative provider the Python server cannot execute", () => {
    const input = request();
    input.speculation = {
      mode: "adaptive",
      controller: "acceptance-adaptive",
      defaultStrategyId: "mtp",
      fallbackStrategyId: "autoregressive",
      acceptanceWindowTokens: 64,
      strategies: [
        {
          id: "mtp",
          kind: "mtp",
          maxDraftTokens: 3,
          minAcceptanceRate: 0.6,
          maxWasteRatio: 0.3,
          priority: 10,
          artifactId: `sha256:${"b".repeat(64)}`,
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
    const current = buildRuntimePipelineManifest(input);
    expect(() => compile(current)).toThrow("python_speculation_strategy_not_supported:mtp");
  });

  it("enforces the Python speculation draft-length limit at compile time", () => {
    const input = request();
    input.speculation = {
      mode: "adaptive",
      controller: "acceptance-adaptive",
      defaultStrategyId: "ngram",
      fallbackStrategyId: "autoregressive",
      acceptanceWindowTokens: 64,
      strategies: [
        {
          id: "ngram",
          kind: "ngram",
          maxDraftTokens: 17,
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
    };
    const current = buildRuntimePipelineManifest(input);
    expect(() => compile(current)).toThrow(
      "python_speculation_draft_length_exceeds_runtime_limit",
    );
  });

  it("rejects phase routes that one root engine cannot switch between", () => {
    const input = request();
    input.phasePlans = { prefill: route(0), decode: route(3) };
    expect(() => compile(buildRuntimePipelineManifest(input))).toThrow(
      "python_server_requires_shared_prefill_decode_route",
    );
  });

  it("rejects a shared route with different codecs or non-in-place KV", () => {
    const input = request();
    input.allowLossyActivation = true;
    input.phasePlans = {
      prefill: route(0, "fp16"),
      decode: route(0, "int8-grouped"),
    };
    expect(() => compile(buildRuntimePipelineManifest(input))).toThrow(
      "python_shared_route_requires_one_codec",
    );

    const recompute = request();
    recompute.kv = { mode: "recompute" };
    const current = buildRuntimePipelineManifest(recompute);
    expect(() => compile(current)).toThrow("python_server_requires_in_place_kv");
  });

  it("rejects a one-stage plan because the distributed engine requires a remote stage", () => {
    expect(() => compile(oneStageManifest())).toThrow(
      "python_distributed_runtime_requires_two_stages",
    );
  });

  it("rejects an unsealed cooperative-stage mutation", () => {
    const current = manifest();
    addCooperativeMember(current.plans.prefill.stages[1]!);
    addCooperativeMember(current.plans.decode.stages[1]!);
    expect(() => compile(current)).toThrow("runtime_stage_id_mismatch:prefill:1");
  });

  it("compiles one materialized intermediate TP cell to the exact stage_cli flags", () => {
    const current = materializeTensorParallelCell(manifest());

    expect(() => compile(current)).toThrow(
      "python_tensor_parallel_requires_measured_collective_profile",
    );
    const certified = certifyTensorParallelCell(current);
    const description = compile(certified);
    const cell = remotes(description).find((stage) => stage.stageIndex === 1)!;
    expect(cell.members).toHaveLength(2);
    expect(cell.cell).toEqual(certified.plans.decode.stages[1]!.execution);
    expect(argumentValue(cell.command.args, "--cell-fixture")).toBe(
      certified.plans.decode.stages[1]!.execution!.fixture.path,
    );
    expect(argumentValue(cell.command.args, "--cell-world-size")).toBe("2");
    expect(argumentValue(cell.command.args, "--cell-manifest-sha256")).toBe(
      "a".repeat(64),
    );
    expect(argumentValue(cell.command.args, "--cell-collective-backend")).toBe("gloo");
    expect(argumentValue(cell.command.args, "--cell-compute-dtype")).toBe("float32");
    expect(argumentValues(cell.command.args, "--cell-device")).toEqual(["cpu", "cpu"]);
    expect(argumentValue(cell.command.args, "--cell-operation-timeout-seconds")).toBe(
      "12.5",
    );
    expect(remotes(description).find((stage) => stage.stageIndex === 2)!.cell).toBeNull();
    expect(root(description).command.args).not.toContain("--cell-fixture");
    expect(() => validatePythonLaunchDescription(description)).not.toThrow();
  });

  it("launches member-local ranks before their external cell anchor", () => {
    const current = certifyTensorParallelCell(
      materializeExternalTensorParallelCell(manifest()),
    );
    const description = compile(current, {
      runtimeModel: {
        source: "C:/models/launcher-model",
        revision: null,
        snapshotIdentity: "12345",
      },
    });
    const member = cellMembers(description)[0]!;
    const anchor = remotes(description).find((stage) => stage.stageIndex === 1)!;

    expect(description.launchOrder.map((process) => process.kind)).toEqual([
      "remote-stage",
      "cell-member",
      "remote-stage",
      "root-engine",
    ]);
    expect(member.launchIndex).toBeLessThan(anchor.launchIndex);
    expect(member.anchor.memberId).toBe("node-1-peer");
    expect(moduleName(member.command.args)).toBe("distributed_runtime.cell_member_cli");
    expect(argumentValue(member.command.args, "--rank")).toBe("1");
    expect(argumentValue(member.command.args, "--pipeline-id")).toBe("12345");
    expect(argumentValue(member.command.args, "--fixture")).toBe(
      "D:/gdlp/cells/stage-1",
    );
    expect(argumentValue(member.command.args, "--control-host")).toBe("10.20.0.10");
    expect(argumentValue(member.command.args, "--cell-collective-backend")).toBe("gloo");
    expect(argumentValue(member.command.args, "--cell-compute-dtype")).toBe("float32");
    expect(argumentValue(member.command.args, "--cell-device")).toBe("cpu");
    expect(argumentValue(anchor.command.args, "--cell-mode")).toBe("external");
    expect(argumentValue(anchor.command.args, "--cell-control-port")).toBe("29100");
    expect(argumentValue(anchor.command.args, "--cell-distributed-port")).toBe("29101");
    expect(description.route.cellMemberProcessIds).toEqual([member.processId]);
    expect(() => validatePythonLaunchDescription(description)).not.toThrow();
  });

  it("propagates the NCCL dtype and every rank device to the anchor stage", () => {
    const current = certifyTensorParallelCell(
      materializeTensorParallelCell(manifest(), 1, 0, {
        computeDtype: "bfloat16",
        computeApi: "rocm",
        rankDevices: ["cuda:0", "cuda:3"],
      }),
    );
    const description = compile(current);
    const anchor = remotes(description).find((stage) => stage.stageIndex === 1)!;

    expect(argumentValue(anchor.command.args, "--cell-collective-backend")).toBe("nccl");
    expect(argumentValue(anchor.command.args, "--cell-compute-dtype")).toBe("bfloat16");
    expect(argumentValues(anchor.command.args, "--cell-device")).toEqual([
      "cuda:0",
      "cuda:3",
    ]);
    expect(() => validatePythonLaunchDescription(description)).not.toThrow();
  });

  it("passes only the selected NCCL rank device to each external member CLI", () => {
    const current = certifyTensorParallelCell(
      materializeExternalTensorParallelCell(manifest(), {
        computeDtype: "float16",
        rankDevices: ["cuda:0", "cuda:7"],
      }),
    );
    const description = compile(current, {
      runtimeModel: {
        source: "C:/models/launcher-model",
        revision: null,
        snapshotIdentity: "12345",
      },
    });
    const member = cellMembers(description)[0]!;
    const anchor = remotes(description).find((stage) => stage.stageIndex === 1)!;

    expect(member.rank).toBe(1);
    expect(member.device).toBe("cuda:7");
    expect(argumentValue(member.command.args, "--cell-collective-backend")).toBe("nccl");
    expect(argumentValue(member.command.args, "--cell-compute-dtype")).toBe("float16");
    expect(argumentValue(member.command.args, "--cell-device")).toBe("cuda:7");
    expect(argumentValues(anchor.command.args, "--cell-device")).toEqual([
      "cuda:0",
      "cuda:7",
    ]);
    expect(() => validatePythonLaunchDescription(description)).not.toThrow();
  });

  it("requires the Python snapshot uint64 before orchestrating external ranks", () => {
    const current = certifyTensorParallelCell(
      materializeExternalTensorParallelCell(manifest()),
    );
    expect(() => compile(current)).toThrow(
      "python_external_cell_requires_snapshot_identity",
    );
  });

  it("materializes by cloning and deterministically reseals stage, plan and pipeline IDs", () => {
    const singleton = manifest();
    const first = materializeTensorParallelCell(singleton);
    const second = materializeTensorParallelCell(singleton);

    expect(first).toEqual(second);
    expect(JSON.stringify(first)).toBe(JSON.stringify(second));
    expect(singleton.plans.prefill.stages[1]!.members).toHaveLength(1);
    expect(singleton.plans.prefill.stages[1]!.execution).toBeUndefined();
    expect(first.plans.prefill.stages[1]!.stageId).not.toBe(
      singleton.plans.prefill.stages[1]!.stageId,
    );
    expect(first.plans.prefill.planId).not.toBe(singleton.plans.prefill.planId);
    expect(first.plans.decode.planId).not.toBe(singleton.plans.decode.planId);
    expect(first.pipelineId).not.toBe(singleton.pipelineId);
    expect(() => validateRuntimePipelineManifest(first)).not.toThrow();
  });

  it("uses aggregate member capacity instead of preserving the singleton limit", () => {
    const singleton = manifest();
    const singletonLimit = singleton.plans.prefill.stages[1]!.memoryLimitBytes;
    const addedCapacity = 32 * MIB;
    const cell = materializeTensorParallelCell(singleton, 1, addedCapacity);

    for (const plan of [cell.plans.prefill, cell.plans.decode]) {
      expect(plan.stages[1]!.memoryLimitBytes).toBe(singletonLimit + addedCapacity);
      expect(plan.predicted.stageMetrics[1]!.memoryLimitBytes).toBe(
        singletonLimit + addedCapacity,
      );
    }
    expect(() => validateRuntimePipelineManifest(cell)).not.toThrow();
  });

  it("rejects stale stage, plan and pipeline IDs after tampering", () => {
    const sealed = materializeTensorParallelCell(manifest());

    const stage = structuredClone(sealed);
    for (const plan of [stage.plans.prefill, stage.plans.decode]) {
      plan.stages[1]!.execution!.fixture.path += "-tampered";
    }
    expect(() => validateRuntimePipelineManifest(stage)).toThrow(
      "runtime_stage_id_mismatch:prefill:1",
    );

    const plan = structuredClone(sealed);
    plan.plans.prefill.planId = "stale-prefill-plan";
    expect(() => validateRuntimePipelineManifest(plan)).toThrow(
      "runtime_plan_id_mismatch:prefill",
    );

    const pipeline = structuredClone(sealed);
    pipeline.pipelineId = "stale-pipeline";
    expect(() => validateRuntimePipelineManifest(pipeline)).toThrow(
      "runtime_pipeline_id_mismatch",
    );
  });

  it("fails closed for cell execution on root/final stages", () => {
    for (const stageIndex of [0, 2]) {
      expect(() => materializeTensorParallelCell(manifest(), stageIndex)).toThrow(
        "runtime_cell_materialization_stage_must_be_intermediate",
      );
    }
  });

  it("fails closed for inconsistent cell backends, locality and world size", () => {
    const backend = materializeTensorParallelCell(manifest());
    for (const phase of [backend.plans.prefill, backend.plans.decode]) {
      phase.stages[1]!.members[1]!.backend.version = "different";
    }
    expect(() => compile(backend)).toThrow(
      "runtime_cell_member_backends_are_inconsistent",
    );

    const locality = materializeTensorParallelCell(manifest());
    for (const phase of [locality.plans.prefill, locality.plans.decode]) {
      phase.stages[1]!.members[1]!.endpoint.host = "another-host.internal";
    }
    expect(() => compile(locality)).toThrow("runtime_cell_members_must_be_anchor_local");

    const world = materializeTensorParallelCell(manifest());
    for (const phase of [world.plans.prefill, world.plans.decode]) {
      phase.stages[1]!.execution!.worldSize = 3;
    }
    expect(() => compile(world)).toThrow("runtime_cell_world_size_mismatch");

    const ranks = materializeTensorParallelCell(manifest());
    for (const phase of [ranks.plans.prefill, ranks.plans.decode]) {
      phase.stages[1]!.execution!.rankMemberIds.reverse();
    }
    expect(() => compile(ranks)).toThrow("runtime_cell_rank_members_mismatch");

    const phaseBackend = materializeTensorParallelCell(manifest());
    for (const member of phaseBackend.plans.decode.stages[1]!.members) {
      member.backend.version = "3";
    }
    expect(() => compile(phaseBackend)).toThrow(
      "runtime_kv_in_place_requires_identical_routes",
    );
  });

  it("does not claim a multi-layer range from the legacy one-layer fixture schema", () => {
    const current = materializeTensorParallelCell(manifest());
    for (const phase of [current.plans.prefill, current.plans.decode]) {
      phase.stages[1]!.execution!.fixture.schema = "gdlp-llama-cell-layer/1";
    }
    expect(() => compile(current)).toThrow("runtime_cell_v1_fixture_requires_one_layer");
  });

  it("passes every codec implemented by stage_cli and server.py", () => {
    for (const codec of ["fp16", "int8", "int8-grouped", "int8-hadamard"] as const) {
      const input = request();
      input.allowLossyActivation = codec !== "fp16";
      input.phasePlans = { prefill: route(0, codec), decode: route(0, codec) };
      const description = compile(buildRuntimePipelineManifest(input));
      expect(description.route.codec).toBe(codec);
      expect(
        description.launchOrder.every(
          (process) => argumentValue(process.command.args, "--codec") === codec,
        ),
      ).toBe(true);
    }
  });

  it("preserves root and remote placement metadata without treating root endpoint as a listener", () => {
    const description = compile();
    const rootLaunch = root(description);
    const manifestRoot = description.sourceManifest.plans.decode.stages[0]!;
    expect(rootLaunch.anchor).toEqual(manifestRoot.anchor);
    expect(rootLaunch.members).toEqual(manifestRoot.members);
    expect(argumentValue(rootLaunch.command.args, "--host")).toBe("0.0.0.0");
    expect(rootLaunch.command.args).not.toContain(manifestRoot.anchor.endpoint.host);
  });

  it("rejects legacy manifests, colliding listeners and invalid normalized options", () => {
    expect(() =>
      compilePythonLaunchDescription(legacy(manifest()), options()),
    ).toThrow("python_launcher_requires_gdlp_2");
    expect(() =>
      compile(manifest(), {
        apiEndpoint: { host: "0.0.0.0", port: 30_000 },
      }),
    ).toThrow("python_api_and_return_ports_conflict");
    expect(() => compile(manifest(), { returnBindHost: "" })).toThrow(
      "python_return_bind_host_is_invalid",
    );
    expect(() => compile(manifest(), { maxPendingRequests: 1 })).toThrow(
      "python_max_pending_requests_is_too_small",
    );
    expect(() =>
      compile(manifest(), {
        stageModule: "third_party.stage",
      } as unknown as Partial<PythonLaunchCompilerOptions>),
    ).toThrow("python_stage_module_must_be_mycellios_native");
    expect(() =>
      compile(manifest(), {
        serverModule: "third_party.server",
      } as unknown as Partial<PythonLaunchCompilerOptions>),
    ).toThrow("python_server_module_must_be_mycellios_native");
  });

  it("rejects duplicate remote listeners and collisions with root listeners", () => {
    const duplicateInput = request();
    duplicateInput.topology.nodes[1]!.endpoint = {
      ...duplicateInput.topology.nodes[2]!.endpoint,
    };
    const duplicate = buildRuntimePipelineManifest(duplicateInput);
    expect(() => compile(duplicate)).toThrow("python_remote_stage_endpoint_reused");

    const rootCollisionInput = request();
    rootCollisionInput.topology.nodes[1]!.endpoint = { host: "0.0.0.0", port: 8_081 };
    const rootCollision = buildRuntimePipelineManifest(rootCollisionInput);
    expect(() => compile(rootCollision)).toThrow(
      "python_remote_stage_endpoint_conflicts_with_root",
    );
  });

  it("detects tampering of root argv, process kind and launch identity", () => {
    const argv = compile();
    root(argv).command.args.push("--unexpected");
    expect(() => validatePythonLaunchDescription(argv)).toThrow(
      "python_launch_description_mismatch",
    );

    const kind = compile();
    (kind.launchOrder.at(-1)! as { kind: string }).kind = "remote-stage";
    expect(() => validatePythonLaunchDescription(kind)).toThrow(
      "python_launch_description_mismatch",
    );

    const identity = compile();
    identity.launchId = "000000000000000000000000";
    expect(() => validatePythonLaunchDescription(identity)).toThrow(
      "python_launch_description_mismatch",
    );

    const cell = certifyTensorParallelCell(
      materializeTensorParallelCell(manifest()),
    );
    const cellArgv = compile(cell);
    remotes(cellArgv).find((stage) => stage.stageIndex === 1)!.command.args[
      remotes(cellArgv).find((stage) => stage.stageIndex === 1)!.command.args.indexOf(
        "--cell-world-size",
      ) + 1
    ] = "99";
    expect(() => validatePythonLaunchDescription(cellArgv)).toThrow(
      "python_launch_description_mismatch",
    );
  });
});
