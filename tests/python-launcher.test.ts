import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import {
  compilePythonLaunchDescription,
  validatePythonLaunchDescription,
  type PythonLaunchCompilerOptions,
  type PythonNativeStageStageInput,
  type PythonCellMemberLaunch,
  type PythonPipelineLaunchDescription,
  type PythonRemoteStageLaunch,
  type PythonRootEngineLaunch,
} from "../src/distribution/python-launcher.js";
import {
  buildRuntimePipelineManifest,
  materializeRuntimeTensorParallelCell,
  validateRuntimePipelineManifest,
  type RuntimePipelineManifestV1,
  type RuntimePipelineManifestV2,
  type RuntimePlanRequest,
  type RuntimeSpeculationPolicy,
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
  overrides: Partial<PythonNativeStageStageInput> = {},
): PythonNativeStageStageInput {
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
): Partial<PythonLaunchCompilerOptions> {
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

  it("places prefill, batching and disabled speculation on the root server argv", () => {
    const args = root(compile()).command.args;
    expect(argumentValue(args, "--prefill-chunk-tokens")).toBe("8");
    expect(argumentValue(args, "--sealed-wave-tokens")).toBe("1");
    expect(argumentValue(args, "--max-prefill-chunk-tokens")).toBe("8");
    expect(argumentValue(args, "--max-batch-size")).toBe("2");
    expect(argumentValue(args, "--max-active-sequences")).toBe("2");
    expect(argumentValue(args, "--speculation")).toBe("off");
    expect(argumentValue(args, "--speculative-max-draft-tokens")).toBe("1");
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

  it("binds one sealed NativeStage package to an exact non-root stage and renders every flag", () => {
    const current = manifest();
    const target = current.plans.prefill.stages[2]!;
    const description = compile(current, native_stageOptions(current));
    const launch = remotes(description).find((stage) => stage.stageId === target.stageId)!;
    const normalized = description.configuration.native_stageStages[target.stageId]!;

    expect(launch.native_stage).toEqual(normalized);
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

  it("binds a local non-Hub snapshot through its content-derived pipeline identity", () => {
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
    const normalized = description.configuration.native_stageStages[target.stageId]!;
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

  it("sorts multiple NativeStage stage bindings before deriving route and launch identities", () => {
    const current = manifest();
    const forward = native_stageOptions(current, [1, 2]);
    const reversedEntries = Object.entries(forward.native_stageStages!).reverse();
    const reversed = {
      ...forward,
      native_stageStages: Object.fromEntries(reversedEntries),
    };
    expect(compile(current, reversed)).toEqual(compile(current, forward));
  });

  it("detects NativeStage configuration, process metadata and argv tampering", () => {
    const current = manifest();
    const stageId = current.plans.prefill.stages[2]!.stageId;

    const configuration = compile(current, native_stageOptions(current));
    configuration.configuration.native_stageStages[stageId]!.packagePath += "-tampered";
    expect(() => validatePythonLaunchDescription(configuration)).toThrow(
      "python_launch_description_mismatch",
    );

    const process = compile(current, native_stageOptions(current));
    remotes(process).find((stage) => stage.stageId === stageId)!.native_stage!.packageId =
      "c".repeat(64);
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

  it("fails closed for invalid NativeStage stage bindings", () => {
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
    delete (missingIdentity.native_stageStages![stageId] as Partial<PythonNativeStageStageInput>)
      .packageId;
    expect(() => compile(current, missingIdentity)).toThrow(
      `python_native_stage_stage_configuration_keys_are_invalid:${stageId}`,
    );

    const cell = materializeTensorParallelCell(manifest());
    const cellStageId = cell.plans.prefill.stages[1]!.stageId;
    expect(() => compile(cell, native_stageOptions(cell, [1]))).toThrow(
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

    const description = compile(current);
    const cell = remotes(description).find((stage) => stage.stageIndex === 1)!;
    expect(cell.members).toHaveLength(2);
    expect(cell.cell).toEqual(current.plans.decode.stages[1]!.execution);
    expect(argumentValue(cell.command.args, "--cell-fixture")).toBe(
      current.plans.decode.stages[1]!.execution!.fixture.path,
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
    const current = materializeExternalTensorParallelCell(manifest());
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
    const current = materializeTensorParallelCell(manifest(), 1, 0, {
      computeDtype: "bfloat16",
      computeApi: "rocm",
      rankDevices: ["cuda:0", "cuda:3"],
    });
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
    const current = materializeExternalTensorParallelCell(manifest(), {
      computeDtype: "float16",
      rankDevices: ["cuda:0", "cuda:7"],
    });
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
    const current = materializeExternalTensorParallelCell(manifest());
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

    const cell = materializeTensorParallelCell(manifest());
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
