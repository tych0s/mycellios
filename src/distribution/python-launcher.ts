import { createHash } from "node:crypto";
import {
  validateRuntimePipelineManifest,
  type RuntimeActivationCodec,
  type RuntimeDecodePlanManifest,
  type RuntimeEndpoint,
  type RuntimePipelineManifestV2,
  type RuntimePrefillPlanManifest,
  type RuntimeSpeculationPolicy,
  type RuntimeStageMemberManifest,
  type RuntimeTensorParallelCellExecutionManifest,
  type RuntimeVirtualStageManifest,
} from "./runtime-manifest.js";
import type {
  MacroWavePlanContractV1,
  MacroWaveStageExecutionContractV1,
} from "./types.js";
import {
  normalizeExecutorIsolationPolicy,
  type ExecutorIsolationPolicyOptions,
  type ExecutorIsolationPolicyV4,
} from "./process-environment.js";

const GDLP_FRAME_HEADER_BYTES = 32n;
const DEFAULT_PREFILL_INFLIGHT_BYTES = 64 * 1024 * 1024;
const MAX_PREFILL_INFLIGHT_BYTES = 1024 * 1024 * 1024;
const MAX_SPECULATIVE_BRANCHES = 64;
const MAX_SPECULATIVE_BRANCH_TOKENS = 1_048_576;
const MAX_SPECULATIVE_KV_BYTES = 2 ** 40;
const DEFAULT_SPECULATIVE_INFLIGHT_WAVES = 1;
const DEFAULT_SPECULATIVE_INFLIGHT_BYTES = 0;
const MAX_SPECULATIVE_INFLIGHT_WAVES = 16;
const MAX_SPECULATIVE_INFLIGHT_BYTES = 1024 * 1024 * 1024;
const DEFAULT_RETAINED_SESSIONS = 4;
const DEFAULT_RETAINED_SESSION_TOKENS = 8_192;
const DEFAULT_RETAINED_SESSION_TTL_SECONDS = 10 * 60;
const RECOVERY_STANDBY_SCHEMA = "gdlp-recovery-standby-route/1";
const PAGED_STAGE_RUNTIME_SCHEMA = "mycellios-hf-paged-stage/2";
const LOCAL_DRAFT_MODEL_SCHEMA = "mycellios-local-draft-model/1";
const MAX_PAGED_BLOCK_SIZE = 256;
const MAX_PAGED_NUM_BLOCKS = 1_048_576;
const MAX_PAGED_BATCH_TOKENS = 65_536;
const MAX_PAGED_ACTIVE_REQUESTS = 4_096;
const MAX_PAGED_SEQUENCE_TOKENS = 1_048_576;
export const MYCELLIOS_STAGE_MODULE = "distributed_runtime.stage_cli";
export const MYCELLIOS_SERVER_MODULE = "distributed_runtime.server";
export const MYCELLIOS_CELL_MEMBER_MODULE =
  "distributed_runtime.cell_member_cli";

export type PythonLaunchPhase = "prefill" | "decode";

/** Every tensor codec currently understood by the Python GDLP/2 runtime. */
export type PythonPrefillReservationCodec =
  | RuntimeActivationCodec
  | "fp32"
  | "int8-grouped-deflate"
  | "int8-hadamard-deflate";

/**
 * Data-independent upper bound for one encoded prefill frame on the wire.
 *
 * This deliberately mirrors Python `_prefill_frame_byte_reservation`: quantized
 * scale bytes are included per row/block and DEFLATE uses zlib's worst-case
 * bound. The return type is bigint so a hostile-but-valid safe-integer shape
 * cannot make the static check optimistic through Number multiplication.
 */
export function pythonPrefillFrameByteReservation(
  codec: PythonPrefillReservationCodec,
  tokenCount: number,
  hiddenSize: number,
): bigint {
  if (
    !Number.isSafeInteger(tokenCount) ||
    tokenCount < 1 ||
    !Number.isSafeInteger(hiddenSize) ||
    hiddenSize < 1
  ) {
    throw new Error("python_prefill_frame_shape_is_invalid");
  }

  const tokens = BigInt(tokenCount);
  const hidden = BigInt(hiddenSize);
  const elements = tokens * hidden;
  const deflated =
    codec === "int8-grouped-deflate" || codec === "int8-hadamard-deflate";
  const baseCodec =
    codec === "int8-grouped-deflate"
      ? "int8-grouped"
      : codec === "int8-hadamard-deflate"
        ? "int8-hadamard"
        : codec;

  let payloadBytes: bigint;
  if (baseCodec === "fp32") {
    payloadBytes = elements * 4n;
  } else if (baseCodec === "fp16") {
    payloadBytes = elements * 2n;
  } else if (baseCodec === "int8") {
    payloadBytes = elements + 4n;
  } else if (baseCodec === "int8-grouped") {
    const blocks = (hidden + 63n) / 64n;
    payloadBytes = elements + tokens * blocks * 4n;
  } else if (baseCodec === "int8-hadamard") {
    const blocks = hadamardQuantizationBlockCount(hidden);
    payloadBytes = elements + tokens * blocks * 4n;
  } else {
    throw new Error("python_prefill_frame_codec_is_invalid");
  }

  if (deflated) {
    // Same bound as Python protocol._deflate_bound: raw DEFLATE stored-block
    // overhead (13) plus the zlib header and Adler-32 trailer (6).
    payloadBytes =
      payloadBytes +
      (payloadBytes >> 12n) +
      (payloadBytes >> 14n) +
      (payloadBytes >> 25n) +
      19n;
  }
  return GDLP_FRAME_HEADER_BYTES + payloadBytes;
}

function hadamardQuantizationBlockCount(hiddenSize: bigint): bigint {
  // Python repeatedly consumes the largest power of two <= min(64, remaining).
  // Full groups contribute one block; the sub-64 tail contributes its popcount.
  let remainder = Number(hiddenSize % 64n);
  let tailBlocks = 0n;
  while (remainder > 0) {
    tailBlocks += BigInt(remainder & 1);
    remainder >>= 1;
  }
  return hiddenSize / 64n + tailBlocks;
}

export interface PythonRuntimeModelInput {
  /** Hugging Face id or local snapshot path consumed by the Python loaders. */
  source: string;
  /** Null deliberately omits --revision, as required by local snapshots. */
  revision?: string | null;
  /** Decimal uint64 returned by Python model_snapshot_identity(). */
  snapshotIdentity?: string;
  /** Full content identity returned by Python model_artifact_reference(). */
  artifactIdentity?: string;
  /** Stable, host-independent source returned with artifactIdentity. */
  canonicalSource?: string;
  /** Must be supplied with the other strong coordinates; null is meaningful. */
  canonicalRevision?: string | null;
}

export type PythonDraftModelDevice =
  | "cpu"
  | "cuda"
  | `cuda:${number}`
  | "mps"
  | "xpu"
  | `xpu:${number}`;

export interface PythonDraftModelInput extends PythonRuntimeModelInput {
  schema: typeof LOCAL_DRAFT_MODEL_SCHEMA;
  device: PythonDraftModelDevice;
  dtype: "float32" | "float16" | "bfloat16";
}

export interface PythonLaunchCompilerOptions {
  /** Bind address for the Mycellios root server. */
  apiEndpoint: RuntimeEndpoint;
  /** Address advertised to remote stages for ACK/token return. */
  returnEndpoint: RuntimeEndpoint;
  /** Local interface used by the root return listener, e.g. 0.0.0.0. */
  returnBindHost: string;
  runtimeModel?: PythonRuntimeModelInput;
  /** Sealed local sibling model used only to propose target-verified tokens. */
  draftModel?: PythonDraftModelInput | null;
  publicModelName?: string;
  pythonExecutable?: string;
  /** Fixed native entrypoint; exposed only for normalized-description validation. */
  stageModule?: typeof MYCELLIOS_STAGE_MODULE;
  /** Fixed native entrypoint; exposed only for normalized-description validation. */
  serverModule?: typeof MYCELLIOS_SERVER_MODULE;
  threadsPerStage?: number;
  connectTimeoutSeconds?: number;
  batchWindowMs?: number;
  /** Maximum ordered prefill chunks allowed on the physical route at once. */
  prefillInflightChunks?: number;
  /** Per-request hard cap for encoded prefill wire bytes currently in flight. */
  prefillInflightBytes?: number;
  /** Concurrent physical KV children. Zero keeps the physical tree disabled. */
  maxSpeculativeBranches?: number;
  /** Absolute context-token ceiling for every physical KV child. */
  maxSpeculativeBranchTokens?: number;
  /** Aggregate stage-local byte ceiling for all physical child KV caches. */
  maxSpeculativeKvBytes?: number;
  /**
   * Ordered linear VERIFY waves. Omitted (and treated as one) preserves the
   * canonical launch/2 identity and historical synchronous path.
   */
  speculativeInflightWaves?: number;
  /**
   * Per-request encoded VERIFY bytes in flight. Omitted (and treated as zero)
   * with one wave preserves the canonical launch/2 identity.
   */
  speculativeInflightBytes?: number;
  maxPendingRequests?: number;
  maxOutputTokens?: number;
  speculationMinimumSpeedup?: number;
  /** Exact completed-chat KV slots kept hot for the next turn. Zero disables reuse. */
  maxRetainedSessions?: number;
  /** Aggregate idle KV token ceiling across retained chats. */
  maxRetainedSessionTokens?: number;
  retainedSessionTtlSeconds?: number;
  /**
   * Exact greedy midstream recovery. Executor identities are supplied by the
   * physical launcher; the compiler never guesses them from a backend label.
   */
  recovery?: PythonRecoveryInput | PythonRecoveryConfiguration | null;
  /**
   * First-class Mycellios GGUF range packages. Every binding carries the
   * complete model coordinates as well as the exact physical stage range.
   */
  nativeGgufStages?: Record<string, PythonNativeGgufStageInput>;
  /**
   * Host-local bindings for every RAM-backed MoE stage. The compiler requires
   * complete coverage and seals them to the stage budgets before execution.
   */
  ramBackedMoeStages?: Record<string, PythonRamBackedMoeStageInput>;
  /**
   * Complete, per-stage Mycellios paged-KV contracts. Supplying one binding
   * requires every logical stage so no process silently falls back to an
   * incompatible cache backend.
   */
  pagedKvStages?: Record<string, PythonPagedKvStageInput>;
  /**
   * Process-boundary controls sealed into every physical launch request.
   * The compact form is expanded to an honest, versioned policy.
   */
  executorIsolation?:
    | ExecutorIsolationPolicyOptions
    | ExecutorIsolationPolicyV4;
}

export interface PythonRecoveryStandbyRouteInput {
  /** Present on normalized descriptions; otherwise derived from the contract. */
  schema?: typeof RECOVERY_STANDBY_SCHEMA;
  /** Present on normalized descriptions and verified against the derived id. */
  routeId?: string;
  firstStage: RuntimeEndpoint;
  /** Root first, followed by every remote stage in logical model order. */
  stageExecutorIds: string[];
}

export interface PythonRecoveryInput {
  maxRetries: number;
  /** Complete active route contract, root first. */
  stageExecutorIds: string[];
  /** One or more idle, independently launched physical routes. */
  standbyRoutes: PythonRecoveryStandbyRouteInput[];
}

export interface PythonRecoveryStandbyRouteConfiguration {
  schema: typeof RECOVERY_STANDBY_SCHEMA;
  routeId: string;
  firstStage: RuntimeEndpoint;
  stageExecutorIds: string[];
}

export interface PythonRecoveryConfiguration {
  maxRetries: number;
  stageExecutorIds: string[];
  standbyRoutes: PythonRecoveryStandbyRouteConfiguration[];
}

/**
 * Authenticated Mycellios-native GGUF package for one exact contiguous range.
 * model* names the complete source model and is deliberately distinct from
 * packageId, which names only this stage package.
 */
export interface PythonNativeGgufStageInput {
  packagePath: string;
  packageId: string;
  modelIdentity: string;
  modelSource: string;
  modelRevision: string | null;
  layerStart: number;
  layerEnd: number;
  totalLayers: number;
}

export interface PythonNativeGgufStageConfiguration
  extends PythonNativeGgufStageInput {}

export interface PythonPagedKvStageInput {
  schema: typeof PAGED_STAGE_RUNTIME_SCHEMA;
  /** Canonical torch spelling: cpu, cuda or cuda:<index>. */
  device: string;
  attentionBackend: "eager" | "sdpa";
  blockSize: number;
  numBlocks: number;
  maxBatchTokens: number;
  maxActiveRequests: number;
  maxSequenceTokens: number;
  /**
   * Maximum KV tensor payload bytes in the CPU tier. This is not a promise
   * about total process RSS or pinned-allocator bookkeeping.
   */
  cpuSpillBytes: number;
}

export interface PythonPagedKvStageConfiguration
  extends PythonPagedKvStageInput {}

export type PythonRamBackedMoeAdapterId =
  | "transformers-qwen3-moe-v1"
  | "transformers-glm4-moe-v1";

export interface PythonRamBackedMoeCacheConfiguration {
  schema: "gdlp-predictive-expert-cache/1";
  /** Active, prefetch and ephemeral expert tensors; excludes fixed VRAM. */
  capacityBytes: number;
  prefetchReserveBytes: number;
  /** Decimal GB/s, matching PredictiveCacheConfig in Python. */
  pcieBandwidthGbytesPerSecond: number;
  hotnessDecay: number;
  minPrefetchConfidence: number;
}

export interface PythonRamBackedMoeStageInput {
  schema: "gdlp-local-safetensors-moe-stage/1";
  /** Absolute path on the stage host; becomes --model, never a duplicate flag. */
  snapshotPath: string;
  /** Full content/commit identity returned by model_artifact_reference(). */
  artifactIdentity: string;
  adapterId: PythonRamBackedMoeAdapterId;
  layerStart: number;
  layerEnd: number;
  totalLayers: number;
  device: string;
  /** False preserves the direct loader's no-second-RAM-copy path. */
  pinMemory: false;
  /** False prevents an unavailable GPU from silently changing execution mode. */
  allowCpuFallback: false;
  /** Optional input assertions; normalized launch descriptions always seal them. */
  residentParameterBudgetBytes?: number;
  totalRoutedExpertBytes?: number;
  largestExpertBytes?: number;
  residentStreamingTransientBytes?: number;
  boundedPinnedStagingReserveBytes?: number;
  hostRamPeakUpperBoundBytes?: number;
  cache: PythonRamBackedMoeCacheConfiguration;
}

export interface PythonRamBackedMoeStageConfiguration
  extends Omit<
    PythonRamBackedMoeStageInput,
    | "residentParameterBudgetBytes"
    | "totalRoutedExpertBytes"
    | "largestExpertBytes"
    | "residentStreamingTransientBytes"
    | "boundedPinnedStagingReserveBytes"
    | "hostRamPeakUpperBoundBytes"
  > {
  residentParameterBudgetBytes: number;
  totalRoutedExpertBytes: number;
  largestExpertBytes: number;
  residentStreamingTransientBytes: number;
  boundedPinnedStagingReserveBytes: number;
  hostRamPeakUpperBoundBytes: number;
}

export interface PythonRuntimeModelSource {
  source: string;
  revision: string | null;
  snapshotIdentity?: string;
  /** Stable executor identity; derived without embedding a host cache path. */
  artifactIdentity?: string;
  canonicalSource?: string;
  canonicalRevision?: string | null;
}

export interface PythonDraftModelConfiguration extends PythonRuntimeModelSource {
  schema: typeof LOCAL_DRAFT_MODEL_SCHEMA;
  artifactIdentity: string;
  canonicalSource: string;
  canonicalRevision: string | null;
  device: PythonDraftModelInput["device"];
  dtype: PythonDraftModelInput["dtype"];
  parameterBytes: number;
  memoryReservationBytes: number;
}

export interface PythonLaunchConfiguration {
  apiEndpoint: RuntimeEndpoint;
  returnEndpoint: RuntimeEndpoint;
  returnBindHost: string;
  runtimeModel: PythonRuntimeModelSource;
  draftModel: PythonDraftModelConfiguration | null;
  publicModelName: string;
  pythonExecutable: string;
  stageModule: typeof MYCELLIOS_STAGE_MODULE;
  serverModule: typeof MYCELLIOS_SERVER_MODULE;
  threadsPerStage: number;
  connectTimeoutSeconds: number;
  batchWindowMs: number;
  prefillInflightChunks: number;
  prefillInflightBytes: number;
  maxSpeculativeBranches: number;
  maxSpeculativeBranchTokens: number;
  maxSpeculativeKvBytes: number;
  /** Present only when the native linear VERIFY conveyor is explicitly enabled. */
  speculativeInflightWaves?: number;
  /** Present only together with speculativeInflightWaves. */
  speculativeInflightBytes?: number;
  maxPendingRequests: number;
  maxOutputTokens: number;
  speculationMinimumSpeedup: number;
  maxRetainedSessions: number;
  maxRetainedSessionTokens: number;
  retainedSessionTtlSeconds: number;
  recovery: PythonRecoveryConfiguration | null;
  /** Sorted by stageId and sealed into route/process/launch identities. */
  nativeGgufStages: Record<string, PythonNativeGgufStageConfiguration>;
  /** Sorted, host-local bindings for the certified physical RAM-backed runner. */
  ramBackedMoeStages: Record<string, PythonRamBackedMoeStageConfiguration>;
  /** Sorted, complete bindings for the native paged-KV runner. */
  pagedKvStages: Record<string, PythonPagedKvStageConfiguration>;
  /** Versioned controls enforced by the worker before spawning a process. */
  executorIsolation: ExecutorIsolationPolicyV4;
}

export interface PythonPrefillLaunchSettings {
  phase: "prefill";
  planId: string;
  activationCodec: RuntimeActivationCodec;
  microBatchSize: number;
  chunkTokens: number;
  macroWave: MacroWavePlanContractV1 | null;
}

export interface PythonDecodeLaunchSettings {
  phase: "decode";
  planId: string;
  activationCodec: RuntimeActivationCodec;
  microBatchSize: number;
  directTokenReturnStage: number;
  speculation: RuntimeSpeculationPolicy;
  macroWave: MacroWavePlanContractV1 | null;
}

export type PythonPhaseLaunchSettings =
  | PythonPrefillLaunchSettings
  | PythonDecodeLaunchSettings;

export interface PythonStageCommand {
  executable: string;
  /** Argument vector intended for execFile/spawn with shell=false. */
  args: string[];
}

export interface PythonDownstreamStage {
  stageId: string;
  stageIndex: number;
  layerEnd: number;
  anchorMemberId: string;
  endpoint: RuntimeEndpoint;
}

interface PythonLaunchBase {
  launchIndex: number;
  processId: string;
  routeId: string;
  phases: ["prefill", "decode"];
  logicalPlanIds: [string, string];
  stageId: string;
  stageIndex: number;
  layerStart: number;
  layerEnd: number;
  totalLayers: number;
  codec: RuntimeActivationCodec;
  /** Hard per-frame limits enforced by root and every physical stage. */
  sealedWaveTokens: number;
  maxPrefillChunkTokens: number;
  anchor: {
    memberId: string;
    endpoint: RuntimeEndpoint;
  };
  members: RuntimeStageMemberManifest[];
  /** Preserved contract; null is the unchanged resident layer-range path. */
  macroWave: MacroWaveStageExecutionContractV1 | null;
  /** Sealed process-boundary controls; remote workers reject mismatches. */
  isolation: ExecutorIsolationPolicyV4;
  command: PythonStageCommand;
}

/** A physical stage_cli process. Stage zero can never use this variant. */
export interface PythonRemoteStageLaunch extends PythonLaunchBase {
  kind: "remote-stage";
  downstream: PythonDownstreamStage | null;
  returnEndpoint: RuntimeEndpoint;
  /** Null selects the ordinary single-member StageModelRunner. */
  cell: RuntimeTensorParallelCellExecutionManifest | null;
  /** Null selects the ordinary Mycellios stage loader. */
  nativeGguf: PythonNativeGgufStageConfiguration | null;
}

/** One nonzero rank that joins a member-local TP cell before its anchor is ready. */
export interface PythonCellMemberLaunch extends PythonLaunchBase {
  kind: "cell-member";
  rank: number;
  fixturePath: string;
  pipelineSnapshotIdentity: string;
  collectiveBackend: RuntimeTensorParallelCellExecutionManifest["collectiveBackend"];
  computeDtype: RuntimeTensorParallelCellExecutionManifest["computeDtype"];
  device: string;
  cellAnchor: {
    memberId: string;
    controlEndpoint: RuntimeEndpoint;
  };
  worldSize: number;
  operationTimeoutSeconds: number;
  startupTimeoutSeconds: number;
}

/** Root partition hosted by DistributedPipelineEngine inside server.py. */
export interface PythonRootEngineLaunch extends PythonLaunchBase {
  kind: "root-engine";
  stageIndex: 0;
  /** Null selects the standard root model loader. */
  nativeGguf: PythonNativeGgufStageConfiguration | null;
  boundaries: number[];
  firstRemoteStage: PythonDownstreamStage;
  apiEndpoint: RuntimeEndpoint;
  returnBindHost: string;
  returnEndpoint: RuntimeEndpoint;
  prefill: PythonPrefillLaunchSettings;
  decode: PythonDecodeLaunchSettings;
}

export type PythonLaunchProcess =
  | PythonCellMemberLaunch
  | PythonRemoteStageLaunch
  | PythonRootEngineLaunch;

export interface PythonPipelineLaunchRoute {
  routeId: string;
  phases: ["prefill", "decode"];
  phaseSettings: [PythonPrefillLaunchSettings, PythonDecodeLaunchSettings];
  codec: RuntimeActivationCodec;
  /** Logical model order, root first. */
  logicalStageIds: string[];
  /** Remote stages downstream-first, then the root server last. */
  downstreamFirstProcessIds: string[];
  cellMemberProcessIds: string[];
  remoteStageProcessIds: string[];
  rootProcessId: string;
}

export interface PythonPipelineLaunchDescription {
  schema: "gdlp-python-launch/2";
  launchId: string;
  sourceProtocol: "gdlp/2";
  sourceManifest: RuntimePipelineManifestV2;
  pipelineId: string;
  modelIdentity: {
    id: string;
    revision: string;
    tokenizerId: string;
  };
  runtimeModel: PythonRuntimeModelSource;
  totalLayers: number;
  configuration: PythonLaunchConfiguration;
  route: PythonPipelineLaunchRoute;
  launchOrder: PythonLaunchProcess[];
}

interface PythonSpeculationArguments {
  provider: "off" | "ngram" | "draft-tree" | "draft-model";
  maxDraftTokens: number;
  maxBranches: number;
  maxBranchTokens: number;
  maxKvBytes: number;
  maxWaveTokens: number;
  artifactId: string | null;
  parameterBytes: number;
  memoryReservationBytes: number;
}

/**
 * Produce a launch plan only. This module never imports child_process and
 * never starts a Python process.
 */
export function compilePythonLaunchDescription(
  manifestValue: unknown,
  optionsValue: PythonLaunchCompilerOptions,
): PythonPipelineLaunchDescription {
  validateRuntimePipelineManifest(manifestValue);
  if (manifestValue.protocol !== "gdlp/2") {
    throw new Error("python_launcher_requires_gdlp_2");
  }
  const configuration = normalizeConfiguration(manifestValue, optionsValue);
  const description = buildDescription(manifestValue, configuration);
  validatePythonLaunchDescription(description);
  return description;
}

/**
 * Closed validation regenerates every derived id, edge and argv from the
 * embedded validated manifest and normalized configuration.
 */
export function validatePythonLaunchDescription(
  value: unknown,
): asserts value is PythonPipelineLaunchDescription {
  if (!isRecord(value)) throw new Error("python_launch_description_must_be_an_object");
  if (value.schema !== "gdlp-python-launch/2") {
    throw new Error("unsupported_python_launch_schema");
  }
  validateRuntimePipelineManifest(value.sourceManifest);
  if (value.sourceManifest.protocol !== "gdlp/2") {
    throw new Error("python_launcher_requires_gdlp_2");
  }
  const configuration = normalizeConfiguration(
    value.sourceManifest,
    value.configuration as unknown as PythonLaunchCompilerOptions,
    true,
  );
  const expected = buildDescription(value.sourceManifest, configuration);
  if (canonicalJson(value) !== canonicalJson(expected)) {
    throw new Error("python_launch_description_mismatch");
  }
}

function buildDescription(
  manifest: RuntimePipelineManifestV2,
  configuration: PythonLaunchConfiguration,
): PythonPipelineLaunchDescription {
  const prefill = manifest.plans.prefill;
  const decode = manifest.plans.decode;
  assertSharedExecutableRoute(manifest, prefill, decode);
  const stages = prefill.stages;
  const phaseSettings: [PythonPrefillLaunchSettings, PythonDecodeLaunchSettings] = [
    prefillSettings(prefill),
    decodeSettings(decode),
  ];
  const frameLimits = executableFrameLimits(prefill, decode);
  const codec = prefill.activationCodec;
  const routeId = routeIdentity(
    manifest.pipelineId,
    phaseSettings,
    stages,
    codec,
    configuration.nativeGgufStages,
    configuration.ramBackedMoeStages,
    configuration.pagedKvStages,
    configuration.recovery,
  );
  const planIds: [string, string] = [prefill.planId, decode.planId];
  const boundaries = [0, ...stages.map((stage) => stage.layerEnd)];
  const launchOrder: PythonLaunchProcess[] = [];
  const occupiedRemoteEndpoints = new Set<string>();

  // Stage N-1 must listen before N-2 connects, continuing back to stage 1.
  for (let index = stages.length - 1; index >= 1; index -= 1) {
    const stage = stages[index]!;
    const listenKey = endpointKey(stage.anchor.endpoint);
    if (occupiedRemoteEndpoints.has(listenKey)) {
      throw new Error("python_remote_stage_endpoint_reused");
    }
    if (
      listenKey === endpointKey(configuration.apiEndpoint) ||
      listenKey === endpointKey(configuration.returnEndpoint)
    ) {
      throw new Error("python_remote_stage_endpoint_conflicts_with_root");
    }
    occupiedRemoteEndpoints.add(listenKey);
    const next = stages[index + 1];
    const downstream = next ? downstreamStage(next) : null;
    const nativeGguf = configuration.nativeGgufStages[stage.stageId] ?? null;
    const cell = tensorParallelCellExecution(stage.execution);
    const macroWave = stage.macroWave;
    if (cell?.fixture.location === "member-local") {
      const external = cell.external!;
      const pipelineSnapshotIdentity = configuration.runtimeModel.snapshotIdentity;
      if (!pipelineSnapshotIdentity) {
        throw new Error("python_external_cell_requires_snapshot_identity");
      }
      for (let rank = 1; rank < cell.worldSize; rank += 1) {
        const memberId = cell.rankMemberIds[rank]!;
        const member = stage.members.find((candidate) => candidate.nodeId === memberId);
        if (!member) throw new Error("python_external_cell_rank_member_is_missing");
        const memberPartial: Omit<PythonCellMemberLaunch, "launchIndex" | "command"> = {
          kind: "cell-member",
          processId: processIdentity(routeId, "cell-member", stage, memberId, rank),
          routeId,
          phases: ["prefill", "decode"],
          logicalPlanIds: [...planIds],
          stageId: stage.stageId,
          stageIndex: stage.index,
          layerStart: stage.layerStart,
          layerEnd: stage.layerEnd,
          totalLayers: manifest.totalLayers,
          codec,
          ...frameLimits,
          anchor: {
            memberId,
            endpoint: { ...member.endpoint },
          },
          members: structuredClone(stage.members),
          macroWave: null,
          isolation: structuredClone(configuration.executorIsolation),
          rank,
          fixturePath: external.rankFixturePaths[rank]!,
          pipelineSnapshotIdentity,
          collectiveBackend: cell.collectiveBackend,
          computeDtype: cell.computeDtype,
          device: cell.rankDevices[rank]!,
          cellAnchor: {
            memberId: stage.anchor.memberId,
            controlEndpoint: {
              host: external.controlAdvertiseHost,
              port: external.controlPort,
            },
          },
          worldSize: cell.worldSize,
          operationTimeoutSeconds: cell.operationTimeoutSeconds,
          startupTimeoutSeconds: external.startupTimeoutSeconds,
        };
        launchOrder.push({
          ...memberPartial,
          launchIndex: launchOrder.length,
          command: {
            executable: configuration.pythonExecutable,
            args: renderCellMemberArguments(memberPartial, configuration),
          },
        });
      }
    }
    const processId = processIdentity(routeId, "remote-stage", stage);
    const partial: Omit<PythonRemoteStageLaunch, "launchIndex" | "command"> = {
      kind: "remote-stage",
      processId,
      routeId,
      phases: ["prefill", "decode"],
      logicalPlanIds: [...planIds],
      stageId: stage.stageId,
      stageIndex: stage.index,
      layerStart: stage.layerStart,
      layerEnd: stage.layerEnd,
      totalLayers: manifest.totalLayers,
      codec,
      ...frameLimits,
      anchor: structuredClone(stage.anchor),
      members: structuredClone(stage.members),
      macroWave: macroWave ? structuredClone(macroWave) : null,
      isolation: structuredClone(configuration.executorIsolation),
      downstream,
      returnEndpoint: { ...configuration.returnEndpoint },
      cell: cell ? structuredClone(cell) : null,
      nativeGguf: nativeGguf ? structuredClone(nativeGguf) : null,
    };
    launchOrder.push({
      ...partial,
      launchIndex: launchOrder.length,
      command: {
        executable: configuration.pythonExecutable,
        args: renderRemoteStageArguments(partial, configuration),
      },
    });
  }

  const rootStage = stages[0]!;
  const rootNativeGguf =
    configuration.nativeGgufStages[rootStage.stageId] ?? null;
  const firstRemoteStage = downstreamStage(stages[1]!);
  const rootProcessId = rootProcessIdentity(
    routeId,
    rootStage,
    configuration.draftModel,
  );
  const rootPartial: Omit<PythonRootEngineLaunch, "launchIndex" | "command"> = {
    kind: "root-engine",
    processId: rootProcessId,
    routeId,
    phases: ["prefill", "decode"],
    logicalPlanIds: [...planIds],
    stageId: rootStage.stageId,
    stageIndex: 0,
    nativeGguf: rootNativeGguf ? structuredClone(rootNativeGguf) : null,
    layerStart: rootStage.layerStart,
    layerEnd: rootStage.layerEnd,
    totalLayers: manifest.totalLayers,
    codec,
    ...frameLimits,
    anchor: structuredClone(rootStage.anchor),
    members: structuredClone(rootStage.members),
    macroWave: rootStage.macroWave
      ? structuredClone(rootStage.macroWave)
      : null,
    isolation: structuredClone(configuration.executorIsolation),
    boundaries,
    firstRemoteStage,
    apiEndpoint: { ...configuration.apiEndpoint },
    returnBindHost: configuration.returnBindHost,
    returnEndpoint: { ...configuration.returnEndpoint },
    prefill: structuredClone(phaseSettings[0]),
    decode: structuredClone(phaseSettings[1]),
  };
  launchOrder.push({
    ...rootPartial,
    launchIndex: launchOrder.length,
    command: {
      executable: configuration.pythonExecutable,
      args: renderRootEngineArguments(rootPartial, configuration),
    },
  });

  const remoteStageProcessIds = launchOrder
    .filter((entry): entry is PythonRemoteStageLaunch => entry.kind === "remote-stage")
    .map((entry) => entry.processId);
  const cellMemberProcessIds = launchOrder
    .filter((entry): entry is PythonCellMemberLaunch => entry.kind === "cell-member")
    .map((entry) => entry.processId);
  const route: PythonPipelineLaunchRoute = {
    routeId,
    phases: ["prefill", "decode"],
    phaseSettings,
    codec,
    logicalStageIds: stages.map((stage) => stage.stageId),
    downstreamFirstProcessIds: launchOrder.map((entry) => entry.processId),
    cellMemberProcessIds,
    remoteStageProcessIds,
    rootProcessId,
  };
  const withoutIdentity: Omit<PythonPipelineLaunchDescription, "launchId"> = {
    schema: "gdlp-python-launch/2",
    sourceProtocol: "gdlp/2",
    sourceManifest: structuredClone(manifest),
    pipelineId: manifest.pipelineId,
    modelIdentity: {
      id: manifest.modelId,
      revision: manifest.modelRevision,
      tokenizerId: manifest.tokenizerId,
    },
    runtimeModel: structuredClone(configuration.runtimeModel),
    totalLayers: manifest.totalLayers,
    configuration: structuredClone(configuration),
    route,
    launchOrder,
  };
  return {
    ...withoutIdentity,
    launchId: digest(canonicalJson(withoutIdentity), 24),
  };
}

function assertSharedExecutableRoute(
  manifest: RuntimePipelineManifestV2,
  prefill: RuntimePrefillPlanManifest,
  decode: RuntimeDecodePlanManifest,
): void {
  if (hasDifferingPhaseCellProfiles(prefill.stages, decode.stages)) {
    throw new Error("python_server_does_not_execute_phase_specific_cell_profiles");
  }
  if (!samePhysicalTopology(prefill.stages, decode.stages)) {
    throw new Error("python_server_requires_shared_prefill_decode_route");
  }
  if (prefill.activationCodec !== decode.activationCodec) {
    throw new Error("python_shared_route_requires_one_codec");
  }
  if (manifest.kvTransition.mode !== "in-place") {
    throw new Error("python_server_requires_in_place_kv");
  }
  if (prefill.stages.length < 2) {
    throw new Error("python_distributed_runtime_requires_two_stages");
  }
  assertExecutableMacroWaveContract(prefill, decode);
  for (const [phase, plan] of [
    ["prefill", prefill],
    ["decode", decode],
  ] as const) {
    for (const stage of plan.stages) {
      const cell = tensorParallelCellExecution(stage.execution);
      if (stage.index === 0 && cell !== undefined) {
        throw new Error(`python_cell_stage_cannot_be_root:${phase}:${stage.stageId}`);
      }
      if (stage.last && cell !== undefined) {
        throw new Error(`python_cell_stage_cannot_be_final:${phase}:${stage.stageId}`);
      }
      if (stage.members.length !== 1 && cell === undefined) {
        throw new Error(`python_stage_runtime_requires_single_member:${phase}:${stage.stageId}`);
      }
      if (stage.members.length === 1 && cell !== undefined) {
        throw new Error(`python_cell_stage_requires_multiple_members:${phase}:${stage.stageId}`);
      }
      if (!stage.members.some((member) => member.nodeId === stage.anchor.memberId)) {
        throw new Error(`python_stage_anchor_is_not_physical_member:${phase}:${stage.stageId}`);
      }
    }
  }
}

function assertExecutableMacroWaveContract(
  prefill: RuntimePrefillPlanManifest,
  decode: RuntimeDecodePlanManifest,
): void {
  const contracts = [prefill.macroWave, decode.macroWave];
  if (contracts.every((contract) => contract === undefined)) return;
  if (contracts.some((contract) => contract === undefined)) {
    throw new Error("python_macro_wave_requires_both_phase_contracts");
  }
  const [prefillContract, decodeContract] = contracts as [
    MacroWavePlanContractV1,
    MacroWavePlanContractV1,
  ];
  if (
    canonicalJson({
      schema: prefillContract.schema,
      routeKind: prefillContract.routeKind,
      waveTokens: prefillContract.waveTokens,
      expectedCommittedTokensPerWave:
        prefillContract.expectedCommittedTokensPerWave,
    }) !==
    canonicalJson({
      schema: decodeContract.schema,
      routeKind: decodeContract.routeKind,
      waveTokens: decodeContract.waveTokens,
      expectedCommittedTokensPerWave:
        decodeContract.expectedCommittedTokensPerWave,
    })
  ) {
    throw new Error("python_macro_wave_phase_contracts_do_not_match");
  }
  const speculation = pythonSpeculation(decode.speculation);
  if (
    speculation.provider === "draft-tree" &&
    decodeContract.waveTokens !== speculation.maxWaveTokens
  ) {
    throw new Error("python_draft_tree_wave_contract_mismatch");
  }
  if (speculation.provider === "off") {
    if (
      decodeContract.waveTokens !== 1 ||
      decodeContract.expectedCommittedTokensPerWave !== 1
    ) {
      throw new Error("python_macro_wave_autoregressive_contract_mismatch");
    }
  } else {
    if (speculation.maxDraftTokens + 1 > decodeContract.waveTokens) {
      throw new Error("python_macro_wave_speculation_exceeds_sealed_wave");
    }
    if (
      decodeContract.expectedCommittedTokensPerWave >
      speculation.maxDraftTokens + 1
    ) {
      throw new Error(
        "python_macro_wave_committed_projection_exceeds_strategy",
      );
    }
  }
  if (decodeContract.waveTokens > 17) {
    throw new Error("python_macro_wave_tokens_exceed_runtime_limit");
  }
  for (const stage of [...prefill.stages, ...decode.stages]) {
    const execution = stage.macroWave;
    if (!execution) {
      throw new Error("python_macro_wave_stage_contract_is_missing");
    }
    if (execution.residentKind === "expert-shard") {
      throw new Error("python_runtime_does_not_execute_implicit_expert_shards");
    }
  }
}

function executableFrameLimits(
  prefill: RuntimePrefillPlanManifest,
  decode: RuntimeDecodePlanManifest,
): { sealedWaveTokens: number; maxPrefillChunkTokens: number } {
  const speculation = pythonSpeculation(decode.speculation);
  return {
    sealedWaveTokens:
      decode.macroWave?.waveTokens ??
      speculation.maxWaveTokens,
    maxPrefillChunkTokens: prefill.chunkTokens,
  };
}

function hasDifferingPhaseCellProfiles(
  prefill: RuntimeVirtualStageManifest[],
  decode: RuntimeVirtualStageManifest[],
): boolean {
  const count = Math.max(prefill.length, decode.length);
  for (let index = 0; index < count; index += 1) {
    const prefillExecution = prefill[index]?.execution;
    const decodeExecution = decode[index]?.execution;
    if (
      (prefillExecution !== undefined || decodeExecution !== undefined) &&
      canonicalJson(prefillExecution ?? null) !== canonicalJson(decodeExecution ?? null)
    ) {
      return true;
    }
  }
  return false;
}

function renderRemoteStageArguments(
  launch: Omit<PythonRemoteStageLaunch, "launchIndex" | "command">,
  configuration: PythonLaunchConfiguration,
): string[] {
  const args = pythonModulePrefix(configuration.stageModule);
  const ramBackedMoe = configuration.ramBackedMoeStages[launch.stageId];
  const pagedKv = configuration.pagedKvStages[launch.stageId];
  if (ramBackedMoe) {
    appendBasicModelArguments(args, ramBackedMoe.snapshotPath, null);
    args.push(
      "--pipeline-snapshot-identity",
      configuration.runtimeModel.snapshotIdentity!,
    );
    appendRamBackedMoeArguments(args, ramBackedMoe);
  } else if (launch.nativeGguf) {
    // The native package is range-local, but the standard model coordinates
    // and pipeline identity remain global and identical on every stage.
    appendModelArguments(args, configuration.runtimeModel);
    args.push(
      "--stage-package-identity",
      `sha256:${launch.nativeGguf.packageId}`,
    );
  } else {
    appendModelArguments(args, configuration.runtimeModel);
  }
  if (pagedKv) appendPagedKvArguments(args, pagedKv);
  args.push(
    "--layer-start",
    String(launch.layerStart),
    "--layer-end",
    String(launch.layerEnd),
    "--total-layers",
    String(launch.totalLayers),
    "--threads",
    String(configuration.threadsPerStage),
    "--listen-host",
    launch.anchor.endpoint.host,
    "--listen-port",
    String(launch.anchor.endpoint.port),
  );
  if (!ramBackedMoe && !launch.cell) {
    args.push("--device", "auto");
    appendDenseTieringArguments(args, launch.macroWave);
  }
  if (launch.downstream) {
    args.push(
      "--next-host",
      launch.downstream.endpoint.host,
      "--next-port",
      String(launch.downstream.endpoint.port),
      "--next-layer-end",
      String(launch.downstream.layerEnd),
    );
  }
  if (launch.cell) {
    args.push(
      "--cell-fixture",
      launch.cell.fixture.path,
      "--cell-world-size",
      String(launch.cell.worldSize),
      "--cell-manifest-sha256",
      launch.cell.fixture.manifestSha256,
      "--cell-collective-backend",
      launch.cell.collectiveBackend,
      "--cell-compute-dtype",
      launch.cell.computeDtype,
      "--cell-operation-timeout-seconds",
      finiteNumber(launch.cell.operationTimeoutSeconds),
    );
    for (const device of launch.cell.rankDevices) {
      args.push("--cell-device", device);
    }
    if (launch.cell.fixture.location === "member-local") {
      const external = launch.cell.external!;
      const reservedPorts = new Set([
        launch.anchor.endpoint.port,
        launch.returnEndpoint.port,
        ...(launch.downstream ? [launch.downstream.endpoint.port] : []),
      ]);
      if (
        reservedPorts.has(external.controlPort) ||
        reservedPorts.has(external.distributedPort)
      ) {
        throw new Error("python_external_cell_ports_conflict_with_pipeline");
      }
      args.push(
        "--cell-mode",
        "external",
        "--cell-control-host",
        external.controlBindHost,
        "--cell-control-port",
        String(external.controlPort),
        "--cell-control-advertise-host",
        external.controlAdvertiseHost,
        "--cell-distributed-advertise-host",
        external.distributedAdvertiseHost,
        "--cell-distributed-port",
        String(external.distributedPort),
        "--cell-startup-timeout-seconds",
        finiteNumber(external.startupTimeoutSeconds),
      );
    }
  }
  if (launch.nativeGguf) {
    args.push(
      "--native-gguf-package",
      launch.nativeGguf.packagePath,
      "--native-gguf-package-id",
      launch.nativeGguf.packageId,
    );
  }
  args.push(
    "--return-host",
    configuration.returnEndpoint.host,
    "--return-port",
    String(configuration.returnEndpoint.port),
    "--codec",
    launch.codec,
    "--sealed-wave-tokens",
    String(launch.sealedWaveTokens),
    "--max-prefill-chunk-tokens",
    String(launch.maxPrefillChunkTokens),
    "--max-speculative-branches",
    String(configuration.maxSpeculativeBranches),
    "--max-speculative-branch-tokens",
    String(configuration.maxSpeculativeBranchTokens),
    "--max-speculative-kv-bytes",
    String(configuration.maxSpeculativeKvBytes),
    "--connect-timeout-seconds",
    finiteNumber(configuration.connectTimeoutSeconds),
  );
  return args;
}

function renderCellMemberArguments(
  launch: Omit<PythonCellMemberLaunch, "launchIndex" | "command">,
  configuration: PythonLaunchConfiguration,
): string[] {
  return [
    ...pythonModulePrefix(MYCELLIOS_CELL_MEMBER_MODULE),
    "--fixture",
    launch.fixturePath,
    "--rank",
    String(launch.rank),
    "--world-size",
    String(launch.worldSize),
    "--pipeline-id",
    launch.pipelineSnapshotIdentity,
    "--cell-collective-backend",
    launch.collectiveBackend,
    "--cell-compute-dtype",
    launch.computeDtype,
    "--cell-device",
    launch.device,
    "--layer-start",
    String(launch.layerStart),
    "--layer-end",
    String(launch.layerEnd),
    "--control-host",
    launch.cellAnchor.controlEndpoint.host,
    "--control-port",
    String(launch.cellAnchor.controlEndpoint.port),
    "--threads",
    String(configuration.threadsPerStage),
    "--connect-timeout-seconds",
    finiteNumber(launch.startupTimeoutSeconds),
    "--operation-timeout-seconds",
    finiteNumber(launch.operationTimeoutSeconds),
  ];
}

function renderRootEngineArguments(
  launch: Omit<PythonRootEngineLaunch, "launchIndex" | "command">,
  configuration: PythonLaunchConfiguration,
): string[] {
  const speculation = pythonSpeculation(launch.decode.speculation);
  const maxActiveSequences = Math.max(
    launch.prefill.microBatchSize,
    launch.decode.microBatchSize,
  );
  if (configuration.maxPendingRequests < maxActiveSequences) {
    throw new Error("python_max_pending_requests_is_too_small");
  }
  const args = pythonModulePrefix(configuration.serverModule);
  const ramBackedMoe = configuration.ramBackedMoeStages[launch.stageId];
  const pagedKv = configuration.pagedKvStages[launch.stageId];
  if (ramBackedMoe) {
    appendBasicModelArguments(args, ramBackedMoe.snapshotPath, null);
    args.push(
      "--pipeline-snapshot-identity",
      configuration.runtimeModel.snapshotIdentity!,
    );
    appendRamBackedMoeArguments(args, ramBackedMoe);
  } else {
    appendModelArguments(args, configuration.runtimeModel);
    if (launch.nativeGguf) {
      args.push(
        "--stage-package-identity",
        `sha256:${launch.nativeGguf.packageId}`,
        "--native-gguf-package",
        launch.nativeGguf.packagePath,
        "--native-gguf-package-id",
        launch.nativeGguf.packageId,
      );
    }
  }
  if (configuration.draftModel) {
    appendDraftModelArguments(args, configuration.draftModel);
  }
  if (pagedKv) appendPagedKvArguments(args, pagedKv);
  args.push(
    "--public-model-name",
    configuration.publicModelName,
    "--host",
    configuration.apiEndpoint.host,
    "--port",
    String(configuration.apiEndpoint.port),
    "--boundaries",
    launch.boundaries.join(","),
    "--codec",
    launch.codec,
    "--threads-per-stage",
    String(configuration.threadsPerStage),
    "--max-batch-size",
    String(launch.decode.microBatchSize),
    "--max-active-sequences",
    String(maxActiveSequences),
    "--max-pending-requests",
    String(configuration.maxPendingRequests),
    "--batch-window-ms",
    finiteNumber(configuration.batchWindowMs),
    "--prefill-chunk-tokens",
    String(launch.prefill.chunkTokens),
    "--prefill-inflight-chunks",
    String(configuration.prefillInflightChunks),
    "--prefill-inflight-bytes",
    String(configuration.prefillInflightBytes),
    "--max-speculative-branches",
    String(configuration.maxSpeculativeBranches),
    "--max-speculative-branch-tokens",
    String(configuration.maxSpeculativeBranchTokens),
    "--max-speculative-kv-bytes",
    String(configuration.maxSpeculativeKvBytes),
    "--sealed-wave-tokens",
    String(launch.sealedWaveTokens),
    "--max-prefill-chunk-tokens",
    String(launch.maxPrefillChunkTokens),
    "--speculation",
    speculation.provider,
    "--speculative-max-draft-tokens",
    String(speculation.maxDraftTokens),
    "--speculation-minimum-speedup",
    finiteNumber(configuration.speculationMinimumSpeedup),
    "--max-output-tokens",
    String(configuration.maxOutputTokens),
    "--max-retained-sessions",
    String(configuration.maxRetainedSessions),
    "--max-retained-session-tokens",
    String(configuration.maxRetainedSessionTokens),
    "--retained-session-ttl-seconds",
    finiteNumber(configuration.retainedSessionTtlSeconds),
    "--first-stage-host",
    launch.firstRemoteStage.endpoint.host,
    "--first-stage-port",
    String(launch.firstRemoteStage.endpoint.port),
    "--return-bind-host",
    configuration.returnBindHost,
    "--return-advertise-host",
    configuration.returnEndpoint.host,
    "--return-port",
    String(configuration.returnEndpoint.port),
    "--startup-timeout-seconds",
    finiteNumber(configuration.connectTimeoutSeconds),
    "--socket-timeout-seconds",
    finiteNumber(configuration.connectTimeoutSeconds),
  );
  if (configuration.speculativeInflightWaves !== undefined) {
    args.push(
      "--speculative-inflight-waves",
      String(configuration.speculativeInflightWaves),
      "--speculative-inflight-bytes",
      String(configuration.speculativeInflightBytes),
    );
  }
  if (configuration.recovery) {
    args.push(
      "--recovery-max-retries",
      String(configuration.recovery.maxRetries),
    );
    for (const executorId of configuration.recovery.stageExecutorIds) {
      args.push("--stage-executor-id", executorId);
    }
    for (const route of configuration.recovery.standbyRoutes) {
      args.push("--recovery-standby-route", canonicalJson(route));
    }
  }
  if (!ramBackedMoe) args.push("--device", "auto");
  if (!ramBackedMoe) appendDenseTieringArguments(args, launch.macroWave);
  return args;
}

function appendDenseTieringArguments(
  args: string[],
  contract: MacroWaveStageExecutionContractV1 | null,
): void {
  if (
    contract === null
    || contract.residentKind !== "layers"
    || contract.ramArtifact !== undefined
  ) return;
  args.push(
    "--dense-host-ram-budget-bytes",
    String(contract.budgets.hostRamBytes),
    "--dense-vram-budget-bytes",
    String(contract.budgets.vramBytes),
    "--dense-activation-reserve-bytes",
    String(contract.requirements.activationBufferBytes),
  );
  if (contract.memoryMode === "resident") {
    args.push("--dense-require-full-residency");
  }
}

function appendPagedKvArguments(
  args: string[],
  contract: PythonPagedKvStageConfiguration,
): void {
  args.push(
    "--paged-kv",
    "--paged-device",
    contract.device,
    "--paged-attention-backend",
    contract.attentionBackend,
    "--paged-block-size",
    String(contract.blockSize),
    "--paged-num-blocks",
    String(contract.numBlocks),
    "--paged-max-batch-tokens",
    String(contract.maxBatchTokens),
    "--paged-max-active-requests",
    String(contract.maxActiveRequests),
    "--paged-max-sequence-tokens",
    String(contract.maxSequenceTokens),
    "--paged-cpu-spill-bytes",
    String(contract.cpuSpillBytes),
  );
}

function pythonSpeculation(policy: RuntimeSpeculationPolicy): PythonSpeculationArguments {
  const selected = policy.strategies.find(
    (strategy) => strategy.id === policy.defaultStrategyId,
  );
  if (!selected) throw new Error("python_speculation_default_is_missing");
  if (policy.mode === "disabled" || selected.kind === "autoregressive") {
    return {
      provider: "off",
      maxDraftTokens: 1,
      maxBranches: 0,
      maxBranchTokens: 0,
      maxKvBytes: 0,
      maxWaveTokens: 1,
      artifactId: null,
      parameterBytes: 0,
      memoryReservationBytes: 0,
    };
  }
  if (
    selected.kind !== "ngram"
    && selected.kind !== "draft-tree"
    && selected.kind !== "draft-model"
  ) {
    throw new Error(`python_speculation_strategy_not_supported:${selected.kind}`);
  }
  if (selected.maxDraftTokens > 16) {
    throw new Error("python_speculation_draft_length_exceeds_runtime_limit");
  }
  if (selected.kind === "ngram") {
    return {
      provider: "ngram",
      maxDraftTokens: selected.maxDraftTokens,
      maxBranches: 0,
      maxBranchTokens: 0,
      maxKvBytes: 0,
      maxWaveTokens: selected.maxDraftTokens + 1,
      artifactId: null,
      parameterBytes: 0,
      memoryReservationBytes: 0,
    };
  }
  if (selected.kind === "draft-model") {
    if (selected.artifactId === undefined) {
      throw new Error("python_draft_model_artifact_is_missing");
    }
    return {
      provider: "draft-model",
      maxDraftTokens: selected.maxDraftTokens,
      maxBranches: 0,
      maxBranchTokens: 0,
      maxKvBytes: 0,
      maxWaveTokens: selected.maxDraftTokens + 1,
      artifactId: selected.artifactId,
      parameterBytes: boundedInteger(
        selected.parameterBytes,
        1,
        Number.MAX_SAFE_INTEGER,
        "python_draft_model_parameter_bytes_are_invalid",
      ),
      memoryReservationBytes: boundedInteger(
        selected.memoryReservationBytes,
        1,
        Number.MAX_SAFE_INTEGER,
        "python_draft_model_memory_reservation_is_invalid",
      ),
    };
  }
  if (
    selected.maxBranches === undefined ||
    selected.maxBranchTokens === undefined ||
    selected.maxKvBytes === undefined ||
    selected.maxWaveTokens === undefined
  ) {
    throw new Error("python_draft_tree_limits_are_missing");
  }
  if (selected.maxWaveTokens !== selected.maxDraftTokens + 1) {
    throw new Error("python_draft_tree_wave_does_not_match_draft_depth");
  }
  return {
    provider: "draft-tree",
    maxDraftTokens: selected.maxDraftTokens,
    maxBranches: selected.maxBranches,
    maxBranchTokens: selected.maxBranchTokens,
    maxKvBytes: selected.maxKvBytes,
    maxWaveTokens: selected.maxWaveTokens,
    artifactId: null,
    parameterBytes: 0,
    memoryReservationBytes: 0,
  };
}

function normalizeConfiguration(
  manifest: RuntimePipelineManifestV2,
  value: PythonLaunchCompilerOptions,
  requireNormalized = false,
): PythonLaunchConfiguration {
  if (!isRecord(value)) throw new Error("python_launch_options_must_be_an_object");
  assertExactKeys(
    value,
    ["apiEndpoint", "returnEndpoint", "returnBindHost"],
    [
      "runtimeModel",
      "draftModel",
      "publicModelName",
      "pythonExecutable",
      "stageModule",
      "serverModule",
      "threadsPerStage",
      "connectTimeoutSeconds",
      "batchWindowMs",
      "prefillInflightChunks",
      "prefillInflightBytes",
      "maxSpeculativeBranches",
      "maxSpeculativeBranchTokens",
      "maxSpeculativeKvBytes",
      "speculativeInflightWaves",
      "speculativeInflightBytes",
      "maxPendingRequests",
      "maxOutputTokens",
      "speculationMinimumSpeedup",
      "maxRetainedSessions",
      "maxRetainedSessionTokens",
      "retainedSessionTtlSeconds",
      "recovery",
      "nativeGgufStages",
      "ramBackedMoeStages",
      "pagedKvStages",
      "executorIsolation",
    ],
    "python_launch_options_have_unknown_or_missing_fields",
  );
  const hasUncalibratedTensorParallel = [
    manifest.plans.prefill,
    manifest.plans.decode,
  ].some(
    (plan) =>
      plan.predicted.calibrationRequired === true
      && (plan.predicted.calibrationReasons ?? []).some(
        (reason) =>
          reason === "tensor_parallel_collective_cost_unprofiled"
          || reason === "external_cell_internal_links_unprofiled"
          || reason === "collective_link_missing"
          || reason.startsWith("tp_cell_"),
      )
      && plan.stages.some(
        (stage) => stage.execution?.mode === "tensor-parallel-cell",
      ),
  );
  if (hasUncalibratedTensorParallel) {
    throw new Error(
      "python_tensor_parallel_requires_measured_collective_profile",
    );
  }
  validateEndpoint(value.apiEndpoint, "python_api_endpoint_is_invalid");
  validateEndpoint(value.returnEndpoint, "python_return_endpoint_is_invalid");
  const returnBindHost = safeString(
    value.returnBindHost,
    "python_return_bind_host_is_invalid",
  );
  if (value.apiEndpoint.port === value.returnEndpoint.port) {
    throw new Error("python_api_and_return_ports_conflict");
  }
  const runtimeModelValue = value.runtimeModel;
  if (runtimeModelValue !== undefined && !isRecord(runtimeModelValue)) {
    throw new Error("python_runtime_model_is_invalid");
  }
  const source = safeString(
    runtimeModelValue?.source ?? manifest.modelId,
    "python_runtime_model_source_is_invalid",
  );
  const revisionValue = runtimeModelValue?.revision;
  const revision =
    revisionValue === undefined || revisionValue === null
      ? null
      : safeString(revisionValue, "python_runtime_model_revision_is_invalid");
  const derivedSnapshotIdentity = deriveHubSnapshotIdentity(source, revision);
  const explicitSnapshotIdentity = runtimeModelValue?.snapshotIdentity;
  const snapshotIdentity =
    explicitSnapshotIdentity === undefined
      ? derivedSnapshotIdentity
      : uint64String(
          explicitSnapshotIdentity,
          "python_runtime_model_snapshot_identity_is_invalid",
        );
  if (
    snapshotIdentity !== undefined &&
    derivedSnapshotIdentity !== undefined &&
    snapshotIdentity !== derivedSnapshotIdentity
  ) {
    throw new Error("python_runtime_model_snapshot_identity_mismatch");
  }
  const artifactCoordinates = normalizeModelArtifactCoordinates(
    runtimeModelValue,
    source,
    revision,
    snapshotIdentity,
    requireNormalized,
  );
  const runtimeModel: PythonRuntimeModelSource = {
    source,
    revision,
    ...(snapshotIdentity ? { snapshotIdentity } : {}),
    ...artifactCoordinates,
  };
  const nativeGgufStages = normalizeNativeGgufStages(
    manifest,
    value.nativeGgufStages,
    runtimeModel,
  );
  const ramBackedMoeStages = normalizeRamBackedMoeStages(
    manifest,
    value.ramBackedMoeStages,
    runtimeModel,
  );
  const pagedKvStages = normalizePagedKvStages(
    manifest,
    value.pagedKvStages,
  );
  validateExclusiveStageBindings(
    nativeGgufStages,
    ramBackedMoeStages,
    pagedKvStages,
  );
  const prefillInflightChunks = boundedInteger(
    value.prefillInflightChunks ??
      Math.min(64, manifest.plans.prefill.stages.length),
    1,
    64,
    "python_prefill_inflight_chunks_is_invalid",
  );
  const prefillInflightBytes = boundedInteger(
    value.prefillInflightBytes ?? DEFAULT_PREFILL_INFLIGHT_BYTES,
    1,
    MAX_PREFILL_INFLIGHT_BYTES,
    "python_prefill_inflight_bytes_is_invalid",
  );
  const minimumPrefillFrameBytes = pythonPrefillFrameByteReservation(
    manifest.plans.prefill.activationCodec,
    manifest.plans.prefill.chunkTokens,
    manifest.hiddenSize,
  );
  if (BigInt(prefillInflightBytes) < minimumPrefillFrameBytes) {
    throw new Error(
      `python_prefill_inflight_bytes_below_frame_reservation:${minimumPrefillFrameBytes}`,
    );
  }
  const speculation = pythonSpeculation(manifest.plans.decode.speculation);
  const draftModel = normalizeDraftModel(
    value.draftModel,
    speculation,
    requireNormalized,
  );
  assertDraftModelCapacity(manifest, draftModel);
  const requestedMaxSpeculativeBranches = boundedInteger(
    value.maxSpeculativeBranches ?? speculation.maxBranches,
    0,
    MAX_SPECULATIVE_BRANCHES,
    "python_max_speculative_branches_is_invalid",
  );
  const requestedMaxSpeculativeBranchTokens = boundedInteger(
    value.maxSpeculativeBranchTokens ?? speculation.maxBranchTokens,
    0,
    MAX_SPECULATIVE_BRANCH_TOKENS,
    "python_max_speculative_branch_tokens_is_invalid",
  );
  const requestedMaxSpeculativeKvBytes = boundedInteger(
    value.maxSpeculativeKvBytes ?? speculation.maxKvBytes,
    0,
    MAX_SPECULATIVE_KV_BYTES,
    "python_max_speculative_kv_bytes_is_invalid",
  );
  if (
    speculation.provider === "draft-tree" &&
    (
      requestedMaxSpeculativeBranches !== speculation.maxBranches ||
      requestedMaxSpeculativeBranchTokens !== speculation.maxBranchTokens ||
      requestedMaxSpeculativeKvBytes !== speculation.maxKvBytes
    )
  ) {
    throw new Error("python_draft_tree_launch_limits_do_not_match_manifest");
  }
  const maxSpeculativeBranches = requestedMaxSpeculativeBranches;
  const maxSpeculativeBranchTokens = requestedMaxSpeculativeBranchTokens;
  const maxSpeculativeKvBytes = requestedMaxSpeculativeKvBytes;
  const speculativeInflightWaves = boundedInteger(
    value.speculativeInflightWaves ?? DEFAULT_SPECULATIVE_INFLIGHT_WAVES,
    1,
    MAX_SPECULATIVE_INFLIGHT_WAVES,
    "python_speculative_inflight_waves_is_invalid",
  );
  const speculativeInflightBytes = boundedInteger(
    value.speculativeInflightBytes ?? DEFAULT_SPECULATIVE_INFLIGHT_BYTES,
    0,
    MAX_SPECULATIVE_INFLIGHT_BYTES,
    "python_speculative_inflight_bytes_is_invalid",
  );
  const speculativeConveyorEnabled =
    speculativeInflightWaves > DEFAULT_SPECULATIVE_INFLIGHT_WAVES
    || speculativeInflightBytes > DEFAULT_SPECULATIVE_INFLIGHT_BYTES;
  if (
    speculativeConveyorEnabled
    && (
      speculativeInflightWaves <= DEFAULT_SPECULATIVE_INFLIGHT_WAVES
      || speculativeInflightBytes <= DEFAULT_SPECULATIVE_INFLIGHT_BYTES
    )
  ) {
    throw new Error(
      "python_speculative_conveyor_limits_must_be_disabled_or_complete",
    );
  }
  if (
    speculativeConveyorEnabled
    && speculation.provider !== "ngram"
    && speculation.provider !== "draft-model"
  ) {
    throw new Error(
      "python_speculative_conveyor_requires_linear_speculation",
    );
  }
  if (
    speculativeConveyorEnabled
    && (
      maxSpeculativeBranches > 0
      || maxSpeculativeBranchTokens > 0
      || maxSpeculativeKvBytes > 0
    )
  ) {
    throw new Error(
      "python_speculative_conveyor_cannot_use_physical_tree_limits",
    );
  }
  if (
    speculativeConveyorEnabled
    && Math.max(
      manifest.plans.prefill.microBatchSize,
      manifest.plans.decode.microBatchSize,
    ) !== 1
  ) {
    throw new Error(
      "python_speculative_conveyor_requires_single_active_sequence",
    );
  }
  if (speculativeConveyorEnabled) {
    const minimumVerifyFrameBytes = pythonPrefillFrameByteReservation(
      manifest.plans.decode.activationCodec,
      speculation.maxDraftTokens + 1,
      manifest.hiddenSize,
    );
    if (BigInt(speculativeInflightBytes) < minimumVerifyFrameBytes) {
      throw new Error(
        `python_speculative_inflight_bytes_below_frame_reservation:${minimumVerifyFrameBytes}`,
      );
    }
  }
  const maxRetainedSessions = boundedInteger(
    value.maxRetainedSessions ?? DEFAULT_RETAINED_SESSIONS,
    0,
    10_000,
    "python_max_retained_sessions_is_invalid",
  );
  const maxRetainedSessionTokens = boundedInteger(
    value.maxRetainedSessionTokens ?? DEFAULT_RETAINED_SESSION_TOKENS,
    0,
    100_000_000,
    "python_max_retained_session_tokens_is_invalid",
  );
  const retainedSessionTtlSeconds = positiveFinite(
    value.retainedSessionTtlSeconds ?? DEFAULT_RETAINED_SESSION_TTL_SECONDS,
    "python_retained_session_ttl_is_invalid",
  );
  const speculativeTreeLimitsEnabled = [
    maxSpeculativeBranches,
    maxSpeculativeBranchTokens,
    maxSpeculativeKvBytes,
  ].map((limit) => limit > 0);
  if (
    speculativeTreeLimitsEnabled.some(Boolean) &&
    !speculativeTreeLimitsEnabled.every(Boolean)
  ) {
    throw new Error("python_speculative_tree_limits_must_be_disabled_or_complete");
  }
  validatePagedKvLaunchCapacity(
    manifest,
    pagedKvStages,
    maxSpeculativeBranches,
    maxSpeculativeBranchTokens,
    maxRetainedSessions,
  );
  const recovery = normalizeRecoveryConfiguration(manifest, value.recovery);
  const executorIsolation = normalizeExecutorIsolationPolicy(
    value.executorIsolation,
  );
  const normalized: PythonLaunchConfiguration = {
    apiEndpoint: { ...value.apiEndpoint },
    returnEndpoint: { ...value.returnEndpoint },
    returnBindHost,
    runtimeModel,
    draftModel,
    publicModelName: safeString(
      value.publicModelName ?? manifest.modelId,
      "python_public_model_name_is_invalid",
    ),
    pythonExecutable: safeString(
      value.pythonExecutable ?? "python",
      "python_executable_is_invalid",
    ),
    stageModule: nativePythonModule(
      value.stageModule,
      MYCELLIOS_STAGE_MODULE,
      "python_stage_module_must_be_mycellios_native",
    ),
    serverModule: nativePythonModule(
      value.serverModule,
      MYCELLIOS_SERVER_MODULE,
      "python_server_module_must_be_mycellios_native",
    ),
    threadsPerStage: boundedInteger(
      value.threadsPerStage ?? 1,
      1,
      1_024,
      "python_threads_per_stage_is_invalid",
    ),
    connectTimeoutSeconds: positiveFinite(
      value.connectTimeoutSeconds ?? 180,
      "python_connect_timeout_is_invalid",
    ),
    batchWindowMs: nonNegativeFinite(
      value.batchWindowMs ?? 2,
      "python_batch_window_is_invalid",
    ),
    prefillInflightChunks,
    prefillInflightBytes,
    maxSpeculativeBranches,
    maxSpeculativeBranchTokens,
    maxSpeculativeKvBytes,
    ...(speculativeConveyorEnabled
      ? { speculativeInflightWaves, speculativeInflightBytes }
      : {}),
    maxPendingRequests: boundedInteger(
      value.maxPendingRequests ?? 128,
      1,
      1_000_000,
      "python_max_pending_requests_is_invalid",
    ),
    maxOutputTokens: boundedInteger(
      value.maxOutputTokens ?? 512,
      1,
      1_000_000,
      "python_max_output_tokens_is_invalid",
    ),
    speculationMinimumSpeedup: atLeastOneFinite(
      value.speculationMinimumSpeedup ?? 1.05,
      "python_speculation_speedup_is_invalid",
    ),
    maxRetainedSessions,
    maxRetainedSessionTokens,
    retainedSessionTtlSeconds,
    recovery,
    nativeGgufStages,
    ramBackedMoeStages,
    pagedKvStages,
    executorIsolation,
  };
  if (requireNormalized && canonicalJson(value) !== canonicalJson(normalized)) {
    throw new Error("python_launch_configuration_is_not_normalized");
  }
  return normalized;
}

function normalizeRecoveryConfiguration(
  manifest: RuntimePipelineManifestV2,
  value: unknown,
): PythonRecoveryConfiguration | null {
  if (value === undefined || value === null) return null;
  if (!isRecord(value)) {
    throw new Error("python_recovery_configuration_is_invalid");
  }
  assertExactKeys(
    value,
    ["maxRetries", "stageExecutorIds", "standbyRoutes"],
    [],
    "python_recovery_configuration_keys_are_invalid",
  );
  const stageCount = manifest.plans.prefill.stages.length;
  const stageExecutorIds = recoveryExecutorIds(
    value.stageExecutorIds,
    stageCount,
    "python_recovery_primary_executor_contract_is_invalid",
  );
  const maxRetries = boundedInteger(
    value.maxRetries,
    1,
    100,
    "python_recovery_max_retries_is_invalid",
  );
  if (!Array.isArray(value.standbyRoutes) || value.standbyRoutes.length === 0) {
    throw new Error("python_recovery_requires_at_least_one_standby_route");
  }

  const primaryFirstStage = manifest.plans.prefill.stages[1]!.anchor.endpoint;
  const standbyRoutes: PythonRecoveryStandbyRouteConfiguration[] =
    value.standbyRoutes.map((candidate, index) => {
    if (!isRecord(candidate)) {
      throw new Error(`python_recovery_standby_is_invalid:${index}`);
    }
    assertExactKeys(
      candidate,
      ["firstStage", "stageExecutorIds"],
      ["schema", "routeId"],
      `python_recovery_standby_keys_are_invalid:${index}`,
    );
    if (
      candidate.schema !== undefined &&
      candidate.schema !== RECOVERY_STANDBY_SCHEMA
    ) {
      throw new Error(`python_recovery_standby_schema_is_invalid:${index}`);
    }
    validateEndpoint(
      candidate.firstStage,
      `python_recovery_standby_endpoint_is_invalid:${index}`,
    );
    const firstStage = { ...candidate.firstStage };
    if (endpointKey(firstStage) === endpointKey(primaryFirstStage)) {
      throw new Error(`python_recovery_standby_reuses_primary_endpoint:${index}`);
    }
    const executorIds = recoveryExecutorIds(
      candidate.stageExecutorIds,
      stageCount,
      `python_recovery_standby_executor_contract_is_invalid:${index}`,
    );
    if (canonicalJson(executorIds) !== canonicalJson(stageExecutorIds)) {
      throw new Error(`python_recovery_standby_executor_contract_mismatch:${index}`);
    }
    const routeId = `standby-${digest(
      canonicalJson({
        schema: RECOVERY_STANDBY_SCHEMA,
        firstStage,
        stageExecutorIds: executorIds,
      }),
      24,
    )}`;
    if (
      candidate.routeId !== undefined &&
      safeString(
        candidate.routeId,
        `python_recovery_standby_route_id_is_invalid:${index}`,
      ) !== routeId
    ) {
      throw new Error(`python_recovery_standby_route_id_mismatch:${index}`);
    }
    return {
      schema: RECOVERY_STANDBY_SCHEMA,
      routeId,
      firstStage,
      stageExecutorIds: executorIds,
    };
  });

  if (
    new Set(standbyRoutes.map((route) => endpointKey(route.firstStage))).size !==
    standbyRoutes.length
  ) {
    throw new Error("python_recovery_standby_endpoints_must_be_unique");
  }
  return { maxRetries, stageExecutorIds, standbyRoutes };
}

function recoveryExecutorIds(
  value: unknown,
  stageCount: number,
  error: string,
): string[] {
  if (!Array.isArray(value) || value.length !== stageCount) {
    throw new Error(error);
  }
  return value.map((executorId) => {
    if (
      typeof executorId !== "string" ||
      !/^[0-9a-f]{32}$/.test(executorId)
    ) {
      throw new Error(error);
    }
    return executorId;
  });
}

function normalizeNativeGgufStages(
  manifest: RuntimePipelineManifestV2,
  value: unknown,
  runtimeModel: PythonRuntimeModelSource,
): Record<string, PythonNativeGgufStageConfiguration> {
  if (value === undefined) return {};
  if (!isRecord(value)) {
    throw new Error("python_native_gguf_stages_must_be_an_object");
  }
  if (Object.keys(value).length === 0) return {};
  if (runtimeModel.snapshotIdentity === undefined) {
    throw new Error("python_native_gguf_requires_pipeline_snapshot_identity");
  }
  if (
    runtimeModel.artifactIdentity === undefined ||
    runtimeModel.canonicalSource === undefined ||
    runtimeModel.canonicalRevision === undefined
  ) {
    throw new Error("python_native_gguf_requires_global_model_coordinates");
  }

  const entries: Array<[string, PythonNativeGgufStageConfiguration]> = [];
  for (const stageIdValue of Object.keys(value).sort()) {
    const stageId = safeString(
      stageIdValue,
      "python_native_gguf_stage_id_is_invalid",
    );
    const input = value[stageId];
    if (!isRecord(input)) {
      throw new Error(`python_native_gguf_stage_configuration_is_invalid:${stageId}`);
    }
    assertExactKeys(
      input,
      [
        "packagePath",
        "packageId",
        "modelIdentity",
        "modelSource",
        "modelRevision",
        "layerStart",
        "layerEnd",
        "totalLayers",
      ],
      [],
      `python_native_gguf_stage_configuration_keys_are_invalid:${stageId}`,
    );
    const normalized: PythonNativeGgufStageConfiguration = {
      packagePath: safeString(
        input.packagePath,
        `python_native_gguf_package_path_is_invalid:${stageId}`,
      ),
      packageId: sha256Digest(
        input.packageId,
        `python_native_gguf_package_id_is_invalid:${stageId}`,
      ),
      modelIdentity: sha256Identity(
        input.modelIdentity,
        `python_native_gguf_model_identity_is_invalid:${stageId}`,
      ),
      modelSource: safeString(
        input.modelSource,
        `python_native_gguf_model_source_is_invalid:${stageId}`,
      ),
      modelRevision: nullableSafeString(
        input.modelRevision,
        `python_native_gguf_model_revision_is_invalid:${stageId}`,
      ),
      layerStart: boundedInteger(
        input.layerStart,
        0,
        manifest.totalLayers,
        `python_native_gguf_layer_start_is_invalid:${stageId}`,
      ),
      layerEnd: boundedInteger(
        input.layerEnd,
        1,
        manifest.totalLayers,
        `python_native_gguf_layer_end_is_invalid:${stageId}`,
      ),
      totalLayers: boundedInteger(
        input.totalLayers,
        1,
        manifest.totalLayers,
        `python_native_gguf_total_layers_is_invalid:${stageId}`,
      ),
    };
    validateNativeGgufStageBinding(manifest, stageId, normalized, runtimeModel);
    entries.push([stageId, normalized]);
  }
  return Object.fromEntries(entries);
}

function validateNativeGgufStageBinding(
  manifest: RuntimePipelineManifestV2,
  stageId: string,
  configuration: PythonNativeGgufStageConfiguration,
  runtimeModel: PythonRuntimeModelSource,
): void {
  const stages = [manifest.plans.prefill, manifest.plans.decode].map((plan) =>
    plan.stages.find((stage) => stage.stageId === stageId),
  );
  if (stages.some((stage) => stage === undefined)) {
    throw new Error(`python_native_gguf_stage_is_not_in_both_phases:${stageId}`);
  }
  for (const stage of stages as RuntimeVirtualStageManifest[]) {
    if (stage.execution !== undefined || stage.macroWave !== undefined) {
      throw new Error(`python_native_gguf_stage_has_conflicting_execution:${stageId}`);
    }
    if (stage.members.length !== 1) {
      throw new Error(`python_native_gguf_stage_requires_single_member:${stageId}`);
    }
    if (
      configuration.layerStart !== stage.layerStart ||
      configuration.layerEnd !== stage.layerEnd ||
      configuration.totalLayers !== manifest.totalLayers
    ) {
      throw new Error(`python_native_gguf_stage_range_mismatch:${stageId}`);
    }
  }
  if (configuration.modelIdentity !== runtimeModel.artifactIdentity) {
    throw new Error(`python_native_gguf_model_identity_mismatch:${stageId}`);
  }
  if (configuration.modelSource !== runtimeModel.canonicalSource) {
    throw new Error(`python_native_gguf_model_source_mismatch:${stageId}`);
  }
  if (configuration.modelRevision !== runtimeModel.canonicalRevision) {
    throw new Error(`python_native_gguf_model_revision_mismatch:${stageId}`);
  }
}

function normalizePagedKvStages(
  manifest: RuntimePipelineManifestV2,
  value: unknown,
): Record<string, PythonPagedKvStageConfiguration> {
  if (value === undefined) return {};
  if (!isRecord(value)) {
    throw new Error("python_paged_kv_stages_must_be_an_object");
  }
  const stages = manifest.plans.prefill.stages;
  const expectedIds = new Set(stages.map((stage) => stage.stageId));
  const suppliedIds = Object.keys(value);
  if (suppliedIds.length === 0) return {};
  for (const stageId of suppliedIds) {
    if (!expectedIds.has(stageId)) {
      throw new Error(`python_paged_kv_stage_binding_is_unexpected:${stageId}`);
    }
  }
  for (const stage of stages) {
    if (!Object.hasOwn(value, stage.stageId)) {
      throw new Error(`python_paged_kv_stage_binding_is_missing:${stage.stageId}`);
    }
  }

  const entries: Array<[string, PythonPagedKvStageConfiguration]> = [];
  for (const stage of stages) {
    const stageId = stage.stageId;
    const input = value[stageId];
    if (!isRecord(input)) {
      throw new Error(`python_paged_kv_stage_configuration_is_invalid:${stageId}`);
    }
    assertExactKeys(
      input,
      [
        "schema",
        "device",
        "attentionBackend",
        "blockSize",
        "numBlocks",
        "maxBatchTokens",
        "maxActiveRequests",
        "maxSequenceTokens",
        "cpuSpillBytes",
      ],
      [],
      `python_paged_kv_stage_configuration_keys_are_invalid:${stageId}`,
    );
    if (input.schema !== PAGED_STAGE_RUNTIME_SCHEMA) {
      throw new Error(`python_paged_kv_stage_schema_is_invalid:${stageId}`);
    }
    const device = safeString(
      input.device,
      `python_paged_kv_device_is_invalid:${stageId}`,
    );
    if (!/^(?:cpu|cuda(?::(?:0|[1-9]\d*))?)$/.test(device)) {
      throw new Error(`python_paged_kv_device_is_invalid:${stageId}`);
    }
    if (device.startsWith("cuda") && device !== "cuda" && device !== "cuda:0") {
      throw new Error(`python_paged_kv_device_index_is_not_verified:${stageId}`);
    }
    if (input.attentionBackend !== "eager" && input.attentionBackend !== "sdpa") {
      throw new Error(`python_paged_kv_attention_backend_is_invalid:${stageId}`);
    }
    const blockSize = boundedInteger(
      input.blockSize,
      4,
      MAX_PAGED_BLOCK_SIZE,
      `python_paged_kv_block_size_is_invalid:${stageId}`,
    );
    const numBlocks = boundedInteger(
      input.numBlocks,
      1,
      MAX_PAGED_NUM_BLOCKS,
      `python_paged_kv_num_blocks_is_invalid:${stageId}`,
    );
    const maxBatchTokens = boundedInteger(
      input.maxBatchTokens,
      1,
      MAX_PAGED_BATCH_TOKENS,
      `python_paged_kv_max_batch_tokens_is_invalid:${stageId}`,
    );
    const maxActiveRequests = boundedInteger(
      input.maxActiveRequests,
      1,
      MAX_PAGED_ACTIVE_REQUESTS,
      `python_paged_kv_max_active_requests_is_invalid:${stageId}`,
    );
    const maxSequenceTokens = boundedInteger(
      input.maxSequenceTokens,
      1,
      MAX_PAGED_SEQUENCE_TOKENS,
      `python_paged_kv_max_sequence_tokens_is_invalid:${stageId}`,
    );
    const cpuSpillBytes = boundedInteger(
      input.cpuSpillBytes,
      0,
      MAX_SPECULATIVE_KV_BYTES,
      `python_paged_kv_cpu_spill_bytes_is_invalid:${stageId}`,
    );
    if (Math.ceil(maxSequenceTokens / blockSize) > numBlocks) {
      throw new Error(`python_paged_kv_sequence_exceeds_pool:${stageId}`);
    }
    if (cpuSpillBytes > 0 && device === "cpu") {
      throw new Error(`python_paged_kv_cpu_spill_requires_gpu:${stageId}`);
    }

    const phaseStages = [manifest.plans.prefill, manifest.plans.decode].map(
      (plan) => plan.stages.find((candidate) => candidate.stageId === stageId),
    );
    if (phaseStages.some((candidate) => candidate === undefined)) {
      throw new Error(`python_paged_kv_stage_is_not_in_both_phases:${stageId}`);
    }
    for (const phaseStage of phaseStages as RuntimeVirtualStageManifest[]) {
      if (phaseStage.execution !== undefined || phaseStage.macroWave !== undefined) {
        throw new Error(`python_paged_kv_stage_has_conflicting_execution:${stageId}`);
      }
      if (phaseStage.members.length !== 1) {
        throw new Error(`python_paged_kv_stage_requires_single_member:${stageId}`);
      }
      const member = phaseStage.members[0]!;
      if (member.backend.engine !== "python-transformers") {
        throw new Error(`python_paged_kv_stage_engine_is_not_supported:${stageId}`);
      }
      if (!member.backend.modelFormats.includes("safetensors")) {
        throw new Error(`python_paged_kv_stage_model_format_is_not_supported:${stageId}`);
      }
      if (!member.backend.executionModes.includes("layer-range")) {
        throw new Error(`python_paged_kv_stage_execution_mode_is_not_supported:${stageId}`);
      }
      if (device === "cpu") {
        if (!member.capabilities.deviceKinds.includes("cpu")) {
          throw new Error(`python_paged_kv_stage_lacks_cpu_capability:${stageId}`);
        }
      } else {
        const capabilities = member.capabilities;
        if (
          !capabilities.deviceKinds.includes("gpu")
          || !capabilities.computeApis.some(
            (api) => api === "cuda" || api === "rocm",
          )
        ) {
          throw new Error(`python_paged_kv_stage_lacks_gpu_capability:${stageId}`);
        }
      }
    }
    entries.push([
      stageId,
      {
        schema: PAGED_STAGE_RUNTIME_SCHEMA,
        device,
        attentionBackend: input.attentionBackend,
        blockSize,
        numBlocks,
        maxBatchTokens,
        maxActiveRequests,
        maxSequenceTokens,
        cpuSpillBytes,
      },
    ]);
  }
  return Object.fromEntries(entries.sort(([left], [right]) => left.localeCompare(right)));
}

function validatePagedKvLaunchCapacity(
  manifest: RuntimePipelineManifestV2,
  stages: Record<string, PythonPagedKvStageConfiguration>,
  maxSpeculativeBranches: number,
  maxSpeculativeBranchTokens: number,
  maxRetainedSessions: number,
): void {
  if (Object.keys(stages).length === 0) return;
  const maxActiveSequences = Math.max(
    manifest.plans.prefill.microBatchSize,
    manifest.plans.decode.microBatchSize,
  );
  const requiredRequests =
    maxActiveSequences + maxSpeculativeBranches + maxRetainedSessions;
  const speculation = pythonSpeculation(manifest.plans.decode.speculation);
  const requiredBatchTokens = Math.max(
    manifest.plans.prefill.chunkTokens,
    speculation.maxWaveTokens,
  );
  for (const [stageId, configuration] of Object.entries(stages)) {
    if (configuration.maxActiveRequests < requiredRequests) {
      throw new Error(`python_paged_kv_active_request_capacity_is_too_small:${stageId}`);
    }
    if (configuration.maxBatchTokens < requiredBatchTokens) {
      throw new Error(`python_paged_kv_batch_token_capacity_is_too_small:${stageId}`);
    }
    if (
      maxSpeculativeBranchTokens > 0
      && configuration.maxSequenceTokens < maxSpeculativeBranchTokens
    ) {
      throw new Error(`python_paged_kv_sequence_capacity_is_too_small:${stageId}`);
    }
  }
}

function validateExclusiveStageBindings(
  nativeGgufStages: Record<string, PythonNativeGgufStageConfiguration>,
  ramBackedMoeStages: Record<string, PythonRamBackedMoeStageConfiguration>,
  pagedKvStages: Record<string, PythonPagedKvStageConfiguration>,
): void {
  for (const stageId of new Set([
    ...Object.keys(nativeGgufStages),
    ...Object.keys(ramBackedMoeStages),
    ...Object.keys(pagedKvStages),
  ])) {
    const hasNativeGguf = Object.hasOwn(nativeGgufStages, stageId);
    const hasRamBackedMoe = Object.hasOwn(ramBackedMoeStages, stageId);
    const hasPagedKv = Object.hasOwn(pagedKvStages, stageId);
    if (hasNativeGguf && hasRamBackedMoe) {
      throw new Error(
        `python_native_gguf_stage_backend_is_not_exclusive:${stageId}`,
      );
    }
    if (hasPagedKv && (hasNativeGguf || hasRamBackedMoe)) {
      throw new Error(
        `python_paged_kv_stage_backend_is_not_exclusive:${stageId}`,
      );
    }
  }
}

function normalizeRamBackedMoeStages(
  manifest: RuntimePipelineManifestV2,
  value: unknown,
  runtimeModel: PythonRuntimeModelSource,
): Record<string, PythonRamBackedMoeStageConfiguration> {
  const requiredStages = manifest.plans.prefill.stages.filter(
    (stage) =>
      stage.macroWave?.memoryMode === "ram-backed"
      && stage.macroWave.ramArtifact !== undefined,
  );
  const requiredIds = new Set(requiredStages.map((stage) => stage.stageId));
  if (value === undefined) {
    if (requiredStages.length > 0) {
      throw new Error(
        `python_ram_backed_moe_stage_binding_is_missing:${requiredStages[0]!.stageId}`,
      );
    }
    return {};
  }
  if (!isRecord(value)) {
    throw new Error("python_ram_backed_moe_stages_must_be_an_object");
  }
  for (const stageId of Object.keys(value)) {
    if (!requiredIds.has(stageId)) {
      throw new Error(`python_ram_backed_moe_stage_binding_is_unexpected:${stageId}`);
    }
  }
  const normalizedEntries: Array<[
    string,
    PythonRamBackedMoeStageConfiguration,
  ]> = [];
  for (const stage of requiredStages) {
    const stageId = stage.stageId;
    const input = value[stageId];
    if (!isRecord(input)) {
      throw new Error(`python_ram_backed_moe_stage_binding_is_missing:${stageId}`);
    }
    assertExactKeys(
      input,
      [
        "schema",
        "snapshotPath",
        "artifactIdentity",
        "adapterId",
        "layerStart",
        "layerEnd",
        "totalLayers",
        "device",
        "pinMemory",
        "allowCpuFallback",
        "cache",
      ],
      [
        "residentParameterBudgetBytes",
        "totalRoutedExpertBytes",
        "largestExpertBytes",
        "residentStreamingTransientBytes",
        "boundedPinnedStagingReserveBytes",
        "hostRamPeakUpperBoundBytes",
      ],
      `python_ram_backed_moe_stage_configuration_keys_are_invalid:${stageId}`,
    );
    if (input.schema !== "gdlp-local-safetensors-moe-stage/1") {
      throw new Error(`python_ram_backed_moe_stage_schema_is_invalid:${stageId}`);
    }
    const snapshotPath = absoluteHostPath(
      input.snapshotPath,
      `python_ram_backed_moe_snapshot_path_is_invalid:${stageId}`,
    );
    const artifactIdentity = sha256Identity(
      input.artifactIdentity,
      `python_ram_backed_moe_artifact_identity_is_invalid:${stageId}`,
    );
    if (runtimeModel.snapshotIdentity === undefined) {
      throw new Error(
        `python_ram_backed_moe_requires_pipeline_snapshot_identity:${stageId}`,
      );
    }
    if (sha256Uint64Identity(artifactIdentity) !== runtimeModel.snapshotIdentity) {
      throw new Error(`python_ram_backed_moe_pipeline_identity_mismatch:${stageId}`);
    }
    const adapterId = ramBackedMoeAdapterId(input.adapterId, stageId);
    const artifactPolicy = stage.macroWave?.ramArtifact;
    if (!artifactPolicy?.adapterIds.includes(adapterId)) {
      throw new Error(`python_ram_backed_moe_adapter_is_not_allowed:${stageId}`);
    }
    const layerStart = boundedInteger(
      input.layerStart,
      0,
      manifest.totalLayers - 1,
      `python_ram_backed_moe_layer_start_is_invalid:${stageId}`,
    );
    const layerEnd = boundedInteger(
      input.layerEnd,
      1,
      manifest.totalLayers,
      `python_ram_backed_moe_layer_end_is_invalid:${stageId}`,
    );
    const totalLayers = boundedInteger(
      input.totalLayers,
      1,
      manifest.totalLayers,
      `python_ram_backed_moe_total_layers_is_invalid:${stageId}`,
    );
    if (
      layerStart !== stage.layerStart ||
      layerEnd !== stage.layerEnd ||
      totalLayers !== manifest.totalLayers
    ) {
      throw new Error(`python_ram_backed_moe_stage_range_mismatch:${stageId}`);
    }
    if (typeof input.device !== "string" || !/^cuda(?::(?:0|[1-9]\d*))?$/.test(input.device)) {
      throw new Error(`python_ram_backed_moe_device_is_invalid:${stageId}`);
    }
    if (input.pinMemory !== false) {
      throw new Error(`python_ram_backed_moe_pin_memory_must_be_false:${stageId}`);
    }
    if (input.allowCpuFallback !== false) {
      throw new Error(`python_ram_backed_moe_cpu_fallback_is_forbidden:${stageId}`);
    }
    const execution = stage.macroWave!;
    if (execution.requirements.weightBufferCopies < 2) {
      throw new Error(`python_ram_backed_moe_requires_double_weight_buffer:${stageId}`);
    }
    const residentParameterBudgetBytes =
      execution.requirements.residentParameterBudgetBytes;
    const totalRoutedExpertBytes = execution.workingSet.totalRoutedExpertBytes;
    const largestExpertBytes = execution.workingSet.largestExpertBytes;
    const residentStreamingTransientBytes =
      execution.requirements.residentStreamingTransientBytes;
    const boundedPinnedStagingReserveBytes =
      execution.requirements.boundedPinnedStagingReserveBytes;
    const hostRamPeakUpperBoundBytes =
      execution.requirements.hostRamPeakUpperBoundBytes;
    for (const [name, supplied, expected] of [
      [
        "resident_parameter",
        input.residentParameterBudgetBytes,
        residentParameterBudgetBytes,
      ],
      ["total_routed_expert", input.totalRoutedExpertBytes, totalRoutedExpertBytes],
      ["largest_expert", input.largestExpertBytes, largestExpertBytes],
      [
        "resident_streaming_transient",
        input.residentStreamingTransientBytes,
        residentStreamingTransientBytes,
      ],
      [
        "bounded_pinned_staging_reserve",
        input.boundedPinnedStagingReserveBytes,
        boundedPinnedStagingReserveBytes,
      ],
      [
        "host_ram_peak_upper_bound",
        input.hostRamPeakUpperBoundBytes,
        hostRamPeakUpperBoundBytes,
      ],
    ] as const) {
      if (supplied !== undefined && supplied !== expected) {
        throw new Error(`python_ram_backed_moe_${name}_budget_mismatch:${stageId}`);
      }
    }
    if (
      residentParameterBudgetBytes < 1 ||
      totalRoutedExpertBytes < 1 ||
      largestExpertBytes < 1 ||
      largestExpertBytes > totalRoutedExpertBytes ||
      residentStreamingTransientBytes < 0 ||
      boundedPinnedStagingReserveBytes < 0 ||
      hostRamPeakUpperBoundBytes !==
        totalRoutedExpertBytes +
          boundedPinnedStagingReserveBytes +
          residentStreamingTransientBytes
    ) {
      throw new Error(`python_ram_backed_moe_manifest_budget_is_invalid:${stageId}`);
    }
    if (!isRecord(input.cache)) {
      throw new Error(`python_ram_backed_moe_cache_is_invalid:${stageId}`);
    }
    assertExactKeys(
      input.cache,
      [
        "schema",
        "capacityBytes",
        "prefetchReserveBytes",
        "pcieBandwidthGbytesPerSecond",
        "hotnessDecay",
        "minPrefetchConfidence",
      ],
      [],
      `python_ram_backed_moe_cache_is_invalid:${stageId}`,
    );
    if (input.cache.schema !== "gdlp-predictive-expert-cache/1") {
      throw new Error(`python_ram_backed_moe_cache_schema_is_invalid:${stageId}`);
    }
    const expectedCapacity =
      execution.requirements.weightBufferBytes + execution.cachePolicy.capacityBytes;
    const capacityBytes = boundedInteger(
      input.cache.capacityBytes,
      2,
      Number.MAX_SAFE_INTEGER,
      `python_ram_backed_moe_cache_capacity_is_invalid:${stageId}`,
    );
    const prefetchReserveBytes = boundedInteger(
      input.cache.prefetchReserveBytes,
      1,
      Number.MAX_SAFE_INTEGER,
      `python_ram_backed_moe_prefetch_reserve_is_invalid:${stageId}`,
    );
    if (
      capacityBytes !== expectedCapacity ||
      prefetchReserveBytes !== largestExpertBytes ||
      prefetchReserveBytes * 2 > capacityBytes
    ) {
      throw new Error(`python_ram_backed_moe_cache_budget_mismatch:${stageId}`);
    }
    const pcieBandwidthGbytesPerSecond = positiveFinite(
      input.cache.pcieBandwidthGbytesPerSecond,
      `python_ram_backed_moe_pcie_bandwidth_is_invalid:${stageId}`,
    );
    const hotnessDecay = positiveFinite(
      input.cache.hotnessDecay,
      `python_ram_backed_moe_hotness_decay_is_invalid:${stageId}`,
    );
    if (hotnessDecay > 1) {
      throw new Error(`python_ram_backed_moe_hotness_decay_is_invalid:${stageId}`);
    }
    const minPrefetchConfidence = probability(
      input.cache.minPrefetchConfidence,
      `python_ram_backed_moe_prefetch_confidence_is_invalid:${stageId}`,
    );
    normalizedEntries.push([
      stageId,
      {
        schema: "gdlp-local-safetensors-moe-stage/1",
        snapshotPath,
        artifactIdentity,
        adapterId,
        layerStart,
        layerEnd,
        totalLayers,
        device: input.device,
        pinMemory: false,
        allowCpuFallback: false,
        residentParameterBudgetBytes,
        totalRoutedExpertBytes,
        largestExpertBytes,
        residentStreamingTransientBytes,
        boundedPinnedStagingReserveBytes,
        hostRamPeakUpperBoundBytes,
        cache: {
          schema: "gdlp-predictive-expert-cache/1",
          capacityBytes,
          prefetchReserveBytes,
          pcieBandwidthGbytesPerSecond,
          hotnessDecay,
          minPrefetchConfidence,
        },
      },
    ]);
  }
  return Object.fromEntries(normalizedEntries.sort(([left], [right]) => left.localeCompare(right)));
}

function prefillSettings(plan: RuntimePrefillPlanManifest): PythonPrefillLaunchSettings {
  return {
    phase: "prefill",
    planId: plan.planId,
    activationCodec: plan.activationCodec,
    microBatchSize: plan.microBatchSize,
    chunkTokens: plan.chunkTokens,
    macroWave: plan.macroWave ? structuredClone(plan.macroWave) : null,
  };
}

function decodeSettings(plan: RuntimeDecodePlanManifest): PythonDecodeLaunchSettings {
  return {
    phase: "decode",
    planId: plan.planId,
    activationCodec: plan.activationCodec,
    microBatchSize: plan.microBatchSize,
    directTokenReturnStage: plan.directTokenReturnStage,
    speculation: structuredClone(plan.speculation),
    macroWave: plan.macroWave ? structuredClone(plan.macroWave) : null,
  };
}

function tensorParallelCellExecution(
  execution: RuntimeVirtualStageManifest["execution"],
): RuntimeTensorParallelCellExecutionManifest | undefined {
  return execution?.mode === "tensor-parallel-cell" ? execution : undefined;
}

function samePhysicalTopology(
  left: RuntimeVirtualStageManifest[],
  right: RuntimeVirtualStageManifest[],
): boolean {
  if (left.length !== right.length) return false;
  return left.every((stage, index) => {
    const other = right[index];
    return (
      other !== undefined &&
      stage.stageId === other.stageId &&
      stage.index === other.index &&
      stage.layerStart === other.layerStart &&
      stage.layerEnd === other.layerEnd &&
      canonicalJson(stage.anchor) === canonicalJson(other.anchor) &&
      canonicalJson(
        stage.members.map((member) => ({
          nodeId: member.nodeId,
          endpoint: member.endpoint,
          backend: member.backend,
          capabilities: member.capabilities,
        })),
      ) ===
        canonicalJson(
          other.members.map((member) => ({
            nodeId: member.nodeId,
            endpoint: member.endpoint,
            backend: member.backend,
            capabilities: member.capabilities,
          })),
        ) &&
      canonicalJson(stage.execution) === canonicalJson(other.execution) &&
      canonicalJson(stage.macroWave) === canonicalJson(other.macroWave)
    );
  });
}

function downstreamStage(stage: RuntimeVirtualStageManifest): PythonDownstreamStage {
  return {
    stageId: stage.stageId,
    stageIndex: stage.index,
    layerEnd: stage.layerEnd,
    anchorMemberId: stage.anchor.memberId,
    endpoint: { ...stage.anchor.endpoint },
  };
}

function routeIdentity(
  pipelineId: string,
  settings: [PythonPrefillLaunchSettings, PythonDecodeLaunchSettings],
  stages: RuntimeVirtualStageManifest[],
  codec: RuntimeActivationCodec,
  nativeGgufStages: Record<string, PythonNativeGgufStageConfiguration>,
  ramBackedMoeStages: Record<string, PythonRamBackedMoeStageConfiguration>,
  pagedKvStages: Record<string, PythonPagedKvStageConfiguration>,
  recovery: PythonRecoveryConfiguration | null,
): string {
  return `route-${digest(
    canonicalJson({
      pipelineId,
      settings,
      codec,
      nativeGgufStages,
      ramBackedMoeStages,
      pagedKvStages,
      recovery,
      stages: stages.map((stage) => ({
        stageId: stage.stageId,
        index: stage.index,
        layers: [stage.layerStart, stage.layerEnd],
        anchor: stage.anchor,
        members: stage.members.map((member) => ({
          nodeId: member.nodeId,
          endpoint: member.endpoint,
          backend: member.backend,
          capabilities: member.capabilities,
        })),
        execution: stage.execution,
        macroWave: stage.macroWave,
      })),
    }),
    20,
  )}`;
}

function rootProcessIdentity(
  routeId: string,
  stage: RuntimeVirtualStageManifest,
  draftModel: PythonDraftModelConfiguration | null,
): string {
  return `root-${digest(
    canonicalJson({
      routeId,
      kind: "root-engine",
      stageId: stage.stageId,
      index: stage.index,
      layers: [stage.layerStart, stage.layerEnd],
      anchor: stage.anchor,
      execution: stage.execution,
      macroWave: stage.macroWave,
      draftModel,
    }),
    20,
  )}`;
}

function processIdentity(
  routeId: string,
  kind: PythonLaunchProcess["kind"],
  stage: RuntimeVirtualStageManifest,
  memberId?: string,
  rank?: number,
): string {
  const prefix =
    kind === "root-engine" ? "root" : kind === "cell-member" ? "cell" : "stage";
  return `${prefix}-${digest(
    canonicalJson({
      routeId,
      kind,
      stageId: stage.stageId,
      index: stage.index,
      layers: [stage.layerStart, stage.layerEnd],
      anchor: stage.anchor,
      execution: stage.execution,
      macroWave: stage.macroWave,
      memberId: memberId ?? null,
      rank: rank ?? null,
    }),
    20,
  )}`;
}

function pythonModulePrefix(module: string): string[] {
  // Readiness is observed over stdout/stderr pipes. Force unbuffered Python so
  // a service cannot be healthy yet remain invisible to the supervisor until
  // its stdio buffer fills or the process exits.
  return ["-u", "-m", module];
}

function appendModelArguments(args: string[], model: PythonRuntimeModelSource): void {
  appendBasicModelArguments(args, model.source, model.revision);
  if (model.artifactIdentity !== undefined) {
    args.push("--model-artifact-identity", model.artifactIdentity);
    args.push("--model-canonical-source", model.canonicalSource!);
    if (model.canonicalRevision != null) {
      args.push("--model-canonical-revision", model.canonicalRevision);
    }
  }
  if (model.snapshotIdentity !== undefined) {
    args.push("--pipeline-snapshot-identity", model.snapshotIdentity);
  }
}

function appendDraftModelArguments(
  args: string[],
  model: PythonDraftModelConfiguration,
): void {
  args.push(
    "--draft-model-source",
    model.source,
    "--draft-model-artifact-identity",
    model.artifactIdentity,
    "--draft-model-canonical-source",
    model.canonicalSource,
    "--draft-model-parameter-bytes",
    String(model.parameterBytes),
    "--draft-model-memory-reservation-bytes",
    String(model.memoryReservationBytes),
    "--draft-model-device",
    model.device,
    "--draft-model-dtype",
    model.dtype,
  );
  if (model.revision !== null) {
    args.push("--draft-model-revision", model.revision);
  }
  if (model.canonicalRevision !== null) {
    args.push("--draft-model-canonical-revision", model.canonicalRevision);
  }
}

function appendBasicModelArguments(
  args: string[],
  source: string,
  revision: string | null,
): void {
  args.push("--model", source);
  if (revision !== null) args.push("--revision", revision);
}

function appendRamBackedMoeArguments(
  args: string[],
  stage: PythonRamBackedMoeStageConfiguration,
): void {
  args.push(
    "--ram-moe-artifact-schema",
    stage.schema,
    "--ram-moe-artifact-identity",
    stage.artifactIdentity,
    "--ram-moe-adapter-id",
    stage.adapterId,
    "--ram-moe-device",
    stage.device,
    "--ram-moe-pin-memory",
    String(stage.pinMemory),
    "--ram-moe-allow-cpu-fallback",
    String(stage.allowCpuFallback),
    "--ram-moe-resident-parameter-budget-bytes",
    String(stage.residentParameterBudgetBytes),
    "--ram-moe-total-routed-expert-bytes",
    String(stage.totalRoutedExpertBytes),
    "--ram-moe-largest-expert-bytes",
    String(stage.largestExpertBytes),
    "--ram-moe-resident-streaming-transient-bytes",
    String(stage.residentStreamingTransientBytes),
    "--ram-moe-bounded-pinned-staging-reserve-bytes",
    String(stage.boundedPinnedStagingReserveBytes),
    "--ram-moe-host-ram-peak-upper-bound-bytes",
    String(stage.hostRamPeakUpperBoundBytes),
    "--ram-moe-cache-schema",
    stage.cache.schema,
    "--ram-moe-cache-capacity-bytes",
    String(stage.cache.capacityBytes),
    "--ram-moe-prefetch-reserve-bytes",
    String(stage.cache.prefetchReserveBytes),
    "--ram-moe-pcie-bandwidth-gbytes-per-second",
    finiteNumber(stage.cache.pcieBandwidthGbytesPerSecond),
    "--ram-moe-hotness-decay",
    finiteNumber(stage.cache.hotnessDecay),
    "--ram-moe-min-prefetch-confidence",
    finiteNumber(stage.cache.minPrefetchConfidence),
  );
}

type PythonModelArtifactCoordinates = Pick<
  PythonRuntimeModelSource,
  "artifactIdentity" | "canonicalSource" | "canonicalRevision"
>;

function normalizeDraftModel(
  value: unknown,
  speculation: PythonSpeculationArguments,
  requireNormalized: boolean,
): PythonDraftModelConfiguration | null {
  if (speculation.provider !== "draft-model") {
    if (value !== undefined && value !== null) {
      throw new Error("python_draft_model_requires_draft_model_strategy");
    }
    return null;
  }
  if (!isRecord(value)) {
    throw new Error("python_draft_model_configuration_is_missing");
  }
  assertExactKeys(
    value,
    ["schema", "source", "device", "dtype"],
    [
      "revision",
      "snapshotIdentity",
      "artifactIdentity",
      "canonicalSource",
      "canonicalRevision",
      "parameterBytes",
      "memoryReservationBytes",
    ],
    "python_draft_model_configuration_keys_are_invalid",
  );
  if (value.schema !== LOCAL_DRAFT_MODEL_SCHEMA) {
    throw new Error("python_draft_model_schema_is_invalid");
  }
  const source = safeString(
    value.source,
    "python_draft_model_source_is_invalid",
  );
  const revision =
    value.revision === undefined || value.revision === null
      ? null
      : safeString(
          value.revision,
          "python_draft_model_revision_is_invalid",
        );
  const derivedSnapshotIdentity = deriveHubSnapshotIdentity(source, revision);
  const snapshotIdentity =
    value.snapshotIdentity === undefined
      ? derivedSnapshotIdentity
      : uint64String(
          value.snapshotIdentity,
          "python_draft_model_snapshot_identity_is_invalid",
        );
  if (
    snapshotIdentity !== undefined
    && derivedSnapshotIdentity !== undefined
    && snapshotIdentity !== derivedSnapshotIdentity
  ) {
    throw new Error("python_draft_model_snapshot_identity_mismatch");
  }
  const coordinates = normalizeModelArtifactCoordinates(
    value,
    source,
    revision,
    snapshotIdentity,
    requireNormalized,
  );
  if (
    coordinates.artifactIdentity === undefined
    || coordinates.canonicalSource === undefined
    || coordinates.canonicalRevision === undefined
  ) {
    throw new Error("python_draft_model_artifact_coordinates_are_missing");
  }
  const strategyArtifact = sha256Identity(
    speculation.artifactId,
    "python_draft_model_strategy_artifact_is_invalid",
  );
  if (coordinates.artifactIdentity !== strategyArtifact) {
    throw new Error("python_draft_model_artifact_does_not_match_strategy");
  }
  for (const [key, expected, error] of [
    [
      "parameterBytes",
      speculation.parameterBytes,
      "python_draft_model_parameter_bytes_do_not_match_strategy",
    ],
    [
      "memoryReservationBytes",
      speculation.memoryReservationBytes,
      "python_draft_model_memory_reservation_does_not_match_strategy",
    ],
  ] as const) {
    const supplied = value[key];
    if (requireNormalized && supplied === undefined) {
      throw new Error(error);
    }
    if (
      supplied !== undefined
      && boundedInteger(supplied, 1, Number.MAX_SAFE_INTEGER, error) !== expected
    ) {
      throw new Error(error);
    }
  }
  const device = draftModelDevice(value.device);
  const dtype = draftModelDtype(value.dtype);
  if (device === "cpu" && dtype === "float16") {
    throw new Error("python_draft_model_float16_cpu_is_unsupported");
  }
  return {
    schema: LOCAL_DRAFT_MODEL_SCHEMA,
    source,
    revision,
    ...(snapshotIdentity === undefined ? {} : { snapshotIdentity }),
    artifactIdentity: coordinates.artifactIdentity,
    canonicalSource: coordinates.canonicalSource,
    canonicalRevision: coordinates.canonicalRevision,
    device,
    dtype,
    parameterBytes: speculation.parameterBytes,
    memoryReservationBytes: speculation.memoryReservationBytes,
  };
}

function assertDraftModelCapacity(
  manifest: RuntimePipelineManifestV2,
  draftModel: PythonDraftModelConfiguration | null,
): void {
  if (draftModel === null) return;
  for (const phase of ["prefill", "decode"] as const) {
    const root = manifest.plans[phase].stages.find((stage) => stage.index === 0);
    if (!root) {
      throw new Error(`python_draft_model_root_stage_is_missing:${phase}`);
    }
    const available = root.memoryLimitBytes - root.memoryBytes;
    if (draftModel.memoryReservationBytes > available) {
      throw new Error(`python_draft_model_memory_capacity_is_too_small:${phase}`);
    }
  }
}

function normalizeModelArtifactCoordinates(
  value: unknown,
  source: string,
  revision: string | null,
  snapshotIdentity: string | undefined,
  requireNormalized: boolean,
): PythonModelArtifactCoordinates | Record<string, never> {
  const derived = deriveModelArtifactCoordinates(source, revision, snapshotIdentity);
  if (!isRecord(value)) return derived;

  const hasArtifactIdentity = Object.prototype.hasOwnProperty.call(
    value,
    "artifactIdentity",
  );
  const hasCanonicalSource = Object.prototype.hasOwnProperty.call(
    value,
    "canonicalSource",
  );
  const hasCanonicalRevision = Object.prototype.hasOwnProperty.call(
    value,
    "canonicalRevision",
  );
  const hasAnyCoordinates =
    hasArtifactIdentity || hasCanonicalSource || hasCanonicalRevision;
  if (!hasAnyCoordinates) return derived;

  const hasAllCoordinates =
    hasArtifactIdentity && hasCanonicalSource && hasCanonicalRevision;
  if (
    requireNormalized &&
    hasAllCoordinates &&
    canonicalJson({
      artifactIdentity: value.artifactIdentity,
      canonicalSource: value.canonicalSource,
      canonicalRevision: value.canonicalRevision,
    }) === canonicalJson(derived)
  ) {
    // Revalidation sees the coordinates emitted by the compiler itself. This
    // preserves the legacy snapshot:uint64 representation for local models.
    return derived;
  }
  if (
    !hasAllCoordinates ||
    !Object.prototype.hasOwnProperty.call(value, "snapshotIdentity") ||
    snapshotIdentity === undefined
  ) {
    throw new Error("python_runtime_model_artifact_coordinates_are_incomplete");
  }

  const artifactIdentity = sha256Identity(
    value.artifactIdentity,
    "python_runtime_model_artifact_identity_is_invalid",
  );
  const canonicalSource = safeString(
    value.canonicalSource,
    "python_runtime_model_canonical_source_is_invalid",
  );
  const canonicalRevision = nullableSafeString(
    value.canonicalRevision,
    "python_runtime_model_canonical_revision_is_invalid",
  );
  if (sha256Uint64Identity(artifactIdentity) !== snapshotIdentity) {
    throw new Error("python_runtime_model_artifact_snapshot_identity_mismatch");
  }

  if (hubSnapshotCommit(source, revision) !== undefined) {
    if (artifactIdentity !== derived.artifactIdentity) {
      throw new Error("python_runtime_model_artifact_identity_mismatch");
    }
    if (canonicalSource !== derived.canonicalSource) {
      throw new Error("python_runtime_model_canonical_source_mismatch");
    }
    if (canonicalRevision !== derived.canonicalRevision) {
      throw new Error("python_runtime_model_canonical_revision_mismatch");
    }
  }
  return { artifactIdentity, canonicalSource, canonicalRevision };
}

function deriveModelArtifactCoordinates(
  source: string,
  revision: string | null,
  snapshotIdentity: string | undefined,
): Pick<
  PythonRuntimeModelSource,
  "artifactIdentity" | "canonicalSource" | "canonicalRevision"
> | Record<string, never> {
  const commit = hubSnapshotCommit(source, revision);
  if (commit !== undefined) {
    const artifactIdentity = `sha256:${createHash("sha256")
      .update("gdlp-hub-snapshot-v1\0")
      .update(commit, "ascii")
      .digest("hex")}`;
    const sourceParts = source.split(/[\\/]+/).filter(Boolean);
    const cacheDirectory = sourceParts.find((part) => part.startsWith("models--"));
    const cacheCoordinates = cacheDirectory?.replace(/^models--/, "").split("--");
    const directCoordinates =
      !source.includes("\\") && !source.includes(":") && source.split("/").length === 2
        ? source
        : undefined;
    const repository =
      cacheCoordinates?.length === 2 && cacheCoordinates.every(Boolean)
        ? cacheCoordinates.join("/")
        : directCoordinates;
    return {
      artifactIdentity,
      canonicalSource: repository ? `hf://${repository}` : `hf-snapshot://${commit}`,
      canonicalRevision: commit,
    };
  }
  if (snapshotIdentity === undefined) return {};
  const artifactIdentity = snapshotArtifactIdentity(snapshotIdentity);
  return {
    artifactIdentity,
    canonicalSource: `content-addressed://${artifactIdentity}`,
    canonicalRevision: null,
  };
}

function deriveHubSnapshotIdentity(
  source: string,
  revision: string | null,
): string | undefined {
  const commit = hubSnapshotCommit(source, revision);
  if (!commit) return undefined;
  const bytes = createHash("sha256")
    .update("gdlp-hub-snapshot-v1\0")
    .update(commit, "ascii")
    .digest()
    .subarray(0, 8);
  let identity = 0n;
  for (const byte of bytes) identity = (identity << 8n) | BigInt(byte);
  return identity.toString();
}

function hubSnapshotCommit(
  source: string,
  revision: string | null,
): string | undefined {
  const revisionCandidate = revision?.trim().toLowerCase();
  const sourceParts = source.split(/[\\/]+/).filter(Boolean);
  const pathCandidate =
    sourceParts.length >= 2 && sourceParts.at(-2)?.toLowerCase() === "snapshots"
      ? sourceParts.at(-1)!.toLowerCase()
      : undefined;
  const commit = [revisionCandidate, pathCandidate].find(
    (candidate) =>
      candidate !== undefined &&
      candidate.length >= 32 &&
      /^[0-9a-f]+$/.test(candidate),
  );
  return commit;
}

function snapshotArtifactIdentity(snapshotIdentity: string): string {
  const hexadecimal = BigInt(snapshotIdentity).toString(16).padStart(16, "0");
  return `snapshot:uint64:${hexadecimal}`;
}

function uint64String(value: unknown, error: string): string {
  if (typeof value !== "string" || !/^(?:0|[1-9]\d*)$/.test(value)) {
    throw new Error(error);
  }
  const parsed = BigInt(value);
  if (parsed > 18_446_744_073_709_551_615n) throw new Error(error);
  return parsed.toString();
}

function validateEndpoint(value: unknown, error: string): asserts value is RuntimeEndpoint {
  if (!isRecord(value) || typeof value.host !== "string" || !value.host.trim()) {
    throw new Error(error);
  }
  if (!Number.isInteger(value.port) || (value.port as number) < 1 || (value.port as number) > 65_535) {
    throw new Error(error);
  }
}

function endpointKey(endpoint: RuntimeEndpoint): string {
  return `${endpoint.host.trim().toLowerCase()}\u0000${endpoint.port}`;
}

function pythonModule(value: unknown, error: string): string {
  const result = safeString(value, error);
  if (!/^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$/.test(result)) throw new Error(error);
  return result;
}

function nativePythonModule<const T extends string>(
  value: unknown,
  expected: T,
  error: string,
): T {
  const result = pythonModule(value ?? expected, error);
  if (result !== expected) throw new Error(error);
  return expected;
}

function safeString(value: unknown, error: string): string {
  if (
    typeof value !== "string" ||
    !value.trim() ||
    value.includes("\u0000") ||
    value.includes("\r") ||
    value.includes("\n")
  ) {
    throw new Error(error);
  }
  return value;
}

function nullableSafeString(value: unknown, error: string): string | null {
  return value === null ? null : safeString(value, error);
}

function sha256Digest(value: unknown, error: string): string {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) {
    throw new Error(error);
  }
  return value;
}

function sha256Identity(value: unknown, error: string): string {
  if (typeof value !== "string" || !/^sha256:[0-9a-f]{64}$/.test(value)) {
    throw new Error(error);
  }
  return value;
}

function draftModelDevice(value: unknown): PythonDraftModelDevice {
  if (
    typeof value !== "string"
    || !/^(?:cpu|mps|cuda(?::(?:0|[1-9]\d*))?|xpu(?::(?:0|[1-9]\d*))?)$/.test(
      value,
    )
  ) {
    throw new Error("python_draft_model_device_is_invalid");
  }
  return value as PythonDraftModelDevice;
}

function draftModelDtype(
  value: unknown,
): PythonDraftModelConfiguration["dtype"] {
  if (
    value !== "float32"
    && value !== "float16"
    && value !== "bfloat16"
  ) {
    throw new Error("python_draft_model_dtype_is_invalid");
  }
  return value;
}

function sha256Uint64Identity(identity: string): string {
  return BigInt(`0x${identity.slice("sha256:".length, "sha256:".length + 16)}`)
    .toString();
}

function absoluteHostPath(value: unknown, error: string): string {
  const result = safeString(value, error);
  const windowsDrive = /^[A-Za-z]:[\\/]/.test(result);
  const windowsUnc = /^\\\\[^\\/]+[\\/][^\\/]+/.test(result);
  const posix = result.startsWith("/");
  const segments = result.split(/[\\/]+/);
  if (
    (!windowsDrive && !windowsUnc && !posix) ||
    segments.some((segment) => segment === "." || segment === "..")
  ) {
    throw new Error(error);
  }
  return result;
}

function ramBackedMoeAdapterId(
  value: unknown,
  stageId: string,
): PythonRamBackedMoeAdapterId {
  if (
    value !== "transformers-qwen3-moe-v1" &&
    value !== "transformers-glm4-moe-v1"
  ) {
    throw new Error(`python_ram_backed_moe_adapter_is_invalid:${stageId}`);
  }
  return value;
}

function assertExactKeys(
  value: Record<string, unknown>,
  required: string[],
  optional: string[],
  error: string,
): void {
  const actual = Object.keys(value).sort();
  const allowed = new Set([...required, ...optional]);
  if (
    required.some((key) => !Object.prototype.hasOwnProperty.call(value, key)) ||
    actual.some((key) => !allowed.has(key))
  ) {
    throw new Error(error);
  }
}

function boundedInteger(value: unknown, min: number, max: number, error: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < min || (value as number) > max) {
    throw new Error(error);
  }
  return value as number;
}

function positiveFinite(value: unknown, error: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(error);
  }
  return value;
}

function nonNegativeFinite(value: unknown, error: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new Error(error);
  }
  return value;
}

function probability(value: unknown, error: string): number {
  const result = nonNegativeFinite(value, error);
  if (result > 1) throw new Error(error);
  return result;
}

function atLeastOneFinite(value: unknown, error: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 1) {
    throw new Error(error);
  }
  return value;
}

function finiteNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(6)));
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(canonicalValue(value));
}

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (!isRecord(value)) return value;
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, canonicalValue(value[key])]),
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function digest(value: string, length: number): string {
  return createHash("sha256").update(value).digest("hex").slice(0, length);
}
