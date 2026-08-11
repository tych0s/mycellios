import { createHash } from "node:crypto";
import { evaluateDistributionPlan } from "./cost-model.js";
import {
  DEFAULT_SEARCH_OPTIONS,
  FleetTopologyPlanner,
  TopologyBeamPlanner,
} from "./planners.js";
import {
  MacroWaveRamVramPlanner,
  type MacroWavePlannerOptions,
} from "./macro-wave.js";
import { tensorParallelCellGateReason } from "./parallelism.js";
import type {
  ActivationCodecId,
  ComputeNodeProfile,
  DirectedLinkProfile,
  DistributedModelProfile,
  DistributionMetrics,
  DistributionPlan,
  DistributionTopology,
  DistributionWorkload,
  MacroWavePlanContractV1,
  MacroWaveStageExecutionContractV1,
  SearchOptions,
} from "./types.js";

export interface RuntimeEndpoint {
  host: string;
  port: number;
}

export type RuntimeActivationCodec = Exclude<ActivationCodecId, "q4">;
export type RuntimePlanTransport =
  | "persistent-tcp"
  | "persistent-tcp-macro-wave-v1";
export const MAX_WAN_VIRTUAL_STAGES = 8;

/** Mycellios executor metadata remains extensible without changing the wire protocol. */
export interface RuntimeBackendProfile {
  engine: string;
  version?: string;
  modelFormats: string[];
  executionModes: string[];
}

/**
 * Capability names are negotiated strings, not a closed enum. That lets
 * Mycellios executors for different accelerators coexist while retaining a
 * small, portable manifest.
 */
export interface RuntimeMemberCapabilities {
  deviceKinds: string[];
  computeApis: string[];
  weightDtypes: string[];
  activationCodecs: RuntimeActivationCodec[];
  features: string[];
}

export interface RuntimeBackendInput {
  engine: string;
  version?: string;
  modelFormats?: string[];
  executionModes?: string[];
}

export interface RuntimeMemberCapabilitiesInput {
  deviceKinds?: string[];
  computeApis?: string[];
  weightDtypes?: string[];
  activationCodecs?: RuntimeActivationCodec[];
  features?: string[];
}

export interface RuntimeNodeProfile {
  id: string;
  region: string;
  memoryBytes: number;
  reserveBytes: number;
  decodeScale: number;
  prefillScale: number;
  codecScale: number;
  batchGain: number;
  maxBatchSpeedup: number;
  powerWatts: number;
  availability: number;
  endpoint: RuntimeEndpoint;
  backend?: RuntimeBackendInput;
  capabilities?: RuntimeMemberCapabilitiesInput;
  /** Required only when the opt-in MacroWave planner is selected. */
  ramVram?: ComputeNodeProfile["ramVram"];
}

export interface RuntimeTopology {
  nodes: RuntimeNodeProfile[];
  links: DirectedLinkProfile[];
}

/** The original, one-node-per-stage GDLP/1 stage. */
export interface RuntimeStageManifest {
  index: number;
  nodeId: string;
  endpoint: RuntimeEndpoint;
  layerStart: number;
  layerEnd: number;
  first: boolean;
  last: boolean;
  memoryBytes: number;
  memoryLimitBytes: number;
}

/** The original wire contract remains readable and validatable. */
export interface RuntimePipelineManifestV1 {
  protocol: "gdlp/1";
  pipelineId: string;
  modelId: string;
  modelRevision: string;
  tokenizerId: string;
  totalLayers: number;
  hiddenSize: number;
  activationCodec: "fp16" | "int8";
  transport: "persistent-tcp";
  prefillChunkTokens: number;
  microBatchSize: number;
  directTokenReturnStage: 0;
  stages: RuntimeStageManifest[];
  predicted: DistributionMetrics;
}

export interface RuntimeStageMemberManifest {
  nodeId: string;
  endpoint: RuntimeEndpoint;
  backend: RuntimeBackendProfile;
  capabilities: RuntimeMemberCapabilities;
  /** Portion of the virtual stage allocation owned by this member. */
  assignedMemoryBytes: number;
  memoryLimitBytes: number;
}

export type RuntimeCellFixtureSchema =
  | "gdlp-llama-cell-layer/1"
  | "gdlp-llama-cell-stage/2";

export type RuntimeTensorParallelCollectiveBackend = "gloo" | "nccl";
export type RuntimeTensorParallelComputeDtype =
  | "float32"
  | "float16"
  | "bfloat16";

export interface RuntimeTensorParallelCellExternalManifest {
  /** Rank-local fixture directory in rank order; entry zero equals fixture.path. */
  rankFixturePaths: string[];
  controlBindHost: string;
  controlAdvertiseHost: string;
  controlPort: number;
  distributedAdvertiseHost: string;
  distributedPort: number;
  startupTimeoutSeconds: number;
}

export interface RuntimeTensorParallelCellRankMemory {
  /** Immutable weights plus any fixed allocations not covered by node reserve. */
  fixedBytes: number;
  /** Rank-local K+V bytes for one cached token and one sequence. */
  kvBytesPerToken: number;
  /** fixedBytes + kvBytesPerToken * planned context * planned concurrency. */
  requiredBytes: number;
}

/**
 * Executable contract for the current Python tensor-parallel cell prototype.
 *
 * `anchor-local` spawns every Gloo rank beside stage_cli. `member-local`
 * launches ranks 1..N-1 through their own agents and uses the explicit LAN
 * control/collective endpoints below. Neither mode transfers fixtures.
 */
export interface RuntimeTensorParallelCellExecutionManifest {
  mode: "tensor-parallel-cell";
  engine: "python-torch";
  collectiveBackend: RuntimeTensorParallelCollectiveBackend;
  computeDtype: RuntimeTensorParallelComputeDtype;
  fixture: {
    schema: RuntimeCellFixtureSchema;
    location: "anchor-local" | "member-local";
    path: string;
    layerCount: number;
    /** SHA-256 of the exact cell.json consumed before READY. */
    manifestSha256: string;
    /** SHA-256 values in rank order. */
    shardSha256: string[];
    /** Exact physical memory contract in rank order. */
    rankMemory: RuntimeTensorParallelCellRankMemory[];
  };
  worldSize: number;
  /** Array position is the collective rank; rank zero is the stage anchor. */
  rankMemberIds: string[];
  /** Relative calibrated capacity used for unequal head/MLP apportionment. */
  rankWeights: number[];
  /** PyTorch device in rank order: cpu for Gloo or cuda:<index> for NCCL/RCCL. */
  rankDevices: string[];
  operationTimeoutSeconds: number;
  /** Required only when fixture.location is member-local. */
  external?: RuntimeTensorParallelCellExternalManifest;
}

export type RuntimeVirtualStageExecutionManifest =
  RuntimeTensorParallelCellExecutionManifest;

export interface RuntimeVirtualStageManifest {
  stageId: string;
  index: number;
  layerStart: number;
  layerEnd: number;
  first: boolean;
  last: boolean;
  /** The stable ingress/egress endpoint for this group. */
  anchor: {
    memberId: string;
    endpoint: RuntimeEndpoint;
  };
  members: RuntimeStageMemberManifest[];
  /** Omitted means the ordinary one-member layer-range runner. */
  execution?: RuntimeVirtualStageExecutionManifest;
  /** Versioned RAM/VRAM contract emitted only by the MacroWave planner. */
  macroWave?: MacroWaveStageExecutionContractV1;
  memoryBytes: number;
  memoryLimitBytes: number;
}

export interface RuntimePrefillPlanManifest {
  phase: "prefill";
  planId: string;
  activationCodec: RuntimeActivationCodec;
  transport: RuntimePlanTransport;
  chunkTokens: number;
  microBatchSize: number;
  stages: RuntimeVirtualStageManifest[];
  /** Omitted for the unchanged standard planner/runtime path. */
  macroWave?: MacroWavePlanContractV1;
  predicted: DistributionMetrics;
}

export type RuntimeSpeculationKind =
  | "autoregressive"
  | "ngram"
  | "draft-model"
  | "draft-tree"
  | "mtp"
  | "intermediate-head";

export interface RuntimeSpeculationStrategy {
  id: string;
  kind: RuntimeSpeculationKind;
  maxDraftTokens: number;
  /**
   * Exact physical-tree limits. They are forbidden for every other strategy
   * kind and required together for `draft-tree`, so an executor never invents
   * capacity that the planner did not seal.
   */
  maxBranches?: number;
  maxBranchTokens?: number;
  maxKvBytes?: number;
  maxWaveTokens?: number;
  minAcceptanceRate: number;
  maxWasteRatio: number;
  priority: number;
  artifactId?: string;
  /** Exact loaded parameter bytes for an artifact-backed local drafter. */
  parameterBytes?: number;
  /** Capacity reserved before loading the optional local drafter. */
  memoryReservationBytes?: number;
}

export interface RuntimeSpeculationPolicy {
  mode: "disabled" | "adaptive";
  controller: "fixed" | "acceptance-adaptive";
  defaultStrategyId: string;
  fallbackStrategyId: string;
  acceptanceWindowTokens: number;
  strategies: RuntimeSpeculationStrategy[];
}

export interface RuntimeDecodePlanManifest {
  phase: "decode";
  planId: string;
  activationCodec: RuntimeActivationCodec;
  transport: RuntimePlanTransport;
  microBatchSize: number;
  directTokenReturnStage: number;
  stages: RuntimeVirtualStageManifest[];
  /** Omitted for the unchanged standard planner/runtime path. */
  macroWave?: MacroWavePlanContractV1;
  speculation: RuntimeSpeculationPolicy;
  predicted: DistributionMetrics;
}

export interface RuntimeKvFormat {
  id: string;
  version: number;
  dtype: string;
  layout: string;
}

export interface RuntimeKvCompatibilityGate {
  policy: "strict";
  modelRevision: string;
  tokenizerId: string;
  layerLayoutHash: string;
  requireContextIdentity: boolean;
  onMismatch: "reject" | "recompute-prefill";
}

export interface RuntimeKvTransitionManifest {
  mode: "in-place" | "transfer" | "recompute";
  format: RuntimeKvFormat;
  gate: RuntimeKvCompatibilityGate;
}

export interface RuntimePipelineManifestV2 {
  protocol: "gdlp/2";
  pipelineId: string;
  modelId: string;
  modelRevision: string;
  tokenizerId: string;
  totalLayers: number;
  hiddenSize: number;
  plans: {
    prefill: RuntimePrefillPlanManifest;
    decode: RuntimeDecodePlanManifest;
  };
  kvTransition: RuntimeKvTransitionManifest;
}

export type RuntimePipelineManifest =
  | RuntimePipelineManifestV1
  | RuntimePipelineManifestV2;

export interface RuntimeKvOptions {
  mode?: RuntimeKvTransitionManifest["mode"];
  format?: Partial<RuntimeKvFormat>;
  requireContextIdentity?: boolean;
  onMismatch?: RuntimeKvCompatibilityGate["onMismatch"];
}

export interface RuntimePlanRequest {
  model: DistributedModelProfile;
  modelRevision: string;
  tokenizerId?: string;
  topology: RuntimeTopology;
  workload: DistributionWorkload;
  /** Opt-in only: INT8 changed greedy tokens in the physical SmolLM2 gate. */
  allowLossyActivation?: boolean;
  /** MacroWave is never selected implicitly; omitted/default preserves legacy planning. */
  planner?:
    | { kind: "default" }
    | { kind: "macro-wave"; options?: MacroWavePlannerOptions };
  /** Optional phase-specific placements; both phases use the winning route by default. */
  phasePlans?: {
    prefill?: DistributionPlan;
    decode?: DistributionPlan;
  };
  speculation?: RuntimeSpeculationPolicy;
  kv?: RuntimeKvOptions;
  /**
   * Optional physical TP cells exposed to the planner as one logical node.
   * Non-anchor members are reserved for their cell and cannot also be chosen
   * as independent pipeline stages.
   */
  tensorParallelCells?: RuntimeTensorParallelCellPlanningInput[];
  /**
   * Optional phase-specific TP cells. Each execution names a complete,
   * already-resident fixture for that phase; this contract never implies
   * live weight reshaping or KV migration. It is mutually exclusive with
   * tensorParallelCells. The current Python launcher deliberately rejects
   * differing phase profiles, so they require separate phase executors and
   * recomputed KV until a real transition runtime exists.
   */
  phaseTensorParallelCells?: {
    prefill?: RuntimeTensorParallelCellPlanningInput[];
    decode?: RuntimeTensorParallelCellPlanningInput[];
  };
}

export interface RuntimeTensorParallelCellPlanningInput {
  /** Index in the target phase route; the current runner permits intermediate stages only. */
  stageIndex: number;
  /** Gloo rank order. Rank zero is the logical route node and stage anchor. */
  memberNodeIds: string[];
  execution: RuntimeTensorParallelCellExecutionManifest;
}

export interface RuntimeTensorParallelCellMaterialization {
  /** Index in both phase routes; the current runner permits intermediate stages only. */
  stageIndex: number;
  /** Rank members in the same order declared by execution.rankMemberIds. */
  members: RuntimeStageMemberManifest[];
  execution: RuntimeTensorParallelCellExecutionManifest;
}

export interface RuntimeKvGateProbe {
  prefillPlanId: string;
  decodePlanId: string;
  modelRevision: string;
  tokenizerId: string;
  layerLayoutHash: string;
  formatId: string;
  sourceContextIdentity?: string;
  targetContextIdentity?: string;
}

export type RuntimeKvGateReason =
  | "prefill_plan_mismatch"
  | "decode_plan_mismatch"
  | "model_revision_mismatch"
  | "tokenizer_mismatch"
  | "layer_layout_mismatch"
  | "kv_format_mismatch"
  | "context_identity_missing"
  | "context_identity_mismatch";

export interface RuntimeKvGateDecision {
  decision: "in-place" | "transfer" | "recompute" | "reject";
  allowed: boolean;
  reasons: RuntimeKvGateReason[];
}

/**
 * Build the current executable contract.  The optimizer still produces a
 * physical route; GDLP/2 exposes it as two independent logical plans so each
 * can be optimized separately without another wire-format migration.
 */
export function buildRuntimePipelineManifest(
  request: RuntimePlanRequest,
): RuntimePipelineManifestV2 {
  validatePlanRequest(request);
  const commonCells = request.tensorParallelCells ?? [];
  const phaseCells = request.phaseTensorParallelCells;
  if (commonCells.length > 0 && phaseCells !== undefined) {
    throw new Error("runtime_common_and_phase_cells_are_mutually_exclusive");
  }
  const prefillCells = phaseCells?.prefill ?? commonCells;
  const decodeCells = phaseCells?.decode ?? commonCells;
  const sharedCellProfiles = samePlannedCellProfiles(prefillCells, decodeCells);
  const baseTopology = distributionTopology(request.topology);
  const prefillTopology = cellAwarePlanningTopology(baseTopology, prefillCells);
  const decodeTopology = sharedCellProfiles
    ? prefillTopology
    : cellAwarePlanningTopology(baseTopology, decodeCells);
  let prefillPlan: DistributionPlan;
  let decodePlan: DistributionPlan;
  if (sharedCellProfiles) {
    const basePlan = planDefaultRoute(request, prefillTopology);
    prefillPlan = request.phasePlans?.prefill ?? basePlan;
    decodePlan = request.phasePlans?.decode ?? basePlan;
  } else {
    prefillPlan =
      request.phasePlans?.prefill ?? planDefaultRoute(request, prefillTopology);
    decodePlan =
      request.phasePlans?.decode ?? planDefaultRoute(request, decodeTopology);
  }
  validateSelectedPlanningMode(request, prefillPlan, decodePlan);
  const prefillPredicted = evaluateRuntimePlan(
    request,
    prefillTopology,
    prefillPlan,
    "prefill",
  );
  const decodePredicted = evaluateRuntimePlan(
    request,
    decodeTopology,
    decodePlan,
    "decode",
  );
  const nodes = new Map(request.topology.nodes.map((node) => [node.id, node]));
  const prefillStages = buildVirtualStages(
    prefillPlan,
    prefillPredicted,
    nodes,
    "prefill",
  );
  const decodeStages = buildVirtualStages(
    decodePlan,
    decodePredicted,
    nodes,
    "decode",
  );
  const speculation = cloneSpeculationPolicy(
    request.speculation ?? defaultRuntimeSpeculationPolicy(),
  );
  const tokenizerId = request.tokenizerId ?? request.model.id;
  const sameRoute =
    sharedCellProfiles && sameExecutionRoute(prefillStages, decodeStages);
  const kvMode = request.kv?.mode ?? (sameRoute ? "in-place" : "recompute");
  const kvFormat: RuntimeKvFormat = {
    id: request.kv?.format?.id ?? "gdlp-kv-v1",
    version: request.kv?.format?.version ?? 1,
    dtype: request.kv?.format?.dtype ?? "backend-native",
    layout: request.kv?.format?.layout ?? "layer-major",
  };
  const kvTransition: RuntimeKvTransitionManifest = {
    mode: kvMode,
    format: kvFormat,
    gate: {
      policy: "strict",
      modelRevision: request.modelRevision,
      tokenizerId,
      layerLayoutHash: modelLayerLayoutHash(request.model),
      requireContextIdentity: request.kv?.requireContextIdentity ?? true,
      onMismatch: request.kv?.onMismatch ?? "recompute-prefill",
    },
  };
  const manifest: RuntimePipelineManifestV2 = {
    protocol: "gdlp/2",
    pipelineId: "pending",
    modelId: request.model.id,
    modelRevision: request.modelRevision,
    tokenizerId,
    totalLayers: request.model.layers.length,
    hiddenSize: request.model.layers[0]!.activationElements,
    plans: {
      prefill: {
        phase: "prefill",
        planId: "pending",
        activationCodec: runtimeCodec(prefillPlan.codec),
        transport: prefillPlan.macroWave
          ? "persistent-tcp-macro-wave-v1"
          : "persistent-tcp",
        chunkTokens: prefillPlan.prefillChunkTokens,
        microBatchSize: prefillPlan.microBatchSize,
        stages: prefillStages,
        ...(prefillPlan.macroWave
          ? { macroWave: structuredClone(prefillPlan.macroWave) }
          : {}),
        predicted: prefillPredicted,
      },
      decode: {
        phase: "decode",
        planId: "pending",
        activationCodec: runtimeCodec(decodePlan.codec),
        transport: decodePlan.macroWave
          ? "persistent-tcp-macro-wave-v1"
          : "persistent-tcp",
        microBatchSize: decodePlan.microBatchSize,
        directTokenReturnStage: 0,
        stages: decodeStages,
        ...(decodePlan.macroWave
          ? { macroWave: structuredClone(decodePlan.macroWave) }
          : {}),
        speculation,
        predicted: decodePredicted,
      },
    },
    kvTransition,
  };
  sealRuntimePipelineIdentities(manifest);
  validateRuntimePipelineManifest(manifest);
  if (phaseCells !== undefined) {
    return materializePlannedTensorParallelCellsByPhase(
      manifest,
      request,
      prefillCells,
      decodeCells,
    );
  }
  return materializePlannedTensorParallelCells(
    manifest,
    request,
    commonCells,
  );
}

/**
 * Clone and seal one already-materialized anchor-local Python TP cell into
 * both phase routes.  No fixture is copied or probed here; stage_cli validates
 * the declared fixture on the anchor before becoming ready.
 */
export function materializeRuntimeTensorParallelCell(
  manifestValue: RuntimePipelineManifestV2,
  input: RuntimeTensorParallelCellMaterialization,
): RuntimePipelineManifestV2 {
  validateRuntimePipelineManifest(manifestValue);
  if (!isRecord(input)) throw new Error("runtime_cell_materialization_is_invalid");
  const stageIndex = asInteger(
    input.stageIndex,
    "runtime_cell_materialization_stage_index_is_invalid",
  );
  const prefillStages = manifestValue.plans.prefill.stages;
  const decodeStages = manifestValue.plans.decode.stages;
  if (
    prefillStages.length !== decodeStages.length ||
    stageIndex <= 0 ||
    stageIndex >= prefillStages.length - 1
  ) {
    throw new Error("runtime_cell_materialization_stage_must_be_intermediate");
  }
  const prefillStage = prefillStages[stageIndex]!;
  const decodeStage = decodeStages[stageIndex]!;
  if (
    prefillStage.macroWave !== undefined ||
    decodeStage.macroWave !== undefined
  ) {
    throw new Error("runtime_macro_wave_and_tensor_parallel_cells_are_not_composable");
  }
  if (
    prefillStage.layerStart !== decodeStage.layerStart ||
    prefillStage.layerEnd !== decodeStage.layerEnd ||
    prefillStage.anchor.memberId !== decodeStage.anchor.memberId ||
    !endpointEquals(prefillStage.anchor.endpoint, decodeStage.anchor.endpoint)
  ) {
    throw new Error("runtime_cell_materialization_requires_shared_phase_stage");
  }
  if (
    prefillStage.memoryBytes !== decodeStage.memoryBytes ||
    prefillStage.memoryLimitBytes !== decodeStage.memoryLimitBytes
  ) {
    throw new Error("runtime_cell_materialization_phase_memory_mismatch");
  }
  if (!Array.isArray(input.members) || input.members.length < 2) {
    throw new Error("runtime_cell_materialization_requires_multiple_members");
  }
  if (!isRecord(input.execution) || input.execution.mode !== "tensor-parallel-cell") {
    throw new Error("runtime_cell_materialization_execution_is_invalid");
  }
  if (
    !Array.isArray(input.execution.rankMemberIds) ||
    !Array.isArray(input.execution.fixture?.rankMemory) ||
    input.execution.rankMemberIds.length !== input.members.length ||
    input.execution.fixture.rankMemory.length !== input.members.length
  ) {
    throw new Error("runtime_cell_materialization_rank_contract_is_invalid");
  }
  let assignedMemoryBytes = 0;
  let memoryLimitBytes = 0;
  for (const [rank, member] of input.members.entries()) {
    if (!isRecord(member)) {
      throw new Error("runtime_cell_materialization_member_is_invalid");
    }
    const assigned = asNonNegativeInteger(
      member.assignedMemoryBytes,
      "runtime_cell_materialization_member_memory_is_invalid",
    );
    const limit = asNonNegativeInteger(
      member.memoryLimitBytes,
      "runtime_cell_materialization_member_memory_limit_is_invalid",
    );
    if (assigned > limit) {
      throw new Error("runtime_cell_materialization_member_memory_exceeded");
    }
    if (member.nodeId !== input.execution.rankMemberIds[rank]) {
      throw new Error("runtime_cell_materialization_rank_members_mismatch");
    }
    const required = asNonNegativeInteger(
      input.execution.fixture.rankMemory[rank]!.requiredBytes,
      "runtime_cell_materialization_rank_memory_is_invalid",
    );
    if (assigned !== required) {
      throw new Error("runtime_cell_materialization_rank_memory_mismatch");
    }
    assignedMemoryBytes += assigned;
    memoryLimitBytes += limit;
  }
  const manifest = structuredClone(manifestValue);
  for (const plan of [manifest.plans.prefill, manifest.plans.decode]) {
    const stage = plan.stages[stageIndex]!;
    stage.members = structuredClone(input.members);
    stage.execution = structuredClone(input.execution);
    // The cell uses the fixture's exact rank-local physical requirement. This
    // can differ from the generic model profile through FP32 conversion,
    // replicated norms/KV or a future quantized fixture.
    stage.memoryBytes = assignedMemoryBytes;
    stage.memoryLimitBytes = memoryLimitBytes;
    const predictedStage = plan.predicted.stageMetrics[stageIndex];
    if (!predictedStage) {
      throw new Error("runtime_cell_materialization_prediction_is_missing");
    }
    predictedStage.memoryBytes = assignedMemoryBytes;
    predictedStage.memoryLimitBytes = memoryLimitBytes;
    plan.predicted.peakStageMemoryBytes = Math.max(
      ...plan.predicted.stageMetrics.map((metrics) => metrics.memoryBytes),
    );
    plan.predicted.calibrationRequired = true;
    plan.predicted.calibrationReasons = [
      ...new Set([
        ...(plan.predicted.calibrationReasons ?? []),
        "tensor_parallel_collective_cost_unprofiled",
        ...(input.execution.fixture.location === "member-local"
          ? ["external_cell_internal_links_unprofiled"]
          : []),
      ]),
    ];
  }
  sealRuntimePipelineIdentities(manifest);
  validateRuntimePipelineManifest(manifest);
  return manifest;
}

/**
 * Certify every materialized TP cell against fresh physical link probes.
 *
 * Materialization deliberately marks collective cost as unprofiled. The only
 * route from that state to a launchable manifest is this fail-closed gate:
 * every ordered member pair must exist in one locality domain and carry fresh,
 * sufficiently sampled runtime-probe evidence within the TP latency ceiling.
 */
export function certifyRuntimeTensorParallelCollectives(
  manifestValue: RuntimePipelineManifestV2,
  topology: RuntimeTopology,
  now = Date.now(),
): RuntimePipelineManifestV2 {
  validateRuntimePipelineManifest(manifestValue);
  const manifest = structuredClone(manifestValue);
  let cells = 0;
  for (const plan of [manifest.plans.prefill, manifest.plans.decode]) {
    for (const stage of plan.stages) {
      if (stage.execution?.mode !== "tensor-parallel-cell") continue;
      cells += 1;
      const memberNodeIds = stage.execution.rankMemberIds;
      const gateReason = tensorParallelCellGateReason(
        memberNodeIds,
        topology,
        now,
      );
      if (gateReason !== null) {
        throw new Error(
          `runtime_tensor_parallel_collective_profile_rejected:${gateReason}`,
        );
      }
      applyTensorParallelEvidence(
        plan.predicted,
        memberNodeIds,
        topology,
        now,
      );
    }
  }
  if (cells === 0) {
    throw new Error("runtime_tensor_parallel_collective_cell_is_missing");
  }
  sealRuntimePipelineIdentities(manifest);
  validateRuntimePipelineManifest(manifest);
  return manifest;
}

export function defaultRuntimeSpeculationPolicy(): RuntimeSpeculationPolicy {
  return {
    mode: "disabled",
    controller: "fixed",
    defaultStrategyId: "autoregressive",
    fallbackStrategyId: "autoregressive",
    acceptanceWindowTokens: 128,
    strategies: [
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
}

/** Parse either supported protocol and fail closed on malformed manifests. */
export function readRuntimePipelineManifest(source: string | unknown): RuntimePipelineManifest {
  let value: unknown = source;
  if (typeof source === "string") {
    try {
      value = JSON.parse(source) as unknown;
    } catch {
      throw new Error("invalid_runtime_manifest_json");
    }
  }
  validateRuntimePipelineManifest(value);
  return value;
}

export function validateRuntimePipelineManifest(
  manifest: unknown,
): asserts manifest is RuntimePipelineManifest {
  if (!isRecord(manifest)) throw new Error("runtime_manifest_must_be_an_object");
  if (manifest.protocol === "gdlp/1") {
    validateRuntimePipelineManifestV1(manifest);
    return;
  }
  if (manifest.protocol === "gdlp/2") {
    validateRuntimePipelineManifestV2(manifest);
    return;
  }
  throw new Error("unsupported_runtime_protocol");
}

export function evaluateRuntimeKvGate(
  manifest: RuntimePipelineManifestV2,
  probe: RuntimeKvGateProbe,
): RuntimeKvGateDecision {
  validateRuntimePipelineManifest(manifest);
  if (manifest.kvTransition.mode === "recompute") {
    return { decision: "recompute", allowed: true, reasons: [] };
  }
  const reasons: RuntimeKvGateReason[] = [];
  if (probe.prefillPlanId !== manifest.plans.prefill.planId) {
    reasons.push("prefill_plan_mismatch");
  }
  if (probe.decodePlanId !== manifest.plans.decode.planId) {
    reasons.push("decode_plan_mismatch");
  }
  if (probe.modelRevision !== manifest.kvTransition.gate.modelRevision) {
    reasons.push("model_revision_mismatch");
  }
  if (probe.tokenizerId !== manifest.kvTransition.gate.tokenizerId) {
    reasons.push("tokenizer_mismatch");
  }
  if (probe.layerLayoutHash !== manifest.kvTransition.gate.layerLayoutHash) {
    reasons.push("layer_layout_mismatch");
  }
  if (probe.formatId !== manifest.kvTransition.format.id) {
    reasons.push("kv_format_mismatch");
  }
  if (manifest.kvTransition.gate.requireContextIdentity) {
    if (!probe.sourceContextIdentity || !probe.targetContextIdentity) {
      reasons.push("context_identity_missing");
    } else if (probe.sourceContextIdentity !== probe.targetContextIdentity) {
      reasons.push("context_identity_mismatch");
    }
  }
  if (reasons.length === 0) {
    return {
      decision: manifest.kvTransition.mode,
      allowed: true,
      reasons,
    };
  }
  return manifest.kvTransition.gate.onMismatch === "recompute-prefill"
    ? { decision: "recompute", allowed: true, reasons }
    : { decision: "reject", allowed: false, reasons };
}

function planDefaultRoute(
  request: RuntimePlanRequest,
  topology: DistributionTopology,
): DistributionPlan {
  if (request.planner?.kind === "macro-wave") {
    const planner = new MacroWaveRamVramPlanner({
      ...request.planner.options,
      candidateCodecs:
        request.planner.options?.candidateCodecs ??
        (request.allowLossyActivation
          ? ["fp16", "int8", "int8-grouped", "int8-hadamard"]
          : ["fp16"]),
    });
    const result = planner.evaluate(request.model, topology, {
      ...request.workload,
      maxStages: Math.min(request.workload.maxStages, MAX_WAN_VIRTUAL_STAGES),
    });
    if (!result.plan) {
      throw new Error(`no_feasible_macro_wave_pipeline:${result.reason}`);
    }
    return result.plan;
  }
  const searchOptions: SearchOptions = {
    ...DEFAULT_SEARCH_OPTIONS,
    // INT8 is implemented but not silently eligible: the real correctness
    // benchmark found deterministic token drift on SmolLM2. Q4 remains purely
    // analytical until a quality-checked kernel exists.
    candidateCodecs: request.allowLossyActivation
      ? ["fp16", "int8", "int8-grouped", "int8-hadamard"]
      : ["fp16"],
  };
  const planner =
    topology.nodes.length > 64
      ? new FleetTopologyPlanner(searchOptions)
      : new TopologyBeamPlanner(searchOptions);
  const plan = planner.plan(request.model, topology, {
    ...request.workload,
    maxStages: Math.min(
      request.workload.maxStages,
      MAX_WAN_VIRTUAL_STAGES,
    ),
  });
  if (!plan) throw new Error("no_feasible_runtime_pipeline");
  return plan;
}

function validateSelectedPlanningMode(
  request: RuntimePlanRequest,
  prefill: DistributionPlan,
  decode: DistributionPlan,
): void {
  const explicitlySelected = request.planner?.kind === "macro-wave";
  const plans = [prefill, decode] as const;
  const containsMacroWave = plans.some(
    (plan) =>
      plan.macroWave !== undefined ||
      plan.stages.some((stage) => stage.macroWave !== undefined) ||
      plan.algorithm.startsWith("macro-wave"),
  );
  if (containsMacroWave && !explicitlySelected) {
    throw new Error("runtime_macro_wave_plan_requires_explicit_selection");
  }
  if (!explicitlySelected) return;
  if (plans.some((plan) => plan.macroWave === undefined)) {
    throw new Error("runtime_macro_wave_selection_requires_contract_plans");
  }
  if (
    (request.tensorParallelCells?.length ?? 0) > 0 ||
    (request.phaseTensorParallelCells?.prefill?.length ?? 0) > 0 ||
    (request.phaseTensorParallelCells?.decode?.length ?? 0) > 0
  ) {
    throw new Error("runtime_macro_wave_and_tensor_parallel_cells_are_not_composable");
  }
  const nodes = new Map(request.topology.nodes.map((node) => [node.id, node]));
  for (const plan of plans) {
    for (const stage of plan.stages) {
      const execution = stage.macroWave;
      if (!execution || execution.mode !== "macro-wave-memory") {
        throw new Error("runtime_macro_wave_stage_contract_is_missing");
      }
      validateMacroWaveStageExecution(
        execution as unknown as Record<string, unknown>,
      );
      const node = nodes.get(stage.nodeId);
      const hierarchy = node?.ramVram;
      if (!hierarchy) {
        throw new Error(`runtime_macro_wave_node_profile_is_missing:${stage.nodeId}`);
      }
      if (
        execution.budgets.hostRamBytes !== hierarchy.usableRamBytes ||
        execution.budgets.vramBytes !== hierarchy.usableVramBytes
      ) {
        throw new Error(`runtime_macro_wave_node_budget_mismatch:${stage.nodeId}`);
      }
      if (
        execution.memoryMode === "resident" &&
        execution.residentKind !== (hierarchy.residentKind ?? "layers")
      ) {
        throw new Error(`runtime_macro_wave_resident_kind_mismatch:${stage.nodeId}`);
      }
    }
  }
}

function evaluateRuntimePlan(
  request: RuntimePlanRequest,
  topology: DistributionTopology,
  plan: DistributionPlan,
  phase: "prefill" | "decode",
): DistributionMetrics {
  if (plan.stages.length > MAX_WAN_VIRTUAL_STAGES) {
    throw new Error(`runtime_${phase}_plan_exceeds_wan_stage_limit`);
  }
  const codec = runtimeCodec(plan.codec);
  if (codec !== "fp16" && !request.allowLossyActivation) {
    throw new Error(`lossy_runtime_codec_requires_opt_in:${phase}`);
  }
  const predicted = evaluateDistributionPlan(
    request.model,
    plan.macroWave ? macroWaveEvaluationTopology(topology, plan) : topology,
    request.workload,
    plan,
  );
  if (!predicted.feasible) {
    throw new Error(
      `planned_${phase}_pipeline_is_infeasible:${predicted.infeasibleReason}`,
    );
  }
  if (plan.macroWave) {
    const projection = plan.macroWave.projection;
    Object.assign(predicted, {
      ttftMs: projection.ttftMs,
      tpotMs: projection.tpotMs,
      responseTimeMs: projection.responseTimeMs,
      pathDecodeMs: projection.pathDecodeMs,
      pipelineCycleMs: projection.pipelineCycleMs,
      tokensPerSecondPerSequence: projection.tokensPerSecondPerSequence,
      aggregateTokensPerSecond: projection.aggregateTokensPerSecond,
      networkBytesPerOutputToken: projection.networkBytesPerOutputToken,
      routeAvailability: projection.routeAvailability,
      calibrationRequired: true,
      calibrationReasons: [
        ...new Set([
          ...(predicted.calibrationReasons ?? []),
          "macro_wave_projection_requires_physical_calibration",
        ]),
      ],
    });
  }
  return predicted;
}

function macroWaveEvaluationTopology(
  topology: DistributionTopology,
  plan: DistributionPlan,
): DistributionTopology {
  const capacities = new Map(
    plan.stages.map((stage) => {
      const execution = stage.macroWave;
      if (!execution) throw new Error("runtime_macro_wave_stage_contract_is_missing");
      const capacity = execution.budgets.hostRamBytes + execution.budgets.vramBytes;
      if (!Number.isSafeInteger(capacity)) {
        throw new Error("runtime_macro_wave_combined_budget_is_invalid");
      }
      return [stage.nodeId, capacity] as const;
    }),
  );
  return {
    nodes: topology.nodes.map((node) => ({
      ...node,
      memoryBytes: capacities.get(node.id) ?? node.memoryBytes,
      reserveBytes: capacities.has(node.id) ? 0 : node.reserveBytes,
    })),
    links: topology.links,
  };
}

function distributionTopology(topology: RuntimeTopology): DistributionTopology {
  return {
    nodes: topology.nodes.map(
      ({
        endpoint: _endpoint,
        backend: _backend,
        capabilities: _capabilities,
        ramVram,
        ...node
      }) => ({
        ...node,
        ...(ramVram ? { ramVram: structuredClone(ramVram) } : {}),
      }),
    ),
    links: topology.links,
  };
}

/**
 * Collapse every declared physical TP cell into its rank-zero route node for
 * planning.  Its usable memory and failure domain are aggregated, while the
 * helper ranks are removed from the ordinary pipeline candidate pool.
 *
 * Compute time is intentionally left at the anchor calibration: until a
 * collective-aware calibration exists, memory feasibility is real but no TP
 * speed-up is invented by the cost model.
 */
function cellAwarePlanningTopology(
  topology: DistributionTopology,
  cells: readonly RuntimeTensorParallelCellPlanningInput[],
): DistributionTopology {
  if (cells.length === 0) return topology;
  const nodes = new Map(topology.nodes.map((node) => [node.id, { ...node }]));
  const reservedMembers = new Set<string>();
  const stageIndexes = new Set<number>();
  for (const cell of cells) {
    if (!isRecord(cell)) throw new Error("runtime_cell_plan_is_invalid");
    const stageIndex = asInteger(
      cell.stageIndex,
      "runtime_cell_plan_stage_index_is_invalid",
    );
    if (stageIndex < 1 || stageIndexes.has(stageIndex)) {
      throw new Error("runtime_cell_plan_stage_index_is_invalid");
    }
    stageIndexes.add(stageIndex);
    validateStringList(
      cell.memberNodeIds,
      "runtime_cell_plan_members_are_invalid",
    );
    if (cell.memberNodeIds.length < 2) {
      throw new Error("runtime_cell_plan_requires_multiple_members");
    }
    if (
      !isRecord(cell.execution) ||
      cell.execution.mode !== "tensor-parallel-cell" ||
      !isRecord(cell.execution.fixture) ||
      (cell.execution.fixture.location !== "anchor-local" &&
        cell.execution.fixture.location !== "member-local") ||
      cell.execution.worldSize !== cell.memberNodeIds.length ||
      !Array.isArray(cell.execution.rankMemberIds) ||
      !stringListEquals(cell.execution.rankMemberIds, cell.memberNodeIds)
    ) {
      throw new Error("runtime_cell_plan_execution_members_mismatch");
    }
    const members = cell.memberNodeIds.map((nodeId) => {
      if (reservedMembers.has(nodeId)) {
        throw new Error(`runtime_cell_plan_member_reused:${nodeId}`);
      }
      reservedMembers.add(nodeId);
      const node = nodes.get(nodeId);
      if (!node) throw new Error(`runtime_cell_plan_unknown_member:${nodeId}`);
      return node;
    });
    const anchor = members[0]!;
    let internalLinkAvailability = 1;
    if (cell.execution.fixture.location === "member-local") {
      for (const member of members.slice(1)) {
        const outbound = topology.links.find(
          (link) => link.from === anchor.id && link.to === member.id,
        );
        const inbound = topology.links.find(
          (link) => link.from === member.id && link.to === anchor.id,
        );
        if (!outbound || !inbound) {
          throw new Error(
            `runtime_cell_plan_missing_internal_link:${anchor.id}<->${member.id}`,
          );
        }
        internalLinkAvailability *= outbound.availability ?? 1;
        internalLinkAvailability *= inbound.availability ?? 1;
      }
    }
    const aggregateLimit = members.reduce(
      (total, member) => total + Math.max(0, member.memoryBytes - member.reserveBytes),
      0,
    );
    if (!Number.isSafeInteger(aggregateLimit) || aggregateLimit < 1) {
      throw new Error("runtime_cell_plan_memory_limit_is_invalid");
    }
    const aggregateMemoryBytes = anchor.reserveBytes + aggregateLimit;
    if (!Number.isSafeInteger(aggregateMemoryBytes)) {
      throw new Error("runtime_cell_plan_memory_limit_is_invalid");
    }
    nodes.set(anchor.id, {
      ...anchor,
      memoryBytes: aggregateMemoryBytes,
      powerWatts: members.reduce((total, member) => total + member.powerWatts, 0),
      availability: members.reduce(
        (availability, member) => availability * member.availability,
        internalLinkAvailability,
      ),
    });
  }
  const helperIds = new Set(
    cells.flatMap((cell) => cell.memberNodeIds.slice(1)),
  );
  const collapsedNodes = [...nodes.values()].filter((node) => !helperIds.has(node.id));
  const collapsedIds = new Set(collapsedNodes.map((node) => node.id));
  return {
    nodes: collapsedNodes,
    links: topology.links.filter(
      (link) => collapsedIds.has(link.from) && collapsedIds.has(link.to),
    ),
  };
}

function materializePlannedTensorParallelCells(
  manifestValue: RuntimePipelineManifestV2,
  request: RuntimePlanRequest,
  cells: readonly RuntimeTensorParallelCellPlanningInput[],
): RuntimePipelineManifestV2 {
  if (cells.length === 0) return manifestValue;
  const nodes = new Map(request.topology.nodes.map((node) => [node.id, node]));
  let manifest = manifestValue;
  for (const cell of [...cells].sort((left, right) => left.stageIndex - right.stageIndex)) {
    const stage = manifest.plans.prefill.stages[cell.stageIndex];
    if (!stage) throw new Error("runtime_cell_plan_stage_is_missing");
    if (stage.anchor.memberId !== cell.memberNodeIds[0]) {
      throw new Error("runtime_cell_plan_anchor_not_selected");
    }
    const physicalNodes = cell.memberNodeIds.map((nodeId) => {
      const node = nodes.get(nodeId);
      if (!node) throw new Error(`runtime_cell_plan_unknown_member:${nodeId}`);
      return node;
    });
    const limits = physicalNodes.map((node) =>
      Math.max(0, node.memoryBytes - node.reserveBytes),
    );
    const assignments = cell.execution.fixture.rankMemory.map((rank, index) => {
      const expected = plannedRankMemoryBytes(
        rank.fixedBytes,
        rank.kvBytesPerToken,
        request.workload.contextTokens,
        request.workload.concurrentSequences,
      );
      if (rank.requiredBytes !== expected) {
        throw new Error(`runtime_cell_plan_rank_memory_mismatch:${index}`);
      }
      if (expected > limits[index]!) {
        throw new Error(`runtime_cell_plan_rank_memory_exceeded:${cell.memberNodeIds[index]}`);
      }
      return expected;
    });
    const members: RuntimeStageMemberManifest[] = physicalNodes.map((node, index) => ({
      nodeId: node.id,
      endpoint: { ...node.endpoint },
      backend: normalizeBackend(node.backend),
      capabilities: normalizeCapabilities(
        node.capabilities,
        manifest.plans.prefill.activationCodec,
      ),
      assignedMemoryBytes: assignments[index]!,
      memoryLimitBytes: limits[index]!,
    }));
    manifest = materializeRuntimeTensorParallelCell(manifest, {
      stageIndex: cell.stageIndex,
      members,
      execution: structuredClone(cell.execution),
    });
    for (const plan of [manifest.plans.prefill, manifest.plans.decode]) {
      applyTensorParallelEvidence(
        plan.predicted,
        cell.memberNodeIds,
        request.topology,
      );
    }
    sealRuntimePipelineIdentities(manifest);
    validateRuntimePipelineManifest(manifest);
  }
  return manifest;
}

/**
 * Materialize independently compiled, resident TP fixtures into each phase.
 * A differing execution changes the stage/plan/pipeline identities and must
 * use recomputed KV. Nothing here copies weights, reshards a live process or
 * claims that the current Python server can switch between the two fixtures.
 */
function materializePlannedTensorParallelCellsByPhase(
  manifestValue: RuntimePipelineManifestV2,
  request: RuntimePlanRequest,
  prefillCells: readonly RuntimeTensorParallelCellPlanningInput[],
  decodeCells: readonly RuntimeTensorParallelCellPlanningInput[],
): RuntimePipelineManifestV2 {
  const manifest = structuredClone(manifestValue);
  const nodes = new Map(request.topology.nodes.map((node) => [node.id, node]));
  for (const [phase, cells] of [
    ["prefill", prefillCells],
    ["decode", decodeCells],
  ] as const) {
    for (const cell of [...cells].sort(
      (left, right) => left.stageIndex - right.stageIndex,
    )) {
      materializePlannedTensorParallelCellForPhase(
        manifest,
        request,
        nodes,
        phase,
        cell,
      );
    }
  }
  sealRuntimePipelineIdentities(manifest);
  validateRuntimePipelineManifest(manifest);
  return manifest;
}

function materializePlannedTensorParallelCellForPhase(
  manifest: RuntimePipelineManifestV2,
  request: RuntimePlanRequest,
  nodes: Map<string, RuntimeNodeProfile>,
  phase: "prefill" | "decode",
  cell: RuntimeTensorParallelCellPlanningInput,
): void {
  const plan = manifest.plans[phase];
  const stage = plan.stages[cell.stageIndex];
  if (!stage || stage.first || stage.last) {
    throw new Error(`runtime_${phase}_cell_plan_stage_must_be_intermediate`);
  }
  if (stage.anchor.memberId !== cell.memberNodeIds[0]) {
    throw new Error(`runtime_${phase}_cell_plan_anchor_not_selected`);
  }
  const physicalNodes = cell.memberNodeIds.map((nodeId) => {
    const node = nodes.get(nodeId);
    if (!node) throw new Error(`runtime_cell_plan_unknown_member:${nodeId}`);
    return node;
  });
  const limits = physicalNodes.map((node) =>
    Math.max(0, node.memoryBytes - node.reserveBytes),
  );
  const assignments = cell.execution.fixture.rankMemory.map((rank, index) => {
    const expected = plannedRankMemoryBytes(
      rank.fixedBytes,
      rank.kvBytesPerToken,
      request.workload.contextTokens,
      request.workload.concurrentSequences,
    );
    if (rank.requiredBytes !== expected) {
      throw new Error(`runtime_${phase}_cell_plan_rank_memory_mismatch:${index}`);
    }
    if (expected > limits[index]!) {
      throw new Error(
        `runtime_${phase}_cell_plan_rank_memory_exceeded:${cell.memberNodeIds[index]}`,
      );
    }
    return expected;
  });
  const members: RuntimeStageMemberManifest[] = physicalNodes.map((node, index) => ({
    nodeId: node.id,
    endpoint: { ...node.endpoint },
    backend: normalizeBackend(node.backend),
    capabilities: normalizeCapabilities(node.capabilities, plan.activationCodec),
    assignedMemoryBytes: assignments[index]!,
    memoryLimitBytes: limits[index]!,
  }));
  stage.members = members;
  stage.execution = structuredClone(cell.execution);
  stage.memoryBytes = assignments.reduce((total, value) => total + value, 0);
  stage.memoryLimitBytes = limits.reduce((total, value) => total + value, 0);
  const predictedStage = plan.predicted.stageMetrics[cell.stageIndex];
  if (!predictedStage) {
    throw new Error(`runtime_${phase}_cell_plan_prediction_is_missing`);
  }
  predictedStage.memoryBytes = stage.memoryBytes;
  predictedStage.memoryLimitBytes = stage.memoryLimitBytes;
  plan.predicted.peakStageMemoryBytes = Math.max(
    ...plan.predicted.stageMetrics.map((metrics) => metrics.memoryBytes),
  );
  plan.predicted.calibrationRequired = true;
  plan.predicted.calibrationReasons = [
    ...new Set([
      ...(plan.predicted.calibrationReasons ?? []),
      "tensor_parallel_collective_cost_unprofiled",
      "phase_specific_tensor_parallel_profile",
      ...(cell.execution.fixture.location === "member-local"
        ? ["external_cell_internal_links_unprofiled"]
        : []),
    ]),
  ];
  applyTensorParallelEvidence(
    plan.predicted,
    cell.memberNodeIds,
    request.topology,
  );
}

function applyTensorParallelEvidence(
  predicted: DistributionMetrics,
  memberNodeIds: string[],
  topology: RuntimeTopology,
  now = Date.now(),
): void {
  const gateReason = tensorParallelCellGateReason(memberNodeIds, topology, now);
  if (gateReason !== null) {
    predicted.calibrationRequired = true;
    predicted.calibrationReasons = [
      ...new Set([
        ...(predicted.calibrationReasons ?? []),
        gateReason,
      ]),
    ];
    return;
  }
  const remainingCalibrationReasons = (
    predicted.calibrationReasons ?? []
  ).filter(
    (reason) =>
      reason !== "tensor_parallel_collective_cost_unprofiled" &&
      reason !== "external_cell_internal_links_unprofiled" &&
      !reason.startsWith("tp_cell_") &&
      reason !== "collective_link_missing",
  );
  predicted.calibrationRequired = remainingCalibrationReasons.length > 0;
  if (remainingCalibrationReasons.length === 0) {
    delete predicted.calibrationReasons;
  } else {
    predicted.calibrationReasons = remainingCalibrationReasons;
  }
}

function plannedRankMemoryBytes(
  fixedBytes: number,
  kvBytesPerToken: number,
  contextTokens: number,
  concurrentSequences: number,
): number {
  for (const value of [fixedBytes, kvBytesPerToken, contextTokens, concurrentSequences]) {
    assertNonNegativeInteger(value, "runtime_cell_plan_rank_memory_is_invalid");
  }
  const required =
    BigInt(fixedBytes) +
    BigInt(kvBytesPerToken) * BigInt(contextTokens) * BigInt(concurrentSequences);
  if (required > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error("runtime_cell_plan_rank_memory_is_invalid");
  }
  return Number(required);
}

function buildVirtualStages(
  plan: DistributionPlan,
  predicted: DistributionMetrics,
  nodes: Map<string, RuntimeNodeProfile>,
  phase: "prefill" | "decode",
): RuntimeVirtualStageManifest[] {
  const codec = runtimeCodec(plan.codec);
  return plan.stages.map((stage, index) => {
    const node = nodes.get(stage.nodeId);
    const metrics = predicted.stageMetrics[index];
    if (!node || !metrics) throw new Error(`runtime_node_missing:${stage.nodeId}`);
    const member: RuntimeStageMemberManifest = {
      nodeId: stage.nodeId,
      endpoint: { ...node.endpoint },
      backend: normalizeBackend(node.backend),
      capabilities: normalizeCapabilities(node.capabilities, codec),
      assignedMemoryBytes: metrics.memoryBytes,
      memoryLimitBytes: metrics.memoryLimitBytes,
    };
    return {
      stageId: "pending",
      index,
      layerStart: stage.layerStart,
      layerEnd: stage.layerEnd,
      first: index === 0,
      last: index === plan.stages.length - 1,
      anchor: {
        memberId: member.nodeId,
        endpoint: { ...member.endpoint },
      },
      members: [member],
      ...(stage.macroWave
        ? { macroWave: structuredClone(stage.macroWave) }
        : {}),
      memoryBytes: metrics.memoryBytes,
      memoryLimitBytes: metrics.memoryLimitBytes,
    };
  });
}

function sealRuntimePipelineIdentities(manifest: RuntimePipelineManifestV2): void {
  for (const plan of [manifest.plans.prefill, manifest.plans.decode]) {
    for (const stage of plan.stages) stage.stageId = runtimeVirtualStageId(stage);
    plan.planId = runtimePhasePlanId(plan);
  }
  manifest.pipelineId = runtimePipelineId(manifest);
}

function validateRuntimePipelineIdentities(manifest: RuntimePipelineManifestV2): void {
  for (const plan of [manifest.plans.prefill, manifest.plans.decode]) {
    for (const stage of plan.stages) {
      if (stage.stageId !== runtimeVirtualStageId(stage)) {
        throw new Error(`runtime_stage_id_mismatch:${plan.phase}:${stage.index}`);
      }
    }
    if (plan.planId !== runtimePhasePlanId(plan)) {
      throw new Error(`runtime_plan_id_mismatch:${plan.phase}`);
    }
  }
  if (manifest.pipelineId !== runtimePipelineId(manifest)) {
    throw new Error("runtime_pipeline_id_mismatch");
  }
}

function runtimeVirtualStageId(stage: RuntimeVirtualStageManifest): string {
  return `vs-${digest(JSON.stringify(runtimeStageIdentity(stage)), 16)}`;
}

function runtimePhasePlanId(
  plan: RuntimePrefillPlanManifest | RuntimeDecodePlanManifest,
): string {
  return digest(
    JSON.stringify({
      phase: plan.phase,
      activationCodec: plan.activationCodec,
      transport: plan.transport,
      microBatchSize: plan.microBatchSize,
      macroWave: plan.macroWave
        ? runtimeMacroWavePlanIdentity(plan.macroWave)
        : null,
      ...(plan.phase === "prefill"
        ? { chunkTokens: plan.chunkTokens }
        : {
            directTokenReturnStage: plan.directTokenReturnStage,
            speculation: runtimeSpeculationIdentity(plan.speculation),
          }),
      stageIds: plan.stages.map((stage) => stage.stageId),
    }),
    20,
  );
}

function runtimeMacroWavePlanIdentity(contract: MacroWavePlanContractV1): object {
  return {
    schema: contract.schema,
    routeKind: contract.routeKind,
    waveTokens: contract.waveTokens,
    expectedCommittedTokensPerWave: contract.expectedCommittedTokensPerWave,
    projection: { ...contract.projection },
  };
}

function runtimePipelineId(manifest: RuntimePipelineManifestV2): string {
  return digest(
    JSON.stringify({
      protocol: manifest.protocol,
      modelId: manifest.modelId,
      modelRevision: manifest.modelRevision,
      tokenizerId: manifest.tokenizerId,
      totalLayers: manifest.totalLayers,
      hiddenSize: manifest.hiddenSize,
      prefillPlanId: manifest.plans.prefill.planId,
      decodePlanId: manifest.plans.decode.planId,
      kvTransition: {
        mode: manifest.kvTransition.mode,
        format: {
          id: manifest.kvTransition.format.id,
          version: manifest.kvTransition.format.version,
          dtype: manifest.kvTransition.format.dtype,
          layout: manifest.kvTransition.format.layout,
        },
        gate: {
          policy: manifest.kvTransition.gate.policy,
          modelRevision: manifest.kvTransition.gate.modelRevision,
          tokenizerId: manifest.kvTransition.gate.tokenizerId,
          layerLayoutHash: manifest.kvTransition.gate.layerLayoutHash,
          requireContextIdentity: manifest.kvTransition.gate.requireContextIdentity,
          onMismatch: manifest.kvTransition.gate.onMismatch,
        },
      },
    }),
    24,
  );
}

function runtimeStageIdentity(stage: RuntimeVirtualStageManifest): object {
  return {
    index: stage.index,
    layers: [stage.layerStart, stage.layerEnd],
    first: stage.first,
    last: stage.last,
    anchor: {
      memberId: stage.anchor.memberId,
      endpoint: runtimeEndpointIdentity(stage.anchor.endpoint),
    },
    members: stage.members.map(runtimeMemberIdentity),
    ...(stage.execution
      ? { execution: runtimeCellExecutionIdentity(stage.execution) }
      : {}),
    ...(stage.macroWave
      ? { macroWave: runtimeMacroWaveStageIdentity(stage.macroWave) }
      : {}),
    memoryBytes: stage.memoryBytes,
    memoryLimitBytes: stage.memoryLimitBytes,
  };
}

function runtimeMacroWaveStageIdentity(
  execution: MacroWaveStageExecutionContractV1,
): object {
  return {
    mode: execution.mode,
    schema: execution.schema,
    memoryMode: execution.memoryMode,
    residentKind: execution.residentKind,
    budgets: { ...execution.budgets },
    requirements: { ...execution.requirements },
    workingSet: { ...execution.workingSet },
    cachePolicy: { ...execution.cachePolicy },
    ...(execution.ramArtifact
      ? { ramArtifact: structuredClone(execution.ramArtifact) }
      : {}),
  };
}

function runtimeMemberIdentity(member: RuntimeStageMemberManifest): object {
  return {
    nodeId: member.nodeId,
    endpoint: runtimeEndpointIdentity(member.endpoint),
    backend: {
      engine: member.backend.engine,
      version: member.backend.version ?? null,
      modelFormats: [...member.backend.modelFormats],
      executionModes: [...member.backend.executionModes],
    },
    capabilities: {
      deviceKinds: [...member.capabilities.deviceKinds],
      computeApis: [...member.capabilities.computeApis],
      weightDtypes: [...member.capabilities.weightDtypes],
      activationCodecs: [...member.capabilities.activationCodecs],
      features: [...member.capabilities.features],
    },
    assignedMemoryBytes: member.assignedMemoryBytes,
    memoryLimitBytes: member.memoryLimitBytes,
  };
}

function runtimeCellExecutionIdentity(
  execution: RuntimeTensorParallelCellExecutionManifest,
): object {
  return {
    mode: execution.mode,
    engine: execution.engine,
    collectiveBackend: execution.collectiveBackend,
    computeDtype: execution.computeDtype,
    fixture: {
      schema: execution.fixture.schema,
      location: execution.fixture.location,
      path: execution.fixture.path,
      layerCount: execution.fixture.layerCount,
      manifestSha256: execution.fixture.manifestSha256,
      shardSha256: [...execution.fixture.shardSha256],
      rankMemory: execution.fixture.rankMemory.map((rank) => ({
        fixedBytes: rank.fixedBytes,
        kvBytesPerToken: rank.kvBytesPerToken,
        requiredBytes: rank.requiredBytes,
      })),
    },
    worldSize: execution.worldSize,
    rankMemberIds: [...execution.rankMemberIds],
    rankWeights: [...execution.rankWeights],
    rankDevices: [...execution.rankDevices],
    operationTimeoutSeconds: execution.operationTimeoutSeconds,
    ...(execution.external
      ? {
          external: {
            rankFixturePaths: [...execution.external.rankFixturePaths],
            controlBindHost: execution.external.controlBindHost,
            controlAdvertiseHost: execution.external.controlAdvertiseHost,
            controlPort: execution.external.controlPort,
            distributedAdvertiseHost: execution.external.distributedAdvertiseHost,
            distributedPort: execution.external.distributedPort,
            startupTimeoutSeconds: execution.external.startupTimeoutSeconds,
          },
        }
      : {}),
  };
}

function runtimeEndpointIdentity(endpoint: RuntimeEndpoint): object {
  return { host: endpoint.host, port: endpoint.port };
}

function runtimeSpeculationIdentity(policy: RuntimeSpeculationPolicy): object {
  return {
    mode: policy.mode,
    controller: policy.controller,
    defaultStrategyId: policy.defaultStrategyId,
    fallbackStrategyId: policy.fallbackStrategyId,
    acceptanceWindowTokens: policy.acceptanceWindowTokens,
    strategies: policy.strategies.map((strategy) => ({
      id: strategy.id,
      kind: strategy.kind,
      maxDraftTokens: strategy.maxDraftTokens,
      ...(strategy.maxBranches !== undefined
        ? { maxBranches: strategy.maxBranches }
        : {}),
      ...(strategy.maxBranchTokens !== undefined
        ? { maxBranchTokens: strategy.maxBranchTokens }
        : {}),
      ...(strategy.maxKvBytes !== undefined
        ? { maxKvBytes: strategy.maxKvBytes }
        : {}),
      ...(strategy.maxWaveTokens !== undefined
        ? { maxWaveTokens: strategy.maxWaveTokens }
        : {}),
      minAcceptanceRate: strategy.minAcceptanceRate,
      maxWasteRatio: strategy.maxWasteRatio,
      priority: strategy.priority,
      artifactId: strategy.artifactId ?? null,
    })),
  };
}

function normalizeBackend(input: RuntimeBackendInput | undefined): RuntimeBackendProfile {
  return {
    engine: input?.engine ?? "unspecified",
    ...(input?.version ? { version: input.version } : {}),
    modelFormats: [...(input?.modelFormats ?? ["unspecified"])],
    executionModes: [...(input?.executionModes ?? ["layer-range"])],
  };
}

function normalizeCapabilities(
  input: RuntimeMemberCapabilitiesInput | undefined,
  codec: RuntimeActivationCodec,
): RuntimeMemberCapabilities {
  return {
    deviceKinds: [...(input?.deviceKinds ?? ["unspecified"])],
    computeApis: [...(input?.computeApis ?? ["unspecified"])],
    weightDtypes: [...(input?.weightDtypes ?? ["unspecified"])],
    activationCodecs: [...(input?.activationCodecs ?? [codec])],
    features: [...(input?.features ?? ["layer-range", "kv-reuse"])],
  };
}

function cloneSpeculationPolicy(policy: RuntimeSpeculationPolicy): RuntimeSpeculationPolicy {
  return {
    ...policy,
    strategies: policy.strategies.map((strategy) => ({ ...strategy })),
  };
}

function runtimeCodec(codec: ActivationCodecId): RuntimeActivationCodec {
  if (codec === "q4") throw new Error("runtime_codec_not_implemented:q4");
  return codec;
}

function isRuntimeActivationCodec(value: unknown): value is RuntimeActivationCodec {
  return (
    value === "fp16" ||
    value === "int8" ||
    value === "int8-grouped" ||
    value === "int8-hadamard"
  );
}

function modelLayerLayoutHash(model: DistributedModelProfile): string {
  return digest(
    JSON.stringify(
      model.layers.map((layer) => [
        layer.index,
        layer.activationElements,
        layer.kvBytesPerToken,
      ]),
    ),
    32,
  );
}

function samePlannedCellProfiles(
  prefill: readonly RuntimeTensorParallelCellPlanningInput[],
  decode: readonly RuntimeTensorParallelCellPlanningInput[],
): boolean {
  if (prefill.length !== decode.length) return false;
  const left = [...prefill].sort((a, b) => a.stageIndex - b.stageIndex);
  const right = [...decode].sort((a, b) => a.stageIndex - b.stageIndex);
  return left.every((cell, index) => {
    const other = right[index];
    return (
      other !== undefined &&
      cell.stageIndex === other.stageIndex &&
      stringListEquals(cell.memberNodeIds, other.memberNodeIds) &&
      stageExecutionEquals(cell.execution, other.execution)
    );
  });
}

function sameExecutionRoute(
  prefill: RuntimeVirtualStageManifest[],
  decode: RuntimeVirtualStageManifest[],
): boolean {
  if (prefill.length !== decode.length) return false;
  return prefill.every((stage, index) => {
    const other = decode[index];
    return (
      other !== undefined &&
      stage.layerStart === other.layerStart &&
      stage.layerEnd === other.layerEnd &&
      stage.anchor.memberId === other.anchor.memberId &&
      endpointEquals(stage.anchor.endpoint, other.anchor.endpoint) &&
      stageExecutionEquals(stage.execution, other.execution) &&
      macroWaveStageEquals(stage.macroWave, other.macroWave) &&
      stage.members.length === other.members.length &&
      stage.members.every((member, memberIndex) => {
        const otherMember = other.members[memberIndex];
        return (
          otherMember !== undefined &&
          stageMemberExecutionEquals(member, otherMember)
        );
      })
    );
  });
}

function stageMemberExecutionEquals(
  left: RuntimeStageMemberManifest,
  right: RuntimeStageMemberManifest,
): boolean {
  return (
    left.nodeId === right.nodeId &&
    endpointEquals(left.endpoint, right.endpoint) &&
    left.backend.engine === right.backend.engine &&
    left.backend.version === right.backend.version &&
    stringListEquals(left.backend.modelFormats, right.backend.modelFormats) &&
    stringListEquals(left.backend.executionModes, right.backend.executionModes) &&
    stringListEquals(left.capabilities.deviceKinds, right.capabilities.deviceKinds) &&
    stringListEquals(left.capabilities.computeApis, right.capabilities.computeApis) &&
    stringListEquals(left.capabilities.weightDtypes, right.capabilities.weightDtypes) &&
    stringListEquals(
      left.capabilities.activationCodecs,
      right.capabilities.activationCodecs,
    ) &&
    stringListEquals(left.capabilities.features, right.capabilities.features)
  );
}

function stringListEquals(left: readonly string[], right: readonly string[]): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function numberListEquals(left: readonly number[], right: readonly number[]): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function stageExecutionEquals(
  left: RuntimeVirtualStageExecutionManifest | undefined,
  right: RuntimeVirtualStageExecutionManifest | undefined,
): boolean {
  if (left === undefined || right === undefined) return left === right;
  const sameExternal =
    left.external === undefined || right.external === undefined
      ? left.external === right.external
      : stringListEquals(
          left.external.rankFixturePaths,
          right.external.rankFixturePaths,
        ) &&
        left.external.controlBindHost === right.external.controlBindHost &&
        left.external.controlAdvertiseHost === right.external.controlAdvertiseHost &&
        left.external.controlPort === right.external.controlPort &&
        left.external.distributedAdvertiseHost ===
          right.external.distributedAdvertiseHost &&
        left.external.distributedPort === right.external.distributedPort &&
        left.external.startupTimeoutSeconds === right.external.startupTimeoutSeconds;
  return (
    left.mode === right.mode &&
    left.engine === right.engine &&
    left.collectiveBackend === right.collectiveBackend &&
    left.computeDtype === right.computeDtype &&
    left.fixture.schema === right.fixture.schema &&
    left.fixture.location === right.fixture.location &&
    left.fixture.path === right.fixture.path &&
    left.fixture.layerCount === right.fixture.layerCount &&
    left.fixture.manifestSha256 === right.fixture.manifestSha256 &&
    stringListEquals(left.fixture.shardSha256, right.fixture.shardSha256) &&
    left.fixture.rankMemory.length === right.fixture.rankMemory.length &&
    left.fixture.rankMemory.every((rank, index) => {
      const other = right.fixture.rankMemory[index];
      return (
        other !== undefined &&
        rank.fixedBytes === other.fixedBytes &&
        rank.kvBytesPerToken === other.kvBytesPerToken &&
        rank.requiredBytes === other.requiredBytes
      );
    }) &&
    left.worldSize === right.worldSize &&
    stringListEquals(left.rankMemberIds, right.rankMemberIds) &&
    numberListEquals(left.rankWeights, right.rankWeights) &&
    stringListEquals(left.rankDevices, right.rankDevices) &&
    left.operationTimeoutSeconds === right.operationTimeoutSeconds &&
    sameExternal
  );
}

function macroWaveStageEquals(
  left: MacroWaveStageExecutionContractV1 | undefined,
  right: MacroWaveStageExecutionContractV1 | undefined,
): boolean {
  if (left === undefined || right === undefined) return left === right;
  return (
    JSON.stringify(runtimeMacroWaveStageIdentity(left)) ===
    JSON.stringify(runtimeMacroWaveStageIdentity(right))
  );
}

function validateRuntimePipelineManifestV1(manifest: Record<string, unknown>): void {
  validateIdentity(manifest);
  if (manifest.activationCodec !== "fp16" && manifest.activationCodec !== "int8") {
    throw new Error("runtime_codec_not_implemented");
  }
  if (manifest.transport !== "persistent-tcp") {
    throw new Error("runtime_transport_not_implemented");
  }
  assertPositiveInteger(manifest.prefillChunkTokens, "runtime_prefill_chunk_is_invalid");
  assertPositiveInteger(manifest.microBatchSize, "runtime_microbatch_is_invalid");
  if (manifest.directTokenReturnStage !== 0) {
    throw new Error("runtime_direct_return_stage_is_invalid");
  }
  if (!Array.isArray(manifest.stages) || manifest.stages.length < 1) {
    throw new Error("runtime_pipeline_has_no_stages");
  }
  let nextLayer = 0;
  const nodeIds = new Set<string>();
  for (let index = 0; index < manifest.stages.length; index += 1) {
    const stage = manifest.stages[index];
    if (!isRecord(stage)) throw new Error("runtime_stage_must_be_an_object");
    if (stage.index !== index) throw new Error("runtime_stage_index_mismatch");
    const layerStart = asInteger(stage.layerStart, "runtime_layers_are_not_contiguous");
    const layerEnd = asInteger(stage.layerEnd, "runtime_layers_are_not_contiguous");
    if (layerStart !== nextLayer || layerEnd <= layerStart) {
      throw new Error("runtime_layers_are_not_contiguous");
    }
    if (
      stage.first !== (index === 0) ||
      stage.last !== (index === manifest.stages.length - 1)
    ) {
      throw new Error("runtime_stage_endpoint_flags_mismatch");
    }
    const memory = asNonNegativeInteger(stage.memoryBytes, "runtime_stage_memory_is_invalid");
    const limit = asNonNegativeInteger(
      stage.memoryLimitBytes,
      "runtime_stage_memory_limit_is_invalid",
    );
    if (memory > limit) throw new Error("runtime_stage_memory_exceeded");
    const nodeId = asNonEmptyString(stage.nodeId, "runtime_node_id_cannot_be_empty");
    if (nodeIds.has(nodeId)) throw new Error("runtime_node_reused");
    nodeIds.add(nodeId);
    validateEndpoint(stage.endpoint);
    nextLayer = layerEnd;
  }
  if (nextLayer !== manifest.totalLayers) throw new Error("runtime_layers_are_not_complete");
}

function validateRuntimePipelineManifestV2(manifest: Record<string, unknown>): void {
  validateIdentity(manifest);
  if (!isRecord(manifest.plans)) throw new Error("runtime_phase_plans_are_missing");
  if (!isRecord(manifest.plans.prefill)) throw new Error("runtime_prefill_plan_is_missing");
  if (!isRecord(manifest.plans.decode)) throw new Error("runtime_decode_plan_is_missing");
  const prefill = manifest.plans.prefill;
  const decode = manifest.plans.decode;
  if (prefill.phase !== "prefill") throw new Error("runtime_prefill_phase_is_invalid");
  if (decode.phase !== "decode") throw new Error("runtime_decode_phase_is_invalid");
  validatePhasePlan(prefill, "prefill", manifest.totalLayers);
  validatePhasePlan(decode, "decode", manifest.totalLayers);
  assertPositiveInteger(prefill.chunkTokens, "runtime_prefill_chunk_is_invalid");
  const decodeStages = decode.stages as RuntimeVirtualStageManifest[];
  const returnStage = asInteger(
    decode.directTokenReturnStage,
    "runtime_direct_return_stage_is_invalid",
  );
  if (returnStage < 0 || returnStage >= decodeStages.length) {
    throw new Error("runtime_direct_return_stage_is_invalid");
  }
  validateSpeculationPolicy(decode.speculation);
  validateMacroWaveSpeculationBinding(
    prefill.macroWave,
    decode.macroWave,
    decode.speculation as unknown as RuntimeSpeculationPolicy,
  );
  validateKvTransition(manifest.kvTransition);
  const kvTransition = manifest.kvTransition as unknown as RuntimeKvTransitionManifest;
  if (kvTransition.gate.modelRevision !== manifest.modelRevision) {
    throw new Error("runtime_kv_gate_model_revision_mismatch");
  }
  if (kvTransition.gate.tokenizerId !== manifest.tokenizerId) {
    throw new Error("runtime_kv_gate_tokenizer_mismatch");
  }
  if (
    kvTransition.mode === "in-place" &&
    !sameExecutionRoute(
      prefill.stages as RuntimeVirtualStageManifest[],
      decode.stages as RuntimeVirtualStageManifest[],
    )
  ) {
    throw new Error("runtime_kv_in_place_requires_identical_routes");
  }
  validateRuntimePipelineIdentities(
    manifest as unknown as RuntimePipelineManifestV2,
  );
}

function validateIdentity(manifest: Record<string, unknown>): void {
  asNonEmptyString(manifest.pipelineId, "runtime_pipeline_id_cannot_be_empty");
  asNonEmptyString(manifest.modelId, "runtime_model_id_cannot_be_empty");
  asNonEmptyString(manifest.modelRevision, "runtime_model_revision_cannot_be_empty");
  asNonEmptyString(manifest.tokenizerId, "runtime_tokenizer_id_cannot_be_empty");
  assertPositiveInteger(manifest.totalLayers, "runtime_total_layers_is_invalid");
  assertPositiveInteger(manifest.hiddenSize, "runtime_hidden_size_is_invalid");
}

function validatePhasePlan(
  plan: Record<string, unknown>,
  phase: "prefill" | "decode",
  totalLayers: unknown,
): void {
  asNonEmptyString(plan.planId, "runtime_plan_id_cannot_be_empty");
  if (!isRuntimeActivationCodec(plan.activationCodec)) {
    throw new Error("runtime_codec_not_implemented");
  }
  const expectedTransport =
    plan.macroWave === undefined
      ? "persistent-tcp"
      : "persistent-tcp-macro-wave-v1";
  if (plan.transport !== expectedTransport) {
    throw new Error("runtime_transport_not_implemented");
  }
  assertPositiveInteger(plan.microBatchSize, "runtime_microbatch_is_invalid");
  if (!Array.isArray(plan.stages) || plan.stages.length < 1) {
    throw new Error("runtime_pipeline_has_no_stages");
  }
  if (!isRecord(plan.predicted) || plan.predicted.feasible !== true) {
    throw new Error(`runtime_${phase}_prediction_is_not_feasible`);
  }
  validatePredictedMetrics(plan.predicted, plan.stages.length, phase);
  validateMacroWavePlanContract(plan.macroWave, plan.stages, plan.predicted);
  let nextLayer = 0;
  const memberIds = new Set<string>();
  const stageIds = new Set<string>();
  for (let index = 0; index < plan.stages.length; index += 1) {
    const stage = plan.stages[index];
    if (!isRecord(stage)) throw new Error("runtime_stage_must_be_an_object");
    if (stage.index !== index) throw new Error("runtime_stage_index_mismatch");
    const stageId = asNonEmptyString(stage.stageId, "runtime_stage_id_cannot_be_empty");
    if (stageIds.has(stageId)) throw new Error("runtime_stage_id_reused");
    stageIds.add(stageId);
    const layerStart = asInteger(stage.layerStart, "runtime_layers_are_not_contiguous");
    const layerEnd = asInteger(stage.layerEnd, "runtime_layers_are_not_contiguous");
    if (layerStart !== nextLayer || layerEnd <= layerStart) {
      throw new Error("runtime_layers_are_not_contiguous");
    }
    if (
      stage.first !== (index === 0) ||
      stage.last !== (index === plan.stages.length - 1)
    ) {
      throw new Error("runtime_stage_endpoint_flags_mismatch");
    }
    if (!Array.isArray(stage.members) || stage.members.length < 1) {
      throw new Error("runtime_virtual_stage_has_no_members");
    }
    if (!isRecord(stage.anchor)) throw new Error("runtime_stage_anchor_is_missing");
    const anchorId = asNonEmptyString(
      stage.anchor.memberId,
      "runtime_stage_anchor_member_is_invalid",
    );
    validateEndpoint(stage.anchor.endpoint);
    let assignedMemory = 0;
    let memoryLimit = 0;
    let anchorMember: RuntimeStageMemberManifest | undefined;
    for (const memberValue of stage.members) {
      if (!isRecord(memberValue)) throw new Error("runtime_stage_member_must_be_an_object");
      const memberId = asNonEmptyString(
        memberValue.nodeId,
        "runtime_node_id_cannot_be_empty",
      );
      if (memberIds.has(memberId)) throw new Error("runtime_node_reused_in_phase");
      memberIds.add(memberId);
      validateEndpoint(memberValue.endpoint);
      validateBackend(memberValue.backend);
      validateMemberCapabilities(memberValue.capabilities, plan.activationCodec);
      const assigned = asNonNegativeInteger(
        memberValue.assignedMemoryBytes,
        "runtime_member_memory_is_invalid",
      );
      const limit = asNonNegativeInteger(
        memberValue.memoryLimitBytes,
        "runtime_member_memory_limit_is_invalid",
      );
      if (assigned > limit) throw new Error("runtime_member_memory_exceeded");
      assignedMemory += assigned;
      memoryLimit += limit;
      if (memberId === anchorId) {
        anchorMember = memberValue as unknown as RuntimeStageMemberManifest;
      }
    }
    if (!anchorMember) throw new Error("runtime_stage_anchor_is_not_a_member");
    if (!endpointEquals(anchorMember.endpoint, stage.anchor.endpoint as RuntimeEndpoint)) {
      throw new Error("runtime_stage_anchor_endpoint_mismatch");
    }
    validateVirtualStageExecution(
      stage.execution,
      stage as unknown as RuntimeVirtualStageManifest,
    );
    const stageMemory = asNonNegativeInteger(
      stage.memoryBytes,
      "runtime_stage_memory_is_invalid",
    );
    const stageLimit = asNonNegativeInteger(
      stage.memoryLimitBytes,
      "runtime_stage_memory_limit_is_invalid",
    );
    if (stageMemory !== assignedMemory) throw new Error("runtime_stage_memory_sum_mismatch");
    if (stageLimit !== memoryLimit) throw new Error("runtime_stage_limit_sum_mismatch");
    if (stageMemory > stageLimit) throw new Error("runtime_stage_memory_exceeded");
    const stageExecution = stage.macroWave as unknown;
    if (isRecord(stageExecution) && stageExecution.mode === "macro-wave-memory") {
      const macro = stageExecution as unknown as MacroWaveStageExecutionContractV1;
      if (stageMemory !== macro.requirements.fullStageStateBytes) {
        throw new Error("runtime_macro_wave_stage_memory_mismatch");
      }
      if (
        stageLimit !== macro.budgets.hostRamBytes + macro.budgets.vramBytes
      ) {
        throw new Error("runtime_macro_wave_stage_budget_mismatch");
      }
    }
    const predictedStage = (plan.predicted.stageMetrics as unknown[])[index];
    if (
      !isRecord(predictedStage) ||
      predictedStage.nodeId !== anchorId ||
      predictedStage.layerStart !== layerStart ||
      predictedStage.layerEnd !== layerEnd ||
      predictedStage.memoryBytes !== stageMemory ||
      predictedStage.memoryLimitBytes !== stageLimit
    ) {
      throw new Error(`runtime_${phase}_stage_prediction_mismatch`);
    }
    nextLayer = layerEnd;
  }
  if (nextLayer !== totalLayers) throw new Error("runtime_layers_are_not_complete");
}

function validateMacroWavePlanContract(
  value: unknown,
  stageValues: unknown[],
  predicted: Record<string, unknown>,
): void {
  const macroStages = stageValues.filter(
    (stage) => isRecord(stage) && stage.macroWave !== undefined,
  );
  if (value === undefined) {
    if (
      macroStages.some(
        (stage) =>
          isRecord((stage as Record<string, unknown>).macroWave) &&
          ((stage as Record<string, unknown>).macroWave as Record<string, unknown>).mode ===
            "macro-wave-memory",
      )
    ) {
      throw new Error("runtime_macro_wave_stage_requires_plan_contract");
    }
    return;
  }
  if (!isRecord(value)) throw new Error("runtime_macro_wave_plan_contract_is_invalid");
  assertExactObjectKeys(
    value,
    [
      "schema",
      "routeKind",
      "waveTokens",
      "expectedCommittedTokensPerWave",
      "projection",
    ],
    "runtime_macro_wave_plan_contract_is_invalid",
  );
  if (value.schema !== "gdlp-macro-wave-plan/1") {
    throw new Error("runtime_macro_wave_plan_schema_is_not_supported");
  }
  if (value.routeKind !== "resident-baseline" && value.routeKind !== "macro-wave") {
    throw new Error("runtime_macro_wave_route_kind_is_invalid");
  }
  assertPositiveInteger(value.waveTokens, "runtime_macro_wave_tokens_are_invalid");
  assertPositiveFinite(
    value.expectedCommittedTokensPerWave,
    "runtime_macro_wave_committed_tokens_are_invalid",
  );
  if ((value.expectedCommittedTokensPerWave as number) < 1) {
    throw new Error("runtime_macro_wave_committed_tokens_are_invalid");
  }
  if ((value.expectedCommittedTokensPerWave as number) > (value.waveTokens as number)) {
    throw new Error("runtime_macro_wave_committed_tokens_exceed_wave");
  }
  if (!isRecord(value.projection)) {
    throw new Error("runtime_macro_wave_projection_is_invalid");
  }
  const projectionFields = [
    "ttftMs",
    "tpotMs",
    "responseTimeMs",
    "pathDecodeMs",
    "pipelineCycleMs",
    "tokensPerSecondPerSequence",
    "aggregateTokensPerSecond",
    "networkBytesPerOutputToken",
    "rawActiveWeightBytesPerOutputToken",
    "expectedWeightCacheMissBytesPerOutputToken",
    "routeAvailability",
  ];
  assertExactObjectKeys(
    value.projection,
    projectionFields,
    "runtime_macro_wave_projection_is_invalid",
  );
  for (const field of projectionFields) {
    if (field === "routeAvailability") {
      assertProbability(
        value.projection[field],
        "runtime_macro_wave_projection_is_invalid",
      );
    } else {
      assertNonNegativeFinite(
        value.projection[field],
        "runtime_macro_wave_projection_is_invalid",
      );
    }
  }
  for (const field of [
    "ttftMs",
    "tpotMs",
    "responseTimeMs",
    "pathDecodeMs",
    "pipelineCycleMs",
    "tokensPerSecondPerSequence",
    "aggregateTokensPerSecond",
    "networkBytesPerOutputToken",
    "routeAvailability",
  ]) {
    if (predicted[field] !== value.projection[field]) {
      throw new Error(`runtime_macro_wave_projection_mismatch:${field}`);
    }
  }
  let ramBackedStages = 0;
  for (const stage of stageValues) {
    if (!isRecord(stage) || !isRecord(stage.macroWave)) {
      throw new Error("runtime_macro_wave_stage_contract_is_missing");
    }
    if (
      stage.execution !== undefined ||
      !Array.isArray(stage.members) ||
      stage.members.length !== 1
    ) {
      throw new Error("runtime_macro_wave_stage_requires_single_member");
    }
    validateMacroWaveStageExecution(stage.macroWave);
    if (stage.macroWave.memoryMode === "ram-backed") {
      ramBackedStages += 1;
      validateMacroWaveRamBackedMember(stage.members[0]);
    }
  }
  if (value.routeKind === "macro-wave" && ramBackedStages === 0) {
    throw new Error("runtime_macro_wave_route_requires_ram_backed_stage");
  }
  if (value.routeKind === "resident-baseline" && ramBackedStages !== 0) {
    throw new Error("runtime_macro_wave_resident_route_cannot_stream_weights");
  }
}

function validateMacroWaveStageExecution(value: Record<string, unknown>): void {
  const hasRamArtifact = Object.prototype.hasOwnProperty.call(value, "ramArtifact");
  assertExactObjectKeys(
    value,
    [
      "mode",
      "schema",
      "memoryMode",
      "residentKind",
      "budgets",
      "requirements",
      "workingSet",
      "cachePolicy",
      ...(hasRamArtifact ? ["ramArtifact"] : []),
    ],
    "runtime_macro_wave_stage_contract_is_invalid",
  );
  if (value.mode !== "macro-wave-memory") {
    throw new Error("runtime_macro_wave_stage_mode_is_invalid");
  }
  if (value.schema !== "gdlp-macro-wave-stage/1") {
    throw new Error("runtime_macro_wave_stage_schema_is_not_supported");
  }
  if (value.memoryMode !== "resident" && value.memoryMode !== "ram-backed") {
    throw new Error("runtime_macro_wave_memory_mode_is_invalid");
  }
  if (value.residentKind !== "layers" && value.residentKind !== "expert-shard") {
    throw new Error("runtime_macro_wave_resident_kind_is_invalid");
  }
  if (
    !isRecord(value.budgets) ||
    !isRecord(value.requirements) ||
    !isRecord(value.workingSet) ||
    !isRecord(value.cachePolicy)
  ) {
    throw new Error("runtime_macro_wave_stage_contract_is_invalid");
  }
  const budgets = value.budgets as unknown as MacroWaveStageExecutionContractV1["budgets"];
  const requirements = value.requirements as unknown as MacroWaveStageExecutionContractV1["requirements"];
  const workingSet = value.workingSet as unknown as MacroWaveStageExecutionContractV1["workingSet"];
  const cachePolicy = value.cachePolicy as unknown as MacroWaveStageExecutionContractV1["cachePolicy"];
  if (value.memoryMode === "ram-backed") {
    if (!isRecord(value.ramArtifact)) {
      throw new Error("runtime_macro_wave_ram_artifact_is_missing");
    }
    validateMacroWaveRamArtifact(value.ramArtifact);
    if (
      value.ramArtifact.largestExpertBytes !== workingSet.largestTransferUnitBytes ||
      value.ramArtifact.largestExpertBytes !== workingSet.largestExpertBytes ||
      value.ramArtifact.weightBufferCopies !== requirements.weightBufferCopies
    ) {
      throw new Error("runtime_macro_wave_ram_artifact_budget_mismatch");
    }
  } else if (hasRamArtifact) {
    throw new Error("runtime_macro_wave_resident_stage_cannot_bind_ram_artifact");
  }
  assertExactObjectKeys(
    budgets as unknown as Record<string, unknown>,
    ["hostRamBytes", "vramBytes"],
    "runtime_macro_wave_budgets_are_invalid",
  );
  assertPositiveInteger(budgets.hostRamBytes, "runtime_macro_wave_ram_budget_is_invalid");
  assertPositiveInteger(budgets.vramBytes, "runtime_macro_wave_vram_budget_is_invalid");
  const requirementFields = [
    "fullStageStateBytes",
    "hostRamBytes",
    "vramBytes",
    "fixedVramBytes",
    "residentParameterBudgetBytes",
    "residentStreamingTransientBytes",
    "boundedPinnedStagingReserveBytes",
    "hostRamPeakUpperBoundBytes",
    "activationBufferBytes",
    "weightBufferBytes",
    "weightBufferCopies",
  ];
  assertExactObjectKeys(
    requirements as unknown as Record<string, unknown>,
    requirementFields,
    "runtime_macro_wave_requirements_are_invalid",
  );
  for (const field of requirementFields) {
    assertNonNegativeInteger(
      requirements[field as keyof typeof requirements],
      "runtime_macro_wave_requirements_are_invalid",
    );
  }
  if (requirements.hostRamBytes > budgets.hostRamBytes) {
    throw new Error("runtime_macro_wave_ram_budget_exceeded");
  }
  if (requirements.vramBytes > budgets.vramBytes) {
    throw new Error("runtime_macro_wave_vram_budget_exceeded");
  }
  const workingSetFields = [
    "totalWeightBytes",
    "residentWeightBytes",
    "totalRoutedExpertBytes",
    "activeWeightBytesPerWave",
    "largestTransferUnitBytes",
    "largestExpertBytes",
  ];
  assertExactObjectKeys(
    workingSet as unknown as Record<string, unknown>,
    workingSetFields,
    "runtime_macro_wave_working_set_is_invalid",
  );
  for (const field of workingSetFields) {
    assertNonNegativeInteger(
      workingSet[field as keyof typeof workingSet],
      "runtime_macro_wave_working_set_is_invalid",
    );
  }
  if (
    workingSet.residentWeightBytes > workingSet.totalWeightBytes ||
    workingSet.totalRoutedExpertBytes > workingSet.totalWeightBytes ||
    workingSet.activeWeightBytesPerWave > workingSet.totalWeightBytes
  ) {
    throw new Error("runtime_macro_wave_working_set_exceeds_weights");
  }
  assertExactObjectKeys(
    cachePolicy as unknown as Record<string, unknown>,
    [
      "kind",
      "capacityBytes",
      "expectedHitRate",
      "expectedMissWeightBytesPerWave",
    ],
    "runtime_macro_wave_cache_policy_is_invalid",
  );
  if (
    cachePolicy.kind !== "full-resident" &&
    cachePolicy.kind !== "disabled" &&
    cachePolicy.kind !== "bounded-lru"
  ) {
    throw new Error("runtime_macro_wave_cache_policy_is_invalid");
  }
  assertNonNegativeInteger(
    cachePolicy.capacityBytes,
    "runtime_macro_wave_cache_capacity_is_invalid",
  );
  assertProbability(
    cachePolicy.expectedHitRate,
    "runtime_macro_wave_cache_hit_rate_is_invalid",
  );
  assertNonNegativeInteger(
    cachePolicy.expectedMissWeightBytesPerWave,
    "runtime_macro_wave_cache_miss_bytes_are_invalid",
  );
  const requiredVram =
    requirements.fixedVramBytes +
    requirements.activationBufferBytes +
    requirements.weightBufferBytes;
  if (value.memoryMode === "ram-backed" && requirements.vramBytes !== requiredVram) {
    throw new Error("runtime_macro_wave_vram_requirement_is_inconsistent");
  }
  if (value.memoryMode === "resident") {
    if (
      requirements.hostRamBytes !== 0 ||
      requirements.residentStreamingTransientBytes !== 0 ||
      requirements.boundedPinnedStagingReserveBytes !== 0 ||
      requirements.hostRamPeakUpperBoundBytes !== 0 ||
      requirements.weightBufferBytes !== 0 ||
      requirements.weightBufferCopies !== 0 ||
      workingSet.activeWeightBytesPerWave !== 0 ||
      cachePolicy.kind !== "full-resident" ||
      cachePolicy.capacityBytes !== workingSet.residentWeightBytes ||
      cachePolicy.expectedHitRate !== 1 ||
      cachePolicy.expectedMissWeightBytesPerWave !== 0
    ) {
      throw new Error("runtime_macro_wave_resident_contract_is_inconsistent");
    }
    if (
      value.residentKind === "layers" &&
      workingSet.residentWeightBytes !== workingSet.totalWeightBytes
    ) {
      throw new Error("runtime_macro_wave_layer_resident_weights_are_incomplete");
    }
    if (
      requirements.residentParameterBudgetBytes !== workingSet.residentWeightBytes ||
      requirements.vramBytes !==
      requirements.fixedVramBytes +
        requirements.activationBufferBytes +
        workingSet.residentWeightBytes
    ) {
      throw new Error("runtime_macro_wave_resident_vram_is_inconsistent");
    }
    return;
  }
  if (
    value.residentKind !== "layers" ||
    requirements.hostRamBytes !== requirements.hostRamPeakUpperBoundBytes ||
    requirements.hostRamPeakUpperBoundBytes !==
      workingSet.totalRoutedExpertBytes +
        requirements.boundedPinnedStagingReserveBytes +
        requirements.residentStreamingTransientBytes ||
    requirements.boundedPinnedStagingReserveBytes !==
      workingSet.largestExpertBytes * 2 ||
    requirements.weightBufferCopies !== 2 ||
    requirements.residentParameterBudgetBytes !== workingSet.residentWeightBytes ||
    workingSet.totalWeightBytes !==
      workingSet.residentWeightBytes + workingSet.totalRoutedExpertBytes ||
    workingSet.totalRoutedExpertBytes < 1 ||
    workingSet.activeWeightBytesPerWave < 1 ||
    workingSet.activeWeightBytesPerWave > workingSet.totalRoutedExpertBytes ||
    workingSet.largestExpertBytes < 1 ||
    workingSet.largestTransferUnitBytes !== workingSet.largestExpertBytes ||
    workingSet.largestExpertBytes > workingSet.activeWeightBytesPerWave ||
    requirements.weightBufferBytes !==
      workingSet.largestExpertBytes * requirements.weightBufferCopies ||
    requirements.fixedVramBytes !==
      requirements.fullStageStateBytes -
        workingSet.totalWeightBytes +
        requirements.residentParameterBudgetBytes
  ) {
    throw new Error("runtime_macro_wave_ram_backed_contract_is_inconsistent");
  }
  const expectedMiss = Math.ceil(
    workingSet.activeWeightBytesPerWave * (1 - cachePolicy.expectedHitRate),
  );
  if (cachePolicy.expectedMissWeightBytesPerWave !== expectedMiss) {
    throw new Error("runtime_macro_wave_cache_expectation_is_inconsistent");
  }
  if (cachePolicy.kind === "disabled") {
    if (
      cachePolicy.capacityBytes !== 0 ||
      cachePolicy.expectedHitRate !== 0
    ) {
      throw new Error("runtime_macro_wave_disabled_cache_is_inconsistent");
    }
  } else if (
    cachePolicy.kind !== "bounded-lru" ||
    cachePolicy.expectedHitRate <= 0 ||
    cachePolicy.capacityBytes < workingSet.largestTransferUnitBytes ||
    requirements.vramBytes + cachePolicy.capacityBytes > budgets.vramBytes
  ) {
    throw new Error("runtime_macro_wave_bounded_cache_is_inconsistent");
  }
}

function validateMacroWaveRamArtifact(value: Record<string, unknown>): void {
  assertExactObjectKeys(
    value,
    [
      "schema",
      "format",
      "locality",
      "loader",
      "weightEncoding",
      "sourceDtypes",
      "adapterIds",
      "expertExecutionMode",
      "largestExpertBytes",
      "weightBufferCopies",
      "fullModelMaterialization",
    ],
    "runtime_macro_wave_ram_artifact_is_invalid",
  );
  if (
    value.schema !== "gdlp-local-safetensors-moe-stage/1" ||
    value.format !== "safetensors" ||
    value.locality !== "host-local-only" ||
    value.loader !== "selective-safetensors-ram-backed-moe" ||
    value.weightEncoding !== "floating-safetensors" ||
    !Array.isArray(value.sourceDtypes) ||
    value.sourceDtypes.length !== 3 ||
    value.sourceDtypes[0] !== "fp16" ||
    value.sourceDtypes[1] !== "bf16" ||
    value.sourceDtypes[2] !== "fp32" ||
    value.expertExecutionMode !== "serial-exact" ||
    value.weightBufferCopies !== 2 ||
    !Number.isSafeInteger(value.largestExpertBytes) ||
    (value.largestExpertBytes as number) <= 0 ||
    value.fullModelMaterialization !== false ||
    !Array.isArray(value.adapterIds) ||
    value.adapterIds.length !== 2 ||
    value.adapterIds[0] !== "transformers-qwen3-moe-v1" ||
    value.adapterIds[1] !== "transformers-glm4-moe-v1"
  ) {
    throw new Error("runtime_macro_wave_ram_artifact_is_invalid");
  }
}

function validateMacroWaveRamBackedMember(value: unknown): void {
  if (!isRecord(value) || !isRecord(value.backend) || !isRecord(value.capabilities)) {
    throw new Error("runtime_macro_wave_ram_backend_is_not_declared");
  }
  const backend = value.backend as unknown as RuntimeBackendProfile;
  const capabilities = value.capabilities as unknown as RuntimeMemberCapabilities;
  if (
    !backend.modelFormats.includes("safetensors") ||
    !backend.executionModes.includes("ram-backed-moe-stage") ||
    !capabilities.deviceKinds.includes("gpu") ||
    !capabilities.computeApis.includes("cuda") ||
    !capabilities.weightDtypes.some((dtype) =>
      ["fp16", "bf16", "fp32"].includes(dtype)
    ) ||
    !capabilities.features.includes("ram-authoritative-routed-experts") ||
    !capabilities.features.includes("device-expert-cache") ||
    !capabilities.features.includes("authoritative-router") ||
    !capabilities.features.includes("predictive-prefetch-only") ||
    !capabilities.features.includes("meta-model-construction") ||
    !capabilities.features.includes("no-full-model-materialization")
  ) {
    throw new Error("runtime_macro_wave_ram_backend_is_not_declared");
  }
}

function validateVirtualStageExecution(
  value: unknown,
  stage: RuntimeVirtualStageManifest,
): void {
  if (value === undefined) return;
  if (!isRecord(value)) {
    throw new Error("runtime_stage_execution_is_invalid");
  }
  if (value.mode !== "tensor-parallel-cell") {
    throw new Error("runtime_stage_execution_is_invalid");
  }
  if (value.engine !== "python-torch") {
    throw new Error("runtime_cell_engine_is_not_executable");
  }
  if (value.collectiveBackend !== "gloo" && value.collectiveBackend !== "nccl") {
    throw new Error("runtime_cell_collective_backend_is_not_executable");
  }
  if (
    value.computeDtype !== "float32" &&
    value.computeDtype !== "float16" &&
    value.computeDtype !== "bfloat16"
  ) {
    throw new Error("runtime_cell_compute_dtype_is_invalid");
  }
  if (!isRecord(value.fixture)) throw new Error("runtime_cell_fixture_is_missing");
  const fixture = value.fixture;
  if (
    fixture.schema !== "gdlp-llama-cell-layer/1" &&
    fixture.schema !== "gdlp-llama-cell-stage/2"
  ) {
    throw new Error("runtime_cell_fixture_schema_is_invalid");
  }
  if (fixture.location !== "anchor-local" && fixture.location !== "member-local") {
    throw new Error("runtime_cell_fixture_location_is_not_executable");
  }
  const fixturePath = asNonEmptyString(
    fixture.path,
    "runtime_cell_fixture_path_is_invalid",
  );
  if (
    fixturePath.includes("\u0000") ||
    fixturePath.includes("\r") ||
    fixturePath.includes("\n")
  ) {
    throw new Error("runtime_cell_fixture_path_is_invalid");
  }
  const layerCount = asPositiveInteger(
    fixture.layerCount,
    "runtime_cell_fixture_layer_count_is_invalid",
  );
  if (layerCount !== stage.layerEnd - stage.layerStart) {
    throw new Error("runtime_cell_fixture_layer_count_mismatch");
  }
  if (fixture.schema === "gdlp-llama-cell-layer/1" && layerCount !== 1) {
    throw new Error("runtime_cell_v1_fixture_requires_one_layer");
  }
  if (fixture.schema === "gdlp-llama-cell-stage/2" && layerCount < 2) {
    throw new Error("runtime_cell_v2_fixture_requires_multiple_layers");
  }
  const worldSize = asPositiveInteger(
    value.worldSize,
    "runtime_cell_world_size_is_invalid",
  );
  if (worldSize < 2 || worldSize !== stage.members.length) {
    throw new Error("runtime_cell_world_size_mismatch");
  }
  validateSha256(
    fixture.manifestSha256,
    "runtime_cell_fixture_manifest_sha256_is_invalid",
  );
  if (!Array.isArray(fixture.shardSha256) || fixture.shardSha256.length !== worldSize) {
    throw new Error("runtime_cell_fixture_shard_sha256_is_invalid");
  }
  for (const digest of fixture.shardSha256) {
    validateSha256(digest, "runtime_cell_fixture_shard_sha256_is_invalid");
  }
  if (!Array.isArray(fixture.rankMemory) || fixture.rankMemory.length !== worldSize) {
    throw new Error("runtime_cell_rank_memory_is_invalid");
  }
  for (const rank of fixture.rankMemory) {
    if (!isRecord(rank)) throw new Error("runtime_cell_rank_memory_is_invalid");
    const fixedBytes = asNonNegativeInteger(
      rank.fixedBytes,
      "runtime_cell_rank_memory_is_invalid",
    );
    asNonNegativeInteger(
      rank.kvBytesPerToken,
      "runtime_cell_rank_memory_is_invalid",
    );
    const requiredBytes = asNonNegativeInteger(
      rank.requiredBytes,
      "runtime_cell_rank_memory_is_invalid",
    );
    if (requiredBytes < fixedBytes) {
      throw new Error("runtime_cell_rank_memory_is_invalid");
    }
  }
  validateStringList(value.rankMemberIds, "runtime_cell_rank_members_are_invalid");
  const rankMemberIds = value.rankMemberIds;
  if (
    rankMemberIds.length !== worldSize ||
    rankMemberIds[0] !== stage.anchor.memberId ||
    !stringListEquals(
      rankMemberIds,
      stage.members.map((member) => member.nodeId),
    )
  ) {
    throw new Error("runtime_cell_rank_members_mismatch");
  }
  if (
    !Array.isArray(value.rankWeights) ||
    value.rankWeights.length !== worldSize ||
    value.rankWeights.some(
      (weight) =>
        typeof weight !== "number" ||
        !Number.isFinite(weight) ||
        weight <= 0,
    )
  ) {
    throw new Error("runtime_cell_rank_weights_are_invalid");
  }
  if (
    !Array.isArray(value.rankDevices) ||
    value.rankDevices.length !== worldSize ||
    value.rankDevices.some((device) => typeof device !== "string" || !device.trim())
  ) {
    throw new Error("runtime_cell_rank_devices_are_invalid");
  }
  const rankDevices = value.rankDevices as string[];
  if (value.collectiveBackend === "gloo") {
    if (
      value.computeDtype !== "float32" ||
      rankDevices.some((device) => device !== "cpu")
    ) {
      throw new Error("runtime_gloo_cell_requires_cpu_float32");
    }
  } else if (
    rankDevices.some((device) => !/^cuda:(?:0|[1-9]\d*)$/.test(device))
  ) {
    throw new Error("runtime_nccl_cell_requires_cuda_devices");
  }
  if (
    typeof value.operationTimeoutSeconds !== "number" ||
    !Number.isFinite(value.operationTimeoutSeconds) ||
    value.operationTimeoutSeconds <= 0
  ) {
    throw new Error("runtime_cell_operation_timeout_is_invalid");
  }
  if (stage.first || stage.last) {
    throw new Error("runtime_cell_stage_must_be_intermediate");
  }

  const external = value.external;
  if (fixture.location === "anchor-local") {
    if (external !== undefined) {
      throw new Error("runtime_local_cell_cannot_declare_external_endpoints");
    }
  } else {
    if (fixture.schema !== "gdlp-llama-cell-stage/2") {
      throw new Error("runtime_external_cell_requires_v2_fixture");
    }
    if (!isRecord(external)) {
      throw new Error("runtime_external_cell_endpoints_are_missing");
    }
    if (
      !Array.isArray(external.rankFixturePaths) ||
      external.rankFixturePaths.length !== worldSize
    ) {
      throw new Error("runtime_external_cell_fixture_paths_are_invalid");
    }
    for (const path of external.rankFixturePaths) {
      validateSafePath(path, "runtime_external_cell_fixture_paths_are_invalid");
    }
    if (external.rankFixturePaths[0] !== fixturePath) {
      throw new Error("runtime_external_cell_anchor_fixture_mismatch");
    }
    for (const [host, error] of [
      [external.controlBindHost, "runtime_external_cell_control_bind_host_is_invalid"],
      [
        external.controlAdvertiseHost,
        "runtime_external_cell_control_advertise_host_is_invalid",
      ],
      [
        external.distributedAdvertiseHost,
        "runtime_external_cell_distributed_host_is_invalid",
      ],
    ] as const) {
      validateSafeHost(host, error);
    }
    const controlPort = asTcpPort(
      external.controlPort,
      "runtime_external_cell_control_port_is_invalid",
    );
    const distributedPort = asTcpPort(
      external.distributedPort,
      "runtime_external_cell_distributed_port_is_invalid",
    );
    if (
      controlPort === distributedPort ||
      controlPort === stage.anchor.endpoint.port ||
      distributedPort === stage.anchor.endpoint.port
    ) {
      throw new Error("runtime_external_cell_ports_conflict");
    }
    if (
      typeof external.startupTimeoutSeconds !== "number" ||
      !Number.isFinite(external.startupTimeoutSeconds) ||
      external.startupTimeoutSeconds <= 0
    ) {
      throw new Error("runtime_external_cell_startup_timeout_is_invalid");
    }
  }

  const anchorHost = stage.anchor.endpoint.host.trim().toLowerCase();
  const referenceBackend = stage.members[0]!.backend;
  const normalizedComputeDtype = runtimeWeightDtype(value.computeDtype);
  const memberEndpoints = new Set<string>();
  for (const member of stage.members) {
    if (
      fixture.location === "anchor-local" &&
      member.endpoint.host.trim().toLowerCase() !== anchorHost
    ) {
      throw new Error("runtime_cell_members_must_be_anchor_local");
    }
    const memberEndpoint = `${member.endpoint.host.trim().toLowerCase()}\u0000${member.endpoint.port}`;
    if (memberEndpoints.has(memberEndpoint)) {
      throw new Error("runtime_cell_member_endpoint_reused");
    }
    memberEndpoints.add(memberEndpoint);
    if (!backendExecutionEquals(member.backend, referenceBackend)) {
      throw new Error("runtime_cell_member_backends_are_inconsistent");
    }
    if (
      member.backend.engine !== value.engine ||
      !member.backend.modelFormats.includes("safetensors") ||
      !member.backend.executionModes.includes("tensor-parallel-cell")
    ) {
      throw new Error("runtime_cell_member_backend_is_not_executable");
    }
    const supportsCollective = member.capabilities.computeApis.includes(
      value.collectiveBackend,
    );
    const supportsDeviceApi =
      value.collectiveBackend === "gloo"
        ? member.capabilities.deviceKinds.includes("cpu")
        : member.capabilities.computeApis.includes("cuda") ||
          member.capabilities.computeApis.includes("rocm");
    if (
      !supportsCollective ||
      !supportsDeviceApi ||
      !member.capabilities.weightDtypes.includes(normalizedComputeDtype) ||
      !member.capabilities.features.includes("rank-local-kv")
    ) {
      throw new Error("runtime_cell_member_capabilities_are_not_executable");
    }
  }
}

function runtimeWeightDtype(
  dtype: RuntimeTensorParallelComputeDtype,
): "fp32" | "fp16" | "bf16" {
  switch (dtype) {
    case "float32":
      return "fp32";
    case "float16":
      return "fp16";
    case "bfloat16":
      return "bf16";
  }
}

function backendExecutionEquals(
  left: RuntimeBackendProfile,
  right: RuntimeBackendProfile,
): boolean {
  return (
    left.engine === right.engine &&
    left.version === right.version &&
    stringListEquals(left.modelFormats, right.modelFormats) &&
    stringListEquals(left.executionModes, right.executionModes)
  );
}

function validateBackend(value: unknown): void {
  if (!isRecord(value)) throw new Error("runtime_backend_is_missing");
  asNonEmptyString(value.engine, "runtime_backend_engine_cannot_be_empty");
  if (value.version !== undefined) {
    asNonEmptyString(value.version, "runtime_backend_version_cannot_be_empty");
  }
  validateStringList(value.modelFormats, "runtime_backend_model_formats_are_invalid");
  validateStringList(value.executionModes, "runtime_backend_execution_modes_are_invalid");
}

function validateMemberCapabilities(
  value: unknown,
  activationCodec: unknown,
): void {
  if (!isRecord(value)) throw new Error("runtime_member_capabilities_are_missing");
  validateStringList(value.deviceKinds, "runtime_device_kinds_are_invalid");
  validateStringList(value.computeApis, "runtime_compute_apis_are_invalid");
  validateStringList(value.weightDtypes, "runtime_weight_dtypes_are_invalid");
  validateStringList(value.activationCodecs, "runtime_activation_codecs_are_invalid");
  validateStringList(value.features, "runtime_member_features_are_invalid");
  if (!(value.activationCodecs as string[]).includes(activationCodec as string)) {
    throw new Error("runtime_member_does_not_support_activation_codec");
  }
}

function validatePredictedMetrics(
  value: Record<string, unknown>,
  stageCount: number,
  phase: "prefill" | "decode",
): void {
  assertPositiveInteger(value.stages, `runtime_${phase}_predicted_stages_are_invalid`);
  if (value.stages !== stageCount) {
    throw new Error(`runtime_${phase}_predicted_stage_count_mismatch`);
  }
  for (const field of [
    "ttftMs",
    "tpotMs",
    "tokensPerSecondPerSequence",
    "aggregateTokensPerSecond",
    "responseTimeMs",
    "pipelineCycleMs",
    "pathDecodeMs",
    "tokenReturnMs",
    "energyWhPerOutputToken",
    "peakStageMemoryBytes",
    "networkBytesPerOutputToken",
    "qualityLoss",
  ]) {
    assertNonNegativeFinite(
      value[field],
      `runtime_${phase}_predicted_metric_is_invalid:${field}`,
    );
  }
  assertProbability(
    value.routeAvailability,
    `runtime_${phase}_predicted_availability_is_invalid`,
  );
  if (!Array.isArray(value.stageMetrics) || value.stageMetrics.length !== stageCount) {
    throw new Error(`runtime_${phase}_predicted_stage_metrics_are_invalid`);
  }
  for (const metric of value.stageMetrics) {
    if (!isRecord(metric)) {
      throw new Error(`runtime_${phase}_predicted_stage_metrics_are_invalid`);
    }
    asNonEmptyString(metric.nodeId, `runtime_${phase}_predicted_node_is_invalid`);
    asNonNegativeInteger(
      metric.layerStart,
      `runtime_${phase}_predicted_layer_range_is_invalid`,
    );
    asPositiveInteger(
      metric.layerEnd,
      `runtime_${phase}_predicted_layer_range_is_invalid`,
    );
    for (const field of ["memoryBytes", "memoryLimitBytes"]) {
      asNonNegativeInteger(
        metric[field],
        `runtime_${phase}_predicted_stage_memory_is_invalid`,
      );
    }
    for (const field of [
      "decodeBatchComputeMs",
      "decodeOutgoingMs",
      "prefillChunkComputeMs",
      "prefillOutgoingMs",
    ]) {
      assertNonNegativeFinite(
        metric[field],
        `runtime_${phase}_predicted_stage_metric_is_invalid:${field}`,
      );
    }
  }
  if (value.calibrationRequired !== undefined && typeof value.calibrationRequired !== "boolean") {
    throw new Error(`runtime_${phase}_calibration_flag_is_invalid`);
  }
  if (value.calibrationReasons !== undefined) {
    validateStringList(
      value.calibrationReasons,
      `runtime_${phase}_calibration_reasons_are_invalid`,
    );
  }
}

function validateSpeculationPolicy(value: unknown): void {
  if (!isRecord(value)) throw new Error("runtime_speculation_policy_is_missing");
  if (value.mode !== "disabled" && value.mode !== "adaptive") {
    throw new Error("runtime_speculation_mode_is_invalid");
  }
  if (value.controller !== "fixed" && value.controller !== "acceptance-adaptive") {
    throw new Error("runtime_speculation_controller_is_invalid");
  }
  assertPositiveInteger(
    value.acceptanceWindowTokens,
    "runtime_speculation_window_is_invalid",
  );
  if (!Array.isArray(value.strategies) || value.strategies.length < 1) {
    throw new Error("runtime_speculation_strategies_are_missing");
  }
  const ids = new Set<string>();
  const strategies = new Map<string, Record<string, unknown>>();
  for (const strategy of value.strategies) {
    if (!isRecord(strategy)) throw new Error("runtime_speculation_strategy_is_invalid");
    const id = asNonEmptyString(strategy.id, "runtime_speculation_strategy_id_is_invalid");
    if (ids.has(id)) throw new Error("runtime_speculation_strategy_id_reused");
    ids.add(id);
    strategies.set(id, strategy);
    if (!isSpeculationKind(strategy.kind)) {
      throw new Error("runtime_speculation_strategy_kind_is_invalid");
    }
    assertPositiveInteger(
      strategy.maxDraftTokens,
      "runtime_speculation_draft_length_is_invalid",
    );
    assertProbability(
      strategy.minAcceptanceRate,
      "runtime_speculation_acceptance_is_invalid",
    );
    assertProbability(strategy.maxWasteRatio, "runtime_speculation_waste_is_invalid");
    assertNonNegativeInteger(
      strategy.priority,
      "runtime_speculation_priority_is_invalid",
    );
    const artifactBacked =
      strategy.kind === "draft-model"
      || strategy.kind === "mtp"
      || strategy.kind === "intermediate-head";
    if (artifactBacked && strategy.artifactId === undefined) {
      throw new Error("runtime_speculation_artifact_is_missing");
    }
    if (strategy.artifactId !== undefined) {
      validateSha256Identity(
        strategy.artifactId,
        "runtime_speculation_artifact_is_invalid",
      );
    }
    const draftMemory = [
      strategy.parameterBytes,
      strategy.memoryReservationBytes,
    ];
    if (strategy.kind === "draft-model") {
      if (draftMemory.some((value) => value === undefined)) {
        throw new Error("runtime_draft_model_memory_contract_is_missing");
      }
      assertPositiveInteger(
        strategy.parameterBytes,
        "runtime_draft_model_parameter_bytes_are_invalid",
      );
      assertPositiveInteger(
        strategy.memoryReservationBytes,
        "runtime_draft_model_memory_reservation_is_invalid",
      );
      if (
        (strategy.memoryReservationBytes as number)
        < (strategy.parameterBytes as number)
      ) {
        throw new Error("runtime_draft_model_memory_reservation_is_too_small");
      }
    } else if (draftMemory.some((value) => value !== undefined)) {
      throw new Error("runtime_draft_model_memory_requires_draft_model_strategy");
    }
    if (strategy.kind === "autoregressive" && strategy.maxDraftTokens !== 1) {
      throw new Error("runtime_autoregressive_draft_length_must_be_one");
    }
    const treeLimits = [
      strategy.maxBranches,
      strategy.maxBranchTokens,
      strategy.maxKvBytes,
      strategy.maxWaveTokens,
    ];
    if (strategy.kind === "draft-tree") {
      if (treeLimits.some((limit) => limit === undefined)) {
        throw new Error("runtime_draft_tree_limits_are_missing");
      }
      assertBoundedPositiveInteger(
        strategy.maxBranches,
        64,
        "runtime_draft_tree_branches_are_invalid",
      );
      assertBoundedPositiveInteger(
        strategy.maxBranchTokens,
        1_048_576,
        "runtime_draft_tree_branch_tokens_are_invalid",
      );
      assertBoundedPositiveInteger(
        strategy.maxKvBytes,
        2 ** 40,
        "runtime_draft_tree_kv_bytes_are_invalid",
      );
      assertBoundedPositiveInteger(
        strategy.maxWaveTokens,
        17,
        "runtime_draft_tree_wave_tokens_are_invalid",
      );
      if ((strategy.maxWaveTokens as number) !== strategy.maxDraftTokens + 1) {
        throw new Error("runtime_draft_tree_wave_does_not_match_draft_depth");
      }
    } else if (treeLimits.some((limit) => limit !== undefined)) {
      throw new Error("runtime_tree_limits_require_draft_tree_strategy");
    }
  }
  const defaultId = asNonEmptyString(
    value.defaultStrategyId,
    "runtime_speculation_default_is_invalid",
  );
  const fallbackId = asNonEmptyString(
    value.fallbackStrategyId,
    "runtime_speculation_fallback_is_invalid",
  );
  if (!strategies.has(defaultId)) throw new Error("runtime_speculation_default_is_unknown");
  const fallback = strategies.get(fallbackId);
  if (!fallback) throw new Error("runtime_speculation_fallback_is_unknown");
  if (fallback.kind !== "autoregressive") {
    throw new Error("runtime_speculation_fallback_must_be_autoregressive");
  }
  if (
    value.mode === "disabled" &&
    (value.controller !== "fixed" ||
      defaultId !== fallbackId ||
      strategies.get(defaultId)?.kind !== "autoregressive")
  ) {
    throw new Error("runtime_disabled_speculation_policy_is_inconsistent");
  }
  if (value.mode === "adaptive" && value.controller !== "acceptance-adaptive") {
    throw new Error("runtime_adaptive_speculation_requires_adaptive_controller");
  }
}

function validateMacroWaveSpeculationBinding(
  prefill: unknown,
  decode: unknown,
  speculation: RuntimeSpeculationPolicy,
): void {
  if (prefill === undefined && decode === undefined) return;
  if (!isRecord(prefill) || !isRecord(decode)) {
    throw new Error("runtime_macro_wave_requires_both_phase_contracts");
  }
  const prefillCore = {
    schema: prefill.schema,
    routeKind: prefill.routeKind,
    waveTokens: prefill.waveTokens,
    expectedCommittedTokensPerWave: prefill.expectedCommittedTokensPerWave,
  };
  const decodeCore = {
    schema: decode.schema,
    routeKind: decode.routeKind,
    waveTokens: decode.waveTokens,
    expectedCommittedTokensPerWave: decode.expectedCommittedTokensPerWave,
  };
  if (JSON.stringify(prefillCore) !== JSON.stringify(decodeCore)) {
    throw new Error("runtime_macro_wave_phase_contracts_do_not_match");
  }

  const waveTokens = decode.waveTokens as number;
  const expectedCommitted = decode.expectedCommittedTokensPerWave as number;
  const selected = speculation.strategies.find(
    (strategy) => strategy.id === speculation.defaultStrategyId,
  );
  if (!selected) {
    // validateSpeculationPolicy normally catches this first; retain a closed
    // boundary here if the helper is reused independently later.
    throw new Error("runtime_speculation_default_is_unknown");
  }
  if (speculation.mode === "disabled" || selected.kind === "autoregressive") {
    if (waveTokens !== 1 || expectedCommitted !== 1) {
      throw new Error("runtime_macro_wave_autoregressive_contract_mismatch");
    }
    return;
  }
  if (
    selected.kind === "draft-tree" &&
    selected.maxWaveTokens !== waveTokens
  ) {
    throw new Error("runtime_draft_tree_wave_contract_mismatch");
  }

  for (const strategy of speculation.strategies) {
    if (
      strategy.kind !== "autoregressive" &&
      strategy.maxDraftTokens + 1 > waveTokens
    ) {
      throw new Error("runtime_macro_wave_speculation_exceeds_wave");
    }
  }
  const maximumCommittedPositions = selected.maxDraftTokens + 1;
  if (expectedCommitted > maximumCommittedPositions) {
    throw new Error(
      "runtime_macro_wave_committed_projection_exceeds_strategy",
    );
  }
}

function validateKvTransition(value: unknown): void {
  if (!isRecord(value)) throw new Error("runtime_kv_transition_is_missing");
  if (value.mode !== "in-place" && value.mode !== "transfer" && value.mode !== "recompute") {
    throw new Error("runtime_kv_transition_mode_is_invalid");
  }
  if (!isRecord(value.format)) throw new Error("runtime_kv_format_is_missing");
  asNonEmptyString(value.format.id, "runtime_kv_format_id_is_invalid");
  assertPositiveInteger(value.format.version, "runtime_kv_format_version_is_invalid");
  asNonEmptyString(value.format.dtype, "runtime_kv_dtype_is_invalid");
  asNonEmptyString(value.format.layout, "runtime_kv_layout_is_invalid");
  if (!isRecord(value.gate)) throw new Error("runtime_kv_gate_is_missing");
  if (value.gate.policy !== "strict") throw new Error("runtime_kv_gate_policy_is_invalid");
  asNonEmptyString(value.gate.modelRevision, "runtime_kv_gate_revision_is_invalid");
  asNonEmptyString(value.gate.tokenizerId, "runtime_kv_gate_tokenizer_is_invalid");
  asNonEmptyString(value.gate.layerLayoutHash, "runtime_kv_gate_layout_is_invalid");
  if (typeof value.gate.requireContextIdentity !== "boolean") {
    throw new Error("runtime_kv_gate_context_policy_is_invalid");
  }
  if (value.gate.onMismatch !== "reject" && value.gate.onMismatch !== "recompute-prefill") {
    throw new Error("runtime_kv_gate_fallback_is_invalid");
  }
}

function validatePlanRequest(request: RuntimePlanRequest): void {
  if (typeof request.modelRevision !== "string" || !request.modelRevision.trim()) {
    throw new Error("model_revision_cannot_be_empty");
  }
  if (request.planner !== undefined) {
    if (
      !isRecord(request.planner) ||
      (request.planner.kind !== "default" && request.planner.kind !== "macro-wave")
    ) {
      throw new Error("runtime_planner_selection_is_invalid");
    }
    if (
      request.planner.kind === "macro-wave" &&
      request.planner.options !== undefined &&
      !isRecord(request.planner.options)
    ) {
      throw new Error("runtime_macro_wave_options_are_invalid");
    }
  }
  if (
    request.tensorParallelCells !== undefined &&
    !Array.isArray(request.tensorParallelCells)
  ) {
    throw new Error("runtime_cell_plans_are_invalid");
  }
  if (request.phaseTensorParallelCells !== undefined) {
    if (!isRecord(request.phaseTensorParallelCells)) {
      throw new Error("runtime_phase_cell_plans_are_invalid");
    }
    for (const phase of ["prefill", "decode"] as const) {
      const cells = request.phaseTensorParallelCells[phase];
      if (cells !== undefined && !Array.isArray(cells)) {
        throw new Error(`runtime_${phase}_cell_plans_are_invalid`);
      }
    }
  }
  const cellPlanGroups = [
    request.tensorParallelCells ?? [],
    request.phaseTensorParallelCells?.prefill ?? [],
    request.phaseTensorParallelCells?.decode ?? [],
  ];
  if (
    request.planner?.kind === "macro-wave" &&
    cellPlanGroups.some((cells) => cells.length > 0)
  ) {
    throw new Error("runtime_macro_wave_and_tensor_parallel_cells_are_not_composable");
  }
  for (const cells of cellPlanGroups) {
    for (const cell of cells) {
      if (!isRecord(cell)) throw new Error("runtime_cell_plan_is_invalid");
      asInteger(cell.stageIndex, "runtime_cell_plan_stage_index_is_invalid");
      validateStringList(cell.memberNodeIds, "runtime_cell_plan_members_are_invalid");
      if (!isRecord(cell.execution)) {
        throw new Error("runtime_cell_plan_execution_is_invalid");
      }
    }
  }
  if (request.topology.nodes.length < 1) throw new Error("runtime_topology_has_no_nodes");
  if (request.model.layers.length < 1) throw new Error("runtime_model_has_no_layers");
  asNonEmptyString(request.model.id, "runtime_model_id_cannot_be_empty");
  for (const [name, value] of [
    ["embedding", request.model.embeddingBytes],
    ["lm_head", request.model.lmHeadBytes],
    ["runtime_overhead", request.model.runtimeOverheadBytesPerStage],
  ] as const) {
    assertNonNegativeInteger(value, `runtime_model_${name}_bytes_are_invalid`);
  }
  for (const [name, value] of [
    ["embedding_decode", request.model.embeddingDecodeMsAtUnit],
    ["lm_head_decode", request.model.lmHeadDecodeMsAtUnit],
    ["embedding_prefill", request.model.embeddingPrefillMsPerTokenAtUnit],
    ["lm_head_prefill", request.model.lmHeadPrefillMsPerTokenAtUnit],
  ] as const) {
    assertNonNegativeFinite(value, `runtime_model_${name}_time_is_invalid`);
  }
  if (
    request.model.tiedEmbeddingAndHead !== undefined &&
    typeof request.model.tiedEmbeddingAndHead !== "boolean"
  ) {
    throw new Error("runtime_model_tied_weights_flag_is_invalid");
  }
  for (const [name, value] of [
    ["largest_embedding_tensor", request.model.largestEmbeddingTensorBytes],
    ["largest_lm_head_tensor", request.model.largestLmHeadTensorBytes],
  ] as const) {
    if (value !== undefined) {
      assertNonNegativeInteger(
        value,
        `runtime_model_${name}_bytes_are_invalid`,
      );
    }
  }
  for (const [index, layer] of request.model.layers.entries()) {
    if (layer.index !== index) throw new Error("runtime_model_layer_index_is_invalid");
    assertNonNegativeInteger(layer.weightBytes, "runtime_model_layer_weight_is_invalid");
    assertPositiveInteger(
      layer.activationElements,
      "runtime_model_layer_activation_size_is_invalid",
    );
    assertNonNegativeInteger(
      layer.kvBytesPerToken,
      "runtime_model_layer_kv_size_is_invalid",
    );
    assertNonNegativeFinite(
      layer.decodeMsAtUnit,
      "runtime_model_layer_decode_time_is_invalid",
    );
    assertNonNegativeFinite(
      layer.prefillMsPerTokenAtUnit,
      "runtime_model_layer_prefill_time_is_invalid",
    );
    if (layer.largestResidentTensorBytes !== undefined) {
      assertNonNegativeInteger(
        layer.largestResidentTensorBytes,
        "runtime_model_layer_largest_resident_tensor_is_invalid",
      );
    }
  }
  const hiddenSizes = new Set(request.model.layers.map((layer) => layer.activationElements));
  if (hiddenSizes.size !== 1) {
    throw new Error("runtime_requires_a_fixed_activation_hidden_size");
  }
  const ids = new Set<string>();
  for (const node of request.topology.nodes) {
    asNonEmptyString(node.id, "runtime_node_id_cannot_be_empty");
    asNonEmptyString(node.region, "runtime_node_region_cannot_be_empty");
    if (ids.has(node.id)) throw new Error(`duplicate_runtime_node:${node.id}`);
    ids.add(node.id);
    assertNonNegativeInteger(node.memoryBytes, "runtime_node_memory_is_invalid");
    assertNonNegativeInteger(node.reserveBytes, "runtime_node_reserve_is_invalid");
    if (node.reserveBytes > node.memoryBytes) {
      throw new Error("runtime_node_reserve_exceeds_memory");
    }
    assertPositiveFinite(node.decodeScale, "runtime_node_decode_scale_is_invalid");
    assertPositiveFinite(node.prefillScale, "runtime_node_prefill_scale_is_invalid");
    assertPositiveFinite(node.codecScale, "runtime_node_codec_scale_is_invalid");
    assertNonNegativeFinite(node.batchGain, "runtime_node_batch_gain_is_invalid");
    if (!Number.isFinite(node.maxBatchSpeedup) || node.maxBatchSpeedup < 1) {
      throw new Error("runtime_node_batch_speedup_is_invalid");
    }
    assertNonNegativeFinite(node.powerWatts, "runtime_node_power_is_invalid");
    assertProbability(node.availability, "runtime_node_availability_is_invalid");
    validateEndpoint(node.endpoint);
  }
  const linkIds = new Set<string>();
  for (const link of request.topology.links) {
    if (!ids.has(link.from) || !ids.has(link.to) || link.from === link.to) {
      throw new Error("runtime_link_endpoint_is_invalid");
    }
    const identity = `${link.from}\u0000${link.to}`;
    if (linkIds.has(identity)) throw new Error("runtime_link_is_duplicated");
    linkIds.add(identity);
    assertNonNegativeFinite(link.oneWayLatencyMs, "runtime_link_latency_is_invalid");
    assertNonNegativeFinite(link.jitterP95Ms, "runtime_link_jitter_is_invalid");
    assertPositiveFinite(link.bandwidthMbps, "runtime_link_bandwidth_is_invalid");
    assertProbability(link.lossRate, "runtime_link_loss_is_invalid");
    if (link.availability !== undefined) {
      assertProbability(link.availability, "runtime_link_availability_is_invalid");
    }
    if (link.evidence !== undefined) {
      const bindingValues = [
        link.evidence.fromEngineProfileId,
        link.evidence.toEngineProfileId,
        link.evidence.fromHardwareFingerprintSha256,
        link.evidence.toHardwareFingerprintSha256,
      ];
      const hasAnyBinding = bindingValues.some((value) => value !== undefined);
      const hasCompleteBinding = bindingValues.every(
        (value) => typeof value === "string" && /^sha256:[0-9a-f]{64}$/.test(value),
      );
      if (
        link.evidence.source !== "runtime-probe"
        || !Number.isInteger(link.evidence.measuredAt)
        || link.evidence.measuredAt <= 0
        || !Number.isInteger(link.evidence.validUntil)
        || link.evidence.validUntil <= link.evidence.measuredAt
        || !Number.isInteger(link.evidence.successfulSamples)
        || link.evidence.successfulSamples < 0
        || !Number.isInteger(link.evidence.failedSamples)
        || link.evidence.failedSamples < 0
        || (hasAnyBinding && !hasCompleteBinding)
      ) {
        throw new Error("runtime_link_evidence_is_invalid");
      }
    }
  }
  for (const [name, value] of [
    ["prompt_tokens", request.workload.promptTokens],
    ["output_tokens", request.workload.outputTokens],
    ["context_tokens", request.workload.contextTokens],
    ["concurrent_sequences", request.workload.concurrentSequences],
    ["max_stages", request.workload.maxStages],
  ] as const) {
    assertPositiveInteger(value, `runtime_workload_${name}_is_invalid`);
  }
  assertProbability(
    request.workload.maxQualityLoss,
    "runtime_workload_quality_budget_is_invalid",
  );
  assertProbability(
    request.workload.minRouteAvailability,
    "runtime_workload_availability_is_invalid",
  );
  assertNonNegativeFinite(
    request.workload.batchWindowMs,
    "runtime_workload_batch_window_is_invalid",
  );
  if (typeof request.workload.p95 !== "boolean") {
    throw new Error("runtime_workload_p95_flag_is_invalid");
  }
}

function validateEndpoint(value: unknown): asserts value is RuntimeEndpoint {
  if (!isRecord(value)) throw new Error("runtime_endpoint_is_missing");
  asNonEmptyString(value.host, "runtime_endpoint_host_cannot_be_empty");
  if (!Number.isInteger(value.port) || (value.port as number) < 1 || (value.port as number) > 65_535) {
    throw new Error("runtime_endpoint_port_is_invalid");
  }
}

function validateSafePath(value: unknown, error: string): asserts value is string {
  if (
    typeof value !== "string" ||
    !value.trim() ||
    value.includes("\u0000") ||
    value.includes("\r") ||
    value.includes("\n")
  ) {
    throw new Error(error);
  }
}

function validateSafeHost(value: unknown, error: string): asserts value is string {
  validateSafePath(value, error);
}

function validateSha256(value: unknown, error: string): asserts value is string {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/i.test(value)) {
    throw new Error(error);
  }
}

function validateSha256Identity(
  value: unknown,
  error: string,
): asserts value is string {
  if (typeof value !== "string" || !/^sha256:[0-9a-f]{64}$/.test(value)) {
    throw new Error(error);
  }
}

function asTcpPort(value: unknown, error: string): number {
  if (!Number.isInteger(value) || (value as number) < 1 || (value as number) > 65_535) {
    throw new Error(error);
  }
  return value as number;
}

function validateStringList(value: unknown, error: string): asserts value is string[] {
  if (
    !Array.isArray(value) ||
    value.length < 1 ||
    value.some((entry) => typeof entry !== "string" || !entry.trim()) ||
    new Set(value).size !== value.length
  ) {
    throw new Error(error);
  }
}

function assertExactObjectKeys(
  value: Record<string, unknown>,
  required: readonly string[],
  error: string,
): void {
  const keys = Object.keys(value);
  const allowed = new Set(required);
  if (
    keys.length !== required.length ||
    required.some((key) => !Object.prototype.hasOwnProperty.call(value, key)) ||
    keys.some((key) => !allowed.has(key))
  ) {
    throw new Error(error);
  }
}

function assertProbability(value: unknown, error: string): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(error);
  }
}

function assertNonNegativeFinite(value: unknown, error: string): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new Error(error);
  }
}

function assertPositiveFinite(value: unknown, error: string): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(error);
  }
}

function assertPositiveInteger(value: unknown, error: string): asserts value is number {
  if (!Number.isSafeInteger(value) || (value as number) < 1) throw new Error(error);
}

function assertBoundedPositiveInteger(
  value: unknown,
  maximum: number,
  error: string,
): asserts value is number {
  if (
    !Number.isSafeInteger(value) ||
    (value as number) < 1 ||
    (value as number) > maximum
  ) {
    throw new Error(error);
  }
}

function assertNonNegativeInteger(value: unknown, error: string): asserts value is number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) throw new Error(error);
}

function asInteger(value: unknown, error: string): number {
  if (!Number.isSafeInteger(value)) throw new Error(error);
  return value as number;
}

function asNonNegativeInteger(value: unknown, error: string): number {
  assertNonNegativeInteger(value, error);
  return value;
}

function asPositiveInteger(value: unknown, error: string): number {
  assertPositiveInteger(value, error);
  return value;
}

function asNonEmptyString(value: unknown, error: string): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(error);
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSpeculationKind(value: unknown): value is RuntimeSpeculationKind {
  return (
    value === "autoregressive" ||
    value === "ngram" ||
    value === "draft-model" ||
    value === "draft-tree" ||
    value === "mtp" ||
    value === "intermediate-head"
  );
}

function endpointEquals(left: RuntimeEndpoint, right: RuntimeEndpoint): boolean {
  return left.host === right.host && left.port === right.port;
}

function digest(value: string, length: number): string {
  return createHash("sha256").update(value).digest("hex").slice(0, length);
}
