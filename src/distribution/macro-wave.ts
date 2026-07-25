import { activationCodec } from "./codecs.js";
import {
  directedLink,
  evaluateDistributionPlan,
  rawTransferMs,
  stageMemoryBytes,
} from "./cost-model.js";
import type {
  ActivationCodecId,
  ComputeNodeProfile,
  DistributedModelProfile,
  DistributionPlan,
  DistributionPlanner,
  DistributionTopology,
  DistributionWorkload,
  MacroWavePlanContractV1,
  MacroWaveRouteProjectionV1,
  MacroWaveStageExecutionContractV1,
  StagePlacement,
} from "./types.js";

const BYTES_PER_GIGABYTE_PER_MILLISECOND = 1_000_000;

export type MacroWaveResidentMode = "layer-resident" | "ep-resident";
export type MacroWaveStageMode = MacroWaveResidentMode | "ram-backed";

export interface MacroWavePlannerOptions {
  beamWidth?: number;
  candidateCodecs?: ActivationCodecId[];
  candidateMicroBatchSizes?: number[];
  candidatePrefillChunks?: number[];
  /** Candidate positions evaluated by the target in one wave. */
  waveTokens?: number;
  /** Conservative expected committed positions from that wave. */
  expectedCommittedTokensPerWave?: number;
  /** Incremental target compute for every position after the first. */
  verificationScalePerExtraToken?: number;
  /** Double buffering is the safe default for overlapped transfer/compute. */
  activationBufferCopies?: number;
  /** Weight buffers needed for compute plus prefetch; defaults to two. */
  weightBufferCopies?: number;
  /** Compute activation dtype; independent from the wire codec. */
  activationBytesPerElement?: number;
  /** Per independently streamed layer/expert transfer. */
  transferSetupMsPerUnit?: number;
  /** Measured/predicted cache hit rate. Zero is the conservative default. */
  expectedWeightCacheHitRate?: number;
}

interface NormalizedMacroWaveOptions {
  beamWidth: number;
  candidateCodecs: ActivationCodecId[];
  candidateMicroBatchSizes: number[];
  candidatePrefillChunks: number[];
  waveTokens: number;
  expectedCommittedTokensPerWave: number;
  verificationScalePerExtraToken: number;
  activationBufferCopies: number;
  weightBufferCopies: number;
  activationBytesPerElement: number;
  transferSetupMsPerUnit: number;
  expectedWeightCacheHitRate: number;
}

export interface MacroWaveModeCost {
  mode: MacroWaveStageMode;
  feasible: boolean;
  reason: string | null;
  usableRamBytes: number;
  usableVramBytes: number;
  fullStageStateBytes: number;
  hostRamRequiredBytes: number;
  vramRequiredBytes: number;
  fixedVramBytes: number;
  residentParameterBudgetBytes: number;
  residentStreamingTransientBytes: number;
  boundedPinnedStagingReserveBytes: number;
  hostRamPeakUpperBoundBytes: number;
  activationBufferBytes: number;
  /** Portion of activationBufferBytes reserved for serial MoE temporaries. */
  expertWorkspaceBytes: number;
  weightBufferBytes: number;
  weightBufferCopies: number;
  totalWeightBytes: number;
  residentWeightBytes: number;
  replicatedWeightBytes: number;
  shardableExpertWeightBytes: number;
  totalRoutedExpertBytes: number;
  expertShardWorldSize: number | null;
  expertShardFraction: number | null;
  activeWeightBytesPerWave: number;
  expectedCacheMissWeightBytesPerWave: number;
  expectedWeightCacheHitRate: number;
  largestTransferUnitBytes: number;
  largestExpertBytes: number;
  ramReadMsPerWave: number;
  pcieTransferMsPerWave: number;
  loadMsPerWave: number;
  amortizedLoadMsPerOutputToken: number;
  /** Cold-prefill routed bytes; cache reuse is not assumed for TTFT. */
  prefillActiveWeightBytesPerChunk: number;
  prefillRamReadMsPerChunk: number;
  prefillPcieTransferMsPerChunk: number;
  prefillLoadMsPerChunk: number;
}

export interface MacroWaveLinkCost {
  from: string;
  to: string;
  oneWayLatencyMs: number;
  rttMs: number;
  bandwidthMbps: number;
  wireBytesPerWave: number;
  transferMsPerWave: number;
  amortizedTransferMsPerOutputToken: number;
}

export interface MacroWaveStageCost {
  nodeId: string;
  layerStart: number;
  layerEnd: number;
  selectedMode: MacroWaveStageMode;
  selectedBecause: string;
  resident: MacroWaveModeCost;
  ramBacked: MacroWaveModeCost;
  computeMsPerWave: number;
  amortizedComputeMsPerOutputToken: number;
  outgoing: MacroWaveLinkCost;
  serviceMsPerOutputToken: number;
}

export interface MacroWaveRejection {
  reason: string;
  count: number;
}

export interface MacroWaveRouteCost {
  kind: "resident-baseline" | "macro-wave";
  feasible: boolean;
  reason: string | null;
  plan: DistributionPlan | null;
  stages: MacroWaveStageCost[];
  ttftMs: number;
  tpotMs: number;
  responseTimeMs: number;
  pathDecodeMs: number;
  pipelineCycleMs: number;
  tokensPerSecondPerSequence: number;
  aggregateTokensPerSecond: number;
  networkBytesPerOutputToken: number;
  rawActiveWeightBytesPerOutputToken: number;
  expectedWeightCacheMissBytesPerOutputToken: number;
  expectedWeightCacheHitRate: number;
  /** Backwards-compatible alias for expectedWeightCacheMissBytesPerOutputToken. */
  ramWeightBytesPerOutputToken: number;
  routeAvailability: number;
  objective: number;
  rejectionBreakdown: MacroWaveRejection[];
}

export interface MacroWavePlanningResult {
  feasible: boolean;
  reason: string | null;
  plan: DistributionPlan | null;
  selected: MacroWaveRouteCost | null;
  alternatives: {
    resident: MacroWaveRouteCost;
    macroWave: MacroWaveRouteCost;
  };
}

interface StageShape {
  placement: StagePlacement;
  first: boolean;
  last: boolean;
  totalWeightBytes: number;
  activeRoutedExpertBytesPerWave: number;
  prefillActiveRoutedExpertBytesPerChunk: number;
  largestExpertBytes: number;
  decodeTransferUnits: number;
  prefillTransferUnits: number;
  fullStageStateBytes: number;
  fixedStageStateBytes: number;
  activationBufferBytes: number;
  expertWorkspaceBytes: number;
  routedExpertWeightBytes: number;
  residentParameterBytes: number;
  residentStreamingTransientBytes: number;
  hasCompleteResidentTensorTelemetry: boolean;
  hasCompleteExpertTelemetry: boolean;
  hasCompleteRamBackedTelemetry: boolean;
  hasCompleteExpertWorkspaceTelemetry: boolean;
}

interface PlacedStage {
  placement: StagePlacement;
  node: ComputeNodeProfile;
  selectedMode: MacroWaveStageMode;
  selectedBecause: string;
  resident: MacroWaveModeCost;
  ramBacked: MacroWaveModeCost;
  partialServiceMs: number;
}

interface BeamState {
  stages: PlacedStage[];
  usedNodeIds: Set<string>;
  nextLayer: number;
  partialCost: number;
  hasRamBackedStage: boolean;
}

interface SearchResult {
  route: MacroWaveRouteCost;
  rejected: Map<string, number>;
}

/**
 * Plan contiguous macro-stages against an explicit RAM/VRAM hierarchy.
 *
 * The ordinary distribution planners remain untouched. This planner fails
 * closed unless every selected node publishes usable RAM, usable VRAM and the
 * two measured bandwidths. Its resident baseline uses the same search and cost
 * model, while the MacroWave alternative may keep complete weights in RAM and
 * stream one layer/expert unit through VRAM per multi-token wave.
 */
export class MacroWaveRamVramPlanner implements DistributionPlanner {
  readonly id = "macro-wave-ram-vram";
  private readonly options: NormalizedMacroWaveOptions;

  constructor(options: MacroWavePlannerOptions = {}) {
    this.options = normalizeOptions(options);
  }

  plan(
    model: DistributedModelProfile,
    topology: DistributionTopology,
    workload: DistributionWorkload,
  ): DistributionPlan | null {
    return this.evaluate(model, topology, workload).plan;
  }

  evaluate(
    model: DistributedModelProfile,
    topology: DistributionTopology,
    workload: DistributionWorkload,
  ): MacroWavePlanningResult {
    const inputFailure = validateInputs(model, topology, workload);
    if (inputFailure !== null) {
      const resident = infeasibleRoute("resident-baseline", inputFailure);
      const macroWave = infeasibleRoute("macro-wave", inputFailure);
      return {
        feasible: false,
        reason: inputFailure,
        plan: null,
        selected: null,
        alternatives: { resident, macroWave },
      };
    }

    const resident = this.search(model, topology, workload, "resident-only");
    const macroWave = this.search(model, topology, workload, "macro-wave");
    const candidates = [resident.route, macroWave.route].filter((route) => route.feasible);
    if (candidates.length === 0) {
      const reason = `resident:${resident.route.reason};macro_wave:${macroWave.route.reason}`;
      return {
        feasible: false,
        reason,
        plan: null,
        selected: null,
        alternatives: { resident: resident.route, macroWave: macroWave.route },
      };
    }
    const selected = candidates.sort(
      (left, right) =>
        left.objective - right.objective ||
        (left.kind === "resident-baseline" ? -1 : 1),
    )[0]!;
    const plan = { ...selected.plan!, algorithm: this.id };
    const normalizedSelected = { ...selected, plan };
    return {
      feasible: true,
      reason: null,
      plan,
      selected: normalizedSelected,
      alternatives: { resident: resident.route, macroWave: macroWave.route },
    };
  }

  private search(
    model: DistributedModelProfile,
    topology: DistributionTopology,
    workload: DistributionWorkload,
    policy: "resident-only" | "macro-wave",
  ): SearchResult {
    const rejected = new Map<string, number>();
    const eligibleNodes = topology.nodes.filter((node) => {
      const reason = validateRamVramNode(node);
      if (reason !== null) recordRejection(rejected, `${reason}:${node.id}`);
      return reason === null && node.availability > 0;
    });
    if (eligibleNodes.length === 0) {
      return {
        route: infeasibleRoute(
          policy === "resident-only" ? "resident-baseline" : "macro-wave",
          "no_profiled_ram_vram_nodes",
          rejected,
        ),
        rejected,
      };
    }

    let best: MacroWaveRouteCost | null = null;
    for (const codec of this.options.candidateCodecs) {
      if (activationCodec(codec).estimatedQualityLoss > workload.maxQualityLoss) {
        recordRejection(rejected, `activation_quality_budget_exceeded:${codec}`);
        continue;
      }
      for (const microBatchSize of this.options.candidateMicroBatchSizes) {
        if (microBatchSize > workload.concurrentSequences) continue;
        for (const prefillChunkTokens of this.options.candidatePrefillChunks) {
          const route = this.searchConfiguration(
            model,
            topology,
            workload,
            eligibleNodes,
            policy,
            codec,
            microBatchSize,
            prefillChunkTokens,
            rejected,
          );
          if (route && (!best || route.objective < best.objective)) best = route;
        }
      }
    }
    if (best === null) {
      return {
        route: infeasibleRoute(
          policy === "resident-only" ? "resident-baseline" : "macro-wave",
          "no_contiguous_route_satisfies_ram_vram_and_links",
          rejected,
        ),
        rejected,
      };
    }
    best.rejectionBreakdown = rejectionBreakdown(rejected);
    return { route: best, rejected };
  }

  private searchConfiguration(
    model: DistributedModelProfile,
    topology: DistributionTopology,
    workload: DistributionWorkload,
    nodes: ComputeNodeProfile[],
    policy: "resident-only" | "macro-wave",
    codec: ActivationCodecId,
    microBatchSize: number,
    prefillChunkTokens: number,
    rejected: Map<string, number>,
  ): MacroWaveRouteCost | null {
    let frontier: BeamState[] = [
      {
        stages: [],
        usedNodeIds: new Set(),
        nextLayer: 0,
        partialCost: 0,
        hasRamBackedStage: false,
      },
    ];
    const completed: BeamState[] = [];
    const maximumStages = Math.min(workload.maxStages, nodes.length, model.layers.length);

    for (let stageCount = 0; stageCount < maximumStages; stageCount += 1) {
      const next: BeamState[] = [];
      for (const state of frontier) {
        for (const node of nodes) {
          if (state.usedNodeIds.has(node.id)) continue;
          const first = state.stages.length === 0;
          for (let layerEnd = state.nextLayer + 1; layerEnd <= model.layers.length; layerEnd += 1) {
            const last = layerEnd === model.layers.length;
            const placement = { nodeId: node.id, layerStart: state.nextLayer, layerEnd };
            const placed = evaluatePlacedStage(
              model,
              workload,
              node,
              placement,
              first,
              last,
              codec,
              microBatchSize,
              prefillChunkTokens,
              policy,
              this.options,
            );
            if (placed === null) {
              const reasons = stageFailureReasons(
                model,
                workload,
                node,
                placement,
                first,
                last,
                codec,
                microBatchSize,
                prefillChunkTokens,
                policy,
                this.options,
              );
              for (const reason of reasons) recordRejection(rejected, `${reason}:${node.id}`);
              // Capacity and working-set requirements only grow with the range.
              break;
            }

            let boundaryCost = 0;
            const previous = state.stages.at(-1);
            if (previous) {
              const link = directedLink(topology, previous.node.id, node.id);
              if (link === null) {
                recordRejection(rejected, `missing_link:${previous.node.id}->${node.id}`);
                continue;
              }
              boundaryCost = linkWaveCost(
                previous.node.id,
                node.id,
                model.layers[previous.placement.layerEnd - 1]!.activationElements,
                codec,
                microBatchSize,
                link,
                workload,
                this.options,
              ).amortizedTransferMsPerOutputToken;
            }
            const candidate: BeamState = {
              stages: [...state.stages, placed],
              usedNodeIds: new Set([...state.usedNodeIds, node.id]),
              nextLayer: layerEnd,
              partialCost: state.partialCost + placed.partialServiceMs + boundaryCost,
              hasRamBackedStage:
                state.hasRamBackedStage || placed.selectedMode === "ram-backed",
            };
            if (last) completed.push(candidate);
            else next.push(candidate);
          }
        }
      }
      frontier = pruneBeam(next, this.options.beamWidth);
      if (frontier.length === 0) break;
    }

    let best: MacroWaveRouteCost | null = null;
    for (const state of completed.sort((left, right) => left.partialCost - right.partialCost)) {
      if (policy === "macro-wave" && !state.hasRamBackedStage) {
        recordRejection(rejected, "macro_wave_route_has_no_ram_backed_stage");
        continue;
      }
      const plan: DistributionPlan = {
        algorithm:
          policy === "resident-only"
            ? "macro-wave-resident-baseline"
            : "macro-wave-ram-vram",
        codec,
        microBatchSize,
        prefillChunkTokens,
        stages: state.stages.map((stage) => stage.placement),
      };
      const route = finalizeRoute(
        policy === "resident-only" ? "resident-baseline" : "macro-wave",
        model,
        topology,
        workload,
        plan,
        state.stages,
        this.options,
        rejected,
      );
      if (route.feasible && (!best || route.objective < best.objective)) best = route;
    }
    return best;
  }
}

function evaluatePlacedStage(
  model: DistributedModelProfile,
  workload: DistributionWorkload,
  node: ComputeNodeProfile,
  placement: StagePlacement,
  first: boolean,
  last: boolean,
  codec: ActivationCodecId,
  microBatchSize: number,
  prefillChunkTokens: number,
  policy: "resident-only" | "macro-wave",
  options: NormalizedMacroWaveOptions,
): PlacedStage | null {
  const shape = stageShape(
    model,
    workload,
    placement,
    first,
    last,
    codec,
    microBatchSize,
    prefillChunkTokens,
    options,
  );
  const resident = residentModeCost(node, shape);
  const ramBacked = ramBackedModeCost(node, shape, options);
  const selected = resident.feasible
    ? resident
    : policy === "macro-wave" && ramBacked.feasible
      ? ramBacked
      : null;
  if (selected === null) return null;

  const decodeAtUnit = sumRange(
    model,
    placement,
    (layer) => layer.decodeMsAtUnit,
  );
  const endpointDecode =
    (first ? model.embeddingDecodeMsAtUnit : 0) +
    (last ? model.lmHeadDecodeMsAtUnit : 0);
  const batchSpeedup = Math.min(
    1 + Math.max(0, microBatchSize - 1) * node.batchGain,
    node.maxBatchSpeedup,
  );
  const targetTokenEquivalent =
    1 +
    Math.max(0, options.waveTokens - 1) *
      options.verificationScalePerExtraToken;
  const computeMsPerWave =
    ((decodeAtUnit + endpointDecode) *
      node.decodeScale *
      microBatchSize *
      targetTokenEquivalent) /
    Math.max(1, batchSpeedup);
  const amortizedCompute =
    computeMsPerWave / options.expectedCommittedTokensPerWave;
  return {
    placement,
    node,
    selectedMode: selected.mode,
    selectedBecause:
      selected.mode === "ram-backed"
        ? `${resident.reason};ram_backed_is_capacity_feasible`
        : selected.mode === "ep-resident"
          ? "profiled_expert_shard_and_replicated_weights_fit_usable_vram"
          : "complete_stage_fits_usable_vram",
    resident,
    ramBacked,
    partialServiceMs:
      amortizedCompute + selected.amortizedLoadMsPerOutputToken,
  };
}

function stageFailureReasons(
  model: DistributedModelProfile,
  workload: DistributionWorkload,
  node: ComputeNodeProfile,
  placement: StagePlacement,
  first: boolean,
  last: boolean,
  codec: ActivationCodecId,
  microBatchSize: number,
  prefillChunkTokens: number,
  policy: "resident-only" | "macro-wave",
  options: NormalizedMacroWaveOptions,
): string[] {
  const shape = stageShape(
    model,
    workload,
    placement,
    first,
    last,
    codec,
    microBatchSize,
    prefillChunkTokens,
    options,
  );
  const resident = residentModeCost(node, shape);
  if (policy === "resident-only") return [resident.reason ?? "resident_stage_rejected"];
  const ramBacked = ramBackedModeCost(node, shape, options);
  return [
    resident.reason ?? "resident_stage_rejected",
    ramBacked.reason ?? "ram_backed_stage_rejected",
  ];
}

function stageShape(
  model: DistributedModelProfile,
  workload: DistributionWorkload,
  placement: StagePlacement,
  first: boolean,
  last: boolean,
  codec: ActivationCodecId,
  microBatchSize: number,
  prefillChunkTokens: number,
  options: NormalizedMacroWaveOptions,
): StageShape {
  const layers = model.layers.slice(placement.layerStart, placement.layerEnd);
  const layerWeightBytes = layers.reduce((total, layer) => total + layer.weightBytes, 0);
  const decodeWorkingSets = layers.map((layer) =>
    conservativeRoutedWorkingSet(
      layer,
      options.waveTokens * microBatchSize,
    ),
  );
  const activeRoutedExpertBytesPerWave = decodeWorkingSets.reduce(
    (total, workingSet) => total + workingSet.bytes,
    0,
  );
  const routedExpertWeightBytes = layers.reduce(
    (total, layer) => total + (layer.expertParallel?.expertWeightBytes ?? 0),
    0,
  );
  // A cold prefill chunk can route at least one token to every expert. The
  // serial runner groups tokens by expert, so each routed expert is transferred
  // at most once per layer/chunk, but assuming the decode top-k union here would
  // make TTFT increasingly optimistic as the chunk grows.
  const prefillActiveRoutedExpertBytesPerChunk = routedExpertWeightBytes;
  const hasCompleteExpertTelemetry = layers.every(
    (layer) => layer.expertParallel !== undefined,
  );
  const hasCompleteRamBackedTelemetry = layers.every(
    (layer) =>
      layer.expertParallel !== undefined &&
      (layer.expertParallel.expertWeightBytes === 0 ||
        layer.macroWave !== undefined),
  );
  const layerUnits = layers.map((layer) => layer.macroWave?.largestTransferUnitBytes ?? 0);
  const endpointWeights = endpointWeightUnits(model, first, last);
  const totalWeightBytes = layerWeightBytes + sum(endpointWeights);
  const residentParameterBytes = totalWeightBytes - routedExpertWeightBytes;
  const layerResidentTensorUnits = layers.map((layer) => {
    const value = layer.largestResidentTensorBytes;
    const residentLayerBytes =
      layer.weightBytes - (layer.expertParallel?.expertWeightBytes ?? 0);
    return value !== undefined &&
      Number.isSafeInteger(value) &&
      value >= 0 &&
      value <= residentLayerBytes
      ? value
      : null;
  });
  const endpointResidentTensorUnits = endpointResidentStreamingUnits(
    model,
    first,
    last,
  );
  const hasCompleteResidentTensorTelemetry =
    layerResidentTensorUnits.every((value) => value !== null) &&
    endpointResidentTensorUnits !== null;
  const residentStreamingTransientBytes = hasCompleteResidentTensorTelemetry
    ? Math.max(
        0,
        ...(layerResidentTensorUnits as number[]),
        ...endpointResidentTensorUnits,
      )
    : residentParameterBytes;
  const largestExpertBytes = Math.max(0, ...layerUnits);
  const decodeTransferUnits = decodeWorkingSets.reduce(
    (total, workingSet) => total + workingSet.transferUnits,
    0,
  );
  const prefillTransferUnits = layers.reduce((total, layer) => {
    const routed = layer.expertParallel?.expertWeightBytes ?? 0;
    const largest = layer.macroWave?.largestTransferUnitBytes ?? 0;
    if (routed <= 0 || largest <= 0) return total;
    const count = layer.expertParallel?.expertCount;
    return total +
      (count !== undefined ? count : Math.ceil(routed / largest));
  }, 0);
  const fullStageStateBytes = stageMemoryBytes(model, placement, workload, first, last);
  const maximumActivationElements = Math.max(
    ...layers.map((layer) => layer.activationElements),
  );
  const activationPositions = Math.max(prefillChunkTokens, options.waveTokens);
  const activationTensorBufferBytes =
    maximumActivationElements *
    options.activationBytesPerElement *
    microBatchSize *
    activationPositions *
    options.activationBufferCopies;
  const workspacePerPosition = layers.map((layer) => {
    if ((layer.expertParallel?.expertWeightBytes ?? 0) === 0) return 0;
    const value = layer.macroWave?.expertWorkspaceBytesPerPosition;
    return value !== undefined && Number.isSafeInteger(value) && value > 0
      ? value
      : null;
  });
  const hasCompleteExpertWorkspaceTelemetry = workspacePerPosition.every(
    (value) => value !== null,
  );
  const expertWorkspaceBytes = hasCompleteExpertWorkspaceTelemetry
    ? Math.max(0, ...(workspacePerPosition as number[])) *
      microBatchSize *
      activationPositions
    : 0;
  // Keep the portable /1 contract stable: activationBufferBytes is the whole
  // non-weight tensor working set, including the now-explicit MoE workspace.
  const activationBufferBytes = activationTensorBufferBytes + expertWorkspaceBytes;
  return {
    placement,
    first,
    last,
    totalWeightBytes,
    activeRoutedExpertBytesPerWave,
    prefillActiveRoutedExpertBytesPerChunk,
    largestExpertBytes,
    decodeTransferUnits,
    prefillTransferUnits,
    fullStageStateBytes,
    fixedStageStateBytes: Math.max(0, fullStageStateBytes - totalWeightBytes),
    activationBufferBytes,
    expertWorkspaceBytes,
    routedExpertWeightBytes,
    residentParameterBytes,
    residentStreamingTransientBytes,
    hasCompleteResidentTensorTelemetry,
    hasCompleteExpertTelemetry,
    hasCompleteRamBackedTelemetry,
    hasCompleteExpertWorkspaceTelemetry,
  };
}

function conservativeRoutedWorkingSet(
  layer: DistributedModelProfile["layers"][number],
  routedTokenPositions: number,
): { bytes: number; transferUnits: number } {
  const routed = layer.expertParallel?.expertWeightBytes ?? 0;
  const largest = layer.macroWave?.largestTransferUnitBytes ?? 0;
  if (routed <= 0 || largest <= 0) return { bytes: 0, transferUnits: 0 };

  const expertCount = layer.expertParallel?.expertCount;
  const expertsPerToken = layer.expertParallel?.expertsPerToken;
  if (expertCount === undefined || expertsPerToken === undefined) {
    // Legacy profiles did not seal the wave/batch geometry behind their active
    // byte measurement. Treat the complete routed set as active instead of
    // extrapolating an ambiguous top-k sample.
    return {
      bytes: routed,
      transferUnits: Math.ceil(routed / largest),
    };
  }
  const uniqueExperts = Math.min(
    expertCount,
    routedTokenPositions * expertsPerToken,
  );
  return {
    bytes: Math.min(routed, uniqueExperts * largest),
    transferUnits: uniqueExperts,
  };
}

function residentModeCost(
  node: ComputeNodeProfile,
  shape: StageShape,
): MacroWaveModeCost {
  const hierarchy = node.ramVram!;
  const mode: MacroWaveResidentMode =
    hierarchy.residentKind === "expert-shard" ? "ep-resident" : "layer-resident";
  const expertShard = hierarchy.expertShard;
  const missingExpertTelemetry =
    mode === "ep-resident" && !shape.hasCompleteExpertTelemetry;
  const missingExpertShard = mode === "ep-resident" && expertShard === undefined;
  const expertShardFraction = mode === "ep-resident" ? (expertShard?.fraction ?? null) : null;
  const expertShardWorldSize = mode === "ep-resident" ? (expertShard?.worldSize ?? null) : null;
  const replicatedWeightBytes = shape.residentParameterBytes;
  const residentWeightBytes =
    mode === "ep-resident" && expertShardFraction !== null
      ? replicatedWeightBytes + shape.routedExpertWeightBytes * expertShardFraction
      : shape.totalWeightBytes;
  const residentStateBytes = shape.fixedStageStateBytes + residentWeightBytes;
  const vramRequiredBytes = residentStateBytes + shape.activationBufferBytes;
  let reason: string | null = null;
  if (missingExpertShard) reason = "missing_expert_shard_profile";
  else if (missingExpertTelemetry) reason = "missing_expert_weight_telemetry";
  else if (vramRequiredBytes > hierarchy.usableVramBytes) {
    reason = "resident_vram_capacity_exceeded";
  }
  return {
    mode,
    feasible: reason === null,
    reason,
    usableRamBytes: hierarchy.usableRamBytes,
    usableVramBytes: hierarchy.usableVramBytes,
    fullStageStateBytes: shape.fullStageStateBytes,
    hostRamRequiredBytes: 0,
    vramRequiredBytes,
    fixedVramBytes: shape.fixedStageStateBytes,
    residentParameterBudgetBytes: residentWeightBytes,
    residentStreamingTransientBytes: 0,
    boundedPinnedStagingReserveBytes: 0,
    hostRamPeakUpperBoundBytes: 0,
    activationBufferBytes: shape.activationBufferBytes,
    expertWorkspaceBytes: shape.expertWorkspaceBytes,
    weightBufferBytes: 0,
    weightBufferCopies: 0,
    totalWeightBytes: shape.totalWeightBytes,
    residentWeightBytes,
    replicatedWeightBytes,
    shardableExpertWeightBytes: shape.routedExpertWeightBytes,
    totalRoutedExpertBytes: shape.routedExpertWeightBytes,
    expertShardWorldSize,
    expertShardFraction,
    activeWeightBytesPerWave: 0,
    expectedCacheMissWeightBytesPerWave: 0,
    expectedWeightCacheHitRate: 0,
    largestTransferUnitBytes: shape.largestExpertBytes,
    largestExpertBytes: shape.largestExpertBytes,
    ramReadMsPerWave: 0,
    pcieTransferMsPerWave: 0,
    loadMsPerWave: 0,
    amortizedLoadMsPerOutputToken: 0,
    prefillActiveWeightBytesPerChunk: 0,
    prefillRamReadMsPerChunk: 0,
    prefillPcieTransferMsPerChunk: 0,
    prefillLoadMsPerChunk: 0,
  };
}

function ramBackedModeCost(
  node: ComputeNodeProfile,
  shape: StageShape,
  options: NormalizedMacroWaveOptions,
): MacroWaveModeCost {
  const hierarchy = node.ramVram!;
  // The direct loader owns every routed expert in host RAM. Attention,
  // routers, shared/dense MLPs and endpoint weights remain device-resident;
  // treating them as streamable would understate VRAM by multiple GiB.
  // Header-complete profiles reserve only the largest one-at-a-time resident
  // checkpoint tensor. Legacy/synthetic profiles retain the complete resident
  // set as a safe upper bound.
  const residentStreamingTransientBytes =
    shape.residentStreamingTransientBytes;
  // Production CUDA always seals two bounded pinned host slots: one executes
  // while the other prefetches. This is a capacity reservation only; the cost
  // model still does not credit transfer/compute overlap before a CUDA gate.
  const boundedPinnedStagingReserveBytes = shape.largestExpertBytes * 2;
  const hostRamPeakUpperBoundBytes =
    shape.routedExpertWeightBytes +
    boundedPinnedStagingReserveBytes +
    residentStreamingTransientBytes;
  const hostRamRequiredBytes = hostRamPeakUpperBoundBytes;
  const fixedVramBytes = shape.fixedStageStateBytes + shape.residentParameterBytes;
  const weightBufferBytes = shape.largestExpertBytes * options.weightBufferCopies;
  const vramRequiredBytes =
    fixedVramBytes + shape.activationBufferBytes + weightBufferBytes;
  let reason: string | null = null;
  if (!shape.hasCompleteRamBackedTelemetry) {
    reason = "missing_ram_backed_expert_telemetry";
  } else if (!shape.hasCompleteExpertWorkspaceTelemetry) {
    reason = "missing_ram_backed_expert_workspace";
  } else if (shape.routedExpertWeightBytes <= 0 || shape.largestExpertBytes <= 0) {
    reason = "ram_backed_stage_has_no_routed_experts";
  } else if (hostRamRequiredBytes > hierarchy.usableRamBytes) {
    reason = "ram_capacity_exceeded";
  } else if (vramRequiredBytes > hierarchy.usableVramBytes) {
    reason = "ram_backed_weight_buffers_vram_exceeded";
  } else if (
    options.expectedWeightCacheHitRate > 0 &&
    hierarchy.usableVramBytes - vramRequiredBytes < shape.largestExpertBytes
  ) {
    reason = "ram_backed_hot_cache_cannot_hold_largest_expert";
  }
  const expectedCacheMissWeightBytesPerWave =
    shape.activeRoutedExpertBytesPerWave * (1 - options.expectedWeightCacheHitRate);
  const ramReadMsPerWave =
    expectedCacheMissWeightBytesPerWave /
    (hierarchy.ramBandwidthGBps * BYTES_PER_GIGABYTE_PER_MILLISECOND);
  const pcieTransferMsPerWave =
    expectedCacheMissWeightBytesPerWave /
    (hierarchy.pcieBandwidthGBps * BYTES_PER_GIGABYTE_PER_MILLISECOND);
  // Add rather than hide one behind the other: overlap must be demonstrated by
  // a physical executor before the planner can safely subtract it.
  const loadMsPerWave =
    ramReadMsPerWave +
    pcieTransferMsPerWave +
    shape.decodeTransferUnits *
      (1 - options.expectedWeightCacheHitRate) *
      options.transferSetupMsPerUnit;
  // TTFT is reported cold: no cache hit is credited before the first chunk has
  // populated the device cache. Every routed expert may be selected by a long
  // prompt, even when the decode union is much smaller.
  const prefillActiveWeightBytesPerChunk =
    shape.prefillActiveRoutedExpertBytesPerChunk;
  const prefillRamReadMsPerChunk =
    prefillActiveWeightBytesPerChunk /
    (hierarchy.ramBandwidthGBps * BYTES_PER_GIGABYTE_PER_MILLISECOND);
  const prefillPcieTransferMsPerChunk =
    prefillActiveWeightBytesPerChunk /
    (hierarchy.pcieBandwidthGBps * BYTES_PER_GIGABYTE_PER_MILLISECOND);
  const prefillLoadMsPerChunk =
    prefillRamReadMsPerChunk +
    prefillPcieTransferMsPerChunk +
    shape.prefillTransferUnits * options.transferSetupMsPerUnit;
  return {
    mode: "ram-backed",
    feasible: reason === null,
    reason,
    usableRamBytes: hierarchy.usableRamBytes,
    usableVramBytes: hierarchy.usableVramBytes,
    fullStageStateBytes: shape.fullStageStateBytes,
    hostRamRequiredBytes,
    vramRequiredBytes,
    fixedVramBytes,
    residentParameterBudgetBytes: shape.residentParameterBytes,
    residentStreamingTransientBytes,
    boundedPinnedStagingReserveBytes,
    hostRamPeakUpperBoundBytes,
    activationBufferBytes: shape.activationBufferBytes,
    expertWorkspaceBytes: shape.expertWorkspaceBytes,
    weightBufferBytes,
    weightBufferCopies: options.weightBufferCopies,
    totalWeightBytes: shape.totalWeightBytes,
    residentWeightBytes: shape.residentParameterBytes,
    replicatedWeightBytes: shape.residentParameterBytes,
    shardableExpertWeightBytes: shape.routedExpertWeightBytes,
    totalRoutedExpertBytes: shape.routedExpertWeightBytes,
    expertShardWorldSize: null,
    expertShardFraction: null,
    activeWeightBytesPerWave: shape.activeRoutedExpertBytesPerWave,
    expectedCacheMissWeightBytesPerWave,
    expectedWeightCacheHitRate: options.expectedWeightCacheHitRate,
    largestTransferUnitBytes: shape.largestExpertBytes,
    largestExpertBytes: shape.largestExpertBytes,
    ramReadMsPerWave,
    pcieTransferMsPerWave,
    loadMsPerWave,
    amortizedLoadMsPerOutputToken:
      loadMsPerWave / options.expectedCommittedTokensPerWave,
    prefillActiveWeightBytesPerChunk,
    prefillRamReadMsPerChunk,
    prefillPcieTransferMsPerChunk,
    prefillLoadMsPerChunk,
  };
}

function finalizeRoute(
  kind: MacroWaveRouteCost["kind"],
  model: DistributedModelProfile,
  topology: DistributionTopology,
  workload: DistributionWorkload,
  plan: DistributionPlan,
  placed: PlacedStage[],
  options: NormalizedMacroWaveOptions,
  rejected: Map<string, number>,
): MacroWaveRouteCost {
  const relaxedTopology: DistributionTopology = {
    nodes: topology.nodes.map((node) => ({
      ...node,
      memoryBytes: node.ramVram
        ? node.ramVram.usableRamBytes + node.ramVram.usableVramBytes
        : node.memoryBytes,
      reserveBytes: 0,
    })),
    links: topology.links,
  };
  const base = evaluateDistributionPlan(model, relaxedTopology, workload, plan);
  if (!base.feasible) {
    recordRejection(rejected, base.infeasibleReason ?? "base_cost_model_rejected");
    return infeasibleRoute(kind, base.infeasibleReason ?? "base_cost_model_rejected", rejected);
  }

  const stageCosts: MacroWaveStageCost[] = [];
  let networkBytesPerOutputToken = 0;
  let rawActiveWeightBytesPerOutputToken = 0;
  let expectedWeightCacheMissBytesPerOutputToken = 0;
  let ramWeightBytesPerOutputToken = 0;
  for (let index = 0; index < placed.length; index += 1) {
    const stage = placed[index]!;
    const last = index === placed.length - 1;
    let outgoing: MacroWaveLinkCost;
    if (last) {
      const first = placed[0]!;
      const link = directedLink(topology, stage.node.id, first.node.id);
      if (link === null) {
        recordRejection(rejected, `missing_return_link:${stage.node.id}->${first.node.id}`);
        return infeasibleRoute(kind, "missing_direct_return_link", rejected);
      }
      outgoing = tokenReturnWaveCost(
        stage.node.id,
        first.node.id,
        plan.microBatchSize,
        link,
        workload,
        options,
      );
      networkBytesPerOutputToken +=
        (4 * options.waveTokens) / options.expectedCommittedTokensPerWave;
    } else {
      const next = placed[index + 1]!;
      const link = directedLink(topology, stage.node.id, next.node.id);
      if (link === null) {
        recordRejection(rejected, `missing_link:${stage.node.id}->${next.node.id}`);
        return infeasibleRoute(kind, "missing_stage_link", rejected);
      }
      outgoing = linkWaveCost(
        stage.node.id,
        next.node.id,
        model.layers[stage.placement.layerEnd - 1]!.activationElements,
        plan.codec,
        plan.microBatchSize,
        link,
        workload,
        options,
      );
      networkBytesPerOutputToken +=
        (model.layers[stage.placement.layerEnd - 1]!.activationElements *
          activationCodec(plan.codec).bytesPerElement *
          options.waveTokens) /
        options.expectedCommittedTokensPerWave;
    }

    const decodeAtUnit = sumRange(
      model,
      stage.placement,
      (layer) => layer.decodeMsAtUnit,
    );
    const endpointDecode =
      (index === 0 ? model.embeddingDecodeMsAtUnit : 0) +
      (last ? model.lmHeadDecodeMsAtUnit : 0);
    const speedup = Math.min(
      1 + Math.max(0, plan.microBatchSize - 1) * stage.node.batchGain,
      stage.node.maxBatchSpeedup,
    );
    const equivalentTokens =
      1 +
      Math.max(0, options.waveTokens - 1) *
        options.verificationScalePerExtraToken;
    const computeMsPerWave =
      ((decodeAtUnit + endpointDecode) *
        stage.node.decodeScale *
        plan.microBatchSize *
        equivalentTokens) /
      Math.max(1, speedup);
    const amortizedComputeMsPerOutputToken =
      computeMsPerWave / options.expectedCommittedTokensPerWave;
    const selected =
      stage.selectedMode === "ram-backed" ? stage.ramBacked : stage.resident;
    rawActiveWeightBytesPerOutputToken +=
      selected.activeWeightBytesPerWave /
      options.expectedCommittedTokensPerWave;
    expectedWeightCacheMissBytesPerOutputToken +=
      selected.expectedCacheMissWeightBytesPerWave /
      options.expectedCommittedTokensPerWave;
    ramWeightBytesPerOutputToken = expectedWeightCacheMissBytesPerOutputToken;
    stageCosts.push({
      nodeId: stage.node.id,
      layerStart: stage.placement.layerStart,
      layerEnd: stage.placement.layerEnd,
      selectedMode: stage.selectedMode,
      selectedBecause: stage.selectedBecause,
      resident: stage.resident,
      ramBacked: stage.ramBacked,
      computeMsPerWave,
      amortizedComputeMsPerOutputToken,
      outgoing,
      serviceMsPerOutputToken:
        amortizedComputeMsPerOutputToken +
        selected.amortizedLoadMsPerOutputToken +
        outgoing.amortizedTransferMsPerOutputToken,
    });
  }

  const services = stageCosts.map((stage) => stage.serviceMsPerOutputToken);
  const pathDecodeMs = sum(services);
  const pipelineCycleMs = Math.max(...services);
  const batchesPerRound = Math.ceil(workload.concurrentSequences / plan.microBatchSize);
  const batchingDelay = plan.microBatchSize > 1 ? workload.batchWindowMs : 0;
  const tpotMs = Math.max(pathDecodeMs, batchesPerRound * pipelineCycleMs) + batchingDelay;

  const prefillChunks = Math.ceil(
    workload.promptTokens / Math.max(1, Math.min(plan.prefillChunkTokens, workload.promptTokens)),
  );
  const prefillServices = base.stageMetrics.map((metric, index) => {
    const stage = stageCosts[index]!;
    const selected = stage.selectedMode === "ram-backed" ? stage.ramBacked : stage.resident;
    return (
      metric.prefillChunkComputeMs +
      metric.prefillOutgoingMs +
      selected.prefillLoadMsPerChunk
    );
  });
  const firstTokenHeadMs =
    model.lmHeadDecodeMsAtUnit * placed.at(-1)!.node.decodeScale;
  const ttftMs =
    sum(prefillServices) +
    Math.max(0, prefillChunks - 1) * Math.max(...prefillServices) +
    firstTokenHeadMs +
    base.tokenReturnMs +
    batchingDelay;
  const responseTimeMs = ttftMs + Math.max(0, workload.outputTokens - 1) * tpotMs;
  const objective = tpotMs + ttftMs * 0.05;
  const projection: MacroWaveRouteProjectionV1 = {
    ttftMs,
    tpotMs,
    responseTimeMs,
    pathDecodeMs,
    pipelineCycleMs,
    tokensPerSecondPerSequence: 1_000 / tpotMs,
    aggregateTokensPerSecond:
      (workload.concurrentSequences * 1_000) / tpotMs,
    networkBytesPerOutputToken,
    rawActiveWeightBytesPerOutputToken,
    expectedWeightCacheMissBytesPerOutputToken,
    routeAvailability: base.routeAvailability,
  };
  const executablePlan = attachMacroWaveExecutionContracts(
    plan,
    kind,
    stageCosts,
    options,
    projection,
  );

  return {
    kind,
    feasible: true,
    reason: null,
    plan: executablePlan,
    stages: stageCosts,
    ttftMs,
    tpotMs,
    responseTimeMs,
    pathDecodeMs,
    pipelineCycleMs,
    tokensPerSecondPerSequence: projection.tokensPerSecondPerSequence,
    aggregateTokensPerSecond: projection.aggregateTokensPerSecond,
    networkBytesPerOutputToken,
    rawActiveWeightBytesPerOutputToken,
    expectedWeightCacheMissBytesPerOutputToken,
    expectedWeightCacheHitRate: options.expectedWeightCacheHitRate,
    ramWeightBytesPerOutputToken,
    routeAvailability: base.routeAvailability,
    objective,
    rejectionBreakdown: rejectionBreakdown(rejected),
  };
}

function attachMacroWaveExecutionContracts(
  plan: DistributionPlan,
  routeKind: MacroWavePlanContractV1["routeKind"],
  stages: readonly MacroWaveStageCost[],
  options: NormalizedMacroWaveOptions,
  projection: MacroWaveRouteProjectionV1,
): DistributionPlan {
  if (plan.stages.length !== stages.length) {
    throw new Error("macro_wave_stage_contract_count_mismatch");
  }
  return {
    ...plan,
    macroWave: {
      schema: "gdlp-macro-wave-plan/1",
      routeKind,
      waveTokens: options.waveTokens,
      expectedCommittedTokensPerWave: options.expectedCommittedTokensPerWave,
      projection: { ...projection },
    },
    stages: plan.stages.map((placement, index) => ({
      ...placement,
      macroWave: macroWaveStageExecutionContract(stages[index]!),
    })),
  };
}

function macroWaveStageExecutionContract(
  stage: MacroWaveStageCost,
): MacroWaveStageExecutionContractV1 {
  const selected =
    stage.selectedMode === "ram-backed" ? stage.ramBacked : stage.resident;
  const memoryMode = stage.selectedMode === "ram-backed" ? "ram-backed" : "resident";
  const residentKind =
    stage.selectedMode === "ep-resident" ? "expert-shard" : "layers";
  const cacheCapacityBytes = Math.ceil(
    memoryMode === "resident"
      ? selected.residentWeightBytes
      : selected.expectedWeightCacheHitRate > 0
        ? Math.max(0, selected.usableVramBytes - selected.vramRequiredBytes)
        : 0,
  );
  const cacheKind =
    memoryMode === "resident"
      ? "full-resident"
      : selected.expectedWeightCacheHitRate > 0
        ? "bounded-lru"
        : "disabled";
  return {
    mode: "macro-wave-memory",
    schema: "gdlp-macro-wave-stage/1",
    memoryMode,
    residentKind,
    budgets: {
      hostRamBytes: selected.usableRamBytes,
      vramBytes: selected.usableVramBytes,
    },
    requirements: {
      fullStageStateBytes: Math.ceil(selected.fullStageStateBytes),
      hostRamBytes: Math.ceil(selected.hostRamRequiredBytes),
      vramBytes: Math.ceil(selected.vramRequiredBytes),
      fixedVramBytes: Math.ceil(selected.fixedVramBytes),
      residentParameterBudgetBytes: Math.ceil(
        selected.residentParameterBudgetBytes,
      ),
      residentStreamingTransientBytes: Math.ceil(
        selected.residentStreamingTransientBytes,
      ),
      boundedPinnedStagingReserveBytes: Math.ceil(
        selected.boundedPinnedStagingReserveBytes,
      ),
      hostRamPeakUpperBoundBytes: Math.ceil(
        selected.hostRamPeakUpperBoundBytes,
      ),
      activationBufferBytes: Math.ceil(selected.activationBufferBytes),
      weightBufferBytes: Math.ceil(selected.weightBufferBytes),
      weightBufferCopies: selected.weightBufferCopies,
    },
    workingSet: {
      totalWeightBytes: Math.ceil(selected.totalWeightBytes),
      residentWeightBytes: Math.ceil(selected.residentWeightBytes),
      totalRoutedExpertBytes: Math.ceil(selected.totalRoutedExpertBytes),
      activeWeightBytesPerWave: Math.ceil(selected.activeWeightBytesPerWave),
      largestTransferUnitBytes: Math.ceil(selected.largestTransferUnitBytes),
      largestExpertBytes: Math.ceil(selected.largestExpertBytes),
    },
    cachePolicy: {
      kind: cacheKind,
      capacityBytes: cacheCapacityBytes,
      expectedHitRate:
        memoryMode === "resident" ? 1 : selected.expectedWeightCacheHitRate,
      expectedMissWeightBytesPerWave: Math.ceil(
        selected.expectedCacheMissWeightBytesPerWave,
      ),
    },
    ...(memoryMode === "ram-backed"
      ? {
          ramArtifact: {
            schema: "gdlp-local-safetensors-moe-stage/1" as const,
            format: "safetensors" as const,
            locality: "host-local-only" as const,
            loader: "selective-safetensors-ram-backed-moe" as const,
            weightEncoding: "floating-safetensors" as const,
            sourceDtypes: ["fp16", "bf16", "fp32"] as [
              "fp16",
              "bf16",
              "fp32",
            ],
            adapterIds: [
              "transformers-qwen3-moe-v1",
              "transformers-glm4-moe-v1",
            ] as [
              "transformers-qwen3-moe-v1",
              "transformers-glm4-moe-v1",
            ],
            expertExecutionMode: "serial-exact" as const,
            largestExpertBytes: Math.ceil(selected.largestExpertBytes),
            weightBufferCopies: 2 as const,
            fullModelMaterialization: false as const,
          },
        }
      : {}),
  };
}

function linkWaveCost(
  from: string,
  to: string,
  activationElements: number,
  codec: ActivationCodecId,
  microBatchSize: number,
  link: NonNullable<ReturnType<typeof directedLink>>,
  workload: DistributionWorkload,
  options: NormalizedMacroWaveOptions,
): MacroWaveLinkCost {
  const wireBytesPerWave =
    activationElements *
    activationCodec(codec).bytesPerElement *
    microBatchSize *
    options.waveTokens;
  return measuredLinkCost(from, to, wireBytesPerWave, link, workload, options);
}

function tokenReturnWaveCost(
  from: string,
  to: string,
  microBatchSize: number,
  link: NonNullable<ReturnType<typeof directedLink>>,
  workload: DistributionWorkload,
  options: NormalizedMacroWaveOptions,
): MacroWaveLinkCost {
  return measuredLinkCost(
    from,
    to,
    4 * microBatchSize * options.waveTokens,
    link,
    workload,
    options,
  );
}

function measuredLinkCost(
  from: string,
  to: string,
  wireBytesPerWave: number,
  link: NonNullable<ReturnType<typeof directedLink>>,
  workload: DistributionWorkload,
  options: NormalizedMacroWaveOptions,
): MacroWaveLinkCost {
  const transferMsPerWave = rawTransferMs(wireBytesPerWave, link, workload.p95);
  return {
    from,
    to,
    oneWayLatencyMs: link.oneWayLatencyMs,
    rttMs: link.oneWayLatencyMs * 2,
    bandwidthMbps: link.bandwidthMbps,
    wireBytesPerWave,
    transferMsPerWave,
    amortizedTransferMsPerOutputToken:
      transferMsPerWave / options.expectedCommittedTokensPerWave,
  };
}

function endpointWeightUnits(
  model: DistributedModelProfile,
  first: boolean,
  last: boolean,
): number[] {
  if (first && last && model.tiedEmbeddingAndHead) {
    return [Math.max(model.embeddingBytes, model.lmHeadBytes)];
  }
  return [first ? model.embeddingBytes : 0, last ? model.lmHeadBytes : 0].filter(
    (value) => value > 0,
  );
}

function endpointResidentStreamingUnits(
  model: DistributedModelProfile,
  first: boolean,
  last: boolean,
): number[] | null {
  const units: number[] = [];
  if (first) {
    const value = model.largestEmbeddingTensorBytes;
    if (
      value === undefined ||
      !Number.isSafeInteger(value) ||
      value < 0 ||
      value > model.embeddingBytes
    ) {
      return null;
    }
    units.push(value);
  }
  if (last) {
    const value = model.largestLmHeadTensorBytes;
    if (
      value === undefined ||
      !Number.isSafeInteger(value) ||
      value < 0 ||
      value > model.lmHeadBytes
    ) {
      return null;
    }
    units.push(value);
  }
  return units;
}

function pruneBeam(states: BeamState[], beamWidth: number): BeamState[] {
  const deduplicated = new Map<string, BeamState>();
  for (const state of states) {
    const key = `${state.nextLayer}|${state.stages.at(-1)?.node.id ?? "root"}|${[
      ...state.usedNodeIds,
    ]
      .sort()
      .join(",")}|${state.hasRamBackedStage}`;
    const previous = deduplicated.get(key);
    if (!previous || state.partialCost < previous.partialCost) deduplicated.set(key, state);
  }
  return [...deduplicated.values()]
    .sort((left, right) => left.partialCost - right.partialCost)
    .slice(0, beamWidth);
}

function normalizeOptions(options: MacroWavePlannerOptions): NormalizedMacroWaveOptions {
  const normalized: NormalizedMacroWaveOptions = {
    beamWidth: options.beamWidth ?? 256,
    candidateCodecs: options.candidateCodecs ?? ["fp16"],
    // See planners.ts DEFAULT_SEARCH_OPTIONS: admission capped at 8 halves measured
    // aggregate throughput on separate GPUs; 32 is the measured knee.
    candidateMicroBatchSizes: options.candidateMicroBatchSizes ?? [1, 2, 4, 8, 16, 32],
    candidatePrefillChunks: options.candidatePrefillChunks ?? [32, 64, 128],
    waveTokens: options.waveTokens ?? 1,
    expectedCommittedTokensPerWave: options.expectedCommittedTokensPerWave ?? 1,
    verificationScalePerExtraToken: options.verificationScalePerExtraToken ?? 1,
    activationBufferCopies: options.activationBufferCopies ?? 2,
    weightBufferCopies: options.weightBufferCopies ?? 2,
    activationBytesPerElement: options.activationBytesPerElement ?? 2,
    transferSetupMsPerUnit: options.transferSetupMsPerUnit ?? 0,
    expectedWeightCacheHitRate: options.expectedWeightCacheHitRate ?? 0,
  };
  requirePositiveInteger("beamWidth", normalized.beamWidth);
  requirePositiveInteger("waveTokens", normalized.waveTokens);
  requirePositiveInteger("activationBufferCopies", normalized.activationBufferCopies);
  requirePositiveInteger("weightBufferCopies", normalized.weightBufferCopies);
  if (normalized.weightBufferCopies !== 2) {
    throw new Error("weightBufferCopies must be 2 for serial-exact ping-pong execution");
  }
  requirePositiveFinite("expectedCommittedTokensPerWave", normalized.expectedCommittedTokensPerWave);
  if (normalized.expectedCommittedTokensPerWave > normalized.waveTokens) {
    throw new Error("expectedCommittedTokensPerWave cannot exceed waveTokens");
  }
  requirePositiveFinite("activationBytesPerElement", normalized.activationBytesPerElement);
  requireNonNegativeFinite(
    "verificationScalePerExtraToken",
    normalized.verificationScalePerExtraToken,
  );
  requireNonNegativeFinite("transferSetupMsPerUnit", normalized.transferSetupMsPerUnit);
  if (
    !Number.isFinite(normalized.expectedWeightCacheHitRate) ||
    normalized.expectedWeightCacheHitRate < 0 ||
    normalized.expectedWeightCacheHitRate > 1
  ) {
    throw new Error("expectedWeightCacheHitRate must be between zero and one");
  }
  if (normalized.candidateCodecs.length === 0) {
    throw new Error("candidateCodecs cannot be empty");
  }
  normalized.candidateCodecs = [...new Set(normalized.candidateCodecs)];
  normalized.candidateMicroBatchSizes = normalizePositiveIntegerList(
    "candidateMicroBatchSizes",
    normalized.candidateMicroBatchSizes,
  );
  normalized.candidatePrefillChunks = normalizePositiveIntegerList(
    "candidatePrefillChunks",
    normalized.candidatePrefillChunks,
  );
  return normalized;
}

function validateInputs(
  model: DistributedModelProfile,
  topology: DistributionTopology,
  workload: DistributionWorkload,
): string | null {
  if (model.layers.length === 0) return "model_has_no_layers";
  if (topology.nodes.length === 0) return "topology_has_no_nodes";
  if (!Number.isInteger(workload.maxStages) || workload.maxStages < 1) {
    return "workload_max_stages_is_invalid";
  }
  for (const [index, layer] of model.layers.entries()) {
    if (
      layer.index !== index ||
      !nonNegativeInteger(layer.weightBytes) ||
      !Number.isInteger(layer.activationElements) ||
      layer.activationElements < 1 ||
      !nonNegativeInteger(layer.kvBytesPerToken) ||
      !positiveFinite(layer.decodeMsAtUnit) ||
      !positiveFinite(layer.prefillMsPerTokenAtUnit)
    ) {
      return `invalid_layer_profile:${index}`;
    }
    if (layer.macroWave) {
      const active = layer.macroWave.activeWeightBytesPerWave;
      const unit = layer.macroWave.largestTransferUnitBytes;
      const workspace = layer.macroWave.expertWorkspaceBytesPerPosition;
      if (
        (active !== undefined &&
          (!nonNegativeInteger(active) || active > layer.weightBytes)) ||
        !nonNegativeInteger(unit) ||
        (workspace !== undefined && !positiveInteger(workspace)) ||
        (active !== undefined && unit > active) ||
        ((active ?? layer.expertParallel?.expertWeightBytes ?? 0) > 0 && unit === 0)
      ) {
        return `invalid_macro_wave_layer_profile:${index}`;
      }
    }
    if (layer.expertParallel) {
      const expertCount = layer.expertParallel.expertCount;
      const expertsPerToken = layer.expertParallel.expertsPerToken;
      const hasExpertCount = expertCount !== undefined;
      const hasExpertsPerToken = expertsPerToken !== undefined;
      if (
        !nonNegativeInteger(layer.expertParallel.expertWeightBytes) ||
        layer.expertParallel.expertWeightBytes > layer.weightBytes ||
        hasExpertCount !== hasExpertsPerToken ||
        (expertCount !== undefined &&
          expertsPerToken !== undefined &&
          (!positiveInteger(expertCount) ||
            !positiveInteger(expertsPerToken) ||
            expertsPerToken > expertCount))
      ) {
        return `invalid_expert_parallel_layer_profile:${index}`;
      }
    }
    if (
      layer.macroWave &&
      layer.expertParallel &&
      ((layer.macroWave.activeWeightBytesPerWave !== undefined &&
        layer.macroWave.activeWeightBytesPerWave >
          layer.expertParallel.expertWeightBytes) ||
        layer.macroWave.largestTransferUnitBytes >
          layer.expertParallel.expertWeightBytes)
    ) {
      return `macro_wave_exceeds_routed_expert_weights:${index}`;
    }
    if (
      layer.macroWave !== undefined &&
      layer.macroWave?.activeWeightBytesPerWave === undefined &&
      (layer.expertParallel?.expertCount === undefined ||
        layer.expertParallel.expertsPerToken === undefined)
    ) {
      return `macro_wave_missing_working_set_geometry:${index}`;
    }
  }
  return null;
}

function validateRamVramNode(node: ComputeNodeProfile): string | null {
  const profile = node.ramVram;
  if (!profile) return "missing_ram_vram_profile";
  if (!positiveInteger(profile.usableRamBytes)) return "invalid_usable_ram";
  if (!positiveInteger(profile.usableVramBytes)) return "invalid_usable_vram";
  if (!positiveFinite(profile.ramBandwidthGBps)) return "invalid_ram_bandwidth";
  if (!positiveFinite(profile.pcieBandwidthGBps)) return "invalid_pcie_bandwidth";
  if (
    profile.residentKind !== undefined &&
    profile.residentKind !== "layers" &&
    profile.residentKind !== "expert-shard"
  ) {
    return "invalid_resident_kind";
  }
  if (profile.residentKind === "expert-shard") {
    const shard = profile.expertShard;
    if (!shard) return "missing_expert_shard_profile";
    if (!positiveInteger(shard.worldSize)) return "invalid_expert_shard_world_size";
    if (!positiveFinite(shard.fraction) || shard.fraction > 1) {
      return "invalid_expert_shard_fraction";
    }
    if (shard.fraction * shard.worldSize < 1) {
      return "expert_shard_fraction_understates_world_coverage";
    }
  } else if (profile.expertShard !== undefined) {
    return "expert_shard_profile_requires_expert_shard_kind";
  }
  return null;
}

function infeasibleRoute(
  kind: MacroWaveRouteCost["kind"],
  reason: string,
  rejected: Map<string, number> = new Map(),
): MacroWaveRouteCost {
  return {
    kind,
    feasible: false,
    reason,
    plan: null,
    stages: [],
    ttftMs: Number.POSITIVE_INFINITY,
    tpotMs: Number.POSITIVE_INFINITY,
    responseTimeMs: Number.POSITIVE_INFINITY,
    pathDecodeMs: Number.POSITIVE_INFINITY,
    pipelineCycleMs: Number.POSITIVE_INFINITY,
    tokensPerSecondPerSequence: 0,
    aggregateTokensPerSecond: 0,
    networkBytesPerOutputToken: Number.POSITIVE_INFINITY,
    rawActiveWeightBytesPerOutputToken: Number.POSITIVE_INFINITY,
    expectedWeightCacheMissBytesPerOutputToken: Number.POSITIVE_INFINITY,
    expectedWeightCacheHitRate: 0,
    ramWeightBytesPerOutputToken: Number.POSITIVE_INFINITY,
    routeAvailability: 0,
    objective: Number.POSITIVE_INFINITY,
    rejectionBreakdown: rejectionBreakdown(rejected),
  };
}

function recordRejection(rejected: Map<string, number>, reason: string): void {
  rejected.set(reason, (rejected.get(reason) ?? 0) + 1);
}

function rejectionBreakdown(rejected: Map<string, number>): MacroWaveRejection[] {
  return [...rejected.entries()]
    .map(([reason, count]) => ({ reason, count }))
    .sort((left, right) => right.count - left.count || left.reason.localeCompare(right.reason));
}

function sumRange(
  model: DistributedModelProfile,
  placement: StagePlacement,
  select: (layer: DistributedModelProfile["layers"][number]) => number,
): number {
  let total = 0;
  for (let index = placement.layerStart; index < placement.layerEnd; index += 1) {
    total += select(model.layers[index]!);
  }
  return total;
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function positiveFinite(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

function positiveInteger(value: number): boolean {
  return Number.isInteger(value) && value > 0;
}

function nonNegativeInteger(value: number): boolean {
  return Number.isInteger(value) && value >= 0;
}

function requirePositiveInteger(name: string, value: number): void {
  if (!positiveInteger(value)) throw new Error(`${name} must be a positive integer`);
}

function requirePositiveFinite(name: string, value: number): void {
  if (!positiveFinite(value)) throw new Error(`${name} must be finite and positive`);
}

function requireNonNegativeFinite(name: string, value: number): void {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`${name} must be finite and non-negative`);
  }
}

function normalizePositiveIntegerList(name: string, values: number[]): number[] {
  if (values.length === 0 || values.some((value) => !positiveInteger(value))) {
    throw new Error(`${name} must contain positive integers`);
  }
  return [...new Set(values)].sort((left, right) => left - right);
}
