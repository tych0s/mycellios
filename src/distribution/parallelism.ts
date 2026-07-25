import { activationCodec } from "./codecs.js";
import {
  directedLink,
  evaluateDistributionPlan,
  rawTransferMs,
  stageMemoryBytes,
} from "./cost-model.js";
import type {
  ComputeNodeProfile,
  DirectedLinkProfile,
  DistributedModelProfile,
  DistributionPlan,
  DistributionTopology,
  DistributionWorkload,
} from "./types.js";

export type ParallelismKind =
  | "single-device"
  | "contiguous-pipeline"
  | "tensor-ring"
  | "tensor-star"
  | "speculative-pipeline";

export interface ParallelismEstimate {
  architecture: ParallelismKind;
  feasible: boolean;
  reason: string | null;
  nodes: number;
  tpotMs: number;
  ttftMs: number;
  tokensPerSecondPerSequence: number;
  aggregateTokensPerSecond: number;
  networkBytesPerOutputToken: number;
  detail: string;
  /** Non-binding advisory; never affects feasibility or selection. */
  warning?: string;
}

/**
 * external runtime B/RDMA-vs-TCP evidence (docs/INVESTIGACION_EXTERNA_OPTIMIZACION_2026-07-19.md
 * section 5.2): tensor parallelism scales on links of tens of microseconds and
 * stops scaling around ~300 us per message, so 0.3 ms one-way is the viability
 * ceiling for a TP cell interconnect.
 */
export const TP_CELL_MAX_ONE_WAY_LATENCY_MS = 0.3;
export const TP_CELL_MIN_SUCCESSFUL_LINK_SAMPLES = 3;
export const TP_CELL_MIN_LINK_AVAILABILITY = 0.9;

export function isViableTensorParallelCellLatency(oneWayLatencyMs: number): boolean {
  return (
    Number.isFinite(oneWayLatencyMs) &&
    oneWayLatencyMs >= 0 &&
    oneWayLatencyMs <= TP_CELL_MAX_ONE_WAY_LATENCY_MS
  );
}

export interface SpeculationProfile {
  /** Time spent by the local draft model for one candidate token. */
  draftMsPerToken: number;
  /** Probability that each consecutive draft token is accepted. */
  acceptanceProbability: number;
  /** Incremental target work for each extra token verified in one pass. */
  verificationScalePerExtraToken: number;
  maxDraftTokens: number;
}

/**
 * Compare the optimized contiguous pipeline with common distributed baselines.
 *
 * Tensor estimates deliberately use the same nodes as the selected pipeline;
 * this exposes the synchronization penalty instead of silently giving one
 * architecture a different fleet. They are analytical estimates, not claims
 * that a particular collective library will hit the exact number.
 */
export function compareParallelismArchitectures(
  model: DistributedModelProfile,
  topology: DistributionTopology,
  workload: DistributionWorkload,
  pipelinePlan: DistributionPlan,
  speculation?: SpeculationProfile,
): ParallelismEstimate[] {
  const pipelineMetrics = evaluateDistributionPlan(model, topology, workload, pipelinePlan);
  const pipeline: ParallelismEstimate = pipelineMetrics.feasible
    ? {
        architecture: "contiguous-pipeline",
        feasible: true,
        reason: null,
        nodes: pipelinePlan.stages.length,
        tpotMs: pipelineMetrics.tpotMs,
        ttftMs: pipelineMetrics.ttftMs,
        tokensPerSecondPerSequence: pipelineMetrics.tokensPerSecondPerSequence,
        aggregateTokensPerSecond: pipelineMetrics.aggregateTokensPerSecond,
        networkBytesPerOutputToken: pipelineMetrics.networkBytesPerOutputToken,
        detail: `${pipelinePlan.codec}, ${pipelinePlan.stages.length} stages, microbatch ${pipelinePlan.microBatchSize}`,
      }
    : infeasible("contiguous-pipeline", pipelinePlan.stages.length, pipelineMetrics.infeasibleReason);

  const nodeIds = pipelinePlan.stages.map((stage) => stage.nodeId);
  const estimates = [
    bestSingleDevice(model, topology, workload),
    pipeline,
    tensorParallelEstimate("tensor-ring", model, topology, workload, nodeIds),
    tensorParallelEstimate("tensor-star", model, topology, workload, nodeIds),
  ];
  if (speculation && pipeline.feasible) {
    estimates.push(speculativePipelineEstimate(pipeline, speculation));
  }
  return estimates;
}

function bestSingleDevice(
  model: DistributedModelProfile,
  topology: DistributionTopology,
  workload: DistributionWorkload,
): ParallelismEstimate {
  let best: ParallelismEstimate | null = null;
  for (const node of topology.nodes) {
    const plan: DistributionPlan = {
      algorithm: "single-device",
      codec: "fp16",
      microBatchSize: 1,
      prefillChunkTokens: Math.max(1, workload.promptTokens),
      stages: [{ nodeId: node.id, layerStart: 0, layerEnd: model.layers.length }],
    };
    const metrics = evaluateDistributionPlan(model, topology, workload, plan);
    if (!metrics.feasible) continue;
    const candidate: ParallelismEstimate = {
      architecture: "single-device",
      feasible: true,
      reason: null,
      nodes: 1,
      tpotMs: metrics.tpotMs,
      ttftMs: metrics.ttftMs,
      tokensPerSecondPerSequence: metrics.tokensPerSecondPerSequence,
      aggregateTokensPerSecond: metrics.aggregateTokensPerSecond,
      networkBytesPerOutputToken: 0,
      detail: node.id,
    };
    if (!best || candidate.tpotMs < best.tpotMs) best = candidate;
  }
  return best ?? infeasible("single-device", 1, "model_does_not_fit_any_node");
}

function tensorParallelEstimate(
  architecture: "tensor-ring" | "tensor-star",
  model: DistributedModelProfile,
  topology: DistributionTopology,
  workload: DistributionWorkload,
  nodeIds: string[],
): ParallelismEstimate {
  if (nodeIds.length < 2) return infeasible(architecture, nodeIds.length, "needs_multiple_nodes");
  const byId = new Map(topology.nodes.map((node) => [node.id, node]));
  const nodes = nodeIds.map((id) => byId.get(id)).filter(Boolean) as ComputeNodeProfile[];
  if (nodes.length !== nodeIds.length) return infeasible(architecture, nodes.length, "unknown_node");
  const linkGateReason = tensorParallelCellGateReason(nodeIds, topology);
  if (linkGateReason !== null) {
    return infeasible(architecture, nodes.length, linkGateReason);
  }
  if (!tensorShardFits(model, workload, nodes)) {
    return infeasible(architecture, nodes.length, "tensor_shard_memory_exceeded");
  }

  const batch = 1;
  let decodeMs = 0;
  let prefillMs = 0;
  let networkBytes = 0;
  for (const layer of model.layers) {
    const decodeCompute = Math.max(
      ...nodes.map((node) => (layer.decodeMsAtUnit * node.decodeScale * batch) / nodes.length),
    );
    const prefillCompute = Math.max(
      ...nodes.map(
        (node) =>
          (layer.prefillMsPerTokenAtUnit * node.prefillScale * workload.promptTokens) /
          nodes.length,
      ),
    );
    const activationBytes = layer.activationElements * activationCodec("fp16").bytesPerElement;
    const decodeCollective =
      architecture === "tensor-ring"
        ? ringAllReduceMs(activationBytes * batch, nodeIds, topology, workload.p95)
        : starAllReduceMs(activationBytes * batch, nodeIds, topology, workload.p95);
    const prefillCollective =
      architecture === "tensor-ring"
        ? ringAllReduceMs(
            activationBytes * workload.promptTokens,
            nodeIds,
            topology,
            workload.p95,
          )
        : starAllReduceMs(
            activationBytes * workload.promptTokens,
            nodeIds,
            topology,
            workload.p95,
          );
    if (!Number.isFinite(decodeCollective) || !Number.isFinite(prefillCollective)) {
      return infeasible(architecture, nodes.length, "collective_link_missing");
    }
    // Decoder TP performs two collectives per transformer block.
    decodeMs += decodeCompute + 2 * decodeCollective;
    prefillMs += prefillCompute + 2 * prefillCollective;
    networkBytes += collectiveWireBytes(architecture, activationBytes, nodes.length) * 2;
  }
  const endpointDecode =
    model.embeddingDecodeMsAtUnit * nodes[0]!.decodeScale +
    model.lmHeadDecodeMsAtUnit * nodes[0]!.decodeScale;
  const endpointPrefill =
    (model.embeddingPrefillMsPerTokenAtUnit + model.lmHeadPrefillMsPerTokenAtUnit) *
    nodes[0]!.prefillScale *
    workload.promptTokens;
  decodeMs += endpointDecode;
  prefillMs += endpointPrefill;

  const batchesPerRound = workload.concurrentSequences;
  const tpotMs = decodeMs * batchesPerRound;
  return {
    architecture,
    feasible: true,
    reason: null,
    nodes: nodes.length,
    tpotMs,
    ttftMs: prefillMs + decodeMs,
    tokensPerSecondPerSequence: 1_000 / tpotMs,
    aggregateTokensPerSecond: (workload.concurrentSequences * 1_000) / tpotMs,
    networkBytesPerOutputToken: networkBytes,
    detail:
      architecture === "tensor-ring"
        ? "2 ring all-reduces per layer (FP16)"
        : "2 root gather/broadcast collectives per layer (FP16)",
  };
}

/**
 * Fail-closed physical admission for a TP cell. Every ordered member pair must
 * have fresh, sufficiently sampled runtime evidence in the same locality
 * domain. Region defaults and analytical links are intentionally ineligible.
 */
export function tensorParallelCellGateReason(
  nodeIds: string[],
  topology: {
    nodes: readonly { id: string; region?: string }[];
    links: readonly DirectedLinkProfile[];
  },
  now = Date.now(),
): string | null {
  const byId = new Map(topology.nodes.map((node) => [node.id, node]));
  const regions = new Set(nodeIds.map((nodeId) => byId.get(nodeId)?.region));
  if (regions.has(undefined) || regions.size !== 1) {
    return "tp_cell_cross_region_forbidden";
  }
  for (const from of nodeIds) {
    for (const to of nodeIds) {
      if (from === to) continue;
      const link = topology.links.find(
        (candidate) => candidate.from === from && candidate.to === to,
      );
      if (!link) return "collective_link_missing";
      const evidence = link.evidence;
      if (!evidence || evidence.source !== "runtime-probe") {
        return "tp_cell_link_evidence_missing";
      }
      if (evidence.validUntil <= now || evidence.measuredAt > now) {
        return "tp_cell_link_evidence_expired";
      }
      if (
        evidence.successfulSamples < TP_CELL_MIN_SUCCESSFUL_LINK_SAMPLES
      ) {
        return "tp_cell_link_confidence_insufficient";
      }
      if (
        (link.availability ?? 0) < TP_CELL_MIN_LINK_AVAILABILITY
      ) {
        return "tp_cell_link_availability_too_low";
      }
      if (!isViableTensorParallelCellLatency(link.oneWayLatencyMs)) {
        return "tp_cell_one_way_latency_above_viability_ceiling";
      }
    }
  }
  return null;
}

function tensorShardFits(
  model: DistributedModelProfile,
  workload: DistributionWorkload,
  nodes: ComputeNodeProfile[],
): boolean {
  const complete = {
    nodeId: nodes[0]!.id,
    layerStart: 0,
    layerEnd: model.layers.length,
  };
  const total = stageMemoryBytes(model, complete, workload, true, true);
  // Runtime state is replicated, while weights and KV are approximately sharded.
  const replicated = model.runtimeOverheadBytesPerStage;
  const perNode = replicated + Math.max(0, total - replicated) / nodes.length;
  return nodes.every((node) => perNode <= node.memoryBytes - node.reserveBytes);
}

function ringAllReduceMs(
  messageBytes: number,
  nodeIds: string[],
  topology: DistributionTopology,
  p95: boolean,
): number {
  let slowestStep = 0;
  for (let index = 0; index < nodeIds.length; index += 1) {
    const from = nodeIds[index]!;
    const to = nodeIds[(index + 1) % nodeIds.length]!;
    const link = directedLink(topology, from, to);
    if (!link) return Number.POSITIVE_INFINITY;
    slowestStep = Math.max(slowestStep, rawTransferMs(messageBytes / nodeIds.length, link, p95));
  }
  return 2 * (nodeIds.length - 1) * slowestStep;
}

function starAllReduceMs(
  messageBytes: number,
  nodeIds: string[],
  topology: DistributionTopology,
  p95: boolean,
): number {
  let bestRoot = Number.POSITIVE_INFINITY;
  for (const root of nodeIds) {
    let worstRoundTrip = 0;
    let valid = true;
    for (const peer of nodeIds) {
      if (peer === root) continue;
      const inbound = directedLink(topology, peer, root);
      const outbound = directedLink(topology, root, peer);
      if (!inbound || !outbound) {
        valid = false;
        break;
      }
      worstRoundTrip = Math.max(
        worstRoundTrip,
        rawTransferMs(messageBytes, inbound, p95) +
          rawTransferMs(messageBytes, outbound, p95),
      );
    }
    if (valid) bestRoot = Math.min(bestRoot, worstRoundTrip);
  }
  return bestRoot;
}

function collectiveWireBytes(
  architecture: "tensor-ring" | "tensor-star",
  messageBytes: number,
  nodes: number,
): number {
  if (architecture === "tensor-ring") {
    return (2 * (nodes - 1) * messageBytes) / nodes;
  }
  return 2 * (nodes - 1) * messageBytes;
}

function speculativePipelineEstimate(
  target: ParallelismEstimate,
  profile: SpeculationProfile,
): ParallelismEstimate {
  const acceptance = Math.max(0, Math.min(0.999, profile.acceptanceProbability));
  let bestTpot = target.tpotMs;
  let bestLength = 0;
  for (let length = 1; length <= profile.maxDraftTokens; length += 1) {
    const expectedTokens = geometricSum(acceptance, length + 1);
    const verificationMs =
      target.tpotMs *
      (1 + Math.max(0, profile.verificationScalePerExtraToken) * Math.max(0, length - 1));
    const cycleMs = profile.draftMsPerToken * length + verificationMs;
    const tpotMs = cycleMs / expectedTokens;
    if (tpotMs < bestTpot) {
      bestTpot = tpotMs;
      bestLength = length;
    }
  }
  if (bestLength === 0) {
    return {
      ...target,
      architecture: "speculative-pipeline",
      detail: "speculation rejected: no predicted speedup",
    };
  }
  const ratio = bestTpot / target.tpotMs;
  return {
    ...target,
    architecture: "speculative-pipeline",
    tpotMs: bestTpot,
    ttftMs: target.ttftMs + profile.draftMsPerToken,
    tokensPerSecondPerSequence: 1_000 / bestTpot,
    aggregateTokensPerSecond: target.aggregateTokensPerSecond / ratio,
    detail: `draft=${bestLength}, acceptance=${acceptance.toFixed(2)}; exploratory upper-bound`,
  };
}

function geometricSum(ratio: number, terms: number): number {
  if (Math.abs(1 - ratio) < 1e-9) return terms;
  return (1 - ratio ** terms) / (1 - ratio);
}

function infeasible(
  architecture: ParallelismKind,
  nodes: number,
  reason: string | null,
): ParallelismEstimate {
  return {
    architecture,
    feasible: false,
    reason: reason ?? "unknown",
    nodes,
    tpotMs: Number.POSITIVE_INFINITY,
    ttftMs: Number.POSITIVE_INFINITY,
    tokensPerSecondPerSequence: 0,
    aggregateTokensPerSecond: 0,
    networkBytesPerOutputToken: Number.POSITIVE_INFINITY,
    detail: reason ?? "infeasible",
  };
}
