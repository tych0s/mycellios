import { activationCodec } from "./codecs.js";
import type {
  ComputeNodeProfile,
  DirectedLinkProfile,
  DistributedModelProfile,
  DistributionMetrics,
  DistributionPlan,
  DistributionTopology,
  DistributionWorkload,
  SearchOptions,
  StagePlacement,
} from "./types.js";

const BYTES_PER_MEGABIT = 125_000;
const MILLISECONDS_PER_SECOND = 1_000;
const MILLISECONDS_PER_HOUR = 3_600_000;

export function evaluateDistributionPlan(
  model: DistributedModelProfile,
  topology: DistributionTopology,
  workload: DistributionWorkload,
  plan: DistributionPlan,
): DistributionMetrics {
  const codec = activationCodec(plan.codec);
  const empty = emptyMetrics(codec.estimatedQualityLoss);
  if (codec.estimatedQualityLoss > workload.maxQualityLoss) {
    return { ...empty, infeasibleReason: "activation_quality_budget_exceeded" };
  }
  if (plan.stages.length < 1 || plan.stages.length > workload.maxStages) {
    return { ...empty, infeasibleReason: "invalid_stage_count" };
  }
  if (!validContiguousCoverage(model, plan.stages)) {
    return { ...empty, infeasibleReason: "layers_are_not_covered_exactly_once" };
  }
  if (new Set(plan.stages.map((stage) => stage.nodeId)).size !== plan.stages.length) {
    return { ...empty, infeasibleReason: "node_reused_in_route" };
  }
  const nodeById = new Map(topology.nodes.map((node) => [node.id, node]));
  const nodes: ComputeNodeProfile[] = [];
  for (const stage of plan.stages) {
    const node = nodeById.get(stage.nodeId);
    if (!node) return { ...empty, infeasibleReason: `unknown_node:${stage.nodeId}` };
    nodes.push(node);
  }

  const batchSize = Math.max(
    1,
    Math.min(plan.microBatchSize, workload.concurrentSequences),
  );
  const prefillChunk = Math.max(1, Math.min(plan.prefillChunkTokens, workload.promptTokens));
  const decodeComputes: number[] = [];
  const decodeOutgoing: number[] = [];
  const prefillComputes: number[] = [];
  const prefillOutgoing: number[] = [];
  const memoryByStage: number[] = [];
  let routeAvailability = 1;
  let networkBytesPerOutputToken = 0;

  for (let index = 0; index < plan.stages.length; index += 1) {
    const stage = plan.stages[index]!;
    const node = nodes[index]!;
    const first = index === 0;
    const last = index === plan.stages.length - 1;
    if (!nodeSupportsStagePosition(node, first, last)) {
      return { ...empty, infeasibleReason: `engine_stage_role_unsupported:${node.id}` };
    }
    if (
      node.maxStageLayers !== undefined
      && stage.layerEnd - stage.layerStart > node.maxStageLayers
    ) {
      return { ...empty, infeasibleReason: `engine_layer_capacity_exceeded:${node.id}` };
    }
    const memoryBytes = stageMemoryBytes(model, stage, workload, first, last);
    const memoryLimitBytes = Math.max(0, node.memoryBytes - node.reserveBytes);
    if (memoryBytes > memoryLimitBytes) {
      return {
        ...empty,
        infeasibleReason: `memory_exceeded:${node.id}`,
        peakStageMemoryBytes: Math.max(...memoryByStage, memoryBytes),
      };
    }
    memoryByStage.push(memoryBytes);
    routeAvailability *= clampProbability(node.availability);

    const speedup = batchSpeedup(node, batchSize);
    const layerDecode = sumLayers(model, stage, (layer) => layer.decodeMsAtUnit);
    const endpointDecode =
      (first ? model.embeddingDecodeMsAtUnit : 0) +
      (last ? model.lmHeadDecodeMsAtUnit : 0);
    const decodeCompute = ((layerDecode + endpointDecode) * node.decodeScale * batchSize) / speedup;
    decodeComputes.push(decodeCompute);

    const layerPrefill = sumLayers(
      model,
      stage,
      (layer) => layer.prefillMsPerTokenAtUnit,
    );
    const endpointPrefill =
      (first ? model.embeddingPrefillMsPerTokenAtUnit : 0) +
      (last ? model.lmHeadPrefillMsPerTokenAtUnit : 0);
    const prefillCompute =
      (layerPrefill + endpointPrefill) * node.prefillScale * prefillChunk;
    prefillComputes.push(prefillCompute);

    if (!last) {
      const nextNode = nodes[index + 1]!;
      const link = directedLink(topology, node.id, nextNode.id);
      if (!link) {
        return { ...empty, infeasibleReason: `missing_link:${node.id}->${nextNode.id}` };
      }
      // TCP/QUIC recover packet loss; do not incorrectly treat every dropped
      // packet as a failed inference. Reachability is a separate measurement.
      routeAvailability *= clampProbability(link.availability ?? 1);
      const activationElements = model.layers[stage.layerEnd - 1]!.activationElements;
      const decodeTransfer = tensorTransferMs(
        activationElements * batchSize,
        codec.bytesPerElement,
        link,
        node,
        nextNode,
        workload.p95,
        codec.encodeGbps,
        codec.decodeGbps,
        codec.fixedEncodeMs,
        codec.fixedDecodeMs,
      );
      const prefillTransfer = tensorTransferMs(
        activationElements * prefillChunk,
        codec.bytesPerElement,
        link,
        node,
        nextNode,
        workload.p95,
        codec.encodeGbps,
        codec.decodeGbps,
        codec.fixedEncodeMs,
        codec.fixedDecodeMs,
      );
      decodeOutgoing.push(decodeTransfer);
      prefillOutgoing.push(prefillTransfer);
      networkBytesPerOutputToken += activationElements * codec.bytesPerElement;
    } else {
      const firstNode = nodes[0]!;
      if (node.id === firstNode.id) {
        decodeOutgoing.push(0);
      } else {
        const returnLink = directedLink(topology, node.id, firstNode.id);
        if (!returnLink) {
          return { ...empty, infeasibleReason: `missing_return_link:${node.id}->${firstNode.id}` };
        }
        routeAvailability *= clampProbability(returnLink.availability ?? 1);
        decodeOutgoing.push(rawTransferMs(4 * batchSize, returnLink, workload.p95));
        networkBytesPerOutputToken += 4;
      }
      prefillOutgoing.push(0);
    }
  }

  if (routeAvailability < workload.minRouteAvailability) {
    return {
      ...empty,
      infeasibleReason: "route_availability_below_minimum",
      routeAvailability,
      peakStageMemoryBytes: Math.max(...memoryByStage),
    };
  }

  const decodeServices = decodeComputes.map(
    (compute, index) => compute + decodeOutgoing[index]!,
  );
  const prefillServices = prefillComputes.map(
    (compute, index) => compute + prefillOutgoing[index]!,
  );
  const pathDecodeMs = sum(decodeServices);
  const pipelineCycleMs = Math.max(...decodeServices);
  const batchesPerRound = Math.ceil(workload.concurrentSequences / batchSize);
  const batchingDelay = batchSize > 1 ? workload.batchWindowMs : 0;
  const tpotMs = Math.max(pathDecodeMs, batchesPerRound * pipelineCycleMs) + batchingDelay;
  const aggregateTokensPerSecond =
    (workload.concurrentSequences * MILLISECONDS_PER_SECOND) / tpotMs;
  const tokensPerSecondPerSequence = MILLISECONDS_PER_SECOND / tpotMs;

  const prefillChunks = Math.ceil(workload.promptTokens / prefillChunk);
  const prefillPipelineMs =
    sum(prefillServices) + Math.max(0, prefillChunks - 1) * Math.max(...prefillServices);
  const tokenReturnMs = decodeOutgoing.at(-1) ?? 0;
  const lastNode = nodes.at(-1)!;
  const firstTokenHeadMs = model.lmHeadDecodeMsAtUnit * lastNode.decodeScale;
  const ttftMs = prefillPipelineMs + firstTokenHeadMs + tokenReturnMs + batchingDelay;
  const responseTimeMs = ttftMs + Math.max(0, workload.outputTokens - 1) * tpotMs;

  const energyPerBatchWh = plan.stages.reduce((total, _stage, index) => {
    const node = nodes[index]!;
    const activeMs = decodeComputes[index]! + decodeOutgoing[index]! * 0.15;
    return total + (node.powerWatts * activeMs) / MILLISECONDS_PER_HOUR;
  }, 0);
  const energyWhPerOutputToken =
    (energyPerBatchWh * batchesPerRound) / workload.concurrentSequences;

  return {
    feasible: true,
    infeasibleReason: null,
    stages: plan.stages.length,
    ttftMs,
    tpotMs,
    tokensPerSecondPerSequence,
    aggregateTokensPerSecond,
    responseTimeMs,
    pipelineCycleMs,
    pathDecodeMs,
    tokenReturnMs,
    routeAvailability,
    energyWhPerOutputToken,
    peakStageMemoryBytes: Math.max(...memoryByStage),
    networkBytesPerOutputToken,
    qualityLoss: codec.estimatedQualityLoss,
    stageMetrics: plan.stages.map((stage, index) => ({
      nodeId: stage.nodeId,
      layerStart: stage.layerStart,
      layerEnd: stage.layerEnd,
      memoryBytes: memoryByStage[index]!,
      memoryLimitBytes: nodes[index]!.memoryBytes - nodes[index]!.reserveBytes,
      decodeBatchComputeMs: decodeComputes[index]!,
      decodeOutgoingMs: decodeOutgoing[index]!,
      prefillChunkComputeMs: prefillComputes[index]!,
      prefillOutgoingMs: prefillOutgoing[index]!,
    })),
  };
}

export function distributionObjective(
  metrics: DistributionMetrics,
  weights: SearchOptions["objectiveWeights"],
): number {
  if (!metrics.feasible) return Number.POSITIVE_INFINITY;
  return (
    weights.tpot * metrics.tpotMs +
    weights.ttft * metrics.ttftMs +
    weights.response * metrics.responseTimeMs +
    weights.energy * metrics.energyWhPerOutputToken * 1_000 +
    weights.unavailability * (1 - metrics.routeAvailability) * 1_000
  );
}

export function stageMemoryBytes(
  model: DistributedModelProfile,
  stage: StagePlacement,
  workload: DistributionWorkload,
  first: boolean,
  last: boolean,
): number {
  const layerWeights = sumLayers(model, stage, (layer) => layer.weightBytes);
  const kvPerToken = sumLayers(model, stage, (layer) => layer.kvBytesPerToken);
  const endpointWeights =
    first && last && model.tiedEmbeddingAndHead
      ? Math.max(model.embeddingBytes, model.lmHeadBytes)
      : (first ? model.embeddingBytes : 0) + (last ? model.lmHeadBytes : 0);
  return (
    model.runtimeOverheadBytesPerStage +
    layerWeights +
    endpointWeights +
    kvPerToken * workload.contextTokens * workload.concurrentSequences
  );
}

function nodeSupportsStagePosition(
  node: ComputeNodeProfile,
  first: boolean,
  last: boolean,
): boolean {
  if (!node.stageRoles) return true;
  if (first && !node.stageRoles.includes("head")) return false;
  if (last && !node.stageRoles.includes("tail")) return false;
  return first || last || node.stageRoles.includes("middle");
}

export function directedLink(
  topology: DistributionTopology,
  from: string,
  to: string,
): DirectedLinkProfile | null {
  if (from === to) {
    return {
      from,
      to,
      oneWayLatencyMs: 0,
      jitterP95Ms: 0,
      bandwidthMbps: Number.POSITIVE_INFINITY,
      lossRate: 0,
      availability: 1,
    };
  }
  return topology.links.find((link) => link.from === from && link.to === to) ?? null;
}

export function rawTransferMs(
  bytes: number,
  link: DirectedLinkProfile,
  p95: boolean,
): number {
  const serializationMs = Number.isFinite(link.bandwidthMbps)
    ? (bytes / (link.bandwidthMbps * BYTES_PER_MEGABIT)) * MILLISECONDS_PER_SECOND
    : 0;
  const loss = Math.min(0.9, clampProbability(link.lossRate));
  const retransmitFactor = 1 / Math.max(0.1, 1 - loss);
  // Established reliable transports pay roughly another RTT when loss causes
  // a retransmission. This remains an approximation, but is materially more
  // realistic on WAN than merely inflating serialization time.
  const expectedRecoveryMs =
    loss > 0 ? (loss / Math.max(0.1, 1 - loss)) * 2 * link.oneWayLatencyMs : 0;
  return (
    link.oneWayLatencyMs +
    (p95 ? link.jitterP95Ms : 0) +
    serializationMs * retransmitFactor +
    expectedRecoveryMs
  );
}

/**
 * Upper bound on link time saved per accepted round by exact distributed
 * speculation, from the DSD cost model (arXiv 2511.11733): with N pipeline
 * stages, one-way link latency t1 and k accepted tokens per round, inter-node
 * communication shrinks by (N-1)*t1*(k-1)/k. Complementary sizing bound for
 * WAN routes; not consumed by any default planner or by
 * evaluateDistributionPlan.
 */
export function speculativeLinkTimeSavedMs(
  stageCount: number,
  linkOneWayLatencyMs: number,
  acceptedTokensPerRound: number,
): number {
  if (!Number.isInteger(stageCount) || stageCount < 2) {
    throw new Error("speculative_stage_count_must_be_an_integer_of_at_least_2");
  }
  if (!Number.isFinite(linkOneWayLatencyMs) || linkOneWayLatencyMs < 0) {
    throw new Error("speculative_link_latency_must_be_finite_and_non_negative");
  }
  if (!Number.isFinite(acceptedTokensPerRound) || acceptedTokensPerRound < 1) {
    throw new Error("speculative_accepted_tokens_must_be_finite_and_at_least_1");
  }
  const saved =
    ((stageCount - 1) * linkOneWayLatencyMs * (acceptedTokensPerRound - 1)) /
    acceptedTokensPerRound;
  // Guard against intermediate overflow to Infinity for extreme (but finite)
  // inputs so the contract always returns a finite bound.
  if (!Number.isFinite(saved)) {
    throw new Error("speculative_link_time_saved_overflowed");
  }
  return saved;
}

function tensorTransferMs(
  elements: number,
  bytesPerElement: number,
  link: DirectedLinkProfile,
  sender: ComputeNodeProfile,
  receiver: ComputeNodeProfile,
  p95: boolean,
  encodeGbps: number,
  decodeGbps: number,
  fixedEncodeMs: number,
  fixedDecodeMs: number,
): number {
  const rawBytes = elements * 2;
  const wireBytes = elements * bytesPerElement;
  const encodeMs = Number.isFinite(encodeGbps)
    ? fixedEncodeMs + (rawBytes * 8) / (encodeGbps * 1_000_000) * sender.codecScale
    : 0;
  const decodeMs = Number.isFinite(decodeGbps)
    ? fixedDecodeMs + (rawBytes * 8) / (decodeGbps * 1_000_000) * receiver.codecScale
    : 0;
  return encodeMs + rawTransferMs(wireBytes, link, p95) + decodeMs;
}

function validContiguousCoverage(
  model: DistributedModelProfile,
  stages: StagePlacement[],
): boolean {
  let nextLayer = 0;
  for (const stage of stages) {
    if (
      stage.layerStart !== nextLayer ||
      stage.layerEnd <= stage.layerStart ||
      stage.layerEnd > model.layers.length
    ) {
      return false;
    }
    nextLayer = stage.layerEnd;
  }
  return nextLayer === model.layers.length;
}

function batchSpeedup(node: ComputeNodeProfile, batchSize: number): number {
  return Math.max(
    1,
    Math.min(node.maxBatchSpeedup, 1 + node.batchGain * Math.max(0, batchSize - 1)),
  );
}

function sumLayers(
  model: DistributedModelProfile,
  stage: StagePlacement,
  value: (layer: DistributedModelProfile["layers"][number]) => number,
): number {
  let result = 0;
  for (let index = stage.layerStart; index < stage.layerEnd; index += 1) {
    result += value(model.layers[index]!);
  }
  return result;
}

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function clampProbability(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function emptyMetrics(qualityLoss: number): DistributionMetrics {
  return {
    feasible: false,
    infeasibleReason: "unknown",
    stages: 0,
    ttftMs: Number.POSITIVE_INFINITY,
    tpotMs: Number.POSITIVE_INFINITY,
    tokensPerSecondPerSequence: 0,
    aggregateTokensPerSecond: 0,
    responseTimeMs: Number.POSITIVE_INFINITY,
    pipelineCycleMs: Number.POSITIVE_INFINITY,
    pathDecodeMs: Number.POSITIVE_INFINITY,
    tokenReturnMs: Number.POSITIVE_INFINITY,
    routeAvailability: 0,
    energyWhPerOutputToken: Number.POSITIVE_INFINITY,
    peakStageMemoryBytes: 0,
    networkBytesPerOutputToken: 0,
    qualityLoss,
    stageMetrics: [],
  };
}
