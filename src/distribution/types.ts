export type ActivationCodecId =
  | "fp16"
  | "int8"
  | "int8-grouped"
  | "int8-hadamard"
  | "q4";

export interface ActivationCodec {
  id: ActivationCodecId;
  bytesPerElement: number;
  encodeGbps: number;
  decodeGbps: number;
  fixedEncodeMs: number;
  fixedDecodeMs: number;
  estimatedQualityLoss: number;
}

export interface LayerProfile {
  index: number;
  weightBytes: number;
  activationElements: number;
  kvBytesPerToken: number;
  decodeMsAtUnit: number;
  prefillMsPerTokenAtUnit: number;
}

export interface DistributedModelProfile {
  id: string;
  layers: LayerProfile[];
  embeddingBytes: number;
  lmHeadBytes: number;
  /** A single-device runtime can alias these weights; split endpoints cannot. */
  tiedEmbeddingAndHead?: boolean;
  runtimeOverheadBytesPerStage: number;
  embeddingDecodeMsAtUnit: number;
  lmHeadDecodeMsAtUnit: number;
  embeddingPrefillMsPerTokenAtUnit: number;
  lmHeadPrefillMsPerTokenAtUnit: number;
}

export interface ComputeNodeProfile {
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
}

export interface DirectedLinkProfile {
  from: string;
  to: string;
  /** One-way propagation and software latency after a connection is warm. */
  oneWayLatencyMs: number;
  jitterP95Ms: number;
  bandwidthMbps: number;
  /** Packet loss is recovered by the reliable transport and therefore adds latency. */
  lossRate: number;
  /** Probability that the peer-to-peer link itself is reachable for a route. */
  availability?: number;
}

export interface DistributionTopology {
  nodes: ComputeNodeProfile[];
  links: DirectedLinkProfile[];
}

export interface DistributionWorkload {
  promptTokens: number;
  outputTokens: number;
  contextTokens: number;
  concurrentSequences: number;
  maxStages: number;
  maxQualityLoss: number;
  minRouteAvailability: number;
  batchWindowMs: number;
  p95: boolean;
}

export interface StagePlacement {
  nodeId: string;
  layerStart: number;
  layerEnd: number;
}

export interface DistributionPlan {
  algorithm: string;
  codec: ActivationCodecId;
  microBatchSize: number;
  prefillChunkTokens: number;
  stages: StagePlacement[];
}

export interface StageMetrics {
  nodeId: string;
  layerStart: number;
  layerEnd: number;
  memoryBytes: number;
  memoryLimitBytes: number;
  decodeBatchComputeMs: number;
  decodeOutgoingMs: number;
  prefillChunkComputeMs: number;
  prefillOutgoingMs: number;
}

export interface DistributionMetrics {
  feasible: boolean;
  infeasibleReason: string | null;
  /** True when feasibility is useful but latency/throughput needs physical calibration. */
  calibrationRequired?: boolean;
  calibrationReasons?: string[];
  stages: number;
  ttftMs: number;
  tpotMs: number;
  tokensPerSecondPerSequence: number;
  aggregateTokensPerSecond: number;
  responseTimeMs: number;
  pipelineCycleMs: number;
  pathDecodeMs: number;
  tokenReturnMs: number;
  routeAvailability: number;
  energyWhPerOutputToken: number;
  peakStageMemoryBytes: number;
  networkBytesPerOutputToken: number;
  qualityLoss: number;
  stageMetrics: StageMetrics[];
}

export interface EvaluatedDistributionPlan {
  plan: DistributionPlan;
  metrics: DistributionMetrics;
  objective: number;
}

export interface DistributionPlanner {
  readonly id: string;
  plan(
    model: DistributedModelProfile,
    topology: DistributionTopology,
    workload: DistributionWorkload,
  ): DistributionPlan | null;
}

export interface SearchOptions {
  beamWidth: number;
  candidateCodecs: ActivationCodecId[];
  candidateMicroBatchSizes: number[];
  candidatePrefillChunks: number[];
  objectiveWeights: {
    tpot: number;
    ttft: number;
    response: number;
    energy: number;
    unavailability: number;
  };
}
