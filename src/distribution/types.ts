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
  /**
   * Largest checkpoint tensor that remains resident for this layer. Certified
   * MoE profiles exclude routed `.mlp.experts.` tensors from this value.
   */
  largestResidentTensorBytes?: number;
  /**
   * Optional, measured working-set description for a RAM-backed MacroWave.
   * `activeWeightBytesPerWave` is retained as legacy telemetry: without the
   * routing geometry below it is not safe to extrapolate that measurement to a
   * different wave or physical batch, so the planner falls back to the complete
   * routed-expert set. The complete `weightBytes` allocation must still fit RAM.
   */
  macroWave?: {
    /**
     * Legacy measured working set. New profiles may omit it when the exact
     * expertCount/expertsPerToken geometry is present, because the planner can
     * then derive the sealed-wave union for the selected wave and batch.
     */
    activeWeightBytesPerWave?: number;
    largestTransferUnitBytes: number;
    /**
     * Certified peak tensor temporaries for one routed position while the
     * serial expert executor processes one expert. The planner multiplies it
     * by the sealed wave/batch width; CUDA-library scratch remains covered by
     * runtimeOverheadBytesPerStage until physically calibrated.
     */
    expertWorkspaceBytesPerPosition?: number;
  };
  /**
   * Measured MoE weight split used when an execution cell is already expert
   * parallel. Only routed experts belong in `expertWeightBytes`; attention,
   * routers and shared experts remain in the replicated remainder
   * (`weightBytes - expertWeightBytes`). Missing telemetry never implies that
   * the whole layer can be sharded.
   */
  expertParallel?: {
    expertWeightBytes: number;
    /** Number of routed experts in this layer (shared experts are excluded). */
    expertCount?: number;
    /** Exact router top-k, normally named num_experts_per_tok by HF models. */
    expertsPerToken?: number;
  };
}

export interface DistributedModelProfile {
  id: string;
  layers: LayerProfile[];
  embeddingBytes: number;
  lmHeadBytes: number;
  /** Header-derived one-at-a-time loader transient for the first endpoint. */
  largestEmbeddingTensorBytes?: number;
  /** Header-derived one-at-a-time loader transient for the final endpoint. */
  largestLmHeadTensorBytes?: number;
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
  /**
   * Explicit host-memory hierarchy used only by the MacroWave planner.
   * Existing planners continue to interpret memoryBytes/reserveBytes exactly
   * as before, so adding this profile is backwards compatible.
   */
  ramVram?: {
    usableRamBytes: number;
    usableVramBytes: number;
    /** Sustained host-memory bandwidth in decimal gigabytes per second. */
    ramBandwidthGBps: number;
    /** Sustained host-to-device bandwidth in decimal gigabytes per second. */
    pcieBandwidthGBps: number;
    /** A resident stage can represent complete layers or a measured EP cell. */
    residentKind?: "layers" | "expert-shard";
    /** Required when residentKind is expert-shard; values describe the EP cell. */
    expertShard?: {
      worldSize: number;
      /** Maximum routed-expert fraction resident on any rank in the cell. */
      fraction: number;
    };
  };
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
  /**
   * Fresh physical probe evidence for this exact directed path. Analytical or
   * region-default links deliberately omit it and can never admit a
   * tensor-parallel collective.
   */
  evidence?: {
    source: "runtime-probe";
    measuredAt: number;
    validUntil: number;
    successfulSamples: number;
    failedSamples: number;
  };
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
  /**
   * Optional executable memory contract emitted only by the opt-in MacroWave
   * planner. Keeping it on the placement makes the contract survive every
   * plan/manifest/launcher transformation instead of relying on array order in
   * an out-of-band report.
   */
  macroWave?: MacroWaveStageExecutionContractV1;
}

export type MacroWaveMemoryMode = "resident" | "ram-backed";
export type MacroWaveResidentKind = "layers" | "expert-shard";

/**
 * Loader policy for a RAM-backed sparse stage.  This is deliberately an
 * artifact *requirement*, not a host path: paths are machine-local launch
 * configuration and must never enter the portable pipeline manifest.
 */
export interface MacroWaveRamArtifactRequirementV1 {
  schema: "gdlp-local-safetensors-moe-stage/1";
  format: "safetensors";
  locality: "host-local-only";
  loader: "selective-safetensors-ram-backed-moe";
  weightEncoding: "floating-safetensors";
  sourceDtypes: ["fp16", "bf16", "fp32"];
  adapterIds: [
    "transformers-qwen3-moe-v1",
    "transformers-glm4-moe-v1",
  ];
  /** Exact expert decisions are executed one at a time, never as a VRAM sum. */
  expertExecutionMode: "serial-exact";
  /** Largest independently transferred expert/tile in this stage. */
  largestExpertBytes: number;
  /** One execution buffer plus one prefetch buffer (ping-pong). */
  weightBufferCopies: 2;
  /** The execution path must not instantiate the complete HF model. */
  fullModelMaterialization: false;
}

export interface MacroWaveStageExecutionContractV1 {
  mode: "macro-wave-memory";
  schema: "gdlp-macro-wave-stage/1";
  memoryMode: MacroWaveMemoryMode;
  residentKind: MacroWaveResidentKind;
  budgets: {
    hostRamBytes: number;
    vramBytes: number;
  };
  requirements: {
    fullStageStateBytes: number;
    hostRamBytes: number;
    vramBytes: number;
    fixedVramBytes: number;
    /** Immutable attention/router/shared/dense/endpoints kept on device. */
    residentParameterBudgetBytes: number;
    /** Largest one-at-a-time checkpoint tensor held while filling device state. */
    residentStreamingTransientBytes: number;
    /** Two bounded pinned host slots when staged CUDA copies are enabled. */
    boundedPinnedStagingReserveBytes: number;
    /** Routed experts + bounded pinned reserve + one streaming tensor. */
    hostRamPeakUpperBoundBytes: number;
    activationBufferBytes: number;
    weightBufferBytes: number;
    weightBufferCopies: number;
  };
  workingSet: {
    totalWeightBytes: number;
    residentWeightBytes: number;
    /** Complete authoritative routed-expert set retained in host RAM. */
    totalRoutedExpertBytes: number;
    activeWeightBytesPerWave: number;
    largestTransferUnitBytes: number;
    largestExpertBytes: number;
  };
  cachePolicy: {
    kind: "full-resident" | "disabled" | "bounded-lru";
    capacityBytes: number;
    expectedHitRate: number;
    expectedMissWeightBytesPerWave: number;
  };
  /** Required for ram-backed stages and forbidden for resident stages. */
  ramArtifact?: MacroWaveRamArtifactRequirementV1;
}

export interface MacroWaveRouteProjectionV1 {
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
  routeAvailability: number;
}

export interface MacroWavePlanContractV1 {
  schema: "gdlp-macro-wave-plan/1";
  routeKind: "resident-baseline" | "macro-wave";
  waveTokens: number;
  expectedCommittedTokensPerWave: number;
  projection: MacroWaveRouteProjectionV1;
}

export interface DistributionPlan {
  algorithm: string;
  codec: ActivationCodecId;
  microBatchSize: number;
  prefillChunkTokens: number;
  stages: StagePlacement[];
  /** Present only when MacroWave was selected explicitly. */
  macroWave?: MacroWavePlanContractV1;
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
