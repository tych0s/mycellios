export interface WanModelSpec {
  id: string;
  modelWeightGb: number;
  activeWeightGbPerToken: number;
  hiddenSize: number;
  hiddenBytes: number;
  layers: number;
  kvHeads: number;
  headDim: number;
  kvBytes: number;
  prefillToDecodeComputeRatio: number;
}

export interface WanRouteProfile {
  id: string;
  label: string;
  /** Physical contributors inside all cells on the route. */
  physicalNodes?: number;
  /** WAN-visible virtual stages; several nearby low-memory nodes may form one. */
  stages: number;
  offeredMemoryGb: number;
  effectiveModelReadGbPerSecond: number;
  linkMbps: number;
  oneWayLatencyMs: number;
  p95NetworkMultiplier: number;
  pipelineEfficiency: number;
  acceptedTokensPerRound: number;
  /** Local drafter cost. Omitted values use a conservative 2 ms/token. */
  draftMsPerToken?: number;
  controlPlaneMs: number;
  activePowerW: number;
  plannedConcurrentChats: number;
  nodeRequestSurvival: number;
}

export interface WanSimulationOptions {
  promptTokens: number;
  outputTokens: number;
  contextTokensPerChat: number;
  activationCompressionRatio: number;
  adaptiveActivationCodec: boolean;
  codecThroughputMbps: number;
  codecFixedMs: number;
  useSpeculativeDecoding: boolean;
  useChunkedPrefill: boolean;
  electricityEuroPerKwh: number;
  interactiveTargetTps: number;
  capacityHeadroom: number;
}

export interface WanRouteEstimate {
  provenance: typeof WAN_PROJECTION_KIND;
  id: string;
  label: string;
  physicalNodes: number;
  virtualStages: number;
  /** Backwards-compatible alias for virtualStages. */
  stages: number;
  modelFits: boolean;
  activationBytesPerToken: number;
  compressedActivationBytesPerToken: number;
  activationCodec: {
    decodeApplied: boolean;
    prefillApplied: boolean;
  };
  speculation: {
    requested: boolean;
    appliedP50: boolean;
    appliedSlowPathP95: boolean;
    autoregressiveTokensPerSecond: number;
    speculativeTokensPerSecond: number;
    predictedSpeedup: number;
  };
  ttftMs: { p50: number; p95: number };
  // slowPathP95 is the P5 rate: throughput observed under P95 latency/network
  // conditions. A larger percentile of tok/s would accidentally select fast paths.
  decodeTokensPerSecond: { p50: number; slowPathP95: number };
  responseSeconds: { p50: number; p95: number };
  sixTurnChatMinutes: { p50: number; p95: number };
  loaded: {
    plannedConcurrentChats: number;
    aggregateTokensPerSecond: number;
    tokensPerSecondPerChat: number;
    recommendedInteractiveChats: number;
    kvCapacityChats: number;
  };
  reliability: {
    requestSuccessPct: number;
    withHotStageSparePct: number;
  };
  economics: {
    activePowerKw: number;
    electricityEuroPerTypicalResponse: number;
    electricityEuroPerMillionOutputTokensAtSaturation: number;
  };
}

export const WAN_PROJECTION_KIND =
  "theoretical_hierarchical_wan_projection_v2_not_benchmark" as const;

export interface FleetEstimate {
  nodes: number;
  expectedOnlineNodes: number;
  usableMemoryTb: number;
  interactiveRegionalRoutes: number;
  fourGbBatchRoutes: number;
  interactiveGlmChatsAtTarget: number;
  allGlmRoutes: number;
}

export const GLM_45_AIR_Q4: WanModelSpec = {
  id: "glm-4.5-air-q4-planning",
  // Planning value includes quantized weights, scales and runtime structures.
  modelWeightGb: 73,
  // 12B active parameters at four bits is 6 GB. The extra 2 GB represents
  // attention, dense blocks, quantization metadata and imperfect kernels.
  activeWeightGbPerToken: 8.26,
  hiddenSize: 4_096,
  hiddenBytes: 2,
  layers: 46,
  kvHeads: 8,
  headDim: 128,
  kvBytes: 2,
  // Prefill matrix operations use accelerators much better than batch-1 decode.
  prefillToDecodeComputeRatio: 30,
};

export const DEFAULT_WAN_OPTIONS: WanSimulationOptions = {
  promptTokens: 800,
  outputTokens: 200,
  contextTokensPerChat: 4_000,
  // BloomBee reports 416.4 -> 312.6 KiB for its measured lossless case (~0.75x).
  activationCompressionRatio: 0.75,
  // The codec is only used when transmission saved is greater than its own
  // encode/decode time. The default throughput approximates BloomBee's
  // byte-split + ZSTD result and is deliberately easy to recalibrate.
  adaptiveActivationCodec: true,
  codecThroughputMbps: 2_000,
  codecFixedMs: 0.1,
  useSpeculativeDecoding: true,
  useChunkedPrefill: true,
  electricityEuroPerKwh: 0.2896,
  interactiveTargetTps: 4,
  capacityHeadroom: 0.6,
};

export const DEFAULT_WAN_ROUTES: WanRouteProfile[] = [
  {
    id: "lan-cell",
    label: "LAN cableada, GPU fuertes",
    physicalNodes: 5,
    stages: 4,
    offeredMemoryGb: 96,
    effectiveModelReadGbPerSecond: 80,
    linkMbps: 10_000,
    oneWayLatencyMs: 0.5,
    p95NetworkMultiplier: 1.25,
    pipelineEfficiency: 0.78,
    acceptedTokensPerRound: 3.2,
    controlPlaneMs: 450,
    activePowerW: 1_250,
    plannedConcurrentChats: 8,
    nodeRequestSurvival: 0.9995,
  },
  {
    id: "metro-fiber",
    label: "Misma ciudad, fibra",
    physicalNodes: 6,
    stages: 4,
    offeredMemoryGb: 90,
    effectiveModelReadGbPerSecond: 65,
    linkMbps: 1_000,
    oneWayLatencyMs: 3,
    p95NetworkMultiplier: 1.4,
    pipelineEfficiency: 0.74,
    acceptedTokensPerRound: 2.8,
    controlPlaneMs: 700,
    activePowerW: 1_200,
    plannedConcurrentChats: 8,
    nodeRequestSurvival: 0.999,
  },
  {
    id: "regional-fiber",
    label: "Same region, residential fiber",
    physicalNodes: 8,
    stages: 6,
    offeredMemoryGb: 88,
    effectiveModelReadGbPerSecond: 50,
    linkMbps: 300,
    oneWayLatencyMs: 8,
    p95NetworkMultiplier: 1.5,
    pipelineEfficiency: 0.72,
    acceptedTokensPerRound: 2.5,
    controlPlaneMs: 1_200,
    activePowerW: 1_200,
    plannedConcurrentChats: 8,
    nodeRequestSurvival: 0.998,
  },
  {
    id: "regional-mixed",
    label: "Mixed region, residential links",
    physicalNodes: 12,
    stages: 6,
    offeredMemoryGb: 84,
    effectiveModelReadGbPerSecond: 32,
    linkMbps: 100,
    oneWayLatencyMs: 15,
    p95NetworkMultiplier: 1.6,
    pipelineEfficiency: 0.65,
    acceptedTokensPerRound: 2.1,
    controlPlaneMs: 1_800,
    activePowerW: 1_200,
    plannedConcurrentChats: 8,
    nodeRequestSurvival: 0.997,
  },
  {
    id: "four-gb-long",
    label: "24 nodos de 4 GB agrupados en 6 celdas",
    physicalNodes: 24,
    stages: 6,
    offeredMemoryGb: 96,
    effectiveModelReadGbPerSecond: 18,
    linkMbps: 50,
    oneWayLatencyMs: 25,
    p95NetworkMultiplier: 1.8,
    pipelineEfficiency: 0.58,
    acceptedTokensPerRound: 1.7,
    controlPlaneMs: 3_000,
    activePowerW: 1_440,
    plannedConcurrentChats: 16,
    nodeRequestSurvival: 0.995,
  },
  {
    id: "intercontinental",
    label: "Intercontinental route",
    physicalNodes: 16,
    stages: 8,
    offeredMemoryGb: 90,
    effectiveModelReadGbPerSecond: 28,
    linkMbps: 50,
    oneWayLatencyMs: 45,
    p95NetworkMultiplier: 1.8,
    pipelineEfficiency: 0.58,
    acceptedTokensPerRound: 1.6,
    controlPlaneMs: 3_500,
    activePowerW: 1_400,
    plannedConcurrentChats: 12,
    nodeRequestSurvival: 0.995,
  },
];

export function estimateWanRoute(
  route: WanRouteProfile,
  options: WanSimulationOptions = DEFAULT_WAN_OPTIONS,
  model: WanModelSpec = GLM_45_AIR_Q4,
): WanRouteEstimate {
  validateInputs(route, options, model);
  const hops = Math.max(0, route.stages - 1);
  const rawActivationBytes = model.hiddenSize * model.hiddenBytes;
  const compressedActivationBytes = Math.ceil(
    rawActivationBytes * options.activationCompressionRatio,
  );
  const computeMs =
    (model.activeWeightGbPerToken / route.effectiveModelReadGbPerSecond) * 1_000;
  const propagationMs = hops * route.oneWayLatencyMs;
  const autoregressiveWire = chooseActivationWire(
    rawActivationBytes,
    route.linkMbps,
    options,
  );
  const autoregressiveRoundP50Ms =
    computeMs +
    propagationMs +
    hops * (autoregressiveWire.serializationMs + autoregressiveWire.codecMs);
  const autoregressiveRoundP95Ms =
    computeMs * 1.2 +
    propagationMs * route.p95NetworkMultiplier +
    hops *
      (autoregressiveWire.serializationMs + autoregressiveWire.codecMs) *
      1.25;
  const autoregressiveP50 = 1_000 / autoregressiveRoundP50Ms;
  const autoregressiveP95 = 1_000 / autoregressiveRoundP95Ms;

  const projectedAccepted = Math.max(1, route.acceptedTokensPerRound);
  const verificationMultiplier = 1 + 0.2 * (projectedAccepted - 1);
  const candidates = Math.max(1, Math.ceil(projectedAccepted));
  const speculativeWire = chooseActivationWire(
    rawActivationBytes * candidates,
    route.linkMbps,
    options,
  );
  const draftMs = (route.draftMsPerToken ?? 2) * Math.max(1, candidates - 1);
  const speculativeRoundP50Ms =
    draftMs +
    computeMs * verificationMultiplier +
    propagationMs +
    hops * (speculativeWire.serializationMs + speculativeWire.codecMs);
  const acceptedP95 = Math.max(1, projectedAccepted * 0.8);
  const speculativeRoundP95Ms =
    draftMs * 1.25 +
    computeMs * verificationMultiplier * 1.2 +
    propagationMs * route.p95NetworkMultiplier +
    hops * (speculativeWire.serializationMs + speculativeWire.codecMs) * 1.25;
  const speculativeP50 = projectedAccepted / (speculativeRoundP50Ms / 1_000);
  const speculativeP95 = acceptedP95 / (speculativeRoundP95Ms / 1_000);
  const applySpeculationP50 =
    options.useSpeculativeDecoding && speculativeP50 > autoregressiveP50;
  const applySpeculationP95 =
    options.useSpeculativeDecoding && speculativeP95 > autoregressiveP95;
  const decodeP50 = applySpeculationP50 ? speculativeP50 : autoregressiveP50;
  const decodeP95 = applySpeculationP95 ? speculativeP95 : autoregressiveP95;
  const roundP50Ms = applySpeculationP50
    ? speculativeRoundP50Ms
    : autoregressiveRoundP50Ms;
  const roundP95Ms = applySpeculationP95
    ? speculativeRoundP95Ms
    : autoregressiveRoundP95Ms;
  const tokensPerRoundP50 = applySpeculationP50 ? projectedAccepted : 1;
  const selectedVerificationMultiplier = applySpeculationP50
    ? verificationMultiplier
    : 1;
  const decodeWire = applySpeculationP50 ? speculativeWire : autoregressiveWire;

  const ttftP50 = estimateTtft(
    route,
    options.promptTokens,
    computeMs,
    rawActivationBytes,
    options,
    model,
    false,
    roundP50Ms,
  );
  const ttftP95 = estimateTtft(
    { ...route, oneWayLatencyMs: route.oneWayLatencyMs * route.p95NetworkMultiplier },
    options.promptTokens,
    computeMs * 1.2,
    rawActivationBytes,
    options,
    model,
    false,
    roundP95Ms,
  );
  const responseP50 = ttftP50 / 1_000 + options.outputTokens / decodeP50;
  const responseP95 = ttftP95 / 1_000 + options.outputTokens / decodeP95;

  const stageVerificationMs =
    (computeMs * selectedVerificationMultiplier) / route.stages;
  const stageLinkMs = decodeWire.serializationMs + decodeWire.codecMs;
  const aggregateTps =
    (tokensPerRoundP50 / (Math.max(stageVerificationMs, stageLinkMs) / 1_000)) *
    route.pipelineEfficiency;
  const perChatLoadedTps = Math.min(
    decodeP50,
    aggregateTps / route.plannedConcurrentChats,
  );
  const kvBytesPerToken =
    model.layers * 2 * model.kvHeads * model.headDim * model.kvBytes;
  const spareBytes = Math.max(0, route.offeredMemoryGb - model.modelWeightGb) * 1_000_000_000;
  const kvCapacityChats = Math.floor(
    spareBytes / (options.contextTokensPerChat * kvBytesPerToken),
  );
  const p95SupportsInteractive = decodeP95 >= options.interactiveTargetTps;
  const recommendedInteractiveChats = p95SupportsInteractive
    ? Math.max(
        1,
        Math.min(
          kvCapacityChats,
          route.plannedConcurrentChats,
          Math.floor(
            (aggregateTps * options.capacityHeadroom) / options.interactiveTargetTps,
          ),
        ),
      )
    : 0;

  const sixTurnP50 = estimateSixTurnChat(
    route,
    computeMs,
    rawActivationBytes,
    options,
    model,
    decodeP50,
    roundP50Ms,
    false,
  );
  const sixTurnP95 = estimateSixTurnChat(
    { ...route, oneWayLatencyMs: route.oneWayLatencyMs * route.p95NetworkMultiplier },
    computeMs * 1.2,
    rawActivationBytes,
    options,
    model,
    decodeP95,
    roundP95Ms,
    true,
  );
  const physicalNodes = route.physicalNodes ?? route.stages;
  const requestSuccess = route.nodeRequestSurvival ** physicalNodes;
  const redundantStageSuccess = 1 - (1 - route.nodeRequestSurvival) ** 2;
  const hotSpareSuccess = redundantStageSuccess ** route.stages;
  const activePowerKw = route.activePowerW / 1_000;
  const electricityPerResponse =
    activePowerKw * options.electricityEuroPerKwh * (responseP50 / 3_600);
  const electricityPerMillion =
    (activePowerKw * options.electricityEuroPerKwh * 1_000_000) /
    (aggregateTps * 3_600);

  return {
    provenance: WAN_PROJECTION_KIND,
    id: route.id,
    label: route.label,
    physicalNodes,
    virtualStages: route.stages,
    stages: route.stages,
    modelFits: route.offeredMemoryGb >= model.modelWeightGb,
    activationBytesPerToken: rawActivationBytes,
    compressedActivationBytesPerToken: compressedActivationBytes,
    activationCodec: {
      decodeApplied: decodeWire.compressed,
      prefillApplied: chooseActivationWire(
        rawActivationBytes * options.promptTokens,
        route.linkMbps,
        options,
      ).compressed,
    },
    speculation: {
      requested: options.useSpeculativeDecoding,
      appliedP50: applySpeculationP50,
      appliedSlowPathP95: applySpeculationP95,
      autoregressiveTokensPerSecond: round(autoregressiveP50, 2),
      speculativeTokensPerSecond: round(speculativeP50, 2),
      predictedSpeedup: round(speculativeP50 / autoregressiveP50, 3),
    },
    ttftMs: { p50: round(ttftP50, 0), p95: round(ttftP95, 0) },
    decodeTokensPerSecond: {
      p50: round(decodeP50, 2),
      slowPathP95: round(decodeP95, 2),
    },
    responseSeconds: { p50: round(responseP50, 1), p95: round(responseP95, 1) },
    sixTurnChatMinutes: { p50: round(sixTurnP50 / 60, 1), p95: round(sixTurnP95 / 60, 1) },
    loaded: {
      plannedConcurrentChats: route.plannedConcurrentChats,
      aggregateTokensPerSecond: round(aggregateTps, 1),
      tokensPerSecondPerChat: round(perChatLoadedTps, 2),
      recommendedInteractiveChats,
      kvCapacityChats,
    },
    reliability: {
      requestSuccessPct: round(requestSuccess * 100, 2),
      withHotStageSparePct: round(hotSpareSuccess * 100, 2),
    },
    economics: {
      activePowerKw: round(activePowerKw, 2),
      electricityEuroPerTypicalResponse: round(electricityPerResponse, 4),
      electricityEuroPerMillionOutputTokensAtSaturation: round(electricityPerMillion, 2),
    },
  };
}

export function estimateDefaultWanRoutes(
  options: WanSimulationOptions = DEFAULT_WAN_OPTIONS,
): WanRouteEstimate[] {
  return DEFAULT_WAN_ROUTES.map((route) => estimateWanRoute(route, options));
}

export function estimateFleet(nodes: number, route: WanRouteEstimate): FleetEstimate {
  if (!Number.isSafeInteger(nodes) || nodes <= 0) throw new Error("nodes must be positive");
  const expectedOnlineNodes = Math.floor(nodes * 0.7);
  // Expected offered inventory: 50% 4 GB, 25% 8 GB, 15% 12 GB,
  // 7.5% 16 GB and 2.5% 24 GB => 7.6 GB/node on average.
  const totalOnlineMemoryGb = nodes * 7.6 * 0.7;
  const usableMemoryGb = totalOnlineMemoryGb * 0.75 * 0.7;
  const interactiveMemoryGb = nodes * 5.6 * 0.7 * 0.75 * 0.7 * 0.88;
  const fourGbMemoryGb = nodes * 2 * 0.7 * 0.75 * 0.7 * 0.75;
  const interactiveRegionalRoutes = Math.floor(
    interactiveMemoryGb / GLM_45_AIR_Q4.modelWeightGb,
  );
  const fourGbBatchRoutes = Math.floor(fourGbMemoryGb / GLM_45_AIR_Q4.modelWeightGb);
  return {
    nodes,
    expectedOnlineNodes,
    usableMemoryTb: round(usableMemoryGb / 1_000, 2),
    interactiveRegionalRoutes,
    fourGbBatchRoutes,
    interactiveGlmChatsAtTarget:
      interactiveRegionalRoutes * route.loaded.recommendedInteractiveChats,
    allGlmRoutes: interactiveRegionalRoutes + fourGbBatchRoutes,
  };
}

function estimateTtft(
  route: WanRouteProfile,
  promptTokens: number,
  computeMs: number,
  rawActivationBytes: number,
  options: WanSimulationOptions,
  model: WanModelSpec,
  warmAffinity: boolean,
  firstDecodeRoundMs: number,
): number {
  const hops = Math.max(0, route.stages - 1);
  const computeOnlyDecodeTps = 1_000 / computeMs;
  const sequentialPrefillMs =
    (promptTokens / (computeOnlyDecodeTps * model.prefillToDecodeComputeRatio)) * 1_000;
  const chunks = options.useChunkedPrefill
    ? Math.max(1, Math.min(route.stages, Math.ceil(promptTokens / 128)))
    : 1;
  const pipelineFactor = options.useChunkedPrefill
    ? (route.stages + chunks - 1) / (route.stages * chunks)
    : 1;
  const prefillComputeMs = sequentialPrefillMs * pipelineFactor;
  const prefillWire = chooseActivationWire(
    rawActivationBytes * promptTokens,
    route.linkMbps,
    options,
  );
  const prefillNetworkMs =
    hops *
    (route.oneWayLatencyMs + prefillWire.serializationMs + prefillWire.codecMs);
  const controlMs = warmAffinity ? route.controlPlaneMs * 0.2 : route.controlPlaneMs;
  return controlMs + prefillComputeMs + prefillNetworkMs + firstDecodeRoundMs;
}

function estimateSixTurnChat(
  route: WanRouteProfile,
  computeMs: number,
  activationBytes: number,
  options: WanSimulationOptions,
  model: WanModelSpec,
  decodeTps: number,
  firstDecodeRoundMs: number,
  p95: boolean,
): number {
  const newPromptTokens = [300, 220, 260, 320, 360, 400];
  const outputTokens = [160, 180, 220, 240, 260, 300];
  return newPromptTokens.reduce((seconds, promptTokens, index) => {
    const ttft = estimateTtft(
      route,
      promptTokens,
      computeMs,
      activationBytes,
      options,
      model,
      index > 0,
      firstDecodeRoundMs,
    );
    const queueMultiplier = p95 ? 1.1 : 1;
    return seconds + (ttft / 1_000 + outputTokens[index]! / decodeTps) * queueMultiplier;
  }, 0);
}

function transferMs(bytes: number, megabitsPerSecond: number): number {
  return (bytes * 8) / (megabitsPerSecond * 1_000);
}

function chooseActivationWire(
  rawBytes: number,
  linkMbps: number,
  options: WanSimulationOptions,
): { compressed: boolean; serializationMs: number; codecMs: number } {
  const rawSerializationMs = transferMs(rawBytes, linkMbps);
  if (
    !options.adaptiveActivationCodec ||
    options.activationCompressionRatio >= 1
  ) {
    return { compressed: false, serializationMs: rawSerializationMs, codecMs: 0 };
  }
  const compressedBytes = Math.ceil(rawBytes * options.activationCompressionRatio);
  const compressedSerializationMs = transferMs(compressedBytes, linkMbps);
  const codecMs =
    options.codecFixedMs + transferMs(rawBytes, options.codecThroughputMbps);
  if (rawSerializationMs - compressedSerializationMs <= codecMs) {
    return { compressed: false, serializationMs: rawSerializationMs, codecMs: 0 };
  }
  return { compressed: true, serializationMs: compressedSerializationMs, codecMs };
}

function validateInputs(
  route: WanRouteProfile,
  options: WanSimulationOptions,
  model: WanModelSpec,
): void {
  const physicalNodes = route.physicalNodes ?? route.stages;
  if (!Number.isSafeInteger(route.stages) || route.stages < 1 || route.stages > 8) {
    throw new Error("stages must be an integer between 1 and 8 virtual stages");
  }
  if (
    !Number.isSafeInteger(physicalNodes) ||
    physicalNodes < route.stages
  ) {
    throw new Error("physicalNodes must be an integer at least as large as stages");
  }
  if (
    route.draftMsPerToken !== undefined &&
    (!Number.isFinite(route.draftMsPerToken) || route.draftMsPerToken < 0)
  ) {
    throw new Error("draftMsPerToken must be finite and non-negative");
  }
  for (const [label, value] of Object.entries({
    offeredMemoryGb: route.offeredMemoryGb,
    effectiveModelReadGbPerSecond: route.effectiveModelReadGbPerSecond,
    linkMbps: route.linkMbps,
    promptTokens: options.promptTokens,
    outputTokens: options.outputTokens,
    contextTokensPerChat: options.contextTokensPerChat,
    activeWeightGbPerToken: model.activeWeightGbPerToken,
    acceptedTokensPerRound: route.acceptedTokensPerRound,
    codecThroughputMbps: options.codecThroughputMbps,
    codecFixedMs: options.codecFixedMs,
  })) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`${label} must be positive`);
  }
  if (route.pipelineEfficiency <= 0 || route.pipelineEfficiency > 1) {
    throw new Error("pipelineEfficiency must be in (0, 1]");
  }
  if (options.activationCompressionRatio <= 0 || options.activationCompressionRatio > 1) {
    throw new Error("activationCompressionRatio must be in (0, 1]");
  }
}

function round(value: number, decimals: number): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}
