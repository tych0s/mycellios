/**
 * Reproducible theoretical physics model for pipeline inference over domestic WAN nodes.
 *
 * This is deliberately labelled as a model, not a benchmark. In particular, the
 * reported slots are ideal cyclic-pipeline slots; they are not guaranteed concurrent
 * users and do not include a full KV-memory admission controller.
 */

export const THEORETICAL_PHYSICS_KIND =
  "theoretical_physics_v1_not_benchmark" as const;

export const IDEAL_SLOTS_DISCLAIMER =
  "ideal_cyclic_pipeline_slots_not_guaranteed" as const;

export interface WanModelProfile {
  layers: number;
  hiddenSize: number;
  totalParams: number;
  activeParams: number;
  weightBytes: number;
  kvHeads: number;
  headDim: number;
  kvBytes: number;
  activationBytesPerElement: number;
}

export interface WanOptions {
  nodes: number;
  trials: number;
  seed: number;
  onlineProbability: number;
  hotReserveShare: number;
  promptTokens: number;
  contextTokens: number;
  microchunkTokens: number;
  maxInteractiveStages: number;
  routeSetupMs: number;
}

export interface PercentilePair {
  p50: number;
  p95: number;
}

export interface WanSummary {
  kind: typeof THEORETICAL_PHYSICS_KIND;
  idealSlotsSemantics: typeof IDEAL_SLOTS_DISCLAIMER;
  nodes: number;
  trials: number;
  probabilityAnyRoute: number;
  routes: PercentilePair;
  stages: PercentilePair;
  tokensPerSecond: {
    p50: number;
    /** P5 of throughput: the slow path corresponding to P95 TPOT. */
    slowPathP95: number;
  };
  ttftMs: PercentilePair;
  /** Sum of stage counts across routes: ideal cyclic pipeline slots only. */
  concurrency: PercentilePair;
  incrementalPowerKw: PercentilePair;
  kvBytesPerConversation: number;
}

export interface TierProfile {
  share: number;
  vramGb: number;
  /** Conservative capacity after runtime, KV and a 15% operating margin. */
  layerCapacity: number;
  decodeMsPerLayer: number;
  ppFullEquivalentTps: number;
  powerW: number;
}

export interface RegionProfile {
  name: string;
  share: number;
  /** Median coordinator RTT, later converted to a one-way relay estimate. */
  rttMs: number;
}

export interface WaterFillInput {
  layerCapacity: number;
  decodeMsPerLayer: number;
}

export interface WanSensitivityInput {
  stages: number;
  computeMs: number;
  oneWayHopMs: number;
  bandwidthMbps: number;
  activationBytes?: number;
  protocolOverheadMs?: number;
}

export interface WanSensitivityPoint {
  stages: number;
  computeMs: number;
  oneWayHopMs: number;
  bandwidthMbps: number;
  activationBytes: number;
  tpotMs: number;
  tokensPerSecond: number;
}

export const GLM45_AIR_Q4KM: Readonly<WanModelProfile> = Object.freeze({
  layers: 46,
  hiddenSize: 4_096,
  totalParams: 106e9,
  activeParams: 12e9,
  weightBytes: 73e9,
  kvHeads: 8,
  headDim: 128,
  kvBytes: 2,
  activationBytesPerElement: 1,
});

export const DOMESTIC_TIERS: readonly Readonly<TierProfile>[] = Object.freeze([
  Object.freeze({
    share: 0.5,
    vramGb: 4,
    layerCapacity: 1,
    decodeMsPerLayer: 5,
    ppFullEquivalentTps: 12,
    powerW: 60,
  }),
  Object.freeze({
    share: 0.25,
    vramGb: 8,
    layerCapacity: 3,
    decodeMsPerLayer: 3.5,
    ppFullEquivalentTps: 25,
    powerW: 90,
  }),
  Object.freeze({
    share: 0.15,
    vramGb: 12,
    layerCapacity: 6,
    decodeMsPerLayer: 2.5,
    ppFullEquivalentTps: 50,
    powerW: 120,
  }),
  Object.freeze({
    share: 0.075,
    vramGb: 16,
    layerCapacity: 8,
    decodeMsPerLayer: 2,
    ppFullEquivalentTps: 80,
    powerW: 150,
  }),
  Object.freeze({
    share: 0.025,
    vramGb: 24,
    layerCapacity: 12,
    decodeMsPerLayer: 1.5,
    ppFullEquivalentTps: 130,
    powerW: 200,
  }),
]);

export const DOMESTIC_REGIONS: readonly Readonly<RegionProfile>[] = Object.freeze([
  Object.freeze({ name: "eu-west", share: 0.35, rttMs: 20 }),
  Object.freeze({ name: "eu-central", share: 0.25, rttMs: 22 }),
  Object.freeze({ name: "us-east", share: 0.2, rttMs: 28 }),
  Object.freeze({ name: "us-west", share: 0.1, rttMs: 32 }),
  Object.freeze({ name: "apac", share: 0.1, rttMs: 38 }),
]);

export const DEFAULT_WAN_PHYSICS_OPTIONS: Readonly<WanOptions> = Object.freeze({
  nodes: 1_000,
  trials: 1_000,
  seed: 123_456_789,
  onlineProbability: 0.7,
  hotReserveShare: 0.25,
  promptTokens: 2_000,
  contextTokens: 2_000,
  microchunkTokens: 128,
  maxInteractiveStages: 16,
  routeSetupMs: 300,
});

interface SimulatedNode {
  cap: number;
  decodeMsPerLayer: number;
  ppFullEquivalentTps: number;
  coordinatorRttMs: number;
  uplinkMbps: number;
  powerW: number;
  assignedLayers: number;
}

interface RouteStats {
  stages: number;
  tokensPerSecond: number;
  ttftMs: number;
  powerW: number;
}

/** Mulberry32 with its exact uint32 transition, used for stable snapshots. */
export function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let result = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    result = (result + Math.imul(result ^ (result >>> 7), 61 | result)) ^ result;
    return ((result ^ (result >>> 14)) >>> 0) / 4_294_967_296;
  };
}

/** Box-Muller normal. It intentionally consumes two draws and caches no spare. */
export function standardNormal(rng: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function lognormal(median: number, sigma: number, rng: () => number): number {
  return median * Math.exp(sigma * standardNormal(rng));
}

function weightedPick<T extends { readonly share: number }>(
  values: readonly T[],
  rng: () => number,
): T {
  if (values.length === 0) throw new Error("weightedPick requires at least one value");
  let draw = rng();
  for (const value of values) {
    draw -= value.share;
    if (draw < 0) return value;
  }
  return values[values.length - 1]!;
}

function nearestRank(values: readonly number[], quantile: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil(quantile * sorted.length) - 1),
  );
  return sorted[index]!;
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function assertOptions(options: WanOptions): void {
  const positiveIntegers: Array<[string, number]> = [
    ["nodes", options.nodes],
    ["trials", options.trials],
    ["promptTokens", options.promptTokens],
    ["contextTokens", options.contextTokens],
    ["microchunkTokens", options.microchunkTokens],
    ["maxInteractiveStages", options.maxInteractiveStages],
  ];
  for (const [name, value] of positiveIntegers) {
    if (!Number.isInteger(value) || value <= 0) {
      throw new RangeError(`${name} must be a positive integer`);
    }
  }
  if (options.onlineProbability < 0 || options.onlineProbability > 1) {
    throw new RangeError("onlineProbability must be between zero and one");
  }
  if (options.hotReserveShare < 0 || options.hotReserveShare > 1) {
    throw new RangeError("hotReserveShare must be between zero and one");
  }
  if (!Number.isFinite(options.routeSetupMs) || options.routeSetupMs < 0) {
    throw new RangeError("routeSetupMs must be a non-negative finite number");
  }
}

export function createWanPhysicsOptions(
  overrides: Partial<WanOptions> = {},
): WanOptions {
  return { ...DEFAULT_WAN_PHYSICS_OPTIONS, ...overrides };
}

/**
 * Assigns every model layer while minimizing the next cumulative stage time.
 * Returns null when the available capacities cannot hold all layers.
 */
export function allocateLayersWaterFill(
  nodes: readonly WaterFillInput[],
  layers = GLM45_AIR_Q4KM.layers,
): number[] | null {
  const assigned = nodes.map(() => 0);
  for (let layer = 0; layer < layers; layer += 1) {
    const candidates = nodes
      .map((node, index) => ({ node, index }))
      .filter(({ node, index }) => assigned[index]! < node.layerCapacity)
      .sort(
        (a, b) =>
          (assigned[a.index]! + 1) * a.node.decodeMsPerLayer -
          (assigned[b.index]! + 1) * b.node.decodeMsPerLayer,
      );
    const selected = candidates[0];
    if (selected === undefined) return null;
    assigned[selected.index] = assigned[selected.index]! + 1;
  }
  return assigned;
}

function assignRouteLayers(nodes: SimulatedNode[]): SimulatedNode[] | null {
  const assigned = allocateLayersWaterFill(
    nodes.map((node) => ({
      layerCapacity: node.cap,
      decodeMsPerLayer: node.decodeMsPerLayer,
    })),
  );
  if (assigned === null) return null;
  for (let index = 0; index < nodes.length; index += 1) {
    nodes[index]!.assignedLayers = assigned[index]!;
  }
  return nodes.filter((node) => node.assignedLayers > 0);
}

function routeStats(nodes: SimulatedNode[], options: WanOptions): RouteStats | null {
  const stages = assignRouteLayers(nodes);
  if (stages === null || stages.length === 0) return null;

  const activationBytes = activationBoundaryBytes();
  const microbatches = Math.ceil(options.promptTokens / options.microchunkTokens);
  let decodeComputeMs = 0;
  let decodeNetworkMs = 0;
  const cycles: number[] = [];

  for (let index = 0; index < stages.length; index += 1) {
    const current = stages[index]!;
    const next = stages[(index + 1) % stages.length]!;
    decodeComputeMs += current.assignedLayers * current.decodeMsPerLayer;

    // Coordinator RTTs are combined into an A -> relay -> B one-way estimate.
    const oneWayMs = (current.coordinatorRttMs + next.coordinatorRttMs) / 2;
    const linkMbps = Math.min(current.uplinkMbps, next.uplinkMbps);
    const decodeSerializationMs =
      ((activationBytes * 8) / (linkMbps * 1e6)) * 1_000;
    decodeNetworkMs += oneWayMs + decodeSerializationMs + 1;

    const ppStageTps =
      (current.ppFullEquivalentTps * GLM45_AIR_Q4KM.layers) /
      current.assignedLayers;
    const chunkComputeMs = (options.microchunkTokens / ppStageTps) * 1_000;
    const chunkSerializationMs =
      ((activationBytes * options.microchunkTokens * 8) /
        (linkMbps * 1e6)) *
      1_000;
    cycles.push(chunkComputeMs + oneWayMs + chunkSerializationMs + 1);
  }

  const tpotMs = decodeComputeMs + decodeNetworkMs;
  const ttftMs =
    options.routeSetupMs +
    sum(cycles) +
    (microbatches - 1) * Math.max(...cycles);
  return {
    stages: stages.length,
    tokensPerSecond: 1_000 / tpotMs,
    ttftMs,
    powerW: sum(stages.map((stage) => stage.powerW)),
  };
}

function summarizeTrials(
  nodes: number,
  options: WanOptions,
  routesByTrial: readonly (readonly RouteStats[])[],
): WanSummary {
  const routeCounts = routesByTrial.map((routes) => routes.length);
  const allRoutes = routesByTrial.flat();
  const idealSlots = routesByTrial.map((routes) =>
    sum(routes.map((route) => route.stages)),
  );
  const powerKw = routesByTrial.map(
    (routes) => sum(routes.map((route) => route.powerW)) / 1_000,
  );
  return {
    kind: THEORETICAL_PHYSICS_KIND,
    idealSlotsSemantics: IDEAL_SLOTS_DISCLAIMER,
    nodes,
    trials: routesByTrial.length,
    probabilityAnyRoute:
      routeCounts.filter((routeCount) => routeCount > 0).length /
      routesByTrial.length,
    routes: {
      p50: nearestRank(routeCounts, 0.5),
      p95: nearestRank(routeCounts, 0.95),
    },
    stages: {
      p50: nearestRank(
        allRoutes.map((route) => route.stages),
        0.5,
      ),
      p95: nearestRank(
        allRoutes.map((route) => route.stages),
        0.95,
      ),
    },
    tokensPerSecond: {
      p50: nearestRank(
        allRoutes.map((route) => route.tokensPerSecond),
        0.5,
      ),
      slowPathP95: nearestRank(
        allRoutes.map((route) => route.tokensPerSecond),
        0.05,
      ),
    },
    ttftMs: {
      p50: nearestRank(
        allRoutes.map((route) => route.ttftMs),
        0.5,
      ),
      p95: nearestRank(
        allRoutes.map((route) => route.ttftMs),
        0.95,
      ),
    },
    concurrency: {
      p50: nearestRank(idealSlots, 0.5),
      p95: nearestRank(idealSlots, 0.95),
    },
    incrementalPowerKw: {
      p50: nearestRank(powerKw, 0.5),
      p95: nearestRank(powerKw, 0.95),
    },
    kvBytesPerConversation: kvBytesForContext(options.contextTokens),
  };
}

function createDomesticNode(
  tier: Readonly<TierProfile>,
  region: Readonly<RegionProfile>,
  rng: () => number,
): SimulatedNode {
  const performanceMultiplier = Math.exp(0.25 * standardNormal(rng));
  return {
    cap: tier.layerCapacity,
    decodeMsPerLayer: tier.decodeMsPerLayer / performanceMultiplier,
    ppFullEquivalentTps: tier.ppFullEquivalentTps * performanceMultiplier,
    coordinatorRttMs: lognormal(region.rttMs, 0.55, rng),
    uplinkMbps: Math.max(5, lognormal(100, 0.8, rng)),
    powerW: tier.powerW * (0.85 + 0.3 * rng()),
    assignedLayers: 0,
  };
}

function packRegionalRoutes(
  regionalNodes: SimulatedNode[][],
  options: WanOptions,
): RouteStats[] {
  const routes: RouteStats[] = [];
  for (const source of regionalNodes) {
    const available = [...source].sort(
      (a, b) => b.cap - a.cap || a.decodeMsPerLayer - b.decodeMsPerLayer,
    );
    while (sum(available.map((node) => node.cap)) >= GLM45_AIR_Q4KM.layers) {
      const selected: SimulatedNode[] = [];
      let capacity = 0;
      while (available.length > 0 && capacity < GLM45_AIR_Q4KM.layers) {
        const node = available.shift()!;
        selected.push(node);
        capacity += node.cap;
      }
      const route = routeStats(selected, options);
      // A longer route is consumed but treated as batch-only, not interactive.
      if (route !== null && route.stages <= options.maxInteractiveStages) {
        routes.push(route);
      }
    }
  }
  return routes;
}

export function simulateWanFleet(
  options: WanOptions,
  rng: () => number = mulberry32(options.seed),
): WanSummary {
  assertOptions(options);
  const routesByTrial: RouteStats[][] = [];
  for (let trial = 0; trial < options.trials; trial += 1) {
    const regionalNodes = DOMESTIC_REGIONS.map(() => [] as SimulatedNode[]);
    for (let nodeIndex = 0; nodeIndex < options.nodes; nodeIndex += 1) {
      // Draw order is part of the reproducibility contract. Generate all physical
      // properties before drawing online and reserve status.
      const tier = weightedPick(DOMESTIC_TIERS, rng);
      const region = weightedPick(DOMESTIC_REGIONS, rng);
      const regionIndex = DOMESTIC_REGIONS.indexOf(region);
      const node = createDomesticNode(tier, region, rng);
      const online = rng() < options.onlineProbability;
      // Reserve status only exists for an online node. The short-circuit is part
      // of the physics-v1 RNG stream and therefore of the published snapshots.
      const usableHot = online && rng() < 1 - options.hotReserveShare;
      if (online && usableHot) regionalNodes[regionIndex]!.push(node);
    }
    routesByTrial.push(packRegionalRoutes(regionalNodes, options));
  }
  return summarizeTrials(options.nodes, options, routesByTrial);
}

/** Runs fleet sizes sequentially against one shared RNG stream. */
export function simulateWanSuite(
  nodeCounts: readonly number[] = [10, 100, 1_000],
  overrides: Partial<Omit<WanOptions, "nodes" | "seed">> = {},
  seed = DEFAULT_WAN_PHYSICS_OPTIONS.seed,
): WanSummary[] {
  const rng = mulberry32(seed);
  return nodeCounts.map((nodes) =>
    simulateWanFleet(
      createWanPhysicsOptions({ ...overrides, nodes, seed }),
      rng,
    ),
  );
}

const CURATED_VRAM_GB = [8, 8, 8, 8, 12, 12, 12, 16, 16, 24] as const;

/** A deliberately viable ten-node regional route, unlike ten random homes. */
export function simulateCuratedTenNodeRoute(
  trials = 10_000,
  seed = 987_654_321,
  overrides: Partial<
    Omit<WanOptions, "nodes" | "trials" | "seed" | "onlineProbability" | "hotReserveShare">
  > = {},
): WanSummary {
  const options = createWanPhysicsOptions({
    ...overrides,
    nodes: 10,
    trials,
    seed,
    onlineProbability: 1,
    hotReserveShare: 0,
  });
  assertOptions(options);
  const rng = mulberry32(seed);
  const routesByTrial: RouteStats[][] = [];
  for (let trial = 0; trial < trials; trial += 1) {
    const nodes = CURATED_VRAM_GB.map((vramGb) => {
      const tier = DOMESTIC_TIERS.find((candidate) => candidate.vramGb === vramGb)!;
      const performanceMultiplier = Math.exp(0.25 * standardNormal(rng));
      return {
        cap: tier.layerCapacity,
        decodeMsPerLayer: tier.decodeMsPerLayer / performanceMultiplier,
        ppFullEquivalentTps: tier.ppFullEquivalentTps * performanceMultiplier,
        coordinatorRttMs: lognormal(18, 0.55, rng),
        uplinkMbps: Math.max(5, lognormal(150, 0.8, rng)),
        powerW: tier.powerW * (0.85 + 0.3 * rng()),
        assignedLayers: 0,
      } satisfies SimulatedNode;
    });
    const route = routeStats([...nodes], options);
    routesByTrial.push(route === null ? [] : [route]);
  }
  return summarizeTrials(10, options, routesByTrial);
}

export function activationBoundaryBytes(
  bytesPerElement = GLM45_AIR_Q4KM.activationBytesPerElement,
): number {
  return GLM45_AIR_Q4KM.hiddenSize * bytesPerElement;
}

export function kvBytesPerContextToken(
  bytesPerElement = GLM45_AIR_Q4KM.kvBytes,
): number {
  return (
    2 *
    GLM45_AIR_Q4KM.layers *
    GLM45_AIR_Q4KM.kvHeads *
    GLM45_AIR_Q4KM.headDim *
    bytesPerElement
  );
}

export function kvBytesForContext(
  contextTokens: number,
  bytesPerElement = GLM45_AIR_Q4KM.kvBytes,
): number {
  if (!Number.isFinite(contextTokens) || contextTokens < 0) {
    throw new RangeError("contextTokens must be a non-negative finite number");
  }
  return kvBytesPerContextToken(bytesPerElement) * contextTokens;
}

export function kvLimitedConversationSlots(
  availableKvBytes: number,
  contextTokens: number,
  bytesPerElement = GLM45_AIR_Q4KM.kvBytes,
): number {
  const bytes = kvBytesForContext(contextTokens, bytesPerElement);
  if (bytes === 0) return 0;
  return Math.max(0, Math.floor(availableKvBytes / bytes));
}

export function activeWeightReadBytesPerToken(): number {
  return (
    (GLM45_AIR_Q4KM.weightBytes * GLM45_AIR_Q4KM.activeParams) /
    GLM45_AIR_Q4KM.totalParams
  );
}

export function approximateActiveFlopsPerToken(): number {
  return 2 * GLM45_AIR_Q4KM.activeParams;
}

export function routeSuccessProbability(
  stages: number,
  perStageFailureProbability: number,
): number {
  if (!Number.isInteger(stages) || stages < 0) {
    throw new RangeError("stages must be a non-negative integer");
  }
  if (
    perStageFailureProbability < 0 ||
    perStageFailureProbability > 1 ||
    !Number.isFinite(perStageFailureProbability)
  ) {
    throw new RangeError("perStageFailureProbability must be between zero and one");
  }
  return (1 - perStageFailureProbability) ** stages;
}

/**
 * Closed-form decode sensitivity. The default has no protocol overhead so the
 * canonical RTT/BW matrix remains easy to audit; Monte Carlo routes add 1 ms/hop.
 */
export function estimateWanSensitivity(
  input: WanSensitivityInput,
): WanSensitivityPoint {
  const activationBytes = input.activationBytes ?? activationBoundaryBytes();
  const protocolOverheadMs = input.protocolOverheadMs ?? 0;
  if (!Number.isInteger(input.stages) || input.stages <= 0) {
    throw new RangeError("stages must be a positive integer");
  }
  if (input.bandwidthMbps <= 0) {
    throw new RangeError("bandwidthMbps must be positive");
  }
  const serializationMs =
    ((activationBytes * 8) / (input.bandwidthMbps * 1e6)) * 1_000;
  const tpotMs =
    input.computeMs +
    input.stages *
      (input.oneWayHopMs + serializationMs + protocolOverheadMs);
  return {
    stages: input.stages,
    computeMs: input.computeMs,
    oneWayHopMs: input.oneWayHopMs,
    bandwidthMbps: input.bandwidthMbps,
    activationBytes,
    tpotMs,
    tokensPerSecond: 1_000 / tpotMs,
  };
}

export function defaultWanSensitivityMatrix(): WanSensitivityPoint[] {
  const computeByStages = [
    { stages: 6, computeMs: 80 },
    { stages: 10, computeMs: 110 },
    { stages: 16, computeMs: 170 },
  ] as const;
  const oneWayHopValues = [5, 15, 50] as const;
  return computeByStages.flatMap(({ stages, computeMs }) =>
    oneWayHopValues.map((oneWayHopMs) =>
      estimateWanSensitivity({
        stages,
        computeMs,
        oneWayHopMs,
        bandwidthMbps: 100,
      }),
    ),
  );
}
