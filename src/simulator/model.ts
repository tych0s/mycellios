import { tierForOfferedVram, type WorkerTier } from "../core/tiers.js";

export type SimulationScenario =
  | "normal"
  | "large_model_spike"
  | "region_outage"
  | "churn_20";

export interface SimulationOptions {
  nodes: number;
  users: number;
  seed: number;
  scenario: SimulationScenario;
  onlineProbability?: number;
}

export interface SimulationResult {
  options: SimulationOptions;
  inventory: {
    totalNodes: number;
    onlineNodes: number;
    onlineVramGb: number;
    glmRoutes: number;
    glmStages: { p50: number; p95: number };
  };
  traffic: Record<
    "small" | "medium" | "glm",
    {
      requested: number;
      accepted: number;
      queued: number;
      tokensPerSecond: { p50: number; p95: number };
      ttftMs: { p50: number; p95: number };
    }
  >;
  network: {
    accepted: number;
    queued: number;
    acceptanceRate: number;
    activeNodes: number;
    utilization: number;
    estimatedPowerKw: number;
    workGini: number;
    typicalSixTurnMinutes: number;
  };
  byTier: Record<WorkerTier, { total: number; online: number; active: number }>;
}

interface SimNode {
  id: number;
  vramGb: number;
  tier: WorkerTier;
  region: string;
  online: boolean;
  powerW: number;
  smallTps: number;
  mediumTps: number;
  slots: number;
  assignments: number;
  reservedForGlm: boolean;
}

interface GlmRoute {
  nodes: SimNode[];
  region: string;
  tokensPerSecond: number;
  ttftMs: number;
  slots: number;
  assignments: number;
}

interface Allocation {
  tps: number[];
  ttft: number[];
  accepted: number;
  queued: number;
}

const INVENTORY = [
  { share: 0.5, vramGb: 4 },
  { share: 0.25, vramGb: 8 },
  { share: 0.15, vramGb: 12 },
  { share: 0.075, vramGb: 16 },
  { share: 0.025, vramGb: 24 },
] as const;

const REGIONS = [
  { name: "eu-west", share: 0.35, rttMs: 24 },
  { name: "eu-central", share: 0.25, rttMs: 31 },
  { name: "us-east", share: 0.2, rttMs: 42 },
  { name: "us-west", share: 0.1, rttMs: 55 },
  { name: "apac", share: 0.1, rttMs: 68 },
] as const;

export function simulateNetwork(options: SimulationOptions): SimulationResult {
  const random = mulberry32(options.seed >>> 0);
  const nodes = generateNodes(options, random);
  const routes = buildGlmRoutes(nodes, random);
  const mix = trafficMix(options.users, options.scenario);
  const glm = allocateGlm(mix.glm, routes);
  const medium = allocateReplicas(
    mix.medium,
    nodes.filter((node) => node.online && !node.reservedForGlm && node.vramGb >= 8),
    "medium",
  );
  const small = allocateReplicas(
    mix.small,
    nodes.filter((node) => node.online && !node.reservedForGlm),
    "small",
  );

  const activeNodes = nodes.filter((node) => node.assignments > 0);
  const onlineNodes = nodes.filter((node) => node.online);
  const accepted = small.accepted + medium.accepted + glm.accepted;
  const queued = small.queued + medium.queued + glm.queued;
  // Include idle online contributors so fairness is not artificially improved
  // by dropping every participant that received no work.
  const assignmentCounts = onlineNodes.map((node) => node.assignments);
  const aggregateTps = [...small.tps, ...medium.tps, ...glm.tps];
  const typicalTps = percentile(aggregateTps, 0.5);
  const typicalTtft = percentile([...small.ttft, ...medium.ttft, ...glm.ttft], 0.5);

  return {
    options,
    inventory: {
      totalNodes: nodes.length,
      onlineNodes: onlineNodes.length,
      onlineVramGb: round(onlineNodes.reduce((sum, node) => sum + node.vramGb, 0), 1),
      glmRoutes: routes.length,
      glmStages: {
        p50: percentile(
          routes.map((route) => route.nodes.length),
          0.5,
        ),
        p95: percentile(
          routes.map((route) => route.nodes.length),
          0.95,
        ),
      },
    },
    traffic: {
      small: summarizeAllocation(mix.small, small),
      medium: summarizeAllocation(mix.medium, medium),
      glm: summarizeAllocation(mix.glm, glm),
    },
    network: {
      accepted,
      queued,
      acceptanceRate: round(options.users === 0 ? 1 : accepted / options.users, 4),
      activeNodes: activeNodes.length,
      utilization: round(onlineNodes.length === 0 ? 0 : activeNodes.length / onlineNodes.length, 4),
      estimatedPowerKw: round(
        activeNodes.reduce((sum, node) => sum + node.powerW, 0) / 1_000,
        2,
      ),
      workGini: round(gini(assignmentCounts), 4),
      typicalSixTurnMinutes: round(
        typicalTps <= 0 ? 0 : (6 * (typicalTtft / 1_000 + 200 / typicalTps)) / 60,
        2,
      ),
    },
    byTier: summarizeTiers(nodes),
  };
}

function generateNodes(options: SimulationOptions, random: () => number): SimNode[] {
  const onlineProbability = options.onlineProbability ?? 0.7;
  const inventory = exactDistribution(options.nodes, INVENTORY);
  return inventory.map((vramGb, id) => {
    const region = weightedPick(REGIONS, random);
    let online = random() < onlineProbability;
    if (options.scenario === "region_outage" && region.name === "eu-west") online = false;
    if (options.scenario === "churn_20" && random() < 0.2) online = false;
    const performance = 0.75 + random() * 0.5;
    return {
      id,
      vramGb,
      tier: tierForOfferedVram(vramGb * 1_024),
      region: region.name,
      online,
      powerW: powerForVram(vramGb) * (0.85 + random() * 0.3),
      smallTps: (8 + vramGb * 0.8) * performance,
      mediumTps: vramGb >= 8 ? (3 + vramGb * 0.45) * performance : 0,
      slots: vramGb >= 24 ? 3 : vramGb >= 12 ? 2 : 1,
      assignments: 0,
      reservedForGlm: false,
    };
  });
}

function buildGlmRoutes(nodes: SimNode[], random: () => number): GlmRoute[] {
  const routes: GlmRoute[] = [];
  for (const region of REGIONS) {
    const candidates = nodes
      .filter((node) => node.online && node.region === region.name && node.vramGb >= 12)
      .sort((left, right) => right.vramGb - left.vramGb || left.id - right.id);
    while (candidates.length >= 3) {
      const selected: SimNode[] = [];
      let usableVram = 0;
      while (candidates.length > 0 && selected.length < 8 && usableVram < 70) {
        const node = candidates.shift()!;
        selected.push(node);
        usableVram += node.vramGb * 0.75;
      }
      if (usableVram < 70 || selected.length < 3) break;
      for (const node of selected) node.reservedForGlm = true;
      const stages = selected.length;
      const baseTps = Math.max(
        1.5,
        8.5 - Math.max(0, stages - 3) * 0.65 - region.rttMs / 65 + (random() - 0.5),
      );
      routes.push({
        nodes: selected,
        region: region.name,
        tokensPerSecond: baseTps,
        ttftMs: 3_000 + stages * 650 + region.rttMs * 25,
        slots: stages <= 5 ? 8 : 6,
        assignments: 0,
      });
    }
  }
  return routes;
}

function allocateGlm(requested: number, routes: GlmRoute[]): Allocation {
  const allocation: Allocation = { tps: [], ttft: [], accepted: 0, queued: 0 };
  for (let index = 0; index < requested; index += 1) {
    const route = routes
      .filter((candidate) => candidate.assignments < candidate.slots)
      .sort(
        (left, right) =>
          left.assignments / left.slots - right.assignments / right.slots ||
          right.tokensPerSecond - left.tokensPerSecond,
      )[0];
    if (!route) {
      allocation.queued += 1;
      continue;
    }
    route.assignments += 1;
    for (const node of route.nodes) node.assignments += 1;
    const degradation = Math.max(0.5, 1 - (route.assignments - 1) * 0.07);
    allocation.tps.push(route.tokensPerSecond * degradation);
    allocation.ttft.push(route.ttftMs * (1 + (route.assignments - 1) * 0.05));
    allocation.accepted += 1;
  }
  return allocation;
}

function allocateReplicas(
  requested: number,
  nodes: SimNode[],
  model: "small" | "medium",
): Allocation {
  const allocation: Allocation = { tps: [], ttft: [], accepted: 0, queued: 0 };
  const remaining = new Map(nodes.map((node) => [node.id, node.slots]));
  for (let index = 0; index < requested; index += 1) {
    const node = nodes
      .filter((candidate) => (remaining.get(candidate.id) ?? 0) > 0)
      .sort((left, right) => {
        const leftTps = model === "small" ? left.smallTps : left.mediumTps;
        const rightTps = model === "small" ? right.smallTps : right.mediumTps;
        return left.assignments - right.assignments || rightTps - leftTps;
      })[0];
    if (!node) {
      allocation.queued += 1;
      continue;
    }
    remaining.set(node.id, (remaining.get(node.id) ?? 1) - 1);
    node.assignments += 1;
    const tps = model === "small" ? node.smallTps : node.mediumTps;
    allocation.tps.push(tps * Math.max(0.65, 1 - (node.assignments - 1) * 0.15));
    allocation.ttft.push((model === "small" ? 1_400 : 2_500) * (1 + node.assignments * 0.1));
    allocation.accepted += 1;
  }
  return allocation;
}

function trafficMix(users: number, scenario: SimulationScenario): {
  small: number;
  medium: number;
  glm: number;
} {
  if (scenario === "large_model_spike") {
    const small = Math.floor(users * 0.1);
    const medium = Math.floor(users * 0.1);
    return { small, medium, glm: users - small - medium };
  }
  const small = Math.floor(users * 0.55);
  const medium = Math.floor(users * 0.3);
  return { small, medium, glm: users - small - medium };
}

function summarizeAllocation(requested: number, allocation: Allocation) {
  return {
    requested,
    accepted: allocation.accepted,
    queued: allocation.queued,
    tokensPerSecond: {
      p50: round(percentile(allocation.tps, 0.5), 2),
      p95: round(percentile(allocation.tps, 0.95), 2),
    },
    ttftMs: {
      p50: Math.round(percentile(allocation.ttft, 0.5)),
      p95: Math.round(percentile(allocation.ttft, 0.95)),
    },
  };
}

function summarizeTiers(nodes: SimNode[]): SimulationResult["byTier"] {
  const tiers: WorkerTier[] = ["INELIGIBLE", "T0", "T1", "T2", "T3", "T4"];
  return Object.fromEntries(
    tiers.map((tier) => {
      const matching = nodes.filter((node) => node.tier === tier);
      return [
        tier,
        {
          total: matching.length,
          online: matching.filter((node) => node.online).length,
          active: matching.filter((node) => node.assignments > 0).length,
        },
      ];
    }),
  ) as SimulationResult["byTier"];
}

function exactDistribution(
  total: number,
  distribution: ReadonlyArray<{ share: number; vramGb: number }>,
): number[] {
  const counts = distribution.map((entry) => Math.floor(total * entry.share));
  let assigned = counts.reduce((sum, count) => sum + count, 0);
  let index = 0;
  while (assigned < total) {
    counts[index % counts.length]! += 1;
    assigned += 1;
    index += 1;
  }
  return counts.flatMap((count, position) =>
    Array.from({ length: count }, () => distribution[position]!.vramGb),
  );
}

function weightedPick<T extends { share: number }>(items: readonly T[], random: () => number): T {
  let cursor = random();
  for (const item of items) {
    cursor -= item.share;
    if (cursor <= 0) return item;
  }
  return items.at(-1)!;
}

function powerForVram(vramGb: number): number {
  if (vramGb <= 4) return 60;
  if (vramGb <= 8) return 90;
  if (vramGb <= 12) return 120;
  if (vramGb <= 16) return 150;
  return 200;
}

function percentile(values: number[], quantile: number): number {
  if (values.length === 0) return 0;
  const sorted = values.slice().sort((left, right) => left - right);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(quantile * sorted.length) - 1));
  return sorted[index]!;
}

function gini(values: number[]): number {
  const filtered = values.filter((value) => value >= 0).sort((left, right) => left - right);
  const total = filtered.reduce((sum, value) => sum + value, 0);
  if (filtered.length === 0 || total === 0) return 0;
  const weighted = filtered.reduce((sum, value, index) => sum + (index + 1) * value, 0);
  return (2 * weighted) / (filtered.length * total) - (filtered.length + 1) / filtered.length;
}

function round(value: number, decimals: number): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let result = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    result = (result + Math.imul(result ^ (result >>> 7), 61 | result)) ^ result;
    return ((result ^ (result >>> 14)) >>> 0) / 4_294_967_296;
  };
}
