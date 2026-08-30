import {
  directedLink,
  distributionObjective,
  evaluateDistributionPlan,
  rawTransferMs,
  stageMemoryBytes,
} from "./cost-model.js";
import { activationCodec } from "./codecs.js";
import { UNMEASURED_RTT_MS } from "../core/rtt.js";
import type {
  ActivationCodecId,
  ComputeNodeProfile,
  DistributedModelProfile,
  DistributionPlan,
  DistributionPlanner,
  DistributionTopology,
  DistributionWorkload,
  EvaluatedDistributionPlan,
  SearchOptions,
  StagePlacement,
} from "./types.js";

export { MacroWaveRamVramPlanner } from "./macro-wave.js";
export type {
  MacroWaveLinkCost,
  MacroWaveModeCost,
  MacroWavePlannerOptions,
  MacroWavePlanningResult,
  MacroWaveRouteCost,
  MacroWaveStageCost,
} from "./macro-wave.js";

export const DEFAULT_SEARCH_OPTIONS: SearchOptions = {
  beamWidth: 512,
  candidateCodecs: ["fp16", "int8", "q4"],
  // Keep the search wide enough to discover the concurrency knee. The cost
  // model still decides which candidate is usable for the current route.
  candidateMicroBatchSizes: [1, 2, 4, 8, 16, 32],
  candidatePrefillChunks: [16, 32, 64, 128, 256],
  objectiveWeights: {
    tpot: 1,
    ttft: 0.05,
    response: 0.002,
    energy: 20,
    unavailability: 5,
  },
};

export class VramGreedyPlanner implements DistributionPlanner {
  readonly id = "vram-greedy";

  constructor(private readonly options: SearchOptions = DEFAULT_SEARCH_OPTIONS) {}

  plan(
    model: DistributedModelProfile,
    topology: DistributionTopology,
    workload: DistributionWorkload,
  ): DistributionPlan | null {
    const ordered = topology.nodes
      .slice()
      .sort(
        (left, right) =>
          usableMemory(right) - usableMemory(left) || left.decodeScale - right.decodeScale,
      );
    const stages: StagePlacement[] = [];
    let layerStart = 0;
    for (const node of ordered.slice(0, workload.maxStages)) {
      const end = furthestLayerThatFits(
        model,
        workload,
        node,
        layerStart,
        stages.length === 0,
      );
      if (end <= layerStart) continue;
      stages.push({ nodeId: node.id, layerStart, layerEnd: end });
      layerStart = end;
      if (layerStart === model.layers.length) {
        return tunePlacement(this.id, stages, model, topology, workload, this.options)?.plan ?? null;
      }
    }
    return null;
  }
}

export class ProportionalComputePlanner implements DistributionPlanner {
  readonly id = "compute-proportional";

  constructor(private readonly options: SearchOptions = DEFAULT_SEARCH_OPTIONS) {}

  plan(
    model: DistributedModelProfile,
    topology: DistributionTopology,
    workload: DistributionWorkload,
  ): DistributionPlan | null {
    const ordered = topology.nodes
      .slice()
      .sort(
        (left, right) =>
          nodeLatencyScore(left, topology) - nodeLatencyScore(right, topology),
      );
    let best: EvaluatedDistributionPlan | null = null;
    const maxStages = Math.min(workload.maxStages, ordered.length, model.layers.length);
    for (let count = 1; count <= maxStages; count += 1) {
      const nodes = ordered.slice(0, count);
      const placement = proportionalPlacement(model, workload, nodes);
      if (!placement) continue;
      const tuned = tunePlacement(this.id, placement, model, topology, workload, this.options);
      if (tuned && (!best || tuned.objective < best.objective)) best = tuned;
    }
    return best?.plan ?? null;
  }
}

export class OrderedDynamicProgrammingPlanner implements DistributionPlanner {
  readonly id = "ordered-dp";

  constructor(private readonly options: SearchOptions = DEFAULT_SEARCH_OPTIONS) {}

  plan(
    model: DistributedModelProfile,
    topology: DistributionTopology,
    workload: DistributionWorkload,
  ): DistributionPlan | null {
    const ordered = topology.nodes
      .slice()
      .sort(
        (left, right) =>
          nodeLatencyScore(left, topology) - nodeLatencyScore(right, topology),
      );
    let best: EvaluatedDistributionPlan | null = null;
    const maxStages = Math.min(workload.maxStages, ordered.length, model.layers.length);
    for (let count = 1; count <= maxStages; count += 1) {
      const placement = orderedDpPlacement(
        model,
        topology,
        workload,
        ordered.slice(0, count),
      );
      if (!placement) continue;
      const tuned = tunePlacement(this.id, placement, model, topology, workload, this.options);
      if (tuned && (!best || tuned.objective < best.objective)) best = tuned;
    }
    return best?.plan ?? null;
  }
}

interface BeamState {
  stages: StagePlacement[];
  usedNodes: Set<string>;
  nextLayer: number;
  routeAvailability: number;
  heuristic: number;
  lowerBound: number;
}

export class TopologyBeamPlanner implements DistributionPlanner {
  readonly id = "topology-beam";

  constructor(private readonly options: SearchOptions = DEFAULT_SEARCH_OPTIONS) {}

  plan(
    model: DistributedModelProfile,
    topology: DistributionTopology,
    workload: DistributionWorkload,
  ): DistributionPlan | null {
    const nodes = topology.nodes
      .filter((node) => node.availability > 0)
      .sort(
        (left, right) =>
          nodeLatencyScore(left, topology) - nodeLatencyScore(right, topology),
      );
    let frontier: BeamState[] = [
      {
        stages: [],
        usedNodes: new Set(),
        nextLayer: 0,
        routeAvailability: 1,
        heuristic: 0,
        lowerBound: optimisticRemainingCompute(model, nodes, 0),
      },
    ];
    const completed: BeamState[] = [];
    for (let stageCount = 0; stageCount < workload.maxStages; stageCount += 1) {
      const next: BeamState[] = [];
      for (const state of frontier) {
        for (const node of nodes) {
          if (state.usedNodes.has(node.id)) continue;
          const first = state.stages.length === 0;
          for (
            let layerEnd = state.nextLayer + 1;
            layerEnd <= model.layers.length;
            layerEnd += 1
          ) {
            if (exceedsStageLayerLimit(node, state.nextLayer, layerEnd)) break;
            const last = layerEnd === model.layers.length;
            if (!nodeSupportsStagePosition(node, first, last)) continue;
            const stage: StagePlacement = {
              nodeId: node.id,
              layerStart: state.nextLayer,
              layerEnd,
            };
            const memory = stageMemoryBytes(model, stage, workload, first, last);
            if (memory > usableMemory(node)) break;
            const previous = state.stages.at(-1);
            const linkCost = previous
              ? boundaryHeuristicMs(model, topology, previous, node.id, workload.p95)
              : 0;
            if (!Number.isFinite(linkCost)) continue;
            const availability =
              state.routeAvailability *
              node.availability *
              (previous
                ? 1 - (directedLink(topology, previous.nodeId, node.id)?.lossRate ?? 1)
                : 1);
            if (availability <= 0) continue;
            const local = stageHeuristicMs(model, stage, node, workload.promptTokens);
            const stages = [...state.stages, stage];
            const heuristic = state.heuristic + local + linkCost;
            const candidate: BeamState = {
              stages,
              usedNodes: new Set([...state.usedNodes, node.id]),
              nextLayer: layerEnd,
              routeAvailability: availability,
              heuristic,
              lowerBound:
                heuristic + optimisticRemainingCompute(model, nodes, layerEnd),
            };
            if (last) completed.push(candidate);
            else if (
              canStillCoverRemainingLayers(
                model,
                workload,
                nodes,
                candidate.usedNodes,
                layerEnd,
                workload.maxStages - stages.length,
              )
            ) {
              next.push(candidate);
            }
          }
        }
      }
      frontier = stratifiedBeamPrune(deduplicateBeamStates(next), this.options.beamWidth);
      if (frontier.length === 0) break;
    }

    let best: EvaluatedDistributionPlan | null = null;
    for (const state of completed
      .sort((left, right) => left.heuristic - right.heuristic)
      .slice(0, this.options.beamWidth * 2)) {
      const tuned = tunePlacement(this.id, state.stages, model, topology, workload, this.options);
      if (tuned && (!best || tuned.objective < best.objective)) best = tuned;
    }
    if (!best) return null;
    return refineEvaluatedPlan(best, model, topology, workload, this.options).plan;
  }
}

export interface FleetPlannerOptions {
  /** Maximum nodes admitted to the expensive joint placement search. */
  candidateLimit: number;
  /** Number of low-latency cells explored independently. */
  cellLimit: number;
  /** Maximum candidates retained from any single cell. */
  nodesPerCell: number;
}

const DEFAULT_FLEET_OPTIONS: FleetPlannerOptions = {
  candidateLimit: 64,
  cellLimit: 4,
  nodesPerCell: 16,
};

/**
 * Scalable front-end for fleets with hundreds or thousands of machines.
 * It keeps the full fleet out of the combinatorial beam search, but preserves
 * diverse candidates (fastest, largest-memory and best-connected) inside the
 * most promising latency cells. The actual route is still optimized by the
 * same exact cost model as TopologyBeamPlanner.
 */
/**
 * Drop nodes whose network position is catastrophically worse than the fleet's.
 *
 * Since the cost model sums latency along the route, one network outlier taxes
 * every token that crosses it. Excluding a catastrophic outlier can therefore
 * be safer than forcing it into a route.
 *
 * The rule is relative, not absolute: a threshold in milliseconds would be wrong
 * for a LAN cell and wrong again for an intercontinental swarm. A node is evicted
 * when its median link is `multiple` times worse than the fleet median.
 *
 * Two guards prevent over-pruning:
 *  - Never evict below `minimumNodes`. A planner with nothing left to place is
 *    worse than a slow route.
 *  - If the rule wants to drop (nearly) everyone, the outlier is the measurement,
 *    not the fleet. peer runtime words it as "if I banned them all, the problem is me".
 */
export function evictFarNodes(
  topology: DistributionTopology,
  options: { multiple?: number; minimumNodes?: number } = {},
): { topology: DistributionTopology; evicted: string[] } {
  const multiple = options.multiple ?? 3;
  const minimumNodes = options.minimumNodes ?? 2;
  if (topology.nodes.length <= minimumNodes) return { topology, evicted: [] };

  const latencies = new Map<string, number[]>();
  for (const link of topology.links) {
    if (!Number.isFinite(link.oneWayLatencyMs) || link.oneWayLatencyMs <= 0) continue;
    for (const id of [link.from, link.to]) {
      const bucket = latencies.get(id);
      if (bucket) bucket.push(link.oneWayLatencyMs);
      else latencies.set(id, [link.oneWayLatencyMs]);
    }
  }
  const nodeMedians = new Map<string, number>();
  for (const [id, samples] of latencies) nodeMedians.set(id, median(samples));
  // Nodes without a single measured link are NOT evicted: unmeasured is not the
  // same as slow, and punishing it would make the fleet shrink as instrumentation
  // lags behind. They are simply invisible to this rule.
  if (nodeMedians.size < 3) return { topology, evicted: [] };

  const fleetMedian = median([...nodeMedians.values()]);
  if (fleetMedian <= 0) return { topology, evicted: [] };
  const ceiling = fleetMedian * multiple;

  const candidates = [...nodeMedians.entries()]
    .filter(([, value]) => value > ceiling)
    .sort((left, right) => right[1] - left[1])
    .map(([id]) => id);
  if (candidates.length === 0) return { topology, evicted: [] };
  if (topology.nodes.length - candidates.length < minimumNodes) {
    // "Si los baneé a todos, el problema soy yo."
    return { topology, evicted: [] };
  }

  const evicted = new Set(candidates);
  return {
    topology: {
      nodes: topology.nodes.filter((node) => !evicted.has(node.id)),
      links: topology.links.filter(
        (link) => !evicted.has(link.from) && !evicted.has(link.to),
      ),
    },
    evicted: candidates,
  };
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = sorted.length >> 1;
  return sorted.length % 2 === 1
    ? sorted[middle]!
    : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

export class FleetTopologyPlanner implements DistributionPlanner {
  readonly id = "fleet-topology";

  constructor(
    private readonly searchOptions: SearchOptions = DEFAULT_SEARCH_OPTIONS,
    private readonly fleetOptions: FleetPlannerOptions = DEFAULT_FLEET_OPTIONS,
  ) {}

  plan(
    model: DistributedModelProfile,
    fullTopology: DistributionTopology,
    workload: DistributionWorkload,
  ): DistributionPlan | null {
    // Antes de gastar la búsqueda combinatoria, quitar de en medio los nodos
    // cuya posición de red arruina cualquier ruta que los cruce.
    const { topology, evicted } = evictFarNodes(fullTopology);
    const pruned = this.planWithin(model, topology, workload);
    if (pruned) return pruned;

    // La poda es una heurística de LATENCIA, no una comprobación de viabilidad.
    // `evictFarNodes` sólo se compromete a conservar dos nodos: no mira las
    // capas del modelo, ni la memoria por nodo, ni el agregado de la flota. En
    // una flota de más de 64 nodos —el único caso que llega hasta aquí— puede
    // dejar un subconjunto donde el modelo ya no cabe, y entonces devolver
    // `null` sería declarar imposible un plan que sí existe: la flota entera
    // tenía sitio, y lo que lo quitó fue un descarte de valores atípicos.
    //
    // Reintentar con la topología completa cuesta una segunda búsqueda sólo en
    // el camino donde la primera ya fracasó, así que el caso normal no paga
    // nada. Preferir un plan lento a ningún plan.
    if (evicted.length === 0) return null;
    return this.planWithin(model, fullTopology, workload);
  }

  private planWithin(
    model: DistributedModelProfile,
    topology: DistributionTopology,
    workload: DistributionWorkload,
  ): DistributionPlan | null {
    if (topology.nodes.length <= this.fleetOptions.candidateLimit) {
      const direct = new TopologyBeamPlanner(this.searchOptions).plan(model, topology, workload);
      return direct ? { ...direct, algorithm: this.id } : null;
    }

    const cells = [...groupNodesByCell(topology).entries()]
      .map(([cell, nodes]) => ({
        cell,
        nodes,
        score: fleetCellScore(nodes, topology),
      }))
      .sort((left, right) => left.score - right.score)
      .slice(0, this.fleetOptions.cellLimit);
    const pools: ComputeNodeProfile[][] = cells.map((entry) =>
      diverseNodeCandidates(
        entry.nodes,
        topology,
        Math.min(this.fleetOptions.nodesPerCell, this.fleetOptions.candidateLimit),
      ),
    );
    const combined = diverseNodeCandidates(
      pools.flat(),
      topology,
      this.fleetOptions.candidateLimit,
    );
    if (combined.length > 0) pools.push(combined);

    let best: EvaluatedDistributionPlan | null = null;
    for (const nodes of deduplicateNodePools(pools)) {
      const candidateTopology = topologySubset(topology, nodes);
      const plan = new TopologyBeamPlanner(this.searchOptions).plan(
        model,
        candidateTopology,
        workload,
      );
      if (!plan) continue;
      const metrics = evaluateDistributionPlan(model, topology, workload, plan);
      const evaluated: EvaluatedDistributionPlan = {
        plan: { ...plan, algorithm: this.id },
        metrics,
        objective: distributionObjective(metrics, this.searchOptions.objectiveWeights),
      };
      if (Number.isFinite(evaluated.objective) && (!best || evaluated.objective < best.objective)) {
        best = evaluated;
      }
    }
    return best?.plan ?? null;
  }
}

export class ExhaustiveTopologyPlanner implements DistributionPlanner {
  readonly id = "exhaustive-optimal";
  private evaluations = 0;

  constructor(
    private readonly options: SearchOptions = DEFAULT_SEARCH_OPTIONS,
    private readonly maxPlacements = 1_000_000,
  ) {}

  plan(
    model: DistributedModelProfile,
    topology: DistributionTopology,
    workload: DistributionWorkload,
  ): DistributionPlan | null {
    this.evaluations = 0;
    let best: EvaluatedDistributionPlan | null = null;
    const visit = (stages: StagePlacement[], used: Set<string>, nextLayer: number): void => {
      if (this.evaluations >= this.maxPlacements) return;
      if (nextLayer === model.layers.length) {
        this.evaluations += 1;
        const tuned = tunePlacement(this.id, stages, model, topology, workload, this.options);
        if (tuned && (!best || tuned.objective < best.objective)) best = tuned;
        return;
      }
      if (stages.length >= workload.maxStages) return;
      for (const node of topology.nodes) {
        if (used.has(node.id)) continue;
        for (let end = nextLayer + 1; end <= model.layers.length; end += 1) {
          if (exceedsStageLayerLimit(node, nextLayer, end)) break;
          const stage = { nodeId: node.id, layerStart: nextLayer, layerEnd: end };
          if (!nodeSupportsStagePosition(
            node,
            stages.length === 0,
            end === model.layers.length,
          )) continue;
          if (
            stageMemoryBytes(
              model,
              stage,
              workload,
              stages.length === 0,
              end === model.layers.length,
            ) > usableMemory(node)
          ) {
            break;
          }
          visit([...stages, stage], new Set([...used, node.id]), end);
        }
      }
    };
    visit([], new Set(), 0);
    const finalBest = best as EvaluatedDistributionPlan | null;
    return finalBest?.plan ?? null;
  }
}

export function defaultDistributionPlanners(
  options: SearchOptions = DEFAULT_SEARCH_OPTIONS,
): DistributionPlanner[] {
  return [
    new VramGreedyPlanner(options),
    new ProportionalComputePlanner(options),
    new OrderedDynamicProgrammingPlanner(options),
    new TopologyBeamPlanner(options),
  ];
}

export function evaluatePlanner(
  planner: DistributionPlanner,
  model: DistributedModelProfile,
  topology: DistributionTopology,
  workload: DistributionWorkload,
  options: SearchOptions = DEFAULT_SEARCH_OPTIONS,
): EvaluatedDistributionPlan | null {
  const plan = planner.plan(model, topology, workload);
  if (!plan) return null;
  const metrics = evaluateDistributionPlan(model, topology, workload, plan);
  return {
    plan,
    metrics,
    objective: distributionObjective(metrics, options.objectiveWeights),
  };
}

function tunePlacement(
  algorithm: string,
  stages: StagePlacement[],
  model: DistributedModelProfile,
  topology: DistributionTopology,
  workload: DistributionWorkload,
  options: SearchOptions,
): EvaluatedDistributionPlan | null {
  let best: EvaluatedDistributionPlan | null = null;
  for (const codec of options.candidateCodecs) {
    for (const microBatchSize of options.candidateMicroBatchSizes) {
      if (microBatchSize > workload.concurrentSequences) continue;
      for (const prefillChunkTokens of options.candidatePrefillChunks) {
        const plan: DistributionPlan = {
          algorithm,
          codec,
          microBatchSize,
          prefillChunkTokens,
          stages,
        };
        const metrics = evaluateDistributionPlan(model, topology, workload, plan);
        const objective = distributionObjective(metrics, options.objectiveWeights);
        if (Number.isFinite(objective) && (!best || objective < best.objective)) {
          best = { plan, metrics, objective };
        }
      }
    }
  }
  return best;
}

function refineEvaluatedPlan(
  initial: EvaluatedDistributionPlan,
  model: DistributedModelProfile,
  topology: DistributionTopology,
  workload: DistributionWorkload,
  options: SearchOptions,
): EvaluatedDistributionPlan {
  let best = exhaustiveLocalBoundaryRefinement(
    initial,
    model,
    topology,
    workload,
    options,
  );
  for (let iteration = 0; iteration < 4; iteration += 1) {
    let improved = best;
    const neighbors = placementNeighbors(best.plan.stages, topology.nodes);
    // A node swap can be temporarily infeasible at the old boundaries even
    // though the swapped order is excellent after rebalancing. Re-run the
    // boundary DP for each neighboring order so local search can cross that
    // barrier without widening the global beam.
    const nodeById = new Map(topology.nodes.map((node) => [node.id, node]));
    for (const neighbor of neighbors.slice()) {
      const orderedNodes = neighbor
        .map((stage) => nodeById.get(stage.nodeId))
        .filter((node): node is ComputeNodeProfile => Boolean(node));
      if (orderedNodes.length !== neighbor.length) continue;
      const rebalanced = orderedDpPlacement(
        model,
        topology,
        workload,
        orderedNodes,
      );
      if (rebalanced) neighbors.push(rebalanced);
    }
    for (const neighbor of neighbors) {
      const tuned = tunePlacement(
        best.plan.algorithm,
        neighbor,
        model,
        topology,
        workload,
        options,
      );
      if (tuned && tuned.objective + 1e-9 < improved.objective) improved = tuned;
    }
    if (improved.objective + 1e-9 >= best.objective) break;
    best = improved;
  }
  return best;
}

function exhaustiveLocalBoundaryRefinement(
  initial: EvaluatedDistributionPlan,
  model: DistributedModelProfile,
  topology: DistributionTopology,
  workload: DistributionWorkload,
  options: SearchOptions,
): EvaluatedDistributionPlan {
  const stageCount = initial.plan.stages.length;
  const orders: string[][] = [initial.plan.stages.map((stage) => stage.nodeId)];
  for (let left = 0; left < stageCount; left += 1) {
    for (let right = left + 1; right < stageCount; right += 1) {
      const order = orders[0]!.slice();
      [order[left], order[right]] = [order[right]!, order[left]!];
      orders.push(order);
    }
  }
  const compositions = binomial(model.layers.length - 1, stageCount - 1);
  // Exact boundary enumeration is valuable for short routes, but the beam must
  // remain bounded for large models and many stages.
  if (compositions * orders.length > 5_000) return initial;
  const nodeById = new Map(topology.nodes.map((node) => [node.id, node]));
  let best = initial;
  for (const order of orders) {
    const nodes = order.map((id) => nodeById.get(id));
    if (nodes.some((node) => !node)) continue;
    const visit = (stages: StagePlacement[], nextLayer: number): void => {
      const index = stages.length;
      const stagesRemaining = stageCount - index;
      if (stagesRemaining === 1) {
        const stage: StagePlacement = {
          nodeId: order[index]!,
          layerStart: nextLayer,
          layerEnd: model.layers.length,
        };
        if (
          stageMemoryBytes(
            model,
            stage,
            workload,
            index === 0,
            true,
          ) > usableMemory(nodes[index]!)
        ) {
          return;
        }
        const tuned = tunePlacement(
          initial.plan.algorithm,
          [...stages, stage],
          model,
          topology,
          workload,
          options,
        );
        if (tuned && tuned.objective < best.objective) best = tuned;
        return;
      }
      const maximumEnd = model.layers.length - (stagesRemaining - 1);
      for (let end = nextLayer + 1; end <= maximumEnd; end += 1) {
        const stage: StagePlacement = {
          nodeId: order[index]!,
          layerStart: nextLayer,
          layerEnd: end,
        };
        if (
          stageMemoryBytes(model, stage, workload, index === 0, false) >
          usableMemory(nodes[index]!)
        ) {
          break;
        }
        visit([...stages, stage], end);
      }
    };
    visit([], 0);
  }
  return best;
}

function binomial(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  const reduced = Math.min(k, n - k);
  let result = 1;
  for (let index = 1; index <= reduced; index += 1) {
    result = (result * (n - reduced + index)) / index;
    if (result > 5_000) return result;
  }
  return Math.round(result);
}

function placementNeighbors(
  stages: StagePlacement[],
  nodes: ComputeNodeProfile[],
): StagePlacement[][] {
  const results: StagePlacement[][] = [];
  for (let boundary = 0; boundary < stages.length - 1; boundary += 1) {
    const left = stages[boundary]!;
    const right = stages[boundary + 1]!;
    for (const delta of [-2, -1, 1, 2]) {
      const split = left.layerEnd + delta;
      if (split <= left.layerStart || split >= right.layerEnd) continue;
      const candidate = stages.map((stage) => ({ ...stage }));
      candidate[boundary]!.layerEnd = split;
      candidate[boundary + 1]!.layerStart = split;
      results.push(candidate);
    }
  }
  for (let leftIndex = 0; leftIndex < stages.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < stages.length; rightIndex += 1) {
      const candidate = stages.map((stage) => ({ ...stage }));
      const leftNode = candidate[leftIndex]!.nodeId;
      candidate[leftIndex]!.nodeId = candidate[rightIndex]!.nodeId;
      candidate[rightIndex]!.nodeId = leftNode;
      results.push(candidate);
    }
  }
  const used = new Set(stages.map((stage) => stage.nodeId));
  for (let stageIndex = 0; stageIndex < stages.length; stageIndex += 1) {
    for (const node of nodes) {
      if (used.has(node.id)) continue;
      const candidate = stages.map((stage) => ({ ...stage }));
      candidate[stageIndex]!.nodeId = node.id;
      results.push(candidate);
    }
  }
  return results;
}

function orderedDpPlacement(
  model: DistributedModelProfile,
  topology: DistributionTopology,
  workload: DistributionWorkload,
  nodes: ComputeNodeProfile[],
): StagePlacement[] | null {
  interface Cell {
    cost: number;
    stages: StagePlacement[];
  }
  let cells = new Map<number, Cell>();
  cells.set(0, { cost: 0, stages: [] });
  for (let stageIndex = 0; stageIndex < nodes.length; stageIndex += 1) {
    const node = nodes[stageIndex]!;
    const nextCells = new Map<number, Cell>();
    for (const [start, cell] of cells) {
      const remainingStages = nodes.length - stageIndex - 1;
      const maximumEnd = model.layers.length - remainingStages;
      for (let end = start + 1; end <= maximumEnd; end += 1) {
        const last = stageIndex === nodes.length - 1;
        if (last && end !== model.layers.length) continue;
        const stage: StagePlacement = { nodeId: node.id, layerStart: start, layerEnd: end };
        if (
          stageMemoryBytes(model, stage, workload, stageIndex === 0, last) >
          usableMemory(node)
        ) {
          break;
        }
        const previous = cell.stages.at(-1);
        const linkCost = previous
          ? boundaryHeuristicMs(model, topology, previous, node.id, workload.p95)
          : 0;
        if (!Number.isFinite(linkCost)) continue;
        const cost =
          cell.cost + stageHeuristicMs(model, stage, node, workload.promptTokens) + linkCost;
        const existing = nextCells.get(end);
        if (!existing || cost < existing.cost) {
          nextCells.set(end, { cost, stages: [...cell.stages, stage] });
        }
      }
    }
    cells = nextCells;
    if (cells.size === 0) return null;
  }
  return cells.get(model.layers.length)?.stages ?? null;
}

function proportionalPlacement(
  model: DistributedModelProfile,
  workload: DistributionWorkload,
  nodes: ComputeNodeProfile[],
): StagePlacement[] | null {
  if (nodes.length > model.layers.length) return null;
  const capacities = nodes.map((node, index) =>
    maximumLayerCount(model, workload, node, index === 0, index === nodes.length - 1),
  );
  if (capacities.some((capacity) => capacity < 1)) return null;
  if (sum(capacities) < model.layers.length) return null;
  const speeds = nodes.map((node) => 1 / Math.max(0.01, node.decodeScale));
  const speedTotal = sum(speeds);
  const counts = nodes.map(() => 1);
  let remaining = model.layers.length - nodes.length;
  while (remaining > 0) {
    let chosen = -1;
    let bestDeficit = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < nodes.length; index += 1) {
      if (counts[index]! >= capacities[index]!) continue;
      const target = (model.layers.length * speeds[index]!) / speedTotal;
      const deficit = target - counts[index]!;
      if (deficit > bestDeficit) {
        bestDeficit = deficit;
        chosen = index;
      }
    }
    if (chosen < 0) return null;
    counts[chosen]! += 1;
    remaining -= 1;
  }
  const stages: StagePlacement[] = [];
  let start = 0;
  for (let index = 0; index < nodes.length; index += 1) {
    const end = start + counts[index]!;
    stages.push({ nodeId: nodes[index]!.id, layerStart: start, layerEnd: end });
    start = end;
  }
  return stages;
}

function furthestLayerThatFits(
  model: DistributedModelProfile,
  workload: DistributionWorkload,
  node: ComputeNodeProfile,
  start: number,
  first: boolean,
): number {
  let best = start;
  for (let end = start + 1; end <= model.layers.length; end += 1) {
    if (exceedsStageLayerLimit(node, start, end)) break;
    const stage = { nodeId: node.id, layerStart: start, layerEnd: end };
    const memory = stageMemoryBytes(model, stage, workload, first, end === model.layers.length);
    if (memory > usableMemory(node)) break;
    best = end;
  }
  return best;
}

function maximumLayerCount(
  model: DistributedModelProfile,
  workload: DistributionWorkload,
  node: ComputeNodeProfile,
  first: boolean,
  last: boolean,
): number {
  let count = 0;
  for (let end = 1; end <= model.layers.length; end += 1) {
    if (exceedsStageLayerLimit(node, 0, end)) break;
    const stage = { nodeId: node.id, layerStart: 0, layerEnd: end };
    if (stageMemoryBytes(model, stage, workload, first, last) > usableMemory(node)) break;
    count = end;
  }
  return count;
}

function stageHeuristicMs(
  model: DistributedModelProfile,
  stage: StagePlacement,
  node: ComputeNodeProfile,
  promptTokens: number,
): number {
  let decode = 0;
  let prefill = 0;
  for (let index = stage.layerStart; index < stage.layerEnd; index += 1) {
    decode += model.layers[index]!.decodeMsAtUnit;
    prefill += model.layers[index]!.prefillMsPerTokenAtUnit;
  }
  return decode * node.decodeScale + prefill * node.prefillScale * promptTokens * 0.05;
}

function boundaryHeuristicMs(
  model: DistributedModelProfile,
  topology: DistributionTopology,
  previous: StagePlacement,
  nextNodeId: string,
  p95: boolean,
): number {
  const link = directedLink(topology, previous.nodeId, nextNodeId);
  if (!link) return Number.POSITIVE_INFINITY;
  const elements = model.layers[previous.layerEnd - 1]!.activationElements;
  return rawTransferMs(elements * 2, link, p95);
}

function optimisticRemainingCompute(
  model: DistributedModelProfile,
  nodes: ComputeNodeProfile[],
  fromLayer: number,
): number {
  const fastest = Math.min(...nodes.map((node) => node.decodeScale));
  let result = 0;
  for (let index = fromLayer; index < model.layers.length; index += 1) {
    result += model.layers[index]!.decodeMsAtUnit * fastest;
  }
  return result;
}

function nodeLatencyScore(node: ComputeNodeProfile, topology: DistributionTopology): number {
  const outgoing = topology.links.filter((link) => link.from === node.id);
  // Un nodo sin aristas salientes ya costaba el centinela. Lo que faltaba es
  // que una arista concreta sin medir cueste lo mismo: antes, un nodo con una
  // sola arista medida y nueve desconocidas promediaba barato y entraba en el
  // plan. `oneWayLatencyMs` ya llega con el centinela desde `auto-distribute`,
  // así que aquí basta con no diluirlo — el promedio lo hace por sí solo.
  const meanLink =
    outgoing.length > 0
      ? outgoing.reduce(
          (total, link) =>
            total + link.oneWayLatencyMs + link.jitterP95Ms + 10 / link.bandwidthMbps,
          0,
        ) / outgoing.length
      : UNMEASURED_RTT_MS;
  return node.decodeScale + node.prefillScale * 0.25 + meanLink * 0.08;
}

function groupNodesByCell(
  topology: DistributionTopology,
): Map<string, ComputeNodeProfile[]> {
  const cells = new Map<string, ComputeNodeProfile[]>();
  for (const node of topology.nodes) {
    const nodes = cells.get(node.region) ?? [];
    nodes.push(node);
    cells.set(node.region, nodes);
  }
  return cells;
}

function fleetCellScore(
  nodes: ComputeNodeProfile[],
  topology: DistributionTopology,
): number {
  const ranked = nodes
    .map((node) => nodeLatencyScore(node, topology))
    .sort((left, right) => left - right);
  const sample = ranked.slice(0, Math.min(8, ranked.length));
  const meanLatency = sum(sample) / Math.max(1, sample.length);
  const usableGiB =
    sum(nodes.map((node) => usableMemory(node))) / (1024 * 1024 * 1024);
  // Memory only helps until a route fits, hence the logarithm. Compute/network
  // remain dominant once that hard constraint is satisfied.
  return meanLatency - Math.log2(1 + usableGiB) * 0.35;
}

function diverseNodeCandidates(
  nodes: ComputeNodeProfile[],
  topology: DistributionTopology,
  limit: number,
): ComputeNodeProfile[] {
  if (nodes.length <= limit) return nodes.slice();
  const orderings = [
    nodes.slice().sort((left, right) => left.decodeScale - right.decodeScale),
    nodes.slice().sort((left, right) => left.prefillScale - right.prefillScale),
    nodes.slice().sort((left, right) => usableMemory(right) - usableMemory(left)),
    nodes
      .slice()
      .sort(
        (left, right) =>
          nodeLatencyScore(left, topology) - nodeLatencyScore(right, topology),
      ),
  ];
  const selected: ComputeNodeProfile[] = [];
  const seen = new Set<string>();
  for (let rank = 0; selected.length < limit; rank += 1) {
    let added = false;
    for (const ordering of orderings) {
      const node = ordering[rank];
      if (!node || seen.has(node.id)) continue;
      seen.add(node.id);
      selected.push(node);
      added = true;
      if (selected.length === limit) break;
    }
    if (!added && orderings.every((ordering) => rank >= ordering.length)) break;
  }
  return selected;
}

function topologySubset(
  topology: DistributionTopology,
  nodes: ComputeNodeProfile[],
): DistributionTopology {
  const ids = new Set(nodes.map((node) => node.id));
  return {
    nodes,
    links: topology.links.filter((link) => ids.has(link.from) && ids.has(link.to)),
  };
}

function deduplicateNodePools(pools: ComputeNodeProfile[][]): ComputeNodeProfile[][] {
  const unique = new Map<string, ComputeNodeProfile[]>();
  for (const pool of pools) {
    if (pool.length === 0) continue;
    const key = pool.map((node) => node.id).sort().join("\u0000");
    unique.set(key, pool);
  }
  return [...unique.values()];
}

function deduplicateBeamStates(states: BeamState[]): BeamState[] {
  const best = new Map<string, BeamState>();
  for (const state of states) {
    const key = `${state.nextLayer}|${[...state.usedNodes].sort().join(",")}|${state.stages.at(-1)?.nodeId ?? ""}`;
    const existing = best.get(key);
    if (!existing || state.lowerBound < existing.lowerBound) best.set(key, state);
  }
  return [...best.values()];
}

function stratifiedBeamPrune(states: BeamState[], beamWidth: number): BeamState[] {
  if (states.length <= beamWidth) return states.sort((left, right) => left.lowerBound - right.lowerBound);
  const byProgress = new Map<number, BeamState[]>();
  for (const state of states) {
    const group = byProgress.get(state.nextLayer) ?? [];
    group.push(state);
    byProgress.set(state.nextLayer, group);
  }
  const groups = [...byProgress.values()];
  const quota = Math.max(1, Math.floor(beamWidth / groups.length));
  const selected: BeamState[] = [];
  const selectedKeys = new Set<BeamState>();
  for (const group of groups) {
    group.sort((left, right) => left.lowerBound - right.lowerBound);
    for (const state of group.slice(0, quota)) {
      selected.push(state);
      selectedKeys.add(state);
    }
  }
  if (selected.length < beamWidth) {
    const remaining = states
      .filter((state) => !selectedKeys.has(state))
      .sort((left, right) => left.lowerBound - right.lowerBound);
    selected.push(...remaining.slice(0, beamWidth - selected.length));
  }
  return selected
    .sort((left, right) => left.lowerBound - right.lowerBound)
    .slice(0, beamWidth);
}

function canStillCoverRemainingLayers(
  model: DistributedModelProfile,
  workload: DistributionWorkload,
  nodes: ComputeNodeProfile[],
  usedNodes: Set<string>,
  nextLayer: number,
  stagesLeft: number,
): boolean {
  const remainingLayers = model.layers.length - nextLayer;
  if (remainingLayers <= 0) return true;
  if (stagesLeft <= 0) return false;
  const capacities = nodes
    .filter((node) => !usedNodes.has(node.id))
    .map((node) => {
      let count = 0;
      for (let end = nextLayer + 1; end <= model.layers.length; end += 1) {
        const stage = { nodeId: node.id, layerStart: nextLayer, layerEnd: end };
        if (stageMemoryBytes(model, stage, workload, false, false) > usableMemory(node)) break;
        count = end - nextLayer;
      }
      return count;
    })
    .sort((left, right) => right - left)
    .slice(0, stagesLeft);
  return sum(capacities) >= remainingLayers;
}

function usableMemory(node: ComputeNodeProfile): number {
  return Math.max(0, node.memoryBytes - node.reserveBytes);
}

function exceedsStageLayerLimit(
  node: ComputeNodeProfile,
  layerStart: number,
  layerEnd: number,
): boolean {
  return node.maxStageLayers !== undefined
    && layerEnd - layerStart > node.maxStageLayers;
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

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

export function codecsWithinQualityBudget(
  workload: DistributionWorkload,
  codecs: ActivationCodecId[],
): ActivationCodecId[] {
  return codecs.filter(
    (codec) =>
      activationCodec(codec).estimatedQualityLoss <= workload.maxQualityLoss,
  );
}
