import { createHash } from "node:crypto";
import {
  requireEligibleEngineRuntimeProfile,
  type EngineRuntimeProfile,
  type EngineRuntimeProfilePolicy,
} from "../contracts/engine-runtime-profile.js";
import {
  distributionObjective,
  evaluateDistributionPlan,
} from "./cost-model.js";
import {
  DEFAULT_SEARCH_OPTIONS,
  ExhaustiveTopologyPlanner,
  FleetTopologyPlanner,
  TopologyBeamPlanner,
} from "./planners.js";
import type {
  DistributionMetrics,
  DistributionPlan,
  DistributionTopology,
  DistributionWorkload,
  DistributedModelProfile,
  SearchOptions,
} from "./types.js";

export type NativeRouteWorkloadClass = "interactive" | "throughput";

export interface NativeFleetNodeState {
  nodeId: string;
  /** Processes on one physical machine share this identifier. */
  physicalHostId: string;
  /** Independent power/network/failure boundary, normally a home or site. */
  failureDomainId: string;
  queuedRequests: number;
  activeRequests: number;
  capacity: number;
  freeSlots: number;
  observedP95ServiceMs: number | null;
  kvSessionIds: readonly string[];
  engineCapability?: NativeEngineNodeCapability | undefined;
}

export interface NativeEngineNodeCapability {
  profileId: `sha256:${string}`;
  descriptorDigest: `sha256:${string}`;
  certificationId: `sha256:${string}`;
  artifactManifestDigest: `sha256:${string}`;
  backend: string;
  runtimeAbi: string;
  quantizations: readonly string[];
  maxContextTokens: number;
  maxLayerCount: number;
  kvBytesPerToken: number;
  maxKvTokens: number;
  decodeScale: number;
  prefillScale: number;
  fastKernel: boolean;
  graphMode: "available" | "unavailable";
  roles: readonly ("head" | "middle" | "tail" | "draft" | "auxiliary")[];
  validUntilMs: number;
}

export interface NativeEngineRouteRequirement {
  descriptorDigest: `sha256:${string}`;
  certificationId: `sha256:${string}`;
  artifactManifestDigest: `sha256:${string}`;
  backend: string;
  runtimeAbi: string;
  quantization: string;
  contextTokens: number;
  requiredRoles: readonly NativeEngineNodeCapability["roles"][number][];
  requireFastKernel: boolean;
  requireGraph: boolean;
  nowMs: number;
}

export interface NativeRouteRequest {
  workloadClass: NativeRouteWorkloadClass;
  sessionId: string | null;
  kvMissPenaltyMs: number;
  engine?: NativeEngineRouteRequirement | undefined;
}

export interface NativeRouteScore {
  total: number;
  computeMs: number;
  queueMs: number;
  p95Ms: number;
  kvPenaltyMs: number;
  physicalBoundaryMs: number;
  physicalBoundaryCount: number;
  failureRisk: number;
}

export interface NativeCompleteChain {
  chainId: string;
  plan: DistributionPlan;
  metrics: DistributionMetrics;
  score: NativeRouteScore;
  nodeIds: string[];
  failureDomainIds: string[];
}

export interface NativeFleetPlacement {
  requestedReplicas: number;
  chains: NativeCompleteChain[];
  complete: boolean;
  reason: string | null;
}

export interface NativeFleetPlacementOptions {
  desiredReplicas: number;
  searchOptions?: SearchOptions;
}

export interface PlannerOracleResult {
  plannerObjective: number;
  oracleObjective: number;
  relativeGap: number;
  withinTolerance: boolean;
  plannerPlan: DistributionPlan;
  oraclePlan: DistributionPlan;
}

/**
 * Two-phase native placement.
 *
 * Phase one creates complete, short chains. Phase two repeats the placement on
 * disjoint failure domains so every result is independently executable. A
 * partial stage set is never published as a replica.
 */
export function planNativeReplicaChains(
  model: DistributedModelProfile,
  topology: DistributionTopology,
  workload: DistributionWorkload,
  nodeStates: readonly NativeFleetNodeState[],
  request: NativeRouteRequest,
  options: NativeFleetPlacementOptions,
): NativeFleetPlacement {
  if (
    !Number.isInteger(options.desiredReplicas)
    || options.desiredReplicas < 1
    || options.desiredReplicas > 64
  ) {
    throw new Error("native_replica_count_is_invalid");
  }
  const stateByNode = validateNodeStates(topology, nodeStates);
  const searchOptions = options.searchOptions ?? DEFAULT_SEARCH_OPTIONS;
  const excludedNodes = new Set<string>();
  const excludedFailureDomains = new Set<string>();
  const chains: NativeCompleteChain[] = [];

  for (let replica = 0; replica < options.desiredReplicas; replica += 1) {
    const nodes = topology.nodes.filter((node) => {
      const state = stateByNode.get(node.id)!;
      return (
        state.freeSlots > 0
        && engineNodeIsEligible(state, request.engine)
        && !excludedNodes.has(node.id)
        && !excludedFailureDomains.has(state.failureDomainId)
      );
    });
    if (nodes.length === 0) break;
    const ids = new Set(nodes.map((node) => node.id));
    const candidateTopology: DistributionTopology = {
      nodes: nodes.map((node) => congestionAdjustedNode(
        node,
        stateByNode.get(node.id)!,
        request.engine,
      )),
      links: topology.links.filter((link) => ids.has(link.from) && ids.has(link.to)),
    };
    const selected = nativePlacementCandidates(
      model,
      candidateTopology,
      workload,
      stateByNode,
      request,
      searchOptions,
    )[0];
    if (!selected) break;
    const { plan, metrics, score } = selected;
    const nodeIds = plan.stages.map((stage) => stage.nodeId);
    const failureDomainIds = [
      ...new Set(nodeIds.map((nodeId) => stateByNode.get(nodeId)!.failureDomainId)),
    ].sort();
    const chainId = chainDigest(plan, failureDomainIds);
    chains.push({
      chainId,
      plan: { ...plan, algorithm: "native-two-phase" },
      metrics,
      score,
      nodeIds,
      failureDomainIds,
    });
    for (const nodeId of nodeIds) excludedNodes.add(nodeId);
    for (const domainId of failureDomainIds) excludedFailureDomains.add(domainId);
  }

  return {
    requestedReplicas: options.desiredReplicas,
    chains,
    complete: chains.length === options.desiredReplicas,
    reason:
      chains.length === options.desiredReplicas
        ? null
        : `independent_complete_chains_unavailable:${chains.length}:${options.desiredReplicas}`,
  };
}

interface NativePlacementCandidate {
  plan: DistributionPlan;
  metrics: DistributionMetrics;
  score: NativeRouteScore;
}

/**
 * The generic beam minimizes model/network cost, while native placement also
 * knows physical hosts, live queues, KV affinity and failure boundaries. Run a
 * bounded set of complete searches whose scopes can actually remove a WAN
 * boundary, then choose only after evaluating the full native loop. This keeps
 * the search scalable and prevents a post-hoc boundary metric from pretending
 * it influenced placement when it did not.
 */
function nativePlacementCandidates(
  model: DistributedModelProfile,
  topology: DistributionTopology,
  workload: DistributionWorkload,
  stateByNode: ReadonlyMap<string, NativeFleetNodeState>,
  request: NativeRouteRequest,
  searchOptions: SearchOptions,
): NativePlacementCandidate[] {
  const nodeSets: string[][] = [topology.nodes.map((node) => node.id)];
  const byHost = new Map<string, string[]>();
  for (const node of topology.nodes) {
    const host = stateByNode.get(node.id)!.physicalHostId;
    byHost.set(host, [...(byHost.get(host) ?? []), node.id]);
    nodeSets.push([node.id]);
  }
  nodeSets.push(...byHost.values());

  const candidates = new Map<string, NativePlacementCandidate>();
  for (const nodeIds of nodeSets) {
    const ids = new Set(nodeIds);
    const scoped: DistributionTopology = {
      nodes: topology.nodes.filter((node) => ids.has(node.id)),
      links: topology.links.filter((link) => ids.has(link.from) && ids.has(link.to)),
    };
    if (scoped.nodes.length === 0) continue;
    const planner = scoped.nodes.length > 64
      ? new FleetTopologyPlanner(searchOptions)
      : new TopologyBeamPlanner(searchOptions);
    const plan = planner.plan(model, scoped, workload);
    if (!plan) continue;
    const key = plan.stages.map((stage) =>
      `${stage.nodeId}:${stage.layerStart}-${stage.layerEnd}`
    ).join("|");
    if (candidates.has(key)) continue;
    const metrics = evaluateDistributionPlan(model, topology, workload, plan);
    const score = scoreNativeRoute(plan, metrics, topology, stateByNode, request);
    if (metrics.feasible && Number.isFinite(score.total)) {
      candidates.set(key, { plan, metrics, score });
    }
  }
  return [...candidates.values()].sort((left, right) =>
    left.score.total - right.score.total
    || left.plan.stages.length - right.plan.stages.length
    || nativePlanIdentity(left.plan).localeCompare(nativePlanIdentity(right.plan))
  );
}

function nativePlanIdentity(plan: DistributionPlan): string {
  return plan.stages.map((stage) =>
    `${stage.nodeId}:${stage.layerStart}-${stage.layerEnd}`
  ).join("|");
}

/**
 * Per-request route selection. Interactive traffic pays the sum along the
 * critical path; throughput traffic pays the slowest pipeline service.
 */
export function selectNativeChain(
  chains: readonly NativeCompleteChain[],
  request: NativeRouteRequest,
  topology: DistributionTopology,
  nodeStates: readonly NativeFleetNodeState[],
): NativeCompleteChain | null {
  const stateByNode = validateNodeStates(topology, nodeStates);
  return chains
    .map((chain) => ({
      ...chain,
      score: scoreNativeRoute(
        chain.plan,
        chain.metrics,
        topology,
        stateByNode,
        request,
      ),
    }))
    .filter((chain) => Number.isFinite(chain.score.total))
    .sort(
      (left, right) =>
        left.score.total - right.score.total
        || left.chainId.localeCompare(right.chainId),
    )[0] ?? null;
}

export function scoreNativeRoute(
  plan: DistributionPlan,
  metrics: DistributionMetrics,
  topology: DistributionTopology,
  stateByNode: ReadonlyMap<string, NativeFleetNodeState>,
  request: NativeRouteRequest,
): NativeRouteScore {
  if (!metrics.feasible) return infiniteRouteScore();
  const states = plan.stages.map((stage) => stateByNode.get(stage.nodeId));
  if (states.some((state) => !state || state.freeSlots < 1)) return infiniteRouteScore();
  const concreteStates = states as NativeFleetNodeState[];
  const queueCosts = concreteStates.map((state) => {
    const service = state.observedP95ServiceMs;
    if (service === null || !Number.isFinite(service) || service <= 0) {
      return Number.POSITIVE_INFINITY;
    }
    return (state.queuedRequests * service) / Math.max(1, state.capacity);
  });
  if (queueCosts.some((cost) => !Number.isFinite(cost))) return infiniteRouteScore();
  const p95Costs = concreteStates.map((state) => state.observedP95ServiceMs!);
  const affinityHit = request.sessionId !== null
    && concreteStates.every((state) => state.kvSessionIds.includes(request.sessionId!));
  const kvPenaltyMs = affinityHit ? 0 : finiteNonNegative(request.kvMissPenaltyMs);
  const boundary = physicalBoundaryCost(plan, topology, stateByNode);
  if (!Number.isFinite(boundary.ms)) return infiniteRouteScore();
  const failureRisk = plan.stages.reduce((risk, stage) => {
    const node = topology.nodes.find((candidate) => candidate.id === stage.nodeId);
    return risk + (node ? 1 - clampProbability(node.availability) : 1);
  }, 0);
  const interactive = request.workloadClass === "interactive";
  const computeMs = interactive ? metrics.pathDecodeMs : metrics.pipelineCycleMs;
  const queueMs = interactive ? sum(queueCosts) : Math.max(...queueCosts);
  const p95Ms = interactive ? sum(p95Costs) : Math.max(...p95Costs);
  return {
    total:
      computeMs
      + queueMs
      + p95Ms
      + kvPenaltyMs
      + boundary.ms
      + failureRisk * 1_000,
    computeMs,
    queueMs,
    p95Ms,
    kvPenaltyMs,
    physicalBoundaryMs: boundary.ms,
    physicalBoundaryCount: boundary.count,
    failureRisk,
  };
}

/**
 * Exact small-instance oracle used only in tests/offline validation. It blocks
 * a fast-planner change whose objective drifts farther from the exhaustive
 * optimum than the declared tolerance.
 */
export function comparePlannerWithExhaustiveOracle(
  model: DistributedModelProfile,
  topology: DistributionTopology,
  workload: DistributionWorkload,
  searchOptions: SearchOptions,
  maximumRelativeGap = 0.1,
): PlannerOracleResult {
  if (
    !Number.isFinite(maximumRelativeGap)
    || maximumRelativeGap < 0
    || maximumRelativeGap > 10
  ) {
    throw new Error("planner_oracle_tolerance_is_invalid");
  }
  const fast = new TopologyBeamPlanner(searchOptions).plan(model, topology, workload);
  const oracle = new ExhaustiveTopologyPlanner(searchOptions).plan(model, topology, workload);
  if (!fast || !oracle) throw new Error("planner_oracle_requires_feasible_plans");
  const fastMetrics = evaluateDistributionPlan(model, topology, workload, fast);
  const oracleMetrics = evaluateDistributionPlan(model, topology, workload, oracle);
  const plannerObjective = distributionObjective(
    fastMetrics,
    searchOptions.objectiveWeights,
  );
  const oracleObjective = distributionObjective(
    oracleMetrics,
    searchOptions.objectiveWeights,
  );
  if (!Number.isFinite(plannerObjective) || !Number.isFinite(oracleObjective)) {
    throw new Error("planner_oracle_objective_is_invalid");
  }
  const relativeGap = oracleObjective === 0
    ? (plannerObjective === 0 ? 0 : Number.POSITIVE_INFINITY)
    : Math.max(0, (plannerObjective - oracleObjective) / oracleObjective);
  return {
    plannerObjective,
    oracleObjective,
    relativeGap,
    withinTolerance: relativeGap <= maximumRelativeGap,
    plannerPlan: fast,
    oraclePlan: oracle,
  };
}

export function requirePlannerWithinOracleTolerance(
  result: PlannerOracleResult,
): void {
  if (!result.withinTolerance) {
    throw new Error(
      `native_planner_oracle_regression:${result.relativeGap.toFixed(6)}`,
    );
  }
}

function validateNodeStates(
  topology: DistributionTopology,
  nodeStates: readonly NativeFleetNodeState[],
): Map<string, NativeFleetNodeState> {
  const result = new Map<string, NativeFleetNodeState>();
  for (const state of nodeStates) {
    if (
      !state.nodeId
      || !state.physicalHostId
      || !state.failureDomainId
      || !Number.isInteger(state.queuedRequests)
      || state.queuedRequests < 0
      || !Number.isInteger(state.activeRequests)
      || state.activeRequests < 0
      || !Number.isInteger(state.capacity)
      || state.capacity < 1
      || !Number.isInteger(state.freeSlots)
      || state.freeSlots < 0
      || state.freeSlots > state.capacity
      || !engineCapabilityIsValid(state.engineCapability)
    ) {
      throw new Error(`native_node_state_is_invalid:${state.nodeId}`);
    }
    if (result.has(state.nodeId)) {
      throw new Error(`native_node_state_is_duplicate:${state.nodeId}`);
    }
    result.set(state.nodeId, {
      ...state,
      kvSessionIds: [...new Set(state.kvSessionIds)].sort(),
    });
  }
  for (const node of topology.nodes) {
    if (!result.has(node.id)) {
      throw new Error(`native_node_state_is_missing:${node.id}`);
    }
  }
  return result;
}

export function engineNodeIsEligible(
  state: NativeFleetNodeState,
  requirement: NativeEngineRouteRequirement | undefined,
): boolean {
  if (!requirement) return true;
  const capability = state.engineCapability;
  if (!capability) return false;
  const requiredPipelineRoles = requirement.requiredRoles.filter(isPipelineRole);
  return (
    capability.validUntilMs > requirement.nowMs
    && capability.descriptorDigest === requirement.descriptorDigest
    && capability.certificationId === requirement.certificationId
    && capability.artifactManifestDigest === requirement.artifactManifestDigest
    && capability.backend === requirement.backend
    && capability.runtimeAbi === requirement.runtimeAbi
    && capability.quantizations.includes(requirement.quantization)
    && capability.maxContextTokens >= requirement.contextTokens
    && capability.maxKvTokens >= requirement.contextTokens
    && capability.maxLayerCount > 0
    && (!requirement.requireFastKernel || capability.fastKernel)
    && (!requirement.requireGraph || capability.graphMode === "available")
    && requiredPipelineRoles.length > 0
    && requiredPipelineRoles.some((role) => capability.roles.includes(role))
    && requirement.requiredRoles
      .filter((role) => !isPipelineRole(role))
      .every((role) => capability.roles.includes(role))
  );
}

function engineCapabilityIsValid(
  capability: NativeEngineNodeCapability | undefined,
): boolean {
  if (!capability) return true;
  return (
    /^sha256:[0-9a-f]{64}$/.test(capability.profileId)
    && /^sha256:[0-9a-f]{64}$/.test(capability.descriptorDigest)
    && /^sha256:[0-9a-f]{64}$/.test(capability.certificationId)
    && /^sha256:[0-9a-f]{64}$/.test(capability.artifactManifestDigest)
    && capability.backend.length > 0
    && capability.runtimeAbi.length > 0
    && capability.quantizations.length > 0
    && new Set(capability.quantizations).size === capability.quantizations.length
    && Number.isSafeInteger(capability.maxContextTokens)
    && capability.maxContextTokens > 0
    && Number.isSafeInteger(capability.maxLayerCount)
    && capability.maxLayerCount > 0
    && Number.isSafeInteger(capability.kvBytesPerToken)
    && capability.kvBytesPerToken > 0
    && Number.isSafeInteger(capability.maxKvTokens)
    && capability.maxKvTokens >= capability.maxContextTokens
    && Number.isFinite(capability.decodeScale)
    && capability.decodeScale > 0
    && Number.isFinite(capability.prefillScale)
    && capability.prefillScale > 0
    && Number.isSafeInteger(capability.validUntilMs)
    && capability.validUntilMs > 0
    && new Set(capability.roles).size === capability.roles.length
  );
}

function congestionAdjustedNode(
  node: DistributionTopology["nodes"][number],
  state: NativeFleetNodeState,
  engineRequirement: NativeEngineRouteRequirement | undefined,
): DistributionTopology["nodes"][number] {
  const occupied = state.activeRequests / Math.max(1, state.capacity);
  const queued = state.queuedRequests / Math.max(1, state.capacity);
  const congestion = 1 + occupied + queued * 2;
  return {
    ...node,
    ...(engineRequirement
      ? {
          maxStageLayers: state.engineCapability!.maxLayerCount,
          stageRoles: state.engineCapability!.roles.filter(isPipelineRole),
          decodeScale: state.engineCapability!.decodeScale * congestion,
          prefillScale: state.engineCapability!.prefillScale * congestion,
        }
      : {}),
    ...(engineRequirement
      ? {}
      : {
          decodeScale: node.decodeScale * congestion,
          prefillScale: node.prefillScale * congestion,
        }),
  };
}

export function nativeEngineNodeCapabilityFromProfile(
  value: EngineRuntimeProfile,
  policy: EngineRuntimeProfilePolicy = {},
): NativeEngineNodeCapability {
  const profile = requireEligibleEngineRuntimeProfile(value, policy);
  return {
    profileId: profile.profileId as `sha256:${string}`,
    descriptorDigest: profile.descriptorDigest as `sha256:${string}`,
    certificationId: profile.certificationId as `sha256:${string}`,
    artifactManifestDigest: profile.artifactManifestDigest as `sha256:${string}`,
    backend: profile.backend,
    runtimeAbi: profile.runtimeAbi,
    quantizations: [profile.quantization],
    maxContextTokens: profile.capacity.contextTokens,
    maxLayerCount: profile.capacity.maxLayerCount,
    kvBytesPerToken: profile.capacity.kvBytesPerToken,
    maxKvTokens: profile.capacity.maxKvTokens,
    decodeScale: profile.costs.decodeScale,
    prefillScale: profile.costs.prefillScale,
    fastKernel: profile.features.fastKernel,
    graphMode: profile.features.graphMode,
    roles: [...profile.features.roles],
    validUntilMs: Date.parse(profile.expiresAt),
  };
}

function isPipelineRole(
  role: NativeEngineNodeCapability["roles"][number],
): role is "head" | "middle" | "tail" {
  return role === "head" || role === "middle" || role === "tail";
}

function physicalBoundaryCost(
  plan: DistributionPlan,
  topology: DistributionTopology,
  stateByNode: ReadonlyMap<string, NativeFleetNodeState>,
): { count: number; ms: number } {
  if (plan.stages.length <= 1) return { count: 0, ms: 0 };
  let count = 0;
  let ms = 0;
  const cycle = [
    ...plan.stages,
    plan.stages[0]!,
  ];
  for (let index = 0; index < cycle.length - 1; index += 1) {
    const from = cycle[index]!.nodeId;
    const to = cycle[index + 1]!.nodeId;
    const fromState = stateByNode.get(from);
    const toState = stateByNode.get(to);
    if (!fromState || !toState) return { count, ms: Number.POSITIVE_INFINITY };
    if (fromState.physicalHostId === toState.physicalHostId) continue;
    const link = topology.links.find(
      (candidate) => candidate.from === from && candidate.to === to,
    );
    if (!link) return { count, ms: Number.POSITIVE_INFINITY };
    count += 1;
    ms += link.oneWayLatencyMs + link.jitterP95Ms;
  }
  return { count, ms };
}

function chainDigest(plan: DistributionPlan, failureDomains: readonly string[]): string {
  const payload = JSON.stringify({
    codec: plan.codec,
    microBatchSize: plan.microBatchSize,
    prefillChunkTokens: plan.prefillChunkTokens,
    stages: plan.stages.map(({ nodeId, layerStart, layerEnd }) => ({
      nodeId,
      layerStart,
      layerEnd,
    })),
    failureDomains,
  });
  return `sha256:${createHash("sha256").update(payload).digest("hex")}`;
}

function infiniteRouteScore(): NativeRouteScore {
  return {
    total: Number.POSITIVE_INFINITY,
    computeMs: Number.POSITIVE_INFINITY,
    queueMs: Number.POSITIVE_INFINITY,
    p95Ms: Number.POSITIVE_INFINITY,
    kvPenaltyMs: Number.POSITIVE_INFINITY,
    physicalBoundaryMs: Number.POSITIVE_INFINITY,
    physicalBoundaryCount: 0,
    failureRisk: Number.POSITIVE_INFINITY,
  };
}

function finiteNonNegative(value: number): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error("native_route_penalty_is_invalid");
  }
  return value;
}

function clampProbability(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
