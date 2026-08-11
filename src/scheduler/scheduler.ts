import type {
  ChatCompletionRequest,
  ExecutionRouteDecisionRecord,
  ModelDeployment,
  RouteStage,
  ScheduledRoute,
  ScheduledRoutePlan,
} from "../contracts/types.js";
import { estimateInputTokens } from "../core/request.js";
import { safeVramBudget } from "../core/tiers.js";
import type { MeshStore, StoredWorker } from "../storage/store.js";
import {
  selectPreferredRuntimeLinkObservation,
  type RuntimeLinkObservation,
} from "../coordinator/runtime-link-observations.js";
import { deploymentMetricsFromCanaryEvidence } from "../contracts/deployment-canary.js";
import {
  decideExecutionRoute,
  type ExecutionRouteCandidate,
  type ExecutionRouteDecision,
  type ExecutionRouteReasonCode,
} from "../distribution/execution-route-policy.js";

export interface SchedulerOptions {
  connectedWorkerIds?: ReadonlySet<string>;
  excludeWorkerIds?: ReadonlySet<string>;
  now?: number;
  allowPipeline?: boolean;
  /** Worker colocated with the request origin, when the caller can prove it. */
  originWorkerId?: string;
  routePolicy?: {
    requireTrustedIdentity?: boolean;
    requireTrustedBoundaryIdentity?: boolean;
    pinnedBoundaryIdentityIds?: ReadonlySet<string>;
    residencyRegion?: string;
    excludedFailureDomainIds?: ReadonlySet<string>;
    maxNormalizedCost?: number;
  };
}

export interface RoutePlanOptions extends SchedulerOptions {
  maxStandbyRoutes?: number;
}

export interface SchedulerEvidenceOptions {
  /**
   * Current observations for the exact worker-to-worker data path. Production
   * enables strict mode so coordinator RTT and advertised uplink never stand
   * in for a link that has not actually carried a probe.
   */
  runtimeLinkObservations?: (() => readonly RuntimeLinkObservation[]) | undefined;
  strictRuntimeLinks?: boolean | undefined;
}

interface Candidate {
  worker: StoredWorker;
  deployment: ModelDeployment;
  score: number;
}

interface EvaluatedCandidate extends Candidate {
  rejectionReasons: ExecutionRouteReasonCode[];
}

const MIN_RUNTIME_LINK_SAMPLES = 3;
const MIN_RUNTIME_LINK_AVAILABILITY = 0.8;
const MIN_RUNTIME_LINK_CONFIDENCE = 0.8;

export interface AvailableModel {
  id: string;
  replicas: number;
  pipelines: number;
}

export class Scheduler {
  constructor(
    private readonly store: MeshStore,
    private readonly evidence: SchedulerEvidenceOptions = {},
  ) {}

  selectRoute(
    request: ChatCompletionRequest,
    sessionId: string,
    options: SchedulerOptions = {},
  ): ScheduledRoute | null {
    return this.selectExecutionRouteDecision(request, sessionId, options).selected?.route ?? null;
  }

  isRouteCurrentlyEligible(
    request: ChatCompletionRequest,
    sessionId: string,
    route: ScheduledRoute,
    options: SchedulerOptions = {},
  ): boolean {
    const candidateId = routeCandidateId(route);
    return this.selectExecutionRouteDecision(request, sessionId, options)
      .evaluations.some((candidate) => candidate.candidateId === candidateId && candidate.eligible);
  }

  selectExecutionRouteDecision(
    request: ChatCompletionRequest,
    sessionId: string,
    options: SchedulerOptions = {},
  ): ExecutionRouteDecision<ScheduledRoute> {
    const maxNormalizedCost = options.routePolicy?.maxNormalizedCost;
    if (
      maxNormalizedCost !== undefined
      && (!Number.isFinite(maxNormalizedCost) || maxNormalizedCost < 0)
    ) {
      throw new RangeError("maxNormalizedCost must be a finite non-negative number");
    }
    const now = options.now ?? Date.now();
    const workers = this.store
      .listSchedulableWorkers(now)
      .filter((worker) => !options.connectedWorkerIds || options.connectedWorkerIds.has(worker.id))
      .filter((worker) => !options.excludeWorkerIds?.has(worker.id));

    const affinity = this.store.getSessionRoute(sessionId, request.model);
    const affinityEligible = Boolean(
      affinity
      && this.routeStillValid(affinity, request, workers)
      && !this.routeIsSaturated(affinity, workers)
    );

    const replicas = this.replicaCandidatesWithReasons(request, workers);
    const routes: ScheduledRoute[] = replicas.map((candidate) => ({
        routeClass: "replica",
        model: request.model,
        region: candidate.worker.capabilities.region,
        stages: [this.toRouteStage(candidate, 0)],
        score: candidate.score,
        affinityHit: false,
      }));
    const eligibilityRejections = new Map(
      replicas.map((candidate, index) => [
        routeCandidateId(routes[index]!),
        candidate.rejectionReasons,
      ]),
    );
    if (options.allowPipeline) {
      routes.push(...this.buildPipelineRoutes(request, workers, now));
    }
    if (affinityEligible && affinity) {
      routes.push({ ...affinity, affinityHit: false });
    }
    const uniqueRoutes = new Map(routes.map((route) => [routeCandidateId(route), route]));
    const workerById = new Map(workers.map((worker) => [worker.id, worker]));
    const candidates: ExecutionRouteCandidate<ScheduledRoute>[] = [...uniqueRoutes.values()].map(
      (route) => {
        const rejectionReasons = [
          ...(eligibilityRejections.get(routeCandidateId(route)) ?? []),
          ...routePolicyRejections(route, workerById, options),
        ];
        return {
          id: routeCandidateId(route),
          kind: executionRouteKind(route, workerById, options.originWorkerId),
          route,
          score: route.score,
          nodeCount: new Set(route.stages.map((stage) => stage.workerId)).size,
          meetsSlo: estimatedRouteServiceMs(route, workerById, request)
            <= (request.deadline_ms ?? 120_000),
          eligible: rejectionReasons.length === 0,
          rejectionReasons,
        };
      },
    );
    const affinityCandidateId = affinityEligible && affinity
      ? routeCandidateId(affinity)
      : undefined;
    const decision = decideExecutionRoute({
      candidates,
      ...(affinityCandidateId ? { affinityCandidateId } : {}),
      // Existing session semantics retain any healthy, unsaturated exact route.
      maxAffinityScorePenalty: Number.MAX_SAFE_INTEGER,
    });
    if (decision.selected) {
      decision.selected.route = {
        ...decision.selected.route,
        affinityHit: decision.selected.reason === "selected_kv_affinity"
          || decision.selected.candidateId === affinityCandidateId,
      };
    }
    return decision;
  }

  selectRoutePlan(
    request: ChatCompletionRequest,
    sessionId: string,
    options: RoutePlanOptions = {},
  ): ScheduledRoutePlan | null {
    const maxStandbyRoutes = options.maxStandbyRoutes ?? 2;
    if (!Number.isInteger(maxStandbyRoutes) || maxStandbyRoutes < 0 || maxStandbyRoutes > 8) {
      throw new RangeError("maxStandbyRoutes must be an integer between 0 and 8");
    }

    const excludedWorkerIds = new Set(options.excludeWorkerIds ?? []);
    const routeOptions: SchedulerOptions = {
      ...(options.connectedWorkerIds ? { connectedWorkerIds: options.connectedWorkerIds } : {}),
      ...(options.now === undefined ? {} : { now: options.now }),
      ...(options.allowPipeline === undefined ? {} : { allowPipeline: options.allowPipeline }),
      ...(options.originWorkerId === undefined ? {} : { originWorkerId: options.originWorkerId }),
      ...(options.routePolicy === undefined ? {} : { routePolicy: options.routePolicy }),
      excludeWorkerIds: excludedWorkerIds,
    };
    const initialDecision = this.selectExecutionRouteDecision(request, sessionId, routeOptions);
    const primary = initialDecision.selected?.route;
    if (!primary) return null;

    const standbys: ScheduledRoute[] = [];
    this.excludeRouteWorkers(primary, excludedWorkerIds);
    const candidateBudget = this.store.listSchedulableWorkers(options.now ?? Date.now()).length;
    for (
      let candidatesInspected = 0;
      candidatesInspected < candidateBudget && standbys.length < maxStandbyRoutes;
      candidatesInspected += 1
    ) {
      const candidate = this.selectRoute(request, sessionId, routeOptions);
      if (!candidate) break;
      this.excludeRouteWorkers(candidate, excludedWorkerIds);
      if (this.hasExactRecoveryContract(primary, candidate)) standbys.push(candidate);
    }
    return { primary, standbys, decision: decisionRecord(initialDecision, standbys) };
  }

  listAvailableModels(options: SchedulerOptions = {}): AvailableModel[] {
    const now = options.now ?? Date.now();
    const workers = this.store
      .listSchedulableWorkers(now)
      .filter((worker) => !options.connectedWorkerIds || options.connectedWorkerIds.has(worker.id));
    const models = new Map<
      string,
      { replicas: number; internalPipelines: number; stages: Set<string> }
    >();
    for (const worker of workers) {
      for (const deployment of worker.capabilities.deployments) {
        if (!this.deploymentEvidenceIsEligible(worker, deployment, now)) continue;
        if (!this.internalPipelineDependenciesConnected(deployment, workers)) continue;
        const entry = models.get(deployment.model) ?? {
          replicas: 0,
          internalPipelines: 0,
          stages: new Set<string>(),
        };
        if (deployment.mode === "replica") {
          if (deployment.internalPipeline) entry.internalPipelines += 1;
          else entry.replicas += 1;
        }
        if (deployment.mode === "pipeline" && deployment.stage) {
          entry.stages.add(`${deployment.model}:${deployment.stage.total}:${deployment.stage.index}`);
        }
        models.set(deployment.model, entry);
      }
    }
    return [...models.entries()].map(([id, value]) => ({
      id,
      replicas: value.replicas,
      pipelines: value.internalPipelines + this.countCompletePipelines(value.stages),
    }));
  }

  scoreWorker(
    worker: StoredWorker,
    deployment: ModelDeployment,
    request: ChatCompletionRequest,
  ): number {
    const active = this.store.countActiveJobs(worker.id);
    const capacity = Math.max(
      1,
      Math.min(worker.capabilities.limits.maxConcurrency, deployment.maxConcurrency),
    );
    const queueRatio = Math.min(1, active / capacity);
    const generationMs = ((request.max_tokens ?? 256) / deployment.tokensPerSecond) * 1_000;
    const estimatedMs = deployment.ttftMs + generationMs;
    const deadline = request.deadline_ms ?? 120_000;
    const slaRisk = Math.min(1, estimatedMs / deadline);
    const workloadClass = request.workload_class ?? "interactive";
    const interactive = workloadClass === "interactive";
    const firstTokenRisk = Math.min(1, deployment.ttftMs / Math.max(250, deadline * 0.2));
    const throughputServiceTime = Math.min(1, generationMs / 120_000);
    const interactiveServiceTime = Math.min(1, estimatedMs / 120_000);
    const failure = 1 - Math.max(0, Math.min(1, worker.reliability));
    const rtt = Math.min(1, worker.capabilities.network.coordinatorRttMs / 250);
    const regionPenalty =
      request.preferred_region && request.preferred_region !== worker.capabilities.region ? 0.35 : 0;
    if (interactive) {
      return (
        0.3 * slaRisk
        + 0.24 * firstTokenRisk
        + 0.14 * interactiveServiceTime
        + 0.18 * queueRatio
        + 0.08 * failure
        + 0.06 * Math.min(1, rtt + regionPenalty)
      );
    }
    // Batch and benchmark traffic optimize the sustained bottleneck. TTFT is
    // deliberately almost absent so a high-throughput route can differ from
    // the interactive route for the same model and prompt.
    return (
      0.14 * slaRisk
      + 0.46 * throughputServiceTime
      + 0.26 * queueRatio
      + 0.08 * failure
      + 0.06 * Math.min(1, rtt + regionPenalty)
    );
  }

  buildPipelineRoute(
    request: ChatCompletionRequest,
    workers: StoredWorker[],
    now = Date.now(),
  ): ScheduledRoute | null {
    return this.buildPipelineRoutes(request, workers, now)[0] ?? null;
  }

  private buildPipelineRoutes(
    request: ChatCompletionRequest,
    workers: StoredWorker[],
    now: number,
  ): ScheduledRoute[] {
    const candidates = workers.flatMap((worker) =>
      worker.capabilities.deployments
        .filter((deployment) => deployment.model === request.model && deployment.mode === "pipeline")
        .filter((deployment) => this.isEligible(worker, deployment, request))
        .map((deployment) => ({
          worker,
          deployment,
          score: this.scoreWorker(worker, deployment, request),
        })),
    );
    const totals = [...new Set(candidates.map((candidate) => candidate.deployment.stage?.total))]
      .filter((value): value is number => value !== undefined)
      .sort((left, right) => left - right);

    const completeRoutes: ScheduledRoute[] = [];
    for (const total of totals) {
      const stageCandidates = Array.from({ length: total }, (_, index) =>
        candidates.filter(
          (candidate) =>
            candidate.deployment.stage?.total === total &&
            candidate.deployment.stage.index === index,
        ),
      );
      if (stageCandidates.some((stage) => stage.length === 0)) continue;

      type PartialPath = { stages: Candidate[]; cost: number; region: string };
      let paths: PartialPath[] = stageCandidates[0]!.map((candidate) => ({
        stages: [candidate],
        cost: candidate.score,
        region: candidate.worker.capabilities.region,
      }));

      for (let index = 1; index < total; index += 1) {
        const next: PartialPath[] = [];
        for (const path of paths) {
          for (const candidate of stageCandidates[index]!) {
            if (path.stages.some((stage) => stage.worker.id === candidate.worker.id)) continue;
            const sameRegion = path.region === candidate.worker.capabilities.region;
            if (request.workload_class === "interactive" && !sameRegion) continue;
            const previous = path.stages.at(-1)!;
            const linkPenalty = this.linkPenalty(previous.worker, candidate.worker, now);
            if (!Number.isFinite(linkPenalty)) continue;
            next.push({
              stages: [...path.stages, candidate],
              cost: path.cost + candidate.score + linkPenalty,
              region: sameRegion ? path.region : "multi-region",
            });
          }
        }
        paths = next.sort((left, right) => left.cost - right.cost).slice(0, 128);
        if (paths.length === 0) break;
      }

      const path = paths.sort((left, right) => left.cost - right.cost)[0];
      if (!path || path.stages.length !== total) continue;
      const returnPenalty = this.linkPenalty(
        path.stages.at(-1)!.worker,
        path.stages[0]!.worker,
        now,
      );
      if (!Number.isFinite(returnPenalty)) continue;
      const route: ScheduledRoute = {
        routeClass: "pipeline",
        model: request.model,
        region: path.region,
        stages: path.stages.map((candidate, index) => this.toRouteStage(candidate, index)),
        score: path.cost + returnPenalty,
        affinityHit: false,
      };
      completeRoutes.push(route);
    }
    return completeRoutes.sort(
      (left, right) => left.score - right.score
        || left.stages.length - right.stages.length
        || routeCandidateId(left).localeCompare(routeCandidateId(right)),
    );
  }

  private replicaCandidatesWithReasons(
    request: ChatCompletionRequest,
    workers: StoredWorker[],
  ): EvaluatedCandidate[] {
    return workers
      .flatMap((worker) =>
        worker.capabilities.deployments
          .filter((deployment) => deployment.model === request.model && deployment.mode === "replica")
          .map((deployment) => {
            const rejectionReasons = this.eligibilityRejections(worker, deployment, request);
            if (!this.internalPipelineDependenciesConnected(deployment, workers)) {
              rejectionReasons.push("candidate_not_ready");
            }
            return {
              worker,
              deployment,
              score: this.scoreWorker(worker, deployment, request),
              rejectionReasons: [...new Set(rejectionReasons)],
            };
          }),
      )
      .sort((left, right) => left.score - right.score);
  }

  private isEligible(
    worker: StoredWorker,
    deployment: ModelDeployment,
    request: ChatCompletionRequest,
  ): boolean {
    return this.eligibilityRejections(worker, deployment, request).length === 0;
  }

  private eligibilityRejections(
    worker: StoredWorker,
    deployment: ModelDeployment,
    request: ChatCompletionRequest,
  ): ExecutionRouteReasonCode[] {
    const reasons: ExecutionRouteReasonCode[] = [];
    if (worker.status !== "online") reasons.push("candidate_not_ready");
    if (deployment.freeSlots < 1) reasons.push("candidate_capacity_exhausted");
    if (!this.deploymentEvidenceIsEligible(worker, deployment)) {
      reasons.push("candidate_evidence_missing");
    }
    if (estimateInputTokens(request) + (request.max_tokens ?? 256) > deployment.contextLimit) {
      reasons.push("candidate_context_exceeded");
    }
    const active = this.store.countActiveJobs(worker.id);
    const concurrency = Math.min(
      worker.capabilities.limits.maxConcurrency,
      deployment.maxConcurrency,
    );
    if (active >= concurrency) reasons.push("candidate_capacity_exhausted");
    const fitsMemory = worker.capabilities.gpus.some(
      (gpu) =>
        safeVramBudget(gpu.offeredVramMb) >= deployment.peakVramMb &&
        gpu.freeOfferedVramMb >= deployment.peakVramMb,
    );
    if (!fitsMemory) reasons.push("candidate_capacity_exhausted");
    return [...new Set(reasons)];
  }

  private routeStillValid(
    route: ScheduledRoute,
    request: ChatCompletionRequest,
    workers: StoredWorker[],
  ): boolean {
    const byId = new Map(workers.map((worker) => [worker.id, worker]));
    return route.stages.every((stage) => {
      const worker = byId.get(stage.workerId);
      const deployment = worker?.capabilities.deployments.find(
        (candidate) => candidate.deploymentId === stage.deploymentId,
      );
      return Boolean(
        worker &&
          deployment &&
          deployment.modelDigest === stage.modelDigest &&
          this.isEligible(worker, deployment, request) &&
          this.internalPipelineDependenciesConnected(deployment, workers),
      );
    });
  }

  private internalPipelineDependenciesConnected(
    deployment: ModelDeployment,
    connectedWorkers: readonly StoredWorker[],
  ): boolean {
    if (!deployment.internalPipeline) return true;
    const executionStages = deployment.execution?.stages;
    if (!executionStages || executionStages.length === 0) return true;
    const connectedNodeIds = new Set(
      connectedWorkers.flatMap((worker) => {
        const nodeId = worker.capabilities.distributedExecutor?.nodeId;
        return nodeId ? [nodeId] : [];
      }),
    );
    return executionStages.every((stage) => connectedNodeIds.has(stage.nodeId));
  }

  private excludeRouteWorkers(route: ScheduledRoute, excludedWorkerIds: Set<string>): void {
    for (const stage of route.stages) excludedWorkerIds.add(stage.workerId);
  }

  private hasExactRecoveryContract(
    primary: ScheduledRoute,
    candidate: ScheduledRoute,
  ): boolean {
    if (
      primary.routeClass !== candidate.routeClass ||
      primary.model !== candidate.model ||
      primary.stages.length !== candidate.stages.length
    ) {
      return false;
    }
    return primary.stages.every((stage, index) => {
      const candidateStage = candidate.stages[index];
      return Boolean(
        candidateStage &&
          candidateStage.stageIndex === stage.stageIndex &&
          candidateStage.modelDigest === stage.modelDigest,
      );
    });
  }

  private toRouteStage(candidate: Candidate, stageIndex: number): RouteStage {
    return {
      workerId: candidate.worker.id,
      deploymentId: candidate.deployment.deploymentId,
      modelDigest: candidate.deployment.modelDigest,
      stageIndex,
      score: candidate.score,
    };
  }

  private linkPenalty(left: StoredWorker, right: StoredWorker, now: number): number {
    const observations = this.evidence.runtimeLinkObservations?.();
    if (observations) {
      const leftNodeId = left.capabilities.distributedExecutor?.nodeId;
      const rightNodeId = right.capabilities.distributedExecutor?.nodeId;
      if (leftNodeId && rightNodeId) {
        const observation = selectPreferredRuntimeLinkObservation(
          observations.filter((candidate) =>
            candidate.fromNodeId === leftNodeId
            && candidate.toNodeId === rightNodeId
          ),
        );
        if (
          observation
          && observation.measuredAt <= now
          && observation.validUntil > now
          && observation.successfulSamples >= MIN_RUNTIME_LINK_SAMPLES
          && observation.availability >= MIN_RUNTIME_LINK_AVAILABILITY
          && observation.confidence >= MIN_RUNTIME_LINK_CONFIDENCE
        ) {
          const latencyPenalty = Math.min(
            1,
            (observation.rttP95Ms + observation.jitterP95Ms) / 500,
          );
          const bandwidthPenalty = Math.min(1, 20 / observation.goodputMbpsP50);
          const availabilityPenalty = 1 - Math.min(1, observation.availability);
          return (
            0.55 * latencyPenalty
            + 0.25 * bandwidthPenalty
            + 0.2 * availabilityPenalty
          );
        }
      }
      if (this.evidence.strictRuntimeLinks) return Number.POSITIVE_INFINITY;
    } else if (this.evidence.strictRuntimeLinks) {
      return Number.POSITIVE_INFINITY;
    }
    const regionPenalty =
      left.capabilities.region === right.capabilities.region ? 0.01 : 0.5;
    const weakestUplink = Math.min(
      left.capabilities.network.uplinkMbps,
      right.capabilities.network.uplinkMbps,
    );
    const bandwidthPenalty = weakestUplink <= 0 ? 1 : Math.min(1, 20 / weakestUplink);
    const rttPenalty = Math.min(
      1,
      (left.capabilities.network.coordinatorRttMs +
        right.capabilities.network.coordinatorRttMs) /
        500,
    );
    return regionPenalty + 0.15 * bandwidthPenalty + 0.1 * rttPenalty;
  }

  private deploymentEvidenceIsEligible(
    worker: StoredWorker,
    deployment: ModelDeployment,
    now = Date.now(),
  ): boolean {
    if (deployment.adapter !== "mycellios-pipeline") return true;
    if (
      deployment.verificationState !== "verified"
      || deployment.throughputSource !== "measured"
      || !deployment.activationId
      || !deployment.canaryEvidence
    ) {
      return false;
    }
    try {
      const metrics = deploymentMetricsFromCanaryEvidence(
        deployment.canaryEvidence,
        {
          model: deployment.model,
          modelDigest: deployment.modelDigest,
          activationId: deployment.activationId,
          workerId: worker.id,
          now,
        },
      );
      return metrics.tokensPerSecond === deployment.tokensPerSecond
        && metrics.ttftMs === deployment.ttftMs;
    } catch {
      return false;
    }
  }

  private routeIsSaturated(
    route: ScheduledRoute,
    workers: readonly StoredWorker[],
  ): boolean {
    const workerById = new Map(workers.map((worker) => [worker.id, worker]));
    return route.stages.some((stage) => {
      const worker = workerById.get(stage.workerId);
      const deployment = worker?.capabilities.deployments.find(
        (candidate) => candidate.deploymentId === stage.deploymentId,
      );
      if (!worker || !deployment) return true;
      const capacity = Math.max(
        1,
        Math.min(worker.capabilities.limits.maxConcurrency, deployment.maxConcurrency),
      );
      return this.store.countActiveJobs(worker.id) / capacity >= 0.75;
    });
  }

  private countCompletePipelines(stages: Set<string>): number {
    const groups = new Map<string, Set<number>>();
    for (const key of stages) {
      const [model, totalText, indexText] = key.split(":");
      const total = Number(totalText);
      const index = Number(indexText);
      if (!model || !Number.isInteger(total) || !Number.isInteger(index)) continue;
      const groupKey = `${model}:${total}`;
      const indexes = groups.get(groupKey) ?? new Set<number>();
      indexes.add(index);
      groups.set(groupKey, indexes);
    }
    return [...groups.entries()].filter(([key, indexes]) => {
      const total = Number(key.split(":").at(-1));
      return indexes.size === total;
    }).length;
  }
}

function routeCandidateId(route: ScheduledRoute): string {
  return route.stages
    .map((stage) => `${stage.workerId}:${stage.deploymentId}:${stage.modelDigest}:${stage.stageIndex}`)
    .join("|");
}

function executionRouteKind(
  route: ScheduledRoute,
  workerById: ReadonlyMap<string, StoredWorker>,
  originWorkerId: string | undefined,
): ExecutionRouteCandidate<ScheduledRoute>["kind"] {
  if (route.routeClass === "pipeline") return "distributed-pipeline";
  const stage = route.stages.length === 1 ? route.stages[0] : undefined;
  if (!stage) return "remote-replica";
  if (stage.workerId === originWorkerId) return "local-complete";
  const worker = workerById.get(stage.workerId);
  const deployment = worker?.capabilities.deployments.find(
    (candidate) => candidate.deploymentId === stage.deploymentId,
  );
  return deployment?.internalPipeline?.stageCount === 1
    ? "local-complete"
    : "remote-replica";
}

function routePolicyRejections(
  route: ScheduledRoute,
  workerById: ReadonlyMap<string, StoredWorker>,
  options: SchedulerOptions,
): ExecutionRouteReasonCode[] {
  const policy = options.routePolicy;
  if (!policy) return [];
  const workers = route.stages
    .map((stage) => workerById.get(stage.workerId))
    .filter((worker): worker is StoredWorker => worker !== undefined);
  const reasons: ExecutionRouteReasonCode[] = [];
  if (
    policy.requireTrustedIdentity
    && workers.some((worker) => !workerHasTrustedIdentity(worker))
  ) {
    reasons.push("candidate_trust_rejected");
  }
  const boundaryWorkers = route.stages.length === 0
    ? []
    : [route.stages[0]!, route.stages.at(-1)!]
      .map((stage) => workerById.get(stage.workerId))
      .filter((worker): worker is StoredWorker => worker !== undefined);
  if (policy.requireTrustedBoundaryIdentity && boundaryWorkers.some((worker) => !workerHasTrustedIdentity(worker))) {
    reasons.push("candidate_boundary_trust_rejected");
  }
  if (policy.pinnedBoundaryIdentityIds && boundaryWorkers.some(
    (worker) => !worker.identityId || !policy.pinnedBoundaryIdentityIds!.has(worker.identityId),
  )) {
    reasons.push("candidate_boundary_pin_rejected");
  }
  if (
    policy.residencyRegion
    && workers.some((worker) => worker.capabilities.region !== policy.residencyRegion)
  ) {
    reasons.push("candidate_residency_rejected");
  }
  if (
    policy.excludedFailureDomainIds
    && workers.some(
      (worker) => worker.identityId && policy.excludedFailureDomainIds!.has(worker.identityId),
    )
  ) {
    reasons.push("candidate_failure_domain_rejected");
  }
  if (
    policy.maxNormalizedCost !== undefined
    && route.score > policy.maxNormalizedCost
  ) {
    reasons.push("candidate_cost_exceeded");
  }
  return reasons;
}

function workerHasTrustedIdentity(worker: StoredWorker): boolean {
  return (worker.identityKind === "device" || worker.identityKind === "cell") && worker.identityId !== null;
}

function estimatedRouteServiceMs(
  route: ScheduledRoute,
  workerById: ReadonlyMap<string, StoredWorker>,
  request: ChatCompletionRequest,
): number {
  const deployments = route.stages.flatMap((stage) => {
    const worker = workerById.get(stage.workerId);
    const deployment = worker?.capabilities.deployments.find(
      (candidate) => candidate.deploymentId === stage.deploymentId,
    );
    return deployment ? [deployment] : [];
  });
  if (deployments.length !== route.stages.length) return Number.POSITIVE_INFINITY;
  const outputTokens = request.max_tokens ?? 256;
  const ttftMs = deployments.reduce((total, deployment) => total + deployment.ttftMs, 0);
  const generationMs = Math.max(
    ...deployments.map((deployment) => outputTokens / deployment.tokensPerSecond * 1_000),
  );
  return ttftMs + generationMs;
}

function decisionRecord(
  decision: ExecutionRouteDecision<ScheduledRoute>,
  standbys: readonly ScheduledRoute[],
): ExecutionRouteDecisionRecord {
  const compact = (
    selection: NonNullable<ExecutionRouteDecision<ScheduledRoute>["selected"]>,
  ) => ({
    candidateId: selection.candidateId,
    kind: selection.kind,
    score: selection.score,
    nodeCount: selection.nodeCount,
    reason: selection.reason,
  });
  return {
    recommendation: decision.recommendation ? compact(decision.recommendation) : null,
    selected: decision.selected ? compact(decision.selected) : null,
    selectedKind: decision.selectedKind,
    fallbacks: decision.fallbacks.map(compact),
    standbys: standbys.map((route) => {
      const candidateId = routeCandidateId(route);
      const evaluation = decision.evaluations.find(
        (candidate) => candidate.candidateId === candidateId,
      );
      return {
        candidateId,
        kind: evaluation?.kind ?? (route.routeClass === "pipeline"
          ? "distributed-pipeline"
          : "remote-replica"),
        compatibility: "exact-model-revision-and-stage-contract" as const,
        stageCount: route.stages.length,
        modelDigests: [...new Set(route.stages.map((stage) => stage.modelDigest))],
      };
    }),
    reasons: [...decision.reasons],
    evaluations: decision.evaluations.map((evaluation) => ({
      ...evaluation,
      reasons: [...evaluation.reasons],
    })),
  };
}
