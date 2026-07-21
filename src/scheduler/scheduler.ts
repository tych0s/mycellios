import type {
  ChatCompletionRequest,
  LlmfitModelAdvisory,
  ModelDeployment,
  RouteStage,
  ScheduledRoute,
  ScheduledRoutePlan,
} from "../contracts/types.js";
import { estimateInputTokens } from "../core/request.js";
import { safeVramBudget } from "../core/tiers.js";
import type { MeshStore, StoredWorker } from "../storage/store.js";

export interface SchedulerOptions {
  connectedWorkerIds?: ReadonlySet<string>;
  excludeWorkerIds?: ReadonlySet<string>;
  now?: number;
  allowPipeline?: boolean;
}

export interface RoutePlanOptions extends SchedulerOptions {
  maxStandbyRoutes?: number;
}

interface Candidate {
  worker: StoredWorker;
  deployment: ModelDeployment;
  score: number;
}

export interface ModelLlmfitSummary {
  advisedReplicas: number;
  bestFit: string;
  quantizations: string[];
  maxEstimatedTokensPerSecond?: number | undefined;
  maxMeasuredTokensPerSecond?: number | undefined;
  minMemoryRequiredMb?: number | undefined;
}

export interface AvailableModel {
  id: string;
  replicas: number;
  pipelines: number;
  llmfit?: ModelLlmfitSummary | undefined;
}

export class Scheduler {
  constructor(private readonly store: MeshStore) {}

  selectRoute(
    request: ChatCompletionRequest,
    sessionId: string,
    options: SchedulerOptions = {},
  ): ScheduledRoute | null {
    const now = options.now ?? Date.now();
    const workers = this.store
      .listSchedulableWorkers(now)
      .filter((worker) => !options.connectedWorkerIds || options.connectedWorkerIds.has(worker.id))
      .filter((worker) => !options.excludeWorkerIds?.has(worker.id));

    const affinity = this.store.getSessionRoute(sessionId, request.model);
    if (affinity && this.routeStillValid(affinity, request, workers)) {
      return { ...affinity, affinityHit: true };
    }

    const replicas = this.replicaCandidates(request, workers);
    if (replicas.length > 0) {
      const chosen = replicas[0]!;
      return {
        routeClass: "replica",
        model: request.model,
        region: chosen.worker.capabilities.region,
        stages: [this.toRouteStage(chosen, 0)],
        score: chosen.score,
        affinityHit: false,
      };
    }

    if (options.allowPipeline) {
      return this.buildPipelineRoute(request, workers);
    }
    return null;
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
      excludeWorkerIds: excludedWorkerIds,
    };
    const primary = this.selectRoute(request, sessionId, routeOptions);
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
    return { primary, standbys };
  }

  listAvailableModels(options: SchedulerOptions = {}): AvailableModel[] {
    const now = options.now ?? Date.now();
    const workers = this.store
      .listSchedulableWorkers(now)
      .filter((worker) => !options.connectedWorkerIds || options.connectedWorkerIds.has(worker.id));
    const models = new Map<
      string,
      { replicas: number; internalPipelines: number; stages: Set<string>; llmfit: LlmfitModelAdvisory[] }
    >();
    for (const worker of workers) {
      for (const deployment of worker.capabilities.deployments) {
        const entry = models.get(deployment.model) ?? {
          replicas: 0,
          internalPipelines: 0,
          stages: new Set<string>(),
          llmfit: [],
        };
        if (deployment.mode === "replica") {
          if (deployment.internalPipeline) entry.internalPipelines += 1;
          else entry.replicas += 1;
          const advisory = worker.capabilities.llmfit?.model;
          if (advisory?.deploymentId === deployment.deploymentId) {
            entry.llmfit.push(advisory);
          }
        }
        if (deployment.mode === "pipeline" && deployment.stage) {
          entry.stages.add(`${deployment.model}:${deployment.stage.total}:${deployment.stage.index}`);
        }
        models.set(deployment.model, entry);
      }
    }
    return [...models.entries()].map(([id, value]) => {
      const llmfit = summarizeLlmfit(value.llmfit);
      return {
        id,
        replicas: value.replicas,
        pipelines: value.internalPipelines + this.countCompletePipelines(value.stages),
        ...(llmfit ? { llmfit } : {}),
      };
    });
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
    const serviceTime = Math.min(1, estimatedMs / 120_000);
    const failure = 1 - Math.max(0, Math.min(1, worker.reliability));
    const rtt = Math.min(1, worker.capabilities.network.coordinatorRttMs / 250);
    const regionPenalty =
      request.preferred_region && request.preferred_region !== worker.capabilities.region ? 0.35 : 0;
    const llmfitPenalty = this.llmfitReplicaPenalty(worker, deployment);
    return (
      0.4 * slaRisk +
      0.25 * serviceTime +
      0.15 * queueRatio +
      0.1 * failure +
      0.1 * Math.min(1, rtt + regionPenalty) +
      llmfitPenalty
    );
  }

  buildPipelineRoute(
    request: ChatCompletionRequest,
    workers: StoredWorker[],
  ): ScheduledRoute | null {
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

    let best: ScheduledRoute | null = null;
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
            const linkPenalty = this.linkPenalty(previous.worker, candidate.worker);
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
      const route: ScheduledRoute = {
        routeClass: "pipeline",
        model: request.model,
        region: path.region,
        stages: path.stages.map((candidate, index) => this.toRouteStage(candidate, index)),
        score: path.cost,
        affinityHit: false,
      };
      if (!best || route.score < best.score) best = route;
    }
    return best;
  }

  private replicaCandidates(
    request: ChatCompletionRequest,
    workers: StoredWorker[],
  ): Candidate[] {
    return workers
      .flatMap((worker) =>
        worker.capabilities.deployments
          .filter((deployment) => deployment.model === request.model && deployment.mode === "replica")
          .filter((deployment) => this.isEligible(worker, deployment, request))
          .map((deployment) => ({
            worker,
            deployment,
            score: this.scoreWorker(worker, deployment, request),
          })),
      )
      .sort((left, right) => left.score - right.score);
  }

  private isEligible(
    worker: StoredWorker,
    deployment: ModelDeployment,
    request: ChatCompletionRequest,
  ): boolean {
    if (worker.status !== "online" || deployment.freeSlots < 1) return false;
    if (estimateInputTokens(request) + (request.max_tokens ?? 256) > deployment.contextLimit) {
      return false;
    }
    const active = this.store.countActiveJobs(worker.id);
    const concurrency = Math.min(
      worker.capabilities.limits.maxConcurrency,
      deployment.maxConcurrency,
    );
    if (active >= concurrency) return false;
    return worker.capabilities.gpus.some(
      (gpu) =>
        safeVramBudget(gpu.offeredVramMb) >= deployment.peakVramMb &&
        gpu.freeOfferedVramMb >= deployment.peakVramMb,
    );
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
          this.isEligible(worker, deployment, request),
      );
    });
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

  private llmfitReplicaPenalty(
    worker: StoredWorker,
    deployment: ModelDeployment,
  ): number {
    if (deployment.mode !== "replica") return 0;
    const advisory = worker.capabilities.llmfit?.model;
    if (!advisory || advisory.deploymentId !== deployment.deploymentId) return 0;
    switch (normalizedFit(advisory.fitLevel)) {
      case "perfect":
        return 0;
      case "good":
        return 0.01;
      case "marginal":
        return 0.06;
      case "tootight":
        // A live deployment remains eligible: llmfit evaluates whole-model
        // fit and may not understand backend offload. This is only a ranking
        // signal, never a hard rejection.
        return 0.15;
      default:
        return 0;
    }
  }

  private linkPenalty(left: StoredWorker, right: StoredWorker): number {
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

function summarizeLlmfit(advisories: LlmfitModelAdvisory[]): ModelLlmfitSummary | null {
  if (advisories.length === 0) return null;
  const fitOrder = new Map([
    ["perfect", 0],
    ["good", 1],
    ["marginal", 2],
    ["tootight", 3],
  ]);
  const best = advisories
    .slice()
    .sort(
      (left, right) =>
        (fitOrder.get(normalizedFit(left.fitLevel)) ?? 99) -
        (fitOrder.get(normalizedFit(right.fitLevel)) ?? 99),
    )[0]!;
  const estimated = advisories
    .map((advisory) => advisory.estimatedTokensPerSecond)
    .filter((value): value is number => value !== undefined);
  const measured = advisories
    .map((advisory) => advisory.measuredTokensPerSecond)
    .filter((value): value is number => value !== undefined);
  const memory = advisories
    .map((advisory) => advisory.memoryRequiredMb)
    .filter((value): value is number => value !== undefined);
  return {
    advisedReplicas: advisories.length,
    bestFit: best.fitLevel,
    quantizations: [
      ...new Set(
        advisories
          .map((advisory) => advisory.bestQuant)
          .filter((value): value is string => value !== undefined),
      ),
    ].sort(),
    ...(estimated.length > 0
      ? { maxEstimatedTokensPerSecond: Math.max(...estimated) }
      : {}),
    ...(measured.length > 0
      ? { maxMeasuredTokensPerSecond: Math.max(...measured) }
      : {}),
    ...(memory.length > 0 ? { minMemoryRequiredMb: Math.min(...memory) } : {}),
  };
}

function normalizedFit(value: string): string {
  return value.toLowerCase().replace(/[\s_-]+/g, "");
}
