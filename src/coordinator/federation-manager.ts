import { createHash } from "node:crypto";
import type {
  FederatedInferenceEvent,
  FederatedModel,
  FederatedNetwork,
  FederatedNetworkId,
  FederatedNode,
  FederatedProviderAdapter,
  FederatedProviderModel,
  FederationSettings,
  FederationSnapshot,
} from "../contracts/federation.js";
import { FEDERATED_NETWORK_IDS } from "../contracts/federation.js";
import type {
  ChatCompletionRequest,
  CompletionResult,
  ScheduledRoute,
} from "../contracts/types.js";
import { AsyncQueue } from "../core/async-queue.js";
import { newId } from "../core/ids.js";
import { inputHashForRequest } from "../core/request.js";
import type { MeshStore } from "../storage/store.js";
import { MeshServiceError, type JobHandle, type JobStreamEvent } from "./mesh-service.js";

const VERIFIED_MODEL_TTL_MS = 30 * 60_000;
const BASE_CIRCUIT_RETRY_MS = 60_000;
const MAX_CIRCUIT_RETRY_MS = 15 * 60_000;
const MAX_ROUTE_ATTEMPTS = 3;
const COMMUNITY_NETWORKS = new Set<FederatedNetworkId>([
  "external-runtime-a",
  "ai-horde",
  "peer-runtime",
]);

interface ProviderRuntime {
  adapter: FederatedProviderAdapter;
  models: Map<string, FederatedModel>;
  nodes: FederatedNode[];
  consecutivePretokenFailures: number;
  circuitOpenCount: number;
  retryAt: number | null;
  lastError: string | null;
  lastCanaryAt: number | null;
  ttftMs: number | null;
  successfulRequests: number;
  failedRequests: number;
  state: FederatedNetwork["actualState"];
}

interface RouteCandidate {
  runtime: ProviderRuntime;
  model: FederatedModel;
  priority: number;
  maximumCostUsd: number;
}

export interface FederationManagerOptions {
  now?: () => number;
  verificationTtlMs?: number;
}

export class FederationManager {
  private readonly runtimes = new Map<FederatedNetworkId, ProviderRuntime>();
  private readonly abortControllers = new Map<string, AbortController>();
  private readonly now: () => number;
  private readonly verificationTtlMs: number;
  private refreshTimer: NodeJS.Timeout | null = null;
  private readonly pendingBackgroundTasks = new Set<Promise<unknown>>();
  private closed = false;

  constructor(
    private readonly store: MeshStore,
    adapters: readonly FederatedProviderAdapter[],
    private readonly featureEnabled: boolean,
    options: FederationManagerOptions = {},
    private readonly rentalProviderConfigured: Partial<Record<FederatedNetworkId, boolean>> = {},
  ) {
    this.now = options.now ?? Date.now;
    this.verificationTtlMs = options.verificationTtlMs ?? VERIFIED_MODEL_TTL_MS;
    this.store.getFederationSettings(featureEnabled);
    for (const [index, id] of FEDERATED_NETWORK_IDS.entries()) {
      this.store.getFederatedNetworkSettings(id, {
        enabled: COMMUNITY_NETWORKS.has(id),
        priority: 100 + index * 10,
        dailyBudgetUsd: 0,
        monthlyBudgetUsd: 0,
      });
    }
    for (const adapter of adapters) {
      this.runtimes.set(adapter.id, {
        adapter,
        models: new Map(),
        nodes: [],
        consecutivePretokenFailures: 0,
        circuitOpenCount: 0,
        retryAt: null,
        lastError: null,
        lastCanaryAt: null,
        ttftMs: null,
        successfulRequests: 0,
        failedRequests: 0,
        state: "disabled",
      });
    }
  }

  start(): void {
    if (this.refreshTimer || this.closed) return;
    this.trackBackground(this.discoverAll());
    this.refreshTimer = setInterval(
      () => this.trackBackground(this.discoverAll()),
      5 * 60_000,
    );
    this.refreshTimer.unref();
  }

  settings(): FederationSettings {
    return this.store.getFederationSettings(this.featureEnabled);
  }

  updateSettings(
    patch: {
      enabled?: boolean | undefined;
      dailyBudgetUsd?: number | undefined;
      monthlyBudgetUsd?: number | undefined;
      autoscalingEnabled?: boolean | undefined;
      maxRentals?: number | undefined;
    },
  ): FederationSettings {
    const current = this.settings();
    return this.store.saveFederationSettings({
      enabled: patch.enabled ?? current.enabled,
      dailyBudgetUsd: patch.dailyBudgetUsd ?? current.dailyBudgetUsd,
      monthlyBudgetUsd: patch.monthlyBudgetUsd ?? current.monthlyBudgetUsd,
      autoscalingEnabled: patch.autoscalingEnabled ?? current.autoscalingEnabled,
      maxRentals: patch.maxRentals ?? current.maxRentals,
    });
  }

  updateNetwork(
    id: FederatedNetworkId,
    patch: {
      enabled?: boolean | undefined;
      priority?: number | undefined;
      dailyBudgetUsd?: number | undefined;
      monthlyBudgetUsd?: number | undefined;
    },
  ) {
    const current = this.store.getFederatedNetworkSettings(id, {
      enabled: COMMUNITY_NETWORKS.has(id),
      priority: 100,
      dailyBudgetUsd: 0,
      monthlyBudgetUsd: 0,
    });
    const updated = this.store.saveFederatedNetworkSettings({
      id,
      enabled: patch.enabled ?? current.enabled,
      priority: patch.priority ?? current.priority,
      dailyBudgetUsd: patch.dailyBudgetUsd ?? current.dailyBudgetUsd,
      monthlyBudgetUsd: patch.monthlyBudgetUsd ?? current.monthlyBudgetUsd,
    });
    const runtime = this.runtimes.get(id);
    if (runtime && !updated.enabled) {
      runtime.state = "draining";
      // Existing streams retain their selected route. New route selection sees
      // the persisted desired state immediately, so draining never swaps a
      // response after its first token.
      runtime.state = "disabled";
    } else if (runtime && updated.enabled) {
      this.trackBackground(this.discover(id));
    }
    return updated;
  }

  async discoverAll(): Promise<void> {
    await Promise.allSettled(
      [...this.runtimes.keys()].map((id) => this.discover(id)),
    );
    await Promise.allSettled(
      [...this.runtimes.entries()]
        .filter(([id, runtime]) =>
          COMMUNITY_NETWORKS.has(id)
          && runtime.models.size > 0
          && this.verifiedModels(runtime).length === 0)
        .map(([id]) => this.probe(id)),
    );
  }

  async discover(id: FederatedNetworkId): Promise<FederatedNetwork> {
    const runtime = this.runtimes.get(id);
    if (!runtime) return this.networkView(id);
    const settings = this.settings();
    const networkSettings = this.store.getFederatedNetworkSettings(id, {
      enabled: COMMUNITY_NETWORKS.has(id),
      priority: 100,
      dailyBudgetUsd: 0,
      monthlyBudgetUsd: 0,
    });
    if (!this.featureEnabled || !settings.enabled || !networkSettings.enabled) {
      runtime.state = "disabled";
      return this.networkView(id);
    }
    if (!runtime.adapter.configured) {
      runtime.state = "blocked";
      runtime.lastError = "Secret not configured";
      return this.networkView(id);
    }
    if (
      runtime.adapter.class !== "community"
      && (effectiveBudget(settings.dailyBudgetUsd, networkSettings.dailyBudgetUsd) <= 0
        || effectiveBudget(settings.monthlyBudgetUsd, networkSettings.monthlyBudgetUsd) <= 0)
    ) {
      runtime.state = "blocked";
      runtime.lastError = "Budget is zero";
      return this.networkView(id);
    }
    if (runtime.retryAt && runtime.retryAt > this.now()) {
      runtime.state = "circuit-open";
      return this.networkView(id);
    }
    runtime.state = "discovering";
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new Error("discovery_timeout")), 15_000);
    try {
      const discovered = await runtime.adapter.discover(controller.signal);
      const previous = runtime.models;
      runtime.models = new Map(discovered.models.map((model) => {
        const existing = previous.get(model.canonicalId);
        return [model.canonicalId, {
          canonicalId: model.canonicalId,
          externalId: model.externalId,
          displayName: model.displayName ?? model.canonicalId,
          verifiedAt: existing?.verifiedAt ?? null,
          advertisedOnly: !existing?.verifiedAt,
          ...(model.contextTokens ? { contextTokens: model.contextTokens } : {}),
          ...(model.estimatedInputUsdPerMillion !== undefined
            ? { estimatedInputUsdPerMillion: model.estimatedInputUsdPerMillion }
            : {}),
          ...(model.estimatedOutputUsdPerMillion !== undefined
            ? { estimatedOutputUsdPerMillion: model.estimatedOutputUsdPerMillion }
            : {}),
        }];
      }));
      runtime.nodes = discovered.nodes.map((node) => ({
        id: anonymizeNodeId(id, node.externalId),
        networkId: id,
        scope: node.scope,
        label: node.label ?? `${networkName(id)} node`,
        models: [...node.models],
        status: node.status,
        reliability: node.reliability ?? 0.5,
        routable: node.models.some((model) => this.isModelVerified(runtime, model)),
        individuallySelectable: node.individuallySelectable ?? false,
        lastVerifiedAt: latestVerifiedAt(runtime, node.models),
        ...(node.capacity ? { capacity: node.capacity } : {}),
      }));
      runtime.state = this.verifiedModels(runtime).length > 0 ? "ready" : "degraded";
      runtime.lastError = this.verifiedModels(runtime).length > 0
        ? null
        : "Discovered capacity has not completed a real generation in the last 30 minutes";
    } catch (error) {
      runtime.state = "error";
      runtime.lastError = errorMessage(error);
    } finally {
      clearTimeout(timer);
    }
    return this.networkView(id);
  }

  async probe(id: FederatedNetworkId, requestedModel?: string): Promise<FederatedNetwork> {
    await this.discover(id);
    const runtime = this.runtimes.get(id);
    if (!runtime || runtime.state === "blocked" || runtime.state === "disabled") {
      return this.networkView(id);
    }
    const model = requestedModel
      ? runtime.models.get(requestedModel)
      : [...runtime.models.values()][0];
    if (!model) {
      runtime.state = "degraded";
      runtime.lastError = requestedModel
        ? `Model not advertised: ${requestedModel}`
        : "No model advertised";
      return this.networkView(id);
    }
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new Error("canary_timeout")), 20_000);
    const request: ChatCompletionRequest = {
      model: model.canonicalId,
      messages: [{ role: "user", content: "Reply with OK." }],
      max_tokens: 16,
      stream: true,
      workload_class: "interactive",
    };
    const startedAt = this.now();
    let firstTokenAt: number | null = null;
    try {
      for await (const event of runtime.adapter.infer({
        requestId: newId("canary"),
        request,
        canonicalModel: model.canonicalId,
        externalModel: model.externalId,
        signal: controller.signal,
      })) {
        if (event.type === "token" && firstTokenAt === null) firstTokenAt = this.now();
        if (event.type === "completed") {
          const verifiedAt = this.now();
          runtime.models.set(model.canonicalId, {
            ...model,
            verifiedAt,
            advertisedOnly: false,
          });
          runtime.lastCanaryAt = verifiedAt;
          runtime.ttftMs = firstTokenAt === null ? verifiedAt - startedAt : firstTokenAt - startedAt;
          runtime.state = "ready";
          runtime.lastError = null;
          runtime.consecutivePretokenFailures = 0;
          runtime.retryAt = null;
          runtime.circuitOpenCount = 0;
          runtime.nodes = runtime.nodes.map((node) => node.models.includes(model.canonicalId)
            ? { ...node, routable: true, lastVerifiedAt: verifiedAt }
            : node);
          return this.networkView(id);
        }
      }
      throw new Error("canary_missing_completion");
    } catch (error) {
      this.recordPretokenFailure(runtime, error);
      return this.networkView(id);
    } finally {
      clearTimeout(timer);
    }
  }

  hasCapacity(request: ChatCompletionRequest): boolean {
    return this.routeCandidates(request).length > 0;
  }

  submit(
    request: ChatCompletionRequest,
    requestedSessionId?: string,
    idempotencyKey?: string,
    maximumAttempts = MAX_ROUTE_ATTEMPTS,
  ): JobHandle {
    const sessionId = requestedSessionId ?? request.session_id ?? newId("ses");
    const jobId = newId("fedjob");
    const requestHash = idempotencyKey ? inputHashForRequest(request) : null;
    if (idempotencyKey) {
      const existing = this.store.getIdempotentJob(idempotencyKey);
      if (existing) {
        throw new MeshServiceError(
          existing.requestHash === requestHash
            ? "idempotency_replayed"
            : "idempotency_key_reused",
          `Idempotency key is already bound to job ${existing.jobId}`,
          409,
        );
      }
    }
    this.store.database.transaction(() => {
      this.store.createJob({
        id: jobId,
        sessionId,
        model: request.model,
        workloadClass: request.workload_class ?? "interactive",
        deadlineAt: this.now() + Math.min(request.deadline_ms ?? 120_000, 120_000),
      });
      if (idempotencyKey) this.store.bindIdempotencyKey(idempotencyKey, requestHash!, jobId);
    });
    const queue = new AsyncQueue<JobStreamEvent>();
    const controller = new AbortController();
    this.abortControllers.set(jobId, controller);
    void this.runRequest(
      jobId,
      sessionId,
      request,
      controller,
      queue,
      Math.max(1, Math.min(MAX_ROUTE_ATTEMPTS, maximumAttempts)),
    );
    return { jobId, sessionId, events: queue };
  }

  cancel(jobId: string): boolean {
    const controller = this.abortControllers.get(jobId);
    if (!controller) return false;
    controller.abort(new Error("cancelled_by_client"));
    return true;
  }

  networks(): FederatedNetwork[] {
    return FEDERATED_NETWORK_IDS.map((id) => this.networkView(id));
  }

  nodes(): FederatedNode[] {
    return [...this.runtimes.values()].flatMap((runtime) => runtime.nodes);
  }

  verifiedModels(): FederatedModel[];
  verifiedModels(runtime: ProviderRuntime): FederatedModel[];
  verifiedModels(runtime?: ProviderRuntime): FederatedModel[] {
    const candidates = runtime ? [runtime] : [...this.runtimes.values()];
    const unique = new Map<string, FederatedModel>();
    for (const candidate of candidates) {
      for (const model of candidate.models.values()) {
        if (!this.isModelVerified(candidate, model.canonicalId)) continue;
        const current = unique.get(model.canonicalId);
        if (!current || (model.verifiedAt ?? 0) > (current.verifiedAt ?? 0)) {
          unique.set(model.canonicalId, { ...model, advertisedOnly: false });
        }
      }
    }
    return [...unique.values()];
  }

  verifiedModelRouteCounts(): Map<string, number> {
    const counts = new Map<string, number>();
    for (const runtime of this.runtimes.values()) {
      for (const model of runtime.models.values()) {
        if (!this.isModelVerified(runtime, model.canonicalId)) continue;
        counts.set(model.canonicalId, (counts.get(model.canonicalId) ?? 0) + 1);
      }
    }
    return counts;
  }

  snapshot(): FederationSnapshot {
    const networks = this.networks();
    const models = this.verifiedModels();
    return {
      enabled: this.featureEnabled && this.settings().enabled,
      notice: "Community and external nodes may process inference content. Mycellios removes user IP, cookies, email, session identifiers and Mycellios names before forwarding; the gateway IP and inference content remain visible to the selected network.",
      readyNetworks: networks.filter((network) => network.actualState === "ready").length,
      routableNodes: this.nodes().filter((node) => node.routable).length,
      verifiedModels: models.length,
      spentTodayUsd: networks.reduce((total, network) => total + network.spentTodayUsd, 0),
      spentMonthUsd: networks.reduce((total, network) => total + network.spentMonthUsd, 0),
      externalRequestsAllowed: true,
    };
  }

  async emergencyStop(): Promise<void> {
    this.updateSettings({ enabled: false, autoscalingEnabled: false });
    for (const controller of this.abortControllers.values()) {
      controller.abort(new Error("federation_emergency_stop"));
    }
    for (const [id, runtime] of this.runtimes) {
      runtime.state = "disabled";
      await runtime.adapter.close?.().catch(() => undefined);
      this.updateNetwork(id, { enabled: false });
    }
  }

  async close(): Promise<void> {
    this.closed = true;
    if (this.refreshTimer) clearInterval(this.refreshTimer);
    this.refreshTimer = null;
    for (const controller of this.abortControllers.values()) controller.abort();
    await Promise.allSettled([...this.pendingBackgroundTasks]);
    await Promise.allSettled(
      [...this.runtimes.values()].map((runtime) => runtime.adapter.close?.()),
    );
  }

  private async runRequest(
    jobId: string,
    sessionId: string,
    request: ChatCompletionRequest,
    controller: AbortController,
    queue: AsyncQueue<JobStreamEvent>,
    maximumAttempts: number,
  ): Promise<void> {
    const candidates = this.routeCandidates(request).slice(0, maximumAttempts);
    if (candidates.length === 0) {
      queue.push({
        type: "failed",
        code: "no_federated_capacity",
        message: `No verified federated route is available for ${request.model}`,
      });
      this.store.setJobStatus(jobId, "failed", "no_federated_capacity");
      queue.close();
      this.abortControllers.delete(jobId);
      return;
    }
    let fallbackReason: string | null = null;
    for (const [index, candidate] of candidates.entries()) {
      const { runtime, model } = candidate;
      const attemptId = newId("froute");
      const reservationId = newId("spend");
      const attemptStartedAt = this.now();
      const networkSettings = this.store.getFederatedNetworkSettings(runtime.adapter.id, {
        enabled: COMMUNITY_NETWORKS.has(runtime.adapter.id),
        priority: 100,
        dailyBudgetUsd: 0,
        monthlyBudgetUsd: 0,
      });
      const globalSettings = this.settings();
      const reserved = candidate.maximumCostUsd <= 0 || this.store.reserveProviderSpend({
        id: reservationId,
        provider: runtime.adapter.id,
        requestId: jobId,
        maximumUsd: candidate.maximumCostUsd,
        dailyBudgetUsd: effectiveBudget(
          globalSettings.dailyBudgetUsd,
          networkSettings.dailyBudgetUsd,
        ),
        monthlyBudgetUsd: effectiveBudget(
          globalSettings.monthlyBudgetUsd,
          networkSettings.monthlyBudgetUsd,
        ),
      });
      if (!reserved) {
        fallbackReason = "budget_exhausted";
        continue;
      }
      this.store.createFederatedRouteAttempt({
        id: attemptId,
        requestId: jobId,
        provider: runtime.adapter.id,
        canonicalModel: model.canonicalId,
        externalModel: model.externalId,
        routeKind: "federated",
        startedAt: attemptStartedAt,
        firstTokenAt: null,
        completedAt: null,
        inputTokens: 0,
        outputTokens: 0,
        reservedCostUsd: candidate.maximumCostUsd,
        actualCostUsd: 0,
        result: "running",
        fallbackReason,
        failureCode: null,
      });
      const route = federatedScheduledRoute(runtime.adapter.id, model.canonicalId);
      queue.push({ type: "accepted", jobId, sessionId, route });
      this.store.setJobStatus(jobId, "running");
      let firstTokenAt: number | null = null;
      try {
        for await (const event of runtime.adapter.infer({
          requestId: jobId,
          request,
          canonicalModel: model.canonicalId,
          externalModel: model.externalId,
          signal: controller.signal,
        })) {
          if (event.type === "heartbeat") {
            queue.push({
              type: "progress",
              phase: "waiting_first_token",
              message: "External network is still processing",
              attempt: index + 1,
            });
            continue;
          }
          if (event.type === "token") {
            if (firstTokenAt === null) {
              firstTokenAt = event.at;
              this.store.updateFederatedRouteAttempt(attemptId, { firstTokenAt });
              this.store.setJobStatus(jobId, "streaming");
            }
            queue.push({ type: "token", token: { index: event.index, text: event.text } });
            continue;
          }
          if (event.type === "completed") {
            const completedAt = this.now();
            const actualCostUsd = event.actualCostUsd ?? candidate.maximumCostUsd;
            const existing = runtime.models.get(model.canonicalId);
            if (existing) {
              runtime.models.set(model.canonicalId, {
                ...existing,
                verifiedAt: completedAt,
                advertisedOnly: false,
              });
            }
            runtime.nodes = runtime.nodes.map((node) =>
              node.models.includes(model.canonicalId)
                ? { ...node, routable: true, lastVerifiedAt: completedAt }
                : node);
            runtime.state = "ready";
            runtime.lastError = null;
            runtime.consecutivePretokenFailures = 0;
            runtime.retryAt = null;
            runtime.circuitOpenCount = 0;
            runtime.successfulRequests += 1;
            runtime.ttftMs = firstTokenAt === null
              ? completedAt - attemptStartedAt
              : firstTokenAt - attemptStartedAt;
            this.store.updateFederatedRouteAttempt(attemptId, {
              completedAt,
              inputTokens: event.inputTokens,
              outputTokens: event.outputTokens,
              actualCostUsd,
              result: "completed",
            });
            this.store.completeJob(jobId, {
              inputTokens: event.inputTokens,
              outputTokens: event.outputTokens,
              ttftMs: firstTokenAt === null
                ? completedAt - attemptStartedAt
                : Math.max(0, firstTokenAt - attemptStartedAt),
              activeMs: firstTokenAt === null ? 0 : Math.max(0, completedAt - firstTokenAt),
            });
            if (candidate.maximumCostUsd > 0) {
              this.store.reconcileProviderSpend(reservationId, actualCostUsd);
            }
            queue.push({
              type: "completed",
              result: completionResult(
                jobId,
                event,
                firstTokenAt,
                completedAt,
                attemptStartedAt,
              ),
            });
            queue.close();
            this.abortControllers.delete(jobId);
            return;
          }
        }
        throw new Error("federated_stream_ended_without_completion");
      } catch (error) {
        const message = errorMessage(error);
        runtime.failedRequests += 1;
        if (candidate.maximumCostUsd > 0) this.store.releaseProviderSpend(reservationId);
        this.store.updateFederatedRouteAttempt(attemptId, {
          completedAt: this.now(),
          result: controller.signal.aborted ? "cancelled" : "failed",
          failureCode: safeFailureCode(message),
        });
        if (firstTokenAt !== null) {
          queue.push({
            type: "failed",
            code: "federated_stream_failed",
            message: "The selected route failed after streaming began; Mycellios did not switch providers.",
          });
          this.store.setJobStatus(jobId, "failed", "federated_stream_failed");
          queue.close();
          this.abortControllers.delete(jobId);
          return;
        }
        this.recordPretokenFailure(runtime, error);
        fallbackReason = safeFailureCode(message);
        if (controller.signal.aborted) break;
      }
    }
    queue.push({
      type: "failed",
      code: controller.signal.aborted ? "cancelled" : "federated_routes_exhausted",
      message: controller.signal.aborted
        ? "The federated request was cancelled"
        : "All eligible routes failed before the first token",
    });
    this.store.setJobStatus(
      jobId,
      controller.signal.aborted ? "cancelled" : "failed",
      controller.signal.aborted ? "cancelled" : "federated_routes_exhausted",
    );
    queue.close();
    this.abortControllers.delete(jobId);
  }

  private routeCandidates(request: ChatCompletionRequest): RouteCandidate[] {
    const now = this.now();
    const globalSettings = this.settings();
    if (!this.featureEnabled || !globalSettings.enabled) return [];
    const alias = isAlias(request.model);
    const candidates: RouteCandidate[] = [];
    for (const runtime of this.runtimes.values()) {
      if (
        runtime.adapter.class === "rental"
        || runtime.retryAt && runtime.retryAt > now
        || !runtime.adapter.configured
      ) continue;
      const settings = this.store.getFederatedNetworkSettings(runtime.adapter.id, {
        enabled: COMMUNITY_NETWORKS.has(runtime.adapter.id),
        priority: 100,
        dailyBudgetUsd: 0,
        monthlyBudgetUsd: 0,
      });
      if (!settings.enabled) continue;
      const eligibleModels = alias
        ? this.aliasModels(request.model, runtime)
        : [...runtime.models.values()].filter((model) => model.canonicalId === request.model);
      for (const model of eligibleModels) {
        if (!this.isModelVerified(runtime, model.canonicalId)) continue;
        const maximumCostUsd = runtime.adapter.estimateMaximumCostUsd({ request, model });
        if (
          maximumCostUsd > 0
          && (effectiveBudget(globalSettings.dailyBudgetUsd, settings.dailyBudgetUsd) <= 0
            || effectiveBudget(globalSettings.monthlyBudgetUsd, settings.monthlyBudgetUsd) <= 0)
        ) continue;
        const costBias = request.workload_class === "batch"
          ? maximumCostUsd * 10_000
          : runtime.adapter.class === "community"
            ? (runtime.ttftMs ?? 10_000) <= 20_000 ? 0 : 1_000
            : 500;
        candidates.push({
          runtime,
          model,
          priority: settings.priority + costBias + (runtime.ttftMs ?? 10_000) / 1_000,
          maximumCostUsd,
        });
      }
    }
    return candidates.sort((left, right) => left.priority - right.priority);
  }

  private aliasModels(alias: string, runtime: ProviderRuntime): FederatedModel[] {
    const verified = this.verifiedModels(runtime);
    const needles = alias === "mycellios-code"
      ? ["code", "coder", "deepseek"]
      : alias === "mycellios-quality"
        ? ["70b", "72b", "large", "pro"]
        : alias === "mycellios-fast"
          ? ["mini", "small", "7b", "8b", "flash"]
          : [];
    const preferred = verified.filter((model) =>
      needles.some((needle) => model.canonicalId.toLowerCase().includes(needle)));
    return preferred.length > 0 ? preferred : verified;
  }

  private isModelVerified(runtime: ProviderRuntime, canonicalId: string): boolean {
    const verifiedAt = runtime.models.get(canonicalId)?.verifiedAt;
    return verifiedAt !== null
      && verifiedAt !== undefined
      && this.now() - verifiedAt <= this.verificationTtlMs;
  }

  private recordPretokenFailure(runtime: ProviderRuntime, error: unknown): void {
    runtime.consecutivePretokenFailures += 1;
    runtime.lastError = errorMessage(error);
    runtime.state = "degraded";
    if (runtime.consecutivePretokenFailures < 3) return;
    runtime.circuitOpenCount += 1;
    runtime.retryAt = this.now() + Math.min(
      BASE_CIRCUIT_RETRY_MS * 2 ** (runtime.circuitOpenCount - 1),
      MAX_CIRCUIT_RETRY_MS,
    );
    runtime.state = "circuit-open";
  }

  private networkView(id: FederatedNetworkId): FederatedNetwork {
    const runtime = this.runtimes.get(id);
    const settings = this.store.getFederatedNetworkSettings(id, {
      enabled: COMMUNITY_NETWORKS.has(id),
      priority: 100,
      dailyBudgetUsd: 0,
      monthlyBudgetUsd: 0,
    });
    const spend = this.store.providerSpend(id);
    const total = (runtime?.successfulRequests ?? 0) + (runtime?.failedRequests ?? 0);
    const globalSettings = this.settings();
    return {
      id,
      name: networkName(id),
      class: runtime?.adapter.class ?? "rental",
      desiredEnabled: settings.enabled,
      actualState: !this.featureEnabled || !globalSettings.enabled
        ? "disabled"
        : runtime?.state ?? (
          settings.enabled && this.rentalConfigured(id) ? "blocked" : "disabled"
        ),
      priority: settings.priority,
      dailyBudgetUsd: settings.dailyBudgetUsd,
      monthlyBudgetUsd: settings.monthlyBudgetUsd,
      configured: runtime?.adapter.configured ?? this.rentalConfigured(id),
      experimental: id === "peer-runtime",
      models: runtime ? [...runtime.models.values()].map((model) => ({
        ...model,
        advertisedOnly: !this.isModelVerified(runtime, model.canonicalId),
      })) : [],
      nodeCount: runtime?.nodes.length ?? 0,
      lastCanaryAt: runtime?.lastCanaryAt ?? null,
      ttftMs: runtime?.ttftMs ?? null,
      reliability: total > 0 ? (runtime?.successfulRequests ?? 0) / total : 0,
      spentTodayUsd: spend.todayUsd,
      spentMonthUsd: spend.monthUsd,
      consecutivePretokenFailures: runtime?.consecutivePretokenFailures ?? 0,
      retryAt: runtime?.retryAt ?? null,
      lastError: runtime?.lastError ?? (
        id === "gpu_cloud" || id === "vast" || id === "clore"
          ? "Rental control is idle until a key, budget and worker image are configured"
          : null
      ),
    };
  }

  private rentalConfigured(_id: FederatedNetworkId): boolean {
    return this.rentalProviderConfigured[_id] ?? false;
  }

  private trackBackground(task: Promise<unknown>): void {
    this.pendingBackgroundTasks.add(task);
    void task.finally(() => this.pendingBackgroundTasks.delete(task));
  }
}

function completionResult(
  jobId: string,
  event: Extract<FederatedInferenceEvent, { type: "completed" }>,
  firstTokenAt: number | null,
  completedAt: number,
  startedAt: number,
): CompletionResult {
  const streamingStartedAt = firstTokenAt ?? completedAt;
  return {
    jobId,
    leaseId: `federated:${jobId}`,
    text: event.text,
    finishReason: normalizeFinishReason(event.finishReason),
    metrics: {
      inputTokens: event.inputTokens,
      outputTokens: event.outputTokens,
      ttftMs: Math.max(0, streamingStartedAt - startedAt),
      activeMs: Math.max(0, completedAt - streamingStartedAt),
    },
  };
}

function normalizeFinishReason(reason: string): CompletionResult["finishReason"] {
  if (reason === "length") return "length";
  if (reason === "cancelled") return "cancelled";
  if (reason === "error") return "error";
  return "stop";
}

function federatedScheduledRoute(
  provider: FederatedNetworkId,
  model: string,
): ScheduledRoute {
  return {
    routeClass: "replica",
    model,
    region: "federated",
    stages: [{
      workerId: `federated:${provider}`,
      deploymentId: `federated:${provider}:${model}`,
      modelDigest: `federated:${createHash("sha256").update(model).digest("hex")}`,
      stageIndex: 0,
      score: 0,
    }],
    score: 0,
    affinityHit: false,
  };
}

function isAlias(model: string): boolean {
  return [
    "mycellios-auto",
    "mycellios-fast",
    "mycellios-code",
    "mycellios-quality",
  ].includes(model);
}

function effectiveBudget(globalBudget: number, providerBudget: number): number {
  if (globalBudget <= 0 || providerBudget <= 0) return 0;
  return Math.min(globalBudget, providerBudget);
}

function latestVerifiedAt(runtime: ProviderRuntime, models: readonly string[]): number | null {
  let latest: number | null = null;
  for (const model of models) {
    const verifiedAt = runtime.models.get(model)?.verifiedAt ?? null;
    if (verifiedAt !== null && (latest === null || verifiedAt > latest)) latest = verifiedAt;
  }
  return latest;
}

function anonymizeNodeId(provider: FederatedNetworkId, externalId: string): string {
  return `fed_${createHash("sha256")
    .update(`${provider}:${externalId}`)
    .digest("hex")
    .slice(0, 20)}`;
}

function networkName(id: FederatedNetworkId): string {
  return ({
    external-runtime-a: "external runtime A",
    "ai-horde": "AI Horde",
    peer-runtime: "peer runtime / Kwaai",
    chutes: "Chutes",
    akashml: "AkashML",
    gpu_cloud: "GpuCloudCloud",
    vast: "Vast.ai",
    clore: "Clore.ai",
  })[id];
}

function safeFailureCode(message: string): string {
  return message.toLowerCase().replace(/[^a-z0-9]+/g, "_").slice(0, 80);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
