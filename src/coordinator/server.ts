import websocket from "@fastify/websocket";
import staticFiles from "@fastify/static";
import Fastify, { type FastifyInstance, type FastifyReply, type FastifyRequest } from "fastify";
import { timingSafeEqual } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { z, ZodError } from "zod";
import { runAndPersistRealSuite } from "../benchlab/run.js";
import { loadBenchmarkRuns } from "../benchlab/history.js";
import {
  chatCompletionRequestSchema,
  workerRegistrationSchema,
} from "../contracts/schemas.js";
import type { ChatCompletionRequest } from "../contracts/types.js";
import type { CoordinatorConfig } from "../core/config.js";
import { Scheduler } from "../scheduler/scheduler.js";
import { MeshDatabase } from "../storage/database.js";
import { MeshStore, type StoredRequestedModel, type StoredWorker } from "../storage/store.js";
import { MeshService, MeshServiceError, type JobStreamEvent } from "./mesh-service.js";
import { MobileComputeHub, type MobileWorkerSnapshot } from "./mobile-compute-hub.js";
import { verifyGitHubReleaseUploadToken } from "./github-oidc.js";
import type { ModelActivationManager } from "./model-activation-manager.js";
import {
  inspectHubModelCapacity,
  requestedModelCapacityViews,
  searchHubModelCatalog,
  shouldQueueAutomaticActivation,
  type ModelActivationProgressEvent,
} from "./model-catalog.js";
import {
  parseReleaseChunkMetadata,
  storeReleaseChunk,
  type ReleaseAssetChannel,
} from "./release-upload.js";
import { WorkerHub } from "./worker-hub.js";
import {
  modelHasGpuFallback,
  verifiedGpuCapacityCanRepairModel,
} from "./connected-executor-activation.js";

export function automaticActivationFailureIsTransient(message: string): boolean {
  const normalized = message.toLowerCase();
  if (normalized.startsWith("automatic_activation_retries_exhausted:")) return false;
  return [
    "distributed_worker_disconnected:",
    "distributed_worker_not_connected:",
    "managed_launch_agent_is_unavailable:",
    "worker_tunnel_prepare_timeout",
    "gpu_only_runtime_not_ready",
    "gpu_model_stage_unavailable_after_retries:",
    "launch_readiness_timeout:",
    "coordinator_worker_registration_timeout",
  ].some((marker) => normalized.includes(marker));
}

export const DEFAULT_AUTOMATIC_ACTIVATION_RETRY_DELAYS_MS = [
  5_000,
  15_000,
  30_000,
  120_000,
  300_000,
] as const;

export interface AutomaticActivationRetryState {
  retryCount: number;
  nextAttemptAt: number;
  lastError: string;
  updatedAt: number;
  launching: boolean;
}

export function nextAutomaticActivationRetry(
  retriesStarted: number,
  message: string,
  now = Date.now(),
  delays: readonly number[] = DEFAULT_AUTOMATIC_ACTIVATION_RETRY_DELAYS_MS,
): AutomaticActivationRetryState | null {
  const delay = delays[retriesStarted];
  if (delay === undefined) return null;
  return {
    retryCount: retriesStarted,
    nextAttemptAt: now + Math.max(0, delay),
    lastError: message,
    updatedAt: now,
    launching: false,
  };
}

export interface CoordinatorRuntime {
  app: FastifyInstance;
  database: MeshDatabase;
  store: MeshStore;
  scheduler: Scheduler;
  hub: WorkerHub;
  mobileHub: MobileComputeHub;
  service: MeshService;
  close(): Promise<void>;
}

export interface CoordinatorActivationContext {
  store: MeshStore;
  hub: WorkerHub;
}

export async function createCoordinator(
  config: CoordinatorConfig,
  options: {
    logger?: boolean;
    activationManager?: ModelActivationManager;
    activationManagerFactory?: (context: CoordinatorActivationContext) => ModelActivationManager;
    releaseTokenVerifier?: (token: string) => Promise<unknown>;
    mobileDisconnectedRetentionMs?: number;
    automaticActivationRetryDelaysMs?: readonly number[];
  } = {},
): Promise<CoordinatorRuntime> {
  const app = Fastify({ logger: options.logger ?? false, bodyLimit: 2 * 1024 * 1024 });
  app.addHook("onRequest", async (_request, reply) => {
    // The public UI and mobile worker must never be embeddable as drive-by
    // compute. Apply the policy to static and dynamic responses so it remains
    // true even when a reverse proxy does not add security headers.
    reply.header("Content-Security-Policy", "frame-ancestors 'none'; object-src 'none'; base-uri 'self'");
    reply.header("X-Frame-Options", "DENY");
    reply.header("X-Content-Type-Options", "nosniff");
    reply.header("Referrer-Policy", "strict-origin-when-cross-origin");
  });
  const internalToken = config.internalToken ?? config.modelAdminToken ?? config.networkToken;
  app.addHook("onRequest", async (request, reply) => {
    const path = request.url.split("?", 1)[0] ?? request.url;
    if (!path.startsWith("/internal/v1/mobile/experts/")) return;
    if (!internalToken) {
      return reply.code(503).send({
        error: {
          code: "internal_administration_not_configured",
          message: "Mobile expert administration requires MYCELLIOS_INTERNAL_TOKEN.",
        },
      });
    }
    const received = parseBearerToken(request.headers.authorization);
    if (!received || !constantTimeEqual(received, internalToken)) {
      return reply.code(401).send({ error: { code: "invalid_internal_token" } });
    }
  });
  if (config.networkToken) {
    const expectedToken = config.networkToken;
    app.addHook("onRequest", async (request, reply) => {
      const path = request.url.split("?", 1)[0] ?? request.url;
      if (!path.startsWith("/internal/v1/") && !path.startsWith("/v1/")) return;
      if (path.startsWith("/internal/v1/releases/")) return;
      // Mobile expert administration has its own stronger control-plane
      // credential above. Requiring both secrets in one Authorization header
      // would make the route impossible to use when the tokens differ.
      if (path.startsWith("/internal/v1/mobile/experts/")) return;
      const received = parseBearerToken(request.headers.authorization);
      if (!received || !constantTimeEqual(received, expectedToken)) {
        return reply.code(401).send({ error: { code: "invalid_network_token" } });
      }
    });
  }
  const authorizeModelMutation = (request: FastifyRequest, reply: FastifyReply): boolean => {
    const expected = config.modelAdminToken;
    if (!expected && isLoopbackAddress(request.ip)) return true;
    if (!expected) {
      void reply.code(503).send({
        error: {
          code: "model_administration_not_configured",
          message: "Remote model administration requires MYCELLIOS_MODEL_ADMIN_TOKEN.",
        },
      });
      return false;
    }
    const received = parseBearerToken(request.headers.authorization);
    if (received && constantTimeEqual(received, expected)) return true;
    void reply.code(401).send({
      error: {
        code: "invalid_model_admin_token",
        message: "The network administrator token is missing or invalid.",
      },
    });
    return false;
  };
  app.addContentTypeParser(
    "application/octet-stream",
    { parseAs: "buffer", bodyLimit: 512 * 1024 * 1024 },
    (_request, body, done) => done(null, body),
  );
  const database = new MeshDatabase(config.databasePath);
  const store = new MeshStore(database);
  const scheduler = new Scheduler(store);
  await app.register(websocket, { options: { maxPayload: 10 * 1024 * 1024 } });
  const mobileAssetsPath = resolveMobileAssetsPath(config.mobileAssetsPath);
  if (mobileAssetsPath) {
    await app.register(staticFiles, {
      root: mobileAssetsPath,
      prefix: "/mobile/",
      decorateReply: false,
      index: "index.html",
      cacheControl: false,
      setHeaders: setPublicAssetCacheHeaders,
    });
    app.get("/mobile", async (_request, reply) => reply.redirect("/mobile/"));
  }
  const desktopUpdatesPath = resolveDesktopUpdatesPath(config.desktopUpdatesPath);
  const releaseDownloadsPath = resolveReleaseDownloadsPath(
    config.releaseDownloadsPath,
    config.landingAssetsPath,
  );
  const publicAssetVersion = readPackageVersion();
  if (desktopUpdatesPath) {
    await app.register(staticFiles, {
      root: desktopUpdatesPath,
      prefix: "/updates/win32/x64/",
      decorateReply: false,
      cacheControl: false,
      setHeaders: setPublicAssetCacheHeaders,
    });
    app.get("/downloads/windows", async (_request, reply) => {
      reply.header("Cache-Control", "no-cache, no-store, must-revalidate");
      return reply.redirect(`/updates/win32/x64/mycellios-setup.exe?v=${publicAssetVersion}`);
    });
  }
  if (releaseDownloadsPath) {
    await app.register(staticFiles, {
      root: releaseDownloadsPath,
      prefix: "/downloads/",
      decorateReply: false,
      cacheControl: false,
      setHeaders: setPublicAssetCacheHeaders,
    });
  }
  const hub = new WorkerHub(store);
  hub.attach(app);
  const mobileHub = new MobileComputeHub({
    joinToken: config.mobileJoinToken,
    expertArtifactsPath: config.mobileExpertArtifactsPath,
    disconnectedRetentionMs: options.mobileDisconnectedRetentionMs,
  });
  mobileHub.attach(app);
  const service = new MeshService(store, scheduler, hub, config.requestTimeoutMs);
  const activationManager = options.activationManager ?? options.activationManagerFactory?.({ store, hub });
  await activationManager?.initialize();
  const automaticRepairState = new Map<string, { attempts: number; nextAttemptAt: number }>();
  const automaticRepairInFlight = new Set<string>();
  const automaticActivationRetryDelaysMs = (
    options.automaticActivationRetryDelaysMs
    ?? DEFAULT_AUTOMATIC_ACTIVATION_RETRY_DELAYS_MS
  ).map((delay) => Math.max(0, Math.round(delay)));
  const automaticActivationRetryState = new Map<string, AutomaticActivationRetryState>();
  const automaticActivationRetryProgressForModel = (
    modelId: string,
  ): readonly ModelActivationProgressEvent[] => {
    const retry = automaticActivationRetryState.get(modelId);
    if (!retry) return [];
    const retryNumber = retry.launching ? retry.retryCount : retry.retryCount + 1;
    const secondsRemaining = Math.max(0, Math.ceil((retry.nextAttemptAt - Date.now()) / 1_000));
    const waitMessage = secondsRemaining > 0
      ? `Automatic retry ${retryNumber} of ${automaticActivationRetryDelaysMs.length} starts in ${secondsRemaining}s.`
      : `Automatic retry ${retryNumber} of ${automaticActivationRetryDelaysMs.length} is ready and waiting for healthy capacity and a free activation slot.`;
    return [{
      phase: retry.launching ? "retrying" : "retry_wait",
      message: retry.launching
        ? `Automatic retry ${retryNumber} of ${automaticActivationRetryDelaysMs.length} is starting.`
        : `A node became unavailable during startup. ${waitMessage}`,
      at: new Date(retry.updatedAt).toISOString(),
      state: "running",
      details: [retry.lastError],
    }];
  };
  const activationProgressForModel = (
    modelId: string,
  ): readonly ModelActivationProgressEvent[] => {
    const managerProgress = activationManager?.activationProgressForModel?.(modelId) ?? [];
    const retryProgress = automaticActivationRetryProgressForModel(modelId);
    if (retryProgress.length === 0) return managerProgress;
    return automaticActivationRetryState.get(modelId)?.launching
      ? [...retryProgress, ...managerProgress]
      : [...managerProgress, ...retryProgress];
  };
  const activationStatusMessageForModel = (modelId: string): string | null => {
    const retry = automaticActivationRetryState.get(modelId);
    if (!retry) return null;
    const retryNumber = retry.launching ? retry.retryCount : retry.retryCount + 1;
    if (retry.launching) {
      return `Automatic retry ${retryNumber} of ${automaticActivationRetryDelaysMs.length} is starting.`;
    }
    const secondsRemaining = Math.max(0, Math.ceil((retry.nextAttemptAt - Date.now()) / 1_000));
    return secondsRemaining > 0
      ? `A node disconnected during startup. Automatic retry ${retryNumber} of ${automaticActivationRetryDelaysMs.length} starts in ${secondsRemaining}s.`
      : `Automatic retry ${retryNumber} of ${automaticActivationRetryDelaysMs.length} is ready and waiting for healthy capacity and a free activation slot.`;
  };
  const handleRequestedModelActivationFailure = (
    modelId: string,
    error: unknown,
  ): void => {
    if (!store.getRequestedModel(modelId)) return;
    const message = error instanceof Error ? error.message : String(error);
    if (!automaticActivationFailureIsTransient(message)) {
      automaticActivationRetryState.delete(modelId);
      store.setRequestedModelActivationError(modelId, message);
      return;
    }
    const previous = automaticActivationRetryState.get(modelId);
    const retriesStarted = previous?.launching
      ? previous.retryCount
      : previous?.retryCount ?? 0;
    const nextRetry = nextAutomaticActivationRetry(
      retriesStarted,
      message,
      Date.now(),
      automaticActivationRetryDelaysMs,
    );
    if (!nextRetry) {
      automaticActivationRetryState.delete(modelId);
      store.setRequestedModelActivationError(
        modelId,
        `automatic_activation_retries_exhausted:${retriesStarted}:${message}`,
      );
      return;
    }
    automaticActivationRetryState.set(modelId, nextRetry);
    store.setRequestedModelActivation(modelId, false);
  };
  const launchRequestedModel = (model: StoredRequestedModel): boolean => {
    if (!activationManager || activationManager.isManaging(model.id) || activationManager.isBusy()) {
      return false;
    }
    try {
      void activationManager.activate(model)
        .then(() => {
          automaticActivationRetryState.delete(model.id);
        })
        .catch((error: unknown) => {
          handleRequestedModelActivationFailure(model.id, error);
        });
      return true;
    } catch (error) {
      handleRequestedModelActivationFailure(model.id, error);
      return true;
    }
  };
  const reconcileAutomaticGpuRepair = (
    model: StoredRequestedModel,
    activeModelIds: ReadonlySet<string>,
    workers: readonly StoredWorker[],
    connectedWorkerIds: ReadonlySet<string>,
  ) => {
    if (!activationManager || !model.autoActivate || !activeModelIds.has(model.id)) return;
    const degraded = modelHasGpuFallback(model.id, workers, connectedWorkerIds);
    if (!degraded) {
      automaticRepairState.delete(model.id);
      return;
    }
    if (
      automaticRepairInFlight.has(model.id)
      || !activationManager.isManaging(model.id)
      || !verifiedGpuCapacityCanRepairModel(model, workers, connectedWorkerIds)
    ) return;
    const now = Date.now();
    const previous = automaticRepairState.get(model.id) ?? { attempts: 0, nextAttemptAt: 0 };
    if (now < previous.nextAttemptAt) return;
    const repairDelays = [30_000, 120_000, 300_000] as const;
    const delay = repairDelays[Math.min(previous.attempts, repairDelays.length - 1)]!;
    automaticRepairState.set(model.id, {
      attempts: previous.attempts + 1,
      nextAttemptAt: now + delay,
    });
    automaticRepairInFlight.add(model.id);
    void (async () => {
      // Keep the CPU fallback online until verified GPU capacity is ready, then
      // recycle the managed topology. The normal activation path runs a fresh
      // health check and real inference canary before publishing it again.
      const stopped = await activationManager.deactivate(model.id);
      if (!stopped) return;
      const latest = store.getRequestedModel(model.id);
      if (!latest?.autoActivate) return;
      store.setRequestedModelActivation(model.id, true);
      launchRequestedModel(store.getRequestedModel(model.id)!);
    })().catch((error: unknown) => {
      app.log.warn({ modelId: model.id, error }, "automatic GPU repair could not be started");
    }).finally(() => {
      automaticRepairInFlight.delete(model.id);
    });
  };
  const reconcileRequestedModels = () => {
    const workers = store.listWorkers();
    const connectedWorkerIds = hub.connectedWorkerIds();
    const activeModelIds = new Set(
      scheduler
        .listAvailableModels({ connectedWorkerIds })
        .map((model) => model.id),
    );
    let requests = store.listRequestedModels();
    const now = Date.now();
    for (const request of requests) {
      if (
        request.autoActivate
        && request.activationError
        && automaticActivationFailureIsTransient(request.activationError)
      ) {
        if (!automaticActivationRetryState.has(request.id)) {
          const retry = nextAutomaticActivationRetry(
            0,
            request.activationError,
            now,
            automaticActivationRetryDelaysMs,
          );
          if (retry) automaticActivationRetryState.set(request.id, retry);
        }
        store.clearRequestedModelActivationError(request.id);
      }
    }
    requests = store.listRequestedModels();
    const views = requestedModelCapacityViews({
      requests,
      workers,
      connectedWorkerIds,
      activeModelIds,
      ...(activationManager
        ? { executionNodesForModel: (modelId: string) => activationManager.capacityNodesForModel(modelId) }
        : {}),
      ...(activationManager
        ? {
            activationProgressForModel,
            activationStatusMessageForModel,
          }
        : {}),
      activationAvailable: activationManager !== undefined,
    });
    for (const view of views) {
      const stored = requests.find((request) => request.id === view.id)!;
      if (view.status === "active") {
        automaticActivationRetryState.delete(view.id);
      }
      if (shouldQueueAutomaticActivation(view)) {
        const retry = automaticActivationRetryState.get(view.id);
        if (retry) {
          if (
            retry.launching
            || Date.now() < retry.nextAttemptAt
            || !activationManager
            || activationManager.isManaging(view.id)
            || activationManager.isBusy()
          ) {
            continue;
          }
          const launchingRetry: AutomaticActivationRetryState = {
            ...retry,
            retryCount: retry.retryCount + 1,
            nextAttemptAt: Date.now(),
            updatedAt: Date.now(),
            launching: true,
          };
          automaticActivationRetryState.set(view.id, launchingRetry);
          store.setRequestedModelActivation(view.id, true);
          if (!launchRequestedModel(store.getRequestedModel(view.id)!)) {
            automaticActivationRetryState.set(view.id, retry);
            store.setRequestedModelActivation(view.id, false);
          }
          continue;
        }
        store.setRequestedModelActivation(view.id, true);
        launchRequestedModel(store.getRequestedModel(view.id)!);
      } else if (
        view.status === "activating" &&
        stored.activationRequestedAt !== null &&
        activationManager &&
        !activationManager.isManaging(view.id) &&
        !activationManager.isBusy()
      ) {
        launchRequestedModel(stored);
      } else if (
        stored.activationRequestedAt !== null &&
        (view.status === "waiting_capacity" || view.status === "incompatible" || view.status === "failed")
      ) {
        store.setRequestedModelActivation(view.id, false);
      }
    }
    for (const model of requests) {
      reconcileAutomaticGpuRepair(model, activeModelIds, workers, connectedWorkerIds);
    }
  };
  const benchmarkRoot = process.env.MYCELLIOS_BENCHMARK_ROOT?.trim() || process.cwd();
  let benchmarkRunInFlight: ReturnType<typeof runAndPersistRealSuite> | null = null;
  const staleTimer = setInterval(() => {
    store.markStaleWorkers();
    hub.closeStaleConnections();
    mobileHub.expireDisconnectedWorkers();
    void activationManager?.refresh();
    reconcileRequestedModels();
  }, 5_000);
  staleTimer.unref();

  app.get("/health", async () => {
    const workers = store.listWorkers();
    const mobileWorkers = mobileHub.listWorkers();
    return {
      status: "ok",
      version: readPackageVersion(),
      revision: readBuildRevision(),
      workers: {
        registered: workers.length + mobileWorkers.length,
        connected: hub.connectedWorkerIds().size + mobileHub.connectedCount(),
        online:
          workers.filter((worker) => worker.status === "online").length +
          mobileHub.onlineCount(),
        mobile: mobileWorkers.length,
      },
      mobilePwa: mobileAssetsPath ? "/mobile/" : null,
      landing: config.landingAssetsPath ? "/" : null,
      desktopUpdates: desktopUpdatesPath ? "/updates/win32/x64/" : null,
      downloads: releaseDownloadsPath ? "/downloads/" : null,
      features: { distributedActivation: activationManager !== undefined },
    };
  });

  app.get("/public/v1/snapshot", async () => {
    reconcileRequestedModels();
    return publicSnapshot(store, scheduler, hub, mobileHub, activationManager, {
      activationProgressForModel,
      activationStatusMessageForModel,
    });
  });

  app.get("/public/v1/huggingface-models", async (request, reply) => {
    const { q, cursor, sort, limit } = huggingFaceModelSearchSchema.parse(request.query);
    try {
      return await searchHubModelCatalog(q, fetch, { ...(cursor ? { cursor } : {}), sort, limit });
    } catch (error) {
      return reply.code(502).send({
        error: {
          code: "huggingface_catalog_unavailable",
          message: error instanceof Error ? error.message : String(error),
        },
      });
    }
  });

  app.post("/public/v1/requested-models", async (request, reply) => {
    if (!authorizeModelMutation(request, reply)) return;
    const body = requestedModelCreateSchema.parse(request.body);
    automaticActivationRetryState.delete(body.id);
    const stored = store.upsertRequestedModel({
      id: body.id,
      source: body.source,
      revision: body.revision,
      contextTokens: body.contextTokens,
      minimumNodes: body.minimumNodes,
      autoActivate: body.autoActivate,
    });
    try {
      const profile = await inspectHubModelCapacity({
        source: stored.source,
        revision: stored.revision,
        contextTokens: stored.contextTokens,
        minimumNodes: stored.minimumNodes,
      });
      store.setRequestedModelProfile(stored.id, profile as unknown as Record<string, unknown>, null);
    } catch (error) {
      store.setRequestedModelProfile(
        stored.id,
        null,
        error instanceof Error ? error.message : String(error),
      );
    }
    reconcileRequestedModels();
    const snapshot = publicSnapshot(store, scheduler, hub, mobileHub, activationManager, {
      activationProgressForModel,
      activationStatusMessageForModel,
    });
    return reply.code(201).send({
      model: snapshot.requestedModels.find((model) => model.id === stored.id),
    });
  });

  app.delete("/public/v1/requested-models/:modelId", async (request, reply) => {
    if (!authorizeModelMutation(request, reply)) return;
    const { modelId } = requestedModelParamsSchema.parse(request.params);
    await activationManager?.deactivate(modelId);
    if (!store.removeRequestedModel(modelId)) {
      return reply.code(404).send({ error: { code: "requested_model_not_found" } });
    }
    automaticActivationRetryState.delete(modelId);
    automaticRepairState.delete(modelId);
    automaticRepairInFlight.delete(modelId);
    return { removed: true, modelId };
  });

  app.get("/internal/v1/model-activation-requests", async () => {
    reconcileRequestedModels();
    return {
      data: store.listRequestedModels()
        .filter((model) => model.activationRequestedAt !== null)
        .map((model) => ({
          id: model.id,
          source: model.source,
          revision: model.revision,
          contextTokens: model.contextTokens,
          minimumNodes: model.minimumNodes,
          profile: model.profile,
          requestedAt: new Date(model.activationRequestedAt!).toISOString(),
        })),
    };
  });

  app.get("/local/v1/benchmarks", async (request, reply) => {
    if (!isLoopbackAddress(request.ip)) {
      return reply.code(403).send({
        error: { code: "local_access_required", message: "Benchmark history is available on the coordinator host only." },
      });
    }
    return { runs: loadBenchmarkRuns(benchmarkRoot).toReversed() };
  });

  app.post("/local/v1/benchmarks/run", async (request, reply) => {
    if (!isLoopbackAddress(request.ip)) {
      return reply.code(403).send({
        error: { code: "local_access_required", message: "Benchmarks can only be started on the coordinator host." },
      });
    }
    if (benchmarkRunInFlight) {
      return reply.code(409).send({
        error: { code: "benchmark_in_progress", message: "A real benchmark is already running." },
      });
    }
    const body = benchmarkRunRequestSchema.parse(request.body ?? {});
    benchmarkRunInFlight = runAndPersistRealSuite({
      cwd: benchmarkRoot,
      coordinatorUrl: `http://127.0.0.1:${config.port}`,
      ...(body.version ? { version: body.version } : {}),
      ...(body.label ? { label: body.label } : {}),
    });
    try {
      const result = await benchmarkRunInFlight;
      return { run: result.run };
    } catch (error) {
      return reply.code(503).send({
        error: {
          code: "benchmark_unavailable",
          message: error instanceof Error ? error.message : String(error),
        },
      });
    } finally {
      benchmarkRunInFlight = null;
    }
  });

  app.delete("/public/v1/workers/:workerId", async (request, reply) => {
    const { workerId } = workerIdParamsSchema.parse(request.params);
    const removed = hub.removeWorker(workerId) || mobileHub.removeWorker(workerId);
    if (!removed) return reply.code(404).send({ error: { code: "worker_not_found" } });
    return { removed: true, workerId };
  });

  app.post("/public/v1/workers/clear-offline", async () => ({
    removed: store.deregisterOfflineWorkers() + mobileHub.removeOfflineWorkers(),
  }));

  app.post("/internal/v1/workers/register", async (request, reply) => {
    const registration = workerRegistrationSchema.parse(request.body);
    const worker = store.registerWorker(registration);
    return reply.code(201).send({ workerId: worker.id, protocolVersion: 1 });
  });

  app.get("/internal/v1/workers", async () => ({
    data: [
      ...store.listWorkers().filter((worker) => storedWorkerIsVisible(worker, hub)).map((worker) => ({
      id: worker.id,
      status: worker.status,
      connected: hub.isConnected(worker.id),
      region: worker.capabilities.region,
      offeredVramMb: worker.capabilities.gpus.reduce(
        (sum, gpu) => sum + gpu.offeredVramMb,
        0,
      ),
      gpus: worker.capabilities.gpus,
      deployments: worker.capabilities.deployments,
      executionNodeId: worker.capabilities.distributedExecutor?.nodeId,
      computeMode: worker.capabilities.distributedExecutor?.computeMode,
      llmfit: worker.capabilities.llmfit,
      reliability: worker.reliability,
      jobsCompleted: worker.jobsCompleted,
      lastSeenAt: new Date(worker.lastSeenAt).toISOString(),
      kind: storedWorkerKind(worker),
      })),
      ...mobileHub.listWorkers().map((worker) => ({
        ...mobileDashboardWorker(worker),
        kind: "browser" as const,
      })),
    ],
  }));

  app.get("/v1/models", async () => {
    const models = scheduler.listAvailableModels({ connectedWorkerIds: hub.connectedWorkerIds() });
    return {
      object: "list",
      data: models.map((model) => ({
        id: model.id,
        object: "model",
        created: 0,
        owned_by: "mycellios",
        x_replicas: model.replicas,
        x_pipelines: model.pipelines,
        ...(model.llmfit
          ? {
              x_llmfit: {
                advised_replicas: model.llmfit.advisedReplicas,
                best_fit: model.llmfit.bestFit,
                quantizations: model.llmfit.quantizations,
                max_estimated_tokens_per_second:
                  model.llmfit.maxEstimatedTokensPerSecond,
                max_measured_tokens_per_second:
                  model.llmfit.maxMeasuredTokensPerSecond,
                min_memory_required_mb: model.llmfit.minMemoryRequiredMb,
              },
            }
          : {}),
      })),
    };
  });

  app.post("/v1/chat/completions", async (request, reply) => {
    const parsed = chatCompletionRequestSchema.parse(request.body) as ChatCompletionRequest;
    const idempotencyKey = parseIdempotencyKey(request.headers["idempotency-key"]);
    const handle = service.submit(parsed, parsed.session_id, idempotencyKey);
    reply.header("x-network-request-id", handle.jobId);
    reply.header("x-network-session-id", handle.sessionId);

    if (parsed.stream) {
      reply.hijack();
      reply.raw.writeHead(200, {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache, no-transform",
        connection: "keep-alive",
        "x-accel-buffering": "no",
        "x-network-request-id": handle.jobId,
        "x-network-session-id": handle.sessionId,
      });
      let finished = false;
      reply.raw.once("close", () => {
        if (!finished) service.cancel(handle.jobId);
      });
      for await (const event of handle.events) {
        writeOpenAiEvent(reply.raw, event, parsed.model, handle.jobId);
      }
      finished = true;
      reply.raw.write("data: [DONE]\n\n");
      reply.raw.end();
      return;
    }

    let result: Extract<JobStreamEvent, { type: "completed" }> | null = null;
    let routeClass = "replica";
    let affinityHit = false;
    for await (const event of handle.events) {
      if (event.type === "accepted") {
        routeClass = event.route.routeClass;
        affinityHit = event.route.affinityHit;
      }
      if (event.type === "failed") {
        throw new MeshServiceError(event.code, event.message, 502);
      }
      if (event.type === "completed") result = event;
    }
    if (!result) {
      throw new MeshServiceError("missing_result", "Worker stream ended without a result", 502);
    }
    reply.header("x-route-class", routeClass);
    return {
      id: handle.jobId,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1_000),
      model: parsed.model,
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: result.result.text },
          finish_reason: result.result.finishReason,
        },
      ],
      usage: {
        prompt_tokens: result.result.metrics.inputTokens,
        completion_tokens: result.result.metrics.outputTokens,
        total_tokens: result.result.metrics.inputTokens + result.result.metrics.outputTokens,
      },
      x_network: {
        session_id: handle.sessionId,
        route_class: routeClass,
        affinity_hit: affinityHit,
        ttft_ms: result.result.metrics.ttftMs,
        active_ms: result.result.metrics.activeMs,
      },
    };
  });

  app.get("/v1/requests/:jobId", async (request, reply) => {
    const { jobId } = jobIdParamsSchema.parse(request.params);
    const job = store.getJob(jobId);
    if (!job) return reply.code(404).send({ error: { code: "not_found" } });
    return {
      id: job.id,
      session_id: job.sessionId,
      model: job.model,
      status: job.status,
      input_tokens: job.inputTokens,
      output_tokens: job.outputTokens,
      failure_code: job.failureCode,
      deadline_at: new Date(job.deadlineAt).toISOString(),
      created_at: new Date(job.createdAt).toISOString(),
      updated_at: new Date(job.updatedAt).toISOString(),
    };
  });

  app.post("/v1/requests/:jobId/cancel", async (request, reply) => {
    const { jobId } = jobIdParamsSchema.parse(request.params);
    if (!service.cancel(jobId)) {
      return reply.code(404).send({ error: { code: "not_found_or_terminal" } });
    }
    return reply.code(202).send({ id: jobId, status: "cancelled" });
  });

  const landingAssetsPath = resolveLandingAssetsPath(config.landingAssetsPath);
  const releaseTokenVerifier = options.releaseTokenVerifier ?? verifyGitHubReleaseUploadToken;
  app.put("/internal/v1/releases/:channel/:fileName", async (request, reply) => {
    const authorization = parseBearerToken(request.headers.authorization);
    if (!authorization) {
      return reply.code(401).send({ error: { code: "release_upload_token_missing" } });
    }
    try {
      await releaseTokenVerifier(authorization);
    } catch {
      return reply.code(401).send({ error: { code: "release_upload_token_invalid" } });
    }
    const params = z.object({
      channel: z.enum(["updates", "downloads"]),
      fileName: z.string().min(1).max(160),
    }).parse(request.params);
    if (!Buffer.isBuffer(request.body)) {
      return reply.code(400).send({ error: { code: "release_chunk_body_invalid" } });
    }
    try {
      const root = releaseAssetRoot(
        params.channel,
        config.desktopUpdatesPath,
        config.releaseDownloadsPath,
        config.landingAssetsPath,
      );
      const result = await storeReleaseChunk({
        root,
        channel: params.channel as ReleaseAssetChannel,
        fileName: params.fileName,
        metadata: parseReleaseChunkMetadata(request.headers),
        body: request.body,
      });
      return reply.code(result.complete ? 201 : 202).send(result);
    } catch (error) {
      return reply.code(400).send({
        error: {
          code: "release_chunk_rejected",
          message: error instanceof Error ? error.message : "release_chunk_rejected",
        },
      });
    }
  });
  if (releaseDownloadsPath) {
    app.get("/downloads/macos-arm64", async (_request, reply) => {
      reply.header("Cache-Control", "no-cache, no-store, must-revalidate");
      return reply.redirect(`/downloads/mycellios-macos-arm64.dmg?v=${publicAssetVersion}`);
    });
    app.get("/downloads/macos-x64", async (_request, reply) => {
      reply.header("Cache-Control", "no-cache, no-store, must-revalidate");
      return reply.redirect(`/downloads/mycellios-macos-x64.dmg?v=${publicAssetVersion}`);
    });
    app.get("/downloads/linux-deb", async (_request, reply) => {
      reply.header("Cache-Control", "no-cache, no-store, must-revalidate");
      return reply.redirect(`/downloads/mycellios-linux-x64.deb?v=${publicAssetVersion}`);
    });
    app.get("/downloads/linux-rpm", async (_request, reply) => {
      reply.header("Cache-Control", "no-cache, no-store, must-revalidate");
      return reply.redirect(`/downloads/mycellios-linux-x64.rpm?v=${publicAssetVersion}`);
    });
  }
  if (landingAssetsPath) {
    await app.register(staticFiles, {
      root: landingAssetsPath,
      prefix: "/",
      decorateReply: true,
      index: "index.html",
      cacheControl: false,
      setHeaders: setPublicAssetCacheHeaders,
    });
    for (const path of ["/network", "/admin", "/join", "/downloads"] as const) {
      app.get(path, async (_request, reply) => reply.sendFile("index.html"));
    }
  }

  app.setErrorHandler((error, _request, reply) => {
    if (error instanceof ZodError) {
      return reply.code(400).send({
        error: { code: "invalid_request", message: "Request validation failed", details: error.issues },
      });
    }
    if (error instanceof MeshServiceError) {
      return reply.code(error.statusCode).send({ error: { code: error.code, message: error.message } });
    }
    app.log.error(error);
    return reply.code(500).send({
      error: { code: "internal_error", message: "The coordinator could not process the request" },
    });
  });

  return {
    app,
    database,
    store,
    scheduler,
    hub,
    mobileHub,
    service,
    async close() {
      clearInterval(staleTimer);
      hub.close();
      mobileHub.close();
      await activationManager?.close();
      await app.close();
      database.close();
    },
  };
}

const benchmarkRunRequestSchema = z.object({
  version: z.string().trim().min(1).max(80).optional(),
  label: z.string().trim().min(1).max(160).optional(),
});

const huggingFaceModelSearchSchema = z.object({
  q: z.string().trim().max(80).default(""),
  cursor: z.string().trim().max(4_096).optional(),
  sort: z.enum(["downloads", "likes", "lastModified"]).default("downloads"),
  limit: z.coerce.number().int().min(10).max(100).default(50),
});

const requestedModelCreateSchema = z.object({
  id: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
  source: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
  revision: z.string().min(1).max(512).nullable().default(null),
  contextTokens: z.number().int().min(128).max(1_048_576).default(4_096),
  minimumNodes: z.number().int().min(2).max(8).default(2),
  autoActivate: z.boolean().default(true),
}).strict();

export function isLoopbackAddress(address: string): boolean {
  const normalized = address.toLowerCase();
  return normalized === "::1" || normalized === "127.0.0.1" || normalized.startsWith("127.") || normalized.startsWith("::ffff:127.");
}

function resolveMobileAssetsPath(configured: string | undefined): string | null {
  const candidates = [configured, resolve(process.cwd(), "mobile-dist")].filter(
    (candidate): candidate is string => Boolean(candidate),
  );
  return candidates.find((candidate) => existsSync(resolve(candidate, "index.html"))) ?? null;
}

function setPublicAssetCacheHeaders(
  reply: FastifyReply,
  filePath: string,
): void {
  const normalized = filePath.replaceAll("\\", "/");
  if (
    normalized.endsWith("/index.html") ||
    normalized.endsWith("/sw.js") ||
    normalized.endsWith("/manifest.webmanifest") ||
    normalized.includes("/downloads/") ||
    normalized.endsWith("/RELEASES") ||
    normalized.endsWith("-setup.exe") ||
    normalized.endsWith("/latest.json")
  ) {
    reply.header("Cache-Control", "no-cache, no-store, must-revalidate");
    return;
  }
  if (normalized.includes("/assets/")) {
    reply.header("Cache-Control", "public, max-age=31536000, immutable");
    return;
  }
  if (normalized.endsWith(".nupkg")) {
    reply.header("Cache-Control", "public, max-age=31536000, immutable");
    return;
  }
  reply.header("Cache-Control", "public, max-age=3600");
}

function resolveDesktopUpdatesPath(configured: string | undefined): string | null {
  const candidates = [configured, resolve(process.cwd(), "updates", "win32", "x64")].filter(
    (candidate): candidate is string => Boolean(candidate),
  );
  return candidates.find((candidate) => existsSync(resolve(candidate, "RELEASES"))) ?? null;
}

function releaseAssetRoot(
  channel: ReleaseAssetChannel,
  configuredUpdates: string | undefined,
  configuredDownloads: string | undefined,
  configuredLanding: string | undefined,
): string {
  if (channel === "updates") {
    return resolve(configuredUpdates ?? resolve(process.cwd(), "updates", "win32", "x64"));
  }
  if (configuredDownloads) return resolve(configuredDownloads);
  return resolve(
    configuredLanding ?? resolve(process.cwd(), "landing-dist"),
    "downloads",
  );
}

function resolveReleaseDownloadsPath(
  configuredDownloads: string | undefined,
  configuredLanding: string | undefined,
): string | null {
  const candidates = [
    configuredDownloads,
    configuredLanding ? resolve(configuredLanding, "downloads") : undefined,
    resolve(process.cwd(), "landing-dist", "downloads"),
  ].filter((candidate): candidate is string => Boolean(candidate));
  return candidates.find((candidate) => existsSync(candidate)) ?? null;
}

function readPackageVersion(): string {
  const environmentVersion = process.env.npm_package_version?.trim();
  if (environmentVersion && /^\d+\.\d+\.\d+$/.test(environmentVersion)) {
    return environmentVersion;
  }
  try {
    const metadata = JSON.parse(
      readFileSync(resolve(process.cwd(), "package.json"), "utf8"),
    ) as { version?: unknown };
    if (typeof metadata.version === "string" && /^\d+\.\d+\.\d+$/.test(metadata.version)) {
      return metadata.version;
    }
  } catch {
    // Packaged clients do not need public download redirects.
  }
  return "0";
}

function readBuildRevision(): string | null {
  const environmentRevision = process.env.MYCELLIOS_REVISION?.trim();
  if (environmentRevision && /^[0-9a-f]{7,40}$/i.test(environmentRevision)) {
    return environmentRevision.toLowerCase();
  }
  try {
    const revision = readFileSync(resolve(process.cwd(), "REVISION"), "utf8").trim();
    return /^[0-9a-f]{7,40}$/i.test(revision) ? revision.toLowerCase() : null;
  } catch {
    return null;
  }
}

function resolveLandingAssetsPath(configured: string | undefined): string | null {
  const candidates = [configured, resolve(process.cwd(), "landing-dist")].filter(
    (candidate): candidate is string => Boolean(candidate),
  );
  return candidates.find((candidate) => existsSync(resolve(candidate, "index.html"))) ?? null;
}

function dashboardWorkers(store: MeshStore, hub: WorkerHub, mobileHub: MobileComputeHub) {
  return [
    ...store.listWorkers().filter((worker) => storedWorkerIsVisible(worker, hub)).map((worker) => ({
      id: worker.id,
      status: worker.status,
      connected: hub.isConnected(worker.id),
      region: worker.capabilities.region,
      offeredVramMb: worker.capabilities.gpus.reduce(
        (sum, gpu) => sum + gpu.offeredVramMb,
        0,
      ),
      gpus: worker.capabilities.gpus,
      deployments: worker.capabilities.deployments,
      executionNodeId: worker.capabilities.distributedExecutor?.nodeId,
      computeMode: worker.capabilities.distributedExecutor?.computeMode,
      reliability: worker.reliability,
      jobsCompleted: worker.jobsCompleted,
      lastSeenAt: new Date(worker.lastSeenAt).toISOString(),
      kind: storedWorkerKind(worker),
    })),
    ...mobileHub.listWorkers().map((worker) => ({
      ...mobileDashboardWorker(worker),
      kind: "browser" as const,
    })),
  ];
}

function storedWorkerKind(worker: StoredWorker): "desktop" | "cell" {
  if (
    worker.identityKind === "cell" ||
    worker.capabilities.gpus.some((gpu) => gpu.vendor === "sidecar-cell")
  ) {
    return "cell";
  }
  return "desktop";
}

function storedWorkerIsVisible(worker: StoredWorker, hub: WorkerHub): boolean {
  return storedWorkerKind(worker) !== "cell" || hub.isConnected(worker.id);
}

function publicSnapshot(
  store: MeshStore,
  scheduler: Scheduler,
  hub: WorkerHub,
  mobileHub: MobileComputeHub,
  activationManager?: ModelActivationManager,
  activationPresentation?: {
    activationProgressForModel(modelId: string): readonly ModelActivationProgressEvent[];
    activationStatusMessageForModel(modelId: string): string | null;
  },
) {
  const workers = dashboardWorkers(store, hub, mobileHub);
  const models = scheduler.listAvailableModels({ connectedWorkerIds: hub.connectedWorkerIds() });
  const requestedModels = requestedModelCapacityViews({
    requests: store.listRequestedModels(),
    workers: store.listWorkers(),
    connectedWorkerIds: hub.connectedWorkerIds(),
    activeModelIds: new Set(models.map((model) => model.id)),
    ...(activationManager
      ? { executionNodesForModel: (modelId: string) => activationManager.capacityNodesForModel(modelId) }
      : {}),
    ...(activationPresentation
      ? {
          activationProgressForModel: activationPresentation.activationProgressForModel,
          activationStatusMessageForModel: activationPresentation.activationStatusMessageForModel,
        }
      : activationManager?.activationProgressForModel
        ? { activationProgressForModel: (modelId: string) => activationManager.activationProgressForModel!(modelId) }
        : {}),
    activationAvailable: activationManager !== undefined,
  });
  const jobs = store.listJobs(100).map((job) => ({
    id: job.id,
    model: job.model,
    status: job.status,
    workerId: job.workerId,
    inputTokens: job.inputTokens,
    outputTokens: job.outputTokens,
    failureCode: job.failureCode,
    createdAt: new Date(job.createdAt).toISOString(),
    updatedAt: new Date(job.updatedAt).toISOString(),
  }));
  return {
    capturedAt: new Date().toISOString(),
    version: readPackageVersion(),
    summary: {
      registered: workers.length,
      connected: workers.filter((worker) => worker.connected).length,
      online: workers.filter((worker) => worker.status === "online").length,
      mobile: workers.filter((worker) => worker.kind === "browser").length,
      offeredVramMb: workers.reduce((sum, worker) => sum + worker.offeredVramMb, 0),
      completedJobs: jobs.filter((job) => job.status === "completed").length,
    },
    workers,
    models: models.map((model) => ({
      id: model.id,
      replicas: model.replicas,
      pipelines: model.pipelines,
    })),
    requestedModels,
    jobs,
  };
}

function mobileDashboardWorker(worker: MobileWorkerSnapshot) {
  return {
    id: worker.id,
    status: worker.status,
    connected: worker.connected,
    region: worker.region,
    offeredVramMb: 0,
    reliability:
      worker.completedTasks + worker.failedTasks === 0
        ? 1
        : worker.completedTasks / (worker.completedTasks + worker.failedTasks),
    jobsCompleted: worker.completedTasks,
    lastSeenAt: worker.lastSeenAt,
    gpus: [
      {
        id: `mobile-${worker.id}`,
        vendor: worker.backend === "webgpu" ? "WebGPU" : "Browser CPU",
        model: `${worker.name} · ${worker.backend.toUpperCase()}`,
        physicalVramMb: 0,
        sharedMemoryMb: worker.capabilities.deviceMemoryGb
          ? Math.round(worker.capabilities.deviceMemoryGb * 1_024)
          : undefined,
        offeredVramMb: 0,
        freeOfferedVramMb: 0,
        utilizationPct: worker.connected && worker.visible ? 100 : 0,
      },
    ],
    deployments: [],
    mobile: {
      platform: worker.platform,
      backend: worker.backend,
      performanceLevel: worker.performanceLevel,
      wakeLock: worker.wakeLock,
      estimatedGflops: worker.estimatedGflops,
      verifiedTasks: worker.verifiedTasks,
      residentExperts: worker.residentExperts,
    },
  };
}

function parseIdempotencyKey(received: string | string[] | undefined): string | undefined {
  const value = Array.isArray(received) ? received[0] : received;
  if (value === undefined) return undefined;
  if (!/^[\x21-\x7E]{1,128}$/.test(value)) {
    throw new MeshServiceError(
      "invalid_idempotency_key",
      "Idempotency-Key must contain 1-128 printable ASCII characters",
      400,
    );
  }
  return value;
}

const jobIdParamsSchema = z.object({ jobId: z.string().min(1).max(128) }).strict();
const workerIdParamsSchema = z.object({ workerId: z.string().min(1).max(256) }).strict();
const requestedModelParamsSchema = z.object({
  modelId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
}).strict();

function parseBearerToken(header: string | undefined): string | undefined {
  const match = /^Bearer (.+)$/i.exec(header ?? "");
  return match?.[1]?.trim() || undefined;
}

function constantTimeEqual(received: string, expected: string): boolean {
  const left = Buffer.from(received);
  const right = Buffer.from(expected);
  return left.length === right.length && timingSafeEqual(left, right);
}

function writeOpenAiEvent(
  stream: NodeJS.WritableStream,
  event: JobStreamEvent,
  model: string,
  jobId: string,
): void {
  if (event.type === "accepted") {
    stream.write(
      `data: ${JSON.stringify({
        id: jobId,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1_000),
        model,
        choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
        x_network: {
          session_id: event.sessionId,
          route_class: event.route.routeClass,
          affinity_hit: event.route.affinityHit,
        },
      })}\n\n`,
    );
  } else if (event.type === "token") {
    stream.write(
      `data: ${JSON.stringify({
        id: jobId,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1_000),
        model,
        choices: [{ index: 0, delta: { content: event.token.text }, finish_reason: null }],
        x_network: { token_index: event.token.index },
      })}\n\n`,
    );
  } else if (event.type === "completed") {
    stream.write(
      `data: ${JSON.stringify({
        id: jobId,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1_000),
        model,
        choices: [{ index: 0, delta: {}, finish_reason: event.result.finishReason }],
        usage: {
          prompt_tokens: event.result.metrics.inputTokens,
          completion_tokens: event.result.metrics.outputTokens,
          total_tokens: event.result.metrics.inputTokens + event.result.metrics.outputTokens,
        },
        x_network: {
          ttft_ms: event.result.metrics.ttftMs,
          active_ms: event.result.metrics.activeMs,
        },
      })}\n\n`,
    );
  } else if (event.type === "failed") {
    stream.write(`data: ${JSON.stringify({ error: { code: event.code, message: event.message } })}\n\n`);
  }
}
