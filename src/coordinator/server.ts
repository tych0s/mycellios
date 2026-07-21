import websocket from "@fastify/websocket";
import staticFiles from "@fastify/static";
import Fastify, { type FastifyInstance, type FastifyReply, type FastifyRequest } from "fastify";
import { timingSafeEqual } from "node:crypto";
import { existsSync } from "node:fs";
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
import { MeshStore } from "../storage/store.js";
import { MeshService, MeshServiceError, type JobStreamEvent } from "./mesh-service.js";
import { MobileComputeHub, type MobileWorkerSnapshot } from "./mobile-compute-hub.js";
import type { ModelActivationManager } from "./model-activation-manager.js";
import {
  inspectHubModelCapacity,
  requestedModelCapacityViews,
  shouldQueueAutomaticActivation,
} from "./model-catalog.js";
import { WorkerHub } from "./worker-hub.js";

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

export async function createCoordinator(
  config: CoordinatorConfig,
  options: { logger?: boolean; activationManager?: ModelActivationManager } = {},
): Promise<CoordinatorRuntime> {
  const app = Fastify({ logger: options.logger ?? false, bodyLimit: 2 * 1024 * 1024 });
  if (config.networkToken) {
    const expectedToken = config.networkToken;
    app.addHook("onRequest", async (request, reply) => {
      const path = request.url.split("?", 1)[0] ?? request.url;
      if (!path.startsWith("/internal/v1/") && !path.startsWith("/v1/")) return;
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
    void reply.code(401).send({ error: { code: "invalid_model_admin_token" } });
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
      return reply.redirect("/updates/win32/x64/mycellios-setup.exe?v=0.2.6");
    });
  }
  const hub = new WorkerHub(store);
  hub.attach(app);
  const mobileHub = new MobileComputeHub({
    joinToken: config.mobileJoinToken,
    expertArtifactsPath: config.mobileExpertArtifactsPath,
  });
  mobileHub.attach(app);
  const service = new MeshService(store, scheduler, hub, config.requestTimeoutMs);
  const activationManager = options.activationManager;
  await activationManager?.initialize();
  const launchRequestedModel = (model: import("../storage/store.js").StoredRequestedModel) => {
    if (!activationManager || activationManager.isManaging(model.id) || activationManager.isBusy()) return;
    void activationManager.activate(model).catch((error: unknown) => {
      if (store.getRequestedModel(model.id)) {
        store.setRequestedModelActivationError(
          model.id,
          error instanceof Error ? error.message : String(error),
        );
      }
    });
  };
  const reconcileRequestedModels = () => {
    const activeModelIds = new Set(
      scheduler
        .listAvailableModels({ connectedWorkerIds: hub.connectedWorkerIds() })
        .map((model) => model.id),
    );
    const requests = store.listRequestedModels();
    const views = requestedModelCapacityViews({
      requests,
      workers: store.listWorkers(),
      connectedWorkerIds: hub.connectedWorkerIds(),
      activeModelIds,
      ...(activationManager
        ? { executionNodesForModel: (modelId: string) => activationManager.capacityNodesForModel(modelId) }
        : {}),
      activationAvailable: activationManager !== undefined,
    });
    for (const view of views) {
      const stored = requests.find((request) => request.id === view.id)!;
      if (shouldQueueAutomaticActivation(view)) {
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
  };
  const benchmarkRoot = process.env.MYCELLIOS_BENCHMARK_ROOT?.trim() || process.cwd();
  let benchmarkRunInFlight: ReturnType<typeof runAndPersistRealSuite> | null = null;
  const staleTimer = setInterval(() => {
    store.markStaleWorkers();
    hub.closeStaleConnections();
    void activationManager?.refresh();
    reconcileRequestedModels();
  }, 5_000);
  staleTimer.unref();

  app.get("/health", async () => {
    const workers = store.listWorkers();
    const mobileWorkers = mobileHub.listWorkers();
    return {
      status: "ok",
      version: "0.2.0",
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
    };
  });

  app.get("/public/v1/snapshot", async () => {
    reconcileRequestedModels();
    return publicSnapshot(store, scheduler, hub, mobileHub, activationManager);
  });

  app.post("/public/v1/requested-models", async (request, reply) => {
    if (!authorizeModelMutation(request, reply)) return;
    const body = requestedModelCreateSchema.parse(request.body);
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
    const snapshot = publicSnapshot(store, scheduler, hub, mobileHub, activationManager);
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
      ...store.listWorkers().map((worker) => ({
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
      llmfit: worker.capabilities.llmfit,
      reliability: worker.reliability,
      jobsCompleted: worker.jobsCompleted,
      lastSeenAt: new Date(worker.lastSeenAt).toISOString(),
      })),
      ...mobileHub.listWorkers().map(mobileDashboardWorker),
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
  if (landingAssetsPath) {
    app.get("/downloads/macos-arm64", async (_request, reply) => {
      reply.header("Cache-Control", "no-cache, no-store, must-revalidate");
      return reply.redirect("/downloads/mycellios-macos-arm64.dmg?v=0.2.6");
    });
    app.get("/downloads/macos-x64", async (_request, reply) => {
      reply.header("Cache-Control", "no-cache, no-store, must-revalidate");
      return reply.redirect("/downloads/mycellios-macos-x64.dmg?v=0.2.6");
    });
    app.get("/downloads/linux-deb", async (_request, reply) => {
      reply.header("Cache-Control", "no-cache, no-store, must-revalidate");
      return reply.redirect("/downloads/mycellios-linux-x64.deb?v=0.2.6");
    });
    app.get("/downloads/linux-rpm", async (_request, reply) => {
      reply.header("Cache-Control", "no-cache, no-store, must-revalidate");
      return reply.redirect("/downloads/mycellios-linux-x64.rpm?v=0.2.6");
    });
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

function resolveLandingAssetsPath(configured: string | undefined): string | null {
  const candidates = [configured, resolve(process.cwd(), "landing-dist")].filter(
    (candidate): candidate is string => Boolean(candidate),
  );
  return candidates.find((candidate) => existsSync(resolve(candidate, "index.html"))) ?? null;
}

function dashboardWorkers(store: MeshStore, hub: WorkerHub, mobileHub: MobileComputeHub) {
  return [
    ...store.listWorkers().map((worker) => ({
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
      reliability: worker.reliability,
      jobsCompleted: worker.jobsCompleted,
      lastSeenAt: new Date(worker.lastSeenAt).toISOString(),
      kind: "desktop" as const,
    })),
    ...mobileHub.listWorkers().map((worker) => ({
      ...mobileDashboardWorker(worker),
      kind: "browser" as const,
    })),
  ];
}

function publicSnapshot(
  store: MeshStore,
  scheduler: Scheduler,
  hub: WorkerHub,
  mobileHub: MobileComputeHub,
  activationManager?: ModelActivationManager,
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
    version: "0.2.0",
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
