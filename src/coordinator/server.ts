import websocket from "@fastify/websocket";
import Fastify, { type FastifyInstance } from "fastify";
import { z, ZodError } from "zod";
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
import { WorkerHub } from "./worker-hub.js";

export interface CoordinatorRuntime {
  app: FastifyInstance;
  database: MeshDatabase;
  store: MeshStore;
  scheduler: Scheduler;
  hub: WorkerHub;
  service: MeshService;
  close(): Promise<void>;
}

export async function createCoordinator(
  config: CoordinatorConfig,
  options: { logger?: boolean } = {},
): Promise<CoordinatorRuntime> {
  const app = Fastify({ logger: options.logger ?? false, bodyLimit: 2 * 1024 * 1024 });
  const database = new MeshDatabase(config.databasePath);
  const store = new MeshStore(database);
  const scheduler = new Scheduler(store);
  await app.register(websocket, { options: { maxPayload: 2 * 1024 * 1024 } });
  const hub = new WorkerHub(store);
  hub.attach(app);
  const service = new MeshService(store, scheduler, hub, config.requestTimeoutMs);
  const staleTimer = setInterval(() => {
    store.markStaleWorkers();
    hub.closeStaleConnections();
  }, 5_000);
  staleTimer.unref();

  app.get("/health", async () => {
    const workers = store.listWorkers();
    return {
      status: "ok",
      version: "0.2.0",
      workers: {
        registered: workers.length,
        connected: hub.connectedWorkerIds().size,
        online: workers.filter((worker) => worker.status === "online").length,
      },
    };
  });

  app.post("/internal/v1/workers/register", async (request, reply) => {
    const registration = workerRegistrationSchema.parse(request.body);
    const worker = store.registerWorker(registration);
    return reply.code(201).send({ workerId: worker.id, protocolVersion: 1 });
  });

  app.get("/internal/v1/workers", async () => ({
    data: store.listWorkers().map((worker) => ({
      id: worker.id,
      status: worker.status,
      connected: hub.isConnected(worker.id),
      region: worker.capabilities.region,
      offeredVramMb: worker.capabilities.gpus.reduce(
        (sum, gpu) => sum + gpu.offeredVramMb,
        0,
      ),
      deployments: worker.capabilities.deployments,
      llmfit: worker.capabilities.llmfit,
      reliability: worker.reliability,
      jobsCompleted: worker.jobsCompleted,
      lastSeenAt: new Date(worker.lastSeenAt).toISOString(),
    })),
  }));

  app.get("/v1/models", async () => {
    const models = scheduler.listAvailableModels({ connectedWorkerIds: hub.connectedWorkerIds() });
    return {
      object: "list",
      data: models.map((model) => ({
        id: model.id,
        object: "model",
        created: 0,
        owned_by: "gpu-distribuida",
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
    service,
    async close() {
      clearInterval(staleTimer);
      hub.close();
      await app.close();
      database.close();
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
