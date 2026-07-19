import type {
  ChatCompletionRequest,
  CompletionResult,
  JobPayload,
  ScheduledRoute,
  TokenEvent,
  WorkerEnvelope,
} from "../contracts/types.js";
import { AsyncQueue } from "../core/async-queue.js";
import { newId } from "../core/ids.js";
import { estimateInputTokens, inputHashForRequest } from "../core/request.js";
import { Scheduler } from "../scheduler/scheduler.js";
import type { MeshStore, StoredJob } from "../storage/store.js";
import type { WorkerHub } from "./worker-hub.js";

export type JobStreamEvent =
  | { type: "accepted"; jobId: string; sessionId: string; route: ScheduledRoute }
  | { type: "token"; token: TokenEvent }
  | { type: "completed"; result: CompletionResult }
  | { type: "failed"; code: string; message: string };

export interface JobHandle {
  jobId: string;
  sessionId: string;
  events: AsyncIterable<JobStreamEvent>;
}

interface RuntimeJob {
  request: ChatCompletionRequest;
  route: ScheduledRoute;
  routePlan: ScheduledRoute[];
  routeIndex: number;
  promptCheckpoint: PromptCheckpoint;
  queue: AsyncQueue<JobStreamEvent>;
  output: string;
  outputBytes: number;
  nextTokenIndex: number;
  timeout: NodeJS.Timeout;
  leaseTimer: NodeJS.Timeout | null;
}

interface PromptCheckpoint {
  request: ChatCompletionRequest;
  requestHash: string;
  inputTokens: number;
}

export class MeshService {
  private readonly runtimes = new Map<string, RuntimeJob>();
  private readonly activeSessions = new Map<string, string>();

  constructor(
    readonly store: MeshStore,
    readonly scheduler: Scheduler,
    readonly hub: WorkerHub,
    private readonly requestTimeoutMs: number,
  ) {
    this.recoverOrphanedJobs();
    hub.on("envelope", (envelope) => this.handleWorkerEnvelope(envelope));
    hub.on("disconnect", (workerId) => this.handleWorkerDisconnect(workerId));
  }

  submit(
    request: ChatCompletionRequest,
    requestedSessionId?: string,
    idempotencyKey?: string,
  ): JobHandle {
    const sessionId = requestedSessionId ?? request.session_id ?? newId("ses");
    const activeJobId = this.activeSessions.get(sessionId);
    if (activeJobId) {
      throw new MeshServiceError(
        "session_busy",
        `Session already has an active request: ${activeJobId}`,
        409,
      );
    }
    const stableRequest = cloneRequest(request);
    const jobId = newId("job");
    const idempotencyRequestHash = idempotencyKey ? inputHashForRequest(stableRequest) : null;
    if (idempotencyKey) {
      const existing = this.store.getIdempotentJob(idempotencyKey);
      if (existing) {
        const code = existing.requestHash === idempotencyRequestHash
          ? "idempotency_replayed"
          : "idempotency_key_reused";
        throw new MeshServiceError(
          code,
          `Idempotency key is already bound to job ${existing.jobId}`,
          409,
        );
      }
    }
    const queue = new AsyncQueue<JobStreamEvent>();
    const deadlineMs = Math.min(request.deadline_ms ?? this.requestTimeoutMs, this.requestTimeoutMs);
    this.store.database.transaction(() => {
      this.store.createJob({
        id: jobId,
        sessionId,
        model: stableRequest.model,
        workloadClass: stableRequest.workload_class ?? "interactive",
        deadlineAt: Date.now() + deadlineMs,
      });
      if (idempotencyKey) {
        this.store.bindIdempotencyKey(idempotencyKey, idempotencyRequestHash!, jobId);
      }
    });

    const routePlan = this.scheduler.selectRoutePlan(stableRequest, sessionId, {
      connectedWorkerIds: this.hub.connectedWorkerIds(),
      allowPipeline: false,
      maxStandbyRoutes: 2,
    });
    if (!routePlan) {
      this.store.setJobStatus(jobId, "failed", "no_capacity");
      throw new MeshServiceError(
        "no_capacity",
        `No healthy replica is currently available for ${stableRequest.model}`,
        503,
      );
    }
    const route = routePlan.primary;

    const timeout = setTimeout(
      () => this.failRuntime(jobId, "deadline_exceeded", "The distributed request timed out"),
      deadlineMs,
    );
    const runtime: RuntimeJob = {
      request: stableRequest,
      route,
      routePlan: [routePlan.primary, ...routePlan.standbys],
      routeIndex: 0,
      promptCheckpoint: createPromptCheckpoint(stableRequest),
      queue,
      output: "",
      outputBytes: 0,
      nextTokenIndex: 0,
      timeout,
      leaseTimer: null,
    };
    this.runtimes.set(jobId, runtime);
    this.activeSessions.set(sessionId, jobId);
    queue.push({ type: "accepted", jobId, sessionId, route });
    this.dispatch(jobId, runtime);

    return { jobId, sessionId, events: queue };
  }

  cancel(jobId: string): boolean {
    const job = this.store.getJob(jobId);
    if (!job) return false;
    const runtime = this.runtimes.get(jobId);
    if (!runtime) return false;
    if (job.workerId) this.hub.send(job.workerId, "task.cancel", { jobId });
    this.store.setJobStatus(jobId, "cancelled", "cancelled_by_client");
    runtime.queue.push({ type: "failed", code: "cancelled", message: "Request cancelled" });
    this.finishRuntime(jobId);
    return true;
  }

  private dispatch(jobId: string, runtime: RuntimeJob): void {
    if (
      inputHashForRequest(runtime.promptCheckpoint.request) !==
      runtime.promptCheckpoint.requestHash
    ) {
      this.failRuntime(
        jobId,
        "prompt_checkpoint_corrupt",
        "The immutable prompt checkpoint failed its integrity check",
      );
      return;
    }
    const stage = runtime.route.stages[0];
    if (!stage) {
      this.failRuntime(jobId, "empty_route", "Scheduler produced an empty route");
      return;
    }
    const leaseId = newId("lea");
    this.store.setJobRoute(jobId, runtime.route, leaseId);
    const job = this.store.getJob(jobId);
    if (!job) return;
    const payload: JobPayload = {
      jobId,
      leaseId,
      modelDigest: stage.modelDigest,
      deadlineAt: job.deadlineAt,
      request: cloneRequest(runtime.promptCheckpoint.request),
    };
    if (!this.hub.send(stage.workerId, "lease.offer", payload)) {
      this.retryOrFail(jobId, "worker_unreachable");
      return;
    }
    if (runtime.leaseTimer) clearTimeout(runtime.leaseTimer);
    runtime.leaseTimer = setTimeout(() => this.retryOrFail(jobId, "lease_accept_timeout"), 3_000);
  }

  private handleWorkerEnvelope(envelope: WorkerEnvelope): void {
    switch (envelope.type) {
      case "lease.accept":
        this.onLeaseAccepted(envelope);
        break;
      case "lease.reject":
        this.onLeaseRejected(envelope);
        break;
      case "task.token":
        this.onToken(envelope);
        break;
      case "task.complete":
        this.onComplete(envelope);
        break;
      case "task.fail":
        this.onFailure(envelope);
        break;
    }
  }

  private onLeaseAccepted(envelope: WorkerEnvelope): void {
    const payload = envelope.payload as { jobId: string; leaseId: string };
    const job = this.validJobEnvelope(envelope.workerId, payload.jobId, payload.leaseId);
    if (job) {
      const runtime = this.runtimes.get(job.id);
      if (runtime?.leaseTimer) clearTimeout(runtime.leaseTimer);
      if (runtime) runtime.leaseTimer = null;
      this.store.setJobStatus(job.id, "running");
    }
  }

  private onLeaseRejected(envelope: WorkerEnvelope): void {
    const payload = envelope.payload as { jobId: string; leaseId: string; reason?: string };
    if (!this.validJobEnvelope(envelope.workerId, payload.jobId, payload.leaseId)) return;
    this.retryOrFail(payload.jobId, payload.reason ?? "lease_rejected");
  }

  private onToken(envelope: WorkerEnvelope): void {
    const payload = envelope.payload as {
      jobId: string;
      leaseId: string;
      index: number;
      text: string;
    };
    const job = this.validJobEnvelope(envelope.workerId, payload.jobId, payload.leaseId);
    const runtime = this.runtimes.get(payload.jobId);
    if (!job || !runtime) return;
    if (runtime.leaseTimer) clearTimeout(runtime.leaseTimer);
    runtime.leaseTimer = null;
    const chunkBytes = Buffer.byteLength(payload.text, "utf8");
    const maxOutputBytes = Math.min(
      2 * 1024 * 1024,
      Math.max(4_096, (runtime.request.max_tokens ?? 256) * 32),
    );
    if (
      payload.index !== runtime.nextTokenIndex ||
      typeof payload.text !== "string" ||
      chunkBytes > 65_536 ||
      runtime.outputBytes + chunkBytes > maxOutputBytes
    ) {
      this.failRuntime(payload.jobId, "invalid_stream", "Worker sent an invalid token sequence");
      return;
    }
    runtime.nextTokenIndex += 1;
    runtime.output += payload.text;
    runtime.outputBytes += chunkBytes;
    if (runtime.nextTokenIndex === 1) this.store.setJobStatus(payload.jobId, "streaming");
    runtime.queue.push({ type: "token", token: { index: payload.index, text: payload.text } });
  }

  private onComplete(envelope: WorkerEnvelope): void {
    const result = envelope.payload as CompletionResult;
    const job = this.validJobEnvelope(
      envelope.workerId,
      result.jobId,
      result.leaseId,
    );
    const runtime = this.runtimes.get(result.jobId);
    if (!job || !runtime) return;
    const validation = this.validateCompletion(job, runtime, result);
    if (!validation.ok) {
      this.failRuntime(job.id, "invalid_completion", validation.reason);
      return;
    }
    this.store.database.transaction(() => {
      this.store.completeJob(job.id, result.metrics);
      this.store.saveSession(job.sessionId, job.model, runtime.route);
    });
    runtime.queue.push({
      type: "completed",
      result: { ...result, text: runtime.output },
    });
    this.finishRuntime(job.id);
  }

  private onFailure(envelope: WorkerEnvelope): void {
    const payload = envelope.payload as {
      jobId: string;
      leaseId: string;
      code?: string;
      message?: string;
    };
    if (!this.validJobEnvelope(envelope.workerId, payload.jobId, payload.leaseId)) return;
    const runtime = this.runtimes.get(payload.jobId);
    if (runtime?.nextTokenIndex === 0) {
      this.retryOrFail(payload.jobId, payload.code ?? "worker_failed");
    }
    else this.failRuntime(payload.jobId, payload.code ?? "worker_failed", payload.message);
  }

  private handleWorkerDisconnect(workerId: string): void {
    for (const job of this.store.listActiveJobsForWorker(workerId)) {
      const runtime = this.runtimes.get(job.id);
      if (runtime?.nextTokenIndex === 0) this.retryOrFail(job.id, "worker_disconnected");
      else this.failRuntime(job.id, "worker_lost_midstream", "Worker disconnected after streaming began");
    }
  }

  private retryOrFail(jobId: string, reason: string): void {
    const job = this.store.getJob(jobId);
    const runtime = this.runtimes.get(jobId);
    if (!job || !runtime) return;
    if (runtime.leaseTimer) clearTimeout(runtime.leaseTimer);
    runtime.leaseTimer = null;
    if (runtime.nextTokenIndex > 0) {
      this.failRuntime(
        jobId,
        reason,
        "Streaming already began; output and KV failover are not implemented",
      );
      return;
    }
    const previousWorkerId = job.workerId;
    if (previousWorkerId) this.hub.send(previousWorkerId, "task.cancel", { jobId });
    this.store.requeueJob(jobId, reason);

    while (runtime.routeIndex + 1 < runtime.routePlan.length) {
      runtime.routeIndex += 1;
      const route = runtime.routePlan[runtime.routeIndex]!;
      if (!route.stages.every((stage) => this.hub.isConnected(stage.workerId))) continue;
      runtime.route = route;
      this.dispatch(jobId, runtime);
      return;
    }
    this.failRuntime(
      jobId,
      reason,
      "No exact-model preplanned standby remains for prompt recomputation",
    );
  }

  private failRuntime(jobId: string, code: string, message = code): void {
    const job = this.store.getJob(jobId);
    const runtime = this.runtimes.get(jobId);
    if (!job || !runtime) return;
    if (job.workerId) this.hub.send(job.workerId, "task.cancel", { jobId });
    this.store.setJobStatus(jobId, code === "deadline_exceeded" ? "expired" : "failed", code);
    runtime.queue.push({ type: "failed", code, message });
    this.finishRuntime(jobId);
  }

  private finishRuntime(jobId: string): void {
    const runtime = this.runtimes.get(jobId);
    if (!runtime) return;
    clearTimeout(runtime.timeout);
    if (runtime.leaseTimer) clearTimeout(runtime.leaseTimer);
    const job = this.store.getJob(jobId);
    if (job) {
      if (this.activeSessions.get(job.sessionId) === jobId) {
        this.activeSessions.delete(job.sessionId);
      }
    }
    runtime.queue.close();
    this.runtimes.delete(jobId);
  }

  private validJobEnvelope(workerId: string, jobId: string, leaseId: string): StoredJob | null {
    const job = this.store.getJob(jobId);
    if (!job || job.workerId !== workerId || job.leaseId !== leaseId) return null;
    if (["completed", "failed", "cancelled", "expired"].includes(job.status)) return null;
    return job;
  }

  private validateCompletion(
    job: StoredJob,
    runtime: RuntimeJob,
    result: CompletionResult,
  ): { ok: true } | { ok: false; reason: string } {
    const worker = job.workerId ? this.store.getWorker(job.workerId) : null;
    const deployment = worker?.capabilities.deployments.find(
      (candidate) => candidate.deploymentId === job.deploymentId,
    );
    if (!worker || !deployment) return { ok: false, reason: "Worker deployment disappeared" };
    if (deployment.modelDigest !== job.modelDigest) {
      return { ok: false, reason: "Model digest mismatch" };
    }
    const expectedInput = runtime.promptCheckpoint.inputTokens;
    const expectedOutput = Math.max(1, Math.ceil(runtime.output.length / 4));
    if (Math.abs(result.metrics.inputTokens - expectedInput) > Math.max(4, expectedInput * 0.1)) {
      return { ok: false, reason: "Implausible input token count" };
    }
    if (Math.abs(result.metrics.outputTokens - expectedOutput) > Math.max(4, expectedOutput * 0.1)) {
      return { ok: false, reason: "Implausible output token count" };
    }
    if (result.text !== runtime.output) return { ok: false, reason: "Completion body mismatch" };
    if (Buffer.byteLength(runtime.output, "utf8") > (runtime.request.max_tokens ?? 256) * 32) {
      return { ok: false, reason: "Completion exceeds the configured expansion limit" };
    }
    return { ok: true };
  }

  private recoverOrphanedJobs(): void {
    for (const job of this.store.listNonterminalJobs()) {
      this.store.setJobStatus(job.id, "failed", "coordinator_restarted");
    }
  }
}

export class MeshServiceError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly statusCode: number,
  ) {
    super(message);
    this.name = "MeshServiceError";
  }
}

function generationOnlyRequest(request: ChatCompletionRequest): ChatCompletionRequest {
  return {
    model: request.model,
    messages: request.messages.map((message) => ({ ...message })),
    ...(request.max_tokens === undefined ? {} : { max_tokens: request.max_tokens }),
    ...(request.temperature === undefined ? {} : { temperature: request.temperature }),
    ...(request.top_p === undefined ? {} : { top_p: request.top_p }),
    ...(request.seed === undefined ? {} : { seed: request.seed }),
  };
}

function cloneRequest(request: ChatCompletionRequest): ChatCompletionRequest {
  return {
    ...request,
    messages: request.messages.map((message) => ({ ...message })),
  };
}

function createPromptCheckpoint(request: ChatCompletionRequest): PromptCheckpoint {
  const checkpointRequest = generationOnlyRequest(request);
  return {
    request: checkpointRequest,
    requestHash: inputHashForRequest(checkpointRequest),
    inputTokens: estimateInputTokens(checkpointRequest),
  };
}
