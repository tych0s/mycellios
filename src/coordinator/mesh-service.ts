import { EventEmitter } from "node:events";
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
  | {
      type: "progress";
      phase: "waiting_first_token" | "recovering";
      message: string;
      attempt: number;
      workerId?: string;
      nodeId?: string;
    }
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
  firstTokenTimer: NodeJS.Timeout | null;
  attempt: number;
}

interface PromptCheckpoint {
  request: ChatCompletionRequest;
  requestHash: string;
  inputTokens: number;
}

const PRETOKEN_DEGRADATION_CODES = new Set([
  "adapter_error",
  "first_token_timeout",
  "pipeline_stage_disconnected",
  "worker_disconnected",
  "worker_unreachable",
  "lease_accept_timeout",
]);

export interface MeshServiceOptions {
  firstTokenTimeoutMs?: number;
}

interface MeshServiceEvents {
  degraded: [{ jobId: string; model: string; code: string; workerId: string | null }];
  healthy: [{ jobId: string; model: string; workerId: string | null }];
}

export class MeshService extends EventEmitter<MeshServiceEvents> {
  private readonly runtimes = new Map<string, RuntimeJob>();
  private readonly activeSessions = new Map<string, string>();

  constructor(
    readonly store: MeshStore,
    readonly scheduler: Scheduler,
    readonly hub: WorkerHub,
    private readonly requestTimeoutMs: number,
    private readonly options: MeshServiceOptions = {},
  ) {
    super();
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
      firstTokenTimer: null,
      attempt: 1,
    };
    this.runtimes.set(jobId, runtime);
    this.activeSessions.set(sessionId, jobId);
    queue.push({ type: "accepted", jobId, sessionId, route });
    this.dispatch(jobId, runtime);

    return { jobId, sessionId, events: queue };
  }

  /**
   * Cancels an orphaned request only when a reconnect repeats the exact same
   * immutable prompt in the same session. A genuinely different concurrent
   * message keeps the normal session_busy protection.
   */
  cancelMatchingActiveSession(
    request: ChatCompletionRequest,
    requestedSessionId?: string,
  ): string | null {
    const sessionId = requestedSessionId ?? request.session_id;
    if (!sessionId) return null;
    const activeJobId = this.activeSessions.get(sessionId);
    if (!activeJobId) return null;
    const runtime = this.runtimes.get(activeJobId);
    if (!runtime) return null;
    if (inputHashForRequest(cloneRequest(request)) !== runtime.promptCheckpoint.requestHash) {
      return null;
    }
    return this.cancel(activeJobId) ? activeJobId : null;
  }

  hasCapacity(request: ChatCompletionRequest, requestedSessionId?: string): boolean {
    const sessionId = requestedSessionId ?? request.session_id ?? newId("capacity");
    return this.scheduler.selectRoutePlan(cloneRequest(request), sessionId, {
      connectedWorkerIds: this.hub.connectedWorkerIds(),
      allowPipeline: false,
      maxStandbyRoutes: 2,
    }) !== null;
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
      if (runtime) {
        if (runtime.firstTokenTimer) clearTimeout(runtime.firstTokenTimer);
        const firstTokenTimeoutMs = this.firstTokenTimeoutFor(job.id, runtime);
        runtime.queue.push({
          type: "progress",
          phase: "waiting_first_token",
          message: "Route accepted. Waiting for the first model token.",
          attempt: runtime.attempt,
          workerId: envelope.workerId,
        });
        runtime.firstTokenTimer = setTimeout(() => {
          const active = this.runtimes.get(job.id);
          if (!active || active.nextTokenIndex > 0) return;
          this.retryOrFail(
            job.id,
            "first_token_timeout",
            `The selected route produced no first token within ${firstTokenTimeoutMs} ms`,
          );
        }, firstTokenTimeoutMs);
      }
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
    if (runtime.firstTokenTimer) clearTimeout(runtime.firstTokenTimer);
    runtime.firstTokenTimer = null;
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
    this.emit("healthy", {
      jobId: job.id,
      model: job.model,
      workerId: job.workerId,
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
      this.retryOrFail(payload.jobId, payload.code ?? "worker_failed", payload.message);
    }
    else this.failRuntime(payload.jobId, payload.code ?? "worker_failed", payload.message);
  }

  private handleWorkerDisconnect(workerId: string): void {
    const handledJobs = new Set<string>();
    for (const job of this.store.listActiveJobsForWorker(workerId)) {
      handledJobs.add(job.id);
      const runtime = this.runtimes.get(job.id);
      if (runtime?.nextTokenIndex === 0) this.retryOrFail(job.id, "worker_disconnected");
      else this.failRuntime(job.id, "worker_lost_midstream", "Worker disconnected after streaming began");
    }
    const disconnectedWorker = this.store.getWorker(workerId);
    const disconnectedNodeId = disconnectedWorker?.capabilities.distributedExecutor?.nodeId;
    if (!disconnectedNodeId) return;
    for (const [jobId, runtime] of this.runtimes) {
      if (handledJobs.has(jobId)) continue;
      const job = this.store.getJob(jobId);
      if (!job?.workerId || !job.deploymentId) continue;
      const routeWorker = this.store.getWorker(job.workerId);
      const deployment = routeWorker?.capabilities.deployments.find(
        (candidate) => candidate.deploymentId === job.deploymentId,
      );
      const affectedStage = deployment?.execution?.stages?.find(
        (stage) => stage.nodeId === disconnectedNodeId,
      );
      if (!affectedStage) continue;
      const message = `${affectedStage.deviceName} (${disconnectedNodeId}) disconnected from the distributed pipeline`;
      runtime.queue.push({
        type: "progress",
        phase: "recovering",
        message,
        attempt: runtime.attempt,
        workerId,
        nodeId: disconnectedNodeId,
      });
      if (runtime.nextTokenIndex === 0) {
        this.retryOrFail(jobId, "pipeline_stage_disconnected", message);
      } else {
        this.failRuntime(jobId, "pipeline_stage_lost_midstream", message);
      }
    }
  }

  private retryOrFail(jobId: string, reason: string, failureMessage?: string): void {
    const job = this.store.getJob(jobId);
    const runtime = this.runtimes.get(jobId);
    if (!job || !runtime) return;
    if (runtime.leaseTimer) clearTimeout(runtime.leaseTimer);
    runtime.leaseTimer = null;
    if (runtime.firstTokenTimer) clearTimeout(runtime.firstTokenTimer);
    runtime.firstTokenTimer = null;
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
      runtime.attempt += 1;
      runtime.queue.push({
        type: "progress",
        phase: "recovering",
        message: "The route failed before token zero. Retrying on an exact-model standby.",
        attempt: runtime.attempt,
        ...(previousWorkerId ? { workerId: previousWorkerId } : {}),
      });
      this.dispatch(jobId, runtime);
      return;
    }
    this.failRuntime(
      jobId,
      reason,
      failureMessage ?? "No exact-model preplanned standby remains for prompt recomputation",
    );
  }

  private failRuntime(jobId: string, code: string, message = code): void {
    const job = this.store.getJob(jobId);
    const runtime = this.runtimes.get(jobId);
    if (!job || !runtime) return;
    if (job.workerId) this.hub.send(job.workerId, "task.cancel", { jobId });
    this.store.setJobStatus(jobId, code === "deadline_exceeded" ? "expired" : "failed", code);
    runtime.queue.push({ type: "failed", code, message });
    if (runtime.nextTokenIndex === 0 && PRETOKEN_DEGRADATION_CODES.has(code)) {
      this.emit("degraded", {
        jobId,
        model: job.model,
        code,
        workerId: job.workerId,
      });
    }
    this.finishRuntime(jobId);
  }

  private finishRuntime(jobId: string): void {
    const runtime = this.runtimes.get(jobId);
    if (!runtime) return;
    clearTimeout(runtime.timeout);
    if (runtime.leaseTimer) clearTimeout(runtime.leaseTimer);
    if (runtime.firstTokenTimer) clearTimeout(runtime.firstTokenTimer);
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

  private firstTokenTimeoutFor(jobId: string, runtime: RuntimeJob): number {
    if (this.options.firstTokenTimeoutMs !== undefined) {
      return Math.max(10, Math.round(this.options.firstTokenTimeoutMs));
    }
    const routeStage = runtime.route.stages[0];
    const routeWorker = routeStage ? this.store.getWorker(routeStage.workerId) : null;
    const deployment = routeWorker?.capabilities.deployments.find(
      (candidate) => candidate.deploymentId === routeStage?.deploymentId,
    );
    const expectedTtftMs = deployment?.ttftMs ?? 2_500;
    const adaptiveTimeoutMs = Math.max(15_000, expectedTtftMs * 6 + 2_000);
    const job = this.store.getJob(jobId);
    const remainingMs = job ? Math.max(1_000, job.deadlineAt - Date.now() - 1_000) : 30_000;
    return Math.max(1_000, Math.min(30_000, adaptiveTimeoutMs, remainingMs));
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
    const promptBytes = runtime.promptCheckpoint.request.messages.reduce(
      (sum, message) => sum + Buffer.byteLength(`${message.role}\n${message.content}`, "utf8"),
      0,
    );
    // The coordinator's characters/4 estimate is useful for scheduling, but
    // it is not an exact tokenizer. Real chat templates add control tokens and
    // byte-level tokenizers can legitimately diverge by far more than 10%.
    // Keep the trust boundary by bounding the worker report against both the
    // request bytes and the deployment's certified context window.
    const plausibleInputMaximum = Math.min(
      deployment.contextLimit,
      Math.max(
        64,
        expectedInput * 8,
        promptBytes * 2 + 256 + runtime.promptCheckpoint.request.messages.length * 64,
      ),
    );
    if (result.metrics.inputTokens < 1 || result.metrics.inputTokens > plausibleInputMaximum) {
      return { ok: false, reason: "Implausible input token count" };
    }
    const maximumOutputTokens = runtime.request.max_tokens ?? 256;
    const outputBytes = Buffer.byteLength(runtime.output, "utf8");
    // Output text length is not a tokenizer. Byte-level tokens, Unicode and
    // model-specific vocabularies can differ dramatically from characters/4,
    // especially for tiny/random validation models. Keep the trust boundary
    // against the caller's token ceiling and the actual emitted bytes instead.
    if (
      result.metrics.outputTokens < 0 ||
      result.metrics.outputTokens > maximumOutputTokens ||
      (runtime.output.length > 0 && result.metrics.outputTokens === 0) ||
      result.metrics.outputTokens > outputBytes + 16
    ) {
      return { ok: false, reason: "Implausible output token count" };
    }
    if (result.text !== runtime.output) return { ok: false, reason: "Completion body mismatch" };
    if (outputBytes > maximumOutputTokens * 32) {
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
    ...(request.session_id === undefined ? {} : { session_id: request.session_id }),
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
