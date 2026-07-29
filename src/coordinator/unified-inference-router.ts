import type { ChatCompletionRequest } from "../contracts/types.js";
import { AsyncQueue } from "../core/async-queue.js";
import type { MeshStore } from "../storage/store.js";
import type { FederationManager } from "./federation-manager.js";
import {
  MeshServiceError,
  type JobHandle,
  type JobStreamEvent,
  type MeshService,
} from "./mesh-service.js";
import type { RentalCapacityManager } from "./rental-capacity-manager.js";

const ALIASES = new Set([
  "mycellios-auto",
  "mycellios-fast",
  "mycellios-code",
  "mycellios-quality",
]);

const PRETOKEN_FALLBACK_CODES = new Set([
  "adapter_error",
  "first_token_timeout",
  "pipeline_stage_disconnected",
  "worker_disconnected",
  "worker_unreachable",
  "lease_accept_timeout",
  "no_capacity",
]);

export class UnifiedInferenceRouter {
  private readonly owners = new Map<string, "native" | "federated">();
  private readonly delegatedJobIds = new Map<string, string>();

  constructor(
    private readonly native: MeshService,
    private readonly federation: FederationManager,
    private readonly store: MeshStore,
    private readonly rentals?: RentalCapacityManager,
  ) {}

  hasCapacity(request: ChatCompletionRequest, requestedSessionId?: string): boolean {
    const nativeRequest = this.resolveNativeRequest(request);
    return (
      nativeRequest !== null && this.native.hasCapacity(nativeRequest, requestedSessionId)
    ) || this.federation.hasCapacity(request);
  }

  submit(
    request: ChatCompletionRequest,
    requestedSessionId?: string,
    idempotencyKey?: string,
  ): JobHandle {
    const nativeRequest = this.resolveNativeRequest(request);
    if (nativeRequest && this.native.hasCapacity(nativeRequest, requestedSessionId)) {
      try {
        const nativeHandle = this.native.submit(
          nativeRequest,
          requestedSessionId,
          idempotencyKey,
        );
        return this.wrapNativeWithPretokenFallback(nativeHandle, request);
      } catch (error) {
        if (!(error instanceof MeshServiceError) || error.code !== "no_capacity") throw error;
      }
    }
    if (!this.federation.hasCapacity(request)) {
      this.rentals?.recordUnmetDemand({
        model: request.model,
        minimumVramMb: estimatedModelVramMb(request.model),
        batch: request.workload_class === "batch",
        projectedTokenCostUsdPerHour:
          ((request.max_tokens ?? 512) * 60 * 2) / 1_000_000,
      });
      throw new MeshServiceError(
        "no_capacity",
        `No verified native or federated route is currently available for ${request.model}`,
        503,
      );
    }
    const federated = this.federation.submit(request, requestedSessionId, idempotencyKey);
    this.owners.set(federated.jobId, "federated");
    return federated;
  }

  cancel(jobId: string): boolean {
    const owner = this.owners.get(jobId);
    if (owner === "federated") {
      return this.federation.cancel(this.delegatedJobIds.get(jobId) ?? jobId);
    }
    return this.native.cancel(jobId) || this.federation.cancel(jobId);
  }

  cancelMatchingActiveSession(
    request: ChatCompletionRequest,
    requestedSessionId?: string,
  ): string | null {
    const nativeRequest = this.resolveNativeRequest(request);
    return nativeRequest
      ? this.native.cancelMatchingActiveSession(nativeRequest, requestedSessionId)
      : null;
  }

  private wrapNativeWithPretokenFallback(
    nativeHandle: JobHandle,
    originalRequest: ChatCompletionRequest,
  ): JobHandle {
    const queue = new AsyncQueue<JobStreamEvent>();
    this.owners.set(nativeHandle.jobId, "native");
    void (async () => {
      let emittedToken = false;
      let nativeAttempts = 1;
      for await (const event of nativeHandle.events) {
        if (event.type === "token") emittedToken = true;
        if (event.type === "progress") nativeAttempts = Math.max(nativeAttempts, event.attempt);
        if (
          event.type === "failed"
          && !emittedToken
          && PRETOKEN_FALLBACK_CODES.has(event.code)
          && this.federation.hasCapacity(originalRequest)
        ) {
          const remainingAttempts = 3 - nativeAttempts;
          if (remainingAttempts <= 0) {
            queue.push(event);
            queue.close();
            this.owners.delete(nativeHandle.jobId);
            return;
          }
          const fallback = this.federation.submit(
            originalRequest,
            nativeHandle.sessionId,
            undefined,
            remainingAttempts,
          );
          this.owners.set(nativeHandle.jobId, "federated");
          this.delegatedJobIds.set(nativeHandle.jobId, fallback.jobId);
          for await (const fallbackEvent of fallback.events) queue.push(fallbackEvent);
          queue.close();
          this.delegatedJobIds.delete(nativeHandle.jobId);
          this.owners.delete(nativeHandle.jobId);
          return;
        }
        queue.push(event);
      }
      queue.close();
      this.delegatedJobIds.delete(nativeHandle.jobId);
      this.owners.delete(nativeHandle.jobId);
    })();
    return {
      jobId: nativeHandle.jobId,
      sessionId: nativeHandle.sessionId,
      events: queue,
    };
  }

  private resolveNativeRequest(request: ChatCompletionRequest): ChatCompletionRequest | null {
    if (!ALIASES.has(request.model)) return request;
    const models = new Set(
      this.store.listWorkers()
        .filter((worker) => worker.status === "online")
        .flatMap((worker) => worker.capabilities.deployments)
        .filter((deployment) => deployment.verificationState !== "pending")
        .map((deployment) => deployment.model),
    );
    const candidates = [...models];
    if (candidates.length === 0) return null;
    const needles = request.model === "mycellios-code"
      ? ["code", "coder", "deepseek"]
      : request.model === "mycellios-quality"
        ? ["70b", "72b", "large", "pro"]
        : request.model === "mycellios-fast"
          ? ["mini", "small", "7b", "8b", "flash"]
          : [];
    const selected = candidates.find((model) =>
      needles.some((needle) => model.toLowerCase().includes(needle)))
      ?? candidates[0];
    return selected ? { ...request, model: selected } : null;
  }
}

function estimatedModelVramMb(model: string): number {
  const match = model.toLowerCase().match(/(\d+(?:\.\d+)?)b(?:\W|$)/);
  const billions = match?.[1] ? Number(match[1]) : 8;
  return Math.max(8_192, Math.ceil(billions * 1_024));
}
