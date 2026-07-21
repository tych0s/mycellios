import WebSocket from "ws";
import { z } from "zod";
import {
  chatCompletionRequestSchema,
  type WorkerConfig,
} from "../contracts/schemas.js";
import type {
  CompletionResult,
  JobPayload,
  WorkerCapabilities,
  WorkerEnvelope,
  WorkerHeartbeat,
} from "../contracts/types.js";
import type { InferenceAdapter } from "../adapters/base.js";
import { createAdapter } from "../adapters/factory.js";
import { sha256Text } from "../core/json.js";
import { estimateInputTokens } from "../core/request.js";
import { safeVramBudget } from "../core/tiers.js";
import { probeHardware } from "./hardware.js";
import { llmfitHardwareFallback, probeLlmfit } from "./llmfit.js";

export interface WorkerAgentOptions {
  coordinatorUrl: string;
  networkToken?: string;
  heartbeatIntervalMs?: number;
  reconnect?: boolean;
  logger?: Pick<Console, "info" | "warn" | "error">;
}

const MAX_SERVER_MESSAGE_BYTES = 2 * 1024 * 1024;
const MAX_OUTPUT_CHUNK_BYTES = 64 * 1024;
const MAX_OUTPUT_BYTES = 2 * 1024 * 1024;
const RECENT_JOB_LIMIT = 2_048;
const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);

const envelopeFields = {
  v: z.literal(1),
};

const serverMessageSchema = z.discriminatedUnion("type", [
  z
    .object({
      ...envelopeFields,
      type: z.literal("server.ready"),
      payload: z.object({ workerId: z.string().min(1).max(256) }).strict(),
    })
    .strict(),
  z
    .object({
      ...envelopeFields,
      type: z.literal("lease.offer"),
      payload: z
        .object({
          jobId: z.string().min(1).max(256),
          leaseId: z.string().min(1).max(256),
          modelDigest: z.string().min(1).max(512),
          deadlineAt: z.number().int().positive(),
          request: chatCompletionRequestSchema,
        })
        .strict(),
    })
    .strict(),
  z
    .object({
      ...envelopeFields,
      type: z.literal("task.cancel"),
      payload: z.object({ jobId: z.string().min(1).max(256) }).strict(),
    })
    .strict(),
]);

const registrationResponseSchema = z
  .object({
    workerId: z.string().min(1).max(256),
    protocolVersion: z.literal(1),
  })
  .strict();

type ValidatedServerMessage = z.infer<typeof serverMessageSchema>;

export class WorkerAgent {
  private readonly adapter: InferenceAdapter;
  private readonly coordinatorBaseUrl: URL;
  private registeredWorkerId: string | undefined;
  private capabilities: WorkerCapabilities | null = null;
  private socket: WebSocket | null = null;
  private stopped = false;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private readonly activeJobs = new Map<string, AbortController>();
  private readonly recentJobs = new Map<string, number>();
  private readonly logger: Pick<Console, "info" | "warn" | "error">;

  constructor(
    private readonly config: WorkerConfig,
    private readonly options: WorkerAgentOptions,
  ) {
    this.coordinatorBaseUrl = validateCoordinatorUrl(options.coordinatorUrl);
    this.adapter = createAdapter(config);
    this.logger = options.logger ?? console;
  }

  async start(): Promise<void> {
    this.capabilities = await this.buildCapabilities();
    await this.register();
    let delayMs = 500;
    do {
      try {
        await this.connectOnce();
        delayMs = 500;
      } catch (error) {
        if (!this.stopped) this.logger.warn(`Worker connection failed: ${errorText(error)}`);
      }
      if (this.stopped || this.options.reconnect === false) break;
      await delay(delayMs);
      delayMs = Math.min(15_000, delayMs * 2);
    } while (!this.stopped);
  }

  async stop(): Promise<void> {
    this.stopped = true;
    if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);
    await this.abortActiveJobs("Worker shutting down");
    await this.sendGoodbye("user_requested");
    await this.closeSocket();
  }

  get workerId(): string | undefined {
    return this.registeredWorkerId;
  }

  private async sendGoodbye(reason: "user_requested" | "shutdown"): Promise<void> {
    const socket = this.socket;
    const workerId = this.registeredWorkerId;
    if (!socket || socket.readyState !== socket.OPEN || !workerId) return;
    const envelope: WorkerEnvelope = {
      v: 1,
      type: "worker.goodbye",
      workerId,
      payload: { reason },
    };
    await new Promise<void>((resolve) => {
      const timer = setTimeout(resolve, 500);
      timer.unref();
      socket.send(JSON.stringify(envelope), () => {
        clearTimeout(timer);
        resolve();
      });
    });
  }

  private async closeSocket(): Promise<void> {
    const socket = this.socket;
    if (!socket || socket.readyState === socket.CLOSED) return;
    await new Promise<void>((resolve) => {
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        socket.off("close", finish);
        resolve();
      };
      const timer = setTimeout(finish, 1_000);
      timer.unref();
      socket.once("close", finish);
      try {
        socket.close(1000, "worker shutting down");
      } catch {
        finish();
      }
    });
  }

  private async buildCapabilities(): Promise<WorkerCapabilities> {
    const [hardware, adapter, llmfit] = await Promise.all([
      probeHardware(),
      this.adapter.probe(),
      this.inspectWithLlmfit(),
    ]);
    const detectedPrimary = hardware.gpus[0]!;
    const primary =
      detectedPrimary.vendor === "unknown" && detectedPrimary.physicalVramMb === 0 && llmfit
        ? (llmfitHardwareFallback(llmfit) ?? detectedPrimary)
        : detectedPrimary;
    const offeredVramMb = this.config.offeredVramMb;
    const safeBudget = safeVramBudget(offeredVramMb);
    const model = this.config.adapter.model;
    const deploymentId = `dep-${sha256Text(`${model}:${adapter.kind}`).slice(-12)}`;
    if (llmfit?.model) llmfit.model.deploymentId = deploymentId;
    const llmfitTokensPerSecond = this.config.llmfit.applyPerformanceEstimate
      ? (llmfit?.model?.measuredTokensPerSecond ??
        llmfit?.model?.estimatedTokensPerSecond)
      : undefined;
    const defaultTokensPerSecond =
      this.config.adapter.kind === "mock"
        ? this.config.adapter.tokensPerSecond
        : (llmfitTokensPerSecond ?? 5);
    const defaultTtft = this.config.adapter.kind === "mock" ? this.config.adapter.ttftMs : 2_000;
    return {
      region: this.config.region,
      agentVersion: "0.1.0",
      gpus: [
        {
          ...(this.config.capacityScope === "cell"
            ? {
                id: "cell-aggregate",
                vendor: "sidecar-cell",
                model: `Aggregate capacity exposed by ${this.config.adapter.model}`,
                // Zero means no claim about a single physical GPU. The quota
                // below represents the independently measured whole cell.
                physicalVramMb: 0,
              }
            : primary),
          offeredVramMb,
          freeOfferedVramMb: offeredVramMb,
        },
      ],
      limits: this.config.limits,
      deployments: [
        {
          deploymentId,
          model,
          modelDigest:
            this.config.deployment.modelDigest ?? sha256Text(`${adapter.kind}:${model}`),
          mode: "replica",
          adapter: adapter.kind,
          peakVramMb:
            this.config.deployment.peakVramMb ?? Math.max(512, Math.floor(safeBudget * 0.9)),
          contextLimit: this.config.deployment.contextLimit,
          maxConcurrency: this.config.limits.maxConcurrency,
          freeSlots: this.config.limits.maxConcurrency,
          tokensPerSecond:
            this.config.deployment.tokensPerSecond ?? defaultTokensPerSecond,
          ttftMs: this.config.deployment.ttftMs ?? defaultTtft,
          dataLocality: adapterDataLocality(this.config),
        },
      ],
      network: {
        coordinatorRttMs: 0,
        uplinkMbps: 100,
        downlinkMbps: 100,
      },
      ...(llmfit ? { llmfit } : {}),
    };
  }

  private async inspectWithLlmfit(): Promise<WorkerCapabilities["llmfit"] | null> {
    if (!this.config.llmfit.enabled) return null;
    if (this.config.capacityScope === "cell") {
      this.logger.warn(
        "llmfit reports the gateway host only; it will not replace aggregate cell capacity",
      );
    }
    try {
      const result = await probeLlmfit({
        executable: this.config.llmfit.executable,
        arguments: this.config.llmfit.arguments,
        timeoutMs: this.config.llmfit.timeoutMs,
        model: this.config.llmfit.model ?? this.config.adapter.model,
        maxContext: this.config.deployment.contextLimit,
      });
      for (const warning of result.warnings) this.logger.warn(warning);
      const model = result.advisory.model;
      if (model) {
        const basis = model.measuredTokensPerSecond ? "measured" : "estimated";
        const tps = model.measuredTokensPerSecond ?? model.estimatedTokensPerSecond;
        this.logger.info(
          `llmfit matched ${model.resolvedModel}: ${model.fitLevel}, ${model.bestQuant ?? "quant unknown"}${tps ? `, ${tps} tok/s ${basis}` : ""}`,
        );
      }
      return result.advisory;
    } catch (error) {
      const message = `llmfit inspection failed: ${errorText(error)}`;
      if (this.config.llmfit.required) throw new Error(message, { cause: error });
      this.logger.warn(`${message}; continuing with the native worker probe`);
      return null;
    }
  }

  private async register(): Promise<void> {
    const response = await fetch(
      coordinatorHttpUrl(this.coordinatorBaseUrl, "internal/v1/workers/register"),
      {
        method: "POST",
        headers: this.requestHeaders({ "content-type": "application/json" }),
        body: JSON.stringify({
          capabilities: this.capabilities,
        }),
        signal: AbortSignal.timeout(10_000),
        redirect: "manual",
      },
    );
    if (!response.ok) {
      throw new Error(`Worker registration failed with HTTP ${response.status}`);
    }
    const serialized = await readResponseTextLimited(response, 64 * 1024);
    let decoded: unknown;
    try {
      decoded = JSON.parse(serialized) as unknown;
    } catch {
      throw new Error("Worker registration returned invalid JSON");
    }
    const body = registrationResponseSchema.parse(decoded);
    this.registeredWorkerId = body.workerId;
  }

  private connectOnce(): Promise<void> {
    if (!this.registeredWorkerId) throw new Error("Worker has not been registered");
    const url = coordinatorWebSocketUrl(
      this.coordinatorBaseUrl,
      "internal/v1/workers/connect",
    );

    return new Promise((resolve, reject) => {
      let opened = false;
      const socket = new WebSocket(url, {
        maxPayload: MAX_SERVER_MESSAGE_BYTES,
        ...(this.options.networkToken
          ? { headers: { authorization: `Bearer ${this.options.networkToken}` } }
          : {}),
      });
      this.socket = socket;
      socket.on("open", () => {
        opened = true;
        this.sendMessage("worker.hello", {});
      });
      socket.on("message", (raw) => {
        let decoded: unknown;
        try {
          const serialized = raw.toString();
          if (Buffer.byteLength(serialized, "utf8") > MAX_SERVER_MESSAGE_BYTES) {
            throw new Error("Server message exceeds the maximum size");
          }
          decoded = JSON.parse(serialized) as unknown;
        } catch (error) {
          this.rejectServerMessage(socket, error);
          return;
        }
        void this.handleServerMessage(decoded).catch((error: unknown) => {
          this.rejectServerMessage(socket, error);
        });
      });
      socket.on("error", (error) => {
        if (!opened) reject(error);
      });
      socket.on("close", () => {
        if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);
        this.heartbeatTimer = null;
        this.socket = null;
        void this.abortActiveJobs("Coordinator disconnected");
        if (opened) resolve();
      });
    });
  }

  private requestHeaders(initial: Record<string, string> = {}): Record<string, string> {
    if (!this.options.networkToken) return initial;
    return { ...initial, authorization: `Bearer ${this.options.networkToken}` };
  }

  private async handleServerMessage(input: unknown): Promise<void> {
    const message = parseServerMessage(input);
    switch (message.type) {
      case "server.ready": {
        if (message.payload.workerId !== this.registeredWorkerId) {
          throw new Error("Coordinator acknowledged a different worker id");
        }
        this.logger.info(`Worker ${this.registeredWorkerId ?? "unknown"} connected`);
        if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);
        await this.sendHeartbeat();
        this.heartbeatTimer = setInterval(
          () => void this.sendHeartbeat(),
          this.options.heartbeatIntervalMs ?? 5_000,
        );
        break;
      }
      case "lease.offer":
        // Zod has already validated and normalized every field. The cast only
        // bridges its optional-property representation under exactOptionalPropertyTypes.
        void this.execute(message.payload as JobPayload);
        break;
      case "task.cancel": {
        const { jobId } = message.payload;
        this.activeJobs.get(jobId)?.abort(new Error("Cancelled by coordinator"));
        await this.adapter.cancel(jobId);
        break;
      }
    }
  }

  private async execute(payload: JobPayload): Promise<void> {
    const deploymentMatches = this.capabilities?.deployments.some(
      (deployment) =>
        deployment.model === payload.request.model &&
        deployment.modelDigest === payload.modelDigest,
    );
    if (!deploymentMatches) {
      this.sendMessage("lease.reject", {
        jobId: payload.jobId,
        leaseId: payload.leaseId,
        reason: "model_digest_mismatch",
      });
      return;
    }
    if (this.activeJobs.has(payload.jobId) || this.recentJobs.has(payload.jobId)) {
      this.sendMessage("lease.reject", {
        jobId: payload.jobId,
        leaseId: payload.leaseId,
        reason: "duplicate_job",
      });
      return;
    }
    if (this.activeJobs.size >= this.config.limits.maxConcurrency) {
      this.sendMessage("lease.reject", {
        jobId: payload.jobId,
        leaseId: payload.leaseId,
        reason: "worker_capacity_exhausted",
      });
      return;
    }
    const controller = new AbortController();
    const remainingMs = payload.deadlineAt - Date.now();
    if (remainingMs <= 0) {
      this.sendMessage("lease.reject", {
        jobId: payload.jobId,
        leaseId: payload.leaseId,
        reason: "deadline_exceeded",
      });
      return;
    }
    const deadlineTimer = setTimeout(
      () => controller.abort(new Error("Inference deadline exceeded")),
      remainingMs,
    );
    this.activeJobs.set(payload.jobId, controller);
    this.rememberJob(payload.jobId);
    this.updateFreeSlots();
    this.sendMessage("lease.accept", { jobId: payload.jobId, leaseId: payload.leaseId });
    const started = performance.now();
    let firstTokenAt: number | null = null;
    let output = "";
    let outputBytes = 0;
    const outputByteLimit = Math.min(
      MAX_OUTPUT_BYTES,
      (payload.request.max_tokens ?? 256) * 32,
    );
    try {
      for await (const chunk of this.adapter.generate(
        { jobId: payload.jobId, request: payload.request },
        controller.signal,
      )) {
        const chunkBytes = Buffer.byteLength(chunk.text, "utf8");
        if (chunkBytes > MAX_OUTPUT_CHUNK_BYTES) {
          const error = new WorkerOutputLimitError(
            `Adapter chunk exceeds ${MAX_OUTPUT_CHUNK_BYTES} bytes`,
          );
          controller.abort(error);
          throw error;
        }
        if (outputBytes + chunkBytes > outputByteLimit) {
          const error = new WorkerOutputLimitError(
            `Generated output exceeds ${outputByteLimit} bytes`,
          );
          controller.abort(error);
          throw error;
        }
        firstTokenAt ??= performance.now();
        output += chunk.text;
        outputBytes += chunkBytes;
        this.sendMessage("task.token", {
          jobId: payload.jobId,
          leaseId: payload.leaseId,
          index: chunk.index,
          text: chunk.text,
        });
      }
      const finished = performance.now();
      const outputTokens = Math.max(1, Math.ceil(output.length / 4));
      const activeMs = Math.max(1, Math.round(finished - started));
      const metrics = {
        inputTokens: estimateInputTokens(payload.request),
        outputTokens,
        ttftMs: Math.round((firstTokenAt ?? finished) - started),
        activeMs,
        energyWh: this.config.limits.maxPowerW
          ? (this.config.limits.maxPowerW * activeMs) / 3_600_000
          : undefined,
      };
      const result: CompletionResult = {
        jobId: payload.jobId,
        leaseId: payload.leaseId,
        text: output,
        finishReason:
          outputTokens >= (payload.request.max_tokens ?? Number.POSITIVE_INFINITY)
            ? "length"
            : "stop",
        metrics,
      };
      this.sendMessage("task.complete", result);
    } catch (error) {
      this.sendMessage("task.fail", {
        jobId: payload.jobId,
        leaseId: payload.leaseId,
        code:
          error instanceof WorkerOutputLimitError
            ? "output_limit_exceeded"
            : controller.signal.aborted
              ? "cancelled"
              : "adapter_error",
        message: errorText(error).slice(0, 300),
      });
    } finally {
      clearTimeout(deadlineTimer);
      this.activeJobs.delete(payload.jobId);
      this.updateFreeSlots();
      await this.sendHeartbeat();
    }
  }

  private rememberJob(jobId: string): void {
    this.recentJobs.set(jobId, Date.now());
    while (this.recentJobs.size > RECENT_JOB_LIMIT) {
      const oldest = this.recentJobs.keys().next().value as string | undefined;
      if (!oldest) break;
      this.recentJobs.delete(oldest);
    }
  }

  private async abortActiveJobs(reason: string): Promise<void> {
    const jobs = [...this.activeJobs.entries()];
    for (const [, controller] of jobs) controller.abort(new Error(reason));
    await Promise.allSettled(jobs.map(([jobId]) => this.adapter.cancel(jobId)));
  }

  private rejectServerMessage(socket: WebSocket, error: unknown): void {
    this.logger.warn(`Rejected invalid coordinator message: ${errorText(error)}`);
    if (socket.readyState === WebSocket.OPEN) {
      socket.close(1008, "invalid coordinator message");
    }
  }

  private async sendHeartbeat(): Promise<void> {
    if (!this.capabilities || !this.socket || this.socket.readyState !== WebSocket.OPEN) return;
    const metrics = await this.adapter.metrics();
    const heartbeat: WorkerHeartbeat = {
      draining: false,
      pausedReason: null,
      activeLeases: [...this.activeJobs.keys()],
      gpus: this.capabilities.gpus.map((gpu) => ({
        id: gpu.id,
        freeOfferedVramMb: gpu.freeOfferedVramMb,
        ...(gpu.utilizationPct === undefined ? {} : { utilizationPct: gpu.utilizationPct }),
        ...(gpu.temperatureC === undefined ? {} : { temperatureC: gpu.temperatureC }),
        ...(gpu.powerW === undefined ? {} : { powerW: gpu.powerW }),
      })),
      deployments: this.capabilities.deployments.map((deployment) => ({
        deploymentId: deployment.deploymentId,
        freeSlots: deployment.freeSlots,
      })),
      network: {
        coordinatorRttMs: this.capabilities.network.coordinatorRttMs,
        uplinkMbps: this.capabilities.network.uplinkMbps,
      },
    };
    this.sendMessage("worker.heartbeat", { heartbeat, capabilities: this.capabilities, metrics });
  }

  private updateFreeSlots(): void {
    if (!this.capabilities) return;
    const freeSlots = Math.max(0, this.config.limits.maxConcurrency - this.activeJobs.size);
    this.capabilities = {
      ...this.capabilities,
      deployments: this.capabilities.deployments.map((deployment) => ({
        ...deployment,
        freeSlots,
      })),
    };
  }

  private sendMessage(type: string, payload: unknown): void {
    if (!this.registeredWorkerId || !this.socket || this.socket.readyState !== WebSocket.OPEN) return;
    const envelope: WorkerEnvelope = {
      v: 1,
      type,
      workerId: this.registeredWorkerId,
      payload,
    };
    this.socket.send(JSON.stringify(envelope));
  }
}

class WorkerOutputLimitError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WorkerOutputLimitError";
  }
}

export function parseServerMessage(input: unknown): ValidatedServerMessage {
  const parsed = serverMessageSchema.safeParse(input);
  if (!parsed.success) {
    throw new Error(`Invalid coordinator message: ${z.prettifyError(parsed.error)}`);
  }
  return parsed.data;
}

export function validateCoordinatorUrl(raw: string): URL {
  const normalized = raw.endsWith("/") ? raw : `${raw}/`;
  const url = new URL(normalized);
  if (!new Set(["http:", "https:", "ws:", "wss:"]).has(url.protocol)) {
    throw new Error("Coordinator URL must use HTTP(S) or WS(S)");
  }
  if (new Set(["http:", "ws:"]).has(url.protocol) && !isLoopback(url.hostname)) {
    throw new Error("Remote coordinators must use HTTPS/WSS");
  }
  if (url.username || url.password) {
    throw new Error("Coordinator URL must not embed credentials");
  }
  return url;
}

function coordinatorHttpUrl(base: URL, path: string): URL {
  const url = new URL(path, base);
  if (url.protocol === "ws:") url.protocol = "http:";
  if (url.protocol === "wss:") url.protocol = "https:";
  return url;
}

function coordinatorWebSocketUrl(base: URL, path: string): URL {
  const url = new URL(path, base);
  if (url.protocol === "http:") url.protocol = "ws:";
  if (url.protocol === "https:") url.protocol = "wss:";
  return url;
}

function isLoopback(hostname: string): boolean {
  return LOOPBACK_HOSTS.has(hostname.toLowerCase());
}

async function readResponseTextLimited(response: Response, limitBytes: number): Promise<string> {
  if (!response.body) return "";
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let result = "";
  let bytes = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > limitBytes) {
        throw new Error(`Response body exceeds ${limitBytes} bytes`);
      }
      result += decoder.decode(value, { stream: true });
    }
    result += decoder.decode();
    return result;
  } finally {
    reader.releaseLock();
  }
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function adapterDataLocality(config: WorkerConfig): "local" | "external" {
  if (config.adapter.kind !== "openai-compatible") return "local";
  const hostname = new URL(config.adapter.baseUrl).hostname.toLowerCase();
  return new Set(["localhost", "127.0.0.1", "::1", "[::1]"]).has(hostname)
    ? "local"
    : "external";
}
