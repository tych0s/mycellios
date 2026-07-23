import WebSocket from "ws";
import { z } from "zod";
import {
  chatCompletionRequestSchema,
  type WorkerConfig,
} from "../contracts/schemas.js";
import type {
  CompletionResult,
  ComputeMode,
  JobPayload,
  WorkerCapabilities,
  WorkerAcceleratorDiagnostics,
  WorkerEnvelope,
  WorkerHeartbeat,
} from "../contracts/types.js";
import type { AdapterChunk, InferenceAdapter } from "../adapters/base.js";
import { createAdapter } from "../adapters/factory.js";
import { sha256Text } from "../core/json.js";
import { estimateInputTokens } from "../core/request.js";
import { safeVramBudget } from "../core/tiers.js";
import {
  probeHardware,
  selectHardwareGpu,
  selectRuntimeCapacityHardware,
  type HardwareProbe,
  type VerifiedGpuRuntimeEvidence,
} from "./hardware.js";
import { llmfitHardwareFallback, probeLlmfit } from "./llmfit.js";
import {
  LaunchProcessExitedError,
  type LaunchAgent,
  type LaunchAgentStartRequest,
  type LaunchProcessHandle,
} from "../distribution/launch-supervisor.js";
import {
  validatePythonLaunchDescription,
  type PythonPipelineLaunchDescription,
} from "../distribution/python-launcher.js";
import { MAX_RUNTIME_STREAM_CHUNK_BYTES } from "../contracts/worker-protocol.js";
import {
  RuntimeStreamTunnel,
  type RuntimeStreamServerMessage,
} from "./runtime-stream-tunnel.js";

export interface WorkerAgentOptions {
  coordinatorUrl: string;
  networkToken?: string;
  heartbeatIntervalMs?: number;
  reconnect?: boolean;
  identity?: {
    kind: "device" | "cell";
    id: string;
  };
  /** Register the physical node without claiming that a model runtime exists. */
  advertiseDeployment?: boolean;
  /** Deterministic hardware source for embedded agents and tests. */
  hardwareProbe?: () => Promise<HardwareProbe>;
  /** Adapter chosen by the verified runtime selector, independent of OS ordering. */
  preferredHardwareGpu?: {
    id?: string;
    vendor: string;
    model: string;
  };
  /** Real CPU-memory capacity used when no reliable GPU memory budget exists. */
  hardwareCapacityOverride?: {
    id: string;
    vendor: string;
    model: string;
    physicalVramMb: number;
    sharedMemoryMb?: number | undefined;
    unifiedMemory?: boolean | undefined;
  };
  /** Physical GPU probe evidence. Without it, host capacity is CPU RAM only. */
  verifiedGpuRuntime?: VerifiedGpuRuntimeEvidence | undefined;
  /** Desktop application version reported to the coordinator. */
  agentVersion?: string | undefined;
  distributedExecutor?: {
    nodeId: string;
    stageHost: string;
    stagePort: number;
    launchAgent: LaunchAgent;
    pythonExecutable?: string;
    computeMode?: ComputeMode;
    cpuEligible?: boolean;
    acceleration?: WorkerAcceleratorDiagnostics;
  };
  logger?: Pick<Console, "info" | "warn" | "error">;
}

const MAX_SERVER_MESSAGE_BYTES = 2 * 1024 * 1024;
const MAX_OUTPUT_CHUNK_BYTES = 64 * 1024;
const MAX_OUTPUT_BYTES = 2 * 1024 * 1024;
const RECENT_JOB_LIMIT = 2_048;
const MAX_WEBSOCKET_BUFFERED_BYTES = 8 * 1024 * 1024;
const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);

const envelopeFields = {
  v: z.literal(1),
};

const runtimeStreamDataSchema = z.string()
  .min(1)
  .max(Math.ceil(MAX_RUNTIME_STREAM_CHUNK_BYTES / 3) * 4)
  .refine((value) => /^[A-Za-z0-9+/]+={0,2}$/.test(value), "Runtime stream data must be base64")
  .refine(
    (value) => Buffer.from(value, "base64").byteLength <= MAX_RUNTIME_STREAM_CHUNK_BYTES,
    `Runtime stream chunks cannot exceed ${MAX_RUNTIME_STREAM_CHUNK_BYTES} bytes`,
  );

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
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.prepare"),
    payload: z.object({ requestId: z.string().min(1).max(256), description: z.unknown() }).strict(),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.start"),
    payload: z.object({ requestId: z.string().min(1).max(256), request: z.unknown() }).strict(),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.stop"),
    payload: z.object({ requestId: z.string().min(1).max(256), reason: z.string().min(1).max(300) }).strict(),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.stream.open"),
    payload: z.object({
      streamId: z.string().min(1).max(256),
      targetPort: z.number().int().min(1).max(65_535),
    }).strict(),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.stream.opened"),
    payload: z.object({ streamId: z.string().min(1).max(256) }).strict(),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.stream.data"),
    payload: z.object({
      streamId: z.string().min(1).max(256),
      sequence: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
      data: runtimeStreamDataSchema,
    }).strict(),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.stream.end"),
    payload: z.object({ streamId: z.string().min(1).max(256) }).strict(),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.stream.error"),
    payload: z.object({
      streamId: z.string().min(1).max(256),
      message: z.string().min(1).max(1_024),
    }).strict(),
  }).strict(),
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
  private readonly authorizedRuntimeProcesses = new Map<string, string>();
  private readonly runtimeProcesses = new Map<string, LaunchProcessHandle>();
  private readonly runtimeTunnel: RuntimeStreamTunnel | null;
  private readonly logger: Pick<Console, "info" | "warn" | "error">;
  private runtimeCapacityGeneration = 0;

  constructor(
    private readonly config: WorkerConfig,
    private readonly options: WorkerAgentOptions,
  ) {
    this.coordinatorBaseUrl = validateCoordinatorUrl(options.coordinatorUrl);
    this.adapter = createAdapter(config);
    this.logger = options.logger ?? console;
    this.runtimeTunnel = options.distributedExecutor
      ? new RuntimeStreamTunnel(
          options.distributedExecutor.nodeId,
          (type, payload) => this.sendMessage(type, payload),
        )
      : null;
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
    await this.resetDistributedRuntime("worker_shutting_down");
    await this.sendGoodbye("user_requested");
    await this.closeSocket();
  }

  get workerId(): string | undefined {
    return this.registeredWorkerId;
  }

  /**
   * Refresh the capacity published to the coordinator after a physical GPU
   * probe succeeds or a native runtime is invalidated. Re-registration updates
   * the full capability document (vendor, model and memory kind); a heartbeat
   * alone can only update values for devices the coordinator already knows.
   * This deliberately keeps the existing socket and active work in place.
   */
  async refreshRuntimeCapacity(
    runtime: VerifiedGpuRuntimeEvidence | undefined,
    executorPolicy?: { computeMode: ComputeMode; cpuEligible: boolean },
    acceleration?: WorkerAcceleratorDiagnostics,
  ): Promise<void> {
    const generation = ++this.runtimeCapacityGeneration;
    this.options.verifiedGpuRuntime = runtime;
    if (executorPolicy && this.options.distributedExecutor) {
      this.options.distributedExecutor.computeMode = executorPolicy.computeMode;
      this.options.distributedExecutor.cpuEligible = executorPolicy.cpuEligible;
    }
    if (acceleration && this.options.distributedExecutor) {
      this.options.distributedExecutor.acceleration = acceleration;
    }
    if (!this.capabilities || this.config.capacityScope !== "host") return;

    const hardware = await (this.options.hardwareProbe?.() ?? probeHardware());
    if (generation !== this.runtimeCapacityGeneration) return;
    const selectedHardwareGpu = selectHardwareGpu(hardware.gpus, this.options.preferredHardwareGpu)
      ?? hardware.gpus[0];
    const primary = this.options.hardwareCapacityOverride
      ?? selectRuntimeCapacityHardware(hardware, selectedHardwareGpu, runtime);
    const capacityMb = primary.physicalVramMb + (primary.sharedMemoryMb ?? 0);
    const offeredVramMb = Math.min(
      this.config.offeredVramMb,
      Math.max(512, capacityMb),
    );
    const previous = this.capabilities.gpus[0];
    const usedVramMb = previous
      ? Math.max(0, previous.offeredVramMb - previous.freeOfferedVramMb)
      : 0;
    const freeOfferedVramMb = Math.max(0, offeredVramMb - usedVramMb);
    const defaultPeakVramMb = Math.max(512, Math.floor(safeVramBudget(offeredVramMb) * 0.9));

    this.capabilities = {
      ...this.capabilities,
      gpus: [{
        ...primary,
        offeredVramMb,
        freeOfferedVramMb,
      }],
      deployments: this.capabilities.deployments.map((deployment) => ({
        ...deployment,
        peakVramMb: this.config.deployment.peakVramMb ?? defaultPeakVramMb,
      })),
      ...(this.capabilities.distributedExecutor
        ? {
            distributedExecutor: {
              ...this.capabilities.distributedExecutor,
              computeMode: this.options.distributedExecutor?.computeMode ?? "automatic",
              cpuEligible: this.options.distributedExecutor?.cpuEligible === true,
              ...(this.options.distributedExecutor?.acceleration
                ? { acceleration: structuredClone(this.options.distributedExecutor.acceleration) }
                : {}),
            },
          }
        : {}),
    };
    if (this.registeredWorkerId) await this.register();
    await this.sendHeartbeat();
  }

  async refreshRuntimeDiagnostics(acceleration: WorkerAcceleratorDiagnostics): Promise<void> {
    if (!this.options.distributedExecutor) return;
    this.options.distributedExecutor.acceleration = acceleration;
    if (!this.capabilities?.distributedExecutor) return;
    this.capabilities = {
      ...this.capabilities,
      distributedExecutor: {
        ...this.capabilities.distributedExecutor,
        acceleration: structuredClone(acceleration),
      },
    };
    if (this.registeredWorkerId) await this.register();
  }

  get activeJobCount(): number {
    return this.activeJobs.size;
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
      this.options.hardwareProbe?.() ?? probeHardware(),
      this.adapter.probe(),
      this.inspectWithLlmfit(),
    ]);
    const selectedHardwareGpu = selectHardwareGpu(hardware.gpus, this.options.preferredHardwareGpu)
      ?? hardware.gpus[0];
    const detectedPrimary = this.options.hardwareCapacityOverride
      ?? selectRuntimeCapacityHardware(
        hardware,
        selectedHardwareGpu,
        this.options.verifiedGpuRuntime,
      );
    const primary =
      detectedPrimary.vendor === "unknown" && detectedPrimary.physicalVramMb === 0 && llmfit
        ? (llmfitHardwareFallback(llmfit) ?? detectedPrimary)
        : detectedPrimary;
    const primaryCapacityMb = primary.physicalVramMb + (primary.sharedMemoryMb ?? 0);
    const offeredVramMb = this.config.capacityScope === "cell"
      ? this.config.offeredVramMb
      : Math.min(this.config.offeredVramMb, Math.max(512, primaryCapacityMb));
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
      agentVersion: this.options.agentVersion?.trim() || "0.1.0",
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
      deployments: this.options.advertiseDeployment === false
        ? []
        : [{
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
          ...(this.config.deployment.internalPipeline
            ? { internalPipeline: structuredClone(this.config.deployment.internalPipeline) }
            : {}),
          ...(this.config.deployment.execution
            ? { execution: structuredClone(this.config.deployment.execution) }
            : {}),
        }],
      network: {
        coordinatorRttMs: 0,
        uplinkMbps: 100,
        downlinkMbps: 100,
      },
      ...(llmfit ? { llmfit } : {}),
      ...(this.options.distributedExecutor
        ? {
            distributedExecutor: {
              protocol: "gdlp-worker-tunnel/2" as const,
              nodeId: this.options.distributedExecutor.nodeId,
              stageHost: this.options.distributedExecutor.stageHost,
              stagePort: this.options.distributedExecutor.stagePort,
              runtime: "python-safetensors" as const,
              computeMode: this.options.distributedExecutor.computeMode ?? "automatic",
              cpuEligible: this.options.distributedExecutor.cpuEligible === true,
              ...(this.options.distributedExecutor.acceleration
                ? { acceleration: structuredClone(this.options.distributedExecutor.acceleration) }
                : {}),
            },
          }
        : {}),
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
    const identity = this.options.identity ?? this.defaultIdentity();
    const response = await fetch(
      coordinatorHttpUrl(this.coordinatorBaseUrl, "internal/v1/workers/register"),
      {
        method: "POST",
        headers: this.requestHeaders({ "content-type": "application/json" }),
        body: JSON.stringify({
          ...(identity ? { identity } : {}),
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

  private defaultIdentity(): WorkerAgentOptions["identity"] {
    if (this.options.distributedExecutor) {
      return { kind: "device", id: this.options.distributedExecutor.nodeId };
    }
    if (this.config.instanceId) {
      return {
        kind: this.config.capacityScope === "cell" ? "cell" : "device",
        id: this.config.instanceId,
      };
    }
    return undefined;
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
        void this.resetDistributedRuntime("coordinator_disconnected").finally(() => {
          if (opened) resolve();
          else reject(new Error("Coordinator connection closed before it became ready"));
        });
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
      case "runtime.prepare":
        await this.prepareDistributedRuntime(message.payload.requestId, message.payload.description);
        break;
      case "runtime.start":
        await this.startDistributedRuntime(message.payload.requestId, message.payload.request);
        break;
      case "runtime.stop":
        await this.stopDistributedRuntime(message.payload.requestId, message.payload.reason);
        break;
      case "runtime.stream.open":
      case "runtime.stream.opened":
      case "runtime.stream.data":
      case "runtime.stream.end":
      case "runtime.stream.error":
        await this.runtimeTunnel?.handle(message as RuntimeStreamServerMessage);
        break;
    }
  }

  private async prepareDistributedRuntime(requestId: string, input: unknown): Promise<void> {
    try {
      const executor = this.options.distributedExecutor;
      if (!executor) throw new Error("distributed_executor_is_not_enabled");
      validatePythonLaunchDescription(input);
      const description = input as PythonPipelineLaunchDescription;
      const local = description.launchOrder.filter((process) => process.anchor.memberId === executor.nodeId);
      if (local.length === 0) throw new Error("distributed_plan_has_no_process_for_this_node");
      await this.runtimeTunnel?.prepare(description);
      this.authorizedRuntimeProcesses.clear();
      for (const process of local) {
        this.authorizedRuntimeProcesses.set(process.processId, JSON.stringify(process));
      }
      this.sendMessage("runtime.prepared", { requestId, ok: true });
    } catch (error) {
      this.sendMessage("runtime.prepared", { requestId, ok: false, error: errorText(error) });
    }
  }

  private async startDistributedRuntime(requestId: string, input: unknown): Promise<void> {
    const executor = this.options.distributedExecutor;
    try {
      if (!executor) throw new Error("distributed_executor_is_not_enabled");
      if (!isLaunchAgentStartRequest(input)) throw new Error("distributed_launch_request_is_invalid");
      if (input.nodeId !== executor.nodeId) throw new Error("distributed_launch_node_mismatch");
      if (this.authorizedRuntimeProcesses.get(input.process.processId) !== JSON.stringify(input.process)) {
        throw new Error("distributed_launch_process_was_not_prepared");
      }
      if (this.runtimeProcesses.has(requestId)) throw new Error("distributed_launch_request_is_duplicate");
      const controller = new AbortController();
      const tunneledProcess = this.runtimeTunnel?.rewriteProcess(input.process) ?? input.process;
      const localRequest: LaunchAgentStartRequest = executor.pythonExecutable
        ? {
            ...input,
            process: {
              ...tunneledProcess,
              command: { ...tunneledProcess.command, executable: executor.pythonExecutable },
            },
          }
        : { ...input, process: tunneledProcess };
      const handle = await executor.launchAgent.start(localRequest, controller.signal);
      this.runtimeProcesses.set(requestId, handle);
      void handle.ready.then(
        () => this.sendMessage("runtime.ready", {
          requestId,
          output: handle.output?.() ?? {
            stdout: "",
            stderr: "",
            stdoutTruncated: false,
            stderrTruncated: false,
          },
        }),
        (error: unknown) => {
          // The exited promise carries the original spawn/runtime error and
          // captured output. Avoid racing it with a lossy wrapper error.
          if (!(error instanceof LaunchProcessExitedError)) {
            this.sendRuntimeExit(requestId, handle, { code: null, signal: null, error: errorText(error) });
          }
        },
      );
      void handle.exited.then(
        (exit) => this.sendRuntimeExit(requestId, handle, exit),
        (error: unknown) => this.sendRuntimeExit(requestId, handle, { code: null, signal: null, error: errorText(error) }),
      );
    } catch (error) {
      this.sendRuntimeExit(requestId, undefined, { code: null, signal: null, error: errorText(error) });
    }
  }

  private async stopDistributedRuntime(requestId: string, reason: string): Promise<void> {
    const handle = this.runtimeProcesses.get(requestId);
    if (handle) await handle.stop(reason).catch(() => undefined);
  }

  private async resetDistributedRuntime(reason: string): Promise<void> {
    const handles = [...this.runtimeProcesses.values()];
    this.runtimeProcesses.clear();
    this.authorizedRuntimeProcesses.clear();
    await Promise.all(handles.map((handle) => handle.stop(reason).catch(() => undefined)));
    await this.runtimeTunnel?.close();
  }

  private sendRuntimeExit(
    requestId: string,
    handle: LaunchProcessHandle | undefined,
    exit: { code: number | null; signal: NodeJS.Signals | null; error?: string },
  ): void {
    if (!this.runtimeProcesses.has(requestId) && handle) return;
    this.runtimeProcesses.delete(requestId);
    const output = handle?.output?.() ?? { stdout: "", stderr: "", stdoutTruncated: false, stderrTruncated: false };
    this.sendMessage("runtime.exited", { requestId, exit, output });
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
    let backendMetrics: AdapterChunk["metrics"] | undefined;
    const outputByteLimit = Math.min(
      MAX_OUTPUT_BYTES,
      (payload.request.max_tokens ?? 256) * 32,
    );
    try {
      for await (const chunk of this.adapter.generate(
        { jobId: payload.jobId, request: payload.request },
        controller.signal,
      )) {
        if (chunk.metrics) backendMetrics = { ...backendMetrics, ...chunk.metrics };
        if (!chunk.text) continue;
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
        inputTokens: Math.max(0, Math.round(backendMetrics?.inputTokens ?? estimateInputTokens(payload.request))),
        outputTokens: Math.max(0, Math.round(backendMetrics?.outputTokens ?? outputTokens)),
        ttftMs: Math.max(0, Math.round(backendMetrics?.ttftMs ?? (firstTokenAt ?? finished) - started)),
        activeMs: Math.max(1, Math.round(backendMetrics?.activeMs ?? activeMs)),
        ...(backendMetrics?.reusedKvTokens === undefined
          ? {}
          : { reusedKvTokens: Math.max(0, Math.round(backendMetrics.reusedKvTokens)) }),
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
      const failureMessage = errorText(error).slice(0, 300);
      this.logger.error(`Inference job ${payload.jobId} failed: ${failureMessage}`);
      this.sendMessage("task.fail", {
        jobId: payload.jobId,
        leaseId: payload.leaseId,
        code:
          error instanceof WorkerOutputLimitError
            ? "output_limit_exceeded"
            : controller.signal.aborted
              ? "cancelled"
              : "adapter_error",
        message: failureMessage,
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
    const [metrics, liveHardware] = await Promise.all([
      this.adapter.metrics(),
      this.config.capacityScope === "host"
        ? (this.options.hardwareProbe?.() ?? probeHardware()).catch(() => null)
        : Promise.resolve(null),
    ]);
    if (liveHardware) {
      this.capabilities = {
        ...this.capabilities,
        gpus: this.capabilities.gpus.map((gpu) => {
          // gpu-N is intentionally synthetic and OS enumeration can change
          // between heartbeats. Require identity as well as id, then fall back
          // to vendor/model only; never copy telemetry from another adapter.
          const live = selectHardwareGpu(liveHardware.gpus, {
            id: gpu.id,
            vendor: gpu.vendor,
            model: gpu.model,
          });
          if (!live) return gpu;
          return {
            ...gpu,
            ...(live.utilizationPct === undefined ? {} : { utilizationPct: live.utilizationPct }),
            ...(live.temperatureC === undefined ? {} : { temperatureC: live.temperatureC }),
            ...(live.powerW === undefined ? {} : { powerW: live.powerW }),
          };
        }),
      };
    }
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
    if (this.socket.bufferedAmount > MAX_WEBSOCKET_BUFFERED_BYTES) {
      this.socket.close(4429, "runtime stream backpressure exceeded");
      return;
    }
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

function isLaunchAgentStartRequest(value: unknown): value is LaunchAgentStartRequest {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const request = value as Record<string, unknown>;
  if (
    typeof request.launchId !== "string" ||
    typeof request.pipelineId !== "string" ||
    typeof request.nodeId !== "string" ||
    !request.process ||
    typeof request.process !== "object" ||
    Array.isArray(request.process)
  ) return false;
  const process = request.process as Record<string, unknown>;
  const anchor = process.anchor;
  return typeof process.processId === "string" && !!anchor && typeof anchor === "object" &&
    !Array.isArray(anchor) && (anchor as Record<string, unknown>).memberId === request.nodeId;
}

function adapterDataLocality(config: WorkerConfig): "local" | "external" {
  if (config.adapter.kind !== "openai-compatible") return "local";
  const hostname = new URL(config.adapter.baseUrl).hostname.toLowerCase();
  return new Set(["localhost", "127.0.0.1", "::1", "[::1]"]).has(hostname)
    ? "local"
    : "external";
}
