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
  type RuntimeStreamTransportSnapshot,
} from "./runtime-stream-tunnel.js";
import type {
  DirectTransportAdvertisement,
  RuntimeDirectTransportOptions,
} from "./runtime-direct-transport.js";
import {
  runtimePerformanceProfileSchema,
  type RuntimePerformanceProfile,
} from "../performance/runtime-profile.js";
import { deploymentMetricsFromCanaryEvidence } from "../contracts/deployment-canary.js";

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
    /** Native peer transport. Enabled by default; options can pin listener/candidates. */
    directTransport?: RuntimeDirectTransportOptions;
  };
  /** Runs the packaged, physical runtime calibration for this exact node. */
  runtimePerformanceProfileProbe?: () =>
    Promise<RuntimePerformanceProfile | null | undefined>;
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
const runtimeStreamIdSchema = z.string().min(1).max(256);
const runtimeStreamRecoveryTokenSchema = z.string()
  .min(16)
  .max(128)
  .regex(/^[A-Za-z0-9_-]+$/);
const runtimeStreamGenerationSchema = z.number().int().nonnegative().max(1_000_000);
const runtimeStreamOffsetSchema = z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER);
const directGrantSchema = z.object({
  protocol: z.literal("mycellios-direct/1"),
  connectionId: runtimeStreamIdSchema,
  sourceNodeId: runtimeStreamIdSchema,
  destinationNodeId: runtimeStreamIdSchema,
  targetPort: z.number().int().min(1).max(65_535),
  expiresAt: z.number().int().positive(),
  secret: z.string().length(43).regex(/^[A-Za-z0-9_-]+$/),
}).strict();
const directCandidateSchema = z.object({
  host: z.string().min(1).max(253),
  port: z.number().int().min(1).max(65_535),
  scope: z.enum(["lan", "configured", "public-mapped"]),
}).strict();

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
    payload: z.union([
      z.object({
        streamId: runtimeStreamIdSchema,
        targetPort: z.number().int().min(1).max(65_535),
      }).strict(),
      z.object({
        streamId: runtimeStreamIdSchema,
        targetPort: z.number().int().min(1).max(65_535),
        generation: runtimeStreamGenerationSchema,
        recoveryToken: runtimeStreamRecoveryTokenSchema,
      }).strict(),
    ]),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.direct.offer"),
    payload: z.object({
      streamId: runtimeStreamIdSchema,
      grant: directGrantSchema,
    }).strict(),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.direct.connect"),
    payload: z.object({
      streamId: runtimeStreamIdSchema,
      destinationNodeId: runtimeStreamIdSchema,
      grant: directGrantSchema,
      candidates: z.array(directCandidateSchema).min(1).max(8),
      timeoutMs: z.number().int().min(250).max(15_000),
    }).strict(),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.direct.commit"),
    payload: z.object({
      streamId: runtimeStreamIdSchema,
      connectionId: runtimeStreamIdSchema,
    }).strict(),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.direct.cancel"),
    payload: z.object({
      streamId: runtimeStreamIdSchema,
      connectionId: runtimeStreamIdSchema,
    }).strict(),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.stream.opened"),
    payload: z.union([
      z.object({ streamId: runtimeStreamIdSchema }).strict(),
      z.object({
        streamId: runtimeStreamIdSchema,
        generation: runtimeStreamGenerationSchema,
        recoveryToken: runtimeStreamRecoveryTokenSchema,
      }).strict(),
    ]),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.stream.data"),
    payload: z.union([
      z.object({
        streamId: runtimeStreamIdSchema,
        sequence: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
        data: runtimeStreamDataSchema,
      }).strict(),
      z.object({
        streamId: runtimeStreamIdSchema,
        sequence: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
        generation: runtimeStreamGenerationSchema,
        recoveryToken: runtimeStreamRecoveryTokenSchema,
        offset: runtimeStreamOffsetSchema,
        data: runtimeStreamDataSchema,
      }).strict(),
    ]),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.stream.ack"),
    payload: z.object({
      streamId: runtimeStreamIdSchema,
      generation: runtimeStreamGenerationSchema,
      recoveryToken: runtimeStreamRecoveryTokenSchema,
      acknowledgedOffset: runtimeStreamOffsetSchema,
    }).strict(),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.stream.suspend"),
    payload: z.object({
      streamId: runtimeStreamIdSchema,
      generation: runtimeStreamGenerationSchema,
      recoveryToken: runtimeStreamRecoveryTokenSchema,
      deadlineAt: z.number().int().positive(),
    }).strict(),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.stream.resumed"),
    payload: z.object({
      streamId: runtimeStreamIdSchema,
      previousGeneration: runtimeStreamGenerationSchema,
      generation: runtimeStreamGenerationSchema,
      recoveryToken: runtimeStreamRecoveryTokenSchema,
      sendFromOffset: runtimeStreamOffsetSchema,
    }).strict(),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.stream.end"),
    payload: z.union([
      z.object({ streamId: runtimeStreamIdSchema }).strict(),
      z.object({
        streamId: runtimeStreamIdSchema,
        generation: runtimeStreamGenerationSchema,
        recoveryToken: runtimeStreamRecoveryTokenSchema,
        finalOffset: runtimeStreamOffsetSchema,
      }).strict(),
    ]),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.stream.error"),
    payload: z.union([
      z.object({
        streamId: runtimeStreamIdSchema,
        message: z.string().min(1).max(1_024),
      }).strict(),
      z.object({
        streamId: runtimeStreamIdSchema,
        generation: runtimeStreamGenerationSchema,
        recoveryToken: runtimeStreamRecoveryTokenSchema,
        message: z.string().min(1).max(1_024),
      }).strict(),
    ]),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.link.probe.start"),
    payload: z.object({
      probeId: z.string().min(1).max(256),
      destinationNodeId: z.string().min(1).max(256),
      timeoutMs: z.number().int().min(100).max(60_000),
      payloadBytes: z.number().int().min(1).max(16 * 1024),
    }).strict(),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.link.probe.ping"),
    payload: z.object({
      probeId: z.string().min(1).max(256),
      data: runtimeStreamDataSchema,
    }).strict(),
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.link.probe.pong"),
    payload: z.object({
      probeId: z.string().min(1).max(256),
      data: runtimeStreamDataSchema,
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

interface PendingRuntimeLinkProbe {
  destinationNodeId: string;
  payloadBytes: number;
  startedAt: bigint;
  timeout: NodeJS.Timeout;
}

const MAX_PENDING_RUNTIME_LINK_PROBES = 64;
const RUNTIME_RECONNECT_GRACE_MS = 45_000;

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
  private readonly preparedRuntimeProcesses = new Map<string, import("../distribution/python-launcher.js").PythonLaunchProcess>();
  private readonly runtimeProcesses = new Map<string, LaunchProcessHandle>();
  private readonly runtimeLinkProbes = new Map<string, PendingRuntimeLinkProbe>();
  private readonly runtimeTunnel: RuntimeStreamTunnel | null;
  private directTransportAdvertisement: DirectTransportAdvertisement | null = null;
  private readonly logger: Pick<Console, "info" | "warn" | "error">;
  private runtimeCapacityGeneration = 0;
  private runtimeDisconnectTimer: NodeJS.Timeout | null = null;

  constructor(
    private readonly config: WorkerConfig,
    private readonly options: WorkerAgentOptions,
  ) {
    this.coordinatorBaseUrl = validateCoordinatorUrl(options.coordinatorUrl);
    if (
      config.adapter.kind === "mycellios-native"
      && options.advertiseDeployment !== false
    ) {
      throw new Error(
        "mycellios_native_control_must_not_advertise_an_inference_deployment",
      );
    }
    this.adapter = createAdapter(config);
    this.logger = options.logger ?? console;
    this.runtimeTunnel = options.distributedExecutor
      ? new RuntimeStreamTunnel(
          options.distributedExecutor.nodeId,
          (type, payload) => this.sendMessage(type, payload),
          {
            ...(options.distributedExecutor.directTransport
              ? { directTransport: options.distributedExecutor.directTransport }
              : {}),
            onDirectTransportAdvertisementChanged: (advertisement) => {
              this.applyDirectTransportAdvertisement(advertisement);
            },
          },
        )
      : null;
  }

  async start(): Promise<void> {
    this.directTransportAdvertisement = await this.runtimeTunnel?.startDirectTransport() ?? null;
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
    this.clearRuntimeDisconnectTimer();
    this.clearRuntimeLinkProbes();
    await this.abortActiveJobs("Worker shutting down");
    await this.resetDistributedRuntime("worker_shutting_down");
    await this.runtimeTunnel?.close();
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
    const performanceProfile = await this.measureRuntimePerformanceProfile();
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
    const publicPrimary = publicHardwareGpu(primary);

    const existingExecutor = this.capabilities.distributedExecutor;
    const executorWithoutProfile = existingExecutor
      ? (() => {
          const { performanceProfile: _previousProfile, ...rest } = existingExecutor;
          return rest;
        })()
      : undefined;
    this.capabilities = {
      ...this.capabilities,
      gpus: [{
        ...publicPrimary,
        offeredVramMb,
        freeOfferedVramMb,
      }],
      deployments: this.capabilities.deployments.map((deployment) => ({
        ...deployment,
        peakVramMb: this.config.deployment.peakVramMb ?? defaultPeakVramMb,
      })),
      ...(executorWithoutProfile
        ? {
            distributedExecutor: {
              ...executorWithoutProfile,
              computeMode: this.options.distributedExecutor?.computeMode ?? "automatic",
              cpuEligible: this.options.distributedExecutor?.cpuEligible === true,
              ...(this.options.distributedExecutor?.acceleration
                ? { acceleration: structuredClone(this.options.distributedExecutor.acceleration) }
                : {}),
              ...(performanceProfile
                ? { performanceProfile: structuredClone(performanceProfile) }
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
    if (this.registeredWorkerId) {
      await this.register();
      await this.sendHeartbeat();
    }
  }

  get activeJobCount(): number {
    return this.activeJobs.size;
  }

  private applyDirectTransportAdvertisement(
    advertisement: DirectTransportAdvertisement,
  ): void {
    this.directTransportAdvertisement = structuredClone(advertisement);
    const executor = this.capabilities?.distributedExecutor;
    if (!this.capabilities || !executor) return;
    this.capabilities = {
      ...this.capabilities,
      distributedExecutor: {
        ...executor,
        directTransport: structuredClone(advertisement),
      },
    };
    // Heartbeats carry the full capability document, so the coordinator drops
    // an expired public candidate on the next normal heartbeat.
  }

  runtimeTransportSnapshot(): RuntimeStreamTransportSnapshot[] {
    return this.runtimeTunnel?.transportSnapshot() ?? [];
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
    const [hardware, adapter, performanceProfile] = await Promise.all([
      this.options.hardwareProbe?.() ?? probeHardware(),
      this.adapter.probe(),
      this.measureRuntimePerformanceProfile(),
    ]);
    const selectedHardwareGpu = selectHardwareGpu(hardware.gpus, this.options.preferredHardwareGpu)
      ?? hardware.gpus[0];
    const detectedPrimary = this.options.hardwareCapacityOverride
      ?? selectRuntimeCapacityHardware(
        hardware,
        selectedHardwareGpu,
        this.options.verifiedGpuRuntime,
      );
    const primary = detectedPrimary;
    const primaryCapacityMb = primary.physicalVramMb + (primary.sharedMemoryMb ?? 0);
    const offeredVramMb = this.config.capacityScope === "cell"
      ? this.config.offeredVramMb
      : Math.min(this.config.offeredVramMb, Math.max(512, primaryCapacityMb));
    const safeBudget = safeVramBudget(offeredVramMb);
    const model = this.config.adapter.model;
    // A restarted native pipeline is a new deployment even when it serves the
    // same public model name. Binding the identifier to the artifact and
    // independently probed pipeline snapshot prevents stale scheduler state
    // from being reused across activations.
    const deploymentId = `dep-${sha256Text([
      model,
      adapter.kind,
      this.config.deployment.modelDigest ?? "",
      this.config.deployment.activationId ?? "",
    ].join(":")).slice(-12)}`;
    const canaryEvidence =
      this.config.adapter.kind === "mycellios-pipeline"
        ? this.config.deployment.canaryEvidence!
        : undefined;
    const canaryPerformance = canaryEvidence
      ? deploymentMetricsFromCanaryEvidence(canaryEvidence, {
          model,
          modelDigest: this.config.deployment.modelDigest!,
          activationId: this.config.deployment.activationId!,
        })
      : undefined;
    const throughputSource =
      this.config.adapter.kind === "mycellios-pipeline"
        ? "measured"
        : this.config.deployment.tokensPerSecond !== undefined
        ? "configured"
        : this.config.adapter.kind === "mock"
          ? "configured"
          : "default";
    const defaultTokensPerSecond =
      this.config.adapter.kind === "mock"
        ? this.config.adapter.tokensPerSecond
        : 5;
    const defaultTtft = this.config.adapter.kind === "mock" ? this.config.adapter.ttftMs : 2_000;
    const publicPrimary = publicHardwareGpu(primary);
    return {
      region: this.config.region,
      agentVersion: this.options.agentVersion?.trim() || "0.1.0",
      gpus: [
        {
          ...(this.config.capacityScope === "cell"
            ? {
                id: "cell-aggregate",
                vendor: "mycellios",
                model: `Native pipeline capacity for ${this.config.adapter.model}`,
                // Zero means no claim about a single physical GPU. The quota
                // below represents the independently measured whole cell.
                physicalVramMb: 0,
              }
            : publicPrimary),
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
          ...(this.config.deployment.activationId
            ? { activationId: this.config.deployment.activationId }
            : {}),
          mode: "replica",
          adapter: deploymentAdapterKind(adapter.kind),
          peakVramMb:
            this.config.deployment.peakVramMb ?? Math.max(512, Math.floor(safeBudget * 0.9)),
          contextLimit: this.config.deployment.contextLimit,
          maxConcurrency: this.config.limits.maxConcurrency,
          freeSlots: this.config.limits.maxConcurrency,
          tokensPerSecond:
            canaryPerformance?.tokensPerSecond
            ?? this.config.deployment.tokensPerSecond
            ?? defaultTokensPerSecond,
          throughputSource,
          ttftMs:
            canaryPerformance?.ttftMs
            ?? this.config.deployment.ttftMs
            ?? defaultTtft,
          ...(canaryEvidence
            ? { canaryEvidence: structuredClone(canaryEvidence) }
            : {}),
          dataLocality: "local",
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
      ...(this.options.distributedExecutor
        ? {
            distributedExecutor: {
              protocol: "gdlp-worker-tunnel/2" as const,
              streamRecovery: "offset-ack-v1" as const,
              nodeId: this.options.distributedExecutor.nodeId,
              stageHost: this.options.distributedExecutor.stageHost,
              stagePort: this.options.distributedExecutor.stagePort,
              runtime: "python-safetensors" as const,
              computeMode: this.options.distributedExecutor.computeMode ?? "automatic",
              cpuEligible: this.options.distributedExecutor.cpuEligible === true,
              ...(this.options.distributedExecutor.acceleration
                ? { acceleration: structuredClone(this.options.distributedExecutor.acceleration) }
                : {}),
              ...(performanceProfile
                ? { performanceProfile: structuredClone(performanceProfile) }
                : {}),
              ...(this.directTransportAdvertisement
                ? { directTransport: structuredClone(this.directTransportAdvertisement) }
                : {}),
            },
          }
        : {}),
    };
  }

  private async measureRuntimePerformanceProfile(): Promise<RuntimePerformanceProfile | undefined> {
    const probe = this.options.runtimePerformanceProfileProbe;
    if (!this.options.distributedExecutor || !probe) return undefined;
    try {
      const measured = await probe();
      if (!measured) return undefined;
      const profile = runtimePerformanceProfileSchema.parse(measured);
      const expectedBackend = this.options.verifiedGpuRuntime?.backend ?? "cpu";
      const expectedPrecision = expectedBackend === "cpu" ? "float32" : "float16";
      if (profile.backend !== expectedBackend) {
        throw new Error("runtime_performance_profile_backend_does_not_match_capacity");
      }
      if (profile.precision !== expectedPrecision) {
        throw new Error("runtime_performance_profile_precision_does_not_match_capacity");
      }
      if (
        this.options.verifiedGpuRuntime
        && normalizeDeviceName(profile.deviceName)
          !== normalizeDeviceName(this.options.verifiedGpuRuntime.deviceName)
      ) {
        throw new Error("runtime_performance_profile_device_does_not_match_capacity");
      }
      return profile;
    } catch (error) {
      this.logger.warn(
        `Native runtime performance calibration unavailable: ${errorText(error)}`,
      );
      return undefined;
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
        this.clearRuntimeLinkProbes();
        this.runtimeTunnel?.transportDisconnected();
        void this.abortActiveJobs("Coordinator disconnected");
        if (
          opened
          && !this.stopped
          && this.options.reconnect !== false
          && this.runtimeTunnel
        ) {
          this.scheduleRuntimeDisconnectReset();
          resolve();
        } else {
          void this.resetDistributedRuntime("coordinator_disconnected").finally(() => {
            if (opened) resolve();
            else reject(new Error("Coordinator connection closed before it became ready"));
          });
        }
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
        this.clearRuntimeDisconnectTimer();
        await this.sendHeartbeat();
        this.runtimeTunnel?.transportConnected();
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
      case "runtime.stream.ack":
      case "runtime.stream.suspend":
      case "runtime.stream.resumed":
      case "runtime.stream.end":
      case "runtime.stream.error":
      case "runtime.direct.offer":
      case "runtime.direct.connect":
      case "runtime.direct.commit":
      case "runtime.direct.cancel":
        await this.runtimeTunnel?.handle(message as RuntimeStreamServerMessage);
        break;
      case "runtime.link.probe.start":
        this.startRuntimeLinkProbe(
          message.payload.probeId,
          message.payload.destinationNodeId,
          message.payload.timeoutMs,
          message.payload.payloadBytes,
        );
        break;
      case "runtime.link.probe.ping":
        this.sendMessage("runtime.link.probe.pong", {
          probeId: message.payload.probeId,
          data: message.payload.data,
        });
        break;
      case "runtime.link.probe.pong":
        this.completeRuntimeLinkProbe(message.payload.probeId, message.payload.data);
        break;
    }
  }

  private startRuntimeLinkProbe(
    probeId: string,
    destinationNodeId: string,
    timeoutMs: number,
    payloadBytes: number,
  ): void {
    if (this.runtimeLinkProbes.has(probeId)) return;
    if (this.runtimeLinkProbes.size >= MAX_PENDING_RUNTIME_LINK_PROBES) {
      this.sendMessage("runtime.link.probe.result", {
        probeId,
        destinationNodeId,
        rttMs: null,
        goodputMbps: null,
      });
      return;
    }
    const timeout = setTimeout(() => {
      const pending = this.runtimeLinkProbes.get(probeId);
      if (!pending || !this.runtimeLinkProbes.delete(probeId)) return;
      this.sendMessage("runtime.link.probe.result", {
        probeId,
        destinationNodeId: pending.destinationNodeId,
        rttMs: null,
        goodputMbps: null,
      });
    }, timeoutMs);
    timeout.unref();
    this.runtimeLinkProbes.set(probeId, {
      destinationNodeId,
      payloadBytes,
      startedAt: process.hrtime.bigint(),
      timeout,
    });
    const fill = probeId.charCodeAt(probeId.length - 1) || 1;
    const data = Buffer.alloc(payloadBytes, fill).toString("base64");
    this.sendMessage("runtime.link.probe.ping", { probeId, destinationNodeId, data });
  }

  private completeRuntimeLinkProbe(probeId: string, data: string): void {
    const pending = this.runtimeLinkProbes.get(probeId);
    if (!pending || !this.runtimeLinkProbes.delete(probeId)) return;
    clearTimeout(pending.timeout);
    const receivedBytes = Buffer.from(data, "base64").byteLength;
    if (receivedBytes !== pending.payloadBytes) {
      this.sendMessage("runtime.link.probe.result", {
        probeId,
        destinationNodeId: pending.destinationNodeId,
        rttMs: null,
        goodputMbps: null,
      });
      return;
    }
    const elapsedNs = process.hrtime.bigint() - pending.startedAt;
    const rttMs = Number(elapsedNs) / 1_000_000;
    const goodputMbps = (2 * pending.payloadBytes * 8) / (rttMs * 1_000);
    this.sendMessage("runtime.link.probe.result", {
      probeId,
      destinationNodeId: pending.destinationNodeId,
      rttMs: Math.min(60_000, Math.max(Number.EPSILON, rttMs)),
      goodputMbps: Math.min(10_000_000, Math.max(Number.EPSILON, goodputMbps)),
    });
  }

  private clearRuntimeLinkProbes(): void {
    for (const probe of this.runtimeLinkProbes.values()) clearTimeout(probe.timeout);
    this.runtimeLinkProbes.clear();
  }

  private async prepareDistributedRuntime(requestId: string, input: unknown): Promise<void> {
    try {
      const executor = this.options.distributedExecutor;
      if (!executor) throw new Error("distributed_executor_is_not_enabled");
      validatePythonLaunchDescription(input);
      const description = input as PythonPipelineLaunchDescription;
      const local = description.launchOrder.filter((process) => process.anchor.memberId === executor.nodeId);
      if (local.length === 0) throw new Error("distributed_plan_has_no_process_for_this_node");
      const prepared = executor.launchAgent.prepareRuntime
        ? await executor.launchAgent.prepareRuntime(
            description,
            executor.nodeId,
            (event) => this.sendMessage("runtime.prepare.progress", {
              requestId,
              ...event,
            }),
          )
        : local;
      const preparedById = new Map(prepared.map((process) => [process.processId, process]));
      if (
        preparedById.size !== local.length
        || local.some((process) => !preparedById.has(process.processId))
      ) {
        throw new Error("distributed_runtime_preparation_did_not_cover_local_plan");
      }
      await this.runtimeTunnel?.prepare(description);
      this.authorizedRuntimeProcesses.clear();
      this.preparedRuntimeProcesses.clear();
      for (const process of local) {
        this.authorizedRuntimeProcesses.set(process.processId, JSON.stringify(process));
        this.preparedRuntimeProcesses.set(process.processId, preparedById.get(process.processId)!);
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
      const preparedProcess = this.preparedRuntimeProcesses.get(input.process.processId);
      if (!preparedProcess) throw new Error("distributed_launch_artifact_was_not_prepared");
      if (this.runtimeProcesses.has(requestId)) throw new Error("distributed_launch_request_is_duplicate");
      const controller = new AbortController();
      const tunneledProcess = this.runtimeTunnel?.rewriteProcess(preparedProcess) ?? preparedProcess;
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
    this.clearRuntimeDisconnectTimer();
    const handles = [...this.runtimeProcesses.values()];
    this.runtimeProcesses.clear();
    this.authorizedRuntimeProcesses.clear();
    this.preparedRuntimeProcesses.clear();
    await Promise.all(handles.map((handle) => handle.stop(reason).catch(() => undefined)));
    await this.runtimeTunnel?.reset();
  }

  private scheduleRuntimeDisconnectReset(): void {
    this.clearRuntimeDisconnectTimer();
    const timer = setTimeout(() => {
      if (this.runtimeDisconnectTimer !== timer) return;
      this.runtimeDisconnectTimer = null;
      void this.resetDistributedRuntime("coordinator_reconnect_timeout");
    }, RUNTIME_RECONNECT_GRACE_MS);
    timer.unref();
    this.runtimeDisconnectTimer = timer;
  }

  private clearRuntimeDisconnectTimer(): void {
    if (!this.runtimeDisconnectTimer) return;
    clearTimeout(this.runtimeDisconnectTimer);
    this.runtimeDisconnectTimer = null;
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
      const measuredTokensPerSecond = metrics.outputTokens > 0
        ? metrics.outputTokens / (metrics.activeMs / 1_000)
        : 0;
      // A normal request is useful operational telemetry, but it is not the
      // sealed multi-sample activation canary. Keep production pipeline
      // scheduling metrics immutable until a new bound canary is published.
      if (
        this.config.adapter.kind !== "mycellios-pipeline"
        && this.capabilities
        && measuredTokensPerSecond > 0
      ) {
        this.capabilities = {
          ...this.capabilities,
          deployments: this.capabilities.deployments.map((deployment) =>
            deployment.model === payload.request.model && deployment.modelDigest === payload.modelDigest
              ? {
                  ...deployment,
                  tokensPerSecond: Math.max(0.001, Number(measuredTokensPerSecond.toFixed(3))),
                  throughputSource: "measured",
                  ttftMs: metrics.ttftMs,
                }
              : deployment
          ),
        };
      }
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

  private sendMessage(type: string, payload: unknown): boolean {
    if (!this.registeredWorkerId || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return false;
    }
    if (this.socket.bufferedAmount > MAX_WEBSOCKET_BUFFERED_BYTES) {
      this.socket.close(4429, "runtime stream backpressure exceeded");
      return false;
    }
    const envelope: WorkerEnvelope = {
      v: 1,
      type,
      workerId: this.registeredWorkerId,
      payload,
    };
    this.socket.send(JSON.stringify(envelope));
    return true;
  }
}

function publicHardwareGpu(gpu: HardwareProbe["gpus"][number]): Omit<HardwareProbe["gpus"][number], "runtimeDeviceIndex"> {
  const { runtimeDeviceIndex, ...capability } = gpu;
  void runtimeDeviceIndex;
  return capability;
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

function normalizeDeviceName(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

function deploymentAdapterKind(
  adapter: InferenceAdapter["kind"],
): "mycellios-pipeline" | "mock" {
  if (adapter === "mycellios-native") {
    throw new Error("mycellios_native_control_cannot_be_a_model_deployment");
  }
  return adapter;
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
