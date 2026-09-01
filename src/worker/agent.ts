import WebSocket from "ws";
import { z } from "zod";
import { createHash } from "node:crypto";
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
  WorkerPhysicalIdentity,
  WorkerEnvelope,
  WorkerExecutorIsolationCapability,
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
  type LaunchCapturedOutput,
  type LaunchProcessHandle,
} from "../distribution/launch-supervisor.js";
import {
  validatePythonLaunchDescription,
  type PythonPipelineLaunchDescription,
} from "../distribution/python-launcher.js";
import { validateExecutorIsolationPolicy } from "../distribution/process-environment.js";
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
  plannerScalesFromProfile,
  runtimePerformanceProfileSchema,
  type RuntimePerformanceProfile,
} from "../performance/runtime-profile.js";
import {
  engineRuntimeMeasurementSchema,
  type EngineRuntimeMeasurement,
} from "../contracts/engine-runtime-profile.js";
import {
  evidenceChallengeSchema,
  type DeploymentCanaryChallenge,
  type EngineRuntimeChallenge,
  type EvidenceChallenge,
  type RuntimePerformanceChallenge,
} from "../contracts/evidence-challenge.js";
import {
  WORKER_PROTOCOL_MAX,
  WORKER_PROTOCOL_MIN,
  workerAdmissionChallengeResponseSchema,
} from "../contracts/worker-admission.js";
import { workerRegistrationDigest } from "../core/worker-admission-digest.js";
import type { WorkerAdmissionSigner } from "./admission-credential.js";
import {
  ACTIVATION_CHECKPOINT_CHUNK_BYTES,
  activationCheckpointChunks,
  activationCheckpointCompatibilitySchema,
  activationCheckpointCommittedSchema,
  activationCheckpointChunkSchema,
  activationCheckpointCommitSchema,
  activationCheckpointFailedSchema,
  activationCheckpointRequestSchema,
  activationCheckpointRestoreBeginSchema,
  activationCheckpointRestoreFailedSchema,
} from "../contracts/activation-checkpoint-transfer.js";
import {
  activationCheckpointSchema,
  signActivationCheckpointWith,
  type ActivationCheckpoint,
} from "../contracts/activation-checkpoint.js";

export interface WorkerAgentOptions {
  coordinatorUrl: string;
  networkToken?: string;
  heartbeatIntervalMs?: number;
  /** How often to probe the coordinator RTT with a WebSocket ping. */
  rttProbeIntervalMs?: number;
  reconnect?: boolean;
  identity?: {
    kind: "device" | "cell";
    id: string;
  };
  /**
   * Stable device proof used by remote coordinators. The private key never
   * leaves the worker; only one-time challenge signatures are transmitted.
   */
  admissionSigner?: WorkerAdmissionSigner | undefined;
  /** One-shot enrollment hook invoked after exact capabilities are known but before admission. */
  beforeSignedAdmission?: (input: {
    identity: { kind: "device" | "cell"; id: string };
    capabilities: WorkerCapabilities;
    protocol: { min: typeof WORKER_PROTOCOL_MIN; max: typeof WORKER_PROTOCOL_MAX };
    signer: WorkerAdmissionSigner;
    registrationDigest: string;
    signal: AbortSignal;
  }) => Promise<void>;
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
  /** Exact sealed source identity reported to the coordinator. */
  buildIdentity?: import("../contracts/build-identity.js").NativeBuildIdentity | undefined;
  /**
   * Keeps the authenticated control channel alive while compute contribution
   * is paused. Omit this for workers that cannot persist administrator
   * commands.
   */
  contributionControl?: {
    initialEnabled: boolean;
    onRemoteChange?: (enabled: boolean) => Promise<void> | void;
  };
  /** Local fail-closed admission policy for schedule and exact model allowlists. */
  workAdmissionPolicy?: (model: string | null, at: Date) => string | null;
  distributedExecutor?: {
    nodeId: string;
    stageHost: string;
    stagePort: number;
    launchAgent: LaunchAgent;
    pythonExecutable?: string;
    computeMode?: ComputeMode;
    cpuEligible?: boolean;
    acceleration?: WorkerAcceleratorDiagnostics;
    isolation?: WorkerExecutorIsolationCapability;
    /** Native peer transport. Enabled by default; options can pin listener/candidates. */
    directTransport?: RuntimeDirectTransportOptions;
    physicalIdentity?: WorkerPhysicalIdentity;
  };
  /** Captures a signed, bounded stage KV checkpoint on coordinator request. */
  activationCheckpointProvider?: (request: {
    stageRequestId: number;
    expected: z.infer<typeof activationCheckpointCompatibilitySchema>;
    maximumBytes: number;
    expiresAt: number;
  }) => Promise<{ checkpoint: ActivationCheckpoint; payload: Uint8Array }>;
  /** Live process capture primitive; WorkerAgent seals its output with the device key. */
  activationCheckpointCapture?: (request: {
    stageRequestId: number;
    expected: z.infer<typeof activationCheckpointCompatibilitySchema>;
    maximumBytes: number;
    expiresAt: number;
  }) => Promise<{ payload: Uint8Array; committedPosition: number }>;
  /** Runs the packaged, physical runtime calibration for this exact node. */
  runtimePerformanceProfileProbe?: (challenge: RuntimePerformanceChallenge) =>
    Promise<RuntimePerformanceProfile | null | undefined>;
  /** Runs the loaded engine's physical layer/KV/verify calibration. */
  engineRuntimeProfileProbe?: (challenge: EngineRuntimeChallenge) =>
    Promise<EngineRuntimeMeasurement | null | undefined>;
  logger?: Pick<Console, "info" | "warn" | "error">;
}

const MAX_SERVER_MESSAGE_BYTES = 2 * 1024 * 1024;
const MAX_OUTPUT_CHUNK_BYTES = 64 * 1024;
const MAX_OUTPUT_BYTES = 2 * 1024 * 1024;
const RECENT_JOB_LIMIT = 2_048;
const MAX_WEBSOCKET_BUFFERED_BYTES = 8 * 1024 * 1024;
const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);
/** Tags our own RTT probes so unsolicited pongs cannot corrupt the estimate. */
const RTT_PROBE_PAYLOAD = Buffer.from("gdlp-rtt");
/**
 * Petals uses 0.2 for the same job (`overhead_delay` / EMA over peer pings) and
 * it is a reasonable default: fast enough to follow a route change, slow enough
 * that one scheduling hiccup does not move placement.
 */
const RTT_EMA_ALPHA = 0.2;

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
      type: z.literal("contribution.set"),
      payload: z
        .object({
          commandId: z.string().min(1).max(256),
          enabled: z.boolean(),
          issuedAt: z.number().int().positive(),
        })
        .strict(),
    })
    .strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("evidence.challenge"),
    payload: evidenceChallengeSchema,
  }).strict(),
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
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.checkpoint.request"),
    payload: activationCheckpointRequestSchema,
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.checkpoint.committed"),
    payload: activationCheckpointCommittedSchema,
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.checkpoint.restore.begin"),
    payload: activationCheckpointRestoreBeginSchema,
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.checkpoint.restore.chunk"),
    payload: activationCheckpointChunkSchema,
  }).strict(),
  z.object({
    ...envelopeFields,
    type: z.literal("runtime.checkpoint.restore.commit"),
    payload: activationCheckpointCommitSchema,
  }).strict(),
]);

const registrationResponseSchema = z
  .object({
    workerId: z.string().min(1).max(256),
    protocolVersion: z.literal(1),
    credentialFingerprint: z.string().regex(/^sha256:[a-f0-9]{64}$/).optional(),
    workerSessionToken: z.string().min(1).max(4_096).optional(),
    nodeGeneration: z.number().int().positive().optional(),
    enrollment: z.enum(["enrolled", "accepted", "local-legacy"]).optional(),
  })
  .strict();

type ValidatedServerMessage = z.infer<typeof serverMessageSchema>;

interface PendingRuntimeLinkProbe {
  destinationNodeId: string;
  payloadBytes: number;
  startedAt: bigint;
  timeout: NodeJS.Timeout;
}

interface PendingActivationCheckpointRestore {
  begin: z.infer<typeof activationCheckpointRestoreBeginSchema>;
  chunks: Buffer[];
  receivedBytes: number;
}

const MAX_PENDING_RUNTIME_LINK_PROBES = 64;
const RUNTIME_RECONNECT_GRACE_MS = 45_000;

export class WorkerAgent {
  private readonly adapter: InferenceAdapter;
  private readonly coordinatorBaseUrl: URL;
  private registeredWorkerId: string | undefined;
  private workerSessionToken: string | undefined;
  private registeredNodeGeneration: number | undefined;
  private capabilities: WorkerCapabilities | null = null;
  private socket: WebSocket | null = null;
  private rttProbeTimer: NodeJS.Timeout | null = null;
  private pendingRttProbe: bigint | null = null;
  private lastRttSampleMs: number | null = null;
  private stopped = false;
  private readonly startupAbortController = new AbortController();
  private directTransportStartPromise: Promise<DirectTransportAdvertisement | null> | null = null;
  private stopPromise: Promise<void> | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private readonly activeJobs = new Map<string, AbortController>();
  private readonly recentJobs = new Map<string, number>();
  private readonly authorizedRuntimeProcesses = new Map<string, string>();
  private readonly preparedRuntimeProcesses = new Map<string, import("../distribution/python-launcher.js").PythonLaunchProcess>();
  private readonly preparedRuntimeModels = new Map<string, string>();
  private runtimePreparationGeneration = 0;
  private runtimePreparationTail: Promise<void> = Promise.resolve();
  private preparedRuntimeFormation: {
    launchId: string;
    pipelineId: string;
    deploymentGeneration: number;
  } | null = null;
  private readonly runtimeProcesses = new Map<string, LaunchProcessHandle>();
  private readonly runtimeStartRequests = new Map<string, string>();
  private readonly readyRuntimeOutputs = new Map<string, LaunchCapturedOutput>();
  private readonly runtimeProcessStages = new Map<string, string>();
  private readonly pendingActivationCheckpointRestores = new Map<string, PendingActivationCheckpointRestore>();
  private readonly runtimeLinkProbes = new Map<string, PendingRuntimeLinkProbe>();
  private readonly activeEvidenceChallenges = new Set<string>();
  private activeRuntimeOperations = 0;
  private readonly runtimeTunnel: RuntimeStreamTunnel | null;
  private directTransportAdvertisement: DirectTransportAdvertisement | null = null;
  private readonly logger: Pick<Console, "info" | "warn" | "error">;
  private runtimeCapacityGeneration = 0;
  private runtimeDisconnectTimer: NodeJS.Timeout | null = null;
  private contributionEnabled: boolean;
  private updateDraining = false;
  private coordinatorReady = false;

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
    this.contributionEnabled = options.contributionControl?.initialEnabled ?? true;
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
    const signal = this.startupAbortController.signal;
    try {
      this.assertStartupActive(signal);
      this.directTransportStartPromise = this.runtimeTunnel?.startDirectTransport()
        ?? Promise.resolve(null);
      const directTransportAdvertisement = await withAbort(
        this.directTransportStartPromise,
        signal,
      );
      this.assertStartupActive(signal);
      this.directTransportAdvertisement = directTransportAdvertisement;
      const capabilities = await withAbort(this.buildCapabilities(), signal);
      this.assertStartupActive(signal);
      this.capabilities = capabilities;
      this.updateFreeSlots();
      await this.register(signal);
      this.assertStartupActive(signal);
      let delayMs = 500;
      do {
        try {
          await this.connectOnce(signal);
          delayMs = 500;
        } catch (error) {
          if (!this.stopped) this.logger.warn(`Worker connection failed: ${errorText(error)}`);
        }
        if (this.stopped || this.options.reconnect === false) break;
        await delay(delayMs, signal);
        delayMs = Math.min(15_000, delayMs * 2);
      } while (!this.stopped);
    } catch (error) {
      if (this.stopped || signal.aborted) return;
      throw error;
    }
  }

  stop(): Promise<void> {
    this.stopped = true;
    this.startupAbortController.abort(new Error("worker_start_cancelled"));
    this.stopPromise ??= this.stopOnce();
    return this.stopPromise;
  }

  private async stopOnce(): Promise<void> {
    if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);
    this.heartbeatTimer = null;
    this.coordinatorReady = false;
    this.clearRuntimeDisconnectTimer();
    this.clearRuntimeLinkProbes();
    await this.sendGoodbye("user_requested");
    await this.closeSocket();
    await this.abortActiveJobs("Worker shutting down");
    await this.resetDistributedRuntime("worker_shutting_down");
    // A direct listener may still be inside asynchronous port discovery while
    // stop() runs. Close once immediately, then wait for that start attempt and
    // close again so it cannot publish a listener after shutdown completed.
    await this.runtimeTunnel?.close();
    await this.directTransportStartPromise?.catch(() => undefined);
    await this.runtimeTunnel?.close();
    this.directTransportAdvertisement = null;
    this.registeredWorkerId = undefined;
    this.workerSessionToken = undefined;
    this.registeredNodeGeneration = undefined;
  }

  private assertStartupActive(signal: AbortSignal): void {
    if (this.stopped || signal.aborted) throw new Error("worker_start_cancelled");
  }

  get workerId(): string | undefined {
    return this.registeredWorkerId;
  }

  /** Narrow control-plane credential; callers must never log or persist it. */
  get nodeControlSession(): { token: string; generation: number } | null {
    return this.workerSessionToken && this.registeredNodeGeneration
      ? { token: this.workerSessionToken, generation: this.registeredNodeGeneration }
      : null;
  }

  get isContributionEnabled(): boolean {
    return this.contributionEnabled;
  }

  get isConnected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  /** True only after the coordinator has accepted the hello and sent server.ready. */
  get isReady(): boolean {
    return this.isConnected && this.coordinatorReady;
  }

  /**
   * Stop admitting new leases without aborting current jobs or persistent
   * stages. The returned release is idempotent and preserves any independent
   * contribution preference change made while the update was prepared.
   */
  async beginRuntimeUpdateDrain(): Promise<() => Promise<void>> {
    if (this.stopped) throw new Error("worker_is_stopped");
    if (this.updateDraining) throw new Error("worker_update_drain_already_active");
    this.updateDraining = true;
    this.updateFreeSlots();
    try {
      await this.sendHeartbeat();
    } catch (error) {
      this.updateDraining = false;
      this.updateFreeSlots();
      throw error;
    }
    let released = false;
    return async () => {
      if (released) return;
      released = true;
      this.updateDraining = false;
      this.updateFreeSlots();
      await this.sendHeartbeat();
    };
  }

  async waitForIdle(timeoutMs: number): Promise<void> {
    const deadline = Date.now() + Math.max(0, timeoutMs);
    while (this.activeJobs.size > 0 || this.activeRuntimeOperations > 0) {
      if (Date.now() >= deadline) throw new Error("worker_drain_timeout");
      await delay(Math.min(100, Math.max(1, deadline - Date.now())));
    }
  }

  async setContributionEnabled(enabled: boolean): Promise<boolean> {
    const changed = this.contributionEnabled !== enabled;
    this.contributionEnabled = enabled;
    this.updateFreeSlots();
    await this.sendHeartbeat();
    if (!enabled && changed) {
      void this.abortActiveJobs("Contribution paused by administrator")
        .then(() => this.resetDistributedRuntime("contribution_paused"))
        .catch((error) => {
          this.logger.warn(`Paused contribution cleanup failed: ${errorText(error)}`);
        });
    }
    return changed;
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
    const publicPrimary = publicHardwareGpu(primary);

    const existingExecutor = this.capabilities.distributedExecutor;
    const executorWithoutEvidence = existingExecutor
      ? (() => {
          const {
            performanceEvidence: _previousEvidence,
            engineProfiles: _engineProfiles,
            ...rest
          } = existingExecutor;
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
      ...(executorWithoutEvidence
        ? {
            distributedExecutor: {
              ...executorWithoutEvidence,
              computeMode: this.options.distributedExecutor?.computeMode ?? "automatic",
              cpuEligible: this.options.distributedExecutor?.cpuEligible === true,
              ...(this.options.distributedExecutor?.acceleration
                ? { acceleration: structuredClone(this.options.distributedExecutor.acceleration) }
                : {}),
            },
          }
        : {}),
    };
    if (this.registeredWorkerId) await this.register(this.startupAbortController.signal);
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
      await this.register(this.startupAbortController.signal);
      await this.sendHeartbeat();
    }
  }

  get activeJobCount(): number {
    return this.activeJobs.size;
  }

  /**
   * All transient work that must finish before the runtime can be replaced.
   * Persistent distributed stages are tracked by the desktop launch service.
   */
  get activeWorkCount(): number {
    return this.activeJobs.size
      + this.activeEvidenceChallenges.size
      + this.activeRuntimeOperations;
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
        if (socket.readyState === WebSocket.CONNECTING) socket.terminate();
        else socket.close(1000, "worker shutting down");
      } catch {
        finish();
      }
    });
  }

  private async buildCapabilities(): Promise<WorkerCapabilities> {
    const [hardware, adapter] = await Promise.all([
      this.options.hardwareProbe?.() ?? probeHardware(),
      this.adapter.probe(),
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
    const throughputSource =
      this.config.adapter.kind === "mycellios-pipeline"
        ? "default"
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
      ...(this.options.buildIdentity
        ? { buildIdentity: this.options.buildIdentity }
        : {}),
      ...(this.options.contributionControl
        ? {
            administration: {
              contributionControl: "mycellios-contribution-control/1" as const,
            },
          }
        : {}),
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
          tokensPerSecond: this.config.adapter.kind === "mycellios-pipeline"
            ? 1
            : this.config.deployment.tokensPerSecond ?? defaultTokensPerSecond,
          throughputSource,
          ttftMs: this.config.adapter.kind === "mycellios-pipeline"
            ? 60_000
            : this.config.deployment.ttftMs ?? defaultTtft,
          ...(this.config.adapter.kind === "mycellios-pipeline"
            ? { verificationState: "pending" as const }
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
              ...(this.options.distributedExecutor.isolation
                ? { isolation: structuredClone(this.options.distributedExecutor.isolation) }
                : {}),
              ...(this.directTransportAdvertisement
                ? { directTransport: structuredClone(this.directTransportAdvertisement) }
                : {}),
              ...(this.options.distributedExecutor.physicalIdentity
                ? { physicalIdentity: structuredClone(this.options.distributedExecutor.physicalIdentity) }
                : {}),
            },
          }
        : {}),
    };
  }

  private async measureRuntimePerformanceProfile(
    challenge: RuntimePerformanceChallenge,
  ): Promise<RuntimePerformanceProfile | undefined> {
    const probe = this.options.runtimePerformanceProfileProbe;
    if (!this.options.distributedExecutor || !probe) return undefined;
    try {
      const measured = await probe(challenge);
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
      // Do not let the coordinator seal evidence that its planner must reject.
      // A noisy physical result is retriable evidence, not usable capacity.
      plannerScalesFromProfile(profile);
      this.logger.info(
        `Native runtime performance calibration verified for ${profile.deviceName} (${profile.backend}).`,
      );
      return profile;
    } catch (error) {
      this.logger.warn(
        `Native runtime performance calibration unavailable: ${errorText(error)}`,
      );
      return undefined;
    }
  }

  private async register(signal: AbortSignal): Promise<void> {
    this.assertStartupActive(signal);
    const identity = this.options.identity ?? this.defaultIdentity();
    if (!this.capabilities) throw new Error("Worker capabilities are not initialized");
    const protocol = { min: WORKER_PROTOCOL_MIN, max: WORKER_PROTOCOL_MAX } as const;
    const registration = {
      ...(identity ? { identity } : {}),
      capabilities: this.capabilities,
      protocol,
    };
    let admission:
      | {
        challengeId: string;
        publicKey: WorkerAdmissionSigner["publicKey"];
        protocol: typeof protocol;
        registrationDigest: string;
        signature: string;
      }
      | undefined;
    if (this.options.admissionSigner) {
      if (!identity) {
        throw new Error("A stable worker identity is required for signed admission");
      }
      const registrationDigest = workerRegistrationDigest({
        identity,
        capabilities: this.capabilities,
        protocol,
      });
      await this.options.beforeSignedAdmission?.({
        identity,
        capabilities: this.capabilities,
        protocol,
        signer: this.options.admissionSigner,
        registrationDigest,
        signal,
      });
      this.assertStartupActive(signal);
      const challengeResponse = await fetch(
        coordinatorHttpUrl(
          this.coordinatorBaseUrl,
          "internal/v1/workers/admission-challenge",
        ),
        {
          method: "POST",
          headers: this.requestHeaders({ "content-type": "application/json" }),
          body: JSON.stringify({
            identity,
            publicKey: this.options.admissionSigner.publicKey,
            protocol,
            registrationDigest,
          }),
          signal: AbortSignal.any([signal, AbortSignal.timeout(10_000)]),
          redirect: "manual",
        },
      );
      if (!challengeResponse.ok) {
        throw new Error(
          `Worker admission challenge failed with HTTP ${challengeResponse.status}`,
        );
      }
      const challengeText = await readResponseTextLimited(challengeResponse, 16 * 1024);
      let challengeJson: unknown;
      try {
        challengeJson = JSON.parse(challengeText) as unknown;
      } catch {
        throw new Error("Worker admission challenge returned invalid JSON");
      }
      const challenge = workerAdmissionChallengeResponseSchema.parse(challengeJson);
      this.assertStartupActive(signal);
      admission = {
        challengeId: challenge.challengeId,
        publicKey: this.options.admissionSigner.publicKey,
        protocol,
        registrationDigest,
        signature: this.options.admissionSigner.sign(challenge.signingPayload),
      };
    }
    const response = await fetch(
      coordinatorHttpUrl(this.coordinatorBaseUrl, "internal/v1/workers/register"),
      {
        method: "POST",
        headers: this.requestHeaders({ "content-type": "application/json" }),
        body: JSON.stringify({
          ...registration,
          ...(admission ? { admission } : {}),
        }),
        signal: AbortSignal.any([signal, AbortSignal.timeout(10_000)]),
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
    this.assertStartupActive(signal);
    this.registeredWorkerId = body.workerId;
    this.workerSessionToken = body.workerSessionToken;
    this.registeredNodeGeneration = body.nodeGeneration;
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

  private connectOnce(signal: AbortSignal): Promise<void> {
    this.assertStartupActive(signal);
    if (!this.registeredWorkerId) throw new Error("Worker has not been registered");
    this.coordinatorReady = false;
    const url = coordinatorWebSocketUrl(
      this.coordinatorBaseUrl,
      "internal/v1/workers/connect",
    );

    return new Promise((resolve, reject) => {
      let opened = false;
      const socket = new WebSocket(url, {
        maxPayload: MAX_SERVER_MESSAGE_BYTES,
        ...(this.options.networkToken || this.workerSessionToken
          ? { headers: { authorization: `Bearer ${this.options.networkToken ?? this.workerSessionToken}` } }
          : {}),
      });
      this.socket = socket;
      const abortStartup = () => {
        if (socket.readyState === WebSocket.CONNECTING) socket.terminate();
      };
      signal.addEventListener("abort", abortStartup, { once: true });
      socket.on("open", () => {
        if (this.stopped || signal.aborted) {
          socket.close(1000, "worker startup cancelled");
          return;
        }
        opened = true;
        this.sendMessage("worker.hello", {});
        this.startRttProbe(socket);
      });
      socket.on("pong", (payload: Buffer) => {
        this.recordRttSample(payload);
      });
      socket.on("message", (raw) => {
        if (this.stopped || signal.aborted) return;
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
        signal.removeEventListener("abort", abortStartup);
        this.coordinatorReady = false;
        if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);
        this.heartbeatTimer = null;
        this.stopRttProbe();
        if (this.socket === socket) this.socket = null;
        this.clearRuntimeLinkProbes();
        this.clearActivationCheckpointRestores();
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

  /**
   * Measure the real round-trip time to the coordinator.
   *
   * Until now `coordinatorRttMs` was hardcoded to 0 and nothing ever wrote it,
   * so every consumer scored placement on a constant: the scheduler's latency
   * term (`scheduler.ts`) was always 0, and the desktop planner concluded that
   * every worker-to-worker link cost 0.1 ms. Measured reality on the fleet is
   * 54-437 ms per hop, and picking the right node is worth ~1.8x — a decision
   * the planner could not make because it never saw the number.
   *
   * WebSocket ping/pong is the honest probe: it rides the same connection as
   * the data, needs nothing from the coordinator, and cannot be confused with
   * application queueing. Samples are smoothed with an EMA so one scheduling
   * hiccup does not move placement.
   */
  private startRttProbe(socket: WebSocket): void {
    this.stopRttProbe();
    const probe = (): void => {
      if (socket.readyState !== WebSocket.OPEN) return;
      // An outstanding probe means the previous pong never came back. Leaving
      // the old timestamp in place would turn a lost pong into an absurd RTT.
      this.pendingRttProbe = process.hrtime.bigint();
      try {
        socket.ping(RTT_PROBE_PAYLOAD);
      } catch {
        this.pendingRttProbe = null;
      }
    };
    probe();
    this.rttProbeTimer = setInterval(probe, this.options.rttProbeIntervalMs ?? 15_000);
    this.rttProbeTimer.unref?.();
  }

  private stopRttProbe(): void {
    if (this.rttProbeTimer) clearInterval(this.rttProbeTimer);
    this.rttProbeTimer = null;
    this.pendingRttProbe = null;
  }

  private recordRttSample(payload: Buffer): void {
    const sentAt = this.pendingRttProbe;
    // Only answer our own probes: `ws` also emits `pong` for unsolicited frames
    // and for the library's own keepalive, which would corrupt the estimate.
    if (sentAt === null || !payload.equals(RTT_PROBE_PAYLOAD)) return;
    this.pendingRttProbe = null;
    const sampleMs = Number(process.hrtime.bigint() - sentAt) / 1_000_000;
    if (!Number.isFinite(sampleMs) || sampleMs < 0) return;
    const previous = this.capabilities?.network.coordinatorRttMs;
    const smoothed = previous === undefined || previous <= 0
      ? sampleMs
      : previous * (1 - RTT_EMA_ALPHA) + sampleMs * RTT_EMA_ALPHA;
    // Report the rounded value: sub-microsecond precision is noise, and the
    // wire schema only promises a non-negative number.
    const rounded = Math.round(smoothed * 100) / 100;
    if (this.capabilities) this.capabilities.network.coordinatorRttMs = rounded;
    this.lastRttSampleMs = sampleMs;
  }

  /** Última muestra cruda de RTT, sin suavizar. Diagnóstico y pruebas. */
  get measuredRttMs(): number | null {
    return this.lastRttSampleMs;
  }

  private requestHeaders(initial: Record<string, string> = {}): Record<string, string> {
    if (!this.options.networkToken) return initial;
    return { ...initial, authorization: `Bearer ${this.options.networkToken}` };
  }

  private async handleServerMessage(input: unknown): Promise<void> {
    if (this.stopped) return;
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
        if (this.stopped) return;
        this.runtimeTunnel?.transportConnected();
        this.coordinatorReady = true;
        this.heartbeatTimer = setInterval(
          () => void this.sendHeartbeat(),
          this.options.heartbeatIntervalMs ?? 5_000,
        );
        break;
      }
      case "contribution.set": {
        const { commandId, enabled } = message.payload;
        if (!this.options.contributionControl) {
          this.sendMessage("contribution.ack", {
            commandId,
            enabled,
            changed: false,
            applied: false,
            error: "contribution_control_not_supported",
          });
          break;
        }
        try {
          await this.options.contributionControl.onRemoteChange?.(enabled);
          const changed = await this.setContributionEnabled(enabled);
          this.sendMessage("contribution.ack", {
            commandId,
            enabled,
            changed,
            applied: true,
          });
        } catch (error) {
          this.sendMessage("contribution.ack", {
            commandId,
            enabled,
            changed: false,
            applied: false,
            error: errorText(error),
          });
        }
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
      case "evidence.challenge":
        void this.handleEvidenceChallenge(message.payload);
        break;
      case "runtime.prepare":
        if (!this.baseAcceptsNewWork()) {
          this.sendMessage("runtime.prepared", {
            requestId: message.payload.requestId,
            ok: false,
            error: "contribution_paused",
          });
          break;
        }
        {
          const generation = ++this.runtimePreparationGeneration;
          this.authorizedRuntimeProcesses.clear();
          this.preparedRuntimeProcesses.clear();
          this.preparedRuntimeFormation = null;
          const preparation = this.runtimePreparationTail.then(() =>
            this.prepareDistributedRuntime(
              message.payload.requestId,
              message.payload.description,
              generation,
            )
          );
          this.runtimePreparationTail = preparation.catch(() => undefined);
          await this.runRuntimeOperation(() => preparation);
        }
        break;
      case "runtime.start":
        if (!this.baseAcceptsNewWork()) {
          this.sendMessage("runtime.exited", {
            requestId: message.payload.requestId,
            exit: { code: null, signal: null, error: "contribution_paused" },
            output: {
              stdout: "",
              stderr: "",
              stdoutTruncated: false,
              stderrTruncated: false,
            },
          });
          break;
        }
        await this.runRuntimeOperation(() =>
          this.startDistributedRuntime(
            message.payload.requestId,
            message.payload.request,
          ),
        );
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
      case "runtime.checkpoint.request":
        await this.publishActivationCheckpoint(message.payload);
        break;
      case "runtime.checkpoint.committed":
        // The coordinator owns durable admission. This acknowledgement is
        // deliberately informational: replaying it cannot mutate worker state.
        break;
      case "runtime.checkpoint.restore.begin":
        this.beginActivationCheckpointRestore(message.payload);
        break;
      case "runtime.checkpoint.restore.chunk":
        this.appendActivationCheckpointRestoreChunk(message.payload);
        break;
      case "runtime.checkpoint.restore.commit":
        await this.commitActivationCheckpointRestore(message.payload);
        break;
    }
  }

  private beginActivationCheckpointRestore(
    begin: z.infer<typeof activationCheckpointRestoreBeginSchema>,
  ): void {
    if (begin.expiresAt <= Date.now()) {
      this.sendMessage("runtime.checkpoint.restore.failed", {
        transferId: begin.transferId,
        checkpointId: begin.checkpoint.checkpointId,
        code: "checkpoint_restore_expired",
      });
      return;
    }
    if (this.pendingActivationCheckpointRestores.has(begin.transferId)) {
      throw new Error("activation_checkpoint_restore_transfer_is_duplicate");
    }
    this.pendingActivationCheckpointRestores.set(begin.transferId, {
      begin: structuredClone(begin),
      chunks: [],
      receivedBytes: 0,
    });
  }

  private appendActivationCheckpointRestoreChunk(
    chunk: z.infer<typeof activationCheckpointChunkSchema>,
  ): void {
    const pending = this.pendingActivationCheckpointRestores.get(chunk.transferId);
    if (!pending) throw new Error("activation_checkpoint_restore_transfer_is_unknown");
    if (pending.begin.expiresAt <= Date.now()) {
      this.failActivationCheckpointRestore(pending, "checkpoint_restore_expired");
      return;
    }
    if (
      chunk.checkpointId !== pending.begin.checkpoint.checkpointId
      || chunk.index !== pending.chunks.length
      || chunk.index >= pending.begin.chunkCount
    ) {
      this.failActivationCheckpointRestore(pending, "checkpoint_restore_incompatible");
      return;
    }
    const bytes = Buffer.from(chunk.data, "base64");
    const expectedLength = chunk.index === pending.begin.chunkCount - 1
      ? pending.begin.checkpoint.bytes - ACTIVATION_CHECKPOINT_CHUNK_BYTES * chunk.index
      : ACTIVATION_CHECKPOINT_CHUNK_BYTES;
    if (
      bytes.byteLength !== expectedLength
      || pending.receivedBytes + bytes.byteLength > pending.begin.maximumBytes
    ) {
      bytes.fill(0);
      this.failActivationCheckpointRestore(pending, "checkpoint_restore_incompatible");
      return;
    }
    pending.chunks.push(bytes);
    pending.receivedBytes += bytes.byteLength;
  }

  private async commitActivationCheckpointRestore(
    commit: z.infer<typeof activationCheckpointCommitSchema>,
  ): Promise<void> {
    const pending = this.pendingActivationCheckpointRestores.get(commit.transferId);
    if (!pending) throw new Error("activation_checkpoint_restore_transfer_is_unknown");
    const { begin } = pending;
    if (begin.expiresAt <= Date.now()) {
      this.failActivationCheckpointRestore(pending, "checkpoint_restore_expired");
      return;
    }
    if (
      commit.checkpointId !== begin.checkpoint.checkpointId
      || pending.chunks.length !== begin.chunkCount
      || pending.receivedBytes !== begin.checkpoint.bytes
    ) {
      this.failActivationCheckpointRestore(pending, "checkpoint_restore_incomplete");
      return;
    }
    const payload = Buffer.concat(pending.chunks, pending.receivedBytes);
    try {
      if (
        begin.checkpoint.payloadDigest
          !== `sha256:${createHash("sha256").update(payload).digest("hex")}`
      ) throw new Error("activation_checkpoint_restore_payload_digest_is_invalid");
      const handle = this.runtimeProcesses.get(begin.targetLaunchRequestId);
      if (
        !handle
        || this.runtimeProcessStages.get(begin.targetLaunchRequestId) !== begin.expected.stageId
        || typeof handle.restoreActivationCheckpoint !== "function"
      ) {
        this.failActivationCheckpointRestore(pending, "checkpoint_restore_process_unavailable");
        return;
      }
      await handle.restoreActivationCheckpoint(
        begin.targetStageRequestId,
        payload,
        begin.checkpoint.committedPosition,
        begin.maximumBytes,
      );
      this.abortActivationCheckpointRestore(begin.transferId);
      this.sendMessage("runtime.checkpoint.restored", {
        transferId: begin.transferId,
        checkpointId: begin.checkpoint.checkpointId,
      });
    } catch {
      this.failActivationCheckpointRestore(pending, "checkpoint_restore_failed");
    } finally {
      payload.fill(0);
    }
  }

  private failActivationCheckpointRestore(
    pending: PendingActivationCheckpointRestore,
    code: z.infer<typeof activationCheckpointRestoreFailedSchema>["code"],
  ): void {
    this.abortActivationCheckpointRestore(pending.begin.transferId);
    this.sendMessage("runtime.checkpoint.restore.failed", {
      transferId: pending.begin.transferId,
      checkpointId: pending.begin.checkpoint.checkpointId,
      code,
    });
  }

  private abortActivationCheckpointRestore(transferId: string): boolean {
    const pending = this.pendingActivationCheckpointRestores.get(transferId);
    if (!pending || !this.pendingActivationCheckpointRestores.delete(transferId)) return false;
    for (const chunk of pending.chunks) chunk.fill(0);
    return true;
  }

  private clearActivationCheckpointRestores(): void {
    for (const transferId of this.pendingActivationCheckpointRestores.keys()) {
      this.abortActivationCheckpointRestore(transferId);
    }
  }

  private async publishActivationCheckpoint(
    request: z.infer<typeof activationCheckpointRequestSchema>,
  ): Promise<void> {
    const fail = (code: z.infer<typeof activationCheckpointFailedSchema>["code"]) => {
      this.sendMessage("runtime.checkpoint.failed", { transferId: request.transferId, code });
    };
    if (request.expiresAt <= Date.now()) {
      fail("checkpoint_request_expired");
      return;
    }
    const provider = this.options.activationCheckpointProvider
      ?? this.signedActivationCheckpointProvider();
    if (!provider) {
      fail("checkpoint_provider_unavailable");
      return;
    }
    try {
      const captured = await provider({
        stageRequestId: request.stageRequestId,
        expected: structuredClone(request.expected),
        maximumBytes: request.maximumBytes,
        expiresAt: request.expiresAt,
      });
      const checkpoint = activationCheckpointSchema.parse(captured.checkpoint);
      const payload = Buffer.from(captured.payload);
      if (payload.byteLength > request.maximumBytes || payload.byteLength !== checkpoint.bytes) {
        fail("checkpoint_too_large");
        return;
      }
      if (
        checkpoint.payloadDigest
          !== `sha256:${createHash("sha256").update(payload).digest("hex")}`
      ) {
        fail("checkpoint_capture_failed");
        return;
      }
      const expected = request.expected;
      if (
        checkpoint.keyId !== expected.keyId
        || checkpoint.requestIdHash !== expected.requestIdHash
        || checkpoint.nodeIdHash !== expected.nodeIdHash
        || checkpoint.topologyGeneration !== expected.topologyGeneration
        || checkpoint.topologyDigest !== expected.topologyDigest
        || checkpoint.engineDescriptorDigest !== expected.engineDescriptorDigest
        || checkpoint.artifactManifestDigest !== expected.artifactManifestDigest
        || checkpoint.configurationDigest !== expected.configurationDigest
        || checkpoint.stageId !== expected.stageId
        || checkpoint.layerStart !== expected.layerStart
        || checkpoint.layerEnd !== expected.layerEnd
        || (
          expected.minimumCommittedPosition !== undefined
          && checkpoint.committedPosition < expected.minimumCommittedPosition
        )
      ) {
        fail("checkpoint_incompatible");
        return;
      }
      const chunks = activationCheckpointChunks(request.transferId, checkpoint, payload);
      if (!this.sendMessage("runtime.checkpoint.begin", {
        transferId: request.transferId,
        checkpoint,
        chunkCount: chunks.length,
      })) throw new Error("checkpoint_delivery_failed");
      for (const chunk of chunks) {
        if (request.expiresAt <= Date.now()) throw new Error("checkpoint_request_expired");
        await this.waitForCheckpointBackpressure(request.expiresAt);
        if (!this.sendMessage("runtime.checkpoint.chunk", chunk)) {
          throw new Error("checkpoint_delivery_failed");
        }
      }
      if (!this.sendMessage("runtime.checkpoint.commit", {
        transferId: request.transferId,
        checkpointId: checkpoint.checkpointId,
      })) throw new Error("checkpoint_delivery_failed");
    } catch (error) {
      const message = errorText(error);
      fail(message.includes("expired")
        ? "checkpoint_request_expired"
        : message.includes("delivery")
          ? "checkpoint_delivery_failed"
          : "checkpoint_capture_failed");
    }
  }

  private signedActivationCheckpointProvider(): WorkerAgentOptions["activationCheckpointProvider"] {
    const capture = this.options.activationCheckpointCapture
      ?? ((request) => this.captureActivationCheckpointFromRuntime(request));
    const signer = this.options.admissionSigner;
    if (!capture || !signer) return undefined;
    return async (request) => {
      const captured = await capture(request);
      const payload = Buffer.from(captured.payload);
      const keyId = `sha256:${createHash("sha256")
        .update(Buffer.from(signer.publicKey.spki, "base64url"))
        .digest("hex")}`;
      if (keyId !== request.expected.keyId) {
        throw new Error("activation_checkpoint_signing_key_is_incompatible");
      }
      const now = Date.now();
      return {
        payload,
        checkpoint: signActivationCheckpointWith({
          schema: "mycellios-activation-checkpoint/1",
          requestIdHash: request.expected.requestIdHash,
          nodeIdHash: request.expected.nodeIdHash,
          topologyGeneration: request.expected.topologyGeneration,
          topologyDigest: request.expected.topologyDigest,
          engineDescriptorDigest: request.expected.engineDescriptorDigest,
          artifactManifestDigest: request.expected.artifactManifestDigest,
          configurationDigest: request.expected.configurationDigest,
          stageId: request.expected.stageId,
          layerStart: request.expected.layerStart,
          layerEnd: request.expected.layerEnd,
          committedPosition: captured.committedPosition,
          payloadDigest: `sha256:${createHash("sha256").update(payload).digest("hex")}`,
          bytes: payload.byteLength,
          createdAt: now,
          expiresAt: Math.min(request.expiresAt, now + 60_000),
          keyId,
        }, (value) => signer.sign(value)),
      };
    };
  }

  private async captureActivationCheckpointFromRuntime(request: {
    stageRequestId: number;
    expected: z.infer<typeof activationCheckpointCompatibilitySchema>;
    maximumBytes: number;
    expiresAt: number;
  }): Promise<{ payload: Uint8Array; committedPosition: number }> {
    const candidates = [...this.runtimeProcesses.entries()].filter(([launchRequestId, handle]) =>
      this.runtimeProcessStages.get(launchRequestId) === request.expected.stageId
      && typeof handle.captureActivationCheckpoint === "function"
    );
    if (candidates.length !== 1) {
      throw new Error("activation_checkpoint_stage_process_is_unavailable");
    }
    const handle = candidates[0]![1];
    const captured = await handle.captureActivationCheckpoint!(
      request.stageRequestId,
      request.maximumBytes,
    );
    return {
      payload: captured.payload,
      committedPosition: captured.committedPosition,
    };
  }

  private async waitForCheckpointBackpressure(expiresAt: number): Promise<void> {
    while (
      this.socket
      && this.socket.readyState === WebSocket.OPEN
      && this.socket.bufferedAmount > MAX_WEBSOCKET_BUFFERED_BYTES / 2
    ) {
      if (Date.now() >= expiresAt) throw new Error("checkpoint_request_expired");
      await delay(5);
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

  private async handleEvidenceChallenge(challenge: EvidenceChallenge): Promise<void> {
    if (
      challenge.workerId !== this.registeredWorkerId
      || Date.now() > Date.parse(challenge.expiresAt)
      || this.activeEvidenceChallenges.has(challenge.challengeId)
    ) {
      return;
    }
    if (!this.baseAcceptsNewWork()) {
      this.sendMessage("evidence.challenge.failed", {
        challengeId: challenge.challengeId,
        nonce: challenge.nonce,
        sessionId: challenge.sessionId,
        reason: "contribution_paused",
      });
      return;
    }
    this.activeEvidenceChallenges.add(challenge.challengeId);
    try {
      if (challenge.kind === "deployment-canary") {
        await this.runDeploymentCanaryChallenge(challenge);
      } else if (challenge.kind === "runtime-performance") {
        const profile = await this.measureRuntimePerformanceProfile(challenge);
        if (!profile) throw new Error("runtime_performance_probe_unavailable");
        this.sendMessage("evidence.runtime.complete", {
          challengeId: challenge.challengeId,
          nonce: challenge.nonce,
          sessionId: challenge.sessionId,
          profile,
        });
      } else {
        const measurement = await this.measureEngineRuntimeProfile(challenge);
        if (!measurement) throw new Error("engine_runtime_probe_unavailable");
        this.sendMessage("evidence.engine-runtime.complete", {
          challengeId: challenge.challengeId,
          nonce: challenge.nonce,
          sessionId: challenge.sessionId,
          measurement,
        });
      }
    } catch (error) {
      this.sendMessage("evidence.challenge.failed", {
        challengeId: challenge.challengeId,
        nonce: challenge.nonce,
        sessionId: challenge.sessionId,
        reason: errorText(error).slice(0, 512),
      });
    } finally {
      this.activeEvidenceChallenges.delete(challenge.challengeId);
    }
  }

  private async runRuntimeOperation<T>(
    operation: () => Promise<T>,
  ): Promise<T> {
    this.activeRuntimeOperations += 1;
    try {
      return await operation();
    } finally {
      this.activeRuntimeOperations -= 1;
    }
  }

  private async measureEngineRuntimeProfile(
    challenge: EngineRuntimeChallenge,
  ): Promise<EngineRuntimeMeasurement | undefined> {
    const probe = this.options.engineRuntimeProfileProbe;
    if (!this.options.distributedExecutor || !probe) return undefined;
    const measured = await probe(challenge);
    if (!measured) return undefined;
    const measurement = engineRuntimeMeasurementSchema.parse(measured);
    if (
      measurement.samples < challenge.minimumSamples
      || measurement.capacity.contextTokens < challenge.contextTokens
      || measurement.capacity.maxKvTokens < challenge.contextTokens
      || measurement.capacity.kvBytesPerToken !== challenge.expectedKvBytesPerToken
      || measurement.capacity.maxLayerCount
        < challenge.expectedLayerEnd - challenge.expectedLayerStart
      || !scaleMatches(
        measurement.costs.decodeScale,
        measurement.costs.decodeMsPerTokenP50 / challenge.referenceDecodeMsPerToken,
      )
      || !scaleMatches(
        measurement.costs.prefillScale,
        measurement.costs.prefillMsPerTokenP50 / challenge.referencePrefillMsPerToken,
      )
      || challenge.requiredRoles.some((role) => !measurement.features.roles.includes(role))
      || Date.parse(measurement.measuredAt) < Date.parse(challenge.issuedAt) - 5_000
      || Date.parse(measurement.measuredAt) > Date.parse(challenge.expiresAt)
    ) throw new Error("engine_runtime_measurement_does_not_match_challenge");
    return measurement;
  }

  private async runDeploymentCanaryChallenge(
    challenge: DeploymentCanaryChallenge,
  ): Promise<void> {
    const deployment = this.capabilities?.deployments.find(
      (candidate) =>
        candidate.deploymentId === challenge.deploymentId
        && candidate.model === challenge.model
        && candidate.modelDigest === challenge.modelDigest
        && candidate.activationId === challenge.activationId,
    );
    if (!deployment || this.config.adapter.kind !== "mycellios-pipeline") {
      throw new Error("deployment_canary_target_is_not_local");
    }
    const deadlineAt = Date.parse(challenge.expiresAt);
    const runSample = async (sampleIndex: number, publish: boolean): Promise<void> => {
      const remainingMs = deadlineAt - Date.now();
      if (remainingMs <= 0) throw new Error("evidence_challenge_expired");
      const controller = new AbortController();
      const timeout = setTimeout(
        () => controller.abort(new Error("evidence_challenge_expired")),
        remainingMs,
      );
      let output = "";
      let backendMetrics: AdapterChunk["metrics"] | undefined;
      let nextObservedIndex = 0;
      try {
        if (publish) {
          this.sendMessage("evidence.canary.started", {
            challengeId: challenge.challengeId,
            nonce: challenge.nonce,
            sessionId: challenge.sessionId,
            sampleIndex,
          });
        }
        for await (const chunk of this.adapter.generate({
          jobId: `evidence-${challenge.challengeId}-${publish ? sampleIndex : `warmup-${sampleIndex}`}`,
          request: {
            model: challenge.model,
            messages: [{ role: "user", content: challenge.prompt }],
            max_tokens: challenge.maxOutputTokens,
            temperature: 0,
            top_p: 1,
            seed: 20_260_725,
            workload_class: "benchmark",
            // Warmups and measured samples must never share a KV/session key:
            // reuse would make sample 0 look faster than a cold independent
            // request and would corrupt the coordinator-observed comparison.
            session_id:
              `evidence-${challenge.challengeId}-${publish ? "sample" : "warmup"}-${sampleIndex}`,
            deadline_ms: Math.max(1_000, remainingMs),
          },
        }, controller.signal)) {
          if (chunk.metrics) backendMetrics = { ...backendMetrics, ...chunk.metrics };
          if (!chunk.text) continue;
          const chunkBytes = Buffer.byteLength(chunk.text, "utf8");
          if (
            chunkBytes > MAX_OUTPUT_CHUNK_BYTES
            || Buffer.byteLength(output, "utf8") + chunkBytes > MAX_OUTPUT_BYTES
          ) {
            throw new Error("deployment_canary_output_limit_exceeded");
          }
          output += chunk.text;
          if (publish) {
            this.sendMessage("evidence.canary.token", {
              challengeId: challenge.challengeId,
              nonce: challenge.nonce,
              sessionId: challenge.sessionId,
              sampleIndex,
              index: nextObservedIndex++,
              text: chunk.text,
            });
          }
        }
        if (!output) throw new Error("deployment_canary_returned_empty_output");
        if (publish) {
          const outputTokens = Math.max(
            1,
            Math.min(
              challenge.maxOutputTokens,
              Math.round(backendMetrics?.outputTokens ?? Math.ceil(output.length / 4)),
            ),
          );
          this.sendMessage("evidence.canary.complete", {
            challengeId: challenge.challengeId,
            nonce: challenge.nonce,
            sessionId: challenge.sessionId,
            sampleIndex,
            outputTokens,
            finishReason: outputTokens >= challenge.maxOutputTokens ? "length" : "stop",
          });
        }
      } finally {
        clearTimeout(timeout);
      }
    };
    for (let index = 0; index < challenge.warmupSamples; index += 1) {
      await runSample(index, false);
    }
    for (let index = 0; index < challenge.samples; index += 1) {
      await runSample(index, true);
    }
  }

  private async prepareDistributedRuntime(
    requestId: string,
    input: unknown,
    generation: number,
  ): Promise<void> {
    try {
      this.assertCurrentRuntimePreparation(generation);
      const executor = this.options.distributedExecutor;
      if (!executor) throw new Error("distributed_executor_is_not_enabled");
      validatePythonLaunchDescription(input);
      const description = input as PythonPipelineLaunchDescription;
      const policyRejection = this.workPolicyRejection(description.modelIdentity.id);
      if (policyRejection) throw new Error(policyRejection);
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
      this.assertCurrentRuntimePreparation(generation);
      await this.runtimeTunnel?.prepare(description);
      this.assertCurrentRuntimePreparation(generation);
      this.authorizedRuntimeProcesses.clear();
      this.preparedRuntimeProcesses.clear();
      this.preparedRuntimeModels.clear();
      for (const process of local) {
        this.authorizedRuntimeProcesses.set(process.processId, JSON.stringify(process));
        this.preparedRuntimeProcesses.set(process.processId, preparedById.get(process.processId)!);
        this.preparedRuntimeModels.set(process.processId, description.modelIdentity.id);
      }
      this.preparedRuntimeFormation = {
        launchId: description.launchId,
        pipelineId: description.pipelineId,
        deploymentGeneration: description.deploymentGeneration,
      };
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
      const requestIdentity = JSON.stringify(input);
      const existing = this.runtimeProcesses.get(requestId);
      if (existing) {
        if (this.runtimeStartRequests.get(requestId) !== requestIdentity) {
          this.socket?.close(4400, "runtime start identity conflict");
          return;
        }
        const output = this.readyRuntimeOutputs.get(requestId);
        if (output) this.sendMessage("runtime.ready", { requestId, output });
        return;
      }
      if (
        input.launchId !== this.preparedRuntimeFormation?.launchId
        || input.pipelineId !== this.preparedRuntimeFormation.pipelineId
        || input.deploymentGeneration !== this.preparedRuntimeFormation.deploymentGeneration
      ) {
        throw new Error("distributed_launch_formation_identity_mismatch");
      }
      if (this.authorizedRuntimeProcesses.get(input.process.processId) !== JSON.stringify(input.process)) {
        throw new Error("distributed_launch_process_was_not_prepared");
      }
      const preparedProcess = this.preparedRuntimeProcesses.get(input.process.processId);
      if (!preparedProcess) throw new Error("distributed_launch_artifact_was_not_prepared");
      const policyRejection = this.workPolicyRejection(this.preparedRuntimeModels.get(input.process.processId) ?? null);
      if (policyRejection) throw new Error(policyRejection);
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
      this.runtimeStartRequests.set(requestId, requestIdentity);
      void handle.ready.then(
        () => {
          const output = handle.output?.() ?? {
            stdout: "",
            stderr: "",
            stdoutTruncated: false,
            stderrTruncated: false,
          };
          this.readyRuntimeOutputs.set(requestId, output);
          this.sendMessage("runtime.ready", { requestId, output });
        },
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
    this.runtimePreparationGeneration += 1;
    const handles = [...this.runtimeProcesses.values()];
    this.runtimeProcesses.clear();
    this.runtimeStartRequests.clear();
    this.readyRuntimeOutputs.clear();
    this.authorizedRuntimeProcesses.clear();
    this.preparedRuntimeProcesses.clear();
    this.preparedRuntimeModels.clear();
    await Promise.all(handles.map((handle) => handle.stop(reason).catch(() => undefined)));
    await this.runtimeTunnel?.reset();
  }

  private assertCurrentRuntimePreparation(generation: number): void {
    if (generation !== this.runtimePreparationGeneration) {
      throw new Error("distributed_runtime_preparation_superseded");
    }
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
    this.runtimeStartRequests.delete(requestId);
    this.readyRuntimeOutputs.delete(requestId);
    const output = handle?.output?.() ?? { stdout: "", stderr: "", stdoutTruncated: false, stderrTruncated: false };
    this.sendMessage("runtime.exited", { requestId, exit, output });
  }

  private async execute(payload: JobPayload): Promise<void> {
    if (!this.baseAcceptsNewWork()) {
      this.sendMessage("lease.reject", {
        jobId: payload.jobId,
        leaseId: payload.leaseId,
        reason: "contribution_paused",
      });
      return;
    }
    const policyRejection = this.workPolicyRejection(payload.request.model);
    if (policyRejection) {
      this.sendMessage("lease.reject", { jobId: payload.jobId, leaseId: payload.leaseId, reason: policyRejection });
      return;
    }
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
      draining: !this.acceptsNewWork(),
      pausedReason: !this.contributionEnabled
        ? "Contribution paused"
        : this.updateDraining
          ? "Runtime update in progress"
          : null,
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
    this.sendMessage("worker.heartbeat", {
      heartbeat,
      capabilities: this.capabilities,
      metrics: {
        ...metrics,
        ready: this.acceptsNewWork() && metrics.ready,
      },
    });
  }

  private updateFreeSlots(): void {
    if (!this.capabilities) return;
    const freeSlots = this.acceptsNewWork()
      ? Math.max(0, this.config.limits.maxConcurrency - this.activeJobs.size)
      : 0;
    this.capabilities = {
      ...this.capabilities,
      deployments: this.capabilities.deployments.map((deployment) => ({
        ...deployment,
        freeSlots,
      })),
    };
  }

  private acceptsNewWork(): boolean {
    return this.baseAcceptsNewWork() && this.workPolicyRejection(null) === null;
  }

  private baseAcceptsNewWork(): boolean {
    return this.contributionEnabled && !this.updateDraining;
  }

  private workPolicyRejection(model: string | null): string | null {
    return this.options.workAdmissionPolicy?.(model, new Date()) ?? null;
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

function delay(milliseconds: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.reject(abortReason(signal));
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", abort);
      resolve();
    }, milliseconds);
    const abort = () => {
      clearTimeout(timer);
      reject(abortReason(signal!));
    };
    signal?.addEventListener("abort", abort, { once: true });
  });
}

function withAbort<T>(operation: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) return Promise.reject(abortReason(signal));
  return new Promise<T>((resolve, reject) => {
    const abort = () => reject(abortReason(signal));
    signal.addEventListener("abort", abort, { once: true });
    void operation.then(resolve, reject).finally(() => {
      signal.removeEventListener("abort", abort);
    });
  });
}

function abortReason(signal: AbortSignal): Error {
  return signal.reason instanceof Error
    ? signal.reason
    : new Error(String(signal.reason ?? "worker_start_cancelled"));
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function normalizeDeviceName(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

function scaleMatches(observed: number, rawExpected: number): boolean {
  const expected = Math.max(0.01, Math.min(100, rawExpected));
  return Number.isFinite(observed)
    && Math.abs(observed - expected) <= Math.max(1e-9, expected * 1e-6);
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
    !Number.isSafeInteger(request.deploymentGeneration) ||
    Number(request.deploymentGeneration) < 0 ||
    typeof request.nodeId !== "string" ||
    !request.process ||
    typeof request.process !== "object" ||
    Array.isArray(request.process)
  ) return false;
  const process = request.process as Record<string, unknown>;
  const anchor = process.anchor;
  if (
    typeof process.processId !== "string"
    || !anchor
    || typeof anchor !== "object"
    || Array.isArray(anchor)
    || (anchor as Record<string, unknown>).memberId !== request.nodeId
  ) {
    return false;
  }
  try {
    validateExecutorIsolationPolicy(process.isolation);
    return true;
  } catch {
    return false;
  }
}
