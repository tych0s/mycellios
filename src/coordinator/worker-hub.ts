import { EventEmitter } from "node:events";
import { randomBytes, randomUUID } from "node:crypto";
import { createServer, type AddressInfo, type Server, type Socket } from "node:net";
import type { FastifyInstance, FastifyRequest } from "fastify";
import type WebSocket from "ws";
import type { ServerEnvelope, WorkerEnvelope } from "../contracts/types.js";
import {
  MAX_RUNTIME_STREAM_CHUNK_BYTES,
  parseWorkerEnvelope,
  workerEnvelopeValidationIssues,
  type WorkerHeartbeatPayload,
} from "../contracts/worker-protocol.js";
import type { MeshStore, StoredWorker } from "../storage/store.js";
import type {
  ModelDeployment,
  WorkerCapabilities,
} from "../contracts/types.js";
import type {
  RuntimeProxyHandle,
  RuntimeTransportSnapshot,
} from "../contracts/runtime-transport.js";
export type {
  RuntimeProxyHandle,
  RuntimeTransportSnapshot,
} from "../contracts/runtime-transport.js";
import {
  createCoordinatorDeploymentCanaryEvidence,
  deploymentMetricsFromCanaryEvidence,
  type DeploymentCanarySample,
} from "../contracts/deployment-canary.js";
import {
  DEPLOYMENT_CANARY_CHALLENGE_MAX_OUTPUT_TOKENS,
  DEPLOYMENT_CANARY_CHALLENGE_PROMPT,
  DEPLOYMENT_CANARY_CHALLENGE_SAMPLES,
  DEPLOYMENT_CANARY_CHALLENGE_WARMUPS,
  EVIDENCE_CHALLENGE_SCHEMA,
  EVIDENCE_CHALLENGE_TTL_MS,
  engineRuntimeChallengeSchema,
  type DeploymentCanaryChallenge,
  type EngineRuntimeChallenge,
  type EngineRuntimeChallengeRequest,
  type RuntimePerformanceChallenge,
} from "../contracts/evidence-challenge.js";
import {
  ENGINE_RUNTIME_PROFILE_DEFAULT_MAXIMUM_AGE_MS,
  engineRuntimeMeasurementSchema,
  sealEngineRuntimeProfile,
} from "../contracts/engine-runtime-profile.js";
import { canonicalEvidenceJson, sha256Text } from "../core/json.js";
import {
  createCoordinatorRuntimePerformanceEvidence,
  runtimePerformanceProfileSchema,
} from "../performance/runtime-profile.js";
import {
  DEFAULT_RUNTIME_LINK_SAMPLE_AGE_MS,
  DEFAULT_RUNTIME_LINK_SAMPLES_PER_LINK,
  RuntimeLinkObservationStore,
  type RuntimeLinkObservation,
} from "./runtime-link-observations.js";
import {
  createDirectSessionGrant,
  type DirectSessionGrant,
} from "../transport/direct-secure-channel.js";
import {
  mergeCurrentSessionEvidence,
} from "./evidence-authority.js";
import { publishCoordinatorEngineRuntimeProfile } from "./engine-profile-authority.js";
import {
  engineRuntimeProfileNeedsChallenge,
  engineRuntimeActivationPlansReady,
  type CertifiedEngineRuntimeChallengePlan,
} from "./engine-runtime-profile-scheduler.js";
import type { EngineRuntimeActivationPlan } from "../contracts/engine-runtime-activation.js";
import type { ActivationCheckpointCompatibility } from "../contracts/activation-checkpoint.js";
export type { EngineRuntimeChallengeRequest } from "../contracts/evidence-challenge.js";
import { ActivationCheckpointTransferAuthority } from "./activation-checkpoint-transfer.js";
import {
  activationCheckpointChunks,
  activationCheckpointRestoreBeginSchema,
} from "../contracts/activation-checkpoint-transfer.js";
import {
  runtimeLinkFailureEvidenceSchema,
  type RuntimeLinkFailureEvidence,
} from "../contracts/runtime-link-failure.js";

interface HubEvents {
  envelope: [WorkerEnvelope];
  disconnect: [string];
  activationCheckpointRestored: [string, string, string];
  activationCheckpointRestoreFailed: [string, string, string, string];
}

export interface AuthorizedWorkerSession {
  workerId: string;
  identityKind: "device" | "cell";
  identityId: string;
  credentialFingerprint: string;
  generation: number | null;
}

interface ConnectionState {
  socket: WebSocket;
  workerId: string | null;
  authorizedWorkerId?: string | null;
  authorizedSession?: AuthorizedWorkerSession | null;
  ready: boolean;
  helloTimer: NodeJS.Timeout;
  pending: boolean;
  messageWindowStartedAt: number;
  messagesInWindow: number;
  sessionId: string;
}

interface PendingCanarySample {
  startedAt: number;
  firstTokenAt: number | null;
  completedAt: number | null;
  nextTokenIndex: number;
  outputBytes: number;
  outputTokens: number | null;
}

interface PendingActivationCheckpointRestoreCompletion {
  workerId: string;
  workerSessionId: string;
  checkpointId: string;
  timeout: NodeJS.Timeout;
  resolve: (value: { transferId: string; checkpointId: string }) => void;
  reject: (error: Error) => void;
}

interface PendingDeploymentCanary {
  kind: "deployment-canary";
  challenge: DeploymentCanaryChallenge;
  timeout: NodeJS.Timeout;
  samples: Map<number, PendingCanarySample>;
}

interface PendingRuntimePerformance {
  kind: "runtime-performance";
  challenge: RuntimePerformanceChallenge;
  timeout: NodeJS.Timeout;
}

interface PendingEngineRuntime {
  kind: "engine-runtime";
  challenge: EngineRuntimeChallenge;
  timeout: NodeJS.Timeout;
}

type PendingEvidenceChallenge =
  | PendingDeploymentCanary
  | PendingRuntimePerformance
  | PendingEngineRuntime;

interface RuntimeStreamSession {
  streamId: string;
  sourceWorkerId: string | null;
  destinationWorkerId: string;
  sourceSequence: number;
  destinationSequence: number;
  opened: boolean;
  localSocket?: Socket;
  recovery: RuntimeStreamRecoverySession | null;
  sourceNodeId: string | null;
  destinationNodeId: string;
  targetPort: number;
  transportMode: "negotiating" | "direct" | "relay";
  direct: {
    grant: DirectSessionGrant;
    state:
      | "offered"
      | "ready"
      | "established"
      | "destination-committing"
      | "committed";
    timeout: NodeJS.Timeout;
    connectRttMs: number | null;
  } | null;
  bytesSourceToDestination: number;
  bytesDestinationToSource: number;
  createdAt: number;
  connectedAt: number | null;
  endedAt: number | null;
}

interface RuntimeStreamRecoveryReport {
  sendOffset: number;
  acknowledgedOffset: number;
  receiveOffset: number;
  bufferedFromOffset: number;
}

interface RuntimeStreamRecoverySession {
  recoveryToken: string;
  generation: number;
  sourceForwardOffset: number;
  destinationForwardOffset: number;
  suspended: boolean;
  reports: Map<string, RuntimeStreamRecoveryReport>;
  timeout: NodeJS.Timeout | null;
}

interface RuntimeLinkProbeSession {
  probeId: string;
  sourceWorkerId: string;
  destinationWorkerId: string;
  sourceNodeId: string;
  destinationNodeId: string;
  timeout: NodeJS.Timeout;
}

const MAX_RUNTIME_STREAMS = 1_024;
const MAX_RUNTIME_STREAM_BUFFERED_BYTES = 8 * 1024 * 1024;
const MAX_WEBSOCKET_BUFFERED_BYTES = 8 * 1024 * 1024;
const MAX_CONCURRENT_RUNTIME_LINK_PROBES = 32;
const RUNTIME_LINK_PROBES_PER_TICK = 8;
const RUNTIME_LINK_PROBE_INTERVAL_MS = 15_000;
const RUNTIME_LINK_PROBE_TIMEOUT_MS = 5_000;
const RUNTIME_LINK_REPROBE_AFTER_MS = 30_000;
const RUNTIME_LINK_PROBE_PAYLOAD_BYTES = 16 * 1024;
const RUNTIME_STREAM_RECOVERY_GRACE_MS = 45_000;
const DIRECT_NEGOTIATION_TIMEOUT_MS = 7_500;
const DIRECT_GRANT_TTL_MS = 10_000;
const DIRECT_ROUTE_MAX_LIFETIME_MS = 4 * 60 * 60 * 1_000;
const EVIDENCE_RETRY_AFTER_MS = 60_000;
const MAX_PENDING_EVIDENCE_CHALLENGES = 1_024;
const WORKER_HEARTBEAT_GRACE_MS = 10_000;

/**
 * Registration and the first heartbeat are separate messages. A freshly
 * authenticated socket may therefore still have the persisted offline status
 * for a few milliseconds and must not be reaped by the stale-worker sweep.
 */
export function workerConnectionIsHeartbeatStale(
  worker: Pick<StoredWorker, "status" | "lastSeenAt"> | null,
  now = Date.now(),
): boolean {
  if (!worker) return true;
  if (worker.status !== "suspect" && worker.status !== "offline") return false;
  return worker.lastSeenAt < now - WORKER_HEARTBEAT_GRACE_MS;
}

export function workerSessionSupersedes(previous: AuthorizedWorkerSession, incoming: AuthorizedWorkerSession): boolean {
  return previous.identityKind === incoming.identityKind
    && previous.identityId === incoming.identityId
    && previous.generation !== null
    && incoming.generation !== null
    && incoming.generation > previous.generation;
}

export class WorkerHub extends EventEmitter<HubEvents> {
  private readonly connections = new Map<string, ConnectionState>();
  private readonly allConnections = new Set<ConnectionState>();
  private logger: FastifyInstance["log"] | null = null;
  private pendingConnections = 0;
  private readonly runtimeStreams = new Map<string, RuntimeStreamSession>();
  private readonly completedRuntimeTransports: RuntimeTransportSnapshot[] = [];
  private readonly runtimeLinkFailures: RuntimeLinkFailureEvidence[] = [];
  private readonly runtimeProxyServers = new Set<Server>();
  private readonly runtimeLinkProbes = new Map<string, RuntimeLinkProbeSession>();
  private readonly runtimeLinkObservationsStore: RuntimeLinkObservationStore;
  private readonly runtimeLinkLastStartedAt = new Map<string, number>();
  private runtimeLinkProbeTimer: NodeJS.Timeout | null = null;
  private runtimeLinkProbeCursor = 0;
  private readonly evidenceChallenges = new Map<string, PendingEvidenceChallenge>();
  private readonly evidenceRetryAfter = new Map<string, number>();
  private readonly activationCheckpointTransfers: ActivationCheckpointTransferAuthority | null;
  private readonly pendingActivationCheckpointRestoreCompletions = new Map<
    string,
    PendingActivationCheckpointRestoreCompletion
  >();
  private sessionIsCurrent: ((session: AuthorizedWorkerSession) => boolean) | undefined;

  constructor(
    private readonly store: MeshStore,
    options: { activationCheckpointTransfers?: ActivationCheckpointTransferAuthority } = {},
  ) {
    super();
    this.activationCheckpointTransfers = options.activationCheckpointTransfers ?? null;
    this.runtimeLinkObservationsStore = new RuntimeLinkObservationStore(
      DEFAULT_RUNTIME_LINK_SAMPLE_AGE_MS,
      DEFAULT_RUNTIME_LINK_SAMPLES_PER_LINK,
      this.store.listRuntimeLinkSamples?.() ?? [],
    );
  }

  attach(
    app: FastifyInstance,
    options: {
      authorizedWorkerId?: (request: FastifyRequest) => string | null;
      authorizedSession?: (request: FastifyRequest) => AuthorizedWorkerSession | null;
      sessionIsCurrent?: (session: AuthorizedWorkerSession) => boolean;
    } = {},
  ): void {
    this.logger = app.log;
    this.sessionIsCurrent = options.sessionIsCurrent;
    app.get("/internal/v1/workers/connect", { websocket: true }, (socket, request) => {
      if (this.pendingConnections >= 256) {
        socket.close(4429, "too many pending connections");
        return;
      }
      const state: ConnectionState = {
        socket,
        workerId: null,
        authorizedWorkerId: options.authorizedWorkerId?.(request) ?? null,
        authorizedSession: options.authorizedSession?.(request) ?? null,
        ready: false,
        helloTimer: setTimeout(() => socket.close(4408, "worker hello timeout"), 5_000),
        pending: true,
        messageWindowStartedAt: Date.now(),
        messagesInWindow: 0,
        sessionId: `session-${randomUUID()}`,
      };
      this.pendingConnections += 1;
      this.allConnections.add(state);
      socket.on("message", (raw) => this.handleRawMessage(state, raw.toString()));
      socket.on("close", (code, reason) => {
        app.log.warn({
          workerId: state.workerId,
          code,
          reason: reason.toString("utf8"),
          ready: state.ready,
        }, "worker websocket closed");
        this.handleClose(state);
      });
      socket.on("error", (error) => {
        app.log.warn({
          workerId: state.workerId,
          error: error instanceof Error ? error.message : String(error),
          ready: state.ready,
        }, "worker websocket error");
        this.handleClose(state);
      });
    });
    if (!this.runtimeLinkProbeTimer) {
      this.runtimeLinkProbeTimer = setInterval(
        () => this.sampleRuntimeLinks(),
        RUNTIME_LINK_PROBE_INTERVAL_MS,
      );
      this.runtimeLinkProbeTimer.unref();
    }
  }

  connectedWorkerIds(): ReadonlySet<string> {
    return new Set(
      [...this.connections.entries()]
        .filter(([, state]) => state.ready)
        .map(([workerId]) => workerId),
    );
  }

  isConnected(workerId: string): boolean {
    return this.connections.get(workerId)?.ready === true;
  }

  runtimeTransportSnapshot(): RuntimeTransportSnapshot[] {
    return [
      ...[...this.runtimeStreams.values()].map((session) =>
        this.runtimeTransportSnapshotForSession(session)
      ),
      ...this.completedRuntimeTransports,
    ].map((snapshot) => ({ ...snapshot }));
  }

  runtimeLinkObservations(now = Date.now()): RuntimeLinkObservation[] {
    return this.runtimeLinkObservationsStore.observations(now);
  }

  runtimeLinkFailureEvidence(): RuntimeLinkFailureEvidence[] {
    return this.runtimeLinkFailures.map((evidence) => ({ ...evidence }));
  }

  requestActivationCheckpoint(
    workerId: string,
    stageRequestId: number,
    expected: ActivationCheckpointCompatibility,
    maximumBytes: number,
    expiresAt: number,
    now = Date.now(),
  ): string {
    const state = this.connections.get(workerId);
    if (!state?.ready || !this.activationCheckpointTransfers) {
      throw new Error("activation_checkpoint_transfer_worker_is_unavailable");
    }
    const transferId = this.activationCheckpointTransfers.expect({
      workerId,
      workerSessionId: state.sessionId,
      expected,
      maximumBytes,
      expiresAt,
    }, now);
    if (!this.send(workerId, "runtime.checkpoint.request", {
      transferId,
      stageRequestId,
      expected,
      maximumBytes,
      expiresAt,
    })) {
      this.activationCheckpointTransfers.abort(transferId);
      throw new Error("activation_checkpoint_transfer_request_delivery_failed");
    }
    return transferId;
  }

  async restoreActivationCheckpoint(
    workerId: string,
    targetLaunchRequestId: string,
    targetStageRequestId: number,
    checkpointId: string,
    expected: ActivationCheckpointCompatibility,
    maximumBytes: number,
    expiresAt: number,
    now = Date.now(),
  ): Promise<{ transferId: string; checkpointId: string }> {
    const state = this.connections.get(workerId);
    if (!state?.ready || !this.activationCheckpointTransfers) {
      throw new Error("activation_checkpoint_restore_worker_is_unavailable");
    }
    const prepared = this.activationCheckpointTransfers.prepareRestore({
      workerId,
      workerSessionId: state.sessionId,
      checkpointId,
      expected,
      expiresAt,
    }, now);
    if (prepared.payload.byteLength > maximumBytes) {
      prepared.payload.fill(0);
      this.activationCheckpointTransfers.abortRestore(prepared.transferId);
      throw new Error("activation_checkpoint_restore_limit_exceeded");
    }
    const chunks = activationCheckpointChunks(
      prepared.transferId, prepared.checkpoint, prepared.payload,
    );
    const completion = new Promise<{ transferId: string; checkpointId: string }>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.activationCheckpointTransfers?.abortRestore(prepared.transferId);
        this.rejectActivationCheckpointRestoreCompletion(
          prepared.transferId,
          new Error("activation_checkpoint_restore_acknowledgement_expired"),
        );
      }, Math.max(1, expiresAt - Date.now()));
      timeout.unref();
      this.pendingActivationCheckpointRestoreCompletions.set(prepared.transferId, {
        workerId,
        workerSessionId: state.sessionId,
        checkpointId: prepared.checkpoint.checkpointId,
        timeout,
        resolve,
        reject,
      });
    });
    void completion.catch(() => undefined);
    try {
      const begin = activationCheckpointRestoreBeginSchema.parse({
        transferId: prepared.transferId,
        targetLaunchRequestId,
        targetStageRequestId,
        expected,
        checkpoint: prepared.checkpoint,
        chunkCount: chunks.length,
        maximumBytes,
        expiresAt,
      });
      if (!this.send(workerId, "runtime.checkpoint.restore.begin", begin)) {
        throw new Error("activation_checkpoint_restore_delivery_failed");
      }
      for (const chunk of chunks) {
        await this.waitForActivationCheckpointBackpressure(state, expiresAt);
        if (!this.send(workerId, "runtime.checkpoint.restore.chunk", chunk)) {
          throw new Error("activation_checkpoint_restore_delivery_failed");
        }
      }
      if (!this.send(workerId, "runtime.checkpoint.restore.commit", {
        transferId: prepared.transferId,
        checkpointId: prepared.checkpoint.checkpointId,
      })) throw new Error("activation_checkpoint_restore_delivery_failed");
      return await completion;
    } catch (error) {
      this.activationCheckpointTransfers.abortRestore(prepared.transferId);
      this.rejectActivationCheckpointRestoreCompletion(
        prepared.transferId,
        error instanceof Error ? error : new Error(String(error)),
      );
      throw error;
    } finally {
      prepared.payload.fill(0);
    }
  }

  private resolveActivationCheckpointRestoreCompletion(
    transferId: string,
    checkpointId: string,
  ): void {
    const pending = this.pendingActivationCheckpointRestoreCompletions.get(transferId);
    if (!pending || pending.checkpointId !== checkpointId) {
      throw new Error("activation_checkpoint_restore_completion_is_unexpected");
    }
    this.pendingActivationCheckpointRestoreCompletions.delete(transferId);
    clearTimeout(pending.timeout);
    pending.resolve({ transferId, checkpointId });
  }

  private rejectActivationCheckpointRestoreCompletion(transferId: string, error: Error): void {
    const pending = this.pendingActivationCheckpointRestoreCompletions.get(transferId);
    if (!pending) return;
    this.pendingActivationCheckpointRestoreCompletions.delete(transferId);
    clearTimeout(pending.timeout);
    pending.reject(error);
  }

  private rejectActivationCheckpointRestoreSession(
    workerId: string,
    workerSessionId: string,
    reason: string,
  ): void {
    for (const [transferId, pending] of this.pendingActivationCheckpointRestoreCompletions) {
      if (pending.workerId === workerId && pending.workerSessionId === workerSessionId) {
        this.rejectActivationCheckpointRestoreCompletion(transferId, new Error(reason));
      }
    }
  }

  startEngineRuntimeProfileChallenge(
    workerId: string,
    request: EngineRuntimeChallengeRequest,
    now = Date.now(),
  ): string | null {
    const state = this.connections.get(workerId);
    const worker = this.store.getWorker(workerId);
    const executor = worker?.capabilities.distributedExecutor;
    if (!state?.ready || !worker || !executor || this.connections.get(workerId) !== state) {
      return null;
    }
    const key = evidenceChallengeKey(workerId, "engine-runtime", executor.nodeId);
    if (!this.challengeMayStart(key, now)) return null;
    const challenge = engineRuntimeChallengeSchema.parse({
      ...this.challengeBinding(state, now),
      kind: "engine-runtime" as const,
      nodeId: executor.nodeId,
      ...request,
    });
    const pending: PendingEngineRuntime = {
      kind: "engine-runtime",
      challenge,
      timeout: this.evidenceChallengeTimeout(challenge),
    };
    this.evidenceChallenges.set(challenge.challengeId, pending);
    if (!this.sendSocket(state.socket, "evidence.challenge", challenge)) {
      this.finishEvidenceChallenge(pending, false);
      return null;
    }
    return challenge.challengeId;
  }

  /**
   * Reconciles coordinator-certified activation plans with current physical
   * profiles. Invalid or disconnected targets fail closed in the existing
   * challenge path; retry cooldowns also prevent duplicate in-flight probes.
   */
  reconcileEngineRuntimeProfileChallenges(
    plans: readonly CertifiedEngineRuntimeChallengePlan[],
    now = Date.now(),
  ): string[] {
    const started: string[] = [];
    const seenWorkers = new Set<string>();
    for (const plan of plans) {
      if (seenWorkers.has(plan.workerId)) continue;
      seenWorkers.add(plan.workerId);
      const profiles = this.store.getWorker(plan.workerId)
        ?.capabilities.distributedExecutor?.engineProfiles ?? [];
      if (!engineRuntimeProfileNeedsChallenge(profiles, plan.request, now)) continue;
      const challengeId = this.startEngineRuntimeProfileChallenge(
        plan.workerId,
        plan.request,
        now,
      );
      if (challengeId) started.push(challengeId);
    }
    return started;
  }

  reconcileStoredEngineRuntimeProfileChallenges(now = Date.now()): string[] {
    return this.reconcileEngineRuntimeProfileChallenges(
      this.store.listEngineRuntimeActivationPlans().map((plan) => ({
        workerId: plan.workerId,
        request: plan.request,
      })),
      now,
    );
  }

  /**
   * Keeps a canaried bootstrap route private until every assigned stage has an
   * exact, coordinator-sealed runtime profile. A timeout fails activation and
   * lets the caller tear the unpublished route down.
   */
  async waitForEngineRuntimeProfiles(
    plans: readonly EngineRuntimeActivationPlan[],
    timeoutMs = EVIDENCE_CHALLENGE_TTL_MS,
    pollIntervalMs = 250,
    onPoll?: () => void | Promise<void>,
  ): Promise<void> {
    if (
      plans.length === 0
      || !Number.isFinite(timeoutMs) || timeoutMs <= 0
      || !Number.isFinite(pollIntervalMs) || pollIntervalMs <= 0
    ) throw new Error("engine_runtime_profile_gate_policy_is_invalid");
    const deadline = Date.now() + timeoutMs;
    while (!engineRuntimeActivationPlansReady(plans, this.store.listWorkers())) {
      await onPoll?.();
      this.reconcileEngineRuntimeProfileChallenges(plans);
      const remaining = deadline - Date.now();
      if (remaining <= 0) throw new Error("engine_runtime_profile_gate_timed_out");
      await new Promise<void>((resolve) => setTimeout(
        resolve,
        Math.min(pollIntervalMs, remaining),
      ));
    }
  }

  /**
   * Starts a bounded rotating sample of directed worker-to-worker relay paths.
   * The source worker owns the monotonic clock, so coordinator event-loop delay
   * before and after the round trip is not charged to the observed link.
   */
  sampleRuntimeLinks(
    maximum = RUNTIME_LINK_PROBES_PER_TICK,
    now = Date.now(),
  ): number {
    if (!Number.isInteger(maximum) || maximum < 1) return 0;
    if (this.runtimeLinkProbes.size >= MAX_CONCURRENT_RUNTIME_LINK_PROBES) return 0;
    const executors = this.store.listWorkers()
      .filter((worker) => this.isConnected(worker.id))
      .map((worker) => ({
        workerId: worker.id,
        executor: worker.capabilities.distributedExecutor,
      }))
      .filter((entry): entry is {
        workerId: string;
        executor: NonNullable<typeof entry.executor>;
      } => entry.executor?.protocol === "gdlp-worker-tunnel/2")
      .sort((left, right) => left.executor.nodeId.localeCompare(right.executor.nodeId));
    const pairs = executors.flatMap((source) => executors
      .filter((destination) => destination.workerId !== source.workerId)
      .map((destination) => ({ source, destination })));
    if (pairs.length === 0) return 0;

    let started = 0;
    let visited = 0;
    while (
      visited < pairs.length
      && started < maximum
      && this.runtimeLinkProbes.size < MAX_CONCURRENT_RUNTIME_LINK_PROBES
    ) {
      const index = this.runtimeLinkProbeCursor % pairs.length;
      this.runtimeLinkProbeCursor = (index + 1) % pairs.length;
      visited += 1;
      const pair = pairs[index]!;
      const key = runtimeLinkKey(pair.source.executor.nodeId, pair.destination.executor.nodeId);
      const alreadyPending = [...this.runtimeLinkProbes.values()].some(
        (probe) =>
          probe.sourceNodeId === pair.source.executor.nodeId
          && probe.destinationNodeId === pair.destination.executor.nodeId,
      );
      if (
        alreadyPending
        || now - (this.runtimeLinkLastStartedAt.get(key) ?? Number.NEGATIVE_INFINITY)
          < RUNTIME_LINK_REPROBE_AFTER_MS
      ) {
        continue;
      }
      if (this.startRuntimeLinkProbe(
        pair.source.workerId,
        pair.source.executor.nodeId,
        pair.destination.workerId,
        pair.destination.executor.nodeId,
        now,
      )) {
        started += 1;
      }
    }
    return started;
  }

  removeWorker(workerId: string, closeReason = "removed from mycellios panel"): boolean {
    const existed = Boolean(this.store.getWorker(workerId));
    const state = this.connections.get(workerId);
    if (state) {
      // Disconnect handling owns lease recovery, stream teardown and the
      // scheduler notification. Run it synchronously before acknowledging a
      // security-sensitive eviction; the later socket close is idempotent.
      this.handleClose(state);
      try {
        state.socket.close(4403, closeReason);
      } catch {
        // State was already detached above.
      }
    }
    return this.store.deregisterWorker(workerId) || existed;
  }

  send(workerId: string, type: string, payload: unknown): boolean {
    const state = this.connections.get(workerId);
    if (!state?.ready || state.socket.readyState !== state.socket.OPEN) return false;
    return this.sendSocket(state.socket, type, payload);
  }

  async createRuntimeProxy(destinationWorkerId: string, targetPort: number): Promise<RuntimeProxyHandle> {
    if (!this.isConnected(destinationWorkerId)) {
      throw new Error(`distributed_worker_not_connected:${destinationWorkerId}`);
    }
    if (!Number.isInteger(targetPort) || targetPort < 1 || targetPort > 65_535) {
      throw new Error("runtime_proxy_target_port_is_invalid");
    }
    const sockets = new Set<Socket>();
    const server = createServer((socket) => {
      sockets.add(socket);
      socket.once("close", () => sockets.delete(socket));
      this.attachLocalRuntimeStream(socket, destinationWorkerId, targetPort);
    });
    this.runtimeProxyServers.add(server);
    await new Promise<void>((resolve, reject) => {
      const onError = (error: Error) => {
        server.off("listening", onListening);
        reject(error);
      };
      const onListening = () => {
        server.off("error", onError);
        resolve();
      };
      server.once("error", onError);
      server.once("listening", onListening);
      server.listen(0, "127.0.0.1");
    });
    const address = server.address() as AddressInfo;
    return {
      host: "127.0.0.1",
      port: address.port,
      close: async () => {
        this.runtimeProxyServers.delete(server);
        // `server.close()` stops accepting new clients but deliberately waits
        // for existing TCP connections. A distributed inference request may
        // leave one of those connections open after a node disappears, which
        // used to block model deactivation and every later reactivation.
        // Destroy the proxy-owned sockets so Wi-Fi recovery cannot be held by
        // a stale HTTP keep-alive or an interrupted streaming response.
        const closed = new Promise<void>((resolve) => server.close(() => resolve()));
        for (const socket of sockets) socket.destroy();
        await closed;
      },
    };
  }

  close(): void {
    if (this.runtimeLinkProbeTimer) clearInterval(this.runtimeLinkProbeTimer);
    this.runtimeLinkProbeTimer = null;
    for (const state of this.allConnections) state.socket.close(1001, "coordinator shutting down");
    this.connections.clear();
    this.allConnections.clear();
    this.pendingConnections = 0;
    for (const stream of [...this.runtimeStreams.values()]) {
      this.terminateRuntimeStream(stream, "coordinator shutting down");
    }
    for (const probe of [...this.runtimeLinkProbes.values()]) {
      this.finishRuntimeLinkProbe(probe, null, null);
    }
    for (const pending of this.evidenceChallenges.values()) clearTimeout(pending.timeout);
    this.evidenceChallenges.clear();
    this.evidenceRetryAfter.clear();
    for (const transferId of this.pendingActivationCheckpointRestoreCompletions.keys()) {
      this.rejectActivationCheckpointRestoreCompletion(
        transferId,
        new Error("activation_checkpoint_restore_coordinator_closed"),
      );
    }
    for (const server of this.runtimeProxyServers) server.close();
    this.runtimeProxyServers.clear();
  }

  closeStaleConnections(): number {
    let closed = 0;
    const now = Date.now();
    for (const [workerId, state] of this.connections) {
      const worker = this.store.getWorker(workerId);
      if (!workerConnectionIsHeartbeatStale(worker, now)) continue;
      closed += 1;
      try {
        state.socket.close(4410, "worker heartbeat stale");
      } catch {
        this.handleClose(state);
      }
    }
    return closed;
  }

  private handleRawMessage(state: ConnectionState, raw: string): void {
    try {
      const now = Date.now();
      if (now - state.messageWindowStartedAt >= 1_000) {
        state.messageWindowStartedAt = now;
        state.messagesInWindow = 0;
      }
      state.messagesInWindow += 1;
      // Runtime byte streams use bounded 48 KiB chunks. A fast consumer can
      // legitimately exceed the control-plane rate while remaining under the
      // per-frame and per-stream memory ceilings enforced below.
      if (state.messagesInWindow > 2_048) {
        this.closeInvalid(state, "worker message rate exceeded", 4429);
        return;
      }
      if (Buffer.byteLength(raw, "utf8") > 2_200_000) {
        this.closeInvalid(state, "frame too large");
        return;
      }
      const decoded = JSON.parse(raw) as unknown;
      const envelope = parseWorkerEnvelope(decoded);
      if (!envelope) {
        this.logger?.warn({
          workerId: state.workerId,
          messageType: messageType(decoded),
          issues: workerEnvelopeValidationIssues(decoded),
        }, "worker message validation failed");
        this.closeInvalid(state, "invalid worker message");
        return;
      }
      if (
        state.authorizedSession
        && (envelope.type === "worker.hello" || envelope.type === "worker.heartbeat")
        && this.sessionIsCurrent
        && !this.sessionIsCurrent(state.authorizedSession)
      ) {
        this.closeInvalid(state, "worker session superseded", 4403);
        return;
      }

      if (!state.workerId) {
        if (
          envelope.type !== "worker.hello"
          || !this.store.getWorker(envelope.workerId)
          || (
            typeof state.authorizedWorkerId === "string"
            && state.authorizedWorkerId !== envelope.workerId
          )
        ) {
          this.closeInvalid(state, "invalid worker hello", 4404);
          return;
        }
        const previous = this.connections.get(envelope.workerId);
        const incomingSupersedes = previous?.authorizedSession && state.authorizedSession
          ? workerSessionSupersedes(previous.authorizedSession, state.authorizedSession)
          : false;
        if (
          previous &&
          previous !== state &&
          previous.ready &&
          previous.socket.readyState === previous.socket.OPEN &&
          !incomingSupersedes
        ) {
          // Two desktop starts can briefly overlap around an application
          // update. Keep the already-healthy connection authoritative so the
          // duplicate cannot create an endless mutual-supersession loop.
          this.closeInvalid(state, "duplicate worker connection", 4409);
          return;
        }
        if (previous && previous !== state) previous.socket.close(4409, "superseded connection");
        state.workerId = envelope.workerId;
        state.pending = false;
        this.pendingConnections = Math.max(0, this.pendingConnections - 1);
        clearTimeout(state.helloTimer);
        this.connections.set(envelope.workerId, state);
        this.sendSocket(state.socket, "server.ready", { workerId: envelope.workerId });
        return;
      }

      if (state.workerId !== envelope.workerId) {
        this.closeInvalid(state, "worker id changed", 4409);
        return;
      }
      if (envelope.type === "worker.hello") {
        this.closeInvalid(state, "duplicate worker hello", 4409);
        return;
      }
      if (envelope.type === "worker.heartbeat") {
        const wasReady = state.ready;
        state.ready = true;
        this.applyHeartbeat(state, envelope.payload);
        this.ensureEvidenceChallenges(state, envelope.payload.capabilities);
        if (!wasReady) queueMicrotask(() => this.sampleRuntimeLinks());
      }
      if (envelope.type === "worker.goodbye") {
        this.store.deregisterWorker(envelope.workerId);
        if (this.connections.get(envelope.workerId) === state) {
          this.connections.delete(envelope.workerId);
          state.ready = false;
          state.workerId = null;
          this.emit("disconnect", envelope.workerId);
        }
        state.socket.close(1000, envelope.payload.reason);
        return;
      }
      if (envelope.type.startsWith("runtime.stream.")) {
        if (this.store.getWorker(envelope.workerId)?.capabilities.distributedExecutor?.protocol
          !== "gdlp-worker-tunnel/2") {
          this.closeInvalid(state, "runtime stream protocol is not enabled", 4403);
          return;
        }
        this.handleRuntimeStreamEnvelope(envelope);
        return;
      }
      if (envelope.type.startsWith("runtime.direct.")) {
        if (
          this.store.getWorker(envelope.workerId)?.capabilities.distributedExecutor
            ?.directTransport?.protocol !== "mycellios-direct/1"
        ) {
          this.closeInvalid(state, "direct runtime transport is not enabled", 4403);
          return;
        }
        this.handleRuntimeDirectEnvelope(envelope);
        return;
      }
      if (envelope.type.startsWith("runtime.checkpoint.")) {
        if (!this.activationCheckpointTransfers) {
          this.closeInvalid(state, "activation checkpoint transfer is not enabled", 4403);
          return;
        }
        if (envelope.type === "runtime.checkpoint.begin") {
          this.activationCheckpointTransfers.begin(
            envelope.workerId, state.sessionId, envelope.payload,
          );
        } else if (envelope.type === "runtime.checkpoint.chunk") {
          this.activationCheckpointTransfers.chunk(
            envelope.workerId, state.sessionId, envelope.payload,
          );
        } else if (envelope.type === "runtime.checkpoint.commit") {
          const checkpoint = this.activationCheckpointTransfers.commit(
            envelope.workerId, state.sessionId, envelope.payload,
          );
          this.send(envelope.workerId, "runtime.checkpoint.committed", {
            transferId: envelope.payload.transferId,
            checkpointId: checkpoint.checkpointId,
          });
        } else if (envelope.type === "runtime.checkpoint.failed") {
          this.activationCheckpointTransfers.abort(envelope.payload.transferId);
        } else if (envelope.type === "runtime.checkpoint.restored") {
          this.activationCheckpointTransfers.completeRestore(
            envelope.workerId,
            state.sessionId,
            envelope.payload.transferId,
            envelope.payload.checkpointId,
          );
          this.emit(
            "activationCheckpointRestored",
            envelope.workerId,
            envelope.payload.transferId,
            envelope.payload.checkpointId,
          );
          this.resolveActivationCheckpointRestoreCompletion(
            envelope.payload.transferId,
            envelope.payload.checkpointId,
          );
        } else if (envelope.type === "runtime.checkpoint.restore.failed") {
          this.activationCheckpointTransfers.failRestore(
            envelope.workerId,
            state.sessionId,
            envelope.payload.transferId,
            envelope.payload.checkpointId,
          );
          this.emit(
            "activationCheckpointRestoreFailed",
            envelope.workerId,
            envelope.payload.transferId,
            envelope.payload.checkpointId,
            envelope.payload.code,
          );
          this.rejectActivationCheckpointRestoreCompletion(
            envelope.payload.transferId,
            new Error(envelope.payload.code),
          );
        }
        return;
      }
      if (envelope.type.startsWith("runtime.link.probe.")) {
        if (this.store.getWorker(envelope.workerId)?.capabilities.distributedExecutor?.protocol
          !== "gdlp-worker-tunnel/2") {
          this.closeInvalid(state, "runtime link probes are not enabled", 4403);
          return;
        }
        this.handleRuntimeLinkProbeEnvelope(envelope);
        return;
      }
      if (envelope.type.startsWith("evidence.")) {
        this.handleEvidenceEnvelope(state, envelope);
        return;
      }
      this.emit("envelope", envelope);
    } catch (error) {
      this.logger?.warn({
        workerId: state.workerId,
        error: error instanceof Error ? error.message : String(error),
      }, "worker message processing failed");
      this.closeInvalid(state, "invalid worker message");
    }
  }

  private applyHeartbeat(state: ConnectionState, payload: WorkerHeartbeatPayload): void {
    const workerId = state.workerId!;
    const status = payload.heartbeat.draining ? "draining" : "online";
    const current = this.store.getWorker(workerId)?.capabilities ?? null;
    this.store.updateWorkerHeartbeat(
      workerId,
      mergeCurrentSessionEvidence(workerId, state.sessionId, payload.capabilities, current),
      status,
    );
  }

  private closeInvalid(state: ConnectionState, reason: string, code = 4400): void {
    try {
      state.socket.close(code, reason);
    } catch {
      this.handleClose(state);
    }
  }

  private handleClose(state: ConnectionState): void {
    clearTimeout(state.helloTimer);
    this.allConnections.delete(state);
    if (state.pending) {
      state.pending = false;
      this.pendingConnections = Math.max(0, this.pendingConnections - 1);
    }
    if (state.workerId && this.connections.get(state.workerId) === state) {
      const disconnectedWorkerId = state.workerId;
      this.activationCheckpointTransfers?.abortSession(disconnectedWorkerId, state.sessionId);
      this.rejectActivationCheckpointRestoreSession(
        disconnectedWorkerId,
        state.sessionId,
        "activation_checkpoint_restore_worker_disconnected",
      );
      this.clearEvidenceChallengesForSession(disconnectedWorkerId, state.sessionId);
      this.connections.delete(state.workerId);
      this.store.setWorkerStatus(state.workerId, "offline");
      this.emit("disconnect", state.workerId);
      for (const stream of [...this.runtimeStreams.values()]) {
        if (
          stream.sourceWorkerId === disconnectedWorkerId ||
          stream.destinationWorkerId === disconnectedWorkerId
        ) {
          if (stream.transportMode === "direct") {
            // A committed peer route no longer depends on coordinator reachability.
            continue;
          }
          if (stream.recovery && stream.transportMode === "relay") {
            this.suspendRuntimeStream(
              stream,
              `distributed_worker_disconnected:${disconnectedWorkerId}`,
            );
          } else {
            this.terminateRuntimeStream(
              stream,
              `distributed_worker_disconnected:${disconnectedWorkerId}`,
            );
          }
        }
      }
      for (const probe of [...this.runtimeLinkProbes.values()]) {
        if (
          probe.sourceWorkerId === disconnectedWorkerId
          || probe.destinationWorkerId === disconnectedWorkerId
        ) {
          this.finishRuntimeLinkProbe(probe, null, null);
        }
      }
    }
  }

  private ensureEvidenceChallenges(
    state: ConnectionState,
    claims: WorkerCapabilities,
    now = Date.now(),
  ): void {
    const workerId = state.workerId;
    if (!workerId || !state.ready || this.connections.get(workerId) !== state) return;
    const stored = this.store.getWorker(workerId);
    if (!stored) return;

    for (const claim of claims.deployments
      .filter((deployment) => deployment.adapter === "mycellios-pipeline")
      .slice(0, 4)) {
      const current = stored.capabilities.deployments.find(
        (deployment) =>
          deployment.deploymentId === claim.deploymentId
          && deployment.modelDigest === claim.modelDigest
          && deployment.activationId === claim.activationId,
      );
      if (
        !claim.activationId
        || current?.verificationState === "verified"
        || !this.challengeMayStart(
          evidenceChallengeKey(workerId, "deployment-canary", claim.deploymentId),
          now,
        )
      ) {
        continue;
      }
      this.startDeploymentCanaryChallenge(state, claim, now);
    }

    const executor = claims.distributedExecutor;
    const currentExecutor = stored.capabilities.distributedExecutor;
    if (
      executor
      && currentExecutor?.nodeId === executor.nodeId
      && !currentExecutor.performanceEvidence
      && this.challengeMayStart(
        evidenceChallengeKey(workerId, "runtime-performance", executor.nodeId),
        now,
      )
    ) {
      const identity = runtimeChallengeIdentity(claims);
      if (identity) this.startRuntimePerformanceChallenge(state, identity, now);
    }
  }

  private challengeMayStart(key: string, now: number): boolean {
    if (this.evidenceChallenges.size >= MAX_PENDING_EVIDENCE_CHALLENGES) return false;
    if (now < (this.evidenceRetryAfter.get(key) ?? 0)) return false;
    return ![...this.evidenceChallenges.values()].some(
      (pending) => evidenceChallengeKey(
        pending.challenge.workerId,
        pending.kind,
        pending.kind === "deployment-canary"
          ? pending.challenge.deploymentId
          : pending.challenge.nodeId,
      ) === key,
    );
  }

  private startDeploymentCanaryChallenge(
    state: ConnectionState,
    deployment: ModelDeployment,
    now: number,
  ): void {
    const workerId = state.workerId!;
    const challenge = {
      ...this.challengeBinding(state, now),
      kind: "deployment-canary" as const,
      deploymentId: deployment.deploymentId,
      model: deployment.model,
      modelDigest: deployment.modelDigest,
      activationId: deployment.activationId!,
      prompt: DEPLOYMENT_CANARY_CHALLENGE_PROMPT,
      promptDigest: sha256Text(DEPLOYMENT_CANARY_CHALLENGE_PROMPT),
      maxOutputTokens: DEPLOYMENT_CANARY_CHALLENGE_MAX_OUTPUT_TOKENS,
      warmupSamples: DEPLOYMENT_CANARY_CHALLENGE_WARMUPS,
      samples: DEPLOYMENT_CANARY_CHALLENGE_SAMPLES,
    } satisfies DeploymentCanaryChallenge;
    const pending: PendingDeploymentCanary = {
      kind: "deployment-canary",
      challenge,
      timeout: this.evidenceChallengeTimeout(challenge),
      samples: new Map(),
    };
    this.evidenceChallenges.set(challenge.challengeId, pending);
    if (!this.sendSocket(state.socket, "evidence.challenge", challenge)) {
      this.finishEvidenceChallenge(pending, false);
      this.evidenceRetryAfter.set(
        evidenceChallengeKey(workerId, pending.kind, deployment.deploymentId),
        now + EVIDENCE_RETRY_AFTER_MS,
      );
    }
  }

  private startRuntimePerformanceChallenge(
    state: ConnectionState,
    identity: {
      nodeId: string;
      backend: RuntimePerformanceChallenge["backend"];
      deviceName: string;
      precision: RuntimePerformanceChallenge["precision"];
    },
    now: number,
  ): void {
    const challenge = {
      ...this.challengeBinding(state, now),
      kind: "runtime-performance" as const,
      ...identity,
      source: "physical-microbenchmark" as const,
      activationCodecId: "fp16" as const,
      minimumWarmupSamples: 1,
      minimumSamples: 7,
    } satisfies RuntimePerformanceChallenge;
    const pending: PendingRuntimePerformance = {
      kind: "runtime-performance",
      challenge,
      timeout: this.evidenceChallengeTimeout(challenge),
    };
    this.evidenceChallenges.set(challenge.challengeId, pending);
    if (!this.sendSocket(state.socket, "evidence.challenge", challenge)) {
      this.finishEvidenceChallenge(pending, false);
    }
  }

  private challengeBinding(
    state: ConnectionState,
    now: number,
  ): Pick<
    DeploymentCanaryChallenge,
    "schema" | "challengeId" | "nonce" | "sessionId" | "workerId" | "issuedAt" | "expiresAt"
  > {
    return {
      schema: EVIDENCE_CHALLENGE_SCHEMA,
      challengeId: `challenge-${randomUUID()}`,
      nonce: randomBytes(32).toString("base64url"),
      sessionId: state.sessionId,
      workerId: state.workerId!,
      issuedAt: new Date(now).toISOString(),
      expiresAt: new Date(now + EVIDENCE_CHALLENGE_TTL_MS).toISOString(),
    };
  }

  private evidenceChallengeTimeout(
    challenge:
      | DeploymentCanaryChallenge
      | RuntimePerformanceChallenge
      | EngineRuntimeChallenge,
  ): NodeJS.Timeout {
    const timeout = setTimeout(() => {
      const pending = this.evidenceChallenges.get(challenge.challengeId);
      if (pending) this.finishEvidenceChallenge(pending, false);
    }, Math.max(1, Date.parse(challenge.expiresAt) - Date.now()));
    timeout.unref();
    return timeout;
  }

  private handleEvidenceEnvelope(
    state: ConnectionState,
    envelope: WorkerEnvelope,
  ): void {
    const payload = envelope.payload as Record<string, unknown>;
    const challengeId = payload.challengeId as string;
    const pending = this.evidenceChallenges.get(challengeId);
    if (!pending || !this.evidenceResponseMatches(state, pending, payload)) {
      this.closeInvalid(state, "invalid or replayed evidence response", 4403);
      return;
    }
    const now = Date.now();
    if (now > Date.parse(pending.challenge.expiresAt)) {
      this.finishEvidenceChallenge(pending, false);
      this.closeInvalid(state, "expired evidence response", 4403);
      return;
    }

    if (envelope.type === "evidence.challenge.failed") {
      this.finishEvidenceChallenge(pending, false);
      return;
    }
    if (pending.kind === "runtime-performance") {
      if (envelope.type !== "evidence.runtime.complete") {
        this.closeInvalid(state, "evidence response kind mismatch", 4403);
        return;
      }
      this.completeRuntimePerformanceChallenge(state, pending, payload.profile, now);
      return;
    }
    if (pending.kind === "engine-runtime") {
      if (envelope.type !== "evidence.engine-runtime.complete") {
        this.closeInvalid(state, "evidence response kind mismatch", 4403);
        return;
      }
      this.completeEngineRuntimeChallenge(
        state,
        pending,
        payload.measurement,
        now,
      );
      return;
    }
    this.handleDeploymentCanaryEnvelope(state, pending, envelope.type, payload, now);
  }

  private evidenceResponseMatches(
    state: ConnectionState,
    pending: PendingEvidenceChallenge,
    payload: Record<string, unknown>,
  ): boolean {
    return state.workerId === pending.challenge.workerId
      && state.sessionId === pending.challenge.sessionId
      && payload.sessionId === pending.challenge.sessionId
      && payload.nonce === pending.challenge.nonce
      && this.connections.get(pending.challenge.workerId) === state;
  }

  private handleDeploymentCanaryEnvelope(
    state: ConnectionState,
    pending: PendingDeploymentCanary,
    type: string,
    payload: Record<string, unknown>,
    now: number,
  ): void {
    const sampleIndex = payload.sampleIndex as number;
    if (
      !Number.isInteger(sampleIndex)
      || sampleIndex < 0
      || sampleIndex >= pending.challenge.samples
    ) {
      this.closeInvalid(state, "invalid canary sample index", 4403);
      return;
    }
    if (type === "evidence.canary.started") {
      if (pending.samples.has(sampleIndex)) {
        this.closeInvalid(state, "duplicate canary sample", 4403);
        return;
      }
      pending.samples.set(sampleIndex, {
        startedAt: now,
        firstTokenAt: null,
        completedAt: null,
        nextTokenIndex: 0,
        outputBytes: 0,
        outputTokens: null,
      });
      return;
    }
    const sample = pending.samples.get(sampleIndex);
    if (!sample || sample.completedAt !== null) {
      this.closeInvalid(state, "canary sample was not started", 4403);
      return;
    }
    if (type === "evidence.canary.token") {
      const text = payload.text as string;
      const index = payload.index as number;
      const bytes = Buffer.byteLength(text, "utf8");
      if (
        !text
        || index !== sample.nextTokenIndex
        || sample.outputBytes + bytes > pending.challenge.maxOutputTokens * 64
      ) {
        this.closeInvalid(state, "invalid canary token sequence", 4403);
        return;
      }
      sample.firstTokenAt ??= now;
      sample.nextTokenIndex += 1;
      sample.outputBytes += bytes;
      return;
    }
    if (type !== "evidence.canary.complete") {
      this.closeInvalid(state, "evidence response kind mismatch", 4403);
      return;
    }
    const outputTokens = payload.outputTokens as number;
    if (
      sample.firstTokenAt === null
      || sample.outputBytes < 1
      || !Number.isInteger(outputTokens)
      || outputTokens < 1
      || outputTokens > pending.challenge.maxOutputTokens
    ) {
      this.closeInvalid(state, "invalid completed canary sample", 4403);
      return;
    }
    sample.completedAt = now;
    sample.outputTokens = outputTokens;
    if (
      pending.samples.size === pending.challenge.samples
      && [...pending.samples.values()].every((candidate) => candidate.completedAt !== null)
    ) {
      this.publishDeploymentCanaryEvidence(state, pending, now);
    }
  }

  private publishDeploymentCanaryEvidence(
    state: ConnectionState,
    pending: PendingDeploymentCanary,
    now: number,
  ): void {
    const worker = this.store.getWorker(pending.challenge.workerId);
    const deployment = worker?.capabilities.deployments.find(
      (candidate) =>
        candidate.deploymentId === pending.challenge.deploymentId
        && candidate.model === pending.challenge.model
        && candidate.modelDigest === pending.challenge.modelDigest
        && candidate.activationId === pending.challenge.activationId,
    );
    if (!worker || !deployment || this.connections.get(worker.id) !== state) {
      this.finishEvidenceChallenge(pending, false);
      return;
    }
    const samples = [...pending.samples.entries()]
      .sort(([left], [right]) => left - right)
      .map(([index, sample]): DeploymentCanarySample => ({
        sampleId: `sample-${index}`,
        outputTokens: sample.outputTokens!,
        activeMs: Math.max(1, sample.completedAt! - sample.startedAt),
        ttftMs: Math.max(0, sample.firstTokenAt! - sample.startedAt),
        completed: true,
      }));
    const evidence = createCoordinatorDeploymentCanaryEvidence({
      challengeId: pending.challenge.challengeId,
      nonce: pending.challenge.nonce,
      workerId: pending.challenge.workerId,
      sessionId: pending.challenge.sessionId,
      issuedAt: pending.challenge.issuedAt,
      expiresAt: pending.challenge.expiresAt,
      model: pending.challenge.model,
      modelDigest: pending.challenge.modelDigest,
      activationId: pending.challenge.activationId,
      promptDigest: pending.challenge.promptDigest,
      maxOutputTokens: pending.challenge.maxOutputTokens,
      observedAt: new Date(now).toISOString(),
      warmupSamples: pending.challenge.warmupSamples,
      samples,
    });
    const metrics = deploymentMetricsFromCanaryEvidence(evidence, {
      model: deployment.model,
      modelDigest: deployment.modelDigest,
      activationId: deployment.activationId!,
      workerId: worker.id,
      sessionId: state.sessionId,
      now,
    });
    const capabilities = structuredClone(worker.capabilities);
    capabilities.deployments = capabilities.deployments.map((candidate) =>
      candidate.deploymentId === deployment.deploymentId
        ? {
            ...candidate,
            verificationState: "verified",
            throughputSource: "measured",
            tokensPerSecond: metrics.tokensPerSecond,
            ttftMs: metrics.ttftMs,
            canaryEvidence: evidence,
          }
        : candidate
    );
    this.store.updateWorkerHeartbeat(worker.id, capabilities, worker.status);
    this.finishEvidenceChallenge(pending, true);
  }

  private completeRuntimePerformanceChallenge(
    state: ConnectionState,
    pending: PendingRuntimePerformance,
    input: unknown,
    now: number,
  ): void {
    try {
      const profile = runtimePerformanceProfileSchema.parse(input);
      const challenge = pending.challenge;
      if (
        profile.backend !== challenge.backend
        || normalizeDeviceName(profile.deviceName) !== normalizeDeviceName(challenge.deviceName)
        || profile.precision !== challenge.precision
        || profile.source !== challenge.source
        || profile.activationCodecId !== challenge.activationCodecId
        || [
          profile.decodeMemory,
          profile.prefillCompute,
          profile.activationCodec,
        ].some((series) =>
          series.warmupSamples < challenge.minimumWarmupSamples
          || series.samples < challenge.minimumSamples
        )
      ) {
        throw new Error("runtime_performance_profile_does_not_match_challenge");
      }
      const worker = this.store.getWorker(challenge.workerId);
      const executor = worker?.capabilities.distributedExecutor;
      if (
        !worker
        || !executor
        || executor.nodeId !== challenge.nodeId
        || this.connections.get(worker.id) !== state
      ) {
        throw new Error("runtime_performance_target_changed");
      }
      const evidence = createCoordinatorRuntimePerformanceEvidence({
        challengeId: challenge.challengeId,
        nonce: challenge.nonce,
        workerId: challenge.workerId,
        sessionId: challenge.sessionId,
        nodeId: challenge.nodeId,
        issuedAt: challenge.issuedAt,
        expiresAt: challenge.expiresAt,
        observedAt: new Date(now).toISOString(),
        profile,
      });
      const capabilities = structuredClone(worker.capabilities);
      capabilities.distributedExecutor!.performanceEvidence = evidence;
      this.store.updateWorkerHeartbeat(worker.id, capabilities, worker.status);
      this.finishEvidenceChallenge(pending, true);
    } catch (error) {
      this.logger?.warn({
        workerId: state.workerId,
        challengeId: pending.challenge.challengeId,
        error: error instanceof Error ? error.message : String(error),
      }, "runtime performance evidence rejected");
      this.finishEvidenceChallenge(pending, false);
    }
  }

  private completeEngineRuntimeChallenge(
    state: ConnectionState,
    pending: PendingEngineRuntime,
    input: unknown,
    now: number,
  ): void {
    try {
      const measurement = engineRuntimeMeasurementSchema.parse(input);
      const challenge = pending.challenge;
      const worker = this.store.getWorker(challenge.workerId);
      const executor = worker?.capabilities.distributedExecutor;
      const performance = executor?.performanceEvidence;
      const physical = executor?.physicalIdentity;
      const build = worker?.capabilities.buildIdentity;
      const activationPlan = this.store.listEngineRuntimeActivationPlans().find(
        (candidate) => candidate.workerId === challenge.workerId
          && candidate.modelId === challenge.modelId
          && canonicalEvidenceJson(candidate.request) === canonicalEvidenceJson({
            probeKind: challenge.probeKind,
            descriptorDigest: challenge.descriptorDigest,
            certificationId: challenge.certificationId,
            artifactManifestDigest: challenge.artifactManifestDigest,
            modelId: challenge.modelId,
            modelRevision: challenge.modelRevision,
            backend: challenge.backend,
            runtimeAbi: challenge.runtimeAbi,
            quantization: challenge.quantization,
            contextTokens: challenge.contextTokens,
            expectedLayerStart: challenge.expectedLayerStart,
            expectedLayerEnd: challenge.expectedLayerEnd,
            expectedKvBytesPerToken: challenge.expectedKvBytesPerToken,
            expectedLayerWeightBytes: challenge.expectedLayerWeightBytes,
            referenceDecodeMsPerToken: challenge.referenceDecodeMsPerToken,
            referencePrefillMsPerToken: challenge.referencePrefillMsPerToken,
            hiddenSize: challenge.hiddenSize,
            attentionHeads: challenge.attentionHeads,
            kvHeads: challenge.kvHeads,
            headDim: challenge.headDim,
            requiredRoles: challenge.requiredRoles,
            minimumSamples: challenge.minimumSamples,
          }),
      );
      if (
        !worker
        || !executor
        || !performance
        || !physical
        || !build
        || !activationPlan
        || executor.nodeId !== challenge.nodeId
        || performance.sessionId !== challenge.sessionId
        || this.connections.get(worker.id) !== state
        || measurement.samples < challenge.minimumSamples
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
        || challenge.requiredRoles.some(
          (role) => !measurement.features.roles.includes(role),
        )
        || Date.parse(measurement.measuredAt) < Date.parse(challenge.issuedAt) - 5_000
        || Date.parse(measurement.measuredAt) > now + 5_000
      ) {
        throw new Error("engine_runtime_measurement_does_not_match_challenge");
      }
      const measuredAtMs = Date.parse(measurement.measuredAt);
      const profile = sealEngineRuntimeProfile({
        descriptorDigest: challenge.descriptorDigest,
        certificationId: challenge.certificationId,
        artifactManifestDigest: challenge.artifactManifestDigest,
        sourceId: build.sourceId,
        hardwareFingerprintSha256: physical.hostFingerprintSha256,
        workerId: challenge.workerId,
        sessionId: challenge.sessionId,
        nodeId: challenge.nodeId,
        modelId: challenge.modelId,
        modelRevision: challenge.modelRevision,
        backend: challenge.backend,
        runtimeAbi: challenge.runtimeAbi,
        quantization: challenge.quantization,
        measuredAt: measurement.measuredAt,
        expiresAt: new Date(
          measuredAtMs + ENGINE_RUNTIME_PROFILE_DEFAULT_MAXIMUM_AGE_MS,
        ).toISOString(),
        samples: measurement.samples,
        confidenceHalfWidthPct: measurement.confidenceHalfWidthPct,
        capacity: measurement.capacity,
        costs: measurement.costs,
        features: measurement.features,
        evidence: {
          deploymentCanaryEvidenceId: activationPlan.evidence.canaryEvidenceId,
          runtimePerformanceEvidenceId: performance.evidenceId,
        },
      });
      publishCoordinatorEngineRuntimeProfile(this.store, worker.id, profile);
      this.finishEvidenceChallenge(pending, true);
    } catch (error) {
      this.logger?.warn({
        workerId: state.workerId,
        challengeId: pending.challenge.challengeId,
        error: error instanceof Error ? error.message : String(error),
      }, "engine runtime evidence rejected");
      this.finishEvidenceChallenge(pending, false);
    }
  }

  private finishEvidenceChallenge(
    pending: PendingEvidenceChallenge,
    succeeded: boolean,
  ): void {
    clearTimeout(pending.timeout);
    this.evidenceChallenges.delete(pending.challenge.challengeId);
    const target = pending.kind === "deployment-canary"
      ? pending.challenge.deploymentId
      : pending.challenge.nodeId;
    const key = evidenceChallengeKey(pending.challenge.workerId, pending.kind, target);
    if (succeeded) this.evidenceRetryAfter.delete(key);
    else this.evidenceRetryAfter.set(key, Date.now() + EVIDENCE_RETRY_AFTER_MS);
  }

  private clearEvidenceChallengesForSession(workerId: string, sessionId: string): void {
    for (const pending of [...this.evidenceChallenges.values()]) {
      if (
        pending.challenge.workerId === workerId
        && pending.challenge.sessionId === sessionId
      ) {
        this.finishEvidenceChallenge(pending, false);
      }
    }
  }

  private sendSocket(socket: WebSocket, type: string, payload: unknown): boolean {
    if (socket.readyState !== socket.OPEN) return false;
    if (socket.bufferedAmount > MAX_WEBSOCKET_BUFFERED_BYTES) {
      socket.close(4429, "runtime stream backpressure exceeded");
      return false;
    }
    const envelope: ServerEnvelope = { v: 1, type, payload };
    socket.send(JSON.stringify(envelope));
    return true;
  }

  private async waitForActivationCheckpointBackpressure(
    state: ConnectionState,
    expiresAt: number,
  ): Promise<void> {
    while (state.socket.bufferedAmount > MAX_WEBSOCKET_BUFFERED_BYTES / 2) {
      if (
        Date.now() >= expiresAt
        || !state.ready
        || this.connections.get(state.workerId ?? "") !== state
        || state.socket.readyState !== state.socket.OPEN
      ) throw new Error("activation_checkpoint_restore_delivery_expired");
      await new Promise<void>((resolve) => setTimeout(resolve, 5));
    }
  }

  private startRuntimeLinkProbe(
    sourceWorkerId: string,
    sourceNodeId: string,
    destinationWorkerId: string,
    destinationNodeId: string,
    now: number,
  ): boolean {
    const probeId = `link-${randomUUID()}`;
    const timeout = setTimeout(() => {
      const pending = this.runtimeLinkProbes.get(probeId);
      if (pending) this.finishRuntimeLinkProbe(pending, null, null);
    }, RUNTIME_LINK_PROBE_TIMEOUT_MS);
    timeout.unref();
    const session: RuntimeLinkProbeSession = {
      probeId,
      sourceWorkerId,
      destinationWorkerId,
      sourceNodeId,
      destinationNodeId,
      timeout,
    };
    this.runtimeLinkProbes.set(probeId, session);
    this.runtimeLinkLastStartedAt.set(runtimeLinkKey(sourceNodeId, destinationNodeId), now);
    if (!this.send(sourceWorkerId, "runtime.link.probe.start", {
      probeId,
      destinationNodeId,
      timeoutMs: RUNTIME_LINK_PROBE_TIMEOUT_MS,
      payloadBytes: RUNTIME_LINK_PROBE_PAYLOAD_BYTES,
    })) {
      this.finishRuntimeLinkProbe(session, null, null);
      return false;
    }
    return true;
  }

  private handleRuntimeLinkProbeEnvelope(envelope: WorkerEnvelope): void {
    const payload = envelope.payload as Record<string, unknown>;
    const probeId = payload.probeId as string;
    const probe = this.runtimeLinkProbes.get(probeId);
    if (!probe) return;

    if (envelope.type === "runtime.link.probe.ping") {
      if (
        envelope.workerId !== probe.sourceWorkerId
        || payload.destinationNodeId !== probe.destinationNodeId
      ) {
        this.finishRuntimeLinkProbe(probe, null, null);
        return;
      }
      if (!this.send(probe.destinationWorkerId, "runtime.link.probe.ping", {
        probeId,
        data: payload.data,
      })) {
        this.finishRuntimeLinkProbe(probe, null, null);
      }
      return;
    }

    if (envelope.type === "runtime.link.probe.pong") {
      if (envelope.workerId !== probe.destinationWorkerId) {
        this.finishRuntimeLinkProbe(probe, null, null);
        return;
      }
      if (!this.send(probe.sourceWorkerId, "runtime.link.probe.pong", {
        probeId,
        data: payload.data,
      })) {
        this.finishRuntimeLinkProbe(probe, null, null);
      }
      return;
    }

    if (envelope.type === "runtime.link.probe.result") {
      if (
        envelope.workerId !== probe.sourceWorkerId
        || payload.destinationNodeId !== probe.destinationNodeId
      ) {
        this.finishRuntimeLinkProbe(probe, null, null);
        return;
      }
      this.finishRuntimeLinkProbe(
        probe,
        payload.rttMs as number | null,
        payload.goodputMbps as number | null,
      );
    }
  }

  private finishRuntimeLinkProbe(
    probe: RuntimeLinkProbeSession,
    rttMs: number | null,
    goodputMbps: number | null,
  ): void {
    if (!this.runtimeLinkProbes.delete(probe.probeId)) return;
    clearTimeout(probe.timeout);
    const measuredAt = Date.now();
    if (rttMs === null || goodputMbps === null) {
      this.runtimeLinkObservationsStore.recordFailure(
        probe.sourceNodeId,
        probe.destinationNodeId,
        measuredAt,
        "relay",
      );
    } else {
      this.runtimeLinkObservationsStore.recordSuccess(
        probe.sourceNodeId,
        probe.destinationNodeId,
        rttMs,
        goodputMbps,
        measuredAt,
        "relay",
      );
    }
    this.store.saveRuntimeLinkSample?.({
      fromNodeId: probe.sourceNodeId,
      toNodeId: probe.destinationNodeId,
      measuredAt,
      rttMs,
      goodputMbps,
      transportMode: "relay",
    });
  }

  private handleRuntimeStreamEnvelope(envelope: WorkerEnvelope): void {
    const payload = envelope.payload as Record<string, unknown>;
    const streamId = payload.streamId as string;
    if (envelope.type === "runtime.stream.open") {
      if (this.runtimeStreams.has(streamId) || this.runtimeStreams.size >= MAX_RUNTIME_STREAMS) {
        this.send(envelope.workerId, "runtime.stream.error", {
          streamId,
          message: this.runtimeStreams.has(streamId)
            ? "runtime_stream_is_duplicate"
            : "runtime_stream_capacity_exceeded",
        });
        return;
      }
      const destinationNodeId = payload.destinationNodeId as string;
      const source = this.store.getWorker(envelope.workerId);
      const destination = this.store.listWorkers().find((worker) =>
        this.isConnected(worker.id) &&
        worker.capabilities.distributedExecutor?.protocol === "gdlp-worker-tunnel/2" &&
        worker.capabilities.distributedExecutor?.nodeId === destinationNodeId
      );
      if (!destination || destination.id === envelope.workerId) {
        this.send(envelope.workerId, "runtime.stream.error", {
          streamId,
          message: `runtime_stream_destination_unavailable:${destinationNodeId}`,
        });
        return;
      }
      const requestedRecovery =
        typeof payload.generation === "number"
        && typeof payload.recoveryToken === "string";
      if (
        requestedRecovery
        && source?.capabilities.distributedExecutor?.streamRecovery !== "offset-ack-v1"
      ) {
        this.send(envelope.workerId, "runtime.stream.error", {
          streamId,
          generation: payload.generation,
          recoveryToken: payload.recoveryToken,
          message: "runtime_stream_recovery_was_not_advertised",
        });
        return;
      }
      // A mixed-version route can safely downgrade before either endpoint has
      // moved a byte. It remains a legacy fail-closed stream and is never
      // labelled recoverable.
      const negotiatedRecovery =
        requestedRecovery
        && destination.capabilities.distributedExecutor?.streamRecovery === "offset-ack-v1";
      const sourceExecutor = source?.capabilities.distributedExecutor;
      const destinationExecutor = destination.capabilities.distributedExecutor;
      const targetPort = payload.targetPort as number;
      const directSupported =
        sourceExecutor?.directTransport?.protocol === "mycellios-direct/1"
        && sourceExecutor.directTransport.commitAck === "destination-v1"
        && destinationExecutor?.directTransport?.protocol === "mycellios-direct/1"
        && destinationExecutor.directTransport.commitAck === "destination-v1"
        && destinationExecutor.directTransport.candidates.length > 0;
      const session: RuntimeStreamSession = {
        streamId,
        sourceWorkerId: envelope.workerId,
        destinationWorkerId: destination.id,
        sourceNodeId: sourceExecutor?.nodeId ?? null,
        destinationNodeId,
        targetPort,
        sourceSequence: 0,
        destinationSequence: 0,
        opened: false,
        transportMode: directSupported ? "negotiating" : "relay",
        direct: null,
        bytesSourceToDestination: 0,
        bytesDestinationToSource: 0,
        createdAt: Date.now(),
        connectedAt: null,
        endedAt: null,
        recovery: negotiatedRecovery
          ? {
              recoveryToken: payload.recoveryToken as string,
              generation: payload.generation as number,
              sourceForwardOffset: 0,
              destinationForwardOffset: 0,
              suspended: false,
              reports: new Map(),
              timeout: null,
            }
          : null,
      };
      this.runtimeStreams.set(streamId, session);
      if (directSupported && session.sourceNodeId) {
        const grant = createDirectSessionGrant({
          connectionId: streamId,
          sourceNodeId: session.sourceNodeId,
          destinationNodeId,
          targetPort,
          expiresAt: Date.now() + DIRECT_GRANT_TTL_MS,
        });
        const timeout = setTimeout(() => {
          this.fallbackRuntimeStreamToRelay(session, "direct_negotiation_timeout");
        }, DIRECT_NEGOTIATION_TIMEOUT_MS);
        timeout.unref();
        session.direct = {
          grant,
          state: "offered",
          timeout,
          connectRttMs: null,
        };
        if (!this.send(destination.id, "runtime.direct.offer", {
          streamId,
          grant,
        })) {
          this.fallbackRuntimeStreamToRelay(session, "direct_destination_disconnected");
        }
      } else {
        this.startRuntimeRelay(session);
      }
      return;
    }

    const session = this.runtimeStreams.get(streamId);
    if (!session) {
      if (
        envelope.type === "runtime.stream.resume"
        && typeof payload.generation === "number"
        && typeof payload.recoveryToken === "string"
      ) {
        this.send(envelope.workerId, "runtime.stream.error", {
          streamId,
          generation: payload.generation,
          recoveryToken: payload.recoveryToken,
          message: "runtime_stream_recovery_session_expired",
        });
      }
      return;
    }
    const fromSource = session.sourceWorkerId === envelope.workerId;
    const fromDestination = session.destinationWorkerId === envelope.workerId;
    if (!fromSource && !fromDestination) return;

    if (envelope.type === "runtime.stream.opened") {
      if (!fromDestination || session.opened) return;
      if (!this.runtimeStreamIdentityMatches(session, payload)) {
        this.terminateRuntimeStream(session, "runtime_stream_opened_identity_mismatch");
        return;
      }
      session.opened = true;
      session.connectedAt ??= Date.now();
      if (session.sourceWorkerId) {
        this.send(session.sourceWorkerId, "runtime.stream.opened", {
          streamId,
          ...(session.recovery
            ? {
                generation: session.recovery.generation,
                recoveryToken: session.recovery.recoveryToken,
              }
            : {}),
        });
      } else {
        session.localSocket?.resume();
      }
      return;
    }

    if (envelope.type === "runtime.stream.data") {
      if (!session.opened) {
        this.terminateRuntimeStream(session, "runtime_stream_data_before_open");
        return;
      }
      if (session.recovery) {
        this.handleRecoverableRuntimeData(session, envelope.workerId, payload);
        return;
      }
      if (!this.runtimeStreamIdentityMatches(session, payload)) {
        this.terminateRuntimeStream(session, "runtime_stream_unnegotiated_recovery_data");
        return;
      }
      const sequence = payload.sequence as number;
      const expected = fromSource ? session.sourceSequence : session.destinationSequence;
      if (sequence !== expected) {
        this.terminateRuntimeStream(session, `runtime_stream_sequence_mismatch:${expected}:${sequence}`);
        return;
      }
      if (fromSource) session.sourceSequence += 1;
      else session.destinationSequence += 1;
      const byteLength = Buffer.from(payload.data as string, "base64").byteLength;
      if (fromSource) session.bytesSourceToDestination += byteLength;
      else session.bytesDestinationToSource += byteLength;
      if (fromSource) {
        if (!this.send(session.destinationWorkerId, "runtime.stream.data", payload)) {
          this.terminateRuntimeStream(session, "runtime_stream_destination_backpressure");
        }
      } else if (session.sourceWorkerId) {
        if (!this.send(session.sourceWorkerId, "runtime.stream.data", payload)) {
          this.terminateRuntimeStream(session, "runtime_stream_source_backpressure");
        }
      } else {
        const socket = session.localSocket;
        if (!socket || socket.destroyed || socket.writableLength > MAX_RUNTIME_STREAM_BUFFERED_BYTES) {
          this.terminateRuntimeStream(session, "runtime_stream_local_buffer_exceeded");
          return;
        }
        socket.write(Buffer.from(payload.data as string, "base64"));
      }
      return;
    }

    if (envelope.type === "runtime.stream.ack") {
      this.handleRuntimeStreamAcknowledgement(session, envelope.workerId, payload);
      return;
    }

    if (envelope.type === "runtime.stream.resume") {
      this.handleRuntimeStreamResume(session, envelope.workerId, payload);
      return;
    }

    if (envelope.type === "runtime.stream.end") {
      if (!this.runtimeStreamIdentityMatches(session, payload)) {
        this.terminateRuntimeStream(session, "runtime_stream_end_identity_mismatch");
        return;
      }
      if (session.recovery) {
        this.forwardRuntimeTerminal(session, envelope.workerId, "runtime.stream.end", payload);
      } else {
        this.forwardRuntimeTerminal(session, envelope.workerId, "runtime.stream.end", { streamId });
      }
      this.terminateRuntimeStream(session);
      return;
    }
    if (envelope.type === "runtime.stream.error") {
      if (!this.runtimeStreamIdentityMatches(session, payload)) {
        this.terminateRuntimeStream(session, "runtime_stream_error_identity_mismatch");
        return;
      }
      this.forwardRuntimeTerminal(session, envelope.workerId, "runtime.stream.error", payload);
      this.terminateRuntimeStream(session);
    }
  }

  private handleRuntimeDirectEnvelope(envelope: WorkerEnvelope): void {
    const payload = envelope.payload as Record<string, unknown>;
    const streamId = payload.streamId as string;
    const connectionId = payload.connectionId as string;
    const session = this.runtimeStreams.get(streamId);
    const direct = session?.direct;
    if (!session || !direct || direct.grant.connectionId !== connectionId) return;
    const fromSource = session.sourceWorkerId === envelope.workerId;
    const fromDestination = session.destinationWorkerId === envelope.workerId;
    if (!fromSource && !fromDestination) return;

    if (envelope.type === "runtime.direct.ready") {
      if (!fromDestination || direct.state !== "offered" || !session.sourceWorkerId) {
        this.terminateRuntimeStream(session, "direct_ready_identity_is_invalid");
        return;
      }
      direct.state = "ready";
      const candidates =
        this.store.getWorker(session.destinationWorkerId)
          ?.capabilities.distributedExecutor?.directTransport?.candidates ?? [];
      if (candidates.length === 0 || !this.send(session.sourceWorkerId, "runtime.direct.connect", {
        streamId,
        destinationNodeId: session.destinationNodeId,
        grant: direct.grant,
        candidates,
        timeoutMs: Math.min(2_000, DIRECT_NEGOTIATION_TIMEOUT_MS),
      })) {
        this.fallbackRuntimeStreamToRelay(session, "direct_source_unavailable");
      }
      return;
    }

    if (envelope.type === "runtime.direct.fallback") {
      if (direct.state === "committed" || session.opened || this.runtimeStreamMovedBytes(session)) {
        this.terminateRuntimeStream(session, "direct_downgrade_after_commit_is_forbidden");
        return;
      }
      this.fallbackRuntimeStreamToRelay(
        session,
        typeof payload.reason === "string" ? payload.reason : "direct_candidate_unreachable",
      );
      return;
    }

    if (envelope.type === "runtime.direct.established") {
      if (
        !fromSource
        || direct.state !== "ready"
        || typeof payload.connectRttMs !== "number"
        || !Number.isFinite(payload.connectRttMs)
        || payload.connectRttMs <= 0
      ) {
        this.terminateRuntimeStream(session, "direct_established_identity_is_invalid");
        return;
      }
      direct.state = "established";
      direct.connectRttMs = payload.connectRttMs;
      clearTimeout(direct.timeout);
      // WebSocket delivery across two peers is not ordered. Release only the
      // destination now and wait for its explicit acknowledgement before the
      // source can emit the first application byte.
      const destinationCommitSent = this.send(
        session.destinationWorkerId,
        "runtime.direct.commit",
        { streamId, connectionId },
      );
      if (!destinationCommitSent) {
        this.terminateRuntimeStream(session, "direct_commit_delivery_failed");
        return;
      }
      direct.state = "destination-committing";
      direct.timeout = setTimeout(() => {
        this.terminateRuntimeStream(session, "direct_destination_commit_ack_timeout");
      }, DIRECT_NEGOTIATION_TIMEOUT_MS);
      direct.timeout.unref();
      return;
    }

    if (envelope.type === "runtime.direct.committed") {
      if (!fromDestination || direct.state !== "destination-committing") {
        this.terminateRuntimeStream(session, "direct_commit_ack_identity_is_invalid");
        return;
      }
      clearTimeout(direct.timeout);
      const sourceCommitted = session.sourceWorkerId
        ? this.send(session.sourceWorkerId, "runtime.direct.commit", {
            streamId,
            connectionId,
          })
        : false;
      if (!sourceCommitted) {
        this.terminateRuntimeStream(session, "direct_commit_delivery_failed");
        return;
      }
      direct.state = "committed";
      session.transportMode = "direct";
      session.opened = true;
      session.connectedAt = Date.now();
      direct.timeout = setTimeout(() => {
        this.terminateRuntimeStream(session, "direct_route_lifetime_exceeded");
      }, DIRECT_ROUTE_MAX_LIFETIME_MS);
      direct.timeout.unref();
      return;
    }

    if (envelope.type === "runtime.direct.telemetry") {
      if (direct.state !== "committed") {
        this.terminateRuntimeStream(session, "direct_telemetry_before_commit");
        return;
      }
      const bytesTx = payload.bytesTx as number;
      const bytesRx = payload.bytesRx as number;
      if (fromSource) {
        session.bytesSourceToDestination = Math.max(session.bytesSourceToDestination, bytesTx);
        session.bytesDestinationToSource = Math.max(session.bytesDestinationToSource, bytesRx);
      } else {
        session.bytesDestinationToSource = Math.max(session.bytesDestinationToSource, bytesTx);
        session.bytesSourceToDestination = Math.max(session.bytesSourceToDestination, bytesRx);
      }
      return;
    }

    if (envelope.type === "runtime.direct.closed") {
      if (direct.state !== "committed") {
        this.terminateRuntimeStream(session, "direct_close_before_commit");
        return;
      }
      const bytesTx = payload.bytesTx as number;
      const bytesRx = payload.bytesRx as number;
      if (fromSource) {
        session.bytesSourceToDestination = Math.max(session.bytesSourceToDestination, bytesTx);
        session.bytesDestinationToSource = Math.max(session.bytesDestinationToSource, bytesRx);
      } else {
        session.bytesDestinationToSource = Math.max(session.bytesDestinationToSource, bytesTx);
        session.bytesSourceToDestination = Math.max(session.bytesSourceToDestination, bytesRx);
      }
      if (typeof payload.reason === "string") {
        this.recordRuntimeLinkFailure(session, "direct", payload.reason);
      }
      this.terminateRuntimeStream(session);
    }
  }

  private startRuntimeRelay(session: RuntimeStreamSession): void {
    if (!this.runtimeStreams.has(session.streamId)) return;
    session.transportMode = "relay";
    session.direct = null;
    if (!this.send(session.destinationWorkerId, "runtime.stream.open", {
      streamId: session.streamId,
      targetPort: session.targetPort,
      ...(session.recovery
        ? {
            generation: session.recovery.generation,
            recoveryToken: session.recovery.recoveryToken,
          }
        : {}),
    })) {
      this.terminateRuntimeStream(session, "runtime_stream_destination_disconnected");
    }
  }

  private fallbackRuntimeStreamToRelay(
    session: RuntimeStreamSession,
    _reason: string,
  ): void {
    const direct = session.direct;
    if (!direct || direct.state === "committed" || this.runtimeStreamMovedBytes(session)) {
      this.terminateRuntimeStream(session, "direct_downgrade_after_bytes_is_forbidden");
      return;
    }
    clearTimeout(direct.timeout);
    if (session.sourceNodeId) {
      const measuredAt = Date.now();
      this.runtimeLinkObservationsStore.recordFailure(
        session.sourceNodeId,
        session.destinationNodeId,
        measuredAt,
        "direct",
      );
      this.store.saveRuntimeLinkSample?.({
        fromNodeId: session.sourceNodeId,
        toNodeId: session.destinationNodeId,
        measuredAt,
        rttMs: null,
        goodputMbps: null,
        transportMode: "direct",
      });
    }
    const cancel = {
      streamId: session.streamId,
      connectionId: direct.grant.connectionId,
    };
    if (session.sourceWorkerId) this.send(session.sourceWorkerId, "runtime.direct.cancel", cancel);
    this.send(session.destinationWorkerId, "runtime.direct.cancel", cancel);
    this.startRuntimeRelay(session);
  }

  private runtimeStreamMovedBytes(session: RuntimeStreamSession): boolean {
    return (
      session.bytesSourceToDestination > 0
      || session.bytesDestinationToSource > 0
    );
  }

  private handleRecoverableRuntimeData(
    session: RuntimeStreamSession,
    originWorkerId: string,
    payload: Record<string, unknown>,
  ): void {
    const recovery = session.recovery;
    if (!recovery || !this.runtimeStreamIdentityMatches(session, payload)) {
      this.terminateRuntimeStream(session, "runtime_stream_data_identity_mismatch");
      return;
    }
    const fromSource = session.sourceWorkerId === originWorkerId;
    const offset = payload.offset as number;
    const byteLength = Buffer.from(payload.data as string, "base64").byteLength;
    const expected = fromSource
      ? recovery.sourceForwardOffset
      : recovery.destinationForwardOffset;
    if (offset < expected && offset + byteLength <= expected) return;
    if (offset !== expected) {
      this.terminateRuntimeStream(
        session,
        `runtime_stream_offset_mismatch:${expected}:${offset}`,
      );
      return;
    }
    if (fromSource) session.bytesSourceToDestination += byteLength;
    else session.bytesDestinationToSource += byteLength;
    if (fromSource) recovery.sourceForwardOffset += byteLength;
    else recovery.destinationForwardOffset += byteLength;
    if (recovery.suspended) return;
    const targetWorkerId = fromSource
      ? session.destinationWorkerId
      : session.sourceWorkerId;
    if (!targetWorkerId || !this.send(targetWorkerId, "runtime.stream.data", payload)) {
      this.suspendRuntimeStream(session, "runtime_stream_relay_delivery_failed");
    }
  }

  private handleRuntimeStreamAcknowledgement(
    session: RuntimeStreamSession,
    originWorkerId: string,
    payload: Record<string, unknown>,
  ): void {
    const recovery = session.recovery;
    if (!recovery || !this.runtimeStreamIdentityMatches(session, payload)) {
      this.terminateRuntimeStream(session, "runtime_stream_ack_identity_mismatch");
      return;
    }
    const fromSource = session.sourceWorkerId === originWorkerId;
    const acknowledgedOffset = payload.acknowledgedOffset as number;
    const maximum = fromSource
      ? recovery.destinationForwardOffset
      : recovery.sourceForwardOffset;
    if (acknowledgedOffset > maximum) {
      this.terminateRuntimeStream(
        session,
        `runtime_stream_ack_exceeds_forwarded_offset:${maximum}:${acknowledgedOffset}`,
      );
      return;
    }
    if (recovery.suspended) return;
    const targetWorkerId = fromSource
      ? session.destinationWorkerId
      : session.sourceWorkerId;
    if (!targetWorkerId || !this.send(targetWorkerId, "runtime.stream.ack", payload)) {
      this.suspendRuntimeStream(session, "runtime_stream_relay_ack_delivery_failed");
    }
  }

  private handleRuntimeStreamResume(
    session: RuntimeStreamSession,
    originWorkerId: string,
    payload: Record<string, unknown>,
  ): void {
    const recovery = session.recovery;
    if (
      !recovery
      || !recovery.suspended
      || !this.runtimeStreamIdentityMatches(session, payload)
    ) {
      this.terminateRuntimeStream(session, "runtime_stream_resume_identity_mismatch");
      return;
    }
    const report: RuntimeStreamRecoveryReport = {
      sendOffset: payload.sendOffset as number,
      acknowledgedOffset: payload.acknowledgedOffset as number,
      receiveOffset: payload.receiveOffset as number,
      bufferedFromOffset: payload.bufferedFromOffset as number,
    };
    if (
      report.acknowledgedOffset > report.sendOffset
      || report.bufferedFromOffset > report.acknowledgedOffset
    ) {
      this.terminateRuntimeStream(session, "runtime_stream_resume_report_is_invalid");
      return;
    }
    const existing = recovery.reports.get(originWorkerId);
    if (existing && !runtimeStreamReportEquals(existing, report)) {
      this.terminateRuntimeStream(session, "runtime_stream_resume_report_changed");
      return;
    }
    recovery.reports.set(originWorkerId, report);
    this.tryResumeRuntimeStream(session);
  }

  private tryResumeRuntimeStream(session: RuntimeStreamSession): void {
    const recovery = session.recovery;
    const sourceWorkerId = session.sourceWorkerId;
    if (!recovery || !sourceWorkerId) return;
    const source = recovery.reports.get(sourceWorkerId);
    const destination = recovery.reports.get(session.destinationWorkerId);
    if (!source || !destination) return;
    if (
      source.bufferedFromOffset > destination.receiveOffset
      || destination.receiveOffset > source.sendOffset
      || destination.bufferedFromOffset > source.receiveOffset
      || source.receiveOffset > destination.sendOffset
      || source.acknowledgedOffset > destination.receiveOffset
      || destination.acknowledgedOffset > source.receiveOffset
    ) {
      this.terminateRuntimeStream(session, "runtime_stream_resume_offsets_are_inconsistent");
      return;
    }
    if (
      !this.isConnected(sourceWorkerId)
      || !this.isConnected(session.destinationWorkerId)
    ) {
      return;
    }
    const previousGeneration = recovery.generation;
    const generation = previousGeneration + 1;
    if (generation > 1_000_000) {
      this.terminateRuntimeStream(session, "runtime_stream_generation_exhausted");
      return;
    }
    recovery.generation = generation;
    recovery.sourceForwardOffset = destination.receiveOffset;
    recovery.destinationForwardOffset = source.receiveOffset;
    recovery.suspended = false;
    recovery.reports.clear();
    if (recovery.timeout) clearTimeout(recovery.timeout);
    recovery.timeout = null;
    const sourceAccepted = this.send(sourceWorkerId, "runtime.stream.resumed", {
      streamId: session.streamId,
      recoveryToken: recovery.recoveryToken,
      previousGeneration,
      generation,
      sendFromOffset: destination.receiveOffset,
    });
    const destinationAccepted = this.send(
      session.destinationWorkerId,
      "runtime.stream.resumed",
      {
        streamId: session.streamId,
        recoveryToken: recovery.recoveryToken,
        previousGeneration,
        generation,
        sendFromOffset: source.receiveOffset,
      },
    );
    if (!sourceAccepted || !destinationAccepted) {
      this.terminateRuntimeStream(session, "runtime_stream_resume_delivery_failed");
    }
  }

  private suspendRuntimeStream(
    session: RuntimeStreamSession,
    reason = "runtime_stream_relay_unavailable",
  ): void {
    const recovery = session.recovery;
    if (!recovery || recovery.suspended) return;
    this.recordRuntimeLinkFailure(session, "relay", reason);
    recovery.suspended = true;
    recovery.reports.clear();
    const deadlineAt = Date.now() + RUNTIME_STREAM_RECOVERY_GRACE_MS;
    const timeout = setTimeout(() => {
      if (session.recovery?.timeout !== timeout) return;
      this.terminateRuntimeStream(session, "runtime_stream_recovery_timeout");
    }, RUNTIME_STREAM_RECOVERY_GRACE_MS);
    timeout.unref();
    recovery.timeout = timeout;
    const suspension = {
      streamId: session.streamId,
      recoveryToken: recovery.recoveryToken,
      generation: recovery.generation,
      deadlineAt,
    };
    if (session.sourceWorkerId && this.isConnected(session.sourceWorkerId)) {
      this.send(session.sourceWorkerId, "runtime.stream.suspend", suspension);
    }
    if (this.isConnected(session.destinationWorkerId)) {
      this.send(session.destinationWorkerId, "runtime.stream.suspend", suspension);
    }
  }

  private recordRuntimeLinkFailure(
    session: RuntimeStreamSession,
    role: "direct" | "relay",
    reason: string,
  ): void {
    const recovery = session.recovery;
    const evidence = runtimeLinkFailureEvidenceSchema.parse({
      schema: "mycellios-runtime-link-failure/1",
      streamId: session.streamId,
      sourceNodeId: session.sourceNodeId,
      destinationNodeId: session.destinationNodeId,
      generation: recovery?.generation ?? 0,
      role,
      failureClass: role === "direct" ? "direct-link-lost" : "relay-link-lost",
      transportMode: role,
      checkpointKind: role === "relay" && recovery ? "stream-offset" : "none",
      sourceOffset: role === "relay" && recovery
        ? recovery.sourceForwardOffset
        : session.bytesSourceToDestination,
      destinationOffset: role === "relay" && recovery
        ? recovery.destinationForwardOffset
        : session.bytesDestinationToSource,
      observedAt: Date.now(),
      reason: reason.slice(0, 256),
    });
    this.runtimeLinkFailures.unshift(evidence);
    if (this.runtimeLinkFailures.length > 512) this.runtimeLinkFailures.length = 512;
  }

  private runtimeStreamIdentityMatches(
    session: RuntimeStreamSession,
    payload: Record<string, unknown>,
  ): boolean {
    if (!session.recovery) {
      return payload.generation === undefined && payload.recoveryToken === undefined;
    }
    return (
      payload.generation === session.recovery.generation
      && payload.recoveryToken === session.recovery.recoveryToken
    );
  }

  private attachLocalRuntimeStream(socket: Socket, destinationWorkerId: string, targetPort: number): void {
    if (this.runtimeStreams.size >= MAX_RUNTIME_STREAMS) {
      socket.destroy(new Error("runtime_stream_capacity_exceeded"));
      return;
    }
    const streamId = `coordinator-${randomUUID()}`;
    const session: RuntimeStreamSession = {
      streamId,
      sourceWorkerId: null,
      destinationWorkerId,
      sourceNodeId: null,
      destinationNodeId: destinationWorkerId,
      targetPort,
      sourceSequence: 0,
      destinationSequence: 0,
      opened: false,
      transportMode: "relay",
      direct: null,
      bytesSourceToDestination: 0,
      bytesDestinationToSource: 0,
      createdAt: Date.now(),
      connectedAt: null,
      endedAt: null,
      localSocket: socket,
      recovery: null,
    };
    this.runtimeStreams.set(streamId, session);
    socket.setNoDelay(true);
    socket.pause();
    socket.on("data", (chunk: Buffer) => {
      if (!session.opened) return;
      for (let offset = 0; offset < chunk.byteLength; offset += MAX_RUNTIME_STREAM_CHUNK_BYTES) {
        const piece = chunk.subarray(
          offset,
          Math.min(chunk.byteLength, offset + MAX_RUNTIME_STREAM_CHUNK_BYTES),
        );
        session.bytesSourceToDestination += piece.byteLength;
        if (!this.send(destinationWorkerId, "runtime.stream.data", {
          streamId,
          sequence: session.sourceSequence++,
          data: piece.toString("base64"),
        })) {
          this.terminateRuntimeStream(session, "runtime_stream_destination_backpressure");
          return;
        }
      }
    });
    socket.once("end", () => {
      this.send(destinationWorkerId, "runtime.stream.end", { streamId });
      this.terminateRuntimeStream(session);
    });
    socket.once("error", (error) => {
      this.send(destinationWorkerId, "runtime.stream.error", {
        streamId,
        message: error.message.slice(0, 1_024),
      });
      this.terminateRuntimeStream(session);
    });
    socket.once("close", () => this.terminateRuntimeStream(session));
    if (!this.send(destinationWorkerId, "runtime.stream.open", { streamId, targetPort })) {
      this.terminateRuntimeStream(session, "runtime_stream_destination_disconnected");
    }
  }

  private forwardRuntimeTerminal(
    session: RuntimeStreamSession,
    originWorkerId: string,
    type: "runtime.stream.end" | "runtime.stream.error",
    payload: unknown,
  ): void {
    if (originWorkerId === session.sourceWorkerId) {
      this.send(session.destinationWorkerId, type, payload);
    } else if (session.sourceWorkerId) {
      this.send(session.sourceWorkerId, type, payload);
    }
  }

  private terminateRuntimeStream(session: RuntimeStreamSession, message?: string): void {
    if (!this.runtimeStreams.delete(session.streamId)) return;
    if (session.recovery?.timeout) clearTimeout(session.recovery.timeout);
    if (session.direct) {
      clearTimeout(session.direct.timeout);
      const cancel = {
        streamId: session.streamId,
        connectionId: session.direct.grant.connectionId,
      };
      if (session.sourceWorkerId) this.send(session.sourceWorkerId, "runtime.direct.cancel", cancel);
      this.send(session.destinationWorkerId, "runtime.direct.cancel", cancel);
    }
    session.endedAt = Date.now();
    this.recordCompletedDirectTransport(session);
    this.completedRuntimeTransports.unshift(this.runtimeTransportSnapshotForSession(session));
    if (this.completedRuntimeTransports.length > 512) {
      this.completedRuntimeTransports.length = 512;
    }
    if (message && session.transportMode === "relay") {
      const recoveryIdentity = session.recovery
        ? {
            generation: session.recovery.generation,
            recoveryToken: session.recovery.recoveryToken,
          }
        : {};
      if (session.sourceWorkerId) {
        this.send(session.sourceWorkerId, "runtime.stream.error", {
          streamId: session.streamId,
          ...recoveryIdentity,
          message: message.slice(0, 1_024),
        });
      }
      this.send(session.destinationWorkerId, "runtime.stream.error", {
        streamId: session.streamId,
        ...recoveryIdentity,
        message: message.slice(0, 1_024),
      });
    }
    session.localSocket?.destroy();
  }

  private recordCompletedDirectTransport(session: RuntimeStreamSession): void {
    if (
      session.transportMode !== "direct"
      || !session.sourceNodeId
      || !session.direct?.connectRttMs
      || session.connectedAt === null
      || session.endedAt === null
    ) return;
    const durationMs = Math.max(1, session.endedAt - session.connectedAt);
    for (const [fromNodeId, toNodeId, bytes] of [
      [session.sourceNodeId, session.destinationNodeId, session.bytesSourceToDestination],
      [session.destinationNodeId, session.sourceNodeId, session.bytesDestinationToSource],
    ] as const) {
      if (bytes <= 0) continue;
      const goodputMbps = bytes * 8 / durationMs / 1_000;
      this.runtimeLinkObservationsStore.recordSuccess(
        fromNodeId,
        toNodeId,
        session.direct.connectRttMs,
        goodputMbps,
        session.endedAt,
        "direct",
      );
      this.store.saveRuntimeLinkSample?.({
        fromNodeId,
        toNodeId,
        measuredAt: session.endedAt,
        rttMs: session.direct.connectRttMs,
        goodputMbps,
        transportMode: "direct",
      });
    }
  }

  private runtimeTransportSnapshotForSession(
    session: RuntimeStreamSession,
  ): RuntimeTransportSnapshot {
    return {
      streamId: session.streamId,
      sourceNodeId: session.sourceNodeId,
      destinationNodeId: session.destinationNodeId,
      targetPort: session.targetPort,
      mode: session.transportMode === "direct" ? "direct" : "relay",
      state: session.endedAt !== null
        ? "closed"
        : session.transportMode === "negotiating"
          ? "negotiating"
          : session.recovery?.suspended
            ? "suspended"
            : "active",
      bytesSourceToDestination: session.bytesSourceToDestination,
      bytesDestinationToSource: session.bytesDestinationToSource,
      createdAt: session.createdAt,
      connectedAt: session.connectedAt,
      endedAt: session.endedAt,
      connectRttMs: session.direct?.connectRttMs ?? null,
    };
  }
}

function messageType(input: unknown): string | null {
  if (!input || typeof input !== "object" || Array.isArray(input)) return null;
  const type = (input as Record<string, unknown>).type;
  return typeof type === "string" ? type : null;
}

function scaleMatches(observed: number, rawExpected: number): boolean {
  const expected = Math.max(0.01, Math.min(100, rawExpected));
  return Number.isFinite(observed)
    && Math.abs(observed - expected) <= Math.max(1e-9, expected * 1e-6);
}

function runtimeLinkKey(fromNodeId: string, toNodeId: string): string {
  return `${fromNodeId}\u0000${toNodeId}`;
}

function evidenceChallengeKey(
  workerId: string,
  kind: PendingEvidenceChallenge["kind"],
  target: string,
): string {
  return `${workerId}\u0000${kind}\u0000${target}`;
}

function runtimeChallengeIdentity(
  capabilities: WorkerCapabilities,
): {
  nodeId: string;
  backend: RuntimePerformanceChallenge["backend"];
  deviceName: string;
  precision: RuntimePerformanceChallenge["precision"];
} | null {
  const executor = capabilities.distributedExecutor;
  if (!executor) return null;
  const acceleration = executor.acceleration;
  if (
    acceleration?.state === "gpu-ready"
    && acceleration.backend
    && acceleration.backend !== "cpu"
    && acceleration.deviceName
    && executor.computeMode !== "cpu-only"
  ) {
    return {
      nodeId: executor.nodeId,
      backend: acceleration.backend,
      deviceName: acceleration.deviceName,
      precision: "float16",
    };
  }
  if (
    executor.cpuEligible === true
    && executor.computeMode !== "gpu-only"
    && acceleration?.backend === "cpu"
    && acceleration.deviceName
  ) {
    return {
      nodeId: executor.nodeId,
      backend: "cpu",
      deviceName: acceleration.deviceName,
      precision: "float32",
    };
  }
  return null;
}

function normalizeDeviceName(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

function runtimeStreamReportEquals(
  left: RuntimeStreamRecoveryReport,
  right: RuntimeStreamRecoveryReport,
): boolean {
  return (
    left.sendOffset === right.sendOffset
    && left.acknowledgedOffset === right.acknowledgedOffset
    && left.receiveOffset === right.receiveOffset
    && left.bufferedFromOffset === right.bufferedFromOffset
  );
}
