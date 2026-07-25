import { EventEmitter } from "node:events";
import { randomUUID } from "node:crypto";
import { createServer, type AddressInfo, type Server, type Socket } from "node:net";
import type { FastifyInstance } from "fastify";
import type WebSocket from "ws";
import type { ServerEnvelope, WorkerEnvelope } from "../contracts/types.js";
import {
  MAX_RUNTIME_STREAM_CHUNK_BYTES,
  parseWorkerEnvelope,
  workerEnvelopeValidationIssues,
  type WorkerHeartbeatPayload,
} from "../contracts/worker-protocol.js";
import type { MeshStore } from "../storage/store.js";
import {
  RuntimeLinkObservationStore,
  type RuntimeLinkObservation,
} from "./runtime-link-observations.js";
import {
  createDirectSessionGrant,
  type DirectSessionGrant,
} from "../transport/direct-secure-channel.js";

interface HubEvents {
  envelope: [WorkerEnvelope];
  disconnect: [string];
}

interface ConnectionState {
  socket: WebSocket;
  workerId: string | null;
  ready: boolean;
  helloTimer: NodeJS.Timeout;
  pending: boolean;
  messageWindowStartedAt: number;
  messagesInWindow: number;
}

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
    state: "offered" | "ready" | "established" | "committed";
    timeout: NodeJS.Timeout;
    connectRttMs: number | null;
  } | null;
  bytesSourceToDestination: number;
  bytesDestinationToSource: number;
  createdAt: number;
  connectedAt: number | null;
  endedAt: number | null;
}

export interface RuntimeTransportSnapshot {
  streamId: string;
  sourceNodeId: string | null;
  destinationNodeId: string;
  targetPort: number;
  mode: "direct" | "relay";
  state: "negotiating" | "active" | "suspended" | "closed";
  bytesSourceToDestination: number;
  bytesDestinationToSource: number;
  createdAt: number;
  connectedAt: number | null;
  endedAt: number | null;
  connectRttMs: number | null;
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

export interface RuntimeProxyHandle {
  host: "127.0.0.1";
  port: number;
  close(): Promise<void>;
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

export class WorkerHub extends EventEmitter<HubEvents> {
  private readonly connections = new Map<string, ConnectionState>();
  private readonly allConnections = new Set<ConnectionState>();
  private logger: FastifyInstance["log"] | null = null;
  private pendingConnections = 0;
  private readonly runtimeStreams = new Map<string, RuntimeStreamSession>();
  private readonly completedRuntimeTransports: RuntimeTransportSnapshot[] = [];
  private readonly runtimeProxyServers = new Set<Server>();
  private readonly runtimeLinkProbes = new Map<string, RuntimeLinkProbeSession>();
  private readonly runtimeLinkObservationsStore = new RuntimeLinkObservationStore();
  private readonly runtimeLinkLastStartedAt = new Map<string, number>();
  private runtimeLinkProbeTimer: NodeJS.Timeout | null = null;
  private runtimeLinkProbeCursor = 0;

  constructor(private readonly store: MeshStore) {
    super();
  }

  attach(app: FastifyInstance): void {
    this.logger = app.log;
    app.get("/internal/v1/workers/connect", { websocket: true }, (socket) => {
      if (this.pendingConnections >= 256) {
        socket.close(4429, "too many pending connections");
        return;
      }
      const state: ConnectionState = {
        socket,
        workerId: null,
        ready: false,
        helloTimer: setTimeout(() => socket.close(4408, "worker hello timeout"), 5_000),
        pending: true,
        messageWindowStartedAt: Date.now(),
        messagesInWindow: 0,
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

  removeWorker(workerId: string): boolean {
    const existed = Boolean(this.store.getWorker(workerId));
    const state = this.connections.get(workerId);
    if (state) {
      this.connections.delete(workerId);
      state.ready = false;
      state.workerId = null;
      try {
        state.socket.close(4000, "removed from mycellios panel");
      } catch {
        this.handleClose(state);
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
    for (const server of this.runtimeProxyServers) server.close();
    this.runtimeProxyServers.clear();
  }

  closeStaleConnections(): number {
    let closed = 0;
    for (const [workerId, state] of this.connections) {
      const worker = this.store.getWorker(workerId);
      if (worker && worker.status !== "suspect" && worker.status !== "offline") continue;
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

      if (!state.workerId) {
        if (envelope.type !== "worker.hello" || !this.store.getWorker(envelope.workerId)) {
          this.closeInvalid(state, "invalid worker hello", 4404);
          return;
        }
        const previous = this.connections.get(envelope.workerId);
        if (
          previous &&
          previous !== state &&
          previous.ready &&
          previous.socket.readyState === previous.socket.OPEN
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
        this.applyHeartbeat(envelope.workerId, envelope.payload);
        state.ready = true;
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
      if (envelope.type.startsWith("runtime.link.probe.")) {
        if (this.store.getWorker(envelope.workerId)?.capabilities.distributedExecutor?.protocol
          !== "gdlp-worker-tunnel/2") {
          this.closeInvalid(state, "runtime link probes are not enabled", 4403);
          return;
        }
        this.handleRuntimeLinkProbeEnvelope(envelope);
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

  private applyHeartbeat(workerId: string, payload: WorkerHeartbeatPayload): void {
    const status = payload.heartbeat.draining ? "draining" : "online";
    this.store.updateWorkerHeartbeat(workerId, payload.capabilities, status);
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
            this.suspendRuntimeStream(stream);
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
    if (rttMs === null || goodputMbps === null) {
      this.runtimeLinkObservationsStore.recordFailure(
        probe.sourceNodeId,
        probe.destinationNodeId,
      );
    } else {
      this.runtimeLinkObservationsStore.recordSuccess(
        probe.sourceNodeId,
        probe.destinationNodeId,
        rttMs,
        goodputMbps,
      );
    }
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
        && destinationExecutor?.directTransport?.protocol === "mycellios-direct/1"
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
      // Commit the destination first. The source remains paused until both
      // authenticated peers have received coordinator authority.
      const destinationCommitted = this.send(
        session.destinationWorkerId,
        "runtime.direct.commit",
        { streamId, connectionId },
      );
      const sourceCommitted = destinationCommitted && session.sourceWorkerId
        ? this.send(session.sourceWorkerId, "runtime.direct.commit", {
            streamId,
            connectionId,
          })
        : false;
      if (!destinationCommitted || !sourceCommitted) {
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
      this.suspendRuntimeStream(session);
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
      this.suspendRuntimeStream(session);
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

  private suspendRuntimeStream(session: RuntimeStreamSession): void {
    const recovery = session.recovery;
    if (!recovery || recovery.suspended) return;
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

function runtimeLinkKey(fromNodeId: string, toNodeId: string): string {
  return `${fromNodeId}\u0000${toNodeId}`;
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
