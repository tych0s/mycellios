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
}

export interface RuntimeProxyHandle {
  host: "127.0.0.1";
  port: number;
  close(): Promise<void>;
}

const MAX_RUNTIME_STREAMS = 1_024;
const MAX_RUNTIME_STREAM_BUFFERED_BYTES = 8 * 1024 * 1024;
const MAX_WEBSOCKET_BUFFERED_BYTES = 8 * 1024 * 1024;

export class WorkerHub extends EventEmitter<HubEvents> {
  private readonly connections = new Map<string, ConnectionState>();
  private readonly allConnections = new Set<ConnectionState>();
  private logger: FastifyInstance["log"] | null = null;
  private pendingConnections = 0;
  private readonly runtimeStreams = new Map<string, RuntimeStreamSession>();
  private readonly runtimeProxyServers = new Set<Server>();

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
    const server = createServer((socket) => this.attachLocalRuntimeStream(
      socket,
      destinationWorkerId,
      targetPort,
    ));
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
        await new Promise<void>((resolve) => server.close(() => resolve()));
      },
    };
  }

  close(): void {
    for (const state of this.allConnections) state.socket.close(1001, "coordinator shutting down");
    this.connections.clear();
    this.allConnections.clear();
    this.pendingConnections = 0;
    for (const stream of [...this.runtimeStreams.values()]) {
      this.terminateRuntimeStream(stream, "coordinator shutting down");
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
        this.applyHeartbeat(envelope.workerId, envelope.payload);
        state.ready = true;
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
          this.terminateRuntimeStream(stream, `distributed_worker_disconnected:${disconnectedWorkerId}`);
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
      const session: RuntimeStreamSession = {
        streamId,
        sourceWorkerId: envelope.workerId,
        destinationWorkerId: destination.id,
        sourceSequence: 0,
        destinationSequence: 0,
        opened: false,
      };
      this.runtimeStreams.set(streamId, session);
      if (!this.send(destination.id, "runtime.stream.open", {
        streamId,
        targetPort: payload.targetPort,
      })) {
        this.terminateRuntimeStream(session, "runtime_stream_destination_disconnected");
      }
      return;
    }

    const session = this.runtimeStreams.get(streamId);
    if (!session) return;
    const fromSource = session.sourceWorkerId === envelope.workerId;
    const fromDestination = session.destinationWorkerId === envelope.workerId;
    if (!fromSource && !fromDestination) return;

    if (envelope.type === "runtime.stream.opened") {
      if (!fromDestination || session.opened) return;
      session.opened = true;
      if (session.sourceWorkerId) {
        this.send(session.sourceWorkerId, "runtime.stream.opened", { streamId });
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
      const sequence = payload.sequence as number;
      const expected = fromSource ? session.sourceSequence : session.destinationSequence;
      if (sequence !== expected) {
        this.terminateRuntimeStream(session, `runtime_stream_sequence_mismatch:${expected}:${sequence}`);
        return;
      }
      if (fromSource) session.sourceSequence += 1;
      else session.destinationSequence += 1;
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

    if (envelope.type === "runtime.stream.end") {
      this.forwardRuntimeTerminal(session, envelope.workerId, "runtime.stream.end", { streamId });
      this.terminateRuntimeStream(session);
      return;
    }
    if (envelope.type === "runtime.stream.error") {
      this.forwardRuntimeTerminal(session, envelope.workerId, "runtime.stream.error", payload);
      this.terminateRuntimeStream(session);
    }
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
      sourceSequence: 0,
      destinationSequence: 0,
      opened: false,
      localSocket: socket,
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
    if (message) {
      if (session.sourceWorkerId) {
        this.send(session.sourceWorkerId, "runtime.stream.error", {
          streamId: session.streamId,
          message: message.slice(0, 1_024),
        });
      }
      this.send(session.destinationWorkerId, "runtime.stream.error", {
        streamId: session.streamId,
        message: message.slice(0, 1_024),
      });
    }
    session.localSocket?.destroy();
  }
}

function messageType(input: unknown): string | null {
  if (!input || typeof input !== "object" || Array.isArray(input)) return null;
  const type = (input as Record<string, unknown>).type;
  return typeof type === "string" ? type : null;
}
