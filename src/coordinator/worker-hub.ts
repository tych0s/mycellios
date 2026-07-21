import { EventEmitter } from "node:events";
import type { FastifyInstance } from "fastify";
import type WebSocket from "ws";
import type { ServerEnvelope, WorkerEnvelope } from "../contracts/types.js";
import {
  parseWorkerEnvelope,
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

export class WorkerHub extends EventEmitter<HubEvents> {
  private readonly connections = new Map<string, ConnectionState>();
  private readonly allConnections = new Set<ConnectionState>();
  private pendingConnections = 0;

  constructor(private readonly store: MeshStore) {
    super();
  }

  attach(app: FastifyInstance): void {
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
      socket.on("close", () => this.handleClose(state));
      socket.on("error", () => this.handleClose(state));
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
    this.sendSocket(state.socket, type, payload);
    return true;
  }

  close(): void {
    for (const state of this.allConnections) state.socket.close(1001, "coordinator shutting down");
    this.connections.clear();
    this.allConnections.clear();
    this.pendingConnections = 0;
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
      if (state.messagesInWindow > 256) {
        this.closeInvalid(state, "worker message rate exceeded", 4429);
        return;
      }
      if (Buffer.byteLength(raw, "utf8") > 2_200_000) {
        this.closeInvalid(state, "frame too large");
        return;
      }
      const envelope = parseWorkerEnvelope(JSON.parse(raw));
      if (!envelope) {
        this.closeInvalid(state, "invalid worker message");
        return;
      }

      if (!state.workerId) {
        if (envelope.type !== "worker.hello" || !this.store.getWorker(envelope.workerId)) {
          this.closeInvalid(state, "invalid worker hello", 4404);
          return;
        }
        const previous = this.connections.get(envelope.workerId);
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
      this.emit("envelope", envelope);
    } catch {
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
      this.connections.delete(state.workerId);
      this.store.setWorkerStatus(state.workerId, "offline");
      this.emit("disconnect", state.workerId);
    }
  }

  private sendSocket(socket: WebSocket, type: string, payload: unknown): void {
    const envelope: ServerEnvelope = { v: 1, type, payload };
    socket.send(JSON.stringify(envelope));
  }
}
