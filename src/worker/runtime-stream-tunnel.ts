import { randomUUID } from "node:crypto";
import { createConnection, createServer, type AddressInfo, type Server, type Socket } from "node:net";
import { MAX_RUNTIME_STREAM_CHUNK_BYTES } from "../contracts/worker-protocol.js";
import type {
  PythonLaunchProcess,
  PythonPipelineLaunchDescription,
  PythonRootEngineLaunch,
} from "../distribution/python-launcher.js";
import type { DirectSessionGrant } from "../transport/direct-secure-channel.js";
import {
  RuntimeDirectTransport,
  type DirectTransportAdvertisement,
  type DirectTransportCandidate,
  type RuntimeDirectTransportOptions,
  type RuntimeDirectTransportSnapshot,
} from "./runtime-direct-transport.js";

type RuntimeStreamServerMessage =
  | {
      type: "runtime.stream.open";
      payload: {
        streamId: string;
        targetPort: number;
        generation?: number;
        recoveryToken?: string;
      };
    }
  | {
      type: "runtime.stream.opened";
      payload: { streamId: string; generation?: number; recoveryToken?: string };
    }
  | {
      type: "runtime.stream.data";
      payload: {
        streamId: string;
        sequence: number;
        generation?: number;
        recoveryToken?: string;
        offset?: number;
        data: string;
      };
    }
  | {
      type: "runtime.stream.ack";
      payload: {
        streamId: string;
        generation: number;
        recoveryToken: string;
        acknowledgedOffset: number;
      };
    }
  | {
      type: "runtime.stream.suspend";
      payload: {
        streamId: string;
        generation: number;
        recoveryToken: string;
        deadlineAt: number;
      };
    }
  | {
      type: "runtime.stream.resumed";
      payload: {
        streamId: string;
        previousGeneration: number;
        generation: number;
        recoveryToken: string;
        sendFromOffset: number;
      };
    }
  | {
      type: "runtime.stream.end";
      payload: {
        streamId: string;
        generation?: number;
        recoveryToken?: string;
        finalOffset?: number;
      };
    }
  | {
      type: "runtime.stream.error";
      payload: {
        streamId: string;
        generation?: number;
        recoveryToken?: string;
        message: string;
      };
    }
  | {
      type: "runtime.direct.offer";
      payload: {
        streamId: string;
        grant: DirectSessionGrant;
      };
    }
  | {
      type: "runtime.direct.connect";
      payload: {
        streamId: string;
        destinationNodeId: string;
        grant: DirectSessionGrant;
        candidates: DirectTransportCandidate[];
        timeoutMs: number;
      };
    }
  | {
      type: "runtime.direct.commit";
      payload: {
        streamId: string;
        connectionId: string;
      };
    }
  | {
      type: "runtime.direct.cancel";
      payload: {
        streamId: string;
        connectionId: string;
      };
    };

interface RecoveryState {
  recoveryToken: string;
  generation: number;
  sendOffset: number;
  acknowledgedOffset: number;
  receiveOffset: number;
  replayBytes: number;
  replay: ReplayChunk[];
  suspended: boolean;
}

interface ReplayChunk {
  sequence: number;
  offset: number;
  data: string;
  byteLength: number;
}

interface StreamSession {
  streamId: string;
  socket: Socket;
  opened: boolean;
  sendSequence: number;
  receiveSequence: number;
  recovery: RecoveryState | null;
  destinationNodeId: string | null;
  targetPort: number;
  transportMode: "negotiating" | "relay" | "direct";
  createdAt: number;
  connectedAt: number | null;
  endedAt: number | null;
  bytesTx: number;
  bytesRx: number;
}

interface CommandRewrite {
  values: Map<string, string>;
}

const LOOPBACK = "127.0.0.1";
const MAX_STREAMS = 256;
const CONNECT_TIMEOUT_MS = 15_000;
const MAX_SOCKET_BUFFERED_BYTES = 8 * 1024 * 1024;
export const MAX_RUNTIME_STREAM_REPLAY_BYTES = 8 * 1024 * 1024;

export interface RuntimeStreamTunnelOptions {
  /** Primarily configurable for deterministic memory-boundary tests. */
  maxReplayBytes?: number;
  /** Tests can start with the relay intentionally unavailable. */
  transportInitiallyAvailable?: boolean;
  directTransport?: RuntimeDirectTransportOptions;
}

export interface RuntimeStreamTransportSnapshot {
  streamId: string;
  sourceNodeId: string;
  destinationNodeId: string | null;
  targetPort: number;
  mode: "direct" | "relay";
  state: "negotiating" | "active" | "suspended" | "closed";
  bytesTx: number;
  bytesRx: number;
  createdAt: number;
  connectedAt: number | null;
  endedAt: number | null;
  connectRttMs: number | null;
}

/**
 * Keeps every Python runtime socket private to the desktop and carries its
 * bytes over the already-authenticated coordinator WebSocket.
 */
export class RuntimeStreamTunnel {
  private readonly proxyServers = new Set<Server>();
  private readonly sessions = new Map<string, StreamSession>();
  private readonly allowedTargetPorts = new Set<number>();
  private readonly rewrites = new Map<string, CommandRewrite>();
  private readonly maxReplayBytes: number;
  private readonly replayHighWaterBytes: number;
  private readonly replayLowWaterBytes: number;
  private readonly directTransport: RuntimeDirectTransport;
  private readonly completedTransportSnapshots: RuntimeStreamTransportSnapshot[] = [];
  private transportAvailable: boolean;

  constructor(
    private readonly nodeId: string,
    private readonly send: (type: string, payload: unknown) => boolean | void,
    options: RuntimeStreamTunnelOptions = {},
  ) {
    const requestedReplayBytes = options.maxReplayBytes ?? MAX_RUNTIME_STREAM_REPLAY_BYTES;
    if (
      !Number.isSafeInteger(requestedReplayBytes)
      || requestedReplayBytes < MAX_RUNTIME_STREAM_CHUNK_BYTES
      || requestedReplayBytes > MAX_RUNTIME_STREAM_REPLAY_BYTES
    ) {
      throw new Error("runtime_stream_replay_limit_is_invalid");
    }
    this.maxReplayBytes = requestedReplayBytes;
    this.replayHighWaterBytes = Math.max(
      MAX_RUNTIME_STREAM_CHUNK_BYTES,
      Math.floor(requestedReplayBytes * 0.75),
    );
    this.replayLowWaterBytes = Math.floor(requestedReplayBytes * 0.5);
    this.transportAvailable = options.transportInitiallyAvailable ?? true;
    this.directTransport = new RuntimeDirectTransport(
      nodeId,
      {
        send: (type, payload) => this.sendEnvelope(type, payload),
        isTargetPortAuthorized: (targetPort) => this.allowedTargetPorts.has(targetPort),
        onSourceCommitted: (streamId) => {
          const session = this.sessions.get(streamId);
          if (!session || session.transportMode !== "negotiating") return;
          session.transportMode = "direct";
          session.opened = true;
          session.connectedAt = Date.now();
          session.socket.resume();
        },
        onSourceFailed: (streamId) => {
          const session = this.sessions.get(streamId);
          if (!session || session.transportMode !== "negotiating") return;
          // The coordinator will explicitly open the existing relay route.
        },
        onSourceClosed: (streamId, error) => {
          this.closeSession(streamId, false, error);
        },
      },
      options.directTransport,
    );
  }

  startDirectTransport(): Promise<DirectTransportAdvertisement | null> {
    return this.directTransport.start();
  }

  async prepare(description: PythonPipelineLaunchDescription): Promise<void> {
    await this.resetPreparedRuntime();
    const root = description.launchOrder.find(
      (process): process is PythonRootEngineLaunch => process.kind === "root-engine",
    );
    if (!root) throw new Error("distributed_runtime_root_is_missing");
    const local = description.launchOrder.filter((process) => process.anchor.memberId === this.nodeId);
    if (local.length === 0) throw new Error("distributed_plan_has_no_process_for_this_node");

    for (const process of local) {
      this.allowedTargetPorts.add(process.anchor.endpoint.port);
      const values = new Map<string, string>();
      if (process.kind === "root-engine") {
        this.allowedTargetPorts.add(process.apiEndpoint.port);
        this.allowedTargetPorts.add(process.returnEndpoint.port);
        const firstStage = await this.createEgressProxy(
          process.firstRemoteStage.anchorMemberId,
          process.firstRemoteStage.endpoint.port,
        );
        values.set("--host", LOOPBACK);
        values.set("--first-stage-host", LOOPBACK);
        values.set("--first-stage-port", String(firstStage));
        values.set("--return-bind-host", LOOPBACK);
        values.set("--return-advertise-host", LOOPBACK);
      } else if (process.kind === "remote-stage") {
        values.set("--listen-host", LOOPBACK);
        if (process.downstream) {
          const downstream = await this.createEgressProxy(
            process.downstream.anchorMemberId,
            process.downstream.endpoint.port,
          );
          values.set("--next-host", LOOPBACK);
          values.set("--next-port", String(downstream));
        }
        const returnProxy = await this.createEgressProxy(
          root.anchor.memberId,
          root.returnEndpoint.port,
        );
        values.set("--return-host", LOOPBACK);
        values.set("--return-port", String(returnProxy));
      }
      this.rewrites.set(process.processId, { values });
    }
  }

  rewriteProcess(process: PythonLaunchProcess): PythonLaunchProcess {
    const rewrite = this.rewrites.get(process.processId);
    if (!rewrite) return process;
    const args = [...process.command.args];
    for (const [flag, value] of rewrite.values) replaceFlagValue(args, flag, value);
    return {
      ...process,
      command: { ...process.command, args },
    } as PythonLaunchProcess;
  }

  async handle(message: RuntimeStreamServerMessage): Promise<void> {
    switch (message.type) {
      case "runtime.stream.open":
        await this.openIncoming(
          message.payload.streamId,
          message.payload.targetPort,
          message.payload.generation,
          message.payload.recoveryToken,
        );
        break;
      case "runtime.stream.opened": {
        const session = this.sessions.get(message.payload.streamId);
        if (!session || session.opened) return;
        if (
          session.recovery
          && message.payload.generation === undefined
          && message.payload.recoveryToken === undefined
          && session.recovery.sendOffset === 0
          && session.recovery.receiveOffset === 0
        ) {
          // A pre-extension coordinator strips unknown open fields. Downgrade
          // only before either direction has carried a byte; after that point
          // losing the recovery identity would be ambiguous and must fail.
          session.recovery = null;
        }
        if (!this.matchesRecoveryIdentity(session, message.payload)) {
          this.closeSession(
            message.payload.streamId,
            true,
            new Error("runtime_stream_opened_recovery_identity_mismatch"),
          );
          return;
        }
        session.transportMode = "relay";
        session.opened = true;
        session.connectedAt ??= Date.now();
        if (!session.recovery?.suspended) session.socket.resume();
        break;
      }
      case "runtime.stream.data":
        this.receiveData(message.payload);
        break;
      case "runtime.stream.ack":
        this.receiveAcknowledgement(message.payload);
        break;
      case "runtime.stream.suspend":
        this.suspendSession(message.payload);
        break;
      case "runtime.stream.resumed":
        this.resumeSession(message.payload);
        break;
      case "runtime.stream.end":
        this.receiveEnd(message.payload);
        break;
      case "runtime.stream.error":
        this.receiveError(message.payload);
        break;
      case "runtime.direct.offer":
        this.directTransport.installDestinationOffer(
          message.payload.streamId,
          message.payload.grant,
        );
        break;
      case "runtime.direct.connect": {
        const session = this.sessions.get(message.payload.streamId);
        if (
          !session
          || session.transportMode !== "negotiating"
          || session.opened
          || session.bytesTx !== 0
          || session.bytesRx !== 0
          || session.destinationNodeId !== message.payload.destinationNodeId
          || session.targetPort !== message.payload.grant.targetPort
        ) {
          this.sendEnvelope("runtime.direct.fallback", {
            streamId: message.payload.streamId,
            connectionId: message.payload.grant.connectionId,
            reason: "direct_source_session_is_invalid",
          });
          break;
        }
        await this.directTransport.connectSource({
          ...message.payload,
          socket: session.socket,
        });
        break;
      }
      case "runtime.direct.commit":
        this.directTransport.commit(
          message.payload.streamId,
          message.payload.connectionId,
        );
        break;
      case "runtime.direct.cancel":
        this.directTransport.cancel(
          message.payload.streamId,
          message.payload.connectionId,
        );
        break;
    }
  }

  /**
   * Freezes recoverable sockets while the authenticated relay is unavailable.
   * Legacy streams cannot prove delivery offsets and are therefore closed.
   */
  transportDisconnected(): void {
    if (!this.transportAvailable) return;
    this.transportAvailable = false;
    for (const session of [...this.sessions.values()]) {
      if (session.transportMode === "direct") continue;
      if (!session.recovery) {
        this.closeSession(
          session.streamId,
          false,
          new Error("runtime_stream_transport_disconnected"),
        );
        continue;
      }
      session.recovery.suspended = true;
      session.socket.pause();
    }
  }

  /** Re-advertises exact offsets; forwarding resumes only after coordinator validation. */
  transportConnected(): void {
    if (this.transportAvailable) return;
    this.transportAvailable = true;
    for (const session of this.sessions.values()) {
      if (session.recovery?.suspended) this.sendResumeState(session);
    }
  }

  recoverySnapshot(): Array<{
    streamId: string;
    generation: number;
    sendOffset: number;
    acknowledgedOffset: number;
    receiveOffset: number;
    replayBytes: number;
    suspended: boolean;
  }> {
    return [...this.sessions.values()]
      .filter((session): session is StreamSession & { recovery: RecoveryState } =>
        session.recovery !== null)
      .map((session) => ({
        streamId: session.streamId,
        generation: session.recovery.generation,
        sendOffset: session.recovery.sendOffset,
        acknowledgedOffset: session.recovery.acknowledgedOffset,
        receiveOffset: session.recovery.receiveOffset,
        replayBytes: session.recovery.replayBytes,
        suspended: session.recovery.suspended,
      }));
  }

  transportSnapshot(): RuntimeStreamTransportSnapshot[] {
    const direct = this.directTransport.snapshot();
    const directIds = new Set(direct.map((snapshot) => snapshot.streamId));
    const relay = [...this.sessions.values()]
      .filter((session) => !directIds.has(session.streamId))
      .map((session): RuntimeStreamTransportSnapshot => ({
        streamId: session.streamId,
        sourceNodeId: this.nodeId,
        destinationNodeId: session.destinationNodeId,
        targetPort: session.targetPort,
        mode: "relay",
        state: session.endedAt !== null
          ? "closed"
          : session.transportMode === "negotiating"
            ? "negotiating"
            : session.recovery?.suspended
              ? "suspended"
              : "active",
        bytesTx: session.bytesTx,
        bytesRx: session.bytesRx,
        createdAt: session.createdAt,
        connectedAt: session.connectedAt,
        endedAt: session.endedAt,
        connectRttMs: null,
      }));
    return [
      ...direct.map((snapshot: RuntimeDirectTransportSnapshot) => ({ ...snapshot })),
      ...relay,
      ...this.completedTransportSnapshots,
    ];
  }

  async close(): Promise<void> {
    await this.resetPreparedRuntime();
    await this.directTransport.close();
  }

  reset(): Promise<void> {
    return this.resetPreparedRuntime();
  }

  private async resetPreparedRuntime(): Promise<void> {
    for (const session of [...this.sessions.values()]) this.closeSession(session.streamId, true);
    const servers = [...this.proxyServers];
    this.proxyServers.clear();
    await Promise.all(servers.map((server) => new Promise<void>((resolve) => server.close(() => resolve()))));
    this.allowedTargetPorts.clear();
    this.rewrites.clear();
  }

  private async createEgressProxy(destinationNodeId: string, targetPort: number): Promise<number> {
    const server = createServer((socket) => this.openOutgoing(socket, destinationNodeId, targetPort));
    this.proxyServers.add(server);
    await listenLoopback(server);
    return (server.address() as AddressInfo).port;
  }

  private openOutgoing(socket: Socket, destinationNodeId: string, targetPort: number): void {
    if (this.sessions.size >= MAX_STREAMS) {
      socket.destroy(new Error("runtime_stream_capacity_exceeded"));
      return;
    }
    const streamId = `${shortNodeId(this.nodeId)}-${randomUUID()}`;
    const recoveryToken = randomUUID();
    this.attachSocket(
      streamId,
      socket,
      false,
      { recoveryToken, generation: 0 },
      { destinationNodeId, targetPort, transportMode: "negotiating" },
    );
    socket.pause();
    if (!this.sendEnvelope("runtime.stream.open", {
      streamId,
      destinationNodeId,
      targetPort,
      generation: 0,
      recoveryToken,
    })) {
      this.closeSession(streamId, false, new Error("runtime_stream_transport_unavailable"));
    }
  }

  private async openIncoming(
    streamId: string,
    targetPort: number,
    generation?: number,
    recoveryToken?: string,
  ): Promise<void> {
    if (
      this.sessions.has(streamId) ||
      this.sessions.size >= MAX_STREAMS ||
      !this.allowedTargetPorts.has(targetPort)
    ) {
      this.sendEnvelope("runtime.stream.error", {
        streamId,
        message: this.sessions.has(streamId)
          ? "runtime_stream_is_duplicate"
          : this.sessions.size >= MAX_STREAMS
            ? "runtime_stream_capacity_exceeded"
            : `runtime_stream_target_is_not_authorized:${targetPort}`,
      });
      return;
    }
    const hasRecoveryGeneration = generation !== undefined;
    const hasRecoveryToken = recoveryToken !== undefined;
    if (hasRecoveryGeneration !== hasRecoveryToken) {
      this.sendEnvelope("runtime.stream.error", {
        streamId,
        message: "runtime_stream_recovery_identity_is_incomplete",
      });
      return;
    }
    const socket = createConnection({ host: LOOPBACK, port: targetPort });
    socket.setNoDelay(true);
    socket.setTimeout(CONNECT_TIMEOUT_MS);
    const recovery = generation !== undefined && recoveryToken !== undefined
      ? { generation, recoveryToken }
      : undefined;
    const session = this.attachSocket(
      streamId,
      socket,
      true,
      recovery,
      { destinationNodeId: null, targetPort, transportMode: "relay" },
    );
    await new Promise<void>((resolve) => {
      const fail = (error: Error) => {
        if (!this.sessions.has(streamId)) return resolve();
        this.sendSessionEnvelope(session, "runtime.stream.error", {
          message: error.message.slice(0, 1_024),
        });
        this.closeSession(streamId, false, error);
        resolve();
      };
      socket.once("connect", () => {
        socket.setTimeout(0);
        session.opened = true;
        this.sendSessionEnvelope(session, "runtime.stream.opened", {});
        resolve();
      });
      socket.once("timeout", () => fail(new Error(`runtime_stream_connect_timeout:${targetPort}`)));
      socket.once("error", fail);
    });
  }

  private attachSocket(
    streamId: string,
    socket: Socket,
    opened: boolean,
    recoveryIdentity?: { recoveryToken: string; generation: number },
    route: {
      destinationNodeId: string | null;
      targetPort: number;
      transportMode: StreamSession["transportMode"];
    } = { destinationNodeId: null, targetPort: 0, transportMode: "relay" },
  ): StreamSession {
    socket.setNoDelay(true);
    const session: StreamSession = {
      streamId,
      socket,
      opened,
      sendSequence: 0,
      receiveSequence: 0,
      destinationNodeId: route.destinationNodeId,
      targetPort: route.targetPort,
      transportMode: route.transportMode,
      createdAt: Date.now(),
      connectedAt: opened ? Date.now() : null,
      endedAt: null,
      bytesTx: 0,
      bytesRx: 0,
      recovery: recoveryIdentity
        ? {
            ...recoveryIdentity,
            sendOffset: 0,
            acknowledgedOffset: 0,
            receiveOffset: 0,
            replayBytes: 0,
            replay: [],
            suspended: !this.transportAvailable,
          }
        : null,
    };
    this.sessions.set(streamId, session);
    socket.on("data", (chunk: Buffer) => {
      if (!session.opened) return;
      if (session.transportMode === "direct") {
        if (!this.directTransport.writeSource(session.streamId, chunk)) {
          this.closeSession(
            session.streamId,
            false,
            new Error("direct_transport_session_is_unavailable"),
          );
        } else {
          session.bytesTx += chunk.byteLength;
        }
        return;
      }
      if (session.transportMode !== "relay") {
        this.closeSession(
          session.streamId,
          false,
          new Error("runtime_stream_data_before_transport_commit"),
        );
        return;
      }
      for (let offset = 0; offset < chunk.byteLength; offset += MAX_RUNTIME_STREAM_CHUNK_BYTES) {
        const piece = chunk.subarray(
          offset,
          Math.min(chunk.byteLength, offset + MAX_RUNTIME_STREAM_CHUNK_BYTES),
        );
        if (!this.sendSocketChunk(session, piece)) return;
      }
    });
    socket.once("end", () => this.closeSession(streamId, true));
    socket.once("close", () => this.closeSession(streamId, true));
    socket.once("error", (error) => this.closeSession(streamId, true, error));
    return session;
  }

  private sendSocketChunk(session: StreamSession, piece: Buffer): boolean {
    session.bytesTx += piece.byteLength;
    const data = piece.toString("base64");
    const sequence = session.sendSequence++;
    const recovery = session.recovery;
    if (!recovery) {
      if (!this.sendEnvelope("runtime.stream.data", { streamId: session.streamId, sequence, data })) {
        this.closeSession(
          session.streamId,
          false,
          new Error("runtime_stream_transport_unavailable"),
        );
        return false;
      }
      return true;
    }

    if (recovery.replayBytes + piece.byteLength > this.maxReplayBytes) {
      this.closeSession(
        session.streamId,
        true,
        new Error("runtime_stream_replay_buffer_exceeded"),
      );
      return false;
    }
    const chunk: ReplayChunk = {
      sequence,
      offset: recovery.sendOffset,
      data,
      byteLength: piece.byteLength,
    };
    recovery.replay.push(chunk);
    recovery.replayBytes += piece.byteLength;
    recovery.sendOffset += piece.byteLength;
    if (recovery.replayBytes >= this.replayHighWaterBytes) session.socket.pause();

    if (!this.transportAvailable || recovery.suspended) return true;
    if (!this.sendRecoveryChunk(session, chunk)) {
      recovery.suspended = true;
      session.socket.pause();
    }
    return true;
  }

  private sendRecoveryChunk(session: StreamSession, chunk: ReplayChunk): boolean {
    const recovery = session.recovery;
    if (!recovery) return false;
    return this.sendEnvelope("runtime.stream.data", {
      streamId: session.streamId,
      sequence: chunk.sequence,
      generation: recovery.generation,
      recoveryToken: recovery.recoveryToken,
      offset: chunk.offset,
      data: chunk.data,
    });
  }

  private receiveData(payload: {
    streamId: string;
    sequence: number;
    generation?: number;
    recoveryToken?: string;
    offset?: number;
    data: string;
  }): void {
    const session = this.sessions.get(payload.streamId);
    if (!session || !session.opened) return;
    if (session.recovery) {
      this.receiveRecoverableData(session, payload);
      return;
    }
    if (
      payload.generation !== undefined
      || payload.recoveryToken !== undefined
      || payload.offset !== undefined
    ) {
      this.closeSession(
        payload.streamId,
        true,
        new Error("runtime_stream_unnegotiated_recovery_data"),
      );
      return;
    }
    if (payload.sequence !== session.receiveSequence) {
      this.closeSession(
        payload.streamId,
        true,
        new Error(`runtime_stream_sequence_mismatch:${session.receiveSequence}:${payload.sequence}`),
      );
      return;
    }
    session.receiveSequence += 1;
    if (session.socket.writableLength > MAX_SOCKET_BUFFERED_BYTES) {
      this.closeSession(payload.streamId, true, new Error("runtime_stream_buffer_exceeded"));
      return;
    }
    const data = Buffer.from(payload.data, "base64");
    session.socket.write(data);
    session.bytesRx += data.byteLength;
  }

  private receiveRecoverableData(
    session: StreamSession,
    payload: {
      streamId: string;
      sequence: number;
      generation?: number;
      recoveryToken?: string;
      offset?: number;
      data: string;
    },
  ): void {
    const recovery = session.recovery!;
    if (
      payload.generation !== recovery.generation
      || payload.recoveryToken !== recovery.recoveryToken
      || payload.offset === undefined
    ) {
      this.closeSession(
        session.streamId,
        true,
        new Error("runtime_stream_recovery_identity_mismatch"),
      );
      return;
    }
    const data = Buffer.from(payload.data, "base64");
    const endOffset = payload.offset + data.byteLength;
    if (payload.offset < recovery.receiveOffset) {
      if (endOffset <= recovery.receiveOffset) {
        // A replayed chunk whose ACK was lost is safe and idempotent.
        this.sendAcknowledgement(session);
        return;
      }
      this.closeSession(
        session.streamId,
        true,
        new Error(
          `runtime_stream_overlapping_replay:${recovery.receiveOffset}:${payload.offset}:${endOffset}`,
        ),
      );
      return;
    }
    if (payload.offset !== recovery.receiveOffset) {
      this.closeSession(
        session.streamId,
        true,
        new Error(
          `runtime_stream_offset_mismatch:${recovery.receiveOffset}:${payload.offset}`,
        ),
      );
      return;
    }
    if (
      session.socket.destroyed
      || session.socket.writableLength + data.byteLength > MAX_SOCKET_BUFFERED_BYTES
    ) {
      this.closeSession(session.streamId, true, new Error("runtime_stream_buffer_exceeded"));
      return;
    }
    session.socket.write(data);
    session.bytesRx += data.byteLength;
    recovery.receiveOffset = endOffset;
    this.sendAcknowledgement(session);
  }

  private sendAcknowledgement(session: StreamSession): void {
    const recovery = session.recovery;
    if (!recovery || !this.transportAvailable || recovery.suspended) return;
    this.sendEnvelope("runtime.stream.ack", {
      streamId: session.streamId,
      generation: recovery.generation,
      recoveryToken: recovery.recoveryToken,
      acknowledgedOffset: recovery.receiveOffset,
    });
  }

  private receiveAcknowledgement(payload: {
    streamId: string;
    generation: number;
    recoveryToken: string;
    acknowledgedOffset: number;
  }): void {
    const session = this.sessions.get(payload.streamId);
    const recovery = session?.recovery;
    if (!session || !recovery) return;
    if (
      payload.generation !== recovery.generation
      || payload.recoveryToken !== recovery.recoveryToken
    ) {
      this.closeSession(
        payload.streamId,
        true,
        new Error("runtime_stream_ack_recovery_identity_mismatch"),
      );
      return;
    }
    if (payload.acknowledgedOffset <= recovery.acknowledgedOffset) return;
    if (
      payload.acknowledgedOffset > recovery.sendOffset
      || !this.trimReplayToOffset(recovery, payload.acknowledgedOffset)
    ) {
      this.closeSession(
        payload.streamId,
        true,
        new Error(
          `runtime_stream_ack_offset_is_invalid:${recovery.acknowledgedOffset}:${payload.acknowledgedOffset}:${recovery.sendOffset}`,
        ),
      );
      return;
    }
    if (
      session.opened
      && !recovery.suspended
      && recovery.replayBytes <= this.replayLowWaterBytes
    ) {
      session.socket.resume();
    }
  }

  private trimReplayToOffset(recovery: RecoveryState, acknowledgedOffset: number): boolean {
    let removedBytes = 0;
    let consumed = 0;
    for (const chunk of recovery.replay) {
      const end = chunk.offset + chunk.byteLength;
      if (end > acknowledgedOffset) break;
      removedBytes += chunk.byteLength;
      consumed += 1;
    }
    const next = recovery.replay[consumed];
    if (
      acknowledgedOffset !== recovery.sendOffset
      && next?.offset !== acknowledgedOffset
    ) {
      return false;
    }
    if (consumed > 0) recovery.replay.splice(0, consumed);
    recovery.replayBytes -= removedBytes;
    recovery.acknowledgedOffset = acknowledgedOffset;
    return true;
  }

  private suspendSession(payload: {
    streamId: string;
    generation: number;
    recoveryToken: string;
    deadlineAt: number;
  }): void {
    const session = this.sessions.get(payload.streamId);
    const recovery = session?.recovery;
    if (!session || !recovery) return;
    if (
      payload.generation !== recovery.generation
      || payload.recoveryToken !== recovery.recoveryToken
      || payload.deadlineAt <= Date.now()
    ) {
      this.closeSession(
        payload.streamId,
        true,
        new Error("runtime_stream_suspend_is_invalid"),
      );
      return;
    }
    recovery.suspended = true;
    session.socket.pause();
    this.sendResumeState(session);
  }

  private sendResumeState(session: StreamSession): void {
    const recovery = session.recovery;
    if (!recovery || !this.transportAvailable) return;
    const bufferedFromOffset = recovery.replay[0]?.offset ?? recovery.acknowledgedOffset;
    if (
      bufferedFromOffset > recovery.acknowledgedOffset
      || recovery.acknowledgedOffset > recovery.sendOffset
    ) {
      this.closeSession(
        session.streamId,
        true,
        new Error("runtime_stream_replay_state_is_invalid"),
      );
      return;
    }
    this.sendEnvelope("runtime.stream.resume", {
      streamId: session.streamId,
      generation: recovery.generation,
      recoveryToken: recovery.recoveryToken,
      sendOffset: recovery.sendOffset,
      acknowledgedOffset: recovery.acknowledgedOffset,
      receiveOffset: recovery.receiveOffset,
      bufferedFromOffset,
    });
  }

  private resumeSession(payload: {
    streamId: string;
    previousGeneration: number;
    generation: number;
    recoveryToken: string;
    sendFromOffset: number;
  }): void {
    const session = this.sessions.get(payload.streamId);
    const recovery = session?.recovery;
    if (!session || !recovery) return;
    const duplicateAcceptance =
      payload.previousGeneration + 1 === payload.generation
      && recovery.generation === payload.generation;
    if (!duplicateAcceptance && (
      payload.previousGeneration !== recovery.generation
      || payload.generation !== recovery.generation + 1
    )) {
      this.closeSession(
        payload.streamId,
        true,
        new Error("runtime_stream_resume_generation_is_invalid"),
      );
      return;
    }
    if (payload.recoveryToken !== recovery.recoveryToken) {
      this.closeSession(
        payload.streamId,
        true,
        new Error("runtime_stream_resume_token_is_invalid"),
      );
      return;
    }
    const bufferedFromOffset = recovery.replay[0]?.offset ?? recovery.acknowledgedOffset;
    if (
      payload.sendFromOffset < bufferedFromOffset
      || payload.sendFromOffset > recovery.sendOffset
      || (
        payload.sendFromOffset !== recovery.sendOffset
        && !recovery.replay.some((chunk) => chunk.offset === payload.sendFromOffset)
      )
    ) {
      this.closeSession(
        payload.streamId,
        true,
        new Error(
          `runtime_stream_resume_offset_is_unavailable:${bufferedFromOffset}:${payload.sendFromOffset}:${recovery.sendOffset}`,
        ),
      );
      return;
    }
    if (!duplicateAcceptance) {
      if (!this.trimReplayToOffset(recovery, payload.sendFromOffset)) {
        this.closeSession(
          payload.streamId,
          true,
          new Error("runtime_stream_resume_offset_is_not_a_chunk_boundary"),
        );
        return;
      }
      recovery.generation = payload.generation;
    }
    recovery.suspended = false;
    for (const chunk of recovery.replay) {
      if (chunk.offset < payload.sendFromOffset) continue;
      if (!this.sendRecoveryChunk(session, chunk)) {
        recovery.suspended = true;
        session.socket.pause();
        return;
      }
    }
    this.sendAcknowledgement(session);
    if (recovery.replayBytes <= this.replayLowWaterBytes) session.socket.resume();
  }

  private receiveEnd(payload: {
    streamId: string;
    generation?: number;
    recoveryToken?: string;
    finalOffset?: number;
  }): void {
    const session = this.sessions.get(payload.streamId);
    if (!session) return;
    if (!this.matchesRecoveryIdentity(session, payload)) {
      this.closeSession(
        payload.streamId,
        true,
        new Error("runtime_stream_end_recovery_identity_mismatch"),
      );
      return;
    }
    if (
      session.recovery
      && payload.finalOffset !== session.recovery.receiveOffset
    ) {
      this.closeSession(
        payload.streamId,
        true,
        new Error(
          `runtime_stream_end_offset_mismatch:${session.recovery.receiveOffset}:${payload.finalOffset ?? -1}`,
        ),
      );
      return;
    }
    this.closeSession(payload.streamId, false);
  }

  private receiveError(payload: {
    streamId: string;
    generation?: number;
    recoveryToken?: string;
    message: string;
  }): void {
    const session = this.sessions.get(payload.streamId);
    if (!session) return;
    if (!this.matchesRecoveryIdentity(session, payload)) {
      this.closeSession(
        payload.streamId,
        false,
        new Error("runtime_stream_error_recovery_identity_mismatch"),
      );
      return;
    }
    this.closeSession(payload.streamId, false, new Error(payload.message));
  }

  private matchesRecoveryIdentity(
    session: StreamSession,
    payload: { generation?: number; recoveryToken?: string },
  ): boolean {
    if (!session.recovery) {
      return payload.generation === undefined && payload.recoveryToken === undefined;
    }
    return (
      payload.generation === session.recovery.generation
      && payload.recoveryToken === session.recovery.recoveryToken
    );
  }

  private sendSessionEnvelope(
    session: StreamSession,
    type: string,
    payload: Record<string, unknown>,
  ): boolean {
    const recovery = session.recovery;
    return this.sendEnvelope(type, {
      streamId: session.streamId,
      ...(recovery
        ? {
            generation: recovery.generation,
            recoveryToken: recovery.recoveryToken,
          }
        : {}),
      ...payload,
    });
  }

  private sendEnvelope(type: string, payload: unknown): boolean {
    if (!this.transportAvailable) return false;
    return this.send(type, payload) !== false;
  }

  private closeSession(streamId: string, notify: boolean, error?: Error): void {
    const session = this.sessions.get(streamId);
    if (!session || !this.sessions.delete(streamId)) return;
    session.endedAt = Date.now();
    if (session.transportMode !== "direct") {
      this.completedTransportSnapshots.unshift({
        streamId: session.streamId,
        sourceNodeId: this.nodeId,
        destinationNodeId: session.destinationNodeId,
        targetPort: session.targetPort,
        mode: "relay",
        state: "closed",
        bytesTx: session.bytesTx,
        bytesRx: session.bytesRx,
        createdAt: session.createdAt,
        connectedAt: session.connectedAt,
        endedAt: session.endedAt,
        connectRttMs: null,
      });
      if (this.completedTransportSnapshots.length > 256) {
        this.completedTransportSnapshots.length = 256;
      }
    } else {
      this.directTransport.closeSource(streamId, error);
    }
    if (notify && session.transportMode !== "direct") {
      this.sendSessionEnvelope(
        session,
        error ? "runtime.stream.error" : "runtime.stream.end",
        error
          ? { message: error.message.slice(0, 1_024) }
          : {
              ...(session.recovery
                ? { finalOffset: session.recovery.sendOffset }
                : {}),
            },
      );
    }
    session.socket.destroy(error);
  }
}

function replaceFlagValue(args: string[], flag: string, value: string): void {
  const index = args.indexOf(flag);
  if (index < 0 || index + 1 >= args.length) throw new Error(`runtime_command_flag_is_missing:${flag}`);
  args[index + 1] = value;
}

function listenLoopback(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    const fail = (error: Error) => {
      server.off("listening", ready);
      reject(error);
    };
    const ready = () => {
      server.off("error", fail);
      resolve();
    };
    server.once("error", fail);
    server.once("listening", ready);
    server.listen(0, LOOPBACK);
  });
}

function shortNodeId(nodeId: string): string {
  return nodeId.replace(/[^A-Za-z0-9_-]/g, "-").slice(0, 64);
}

export type { RuntimeStreamServerMessage };
