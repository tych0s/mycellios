import { randomUUID } from "node:crypto";
import { createConnection, createServer, type AddressInfo, type Server, type Socket } from "node:net";
import { MAX_RUNTIME_STREAM_CHUNK_BYTES } from "../contracts/worker-protocol.js";
import type {
  PythonLaunchProcess,
  PythonPipelineLaunchDescription,
  PythonRootEngineLaunch,
} from "../distribution/python-launcher.js";

type RuntimeStreamServerMessage =
  | { type: "runtime.stream.open"; payload: { streamId: string; targetPort: number } }
  | { type: "runtime.stream.opened"; payload: { streamId: string } }
  | { type: "runtime.stream.data"; payload: { streamId: string; sequence: number; data: string } }
  | { type: "runtime.stream.end"; payload: { streamId: string } }
  | { type: "runtime.stream.error"; payload: { streamId: string; message: string } };

interface StreamSession {
  streamId: string;
  socket: Socket;
  opened: boolean;
  sendSequence: number;
  receiveSequence: number;
}

interface CommandRewrite {
  values: Map<string, string>;
}

const LOOPBACK = "127.0.0.1";
const MAX_STREAMS = 256;
const CONNECT_TIMEOUT_MS = 15_000;
const MAX_SOCKET_BUFFERED_BYTES = 8 * 1024 * 1024;

/**
 * Keeps every Python runtime socket private to the desktop and carries its
 * bytes over the already-authenticated coordinator WebSocket.
 */
export class RuntimeStreamTunnel {
  private readonly proxyServers = new Set<Server>();
  private readonly sessions = new Map<string, StreamSession>();
  private readonly allowedTargetPorts = new Set<number>();
  private readonly rewrites = new Map<string, CommandRewrite>();

  constructor(
    private readonly nodeId: string,
    private readonly send: (type: string, payload: unknown) => void,
  ) {}

  async prepare(description: PythonPipelineLaunchDescription): Promise<void> {
    await this.close();
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
        await this.openIncoming(message.payload.streamId, message.payload.targetPort);
        break;
      case "runtime.stream.opened": {
        const session = this.sessions.get(message.payload.streamId);
        if (!session || session.opened) return;
        session.opened = true;
        session.socket.resume();
        break;
      }
      case "runtime.stream.data":
        this.receiveData(message.payload);
        break;
      case "runtime.stream.end":
        this.closeSession(message.payload.streamId, false);
        break;
      case "runtime.stream.error":
        this.closeSession(message.payload.streamId, false, new Error(message.payload.message));
        break;
    }
  }

  async close(): Promise<void> {
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
    this.attachSocket(streamId, socket, false);
    socket.pause();
    this.send("runtime.stream.open", { streamId, destinationNodeId, targetPort });
  }

  private async openIncoming(streamId: string, targetPort: number): Promise<void> {
    if (
      this.sessions.has(streamId) ||
      this.sessions.size >= MAX_STREAMS ||
      !this.allowedTargetPorts.has(targetPort)
    ) {
      this.send("runtime.stream.error", {
        streamId,
        message: this.sessions.has(streamId)
          ? "runtime_stream_is_duplicate"
          : this.sessions.size >= MAX_STREAMS
            ? "runtime_stream_capacity_exceeded"
            : `runtime_stream_target_is_not_authorized:${targetPort}`,
      });
      return;
    }
    const socket = createConnection({ host: LOOPBACK, port: targetPort });
    socket.setNoDelay(true);
    socket.setTimeout(CONNECT_TIMEOUT_MS);
    const session = this.attachSocket(streamId, socket, true);
    await new Promise<void>((resolve) => {
      const fail = (error: Error) => {
        if (!this.sessions.has(streamId)) return resolve();
        this.send("runtime.stream.error", { streamId, message: error.message.slice(0, 1_024) });
        this.closeSession(streamId, false, error);
        resolve();
      };
      socket.once("connect", () => {
        socket.setTimeout(0);
        session.opened = true;
        this.send("runtime.stream.opened", { streamId });
        resolve();
      });
      socket.once("timeout", () => fail(new Error(`runtime_stream_connect_timeout:${targetPort}`)));
      socket.once("error", fail);
    });
  }

  private attachSocket(streamId: string, socket: Socket, opened: boolean): StreamSession {
    socket.setNoDelay(true);
    const session: StreamSession = {
      streamId,
      socket,
      opened,
      sendSequence: 0,
      receiveSequence: 0,
    };
    this.sessions.set(streamId, session);
    socket.on("data", (chunk: Buffer) => {
      if (!session.opened) return;
      for (let offset = 0; offset < chunk.byteLength; offset += MAX_RUNTIME_STREAM_CHUNK_BYTES) {
        const piece = chunk.subarray(
          offset,
          Math.min(chunk.byteLength, offset + MAX_RUNTIME_STREAM_CHUNK_BYTES),
        );
        this.send("runtime.stream.data", {
          streamId,
          sequence: session.sendSequence++,
          data: piece.toString("base64"),
        });
      }
    });
    socket.once("end", () => this.closeSession(streamId, true));
    socket.once("close", () => this.closeSession(streamId, true));
    socket.once("error", (error) => this.closeSession(streamId, true, error));
    return session;
  }

  private receiveData(payload: { streamId: string; sequence: number; data: string }): void {
    const session = this.sessions.get(payload.streamId);
    if (!session || !session.opened) return;
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
    session.socket.write(Buffer.from(payload.data, "base64"));
  }

  private closeSession(streamId: string, notify: boolean, error?: Error): void {
    const session = this.sessions.get(streamId);
    if (!session || !this.sessions.delete(streamId)) return;
    if (notify) {
      this.send(error ? "runtime.stream.error" : "runtime.stream.end", error
        ? { streamId, message: error.message.slice(0, 1_024) }
        : { streamId });
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
