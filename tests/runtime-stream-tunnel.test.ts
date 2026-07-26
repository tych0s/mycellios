import { createServer, connect, type AddressInfo, type Socket } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import type { PythonPipelineLaunchDescription } from "../src/distribution/python-launcher.js";
import {
  RuntimeStreamTunnel,
  type RuntimeStreamServerMessage,
} from "../src/worker/runtime-stream-tunnel.js";

const cleanup: Array<() => Promise<void>> = [];

afterEach(async () => {
  await Promise.all(cleanup.splice(0).map((close) => close()));
});

describe("RuntimeStreamTunnel", () => {
  it("carries a TCP stream between two private workers without using either advertised host", async () => {
    const echo = createServer((socket) => socket.pipe(socket));
    await listen(echo);
    cleanup.push(() => closeServer(echo));
    const targetPort = (echo.address() as AddressInfo).port;
    const description = launchDescription(targetPort);

    let left!: RuntimeStreamTunnel;
    let right!: RuntimeStreamTunnel;
    const relay = (
      sender: "root-node" | "stage-node",
      type: string,
      payload: unknown,
    ) => {
      const target = sender === "root-node" ? right : left;
      const value = payload as Record<string, unknown>;
      const message = type === "runtime.stream.open"
        ? {
            type,
            payload: {
              streamId: value.streamId,
              targetPort: value.targetPort,
              ...(typeof value.generation === "number"
                && typeof value.recoveryToken === "string"
                ? {
                    generation: value.generation,
                    recoveryToken: value.recoveryToken,
                  }
                : {}),
            },
          }
        : { type, payload };
      queueMicrotask(() => void target.handle(message as RuntimeStreamServerMessage));
    };
    left = new RuntimeStreamTunnel("root-node", (type, payload) => relay("root-node", type, payload));
    right = new RuntimeStreamTunnel("stage-node", (type, payload) => relay("stage-node", type, payload));
    cleanup.push(() => left.close(), () => right.close());
    await Promise.all([left.prepare(description), right.prepare(description)]);

    const root = description.launchOrder.find((process) => process.kind === "root-engine")!;
    const rewritten = left.rewriteProcess(root);
    expect(flag(rewritten.command.args, "--host")).toBe("127.0.0.1");
    expect(flag(rewritten.command.args, "--first-stage-host")).toBe("127.0.0.1");
    expect(flag(rewritten.command.args, "--first-stage-port")).not.toBe(String(targetPort));
    expect(flag(rewritten.command.args, "--return-bind-host")).toBe("127.0.0.1");

    const socket = connect({ host: "127.0.0.1", port: Number(flag(rewritten.command.args, "--first-stage-port")) });
    cleanup.push(() => closeSocket(socket));
    await onceConnected(socket);
    const received = readOnce(socket);
    socket.write(Buffer.from("mycelium-over-wss"));
    expect((await received).toString("utf8")).toBe("mycelium-over-wss");
  });

  it("rejects coordinator stream requests for ports outside the prepared runtime", async () => {
    const sent: Array<{ type: string; payload: unknown }> = [];
    const tunnel = new RuntimeStreamTunnel("root-node", (type, payload) => {
      sent.push({ type, payload });
    });
    cleanup.push(() => tunnel.close());
    await tunnel.prepare(launchDescription(19_850));
    await tunnel.handle({
      type: "runtime.stream.open",
      payload: { streamId: "unauthorized", targetPort: 22 },
    });
    expect(sent).toContainEqual({
      type: "runtime.stream.error",
      payload: { streamId: "unauthorized", message: "runtime_stream_target_is_not_authorized:22" },
    });
  });

  it("fails closed before its replay buffer can exceed the configured memory ceiling", async () => {
    const sent: Array<{ type: string; payload: Record<string, unknown> }> = [];
    const tunnel = new RuntimeStreamTunnel(
      "root-node",
      (type, payload) => {
        sent.push({ type, payload: payload as Record<string, unknown> });
        return true;
      },
      { maxReplayBytes: 96 * 1024 },
    );
    cleanup.push(() => tunnel.close());
    const description = launchDescription(19_850);
    await tunnel.prepare(description);
    const root = description.launchOrder.find((process) => process.kind === "root-engine")!;
    const rewritten = tunnel.rewriteProcess(root);
    const socket = connect({
      host: "127.0.0.1",
      port: Number(flag(rewritten.command.args, "--first-stage-port")),
    });
    socket.on("error", () => undefined);
    cleanup.push(() => closeSocket(socket));
    await onceConnected(socket);
    await waitUntil(() => sent.some((message) => message.type === "runtime.stream.open"));
    const opened = sent.find((message) => message.type === "runtime.stream.open")!;
    await tunnel.handle({
      type: "runtime.stream.opened",
      payload: {
        streamId: opened.payload.streamId as string,
        generation: opened.payload.generation as number,
        recoveryToken: opened.payload.recoveryToken as string,
      },
    });

    // Two 48 KiB chunks fill the exact ceiling. The third byte range must
    // terminate the stream; it can never become an untracked in-memory tail.
    socket.write(Buffer.alloc(96 * 1024 + 1, 7));
    await waitUntil(() => sent.some((message) =>
      message.type === "runtime.stream.error"
      && message.payload.message === "runtime_stream_replay_buffer_exceeded"
    ));
    expect(tunnel.recoverySnapshot()).toEqual([]);
  });

  it("acknowledges a duplicate idempotently and rejects a forward offset gap", async () => {
    const received: Buffer[] = [];
    const target = createServer((socket) => {
      socket.on("data", (data) => received.push(Buffer.from(data)));
    });
    await listen(target);
    cleanup.push(() => closeServer(target));
    const targetPort = (target.address() as AddressInfo).port;
    const sent: Array<{ type: string; payload: Record<string, unknown> }> = [];
    const tunnel = new RuntimeStreamTunnel("stage-node", (type, payload) => {
      sent.push({ type, payload: payload as Record<string, unknown> });
      return true;
    });
    cleanup.push(() => tunnel.close());
    await tunnel.prepare(launchDescription(targetPort));
    const recoveryToken = "recovery_token_0123456789";
    await tunnel.handle({
      type: "runtime.stream.open",
      payload: {
        streamId: "recoverable-incoming",
        targetPort,
        generation: 0,
        recoveryToken,
      },
    });
    const data = Buffer.from("one").toString("base64");
    const chunk: RuntimeStreamServerMessage = {
      type: "runtime.stream.data",
      payload: {
        streamId: "recoverable-incoming",
        generation: 0,
        recoveryToken,
        sequence: 0,
        offset: 0,
        data,
      },
    };
    await tunnel.handle(chunk);
    await waitUntil(() => Buffer.concat(received).toString("utf8") === "one");
    await tunnel.handle(chunk);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(Buffer.concat(received).toString("utf8")).toBe("one");
    expect(sent.filter((message) =>
      message.type === "runtime.stream.ack"
      && message.payload.acknowledgedOffset === 3
    )).toHaveLength(2);

    await tunnel.handle({
      type: "runtime.stream.data",
      payload: {
        ...chunk.payload,
        sequence: 1,
        offset: 7,
      },
    });
    expect(sent).toContainEqual(expect.objectContaining({
      type: "runtime.stream.error",
      payload: expect.objectContaining({
        message: "runtime_stream_offset_mismatch:3:7",
      }),
    }));
    expect(tunnel.recoverySnapshot()).toEqual([]);
  });
});

function launchDescription(stagePort: number): PythonPipelineLaunchDescription {
  return {
    launchOrder: [
      {
        kind: "remote-stage",
        processId: "stage-process",
        anchor: { memberId: "stage-node", endpoint: { host: "10.0.0.20", port: stagePort } },
        downstream: null,
        returnEndpoint: { host: "10.0.0.10", port: 30_092 },
        command: {
          executable: "python",
          args: [
            "-m", "distributed_runtime.stage_cli",
            "--listen-host", "10.0.0.20",
            "--listen-port", String(stagePort),
            "--return-host", "10.0.0.10",
            "--return-port", "30092",
          ],
        },
      },
      {
        kind: "root-engine",
        processId: "root-process",
        anchor: { memberId: "root-node", endpoint: { host: "10.0.0.10", port: 9_850 } },
        firstRemoteStage: {
          stageId: "stage-1",
          stageIndex: 1,
          layerEnd: 28,
          anchorMemberId: "stage-node",
          endpoint: { host: "10.0.0.20", port: stagePort },
        },
        apiEndpoint: { host: "0.0.0.0", port: 9_860 },
        returnEndpoint: { host: "10.0.0.10", port: 30_092 },
        command: {
          executable: "python",
          args: [
            "-m", "distributed_runtime.server",
            "--host", "0.0.0.0",
            "--first-stage-host", "10.0.0.20",
            "--first-stage-port", String(stagePort),
            "--return-bind-host", "0.0.0.0",
            "--return-advertise-host", "10.0.0.10",
            "--return-port", "30092",
          ],
        },
      },
    ],
  } as unknown as PythonPipelineLaunchDescription;
}

function flag(args: string[], name: string): string {
  const index = args.indexOf(name);
  if (index < 0 || !args[index + 1]) throw new Error(`missing ${name}`);
  return args[index + 1]!;
}

function listen(server: ReturnType<typeof createServer>): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolve());
  });
}

function closeServer(server: ReturnType<typeof createServer>): Promise<void> {
  return new Promise((resolve) => server.close(() => resolve()));
}

function closeSocket(socket: Socket): Promise<void> {
  if (socket.destroyed) return Promise.resolve();
  return new Promise((resolve) => {
    socket.once("close", () => resolve());
    socket.destroy();
  });
}

function onceConnected(socket: Socket): Promise<void> {
  if (socket.readyState === "open") return Promise.resolve();
  return new Promise((resolve, reject) => {
    socket.once("connect", resolve);
    socket.once("error", reject);
  });
}

function readOnce(socket: Socket): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    socket.once("data", resolve);
    socket.once("error", reject);
  });
}

async function waitUntil(check: () => boolean, timeoutMs = 2_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!check()) {
    if (Date.now() >= deadline) throw new Error("condition_not_met");
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}
