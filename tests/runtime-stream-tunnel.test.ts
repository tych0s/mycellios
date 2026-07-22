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
        ? { type, payload: { streamId: value.streamId, targetPort: value.targetPort } }
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
    const tunnel = new RuntimeStreamTunnel("root-node", (type, payload) => sent.push({ type, payload }));
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
