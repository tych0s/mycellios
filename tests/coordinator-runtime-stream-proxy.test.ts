import { connect, type Socket } from "node:net";
import { afterEach, describe, expect, it, vi } from "vitest";
import { WorkerHub } from "../src/coordinator/worker-hub.js";
import type { WorkerEnvelope } from "../src/contracts/types.js";
import type { MeshStore } from "../src/storage/store.js";

const cleanup: Array<() => Promise<void>> = [];

afterEach(async () => {
  await Promise.all(cleanup.splice(0).map((close) => close()));
  vi.restoreAllMocks();
});

describe("WorkerHub runtime proxy", () => {
  it("bridges a coordinator-local TCP client through one connected worker", async () => {
    const hub = new WorkerHub({} as MeshStore);
    const sent: Array<{ workerId: string; type: string; payload: Record<string, unknown> }> = [];
    vi.spyOn(hub, "isConnected").mockReturnValue(true);
    vi.spyOn(hub, "send").mockImplementation((workerId, type, payload) => {
      sent.push({ workerId, type, payload: payload as Record<string, unknown> });
      return true;
    });
    const proxy = await hub.createRuntimeProxy("worker-root", 9_860);
    cleanup.push(() => proxy.close());
    const client = connect({ host: proxy.host, port: proxy.port });
    cleanup.push(() => closeSocket(client));
    await connected(client);
    await waitUntil(() => sent.some((message) => message.type === "runtime.stream.open"));
    const opened = sent.find((message) => message.type === "runtime.stream.open")!;
    expect(opened.workerId).toBe("worker-root");
    expect(opened.payload.targetPort).toBe(9_860);
    const streamId = opened.payload.streamId as string;

    deliver(hub, "worker-root", "runtime.stream.opened", { streamId });
    client.write(Buffer.from("health request"));
    await waitUntil(() => sent.some((message) => message.type === "runtime.stream.data"));
    const outbound = sent.find((message) => message.type === "runtime.stream.data")!;
    expect(Buffer.from(outbound.payload.data as string, "base64").toString("utf8")).toBe("health request");

    const response = readOnce(client);
    deliver(hub, "worker-root", "runtime.stream.data", {
      streamId,
      sequence: 0,
      data: Buffer.from("HTTP/1.1 200 OK\r\n\r\nready").toString("base64"),
    });
    expect((await response).toString("utf8")).toContain("ready");
  });

  it("closes active client sockets instead of blocking route deactivation", async () => {
    const hub = new WorkerHub({} as MeshStore);
    vi.spyOn(hub, "isConnected").mockReturnValue(true);
    vi.spyOn(hub, "send").mockReturnValue(true);
    const proxy = await hub.createRuntimeProxy("worker-root", 9_860);
    const client = connect({ host: proxy.host, port: proxy.port });
    cleanup.push(() => closeSocket(client));
    await connected(client);

    const clientClosed = new Promise<void>((resolve) => client.once("close", () => resolve()));
    await expect(Promise.race([
      proxy.close().then(() => "closed"),
      new Promise<string>((resolve) => setTimeout(() => resolve("timeout"), 500)),
    ])).resolves.toBe("closed");
    await expect(clientClosed).resolves.toBeUndefined();
  });
});

function deliver(
  hub: WorkerHub,
  workerId: string,
  type: string,
  payload: Record<string, unknown>,
): void {
  const internal = hub as unknown as {
    handleRuntimeStreamEnvelope(envelope: WorkerEnvelope): void;
  };
  internal.handleRuntimeStreamEnvelope({ v: 1, workerId, type, payload });
}

function connected(socket: Socket): Promise<void> {
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

function closeSocket(socket: Socket): Promise<void> {
  if (socket.destroyed) return Promise.resolve();
  return new Promise((resolve) => {
    socket.once("close", () => resolve());
    socket.destroy();
  });
}

async function waitUntil(check: () => boolean, timeoutMs = 2_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!check()) {
    if (Date.now() >= deadline) throw new Error("condition_not_met");
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}
