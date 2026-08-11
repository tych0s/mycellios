import { connect, createServer, type AddressInfo, type Socket } from "node:net";
import { afterEach, describe, expect, it, vi } from "vitest";
import { WorkerHub } from "../src/coordinator/worker-hub.js";
import type { WorkerEnvelope } from "../src/contracts/types.js";
import type { PythonPipelineLaunchDescription } from "../src/distribution/python-launcher.js";
import type { MeshStore } from "../src/storage/store.js";
import {
  RuntimeStreamTunnel,
  type RuntimeStreamServerMessage,
} from "../src/worker/runtime-stream-tunnel.js";

const cleanup: Array<() => Promise<void>> = [];

afterEach(async () => {
  await Promise.all(cleanup.splice(0).map((close) => close()));
  vi.restoreAllMocks();
});

describe("coordinator-negotiated native direct runtime transport", () => {
  it("carries a real loopback TCP stream directly and exposes secret-free telemetry", async () => {
    const fixture = await createFixture();
    const client = await fixture.openClient();
    const response = readOnce(client);
    client.write(Buffer.from("native-direct"));
    expect((await response).toString()).toBe("native-direct");
    await waitUntil(() =>
      fixture.hub.runtimeTransportSnapshot().some((item) =>
        item.mode === "direct" && item.state === "active"
      )
    );
    const snapshot = fixture.hub.runtimeTransportSnapshot().find((item) =>
      item.mode === "direct"
    )!;
    expect(snapshot).toMatchObject({
      sourceNodeId: "root-node",
      destinationNodeId: "stage-node",
      mode: "direct",
    });
    expect(snapshot.bytesSourceToDestination).toBeGreaterThan(0);
    expect(JSON.stringify(snapshot)).not.toMatch(/secret|grant/i);
    expect(fixture.sent.some((message) => message.type === "runtime.stream.data")).toBe(false);
    client.end();
    await waitUntil(() => fixture.hub.runtimeLinkObservations().some((item) =>
      item.fromNodeId === "root-node"
      && item.toNodeId === "stage-node"
      && item.transportMode === "direct"
    ));
    expect(fixture.samples).toContainEqual(expect.objectContaining({
      fromNodeId: "root-node",
      toNodeId: "stage-node",
      transportMode: "direct",
    }));
  });

  it("attributes an authenticated abnormal direct close without exposing the grant", async () => {
    const fixture = await createFixture();
    const client = await fixture.openClient();
    const offer = fixture.sent.find((message) => message.type === "runtime.direct.offer")!;
    const grant = offer.payload.grant as { connectionId: string };
    fixture.deliverWorker("worker-root", "runtime.direct.closed", {
      streamId: offer.payload.streamId,
      connectionId: grant.connectionId,
      bytesTx: 7,
      bytesRx: 3,
      reason: "authenticated_direct_channel_reset",
    });
    await waitUntil(() => fixture.hub.runtimeLinkFailureEvidence().some((item) =>
      item.reason === "authenticated_direct_channel_reset"
    ));
    const evidence = fixture.hub.runtimeLinkFailureEvidence().find((item) =>
      item.reason === "authenticated_direct_channel_reset"
    )!;
    expect(evidence).toMatchObject({
      role: "direct",
      failureClass: "direct-link-lost",
      sourceNodeId: "root-node",
      destinationNodeId: "stage-node",
      sourceOffset: 7,
      destinationOffset: 3,
    });
    expect(evidence).not.toHaveProperty("connectionId");
    expect(evidence).not.toHaveProperty("recoveryToken");
    client.destroy();
  });

  it("records coordinator loss only for relay-bound sessions", async () => {
    const fixture = await createFixture({ advertiseUnreachableCandidate: true });
    const client = await fixture.openClient();
    fixture.left.transportDisconnected();
    const evidence = fixture.left.runtimeLinkFailureEvidence()[0]!;
    expect(evidence).toMatchObject({
      role: "coordinator",
      failureClass: "coordinator-lost",
      transportMode: "relay",
      checkpointKind: "stream-offset",
      sourceNodeId: "root-node",
      destinationNodeId: "stage-node",
    });
    expect(fixture.right.runtimeLinkFailureEvidence()).toEqual([]);
    client.destroy();
  });

  it("waits for the destination commit acknowledgement before releasing the source", async () => {
    const fixture = await createFixture({ delayDestinationCommitMs: 40 });
    const client = await fixture.openClient();
    const response = readOnce(client);
    client.write(Buffer.from("ordered-commit"));
    expect((await response).toString()).toBe("ordered-commit");

    const destinationCommit = fixture.sent.findIndex((message) =>
      message.workerId === "worker-stage" && message.type === "runtime.direct.commit"
    );
    const destinationAcknowledgement = fixture.sent.findIndex((message) =>
      message.workerId === "worker-stage" && message.type === "runtime.direct.committed"
    );
    const sourceCommit = fixture.sent.findIndex((message) =>
      message.workerId === "worker-root" && message.type === "runtime.direct.commit"
    );
    expect(destinationCommit).toBeGreaterThanOrEqual(0);
    expect(destinationAcknowledgement).toBeGreaterThan(destinationCommit);
    expect(sourceCommit).toBeGreaterThan(destinationAcknowledgement);
  });

  it("falls back to offset-ACK relay when every advertised candidate is unreachable", async () => {
    const fixture = await createFixture({ advertiseUnreachableCandidate: true });
    const client = await fixture.openClient();
    const response = readOnce(client);
    client.write(Buffer.from("honest-relay-fallback"));
    expect((await response).toString()).toBe("honest-relay-fallback");
    await waitUntil(() =>
      fixture.hub.runtimeTransportSnapshot().some((item) =>
        item.mode === "relay" && item.state === "active"
      )
    );
    expect(fixture.sent.some((message) => message.type === "runtime.direct.fallback")).toBe(true);
    expect(fixture.sent.some((message) => message.type === "runtime.stream.data")).toBe(true);
    expect(fixture.samples).toContainEqual(expect.objectContaining({
      fromNodeId: "root-node",
      toNodeId: "stage-node",
      transportMode: "direct",
      rttMs: null,
      goodputMbps: null,
    }));
  });

  it("carries the rewritten tail-to-root return stream directly exactly once", async () => {
    const fixture = await createFixture();
    const tail = await fixture.openTailReturn();
    const responses: string[] = [];
    tail.on("data", (data: Buffer) => responses.push(data.toString()));
    tail.write(Buffer.from("tail-token-once"));
    await waitUntil(() => responses.length === 1);
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(responses).toEqual(["tail-token-once"]);
    expect(fixture.returnInputs).toEqual(["tail-token-once"]);
    expect(fixture.hub.runtimeTransportSnapshot()).toContainEqual(
      expect.objectContaining({
        sourceNodeId: "stage-node",
        destinationNodeId: "root-node",
        mode: "direct",
        state: "active",
      }),
    );
    expect(fixture.sent.some((message) => message.type === "runtime.stream.data")).toBe(false);
  });

  it("falls the tail return back to relay before bytes and still delivers once", async () => {
    const fixture = await createFixture({ advertiseUnreachableRootCandidate: true });
    const tail = await fixture.openTailReturn();
    const responses: string[] = [];
    tail.on("data", (data: Buffer) => responses.push(data.toString()));
    tail.write(Buffer.from("tail-relay-once"));
    await waitUntil(() => responses.length === 1);
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(responses).toEqual(["tail-relay-once"]);
    expect(fixture.returnInputs).toEqual(["tail-relay-once"]);
    expect(fixture.hub.runtimeTransportSnapshot()).toContainEqual(
      expect.objectContaining({
        sourceNodeId: "stage-node",
        destinationNodeId: "root-node",
        mode: "relay",
        state: "active",
      }),
    );
    expect(fixture.sent.some((message) => message.type === "runtime.direct.fallback")).toBe(true);
    expect(fixture.sent.some((message) => message.type === "runtime.stream.data")).toBe(true);
  });

  it("keeps legacy direct peers on the race-free relay path", async () => {
    const fixture = await createFixture({ legacyDestinationCommit: true });
    const client = await fixture.openClient();
    const response = readOnce(client);
    client.write(Buffer.from("legacy-relay"));
    expect((await response).toString()).toBe("legacy-relay");
    expect(fixture.sent.some((message) => message.type === "runtime.direct.offer")).toBe(false);
    expect(fixture.sent.some((message) => message.type === "runtime.stream.data")).toBe(true);
  });

  it("tries the next coordinator-advertised candidate before choosing relay", async () => {
    const fixture = await createFixture({ prependUnreachableCandidate: true });
    const client = await fixture.openClient();
    const response = readOnce(client);
    client.write(Buffer.from("second-candidate"));
    expect((await response).toString()).toBe("second-candidate");
    expect(fixture.hub.runtimeTransportSnapshot()).toContainEqual(
      expect.objectContaining({ mode: "direct", state: "active" }),
    );
    expect(fixture.sent.some((message) => message.type === "runtime.stream.data")).toBe(false);
  });

  it("forbids a relay downgrade after the direct route has moved bytes", async () => {
    const fixture = await createFixture();
    const client = await fixture.openClient();
    client.on("error", () => undefined);
    const response = readOnce(client);
    client.write(Buffer.from("committed-byte"));
    await response;
    const active = fixture.hub.runtimeTransportSnapshot().find((item) =>
      item.mode === "direct" && item.state === "active"
    )!;
    const relayOpensBefore = fixture.sent.filter((item) =>
      item.type === "runtime.stream.open"
    ).length;
    fixture.deliverWorker("worker-root", "runtime.direct.fallback", {
      streamId: active.streamId,
      connectionId: active.streamId,
      reason: "late_candidate_failure",
    });
    await waitUntil(() =>
      fixture.hub.runtimeTransportSnapshot().some((item) =>
        item.streamId === active.streamId && item.state === "closed"
      )
    );
    expect(fixture.sent.filter((item) => item.type === "runtime.stream.open")).toHaveLength(
      relayOpensBefore,
    );
  });

  it("closes listeners, grants and active channels without leaving sessions", async () => {
    const fixture = await createFixture();
    const client = await fixture.openClient();
    await fixture.left.close();
    await fixture.right.close();
    await waitUntil(() => client.destroyed);
    expect(fixture.left.transportSnapshot().some((item) => item.state === "active")).toBe(false);
    expect(fixture.right.transportSnapshot().some((item) => item.state === "active")).toBe(false);
  });
});

async function createFixture(
  options: {
    advertiseUnreachableCandidate?: boolean;
    prependUnreachableCandidate?: boolean;
    delayDestinationCommitMs?: number;
    legacyDestinationCommit?: boolean;
    advertiseUnreachableRootCandidate?: boolean;
  } = {},
) {
  const echo = createServer((socket) => socket.pipe(socket));
  await listen(echo);
  cleanup.push(() => closeServer(echo));
  const targetPort = (echo.address() as AddressInfo).port;
  const returnInputs: string[] = [];
  const returnEcho = createServer((socket) => {
    socket.on("data", (data: Buffer) => returnInputs.push(data.toString()));
    socket.pipe(socket);
  });
  await listen(returnEcho);
  cleanup.push(() => closeServer(returnEcho));
  const returnPort = (returnEcho.address() as AddressInfo).port;
  const description = launchDescription(targetPort, returnPort);
  const sent: Array<{ workerId: string; type: string; payload: Record<string, unknown> }> = [];
  let hub!: WorkerHub;
  let left!: RuntimeStreamTunnel;
  let right!: RuntimeStreamTunnel;
  const deliverWorker = (
    workerId: string,
    type: string,
    payload: Record<string, unknown>,
  ) => {
    sent.push({ workerId, type, payload });
    const internal = hub as unknown as {
      handleRuntimeStreamEnvelope(envelope: WorkerEnvelope): void;
      handleRuntimeDirectEnvelope(envelope: WorkerEnvelope): void;
    };
    queueMicrotask(() => {
      const envelope = { v: 1, workerId, type, payload } as WorkerEnvelope;
      if (type.startsWith("runtime.direct.")) {
        internal.handleRuntimeDirectEnvelope(envelope);
      } else {
        internal.handleRuntimeStreamEnvelope(envelope);
      }
    });
    return true;
  };
  left = new RuntimeStreamTunnel(
    "root-node",
    (type, payload) => deliverWorker(
      "worker-root",
      type,
      payload as Record<string, unknown>,
    ),
    {
      directTransport: {
        listenHost: "127.0.0.1",
        candidateHosts: ["127.0.0.1"],
        connectTimeoutMs: 500,
      },
    },
  );
  right = new RuntimeStreamTunnel(
    "stage-node",
    (type, payload) => deliverWorker(
      "worker-stage",
      type,
      payload as Record<string, unknown>,
    ),
    {
      directTransport: {
        listenHost: "127.0.0.1",
        candidateHosts: ["127.0.0.1"],
        connectTimeoutMs: 500,
      },
    },
  );
  cleanup.push(() => left.close(), () => right.close());
  await Promise.all([left.prepare(description), right.prepare(description)]);
  const [leftDirect, rightDirect] = await Promise.all([
    left.startDirectTransport(),
    right.startDirectTransport(),
  ]);
  if (!leftDirect || !rightDirect) throw new Error("direct_listener_not_started");
  const deadCandidate = {
    host: "127.0.0.1",
    port: await unusedPort(),
    scope: "configured" as const,
  };
  const advertisedRight = options.legacyDestinationCommit
    ? (() => {
        const { commitAck: _commitAck, ...legacy } = rightDirect;
        return legacy as typeof rightDirect;
      })()
    : options.advertiseUnreachableCandidate
    ? { ...rightDirect, candidates: [deadCandidate] }
    : options.prependUnreachableCandidate
      ? { ...rightDirect, candidates: [deadCandidate, ...rightDirect.candidates] }
      : rightDirect;
  const advertisedLeft = options.advertiseUnreachableRootCandidate
    ? { ...leftDirect, candidates: [deadCandidate] }
    : leftDirect;
  const workers = [
    worker("worker-root", "root-node", advertisedLeft),
    worker("worker-stage", "stage-node", advertisedRight),
  ];
  const samples: Array<import("../src/storage/store.js").StoredRuntimeLinkSample> = [];
  const store = {
    getWorker: (workerId: string) => workers.find((worker) => worker.id === workerId),
    listWorkers: () => workers,
    listRuntimeLinkSamples: () => [],
    saveRuntimeLinkSample: (sample: import("../src/storage/store.js").StoredRuntimeLinkSample) => {
      samples.push(structuredClone(sample));
    },
  } as unknown as MeshStore;
  hub = new WorkerHub(store);
  vi.spyOn(hub, "isConnected").mockReturnValue(true);
  vi.spyOn(hub, "send").mockImplementation((workerId, type, payload) => {
    sent.push({ workerId, type, payload: payload as Record<string, unknown> });
    const tunnel = workerId === "worker-root" ? left : right;
    const deliver = () => {
      void tunnel.handle({ type, payload } as RuntimeStreamServerMessage);
    };
    if (
      workerId === "worker-stage"
      && type === "runtime.direct.commit"
      && options.delayDestinationCommitMs
    ) {
      setTimeout(deliver, options.delayDestinationCommitMs);
    } else {
      queueMicrotask(deliver);
    }
    return true;
  });
  const root = description.launchOrder.find((process) => process.kind === "root-engine")!;
  const rewritten = left.rewriteProcess(root);
  const proxyPort = Number(flag(rewritten.command.args, "--first-stage-port"));
  const stage = description.launchOrder.find((process) => process.kind === "remote-stage")!;
  const rewrittenStage = right.rewriteProcess(stage);
  const returnProxyPort = Number(flag(rewrittenStage.command.args, "--return-port"));
  const sockets: Socket[] = [];
  return {
    hub,
    left,
    right,
    sent,
    samples,
    returnInputs,
    deliverWorker,
    async openClient() {
      const socket = connect({ host: "127.0.0.1", port: proxyPort });
      socket.on("error", () => undefined);
      sockets.push(socket);
      cleanup.push(() => closeSocket(socket));
      await connected(socket);
      await waitUntil(() =>
        hub.runtimeTransportSnapshot().some((item) => item.state === "active")
      );
      return socket;
    },
    async openTailReturn() {
      const socket = connect({ host: "127.0.0.1", port: returnProxyPort });
      socket.on("error", () => undefined);
      sockets.push(socket);
      cleanup.push(() => closeSocket(socket));
      await connected(socket);
      await waitUntil(() =>
        hub.runtimeTransportSnapshot().some((item) =>
          item.sourceNodeId === "stage-node"
          && item.destinationNodeId === "root-node"
          && item.state === "active"
        )
      );
      return socket;
    },
  };
}

function worker(
  id: string,
  nodeId: string,
  directTransport: NonNullable<
    ReturnType<RuntimeStreamTunnel["startDirectTransport"]> extends Promise<infer T> ? T : never
  >,
) {
  return {
    id,
    capabilities: {
      distributedExecutor: {
        protocol: "gdlp-worker-tunnel/2",
        streamRecovery: "offset-ack-v1",
        nodeId,
        directTransport,
      },
    },
  };
}

function launchDescription(
  stagePort: number,
  returnPort: number,
): PythonPipelineLaunchDescription {
  return {
    launchOrder: [
      {
        kind: "remote-stage",
        processId: "stage-process",
        anchor: { memberId: "stage-node", endpoint: { host: "10.0.0.20", port: stagePort } },
        downstream: null,
        returnEndpoint: { host: "10.0.0.10", port: returnPort },
        command: {
          executable: "python",
          args: [
            "-m", "distributed_runtime.stage_cli",
            "--listen-host", "10.0.0.20",
            "--listen-port", String(stagePort),
            "--return-host", "10.0.0.10",
            "--return-port", String(returnPort),
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
        returnEndpoint: { host: "10.0.0.10", port: returnPort },
        command: {
          executable: "python",
          args: [
            "-m", "distributed_runtime.server",
            "--host", "0.0.0.0",
            "--first-stage-host", "10.0.0.20",
            "--first-stage-port", String(stagePort),
            "--return-bind-host", "0.0.0.0",
            "--return-advertise-host", "10.0.0.10",
            "--return-port", String(returnPort),
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
    server.listen(0, "127.0.0.1", resolve);
  });
}

function closeServer(server: ReturnType<typeof createServer>): Promise<void> {
  if (!server.listening) return Promise.resolve();
  return new Promise((resolve) => server.close(() => resolve()));
}

function closeSocket(socket: Socket): Promise<void> {
  if (socket.destroyed) return Promise.resolve();
  return new Promise((resolve) => {
    socket.once("close", resolve);
    socket.destroy();
  });
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

async function unusedPort(): Promise<number> {
  const server = createServer();
  await listen(server);
  const port = (server.address() as AddressInfo).port;
  await closeServer(server);
  return port;
}

async function waitUntil(check: () => boolean, timeoutMs = 4_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!check()) {
    if (Date.now() >= deadline) throw new Error("condition_not_met");
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}
