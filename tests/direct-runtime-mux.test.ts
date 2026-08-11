import { once } from "node:events";
import { describe, expect, it } from "vitest";
import {
  createDirectSessionGrant,
  DirectSecureChannel,
  DirectSecureServer,
} from "../src/transport/direct-secure-channel.js";
import {
  DirectRuntimeMux,
  type DirectRuntimeStream,
} from "../src/transport/direct-runtime-mux.js";

describe("direct runtime stream multiplexer", () => {
  it("carries several bidirectional streams with exact offsets", async () => {
    const pair = await muxPair();
    const incoming: DirectRuntimeStream[] = [];
    pair.serverMux.on("stream", (stream) => incoming.push(stream));
    const alpha = await pair.clientMux.openStream("alpha", Buffer.from("stage-a"));
    const beta = await pair.clientMux.openStream("beta", Buffer.from("stage-b"));
    await waitUntil(() => incoming.length === 2);
    const remoteAlpha = incoming.find((stream) => stream.id === "alpha")!;
    const remoteBeta = incoming.find((stream) => stream.id === "beta")!;
    expect(remoteAlpha.metadata.toString()).toBe("stage-a");
    expect(remoteBeta.metadata.toString()).toBe("stage-b");
    const alphaValues: string[] = [];
    const betaValues: string[] = [];
    remoteAlpha.setDataHandler((data, offset) => {
      expect(offset).toBe(alphaValues.join("").length);
      alphaValues.push(data.toString());
    });
    remoteBeta.setDataHandler((data) => {
      betaValues.push(data.toString());
    });
    const reverse: string[] = [];
    alpha.setDataHandler((data) => {
      reverse.push(data.toString());
    });
    beta.setDataHandler(() => undefined);

    await Promise.all([
      alpha.write(Buffer.from("hello")),
      beta.write(Buffer.from("parallel")),
      alpha.write(Buffer.from("-world")),
    ]);
    await waitUntil(() => alphaValues.join("") === "hello-world");
    expect(betaValues).toEqual(["parallel"]);

    await remoteAlpha.write(Buffer.from("return"));
    await waitUntil(() => reverse.length === 1);
    expect(reverse).toEqual(["return"]);
    expect(remoteAlpha.snapshot()).toMatchObject({
      sendOffset: 6,
      receiveOffset: 11,
    });
    await pair.close();
  });

  it("stops only the slow stream at its credit boundary", async () => {
    const pair = await muxPair({ receiveWindowBytes: 1_024, maximumChunkBytes: 512 });
    const incoming: DirectRuntimeStream[] = [];
    pair.serverMux.on("stream", (stream) => incoming.push(stream));
    const slow = await pair.clientMux.openStream("slow");
    const fast = await pair.clientMux.openStream("fast");
    await waitUntil(() => incoming.length === 2);
    const remoteSlow = incoming.find((stream) => stream.id === "slow")!;
    const remoteFast = incoming.find((stream) => stream.id === "fast")!;
    const fastValues: Buffer[] = [];
    remoteFast.setDataHandler((data) => {
      fastValues.push(data);
    });

    let slowFinished = false;
    const slowWrite = slow.write(Buffer.alloc(4_096, 7)).then(() => {
      slowFinished = true;
    });
    await fast.write(Buffer.from("control-path-still-live"));
    await waitUntil(() => fastValues.length === 1);
    expect(slowFinished).toBe(false);
    expect(remoteSlow.snapshot().receiveBufferedBytes).toBe(1_024);

    let slowBytes = 0;
    remoteSlow.setDataHandler(async (data) => {
      slowBytes += data.byteLength;
      await new Promise((resolve) => setTimeout(resolve, 1));
    });
    await slowWrite;
    await waitUntil(() => slowBytes === 4_096);
    expect(slowFinished).toBe(true);
    await pair.close();
  });

  it("propagates reset to one stream without killing its siblings", async () => {
    const pair = await muxPair();
    const incoming: DirectRuntimeStream[] = [];
    pair.serverMux.on("stream", (stream) => incoming.push(stream));
    const rejected = await pair.clientMux.openStream("rejected");
    const healthy = await pair.clientMux.openStream("healthy");
    await waitUntil(() => incoming.length === 2);
    const remoteRejected = incoming.find((stream) => stream.id === "rejected")!;
    const remoteHealthy = incoming.find((stream) => stream.id === "healthy")!;
    const reset = once(rejected, "reset");
    await remoteRejected.reset("stage_not_authorized");
    expect(((await reset)[0] as Error).message).toBe("stage_not_authorized");

    const values: string[] = [];
    remoteHealthy.setDataHandler((data) => {
      values.push(data.toString());
    });
    await healthy.write(Buffer.from("still-running"));
    await waitUntil(() => values.length === 1);
    expect(values).toEqual(["still-running"]);
    await pair.close();
  });

  it("bounds queued sender memory without blocking control or sibling streams", async () => {
    const pair = await muxPair({
      receiveWindowBytes: 1_024,
      maximumChunkBytes: 512,
      maximumPendingWriteBytes: 1_024,
    });
    const incoming: DirectRuntimeStream[] = [];
    pair.serverMux.on("stream", (stream) => incoming.push(stream));
    const bulk = await pair.clientMux.openStream("bulk");
    const control = await pair.clientMux.openStream("control");
    await waitUntil(() => incoming.length === 2);
    await bulk.write(Buffer.alloc(1_024));
    expect(bulk.snapshot()).toMatchObject({ sendOffset: 1_024, pendingWriteBytes: 0 });
    await expect(bulk.write(Buffer.alloc(1_025))).rejects.toThrow(
      "direct_mux_pending_write_capacity_exceeded",
    );
    expect(bulk.snapshot()).toMatchObject({ sendOffset: 1_024, pendingWriteBytes: 0 });

    const controlValues: string[] = [];
    incoming.find((stream) => stream.id === "control")!.setDataHandler((data) => {
      controlValues.push(data.toString());
    });
    await control.write(Buffer.from("cancel-reset-health"));
    await waitUntil(() => controlValues.length === 1);
    expect(controlValues).toEqual(["cancel-reset-health"]);
    await pair.close();
  });

  it("preserves control latency and an RSS ceiling under sustained bulk pressure", async () => {
    const pair = await muxPair({
      receiveWindowBytes: 4 * 1_024,
      maximumChunkBytes: 1_024,
      maximumPendingWriteBytes: 64 * 1_024,
    });
    const incoming: DirectRuntimeStream[] = [];
    pair.serverMux.on("stream", (stream) => incoming.push(stream));
    const bulk = await Promise.all(
      ["bulk-a", "bulk-b", "bulk-c"].map((id) => pair.clientMux.openStream(id)),
    );
    const control = await pair.clientMux.openStream("health-control");
    await waitUntil(() => incoming.length === 4);
    for (const stream of incoming.filter((candidate) => candidate.id.startsWith("bulk-"))) {
      stream.setDataHandler(async () => {
        await new Promise((resolve) => setTimeout(resolve, 1));
      });
    }
    let observedControls = 0;
    incoming.find((stream) => stream.id === "health-control")!.setDataHandler(() => {
      observedControls += 1;
    });

    const rssBefore = process.memoryUsage.rss();
    const bulkRuns = bulk.map(async (stream, streamIndex) => {
      for (let batch = 0; batch < 8; batch += 1) {
        await Promise.all(Array.from({ length: 64 }, (_, chunkIndex) =>
          stream.write(Buffer.alloc(1_024, streamIndex + chunkIndex + batch))
        ));
      }
    });
    const controlLatenciesMs: number[] = [];
    for (let index = 0; index < 24; index += 1) {
      const started = performance.now();
      await control.write(Buffer.from(`health-${index}`));
      await waitUntil(() => observedControls === index + 1);
      controlLatenciesMs.push(performance.now() - started);
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    await Promise.all(bulkRuns);
    const rssDelta = Math.max(0, process.memoryUsage.rss() - rssBefore);
    const sortedLatency = [...controlLatenciesMs].sort((left, right) => left - right);
    const p95 = sortedLatency[Math.ceil(sortedLatency.length * 0.95) - 1]!;
    expect(p95).toBeLessThan(100);
    expect(rssDelta).toBeLessThan(64 * 1_024 * 1_024);
    for (const stream of bulk) expect(stream.snapshot().pendingWriteBytes).toBe(0);
    await pair.close();
  });
});

async function muxPair(
  options: {
    receiveWindowBytes?: number;
    maximumChunkBytes?: number;
    maximumPendingWriteBytes?: number;
  } = {},
) {
  const grant = createDirectSessionGrant({
    connectionId: `mux-${Math.random().toString(16).slice(2)}`,
    sourceNodeId: "node-a",
    destinationNodeId: "node-b",
    targetPort: 9_850,
    expiresAt: Date.now() + 60_000,
  });
  let serverChannel: DirectSecureChannel | null = null;
  const server = new DirectSecureServer({
    resolveGrant: () => grant,
    onChannel: (channel) => {
      serverChannel = channel;
    },
  });
  const address = await server.listen();
  const clientChannel = await DirectSecureChannel.connect({
    host: address.host,
    port: address.port,
    grant,
  });
  await waitUntil(() => serverChannel !== null);
  const clientMux = new DirectRuntimeMux(clientChannel, options);
  const serverMux = new DirectRuntimeMux(serverChannel!, options);
  return {
    clientMux,
    serverMux,
    async close() {
      await clientMux.close();
      await serverMux.close().catch(() => undefined);
      await server.close();
    },
  };
}

async function waitUntil(predicate: () => boolean): Promise<void> {
  const deadline = Date.now() + 3_000;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error("condition was not reached");
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}
