import { describe, expect, it, vi } from "vitest";
import type { WorkerEnvelope } from "../src/contracts/types.js";
import { WorkerHub } from "../src/coordinator/worker-hub.js";
import type { MeshStore } from "../src/storage/store.js";

const RECOVERY_TOKEN = "recovery_token_0123456789";

describe("WorkerHub recoverable runtime streams", () => {
  it("resumes from the peer-confirmed offset and ignores an idempotent duplicate", () => {
    const { hub, sent, store } = recoveryHub(true);
    openRecoverableStream(hub);
    deliver(hub, "worker-b", "runtime.stream.opened", recoveryIdentity());
    sent.splice(0);

    deliver(hub, "worker-a", "runtime.stream.data", {
      ...recoveryIdentity(),
      sequence: 0,
      offset: 0,
      data: Buffer.from("lost").toString("base64"),
    });
    expect(sent.filter((message) => message.type === "runtime.stream.data")).toHaveLength(1);

    disconnectWorker(hub, "worker-b");
    expect(store.setWorkerStatus).toHaveBeenCalledWith("worker-b", "offline");
    expect(sent).toContainEqual(expect.objectContaining({
      workerId: "worker-a",
      type: "runtime.stream.suspend",
    }));

    // Source retained four bytes; destination proves it delivered none.
    deliver(hub, "worker-a", "runtime.stream.resume", {
      ...recoveryIdentity(),
      sendOffset: 4,
      acknowledgedOffset: 0,
      receiveOffset: 0,
      bufferedFromOffset: 0,
    });
    deliver(hub, "worker-b", "runtime.stream.resume", {
      ...recoveryIdentity(),
      sendOffset: 0,
      acknowledgedOffset: 0,
      receiveOffset: 0,
      bufferedFromOffset: 0,
    });

    const resumed = sent.filter((message) => message.type === "runtime.stream.resumed");
    expect(resumed).toHaveLength(2);
    expect(resumed).toContainEqual(expect.objectContaining({
      workerId: "worker-a",
      payload: expect.objectContaining({
        previousGeneration: 0,
        generation: 1,
        sendFromOffset: 0,
      }),
    }));

    const replay = {
      streamId: "stream-recovery",
      generation: 1,
      recoveryToken: RECOVERY_TOKEN,
      sequence: 0,
      offset: 0,
      data: Buffer.from("lost").toString("base64"),
    };
    const beforeReplay = sent.filter((message) => message.type === "runtime.stream.data").length;
    deliver(hub, "worker-a", "runtime.stream.data", replay);
    deliver(hub, "worker-a", "runtime.stream.data", replay);
    // The second copy is wholly behind the contiguous offset and is ignored.
    expect(sent.filter((message) => message.type === "runtime.stream.data")).toHaveLength(
      beforeReplay + 1,
    );

    deliver(hub, "worker-a", "runtime.stream.data", {
      ...replay,
      sequence: 1,
      offset: 9,
    });
    expect(sent.some((message) =>
      message.type === "runtime.stream.error"
      && String(message.payload.message).includes("runtime_stream_offset_mismatch:4:9")
    )).toBe(true);
    expect(runtimeStreamCount(hub)).toBe(0);
    hub.close();
  });

  it("downgrades a mixed v2 route before its first byte without claiming recovery", () => {
    const legacy = recoveryHub(false);
    openRecoverableStream(legacy.hub);
    expect(legacy.sent).toContainEqual(expect.objectContaining({
      workerId: "worker-b",
      type: "runtime.stream.open",
      payload: { streamId: "stream-recovery", targetPort: 9_850 },
    }));
    expect(runtimeStreamCount(legacy.hub)).toBe(1);

    deliver(legacy.hub, "worker-a", "runtime.stream.open", {
      streamId: "legacy-stream",
      destinationNodeId: "node-b",
      targetPort: 9_850,
    });
    expect(legacy.sent).toContainEqual(expect.objectContaining({
      workerId: "worker-b",
      type: "runtime.stream.open",
      payload: { streamId: "legacy-stream", targetPort: 9_850 },
    }));
    expect(runtimeStreamCount(legacy.hub)).toBe(2);
    legacy.hub.close();
  });
});

function recoveryHub(destinationSupportsRecovery: boolean): {
  hub: WorkerHub;
  sent: Array<{ workerId: string; type: string; payload: Record<string, unknown> }>;
  store: { setWorkerStatus: ReturnType<typeof vi.fn> };
} {
  const workers = [
    worker("worker-a", "node-a", true),
    worker("worker-b", "node-b", destinationSupportsRecovery),
  ];
  const store = {
    getWorker: vi.fn((workerId: string) => workers.find((candidate) => candidate.id === workerId)),
    listWorkers: vi.fn(() => workers),
    setWorkerStatus: vi.fn(),
  };
  const hub = new WorkerHub(store as unknown as MeshStore);
  const sent: Array<{ workerId: string; type: string; payload: Record<string, unknown> }> = [];
  vi.spyOn(hub, "isConnected").mockReturnValue(true);
  vi.spyOn(hub, "send").mockImplementation((workerId, type, payload) => {
    sent.push({ workerId, type, payload: payload as Record<string, unknown> });
    return true;
  });
  return { hub, sent, store };
}

function worker(workerId: string, nodeId: string, recovery: boolean) {
  return {
    id: workerId,
    capabilities: {
      distributedExecutor: {
        protocol: "gdlp-worker-tunnel/2",
        ...(recovery ? { streamRecovery: "offset-ack-v1" } : {}),
        nodeId,
        stageHost: "127.0.0.1",
        stagePort: 9_850,
        runtime: "python-safetensors",
      },
    },
  };
}

function openRecoverableStream(hub: WorkerHub): void {
  deliver(hub, "worker-a", "runtime.stream.open", {
    ...recoveryIdentity(),
    destinationNodeId: "node-b",
    targetPort: 9_850,
  });
}

function recoveryIdentity(): Record<string, unknown> {
  return {
    streamId: "stream-recovery",
    generation: 0,
    recoveryToken: RECOVERY_TOKEN,
  };
}

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

function disconnectWorker(hub: WorkerHub, workerId: string): void {
  const timer = setTimeout(() => undefined, 60_000);
  timer.unref();
  const state = {
    socket: {},
    workerId,
    ready: true,
    helloTimer: timer,
    pending: false,
    messageWindowStartedAt: Date.now(),
    messagesInWindow: 0,
  };
  const internal = hub as unknown as {
    connections: Map<string, typeof state>;
    allConnections: Set<typeof state>;
    handleClose(connection: typeof state): void;
  };
  internal.connections.set(workerId, state);
  internal.allConnections.add(state);
  internal.handleClose(state);
}

function runtimeStreamCount(hub: WorkerHub): number {
  return (hub as unknown as { runtimeStreams: Map<string, unknown> }).runtimeStreams.size;
}
