import { describe, expect, it, vi } from "vitest";
import { WorkerHub } from "../src/coordinator/worker-hub.js";
import type { WorkerEnvelope } from "../src/contracts/types.js";
import type { MeshStore } from "../src/storage/store.js";

describe("WorkerHub runtime link probes", () => {
  it("measures the directed path used by live relay traffic", () => {
    const workers = [
      runtimeWorker("worker-a", "node-a"),
      runtimeWorker("worker-b", "node-b"),
    ];
    const hub = new WorkerHub({
      listWorkers: () => workers,
    } as unknown as MeshStore);
    const sent: Array<{
      workerId: string;
      type: string;
      payload: Record<string, unknown>;
    }> = [];
    vi.spyOn(hub, "isConnected").mockReturnValue(true);
    vi.spyOn(hub, "send").mockImplementation((workerId, type, payload) => {
      sent.push({ workerId, type, payload: payload as Record<string, unknown> });
      return true;
    });

    expect(hub.sampleRuntimeLinks(1, 10_000)).toBe(1);
    const start = sent.find((message) => message.type === "runtime.link.probe.start")!;
    expect(start.workerId).toBe("worker-a");
    expect(start.payload.destinationNodeId).toBe("node-b");
    expect(start.payload.payloadBytes).toBe(16 * 1024);
    const probeId = start.payload.probeId as string;
    const data = Buffer.alloc(16 * 1024, 7).toString("base64");

    deliver(hub, "worker-a", "runtime.link.probe.ping", {
      probeId,
      destinationNodeId: "node-b",
      data,
    });
    expect(sent.at(-1)).toMatchObject({
      workerId: "worker-b",
      type: "runtime.link.probe.ping",
      payload: { probeId, data },
    });

    deliver(hub, "worker-b", "runtime.link.probe.pong", { probeId, data });
    expect(sent.at(-1)).toMatchObject({
      workerId: "worker-a",
      type: "runtime.link.probe.pong",
      payload: { probeId, data },
    });

    deliver(hub, "worker-a", "runtime.link.probe.result", {
      probeId,
      destinationNodeId: "node-b",
      rttMs: 42,
      goodputMbps: 6.24,
    });
    expect(hub.runtimeLinkObservations()).toEqual([
      expect.objectContaining({
        fromNodeId: "node-a",
        toNodeId: "node-b",
        rttP50Ms: 42,
        goodputMbpsP50: 6.24,
        availability: 1,
      }),
    ]);
    hub.close();
  });

  it("rejects a result from a worker that does not own the probe", () => {
    const workers = [
      runtimeWorker("worker-a", "node-a"),
      runtimeWorker("worker-b", "node-b"),
    ];
    const hub = new WorkerHub({
      listWorkers: () => workers,
    } as unknown as MeshStore);
    const sent: Array<{ workerId: string; type: string; payload: Record<string, unknown> }> = [];
    vi.spyOn(hub, "isConnected").mockReturnValue(true);
    vi.spyOn(hub, "send").mockImplementation((workerId, type, payload) => {
      sent.push({ workerId, type, payload: payload as Record<string, unknown> });
      return true;
    });

    hub.sampleRuntimeLinks(1, 10_000);
    const probeId = sent[0]!.payload.probeId as string;
    deliver(hub, "worker-b", "runtime.link.probe.result", {
      probeId,
      destinationNodeId: "node-b",
      rttMs: 1,
      goodputMbps: 1_000,
    });
    expect(hub.runtimeLinkObservations()).toEqual([]);
    hub.close();
  });
});

function deliver(
  hub: WorkerHub,
  workerId: string,
  type: string,
  payload: Record<string, unknown>,
): void {
  const internal = hub as unknown as {
    handleRuntimeLinkProbeEnvelope(envelope: WorkerEnvelope): void;
  };
  internal.handleRuntimeLinkProbeEnvelope({ v: 1, workerId, type, payload });
}

function runtimeWorker(id: string, nodeId: string) {
  return {
    id,
    capabilities: {
      distributedExecutor: {
        protocol: "gdlp-worker-tunnel/2",
        nodeId,
      },
    },
  };
}
