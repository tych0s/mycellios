import { describe, expect, it, vi } from "vitest";
import { WorkerHub, workerSessionSupersedes } from "../src/coordinator/worker-hub.js";
import type { MeshStore } from "../src/storage/store.js";

describe("WorkerHub duplicate desktop connections", () => {
  it("keeps the healthy connection instead of letting duplicates supersede each other", () => {
    const store = {
      getWorker: vi.fn(() => ({ id: "worker-stable" })),
    } as unknown as MeshStore;
    const hub = new WorkerHub(store);
    const first = connectionState();
    const duplicate = connectionState();
    const deliver = (state: ReturnType<typeof connectionState>) => {
      (hub as unknown as {
        handleRawMessage(connection: unknown, raw: string): void;
      }).handleRawMessage(state, JSON.stringify({
        v: 1,
        type: "worker.hello",
        workerId: "worker-stable",
        payload: {},
      }));
    };

    deliver(first);
    first.ready = true;
    deliver(duplicate);

    expect(first.socket.close).not.toHaveBeenCalled();
    expect(duplicate.socket.close).toHaveBeenCalledWith(4409, "duplicate worker connection");
    expect(hub.isConnected("worker-stable")).toBe(true);
    clearTimeout(duplicate.helloTimer);
  });

  it("lets a higher ownership generation replace the still-healthy old socket", () => {
    const store = { getWorker: vi.fn(() => ({ id: "worker-stable" })) } as unknown as MeshStore;
    const hub = new WorkerHub(store);
    const first = connectionState(1);
    const recovered = connectionState(2);
    const deliver = (state: ReturnType<typeof connectionState>) => (hub as unknown as {
      handleRawMessage(connection: unknown, raw: string): void;
    }).handleRawMessage(state, JSON.stringify({ v: 1, type: "worker.hello", workerId: "worker-stable", payload: {} }));
    deliver(first); first.ready = true; deliver(recovered);
    expect(first.socket.close).toHaveBeenCalledWith(4409, "superseded connection");
    expect(recovered.socket.close).not.toHaveBeenCalled();
    expect(hub.isConnected("worker-stable")).toBe(false);
    clearTimeout(first.helloTimer); clearTimeout(recovered.helloTimer);
  });

  it("orders sessions only within one stable identity", () => {
    const first = session(1);
    expect(workerSessionSupersedes(first, session(2))).toBe(true);
    expect(workerSessionSupersedes(first, { ...session(2), identityId: "another-node" })).toBe(false);
    expect(workerSessionSupersedes(session(2), first)).toBe(false);
  });

  it("closes a bound socket when its credential generation is no longer current", () => {
    const store = { getWorker: vi.fn(() => ({ id: "worker-stable" })) } as unknown as MeshStore;
    const hub = new WorkerHub(store);
    (hub as unknown as { sessionIsCurrent: () => boolean }).sessionIsCurrent = () => false;
    const stale = connectionState(1);
    (hub as unknown as { handleRawMessage(connection: unknown, raw: string): void }).handleRawMessage(
      stale, JSON.stringify({ v: 1, type: "worker.hello", workerId: "worker-stable", payload: {} }),
    );
    expect(stale.socket.close).toHaveBeenCalledWith(4403, "worker session superseded");
    clearTimeout(stale.helloTimer);
  });

  it("emits disconnect synchronously when a security action evicts a worker", () => {
    const store = { getWorker: vi.fn(() => ({ id: "worker-stable" })),
      setWorkerStatus: vi.fn(), deregisterWorker: vi.fn(() => true) } as unknown as MeshStore;
    const hub = new WorkerHub(store);
    const active = connectionState();
    (hub as unknown as { connections: Map<string, unknown>; allConnections: Set<unknown> }).connections.set("worker-stable", active);
    (hub as unknown as { connections: Map<string, unknown>; allConnections: Set<unknown> }).allConnections.add(active);
    active.workerId = "worker-stable"; active.ready = true; active.pending = false;
    const disconnected = vi.fn(); hub.on("disconnect", disconnected);
    expect(hub.removeWorker("worker-stable", "node credential revoked")).toBe(true);
    expect(disconnected).toHaveBeenCalledWith("worker-stable");
    expect(active.socket.close).toHaveBeenCalledWith(4403, "node credential revoked");
    clearTimeout(active.helloTimer);
  });
});

function session(generation: number) {
  return { workerId: "worker-stable", identityKind: "device" as const, identityId: "node-stable",
    credentialFingerprint: `sha256:${String(generation).repeat(64)}`, generation };
}

function connectionState(generation?: number) {
  const socket = {
    OPEN: 1,
    readyState: 1,
    bufferedAmount: 0,
    send: vi.fn(),
    close: vi.fn(),
  };
  return {
    socket,
    workerId: null as string | null,
    authorizedSession: generation === undefined ? null : session(generation),
    ready: false,
    helloTimer: setTimeout(() => undefined, 60_000),
    pending: true,
    messageWindowStartedAt: Date.now(),
    messagesInWindow: 0,
  };
}
