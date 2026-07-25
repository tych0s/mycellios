import { describe, expect, it, vi } from "vitest";
import { WorkerHub } from "../src/coordinator/worker-hub.js";
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
});

function connectionState() {
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
    ready: false,
    helloTimer: setTimeout(() => undefined, 60_000),
    pending: true,
    messageWindowStartedAt: Date.now(),
    messagesInWindow: 0,
  };
}
