import { describe, expect, it } from "vitest";
import { workerConnectionIsHeartbeatStale } from "../src/coordinator/worker-hub.js";

describe("WorkerHub stale connection policy", () => {
  it("does not close a freshly registered worker before its first heartbeat", () => {
    const now = Date.now();
    expect(workerConnectionIsHeartbeatStale({
      status: "offline",
      lastSeenAt: now - 5,
    }, now)).toBe(false);
  });

  it("closes an offline worker after the heartbeat grace window", () => {
    const now = Date.now();
    expect(workerConnectionIsHeartbeatStale({
      status: "offline",
      lastSeenAt: now - 10_001,
    }, now)).toBe(true);
  });
});
