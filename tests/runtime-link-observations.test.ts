import { describe, expect, it } from "vitest";
import {
  RuntimeLinkObservationStore,
  selectPreferredRuntimeLinkObservation,
} from "../src/coordinator/runtime-link-observations.js";
import { WorkerHub } from "../src/coordinator/worker-hub.js";
import { MeshDatabase } from "../src/storage/database.js";
import { MeshStore } from "../src/storage/store.js";

describe("runtime link observations", () => {
  it("derives bounded P50/P95, goodput, and reachability from real attempts", () => {
    const store = new RuntimeLinkObservationStore(60_000, 8);
    store.recordSuccess("node-a", "node-b", 20, 80, 1_000);
    store.recordSuccess("node-a", "node-b", 30, 60, 2_000);
    store.recordSuccess("node-a", "node-b", 50, 40, 3_000);
    store.recordFailure("node-a", "node-b", 4_000);

    expect(store.observations(5_000)).toEqual([{
      fromNodeId: "node-a",
      toNodeId: "node-b",
      measuredAt: 4_000,
      validUntil: 64_000,
      rttP50Ms: 30,
      rttP95Ms: 50,
      jitterP95Ms: 20,
      goodputMbpsP50: 60,
      successfulSamples: 3,
      failedSamples: 1,
      availability: 0.75,
      transportMode: "relay",
      confidence: 0.75,
    }]);
  });

  it("expires stale evidence instead of silently reusing an old route", () => {
    const store = new RuntimeLinkObservationStore(1_000, 8);
    store.recordSuccess("node-a", "node-b", 20, 80, 1_000);
    expect(store.observations(2_000)).toHaveLength(1);
    expect(store.observations(2_001)).toEqual([]);
  });

  it("keeps directed links independent and can invalidate a departed node", () => {
    const store = new RuntimeLinkObservationStore();
    store.recordSuccess("node-a", "node-b", 20, 80, 1_000);
    store.recordSuccess("node-b", "node-a", 40, 70, 1_000);
    expect(store.observations(1_000).map((entry) => entry.rttP50Ms)).toEqual([20, 40]);
    store.clearNode("node-a");
    expect(store.observations(1_000)).toEqual([]);
  });

  it("never merges direct reachability with relay measurements", () => {
    const store = new RuntimeLinkObservationStore(60_000, 8);
    store.recordSuccess("node-a", "node-b", 12, 900, 1_000, "direct");
    store.recordFailure("node-a", "node-b", 2_000, "direct");
    store.recordSuccess("node-a", "node-b", 40, 80, 2_000, "relay");
    const observations = store.observations(2_000);
    expect(observations).toEqual([
      expect.objectContaining({ transportMode: "direct", availability: 0.5, rttP50Ms: 12 }),
      expect.objectContaining({ transportMode: "relay", availability: 1, rttP50Ms: 40 }),
    ]);
    expect(selectPreferredRuntimeLinkObservation(observations)?.transportMode).toBe("relay");
    store.recordSuccess("node-a", "node-b", 10, 850, 3_000, "direct");
    expect(selectPreferredRuntimeLinkObservation(
      store.observations(3_000),
      0.6,
    )?.transportMode).toBe("direct");
  });

  it("rehydrates bounded fresh probe evidence after coordinator restart", () => {
    const database = new MeshDatabase(":memory:");
    const durable = new MeshStore(database);
    const now = Date.now();
    for (let index = 0; index < 40; index += 1) {
      durable.saveRuntimeLinkSample({
        fromNodeId: "node-a",
        toNodeId: "node-b",
        measuredAt: now - 40 + index,
        rttMs: 20 + index,
        goodputMbps: 100 - index,
        transportMode: "relay",
      }, 60_000, 32);
    }
    durable.saveRuntimeLinkSample({
      fromNodeId: "node-b",
      toNodeId: "node-a",
      measuredAt: now,
      rttMs: null,
      goodputMbps: null,
      transportMode: "relay",
    }, 60_000, 32);
    durable.saveRuntimeLinkSample({
      fromNodeId: "node-a",
      toNodeId: "node-b",
      measuredAt: now,
      rttMs: 8,
      goodputMbps: 900,
      transportMode: "direct",
    }, 60_000, 32);

    expect(durable.listRuntimeLinkSamples(now, 60_000)).toHaveLength(34);
    const rehydrated = new RuntimeLinkObservationStore(
      60_000,
      32,
      durable.listRuntimeLinkSamples(now, 60_000),
    );
    expect(rehydrated.observations(now)).toEqual([
      expect.objectContaining({
        fromNodeId: "node-a",
        toNodeId: "node-b",
        transportMode: "direct",
        successfulSamples: 1,
      }),
      expect.objectContaining({
        fromNodeId: "node-a",
        toNodeId: "node-b",
        transportMode: "relay",
        successfulSamples: 32,
      }),
    ]);

    const restartedHub = new WorkerHub(durable);
    expect(restartedHub.runtimeLinkObservations(now)).toEqual([
      expect.objectContaining({ transportMode: "direct", successfulSamples: 1 }),
      expect.objectContaining({
        fromNodeId: "node-a",
        toNodeId: "node-b",
        transportMode: "relay",
        successfulSamples: 32,
      }),
    ]);
    restartedHub.close();
    database.close();
  });
});
