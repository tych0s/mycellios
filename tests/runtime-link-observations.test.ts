import { describe, expect, it } from "vitest";
import { RuntimeLinkObservationStore } from "../src/coordinator/runtime-link-observations.js";

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
      rttP50Ms: 30,
      rttP95Ms: 50,
      goodputMbpsP50: 60,
      successfulSamples: 3,
      failedSamples: 1,
      availability: 0.75,
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
});
