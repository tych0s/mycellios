import { describe, expect, it } from "vitest";
import { deriveDecodeScales } from "../src/distribution/node-scale.js";
import { CoordinatorRttTracker } from "../src/worker/link-telemetry.js";

describe("CoordinatorRttTracker", () => {
  it("reports null until a sample lands, so an unmeasured link cannot pass for adjacent", () => {
    const tracker = new CoordinatorRttTracker();
    expect(tracker.publishedRttMs()).toBeNull();
    expect(tracker.snapshot()).toEqual({ ewmaMs: null, minMs: null, lastMs: null, samples: 0 });
  });

  it("pairs a pong with the outstanding ping", () => {
    const tracker = new CoordinatorRttTracker();
    tracker.markPingSent(1_000);
    expect(tracker.markPongReceived(1_065)).toBe(65);
    expect(tracker.publishedRttMs()).toBe(65);
  });

  it("ignores a pong with no ping outstanding", () => {
    const tracker = new CoordinatorRttTracker();
    expect(tracker.markPongReceived(1_000)).toBeNull();
    expect(tracker.publishedRttMs()).toBeNull();
  });

  it("does not pair a pong across a reconnect", () => {
    const tracker = new CoordinatorRttTracker();
    tracker.markPingSent(1_000);
    tracker.reset();
    expect(tracker.markPongReceived(1_400)).toBeNull();
  });

  it("keeps only the newest ping when two go out before a pong", () => {
    const tracker = new CoordinatorRttTracker();
    tracker.markPingSent(1_000);
    tracker.markPingSent(1_500);
    // Pairing with the older ping would report 600 ms instead of the true 100.
    expect(tracker.markPongReceived(1_600)).toBe(100);
  });

  it("rejects implausible samples instead of poisoning the average", () => {
    const tracker = new CoordinatorRttTracker();
    expect(tracker.observe(-5)).toBeNull();
    expect(tracker.observe(120_000)).toBeNull();
    expect(tracker.observe(Number.NaN)).toBeNull();
    expect(tracker.publishedRttMs()).toBeNull();
  });

  it("smooths jitter but still tracks a node that moved continents", () => {
    const tracker = new CoordinatorRttTracker({ alpha: 0.25 });
    for (let index = 0; index < 12; index += 1) tracker.observe(65);
    expect(tracker.publishedRttMs()).toBeCloseTo(65, 5);

    // One bad sample must not dominate...
    tracker.observe(437);
    expect(tracker.publishedRttMs()!).toBeLessThan(160);

    // ...but a sustained change must land.
    for (let index = 0; index < 20; index += 1) tracker.observe(437);
    expect(tracker.publishedRttMs()!).toBeGreaterThan(430);
  });

  it("tracks the minimum separately from the average", () => {
    const tracker = new CoordinatorRttTracker();
    tracker.observe(120);
    tracker.observe(54);
    tracker.observe(200);
    const snapshot = tracker.snapshot();
    expect(snapshot.minMs).toBe(54);
    expect(snapshot.lastMs).toBe(200);
    expect(snapshot.samples).toBe(3);
  });

  it("rejects a nonsensical smoothing weight", () => {
    expect(() => new CoordinatorRttTracker({ alpha: 0 })).toThrow();
    expect(() => new CoordinatorRttTracker({ alpha: 1.5 })).toThrow();
  });
});

describe("deriveDecodeScales", () => {
  it("gives the fastest node scale 1 and scales the rest above it", () => {
    const result = deriveDecodeScales([
      { nodeId: "fast", measuredTokensPerSecond: 40 },
      { nodeId: "slow", measuredTokensPerSecond: 10 },
    ]);
    expect(result.referenceTokensPerSecond).toBe(40);
    expect(result.fullyMeasured).toBe(true);
    expect(result.scales).toEqual([
      { nodeId: "fast", decodeScale: 1, measured: true },
      { nodeId: "slow", decodeScale: 4, measured: true },
    ]);
  });

  it("does NOT degenerate to an equal split when nodes differ", () => {
    // This is the regression the whole module exists for: decodeScale was
    // hardcoded to 1, so ProportionalComputePlanner saw a constant vector.
    const result = deriveDecodeScales([
      { nodeId: "rtx3060", measuredTokensPerSecond: 27.4 },
      { nodeId: "gtx1050ti", measuredTokensPerSecond: 17.1 },
    ]);
    const scales = result.scales.map((entry) => entry.decodeScale);
    expect(new Set(scales).size).toBe(2);
    expect(scales[1]!).toBeGreaterThan(scales[0]!);
  });

  it("falls back to 1 and says so when nothing was measured", () => {
    const result = deriveDecodeScales([
      { nodeId: "a", measuredTokensPerSecond: null },
      { nodeId: "b", measuredTokensPerSecond: null },
    ]);
    expect(result.fullyMeasured).toBe(false);
    expect(result.referenceTokensPerSecond).toBeNull();
    expect(result.scales.every((entry) => entry.decodeScale === 1 && !entry.measured)).toBe(true);
  });

  it("flags a partially measured fleet rather than pretending it is informed", () => {
    const result = deriveDecodeScales([
      { nodeId: "measured", measuredTokensPerSecond: 20 },
      { nodeId: "unknown", measuredTokensPerSecond: null },
    ]);
    expect(result.fullyMeasured).toBe(false);
    expect(result.scales[0]).toEqual({ nodeId: "measured", decodeScale: 1, measured: true });
    expect(result.scales[1]).toEqual({ nodeId: "unknown", decodeScale: 1, measured: false });
  });

  it("clamps a pathological node so it is neither handed the model nor starved", () => {
    const result = deriveDecodeScales([
      { nodeId: "fast", measuredTokensPerSecond: 1_000 },
      { nodeId: "crawling", measuredTokensPerSecond: 0.01 },
    ]);
    expect(result.scales[1]!.decodeScale).toBe(20);
  });

  it("ignores zero and non-finite throughput", () => {
    const result = deriveDecodeScales([
      { nodeId: "zero", measuredTokensPerSecond: 0 },
      { nodeId: "inf", measuredTokensPerSecond: Number.POSITIVE_INFINITY },
      { nodeId: "real", measuredTokensPerSecond: 12 },
    ]);
    expect(result.referenceTokensPerSecond).toBe(12);
    expect(result.scales[0]!.measured).toBe(false);
    expect(result.scales[1]!.measured).toBe(false);
    expect(result.scales[2]!.measured).toBe(true);
  });

  it("handles an empty fleet", () => {
    const result = deriveDecodeScales([]);
    expect(result.scales).toEqual([]);
    expect(result.fullyMeasured).toBe(false);
  });
});
