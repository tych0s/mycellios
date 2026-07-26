import { describe, expect, it } from "vitest";
import { estimateLinkLatencyMs } from "../src/coordinator/connected-executor-activation.js";

/**
 * Regression: `coordinatorRttMs` was hardcoded to 0 in the worker agent and
 * nothing ever wrote it, so every link fell into the `Math.max(0.1, …)` floor
 * and the planner modelled the whole fleet as 0.1 ms LAN. Measured reality is
 * 54-437 ms per hop, and the latency term dominates the cost model — so the one
 * number that decides placement was the one number the planner never saw.
 */
describe("link latency estimate", () => {
  it("does not pretend an unmeasured link is a LAN cable", () => {
    // The old code returned 0.1 here.
    expect(estimateLinkLatencyMs(0, 0)).toBe(65);
  });

  it("averages two measured round trips", () => {
    expect(estimateLinkLatencyMs(60, 80)).toBe(70);
  });

  it("prefers one real measurement over the fleet default", () => {
    expect(estimateLinkLatencyMs(120, 0)).toBe(120);
    expect(estimateLinkLatencyMs(0, 120)).toBe(120);
  });

  it("keeps a genuinely fast measured link fast", () => {
    // A LAN pair that really measured sub-millisecond must not be inflated.
    expect(estimateLinkLatencyMs(0.4, 0.6)).toBeCloseTo(0.5, 5);
  });

  it("never returns zero, so downstream division stays safe", () => {
    expect(estimateLinkLatencyMs(0.00001, 0.00001)).toBeGreaterThan(0);
  });

  it("ignores values that are not finite", () => {
    expect(estimateLinkLatencyMs(Number.NaN, Number.POSITIVE_INFINITY)).toBe(65);
    expect(estimateLinkLatencyMs(Number.NaN, 90)).toBe(90);
  });

  it("treats a negative reading as missing rather than as a discount", () => {
    expect(estimateLinkLatencyMs(-10, -10)).toBe(65);
  });

  it("ranks a measured slow link above an unmeasured one, so blindness is not rewarded", () => {
    const measuredSlow = estimateLinkLatencyMs(400, 400);
    const unmeasured = estimateLinkLatencyMs(0, 0);
    expect(measuredSlow).toBeGreaterThan(unmeasured);
    // ...and a measured good link still wins outright.
    expect(estimateLinkLatencyMs(20, 20)).toBeLessThan(unmeasured);
  });
});
