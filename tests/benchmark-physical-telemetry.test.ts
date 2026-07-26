import { describe, expect, it } from "vitest";
import {
  PhysicalTelemetryWindow,
  type BenchmarkNodeTelemetry,
} from "../src/benchlab/physical-telemetry.js";

describe("physical benchmark telemetry window", () => {
  it("integrates measured power and reports peak memory, heat and utilization", () => {
    const window = new PhysicalTelemetryWindow({ maximumIntegrationGapMs: 6_000 });
    window.observe({ atMs: 0, nodes: [node("a", 100, 30, 55, 8, 6)] });
    window.observe({ atMs: 5_000, nodes: [node("a", 140, 70, 65, 8, 4)] });
    window.observe({ atMs: 10_000, nodes: [node("a", 120, 50, 60, 8, 5)] });

    expect(window.summarize(new Set(["a"]), 100)).toEqual({
      sampleCount: 3,
      durationMs: 10_000,
      energyCoveragePct: 100,
      powerWattsP50: 120,
      powerWattsP95: 138,
      powerWattsPeak: 140,
      energyWh: 0.34722222,
      energyWhPerToken: 0.0034722222,
      utilizationPctP50: 50,
      utilizationPctP95: 68,
      temperatureCPeak: 65,
      usedMemoryGbPeak: 4,
    });
  });

  it("does not invent energy across missing data or a long heartbeat gap", () => {
    const window = new PhysicalTelemetryWindow({ maximumIntegrationGapMs: 5_000 });
    window.observe({ atMs: 0, nodes: [node("a", 100, 30, 55, 8, 6)] });
    window.observe({ atMs: 4_000, nodes: [node("a", null, 40, 56, 8, 6)] });
    window.observe({ atMs: 20_000, nodes: [node("a", 120, 50, 57, 8, 5)] });

    const summary = window.summarize(new Set(["a"]), 20);
    expect(summary.energyWh).toBeNull();
    expect(summary.energyWhPerToken).toBeNull();
    expect(summary.energyCoveragePct).toBe(0);
    expect(summary.powerWattsP50).toBe(110);
  });

  it("restricts aggregation to nodes that actually executed the route", () => {
    const window = new PhysicalTelemetryWindow();
    window.observe({
      atMs: 0,
      nodes: [
        node("used", 100, 50, 60, 8, 4),
        node("idle", 300, 90, 80, 24, 1),
      ],
    });
    window.observe({
      atMs: 1_000,
      nodes: [
        node("used", 100, 50, 60, 8, 4),
        node("idle", 300, 90, 80, 24, 1),
      ],
    });

    const summary = window.summarize(new Set(["used"]), 10);
    expect(summary.powerWattsP50).toBe(100);
    expect(summary.usedMemoryGbPeak).toBe(4);
    expect(summary.temperatureCPeak).toBe(60);
  });

  it("fails closed on impossible physical telemetry", () => {
    const window = new PhysicalTelemetryWindow();
    expect(() => window.observe({
      atMs: 0,
      nodes: [node("a", 10, 20, 30, 4, 5)],
    })).toThrow("benchmark_telemetry_free_memory_exceeds_offer");
    window.observe({ atMs: 10, nodes: [node("a", 10, 20, 30, 4, 2)] });
    expect(() => window.observe({
      atMs: 10,
      nodes: [node("a", 10, 20, 30, 4, 2)],
    })).toThrow("benchmark_telemetry_timestamp_must_increase");
  });
});

function node(
  nodeId: string,
  powerWatts: number | null,
  utilizationPct: number | null,
  temperatureC: number | null,
  offeredMemoryGb: number | null,
  freeOfferedMemoryGb: number | null,
): BenchmarkNodeTelemetry {
  return {
    nodeId,
    powerWatts,
    utilizationPct,
    temperatureC,
    offeredMemoryGb,
    freeOfferedMemoryGb,
  };
}
