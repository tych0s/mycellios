import { describe, expect, it } from "vitest";
import { speculativeLinkTimeSavedMs } from "../src/distribution/cost-model.js";
import {
  TP_CELL_MAX_ONE_WAY_LATENCY_MS,
  compareParallelismArchitectures,
  isViableTensorParallelCellLatency,
} from "../src/distribution/parallelism.js";
import { evaluatePlanner, TopologyBeamPlanner } from "../src/distribution/planners.js";
import { fixedDistributionScenarios } from "../src/distribution/scenarios.js";

describe("DSD speculative link-time bound", () => {
  it("matches hand-computed values of (N-1)*t1*(k-1)/k", () => {
    expect(speculativeLinkTimeSavedMs(2, 10, 2)).toBeCloseTo(5, 12);
    expect(speculativeLinkTimeSavedMs(4, 20, 2.4)).toBeCloseTo(35, 12);
    expect(speculativeLinkTimeSavedMs(3, 40, 4)).toBeCloseTo(60, 12);
  });

  it("saves nothing with one accepted token per round or a free link", () => {
    expect(speculativeLinkTimeSavedMs(5, 100, 1)).toBe(0);
    expect(speculativeLinkTimeSavedMs(4, 0, 6)).toBe(0);
  });

  it("rejects invalid stage counts, latencies and acceptance rates", () => {
    expect(() => speculativeLinkTimeSavedMs(1, 10, 2)).toThrow(
      "speculative_stage_count_must_be_an_integer_of_at_least_2",
    );
    expect(() => speculativeLinkTimeSavedMs(2.5, 10, 2)).toThrow();
    expect(() => speculativeLinkTimeSavedMs(Number.NaN, 10, 2)).toThrow();
    expect(() => speculativeLinkTimeSavedMs(3, -1, 2)).toThrow(
      "speculative_link_latency_must_be_finite_and_non_negative",
    );
    expect(() => speculativeLinkTimeSavedMs(3, Number.POSITIVE_INFINITY, 2)).toThrow();
    expect(() => speculativeLinkTimeSavedMs(3, Number.NaN, 2)).toThrow();
    expect(() => speculativeLinkTimeSavedMs(3, 10, 0.5)).toThrow(
      "speculative_accepted_tokens_must_be_finite_and_at_least_1",
    );
    expect(() => speculativeLinkTimeSavedMs(3, 10, Number.POSITIVE_INFINITY)).toThrow();
    expect(() => speculativeLinkTimeSavedMs(3, 10, Number.NaN)).toThrow();
  });

  it("throws instead of returning Infinity when the product overflows", () => {
    expect(() => speculativeLinkTimeSavedMs(4, Number.MAX_VALUE, 2)).toThrow(
      "speculative_link_time_saved_overflowed",
    );
  });

  it("grows monotonically with acceptance and stage count, below (N-1)*t1", () => {
    const acceptances = [1, 1.5, 2.4, 4, 8, 64];
    for (let index = 1; index < acceptances.length; index += 1) {
      expect(speculativeLinkTimeSavedMs(4, 25, acceptances[index]!)).toBeGreaterThan(
        speculativeLinkTimeSavedMs(4, 25, acceptances[index - 1]!),
      );
    }
    expect(speculativeLinkTimeSavedMs(4, 25, 64)).toBeLessThan(3 * 25);
    const stageCounts = [2, 3, 4, 6, 9];
    for (let index = 1; index < stageCounts.length; index += 1) {
      expect(speculativeLinkTimeSavedMs(stageCounts[index]!, 25, 2.4)).toBeGreaterThan(
        speculativeLinkTimeSavedMs(stageCounts[index - 1]!, 25, 2.4),
      );
    }
  });
});

describe("tensor-parallel cell latency gate", () => {
  it("accepts microsecond-class links and rejects beyond the 0.3 ms ceiling", () => {
    expect(TP_CELL_MAX_ONE_WAY_LATENCY_MS).toBe(0.3);
    expect(isViableTensorParallelCellLatency(0.05)).toBe(true);
    expect(isViableTensorParallelCellLatency(0.3)).toBe(true);
    expect(isViableTensorParallelCellLatency(0.31)).toBe(false);
    expect(isViableTensorParallelCellLatency(12)).toBe(false);
    expect(isViableTensorParallelCellLatency(-0.01)).toBe(false);
    expect(isViableTensorParallelCellLatency(Number.NaN)).toBe(false);
    expect(isViableTensorParallelCellLatency(Number.POSITIVE_INFINITY)).toBe(false);
  });

  it("annotates tensor estimates with a non-binding warning on slow cells only", () => {
    const base = fixedDistributionScenarios()[0]!;
    const withLatency = (oneWayLatencyMs: number) => ({
      ...base,
      topology: {
        ...base.topology,
        links: base.topology.links.map((link) => ({ ...link, oneWayLatencyMs })),
      },
    });

    const slow = withLatency(12);
    const slowPipeline = evaluatePlanner(
      new TopologyBeamPlanner(),
      slow.model,
      slow.topology,
      slow.workload,
    )!;
    const slowEstimates = compareParallelismArchitectures(
      slow.model,
      slow.topology,
      slow.workload,
      slowPipeline.plan,
    );
    const slowRing = slowEstimates.find((entry) => entry.architecture === "tensor-ring")!;
    expect(slowRing.feasible).toBe(true);
    expect(slowRing.warning).toBe("tp_cell_one_way_latency_above_viability_ceiling");
    const slowPipelineEstimate = slowEstimates.find(
      (entry) => entry.architecture === "contiguous-pipeline",
    )!;
    expect(slowPipelineEstimate.warning).toBeUndefined();

    const fast = withLatency(0.05);
    const fastPipeline = evaluatePlanner(
      new TopologyBeamPlanner(),
      fast.model,
      fast.topology,
      fast.workload,
    )!;
    const fastEstimates = compareParallelismArchitectures(
      fast.model,
      fast.topology,
      fast.workload,
      fastPipeline.plan,
    );
    const fastRing = fastEstimates.find((entry) => entry.architecture === "tensor-ring")!;
    expect(fastRing.feasible).toBe(true);
    expect(fastRing.warning).toBeUndefined();
  });
});
