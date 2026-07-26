import { describe, expect, it } from "vitest";
import {
  runAdaptiveBenchmarkCampaign,
  summarizeBenchmarkSeries,
} from "../src/benchlab/campaign.js";

describe("adaptive benchmark campaign", () => {
  it("separates warmup and reaches confidence only after the minimum samples", async () => {
    const attempts: string[] = [];
    const result = await runAdaptiveBenchmarkCampaign({
      warmupSamples: 2,
      minimumSamples: 7,
      maximumSamples: 15,
      targetConfidenceHalfWidthPct: 3,
      measure: async ({ phase, sampleIndex }) => {
        attempts.push(`${phase}:${sampleIndex}`);
        return phase === "warmup" ? 1 : 100 + (sampleIndex % 2);
      },
      score: (sample) => sample,
    });

    expect(result.warmups).toEqual([1, 1]);
    expect(result.samples).toHaveLength(7);
    expect(result.stable).toBe(true);
    expect(result.stoppedBecause).toBe("confidence_reached");
    expect(result.summary).toMatchObject({ count: 7, p50: 100 });
    expect(attempts.slice(0, 2)).toEqual(["warmup:0", "warmup:1"]);
  });

  it("retains a temporary availability failure after the sample recovers", async () => {
    let calls = 0;
    const sleeps: number[] = [];
    const result = await runAdaptiveBenchmarkCampaign({
      warmupSamples: 0,
      minimumSamples: 1,
      maximumSamples: 1,
      retriesPerSample: 2,
      retryDelayMs: 25,
      targetConfidenceHalfWidthPct: 1,
      measure: async () => {
        calls += 1;
        if (calls === 1) throw new Error("HTTP 503: route warming");
        return 8;
      },
      score: (sample) => sample,
      retryable: (error) => String(error).includes("503"),
      sleep: async (milliseconds) => {
        sleeps.push(milliseconds);
      },
    });

    expect(result.samples).toEqual([8]);
    expect(result.failures).toEqual([{
      phase: "measurement",
      sampleIndex: 0,
      attempt: 0,
      retryable: true,
      recovered: true,
      message: "HTTP 503: route warming",
    }]);
    expect(sleeps).toEqual([25]);
  });

  it("fails closed on permanent errors and never manufactures a score", async () => {
    const result = await runAdaptiveBenchmarkCampaign({
      warmupSamples: 0,
      minimumSamples: 3,
      maximumSamples: 3,
      measure: async () => {
        throw new Error("model digest mismatch");
      },
      score: () => 999,
      retryable: () => false,
    });

    expect(result.samples).toEqual([]);
    expect(result.summary).toBeNull();
    expect(result.stable).toBe(false);
    expect(result.stoppedBecause).toBe("insufficient_samples");
    expect(result.failures).toHaveLength(3);
  });

  it("reports noise rather than hiding it behind the mean", () => {
    const summary = summarizeBenchmarkSeries([2.174, 2.4, 2.9, 3.1, 3.973]);
    expect(summary.p5).toBeLessThan(summary.p50);
    expect(summary.p95).toBeGreaterThan(summary.p50);
    expect(summary.coefficientOfVariationPct).toBeGreaterThan(20);
    expect(summary.confidenceHalfWidthPct).toBeGreaterThan(10);
  });

  it("rejects invalid physical measurements", () => {
    expect(() => summarizeBenchmarkSeries([])).toThrow(
      "benchmark_series_requires_at_least_one_value",
    );
    expect(() => summarizeBenchmarkSeries([1, Number.NaN])).toThrow(
      "benchmark_series_values_must_be_finite_and_non_negative",
    );
  });
});
