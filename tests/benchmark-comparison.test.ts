import { describe, expect, it } from "vitest";
import {
  benchmarkRunSnapshot,
  compareBenchmarkRuns,
  defaultBenchmarkBaseline,
} from "../landing/src/benchmark-comparison.js";
import {
  BENCHMARK_RUN_SCHEMA,
  emptyComparison,
  type BenchmarkEvidence,
  type BenchmarkRun,
} from "../src/benchlab/types.js";

function makeRun(input: {
  runId: string;
  model?: string;
  evidence?: BenchmarkEvidence;
  nodes?: number;
  memoryGb?: number;
  tokensPerSecond?: number;
  ttftMsP95?: number;
  measurementId?: string;
}): BenchmarkRun {
  const model = input.model ?? "qwen3-0.6b";
  const nodes = input.nodes ?? 2;
  const memoryGb = input.memoryGb ?? 8;
  return {
    schema: BENCHMARK_RUN_SCHEMA,
    runId: input.runId,
    version: input.runId,
    label: input.runId,
    gitCommit: null,
    gitBranch: "main",
    gitDirty: false,
    startedAt: "2026-07-23T10:00:00.000Z",
    finishedAt: "2026-07-23T10:00:01.000Z",
    suite: "real-runtime",
    status: "passed",
    measurements: [{
      id: input.measurementId ?? `startup-${model}-n${nodes}-o24`,
      title: "Real inference",
      description: "Measured through the real coordinator route.",
      evidence: input.evidence ?? "physical",
      environment: "lan",
      model: { id: model, label: model, revision: "rev-1", precision: "bf16" },
      inventory: {
        totalDevices: nodes,
        connectedDevices: nodes,
        selectedDevices: nodes,
        offeredMemoryGb: memoryGb,
        observedPowerWatts: 100,
        profiles: [{
          label: "GPU",
          kind: "gpu",
          count: nodes,
          memoryGb,
          offeredMemoryGb: memoryGb,
          observedPowerWatts: 100,
        }],
      },
      workload: {
        promptTokens: 8,
        outputTokens: 24,
        concurrentSequences: 1,
      },
      metrics: {
        tokensPerSecond: input.tokensPerSecond ?? 20,
        tokensPerSecondP95: input.tokensPerSecond ?? 20,
        aggregateTokensPerSecond: input.tokensPerSecond ?? 20,
        ttftMsP50: input.ttftMsP95 ?? 500,
        ttftMsP95: input.ttftMsP95 ?? 500,
        tpotMsP50: 50,
        tpotMsP95: 50,
        acceptanceRate: 1,
        energyWhPerToken: null,
      },
      status: "passed",
      comparison: emptyComparison(),
      notes: [],
    }],
  };
}

describe("benchmark comparison", () => {
  it("normalizes throughput by node, offered GB and observed watts", () => {
    const snapshot = benchmarkRunSnapshot(makeRun({
      runId: "current",
      nodes: 2,
      memoryGb: 8,
      tokensPerSecond: 24,
    }));

    expect(snapshot?.tokensPerSecondPerNode).toBe(12);
    expect(snapshot?.tokensPerSecondPerGb).toBe(3);
    expect(snapshot?.tokensPerSecondPerWatt).toBe(0.24);
  });

  it("marks equal hardware and workload as a comparable program improvement", () => {
    const baseline = makeRun({ runId: "baseline", tokensPerSecond: 20, ttftMsP95: 500 });
    const current = makeRun({ runId: "current", tokensPerSecond: 25, ttftMsP95: 400 });
    const comparison = compareBenchmarkRuns(current, baseline);

    expect(comparison.mode).toBe("same-scenario");
    expect(comparison.speedChangePct).toBe(25);
    expect(comparison.latencyImprovementPct).toBe(20);
    expect(comparison.efficiencyPerNodeChangePct).toBe(25);
  });

  it("separates capacity scaling from a code improvement", () => {
    const baseline = makeRun({ runId: "baseline", nodes: 1, memoryGb: 4, tokensPerSecond: 12 });
    const current = makeRun({ runId: "current", nodes: 2, memoryGb: 8, tokensPerSecond: 20 });
    const comparison = compareBenchmarkRuns(current, baseline);

    expect(comparison.mode).toBe("capacity-change");
    expect(comparison.nodesDelta).toBe(1);
    expect(comparison.speedChangePct).toBeCloseTo(66.666, 2);
    expect(comparison.efficiencyPerNodeChangePct).toBeCloseTo(-16.666, 2);
  });

  it("does not calculate an improvement across models or evidence classes", () => {
    const baseline = makeRun({ runId: "baseline", model: "qwen3", evidence: "loopback" });
    const current = makeRun({ runId: "current", model: "llama3", evidence: "physical" });
    const comparison = compareBenchmarkRuns(current, baseline);

    expect(comparison.mode).toBe("different-model");
    expect(comparison.speedChangePct).toBeNull();
  });

  it("chooses the newest same-scenario baseline before a different configuration", () => {
    const current = makeRun({ runId: "current" });
    const exact = makeRun({ runId: "exact" });
    const capacity = makeRun({ runId: "capacity", nodes: 4, memoryGb: 16 });

    expect(defaultBenchmarkBaseline(current, [current, capacity, exact])?.runId).toBe("exact");
  });

  it("does not silently choose a different model as the default reference", () => {
    const current = makeRun({ runId: "current", model: "qwen3" });
    const other = makeRun({ runId: "other", model: "llama3" });

    expect(defaultBenchmarkBaseline(current, [current, other])).toBeNull();
  });
});
