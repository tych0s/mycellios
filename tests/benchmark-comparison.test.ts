import { describe, expect, it } from "vitest";
import {
  benchmarkRunSnapshot,
  benchmarkComparisonNarrative,
  benchmarkFilterValues,
  benchmarkTrendRuns,
  compareBenchmarkRuns,
  defaultBenchmarkBaseline,
  EMPTY_BENCHMARK_FILTERS,
  filterBenchmarkRuns,
} from "../landing/src/benchmark-comparison.js";
import {
  BENCHMARK_RUN_SCHEMA,
  emptyComparison,
  type BenchmarkEvidence,
  type BenchmarkRun,
} from "../src/benchlab/types.js";
import { sealBenchmarkScenario } from "../src/benchlab/scenario.js";

function makeRun(input: {
  runId: string;
  model?: string;
  evidence?: BenchmarkEvidence;
  nodes?: number;
  memoryGb?: number;
  tokensPerSecond?: number;
  ttftMsP95?: number;
  measurementId?: string;
  modelDigest?: string | null;
  backend?: string;
  promptTokens?: number;
  version?: string;
  status?: BenchmarkRun["status"];
  stable?: boolean;
  p5?: number;
  p95?: number;
}): BenchmarkRun {
  const model = input.model ?? "qwen3-0.6b";
  const nodes = input.nodes ?? 2;
  const memoryGb = input.memoryGb ?? 8;
  return {
    schema: BENCHMARK_RUN_SCHEMA,
    runId: input.runId,
    version: input.version ?? input.runId,
    label: input.runId,
    gitCommit: null,
    gitBranch: "main",
    gitDirty: false,
    build: {
      release: input.runId,
      releaseSource: "override",
      revision: null,
      revisionSource: "unknown",
      sourceId: null,
      sourceIdSource: "unknown",
      participantSourceIds: [],
    },
    startedAt: "2026-07-23T10:00:00.000Z",
    finishedAt: "2026-07-23T10:00:01.000Z",
    suite: "real-runtime",
    status: input.status ?? "passed",
    measurements: [sealBenchmarkScenario({
      id: input.measurementId ?? `startup-${model}-n${nodes}-o24`,
      title: "Real inference",
      description: "Measured through the real coordinator route.",
      evidence: input.evidence ?? "physical",
      environment: "lan",
      model: {
        id: model,
        label: model,
        revision: "rev-1",
        digest: input.modelDigest === undefined ? `sha256:${"b".repeat(64)}` : input.modelDigest,
        precision: "bf16",
      },
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
          backend: input.backend ?? "cuda",
          offeredMemoryGb: memoryGb,
          observedPowerWatts: 100,
        }],
      },
      topology: {
        digest: null,
        stageCount: nodes,
        boundaries: [],
        nodeIds: Array.from({ length: nodes }, (_, index) => `node-${index + 1}`),
        routeClasses: ["pipeline"],
      },
      workload: {
        promptTokens: input.promptTokens ?? 8,
        outputTokens: 24,
        concurrentSequences: 1,
        ...(input.stable === undefined ? {} : { statisticallyStable: input.stable }),
      },
      metrics: {
        tokensPerSecond: input.tokensPerSecond ?? 20,
        tokensPerSecondP5: input.p5 ?? input.tokensPerSecond ?? 20,
        tokensPerSecondP50: input.tokensPerSecond ?? 20,
        tokensPerSecondP95: input.p95 ?? input.tokensPerSecond ?? 20,
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
    })],
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

  it("only attributes a faster result to code when both stable bands are separated", () => {
    const baseline = makeRun({
      runId: "baseline",
      tokensPerSecond: 20,
      p5: 19,
      p95: 21,
      stable: true,
    });
    const current = makeRun({
      runId: "current",
      tokensPerSecond: 25,
      p5: 24,
      p95: 26,
      stable: true,
    });

    const comparison = compareBenchmarkRuns(current, baseline);
    expect(comparison.verdict).toBe("faster-code-signal");
    expect(benchmarkComparisonNarrative(comparison)).toMatchObject({
      eyebrow: "SEÑAL DE MEJORA DEL CÓDIGO",
      title: "Más rápido con la misma capacidad",
    });
  });

  it("labels overlapping ranges as normal variation and unstable data as inconclusive", () => {
    const baseline = makeRun({
      runId: "baseline",
      tokensPerSecond: 20,
      p5: 18,
      p95: 22,
      stable: true,
    });
    const overlapping = makeRun({
      runId: "overlapping",
      tokensPerSecond: 21,
      p5: 20,
      p95: 23,
      stable: true,
    });
    const unstable = makeRun({
      runId: "unstable",
      tokensPerSecond: 25,
      p5: 24,
      p95: 26,
      stable: false,
    });

    expect(compareBenchmarkRuns(overlapping, baseline).verdict).toBe("within-variation");
    const inconclusive = compareBenchmarkRuns(unstable, baseline);
    expect(inconclusive.verdict).toBe("insufficient-confidence");
    expect(benchmarkComparisonNarrative(inconclusive).title)
      .toBe("Mismo escenario, faltan muestras estables");
  });

  it("separates capacity scaling from a code improvement", () => {
    const baseline = makeRun({ runId: "baseline", nodes: 1, memoryGb: 4, tokensPerSecond: 12 });
    const current = makeRun({ runId: "current", nodes: 2, memoryGb: 8, tokensPerSecond: 20 });
    const comparison = compareBenchmarkRuns(current, baseline);

    expect(comparison.mode).toBe("capacity-change");
    expect(comparison.nodesDelta).toBe(1);
    expect(comparison.speedChangePct).toBeNull();
    expect(comparison.efficiencyPerNodeChangePct).toBeNull();
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

  it("changes the fingerprint for backend, workload and model digest", () => {
    const baseline = makeRun({ runId: "baseline" });
    const backend = makeRun({ runId: "backend", backend: "rocm" });
    const workload = makeRun({ runId: "workload", promptTokens: 9 });
    const digest = makeRun({ runId: "digest", modelDigest: `sha256:${"c".repeat(64)}` });

    expect(backend.measurements[0]?.scenarioFingerprint)
      .not.toBe(baseline.measurements[0]?.scenarioFingerprint);
    expect(workload.measurements[0]?.scenarioFingerprint)
      .not.toBe(baseline.measurements[0]?.scenarioFingerprint);
    expect(digest.measurements[0]?.scenarioFingerprint)
      .not.toBe(baseline.measurements[0]?.scenarioFingerprint);
    expect(compareBenchmarkRuns(backend, baseline).speedChangePct).toBeNull();
    expect(compareBenchmarkRuns(backend, baseline).mode).toBe("configuration-change");
    expect(compareBenchmarkRuns(workload, baseline).speedChangePct).toBeNull();
    expect(compareBenchmarkRuns(digest, baseline).speedChangePct).toBeNull();
  });

  it("keeps the fingerprint deterministic when unordered inventory changes order", () => {
    const run = makeRun({ runId: "ordered" });
    const measurement = run.measurements[0]!;
    const { scenarioFingerprint: _discarded, ...unsealed } = measurement;
    const cpu = {
      label: "CPU host",
      kind: "cpu" as const,
      count: 1,
      memoryGb: 32,
      backend: "cpu",
    };
    const forward = sealBenchmarkScenario({
      ...unsealed,
      inventory: {
        ...unsealed.inventory,
        profiles: [...unsealed.inventory.profiles, cpu],
      },
    });
    const reverse = sealBenchmarkScenario({
      ...unsealed,
      inventory: {
        ...unsealed.inventory,
        profiles: [cpu, ...unsealed.inventory.profiles],
      },
    });

    expect(reverse.scenarioFingerprint).toBe(forward.scenarioFingerprint);
  });

  it("fails closed when the model digest is missing", () => {
    const baseline = makeRun({ runId: "baseline", modelDigest: null });
    const current = makeRun({ runId: "current", modelDigest: null });

    expect(compareBenchmarkRuns(current, baseline).mode).toBe("unavailable");
    expect(compareBenchmarkRuns(current, baseline).speedChangePct).toBeNull();
    expect(defaultBenchmarkBaseline(current, [baseline])).toBeNull();
  });

  it("filters by model, version, status, backend and evidence without inventing missing backends", () => {
    const physical = makeRun({
      runId: "physical",
      version: "0.3.0",
      backend: "cuda",
      status: "passed",
    });
    const loopback = makeRun({
      runId: "loopback",
      version: "0.2.0",
      backend: "",
      evidence: "loopback",
      status: "inconclusive",
    });
    const values = benchmarkFilterValues([physical, loopback]);

    expect(values.backends).toContain("cuda");
    expect(values.backends).toContain("__without_reading__");
    expect(filterBenchmarkRuns([physical, loopback], {
      ...EMPTY_BENCHMARK_FILTERS,
      version: "0.3.0",
      status: "passed",
      backend: "cuda",
      evidence: "physical",
    })).toEqual([physical]);
    expect(filterBenchmarkRuns([physical, loopback], {
      ...EMPTY_BENCHMARK_FILTERS,
      backend: "__without_reading__",
    })).toEqual([loopback]);
  });

  it("builds trends only from the exact selected fingerprint", () => {
    const selected = makeRun({ runId: "selected", nodes: 2, memoryGb: 8 });
    const sameScenario = makeRun({ runId: "same", nodes: 2, memoryGb: 8 });
    const moreCapacity = makeRun({ runId: "capacity", nodes: 4, memoryGb: 16 });
    const otherBackend = makeRun({ runId: "backend", nodes: 2, memoryGb: 8, backend: "rocm" });

    expect(benchmarkTrendRuns(selected, [
      moreCapacity,
      sameScenario,
      otherBackend,
      selected,
    ]).map((run) => run.runId)).toEqual(["same", "selected"]);
  });
});
