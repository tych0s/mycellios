import { describe, expect, it } from "vitest";
import { runDistributionBenchmark } from "../src/distribution/benchmark.js";
import { evaluateDistributionPlan } from "../src/distribution/cost-model.js";
import { compareParallelismArchitectures } from "../src/distribution/parallelism.js";
import {
  DEFAULT_SEARCH_OPTIONS,
  ExhaustiveTopologyPlanner,
  FleetTopologyPlanner,
  TopologyBeamPlanner,
  defaultDistributionPlanners,
  evaluatePlanner,
} from "../src/distribution/planners.js";
import {
  fixedDistributionScenarios,
  randomDistributionScenario,
} from "../src/distribution/scenarios.js";
import { runSensitivitySweep } from "../src/distribution/sensitivity.js";
import {
  buildRuntimePipelineManifest,
  validateRuntimePipelineManifest,
} from "../src/distribution/runtime-manifest.js";

describe("distribution optimization laboratory", () => {
  it("builds contiguous, memory-feasible routes for every fixed scenario", () => {
    const planner = new TopologyBeamPlanner();
    for (const scenario of fixedDistributionScenarios()) {
      const evaluated = evaluatePlanner(
        planner,
        scenario.model,
        scenario.topology,
        scenario.workload,
      );
      expect(evaluated, scenario.id).not.toBeNull();
      expect(evaluated!.metrics.feasible, scenario.id).toBe(true);
      expect(evaluated!.plan.stages.length).toBeLessThanOrEqual(scenario.workload.maxStages);
      expect(evaluated!.plan.stages[0]!.layerStart).toBe(0);
      expect(evaluated!.plan.stages.at(-1)!.layerEnd).toBe(scenario.model.layers.length);
      for (let index = 1; index < evaluated!.plan.stages.length; index += 1) {
        expect(evaluated!.plan.stages[index]!.layerStart).toBe(
          evaluated!.plan.stages[index - 1]!.layerEnd,
        );
      }
      for (const stage of evaluated!.metrics.stageMetrics) {
        expect(stage.memoryBytes).toBeLessThanOrEqual(stage.memoryLimitBytes);
      }
    }
  });

  it("stays within 0.2% of exhaustive optimum on a tractable topology", () => {
    const scenario = fixedDistributionScenarios()[0]!;
    const beam = evaluatePlanner(
      new TopologyBeamPlanner(),
      scenario.model,
      scenario.topology,
      scenario.workload,
    )!;
    const exact = evaluatePlanner(
      new ExhaustiveTopologyPlanner(),
      scenario.model,
      scenario.topology,
      scenario.workload,
    )!;
    expect(beam.objective / exact.objective - 1).toBeLessThan(0.002);
  });

  it("accounts for direct token return and worsens when every link gains latency", () => {
    const scenario = fixedDistributionScenarios()[1]!;
    const evaluated = evaluatePlanner(
      new TopologyBeamPlanner(),
      scenario.model,
      scenario.topology,
      scenario.workload,
    )!;
    expect(evaluated.metrics.tokenReturnMs).toBeGreaterThan(0);
    const slowerTopology = {
      nodes: scenario.topology.nodes,
      links: scenario.topology.links.map((link) => ({
        ...link,
        oneWayLatencyMs: link.oneWayLatencyMs + 20,
      })),
    };
    const slower = evaluateDistributionPlan(
      scenario.model,
      slowerTopology,
      scenario.workload,
      evaluated.plan,
    );
    expect(slower.tpotMs).toBeGreaterThan(evaluated.metrics.tpotMs);
    expect(slower.ttftMs).toBeGreaterThan(evaluated.metrics.ttftMs);
  });

  it("treats KV memory per conversation as a hard placement constraint", () => {
    const scenario = fixedDistributionScenarios()[1]!;
    const evaluated = evaluatePlanner(
      new TopologyBeamPlanner(),
      scenario.model,
      scenario.topology,
      scenario.workload,
    )!;
    const overloaded = evaluateDistributionPlan(
      scenario.model,
      scenario.topology,
      { ...scenario.workload, contextTokens: scenario.workload.contextTokens * 64 },
      evaluated.plan,
    );
    expect(overloaded.feasible).toBe(false);
    expect(overloaded.infeasibleReason).toMatch(/^memory_exceeded:/);
  });

  it("never chooses a lossy codec beyond the configured quality budget", () => {
    const scenario = fixedDistributionScenarios()[0]!;
    const strict = evaluatePlanner(
      new TopologyBeamPlanner(),
      scenario.model,
      scenario.topology,
      { ...scenario.workload, maxQualityLoss: 0 },
    )!;
    expect(strict.plan.codec).toBe("fp16");
    expect(strict.metrics.qualityLoss).toBe(0);
  });

  it("selects a short route instead of chaining the whole 32-node fleet", () => {
    const scenario = fixedDistributionScenarios()[3]!;
    const evaluated = evaluatePlanner(
      new TopologyBeamPlanner(),
      scenario.model,
      scenario.topology,
      scenario.workload,
    )!;
    expect(scenario.topology.nodes).toHaveLength(32);
    expect(evaluated.metrics.stages).toBeLessThanOrEqual(4);
  });

  it("uses recurrent microbatching when eight sequences share a route", () => {
    const scenario = fixedDistributionScenarios()[4]!;
    const evaluated = evaluatePlanner(
      new TopologyBeamPlanner(),
      scenario.model,
      scenario.topology,
      scenario.workload,
    )!;
    expect(evaluated.plan.microBatchSize).toBeGreaterThan(1);
    expect(evaluated.metrics.aggregateTokensPerSecond).toBeGreaterThan(
      evaluated.metrics.tokensPerSecondPerSequence,
    );
  });

  it("wins the seeded heterogeneous comparison without sacrificing feasibility", () => {
    const scenarios = Array.from({ length: 5 }, (_, index) =>
      randomDistributionScenario(0x5eedc0de, index),
    );
    const result = runDistributionBenchmark(scenarios, defaultDistributionPlanners());
    const beam = result.summaries.find((summary) => summary.algorithm === "topology-beam")!;
    expect(beam.feasible).toBe(scenarios.length);
    expect(beam.wins).toBeGreaterThanOrEqual(4);
    expect(beam.meanRegret).toBeLessThanOrEqual(0.02);
  });

  it("is deterministic for a fixed scenario and search configuration", () => {
    const scenario = randomDistributionScenario(12345, 3);
    const first = evaluatePlanner(
      new TopologyBeamPlanner(DEFAULT_SEARCH_OPTIONS),
      scenario.model,
      scenario.topology,
      scenario.workload,
    );
    const second = evaluatePlanner(
      new TopologyBeamPlanner(DEFAULT_SEARCH_OPTIONS),
      scenario.model,
      scenario.topology,
      scenario.workload,
    );
    expect(second).toEqual(first);
  });

  it("models packet loss as retransmission latency, not route disappearance", () => {
    const scenario = fixedDistributionScenarios()[0]!;
    const evaluated = evaluatePlanner(
      new TopologyBeamPlanner(),
      scenario.model,
      scenario.topology,
      scenario.workload,
    )!;
    const lossy = evaluateDistributionPlan(
      scenario.model,
      {
        ...scenario.topology,
        links: scenario.topology.links.map((link) => ({ ...link, lossRate: 0.05 })),
      },
      scenario.workload,
      evaluated.plan,
    );
    expect(lossy.feasible).toBe(true);
    expect(lossy.routeAvailability).toBeCloseTo(evaluated.metrics.routeAvailability, 12);
    expect(lossy.tpotMs).toBeGreaterThan(evaluated.metrics.tpotMs);
  });

  it("shows why per-layer tensor collectives are a poor WAN baseline", () => {
    const base = fixedDistributionScenarios()[0]!;
    const scenario = {
      ...base,
      topology: {
        ...base.topology,
        links: base.topology.links.map((link) => ({
          ...link,
          oneWayLatencyMs: 12,
          jitterP95Ms: 4,
          bandwidthMbps: 150,
        })),
      },
    };
    const pipeline = evaluatePlanner(
      new TopologyBeamPlanner(),
      scenario.model,
      scenario.topology,
      scenario.workload,
    )!;
    const comparison = compareParallelismArchitectures(
      scenario.model,
      scenario.topology,
      scenario.workload,
      pipeline.plan,
    );
    const ring = comparison.find((entry) => entry.architecture === "tensor-ring")!;
    const contiguous = comparison.find(
      (entry) => entry.architecture === "contiguous-pipeline",
    )!;
    expect(ring.feasible).toBe(true);
    expect(ring.tpotMs).toBeGreaterThan(contiguous.tpotMs * 2);
    expect(ring.networkBytesPerOutputToken).toBeGreaterThan(
      contiguous.networkBytesPerOutputToken,
    );
  });

  it("produces deterministic latency, bandwidth, concurrency and stage sweeps", () => {
    const points = runSensitivitySweep(fixedDistributionScenarios()[1]!);
    expect(points).toHaveLength(26);
    expect(new Set(points.map((point) => point.dimension))).toEqual(
      new Set(["one_way_latency_ms", "bandwidth_mbps", "conversations", "max_stages"]),
    );
    const latency = points.filter((point) => point.dimension === "one_way_latency_ms");
    expect(latency.at(-1)!.tpotMs).toBeGreaterThan(latency[0]!.tpotMs);
  });

  it("plans a short route without feeding a thousand-node fleet into the beam", () => {
    const template = fixedDistributionScenarios()[3]!;
    const nodeTemplate = template.topology.nodes[0]!;
    const nodes = Array.from({ length: 1_000 }, (_, index) => ({
      ...nodeTemplate,
      id: `large-${index}`,
      region: `cell-${Math.floor(index / 10)}`,
      decodeScale: 0.55 + (index % 7) * 0.08,
      prefillScale: 0.62 + (index % 5) * 0.09,
      memoryBytes: (384 + (index % 3) * 128) * 1024 * 1024,
    }));
    const links = [];
    for (let cell = 0; cell < 100; cell += 1) {
      const start = cell * 10;
      for (let left = start; left < start + 10; left += 1) {
        for (let right = start; right < start + 10; right += 1) {
          if (left === right) continue;
          links.push({
            from: `large-${left}`,
            to: `large-${right}`,
            oneWayLatencyMs: 0.8 + (cell % 4) * 0.2,
            jitterP95Ms: 0.4,
            bandwidthMbps: 500,
            lossRate: 0.001,
          });
        }
      }
    }
    const evaluated = evaluatePlanner(
      new FleetTopologyPlanner(),
      template.model,
      { nodes, links },
      template.workload,
    );
    expect(evaluated?.metrics.feasible).toBe(true);
    expect(evaluated!.plan.algorithm).toBe("fleet-topology");
    expect(evaluated!.plan.stages.length).toBeLessThanOrEqual(template.workload.maxStages);
    expect(new Set(evaluated!.plan.stages.map((stage) => stage.nodeId)).size).toBe(
      evaluated!.plan.stages.length,
    );
  });

  it("turns the winning plan into an executable contiguous runtime manifest", () => {
    const scenario = fixedDistributionScenarios()[1]!;
    const manifest = buildRuntimePipelineManifest({
      model: scenario.model,
      modelRevision: "test-revision-1",
      topology: {
        nodes: scenario.topology.nodes.map((node, index) => ({
          ...node,
          endpoint: { host: "127.0.0.1", port: 20_000 + index },
        })),
        links: scenario.topology.links,
      },
      workload: scenario.workload,
    });
    expect(() => validateRuntimePipelineManifest(manifest)).not.toThrow();
    expect(manifest.protocol).toBe("gdlp/2");
    expect(manifest.plans.prefill.stages[0]!.layerStart).toBe(0);
    expect(manifest.plans.decode.stages.at(-1)!.layerEnd).toBe(
      scenario.model.layers.length,
    );
    expect(manifest.plans.prefill.predicted.feasible).toBe(true);
    expect(manifest.plans.decode.predicted.feasible).toBe(true);
    expect(manifest.plans.decode.activationCodec).toBe("fp16");
  });
});
