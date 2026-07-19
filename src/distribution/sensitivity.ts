import { pathToFileURL } from "node:url";
import { evaluatePlanner, TopologyBeamPlanner } from "./planners.js";
import { fixedDistributionScenarios } from "./scenarios.js";
import type { DistributionScenario } from "./scenarios.js";

export interface SensitivityPoint {
  dimension: "one_way_latency_ms" | "bandwidth_mbps" | "conversations" | "max_stages";
  value: number;
  feasible: boolean;
  selectedStages: number;
  codec: string;
  tpotMs: number;
  ttftMs: number;
  perUserTokensPerSecond: number;
  aggregateTokensPerSecond: number;
}

export function runSensitivitySweep(base: DistributionScenario): SensitivityPoint[] {
  const planner = new TopologyBeamPlanner();
  const dimensions: Array<{
    dimension: SensitivityPoint["dimension"];
    values: number[];
    scenario: (value: number) => DistributionScenario;
  }> = [
    {
      dimension: "one_way_latency_ms",
      values: [0.25, 1, 3, 10, 25, 50, 100],
      scenario: (value) => ({
        ...base,
        topology: {
          ...base.topology,
          links: base.topology.links.map((link) => ({ ...link, oneWayLatencyMs: value })),
        },
      }),
    },
    {
      dimension: "bandwidth_mbps",
      values: [25, 50, 100, 250, 500, 1_000, 2_500],
      scenario: (value) => ({
        ...base,
        topology: {
          ...base.topology,
          links: base.topology.links.map((link) => ({ ...link, bandwidthMbps: value })),
        },
      }),
    },
    {
      dimension: "conversations",
      values: [1, 2, 4, 8, 16, 32],
      scenario: (value) => ({
        ...base,
        workload: {
          ...base.workload,
          concurrentSequences: value,
          contextTokens: value >= 16 ? Math.min(384, base.workload.contextTokens) : base.workload.contextTokens,
        },
      }),
    },
    {
      dimension: "max_stages",
      values: [2, 3, 4, 5, 6, 8],
      scenario: (value) => ({
        ...base,
        workload: { ...base.workload, maxStages: value },
      }),
    },
  ];

  const result: SensitivityPoint[] = [];
  for (const dimension of dimensions) {
    for (const value of dimension.values) {
      const scenario = dimension.scenario(value);
      const evaluated = evaluatePlanner(
        planner,
        scenario.model,
        scenario.topology,
        scenario.workload,
      );
      result.push({
        dimension: dimension.dimension,
        value,
        feasible: Boolean(evaluated?.metrics.feasible),
        selectedStages: evaluated?.metrics.stages ?? 0,
        codec: evaluated?.plan.codec ?? "-",
        tpotMs: evaluated?.metrics.tpotMs ?? Number.POSITIVE_INFINITY,
        ttftMs: evaluated?.metrics.ttftMs ?? Number.POSITIVE_INFINITY,
        perUserTokensPerSecond: evaluated?.metrics.tokensPerSecondPerSequence ?? 0,
        aggregateTokensPerSecond: evaluated?.metrics.aggregateTokensPerSecond ?? 0,
      });
    }
  }
  return result;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const scenario = fixedDistributionScenarios()[1]!;
  console.log(`Sensitivity: ${scenario.id}`);
  const points = runSensitivitySweep(scenario);
  for (const dimension of [...new Set(points.map((point) => point.dimension))]) {
    console.log(`\n${dimension}`);
    console.table(
      points
        .filter((point) => point.dimension === dimension)
        .map((point) => ({
          value: point.value,
          viable: point.feasible,
          stages: point.selectedStages,
          codec: point.codec,
          tpot_ms: round(point.tpotMs),
          ttft_ms: round(point.ttftMs),
          tok_s_user: round(point.perUserTokensPerSecond),
          tok_s_total: round(point.aggregateTokensPerSecond),
        })),
    );
  }
}

function round(value: number): number | string {
  return Number.isFinite(value) ? Math.round(value * 100) / 100 : "-";
}
