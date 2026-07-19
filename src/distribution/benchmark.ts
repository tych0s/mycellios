import { evaluatePlanner, DEFAULT_SEARCH_OPTIONS } from "./planners.js";
import type { DistributionScenario } from "./scenarios.js";
import type {
  DistributionPlanner,
  EvaluatedDistributionPlan,
  SearchOptions,
} from "./types.js";

export interface DistributionBenchmarkRow {
  scenarioId: string;
  algorithm: string;
  evaluated: EvaluatedDistributionPlan | null;
  winner: boolean;
  regret: number;
}

export interface DistributionAlgorithmSummary {
  algorithm: string;
  scenarios: number;
  feasible: number;
  wins: number;
  meanRegret: number;
  p95Regret: number;
  meanStages: number;
  meanTpotMs: number;
  meanTtftMs: number;
  meanEnergyWhPerToken: number;
}

export interface DistributionBenchmarkResult {
  rows: DistributionBenchmarkRow[];
  summaries: DistributionAlgorithmSummary[];
}

export function runDistributionBenchmark(
  scenarios: DistributionScenario[],
  planners: DistributionPlanner[],
  options: SearchOptions = DEFAULT_SEARCH_OPTIONS,
): DistributionBenchmarkResult {
  const rows: DistributionBenchmarkRow[] = [];
  for (const scenario of scenarios) {
    const evaluated = planners.map((planner) => ({
      planner,
      result: evaluatePlanner(
        planner,
        scenario.model,
        scenario.topology,
        scenario.workload,
        options,
      ),
    }));
    const bestObjective = Math.min(
      ...evaluated
        .map((entry) => entry.result?.objective ?? Number.POSITIVE_INFINITY)
        .filter(Number.isFinite),
    );
    for (const entry of evaluated) {
      const objective = entry.result?.objective ?? Number.POSITIVE_INFINITY;
      const winner = Number.isFinite(objective) && objective <= bestObjective * (1 + 1e-9);
      rows.push({
        scenarioId: scenario.id,
        algorithm: entry.planner.id,
        evaluated: entry.result,
        winner,
        regret:
          Number.isFinite(objective) && Number.isFinite(bestObjective) && bestObjective > 0
            ? objective / bestObjective - 1
            : Number.POSITIVE_INFINITY,
      });
    }
  }
  const summaries = planners.map((planner) => summarizeAlgorithm(planner.id, rows));
  summaries.sort(
    (left, right) =>
      right.wins - left.wins ||
      left.meanRegret - right.meanRegret ||
      right.feasible - left.feasible,
  );
  return { rows, summaries };
}

function summarizeAlgorithm(
  algorithm: string,
  rows: DistributionBenchmarkRow[],
): DistributionAlgorithmSummary {
  const matching = rows.filter((row) => row.algorithm === algorithm);
  const feasible = matching.filter((row) => row.evaluated?.metrics.feasible);
  const regrets = matching
    .map((row) => row.regret)
    .filter(Number.isFinite)
    .sort((left, right) => left - right);
  return {
    algorithm,
    scenarios: matching.length,
    feasible: feasible.length,
    wins: matching.filter((row) => row.winner).length,
    meanRegret: mean(regrets),
    p95Regret: percentile(regrets, 0.95),
    meanStages: mean(feasible.map((row) => row.evaluated!.metrics.stages)),
    meanTpotMs: mean(feasible.map((row) => row.evaluated!.metrics.tpotMs)),
    meanTtftMs: mean(feasible.map((row) => row.evaluated!.metrics.ttftMs)),
    meanEnergyWhPerToken: mean(
      feasible.map((row) => row.evaluated!.metrics.energyWhPerOutputToken),
    ),
  };
}

function mean(values: number[]): number {
  if (values.length === 0) return Number.POSITIVE_INFINITY;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function percentile(values: number[], fraction: number): number {
  if (values.length === 0) return Number.POSITIVE_INFINITY;
  const index = Math.min(values.length - 1, Math.ceil(values.length * fraction) - 1);
  return values[index]!;
}
