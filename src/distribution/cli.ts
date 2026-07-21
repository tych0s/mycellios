import { runDistributionBenchmark } from "./benchmark.js";
import { defaultDistributionPlanners } from "./planners.js";
import {
  fixedDistributionScenarios,
  randomDistributionScenario,
} from "./scenarios.js";

const seed = 0x5eedc0de;
const randomTrials = Number.parseInt(process.env.DISTRIBUTION_TRIALS ?? "50", 10);
const fixed = fixedDistributionScenarios();
const random = Array.from({ length: randomTrials }, (_, index) =>
  randomDistributionScenario(seed, index),
);
const planners = defaultDistributionPlanners();

console.log("LLM distribution lab: reproducible analytical model, not a physical benchmark");
console.log(`Escenarios: ${fixed.length} fijos + ${random.length} Monte Carlo; seed=${seed}`);

const fixedResult = runDistributionBenchmark(fixed, planners);
for (const scenario of fixed) {
  const rows = fixedResult.rows.filter((row) => row.scenarioId === scenario.id);
  console.log(`\n${scenario.id}: ${scenario.description}`);
  console.table(
    rows.map((row) => ({
      algoritmo: row.algorithm,
      viable: Boolean(row.evaluated?.metrics.feasible),
      ganador: row.winner,
      etapas: row.evaluated?.metrics.stages ?? 0,
      codec: row.evaluated?.plan.codec ?? "-",
      microbatch: row.evaluated?.plan.microBatchSize ?? 0,
      chunk_prefill: row.evaluated?.plan.prefillChunkTokens ?? 0,
      tpot_p95_ms: round(row.evaluated?.metrics.tpotMs),
      tok_s_usuario: round(row.evaluated?.metrics.tokensPerSecondPerSequence),
      ttft_p95_ms: round(row.evaluated?.metrics.ttftMs),
      tok_s_agregados: round(row.evaluated?.metrics.aggregateTokensPerSecond),
      Wh_token: round(row.evaluated?.metrics.energyWhPerOutputToken, 6),
      route: row.evaluated?.plan.stages
        .map((stage) => `${stage.nodeId}[${stage.layerStart}:${stage.layerEnd}]`)
        .join(" -> ") ?? "no route",
    })),
  );
}

const monteCarlo = runDistributionBenchmark(random, planners);
console.log("\nResumen Monte Carlo");
console.table(
  monteCarlo.summaries.map((summary) => ({
    algoritmo: summary.algorithm,
    viable: `${summary.feasible}/${summary.scenarios}`,
    victorias: summary.wins,
    regret_medio_pct: round(summary.meanRegret * 100),
    regret_p95_pct: round(summary.p95Regret * 100),
    etapas_medias: round(summary.meanStages),
    tpot_medio_ms: round(summary.meanTpotMs),
    ttft_medio_ms: round(summary.meanTtftMs),
    Wh_token_medio: round(summary.meanEnergyWhPerToken, 6),
  })),
);

function round(value: number | undefined, digits = 2): number | string {
  if (value === undefined || !Number.isFinite(value)) return "-";
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}
