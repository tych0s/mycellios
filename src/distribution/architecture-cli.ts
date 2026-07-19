import { compareParallelismArchitectures } from "./parallelism.js";
import { evaluatePlanner, TopologyBeamPlanner } from "./planners.js";
import { fixedDistributionScenarios } from "./scenarios.js";

console.log("Comparación de arquitecturas (modelo analítico reproducible)");
for (const scenario of fixedDistributionScenarios()) {
  const planned = evaluatePlanner(
    new TopologyBeamPlanner(),
    scenario.model,
    scenario.topology,
    scenario.workload,
  );
  if (!planned) continue;
  const estimates = compareParallelismArchitectures(
    scenario.model,
    scenario.topology,
    scenario.workload,
    planned.plan,
    {
      draftMsPerToken: Math.max(1, planned.metrics.tpotMs * 0.08),
      acceptanceProbability: 0.75,
      verificationScalePerExtraToken: 0.22,
      maxDraftTokens: 8,
    },
  );
  console.log(`\n${scenario.id}: ${scenario.description}`);
  console.table(
    estimates.map((estimate) => ({
      arquitectura: estimate.architecture,
      viable: estimate.feasible,
      nodos: estimate.nodes,
      tpot_ms: finite(estimate.tpotMs),
      ttft_ms: finite(estimate.ttftMs),
      tok_s_usuario: finite(estimate.tokensPerSecondPerSequence),
      tok_s_total: finite(estimate.aggregateTokensPerSecond),
      bytes_red_token: finite(estimate.networkBytesPerOutputToken, 0),
      detalle: estimate.detail,
    })),
  );
}

function finite(value: number, digits = 2): number | string {
  if (!Number.isFinite(value)) return "-";
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}
