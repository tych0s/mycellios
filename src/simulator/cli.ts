import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  simulateNetwork,
  type SimulationResult,
  type SimulationScenario,
} from "./model.js";

const args = parseArguments(process.argv.slice(2));
const scenarios: SimulationScenario[] = args.scenario
  ? [args.scenario]
  : ["normal", "large_model_spike", "region_outage", "churn_20"];
const results = scenarios.map((scenario) =>
  simulateNetwork({ nodes: args.nodes, users: args.users, seed: args.seed, scenario }),
);

if (args.output) {
  writeFileSync(resolve(args.output), `${JSON.stringify(results, null, 2)}\n`, "utf8");
}

if (args.json) {
  console.log(JSON.stringify(results, null, 2));
} else {
  printSummary(results);
}

function parseArguments(values: string[]): {
  nodes: number;
  users: number;
  seed: number;
  scenario?: SimulationScenario;
  output?: string;
  json: boolean;
} {
  const parsed: {
    nodes: number;
    users: number;
    seed: number;
    scenario?: SimulationScenario;
    output?: string;
    json: boolean;
  } = { nodes: 1_000, users: 1_000, seed: 42, json: false };
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    const next = values[index + 1];
    if (value === "--nodes" && next) parsed.nodes = positiveInteger(next, "nodes");
    else if (value === "--users" && next) parsed.users = positiveInteger(next, "users");
    else if (value === "--seed" && next) parsed.seed = positiveInteger(next, "seed");
    else if (value === "--scenario" && next) parsed.scenario = scenario(next);
    else if (value === "--output" && next) parsed.output = next;
    else if (value === "--json") parsed.json = true;
    else continue;
    if (value !== "--json") index += 1;
  }
  return parsed;
}

function scenario(value: string): SimulationScenario {
  const allowed: SimulationScenario[] = [
    "normal",
    "large_model_spike",
    "region_outage",
    "churn_20",
  ];
  if (!allowed.includes(value as SimulationScenario)) {
    throw new Error(`Unknown scenario ${value}. Expected one of: ${allowed.join(", ")}`);
  }
  return value as SimulationScenario;
}

function positiveInteger(value: string, label: string): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) throw new Error(`${label} must be positive`);
  return parsed;
}

function printSummary(results: SimulationResult[]): void {
  console.log("Simulación de GPU Distribuida (modelo de planificación, no benchmark real)\n");
  console.table(
    results.map((result) => ({
      escenario: result.options.scenario,
      nodos_online: result.inventory.onlineNodes,
      rutas_glm: result.inventory.glmRoutes,
      aceptados: result.network.accepted,
      cola: result.network.queued,
      aceptacion_pct: Math.round(result.network.acceptanceRate * 10_000) / 100,
      glm_tps_p50: result.traffic.glm.tokensPerSecond.p50,
      potencia_kw: result.network.estimatedPowerKw,
      chat_6_turnos_min: result.network.typicalSixTurnMinutes,
    })),
  );
}
