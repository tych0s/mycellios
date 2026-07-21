import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  DEFAULT_WAN_OPTIONS,
  WAN_PROJECTION_KIND,
  estimateDefaultWanRoutes,
  estimateFleet,
  type WanSimulationOptions,
} from "./wan-model.js";

const args = parseArguments(process.argv.slice(2));
const options: WanSimulationOptions = {
  ...DEFAULT_WAN_OPTIONS,
  promptTokens: args.promptTokens,
  outputTokens: args.outputTokens,
  contextTokensPerChat: args.contextTokens,
  useSpeculativeDecoding: !args.noSpeculation,
  activationCompressionRatio: args.noCompression
    ? 1
    : DEFAULT_WAN_OPTIONS.activationCompressionRatio,
};
const routes = estimateDefaultWanRoutes(options);
const regional = routes.find((route) => route.id === "regional-fiber")!;
const fleet = [10, 100, 1_000].map((nodes) => estimateFleet(nodes, regional));
const result = {
  kind: WAN_PROJECTION_KIND,
  assumptions: options,
  routes,
  fleet,
};

if (args.output) {
  writeFileSync(resolve(args.output), `${JSON.stringify(result, null, 2)}\n`, "utf8");
}
if (args.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log("Theoretical WAN model for GLM-4.5-Air Q4 (not a real benchmark)\n");
  console.table(
    routes.map((route) => ({
      route: route.label,
      nodos_fisicos: route.physicalNodes,
      etapas_virtuales: route.virtualStages,
      speculation: route.speculation.appliedP50 ? "yes" : "AR fallback",
      ttft_p50_s: route.ttftMs.p50 / 1_000,
      ttft_p95_s: route.ttftMs.p95 / 1_000,
      tok_s_p50: route.decodeTokensPerSecond.p50,
      tok_s_slow_p95: route.decodeTokensPerSecond.slowPathP95,
      respuesta_200t_p50_s: route.responseSeconds.p50,
      respuesta_200t_p95_s: route.responseSeconds.p95,
      chats_4t_s: route.loaded.recommendedInteractiveChats,
      tok_s_agregados: route.loaded.aggregateTokensPerSecond,
      exito_sin_repuesto_pct: route.reliability.requestSuccessPct,
      electricidad_eur_M: route.economics.electricityEuroPerMillionOutputTokensAtSaturation,
    })),
  );
  console.log("\nMemory bound for the residential mix if every route were strong\n");
  console.table(
    fleet.map((entry) => ({
      nodos: entry.nodes,
      online: entry.expectedOnlineNodes,
      memoria_util_TB: entry.usableMemoryTb,
      rutas_interactivas: entry.interactiveRegionalRoutes,
      rutas_4GB_batch: entry.fourGbBatchRoutes,
      cota_chats_4t_s_rutas_fuertes: entry.interactiveGlmChatsAtTarget,
    })),
  );
}

function parseArguments(values: string[]): {
  promptTokens: number;
  outputTokens: number;
  contextTokens: number;
  noSpeculation: boolean;
  noCompression: boolean;
  json: boolean;
  output?: string;
} {
  const parsed: {
    promptTokens: number;
    outputTokens: number;
    contextTokens: number;
    noSpeculation: boolean;
    noCompression: boolean;
    json: boolean;
    output?: string;
  } = {
    promptTokens: 800,
    outputTokens: 200,
    contextTokens: 4_000,
    noSpeculation: false,
    noCompression: false,
    json: false,
  };
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    const next = values[index + 1];
    if (value === "--prompt-tokens" && next) parsed.promptTokens = positiveInteger(next, value);
    else if (value === "--output-tokens" && next) parsed.outputTokens = positiveInteger(next, value);
    else if (value === "--context-tokens" && next) parsed.contextTokens = positiveInteger(next, value);
    else if (value === "--output" && next) parsed.output = next;
    else if (value === "--no-speculation") parsed.noSpeculation = true;
    else if (value === "--no-compression") parsed.noCompression = true;
    else if (value === "--json") parsed.json = true;
    else continue;
    if (!["--no-speculation", "--no-compression", "--json"].includes(value)) index += 1;
  }
  return parsed;
}

function positiveInteger(value: string, label: string): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) throw new Error(`${label} must be positive`);
  return parsed;
}
