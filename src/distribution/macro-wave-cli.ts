import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { z } from "zod";
import { canonicalJson, sha256Text } from "../core/json.js";
import {
  MacroWaveRamVramPlanner,
  type MacroWavePlannerOptions,
  type MacroWavePlanningResult,
  type MacroWaveRouteCost,
} from "./macro-wave.js";
import type {
  DistributedModelProfile,
  DistributionTopology,
  DistributionWorkload,
  MacroWavePlanContractV1,
  MacroWaveStageExecutionContractV1,
} from "./types.js";

export type MacroWaveCliFormat = "json" | "markdown";

export interface MacroWaveCliInput {
  schema: "gdlp-macro-wave-input/1";
  model: DistributedModelProfile;
  topology: DistributionTopology;
  workload: DistributionWorkload;
  plannerOptions?: MacroWavePlannerOptions;
}

export interface MacroWaveCliArguments {
  inputPath: string | null;
  format: MacroWaveCliFormat;
  help: boolean;
}

export interface MacroWaveCliDocument {
  schema: "gdlp-macro-wave-report/1";
  inputSha256: string;
  modelId: string;
  feasible: boolean;
  reason: string | null;
  selectedAlternative: MacroWaveRouteCost["kind"] | null;
  plan: {
    algorithm: string;
    codec: string;
    microBatchSize: number;
    prefillChunkTokens: number;
    macroWave: MacroWavePlanContractV1 | null;
    stages: Array<{
      nodeId: string;
      layerStart: number;
      layerEnd: number;
      macroWave: MacroWaveStageExecutionContractV1 | null;
    }>;
  } | null;
  metrics: MacroWaveCliMetrics;
  traffic: {
    networkBytesPerOutputToken: number | null;
    rawActiveWeightBytesPerOutputToken: number | null;
    expectedWeightCacheMissBytesPerOutputToken: number | null;
    expectedWeightCacheHitRate: number | null;
  };
  stages: MacroWaveCliStage[];
  alternatives: Array<{
    kind: MacroWaveRouteCost["kind"];
    feasible: boolean;
    reason: string | null;
    stages: number;
    metrics: MacroWaveCliMetrics;
  }>;
  discardReasons: Array<{
    alternative: "resident" | "macro-wave";
    reason: string;
    count: number;
  }>;
}

export interface MacroWaveCliMetrics {
  ttftMs: number | null;
  tpotMs: number | null;
  responseTimeMs: number | null;
  pathDecodeMs: number | null;
  pipelineCycleMs: number | null;
  tokensPerSecondPerSequence: number | null;
  aggregateTokensPerSecond: number | null;
  routeAvailability: number | null;
}

export interface MacroWaveCliStage {
  index: number;
  nodeId: string;
  layerStart: number;
  layerEnd: number;
  mode: string;
  selectedBecause: string;
  ram: {
    requiredBytes: number;
    usableBytes: number;
  };
  vram: {
    requiredBytes: number;
    usableBytes: number;
    activationBufferBytes: number;
    expertWorkspaceBytes: number;
  };
  weights: {
    totalBytes: number;
    residentBytes: number;
    replicatedBytes: number;
    shardableExpertBytes: number;
    expertShardWorldSize: number | null;
    expertShardFraction: number | null;
    activeBytesPerWave: number;
    expectedCacheMissBytesPerWave: number;
    expectedCacheHitRate: number;
    largestTransferUnitBytes: number;
    bufferCopies: number;
    bufferBytes: number;
  };
  cost: {
    computeMsPerWave: number;
    loadMsPerWave: number;
    ramReadMsPerWave: number;
    pcieTransferMsPerWave: number;
    networkMsPerWave: number;
    serviceMsPerOutputToken: number;
  };
  link: {
    to: string;
    oneWayLatencyMs: number;
    rttMs: number;
    bandwidthMbps: number | null;
    wireBytesPerWave: number;
  };
  comparison: {
    resident: { feasible: boolean; reason: string | null; vramRequiredBytes: number };
    ramBacked: {
      feasible: boolean;
      reason: string | null;
      ramRequiredBytes: number;
      vramRequiredBytes: number;
    };
  };
}

const nonNegativeInteger = z.number().int().nonnegative().finite();
const positiveInteger = z.number().int().positive().finite();
const probability = z.number().min(0).max(1).finite();
const positiveFinite = z.number().positive().finite();
const nonNegativeFinite = z.number().nonnegative().finite();

const layerSchema = z
  .object({
    index: nonNegativeInteger,
    weightBytes: nonNegativeInteger,
    activationElements: positiveInteger,
    kvBytesPerToken: nonNegativeInteger,
    decodeMsAtUnit: positiveFinite,
    prefillMsPerTokenAtUnit: positiveFinite,
    largestResidentTensorBytes: nonNegativeInteger.optional(),
    macroWave: z
      .object({
        activeWeightBytesPerWave: nonNegativeInteger.optional(),
        largestTransferUnitBytes: nonNegativeInteger,
        expertWorkspaceBytesPerPosition: positiveInteger.optional(),
      })
      .strict()
      .optional(),
    expertParallel: z
      .object({
        expertWeightBytes: nonNegativeInteger,
        expertCount: positiveInteger.optional(),
        expertsPerToken: positiveInteger.optional(),
      })
      .strict()
      .optional(),
  })
  .strict();

const modelSchema = z
  .object({
    id: z.string().min(1),
    layers: z.array(layerSchema).min(1),
    embeddingBytes: nonNegativeInteger,
    lmHeadBytes: nonNegativeInteger,
    largestEmbeddingTensorBytes: nonNegativeInteger.optional(),
    largestLmHeadTensorBytes: nonNegativeInteger.optional(),
    tiedEmbeddingAndHead: z.boolean().optional(),
    runtimeOverheadBytesPerStage: nonNegativeInteger,
    embeddingDecodeMsAtUnit: nonNegativeFinite,
    lmHeadDecodeMsAtUnit: nonNegativeFinite,
    embeddingPrefillMsPerTokenAtUnit: nonNegativeFinite,
    lmHeadPrefillMsPerTokenAtUnit: nonNegativeFinite,
  })
  .strict();

const ramVramSchema = z
  .object({
    usableRamBytes: positiveInteger,
    usableVramBytes: positiveInteger,
    ramBandwidthGBps: positiveFinite,
    pcieBandwidthGBps: positiveFinite,
    residentKind: z.enum(["layers", "expert-shard"]).optional(),
    expertShard: z
      .object({
        worldSize: positiveInteger,
        fraction: z.number().positive().max(1).finite(),
      })
      .strict()
      .optional(),
  })
  .strict();

const nodeSchema = z
  .object({
    id: z.string().min(1),
    region: z.string().min(1),
    memoryBytes: nonNegativeInteger,
    reserveBytes: nonNegativeInteger,
    decodeScale: positiveFinite,
    prefillScale: positiveFinite,
    codecScale: positiveFinite,
    batchGain: nonNegativeFinite,
    maxBatchSpeedup: positiveFinite,
    powerWatts: nonNegativeFinite,
    availability: probability,
    ramVram: ramVramSchema.optional(),
  })
  .strict();

const linkSchema = z
  .object({
    from: z.string().min(1),
    to: z.string().min(1),
    oneWayLatencyMs: nonNegativeFinite,
    jitterP95Ms: nonNegativeFinite,
    bandwidthMbps: positiveFinite,
    lossRate: probability,
    availability: probability.optional(),
  })
  .strict();

const workloadSchema = z
  .object({
    promptTokens: positiveInteger,
    outputTokens: positiveInteger,
    contextTokens: positiveInteger,
    concurrentSequences: positiveInteger,
    maxStages: positiveInteger,
    maxQualityLoss: nonNegativeFinite,
    minRouteAvailability: probability,
    batchWindowMs: nonNegativeFinite,
    p95: z.boolean(),
  })
  .strict();

const codecSchema = z.enum(["fp16", "int8", "int8-grouped", "int8-hadamard", "q4"]);

const plannerOptionsSchema = z
  .object({
    beamWidth: positiveInteger.optional(),
    candidateCodecs: z.array(codecSchema).min(1).optional(),
    candidateMicroBatchSizes: z.array(positiveInteger).min(1).optional(),
    candidatePrefillChunks: z.array(positiveInteger).min(1).optional(),
    waveTokens: positiveInteger.optional(),
    expectedCommittedTokensPerWave: positiveFinite.optional(),
    verificationScalePerExtraToken: nonNegativeFinite.optional(),
    activationBufferCopies: positiveInteger.optional(),
    weightBufferCopies: positiveInteger.optional(),
    activationBytesPerElement: positiveFinite.optional(),
    transferSetupMsPerUnit: nonNegativeFinite.optional(),
    expectedWeightCacheHitRate: probability.optional(),
  })
  .strict();

const inputSchema = z
  .object({
    schema: z.literal("gdlp-macro-wave-input/1"),
    model: modelSchema,
    topology: z
      .object({
        nodes: z.array(nodeSchema).min(1),
        links: z.array(linkSchema),
      })
      .strict(),
    workload: workloadSchema,
    plannerOptions: plannerOptionsSchema.optional(),
  })
  .strict();

export function parseMacroWaveCliInput(value: unknown): MacroWaveCliInput {
  const parsed = inputSchema.safeParse(value);
  if (!parsed.success) {
    const issue = parsed.error.issues[0];
    const path = issue?.path.length ? issue.path.join(".") : "input";
    throw new Error(`invalid_macro_wave_input:${path}:${issue?.message ?? "unknown"}`);
  }
  return parsed.data as MacroWaveCliInput;
}

export function parseMacroWaveCliArguments(argv: string[]): MacroWaveCliArguments {
  let inputPath: string | null = null;
  let format: MacroWaveCliFormat = "json";
  let help = false;
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index]!;
    if (argument === "--help" || argument === "-h") {
      help = true;
      continue;
    }
    if (argument === "--format") {
      const value = argv[++index];
      if (value !== "json" && value !== "markdown") {
        throw new Error("macro_wave_format_must_be_json_or_markdown");
      }
      format = value;
      continue;
    }
    if (argument.startsWith("--format=")) {
      const value = argument.slice("--format=".length);
      if (value !== "json" && value !== "markdown") {
        throw new Error("macro_wave_format_must_be_json_or_markdown");
      }
      format = value;
      continue;
    }
    if (argument.startsWith("-")) throw new Error(`unknown_macro_wave_option:${argument}`);
    if (inputPath !== null) throw new Error("macro_wave_cli_accepts_one_input_file");
    inputPath = argument;
  }
  if (!help && inputPath === null) throw new Error("macro_wave_input_file_is_required");
  return { inputPath, format, help };
}

export function planMacroWaveCliDocument(inputValue: unknown): MacroWaveCliDocument {
  const input = parseMacroWaveCliInput(inputValue);
  const planner = new MacroWaveRamVramPlanner(input.plannerOptions);
  const result = planner.evaluate(input.model, input.topology, input.workload);
  return buildCliDocument(input, result);
}

export async function executeMacroWaveCli(
  argv: string[],
  cwd = process.cwd(),
): Promise<string> {
  const parsed = parseMacroWaveCliArguments(argv);
  if (parsed.help) return macroWaveCliUsage();
  const source = await readFile(resolve(cwd, parsed.inputPath!), "utf8");
  let input: unknown;
  try {
    input = JSON.parse(source);
  } catch (error) {
    throw new Error(
      `invalid_macro_wave_json:${error instanceof Error ? error.message : String(error)}`,
    );
  }
  const document = planMacroWaveCliDocument(input);
  return parsed.format === "markdown"
    ? renderMacroWaveMarkdown(document)
    : `${JSON.stringify(document, null, 2)}\n`;
}

export function renderMacroWaveMarkdown(document: MacroWaveCliDocument): string {
  const lines = [
    `# MacroWave RAM+VRAM — ${escapeMarkdown(document.modelId)}`,
    "",
    `- Viable: **${document.feasible ? "sí" : "no"}**`,
    `- Alternativa elegida: **${document.selectedAlternative ?? "ninguna"}**`,
    `- Motivo: ${escapeMarkdown(document.reason ?? "ok")}`,
    `- Hit esperado de caché de pesos (entrada, no garantía): **${formatPercent(document.traffic.expectedWeightCacheHitRate)}**`,
    `- Entrada: \`${document.inputSha256}\``,
    "",
    "## Métricas",
    "",
    "| TTFT | TPOT | tok/s usuario | tok/s total | Pesos activos/token | Miss RAM/PCIe/token | Red/token |",
    "|---:|---:|---:|---:|---:|---:|---:|",
    `| ${formatMs(document.metrics.ttftMs)} | ${formatMs(document.metrics.tpotMs)} | ${formatNumber(document.metrics.tokensPerSecondPerSequence)} | ${formatNumber(document.metrics.aggregateTokensPerSecond)} | ${formatBytes(document.traffic.rawActiveWeightBytesPerOutputToken)} | ${formatBytes(document.traffic.expectedWeightCacheMissBytesPerOutputToken)} | ${formatBytes(document.traffic.networkBytesPerOutputToken)} |`,
    "",
    "## Etapas",
    "",
    "| # | Nodo | Capas | Modo | RAM requerida/útil | VRAM requerida/útil | Buffer pesos | Activos/miss por onda | Cómputo/onda | Carga/onda | Red/onda | RTT | Servicio/token |",
    "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ];
  if (document.stages.length === 0) {
    lines.push("| - | - | - | - | - | - | - | - | - | - | - | - | - |");
  } else {
    for (const stage of document.stages) {
      lines.push(
        `| ${stage.index} | ${escapeMarkdown(stage.nodeId)} | ${stage.layerStart}-${stage.layerEnd - 1} | ${stage.mode} | ${formatBytes(stage.ram.requiredBytes)} / ${formatBytes(stage.ram.usableBytes)} | ${formatBytes(stage.vram.requiredBytes)} / ${formatBytes(stage.vram.usableBytes)} | ${stage.weights.bufferCopies} × ${formatBytes(stage.weights.largestTransferUnitBytes)} = ${formatBytes(stage.weights.bufferBytes)} | ${formatBytes(stage.weights.activeBytesPerWave)} / ${formatBytes(stage.weights.expectedCacheMissBytesPerWave)} | ${formatMs(stage.cost.computeMsPerWave)} | ${formatMs(stage.cost.loadMsPerWave)} | ${formatMs(stage.cost.networkMsPerWave)} | ${formatMs(stage.link.rttMs)} | ${formatMs(stage.cost.serviceMsPerOutputToken)} |`,
      );
    }
  }
  lines.push(
    "",
    "## Comparación",
    "",
    "| Alternativa | Viable | Etapas | TTFT | TPOT | tok/s usuario | Razón |",
    "|---|---|---:|---:|---:|---:|---|",
  );
  for (const alternative of document.alternatives) {
    lines.push(
      `| ${alternative.kind} | ${alternative.feasible ? "sí" : "no"} | ${alternative.stages} | ${formatMs(alternative.metrics.ttftMs)} | ${formatMs(alternative.metrics.tpotMs)} | ${formatNumber(alternative.metrics.tokensPerSecondPerSequence)} | ${escapeMarkdown(alternative.reason ?? "ok")} |`,
    );
  }
  lines.push(
    "",
    "## Razones de descarte",
    "",
    "| Alternativa | Razón | Ocurrencias |",
    "|---|---|---:|",
  );
  if (document.discardReasons.length === 0) {
    lines.push("| - | ninguna | 0 |");
  } else {
    for (const rejection of document.discardReasons) {
      lines.push(
        `| ${rejection.alternative} | ${escapeMarkdown(rejection.reason)} | ${rejection.count} |`,
      );
    }
  }
  return `${lines.join("\n")}\n`;
}

export function macroWaveCliUsage(): string {
  return [
    "Uso: npm run plan:macro-wave -- <perfil.json> [--format json|markdown]",
    "",
    "El JSON debe usar schema gdlp-macro-wave-input/1.",
    "",
  ].join("\n");
}

function buildCliDocument(
  input: MacroWaveCliInput,
  result: MacroWavePlanningResult,
): MacroWaveCliDocument {
  const selected = result.selected;
  return {
    schema: "gdlp-macro-wave-report/1",
    inputSha256: sha256Text(canonicalJson(input)),
    modelId: input.model.id,
    feasible: result.feasible,
    reason: result.reason,
    selectedAlternative: selected?.kind ?? null,
    plan: result.plan
      ? {
          algorithm: result.plan.algorithm,
          codec: result.plan.codec,
          microBatchSize: result.plan.microBatchSize,
          prefillChunkTokens: result.plan.prefillChunkTokens,
          macroWave: result.plan.macroWave
            ? structuredClone(result.plan.macroWave)
            : null,
          stages: result.plan.stages.map((stage) => ({
            nodeId: stage.nodeId,
            layerStart: stage.layerStart,
            layerEnd: stage.layerEnd,
            macroWave: stage.macroWave ? structuredClone(stage.macroWave) : null,
          })),
        }
      : null,
    metrics: routeMetrics(selected),
    traffic: {
      networkBytesPerOutputToken: finiteOrNull(selected?.networkBytesPerOutputToken),
      rawActiveWeightBytesPerOutputToken: finiteOrNull(
        selected?.rawActiveWeightBytesPerOutputToken,
      ),
      expectedWeightCacheMissBytesPerOutputToken: finiteOrNull(
        selected?.expectedWeightCacheMissBytesPerOutputToken,
      ),
      expectedWeightCacheHitRate:
        input.plannerOptions?.expectedWeightCacheHitRate ?? 0,
    },
    stages:
      selected?.stages.map((stage, index) => {
        const mode = stage.selectedMode === "ram-backed" ? stage.ramBacked : stage.resident;
        return {
          index,
          nodeId: stage.nodeId,
          layerStart: stage.layerStart,
          layerEnd: stage.layerEnd,
          mode: stage.selectedMode,
          selectedBecause: stage.selectedBecause,
          ram: {
            requiredBytes: mode.hostRamRequiredBytes,
            usableBytes: mode.usableRamBytes,
          },
          vram: {
            requiredBytes: mode.vramRequiredBytes,
            usableBytes: mode.usableVramBytes,
            activationBufferBytes: mode.activationBufferBytes,
            expertWorkspaceBytes: mode.expertWorkspaceBytes,
          },
          weights: {
            totalBytes: mode.totalWeightBytes,
            residentBytes: mode.residentWeightBytes,
            replicatedBytes: mode.replicatedWeightBytes,
            shardableExpertBytes: mode.shardableExpertWeightBytes,
            expertShardWorldSize: mode.expertShardWorldSize,
            expertShardFraction: mode.expertShardFraction,
            activeBytesPerWave: mode.activeWeightBytesPerWave,
            expectedCacheMissBytesPerWave: mode.expectedCacheMissWeightBytesPerWave,
            expectedCacheHitRate: mode.expectedWeightCacheHitRate,
            largestTransferUnitBytes: mode.largestTransferUnitBytes,
            bufferCopies: mode.weightBufferCopies,
            bufferBytes: mode.weightBufferBytes,
          },
          cost: {
            computeMsPerWave: stage.computeMsPerWave,
            loadMsPerWave: mode.loadMsPerWave,
            ramReadMsPerWave: mode.ramReadMsPerWave,
            pcieTransferMsPerWave: mode.pcieTransferMsPerWave,
            networkMsPerWave: stage.outgoing.transferMsPerWave,
            serviceMsPerOutputToken: stage.serviceMsPerOutputToken,
          },
          link: {
            to: stage.outgoing.to,
            oneWayLatencyMs: stage.outgoing.oneWayLatencyMs,
            rttMs: stage.outgoing.rttMs,
            bandwidthMbps: finiteOrNull(stage.outgoing.bandwidthMbps),
            wireBytesPerWave: stage.outgoing.wireBytesPerWave,
          },
          comparison: {
            resident: {
              feasible: stage.resident.feasible,
              reason: stage.resident.reason,
              vramRequiredBytes: stage.resident.vramRequiredBytes,
            },
            ramBacked: {
              feasible: stage.ramBacked.feasible,
              reason: stage.ramBacked.reason,
              ramRequiredBytes: stage.ramBacked.hostRamRequiredBytes,
              vramRequiredBytes: stage.ramBacked.vramRequiredBytes,
            },
          },
        };
      }) ?? [],
    alternatives: [result.alternatives.resident, result.alternatives.macroWave].map(
      (route) => ({
        kind: route.kind,
        feasible: route.feasible,
        reason: route.reason,
        stages: route.plan?.stages.length ?? 0,
        metrics: routeMetrics(route),
      }),
    ),
    discardReasons: [
      ...result.alternatives.resident.rejectionBreakdown.map((reason) => ({
        alternative: "resident" as const,
        ...reason,
      })),
      ...result.alternatives.macroWave.rejectionBreakdown.map((reason) => ({
        alternative: "macro-wave" as const,
        ...reason,
      })),
    ],
  };
}

function routeMetrics(route: MacroWaveRouteCost | null | undefined): MacroWaveCliMetrics {
  return {
    ttftMs: finiteOrNull(route?.ttftMs),
    tpotMs: finiteOrNull(route?.tpotMs),
    responseTimeMs: finiteOrNull(route?.responseTimeMs),
    pathDecodeMs: finiteOrNull(route?.pathDecodeMs),
    pipelineCycleMs: finiteOrNull(route?.pipelineCycleMs),
    tokensPerSecondPerSequence: finiteOrNull(route?.tokensPerSecondPerSequence),
    aggregateTokensPerSecond: finiteOrNull(route?.aggregateTokensPerSecond),
    routeAvailability: finiteOrNull(route?.routeAvailability),
  };
}

function finiteOrNull(value: number | undefined): number | null {
  return value !== undefined && Number.isFinite(value) ? value : null;
}

function formatMs(value: number | null): string {
  return value === null ? "-" : `${formatNumber(value)} ms`;
}

function formatNumber(value: number | null, digits = 2): string {
  if (value === null) return "-";
  return value.toFixed(digits).replace(/\.00$/, "");
}

function formatBytes(value: number | null): string {
  if (value === null) return "-";
  if (value >= 1024 ** 3) return `${formatNumber(value / 1024 ** 3)} GiB`;
  if (value >= 1024 ** 2) return `${formatNumber(value / 1024 ** 2)} MiB`;
  if (value >= 1024) return `${formatNumber(value / 1024)} KiB`;
  return `${formatNumber(value)} B`;
}

function formatPercent(value: number | null): string {
  return value === null ? "-" : `${formatNumber(value * 100)} %`;
}

function escapeMarkdown(value: string): string {
  return value.replace(/\|/g, "\\|").replace(/[\r\n]+/g, " ");
}

async function main(): Promise<void> {
  try {
    process.stdout.write(await executeMacroWaveCli(process.argv.slice(2)));
  } catch (error) {
    process.stderr.write(
      `macro-wave-cli: ${error instanceof Error ? error.message : String(error)}\n`,
    );
    process.exitCode = 2;
  }
}

const invokedPath = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : null;
if (invokedPath === import.meta.url) await main();
