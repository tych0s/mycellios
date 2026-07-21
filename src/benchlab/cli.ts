import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  compareRunWithHistory,
  createRunIdentity,
  loadBenchmarkRuns,
  saveBenchmarkRun,
} from "./history.js";
import { importPhysicalCampaign } from "./physical-import.js";
import { runRealRuntimeSuite, type RealRuntimeOptions } from "./real-suite.js";
import type { BenchmarkRun } from "./types.js";

export interface BenchmarkRunOptions extends Omit<RealRuntimeOptions, "cwd"> {
  cwd: string;
  version?: string;
  label?: string;
  historyDirectory?: string;
}

export async function runAndPersistRealSuite(options: BenchmarkRunOptions): Promise<{
  run: BenchmarkRun;
  path: string;
}> {
  const history = loadBenchmarkRuns(options.cwd, options.historyDirectory);
  const identity = createRunIdentity(options.cwd, options.version, options.label);
  const run = compareRunWithHistory(await runRealRuntimeSuite(identity, options), history);
  const path = saveBenchmarkRun(options.cwd, run, options.historyDirectory);
  return { run, path };
}

function importAndPersistPhysicalSuite(
  options: BenchmarkRunOptions,
  observationPath: string,
  configPath: string,
): { run: BenchmarkRun; path: string } {
  const history = loadBenchmarkRuns(options.cwd, options.historyDirectory);
  const identity = createRunIdentity(options.cwd, options.version, options.label);
  const observation = JSON.parse(readFileSync(resolve(options.cwd, observationPath), "utf8")) as unknown;
  const config = JSON.parse(readFileSync(resolve(options.cwd, configPath), "utf8")) as unknown;
  const run = compareRunWithHistory(importPhysicalCampaign(identity, observation, config), history);
  const path = saveBenchmarkRun(options.cwd, run, options.historyDirectory);
  return { run, path };
}

if (isEntryPoint()) {
  void (async () => {
    try {
      const args = parseArguments(process.argv.slice(2));
      const result = args.command === "import-physical"
        ? importAndPersistPhysicalSuite(args, args.observationPath!, args.configPath!)
        : await runAndPersistRealSuite(args);
      if (args.json) console.log(JSON.stringify(result.run, null, 2));
      else printSummary(result.run, result.path);
      if (result.run.status === "failed" || result.run.status === "regression") {
        process.exitCode = 1;
      }
    } catch (error) {
      console.error(error instanceof Error ? error.message : String(error));
      process.exitCode = 1;
    }
  })();
}

interface ParsedArguments extends BenchmarkRunOptions {
  command: "run" | "import-physical";
  json: boolean;
  observationPath: string | null;
  configPath: string | null;
}

function parseArguments(values: string[]): ParsedArguments {
  const command = values[0] === "import-physical" ? "import-physical" : "run";
  const args: ParsedArguments = {
    command,
    cwd: process.cwd(),
    json: false,
    observationPath: null,
    configPath: null,
  };
  const start = command === "run" && values[0] === "run" ? 1 : command === "import-physical" ? 1 : 0;
  for (let index = start; index < values.length; index += 1) {
    const value = values[index]!;
    const next = values[index + 1];
    if (value === "--version" && next) args.version = next;
    else if (value === "--label" && next) args.label = next;
    else if (value === "--history" && next) args.historyDirectory = next;
    else if (value === "--base-url" && next) args.baseUrl = next;
    else if (value === "--coordinator-url" && next) args.coordinatorUrl = next;
    else if (value === "--concurrencies" && next) args.concurrencies = positiveCsv(next);
    else if (value === "--iterations" && next) args.iterations = positiveInteger(next, "iterations");
    else if (value === "--warmups" && next) args.warmups = nonnegativeInteger(next, "warmups");
    else if (value === "--output-tokens" && next) args.outputTokens = positiveInteger(next, "output-tokens");
    else if (value === "--observation" && next) args.observationPath = next;
    else if (value === "--config" && next) args.configPath = next;
    else if (value === "--json") {
      args.json = true;
      continue;
    } else {
      throw new Error(`Argumento desconocido: ${value}`);
    }
    index += 1;
  }
  if (command === "import-physical" && (!args.observationPath || !args.configPath)) {
    throw new Error("import-physical requiere --observation y --config.");
  }
  return args;
}

function printSummary(run: BenchmarkRun, path: string): void {
  console.log(`\nBanco de pruebas mycellios · ${run.label}`);
  console.log(`Estado: ${run.status} · ${run.measurements.length} escenarios`);
  console.table(
    run.measurements.map((measurement) => ({
      test: measurement.id,
      evidencia: measurement.evidence,
      dispositivos: `${measurement.inventory.connectedDevices}/${measurement.inventory.totalDevices}`,
      modelo: measurement.model.label,
      "tokens/s": measurement.metrics.tokensPerSecond,
      "TTFT P95 ms": measurement.metrics.ttftMsP95,
      estado: measurement.status,
    })),
  );
  console.log(`Guardado en ${path}`);
}

function positiveCsv(value: string): number[] {
  const parsed = value.split(",").map((item) => positiveInteger(item.trim(), "concurrencies"));
  if (parsed.length === 0) throw new Error("concurrencies no puede estar vacío.");
  return parsed;
}

function positiveInteger(value: string, label: string): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isSafeInteger(parsed) || parsed < 1) throw new Error(`${label} debe ser positivo.`);
  return parsed;
}

function nonnegativeInteger(value: string, label: string): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isSafeInteger(parsed) || parsed < 0) throw new Error(`${label} no puede ser negativo.`);
  return parsed;
}

function isEntryPoint(): boolean {
  const entry = process.argv[1];
  if (!entry) return false;
  return import.meta.url === new URL(`file:///${entry.replaceAll("\\", "/")}`).href;
}
