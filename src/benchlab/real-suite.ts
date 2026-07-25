import { execFile } from "node:child_process";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { hostname } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { sha256Text } from "../core/json.js";
import { probeHardware, type HardwareProbe } from "../worker/hardware.js";
import { sealBenchmarkScenario } from "./scenario.js";
import {
  BENCHMARK_RUN_SCHEMA,
  emptyComparison,
  type BenchmarkDeviceProfile,
  type BenchmarkMeasurement,
  type BenchmarkRun,
} from "./types.js";
import type { RunIdentity } from "./history.js";

const execFileAsync = promisify(execFile);

export interface RealRuntimeOptions {
  cwd: string;
  baseUrl?: string;
  coordinatorUrl?: string;
  concurrencies?: number[];
  iterations?: number;
  warmups?: number;
  outputTokens?: number;
  prompt?: string;
  timeoutSeconds?: number;
}

interface RuntimeHealth {
  status: string;
  error: string | null;
  model: string;
  artifact_identity: string;
  canonical_model_source: string;
  canonical_model_revision: string | null;
  pipeline_snapshot_identity: string;
  stages: number;
  boundaries: number[];
  codec: string;
  [key: string]: unknown;
}

interface ApiBenchmarkSample {
  prompt_tokens: number;
  completion_tokens: number;
  response_ms: number;
  server_token_ttft_ms: number;
  server_token_tpot_ms: number;
  text_nonempty: boolean;
}

interface ApiBenchmarkRow {
  concurrency: number;
  measured_requests: number;
  actual_completion_tokens: number;
  aggregate_actual_tok_s: number;
  per_request_actual_tok_s_mean: number;
  server_token_ttft_mean_ms: number;
  server_token_ttft_p95_ms: number;
  server_token_tpot_mean_ms: number;
  server_token_tpot_p95_ms: number;
  nonempty: number;
  request_samples: ApiBenchmarkSample[];
}

export interface ApiBenchmarkDocument {
  schema_version: 2;
  /** The second value is read-only compatibility for archived benchmark evidence. */
  kind:
    | "mycellios_api_continuous_scheduler"
    | "openai_api_continuous_scheduler";
  configuration: {
    base_url: string;
    model: string;
    output_tokens: number;
    warm_batches_per_scenario: number;
    measured_batches_per_scenario: number;
    prompt_digest?: string;
  };
  rows: ApiBenchmarkRow[];
  server_after_measurement: RuntimeHealth;
}

interface CoordinatorSnapshot {
  summary: {
    registered: number;
    connected: number;
    online: number;
    offeredVramMb: number;
  };
  workers: Array<{ id: string; status: string; connected: boolean }>;
}

export async function runRealRuntimeSuite(
  identity: RunIdentity,
  options: RealRuntimeOptions,
): Promise<BenchmarkRun> {
  const startedAt = new Date().toISOString();
  const baseUrl = options.baseUrl ?? (await discoverRuntimeUrl(options.cwd));
  const health = await fetchRuntimeHealth(baseUrl);
  const concurrencies = options.concurrencies ?? [1, 2];
  const iterations = options.iterations ?? 3;
  const warmups = options.warmups ?? 1;
  const outputTokens = options.outputTokens ?? 12;
  const benchmark = await executeApiBenchmark({
    ...options,
    baseUrl,
    model: health.model,
    concurrencies,
    iterations,
    warmups,
    outputTokens,
  });
  const [hardware, coordinator] = await Promise.all([
    probeHardware(),
    fetchCoordinatorSnapshot(options.coordinatorUrl ?? "http://127.0.0.1:4180"),
  ]);
  return buildRealBenchmarkRun(identity, benchmark, hardware, coordinator, startedAt);
}

export function buildRealBenchmarkRun(
  identity: RunIdentity,
  benchmark: ApiBenchmarkDocument,
  hardware: HardwareProbe,
  coordinator: CoordinatorSnapshot | null,
  startedAt = new Date().toISOString(),
): BenchmarkRun {
  validateBenchmarkDocument(benchmark);
  const health = benchmark.server_after_measurement;
  const profiles = realDeviceProfiles(hardware, health.codec);
  const remoteConnected = coordinator?.summary.connected ?? 0;
  const measurements = benchmark.rows.map((row) =>
    rowToMeasurement(row, benchmark, health, profiles, hardware, coordinator, remoteConnected),
  );
  return {
    schema: BENCHMARK_RUN_SCHEMA,
    ...identity,
    startedAt,
    finishedAt: new Date().toISOString(),
    suite: "real-runtime",
    status: measurements.every((measurement) => measurement.status !== "failed")
      ? "baseline"
      : "failed",
    measurements,
  };
}

async function executeApiBenchmark(options: RealRuntimeOptions & {
  baseUrl: string;
  model: string;
  concurrencies: number[];
  iterations: number;
  warmups: number;
  outputTokens: number;
}): Promise<ApiBenchmarkDocument> {
  const python = distributionPython(options.cwd);
  const pythonPath = join(options.cwd, "python");
  const { stdout } = await execFileAsync(
    python,
    [
      "-m", "distributed_runtime.api_benchmark",
      "--base-url", options.baseUrl,
      "--model", options.model,
      "--concurrencies", options.concurrencies.join(","),
      "--iterations", String(options.iterations),
      "--warmups", String(options.warmups),
      "--output-tokens", String(options.outputTokens),
      "--prompt", options.prompt ?? "Explica en una frase que hace una GPU. Muestra {request}.",
      "--timeout-seconds", String(options.timeoutSeconds ?? 180),
    ],
    {
      cwd: options.cwd,
      windowsHide: true,
      encoding: "utf8",
      timeout: (options.timeoutSeconds ?? 180) * 1_000 * Math.max(2, options.concurrencies.length),
      maxBuffer: 32 * 1024 * 1024,
      env: {
        ...process.env,
        PYTHONPATH: pythonPath,
        HF_HOME: join(options.cwd, "runtime", "hf-cache"),
        TOKENIZERS_PARALLELISM: "false",
        PYTHONIOENCODING: "utf-8",
      },
    },
  );
  const document = JSON.parse(stdout) as ApiBenchmarkDocument;
  document.configuration.prompt_digest = sha256Text(
    options.prompt ?? "Explica en una frase que hace una GPU. Muestra {request}.",
  );
  return document;
}

async function discoverRuntimeUrl(cwd: string): Promise<string> {
  const configured = process.env.MYCELLIOS_BENCHMARK_URL?.trim();
  const candidates = configured
    ? [configured]
    : [
        ...configuredAutoDistributionUrls(cwd),
        "http://127.0.0.1:8082",
        "http://127.0.0.1:8081",
      ];
  for (const candidate of candidates) {
    try {
      await fetchRuntimeHealth(candidate);
      return candidate;
    } catch {
      // Try the next local physical runtime.
    }
  }
  throw new Error(
    `No hay un runtime físico GDLP activo en ${Array.from(new Set(candidates)).join(", ")}. No se han generado datos.`,
  );
}

function configuredAutoDistributionUrls(cwd: string): string[] {
  const directory = join(cwd, "config");
  let files: string[];
  try {
    files = readdirSync(directory).filter((file) => file.startsWith("auto-distribute") && file.endsWith(".json"));
  } catch {
    return [];
  }
  return files.flatMap((file) => {
    try {
      const document = JSON.parse(readFileSync(join(directory, file), "utf8")) as {
        runtime?: { apiEndpoint?: { host?: unknown; port?: unknown } };
      };
      const host = document.runtime?.apiEndpoint?.host;
      const port = document.runtime?.apiEndpoint?.port;
      return typeof host === "string" && Number.isSafeInteger(port)
        ? [`http://${host}:${String(port)}`]
        : [];
    } catch {
      return [];
    }
  });
}

async function fetchRuntimeHealth(baseUrl: string): Promise<RuntimeHealth> {
  const response = await fetch(new URL("health", `${baseUrl.replace(/\/$/, "")}/`), {
    signal: AbortSignal.timeout(5_000),
  });
  if (!response.ok) throw new Error(`El runtime respondió HTTP ${response.status}.`);
  const health = (await response.json()) as RuntimeHealth;
  if (health.status !== "ready" || health.error !== null || !health.model || !health.artifact_identity) {
    throw new Error("El runtime no está READY o no ofrece identidad física verificable.");
  }
  return health;
}

async function fetchCoordinatorSnapshot(baseUrl: string): Promise<CoordinatorSnapshot | null> {
  try {
    const response = await fetch(new URL("public/v1/snapshot", `${baseUrl.replace(/\/$/, "")}/`), {
      signal: AbortSignal.timeout(3_000),
    });
    if (!response.ok) return null;
    return (await response.json()) as CoordinatorSnapshot;
  } catch {
    return null;
  }
}

function rowToMeasurement(
  row: ApiBenchmarkRow,
  benchmark: ApiBenchmarkDocument,
  health: RuntimeHealth,
  profiles: BenchmarkDeviceProfile[],
  hardware: HardwareProbe,
  coordinator: CoordinatorSnapshot | null,
  remoteConnected: number,
): BenchmarkMeasurement {
  const samples = row.request_samples;
  const successful = row.nonempty === row.measured_requests && row.measured_requests > 0;
  const requestTps = samples.map((sample) => sample.completion_tokens / (sample.response_ms / 1_000));
  return sealBenchmarkScenario({
    id: `runtime-${health.artifact_identity.slice(0, 18)}-c${row.concurrency}`,
    title: `${health.model} real · concurrencia ${row.concurrency}`,
    description: "Peticiones reales por la API Mycellios y el pipeline GDLP/2 activo.",
    evidence: "loopback",
    environment: "local-loopback",
    model: {
      id: health.canonical_model_source,
      label: health.model,
      revision: health.canonical_model_revision,
      digest: health.artifact_identity,
      precision: health.codec,
    },
    inventory: {
      totalDevices: 1 + remoteConnected,
      connectedDevices: 1 + remoteConnected,
      selectedDevices: 1,
      profiles,
    },
    topology: {
      digest: health.pipeline_snapshot_identity,
      stageCount: health.stages,
      boundaries: health.boundaries.slice(),
      nodeIds: [hardware.hostname || hostname()],
      routeClasses: [],
    },
    workload: {
      promptTokens: Math.round(median(samples.map((sample) => sample.prompt_tokens))),
      outputTokens: benchmark.configuration.output_tokens,
      concurrentSequences: row.concurrency,
      promptDigest: benchmark.configuration.prompt_digest ?? null,
      requests: row.measured_requests,
    },
    metrics: {
      tokensPerSecond: round(row.per_request_actual_tok_s_mean, 3),
      tokensPerSecondP95: round(percentile(requestTps, 0.95), 3),
      aggregateTokensPerSecond: round(row.aggregate_actual_tok_s, 3),
      ttftMsP50: round(percentile(samples.map((sample) => sample.server_token_ttft_ms), 0.5), 2),
      ttftMsP95: round(row.server_token_ttft_p95_ms, 2),
      tpotMsP50: round(percentile(samples.map((sample) => sample.server_token_tpot_ms), 0.5), 2),
      tpotMsP95: round(row.server_token_tpot_p95_ms, 2),
      acceptanceRate: row.measured_requests === 0 ? 0 : row.nonempty / row.measured_requests,
      energyWhPerToken: null,
    },
    status: successful ? "baseline" : "failed",
    comparison: emptyComparison(),
    notes: [
      "MEDICIÓN REAL LOCAL: no es una simulación ni una proyección.",
      `Endpoint ${benchmark.configuration.base_url}; ${row.measured_requests} peticiones medidas y ${row.actual_completion_tokens} tokens reales.`,
      `${health.stages} etapas sobre el host ${hardware.hostname || hostname()}; ejecución PyTorch CPU; GPU detectada pero no usada por este runtime.`,
      `Coordinador: ${coordinator?.summary.connected ?? 0} workers remotos conectados de ${coordinator?.summary.registered ?? 0} registrados.`,
      `Artefacto ${health.artifact_identity}; snapshot ${health.pipeline_snapshot_identity}.`,
    ],
  });
}

function realDeviceProfiles(hardware: HardwareProbe, precision: string): BenchmarkDeviceProfile[] {
  return [
    {
      label: `CPU host activo · ${hardware.hostname}`,
      kind: "cpu",
      count: 1,
      memoryGb: round(hardware.ramMb / 1024, 1),
      backend: "pytorch-cpu",
      precision,
    },
    ...hardware.gpus.map((gpu) => ({
      label: `GPU detectada, no usada · ${gpu.model}`,
      kind: "gpu" as const,
      count: 1,
      // Only expose the dedicated memory reported by the OS. The Windows UMA
      // shared-memory ceiling in probeHardware is an estimate, not a measurement.
      memoryGb: round(gpu.physicalVramMb / 1024, 1),
    })),
  ];
}

function distributionPython(cwd: string): string {
  const candidates = process.platform === "win32"
    ? [join(cwd, "runtime", "distribution-venv", "Scripts", "python.exe")]
    : [join(cwd, "runtime", "distribution-venv", "bin", "python")];
  const found = candidates.find(existsSync);
  if (!found) throw new Error("Falta runtime/distribution-venv; no se han generado datos.");
  return found;
}

function validateBenchmarkDocument(value: ApiBenchmarkDocument): void {
  if (
    value.schema_version !== 2
    || (
      value.kind !== "mycellios_api_continuous_scheduler"
      && value.kind !== "openai_api_continuous_scheduler"
    )
  ) {
    throw new Error("La salida del benchmark físico no tiene un esquema compatible.");
  }
  if (!Array.isArray(value.rows) || value.rows.length === 0) {
    throw new Error("El benchmark físico no produjo mediciones.");
  }
}

function median(values: number[]): number {
  return percentile(values, 0.5);
}

function percentile(values: number[], fraction: number): number {
  if (values.length === 0) return 0;
  const ordered = values.slice().sort((left, right) => left - right);
  const position = (ordered.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return ordered[lower]!;
  const weight = position - lower;
  return ordered[lower]! * (1 - weight) + ordered[upper]! * weight;
}

function round(value: number, decimals: number): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}
