import type { ModelDeployment } from "../contracts/types.js";
import type { StoredWorker } from "../storage/store.js";
import {
  BENCHMARK_RUN_SCHEMA,
  emptyComparison,
  type BenchmarkDeviceProfile,
  type BenchmarkInventory,
  type BenchmarkMeasurement,
  type BenchmarkRun,
} from "./types.js";
import {
  compareRunWithHistory,
  createRunIdentity,
  loadBenchmarkRuns,
  saveBenchmarkRun,
  type RunIdentity,
} from "./history.js";

export interface CoordinatorBenchmarkModel {
  id: string;
  source: string;
  revision: string | null;
}

export interface CoordinatorBenchmarkOptions {
  cwd: string;
  coordinatorUrl: string;
  model: CoordinatorBenchmarkModel;
  inventory(routedWorkerIds: ReadonlySet<string>): BenchmarkInventory;
  resolveWorkerId?(jobId: string): string | null;
  networkToken?: string;
  samples?: number;
  outputTokens?: number;
  timeoutMs?: number;
  prompt?: string;
  version?: string;
  label?: string;
  historyDirectory?: string;
  trigger?: "automatic-model-start" | "manual";
  fetchImpl?: typeof fetch;
}

interface CoordinatorCompletion {
  id: string;
  model: string;
  choices: Array<{ message?: { content?: string } }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  x_network: {
    route_class: string;
    affinity_hit: boolean;
    ttft_ms: number;
    active_ms: number;
    reused_kv_tokens: number;
  };
}

interface CoordinatorSample {
  promptTokens: number;
  outputTokens: number;
  ttftMs: number;
  activeMs: number;
  latencyMs: number;
  routeClass: string;
  workerId: string | null;
}

export async function runAndPersistCoordinatorSuite(
  options: CoordinatorBenchmarkOptions,
): Promise<{ run: BenchmarkRun; path: string }> {
  const history = loadBenchmarkRuns(options.cwd, options.historyDirectory);
  const identity = createRunIdentity(options.cwd, options.version, options.label);
  const run = compareRunWithHistory(
    await runCoordinatorModelSuite(identity, options),
    history,
  );
  const path = saveBenchmarkRun(options.cwd, run, options.historyDirectory);
  return { run, path };
}

export async function runCoordinatorModelSuite(
  identity: RunIdentity,
  options: CoordinatorBenchmarkOptions,
): Promise<BenchmarkRun> {
  const startedAt = new Date().toISOString();
  const sampleCount = boundedInteger(options.samples, 3, 1, 10);
  const outputTokens = boundedInteger(options.outputTokens, 24, 4, 256);
  const timeoutMs = boundedInteger(options.timeoutMs, 120_000, 1_000, 3_600_000);
  const fetchImpl = options.fetchImpl ?? fetch;
  const samples: CoordinatorSample[] = [];
  const failures: string[] = [];
  const routedWorkerIds = new Set<string>();

  for (let index = 0; index < sampleCount; index += 1) {
    try {
      const sample = await requestCoordinatorSample({
        fetchImpl,
        coordinatorUrl: options.coordinatorUrl,
        modelId: options.model.id,
        outputTokens,
        timeoutMs,
        prompt: options.prompt ?? defaultPrompt(outputTokens),
        sampleIndex: index,
        ...(options.networkToken ? { networkToken: options.networkToken } : {}),
        ...(options.resolveWorkerId ? { resolveWorkerId: options.resolveWorkerId } : {}),
      });
      samples.push(sample);
      if (sample.workerId) routedWorkerIds.add(sample.workerId);
    } catch (error) {
      failures.push(error instanceof Error ? error.message : String(error));
    }
  }

  const inventory = options.inventory(routedWorkerIds);
  const measurement = buildCoordinatorMeasurement(
    options.model,
    inventory,
    samples,
    failures,
    sampleCount,
    outputTokens,
  );
  return {
    schema: BENCHMARK_RUN_SCHEMA,
    ...identity,
    startedAt,
    finishedAt: new Date().toISOString(),
    suite: "real-runtime",
    trigger: options.trigger ?? "manual",
    triggerModelId: options.model.id,
    status: measurement.status,
    measurements: [measurement],
  };
}

export function detectNewActiveModels(
  activeModelIds: ReadonlySet<string>,
  observedActiveModelIds: Set<string>,
): string[] {
  const started: string[] = [];
  for (const modelId of activeModelIds) {
    if (observedActiveModelIds.has(modelId)) continue;
    observedActiveModelIds.add(modelId);
    started.push(modelId);
  }
  for (const modelId of [...observedActiveModelIds]) {
    if (!activeModelIds.has(modelId)) observedActiveModelIds.delete(modelId);
  }
  return started;
}

export function buildCoordinatorBenchmarkInventory(
  modelId: string,
  workers: readonly StoredWorker[],
  connectedWorkerIds: ReadonlySet<string>,
  routedWorkerIds: ReadonlySet<string> = new Set(),
): BenchmarkInventory {
  const connectedWorkers = workers.filter((worker) => connectedWorkerIds.has(worker.id));
  const eligibleWorkers = connectedWorkers.filter((worker) =>
    worker.capabilities.deployments.some((deployment) => deployment.model === modelId)
  );
  const routedWorkers = routedWorkerIds.size > 0
    ? eligibleWorkers.filter((worker) => routedWorkerIds.has(worker.id))
    : eligibleWorkers;
  const roots = routedWorkers.length > 0 ? routedWorkers : eligibleWorkers;
  const profiles = participantProfiles(modelId, roots, connectedWorkers);

  return {
    totalDevices: workers.length,
    connectedDevices: connectedWorkers.length,
    selectedDevices: profiles.reduce((sum, profile) => sum + profile.count, 0),
    profiles,
    physicalMemoryGb: nullableSum(profiles.map((profile) => profile.physicalMemoryGb)),
    offeredMemoryGb: nullableSum(profiles.map((profile) => profile.offeredMemoryGb)),
    observedPowerWatts: nullableSum(profiles.map((profile) => profile.observedPowerWatts)),
    powerLimitWatts: nullableSum(profiles.map((profile) => profile.powerLimitWatts)),
  };
}

function participantProfiles(
  modelId: string,
  roots: readonly StoredWorker[],
  connectedWorkers: readonly StoredWorker[],
): BenchmarkDeviceProfile[] {
  const byNodeId = new Map(
    connectedWorkers
      .filter((worker) => worker.capabilities.distributedExecutor?.nodeId)
      .map((worker) => [worker.capabilities.distributedExecutor!.nodeId, worker] as const),
  );
  const stageProfiles = roots.flatMap((worker) =>
    worker.capabilities.deployments
      .filter((deployment) => deployment.model === modelId)
      .flatMap((deployment) =>
        deployment.execution?.stages?.map((stage) =>
          profileForStage(stage, byNodeId.get(stage.nodeId))
        ) ?? []
      )
  );
  if (stageProfiles.length > 0) return uniqueProfiles(stageProfiles);

  return uniqueProfiles(roots.flatMap((worker) =>
    worker.capabilities.deployments
      .filter((deployment) => deployment.model === modelId)
      .slice(0, 1)
      .map((deployment) => profileForDeployment(worker, deployment))
  ));
}

function profileForStage(
  stage: NonNullable<NonNullable<ModelDeployment["execution"]>["stages"]>[number],
  worker: StoredWorker | undefined,
): BenchmarkDeviceProfile {
  const gpu = stage.deviceType === "gpu" ? matchingGpu(worker, stage.deviceName) : undefined;
  return profileFromTelemetry({
    nodeId: stage.nodeId,
    label: `${stage.deviceName} · etapa ${stage.stageIndex + 1}`,
    kind: stage.deviceType,
    backend: stage.backend,
    precision: stage.precision,
    worker,
    gpu,
  });
}

function profileForDeployment(
  worker: StoredWorker,
  deployment: ModelDeployment,
): BenchmarkDeviceProfile {
  const execution = deployment.execution;
  const kind = execution?.deviceType === "gpu" || execution?.deviceType === "mixed" ? "gpu" : "cpu";
  const gpu = kind === "gpu" ? matchingGpu(worker, execution?.deviceName) : undefined;
  return profileFromTelemetry({
    nodeId: worker.capabilities.distributedExecutor?.nodeId ?? worker.id,
    label: execution?.deviceName ?? gpu?.model ?? `CPU · ${worker.id}`,
    kind,
    backend: execution?.backend ?? (kind === "gpu" ? worker.capabilities.distributedExecutor?.acceleration?.backend : "cpu"),
    precision: execution?.precision ?? null,
    worker,
    gpu,
  });
}

function profileFromTelemetry(input: {
  nodeId: string;
  label: string;
  kind: "gpu" | "cpu";
  backend: string | null | undefined;
  precision: string | null | undefined;
  worker: StoredWorker | undefined;
  gpu: StoredWorker["capabilities"]["gpus"][number] | undefined;
}): BenchmarkDeviceProfile {
  const physicalMemoryGb = input.gpu ? round(input.gpu.physicalVramMb / 1_024, 2) : null;
  const offeredMemoryGb = input.gpu ? round(input.gpu.offeredVramMb / 1_024, 2) : null;
  return {
    nodeId: input.nodeId,
    label: input.label,
    kind: input.kind,
    count: 1,
    memoryGb: offeredMemoryGb,
    backend: input.backend ?? null,
    precision: input.precision ?? null,
    physicalMemoryGb,
    offeredMemoryGb,
    observedPowerWatts: finiteOrNull(input.gpu?.powerW),
    powerLimitWatts: finiteOrNull(input.worker?.capabilities.limits.maxPowerW),
    utilizationPct: finiteOrNull(input.gpu?.utilizationPct),
    temperatureC: finiteOrNull(input.gpu?.temperatureC),
  };
}

function matchingGpu(
  worker: StoredWorker | undefined,
  deviceName?: string,
): StoredWorker["capabilities"]["gpus"][number] | undefined {
  if (!worker) return undefined;
  const normalized = deviceName?.toLowerCase();
  return worker.capabilities.gpus.find((gpu) =>
    normalized && (normalized.includes(gpu.model.toLowerCase()) || gpu.model.toLowerCase().includes(normalized))
  ) ?? worker.capabilities.gpus[0];
}

function uniqueProfiles(profiles: BenchmarkDeviceProfile[]): BenchmarkDeviceProfile[] {
  const unique = new Map<string, BenchmarkDeviceProfile>();
  for (const profile of profiles) unique.set(profile.nodeId ?? profile.label, profile);
  return [...unique.values()];
}

async function requestCoordinatorSample(input: {
  fetchImpl: typeof fetch;
  coordinatorUrl: string;
  modelId: string;
  networkToken?: string;
  outputTokens: number;
  timeoutMs: number;
  prompt: string;
  sampleIndex: number;
  resolveWorkerId?: (jobId: string) => string | null;
}): Promise<CoordinatorSample> {
  const started = performance.now();
  const response = await input.fetchImpl(
    new URL("/v1/chat/completions", input.coordinatorUrl),
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(input.networkToken ? { authorization: `Bearer ${input.networkToken}` } : {}),
      },
      body: JSON.stringify({
        model: input.modelId,
        messages: [{ role: "user", content: input.prompt }],
        stream: false,
        max_tokens: input.outputTokens,
        temperature: 0,
        top_p: 1,
        seed: 10_000 + input.sampleIndex,
        session_id: `benchmark-${Date.now()}-${input.sampleIndex}`,
        workload_class: "benchmark",
        deadline_ms: input.timeoutMs,
      }),
      signal: AbortSignal.timeout(input.timeoutMs + 2_000),
    },
  );
  const latencyMs = performance.now() - started;
  const raw = await response.text();
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${errorMessage(raw)}`);
  }
  const completion = parseCoordinatorCompletion(raw);
  return {
    promptTokens: completion.usage.prompt_tokens,
    outputTokens: completion.usage.completion_tokens,
    ttftMs: completion.x_network.ttft_ms,
    activeMs: completion.x_network.active_ms,
    latencyMs,
    routeClass: completion.x_network.route_class,
    workerId: input.resolveWorkerId?.(completion.id) ?? null,
  };
}

function buildCoordinatorMeasurement(
  model: CoordinatorBenchmarkModel,
  inventory: BenchmarkInventory,
  samples: CoordinatorSample[],
  failures: string[],
  requestedSamples: number,
  requestedOutputTokens: number,
): BenchmarkMeasurement {
  const throughput = samples.map((sample) =>
    sample.outputTokens > 0 && sample.activeMs > 0
      ? sample.outputTokens / (sample.activeMs / 1_000)
      : 0
  );
  const tpot = samples.map((sample) =>
    Math.max(0, sample.activeMs - sample.ttftMs) / Math.max(1, sample.outputTokens - 1)
  );
  const routeClasses = [...new Set(samples.map((sample) => sample.routeClass))];
  const precision = [...new Set(
    inventory.profiles.map((profile) => profile.precision).filter((value): value is string => Boolean(value)),
  )].join(" + ") || "runtime";
  const successful = samples.length;
  const durationMs = samples.reduce((sum, sample) => sum + sample.latencyMs, 0);
  const status = successful === requestedSamples && successful > 0 ? "baseline" : "failed";
  return {
    id: `startup-${safeId(model.id)}-n${inventory.selectedDevices}-o${requestedOutputTokens}`,
    title: `${model.id} · arranque real`,
    description: "Prueba automática sobre la ruta OpenAI-compatible del coordinador y el modelo recién activado.",
    evidence: "physical",
    environment: "lan",
    model: {
      id: model.source,
      label: model.id,
      revision: model.revision,
      precision,
    },
    inventory,
    workload: {
      promptTokens: Math.round(median(samples.map((sample) => sample.promptTokens))),
      outputTokens: samples.reduce((sum, sample) => sum + sample.outputTokens, 0),
      concurrentSequences: 1,
      requests: requestedSamples,
      successfulRequests: successful,
      durationMs: round(durationMs, 1),
      routeClasses,
    },
    metrics: {
      tokensPerSecond: samples.length > 0 ? round(mean(throughput), 3) : null,
      tokensPerSecondP95: samples.length > 0 ? round(percentile(throughput, 0.95), 3) : null,
      aggregateTokensPerSecond:
        samples.length > 0 && durationMs > 0
          ? round(samples.reduce((sum, sample) => sum + sample.outputTokens, 0) / (durationMs / 1_000), 3)
          : null,
      ttftMsP50: samples.length > 0 ? round(percentile(samples.map((sample) => sample.ttftMs), 0.5), 2) : null,
      ttftMsP95: samples.length > 0 ? round(percentile(samples.map((sample) => sample.ttftMs), 0.95), 2) : null,
      tpotMsP50: samples.length > 0 ? round(percentile(tpot, 0.5), 2) : null,
      tpotMsP95: samples.length > 0 ? round(percentile(tpot, 0.95), 2) : null,
      latencyMsP50: samples.length > 0 ? round(percentile(samples.map((sample) => sample.latencyMs), 0.5), 2) : null,
      latencyMsP95: samples.length > 0 ? round(percentile(samples.map((sample) => sample.latencyMs), 0.95), 2) : null,
      acceptanceRate: requestedSamples === 0 ? 0 : successful / requestedSamples,
      energyWhPerToken: null,
    },
    status,
    comparison: emptyComparison(),
    notes: [
      "PRUEBA FÍSICA REAL: las peticiones atravesaron el coordinador y un runtime conectado; no hay datos simulados.",
      `${successful}/${requestedSamples} peticiones completadas; ${routeClasses.length > 0 ? `rutas ${routeClasses.join(", ")}` : "sin ruta completada"}.`,
      `${inventory.selectedDevices} nodos usados o integrantes del pipeline; ${inventory.connectedDevices}/${inventory.totalDevices} conectados al iniciar la prueba.`,
      inventory.observedPowerWatts === null || inventory.observedPowerWatts === undefined
        ? "La potencia no fue reportada por el controlador; no se ha estimado."
        : `${round(inventory.observedPowerWatts, 1)} W observados en el último heartbeat físico de los nodos participantes.`,
      ...failures.map((failure) => `Fallo real: ${failure}`),
    ],
  };
}

function parseCoordinatorCompletion(raw: string): CoordinatorCompletion {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    throw new Error("El coordinador devolvió una respuesta que no es JSON.");
  }
  if (!value || typeof value !== "object") throw new Error("La respuesta del coordinador está vacía.");
  const completion = value as Partial<CoordinatorCompletion>;
  if (
    typeof completion.id !== "string"
    || !completion.usage
    || !completion.x_network
    || !Number.isFinite(completion.usage.completion_tokens)
    || !Number.isFinite(completion.x_network.ttft_ms)
    || !Number.isFinite(completion.x_network.active_ms)
  ) {
    throw new Error("La respuesta no contiene métricas físicas verificables.");
  }
  return completion as CoordinatorCompletion;
}

function errorMessage(raw: string): string {
  try {
    const parsed = JSON.parse(raw) as { error?: { message?: unknown; code?: unknown } };
    return String(parsed.error?.message ?? parsed.error?.code ?? raw);
  } catch {
    return raw.slice(0, 240) || "respuesta vacía";
  }
}

function defaultPrompt(outputTokens: number): string {
  return `Prueba de rendimiento mycellios. Responde en una sola línea con aproximadamente ${outputTokens} tokens sobre cómo medir la inferencia ayuda a mejorarla.`;
}

function finiteOrNull(value: number | undefined): number | null {
  return value !== undefined && Number.isFinite(value) ? value : null;
}

function nullableSum(values: Array<number | null | undefined>): number | null {
  const present = values.filter((value): value is number => value !== null && value !== undefined && Number.isFinite(value));
  return present.length > 0 ? round(present.reduce((sum, value) => sum + value, 0), 2) : null;
}

function mean(values: number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
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
  return ordered[lower]! * (1 - (position - lower)) + ordered[upper]! * (position - lower);
}

function boundedInteger(value: number | undefined, fallback: number, minimum: number, maximum: number): number {
  if (value === undefined) return fallback;
  return Math.min(maximum, Math.max(minimum, Math.round(value)));
}

function round(value: number, decimals: number): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function safeId(value: string): string {
  return value.toLowerCase().replaceAll(/[^a-z0-9._-]/g, "-").slice(0, 90);
}
