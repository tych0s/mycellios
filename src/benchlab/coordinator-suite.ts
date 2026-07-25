import type { ModelDeployment } from "../contracts/types.js";
import type { NativeBuildIdentity } from "../contracts/build-identity.js";
import { sha256CanonicalEvidence, sha256Text } from "../core/json.js";
import type { StoredWorker } from "../storage/store.js";
import {
  sealBenchmarkActivation,
  type BenchmarkActivation,
} from "./activation-tracker.js";
import {
  runAdaptiveBenchmarkCampaign,
  type AdaptiveBenchmarkCampaignResult,
} from "./campaign.js";
import {
  PhysicalTelemetryWindow,
  type BenchmarkPhysicalTelemetrySummary,
  type BenchmarkTelemetrySnapshot,
} from "./physical-telemetry.js";
import { sealBenchmarkScenario } from "./scenario.js";
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
import type { NetworkExecutionTrace } from "../contracts/types.js";
import { parseNetworkExecutionTrace } from "../telemetry/network-execution-trace.js";

export interface CoordinatorBenchmarkModel {
  id: string;
  source: string;
  revision: string | null;
  digest: string | null;
}

export function coordinatorBenchmarkModelIdentity(
  model: Omit<CoordinatorBenchmarkModel, "digest">,
  workers: readonly StoredWorker[],
  connectedWorkerIds: ReadonlySet<string>,
): CoordinatorBenchmarkModel {
  const digests = new Set(
    workers
      .filter((worker) => connectedWorkerIds.has(worker.id))
      .flatMap((worker) => worker.capabilities.deployments)
      .filter((deployment) => deployment.model === model.id && deployment.adapter !== "mock")
      .map((deployment) => deployment.modelDigest.trim())
      .filter(Boolean),
  );
  return {
    ...model,
    // Conflicting or absent declarations are not a model identity.
    digest: digests.size === 1 ? [...digests][0]! : null,
  };
}

export function coordinatorBenchmarkActivations(
  modelIds: ReadonlySet<string>,
  workers: readonly StoredWorker[],
  connectedWorkerIds: ReadonlySet<string>,
  expectedBuildIdentity?: NativeBuildIdentity | null,
): BenchmarkActivation[] {
  if (expectedBuildIdentity === null) return [];
  const connectedWorkers = workers.filter((worker) =>
    connectedWorkerIds.has(worker.id) && worker.status === "online"
  );
  const activations: BenchmarkActivation[] = [];
  for (const modelId of [...modelIds].sort()) {
    const participants = connectedWorkers.flatMap((worker) =>
      worker.capabilities.buildIdentity?.version
        === worker.capabilities.agentVersion
        && (
          expectedBuildIdentity === undefined
          || (
            worker.capabilities.buildIdentity.schema === expectedBuildIdentity.schema
            && worker.capabilities.buildIdentity.version === expectedBuildIdentity.version
            && worker.capabilities.buildIdentity.sourceId === expectedBuildIdentity.sourceId
          )
        )
        ?
      worker.capabilities.deployments
        .filter((deployment) =>
          deployment.model === modelId && deployment.adapter !== "mock"
        )
        .map((deployment) => {
          const stageRanges = deployment.execution?.stages?.map((stage) => ({
            nodeId: stage.nodeId,
            stageIndex: stage.stageIndex,
            layerStart: stage.layerStart,
            layerEnd: stage.layerEnd,
          })) ?? (deployment.stage
            ? [{
                nodeId: worker.capabilities.distributedExecutor?.nodeId ?? worker.id,
                stageIndex: deployment.stage.index,
                layerStart: deployment.stage.layerStart,
                layerEnd: deployment.stage.layerEnd,
              }]
            : []);
          return {
            workerId: worker.id,
            agentVersion: worker.capabilities.agentVersion,
            buildSourceId: worker.capabilities.buildIdentity!.sourceId,
            deploymentId: deployment.deploymentId,
            modelDigest: deployment.modelDigest,
            nodeIds: stageRanges.length > 0
              ? stageRanges.map((stage) => stage.nodeId)
              : [worker.capabilities.distributedExecutor?.nodeId ?? worker.id],
            stageRanges,
          };
        })
        : []
    );
    if (participants.length === 0) continue;
    try {
      activations.push(sealBenchmarkActivation(modelId, participants));
    } catch {
      // Conflicting digests, invalid layer ranges or incomplete physical
      // identity must not produce a benchmark activation.
    }
  }
  return activations;
}

export interface CoordinatorBenchmarkOptions {
  cwd: string;
  coordinatorUrl: string;
  model: CoordinatorBenchmarkModel;
  inventory(routedWorkerIds: ReadonlySet<string>): BenchmarkInventory;
  resolveWorkerId?(jobId: string): string | null;
  networkToken?: string;
  buildIdentity?: NativeBuildIdentity | null;
  activation?: BenchmarkActivation;
  validateActivation?(
    activation: BenchmarkActivation,
    routedWorkerIds: ReadonlySet<string>,
  ): void;
  samples?: number;
  maximumSamples?: number;
  warmupSamples?: number;
  retriesPerSample?: number;
  retryDelayMs?: number;
  targetConfidenceHalfWidthPct?: number;
  outputTokens?: number;
  timeoutMs?: number;
  prompt?: string;
  version?: string;
  label?: string;
  historyDirectory?: string;
  trigger?: "automatic-model-start" | "manual";
  telemetrySnapshot?(): BenchmarkTelemetrySnapshot;
  telemetryIntervalMs?: number;
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
    execution_trace?: NetworkExecutionTrace | null;
  };
}

interface CoordinatorSample {
  sampleIndex: number;
  promptTokens: number;
  outputTokens: number;
  outputDigest: string;
  ttftMs: number;
  activeMs: number;
  latencyMs: number;
  routeClass: string;
  workerId: string | null;
  executionTrace: NetworkExecutionTrace | null;
}

export async function runAndPersistCoordinatorSuite(
  options: CoordinatorBenchmarkOptions,
): Promise<{ run: BenchmarkRun; path: string }> {
  const history = loadBenchmarkRuns(options.cwd, options.historyDirectory);
  const identity = createRunIdentity(options.cwd, options.version, options.label);
  if (options.buildIdentity) {
    if (options.buildIdentity.version !== identity.version) {
      throw new Error(
        `benchmark_build_version_mismatch:${options.buildIdentity.version}:${identity.version}`,
      );
    }
    identity.build.sourceId = options.buildIdentity.sourceId;
    identity.build.sourceIdSource = "runtime-local";
  }
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
  const minimumSamples = boundedInteger(options.samples, 7, 1, 100);
  const maximumSamples = boundedInteger(
    options.maximumSamples,
    options.samples === undefined ? 15 : minimumSamples,
    minimumSamples,
    250,
  );
  const warmupSamples = boundedInteger(options.warmupSamples, 1, 0, 10);
  const outputTokens = boundedInteger(options.outputTokens, 64, 4, 256);
  const timeoutMs = boundedInteger(options.timeoutMs, 120_000, 1_000, 3_600_000);
  const fetchImpl = options.fetchImpl ?? fetch;
  const routedWorkerIds = new Set<string>();
  const prompt = options.prompt ?? defaultPrompt(outputTokens);
  const telemetry = new PhysicalTelemetryWindow();
  const telemetryErrors: string[] = [];
  const observeTelemetry = (): void => {
    if (!options.telemetrySnapshot) return;
    try {
      telemetry.observe(options.telemetrySnapshot());
    } catch (error) {
      telemetryErrors.push(error instanceof Error ? error.message : String(error));
    }
  };
  observeTelemetry();
  const telemetryTimer = options.telemetrySnapshot
    ? setInterval(
        observeTelemetry,
        boundedInteger(options.telemetryIntervalMs, 1_000, 250, 30_000),
      )
    : null;
  telemetryTimer?.unref();

  let campaign: AdaptiveBenchmarkCampaignResult<CoordinatorSample>;
  try {
    campaign = await runAdaptiveBenchmarkCampaign({
      warmupSamples,
      minimumSamples,
      maximumSamples,
      retriesPerSample: boundedInteger(options.retriesPerSample, 2, 0, 10),
      retryDelayMs: boundedInteger(options.retryDelayMs, 5_000, 0, 60_000),
      targetConfidenceHalfWidthPct:
        finitePositive(options.targetConfidenceHalfWidthPct, 8),
      measure: async ({ phase, sampleIndex, attempt }) => {
        const sample = await requestCoordinatorSample({
          fetchImpl,
          coordinatorUrl: options.coordinatorUrl,
          modelId: options.model.id,
          outputTokens,
          timeoutMs,
          prompt,
          sampleIndex,
          phase,
          attempt,
          ...(options.networkToken ? { networkToken: options.networkToken } : {}),
          ...(options.resolveWorkerId ? { resolveWorkerId: options.resolveWorkerId } : {}),
        });
        return sample;
      },
      score: sampleTokensPerSecond,
      retryable: isRetryableBenchmarkAvailabilityError,
    });
  } finally {
    if (telemetryTimer) clearInterval(telemetryTimer);
    observeTelemetry();
  }
  for (const sample of campaign.samples) {
    if (sample.workerId) routedWorkerIds.add(sample.workerId);
  }
  if (options.activation) {
    const participantWorkerIds = new Set(
      options.activation.participants.map((participant) => participant.workerId),
    );
    for (const workerId of routedWorkerIds) {
      if (!participantWorkerIds.has(workerId)) {
        throw new Error(
          `benchmark_routed_outside_activation:${workerId}:${options.activation.activationId}`,
        );
      }
    }
    options.validateActivation?.(options.activation, routedWorkerIds);
  }

  const inventory = options.inventory(routedWorkerIds);
  const selectedNodeIds = new Set(
    inventory.profiles
      .map((profile) => profile.nodeId)
      .filter((nodeId): nodeId is string => Boolean(nodeId)),
  );
  const physicalTelemetry = telemetry.summarize(
    selectedNodeIds,
    campaign.samples.reduce((sum, sample) => sum + sample.outputTokens, 0),
  );
  const measurement = buildCoordinatorMeasurement(
    options.model,
    inventory,
    campaign,
    physicalTelemetry,
    telemetryErrors,
    minimumSamples,
    outputTokens,
    sha256Text(prompt),
    options.activation,
  );
  return {
    schema: BENCHMARK_RUN_SCHEMA,
    ...identity,
    build: {
      ...identity.build,
      participantSourceIds: [...new Set(
        options.activation?.participants.map(
          (participant) => participant.buildSourceId,
        ) ?? identity.build.participantSourceIds,
      )].sort(),
    },
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

export function buildCoordinatorBenchmarkTelemetrySnapshot(
  workers: readonly StoredWorker[],
  connectedWorkerIds: ReadonlySet<string>,
  atMs = Date.now(),
): BenchmarkTelemetrySnapshot {
  const nodes = new Map<string, BenchmarkTelemetrySnapshot["nodes"][number]>();
  for (const worker of workers) {
    if (!connectedWorkerIds.has(worker.id) || worker.status !== "online") continue;
    const nodeId = worker.capabilities.distributedExecutor?.nodeId ?? worker.id;
    if (nodes.has(nodeId)) continue;
    const gpus = worker.capabilities.gpus;
    const powers = gpus.map((gpu) => gpu.powerW);
    const utilizations = gpus.map((gpu) => gpu.utilizationPct);
    const temperatures = gpus
      .map((gpu) => gpu.temperatureC)
      .filter((value): value is number => value !== undefined && Number.isFinite(value));
    nodes.set(nodeId, {
      nodeId,
      powerWatts: powers.length > 0 && powers.every(isFiniteNumber)
        ? powers.reduce((sum, value) => sum + value, 0)
        : null,
      utilizationPct: utilizations.length > 0 && utilizations.every(isFiniteNumber)
        ? utilizations.reduce((sum, value) => sum + value, 0) / utilizations.length
        : null,
      temperatureC: temperatures.length > 0 ? Math.max(...temperatures) : null,
      offeredMemoryGb: gpus.length > 0
        ? gpus.reduce((sum, gpu) => sum + gpu.offeredVramMb, 0) / 1_024
        : null,
      freeOfferedMemoryGb: gpus.length > 0
        ? gpus.reduce((sum, gpu) => sum + gpu.freeOfferedVramMb, 0) / 1_024
        : null,
    });
  }
  return { atMs, nodes: [...nodes.values()] };
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
  phase: "warmup" | "measurement";
  attempt: number;
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
        seed: 10_000,
        session_id:
          `benchmark-${input.phase}-${input.sampleIndex}-${input.attempt}-${Date.now()}`,
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
  if (completion.model !== input.modelId) {
    throw new Error(
      `El runtime respondió como ${completion.model}; se esperaba ${input.modelId}.`,
    );
  }
  const outputText = completion.choices
    .map((choice) => choice.message?.content ?? "")
    .join("")
    .trim();
  if (!outputText || completion.usage.completion_tokens < 1) {
    throw new Error("El runtime no produjo una salida física verificable.");
  }
  return {
    sampleIndex: input.sampleIndex,
    promptTokens: completion.usage.prompt_tokens,
    outputTokens: completion.usage.completion_tokens,
    outputDigest: sha256Text(outputText),
    ttftMs: completion.x_network.ttft_ms,
    activeMs: completion.x_network.active_ms,
    latencyMs,
    routeClass: completion.x_network.route_class,
    workerId: input.resolveWorkerId?.(completion.id) ?? null,
    executionTrace: completion.x_network.execution_trace ?? null,
  };
}

function buildCoordinatorMeasurement(
  model: CoordinatorBenchmarkModel,
  inventory: BenchmarkInventory,
  campaign: AdaptiveBenchmarkCampaignResult<CoordinatorSample>,
  physicalTelemetry: BenchmarkPhysicalTelemetrySummary,
  telemetryErrors: readonly string[],
  minimumSamples: number,
  requestedOutputTokens: number,
  promptDigest: string,
  activation: BenchmarkActivation | undefined,
): BenchmarkMeasurement {
  const samples = campaign.samples;
  const throughput = samples.map((sample) =>
    sampleTokensPerSecond(sample)
  );
  const tpot = samples.map((sample) =>
    Math.max(0, sample.activeMs - sample.ttftMs) / Math.max(1, sample.outputTokens - 1)
  );
  const routeClasses = [...new Set(samples.map((sample) => sample.routeClass))];
  const precision = [...new Set(
    inventory.profiles.map((profile) => profile.precision).filter((value): value is string => Boolean(value)),
  )].join(" + ") || "runtime";
  const successful = samples.length;
  const failedMeasurementSlots = new Set(
    campaign.failures
      .filter((failure) => failure.phase === "measurement" && !failure.recovered)
      .map((failure) => failure.sampleIndex),
  ).size;
  const measurementSlotsAttempted = successful + failedMeasurementSlots;
  const recoveredFailures = campaign.failures.filter((failure) => failure.recovered).length;
  const durationMs = samples.reduce((sum, sample) => sum + sample.latencyMs, 0);
  const status = successful < minimumSamples
    ? "failed"
    : campaign.stable
      ? "baseline"
      : "inconclusive";
  const inventoryNodeIds = inventory.profiles
    .map((profile) => profile.nodeId)
    .filter((nodeId): nodeId is string => Boolean(nodeId));
  const networkTraces = samples.flatMap((sample) =>
    sample.executionTrace ? [sample.executionTrace] : []
  );
  const tracedNodeIds = networkTraces.flatMap((trace) =>
    trace.stages.flatMap((stage) => stage.nodeId ? [stage.nodeId] : [])
  );
  const topologyNodeIds = tracedNodeIds.length > 0
    ? tracedNodeIds
    : activation?.participants
    .flatMap((participant) => participant.nodeIds)
    ?? inventoryNodeIds;
  const tracedStageRanges = networkTraces.flatMap((trace) =>
    trace.stages.flatMap((stage) =>
      stage.layerStart !== null && stage.layerEnd !== null
        ? [{
            stageIndex: stage.stageIndex,
            layerStart: stage.layerStart,
            layerEnd: stage.layerEnd,
          }]
        : []
    )
  );
  const stageRanges = tracedStageRanges.length > 0
    ? tracedStageRanges
    : activation?.participants.flatMap(
    (participant) => participant.stageRanges,
  ) ?? [];
  const topologyBoundaries = [...new Set(
    stageRanges
      .flatMap((stage) => [stage.layerStart, stage.layerEnd])
      .filter((boundary) => boundary > 0),
  )].sort((left, right) => left - right);
  const deterministicConsistencyRate = modalDigestRate(
    samples.map((sample) => sample.outputDigest),
  );
  const requestSuccessRate = measurementSlotsAttempted > 0
    ? successful / measurementSlotsAttempted
    : 0;
  const environment = inventory.selectedDevices <= 1
    ? "local-loopback" as const
    : "unknown" as const;
  return sealBenchmarkScenario({
    id: `startup-${safeId(model.id)}-n${inventory.selectedDevices}-o${requestedOutputTokens}`,
    title: `${model.id} · arranque real`,
    description:
      "Campaña automática sobre la ruta real del coordinador, con warmup separado, reintentos registrados y control estadístico.",
    evidence: "physical",
    environment,
    model: {
      id: model.source,
      label: model.id,
      revision: model.revision,
      digest: model.digest,
      precision,
    },
    inventory,
    topology: {
      digest: networkTraces.length > 0
        ? benchmarkTraceTopologyDigest(networkTraces)
        : activation?.topologyDigest ?? null,
      stageCount: stageRanges.length > 0
        ? new Set(stageRanges.map((stage) => stage.stageIndex)).size
        : topologyNodeIds.length > 0
          ? topologyNodeIds.length
          : null,
      boundaries: topologyBoundaries,
      nodeIds: [...new Set(topologyNodeIds)].sort(),
      routeClasses,
    },
    workload: {
      promptTokens: Math.round(median(samples.map((sample) => sample.promptTokens))),
      outputTokens: requestedOutputTokens,
      concurrentSequences: 1,
      promptDigest,
      requests: minimumSamples,
      successfulRequests: successful,
      warmupRequests: campaign.warmups.length,
      recoveredFailures,
      observedOutputTokens: samples.reduce((sum, sample) => sum + sample.outputTokens, 0),
      statisticallyStable: campaign.stable,
      campaignStopReason: campaign.stoppedBecause,
      durationMs: round(durationMs, 1),
      routeClasses,
    },
    metrics: {
      tokensPerSecond:
        campaign.summary !== null ? round(campaign.summary.p50, 3) : null,
      tokensPerSecondP5:
        campaign.summary !== null ? round(campaign.summary.p5, 3) : null,
      tokensPerSecondP50:
        campaign.summary !== null ? round(campaign.summary.p50, 3) : null,
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
      coefficientOfVariationPct:
        campaign.summary !== null
          ? round(campaign.summary.coefficientOfVariationPct ?? 0, 3)
          : null,
      confidenceHalfWidthPct:
        campaign.summary !== null
          ? round(campaign.summary.confidenceHalfWidthPct ?? 0, 3)
          : null,
      requestSuccessRate,
      // A deterministic reference token sequence is not yet available for
      // every certified model. Keep exactness unknown instead of equating
      // HTTP success or repeated output with correctness.
      exactnessRate: null,
      deterministicConsistencyRate,
      speculativeAcceptanceRate: null,
      powerWattsP50: physicalTelemetry.powerWattsP50,
      powerWattsP95: physicalTelemetry.powerWattsP95,
      powerWattsPeak: physicalTelemetry.powerWattsPeak,
      energyWh: physicalTelemetry.energyWh,
      energyCoveragePct: physicalTelemetry.energyCoveragePct,
      utilizationPctP50: physicalTelemetry.utilizationPctP50,
      utilizationPctP95: physicalTelemetry.utilizationPctP95,
      temperatureCPeak: physicalTelemetry.temperatureCPeak,
      usedMemoryGbPeak: physicalTelemetry.usedMemoryGbPeak,
      acceptanceRate: requestSuccessRate,
      energyWhPerToken: physicalTelemetry.energyWhPerToken,
    },
    networkTraces,
    status,
    comparison: emptyComparison(),
    notes: [
      "PRUEBA FÍSICA REAL: las peticiones atravesaron el coordinador y un runtime conectado; no hay datos simulados y las muestras de warmup no cuentan como rendimiento.",
      `${successful} muestras calientes válidas (mínimo ${minimumSamples}); ${campaign.stable ? "intervalo de confianza suficiente" : "resultado demasiado variable para afirmar una mejora"}.`,
      `${campaign.warmups.length} warmups completados y ${recoveredFailures} fallos temporales recuperados sin ocultarlos.`,
      `${inventory.selectedDevices} nodos integrantes de la ruta observada; ${inventory.connectedDevices}/${inventory.totalDevices} conectados durante la campaña.`,
      activation
        ? `Activación ${activation.activationId}; topología ${activation.topologyDigest}; builds ${activation.participants.map((participant) => participant.buildSourceId).join(", ")}.`
        : "Ejecución manual sin activation ID sellado; la comparación exige el resto del fingerprint exacto.",
      networkTraces.length > 0
        ? `${networkTraces.length} trazas de ruta ligadas a peticiones completadas; transporte y bytes sólo aparecen cuando hubo evidencia efectiva no ambigua.`
        : "El coordinador no entregó una traza física de ruta; transporte, RTT y bytes quedan sin lectura.",
      physicalTelemetry.energyWh === null
        ? "La ventana no tuvo suficientes muestras físicas continuas para calcular energía; no se ha estimado."
        : `${round(physicalTelemetry.energyWh, 6)} Wh integrados con ${round(physicalTelemetry.energyCoveragePct, 1)}% de cobertura temporal.`,
      ...campaign.failures.map((failure) =>
        `${failure.recovered ? "Fallo recuperado" : "Fallo real"} (${failure.phase} ${failure.sampleIndex + 1}, intento ${failure.attempt + 1}): ${failure.message}`
      ),
      ...telemetryErrors.map((failure) => `Telemetría física rechazada: ${failure}`),
    ],
  });
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
    || typeof completion.model !== "string"
    || !Array.isArray(completion.choices)
    || !completion.usage
    || !completion.x_network
    || !Number.isFinite(completion.usage.prompt_tokens)
    || !Number.isFinite(completion.usage.completion_tokens)
    || !Number.isFinite(completion.x_network.ttft_ms)
    || !Number.isFinite(completion.x_network.active_ms)
  ) {
    throw new Error("La respuesta no contiene métricas físicas verificables.");
  }
  const traceValue = completion.x_network.execution_trace;
  const executionTrace = traceValue === undefined || traceValue === null
    ? null
    : parseNetworkExecutionTrace(traceValue);
  if (traceValue !== undefined && traceValue !== null && executionTrace === null) {
    throw new Error("La traza física de ejecución no supera la validación estricta.");
  }
  return {
    ...(completion as CoordinatorCompletion),
    x_network: {
      ...(completion.x_network as CoordinatorCompletion["x_network"]),
      execution_trace: executionTrace,
    },
  };
}

function benchmarkTraceTopologyDigest(
  traces: readonly NetworkExecutionTrace[],
): string {
  const identities = [...new Set(traces.map((trace) =>
    sha256CanonicalEvidence({
      routeClass: trace.routeClass,
      selectedRoute: trace.selectedRoute,
      stages: trace.stages.map((stage) => ({
        routeStageIndex: stage.routeStageIndex,
        stageIndex: stage.stageIndex,
        nodeId: stage.nodeId,
        workerId: stage.workerId,
        deploymentId: stage.deploymentId,
        deploymentOwnerWorkerId: stage.deploymentOwnerWorkerId,
        modelDigest: stage.modelDigest,
        layerStart: stage.layerStart,
        layerEnd: stage.layerEnd,
        deviceType: stage.deviceType,
        backend: stage.backend,
        precision: stage.precision,
        deviceName: stage.deviceName,
      })),
      boundaries: trace.boundaries.map((boundary) => ({
        fromStageIndex: boundary.fromStageIndex,
        toStageIndex: boundary.toStageIndex,
        sourceNodeId: boundary.sourceNodeId,
        destinationNodeId: boundary.destinationNodeId,
        physicalBoundary: boundary.physicalBoundary,
        transport: boundary.transport,
      })),
    })
  ))].sort();
  return sha256CanonicalEvidence({
    schema: "mycellios-benchmark-observed-topology/1",
    identities,
  });
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

function sampleTokensPerSecond(sample: CoordinatorSample): number {
  return sample.outputTokens > 0 && sample.activeMs > 0
    ? sample.outputTokens / (sample.activeMs / 1_000)
    : 0;
}

function modalDigestRate(digests: readonly string[]): number | null {
  if (digests.length === 0) return null;
  const counts = new Map<string, number>();
  for (const digest of digests) counts.set(digest, (counts.get(digest) ?? 0) + 1);
  return Math.max(...counts.values()) / digests.length;
}

function isRetryableBenchmarkAvailabilityError(error: unknown): boolean {
  const message = (error instanceof Error ? error.message : String(error)).toLowerCase();
  return /http (409|429|502|503|504)\b/.test(message)
    || message.includes("no_capacity")
    || message.includes("no model route")
    || message.includes("route warming")
    || message.includes("worker shutting down")
    || message.includes("worker disconnected");
}

function isFiniteNumber(value: number | undefined): value is number {
  return value !== undefined && Number.isFinite(value) && value >= 0;
}

function finitePositive(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value) && value > 0
    ? value
    : fallback;
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
