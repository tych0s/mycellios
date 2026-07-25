import type {
  PhysicalGpuCampaignCliConfig,
  PhysicalGpuCampaignCliObservationV1,
} from "../distribution/physical-gpu-campaign-cli.js";
import { sha256CanonicalEvidence } from "../core/json.js";
import {
  BENCHMARK_RUN_SCHEMA,
  emptyComparison,
  type BenchmarkDeviceProfile,
  type BenchmarkMeasurement,
  type BenchmarkRun,
} from "./types.js";
import type { RunIdentity } from "./history.js";
import { sealBenchmarkScenario } from "./scenario.js";

const OBSERVATION_SCHEMA = "gdlp-physical-gpu-campaign-cli-observation/1";
const CONFIG_SCHEMA = "gdlp-physical-gpu-campaign-config/1";

export function importPhysicalCampaign(
  identity: RunIdentity,
  observationValue: unknown,
  configValue: unknown,
): BenchmarkRun {
  assertPhysicalInput(observationValue, configValue);
  const observation = observationValue;
  const config = configValue as PhysicalGpuCampaignCliConfig;
  const campaign = observation.campaign;
  if (!observation.passed || campaign === null || !campaign.passed || campaign.apiHealth === null) {
    throw new Error("La campaña física no terminó correctamente y no se importará como resultado válido.");
  }
  const health = campaign.apiHealth;
  const profiles = physicalProfiles(observation, health.codec);
  const connectedDevices = profiles.reduce((total, profile) => total + profile.count, 0);
  const measurements = campaign.summary.byConcurrency.map((summary) => {
    const samples = campaign.samples.filter(
      (sample) => sample.phase === "measure" && sample.concurrency === summary.concurrency,
    );
    const passedSamples = samples.filter((sample) => sample.passed).length;
    const acceptanceRate = samples.length === 0 ? null : passedSamples / samples.length;
    return physicalMeasurement(
      observation,
      config,
      summary.concurrency,
      summary.aggregateOutputTokensPerSecondIncludingTtft,
      acceptanceRate,
      profiles,
      connectedDevices,
    );
  });
  return {
    schema: BENCHMARK_RUN_SCHEMA,
    ...identity,
    startedAt: observation.capturedAt,
    finishedAt: new Date().toISOString(),
    suite: "physical-import",
    status: "baseline",
    measurements,
  };
}

function physicalMeasurement(
  observation: PhysicalGpuCampaignCliObservationV1,
  config: PhysicalGpuCampaignCliConfig,
  concurrency: number,
  tokensPerSecond: number | null,
  acceptanceRate: number | null,
  profiles: BenchmarkDeviceProfile[],
  connectedDevices: number,
): BenchmarkMeasurement {
  const campaign = observation.campaign!;
  const health = campaign.apiHealth!;
  return sealBenchmarkScenario({
    id: `physical-${config.networkScope}-${health.model}-c${concurrency}`,
    title: `${health.model} físico · concurrencia ${concurrency}`,
    description: `Campaña certificada sobre ${config.networkScope.toUpperCase()} con canarios antes y después.`,
    evidence: "physical",
    environment: config.networkScope,
    model: {
      id: health.canonicalModelSource,
      label: health.model,
      revision: health.canonicalModelRevision,
      digest: health.artifactIdentity,
      precision: health.codec,
    },
    inventory: {
      totalDevices: connectedDevices,
      connectedDevices,
      selectedDevices: config.hosts.length,
      profiles,
    },
    topology: {
      digest: health.pipelineSnapshotIdentity || observation.pipelineId,
      stageCount: health.stages,
      boundaries: health.boundaries?.slice() ?? [],
      nodeIds: config.hosts.map((host) => host.rankNodeId),
      routeClasses: ["pipeline"],
    },
    workload: {
      promptTokens: median(
        campaign.samples
          .filter((sample) => sample.phase === "measure" && sample.concurrency === concurrency)
          .map((sample) => sample.promptTokens)
          .filter((value): value is number => value !== null),
      ),
      outputTokens: median(
        campaign.samples
          .filter((sample) => sample.phase === "measure" && sample.concurrency === concurrency)
          .map((sample) => sample.completionTokens)
          .filter((value): value is number => value !== null),
      ),
      concurrentSequences: concurrency,
      promptDigest: Array.isArray(config.canaries) && config.canaries.length > 0
        ? sha256CanonicalEvidence(config.canaries)
        : null,
      requests: campaign.samples.filter(
        (sample) => sample.phase === "measure" && sample.concurrency === concurrency,
      ).length,
      routeClasses: ["pipeline"],
    },
    metrics: {
      tokensPerSecond,
      tokensPerSecondP95: null,
      aggregateTokensPerSecond: tokensPerSecond,
      ttftMsP50: campaign.summary.serverTtftMs.p50,
      ttftMsP95: campaign.summary.serverTtftMs.p95,
      tpotMsP50: campaign.summary.serverTpotMs.p50,
      tpotMsP95: campaign.summary.serverTpotMs.p95,
      acceptanceRate,
      energyWhPerToken: null,
    },
    status: "baseline",
    comparison: emptyComparison(),
    notes: [
      "MEDICIÓN FÍSICA: importada de la campaña GPU con paridad y limpieza verificadas.",
      `Pipeline ${observation.pipelineId ?? "desconocido"}; ${health.stages} etapas; codec ${health.codec}.`,
    ],
  });
}

function physicalProfiles(
  observation: PhysicalGpuCampaignCliObservationV1,
  precision: string,
): BenchmarkDeviceProfile[] {
  return observation.probes.flatMap((probe) => {
    const backend = probe.probe.runtime?.cudaApiAvailable
      ? probe.probe.runtime.rocmVersion ? "rocm" : "cuda"
      : null;
    return probe.probe.devices.map((device) => ({
      nodeId: probe.rankNodeId,
      label: device.name,
      kind: "gpu" as const,
      count: 1,
      memoryGb: Math.round((device.totalMemoryBytes / 1024 ** 3) * 10) / 10,
      backend,
      precision,
      physicalMemoryGb: Math.round((device.totalMemoryBytes / 1024 ** 3) * 10) / 10,
    }));
  });
}

function assertPhysicalInput(
  observation: unknown,
  config: unknown,
): asserts observation is PhysicalGpuCampaignCliObservationV1 {
  if (!isRecord(observation) || observation.schema !== OBSERVATION_SCHEMA) {
    throw new Error("El archivo no es una observación de campaña física compatible.");
  }
  if (!isRecord(config) || config.schema !== CONFIG_SCHEMA) {
    throw new Error("El archivo de configuración física no es compatible.");
  }
  if (!isRecord(observation.campaign) || !Array.isArray(observation.probes)) {
    throw new Error("La observación física está incompleta.");
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = values.slice().sort((left, right) => left - right);
  return Math.round(sorted[Math.floor((sorted.length - 1) / 2)]!);
}
