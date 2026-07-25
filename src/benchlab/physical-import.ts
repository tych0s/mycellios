import type {
  PhysicalGateRequestSampleV1,
  PhysicalTwoHostGpuGateReportV1,
} from "../distribution/physical-gpu-gate-report.js";
import {
  requirePassingPhysicalTwoHostGpuGate,
} from "../distribution/physical-gpu-gate-report.js";
import {
  BENCHMARK_RUN_SCHEMA,
  emptyComparison,
  type BenchmarkDeviceProfile,
  type BenchmarkMeasurement,
  type BenchmarkRun,
} from "./types.js";
import type { RunIdentity } from "./history.js";
import { sealBenchmarkScenario } from "./scenario.js";

/**
 * Imports only the sealed two-host physical gate. Raw campaign observations
 * and configuration claims are deliberately insufficient: all metrics,
 * topology, hardware and declared build-cohort fields consumed below are
 * covered by the gate's canonical SHA-256 and recomputed pass decision.
 */
export function importPhysicalCampaign(
  identity: RunIdentity,
  reportValue: unknown,
): BenchmarkRun {
  const report = requirePassingPhysicalTwoHostGpuGate(reportValue);
  const participantSourceIds = physicalParticipantSourceIds(report);
  const profiles = physicalProfiles(report);
  const measured = report.samples.filter((sample) => sample.phase === "measure");
  const concurrencies = [
    ...new Set(measured.map((sample) => sample.concurrency)),
  ].sort((left, right) => left - right);
  if (concurrencies.length === 0) {
    throw new Error("La puerta física sellada no contiene muestras medidas.");
  }
  const measurements = concurrencies.map((concurrency) =>
    physicalMeasurement(
      report,
      concurrency,
      measured.filter((sample) => sample.concurrency === concurrency),
      profiles,
    )
  );
  return {
    schema: BENCHMARK_RUN_SCHEMA,
    ...identity,
    build: {
      ...identity.build,
      participantSourceIds,
    },
    startedAt: report.capturedAt,
    finishedAt: new Date().toISOString(),
    suite: "physical-import",
    trigger: "physical-import",
    triggerModelId: report.reference.modelId,
    status: "baseline",
    measurements,
  };
}

function physicalMeasurement(
  report: PhysicalTwoHostGpuGateReportV1,
  concurrency: number,
  samples: PhysicalGateRequestSampleV1[],
  profiles: BenchmarkDeviceProfile[],
): BenchmarkMeasurement {
  const launch = report.launch.description;
  const runtimeModel = launch.runtimeModel;
  const perRequestRates = samples.map(
    (sample) => sample.completionTokens / (sample.responseMs / 1_000),
  );
  const promptTokens = samples.map((sample) => sample.promptTokens);
  const completionTokens = samples.map((sample) => sample.completionTokens);
  const ttft = samples.map((sample) => sample.ttftMs);
  const tpot = samples.map((sample) => sample.tpotMs);
  const latency = samples.map((sample) => sample.responseMs);
  const nodeIds = report.hosts.map((host) => host.rankNodeId).sort();
  return sealBenchmarkScenario({
    id: `physical-${report.provenance.networkScope}-${report.reference.modelId}-c${concurrency}`,
    title: `${report.lifecycle.health.model} físico · concurrencia ${concurrency}`,
    description:
      `Puerta física sellada sobre ${report.provenance.networkScope.toUpperCase()} con paridad, red y limpieza verificadas.`,
    evidence: "physical",
    environment: report.provenance.networkScope === "loopback"
      ? "local-loopback"
      : report.provenance.networkScope,
    model: {
      id: runtimeModel.canonicalSource ?? report.reference.modelId,
      label: report.lifecycle.health.model,
      revision:
        runtimeModel.canonicalRevision ?? report.reference.modelRevision ?? null,
      digest: runtimeModel.artifactIdentity ?? null,
      precision: report.lifecycle.health.codec,
    },
    inventory: {
      totalDevices: profiles.reduce((total, profile) => total + profile.count, 0),
      connectedDevices: profiles.reduce(
        (total, profile) => total + profile.count,
        0,
      ),
      selectedDevices: report.hosts.length,
      profiles,
      physicalMemoryGb: sumNullable(
        profiles.map((profile) => profile.physicalMemoryGb ?? null),
      ),
      offeredMemoryGb: sumNullable(
        profiles.map((profile) => profile.offeredMemoryGb ?? null),
      ),
    },
    topology: {
      digest: runtimeModel.snapshotIdentity ?? report.launch.canonicalSha256,
      stageCount: report.lifecycle.health.stages,
      boundaries: [...report.lifecycle.health.boundaries],
      nodeIds,
      routeClasses: ["pipeline"],
    },
    workload: {
      promptTokens: integerMedian(promptTokens),
      outputTokens: integerMedian(completionTokens),
      concurrentSequences: concurrency,
      promptDigest: report.reference.promptTokenIdsSha256,
      requests: samples.length,
      successfulRequests: samples.length,
      warmupRequests: report.samples.filter(
        (sample) =>
          sample.phase === "warmup" && sample.concurrency === concurrency,
      ).length,
      recoveredFailures: 0,
      observedOutputTokens: completionTokens.reduce(
        (total, value) => total + value,
        0,
      ),
      statisticallyStable: samples.length >= report.policy.minimumMeasuredSamples,
      campaignStopReason: samples.length >= report.policy.minimumMeasuredSamples
        ? "confidence_reached"
        : "insufficient_samples",
      routeClasses: ["pipeline"],
    },
    metrics: {
      tokensPerSecond: percentile(perRequestRates, 0.5),
      tokensPerSecondP5: percentile(perRequestRates, 0.05),
      tokensPerSecondP50: percentile(perRequestRates, 0.5),
      tokensPerSecondP95: percentile(perRequestRates, 0.95),
      // The gate seals individual request wall times but not a batch wall
      // clock, so aggregate throughput is intentionally left unknown.
      aggregateTokensPerSecond: concurrency === 1
        ? percentile(perRequestRates, 0.5)
        : null,
      ttftMsP50: percentile(ttft, 0.5),
      ttftMsP95: percentile(ttft, 0.95),
      tpotMsP50: percentile(tpot, 0.5),
      tpotMsP95: percentile(tpot, 0.95),
      latencyMsP50: percentile(latency, 0.5),
      latencyMsP95: percentile(latency, 0.95),
      requestSuccessRate: 1,
      exactnessRate: 1,
      deterministicConsistencyRate: 1,
      acceptanceRate: 1,
      energyWhPerToken: null,
    },
    status: "baseline",
    comparison: emptyComparison(),
    notes: [
      "MEDICIÓN FÍSICA: importada exclusivamente desde la puerta GPU sellada y revalidada.",
      `Sello ${report.seal.digest}; ${report.lifecycle.health.stages} etapas; codec ${report.lifecycle.health.codec}.`,
      "La identidad de build de los agentes es una declaración remota sellada en el informe, no attestation de hardware.",
    ],
  });
}

function physicalProfiles(
  report: PhysicalTwoHostGpuGateReportV1,
): BenchmarkDeviceProfile[] {
  return report.hosts.map((host) => ({
    nodeId: host.rankNodeId,
    label: host.gpu.model,
    kind: "gpu",
    count: 1,
    memoryGb: roundGb(host.gpu.physicalVramBytes),
    backend: host.gpu.computeApi,
    precision: report.lifecycle.health.codec,
    physicalMemoryGb: roundGb(host.gpu.physicalVramBytes),
    offeredMemoryGb: roundGb(host.gpu.offeredVramBytes),
  }));
}

function physicalParticipantSourceIds(
  report: PhysicalTwoHostGpuGateReportV1,
): Array<`sha256:${string}`> {
  const identities = report.hosts.map((host) => host.buildIdentity);
  const cohorts = new Set(
    identities.map((identity) =>
      `${identity.schema}\0${identity.version}\0${identity.sourceId}`
    ),
  );
  if (identities.length !== 2 || cohorts.size !== 1) {
    throw new Error(
      "La puerta física no demuestra una cohorte única de builds nativos declarados.",
    );
  }
  return [identities[0]!.sourceId];
}

function percentile(values: number[], quantile: number): number | null {
  if (values.length === 0) return null;
  const sorted = values.slice().sort((left, right) => left - right);
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil(quantile * sorted.length) - 1),
  );
  return round(sorted[index]!, 4);
}

function integerMedian(values: number[]): number {
  return Math.round(percentile(values, 0.5) ?? 0);
}

function roundGb(bytes: number): number {
  return round(bytes / 1024 ** 3, 3);
}

function sumNullable(values: Array<number | null>): number | null {
  return values.every((value) => value === null)
    ? null
    : round(
        values.reduce<number>(
          (total, value) => total + (value ?? 0),
          0,
        ),
        3,
      );
}

function round(value: number, decimals: number): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}
