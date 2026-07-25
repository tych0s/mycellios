import type { NetworkExecutionTrace } from "../contracts/types.js";

export const LEGACY_BENCHMARK_RUN_SCHEMA = "mycellios-benchmark-run/1" as const;
export const PREVIOUS_BENCHMARK_RUN_SCHEMA = "mycellios-benchmark-run/2" as const;
export const BENCHMARK_RUN_SCHEMA = "mycellios-benchmark-run/3" as const;

export type BenchmarkEvidence = "physical" | "loopback";
export type BenchmarkStatus =
  | "baseline"
  | "passed"
  | "regression"
  | "inconclusive"
  | "failed";

export interface BenchmarkDeviceProfile {
  label: string;
  kind: "gpu" | "cpu";
  count: number;
  memoryGb: number | null;
  nodeId?: string;
  backend?: string | null;
  precision?: string | null;
  physicalMemoryGb?: number | null;
  offeredMemoryGb?: number | null;
  observedPowerWatts?: number | null;
  powerLimitWatts?: number | null;
  utilizationPct?: number | null;
  temperatureC?: number | null;
}

export interface BenchmarkInventory {
  totalDevices: number;
  connectedDevices: number;
  selectedDevices: number;
  profiles: BenchmarkDeviceProfile[];
  physicalMemoryGb?: number | null;
  offeredMemoryGb?: number | null;
  observedPowerWatts?: number | null;
  powerLimitWatts?: number | null;
}

export interface BenchmarkModel {
  id: string;
  label: string;
  revision: string | null;
  digest: string | null;
  precision: string;
}

export interface BenchmarkTopology {
  digest: string | null;
  stageCount: number | null;
  boundaries: number[];
  nodeIds: string[];
  routeClasses: string[];
}

export interface BenchmarkMetrics {
  tokensPerSecond: number | null;
  tokensPerSecondP5?: number | null;
  tokensPerSecondP50?: number | null;
  tokensPerSecondP95: number | null;
  aggregateTokensPerSecond: number | null;
  ttftMsP50: number | null;
  ttftMsP95: number | null;
  tpotMsP50: number | null;
  tpotMsP95: number | null;
  latencyMsP50?: number | null;
  latencyMsP95?: number | null;
  coefficientOfVariationPct?: number | null;
  confidenceHalfWidthPct?: number | null;
  requestSuccessRate?: number | null;
  exactnessRate?: number | null;
  deterministicConsistencyRate?: number | null;
  speculativeAcceptanceRate?: number | null;
  powerWattsP50?: number | null;
  powerWattsP95?: number | null;
  powerWattsPeak?: number | null;
  energyWh?: number | null;
  energyCoveragePct?: number | null;
  utilizationPctP50?: number | null;
  utilizationPctP95?: number | null;
  temperatureCPeak?: number | null;
  usedMemoryGbPeak?: number | null;
  /** Legacy alias retained while v1 history is migrated. */
  acceptanceRate: number | null;
  energyWhPerToken: number | null;
}

export interface BenchmarkComparison {
  baselineRunId: string | null;
  tokensPerSecondPct: number | null;
  ttftP95Pct: number | null;
  acceptancePoints: number | null;
  reasons: string[];
}

export interface BenchmarkMeasurement {
  id: string;
  scenarioFingerprint: string;
  title: string;
  description: string;
  evidence: BenchmarkEvidence;
  environment: "local-loopback" | "lan" | "wan" | "unknown";
  model: BenchmarkModel;
  inventory: BenchmarkInventory;
  topology: BenchmarkTopology;
  workload: {
    promptTokens: number;
    outputTokens: number;
    concurrentSequences: number;
    promptDigest?: string | null;
    requests?: number;
    successfulRequests?: number;
    warmupRequests?: number;
    recoveredFailures?: number;
    observedOutputTokens?: number;
    statisticallyStable?: boolean;
    campaignStopReason?:
      | "confidence_reached"
      | "maximum_samples"
      | "insufficient_samples";
    durationMs?: number;
    routeClasses?: string[];
  };
  metrics: BenchmarkMetrics;
  /** Per-request physical route evidence emitted by the coordinator. */
  networkTraces?: NetworkExecutionTrace[];
  status: BenchmarkStatus;
  comparison: BenchmarkComparison;
  notes: string[];
}

export interface BenchmarkRun {
  schema: typeof BENCHMARK_RUN_SCHEMA;
  runId: string;
  version: string;
  label: string;
  gitCommit: string | null;
  gitBranch: string | null;
  gitDirty: boolean | null;
  build: {
    release: string;
    releaseSource: "override" | "environment" | "package" | "unknown";
    revision: string | null;
    revisionSource: "environment" | "revision-file" | "git" | "unknown";
    sourceId: `sha256:${string}` | null;
    sourceIdSource: "provenance-file" | "runtime-local" | "unknown";
    /**
     * Build IDs declared by participating remote runtimes. They identify a
     * cohort for drift detection but are not hardware or binary attestation.
     */
    participantSourceIds: Array<`sha256:${string}`>;
  };
  startedAt: string;
  finishedAt: string;
  suite: "real-runtime" | "physical-import";
  trigger?: "automatic-model-start" | "manual" | "physical-import";
  triggerModelId?: string | null;
  status: BenchmarkStatus;
  measurements: BenchmarkMeasurement[];
}

export interface BenchmarkThresholds {
  tokensPerSecondRegressionPct: number;
  ttftP95RegressionPct: number;
  acceptanceRegressionPoints: number;
}

export const DEFAULT_BENCHMARK_THRESHOLDS: BenchmarkThresholds = Object.freeze({
  tokensPerSecondRegressionPct: 5,
  ttftP95RegressionPct: 10,
  acceptanceRegressionPoints: 5,
});

export function emptyComparison(): BenchmarkComparison {
  return {
    baselineRunId: null,
    tokensPerSecondPct: null,
    ttftP95Pct: null,
    acceptancePoints: null,
    reasons: [],
  };
}
