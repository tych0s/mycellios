export const BENCHMARK_RUN_SCHEMA = "mycellios-benchmark-run/1" as const;

export type BenchmarkEvidence = "physical" | "loopback";
export type BenchmarkStatus = "baseline" | "passed" | "regression" | "failed";

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
  precision: string;
}

export interface BenchmarkMetrics {
  tokensPerSecond: number | null;
  tokensPerSecondP95: number | null;
  aggregateTokensPerSecond: number | null;
  ttftMsP50: number | null;
  ttftMsP95: number | null;
  tpotMsP50: number | null;
  tpotMsP95: number | null;
  latencyMsP50?: number | null;
  latencyMsP95?: number | null;
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
  title: string;
  description: string;
  evidence: BenchmarkEvidence;
  environment: "local-loopback" | "lan" | "wan";
  model: BenchmarkModel;
  inventory: BenchmarkInventory;
  workload: {
    promptTokens: number;
    outputTokens: number;
    concurrentSequences: number;
    requests?: number;
    successfulRequests?: number;
    durationMs?: number;
    routeClasses?: string[];
  };
  metrics: BenchmarkMetrics;
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
  gitDirty: boolean;
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
