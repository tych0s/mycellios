import type {
  BenchmarkEvidence,
  BenchmarkMeasurement,
  BenchmarkRun,
  BenchmarkStatus,
} from "../../src/benchlab/types.js";

export type BenchmarkComparisonMode =
  | "same-scenario"
  | "capacity-change"
  | "configuration-change"
  | "different-evidence"
  | "different-model"
  | "unavailable";

export type BenchmarkPerformanceVerdict =
  | "faster-code-signal"
  | "slower-code-signal"
  | "within-variation"
  | "insufficient-confidence"
  | "capacity-change"
  | "configuration-change"
  | "not-comparable";

export interface BenchmarkFilters {
  model: string;
  version: string;
  status: "" | BenchmarkStatus;
  backend: string;
  evidence: "" | BenchmarkEvidence;
}

export const EMPTY_BENCHMARK_FILTERS: BenchmarkFilters = Object.freeze({
  model: "",
  version: "",
  status: "",
  backend: "",
  evidence: "",
});

export interface BenchmarkRunSnapshot {
  run: BenchmarkRun;
  measurement: BenchmarkMeasurement;
  scenarioFingerprint: string;
  scenarioComparable: boolean;
  modelKey: string;
  nodes: number;
  offeredMemoryGb: number | null;
  observedPowerWatts: number | null;
  tokensPerSecond: number | null;
  tokensPerSecondP5: number | null;
  tokensPerSecondP50: number | null;
  tokensPerSecondP95: number | null;
  ttftMsP95: number | null;
  tokensPerSecondPerNode: number | null;
  tokensPerSecondPerGb: number | null;
  tokensPerSecondPerWatt: number | null;
  statisticallyStable: boolean | null;
  confidenceHalfWidthPct: number | null;
  coefficientOfVariationPct: number | null;
}

export interface BenchmarkRunComparison {
  mode: BenchmarkComparisonMode;
  verdict: BenchmarkPerformanceVerdict;
  current: BenchmarkRunSnapshot | null;
  baseline: BenchmarkRunSnapshot | null;
  speedChangePct: number | null;
  latencyImprovementPct: number | null;
  efficiencyPerNodeChangePct: number | null;
  efficiencyPerGbChangePct: number | null;
  efficiencyPerWattChangePct: number | null;
  nodesDelta: number | null;
  offeredMemoryDeltaGb: number | null;
}

export interface BenchmarkComparisonNarrative {
  eyebrow: string;
  title: string;
  description: string;
}

export function benchmarkRunSnapshot(run: BenchmarkRun | null | undefined): BenchmarkRunSnapshot | null {
  if (!run) return null;
  const measurement = run.measurements.find((item) => item.metrics.tokensPerSecond !== null)
    ?? run.measurements[0];
  if (!measurement) return null;

  const nodes = measurement.inventory.selectedDevices;
  const offeredMemoryGb = measurement.inventory.offeredMemoryGb
    ?? nullableSum(measurement.inventory.profiles
      .filter((profile) => profile.kind === "gpu")
      .map((profile) => profile.offeredMemoryGb ?? profile.memoryGb));
  const observedPowerWatts = measurement.inventory.observedPowerWatts
    ?? nullableSum(measurement.inventory.profiles.map((profile) => profile.observedPowerWatts));
  const tokensPerSecond = measurement.metrics.tokensPerSecondP50
    ?? measurement.metrics.tokensPerSecond;

  return {
    run,
    measurement,
    scenarioFingerprint: measurement.scenarioFingerprint,
    scenarioComparable: Boolean(
      measurement.model.digest?.trim()
      && /^sha256:[0-9a-f]{64}$/i.test(measurement.scenarioFingerprint),
    ),
    modelKey: modelKey(measurement),
    nodes,
    offeredMemoryGb,
    observedPowerWatts: measurement.metrics.powerWattsP50 ?? observedPowerWatts,
    tokensPerSecond,
    tokensPerSecondP5: measurement.metrics.tokensPerSecondP5 ?? null,
    tokensPerSecondP50: measurement.metrics.tokensPerSecondP50 ?? measurement.metrics.tokensPerSecond,
    tokensPerSecondP95: measurement.metrics.tokensPerSecondP95,
    ttftMsP95: measurement.metrics.ttftMsP95,
    tokensPerSecondPerNode: safeDivide(tokensPerSecond, nodes),
    tokensPerSecondPerGb: safeDivide(tokensPerSecond, offeredMemoryGb),
    tokensPerSecondPerWatt: safeDivide(
      tokensPerSecond,
      measurement.metrics.powerWattsP50 ?? observedPowerWatts,
    ),
    statisticallyStable: measurement.workload.statisticallyStable ?? null,
    confidenceHalfWidthPct: measurement.metrics.confidenceHalfWidthPct ?? null,
    coefficientOfVariationPct: measurement.metrics.coefficientOfVariationPct ?? null,
  };
}

export function compareBenchmarkRuns(
  currentRun: BenchmarkRun | null | undefined,
  baselineRun: BenchmarkRun | null | undefined,
): BenchmarkRunComparison {
  const current = benchmarkRunSnapshot(currentRun);
  const baseline = benchmarkRunSnapshot(baselineRun);
  const mode = comparisonMode(current, baseline);
  const canComparePerformance = mode === "same-scenario";
  const verdict = comparisonVerdict(mode, current, baseline);

  return {
    mode,
    verdict,
    current,
    baseline,
    speedChangePct: canComparePerformance
      ? percentChange(current?.tokensPerSecond, baseline?.tokensPerSecond)
      : null,
    latencyImprovementPct: canComparePerformance
      ? inversePercentChange(current?.ttftMsP95, baseline?.ttftMsP95)
      : null,
    efficiencyPerNodeChangePct: canComparePerformance
      ? percentChange(current?.tokensPerSecondPerNode, baseline?.tokensPerSecondPerNode)
      : null,
    efficiencyPerGbChangePct: canComparePerformance
      ? percentChange(current?.tokensPerSecondPerGb, baseline?.tokensPerSecondPerGb)
      : null,
    efficiencyPerWattChangePct: canComparePerformance
      ? percentChange(current?.tokensPerSecondPerWatt, baseline?.tokensPerSecondPerWatt)
      : null,
    nodesDelta: current && baseline ? current.nodes - baseline.nodes : null,
    offeredMemoryDeltaGb: current?.offeredMemoryGb !== null
      && current?.offeredMemoryGb !== undefined
      && baseline?.offeredMemoryGb !== null
      && baseline?.offeredMemoryGb !== undefined
      ? current.offeredMemoryGb - baseline.offeredMemoryGb
      : null,
  };
}

export function defaultBenchmarkBaseline(current: BenchmarkRun | null, runs: BenchmarkRun[]): BenchmarkRun | null {
  if (!current) return null;
  const currentSnapshot = benchmarkRunSnapshot(current);
  if (!currentSnapshot || !currentSnapshot.scenarioComparable) return null;

  const currentFinishedAt = Date.parse(current.finishedAt);
  const candidates = runs
    .filter((run) => run.runId !== current.runId)
    .sort((left, right) => Date.parse(right.finishedAt) - Date.parse(left.finishedAt));
  const earlierCandidates = candidates.filter((run) => {
    const finishedAt = Date.parse(run.finishedAt);
    return !Number.isFinite(currentFinishedAt) || !Number.isFinite(finishedAt) || finishedAt < currentFinishedAt;
  });
  const orderedCandidates = earlierCandidates.length > 0 ? earlierCandidates : candidates;
  return orderedCandidates.find((run) => {
    const candidate = benchmarkRunSnapshot(run);
    return candidate
      && candidate.scenarioComparable
      && candidate.scenarioFingerprint === currentSnapshot.scenarioFingerprint;
  }) ?? null;
}

export function benchmarkRunHasModel(run: BenchmarkRun, modelKeyValue: string): boolean {
  return run.measurements.some((measurement) => modelKey(measurement) === modelKeyValue);
}

export function benchmarkMeasurementModelKey(measurement: BenchmarkMeasurement): string {
  return modelKey(measurement);
}

export function filterBenchmarkRuns(
  runs: BenchmarkRun[],
  filters: BenchmarkFilters,
): BenchmarkRun[] {
  return runs.filter((run) => {
    if (filters.version && run.version !== filters.version) return false;
    if (filters.status && run.status !== filters.status) return false;
    return run.measurements.some((measurement) => {
      if (filters.model && modelKey(measurement) !== filters.model) return false;
      if (filters.evidence && measurement.evidence !== filters.evidence) return false;
      if (filters.backend && !measurementBackends(measurement).includes(filters.backend)) {
        return false;
      }
      return true;
    });
  });
}

export function benchmarkTrendRuns(
  currentRun: BenchmarkRun | null,
  runs: BenchmarkRun[],
): BenchmarkRun[] {
  const current = benchmarkRunSnapshot(currentRun);
  if (!current?.scenarioComparable) return currentRun ? [currentRun] : [];
  return runs.filter((run) => {
    const candidate = benchmarkRunSnapshot(run);
    return candidate?.scenarioComparable === true
      && candidate.scenarioFingerprint === current.scenarioFingerprint;
  });
}

export function benchmarkFilterValues(runs: BenchmarkRun[]): {
  models: Array<{ value: string; label: string }>;
  versions: string[];
  statuses: BenchmarkStatus[];
  backends: string[];
  evidence: BenchmarkEvidence[];
} {
  const models = new Map<string, string>();
  const versions = new Set<string>();
  const statuses = new Set<BenchmarkStatus>();
  const backends = new Set<string>();
  const evidence = new Set<BenchmarkEvidence>();
  for (const run of runs) {
    versions.add(run.version);
    statuses.add(run.status);
    for (const measurement of run.measurements) {
      models.set(modelKey(measurement), measurement.model.label);
      evidence.add(measurement.evidence);
      for (const backend of measurementBackends(measurement)) backends.add(backend);
    }
  }
  return {
    models: [...models].map(([value, label]) => ({ value, label }))
      .sort((left, right) => left.label.localeCompare(right.label)),
    versions: [...versions].sort((left, right) =>
      right.localeCompare(left, undefined, { numeric: true })
    ),
    statuses: [...statuses],
    backends: [...backends].sort(),
    evidence: [...evidence],
  };
}

export function measurementBackends(measurement: BenchmarkMeasurement): string[] {
  const values = measurement.inventory.profiles
    .map((profile) => normalized(profile.backend))
    .filter((value): value is string => value !== null);
  return values.length > 0 ? [...new Set(values)] : ["__without_reading__"];
}

export function benchmarkComparisonNarrative(
  comparison: BenchmarkRunComparison,
): BenchmarkComparisonNarrative {
  if (comparison.verdict === "faster-code-signal") {
    return {
      eyebrow: "CODE IMPROVEMENT SIGNAL",
      title: "Faster with the same capacity",
      description: "Model, workload, backend, nodes and topology match; both campaigns are stable and their P5–P95 bands do not overlap.",
    };
  }
  if (comparison.verdict === "slower-code-signal") {
    return {
      eyebrow: "CODE REGRESSION SIGNAL",
      title: "Slower with the same capacity",
      description: "The scenario is identical and stable, but the entire current P5–P95 band is below the reference.",
    };
  }
  if (comparison.verdict === "within-variation") {
    return {
      eyebrow: "WITHIN VARIATION",
      title: "No demonstrated improvement",
      description: "The scenario is identical and stable, but the P5–P95 bands overlap. The difference may be normal execution noise.",
    };
  }
  if (comparison.verdict === "insufficient-confidence") {
    return {
      eyebrow: "INCONCLUSIVE COMPARISON",
      title: "Same scenario, stable samples missing",
      description: "The configuration matches, but variation or statistical coverage cannot yet attribute the change to code.",
    };
  }
  if (comparison.mode === "capacity-change") {
    return {
      eyebrow: "CAPACITY CHANGE",
      title: "Hardware or topology changed",
      description: "Total speed can still show scaling. Also inspect tok/s per node and GB; this is not a clean code improvement.",
    };
  }
  if (comparison.mode === "configuration-change") {
    return {
      eyebrow: "DIFFERENT CONFIGURATION",
      title: "Backend or workload changed",
      description: "The model and capacity may match, but the fingerprint does not. Values are shown without attributing the change to code.",
    };
  }
  if (comparison.mode === "different-evidence") {
    return {
      eyebrow: "DIFFERENT EVIDENCE",
      title: "Physical and loopback do not mix",
      description: "Real values from each run are shown, but no improvement is calculated between different environments.",
    };
  }
  if (comparison.mode === "different-model") {
    return {
      eyebrow: "DIFFERENT MODELS",
      title: "This selection is not comparable",
      description: "Select a reference from the same model to measure speed and efficiency over time.",
    };
  }
  return {
    eyebrow: "NO REFERENCE RUN",
    title: "No comparable reference yet",
    description: "Repeat the same model and workload with matching physical evidence to compare speed, latency and efficiency.",
  };
}

function comparisonMode(
  current: BenchmarkRunSnapshot | null,
  baseline: BenchmarkRunSnapshot | null,
): BenchmarkComparisonMode {
  if (!current || !baseline) return "unavailable";
  if (!current.scenarioComparable || !baseline.scenarioComparable) return "unavailable";
  if (
    normalized(current.measurement.model.id) !== normalized(baseline.measurement.model.id)
    || normalized(current.measurement.model.revision) !== normalized(baseline.measurement.model.revision)
    || normalized(current.measurement.model.digest) !== normalized(baseline.measurement.model.digest)
    || normalized(current.measurement.model.precision) !== normalized(baseline.measurement.model.precision)
  ) return "different-model";
  if (current.measurement.evidence !== baseline.measurement.evidence) return "different-evidence";
  if (current.scenarioFingerprint === baseline.scenarioFingerprint) return "same-scenario";
  if (capacitySignature(current.measurement) !== capacitySignature(baseline.measurement)) {
    return "capacity-change";
  }
  return "configuration-change";
}

function comparisonVerdict(
  mode: BenchmarkComparisonMode,
  current: BenchmarkRunSnapshot | null,
  baseline: BenchmarkRunSnapshot | null,
): BenchmarkPerformanceVerdict {
  if (mode === "capacity-change") return "capacity-change";
  if (mode === "configuration-change") return "configuration-change";
  if (mode !== "same-scenario" || !current || !baseline) return "not-comparable";
  if (current.statisticallyStable !== true || baseline.statisticallyStable !== true) {
    return "insufficient-confidence";
  }
  if (
    current.tokensPerSecondP5 === null
    || current.tokensPerSecondP95 === null
    || baseline.tokensPerSecondP5 === null
    || baseline.tokensPerSecondP95 === null
  ) {
    return "insufficient-confidence";
  }
  if (current.tokensPerSecondP5 > baseline.tokensPerSecondP95) {
    return "faster-code-signal";
  }
  if (current.tokensPerSecondP95 < baseline.tokensPerSecondP5) {
    return "slower-code-signal";
  }
  return "within-variation";
}

function capacitySignature(measurement: BenchmarkMeasurement): string {
  const profileCapacity = measurement.inventory.profiles.map((profile) => ({
    nodeId: normalized(profile.nodeId),
    kind: profile.kind,
    count: profile.count,
    memoryGb: profile.memoryGb,
    physicalMemoryGb: profile.physicalMemoryGb ?? null,
    offeredMemoryGb: profile.offeredMemoryGb ?? null,
  })).sort((left, right) => JSON.stringify(left).localeCompare(JSON.stringify(right)));
  return JSON.stringify({
    totalDevices: measurement.inventory.totalDevices,
    connectedDevices: measurement.inventory.connectedDevices,
    selectedDevices: measurement.inventory.selectedDevices,
    physicalMemoryGb: measurement.inventory.physicalMemoryGb ?? null,
    offeredMemoryGb: measurement.inventory.offeredMemoryGb ?? null,
    profileCapacity,
    topology: {
      digest: normalized(measurement.topology.digest),
      stageCount: measurement.topology.stageCount,
      boundaries: measurement.topology.boundaries,
      nodeIds: [...measurement.topology.nodeIds].map(normalized).sort(),
    },
  });
}

function modelKey(measurement: BenchmarkMeasurement): string {
  return `${measurement.model.id.trim().toLowerCase()}::${measurement.model.label.trim().toLowerCase()}`;
}

function normalized(value: string | null | undefined): string | null {
  const result = value?.trim().toLowerCase();
  return result ? result : null;
}

function nullableSum(values: Array<number | null | undefined>): number | null {
  const available = values.filter((value): value is number => value !== null && value !== undefined);
  return available.length > 0 ? available.reduce((sum, value) => sum + value, 0) : null;
}

function safeDivide(numerator: number | null | undefined, denominator: number | null | undefined): number | null {
  if (numerator === null || numerator === undefined || denominator === null || denominator === undefined || denominator <= 0) return null;
  return numerator / denominator;
}

function percentChange(current: number | null | undefined, baseline: number | null | undefined): number | null {
  if (current === null || current === undefined || baseline === null || baseline === undefined || baseline === 0) return null;
  return ((current - baseline) / Math.abs(baseline)) * 100;
}

function inversePercentChange(current: number | null | undefined, baseline: number | null | undefined): number | null {
  if (current === null || current === undefined || baseline === null || baseline === undefined || baseline === 0) return null;
  return ((baseline - current) / Math.abs(baseline)) * 100;
}
