import type { BenchmarkMeasurement, BenchmarkRun } from "../../src/benchlab/types.js";

export type BenchmarkComparisonMode =
  | "same-scenario"
  | "capacity-change"
  | "different-evidence"
  | "different-model"
  | "unavailable";

export interface BenchmarkRunSnapshot {
  run: BenchmarkRun;
  measurement: BenchmarkMeasurement;
  modelKey: string;
  nodes: number;
  offeredMemoryGb: number | null;
  observedPowerWatts: number | null;
  tokensPerSecond: number | null;
  ttftMsP95: number | null;
  tokensPerSecondPerNode: number | null;
  tokensPerSecondPerGb: number | null;
  tokensPerSecondPerWatt: number | null;
}

export interface BenchmarkRunComparison {
  mode: BenchmarkComparisonMode;
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
  const tokensPerSecond = measurement.metrics.tokensPerSecond;

  return {
    run,
    measurement,
    modelKey: modelKey(measurement),
    nodes,
    offeredMemoryGb,
    observedPowerWatts,
    tokensPerSecond,
    ttftMsP95: measurement.metrics.ttftMsP95,
    tokensPerSecondPerNode: safeDivide(tokensPerSecond, nodes),
    tokensPerSecondPerGb: safeDivide(tokensPerSecond, offeredMemoryGb),
    tokensPerSecondPerWatt: safeDivide(tokensPerSecond, observedPowerWatts),
  };
}

export function compareBenchmarkRuns(
  currentRun: BenchmarkRun | null | undefined,
  baselineRun: BenchmarkRun | null | undefined,
): BenchmarkRunComparison {
  const current = benchmarkRunSnapshot(currentRun);
  const baseline = benchmarkRunSnapshot(baselineRun);
  const mode = comparisonMode(current, baseline);
  const canComparePerformance = mode === "same-scenario" || mode === "capacity-change";

  return {
    mode,
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
  if (!currentSnapshot) return runs.find((run) => run.runId !== current.runId) ?? null;

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
      && candidate.modelKey === currentSnapshot.modelKey
      && candidate.measurement.evidence === currentSnapshot.measurement.evidence
      && candidate.measurement.model.revision === currentSnapshot.measurement.model.revision
      && candidate.measurement.model.precision === currentSnapshot.measurement.model.precision
      && candidate.measurement.id === currentSnapshot.measurement.id;
  }) ?? orderedCandidates.find((run) => {
    const candidate = benchmarkRunSnapshot(run);
    return candidate
      && candidate.modelKey === currentSnapshot.modelKey
      && candidate.measurement.evidence === currentSnapshot.measurement.evidence;
  }) ?? null;
}

export function benchmarkRunHasModel(run: BenchmarkRun, modelKeyValue: string): boolean {
  return run.measurements.some((measurement) => modelKey(measurement) === modelKeyValue);
}

export function benchmarkMeasurementModelKey(measurement: BenchmarkMeasurement): string {
  return modelKey(measurement);
}

function comparisonMode(
  current: BenchmarkRunSnapshot | null,
  baseline: BenchmarkRunSnapshot | null,
): BenchmarkComparisonMode {
  if (!current || !baseline) return "unavailable";
  if (current.modelKey !== baseline.modelKey) return "different-model";
  if (
    current.measurement.model.revision !== baseline.measurement.model.revision
    || current.measurement.model.precision !== baseline.measurement.model.precision
  ) return "different-model";
  if (current.measurement.evidence !== baseline.measurement.evidence) return "different-evidence";
  if (current.measurement.id !== baseline.measurement.id) return "capacity-change";
  if (
    current.nodes !== baseline.nodes
    || current.offeredMemoryGb !== baseline.offeredMemoryGb
    || current.measurement.workload.concurrentSequences !== baseline.measurement.workload.concurrentSequences
    || current.measurement.workload.outputTokens !== baseline.measurement.workload.outputTokens
    || current.measurement.workload.promptTokens !== baseline.measurement.workload.promptTokens
  ) return "capacity-change";
  return "same-scenario";
}

function modelKey(measurement: BenchmarkMeasurement): string {
  return `${measurement.model.id.trim().toLowerCase()}::${measurement.model.label.trim().toLowerCase()}`;
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
