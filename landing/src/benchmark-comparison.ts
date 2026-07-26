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
      eyebrow: "SEÑAL DE MEJORA DEL CÓDIGO",
      title: "Más rápido con la misma capacidad",
      description: "Modelo, workload, backend, nodos y topología coinciden; ambas campañas son estables y sus bandas P5–P95 no se solapan.",
    };
  }
  if (comparison.verdict === "slower-code-signal") {
    return {
      eyebrow: "SEÑAL DE REGRESIÓN DEL CÓDIGO",
      title: "Más lento con la misma capacidad",
      description: "El escenario es idéntico y estable, pero toda la banda P5–P95 actual queda por debajo de la referencia.",
    };
  }
  if (comparison.verdict === "within-variation") {
    return {
      eyebrow: "DENTRO DE LA VARIACIÓN",
      title: "No hay una mejora demostrada",
      description: "El escenario es idéntico y estable, pero las bandas P5–P95 se solapan. La diferencia puede ser ruido normal de ejecución.",
    };
  }
  if (comparison.verdict === "insufficient-confidence") {
    return {
      eyebrow: "COMPARACIÓN INCONCLUSA",
      title: "Mismo escenario, faltan muestras estables",
      description: "La configuración coincide, pero la variación o la cobertura estadística todavía no permiten atribuir el cambio al código.",
    };
  }
  if (comparison.mode === "capacity-change") {
    return {
      eyebrow: "CAMBIO DE CAPACIDAD",
      title: "Ha cambiado el hardware o la topología",
      description: "La velocidad total permite estudiar el escalado. Mira también tok/s por nodo y por GB: no se atribuye al código como mejora limpia.",
    };
  }
  if (comparison.mode === "configuration-change") {
    return {
      eyebrow: "CONFIGURACIÓN DISTINTA",
      title: "Ha cambiado el backend o la carga",
      description: "El modelo y la capacidad pueden coincidir, pero el fingerprint no. Los valores se muestran sin atribuir el cambio al código.",
    };
  }
  if (comparison.mode === "different-evidence") {
    return {
      eyebrow: "EVIDENCIA DISTINTA",
      title: "Físico y loopback no se mezclan",
      description: "Se muestran los valores reales de cada ejecución, pero no se calcula una mejora entre entornos diferentes.",
    };
  }
  if (comparison.mode === "different-model") {
    return {
      eyebrow: "MODELOS DISTINTOS",
      title: "Esta selección no es comparable",
      description: "Selecciona una referencia del mismo modelo para medir evolución de velocidad y eficiencia.",
    };
  }
  return {
    eyebrow: "SIN REFERENCIA",
    title: "Hace falta una segunda ejecución",
    description: "Cuando haya otra prueba real podrás comparar versiones, nodos, VRAM, latencia y eficiencia.",
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
