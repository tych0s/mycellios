export interface BenchmarkNodeTelemetry {
  nodeId: string;
  powerWatts: number | null;
  utilizationPct: number | null;
  temperatureC: number | null;
  offeredMemoryGb: number | null;
  freeOfferedMemoryGb: number | null;
}

export interface BenchmarkTelemetrySnapshot {
  atMs: number;
  nodes: BenchmarkNodeTelemetry[];
}

export interface BenchmarkPhysicalTelemetrySummary {
  sampleCount: number;
  durationMs: number;
  energyCoveragePct: number;
  powerWattsP50: number | null;
  powerWattsP95: number | null;
  powerWattsPeak: number | null;
  energyWh: number | null;
  energyWhPerToken: number | null;
  utilizationPctP50: number | null;
  utilizationPctP95: number | null;
  temperatureCPeak: number | null;
  usedMemoryGbPeak: number | null;
}

export interface PhysicalTelemetryWindowOptions {
  maximumSamples?: number;
  maximumIntegrationGapMs?: number;
}

/**
 * Bounded physical telemetry window for one benchmark campaign.
 *
 * Energy is integrated only across adjacent measured power samples. Missing
 * heartbeats remain missing and reduce coverage; they are never interpolated
 * from a power limit or a GPU catalogue value.
 */
export class PhysicalTelemetryWindow {
  private readonly snapshots: BenchmarkTelemetrySnapshot[] = [];
  private readonly maximumSamples: number;
  private readonly maximumIntegrationGapMs: number;

  constructor(options: PhysicalTelemetryWindowOptions = {}) {
    this.maximumSamples = boundedInteger(options.maximumSamples, 3_600, 2, 100_000);
    this.maximumIntegrationGapMs = boundedInteger(
      options.maximumIntegrationGapMs,
      15_000,
      1,
      5 * 60_000,
    );
  }

  observe(snapshot: BenchmarkTelemetrySnapshot): void {
    validateSnapshot(snapshot);
    const previous = this.snapshots.at(-1);
    if (previous && snapshot.atMs <= previous.atMs) {
      throw new Error("benchmark_telemetry_timestamp_must_increase");
    }
    this.snapshots.push(structuredClone(snapshot));
    if (this.snapshots.length > this.maximumSamples) this.snapshots.shift();
  }

  summarize(
    selectedNodeIds: ReadonlySet<string>,
    outputTokens: number,
  ): BenchmarkPhysicalTelemetrySummary {
    const selected = this.snapshots.map((snapshot) => ({
      atMs: snapshot.atMs,
      nodes: snapshot.nodes.filter((node) => selectedNodeIds.has(node.nodeId)),
    }));
    const usable = selected.filter((snapshot) => snapshot.nodes.length > 0);
    const startedAt = usable[0]?.atMs ?? null;
    const finishedAt = usable.at(-1)?.atMs ?? null;
    const durationMs = startedAt !== null && finishedAt !== null
      ? Math.max(0, finishedAt - startedAt)
      : 0;
    const totalPower = usable.flatMap((snapshot) => {
      const powers = snapshot.nodes.map((node) => node.powerWatts);
      return powers.every(isFiniteNonNegative)
        ? [powers.reduce((sum, value) => sum + value, 0)]
        : [];
    });
    let integratedWattMilliseconds = 0;
    let integratedDurationMs = 0;
    for (let index = 1; index < usable.length; index += 1) {
      const previous = usable[index - 1]!;
      const current = usable[index]!;
      const gap = current.atMs - previous.atMs;
      if (gap <= 0 || gap > this.maximumIntegrationGapMs) continue;
      const previousPower = totalNodePower(previous.nodes);
      const currentPower = totalNodePower(current.nodes);
      if (previousPower === null || currentPower === null) continue;
      integratedWattMilliseconds += ((previousPower + currentPower) / 2) * gap;
      integratedDurationMs += gap;
    }
    const energyWh = integratedDurationMs > 0
      ? integratedWattMilliseconds / 3_600_000
      : null;
    const utilization = usable.flatMap((snapshot) =>
      snapshot.nodes.flatMap((node) =>
        node.utilizationPct === null ? [] : [node.utilizationPct]
      )
    );
    const temperatures = usable.flatMap((snapshot) =>
      snapshot.nodes.flatMap((node) =>
        node.temperatureC === null ? [] : [node.temperatureC]
      )
    );
    const usedMemory = usable.flatMap((snapshot) => {
      const perNode = snapshot.nodes.map((node) =>
        node.offeredMemoryGb !== null && node.freeOfferedMemoryGb !== null
          ? Math.max(0, node.offeredMemoryGb - node.freeOfferedMemoryGb)
          : null
      );
      return perNode.every((value): value is number => value !== null)
        ? [perNode.reduce((sum, value) => sum + value, 0)]
        : [];
    });
    return {
      sampleCount: usable.length,
      durationMs,
      energyCoveragePct: durationMs > 0
        ? round((integratedDurationMs / durationMs) * 100, 2)
        : 0,
      powerWattsP50: nullablePercentile(totalPower, 0.5),
      powerWattsP95: nullablePercentile(totalPower, 0.95),
      powerWattsPeak: totalPower.length > 0 ? Math.max(...totalPower) : null,
      energyWh: energyWh === null ? null : round(energyWh, 8),
      energyWhPerToken:
        energyWh !== null && Number.isFinite(outputTokens) && outputTokens > 0
          ? round(energyWh / outputTokens, 10)
          : null,
      utilizationPctP50: nullablePercentile(utilization, 0.5),
      utilizationPctP95: nullablePercentile(utilization, 0.95),
      temperatureCPeak: temperatures.length > 0 ? Math.max(...temperatures) : null,
      usedMemoryGbPeak: usedMemory.length > 0 ? Math.max(...usedMemory) : null,
    };
  }
}

function validateSnapshot(snapshot: BenchmarkTelemetrySnapshot): void {
  if (!Number.isFinite(snapshot.atMs) || snapshot.atMs < 0) {
    throw new Error("benchmark_telemetry_timestamp_is_invalid");
  }
  const nodeIds = new Set<string>();
  for (const node of snapshot.nodes) {
    const nodeId = node.nodeId.trim();
    if (!nodeId || nodeIds.has(nodeId)) {
      throw new Error("benchmark_telemetry_node_id_is_invalid_or_duplicate");
    }
    nodeIds.add(nodeId);
    for (const [key, value] of Object.entries(node)) {
      if (key === "nodeId" || value === null) continue;
      if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
        throw new Error(`benchmark_telemetry_${key}_is_invalid`);
      }
    }
    if (
      node.offeredMemoryGb !== null
      && node.freeOfferedMemoryGb !== null
      && node.freeOfferedMemoryGb > node.offeredMemoryGb
    ) {
      throw new Error("benchmark_telemetry_free_memory_exceeds_offer");
    }
  }
}

function totalNodePower(nodes: readonly BenchmarkNodeTelemetry[]): number | null {
  const powers = nodes.map((node) => node.powerWatts);
  return powers.length > 0 && powers.every(isFiniteNonNegative)
    ? powers.reduce((sum, value) => sum + value, 0)
    : null;
}

function isFiniteNonNegative(value: number | null): value is number {
  return value !== null && Number.isFinite(value) && value >= 0;
}

function nullablePercentile(values: readonly number[], fraction: number): number | null {
  if (values.length === 0) return null;
  const ordered = values.slice().sort((left, right) => left - right);
  const position = (ordered.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const value = lower === upper
    ? ordered[lower]!
    : ordered[lower]! * (1 - (position - lower))
      + ordered[upper]! * (position - lower);
  return round(value, 4);
}

function boundedInteger(
  value: number | undefined,
  fallback: number,
  minimum: number,
  maximum: number,
): number {
  const candidate = value === undefined ? fallback : Math.round(value);
  return Number.isFinite(candidate)
    ? Math.min(maximum, Math.max(minimum, candidate))
    : fallback;
}

function round(value: number, decimals: number): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}
