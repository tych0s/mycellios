import { sha256CanonicalEvidence } from "../core/json.js";
import type {
  BenchmarkDeviceProfile,
  BenchmarkMeasurement,
  BenchmarkTopology,
} from "./types.js";

export const BENCHMARK_SCENARIO_SCHEMA = "mycellios-benchmark-scenario/1" as const;

type UnsealedMeasurement = Omit<BenchmarkMeasurement, "scenarioFingerprint">;

/**
 * Seals only scenario inputs. Result metrics, timestamps and application build
 * identity are deliberately excluded so two builds can be compared under the
 * exact same physical scenario.
 */
export function sealBenchmarkScenario(measurement: UnsealedMeasurement): BenchmarkMeasurement {
  return {
    ...measurement,
    scenarioFingerprint: benchmarkScenarioFingerprint(measurement),
  };
}

export function benchmarkScenarioFingerprint(
  measurement: Pick<
    UnsealedMeasurement,
    "evidence" | "environment" | "model" | "inventory" | "topology" | "workload"
  >,
): string {
  const profiles = measurement.inventory.profiles
    .map(canonicalProfile)
    .sort((left, right) => ordinal(JSON.stringify(left), JSON.stringify(right)));
  const topology = canonicalTopology(measurement.topology);
  return sha256CanonicalEvidence({
    schema: BENCHMARK_SCENARIO_SCHEMA,
    model: {
      id: normalized(measurement.model.id),
      revision: nullableNormalized(measurement.model.revision),
      digest: nullableNormalized(measurement.model.digest),
      precision: normalized(measurement.model.precision),
    },
    workload: {
      promptTokens: measurement.workload.promptTokens,
      outputTokens: measurement.workload.outputTokens,
      concurrentSequences: measurement.workload.concurrentSequences,
      promptDigest: nullableNormalized(measurement.workload.promptDigest ?? null),
      requests: measurement.workload.requests ?? null,
      routeClasses: sortedNormalized(measurement.workload.routeClasses ?? []),
    },
    evidence: {
      class: measurement.evidence,
      environment: measurement.environment,
    },
    hardware: {
      totalDevices: measurement.inventory.totalDevices,
      connectedDevices: measurement.inventory.connectedDevices,
      selectedDevices: measurement.inventory.selectedDevices,
      physicalMemoryGb: measurement.inventory.physicalMemoryGb ?? null,
      offeredMemoryGb: measurement.inventory.offeredMemoryGb ?? null,
      powerLimitWatts: measurement.inventory.powerLimitWatts ?? null,
      profiles,
    },
    topology,
  });
}

/**
 * A digest is mandatory for performance comparison. Matching aliases and
 * revisions are not proof that two workers loaded the same model bytes.
 */
export function benchmarkScenarioCanBeCompared(
  measurement: Pick<BenchmarkMeasurement, "model" | "scenarioFingerprint">,
): boolean {
  return Boolean(
    measurement.model.digest?.trim()
    && /^sha256:[0-9a-f]{64}$/i.test(measurement.scenarioFingerprint),
  );
}

function canonicalProfile(profile: BenchmarkDeviceProfile) {
  return {
    nodeId: nullableNormalized(profile.nodeId ?? null),
    label: normalized(profile.label),
    kind: profile.kind,
    count: profile.count,
    backend: nullableNormalized(profile.backend ?? null),
    precision: nullableNormalized(profile.precision ?? null),
    memoryGb: profile.memoryGb,
    physicalMemoryGb: profile.physicalMemoryGb ?? null,
    offeredMemoryGb: profile.offeredMemoryGb ?? null,
    powerLimitWatts: profile.powerLimitWatts ?? null,
  };
}

function canonicalTopology(topology: BenchmarkTopology) {
  return {
    digest: nullableNormalized(topology.digest),
    stageCount: topology.stageCount,
    boundaries: topology.boundaries.slice(),
    nodeIds: sortedNormalized(topology.nodeIds),
    routeClasses: sortedNormalized(topology.routeClasses),
  };
}

function normalized(value: string): string {
  return value.trim().toLowerCase();
}

function nullableNormalized(value: string | null): string | null {
  const result = value?.trim().toLowerCase();
  return result ? result : null;
}

function sortedNormalized(values: readonly string[]): string[] {
  return [...new Set(values.map(normalized).filter(Boolean))].sort();
}

function ordinal(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
