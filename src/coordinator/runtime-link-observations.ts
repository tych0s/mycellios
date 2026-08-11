export interface RuntimeLinkObservation {
  fromNodeId: string;
  toNodeId: string;
  measuredAt: number;
  validUntil: number;
  rttP50Ms: number;
  rttP95Ms: number;
  jitterP95Ms: number;
  goodputMbpsP50: number;
  successfulSamples: number;
  failedSamples: number;
  availability: number;
  confidence: number;
  transportMode?: "direct" | "relay";
}

export interface RuntimeLinkSample {
  fromNodeId: string;
  toNodeId: string;
  measuredAt: number;
  rttMs: number | null;
  goodputMbps: number | null;
  transportMode: "direct" | "relay";
}

export const DEFAULT_RUNTIME_LINK_SAMPLE_AGE_MS = 5 * 60_000;
export const DEFAULT_RUNTIME_LINK_SAMPLES_PER_LINK = 32;

export function selectPreferredRuntimeLinkObservation(
  observations: readonly RuntimeLinkObservation[],
  minimumDirectAvailability = 0.9,
): RuntimeLinkObservation | undefined {
  if (
    !Number.isFinite(minimumDirectAvailability)
    || minimumDirectAvailability <= 0
    || minimumDirectAvailability > 1
  ) throw new Error("runtime_link_direct_availability_policy_is_invalid");
  const direct = observations.find((observation) => observation.transportMode === "direct");
  const relay = observations.find((observation) =>
    observation.transportMode === "relay" || observation.transportMode === undefined
  );
  return direct && direct.availability >= minimumDirectAvailability
    ? direct
    : relay ?? direct;
}

/**
 * Rolling, bounded evidence for the exact path currently carrying stage bytes.
 * Relay probes and observed direct streams are independent populations. A NAT
 * candidate is never evidence: only a committed transport that moved bytes or
 * a failed direct negotiation contributes a sample.
 */
export class RuntimeLinkObservationStore {
  private readonly samples = new Map<string, RuntimeLinkSample[]>();

  constructor(
    private readonly maxSampleAgeMs = DEFAULT_RUNTIME_LINK_SAMPLE_AGE_MS,
    private readonly maxSamplesPerLink = DEFAULT_RUNTIME_LINK_SAMPLES_PER_LINK,
    initialSamples: readonly RuntimeLinkSample[] = [],
  ) {
    if (!Number.isFinite(maxSampleAgeMs) || maxSampleAgeMs <= 0) {
      throw new Error("runtime_link_sample_age_must_be_positive");
    }
    if (!Number.isInteger(maxSamplesPerLink) || maxSamplesPerLink < 1) {
      throw new Error("runtime_link_sample_limit_must_be_positive");
    }
    for (const sample of initialSamples) {
      this.record(sample.fromNodeId, sample.toNodeId, sample);
    }
  }

  recordSuccess(
    fromNodeId: string,
    toNodeId: string,
    rttMs: number,
    goodputMbps: number,
    measuredAt = Date.now(),
    transportMode: RuntimeLinkSample["transportMode"] = "relay",
  ): void {
    if (!Number.isFinite(rttMs) || rttMs <= 0) {
      throw new Error("runtime_link_rtt_must_be_positive");
    }
    if (!Number.isFinite(goodputMbps) || goodputMbps <= 0) {
      throw new Error("runtime_link_goodput_must_be_positive");
    }
    this.record(fromNodeId, toNodeId, {
      fromNodeId,
      toNodeId,
      measuredAt,
      rttMs,
      goodputMbps,
      transportMode,
    });
  }

  recordFailure(
    fromNodeId: string,
    toNodeId: string,
    measuredAt = Date.now(),
    transportMode: RuntimeLinkSample["transportMode"] = "relay",
  ): void {
    this.record(fromNodeId, toNodeId, {
      fromNodeId,
      toNodeId,
      measuredAt,
      rttMs: null,
      goodputMbps: null,
      transportMode,
    });
  }

  observations(now = Date.now()): RuntimeLinkObservation[] {
    const result: RuntimeLinkObservation[] = [];
    for (const [key, samples] of this.samples) {
      const fresh = samples.filter((sample) => now - sample.measuredAt <= this.maxSampleAgeMs);
      if (fresh.length === 0) {
        this.samples.delete(key);
        continue;
      }
      if (fresh.length !== samples.length) this.samples.set(key, fresh);
      const successes = fresh
        .filter((sample): sample is RuntimeLinkSample & {
          rttMs: number;
          goodputMbps: number;
        } => sample.rttMs !== null && sample.goodputMbps !== null);
      const rtts = successes
        .map((sample) => sample.rttMs)
        .sort((left, right) => left - right);
      if (rtts.length === 0) continue;
      const goodputs = successes
        .map((sample) => sample.goodputMbps)
        .sort((left, right) => left - right);
      const [fromNodeId, toNodeId, transportMode] = splitKey(key);
      const measuredAt = Math.max(...fresh.map((sample) => sample.measuredAt));
      const medianRtt = percentile(rtts, 0.5);
      const jitter = successes
        .map((sample) => Math.abs(sample.rttMs - medianRtt))
        .sort((left, right) => left - right);
      const availability = rtts.length / fresh.length;
      result.push({
        fromNodeId,
        toNodeId,
        measuredAt,
        validUntil: measuredAt + this.maxSampleAgeMs,
        rttP50Ms: medianRtt,
        rttP95Ms: percentile(rtts, 0.95),
        jitterP95Ms: percentile(jitter, 0.95),
        goodputMbpsP50: percentile(goodputs, 0.5),
        successfulSamples: rtts.length,
        failedSamples: fresh.length - rtts.length,
        availability,
        confidence: Math.min(1, rtts.length / 3) * availability,
        transportMode,
      });
    }
    return result.sort(
      (left, right) =>
        left.fromNodeId.localeCompare(right.fromNodeId)
        || left.toNodeId.localeCompare(right.toNodeId)
        || (left.transportMode ?? "relay").localeCompare(right.transportMode ?? "relay"),
    );
  }

  clearNode(nodeId: string): void {
    for (const key of this.samples.keys()) {
      const [fromNodeId, toNodeId] = splitKey(key);
      if (fromNodeId === nodeId || toNodeId === nodeId) this.samples.delete(key);
    }
  }

  private record(fromNodeId: string, toNodeId: string, sample: RuntimeLinkSample): void {
    if (!fromNodeId || !toNodeId || fromNodeId === toNodeId) {
      throw new Error("runtime_link_nodes_must_be_distinct");
    }
    if (!Number.isFinite(sample.measuredAt) || sample.measuredAt <= 0) {
      throw new Error("runtime_link_measurement_time_is_invalid");
    }
    if (sample.transportMode !== "direct" && sample.transportMode !== "relay") {
      throw new Error("runtime_link_transport_mode_is_invalid");
    }
    const key = linkKey(fromNodeId, toNodeId, sample.transportMode);
    const values = [...(this.samples.get(key) ?? []), sample]
      .sort((left, right) => left.measuredAt - right.measuredAt)
      .slice(-this.maxSamplesPerLink);
    this.samples.set(key, values);
  }
}

function percentile(sorted: readonly number[], probability: number): number {
  const index = Math.max(0, Math.ceil(sorted.length * probability) - 1);
  return sorted[index]!;
}

function linkKey(
  fromNodeId: string,
  toNodeId: string,
  transportMode: RuntimeLinkSample["transportMode"],
): string {
  return `${fromNodeId}\u0000${toNodeId}\u0000${transportMode}`;
}

function splitKey(key: string): [string, string, RuntimeLinkSample["transportMode"]] {
  const first = key.indexOf("\u0000");
  const second = key.indexOf("\u0000", first + 1);
  if (first <= 0 || second <= first + 1 || second === key.length - 1) {
    throw new Error("runtime_link_key_is_corrupt");
  }
  const mode = key.slice(second + 1);
  if (mode !== "direct" && mode !== "relay") throw new Error("runtime_link_key_is_corrupt");
  return [key.slice(0, first), key.slice(first + 1, second), mode];
}
