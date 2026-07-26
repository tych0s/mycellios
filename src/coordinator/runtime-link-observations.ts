export interface RuntimeLinkObservation {
  fromNodeId: string;
  toNodeId: string;
  measuredAt: number;
  rttP50Ms: number;
  rttP95Ms: number;
  goodputMbpsP50: number;
  successfulSamples: number;
  failedSamples: number;
  availability: number;
}

interface RuntimeLinkSample {
  measuredAt: number;
  rttMs: number | null;
  goodputMbps: number | null;
}

const DEFAULT_MAX_SAMPLE_AGE_MS = 5 * 60_000;
const DEFAULT_MAX_SAMPLES_PER_LINK = 32;

/**
 * Rolling, bounded evidence for the exact path currently carrying stage bytes.
 * Today that path is the authenticated worker relay; when direct transports are
 * added the same planner input can be populated by their own probes.
 */
export class RuntimeLinkObservationStore {
  private readonly samples = new Map<string, RuntimeLinkSample[]>();

  constructor(
    private readonly maxSampleAgeMs = DEFAULT_MAX_SAMPLE_AGE_MS,
    private readonly maxSamplesPerLink = DEFAULT_MAX_SAMPLES_PER_LINK,
  ) {
    if (!Number.isFinite(maxSampleAgeMs) || maxSampleAgeMs <= 0) {
      throw new Error("runtime_link_sample_age_must_be_positive");
    }
    if (!Number.isInteger(maxSamplesPerLink) || maxSamplesPerLink < 1) {
      throw new Error("runtime_link_sample_limit_must_be_positive");
    }
  }

  recordSuccess(
    fromNodeId: string,
    toNodeId: string,
    rttMs: number,
    goodputMbps: number,
    measuredAt = Date.now(),
  ): void {
    if (!Number.isFinite(rttMs) || rttMs <= 0) {
      throw new Error("runtime_link_rtt_must_be_positive");
    }
    if (!Number.isFinite(goodputMbps) || goodputMbps <= 0) {
      throw new Error("runtime_link_goodput_must_be_positive");
    }
    this.record(fromNodeId, toNodeId, { measuredAt, rttMs, goodputMbps });
  }

  recordFailure(
    fromNodeId: string,
    toNodeId: string,
    measuredAt = Date.now(),
  ): void {
    this.record(fromNodeId, toNodeId, { measuredAt, rttMs: null, goodputMbps: null });
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
      const [fromNodeId, toNodeId] = splitKey(key);
      result.push({
        fromNodeId,
        toNodeId,
        measuredAt: Math.max(...fresh.map((sample) => sample.measuredAt)),
        rttP50Ms: percentile(rtts, 0.5),
        rttP95Ms: percentile(rtts, 0.95),
        goodputMbpsP50: percentile(goodputs, 0.5),
        successfulSamples: rtts.length,
        failedSamples: fresh.length - rtts.length,
        availability: rtts.length / fresh.length,
      });
    }
    return result.sort(
      (left, right) =>
        left.fromNodeId.localeCompare(right.fromNodeId)
        || left.toNodeId.localeCompare(right.toNodeId),
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
    const key = linkKey(fromNodeId, toNodeId);
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

function linkKey(fromNodeId: string, toNodeId: string): string {
  return `${fromNodeId}\u0000${toNodeId}`;
}

function splitKey(key: string): [string, string] {
  const separator = key.indexOf("\u0000");
  if (separator <= 0 || separator === key.length - 1) {
    throw new Error("runtime_link_key_is_corrupt");
  }
  return [key.slice(0, separator), key.slice(separator + 1)];
}
