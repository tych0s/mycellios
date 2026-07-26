/**
 * Derives `decodeScale` from measured worker throughput.
 *
 * Why this exists: `decodeScale` was hardcoded to 1 in both paths that build
 * real node profiles (`connected-executor-activation.ts` and `desktop/main.ts`),
 * so `ProportionalComputePlanner` (`planners.ts:690`, `1 / decodeScale`)
 * divided by a vector of ones and degenerated to an equal split, and the
 * minimax in `cost-model.ts:166` minimised the maximum of a constant vector.
 * The proportional planner was already written; it was switched off by missing
 * telemetry, not by missing design.
 *
 * `decodeScale` is a COST multiplier, not a speed: `parallelism.ts:162` uses
 * `layer.decodeMsAtUnit * node.decodeScale`. A node twice as slow as the
 * reference gets 2.
 *
 * ⚠️ Only `measured` throughput is used. An estimated or defaulted number is a
 * guess about hardware, and planning a layer split on a guess is how the
 * planner ended up trusting a constant in the first place. When a node has no
 * measurement, this returns scale 1 AND says so, so the caller can decide
 * whether to fall back to an equal split rather than silently pretending the
 * split was informed.
 */

export interface ThroughputObservation {
  readonly nodeId: string;
  /** Decode throughput in tokens/second, or null when never measured. */
  readonly measuredTokensPerSecond: number | null;
}

export interface NodeDecodeScale {
  readonly nodeId: string;
  readonly decodeScale: number;
  /** False when this node fell back to 1 because nothing was measured. */
  readonly measured: boolean;
}

export interface DecodeScaleResult {
  readonly scales: readonly NodeDecodeScale[];
  /** True only when EVERY node carried a measurement. */
  readonly fullyMeasured: boolean;
  /** tokens/s of the fastest measured node — the normalisation reference. */
  readonly referenceTokensPerSecond: number | null;
}

/** Keeps one pathological node from being handed the whole model, or none of it. */
const MIN_DECODE_SCALE = 0.1;
const MAX_DECODE_SCALE = 20;

/**
 * Normalises to the FASTEST measured node, so the reference always has
 * `decodeScale = 1` and slower nodes grow above it. Normalising to a fixed
 * absolute reference would make every scale drift when the fleet changes
 * hardware, which is not what the planners expect.
 */
export function deriveDecodeScales(
  observations: readonly ThroughputObservation[],
): DecodeScaleResult {
  const measured = observations.filter(
    (observation): observation is ThroughputObservation & { measuredTokensPerSecond: number } =>
      observation.measuredTokensPerSecond !== null
      && Number.isFinite(observation.measuredTokensPerSecond)
      && observation.measuredTokensPerSecond > 0,
  );

  if (measured.length === 0) {
    return {
      scales: observations.map((observation) => ({
        nodeId: observation.nodeId,
        decodeScale: 1,
        measured: false,
      })),
      fullyMeasured: false,
      referenceTokensPerSecond: null,
    };
  }

  const reference = Math.max(...measured.map((observation) => observation.measuredTokensPerSecond));
  const byId = new Map(measured.map((observation) => [observation.nodeId, observation]));

  const scales = observations.map((observation) => {
    const hit = byId.get(observation.nodeId);
    if (!hit) return { nodeId: observation.nodeId, decodeScale: 1, measured: false };
    const raw = reference / hit.measuredTokensPerSecond;
    const clamped = Math.min(MAX_DECODE_SCALE, Math.max(MIN_DECODE_SCALE, raw));
    return {
      nodeId: observation.nodeId,
      // Two decimals: more precision than the ±20-25% bench noise supports
      // would be false confidence.
      decodeScale: Math.round(clamped * 100) / 100,
      measured: true,
    };
  });

  return {
    scales,
    fullyMeasured: measured.length === observations.length,
    referenceTokensPerSecond: reference,
  };
}

/**
 * ⚠️ Reminder carried from the audit of `external-runtime-c` (25-07-2026):
 * they measured the SAME 13-layer block at 11.5 ms behind an idle desktop CPU
 * and 35-50 ms behind an old or co-tenanted server CPU, while all five of their
 * GPUs benchmarked identically (1523-1527 GB/s). The GPU is not the variable —
 * a block forward is hundreds of small kernel launches and launch cost tracks
 * single-thread CPU speed. Throughput measured on the node captures that
 * automatically; a spec-sheet estimate from GPU model never would. This is the
 * reason `measured` is the only source accepted above.
 */
export const DECODE_SCALE_SOURCE_NOTE =
  "decodeScale is derived only from measured decode throughput; GPU specs alone "
  + "do not predict it because block forward cost tracks single-thread CPU speed.";
