/**
 * Coordinator round-trip telemetry for the worker agent.
 *
 * Why this exists: `agent.ts` used to publish `coordinatorRttMs: 0` at startup
 * and nobody ever wrote it again, so every consumer downstream —
 * `connected-executor-activation.ts` (link cost), `planners.ts::nodeLatencyScore`,
 * `cost-model.ts::rawTransferMs` and `scheduler.ts` — planned placement on a
 * constant zero. Measured RTT on the swarm ranges 54-437 ms, and picking the
 * wrong node costs up to 8x, so the zero is not a rounding error: it silently
 * disables latency-aware placement.
 *
 * The samples come from WebSocket-level ping/pong, which the peer's `ws`
 * library answers automatically. That means no protocol change, no extra
 * message type, and no coordinator-side work — and it measures the same path
 * the control plane actually uses.
 */

/** Ignore absurd samples: a scheduler timer hiccup is not a network fact. */
const MAX_PLAUSIBLE_RTT_MS = 60_000;

export interface RttTrackerOptions {
  /**
   * EWMA weight for each new sample. Higher reacts faster to route changes,
   * lower rejects jitter. 0.25 keeps ~90% of the weight in the last ~8 samples,
   * which at a 5 s heartbeat is a ~40 s window — long enough to survive one
   * bad sample, short enough to notice a node moving continents.
   */
  alpha?: number;
}

export interface RttSnapshot {
  /** EWMA of the round trip, or null while no sample has landed yet. */
  readonly ewmaMs: number | null;
  /** Lowest sample seen. Useful to separate propagation from queueing. */
  readonly minMs: number | null;
  /** Most recent sample. */
  readonly lastMs: number | null;
  readonly samples: number;
}

/**
 * Exponentially weighted round-trip tracker.
 *
 * Deliberately does no I/O and owns no timer: the caller drives it. That keeps
 * it testable without a socket and without faking time.
 */
export class CoordinatorRttTracker {
  private readonly alpha: number;
  private ewma: number | null = null;
  private min: number | null = null;
  private last: number | null = null;
  private count = 0;
  private pendingSince: number | null = null;

  constructor(options: RttTrackerOptions = {}) {
    const alpha = options.alpha ?? 0.25;
    if (!(alpha > 0 && alpha <= 1)) {
      throw new Error("RTT EWMA alpha must be in (0, 1]");
    }
    this.alpha = alpha;
  }

  /**
   * Records that a ping left at `nowMs`. A second call before the pong lands
   * replaces the timestamp rather than queueing: we only ever have one ping in
   * flight, and pairing a pong with the older ping would overstate the RTT.
   */
  markPingSent(nowMs: number): void {
    this.pendingSince = nowMs;
  }

  /**
   * Pairs a pong with the outstanding ping and folds the sample in.
   * Returns the sample in ms, or null when there was nothing outstanding or
   * the sample was not plausible.
   */
  markPongReceived(nowMs: number): number | null {
    const sentAt = this.pendingSince;
    this.pendingSince = null;
    if (sentAt === null) return null;
    const sample = nowMs - sentAt;
    if (!Number.isFinite(sample) || sample < 0 || sample > MAX_PLAUSIBLE_RTT_MS) {
      return null;
    }
    return this.observe(sample);
  }

  /** Folds a round-trip sample in directly. Exposed for probes that time their own request. */
  observe(sampleMs: number): number | null {
    if (!Number.isFinite(sampleMs) || sampleMs < 0 || sampleMs > MAX_PLAUSIBLE_RTT_MS) {
      return null;
    }
    this.ewma = this.ewma === null ? sampleMs : this.alpha * sampleMs + (1 - this.alpha) * this.ewma;
    this.min = this.min === null ? sampleMs : Math.min(this.min, sampleMs);
    this.last = sampleMs;
    this.count += 1;
    return sampleMs;
  }

  /** Drops the outstanding ping. Call on reconnect so a stale one cannot pair. */
  reset(): void {
    this.pendingSince = null;
  }

  snapshot(): RttSnapshot {
    return { ewmaMs: this.ewma, minMs: this.min, lastMs: this.last, samples: this.count };
  }

  /**
   * The value to publish in capabilities.
   *
   * Returns `null` — not 0 — while unmeasured, so callers must decide what an
   * unmeasured edge means instead of inheriting a free "this node is adjacent".
   * That inversion is the whole point: the previous default made an unmeasured
   * link look like the best possible one.
   */
  publishedRttMs(): number | null {
    return this.ewma === null ? null : Math.round(this.ewma * 100) / 100;
  }
}
