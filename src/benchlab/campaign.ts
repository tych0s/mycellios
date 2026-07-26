export type BenchmarkCampaignPhase = "warmup" | "measurement";

export interface BenchmarkCampaignAttempt {
  phase: BenchmarkCampaignPhase;
  sampleIndex: number;
  attempt: number;
}

export interface BenchmarkCampaignFailure {
  phase: BenchmarkCampaignPhase;
  sampleIndex: number;
  attempt: number;
  retryable: boolean;
  recovered: boolean;
  message: string;
}

export interface BenchmarkSeriesSummary {
  count: number;
  minimum: number;
  maximum: number;
  mean: number;
  standardDeviation: number;
  p5: number;
  p50: number;
  p95: number;
  coefficientOfVariationPct: number | null;
  confidenceHalfWidthPct: number | null;
}

export interface AdaptiveBenchmarkCampaignOptions<T> {
  measure(attempt: BenchmarkCampaignAttempt): Promise<T>;
  score(sample: T): number;
  retryable?(error: unknown): boolean;
  sleep?(milliseconds: number): Promise<void>;
  warmupSamples?: number;
  minimumSamples?: number;
  maximumSamples?: number;
  retriesPerSample?: number;
  retryDelayMs?: number;
  targetConfidenceHalfWidthPct?: number;
}

export interface AdaptiveBenchmarkCampaignResult<T> {
  warmups: T[];
  samples: T[];
  failures: BenchmarkCampaignFailure[];
  summary: BenchmarkSeriesSummary | null;
  stable: boolean;
  stoppedBecause: "confidence_reached" | "maximum_samples" | "insufficient_samples";
}

const DEFAULT_WARMUP_SAMPLES = 1;
const DEFAULT_MINIMUM_SAMPLES = 7;
const DEFAULT_MAXIMUM_SAMPLES = 15;
const DEFAULT_RETRIES_PER_SAMPLE = 2;
const DEFAULT_RETRY_DELAY_MS = 1_000;
const DEFAULT_TARGET_CONFIDENCE_HALF_WIDTH_PCT = 8;
const Z_95 = 1.96;

/**
 * Runs a real benchmark campaign without counting model warm-up as performance.
 *
 * Availability failures are retained as evidence even when a later attempt
 * recovers. The campaign only stops early after the minimum number of samples
 * and a 95% confidence half-width below the configured target. This keeps a
 * lucky first response from being reported as a speed improvement.
 */
export async function runAdaptiveBenchmarkCampaign<T>(
  options: AdaptiveBenchmarkCampaignOptions<T>,
): Promise<AdaptiveBenchmarkCampaignResult<T>> {
  const warmupSamples = boundedInteger(
    options.warmupSamples,
    DEFAULT_WARMUP_SAMPLES,
    0,
    10,
  );
  const minimumSamples = boundedInteger(
    options.minimumSamples,
    DEFAULT_MINIMUM_SAMPLES,
    1,
    100,
  );
  const maximumSamples = boundedInteger(
    options.maximumSamples,
    DEFAULT_MAXIMUM_SAMPLES,
    minimumSamples,
    250,
  );
  const retriesPerSample = boundedInteger(
    options.retriesPerSample,
    DEFAULT_RETRIES_PER_SAMPLE,
    0,
    10,
  );
  const retryDelayMs = boundedInteger(
    options.retryDelayMs,
    DEFAULT_RETRY_DELAY_MS,
    0,
    60_000,
  );
  const targetConfidenceHalfWidthPct = finitePositive(
    options.targetConfidenceHalfWidthPct,
    DEFAULT_TARGET_CONFIDENCE_HALF_WIDTH_PCT,
  );
  const retryable = options.retryable ?? (() => false);
  const sleep = options.sleep ?? defaultSleep;
  const failures: BenchmarkCampaignFailure[] = [];
  const warmups: T[] = [];
  const samples: T[] = [];

  for (let sampleIndex = 0; sampleIndex < warmupSamples; sampleIndex += 1) {
    const result = await attemptSample(
      "warmup",
      sampleIndex,
      retriesPerSample,
      retryDelayMs,
      options.measure,
      retryable,
      sleep,
      failures,
    );
    if (result !== null) warmups.push(result);
  }

  let stoppedBecause: AdaptiveBenchmarkCampaignResult<T>["stoppedBecause"] =
    "maximum_samples";
  for (let sampleIndex = 0; sampleIndex < maximumSamples; sampleIndex += 1) {
    const result = await attemptSample(
      "measurement",
      sampleIndex,
      retriesPerSample,
      retryDelayMs,
      options.measure,
      retryable,
      sleep,
      failures,
    );
    if (result === null) continue;
    const score = options.score(result);
    if (!Number.isFinite(score) || score < 0) {
      failures.push({
        phase: "measurement",
        sampleIndex,
        attempt: 0,
        retryable: false,
        recovered: false,
        message: "benchmark_sample_score_is_not_finite_and_non_negative",
      });
      continue;
    }
    samples.push(result);

    if (samples.length < minimumSamples) continue;
    const current = summarizeBenchmarkSeries(samples.map(options.score));
    if (
      current.confidenceHalfWidthPct !== null
      && current.confidenceHalfWidthPct <= targetConfidenceHalfWidthPct
    ) {
      stoppedBecause = "confidence_reached";
      break;
    }
  }

  const summary = samples.length > 0
    ? summarizeBenchmarkSeries(samples.map(options.score))
    : null;
  const stable = samples.length >= minimumSamples
    && summary !== null
    && summary.confidenceHalfWidthPct !== null
    && summary.confidenceHalfWidthPct <= targetConfidenceHalfWidthPct;
  if (samples.length < minimumSamples) stoppedBecause = "insufficient_samples";

  return {
    warmups,
    samples,
    failures,
    summary,
    stable,
    stoppedBecause,
  };
}

export function summarizeBenchmarkSeries(values: readonly number[]): BenchmarkSeriesSummary {
  if (values.length === 0) {
    throw new Error("benchmark_series_requires_at_least_one_value");
  }
  if (values.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error("benchmark_series_values_must_be_finite_and_non_negative");
  }
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.length > 1
    ? values.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / (values.length - 1)
    : 0;
  const standardDeviation = Math.sqrt(variance);
  const coefficientOfVariationPct = mean > 0 ? (standardDeviation / mean) * 100 : null;
  const confidenceHalfWidthPct = mean > 0
    ? ((Z_95 * standardDeviation) / Math.sqrt(values.length) / mean) * 100
    : null;
  return {
    count: values.length,
    minimum: Math.min(...values),
    maximum: Math.max(...values),
    mean,
    standardDeviation,
    p5: percentile(values, 0.05),
    p50: percentile(values, 0.5),
    p95: percentile(values, 0.95),
    coefficientOfVariationPct,
    confidenceHalfWidthPct,
  };
}

async function attemptSample<T>(
  phase: BenchmarkCampaignPhase,
  sampleIndex: number,
  retriesPerSample: number,
  retryDelayMs: number,
  measure: AdaptiveBenchmarkCampaignOptions<T>["measure"],
  retryable: (error: unknown) => boolean,
  sleep: (milliseconds: number) => Promise<void>,
  failures: BenchmarkCampaignFailure[],
): Promise<T | null> {
  const failureIndexes: number[] = [];
  for (let attempt = 0; attempt <= retriesPerSample; attempt += 1) {
    try {
      const sample = await measure({ phase, sampleIndex, attempt });
      for (const failureIndex of failureIndexes) {
        failures[failureIndex] = { ...failures[failureIndex]!, recovered: true };
      }
      return sample;
    } catch (error) {
      const canRetry = retryable(error);
      failureIndexes.push(failures.length);
      failures.push({
        phase,
        sampleIndex,
        attempt,
        retryable: canRetry,
        recovered: false,
        message: error instanceof Error ? error.message : String(error),
      });
      if (!canRetry || attempt >= retriesPerSample) return null;
      if (retryDelayMs > 0) await sleep(retryDelayMs);
    }
  }
  return null;
}

function percentile(values: readonly number[], fraction: number): number {
  const ordered = values.slice().sort((left, right) => left - right);
  const position = (ordered.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return ordered[lower]!;
  return ordered[lower]! * (1 - (position - lower))
    + ordered[upper]! * (position - lower);
}

function boundedInteger(
  value: number | undefined,
  fallback: number,
  minimum: number,
  maximum: number,
): number {
  const candidate = value === undefined ? fallback : Math.round(value);
  if (!Number.isFinite(candidate)) return fallback;
  return Math.min(maximum, Math.max(minimum, candidate));
}

function finitePositive(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value) && value > 0 ? value : fallback;
}

function defaultSleep(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
