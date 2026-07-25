import { createHash } from "node:crypto";
import { z } from "zod";

export const RUNTIME_PERFORMANCE_PROFILE_SCHEMA =
  "mycellios-runtime-performance/1" as const;
export const RUNTIME_PERFORMANCE_PROFILE_MINIMUM_SAMPLES = 7;
export const RUNTIME_PERFORMANCE_PROFILE_DEFAULT_MAXIMUM_AGE_MS =
  7 * 24 * 60 * 60_000;
export const RUNTIME_PERFORMANCE_PROFILE_DEFAULT_MAXIMUM_CONFIDENCE_HALF_WIDTH_PCT =
  20;

export interface PhysicalPerformanceSeries {
  unit: "GB/s" | "TFLOP/s";
  warmupSamples: number;
  samples: number;
  p5: number;
  p50: number;
  p95: number;
  confidenceHalfWidthPct: number;
}

export interface RuntimePerformanceProfileInput {
  measuredAt: string;
  backend: "cuda" | "rocm" | "mps" | "xpu" | "cpu";
  deviceName: string;
  precision: "float16" | "float32";
  source: "physical-microbenchmark" | "runtime-calibration";
  activationCodecId: "fp16";
  decodeMemory: PhysicalPerformanceSeries;
  prefillCompute: PhysicalPerformanceSeries;
  activationCodec: PhysicalPerformanceSeries;
}

export interface RuntimePerformanceProfile extends RuntimePerformanceProfileInput {
  schema: typeof RUNTIME_PERFORMANCE_PROFILE_SCHEMA;
  profileId: string;
}

export interface PlannerPerformanceScales {
  decodeScale: number;
  prefillScale: number;
  codecScale: number;
  profileId: string;
  measuredAt: string;
}

export interface PlannerProfilePolicy {
  maximumAgeMs?: number;
  maximumConfidenceHalfWidthPct?: number;
  requiredSource?: RuntimePerformanceProfileInput["source"];
  now?: number;
}

// Unit anchors make profiles from different machines comparable. They are not
// claimed hardware speeds: measured speed divided by the same fixed anchor
// only establishes a relative planner scale (lower is faster).
const REFERENCE_DECODE_MEMORY_GBPS = 200;
const REFERENCE_PREFILL_TFLOPS = 10;
const REFERENCE_CODEC_GBPS = 1;

export function sealRuntimePerformanceProfile(
  input: RuntimePerformanceProfileInput,
): RuntimePerformanceProfile {
  const normalized = normalizeProfileInput(input);
  return {
    schema: RUNTIME_PERFORMANCE_PROFILE_SCHEMA,
    profileId: profileDigest(normalized),
    ...normalized,
  };
}

export function plannerScalesFromProfile(
  profile: RuntimePerformanceProfile,
  policy: PlannerProfilePolicy = {},
): PlannerPerformanceScales {
  validateRuntimePerformanceProfile(profile);
  if (profile.source !== (policy.requiredSource ?? "physical-microbenchmark")) {
    throw new Error("runtime_performance_profile_source_is_not_physical");
  }
  const now = policy.now ?? Date.now();
  const measuredAt = Date.parse(profile.measuredAt);
  const maximumAgeMs = finitePositive(
    policy.maximumAgeMs,
    RUNTIME_PERFORMANCE_PROFILE_DEFAULT_MAXIMUM_AGE_MS,
  );
  if (measuredAt > now + 60_000) {
    throw new Error("runtime_performance_profile_is_from_the_future");
  }
  if (now - measuredAt > maximumAgeMs) {
    throw new Error("runtime_performance_profile_is_stale");
  }
  const maximumConfidence = finitePositive(
    policy.maximumConfidenceHalfWidthPct,
    RUNTIME_PERFORMANCE_PROFILE_DEFAULT_MAXIMUM_CONFIDENCE_HALF_WIDTH_PCT,
  );
  for (const [name, measurement] of [
    ["decode", profile.decodeMemory],
    ["prefill", profile.prefillCompute],
    ["codec", profile.activationCodec],
  ] as const) {
    if (measurement.confidenceHalfWidthPct > maximumConfidence) {
      throw new Error(`runtime_performance_profile_${name}_confidence_is_too_low`);
    }
  }
  return {
    decodeScale: plannerScale(
      REFERENCE_DECODE_MEMORY_GBPS,
      profile.decodeMemory.p50,
    ),
    prefillScale: plannerScale(
      REFERENCE_PREFILL_TFLOPS,
      profile.prefillCompute.p50,
    ),
    codecScale: plannerScale(
      REFERENCE_CODEC_GBPS,
      profile.activationCodec.p50,
    ),
    profileId: profile.profileId,
    measuredAt: profile.measuredAt,
  };
}

export function validateRuntimePerformanceProfile(
  value: RuntimePerformanceProfile,
): void {
  if (value.schema !== RUNTIME_PERFORMANCE_PROFILE_SCHEMA) {
    throw new Error("runtime_performance_profile_schema_is_invalid");
  }
  const normalized = normalizeProfileInput(value);
  if (value.profileId !== profileDigest(normalized)) {
    throw new Error("runtime_performance_profile_seal_is_invalid");
  }
}

function normalizeProfileInput(
  input: RuntimePerformanceProfileInput,
): RuntimePerformanceProfileInput {
  const measuredAtMs = Date.parse(input.measuredAt);
  if (!Number.isFinite(measuredAtMs)) {
    throw new Error("runtime_performance_profile_measured_at_is_invalid");
  }
  if (!["cuda", "rocm", "mps", "xpu", "cpu"].includes(input.backend)) {
    throw new Error("runtime_performance_profile_backend_is_invalid");
  }
  if (!["float16", "float32"].includes(input.precision)) {
    throw new Error("runtime_performance_profile_precision_is_invalid");
  }
  if (!["physical-microbenchmark", "runtime-calibration"].includes(input.source)) {
    throw new Error("runtime_performance_profile_source_is_invalid");
  }
  if (input.activationCodecId !== "fp16") {
    throw new Error("runtime_performance_profile_codec_is_invalid");
  }
  const deviceName = input.deviceName.trim();
  if (!deviceName || deviceName.length > 256) {
    throw new Error("runtime_performance_profile_device_name_is_invalid");
  }
  return {
    measuredAt: new Date(measuredAtMs).toISOString(),
    backend: input.backend,
    deviceName,
    precision: input.precision,
    source: input.source,
    activationCodecId: input.activationCodecId,
    decodeMemory: normalizeSeries(input.decodeMemory, "GB/s", "decode"),
    prefillCompute: normalizeSeries(input.prefillCompute, "TFLOP/s", "prefill"),
    activationCodec: normalizeSeries(input.activationCodec, "GB/s", "codec"),
  };
}

function normalizeSeries(
  series: PhysicalPerformanceSeries,
  expectedUnit: PhysicalPerformanceSeries["unit"],
  name: string,
): PhysicalPerformanceSeries {
  if (series.unit !== expectedUnit) {
    throw new Error(`runtime_performance_profile_${name}_unit_is_invalid`);
  }
  if (
    !Number.isInteger(series.warmupSamples)
    || series.warmupSamples < 1
    || series.warmupSamples > 1_000
    || !Number.isInteger(series.samples)
    || series.samples < RUNTIME_PERFORMANCE_PROFILE_MINIMUM_SAMPLES
    || series.samples > 10_000
  ) {
    throw new Error(`runtime_performance_profile_${name}_sample_count_is_invalid`);
  }
  const values = [series.p5, series.p50, series.p95, series.confidenceHalfWidthPct];
  if (values.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error(`runtime_performance_profile_${name}_measurement_is_invalid`);
  }
  if (
    series.p5 <= 0
    || series.p50 <= 0
    || series.p95 <= 0
    || series.p5 > series.p50
    || series.p50 > series.p95
    || series.confidenceHalfWidthPct > 100
  ) {
    throw new Error(`runtime_performance_profile_${name}_distribution_is_invalid`);
  }
  return {
    unit: expectedUnit,
    warmupSamples: series.warmupSamples,
    samples: series.samples,
    p5: series.p5,
    p50: series.p50,
    p95: series.p95,
    confidenceHalfWidthPct: series.confidenceHalfWidthPct,
  };
}

function profileDigest(input: RuntimePerformanceProfileInput): string {
  return `sha256:${createHash("sha256")
    .update(JSON.stringify(input))
    .digest("hex")}`;
}

function plannerScale(reference: number, measured: number): number {
  return Math.min(20, Math.max(0.05, reference / measured));
}

function finitePositive(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value) && value > 0
    ? value
    : fallback;
}

const performanceSeriesSchema = z.object({
  unit: z.enum(["GB/s", "TFLOP/s"]),
  warmupSamples: z.number().int().min(1).max(1_000),
  samples: z.number().int()
    .min(RUNTIME_PERFORMANCE_PROFILE_MINIMUM_SAMPLES)
    .max(10_000),
  p5: z.number().positive().finite().max(1_000_000_000),
  p50: z.number().positive().finite().max(1_000_000_000),
  p95: z.number().positive().finite().max(1_000_000_000),
  confidenceHalfWidthPct: z.number().nonnegative().finite().max(100),
}).strict();

/**
 * Strict wire schema for a self-contained physical profile. The structural
 * checks run before the content seal so malformed network input can never
 * reach the normalizer as an unchecked object.
 */
export const runtimePerformanceProfileInputSchema = z.object({
  measuredAt: z.string().datetime({ offset: true }),
  backend: z.enum(["cuda", "rocm", "mps", "xpu", "cpu"]),
  deviceName: z.string().trim().min(1).max(256),
  precision: z.enum(["float16", "float32"]),
  source: z.enum(["physical-microbenchmark", "runtime-calibration"]),
  activationCodecId: z.literal("fp16"),
  decodeMemory: performanceSeriesSchema,
  prefillCompute: performanceSeriesSchema,
  activationCodec: performanceSeriesSchema,
}).strict();

export const runtimePerformanceProfileSchema = runtimePerformanceProfileInputSchema.extend({
  schema: z.literal(RUNTIME_PERFORMANCE_PROFILE_SCHEMA),
  profileId: z.string().regex(/^sha256:[0-9a-f]{64}$/),
}).strict().superRefine((profile, context) => {
  try {
    validateRuntimePerformanceProfile(profile);
  } catch (error) {
    context.addIssue({
      code: "custom",
      message: error instanceof Error ? error.message : String(error),
    });
  }
});
