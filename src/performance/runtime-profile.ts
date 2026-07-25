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

export const COORDINATOR_RUNTIME_PERFORMANCE_EVIDENCE_SCHEMA =
  "mycellios-coordinator-runtime-performance/1" as const;

export interface CoordinatorRuntimePerformanceEvidence {
  schema: typeof COORDINATOR_RUNTIME_PERFORMANCE_EVIDENCE_SCHEMA;
  evidenceId: string;
  challengeId: string;
  nonce: string;
  workerId: string;
  sessionId: string;
  nodeId: string;
  issuedAt: string;
  expiresAt: string;
  observedAt: string;
  profile: RuntimePerformanceProfile;
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

export interface CoordinatorPlannerProfilePolicy extends PlannerProfilePolicy {
  workerId?: string;
  nodeId?: string;
  sessionId?: string;
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

/**
 * Placement consumes only coordinator-owned observations. The nested profile
 * hash protects content integrity; the outer evidence proves that the profile
 * arrived in response to a fresh challenge on one authenticated worker
 * session. It deliberately does not pretend to be hardware attestation.
 */
export function plannerScalesFromCoordinatorEvidence(
  evidence: CoordinatorRuntimePerformanceEvidence,
  policy: CoordinatorPlannerProfilePolicy = {},
): PlannerPerformanceScales {
  validateCoordinatorRuntimePerformanceEvidence(evidence);
  if (policy.workerId && evidence.workerId !== policy.workerId) {
    throw new Error("runtime_performance_evidence_worker_mismatch");
  }
  if (policy.nodeId && evidence.nodeId !== policy.nodeId) {
    throw new Error("runtime_performance_evidence_node_mismatch");
  }
  if (policy.sessionId && evidence.sessionId !== policy.sessionId) {
    throw new Error("runtime_performance_evidence_session_mismatch");
  }
  return plannerScalesFromProfile(evidence.profile, policy);
}

export function createCoordinatorRuntimePerformanceEvidence(
  input: Omit<CoordinatorRuntimePerformanceEvidence, "schema" | "evidenceId">,
): CoordinatorRuntimePerformanceEvidence {
  const normalized = normalizeCoordinatorEvidence(input);
  return {
    schema: COORDINATOR_RUNTIME_PERFORMANCE_EVIDENCE_SCHEMA,
    evidenceId: coordinatorEvidenceDigest(normalized),
    ...normalized,
  };
}

export function validateCoordinatorRuntimePerformanceEvidence(
  evidence: CoordinatorRuntimePerformanceEvidence,
): void {
  if (evidence.schema !== COORDINATOR_RUNTIME_PERFORMANCE_EVIDENCE_SCHEMA) {
    throw new Error("runtime_performance_evidence_schema_is_invalid");
  }
  const normalized = normalizeCoordinatorEvidence({
    challengeId: evidence.challengeId,
    nonce: evidence.nonce,
    workerId: evidence.workerId,
    sessionId: evidence.sessionId,
    nodeId: evidence.nodeId,
    issuedAt: evidence.issuedAt,
    expiresAt: evidence.expiresAt,
    observedAt: evidence.observedAt,
    profile: evidence.profile,
  });
  if (evidence.evidenceId !== coordinatorEvidenceDigest(normalized)) {
    throw new Error("runtime_performance_evidence_integrity_is_invalid");
  }
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

const coordinatorEvidenceInputSchema = z.object({
  challengeId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/),
  nonce: z.string().length(43).regex(/^[A-Za-z0-9_-]+$/),
  workerId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/),
  sessionId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/),
  nodeId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
  issuedAt: z.string().datetime({ offset: true }),
  expiresAt: z.string().datetime({ offset: true }),
  observedAt: z.string().datetime({ offset: true }),
  profile: runtimePerformanceProfileSchema,
}).strict().superRefine((input, context) => {
  const issuedAt = Date.parse(input.issuedAt);
  const expiresAt = Date.parse(input.expiresAt);
  const observedAt = Date.parse(input.observedAt);
  if (expiresAt <= issuedAt) {
    context.addIssue({
      code: "custom",
      message: "runtime_performance_challenge_expiry_is_invalid",
      path: ["expiresAt"],
    });
  }
  if (observedAt < issuedAt || observedAt > expiresAt) {
    context.addIssue({
      code: "custom",
      message: "runtime_performance_observation_is_outside_challenge",
      path: ["observedAt"],
    });
  }
  const measuredAt = Date.parse(input.profile.measuredAt);
  if (measuredAt < issuedAt - 5_000 || measuredAt > observedAt + 5_000) {
    context.addIssue({
      code: "custom",
      message: "runtime_performance_profile_was_not_measured_for_challenge",
      path: ["profile", "measuredAt"],
    });
  }
});

export const coordinatorRuntimePerformanceEvidenceSchema =
  coordinatorEvidenceInputSchema.extend({
    schema: z.literal(COORDINATOR_RUNTIME_PERFORMANCE_EVIDENCE_SCHEMA),
    evidenceId: z.string().regex(/^sha256:[0-9a-f]{64}$/),
  }).strict().superRefine((evidence, context) => {
    try {
      validateCoordinatorRuntimePerformanceEvidence(evidence);
    } catch (error) {
      context.addIssue({
        code: "custom",
        message: error instanceof Error ? error.message : String(error),
      });
    }
  });

function normalizeCoordinatorEvidence(
  input: Omit<CoordinatorRuntimePerformanceEvidence, "schema" | "evidenceId">,
): Omit<CoordinatorRuntimePerformanceEvidence, "schema" | "evidenceId"> {
  const parsed = coordinatorEvidenceInputSchema.parse(input);
  return {
    challengeId: parsed.challengeId,
    nonce: parsed.nonce,
    workerId: parsed.workerId,
    sessionId: parsed.sessionId,
    nodeId: parsed.nodeId,
    issuedAt: new Date(Date.parse(parsed.issuedAt)).toISOString(),
    expiresAt: new Date(Date.parse(parsed.expiresAt)).toISOString(),
    observedAt: new Date(Date.parse(parsed.observedAt)).toISOString(),
    profile: structuredClone(parsed.profile),
  };
}

function coordinatorEvidenceDigest(
  input: Omit<CoordinatorRuntimePerformanceEvidence, "schema" | "evidenceId">,
): string {
  return `sha256:${createHash("sha256")
    .update(JSON.stringify(input))
    .digest("hex")}`;
}
