import { z } from "zod";
import { sha256CanonicalEvidence } from "../core/json.js";

export const ENGINE_RUNTIME_PROFILE_SCHEMA =
  "mycellios-engine-runtime-profile/1" as const;
export const ENGINE_RUNTIME_PROFILE_DEFAULT_MAXIMUM_AGE_MS = 24 * 60 * 60_000;
export const ENGINE_RUNTIME_PROFILE_DEFAULT_MAXIMUM_CONFIDENCE_HALF_WIDTH_PCT = 20;

const digestSchema = z.string().regex(/^sha256:[0-9a-f]{64}$/);
const identifierSchema = z.string()
  .min(1)
  .max(128)
  .regex(/^[A-Za-z0-9][A-Za-z0-9._:+-]*$/);
const positiveSafeInteger = z.number().int().positive().safe();
const positiveFinite = z.number().positive().finite().max(1_000_000_000_000);

const engineRuntimeCapacitySchema = z.object({
  contextTokens: positiveSafeInteger.max(16_777_216),
  maxLayerCount: positiveSafeInteger.max(1_000_000),
  kvBytesPerToken: positiveSafeInteger,
  maxKvTokens: positiveSafeInteger.max(16_777_216),
  usableMemoryBytes: positiveSafeInteger,
}).strict();

const engineRuntimeCostsSchema = z.object({
  decodeMsPerTokenP50: positiveFinite,
  decodeMsPerTokenP95: positiveFinite,
  prefillMsPerTokenP50: positiveFinite,
  prefillMsPerTokenP95: positiveFinite,
  verifyMsPerTokenP50: positiveFinite,
  verifyMsPerTokenP95: positiveFinite,
  decodeScale: z.number().positive().finite().min(0.01).max(100),
  prefillScale: z.number().positive().finite().min(0.01).max(100),
}).strict();

const engineRuntimeFeaturesSchema = z.object({
  fastKernel: z.boolean(),
  graphMode: z.enum(["available", "unavailable"]),
  roles: z.array(z.enum(["head", "middle", "tail", "draft", "auxiliary"]))
    .min(1)
    .max(5),
}).strict();

const engineRuntimeMeasurementBaseSchema = z.object({
  measuredAt: z.string().datetime({ offset: true }),
  samples: z.number().int().min(7).max(100_000),
  confidenceHalfWidthPct: z.number().nonnegative().finite().max(100),
  capacity: engineRuntimeCapacitySchema,
  costs: engineRuntimeCostsSchema,
  features: engineRuntimeFeaturesSchema,
}).strict();

export const engineRuntimeMeasurementSchema = engineRuntimeMeasurementBaseSchema
  .superRefine((measurement, context) => {
  validateMeasurement(measurement, context);
});

const engineRuntimeProfileInputSchema = z.object({
  descriptorDigest: digestSchema,
  certificationId: digestSchema,
  artifactManifestDigest: digestSchema,
  sourceId: digestSchema,
  hardwareFingerprintSha256: digestSchema,
  workerId: identifierSchema,
  sessionId: identifierSchema,
  nodeId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
  modelId: z.string().trim().min(1).max(512),
  modelRevision: z.string().regex(/^[0-9a-f]{40}$/),
  backend: z.enum(["cpu", "cuda", "rocm", "directml", "mps", "vulkan", "webgpu"]),
  runtimeAbi: identifierSchema,
  quantization: z.string().trim().min(1).max(64),
  measuredAt: z.string().datetime({ offset: true }),
  expiresAt: z.string().datetime({ offset: true }),
  samples: engineRuntimeMeasurementBaseSchema.shape.samples,
  confidenceHalfWidthPct: engineRuntimeMeasurementBaseSchema.shape.confidenceHalfWidthPct,
  capacity: engineRuntimeCapacitySchema,
  costs: engineRuntimeCostsSchema,
  features: engineRuntimeFeaturesSchema,
  evidence: z.object({
    deploymentCanaryEvidenceId: digestSchema,
    runtimePerformanceEvidenceId: digestSchema,
  }).strict(),
}).strict().superRefine((profile, context) => {
  if (Date.parse(profile.expiresAt) <= Date.parse(profile.measuredAt)) {
    context.addIssue({
      code: "custom",
      message: "engine_runtime_profile_window_is_invalid",
      path: ["expiresAt"],
    });
  }
  validateMeasurement(profile, context);
});

export const engineRuntimeProfileSchema = engineRuntimeProfileInputSchema.extend({
  schema: z.literal(ENGINE_RUNTIME_PROFILE_SCHEMA),
  profileId: digestSchema,
}).strict().superRefine((profile, context) => {
  const { schema: _schema, profileId: _profileId, ...input } = profile;
  if (profile.profileId !== sha256CanonicalEvidence(input)) {
    context.addIssue({
      code: "custom",
      message: "engine_runtime_profile_seal_is_invalid",
      path: ["profileId"],
    });
  }
});

export type EngineRuntimeProfileInput = z.infer<typeof engineRuntimeProfileInputSchema>;
export type EngineRuntimeProfile = z.infer<typeof engineRuntimeProfileSchema>;
export type EngineRuntimeMeasurement = z.infer<typeof engineRuntimeMeasurementSchema>;

export interface EngineRuntimeProfilePolicy {
  nowMs?: number;
  maximumAgeMs?: number;
  maximumConfidenceHalfWidthPct?: number;
  descriptorDigest?: string;
  certificationId?: string;
  artifactManifestDigest?: string;
  backend?: EngineRuntimeProfile["backend"];
  runtimeAbi?: string;
  quantization?: string;
  contextTokens?: number;
}

function validateMeasurement(
  measurement: Pick<EngineRuntimeProfileInput,
    "capacity" | "costs" | "features">,
  context: z.RefinementCtx,
): void {
  if (measurement.capacity.maxKvTokens < measurement.capacity.contextTokens) {
    context.addIssue({
      code: "custom",
      message: "engine_runtime_profile_kv_capacity_is_insufficient",
      path: ["capacity", "maxKvTokens"],
    });
  }
  if (
    measurement.costs.decodeMsPerTokenP50 > measurement.costs.decodeMsPerTokenP95
    || measurement.costs.prefillMsPerTokenP50 > measurement.costs.prefillMsPerTokenP95
    || measurement.costs.verifyMsPerTokenP50 > measurement.costs.verifyMsPerTokenP95
  ) {
    context.addIssue({
      code: "custom",
      message: "engine_runtime_profile_cost_distribution_is_invalid",
      path: ["costs"],
    });
  }
  if (new Set(measurement.features.roles).size !== measurement.features.roles.length) {
    context.addIssue({
      code: "custom",
      message: "engine_runtime_profile_role_is_duplicated",
      path: ["features", "roles"],
    });
  }
}

export function sealEngineRuntimeProfile(
  value: EngineRuntimeProfileInput,
): EngineRuntimeProfile {
  const input = engineRuntimeProfileInputSchema.parse(value);
  return engineRuntimeProfileSchema.parse({
    schema: ENGINE_RUNTIME_PROFILE_SCHEMA,
    profileId: sha256CanonicalEvidence(input),
    ...input,
  });
}

export function requireEligibleEngineRuntimeProfile(
  value: unknown,
  policy: EngineRuntimeProfilePolicy = {},
): EngineRuntimeProfile {
  const profile = engineRuntimeProfileSchema.parse(value);
  const nowMs = policy.nowMs ?? Date.now();
  const maximumAgeMs = policy.maximumAgeMs
    ?? ENGINE_RUNTIME_PROFILE_DEFAULT_MAXIMUM_AGE_MS;
  const maximumConfidence = policy.maximumConfidenceHalfWidthPct
    ?? ENGINE_RUNTIME_PROFILE_DEFAULT_MAXIMUM_CONFIDENCE_HALF_WIDTH_PCT;
  if (!Number.isFinite(nowMs) || !Number.isFinite(maximumAgeMs) || maximumAgeMs <= 0) {
    throw new Error("engine_runtime_profile_policy_is_invalid");
  }
  const measuredAt = Date.parse(profile.measuredAt);
  if (measuredAt > nowMs + 60_000) {
    throw new Error("engine_runtime_profile_is_from_the_future");
  }
  if (nowMs - measuredAt > maximumAgeMs || Date.parse(profile.expiresAt) <= nowMs) {
    throw new Error("engine_runtime_profile_is_stale");
  }
  if (profile.confidenceHalfWidthPct > maximumConfidence) {
    throw new Error("engine_runtime_profile_confidence_is_too_low");
  }
  for (const [field, expected] of [
    ["descriptorDigest", policy.descriptorDigest],
    ["certificationId", policy.certificationId],
    ["artifactManifestDigest", policy.artifactManifestDigest],
    ["backend", policy.backend],
    ["runtimeAbi", policy.runtimeAbi],
    ["quantization", policy.quantization],
  ] as const) {
    if (expected !== undefined && profile[field] !== expected) {
      throw new Error(`engine_runtime_profile_${field}_mismatch`);
    }
  }
  if (
    policy.contextTokens !== undefined
    && (
      !Number.isSafeInteger(policy.contextTokens)
      || policy.contextTokens <= 0
      || profile.capacity.contextTokens < policy.contextTokens
      || profile.capacity.maxKvTokens < policy.contextTokens
    )
  ) {
    throw new Error("engine_runtime_profile_context_is_insufficient");
  }
  return profile;
}
