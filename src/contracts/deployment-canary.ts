import { z } from "zod";
import { sha256CanonicalEvidence } from "../core/json.js";

export const DEPLOYMENT_CANARY_EVIDENCE_SCHEMA =
  "mycellios-coordinator-deployment-canary/1" as const;
export const DEPLOYMENT_CANARY_MINIMUM_SAMPLES = 3;
export const DEPLOYMENT_CANARY_MAXIMUM_AGE_MS = 7 * 24 * 60 * 60_000;

export interface DeploymentCanarySample {
  sampleId: string;
  outputTokens: number;
  /**
   * Coordinator wall-clock duration between the observed sample start and
   * completion messages. This intentionally ignores a worker-provided clock.
   */
  activeMs: number;
  /** Coordinator wall-clock delay until the first non-empty token frame. */
  ttftMs: number;
  completed: true;
}

export interface CoordinatorChallengeBinding {
  challengeId: string;
  nonce: string;
  workerId: string;
  sessionId: string;
  issuedAt: string;
  expiresAt: string;
}

export interface DeploymentCanaryEvidenceInput
  extends CoordinatorChallengeBinding {
  model: string;
  modelDigest: string;
  activationId: string;
  promptDigest: string;
  maxOutputTokens: number;
  observedAt: string;
  warmupSamples: number;
  samples: DeploymentCanarySample[];
}

/**
 * Evidence that can only enter coordinator state after a live authenticated
 * session completes a coordinator-issued challenge. `evidenceId` is a content
 * integrity identifier, not a worker signature or remote-attestation claim.
 */
export interface DeploymentCanaryEvidence
  extends DeploymentCanaryEvidenceInput {
  schema: typeof DEPLOYMENT_CANARY_EVIDENCE_SCHEMA;
  evidenceId: string;
}

export interface DeploymentCanaryMetrics {
  tokensPerSecond: number;
  ttftMs: number;
  evidenceId: string;
  activationId: string;
  observedAt: string;
}

const identifierSchema = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/);
const nonceSchema = z.string().length(43).regex(/^[A-Za-z0-9_-]+$/);

const canarySampleSchema = z.object({
  sampleId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
  outputTokens: z.number().int().min(1).max(1_000_000),
  activeMs: z.number().int().min(1).max(3_600_000),
  ttftMs: z.number().int().nonnegative().max(3_600_000),
  completed: z.literal(true),
}).strict();

const canaryInputSchema = z.object({
  challengeId: identifierSchema,
  nonce: nonceSchema,
  workerId: identifierSchema,
  sessionId: identifierSchema,
  issuedAt: z.string().datetime({ offset: true }),
  expiresAt: z.string().datetime({ offset: true }),
  model: z.string().trim().min(1).max(256),
  modelDigest: z.string().regex(/^sha256:[0-9a-f]{64}$/),
  activationId: z.string().trim().min(1).max(256),
  promptDigest: z.string().regex(/^sha256:[0-9a-f]{64}$/),
  maxOutputTokens: z.number().int().min(1).max(32_768),
  observedAt: z.string().datetime({ offset: true }),
  warmupSamples: z.number().int().min(1).max(1_000),
  samples: z.array(canarySampleSchema)
    .min(DEPLOYMENT_CANARY_MINIMUM_SAMPLES)
    .max(1_000),
}).strict().superRefine((input, context) => {
  if (new Set(input.samples.map((sample) => sample.sampleId)).size !== input.samples.length) {
    context.addIssue({
      code: "custom",
      message: "deployment_canary_sample_ids_are_not_unique",
      path: ["samples"],
    });
  }
  const issuedAt = Date.parse(input.issuedAt);
  const expiresAt = Date.parse(input.expiresAt);
  const observedAt = Date.parse(input.observedAt);
  if (expiresAt <= issuedAt) {
    context.addIssue({
      code: "custom",
      message: "deployment_canary_challenge_expiry_is_invalid",
      path: ["expiresAt"],
    });
  }
  if (observedAt < issuedAt || observedAt > expiresAt) {
    context.addIssue({
      code: "custom",
      message: "deployment_canary_observation_is_outside_challenge",
      path: ["observedAt"],
    });
  }
});

export const deploymentCanaryEvidenceSchema = canaryInputSchema.safeExtend({
  schema: z.literal(DEPLOYMENT_CANARY_EVIDENCE_SCHEMA),
  evidenceId: z.string().regex(/^sha256:[0-9a-f]{64}$/),
}).strict().superRefine((evidence, context) => {
  try {
    validateDeploymentCanaryEvidence(evidence);
  } catch (error) {
    context.addIssue({
      code: "custom",
      message: error instanceof Error ? error.message : String(error),
    });
  }
});

/**
 * Called by the coordinator only, after it has matched and consumed the live
 * challenge. Worker registration/configuration paths never call this.
 */
export function createCoordinatorDeploymentCanaryEvidence(
  input: DeploymentCanaryEvidenceInput,
): DeploymentCanaryEvidence {
  const normalized = normalizeCanaryInput(input);
  return {
    schema: DEPLOYMENT_CANARY_EVIDENCE_SCHEMA,
    evidenceId: sha256CanonicalEvidence(normalized),
    ...normalized,
  };
}

export function validateDeploymentCanaryEvidence(
  evidence: DeploymentCanaryEvidence,
): void {
  if (evidence.schema !== DEPLOYMENT_CANARY_EVIDENCE_SCHEMA) {
    throw new Error("deployment_canary_schema_is_invalid");
  }
  const normalized = normalizeCanaryInput({
    challengeId: evidence.challengeId,
    nonce: evidence.nonce,
    workerId: evidence.workerId,
    sessionId: evidence.sessionId,
    issuedAt: evidence.issuedAt,
    expiresAt: evidence.expiresAt,
    model: evidence.model,
    modelDigest: evidence.modelDigest,
    activationId: evidence.activationId,
    promptDigest: evidence.promptDigest,
    maxOutputTokens: evidence.maxOutputTokens,
    observedAt: evidence.observedAt,
    warmupSamples: evidence.warmupSamples,
    samples: evidence.samples,
  });
  if (evidence.evidenceId !== sha256CanonicalEvidence(normalized)) {
    throw new Error("deployment_canary_integrity_is_invalid");
  }
}

export function deploymentMetricsFromCanaryEvidence(
  evidence: DeploymentCanaryEvidence,
  expected: {
    model: string;
    modelDigest: string;
    activationId: string;
    workerId?: string;
    sessionId?: string;
    now?: number;
    maximumAgeMs?: number;
  },
): DeploymentCanaryMetrics {
  validateDeploymentCanaryEvidence(evidence);
  if (evidence.model !== expected.model) {
    throw new Error("deployment_canary_model_mismatch");
  }
  if (evidence.modelDigest !== expected.modelDigest) {
    throw new Error("deployment_canary_model_digest_mismatch");
  }
  if (evidence.activationId !== expected.activationId) {
    throw new Error("deployment_canary_activation_mismatch");
  }
  if (expected.workerId && evidence.workerId !== expected.workerId) {
    throw new Error("deployment_canary_worker_mismatch");
  }
  if (expected.sessionId && evidence.sessionId !== expected.sessionId) {
    throw new Error("deployment_canary_session_mismatch");
  }
  const now = expected.now ?? Date.now();
  const observedAt = Date.parse(evidence.observedAt);
  if (observedAt > now + 60_000) {
    throw new Error("deployment_canary_is_from_the_future");
  }
  if (
    now - observedAt
    > (expected.maximumAgeMs ?? DEPLOYMENT_CANARY_MAXIMUM_AGE_MS)
  ) {
    throw new Error("deployment_canary_is_stale");
  }
  const totalTokens = evidence.samples.reduce(
    (total, sample) => total + sample.outputTokens,
    0,
  );
  const totalActiveMs = evidence.samples.reduce(
    (total, sample) => total + sample.activeMs,
    0,
  );
  const orderedTtft = evidence.samples
    .map((sample) => sample.ttftMs)
    .sort((left, right) => left - right);
  return {
    tokensPerSecond: Number(
      ((totalTokens * 1_000) / totalActiveMs).toFixed(3),
    ),
    ttftMs: orderedTtft[Math.floor(orderedTtft.length / 2)]!,
    evidenceId: evidence.evidenceId,
    activationId: evidence.activationId,
    observedAt: evidence.observedAt,
  };
}

function normalizeCanaryInput(
  input: DeploymentCanaryEvidenceInput,
): DeploymentCanaryEvidenceInput {
  const parsed = canaryInputSchema.parse(input);
  return {
    challengeId: parsed.challengeId,
    nonce: parsed.nonce,
    workerId: parsed.workerId,
    sessionId: parsed.sessionId,
    issuedAt: new Date(Date.parse(parsed.issuedAt)).toISOString(),
    expiresAt: new Date(Date.parse(parsed.expiresAt)).toISOString(),
    model: parsed.model.trim(),
    modelDigest: parsed.modelDigest,
    activationId: parsed.activationId.trim(),
    promptDigest: parsed.promptDigest,
    maxOutputTokens: parsed.maxOutputTokens,
    observedAt: new Date(Date.parse(parsed.observedAt)).toISOString(),
    warmupSamples: parsed.warmupSamples,
    samples: parsed.samples
      .map((sample) => ({ ...sample }))
      .sort((left, right) => left.sampleId.localeCompare(right.sampleId)),
  };
}
