import { z } from "zod";

export const EVIDENCE_CHALLENGE_TTL_MS = 10 * 60_000;
export const EVIDENCE_CHALLENGE_SCHEMA = "mycellios-evidence-challenge/1" as const;
export const DEPLOYMENT_CANARY_CHALLENGE_WARMUPS = 1;
export const DEPLOYMENT_CANARY_CHALLENGE_SAMPLES = 3;
export const DEPLOYMENT_CANARY_CHALLENGE_MAX_OUTPUT_TOKENS = 64;
export const DEPLOYMENT_CANARY_CHALLENGE_PROMPT =
  "Produce a concise deterministic explanation of why measured distributed inference matters.";

const identifierSchema = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/);
const nonceSchema = z.string().length(43).regex(/^[A-Za-z0-9_-]+$/);
const challengeBase = {
  schema: z.literal(EVIDENCE_CHALLENGE_SCHEMA),
  challengeId: identifierSchema,
  nonce: nonceSchema,
  sessionId: identifierSchema,
  workerId: identifierSchema,
  issuedAt: z.string().datetime({ offset: true }),
  expiresAt: z.string().datetime({ offset: true }),
};

export const deploymentCanaryChallengeSchema = z.object({
  ...challengeBase,
  kind: z.literal("deployment-canary"),
  deploymentId: identifierSchema,
  model: z.string().trim().min(1).max(256),
  modelDigest: z.string().regex(/^sha256:[0-9a-f]{64}$/),
  activationId: z.string().trim().min(1).max(256),
  prompt: z.string().min(1).max(4_096),
  promptDigest: z.string().regex(/^sha256:[0-9a-f]{64}$/),
  maxOutputTokens: z.number().int().min(1).max(32_768),
  warmupSamples: z.number().int().min(1).max(16),
  samples: z.number().int().min(3).max(16),
}).strict();

export const runtimePerformanceChallengeSchema = z.object({
  ...challengeBase,
  kind: z.literal("runtime-performance"),
  nodeId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
  backend: z.enum(["cuda", "rocm", "mps", "xpu", "cpu"]),
  deviceName: z.string().trim().min(1).max(256),
  precision: z.enum(["float16", "float32"]),
  source: z.literal("physical-microbenchmark"),
  activationCodecId: z.literal("fp16"),
  minimumWarmupSamples: z.number().int().min(1).max(1_000),
  minimumSamples: z.number().int().min(7).max(10_000),
}).strict();

export const evidenceChallengeSchema = z.discriminatedUnion("kind", [
  deploymentCanaryChallengeSchema,
  runtimePerformanceChallengeSchema,
]);

export type DeploymentCanaryChallenge = z.infer<typeof deploymentCanaryChallengeSchema>;
export type RuntimePerformanceChallenge = z.infer<typeof runtimePerformanceChallengeSchema>;
export type EvidenceChallenge = z.infer<typeof evidenceChallengeSchema>;

export const evidenceResponseBindingSchema = z.object({
  challengeId: identifierSchema,
  nonce: nonceSchema,
  sessionId: identifierSchema,
}).strict();
