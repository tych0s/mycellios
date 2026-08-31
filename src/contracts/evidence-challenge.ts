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
const digestSchema = z.string().regex(/^sha256:[0-9a-f]{64}$/);
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

const engineRuntimeChallengeRequestFields = {
  probeKind: z.literal("qwen3-dense-v1"),
  descriptorDigest: digestSchema,
  certificationId: digestSchema,
  artifactManifestDigest: digestSchema,
  modelId: z.string().trim().min(1).max(512),
  modelRevision: z.string().regex(/^[0-9a-f]{40}$/),
  backend: z.enum(["cpu", "cuda", "rocm", "directml", "mps", "vulkan", "webgpu"]),
  runtimeAbi: identifierSchema,
  quantization: z.string().trim().min(1).max(64),
  contextTokens: z.number().int().positive().max(16_777_216),
  expectedLayerStart: z.number().int().nonnegative().max(1_000_000),
  expectedLayerEnd: z.number().int().positive().max(1_000_000),
  expectedKvBytesPerToken: z.number().int().positive().safe(),
  expectedLayerWeightBytes: z.number().int().positive().safe(),
  referenceDecodeMsPerToken: z.number().positive().finite().max(1_000_000_000),
  referencePrefillMsPerToken: z.number().positive().finite().max(1_000_000_000),
  hiddenSize: z.number().int().positive().max(1_000_000),
  attentionHeads: z.number().int().positive().max(100_000),
  kvHeads: z.number().int().positive().max(100_000),
  headDim: z.number().int().positive().max(100_000),
  requiredRoles: z.array(z.enum(["head", "middle", "tail", "draft", "auxiliary"]))
    .min(1)
    .max(5),
  minimumSamples: z.number().int().min(7).max(100_000),
} as const;

function validateEngineRuntimeChallengeRequest(
  challenge: z.infer<z.ZodObject<typeof engineRuntimeChallengeRequestFields>>,
  context: z.RefinementCtx,
): void {
  if (challenge.expectedLayerEnd <= challenge.expectedLayerStart) {
    context.addIssue({
      code: "custom",
      message: "engine_runtime_challenge_layer_range_is_invalid",
      path: ["expectedLayerEnd"],
    });
  }
  const layerCount = challenge.expectedLayerEnd - challenge.expectedLayerStart;
  if (
    !["cuda", "rocm"].includes(challenge.backend)
    ||
    challenge.hiddenSize !== challenge.attentionHeads * challenge.headDim
    || challenge.attentionHeads % challenge.kvHeads !== 0
    || challenge.expectedKvBytesPerToken
      !== 2 * challenge.kvHeads * challenge.headDim * 2 * layerCount
  ) {
    context.addIssue({
      code: "custom",
      message: "engine_runtime_challenge_qwen3_shape_is_invalid",
      path: ["hiddenSize"],
    });
  }
  if (new Set(challenge.requiredRoles).size !== challenge.requiredRoles.length) {
    context.addIssue({
      code: "custom",
      message: "engine_runtime_challenge_role_is_duplicated",
      path: ["requiredRoles"],
    });
  }
}

export const engineRuntimeChallengeRequestSchema = z.object(
  engineRuntimeChallengeRequestFields,
).strict().superRefine(validateEngineRuntimeChallengeRequest);

export const engineRuntimeChallengeSchema = z.object({
  ...challengeBase,
  kind: z.literal("engine-runtime"),
  nodeId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
  ...engineRuntimeChallengeRequestFields,
}).strict().superRefine(validateEngineRuntimeChallengeRequest);

export const evidenceChallengeSchema = z.discriminatedUnion("kind", [
  deploymentCanaryChallengeSchema,
  runtimePerformanceChallengeSchema,
  engineRuntimeChallengeSchema,
]);

export type DeploymentCanaryChallenge = z.infer<typeof deploymentCanaryChallengeSchema>;
export type RuntimePerformanceChallenge = z.infer<typeof runtimePerformanceChallengeSchema>;
export type EngineRuntimeChallengeRequest = z.infer<typeof engineRuntimeChallengeRequestSchema>;
export type EngineRuntimeChallenge = z.infer<typeof engineRuntimeChallengeSchema>;
export type EvidenceChallenge = z.infer<typeof evidenceChallengeSchema>;

export const evidenceResponseBindingSchema = z.object({
  challengeId: identifierSchema,
  nonce: nonceSchema,
  sessionId: identifierSchema,
}).strict();
