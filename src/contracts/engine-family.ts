import {
  sign as signBytes,
  verify as verifyBytes,
  type KeyLike,
} from "node:crypto";
import { z } from "zod";
import {
  canonicalEvidenceJson,
  sha256CanonicalEvidence,
} from "../core/json.js";
import type { ActivationCheckpoint as StageRuntimeCheckpoint } from "./activation-checkpoint.js";
export type { ActivationCheckpoint as StageRuntimeCheckpoint } from "./activation-checkpoint.js";

export const ENGINE_FAMILY_DESCRIPTOR_SCHEMA =
  "mycellios-engine-family/1" as const;
export const ENGINE_CERTIFICATION_SCHEMA =
  "mycellios-engine-certification/1" as const;
export const DRAFT_STRATEGY_DESCRIPTOR_SCHEMA =
  "mycellios-draft-strategy/1" as const;
export const DRAFT_STRATEGY_CERTIFICATION_SCHEMA =
  "mycellios-draft-strategy-certification/1" as const;

const digestSchema = z.string().regex(/^sha256:[0-9a-f]{64}$/);
const revisionSchema = z.string().regex(/^[0-9a-f]{40}$/);
const identifierSchema = z
  .string()
  .min(1)
  .max(128)
  .regex(/^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$/);
const versionSchema = z
  .string()
  .min(1)
  .max(128)
  .regex(/^[A-Za-z0-9](?:[A-Za-z0-9._+-]*[A-Za-z0-9])?$/);
const backendSchema = z.enum([
  "cpu",
  "cuda",
  "rocm",
  "directml",
  "mps",
  "xpu",
  "vulkan",
  "webgpu",
]);
const platformSchema = z.enum(["linux", "win32", "darwin"]);
const architectureSchema = z.enum(["x64", "arm64"]);
const quantizationSchema = z
  .string()
  .min(1)
  .max(64)
  .regex(/^[A-Za-z0-9][A-Za-z0-9._+-]*$/);
const keyIdSchema = z
  .string()
  .min(1)
  .max(128)
  .regex(/^[A-Za-z0-9][A-Za-z0-9._-]*$/);
const signatureSchema = z
  .string()
  .regex(/^[A-Za-z0-9_-]{86}$/)
  .refine(
    (value) => Buffer.from(value, "base64url").byteLength === 64,
    "draft_strategy_certification_signature_is_invalid",
  );

const contextRangeSchema = z
  .object({
    minTokens: z.number().int().positive().max(16_777_216),
    maxTokens: z.number().int().positive().max(16_777_216),
  })
  .strict()
  .refine(
    ({ minTokens, maxTokens }) => minTokens <= maxTokens,
    "engine_context_range_is_invalid",
  );

const engineTargetSchema = z
  .object({
    platform: platformSchema,
    arch: architectureSchema,
    backend: backendSchema,
    runtimeAbi: identifierSchema,
    quantizations: z.array(quantizationSchema).min(1).max(64),
    context: contextRangeSchema,
    graphMode: z.enum(["required", "optional", "unsupported"]),
    minDriver: versionSchema.nullable(),
  })
  .strict()
  .superRefine((target, context) => {
    const seen = new Set<string>();
    for (const [index, quantization] of target.quantizations.entries()) {
      if (seen.has(quantization)) {
        context.addIssue({
          code: "custom",
          message: "engine_target_quantization_is_duplicated",
          path: ["quantizations", index],
        });
      }
      seen.add(quantization);
    }
  });

const boundaryConstraintSchema = z
  .object({
    role: z.enum(["head", "tail", "draft", "auxiliary", "privacy"]),
    trust: z.enum(["any", "account", "private-network", "operator"]),
    requiredBackends: z.array(backendSchema).max(8),
    requiredFailureDomain: z.enum(["any", "distinct-host", "distinct-network"]),
  })
  .strict();

const tensorOwnershipSchema = z
  .object({
    embeddings: z.enum(["root", "head"]),
    outputHead: z.enum(["root", "tail"]),
    tokenizer: z.literal("root"),
    sharedTensorIds: z.array(identifierSchema).max(128),
  })
  .strict();

const requiredOperations = [
  "prepare",
  "load",
  "prefill",
  "verify",
  "decode",
  "reset",
  "checkpoint",
  "rollback",
  "health",
  "receipt-observe",
  "unload",
] as const;

const operationSchema = z.enum(requiredOperations);

export const engineFamilyDescriptorSchema = z
  .object({
    schema: z.literal(ENGINE_FAMILY_DESCRIPTOR_SCHEMA),
    familyId: identifierSchema,
    version: versionSchema,
    implementationDigest: digestSchema,
    tensorAbi: identifierSchema,
    model: z
      .object({
        modelId: z.string().min(1).max(512),
        revision: revisionSchema,
        manifestDigest: digestSchema,
        nativeAdapterId: identifierSchema,
        implementationContract: identifierSchema,
      })
      .strict(),
    ownership: tensorOwnershipSchema,
    operations: z.array(operationSchema).min(requiredOperations.length).max(64),
    targets: z.array(engineTargetSchema).min(1).max(128),
    boundaryConstraints: z.array(boundaryConstraintSchema).max(32),
  })
  .strict()
  .superRefine((descriptor, context) => {
    const operations = new Set(descriptor.operations);
    for (const operation of requiredOperations) {
      if (!operations.has(operation)) {
        context.addIssue({
          code: "custom",
          message: `engine_required_operation_is_missing:${operation}`,
          path: ["operations"],
        });
      }
    }
    if (operations.size !== descriptor.operations.length) {
      context.addIssue({
        code: "custom",
        message: "engine_operation_is_duplicated",
        path: ["operations"],
      });
    }

    const targets = new Set<string>();
    for (const [index, target] of descriptor.targets.entries()) {
      const identity = [
        target.platform,
        target.arch,
        target.backend,
        target.runtimeAbi,
      ].join(":");
      if (targets.has(identity)) {
        context.addIssue({
          code: "custom",
          message: "engine_target_is_duplicated",
          path: ["targets", index],
        });
      }
      targets.add(identity);
    }
  });

const parityEvidenceBaseShape = {
  evidenceDigest: digestSchema,
  evidenceClass: z.enum(["fixture", "loopback", "lan", "wan"]),
} as const;

export const parityEvidenceSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.enum(["logits", "tokens"]),
    ...parityEvidenceBaseShape,
  }).strict(),
  z.object({
    kind: z.literal("sampling-distribution"),
    ...parityEvidenceBaseShape,
    samplerSchema: identifierSchema,
    seedCount: z.number().int().min(32).max(1_000_000),
    sampleCount: z.number().int().min(4_096).max(1_000_000_000),
    maximumTotalVariationDistance: z.number().min(0).max(0.01),
    minimumGoodnessOfFitPValue: z.number().min(0.01).max(1),
  }).strict(),
]);

export const engineCertificationSchema = z
  .object({
    schema: z.literal(ENGINE_CERTIFICATION_SCHEMA),
    certificationId: digestSchema,
    descriptorDigest: digestSchema,
    sourceId: digestSchema,
    artifactManifestDigest: digestSchema,
    status: z.enum(["candidate", "certified", "revoked"]),
    validFrom: z.string().datetime({ offset: true }),
    expiresAt: z.string().datetime({ offset: true }),
    targets: z.array(engineTargetSchema).min(1).max(128),
    hardwareClasses: z.array(identifierSchema).min(1).max(128),
    parity: z.array(parityEvidenceSchema).min(1).max(128),
  })
  .strict()
  .superRefine((certification, context) => {
    if (
      Date.parse(certification.validFrom) >= Date.parse(certification.expiresAt)
    ) {
      context.addIssue({
        code: "custom",
        message: "engine_certification_window_is_invalid",
        path: ["expiresAt"],
      });
    }
  });

export const draftStrategyDescriptorSchema = z
  .object({
    schema: z.literal(DRAFT_STRATEGY_DESCRIPTOR_SCHEMA),
    strategyId: identifierSchema,
    version: versionSchema,
    kind: z.enum(["ngram", "sibling-model", "mtp", "model-specific"]),
    componentDigest: digestSchema,
    targetDescriptorDigest: digestSchema,
    tokenizerDigest: digestSchema,
    vocabularyDigest: digestSchema,
    resource: z
      .object({
        minimumRamBytes: z.number().int().nonnegative().safe(),
        minimumVramBytes: z.number().int().nonnegative().safe(),
        allowedBackends: z.array(backendSchema).min(1).max(8),
      })
      .strict(),
    limits: z
      .object({
        minDraftTokens: z.number().int().positive().max(4_096),
        maxDraftTokens: z.number().int().positive().max(4_096),
        maxInflightWaves: z.number().int().positive().max(256),
      })
      .strict(),
    evidenceDigest: digestSchema,
  })
  .strict()
  .refine(
    ({ limits }) => limits.minDraftTokens <= limits.maxDraftTokens,
    "draft_strategy_token_range_is_invalid",
  );

const draftStrategyCertificationContentShape = {
  schema: z.literal(DRAFT_STRATEGY_CERTIFICATION_SCHEMA),
  descriptorDigest: digestSchema,
  sourceId: digestSchema,
  status: z.enum(["candidate", "certified", "revoked"]),
  validFrom: z.string().datetime({ offset: true }),
  expiresAt: z.string().datetime({ offset: true }),
  publisherKeyId: keyIdSchema,
} as const;

export const draftStrategyCertificationContentSchema = z
  .object(draftStrategyCertificationContentShape)
  .strict()
  .refine(
    ({ validFrom, expiresAt }) => Date.parse(validFrom) < Date.parse(expiresAt),
    "draft_strategy_certification_window_is_invalid",
  );

export const draftStrategyCertificationSchema = z
  .object({
    ...draftStrategyCertificationContentShape,
    certificationId: digestSchema,
    signature: signatureSchema,
  })
  .strict()
  .refine(
    ({ validFrom, expiresAt }) => Date.parse(validFrom) < Date.parse(expiresAt),
    "draft_strategy_certification_window_is_invalid",
  );

export type EngineFamilyDescriptor = z.infer<
  typeof engineFamilyDescriptorSchema
>;
export type EngineTarget = z.infer<typeof engineTargetSchema>;
export type EngineCertification = z.infer<typeof engineCertificationSchema>;
export type DraftStrategyDescriptor = z.infer<
  typeof draftStrategyDescriptorSchema
>;
export type DraftStrategyCertificationContent = z.infer<
  typeof draftStrategyCertificationContentSchema
>;
export type DraftStrategyCertification = z.infer<
  typeof draftStrategyCertificationSchema
>;

export function draftStrategyDescriptorIdentity(
  value: DraftStrategyDescriptor,
): `sha256:${string}` {
  return sha256CanonicalEvidence(
    draftStrategyDescriptorSchema.parse(value),
  ) as `sha256:${string}`;
}

export function signDraftStrategyCertification(
  value: DraftStrategyCertificationContent,
  privateKey: KeyLike,
): DraftStrategyCertification {
  const content = draftStrategyCertificationContentSchema.parse(value);
  const certificationId = sha256CanonicalEvidence(content);
  const signed = { ...content, certificationId };
  const signature = signBytes(
    null,
    Buffer.from(canonicalEvidenceJson(signed)),
    privateKey,
  ).toString("base64url");
  return draftStrategyCertificationSchema.parse({ ...signed, signature });
}

export function draftStrategyCertificationIdentity(
  value: DraftStrategyCertification,
): `sha256:${string}` {
  const certification = draftStrategyCertificationSchema.parse(value);
  const { signature: _signature, certificationId: _certificationId, ...content } =
    certification;
  return sha256CanonicalEvidence(content) as `sha256:${string}`;
}

export function verifyDraftStrategyCertification(
  value: unknown,
  pinnedPublisherKeys: ReadonlyMap<string, KeyLike>,
): DraftStrategyCertification {
  const certification = draftStrategyCertificationSchema.parse(value);
  const { signature, certificationId, ...content } = certification;
  if (draftStrategyCertificationIdentity(certification) !== certificationId) {
    throw new Error("draft_strategy_certification_identity_mismatch");
  }
  const publicKey = pinnedPublisherKeys.get(certification.publisherKeyId);
  if (!publicKey) {
    throw new Error(
      `draft_strategy_certification_key_is_unknown:${certification.publisherKeyId}`,
    );
  }
  if (
    !verifyBytes(
      null,
      Buffer.from(canonicalEvidenceJson({ ...content, certificationId })),
      publicKey,
      Buffer.from(signature, "base64url"),
    )
  ) {
    throw new Error("draft_strategy_certification_signature_is_invalid");
  }
  return certification;
}

export function engineCertificationIdentity(
  value: EngineCertification,
): `sha256:${string}` {
  const certification = engineCertificationSchema.parse(value);
  const { certificationId: _certificationId, ...content } = certification;
  return sha256CanonicalEvidence(content) as `sha256:${string}`;
}

export function sealEngineCertification(
  value: Omit<EngineCertification, "certificationId">,
): EngineCertification {
  const placeholder = engineCertificationSchema.parse({
    ...value,
    certificationId: `sha256:${"0".repeat(64)}`,
  });
  return engineCertificationSchema.parse({
    ...value,
    certificationId: engineCertificationIdentity(placeholder),
  });
}

export interface StageRuntimeHealth {
  readonly ready: boolean;
  readonly loadedArtifactDigest: `sha256:${string}` | null;
  readonly topologyGeneration: number | null;
  readonly activeRequests: number;
}

/**
 * Runtime behavior is deliberately an interface rather than a serializable
 * schema. Only descriptors and operation inputs cross trust boundaries.
 */
export interface StageRuntime {
  prepare(): Promise<void>;
  load(): Promise<void>;
  prefill(requestId: string, startPosition: number): Promise<void>;
  verify(requestId: string, waveId: number): Promise<void>;
  decode(requestId: string): Promise<void>;
  reset(requestId: string): Promise<void>;
  checkpoint(requestId: string): Promise<StageRuntimeCheckpoint>;
  rollback(checkpoint: StageRuntimeCheckpoint): Promise<void>;
  health(): Promise<StageRuntimeHealth>;
  observeReceipt(requestId: string): Promise<unknown>;
  unload(): Promise<void>;
}
