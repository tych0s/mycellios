import {
  createPrivateKey,
  createPublicKey,
  sign as signBytes,
  verify as verifyBytes,
  type KeyLike,
  type KeyObject,
} from "node:crypto";
import { z } from "zod";

import { canonicalEvidenceJson, sha256CanonicalEvidence } from "../core/json.js";

export const MODEL_CERTIFICATION_SCHEMA = "mycellios-model-certification/1" as const;

const sha256Schema = z.string().regex(/^sha256:[0-9a-f]{64}$/);
const revisionSchema = z.string().regex(/^[0-9a-f]{40}$/);
const identifierSchema = z.string().min(1).max(256).regex(/^[A-Za-z0-9][A-Za-z0-9._+:/-]*$/);
const protocolRangeSchema = z.object({
  min: z.number().int().positive().max(65_535),
  max: z.number().int().positive().max(65_535),
}).strict().refine(({ min, max }) => min <= max, "model_certification_protocol_range_is_invalid");

const unsignedCertificationSchema = z.object({
  schema: z.literal(MODEL_CERTIFICATION_SCHEMA),
  decision: z.enum(["certified", "revoked"]),
  modelFamily: identifierSchema,
  distributionManifestId: sha256Schema,
  adapterContractId: sha256Schema,
  componentManifestId: sha256Schema,
  topology: z.object({
    kind: z.enum(["local-complete", "remote-replica", "distributed-pipeline"]),
    stageCount: z.number().int().positive().max(1_024),
    tensorParallelDegree: z.number().int().positive().max(1_024),
  }).strict().superRefine((topology, context) => {
    if (topology.kind !== "distributed-pipeline" && topology.stageCount !== 1) {
      context.addIssue({ code: "custom", message: "model_certification_topology_stage_count_is_invalid" });
    }
  }),
  hardware: z.object({
    platform: z.enum(["win32", "linux", "darwin"]),
    arch: z.enum(["x64", "arm64"]),
    backend: z.enum(["cpu", "cuda", "rocm", "metal", "vulkan"]),
    deviceFamily: identifierSchema,
    minimumMemoryBytes: z.number().int().positive().safe(),
    driverFingerprint: sha256Schema,
  }).strict(),
  context: z.object({
    maximumTokens: z.number().int().positive().max(10_000_000),
    codecs: z.array(identifierSchema).min(1).max(32).refine((values) => new Set(values).size === values.length, "model_certification_codecs_are_duplicated"),
    workerProtocol: protocolRangeSchema,
    tensorAbi: identifierSchema,
  }).strict(),
  evidence: z.object({
    class: z.literal("physical"),
    receiptId: sha256Schema,
    sourceRevision: revisionSchema,
    measuredAt: z.string().datetime({ offset: true }),
  }).strict(),
  review: z.object({
    reviewerId: identifierSchema,
    reviewedAt: z.string().datetime({ offset: true }),
  }).strict(),
  expiresAt: z.string().datetime({ offset: true }),
}).strict().superRefine((value, context) => {
  if (Date.parse(value.expiresAt) <= Date.parse(value.review.reviewedAt)) {
    context.addIssue({ code: "custom", message: "model_certification_expiry_is_invalid", path: ["expiresAt"] });
  }
});

export const modelCertificationSchema = unsignedCertificationSchema.extend({
  certificationId: sha256Schema,
  keyId: identifierSchema,
  signature: z.string().max(128).refine((value) => {
    const bytes = Buffer.from(value, "base64url");
    return bytes.length === 64 && bytes.toString("base64url") === value;
  }, "model_certification_signature_is_invalid"),
}).strict();

export type UnsignedModelCertification = z.infer<typeof unsignedCertificationSchema>;
export type ModelCertification = z.infer<typeof modelCertificationSchema>;
export type ModelCertificationBuildInput = Omit<UnsignedModelCertification, "schema">;

export function buildModelCertification(input: ModelCertificationBuildInput): UnsignedModelCertification {
  return unsignedCertificationSchema.parse({ schema: MODEL_CERTIFICATION_SCHEMA, ...input });
}

export function signModelCertification(
  unsigned: UnsignedModelCertification,
  options: { keyId: string; privateKey: KeyLike },
): ModelCertification {
  const parsed = unsignedCertificationSchema.parse(unsigned);
  const keyId = identifierSchema.parse(options.keyId);
  const privateKey = isKeyObject(options.privateKey)
    ? options.privateKey
    : createPrivateKey(options.privateKey);
  if (privateKey.asymmetricKeyType !== "ed25519") throw new Error("model_certification_private_key_is_not_ed25519");
  const certificationId = sha256CanonicalEvidence(parsed);
  const document = { ...parsed, certificationId, keyId };
  return modelCertificationSchema.parse({
    ...document,
    signature: signBytes(null, Buffer.from(canonicalEvidenceJson(document)), privateKey).toString("base64url"),
  });
}

export function verifyModelCertification(
  value: unknown,
  options: {
    pinnedKey: { keyId: string; spki: string };
    expectedDistributionManifestId?: string;
    now?: Date;
  },
): ModelCertification {
  const certification = modelCertificationSchema.parse(value);
  const { signature, certificationId, keyId, ...unsigned } = certification;
  if (sha256CanonicalEvidence(unsigned) !== certificationId) throw new Error("model_certification_identity_mismatch");
  if (keyId !== options.pinnedKey.keyId) throw new Error("model_certification_key_id_mismatch");
  if (options.expectedDistributionManifestId !== undefined && certification.distributionManifestId !== options.expectedDistributionManifestId) {
    throw new Error("model_certification_distribution_mismatch");
  }
  if (Date.parse(certification.expiresAt) <= (options.now ?? new Date()).getTime()) throw new Error("model_certification_expired");
  const publicKey = createPublicKey({ key: Buffer.from(options.pinnedKey.spki, "base64url"), format: "der", type: "spki" });
  if (publicKey.asymmetricKeyType !== "ed25519") throw new Error("model_certification_public_key_is_not_ed25519");
  const document = { ...unsigned, certificationId, keyId };
  if (!verifyBytes(null, Buffer.from(canonicalEvidenceJson(document)), publicKey, Buffer.from(signature, "base64url"))) {
    throw new Error("model_certification_signature_verification_failed");
  }
  return certification;
}

export function assertProductiveModelCertification(certification: ModelCertification): void {
  if (certification.decision !== "certified") throw new Error("model_certification_is_not_productive");
}

function isKeyObject(value: KeyLike): value is KeyObject {
  return typeof value === "object" && value !== null && "asymmetricKeyType" in value;
}
