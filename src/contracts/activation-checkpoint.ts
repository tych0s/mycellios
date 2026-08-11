import { createHash, sign, verify, type KeyLike } from "node:crypto";
import { z } from "zod";
import { canonicalEvidenceJson, sha256CanonicalEvidence } from "../core/json.js";

export const ACTIVATION_CHECKPOINT_SCHEMA = "mycellios-activation-checkpoint/1" as const;
export const MAX_ACTIVATION_CHECKPOINT_BYTES = 512 * 1024 * 1024;

const digestSchema = z.string().regex(/^sha256:[0-9a-f]{64}$/);
const identifierSchema = z.string().min(1).max(256).regex(/^[A-Za-z0-9][A-Za-z0-9._:-]*$/);
const signatureSchema = z.string().regex(/^[A-Za-z0-9_-]{86}$/);

const unsignedShape = {
  schema: z.literal(ACTIVATION_CHECKPOINT_SCHEMA),
  requestIdHash: digestSchema,
  stageId: identifierSchema,
  nodeIdHash: digestSchema,
  topologyGeneration: z.number().int().nonnegative().safe(),
  topologyDigest: digestSchema,
  engineDescriptorDigest: digestSchema,
  artifactManifestDigest: digestSchema,
  configurationDigest: digestSchema,
  layerStart: z.number().int().nonnegative().safe(),
  layerEnd: z.number().int().positive().safe(),
  committedPosition: z.number().int().nonnegative().safe(),
  payloadDigest: digestSchema,
  bytes: z.number().int().positive().max(MAX_ACTIVATION_CHECKPOINT_BYTES),
  createdAt: z.number().int().nonnegative().safe(),
  expiresAt: z.number().int().positive().safe(),
  keyId: identifierSchema,
} as const;

export const unsignedActivationCheckpointSchema = z.object(unsignedShape).strict()
  .superRefine((value, context) => {
    if (value.layerEnd <= value.layerStart) {
      context.addIssue({ code: "custom", message: "activation_checkpoint_layer_range_is_invalid" });
    }
    if (value.expiresAt <= value.createdAt) {
      context.addIssue({ code: "custom", message: "activation_checkpoint_expiry_is_invalid" });
    }
  });

export const activationCheckpointSchema = z.object({
  ...unsignedShape,
  checkpointId: digestSchema,
  signature: signatureSchema,
}).strict().superRefine((value, context) => {
  if (value.layerEnd <= value.layerStart) {
    context.addIssue({ code: "custom", message: "activation_checkpoint_layer_range_is_invalid" });
  }
  if (value.expiresAt <= value.createdAt) {
    context.addIssue({ code: "custom", message: "activation_checkpoint_expiry_is_invalid" });
  }
});

export type UnsignedActivationCheckpoint = z.infer<typeof unsignedActivationCheckpointSchema>;
export type ActivationCheckpoint = z.infer<typeof activationCheckpointSchema>;

export interface ActivationCheckpointCompatibility {
  keyId: string;
  requestIdHash: string;
  nodeIdHash: string;
  topologyGeneration: number;
  topologyDigest: string;
  engineDescriptorDigest: string;
  artifactManifestDigest: string;
  configurationDigest: string;
  stageId: string;
  layerStart: number;
  layerEnd: number;
  minimumCommittedPosition?: number;
}

export function signActivationCheckpoint(
  input: UnsignedActivationCheckpoint,
  privateKey: KeyLike,
): ActivationCheckpoint {
  const unsigned = unsignedActivationCheckpointSchema.parse(input);
  const checkpointId = sha256CanonicalEvidence(unsigned);
  const signature = sign(
    null,
    Buffer.from(canonicalEvidenceJson({ ...unsigned, checkpointId })),
    privateKey,
  ).toString("base64url");
  return activationCheckpointSchema.parse({ ...unsigned, checkpointId, signature });
}

export function signActivationCheckpointWith(
  input: UnsignedActivationCheckpoint,
  signer: (payload: string) => string,
): ActivationCheckpoint {
  const unsigned = unsignedActivationCheckpointSchema.parse(input);
  const checkpointId = sha256CanonicalEvidence(unsigned);
  const signature = signer(canonicalEvidenceJson({ ...unsigned, checkpointId }));
  return activationCheckpointSchema.parse({ ...unsigned, checkpointId, signature });
}

export function verifyActivationCheckpoint(
  value: unknown,
  payload: Uint8Array,
  pinnedKeys: ReadonlyMap<string, KeyLike>,
  expected: ActivationCheckpointCompatibility,
  now = Date.now(),
): ActivationCheckpoint {
  const checkpoint = activationCheckpointSchema.parse(value);
  const { checkpointId, signature, ...unsigned } = checkpoint;
  if (sha256CanonicalEvidence(unsigned) !== checkpointId) {
    throw new Error("activation_checkpoint_identity_is_invalid");
  }
  const publicKey = pinnedKeys.get(checkpoint.keyId);
  if (!publicKey || !verify(
    null,
    Buffer.from(canonicalEvidenceJson({ ...unsigned, checkpointId })),
    publicKey,
    Buffer.from(signature, "base64url"),
  )) throw new Error("activation_checkpoint_signature_is_invalid");
  if (payload.byteLength !== checkpoint.bytes || payload.byteLength > MAX_ACTIVATION_CHECKPOINT_BYTES) {
    throw new Error("activation_checkpoint_size_is_invalid");
  }
  const payloadDigest = `sha256:${createHash("sha256").update(payload).digest("hex")}`;
  if (payloadDigest !== checkpoint.payloadDigest) throw new Error("activation_checkpoint_payload_is_corrupt");
  if (checkpoint.expiresAt <= now) throw new Error("activation_checkpoint_is_expired");
  for (const key of [
    "keyId", "requestIdHash", "nodeIdHash", "topologyGeneration", "topologyDigest", "engineDescriptorDigest",
    "artifactManifestDigest", "configurationDigest", "stageId", "layerStart", "layerEnd",
  ] as const) {
    if (checkpoint[key] !== expected[key]) {
      throw new Error(`activation_checkpoint_${key}_is_incompatible`);
    }
  }
  if (
    expected.minimumCommittedPosition !== undefined
    && checkpoint.committedPosition < expected.minimumCommittedPosition
  ) throw new Error("activation_checkpoint_position_is_incompatible");
  return checkpoint;
}
