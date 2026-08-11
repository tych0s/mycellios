import { z } from "zod";
import { createHash } from "node:crypto";
import {
  activationCheckpointSchema,
  type ActivationCheckpoint,
} from "./activation-checkpoint.js";

export const ACTIVATION_CHECKPOINT_CHUNK_BYTES = 256 * 1024;
export const MAX_ACTIVATION_CHECKPOINT_CHUNKS = 2_048;

export const activationCheckpointTransferIdSchema = z.string().uuid();
export const activationCheckpointIdSchema = z.string().regex(/^sha256:[0-9a-f]{64}$/);

export const activationCheckpointCompatibilitySchema = z.object({
  keyId: z.string().min(1).max(256).regex(/^[A-Za-z0-9][A-Za-z0-9._:-]*$/),
  requestIdHash: activationCheckpointIdSchema,
  nodeIdHash: activationCheckpointIdSchema,
  topologyGeneration: z.number().int().nonnegative().safe(),
  topologyDigest: activationCheckpointIdSchema,
  engineDescriptorDigest: activationCheckpointIdSchema,
  artifactManifestDigest: activationCheckpointIdSchema,
  configurationDigest: activationCheckpointIdSchema,
  stageId: z.string().min(1).max(256),
  layerStart: z.number().int().nonnegative().safe(),
  layerEnd: z.number().int().positive().safe(),
  minimumCommittedPosition: z.number().int().nonnegative().safe().optional(),
}).strict().refine((value) => value.layerEnd > value.layerStart, {
  message: "activation_checkpoint_layer_range_is_invalid",
});

export const activationCheckpointBeginSchema = z.object({
  transferId: activationCheckpointTransferIdSchema,
  checkpoint: activationCheckpointSchema,
  chunkCount: z.number().int().min(1).max(MAX_ACTIVATION_CHECKPOINT_CHUNKS),
}).strict();

export const activationCheckpointChunkSchema = z.object({
  transferId: activationCheckpointTransferIdSchema,
  checkpointId: activationCheckpointIdSchema,
  index: z.number().int().nonnegative().max(MAX_ACTIVATION_CHECKPOINT_CHUNKS - 1),
  data: z.string().min(1).max(Math.ceil(ACTIVATION_CHECKPOINT_CHUNK_BYTES * 4 / 3) + 4)
    .regex(/^[A-Za-z0-9+/]+={0,2}$/)
    .refine(
      (value) => Buffer.from(value, "base64").byteLength <= ACTIVATION_CHECKPOINT_CHUNK_BYTES,
      "activation_checkpoint_chunk_is_too_large",
    ),
}).strict();

export const activationCheckpointCommitSchema = z.object({
  transferId: activationCheckpointTransferIdSchema,
  checkpointId: activationCheckpointIdSchema,
}).strict();

export const activationCheckpointRequestSchema = z.object({
  transferId: activationCheckpointTransferIdSchema,
  stageRequestId: z.number().int().nonnegative().safe(),
  expected: activationCheckpointCompatibilitySchema,
  maximumBytes: z.number().int().positive().max(512 * 1024 * 1024),
  expiresAt: z.number().int().positive().safe(),
}).strict().superRefine((value, context) => {
  if (value.expected.requestIdHash !== activationCheckpointRequestIdHash(value.stageRequestId)) {
    context.addIssue({ code: "custom", message: "activation_checkpoint_request_hash_is_invalid" });
  }
});

export const activationCheckpointCommittedSchema = activationCheckpointCommitSchema;

export const activationCheckpointRestoreBeginSchema = z.object({
  transferId: activationCheckpointTransferIdSchema,
  targetLaunchRequestId: z.string().min(1).max(256),
  targetStageRequestId: z.number().int().nonnegative().safe(),
  expected: activationCheckpointCompatibilitySchema,
  checkpoint: activationCheckpointSchema,
  chunkCount: z.number().int().min(1).max(MAX_ACTIVATION_CHECKPOINT_CHUNKS),
  maximumBytes: z.number().int().positive().max(512 * 1024 * 1024),
  expiresAt: z.number().int().positive().safe(),
}).strict().superRefine((value, context) => {
  const { checkpoint, expected } = value;
  if (expected.requestIdHash !== activationCheckpointRequestIdHash(value.targetStageRequestId)) {
    context.addIssue({ code: "custom", message: "activation_checkpoint_restore_request_hash_is_invalid" });
  }
  if (
    checkpoint.keyId !== expected.keyId
    || checkpoint.requestIdHash !== expected.requestIdHash
    || checkpoint.nodeIdHash !== expected.nodeIdHash
    || checkpoint.topologyGeneration !== expected.topologyGeneration
    || checkpoint.topologyDigest !== expected.topologyDigest
    || checkpoint.engineDescriptorDigest !== expected.engineDescriptorDigest
    || checkpoint.artifactManifestDigest !== expected.artifactManifestDigest
    || checkpoint.configurationDigest !== expected.configurationDigest
    || checkpoint.stageId !== expected.stageId
    || checkpoint.layerStart !== expected.layerStart
    || checkpoint.layerEnd !== expected.layerEnd
    || checkpoint.bytes > value.maximumBytes
    || value.chunkCount !== Math.ceil(checkpoint.bytes / ACTIVATION_CHECKPOINT_CHUNK_BYTES)
    || (
      expected.minimumCommittedPosition !== undefined
      && checkpoint.committedPosition < expected.minimumCommittedPosition
    )
  ) {
    context.addIssue({ code: "custom", message: "activation_checkpoint_restore_compatibility_is_invalid" });
  }
});

export const activationCheckpointRestoredSchema = activationCheckpointCommitSchema;

export const activationCheckpointRestoreFailedSchema = z.object({
  transferId: activationCheckpointTransferIdSchema,
  checkpointId: activationCheckpointIdSchema,
  code: z.enum([
    "checkpoint_restore_expired",
    "checkpoint_restore_incomplete",
    "checkpoint_restore_incompatible",
    "checkpoint_restore_process_unavailable",
    "checkpoint_restore_failed",
  ]),
}).strict();

export const activationCheckpointFailedSchema = z.object({
  transferId: activationCheckpointTransferIdSchema,
  code: z.enum([
    "checkpoint_provider_unavailable",
    "checkpoint_capture_failed",
    "checkpoint_incompatible",
    "checkpoint_too_large",
    "checkpoint_request_expired",
    "checkpoint_delivery_failed",
  ]),
}).strict();

export function activationCheckpointChunks(
  transferId: string,
  checkpoint: ActivationCheckpoint,
  payload: Uint8Array,
): Array<z.infer<typeof activationCheckpointChunkSchema>> {
  activationCheckpointTransferIdSchema.parse(transferId);
  if (payload.byteLength !== checkpoint.bytes) {
    throw new Error("activation_checkpoint_transfer_payload_size_is_invalid");
  }
  const chunks = [];
  for (
    let offset = 0, index = 0;
    offset < payload.byteLength;
    offset += ACTIVATION_CHECKPOINT_CHUNK_BYTES, index += 1
  ) {
    chunks.push(activationCheckpointChunkSchema.parse({
      transferId,
      checkpointId: checkpoint.checkpointId,
      index,
      data: Buffer.from(
        payload.subarray(offset, offset + ACTIVATION_CHECKPOINT_CHUNK_BYTES),
      ).toString("base64"),
    }));
  }
  return chunks;
}

export function activationCheckpointRequestIdHash(requestId: number): string {
  if (!Number.isSafeInteger(requestId) || requestId < 0) {
    throw new Error("activation_checkpoint_stage_request_id_is_invalid");
  }
  return `sha256:${createHash("sha256")
    .update("mycellios-stage-request/1\0")
    .update(String(requestId))
    .digest("hex")}`;
}
