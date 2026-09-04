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
import { recoveryEventSchema } from "./recovery-outcome.js";

export const EXECUTION_RECEIPT_SCHEMA =
  "mycellios-execution-receipt/1" as const;
export const STAGE_EXECUTION_OBSERVATION_SCHEMA =
  "mycellios-stage-execution-observation/1" as const;
export const EXECUTION_RECEIPT_ENVELOPE_SCHEMA =
  "mycellios-execution-receipt-envelope/1" as const;
export const MAX_EXECUTION_RECEIPT_ENVELOPE_BYTES = 1024 * 1024;

const digestSchema = z.string().regex(/^sha256:[0-9a-f]{64}$/);
const gitShaSchema = z.string().regex(/^[0-9a-f]{40}$/);
const identifierSchema = z
  .string()
  .min(1)
  .max(256)
  .regex(/^[A-Za-z0-9][A-Za-z0-9._:-]*$/);
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
    "execution_receipt_signature_is_invalid",
  );
const wireDigestSchema = z.string().regex(/^[0-9a-f]{32}$/);

const stageCountersSchema = z
  .object({
    frames: z.number().int().nonnegative().safe(),
    inputBytes: z.number().int().nonnegative().safe(),
    outputBytes: z.number().int().nonnegative().safe(),
    computeMs: z.number().nonnegative().finite(),
  })
  .strict();

const unsignedStageObservationShape = {
  schema: z.literal(STAGE_EXECUTION_OBSERVATION_SCHEMA),
  executionId: identifierSchema,
  requestIdHash: digestSchema,
  stageId: identifierSchema,
  nodeIdHash: digestSchema,
  keyId: keyIdSchema,
  topologyGeneration: z.number().int().nonnegative().safe(),
  layerStart: z.number().int().nonnegative().safe(),
  layerEnd: z.number().int().positive().safe(),
  engineDescriptorDigest: digestSchema,
  artifactManifestDigest: digestSchema,
  inputRoot: digestSchema,
  outputRoot: digestSchema,
  counters: stageCountersSchema,
  outcome: z.enum(["completed", "recovered", "failed"]),
} as const;

export const unsignedStageExecutionObservationSchema = z
  .object(unsignedStageObservationShape)
  .strict()
  .refine(
    ({ layerStart, layerEnd }) => layerEnd > layerStart,
    "execution_stage_layer_range_is_invalid",
  );

export const stageExecutionObservationSchema = z
  .object({
    ...unsignedStageObservationShape,
    signature: signatureSchema,
  })
  .strict()
  .refine(
    ({ layerStart, layerEnd }) => layerEnd > layerStart,
    "execution_stage_layer_range_is_invalid",
  );

const executionMetricsSchema = z
  .object({
    ttftMs: z.number().nonnegative().finite(),
    tpotMs: z.number().nonnegative().finite(),
    outputTokens: z.number().int().nonnegative().safe(),
    acceptedDraftTokens: z.number().int().nonnegative().safe(),
    proposedDraftTokens: z.number().int().nonnegative().safe(),
  })
  .strict()
  .refine(
    ({ acceptedDraftTokens, proposedDraftTokens }) =>
      acceptedDraftTokens <= proposedDraftTokens,
    "execution_receipt_acceptance_is_invalid",
  );

const unsignedExecutionReceiptShape = {
  schema: z.literal(EXECUTION_RECEIPT_SCHEMA),
  sourceSha: gitShaSchema,
  sourceId: digestSchema,
  createdAt: z.string().datetime({ offset: true }),
  evidenceClass: z.enum([
    "simulation",
    "loopback",
    "lan",
    "wan-direct",
    "wan-relay",
  ]),
  executionId: identifierSchema,
  requestIdHash: digestSchema,
  modelDigest: digestSchema,
  engineDescriptorDigest: digestSchema,
  artifactManifestDigest: digestSchema,
  topologyDigest: digestSchema,
  topologyGeneration: z.number().int().nonnegative().safe(),
  transport: z.enum(["local", "direct", "relay", "mixed"]),
  totalLayers: z.number().int().positive().safe(),
  requestRoot: digestSchema,
  resultRoot: digestSchema,
  outputTokenHash: digestSchema,
  stages: z.array(stageExecutionObservationSchema).min(1).max(256),
  recovery: z.array(recoveryEventSchema).max(256),
  metrics: executionMetricsSchema,
  redactionVersion: z.literal(1),
} as const;

export const unsignedExecutionReceiptSchema = z
  .object(unsignedExecutionReceiptShape)
  .strict();

export const executionReceiptSchema = z
  .object({
    ...unsignedExecutionReceiptShape,
    receiptId: digestSchema,
  })
  .strict();

const unsignedExecutionReceiptEnvelopeShape = {
  schema: z.literal(EXECUTION_RECEIPT_ENVELOPE_SCHEMA),
  deploymentGeneration: z.number().int().nonnegative().safe(),
  routeDigest: wireDigestSchema,
  waveStrategyDigest: wireDigestSchema,
  waveArtifactDigest: wireDigestSchema,
  keyId: keyIdSchema,
  receipt: executionReceiptSchema,
} as const;

export const unsignedExecutionReceiptEnvelopeSchema = z
  .object(unsignedExecutionReceiptEnvelopeShape)
  .strict();

export const executionReceiptEnvelopeSchema = z
  .object({
    ...unsignedExecutionReceiptEnvelopeShape,
    signature: signatureSchema,
  })
  .strict();

export type UnsignedStageExecutionObservation = z.infer<
  typeof unsignedStageExecutionObservationSchema
>;
export type StageExecutionObservation = z.infer<
  typeof stageExecutionObservationSchema
>;
export type UnsignedExecutionReceipt = z.infer<
  typeof unsignedExecutionReceiptSchema
>;
export type ExecutionReceipt = z.infer<typeof executionReceiptSchema>;
export type UnsignedExecutionReceiptEnvelope = z.infer<
  typeof unsignedExecutionReceiptEnvelopeSchema
>;
export type ExecutionReceiptEnvelope = z.infer<
  typeof executionReceiptEnvelopeSchema
>;

export interface StageObservationSigningOptions {
  keyId: string;
  privateKey: KeyLike;
}

export interface ExecutionReceiptVerificationOptions {
  pinnedStageKeys: ReadonlyMap<string, KeyLike>;
}

export interface ExecutionReceiptEnvelopeSigningOptions {
  keyId: string;
  privateKey: KeyLike;
  deploymentGeneration: number;
  routeDigest: string;
  waveStrategyDigest: string;
  waveArtifactDigest: string;
}

export interface ExecutionReceiptEnvelopeVerificationOptions
  extends ExecutionReceiptVerificationOptions {
  pinnedEnvelopeKeys: ReadonlyMap<string, KeyLike>;
}

export function signStageExecutionObservation(
  input: Omit<UnsignedStageExecutionObservation, "keyId">,
  options: StageObservationSigningOptions,
): StageExecutionObservation {
  const unsigned = unsignedStageExecutionObservationSchema.parse({
    ...input,
    keyId: options.keyId,
  });
  const signature = signBytes(
    null,
    Buffer.from(canonicalEvidenceJson(unsigned)),
    options.privateKey,
  ).toString("base64url");
  return stageExecutionObservationSchema.parse({ ...unsigned, signature });
}

export function buildExecutionReceipt(
  input: UnsignedExecutionReceipt,
): ExecutionReceipt {
  const unsigned = unsignedExecutionReceiptSchema.parse(input);
  validateExecutionReceiptChain(unsigned);
  return executionReceiptSchema.parse({
    ...unsigned,
    receiptId: sha256CanonicalEvidence(unsigned),
  });
}

export function verifyExecutionReceipt(
  value: unknown,
  options: ExecutionReceiptVerificationOptions,
): ExecutionReceipt {
  const receipt = executionReceiptSchema.parse(value);
  validateExecutionReceiptIdentityAndChain(receipt);
  for (const observation of receipt.stages) {
    const publicKey = options.pinnedStageKeys.get(observation.keyId);
    if (!publicKey) {
      throw new Error(`execution_receipt_stage_key_is_unknown:${observation.keyId}`);
    }
    const { signature, ...signed } = observation;
    if (
      !verifyBytes(
        null,
        Buffer.from(canonicalEvidenceJson(signed)),
        publicKey,
        Buffer.from(signature, "base64url"),
      )
    ) {
      throw new Error(`execution_receipt_stage_signature_is_invalid:${observation.stageId}`);
    }
  }
  return receipt;
}

export function signExecutionReceiptEnvelope(
  receiptValue: unknown,
  options: ExecutionReceiptEnvelopeSigningOptions,
): ExecutionReceiptEnvelope {
  const receipt = executionReceiptSchema.parse(receiptValue);
  validateExecutionReceiptIdentityAndChain(receipt);
  const unsigned = unsignedExecutionReceiptEnvelopeSchema.parse({
    schema: EXECUTION_RECEIPT_ENVELOPE_SCHEMA,
    deploymentGeneration: options.deploymentGeneration,
    routeDigest: options.routeDigest,
    waveStrategyDigest: options.waveStrategyDigest,
    waveArtifactDigest: options.waveArtifactDigest,
    keyId: options.keyId,
    receipt,
  });
  const signature = signBytes(
    null,
    Buffer.from(canonicalEvidenceJson(unsigned)),
    options.privateKey,
  ).toString("base64url");
  return executionReceiptEnvelopeSchema.parse({ ...unsigned, signature });
}

export function verifyExecutionReceiptEnvelope(
  value: unknown,
  options: ExecutionReceiptEnvelopeVerificationOptions,
): ExecutionReceiptEnvelope {
  const envelope = executionReceiptEnvelopeSchema.parse(value);
  const publicKey = options.pinnedEnvelopeKeys.get(envelope.keyId);
  if (!publicKey) {
    throw new Error(`execution_receipt_envelope_key_is_unknown:${envelope.keyId}`);
  }
  const { signature, ...signed } = envelope;
  if (
    !verifyBytes(
      null,
      Buffer.from(canonicalEvidenceJson(signed)),
      publicKey,
      Buffer.from(signature, "base64url"),
    )
  ) {
    throw new Error("execution_receipt_envelope_signature_is_invalid");
  }
  verifyExecutionReceipt(envelope.receipt, options);
  return envelope;
}

export function encodeExecutionReceiptEnvelope(
  value: unknown,
): Buffer {
  const envelope = executionReceiptEnvelopeSchema.parse(value);
  const encoded = Buffer.from(canonicalEvidenceJson(envelope));
  if (encoded.byteLength > MAX_EXECUTION_RECEIPT_ENVELOPE_BYTES) {
    throw new Error("execution_receipt_envelope_exceeds_wire_limit");
  }
  return encoded;
}

export function verifyExecutionReceiptEnvelopeBytes(
  value: Uint8Array,
  options: ExecutionReceiptEnvelopeVerificationOptions,
): ExecutionReceiptEnvelope {
  if (!(value instanceof Uint8Array) || value.byteLength < 1) {
    throw new Error("execution_receipt_envelope_wire_payload_is_empty");
  }
  if (value.byteLength > MAX_EXECUTION_RECEIPT_ENVELOPE_BYTES) {
    throw new Error("execution_receipt_envelope_exceeds_wire_limit");
  }
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(value);
  } catch {
    throw new Error("execution_receipt_envelope_wire_utf8_is_invalid");
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("execution_receipt_envelope_wire_json_is_invalid");
  }
  const envelope = executionReceiptEnvelopeSchema.parse(parsed);
  if (text !== canonicalEvidenceJson(envelope)) {
    throw new Error("execution_receipt_envelope_wire_json_is_not_canonical");
  }
  return verifyExecutionReceiptEnvelope(envelope, options);
}

function validateExecutionReceiptChain(
  receipt: UnsignedExecutionReceipt,
): void {
  let expectedLayer = 0;
  let expectedRoot = receipt.requestRoot;
  const stageIds = new Set<string>();
  for (const observation of receipt.stages) {
    // Settlement is fail-closed: a terminal receipt proves completed work, not
    // merely a valid signature over an assigned range. This invariant is
    // adapted from leyten/shard's Apache-2.0 receipt coverage verifier; see
    // THIRD_PARTY.md for the pinned source revision.
    if (observation.outcome === "failed") {
      throw new Error("execution_receipt_contains_failed_stage");
    }
    if (
      observation.counters.frames < 1
      || observation.counters.inputBytes < 1
      || observation.counters.outputBytes < 1
    ) {
      throw new Error("execution_receipt_stage_attests_zero_work");
    }
    if (stageIds.has(observation.stageId)) {
      throw new Error("execution_receipt_stage_is_duplicated");
    }
    stageIds.add(observation.stageId);
    if (
      observation.executionId !== receipt.executionId
      || observation.requestIdHash !== receipt.requestIdHash
      || observation.topologyGeneration !== receipt.topologyGeneration
      || observation.engineDescriptorDigest !== receipt.engineDescriptorDigest
      || observation.artifactManifestDigest !== receipt.artifactManifestDigest
    ) {
      throw new Error("execution_receipt_stage_identity_mismatch");
    }
    if (observation.layerStart !== expectedLayer) {
      throw new Error("execution_receipt_layer_coverage_is_invalid");
    }
    if (observation.inputRoot !== expectedRoot) {
      throw new Error("execution_receipt_stage_chain_is_invalid");
    }
    expectedLayer = observation.layerEnd;
    expectedRoot = observation.outputRoot;
  }
  if (expectedLayer !== receipt.totalLayers) {
    throw new Error("execution_receipt_layer_coverage_is_invalid");
  }
  if (expectedRoot !== receipt.resultRoot) {
    throw new Error("execution_receipt_result_root_is_invalid");
  }
}

function validateExecutionReceiptIdentityAndChain(
  receipt: ExecutionReceipt,
): void {
  const { receiptId: _receiptId, ...unsigned } = receipt;
  if (sha256CanonicalEvidence(unsigned) !== receipt.receiptId) {
    throw new Error("execution_receipt_identity_mismatch");
  }
  validateExecutionReceiptChain(unsigned);
}
