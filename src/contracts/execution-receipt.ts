import { createPrivateKey, createPublicKey, sign, verify, type KeyLike, type KeyObject } from "node:crypto";
import { z } from "zod";
import { canonicalEvidenceJson, sha256CanonicalEvidence } from "../core/json.js";

export const EXECUTION_RECEIPT_SCHEMA = "mycellios-execution-receipt/1" as const;
const sha256 = z.string().regex(/^sha256:[a-f0-9]{64}$/);
const identifier = z.string().min(1).max(256).regex(/^[A-Za-z0-9][A-Za-z0-9._:/-]*$/);

export const executionReceiptBodySchema = z.object({
  schema: z.literal(EXECUTION_RECEIPT_SCHEMA),
  jobId: identifier,
  modelIdHash: sha256,
  routeClass: z.enum(["replica", "pipeline"]),
  metrics: z.object({
    inputTokens: z.number().int().nonnegative().safe(),
    outputTokens: z.number().int().nonnegative().safe(),
    ttftMs: z.number().nonnegative().finite(),
    activeMs: z.number().nonnegative().finite(),
  }).strict(),
  networkTraceDigest: sha256,
  recovery: z.object({
    mode: z.enum(["none", "prompt-replay", "deterministic-prefix-replay"]),
    attempts: z.number().int().positive().safe(),
    replayedTokenEvents: z.number().int().nonnegative().safe(),
  }).strict(),
  privacy: z.object({
    trust: z.enum(["default", "trusted-only"]),
    boundary: z.enum(["trusted-edges", "pinned-edges"]),
    pinnedIdentityHashes: z.array(sha256).max(64),
  }).strict(),
  completedAt: z.number().int().nonnegative().safe(),
}).strict();

export const executionReceiptSchema = executionReceiptBodySchema.extend({
  receiptId: sha256,
  keyId: identifier,
  signature: z.string().max(128).refine((value) => {
    const bytes = Buffer.from(value, "base64url");
    return bytes.length === 64 && bytes.toString("base64url") === value;
  }, "execution_receipt_signature_is_invalid"),
}).strict();

export type ExecutionReceiptBody = z.infer<typeof executionReceiptBodySchema>;
export type ExecutionReceipt = z.infer<typeof executionReceiptSchema>;

export function signExecutionReceipt(body: unknown, options: { keyId: string; privateKey: KeyLike }): ExecutionReceipt {
  const parsed = executionReceiptBodySchema.parse(body);
  const keyId = identifier.parse(options.keyId);
  const privateKey = isKeyObject(options.privateKey) ? options.privateKey : createPrivateKey(options.privateKey);
  if (privateKey.type !== "private" || privateKey.asymmetricKeyType !== "ed25519") throw new Error("execution_receipt_private_key_is_not_ed25519");
  const receiptId = sha256CanonicalEvidence(parsed);
  const document = { ...parsed, receiptId, keyId };
  return executionReceiptSchema.parse({ ...document,
    signature: sign(null, Buffer.from(canonicalEvidenceJson(document)), privateKey).toString("base64url") });
}

export function verifyExecutionReceipt(value: unknown, pinnedKey: { keyId: string; spki: string }): ExecutionReceipt {
  const receipt = executionReceiptSchema.parse(value);
  const { signature, receiptId, keyId, ...body } = receipt;
  if (sha256CanonicalEvidence(body) !== receiptId) throw new Error("execution_receipt_identity_mismatch");
  if (keyId !== pinnedKey.keyId) throw new Error("execution_receipt_key_id_mismatch");
  const bytes = Buffer.from(pinnedKey.spki, "base64url");
  if (bytes.toString("base64url") !== pinnedKey.spki) throw new Error("execution_receipt_public_key_is_invalid");
  let publicKey: KeyObject;
  try { publicKey = createPublicKey({ key: bytes, format: "der", type: "spki" }); }
  catch { throw new Error("execution_receipt_public_key_is_invalid"); }
  if (publicKey.asymmetricKeyType !== "ed25519") throw new Error("execution_receipt_public_key_is_not_ed25519");
  const document = { ...body, receiptId, keyId };
  if (!verify(null, Buffer.from(canonicalEvidenceJson(document)), publicKey, Buffer.from(signature, "base64url"))) {
    throw new Error("execution_receipt_signature_verification_failed");
  }
  return receipt;
}

function isKeyObject(value: KeyLike): value is KeyObject {
  return typeof value === "object" && value !== null && "asymmetricKeyType" in value;
}
