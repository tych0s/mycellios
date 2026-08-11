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

export const ECONOMIC_SETTLEMENT_RECEIPT_SCHEMA =
  "mycellios-economic-settlement-receipt/1" as const;

const sha256Schema = z.string().regex(/^sha256:[0-9a-f]{64}$/);
const identifierSchema = z.string().min(1).max(256).regex(/^[A-Za-z0-9][A-Za-z0-9._:/-]*$/);
const safeAmountSchema = z.number().int().nonnegative().safe();
const signedBodySchema = z.object({
  schema: z.literal(ECONOMIC_SETTLEMENT_RECEIPT_SCHEMA),
  settlementId: sha256Schema,
  jobId: identifierSchema,
  executionReceiptId: sha256Schema,
  contributionEvidenceId: sha256Schema,
  pricingPolicyId: sha256Schema,
  executionRecovery: z.object({
    mode: z.enum(["none", "prompt-replay", "deterministic-prefix-replay"]),
    attempts: z.number().int().positive().safe(),
    replayedTokenEvents: z.number().int().nonnegative().safe(),
  }).strict().optional(),
  asset: z.literal("MYC_MICROCREDITS"),
  payerAccountHash: sha256Schema,
  grossMicrounits: safeAmountSchema,
  providerMicrounits: safeAmountSchema,
  platformMicrounits: safeAmountSchema,
  contributorCredits: z.array(z.object({
    nodeIdHash: sha256Schema,
    amountMicrounits: safeAmountSchema,
  }).strict()).min(1).max(1_024),
  createdAt: z.number().int().nonnegative().safe(),
}).strict().superRefine((receipt, context) => {
  if (receipt.providerMicrounits + receipt.platformMicrounits !== receipt.grossMicrounits) {
    context.addIssue({ code: "custom", message: "economic_receipt_amounts_are_unbalanced" });
  }
  if (receipt.contributorCredits.reduce((sum, entry) => sum + entry.amountMicrounits, 0) !== receipt.providerMicrounits) {
    context.addIssue({ code: "custom", message: "economic_receipt_contributors_are_unbalanced" });
  }
  const hashes = receipt.contributorCredits.map(({ nodeIdHash }) => nodeIdHash);
  if (new Set(hashes).size !== hashes.length || hashes.some((hash, index) => index > 0 && hashes[index - 1]! > hash)) {
    context.addIssue({ code: "custom", message: "economic_receipt_contributors_are_not_canonical" });
  }
});

export const economicSettlementReceiptSchema = signedBodySchema.extend({
  receiptId: sha256Schema,
  keyId: identifierSchema,
  signature: z.string().max(128).refine((value) => {
    const bytes = Buffer.from(value, "base64url");
    return bytes.length === 64 && bytes.toString("base64url") === value;
  }, "economic_receipt_signature_is_invalid"),
}).strict();

export type EconomicSettlementReceiptBody = z.infer<typeof signedBodySchema>;
export type EconomicSettlementReceipt = z.infer<typeof economicSettlementReceiptSchema>;

export function signEconomicSettlementReceipt(
  body: EconomicSettlementReceiptBody,
  options: { keyId: string; privateKey: KeyLike },
): EconomicSettlementReceipt {
  const parsed = signedBodySchema.parse(body);
  const keyId = identifierSchema.parse(options.keyId);
  const privateKey = isKeyObject(options.privateKey) ? options.privateKey : createPrivateKey(options.privateKey);
  if (privateKey.type !== "private" || privateKey.asymmetricKeyType !== "ed25519") {
    throw new Error("economic_receipt_private_key_is_not_ed25519");
  }
  const receiptId = sha256CanonicalEvidence(parsed);
  const document = { ...parsed, receiptId, keyId };
  return economicSettlementReceiptSchema.parse({
    ...document,
    signature: signBytes(null, Buffer.from(canonicalEvidenceJson(document)), privateKey).toString("base64url"),
  });
}

export function verifyEconomicSettlementReceipt(
  value: unknown,
  pinnedKey: { keyId: string; spki: string },
): EconomicSettlementReceipt {
  const receipt = economicSettlementReceiptSchema.parse(value);
  const { signature, receiptId, keyId, ...body } = receipt;
  if (sha256CanonicalEvidence(body) !== receiptId) throw new Error("economic_receipt_identity_mismatch");
  if (keyId !== pinnedKey.keyId) throw new Error("economic_receipt_key_id_mismatch");
  let publicKey: KeyObject;
  try {
    publicKey = createPublicKey({ key: Buffer.from(pinnedKey.spki, "base64url"), format: "der", type: "spki" });
  } catch {
    throw new Error("economic_receipt_public_key_is_invalid");
  }
  if (publicKey.asymmetricKeyType !== "ed25519") throw new Error("economic_receipt_public_key_is_not_ed25519");
  const document = { ...body, receiptId, keyId };
  if (!verifyBytes(null, Buffer.from(canonicalEvidenceJson(document)), publicKey, Buffer.from(signature, "base64url"))) {
    throw new Error("economic_receipt_signature_verification_failed");
  }
  return receipt;
}

function isKeyObject(value: KeyLike): value is KeyObject {
  return typeof value === "object" && value !== null && "asymmetricKeyType" in value;
}
