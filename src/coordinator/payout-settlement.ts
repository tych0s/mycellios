import { createPublicKey, verify } from "node:crypto";

export interface PayoutSettlementAttestation {
  schema: "mycellios.payout-settlement.v1";
  batchId: string;
  externalReference: string;
  settlementReference: string;
  paidAt: number;
  issuedAt: number;
  verifierKeyId: string;
  signature: string;
}

export class PayoutSettlementError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "PayoutSettlementError";
  }
}

export class PayoutSettlementVerifier {
  constructor(
    private readonly trustedKeys: ReadonlyMap<string, string>,
    private readonly maxAgeMs = 15 * 60_000,
    private readonly now: () => number = Date.now,
  ) {
    if (!Number.isSafeInteger(maxAgeMs) || maxAgeMs <= 0) throw new Error("invalid_settlement_max_age");
  }

  verify(attestation: PayoutSettlementAttestation): PayoutSettlementAttestation {
    const now = this.now();
    validateAttestation(attestation, now, this.maxAgeMs);
    const publicKey = this.trustedKeys.get(attestation.verifierKeyId);
    if (!publicKey) {
      throw new PayoutSettlementError("untrusted_settlement_verifier", "The settlement verifier is not trusted.");
    }
    let valid = false;
    try {
      valid = verify(
        null,
        payoutSettlementSigningBytes(attestation),
        createPublicKey(publicKey),
        Buffer.from(attestation.signature, "base64url"),
      );
    } catch {
      valid = false;
    }
    if (!valid) {
      throw new PayoutSettlementError("invalid_settlement_signature", "The settlement signature is invalid.");
    }
    return { ...attestation };
  }
}

export function payoutSettlementSigningBytes(
  attestation: Omit<PayoutSettlementAttestation, "signature">,
): Buffer {
  return Buffer.from(JSON.stringify([
    attestation.schema,
    attestation.batchId,
    attestation.externalReference,
    attestation.settlementReference,
    attestation.paidAt,
    attestation.issuedAt,
    attestation.verifierKeyId,
  ]), "utf8");
}

function validateAttestation(
  input: PayoutSettlementAttestation,
  now: number,
  maxAgeMs: number,
): void {
  if (input.schema !== "mycellios.payout-settlement.v1") {
    throw new PayoutSettlementError("unsupported_settlement_schema", "The settlement schema is unsupported.");
  }
  for (const value of [input.batchId, input.externalReference, input.settlementReference, input.verifierKeyId]) {
    if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$/.test(value)) {
      throw new PayoutSettlementError("invalid_settlement_identity", "The settlement identity is invalid.");
    }
  }
  if (!Number.isSafeInteger(input.paidAt) || input.paidAt <= 0
    || !Number.isSafeInteger(input.issuedAt)
    || input.issuedAt > now + 60_000
    || input.issuedAt < now - maxAgeMs
    || input.paidAt > input.issuedAt + 60_000) {
    throw new PayoutSettlementError("invalid_settlement_time", "The settlement timestamps are invalid or stale.");
  }
  if (!/^[A-Za-z0-9_-]{16,1024}$/.test(input.signature)) {
    throw new PayoutSettlementError("invalid_settlement_signature", "The settlement signature encoding is invalid.");
  }
}
