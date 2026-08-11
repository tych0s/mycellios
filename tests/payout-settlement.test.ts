import { generateKeyPairSync, sign } from "node:crypto";
import { describe, expect, it } from "vitest";
import {
  PayoutSettlementError,
  PayoutSettlementVerifier,
  payoutSettlementSigningBytes,
  type PayoutSettlementAttestation,
} from "../src/coordinator/payout-settlement.js";

describe("signed payout settlement evidence", () => {
  const now = 1_786_000_000_000;
  const keyPair = generateKeyPairSync("ed25519");
  const verifier = new PayoutSettlementVerifier(new Map([[
    "settlement-verifier-1",
    keyPair.publicKey.export({ type: "spki", format: "pem" }).toString(),
  ]]), 15 * 60_000, () => now);

  function attestation(overrides: Partial<PayoutSettlementAttestation> = {}): PayoutSettlementAttestation {
    const unsigned = {
      schema: "mycellios.payout-settlement.v1" as const,
      batchId: "payout-batch-1",
      externalReference: "tr_transfer123",
      settlementReference: "po_payout123",
      paidAt: now - 120_000,
      issuedAt: now - 60_000,
      verifierKeyId: "settlement-verifier-1",
      ...overrides,
    };
    return {
      ...unsigned,
      signature: sign(null, payoutSettlementSigningBytes(unsigned), keyPair.privateKey).toString("base64url"),
      ...("signature" in overrides ? { signature: overrides.signature! } : {}),
    };
  }

  it("accepts fresh evidence signed by a trusted verifier", () => {
    expect(verifier.verify(attestation())).toMatchObject({
      batchId: "payout-batch-1",
      externalReference: "tr_transfer123",
      settlementReference: "po_payout123",
    });
  });

  it("rejects tampering, stale evidence and untrusted verifiers", () => {
    const tampered = attestation();
    tampered.externalReference = "tr_different123";
    expect(() => verifier.verify(tampered)).toThrowError(PayoutSettlementError);
    expect(() => verifier.verify(attestation({ issuedAt: now - 15 * 60_000 - 1 })))
      .toThrowError(expect.objectContaining({ code: "invalid_settlement_time" }));
    expect(() => verifier.verify(attestation({ verifierKeyId: "unknown-verifier" })))
      .toThrowError(expect.objectContaining({ code: "untrusted_settlement_verifier" }));
  });
});
