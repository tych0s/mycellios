import { generateKeyPairSync } from "node:crypto";
import { describe, expect, it } from "vitest";

import {
  signEconomicSettlementReceipt,
  verifyEconomicSettlementReceipt,
} from "../src/contracts/economic-settlement-receipt.js";

const sha = (character: string) => `sha256:${character.repeat(64)}` as const;

describe("economic settlement receipt", () => {
  it("signs and verifies a redacted canonical balanced receipt", () => {
    const keys = generateKeyPairSync("ed25519");
    const receipt = signEconomicSettlementReceipt({
      schema: "mycellios-economic-settlement-receipt/1",
      settlementId: sha("a"), jobId: "job-1", executionReceiptId: sha("b"), contributionEvidenceId: sha("f"), pricingPolicyId: sha("c"),
      executionRecovery: { mode: "deterministic-prefix-replay", attempts: 2, replayedTokenEvents: 3 },
      asset: "MYC_MICROCREDITS", payerAccountHash: sha("d"), grossMicrounits: 100,
      providerMicrounits: 85, platformMicrounits: 15,
      contributorCredits: [{ nodeIdHash: sha("e"), amountMicrounits: 85 }], createdAt: 1,
    }, { keyId: "economic-key-1", privateKey: keys.privateKey });
    expect(verifyEconomicSettlementReceipt(receipt, {
      keyId: "economic-key-1",
      spki: keys.publicKey.export({ format: "der", type: "spki" }).toString("base64url"),
    })).toEqual(receipt);
    expect(JSON.stringify(receipt)).not.toContain("customer@example");
    expect(receipt).not.toHaveProperty("payerAccountId");
    expect(receipt.contributorCredits[0]).not.toHaveProperty("nodeId");
    expect(receipt.executionRecovery).toEqual({ mode: "deterministic-prefix-replay", attempts: 2, replayedTokenEvents: 3 });
  });

  it("rejects unbalanced, tampered and wrongly pinned receipts", () => {
    const keys = generateKeyPairSync("ed25519");
    const body = {
      schema: "mycellios-economic-settlement-receipt/1" as const,
      settlementId: sha("a"), jobId: "job-1", executionReceiptId: sha("b"), contributionEvidenceId: sha("f"), pricingPolicyId: sha("c"),
      executionRecovery: { mode: "prompt-replay" as const, attempts: 2, replayedTokenEvents: 0 },
      asset: "MYC_MICROCREDITS" as const, payerAccountHash: sha("d"), grossMicrounits: 100,
      providerMicrounits: 85, platformMicrounits: 15,
      contributorCredits: [{ nodeIdHash: sha("e"), amountMicrounits: 85 }], createdAt: 1,
    };
    expect(() => signEconomicSettlementReceipt({ ...body, platformMicrounits: 14 }, { keyId: "key", privateKey: keys.privateKey })).toThrow("economic_receipt_amounts_are_unbalanced");
    const receipt = signEconomicSettlementReceipt(body, { keyId: "key", privateKey: keys.privateKey });
    expect(() => verifyEconomicSettlementReceipt({ ...receipt, executionRecovery: { ...receipt.executionRecovery!, mode: "none" } }, {
      keyId: "key", spki: keys.publicKey.export({ format: "der", type: "spki" }).toString("base64url"),
    })).toThrow("economic_receipt_identity_mismatch");
    expect(() => verifyEconomicSettlementReceipt({ ...receipt, grossMicrounits: 101 }, {
      keyId: "key", spki: keys.publicKey.export({ format: "der", type: "spki" }).toString("base64url"),
    })).toThrow();
    expect(() => verifyEconomicSettlementReceipt(receipt, {
      keyId: "other", spki: keys.publicKey.export({ format: "der", type: "spki" }).toString("base64url"),
    })).toThrow("economic_receipt_key_id_mismatch");
  });
});
