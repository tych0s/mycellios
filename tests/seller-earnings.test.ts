import { generateKeyPairSync, sign } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  SellerEarningsError,
  SellerEarningsManager,
  earningReversalSigningBytes,
  type SignedEarningReversal,
  type SignedWorkReceipt,
  workReceiptSigningBytes,
} from "../src/coordinator/seller-earnings.js";
import { PayoutManager } from "../src/coordinator/payouts.js";
import { MeshDatabase } from "../src/storage/database.js";

describe("receipt-driven seller earnings", () => {
  let database: MeshDatabase;
  let manager: SellerEarningsManager;
  let privateKey: ReturnType<typeof generateKeyPairSync>["privateKey"];

  beforeEach(() => {
    database = new MeshDatabase(":memory:");
    const pair = generateKeyPairSync("ed25519");
    privateKey = pair.privateKey;
    manager = new SellerEarningsManager(database, new Map([
      ["coordinator-2026-01", pair.publicKey.export({ type: "spki", format: "pem" }).toString()],
    ]));
  });

  afterEach(() => database.close());

  function receipt(overrides: Partial<SignedWorkReceipt> = {}): SignedWorkReceipt {
    const payload = {
      schema: "mycellios.work-receipt.v1" as const,
      receiptId: "receipt-1",
      sellerId: "seller-1",
      workerId: "worker-1",
      jobId: "job-1",
      stageId: "stage-1",
      pricingVersion: "pricing-1",
      amountUsdMicros: 125_000,
      evidenceDigest: "a".repeat(64),
      accepted: true,
      acceptedAt: 1_786_000_000_000,
      verifierKeyId: "coordinator-2026-01",
      ...overrides,
    };
    return {
      ...payload,
      signature: sign(null, workReceiptSigningBytes(payload), privateKey).toString("base64url"),
    };
  }

  function reversal(
    earningId: string,
    overrides: Partial<SignedEarningReversal> = {},
  ): SignedEarningReversal {
    const payload = {
      schema: "mycellios.earning-reversal.v1" as const,
      reversalId: `reversal-${earningId}`,
      earningId,
      sellerId: "seller-1",
      reason: "verification_error" as const,
      evidenceDigest: "b".repeat(64),
      reversedAt: 1_786_100_000_000,
      verifierKeyId: "coordinator-2026-01",
      ...overrides,
    };
    return {
      ...payload,
      signature: sign(null, earningReversalSigningBytes(payload), privateKey).toString("base64url"),
    };
  }

  it("creates one stable earning from one valid accepted receipt", () => {
    const signed = receipt();
    const first = manager.recordAcceptedReceipt(signed);
    const replay = manager.recordAcceptedReceipt(signed);

    expect(first).toMatchObject({
      sellerId: "seller-1",
      amountUsdMicros: 125_000,
      payoutMethod: "stable",
      status: "available",
    });
    expect(replay).toEqual(first);
    expect(manager.listAvailable("seller-1")).toEqual([first]);
  });

  it("seals the verified receipt and derived earning economic snapshot", () => {
    const earning = manager.recordAcceptedReceipt(receipt());

    expect(() => database.raw.prepare(
      "UPDATE verified_work_receipts SET amount_usd_micros = 999999 WHERE receipt_id = ?",
    ).run(earning.receiptId)).toThrow(/immutable_verified_work_receipt/);
    expect(() => database.raw.prepare(
      "UPDATE seller_earnings SET payout_method = 'spore' WHERE id = ?",
    ).run(earning.id)).toThrow(/immutable_seller_earning_economic_snapshot/);
    expect(() => database.raw.prepare(
      "UPDATE seller_earnings SET amount_usd_micros = 999999 WHERE id = ?",
    ).run(earning.id)).toThrow(/immutable_seller_earning_economic_snapshot/);
    expect(() => database.raw.prepare(
      "DELETE FROM seller_earnings WHERE id = ?",
    ).run(earning.id)).toThrow(/immutable_seller_earning/);
    expect(() => database.raw.prepare(
      "DELETE FROM verified_work_receipts WHERE receipt_id = ?",
    ).run(earning.receiptId)).toThrow(/immutable_verified_work_receipt/);

    const now = Date.now();
    database.raw.prepare(
      `INSERT INTO verified_work_receipts(
         receipt_id, seller_id, worker_id, job_id, stage_id, pricing_version,
         amount_usd_micros, evidence_digest, accepted_at, verifier_key_id, signature, recorded_at
       ) VALUES ('receipt-binding', 'seller-1', 'worker-binding', 'job-binding',
                 'stage-binding', 'pricing-1', 125000, ?, ?, 'verifier-1', 'signed-fixture', ?)`,
    ).run("c".repeat(64), now, now);
    expect(() => database.raw.prepare(
      `INSERT INTO seller_earnings(
         id, seller_id, receipt_id, amount_usd_micros, payout_method, status, created_at
       ) VALUES ('earning-binding', 'seller-1', 'receipt-binding', 999999, 'stable', 'available', ?)`,
    ).run(now)).toThrow(/invalid_seller_earning_receipt_binding/);
  });

  it("snapshots an explicit SPORE payout preference without changing stable value", () => {
    manager.setPayoutPreference("seller-1", "spore");
    const earning = manager.recordAcceptedReceipt(receipt());
    expect(earning).toMatchObject({ payoutMethod: "spore", amountUsdMicros: 125_000 });
  });

  it("fails closed for tampering, rejected work and untrusted verifiers", () => {
    const signed = receipt();
    expect(() => manager.recordAcceptedReceipt({ ...signed, amountUsdMicros: 999_000 })).toThrowError(
      expect.objectContaining({ code: "invalid_receipt_signature" } satisfies Partial<SellerEarningsError>),
    );
    expect(() => manager.recordAcceptedReceipt(receipt({ receiptId: "receipt-rejected", accepted: false }))).toThrowError(
      expect.objectContaining({ code: "work_not_accepted" } satisfies Partial<SellerEarningsError>),
    );
    expect(() => manager.recordAcceptedReceipt(receipt({ receiptId: "receipt-untrusted", verifierKeyId: "unknown-key" }))).toThrowError(
      expect.objectContaining({ code: "untrusted_receipt_verifier" } satisfies Partial<SellerEarningsError>),
    );
    expect(database.raw.prepare("SELECT COUNT(*) AS count FROM seller_earnings").get()).toEqual({ count: 0 });
  });

  it("applies a signed reversal once while an earning is still available", () => {
    const earning = manager.recordAcceptedReceipt(receipt());
    const signed = reversal(earning.id);
    const first = manager.recordEarningReversal(signed);
    const replay = manager.recordEarningReversal(signed);

    expect(first).toMatchObject({ state: "applied", amountUsdMicros: 125_000, duplicate: false });
    expect(replay).toMatchObject({ id: first.id, state: "applied", duplicate: true });
    expect(manager.listAvailable("seller-1")).toEqual([]);
    expect(manager.getSellerDebt("seller-1")).toBe(0);
  });

  it("creates seller debt when already-paid work is later revoked", () => {
    const earning = manager.recordAcceptedReceipt(receipt());
    const payouts = new PayoutManager(database, {
      minimumUsdMicros: 100_000,
      spore: {
        legalApproved: false,
        custodyApproved: false,
        liquidityApproved: false,
        antifraudApproved: false,
      },
    });
    const batch = payouts.prepare({
      sellerId: "seller-1",
      payoutMethod: "stable",
      earningIds: [earning.id],
      idempotencyKey: "paid-reversal-fixture",
    }).batch;
    payouts.markSubmitted({
      batchId: batch.id,
      dispatchKey: "paid-reversal-dispatch",
      externalReference: "paid-reversal-transfer",
    });
    payouts.markPaid({
      batchId: batch.id,
      settlementReference: "paid-reversal-settlement",
      paidAt: Date.now(),
    });

    const result = manager.recordEarningReversal(reversal(earning.id));
    expect(result.state).toBe("seller_debt");
    expect(manager.getSellerDebt("seller-1")).toBe(125_000);
    expect(database.raw.prepare("SELECT status FROM seller_earnings WHERE id = ?").get(earning.id)).toEqual({
      status: "reversed",
    });
  });

  it("holds a submitted payout reversal until settlement then creates debt", () => {
    const earning = manager.recordAcceptedReceipt(receipt());
    const payouts = new PayoutManager(database, {
      minimumUsdMicros: 100_000,
      spore: {
        legalApproved: false,
        custodyApproved: false,
        liquidityApproved: false,
        antifraudApproved: false,
      },
    });
    const batch = payouts.prepare({
      sellerId: "seller-1",
      payoutMethod: "stable",
      earningIds: [earning.id],
      idempotencyKey: "pending-reversal-batch",
    }).batch;
    payouts.markSubmitted({
      batchId: batch.id,
      dispatchKey: "pending-reversal-dispatch",
      externalReference: "pending-reversal-transfer",
    });

    const pending = manager.recordEarningReversal(reversal(earning.id));
    expect(pending.state).toBe("pending_payout");
    expect(manager.getSellerDebt("seller-1")).toBe(0);
    expect(() => database.raw.prepare(
      "UPDATE seller_earning_reversals SET amount_usd_micros = 999999 WHERE id = ?",
    ).run(pending.id)).toThrow(/immutable_seller_earning_reversal_content/);
    expect(() => database.raw.prepare(
      "UPDATE seller_earning_reversals SET state = 'applied', resolved_at = ? WHERE id = ?",
    ).run(Date.now(), pending.id)).toThrow(/invalid_seller_earning_reversal_lifecycle/);
    expect(() => database.raw.prepare(
      "DELETE FROM seller_earning_reversals WHERE id = ?",
    ).run(pending.id)).toThrow(/immutable_seller_earning_reversal/);

    payouts.markPaid({
      batchId: batch.id,
      settlementReference: "pending-reversal-settlement",
      paidAt: 1_786_200_000_000,
    });
    expect(manager.getSellerDebt("seller-1")).toBe(125_000);
    expect(database.raw.prepare(
      "SELECT state, resolved_at FROM seller_earning_reversals WHERE reversal_id = ?",
    ).get(`reversal-${earning.id}`)).toEqual({ state: "seller_debt", resolved_at: expect.any(Number) });
    expect(() => database.raw.prepare(
      "UPDATE seller_earning_reversals SET state = 'pending_payout', resolved_at = NULL WHERE id = ?",
    ).run(pending.id)).toThrow(/invalid_seller_earning_reversal_lifecycle/);
  });

  it("rejects tampered reversals and requires prepared payouts to be cancelled first", () => {
    const earning = manager.recordAcceptedReceipt(receipt());
    const signed = reversal(earning.id);
    expect(() => manager.recordEarningReversal({ ...signed, reason: "fraud" })).toThrowError(
      expect.objectContaining({ code: "invalid_reversal_signature" }),
    );

    const payouts = new PayoutManager(database, {
      minimumUsdMicros: 100_000,
      spore: {
        legalApproved: false,
        custodyApproved: false,
        liquidityApproved: false,
        antifraudApproved: false,
      },
    });
    payouts.prepare({
      sellerId: "seller-1",
      payoutMethod: "stable",
      earningIds: [earning.id],
      idempotencyKey: "prepared-reversal-batch",
    });
    expect(() => manager.recordEarningReversal(signed)).toThrowError(
      expect.objectContaining({ code: "earning_in_prepared_payout" }),
    );
  });
});
