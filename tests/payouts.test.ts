import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { PayoutError, PayoutManager, type PayoutPolicy } from "../src/coordinator/payouts.js";
import { MeshDatabase } from "../src/storage/database.js";

const closedPolicy: PayoutPolicy = {
  minimumUsdMicros: 100_000,
  spore: {
    legalApproved: false,
    custodyApproved: false,
    liquidityApproved: false,
    antifraudApproved: false,
  },
};

describe("provider-neutral seller payout batches", () => {
  let database: MeshDatabase;

  beforeEach(() => {
    database = new MeshDatabase(":memory:");
  });

  afterEach(() => database.close());

  function earning(id: string, sellerId = "seller-1", method: "stable" | "spore" = "stable", amount = 125_000) {
    database.raw.prepare(
      `INSERT INTO verified_work_receipts(
         receipt_id, seller_id, worker_id, job_id, stage_id, pricing_version,
         amount_usd_micros, evidence_digest, accepted_at, verifier_key_id, signature, recorded_at
       ) VALUES (?, ?, ?, ?, ?, 'pricing-1', ?, ?, ?, 'verifier-1', ?, ?)`,
    ).run(
      `receipt-${id}`, sellerId, `worker-${id}`, `job-${id}`, `stage-${id}`,
      amount, "a".repeat(64), Date.now(), "signed-fixture", Date.now(),
    );
    database.raw.prepare(
      `INSERT INTO seller_earnings(
         id, seller_id, receipt_id, amount_usd_micros, payout_method, status, created_at
       ) VALUES (?, ?, ?, ?, ?, 'available', ?)`,
    ).run(id, sellerId, `receipt-${id}`, amount, method, Date.now());
  }

  it("prepares, submits and settles a stable payout exactly once", () => {
    earning("earning-1");
    earning("earning-2", "seller-1", "stable", 225_000);
    const payouts = new PayoutManager(database, closedPolicy);

    const prepared = payouts.prepare({
      sellerId: "seller-1",
      payoutMethod: "stable",
      earningIds: ["earning-2", "earning-1"],
      idempotencyKey: "seller-1:2026-08-10",
    });
    const replay = payouts.prepare({
      sellerId: "seller-1",
      payoutMethod: "stable",
      earningIds: ["earning-1", "earning-2"],
      idempotencyKey: "seller-1:2026-08-10",
    });
    expect(prepared).toMatchObject({ duplicate: false, batch: { status: "prepared", amountUsdMicros: 350_000 } });
    expect(replay).toMatchObject({ duplicate: true, batch: { id: prepared.batch.id } });

    const submitted = payouts.markSubmitted({
      batchId: prepared.batch.id,
      dispatchKey: "dispatch-1",
      externalReference: "usd-transfer-1",
    });
    expect(submitted.batch.status).toBe("submitted");
    expect(payouts.markSubmitted({
      batchId: prepared.batch.id,
      dispatchKey: "dispatch-1",
      externalReference: "usd-transfer-1",
    }).duplicate).toBe(true);

    const paid = payouts.markPaid({
      batchId: prepared.batch.id,
      settlementReference: "settlement-1",
      paidAt: 1_786_000_000_000,
    });
    expect(paid.batch.status).toBe("paid");
    expect(payouts.markPaid({
      batchId: prepared.batch.id,
      settlementReference: "settlement-1",
      paidAt: 1_786_000_000_000,
    }).duplicate).toBe(true);
    expect(database.raw.prepare(
      "SELECT COUNT(*) AS count FROM seller_earnings WHERE status = 'paid'",
    ).get()).toEqual({ count: 2 });
  });

  it("cancels only before submission and releases earnings", () => {
    earning("earning-cancel");
    const payouts = new PayoutManager(database, closedPolicy);
    const prepared = payouts.prepare({
      sellerId: "seller-1",
      payoutMethod: "stable",
      earningIds: ["earning-cancel"],
      idempotencyKey: "cancel-1",
    });
    expect(payouts.cancelPrepared(prepared.batch.id).batch.status).toBe("cancelled");
    expect(database.raw.prepare(
      "SELECT status FROM seller_earnings WHERE id = 'earning-cancel'",
    ).get()).toEqual({ status: "available" });

    const next = payouts.prepare({
      sellerId: "seller-1",
      payoutMethod: "stable",
      earningIds: ["earning-cancel"],
      idempotencyKey: "cancel-2",
    });
    payouts.markSubmitted({ batchId: next.batch.id, dispatchKey: "dispatch-2", externalReference: "transfer-2" });
    expect(() => payouts.cancelPrepared(next.batch.id)).toThrowError(
      expect.objectContaining({ code: "payout_cancellation_unsafe" } satisfies Partial<PayoutError>),
    );
  });

  it("preserves the payout batch audit timeline without ordering provider time", () => {
    earning("earning-batch-time");
    const payouts = new PayoutManager(database, closedPolicy);
    const prepared = payouts.prepare({
      sellerId: "seller-1",
      payoutMethod: "stable",
      earningIds: ["earning-batch-time"],
      idempotencyKey: "batch-time",
    });
    const initial = database.raw.prepare(
      "SELECT created_at, updated_at FROM payout_batches WHERE id = ?",
    ).get(prepared.batch.id) as { created_at: number; updated_at: number };
    expect(() => database.raw.prepare(
      "UPDATE payout_batches SET updated_at = ? WHERE id = ?",
    ).run(initial.created_at - 1, prepared.batch.id)).toThrow(/invalid_payout_batch_time/);

    const submitted = payouts.markSubmitted({
      batchId: prepared.batch.id,
      dispatchKey: "dispatch-batch-time",
      externalReference: "transfer-batch-time",
    });
    expect(() => database.raw.prepare(
      "UPDATE payout_batches SET submitted_at = submitted_at + 1 WHERE id = ?",
    ).run(prepared.batch.id)).toThrow(/invalid_payout_batch_time/);

    payouts.markPaid({
      batchId: prepared.batch.id,
      settlementReference: "settlement-batch-time",
      paidAt: 1,
    });
    expect(() => database.raw.prepare(
      "UPDATE payout_batches SET paid_at = paid_at + 1 WHERE id = ?",
    ).run(prepared.batch.id)).toThrow(/invalid_payout_batch_time/);
    expect(submitted.batch.submittedAt).toBeGreaterThanOrEqual(initial.created_at);
  });

  it("fails closed for SPORE until every gate is approved", () => {
    earning("earning-spore", "seller-1", "spore");
    const closed = new PayoutManager(database, closedPolicy);
    expect(() => closed.prepare({
      sellerId: "seller-1",
      payoutMethod: "spore",
      earningIds: ["earning-spore"],
      idempotencyKey: "spore-closed",
    })).toThrowError(expect.objectContaining({ code: "spore_payout_gates_closed" }));

    const open = new PayoutManager(database, {
      ...closedPolicy,
      spore: {
        legalApproved: true,
        custodyApproved: true,
        liquidityApproved: true,
        antifraudApproved: true,
      },
    });
    expect(open.prepare({
      sellerId: "seller-1",
      payoutMethod: "spore",
      earningIds: ["earning-spore"],
      idempotencyKey: "spore-open",
    }).batch).toMatchObject({ payoutMethod: "spore", amountUsdMicros: 125_000 });
  });

  it("rejects item reuse, ownership, method and external identity conflicts", () => {
    earning("earning-owned");
    earning("earning-other", "seller-2");
    const payouts = new PayoutManager(database, closedPolicy);
    expect(() => payouts.prepare({
      sellerId: "seller-1", payoutMethod: "stable",
      earningIds: ["earning-other"], idempotencyKey: "wrong-owner",
    })).toThrowError(expect.objectContaining({ code: "earning_owner_conflict" }));
    const first = payouts.prepare({
      sellerId: "seller-1", payoutMethod: "stable",
      earningIds: ["earning-owned"], idempotencyKey: "first",
    });
    expect(() => payouts.prepare({
      sellerId: "seller-1", payoutMethod: "stable",
      earningIds: ["earning-owned"], idempotencyKey: "second",
    })).toThrowError(expect.objectContaining({ code: "earning_not_available" }));
    payouts.markSubmitted({ batchId: first.batch.id, dispatchKey: "dispatch-unique", externalReference: "transfer-unique" });
    expect(() => payouts.markSubmitted({
      batchId: first.batch.id,
      dispatchKey: "dispatch-changed",
      externalReference: "transfer-changed",
    })).toThrowError(expect.objectContaining({ code: "payout_submission_conflict" }));
    expect(() => database.raw.prepare(
      "UPDATE payout_batches SET dispatch_key = 'dispatch-lateral' WHERE id = ?",
    ).run(first.batch.id)).toThrow(/immutable_payout_batch_provider_identity/);
    expect(() => database.raw.prepare(
      "UPDATE payout_batches SET external_reference = 'transfer-lateral' WHERE id = ?",
    ).run(first.batch.id)).toThrow(/immutable_payout_batch_provider_identity/);

    payouts.markPaid({
      batchId: first.batch.id,
      settlementReference: "settlement-unique",
      paidAt: 1,
    });
    expect(() => database.raw.prepare(
      "UPDATE payout_batches SET settlement_reference = 'settlement-lateral' WHERE id = ?",
    ).run(first.batch.id)).toThrow(/immutable_payout_batch_provider_identity/);
  });

  it("nets seller debt from a payout and restores it on safe cancellation", () => {
    earning("earning-net", "seller-1", "stable", 300_000);
    database.raw.prepare(
      "INSERT INTO seller_debts(seller_id, amount_usd_micros, updated_at) VALUES ('seller-1', 125000, 1)",
    ).run();
    const payouts = new PayoutManager(database, closedPolicy);
    const prepared = payouts.prepare({
      sellerId: "seller-1",
      payoutMethod: "stable",
      earningIds: ["earning-net"],
      idempotencyKey: "debt-net-1",
    });

    expect(prepared.batch).toMatchObject({
      grossUsdMicros: 300_000,
      debtOffsetUsdMicros: 125_000,
      amountUsdMicros: 175_000,
    });
    expect(database.raw.prepare(
      "SELECT amount_usd_micros FROM seller_debts WHERE seller_id = 'seller-1'",
    ).get()).toEqual({ amount_usd_micros: 0 });

    payouts.cancelPrepared(prepared.batch.id);
    expect(database.raw.prepare(
      "SELECT amount_usd_micros FROM seller_debts WHERE seller_id = 'seller-1'",
    ).get()).toEqual({ amount_usd_micros: 125_000 });
    expect(() => database.raw.prepare(
      "UPDATE seller_debts SET seller_id = 'seller-other' WHERE seller_id = 'seller-1'",
    ).run()).toThrow(/invalid_seller_debt_identity_time/);
    expect(() => database.raw.prepare(
      "UPDATE seller_debts SET updated_at = 0 WHERE seller_id = 'seller-1'",
    ).run()).toThrow(/invalid_seller_debt_identity_time/);
    expect(() => database.raw.prepare(
      "DELETE FROM seller_debts WHERE seller_id = 'seller-1'",
    ).run()).toThrow(/immutable_seller_debt/);
  });

  it("rejects a payout batch whose net amount does not balance", () => {
    expect(() => database.raw.prepare(`
      INSERT INTO payout_batches(
        id, seller_id, payout_method, idempotency_key, request_digest,
        gross_usd_micros, debt_offset_usd_micros, amount_usd_micros,
        status, created_at, updated_at
      ) VALUES ('batch-unbalanced', 'seller-1', 'stable', 'batch-unbalanced-key',
        'digest-unbalanced', 200000, 50000, 175000, 'prepared', 1, 1)
    `).run()).toThrow(/invalid_payout_batch_amount_equation/);
  });

  it("keeps the seller debt offset consumed after external settlement", () => {
    earning("earning-net-paid", "seller-1", "stable", 300_000);
    database.raw.prepare(
      "INSERT INTO seller_debts(seller_id, amount_usd_micros, updated_at) VALUES ('seller-1', 125000, 1)",
    ).run();
    const payouts = new PayoutManager(database, closedPolicy);
    const batch = payouts.prepare({
      sellerId: "seller-1",
      payoutMethod: "stable",
      earningIds: ["earning-net-paid"],
      idempotencyKey: "debt-net-paid",
    }).batch;
    payouts.markSubmitted({
      batchId: batch.id,
      dispatchKey: "debt-net-dispatch",
      externalReference: "debt-net-transfer",
    });
    payouts.markPaid({
      batchId: batch.id,
      settlementReference: "debt-net-settlement",
      paidAt: 1_786_000_000_000,
    });
    expect(database.raw.prepare(
      "SELECT amount_usd_micros FROM seller_debts WHERE seller_id = 'seller-1'",
    ).get()).toEqual({ amount_usd_micros: 0 });
    expect(payouts.getBatch(batch.id)).toMatchObject({ amountUsdMicros: 175_000, status: "paid" });
  });

  it("requires and snapshots a verified destination when policy enables self-service", () => {
    earning("earning-destination-one", "seller-1", "stable", 200_000);
    const strict = new PayoutManager(database, { ...closedPolicy, requireVerifiedDestination: true });
    expect(() => strict.prepare({
      sellerId: "seller-1", payoutMethod: "stable",
      earningIds: ["earning-destination-one"], idempotencyKey: "destination-missing",
    })).toThrowError(expect.objectContaining({ code: "verified_payout_destination_required" }));

    const now = Date.now();
    database.raw.prepare(
      `INSERT INTO seller_payout_destinations(
         id, seller_id, payout_method, destination_kind, destination_reference,
         destination_fingerprint, verifier_key_id, attestation_signature, status,
         verified_at, expires_at, created_at
       ) VALUES ('destination-old', 'seller-1', 'stable', 'provider_account',
                 'provider-account-old', ?, 'verifier-1', 'signature-fixture',
                 'active', ?, ?, ?)`,
    ).run("a".repeat(64), now, now + 86_400_000, now);
    const first = strict.prepare({
      sellerId: "seller-1", payoutMethod: "stable",
      earningIds: ["earning-destination-one"], idempotencyKey: "destination-first",
    }).batch;
    expect(first).toMatchObject({
      destinationId: "destination-old",
      destinationReference: "provider-account-old",
      destinationFingerprint: "a".repeat(64),
    });

    database.raw.prepare(
      "UPDATE seller_payout_destinations SET status = 'revoked', revoked_at = ? WHERE id = 'destination-old'",
    ).run(now + 1);
    database.raw.prepare(
      `INSERT INTO seller_payout_destinations(
         id, seller_id, payout_method, destination_kind, destination_reference,
         destination_fingerprint, verifier_key_id, attestation_signature, status,
         verified_at, expires_at, created_at
       ) VALUES ('destination-new', 'seller-1', 'stable', 'provider_account',
                 'provider-account-new', ?, 'verifier-1', 'signature-fixture',
                 'active', ?, ?, ?)`,
    ).run("b".repeat(64), now + 1, now + 86_400_000, now + 1);
    expect(strict.getBatch(first.id)).toMatchObject({
      destinationId: "destination-old",
      destinationFingerprint: "a".repeat(64),
    });
  });
});
