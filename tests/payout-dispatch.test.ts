import { generateKeyPairSync, sign } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  PayoutDispatchService,
  type PayoutGateway,
  type PayoutProviderObservation,
} from "../src/coordinator/payout-dispatch.js";
import { PayoutError, PayoutManager, type PayoutPolicy } from "../src/coordinator/payouts.js";
import {
  SporeConversionQuoteStore,
  SporeConversionQuoteVerifier,
  sporeConversionSigningBytes,
} from "../src/coordinator/spore-conversion.js";
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

describe("durable payout dispatch and reconciliation", () => {
  let database: MeshDatabase;
  let payouts: PayoutManager;

  beforeEach(() => {
    database = new MeshDatabase(":memory:");
    payouts = new PayoutManager(database, closedPolicy);
  });

  afterEach(() => database.close());

  function prepare(
    method: "stable" | "spore" = "stable",
    manager = payouts,
    destination?: { reference: string; fingerprint: string },
  ) {
    const id = `earning-${method}`;
    const now = Date.now();
    if (destination) {
      database.raw.prepare(
        `INSERT INTO seller_payout_destinations(
           id, seller_id, payout_method, destination_kind, destination_reference,
           destination_fingerprint, verifier_key_id, attestation_signature,
           status, verified_at, expires_at, created_at
         ) VALUES (?, 'seller-1', ?, ?, ?, ?, 'verifier-1', 'signed-fixture',
                   'active', ?, ?, ?)`,
      ).run(
        `destination-${method}`, method, method === "stable" ? "provider_account" : "wallet",
        destination.reference, destination.fingerprint, now, now + 300_000, now,
      );
    }
    database.raw.prepare(
      `INSERT INTO verified_work_receipts(
         receipt_id, seller_id, worker_id, job_id, stage_id, pricing_version,
         amount_usd_micros, evidence_digest, accepted_at, verifier_key_id, signature, recorded_at
       ) VALUES (?, 'seller-1', 'worker-1', ?, 'stage-1', 'pricing-1',
                 250000, ?, ?, 'verifier-1', 'signed-fixture', ?)`,
    ).run(`receipt-${method}`, `job-${method}`, "a".repeat(64), now, now);
    database.raw.prepare(
      `INSERT INTO seller_earnings(
         id, seller_id, receipt_id, amount_usd_micros, payout_method, status, created_at
       ) VALUES (?, 'seller-1', ?, 250000, ?, 'available', ?)`,
    ).run(id, `receipt-${method}`, method, now);
    return manager.prepare({
      sellerId: "seller-1",
      payoutMethod: method,
      earningIds: [id],
      idempotencyKey: `prepare-${method}`,
    }).batch;
  }

  function gateway(initial: PayoutProviderObservation, inspected = initial): PayoutGateway {
    return {
      createTransfer: vi.fn(async () => {
        if (initial.state === "absent") throw new Error("fixture cannot dispatch absent");
        return initial;
      }),
      inspectTransfer: vi.fn(async () => inspected),
    };
  }

  it("commits one stable dispatch identity and never sends a duplicate", async () => {
    const batch = prepare("stable", payouts, {
      reference: "provider-account-seller-1",
      fingerprint: "c".repeat(64),
    });
    const stable = gateway({ state: "submitted", externalReference: "transfer-1" });
    const service = new PayoutDispatchService(database, payouts, { stable });

    const first = await service.dispatch(batch.id);
    const replay = await service.dispatch(batch.id);

    expect(first).toMatchObject({
      duplicate: false,
      operation: { state: "submitted", dispatchKey: `payout:${batch.id}` },
      batch: { status: "submitted", externalReference: "transfer-1" },
    });
    expect(replay).toMatchObject({ duplicate: true, operation: { id: first.operation.id } });
    expect(stable.createTransfer).toHaveBeenCalledTimes(1);
    expect(stable.createTransfer).toHaveBeenCalledWith(expect.objectContaining({
      destinationReference: "provider-account-seller-1",
      destinationFingerprint: "c".repeat(64),
    }));
  });

  it("rejects an operation whose initial state contradicts its batch", () => {
    const batch = prepare();
    expect(() => database.raw.prepare(`
      INSERT INTO payout_dispatch_operations(
        id, batch_id, payout_method, dispatch_key, state, external_reference,
        created_at, updated_at
      ) VALUES ('dispatch-impossible', ?, 'stable', ?, 'submitted',
        'transfer-impossible', 1, 1)
    `).run(batch.id, `payout:${batch.id}`))
      .toThrow(/invalid_payout_dispatch_batch_projection/);
  });

  it("rejects dispatch allocation for a batch without its full reserved composition", () => {
    database.raw.prepare(`
      INSERT INTO payout_batches(
        id, seller_id, payout_method, idempotency_key, request_digest,
        gross_usd_micros, debt_offset_usd_micros, amount_usd_micros,
        status, created_at, updated_at
      ) VALUES ('batch-empty-composition', 'seller-1', 'stable',
        'batch-empty-composition-key', 'digest', 250000, 0, 250000,
        'prepared', 1, 1)
    `).run();
    expect(() => database.raw.prepare(`
      INSERT INTO payout_dispatch_operations(
        id, batch_id, payout_method, dispatch_key, state, created_at, updated_at
      ) VALUES ('dispatch-empty-composition', 'batch-empty-composition', 'stable',
        'payout:batch-empty-composition', 'dispatching', 1, 1)
    `).run()).toThrow(/invalid_payout_dispatch_economic_composition/);
  });

  it("rejects a lateral operation transition that leaves its batch behind", () => {
    const batch = prepare();
    database.raw.prepare(`
      INSERT INTO payout_dispatch_operations(
        id, batch_id, payout_method, dispatch_key, state, created_at, updated_at
      ) VALUES ('dispatch-transition', ?, 'stable', ?, 'dispatching', 1, 1)
    `).run(batch.id, `payout:${batch.id}`);
    expect(() => database.raw.prepare(`
      UPDATE payout_dispatch_operations
      SET state = 'submitted', external_reference = 'transfer-lateral'
      WHERE id = 'dispatch-transition'
    `).run()).toThrow(/invalid_payout_dispatch_batch_projection/);
  });

  it("preserves the monotonic dispatch audit timeline", () => {
    const batch = prepare();
    database.raw.prepare(`
      INSERT INTO payout_dispatch_operations(
        id, batch_id, payout_method, dispatch_key, state, created_at, updated_at
      ) VALUES ('dispatch-timeline', ?, 'stable', ?, 'dispatching', 100, 100)
    `).run(batch.id, `payout:${batch.id}`);
    expect(() => database.raw.prepare(
      "UPDATE payout_dispatch_operations SET created_at = 99 WHERE id = 'dispatch-timeline'",
    ).run()).toThrow(/invalid_payout_dispatch_time/);
    expect(() => database.raw.prepare(
      "UPDATE payout_dispatch_operations SET updated_at = 99 WHERE id = 'dispatch-timeline'",
    ).run()).toThrow(/invalid_payout_dispatch_time/);
    expect(() => database.raw.prepare(
      "UPDATE payout_dispatch_operations SET reconciled_at = 101 WHERE id = 'dispatch-timeline'",
    ).run()).toThrow(/invalid_payout_dispatch_time/);
  });

  it("turns a timeout into uncertain state and reconciles without resending", async () => {
    const batch = prepare();
    const stable: PayoutGateway = {
      createTransfer: vi.fn(async () => { throw new Error("timeout after send"); }),
      inspectTransfer: vi.fn(async (): Promise<PayoutProviderObservation> => ({
        state: "paid",
        externalReference: "transfer-timeout",
        settlementReference: "settlement-timeout",
        paidAt: 1_786_000_000_000,
      })),
    };
    const service = new PayoutDispatchService(database, payouts, { stable });

    const uncertain = await service.dispatch(batch.id);
    expect(uncertain).toMatchObject({ operation: { state: "uncertain", lastError: "timeout after send" } });
    expect(uncertain.batch.status).toBe("prepared");
    expect((await service.dispatch(batch.id)).duplicate).toBe(true);
    expect(stable.createTransfer).toHaveBeenCalledTimes(1);
    expect(() => payouts.cancelPrepared(batch.id)).toThrowError(expect.objectContaining({
      code: "payout_cancellation_unsafe",
    }));
    expect(() => database.raw.prepare(
      "UPDATE payout_batches SET status = 'cancelled', cancelled_at = ? WHERE id = ?",
    ).run(Date.now(), batch.id)).toThrow(/payout_cancellation_unsafe|invalid_payout_batch_time/);
    expect(database.raw.prepare(
      "SELECT status FROM seller_earnings WHERE id = 'earning-stable'",
    ).get()).toEqual({ status: "batched" });

    const reconciled = await service.reconcile(batch.id);
    expect(reconciled).toMatchObject({
      operation: {
        state: "paid",
        externalReference: "transfer-timeout",
        settlementReference: "settlement-timeout",
      },
      batch: { status: "paid" },
    });
    expect(stable.inspectTransfer).toHaveBeenCalledWith({
      dispatchKey: `payout:${batch.id}`,
      externalReference: null,
    });
    expect(stable.createTransfer).toHaveBeenCalledTimes(1);
  });

  it("rolls back batch and earnings when the durable operation cannot record paid", async () => {
    const batch = prepare();
    database.raw.exec(`
      CREATE TRIGGER fixture_reject_paid_operation
      BEFORE UPDATE OF state ON payout_dispatch_operations
      WHEN NEW.state = 'paid'
      BEGIN
        SELECT RAISE(ABORT, 'fixture_paid_operation_failure');
      END;
    `);
    const stable = gateway({
      state: "paid",
      externalReference: "transfer-atomic",
      settlementReference: "settlement-atomic",
      paidAt: 1_786_000_000_000,
    });
    const service = new PayoutDispatchService(database, payouts, { stable });

    const uncertain = await service.dispatch(batch.id);

    expect(uncertain).toMatchObject({
      operation: {
        state: "uncertain",
        externalReference: null,
        settlementReference: null,
        lastError: "fixture_paid_operation_failure",
      },
      batch: { status: "prepared", externalReference: null, settlementReference: null },
    });
    expect(database.raw.prepare(
      "SELECT status FROM seller_earnings WHERE id = 'earning-stable'",
    ).get()).toEqual({ status: "batched" });
  });

  it("blocks a divergent projection before provider inspection", async () => {
    const batch = prepare();
    const stable: PayoutGateway = {
      createTransfer: vi.fn(async () => { throw new Error("timeout after send"); }),
      inspectTransfer: vi.fn(async (): Promise<PayoutProviderObservation> => ({ state: "absent" })),
    };
    const service = new PayoutDispatchService(database, payouts, { stable });
    await service.dispatch(batch.id);
    expect(() => database.raw.prepare(
      `UPDATE payout_dispatch_operations
       SET state = 'submitted', external_reference = 'transfer-lateral'
       WHERE batch_id = ?`,
    ).run(batch.id)).toThrow(/invalid_payout_dispatch_batch_projection/);
    expect(stable.inspectTransfer).not.toHaveBeenCalled();
    expect(stable.createTransfer).toHaveBeenCalledTimes(1);
  });

  it("blocks recovery when a reserved earning was released laterally", async () => {
    const batch = prepare();
    const stable: PayoutGateway = {
      createTransfer: vi.fn(async () => { throw new Error("timeout after send"); }),
      inspectTransfer: vi.fn(async (): Promise<PayoutProviderObservation> => ({ state: "absent" })),
    };
    const service = new PayoutDispatchService(database, payouts, { stable });
    await service.dispatch(batch.id);
    expect(() => database.raw.prepare(
      "UPDATE seller_earnings SET status = 'available' WHERE id = 'earning-stable'",
    ).run()).toThrow(/invalid_seller_earning_status_transition/);
    database.raw.exec("DROP TRIGGER seller_earning_status_transition_guard");
    database.raw.prepare(
      "UPDATE seller_earnings SET status = 'available' WHERE id = 'earning-stable'",
    ).run();

    expect(() => payouts.prepare({
      sellerId: "seller-1",
      payoutMethod: "stable",
      earningIds: ["earning-stable"],
      idempotencyKey: "lateral-double-allocation",
    })).toThrowError(expect.objectContaining({ code: "earning_already_allocated" }));
    expect(database.raw.prepare(
      "SELECT COUNT(*) AS count FROM payout_batches WHERE idempotency_key = 'lateral-double-allocation'",
    ).get()).toEqual({ count: 0 });
    const now = Date.now();
    database.raw.prepare(
      `INSERT INTO payout_batches(
         id, seller_id, payout_method, idempotency_key, request_digest,
         gross_usd_micros, amount_usd_micros, status, created_at, updated_at
       ) VALUES ('payout-lateral', 'seller-1', 'stable', 'sql-lateral-allocation',
                 'fixture-digest', 250000, 250000, 'prepared', ?, ?)`,
    ).run(now, now);
    expect(() => database.raw.prepare(
      `INSERT INTO payout_batch_items(batch_id, earning_id, amount_usd_micros)
       VALUES ('payout-lateral', 'earning-stable', 250000)`,
    ).run()).toThrow(/invalid_payout_batch_item/);

    await expect(service.reconcile(batch.id)).rejects.toMatchObject({
      code: "payout_dispatch_projection_conflict",
    });
    await expect(service.dispatch(batch.id)).rejects.toMatchObject({
      code: "payout_dispatch_projection_conflict",
    });
    expect(stable.inspectTransfer).not.toHaveBeenCalled();
    expect(stable.createTransfer).toHaveBeenCalledTimes(1);
  });

  it("keeps a rejected dispatch locked when provider identity was already observed", async () => {
    const batch = prepare();
    const stable = gateway(
      { state: "submitted", externalReference: "transfer-before-rejection" },
      { state: "rejected", reason: "provider_reversed" },
    );
    const service = new PayoutDispatchService(database, payouts, { stable });

    await service.dispatch(batch.id);
    const rejected = await service.reconcile(batch.id);

    expect(rejected).toMatchObject({
      operation: { state: "rejected", externalReference: "transfer-before-rejection" },
      batch: { status: "submitted" },
    });
    expect(() => payouts.cancelPrepared(batch.id)).toThrowError(expect.objectContaining({
      code: "payout_cancellation_unsafe",
    }));
    expect(database.raw.prepare(
      "SELECT status FROM seller_earnings WHERE id = 'earning-stable'",
    ).get()).toEqual({ status: "batched" });
  });

  it("rejects a payout whose sealed item composition was polluted", async () => {
    const batch = prepare();
    const now = Date.now();
    database.raw.prepare(
      `INSERT INTO verified_work_receipts(
         receipt_id, seller_id, worker_id, job_id, stage_id, pricing_version,
         amount_usd_micros, evidence_digest, accepted_at, verifier_key_id, signature, recorded_at
       ) VALUES ('receipt-extra', 'seller-1', 'worker-2', 'job-extra', 'stage-1', 'pricing-1',
                 250000, ?, ?, 'verifier-1', 'signed-fixture', ?)`,
    ).run("d".repeat(64), now, now);
    database.raw.prepare(
      `INSERT INTO seller_earnings(
         id, seller_id, receipt_id, amount_usd_micros, payout_method, status, created_at
       ) VALUES ('earning-extra', 'seller-1', 'receipt-extra', 250000, 'stable', 'available', ?)`,
    ).run(now);
    database.raw.prepare(
      `INSERT INTO payout_batch_items(batch_id, earning_id, amount_usd_micros)
       VALUES (?, 'earning-extra', 250000)`,
    ).run(batch.id);
    expect(() => database.raw.prepare(
      `UPDATE payout_batch_items SET amount_usd_micros = 1
       WHERE batch_id = ? AND earning_id = 'earning-extra'`,
    ).run(batch.id)).toThrow(/immutable_payout_batch_item/);
    expect(() => database.raw.prepare(
      "DELETE FROM payout_batch_items WHERE batch_id = ? AND earning_id = 'earning-extra'",
    ).run(batch.id)).toThrow(/immutable_payout_batch_item/);

    const stable = gateway({ state: "submitted", externalReference: "transfer-polluted" });
    const service = new PayoutDispatchService(database, payouts, { stable });
    await expect(service.dispatch(batch.id)).rejects.toMatchObject({ code: "payout_batch_audit_conflict" });
    expect(stable.createTransfer).not.toHaveBeenCalled();
    expect(service.getByBatchId(batch.id)).toBeNull();
  });

  it("keeps cancelled payout batches terminal and evidence-consistent", () => {
    const batch = prepare();
    expect(payouts.cancelPrepared(batch.id)).toMatchObject({
      duplicate: false,
      batch: { status: "cancelled", cancelledAt: expect.any(Number) },
    });
    expect(() => database.raw.prepare(
      "UPDATE payout_batches SET status = 'prepared', cancelled_at = NULL WHERE id = ?",
    ).run(batch.id)).toThrow(/invalid_payout_batch_/);
    expect(() => database.raw.prepare(
      `UPDATE payout_batches
       SET status = 'paid', dispatch_key = 'payout:forged',
           external_reference = 'transfer-forged', settlement_reference = 'settlement-forged',
           submitted_at = 1, paid_at = 1, cancelled_at = NULL
       WHERE id = ?`,
    ).run(batch.id)).toThrow(/invalid_payout_batch_state_transition|invalid_payout_batch_time/);
    expect(payouts.getBatch(batch.id)).toMatchObject({ status: "cancelled" });
  });

  it("promotes only identity-bound settlement evidence and replays idempotently", async () => {
    const batch = prepare();
    const stable = gateway({ state: "submitted", externalReference: "transfer-settlement" });
    const service = new PayoutDispatchService(database, payouts, { stable });
    await service.dispatch(batch.id);
    const evidence = {
      schema: "mycellios.payout-settlement.v1" as const,
      batchId: batch.id,
      externalReference: "transfer-settlement",
      settlementReference: "bank-payout-settlement",
      paidAt: 1_786_000_000_000,
      issuedAt: 1_786_000_001_000,
      verifierKeyId: "settlement-verifier",
      signature: "signed-settlement-evidence",
    };

    const paid = service.recordSettlement(evidence);
    const replay = service.recordSettlement(evidence);
    expect(paid).toMatchObject({ duplicate: false, operation: { state: "paid" }, batch: { status: "paid" } });
    expect(replay).toMatchObject({ duplicate: true, operation: { state: "paid" }, batch: { status: "paid" } });
    expect(database.raw.prepare(
      "SELECT verifier_key_id, signature FROM payout_settlement_evidence WHERE batch_id = ?",
    ).get(batch.id)).toEqual({
      verifier_key_id: "settlement-verifier",
      signature: "signed-settlement-evidence",
    });
    expect(() => service.recordSettlement({ ...evidence, externalReference: "transfer-other" }))
      .toThrowError(expect.objectContaining({ code: "payout_settlement_identity_conflict" }));
    expect(() => service.recordSettlement({ ...evidence, settlementReference: "bank-payout-other" }))
      .toThrowError(expect.objectContaining({ code: "payout_settlement_evidence_conflict" }));
    expect(() => database.raw.prepare(
      "UPDATE payout_dispatch_operations SET settlement_reference = 'bank-payout-other' WHERE batch_id = ?",
    ).run(batch.id)).toThrow(/immutable_payout_dispatch_provider_identity|invalid_payout_settlement_evidence_binding|invalid_payout_dispatch_batch_projection/);
    expect(() => database.raw.prepare(
      "UPDATE payout_settlement_evidence SET signature = 'replacement-signature' WHERE batch_id = ?",
    ).run(batch.id)).toThrow(/immutable_payout_settlement_evidence/);
    expect(() => database.raw.prepare(
      "UPDATE payout_dispatch_operations SET state = 'uncertain' WHERE batch_id = ?",
    ).run(batch.id)).toThrow(/invalid_payout_dispatch_/);
    expect(() => database.raw.prepare(
      "DELETE FROM payout_settlement_evidence WHERE batch_id = ?",
    ).run(batch.id)).toThrow(/immutable_payout_settlement_evidence/);
    expect(() => database.raw.prepare(
      "DELETE FROM payout_dispatch_operations WHERE batch_id = ?",
    ).run(batch.id)).toThrow(/immutable_payout_dispatch_operation/);
    expect(() => database.raw.prepare(
      "DELETE FROM payout_batches WHERE id = ?",
    ).run(batch.id)).toThrow(/immutable_payout_/);
    expect(service.getByBatchId(batch.id)).not.toBeNull();
    expect(service.getSettlementEvidence(batch.id)).not.toBeNull();
  });

  it("rejects settlement evidence that is not bound to the durable dispatch", async () => {
    const batch = prepare();
    expect(() => database.raw.prepare(`
      INSERT INTO payout_settlement_evidence(
        batch_id, schema_name, external_reference, settlement_reference,
        paid_at, issued_at, verifier_key_id, signature, recorded_at
      ) VALUES (?, 'mycellios.payout-settlement.v1', 'transfer-forged',
        'settlement-forged', 1786000000000, 1786000001000,
        'settlement-verifier', 'signature', 1786000002000)
    `).run(batch.id)).toThrow(/invalid_payout_settlement_evidence_binding/);

    const stable = gateway({ state: "submitted", externalReference: "transfer-real" });
    const service = new PayoutDispatchService(database, payouts, { stable });
    await service.dispatch(batch.id);
    expect(() => database.raw.prepare(`
      INSERT INTO payout_settlement_evidence(
        batch_id, schema_name, external_reference, settlement_reference,
        paid_at, issued_at, verifier_key_id, signature, recorded_at
      ) VALUES (?, 'mycellios.payout-settlement.v1', 'transfer-other',
        'settlement-other', 1786000000000, 1786000001000,
        'settlement-verifier', 'signature', 1786000002000)
    `).run(batch.id)).toThrow(/invalid_payout_settlement_evidence_binding/);
    expect(() => database.raw.prepare(`
      INSERT INTO payout_settlement_evidence(
        batch_id, schema_name, external_reference, settlement_reference,
        paid_at, issued_at, verifier_key_id, signature, recorded_at
      ) VALUES (?, 'mycellios.payout-settlement.v1', 'transfer-real',
        'settlement-impossible-time', 1786000060001, 1786000000000,
        'settlement-verifier', 'signature', 1786000001000)
    `).run(batch.id)).toThrow(/invalid_payout_settlement_evidence_binding/);
  });

  it("keeps an absent provider lookup uncertain", async () => {
    const batch = prepare();
    const stable: PayoutGateway = {
      createTransfer: vi.fn(async () => { throw new Error("connection reset"); }),
      inspectTransfer: vi.fn(async (): Promise<PayoutProviderObservation> => ({ state: "absent" })),
    };
    const service = new PayoutDispatchService(database, payouts, { stable });
    await service.dispatch(batch.id);

    const result = await service.reconcile(batch.id);
    expect(result).toMatchObject({
      duplicate: true,
      operation: { state: "uncertain", lastError: "provider_transfer_absent" },
      batch: { status: "prepared" },
    });
    expect(stable.createTransfer).toHaveBeenCalledTimes(1);
  });

  it("atomically releases a definitive rejection without provider identity", async () => {
    const batch = prepare();
    const stable = gateway({ state: "rejected", reason: "destination_unverified" });
    const service = new PayoutDispatchService(database, payouts, { stable });

    const result = await service.dispatch(batch.id);
    expect(result).toMatchObject({
      operation: { state: "rejected", lastError: "destination_unverified" },
      batch: { status: "cancelled" },
    });
    expect(database.raw.prepare(
      "SELECT status FROM seller_earnings WHERE id = 'earning-stable'",
    ).get()).toEqual({ status: "available" });
    expect(() => database.raw.prepare(
      `UPDATE payout_dispatch_operations
       SET external_reference = 'transfer-late' WHERE batch_id = ?`,
    ).run(batch.id)).toThrow(/immutable_payout_dispatch_provider_identity|invalid_payout_dispatch_batch_projection/);
    expect(await service.reconcile(batch.id)).toMatchObject({
      duplicate: true,
      operation: { state: "rejected" },
      batch: { status: "cancelled" },
    });
    expect(() => service.recordSettlement({
      schema: "mycellios.payout-settlement.v1",
      batchId: batch.id,
      externalReference: "rejected-transfer",
      settlementReference: "rejected-settlement",
      paidAt: Date.now(),
      issuedAt: Date.now(),
      verifierKeyId: "settlement-verifier",
      signature: "signed-rejected-settlement",
    })).toThrowError(expect.objectContaining({ code: "payout_dispatch_terminal" }));
    expect(service.getSettlementEvidence(batch.id)).toBeNull();
  });

  it("rechecks SPORE gates at dispatch even when the batch was prepared earlier", async () => {
    const openManager = new PayoutManager(database, {
      ...closedPolicy,
      spore: {
        legalApproved: true,
        custodyApproved: true,
        liquidityApproved: true,
        antifraudApproved: true,
      },
    });
    const destinationFingerprint = "a".repeat(64);
    const batch = prepare("spore", openManager, {
      reference: "wallet-seller-1",
      fingerprint: destinationFingerprint,
    });
    const spore = gateway({ state: "submitted", externalReference: "spore-transfer-1" });
    const service = new PayoutDispatchService(database, payouts, { spore });

    await expect(service.dispatch(batch.id)).rejects.toMatchObject({
      code: "spore_payout_gates_closed",
    } satisfies Partial<PayoutError>);
    expect(spore.createTransfer).not.toHaveBeenCalled();
    expect(service.getByBatchId(batch.id)).toBeNull();
  });

  it("requires a fresh verified SPORE quote and passes its snapshot to the gateway", async () => {
    const keys = generateKeyPairSync("ed25519");
    const publicKey = keys.publicKey.export({ type: "spki", format: "pem" }).toString();
    const openManager = new PayoutManager(database, {
      ...closedPolicy,
      spore: {
        legalApproved: true,
        custodyApproved: true,
        liquidityApproved: true,
        antifraudApproved: true,
      },
    });
    const destinationFingerprint = "a".repeat(64);
    const batch = prepare("spore", openManager, {
      reference: "wallet-seller-1",
      fingerprint: destinationFingerprint,
    });
    const spore = gateway({ state: "submitted", externalReference: "spore-transfer-1" });
    const missingService = new PayoutDispatchService(database, openManager, { spore });
    await expect(missingService.dispatch(batch.id)).rejects.toMatchObject({ code: "spore_quote_required" });
    expect(spore.createTransfer).not.toHaveBeenCalled();
    expect(missingService.getByBatchId(batch.id)).toBeNull();

    const now = Date.now();
    let clock = now;
    const unsigned = {
      schema: "mycellios.spore-conversion-quote.v1" as const,
      quoteId: "dispatch-quote-1", batchId: batch.id, sellerId: batch.sellerId,
      usdMicros: batch.amountUsdMicros, chainId: "base", assetId: "spore-1",
      tokenDecimals: 18, tokenAtomicAmount: "250000000000000000",
      destinationFingerprint, issuedAt: now, expiresAt: now + 60_000,
      oracleKeyId: "oracle",
    };
    const quotes = new SporeConversionQuoteStore(database, new SporeConversionQuoteVerifier({
      trustedOracleKeys: new Map([["oracle", publicKey]]),
      approvedAssets: new Map([["base:spore-1", { tokenDecimals: 18 }]]),
      maxQuoteAgeMs: 300_000,
      now: () => clock,
    }), () => clock);
    const quote = {
      ...unsigned,
      signature: sign(null, sporeConversionSigningBytes(unsigned), keys.privateKey).toString("base64url"),
    };
    quotes.record(quote);
    const service = new PayoutDispatchService(database, openManager, { spore }, quotes);
    await expect(service.dispatch(batch.id)).resolves.toMatchObject({ operation: { state: "submitted" } });
    expect(service.getByBatchId(batch.id)).toMatchObject({
      sporeQuoteId: quote.quoteId,
      sporeQuoteAttestationDigest: expect.stringMatching(/^[a-f0-9]{64}$/),
    });
    expect(spore.createTransfer).toHaveBeenCalledWith(expect.objectContaining({
      batchId: batch.id,
      sporeQuote: quote,
    }));
    clock = quote.expiresAt + 1;
    await expect(service.reconcile(batch.id)).resolves.toMatchObject({
      operation: { state: "submitted" },
    });
    expect(spore.inspectTransfer).toHaveBeenCalledTimes(1);
    database.raw.prepare(
      "UPDATE spore_conversion_quotes SET status = 'superseded', replaced_at = ? WHERE quote_id = ?",
    ).run(quote.expiresAt + 2, quote.quoteId);
    expect(() => database.raw.prepare(
      "UPDATE spore_conversion_quotes SET status = 'active', replaced_at = NULL WHERE quote_id = ?",
    ).run(quote.quoteId)).toThrow(/invalid_spore_conversion_quote_lifecycle/);
    expect(() => database.raw.prepare(
      "DELETE FROM spore_conversion_quotes WHERE quote_id = ?",
    ).run(quote.quoteId)).toThrow(/immutable_spore_conversion_quote/);
    expect(() => database.raw.prepare(
      "UPDATE spore_conversion_quotes SET token_atomic_amount = ? WHERE quote_id = ?",
    ).run("999", quote.quoteId)).toThrow(/immutable_spore_conversion_quote_content/);
    database.raw.exec("DROP TRIGGER spore_conversion_quote_content_immutable_guard");
    database.raw.prepare("UPDATE spore_conversion_quotes SET token_atomic_amount = ? WHERE quote_id = ?")
      .run("999", quote.quoteId);
    await expect(service.reconcile(batch.id)).rejects.toMatchObject({
      code: "spore_dispatch_quote_audit_conflict",
    });
    expect(spore.inspectTransfer).toHaveBeenCalledTimes(1);
    expect(() => service.recordSettlement({
      schema: "mycellios.payout-settlement.v1",
      batchId: batch.id,
      externalReference: "spore-transfer-1",
      settlementReference: "spore-settlement-1",
      paidAt: quote.expiresAt + 2,
      issuedAt: quote.expiresAt + 3,
      verifierKeyId: "settlement-verifier",
      signature: "signed-spore-settlement",
    })).toThrowError(expect.objectContaining({ code: "spore_dispatch_quote_audit_conflict" }));
    expect(database.raw.prepare(
      "SELECT COUNT(*) AS count FROM payout_settlement_evidence WHERE batch_id = ?",
    ).get(batch.id)).toEqual({ count: 0 });
  });

  it("enforces stable/SPORE quote audit consistency at the database boundary", () => {
    const openManager = new PayoutManager(database, {
      ...closedPolicy,
      spore: {
        legalApproved: true, custodyApproved: true,
        liquidityApproved: true, antifraudApproved: true,
      },
    });
    const sporeBatch = prepare("spore", openManager, {
      reference: "wallet-spore-audit",
      fingerprint: "d".repeat(64),
    });
    expect(() => database.raw.prepare(
      `INSERT INTO payout_dispatch_operations(
         id, batch_id, payout_method, dispatch_key, state, created_at, updated_at
       ) VALUES ('invalid-spore', ?, 'spore', ?, 'dispatching', 1, 1)`,
    ).run(sporeBatch.id, `payout:${sporeBatch.id}`)).toThrow(/invalid_payout_dispatch_spore_quote_audit/);
    expect(() => database.raw.prepare(
      `INSERT INTO payout_dispatch_operations(
         id, batch_id, payout_method, dispatch_key, state, spore_quote_id,
         spore_quote_attestation_digest, created_at, updated_at
       ) VALUES ('invalid-quote-binding', ?, 'spore', ?, 'dispatching',
         'quote-missing', ?, 1, 1)`,
    ).run(sporeBatch.id, `payout:${sporeBatch.id}`, "b".repeat(64)))
      .toThrow(/invalid_payout_dispatch_spore_quote_binding/);

    database.raw.prepare(`
      INSERT INTO spore_conversion_quotes(
        quote_id, batch_id, seller_id, usd_micros, chain_id, asset_id,
        token_decimals, token_atomic_amount, destination_fingerprint,
        issued_at, expires_at, oracle_key_id, signature, status, recorded_at
      ) VALUES ('quote-1', ?, 'seller-1', 250000, 'base', 'spore-1',
        18, '1', ?, 1, 2, 'oracle', 'signature', 'active', 1)
    `).run(sporeBatch.id, "d".repeat(64));

    const stableBatch = prepare("stable", openManager);
    expect(() => database.raw.prepare(
      `INSERT INTO payout_dispatch_operations(
         id, batch_id, payout_method, dispatch_key, state, created_at, updated_at
       ) VALUES ('invalid-binding', ?, 'stable', 'payout:other', 'dispatching', 1, 1)`,
    ).run(stableBatch.id)).toThrow(/invalid_payout_dispatch_batch_binding/);
    expect(() => database.raw.prepare(
      `INSERT INTO payout_dispatch_operations(
         id, batch_id, payout_method, dispatch_key, state, created_at, updated_at
       ) VALUES ('invalid-rail', ?, 'stable', ?, 'dispatching', 1, 1)`,
    ).run(sporeBatch.id, `payout:${sporeBatch.id}`)).toThrow(/invalid_payout_dispatch_batch_binding/);
    expect(() => database.raw.prepare(
      `INSERT INTO payout_dispatch_operations(
         id, batch_id, payout_method, dispatch_key, state, spore_quote_id,
         spore_quote_attestation_digest, created_at, updated_at
       ) VALUES ('invalid-stable', ?, 'stable', ?, 'dispatching', 'quote', ?, 1, 1)`,
    ).run(stableBatch.id, `payout:${stableBatch.id}`, "a".repeat(64)))
      .toThrow(/invalid_payout_dispatch_spore_quote_audit/);
    database.raw.prepare(
      `INSERT INTO payout_dispatch_operations(
         id, batch_id, payout_method, dispatch_key, state, created_at, updated_at
       ) VALUES ('valid-stable', ?, 'stable', ?, 'dispatching', 1, 1)`,
    ).run(stableBatch.id, `payout:${stableBatch.id}`);
    expect(() => database.raw.prepare(
      `UPDATE payout_dispatch_operations
       SET spore_quote_id = 'quote', spore_quote_attestation_digest = ?
       WHERE id = 'valid-stable'`,
    ).run("a".repeat(64))).toThrow(/payout_dispatch_spore_quote_audit/);

    database.raw.prepare(
      `INSERT INTO payout_dispatch_operations(
         id, batch_id, payout_method, dispatch_key, state, spore_quote_id,
         spore_quote_attestation_digest, created_at, updated_at
       ) VALUES ('valid-spore', ?, 'spore', ?, 'dispatching', 'quote-1', ?, 1, 1)`,
    ).run(sporeBatch.id, `payout:${sporeBatch.id}`, "b".repeat(64));
    expect(() => database.raw.prepare(
      `UPDATE payout_dispatch_operations
       SET spore_quote_id = 'quote-2', spore_quote_attestation_digest = ?
       WHERE id = 'valid-spore'`,
    ).run("c".repeat(64))).toThrow(/immutable_payout_dispatch_spore_quote_audit/);
    expect(() => database.raw.prepare(
      "UPDATE payout_dispatch_operations SET payout_method = 'stable' WHERE id = 'valid-spore'",
    ).run()).toThrow(/immutable_payout_dispatch_spore_quote_audit/);
    expect(() => database.raw.prepare(
      "UPDATE payout_dispatch_operations SET state = 'submitted' WHERE id = 'valid-spore'",
    ).run()).toThrow(/invalid_payout_dispatch_state_evidence/);

    payouts.markSubmitted({
      batchId: stableBatch.id,
      dispatchKey: `payout:${stableBatch.id}`,
      externalReference: "transfer-1",
    });
    database.raw.prepare(
      `UPDATE payout_dispatch_operations
       SET state = 'submitted', external_reference = 'transfer-1'
       WHERE id = 'valid-stable'`,
    ).run();
    expect(() => database.raw.prepare(
      "UPDATE payout_dispatch_operations SET external_reference = 'transfer-2' WHERE id = 'valid-stable'",
    ).run()).toThrow(/immutable_payout_dispatch_provider_identity|invalid_payout_dispatch_batch_projection/);
    expect(() => database.raw.prepare(
      "UPDATE payout_dispatch_operations SET dispatch_key = 'payout:other' WHERE id = 'valid-stable'",
    ).run()).toThrow(/immutable_payout_dispatch_provider_identity/);
    database.raw.prepare(
      "UPDATE payout_dispatch_operations SET state = 'uncertain' WHERE id = 'valid-stable'",
    ).run();
    expect(() => database.raw.prepare(
      "UPDATE payout_dispatch_operations SET state = 'paid' WHERE id = 'valid-stable'",
    ).run()).toThrow(/invalid_payout_dispatch_state_evidence/);
    expect(() => database.raw.prepare(
      `UPDATE payout_dispatch_operations
       SET settlement_reference = 'premature-settlement' WHERE id = 'valid-stable'`,
    ).run()).toThrow(/invalid_payout_dispatch_state_evidence/);
    expect(() => database.raw.prepare(
      "UPDATE payout_dispatch_operations SET state = 'dispatching' WHERE id = 'valid-stable'",
    ).run()).toThrow(/invalid_payout_dispatch_state_transition|invalid_payout_dispatch_batch_projection/);
    expect(() => database.raw.prepare(
      "UPDATE payout_batches SET amount_usd_micros = 999 WHERE id = ?",
    ).run(stableBatch.id)).toThrow(/immutable_payout_batch_economic_snapshot/);
    expect(() => database.raw.prepare(
      "UPDATE payout_batches SET destination_reference = 'other-destination' WHERE id = ?",
    ).run(stableBatch.id)).toThrow(/immutable_payout_batch_economic_snapshot/);
  });
});
