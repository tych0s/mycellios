import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BillingOperationsMonitor } from "../src/coordinator/billing-operations.js";
import { createCoordinator, type CoordinatorRuntime } from "../src/coordinator/server.js";
import type { SupabaseAuthService } from "../src/coordinator/supabase-auth.js";
import { MeshDatabase } from "../src/storage/database.js";

const NOW = 1_800_000_000_000;

describe("billing operational reconciliation monitor", () => {
  let database: MeshDatabase;

  beforeEach(() => {
    database = new MeshDatabase(":memory:");
  });

  afterEach(() => database.close());

  it("reports bounded stablecoin and payout reconciliation alerts without payload secrets", () => {
    seedAccount(database);
    seedIntent(database, "intent-allocating", "allocating", NOW - 20_000, NOW + 60_000);
    seedIntent(database, "intent-expired", "awaiting_payment", NOW - 20_000, NOW - 1_000);
    seedIntent(database, "intent-orphan", "consumed", NOW - 20_000, NOW + 60_000, "stablecoin:missing-event");
    seedPayout(database, "payout-uncertain", "prepared", NOW - 20_000);
    seedDispatch(database, "payout-uncertain", "uncertain", NOW - 20_000);
    seedPayout(database, "payout-awaiting-settlement", "submitted", NOW - 20_000);
    seedDispatch(database, "payout-awaiting-settlement", "submitted", NOW - 20_000);
    seedPayout(database, "payout-manual", "submitted", NOW - 20_000);
    seedPayout(database, "spore-missing", "prepared", NOW - 30_000, "spore");
    seedPayout(database, "spore-expired", "prepared", NOW - 20_000, "spore");
    seedSporeQuote(database, "spore-expired", NOW - 1_000);

    const snapshot = new BillingOperationsMonitor(database, {
      stablecoinAllocationStuckMs: 10_000,
      payoutDispatchStuckMs: 10_000,
      payoutSettlementStuckMs: 10_000,
      maxAlerts: 10,
      now: () => NOW,
    }).snapshot();

    expect(snapshot.stablecoinIntents).toEqual({
      allocating: 1,
      awaitingPayment: 1,
      consumed: 1,
      expiredAwaitingPayment: 1,
    });
    expect(snapshot.payoutDispatches).toMatchObject({ uncertain: 1 });
    expect(snapshot.settlementCandidates).toEqual([{
      batchId: "payout-awaiting-settlement",
      externalReference: "transfer-payout-awaiting-settlement",
      amountUsdMicros: 200_000,
      submittedAt: NOW - 20_000,
      ageMs: 20_000,
    }]);
    expect(snapshot.settlementCandidatesTruncated).toBe(false);
    expect(snapshot.sporeQuoteCandidates).toEqual([
      {
        batchId: "spore-missing", amountUsdMicros: 200_000,
        preparedAt: NOW - 30_000, ageMs: 30_000,
        reason: "missing", quoteExpiresAt: null,
      },
      {
        batchId: "spore-expired", amountUsdMicros: 200_000,
        preparedAt: NOW - 20_000, ageMs: 20_000,
        reason: "expired", quoteExpiresAt: NOW - 1_000,
      },
    ]);
    expect(snapshot.sporeQuoteCandidatesTruncated).toBe(false);
    expect(snapshot.alerts.map((alert) => alert.code)).toEqual(expect.arrayContaining([
      "stablecoin_allocation_stuck",
      "stablecoin_consumption_orphaned",
      "payout_dispatch_stuck",
      "payout_submitted_without_dispatch",
      "stablecoin_payment_expired",
    ]));
    expect(snapshot.alerts.slice(0, 4).every((alert) => alert.severity === "critical")).toBe(true);
    expect(JSON.stringify(snapshot)).not.toContain("request_digest");
    expect(JSON.stringify(snapshot)).not.toContain("recipient-secret");
    expect(snapshot.alertsTruncated).toBe(false);
  });

  it("caps alert output while preserving aggregate counts", () => {
    seedAccount(database);
    seedIntent(database, "intent-one", "allocating", NOW - 20_000, NOW + 60_000);
    seedIntent(database, "intent-two", "allocating", NOW - 30_000, NOW + 60_000);
    seedPayout(database, "payout-candidate-one", "submitted", NOW - 30_000);
    seedDispatch(database, "payout-candidate-one", "submitted", NOW - 30_000);
    seedPayout(database, "payout-candidate-two", "submitted", NOW - 20_000);
    seedDispatch(database, "payout-candidate-two", "submitted", NOW - 20_000);
    seedPayout(database, "spore-candidate-one", "prepared", NOW - 30_000, "spore");
    seedPayout(database, "spore-candidate-two", "prepared", NOW - 20_000, "spore");
    const snapshot = new BillingOperationsMonitor(database, {
      stablecoinAllocationStuckMs: 10_000,
      payoutDispatchStuckMs: 10_000,
      payoutSettlementStuckMs: 10_000,
      maxAlerts: 1,
      now: () => NOW,
    }).snapshot();
    expect(snapshot.stablecoinIntents.allocating).toBe(2);
    expect(snapshot.alerts).toHaveLength(1);
    expect(snapshot.alertsTruncated).toBe(true);
    expect(snapshot.settlementCandidates.map((candidate) => candidate.batchId)).toEqual([
      "payout-candidate-one",
    ]);
    expect(snapshot.settlementCandidatesTruncated).toBe(true);
    expect(snapshot.sporeQuoteCandidates.map((candidate) => candidate.batchId)).toEqual([
      "spore-candidate-one",
    ]);
    expect(snapshot.sporeQuoteCandidatesTruncated).toBe(true);
  });
});

describe("billing operations administrative route", () => {
  const runtimes: CoordinatorRuntime[] = [];
  afterEach(async () => {
    await Promise.all(runtimes.splice(0).map((runtime) => runtime.close()));
  });

  it("allows owner/admin sessions and rejects an ordinary account", async () => {
    let role: "owner" | null = null;
    const auth = {
      authenticate: vi.fn(async () => ({ id: "operator-1", email: null, role })),
    } as unknown as SupabaseAuthService;
    const runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 0,
      databasePath: ":memory:",
      requestTimeoutMs: 1_000,
      modelAdminToken: "separate-admin-secret",
    }, { logger: false, supabaseAuthService: auth });
    runtimes.push(runtime);

    const forbidden = await runtime.app.inject({
      method: "GET",
      url: "/public/v1/admin/billing-operations",
      headers: { authorization: "Bearer ordinary-session" },
      remoteAddress: "203.0.113.8",
    });
    expect(forbidden.statusCode).toBe(403);

    role = "owner";
    const accepted = await runtime.app.inject({
      method: "GET",
      url: "/public/v1/admin/billing-operations",
      headers: { authorization: "Bearer owner-session" },
      remoteAddress: "203.0.113.8",
    });
    expect(accepted.statusCode).toBe(200);
    expect(accepted.json()).toMatchObject({
      stablecoinIntents: { allocating: 0, awaitingPayment: 0, consumed: 0 },
      payoutDispatches: { uncertain: 0, submitted: 0, paid: 0 },
      settlementCandidates: [],
      sporeQuoteCandidates: [],
      alerts: [],
    });
  });
});

function seedAccount(database: MeshDatabase): void {
  database.raw.prepare(
    `INSERT INTO api_accounts(user_id, token_balance, usd_micros, created_at, updated_at)
     VALUES ('buyer-1', 0, 0, ?, ?)`,
  ).run(NOW, NOW);
}

function seedIntent(
  database: MeshDatabase,
  id: string,
  status: "allocating" | "awaiting_payment" | "consumed",
  updatedAt: number,
  expiresAt: number,
  consumedEventKey: string | null = null,
): void {
  database.raw.prepare(
    `INSERT INTO billing_stablecoin_intents(
       id, user_id, idempotency_key, request_digest, kind, amount_micros, currency,
       chain_id, asset, asset_atomic_amount, recipient, status, consumed_event_key,
       expires_at, created_at, updated_at, consumed_at
     ) VALUES (?, 'buyer-1', ?, ?, 'topup', 1000000, 'EUR', 'base', 'USDC',
               '1000000', 'recipient-secret', ?, ?, ?, ?, ?, ?)`,
  ).run(
    id, `key-${id}`, "a".repeat(64), status, consumedEventKey,
    expiresAt, updatedAt, updatedAt, status === "consumed" ? updatedAt : null,
  );
}

function seedPayout(
  database: MeshDatabase,
  id: string,
  status: "prepared" | "submitted",
  updatedAt: number,
  payoutMethod: "stable" | "spore" = "stable",
): void {
  if (payoutMethod === "spore") {
    database.raw.prepare(`
      INSERT OR IGNORE INTO seller_payout_destinations(
        id, seller_id, payout_method, destination_kind, destination_reference,
        destination_fingerprint, verifier_key_id, attestation_signature, status,
        verified_at, expires_at, created_at
      ) VALUES ('destination-spore-operations', 'seller-1', 'spore', 'wallet',
        'wallet-spore-operations', ?, 'verifier', 'signature', 'active', ?, ?, ?)
    `).run("a".repeat(64), updatedAt - 1_000, updatedAt + 86_400_000, updatedAt);
  }
  database.raw.prepare(
    `INSERT INTO payout_batches(
       id, seller_id, payout_method, idempotency_key, request_digest,
       gross_usd_micros, debt_offset_usd_micros, amount_usd_micros,
       destination_id, destination_reference, destination_fingerprint,
       status, created_at, updated_at
     ) VALUES (?, 'seller-1', ?, ?, ?, 200000, 0, 200000, ?, ?, ?, 'prepared', ?, ?)`,
  ).run(
    id, payoutMethod, `key-${id}`, "b".repeat(64),
    payoutMethod === "spore" ? "destination-spore-operations" : null,
    payoutMethod === "spore" ? "wallet-spore-operations" : null,
    payoutMethod === "spore" ? "a".repeat(64) : null,
    updatedAt, updatedAt,
  );
  database.raw.prepare(`
    INSERT INTO verified_work_receipts(
      receipt_id, seller_id, worker_id, job_id, stage_id, pricing_version,
      amount_usd_micros, evidence_digest, accepted_at, verifier_key_id,
      signature, recorded_at
    ) VALUES (?, 'seller-1', 'worker-operations', ?, 'stage-operations',
      'pricing-operations', 200000, ?, ?, 'verifier', 'signature', ?)
  `).run(`receipt-${id}`, `job-${id}`, "c".repeat(64), updatedAt, updatedAt);
  database.raw.prepare(`
    INSERT INTO seller_earnings(
      id, seller_id, receipt_id, amount_usd_micros, payout_method, status, created_at
    ) VALUES (?, 'seller-1', ?, 200000, ?, 'available', ?)
  `).run(`earning-${id}`, `receipt-${id}`, payoutMethod, updatedAt);
  database.raw.prepare(
    "INSERT INTO payout_batch_items(batch_id, earning_id, amount_usd_micros) VALUES (?, ?, 200000)",
  ).run(id, `earning-${id}`);
  database.raw.prepare(
    "UPDATE seller_earnings SET status = 'batched' WHERE id = ?",
  ).run(`earning-${id}`);
  if (status === "submitted") {
    database.raw.prepare(
      `UPDATE payout_batches
       SET status = 'submitted', dispatch_key = ?, external_reference = ?,
           submitted_at = ?, updated_at = ? WHERE id = ?`,
    ).run(`payout:${id}`, `transfer-${id}`, updatedAt, updatedAt, id);
  }
}

function seedSporeQuote(database: MeshDatabase, batchId: string, expiresAt: number): void {
  database.raw.prepare(
    `INSERT INTO spore_conversion_quotes(
       quote_id, batch_id, seller_id, usd_micros, chain_id, asset_id,
       token_decimals, token_atomic_amount, destination_fingerprint,
       issued_at, expires_at, oracle_key_id, signature, status, recorded_at
     ) VALUES (?, ?, 'seller-1', 200000, 'base', 'spore-1', 18, '1', ?,
       ?, ?, 'oracle', ?, 'active', ?)`,
  ).run(
    `quote-${batchId}`, batchId, "a".repeat(64), expiresAt - 60_000,
    expiresAt, "s".repeat(32), expiresAt - 60_000,
  );
}

function seedDispatch(
  database: MeshDatabase,
  batchId: string,
  state: "uncertain" | "submitted",
  updatedAt: number,
): void {
  database.raw.prepare(
    `INSERT INTO payout_dispatch_operations(
       id, batch_id, payout_method, dispatch_key, state, external_reference,
       created_at, updated_at
     ) VALUES (?, ?, 'stable', ?, ?, ?, ?, ?)`,
  ).run(
    `dispatch-${batchId}`, batchId, `payout:${batchId}`, state,
    state === "submitted" ? `transfer-${batchId}` : null, updatedAt, updatedAt,
  );
}
