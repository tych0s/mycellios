import { generateKeyPairSync, sign } from "node:crypto";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createCoordinator, type CoordinatorRuntime } from "../src/coordinator/server.js";
import type { SupabaseAuthService } from "../src/coordinator/supabase-auth.js";
import {
  sellerDestinationFingerprint,
  sellerDestinationSigningBytes,
  type SellerDestinationAttestation,
} from "../src/coordinator/seller-destinations.js";
import type { PayoutGateway } from "../src/coordinator/payout-dispatch.js";
import {
  PayoutSettlementVerifier,
  payoutSettlementSigningBytes,
  type PayoutSettlementAttestation,
} from "../src/coordinator/payout-settlement.js";

describe("account-bound seller finance routes", () => {
  const runtimes: CoordinatorRuntime[] = [];
  afterEach(async () => {
    await Promise.all(runtimes.splice(0).map((runtime) => runtime.close()));
  });

  async function setup() {
    const verifier = generateKeyPairSync("ed25519");
    let activeUser = "seller-one";
    let activeRole: "owner" | null = null;
    const auth = {
      authenticate: vi.fn(async () => ({ id: activeUser, email: null, role: activeRole })),
    } as unknown as SupabaseAuthService;
    const stableGateway: PayoutGateway = {
      createTransfer: vi.fn(async (input) => ({
        state: "submitted" as const,
        externalReference: `tr_${input.batchId.replace(/[^A-Za-z0-9]/g, "")}`,
      })),
      inspectTransfer: vi.fn(async () => ({ state: "absent" as const })),
    };
    const runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 0,
      databasePath: ":memory:",
      requestTimeoutMs: 1_000,
      apiAccessEnabled: true,
      apiStarterTokens: 0,
      modelAdminToken: "separate-admin-secret",
    }, {
      logger: false,
      supabaseAuthService: auth,
      sellerPayoutPolicy: {
        minimumUsdMicros: 100_000,
        spore: {
          legalApproved: false,
          custodyApproved: false,
          liquidityApproved: false,
          antifraudApproved: false,
        },
      },
      sellerDestinationVerifierKeys: new Map([[
        "destination-verifier",
        verifier.publicKey.export({ type: "spki", format: "pem" }).toString(),
      ]]),
      stablePayoutGateway: stableGateway,
      payoutSettlementVerifier: new PayoutSettlementVerifier(new Map([[
        "settlement-verifier",
        verifier.publicKey.export({ type: "spki", format: "pem" }).toString(),
      ]])),
    });
    runtimes.push(runtime);
    const asUser = (userId: string, role: "owner" | null = null) => {
      activeUser = userId;
      activeRole = role;
      return { authorization: `Bearer session-${userId}` };
    };
    const destinationAttestation = (sellerId: string): SellerDestinationAttestation => {
      const unsigned = {
        schema: "mycellios.seller-destination.v1" as const,
        destinationId: `destination-${sellerId}`,
        sellerId,
        payoutMethod: "stable" as const,
        destinationKind: "provider_account" as const,
        destinationReference: `provider-account-${sellerId}`,
        destinationFingerprint: sellerDestinationFingerprint("provider_account", `provider-account-${sellerId}`),
        verifiedAt: Date.now(),
        expiresAt: Date.now() + 86_400_000,
        verifierKeyId: "destination-verifier",
      };
      return {
        ...unsigned,
        signature: sign(null, sellerDestinationSigningBytes(unsigned), verifier.privateKey).toString("base64url"),
      };
    };
    const settlementAttestation = (
      batchId: string,
      externalReference: string,
    ): PayoutSettlementAttestation => {
      const unsigned = {
        schema: "mycellios.payout-settlement.v1" as const,
        batchId,
        externalReference,
        settlementReference: `po_${batchId.replace(/[^A-Za-z0-9]/g, "")}`,
        paidAt: Date.now() - 1_000,
        issuedAt: Date.now(),
        verifierKeyId: "settlement-verifier",
      };
      return {
        ...unsigned,
        signature: sign(null, payoutSettlementSigningBytes(unsigned), verifier.privateKey).toString("base64url"),
      };
    };
    return { runtime, asUser, destinationAttestation, settlementAttestation, stableGateway };
  }

  it("derives seller identity from the account session and hides other earnings", async () => {
    const { runtime, asUser } = await setup();
    seedEarning(runtime, "earning-one", "seller-one", 250_000);
    seedEarning(runtime, "earning-two", "seller-two", 350_000);

    const summary = await runtime.app.inject({
      method: "GET", url: "/v1/seller", headers: asUser("seller-one"),
    });
    expect(summary.statusCode).toBe(200);
    expect(summary.json()).toMatchObject({
      sellerId: "seller-one",
      payoutPreference: "stable",
      availableUsdMicros: 250_000,
    });
    const earnings = await runtime.app.inject({
      method: "GET", url: "/v1/seller/earnings", headers: asUser("seller-one"),
    });
    expect(earnings.json().data).toEqual([
      expect.objectContaining({ id: "earning-one", sellerId: "seller-one" }),
    ]);

    const apiKey = runtime.apiAccess.createKey("seller-one", "seller-finance-denied");
    const denied = await runtime.app.inject({
      method: "GET",
      url: "/v1/seller/earnings",
      headers: { authorization: `Bearer ${apiKey.secret}` },
    });
    expect(denied.statusCode).toBe(403);
    expect(denied.json().error.code).toBe("account_session_required");
  });

  it("prepares idempotently, isolates ownership and cancels only its own payout", async () => {
    const { runtime, asUser, destinationAttestation } = await setup();
    seedEarning(runtime, "earning-payout", "seller-one", 250_000);
    const destination = await runtime.app.inject({
      method: "PUT",
      url: "/v1/seller/payout-destinations",
      headers: asUser("seller-one"),
      payload: destinationAttestation("seller-one"),
    });
    expect(destination.statusCode).toBe(201);
    const crossAccountPrepare = await runtime.app.inject({
      method: "POST",
      url: "/v1/seller/payouts",
      headers: { ...asUser("seller-two"), "idempotency-key": "cross-account-payout" },
      payload: { payoutMethod: "stable", earningIds: ["earning-payout"] },
    });
    expect(crossAccountPrepare.statusCode).toBe(409);
    expect(crossAccountPrepare.json().error.code).toBe("earning_owner_conflict");
    const create = () => runtime.app.inject({
      method: "POST",
      url: "/v1/seller/payouts",
      headers: { ...asUser("seller-one"), "idempotency-key": "seller-payout-one" },
      payload: { payoutMethod: "stable", earningIds: ["earning-payout"] },
    });
    const created = await create();
    expect(created.statusCode).toBe(201);
    expect(created.json()).toMatchObject({
      duplicate: false,
      batch: { sellerId: "seller-one", payoutMethod: "stable", status: "prepared" },
    });
    expect(created.json().batch.destinationFingerprint).toBe(
      destinationAttestation("seller-one").destinationFingerprint,
    );
    const batchId = created.json().batch.id as string;
    const replay = await create();
    expect(replay.statusCode).toBe(200);
    expect(replay.json()).toMatchObject({ duplicate: true, batch: { id: batchId } });

    const hidden = await runtime.app.inject({
      method: "GET", url: `/v1/seller/payouts/${batchId}`, headers: asUser("seller-two"),
    });
    expect(hidden.statusCode).toBe(404);
    const forbiddenCancel = await runtime.app.inject({
      method: "DELETE", url: `/v1/seller/payouts/${batchId}`, headers: asUser("seller-two"),
    });
    expect(forbiddenCancel.statusCode).toBe(404);

    const cancelled = await runtime.app.inject({
      method: "DELETE", url: `/v1/seller/payouts/${batchId}`, headers: asUser("seller-one"),
    });
    expect(cancelled.statusCode).toBe(200);
    expect(cancelled.json()).toMatchObject({ batch: { status: "cancelled" } });
  });

  it("keeps SPORE preference closed until every gate is approved", async () => {
    const { runtime, asUser } = await setup();
    const response = await runtime.app.inject({
      method: "PUT",
      url: "/v1/seller/payout-preference",
      headers: asUser("seller-one"),
      payload: { method: "spore" },
    });
    expect(response.statusCode).toBe(409);
    expect(response.json().error.code).toBe("spore_payout_gates_closed");
    expect(runtime.sellerEarnings.getPayoutPreference("seller-one")).toBe("stable");
  });

  it("allows only an administrator to dispatch a prepared stable payout", async () => {
    const { runtime, asUser, destinationAttestation, settlementAttestation, stableGateway } = await setup();
    seedEarning(runtime, "earning-admin-dispatch", "seller-one", 250_000);
    await runtime.app.inject({
      method: "PUT", url: "/v1/seller/payout-destinations",
      headers: asUser("seller-one"), payload: destinationAttestation("seller-one"),
    });
    const prepared = await runtime.app.inject({
      method: "POST", url: "/v1/seller/payouts",
      headers: { ...asUser("seller-one"), "idempotency-key": "admin-dispatch-payout" },
      payload: { payoutMethod: "stable", earningIds: ["earning-admin-dispatch"] },
    });
    const batchId = prepared.json().batch.id as string;

    const forbidden = await runtime.app.inject({
      method: "POST",
      url: `/public/v1/admin/seller-payouts/${batchId}/dispatch`,
      headers: asUser("seller-one"),
      remoteAddress: "203.0.113.20",
    });
    expect(forbidden.statusCode).toBe(403);
    const dispatched = await runtime.app.inject({
      method: "POST",
      url: `/public/v1/admin/seller-payouts/${batchId}/dispatch`,
      headers: asUser("network-owner", "owner"),
      remoteAddress: "203.0.113.20",
    });
    expect(dispatched.statusCode).toBe(202);
    expect(dispatched.json()).toMatchObject({
      operation: { state: "submitted" }, batch: { status: "submitted" },
    });
    expect(stableGateway.createTransfer).toHaveBeenCalledTimes(1);
    expect(JSON.stringify(dispatched.json())).not.toContain("provider-account-seller-one");

    const dispatchAudit = await runtime.app.inject({
      method: "GET",
      url: `/public/v1/admin/seller-payouts/${batchId}/dispatch`,
      headers: asUser("network-owner", "owner"),
      remoteAddress: "203.0.113.20",
    });
    expect(dispatchAudit.statusCode).toBe(200);
    expect(dispatchAudit.json()).toMatchObject({
      object: "payout_dispatch",
      operation: {
        batchId, state: "submitted",
        sporeQuoteId: null, sporeQuoteAttestationDigest: null,
      },
    });
    const forbiddenDispatchAudit = await runtime.app.inject({
      method: "GET",
      url: `/public/v1/admin/seller-payouts/${batchId}/dispatch`,
      headers: asUser("seller-one"),
      remoteAddress: "203.0.113.20",
    });
    expect(forbiddenDispatchAudit.statusCode).toBe(403);

    const settled = await runtime.app.inject({
      method: "POST",
      url: `/public/v1/admin/seller-payouts/${batchId}/settlement`,
      headers: asUser("network-owner", "owner"),
      remoteAddress: "203.0.113.20",
      payload: settlementAttestation(batchId, dispatched.json().operation.externalReference),
    });
    expect(settled.statusCode).toBe(202);
    expect(settled.json()).toMatchObject({
      operation: { state: "paid" }, batch: { status: "paid" },
    });
    const evidence = await runtime.app.inject({
      method: "GET",
      url: `/public/v1/admin/seller-payouts/${batchId}/settlement`,
      headers: asUser("network-owner", "owner"),
      remoteAddress: "203.0.113.20",
    });
    expect(evidence.statusCode).toBe(200);
    expect(evidence.json()).toMatchObject({
      object: "payout_settlement_evidence",
      evidence: {
        batchId,
        verifierKeyId: "settlement-verifier",
        signatureDigest: expect.stringMatching(/^[a-f0-9]{64}$/),
      },
    });
    expect(JSON.stringify(evidence.json())).not.toContain("signature\"");

    const forbiddenEvidence = await runtime.app.inject({
      method: "GET",
      url: `/public/v1/admin/seller-payouts/${batchId}/settlement`,
      headers: asUser("seller-one"),
      remoteAddress: "203.0.113.20",
    });
    expect(forbiddenEvidence.statusCode).toBe(403);
  });
});

function seedEarning(
  runtime: CoordinatorRuntime,
  id: string,
  sellerId: string,
  amountUsdMicros: number,
): void {
  const now = Date.now();
  runtime.database.raw.prepare(
    `INSERT INTO verified_work_receipts(
       receipt_id, seller_id, worker_id, job_id, stage_id, pricing_version,
       amount_usd_micros, evidence_digest, accepted_at, verifier_key_id, signature, recorded_at
     ) VALUES (?, ?, ?, ?, ?, 'pricing-1', ?, ?, ?, 'verifier-1', 'fixture-signature', ?)`,
  ).run(
    `receipt-${id}`, sellerId, `worker-${id}`, `job-${id}`, `stage-${id}`,
    amountUsdMicros, "a".repeat(64), now, now,
  );
  runtime.database.raw.prepare(
    `INSERT INTO seller_earnings(
       id, seller_id, receipt_id, amount_usd_micros, payout_method, status, created_at
     ) VALUES (?, ?, ?, ?, 'stable', 'available', ?)`,
  ).run(id, sellerId, `receipt-${id}`, amountUsdMicros, now);
}
