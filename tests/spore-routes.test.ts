import { generateKeyPairSync, sign } from "node:crypto";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createCoordinator, type CoordinatorRuntime } from "../src/coordinator/server.js";
import { sporeConversionSigningBytes } from "../src/coordinator/spore-conversion.js";
import type { SupabaseAuthService } from "../src/coordinator/supabase-auth.js";
import type { PayoutGateway } from "../src/coordinator/payout-dispatch.js";

describe("SPORE quote administrative routes", () => {
  const runtimes: CoordinatorRuntime[] = [];
  afterEach(async () => Promise.all(runtimes.splice(0).map((runtime) => runtime.close())));

  it("stores a signed quote for owner/admin and returns only redacted metadata", async () => {
    const keys = generateKeyPairSync("ed25519");
    const publicKey = keys.publicKey.export({ type: "spki", format: "pem" }).toString();
    let role: "owner" | null = null;
    let userId = "operator";
    const auth = { authenticate: vi.fn(async () => ({ id: userId, email: null, role })) } as unknown as SupabaseAuthService;
    const sporeGateway: PayoutGateway = {
      createTransfer: vi.fn(async () => ({ state: "submitted" as const, externalReference: "spore-transfer" })),
      inspectTransfer: vi.fn(async () => ({ state: "absent" as const })),
    };
    const runtime = await createCoordinator({
      host: "127.0.0.1", port: 0, databasePath: ":memory:", requestTimeoutMs: 1_000,
      apiAccessEnabled: true, apiStarterTokens: 0,
      sellerPayout: {
        minimumUsdMicros: 100_000,
        spore: { legalApproved: true, custodyApproved: true, liquidityApproved: true, antifraudApproved: true },
        destinationVerifierKeys: [{ keyId: "destination", publicKey }],
        settlementVerifierKeys: [{ keyId: "settlement", publicKey }],
        settlementEvidenceMaxAgeMs: 900_000,
        sporeConversion: {
          trustedOracleKeys: [{ keyId: "oracle", publicKey }],
          approvedAssets: [{ chainId: "base", assetId: "spore-1", tokenDecimals: 18 }],
          maxQuoteAgeMs: 300_000,
        },
      },
    }, { logger: false, supabaseAuthService: auth, sporePayoutGateway: sporeGateway });
    runtimes.push(runtime);
    expect(runtime.payoutDispatch).not.toBeNull();
    const now = Date.now();
    runtime.database.raw.prepare(`
      INSERT INTO seller_payout_destinations (
        id, seller_id, payout_method, destination_kind, destination_reference,
        destination_fingerprint, verifier_key_id, attestation_signature, status,
        verified_at, expires_at, created_at
      ) VALUES ('destination-spore', 'seller-1', 'spore', 'wallet', 'wallet-spore',
        ?, 'destination', 'signature', 'active', ?, ?, ?)
    `).run("a".repeat(64), now - 1_000, now + 86_400_000, now);
    runtime.database.raw.prepare(
      `INSERT INTO payout_batches(id, seller_id, payout_method, idempotency_key, request_digest,
       gross_usd_micros, debt_offset_usd_micros, amount_usd_micros,
       destination_id, destination_reference, destination_fingerprint,
       status, created_at, updated_at) VALUES ('batch-spore', 'seller-1', 'spore', 'key', ?,
       1000000, 0, 1000000, 'destination-spore', 'wallet-spore', ?, 'prepared', ?, ?)`,
    ).run("b".repeat(64), "a".repeat(64), now, now);
    const unsigned = {
      schema: "mycellios.spore-conversion-quote.v1" as const, quoteId: "quote-1",
      batchId: "batch-spore", sellerId: "seller-1", usdMicros: 1_000_000,
      chainId: "base", assetId: "spore-1", tokenDecimals: 18,
      tokenAtomicAmount: "1000000000000000000", destinationFingerprint: "a".repeat(64),
      issuedAt: now, expiresAt: now + 60_000, oracleKeyId: "oracle",
    };
    const payload = { ...unsigned, signature: sign(null, sporeConversionSigningBytes(unsigned), keys.privateKey).toString("base64url") };
    const forbidden = await runtime.app.inject({ method: "POST", url: "/public/v1/admin/seller-payouts/batch-spore/spore-quote", payload, headers: { authorization: "Bearer user" }, remoteAddress: "203.0.113.1" });
    expect(forbidden.statusCode).toBe(403);
    role = "owner";
    const created = await runtime.app.inject({ method: "POST", url: "/public/v1/admin/seller-payouts/batch-spore/spore-quote", payload, headers: { authorization: "Bearer owner" }, remoteAddress: "203.0.113.1" });
    expect(created.statusCode).toBe(201);
    expect(created.json().quote.signatureDigest).toMatch(/^[a-f0-9]{64}$/);
    expect(JSON.stringify(created.json())).not.toContain("seller-1");
    expect(JSON.stringify(created.json())).not.toContain(payload.signature);

    const history = await runtime.app.inject({
      method: "GET", url: "/public/v1/admin/seller-payouts/batch-spore/spore-quote-history",
      headers: { authorization: "Bearer owner" }, remoteAddress: "203.0.113.1",
    });
    expect(history.statusCode).toBe(200);
    expect(history.json()).toMatchObject({
      object: "spore_conversion_quote_history",
      hasMore: false,
      nextCursor: null,
      quotes: [{ quoteId: "quote-1", status: "active", replacedAt: null }],
    });
    expect(history.json().quotes[0].recordedAt).toEqual(expect.any(Number));
    expect(history.json().quotes[0].signatureDigest).toMatch(/^[a-f0-9]{64}$/);
    expect(JSON.stringify(history.json())).not.toContain("seller-1");
    expect(JSON.stringify(history.json())).not.toContain(payload.signature);

    runtime.database.transaction(() => {
      runtime.database.raw.prepare(
        "UPDATE spore_conversion_quotes SET status = 'superseded', replaced_at = ? WHERE quote_id = 'quote-1'",
      ).run(now + 1);
      runtime.database.raw.prepare(
        `INSERT INTO spore_conversion_quotes(
           quote_id, batch_id, seller_id, usd_micros, chain_id, asset_id,
           token_decimals, token_atomic_amount, destination_fingerprint,
           issued_at, expires_at, oracle_key_id, signature, status, recorded_at)
         SELECT 'quote-2', batch_id, seller_id, usd_micros, chain_id, asset_id,
           token_decimals, token_atomic_amount, destination_fingerprint,
           issued_at, expires_at, oracle_key_id, signature, 'active', recorded_at + 1
         FROM spore_conversion_quotes WHERE quote_id = 'quote-1'`,
      ).run();
    });

    const bounded = await runtime.app.inject({
      method: "GET", url: "/public/v1/admin/seller-payouts/batch-spore/spore-quote-history?limit=1",
      headers: { authorization: "Bearer owner" }, remoteAddress: "203.0.113.1",
    });
    expect(bounded.statusCode).toBe(200);
    expect(bounded.json()).toMatchObject({
      hasMore: true,
      quotes: [{ quoteId: "quote-1", status: "superseded" }],
    });
    const cursor = bounded.json().nextCursor as { recordedAt: number; quoteId: string };
    expect(cursor).toMatchObject({ quoteId: "quote-1" });
    const nextPage = await runtime.app.inject({
      method: "GET",
      url: `/public/v1/admin/seller-payouts/batch-spore/spore-quote-history?limit=1&afterRecordedAt=${cursor.recordedAt}&afterQuoteId=${cursor.quoteId}`,
      headers: { authorization: "Bearer owner" }, remoteAddress: "203.0.113.1",
    });
    expect(nextPage.statusCode).toBe(200);
    expect(nextPage.json()).toMatchObject({
      hasMore: false,
      nextCursor: null,
      quotes: [{ quoteId: "quote-2", status: "active" }],
    });

    role = null;
    userId = "seller-1";
    const sellerView = await runtime.app.inject({
      method: "GET", url: "/v1/seller/payouts/batch-spore/spore-quote",
      headers: { authorization: "Bearer session-seller-1" },
    });
    expect(sellerView.statusCode).toBe(200);
    expect(sellerView.json().quote).toMatchObject({
      tokenAtomicAmount: "1000000000000000000", assetId: "spore-1", expired: false,
    });
    userId = "seller-other";
    const hidden = await runtime.app.inject({
      method: "GET", url: "/v1/seller/payouts/batch-spore/spore-quote",
      headers: { authorization: "Bearer session-seller-other" },
    });
    expect(hidden.statusCode).toBe(404);
  });

  it("protects quote history and returns 404 when the batch has no quotes", async () => {
    const keys = generateKeyPairSync("ed25519");
    const publicKey = keys.publicKey.export({ type: "spki", format: "pem" }).toString();
    let role: "owner" | null = null;
    const auth = { authenticate: vi.fn(async () => ({ id: "operator", email: null, role })) } as unknown as SupabaseAuthService;
    const runtime = await createCoordinator({
      host: "127.0.0.1", port: 0, databasePath: ":memory:", requestTimeoutMs: 1_000,
      apiAccessEnabled: true, apiStarterTokens: 0,
      sellerPayout: {
        minimumUsdMicros: 100_000,
        spore: { legalApproved: true, custodyApproved: true, liquidityApproved: true, antifraudApproved: true },
        destinationVerifierKeys: [{ keyId: "destination", publicKey }],
        settlementVerifierKeys: [{ keyId: "settlement", publicKey }],
        settlementEvidenceMaxAgeMs: 900_000,
        sporeConversion: {
          trustedOracleKeys: [{ keyId: "oracle", publicKey }],
          approvedAssets: [{ chainId: "base", assetId: "spore-1", tokenDecimals: 18 }],
          maxQuoteAgeMs: 300_000,
        },
      },
    }, { logger: false, supabaseAuthService: auth });
    runtimes.push(runtime);
    const forbidden = await runtime.app.inject({
      method: "GET", url: "/public/v1/admin/seller-payouts/unknown/spore-quote-history",
      headers: { authorization: "Bearer user" }, remoteAddress: "203.0.113.1",
    });
    expect(forbidden.statusCode).toBe(403);
    role = "owner";
    const missing = await runtime.app.inject({
      method: "GET", url: "/public/v1/admin/seller-payouts/unknown/spore-quote-history",
      headers: { authorization: "Bearer owner" }, remoteAddress: "203.0.113.1",
    });
    expect(missing.statusCode).toBe(404);
  });
});
