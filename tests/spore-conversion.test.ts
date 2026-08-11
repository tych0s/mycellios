import { generateKeyPairSync, sign } from "node:crypto";
import { describe, expect, it } from "vitest";
import {
  SporeConversionQuoteStore,
  SporeConversionQuoteVerifier,
  sporeConversionSigningBytes,
  type SporeConversionQuote,
} from "../src/coordinator/spore-conversion.js";
import { MeshDatabase } from "../src/storage/database.js";
import { PayoutManager } from "../src/coordinator/payouts.js";

describe("SPORE conversion quote boundary", () => {
  const now = 1_800_000_000_000;
  const keys = generateKeyPairSync("ed25519");
  const verifier = new SporeConversionQuoteVerifier({
    trustedOracleKeys: new Map([["oracle-1", keys.publicKey.export({ type: "spki", format: "pem" }).toString()]]),
    approvedAssets: new Map([["base:spore-contract-1", { tokenDecimals: 18 }]]),
    maxQuoteAgeMs: 300_000,
    now: () => now,
  });

  function quote(overrides: Partial<SporeConversionQuote> = {}): SporeConversionQuote {
    const unsigned = {
      schema: "mycellios.spore-conversion-quote.v1" as const,
      quoteId: "spore-quote-1", batchId: "payout-batch-1", sellerId: "seller-1",
      usdMicros: 10_000_000, chainId: "base", assetId: "spore-contract-1",
      tokenDecimals: 18, tokenAtomicAmount: "25000000000000000000",
      destinationFingerprint: "a".repeat(64), issuedAt: now - 1_000,
      expiresAt: now + 60_000, oracleKeyId: "oracle-1", ...overrides,
    };
    return { ...unsigned, signature: sign(null, sporeConversionSigningBytes(unsigned), keys.privateKey).toString("base64url") };
  }

  function seedDestination(database: MeshDatabase): void {
    database.raw.prepare(`
      INSERT INTO seller_payout_destinations (
        id, seller_id, payout_method, destination_kind, destination_reference,
        destination_fingerprint, verifier_key_id, attestation_signature, status,
        verified_at, expires_at, created_at
      ) VALUES ('destination-spore', 'seller-1', 'spore', 'wallet',
        'wallet-spore', ?, 'destination-verifier', 'signature', 'active', ?, ?, ?)
    `).run("a".repeat(64), now - 1_000, now + 86_400_000, now);
  }

  it("accepts an approved, fresh and signed quote", () => {
    expect(verifier.verify(quote())).toMatchObject({ usdMicros: 10_000_000, tokenDecimals: 18 });
  });

  it("rejects tampering and unapproved assets", () => {
    const tampered = quote();
    tampered.usdMicros += 1;
    expect(() => verifier.verify(tampered)).toThrowError(expect.objectContaining({ code: "invalid_spore_quote_signature" }));
    expect(() => verifier.verify(quote({ chainId: "unknown" }))).toThrowError(expect.objectContaining({ code: "spore_asset_not_approved" }));
  });

  it("rejects expired quotes and non-canonical atomic amounts", () => {
    expect(() => verifier.verify(quote({ expiresAt: now }))).toThrowError(expect.objectContaining({ code: "invalid_spore_quote_validity" }));
    expect(() => verifier.verify(quote({ tokenAtomicAmount: "025" }))).toThrowError(expect.objectContaining({ code: "invalid_spore_quote_value" }));
  });

  it("persists only a quote matching one prepared SPORE batch", () => {
    const database = new MeshDatabase(":memory:");
    seedDestination(database);
    database.raw.prepare(
      `INSERT INTO payout_batches(
         id, seller_id, payout_method, idempotency_key, request_digest,
         gross_usd_micros, debt_offset_usd_micros, amount_usd_micros,
         destination_id, destination_reference, destination_fingerprint, status, created_at, updated_at
       ) VALUES ('payout-batch-1', 'seller-1', 'spore', 'key-1', ?,
                 10000000, 0, 10000000, 'destination-spore', 'wallet-spore', ?, 'prepared', ?, ?)`,
    ).run("b".repeat(64), "a".repeat(64), now, now);
    const store = new SporeConversionQuoteStore(database, verifier, () => now);
    const first = store.record(quote());
    const replay = store.record(quote());
    expect(first.duplicate).toBe(false);
    expect(replay.duplicate).toBe(true);
    expect(store.getByBatchId("payout-batch-1")).toMatchObject({ quoteId: "spore-quote-1" });
    expect(() => store.record(quote({ usdMicros: 9_000_000 })))
      .toThrowError(expect.objectContaining({ code: "spore_quote_batch_mismatch" }));
    expect(() => database.raw.prepare(`
      INSERT INTO spore_conversion_quotes(
        quote_id, batch_id, seller_id, usd_micros, chain_id, asset_id,
        token_decimals, token_atomic_amount, destination_fingerprint,
        issued_at, expires_at, oracle_key_id, signature, status, recorded_at
      ) VALUES ('quote-lateral', 'payout-batch-1', 'seller-other', 10000000,
        'base', 'spore-contract-1', 18, '1', ?, ?, ?, 'oracle-1',
        'signature', 'superseded', ?)
    `).run("a".repeat(64), now - 1_000, now + 60_000, now))
      .toThrow(/invalid_spore_conversion_quote_batch_binding/);
    database.close();
  });

  it("supersedes only an expired quote and retains its audit history", () => {
    let clock = now;
    const timedVerifier = new SporeConversionQuoteVerifier({
      trustedOracleKeys: new Map([["oracle-1", keys.publicKey.export({ type: "spki", format: "pem" }).toString()]]),
      approvedAssets: new Map([["base:spore-contract-1", { tokenDecimals: 18 }]]),
      maxQuoteAgeMs: 300_000, now: () => clock,
    });
    const database = new MeshDatabase(":memory:");
    seedDestination(database);
    database.raw.prepare(
      `INSERT INTO payout_batches(id, seller_id, payout_method, idempotency_key, request_digest,
       gross_usd_micros, debt_offset_usd_micros, amount_usd_micros,
       destination_id, destination_reference, destination_fingerprint,
       status, created_at, updated_at) VALUES ('payout-batch-1', 'seller-1', 'spore', 'key', ?,
       10000000, 0, 10000000, 'destination-spore', 'wallet-spore', ?, 'prepared', ?, ?)`,
    ).run("b".repeat(64), "a".repeat(64), now, now);
    const store = new SporeConversionQuoteStore(database, timedVerifier, () => clock);
    const first = quote({ expiresAt: clock + 100 });
    store.record(first);
    expect(() => store.record(quote({ quoteId: "spore-quote-live", expiresAt: clock + 200 })))
      .toThrowError(expect.objectContaining({ code: "spore_quote_replay_conflict" }));
    clock += 101;
    const replacement = quote({ quoteId: "spore-quote-2", issuedAt: clock, expiresAt: clock + 60_000 });
    store.record(replacement);
    expect(store.getByBatchId("payout-batch-1")?.quoteId).toBe("spore-quote-2");
    expect(database.raw.prepare(
      "SELECT quote_id, status FROM spore_conversion_quotes WHERE batch_id = ? ORDER BY recorded_at, quote_id",
    ).all("payout-batch-1")).toEqual(expect.arrayContaining([
      { quote_id: "spore-quote-1", status: "superseded" },
      { quote_id: "spore-quote-2", status: "active" },
    ]));
    database.close();
  });

  it("retires the active quote atomically when its prepared payout is cancelled", () => {
    const database = new MeshDatabase(":memory:");
    const batchNow = Date.now();
    seedDestination(database);
    database.raw.prepare(
      `INSERT INTO payout_batches(id, seller_id, payout_method, idempotency_key, request_digest,
       gross_usd_micros, debt_offset_usd_micros, amount_usd_micros,
       destination_id, destination_reference, destination_fingerprint,
       status, created_at, updated_at) VALUES ('payout-batch-1', 'seller-1', 'spore', 'key', ?,
       10000000, 0, 10000000, 'destination-spore', 'wallet-spore', ?, 'prepared', ?, ?)`,
    ).run("b".repeat(64), "a".repeat(64), batchNow, batchNow);
    const store = new SporeConversionQuoteStore(database, verifier, () => now);
    store.record(quote());
    new PayoutManager(database, {
      minimumUsdMicros: 1,
      spore: { legalApproved: true, custodyApproved: true, liquidityApproved: true, antifraudApproved: true },
    }).cancelPrepared("payout-batch-1");
    expect(store.getByBatchId("payout-batch-1")).toBeNull();
    expect(database.raw.prepare(
      "SELECT status FROM spore_conversion_quotes WHERE batch_id = 'payout-batch-1'",
    ).get()).toEqual({ status: "superseded" });
    database.close();
  });
});
