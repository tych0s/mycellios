import { createHash, createPublicKey, verify } from "node:crypto";
import type { MeshDatabase } from "../storage/database.js";

export interface SporeConversionQuote {
  schema: "mycellios.spore-conversion-quote.v1";
  quoteId: string;
  batchId: string;
  sellerId: string;
  usdMicros: number;
  chainId: string;
  assetId: string;
  tokenDecimals: number;
  tokenAtomicAmount: string;
  destinationFingerprint: string;
  issuedAt: number;
  expiresAt: number;
  oracleKeyId: string;
  signature: string;
}

export interface StoredSporeConversionQuote extends SporeConversionQuote {
  status: "active" | "superseded";
  recordedAt: number;
  replacedAt: number | null;
}

export class SporeConversionError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "SporeConversionError";
  }
}

export class SporeConversionQuoteVerifier {
  constructor(private readonly config: {
    trustedOracleKeys: ReadonlyMap<string, string>;
    approvedAssets: ReadonlyMap<string, { tokenDecimals: number }>;
    maxQuoteAgeMs: number;
    now?: () => number;
  }) {
    if (!Number.isSafeInteger(config.maxQuoteAgeMs) || config.maxQuoteAgeMs <= 0) {
      throw new Error("invalid_spore_quote_max_age");
    }
  }

  verify(quote: SporeConversionQuote): SporeConversionQuote {
    const now = this.config.now?.() ?? Date.now();
    validateQuote(quote, now, this.config.maxQuoteAgeMs);
    const asset = this.config.approvedAssets.get(`${quote.chainId}:${quote.assetId}`);
    if (!asset || asset.tokenDecimals !== quote.tokenDecimals) {
      throw new SporeConversionError("spore_asset_not_approved", "The SPORE asset is not approved.");
    }
    const publicKey = this.config.trustedOracleKeys.get(quote.oracleKeyId);
    if (!publicKey) throw new SporeConversionError("untrusted_spore_oracle", "The SPORE quote oracle is not trusted.");
    let valid = false;
    try {
      valid = verify(null, sporeConversionSigningBytes(quote), createPublicKey(publicKey), Buffer.from(quote.signature, "base64url"));
    } catch { valid = false; }
    if (!valid) throw new SporeConversionError("invalid_spore_quote_signature", "The SPORE quote signature is invalid.");
    return { ...quote };
  }
}

export class SporeConversionQuoteStore {
  constructor(
    private readonly database: MeshDatabase,
    private readonly verifier: SporeConversionQuoteVerifier,
    private readonly now: () => number = Date.now,
  ) {}

  record(input: SporeConversionQuote): { quote: SporeConversionQuote; duplicate: boolean } {
    const quote = this.verifier.verify(input);
    return this.database.transaction(() => {
      const batch = this.database.raw.prepare(
        `SELECT seller_id, payout_method, amount_usd_micros, destination_fingerprint, status
         FROM payout_batches WHERE id = ?`,
      ).get(quote.batchId) as {
        seller_id: string; payout_method: string; amount_usd_micros: number;
        destination_fingerprint: string | null; status: string;
      } | undefined;
      if (!batch) throw new SporeConversionError("spore_quote_batch_not_found", "The payout batch does not exist.");
      if (batch.payout_method !== "spore" || batch.status !== "prepared"
        || batch.seller_id !== quote.sellerId
        || Number(batch.amount_usd_micros) !== quote.usdMicros
        || batch.destination_fingerprint !== quote.destinationFingerprint) {
        throw new SporeConversionError("spore_quote_batch_mismatch", "The SPORE quote does not match the prepared payout.");
      }
      const existing = this.getByBatchId(quote.batchId);
      if (existing) {
        if (JSON.stringify(existing) !== JSON.stringify(quote)) {
          if (existing.expiresAt > this.now()) {
            throw new SporeConversionError("spore_quote_replay_conflict", "The payout already has a live SPORE quote.");
          }
          this.database.raw.prepare(
            "UPDATE spore_conversion_quotes SET status = 'superseded', replaced_at = ? WHERE quote_id = ? AND status = 'active'",
          ).run(this.now(), existing.quoteId);
        } else {
          return { quote: existing, duplicate: true };
        }
      }
      try {
        this.database.raw.prepare(
          `INSERT INTO spore_conversion_quotes(
             quote_id, batch_id, seller_id, usd_micros, chain_id, asset_id,
             token_decimals, token_atomic_amount, destination_fingerprint,
             issued_at, expires_at, oracle_key_id, signature, status, recorded_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)`,
        ).run(
          quote.quoteId, quote.batchId, quote.sellerId, quote.usdMicros,
          quote.chainId, quote.assetId, quote.tokenDecimals, quote.tokenAtomicAmount,
          quote.destinationFingerprint, quote.issuedAt, quote.expiresAt,
          quote.oracleKeyId, quote.signature, this.now(),
        );
      } catch {
        throw new SporeConversionError("spore_quote_replay_conflict", "The SPORE quote identity is already used.");
      }
      return { quote, duplicate: false };
    });
  }

  getByBatchId(batchId: string): SporeConversionQuote | null {
    const row = this.database.raw.prepare(
      `SELECT quote_id, batch_id, seller_id, usd_micros, chain_id, asset_id,
              token_decimals, token_atomic_amount, destination_fingerprint,
              issued_at, expires_at, oracle_key_id, signature
       FROM spore_conversion_quotes WHERE batch_id = ? AND status = 'active'`,
    ).get(batchId) as Record<string, unknown> | undefined;
    return row ? {
      schema: "mycellios.spore-conversion-quote.v1",
      quoteId: String(row.quote_id), batchId: String(row.batch_id), sellerId: String(row.seller_id),
      usdMicros: Number(row.usd_micros), chainId: String(row.chain_id), assetId: String(row.asset_id),
      tokenDecimals: Number(row.token_decimals), tokenAtomicAmount: String(row.token_atomic_amount),
      destinationFingerprint: String(row.destination_fingerprint), issuedAt: Number(row.issued_at),
      expiresAt: Number(row.expires_at), oracleKeyId: String(row.oracle_key_id), signature: String(row.signature),
    } : null;
  }

  getVerifiedByBatchId(batchId: string): SporeConversionQuote | null {
    const quote = this.getByBatchId(batchId);
    return quote ? this.verifier.verify(quote) : null;
  }

  getStoredByQuoteId(quoteId: string): StoredSporeConversionQuote | null {
    const row = this.database.raw.prepare(
      `SELECT quote_id, batch_id, seller_id, usd_micros, chain_id, asset_id,
              token_decimals, token_atomic_amount, destination_fingerprint,
              issued_at, expires_at, oracle_key_id, signature, status,
              recorded_at, replaced_at
       FROM spore_conversion_quotes WHERE quote_id = ?`,
    ).get(quoteId) as Record<string, unknown> | undefined;
    return row ? mapStoredQuote(row) : null;
  }

  listHistoryByBatchId(
    batchId: string,
    limit = 50,
    after?: { recordedAt: number; quoteId: string },
  ): StoredSporeConversionQuote[] {
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > 101) {
      throw new SporeConversionError("invalid_spore_quote_history_limit", "Invalid SPORE quote history limit.");
    }
    if (after && (!Number.isSafeInteger(after.recordedAt) || after.recordedAt < 0
      || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$/.test(after.quoteId))) {
      throw new SporeConversionError("invalid_spore_quote_history_cursor", "Invalid SPORE quote history cursor.");
    }
    const rows = this.database.raw.prepare(
      `SELECT quote_id, batch_id, seller_id, usd_micros, chain_id, asset_id,
              token_decimals, token_atomic_amount, destination_fingerprint,
              issued_at, expires_at, oracle_key_id, signature, status,
              recorded_at, replaced_at
       FROM spore_conversion_quotes
       WHERE batch_id = ?
         AND (? IS NULL OR recorded_at > ? OR (recorded_at = ? AND quote_id > ?))
       ORDER BY recorded_at ASC, quote_id ASC
       LIMIT ?`,
    ).all(
      batchId,
      after?.recordedAt ?? null,
      after?.recordedAt ?? null,
      after?.recordedAt ?? null,
      after?.quoteId ?? "",
      limit,
    ) as Record<string, unknown>[];
    return rows.map(mapStoredQuote);
  }
}

function mapStoredQuote(row: Record<string, unknown>): StoredSporeConversionQuote {
  return {
    schema: "mycellios.spore-conversion-quote.v1",
    quoteId: String(row.quote_id), batchId: String(row.batch_id), sellerId: String(row.seller_id),
    usdMicros: Number(row.usd_micros), chainId: String(row.chain_id), assetId: String(row.asset_id),
    tokenDecimals: Number(row.token_decimals), tokenAtomicAmount: String(row.token_atomic_amount),
    destinationFingerprint: String(row.destination_fingerprint), issuedAt: Number(row.issued_at),
    expiresAt: Number(row.expires_at), oracleKeyId: String(row.oracle_key_id), signature: String(row.signature),
    status: row.status === "active" ? "active" : "superseded",
    recordedAt: Number(row.recorded_at), replacedAt: row.replaced_at === null ? null : Number(row.replaced_at),
  };
}

export function sporeConversionSigningBytes(quote: Omit<SporeConversionQuote, "signature">): Buffer {
  return Buffer.from(JSON.stringify([
    quote.schema, quote.quoteId, quote.batchId, quote.sellerId, quote.usdMicros,
    quote.chainId, quote.assetId, quote.tokenDecimals, quote.tokenAtomicAmount,
    quote.destinationFingerprint, quote.issuedAt, quote.expiresAt, quote.oracleKeyId,
  ]), "utf8");
}

export function sporeConversionAttestationDigest(quote: SporeConversionQuote): string {
  return createHash("sha256")
    .update(sporeConversionSigningBytes(quote))
    .update(".", "utf8")
    .update(quote.signature, "utf8")
    .digest("hex");
}

function validateQuote(quote: SporeConversionQuote, now: number, maxAgeMs: number): void {
  if (quote.schema !== "mycellios.spore-conversion-quote.v1") throw new SporeConversionError("unsupported_spore_quote_schema", "Unsupported SPORE quote schema.");
  for (const value of [quote.quoteId, quote.batchId, quote.sellerId, quote.chainId, quote.assetId, quote.oracleKeyId]) {
    if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$/.test(value)) throw new SporeConversionError("invalid_spore_quote_identity", "Invalid SPORE quote identity.");
  }
  if (!Number.isSafeInteger(quote.usdMicros) || quote.usdMicros <= 0
    || !Number.isSafeInteger(quote.tokenDecimals) || quote.tokenDecimals < 0 || quote.tokenDecimals > 30
    || !/^[1-9][0-9]{0,77}$/.test(quote.tokenAtomicAmount)
    || !/^[a-f0-9]{64}$/.test(quote.destinationFingerprint)) {
    throw new SporeConversionError("invalid_spore_quote_value", "Invalid SPORE quote value.");
  }
  if (!Number.isSafeInteger(quote.issuedAt) || !Number.isSafeInteger(quote.expiresAt)
    || quote.issuedAt > now + 60_000 || quote.issuedAt < now - maxAgeMs
    || quote.expiresAt <= now || quote.expiresAt <= quote.issuedAt) {
    throw new SporeConversionError("invalid_spore_quote_validity", "The SPORE quote is stale or expired.");
  }
  if (!/^[A-Za-z0-9_-]{16,1024}$/.test(quote.signature)) throw new SporeConversionError("invalid_spore_quote_signature", "Invalid SPORE quote signature encoding.");
}
