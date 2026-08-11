import { createHash } from "node:crypto";
import { newId } from "../core/ids.js";
import type { MeshDatabase } from "../storage/database.js";

export type PayoutMethod = "stable" | "spore";
export type PayoutStatus = "prepared" | "submitted" | "paid" | "cancelled";

export interface SporePayoutGates {
  legalApproved: boolean;
  custodyApproved: boolean;
  liquidityApproved: boolean;
  antifraudApproved: boolean;
}

export interface PayoutPolicy {
  minimumUsdMicros: number;
  spore: SporePayoutGates;
  requireVerifiedDestination?: boolean;
}

export interface PayoutBatch {
  id: string;
  sellerId: string;
  payoutMethod: PayoutMethod;
  idempotencyKey: string;
  grossUsdMicros: number;
  debtOffsetUsdMicros: number;
  amountUsdMicros: number;
  destinationId: string | null;
  destinationReference: string | null;
  destinationFingerprint: string | null;
  status: PayoutStatus;
  dispatchKey: string | null;
  externalReference: string | null;
  settlementReference: string | null;
  earningIds: string[];
  createdAt: number;
  updatedAt: number;
  submittedAt: number | null;
  paidAt: number | null;
  cancelledAt: number | null;
}

export interface PayoutMutationResult {
  batch: PayoutBatch;
  duplicate: boolean;
}

export class PayoutError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "PayoutError";
  }
}

export class PayoutManager {
  constructor(
    private readonly database: MeshDatabase,
    private readonly policy: PayoutPolicy,
  ) {
    if (!Number.isSafeInteger(policy.minimumUsdMicros) || policy.minimumUsdMicros <= 0) {
      throw new PayoutError("invalid_payout_policy", "The payout minimum must be a positive safe integer.");
    }
  }

  prepare(input: {
    sellerId: string;
    payoutMethod: PayoutMethod;
    earningIds: string[];
    idempotencyKey: string;
  }): PayoutMutationResult {
    validateId(input.sellerId, "seller id");
    validateId(input.idempotencyKey, "idempotency key");
    validateMethod(input.payoutMethod);
    this.assertMethodEnabled(input.payoutMethod);
    const earningIds = [...new Set(input.earningIds)].sort();
    if (earningIds.length === 0 || earningIds.length !== input.earningIds.length) {
      throw new PayoutError("invalid_payout_items", "A payout requires unique earning ids.");
    }
    for (const id of earningIds) validateId(id, "earning id");
    const digest = requestDigest(input.sellerId, input.payoutMethod, earningIds);

    return this.database.transaction(() => {
      const existing = this.database.raw.prepare(
        "SELECT id, request_digest FROM payout_batches WHERE idempotency_key = ?",
      ).get(input.idempotencyKey) as { id: string; request_digest: string } | undefined;
      if (existing) {
        if (existing.request_digest !== digest) {
          throw new PayoutError("payout_idempotency_conflict", "The payout key was reused with different items.");
        }
        return { batch: this.getBatchRequired(existing.id), duplicate: true };
      }

      const placeholders = earningIds.map(() => "?").join(",");
      const rows = this.database.raw.prepare(
        `SELECT e.id, e.seller_id, e.amount_usd_micros, e.payout_method, e.status,
                (SELECT COUNT(*)
                 FROM payout_batch_items i
                 JOIN payout_batches b ON b.id = i.batch_id
                 WHERE i.earning_id = e.id AND b.status <> 'cancelled') AS active_batch_count
         FROM seller_earnings e WHERE e.id IN (${placeholders}) ORDER BY e.id`,
      ).all(...earningIds) as unknown as Array<{
        id: string; seller_id: string; amount_usd_micros: number;
        payout_method: PayoutMethod; status: string; active_batch_count: number;
      }>;
      if (rows.length !== earningIds.length) {
        throw new PayoutError("earning_not_found", "At least one payout earning does not exist.");
      }
      for (const row of rows) {
        if (row.seller_id !== input.sellerId) {
          throw new PayoutError("earning_owner_conflict", "A payout cannot include another seller's earning.");
        }
        if (row.payout_method !== input.payoutMethod) {
          throw new PayoutError("earning_method_conflict", "All earnings must match the payout method snapshot.");
        }
        if (row.status !== "available") {
          throw new PayoutError("earning_not_available", "Every earning must still be available.");
        }
        if (Number(row.active_batch_count) !== 0) {
          throw new PayoutError(
            "earning_already_allocated",
            "An earning cannot be allocated while a previous payout batch remains active.",
          );
        }
      }
      const destination = this.database.raw.prepare(
        `SELECT id, destination_reference, destination_fingerprint
         FROM seller_payout_destinations
         WHERE seller_id = ? AND payout_method = ? AND status = 'active' AND expires_at > ?`,
      ).get(input.sellerId, input.payoutMethod, Date.now()) as {
        id: string; destination_reference: string; destination_fingerprint: string;
      } | undefined;
      if (this.policy.requireVerifiedDestination && !destination) {
        throw new PayoutError(
          "verified_payout_destination_required",
          "A verified, non-expired payout destination is required.",
        );
      }
      const grossUsdMicros = rows.reduce((sum, row) => sum + Number(row.amount_usd_micros), 0);
      const debtRow = this.database.raw.prepare(
        "SELECT amount_usd_micros FROM seller_debts WHERE seller_id = ?",
      ).get(input.sellerId) as { amount_usd_micros: number } | undefined;
      const sellerDebt = Number(debtRow?.amount_usd_micros ?? 0);
      const debtOffsetUsdMicros = Math.min(sellerDebt, grossUsdMicros);
      const amountUsdMicros = grossUsdMicros - debtOffsetUsdMicros;
      if (
        !Number.isSafeInteger(grossUsdMicros)
        || !Number.isSafeInteger(amountUsdMicros)
        || amountUsdMicros < this.policy.minimumUsdMicros
      ) {
        throw new PayoutError("payout_below_minimum", "The selected earnings do not reach the payout minimum.");
      }
      const now = Date.now();
      const batchId = newId("payout");
      this.database.raw.prepare(
         `INSERT INTO payout_batches(
           id, seller_id, payout_method, idempotency_key, request_digest,
           gross_usd_micros, debt_offset_usd_micros, amount_usd_micros,
           destination_id, destination_reference, destination_fingerprint,
           status, created_at, updated_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?, ?)`,
      ).run(
        batchId, input.sellerId, input.payoutMethod, input.idempotencyKey,
        digest, grossUsdMicros, debtOffsetUsdMicros, amountUsdMicros,
        destination?.id ?? null, destination?.destination_reference ?? null,
        destination?.destination_fingerprint ?? null, now, now,
      );
      if (debtOffsetUsdMicros > 0) {
        this.database.raw.prepare(
          "UPDATE seller_debts SET amount_usd_micros = amount_usd_micros - ?, updated_at = ? WHERE seller_id = ?",
        ).run(debtOffsetUsdMicros, now, input.sellerId);
      }
      const insertItem = this.database.raw.prepare(
        "INSERT INTO payout_batch_items(batch_id, earning_id, amount_usd_micros) VALUES (?, ?, ?)",
      );
      for (const row of rows) insertItem.run(batchId, row.id, row.amount_usd_micros);
      this.database.raw.prepare(
        `UPDATE seller_earnings SET status = 'batched'
         WHERE id IN (${placeholders}) AND status = 'available'`,
      ).run(...earningIds);
      return { batch: this.getBatchRequired(batchId), duplicate: false };
    });
  }

  markSubmitted(input: {
    batchId: string;
    dispatchKey: string;
    externalReference: string;
  }): PayoutMutationResult {
    validateId(input.batchId, "batch id");
    validateId(input.dispatchKey, "dispatch key");
    validateId(input.externalReference, "external reference");
    return this.database.transaction(() => {
      const batch = this.getBatchRequired(input.batchId);
      if (batch.status === "submitted" || batch.status === "paid") {
        if (batch.dispatchKey !== input.dispatchKey || batch.externalReference !== input.externalReference) {
          throw new PayoutError("payout_submission_conflict", "A submitted batch cannot change external identity.");
        }
        return { batch, duplicate: true };
      }
      if (batch.status !== "prepared") {
        throw new PayoutError("payout_not_submittable", "Only a prepared payout can be submitted.");
      }
      const now = Date.now();
      try {
        this.database.raw.prepare(
          `UPDATE payout_batches
           SET status = 'submitted', dispatch_key = ?, external_reference = ?,
               submitted_at = ?, updated_at = ? WHERE id = ?`,
        ).run(input.dispatchKey, input.externalReference, now, now, input.batchId);
      } catch (error) {
        throw new PayoutError("payout_external_identity_conflict", errorMessage(error));
      }
      return { batch: this.getBatchRequired(input.batchId), duplicate: false };
    });
  }

  markPaid(input: {
    batchId: string;
    settlementReference: string;
    paidAt: number;
  }): PayoutMutationResult {
    validateId(input.batchId, "batch id");
    validateId(input.settlementReference, "settlement reference");
    if (!Number.isSafeInteger(input.paidAt) || input.paidAt <= 0) {
      throw new PayoutError("invalid_paid_at", "The payout settlement timestamp is invalid.");
    }
    return this.database.transaction(() => {
      const batch = this.getBatchRequired(input.batchId);
      if (batch.status === "paid") {
        if (batch.settlementReference !== input.settlementReference || batch.paidAt !== input.paidAt) {
          throw new PayoutError("payout_settlement_conflict", "A paid batch cannot change settlement evidence.");
        }
        return { batch, duplicate: true };
      }
      if (batch.status !== "submitted") {
        throw new PayoutError("payout_not_submitted", "A payout must be submitted before it can be paid.");
      }
      const now = Date.now();
      this.database.raw.prepare(
        `UPDATE payout_batches SET status = 'paid', settlement_reference = ?, paid_at = ?, updated_at = ?
         WHERE id = ?`,
      ).run(input.settlementReference, input.paidAt, now, input.batchId);
      this.database.raw.prepare(
        `UPDATE seller_earnings SET status = 'paid'
         WHERE id IN (SELECT earning_id FROM payout_batch_items WHERE batch_id = ?)`,
      ).run(input.batchId);
      const pendingReversals = this.database.raw.prepare(
        `SELECT r.id, r.earning_id, r.seller_id, r.amount_usd_micros
         FROM seller_earning_reversals r
         JOIN payout_batch_items i ON i.earning_id = r.earning_id
         WHERE i.batch_id = ? AND r.state = 'pending_payout'`,
      ).all(input.batchId) as unknown as Array<{
        id: string; earning_id: string; seller_id: string; amount_usd_micros: number;
      }>;
      for (const reversal of pendingReversals) {
        this.database.raw.prepare(
          `INSERT INTO seller_debts(seller_id, amount_usd_micros, updated_at)
           VALUES (?, ?, ?)
           ON CONFLICT(seller_id) DO UPDATE
           SET amount_usd_micros = amount_usd_micros + excluded.amount_usd_micros,
               updated_at = excluded.updated_at`,
        ).run(reversal.seller_id, reversal.amount_usd_micros, now);
        this.database.raw.prepare(
          "UPDATE seller_earnings SET status = 'reversed' WHERE id = ? AND status = 'paid'",
        ).run(reversal.earning_id);
        this.database.raw.prepare(
          `UPDATE seller_earning_reversals
           SET state = 'seller_debt', resolved_at = ? WHERE id = ? AND state = 'pending_payout'`,
        ).run(now, reversal.id);
      }
      return { batch: this.getBatchRequired(input.batchId), duplicate: false };
    });
  }

  cancelPrepared(batchId: string): PayoutMutationResult {
    validateId(batchId, "batch id");
    return this.database.transaction(() => {
      const batch = this.getBatchRequired(batchId);
      if (batch.status === "cancelled") return { batch, duplicate: true };
      if (batch.status !== "prepared") {
        throw new PayoutError("payout_cancellation_unsafe", "A submitted or paid payout cannot be released automatically.");
      }
      const dispatch = this.database.raw.prepare(
        `SELECT state, external_reference, settlement_reference
         FROM payout_dispatch_operations WHERE batch_id = ?`,
      ).get(batchId) as {
        state: string;
        external_reference: string | null;
        settlement_reference: string | null;
      } | undefined;
      const safelyRejected = dispatch?.state === "rejected"
        && dispatch.external_reference === null
        && dispatch.settlement_reference === null;
      if (dispatch && !safelyRejected) {
        throw new PayoutError(
          "payout_cancellation_unsafe",
          "A payout with a durable dispatch operation must be reconciled before funds can be released.",
        );
      }
      const now = Date.now();
      this.database.raw.prepare(
        "UPDATE payout_batches SET status = 'cancelled', cancelled_at = ?, updated_at = ? WHERE id = ?",
      ).run(now, now, batchId);
      if (batch.payoutMethod === "spore") {
        this.database.raw.prepare(
          `UPDATE spore_conversion_quotes SET status = 'superseded', replaced_at = ?
           WHERE batch_id = ? AND status = 'active'`,
        ).run(now, batchId);
      }
      this.database.raw.prepare(
        `UPDATE seller_earnings SET status = 'available'
         WHERE id IN (SELECT earning_id FROM payout_batch_items WHERE batch_id = ?)
           AND status = 'batched'`,
      ).run(batchId);
      if (batch.debtOffsetUsdMicros > 0) {
        this.database.raw.prepare(
          `INSERT INTO seller_debts(seller_id, amount_usd_micros, updated_at)
           VALUES (?, ?, ?)
           ON CONFLICT(seller_id) DO UPDATE
           SET amount_usd_micros = amount_usd_micros + excluded.amount_usd_micros,
               updated_at = excluded.updated_at`,
        ).run(batch.sellerId, batch.debtOffsetUsdMicros, now);
      }
      return { batch: this.getBatchRequired(batchId), duplicate: false };
    });
  }

  getBatch(batchId: string): PayoutBatch | null {
    const row = this.database.raw.prepare(
      `SELECT id, seller_id, payout_method, idempotency_key, gross_usd_micros,
              debt_offset_usd_micros, amount_usd_micros, status,
              destination_id, destination_reference, destination_fingerprint,
              dispatch_key, external_reference, settlement_reference, created_at, updated_at,
              submitted_at, paid_at, cancelled_at
       FROM payout_batches WHERE id = ?`,
    ).get(batchId) as {
      id: string; seller_id: string; payout_method: PayoutMethod; idempotency_key: string;
      gross_usd_micros: number; debt_offset_usd_micros: number;
      amount_usd_micros: number; status: PayoutStatus; dispatch_key: string | null;
      destination_id: string | null; destination_reference: string | null;
      destination_fingerprint: string | null;
      external_reference: string | null; settlement_reference: string | null;
      created_at: number; updated_at: number; submitted_at: number | null;
      paid_at: number | null; cancelled_at: number | null;
    } | undefined;
    if (!row) return null;
    const items = this.database.raw.prepare(
      "SELECT earning_id FROM payout_batch_items WHERE batch_id = ? ORDER BY earning_id",
    ).all(batchId) as unknown as Array<{ earning_id: string }>;
    return {
      id: row.id,
      sellerId: row.seller_id,
      payoutMethod: row.payout_method,
      idempotencyKey: row.idempotency_key,
      grossUsdMicros: Number(row.gross_usd_micros),
      debtOffsetUsdMicros: Number(row.debt_offset_usd_micros),
      amountUsdMicros: Number(row.amount_usd_micros),
      destinationId: row.destination_id,
      destinationReference: row.destination_reference,
      destinationFingerprint: row.destination_fingerprint,
      status: row.status,
      dispatchKey: row.dispatch_key,
      externalReference: row.external_reference,
      settlementReference: row.settlement_reference,
      earningIds: items.map((item) => item.earning_id),
      createdAt: Number(row.created_at),
      updatedAt: Number(row.updated_at),
      submittedAt: row.submitted_at === null ? null : Number(row.submitted_at),
      paidAt: row.paid_at === null ? null : Number(row.paid_at),
      cancelledAt: row.cancelled_at === null ? null : Number(row.cancelled_at),
    };
  }

  listSellerBatches(sellerId: string, limit = 50): PayoutBatch[] {
    validateId(sellerId, "seller id");
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > 100) {
      throw new PayoutError("invalid_limit", "The payout limit must be between 1 and 100.");
    }
    const rows = this.database.raw.prepare(
      `SELECT id FROM payout_batches WHERE seller_id = ?
       ORDER BY created_at DESC, id DESC LIMIT ?`,
    ).all(sellerId, limit) as unknown as Array<{ id: string }>;
    return rows.map((row) => this.getBatchRequired(row.id));
  }

  assertPayoutMethodEnabled(method: PayoutMethod): void {
    validateMethod(method);
    this.assertMethodEnabled(method);
  }

  private getBatchRequired(batchId: string): PayoutBatch {
    const batch = this.getBatch(batchId);
    if (!batch) throw new PayoutError("payout_not_found", "The payout batch does not exist.");
    return batch;
  }

  private assertMethodEnabled(method: PayoutMethod): void {
    if (method !== "spore") return;
    const missing = Object.entries(this.policy.spore)
      .filter(([, approved]) => !approved)
      .map(([gate]) => gate);
    if (missing.length > 0) {
      throw new PayoutError("spore_payout_gates_closed", `SPORE payout gates are closed: ${missing.join(", ")}.`);
    }
  }
}

function requestDigest(sellerId: string, method: PayoutMethod, earningIds: string[]): string {
  return createHash("sha256").update(JSON.stringify([sellerId, method, earningIds]), "utf8").digest("hex");
}

function validateMethod(method: PayoutMethod): void {
  if (method !== "stable" && method !== "spore") {
    throw new PayoutError("invalid_payout_method", "The payout method is invalid.");
  }
}

function validateId(value: string, label: string): void {
  if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$/.test(value)) {
    throw new PayoutError("invalid_identifier", `The ${label} is invalid.`);
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
