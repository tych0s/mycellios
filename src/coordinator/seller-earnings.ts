import { createPublicKey, verify } from "node:crypto";
import { newId } from "../core/ids.js";
import type { MeshDatabase } from "../storage/database.js";

export interface WorkReceiptPayload {
  schema: "mycellios.work-receipt.v1";
  receiptId: string;
  sellerId: string;
  workerId: string;
  jobId: string;
  stageId: string;
  pricingVersion: string;
  amountUsdMicros: number;
  evidenceDigest: string;
  accepted: boolean;
  acceptedAt: number;
  verifierKeyId: string;
}

export interface SignedWorkReceipt extends WorkReceiptPayload {
  signature: string;
}

export interface EarningReversalPayload {
  schema: "mycellios.earning-reversal.v1";
  reversalId: string;
  earningId: string;
  sellerId: string;
  reason: "fraud" | "verification_error" | "buyer_dispute";
  evidenceDigest: string;
  reversedAt: number;
  verifierKeyId: string;
}

export interface SignedEarningReversal extends EarningReversalPayload {
  signature: string;
}

export interface EarningReversalResult {
  id: string;
  reversalId: string;
  earningId: string;
  sellerId: string;
  amountUsdMicros: number;
  state: "applied" | "seller_debt" | "pending_payout";
  duplicate: boolean;
}

export interface SellerEarning {
  id: string;
  sellerId: string;
  receiptId: string;
  amountUsdMicros: number;
  payoutMethod: "stable" | "spore";
  status: "available" | "batched" | "paid" | "reversed";
  createdAt: number;
}

export class SellerEarningsError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "SellerEarningsError";
  }
}

export class SellerEarningsManager {
  constructor(
    private readonly database: MeshDatabase,
    private readonly trustedVerifierKeys: ReadonlyMap<string, string>,
  ) {}

  setPayoutPreference(sellerId: string, method: "stable" | "spore"): void {
    validateId(sellerId, "seller id");
    if (method !== "stable" && method !== "spore") {
      throw new SellerEarningsError("invalid_payout_method", "The payout method is invalid.");
    }
    this.database.raw.prepare(
      `INSERT INTO seller_payout_preferences(seller_id, method, updated_at)
       VALUES (?, ?, ?)
       ON CONFLICT(seller_id) DO UPDATE SET method = excluded.method, updated_at = excluded.updated_at`,
    ).run(sellerId, method, Date.now());
  }

  getPayoutPreference(sellerId: string): "stable" | "spore" {
    validateId(sellerId, "seller id");
    const row = this.database.raw.prepare(
      "SELECT method FROM seller_payout_preferences WHERE seller_id = ?",
    ).get(sellerId) as { method: "stable" | "spore" } | undefined;
    return row?.method ?? "stable";
  }

  list(sellerId: string, limit = 100): SellerEarning[] {
    validateId(sellerId, "seller id");
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > 100) {
      throw new SellerEarningsError("invalid_limit", "The earnings limit must be between 1 and 100.");
    }
    const rows = this.database.raw.prepare(
      `SELECT id, seller_id, receipt_id, amount_usd_micros, payout_method, status, created_at
       FROM seller_earnings WHERE seller_id = ? ORDER BY created_at DESC, id DESC LIMIT ?`,
    ).all(sellerId, limit) as unknown as Array<{
      id: string; seller_id: string; receipt_id: string; amount_usd_micros: number;
      payout_method: SellerEarning["payoutMethod"]; status: SellerEarning["status"]; created_at: number;
    }>;
    return rows.map(mapSellerEarning);
  }

  recordAcceptedReceipt(receipt: SignedWorkReceipt): SellerEarning {
    validateReceipt(receipt);
    const publicKey = this.trustedVerifierKeys.get(receipt.verifierKeyId);
    if (!publicKey) {
      throw new SellerEarningsError("untrusted_receipt_verifier", "The receipt verifier is not trusted.");
    }
    let signatureValid = false;
    try {
      signatureValid = verify(
        null,
        workReceiptSigningBytes(receipt),
        createPublicKey(publicKey),
        Buffer.from(receipt.signature, "base64url"),
      );
    } catch {
      signatureValid = false;
    }
    if (!signatureValid) {
      throw new SellerEarningsError("invalid_receipt_signature", "The work receipt signature is invalid.");
    }
    if (!receipt.accepted) {
      throw new SellerEarningsError("work_not_accepted", "Rejected work cannot create seller earnings.");
    }

    return this.database.transaction(() => {
      const existing = this.readEarningByReceipt(receipt.receiptId);
      if (existing) {
        const stored = this.database.raw.prepare(
          `SELECT seller_id, worker_id, job_id, stage_id, pricing_version,
                  amount_usd_micros, evidence_digest, accepted_at, verifier_key_id, signature
           FROM verified_work_receipts WHERE receipt_id = ?`,
        ).get(receipt.receiptId) as Record<string, unknown>;
        if (!storedReceiptMatches(stored, receipt)) {
          throw new SellerEarningsError("receipt_replay_conflict", "A receipt id was reused with different work evidence.");
        }
        return existing;
      }

      const methodRow = this.database.raw.prepare(
        "SELECT method FROM seller_payout_preferences WHERE seller_id = ?",
      ).get(receipt.sellerId) as { method: "stable" | "spore" } | undefined;
      const payoutMethod = methodRow?.method ?? "stable";
      const now = Date.now();
      this.database.raw.prepare(
        `INSERT INTO verified_work_receipts(
           receipt_id, seller_id, worker_id, job_id, stage_id, pricing_version,
           amount_usd_micros, evidence_digest, accepted_at, verifier_key_id, signature, recorded_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        receipt.receiptId, receipt.sellerId, receipt.workerId, receipt.jobId,
        receipt.stageId, receipt.pricingVersion, receipt.amountUsdMicros,
        receipt.evidenceDigest, receipt.acceptedAt, receipt.verifierKeyId,
        receipt.signature, now,
      );
      const earning: SellerEarning = {
        id: newId("earn"),
        sellerId: receipt.sellerId,
        receiptId: receipt.receiptId,
        amountUsdMicros: receipt.amountUsdMicros,
        payoutMethod,
        status: "available",
        createdAt: now,
      };
      this.database.raw.prepare(
        `INSERT INTO seller_earnings(
           id, seller_id, receipt_id, amount_usd_micros, payout_method, status, created_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        earning.id, earning.sellerId, earning.receiptId, earning.amountUsdMicros,
        earning.payoutMethod, earning.status, earning.createdAt,
      );
      return earning;
    });
  }

  recordEarningReversal(reversal: SignedEarningReversal): EarningReversalResult {
    validateEarningReversal(reversal);
    const publicKey = this.trustedVerifierKeys.get(reversal.verifierKeyId);
    if (!publicKey) {
      throw new SellerEarningsError("untrusted_reversal_verifier", "The reversal verifier is not trusted.");
    }
    let signatureValid = false;
    try {
      signatureValid = verify(
        null,
        earningReversalSigningBytes(reversal),
        createPublicKey(publicKey),
        Buffer.from(reversal.signature, "base64url"),
      );
    } catch {
      signatureValid = false;
    }
    if (!signatureValid) {
      throw new SellerEarningsError("invalid_reversal_signature", "The earning reversal signature is invalid.");
    }

    return this.database.transaction(() => {
      const existing = this.database.raw.prepare(
        `SELECT id, reversal_id, earning_id, seller_id, reason, evidence_digest,
                verifier_key_id, signature, amount_usd_micros, state, reversed_at
         FROM seller_earning_reversals WHERE reversal_id = ?`,
      ).get(reversal.reversalId) as Record<string, unknown> | undefined;
      if (existing) {
        if (!storedReversalMatches(existing, reversal)) {
          throw new SellerEarningsError("reversal_replay_conflict", "A reversal id was reused with different evidence.");
        }
        return reversalResult(existing, true);
      }

      const earning = this.database.raw.prepare(
        `SELECT id, seller_id, amount_usd_micros, status
         FROM seller_earnings WHERE id = ?`,
      ).get(reversal.earningId) as {
        id: string; seller_id: string; amount_usd_micros: number;
        status: SellerEarning["status"];
      } | undefined;
      if (!earning) throw new SellerEarningsError("earning_not_found", "The reversed earning does not exist.");
      if (earning.seller_id !== reversal.sellerId) {
        throw new SellerEarningsError("earning_owner_conflict", "An earning reversal cannot move between sellers.");
      }
      const priorForEarning = this.database.raw.prepare(
        "SELECT 1 AS found FROM seller_earning_reversals WHERE earning_id = ?",
      ).get(reversal.earningId) as { found: number } | undefined;
      if (priorForEarning) {
        throw new SellerEarningsError("earning_already_reversed", "The earning already has a reversal.");
      }

      let state: EarningReversalResult["state"];
      if (earning.status === "available") {
        state = "applied";
      } else if (earning.status === "paid") {
        state = "seller_debt";
      } else if (earning.status === "batched") {
        const batch = this.database.raw.prepare(
          `SELECT b.status FROM payout_batches b
           JOIN payout_batch_items i ON i.batch_id = b.id
           WHERE i.earning_id = ? AND b.status IN ('prepared', 'submitted')
           ORDER BY b.created_at DESC LIMIT 1`,
        ).get(reversal.earningId) as { status: "prepared" | "submitted" } | undefined;
        if (!batch) {
          throw new SellerEarningsError("earning_batch_inconsistent", "The batched earning has no active payout batch.");
        }
        if (batch.status === "prepared") {
          throw new SellerEarningsError(
            "earning_in_prepared_payout",
            "Cancel the prepared payout before reversing this earning.",
          );
        }
        state = "pending_payout";
      } else {
        throw new SellerEarningsError("earning_already_reversed", "The earning is already reversed.");
      }

      const now = Date.now();
      const id = newId("earnrev");
      this.database.raw.prepare(
        `INSERT INTO seller_earning_reversals(
           id, reversal_id, earning_id, seller_id, reason, evidence_digest,
           verifier_key_id, signature, amount_usd_micros, state,
           reversed_at, recorded_at, resolved_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        id, reversal.reversalId, reversal.earningId, reversal.sellerId,
        reversal.reason, reversal.evidenceDigest, reversal.verifierKeyId,
        reversal.signature, earning.amount_usd_micros, state,
        reversal.reversedAt, now, state === "pending_payout" ? null : now,
      );
      if (state === "applied") {
        this.database.raw.prepare(
          "UPDATE seller_earnings SET status = 'reversed' WHERE id = ? AND status = 'available'",
        ).run(reversal.earningId);
      } else if (state === "seller_debt") {
        addSellerDebt(this.database, reversal.sellerId, Number(earning.amount_usd_micros), now);
        this.database.raw.prepare(
          "UPDATE seller_earnings SET status = 'reversed' WHERE id = ? AND status = 'paid'",
        ).run(reversal.earningId);
      }
      return {
        id,
        reversalId: reversal.reversalId,
        earningId: reversal.earningId,
        sellerId: reversal.sellerId,
        amountUsdMicros: Number(earning.amount_usd_micros),
        state,
        duplicate: false,
      };
    });
  }

  getSellerDebt(sellerId: string): number {
    validateId(sellerId, "seller id");
    const row = this.database.raw.prepare(
      "SELECT amount_usd_micros FROM seller_debts WHERE seller_id = ?",
    ).get(sellerId) as { amount_usd_micros: number } | undefined;
    return Number(row?.amount_usd_micros ?? 0);
  }

  listAvailable(sellerId: string): SellerEarning[] {
    validateId(sellerId, "seller id");
    const rows = this.database.raw.prepare(
      `SELECT id, seller_id, receipt_id, amount_usd_micros, payout_method, status, created_at
       FROM seller_earnings WHERE seller_id = ? AND status = 'available'
       ORDER BY created_at, id`,
    ).all(sellerId) as unknown as Array<{
      id: string; seller_id: string; receipt_id: string; amount_usd_micros: number;
      payout_method: SellerEarning["payoutMethod"]; status: SellerEarning["status"]; created_at: number;
    }>;
    return rows.map(mapSellerEarning);
  }

  private readEarningByReceipt(receiptId: string): SellerEarning | null {
    const row = this.database.raw.prepare(
      `SELECT id, seller_id, receipt_id, amount_usd_micros, payout_method, status, created_at
       FROM seller_earnings WHERE receipt_id = ?`,
    ).get(receiptId) as {
      id: string; seller_id: string; receipt_id: string; amount_usd_micros: number;
      payout_method: SellerEarning["payoutMethod"]; status: SellerEarning["status"]; created_at: number;
    } | undefined;
    return row ? {
      id: row.id,
      sellerId: row.seller_id,
      receiptId: row.receipt_id,
      amountUsdMicros: Number(row.amount_usd_micros),
      payoutMethod: row.payout_method,
      status: row.status,
      createdAt: Number(row.created_at),
    } : null;
  }
}

function mapSellerEarning(row: {
  id: string; seller_id: string; receipt_id: string; amount_usd_micros: number;
  payout_method: SellerEarning["payoutMethod"]; status: SellerEarning["status"];
  created_at: number;
}): SellerEarning {
  return {
    id: row.id,
    sellerId: row.seller_id,
    receiptId: row.receipt_id,
    amountUsdMicros: Number(row.amount_usd_micros),
    payoutMethod: row.payout_method,
    status: row.status,
    createdAt: Number(row.created_at),
  };
}

export function workReceiptSigningBytes(receipt: WorkReceiptPayload): Buffer {
  return Buffer.from(JSON.stringify([
    receipt.schema,
    receipt.receiptId,
    receipt.sellerId,
    receipt.workerId,
    receipt.jobId,
    receipt.stageId,
    receipt.pricingVersion,
    receipt.amountUsdMicros,
    receipt.evidenceDigest,
    receipt.accepted,
    receipt.acceptedAt,
    receipt.verifierKeyId,
  ]), "utf8");
}

export function earningReversalSigningBytes(reversal: EarningReversalPayload): Buffer {
  return Buffer.from(JSON.stringify([
    reversal.schema,
    reversal.reversalId,
    reversal.earningId,
    reversal.sellerId,
    reversal.reason,
    reversal.evidenceDigest,
    reversal.reversedAt,
    reversal.verifierKeyId,
  ]), "utf8");
}

function validateReceipt(receipt: SignedWorkReceipt): void {
  if (receipt.schema !== "mycellios.work-receipt.v1") {
    throw new SellerEarningsError("unsupported_receipt_schema", "The work receipt schema is unsupported.");
  }
  for (const [label, value] of [
    ["receipt id", receipt.receiptId], ["seller id", receipt.sellerId],
    ["worker id", receipt.workerId], ["job id", receipt.jobId],
    ["stage id", receipt.stageId], ["pricing version", receipt.pricingVersion],
    ["verifier key id", receipt.verifierKeyId],
  ] as const) validateId(value, label);
  if (!Number.isSafeInteger(receipt.amountUsdMicros) || receipt.amountUsdMicros <= 0) {
    throw new SellerEarningsError("invalid_earning_amount", "The receipt amount must be a positive safe integer.");
  }
  if (!/^[a-f0-9]{64}$/.test(receipt.evidenceDigest)) {
    throw new SellerEarningsError("invalid_evidence_digest", "The evidence digest must be canonical SHA-256 hex.");
  }
  if (!Number.isSafeInteger(receipt.acceptedAt) || receipt.acceptedAt <= 0) {
    throw new SellerEarningsError("invalid_accepted_at", "The accepted timestamp is invalid.");
  }
  if (!/^[A-Za-z0-9_-]{80,120}$/.test(receipt.signature)) {
    throw new SellerEarningsError("invalid_receipt_signature", "The work receipt signature encoding is invalid.");
  }
}

function validateEarningReversal(reversal: SignedEarningReversal): void {
  if (reversal.schema !== "mycellios.earning-reversal.v1") {
    throw new SellerEarningsError("unsupported_reversal_schema", "The earning reversal schema is unsupported.");
  }
  for (const [label, value] of [
    ["reversal id", reversal.reversalId], ["earning id", reversal.earningId],
    ["seller id", reversal.sellerId], ["verifier key id", reversal.verifierKeyId],
  ] as const) validateId(value, label);
  if (!/^[a-f0-9]{64}$/.test(reversal.evidenceDigest)) {
    throw new SellerEarningsError("invalid_evidence_digest", "The reversal evidence digest must be canonical SHA-256 hex.");
  }
  if (!Number.isSafeInteger(reversal.reversedAt) || reversal.reversedAt <= 0) {
    throw new SellerEarningsError("invalid_reversed_at", "The reversal timestamp is invalid.");
  }
  if (!/^[A-Za-z0-9_-]{80,120}$/.test(reversal.signature)) {
    throw new SellerEarningsError("invalid_reversal_signature", "The reversal signature encoding is invalid.");
  }
}

function validateId(value: string, label: string): void {
  if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$/.test(value)) {
    throw new SellerEarningsError("invalid_identifier", `The ${label} is invalid.`);
  }
}

function storedReceiptMatches(stored: Record<string, unknown>, receipt: SignedWorkReceipt): boolean {
  return stored.seller_id === receipt.sellerId
    && stored.worker_id === receipt.workerId
    && stored.job_id === receipt.jobId
    && stored.stage_id === receipt.stageId
    && stored.pricing_version === receipt.pricingVersion
    && Number(stored.amount_usd_micros) === receipt.amountUsdMicros
    && stored.evidence_digest === receipt.evidenceDigest
    && Number(stored.accepted_at) === receipt.acceptedAt
    && stored.verifier_key_id === receipt.verifierKeyId
    && stored.signature === receipt.signature;
}

function storedReversalMatches(stored: Record<string, unknown>, reversal: SignedEarningReversal): boolean {
  return stored.reversal_id === reversal.reversalId
    && stored.earning_id === reversal.earningId
    && stored.seller_id === reversal.sellerId
    && stored.reason === reversal.reason
    && stored.evidence_digest === reversal.evidenceDigest
    && stored.verifier_key_id === reversal.verifierKeyId
    && stored.signature === reversal.signature
    && Number(stored.reversed_at) === reversal.reversedAt;
}

function reversalResult(stored: Record<string, unknown>, duplicate: boolean): EarningReversalResult {
  return {
    id: String(stored.id),
    reversalId: String(stored.reversal_id),
    earningId: String(stored.earning_id),
    sellerId: String(stored.seller_id),
    amountUsdMicros: Number(stored.amount_usd_micros),
    state: stored.state as EarningReversalResult["state"],
    duplicate,
  };
}

function addSellerDebt(database: MeshDatabase, sellerId: string, amountUsdMicros: number, now: number): void {
  database.raw.prepare(
    `INSERT INTO seller_debts(seller_id, amount_usd_micros, updated_at)
     VALUES (?, ?, ?)
     ON CONFLICT(seller_id) DO UPDATE
     SET amount_usd_micros = amount_usd_micros + excluded.amount_usd_micros,
         updated_at = excluded.updated_at`,
  ).run(sellerId, amountUsdMicros, now);
}
