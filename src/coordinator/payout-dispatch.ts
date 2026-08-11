import { newId } from "../core/ids.js";
import type { MeshDatabase } from "../storage/database.js";
import { PayoutError, PayoutManager, type PayoutBatch, type PayoutMethod } from "./payouts.js";
import type { PayoutSettlementAttestation } from "./payout-settlement.js";
import {
  SporeConversionError,
  sporeConversionAttestationDigest,
  type SporeConversionQuote,
  type SporeConversionQuoteStore,
} from "./spore-conversion.js";

export type PayoutDispatchState = "dispatching" | "uncertain" | "submitted" | "paid" | "rejected";

export interface PayoutDispatchOperation {
  id: string;
  batchId: string;
  payoutMethod: PayoutMethod;
  dispatchKey: string;
  state: PayoutDispatchState;
  externalReference: string | null;
  settlementReference: string | null;
  sporeQuoteId: string | null;
  sporeQuoteAttestationDigest: string | null;
  lastError: string | null;
  createdAt: number;
  updatedAt: number;
  reconciledAt: number | null;
}

export type PayoutProviderObservation =
  | { state: "absent" }
  | { state: "rejected"; reason: string }
  | { state: "submitted"; externalReference: string }
  | {
      state: "paid";
      externalReference: string;
      settlementReference: string;
      paidAt: number;
    };

export interface PayoutGateway {
  createTransfer(input: {
    batchId: string;
    sellerId: string;
    payoutMethod: PayoutMethod;
    amountUsdMicros: number;
    destinationReference: string | null;
    destinationFingerprint: string | null;
    dispatchKey: string;
    sporeQuote?: SporeConversionQuote | null;
  }): Promise<Exclude<PayoutProviderObservation, { state: "absent" }>>;
  inspectTransfer(input: {
    dispatchKey: string;
    externalReference: string | null;
  }): Promise<PayoutProviderObservation>;
}

export interface PayoutDispatchResult {
  operation: PayoutDispatchOperation;
  batch: PayoutBatch;
  duplicate: boolean;
}

export interface PayoutSettlementEvidenceRecord {
  batchId: string;
  schema: "mycellios.payout-settlement.v1";
  externalReference: string;
  settlementReference: string;
  paidAt: number;
  issuedAt: number;
  verifierKeyId: string;
  signature: string;
  recordedAt: number;
}

/**
 * Persists the provider idempotency identity before money can move. A failed
 * network call is intentionally uncertain and can only advance through
 * provider reconciliation; dispatch() never blindly sends it again.
 */
export class PayoutDispatchService {
  constructor(
    private readonly database: MeshDatabase,
    private readonly payouts: PayoutManager,
    private readonly gateways: Partial<Record<PayoutMethod, PayoutGateway>>,
    private readonly sporeQuotes: SporeConversionQuoteStore | null = null,
  ) {}

  async dispatch(batchId: string): Promise<PayoutDispatchResult> {
    const current = this.requireBatch(batchId);
    const existing = this.getByBatchId(current.id);
    if (existing) {
      this.assertProjectionIntegrity(existing, current);
      return { operation: existing, batch: current, duplicate: true };
    }
    const batch = this.requirePreparedBatch(batchId);
    this.assertBatchEconomicIntegrity(batch);
    this.payouts.assertPayoutMethodEnabled(batch.payoutMethod);
    const gateway = this.requireGateway(batch.payoutMethod);
    const sporeQuote = this.requireDispatchableSporeQuote(batch);

    const allocation = this.createOperation(batch, sporeQuote);
    if (!allocation.created) {
      this.assertProjectionIntegrity(allocation.operation, batch);
      return { operation: allocation.operation, batch, duplicate: true };
    }
    const operation = allocation.operation;
    try {
      const observation = await gateway.createTransfer({
        batchId: batch.id,
        sellerId: batch.sellerId,
        payoutMethod: batch.payoutMethod,
        amountUsdMicros: batch.amountUsdMicros,
        destinationReference: batch.destinationReference,
        destinationFingerprint: batch.destinationFingerprint,
        dispatchKey: operation.dispatchKey,
        sporeQuote,
      });
      return this.applyObservation(operation, observation, false);
    } catch (error) {
      const uncertain = this.updateOperation(operation.id, {
        state: "uncertain",
        lastError: boundedError(error),
        reconciledAt: null,
      });
      return { operation: uncertain, batch: this.requireBatch(batch.id), duplicate: false };
    }
  }

  async reconcile(batchId: string): Promise<PayoutDispatchResult> {
    const operation = this.getByBatchId(batchId);
    if (!operation) throw new PayoutError("payout_dispatch_not_found", "The payout has no dispatch operation.");
    const batch = this.requireBatch(batchId);
    this.assertProjectionIntegrity(operation, batch);
    if (operation.state === "paid" || operation.state === "rejected") {
      return { operation, batch, duplicate: true };
    }
    this.assertSporeQuoteAuditIntegrity(operation);
    const gateway = this.requireGateway(operation.payoutMethod);
    const observation = await gateway.inspectTransfer({
      dispatchKey: operation.dispatchKey,
      externalReference: operation.externalReference,
    });
    if (observation.state === "absent") {
      const uncertain = this.updateOperation(operation.id, {
        state: "uncertain",
        lastError: "provider_transfer_absent",
        reconciledAt: Date.now(),
      });
      return { operation: uncertain, batch: this.requireBatch(batchId), duplicate: true };
    }
    return this.applyObservation(operation, observation, true);
  }

  recordSettlement(
    evidence: PayoutSettlementAttestation,
  ): PayoutDispatchResult {
    return this.database.transaction(() => {
      const operation = this.getByBatchId(evidence.batchId);
      if (!operation) throw new PayoutError("payout_dispatch_not_found", "The payout has no dispatch operation.");
      this.assertProjectionIntegrity(operation, this.requireBatch(evidence.batchId));
      if (operation.state === "rejected") {
        throw new PayoutError("payout_dispatch_terminal", "A rejected payout dispatch cannot be settled.");
      }
      this.assertSporeQuoteAuditIntegrity(operation);
      if (operation.externalReference !== evidence.externalReference) {
        throw new PayoutError("payout_settlement_identity_conflict", "Settlement evidence does not match the dispatched transfer.");
      }
      const existing = this.database.raw.prepare(
        `SELECT schema_name, external_reference, settlement_reference, paid_at,
                issued_at, verifier_key_id, signature
         FROM payout_settlement_evidence WHERE batch_id = ?`,
      ).get(evidence.batchId) as SettlementEvidenceRow | undefined;
      if (existing && !settlementEvidenceMatches(existing, evidence)) {
        throw new PayoutError("payout_settlement_evidence_conflict", "The payout already has different signed settlement evidence.");
      }
      if (!existing) {
        try {
          this.database.raw.prepare(
            `INSERT INTO payout_settlement_evidence(
               batch_id, schema_name, external_reference, settlement_reference,
               paid_at, issued_at, verifier_key_id, signature, recorded_at
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
          ).run(
            evidence.batchId, evidence.schema, evidence.externalReference,
            evidence.settlementReference, evidence.paidAt, evidence.issuedAt,
            evidence.verifierKeyId, evidence.signature, Date.now(),
          );
        } catch (error) {
          throw new PayoutError("payout_settlement_evidence_conflict", boundedError(error));
        }
      }
      const settled = this.applyObservation(operation, {
        state: "paid",
        externalReference: evidence.externalReference,
        settlementReference: evidence.settlementReference,
        paidAt: evidence.paidAt,
      }, true);
      return { ...settled, duplicate: Boolean(existing) || operation.state === "paid" };
    });
  }

  getByBatchId(batchId: string): PayoutDispatchOperation | null {
    const row = this.database.raw.prepare(
      `SELECT id, batch_id, payout_method, dispatch_key, state, external_reference,
              settlement_reference, spore_quote_id, spore_quote_attestation_digest,
              last_error, created_at, updated_at, reconciled_at
       FROM payout_dispatch_operations WHERE batch_id = ?`,
    ).get(batchId) as DispatchRow | undefined;
    return row ? mapOperation(row) : null;
  }

  getSettlementEvidence(batchId: string): PayoutSettlementEvidenceRecord | null {
    const row = this.database.raw.prepare(
      `SELECT batch_id, schema_name, external_reference, settlement_reference,
              paid_at, issued_at, verifier_key_id, signature, recorded_at
       FROM payout_settlement_evidence WHERE batch_id = ?`,
    ).get(batchId) as (SettlementEvidenceRow & { batch_id: string; recorded_at: number }) | undefined;
    return row ? {
      batchId: row.batch_id,
      schema: row.schema_name as "mycellios.payout-settlement.v1",
      externalReference: row.external_reference,
      settlementReference: row.settlement_reference,
      paidAt: Number(row.paid_at),
      issuedAt: Number(row.issued_at),
      verifierKeyId: row.verifier_key_id,
      signature: row.signature,
      recordedAt: Number(row.recorded_at),
    } : null;
  }

  private createOperation(batch: PayoutBatch, sporeQuote: SporeConversionQuote | null): {
    operation: PayoutDispatchOperation;
    created: boolean;
  } {
    return this.database.transaction(() => {
      const existing = this.getByBatchId(batch.id);
      if (existing) return { operation: existing, created: false };
      const now = Date.now();
      const id = newId("payout-dispatch");
      const dispatchKey = `payout:${batch.id}`;
      const sporeQuoteAttestationDigest = sporeQuote
        ? sporeConversionAttestationDigest(sporeQuote)
        : null;
      this.database.raw.prepare(
        `INSERT INTO payout_dispatch_operations(
           id, batch_id, payout_method, dispatch_key, state, spore_quote_id,
           spore_quote_attestation_digest, created_at, updated_at
         ) VALUES (?, ?, ?, ?, 'dispatching', ?, ?, ?, ?)`,
      ).run(
        id, batch.id, batch.payoutMethod, dispatchKey,
        sporeQuote?.quoteId ?? null, sporeQuoteAttestationDigest, now, now,
      );
      return { operation: this.getByBatchId(batch.id)!, created: true };
    });
  }

  private applyObservation(
    operation: PayoutDispatchOperation,
    observation: Exclude<PayoutProviderObservation, { state: "absent" }>,
    reconciled: boolean,
  ): PayoutDispatchResult {
    if (observation.state === "rejected") {
      return this.database.transaction(() => {
        const rejected = this.updateOperation(operation.id, {
          state: "rejected",
          lastError: observation.reason.slice(0, 500),
          reconciledAt: reconciled ? Date.now() : null,
        });
        const batch = rejected.externalReference === null
          ? this.payouts.cancelPrepared(operation.batchId).batch
          : this.requireBatch(operation.batchId);
        return { operation: rejected, batch, duplicate: false };
      });
    }

    return this.database.transaction(() => {
      this.payouts.markSubmitted({
        batchId: operation.batchId,
        dispatchKey: operation.dispatchKey,
        externalReference: observation.externalReference,
      });
      if (observation.state === "paid") {
        this.payouts.markPaid({
          batchId: operation.batchId,
          settlementReference: observation.settlementReference,
          paidAt: observation.paidAt,
        });
      }
      const updated = this.updateOperation(operation.id, {
        state: observation.state,
        externalReference: observation.externalReference,
        settlementReference: observation.state === "paid" ? observation.settlementReference : null,
        lastError: null,
        reconciledAt: reconciled ? Date.now() : null,
      });
      return { operation: updated, batch: this.requireBatch(operation.batchId), duplicate: false };
    });
  }

  private updateOperation(id: string, input: {
    state: PayoutDispatchState;
    externalReference?: string | null;
    settlementReference?: string | null;
    lastError: string | null;
    reconciledAt: number | null;
  }): PayoutDispatchOperation {
    const now = Date.now();
    this.database.raw.prepare(
      `UPDATE payout_dispatch_operations
       SET state = ?, external_reference = COALESCE(?, external_reference),
           settlement_reference = COALESCE(?, settlement_reference), last_error = ?,
           updated_at = ?, reconciled_at = ? WHERE id = ?`,
    ).run(
      input.state,
      input.externalReference ?? null,
      input.settlementReference ?? null,
      input.lastError,
      now,
      input.reconciledAt,
      id,
    );
    const row = this.database.raw.prepare(
      `SELECT id, batch_id, payout_method, dispatch_key, state, external_reference,
              settlement_reference, spore_quote_id, spore_quote_attestation_digest,
              last_error, created_at, updated_at, reconciled_at
       FROM payout_dispatch_operations WHERE id = ?`,
    ).get(id) as DispatchRow | undefined;
    if (!row) throw new PayoutError("payout_dispatch_not_found", "The payout dispatch operation disappeared.");
    return mapOperation(row);
  }

  private requirePreparedBatch(batchId: string): PayoutBatch {
    const batch = this.requireBatch(batchId);
    if (batch.status !== "prepared") {
      throw new PayoutError("payout_not_dispatchable", "Only a prepared payout can be dispatched.");
    }
    return batch;
  }

  private assertProjectionIntegrity(operation: PayoutDispatchOperation, batch: PayoutBatch): void {
    const earningRows = this.database.raw.prepare(
      `SELECT e.status
       FROM payout_batch_items i
       JOIN seller_earnings e ON e.id = i.earning_id
       WHERE i.batch_id = ?`,
    ).all(batch.id) as unknown as Array<{ status: string }>;
    const allowedEarningStatuses = batch.status === "cancelled"
      ? new Set(["available"])
      : batch.status === "paid"
        ? new Set(["paid", "reversed"])
        : new Set(["batched"]);
    const earningsMatch = earningRows.length > 0
      && earningRows.every((row) => allowedEarningStatuses.has(row.status));
    const submittedIdentityMatches = batch.status === "submitted"
      && batch.dispatchKey === operation.dispatchKey
      && batch.externalReference === operation.externalReference
      && batch.settlementReference === null;
    const paidIdentityMatches = batch.status === "paid"
      && batch.dispatchKey === operation.dispatchKey
      && batch.externalReference === operation.externalReference
      && batch.settlementReference === operation.settlementReference;
    const consistent = earningsMatch && batch.payoutMethod === operation.payoutMethod && (
      (operation.state === "dispatching" && batch.status === "prepared")
      || (operation.state === "uncertain" && (
        (operation.externalReference === null && batch.status === "prepared")
        || (operation.externalReference !== null && submittedIdentityMatches)
      ))
      || (operation.state === "submitted" && submittedIdentityMatches)
      || (operation.state === "paid" && paidIdentityMatches)
      || (operation.state === "rejected" && (
        (operation.externalReference === null && batch.status === "cancelled")
        || (operation.externalReference !== null && submittedIdentityMatches)
      ))
    );
    if (!consistent) {
      throw new PayoutError(
        "payout_dispatch_projection_conflict",
        "The durable dispatch operation and payout batch projections do not match.",
      );
    }
  }

  private requireDispatchableSporeQuote(batch: PayoutBatch): SporeConversionQuote | null {
    if (batch.payoutMethod !== "spore") return null;
    if (!this.sporeQuotes) {
      throw new PayoutError("spore_quote_required", "SPORE payout dispatch requires a configured quote verifier.");
    }
    try {
      const quote = this.sporeQuotes.getVerifiedByBatchId(batch.id);
      if (!quote) throw new PayoutError("spore_quote_required", "SPORE payout dispatch requires an active quote.");
      if (quote.sellerId !== batch.sellerId || quote.usdMicros !== batch.amountUsdMicros
        || quote.destinationFingerprint !== batch.destinationFingerprint) {
        throw new PayoutError("spore_quote_batch_mismatch", "The active SPORE quote no longer matches the payout batch.");
      }
      return quote;
    } catch (error) {
      if (error instanceof PayoutError) throw error;
      if (error instanceof SporeConversionError) {
        throw new PayoutError("spore_quote_not_dispatchable", error.message);
      }
      throw error;
    }
  }

  private assertBatchEconomicIntegrity(batch: PayoutBatch): void {
    const rows = this.database.raw.prepare(
      `SELECT i.amount_usd_micros AS item_amount_usd_micros,
              e.amount_usd_micros AS earning_amount_usd_micros,
              e.seller_id, e.payout_method, e.status
       FROM payout_batch_items i
       JOIN seller_earnings e ON e.id = i.earning_id
       WHERE i.batch_id = ?`,
    ).all(batch.id) as unknown as Array<{
      item_amount_usd_micros: number;
      earning_amount_usd_micros: number;
      seller_id: string;
      payout_method: PayoutMethod;
      status: string;
    }>;
    const grossUsdMicros = rows.reduce((sum, row) => sum + Number(row.item_amount_usd_micros), 0);
    const invalid = rows.length === 0 || grossUsdMicros !== batch.grossUsdMicros
      || rows.some((row) => row.seller_id !== batch.sellerId
        || row.payout_method !== batch.payoutMethod
        || row.status !== "batched"
        || Number(row.item_amount_usd_micros) !== Number(row.earning_amount_usd_micros));
    if (invalid) {
      throw new PayoutError("payout_batch_audit_conflict", "The payout batch item audit does not match its sealed economics.");
    }
  }

  private assertSporeQuoteAuditIntegrity(operation: PayoutDispatchOperation): void {
    if (operation.payoutMethod !== "spore") return;
    if (!this.sporeQuotes || !operation.sporeQuoteId || !operation.sporeQuoteAttestationDigest) {
      throw new PayoutError("spore_dispatch_quote_audit_missing", "The SPORE dispatch has no durable quote audit link.");
    }
    const quote = this.sporeQuotes.getStoredByQuoteId(operation.sporeQuoteId);
    const digest = quote ? sporeConversionAttestationDigest(quote) : null;
    if (!quote || quote.batchId !== operation.batchId || digest !== operation.sporeQuoteAttestationDigest) {
      throw new PayoutError("spore_dispatch_quote_audit_conflict", "The SPORE dispatch quote audit link is inconsistent.");
    }
  }

  private requireBatch(batchId: string): PayoutBatch {
    const batch = this.payouts.getBatch(batchId);
    if (!batch) throw new PayoutError("payout_not_found", "The payout batch does not exist.");
    return batch;
  }

  private requireGateway(method: PayoutMethod): PayoutGateway {
    const gateway = this.gateways[method];
    if (!gateway) throw new PayoutError("payout_gateway_unavailable", `No ${method} payout gateway is configured.`);
    return gateway;
  }
}

interface DispatchRow {
  id: string;
  batch_id: string;
  payout_method: PayoutMethod;
  dispatch_key: string;
  state: PayoutDispatchState;
  external_reference: string | null;
  settlement_reference: string | null;
  spore_quote_id: string | null;
  spore_quote_attestation_digest: string | null;
  last_error: string | null;
  created_at: number;
  updated_at: number;
  reconciled_at: number | null;
}

interface SettlementEvidenceRow {
  schema_name: string;
  external_reference: string;
  settlement_reference: string;
  paid_at: number;
  issued_at: number;
  verifier_key_id: string;
  signature: string;
}

function settlementEvidenceMatches(row: SettlementEvidenceRow, evidence: PayoutSettlementAttestation): boolean {
  return row.schema_name === evidence.schema
    && row.external_reference === evidence.externalReference
    && row.settlement_reference === evidence.settlementReference
    && Number(row.paid_at) === evidence.paidAt
    && Number(row.issued_at) === evidence.issuedAt
    && row.verifier_key_id === evidence.verifierKeyId
    && row.signature === evidence.signature;
}

function mapOperation(row: DispatchRow): PayoutDispatchOperation {
  return {
    id: row.id,
    batchId: row.batch_id,
    payoutMethod: row.payout_method,
    dispatchKey: row.dispatch_key,
    state: row.state,
    externalReference: row.external_reference,
    settlementReference: row.settlement_reference,
    sporeQuoteId: row.spore_quote_id,
    sporeQuoteAttestationDigest: row.spore_quote_attestation_digest,
    lastError: row.last_error,
    createdAt: Number(row.created_at),
    updatedAt: Number(row.updated_at),
    reconciledAt: row.reconciled_at === null ? null : Number(row.reconciled_at),
  };
}

function boundedError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return message.slice(0, 500) || "provider_dispatch_failed";
}
