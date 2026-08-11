import type { MeshDatabase } from "../storage/database.js";

export type BillingOperationalAlertCode =
  | "stablecoin_allocation_stuck"
  | "stablecoin_payment_expired"
  | "stablecoin_consumption_orphaned"
  | "payout_dispatch_stuck"
  | "payout_settlement_stuck"
  | "payout_submitted_without_dispatch";

export interface BillingOperationalAlert {
  code: BillingOperationalAlertCode;
  severity: "warning" | "critical";
  resourceType: "stablecoin_intent" | "payout_batch";
  resourceId: string;
  ageMs: number;
}

export interface BillingOperationsSnapshot {
  generatedAt: number;
  stablecoinIntents: {
    allocating: number;
    awaitingPayment: number;
    consumed: number;
    expiredAwaitingPayment: number;
  };
  payoutDispatches: {
    dispatching: number;
    uncertain: number;
    submitted: number;
    paid: number;
    rejected: number;
  };
  settlementCandidates: Array<{
    batchId: string;
    externalReference: string;
    amountUsdMicros: number;
    submittedAt: number;
    ageMs: number;
  }>;
  settlementCandidatesTruncated: boolean;
  sporeQuoteCandidates: Array<{
    batchId: string;
    amountUsdMicros: number;
    preparedAt: number;
    ageMs: number;
    reason: "missing" | "expired";
    quoteExpiresAt: number | null;
  }>;
  sporeQuoteCandidatesTruncated: boolean;
  alerts: BillingOperationalAlert[];
  alertsTruncated: boolean;
}

export interface BillingOperationsPolicy {
  stablecoinAllocationStuckMs: number;
  payoutDispatchStuckMs: number;
  payoutSettlementStuckMs: number;
  maxAlerts?: number;
  now?: () => number;
}

export class BillingOperationsMonitor {
  private readonly maxAlerts: number;

  constructor(
    private readonly database: MeshDatabase,
    private readonly policy: BillingOperationsPolicy,
  ) {
    for (const [name, value] of [
      ["stablecoinAllocationStuckMs", policy.stablecoinAllocationStuckMs],
      ["payoutDispatchStuckMs", policy.payoutDispatchStuckMs],
      ["payoutSettlementStuckMs", policy.payoutSettlementStuckMs],
    ] as const) {
      if (!Number.isSafeInteger(value) || value <= 0) {
        throw new Error(`invalid_billing_operations_policy:${name}`);
      }
    }
    this.maxAlerts = policy.maxAlerts ?? 100;
    if (!Number.isSafeInteger(this.maxAlerts) || this.maxAlerts < 1 || this.maxAlerts > 1_000) {
      throw new Error("invalid_billing_operations_policy:maxAlerts");
    }
  }

  snapshot(): BillingOperationsSnapshot {
    const now = this.policy.now?.() ?? Date.now();
    const intentCounts = statusCounts(
      this.database,
      "billing_stablecoin_intents",
      "status",
      ["allocating", "awaiting_payment", "consumed"] as const,
    );
    const dispatchCounts = statusCounts(
      this.database,
      "payout_dispatch_operations",
      "state",
      ["dispatching", "uncertain", "submitted", "paid", "rejected"] as const,
    );
    const alerts = this.collectAlerts(now);
    const candidateRows = this.database.raw.prepare(
      `SELECT d.batch_id, d.external_reference, b.amount_usd_micros, b.submitted_at
       FROM payout_dispatch_operations d
       JOIN payout_batches b ON b.id = d.batch_id
       LEFT JOIN payout_settlement_evidence e ON e.batch_id = d.batch_id
       WHERE d.state = 'submitted' AND d.external_reference IS NOT NULL
         AND b.status = 'submitted' AND e.batch_id IS NULL
       ORDER BY b.submitted_at ASC, d.batch_id ASC LIMIT ?`,
    ).all(this.maxAlerts + 1) as unknown as Array<{
      batch_id: string; external_reference: string;
      amount_usd_micros: number; submitted_at: number;
    }>;
    const sporeQuoteRows = this.database.raw.prepare(
      `SELECT b.id AS batch_id, b.amount_usd_micros, b.created_at,
              q.quote_id, q.expires_at
       FROM payout_batches b
       LEFT JOIN spore_conversion_quotes q
         ON q.batch_id = b.id AND q.status = 'active'
       WHERE b.payout_method = 'spore' AND b.status = 'prepared'
         AND (q.quote_id IS NULL OR q.expires_at <= ?)
       ORDER BY b.created_at ASC, b.id ASC LIMIT ?`,
    ).all(now, this.maxAlerts + 1) as unknown as Array<{
      batch_id: string; amount_usd_micros: number; created_at: number;
      quote_id: string | null; expires_at: number | null;
    }>;
    return {
      generatedAt: now,
      stablecoinIntents: {
        allocating: intentCounts.allocating,
        awaitingPayment: intentCounts.awaiting_payment,
        consumed: intentCounts.consumed,
        expiredAwaitingPayment: scalarCount(
          this.database,
          "SELECT COUNT(*) AS count FROM billing_stablecoin_intents WHERE status = 'awaiting_payment' AND expires_at < ?",
          now,
        ),
      },
      payoutDispatches: {
        dispatching: dispatchCounts.dispatching,
        uncertain: dispatchCounts.uncertain,
        submitted: dispatchCounts.submitted,
        paid: dispatchCounts.paid,
        rejected: dispatchCounts.rejected,
      },
      settlementCandidates: candidateRows.slice(0, this.maxAlerts).map((row) => ({
        batchId: row.batch_id,
        externalReference: row.external_reference,
        amountUsdMicros: Number(row.amount_usd_micros),
        submittedAt: Number(row.submitted_at),
        ageMs: Math.max(0, now - Number(row.submitted_at)),
      })),
      settlementCandidatesTruncated: candidateRows.length > this.maxAlerts,
      sporeQuoteCandidates: sporeQuoteRows.slice(0, this.maxAlerts).map((row) => ({
        batchId: row.batch_id,
        amountUsdMicros: Number(row.amount_usd_micros),
        preparedAt: Number(row.created_at),
        ageMs: Math.max(0, now - Number(row.created_at)),
        reason: row.quote_id === null ? "missing" : "expired",
        quoteExpiresAt: row.expires_at === null ? null : Number(row.expires_at),
      })),
      sporeQuoteCandidatesTruncated: sporeQuoteRows.length > this.maxAlerts,
      alerts: alerts.slice(0, this.maxAlerts),
      alertsTruncated: alerts.length > this.maxAlerts,
    };
  }

  private collectAlerts(now: number): BillingOperationalAlert[] {
    const alerts: BillingOperationalAlert[] = [];
    const addRows = (
      rows: Array<{ id: string; timestamp: number }>,
      code: BillingOperationalAlertCode,
      severity: BillingOperationalAlert["severity"],
      resourceType: BillingOperationalAlert["resourceType"],
    ) => {
      for (const row of rows) {
        alerts.push({
          code,
          severity,
          resourceType,
          resourceId: row.id,
          ageMs: Math.max(0, now - Number(row.timestamp)),
        });
      }
    };

    addRows(this.database.raw.prepare(
      `SELECT id, updated_at AS timestamp FROM billing_stablecoin_intents
       WHERE status = 'allocating' AND updated_at <= ? ORDER BY updated_at ASC`,
    ).all(now - this.policy.stablecoinAllocationStuckMs) as unknown as Array<{ id: string; timestamp: number }>,
    "stablecoin_allocation_stuck", "critical", "stablecoin_intent");

    addRows(this.database.raw.prepare(
      `SELECT id, expires_at AS timestamp FROM billing_stablecoin_intents
       WHERE status = 'awaiting_payment' AND expires_at < ? ORDER BY expires_at ASC`,
    ).all(now) as unknown as Array<{ id: string; timestamp: number }>,
    "stablecoin_payment_expired", "warning", "stablecoin_intent");

    addRows(this.database.raw.prepare(
      `SELECT i.id, i.consumed_at AS timestamp FROM billing_stablecoin_intents i
       LEFT JOIN billing_events e
         ON i.consumed_event_key = ('stablecoin:' || e.provider_event_id)
        AND e.provider = 'stablecoin'
       WHERE i.status = 'consumed' AND e.id IS NULL ORDER BY i.consumed_at ASC`,
    ).all() as unknown as Array<{ id: string; timestamp: number }>,
    "stablecoin_consumption_orphaned", "critical", "stablecoin_intent");

    addRows(this.database.raw.prepare(
      `SELECT batch_id AS id, updated_at AS timestamp FROM payout_dispatch_operations
       WHERE state IN ('dispatching', 'uncertain') AND updated_at <= ? ORDER BY updated_at ASC`,
    ).all(now - this.policy.payoutDispatchStuckMs) as unknown as Array<{ id: string; timestamp: number }>,
    "payout_dispatch_stuck", "critical", "payout_batch");

    addRows(this.database.raw.prepare(
      `SELECT batch_id AS id, updated_at AS timestamp FROM payout_dispatch_operations
       WHERE state = 'submitted' AND updated_at <= ? ORDER BY updated_at ASC`,
    ).all(now - this.policy.payoutSettlementStuckMs) as unknown as Array<{ id: string; timestamp: number }>,
    "payout_settlement_stuck", "warning", "payout_batch");

    addRows(this.database.raw.prepare(
      `SELECT b.id, b.updated_at AS timestamp FROM payout_batches b
       LEFT JOIN payout_dispatch_operations d ON d.batch_id = b.id
       WHERE b.status = 'submitted' AND d.id IS NULL ORDER BY b.updated_at ASC`,
    ).all() as unknown as Array<{ id: string; timestamp: number }>,
    "payout_submitted_without_dispatch", "critical", "payout_batch");

    return alerts.sort((left, right) => {
      if (left.severity !== right.severity) return left.severity === "critical" ? -1 : 1;
      if (left.ageMs !== right.ageMs) return right.ageMs - left.ageMs;
      return left.resourceId.localeCompare(right.resourceId);
    });
  }
}

function scalarCount(database: MeshDatabase, sql: string, ...params: Array<string | number | null>): number {
  const row = database.raw.prepare(sql).get(...params) as { count: number };
  return Number(row.count);
}

function statusCounts<const T extends readonly string[]>(
  database: MeshDatabase,
  table: string,
  column: "status" | "state",
  statuses: T,
): Record<T[number], number> {
  const result = Object.fromEntries(statuses.map((status) => [status, 0])) as Record<T[number], number>;
  const rows = database.raw.prepare(
    `SELECT ${column} AS status, COUNT(*) AS count FROM ${table} GROUP BY ${column}`,
  ).all() as unknown as Array<{ status: T[number]; count: number }>;
  for (const row of rows) {
    if (Object.hasOwn(result, row.status)) result[row.status] = Number(row.count);
  }
  return result;
}
