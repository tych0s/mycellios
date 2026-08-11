import { createHash } from "node:crypto";
import { newId } from "../core/ids.js";
import type { MeshDatabase } from "../storage/database.js";
import type { ApiAccessManager } from "./api-access.js";

export type BillingProvider = "stripe" | "stablecoin";
export type BillingEventType = "subscription_paid" | "topup_paid";

export interface BillingPlan {
  planId: string;
  version: number;
  priceCurrency: string;
  priceMicros: number;
  includedTokens: number;
  status: "draft" | "active" | "retired";
  createdAt: number;
}

export interface PaidSubscriptionEvent {
  provider: BillingProvider;
  providerEventId: string;
  providerSubscriptionId: string;
  userId: string;
  planId: string;
  planVersion: number;
  amountMicros: number;
  currency: string;
  periodStart: number;
  periodEnd: number;
  occurredAt: number;
  finalized: boolean;
  stablecoinPaymentIntentId?: string;
  stablecoinChainId?: string;
  stablecoinAsset?: string;
  stablecoinAssetAtomicAmount?: string;
  stablecoinRecipient?: string;
}

export interface PaidTopUpEvent {
  provider: BillingProvider;
  providerEventId: string;
  paymentReference: string;
  quoteId: string;
  userId: string;
  amountMicros: number;
  currency: string;
  tokenAmount: number;
  occurredAt: number;
  finalized: boolean;
  stablecoinPaymentIntentId?: string;
  stablecoinChainId?: string;
  stablecoinAsset?: string;
  stablecoinAssetAtomicAmount?: string;
  stablecoinRecipient?: string;
}

export interface TopUpQuote {
  quoteId: string;
  userId: string;
  amountMicros: number;
  currency: string;
  tokenAmount: number;
  expiresAt: number;
  createdAt: number;
}

export interface BillingApplicationResult {
  eventId: string;
  grantId: string;
  tokenAmount: number;
  debtRepaidTokens: number;
  tokenBalance: number;
  duplicate: boolean;
}

export interface PaymentReversalEvent {
  provider: BillingProvider;
  providerEventId: string;
  originalProviderEventId: string;
  userId: string;
  reason: "refund" | "dispute" | "stablecoin_reorg";
  tokenAmount: number;
  occurredAt: number;
  finalized: boolean;
}

export interface BillingReversalResult {
  reversalId: string;
  recoveredTokens: number;
  debtTokens: number;
  tokenBalance: number;
  totalTokenDebt: number;
  duplicate: boolean;
}

export interface SubscriptionStatusEvent {
  provider: BillingProvider;
  providerEventId: string;
  providerSubscriptionId: string;
  userId: string;
  status: "past_due" | "cancelled";
  occurredAt: number;
  finalized: boolean;
}

export interface SubscriptionStatusResult {
  eventId: string;
  status: "past_due" | "cancelled";
  applied: boolean;
  duplicate: boolean;
}

export interface BillingSubscription {
  id: string;
  planId: string;
  planVersion: number;
  provider: BillingProvider;
  status: "active" | "past_due" | "cancelled";
  periodStart: number;
  periodEnd: number;
  statusChangedAt: number;
  createdAt: number;
  updatedAt: number;
}

interface BillingEventRow {
  id: string;
  event_digest: string;
}

interface CreditGrantRow {
  id: string;
  token_amount: number;
  debt_repaid_tokens: number;
}

export class BillingError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "BillingError";
  }
}

export class BillingManager {
  constructor(
    private readonly database: MeshDatabase,
    private readonly apiAccess: ApiAccessManager,
  ) {}

  registerPlan(input: Omit<BillingPlan, "createdAt">): BillingPlan {
    validateIdentifier(input.planId, "plan_id");
    validatePositiveInteger(input.version, "plan version");
    validateCurrency(input.priceCurrency);
    validatePositiveInteger(input.priceMicros, "plan price");
    validatePositiveInteger(input.includedTokens, "included tokens");
    return this.database.transaction(() => {
      const existing = this.getPlan(input.planId, input.version);
      if (existing) {
        const matches = existing.priceCurrency === input.priceCurrency.toUpperCase()
          && existing.priceMicros === input.priceMicros
          && existing.includedTokens === input.includedTokens
          && existing.status === input.status;
        if (!matches) {
          throw new BillingError("plan_version_conflict", "A plan version is immutable once registered.");
        }
        return existing;
      }
      const createdAt = Date.now();
      this.database.raw.prepare(
        `INSERT INTO billing_plans(
           plan_id, version, price_currency, price_micros, included_tokens, status, created_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        input.planId,
        input.version,
        input.priceCurrency.toUpperCase(),
        input.priceMicros,
        input.includedTokens,
        input.status,
        createdAt,
      );
      return { ...input, priceCurrency: input.priceCurrency.toUpperCase(), createdAt };
    });
  }

  createTopUpQuote(input: Omit<TopUpQuote, "createdAt">): TopUpQuote {
    validateIdentifier(input.quoteId, "quote id");
    validateIdentifier(input.userId, "user id");
    validatePositiveInteger(input.amountMicros, "quote amount");
    validateCurrency(input.currency);
    validatePositiveInteger(input.tokenAmount, "quote tokens");
    if (!Number.isSafeInteger(input.expiresAt) || input.expiresAt <= Date.now()) {
      throw new BillingError("invalid_quote_expiry", "A top-up quote must expire in the future.");
    }
    return this.database.transaction(() => {
      this.apiAccess.getOrCreateAccount(input.userId);
      const existing = this.database.raw.prepare(
        `SELECT user_id, amount_micros, currency, token_amount, expires_at, created_at
         FROM billing_topup_quotes WHERE quote_id = ?`,
      ).get(input.quoteId) as {
        user_id: string; amount_micros: number; currency: string;
        token_amount: number; expires_at: number; created_at: number;
      } | undefined;
      if (existing) {
        const matches = existing.user_id === input.userId
          && Number(existing.amount_micros) === input.amountMicros
          && existing.currency === input.currency.toUpperCase()
          && Number(existing.token_amount) === input.tokenAmount
          && Number(existing.expires_at) === input.expiresAt;
        if (!matches) throw new BillingError("topup_quote_conflict", "A quote id is immutable once created.");
        return {
          ...input,
          currency: input.currency.toUpperCase(),
          createdAt: Number(existing.created_at),
        };
      }
      const createdAt = Date.now();
      this.database.raw.prepare(
        `INSERT INTO billing_topup_quotes(
           quote_id, user_id, amount_micros, currency, token_amount,
           status, expires_at, created_at
         ) VALUES (?, ?, ?, ?, ?, 'open', ?, ?)`,
      ).run(
        input.quoteId, input.userId, input.amountMicros, input.currency.toUpperCase(),
        input.tokenAmount, input.expiresAt, createdAt,
      );
      return { ...input, currency: input.currency.toUpperCase(), createdAt };
    });
  }

  applyPaidSubscription(input: PaidSubscriptionEvent): BillingApplicationResult {
    validatePaidEvent(input);
    this.validateStablecoinIntent(input, "subscription");
    validateIdentifier(input.providerSubscriptionId, "provider subscription id");
    validateIdentifier(input.planId, "plan id");
    validatePositiveInteger(input.planVersion, "plan version");
    if (!Number.isSafeInteger(input.periodStart) || input.periodEnd <= input.periodStart) {
      throw new BillingError("invalid_billing_period", "The billing period must have an ordered start and end.");
    }
    const plan = this.getPlan(input.planId, input.planVersion);
    if (!plan || plan.status !== "active") {
      throw new BillingError("plan_not_active", "The paid event references a plan version that is not active.");
    }
    if (plan.priceCurrency !== input.currency.toUpperCase() || plan.priceMicros !== input.amountMicros) {
      throw new BillingError("plan_price_mismatch", "The paid amount does not match the immutable plan version.");
    }
    const digest = eventDigest("subscription_paid", input);
    return this.applyGrant({
      provider: input.provider,
      providerEventId: input.providerEventId,
      eventType: "subscription_paid",
      userId: input.userId,
      digest,
      externalReference: input.providerSubscriptionId,
      amountMicros: input.amountMicros,
      currency: input.currency,
      occurredAt: input.occurredAt,
      tokenAmount: plan.includedTokens,
      sourceType: "subscription",
      onFirstApplication: (now) => {
        const existing = this.database.raw.prepare(
          `SELECT id, user_id, status_changed_at FROM billing_subscriptions
           WHERE provider = ? AND provider_subscription_id = ?`,
        ).get(input.provider, input.providerSubscriptionId) as {
          id: string; user_id: string; status_changed_at: number;
        } | undefined;
        if (existing && existing.user_id !== input.userId) {
          throw new BillingError("subscription_owner_conflict", "A provider subscription cannot move between accounts.");
        }
        if (existing && input.occurredAt >= Number(existing.status_changed_at)) {
          this.database.raw.prepare(
            `UPDATE billing_subscriptions
             SET plan_id = ?, plan_version = ?, status = 'active', period_start = ?, period_end = ?,
                 status_changed_at = ?, updated_at = ?
             WHERE id = ?`,
          ).run(
            input.planId, input.planVersion, input.periodStart, input.periodEnd,
            input.occurredAt, now, existing.id,
          );
        } else {
          if (!existing) {
            this.database.raw.prepare(
              `INSERT INTO billing_subscriptions(
                 id, user_id, plan_id, plan_version, provider, provider_subscription_id,
                 status, period_start, period_end, status_changed_at, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)`,
            ).run(
              newId("sub"), input.userId, input.planId, input.planVersion, input.provider,
              input.providerSubscriptionId, input.periodStart, input.periodEnd,
              input.occurredAt, now, now,
            );
          }
        }
        this.consumeStablecoinIntent(input, now);
      },
    });
  }

  applyPaidTopUp(input: PaidTopUpEvent): BillingApplicationResult {
    validatePaidEvent(input);
    this.validateStablecoinIntent(input, "topup");
    validateIdentifier(input.paymentReference, "payment reference");
    validateIdentifier(input.quoteId, "quote id");
    validatePositiveInteger(input.tokenAmount, "top-up tokens");
    const quote = this.database.raw.prepare(
      `SELECT user_id, amount_micros, currency, token_amount, status,
              consumed_event_key, expires_at
       FROM billing_topup_quotes WHERE quote_id = ?`,
    ).get(input.quoteId) as {
      user_id: string; amount_micros: number; currency: string; token_amount: number;
      status: "open" | "consumed"; consumed_event_key: string | null; expires_at: number;
    } | undefined;
    if (!quote) throw new BillingError("topup_quote_not_found", "The paid top-up quote does not exist.");
    const eventKey = `${input.provider}:${input.providerEventId}`;
    if (quote.user_id !== input.userId) {
      throw new BillingError("topup_quote_owner_conflict", "A top-up quote cannot move between accounts.");
    }
    if (
      Number(quote.amount_micros) !== input.amountMicros
      || quote.currency !== input.currency.toUpperCase()
      || Number(quote.token_amount) !== input.tokenAmount
    ) {
      throw new BillingError("topup_quote_value_mismatch", "The paid event does not match the immutable quote.");
    }
    if (input.occurredAt > Number(quote.expires_at)) {
      throw new BillingError("topup_quote_expired", "The payment occurred after the quote expired.");
    }
    if (quote.status === "consumed" && quote.consumed_event_key !== eventKey) {
      throw new BillingError("topup_quote_already_consumed", "The quote was already consumed by another payment.");
    }
    const digest = eventDigest("topup_paid", input);
    return this.applyGrant({
      provider: input.provider,
      providerEventId: input.providerEventId,
      eventType: "topup_paid",
      userId: input.userId,
      digest,
      externalReference: input.paymentReference,
      amountMicros: input.amountMicros,
      currency: input.currency,
      occurredAt: input.occurredAt,
      tokenAmount: input.tokenAmount,
      sourceType: "topup",
      onFirstApplication: (now) => {
        const result = this.database.raw.prepare(
          `UPDATE billing_topup_quotes
           SET status = 'consumed', consumed_event_key = ?, consumed_at = ?
           WHERE quote_id = ? AND status = 'open'`,
        ).run(eventKey, now, input.quoteId);
        if (Number(result.changes) !== 1) {
          throw new BillingError("topup_quote_consume_conflict", "The quote could not be consumed atomically.");
        }
        this.consumeStablecoinIntent(input, now);
      },
    });
  }

  applyPaymentReversal(input: PaymentReversalEvent): BillingReversalResult {
    validateProvider(input.provider);
    validateIdentifier(input.providerEventId, "provider event id");
    validateIdentifier(input.originalProviderEventId, "original provider event id");
    validateIdentifier(input.userId, "user id");
    validatePositiveInteger(input.tokenAmount, "reversed token amount");
    if (!Number.isSafeInteger(input.occurredAt) || input.occurredAt <= 0) {
      throw new BillingError("invalid_occurred_at", "The reversal timestamp is invalid.");
    }
    if (!input.finalized) {
      throw new BillingError("reversal_not_finalized", "Only finalized reversals may change credit state.");
    }
    if (input.reason === "stablecoin_reorg" && input.provider !== "stablecoin") {
      throw new BillingError("invalid_reversal_reason", "A stablecoin reorg must come from the stablecoin provider.");
    }
    const digest = eventDigest("payment_reversed", input);

    return this.database.transaction(() => {
      const paidCollision = this.database.raw.prepare(
        "SELECT 1 AS found FROM billing_events WHERE provider = ? AND provider_event_id = ?",
      ).get(input.provider, input.providerEventId) as { found: number } | undefined;
      if (paidCollision) {
        throw new BillingError("billing_event_conflict", "A provider event id is already used by a paid event.");
      }
      const statusCollision = this.database.raw.prepare(
        "SELECT 1 AS found FROM billing_subscription_events WHERE provider = ? AND provider_event_id = ?",
      ).get(input.provider, input.providerEventId) as { found: number } | undefined;
      if (statusCollision) {
        throw new BillingError("billing_event_conflict", "A provider event id is already used by a subscription status event.");
      }
      const existing = this.database.raw.prepare(
        `SELECT id, event_digest, recovered_tokens, debt_tokens
         FROM billing_reversals WHERE provider = ? AND provider_event_id = ?`,
      ).get(input.provider, input.providerEventId) as {
        id: string; event_digest: string; recovered_tokens: number; debt_tokens: number;
      } | undefined;
      if (existing) {
        if (existing.event_digest !== digest) {
          throw new BillingError("billing_event_conflict", "A reversal event id was reused with different economic content.");
        }
        const account = this.apiAccess.getOrCreateAccount(input.userId);
        return {
          reversalId: existing.id,
          recoveredTokens: Number(existing.recovered_tokens),
          debtTokens: Number(existing.debt_tokens),
          tokenBalance: account.tokenBalance,
          totalTokenDebt: this.getTokenDebt(input.userId),
          duplicate: true,
        };
      }

      const original = this.database.raw.prepare(
        `SELECT id, user_id FROM billing_events
         WHERE provider = ? AND provider_event_id = ?`,
      ).get(input.provider, input.originalProviderEventId) as { id: string; user_id: string } | undefined;
      if (!original) {
        throw new BillingError("original_payment_not_found", "The reversed paid event does not exist.");
      }
      if (original.user_id !== input.userId) {
        throw new BillingError("reversal_owner_conflict", "A payment reversal cannot move between accounts.");
      }
      const grant = this.database.raw.prepare(
        "SELECT id, token_amount FROM credit_grants WHERE source_event_id = ?",
      ).get(original.id) as unknown as { id: string; token_amount: number } | undefined;
      if (!grant) {
        throw new BillingError("billing_ledger_incomplete", "The original paid event has no credit grant.");
      }
      const reversed = this.database.raw.prepare(
        "SELECT COALESCE(SUM(token_amount), 0) AS total FROM billing_reversals WHERE original_event_id = ?",
      ).get(original.id) as { total: number };
      const remaining = Number(grant.token_amount) - Number(reversed.total);
      if (input.tokenAmount > remaining) {
        throw new BillingError("reversal_exceeds_grant", "The reversal exceeds the unreversed part of the original grant.");
      }

      const account = this.apiAccess.getOrCreateAccount(input.userId);
      const recoveredTokens = Math.min(account.tokenBalance, input.tokenAmount);
      const debtTokens = input.tokenAmount - recoveredTokens;
      const now = Date.now();
      const reversalId = newId("reverse");
      this.database.raw.prepare(
        `INSERT INTO billing_reversals(
           id, provider, provider_event_id, original_event_id, user_id, reason,
           event_digest, token_amount, recovered_tokens, debt_tokens, occurred_at, processed_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        reversalId, input.provider, input.providerEventId, original.id, input.userId,
        input.reason, digest, input.tokenAmount, recoveredTokens, debtTokens,
        input.occurredAt, now,
      );
      this.database.raw.prepare(
        "UPDATE api_accounts SET token_balance = token_balance - ?, updated_at = ? WHERE user_id = ?",
      ).run(recoveredTokens, now, input.userId);
      if (debtTokens > 0) {
        this.database.raw.prepare(
          `INSERT INTO account_credit_debts(user_id, token_debt, updated_at)
           VALUES (?, ?, ?)
           ON CONFLICT(user_id) DO UPDATE
           SET token_debt = token_debt + excluded.token_debt, updated_at = excluded.updated_at`,
        ).run(input.userId, debtTokens, now);
      }
      return {
        reversalId,
        recoveredTokens,
        debtTokens,
        tokenBalance: account.tokenBalance - recoveredTokens,
        totalTokenDebt: this.getTokenDebt(input.userId),
        duplicate: false,
      };
    });
  }

  applySubscriptionStatus(input: SubscriptionStatusEvent): SubscriptionStatusResult {
    validateProvider(input.provider);
    validateIdentifier(input.providerEventId, "provider event id");
    validateIdentifier(input.providerSubscriptionId, "provider subscription id");
    validateIdentifier(input.userId, "user id");
    if (input.status !== "past_due" && input.status !== "cancelled") {
      throw new BillingError("invalid_subscription_status", "The subscription status event is invalid.");
    }
    if (!Number.isSafeInteger(input.occurredAt) || input.occurredAt <= 0) {
      throw new BillingError("invalid_occurred_at", "The subscription event timestamp is invalid.");
    }
    if (!input.finalized) {
      throw new BillingError("subscription_event_not_finalized", "Only finalized subscription events may change state.");
    }
    const digest = eventDigest("subscription_status", input);

    return this.database.transaction(() => {
      const paidCollision = this.database.raw.prepare(
        "SELECT 1 AS found FROM billing_events WHERE provider = ? AND provider_event_id = ?",
      ).get(input.provider, input.providerEventId) as { found: number } | undefined;
      const reversalCollision = this.database.raw.prepare(
        "SELECT 1 AS found FROM billing_reversals WHERE provider = ? AND provider_event_id = ?",
      ).get(input.provider, input.providerEventId) as { found: number } | undefined;
      if (paidCollision || reversalCollision) {
        throw new BillingError("billing_event_conflict", "A provider event id is already used by another economic event.");
      }
      const existingEvent = this.database.raw.prepare(
        `SELECT id, event_digest, status, applied FROM billing_subscription_events
         WHERE provider = ? AND provider_event_id = ?`,
      ).get(input.provider, input.providerEventId) as {
        id: string; event_digest: string; status: SubscriptionStatusEvent["status"]; applied: number;
      } | undefined;
      if (existingEvent) {
        if (existingEvent.event_digest !== digest) {
          throw new BillingError("billing_event_conflict", "A subscription event id was reused with different content.");
        }
        return {
          eventId: existingEvent.id,
          status: existingEvent.status,
          applied: existingEvent.applied === 1,
          duplicate: true,
        };
      }

      const subscription = this.database.raw.prepare(
        `SELECT id, user_id, status_changed_at FROM billing_subscriptions
         WHERE provider = ? AND provider_subscription_id = ?`,
      ).get(input.provider, input.providerSubscriptionId) as {
        id: string; user_id: string; status_changed_at: number;
      } | undefined;
      if (!subscription) {
        throw new BillingError("subscription_not_found", "The provider subscription does not exist.");
      }
      if (subscription.user_id !== input.userId) {
        throw new BillingError("subscription_owner_conflict", "A subscription event cannot move between accounts.");
      }
      const applied = input.occurredAt >= Number(subscription.status_changed_at);
      const now = Date.now();
      if (applied) {
        this.database.raw.prepare(
          `UPDATE billing_subscriptions
           SET status = ?, status_changed_at = ?, updated_at = ? WHERE id = ?`,
        ).run(input.status, input.occurredAt, now, subscription.id);
      }
      const eventId = newId("subevt");
      this.database.raw.prepare(
        `INSERT INTO billing_subscription_events(
           id, provider, provider_event_id, provider_subscription_id, user_id,
           status, event_digest, occurred_at, processed_at, applied
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        eventId, input.provider, input.providerEventId, input.providerSubscriptionId,
        input.userId, input.status, digest, input.occurredAt, now, applied ? 1 : 0,
      );
      return { eventId, status: input.status, applied, duplicate: false };
    });
  }

  getTokenDebt(userId: string): number {
    const row = this.database.raw.prepare(
      "SELECT token_debt FROM account_credit_debts WHERE user_id = ?",
    ).get(userId) as { token_debt: number } | undefined;
    return Number(row?.token_debt ?? 0);
  }

  getSubscription(userId: string): BillingSubscription | null {
    validateIdentifier(userId, "user id");
    const row = this.database.raw.prepare(
      `SELECT id, plan_id, plan_version, provider, status, period_start,
              period_end, status_changed_at, created_at, updated_at
       FROM billing_subscriptions
       WHERE user_id = ?
       ORDER BY CASE status WHEN 'active' THEN 0 WHEN 'past_due' THEN 1 ELSE 2 END,
                status_changed_at DESC, id DESC
       LIMIT 1`,
    ).get(userId) as {
      id: string; plan_id: string; plan_version: number; provider: BillingProvider;
      status: BillingSubscription["status"]; period_start: number; period_end: number;
      status_changed_at: number; created_at: number; updated_at: number;
    } | undefined;
    return row ? {
      id: row.id,
      planId: row.plan_id,
      planVersion: Number(row.plan_version),
      provider: row.provider,
      status: row.status,
      periodStart: Number(row.period_start),
      periodEnd: Number(row.period_end),
      statusChangedAt: Number(row.status_changed_at),
      createdAt: Number(row.created_at),
      updatedAt: Number(row.updated_at),
    } : null;
  }

  getPlan(planId: string, version: number): BillingPlan | null {
    const row = this.database.raw.prepare(
      `SELECT plan_id, version, price_currency, price_micros, included_tokens, status, created_at
       FROM billing_plans WHERE plan_id = ? AND version = ?`,
    ).get(planId, version) as {
      plan_id: string; version: number; price_currency: string; price_micros: number;
      included_tokens: number; status: BillingPlan["status"]; created_at: number;
    } | undefined;
    return row ? {
      planId: row.plan_id,
      version: Number(row.version),
      priceCurrency: row.price_currency,
      priceMicros: Number(row.price_micros),
      includedTokens: Number(row.included_tokens),
      status: row.status,
      createdAt: Number(row.created_at),
    } : null;
  }

  private validateStablecoinIntent(
    input: PaidSubscriptionEvent | PaidTopUpEvent,
    kind: "subscription" | "topup",
  ): void {
    const binding = [
      input.stablecoinPaymentIntentId,
      input.stablecoinChainId,
      input.stablecoinAsset,
      input.stablecoinAssetAtomicAmount,
      input.stablecoinRecipient,
    ];
    if (input.provider !== "stablecoin") {
      if (binding.some((value) => value !== undefined)) {
        throw new BillingError("unexpected_stablecoin_binding", "Stripe events cannot carry stablecoin intent fields.");
      }
      return;
    }
    if (binding.some((value) => typeof value !== "string" || value.length === 0)) {
      throw new BillingError("stablecoin_intent_required", "Stablecoin payments require a complete server-side intent binding.");
    }
    const intent = this.database.raw.prepare(
      `SELECT user_id, kind, quote_id, plan_id, plan_version, subscription_id,
              period_duration_ms, amount_micros, currency, chain_id, asset,
              asset_atomic_amount, recipient, status, consumed_event_key, expires_at
       FROM billing_stablecoin_intents WHERE id = ?`,
    ).get(input.stablecoinPaymentIntentId!) as {
      user_id: string; kind: "subscription" | "topup"; quote_id: string | null;
      plan_id: string | null; plan_version: number | null; subscription_id: string | null;
      period_duration_ms: number | null; amount_micros: number; currency: string;
      chain_id: string; asset: string; asset_atomic_amount: string; recipient: string | null;
      status: "allocating" | "awaiting_payment" | "consumed";
      consumed_event_key: string | null; expires_at: number;
    } | undefined;
    if (!intent) throw new BillingError("stablecoin_intent_not_found", "The stablecoin payment intent does not exist.");
    const eventKey = `stablecoin:${input.providerEventId}`;
    const commonMatches = intent.user_id === input.userId
      && intent.kind === kind
      && Number(intent.amount_micros) === input.amountMicros
      && intent.currency === input.currency.toUpperCase()
      && intent.chain_id === input.stablecoinChainId
      && intent.asset === input.stablecoinAsset!.toUpperCase()
      && intent.asset_atomic_amount === input.stablecoinAssetAtomicAmount
      && intent.recipient === input.stablecoinRecipient;
    const economicMatches = kind === "topup"
      ? intent.quote_id === (input as PaidTopUpEvent).quoteId
      : intent.plan_id === (input as PaidSubscriptionEvent).planId
        && Number(intent.plan_version) === (input as PaidSubscriptionEvent).planVersion
        && intent.subscription_id === (input as PaidSubscriptionEvent).providerSubscriptionId
        && (input as PaidSubscriptionEvent).periodStart === input.occurredAt
        && (input as PaidSubscriptionEvent).periodEnd - (input as PaidSubscriptionEvent).periodStart
          === Number(intent.period_duration_ms);
    if (!commonMatches || !economicMatches) {
      throw new BillingError("stablecoin_intent_mismatch", "The watcher event does not match the server-side payment intent.");
    }
    if (input.occurredAt > Number(intent.expires_at)) {
      throw new BillingError("stablecoin_intent_expired", "The stablecoin transfer occurred after the payment intent expired.");
    }
    if (intent.status === "allocating") {
      throw new BillingError("stablecoin_intent_not_ready", "The stablecoin payment intent has no allocated recipient.");
    }
    if (intent.status === "consumed" && intent.consumed_event_key !== eventKey) {
      throw new BillingError("stablecoin_intent_consumed", "The stablecoin payment intent was already consumed by another transfer.");
    }
  }

  private consumeStablecoinIntent(
    input: PaidSubscriptionEvent | PaidTopUpEvent,
    now: number,
  ): void {
    if (input.provider !== "stablecoin") return;
    const result = this.database.raw.prepare(
      `UPDATE billing_stablecoin_intents
       SET status = 'consumed', consumed_event_key = ?, consumed_at = ?, updated_at = ?
       WHERE id = ? AND status = 'awaiting_payment'`,
    ).run(
      `stablecoin:${input.providerEventId}`,
      now,
      now,
      input.stablecoinPaymentIntentId!,
    );
    if (Number(result.changes) !== 1) {
      throw new BillingError("stablecoin_intent_consume_conflict", "The stablecoin payment intent could not be consumed atomically.");
    }
  }

  private applyGrant(input: {
    provider: BillingProvider;
    providerEventId: string;
    eventType: BillingEventType;
    userId: string;
    digest: string;
    externalReference: string;
    amountMicros: number;
    currency: string;
    occurredAt: number;
    tokenAmount: number;
    sourceType: "subscription" | "topup";
    onFirstApplication: (now: number) => void;
  }): BillingApplicationResult {
    return this.database.transaction(() => {
      const existingEvent = this.database.raw.prepare(
        `SELECT id, event_digest FROM billing_events
         WHERE provider = ? AND provider_event_id = ?`,
      ).get(input.provider, input.providerEventId) as BillingEventRow | undefined;
      const reversalCollision = this.database.raw.prepare(
        "SELECT 1 AS found FROM billing_reversals WHERE provider = ? AND provider_event_id = ?",
      ).get(input.provider, input.providerEventId) as { found: number } | undefined;
      if (reversalCollision) {
        throw new BillingError("billing_event_conflict", "A provider event id is already used by a reversal.");
      }
      const statusCollision = this.database.raw.prepare(
        "SELECT 1 AS found FROM billing_subscription_events WHERE provider = ? AND provider_event_id = ?",
      ).get(input.provider, input.providerEventId) as { found: number } | undefined;
      if (statusCollision) {
        throw new BillingError("billing_event_conflict", "A provider event id is already used by a subscription status event.");
      }
      if (existingEvent) {
        if (existingEvent.event_digest !== input.digest) {
          throw new BillingError("billing_event_conflict", "A provider event id was reused with different economic content.");
        }
        const grant = this.database.raw.prepare(
          "SELECT id, token_amount, debt_repaid_tokens FROM credit_grants WHERE source_event_id = ?",
        ).get(existingEvent.id) as unknown as CreditGrantRow | undefined;
        if (!grant) {
          throw new BillingError(
            "billing_ledger_incomplete",
            "The billing event exists without its atomic credit grant.",
          );
        }
        const account = this.apiAccess.getOrCreateAccount(input.userId);
        return {
          eventId: existingEvent.id,
          grantId: grant.id,
          tokenAmount: Number(grant.token_amount),
          debtRepaidTokens: Number(grant.debt_repaid_tokens),
          tokenBalance: account.tokenBalance,
          duplicate: true,
        };
      }

      const now = Date.now();
      this.apiAccess.getOrCreateAccount(input.userId);
      const eventId = newId("bill");
      this.database.raw.prepare(
        `INSERT INTO billing_events(
           id, provider, provider_event_id, event_type, user_id, event_digest,
           external_reference, amount_micros, currency, occurred_at, processed_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        eventId, input.provider, input.providerEventId, input.eventType, input.userId,
        input.digest, input.externalReference, input.amountMicros,
        input.currency.toUpperCase(), input.occurredAt, now,
      );
      input.onFirstApplication(now);
      const debt = this.getTokenDebt(input.userId);
      const debtRepaidTokens = Math.min(debt, input.tokenAmount);
      const availableTokens = input.tokenAmount - debtRepaidTokens;
      const grantId = newId("grant");
      this.database.raw.prepare(
        `INSERT INTO credit_grants(
           id, user_id, idempotency_key, source_type, source_event_id,
           token_amount, debt_repaid_tokens, created_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        grantId, input.userId, `${input.provider}:${input.providerEventId}`,
        input.sourceType, eventId, input.tokenAmount, debtRepaidTokens, now,
      );
      if (debtRepaidTokens > 0) {
        this.database.raw.prepare(
          "UPDATE account_credit_debts SET token_debt = token_debt - ?, updated_at = ? WHERE user_id = ?",
        ).run(debtRepaidTokens, now, input.userId);
      }
      this.database.raw.prepare(
        "UPDATE api_accounts SET token_balance = token_balance + ?, updated_at = ? WHERE user_id = ?",
      ).run(availableTokens, now, input.userId);
      const account = this.apiAccess.getOrCreateAccount(input.userId);
      return {
        eventId,
        grantId,
        tokenAmount: input.tokenAmount,
        debtRepaidTokens,
        tokenBalance: account.tokenBalance,
        duplicate: false,
      };
    });
  }
}

function validatePaidEvent(input: {
  provider: BillingProvider;
  providerEventId: string;
  userId: string;
  amountMicros: number;
  currency: string;
  occurredAt: number;
  finalized: boolean;
}): void {
  validateProvider(input.provider);
  validateIdentifier(input.providerEventId, "provider event id");
  validateIdentifier(input.userId, "user id");
  validatePositiveInteger(input.amountMicros, "paid amount");
  validateCurrency(input.currency);
  if (!Number.isSafeInteger(input.occurredAt) || input.occurredAt <= 0) {
    throw new BillingError("invalid_occurred_at", "The provider event timestamp is invalid.");
  }
  if (!input.finalized) {
    throw new BillingError("payment_not_finalized", "Only finalized payments may create entitlements or grants.");
  }
}

function validateProvider(provider: BillingProvider): void {
  if (provider !== "stripe" && provider !== "stablecoin") {
    throw new BillingError("unsupported_billing_provider", "The billing provider is not supported.");
  }
}

function validateIdentifier(value: string, label: string): void {
  if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$/.test(value)) {
    throw new BillingError("invalid_identifier", `The ${label} is invalid.`);
  }
}

function validateCurrency(value: string): void {
  if (!/^[A-Z]{3,8}$/.test(value.toUpperCase())) {
    throw new BillingError("invalid_currency", "The currency must be a canonical uppercase code.");
  }
}

function validatePositiveInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new BillingError("invalid_amount", `The ${label} must be a positive safe integer.`);
  }
}

function eventDigest(
  type: BillingEventType | "payment_reversed" | "subscription_status",
  input: object,
): string {
  const normalized = Object.entries(input)
    .filter(([key]) => key !== "finalized")
    .sort(([left], [right]) => left.localeCompare(right));
  return createHash("sha256").update(JSON.stringify([type, normalized]), "utf8").digest("hex");
}
