import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ApiAccessManager } from "../src/coordinator/api-access.js";
import { BillingError, BillingManager } from "../src/coordinator/billing.js";
import { MeshDatabase } from "../src/storage/database.js";

describe("provider-neutral billing ledger", () => {
  let database: MeshDatabase;
  let access: ApiAccessManager;
  let billing: BillingManager;

  beforeEach(() => {
    database = new MeshDatabase(":memory:");
    access = new ApiAccessManager(database, {
      starterTokens: 0,
      requestsPerMinute: 10,
      maxConcurrent: 2,
      maxActiveKeys: 2,
    });
    billing = new BillingManager(database, access);
    billing.registerPlan({
      planId: "mycellios-go",
      version: 1,
      priceCurrency: "EUR",
      priceMicros: 10_000_000,
      includedTokens: 50_000,
      status: "active",
    });
  });

  afterEach(() => database.close());

  function quote(
    quoteId: string,
    userId: string,
    amountMicros: number,
    currency: string,
    tokenAmount: number,
  ) {
    billing.createTopUpQuote({
      quoteId,
      userId,
      amountMicros,
      currency,
      tokenAmount,
      expiresAt: Date.now() + 60_000,
    });
  }

  function stablecoinIntent(
    id: string,
    userId: string,
    quoteId: string,
    amountMicros: number,
    currency: string,
    occurredAt: number,
  ) {
    const recipient = `0xrecipient${id.replaceAll(/[^A-Za-z0-9]/g, "")}`;
    database.raw.prepare(
      `INSERT INTO billing_stablecoin_intents(
         id, user_id, idempotency_key, request_digest, kind, quote_id,
         amount_micros, currency, chain_id, asset, asset_atomic_amount,
         recipient, provider_reference, status, expires_at, created_at, updated_at
       ) VALUES (?, ?, ?, 'fixture', 'topup', ?, ?, ?, 'base', 'USDC', ?, ?, ?,
                 'awaiting_payment', ?, ?, ?)`,
    ).run(
      id, userId, `fixture:${id}`, quoteId, amountMicros, currency,
      String(amountMicros), recipient, `deposit:${id}`, occurredAt + 60_000,
      occurredAt - 1_000, occurredAt - 1_000,
    );
    return {
      stablecoinPaymentIntentId: id,
      stablecoinChainId: "base",
      stablecoinAsset: "USDC",
      stablecoinAssetAtomicAmount: String(amountMicros),
      stablecoinRecipient: recipient,
    };
  }

  it("applies a paid Stripe subscription exactly once", () => {
    const input = {
      provider: "stripe" as const,
      providerEventId: "evt_paid_1",
      providerSubscriptionId: "sub_1",
      userId: "buyer-1",
      planId: "mycellios-go",
      planVersion: 1,
      amountMicros: 10_000_000,
      currency: "EUR",
      periodStart: 1_786_000_000_000,
      periodEnd: 1_788_678_400_000,
      occurredAt: 1_786_000_000_000,
      finalized: true,
    };

    const first = billing.applyPaidSubscription(input);
    const replay = billing.applyPaidSubscription(input);

    expect(first).toMatchObject({ duplicate: false, tokenAmount: 50_000, tokenBalance: 50_000 });
    expect(replay).toMatchObject({
      eventId: first.eventId,
      grantId: first.grantId,
      duplicate: true,
      tokenBalance: 50_000,
    });
    expect(database.raw.prepare("SELECT COUNT(*) AS count FROM billing_events").get()).toEqual({ count: 1 });
    expect(database.raw.prepare("SELECT COUNT(*) AS count FROM credit_grants").get()).toEqual({ count: 1 });
    expect(billing.getSubscription("buyer-1")).toMatchObject({
      planId: "mycellios-go",
      planVersion: 1,
      provider: "stripe",
      status: "active",
      periodStart: input.periodStart,
      periodEnd: input.periodEnd,
    });
  });

  it("returns no subscription for an account without verified payment evidence", () => {
    expect(billing.getSubscription("buyer-without-payment")).toBeNull();
  });

  it("rejects a provider event replay with different economic content", () => {
    quote("quote-conflict", "buyer-2", 5_000_000, "EUR", 20_000);
    const input = {
      provider: "stripe" as const,
      providerEventId: "evt_paid_conflict",
      paymentReference: "pi_1",
      quoteId: "quote-conflict",
      userId: "buyer-2",
      amountMicros: 5_000_000,
      currency: "EUR",
      tokenAmount: 20_000,
      occurredAt: 1_786_000_000_000,
      finalized: true,
    };
    billing.applyPaidTopUp(input);

    expect(() => billing.applyPaidTopUp({ ...input, paymentReference: "pi_2" })).toThrowError(
      expect.objectContaining({ code: "billing_event_conflict" } satisfies Partial<BillingError>),
    );
    expect(access.getOrCreateAccount("buyer-2").tokenBalance).toBe(20_000);
  });

  it("requires stablecoin finality and then grants a top-up once", () => {
    quote("quote-stablecoin", "buyer-3", 20_000_000, "USDC", 80_000);
    const binding = stablecoinIntent(
      "coinpay-stablecoin", "buyer-3", "quote-stablecoin", 20_000_000,
      "USDC", 1_786_000_000_000,
    );
    const input = {
      provider: "stablecoin" as const,
      providerEventId: "base:0xabc:12",
      paymentReference: "0xabc",
      quoteId: "quote-stablecoin",
      userId: "buyer-3",
      amountMicros: 20_000_000,
      currency: "USDC",
      tokenAmount: 80_000,
      occurredAt: 1_786_000_000_000,
      finalized: false,
      ...binding,
    };
    expect(() => billing.applyPaidTopUp(input)).toThrowError(
      expect.objectContaining({ code: "payment_not_finalized" } satisfies Partial<BillingError>),
    );
    expect(database.raw.prepare("SELECT COUNT(*) AS count FROM billing_events").get()).toEqual({ count: 0 });

    expect(() => billing.applyPaidTopUp({
      ...input,
      finalized: true,
      stablecoinAssetAtomicAmount: "20000001",
    })).toThrowError(expect.objectContaining({ code: "stablecoin_intent_mismatch" }));

    const applied = billing.applyPaidTopUp({ ...input, finalized: true });
    const replay = billing.applyPaidTopUp({ ...input, finalized: true });
    expect(applied).toMatchObject({ duplicate: false, tokenBalance: 80_000 });
    expect(replay).toMatchObject({ duplicate: true, tokenBalance: 80_000 });
  });

  it("keeps plan versions immutable and rejects a price mismatch", () => {
    expect(() => billing.registerPlan({
      planId: "mycellios-go",
      version: 1,
      priceCurrency: "EUR",
      priceMicros: 12_000_000,
      includedTokens: 50_000,
      status: "active",
    })).toThrowError(expect.objectContaining({ code: "plan_version_conflict" }));

    expect(() => billing.applyPaidSubscription({
      provider: "stripe",
      providerEventId: "evt_wrong_price",
      providerSubscriptionId: "sub_wrong_price",
      userId: "buyer-4",
      planId: "mycellios-go",
      planVersion: 1,
      amountMicros: 9_000_000,
      currency: "EUR",
      periodStart: 1_786_000_000_000,
      periodEnd: 1_788_678_400_000,
      occurredAt: 1_786_000_000_000,
      finalized: true,
    })).toThrowError(expect.objectContaining({ code: "plan_price_mismatch" }));
  });

  it("records an append-only refund and turns consumed credit into debt", () => {
    quote("quote-refunded", "buyer-refund", 10_000_000, "EUR", 50_000);
    billing.applyPaidTopUp({
      provider: "stripe",
      providerEventId: "evt_topup_refunded",
      paymentReference: "pi_refunded",
      quoteId: "quote-refunded",
      userId: "buyer-refund",
      amountMicros: 10_000_000,
      currency: "EUR",
      tokenAmount: 50_000,
      occurredAt: 1_786_000_000_000,
      finalized: true,
    });
    database.raw.prepare(
      "UPDATE api_accounts SET token_balance = 10_000 WHERE user_id = 'buyer-refund'",
    ).run();

    const reversalInput = {
      provider: "stripe" as const,
      providerEventId: "evt_refund_1",
      originalProviderEventId: "evt_topup_refunded",
      userId: "buyer-refund",
      reason: "refund" as const,
      tokenAmount: 50_000,
      occurredAt: 1_786_100_000_000,
      finalized: true,
    };
    const reversed = billing.applyPaymentReversal(reversalInput);
    const replay = billing.applyPaymentReversal(reversalInput);

    expect(reversed).toMatchObject({
      recoveredTokens: 10_000,
      debtTokens: 40_000,
      tokenBalance: 0,
      totalTokenDebt: 40_000,
      duplicate: false,
    });
    expect(replay).toMatchObject({
      reversalId: reversed.reversalId,
      tokenBalance: 0,
      totalTokenDebt: 40_000,
      duplicate: true,
    });
    expect(database.raw.prepare("SELECT COUNT(*) AS count FROM billing_reversals").get()).toEqual({ count: 1 });
  });

  it("uses a later grant to repay reversal debt before exposing spendable balance", () => {
    quote("quote-debt-origin", "buyer-debt", 10_000_000, "EUR", 50_000);
    billing.applyPaidTopUp({
      provider: "stripe",
      providerEventId: "evt_debt_origin",
      paymentReference: "pi_debt_origin",
      quoteId: "quote-debt-origin",
      userId: "buyer-debt",
      amountMicros: 10_000_000,
      currency: "EUR",
      tokenAmount: 50_000,
      occurredAt: 1_786_000_000_000,
      finalized: true,
    });
    database.raw.prepare(
      "UPDATE api_accounts SET token_balance = 0 WHERE user_id = 'buyer-debt'",
    ).run();
    billing.applyPaymentReversal({
      provider: "stripe",
      providerEventId: "evt_debt_reversal",
      originalProviderEventId: "evt_debt_origin",
      userId: "buyer-debt",
      reason: "dispute",
      tokenAmount: 50_000,
      occurredAt: 1_786_100_000_000,
      finalized: true,
    });

    quote("quote-debt-next", "buyer-debt", 15_000_000, "USDC", 60_000);
    const debtBinding = stablecoinIntent(
      "coinpay-debt", "buyer-debt", "quote-debt-next", 15_000_000,
      "USDC", 1_786_200_000_000,
    );
    const next = billing.applyPaidTopUp({
      provider: "stablecoin",
      providerEventId: "base:0xdebt:44",
      paymentReference: "0xdebt",
      quoteId: "quote-debt-next",
      userId: "buyer-debt",
      amountMicros: 15_000_000,
      currency: "USDC",
      tokenAmount: 60_000,
      occurredAt: 1_786_200_000_000,
      finalized: true,
      ...debtBinding,
    });

    expect(next).toMatchObject({
      tokenAmount: 60_000,
      debtRepaidTokens: 50_000,
      tokenBalance: 10_000,
    });
    expect(billing.getTokenDebt("buyer-debt")).toBe(0);
  });

  it("fails closed for excessive, conflicting or non-finalized reversals", () => {
    quote("quote-reorg", "buyer-reorg", 10_000_000, "USDC", 30_000);
    const reorgBinding = stablecoinIntent(
      "coinpay-reorg", "buyer-reorg", "quote-reorg", 10_000_000,
      "USDC", 1_786_000_000_000,
    );
    billing.applyPaidTopUp({
      provider: "stablecoin",
      providerEventId: "base:0xreorg:20",
      paymentReference: "0xreorg",
      quoteId: "quote-reorg",
      userId: "buyer-reorg",
      amountMicros: 10_000_000,
      currency: "USDC",
      tokenAmount: 30_000,
      occurredAt: 1_786_000_000_000,
      finalized: true,
      ...reorgBinding,
    });
    const reversal = {
      provider: "stablecoin" as const,
      providerEventId: "base:0xreorged:21",
      originalProviderEventId: "base:0xreorg:20",
      userId: "buyer-reorg",
      reason: "stablecoin_reorg" as const,
      tokenAmount: 30_000,
      occurredAt: 1_786_100_000_000,
      finalized: false,
    };
    expect(() => billing.applyPaymentReversal(reversal)).toThrowError(
      expect.objectContaining({ code: "reversal_not_finalized" }),
    );
    billing.applyPaymentReversal({ ...reversal, tokenAmount: 10_000, finalized: true });
    expect(() => billing.applyPaymentReversal({ ...reversal, tokenAmount: 11_000, finalized: true })).toThrowError(
      expect.objectContaining({ code: "billing_event_conflict" }),
    );
    expect(() => billing.applyPaymentReversal({
      ...reversal,
      providerEventId: "base:0xreorged:22",
      tokenAmount: 21_000,
      finalized: true,
    })).toThrowError(expect.objectContaining({ code: "reversal_exceeds_grant" }));
  });

  it("applies subscription cancellation idempotently and ignores stale status events", () => {
    const paid = {
      provider: "stripe" as const,
      providerEventId: "evt_subscription_initial",
      providerSubscriptionId: "sub_status_1",
      userId: "buyer-status",
      planId: "mycellios-go",
      planVersion: 1,
      amountMicros: 10_000_000,
      currency: "EUR",
      periodStart: 1_786_000_000_000,
      periodEnd: 1_788_678_400_000,
      occurredAt: 1_786_000_000_000,
      finalized: true,
    };
    billing.applyPaidSubscription(paid);
    const cancelled = billing.applySubscriptionStatus({
      provider: "stripe",
      providerEventId: "evt_subscription_cancelled",
      providerSubscriptionId: "sub_status_1",
      userId: "buyer-status",
      status: "cancelled",
      occurredAt: 1_786_200_000_000,
      finalized: true,
    });
    const replay = billing.applySubscriptionStatus({
      provider: "stripe",
      providerEventId: "evt_subscription_cancelled",
      providerSubscriptionId: "sub_status_1",
      userId: "buyer-status",
      status: "cancelled",
      occurredAt: 1_786_200_000_000,
      finalized: true,
    });
    const stale = billing.applySubscriptionStatus({
      provider: "stripe",
      providerEventId: "evt_subscription_stale_past_due",
      providerSubscriptionId: "sub_status_1",
      userId: "buyer-status",
      status: "past_due",
      occurredAt: 1_786_100_000_000,
      finalized: true,
    });

    expect(cancelled).toMatchObject({ status: "cancelled", applied: true, duplicate: false });
    expect(replay).toMatchObject({ eventId: cancelled.eventId, applied: true, duplicate: true });
    expect(stale).toMatchObject({ status: "past_due", applied: false, duplicate: false });
    expect(database.raw.prepare(
      "SELECT status FROM billing_subscriptions WHERE provider_subscription_id = 'sub_status_1'",
    ).get()).toEqual({ status: "cancelled" });

    billing.applyPaidSubscription({
      ...paid,
      providerEventId: "evt_subscription_renewed",
      periodStart: 1_788_678_400_000,
      periodEnd: 1_791_270_400_000,
      occurredAt: 1_788_678_400_000,
    });
    expect(database.raw.prepare(
      "SELECT status FROM billing_subscriptions WHERE provider_subscription_id = 'sub_status_1'",
    ).get()).toEqual({ status: "active" });
  });

  it("rejects non-finalized or cross-account subscription status events", () => {
    billing.applyPaidSubscription({
      provider: "stripe",
      providerEventId: "evt_subscription_owner",
      providerSubscriptionId: "sub_owner",
      userId: "buyer-owner",
      planId: "mycellios-go",
      planVersion: 1,
      amountMicros: 10_000_000,
      currency: "EUR",
      periodStart: 1_786_000_000_000,
      periodEnd: 1_788_678_400_000,
      occurredAt: 1_786_000_000_000,
      finalized: true,
    });
    const status = {
      provider: "stripe" as const,
      providerEventId: "evt_subscription_past_due",
      providerSubscriptionId: "sub_owner",
      userId: "buyer-owner",
      status: "past_due" as const,
      occurredAt: 1_786_100_000_000,
      finalized: false,
    };
    expect(() => billing.applySubscriptionStatus(status)).toThrowError(
      expect.objectContaining({ code: "subscription_event_not_finalized" }),
    );
    expect(() => billing.applySubscriptionStatus({
      ...status,
      providerEventId: "evt_subscription_wrong_owner",
      userId: "buyer-attacker",
      finalized: true,
    })).toThrowError(expect.objectContaining({ code: "subscription_owner_conflict" }));
  });

  it("binds top-ups to immutable, single-use, unexpired quotes", () => {
    const expiresAt = Date.now() + 60_000;
    billing.createTopUpQuote({
      quoteId: "quote-secure",
      userId: "buyer-quote",
      amountMicros: 10_000_000,
      currency: "EUR",
      tokenAmount: 40_000,
      expiresAt,
    });
    expect(() => billing.createTopUpQuote({
      quoteId: "quote-secure",
      userId: "buyer-quote",
      amountMicros: 10_000_000,
      currency: "EUR",
      tokenAmount: 400_000,
      expiresAt,
    })).toThrowError(expect.objectContaining({ code: "topup_quote_conflict" }));

    const paid = {
      provider: "stripe" as const,
      providerEventId: "evt_quote_paid",
      paymentReference: "pi_quote_paid",
      quoteId: "quote-secure",
      userId: "buyer-quote",
      amountMicros: 10_000_000,
      currency: "EUR",
      tokenAmount: 40_000,
      occurredAt: Date.now(),
      finalized: true,
    };
    expect(() => billing.applyPaidTopUp({ ...paid, tokenAmount: 400_000 })).toThrowError(
      expect.objectContaining({ code: "topup_quote_value_mismatch" }),
    );
    billing.applyPaidTopUp(paid);
    expect(() => billing.applyPaidTopUp({
      ...paid,
      providerEventId: "evt_duplicate_quote",
      paymentReference: "pi_duplicate_quote",
    })).toThrowError(expect.objectContaining({ code: "topup_quote_already_consumed" }));

    const shortExpiry = Date.now() + 1_000;
    billing.createTopUpQuote({
      quoteId: "quote-expired",
      userId: "buyer-quote",
      amountMicros: 5_000_000,
      currency: "EUR",
      tokenAmount: 20_000,
      expiresAt: shortExpiry,
    });
    expect(() => billing.applyPaidTopUp({
      provider: "stripe",
      providerEventId: "evt_quote_late",
      paymentReference: "pi_quote_late",
      quoteId: "quote-expired",
      userId: "buyer-quote",
      amountMicros: 5_000_000,
      currency: "EUR",
      tokenAmount: 20_000,
      occurredAt: shortExpiry + 1,
      finalized: true,
    })).toThrowError(expect.objectContaining({ code: "topup_quote_expired" }));
  });
});
