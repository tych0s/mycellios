import { createHmac, generateKeyPairSync, sign } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ApiAccessManager } from "../src/coordinator/api-access.js";
import {
  BillingInboundError,
  StablecoinBillingInbound,
  StripeBillingInbound,
  applyBillingCommand,
  type SignedStablecoinWatcherEvent,
  stablecoinWatcherSigningBytes,
} from "../src/coordinator/billing-inbound.js";
import { BillingManager } from "../src/coordinator/billing.js";
import { MeshDatabase } from "../src/storage/database.js";

const stripeSecret = "whsec_test_secret_at_least_32_chars";
const nowSeconds = 1_786_000_100;

function stripeDelivery(event: object, timestamp = nowSeconds, secret = stripeSecret) {
  const rawBody = Buffer.from(JSON.stringify(event), "utf8");
  const signature = createHmac("sha256", secret)
    .update(Buffer.concat([Buffer.from(`${timestamp}.`), rawBody]))
    .digest("hex");
  return { rawBody, header: `t=${timestamp},v1=${signature}` };
}

describe("signed billing inbound adapters", () => {
  let database: MeshDatabase;
  let billing: BillingManager;

  beforeEach(() => {
    database = new MeshDatabase(":memory:");
    billing = new BillingManager(database, new ApiAccessManager(database, {
      starterTokens: 0,
      requestsPerMinute: 10,
      maxConcurrent: 2,
      maxActiveKeys: 2,
    }));
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

  it("verifies raw Stripe bytes and applies a subscription invoice once", () => {
    const stripe = new StripeBillingInbound({
      endpointSecrets: ["old_rotating_secret_at_least_32_chars", stripeSecret],
      expectedLivemode: false,
      nowSeconds: () => nowSeconds,
    });
    const delivery = stripeDelivery({
      id: "evt_invoice_paid",
      type: "invoice.paid",
      created: nowSeconds,
      livemode: false,
      data: { object: {
        id: "in_1",
        subscription: "sub_1",
        amount_paid: 1_000,
        currency: "eur",
        metadata: {
          mycellios_user_id: "buyer-stripe",
          mycellios_plan_id: "mycellios-go",
          mycellios_plan_version: "1",
          mycellios_period_start_ms: "1786000000000",
          mycellios_period_end_ms: "1788678400000",
        },
      } },
    });
    const command = stripe.verifyAndNormalize(delivery.rawBody, delivery.header);
    expect(command?.kind).toBe("paid_subscription");
    const first = applyBillingCommand(billing, command!);
    const replay = applyBillingCommand(billing, command!);
    expect(first).toMatchObject({ tokenBalance: 50_000, duplicate: false });
    expect(replay).toMatchObject({ tokenBalance: 50_000, duplicate: true });
  });

  it("reads current Stripe invoice subscription metadata and period fields", () => {
    const stripe = new StripeBillingInbound({
      endpointSecrets: [stripeSecret], expectedLivemode: false, nowSeconds: () => nowSeconds,
    });
    const delivery = stripeDelivery({
      id: "evt_invoice_parent_metadata",
      type: "invoice.paid",
      created: nowSeconds,
      livemode: false,
      data: { object: {
        id: "in_parent",
        amount_paid: 1_000,
        currency: "eur",
        period_start: 1_786_000_000,
        period_end: 1_788_678_400,
        metadata: {},
        parent: { subscription_details: {
          subscription: "sub_parent",
          metadata: {
            mycellios_user_id: "buyer-parent",
            mycellios_plan_id: "mycellios-go",
            mycellios_plan_version: "1",
          },
        } },
      } },
    });
    const command = stripe.verifyAndNormalize(delivery.rawBody, delivery.header);
    expect(command).toMatchObject({
      kind: "paid_subscription",
      event: {
        userId: "buyer-parent",
        periodStart: 1_786_000_000_000,
        periodEnd: 1_788_678_400_000,
      },
    });
    expect(applyBillingCommand(billing, command!)).toMatchObject({ tokenBalance: 50_000 });
  });

  it("maps actionable subscription updates and ignores non-terminal noise", () => {
    const stripe = new StripeBillingInbound({
      endpointSecrets: [stripeSecret], expectedLivemode: false, nowSeconds: () => nowSeconds,
    });
    billing.applyPaidSubscription({
      provider: "stripe", providerEventId: "evt_update_seed",
      providerSubscriptionId: "sub_update", userId: "buyer-update",
      planId: "mycellios-go", planVersion: 1, amountMicros: 10_000_000,
      currency: "EUR", periodStart: 1_786_000_000_000,
      periodEnd: 1_788_678_400_000, occurredAt: 1_786_000_000_000,
      finalized: true,
    });
    const base = {
      type: "customer.subscription.updated", created: nowSeconds,
      livemode: false, data: { object: {
        id: "sub_update", metadata: { mycellios_user_id: "buyer-update" },
      } },
    };
    const active = stripeDelivery({ ...base, id: "evt_update_active", data: { object: { ...base.data.object, status: "active" } } });
    expect(stripe.verifyAndNormalize(active.rawBody, active.header)).toBeNull();

    const pastDue = stripeDelivery({ ...base, id: "evt_update_past_due", data: { object: { ...base.data.object, status: "past_due" } } });
    const command = stripe.verifyAndNormalize(pastDue.rawBody, pastDue.header);
    expect(command).toMatchObject({ kind: "subscription_status", event: { status: "past_due" } });
    applyBillingCommand(billing, command!);
    expect(billing.getSubscription("buyer-update")?.status).toBe("past_due");
  });

  it("maps paid Checkout top-ups and rejects unpaid sessions", () => {
    const stripe = new StripeBillingInbound({
      endpointSecrets: [stripeSecret], expectedLivemode: false, nowSeconds: () => nowSeconds,
    });
    billing.createTopUpQuote({
      quoteId: "quote-stripe-topup",
      userId: "buyer-topup",
      amountMicros: 20_000_000,
      currency: "EUR",
      tokenAmount: 80_000,
      expiresAt: Date.now() + 60_000,
    });
    const base = {
      id: "evt_checkout_paid",
      type: "checkout.session.completed",
      created: nowSeconds,
      livemode: false,
      data: { object: {
        id: "cs_1", mode: "payment", payment_status: "paid",
        payment_intent: "pi_1", amount_total: 2_000, currency: "eur",
        metadata: {
          mycellios_user_id: "buyer-topup",
          mycellios_quote_id: "quote-stripe-topup",
          mycellios_token_amount: "80000",
        },
      } },
    };
    const paid = stripeDelivery(base);
    const result = applyBillingCommand(billing, stripe.verifyAndNormalize(paid.rawBody, paid.header)!);
    expect(result).toMatchObject({ tokenBalance: 80_000 });

    const unpaid = stripeDelivery({
      ...base,
      id: "evt_checkout_unpaid",
      data: { object: { ...base.data.object, payment_status: "unpaid" } },
    });
    expect(() => stripe.verifyAndNormalize(unpaid.rawBody, unpaid.header)).toThrowError(
      expect.objectContaining({ code: "stripe_topup_not_paid" }),
    );
  });

  it("rejects Stripe body tampering, stale signatures and mode confusion", () => {
    const stripe = new StripeBillingInbound({
      endpointSecrets: [stripeSecret], expectedLivemode: false, nowSeconds: () => nowSeconds,
    });
    const event = {
      id: "evt_security", type: "unknown.event", created: nowSeconds, livemode: false,
      data: { object: {} },
    };
    const delivery = stripeDelivery(event);
    expect(() => stripe.verifyAndNormalize(
      Buffer.from(delivery.rawBody.toString("utf8").replace("evt_security", "evt_tampered")),
      delivery.header,
    )).toThrowError(expect.objectContaining({ code: "invalid_stripe_signature" }));

    const stale = stripeDelivery(event, nowSeconds - 301);
    expect(() => stripe.verifyAndNormalize(stale.rawBody, stale.header)).toThrowError(
      expect.objectContaining({ code: "stripe_signature_expired" }),
    );

    const live = stripeDelivery({ ...event, livemode: true });
    expect(() => stripe.verifyAndNormalize(live.rawBody, live.header)).toThrowError(
      expect.objectContaining({ code: "stripe_mode_mismatch" }),
    );
  });

  it("verifies a finalized stablecoin watcher event and applies its top-up once", () => {
    const pair = generateKeyPairSync("ed25519");
    const adapter = new StablecoinBillingInbound({
      trustedWatcherKeys: new Map([
        ["watcher-1", pair.publicKey.export({ type: "spki", format: "pem" }).toString()],
      ]),
      chains: new Map([["base-mainnet", { asset: "USDC", minimumConfirmations: 20 }]]),
    });
    billing.createTopUpQuote({
      quoteId: "quote-crypto-topup",
      userId: "buyer-crypto",
      amountMicros: 20_000_000,
      currency: "USDC",
      tokenAmount: 80_000,
      expiresAt: Date.now() + 60_000,
    });
    createStablecoinTopUpIntent({
      id: "coinpay-crypto-topup",
      userId: "buyer-crypto",
      quoteId: "quote-crypto-topup",
      amountMicros: 20_000_000,
      currency: "USDC",
      atomicAmount: "20000000",
      recipient: "0xrecipientcrypto001",
    });
    const payload = {
      schema: "mycellios.stablecoin-payment.v1" as const,
      eventId: "base-mainnet:0xabc123:7",
      kind: "topup_paid" as const,
      chainId: "base-mainnet",
      asset: "USDC",
      txHash: "0xabc123",
      logIndex: 7,
      blockNumber: 25_000_000,
      blockHash: "0xblock123",
      confirmations: 20,
      finality: "finalized" as const,
      userId: "buyer-crypto",
      amountMicros: 20_000_000,
      currency: "USDC",
      quoteId: "quote-crypto-topup",
      tokenAmount: 80_000,
      occurredAt: 1_786_000_000_000,
      watcherKeyId: "watcher-1",
      paymentIntentId: "coinpay-crypto-topup",
      assetAtomicAmount: "20000000",
      recipient: "0xrecipientcrypto001",
    };
    const signed: SignedStablecoinWatcherEvent = {
      ...payload,
      signature: sign(null, stablecoinWatcherSigningBytes(payload), pair.privateKey).toString("base64url"),
    };
    const command = adapter.verifyAndNormalize(signed);
    const first = applyBillingCommand(billing, command);
    const replay = applyBillingCommand(billing, command);
    expect(first).toMatchObject({ tokenBalance: 80_000, duplicate: false });
    expect(replay).toMatchObject({ tokenBalance: 80_000, duplicate: true });
  });

  it("fails closed for stablecoin tampering, insufficient finality and unknown chains", () => {
    const pair = generateKeyPairSync("ed25519");
    const adapter = new StablecoinBillingInbound({
      trustedWatcherKeys: new Map([
        ["watcher-1", pair.publicKey.export({ type: "spki", format: "pem" }).toString()],
      ]),
      chains: new Map([["base-mainnet", { asset: "USDC", minimumConfirmations: 20 }]]),
    });
    const payload = {
      schema: "mycellios.stablecoin-payment.v1" as const,
      eventId: "base-mainnet:0xdef456:2",
      kind: "topup_paid" as const,
      chainId: "base-mainnet",
      asset: "USDC",
      txHash: "0xdef456",
      logIndex: 2,
      blockNumber: 25_000_001,
      blockHash: "0xblock456",
      confirmations: 20,
      finality: "finalized" as const,
      userId: "buyer-security",
      amountMicros: 20_000_000,
      currency: "USDC",
      quoteId: "quote-security",
      tokenAmount: 80_000,
      occurredAt: 1_786_000_000_000,
      watcherKeyId: "watcher-1",
      paymentIntentId: "coinpay-security",
      assetAtomicAmount: "20000000",
      recipient: "0xrecipientsecurity1",
    };
    const signed = {
      ...payload,
      signature: sign(null, stablecoinWatcherSigningBytes(payload), pair.privateKey).toString("base64url"),
    };
    expect(() => adapter.verifyAndNormalize({ ...signed, tokenAmount: 800_000 })).toThrowError(
      expect.objectContaining({ code: "invalid_stablecoin_signature" }),
    );
    const lowConfirmations = { ...payload, confirmations: 5 };
    expect(() => adapter.verifyAndNormalize({
      ...lowConfirmations,
      signature: sign(null, stablecoinWatcherSigningBytes(lowConfirmations), pair.privateKey).toString("base64url"),
    })).toThrowError(expect.objectContaining({ code: "stablecoin_not_final" }));
    const unknownChain = { ...payload, chainId: "unknown", eventId: "unknown:0xdef456:2" };
    expect(() => adapter.verifyAndNormalize({
      ...unknownChain,
      signature: sign(null, stablecoinWatcherSigningBytes(unknownChain), pair.privateKey).toString("base64url"),
    })).toThrowError(expect.objectContaining({ code: "stablecoin_chain_not_allowed" }));
  });

  function createStablecoinTopUpIntent(input: {
    id: string; userId: string; quoteId: string; amountMicros: number;
    currency: string; atomicAmount: string; recipient: string;
  }) {
    const now = Date.now();
    database.raw.prepare(
      `INSERT INTO billing_stablecoin_intents(
         id, user_id, idempotency_key, request_digest, kind, quote_id,
         amount_micros, currency, chain_id, asset, asset_atomic_amount,
         recipient, provider_reference, status, expires_at, created_at, updated_at
       ) VALUES (?, ?, ?, 'fixture', 'topup', ?, ?, ?, 'base-mainnet', 'USDC', ?, ?, ?,
                 'awaiting_payment', ?, ?, ?)`,
    ).run(
      input.id, input.userId, `fixture:${input.id}`, input.quoteId,
      input.amountMicros, input.currency, input.atomicAmount, input.recipient,
      `deposit:${input.id}`, now + 60_000, now, now,
    );
  }
});
