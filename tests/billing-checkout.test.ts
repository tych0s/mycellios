import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiAccessManager } from "../src/coordinator/api-access.js";
import {
  BillingCheckoutError,
  BillingCheckoutService,
  StripeRestGateway,
  type StripeCheckoutGateway,
} from "../src/coordinator/billing-checkout.js";
import { BillingManager } from "../src/coordinator/billing.js";
import { MeshDatabase } from "../src/storage/database.js";
import { createCoordinator } from "../src/coordinator/server.js";
import type { SupabaseAuthService } from "../src/coordinator/supabase-auth.js";
import { loadCoordinatorConfig } from "../src/core/config.js";

describe("buyer checkout service", () => {
  let database: MeshDatabase;
  let billing: BillingManager;
  let gateway: FakeStripeGateway;
  let checkout: BillingCheckoutService;

  beforeEach(() => {
    database = new MeshDatabase(":memory:");
    billing = new BillingManager(database, new ApiAccessManager(database, {
      starterTokens: 0, requestsPerMinute: 30, maxConcurrent: 2, maxActiveKeys: 10,
    }));
    billing.registerPlan({
      planId: "mycellios-go", version: 1, priceCurrency: "EUR",
      priceMicros: 10_000_000, includedTokens: 50_000, status: "active",
    });
    gateway = new FakeStripeGateway();
    checkout = service(gateway);
  });

  afterEach(() => database.close());

  it("creates one owned customer and one subscription Checkout across replays", async () => {
    const first = await checkout.createSubscription({
      userId: "buyer-one", email: "buyer@example.com", idempotencyKey: "subscribe-1",
    });
    const replay = await checkout.createSubscription({
      userId: "buyer-one", email: "changed@example.com", idempotencyKey: "subscribe-1",
    });
    expect(first).toMatchObject({ id: "cs_subscription_1", duplicate: false });
    expect(replay).toEqual({ ...first, duplicate: true });
    expect(gateway.customers).toHaveLength(1);
    expect(gateway.subscriptions).toHaveLength(1);
    expect(gateway.subscriptions[0]).toMatchObject({
      userId: "buyer-one", customerId: "cus_buyer_one", priceId: "price_go_1",
    });
    expect(database.raw.prepare(
      "SELECT user_id, provider_customer_id FROM billing_customers",
    ).all()).toEqual([{ user_id: "buyer-one", provider_customer_id: "cus_buyer_one" }]);
  });

  it("resumes one recent subscription Checkout across different browser attempts", async () => {
    const first = await checkout.createSubscription({
      userId: "buyer-resume", email: null, idempotencyKey: "subscribe-first",
    });
    const resumed = await checkout.createSubscription({
      userId: "buyer-resume", email: null, idempotencyKey: "subscribe-second",
    });
    expect(resumed).toEqual({ ...first, duplicate: true });
    expect(gateway.subscriptions).toHaveLength(1);
  });

  it("rejects another Checkout after verified activation", async () => {
    billing.applyPaidSubscription({
      provider: "stripe", providerEventId: "evt_already_active",
      providerSubscriptionId: "sub_already_active", userId: "buyer-active",
      planId: "mycellios-go", planVersion: 1, amountMicros: 10_000_000,
      currency: "EUR", periodStart: 1_786_000_000_000,
      periodEnd: 1_788_678_400_000, occurredAt: 1_786_000_000_000,
      finalized: true,
    });
    await expect(checkout.createSubscription({
      userId: "buyer-active", email: null, idempotencyKey: "subscribe-again",
    })).rejects.toMatchObject({ code: "subscription_already_exists", statusCode: 409 });
    expect(gateway.subscriptions).toHaveLength(0);
  });

  it("rejects reusing an operation key for a different purchase", async () => {
    await checkout.createTopUp({
      userId: "buyer-two", email: null, packId: "boost-5", idempotencyKey: "purchase-1",
    });
    await expect(checkout.createSubscription({
      userId: "buyer-two", email: null, idempotencyKey: "purchase-1",
    })).rejects.toMatchObject({ code: "checkout_idempotency_conflict", statusCode: 409 });
  });

  it("builds top-ups only from server packs and reuses the quote on a safe retry", async () => {
    gateway.failNextTopUp = true;
    await expect(checkout.createTopUp({
      userId: "buyer-three", email: null, packId: "boost-5", idempotencyKey: "topup-retry",
    })).rejects.toThrow("temporary Stripe failure");
    const pending = database.raw.prepare(
      "SELECT quote_id, status FROM billing_checkout_operations WHERE user_id = 'buyer-three'",
    ).get() as { quote_id: string; status: string };
    expect(pending.status).toBe("pending");

    const completed = await checkout.createTopUp({
      userId: "buyer-three", email: null, packId: "boost-5", idempotencyKey: "topup-retry",
    });
    expect(completed.quoteId).toBe(pending.quote_id);
    expect(gateway.topups[0]).toMatchObject({
      quoteId: pending.quote_id,
      tokenAmount: 20_000,
      priceId: "price_boost_5",
    });
    expect(gateway.topupAttempts[0]?.idempotencyKey).toBe(gateway.topupAttempts[1]?.idempotencyKey);
    expect(database.raw.prepare(
      "SELECT amount_micros, currency, token_amount, status FROM billing_topup_quotes WHERE quote_id = ?",
    ).get(pending.quote_id)).toEqual({
      amount_micros: 5_000_000, currency: "EUR", token_amount: 20_000, status: "open",
    });
    await expect(checkout.createTopUp({
      userId: "buyer-three", email: null, packId: "client-invented", idempotencyKey: "topup-evil",
    })).rejects.toMatchObject({ code: "topup_pack_not_found" });
  });

  it("reuses the owned customer for the portal and rejects cross-account customer collisions", async () => {
    await checkout.createSubscription({
      userId: "buyer-four", email: null, idempotencyKey: "subscribe-four",
    });
    const portal = await checkout.createPortal({
      userId: "buyer-four", email: null, idempotencyKey: "portal-four",
    });
    expect(portal.id).toBe("bps_1");
    expect(gateway.customers).toHaveLength(1);
    expect(gateway.portals[0]?.customerId).toBe("cus_buyer_four");

    gateway.forcedCustomerId = "cus_buyer_four";
    await expect(checkout.createSubscription({
      userId: "buyer-five", email: null, idempotencyKey: "subscribe-five",
    })).rejects.toMatchObject({ code: "stripe_customer_conflict" });
  });

  it("sends form-encoded Stripe requests with provider idempotency and metadata", async () => {
    const fetchMock = vi.fn<typeof fetch>(async (_url, init) => Response.json({
      id: "cs_test_123",
      url: "https://checkout.stripe.com/c/pay/cs_test_123",
    }, { status: 200 }));
    const stripe = new StripeRestGateway({
      secretKey: "sk_test_1234567890abcdef",
      fetch: fetchMock,
    });
    await stripe.createTopUpCheckout({
      userId: "buyer-six", customerId: "cus_buyer_six", quoteId: "quote_six",
      tokenAmount: 20_000, priceId: "price_boost_5",
      successUrl: "https://mycellios.com/account?paid=1",
      cancelUrl: "https://mycellios.com/account?cancelled=1",
      idempotencyKey: "provider-op-six",
    });
    const [, init] = fetchMock.mock.calls[0]!;
    expect(init?.headers).toMatchObject({
      authorization: "Bearer sk_test_1234567890abcdef",
      "idempotency-key": "provider-op-six",
    });
    const form = new URLSearchParams(String(init?.body));
    expect(form.get("mode")).toBe("payment");
    expect(form.get("metadata[mycellios_quote_id]")).toBe("quote_six");
    expect(form.get("payment_intent_data[metadata][mycellios_user_id]")).toBe("buyer-six");
    expect(form.get("line_items[0][price]")).toBe("price_boost_5");
  });

  it("exposes Checkout only to account sessions with mandatory idempotency", async () => {
    const routeGateway = new FakeStripeGateway();
    const auth = {
      authenticate: vi.fn(async () => ({
        id: "buyer-route", email: "route@example.com", role: null,
      })),
    } as unknown as SupabaseAuthService;
    const runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 0,
      databasePath: ":memory:",
      requestTimeoutMs: 1_000,
      apiAccessEnabled: true,
      apiStarterTokens: 0,
    }, {
      logger: false,
      supabaseAuthService: auth,
      billingCheckout: { gateway: routeGateway, config: checkoutConfig() },
    });
    try {
      runtime.billing.registerPlan({
        planId: "mycellios-go", version: 1, priceCurrency: "EUR",
        priceMicros: 10_000_000, includedTokens: 50_000, status: "active",
      });
      const missingKey = await runtime.app.inject({
        method: "POST", url: "/v1/billing/checkout/subscription",
        headers: { authorization: "Bearer account-session" },
        payload: {},
      });
      expect(missingKey.statusCode).toBe(400);
      expect(missingKey.json()).toEqual({
        error: {
          code: "idempotency_key_required",
          message: "Billing mutations require an Idempotency-Key header.",
        },
      });

      const accepted = await runtime.app.inject({
        method: "POST", url: "/v1/billing/checkout/subscription",
        headers: { authorization: "Bearer account-session", "idempotency-key": "route-subscribe" },
        payload: {},
      });
      expect(accepted.statusCode).toBe(201);
      expect(accepted.json()).toMatchObject({
        object: "billing_checkout_session", duplicate: false,
      });
      const replay = await runtime.app.inject({
        method: "POST", url: "/v1/billing/checkout/subscription",
        headers: { authorization: "Bearer account-session", "idempotency-key": "route-subscribe" },
        payload: {},
      });
      expect(replay.statusCode).toBe(200);
      expect(replay.json()).toMatchObject({ duplicate: true });

      const overview = await runtime.app.inject({
        method: "GET", url: "/v1/billing/account",
        headers: { authorization: "Bearer account-session" },
      });
      expect(overview.statusCode).toBe(200);
      expect(overview.json()).toEqual({
        object: "billing_account",
        checkoutAvailable: true,
        portalAvailable: true,
        topUpsAvailable: true,
        tokenDebt: 0,
        plan: {
          id: "mycellios-go", version: 1, currency: "EUR",
          amountMicros: 10_000_000, includedTokens: 50_000, status: "active",
        },
        subscription: null,
      });

      const apiKey = runtime.apiAccess.createKey("buyer-route", "billing-forbidden");
      const forbidden = await runtime.app.inject({
        method: "POST", url: "/v1/billing/portal",
        headers: { authorization: `Bearer ${apiKey.secret}`, "idempotency-key": "portal-api-key" },
        payload: {},
      });
      expect(forbidden.statusCode).toBe(403);
      expect(forbidden.json().error.code).toBe("account_session_required");
      const forbiddenOverview = await runtime.app.inject({
        method: "GET", url: "/v1/billing/account",
        headers: { authorization: `Bearer ${apiKey.secret}` },
      });
      expect(forbiddenOverview.statusCode).toBe(403);
      expect(forbiddenOverview.json().error.code).toBe("account_session_required");
    } finally {
      await runtime.close();
    }
  });

  it("activates the immutable Go plan when Checkout is configured", async () => {
    const runtime = await createCoordinator(loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_STRIPE_SECRET_KEY: "sk_test_1234567890abcdef",
      MYCELLIOS_STRIPE_GO_PRICE_ID: "price_go_1",
      MYCELLIOS_GO_INCLUDED_TOKENS: "50000",
      MYCELLIOS_BILLING_SUCCESS_URL: "https://mycellios.com/account?checkout=success",
      MYCELLIOS_BILLING_CANCEL_URL: "https://mycellios.com/account?checkout=cancelled",
      MYCELLIOS_BILLING_PORTAL_RETURN_URL: "https://mycellios.com/account",
    }), { logger: false });
    try {
      expect(runtime.billingCheckout).not.toBeNull();
      expect(runtime.billing.getPlan("mycellios-go", 1)).toMatchObject({
        priceCurrency: "EUR",
        priceMicros: 10_000_000,
        includedTokens: 50_000,
        status: "active",
      });
    } finally {
      await runtime.close();
    }
  });

  function service(stripe: StripeCheckoutGateway) {
    return new BillingCheckoutService(database, billing, stripe, checkoutConfig());
  }

  function checkoutConfig() {
    return {
      planId: "mycellios-go",
      planVersion: 1,
      subscriptionPriceId: "price_go_1",
      successUrl: "https://mycellios.com/account?checkout=success",
      cancelUrl: "https://mycellios.com/account?checkout=cancelled",
      portalReturnUrl: "https://mycellios.com/account",
      topUpPacks: [{
        packId: "boost-5", amountMicros: 5_000_000, currency: "EUR",
        tokenAmount: 20_000, stripePriceId: "price_boost_5",
      }],
      now: () => 1_800_000_000_000,
    } as const;
  }
});

class FakeStripeGateway implements StripeCheckoutGateway {
  customers: Array<Parameters<StripeCheckoutGateway["createCustomer"]>[0]> = [];
  subscriptions: Array<Parameters<StripeCheckoutGateway["createSubscriptionCheckout"]>[0]> = [];
  topupAttempts: Array<Parameters<StripeCheckoutGateway["createTopUpCheckout"]>[0]> = [];
  topups: Array<Parameters<StripeCheckoutGateway["createTopUpCheckout"]>[0]> = [];
  portals: Array<Parameters<StripeCheckoutGateway["createPortalSession"]>[0]> = [];
  failNextTopUp = false;
  forcedCustomerId?: string;

  async createCustomer(input: Parameters<StripeCheckoutGateway["createCustomer"]>[0]) {
    this.customers.push(input);
    return { id: this.forcedCustomerId ?? `cus_${input.userId.replaceAll("-", "_")}` };
  }

  async createSubscriptionCheckout(input: Parameters<StripeCheckoutGateway["createSubscriptionCheckout"]>[0]) {
    this.subscriptions.push(input);
    return { id: `cs_subscription_${this.subscriptions.length}`, url: "https://checkout.stripe.com/subscription" };
  }

  async createTopUpCheckout(input: Parameters<StripeCheckoutGateway["createTopUpCheckout"]>[0]) {
    this.topupAttempts.push(input);
    if (this.failNextTopUp) {
      this.failNextTopUp = false;
      throw new Error("temporary Stripe failure");
    }
    this.topups.push(input);
    return { id: `cs_topup_${this.topups.length}`, url: "https://checkout.stripe.com/topup" };
  }

  async createPortalSession(input: Parameters<StripeCheckoutGateway["createPortalSession"]>[0]) {
    this.portals.push(input);
    return { id: `bps_${this.portals.length}`, url: "https://billing.stripe.com/portal" };
  }
}
