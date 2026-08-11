import { generateKeyPairSync, sign } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiAccessManager } from "../src/coordinator/api-access.js";
import { BillingManager } from "../src/coordinator/billing.js";
import {
  StablecoinBillingInbound,
  applyBillingCommand,
  stablecoinWatcherSigningBytes,
} from "../src/coordinator/billing-inbound.js";
import {
  StablecoinCheckoutService,
  type StablecoinPaymentGateway,
} from "../src/coordinator/stablecoin-checkout.js";
import { MeshDatabase } from "../src/storage/database.js";
import { createCoordinator } from "../src/coordinator/server.js";
import type { SupabaseAuthService } from "../src/coordinator/supabase-auth.js";

describe("stablecoin buyer checkout", () => {
  let database: MeshDatabase;
  let billing: BillingManager;
  let gateway: FakeStablecoinGateway;
  let checkout: StablecoinCheckoutService;

  beforeEach(() => {
    database = new MeshDatabase(":memory:");
    billing = new BillingManager(database, new ApiAccessManager(database, {
      starterTokens: 0, requestsPerMinute: 30, maxConcurrent: 2, maxActiveKeys: 10,
    }));
    billing.registerPlan({
      planId: "mycellios-go", version: 1, priceCurrency: "EUR",
      priceMicros: 10_000_000, includedTokens: 50_000, status: "active",
    });
    gateway = new FakeStablecoinGateway();
    checkout = service(gateway);
  });

  afterEach(() => database.close());

  it("creates a server-bound subscription payment intent and replays it", async () => {
    const first = await checkout.createSubscription({
      userId: "coin-buyer", idempotencyKey: "coin-sub-1",
    });
    const replay = await checkout.createSubscription({
      userId: "coin-buyer", idempotencyKey: "coin-sub-1",
    });
    expect(first).toMatchObject({
      kind: "subscription", chainId: "base-mainnet", asset: "USDC",
      assetAtomicAmount: "10000000", recipient: "0xrecipient00000001", duplicate: false,
    });
    expect(replay).toEqual({ ...first, duplicate: true });
    expect(gateway.attempts).toHaveLength(1);
    expect(database.raw.prepare(
      `SELECT user_id, plan_id, plan_version, amount_micros, currency,
              asset_atomic_amount, status
       FROM billing_stablecoin_intents WHERE id = ?`,
    ).get(first.id)).toEqual({
      user_id: "coin-buyer", plan_id: "mycellios-go", plan_version: 1,
      amount_micros: 10_000_000, currency: "EUR",
      asset_atomic_amount: "10000000", status: "awaiting_payment",
    });

    const keys = generateKeyPairSync("ed25519");
    const inbound = new StablecoinBillingInbound({
      trustedWatcherKeys: new Map([[
        "watcher-subscription",
        keys.publicKey.export({ type: "spki", format: "pem" }).toString(),
      ]]),
      chains: new Map([["base-mainnet", { asset: "USDC", minimumConfirmations: 12 }]]),
    });
    const occurredAt = 1_800_000_001_000;
    const watcherPayload = {
      schema: "mycellios.stablecoin-payment.v1" as const,
      eventId: "base-mainnet:0xsubscription:0",
      kind: "subscription_paid" as const,
      chainId: "base-mainnet",
      asset: "USDC",
      txHash: "0xsubscription",
      logIndex: 0,
      blockNumber: 100,
      blockHash: "0xblocksubscription",
      confirmations: 12,
      finality: "finalized" as const,
      userId: "coin-buyer",
      amountMicros: 10_000_000,
      currency: "EUR",
      planId: "mycellios-go",
      planVersion: 1,
      subscriptionId: "stablecoin:coin-buyer",
      periodStart: occurredAt,
      periodEnd: occurredAt + 30 * 24 * 60 * 60_000,
      occurredAt,
      watcherKeyId: "watcher-subscription",
      paymentIntentId: first.id,
      assetAtomicAmount: first.assetAtomicAmount,
      recipient: first.recipient,
    };
    const command = inbound.verifyAndNormalize({
      ...watcherPayload,
      signature: sign(
        null,
        stablecoinWatcherSigningBytes(watcherPayload),
        keys.privateKey,
      ).toString("base64url"),
    });
    expect(applyBillingCommand(billing, command)).toMatchObject({ tokenBalance: 50_000 });
    expect(database.raw.prepare(
      "SELECT status, consumed_event_key FROM billing_stablecoin_intents WHERE id = ?",
    ).get(first.id)).toEqual({
      status: "consumed",
      consumed_event_key: "stablecoin:base-mainnet:0xsubscription:0",
    });

    const secondPayload = {
      ...watcherPayload,
      eventId: "base-mainnet:0xsubscriptionduplicate:1",
      txHash: "0xsubscriptionduplicate",
      logIndex: 1,
    };
    const secondCommand = inbound.verifyAndNormalize({
      ...secondPayload,
      signature: sign(
        null,
        stablecoinWatcherSigningBytes(secondPayload),
        keys.privateKey,
      ).toString("base64url"),
    });
    expect(() => applyBillingCommand(billing, secondCommand)).toThrowError(
      expect.objectContaining({ code: "stablecoin_intent_consumed" }),
    );
  });

  it("creates top-ups only from configured packs and binds an immutable quote", async () => {
    const result = await checkout.createTopUp({
      userId: "coin-topup", packId: "boost-5", idempotencyKey: "coin-topup-1",
    });
    expect(result).toMatchObject({ kind: "topup", assetAtomicAmount: "5000000" });
    expect(result.quoteId).toBeDefined();
    expect(database.raw.prepare(
      `SELECT amount_micros, currency, token_amount, status
       FROM billing_topup_quotes WHERE quote_id = ?`,
    ).get(result.quoteId!)).toEqual({
      amount_micros: 5_000_000, currency: "EUR", token_amount: 20_000, status: "open",
    });
    await expect(checkout.createTopUp({
      userId: "coin-topup", packId: "invented", idempotencyKey: "coin-topup-evil",
    })).rejects.toMatchObject({ code: "stablecoin_pack_not_found" });
  });

  it("retries allocation with one provider key and rejects changed economic intent", async () => {
    gateway.failNext = true;
    await expect(checkout.createTopUp({
      userId: "coin-retry", packId: "boost-5", idempotencyKey: "coin-retry-1",
    })).rejects.toThrow("temporary allocator failure");
    const completed = await checkout.createTopUp({
      userId: "coin-retry", packId: "boost-5", idempotencyKey: "coin-retry-1",
    });
    expect(completed.duplicate).toBe(false);
    expect(gateway.attempts[0]?.idempotencyKey).toBe(gateway.attempts[1]?.idempotencyKey);
    await expect(checkout.createSubscription({
      userId: "coin-retry", idempotencyKey: "coin-retry-1",
    })).rejects.toMatchObject({ code: "stablecoin_idempotency_conflict" });
  });

  it("exposes stablecoin intents only to account sessions with idempotency", async () => {
    const routeGateway = new FakeStablecoinGateway();
    const auth = {
      authenticate: vi.fn(async () => ({ id: "coin-route", email: null, role: null })),
    } as unknown as SupabaseAuthService;
    const runtime = await createCoordinator({
      host: "127.0.0.1", port: 0, databasePath: ":memory:", requestTimeoutMs: 1_000,
      apiAccessEnabled: true, apiStarterTokens: 0,
    }, {
      logger: false,
      supabaseAuthService: auth,
      stablecoinCheckout: { gateway: routeGateway, config: stablecoinConfig() },
    });
    try {
      runtime.billing.registerPlan({
        planId: "mycellios-go", version: 1, priceCurrency: "EUR",
        priceMicros: 10_000_000, includedTokens: 50_000, status: "active",
      });
      const accepted = await runtime.app.inject({
        method: "POST",
        url: "/v1/billing/crypto/topup",
        headers: {
          authorization: "Bearer account-session",
          "idempotency-key": "coin-route-topup",
        },
        payload: { packId: "boost-5" },
      });
      expect(accepted.statusCode).toBe(201);
      expect(accepted.json()).toMatchObject({
        object: "stablecoin_payment_intent",
        kind: "topup",
        asset: "USDC",
        duplicate: false,
      });
      const replay = await runtime.app.inject({
        method: "POST",
        url: "/v1/billing/crypto/topup",
        headers: {
          authorization: "Bearer account-session",
          "idempotency-key": "coin-route-topup",
        },
        payload: { packId: "boost-5" },
      });
      expect(replay.statusCode).toBe(200);
      expect(replay.json().duplicate).toBe(true);

      const apiKey = runtime.apiAccess.createKey("coin-route", "crypto-forbidden");
      const forbidden = await runtime.app.inject({
        method: "POST",
        url: "/v1/billing/crypto/subscription",
        headers: {
          authorization: `Bearer ${apiKey.secret}`,
          "idempotency-key": "coin-route-api-key",
        },
        payload: {},
      });
      expect(forbidden.statusCode).toBe(403);
      expect(forbidden.json().error.code).toBe("account_session_required");
    } finally {
      await runtime.close();
    }
  });

  function service(provider: StablecoinPaymentGateway) {
    return new StablecoinCheckoutService(database, billing, provider, stablecoinConfig());
  }

  function stablecoinConfig() {
    return {
      chainId: "base-mainnet",
      asset: "USDC",
      subscriptionAssetAtomicAmount: "10000000",
      subscriptionPeriodMs: 30 * 24 * 60 * 60_000,
      planId: "mycellios-go",
      planVersion: 1,
      topUpPacks: [{
        packId: "boost-5", amountMicros: 5_000_000, currency: "EUR",
        tokenAmount: 20_000, assetAtomicAmount: "5000000",
      }],
      now: () => 1_800_000_000_000,
    } as const;
  }
});

class FakeStablecoinGateway implements StablecoinPaymentGateway {
  attempts: Array<Parameters<StablecoinPaymentGateway["allocatePayment"]>[0]> = [];
  failNext = false;

  async allocatePayment(input: Parameters<StablecoinPaymentGateway["allocatePayment"]>[0]) {
    this.attempts.push(input);
    if (this.failNext) {
      this.failNext = false;
      throw new Error("temporary allocator failure");
    }
    return {
      providerReference: `deposit:${input.intentId}`,
      recipient: "0xrecipient00000001",
      checkoutUrl: `https://pay.example.com/${input.intentId}`,
    };
  }
}
