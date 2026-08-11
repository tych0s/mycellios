import Fastify from "fastify";
import { createHmac, generateKeyPairSync, sign } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ApiAccessManager } from "../src/coordinator/api-access.js";
import {
  StablecoinBillingInbound,
  StripeBillingInbound,
  stablecoinWatcherSigningBytes,
  type StablecoinWatcherPayload,
} from "../src/coordinator/billing-inbound.js";
import { registerBillingInboundRoutes } from "../src/coordinator/billing-http.js";
import { BillingManager } from "../src/coordinator/billing.js";
import { MeshDatabase } from "../src/storage/database.js";
import { loadCoordinatorConfig } from "../src/core/config.js";
import { createCoordinator } from "../src/coordinator/server.js";

describe("billing HTTP ingress", () => {
  const secret = "whsec_http_test_secret_long_enough";
  const timestamp = 1_800_000_000;
  let app: ReturnType<typeof Fastify>;
  let database: MeshDatabase;
  let billing: BillingManager;

  beforeEach(() => {
    app = Fastify();
    database = new MeshDatabase(":memory:");
    billing = new BillingManager(database, new ApiAccessManager(database, {
      starterTokens: 0,
      requestsPerMinute: 30,
      maxConcurrent: 2,
      maxActiveKeys: 10,
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

  afterEach(async () => {
    await app.close();
    database.close();
  });

  it("preserves Stripe raw bytes, applies once and ignores unsupported signed events", async () => {
    await registerBillingInboundRoutes(app, {
      manager: billing,
      stripe: new StripeBillingInbound({
        endpointSecrets: [secret],
        expectedLivemode: false,
        nowSeconds: () => timestamp,
      }),
    });
    const event = stripeInvoiceEvent();
    const raw = Buffer.from(JSON.stringify(event));
    const headers = {
      "content-type": "application/json",
      "stripe-signature": stripeSignature(raw),
    };
    const accepted = await app.inject({ method: "POST", url: "/webhooks/v1/stripe", headers, payload: raw });
    expect(accepted.statusCode).toBe(202);
    expect(accepted.json()).toEqual({ accepted: true, duplicate: false });
    const replay = await app.inject({ method: "POST", url: "/webhooks/v1/stripe", headers, payload: raw });
    expect(replay.statusCode).toBe(200);
    expect(replay.json()).toEqual({ accepted: true, duplicate: true });
    expect(database.raw.prepare("SELECT token_balance FROM api_accounts WHERE user_id = 'buyer-http'").get())
      .toEqual({ token_balance: 50_000 });

    const ignoredRaw = Buffer.from(JSON.stringify({ ...event, id: "evt_ignored", type: "customer.created" }));
    const ignored = await app.inject({
      method: "POST",
      url: "/webhooks/v1/stripe",
      headers: { ...headers, "stripe-signature": stripeSignature(ignoredRaw) },
      payload: ignoredRaw,
    });
    expect(ignored.statusCode).toBe(200);
    expect(ignored.json()).toEqual({ accepted: true, ignored: true });
  });

  it("rejects a changed Stripe body and fails closed when the adapter is absent", async () => {
    await registerBillingInboundRoutes(app, {
      manager: billing,
      stripe: new StripeBillingInbound({
        endpointSecrets: [secret], expectedLivemode: false, nowSeconds: () => timestamp,
      }),
    });
    const raw = Buffer.from(JSON.stringify(stripeInvoiceEvent()));
    const changed = Buffer.from(raw.toString("utf8").replace("buyer-http", "buyer-evil"));
    const response = await app.inject({
      method: "POST", url: "/webhooks/v1/stripe",
      headers: { "content-type": "application/json", "stripe-signature": stripeSignature(raw) },
      payload: changed,
    });
    expect(response.statusCode).toBe(401);
    expect(response.json().error.code).toBe("invalid_stripe_signature");

    const closed = Fastify();
    await registerBillingInboundRoutes(closed, { manager: billing });
    const unavailable = await closed.inject({
      method: "POST", url: "/webhooks/v1/stripe",
      headers: { "content-type": "application/json", "stripe-signature": "t=1,v1=bad" },
      payload: "{}",
    });
    expect(unavailable.statusCode).toBe(503);
    await closed.close();
  });

  it("accepts only signed finalized stablecoin watcher events", async () => {
    const keys = generateKeyPairSync("ed25519");
    const inbound = new StablecoinBillingInbound({
      trustedWatcherKeys: new Map([["watcher-http", keys.publicKey.export({ type: "spki", format: "pem" }).toString()]]),
      chains: new Map([["base-mainnet", { asset: "USDC", minimumConfirmations: 12 }]]),
    });
    await registerBillingInboundRoutes(app, { manager: billing, stablecoin: inbound });
    billing.createTopUpQuote({
      quoteId: "quote-http", userId: "buyer-http-coin", amountMicros: 5_000_000,
      currency: "EUR", tokenAmount: 20_000, expiresAt: timestamp * 1_000 + 60_000,
    });
    database.raw.prepare(
      `INSERT INTO billing_stablecoin_intents(
         id, user_id, idempotency_key, request_digest, kind, quote_id,
         amount_micros, currency, chain_id, asset, asset_atomic_amount,
         recipient, provider_reference, status, expires_at, created_at, updated_at
       ) VALUES (
         'coinpay-http', 'buyer-http-coin', 'fixture-http', 'fixture', 'topup', 'quote-http',
         5000000, 'EUR', 'base-mainnet', 'USDC', '5000000',
         '0xrecipienthttp001', 'deposit:coinpay-http', 'awaiting_payment', ?, ?, ?
       )`,
    ).run(timestamp * 1_000 + 60_000, timestamp * 1_000, timestamp * 1_000);
    const payload: StablecoinWatcherPayload = {
      schema: "mycellios.stablecoin-payment.v1",
      eventId: "base-mainnet:0xabc123:0",
      kind: "topup_paid",
      chainId: "base-mainnet",
      asset: "USDC",
      txHash: "0xabc123",
      logIndex: 0,
      blockNumber: 100,
      blockHash: "0xblock123",
      confirmations: 12,
      finality: "finalized",
      userId: "buyer-http-coin",
      amountMicros: 5_000_000,
      currency: "EUR",
      quoteId: "quote-http",
      tokenAmount: 20_000,
      occurredAt: timestamp * 1_000,
      watcherKeyId: "watcher-http",
      paymentIntentId: "coinpay-http",
      assetAtomicAmount: "5000000",
      recipient: "0xrecipienthttp001",
    };
    const signature = sign(null, stablecoinWatcherSigningBytes(payload), keys.privateKey).toString("base64url");
    const accepted = await app.inject({
      method: "POST", url: "/webhooks/v1/stablecoin", payload: { ...payload, signature },
    });
    expect(accepted.statusCode).toBe(202);
    const malformed = await app.inject({
      method: "POST", url: "/webhooks/v1/stablecoin",
      headers: { "content-type": "application/json" },
      payload: JSON.stringify("not-an-event"),
    });
    expect(malformed.statusCode).toBe(400);
    expect(malformed.json().error.code).toBe("invalid_stablecoin_payload");
    const tampered = await app.inject({
      method: "POST", url: "/webhooks/v1/stablecoin",
      payload: { ...payload, eventId: "base-mainnet:0xabc124:0", txHash: "0xabc124", signature },
    });
    expect(tampered.statusCode).toBe(401);
    expect(tampered.json().error.code).toBe("invalid_stablecoin_signature");
  });

  it("loads current and previous Stripe secrets only with an explicit endpoint mode", () => {
    expect(() => loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_STRIPE_WEBHOOK_SECRET: "whsec_current_long_enough",
    })).toThrow("MYCELLIOS_STRIPE_WEBHOOK_LIVEMODE");

    const config = loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_STRIPE_WEBHOOK_SECRET: "whsec_current_long_enough",
      MYCELLIOS_STRIPE_PREVIOUS_WEBHOOK_SECRET: "whsec_previous_long_enough",
      MYCELLIOS_STRIPE_WEBHOOK_LIVEMODE: "false",
    });
    expect(config.stripeWebhookSecrets).toEqual([
      "whsec_current_long_enough",
      "whsec_previous_long_enough",
    ]);
    expect(config.stripeWebhookLivemode).toBe(false);
  });

  it("wires configured Stripe verification into the real coordinator", async () => {
    const realTimestamp = Math.floor(Date.now() / 1_000);
    const config = loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_API_STARTER_TOKENS: "0",
      MYCELLIOS_STRIPE_WEBHOOK_SECRET: secret,
      MYCELLIOS_STRIPE_WEBHOOK_LIVEMODE: "false",
    });
    const runtime = await createCoordinator(config, { logger: false });
    runtime.billing.registerPlan({
      planId: "mycellios-go",
      version: 1,
      priceCurrency: "EUR",
      priceMicros: 10_000_000,
      includedTokens: 50_000,
      status: "active",
    });
    const event = stripeInvoiceEvent(realTimestamp);
    const raw = Buffer.from(JSON.stringify(event));
    const accepted = await runtime.app.inject({
      method: "POST",
      url: "/webhooks/v1/stripe",
      headers: {
        "content-type": "application/json",
        "stripe-signature": stripeSignature(raw, realTimestamp),
      },
      payload: raw,
    });
    expect(accepted.statusCode).toBe(202);
    expect(runtime.apiAccess.getOrCreateAccount("buyer-http").tokenBalance).toBe(50_000);
    await runtime.close();
  });

  function stripeSignature(raw: Buffer, signedAt = timestamp): string {
    const digest = createHmac("sha256", secret).update(`${signedAt}.`).update(raw).digest("hex");
    return `t=${signedAt},v1=${digest}`;
  }

  function stripeInvoiceEvent(created = timestamp) {
    return {
      id: "evt_http_invoice",
      type: "invoice.paid",
      created,
      livemode: false,
      data: {
        object: {
          subscription: "sub_http",
          amount_paid: 1_000,
          currency: "eur",
          metadata: {
            mycellios_user_id: "buyer-http",
            mycellios_plan_id: "mycellios-go",
            mycellios_plan_version: 1,
            mycellios_period_start_ms: created * 1_000,
            mycellios_period_end_ms: created * 1_000 + 30 * 24 * 60 * 60_000,
          },
        },
      },
    };
  }
});
