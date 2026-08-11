#!/usr/bin/env node

import { readFileSync } from "node:fs";

const REQUIRED_EVENTS = new Set([
  "checkout.session.completed",
  "invoice.paid",
  "invoice.payment_failed",
  "customer.subscription.updated",
  "customer.subscription.deleted",
  "charge.refunded",
]);

function secretFromEnvironment() {
  const direct = process.env.MYCELLIOS_STRIPE_SECRET_KEY?.trim();
  const file = process.env.MYCELLIOS_STRIPE_SECRET_KEY_FILE?.trim();
  if (direct && file) {
    throw new Error("Configure the Stripe test key directly or by file, never both.");
  }
  const value = direct ?? (file ? readFileSync(file, "utf8").trim() : "");
  if (!value) {
    throw new Error("Missing MYCELLIOS_STRIPE_SECRET_KEY(_FILE).");
  }
  if (!/^(sk|rk)_test_/.test(value)) {
    throw new Error("Sandbox verification refuses non-test Stripe credentials.");
  }
  return value;
}

async function stripeGet(path, secretKey) {
  const response = await fetch(`https://api.stripe.com${path}`, {
    headers: { Authorization: `Bearer ${secretKey}` },
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(`Stripe ${path} failed (${response.status}): ${body?.error?.code ?? "unknown_error"}`);
  }
  return body;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function main() {
  const secretKey = secretFromEnvironment();
  const priceId = process.env.MYCELLIOS_STRIPE_GO_PRICE_ID?.trim();
  const webhookUrl = process.env.MYCELLIOS_STRIPE_SANDBOX_WEBHOOK_URL?.trim();
  assert(priceId?.startsWith("price_"), "Missing or invalid MYCELLIOS_STRIPE_GO_PRICE_ID.");
  assert(webhookUrl, "Missing MYCELLIOS_STRIPE_SANDBOX_WEBHOOK_URL.");

  const expectedWebhookUrl = new URL(webhookUrl).toString();
  const [account, price, endpointPage] = await Promise.all([
    stripeGet("/v1/account", secretKey),
    stripeGet(`/v1/prices/${encodeURIComponent(priceId)}?expand[]=product`, secretKey),
    stripeGet("/v1/webhook_endpoints?limit=100", secretKey),
  ]);

  assert(price.livemode === false, "Configured subscription Price is not in test mode.");
  assert(price.active === true, "Configured subscription Price is inactive.");
  assert(price.type === "recurring", "Configured subscription Price is not recurring.");
  assert(price.recurring?.interval === "month", "Configured subscription Price is not monthly.");
  assert(price.currency === "eur", "Configured subscription Price is not denominated in EUR.");
  assert(price.unit_amount === 1000, "Configured subscription Price is not EUR 10.00.");
  assert(price.product?.active === true, "Configured subscription Product is inactive.");

  const endpoint = endpointPage.data?.find((candidate) => candidate.url === expectedWebhookUrl);
  assert(endpoint, "No Stripe test webhook endpoint matches MYCELLIOS_STRIPE_SANDBOX_WEBHOOK_URL.");
  assert(endpoint.status === "enabled", "Stripe test webhook endpoint is disabled.");
  const enabledEvents = new Set(endpoint.enabled_events ?? []);
  const receivesAllEvents = enabledEvents.has("*");
  const missingEvents = receivesAllEvents
    ? []
    : [...REQUIRED_EVENTS].filter((event) => !enabledEvents.has(event));
  assert(missingEvents.length === 0, `Stripe webhook is missing events: ${missingEvents.join(", ")}`);

  process.stdout.write(`${JSON.stringify({
    ok: true,
    mode: "test",
    accountCountry: account.country,
    chargesEnabled: account.charges_enabled,
    price: {
      id: price.id,
      active: price.active,
      currency: price.currency,
      unitAmount: price.unit_amount,
      interval: price.recurring.interval,
    },
    webhook: {
      id: endpoint.id,
      status: endpoint.status,
      requiredEvents: REQUIRED_EVENTS.size,
    },
  }, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`Stripe sandbox verification failed: ${error.message}\n`);
  process.exitCode = 1;
});
