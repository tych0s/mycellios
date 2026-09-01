#!/usr/bin/env node

import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";

const REQUIRED_WEBHOOK_EVENTS = [
  "checkout.session.completed",
  "invoice.paid",
  "invoice.payment_failed",
  "customer.subscription.updated",
  "customer.subscription.deleted",
  "charge.refunded",
];

function secretFromEnvironment() {
  const direct = process.env.MYCELLIOS_STRIPE_SECRET_KEY?.trim();
  const file = process.env.MYCELLIOS_STRIPE_SECRET_KEY_FILE?.trim();
  if (direct && file) throw new Error("Configure the Stripe test key directly or by file, never both.");
  const value = direct ?? (file ? readFileSync(file, "utf8").trim() : "");
  if (!/^(sk|rk)_test_/.test(value)) {
    throw new Error("Sandbox provisioning requires a Stripe test credential.");
  }
  return value;
}

async function stripeRequest(path, secretKey, body) {
  const response = await fetch(`https://api.stripe.com${path}`, {
    method: body ? "POST" : "GET",
    headers: {
      Authorization: `Bearer ${secretKey}`,
      ...(body ? { "Content-Type": "application/x-www-form-urlencoded" } : {}),
    },
    body,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(`Stripe ${path} failed (${response.status}): ${payload?.error?.code ?? "unknown_error"}`);
  }
  return payload;
}

function retainWebhookSecret(secret) {
  const projectId = process.env.MYCELLIOS_INFISICAL_PROJECT_ID?.trim();
  const environment = process.env.MYCELLIOS_INFISICAL_ENVIRONMENT?.trim();
  const path = process.env.MYCELLIOS_INFISICAL_PATH?.trim();
  if (!projectId || !environment || !path) {
    throw new Error("Webhook creation requires MYCELLIOS_INFISICAL_PROJECT_ID, _ENVIRONMENT and _PATH.");
  }
  const result = spawnSync("infisical", [
    "secrets", "set", "--silent",
    "--projectId", projectId,
    "--env", environment,
    "--path", path,
    `MYCELLIOS_STRIPE_WEBHOOK_SECRET=${secret}`,
    "MYCELLIOS_STRIPE_WEBHOOK_LIVEMODE=false",
  ], { encoding: "utf8", stdio: "pipe" });
  if (result.status !== 0) {
    throw new Error("Unable to retain the Stripe webhook secret in Infisical.");
  }
}

async function provisionWebhook(secretKey) {
  const webhookUrl = process.env.MYCELLIOS_STRIPE_SANDBOX_WEBHOOK_URL?.trim();
  if (!webhookUrl) return { configured: false };
  const canonicalUrl = new URL(webhookUrl).toString();
  const endpoints = await stripeRequest("/v1/webhook_endpoints?limit=100", secretKey);
  let endpoint = endpoints.data.find((candidate) => candidate.url === canonicalUrl);
  let created = false;
  if (!endpoint) {
    const form = new URLSearchParams({
      url: canonicalUrl,
      description: "Mycellios Stripe sandbox",
    });
    for (const event of REQUIRED_WEBHOOK_EVENTS) form.append("enabled_events[]", event);
    endpoint = await stripeRequest("/v1/webhook_endpoints", secretKey, form);
    retainWebhookSecret(endpoint.secret);
    created = true;
  }
  if (endpoint.livemode !== false) throw new Error("Stripe webhook endpoint is not in test mode.");
  const enabled = new Set(endpoint.enabled_events ?? []);
  const missing = enabled.has("*") ? [] : REQUIRED_WEBHOOK_EVENTS.filter((event) => !enabled.has(event));
  if (missing.length > 0) throw new Error(`Stripe webhook is missing events: ${missing.join(", ")}`);
  return { configured: true, id: endpoint.id, created, eventCount: REQUIRED_WEBHOOK_EVENTS.length };
}

async function main() {
  const secretKey = secretFromEnvironment();
  const account = await stripeRequest("/v1/account", secretKey);

  const products = await stripeRequest("/v1/products?active=true&limit=100", secretKey);
  let product = products.data.find((entry) => entry.metadata?.mycellios_role === "go_subscription");
  let productCreated = false;
  if (!product) {
    product = await stripeRequest("/v1/products", secretKey, new URLSearchParams({
      name: "Mycellios Go",
      description: "Mycellios Go monthly subscription",
      "metadata[mycellios_role]": "go_subscription",
      "metadata[managed_by]": "mycellios",
    }));
    productCreated = true;
  }
  if (product.livemode !== false) throw new Error("Stripe Product is not in test mode.");

  const prices = await stripeRequest(`/v1/prices?active=true&limit=100&product=${encodeURIComponent(product.id)}`, secretKey);
  let price = prices.data.find((entry) => entry.currency === "eur"
    && entry.unit_amount === 1000
    && entry.type === "recurring"
    && entry.recurring?.interval === "month"
    && entry.recurring?.interval_count === 1);
  let priceCreated = false;
  if (!price) {
    price = await stripeRequest("/v1/prices", secretKey, new URLSearchParams({
      product: product.id,
      currency: "eur",
      unit_amount: "1000",
      "recurring[interval]": "month",
      "recurring[interval_count]": "1",
      "metadata[mycellios_role]": "go_subscription",
      "metadata[managed_by]": "mycellios",
    }));
    priceCreated = true;
  }
  if (price.livemode !== false) throw new Error("Stripe Price is not in test mode.");
  const webhook = await provisionWebhook(secretKey);

  process.stdout.write(`${JSON.stringify({
    ok: true,
    mode: "test",
    accountCountry: account.country,
    productId: product.id,
    productCreated,
    priceId: price.id,
    priceCreated,
    currency: price.currency,
    unitAmount: price.unit_amount,
    interval: price.recurring.interval,
    webhook,
  })}\n`);
}

main().catch((error) => {
  process.stderr.write(`Stripe sandbox provisioning failed: ${error.message}\n`);
  process.exitCode = 1;
});
