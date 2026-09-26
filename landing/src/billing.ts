export type BillingSubscriptionStatus = "active" | "past_due" | "cancelled";

export interface BillingAccountOverview {
  object: "billing_account";
  checkoutAvailable: boolean;
  portalAvailable: boolean;
  topUpsAvailable: boolean;
  tokenDebt: number;
  plan: {
    id: string;
    version: number;
    currency: string;
    amountMicros: number;
    includedTokens: number;
    status: "draft" | "active" | "retired";
  } | null;
  subscription: {
    planId: string;
    planVersion: number;
    provider: "stripe" | "stablecoin";
    status: BillingSubscriptionStatus;
    periodStart: number;
    periodEnd: number;
    statusChangedAt: number;
  } | null;
}

interface BillingRedirect {
  object: "billing_checkout_session" | "billing_portal_session";
  id: string;
  url: string;
  duplicate: boolean;
}

export function loadBillingAccount(accessToken: string): Promise<BillingAccountOverview> {
  return billingRequest<BillingAccountOverview>("/v1/billing/account", accessToken);
}

export function createSubscriptionCheckout(accessToken: string, idempotencyKey: string): Promise<BillingRedirect> {
  return billingRequest<BillingRedirect>("/v1/billing/checkout/subscription", accessToken, {
    method: "POST",
    headers: { "idempotency-key": idempotencyKey, "content-type": "application/json" },
    body: "{}",
  });
}

export function createBillingPortal(accessToken: string, idempotencyKey: string): Promise<BillingRedirect> {
  return billingRequest<BillingRedirect>("/v1/billing/portal", accessToken, {
    method: "POST",
    headers: { "idempotency-key": idempotencyKey, "content-type": "application/json" },
    body: "{}",
  });
}

export function assertStripeRedirect(value: string, kind: "checkout" | "portal"): string {
  const url = new URL(value);
  const expected = kind === "checkout" ? "checkout.stripe.com" : "billing.stripe.com";
  if (url.protocol !== "https:" || (url.hostname !== expected && !url.hostname.endsWith(`.${expected}`))) {
    throw new Error("Stripe returned an unexpected redirect destination.");
  }
  return url.toString();
}

export function newBillingIdempotencyKey(kind: "subscription" | "portal"): string {
  const random = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  return `mycellios-${kind}-${random}`;
}

async function billingRequest<T>(path: string, accessToken: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      cache: "no-store",
      signal: init.signal ? AbortSignal.any([init.signal, AbortSignal.timeout(30_000)]) : AbortSignal.timeout(30_000),
      headers: { ...init.headers, authorization: `Bearer ${accessToken}` },
    });
  } catch (error) {
    if (error instanceof Error && error.name === "TimeoutError") throw new Error("The billing service did not respond. Try again.");
    throw error;
  }
  const body = await response.json().catch(() => null) as {
    error?: { code?: string; message?: string };
  } | null;
  if (!response.ok) {
    throw new Error(body?.error?.message ?? body?.error?.code ?? `Billing request failed (HTTP ${response.status}).`);
  }
  return body as T;
}
