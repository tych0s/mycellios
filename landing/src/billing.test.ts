import { afterEach, describe, expect, it, vi } from "vitest";
import {
  assertStripeRedirect,
  createBillingPortal,
  createSubscriptionCheckout,
  loadBillingAccount,
  newBillingIdempotencyKey,
} from "./billing";
import { billingPriceLabel } from "./BillingAccountPanel";

describe("buyer billing client", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("loads only the signed-in account projection", async () => {
    const fetch = vi.fn(async () => new Response(JSON.stringify({
      object: "billing_account", checkoutAvailable: true, portalAvailable: false,
      topUpsAvailable: false, tokenDebt: 0, plan: null, subscription: null,
    }), { status: 200, headers: { "content-type": "application/json" } }));
    vi.stubGlobal("fetch", fetch);
    await expect(loadBillingAccount("session-token")).resolves.toMatchObject({ object: "billing_account" });
    expect(fetch).toHaveBeenCalledWith("/v1/billing/account", expect.objectContaining({
      cache: "no-store", headers: expect.objectContaining({ authorization: "Bearer session-token" }),
    }));
  });

  it("sends explicit idempotency for Checkout and Portal", async () => {
    const fetch = vi.fn(async (url: RequestInfo | URL, _init?: RequestInit) => new Response(JSON.stringify({
      object: String(url).includes("portal") ? "billing_portal_session" : "billing_checkout_session",
      id: "cs_test_1", url: "https://checkout.stripe.com/c/pay/cs_test_1", duplicate: false,
    }), { status: 201, headers: { "content-type": "application/json" } }));
    vi.stubGlobal("fetch", fetch);
    await createSubscriptionCheckout("session-token", "subscription-attempt");
    await createBillingPortal("session-token", "portal-attempt");
    expect(fetch.mock.calls[0]?.[1]).toEqual(expect.objectContaining({
      method: "POST",
      headers: expect.objectContaining({ "idempotency-key": "subscription-attempt" }),
    }));
    expect(fetch.mock.calls[1]?.[1]).toEqual(expect.objectContaining({
      method: "POST",
      headers: expect.objectContaining({ "idempotency-key": "portal-attempt" }),
    }));
  });

  it("accepts only the expected Stripe HTTPS redirect surfaces", () => {
    expect(assertStripeRedirect("https://checkout.stripe.com/c/pay/cs_test_1", "checkout"))
      .toBe("https://checkout.stripe.com/c/pay/cs_test_1");
    expect(assertStripeRedirect("https://billing.stripe.com/p/session/test", "portal"))
      .toBe("https://billing.stripe.com/p/session/test");
    expect(() => assertStripeRedirect("https://evil.example/checkout", "checkout"))
      .toThrow("unexpected redirect destination");
    expect(() => assertStripeRedirect("http://checkout.stripe.com/c/pay/test", "checkout"))
      .toThrow("unexpected redirect destination");
  });

  it("creates scoped, unique operation identities", () => {
    const first = newBillingIdempotencyKey("subscription");
    const second = newBillingIdempotencyKey("subscription");
    expect(first).toMatch(/^mycellios-subscription-/);
    expect(second).not.toBe(first);
  });

  it("does not invent a price when the subscription plan is unavailable", () => {
    expect(billingPriceLabel(null)).toBe("Price unavailable");
    expect(billingPriceLabel({ id: "go", version: 1, currency: "EUR", amountMicros: 10_000_000, includedTokens: 1_000, status: "draft" }))
      .toBe("Price unavailable");
    expect(billingPriceLabel({ id: "go", version: 1, currency: "EUR", amountMicros: 12_000_000, includedTokens: 1_000, status: "active" }))
      .toContain("12");
  });

  it("ends a stalled billing request with a readable error", async () => {
    const fetch = vi.fn().mockRejectedValue(new DOMException("Timed out", "TimeoutError"));
    vi.stubGlobal("fetch", fetch);
    await expect(loadBillingAccount("session-token"))
      .rejects.toThrow("The billing service did not respond. Try again.");
    expect(fetch.mock.calls[0]?.[1]?.signal).toBeInstanceOf(AbortSignal);
  });
});
