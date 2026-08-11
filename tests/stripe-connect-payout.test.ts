import { describe, expect, it, vi } from "vitest";
import { StripeConnectPayoutGateway } from "../src/coordinator/stripe-connect-payout.js";

const API_VERSION = "2025-10-29.clover";

describe("Stripe Connect stable payout gateway", () => {
  it("creates an exact USD transfer with the durable dispatch identity", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) =>
      jsonResponse(200, transfer("payout:batch-1", "tr_transfer123")));
    const gateway = service(fetchMock);
    const result = await gateway.createTransfer({
      batchId: "batch-1",
      sellerId: "seller-1",
      payoutMethod: "stable",
      amountUsdMicros: 12_340_000,
      destinationReference: "acct_1234567890",
      destinationFingerprint: "a".repeat(64),
      dispatchKey: "payout:batch-1",
    });
    expect(result).toEqual({ state: "submitted", externalReference: "tr_transfer123" });
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(String(url)).toBe("https://stripe.test/v1/transfers");
    expect(init?.headers).toMatchObject({
      "idempotency-key": "payout:batch-1",
      "stripe-version": API_VERSION,
    });
    const form = new URLSearchParams(String(init?.body));
    expect(Object.fromEntries(form)).toMatchObject({
      amount: "1234",
      currency: "usd",
      destination: "acct_1234567890",
      transfer_group: "payout:batch-1",
      "metadata[mycellios_batch_id]": "batch-1",
      "metadata[mycellios_destination_fingerprint]": "a".repeat(64),
    });
  });

  it("rejects non-cent amounts and invalid connected accounts before network I/O", async () => {
    const fetchMock = vi.fn();
    const gateway = service(fetchMock);
    const base = {
      batchId: "batch-1", sellerId: "seller-1", payoutMethod: "stable" as const,
      destinationReference: "acct_1234567890", destinationFingerprint: "a".repeat(64),
      dispatchKey: "payout:batch-1",
    };
    await expect(gateway.createTransfer({ ...base, amountUsdMicros: 12_340_001 })).resolves.toEqual({
      state: "rejected", reason: "amount_not_cent_aligned",
    });
    await expect(gateway.createTransfer({
      ...base, amountUsdMicros: 12_340_000, destinationReference: "bank-account-user-input",
    })).resolves.toEqual({ state: "rejected", reason: "invalid_stripe_connected_account" });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("recovers an uncertain create by its unique transfer group", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => jsonResponse(200, {
      object: "list", has_more: false, data: [transfer("payout:batch-timeout", "tr_timeout123")],
    }));
    const result = await service(fetchMock).inspectTransfer({
      dispatchKey: "payout:batch-timeout", externalReference: null,
    });
    expect(result).toEqual({ state: "submitted", externalReference: "tr_timeout123" });
    expect(String(fetchMock.mock.calls[0]![0])).toContain(
      "/v1/transfers?transfer_group=payout%3Abatch-timeout&limit=2",
    );
  });

  it("fails closed on duplicate or identity-mismatched provider transfers", async () => {
    const duplicateFetch = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => jsonResponse(200, {
      object: "list",
      data: [transfer("payout:batch-1", "tr_duplicate1"), transfer("payout:batch-1", "tr_duplicate2")],
    }));
    await expect(service(duplicateFetch).inspectTransfer({
      dispatchKey: "payout:batch-1", externalReference: null,
    })).resolves.toEqual({ state: "rejected", reason: "duplicate_stripe_transfers" });

    const mismatchFetch = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) =>
      jsonResponse(200, transfer("payout:another", "tr_mismatch1")));
    await expect(service(mismatchFetch).inspectTransfer({
      dispatchKey: "payout:batch-1", externalReference: "tr_mismatch1",
    })).rejects.toThrow("invalid_stripe_transfer_identity");
  });

  it("distinguishes definitive Stripe rejection from uncertain provider failure", async () => {
    const rejectedFetch = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => jsonResponse(400, {
      error: { code: "insufficient_funds", message: "redacted" },
    }));
    await expect(service(rejectedFetch).createTransfer(validInput())).resolves.toEqual({
      state: "rejected", reason: "stripe_insufficient_funds",
    });
    const uncertainFetch = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) =>
      jsonResponse(500, { error: { code: "api_error" } }));
    await expect(service(uncertainFetch).createTransfer(validInput())).rejects.toThrow(
      "stripe_connect_uncertain_http:500",
    );
  });

  function service(fetchMock: typeof fetch | ReturnType<typeof vi.fn>) {
    return new StripeConnectPayoutGateway({
      secretKey: "sk_test_1234567890abcdef",
      apiVersion: API_VERSION,
      apiBaseUrl: "https://stripe.test",
      fetch: fetchMock as typeof fetch,
    });
  }
});

function validInput() {
  return {
    batchId: "batch-1", sellerId: "seller-1", payoutMethod: "stable" as const,
    amountUsdMicros: 12_340_000, destinationReference: "acct_1234567890",
    destinationFingerprint: "a".repeat(64), dispatchKey: "payout:batch-1",
  };
}

function transfer(dispatchKey: string, id: string) {
  return {
    id, object: "transfer", amount: 1234, amount_reversed: 0, currency: "usd",
    destination: "acct_1234567890", transfer_group: dispatchKey, reversed: false,
    metadata: { mycellios_dispatch_key: dispatchKey },
  };
}

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}
