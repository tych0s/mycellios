import type { PayoutGateway, PayoutProviderObservation } from "./payout-dispatch.js";

const USD_MICROS_PER_CENT = 10_000;

export class StripeConnectPayoutGateway implements PayoutGateway {
  constructor(private readonly config: {
    secretKey: string;
    apiVersion: string;
    apiBaseUrl?: string;
    fetch?: typeof fetch;
  }) {
    if (!/^sk_(test|live)_[A-Za-z0-9_]{8,}$/.test(config.secretKey)) {
      throw new Error("invalid_stripe_connect_api_key");
    }
    if (!/^\d{4}-\d{2}-\d{2}\.[a-z]+$/.test(config.apiVersion)) {
      throw new Error("invalid_stripe_connect_api_version");
    }
  }

  async createTransfer(input: Parameters<PayoutGateway["createTransfer"]>[0]) {
    if (input.payoutMethod !== "stable") {
      return { state: "rejected", reason: "unsupported_payout_method" } as const;
    }
    if (!input.destinationReference || !/^acct_[A-Za-z0-9]{8,}$/.test(input.destinationReference)) {
      return { state: "rejected", reason: "invalid_stripe_connected_account" } as const;
    }
    if (!input.destinationFingerprint || !/^[a-f0-9]{64}$/.test(input.destinationFingerprint)) {
      return { state: "rejected", reason: "invalid_destination_fingerprint" } as const;
    }
    if (!Number.isSafeInteger(input.amountUsdMicros) || input.amountUsdMicros <= 0
      || input.amountUsdMicros % USD_MICROS_PER_CENT !== 0) {
      return { state: "rejected", reason: "amount_not_cent_aligned" } as const;
    }
    const response = await this.request("POST", "/v1/transfers", input.dispatchKey, {
      amount: String(input.amountUsdMicros / USD_MICROS_PER_CENT),
      currency: "usd",
      destination: input.destinationReference,
      transfer_group: input.dispatchKey,
      "metadata[mycellios_batch_id]": input.batchId,
      "metadata[mycellios_seller_id]": input.sellerId,
      "metadata[mycellios_destination_fingerprint]": input.destinationFingerprint,
      "metadata[mycellios_dispatch_key]": input.dispatchKey,
    });
    if (!response.ok) return rejectedOrThrow(response);
    return transferObservation(response.body, input.dispatchKey);
  }

  async inspectTransfer(input: Parameters<PayoutGateway["inspectTransfer"]>[0]) {
    if (input.externalReference) {
      if (!/^tr_[A-Za-z0-9]{8,}$/.test(input.externalReference)) {
        return { state: "rejected", reason: "invalid_stripe_transfer_reference" } as const;
      }
      const response = await this.request(
        "GET",
        `/v1/transfers/${encodeURIComponent(input.externalReference)}`,
      );
      if (response.status === 404) return { state: "absent" } as const;
      if (!response.ok) return rejectedOrThrow(response);
      return transferObservation(response.body, input.dispatchKey);
    }

    const query = new URLSearchParams({ transfer_group: input.dispatchKey, limit: "2" });
    const response = await this.request("GET", `/v1/transfers?${query}`);
    if (!response.ok) return rejectedOrThrow(response);
    const data = Array.isArray(response.body.data) ? response.body.data : null;
    if (!data) throw new Error("invalid_stripe_transfer_list");
    if (data.length === 0) return { state: "absent" } as const;
    if (data.length > 1) return { state: "rejected", reason: "duplicate_stripe_transfers" } as const;
    return transferObservation(data[0], input.dispatchKey);
  }

  private async request(
    method: "GET" | "POST",
    path: string,
    idempotencyKey?: string,
    fields?: Record<string, string>,
  ): Promise<{ ok: boolean; status: number; body: Record<string, unknown> }> {
    const request = this.config.fetch ?? fetch;
    const response = await request(new URL(path, this.config.apiBaseUrl ?? "https://api.stripe.com"), {
      method,
      signal: AbortSignal.timeout(15_000),
      headers: {
        authorization: `Bearer ${this.config.secretKey}`,
        "stripe-version": this.config.apiVersion,
        ...(idempotencyKey ? { "idempotency-key": idempotencyKey } : {}),
        ...(fields ? { "content-type": "application/x-www-form-urlencoded" } : {}),
      },
      ...(fields ? { body: new URLSearchParams(fields) } : {}),
    });
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      throw new Error(`stripe_connect_invalid_json:${response.status}`);
    }
    if (typeof body !== "object" || body === null || Array.isArray(body)) {
      throw new Error("stripe_connect_invalid_response");
    }
    return { ok: response.ok, status: response.status, body: body as Record<string, unknown> };
  }
}

function transferObservation(body: unknown, dispatchKey: string): Exclude<PayoutProviderObservation, { state: "absent" }> {
  if (typeof body !== "object" || body === null || Array.isArray(body)) {
    throw new Error("invalid_stripe_transfer");
  }
  const transfer = body as Record<string, unknown>;
  const metadata = typeof transfer.metadata === "object" && transfer.metadata !== null
    ? transfer.metadata as Record<string, unknown>
    : {};
  if (typeof transfer.id !== "string" || !/^tr_[A-Za-z0-9]{8,}$/.test(transfer.id)
    || transfer.object !== "transfer"
    || transfer.transfer_group !== dispatchKey
    || metadata.mycellios_dispatch_key !== dispatchKey) {
    throw new Error("invalid_stripe_transfer_identity");
  }
  if (transfer.reversed === true || (typeof transfer.amount_reversed === "number" && transfer.amount_reversed > 0)) {
    return { state: "rejected", reason: "stripe_transfer_reversed" };
  }
  return { state: "submitted", externalReference: transfer.id };
}

function rejectedOrThrow(response: {
  status: number;
  body: Record<string, unknown>;
}): { state: "rejected"; reason: string } {
  if (response.status >= 400 && response.status < 500 && response.status !== 409 && response.status !== 429) {
    const stripeError = typeof response.body.error === "object" && response.body.error !== null
      ? response.body.error as Record<string, unknown>
      : {};
    const code = typeof stripeError.code === "string" && /^[a-z0-9_]{1,80}$/.test(stripeError.code)
      ? stripeError.code
      : `http_${response.status}`;
    return { state: "rejected", reason: `stripe_${code}` };
  }
  throw new Error(`stripe_connect_uncertain_http:${response.status}`);
}
