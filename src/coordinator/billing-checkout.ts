import { createHash } from "node:crypto";
import { newId } from "../core/ids.js";
import type { MeshDatabase } from "../storage/database.js";
import type { BillingManager, BillingPlan } from "./billing.js";

export interface StripeCheckoutSessionResult {
  id: string;
  url: string;
}

export interface StripeCheckoutGateway {
  createCustomer(input: {
    userId: string;
    email: string | null;
    idempotencyKey: string;
  }): Promise<{ id: string }>;
  createSubscriptionCheckout(input: {
    userId: string;
    customerId: string;
    plan: BillingPlan;
    priceId: string;
    successUrl: string;
    cancelUrl: string;
    idempotencyKey: string;
  }): Promise<StripeCheckoutSessionResult>;
  createTopUpCheckout(input: {
    userId: string;
    customerId: string;
    quoteId: string;
    tokenAmount: number;
    priceId: string;
    successUrl: string;
    cancelUrl: string;
    idempotencyKey: string;
  }): Promise<StripeCheckoutSessionResult>;
  createPortalSession(input: {
    customerId: string;
    returnUrl: string;
    idempotencyKey: string;
  }): Promise<StripeCheckoutSessionResult>;
}

export interface BillingTopUpPack {
  packId: string;
  amountMicros: number;
  currency: string;
  tokenAmount: number;
  stripePriceId: string;
}

export interface BillingCheckoutServiceConfig {
  planId: string;
  planVersion: number;
  subscriptionPriceId: string;
  successUrl: string;
  cancelUrl: string;
  portalReturnUrl: string;
  topUpPacks: readonly BillingTopUpPack[];
  quoteTtlMs?: number;
  subscriptionCheckoutTtlMs?: number;
  now?: () => number;
}

export interface BillingCheckoutResult extends StripeCheckoutSessionResult {
  duplicate: boolean;
  quoteId?: string;
}

export class BillingCheckoutError extends Error {
  constructor(readonly code: string, message: string, readonly statusCode = 400) {
    super(message);
    this.name = "BillingCheckoutError";
  }
}

interface OperationRow {
  id: string;
  request_digest: string;
  quote_id: string | null;
  status: "pending" | "completed";
  provider_session_id: string | null;
  provider_url: string | null;
  created_at: number;
}

export class BillingCheckoutService {
  private readonly packs: ReadonlyMap<string, BillingTopUpPack>;

  constructor(
    private readonly database: MeshDatabase,
    private readonly billing: BillingManager,
    private readonly stripe: StripeCheckoutGateway,
    private readonly config: BillingCheckoutServiceConfig,
  ) {
    this.packs = new Map(config.topUpPacks.map((pack) => [pack.packId, pack]));
    if (this.packs.size !== config.topUpPacks.length) {
      throw new BillingCheckoutError("duplicate_topup_pack", "Top-up pack ids must be unique.");
    }
    validateStripeId(config.subscriptionPriceId, "subscription price");
    for (const url of [config.successUrl, config.cancelUrl, config.portalReturnUrl]) validateHttpsUrl(url);
    for (const pack of config.topUpPacks) validatePack(pack);
  }

  async createSubscription(input: {
    userId: string; email: string | null; idempotencyKey: string;
  }): Promise<BillingCheckoutResult> {
    const plan = this.billing.getPlan(this.config.planId, this.config.planVersion);
    if (!plan || plan.status !== "active") {
      throw new BillingCheckoutError("billing_plan_not_active", "The configured subscription plan is not active.", 503);
    }
    const subscription = this.billing.getSubscription(input.userId);
    if (subscription && subscription.status !== "cancelled") {
      throw new BillingCheckoutError(
        "subscription_already_exists",
        subscription.status === "past_due"
          ? "Resolve the existing subscription in the billing portal."
          : "This account already has an active subscription.",
        409,
      );
    }
    const resumable = this.resumableSubscriptionCheckout(input.userId);
    if (resumable) return { ...resumable, duplicate: true };
    const operation = this.beginOperation(input.userId, input.idempotencyKey, "subscription", {
      planId: plan.planId,
      planVersion: plan.version,
      priceId: this.config.subscriptionPriceId,
    });
    const completed = completedResult(operation);
    if (completed) return completed;
    const customerId = await this.customerFor(input.userId, input.email);
    const session = await this.stripe.createSubscriptionCheckout({
      userId: input.userId,
      customerId,
      plan,
      priceId: this.config.subscriptionPriceId,
      successUrl: this.config.successUrl,
      cancelUrl: this.config.cancelUrl,
      idempotencyKey: providerOperationKey(operation.id),
    });
    return this.completeOperation(operation, session);
  }

  hasCustomer(userId: string): boolean {
    return Boolean(this.database.raw.prepare(
      "SELECT 1 AS found FROM billing_customers WHERE provider = 'stripe' AND user_id = ?",
    ).get(userId));
  }

  hasTopUpPacks(): boolean {
    return this.packs.size > 0;
  }

  private resumableSubscriptionCheckout(userId: string): StripeCheckoutSessionResult | null {
    const row = this.database.raw.prepare(
      `SELECT provider_session_id, provider_url, created_at
       FROM billing_checkout_operations
       WHERE user_id = ? AND kind = 'subscription' AND status = 'completed'
       ORDER BY created_at DESC, id DESC LIMIT 1`,
    ).get(userId) as {
      provider_session_id: string | null; provider_url: string | null; created_at: number;
    } | undefined;
    if (!row?.provider_session_id || !row.provider_url) return null;
    const now = this.config.now?.() ?? Date.now();
    if (now >= Number(row.created_at) + (this.config.subscriptionCheckoutTtlMs ?? 24 * 60 * 60_000)) {
      return null;
    }
    return { id: row.provider_session_id, url: row.provider_url };
  }

  async createTopUp(input: {
    userId: string; email: string | null; packId: string; idempotencyKey: string;
  }): Promise<BillingCheckoutResult> {
    const pack = this.packs.get(input.packId);
    if (!pack) throw new BillingCheckoutError("topup_pack_not_found", "The selected top-up pack is unavailable.", 404);
    const operation = this.beginOperation(input.userId, input.idempotencyKey, "topup", { packId: input.packId });
    const completed = completedResult(operation);
    if (completed) return completed;
    const quoteId = operation.quote_id ?? deterministicQuoteId(operation.id);
    const now = this.config.now?.() ?? Date.now();
    const expiresAt = operation.created_at + (this.config.quoteTtlMs ?? 30 * 60_000);
    if (now >= expiresAt) {
      throw new BillingCheckoutError("checkout_operation_expired", "The pending top-up checkout has expired.", 409);
    }
    this.billing.createTopUpQuote({
      quoteId,
      userId: input.userId,
      amountMicros: pack.amountMicros,
      currency: pack.currency,
      tokenAmount: pack.tokenAmount,
      expiresAt,
    });
    if (!operation.quote_id) {
      this.database.raw.prepare(
        "UPDATE billing_checkout_operations SET quote_id = ?, updated_at = ? WHERE id = ? AND quote_id IS NULL",
      ).run(quoteId, now, operation.id);
      operation.quote_id = quoteId;
    }
    const customerId = await this.customerFor(input.userId, input.email);
    const session = await this.stripe.createTopUpCheckout({
      userId: input.userId,
      customerId,
      quoteId,
      tokenAmount: pack.tokenAmount,
      priceId: pack.stripePriceId,
      successUrl: this.config.successUrl,
      cancelUrl: this.config.cancelUrl,
      idempotencyKey: providerOperationKey(operation.id),
    });
    return this.completeOperation(operation, session, quoteId);
  }

  async createPortal(input: {
    userId: string; email: string | null; idempotencyKey: string;
  }): Promise<BillingCheckoutResult> {
    const operation = this.beginOperation(input.userId, input.idempotencyKey, "portal", {
      returnUrl: this.config.portalReturnUrl,
    });
    const completed = completedResult(operation);
    if (completed) return completed;
    const customerId = await this.customerFor(input.userId, input.email);
    const session = await this.stripe.createPortalSession({
      customerId,
      returnUrl: this.config.portalReturnUrl,
      idempotencyKey: providerOperationKey(operation.id),
    });
    return this.completeOperation(operation, session);
  }

  private beginOperation(
    userId: string,
    idempotencyKey: string,
    kind: "subscription" | "topup" | "portal",
    economicInput: object,
  ): OperationRow {
    validateIdempotencyKey(idempotencyKey);
    const digest = createHash("sha256").update(JSON.stringify([kind, economicInput])).digest("hex");
    return this.database.transaction(() => {
      const existing = this.database.raw.prepare(
        `SELECT id, request_digest, quote_id, status, provider_session_id, provider_url, created_at
         FROM billing_checkout_operations WHERE user_id = ? AND idempotency_key = ?`,
      ).get(userId, idempotencyKey) as OperationRow | undefined;
      if (existing) {
        if (existing.request_digest !== digest) {
          throw new BillingCheckoutError(
            "checkout_idempotency_conflict",
            "The idempotency key was already used for different checkout parameters.",
            409,
          );
        }
        return existing;
      }
      const now = this.config.now?.() ?? Date.now();
      const id = newId("checkout");
      // Ensure the FK account exists before reserving the operation key.
      this.database.raw.prepare(
        `INSERT OR IGNORE INTO api_accounts(
           user_id, token_balance, usd_micros, created_at, updated_at
         ) VALUES (?, 0, 0, ?, ?)`,
      ).run(userId, now, now);
      this.database.raw.prepare(
        `INSERT INTO billing_checkout_operations(
           id, user_id, idempotency_key, kind, request_digest, status, created_at, updated_at
         ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)`,
      ).run(id, userId, idempotencyKey, kind, digest, now, now);
      return {
        id, request_digest: digest, quote_id: null, status: "pending",
        provider_session_id: null, provider_url: null, created_at: now,
      };
    });
  }

  private async customerFor(userId: string, email: string | null): Promise<string> {
    const existing = this.database.raw.prepare(
      "SELECT provider_customer_id FROM billing_customers WHERE provider = 'stripe' AND user_id = ?",
    ).get(userId) as { provider_customer_id: string } | undefined;
    if (existing) return existing.provider_customer_id;
    const created = await this.stripe.createCustomer({
      userId,
      email,
      idempotencyKey: `mycellios-customer-${createHash("sha256").update(userId).digest("hex")}`,
    });
    validateStripeId(created.id, "customer");
    const now = this.config.now?.() ?? Date.now();
    try {
      this.database.raw.prepare(
        `INSERT INTO billing_customers(provider, user_id, provider_customer_id, created_at, updated_at)
         VALUES ('stripe', ?, ?, ?, ?)`,
      ).run(userId, created.id, now, now);
    } catch {
      const raced = this.database.raw.prepare(
        "SELECT provider_customer_id FROM billing_customers WHERE provider = 'stripe' AND user_id = ?",
      ).get(userId) as { provider_customer_id: string } | undefined;
      if (raced?.provider_customer_id === created.id) return created.id;
      throw new BillingCheckoutError("stripe_customer_conflict", "The Stripe customer mapping conflicts with this account.", 409);
    }
    return created.id;
  }

  private completeOperation(
    operation: OperationRow,
    session: StripeCheckoutSessionResult,
    quoteId?: string,
  ): BillingCheckoutResult {
    validateStripeId(session.id, "session");
    validateHttpsUrl(session.url);
    const now = this.config.now?.() ?? Date.now();
    this.database.raw.prepare(
      `UPDATE billing_checkout_operations
       SET status = 'completed', provider_session_id = ?, provider_url = ?, updated_at = ?
       WHERE id = ? AND status = 'pending'`,
    ).run(session.id, session.url, now, operation.id);
    return { ...session, ...(quoteId ? { quoteId } : {}), duplicate: false };
  }
}

export class StripeRestGateway implements StripeCheckoutGateway {
  constructor(private readonly config: {
    secretKey: string;
    apiBaseUrl?: string;
    fetch?: typeof fetch;
  }) {
    if (!/^sk_(test|live)_[A-Za-z0-9_]{8,}$/.test(config.secretKey)) {
      throw new BillingCheckoutError("invalid_stripe_api_key", "The Stripe API key configuration is invalid.", 503);
    }
  }

  createCustomer(input: Parameters<StripeCheckoutGateway["createCustomer"]>[0]) {
    return this.post("/v1/customers", {
      ...(input.email ? { email: input.email } : {}),
      "metadata[mycellios_user_id]": input.userId,
    }, input.idempotencyKey);
  }

  createSubscriptionCheckout(input: Parameters<StripeCheckoutGateway["createSubscriptionCheckout"]>[0]) {
    const metadata = {
      mycellios_user_id: input.userId,
      mycellios_plan_id: input.plan.planId,
      mycellios_plan_version: String(input.plan.version),
    };
    return this.post("/v1/checkout/sessions", {
      mode: "subscription", customer: input.customerId,
      "line_items[0][price]": input.priceId, "line_items[0][quantity]": "1",
      success_url: input.successUrl, cancel_url: input.cancelUrl,
      client_reference_id: input.userId,
      ...stripeMetadata("metadata", metadata),
      ...stripeMetadata("subscription_data[metadata]", metadata),
    }, input.idempotencyKey);
  }

  createTopUpCheckout(input: Parameters<StripeCheckoutGateway["createTopUpCheckout"]>[0]) {
    const metadata = {
      mycellios_user_id: input.userId,
      mycellios_quote_id: input.quoteId,
      mycellios_token_amount: String(input.tokenAmount),
    };
    return this.post("/v1/checkout/sessions", {
      mode: "payment", customer: input.customerId,
      "line_items[0][price]": input.priceId, "line_items[0][quantity]": "1",
      success_url: input.successUrl, cancel_url: input.cancelUrl,
      client_reference_id: input.userId,
      ...stripeMetadata("metadata", metadata),
      ...stripeMetadata("payment_intent_data[metadata]", metadata),
    }, input.idempotencyKey);
  }

  createPortalSession(input: Parameters<StripeCheckoutGateway["createPortalSession"]>[0]) {
    return this.post("/v1/billing_portal/sessions", {
      customer: input.customerId,
      return_url: input.returnUrl,
    }, input.idempotencyKey);
  }

  private async post(path: string, fields: Record<string, string>, idempotencyKey: string) {
    const request = this.config.fetch ?? fetch;
    const response = await request(new URL(path, this.config.apiBaseUrl ?? "https://api.stripe.com"), {
      method: "POST",
      signal: AbortSignal.timeout(15_000),
      headers: {
        authorization: `Bearer ${this.config.secretKey}`,
        "content-type": "application/x-www-form-urlencoded",
        "idempotency-key": idempotencyKey,
      },
      body: new URLSearchParams(fields),
    });
    const body = await response.json() as Record<string, unknown>;
    if (!response.ok) {
      const error = typeof body.error === "object" && body.error !== null
        ? body.error as Record<string, unknown>
        : {};
      throw new BillingCheckoutError(
        "stripe_api_error",
        typeof error.message === "string" ? error.message : `Stripe returned HTTP ${response.status}.`,
        502,
      );
    }
    if (typeof body.id !== "string") {
      throw new BillingCheckoutError("invalid_stripe_response", "Stripe returned an invalid object.", 502);
    }
    return {
      id: body.id,
      ...(typeof body.url === "string" ? { url: body.url } : {}),
    } as { id: string; url: string };
  }
}

function completedResult(operation: OperationRow): BillingCheckoutResult | null {
  if (operation.status !== "completed") return null;
  if (!operation.provider_session_id || !operation.provider_url) {
    throw new BillingCheckoutError("checkout_operation_corrupt", "The completed checkout operation is incomplete.", 500);
  }
  return {
    id: operation.provider_session_id,
    url: operation.provider_url,
    ...(operation.quote_id ? { quoteId: operation.quote_id } : {}),
    duplicate: true,
  };
}

function providerOperationKey(operationId: string): string {
  return `mycellios-${operationId}`;
}

function deterministicQuoteId(operationId: string): string {
  return `quote_${createHash("sha256").update(operationId).digest("hex").slice(0, 32)}`;
}

function stripeMetadata(prefix: string, values: Record<string, string>): Record<string, string> {
  return Object.fromEntries(Object.entries(values).map(([key, value]) => [`${prefix}[${key}]`, value]));
}

function validateIdempotencyKey(value: string): void {
  if (!/^[\x21-\x7E]{1,128}$/.test(value)) {
    throw new BillingCheckoutError("invalid_idempotency_key", "Idempotency-Key must contain 1-128 printable ASCII characters.");
  }
}

function validateStripeId(value: string, label: string): void {
  if (!/^[A-Za-z0-9][A-Za-z0-9_]{2,191}$/.test(value)) {
    throw new BillingCheckoutError("invalid_stripe_identifier", `The Stripe ${label} id is invalid.`, 502);
  }
}

function validateHttpsUrl(value: string): void {
  let parsed: URL;
  try { parsed = new URL(value); } catch {
    throw new BillingCheckoutError("invalid_billing_url", "Billing redirect URLs must be valid HTTPS URLs.", 503);
  }
  if (parsed.protocol !== "https:" || parsed.username || parsed.password) {
    throw new BillingCheckoutError("invalid_billing_url", "Billing redirect URLs must be valid HTTPS URLs.", 503);
  }
}

function validatePack(pack: BillingTopUpPack): void {
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(pack.packId)) {
    throw new BillingCheckoutError("invalid_topup_pack", "A top-up pack id is invalid.", 503);
  }
  if (!Number.isSafeInteger(pack.amountMicros) || pack.amountMicros <= 0
    || !Number.isSafeInteger(pack.tokenAmount) || pack.tokenAmount <= 0
    || !/^[A-Z]{3}$/.test(pack.currency.toUpperCase())) {
    throw new BillingCheckoutError("invalid_topup_pack", "A top-up pack value is invalid.", 503);
  }
  validateStripeId(pack.stripePriceId, "top-up price");
}
