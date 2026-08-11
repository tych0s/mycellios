import { createHash } from "node:crypto";
import { newId } from "../core/ids.js";
import type { MeshDatabase } from "../storage/database.js";
import type { BillingManager } from "./billing.js";
import { BillingCheckoutError } from "./billing-checkout.js";

export interface StablecoinPaymentGateway {
  allocatePayment(input: {
    intentId: string;
    userId: string;
    chainId: string;
    asset: string;
    assetAtomicAmount: string;
    expiresAt: number;
    idempotencyKey: string;
  }): Promise<{
    providerReference: string;
    recipient: string;
    checkoutUrl?: string;
  }>;
}

export interface StablecoinTopUpPack {
  packId: string;
  amountMicros: number;
  currency: string;
  tokenAmount: number;
  assetAtomicAmount: string;
}

export interface StablecoinCheckoutConfig {
  chainId: string;
  asset: string;
  subscriptionAssetAtomicAmount: string;
  subscriptionPeriodMs: number;
  intentTtlMs?: number;
  planId: string;
  planVersion: number;
  topUpPacks: readonly StablecoinTopUpPack[];
  now?: () => number;
}

export interface StablecoinCheckoutResult {
  id: string;
  kind: "subscription" | "topup";
  chainId: string;
  asset: string;
  assetAtomicAmount: string;
  recipient: string;
  expiresAt: number;
  providerReference: string;
  checkoutUrl?: string;
  quoteId?: string;
  duplicate: boolean;
}

interface IntentRow {
  id: string;
  request_digest: string;
  kind: "subscription" | "topup";
  quote_id: string | null;
  amount_micros: number;
  currency: string;
  chain_id: string;
  asset: string;
  asset_atomic_amount: string;
  recipient: string | null;
  provider_reference: string | null;
  checkout_url: string | null;
  status: "allocating" | "awaiting_payment" | "consumed";
  expires_at: number;
}

export class StablecoinCheckoutService {
  private readonly packs: ReadonlyMap<string, StablecoinTopUpPack>;

  constructor(
    private readonly database: MeshDatabase,
    private readonly billing: BillingManager,
    private readonly gateway: StablecoinPaymentGateway,
    private readonly config: StablecoinCheckoutConfig,
  ) {
    validateIdentifier(config.chainId, "chain id");
    validateIdentifier(config.asset, "asset");
    validateAtomicAmount(config.subscriptionAssetAtomicAmount);
    if (!Number.isSafeInteger(config.subscriptionPeriodMs) || config.subscriptionPeriodMs <= 0) {
      throw new BillingCheckoutError("invalid_stablecoin_period", "The stablecoin subscription period is invalid.", 503);
    }
    const ttl = config.intentTtlMs ?? 30 * 60_000;
    if (!Number.isSafeInteger(ttl) || ttl <= 0) {
      throw new BillingCheckoutError("invalid_stablecoin_ttl", "The stablecoin intent TTL is invalid.", 503);
    }
    this.packs = new Map(config.topUpPacks.map((pack) => [pack.packId, pack]));
    if (this.packs.size !== config.topUpPacks.length) {
      throw new BillingCheckoutError("duplicate_stablecoin_pack", "Stablecoin pack ids must be unique.", 503);
    }
    for (const pack of config.topUpPacks) validatePack(pack);
  }

  async createSubscription(input: {
    userId: string;
    idempotencyKey: string;
  }): Promise<StablecoinCheckoutResult> {
    const plan = this.billing.getPlan(this.config.planId, this.config.planVersion);
    if (!plan || plan.status !== "active") {
      throw new BillingCheckoutError("billing_plan_not_active", "The stablecoin subscription plan is not active.", 503);
    }
    return this.createIntent({
      userId: input.userId,
      idempotencyKey: input.idempotencyKey,
      kind: "subscription",
      amountMicros: plan.priceMicros,
      currency: plan.priceCurrency,
      assetAtomicAmount: this.config.subscriptionAssetAtomicAmount,
      economicInput: {
        planId: plan.planId,
        planVersion: plan.version,
        amountMicros: plan.priceMicros,
        currency: plan.priceCurrency,
      },
      planId: plan.planId,
      planVersion: plan.version,
      subscriptionId: `stablecoin:${input.userId}`,
      periodDurationMs: this.config.subscriptionPeriodMs,
    });
  }

  async createTopUp(input: {
    userId: string;
    packId: string;
    idempotencyKey: string;
  }): Promise<StablecoinCheckoutResult> {
    const pack = this.packs.get(input.packId);
    if (!pack) throw new BillingCheckoutError("stablecoin_pack_not_found", "The stablecoin top-up pack is unavailable.", 404);
    return this.createIntent({
      userId: input.userId,
      idempotencyKey: input.idempotencyKey,
      kind: "topup",
      amountMicros: pack.amountMicros,
      currency: pack.currency,
      assetAtomicAmount: pack.assetAtomicAmount,
      tokenAmount: pack.tokenAmount,
      economicInput: { packId: pack.packId },
    });
  }

  private async createIntent(input: {
    userId: string;
    idempotencyKey: string;
    kind: "subscription" | "topup";
    amountMicros: number;
    currency: string;
    assetAtomicAmount: string;
    economicInput: object;
    tokenAmount?: number;
    planId?: string;
    planVersion?: number;
    subscriptionId?: string;
    periodDurationMs?: number;
  }): Promise<StablecoinCheckoutResult> {
    validateIdempotencyKey(input.idempotencyKey);
    const digest = createHash("sha256").update(JSON.stringify([
      input.kind,
      input.economicInput,
      this.config.chainId,
      this.config.asset,
      input.assetAtomicAmount,
    ])).digest("hex");
    const now = this.config.now?.() ?? Date.now();
    const ttl = this.config.intentTtlMs ?? 30 * 60_000;
    let intent = this.database.transaction(() => {
      const existing = this.readIntent(input.userId, input.idempotencyKey);
      if (existing) {
        if (existing.request_digest !== digest) {
          throw new BillingCheckoutError(
            "stablecoin_idempotency_conflict",
            "The idempotency key was already used for a different stablecoin payment.",
            409,
          );
        }
        return existing;
      }
      ensureAccount(this.database, input.userId, now);
      const id = newId("coinpay");
      const expiresAt = now + ttl;
      let quoteId: string | null = null;
      if (input.kind === "topup") {
        quoteId = `quote_${createHash("sha256").update(id).digest("hex").slice(0, 32)}`;
        this.billing.createTopUpQuote({
          quoteId,
          userId: input.userId,
          amountMicros: input.amountMicros,
          currency: input.currency,
          tokenAmount: input.tokenAmount!,
          expiresAt,
        });
      }
      this.database.raw.prepare(
        `INSERT INTO billing_stablecoin_intents(
           id, user_id, idempotency_key, request_digest, kind, quote_id,
           plan_id, plan_version, subscription_id, period_duration_ms,
           amount_micros, currency, chain_id, asset, asset_atomic_amount,
           status, expires_at, created_at, updated_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'allocating', ?, ?, ?)`,
      ).run(
        id, input.userId, input.idempotencyKey, digest, input.kind, quoteId,
        input.planId ?? null, input.planVersion ?? null, input.subscriptionId ?? null,
        input.periodDurationMs ?? null, input.amountMicros, input.currency.toUpperCase(),
        this.config.chainId, this.config.asset.toUpperCase(), input.assetAtomicAmount,
        expiresAt, now, now,
      );
      return this.readIntent(input.userId, input.idempotencyKey)!;
    });
    if (intent.status === "awaiting_payment" || intent.status === "consumed") {
      return resultFromRow(intent, true);
    }
    if (now >= intent.expires_at) {
      throw new BillingCheckoutError("stablecoin_intent_expired", "The stablecoin payment intent has expired.", 409);
    }
    const allocated = await this.gateway.allocatePayment({
      intentId: intent.id,
      userId: input.userId,
      chainId: intent.chain_id,
      asset: intent.asset,
      assetAtomicAmount: intent.asset_atomic_amount,
      expiresAt: intent.expires_at,
      idempotencyKey: `mycellios-${intent.id}`,
    });
    validateProviderReference(allocated.providerReference);
    validateRecipient(allocated.recipient);
    if (allocated.checkoutUrl !== undefined) validateHttpsUrl(allocated.checkoutUrl);
    this.database.raw.prepare(
      `UPDATE billing_stablecoin_intents
       SET recipient = ?, provider_reference = ?, checkout_url = ?,
           status = 'awaiting_payment', updated_at = ?
       WHERE id = ? AND status = 'allocating'`,
    ).run(
      allocated.recipient, allocated.providerReference, allocated.checkoutUrl ?? null,
      now, intent.id,
    );
    intent = this.readIntent(input.userId, input.idempotencyKey)!;
    return resultFromRow(intent, false);
  }

  private readIntent(userId: string, idempotencyKey: string): IntentRow | null {
    return (this.database.raw.prepare(
      `SELECT id, request_digest, kind, quote_id, amount_micros, currency,
              chain_id, asset, asset_atomic_amount, recipient, provider_reference,
              checkout_url, status, expires_at
       FROM billing_stablecoin_intents WHERE user_id = ? AND idempotency_key = ?`,
    ).get(userId, idempotencyKey) as IntentRow | undefined) ?? null;
  }
}

function resultFromRow(row: IntentRow, duplicate: boolean): StablecoinCheckoutResult {
  if (!row.recipient || !row.provider_reference) {
    throw new BillingCheckoutError("stablecoin_intent_corrupt", "The stablecoin payment intent is incomplete.", 500);
  }
  return {
    id: row.id,
    kind: row.kind,
    chainId: row.chain_id,
    asset: row.asset,
    assetAtomicAmount: row.asset_atomic_amount,
    recipient: row.recipient,
    expiresAt: Number(row.expires_at),
    providerReference: row.provider_reference,
    ...(row.checkout_url ? { checkoutUrl: row.checkout_url } : {}),
    ...(row.quote_id ? { quoteId: row.quote_id } : {}),
    duplicate,
  };
}

function ensureAccount(database: MeshDatabase, userId: string, now: number): void {
  database.raw.prepare(
    `INSERT OR IGNORE INTO api_accounts(user_id, token_balance, usd_micros, created_at, updated_at)
     VALUES (?, 0, 0, ?, ?)`,
  ).run(userId, now, now);
}

function validatePack(pack: StablecoinTopUpPack): void {
  validateIdentifier(pack.packId, "pack id");
  if (!Number.isSafeInteger(pack.amountMicros) || pack.amountMicros <= 0
    || !Number.isSafeInteger(pack.tokenAmount) || pack.tokenAmount <= 0
    || !/^[A-Z]{3}$/.test(pack.currency.toUpperCase())) {
    throw new BillingCheckoutError("invalid_stablecoin_pack", "A stablecoin pack value is invalid.", 503);
  }
  validateAtomicAmount(pack.assetAtomicAmount);
}

function validateIdentifier(value: string, label: string): void {
  if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$/.test(value)) {
    throw new BillingCheckoutError("invalid_stablecoin_config", `The stablecoin ${label} is invalid.`, 503);
  }
}

function validateAtomicAmount(value: string): void {
  if (!/^[1-9][0-9]{0,77}$/.test(value)) {
    throw new BillingCheckoutError("invalid_stablecoin_amount", "The stablecoin atomic amount is invalid.", 503);
  }
}

function validateIdempotencyKey(value: string): void {
  if (!/^[\x21-\x7E]{1,128}$/.test(value)) {
    throw new BillingCheckoutError("invalid_idempotency_key", "Idempotency-Key must contain 1-128 printable ASCII characters.");
  }
}

function validateProviderReference(value: string): void {
  if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$/.test(value)) {
    throw new BillingCheckoutError("invalid_stablecoin_provider_response", "The payment provider reference is invalid.", 502);
  }
}

function validateRecipient(value: string): void {
  if (value.length < 8 || value.length > 256 || /\s/.test(value)) {
    throw new BillingCheckoutError("invalid_stablecoin_provider_response", "The payment recipient is invalid.", 502);
  }
}

function validateHttpsUrl(value: string): void {
  let parsed: URL;
  try { parsed = new URL(value); } catch {
    throw new BillingCheckoutError("invalid_stablecoin_provider_response", "The provider checkout URL is invalid.", 502);
  }
  if (parsed.protocol !== "https:" || parsed.username || parsed.password) {
    throw new BillingCheckoutError("invalid_stablecoin_provider_response", "The provider checkout URL is invalid.", 502);
  }
}
