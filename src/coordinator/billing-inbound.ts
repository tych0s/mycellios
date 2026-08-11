import {
  createHmac,
  createPublicKey,
  timingSafeEqual,
  verify,
} from "node:crypto";
import type {
  BillingApplicationResult,
  BillingManager,
  BillingReversalResult,
  PaidSubscriptionEvent,
  PaidTopUpEvent,
  PaymentReversalEvent,
  SubscriptionStatusEvent,
  SubscriptionStatusResult,
} from "./billing.js";

export type NormalizedBillingCommand =
  | { kind: "paid_subscription"; event: PaidSubscriptionEvent }
  | { kind: "paid_topup"; event: PaidTopUpEvent }
  | { kind: "payment_reversal"; event: PaymentReversalEvent }
  | { kind: "subscription_status"; event: SubscriptionStatusEvent };

export type BillingCommandResult =
  | BillingApplicationResult
  | BillingReversalResult
  | SubscriptionStatusResult;

export class BillingInboundError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "BillingInboundError";
  }
}

export function applyBillingCommand(
  manager: BillingManager,
  command: NormalizedBillingCommand,
): BillingCommandResult {
  switch (command.kind) {
    case "paid_subscription": return manager.applyPaidSubscription(command.event);
    case "paid_topup": return manager.applyPaidTopUp(command.event);
    case "payment_reversal": return manager.applyPaymentReversal(command.event);
    case "subscription_status": return manager.applySubscriptionStatus(command.event);
  }
}

interface StripeEnvelope {
  id: string;
  type: string;
  created: number;
  livemode: boolean;
  data: { object: Record<string, unknown> };
}

export class StripeBillingInbound {
  constructor(private readonly config: {
    endpointSecrets: readonly string[];
    toleranceSeconds?: number;
    expectedLivemode: boolean;
    nowSeconds?: () => number;
  }) {
    if (config.endpointSecrets.length === 0 || config.endpointSecrets.some((secret) => secret.length < 16)) {
      throw new BillingInboundError("invalid_stripe_secret_config", "At least one non-trivial Stripe endpoint secret is required.");
    }
    const tolerance = config.toleranceSeconds ?? 300;
    if (!Number.isSafeInteger(tolerance) || tolerance <= 0) {
      throw new BillingInboundError("invalid_stripe_tolerance", "Stripe signature tolerance must be positive.");
    }
  }

  verifyAndNormalize(rawBody: Buffer, signatureHeader: string): NormalizedBillingCommand | null {
    const envelope = this.verifyEnvelope(rawBody, signatureHeader);
    if (envelope.livemode !== this.config.expectedLivemode) {
      throw new BillingInboundError("stripe_mode_mismatch", "The Stripe event livemode does not match this endpoint.");
    }
    const object = envelope.data.object;
    switch (envelope.type) {
      case "invoice.paid":
        return { kind: "paid_subscription", event: stripePaidSubscription(envelope, object) };
      case "checkout.session.completed":
        return { kind: "paid_topup", event: stripePaidTopUp(envelope, object) };
      case "invoice.payment_failed":
        return { kind: "subscription_status", event: stripeSubscriptionStatus(envelope, object, "past_due") };
      case "customer.subscription.updated":
        return stripeUpdatedSubscription(envelope, object);
      case "customer.subscription.deleted":
        return { kind: "subscription_status", event: stripeSubscriptionStatus(envelope, object, "cancelled") };
      case "charge.refunded":
        return { kind: "payment_reversal", event: stripePaymentReversal(envelope, object) };
      default:
        return null;
    }
  }

  private verifyEnvelope(rawBody: Buffer, signatureHeader: string): StripeEnvelope {
    if (rawBody.length === 0 || rawBody.length > 1_000_000) {
      throw new BillingInboundError("invalid_stripe_payload_size", "The Stripe webhook body size is invalid.");
    }
    const parts = signatureHeader.split(",").map((part) => part.trim());
    const timestampPart = parts.find((part) => part.startsWith("t="));
    const signatures = parts.filter((part) => part.startsWith("v1=")).map((part) => part.slice(3));
    const timestamp = Number(timestampPart?.slice(2));
    if (!Number.isSafeInteger(timestamp) || timestamp <= 0 || signatures.length === 0) {
      throw new BillingInboundError("invalid_stripe_signature_header", "The Stripe-Signature header is malformed.");
    }
    const now = this.config.nowSeconds?.() ?? Math.floor(Date.now() / 1_000);
    const tolerance = this.config.toleranceSeconds ?? 300;
    if (Math.abs(now - timestamp) > tolerance) {
      throw new BillingInboundError("stripe_signature_expired", "The Stripe signature timestamp is outside tolerance.");
    }
    const signedPayload = Buffer.concat([Buffer.from(`${timestamp}.`, "utf8"), rawBody]);
    const valid = this.config.endpointSecrets.some((secret) => {
      const expected = createHmac("sha256", secret).update(signedPayload).digest();
      return signatures.some((candidate) => safeHexEqual(expected, candidate));
    });
    if (!valid) {
      throw new BillingInboundError("invalid_stripe_signature", "The Stripe webhook signature is invalid.");
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(rawBody.toString("utf8"));
    } catch {
      throw new BillingInboundError("invalid_stripe_json", "The signed Stripe body is not valid JSON.");
    }
    const envelope = asRecord(parsed, "Stripe event");
    const data = asRecord(envelope.data, "Stripe event data");
    return {
      id: requiredId(envelope.id, "Stripe event id"),
      type: requiredString(envelope.type, "Stripe event type"),
      created: requiredPositiveInteger(envelope.created, "Stripe created timestamp"),
      livemode: requiredBoolean(envelope.livemode, "Stripe livemode"),
      data: { object: asRecord(data.object, "Stripe data object") },
    };
  }
}

function stripeUpdatedSubscription(
  envelope: StripeEnvelope,
  object: Record<string, unknown>,
): NormalizedBillingCommand | null {
  const providerStatus = requiredString(object.status, "Stripe subscription status");
  const status = providerStatus === "past_due" || providerStatus === "unpaid" || providerStatus === "paused"
    ? "past_due"
    : providerStatus === "canceled"
      ? "cancelled"
      : null;
  if (!status) return null;
  return {
    kind: "subscription_status",
    event: stripeSubscriptionStatus(envelope, object, status),
  };
}

export interface StablecoinWatcherPayload {
  schema: "mycellios.stablecoin-payment.v1";
  eventId: string;
  kind: "topup_paid" | "subscription_paid";
  chainId: string;
  asset: string;
  txHash: string;
  logIndex: number;
  blockNumber: number;
  blockHash: string;
  confirmations: number;
  finality: "finalized";
  userId: string;
  amountMicros: number;
  currency: string;
  quoteId?: string;
  tokenAmount?: number;
  planId?: string;
  planVersion?: number;
  subscriptionId?: string;
  periodStart?: number;
  periodEnd?: number;
  occurredAt: number;
  watcherKeyId: string;
  paymentIntentId: string;
  assetAtomicAmount: string;
  recipient: string;
}

export interface SignedStablecoinWatcherEvent extends StablecoinWatcherPayload {
  signature: string;
}

export class StablecoinBillingInbound {
  constructor(private readonly config: {
    trustedWatcherKeys: ReadonlyMap<string, string>;
    chains: ReadonlyMap<string, { asset: string; minimumConfirmations: number }>;
  }) {
    if (config.trustedWatcherKeys.size === 0 || config.chains.size === 0) {
      throw new BillingInboundError("invalid_stablecoin_config", "Stablecoin watcher keys and chains must be configured.");
    }
    for (const [keyId, publicKey] of config.trustedWatcherKeys) {
      requiredId(keyId, "stablecoin watcher key id");
      try {
        if (createPublicKey(publicKey).asymmetricKeyType !== "ed25519") throw new Error("wrong key type");
      } catch {
        throw new BillingInboundError("invalid_stablecoin_config", "A stablecoin watcher key is not Ed25519.");
      }
    }
    for (const [chainId, chain] of config.chains) {
      requiredId(chainId, "stablecoin chain id");
      requiredId(chain.asset, "stablecoin asset");
      requiredPositiveInteger(chain.minimumConfirmations, "stablecoin minimum confirmations");
    }
  }

  verifyAndNormalize(input: SignedStablecoinWatcherEvent): NormalizedBillingCommand {
    validateStablecoinPayload(input);
    const chain = this.config.chains.get(input.chainId);
    if (!chain) throw new BillingInboundError("stablecoin_chain_not_allowed", "The stablecoin chain is not allowed.");
    if (chain.asset.toUpperCase() !== input.asset.toUpperCase()) {
      throw new BillingInboundError("stablecoin_asset_not_allowed", "The stablecoin asset is not allowed on this chain.");
    }
    if (input.confirmations < chain.minimumConfirmations) {
      throw new BillingInboundError("stablecoin_not_final", "The watcher event has insufficient confirmations.");
    }
    const publicKey = this.config.trustedWatcherKeys.get(input.watcherKeyId);
    if (!publicKey) throw new BillingInboundError("untrusted_stablecoin_watcher", "The stablecoin watcher is not trusted.");
    let valid = false;
    try {
      valid = verify(
        null,
        stablecoinWatcherSigningBytes(input),
        createPublicKey(publicKey),
        Buffer.from(input.signature, "base64url"),
      );
    } catch {
      valid = false;
    }
    if (!valid) throw new BillingInboundError("invalid_stablecoin_signature", "The stablecoin watcher signature is invalid.");

    const providerEventId = `${input.chainId}:${input.txHash}:${input.logIndex}`;
    if (input.eventId !== providerEventId) {
      throw new BillingInboundError("stablecoin_event_identity_mismatch", "The watcher event id is not canonical.");
    }
    if (input.kind === "topup_paid") {
      return {
        kind: "paid_topup",
        event: {
          provider: "stablecoin",
          providerEventId,
          paymentReference: input.txHash,
          quoteId: requiredId(input.quoteId, "stablecoin quote id"),
          userId: input.userId,
          amountMicros: input.amountMicros,
          currency: input.currency,
          tokenAmount: requiredPositiveInteger(input.tokenAmount, "stablecoin top-up tokens"),
          occurredAt: input.occurredAt,
          finalized: true,
          stablecoinPaymentIntentId: input.paymentIntentId,
          stablecoinChainId: input.chainId,
          stablecoinAsset: input.asset,
          stablecoinAssetAtomicAmount: input.assetAtomicAmount,
          stablecoinRecipient: input.recipient,
        },
      };
    }
    return {
      kind: "paid_subscription",
      event: {
        provider: "stablecoin",
        providerEventId,
        providerSubscriptionId: requiredId(input.subscriptionId, "stablecoin subscription id"),
        userId: input.userId,
        planId: requiredId(input.planId, "stablecoin plan id"),
        planVersion: requiredPositiveInteger(input.planVersion, "stablecoin plan version"),
        amountMicros: input.amountMicros,
        currency: input.currency,
        periodStart: requiredPositiveInteger(input.periodStart, "stablecoin period start"),
        periodEnd: requiredPositiveInteger(input.periodEnd, "stablecoin period end"),
        occurredAt: input.occurredAt,
        finalized: true,
        stablecoinPaymentIntentId: input.paymentIntentId,
        stablecoinChainId: input.chainId,
        stablecoinAsset: input.asset,
        stablecoinAssetAtomicAmount: input.assetAtomicAmount,
        stablecoinRecipient: input.recipient,
      },
    };
  }
}

export function stablecoinWatcherSigningBytes(input: StablecoinWatcherPayload): Buffer {
  return Buffer.from(JSON.stringify([
    input.schema, input.eventId, input.kind, input.chainId, input.asset,
    input.txHash, input.logIndex, input.blockNumber, input.blockHash,
    input.confirmations, input.finality, input.userId, input.amountMicros,
    input.currency, input.quoteId ?? null, input.tokenAmount ?? null, input.planId ?? null,
    input.planVersion ?? null, input.subscriptionId ?? null,
    input.periodStart ?? null, input.periodEnd ?? null, input.occurredAt,
    input.watcherKeyId, input.paymentIntentId, input.assetAtomicAmount,
    input.recipient,
  ]), "utf8");
}

function stripePaidSubscription(envelope: StripeEnvelope, object: Record<string, unknown>): PaidSubscriptionEvent {
  const metadata = stripeSubscriptionMetadata(object);
  return {
    provider: "stripe",
    providerEventId: envelope.id,
    providerSubscriptionId: stripeInvoiceSubscriptionId(object),
    userId: requiredMetadataId(metadata, "mycellios_user_id"),
    planId: requiredMetadataId(metadata, "mycellios_plan_id"),
    planVersion: requiredMetadataInteger(metadata, "mycellios_plan_version"),
    amountMicros: stripeMinorToMicros(object.amount_paid, "Stripe invoice amount_paid"),
    currency: requiredString(object.currency, "Stripe invoice currency").toUpperCase(),
    periodStart: stripeInvoicePeriod(object.period_start, metadata, "mycellios_period_start_ms"),
    periodEnd: stripeInvoicePeriod(object.period_end, metadata, "mycellios_period_end_ms"),
    occurredAt: envelope.created * 1_000,
    finalized: true,
  };
}

function stripePaidTopUp(envelope: StripeEnvelope, object: Record<string, unknown>): PaidTopUpEvent {
  if (object.mode !== "payment" || object.payment_status !== "paid") {
    throw new BillingInboundError("stripe_topup_not_paid", "The Checkout Session is not a finalized one-time payment.");
  }
  const metadata = requiredMetadata(object);
  return {
    provider: "stripe",
    providerEventId: envelope.id,
    paymentReference: requiredExpandableId(object.payment_intent, "Stripe PaymentIntent"),
    quoteId: requiredMetadataId(metadata, "mycellios_quote_id"),
    userId: requiredMetadataId(metadata, "mycellios_user_id"),
    amountMicros: stripeMinorToMicros(object.amount_total, "Stripe Checkout amount_total"),
    currency: requiredString(object.currency, "Stripe Checkout currency").toUpperCase(),
    tokenAmount: requiredMetadataInteger(metadata, "mycellios_token_amount"),
    occurredAt: envelope.created * 1_000,
    finalized: true,
  };
}

function stripeSubscriptionStatus(
  envelope: StripeEnvelope,
  object: Record<string, unknown>,
  status: "past_due" | "cancelled",
): SubscriptionStatusEvent {
  const subscriptionObject = envelope.type === "customer.subscription.deleted"
    || envelope.type === "customer.subscription.updated";
  const metadata = subscriptionObject
    ? requiredMetadata(object)
    : stripeSubscriptionMetadata(object);
  const subscriptionId = subscriptionObject
    ? requiredId(object.id, "Stripe subscription id")
    : stripeInvoiceSubscriptionId(object);
  return {
    provider: "stripe",
    providerEventId: envelope.id,
    providerSubscriptionId: subscriptionId,
    userId: requiredMetadataId(metadata, "mycellios_user_id"),
    status,
    occurredAt: envelope.created * 1_000,
    finalized: true,
  };
}

function stripePaymentReversal(envelope: StripeEnvelope, object: Record<string, unknown>): PaymentReversalEvent {
  const metadata = requiredMetadata(object);
  return {
    provider: "stripe",
    providerEventId: envelope.id,
    originalProviderEventId: requiredMetadataId(metadata, "mycellios_original_event_id"),
    userId: requiredMetadataId(metadata, "mycellios_user_id"),
    reason: "refund",
    tokenAmount: requiredMetadataInteger(metadata, "mycellios_reversed_token_amount"),
    occurredAt: envelope.created * 1_000,
    finalized: true,
  };
}

function validateStablecoinPayload(input: SignedStablecoinWatcherEvent): void {
  if (input.schema !== "mycellios.stablecoin-payment.v1") {
    throw new BillingInboundError("unsupported_stablecoin_schema", "The stablecoin watcher schema is unsupported.");
  }
  if (input.finality !== "finalized") {
    throw new BillingInboundError("stablecoin_not_final", "The stablecoin watcher event is not finalized.");
  }
  for (const [label, value] of [
    ["event id", input.eventId], ["chain id", input.chainId], ["asset", input.asset],
    ["transaction hash", input.txHash], ["block hash", input.blockHash],
    ["user id", input.userId], ["watcher key id", input.watcherKeyId],
    ["payment intent id", input.paymentIntentId],
  ] as const) requiredId(value, label);
  requiredPositiveInteger(input.blockNumber, "stablecoin block number");
  if (!Number.isSafeInteger(input.logIndex) || input.logIndex < 0) {
    throw new BillingInboundError("invalid_stablecoin_log_index", "The stablecoin log index is invalid.");
  }
  if (!Number.isSafeInteger(input.confirmations) || input.confirmations < 0) {
    throw new BillingInboundError("invalid_stablecoin_confirmations", "Stablecoin confirmations are invalid.");
  }
  requiredPositiveInteger(input.amountMicros, "stablecoin amount");
  requiredPositiveInteger(input.occurredAt, "stablecoin occurredAt");
  if (!/^[1-9][0-9]{0,77}$/.test(input.assetAtomicAmount)) {
    throw new BillingInboundError("invalid_stablecoin_atomic_amount", "Stablecoin atomic amount is invalid.");
  }
  if (input.recipient.length < 8 || input.recipient.length > 256 || /\s/.test(input.recipient)) {
    throw new BillingInboundError("invalid_stablecoin_recipient", "Stablecoin recipient is invalid.");
  }
  if (!/^[A-Z]{3,8}$/.test(input.currency.toUpperCase())) {
    throw new BillingInboundError("invalid_stablecoin_currency", "The stablecoin currency code is invalid.");
  }
  if (!/^[A-Za-z0-9_-]{80,120}$/.test(input.signature)) {
    throw new BillingInboundError("invalid_stablecoin_signature", "The stablecoin watcher signature encoding is invalid.");
  }
}

function requiredMetadata(object: Record<string, unknown>): Record<string, unknown> {
  return asRecord(object.metadata, "Stripe metadata");
}

function stripeSubscriptionMetadata(object: Record<string, unknown>): Record<string, unknown> {
  const parent = optionalRecord(object.parent);
  const subscriptionDetails = optionalRecord(parent?.subscription_details);
  const nested = optionalRecord(subscriptionDetails?.metadata);
  const direct = optionalRecord(object.metadata);
  const metadata = { ...(direct ?? {}), ...(nested ?? {}) };
  if (Object.keys(metadata).length === 0) {
    throw new BillingInboundError("invalid_inbound_object", "Stripe subscription metadata must be an object.");
  }
  return metadata;
}

function stripeInvoiceSubscriptionId(object: Record<string, unknown>): string {
  if (object.subscription !== undefined && object.subscription !== null) {
    return requiredExpandableId(object.subscription, "Stripe subscription");
  }
  const parent = optionalRecord(object.parent);
  const subscriptionDetails = optionalRecord(parent?.subscription_details);
  return requiredExpandableId(subscriptionDetails?.subscription, "Stripe subscription");
}

function stripeInvoicePeriod(
  seconds: unknown,
  metadata: Record<string, unknown>,
  legacyMetadataKey: string,
): number {
  if (typeof seconds === "number" && Number.isSafeInteger(seconds) && seconds > 0) {
    const milliseconds = seconds * 1_000;
    if (Number.isSafeInteger(milliseconds)) return milliseconds;
  }
  return requiredMetadataInteger(metadata, legacyMetadataKey);
}

function optionalRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function requiredMetadataId(metadata: Record<string, unknown>, key: string): string {
  return requiredId(metadata[key], `Stripe metadata ${key}`);
}

function requiredMetadataInteger(metadata: Record<string, unknown>, key: string): number {
  const value = typeof metadata[key] === "string" ? Number(metadata[key]) : metadata[key];
  return requiredPositiveInteger(value, `Stripe metadata ${key}`);
}

function stripeMinorToMicros(value: unknown, label: string): number {
  const minor = requiredPositiveInteger(value, label);
  const micros = minor * 10_000;
  if (!Number.isSafeInteger(micros)) throw new BillingInboundError("stripe_amount_overflow", `${label} is too large.`);
  return micros;
}

function requiredExpandableId(value: unknown, label: string): string {
  if (typeof value === "string") return requiredId(value, label);
  return requiredId(asRecord(value, label).id, label);
}

function requiredId(value: unknown, label: string): string {
  const text = requiredString(value, label);
  if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$/.test(text)) {
    throw new BillingInboundError("invalid_inbound_identifier", `${label} is invalid.`);
  }
  return text;
}

function requiredString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new BillingInboundError("invalid_inbound_field", `${label} must be a non-empty string.`);
  }
  return value;
}

function requiredPositiveInteger(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0) {
    throw new BillingInboundError("invalid_inbound_integer", `${label} must be a positive safe integer.`);
  }
  return value;
}

function requiredBoolean(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") throw new BillingInboundError("invalid_inbound_field", `${label} must be boolean.`);
  return value;
}

function asRecord(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new BillingInboundError("invalid_inbound_object", `${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function safeHexEqual(expected: Buffer, candidate: string): boolean {
  if (!/^[a-f0-9]{64}$/i.test(candidate)) return false;
  const received = Buffer.from(candidate, "hex");
  return received.length === expected.length && timingSafeEqual(expected, received);
}
