import { readFileSync } from "node:fs";
import { createPublicKey } from "node:crypto";
import { isIP } from "node:net";
import { resolve } from "node:path";
import { workerConfigSchema, type WorkerConfig } from "../contracts/schemas.js";

export interface CoordinatorConfig {
  host: string;
  port: number;
  databasePath: string;
  requestTimeoutMs: number;
  networkToken?: string | undefined;
  internalToken?: string | undefined;
  modelAdminToken?: string | undefined;
  mobileJoinToken?: string | undefined;
  mobileAssetsPath?: string | undefined;
  landingAssetsPath?: string | undefined;
  nodeUpdatesPath?: string | undefined;
  componentUpdatesPath?: string | undefined;
  componentUpdatePinnedKeys?: {
    dev?: readonly { keyId: string; spki: string }[];
    stable?: readonly { keyId: string; spki: string }[];
  } | undefined;
  modelCertificationPinnedKeys?: readonly { keyId: string; spki: string }[] | undefined;
  releaseDownloadsPath?: string | undefined;
  mobileExpertArtifactsPath?: string | undefined;
  contentHubApiUrl?: string | undefined;
  publicationWebhookSecret?: string | undefined;
  supabaseUrl?: string | undefined;
  publicSupabaseUrl?: string | undefined;
  supabaseServiceRoleKey?: string | undefined;
  supabaseAnonKey?: string | undefined;
  supabasePersistenceRequired?: boolean | undefined;
  /** Explicit test/development escape hatch; production defaults to native adapters only. */
  allowDevelopmentAdapters?: boolean | undefined;
  apiAccessEnabled?: boolean | undefined;
  apiStarterTokens?: number | undefined;
  apiRequestsPerMinute?: number | undefined;
  apiMaxConcurrent?: number | undefined;
  apiMaxActiveKeys?: number | undefined;
  publicApiBaseUrl?: string | undefined;
  stripeWebhookSecrets?: readonly string[] | undefined;
  stripeWebhookLivemode?: boolean | undefined;
  stripeCheckout?: {
    secretKey: string;
    subscriptionPriceId: string;
    includedTokens: number;
    successUrl: string;
    cancelUrl: string;
    portalReturnUrl: string;
    topUpPacks: readonly {
      packId: string;
      amountMicros: number;
      currency: string;
      tokenAmount: number;
      stripePriceId: string;
    }[];
  } | undefined;
  stablecoinWatcher?: {
    trustedWatcherKeys: readonly { keyId: string; publicKey: string }[];
    chains: readonly { chainId: string; asset: string; minimumConfirmations: number }[];
  } | undefined;
  sellerPayout?: {
    minimumUsdMicros: number;
    spore: {
      legalApproved: boolean;
      custodyApproved: boolean;
      liquidityApproved: boolean;
      antifraudApproved: boolean;
    };
    destinationVerifierKeys: readonly { keyId: string; publicKey: string }[];
    settlementVerifierKeys: readonly { keyId: string; publicKey: string }[];
    settlementEvidenceMaxAgeMs: number;
    sporeConversion?: {
      trustedOracleKeys: readonly { keyId: string; publicKey: string }[];
      approvedAssets: readonly { chainId: string; assetId: string; tokenDecimals: number }[];
      maxQuoteAgeMs: number;
    };
  } | undefined;
  stripeConnectPayout?: {
    secretKey: string;
    apiVersion: string;
  } | undefined;
  economicReceiptSigningKeyId?: string | undefined;
  economicReceiptSigningPrivateKey?: string | undefined;
}

export function isCoordinatorLoopbackHost(host: string): boolean {
  const normalized = host.trim().toLowerCase().replace(/^\[(.*)\]$/, "$1");
  if (
    normalized === "localhost"
    || normalized === "::1"
    || normalized === "0:0:0:0:0:0:0:1"
  ) return true;
  if (isIP(normalized) === 4) return normalized.split(".", 1)[0] === "127";
  if (isIP(normalized) === 6 && normalized.startsWith("::ffff:")) {
    const mapped = normalized.slice("::ffff:".length);
    return isIP(mapped) === 4 && mapped.split(".", 1)[0] === "127";
  }
  return false;
}

export function assertCoordinatorNetworkSecurity(
  config: Pick<CoordinatorConfig, "host" | "networkToken">,
): void {
  if (!config.host.trim()) throw new Error("GPU_MESH_HOST must not be empty.");
  if (!isCoordinatorLoopbackHost(config.host) && !config.networkToken?.trim()) {
    throw new Error(
      "MYCELLIOS_NETWORK_TOKEN is required when GPU_MESH_HOST is not loopback.",
    );
  }
}

export function loadCoordinatorConfig(
  environment: NodeJS.ProcessEnv = process.env,
): CoordinatorConfig {
  const networkToken = loadSecret(
    environment.MYCELLIOS_NETWORK_TOKEN,
    environment.MYCELLIOS_NETWORK_TOKEN_FILE,
  );
  const mobileJoinToken = loadSecret(
    environment.MYCELLIOS_MOBILE_JOIN_TOKEN,
    environment.MYCELLIOS_MOBILE_JOIN_TOKEN_FILE,
  );
  const internalToken = loadSecret(
    environment.MYCELLIOS_INTERNAL_TOKEN,
    environment.MYCELLIOS_INTERNAL_TOKEN_FILE,
  );
  const modelAdminToken = loadSecret(
    environment.MYCELLIOS_MODEL_ADMIN_TOKEN,
    environment.MYCELLIOS_MODEL_ADMIN_TOKEN_FILE,
  );
  const mobileAssetsPath = environment.MYCELLIOS_MOBILE_DIST?.trim();
  const landingAssetsPath = environment.MYCELLIOS_LANDING_DIST?.trim();
  const nodeUpdatesPath = environment.MYCELLIOS_NODE_UPDATES_DIST?.trim();
  const componentUpdatesPath = environment.MYCELLIOS_COMPONENT_UPDATES_DIST?.trim();
  const componentUpdateDevKeys = loadComponentUpdatePublicKeys(
    "dev",
    environment.MYCELLIOS_COMPONENT_UPDATE_DEV_KEY_ID,
    environment.MYCELLIOS_COMPONENT_UPDATE_DEV_PUBLIC_KEY,
    environment.MYCELLIOS_COMPONENT_UPDATE_DEV_PUBLIC_KEY_FILE,
    environment.MYCELLIOS_COMPONENT_UPDATE_DEV_KEYRING_FILE,
  );
  const componentUpdateStableKeys = loadComponentUpdatePublicKeys(
    "stable",
    environment.MYCELLIOS_COMPONENT_UPDATE_STABLE_KEY_ID,
    environment.MYCELLIOS_COMPONENT_UPDATE_STABLE_PUBLIC_KEY,
    environment.MYCELLIOS_COMPONENT_UPDATE_STABLE_PUBLIC_KEY_FILE,
    environment.MYCELLIOS_COMPONENT_UPDATE_STABLE_KEYRING_FILE,
  );
  const modelCertificationKeyId = environment.MYCELLIOS_MODEL_CERTIFICATION_KEY_ID?.trim();
  const modelCertificationSpki = loadSecret(environment.MYCELLIOS_MODEL_CERTIFICATION_PUBLIC_KEY,
    environment.MYCELLIOS_MODEL_CERTIFICATION_PUBLIC_KEY_FILE);
  if (Boolean(modelCertificationKeyId) !== Boolean(modelCertificationSpki)) {
    throw new Error("MYCELLIOS_MODEL_CERTIFICATION_KEY_ID and MYCELLIOS_MODEL_CERTIFICATION_PUBLIC_KEY(_FILE) must be configured together.");
  }
  const modelCertificationPinnedKeys = modelCertificationKeyId && modelCertificationSpki
    ? (() => {
        if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(modelCertificationKeyId)
          || Buffer.from(modelCertificationSpki, "base64url").toString("base64url") !== modelCertificationSpki
          || !isEd25519Spki(modelCertificationSpki)) throw new Error("Invalid model certification public key configuration.");
        return [{ keyId: modelCertificationKeyId, spki: modelCertificationSpki }];
      })() : [];
  const releaseDownloadsPath = environment.MYCELLIOS_DOWNLOADS_DIST?.trim();
  const mobileExpertArtifactsPath = environment.MYCELLIOS_MOBILE_EXPERTS?.trim();
  const contentHubApiUrl = environment.CONTENT_HUB_API_URL?.trim();
  const publicationWebhookSecret = loadSecret(
    environment.MYCELLIOS_PUBLICATION_WEBHOOK_SECRET,
    environment.MYCELLIOS_PUBLICATION_WEBHOOK_SECRET_FILE,
  ) ?? loadSecret(
    environment.PUBLICATION_WEBHOOK_SECRET,
    environment.PUBLICATION_WEBHOOK_SECRET_FILE,
  );
  const supabaseUrl = environment.MYCELLIOS_SUPABASE_URL?.trim();
  const publicSupabaseUrlValue = environment.MYCELLIOS_SUPABASE_PUBLIC_URL?.trim();
  let publicSupabaseUrl: string | undefined;
  if (publicSupabaseUrlValue) {
    let url: URL;
    try { url = new URL(publicSupabaseUrlValue); }
    catch { throw new Error("MYCELLIOS_SUPABASE_PUBLIC_URL must be an HTTPS origin (or a loopback HTTP origin in development)."); }
    if (
      (url.protocol !== "https:" && !(url.protocol === "http:" && isCoordinatorLoopbackHost(url.hostname)))
      || url.username || url.password || url.pathname !== "/" || url.search || url.hash
    ) throw new Error("MYCELLIOS_SUPABASE_PUBLIC_URL must be an HTTPS origin (or a loopback HTTP origin in development).");
    publicSupabaseUrl = url.origin;
  }
  const supabaseServiceRoleKey = loadSecret(
    environment.MYCELLIOS_SUPABASE_SERVICE_ROLE_KEY,
    environment.MYCELLIOS_SUPABASE_SERVICE_ROLE_KEY_FILE,
  );
  const supabaseAnonKey = loadSecret(
    environment.MYCELLIOS_SUPABASE_ANON_KEY,
    environment.MYCELLIOS_SUPABASE_ANON_KEY_FILE,
  );
  if ((supabaseUrl && !supabaseServiceRoleKey) || (!supabaseUrl && supabaseServiceRoleKey)) {
    throw new Error(
      "MYCELLIOS_SUPABASE_URL and MYCELLIOS_SUPABASE_SERVICE_ROLE_KEY(_FILE) must be configured together.",
    );
  }
  if (publicSupabaseUrl && (!supabaseUrl || !supabaseAnonKey)) {
    throw new Error("MYCELLIOS_SUPABASE_PUBLIC_URL requires the Supabase server URL and anon key.");
  }
  const apiAccessEnabled = environment.MYCELLIOS_API_ACCESS_ENABLED === undefined
    ? Boolean(supabaseUrl && supabaseServiceRoleKey)
    : parseBoolean(environment.MYCELLIOS_API_ACCESS_ENABLED);
  const publicApiBaseUrl = environment.MYCELLIOS_PUBLIC_API_BASE_URL?.trim();
  const stripeWebhookSecret = loadSecret(
    environment.MYCELLIOS_STRIPE_WEBHOOK_SECRET,
    environment.MYCELLIOS_STRIPE_WEBHOOK_SECRET_FILE,
  );
  const stripePreviousWebhookSecret = loadSecret(
    environment.MYCELLIOS_STRIPE_PREVIOUS_WEBHOOK_SECRET,
    environment.MYCELLIOS_STRIPE_PREVIOUS_WEBHOOK_SECRET_FILE,
  );
  const stripeWebhookSecrets = [stripeWebhookSecret, stripePreviousWebhookSecret]
    .filter((value): value is string => Boolean(value));
  const stripeLivemodeValue = environment.MYCELLIOS_STRIPE_WEBHOOK_LIVEMODE;
  if (stripeWebhookSecrets.length > 0 && stripeLivemodeValue === undefined) {
    throw new Error(
      "MYCELLIOS_STRIPE_WEBHOOK_LIVEMODE must be explicitly configured when Stripe webhooks are enabled.",
    );
  }
  const stripeSecretKey = loadSecret(
    environment.MYCELLIOS_STRIPE_SECRET_KEY,
    environment.MYCELLIOS_STRIPE_SECRET_KEY_FILE,
  );
  const stripeSubscriptionPriceId = environment.MYCELLIOS_STRIPE_GO_PRICE_ID?.trim();
  const billingIncludedTokensValue = environment.MYCELLIOS_GO_INCLUDED_TOKENS?.trim();
  const billingSuccessUrl = environment.MYCELLIOS_BILLING_SUCCESS_URL?.trim();
  const billingCancelUrl = environment.MYCELLIOS_BILLING_CANCEL_URL?.trim();
  const billingPortalReturnUrl = environment.MYCELLIOS_BILLING_PORTAL_RETURN_URL?.trim();
  const topUpPacksValue = loadSecret(
    environment.MYCELLIOS_BILLING_TOPUP_PACKS_JSON,
    environment.MYCELLIOS_BILLING_TOPUP_PACKS_FILE,
  );
  const checkoutParts = [
    stripeSecretKey,
    stripeSubscriptionPriceId,
    billingIncludedTokensValue,
    billingSuccessUrl,
    billingCancelUrl,
    billingPortalReturnUrl,
  ];
  if (checkoutParts.some(Boolean) && !checkoutParts.every(Boolean)) {
    throw new Error(
      "Stripe Checkout requires MYCELLIOS_STRIPE_SECRET_KEY, MYCELLIOS_STRIPE_GO_PRICE_ID, "
      + "MYCELLIOS_GO_INCLUDED_TOKENS and all MYCELLIOS_BILLING_*_URL values.",
    );
  }
  const stripeCheckout = stripeSecretKey
    ? {
        secretKey: stripeSecretKey,
        subscriptionPriceId: stripeSubscriptionPriceId!,
        includedTokens: parseInteger(billingIncludedTokensValue, 0),
        successUrl: billingSuccessUrl!,
        cancelUrl: billingCancelUrl!,
        portalReturnUrl: billingPortalReturnUrl!,
        topUpPacks: parseTopUpPacks(topUpPacksValue),
      }
    : undefined;
  const stablecoinWatcherValue = loadSecret(
    environment.MYCELLIOS_STABLECOIN_WATCHER_CONFIG_JSON,
    environment.MYCELLIOS_STABLECOIN_WATCHER_CONFIG_FILE,
  );
  const stablecoinWatcher = stablecoinWatcherValue
    ? parseStablecoinWatcherConfig(stablecoinWatcherValue)
    : undefined;
  const sellerPayoutValue = loadSecret(
    environment.MYCELLIOS_SELLER_PAYOUT_CONFIG_JSON,
    environment.MYCELLIOS_SELLER_PAYOUT_CONFIG_FILE,
  );
  const sellerPayout = sellerPayoutValue
    ? parseSellerPayoutConfig(sellerPayoutValue)
    : undefined;
  const stripeConnectSecretKey = loadSecret(
    environment.MYCELLIOS_STRIPE_CONNECT_SECRET_KEY,
    environment.MYCELLIOS_STRIPE_CONNECT_SECRET_KEY_FILE,
  );
  const stripeConnectApiVersion = environment.MYCELLIOS_STRIPE_CONNECT_API_VERSION?.trim();
  if (Boolean(stripeConnectSecretKey) !== Boolean(stripeConnectApiVersion)) {
    throw new Error(
      "Stripe Connect payout requires MYCELLIOS_STRIPE_CONNECT_SECRET_KEY(_FILE) "
      + "and MYCELLIOS_STRIPE_CONNECT_API_VERSION together.",
    );
  }
  if (stripeConnectSecretKey && !sellerPayout) {
    throw new Error("Stripe Connect payout requires MYCELLIOS_SELLER_PAYOUT_CONFIG_JSON or _FILE.");
  }
  const stripeConnectPayout = stripeConnectSecretKey
    ? { secretKey: stripeConnectSecretKey, apiVersion: stripeConnectApiVersion! }
    : undefined;
  const economicReceiptSigningKeyId = environment.MYCELLIOS_ECONOMIC_RECEIPT_SIGNING_KEY_ID?.trim();
  const economicReceiptSigningPrivateKey = loadSecret(
    environment.MYCELLIOS_ECONOMIC_RECEIPT_SIGNING_PRIVATE_KEY,
    environment.MYCELLIOS_ECONOMIC_RECEIPT_SIGNING_PRIVATE_KEY_FILE,
  );
  if (Boolean(economicReceiptSigningKeyId) !== Boolean(economicReceiptSigningPrivateKey)) {
    throw new Error("MYCELLIOS_ECONOMIC_RECEIPT_SIGNING_KEY_ID and MYCELLIOS_ECONOMIC_RECEIPT_SIGNING_PRIVATE_KEY(_FILE) must be configured together.");
  }
  const config: CoordinatorConfig = {
    host: environment.GPU_MESH_HOST?.trim() || "127.0.0.1",
    port: parseInteger(environment.GPU_MESH_PORT, 8_787),
    databasePath:
      environment.GPU_MESH_DB === ":memory:"
        ? ":memory:"
        : resolve(environment.GPU_MESH_DB ?? "./data/gpu-mesh.db"),
    requestTimeoutMs: parseInteger(environment.GPU_MESH_REQUEST_TIMEOUT_MS, 120_000),
    supabasePersistenceRequired: environment.MYCELLIOS_SUPABASE_PERSISTENCE_REQUIRED?.trim() === "1",
    allowDevelopmentAdapters: environment.MYCELLIOS_ALLOW_DEV_ADAPTERS?.trim() === "1",
    apiAccessEnabled,
    apiStarterTokens: parseNonNegativeInteger(environment.MYCELLIOS_API_STARTER_TOKENS, 25_000),
    apiRequestsPerMinute: parseInteger(environment.MYCELLIOS_API_REQUESTS_PER_MINUTE, 30),
    apiMaxConcurrent: parseInteger(environment.MYCELLIOS_API_MAX_CONCURRENT, 2),
    apiMaxActiveKeys: parseInteger(environment.MYCELLIOS_API_MAX_ACTIVE_KEYS, 10),
    ...(networkToken ? { networkToken } : {}),
    ...(internalToken ? { internalToken } : {}),
    ...(modelAdminToken ? { modelAdminToken } : {}),
    ...(mobileJoinToken ? { mobileJoinToken } : {}),
    ...(mobileAssetsPath ? { mobileAssetsPath: resolve(mobileAssetsPath) } : {}),
    ...(landingAssetsPath ? { landingAssetsPath: resolve(landingAssetsPath) } : {}),
    ...(nodeUpdatesPath ? { nodeUpdatesPath: resolve(nodeUpdatesPath) } : {}),
    ...(componentUpdatesPath
      ? { componentUpdatesPath: resolve(componentUpdatesPath) }
      : {}),
    ...(componentUpdateDevKeys.length > 0 || componentUpdateStableKeys.length > 0
      ? {
          componentUpdatePinnedKeys: {
            ...(componentUpdateDevKeys.length > 0
              ? { dev: componentUpdateDevKeys }
              : {}),
            ...(componentUpdateStableKeys.length > 0
              ? { stable: componentUpdateStableKeys }
              : {}),
          },
        }
      : {}),
    ...(modelCertificationPinnedKeys.length ? { modelCertificationPinnedKeys } : {}),
    ...(releaseDownloadsPath ? { releaseDownloadsPath: resolve(releaseDownloadsPath) } : {}),
    ...(mobileExpertArtifactsPath
      ? { mobileExpertArtifactsPath: resolve(mobileExpertArtifactsPath) }
      : {}),
    ...(contentHubApiUrl ? { contentHubApiUrl } : {}),
    ...(publicationWebhookSecret ? { publicationWebhookSecret } : {}),
    ...(supabaseUrl ? { supabaseUrl } : {}),
    ...(publicSupabaseUrl ? { publicSupabaseUrl } : {}),
    ...(supabaseServiceRoleKey ? { supabaseServiceRoleKey } : {}),
    ...(supabaseAnonKey ? { supabaseAnonKey } : {}),
    ...(publicApiBaseUrl ? { publicApiBaseUrl: publicApiBaseUrl.replace(/\/+$/, "") } : {}),
    ...(stripeWebhookSecrets.length > 0
      ? {
          stripeWebhookSecrets,
          stripeWebhookLivemode: parseBoolean(stripeLivemodeValue!),
        }
      : {}),
    ...(stripeCheckout ? { stripeCheckout } : {}),
    ...(stablecoinWatcher ? { stablecoinWatcher } : {}),
    ...(sellerPayout ? { sellerPayout } : {}),
    ...(stripeConnectPayout ? { stripeConnectPayout } : {}),
    ...(economicReceiptSigningKeyId && economicReceiptSigningPrivateKey
      ? { economicReceiptSigningKeyId, economicReceiptSigningPrivateKey }
      : {}),
  };
  assertCoordinatorNetworkSecurity(config);
  return config;
}

function loadSecret(value: string | undefined, file: string | undefined): string | undefined {
  const inline = value?.trim();
  if (inline) return inline;
  const path = file?.trim();
  if (!path) return undefined;
  const secret = readFileSync(resolve(path), "utf8").trim();
  if (!secret) throw new Error(`Secret file is empty: ${path}`);
  return secret;
}

function loadComponentUpdatePublicKeys(
  channel: "dev" | "stable",
  keyIdValue: string | undefined,
  publicKeyValue: string | undefined,
  publicKeyFile: string | undefined,
  keyringFile: string | undefined,
): readonly { keyId: string; spki: string }[] {
  const keyId = keyIdValue?.trim();
  const spki = loadSecret(publicKeyValue, publicKeyFile);
  const candidates: unknown[] = [];
  if (keyId || spki) candidates.push({ keyId, spki });
  if (keyringFile?.trim()) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(
        readFileSync(resolve(keyringFile.trim()), "utf8"),
      ) as unknown;
    } catch (error) {
      throw new Error(
        `Invalid ${channel} component update keyring file.`,
        { cause: error },
      );
    }
    if (!Array.isArray(parsed)) {
      throw new Error(
        `Invalid ${channel} component update keyring file.`,
      );
    }
    candidates.push(...parsed);
  }
  const keys = candidates.map((candidate) => {
    const record = candidate && typeof candidate === "object"
      ? candidate as Record<string, unknown>
      : {};
    if (
      Object.keys(record).sort().join(",") !== "keyId,spki"
      || typeof record.keyId !== "string"
      || typeof record.spki !== "string"
      || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(record.keyId)
      || !/^[A-Za-z0-9_-]+$/.test(record.spki)
      || Buffer.from(record.spki, "base64url").toString("base64url")
        !== record.spki
      || !isEd25519Spki(record.spki)
    ) {
      throw new Error(
        `Invalid ${channel} component update public key configuration.`,
      );
    }
    return { keyId: record.keyId, spki: record.spki };
  });
  const byId = new Map<string, string>();
  for (const key of keys) {
    const existing = byId.get(key.keyId);
    if (existing && existing !== key.spki) {
      throw new Error(
        `Conflicting ${channel} component update key ID: ${key.keyId}.`,
      );
    }
    byId.set(key.keyId, key.spki);
  }
  return [...byId].map(([resolvedKeyId, resolvedSpki]) => ({
    keyId: resolvedKeyId,
    spki: resolvedSpki,
  }));
}

function isEd25519Spki(value: string): boolean {
  try {
    const encoded = Buffer.from(value, "base64url");
    if (encoded.length > 128) return false;
    const key = createPublicKey({
      key: encoded,
      format: "der",
      type: "spki",
    });
    const canonical = key.export({ format: "der", type: "spki" });
    return key.asymmetricKeyType === "ed25519"
      && Buffer.isBuffer(canonical)
      && canonical.equals(encoded);
  } catch {
    return false;
  }
}

export function loadWorkerConfig(path: string): WorkerConfig {
  const parsed = JSON.parse(readFileSync(resolve(path), "utf8")) as unknown;
  return workerConfigSchema.parse(parsed);
}

function parseInteger(value: string | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  const parsed = Number.parseInt(value, 10);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`Invalid positive integer configuration value: ${value}`);
  }
  return parsed;
}

function parseNonNegativeInteger(value: string | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  const parsed = Number.parseInt(value, 10);
  if (!Number.isSafeInteger(parsed) || parsed < 0) {
    throw new Error(`Invalid non-negative integer configuration value: ${value}`);
  }
  return parsed;
}

function parseBoolean(value: string): boolean {
  const normalized = value.trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(normalized)) return true;
  if (["0", "false", "no", "off"].includes(normalized)) return false;
  throw new Error(`Invalid boolean configuration value: ${value}`);
}

function parseTopUpPacks(value: string | undefined): Array<{
  packId: string;
  amountMicros: number;
  currency: string;
  tokenAmount: number;
  stripePriceId: string;
}> {
  if (!value) return [];
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    throw new Error("MYCELLIOS_BILLING_TOPUP_PACKS_JSON must contain valid JSON.");
  }
  if (!Array.isArray(parsed)) {
    throw new Error("MYCELLIOS_BILLING_TOPUP_PACKS_JSON must contain an array.");
  }
  return parsed as Array<{
    packId: string; amountMicros: number; currency: string;
    tokenAmount: number; stripePriceId: string;
  }>;
}

function parseStablecoinWatcherConfig(value: string): {
  trustedWatcherKeys: Array<{ keyId: string; publicKey: string }>;
  chains: Array<{ chainId: string; asset: string; minimumConfirmations: number }>;
} {
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    throw new Error("MYCELLIOS_STABLECOIN_WATCHER_CONFIG_JSON must contain valid JSON.");
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("MYCELLIOS_STABLECOIN_WATCHER_CONFIG_JSON must contain an object.");
  }
  const record = parsed as Record<string, unknown>;
  if (!Array.isArray(record.trustedWatcherKeys) || !Array.isArray(record.chains)) {
    throw new Error("Stablecoin watcher config must contain trustedWatcherKeys and chains arrays.");
  }
  return {
    trustedWatcherKeys: record.trustedWatcherKeys as Array<{ keyId: string; publicKey: string }>,
    chains: record.chains as Array<{ chainId: string; asset: string; minimumConfirmations: number }>,
  };
}

function parseSellerPayoutConfig(value: string): NonNullable<CoordinatorConfig["sellerPayout"]> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    throw new Error("MYCELLIOS_SELLER_PAYOUT_CONFIG_JSON must contain valid JSON.");
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("MYCELLIOS_SELLER_PAYOUT_CONFIG_JSON must contain an object.");
  }
  const record = parsed as Record<string, unknown>;
  const spore = record.spore;
  if (!Number.isSafeInteger(record.minimumUsdMicros) || Number(record.minimumUsdMicros) <= 0) {
    throw new Error("Seller payout config requires a positive minimumUsdMicros safe integer.");
  }
  if (typeof spore !== "object" || spore === null || Array.isArray(spore)) {
    throw new Error("Seller payout config requires a complete spore gate object.");
  }
  const gates = spore as Record<string, unknown>;
  const gateNames = ["legalApproved", "custodyApproved", "liquidityApproved", "antifraudApproved"] as const;
  if (gateNames.some((name) => typeof gates[name] !== "boolean")) {
    throw new Error("Seller payout config requires all SPORE gates as booleans.");
  }
  if (!Array.isArray(record.destinationVerifierKeys) || record.destinationVerifierKeys.length === 0) {
    throw new Error("Seller payout config requires at least one destination verifier key.");
  }
  if (!Array.isArray(record.settlementVerifierKeys) || record.settlementVerifierKeys.length === 0) {
    throw new Error("Seller payout config requires at least one settlement verifier key.");
  }
  if (!Number.isSafeInteger(record.settlementEvidenceMaxAgeMs)
    || Number(record.settlementEvidenceMaxAgeMs) < 60_000
    || Number(record.settlementEvidenceMaxAgeMs) > 86_400_000) {
    throw new Error("Seller payout config requires settlementEvidenceMaxAgeMs between 60000 and 86400000.");
  }
  const parseVerifierKeys = (entries: unknown[], purpose: string) => entries.map((entry) => {
    if (typeof entry !== "object" || entry === null || Array.isArray(entry)) {
      throw new Error(`Seller payout ${purpose} verifier entries must be objects.`);
    }
    const key = entry as Record<string, unknown>;
    if (typeof key.keyId !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$/.test(key.keyId)
      || typeof key.publicKey !== "string" || key.publicKey.trim().length < 32) {
      throw new Error(`Seller payout ${purpose} verifier keys are invalid.`);
    }
    try {
      if (createPublicKey(key.publicKey).asymmetricKeyType !== "ed25519") {
        throw new Error("not_ed25519");
      }
    } catch {
      throw new Error(`Seller payout ${purpose} verifier keys must be valid Ed25519 public keys.`);
    }
    return { keyId: key.keyId, publicKey: key.publicKey };
  });
  const destinationVerifierKeys = parseVerifierKeys(record.destinationVerifierKeys, "destination");
  const settlementVerifierKeys = parseVerifierKeys(record.settlementVerifierKeys, "settlement");
  if (new Set(destinationVerifierKeys.map((entry) => entry.keyId)).size !== destinationVerifierKeys.length
    || new Set(settlementVerifierKeys.map((entry) => entry.keyId)).size !== settlementVerifierKeys.length) {
    throw new Error("Seller payout verifier key ids must be unique per purpose.");
  }
  let sporeConversion: NonNullable<CoordinatorConfig["sellerPayout"]>["sporeConversion"];
  if (record.sporeConversion !== undefined) {
    if (typeof record.sporeConversion !== "object" || record.sporeConversion === null
      || Array.isArray(record.sporeConversion)) {
      throw new Error("Seller payout sporeConversion must be an object.");
    }
    const conversion = record.sporeConversion as Record<string, unknown>;
    if (!Array.isArray(conversion.trustedOracleKeys) || conversion.trustedOracleKeys.length === 0
      || !Array.isArray(conversion.approvedAssets) || conversion.approvedAssets.length === 0
      || !Number.isSafeInteger(conversion.maxQuoteAgeMs) || Number(conversion.maxQuoteAgeMs) < 10_000
      || Number(conversion.maxQuoteAgeMs) > 3_600_000) {
      throw new Error("Seller payout sporeConversion requires oracle keys, assets and maxQuoteAgeMs 10000..3600000.");
    }
    const trustedOracleKeys = parseVerifierKeys(conversion.trustedOracleKeys, "SPORE oracle");
    const approvedAssets = conversion.approvedAssets.map((entry) => {
      if (typeof entry !== "object" || entry === null || Array.isArray(entry)) {
        throw new Error("Seller payout SPORE assets must be objects.");
      }
      const asset = entry as Record<string, unknown>;
      if (typeof asset.chainId !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$/.test(asset.chainId)
        || typeof asset.assetId !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$/.test(asset.assetId)
        || !Number.isSafeInteger(asset.tokenDecimals) || Number(asset.tokenDecimals) < 0
        || Number(asset.tokenDecimals) > 30) {
        throw new Error("Seller payout SPORE assets are invalid.");
      }
      return { chainId: asset.chainId, assetId: asset.assetId, tokenDecimals: Number(asset.tokenDecimals) };
    });
    if (new Set(approvedAssets.map((asset) => `${asset.chainId}:${asset.assetId}`)).size !== approvedAssets.length) {
      throw new Error("Seller payout SPORE assets must be unique.");
    }
    sporeConversion = { trustedOracleKeys, approvedAssets, maxQuoteAgeMs: Number(conversion.maxQuoteAgeMs) };
  }
  if (gateNames.every((name) => gates[name] === true) && !sporeConversion) {
    throw new Error("Opening all SPORE gates requires a complete sporeConversion policy.");
  }
  return {
    minimumUsdMicros: Number(record.minimumUsdMicros),
    spore: {
      legalApproved: gates.legalApproved as boolean,
      custodyApproved: gates.custodyApproved as boolean,
      liquidityApproved: gates.liquidityApproved as boolean,
      antifraudApproved: gates.antifraudApproved as boolean,
    },
    destinationVerifierKeys,
    settlementVerifierKeys,
    settlementEvidenceMaxAgeMs: Number(record.settlementEvidenceMaxAgeMs),
    ...(sporeConversion ? { sporeConversion } : {}),
  };
}
