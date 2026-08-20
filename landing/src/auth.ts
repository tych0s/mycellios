const SESSION_KEY = "mycellios.auth.session";
const SESSION_REFRESH_MARGIN_MS = 90_000;
let refreshInFlight: Promise<AuthSession> | null = null;

export interface PublicAuthConfig {
  enabled: boolean;
  url?: string;
  anonKey?: string;
  providers?: {
    email: boolean;
    google: boolean;
    twitter: boolean;
  };
  apiAccessEnabled?: boolean;
  publicApiBaseUrl?: string;
  starterTokens?: number;
  limits?: {
    requestsPerMinute: number;
    maxConcurrent: number;
    maxActiveKeys: number;
  };
}

export interface AuthSession {
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
  user: { id: string; email: string | null };
}

export interface NetworkIdentity {
  id: string;
  email: string | null;
  role: "owner" | "admin" | "operator" | "viewer" | null;
}

export type OAuthProvider = "google" | "twitter";

export async function loadAuthConfig(): Promise<PublicAuthConfig> {
  const response = await fetch("/public/v1/auth-config", { cache: "no-store" });
  if (!response.ok) return { enabled: false };
  const config = await response.json() as PublicAuthConfig;
  if (!config.enabled || !config.url || !config.anonKey) return config;
  try {
    const settingsUrl = new URL("/auth/v1/settings", config.url);
    const settingsResponse = await fetch(settingsUrl, {
      cache: "no-store",
      headers: {
        apikey: config.anonKey,
        authorization: `Bearer ${config.anonKey}`,
      },
    });
    if (!settingsResponse.ok) return config;
    const settings = await settingsResponse.json() as {
      external?: Partial<Record<"email" | OAuthProvider, boolean>>;
    };
    return {
      ...config,
      providers: {
        email: settings.external?.email !== false,
        google: settings.external?.google === true,
        twitter: settings.external?.twitter === true,
      },
    };
  } catch {
    return config;
  }
}

export function storedAuthSession(): AuthSession | null {
  try {
    const value = JSON.parse(window.localStorage.getItem(SESSION_KEY) ?? "null") as unknown;
    return isAuthSession(value) ? value : null;
  } catch {
    return null;
  }
}

export async function restoreAuthSession(config: PublicAuthConfig): Promise<AuthSession | null> {
  const redirected = await consumeOAuthRedirect(config);
  if (redirected) return redirected;
  const current = storedAuthSession();
  if (!current || !config.enabled || !config.url || !config.anonKey) return null;
  if (current.expiresAt > Date.now() + SESSION_REFRESH_MARGIN_MS) return current;
  try {
    return await refreshAuthSession(config, current);
  } catch {
    clearAuthSession();
    return null;
  }
}

export function authSessionNeedsRefresh(session: AuthSession, now = Date.now()): boolean {
  return session.expiresAt <= now + SESSION_REFRESH_MARGIN_MS;
}

export async function refreshAuthSession(
  config: PublicAuthConfig,
  session: AuthSession,
  force = false,
): Promise<AuthSession> {
  const stored = storedAuthSession();
  const freshest = stored?.user.id === session.user.id && stored.expiresAt > session.expiresAt
    ? stored
    : session;
  if (!force && !authSessionNeedsRefresh(freshest)) return freshest;
  if (!refreshInFlight) {
    refreshInFlight = authRequest(config, "/auth/v1/token?grant_type=refresh_token", {
      refresh_token: freshest.refreshToken,
    }).finally(() => { refreshInFlight = null; });
  }
  return refreshInFlight;
}

export async function validAuthSession(
  config: PublicAuthConfig,
  session: AuthSession,
  forceRefresh = false,
): Promise<AuthSession> {
  const stored = storedAuthSession();
  const freshest = stored?.user.id === session.user.id && stored.expiresAt > session.expiresAt
    ? stored
    : session;
  return forceRefresh || authSessionNeedsRefresh(freshest)
    ? refreshAuthSession(config, freshest, forceRefresh)
    : freshest;
}

export function subscribeAuthSession(listener: (session: AuthSession | null) => void): () => void {
  const onStorage = (event: StorageEvent) => {
    if (event.key === SESSION_KEY) listener(storedAuthSession());
  };
  window.addEventListener("storage", onStorage);
  return () => window.removeEventListener("storage", onStorage);
}

export function signIn(
  config: PublicAuthConfig,
  email: string,
  password: string,
): Promise<AuthSession> {
  return authRequest(config, "/auth/v1/token?grant_type=password", { email, password });
}

export function signUp(
  config: PublicAuthConfig,
  email: string,
  password: string,
): Promise<AuthSession> {
  return authRequest(config, "/auth/v1/signup", { email, password });
}

export function signInWithOAuth(config: PublicAuthConfig, provider: OAuthProvider): void {
  assertAuthConfig(config);
  assertOAuthProviderEnabled(config, provider);
  const authorizeUrl = new URL("/auth/v1/authorize", config.url);
  authorizeUrl.searchParams.set("provider", provider);
  authorizeUrl.searchParams.set("redirect_to", `${window.location.origin}${window.location.pathname}${window.location.search}`);
  window.location.assign(authorizeUrl);
}

export function signInWithGoogle(config: PublicAuthConfig): void {
  signInWithOAuth(config, "google");
}

export function signInWithX(config: PublicAuthConfig): void {
  signInWithOAuth(config, "twitter");
}

export async function linkOAuthIdentity(
  config: PublicAuthConfig,
  session: Pick<AuthSession, "accessToken">,
  provider: OAuthProvider,
): Promise<void> {
  assertAuthConfig(config);
  assertOAuthProviderEnabled(config, provider);
  const authorizeUrl = new URL("/auth/v1/user/identities/authorize", config.url);
  authorizeUrl.searchParams.set("provider", provider);
  authorizeUrl.searchParams.set("redirect_to", `${window.location.origin}${window.location.pathname}${window.location.search}`);
  authorizeUrl.searchParams.set("skip_http_redirect", "true");
  const response = await fetch(authorizeUrl, {
    cache: "no-store",
    headers: {
      apikey: config.anonKey,
      authorization: `Bearer ${session.accessToken}`,
    },
  });
  const payload = await response.json().catch(() => null) as { url?: unknown; message?: unknown; error_description?: unknown } | null;
  if (!response.ok || typeof payload?.url !== "string") {
    const message = payload?.message ?? payload?.error_description;
    throw new Error(typeof message === "string" ? message : `Could not link ${provider} (HTTP ${response.status}).`);
  }
  window.location.assign(payload.url);
}

export async function signInWithMetaMask(config: PublicAuthConfig): Promise<AuthSession> {
  assertAuthConfig(config);
  const ethereum = (window as Window & {
    ethereum?: { request: (request: { method: string; params?: unknown[] }) => Promise<unknown> };
  }).ethereum;
  if (!ethereum) throw new Error("MetaMask is not installed in this browser.");

  const accounts = await ethereum.request({ method: "eth_requestAccounts" });
  const address = Array.isArray(accounts) && typeof accounts[0] === "string" ? accounts[0] : null;
  if (!address) throw new Error("MetaMask did not provide an Ethereum account.");
  const rawChainId = await ethereum.request({ method: "eth_chainId" });
  const chainId = typeof rawChainId === "string" ? Number.parseInt(rawChainId, 16) : Number.NaN;
  if (!Number.isSafeInteger(chainId)) throw new Error("MetaMask returned an invalid chain ID.");

  const uri = `${window.location.origin}${window.location.pathname}`;
  const nonce = [...crypto.getRandomValues(new Uint8Array(8))]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
  const message = `${window.location.host} wants you to sign in with your Ethereum account:\n${address}\n\nSign in to Mycellios.\n\nURI: ${uri}\nVersion: 1\nChain ID: ${chainId}\nNonce: ${nonce}\nIssued At: ${new Date().toISOString()}`;
  const encodedMessage = `0x${[...new TextEncoder().encode(message)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("")}`;
  const signature = await ethereum.request({
    method: "personal_sign",
    params: [encodedMessage, address],
  });
  if (typeof signature !== "string") throw new Error("MetaMask did not return a valid signature.");
  return authRequest(config, "/auth/v1/token?grant_type=web3", {
    chain: "ethereum",
    message,
    signature,
  });
}

export async function loadNetworkIdentity(accessToken: string): Promise<NetworkIdentity> {
  const response = await fetch("/v1/auth/me", {
    cache: "no-store",
    headers: { authorization: `Bearer ${accessToken}` },
  });
  if (!response.ok) throw new Error("Your Mycellios session is no longer valid.");
  const body = await response.json() as { user: NetworkIdentity };
  return body.user;
}

export async function signOut(config: PublicAuthConfig, session: AuthSession): Promise<void> {
  clearAuthSession();
  if (!config.url || !config.anonKey) return;
  await fetch(new URL("/auth/v1/logout", config.url), {
    method: "POST",
    headers: {
      apikey: config.anonKey,
      authorization: `Bearer ${session.accessToken}`,
    },
  }).catch(() => undefined);
}

export function sessionHasRecentAal2(session: Pick<AuthSession, "accessToken">, now = Date.now()): boolean {
  const [, payload] = session.accessToken.split(".");
  if (!payload) return false;
  try {
    const claims = JSON.parse(atob(payload.replaceAll("-", "+").replaceAll("_", "/"))) as {
      aal?: unknown; auth_time?: unknown; amr?: unknown;
    };
    const amrTimes = Array.isArray(claims.amr) ? claims.amr.flatMap((entry) => {
      if (!entry || typeof entry !== "object") return [];
      const record = entry as Record<string, unknown>;
      return typeof record.timestamp === "number" && record.method !== "token_refresh" ? [record.timestamp] : [];
    }) : [];
    const authenticationTimes = [...(typeof claims.auth_time === "number" ? [claims.auth_time] : []), ...amrTimes];
    const authenticationTime = authenticationTimes.length > 0 ? Math.max(...authenticationTimes) : null;
    return claims.aal === "aal2" && typeof authenticationTime === "number"
      && authenticationTime * 1_000 <= now + 30_000 && authenticationTime * 1_000 >= now - 5 * 60_000;
  } catch { return false; }
}

export async function verifyTotpStepUp(
  config: PublicAuthConfig,
  session: AuthSession,
  code: string,
): Promise<AuthSession> {
  assertAuthConfig(config);
  const headers = { apikey: config.anonKey, authorization: `Bearer ${session.accessToken}`, "content-type": "application/json" };
  const factorsResponse = await fetch(new URL("/auth/v1/factors", config.url), { headers, cache: "no-store" });
  const factorsPayload = await factorsResponse.json().catch(() => null) as {
    totp?: Array<{ id?: unknown; status?: unknown }>;
    all?: Array<{ id?: unknown; factor_type?: unknown; status?: unknown }>;
    message?: unknown;
  } | null;
  if (!factorsResponse.ok) throw new Error(authPayloadMessage(factorsPayload, "Could not load MFA factors."));
  const factor = factorsPayload?.totp?.find((candidate) => candidate.status === "verified")
    ?? factorsPayload?.all?.find((candidate) => candidate.factor_type === "totp" && candidate.status === "verified");
  if (typeof factor?.id !== "string") throw new Error("No verified TOTP factor is enrolled for this account.");
  const challengeResponse = await fetch(new URL(`/auth/v1/factors/${encodeURIComponent(factor.id)}/challenge`, config.url), {
    method: "POST", headers, body: "{}",
  });
  const challenge = await challengeResponse.json().catch(() => null) as { id?: unknown; message?: unknown } | null;
  if (!challengeResponse.ok || typeof challenge?.id !== "string") {
    throw new Error(authPayloadMessage(challenge, "Could not start MFA verification."));
  }
  const verifyResponse = await fetch(new URL(`/auth/v1/factors/${encodeURIComponent(factor.id)}/verify`, config.url), {
    method: "POST", headers, body: JSON.stringify({ challenge_id: challenge.id, code: code.trim() }),
  });
  const verified = await verifyResponse.json().catch(() => null) as {
    access_token?: unknown; refresh_token?: unknown; expires_in?: unknown; message?: unknown;
  } | null;
  if (!verifyResponse.ok || typeof verified?.access_token !== "string") {
    throw new Error(authPayloadMessage(verified, "MFA verification failed."));
  }
  const elevated: AuthSession = {
    ...session,
    accessToken: verified.access_token,
    refreshToken: typeof verified.refresh_token === "string" ? verified.refresh_token : session.refreshToken,
    expiresAt: Date.now() + (typeof verified.expires_in === "number" ? verified.expires_in : 3600) * 1_000,
  };
  window.localStorage.setItem(SESSION_KEY, JSON.stringify(elevated));
  return elevated;
}

function authPayloadMessage(payload: { message?: unknown } | null, fallback: string): string {
  return typeof payload?.message === "string" ? payload.message : fallback;
}

function clearAuthSession(): void {
  window.localStorage.removeItem(SESSION_KEY);
}

function assertAuthConfig(config: PublicAuthConfig): asserts config is PublicAuthConfig & { url: string; anonKey: string } {
  if (!config.enabled || !config.url || !config.anonKey) {
    throw new Error("Mycellios accounts are not available yet.");
  }
}

function assertOAuthProviderEnabled(config: PublicAuthConfig, provider: OAuthProvider): void {
  if (config.providers && !config.providers[provider]) {
    throw new Error(`${provider === "twitter" ? "X" : "Google"} sign-in is not configured yet.`);
  }
}

async function consumeOAuthRedirect(config: PublicAuthConfig): Promise<AuthSession | null> {
  if (!window.location.hash.includes("access_token=") && !window.location.hash.includes("error=")) return null;
  const params = new URLSearchParams(window.location.hash.slice(1));
  const oauthError = params.get("error_description") ?? params.get("error");
  window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
  if (oauthError) throw new Error(oauthError);
  assertAuthConfig(config);
  const accessToken = params.get("access_token");
  const refreshToken = params.get("refresh_token");
  if (!accessToken || !refreshToken) throw new Error("The identity provider did not return a valid Mycellios session.");
  const response = await fetch(new URL("/auth/v1/user", config.url), {
    headers: { apikey: config.anonKey, authorization: `Bearer ${accessToken}` },
  });
  const user = await response.json().catch(() => null) as { id?: unknown; email?: unknown } | null;
  if (!response.ok || typeof user?.id !== "string") throw new Error("The identity provider returned an invalid Mycellios identity.");
  const session: AuthSession = {
    accessToken,
    refreshToken,
    expiresAt: Date.now() + Number(params.get("expires_in") ?? 3600) * 1_000,
    user: { id: user.id, email: typeof user.email === "string" ? user.email : null },
  };
  window.localStorage.setItem(SESSION_KEY, JSON.stringify(session));
  return session;
}

async function authRequest(
  config: PublicAuthConfig,
  path: string,
  body: Record<string, string>,
): Promise<AuthSession> {
  assertAuthConfig(config);
  const response = await fetch(new URL(path, config.url), {
    method: "POST",
    headers: { apikey: config.anonKey, "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => null) as {
    access_token?: unknown;
    refresh_token?: unknown;
    expires_in?: unknown;
    user?: { id?: unknown; email?: unknown };
    msg?: unknown;
    message?: unknown;
    error_description?: unknown;
  } | null;
  if (!response.ok) {
    const message = payload?.msg ?? payload?.message ?? payload?.error_description;
    throw new Error(typeof message === "string" ? message : `Authentication failed (HTTP ${response.status}).`);
  }
  if (
    typeof payload?.access_token !== "string"
    || typeof payload.refresh_token !== "string"
    || typeof payload.user?.id !== "string"
  ) {
    throw new Error("Check your email to confirm the account, then sign in.");
  }
  const session: AuthSession = {
    accessToken: payload.access_token,
    refreshToken: payload.refresh_token,
    expiresAt: Date.now() + (
      typeof payload.expires_in === "number" ? payload.expires_in : 3600
    ) * 1_000,
    user: {
      id: payload.user.id,
      email: typeof payload.user.email === "string" ? payload.user.email : null,
    },
  };
  window.localStorage.setItem(SESSION_KEY, JSON.stringify(session));
  return session;
}

function isAuthSession(value: unknown): value is AuthSession {
  if (!value || typeof value !== "object") return false;
  const session = value as Partial<AuthSession>;
  return typeof session.accessToken === "string"
    && typeof session.refreshToken === "string"
    && typeof session.expiresAt === "number"
    && typeof session.user?.id === "string";
}
