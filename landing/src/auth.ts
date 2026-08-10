const SESSION_KEY = "mycellios.auth.session";

export interface PublicAuthConfig {
  enabled: boolean;
  url?: string;
  anonKey?: string;
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

export async function loadAuthConfig(): Promise<PublicAuthConfig> {
  const response = await fetch("/public/v1/auth-config", { cache: "no-store" });
  if (!response.ok) return { enabled: false };
  return response.json() as Promise<PublicAuthConfig>;
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
  if (current.expiresAt > Date.now() + 60_000) return current;
  try {
    return await authRequest(config, "/auth/v1/token?grant_type=refresh_token", {
      refresh_token: current.refreshToken,
    });
  } catch {
    clearAuthSession();
    return null;
  }
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

export function signInWithX(config: PublicAuthConfig): void {
  assertAuthConfig(config);
  const authorizeUrl = new URL("/auth/v1/authorize", config.url);
  authorizeUrl.searchParams.set("provider", "x");
  authorizeUrl.searchParams.set("redirect_to", `${window.location.origin}${window.location.pathname}${window.location.search}`);
  window.location.assign(authorizeUrl);
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

function clearAuthSession(): void {
  window.localStorage.removeItem(SESSION_KEY);
}

function assertAuthConfig(config: PublicAuthConfig): asserts config is PublicAuthConfig & { url: string; anonKey: string } {
  if (!config.enabled || !config.url || !config.anonKey) {
    throw new Error("Mycellios accounts are not available yet.");
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
  if (!accessToken || !refreshToken) throw new Error("X did not return a valid Mycellios session.");
  const response = await fetch(new URL("/auth/v1/user", config.url), {
    headers: { apikey: config.anonKey, authorization: `Bearer ${accessToken}` },
  });
  const user = await response.json().catch(() => null) as { id?: unknown; email?: unknown } | null;
  if (!response.ok || typeof user?.id !== "string") throw new Error("X returned an invalid Mycellios identity.");
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
