const SESSION_KEY = "mycellios.auth.session";

export interface PublicAuthConfig {
  enabled: boolean;
  url?: string;
  anonKey?: string;
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

async function authRequest(
  config: PublicAuthConfig,
  path: string,
  body: Record<string, string>,
): Promise<AuthSession> {
  if (!config.enabled || !config.url || !config.anonKey) {
    throw new Error("Mycellios accounts are not available yet.");
  }
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
