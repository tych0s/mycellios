import { afterEach, describe, expect, it, vi } from "vitest";
import {
  linkOAuthIdentity,
  loadAuthConfig,
  loadNetworkIdentity,
  authSessionNeedsRefresh,
  validAuthSession,
  restoreAuthSession,
  sessionSurvivesRefreshFailure,
  signInWithGoogle,
  signInWithX,
  signIn,
  signOut,
  signUp,
  EmailConfirmationRequired,
  AuthServiceError,
  type PublicAuthConfig,
} from "./auth";

const config: PublicAuthConfig = { enabled: true, url: "https://auth.example", anonKey: "anon" };

function installWindow(hash = "") {
  const assign = vi.fn();
  const setItem = vi.fn();
  const removeItem = vi.fn();
  const replaceState = vi.fn();
  const sessionValues = new Map<string, string>();
  const sessionStorage = {
    getItem: vi.fn((key: string) => sessionValues.get(key) ?? null),
    setItem: vi.fn((key: string, value: string) => { sessionValues.set(key, value); }),
    removeItem: vi.fn((key: string) => { sessionValues.delete(key); }),
  };
  vi.stubGlobal("window", {
    location: {
      origin: "https://app.example",
      pathname: "/dashboard",
      search: "?view=overview",
      hash,
      host: "app.example",
      assign,
    },
    history: { replaceState },
    localStorage: { getItem: vi.fn(() => null), setItem, removeItem },
    sessionStorage,
  });
  return { assign, setItem, removeItem, replaceState, sessionStorage };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("session renewal", () => {
  it("keeps a session that is safely inside its lifetime", async () => {
    installWindow();
    const session = { accessToken: "access", refreshToken: "refresh", expiresAt: Date.now() + 10 * 60_000, user: { id: "u1", email: null } };
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    expect(authSessionNeedsRefresh(session)).toBe(false);
    await expect(validAuthSession(config, session)).resolves.toBe(session);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("renews and persists a session near expiry", async () => {
    const { setItem } = installWindow();
    const session = { accessToken: "old", refreshToken: "refresh", expiresAt: Date.now() + 10_000, user: { id: "u1", email: null } };
    window.localStorage.getItem = vi.fn(() => JSON.stringify(session));
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({ access_token: "new", refresh_token: "next", expires_in: 3600, user: { id: "u1", email: null } })));
    const renewed = await validAuthSession(config, session);
    expect(renewed.accessToken).toBe("new");
    expect(setItem).toHaveBeenCalledWith("mycellios.auth.session", expect.stringContaining('"accessToken":"new"'));
  });

  it("deduplicates simultaneous forced renewals", async () => {
    installWindow();
    const session = { accessToken: "old", refreshToken: "refresh", expiresAt: Date.now() + 60_000, user: { id: "u1", email: null } };
    window.localStorage.getItem = vi.fn(() => JSON.stringify(session));
    const fetchMock = vi.fn(async () => Response.json({ access_token: "new", refresh_token: "next", expires_in: 3600, user: { id: "u1", email: null } }));
    vi.stubGlobal("fetch", fetchMock);
    const [first, second] = await Promise.all([validAuthSession(config, session, true), validAuthSession(config, session, true)]);
    expect(first.accessToken).toBe("new");
    expect(second.accessToken).toBe("new");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("does not share refresh responses between different accounts", async () => {
    installWindow();
    const first = { accessToken: "a", refreshToken: "refresh-a", expiresAt: Date.now() + 10_000, user: { id: "u1", email: null } };
    const second = { accessToken: "b", refreshToken: "refresh-b", expiresAt: Date.now() + 10_000, user: { id: "u2", email: null } };
    window.localStorage.getItem = vi.fn(() => JSON.stringify(second));
    const fetchMock = vi.fn(async (_url: URL, init: RequestInit) => {
      const token = JSON.parse(String(init.body)).refresh_token as string;
      return Response.json({ access_token: `new-${token}`, refresh_token: `next-${token}`, expires_in: 3600, user: { id: token === "refresh-a" ? "u1" : "u2", email: null } });
    });
    vi.stubGlobal("fetch", fetchMock);
    const [firstResult, secondResult] = await Promise.all([validAuthSession(config, first), validAuthSession(config, second)]);
    expect(firstResult.accessToken).toBe("new-refresh-a");
    expect(secondResult.accessToken).toBe("new-refresh-b");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("uses a newer session already rotated by another tab when it is still valid", async () => {
    const current = { accessToken: "old", refreshToken: "old-refresh", expiresAt: Date.now() + 10_000, user: { id: "u1", email: null } };
    const newer = { ...current, accessToken: "newer", refreshToken: "newer-refresh", expiresAt: Date.now() + 10 * 60_000 };
    installWindow();
    window.localStorage.getItem = vi.fn(() => JSON.stringify(newer));
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    await expect(validAuthSession(config, current)).resolves.toStrictEqual(newer);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("really rotates a rejected token even when its local expiry is in the future", async () => {
    installWindow();
    const session = { accessToken: "rejected", refreshToken: "refresh", expiresAt: Date.now() + 10 * 60_000, user: { id: "u1", email: null } };
    window.localStorage.getItem = vi.fn(() => JSON.stringify(session));
    const fetchMock = vi.fn(async () => Response.json({ access_token: "accepted", refresh_token: "next", expires_in: 3600, user: { id: "u1", email: null } }));
    vi.stubGlobal("fetch", fetchMock);
    const renewed = await validAuthSession(config, session, true);
    expect(renewed.accessToken).toBe("accepted");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("does not restore a signed-out session when an older refresh finishes", async () => {
    const { setItem, removeItem } = installWindow();
    const session = { accessToken: "old", refreshToken: "refresh", expiresAt: Date.now() + 10_000, user: { id: "u1", email: null } };
    let stored: string | null = JSON.stringify(session);
    window.localStorage.getItem = vi.fn(() => stored);
    setItem.mockImplementation((_key: string, value: string) => { stored = value; });
    removeItem.mockImplementation(() => { stored = null; });
    let finishRefresh!: (response: Response) => void;
    const fetchMock = vi.fn().mockImplementationOnce(() => new Promise<Response>((resolve) => { finishRefresh = resolve; }))
      .mockResolvedValueOnce(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);
    const renewal = validAuthSession(config, session);
    await signOut(config, session);
    finishRefresh(Response.json({ access_token: "old-returned", refresh_token: "rotated", expires_in: 3600, user: { id: "u1", email: null } }));
    await renewal;
    expect(stored).toBeNull();
    expect(setItem).not.toHaveBeenCalled();
  });
});

describe("Supabase OAuth providers", () => {
  it("can restore the base configuration without waiting for provider discovery", async () => {
    const fetchMock = vi.fn().mockResolvedValue(Response.json(config));
    vi.stubGlobal("fetch", fetchMock);
    await expect(loadAuthConfig(false)).resolves.toEqual(config);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("loads live provider availability from Supabase settings", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(Response.json(config))
      .mockResolvedValueOnce(Response.json({ external: { email: true, google: true, twitter: false } }));
    vi.stubGlobal("fetch", fetchMock);
    const loaded = await loadAuthConfig();
    expect(loaded.providers).toEqual({ email: true, google: true, twitter: false, web3: false });
    expect(fetchMock).toHaveBeenNthCalledWith(2, new URL("https://auth.example/auth/v1/settings"), expect.objectContaining({
      headers: { apikey: "anon", authorization: "Bearer anon" },
    }));
  });

  it("rejects a known disabled provider before redirecting", () => {
    const { assign } = installWindow();
    expect(() => signInWithGoogle({ ...config, providers: { email: true, google: false, twitter: false } }))
      .toThrow("Google sign-in is not configured yet.");
    expect(assign).not.toHaveBeenCalled();
  });

  it("builds Google and X sign-in URLs with the current return location", () => {
    const { assign } = installWindow();
    signInWithGoogle(config);
    signInWithX(config);
    expect(assign).toHaveBeenCalledTimes(2);
    const google = new URL(assign.mock.calls[0]![0] as string);
    const x = new URL(assign.mock.calls[1]![0] as string);
    expect(google.pathname).toBe("/auth/v1/authorize");
    expect(google.searchParams.get("provider")).toBe("google");
    expect(x.searchParams.get("provider")).toBe("twitter");
    const redirect = new URL(google.searchParams.get("redirect_to")!);
    expect(redirect.origin + redirect.pathname).toBe("https://app.example/dashboard");
    expect(redirect.searchParams.get("view")).toBe("overview");
    expect(redirect.searchParams.get("mycellios_oauth_state")).toMatch(/^[0-9a-f]{32}$/);
  });

  it("requests a bearer-authenticated manual link before redirecting", async () => {
    const { assign } = installWindow();
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) =>
      Response.json({ url: "https://accounts.example/link" }),
    );
    vi.stubGlobal("fetch", fetchMock);
    await linkOAuthIdentity(config, { accessToken: "access" }, "google");
    const [request, init] = fetchMock.mock.calls[0]!;
    const url = new URL(String(request));
    expect(url.pathname).toBe("/auth/v1/user/identities/authorize");
    expect(url.searchParams.get("skip_http_redirect")).toBe("true");
    expect(init?.headers).toMatchObject({ apikey: "anon", authorization: "Bearer access" });
    expect(init?.signal).toBeInstanceOf(AbortSignal);
    expect(assign).toHaveBeenCalledWith("https://accounts.example/link");
  });

  it("refuses an unsafe account-link redirect", async () => {
    const { assign, sessionStorage } = installWindow();
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({ url: "javascript:alert(1)" })));
    await expect(linkOAuthIdentity(config, { accessToken: "access" }, "google")).rejects.toThrow("unsafe link");
    expect(assign).not.toHaveBeenCalled();
    expect(sessionStorage.getItem("mycellios.auth.oauth-state")).toBeNull();
  });

  it("accepts only the OAuth return started in this tab", async () => {
    const { assign, setItem, replaceState, sessionStorage } = installWindow();
    signInWithGoogle(config);
    const authorize = new URL(assign.mock.calls[0]![0] as string);
    window.location.search = new URL(authorize.searchParams.get("redirect_to")!).search;
    window.location.hash = "#access_token=access&refresh_token=refresh&expires_in=3600";
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({ id: "user-1", email: "same@example.com" })));
    const session = await restoreAuthSession(config);
    expect(session?.user).toEqual({ id: "user-1", email: "same@example.com" });
    expect(setItem).toHaveBeenCalledWith("mycellios.auth.session", expect.any(String));
    expect(replaceState).toHaveBeenCalledWith(null, "", "/dashboard?view=overview");
    expect(sessionStorage.getItem("mycellios.auth.oauth-state")).toBeNull();
  });

  it("ignores an unsolicited token fragment without contacting the identity service", async () => {
    const { setItem, replaceState } = installWindow("#access_token=attacker&refresh_token=attacker");
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    await expect(restoreAuthSession(config)).resolves.toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(setItem).not.toHaveBeenCalled();
    expect(replaceState).toHaveBeenCalledWith(null, "", "/dashboard?view=overview");
  });

  it("rejects a mismatched OAuth return and consumes the pending state", async () => {
    const { setItem, sessionStorage } = installWindow();
    signInWithGoogle(config);
    window.location.search = "?view=overview&mycellios_oauth_state=wrong";
    window.location.hash = "#access_token=attacker&refresh_token=attacker";
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    await expect(restoreAuthSession(config)).resolves.toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(setItem).not.toHaveBeenCalled();
    expect(sessionStorage.getItem("mycellios.auth.oauth-state")).toBeNull();
  });

  it("rejects an expired OAuth return", async () => {
    const { assign, sessionStorage } = installWindow();
    signInWithGoogle(config);
    const authorize = new URL(assign.mock.calls[0]![0] as string);
    window.location.search = new URL(authorize.searchParams.get("redirect_to")!).search;
    window.location.hash = "#access_token=old&refresh_token=old";
    const pending = JSON.parse(sessionStorage.getItem("mycellios.auth.oauth-state")!) as { state: string; startedAt: number };
    sessionStorage.setItem("mycellios.auth.oauth-state", JSON.stringify({ ...pending, startedAt: Date.now() - 11 * 60_000 }));
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    await expect(restoreAuthSession(config)).resolves.toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("account service failures", () => {
  it("keeps a verified identity only while a transiently affected session is still valid", () => {
    const valid = { expiresAt: 10_000 };
    expect(sessionSurvivesRefreshFailure(valid, new AuthServiceError("Unavailable", 503), 9_000)).toBe(true);
    expect(sessionSurvivesRefreshFailure(valid, new AuthServiceError("Unauthorized", 401), 9_000)).toBe(false);
    expect(sessionSurvivesRefreshFailure(valid, new AuthServiceError("Unavailable", 503), 10_000)).toBe(false);
  });

  it("identifies a successful signup that requires email confirmation", async () => {
    const { setItem } = installWindow();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json({ user: { id: "u1", email: "person@example.com" }, session: null })));
    await expect(signUp(config, "person@example.com", "password123"))
      .rejects.toBeInstanceOf(EmailConfirmationRequired);
    expect(setItem).not.toHaveBeenCalled();
  });

  it("preserves a still-valid session through a temporary refresh outage", async () => {
    const { removeItem } = installWindow();
    const session = { accessToken: "access", refreshToken: "refresh", expiresAt: Date.now() + 30_000, user: { id: "u1", email: null } };
    window.localStorage.getItem = vi.fn(() => JSON.stringify(session));
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 503 })));
    await expect(restoreAuthSession(config)).resolves.toEqual(session);
    expect(removeItem).not.toHaveBeenCalled();
  });

  it("clears a refresh token only after a definitive rejection", async () => {
    const { removeItem } = installWindow();
    const session = { accessToken: "access", refreshToken: "refresh", expiresAt: Date.now() - 1_000, user: { id: "u1", email: null } };
    window.localStorage.getItem = vi.fn(() => JSON.stringify(session));
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 401 })));
    await expect(restoreAuthSession(config)).resolves.toBeNull();
    expect(removeItem).toHaveBeenCalledWith("mycellios.auth.session");
  });

  it("keeps a replacement session if an older refresh is rejected", async () => {
    const { removeItem } = installWindow();
    const oldSession = { accessToken: "old", refreshToken: "old-refresh", expiresAt: Date.now() + 10_000, user: { id: "u1", email: null } };
    const replacement = { accessToken: "other", refreshToken: "other-refresh", expiresAt: Date.now() + 60_000, user: { id: "u2", email: null } };
    let stored = oldSession;
    window.localStorage.getItem = vi.fn(() => JSON.stringify(stored));
    let rejectRefresh!: (response: Response) => void;
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>((resolve) => { rejectRefresh = resolve; })));
    const restoring = restoreAuthSession(config);
    await Promise.resolve();
    stored = replacement;
    rejectRefresh(new Response(null, { status: 401 }));
    await expect(restoring).resolves.toEqual(replacement);
    expect(removeItem).not.toHaveBeenCalled();
  });

  it("distinguishes an unavailable network role from an invalid session", async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(new Response(null, { status: 503 }))
      .mockResolvedValueOnce(new Response(null, { status: 401 }));
    vi.stubGlobal("fetch", fetchMock);
    await expect(loadNetworkIdentity("access")).rejects.toThrow("Network permissions could not be verified (HTTP 503).");
    await expect(loadNetworkIdentity("access")).rejects.toThrow("Your Mycellios session is no longer valid.");
    expect(fetchMock.mock.calls[0]?.[1]?.signal).toBeInstanceOf(AbortSignal);
  });

  it("ends a stalled sign-in with an actionable message", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new DOMException("Timed out", "TimeoutError")));
    await expect(signIn(config, "person@example.com", "password"))
      .rejects.toThrow("The account service did not respond. Try again.");
  });
});
