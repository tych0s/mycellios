import { afterEach, describe, expect, it, vi } from "vitest";
import {
  linkOAuthIdentity,
  loadAuthConfig,
  authSessionNeedsRefresh,
  validAuthSession,
  restoreAuthSession,
  signInWithGoogle,
  signInWithX,
  type PublicAuthConfig,
} from "./auth";

const config: PublicAuthConfig = { enabled: true, url: "https://auth.example", anonKey: "anon" };

function installWindow(hash = "") {
  const assign = vi.fn();
  const setItem = vi.fn();
  const removeItem = vi.fn();
  const replaceState = vi.fn();
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
  });
  return { assign, setItem, replaceState };
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
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({ access_token: "new", refresh_token: "next", expires_in: 3600, user: { id: "u1", email: null } })));
    const renewed = await validAuthSession(config, session);
    expect(renewed.accessToken).toBe("new");
    expect(setItem).toHaveBeenCalledWith("mycellios.auth.session", expect.stringContaining('"accessToken":"new"'));
  });

  it("deduplicates simultaneous forced renewals", async () => {
    installWindow();
    const session = { accessToken: "old", refreshToken: "refresh", expiresAt: Date.now() + 60_000, user: { id: "u1", email: null } };
    const fetchMock = vi.fn(async () => Response.json({ access_token: "new", refresh_token: "next", expires_in: 3600, user: { id: "u1", email: null } }));
    vi.stubGlobal("fetch", fetchMock);
    const [first, second] = await Promise.all([validAuthSession(config, session, true), validAuthSession(config, session, true)]);
    expect(first.accessToken).toBe("new");
    expect(second.accessToken).toBe("new");
    expect(fetchMock).toHaveBeenCalledTimes(1);
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
    const fetchMock = vi.fn(async () => Response.json({ access_token: "accepted", refresh_token: "next", expires_in: 3600, user: { id: "u1", email: null } }));
    vi.stubGlobal("fetch", fetchMock);
    const renewed = await validAuthSession(config, session, true);
    expect(renewed.accessToken).toBe("accepted");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe("Supabase OAuth providers", () => {
  it("loads live provider availability from Supabase settings", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(Response.json(config))
      .mockResolvedValueOnce(Response.json({ external: { email: true, google: true, twitter: false } }));
    vi.stubGlobal("fetch", fetchMock);
    const loaded = await loadAuthConfig();
    expect(loaded.providers).toEqual({ email: true, google: true, twitter: false });
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
    expect(google.searchParams.get("redirect_to")).toBe("https://app.example/dashboard?view=overview");
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
    expect(assign).toHaveBeenCalledWith("https://accounts.example/link");
  });

  it("consumes a provider-neutral OAuth fragment and persists the Supabase user", async () => {
    const { setItem, replaceState } = installWindow("#access_token=access&refresh_token=refresh&expires_in=3600");
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({ id: "user-1", email: "same@example.com" })));
    const session = await restoreAuthSession(config);
    expect(session?.user).toEqual({ id: "user-1", email: "same@example.com" });
    expect(setItem).toHaveBeenCalledWith("mycellios.auth.session", expect.any(String));
    expect(replaceState).toHaveBeenCalledWith(null, "", "/dashboard?view=overview");
  });
});
