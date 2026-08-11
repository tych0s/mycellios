import { afterEach, describe, expect, it, vi } from "vitest";
import { sessionHasRecentAal2, verifyTotpStepUp, type AuthSession, type PublicAuthConfig } from "./auth";

const config: PublicAuthConfig = { enabled: true, url: "https://auth.example", anonKey: "anon" };
const session: AuthSession = {
  accessToken: token({ aal: "aal1", auth_time: 1 }), refreshToken: "refresh-1", expiresAt: 1,
  user: { id: "account-1", email: "owner@example.com" },
};

afterEach(() => vi.unstubAllGlobals());

describe("account MFA step-up", () => {
  it("recognizes only recent AAL2 access tokens", () => {
    const now = Date.parse("2026-08-10T12:00:00Z");
    expect(sessionHasRecentAal2({ accessToken: token({ aal: "aal2", auth_time: now / 1_000 - 60 }) }, now)).toBe(true);
    expect(sessionHasRecentAal2({ accessToken: token({ aal: "aal2", auth_time: now / 1_000 - 3_600,
      amr: [{ method: "totp", timestamp: now / 1_000 - 30 }] }) }, now)).toBe(true);
    expect(sessionHasRecentAal2({ accessToken: token({ aal: "aal2", auth_time: now / 1_000 - 3_600,
      amr: [{ method: "token_refresh", timestamp: now / 1_000 }] }) }, now)).toBe(false);
    expect(sessionHasRecentAal2({ accessToken: token({ aal: "aal2", auth_time: now / 1_000 - 301 }) }, now)).toBe(false);
    expect(sessionHasRecentAal2(session, now)).toBe(false);
  });

  it("challenges a verified TOTP factor and persists the elevated token", async () => {
    const elevated = token({ aal: "aal2", auth_time: Math.floor(Date.now() / 1_000) });
    const setItem = vi.fn();
    vi.stubGlobal("window", { localStorage: { setItem } });
    const request = vi.fn()
      .mockResolvedValueOnce(Response.json({ totp: [{ id: "factor-1", status: "verified" }] }))
      .mockResolvedValueOnce(Response.json({ id: "challenge-1" }))
      .mockResolvedValueOnce(Response.json({ access_token: elevated, refresh_token: "refresh-2", expires_in: 600 }));
    vi.stubGlobal("fetch", request);

    await expect(verifyTotpStepUp(config, session, "123456")).resolves.toMatchObject({
      accessToken: elevated, refreshToken: "refresh-2", user: session.user,
    });
    expect(request.mock.calls.map(([url]) => String(url))).toEqual([
      "https://auth.example/auth/v1/factors",
      "https://auth.example/auth/v1/factors/factor-1/challenge",
      "https://auth.example/auth/v1/factors/factor-1/verify",
    ]);
    expect(JSON.parse(String(request.mock.calls[2]?.[1]?.body))).toEqual({ challenge_id: "challenge-1", code: "123456" });
    expect(setItem).toHaveBeenCalledOnce();
  });

  it("fails closed when no verified TOTP factor exists", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({ totp: [{ id: "factor-1", status: "unverified" }] })));
    await expect(verifyTotpStepUp(config, session, "123456")).rejects.toThrow("No verified TOTP factor");
  });
});

function token(claims: Record<string, unknown>): string {
  return `header.${Buffer.from(JSON.stringify(claims)).toString("base64url")}.signature`;
}
