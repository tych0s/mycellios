import { describe, expect, it } from "vitest";
import { recentAal2ClaimsAreValid } from "../src/coordinator/server.js";

function token(payload: Record<string, unknown>): string {
  return `header.${Buffer.from(JSON.stringify(payload)).toString("base64url")}.signature`;
}

describe("node sensitive-action reauthentication", () => {
  it("requires recent AAL2 claims", () => {
    const now = Date.parse("2026-08-10T12:00:00.000Z");
    expect(recentAal2ClaimsAreValid(token({ aal: "aal2", auth_time: now / 1_000 - 60 }), now)).toBe(true);
    expect(recentAal2ClaimsAreValid(token({ aal: "aal2", amr: [{ method: "totp", timestamp: now / 1_000 - 30 }] }), now)).toBe(true);
    expect(recentAal2ClaimsAreValid(token({ aal: "aal2", auth_time: now / 1_000 - 3_600, amr: [{ method: "totp", timestamp: now / 1_000 - 30 }] }), now)).toBe(true);
    expect(recentAal2ClaimsAreValid(token({ aal: "aal2", amr: [{ method: "token_refresh", timestamp: now / 1_000 }] }), now)).toBe(false);
    expect(recentAal2ClaimsAreValid(token({ aal: "aal1", auth_time: now / 1_000 }), now)).toBe(false);
    expect(recentAal2ClaimsAreValid(token({ aal: "aal2", auth_time: now / 1_000 - 301 }), now)).toBe(false);
    expect(recentAal2ClaimsAreValid("opaque-session", now)).toBe(false);
  });
});
