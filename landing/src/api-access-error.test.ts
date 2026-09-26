import { describe, expect, it } from "vitest";
import { apiAccessErrorPresentation, apiAccessSessionNeedsRefresh, copyApiValue } from "./ApiAccessPanel";

describe("apiAccessErrorPresentation", () => {
  it("explains invalid network tokens and offers re-authentication", () => {
    expect(apiAccessErrorPresentation("invalid_network_token")).toMatchObject({
      title: "Your API session has expired",
      action: "Sign in",
      requiresAuth: true,
    });
  });
  it.each([
    "invalid_network_token",
    "Invalid network token",
    "invalid_access_token",
    "Account session is invalid",
    "Session is invalid or expired",
  ])("automatically refreshes rejected authenticated sessions: %s", (message) => {
    expect(apiAccessSessionNeedsRefresh(new Error(message))).toBe(true);
  });
  it("does not refresh ordinary API failures", () => {
    expect(apiAccessSessionNeedsRefresh(new Error("HTTP 503"))).toBe(false);
  });
  it("keeps ordinary API errors actionable", () => {
    expect(apiAccessErrorPresentation("HTTP 503")).toMatchObject({
      title: "Could not check the API",
      detail: "HTTP 503",
      action: "Retry",
      requiresAuth: false,
    });
  });
  it("offers the right action for key loading and mutations", () => {
    expect(apiAccessErrorPresentation("HTTP 503", "keys")).toMatchObject({
      title: "Could not load API keys", action: "Retry",
    });
    expect(apiAccessErrorPresentation("HTTP 503", "mutation")).toMatchObject({
      title: "Could not change the API key", action: "Dismiss",
    });
    expect(apiAccessErrorPresentation("invalid_access_token", "keys")).toMatchObject({
      action: "Sign in", requiresAuth: true,
    });
  });
});

describe("copyApiValue", () => {
  it("reports successful copying without changing the API key", async () => {
    let copied = "";
    expect(await copyApiValue("secret-key", async (text) => { copied = text; })).toBe("copied");
    expect(copied).toBe("secret-key");
  });

  it("reports clipboard denial so the user can copy the visible key manually", async () => {
    expect(await copyApiValue("secret-key", async () => { throw new Error("denied"); })).toBe("failed");
  });
});
