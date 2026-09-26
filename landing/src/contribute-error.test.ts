import { describe, expect, it } from "vitest";
import { contributionErrorPresentation } from "./Contribute";

describe("contributionErrorPresentation", () => {
  it("explains an expired network token and routes the user back to auth", () => {
    expect(contributionErrorPresentation("invalid_network_token")).toEqual({
      title: "Your contribution session has expired",
      detail: "This browser's network credential is no longer valid. Sign in again to reconnect your contribution.",
      action: "Sign in",
      requiresAuth: true,
    });
  });

  it("keeps unknown failures useful without exposing a detached technical banner", () => {
    expect(contributionErrorPresentation("Coordinator unavailable")).toMatchObject({
      title: "Could not update contribution",
      detail: "Coordinator unavailable",
      action: "Retry",
      requiresAuth: false,
    });
  });
});
