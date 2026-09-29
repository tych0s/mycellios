import { describe, expect, it } from "vitest";
import { canonicalLandingRedirectLocation } from "../src/coordinator/landing-entry-routes.js";

describe("canonical landing host", () => {
  it("permanently redirects public pages on the apex and keeps their query", () => {
    expect(canonicalLandingRedirectLocation("GET", "mycellios.com", "/network/?view=globe"))
      .toBe("https://www.mycellios.com/network?view=globe");
    expect(canonicalLandingRedirectLocation("GET", "MYCELLIOS.COM", "/blog/research/"))
      .toBe("https://www.mycellios.com/blog/research");
    expect(canonicalLandingRedirectLocation("GET", "mycellios.com", "/llms.txt"))
      .toBe("https://www.mycellios.com/llms.txt");
  });

  it("routes browser and mobile aliases to the canonical browser worker", () => {
    expect(canonicalLandingRedirectLocation("GET", "mycellios.com", "/browser"))
      .toBe("https://www.mycellios.com/browser/");
    expect(canonicalLandingRedirectLocation("GET", "mycellios.com", "/mobile?autostart=1"))
      .toBe("https://www.mycellios.com/browser/?autostart=1");
  });

  it("leaves the canonical host, APIs, and non-read requests alone", () => {
    expect(canonicalLandingRedirectLocation("GET", "www.mycellios.com", "/network")).toBeNull();
    expect(canonicalLandingRedirectLocation("GET", "mycellios.com", "/v1/chat/completions")).toBeNull();
    expect(canonicalLandingRedirectLocation("POST", "mycellios.com", "/network")).toBeNull();
  });
});
