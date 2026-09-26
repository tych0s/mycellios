import { afterEach, describe, expect, it, vi } from "vitest";
import { apiRequestUrl, loadApiAccount } from "./api-access";

afterEach(() => vi.unstubAllGlobals());

describe("apiRequestUrl", () => {
  it("keeps authenticated routes below a /v1 API base", () => {
    expect(apiRequestUrl("/v1/account", "https://www.mycellios.com/v1"))
      .toBe("https://www.mycellios.com/v1/account");
  });

  it("resolves public routes beside a /v1 API base", () => {
    expect(apiRequestUrl("/public/v1/admin/operations", "https://www.mycellios.com/v1"))
      .toBe("/public/v1/admin/operations");
  });
});

describe("authenticated API requests", () => {
  it("bounds account loading and reports a readable timeout", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new DOMException("Timed out", "TimeoutError"));
    vi.stubGlobal("fetch", fetchMock);
    await expect(loadApiAccount("access")).rejects.toThrow("The API request timed out. Try again.");
    expect(fetchMock.mock.calls[0]?.[1]?.signal).toBeInstanceOf(AbortSignal);
  });
});
