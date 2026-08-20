import { describe, expect, it } from "vitest";
import { apiRequestUrl } from "./api-access";

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
