import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { verifySecurityP0Controls } from "../scripts/verify-security-p0-controls.js";

const fixture = () => JSON.parse(readFileSync("config/security-p0-controls.json", "utf8")) as Record<string, unknown>;

describe("security P0 control matrix", () => {
  it("binds every control and residual to existing evidence and an owner", () => {
    const matrix = verifySecurityP0Controls(fixture());
    expect(matrix.publicNetworkAllowed).toBe(false);
    expect(matrix.controls.filter(({ status }) => status === "implemented").map(({ id }) => id)).toEqual([
      "os_isolation", "account_rate_limits", "append_only_audit", "owner_key_recovery",
    ]);
  });

  it("fails closed if public enrollment opens before every control is implemented", () => {
    expect(() => verifySecurityP0Controls({ ...fixture(), publicNetworkAllowed: true }))
      .toThrow("public network cannot open with blocked security controls");
  });

  it("requires an owner, prerequisite and evidence for each residual", () => {
    const value = fixture() as { controls: Array<Record<string, unknown>> };
    value.controls = value.controls.map((control) => control.id === "production_tls_perimeter"
      ? { ...control, prerequisite: undefined }
      : control);
    expect(() => verifySecurityP0Controls(value)).toThrow();
  });
});
