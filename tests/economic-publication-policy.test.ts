import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { verifyEconomicPublicationPolicy } from "../scripts/verify-economic-publication-policy.js";

const fixture = () => JSON.parse(readFileSync("config/economic-publication-policy.json", "utf8")) as Record<string, unknown>;

describe("economic publication policy", () => {
  it("keeps the verified ledger internal while reviews are blocked", () => {
    const policy = verifyEconomicPublicationPolicy(fixture());
    expect(policy).toMatchObject({ internalLedgerOnly: true, publicEarningsEnabled: false, payoutEnabled: false });
    expect(policy.controls.every(({ owner, prerequisite }) => owner.length > 0 && prerequisite.length > 0)).toBe(true);
  });

  it("fails closed if earnings or payout are enabled prematurely", () => {
    expect(() => verifyEconomicPublicationPolicy({ ...fixture(), publicEarningsEnabled: true })).toThrow("public economics cannot open");
    expect(() => verifyEconomicPublicationPolicy({ ...fixture(), payoutEnabled: true })).toThrow("public economics cannot open");
  });
});
