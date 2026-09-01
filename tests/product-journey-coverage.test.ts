import { access } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { PRODUCT_JOURNEY_GATE_TESTS, PRODUCT_JOURNEY_TESTS } from "../scripts/gate-program.js";

const REQUIRED_JOURNEYS = [
  "J1_NEW_CONTRIBUTOR",
  "J2_FIRST_INFERENCE",
  "J3_DISTRIBUTED_ACTIVATION",
  "J4_UPDATE_AND_ROLLBACK",
  "J5_EXECUTION_FAILURE",
  "J6_REVOCATION_AND_UNINSTALL",
] as const;

describe("P6-D automatic product journey coverage", () => {
  it("binds every J1-J6 acceptance journey to dedicated executable evidence", async () => {
    expect(Object.keys(PRODUCT_JOURNEY_TESTS)).toEqual(REQUIRED_JOURNEYS);

    for (const journey of REQUIRED_JOURNEYS) {
      const evidence = PRODUCT_JOURNEY_TESTS[journey];
      expect(evidence.length, `${journey} must not rely on one narrow test`).toBeGreaterThanOrEqual(3);
      await Promise.all(evidence.map((file) => access(file)));
      expect(evidence.every((file) => PRODUCT_JOURNEY_GATE_TESTS.includes(file))).toBe(true);
    }
  });

  it("does not silently duplicate evidence or omit the coverage contract from the gate", () => {
    expect(new Set(PRODUCT_JOURNEY_GATE_TESTS).size).toBe(PRODUCT_JOURNEY_GATE_TESTS.length);
    expect(PRODUCT_JOURNEY_GATE_TESTS).toContain("tests/product-journey-coverage.test.ts");
  });
});
