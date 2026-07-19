import { describe, expect, it } from "vitest";
import { safeVramBudget, tierForOfferedVram } from "../src/core/tiers.js";

describe("worker tiers", () => {
  it("classifies using offered memory rather than physical memory", () => {
    expect(tierForOfferedVram(4_095)).toBe("INELIGIBLE");
    expect(tierForOfferedVram(4_096)).toBe("T0");
    expect(tierForOfferedVram(8_192)).toBe("T1");
    expect(tierForOfferedVram(12_288)).toBe("T2");
    expect(tierForOfferedVram(16_384)).toBe("T3");
    expect(tierForOfferedVram(24_576)).toBe("T4");
  });

  it("keeps a memory safety margin", () => {
    expect(safeVramBudget(4_096)).toBe(3_481);
    expect(safeVramBudget(16_384)).toBe(13_926);
  });
});
