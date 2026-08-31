import { describe, expect, it } from "vitest";
import {
  analyzeArchitecture,
  countLines,
  type ModuleSizeBudget,
} from "../src/program/architecture-health.js";

const budget: ModuleSizeBudget = {
  schema: "mycellios-module-size-budget/2",
  defaultMaximumLines: 3,
  trackedDebt: {},
};

describe("architecture health", () => {
  it("counts files with and without trailing newlines consistently", () => {
    expect(countLines("one\ntwo\n")).toBe(2);
    expect(countLines("one\ntwo")).toBe(2);
    expect(countLines("")).toBe(0);
  });

  it("reports cycles and unowned TypeScript areas", () => {
    const result = analyzeArchitecture([
      { path: "src/coordinator/a.ts", content: 'import "./b.js";\n' },
      { path: "src/coordinator/b.ts", content: 'import "./a.js";\n' },
      { path: "src/unowned/value.ts", content: "export const value = 1;\n" },
    ], [], budget);

    expect(result.cycles).toEqual([[
      "src/coordinator/a.ts",
      "src/coordinator/b.ts",
    ]]);
    expect(result.unknownAreas).toEqual(["src/unowned/value.ts"]);
  });

  it("ratchets known debt and rejects new oversized modules", () => {
    const result = analyzeArchitecture([], [
      { path: "src/coordinator/known.ts", content: "1\n2\n3\n4\n" },
      { path: "src/coordinator/new.ts", content: "1\n2\n3\n4\n" },
    ], {
      ...budget,
      trackedDebt: { "src/coordinator/known.ts": { ceiling: 4, target: 3, owner: "coordinator" } },
    });

    expect(result.sizeViolations).toEqual(["src/coordinator/new.ts:4>3"]);
    expect(result.staleSizeDebt).toEqual([]);
    expect(result.reducibleSizeDebt).toEqual([]);
  });

  it("rejects debt entries after their module disappears", () => {
    const result = analyzeArchitecture([], [], {
      ...budget,
      trackedDebt: { "src/coordinator/removed.ts": { ceiling: 10, target: 8, owner: "coordinator" } },
    });
    expect(result.staleSizeDebt).toEqual(["src/coordinator/removed.ts"]);
  });

  it("requires lowering a ceiling once its target is reached", () => {
    const result = analyzeArchitecture([], [{ path: "src/coordinator/known.ts", content: "1\n2\n3\n" }], {
      ...budget, trackedDebt: { "src/coordinator/known.ts": { ceiling: 4, target: 3, owner: "coordinator" } },
    });
    expect(result.reducibleSizeDebt).toEqual(["src/coordinator/known.ts:3<4:target=3"]);
  });
});
