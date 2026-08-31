import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { verifyWebBundleBudget, type WebBundleSurfaceDefinition } from "../scripts/verify-web-bundle-budget.mjs";

function fixture(files: Record<string, number | string>) {
  const root = mkdtempSync(join(tmpdir(), "mycellios-bundle-"));
  mkdirSync(join(root, "out/assets"), { recursive: true });
  for (const [name, content] of Object.entries(files)) {
    writeFileSync(join(root, "out/assets", name), typeof content === "number" ? "x".repeat(content) : content);
  }
  return root;
}

const definition: WebBundleSurfaceDefinition = {
  outputDirectory: "out",
  sourceMapsAllowed: false,
  defaultMaximumBytes: 5,
  budgets: { shell: { match: "index-", maximumBytes: 10 } },
  criticalPaths: { home: { entry: "index.html", highPriorityMatches: ["hero-"], maximumBytes: 16 } },
};
const manifest = {
  "index.html": { file: "assets/index-test.js", imports: ["_runtime.js"] },
  "_runtime.js": { file: "assets/runtime-test.js" },
  "hero.js": { file: "assets/hero-test.js" },
};

describe("closed web bundle policy", () => {
  it("accepts named, default and critical-path assets within budget", () => {
    const root = fixture({ "index-test.js": 10, "runtime-test.js": 1, "hero-test.js": 5 });
    expect(verifyWebBundleBudget({ root, outputDirectory: "out", manifest, definition }))
      .toEqual(expect.arrayContaining([expect.objectContaining({ kind: "critical-path", bytes: 16 })]));
  });

  it("rejects an unknown chunk above the explicit default", () => {
    const root = fixture({ "index-test.js": 10, "runtime-test.js": 6, "hero-test.js": 1 });
    expect(() => verifyWebBundleBudget({ root, outputDirectory: "out", manifest, definition }))
      .toThrow("unknown_bundle_exceeds_default:assets/runtime-test.js:6>5");
  });

  it("rejects a high-priority critical-path regression", () => {
    const root = fixture({ "index-test.js": 10, "runtime-test.js": 1, "hero-test.js": 6 });
    const relaxed = { ...definition, defaultMaximumBytes: 6 };
    expect(() => verifyWebBundleBudget({ root, outputDirectory: "out", manifest, definition: relaxed }))
      .toThrow("critical_path_budget_exceeded:home:17>16");
  });

  it("rejects an emitted script omitted from the manifest", () => {
    const root = fixture({ "index-test.js": 1, "runtime-test.js": 1, "hero-test.js": 1, "unknown.js": 1 });
    expect(() => verifyWebBundleBudget({ root, outputDirectory: "out", manifest, definition }))
      .toThrow("bundle_not_in_manifest:assets/unknown.js");
  });

  it("rejects public source maps", () => {
    const root = fixture({ "index-test.js": 1, "runtime-test.js": 1, "hero-test.js": 1, "index-test.js.map": "{}" });
    expect(() => verifyWebBundleBudget({ root, outputDirectory: "out", manifest, definition }))
      .toThrow("public_source_maps_forbidden:index-test.js.map");
  });
});
