import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { CANONICAL_DOCS, verifyRepositoryStructure } from "../scripts/verify-repository-structure.mjs";

const roots: string[] = [];

function fixture(): string {
  const root = join(tmpdir(), `mycellios-structure-${crypto.randomUUID()}`);
  roots.push(root);
  mkdirSync(root, { recursive: true });
  writeFileSync(join(root, "package.json"), JSON.stringify({ packageManager: "npm@10.9.8" }));
  writeFileSync(join(root, "package-lock.json"), "{}\n");
  writeFileSync(join(root, "AGENTS.md"), "# Rules\n");
  for (const area of [
    "src/contracts", "src/coordinator", "src/distribution", "src/transport",
    "src/worker", "python/distributed_runtime", "landing", "tests", "config", "docs",
  ]) mkdirSync(join(root, area), { recursive: true });
  for (const doc of CANONICAL_DOCS) writeFileSync(join(root, "docs", doc), "\n");
  return root;
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("repository structure", () => {
  it("accepts the canonical single npm workspace", () => {
    expect(verifyRepositoryStructure(fixture())).toEqual({
      packageManager: "npm",
      docs: CANONICAL_DOCS.length,
      areas: 10,
    });
  });

  it("rejects a second package manager", () => {
    const root = fixture();
    writeFileSync(join(root, "pnpm-lock.yaml"), "lockfileVersion: 9\n");
    expect(() => verifyRepositoryStructure(root)).toThrow("unsupported_package_manager:pnpm-lock.yaml");
  });

  it("rejects documentation drift", () => {
    const root = fixture();
    writeFileSync(join(root, "docs", "OLD_HANDOFF.md"), "stale\n");
    expect(() => verifyRepositoryStructure(root)).toThrow("canonical_docs_mismatch");
  });

  it("rejects a duplicate documentation site", () => {
    const root = fixture();
    mkdirSync(join(root, "docs-site"));
    expect(() => verifyRepositoryStructure(root)).toThrow("duplicate_documentation_site:docs-site");
  });
});
