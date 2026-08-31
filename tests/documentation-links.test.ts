import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { verifyDocumentationLinks } from "../scripts/verify-documentation.mjs";

const roots: string[] = [];

function fixture(): string {
  const root = join(tmpdir(), `mycellios-docs-${crypto.randomUUID()}`);
  roots.push(root);
  mkdirSync(join(root, "docs"), { recursive: true });
  writeFileSync(join(root, "README.md"), "Read [architecture](docs/ARCHITECTURE.md).\n");
  writeFileSync(join(root, "docs", "ARCHITECTURE.md"), "# Architecture\n");
  return root;
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("documentation links", () => {
  it("accepts repository-local links and ignores external links", () => {
    const root = fixture();
    writeFileSync(
      join(root, "README.md"),
      "[Architecture](docs/ARCHITECTURE.md) and [website](https://example.com).\n",
    );
    expect(verifyDocumentationLinks(root, ["README.md", "docs/ARCHITECTURE.md"]))
      .toEqual({ documents: 2, localLinks: 1 });
  });

  it("rejects missing and escaping targets", () => {
    const root = fixture();
    writeFileSync(join(root, "README.md"), "[Missing](docs/MISSING.md) [Escape](../secret.md)\n");
    expect(() => verifyDocumentationLinks(root, ["README.md"]))
      .toThrow(/broken_link:README\.md:docs\/MISSING\.md/);
    expect(() => verifyDocumentationLinks(root, ["README.md"]))
      .toThrow(/link_escapes_repository:README\.md:\.\.\/secret\.md/);
  });
});
