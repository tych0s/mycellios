import { existsSync, readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

export const CANONICAL_DOCS = Object.freeze([
  "ARCHITECTURE.md",
  "DEVELOPMENT.md",
  "README.md",
  "REPOSITORY_STRUCTURE.md",
  "ROADMAP.md",
  "SECURITY.md",
  "STATUS_AND_EVIDENCE.md",
  "TWO_HOST_QUICKSTART.md",
  "openapi.yaml",
]);

const REQUIRED_AREAS = Object.freeze([
  "src/contracts",
  "src/coordinator",
  "src/distribution",
  "src/transport",
  "src/worker",
  "python/distributed_runtime",
  "landing",
  "tests",
  "config",
  "docs",
]);

function sameMembers(actual, expected) {
  return actual.length === expected.length
    && actual.every((entry, index) => entry === expected[index]);
}

export function verifyRepositoryStructure(root = process.cwd()) {
  const failures = [];
  const manifestPath = resolve(root, "package.json");
  if (!existsSync(manifestPath)) {
    failures.push("missing:package.json");
  } else {
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
    if (!/^npm@\d+\.\d+\.\d+$/.test(manifest.packageManager ?? "")) {
      failures.push("packageManager_must_pin_npm");
    }
    if (manifest.workspaces !== undefined) {
      failures.push("npm_workspaces_require_an_explicit_extraction_change");
    }
  }

  if (!existsSync(resolve(root, "package-lock.json"))) failures.push("missing:package-lock.json");
  for (const stale of ["pnpm-lock.yaml", "pnpm-workspace.yaml", "yarn.lock", ".yarn"]) {
    if (existsSync(resolve(root, stale))) failures.push(`unsupported_package_manager:${stale}`);
  }

  if (existsSync(resolve(root, "docs-site"))) failures.push("duplicate_documentation_site:docs-site");
  if (!existsSync(resolve(root, "AGENTS.md"))) failures.push("missing:AGENTS.md");
  for (const area of REQUIRED_AREAS) {
    if (!existsSync(resolve(root, area))) failures.push(`missing:${area}`);
  }

  const docsPath = resolve(root, "docs");
  if (existsSync(docsPath)) {
    const docs = readdirSync(docsPath, { withFileTypes: true })
      .map((entry) => entry.name)
      .sort();
    if (!sameMembers(docs, [...CANONICAL_DOCS].sort())) {
      failures.push(`canonical_docs_mismatch:${docs.join(",")}`);
    }
  }

  if (failures.length > 0) {
    throw new Error(`repository_structure_invalid\n${failures.join("\n")}`);
  }
  return { packageManager: "npm", docs: CANONICAL_DOCS.length, areas: REQUIRED_AREAS.length };
}

const invokedDirectly = process.argv[1]
  ? pathToFileURL(resolve(process.argv[1])).href === import.meta.url
  : false;
if (invokedDirectly) {
  const result = verifyRepositoryStructure(resolve(process.argv[2] ?? "."));
  console.log(`Repository structure verified: ${result.packageManager}, ${result.docs} canonical docs, ${result.areas} required areas.`);
}
