import { existsSync, readFileSync, realpathSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { CANONICAL_DOCS } from "./verify-repository-structure.mjs";

export const ROOT_DOCUMENTS = Object.freeze([
  "AGENTS.md",
  "CHANGELOG.md",
  "CODE_OF_CONDUCT.md",
  "CONTRIBUTING.md",
  "GOVERNANCE.md",
  "README.md",
  "SECURITY.md",
  "THIRD_PARTY.md",
]);

const markdownLinkPattern = /(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+["'][^"']*["'])?\)/gu;

function localTarget(link) {
  if (/^(?:https?:|mailto:|tel:|data:)/iu.test(link) || link.startsWith("#")) return null;
  const withoutFragment = link.split("#", 1)[0]?.split("?", 1)[0] ?? "";
  return withoutFragment.length > 0 ? decodeURIComponent(withoutFragment) : null;
}

export function verifyDocumentationLinks(root = process.cwd(), files = [
  ...ROOT_DOCUMENTS,
  ...CANONICAL_DOCS.filter((file) => file.endsWith(".md")).map((file) => `docs/${file}`),
]) {
  const absoluteRoot = realpathSync(root);
  const failures = [];
  let checkedLinks = 0;
  for (const file of files) {
    const absoluteFile = resolve(absoluteRoot, file);
    if (!existsSync(absoluteFile)) {
      failures.push(`missing_document:${file}`);
      continue;
    }
    const content = readFileSync(absoluteFile, "utf8");
    for (const match of content.matchAll(markdownLinkPattern)) {
      const target = localTarget(match[1]);
      if (!target) continue;
      checkedLinks += 1;
      const absoluteTarget = resolve(dirname(absoluteFile), target);
      const portable = relative(absoluteRoot, absoluteTarget).replaceAll("\\", "/");
      if (portable === ".." || portable.startsWith("../")) {
        failures.push(`link_escapes_repository:${file}:${match[1]}`);
      } else if (!existsSync(absoluteTarget)) {
        failures.push(`broken_link:${file}:${match[1]}`);
      }
    }
  }
  if (failures.length > 0) {
    throw new Error(`documentation_links_invalid\n${failures.join("\n")}`);
  }
  return { documents: files.length, localLinks: checkedLinks };
}

const invokedDirectly = process.argv[1]
  ? pathToFileURL(resolve(process.argv[1])).href === import.meta.url
  : false;
if (invokedDirectly) {
  const result = verifyDocumentationLinks(resolve(process.argv[2] ?? "."));
  console.log(`Documentation verified: ${result.documents} documents, ${result.localLinks} local links.`);
}
