import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

const root = resolve(process.argv[2] ?? ".");
const thirdPartyPath = resolve(root, "THIRD_PARTY.md");
const pendingLicense = `The Mycellios kernel licence is pending a decision by the owner. Until a
\`LICENSE\` file exists, no open-source licence should be assumed for Mycellios
code.`;
const publicLicense = `Unless a file states otherwise, Mycellios source code is licensed under the
GNU General Public License, version 3 only (\`GPL-3.0-only\`).`;

const thirdParty = readFileSync(thirdPartyPath, "utf8");
if (thirdParty.includes(pendingLicense)) {
  writeFileSync(thirdPartyPath, thirdParty.replace(pendingLicense, publicLicense));
} else if (!thirdParty.includes(publicLicense)) {
  throw new Error("THIRD_PARTY.md contains neither the pending nor public license notice");
}

for (const relativePath of [
  "README.md",
  "docs/README.md",
  "docs/ARCHITECTURE.md",
  "docs/TWO_HOST_QUICKSTART.md",
  "docs/STATUS_AND_EVIDENCE.md",
  "docs/DEVELOPMENT.md",
  "docs/REPOSITORY_STRUCTURE.md",
  "docs/ROADMAP.md",
  "docs/SECURITY.md",
]) {
  const path = resolve(root, relativePath);
  const source = readFileSync(path, "utf8");
  for (const match of source.matchAll(/\[[^\]]+\]\(([^)]+)\)/g)) {
    const target = match[1].split("#", 1)[0];
    if (!target || /^(?:https?:|mailto:|#|\/)/.test(target)) continue;
    if (!existsSync(resolve(dirname(path), target))) {
      throw new Error(`broken public documentation link: ${relativePath} -> ${target}`);
    }
  }
}
