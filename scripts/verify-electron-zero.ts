import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { verifyElectronZero } from "../src/program/electron-zero-verifier.js";

const trackedFiles = execFileSync("git", [
  "ls-files", "--cached", "--others", "--exclude-standard",
], { encoding: "utf8" }).trim().split("\n").filter((file) => Boolean(file) && existsSync(file));
const inspectedFiles = trackedFiles.filter((file) =>
  file.startsWith(".github/workflows/")
  || file === "README.md"
  || file.startsWith("docs/")
  || (/^(?:src|landing|scripts)\//u.test(file) && /\.(?:ts|tsx|mts|mjs|js|json|ya?ml|md|ps1)$/u.test(file)));
const fileContents = new Map(await Promise.all(inspectedFiles.map(async (file) =>
  [file, await readFile(file, "utf8")] as const)));
const violations = verifyElectronZero({
  trackedFiles,
  packageDocument: JSON.parse(await readFile("package.json", "utf8")) as Record<string, unknown>,
  lockDocument: JSON.parse(await readFile("package-lock.json", "utf8")) as Record<string, unknown>,
  fileContents,
});

if (violations.length > 0) {
  console.error(`ELECTRON_ZERO failed with ${violations.length} violation(s):`);
  for (const violation of violations) {
    console.error(`${violation.category}:${violation.subject}:${violation.reason}`);
  }
  process.exitCode = 1;
} else {
  console.log(`ELECTRON_ZERO verified across ${trackedFiles.length} tracked files.`);
}
