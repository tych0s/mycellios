import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { boundaryViolation } from "../src/program/component-boundaries.js";

const files = execFileSync("git", [
  "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.ts", "*.tsx",
], { encoding: "utf8" }).trim().split("\n").filter((file) => Boolean(file) && existsSync(file));
const importPattern = /(?:from\s+|import\s*\(|export\s+[^"']*from\s+)["']([^"']+)["']/gu;
const violations: string[] = [];
const debtDocument = JSON.parse(await readFile("config/component-boundary-debt.json", "utf8")) as {
  schema: string; expiresAtGate: string; entries: string[];
};
if (debtDocument.schema !== "mycellios-component-boundary-debt/1") throw new Error("Invalid component boundary debt schema");
const allowedDebt = new Set(debtDocument.entries);
const observedDebt = new Set<string>();

for (const file of files) {
  const source = await readFile(file, "utf8");
  for (const match of source.matchAll(importPattern)) {
    const specifier = match[1];
    if (!specifier) continue;
    const reason = boundaryViolation(file, specifier);
    if (reason) {
      const identity = `${file}|${specifier}|${reason}`;
      if (allowedDebt.has(identity)) observedDebt.add(identity);
      else violations.push(`${file}: ${specifier} (${reason})`);
    }
  }
}

if (violations.length > 0) {
  console.error(["Component boundary violations:", ...violations].join("\n"));
  process.exitCode = 1;
} else {
  const staleDebt = [...allowedDebt].filter((entry) => !observedDebt.has(entry));
  if (staleDebt.length > 0) {
    console.error(["Remove resolved entries from component-boundary-debt.json:", ...staleDebt].join("\n"));
    process.exitCode = 1;
  } else {
    console.log(`Component boundaries verified across ${files.length} TypeScript files; ${observedDebt.size} migration debts expire at ${debtDocument.expiresAtGate}.`);
  }
}
