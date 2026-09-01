import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";

const files = execFileSync("git", ["ls-files", "--cached", "--others", "--exclude-standard", "--", "*.ts", "*.tsx", "*.mts", "*.mjs", "*.py", "*.json", "*.yml", "*.yaml"], { encoding: "utf8" })
  .trim().split("\n").filter((file) => (
    file
    && !file.startsWith("native-helpers/")
    && !file.startsWith("sidecars/")
    && !file.includes("/vendor/")
  ));
const failures = [];
for (const file of files) {
  const text = readFileSync(file, "utf8");
  if (text.includes("\r\n")) failures.push(`${file}:crlf`);
  if (text.length > 0 && !text.endsWith("\n")) failures.push(`${file}:missing_final_newline`);
  const trailing = text.split("\n").findIndex((line) => /[ \t]+$/.test(line));
  if (trailing >= 0) failures.push(`${file}:trailing_whitespace:${trailing + 1}`);
}
if (failures.length) throw new Error(`text_format_invalid\n${failures.join("\n")}`);
console.log(`Text format verified: ${files.length} source and configuration files.`);
