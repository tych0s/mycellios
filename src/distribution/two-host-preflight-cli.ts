import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parseAutoDistributionConfig } from "./auto-distribute.js";
import { evaluateTwoHostPreflight } from "./two-host-preflight.js";

export async function runTwoHostPreflight(argv: readonly string[], environment: NodeJS.ProcessEnv = process.env): Promise<number> {
  const configAt = argv.indexOf("--config");
  const configPath = configAt >= 0 ? argv[configAt + 1] : undefined;
  if (!configPath || configPath.startsWith("--")) throw new Error("two_host_preflight_config_is_required");
  const json = argv.includes("--json");
  const config = parseAutoDistributionConfig(JSON.parse(await readFile(resolve(configPath), "utf8")));
  const environmentNames = new Set(Object.entries(environment).filter(([, value]) => typeof value === "string" && value.length > 0).map(([name]) => name));
  const report = evaluateTwoHostPreflight(config, { python312: hasPython312(config.runtime.pythonExecutable), configuredEnvironmentNames: environmentNames });
  if (json) process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  else {
    process.stdout.write(`Two-host preflight: ${report.ready ? "READY" : report.dryRun ? "DRY RUN ONLY" : "NOT READY"}\n`);
    for (const diagnostic of report.diagnostics) process.stdout.write(`${diagnostic.severity.toUpperCase()} ${diagnostic.code} [${diagnostic.subject}] ${diagnostic.message}\n`);
  }
  return report.ready ? 0 : 2;
}

function hasPython312(executable: string): boolean {
  try {
    return /^Python 3\.12(?:\.|$)/.test(execFileSync(executable, ["--version"], {
      encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], timeout: 5_000, windowsHide: true,
    }).trim());
  } catch { return false; }
}

const invokedDirectly = process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url;
if (invokedDirectly) runTwoHostPreflight(process.argv.slice(2)).then((code) => { process.exitCode = code; }).catch((error) => {
  process.stderr.write(`two-host-preflight: ${error instanceof Error ? error.message : String(error)}\n`); process.exitCode = 2;
});
