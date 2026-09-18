import { existsSync, readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { isAbsolute, resolve } from "node:path";
import { pathToFileURL } from "node:url";

export function evaluateDeveloperEnvironment(input) {
  const checks = [];
  const nodeMajor = Number(String(input.nodeVersion ?? "").replace(/^v/, "").split(".")[0]);
  checks.push({ id: "node", required: true, ok: nodeMajor >= 24, detail: input.nodeVersion || "not found", fix: "Install Node.js 24 or newer." });
  checks.push({ id: "npm", required: true, ok: /^10\./.test(input.npmVersion ?? ""), detail: input.npmVersion || "not found", fix: "Use the npm version pinned by packageManager (Corepack is supported)." });
  checks.push({ id: "lockfile", required: true, ok: input.hasLockfile === true, detail: input.hasLockfile ? "package-lock.json" : "missing", fix: "Restore package-lock.json and run npm ci." });
  checks.push({ id: "package-manager", required: true, ok: input.hasAlternativeLock !== true, detail: input.hasAlternativeLock ? "alternative lockfile found" : "npm only", fix: "Remove pnpm/Yarn lockfiles; this repository is npm-only." });
  checks.push({ id: "python", required: false, ok: /^Python 3\.12(?:\.|$)/.test(input.pythonVersion ?? ""), detail: input.pythonVersion ?? "not found (needed for the physical runtime)", fix: "Install Python 3.12 for runtime and Python tests." });
  checks.push({ id: "git", required: true, ok: input.gitVersion !== null, detail: input.gitVersion ?? "not found", fix: "Install Git before cloning or validating source provenance." });
  const powershellAvailable = input.powershellVersion !== null;
  return {
    ok: checks.every((check) => !check.required || check.ok),
    fullRuntimeReady: checks.every((check) => check.ok),
    checks,
    platform: input.platform,
    capabilities: {
      controlPlane: checks.filter((check) => check.required).every((check) => check.ok),
      pythonRuntime: /^Python 3\.12(?:\.|$)/.test(input.pythonVersion ?? ""),
      portableTwoHostPreflight: checks.filter((check) => check.required).every((check) => check.ok),
      windowsPhysicalScripts: input.platform === "win32" && powershellAvailable,
      powershell: powershellAvailable,
    },
  };
}

function commandVersion(command, args) {
  try {
    return execFileSync(command, args, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], timeout: 10_000, windowsHide: true }).trim();
  } catch {
    return null;
  }
}

export function inspectDeveloperEnvironment(root = process.cwd(), options = {}) {
  const probe = options.commandVersion ?? commandVersion;
  const platform = options.platform ?? process.platform;
  const environment = options.environment ?? process.env;
  const nodeExecutable = options.nodeExecutable ?? process.execPath;
  // npm.cmd cannot be executed directly by execFileSync on Windows. Prefer
  // the actual CLI supplied by npm, preserving paths with spaces as arguments.
  const npmVersion = environment.npm_execpath
    ? probe(nodeExecutable, [environment.npm_execpath, "--version"])
    : platform === "win32"
      ? probe("cmd.exe", ["/d", "/s", "/c", "npm --version"])
      : probe("npm", ["--version"]);
  const runtimeRoot = environment.MYCELLIOS_DESKTOP_RUNTIME_ROOT
    || resolve(root, "runtime", "distribution-venv");
  const candidates = [
    ...(environment.MYCELLIOS_PYTHON ? [environment.MYCELLIOS_PYTHON] : []),
    resolve(runtimeRoot, platform === "win32" ? "python.exe" : "bin/python"),
    ...(platform === "win32" ? [resolve(runtimeRoot, "Scripts", "python.exe")] : []),
    ...(platform === "win32" ? ["py", "python"] : ["python3.12", "python3", "python"]),
  ];
  let pythonVersion = null;
  for (const candidate of candidates) {
    const args = candidate === "py" ? ["-3.12", "--version"] : ["--version"];
    const found = probe(candidate, args);
    pythonVersion ??= found;
    if (/^Python 3\.12(?:\.|$)/.test(found ?? "")) {
      pythonVersion = found;
      break;
    }
  }
  if (!/^Python 3\.12(?:\.|$)/.test(pythonVersion ?? "")) {
    const managedPython = probe("uv", ["python", "find", "3.12"]);
    if (managedPython && isAbsolute(managedPython)) {
      const found = probe(managedPython, ["--version"]);
      if (/^Python 3\.12(?:\.|$)/.test(found ?? "")) pythonVersion = found;
    }
  }
  return evaluateDeveloperEnvironment({
    nodeVersion: process.version,
    npmVersion,
    pythonVersion,
    gitVersion: probe("git", ["--version"]),
    powershellVersion: probe(platform === "win32" ? "powershell.exe" : "pwsh", ["-NoProfile", "-Command", "$PSVersionTable.PSVersion.ToString()"]),
    platform,
    hasLockfile: existsSync(resolve(root, "package-lock.json")),
    hasAlternativeLock: ["pnpm-lock.yaml", "pnpm-workspace.yaml", "yarn.lock"].some((name) => existsSync(resolve(root, name))),
  });
}

function printHuman(result) {
  for (const check of result.checks) {
    const mark = check.ok ? "PASS" : check.required ? "FAIL" : "WARN";
    console.log(`${mark.padEnd(4)} ${check.id.padEnd(16)} ${check.detail}`);
    if (!check.ok) console.log(`     ${check.fix}`);
  }
  console.log(`Platform capabilities: portable-preflight=${result.capabilities.portableTwoHostPreflight ? "yes" : "no"}, python-runtime=${result.capabilities.pythonRuntime ? "yes" : "no"}, windows-physical-scripts=${result.capabilities.windowsPhysicalScripts ? "yes" : "no"}.`);
  console.log(result.fullRuntimeReady ? "Environment ready for full runtime development." : result.ok ? "Control-plane environment ready; optional runtime prerequisites are missing." : "Environment is not ready. Fix required checks above.");
}

const invokedDirectly = process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url;
if (invokedDirectly) {
  const result = inspectDeveloperEnvironment();
  if (process.argv.includes("--json")) console.log(JSON.stringify(result, null, 2));
  else printHuman(result);
  if (!result.ok) process.exitCode = 1;
}
