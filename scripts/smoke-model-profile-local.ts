import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { delimiter, dirname, join, resolve } from "node:path";
import process from "node:process";
import { profileCompatibleModelWithRuntime } from "../src/distribution/auto-distribute.js";

const args = process.argv.slice(2);
if (args.includes("--help")) {
  console.log(`Profile a real model with the current source and an installed Mycellios runtime.

Usage:
  npm run smoke:model-profile:local -- [options]

Options:
  --model <id>       Hugging Face or local model (tiny test model by default)
  --revision <rev>   Exact optional revision
  --runtime <path>   Installed distribution runtime root
  --user-data <path> Mycellios user-data directory
  --json             Print the complete compiled profile

This command does not build an image, publish or deploy anything.`);
  process.exit(0);
}

const workspace = resolve(import.meta.dirname, "..");
const userData = resolve(option("--user-data") ?? defaultUserData());
const runtimeRoot = resolve(
  option("--runtime")
    ?? process.env.MYCELLIOS_DESKTOP_RUNTIME_ROOT
    ?? defaultRuntimeRoot(userData, workspace),
);
const pythonExecutable = process.platform === "win32"
  ? join(runtimeRoot, "python.exe")
  : join(runtimeRoot, "bin", "python");
if (!existsSync(pythonExecutable)) {
  throw new Error(`Installed Mycellios Python is missing: ${pythonExecutable}`);
}

const model = option("--model") ?? "hmellor/tiny-random-LlamaForCausalLM";
const profile = await profileCompatibleModelWithRuntime({
  source: model,
  revision: option("--revision") ?? null,
  pythonExecutable,
  pythonPath: join(workspace, "python"),
  hfHome: join(userData, "model-shards"),
  cwd: workspace,
  environment: {
    ...process.env,
    PATH: [runtimeRoot, dirname(pythonExecutable), process.env.PATH ?? ""]
      .filter(Boolean)
      .join(delimiter),
  },
});

if (args.includes("--json")) {
  console.log(JSON.stringify(profile, null, 2));
} else {
  console.log("Model profile accepted.");
  console.log(`  Model: ${profile.source.model}`);
  console.log(`  Architecture: ${profile.inspection.architecture ?? "not reported"}`);
  console.log(`  Layers: ${profile.model.layers.length}`);
  console.log(`  Snapshot: ${profile.source.snapshotCommit ?? profile.source.revision ?? "default"}`);
}

function defaultUserData(): string {
  if (process.platform === "win32") {
    const roaming = process.env.APPDATA;
    if (!roaming) throw new Error("APPDATA is required on Windows.");
    return join(roaming, "mycellios");
  }
  if (process.platform === "darwin") {
    return join(homedir(), "Library", "Application Support", "mycellios");
  }
  return join(process.env.XDG_CONFIG_HOME ?? join(homedir(), ".config"), "mycellios");
}

function defaultRuntimeRoot(userData: string, workspace: string): string {
  const installed = join(userData, "distribution-runtime-v4");
  if (existsSync(installed)) return installed;
  return join(workspace, "runtime", "distribution-venv");
}

function option(name: string): string | undefined {
  const index = args.indexOf(name);
  if (index === -1) return undefined;
  const value = args[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} requires a value.`);
  return value;
}
