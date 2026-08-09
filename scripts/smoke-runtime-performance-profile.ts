import { existsSync } from "node:fs";
import { readFile, readdir, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { delimiter, join, resolve } from "node:path";
import process from "node:process";
import { probeRuntimePerformanceProfile } from "../src/performance/runtime-profile-probe.js";
import {
  plannerScalesFromProfile,
  type RuntimePerformanceProfileInput,
} from "../src/performance/runtime-profile.js";

interface CachedAcceleratorManifest {
  schema: "mycellios-accelerator-runtime/1";
  backend: RuntimePerformanceProfileInput["backend"];
  probe: {
    backend: RuntimePerformanceProfileInput["backend"];
    device: string;
    deviceName: string;
    precision: RuntimePerformanceProfileInput["precision"];
  };
  environment: {
    pythonPathAdditions: string[];
    pathAdditions: string[];
  };
}

const args = process.argv.slice(2);
if (args.includes("--help")) {
  console.log(`Run the current source parser against an already installed physical accelerator runtime.

Usage:
  npm run smoke:runtime-profile:local -- [options]

Options:
  --backend <name>   cuda, rocm, mps or xpu; newest cached runtime by default
  --user-data <path> Mycellios user-data directory
  --warmups <count>  Physical warm-up samples; probe default when omitted
  --samples <count>  Physical measurement samples; probe default when omitted
  --json             Print the complete sealed profile

This command does not package, publish or deploy anything.`);
  process.exit(0);
}

const workspace = resolve(import.meta.dirname, "..");
const userData = resolve(
  option("--user-data")
    ?? process.env.MYCELLIOS_DESKTOP_USER_DATA
    ?? defaultUserData(),
);
const requestedBackend = option("--backend");
const warmupSamples = integerOption("--warmups");
const samples = integerOption("--samples");
const cachedRoot = join(userData, "accelerator-runtimes-v1");
const selected = await selectCachedRuntime(cachedRoot, requestedBackend);
const pythonExecutable = process.platform === "win32"
  ? join(selected.root, "python.exe")
  : join(selected.root, "bin", "python3");
if (!existsSync(pythonExecutable)) {
  throw new Error(`Cached accelerator Python is missing: ${pythonExecutable}`);
}

const profile = await probeRuntimePerformanceProfile({
  pythonExecutable,
  pythonPath: [
    ...selected.manifest.environment.pythonPathAdditions.map((entry) =>
      resolve(selected.root, entry)
    ),
    join(workspace, "python"),
  ],
  pathAdditions: selected.manifest.environment.pathAdditions.map((entry) =>
    resolve(selected.root, entry)
  ),
  backend: selected.manifest.probe.backend,
  device: selected.manifest.probe.device,
  precision: selected.manifest.probe.precision,
  expectedDeviceName: selected.manifest.probe.deviceName,
  ...(warmupSamples === undefined ? {} : { warmupSamples }),
  ...(samples === undefined ? {} : { samples }),
  cwd: workspace,
  env: {
    HF_HOME: join(userData, "model-shards"),
    PATH: [
      ...selected.manifest.environment.pathAdditions.map((entry) =>
        resolve(selected.root, entry)
      ),
      process.env.PATH ?? "",
    ].filter(Boolean).join(delimiter),
  },
});
const plannerScales = plannerScalesFromProfile(profile);

if (args.includes("--json")) {
  console.log(JSON.stringify(profile, null, 2));
} else {
  console.log("Physical runtime profile accepted and routable.");
  console.log(`  Pack: ${selected.packId}`);
  console.log(`  Backend: ${profile.backend}`);
  console.log(`  Device: ${profile.deviceName}`);
  console.log(`  Profile: ${profile.profileId}`);
  console.log(`  Decode memory p50: ${profile.decodeMemory.p50} ${profile.decodeMemory.unit}`);
  console.log(`  Prefill compute p50: ${profile.prefillCompute.p50} ${profile.prefillCompute.unit}`);
  console.log(`  Activation codec p50: ${profile.activationCodec.p50} ${profile.activationCodec.unit}`);
  console.log(`  Planner decode scale: ${plannerScales.decodeScale}`);
}

async function selectCachedRuntime(
  root: string,
  backend: string | undefined,
): Promise<{
  packId: string;
  root: string;
  manifest: CachedAcceleratorManifest;
}> {
  if (!existsSync(root)) {
    throw new Error(`No cached Mycellios accelerator runtimes found at ${root}`);
  }
  const candidates = await Promise.all((await readdir(root, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory())
    .map(async (entry) => {
      const runtimeRoot = join(root, entry.name);
      const manifestPath = join(runtimeRoot, "accelerator-runtime.json");
      if (!existsSync(manifestPath)) return null;
      const manifest = parseManifest(await readFile(manifestPath, "utf8"), manifestPath);
      const metadata = await stat(manifestPath);
      return {
        packId: entry.name,
        root: runtimeRoot,
        manifest,
        modifiedAt: metadata.mtimeMs,
      };
    }));
  const matches = candidates
    .filter((candidate) => candidate !== null)
    .filter((candidate) => !backend || candidate.manifest.backend === backend)
    .sort((left, right) => right.modifiedAt - left.modifiedAt);
  const selected = matches[0];
  if (!selected) {
    throw new Error(
      backend
        ? `No cached Mycellios ${backend} accelerator runtime was found.`
        : "No complete cached Mycellios accelerator runtime was found.",
    );
  }
  return selected;
}

function parseManifest(serialized: string, source: string): CachedAcceleratorManifest {
  const raw = JSON.parse(serialized) as Partial<CachedAcceleratorManifest>;
  if (
    raw.schema !== "mycellios-accelerator-runtime/1"
    || typeof raw.backend !== "string"
    || !raw.probe
    || raw.probe.backend !== raw.backend
    || typeof raw.probe.device !== "string"
    || typeof raw.probe.deviceName !== "string"
    || (raw.probe.precision !== "float16" && raw.probe.precision !== "float32")
    || !raw.environment
    || !Array.isArray(raw.environment.pythonPathAdditions)
    || !Array.isArray(raw.environment.pathAdditions)
  ) {
    throw new Error(`Cached accelerator manifest is invalid: ${source}`);
  }
  return raw as CachedAcceleratorManifest;
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

function option(name: string): string | undefined {
  const index = args.indexOf(name);
  if (index === -1) return undefined;
  const value = args[index + 1];
  if (!value || value.startsWith("--")) {
    throw new Error(`${name} requires a value.`);
  }
  return value;
}

function integerOption(name: string): number | undefined {
  const value = option(name);
  if (value === undefined) return undefined;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
  return parsed;
}
