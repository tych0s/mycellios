import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { join, posix, resolve } from "node:path";
import { tmpdir } from "node:os";
import { spawnSync } from "node:child_process";
import {
  PORTABLE_PYTHON_PROVENANCE_FILE,
  PORTABLE_PYTHON_PROVENANCE_SCHEMA,
  PORTABLE_RUNTIME_SCHEMA,
  matchesPinnedPythonArtifact,
  normalizePythonMachine,
  portableRuntimeSpec,
} from "./portable-runtime-policy.mjs";
import {
  assertNoEscapingSymlinks,
  samePath,
} from "./portable-runtime-filesystem.mjs";
import { runInstalledStageCanary } from "./installed-stage-canary.mjs";

const platform = readArgument("platform") ?? process.platform;
const arch = readArgument("arch") ?? process.arch;
const spec = portableRuntimeSpec(platform, arch);
if (!spec.supported) throw new Error(spec.reason);

const workspace = resolve(import.meta.dirname, "..");
const archive = resolve(readArgument("archive") ?? join(workspace, "build", "distribution-runtime.tar.gz"));
const pythonSource = resolve(
  readArgument("python-source") ?? join(workspace, "build", "python"),
);
if (!existsSync(archive)) throw new Error(`Portable runtime archive is missing: ${archive}.`);

const entries = tar(["-tzf", archive])
  .split(/\r?\n/)
  .map((entry) => entry.trim())
  .filter(Boolean);
for (const entry of entries) {
  const portableEntry = entry.replaceAll("\\", "/");
  const normalized = posix.normalize(portableEntry.replace(/^\.\//, ""));
  if (
    portableEntry.startsWith("/") ||
    /^[A-Za-z]:/.test(portableEntry) ||
    normalized === ".." ||
    normalized.startsWith("../")
  ) {
    throw new Error(`Portable runtime archive contains an unsafe path: ${entry}.`);
  }
}

const manifest = JSON.parse(tarEntry(archive, ["./runtime-manifest.json", "runtime-manifest.json"]));
if (
  manifest.schema !== PORTABLE_RUNTIME_SCHEMA ||
  manifest.platform !== platform ||
  manifest.arch !== arch ||
  manifest.pythonVersion !== spec.pythonVersion ||
  manifest.pythonAbi !== "cp312" ||
  manifest.executable !== spec.pythonExecutable ||
  !matchesPinnedPythonArtifact(manifest.pythonArtifact, spec.pythonArtifact) ||
  manifest.torchVersion !== spec.torchVersion ||
  manifest.transformersVersion !== spec.packageVersions.transformers ||
  manifest.accelerateVersion !== spec.packageVersions.accelerate ||
  manifest.safetensorsVersion !== spec.packageVersions.safetensors ||
  manifest.aiohttpVersion !== spec.packageVersions.aiohttp ||
  manifest.sentencepieceVersion !== spec.packageVersions.sentencepiece ||
  manifest.numpyVersion !== spec.packageVersions.numpy ||
  manifest.backend !== "cpu" ||
  JSON.stringify(manifest.bundledAccelerators ?? []) !== JSON.stringify(spec.bundledAccelerators)
) {
  throw new Error(`Portable runtime manifest does not match ${platform}/${arch}: ${JSON.stringify(manifest)}.`);
}
const provenance = JSON.parse(tarEntry(archive, [
  `./${PORTABLE_PYTHON_PROVENANCE_FILE}`,
  PORTABLE_PYTHON_PROVENANCE_FILE,
]));
if (
  provenance.schema !== PORTABLE_PYTHON_PROVENANCE_SCHEMA ||
  provenance.platform !== platform ||
  provenance.arch !== arch ||
  !matchesPinnedPythonArtifact(provenance.artifact, spec.pythonArtifact)
) {
  throw new Error(`Portable runtime Python provenance does not match ${platform}/${arch}: ${JSON.stringify(provenance)}.`);
}

const temporary = mkdtempSync(join(tmpdir(), "mycellios-portable-runtime-"));
try {
  tar(["-xzf", archive, "-C", temporary]);
  assertNoEscapingSymlinks(temporary);
  const python = join(temporary, ...spec.pythonExecutable.split("/"));
  if (!existsSync(python)) throw new Error(`Relocated Python is missing: ${python}.`);
  const extractedProvenance = JSON.parse(
    readFileSync(join(temporary, PORTABLE_PYTHON_PROVENANCE_FILE), "utf8"),
  );
  if (!matchesPinnedPythonArtifact(extractedProvenance.artifact, spec.pythonArtifact)) {
    throw new Error("Relocated Python provenance drifted after extraction.");
  }
  const probe = spawnSync(python, ["-c", [
    "import accelerate,aiohttp,json,numpy,platform,safetensors,sentencepiece,struct,sys,torch,transformers",
    "mps=getattr(getattr(torch,'backends',None),'mps',None)",
    "print(json.dumps({'pythonVersion':'.'.join(map(str,sys.version_info[:3])),'machine':platform.machine(),'bits':struct.calcsize('P')*8,'prefix':sys.prefix,'basePrefix':sys.base_prefix,'torchVersion':torch.__version__,'cudaVersion':torch.version.cuda,'hipVersion':getattr(torch.version,'hip',None),'cudaAvailable':bool(torch.cuda.is_available()),'mpsBuilt':bool(mps and mps.is_built()),'packages':{'numpy':numpy.__version__,'aiohttp':aiohttp.__version__,'accelerate':accelerate.__version__,'transformers':transformers.__version__,'safetensors':safetensors.__version__,'sentencepiece':sentencepiece.__version__}},sort_keys=True))",
  ].join("; ")], {
    encoding: "utf8",
    env: { ...process.env, PYTHONNOUSERSITE: "1" },
    maxBuffer: 16 * 1024 * 1024,
    shell: false,
    windowsHide: true,
  });
  if (probe.error) throw probe.error;
  if (probe.status !== 0) {
    throw new Error(probe.stderr?.trim() || `Relocated Python exited with ${probe.status ?? "no status"}.`);
  }
  const runtime = JSON.parse(probe.stdout.trim());
  if (
    runtime.pythonVersion !== spec.pythonVersion ||
    runtime.bits !== 64 ||
    normalizePythonMachine(runtime.machine) !== spec.arch ||
    !samePath(runtime.prefix, temporary) ||
    !samePath(runtime.basePrefix, temporary) ||
    runtime.torchVersion !== spec.torchVersion ||
    runtime.cudaVersion !== null ||
    runtime.hipVersion !== null ||
    runtime.cudaAvailable !== false
  ) {
    throw new Error(`Relocated runtime failed its backend verification: ${JSON.stringify(runtime)}.`);
  }
  if (spec.bundledAccelerators.includes("mps") && runtime.mpsBuilt !== true) {
    throw new Error("Relocated macOS arm64 runtime does not contain the MPS backend.");
  }
  for (const [name, version] of Object.entries(spec.packageVersions)) {
    if (runtime.packages?.[name] !== version) {
      throw new Error(`Relocated runtime has ${name} ${runtime.packages?.[name] ?? "missing"}; expected ${version}.`);
    }
  }
  const canary = runInstalledStageCanary({
    pythonExecutable: python,
    pythonSourceRoot: pythonSource,
    expectedPythonPrefix: temporary,
    expectedTorchVersion: spec.torchVersion,
    expectedTransformersVersion: spec.packageVersions.transformers,
  });
  process.stdout.write(
    `Installed stage canary verified: ${canary.engine}/${canary.adapter}, ` +
      `${canary.kvBytes} KV bytes, batch ${canary.batchSize}.\n`,
  );
} finally {
  rmSync(temporary, { recursive: true, force: true });
}

process.stdout.write(`Relocated portable runtime verified: ${platform}/${arch}, torch ${spec.torchVersion}.\n`);

function readArgument(name) {
  const prefix = `--${name}=`;
  const value = process.argv.slice(2).find((argument) => argument.startsWith(prefix));
  return value?.slice(prefix.length).trim() || undefined;
}

function tar(args) {
  const result = spawnSync("tar", args, {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
    shell: false,
    windowsHide: true,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(result.stderr?.trim() || `tar exited with ${result.status ?? "no status"}.`);
  }
  return result.stdout;
}

function tarEntry(path, candidates) {
  for (const candidate of candidates) {
    const result = spawnSync("tar", ["-xOf", path, candidate], {
      encoding: "utf8",
      maxBuffer: 4 * 1024 * 1024,
      shell: false,
      windowsHide: true,
    });
    if (result.status === 0 && result.stdout.trim()) return result.stdout;
  }
  throw new Error(`Portable runtime ${path} has no runtime-manifest.json.`);
}
