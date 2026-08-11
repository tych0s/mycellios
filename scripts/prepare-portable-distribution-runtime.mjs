import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
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
  normalizeCopiedInternalAbsoluteSymlinks,
  samePath,
} from "./portable-runtime-filesystem.mjs";
import { readPortableRuntimeWheelLock } from "./portable-runtime-wheel-lock.mjs";

const workspace = resolve(import.meta.dirname, "..");
const spec = portableRuntimeSpec(process.platform, process.arch);
const standalone = resolve(workspace, "runtime", "distribution-venv");
const output = resolve(workspace, "build", "distribution-runtime");
const archive = resolve(workspace, "build", "distribution-runtime.tar.gz");
const packagedPythonSource = resolve(workspace, "build", "python");
const resolvedBuild = resolve(workspace, "build");

if (!spec.supported) {
  rmSync(output, { recursive: true, force: true });
  rmSync(archive, { force: true });
  process.stdout.write(`${spec.reason}\n`);
  process.exit(0);
}
const wheelLock = readPortableRuntimeWheelLock(workspace, spec);
if (dirname(standalone) !== resolve(workspace, "runtime")) {
  throw new Error("Standalone Python escaped the managed runtime directory.");
}
if (dirname(output) !== resolvedBuild || dirname(archive) !== resolvedBuild) {
  throw new Error("Portable runtime output escaped the build directory.");
}

const standalonePython = join(standalone, ...spec.pythonExecutable.split("/"));
if (!existsSync(standalonePython)) {
  throw new Error("Standalone distribution Python is missing; run npm run node:runtime:setup first.");
}
const provenancePath = join(standalone, PORTABLE_PYTHON_PROVENANCE_FILE);
if (!existsSync(provenancePath)) {
  throw new Error(`Standalone Python has no pinned provenance file: ${provenancePath}.`);
}
const provenance = JSON.parse(readFileSync(provenancePath, "utf8"));
if (
  provenance.schema !== PORTABLE_PYTHON_PROVENANCE_SCHEMA ||
  provenance.platform !== spec.platform ||
  provenance.arch !== spec.arch ||
  !matchesPinnedPythonArtifact(provenance.artifact, spec.pythonArtifact)
) {
  throw new Error(`Standalone Python provenance does not match policy: ${JSON.stringify(provenance)}.`);
}
assertNoEscapingSymlinks(standalone);

const sourceRuntime = inspectRuntime(standalonePython);
assertCertifiedRuntime(sourceRuntime, standalone, "Installed standalone runtime");

rmSync(output, { recursive: true, force: true });
mkdirSync(resolvedBuild, { recursive: true });
cpSync(standalone, output, { recursive: true, verbatimSymlinks: true });
normalizeCopiedInternalAbsoluteSymlinks(output, standalone);
assertNoEscapingSymlinks(output);

// CPython's Unix distribution carries a terminal database with entries that
// differ only by case (for example 2621a/2621A). The headless inference runtime
// does not consume terminfo, and retaining it would make the sealed installer
// impossible to verify or move safely across case-insensitive filesystems.
rmSync(join(output, "share", "terminfo"), { recursive: true, force: true });

const packagedPython = join(output, ...spec.pythonExecutable.split("/"));
if (!existsSync(packagedPython)) {
  throw new Error(`Portable Python executable was not copied to ${packagedPython}.`);
}
const runtime = inspectRuntime(packagedPython);
assertCertifiedRuntime(runtime, output, "Relocated portable runtime");

const manifest = {
  schema: PORTABLE_RUNTIME_SCHEMA,
  platform: spec.platform,
  arch: spec.arch,
  pythonVersion: runtime.pythonVersion,
  pythonAbi: runtime.pythonAbi,
  executable: spec.pythonExecutable,
  pythonArtifact: { ...spec.pythonArtifact },
  wheelLock: {
    path: wheelLock.path,
    sha256: wheelLock.sha256,
  },
  torchVersion: runtime.torchVersion,
  transformersVersion: runtime.transformersVersion,
  accelerateVersion: runtime.accelerateVersion,
  safetensorsVersion: runtime.safetensorsVersion,
  aiohttpVersion: runtime.aiohttpVersion,
  sentencepieceVersion: runtime.sentencepieceVersion,
  numpyVersion: runtime.numpyVersion,
  backend: "cpu",
  bundledAccelerators: [...spec.bundledAccelerators],
};
writeFileSync(
  join(output, "runtime-manifest.json"),
  `${JSON.stringify(manifest, null, 2)}\n`,
  "utf8",
);

rmSync(archive, { force: true });
const packed = spawnSync("tar", ["-czf", archive, "-C", output, "."], {
  encoding: "utf8",
  shell: false,
  windowsHide: true,
});
if (packed.error) throw packed.error;
if (packed.status !== 0 || !existsSync(archive)) {
  throw new Error(packed.stderr?.trim() || "Could not archive the portable distribution runtime.");
}
const verified = spawnSync(process.execPath, [
  join(import.meta.dirname, "verify-portable-runtime-archive.mjs"),
  `--archive=${archive}`,
  `--platform=${spec.platform}`,
  `--arch=${spec.arch}`,
  `--python-source=${packagedPythonSource}`,
], { stdio: "inherit", shell: false, windowsHide: true });
if (verified.error) throw verified.error;
if (verified.status !== 0) {
  throw new Error("The relocated portable runtime archive did not pass verification.");
}
process.stdout.write(
  `Portable distribution runtime v4 ready: Python ${manifest.pythonVersion}, ${manifest.torchVersion} ` +
  `(${manifest.platform}/${manifest.arch}${manifest.bundledAccelerators.length ? `, ${manifest.bundledAccelerators.join(",")}` : ""})\n`,
);

function inspectRuntime(python) {
  const script = [
    "import accelerate,aiohttp,json,numpy,platform,safetensors,sentencepiece,struct,sys,torch,transformers",
    "mps=getattr(getattr(torch,'backends',None),'mps',None)",
    "print(json.dumps({'pythonVersion':'.'.join(map(str,sys.version_info[:3])),'pythonAbi':f'cp{sys.version_info[0]}{sys.version_info[1]}','machine':platform.machine(),'bits':struct.calcsize('P')*8,'prefix':sys.prefix,'basePrefix':sys.base_prefix,'torchVersion':torch.__version__,'transformersVersion':transformers.__version__,'accelerateVersion':accelerate.__version__,'safetensorsVersion':safetensors.__version__,'aiohttpVersion':aiohttp.__version__,'sentencepieceVersion':sentencepiece.__version__,'numpyVersion':numpy.__version__,'cudaVersion':torch.version.cuda,'hipVersion':getattr(torch.version,'hip',None),'cudaAvailable':bool(torch.cuda.is_available()),'mpsBuilt':bool(mps and mps.is_built())},sort_keys=True))",
  ].join("; ");
  const result = spawnSync(python, ["-I", "-c", script], {
    encoding: "utf8",
    env: { ...process.env, PYTHONNOUSERSITE: "1" },
    maxBuffer: 16 * 1024 * 1024,
    shell: false,
    windowsHide: true,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(result.stderr?.trim() || "Portable distribution runtime verification failed.");
  }
  return JSON.parse(result.stdout.trim());
}

function assertCertifiedRuntime(runtime, expectedRoot, label) {
  if (
    runtime.pythonVersion !== spec.pythonVersion ||
    runtime.pythonAbi !== "cp312" ||
    runtime.bits !== 64 ||
    normalizePythonMachine(runtime.machine) !== spec.arch ||
    !samePath(runtime.prefix, expectedRoot) ||
    !samePath(runtime.basePrefix, expectedRoot) ||
    runtime.torchVersion !== spec.torchVersion ||
    runtime.transformersVersion !== spec.packageVersions.transformers ||
    runtime.accelerateVersion !== spec.packageVersions.accelerate ||
    runtime.safetensorsVersion !== spec.packageVersions.safetensors ||
    runtime.aiohttpVersion !== spec.packageVersions.aiohttp ||
    runtime.sentencepieceVersion !== spec.packageVersions.sentencepiece ||
    runtime.numpyVersion !== spec.packageVersions.numpy ||
    runtime.cudaVersion !== null ||
    runtime.hipVersion !== null ||
    runtime.cudaAvailable !== false
  ) {
    throw new Error(`${label} is not the certified standalone bootstrap: ${JSON.stringify(runtime)}.`);
  }
  if (spec.bundledAccelerators.includes("mps") && runtime.mpsBuilt !== true) {
    throw new Error(`${label} does not contain an MPS-enabled PyTorch build.`);
  }
}
