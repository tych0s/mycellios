import { createHash } from "node:crypto";
import {
  createReadStream,
  createWriteStream,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readlinkSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, posix, relative, resolve } from "node:path";
import { tmpdir } from "node:os";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { spawnSync } from "node:child_process";
import {
  PORTABLE_PYTHON_PROVENANCE_FILE,
  PORTABLE_PYTHON_PROVENANCE_SCHEMA,
  matchesPinnedPythonArtifact,
  normalizePythonMachine,
  portableRuntimeSpec,
} from "./portable-runtime-policy.mjs";

const workspace = resolve(import.meta.dirname, "..");
const runtimeDirectory = resolve(workspace, "runtime");
const standalone = resolve(runtimeDirectory, "distribution-venv");
const buildDirectory = resolve(workspace, "build");
const portableOutput = resolve(buildDirectory, "distribution-runtime");
const portableArchive = resolve(buildDirectory, "distribution-runtime.tar.gz");
const spec = portableRuntimeSpec(process.platform, process.arch);

assertDirectChild(runtimeDirectory, standalone, "standalone Python runtime");
assertDirectChild(buildDirectory, portableOutput, "portable runtime output");
if (dirname(portableArchive) !== buildDirectory) {
  throw new Error("Portable runtime archive escaped the build directory.");
}

if (!spec.supported) {
  rmSync(portableOutput, { recursive: true, force: true });
  rmSync(portableArchive, { force: true });
  process.stdout.write(`${spec.reason}\n`);
  process.exit(0);
}

const suppliedArchive = readArgument("python-archive");
const verifyArtifactOnly = process.argv.includes("--verify-python-artifact-only");
const temporary = mkdtempSync(join(tmpdir(), "mycellios-python-standalone-"));

try {
  const artifactArchive = suppliedArchive
    ? resolve(suppliedArchive)
    : await downloadArtifact(spec.pythonArtifact, join(temporary, "download"));
  await verifyArtifactFile(artifactArchive, spec.pythonArtifact);

  const extracted = join(temporary, "python");
  extractPinnedPython(artifactArchive, temporary, extracted);
  const extractedPython = join(extracted, ...spec.pythonExecutable.split("/"));
  const python = verifyStandalonePython(extractedPython, extracted);

  if (verifyArtifactOnly) {
    process.stdout.write(
      `Pinned Python artifact verified: CPython ${python.version}, ${spec.platform}/${spec.arch}, ` +
      `${spec.pythonArtifact.sha256}.\n`,
    );
    process.exitCode = 0;
  } else {
    mkdirSync(runtimeDirectory, { recursive: true });
    rmSync(standalone, { recursive: true, force: true });
    renameSync(extracted, standalone);
    writeProvenance(standalone, spec);

    const standalonePython = join(standalone, ...spec.pythonExecutable.split("/"));
    installPackages(standalonePython);
    const verification = verifyInstalledRuntime(standalonePython);
    process.stdout.write(
      `Platform runtime ready: Python ${verification.pythonVersion}, ${verification.torchVersion} ` +
      `(${spec.platform}/${spec.arch}${spec.bundledAccelerators.length ? `, ${spec.bundledAccelerators.join(",")}` : ""}).\n`,
    );
  }
} finally {
  rmSync(temporary, { recursive: true, force: true });
}

async function downloadArtifact(artifact, directory) {
  mkdirSync(directory, { recursive: true });
  const destination = join(directory, artifact.filename);
  const response = await fetch(artifact.url, {
    redirect: "follow",
    headers: { "user-agent": "mycellios-desktop-runtime-builder" },
  });
  if (!response.ok || !response.body) {
    throw new Error(`Could not download ${artifact.url}: HTTP ${response.status}.`);
  }
  const declaredLength = Number(response.headers.get("content-length"));
  if (Number.isFinite(declaredLength) && declaredLength !== artifact.size) {
    throw new Error(
      `Python artifact server declared ${declaredLength} bytes; expected ${artifact.size}.`,
    );
  }
  const source = Readable.fromWeb(response.body);
  let received = 0;
  source.on("data", (chunk) => {
    received += chunk.length;
    if (received > artifact.size) {
      source.destroy(new Error(`Python artifact exceeded its pinned size of ${artifact.size} bytes.`));
    }
  });
  await pipeline(source, createWriteStream(destination, { flags: "wx" }));
  if (received !== artifact.size) {
    throw new Error(`Downloaded Python artifact has ${received} bytes; expected ${artifact.size}.`);
  }
  return destination;
}

async function verifyArtifactFile(path, artifact) {
  if (!existsSync(path)) throw new Error(`Pinned Python artifact is missing: ${path}.`);
  const size = statSync(path).size;
  if (size !== artifact.size) {
    throw new Error(`Python artifact has ${size} bytes; expected ${artifact.size}: ${path}.`);
  }
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  const digest = hash.digest("hex");
  if (digest !== artifact.sha256) {
    throw new Error(`Python artifact SHA-256 ${digest} does not match ${artifact.sha256}: ${path}.`);
  }
}

function extractPinnedPython(archive, directory, extracted) {
  const entries = tar(["-tzf", archive])
    .split(/\r?\n/)
    .map((entry) => entry.trim())
    .filter(Boolean);
  if (entries.length === 0) throw new Error(`Pinned Python artifact is empty: ${archive}.`);
  for (const entry of entries) {
    if (entry.includes("\\")) {
      throw new Error(`Pinned Python artifact contains a non-portable path: ${entry}.`);
    }
    const normalized = posix.normalize(entry.replace(/^\.\//, "").replace(/\/$/, ""));
    if (
      entry.startsWith("/") ||
      /^[A-Za-z]:/.test(entry) ||
      normalized === ".." ||
      normalized.startsWith("../") ||
      (normalized !== "python" && !normalized.startsWith("python/"))
    ) {
      throw new Error(`Pinned Python artifact contains an unsafe path: ${entry}.`);
    }
  }
  tar(["-xzf", archive, "-C", directory]);
  if (!existsSync(extracted)) {
    throw new Error(`Pinned Python artifact did not contain its required python/ directory: ${archive}.`);
  }
  assertSafeTree(extracted);
}

function verifyStandalonePython(executable, root) {
  if (!existsSync(executable)) {
    throw new Error(`Standalone Python executable is missing: ${executable}.`);
  }
  const result = runJson(executable, ["-I", "-c", [
    "import json,platform,struct,sys",
    "print(json.dumps({'version':'.'.join(map(str,sys.version_info[:3])),'machine':platform.machine(),'bits':struct.calcsize('P')*8,'prefix':sys.prefix,'basePrefix':sys.base_prefix}))",
  ].join("; ")]);
  if (
    result.version !== spec.pythonVersion ||
    result.bits !== 64 ||
    normalizePythonMachine(result.machine) !== spec.arch ||
    !samePath(result.prefix, root) ||
    !samePath(result.basePrefix, root)
  ) {
    throw new Error(`Standalone Python failed its pinned platform verification: ${JSON.stringify(result)}.`);
  }
  return result;
}

function writeProvenance(root, runtimeSpec) {
  const provenance = {
    schema: PORTABLE_PYTHON_PROVENANCE_SCHEMA,
    platform: runtimeSpec.platform,
    arch: runtimeSpec.arch,
    artifact: { ...runtimeSpec.pythonArtifact },
  };
  if (!matchesPinnedPythonArtifact(provenance.artifact, runtimeSpec.pythonArtifact)) {
    throw new Error("Internal Python artifact policy drifted before provenance was written.");
  }
  writeFileSync(
    join(root, PORTABLE_PYTHON_PROVENANCE_FILE),
    `${JSON.stringify(provenance, null, 2)}\n`,
    "utf8",
  );
}

function installPackages(python) {
  const pipEnvironment = {
    ...process.env,
    PIP_DISABLE_PIP_VERSION_CHECK: "1",
    PIP_NO_INPUT: "1",
    PYTHONNOUSERSITE: "1",
  };
  run(python, [
    "-m", "pip", "install",
    "--isolated",
    "--disable-pip-version-check",
    "--no-input",
    "--only-binary=:all:",
    "--index-url", spec.torchIndex,
    spec.torchRequirement,
  ], pipEnvironment);
  run(python, [
    "-m", "pip", "install",
    "--isolated",
    "--disable-pip-version-check",
    "--no-input",
    "--only-binary=:all:",
    "--index-url", "https://pypi.org/simple",
    ...Object.entries(spec.packageVersions).map(([name, version]) => `${name}==${version}`),
  ], pipEnvironment);
  run(python, ["-m", "pip", "check"], pipEnvironment);
}

function verifyInstalledRuntime(python) {
  const verification = runJson(python, ["-I", "-c", [
    "import importlib.metadata,json,platform,struct,sys,torch",
    "mps=getattr(getattr(torch,'backends',None),'mps',None)",
    "payload={'pythonVersion':'.'.join(map(str,sys.version_info[:3])),'machine':platform.machine(),'bits':struct.calcsize('P')*8,'torchVersion':torch.__version__,'cudaVersion':torch.version.cuda,'hipVersion':getattr(torch.version,'hip',None),'cudaAvailable':bool(torch.cuda.is_available()),'mpsBuilt':bool(mps and mps.is_built()),'packages':{name:importlib.metadata.version(name) for name in ('numpy','aiohttp','accelerate','transformers','safetensors','sentencepiece')}}",
    "print(json.dumps(payload,sort_keys=True))",
  ].join("; ")], { ...process.env, PYTHONNOUSERSITE: "1" });

  if (
    verification.pythonVersion !== spec.pythonVersion ||
    verification.bits !== 64 ||
    normalizePythonMachine(verification.machine) !== spec.arch ||
    verification.torchVersion !== spec.torchVersion ||
    verification.cudaVersion !== null ||
    verification.hipVersion !== null ||
    verification.cudaAvailable !== false
  ) {
    throw new Error(`Installed runtime failed its platform/backend verification: ${JSON.stringify(verification)}.`);
  }
  if (spec.bundledAccelerators.includes("mps") && verification.mpsBuilt !== true) {
    throw new Error("The macOS arm64 PyTorch wheel was not built with the required MPS backend.");
  }
  for (const [name, version] of Object.entries(spec.packageVersions)) {
    if (verification.packages?.[name] !== version) {
      throw new Error(`Installed ${name} ${verification.packages?.[name] ?? "missing"}; expected ${version}.`);
    }
  }
  return verification;
}

function readArgument(name) {
  const prefix = `--${name}=`;
  const value = process.argv.slice(2).find((argument) => argument.startsWith(prefix));
  return value?.slice(prefix.length).trim() || undefined;
}

function run(executable, args, env = process.env) {
  const result = spawnSync(executable, args, {
    cwd: workspace,
    env,
    stdio: "inherit",
    shell: false,
    windowsHide: true,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${executable} exited with ${result.status ?? "no status"}.`);
  }
}

function runJson(executable, args, env = process.env) {
  const result = spawnSync(executable, args, {
    cwd: workspace,
    env,
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
    shell: false,
    windowsHide: true,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(result.stderr?.trim() || `${executable} exited with ${result.status ?? "no status"}.`);
  }
  try {
    return JSON.parse(result.stdout.trim());
  } catch {
    throw new Error(`Could not parse the Python verification output: ${result.stdout.trim()}.`);
  }
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

function assertSafeTree(root) {
  const visit = (directory) => {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isSymbolicLink()) {
        const target = readlinkSync(path);
        const resolvedTarget = resolve(dirname(path), target);
        const back = relative(root, resolvedTarget);
        if (back === ".." || back.startsWith(`..${process.platform === "win32" ? "\\" : "/"}`)) {
          throw new Error(`Standalone Python contains an escaping symlink: ${relative(root, path)} -> ${target}`);
        }
      } else if (entry.isDirectory()) {
        visit(path);
      } else if (!entry.isFile()) {
        throw new Error(`Standalone Python contains an unsupported filesystem entry: ${relative(root, path)}.`);
      }
    }
  };
  visit(root);
}

function samePath(left, right) {
  const normalize = (value) => {
    const normalized = resolve(String(value)).replaceAll("\\", "/");
    return process.platform === "win32" ? normalized.toLowerCase() : normalized;
  };
  return normalize(left) === normalize(right);
}

function assertDirectChild(parent, child, label) {
  if (dirname(child) !== parent) throw new Error(`${label} escaped its managed directory.`);
}
