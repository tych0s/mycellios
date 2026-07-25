import {
  copyFileSync,
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from "node:fs";
import { createHash } from "node:crypto";
import { dirname, join, relative, resolve, sep } from "node:path";

export const NATIVE_PYTHON_PRODUCT_SCHEMA =
  "mycellios-native-python-product/1";
export const NATIVE_PYTHON_PRODUCT_MANIFEST =
  "mycellios-native-python-manifest.json";

/**
 * These are the only executable Python surfaces shipped in a Mycellios
 * product. Everything else under python/distributed_runtime is laboratory or
 * source-only material and cannot enter a release by accident.
 */
export const NATIVE_PYTHON_ENTRY_MODULES = Object.freeze([
  "distributed_runtime.server",
  "distributed_runtime.stage_cli",
  "distributed_runtime.cell_member_cli",
  "distributed_runtime.native_gguf_cli",
  "distributed_runtime.profile",
  "distributed_runtime.physical_probe",
  "distributed_runtime.runtime_profile",
  "distributed_runtime.stage_artifact_cache",
  "distributed_runtime.installed_stage_canary",
]);

/**
 * Closed transitive source set for the entrypoints above. The closure checker
 * below fails whenever one of these modules starts importing another local
 * module that has not been deliberately admitted here.
 */
export const NATIVE_PYTHON_PRODUCT_FILES = Object.freeze([
  "distributed_runtime/__init__.py",
  "distributed_runtime/batching.py",
  "distributed_runtime/cell_backend.py",
  "distributed_runtime/cell_member_cli.py",
  "distributed_runtime/cell_parallel.py",
  "distributed_runtime/cell_stage.py",
  "distributed_runtime/decode_attention.py",
  "distributed_runtime/dense_tiering.py",
  "distributed_runtime/device.py",
  "distributed_runtime/engine.py",
  "distributed_runtime/executor_abi.py",
  "distributed_runtime/external_cell.py",
  "distributed_runtime/installed_stage_canary.py",
  "distributed_runtime/kv_arena.py",
  "distributed_runtime/macro_wave.py",
  "distributed_runtime/macro_wave_adapter.py",
  "distributed_runtime/model.py",
  "distributed_runtime/model_adapter_registry.json",
  "distributed_runtime/model_adapters.py",
  "distributed_runtime/native_gguf.py",
  "distributed_runtime/native_gguf_cli.py",
  "distributed_runtime/native_gguf_disk_tiering.py",
  "distributed_runtime/native_gguf_runtime.py",
  "distributed_runtime/paged_stage.py",
  "distributed_runtime/physical_probe.py",
  "distributed_runtime/physical_tree.py",
  "distributed_runtime/profile.py",
  "distributed_runtime/protocol.py",
  "distributed_runtime/ram_backed_moe_runtime.py",
  "distributed_runtime/ram_backed_moe_stage.py",
  "distributed_runtime/ram_expert_cache.py",
  "distributed_runtime/recovery.py",
  "distributed_runtime/resident_expert_mesh.py",
  "distributed_runtime/rocm_compat.py",
  "distributed_runtime/runtime_policy.py",
  "distributed_runtime/runtime_profile.py",
  "distributed_runtime/safetensors_moe_stage_loader.py",
  "distributed_runtime/server.py",
  "distributed_runtime/speculation.py",
  "distributed_runtime/stage.py",
  "distributed_runtime/stage_artifact.py",
  "distributed_runtime/stage_artifact_cache.py",
  "distributed_runtime/stage_cli.py",
  "distributed_runtime/torch_ram_expert_store.py",
]);

export const NATIVE_PYTHON_IMPORT_SMOKE_MODULES = Object.freeze([
  ...NATIVE_PYTHON_ENTRY_MODULES,
  "distributed_runtime.native_gguf",
  "distributed_runtime.native_gguf_runtime",
  "distributed_runtime.native_gguf_disk_tiering",
  "distributed_runtime.dense_tiering",
]);

const allowedModules = new Set(
  NATIVE_PYTHON_PRODUCT_FILES
    .filter((path) => path.endsWith(".py"))
    .map(pathToModule),
);
const policyId = `sha256:${sha256(Buffer.from(JSON.stringify({
  schema: NATIVE_PYTHON_PRODUCT_SCHEMA,
  entryModules: NATIVE_PYTHON_ENTRY_MODULES,
  files: NATIVE_PYTHON_PRODUCT_FILES,
})))}`;

export function prepareNativePythonProductSource(sourceRoot, destinationRoot) {
  const source = resolve(sourceRoot);
  const destination = resolve(destinationRoot);
  if (source === destination || source.startsWith(`${destination}${sep}`)) {
    throw new Error("Native Python source and destination must be separate.");
  }
  assertNativePythonSourceClosure(source);
  mkdirSync(destination, { recursive: true });
  for (const portable of NATIVE_PYTHON_PRODUCT_FILES) {
    const input = resolveInside(source, portable);
    const output = resolveInside(destination, portable);
    mkdirSync(dirname(output), { recursive: true });
    copyFileSync(input, output);
  }
  const manifest = buildNativePythonProductManifest(destination);
  writeFileSync(
    join(destination, NATIVE_PYTHON_PRODUCT_MANIFEST),
    `${JSON.stringify(manifest, null, 2)}\n`,
    "utf8",
  );
  verifyNativePythonProductSource(destination);
  return manifest;
}

export function assertNativePythonSourceClosure(sourceRoot) {
  const root = resolve(sourceRoot);
  const graph = new Map();
  for (const portable of NATIVE_PYTHON_PRODUCT_FILES) {
    const absolute = resolveInside(root, portable);
    if (!existsSync(absolute) || !lstatSync(absolute).isFile()) {
      throw new Error(`Native Python product source is missing: ${portable}.`);
    }
    if (!portable.endsWith(".py")) continue;
    const module = pathToModule(portable);
    const dependencies = internalImports(readFileSync(absolute, "utf8"));
    for (const dependency of dependencies) {
      if (!allowedModules.has(dependency)) {
        throw new Error(
          `Native Python import closure escaped the allowlist: ${module} -> ${dependency}.`,
        );
      }
    }
    graph.set(module, dependencies);
  }
  const reachable = new Set();
  const pending = [...NATIVE_PYTHON_ENTRY_MODULES, "distributed_runtime.__init__"];
  while (pending.length > 0) {
    const module = pending.pop();
    if (reachable.has(module)) continue;
    reachable.add(module);
    for (const dependency of graph.get(module) ?? []) {
      if (!reachable.has(dependency)) pending.push(dependency);
    }
  }
  for (const module of allowedModules) {
    if (!reachable.has(module)) {
      throw new Error(
        `Native Python allowlist contains unreachable product module: ${module}.`,
      );
    }
  }
}

export function verifyNativePythonProductSource(sourceRoot) {
  const root = resolve(sourceRoot);
  const manifestPath = join(root, NATIVE_PYTHON_PRODUCT_MANIFEST);
  if (!existsSync(manifestPath)) {
    throw new Error(`Native Python manifest is missing: ${manifestPath}.`);
  }
  let manifest;
  try {
    manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  } catch {
    throw new Error(`Native Python manifest is invalid: ${manifestPath}.`);
  }
  const expectedPaths = [...NATIVE_PYTHON_PRODUCT_FILES];
  if (
    manifest?.schema !== NATIVE_PYTHON_PRODUCT_SCHEMA
    || manifest?.policyId !== policyId
    || JSON.stringify(manifest?.entryModules) !== JSON.stringify(NATIVE_PYTHON_ENTRY_MODULES)
    || !Array.isArray(manifest?.files)
    || JSON.stringify(manifest.files.map((entry) => entry?.path)) !== JSON.stringify(expectedPaths)
  ) {
    throw new Error("Native Python manifest does not match the product allowlist.");
  }
  const actualFiles = listRegularFiles(root);
  const expectedFiles = new Set([
    ...NATIVE_PYTHON_PRODUCT_FILES,
    NATIVE_PYTHON_PRODUCT_MANIFEST,
  ]);
  if (
    actualFiles.length !== expectedFiles.size
    || actualFiles.some((path) => !expectedFiles.has(path))
  ) {
    throw new Error(
      `Native Python package contains files outside the allowlist: ${actualFiles
        .filter((path) => !expectedFiles.has(path))
        .join(", ") || "file count mismatch"}.`,
    );
  }
  for (const entry of manifest.files) {
    const absolute = resolveInside(root, entry.path);
    const bytes = readFileSync(absolute);
    if (
      !Number.isInteger(entry.bytes)
      || entry.bytes !== bytes.byteLength
      || entry.sha256 !== sha256(bytes)
    ) {
      throw new Error(`Native Python package digest mismatch: ${entry.path}.`);
    }
  }
  assertNativePythonSourceClosure(root);
  return manifest;
}

export function buildNativePythonProductManifest(sourceRoot) {
  const root = resolve(sourceRoot);
  return {
    schema: NATIVE_PYTHON_PRODUCT_SCHEMA,
    policyId,
    entryModules: [...NATIVE_PYTHON_ENTRY_MODULES],
    files: NATIVE_PYTHON_PRODUCT_FILES.map((portable) => {
      const bytes = readFileSync(resolveInside(root, portable));
      return {
        path: portable,
        bytes: bytes.byteLength,
        sha256: sha256(bytes),
      };
    }),
  };
}

function internalImports(source) {
  const imports = new Set();
  const patterns = [
    /^\s*from\s+\.(?!\.)([A-Za-z_][A-Za-z0-9_]*)/gm,
    /^\s*from\s+distributed_runtime\.([A-Za-z_][A-Za-z0-9_]*)/gm,
    /^\s*import\s+distributed_runtime\.([A-Za-z_][A-Za-z0-9_]*)/gm,
    /(?:import_module|run_module)\(\s*["']distributed_runtime\.([A-Za-z_][A-Za-z0-9_]*)/g,
  ];
  for (const pattern of patterns) {
    let match;
    while ((match = pattern.exec(source)) !== null) {
      imports.add(`distributed_runtime.${match[1]}`);
    }
  }
  return [...imports].sort();
}

function listRegularFiles(root) {
  const files = [];
  const visit = (directory) => {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const absolute = join(directory, entry.name);
      if (entry.isSymbolicLink()) {
        throw new Error(`Native Python package contains a symbolic link: ${absolute}.`);
      }
      if (entry.isDirectory()) {
        visit(absolute);
      } else if (entry.isFile()) {
        files.push(relative(root, absolute).replaceAll("\\", "/"));
      } else {
        throw new Error(`Native Python package contains a special file: ${absolute}.`);
      }
    }
  };
  visit(root);
  return files.sort();
}

function pathToModule(path) {
  return path
    .replace(/\.py$/, "")
    .replaceAll("/", ".");
}

function resolveInside(root, portable) {
  const absolute = resolve(root, ...portable.split("/"));
  if (absolute !== root && !absolute.startsWith(`${root}${sep}`)) {
    throw new Error(`Native Python path escaped its root: ${portable}.`);
  }
  return absolute;
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
