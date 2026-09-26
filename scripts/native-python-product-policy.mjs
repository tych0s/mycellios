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
import { spawnSync } from "node:child_process";
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
  "distributed_runtime.engine_runtime_profile",
  "distributed_runtime.activation_integrity",
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
  "distributed_runtime/activation_integrity.py",
  "distributed_runtime/batching.py",
  "distributed_runtime/cell_backend.py",
  "distributed_runtime/cell_member_cli.py",
  "distributed_runtime/cell_parallel.py",
  "distributed_runtime/cell_stage.py",
  "distributed_runtime/checkpoint_control.py",
  "distributed_runtime/compute_timing.py",
  "distributed_runtime/decode_attention.py",
  "distributed_runtime/dense_tiering.py",
  "distributed_runtime/device.py",
  "distributed_runtime/draft_model.py",
  "distributed_runtime/engine.py",
  "distributed_runtime/engine_runtime_profile.py",
  "distributed_runtime/executor_abi.py",
  "distributed_runtime/external_cell.py",
  "distributed_runtime/failure_evidence.py",
  "distributed_runtime/installed_stage_canary.py",
  "distributed_runtime/kv_arena.py",
  "distributed_runtime/lossless_sampling.py",
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
  "distributed_runtime/recovery_outcome.py",
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
  "distributed_runtime.draft_model",
  "distributed_runtime.lossless_sampling",
]);

const allowedModules = new Set(
  NATIVE_PYTHON_PRODUCT_FILES
    .filter((path) => path.endsWith(".py"))
    .map(pathToModule),
);
let cachedUvPython;
export const NATIVE_PYTHON_PRODUCT_POLICY_ID = `sha256:${sha256(Buffer.from(JSON.stringify({
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
  const pythonSources = [];
  for (const portable of NATIVE_PYTHON_PRODUCT_FILES) {
    const absolute = resolveInside(root, portable);
    if (!existsSync(absolute) || !lstatSync(absolute).isFile()) {
      throw new Error(`Native Python product source is missing: ${portable}.`);
    }
    if (!portable.endsWith(".py")) continue;
    pythonSources.push({
      path: portable,
      module: pathToModule(portable),
      source: readFileSync(absolute, "utf8"),
    });
  }
  const analyzedSources = analyzeNativePythonImportSources(
    pythonSources.map(({ path, source }) => ({ path, source })),
  );
  for (const [index, analyzed] of analyzedSources.entries()) {
    const module = pythonSources[index].module;
    const dependencies = analyzed.imports;
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
    || manifest?.policyId !== NATIVE_PYTHON_PRODUCT_POLICY_ID
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

export function assertNativePythonProductMatchesSource(
  sourceRoot,
  productRoot,
) {
  assertNativePythonSourceClosure(sourceRoot);
  const expected = buildNativePythonProductManifest(sourceRoot);
  const actual = verifyNativePythonProductSource(productRoot);
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(
      "Packaged native Python source does not match the current product source.",
    );
  }
  return actual;
}

export function buildNativePythonProductManifest(sourceRoot) {
  const root = resolve(sourceRoot);
  return {
    schema: NATIVE_PYTHON_PRODUCT_SCHEMA,
    policyId: NATIVE_PYTHON_PRODUCT_POLICY_ID,
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

export function analyzeNativePythonImports(source) {
  if (typeof source !== "string") {
    throw new TypeError("Native Python source must be a string.");
  }
  return analyzeNativePythonImportSources([{
    path: "<native-python-source>",
    source,
  }])[0].imports;
}

const PYTHON_IMPORT_AST_SCHEMA =
  "mycellios-native-python-import-ast/1";
const PYTHON_IMPORT_ANALYZER = join(
  import.meta.dirname,
  "native-python-import-analyzer.py",
);

function analyzeNativePythonImportSources(sources) {
  const request = JSON.stringify({
    schema: PYTHON_IMPORT_AST_SCHEMA,
    sources,
  });
  const stdout = runNativePythonImportAnalyzer(request);
  let response;
  try {
    response = JSON.parse(stdout);
  } catch {
    throw new Error("Native Python AST analyzer returned invalid JSON.");
  }
  if (
    !isPlainObject(response)
    || !hasExactKeys(response, ["pythonVersion", "schema", "sources"])
    || response.schema !== PYTHON_IMPORT_AST_SCHEMA
    || !Array.isArray(response.pythonVersion)
    || response.pythonVersion.length !== 2
    || response.pythonVersion[0] !== 3
    || !Number.isInteger(response.pythonVersion[1])
    || response.pythonVersion[1] < 12
    || !Array.isArray(response.sources)
    || response.sources.length !== sources.length
  ) {
    throw new Error("Native Python AST analyzer returned an invalid response.");
  }
  for (const [index, analyzed] of response.sources.entries()) {
    if (
      !isPlainObject(analyzed)
      || !hasExactKeys(analyzed, ["imports", "path"])
      || analyzed.path !== sources[index].path
      || !Array.isArray(analyzed.imports)
      || analyzed.imports.some(
        (module) =>
          typeof module !== "string"
          || !/^distributed_runtime(?:\.[A-Za-z_][A-Za-z0-9_]*)+$/.test(module),
      )
      || JSON.stringify(analyzed.imports)
        !== JSON.stringify([...new Set(analyzed.imports)].sort())
    ) {
      throw new Error("Native Python AST analyzer returned invalid source evidence.");
    }
  }
  return response.sources;
}

function runNativePythonImportAnalyzer(request) {
  const commands = nativePythonAnalyzerCommands();
  const uvPython = resolveUvManagedPython();
  // On Windows, `py -3.12` can exist while that interpreter is not registered.
  // Use an already installed uv-managed runtime before trying the launcher.
  if (uvPython) commands.splice(process.env.MYCELLIOS_PYTHON?.trim() ? 1 : 0, 0, {
    executable: uvPython,
    prefix: [],
  });
  const missing = [];
  for (const command of commands) {
    const result = spawnSync(
      command.executable,
      [
        ...command.prefix,
        "-I",
        "-B",
        PYTHON_IMPORT_ANALYZER,
      ],
      {
        encoding: "utf8",
        input: request,
        maxBuffer: 16 * 1024 * 1024,
        shell: false,
        timeout: 30_000,
        windowsHide: true,
      },
    );
    if (result.error?.code === "ENOENT") {
      missing.push(command.executable);
      continue;
    }
    if (result.error) {
      throw new Error(
        `Native Python AST analyzer process failed: ${result.error.message}.`,
        { cause: result.error },
      );
    }
    const stderr = typeof result.stderr === "string"
      ? result.stderr.trim()
      : "";
    if (result.status !== 0 || result.signal !== null) {
      throw new Error(
        `Native Python AST analyzer failed via ${command.executable} ` +
        `(exit ${result.status ?? "none"}, signal ${result.signal ?? "none"}): ` +
        `${stderr || "no diagnostic output"}.`,
      );
    }
    if (
      typeof result.stdout !== "string"
      || typeof result.stderr !== "string"
    ) {
      throw new Error("Native Python AST analyzer returned invalid process streams.");
    }
    if (stderr) {
      throw new Error(
        `Native Python AST analyzer emitted unexpected diagnostics: ${stderr}.`,
      );
    }
    return result.stdout.trim();
  }
  throw new Error(
    `Python 3.12 is required for native Python AST analysis; commands not found: ${missing.join(", ")}.`,
  );
}

export function nativePythonAnalyzerCommands(
  platform = process.platform,
  configuredPython = process.env.MYCELLIOS_PYTHON,
) {
  const commands = [];
  if (typeof configuredPython === "string" && configuredPython.trim()) {
    commands.push({ executable: configuredPython.trim(), prefix: [] });
  }
  commands.push(...(platform === "win32"
    ? [
        { executable: "py", prefix: ["-3.12"] },
        { executable: "python", prefix: [] },
      ]
    : [
        { executable: "python3.12", prefix: [] },
        { executable: "python3", prefix: [] },
        { executable: "python", prefix: [] },
      ]));
  return commands;
}

function resolveUvManagedPython() {
  if (cachedUvPython !== undefined) return cachedUvPython;
  const result = spawnSync("uv", ["python", "find", "3.12"], {
    encoding: "utf8",
    shell: false,
    timeout: 10_000,
    windowsHide: true,
  });
  if (result.error?.code === "ENOENT" || result.status !== 0) {
    cachedUvPython = null;
    return cachedUvPython;
  }
  if (result.error || result.signal !== null || typeof result.stdout !== "string") {
    cachedUvPython = null;
    return cachedUvPython;
  }
  const executable = result.stdout.trim();
  cachedUvPython = executable && resolve(executable) === executable
    ? executable
    : null;
  return cachedUvPython;
}

function isPlainObject(value) {
  return (
    value !== null
    && typeof value === "object"
    && !Array.isArray(value)
    && Object.getPrototypeOf(value) === Object.prototype
  );
}

function hasExactKeys(value, expected) {
  return JSON.stringify(Object.keys(value).sort())
    === JSON.stringify([...expected].sort());
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
