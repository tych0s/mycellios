import {
  copyFileSync,
  cpSync,
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from "node:fs";
import { createHash } from "node:crypto";
import { dirname, join, relative, resolve, sep } from "node:path";
import {
  prepareNativePythonProductSource,
  verifyNativePythonProductSource,
} from "./native-python-product-policy.mjs";
import {
  NATIVE_BUILD_PROVENANCE_FILE,
  buildNativeSourceProvenance,
  verifyNativeBuildProvenanceDocument,
} from "./native-build-provenance.mjs";

export const COORDINATOR_RELEASE_SCHEMA =
  "mycellios-native-coordinator-release/3";
export const COORDINATOR_RELEASE_MANIFEST =
  "mycellios-coordinator-release-manifest.json";
export const COORDINATOR_OUTPUT_RECEIPT_SCHEMA =
  "mycellios-native-coordinator-output-receipt/1";
export const COORDINATOR_OUTPUT_RECEIPT =
  "mycellios-coordinator-output-receipt.json";
export const COORDINATOR_WORKSPACE_OUTPUT_RECEIPT =
  `build/${COORDINATOR_OUTPUT_RECEIPT}`;

const coordinatorEntry = "coordinator/main.js";
const coordinatorOutputs = Object.freeze([
  Object.freeze({
    path: "dist",
    selection: "coordinator-js-closure",
  }),
  Object.freeze({
    path: "landing-dist",
    selection: "complete-tree",
  }),
  Object.freeze({
    path: "mobile-dist",
    selection: "complete-tree",
  }),
]);
const systemdFiles = Object.freeze([
  "mycellios-content-hub.conf",
  "mycellios-dynamic-workers.conf",
  "mycellios-release-storage.conf",
  "mycellios-supabase-persistence.conf",
]);
const requiredTopLevel = new Set([
  "REVISION",
  "deploy",
  "dist",
  "landing-dist",
  "mobile-dist",
  "node_modules",
  "package-lock.json",
  "package.json",
  "python",
  NATIVE_BUILD_PROVENANCE_FILE,
  COORDINATOR_OUTPUT_RECEIPT,
  COORDINATOR_RELEASE_MANIFEST,
]);

export function buildCoordinatorOutputReceipt(workspaceRoot) {
  const workspace = resolve(workspaceRoot);
  const provenance = buildNativeSourceProvenance(workspace);
  return buildCoordinatorOutputReceiptForSourceId(
    workspace,
    provenance.sourceId,
  );
}

export function writeCoordinatorOutputReceipt(
  workspaceRoot,
  receiptPath = COORDINATOR_WORKSPACE_OUTPUT_RECEIPT,
) {
  const workspace = resolve(workspaceRoot);
  const receipt = buildCoordinatorOutputReceipt(workspace);
  const destination = resolveInside(workspace, receiptPath);
  mkdirSync(dirname(destination), { recursive: true });
  writeFileSync(
    destination,
    `${JSON.stringify(receipt, null, 2)}\n`,
    "utf8",
  );
  assertCoordinatorOutputReceiptMatches(workspace, receipt);
  return receipt;
}

export function verifyCoordinatorOutputReceiptDocument(candidate) {
  assertPlainObject(candidate, "Coordinator output receipt");
  assertExactKeys(
    candidate,
    ["outputs", "receiptId", "schema", "sourceId"],
    "Coordinator output receipt",
  );
  if (candidate.schema !== COORDINATOR_OUTPUT_RECEIPT_SCHEMA) {
    throw new Error(
      `Unsupported coordinator output receipt schema: ${candidate.schema}.`,
    );
  }
  if (
    typeof candidate.sourceId !== "string"
    || !/^sha256:[0-9a-f]{64}$/.test(candidate.sourceId)
  ) {
    throw new Error("Coordinator output receipt sourceId is invalid.");
  }
  if (
    typeof candidate.receiptId !== "string"
    || !/^sha256:[0-9a-f]{64}$/.test(candidate.receiptId)
  ) {
    throw new Error("Coordinator output receipt receiptId is invalid.");
  }
  if (
    !Array.isArray(candidate.outputs)
    || candidate.outputs.length !== coordinatorOutputs.length
  ) {
    throw new Error(
      "Coordinator output receipt must seal exactly three native outputs.",
    );
  }
  for (const [outputIndex, output] of candidate.outputs.entries()) {
    const definition = coordinatorOutputs[outputIndex];
    const label = `Coordinator output ${outputIndex}`;
    assertPlainObject(output, label);
    assertExactKeys(
      output,
      ["files", "outputId", "path", "selection"],
      label,
    );
    if (
      output.path !== definition.path
      || output.selection !== definition.selection
    ) {
      throw new Error(
        "Coordinator outputs must use the canonical order and selection.",
      );
    }
    if (
      typeof output.outputId !== "string"
      || !/^sha256:[0-9a-f]{64}$/.test(output.outputId)
    ) {
      throw new Error(`${label} outputId is invalid.`);
    }
    if (!Array.isArray(output.files) || output.files.length < 1) {
      throw new Error(`${label} must contain at least one file.`);
    }
    let previousPath = null;
    for (const [fileIndex, file] of output.files.entries()) {
      const fileLabel = `${label} file ${fileIndex}`;
      assertPlainObject(file, fileLabel);
      assertExactKeys(file, ["bytes", "path", "sha256"], fileLabel);
      if (
        typeof file.path !== "string"
        || normalizePortable(file.path) !== file.path
      ) {
        throw new Error(`${fileLabel} has an invalid path.`);
      }
      if (
        previousPath !== null
        && comparePortablePaths(previousPath, file.path) >= 0
      ) {
        throw new Error(
          `${label} file paths must be unique and strictly sorted.`,
        );
      }
      if (!Number.isSafeInteger(file.bytes) || file.bytes < 0) {
        throw new Error(`${fileLabel} has an invalid byte length.`);
      }
      if (
        typeof file.sha256 !== "string"
        || !/^[0-9a-f]{64}$/.test(file.sha256)
      ) {
        throw new Error(`${fileLabel} has an invalid SHA-256 digest.`);
      }
      previousPath = file.path;
    }
    const outputIdentity = {
      files: output.files,
      path: output.path,
      selection: output.selection,
    };
    const expectedOutputId =
      `sha256:${sha256(Buffer.from(canonicalJson(outputIdentity)))}`;
    if (output.outputId !== expectedOutputId) {
      throw new Error(`${label} outputId does not seal its file evidence.`);
    }
  }
  const identity = {
    outputs: candidate.outputs,
    schema: candidate.schema,
    sourceId: candidate.sourceId,
  };
  const expectedReceiptId =
    `sha256:${sha256(Buffer.from(canonicalJson(identity)))}`;
  if (candidate.receiptId !== expectedReceiptId) {
    throw new Error(
      "Coordinator output receiptId does not seal its source and outputs.",
    );
  }
  return candidate;
}

export function assertCoordinatorOutputReceiptMatches(
  workspaceRoot,
  candidate,
) {
  const workspace = resolve(workspaceRoot);
  const verified = verifyCoordinatorOutputReceiptDocument(candidate);
  const expected = buildCoordinatorOutputReceipt(workspace);
  if (verified.sourceId !== expected.sourceId) {
    throw new Error(
      `Coordinator output receipt sourceId is stale: sealed ${verified.sourceId}, current ${expected.sourceId}. Rebuild the coordinator outputs before packaging.`,
    );
  }
  const expectedOutputs = new Map(
    expected.outputs.map((output) => [output.path, output]),
  );
  for (const output of verified.outputs) {
    const current = expectedOutputs.get(output.path);
    if (!current || canonicalJson(output) !== canonicalJson(current)) {
      throw new Error(
        `Coordinator output receipt is stale for ${output.path}: rebuild that output before packaging.`,
      );
    }
  }
  if (verified.receiptId !== expected.receiptId) {
    throw new Error(
      "Coordinator output receiptId is stale for the current source and outputs.",
    );
  }
  return expected;
}

export function prepareCoordinatorRelease(
  workspaceRoot,
  destinationRoot,
  options = {},
) {
  const workspace = resolve(workspaceRoot);
  const destination = resolve(destinationRoot);
  const revision = String(options.revision ?? "").trim();
  if (!/^[0-9a-f]{40}$/.test(revision)) {
    throw new Error("Coordinator release revision must be an exact Git SHA.");
  }
  const requiredInputs = [
    "dist/coordinator/main.js",
    "landing-dist",
    "mobile-dist",
    "package.json",
    "package-lock.json",
    "python/distributed_runtime/server.py",
    ...systemdFiles.map((file) => `deploy/systemd/${file}`),
  ];
  for (const portable of requiredInputs) {
    if (!existsSync(resolveInside(workspace, portable))) {
      throw new Error(`Coordinator release input is missing: ${portable}.`);
    }
  }
  const receiptPath = resolveInside(
    workspace,
    COORDINATOR_WORKSPACE_OUTPUT_RECEIPT,
  );
  if (!existsSync(receiptPath) || !lstatSync(receiptPath).isFile()) {
    throw new Error(
      `Coordinator build output receipt is missing: ${COORDINATOR_WORKSPACE_OUTPUT_RECEIPT}.`,
    );
  }
  let outputReceipt;
  try {
    outputReceipt = assertCoordinatorOutputReceiptMatches(
      workspace,
      JSON.parse(readFileSync(receiptPath, "utf8")),
    );
  } catch (error) {
    throw new Error(
      `Coordinator build output receipt is invalid: ${error instanceof Error ? error.message : String(error)}.`,
    );
  }

  copyJavaScriptClosure(
    resolve(workspace, "dist"),
    resolve(destination, "dist"),
    [coordinatorEntry],
  );
  copyDirectory(workspace, destination, "landing-dist");
  copyDirectory(workspace, destination, "mobile-dist");
  copyRegularFile(workspace, destination, "package.json");
  copyRegularFile(workspace, destination, "package-lock.json");
  writeFileSync(
    resolveInside(destination, COORDINATOR_OUTPUT_RECEIPT),
    `${JSON.stringify(outputReceipt, null, 2)}\n`,
    "utf8",
  );
  writeFileSync(resolveInside(destination, "REVISION"), `${revision}\n`, "utf8");
  for (const file of systemdFiles) {
    copyRegularFile(workspace, destination, `deploy/systemd/${file}`);
  }
  prepareNativePythonProductSource(
    resolve(workspace, "python"),
    resolve(destination, "python"),
  );
  const buildProvenance = buildNativeSourceProvenance(workspace);
  writeFileSync(
    resolveInside(destination, NATIVE_BUILD_PROVENANCE_FILE),
    `${JSON.stringify(buildProvenance, null, 2)}\n`,
    "utf8",
  );
  if (typeof options.populateProductionDependencies !== "function") {
    throw new Error(
      "Coordinator release requires a production dependency installer before sealing.",
    );
  }
  options.populateProductionDependencies(destination);
  verifyProductionDependencyTree(destination);

  const manifest = buildCoordinatorReleaseManifest(destination);
  writeFileSync(
    join(destination, COORDINATOR_RELEASE_MANIFEST),
    `${JSON.stringify(manifest, null, 2)}\n`,
    "utf8",
  );
  verifyCoordinatorReleaseDirectory(destination);
  const finalSourceId = buildNativeSourceProvenance(workspace).sourceId;
  if (finalSourceId !== outputReceipt.sourceId) {
    throw new Error(
      `Coordinator source changed while the release was being sealed: expected ${outputReceipt.sourceId}, current ${finalSourceId}. Rebuild before packaging.`,
    );
  }
  try {
    assertCoordinatorOutputReceiptMatches(workspace, outputReceipt);
  } catch (error) {
    throw new Error(
      `Coordinator build outputs changed while the release was being sealed: ${error instanceof Error ? error.message : String(error)}.`,
    );
  }
  return manifest;
}

export function verifyCoordinatorReleaseDirectory(releaseRoot) {
  const root = resolve(releaseRoot);
  const topLevel = readdirSync(root).sort();
  if (
    topLevel.length !== requiredTopLevel.size
    || topLevel.some((entry) => !requiredTopLevel.has(entry))
  ) {
    throw new Error(
      `Coordinator release has unexpected top-level entries: ${topLevel
        .filter((entry) => !requiredTopLevel.has(entry))
        .join(", ") || "entry count mismatch"}.`,
    );
  }
  for (const file of systemdFiles) {
    const path = resolveInside(root, `deploy/systemd/${file}`);
    if (!existsSync(path) || !lstatSync(path).isFile()) {
      throw new Error(`Coordinator release is missing systemd policy: ${file}.`);
    }
  }
  const deployFiles = listRegularFiles(resolve(root, "deploy"));
  const expectedDeploy = systemdFiles.map((file) => `systemd/${file}`).sort();
  if (JSON.stringify(deployFiles) !== JSON.stringify(expectedDeploy)) {
    throw new Error("Coordinator release contains non-production deploy material.");
  }

  verifyJavaScriptClosure(resolve(root, "dist"), [coordinatorEntry]);
  verifyNativePythonProductSource(resolve(root, "python"));
  verifyProductionDependencyTree(root);
  let buildProvenance;
  try {
    buildProvenance = verifyNativeBuildProvenanceDocument(
      JSON.parse(
        readFileSync(
          resolveInside(root, NATIVE_BUILD_PROVENANCE_FILE),
          "utf8",
        ),
      ),
    );
  } catch (error) {
    throw new Error(
      `Coordinator build provenance is invalid: ${error instanceof Error ? error.message : String(error)}.`,
    );
  }
  let outputReceipt;
  try {
    outputReceipt = verifyCoordinatorOutputReceiptDocument(
      JSON.parse(
        readFileSync(
          resolveInside(root, COORDINATOR_OUTPUT_RECEIPT),
          "utf8",
        ),
      ),
    );
  } catch (error) {
    throw new Error(
      `Coordinator output receipt is invalid: ${error instanceof Error ? error.message : String(error)}.`,
    );
  }
  if (outputReceipt.sourceId !== buildProvenance.sourceId) {
    throw new Error(
      "Coordinator output receipt source identity drifted from provenance.",
    );
  }
  const expectedOutputReceipt = buildCoordinatorOutputReceiptForSourceId(
    root,
    buildProvenance.sourceId,
  );
  if (canonicalJson(outputReceipt) !== canonicalJson(expectedOutputReceipt)) {
    throw new Error(
      "Coordinator output receipt does not match the exact packaged outputs.",
    );
  }

  const manifestPath = join(root, COORDINATOR_RELEASE_MANIFEST);
  let manifest;
  try {
    manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  } catch {
    throw new Error(`Coordinator release manifest is invalid: ${manifestPath}.`);
  }
  const expected = buildCoordinatorReleaseManifest(root);
  if (canonicalJson(manifest) !== canonicalJson(expected)) {
    throw new Error(
      "Coordinator release manifest does not match its exact build identity and file set.",
    );
  }
  if (manifest.sourceId !== buildProvenance.sourceId) {
    throw new Error("Coordinator release source identity drifted from provenance.");
  }
  return manifest;
}

export function buildCoordinatorReleaseManifest(releaseRoot) {
  const root = resolve(releaseRoot);
  const revision = readFileSync(resolveInside(root, "REVISION"), "utf8").trim();
  if (!/^[0-9a-f]{40}$/.test(revision)) {
    throw new Error("Coordinator release REVISION is not an exact Git SHA.");
  }
  const metadata = JSON.parse(
    readFileSync(resolveInside(root, "package.json"), "utf8"),
  );
  if (
    metadata === null
    || typeof metadata !== "object"
    || Array.isArray(metadata)
    || typeof metadata.version !== "string"
    || !metadata.version
  ) {
    throw new Error("Coordinator release package version is invalid.");
  }
  const provenance = verifyNativeBuildProvenanceDocument(
    JSON.parse(
      readFileSync(
        resolveInside(root, NATIVE_BUILD_PROVENANCE_FILE),
        "utf8",
      ),
    ),
  );
  if (provenance.version !== metadata.version) {
    throw new Error(
      "Coordinator release provenance version does not match package.json.",
    );
  }
  const identity = {
    schema: COORDINATOR_RELEASE_SCHEMA,
    entrypoint: `dist/${coordinatorEntry}`,
    revision,
    version: metadata.version,
    sourceId: provenance.sourceId,
    files: listRegularFiles(root)
      .filter((path) => path !== COORDINATOR_RELEASE_MANIFEST)
      .map((portable) => {
        const bytes = readFileSync(resolveInside(root, portable));
        return {
          path: portable,
          bytes: bytes.byteLength,
          sha256: sha256(bytes),
        };
      }),
  };
  return {
    ...identity,
    releaseId: `sha256:${sha256(Buffer.from(canonicalJson(identity)))}`,
  };
}

function buildCoordinatorOutputReceiptForSourceId(root, sourceId) {
  const base = resolve(root);
  const outputs = coordinatorOutputs.map((definition) => {
    const files = definition.path === "dist"
      ? javascriptClosure(resolve(base, definition.path), [coordinatorEntry])
        .map((portable) =>
          fileEvidence(resolve(base, definition.path), portable))
      : listRegularFiles(resolve(base, definition.path))
        .map((portable) =>
          fileEvidence(resolve(base, definition.path), portable));
    if (files.length < 1) {
      throw new Error(
        `Coordinator output is empty: ${definition.path}.`,
      );
    }
    const identity = {
      files,
      path: definition.path,
      selection: definition.selection,
    };
    return {
      ...identity,
      outputId: `sha256:${sha256(Buffer.from(canonicalJson(identity)))}`,
    };
  });
  const identity = {
    schema: COORDINATOR_OUTPUT_RECEIPT_SCHEMA,
    sourceId,
    outputs,
  };
  return {
    ...identity,
    receiptId: `sha256:${sha256(Buffer.from(canonicalJson(identity)))}`,
  };
}

function verifyProductionDependencyTree(releaseRoot) {
  const root = resolve(releaseRoot);
  const nodeModules = resolveInside(root, "node_modules");
  if (!existsSync(nodeModules) || !lstatSync(nodeModules).isDirectory()) {
    throw new Error(
      "Coordinator release production dependencies are missing.",
    );
  }
  const installedLock = resolveInside(
    root,
    "node_modules/.package-lock.json",
  );
  if (!existsSync(installedLock) || !lstatSync(installedLock).isFile()) {
    throw new Error(
      "Coordinator release production dependency receipt is missing.",
    );
  }
  const executableLinks = resolveInside(root, "node_modules/.bin");
  if (existsSync(executableLinks)) {
    throw new Error(
      "Coordinator release must remove the npm .bin link surface before sealing.",
    );
  }
  listRegularFiles(nodeModules);
}

function fileEvidence(root, portable) {
  const bytes = readFileSync(resolveInside(root, portable));
  return {
    path: portable,
    bytes: bytes.byteLength,
    sha256: sha256(bytes),
  };
}

function copyJavaScriptClosure(sourceRoot, destinationRoot, entries) {
  const source = resolve(sourceRoot);
  const destination = resolve(destinationRoot);
  for (const portable of javascriptClosure(source, entries)) {
    const input = resolveInside(source, portable);
    const output = resolveInside(destination, portable);
    mkdirSync(dirname(output), { recursive: true });
    copyFileSync(input, output);
  }
}

function verifyJavaScriptClosure(root, entries) {
  const expected = javascriptClosure(root, entries);
  const actual = listRegularFiles(root);
  if (
    actual.some((path) => !path.endsWith(".js"))
    || JSON.stringify(actual) !== JSON.stringify(expected)
  ) {
    throw new Error(
      "Coordinator dist is not the exact JavaScript closure of coordinator/main.js.",
    );
  }
  for (const portable of actual) {
    if (
      portable.startsWith("simulator/")
      || portable.endsWith("/demo.js")
      || portable === "demo.js"
    ) {
      throw new Error(`Coordinator release contains a simulation surface: ${portable}.`);
    }
  }
}

function javascriptClosure(root, entries) {
  const base = resolve(root);
  const visited = new Set();
  const pending = [...entries];
  while (pending.length > 0) {
    const portable = normalizePortable(pending.pop());
    if (visited.has(portable)) continue;
    const absolute = resolveInside(base, portable);
    if (!existsSync(absolute) || !lstatSync(absolute).isFile()) {
      throw new Error(`Coordinator JavaScript closure is missing: ${portable}.`);
    }
    if (!portable.endsWith(".js")) {
      throw new Error(`Coordinator JavaScript import is not executable JS: ${portable}.`);
    }
    visited.add(portable);
    const source = readFileSync(absolute, "utf8");
    for (const specifier of relativeJavaScriptImports(source)) {
      const dependency = normalizePortable(
        relative(base, resolve(dirname(absolute), specifier)).replaceAll("\\", "/"),
      );
      if (!dependency.endsWith(".js")) {
        throw new Error(
          `Coordinator JavaScript import is not pinned to .js: ${portable} -> ${specifier}.`,
        );
      }
      pending.push(dependency);
    }
  }
  return [...visited].sort();
}

function relativeJavaScriptImports(source) {
  const imports = new Set();
  const patterns = [
    /\bfrom\s*["'](\.{1,2}\/[^"']+)["']/g,
    /^\s*import\s*["'](\.{1,2}\/[^"']+)["']/gm,
    /\bimport\s*\(\s*["'](\.{1,2}\/[^"']+)["']\s*\)/g,
  ];
  for (const pattern of patterns) {
    let match;
    while ((match = pattern.exec(source)) !== null) imports.add(match[1]);
  }
  return [...imports];
}

function copyDirectory(workspace, destination, portable) {
  const input = resolveInside(workspace, portable);
  const output = resolveInside(destination, portable);
  cpSync(input, output, {
    recursive: true,
    dereference: true,
    errorOnExist: true,
  });
}

function copyRegularFile(workspace, destination, portable) {
  const input = resolveInside(workspace, portable);
  if (!lstatSync(input).isFile()) {
    throw new Error(`Coordinator release input is not a regular file: ${portable}.`);
  }
  const output = resolveInside(destination, portable);
  mkdirSync(dirname(output), { recursive: true });
  copyFileSync(input, output);
}

function listRegularFiles(root) {
  const base = resolve(root);
  const files = [];
  const visit = (directory) => {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const absolute = join(directory, entry.name);
      if (entry.isSymbolicLink()) {
        throw new Error(`Coordinator release contains a symbolic link: ${absolute}.`);
      }
      if (entry.isDirectory()) visit(absolute);
      else if (entry.isFile()) {
        files.push(relative(base, absolute).replaceAll("\\", "/"));
      } else {
        throw new Error(`Coordinator release contains a special file: ${absolute}.`);
      }
    }
  };
  visit(base);
  return files.sort();
}

function normalizePortable(value) {
  const portable = value.replaceAll("\\", "/").replace(/^\.\//, "");
  if (
    !portable
    || portable.startsWith("/")
    || /^[A-Za-z]:/.test(portable)
    || portable === ".."
    || portable.startsWith("../")
    || portable.includes("/../")
  ) {
    throw new Error(`Coordinator release path escaped its root: ${value}.`);
  }
  return portable;
}

function assertPlainObject(value, label) {
  if (
    value === null
    || typeof value !== "object"
    || Array.isArray(value)
    || Object.getPrototypeOf(value) !== Object.prototype
  ) {
    throw new Error(`${label} must be a plain JSON object.`);
  }
}

function assertExactKeys(value, expected, label) {
  const actual = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  if (JSON.stringify(actual) !== JSON.stringify(sortedExpected)) {
    throw new Error(`${label} contains unexpected or missing fields.`);
  }
}

function comparePortablePaths(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function resolveInside(root, portable) {
  const base = resolve(root);
  const normalized = normalizePortable(portable);
  const absolute = resolve(base, ...normalized.split("/"));
  if (absolute !== base && !absolute.startsWith(`${base}${sep}`)) {
    throw new Error(`Coordinator release path escaped its root: ${portable}.`);
  }
  return absolute;
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function canonicalJson(value) {
  if (Array.isArray(value)) {
    return `[${value.map((entry) => canonicalJson(entry)).join(",")}]`;
  }
  if (value !== null && typeof value === "object") {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}
