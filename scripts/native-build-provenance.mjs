import { createHash } from "node:crypto";
import {
  lstatSync,
  readFileSync,
  readdirSync,
} from "node:fs";
import { join, relative, resolve, sep } from "node:path";
import { NATIVE_PYTHON_PRODUCT_FILES } from "./native-python-product-policy.mjs";

export const NATIVE_BUILD_PROVENANCE_SCHEMA =
  "mycellios-native-build-provenance/1";
export const NATIVE_BUILD_PROVENANCE_FILE =
  "mycellios-native-build-provenance.json";

const SOURCE_INPUTS = Object.freeze([
  ".gitattributes",
  ".github/actionlint.yaml",
  ".github/workflows",
  "package.json",
  "package-lock.json",
  "python/requirements-distribution.txt",
  "vite.landing.config.ts",
  "vite.mobile.config.ts",
  "tsconfig.json",
  "tsconfig.build.json",
  "tsconfig.landing.json",
  "tsconfig.mobile.json",
  "assets/mycellios-logo.png",
  "build/icons",
  "config",
  "deploy",
  "landing/index.html",
  "landing/public",
  "landing/src",
  "src",
  "scripts",
  ...NATIVE_PYTHON_PRODUCT_FILES.map((portable) => `python/${portable}`),
]);

const GENERATED_DIRECTORY_NAMES = new Set([".vite", "node_modules"]);

export function buildNativeSourceProvenance(workspaceRoot) {
  const root = resolve(workspaceRoot);
  const packageMetadata = parsePackageJson(root);
  const files = [];
  const observedPaths = new Set();
  for (const portable of SOURCE_INPUTS) {
    const absolute = resolveInside(root, portable);
    const stats = lstatSync(absolute);
    if (stats.isSymbolicLink()) {
      throw new Error(`Build input cannot be a symbolic link: ${portable}.`);
    }
    if (stats.isDirectory()) {
      collectFiles(root, absolute, files, observedPaths);
    } else if (stats.isFile()) {
      addFileEvidence(root, absolute, files, observedPaths);
    } else {
      throw new Error(`Build input is not a regular file or directory: ${portable}.`);
    }
  }
  files.sort((left, right) => comparePortablePaths(left.path, right.path));
  const identityDocument = {
    schema: NATIVE_BUILD_PROVENANCE_SCHEMA,
    version: packageMetadata.version,
    files,
  };
  return {
    ...identityDocument,
    sourceId: sourceIdFor(identityDocument),
  };
}

export function verifyNativeBuildProvenanceDocument(candidate) {
  assertPlainObject(candidate, "Build provenance");
  assertExactKeys(
    candidate,
    ["files", "schema", "sourceId", "version"],
    "Build provenance",
  );
  if (candidate.schema !== NATIVE_BUILD_PROVENANCE_SCHEMA) {
    throw new Error(`Unsupported build provenance schema: ${candidate.schema}.`);
  }
  if (
    typeof candidate.version !== "string"
    || candidate.version.length < 1
    || candidate.version.length > 128
    || /[\u0000-\u001f\u007f]/u.test(candidate.version)
  ) {
    throw new Error("Build provenance version is invalid.");
  }
  if (
    typeof candidate.sourceId !== "string"
    || !/^sha256:[0-9a-f]{64}$/.test(candidate.sourceId)
  ) {
    throw new Error("Build provenance sourceId is invalid.");
  }
  if (!Array.isArray(candidate.files) || candidate.files.length < 1) {
    throw new Error("Build provenance must contain at least one source file.");
  }

  let previousPath = null;
  for (const [index, entry] of candidate.files.entries()) {
    const label = `Build provenance file ${index}`;
    assertPlainObject(entry, label);
    assertExactKeys(entry, ["bytes", "path", "sha256"], label);
    if (
      typeof entry.path !== "string"
      || !isSafePortablePath(entry.path)
    ) {
      throw new Error(`${label} has an invalid path.`);
    }
    if (
      previousPath !== null
      && comparePortablePaths(previousPath, entry.path) >= 0
    ) {
      throw new Error(
        "Build provenance file paths must be unique and strictly sorted.",
      );
    }
    if (
      !Number.isSafeInteger(entry.bytes)
      || entry.bytes < 0
    ) {
      throw new Error(`${label} has an invalid byte length.`);
    }
    if (
      typeof entry.sha256 !== "string"
      || !/^[0-9a-f]{64}$/.test(entry.sha256)
    ) {
      throw new Error(`${label} has an invalid SHA-256 digest.`);
    }
    previousPath = entry.path;
  }

  const identityDocument = {
    schema: candidate.schema,
    version: candidate.version,
    files: candidate.files,
  };
  const expectedSourceId = sourceIdFor(identityDocument);
  if (candidate.sourceId !== expectedSourceId) {
    throw new Error(
      `Build provenance sourceId does not seal its file evidence: expected ${expectedSourceId}.`,
    );
  }
  return candidate;
}

export function assertNativeSourceProvenanceMatches(
  workspaceRoot,
  candidate,
) {
  const verified = verifyNativeBuildProvenanceDocument(candidate);
  const expected = buildNativeSourceProvenance(workspaceRoot);
  if (canonicalJson(verified) !== canonicalJson(expected)) {
    throw new Error(
      "Packaged app source provenance does not match the current Mycellios source tree.",
    );
  }
  return expected;
}

function collectFiles(root, directory, output, observedPaths) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const absolute = join(directory, entry.name);
    if (entry.isSymbolicLink()) {
      throw new Error(
        `Build input cannot contain a symbolic link: ${portablePath(root, absolute)}.`,
      );
    }
    if (entry.isDirectory()) {
      if (GENERATED_DIRECTORY_NAMES.has(entry.name)) continue;
      collectFiles(root, absolute, output, observedPaths);
    } else if (entry.isFile()) {
      addFileEvidence(root, absolute, output, observedPaths);
    } else {
      throw new Error(
        `Build input contains a special file: ${portablePath(root, absolute)}.`,
      );
    }
  }
}

function addFileEvidence(root, absolute, output, observedPaths) {
  const evidence = fileEvidence(root, absolute);
  if (observedPaths.has(evidence.path)) {
    throw new Error(`Build input is listed more than once: ${evidence.path}.`);
  }
  observedPaths.add(evidence.path);
  output.push(evidence);
}

function fileEvidence(root, absolute) {
  const bytes = readFileSync(absolute);
  return {
    path: portablePath(root, absolute),
    bytes: bytes.byteLength,
    sha256: sha256(bytes),
  };
}

function parsePackageJson(root) {
  const path = resolveInside(root, "package.json");
  let value;
  try {
    value = JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    throw new Error(`Cannot parse ${path}: ${error.message}`);
  }
  if (
    value === null
    || typeof value !== "object"
    || Array.isArray(value)
    || typeof value.version !== "string"
    || !value.version
  ) {
    throw new Error("package.json does not expose a valid version.");
  }
  return value;
}

function portablePath(root, absolute) {
  const portable = relative(root, absolute).replaceAll("\\", "/");
  if (!isSafePortablePath(portable)) {
    throw new Error(`Build input escaped its root: ${absolute}.`);
  }
  return portable;
}

function isSafePortablePath(portable) {
  if (
    !portable
    || portable.length > 4096
    || portable.includes("\\")
    || portable.startsWith("/")
    || portable.endsWith("/")
    || /^[A-Za-z]:/u.test(portable)
    || /[\u0000-\u001f\u007f]/u.test(portable)
  ) {
    return false;
  }
  const segments = portable.split("/");
  return segments.every(
    (segment) => segment.length > 0 && segment !== "." && segment !== "..",
  );
}

function resolveInside(root, portable) {
  const absolute = resolve(root, ...portable.split("/"));
  if (absolute !== root && !absolute.startsWith(`${root}${sep}`)) {
    throw new Error(`Build input escaped its root: ${portable}.`);
  }
  return absolute;
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

function sourceIdFor(identityDocument) {
  return `sha256:${sha256(Buffer.from(canonicalJson(identityDocument)))}`;
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

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
