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

export const COORDINATOR_RELEASE_SCHEMA =
  "mycellios-native-coordinator-release/1";
export const COORDINATOR_RELEASE_MANIFEST =
  "mycellios-coordinator-release-manifest.json";

const coordinatorEntry = "coordinator/main.js";
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
  "package-lock.json",
  "package.json",
  "python",
  COORDINATOR_RELEASE_MANIFEST,
]);

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

  copyJavaScriptClosure(
    resolve(workspace, "dist"),
    resolve(destination, "dist"),
    [coordinatorEntry],
  );
  copyDirectory(workspace, destination, "landing-dist");
  copyDirectory(workspace, destination, "mobile-dist");
  copyRegularFile(workspace, destination, "package.json");
  copyRegularFile(workspace, destination, "package-lock.json");
  writeFileSync(resolveInside(destination, "REVISION"), `${revision}\n`, "utf8");
  for (const file of systemdFiles) {
    copyRegularFile(workspace, destination, `deploy/systemd/${file}`);
  }
  prepareNativePythonProductSource(
    resolve(workspace, "python"),
    resolve(destination, "python"),
  );

  const manifest = buildCoordinatorReleaseManifest(destination);
  writeFileSync(
    join(destination, COORDINATOR_RELEASE_MANIFEST),
    `${JSON.stringify(manifest, null, 2)}\n`,
    "utf8",
  );
  verifyCoordinatorReleaseDirectory(destination);
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

  const manifestPath = join(root, COORDINATOR_RELEASE_MANIFEST);
  let manifest;
  try {
    manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  } catch {
    throw new Error(`Coordinator release manifest is invalid: ${manifestPath}.`);
  }
  const expectedFiles = listRegularFiles(root)
    .filter((path) => path !== COORDINATOR_RELEASE_MANIFEST);
  if (
    manifest?.schema !== COORDINATOR_RELEASE_SCHEMA
    || !Array.isArray(manifest?.files)
    || JSON.stringify(manifest.files.map((entry) => entry?.path))
      !== JSON.stringify(expectedFiles)
  ) {
    throw new Error("Coordinator release manifest does not match its exact file set.");
  }
  for (const entry of manifest.files) {
    const bytes = readFileSync(resolveInside(root, entry.path));
    if (
      entry.bytes !== bytes.byteLength
      || entry.sha256 !== sha256(bytes)
    ) {
      throw new Error(`Coordinator release digest mismatch: ${entry.path}.`);
    }
  }
  return manifest;
}

export function buildCoordinatorReleaseManifest(releaseRoot) {
  const root = resolve(releaseRoot);
  return {
    schema: COORDINATOR_RELEASE_SCHEMA,
    entrypoint: `dist/${coordinatorEntry}`,
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
