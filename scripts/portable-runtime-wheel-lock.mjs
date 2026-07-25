import { createHash } from "node:crypto";
import {
  createReadStream,
  lstatSync,
  readFileSync,
  readdirSync,
} from "node:fs";
import { basename, relative, resolve, sep } from "node:path";

export const PORTABLE_RUNTIME_WHEEL_LOCK_SCHEMA =
  "mycellios-python-wheel-lock/1";

const ARTIFACT_LINE =
  /^([a-z0-9]+(?:-[a-z0-9]+)*) @ (https:\/\/\S+) --hash=sha256:([0-9a-f]{64}) # version=([A-Za-z0-9][A-Za-z0-9.+!_-]{0,127})$/;
const APPROVED_HOSTS = new Set([
  "files.pythonhosted.org",
  "download.pytorch.org",
  "download-r2.pytorch.org",
]);

export function readPortableRuntimeWheelLock(workspaceRoot, spec) {
  if (!spec?.supported || !spec.wheelLock) {
    throw new Error("Portable runtime has no supported wheel-lock policy.");
  }
  const root = resolve(workspaceRoot);
  const path = resolve(root, ...spec.wheelLock.path.split("/"));
  const back = relative(root, path);
  if (
    !back
    || back === ".."
    || back.startsWith(`..${sep}`)
    || resolve(root, back) !== path
  ) {
    throw new Error("Portable runtime wheel lock escaped the workspace.");
  }
  const source = readFileSync(path);
  const digest = createHash("sha256").update(source).digest("hex");
  if (digest !== spec.wheelLock.sha256) {
    throw new Error(
      `Portable runtime wheel lock ${spec.wheelLock.path} has SHA-256 ${digest}; expected ${spec.wheelLock.sha256}.`,
    );
  }
  return parsePortableRuntimeWheelLock(source.toString("utf8"), spec);
}

export function parsePortableRuntimeWheelLock(source, spec) {
  if (typeof source !== "string" || source.includes("\r")) {
    throw new Error("Portable runtime wheel lock must use canonical LF text.");
  }
  const lines = source.split("\n");
  if (lines.at(-1) !== "") {
    throw new Error("Portable runtime wheel lock must end with one newline.");
  }
  lines.pop();
  const expectedTarget = `# target=${spec.platform}/${spec.arch}/cp312`;
  if (
    lines[0] !== `# ${PORTABLE_RUNTIME_WHEEL_LOCK_SCHEMA}`
    || lines[1] !== expectedTarget
    || lines[2]
      !== "# Regenerate deliberately from a reviewed pip JSON report; never edit hashes by hand."
  ) {
    throw new Error("Portable runtime wheel lock header is invalid.");
  }
  const artifacts = [];
  let previousName = null;
  const filenames = new Set();
  for (const [index, line] of lines.slice(3).entries()) {
    const match = ARTIFACT_LINE.exec(line);
    if (!match) {
      throw new Error(`Portable runtime wheel lock line ${index + 4} is invalid.`);
    }
    const [, name, urlValue, sha256, version] = match;
    if (previousName !== null && previousName >= name) {
      throw new Error(
        "Portable runtime wheel-lock packages must be unique and strictly sorted.",
      );
    }
    previousName = name;
    const url = new URL(urlValue);
    if (
      !APPROVED_HOSTS.has(url.hostname)
      || url.username
      || url.password
      || url.search
      || url.hash
    ) {
      throw new Error(`Portable runtime wheel URL is not approved: ${urlValue}.`);
    }
    const filename = decodeURIComponent(basename(url.pathname));
    if (
      !filename.endsWith(".whl")
      || /[\0\r\n/\\]/.test(filename)
      || filenames.has(filename)
    ) {
      throw new Error(`Portable runtime wheel filename is invalid: ${filename}.`);
    }
    filenames.add(filename);
    const wheelParts = filename.split("-");
    if (
      normalizeName(wheelParts[0]) !== name
      || wheelParts[1]?.replaceAll("_", "-") !== version.replaceAll("_", "-")
    ) {
      throw new Error(
        `Portable runtime wheel ${filename} does not match ${name} ${version}.`,
      );
    }
    artifacts.push({ name, version, url: url.href, sha256, filename });
  }
  if (artifacts.length === 0) {
    throw new Error("Portable runtime wheel lock is empty.");
  }
  const resolvedVersions = Object.fromEntries(
    artifacts.map(({ name, version }) => [name, version]),
  );
  const expected = {
    torch: spec.torchVersion,
    ...Object.fromEntries(
      Object.entries(spec.packageVersions).map(([name, version]) => [
        normalizeName(name),
        version,
      ]),
    ),
  };
  for (const [name, version] of Object.entries(expected)) {
    if (resolvedVersions[name] !== version) {
      throw new Error(
        `Portable runtime wheel lock resolved ${name} ${resolvedVersions[name] ?? "missing"}; expected ${version}.`,
      );
    }
  }
  return Object.freeze({
    schema: PORTABLE_RUNTIME_WHEEL_LOCK_SCHEMA,
    platform: spec.platform,
    arch: spec.arch,
    sha256: spec.wheelLock.sha256,
    path: spec.wheelLock.path,
    artifacts: Object.freeze(
      artifacts.map((artifact) => Object.freeze(artifact)),
    ),
  });
}

export function portableRuntimeOfflineRequirements(lock) {
  if (lock?.schema !== PORTABLE_RUNTIME_WHEEL_LOCK_SCHEMA) {
    throw new Error("Portable runtime wheel lock is missing or invalid.");
  }
  return [
    "# Generated from the sealed Mycellios wheel lock. Do not persist or edit.",
    ...lock.artifacts.map(
      ({ name, version, sha256 }) =>
        `${name}==${version} --hash=sha256:${sha256}`,
    ),
    "",
  ].join("\n");
}

export async function verifyPortableRuntimeWheelhouse(directory, lock) {
  if (lock?.schema !== PORTABLE_RUNTIME_WHEEL_LOCK_SCHEMA) {
    throw new Error("Portable runtime wheel lock is missing or invalid.");
  }
  const root = resolve(directory);
  const expected = new Map(
    lock.artifacts.map((artifact) => [artifact.filename, artifact]),
  );
  const entries = readdirSync(root, { withFileTypes: true });
  if (entries.length !== expected.size) {
    throw new Error(
      `Portable runtime wheelhouse contains ${entries.length} entries; expected ${expected.size}.`,
    );
  }
  for (const entry of entries) {
    const artifact = expected.get(entry.name);
    const path = resolve(root, entry.name);
    if (
      !artifact
      || !entry.isFile()
      || entry.isSymbolicLink()
      || !lstatSync(path).isFile()
    ) {
      throw new Error(
        `Portable runtime wheelhouse contains an unexpected entry: ${entry.name}.`,
      );
    }
    const hash = createHash("sha256");
    for await (const chunk of createReadStream(path)) hash.update(chunk);
    const digest = hash.digest("hex");
    if (digest !== artifact.sha256) {
      throw new Error(
        `Portable runtime wheel ${entry.name} has SHA-256 ${digest}; expected ${artifact.sha256}.`,
      );
    }
  }
  return true;
}

function normalizeName(value) {
  return String(value ?? "").toLowerCase().replace(/[_.]+/g, "-");
}
