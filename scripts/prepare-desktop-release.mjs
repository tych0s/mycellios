import { createHash, randomBytes } from "node:crypto";
import {
  closeSync,
  copyFileSync,
  cpSync,
  existsSync,
  lstatSync,
  mkdirSync,
  openSync,
  readSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, resolve, sep } from "node:path";

const root = resolve(import.meta.dirname, "..");
const metadata = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
const version = metadata.version;
const makerDirectory = resolve(root, "out", "make", "squirrel.windows", "x64");
const updateDirectory = resolve(root, "updates", "win32", "x64");
const downloadDirectory = resolve(root, "landing-dist", "downloads");
const fullPackageName = `mycellios-${version}-full.nupkg`;
const expected = ["RELEASES", "mycellios-setup.exe", fullPackageName];

const sources = expected.map((fileName) => {
  const path = inside(makerDirectory, fileName);
  const stats = requireRegularFile(path);
  return {
    fileName,
    path,
    bytes: stats.size,
    sha256: sha256File(path),
  };
});

const updateParent = dirname(updateDirectory);
mkdirSync(updateParent, { recursive: true });
const nonce = `${process.pid}-${randomBytes(8).toString("hex")}`;
const stagingDirectory = inside(updateParent, `.x64-stage-${nonce}`);
const backupDirectory = inside(updateParent, `.x64-backup-${nonce}`);
if (existsSync(stagingDirectory) || existsSync(backupDirectory)) {
  throw new Error("Release staging path unexpectedly already exists.");
}

if (existsSync(updateDirectory)) {
  requireDirectory(updateDirectory);
  cpSync(updateDirectory, stagingDirectory, {
    recursive: true,
    errorOnExist: true,
    force: false,
  });
} else {
  mkdirSync(stagingDirectory, { recursive: true });
}

for (const fileName of readdirSync(stagingDirectory)) {
  if (
    fileName === "RELEASES"
    || fileName === "latest.json"
    || fileName.endsWith(".nupkg")
    || fileName.endsWith("-setup.exe")
  ) {
    rmSync(inside(stagingDirectory, fileName), { force: true });
  }
}

for (const source of sources) {
  const destination = inside(stagingDirectory, source.fileName);
  copyFileSync(source.path, destination);
  assertCopy(source, destination);
}

const latest = {
  schema: "mycellios-windows-update-feed/1",
  version,
  publishedAt: new Date().toISOString(),
  files: sources.map(({ fileName, bytes, sha256 }) => ({
    name: basename(fileName),
    bytes,
    sha256,
  })),
};
writeFileSync(
  inside(stagingDirectory, "latest.json"),
  `${JSON.stringify(latest, null, 2)}\n`,
  "utf8",
);

let oldFeedMoved = false;
let newFeedInstalled = false;
try {
  if (existsSync(updateDirectory)) {
    renameSync(updateDirectory, backupDirectory);
    oldFeedMoved = true;
  }
  renameSync(stagingDirectory, updateDirectory);
  newFeedInstalled = true;

  for (const source of sources) {
    assertCopy(source, inside(updateDirectory, source.fileName));
  }
  const publishedLatest = JSON.parse(
    readFileSync(inside(updateDirectory, "latest.json"), "utf8"),
  );
  if (JSON.stringify(publishedLatest) !== JSON.stringify(latest)) {
    throw new Error("Published latest.json changed during the atomic feed swap.");
  }
} catch (error) {
  if (newFeedInstalled && existsSync(updateDirectory)) {
    rmSync(updateDirectory, { recursive: true, force: true });
  }
  if (oldFeedMoved && existsSync(backupDirectory)) {
    renameSync(backupDirectory, updateDirectory);
  }
  throw error;
}
if (existsSync(backupDirectory)) {
  rmSync(backupDirectory, { recursive: true, force: true });
}

mkdirSync(downloadDirectory, { recursive: true });
const setup = sources.find(({ fileName }) => fileName === "mycellios-setup.exe");
if (!setup) throw new Error("Internal error: Setup source evidence is missing.");
const stagedDownload = inside(downloadDirectory, `.mycellios-windows-x64-${nonce}.tmp`);
const backupDownload = inside(downloadDirectory, `.mycellios-windows-x64-${nonce}.bak`);
const finalDownload = inside(downloadDirectory, "mycellios-windows-x64.exe");
copyFileSync(setup.path, stagedDownload);
assertCopy(setup, stagedDownload);
let oldDownloadMoved = false;
let newDownloadInstalled = false;
try {
  if (existsSync(finalDownload)) {
    requireRegularFile(finalDownload);
    renameSync(finalDownload, backupDownload);
    oldDownloadMoved = true;
  }
  renameSync(stagedDownload, finalDownload);
  newDownloadInstalled = true;
  assertCopy(setup, finalDownload);
} catch (error) {
  if (newDownloadInstalled && existsSync(finalDownload)) {
    rmSync(finalDownload, { force: true });
  }
  if (oldDownloadMoved && existsSync(backupDownload)) {
    renameSync(backupDownload, finalDownload);
  }
  throw error;
}
if (existsSync(backupDownload)) {
  rmSync(backupDownload, { force: true });
}

console.log(`Prepared and re-verified mycellios ${version} at ${updateDirectory}`);

function assertCopy(source, destination) {
  const stats = requireRegularFile(destination);
  const digest = sha256File(destination);
  if (stats.size !== source.bytes || digest !== source.sha256) {
    throw new Error(
      `Release copy mismatch for ${destination}: ${stats.size}/${digest} instead of ${source.bytes}/${source.sha256}.`,
    );
  }
}

function requireRegularFile(path) {
  const stats = lstatSync(path);
  if (!stats.isFile() || stats.isSymbolicLink() || stats.size < 1) {
    throw new Error(`Expected a non-empty regular release file: ${path}`);
  }
  return stats;
}

function requireDirectory(path) {
  const stats = lstatSync(path);
  if (!stats.isDirectory() || stats.isSymbolicLink()) {
    throw new Error(`Expected a real release directory: ${path}`);
  }
}

function inside(parent, name) {
  const candidate = resolve(parent, name);
  if (candidate !== parent && !candidate.startsWith(`${parent}${sep}`)) {
    throw new Error(`Release path escaped ${parent}: ${name}`);
  }
  return candidate;
}

function sha256File(path) {
  const hash = createHash("sha256");
  const descriptor = openSync(path, "r");
  const buffer = Buffer.allocUnsafe(8 * 1024 * 1024);
  try {
    while (true) {
      const bytesRead = readSync(
        descriptor,
        buffer,
        0,
        buffer.byteLength,
        null,
      );
      if (bytesRead === 0) break;
      hash.update(buffer.subarray(0, bytesRead));
    }
  } finally {
    closeSync(descriptor);
  }
  return hash.digest("hex");
}
