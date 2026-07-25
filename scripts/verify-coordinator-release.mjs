import {
  existsSync,
  mkdtempSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, posix, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { verifyCoordinatorReleaseDirectory } from "./coordinator-release-policy.mjs";

const archiveArgument = readArgument("archive");
const rootArgument = readArgument("root");
if (Boolean(archiveArgument) === Boolean(rootArgument)) {
  throw new Error("Pass exactly one of --archive or --root.");
}

if (rootArgument) {
  const root = resolve(rootArgument);
  verifyCoordinatorReleaseDirectory(root);
  process.stdout.write(`Native coordinator staging verified: ${root}.\n`);
} else {
  const archive = resolve(archiveArgument);
  if (!existsSync(archive)) {
    throw new Error(`Coordinator release archive is missing: ${archive}.`);
  }
  const entries = tar(["-tzf", archive])
    .split(/\r?\n/)
    .map((entry) => entry.trim())
    .filter(Boolean);
  for (const entry of entries) {
    const portable = entry.replaceAll("\\", "/");
    const normalized = posix.normalize(portable.replace(/^\.\//, ""));
    if (
      portable.startsWith("/")
      || /^[A-Za-z]:/.test(portable)
      || normalized === ".."
      || normalized.startsWith("../")
    ) {
      throw new Error(`Coordinator archive contains an unsafe path: ${entry}.`);
    }
  }
  const temporary = mkdtempSync(join(tmpdir(), "mycellios-coordinator-release-"));
  try {
    tar(["-xzf", archive, "-C", temporary]);
    verifyCoordinatorReleaseDirectory(temporary);
  } finally {
    rmSync(temporary, { recursive: true, force: true });
  }
  process.stdout.write(`Native coordinator archive verified: ${archive}.\n`);
}

function readArgument(name) {
  const prefix = `--${name}=`;
  return process.argv
    .slice(2)
    .find((argument) => argument.startsWith(prefix))
    ?.slice(prefix.length)
    .trim();
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
    throw new Error(
      result.stderr?.trim() || `tar exited with ${result.status ?? "no status"}.`,
    );
  }
  return result.stdout;
}
