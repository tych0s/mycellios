import { mkdirSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { prepareCoordinatorRelease } from "./coordinator-release-policy.mjs";

const workspace = resolve(import.meta.dirname, "..");
const buildRoot = resolve(workspace, "build");
const destination = resolve(buildRoot, "coordinator-release");

if (dirname(destination) !== buildRoot) {
  throw new Error("Coordinator staging escaped the managed build directory.");
}
const revision = process.argv
  .slice(2)
  .find((argument) => argument.startsWith("--revision="))
  ?.slice("--revision=".length)
  .trim();
if (!revision) throw new Error("--revision=<40-character Git SHA> is required.");

rmSync(destination, { recursive: true, force: true });
mkdirSync(destination, { recursive: true });
const manifest = prepareCoordinatorRelease(workspace, destination, { revision });
process.stdout.write(
  `Native coordinator staging ready: ${manifest.files.length} sealed files.\n`,
);
