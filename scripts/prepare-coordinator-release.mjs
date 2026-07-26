import { existsSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
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
const manifest = prepareCoordinatorRelease(workspace, destination, {
  revision,
  populateProductionDependencies: installProductionDependencies,
});
process.stdout.write(
  `Native coordinator staging ready: ${manifest.files.length} sealed files.\n`,
);

function installProductionDependencies(stagingRoot) {
  const npmCli = [
    process.env.npm_execpath,
    join(dirname(process.execPath), "node_modules", "npm", "bin", "npm-cli.js"),
    join(
      dirname(process.execPath),
      "..",
      "lib",
      "node_modules",
      "npm",
      "bin",
      "npm-cli.js",
    ),
  ].find((candidate) => candidate && existsSync(candidate));
  if (!npmCli) {
    throw new Error(
      "Cannot locate npm-cli.js for the sealed production dependency install.",
    );
  }
  const result = spawnSync(
    process.execPath,
    [
      npmCli,
      "ci",
      "--omit=dev",
      "--ignore-scripts",
      "--no-audit",
      "--no-fund",
      "--prefix",
      stagingRoot,
    ],
    {
      cwd: stagingRoot,
      encoding: "utf8",
      maxBuffer: 64 * 1024 * 1024,
      shell: false,
      windowsHide: true,
    },
  );
  if (result.stdout) process.stdout.write(result.stdout);
  if (result.stderr) process.stderr.write(result.stderr);
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(
      `Production dependency installation failed with exit code ${result.status ?? "unknown"}.`,
    );
  }
  rmSync(join(stagingRoot, "node_modules", ".bin"), {
    recursive: true,
    force: true,
  });
}
