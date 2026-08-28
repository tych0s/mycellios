import { createHash } from "node:crypto";
import { chmod, cp, lstat, mkdir, readFile, readdir, realpath, rm, writeFile } from "node:fs/promises";
import { basename, dirname, join, relative, resolve, sep } from "node:path";
import { pathToFileURL } from "node:url";
import { buildNativeSourceProvenance, verifyNativeBuildProvenanceDocument } from "./native-build-provenance.mjs";

export const NODE_INSTALLER_LAYOUT_SCHEMA = "mycellios-node-installer-layout/2";
const PRODUCTION_DEPENDENCIES = ["ws", "zod"];

export async function stageNodeInstaller(input) {
  const output = resolve(input.output);
  const dist = resolve(input.dist);
  const runtime = resolve(input.runtime);
  const nodeExecutable = resolve(input.nodeExecutable);
  const modules = resolve(input.nodeModules);
  if (!/^[a-f0-9]{40}$/.test(input.sourceRevision)) throw new Error("node_installer_source_revision_is_invalid");
  const provenance = verifyNativeBuildProvenanceDocument(input.sourceProvenance);
  if (output === dist || output === runtime || output === modules || output === dirname(output)) {
    throw new Error("node_installer_output_is_unsafe");
  }
  await assertDirectory(dist, "node_installer_dist_is_missing");
  await assertDirectory(runtime, "node_installer_runtime_is_missing");
  await assertRegularFile(nodeExecutable, "node_installer_node_executable_is_missing");
  await assertRegularFile(join(dist, "node", "main.js"), "node_installer_main_is_missing");
  await assertRegularFile(join(dist, "node", "install-main.js"), "node_installer_bootstrap_is_missing");
  await assertRegularFile(join(dist, "node", "uninstall-main.js"), "node_installer_uninstall_helper_is_missing");
  const runtimeManifest = JSON.parse(await readFile(join(runtime, "runtime-manifest.json"), "utf8"));
  const [platform, arch] = parseTarget(input.target);
  if (runtimeManifest.platform !== platform || runtimeManifest.arch !== arch) {
    throw new Error("node_installer_runtime_target_mismatch");
  }
  await assertSymlinksStayInside(runtime);
  await rm(output, { recursive: true, force: true });
  await mkdir(join(output, "bin"), { recursive: true });
  await Promise.all([
    copyTree(dist, join(output, "app")),
    copyTree(runtime, join(output, "runtime"), { materializeInternalSymlinks: true }),
    cp(nodeExecutable, join(output, "bin", platform === "win32" ? "node.exe" : "node")),
    ...PRODUCTION_DEPENDENCIES.map(async (dependency) => {
      const source = join(modules, dependency);
      await assertDirectory(source, `node_installer_dependency_is_missing:${dependency}`);
      await copyTree(source, join(output, "node_modules", dependency));
    }),
  ]);
  const provenanceBytes = Buffer.from(`${JSON.stringify(provenance, null, 2)}\n`);
  await writeFile(join(output, "app", "mycellios-native-build-provenance.json"), provenanceBytes, { flag: "wx" });
  await writeFile(join(output, "app", "manifest.json"), `${JSON.stringify({
    schema: "mycellios-node-runtime-identity/1", version: provenance.version, sourceRevision: input.sourceRevision,
    sourceId: provenance.sourceId, provenanceSha256: sha256(provenanceBytes),
  }, null, 2)}\n`, { flag: "wx" });
  if (platform !== "win32") await (await import("node:fs/promises")).chmod(join(output, "bin", "node"), 0o755);
  const launcher = await writeInstallerLauncher(output, platform);
  const files = await inventoryNodeInstaller(output);
  const manifest = {
    schema: NODE_INSTALLER_LAYOUT_SCHEMA,
    target: input.target,
    sourceRevision: input.sourceRevision,
    source: { revision: input.sourceRevision, sourceId: provenance.sourceId, provenanceSha256: sha256(provenanceBytes), version: provenance.version },
    toolchain: { nodeVersion: process.version, pythonVersion: runtimeManifest.pythonVersion ?? "unknown",
      pythonAbi: runtimeManifest.pythonAbi ?? "unknown", backend: runtimeManifest.backend ?? "unknown" },
    entrypoints: {
      service: "app/node/main.js",
      install: "app/node/install-main.js",
      uninstall: "app/node/uninstall-main.js",
      launcher,
    },
    runtimeManifestSha256: sha256(await readFile(join(output, "runtime", "runtime-manifest.json"))),
    files,
  };
  await writeFile(join(output, "layout-manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, { flag: "wx" });
  return manifest;
}

async function writeInstallerLauncher(root, platform) {
  if (platform === "win32") {
    const path = join(root, "install.ps1");
    await writeFile(path, [
      "param([Parameter(Mandatory=$true)][string]$Enrollment)",
      "$ErrorActionPreference='Stop'",
      "$principal=New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())",
      "if(-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)){",
      '  $arguments="-NoProfile -ExecutionPolicy Bypass -File `"$($MyInvocation.MyCommand.Path)`" -Enrollment `"$Enrollment`""',
      "  $process=Start-Process powershell.exe -Verb RunAs -ArgumentList $arguments -Wait -PassThru",
      "  exit $process.ExitCode",
      "}",
      "$root=Split-Path -Parent $MyInvocation.MyCommand.Path",
      "& (Join-Path $root 'bin\\node.exe') (Join-Path $root 'app\\node\\install-main.js') --install-root $root --enrollment $Enrollment",
      "if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}",
      "",
    ].join("\r\n"), { flag: "wx" });
    return "install.ps1";
  }
  const path = join(root, "install");
  await writeFile(path, [
    "#!/bin/sh", "set -eu", "if [ \"$(id -u)\" -ne 0 ]; then echo 'Root privileges are required' >&2; exit 1; fi",
    "if [ \"$#\" -ne 1 ]; then echo 'usage: install <pairing.mycellios-enrollment>' >&2; exit 2; fi",
    "root=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd -P)",
    "exec \"$root/bin/node\" \"$root/app/node/install-main.js\" --install-root \"$root\" --enrollment \"$1\"", "",
  ].join("\n"), { flag: "wx", mode: 0o755 });
  await chmod(path, 0o755);
  return "install";
}

export async function inventoryNodeInstaller(root, excluded = new Set()) {
  const files = [];
  async function visit(directory) {
    const entries = await readdir(directory, { withFileTypes: true });
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      const absolute = join(directory, entry.name);
      const portablePath = relative(root, absolute).replaceAll("\\", "/");
      if (excluded.has(portablePath)) continue;
      if (entry.isSymbolicLink()) throw new Error(`node_installer_symlink_is_forbidden:${relative(root, absolute)}`);
      if (entry.isDirectory()) await visit(absolute);
      else if (entry.isFile()) {
        const bytes = await readFile(absolute);
        files.push({ path: portablePath, bytes: bytes.byteLength, sha256: sha256(bytes) });
      } else throw new Error(`node_installer_special_file_is_forbidden:${relative(root, absolute)}`);
    }
  }
  await visit(root);
  return files;
}

async function copyTree(source, destination, options = {}) {
  await cp(source, destination, {
    recursive: true,
    dereference: options.materializeInternalSymlinks === true,
    errorOnExist: true,
    force: false,
  });
}

async function assertSymlinksStayInside(root) {
  const canonicalRoot = await realpath(root);
  async function visit(directory) {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const absolute = join(directory, entry.name);
      if (entry.isSymbolicLink()) {
        const target = await realpath(absolute).catch(() => null);
        if (!target || (target !== canonicalRoot && !target.startsWith(`${canonicalRoot}${sep}`))) {
          throw new Error(`node_installer_runtime_symlink_escapes_root:${relative(root, absolute)}`);
        }
      } else if (entry.isDirectory()) await visit(absolute);
    }
  }
  await visit(root);
}

async function assertDirectory(path, error) {
  const stats = await lstat(path).catch(() => null);
  if (!stats?.isDirectory() || stats.isSymbolicLink()) throw new Error(error);
}

async function assertRegularFile(path, error) {
  const stats = await lstat(path).catch(() => null);
  if (!stats?.isFile() || stats.isSymbolicLink()) throw new Error(error);
}

function parseTarget(target) {
  const match = /^(linux|macos|windows|darwin|win32)-(x64|arm64)$/.exec(target);
  if (!match) throw new Error("node_installer_target_is_invalid");
  const platform = match[1] === "windows" ? "win32" : match[1] === "macos" ? "darwin" : match[1];
  return [platform, match[2]];
}

function sha256(bytes) { return createHash("sha256").update(bytes).digest("hex"); }

async function main(argv) {
  const values = Object.fromEntries(argv.map((argument) => {
    const match = /^--([^=]+)=(.+)$/.exec(argument);
    if (!match) throw new Error(`node_installer_argument_is_invalid:${argument}`);
    return [match[1], match[2]];
  }));
  for (const key of ["output", "dist", "runtime", "node-executable", "node-modules", "target", "source-revision", "source-root"]) {
    if (!values[key]) throw new Error(`node_installer_argument_is_missing:${key}`);
  }
  await stageNodeInstaller({ output: values.output, dist: values.dist, runtime: values.runtime,
    nodeExecutable: values["node-executable"], nodeModules: values["node-modules"], target: values.target,
    sourceRevision: values["source-revision"], sourceProvenance: buildNativeSourceProvenance(values["source-root"]) });
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) await main(process.argv.slice(2));
