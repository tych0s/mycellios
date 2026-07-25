import {
  cpSync,
  existsSync,
  mkdirSync,
  rmSync,
} from "node:fs";
import { dirname, join, relative, resolve } from "node:path";

const workspace = resolve(import.meta.dirname, "..");
const sourceRoot = resolve(workspace, "python");
const buildRoot = resolve(workspace, "build");
const destinationRoot = resolve(buildRoot, "python");

const excludedRuntimeFiles = new Set([
  "distributed_runtime/external_gguf_runtime.py",
  "distributed_runtime/external_gguf_runtime_probe_cli.py",
  "distributed_runtime/external_gguf_runtime_rpc.py",
  "distributed_runtime/native_stage.py",
  "distributed_runtime/native_stage_package.py",
  "distributed_runtime/native_stage_package_cli.py",
  "distributed_runtime/gpu_cloud_probe.py",
]);

if (dirname(destinationRoot) !== buildRoot) {
  throw new Error("Packaged Python source escaped the managed build directory.");
}
if (!existsSync(sourceRoot)) {
  throw new Error(`Mycellios Python source is missing: ${sourceRoot}.`);
}

rmSync(destinationRoot, { recursive: true, force: true });
mkdirSync(buildRoot, { recursive: true });
cpSync(sourceRoot, destinationRoot, {
  recursive: true,
  filter(source) {
    const portable = relative(sourceRoot, source).replaceAll("\\", "/");
    if (!portable) return true;
    const parts = portable.split("/");
    if (
      parts.includes("__pycache__")
      || parts.includes(".ruff_cache")
      || parts.some((part) => part.endsWith(".pyc"))
      || parts[0] === "tests"
    ) {
      return false;
    }
    return !excludedRuntimeFiles.has(portable);
  },
});

for (const required of [
  "distributed_runtime/__init__.py",
  "distributed_runtime/runtime_policy.py",
  "distributed_runtime/model_adapter_registry.json",
  "distributed_runtime/native_gguf.py",
  "distributed_runtime/server.py",
  "distributed_runtime/stage_cli.py",
]) {
  if (!existsSync(join(destinationRoot, ...required.split("/")))) {
    throw new Error(`Packaged Mycellios runtime source is incomplete: ${required}.`);
  }
}
for (const excluded of excludedRuntimeFiles) {
  if (existsSync(join(destinationRoot, ...excluded.split("/")))) {
    throw new Error(`External research backend leaked into the package: ${excluded}.`);
  }
}

process.stdout.write(
  "Packaged Python source ready: Mycellios native runtime only; research backends excluded.\n",
);
