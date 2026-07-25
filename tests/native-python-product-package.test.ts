import {
  cpSync,
  mkdtempSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  NATIVE_PYTHON_ENTRY_MODULES,
  NATIVE_PYTHON_IMPORT_SMOKE_MODULES,
  NATIVE_PYTHON_PRODUCT_FILES,
  NATIVE_PYTHON_PRODUCT_MANIFEST,
  assertNativePythonSourceClosure,
  prepareNativePythonProductSource,
  verifyNativePythonProductSource,
} from "../scripts/native-python-product-policy.mjs";

const temporaryRoots: string[] = [];

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) {
    rmSync(root, { recursive: true, force: true });
  }
});

describe("native Python product package", () => {
  it("is a closed allowlist covering every native product entrypoint", () => {
    expect(() =>
      assertNativePythonSourceClosure(resolve("python")),
    ).not.toThrow();
    expect(NATIVE_PYTHON_ENTRY_MODULES).toEqual(expect.arrayContaining([
      "distributed_runtime.server",
      "distributed_runtime.stage_cli",
      "distributed_runtime.cell_member_cli",
      "distributed_runtime.native_gguf_cli",
      "distributed_runtime.installed_stage_canary",
    ]));
    expect(NATIVE_PYTHON_IMPORT_SMOKE_MODULES).toEqual(expect.arrayContaining([
      "distributed_runtime.native_gguf_runtime",
      "distributed_runtime.native_gguf_disk_tiering",
      "distributed_runtime.dense_tiering",
    ]));
    expect(NATIVE_PYTHON_PRODUCT_FILES).toEqual(expect.arrayContaining([
      "distributed_runtime/native_gguf_runtime.py",
      "distributed_runtime/native_gguf_disk_tiering.py",
      "distributed_runtime/dense_tiering.py",
    ]));
    for (const forbidden of [
      "distributed_runtime/external_gguf_runtime.py",
      "distributed_runtime/native_stage.py",
      "distributed_runtime/gpu_cloud_probe.py",
      "distributed_runtime/benchmark.py",
      "distributed_runtime/resident_expert_mesh_cli.py",
    ]) {
      expect(NATIVE_PYTHON_PRODUCT_FILES).not.toContain(forbidden);
    }
  });

  it("copies only admitted files and seals their exact bytes", () => {
    const root = temporaryDirectory();
    const destination = join(root, "python");
    prepareNativePythonProductSource(resolve("python"), destination);
    const manifest = verifyNativePythonProductSource(destination);
    expect(manifest.files).toHaveLength(NATIVE_PYTHON_PRODUCT_FILES.length);
    expect(JSON.parse(
      readFileSync(join(destination, NATIVE_PYTHON_PRODUCT_MANIFEST), "utf8"),
    )).toEqual(manifest);
  });

  it("rejects tampering and any extra research or simulation file", () => {
    const root = temporaryDirectory();
    const destination = join(root, "python");
    prepareNativePythonProductSource(resolve("python"), destination);
    const server = join(destination, "distributed_runtime", "server.py");
    writeFileSync(server, `${readFileSync(server, "utf8")}\n# tampered\n`, "utf8");
    expect(() => verifyNativePythonProductSource(destination)).toThrow(
      "digest mismatch",
    );

    rmSync(destination, { recursive: true, force: true });
    prepareNativePythonProductSource(resolve("python"), destination);
    const research = join(destination, "distributed_runtime", "benchmark.py");
    mkdirSync(join(destination, "distributed_runtime"), { recursive: true });
    cpSync(resolve("python/distributed_runtime/benchmark.py"), research);
    expect(() => verifyNativePythonProductSource(destination)).toThrow(
      "outside the allowlist",
    );
  });
});

function temporaryDirectory(): string {
  const root = mkdtempSync(join(tmpdir(), "mycellios-native-python-test-"));
  temporaryRoots.push(root);
  return root;
}
