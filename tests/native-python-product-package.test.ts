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
  analyzeNativePythonImports,
  assertNativePythonProductMatchesSource,
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
      "distributed_runtime.draft_model",
      "distributed_runtime.lossless_sampling",
    ]));
    expect(NATIVE_PYTHON_PRODUCT_FILES).toEqual(expect.arrayContaining([
      "distributed_runtime/native_gguf_runtime.py",
      "distributed_runtime/native_gguf_disk_tiering.py",
      "distributed_runtime/dense_tiering.py",
      "distributed_runtime/draft_model.py",
      "distributed_runtime/lossless_sampling.py",
      "distributed_runtime/failure_evidence.py",
      "distributed_runtime/recovery_outcome.py",
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

  it("closes imports with the Python 3.12 AST across aliases and multiline forms", () => {
    const source = `
DECOY = """
from . import string_decoy
import distributed_runtime.string_decoy
"""
# importlib.import_module("distributed_runtime.comment_decoy")

from . import (
    benchmark as relative_alias,
    gpu_cloud_probe,
)
from .native_stage import run as run_native_stage
from distributed_runtime import (
    external_gguf_runtime as llama,
    resident_expert_mesh_cli,
)
import distributed_runtime.cell_fixture_compiler as compiler, distributed_runtime.benchmark as benchmark_module

import importlib as imports, runpy as runner
from importlib import import_module as load_module, invalidate_caches
from runpy import run_module as execute_module
from builtins import __import__ as builtin_import

assigned_loader = imports.import_module
assigned_runner, assigned_builtin = runner.run_module, builtin_import

imports.import_module("distributed_runtime.dynamic_one")
runner.run_module(mod_name="distributed_runtime.dynamic_two")
__import__("distributed_runtime.dynamic_three")
load_module("distributed_runtime.dynamic_four")
execute_module("distributed_runtime.dynamic_five")
builtin_import("distributed_runtime.dynamic_six")
assigned_loader("distributed_runtime." + "dynamic_seven")
assigned_runner("distributed_runtime.dynamic_eight")
assigned_builtin("distributed_runtime.dynamic_nine")
`;
    expect(analyzeNativePythonImports(source)).toEqual([
      "distributed_runtime.benchmark",
      "distributed_runtime.cell_fixture_compiler",
      "distributed_runtime.dynamic_eight",
      "distributed_runtime.dynamic_five",
      "distributed_runtime.dynamic_four",
      "distributed_runtime.dynamic_nine",
      "distributed_runtime.dynamic_one",
      "distributed_runtime.dynamic_seven",
      "distributed_runtime.dynamic_six",
      "distributed_runtime.dynamic_three",
      "distributed_runtime.dynamic_two",
      "distributed_runtime.external_gguf_runtime",
      "distributed_runtime.gpu_cloud_probe",
      "distributed_runtime.native_stage",
      "distributed_runtime.resident_expert_mesh_cli",
    ]);
  });

  it("fails closed when product Python is not valid Python 3.12 syntax", () => {
    expect(() =>
      analyzeNativePythonImports("def invalid(:\n    pass\n"),
    ).toThrow(/AST analyzer failed.*invalid syntax/i);
  });

  it("fails closed when a dynamic import target cannot be proven statically", () => {
    expect(() =>
      analyzeNativePythonImports(`
import importlib
module_name = input()
importlib.import_module(module_name)
`),
    ).toThrow(/target must be a statically known string/i);
  });

  it("rejects AST-discovered imports that escape the product allowlist", () => {
    const root = temporaryDirectory();
    const source = join(root, "python");
    prepareNativePythonProductSource(resolve("python"), source);
    const server = join(source, "distributed_runtime", "server.py");
    writeFileSync(
      server,
      `${readFileSync(server, "utf8")}
if False:
    from distributed_runtime import benchmark as hidden_benchmark, gpu_cloud_probe
`,
      "utf8",
    );
    expect(() => assertNativePythonSourceClosure(source)).toThrow(
      "distributed_runtime.server -> distributed_runtime.benchmark",
    );
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

  it("rejects a valid but stale packaged source tree", () => {
    const root = temporaryDirectory();
    const source = join(root, "source");
    const product = join(root, "product");
    prepareNativePythonProductSource(resolve("python"), source);
    prepareNativePythonProductSource(source, product);
    expect(() =>
      assertNativePythonProductMatchesSource(source, product),
    ).not.toThrow();

    const server = join(source, "distributed_runtime", "server.py");
    writeFileSync(server, `${readFileSync(server, "utf8")}\n# newer source\n`, "utf8");
    expect(() =>
      assertNativePythonProductMatchesSource(source, product),
    ).toThrow("does not match the current product source");
  });
});

function temporaryDirectory(): string {
  const root = mkdtempSync(join(tmpdir(), "mycellios-native-python-test-"));
  temporaryRoots.push(root);
  return root;
}
