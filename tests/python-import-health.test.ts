import { describe, expect, it } from "vitest";
import { analyzePythonImportBoundaries, type PythonImportPolicy } from "../src/program/python-import-health.js";

const policy: PythonImportPolicy = { schema: "mycellios-python-import-boundaries/1", forbiddenAbsoluteRoots: ["tests", "research"], allowedDebt: [], maximumAllowedDebt: 0 };

describe("Python import boundaries", () => {
  it("allows standard, dependency and package-relative imports", () => {
    expect(analyzePythonImportBoundaries([{ path: "python/distributed_runtime/a.py", content: "import json\nfrom .engine import Engine\nfrom torch import Tensor\n" }], policy)).toEqual([]);
  });
  it("rejects imports from non-product areas and opaque dynamic imports", () => {
    expect(analyzePythonImportBoundaries([{ path: "python/distributed_runtime/a.py", content: "from tests.fixtures import model\nimportlib.import_module(module_name)\n" }], policy)).toEqual([
      "python/distributed_runtime/a.py:1:forbidden_absolute_import:tests.fixtures",
      "python/distributed_runtime/a.py:2:dynamic_import_target_is_not_static",
    ]);
  });
  it("prevents adding an import-debt allowance", () => {
    expect(() => analyzePythonImportBoundaries([], { ...policy, allowedDebt: ["legacy"], maximumAllowedDebt: 1 })).toThrow("python_import_debt_must_only_decrease");
  });
});
