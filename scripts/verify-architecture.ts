import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import {
  analyzeArchitecture,
  type ArchitectureSource,
  type ModuleSizeBudget,
} from "../src/program/architecture-health.js";
import { analyzePythonImportBoundaries, type PythonImportPolicy } from "../src/program/python-import-health.js";

function trackedFiles(...patterns: string[]): string[] {
  const output = execFileSync(
    "git",
    ["ls-files", "--cached", "--others", "--exclude-standard", "--", ...patterns],
    { encoding: "utf8" },
  );
  return output.trim().split("\n").filter(Boolean).sort();
}

function readSources(files: readonly string[]): ArchitectureSource[] {
  return files.map((file) => ({ path: file, content: readFileSync(file, "utf8") }));
}

const graphSources = readSources(
  trackedFiles("src", "landing").filter((file) => /\.(?:ts|tsx)$/u.test(file)),
);
const sizedSources = readSources(
  trackedFiles("src", "landing", "python/distributed_runtime", "scripts")
    .filter((file) => /\.(?:ts|tsx|mjs|py)$/u.test(file)),
);
const budget = JSON.parse(readFileSync("config/module-size-budget.json", "utf8")) as ModuleSizeBudget;
const result = analyzeArchitecture(graphSources, sizedSources, budget);
const pythonImportPolicy = JSON.parse(readFileSync("config/python-import-boundaries.json", "utf8")) as PythonImportPolicy;
const pythonImportViolations = analyzePythonImportBoundaries(
  readSources(trackedFiles("python/distributed_runtime").filter((file) => file.endsWith(".py"))),
  pythonImportPolicy,
);
const failures = [
  ...result.unknownAreas.map((file) => `unknown_component:${file}`),
  ...result.cycles.map((cycle) => `dependency_cycle:${cycle.join(" -> ")}`),
  ...result.sizeViolations.map((violation) => `module_size_growth:${violation}`),
  ...result.staleSizeDebt.map((file) => `stale_module_size_debt:${file}`),
  ...result.reducibleSizeDebt.map((file) => `module_size_ceiling_must_decrease:${file}`),
  ...pythonImportViolations.map((violation) => `python_import_boundary:${violation}`),
];

if (failures.length > 0) {
  console.error(["Architecture health check failed:", ...failures].join("\n"));
  process.exitCode = 1;
} else {
  console.log(
    `Architecture verified: ${result.files} TypeScript files, ${result.edges} relative imports, `
    + `${Object.keys(budget.trackedDebt).length} shrinking module-size debts, zero cycles.`,
  );
}
