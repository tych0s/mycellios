export interface PythonImportPolicy {
  schema: "mycellios-python-import-boundaries/1";
  forbiddenAbsoluteRoots: string[];
  allowedDebt: string[];
  maximumAllowedDebt: number;
}

export interface PythonSource { path: string; content: string }

export function analyzePythonImportBoundaries(sources: readonly PythonSource[], policy: PythonImportPolicy): string[] {
  if (policy.schema !== "mycellios-python-import-boundaries/1") throw new Error("python_import_policy_schema_is_invalid");
  if (policy.maximumAllowedDebt !== 0 || policy.allowedDebt.length > policy.maximumAllowedDebt) throw new Error("python_import_debt_must_only_decrease");
  const forbidden = new Set(policy.forbiddenAbsoluteRoots);
  const violations = [];
  for (const source of sources) {
    const lines = source.content.split("\n");
    for (const [index, line] of lines.entries()) {
      const match = line.match(/^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import\b|import\s+([A-Za-z_][\w.]*))/);
      const target = match?.[1] ?? match?.[2];
      if (target && forbidden.has(target.split(".")[0]!)) violations.push(`${source.path}:${index + 1}:forbidden_absolute_import:${target}`);
      if (/\b(?:importlib\.import_module|__import__)\s*\(\s*(?!["'])/.test(line)) violations.push(`${source.path}:${index + 1}:dynamic_import_target_is_not_static`);
    }
  }
  const allowed = new Set(policy.allowedDebt);
  return violations.filter((violation) => !allowed.has(violation)).sort();
}
