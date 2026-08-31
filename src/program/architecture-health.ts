import path from "node:path";
import { componentFor } from "./component-boundaries.js";

export interface ArchitectureSource {
  path: string;
  content: string;
}

export interface ModuleSizeBudget {
  schema: "mycellios-module-size-budget/2";
  defaultMaximumLines: number;
  trackedDebt: Record<string, { ceiling: number; target: number; owner: string }>;
}

export interface ArchitectureHealthResult {
  files: number;
  edges: number;
  unknownAreas: string[];
  cycles: string[][];
  sizeViolations: string[];
  staleSizeDebt: string[];
  reducibleSizeDebt: string[];
}

const importPattern = /(?:\bfrom\s+|\bimport\s*(?:\(\s*)?|\bexport\s+[^"']*\bfrom\s+)["']([^"']+)["']/gu;

function resolveRelativeImport(
  fromFile: string,
  specifier: string,
  files: ReadonlySet<string>,
): string | null {
  if (!specifier.startsWith(".")) return null;
  const base = path.posix
    .normalize(path.posix.join(path.posix.dirname(fromFile), specifier))
    .replace(/\.(?:js|mjs|cjs)$/u, "");
  for (const candidate of [
    `${base}.ts`, `${base}.tsx`, `${base}.mts`, `${base}.cts`,
    `${base}/index.ts`, `${base}/index.tsx`,
  ]) {
    if (files.has(candidate)) return candidate;
  }
  return null;
}

function stronglyConnectedComponents(graph: ReadonlyMap<string, readonly string[]>): string[][] {
  let nextIndex = 0;
  const indices = new Map<string, number>();
  const lowLinks = new Map<string, number>();
  const stack: string[] = [];
  const onStack = new Set<string>();
  const components: string[][] = [];

  const visit = (node: string): void => {
    indices.set(node, nextIndex);
    lowLinks.set(node, nextIndex);
    nextIndex += 1;
    stack.push(node);
    onStack.add(node);

    for (const dependency of graph.get(node) ?? []) {
      if (!indices.has(dependency)) {
        visit(dependency);
        lowLinks.set(node, Math.min(lowLinks.get(node)!, lowLinks.get(dependency)!));
      } else if (onStack.has(dependency)) {
        lowLinks.set(node, Math.min(lowLinks.get(node)!, indices.get(dependency)!));
      }
    }

    if (lowLinks.get(node) !== indices.get(node)) return;
    const component: string[] = [];
    let current: string;
    do {
      current = stack.pop()!;
      onStack.delete(current);
      component.push(current);
    } while (current !== node);
    if (component.length > 1) components.push(component.sort());
  };

  for (const node of graph.keys()) if (!indices.has(node)) visit(node);
  return components.sort((left, right) => left[0]!.localeCompare(right[0]!));
}

export function countLines(content: string): number {
  if (content.length === 0) return 0;
  return content.endsWith("\n")
    ? content.slice(0, -1).split("\n").length
    : content.split("\n").length;
}

export function analyzeArchitecture(
  graphSources: readonly ArchitectureSource[],
  sizedSources: readonly ArchitectureSource[],
  budget: ModuleSizeBudget,
): ArchitectureHealthResult {
  if (budget.schema !== "mycellios-module-size-budget/2") {
    throw new Error("module_size_budget_schema_is_invalid");
  }
  if (!Number.isInteger(budget.defaultMaximumLines) || budget.defaultMaximumLines < 1) {
    throw new Error("module_size_default_limit_is_invalid");
  }

  const sourcePaths = new Set(graphSources.map((source) => source.path));
  const graph = new Map<string, string[]>();
  const unknownAreas = new Set<string>();
  let edges = 0;
  for (const source of graphSources) {
    if (componentFor(source.path) === "unknown") unknownAreas.add(source.path);
    const dependencies: string[] = [];
    for (const match of source.content.matchAll(importPattern)) {
      const dependency = resolveRelativeImport(source.path, match[1]!, sourcePaths);
      if (dependency) {
        dependencies.push(dependency);
        edges += 1;
      }
    }
    graph.set(source.path, dependencies);
  }

  const sizedByPath = new Map(sizedSources.map((source) => [source.path, source.content]));
  const sizeViolations: string[] = [];
  const reducibleSizeDebt: string[] = [];
  for (const source of sizedSources) {
    const lines = countLines(source.content);
    const debt = budget.trackedDebt[source.path];
    const limit = debt?.ceiling ?? budget.defaultMaximumLines;
    if (lines > limit) sizeViolations.push(`${source.path}:${lines}>${limit}`);
    if (debt && lines <= debt.target && debt.ceiling > lines) reducibleSizeDebt.push(`${source.path}:${lines}<${debt.ceiling}:target=${debt.target}`);
  }
  const staleSizeDebt = Object.keys(budget.trackedDebt)
    .filter((file) => !sizedByPath.has(file))
    .sort();

  return {
    files: graphSources.length,
    edges,
    unknownAreas: [...unknownAreas].sort(),
    cycles: stronglyConnectedComponents(graph),
    sizeViolations: sizeViolations.sort(),
    staleSizeDebt,
    reducibleSizeDebt: reducibleSizeDebt.sort(),
  };
}
