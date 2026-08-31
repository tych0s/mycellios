export interface WebBundleSurfaceDefinition {
  outputDirectory: string;
  sourceMapsAllowed: boolean;
  defaultMaximumBytes: number;
  budgets: Record<string, { match: string; maximumBytes: number }>;
  criticalPaths: Record<string, { entry: string; highPriorityMatches: string[]; maximumBytes: number }>;
}
export interface WebBundleBudgetResult {
  surface?: string;
  kind: "asset" | "default" | "critical-path";
  role: string;
  file: string;
  bytes: number;
  maximumBytes: number;
}
export function verifyWebBundleBudget(input: {
  root?: string;
  outputDirectory: string;
  manifest: Record<string, { file?: string; imports?: string[] }>;
  definition: WebBundleSurfaceDefinition;
}): WebBundleBudgetResult[];
export function loadAndVerifyWebBundleBudget(root?: string, selectedSurface?: string): WebBundleBudgetResult[];
