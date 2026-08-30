export const CANONICAL_DOCS: readonly string[];

export interface RepositoryStructureResult {
  packageManager: "npm";
  docs: number;
  areas: number;
}

export function verifyRepositoryStructure(root?: string): RepositoryStructureResult;
