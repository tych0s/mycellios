export interface CoordinatorRuntimeIdentity {
  revision: string;
  sourceId: `sha256:${string}`;
  version: string;
}

export function prepareCoordinatorRuntimeIdentity(
  sourceRoot: string,
  runtimeRoot: string,
  revision: string | undefined,
): CoordinatorRuntimeIdentity;
