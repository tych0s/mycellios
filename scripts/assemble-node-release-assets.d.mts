export interface AssembleNodeReleaseAssetsInput {
  artifactsRoot: string;
  outputRoot: string;
  version: string;
  revision: string;
  sourceId: string;
  publishedAt: string;
}

export interface AssembledNodeReleaseAssets {
  outputRoot: string;
  version: string;
  revision: string;
  sourceId: string;
  signatureState: "pending" | "signed";
  packages: Array<{ target: string; name: string; bytes: number; sha256: string }>;
}

export function assembleNodeReleaseAssets(
  input: AssembleNodeReleaseAssetsInput,
): Promise<AssembledNodeReleaseAssets>;
