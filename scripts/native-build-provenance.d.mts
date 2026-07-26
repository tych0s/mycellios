export const NATIVE_BUILD_PROVENANCE_SCHEMA:
  "mycellios-native-build-provenance/1";
export const NATIVE_BUILD_PROVENANCE_FILE:
  "mycellios-native-build-provenance.json";

export interface NativeBuildProvenance {
  schema: typeof NATIVE_BUILD_PROVENANCE_SCHEMA;
  version: string;
  sourceId: `sha256:${string}`;
  files: Array<{
    path: string;
    bytes: number;
    sha256: string;
  }>;
}

export function buildNativeSourceProvenance(
  workspaceRoot: string,
): NativeBuildProvenance;
export function verifyNativeBuildProvenanceDocument(
  candidate: unknown,
): NativeBuildProvenance;
export function assertNativeSourceProvenanceMatches(
  workspaceRoot: string,
  candidate: unknown,
): NativeBuildProvenance;
