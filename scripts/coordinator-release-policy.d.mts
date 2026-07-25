export interface CoordinatorReleaseManifestEntry {
  path: string;
  bytes: number;
  sha256: string;
}

export interface CoordinatorReleaseManifest {
  schema: "mycellios-native-coordinator-release/1";
  entrypoint: "dist/coordinator/main.js";
  files: CoordinatorReleaseManifestEntry[];
}

export const COORDINATOR_RELEASE_SCHEMA:
  "mycellios-native-coordinator-release/1";
export const COORDINATOR_RELEASE_MANIFEST:
  "mycellios-coordinator-release-manifest.json";
export function prepareCoordinatorRelease(
  workspaceRoot: string,
  destinationRoot: string,
  options: { revision: string },
): CoordinatorReleaseManifest;
export function verifyCoordinatorReleaseDirectory(
  releaseRoot: string,
): CoordinatorReleaseManifest;
export function buildCoordinatorReleaseManifest(
  releaseRoot: string,
): CoordinatorReleaseManifest;
