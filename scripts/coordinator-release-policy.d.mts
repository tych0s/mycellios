export interface CoordinatorReleaseManifestEntry {
  path: string;
  bytes: number;
  sha256: string;
}

export interface CoordinatorReleaseManifest {
  schema: "mycellios-native-coordinator-release/3";
  entrypoint: "dist/coordinator/main.js";
  revision: string;
  version: string;
  sourceId: `sha256:${string}`;
  releaseId: `sha256:${string}`;
  files: CoordinatorReleaseManifestEntry[];
}

export const COORDINATOR_RELEASE_SCHEMA:
  "mycellios-native-coordinator-release/3";
export const COORDINATOR_RELEASE_MANIFEST:
  "mycellios-coordinator-release-manifest.json";
export interface CoordinatorOutputReceiptFile {
  path: string;
  bytes: number;
  sha256: string;
}
export interface CoordinatorOutputReceiptOutput {
  path: "dist" | "landing-dist" | "mobile-dist";
  selection: "coordinator-js-closure" | "complete-tree";
  outputId: `sha256:${string}`;
  files: CoordinatorOutputReceiptFile[];
}
export interface CoordinatorOutputReceipt {
  schema: "mycellios-native-coordinator-output-receipt/1";
  sourceId: `sha256:${string}`;
  receiptId: `sha256:${string}`;
  outputs: CoordinatorOutputReceiptOutput[];
}
export const COORDINATOR_OUTPUT_RECEIPT_SCHEMA:
  "mycellios-native-coordinator-output-receipt/1";
export const COORDINATOR_OUTPUT_RECEIPT:
  "mycellios-coordinator-output-receipt.json";
export const COORDINATOR_WORKSPACE_OUTPUT_RECEIPT:
  "build/mycellios-coordinator-output-receipt.json";
export function buildCoordinatorOutputReceipt(
  workspaceRoot: string,
): CoordinatorOutputReceipt;
export function writeCoordinatorOutputReceipt(
  workspaceRoot: string,
  receiptPath?: string,
): CoordinatorOutputReceipt;
export function verifyCoordinatorOutputReceiptDocument(
  candidate: unknown,
): CoordinatorOutputReceipt;
export function assertCoordinatorOutputReceiptMatches(
  workspaceRoot: string,
  candidate: unknown,
): CoordinatorOutputReceipt;
export function prepareCoordinatorRelease(
  workspaceRoot: string,
  destinationRoot: string,
  options: {
    revision: string;
    populateProductionDependencies: (destinationRoot: string) => void;
  },
): CoordinatorReleaseManifest;
export function verifyCoordinatorReleaseDirectory(
  releaseRoot: string,
): CoordinatorReleaseManifest;
export function buildCoordinatorReleaseManifest(
  releaseRoot: string,
): CoordinatorReleaseManifest;
