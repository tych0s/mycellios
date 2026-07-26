export interface WindowsReleaseRecord {
  sha1: string;
  name: string;
  size: bigint;
}

export interface ArchivePathEvidence {
  path: string;
  directory: boolean;
}

export interface EmbeddedZipEvidence {
  bytes: bigint;
  sha256: string;
  buffer?: Buffer;
}

export function verifyWindowsRelease(workspaceRoot: string): Promise<void>;
export function assertNativePythonManifest(
  manifest: unknown,
  source: string,
): void;
export function parseNuspec(
  bytes: Uint8Array,
  source: string,
): { id: string; version: string };
export function parseReleases(
  bytes: Uint8Array,
  releasesPath: string,
): WindowsReleaseRecord;
export function validateArchivePath(rawEntry: string): ArchivePathEvidence;
export function readZipEvidence(
  buffer: Buffer,
  expectedNames: Set<string>,
): Promise<Map<string, EmbeddedZipEvidence>>;
