export type InstallerTreeEntry =
  | { kind: "directory"; mode: number | null }
  | { kind: "file"; mode: number | null; bytes: number; sha256: string }
  | { kind: "symlink"; mode: number | null; target: string };

export type DmgRootEntry =
  | { kind: "directory"; mode: number | null }
  | { kind: "file"; mode: number | null }
  | { kind: "symlink"; mode: number | null; target: string }
  | { kind: "special" };

export function collectTreeEvidence(
  root: string,
): Promise<Map<string, InstallerTreeEntry>>;
export function assertInstallerPackageTreeMatches(
  expectedRoot: string,
  actualRoot: string,
): Promise<void>;
export function assertExactDmgRootInventory(
  rootEntries: Map<string, DmgRootEntry>,
  appName: string,
): void;
export function assertExactDmgInstallerPayload(
  expectedAppRoot: string,
  mountedDmgRoot: string,
): Promise<void>;
export function assertDebianControlArchiveSafe(
  controlRoot: string,
): Promise<void>;
export function parseDebianControl(text: string): Map<string, string>;
export function expectedDebianControlFields(options: {
  sourceApp: string;
  expectedVersion: string;
  expectedArch: string;
}): Promise<Map<string, string>>;
export function assertDebianMd5Sums(
  md5Path: string,
  payloadRoot: string,
): Promise<void>;
export function assertDebianPackageMetadata(options: {
  controlRoot: string;
  payloadRoot: string;
  sourceApp: string;
  expectedVersion: string;
  expectedArch: string;
}): Promise<void>;
export function assertRpmMetadataEvidence(options: {
  headerOutput: string;
  fileOutput: string;
  expectedEntries: Map<string, InstallerTreeEntry>;
  expectedVersion: string;
  expectedArch: string;
}): void;
export function assertRpmPackageMetadata(options: {
  rpmPackage: string;
  expectedRoot: string;
  expectedVersion: string;
  expectedArch: string;
}): Promise<void>;
export function portablePermissionMode(
  mode: number,
  platform?: NodeJS.Platform,
): number | null;
