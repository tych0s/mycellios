import { createHash } from "node:crypto";
import {
  closeSync,
  constants,
  existsSync,
  fchmodSync,
  fstatSync,
  fsyncSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
  type Stats,
} from "node:fs";
import {
  dirname,
  isAbsolute,
  join,
  relative,
  resolve,
} from "node:path";
import { gunzipSync, gzipSync } from "node:zlib";

import {
  canonicalEvidenceJson,
  sha256CanonicalEvidence,
} from "../core/json.js";

export const COMPONENT_FILES_PACKAGE_SCHEMA =
  "mycellios-component-files/1" as const;
export const COMPONENT_FILES_MANIFEST_SCHEMA =
  "mycellios-component-files-manifest/1" as const;
export const COMPONENT_FILES_FORMAT = "json-gzip-v1" as const;

export type ComponentFileMode = "0644" | "0755";

export interface ComponentFileManifestEntry {
  path: string;
  mode: ComponentFileMode;
  bytes: number;
  sha256: `sha256:${string}`;
}

export interface ComponentFilePackageEntry extends ComponentFileManifestEntry {
  base64: string;
}

export interface ComponentFilesPackageDocument {
  schema: typeof COMPONENT_FILES_PACKAGE_SCHEMA;
  format: typeof COMPONENT_FILES_FORMAT;
  filesManifestSha256: `sha256:${string}`;
  files: ComponentFilePackageEntry[];
}

export interface ComponentFilesLimits {
  maxFiles: number;
  maxFileBytes: number;
  maxTotalFileBytes: number;
  maxCompressedBytes: number;
  maxDocumentBytes: number;
  maxPathBytes: number;
}

export interface ComponentFilesBuildOptions {
  limits?: Partial<ComponentFilesLimits> | undefined;
  /**
   * Portable paths that must be restored as executable. This is useful when a
   * package for Linux or macOS is assembled on Windows, where source mode bits
   * are not meaningful.
   */
  executablePaths?: readonly string[] | undefined;
}

export interface ComponentFilesVerificationOptions {
  limits?: Partial<ComponentFilesLimits> | undefined;
  expectedFilesManifestSha256?: `sha256:${string}` | undefined;
}

export interface BuiltComponentFilesPackage {
  format: typeof COMPONENT_FILES_FORMAT;
  packageBytes: Buffer;
  artifactSha256: `sha256:${string}`;
  filesManifestSha256: `sha256:${string}`;
  files: ComponentFileManifestEntry[];
  fileCount: number;
  totalFileBytes: number;
  documentBytes: number;
}

export interface InspectedComponentFilesPackage {
  format: typeof COMPONENT_FILES_FORMAT;
  artifactSha256: `sha256:${string}`;
  filesManifestSha256: `sha256:${string}`;
  files: ComponentFileManifestEntry[];
  fileCount: number;
  totalFileBytes: number;
  documentBytes: number;
}

export interface ExtractedComponentFilesPackage
  extends InspectedComponentFilesPackage {
  stagingDirectory: string;
}

const DEFAULT_LIMITS: Readonly<ComponentFilesLimits> = Object.freeze({
  maxFiles: 16_384,
  maxFileBytes: 512 * 1024 * 1024,
  maxTotalFileBytes: 2 * 1024 * 1024 * 1024,
  maxCompressedBytes: 2 * 1024 * 1024 * 1024,
  maxDocumentBytes: 3 * 1024 * 1024 * 1024,
  maxPathBytes: 4_096,
});

const SHA256_IDENTITY = /^sha256:[0-9a-f]{64}$/;
const WINDOWS_DEVICE_NAME =
  /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$/iu;
const BASE64 =
  /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/u;

interface VerifiedPackage {
  inspection: InspectedComponentFilesPackage;
  decodedFiles: Array<{
    manifest: ComponentFileManifestEntry;
    contents: Buffer;
  }>;
}

/**
 * Builds a deterministic component artifact from one directory tree.
 *
 * Symbolic links and non-regular filesystem entries are rejected rather than
 * followed. Directories are represented implicitly by their child files.
 */
export function buildComponentFilesPackage(
  sourceDirectory: string,
  options: ComponentFilesBuildOptions = {},
): BuiltComponentFilesPackage {
  const limits = normalizeLimits(options.limits);
  const sourceRoot = requiredAbsoluteDirectory(sourceDirectory);
  const executablePaths = normalizeExecutablePaths(
    options.executablePaths ?? [],
    limits.maxPathBytes,
  );
  const packageEntries: ComponentFilePackageEntry[] = [];
  let totalFileBytes = 0;

  walkSourceDirectory(
    sourceRoot,
    "",
    limits.maxPathBytes,
    (portablePath, absolutePath) => {
      if (packageEntries.length >= limits.maxFiles) {
        throw new Error("component_files_file_limit_exceeded");
      }
      assertSafePortablePath(portablePath, limits.maxPathBytes);
      const { contents, stats } = readRegularFile(absolutePath);
      if (contents.length > limits.maxFileBytes) {
        throw new Error("component_files_individual_size_limit_exceeded");
      }
      totalFileBytes = checkedTotal(
        totalFileBytes,
        contents.length,
        limits.maxTotalFileBytes,
      );
      const mode: ComponentFileMode =
        executablePaths.has(portablePath) || (stats.mode & 0o111) !== 0
          ? "0755"
          : "0644";
      packageEntries.push({
        path: portablePath,
        mode,
        bytes: contents.length,
        sha256: sha256Identity(contents),
        base64: contents.toString("base64"),
      });
    },
  );

  if (packageEntries.length === 0) {
    throw new Error("component_files_package_is_empty");
  }
  packageEntries.sort(compareEntries);
  assertNoDuplicateOrConflictingPaths(packageEntries);
  for (const executablePath of executablePaths) {
    if (!packageEntries.some((entry) => entry.path === executablePath)) {
      throw new Error(
        `component_files_executable_path_is_missing:${executablePath}`,
      );
    }
  }

  const manifestEntries = packageEntries.map(logicalEntry);
  const filesManifestSha256 =
    computeFilesManifestSha256(manifestEntries, limits);
  const document: ComponentFilesPackageDocument = {
    schema: COMPONENT_FILES_PACKAGE_SCHEMA,
    format: COMPONENT_FILES_FORMAT,
    filesManifestSha256,
    files: packageEntries,
  };
  const canonicalDocument = Buffer.from(
    canonicalEvidenceJson(document),
    "utf8",
  );
  if (canonicalDocument.length > limits.maxDocumentBytes) {
    throw new Error("component_files_document_size_limit_exceeded");
  }
  const packageBytes = gzipSync(canonicalDocument, { level: 9 });
  if (packageBytes.length > limits.maxCompressedBytes) {
    throw new Error("component_files_compressed_size_limit_exceeded");
  }

  return {
    format: COMPONENT_FILES_FORMAT,
    packageBytes,
    artifactSha256: sha256Identity(packageBytes),
    filesManifestSha256,
    files: manifestEntries,
    fileCount: manifestEntries.length,
    totalFileBytes,
    documentBytes: canonicalDocument.length,
  };
}

/**
 * Parses and fully verifies a package without writing it to disk.
 */
export function inspectComponentFilesPackage(
  packageBytes: Uint8Array,
  limits?: Partial<ComponentFilesLimits>,
): InspectedComponentFilesPackage {
  return verifyPackage(packageBytes, { limits }).inspection;
}

/**
 * Verifies package structure, canonical encoding, every file digest and an
 * optional logical files-manifest identity supplied by a signed update
 * manifest.
 */
export function verifyComponentFilesPackage(
  packageBytes: Uint8Array,
  options: ComponentFilesVerificationOptions = {},
): InspectedComponentFilesPackage {
  return verifyPackage(packageBytes, options).inspection;
}

/**
 * Extracts a verified package into a newly created staging directory.
 *
 * Requiring a non-existent destination avoids overwrites and prevents an
 * existing symlink tree from influencing extraction. Any partial staging tree
 * created by this call is removed if extraction fails.
 */
export function extractComponentFilesPackage(
  packageBytes: Uint8Array,
  stagingDirectory: string,
  options: ComponentFilesVerificationOptions = {},
): ExtractedComponentFilesPackage {
  const verified = verifyPackage(packageBytes, options);
  const stagingRoot = requiredNewStagingPath(stagingDirectory);
  let stagingCreated = false;

  try {
    mkdirSync(stagingRoot, { mode: 0o700 });
    stagingCreated = true;

    for (const file of verified.decodedFiles) {
      const target = safeExtractionTarget(stagingRoot, file.manifest.path);
      createSafeParentDirectories(stagingRoot, dirname(target));
      writeNewRegularFile(target, file.contents, file.manifest.mode);
    }

    return {
      ...verified.inspection,
      stagingDirectory: stagingRoot,
    };
  } catch (error) {
    if (stagingCreated) {
      try {
        rmSync(stagingRoot, {
          recursive: true,
          force: true,
          maxRetries: 2,
          retryDelay: 25,
        });
      } catch {
        // Preserve the extraction error. The caller still knows the exact
        // staging path and can quarantine it if external interference won.
      }
    }
    throw error;
  }
}

/**
 * Computes the transport-independent identity referenced by
 * `filesManifestSha256`. File data is represented by size and digest; base64
 * and gzip encoding deliberately do not affect this identity.
 */
export function computeFilesManifestSha256(
  entries: readonly ComponentFileManifestEntry[],
  limitOverrides?: Partial<ComponentFilesLimits>,
): `sha256:${string}` {
  const limits = normalizeLimits(limitOverrides);
  if (entries.length === 0) {
    throw new Error("component_files_manifest_is_empty");
  }
  if (entries.length > limits.maxFiles) {
    throw new Error("component_files_file_limit_exceeded");
  }
  const normalized = entries.map((entry) =>
    validateLogicalEntry(entry, limits),
  );
  normalized.sort(compareEntries);
  assertNoDuplicateOrConflictingPaths(normalized);
  let total = 0;
  for (const entry of normalized) {
    total = checkedTotal(total, entry.bytes, limits.maxTotalFileBytes);
  }
  return sha256CanonicalEvidence({
    schema: COMPONENT_FILES_MANIFEST_SCHEMA,
    files: normalized,
  }) as `sha256:${string}`;
}

function verifyPackage(
  packageBytes: Uint8Array,
  options: ComponentFilesVerificationOptions,
): VerifiedPackage {
  const limits = normalizeLimits(options.limits);
  if (!(packageBytes instanceof Uint8Array)) {
    throw new TypeError("component_files_package_bytes_are_invalid");
  }
  if (packageBytes.byteLength === 0) {
    throw new Error("component_files_package_is_empty");
  }
  if (packageBytes.byteLength > limits.maxCompressedBytes) {
    throw new Error("component_files_compressed_size_limit_exceeded");
  }
  const artifactBytes = Buffer.from(
    packageBytes.buffer,
    packageBytes.byteOffset,
    packageBytes.byteLength,
  );

  let documentBytes: Buffer;
  try {
    documentBytes = gunzipSync(artifactBytes, {
      maxOutputLength: limits.maxDocumentBytes,
    });
  } catch {
    throw new Error("component_files_package_decompression_failed");
  }
  if (
    documentBytes.length === 0
    || documentBytes.length > limits.maxDocumentBytes
  ) {
    throw new Error("component_files_document_size_limit_exceeded");
  }

  let value: unknown;
  try {
    value = JSON.parse(documentBytes.toString("utf8")) as unknown;
  } catch {
    throw new Error("component_files_package_json_is_invalid");
  }
  let canonical: Buffer;
  try {
    canonical = Buffer.from(canonicalEvidenceJson(value), "utf8");
  } catch {
    throw new Error("component_files_package_json_is_invalid");
  }
  if (!canonical.equals(documentBytes)) {
    throw new Error("component_files_package_json_is_not_canonical");
  }

  const document = validatePackageDocument(value, limits);
  const decodedFiles: VerifiedPackage["decodedFiles"] = [];
  const manifestEntries: ComponentFileManifestEntry[] = [];
  let totalFileBytes = 0;

  for (const entry of document.files) {
    totalFileBytes = checkedTotal(
      totalFileBytes,
      entry.bytes,
      limits.maxTotalFileBytes,
    );
    const expectedBase64Length = Math.ceil(entry.bytes / 3) * 4;
    if (
      entry.base64.length !== expectedBase64Length
      || !BASE64.test(entry.base64)
    ) {
      throw new Error(`component_files_base64_is_invalid:${entry.path}`);
    }
    const contents = Buffer.from(entry.base64, "base64");
    if (
      contents.length !== entry.bytes
      || contents.toString("base64") !== entry.base64
    ) {
      throw new Error(`component_files_byte_count_mismatch:${entry.path}`);
    }
    if (sha256Identity(contents) !== entry.sha256) {
      throw new Error(`component_files_sha256_mismatch:${entry.path}`);
    }
    const manifest = logicalEntry(entry);
    manifestEntries.push(manifest);
    decodedFiles.push({ manifest, contents });
  }

  const computedManifestSha256 =
    computeFilesManifestSha256(manifestEntries, limits);
  if (document.filesManifestSha256 !== computedManifestSha256) {
    throw new Error("component_files_manifest_sha256_mismatch");
  }
  if (
    options.expectedFilesManifestSha256 !== undefined
    && (
      !SHA256_IDENTITY.test(options.expectedFilesManifestSha256)
      || options.expectedFilesManifestSha256 !== computedManifestSha256
    )
  ) {
    throw new Error("component_files_expected_manifest_sha256_mismatch");
  }

  return {
    inspection: {
      format: COMPONENT_FILES_FORMAT,
      artifactSha256: sha256Identity(artifactBytes),
      filesManifestSha256: computedManifestSha256,
      files: manifestEntries,
      fileCount: manifestEntries.length,
      totalFileBytes,
      documentBytes: documentBytes.length,
    },
    decodedFiles,
  };
}

function validatePackageDocument(
  value: unknown,
  limits: ComponentFilesLimits,
): ComponentFilesPackageDocument {
  const record = strictRecord(value, [
    "schema",
    "format",
    "filesManifestSha256",
    "files",
  ], "component_files_package_shape_is_invalid");
  if (record.schema !== COMPONENT_FILES_PACKAGE_SCHEMA) {
    throw new Error("component_files_package_schema_is_unsupported");
  }
  if (record.format !== COMPONENT_FILES_FORMAT) {
    throw new Error("component_files_package_format_is_unsupported");
  }
  if (
    typeof record.filesManifestSha256 !== "string"
    || !SHA256_IDENTITY.test(record.filesManifestSha256)
  ) {
    throw new Error("component_files_manifest_sha256_is_invalid");
  }
  if (!Array.isArray(record.files) || record.files.length === 0) {
    throw new Error("component_files_package_is_empty");
  }
  if (record.files.length > limits.maxFiles) {
    throw new Error("component_files_file_limit_exceeded");
  }

  const files = record.files.map((entry) =>
    validatePackageEntry(entry, limits),
  );
  for (let index = 1; index < files.length; index += 1) {
    if (compareEntries(files[index - 1]!, files[index]!) >= 0) {
      throw new Error("component_files_paths_are_not_canonical");
    }
  }
  assertNoDuplicateOrConflictingPaths(files);
  return {
    schema: COMPONENT_FILES_PACKAGE_SCHEMA,
    format: COMPONENT_FILES_FORMAT,
    filesManifestSha256:
      record.filesManifestSha256 as `sha256:${string}`,
    files,
  };
}

function validatePackageEntry(
  value: unknown,
  limits: ComponentFilesLimits,
): ComponentFilePackageEntry {
  const record = strictRecord(value, [
    "path",
    "mode",
    "bytes",
    "sha256",
    "base64",
  ], "component_files_entry_shape_is_invalid");
  const logical = validateLogicalEntry({
    path: record.path,
    mode: record.mode,
    bytes: record.bytes,
    sha256: record.sha256,
  }, limits);
  if (typeof record.base64 !== "string") {
    throw new Error("component_files_base64_is_invalid");
  }
  return { ...logical, base64: record.base64 };
}

function validateLogicalEntry(
  value: unknown,
  limits: ComponentFilesLimits,
): ComponentFileManifestEntry {
  const record = strictRecord(value, [
    "path",
    "mode",
    "bytes",
    "sha256",
  ], "component_files_manifest_entry_shape_is_invalid");
  if (typeof record.path !== "string") {
    throw new Error("component_files_path_is_invalid");
  }
  assertSafePortablePath(record.path, limits.maxPathBytes);
  if (record.mode !== "0644" && record.mode !== "0755") {
    throw new Error(`component_files_mode_is_invalid:${record.path}`);
  }
  if (
    typeof record.bytes !== "number"
    || !Number.isSafeInteger(record.bytes)
    || record.bytes < 0
  ) {
    throw new Error(`component_files_byte_count_is_invalid:${record.path}`);
  }
  if (record.bytes > limits.maxFileBytes) {
    throw new Error("component_files_individual_size_limit_exceeded");
  }
  if (
    typeof record.sha256 !== "string"
    || !SHA256_IDENTITY.test(record.sha256)
  ) {
    throw new Error(`component_files_sha256_is_invalid:${record.path}`);
  }
  return {
    path: record.path,
    mode: record.mode,
    bytes: record.bytes,
    sha256: record.sha256 as `sha256:${string}`,
  };
}

function strictRecord(
  value: unknown,
  keys: readonly string[],
  errorCode: string,
): Record<string, unknown> {
  if (
    typeof value !== "object"
    || value === null
    || Array.isArray(value)
    || Object.getPrototypeOf(value) !== Object.prototype
  ) {
    throw new Error(errorCode);
  }
  const actual = Object.keys(value).sort(compareText);
  const expected = [...keys].sort(compareText);
  if (
    actual.length !== expected.length
    || actual.some((key, index) => key !== expected[index])
  ) {
    throw new Error(errorCode);
  }
  return value as Record<string, unknown>;
}

function logicalEntry(
  entry: ComponentFilePackageEntry,
): ComponentFileManifestEntry {
  return {
    path: entry.path,
    mode: entry.mode,
    bytes: entry.bytes,
    sha256: entry.sha256,
  };
}

function normalizeLimits(
  overrides: Partial<ComponentFilesLimits> | undefined,
): ComponentFilesLimits {
  if (overrides !== undefined) {
    strictPartialLimits(overrides);
  }
  return {
    maxFiles: positiveSafeInteger(
      overrides?.maxFiles ?? DEFAULT_LIMITS.maxFiles,
      "maxFiles",
    ),
    maxFileBytes: positiveSafeInteger(
      overrides?.maxFileBytes ?? DEFAULT_LIMITS.maxFileBytes,
      "maxFileBytes",
    ),
    maxTotalFileBytes: positiveSafeInteger(
      overrides?.maxTotalFileBytes ?? DEFAULT_LIMITS.maxTotalFileBytes,
      "maxTotalFileBytes",
    ),
    maxCompressedBytes: positiveSafeInteger(
      overrides?.maxCompressedBytes ?? DEFAULT_LIMITS.maxCompressedBytes,
      "maxCompressedBytes",
    ),
    maxDocumentBytes: positiveSafeInteger(
      overrides?.maxDocumentBytes ?? DEFAULT_LIMITS.maxDocumentBytes,
      "maxDocumentBytes",
    ),
    maxPathBytes: positiveSafeInteger(
      overrides?.maxPathBytes ?? DEFAULT_LIMITS.maxPathBytes,
      "maxPathBytes",
    ),
  };
}

function strictPartialLimits(value: Partial<ComponentFilesLimits>): void {
  if (
    typeof value !== "object"
    || value === null
    || Array.isArray(value)
  ) {
    throw new TypeError("component_files_limits_are_invalid");
  }
  const allowed = new Set<keyof ComponentFilesLimits>([
    "maxFiles",
    "maxFileBytes",
    "maxTotalFileBytes",
    "maxCompressedBytes",
    "maxDocumentBytes",
    "maxPathBytes",
  ]);
  for (const key of Object.keys(value)) {
    if (!allowed.has(key as keyof ComponentFilesLimits)) {
      throw new Error(`component_files_limit_is_unknown:${key}`);
    }
  }
}

function positiveSafeInteger(value: unknown, name: string): number {
  if (
    typeof value !== "number"
    || !Number.isSafeInteger(value)
    || value <= 0
  ) {
    throw new Error(`component_files_limit_is_invalid:${name}`);
  }
  return value;
}

function requiredAbsoluteDirectory(sourceDirectory: string): string {
  if (typeof sourceDirectory !== "string" || sourceDirectory.length === 0) {
    throw new TypeError("component_files_source_directory_is_invalid");
  }
  const sourceRoot = resolve(sourceDirectory);
  let stats: Stats;
  try {
    stats = lstatSync(sourceRoot);
  } catch {
    throw new Error("component_files_source_directory_is_unavailable");
  }
  if (stats.isSymbolicLink() || !stats.isDirectory()) {
    throw new Error("component_files_source_root_must_be_a_real_directory");
  }
  return sourceRoot;
}

function walkSourceDirectory(
  absoluteDirectory: string,
  portableDirectory: string,
  maxPathBytes: number,
  onFile: (portablePath: string, absolutePath: string) => void,
): void {
  const children = readdirSync(absoluteDirectory, { withFileTypes: true })
    .sort((left, right) => compareText(left.name, right.name));
  for (const child of children) {
    const absolutePath = join(absoluteDirectory, child.name);
    const portablePath = portableDirectory.length === 0
      ? child.name
      : `${portableDirectory}/${child.name}`;
    const stats = lstatSync(absolutePath);
    if (stats.isSymbolicLink() || child.isSymbolicLink()) {
      throw new Error(`component_files_symbolic_link_is_forbidden:${portablePath}`);
    }
    if (stats.isDirectory() && child.isDirectory()) {
      assertSafePortablePath(portablePath, maxPathBytes);
      walkSourceDirectory(
        absolutePath,
        portablePath,
        maxPathBytes,
        onFile,
      );
      continue;
    }
    if (stats.isFile() && child.isFile()) {
      onFile(portablePath, absolutePath);
      continue;
    }
    throw new Error(`component_files_source_type_is_unsupported:${portablePath}`);
  }
}

function readRegularFile(
  absolutePath: string,
): { contents: Buffer; stats: Stats } {
  let descriptor: number | undefined;
  try {
    descriptor = openSync(
      absolutePath,
      constants.O_RDONLY | constants.O_NOFOLLOW,
    );
    const before = fstatSync(descriptor);
    if (!before.isFile()) {
      throw new Error("component_files_source_type_is_unsupported");
    }
    const pathStats = lstatSync(absolutePath);
    if (pathStats.isSymbolicLink() || !pathStats.isFile()) {
      throw new Error("component_files_source_changed_during_read");
    }
    if (!sameFileIdentity(before, pathStats)) {
      throw new Error("component_files_source_changed_during_read");
    }
    const contents = readFileSync(descriptor);
    const after = fstatSync(descriptor);
    if (
      !after.isFile()
      || after.size !== contents.length
      || before.size !== after.size
      || before.mtimeMs !== after.mtimeMs
      || before.ctimeMs !== after.ctimeMs
    ) {
      throw new Error("component_files_source_changed_during_read");
    }
    return { contents, stats: after };
  } finally {
    if (descriptor !== undefined) closeSync(descriptor);
  }
}

function sameFileIdentity(left: Stats, right: Stats): boolean {
  if (left.dev === 0 && left.ino === 0 && right.dev === 0 && right.ino === 0) {
    return true;
  }
  return left.dev === right.dev && left.ino === right.ino;
}

function normalizeExecutablePaths(
  paths: readonly string[],
  maxPathBytes: number,
): Set<string> {
  if (!Array.isArray(paths)) {
    throw new TypeError("component_files_executable_paths_are_invalid");
  }
  const normalized = new Set<string>();
  for (const path of paths) {
    if (typeof path !== "string") {
      throw new TypeError("component_files_executable_path_is_invalid");
    }
    assertSafePortablePath(path, maxPathBytes);
    if (normalized.has(path)) {
      throw new Error(`component_files_executable_path_is_duplicated:${path}`);
    }
    normalized.add(path);
  }
  return normalized;
}

function assertSafePortablePath(path: string, maxPathBytes: number): void {
  if (
    path.length === 0
    || Buffer.byteLength(path, "utf8") > maxPathBytes
    || path !== path.normalize("NFC")
    || path.includes("\\")
    || path.includes(":")
    || path.startsWith("/")
    || path.endsWith("/")
    || isAbsolute(path)
    || /^[A-Za-z]:/u.test(path)
    || /[\u0000-\u001f\u007f]/u.test(path)
  ) {
    throw new Error(`component_files_path_is_unsafe:${path}`);
  }
  const segments = path.split("/");
  for (const segment of segments) {
    if (
      segment.length === 0
      || segment === "."
      || segment === ".."
      || segment.endsWith(".")
      || segment.endsWith(" ")
      || WINDOWS_DEVICE_NAME.test(segment)
    ) {
      throw new Error(`component_files_path_is_unsafe:${path}`);
    }
  }
}

function assertNoDuplicateOrConflictingPaths(
  entries: readonly Pick<ComponentFileManifestEntry, "path">[],
): void {
  const portableKeys = new Set<string>();
  const platformKeys = new Set<string>();
  const sortedPaths = entries.map(({ path }) => path).sort(compareText);
  for (const path of sortedPaths) {
    if (portableKeys.has(path)) {
      throw new Error(`component_files_path_is_duplicated:${path}`);
    }
    portableKeys.add(path);

    // Windows extraction is case-insensitive in the common configuration.
    // Reject portable names that could alias there even when built on Linux.
    const platformKey = path.toLocaleLowerCase("en-US");
    if (platformKeys.has(platformKey)) {
      throw new Error(`component_files_path_is_ambiguous:${path}`);
    }
    platformKeys.add(platformKey);
  }
  for (let index = 1; index < sortedPaths.length; index += 1) {
    const previous = sortedPaths[index - 1]!;
    const current = sortedPaths[index]!;
    if (current.startsWith(`${previous}/`)) {
      throw new Error(`component_files_path_conflicts_with_file:${current}`);
    }
  }
  const foldedPaths = sortedPaths
    .map((path) => path.toLocaleLowerCase("en-US"))
    .sort(compareText);
  for (let index = 1; index < foldedPaths.length; index += 1) {
    const previous = foldedPaths[index - 1]!;
    const current = foldedPaths[index]!;
    if (current.startsWith(`${previous}/`)) {
      throw new Error(`component_files_path_is_ambiguous:${current}`);
    }
  }
}

function checkedTotal(current: number, next: number, maximum: number): number {
  const total = current + next;
  if (!Number.isSafeInteger(total) || total > maximum) {
    throw new Error("component_files_total_size_limit_exceeded");
  }
  return total;
}

function requiredNewStagingPath(stagingDirectory: string): string {
  if (typeof stagingDirectory !== "string" || stagingDirectory.length === 0) {
    throw new TypeError("component_files_staging_directory_is_invalid");
  }
  const stagingRoot = resolve(stagingDirectory);
  if (existsSync(stagingRoot)) {
    throw new Error("component_files_staging_directory_already_exists");
  }
  const parent = dirname(stagingRoot);
  let parentStats: Stats;
  try {
    parentStats = lstatSync(parent);
  } catch {
    throw new Error("component_files_staging_parent_is_unavailable");
  }
  if (parentStats.isSymbolicLink() || !parentStats.isDirectory()) {
    throw new Error("component_files_staging_parent_must_be_a_real_directory");
  }
  return stagingRoot;
}

function safeExtractionTarget(stagingRoot: string, portablePath: string): string {
  const target = resolve(stagingRoot, ...portablePath.split("/"));
  const traversal = relative(stagingRoot, target);
  if (
    traversal.length === 0
    || isAbsolute(traversal)
    || traversal === ".."
    || traversal.startsWith(`..${process.platform === "win32" ? "\\" : "/"}`)
  ) {
    throw new Error(`component_files_extraction_target_is_unsafe:${portablePath}`);
  }
  return target;
}

function createSafeParentDirectories(
  stagingRoot: string,
  targetParent: string,
): void {
  const traversal = relative(stagingRoot, targetParent);
  if (traversal.length === 0) return;
  if (isAbsolute(traversal) || traversal.startsWith("..")) {
    throw new Error("component_files_extraction_parent_is_unsafe");
  }
  let current = stagingRoot;
  for (const segment of traversal.split(/[\\/]/u)) {
    current = join(current, segment);
    if (!existsSync(current)) {
      mkdirSync(current, { mode: 0o755 });
      continue;
    }
    const stats = lstatSync(current);
    if (stats.isSymbolicLink() || !stats.isDirectory()) {
      throw new Error("component_files_extraction_parent_is_unsafe");
    }
  }
}

function writeNewRegularFile(
  target: string,
  contents: Buffer,
  mode: ComponentFileMode,
): void {
  const numericMode = mode === "0755" ? 0o755 : 0o644;
  let descriptor: number | undefined;
  try {
    descriptor = openSync(
      target,
      constants.O_WRONLY
        | constants.O_CREAT
        | constants.O_EXCL
        | constants.O_NOFOLLOW,
      numericMode,
    );
    writeFileSync(descriptor, contents);
    fchmodSync(descriptor, numericMode);
    fsyncSync(descriptor);
  } finally {
    if (descriptor !== undefined) closeSync(descriptor);
  }
}

function sha256Identity(value: Uint8Array): `sha256:${string}` {
  return `sha256:${createHash("sha256").update(value).digest("hex")}`;
}

function compareEntries(
  left: Pick<ComponentFileManifestEntry, "path">,
  right: Pick<ComponentFileManifestEntry, "path">,
): number {
  return compareText(left.path, right.path);
}

function compareText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
