import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  NATIVE_BUILD_PROVENANCE_FILE,
  nativeBuildProvenanceDocumentSchema,
  type NativeBuildIdentity,
  type NativeBuildProvenanceDocument,
} from "../contracts/build-identity.js";

export interface NativeRuntimeBuildMetadata {
  root: string;
  version: string;
  revision: string | null;
  buildIdentity: NativeBuildIdentity | null;
}

/**
 * Reads build metadata from one explicit application root. It never searches
 * process.cwd() or a parent directory, so a service cannot accidentally report
 * the identity of a sibling checkout.
 *
 * Source checkouts legitimately have no generated provenance or REVISION.
 * When either file is present it is treated as release evidence and therefore
 * must be valid; malformed evidence fails closed instead of becoming `null`.
 */
export function readNativeRuntimeBuildMetadata(
  runtimeRoot: string,
): NativeRuntimeBuildMetadata {
  const root = resolve(runtimeRoot);
  const packagePath = resolve(root, "package.json");
  let packageMetadata: unknown;
  try {
    packageMetadata = JSON.parse(readFileSync(packagePath, "utf8"));
  } catch (error) {
    throw new Error(
      `Cannot read native runtime package metadata at ${packagePath}: ${errorText(error)}`,
    );
  }
  if (
    packageMetadata === null
    || typeof packageMetadata !== "object"
    || Array.isArray(packageMetadata)
    || typeof (packageMetadata as { version?: unknown }).version !== "string"
  ) {
    throw new Error("Native runtime package metadata does not contain a version.");
  }
  const version = (packageMetadata as { version: string }).version.trim();
  if (
    version.length < 1
    || version.length > 128
    || /[\u0000-\u001f\u007f]/u.test(version)
  ) {
    throw new Error("Native runtime package version is invalid.");
  }

  const provenancePath = resolve(root, NATIVE_BUILD_PROVENANCE_FILE);
  const buildIdentity = existsSync(provenancePath)
    ? readNativeBuildIdentity(provenancePath, version)
    : null;

  const revisionPath = resolve(root, "REVISION");
  let revision: string | null = null;
  if (existsSync(revisionPath)) {
    revision = readFileSync(revisionPath, "utf8").trim().toLowerCase();
    if (!/^[0-9a-f]{40}$/.test(revision)) {
      throw new Error(
        `Native runtime revision at ${revisionPath} is not an exact Git SHA.`,
      );
    }
  }

  return { root, version, revision, buildIdentity };
}

export function readNativeBuildIdentity(
  provenancePath: string,
  expectedVersion?: string,
): NativeBuildIdentity {
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(provenancePath, "utf8"));
  } catch (error) {
    throw new Error(
      `Cannot read native build provenance at ${provenancePath}: ${errorText(error)}`,
    );
  }
  const document = nativeBuildProvenanceDocumentSchema.parse(
    parsed,
  ) as NativeBuildProvenanceDocument;
  assertStrictlySortedUniqueFiles(document);
  const identityDocument = {
    schema: document.schema,
    version: document.version,
    files: document.files,
  };
  const expectedSourceId =
    `sha256:${createHash("sha256")
      .update(Buffer.from(canonicalJson(identityDocument)))
      .digest("hex")}` as const;
  if (document.sourceId !== expectedSourceId) {
    throw new Error(
      `Native build provenance is not self-consistent: expected ${expectedSourceId}.`,
    );
  }
  if (expectedVersion !== undefined && document.version !== expectedVersion) {
    throw new Error(
      `Native build provenance version ${document.version} does not match runtime ${expectedVersion}.`,
    );
  }
  return {
    schema: document.schema,
    version: document.version,
    sourceId: document.sourceId,
  };
}

function assertStrictlySortedUniqueFiles(
  document: NativeBuildProvenanceDocument,
): void {
  let previous: string | null = null;
  for (const entry of document.files) {
    if (previous !== null && previous >= entry.path) {
      throw new Error(
        "Native build provenance file paths are not unique and strictly sorted.",
      );
    }
    previous = entry.path;
  }
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map((entry) => canonicalJson(entry)).join(",")}]`;
  }
  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
