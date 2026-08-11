import {
  constants,
  type Stats,
} from "node:fs";
import {
  lstat,
  mkdir,
  open,
  readFile,
  readdir,
  rename,
  rm,
  type FileHandle,
} from "node:fs/promises";
import { createHash, randomUUID } from "node:crypto";
import {
  basename,
  dirname,
  relative,
  parse as parsePath,
  resolve,
} from "node:path";

import {
  parseComponentUpdateManifest,
  verifyComponentUpdateManifest,
  type ComponentUpdateChannel,
  type ComponentUpdateComponent,
  type ComponentUpdateManifest,
  type PinnedComponentUpdateKey,
} from "../contracts/component-update-manifest.js";
import { canonicalEvidenceJson } from "../core/json.js";

const SHA256_IDENTITY = /^sha256:([0-9a-f]{64})$/;
const MAX_MANIFEST_BYTES = 2 * 1_024 * 1_024;
const READ_BUFFER_BYTES = 64 * 1_024;
const STAGING_SUFFIX = ".component-release-staging";

const PLATFORMS = ["any", "win32", "linux", "darwin"] as const;
const ARCHITECTURES = ["any", "x64", "arm64"] as const;

export type ComponentReleasePlatform = (typeof PLATFORMS)[number];
export type ComponentReleaseArchitecture = (typeof ARCHITECTURES)[number];

export interface ComponentReleaseTarget {
  channel: ComponentUpdateChannel;
  platform: ComponentReleasePlatform;
  arch: ComponentReleaseArchitecture;
}

export interface ComponentReleaseVerificationContext {
  target: ComponentReleaseTarget;
  minimumSequence: number | null;
}

export type ComponentReleaseManifestVerifier = (
  manifest: ComponentUpdateManifest,
  context: ComponentReleaseVerificationContext,
) => void | Promise<void>;

export type ComponentReleaseStoreVerification =
  | {
    pinnedKeys: Readonly<
      Partial<
        Record<
          ComponentUpdateChannel,
          PinnedComponentUpdateKey | readonly PinnedComponentUpdateKey[]
        >
      >
    >;
    verifyManifest?: never;
  }
  | {
    verifyManifest: ComponentReleaseManifestVerifier;
    pinnedKeys?: never;
  };

export type ComponentReleaseStoreOptions = {
  root: string;
  verification: ComponentReleaseStoreVerification;
};

export type ComponentReleaseArtifactBody =
  | Uint8Array
  | AsyncIterable<Uint8Array>;

export interface PutComponentReleaseArtifactInput {
  sha256: string;
  bytes: number;
  body: ComponentReleaseArtifactBody;
}

export interface StoredComponentReleaseArtifact {
  sha256: `sha256:${string}`;
  bytes: number;
  alreadyStored: boolean;
}

export interface ReadComponentReleaseArtifactInput {
  sha256: string;
  bytes?: number | undefined;
}

export interface OpenComponentReleaseArtifact {
  handle: FileHandle;
  sha256: `sha256:${string}`;
  bytes: number;
}

export interface PublishComponentReleaseManifestInput {
  target: ComponentReleaseTarget;
  manifest: unknown;
}

export interface PublishedComponentReleaseManifest {
  manifest: ComponentUpdateManifest;
  target: ComponentReleaseTarget;
  alreadyPublished: boolean;
  previousManifestId: string | null;
}

export interface ComponentReleaseGarbageCollectionResult {
  referencedArtifacts: number;
  retainedUnreferencedArtifacts: number;
  removedArtifacts: number;
}

export type ComponentReleaseStoreErrorCode =
  | "component_release_store_configuration_invalid"
  | "component_release_target_invalid"
  | "component_release_artifact_identity_invalid"
  | "component_release_artifact_size_invalid"
  | "component_release_artifact_body_invalid"
  | "component_release_artifact_missing"
  | "component_release_artifact_path_unsafe"
  | "component_release_artifact_size_mismatch"
  | "component_release_artifact_hash_mismatch"
  | "component_release_artifact_io_failed"
  | "component_release_manifest_invalid"
  | "component_release_manifest_signature_key_missing"
  | "component_release_manifest_signature_rejected"
  | "component_release_manifest_channel_mismatch"
  | "component_release_manifest_target_empty"
  | "component_release_manifest_artifact_missing"
  | "component_release_manifest_artifact_invalid"
  | "component_release_manifest_rollback_rejected"
  | "component_release_manifest_sequence_conflict"
  | "component_release_manifest_missing"
  | "component_release_manifest_path_unsafe"
  | "component_release_manifest_too_large"
  | "component_release_manifest_corrupt"
  | "component_release_manifest_io_failed";

export class ComponentReleaseStoreError extends Error {
  override readonly name = "ComponentReleaseStoreError";

  constructor(
    readonly code: ComponentReleaseStoreErrorCode,
    options?: { cause?: unknown },
  ) {
    super(code, options);
  }
}

/**
 * Filesystem-backed release store for fast component channels.
 *
 * Artifacts are immutable and addressed by their SHA-256 identity. A channel
 * publication becomes visible only after its signed manifest and every
 * referenced artifact have passed verification.
 */
export class ComponentReleaseStore {
  readonly #root: string;
  readonly #verification: ComponentReleaseStoreVerification;
  #mutation: Promise<void> = Promise.resolve();

  constructor(options: ComponentReleaseStoreOptions) {
    if (
      typeof options?.root !== "string"
      || options.root.trim().length === 0
      || !isVerificationConfiguration(options.verification)
    ) {
      throw new ComponentReleaseStoreError(
        "component_release_store_configuration_invalid",
      );
    }
    const root = resolve(options.root);
    if (root === parsePath(root).root) {
      throw new ComponentReleaseStoreError(
        "component_release_store_configuration_invalid",
      );
    }
    this.#root = root;
    this.#verification = options.verification;
  }

  async putArtifact(
    input: PutComponentReleaseArtifactInput,
  ): Promise<StoredComponentReleaseArtifact> {
    return this.#exclusive(async () => {
      const identity = parseArtifactIdentity(input.sha256);
      const expectedBytes = parseArtifactSize(input.bytes);
      const target = this.#artifactPath(identity.hex);

      let existing: Awaited<ReturnType<typeof inspectArtifact>>;
      try {
        existing = await inspectArtifact(target, {
          root: this.#root,
          digest: identity.digest,
          expectedBytes,
          collect: false,
        });
      } catch (error) {
        if (error instanceof ComponentReleaseStoreError) throw error;
        throw new ComponentReleaseStoreError(
          "component_release_artifact_io_failed",
          { cause: error },
        );
      }
      if (existing !== null) {
        return {
          sha256: identity.digest,
          bytes: existing.bytes,
          alreadyStored: true,
        };
      }

      const staging = resolve(
        dirname(target),
        `.${basename(target)}.${randomUUID()}${STAGING_SUFFIX}`,
      );
      assertInside(this.#root, staging);
      let renamed = false;
      let alreadyStored = false;
      try {
        await ensureSafeStoreDirectory(
          this.#root,
          dirname(target),
          "component_release_artifact_path_unsafe",
        );
        const handle = await open(staging, "wx");
        try {
          const observed = await writeArtifactBody(
            handle,
            input.body,
            expectedBytes,
          );
          if (observed.bytes !== expectedBytes) {
            throw new ComponentReleaseStoreError(
              "component_release_artifact_size_mismatch",
            );
          }
          if (observed.digest !== identity.digest) {
            throw new ComponentReleaseStoreError(
              "component_release_artifact_hash_mismatch",
            );
          }
          await handle.sync();
        } finally {
          await handle.close();
        }

        try {
          await rename(staging, target);
          renamed = true;
        } catch (error) {
          // A second publisher may have won the same immutable digest race.
          const raced = await inspectArtifact(target, {
            root: this.#root,
            digest: identity.digest,
            expectedBytes,
            collect: false,
          });
          if (raced === null) throw error;
          alreadyStored = true;
        }
        // A durability failure after our rename is not a name collision and
        // must reach the publisher. A retry will then verify the immutable
        // target instead of silently claiming a durable write.
        if (renamed) await syncDirectory(dirname(target));
      } catch (error) {
        if (error instanceof ComponentReleaseStoreError) throw error;
        throw new ComponentReleaseStoreError(
          "component_release_artifact_io_failed",
          { cause: error },
        );
      } finally {
        if (!renamed) await rm(staging, { force: true }).catch(() => undefined);
      }

      return {
        sha256: identity.digest,
        bytes: expectedBytes,
        alreadyStored,
      };
    });
  }

  async readArtifact(
    input: ReadComponentReleaseArtifactInput,
  ): Promise<Buffer> {
    const identity = parseArtifactIdentity(input.sha256);
    const expectedBytes =
      input.bytes === undefined ? undefined : parseArtifactSize(input.bytes);
    try {
      const artifact = await inspectArtifact(
        this.#artifactPath(identity.hex),
        {
          root: this.#root,
          digest: identity.digest,
          expectedBytes,
          collect: true,
        },
      );
      if (artifact === null) {
        throw new ComponentReleaseStoreError(
          "component_release_artifact_missing",
        );
      }
      return artifact.body!;
    } catch (error) {
      if (error instanceof ComponentReleaseStoreError) throw error;
      throw new ComponentReleaseStoreError(
        "component_release_artifact_io_failed",
        { cause: error },
      );
    }
  }

  /**
   * Returns an immutable, path-safe blob for HTTP streaming. Publication has
   * already hashed it; clients independently enforce the digest from the
   * signed manifest, so a public GET need not rehash or buffer the whole file.
   */
  async openArtifact(
    input: ReadComponentReleaseArtifactInput,
  ): Promise<OpenComponentReleaseArtifact> {
    const identity = parseArtifactIdentity(input.sha256);
    const path = this.#artifactPath(identity.hex);
    try {
      if (
        !await safeStoreDirectoryExists(
          this.#root,
          dirname(path),
          "component_release_artifact_path_unsafe",
        )
      ) {
        throw new ComponentReleaseStoreError(
          "component_release_artifact_missing",
        );
      }
      const metadata = await safeRegularFileMetadata(
        path,
        "component_release_artifact_path_unsafe",
      );
      if (metadata === null) {
        throw new ComponentReleaseStoreError(
          "component_release_artifact_missing",
        );
      }
      if (
        input.bytes !== undefined
        && metadata.size !== parseArtifactSize(input.bytes)
      ) {
        throw new ComponentReleaseStoreError(
          "component_release_artifact_size_mismatch",
        );
      }
      const handle = await open(path, constants.O_RDONLY);
      try {
        const opened = await handle.stat();
        if (!opened.isFile() || !sameFile(metadata, opened)) {
          throw new ComponentReleaseStoreError(
            "component_release_artifact_path_unsafe",
          );
        }
        return {
          handle,
          sha256: identity.digest,
          bytes: opened.size,
        };
      } catch (error) {
        await handle.close();
        throw error;
      }
    } catch (error) {
      if (error instanceof ComponentReleaseStoreError) throw error;
      throw new ComponentReleaseStoreError(
        "component_release_artifact_io_failed",
        { cause: error },
      );
    }
  }

  async publishManifest(
    input: PublishComponentReleaseManifestInput,
  ): Promise<PublishedComponentReleaseManifest> {
    return this.#exclusive(async () => {
      const target = parseTarget(input.target);
      const manifest = parseManifest(input.manifest);
      this.#assertManifestTarget(manifest, target);

      const current = await this.#loadPublishedManifest(target, false);
      await this.#verifyManifest(
        manifest,
        target,
        current?.sequence ?? null,
      );
      assertMonotonicManifest(manifest, current);
      await this.#verifyManifestArtifacts(manifest);

      if (current?.manifestId === manifest.manifestId) {
        return {
          manifest,
          target,
          alreadyPublished: true,
          previousManifestId: current.manifestId,
        };
      }

      const targetPath = this.#manifestPath(target);
      const bytes = Buffer.from(
        `${canonicalEvidenceJson(manifest)}\n`,
        "utf8",
      );
      try {
        await atomicReplace(targetPath, bytes, this.#root);
      } catch (error) {
        throw new ComponentReleaseStoreError(
          "component_release_manifest_io_failed",
          { cause: error },
        );
      }
      return {
        manifest,
        target,
        alreadyPublished: false,
        previousManifestId: current?.manifestId ?? null,
      };
    });
  }

  async readPublishedManifest(
    targetInput: ComponentReleaseTarget,
    options: { verifyArtifacts?: boolean } = {},
  ): Promise<ComponentUpdateManifest | null> {
    const target = parseTarget(targetInput);
    // Manifests are replaced atomically, so a lock-free reader sees either the
    // previous complete release or the next complete release. Public polling
    // must not enqueue behind unrelated publishers or other readers.
    return this.#loadPublishedManifest(
      target,
      options.verifyArtifacts !== false,
    );
  }

  async collectGarbage(options: {
    retainUnreferenced?: number;
    minimumAgeMs?: number;
  } = {}): Promise<ComponentReleaseGarbageCollectionResult> {
    return this.#exclusive(async () => {
      const retainUnreferenced = options.retainUnreferenced ?? 32;
      const minimumAgeMs = options.minimumAgeMs ?? 6 * 60 * 60_000;
      if (
        !Number.isSafeInteger(retainUnreferenced)
        || retainUnreferenced < 0
        || !Number.isSafeInteger(minimumAgeMs)
        || minimumAgeMs < 0
      ) {
        throw new ComponentReleaseStoreError(
          "component_release_store_configuration_invalid",
        );
      }
      const referenced = new Set<string>();
      for (const channel of ["dev", "stable"] as const) {
        for (const platform of PLATFORMS) {
          for (const arch of ARCHITECTURES) {
            const manifest = await this.#loadPublishedManifest(
              { channel, platform, arch },
              false,
            );
            for (const component of manifest?.components ?? []) {
              referenced.add(component.artifact.sha256);
            }
          }
        }
      }
      const artifactRoot = resolve(this.#root, "artifacts", "sha256");
      assertInside(this.#root, artifactRoot);
      if (
        !await safeStoreDirectoryExists(
          this.#root,
          artifactRoot,
          "component_release_artifact_path_unsafe",
        )
      ) {
        return {
          referencedArtifacts: referenced.size,
          retainedUnreferencedArtifacts: 0,
          removedArtifacts: 0,
        };
      }
      const candidates: Array<{
        path: string;
        sha256: `sha256:${string}`;
        mtimeMs: number;
      }> = [];
      for (
        const prefix of await readdir(artifactRoot, { withFileTypes: true })
      ) {
        const prefixPath = resolve(artifactRoot, prefix.name);
        assertInside(this.#root, prefixPath);
        if (
          prefix.isSymbolicLink()
          || !prefix.isDirectory()
          || !/^[0-9a-f]{2}$/.test(prefix.name)
        ) {
          throw new ComponentReleaseStoreError(
            "component_release_artifact_path_unsafe",
          );
        }
        for (
          const entry of await readdir(prefixPath, { withFileTypes: true })
        ) {
          const match = /^([0-9a-f]{64})\.blob$/.exec(entry.name);
          const path = resolve(prefixPath, entry.name);
          assertInside(this.#root, path);
          if (!match || entry.isSymbolicLink() || !entry.isFile()) {
            throw new ComponentReleaseStoreError(
              "component_release_artifact_path_unsafe",
            );
          }
          const metadata = await lstat(path);
          const sha256 = `sha256:${match[1]}` as const;
          if (!referenced.has(sha256)) {
            candidates.push({ path, sha256, mtimeMs: metadata.mtimeMs });
          }
        }
      }
      candidates.sort((left, right) => right.mtimeMs - left.mtimeMs);
      let removedArtifacts = 0;
      const now = Date.now();
      for (
        let index = retainUnreferenced;
        index < candidates.length;
        index += 1
      ) {
        const candidate = candidates[index]!;
        if (now - candidate.mtimeMs < minimumAgeMs) continue;
        await rm(candidate.path, { force: true });
        removedArtifacts += 1;
      }
      return {
        referencedArtifacts: referenced.size,
        retainedUnreferencedArtifacts:
          candidates.length - removedArtifacts,
        removedArtifacts,
      };
    });
  }

  async #loadPublishedManifest(
    target: ComponentReleaseTarget,
    verifyArtifacts: boolean,
  ): Promise<ComponentUpdateManifest | null> {
    let bytes: Buffer;
    try {
      const path = this.#manifestPath(target);
      if (
        !await safeStoreDirectoryExists(
          this.#root,
          dirname(path),
          "component_release_manifest_path_unsafe",
        )
      ) {
        return null;
      }
      const opened = await readStableRegularFile(
        path,
        "component_release_manifest_path_unsafe",
        MAX_MANIFEST_BYTES,
      );
      if (opened === null) return null;
      bytes = opened;
    } catch (error) {
      if (error instanceof ComponentReleaseStoreError) throw error;
      throw new ComponentReleaseStoreError(
        "component_release_manifest_io_failed",
        { cause: error },
      );
    }

    let candidate: unknown;
    try {
      candidate = JSON.parse(bytes.toString("utf8"));
    } catch (error) {
      throw new ComponentReleaseStoreError(
        "component_release_manifest_corrupt",
        { cause: error },
      );
    }
    const manifest = parseManifest(candidate, true);
    this.#assertManifestTarget(manifest, target);
    await this.#verifyManifest(manifest, target, null);
    if (verifyArtifacts) await this.#verifyManifestArtifacts(manifest);
    return manifest;
  }

  async #verifyManifest(
    manifest: ComponentUpdateManifest,
    target: ComponentReleaseTarget,
    minimumSequence: number | null,
  ): Promise<void> {
    try {
      if ("verifyManifest" in this.#verification) {
        await this.#verification.verifyManifest(
          structuredClone(manifest),
          {
            target: { ...target },
            minimumSequence,
          },
        );
        return;
      }
      const configured = this.#verification.pinnedKeys[manifest.channel];
      const pinnedKeys = configured === undefined
        ? []
        : Array.isArray(configured)
          ? configured
          : [configured];
      const pinnedKey = pinnedKeys.find(
        (candidate) => candidate.keyId === manifest.keyId,
      );
      if (pinnedKey === undefined) {
        throw new ComponentReleaseStoreError(
          "component_release_manifest_signature_key_missing",
        );
      }
      verifyComponentUpdateManifest(manifest, {
        pinnedKey,
        expectedChannel: target.channel,
      });
    } catch (error) {
      if (
        error instanceof ComponentReleaseStoreError
        && error.code === "component_release_manifest_signature_key_missing"
      ) {
        throw error;
      }
      throw new ComponentReleaseStoreError(
        "component_release_manifest_signature_rejected",
        { cause: error },
      );
    }
  }

  async #verifyManifestArtifacts(
    manifest: ComponentUpdateManifest,
  ): Promise<void> {
    const checked = new Map<string, number>();
    for (const component of manifest.components) {
      const { sha256, bytes } = component.artifact;
      const previousBytes = checked.get(sha256);
      if (previousBytes !== undefined) {
        if (previousBytes !== bytes) {
          throw new ComponentReleaseStoreError(
            "component_release_manifest_artifact_invalid",
          );
        }
        continue;
      }
      const identity = parseArtifactIdentity(sha256);
      try {
        const artifact = await inspectArtifact(
          this.#artifactPath(identity.hex),
          {
            root: this.#root,
            digest: identity.digest,
            expectedBytes: bytes,
            collect: false,
          },
        );
        if (artifact === null) {
          throw new ComponentReleaseStoreError(
            "component_release_manifest_artifact_missing",
          );
        }
      } catch (error) {
        if (
          error instanceof ComponentReleaseStoreError
          && error.code === "component_release_manifest_artifact_missing"
        ) {
          throw error;
        }
        throw new ComponentReleaseStoreError(
          "component_release_manifest_artifact_invalid",
          { cause: error },
        );
      }
      checked.set(sha256, bytes);
    }
  }

  #assertManifestTarget(
    manifest: ComponentUpdateManifest,
    target: ComponentReleaseTarget,
  ): void {
    if (manifest.channel !== target.channel) {
      throw new ComponentReleaseStoreError(
        "component_release_manifest_channel_mismatch",
      );
    }
    if (
      !manifest.components.some((component) =>
        componentMatchesTarget(component, target))
    ) {
      throw new ComponentReleaseStoreError(
        "component_release_manifest_target_empty",
      );
    }
  }

  #artifactPath(hex: string): string {
    const path = resolve(
      this.#root,
      "artifacts",
      "sha256",
      hex.slice(0, 2),
      `${hex}.blob`,
    );
    assertInside(this.#root, path);
    return path;
  }

  #manifestPath(target: ComponentReleaseTarget): string {
    const path = resolve(
      this.#root,
      "channels",
      target.channel,
      target.platform,
      target.arch,
      "manifest.json",
    );
    assertInside(this.#root, path);
    return path;
  }

  #exclusive<T>(operation: () => Promise<T>): Promise<T> {
    const result = this.#mutation.then(operation, operation);
    this.#mutation = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }
}

function isVerificationConfiguration(
  value: unknown,
): value is ComponentReleaseStoreVerification {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const candidate = value as Record<string, unknown>;
  const hasCallback =
    typeof candidate.verifyManifest === "function"
    && candidate.pinnedKeys === undefined;
  const hasPinnedKeys =
    candidate.verifyManifest === undefined
    && !!candidate.pinnedKeys
    && typeof candidate.pinnedKeys === "object"
    && !Array.isArray(candidate.pinnedKeys);
  return hasCallback || hasPinnedKeys;
}

function parseTarget(value: ComponentReleaseTarget): ComponentReleaseTarget {
  if (
    !value
    || typeof value !== "object"
    || !["dev", "stable"].includes(value.channel)
    || !PLATFORMS.includes(value.platform)
    || !ARCHITECTURES.includes(value.arch)
  ) {
    throw new ComponentReleaseStoreError("component_release_target_invalid");
  }
  return {
    channel: value.channel,
    platform: value.platform,
    arch: value.arch,
  };
}

function parseManifest(
  value: unknown,
  stored = false,
): ComponentUpdateManifest {
  try {
    return parseComponentUpdateManifest(value);
  } catch (error) {
    throw new ComponentReleaseStoreError(
      stored
        ? "component_release_manifest_corrupt"
        : "component_release_manifest_invalid",
      { cause: error },
    );
  }
}

function parseArtifactIdentity(value: string): {
  digest: `sha256:${string}`;
  hex: string;
} {
  const match =
    typeof value === "string" ? SHA256_IDENTITY.exec(value) : null;
  if (match === null) {
    throw new ComponentReleaseStoreError(
      "component_release_artifact_identity_invalid",
    );
  }
  return {
    digest: value as `sha256:${string}`,
    hex: match[1]!,
  };
}

function parseArtifactSize(value: number): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new ComponentReleaseStoreError(
      "component_release_artifact_size_invalid",
    );
  }
  return value;
}

function assertMonotonicManifest(
  manifest: ComponentUpdateManifest,
  current: ComponentUpdateManifest | null,
): void {
  if (current === null) return;
  if (manifest.sequence < current.sequence) {
    throw new ComponentReleaseStoreError(
      "component_release_manifest_rollback_rejected",
    );
  }
  if (
    manifest.sequence === current.sequence
    && manifest.manifestId !== current.manifestId
  ) {
    throw new ComponentReleaseStoreError(
      "component_release_manifest_sequence_conflict",
    );
  }
}

function componentMatchesTarget(
  component: ComponentUpdateComponent,
  target: ComponentReleaseTarget,
): boolean {
  const platformMatches =
    target.platform === "any"
      ? component.platform === "any"
      : component.platform === "any" || component.platform === target.platform;
  const architectureMatches =
    target.arch === "any"
      ? component.arch === "any"
      : component.arch === "any" || component.arch === target.arch;
  return platformMatches && architectureMatches;
}

async function writeArtifactBody(
  handle: FileHandle,
  body: ComponentReleaseArtifactBody,
  maximumBytes: number,
): Promise<{ digest: `sha256:${string}`; bytes: number }> {
  const hash = createHash("sha256");
  let bytes = 0;
  for await (const chunk of artifactChunks(body)) {
    if (!(chunk instanceof Uint8Array) || chunk.byteLength === 0) {
      throw new ComponentReleaseStoreError(
        "component_release_artifact_body_invalid",
      );
    }
    bytes += chunk.byteLength;
    if (bytes > maximumBytes) {
      throw new ComponentReleaseStoreError(
        "component_release_artifact_size_mismatch",
      );
    }
    hash.update(chunk);
    await writeAll(handle, chunk);
  }
  return {
    digest: `sha256:${hash.digest("hex")}`,
    bytes,
  };
}

async function* artifactChunks(
  body: ComponentReleaseArtifactBody,
): AsyncGenerator<Uint8Array> {
  if (body instanceof Uint8Array) {
    yield body;
    return;
  }
  if (
    !body
    || typeof body !== "object"
    || !(Symbol.asyncIterator in body)
  ) {
    throw new ComponentReleaseStoreError(
      "component_release_artifact_body_invalid",
    );
  }
  for await (const chunk of body) yield chunk;
}

async function writeAll(
  handle: FileHandle,
  bytes: Uint8Array,
): Promise<void> {
  let offset = 0;
  while (offset < bytes.byteLength) {
    const { bytesWritten } = await handle.write(
      bytes,
      offset,
      bytes.byteLength - offset,
      null,
    );
    if (bytesWritten < 1) {
      throw new ComponentReleaseStoreError(
        "component_release_artifact_io_failed",
      );
    }
    offset += bytesWritten;
  }
}

async function inspectArtifact(
  path: string,
  options: {
    root: string;
    digest: `sha256:${string}`;
    expectedBytes?: number | undefined;
    collect: boolean;
  },
): Promise<{ bytes: number; body?: Buffer | undefined } | null> {
  if (
    !await safeStoreDirectoryExists(
      options.root,
      dirname(path),
      "component_release_artifact_path_unsafe",
    )
  ) {
    return null;
  }
  const metadata = await safeRegularFileMetadata(
    path,
    "component_release_artifact_path_unsafe",
  );
  if (metadata === null) return null;
  if (
    options.expectedBytes !== undefined
    && metadata.size !== options.expectedBytes
  ) {
    throw new ComponentReleaseStoreError(
      "component_release_artifact_size_mismatch",
    );
  }

  const handle = await open(path, constants.O_RDONLY);
  const chunks: Buffer[] = [];
  const hash = createHash("sha256");
  let bytes = 0;
  try {
    const opened = await handle.stat();
    if (!sameFile(metadata, opened) || !opened.isFile()) {
      throw new ComponentReleaseStoreError(
        "component_release_artifact_path_unsafe",
      );
    }
    const buffer = Buffer.allocUnsafe(READ_BUFFER_BYTES);
    while (true) {
      const { bytesRead } = await handle.read(
        buffer,
        0,
        buffer.length,
        null,
      );
      if (bytesRead === 0) break;
      const chunk = buffer.subarray(0, bytesRead);
      hash.update(chunk);
      if (options.collect) chunks.push(Buffer.from(chunk));
      bytes += bytesRead;
    }
    const final = await handle.stat();
    if (!sameFile(opened, final) || final.size !== bytes) {
      throw new ComponentReleaseStoreError(
        "component_release_artifact_path_unsafe",
      );
    }
  } finally {
    await handle.close();
  }

  if (
    options.expectedBytes !== undefined
    && bytes !== options.expectedBytes
  ) {
    throw new ComponentReleaseStoreError(
      "component_release_artifact_size_mismatch",
    );
  }
  if (`sha256:${hash.digest("hex")}` !== options.digest) {
    throw new ComponentReleaseStoreError(
      "component_release_artifact_hash_mismatch",
    );
  }
  return {
    bytes,
    ...(options.collect ? { body: Buffer.concat(chunks, bytes) } : {}),
  };
}

async function safeRegularFileMetadata(
  path: string,
  unsafeCode:
    | "component_release_artifact_path_unsafe"
    | "component_release_manifest_path_unsafe",
): Promise<Stats | null> {
  try {
    const metadata = await lstat(path);
    if (metadata.isSymbolicLink() || !metadata.isFile()) {
      throw new ComponentReleaseStoreError(unsafeCode);
    }
    return metadata;
  } catch (error) {
    if (isFileSystemError(error) && error.code === "ENOENT") return null;
    throw error;
  }
}

async function readStableRegularFile(
  path: string,
  unsafeCode:
    | "component_release_artifact_path_unsafe"
    | "component_release_manifest_path_unsafe",
  maximumBytes: number,
): Promise<Buffer | null> {
  const metadata = await safeRegularFileMetadata(path, unsafeCode);
  if (metadata === null) return null;
  if (metadata.size > maximumBytes) {
    throw new ComponentReleaseStoreError(
      "component_release_manifest_too_large",
    );
  }
  const handle = await open(path, constants.O_RDONLY);
  try {
    const opened = await handle.stat();
    if (!opened.isFile() || !sameFile(metadata, opened)) {
      throw new ComponentReleaseStoreError(unsafeCode);
    }
    const bytes = await handle.readFile();
    const final = await handle.stat();
    if (
      !sameFile(opened, final)
      || bytes.length !== opened.size
      || bytes.length > maximumBytes
    ) {
      throw new ComponentReleaseStoreError(
        "component_release_manifest_corrupt",
      );
    }
    return bytes;
  } finally {
    await handle.close();
  }
}

function sameFile(left: Stats, right: Stats): boolean {
  return (
    left.dev === right.dev
    && left.ino === right.ino
    && left.size === right.size
  );
}

async function atomicReplace(
  target: string,
  bytes: Buffer,
  root: string,
): Promise<void> {
  const directory = dirname(target);
  await ensureSafeStoreDirectory(
    root,
    directory,
    "component_release_manifest_path_unsafe",
  );
  const staging = resolve(
    directory,
    `.${basename(target)}.${randomUUID()}${STAGING_SUFFIX}`,
  );
  assertInside(root, staging);
  let renamed = false;
  try {
    const handle = await open(staging, "wx");
    try {
      await writeAll(handle, bytes);
      await handle.sync();
    } finally {
      await handle.close();
    }
    await rename(staging, target);
    renamed = true;
    await syncDirectory(directory);
  } finally {
    if (!renamed) await rm(staging, { force: true }).catch(() => undefined);
  }
}

async function syncDirectory(path: string): Promise<void> {
  let handle: FileHandle | null = null;
  try {
    handle = await open(path, "r");
    await handle.sync();
  } catch (error) {
    if (
      process.platform === "win32"
      && isFileSystemError(error)
      && ["EINVAL", "ENOTSUP", "EPERM"].includes(error.code ?? "")
    ) {
      return;
    }
    throw error;
  } finally {
    await handle?.close();
  }
}

async function ensureSafeStoreDirectory(
  root: string,
  target: string,
  unsafeCode:
    | "component_release_artifact_path_unsafe"
    | "component_release_manifest_path_unsafe",
): Promise<void> {
  assertInside(root, target);
  await mkdir(root, { recursive: true });
  assertDirectory(await lstat(root), unsafeCode);

  const portable = relative(root, target);
  if (portable === "") return;
  let current = root;
  for (const segment of portable.split(/[\\/]/u)) {
    if (segment === "" || segment === "." || segment === "..") {
      throw new ComponentReleaseStoreError(unsafeCode);
    }
    current = resolve(current, segment);
    try {
      await mkdir(current);
    } catch (error) {
      if (!isFileSystemError(error) || error.code !== "EEXIST") throw error;
    }
    assertDirectory(await lstat(current), unsafeCode);
  }
}

async function safeStoreDirectoryExists(
  root: string,
  target: string,
  unsafeCode:
    | "component_release_artifact_path_unsafe"
    | "component_release_manifest_path_unsafe",
): Promise<boolean> {
  assertInside(root, target);
  const portable = relative(root, target);
  const directories = portable === ""
    ? [root]
    : [
      root,
      ...portable
        .split(/[\\/]/u)
        .reduce<string[]>((paths, segment) => {
          const parent = paths.at(-1) ?? root;
          paths.push(resolve(parent, segment));
          return paths;
        }, []),
    ];
  for (const directory of directories) {
    try {
      assertDirectory(await lstat(directory), unsafeCode);
    } catch (error) {
      if (isFileSystemError(error) && error.code === "ENOENT") return false;
      throw error;
    }
  }
  return true;
}

function assertDirectory(
  metadata: Stats,
  unsafeCode:
    | "component_release_artifact_path_unsafe"
    | "component_release_manifest_path_unsafe",
): void {
  if (!metadata.isDirectory() || metadata.isSymbolicLink()) {
    throw new ComponentReleaseStoreError(unsafeCode);
  }
}

function assertInside(root: string, candidate: string): void {
  const prefix = root.endsWith("\\") || root.endsWith("/")
    ? root
    : `${root}${process.platform === "win32" ? "\\" : "/"}`;
  if (candidate !== root && !candidate.startsWith(prefix)) {
    throw new ComponentReleaseStoreError(
      "component_release_store_configuration_invalid",
    );
  }
}

function isFileSystemError(
  error: unknown,
): error is NodeJS.ErrnoException {
  return error instanceof Error && "code" in error;
}
