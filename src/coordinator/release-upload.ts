import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import {
  type FileHandle,
  lstat,
  mkdir,
  open,
  readFile,
  readdir,
  rename,
  rm,
  stat,
} from "node:fs/promises";
import { basename, dirname, join, relative, resolve, sep } from "node:path";
import { TextDecoder } from "node:util";

export type ReleaseAssetChannel = "updates" | "downloads";

export interface ReleaseChunkMetadata {
  chunkIndex: number;
  chunkCount: number;
  chunkSha256: string;
  fileSha256: string;
  fileSize: number;
}

export interface StoredReleaseChunk {
  complete: boolean;
  receivedChunk: number;
  chunkCount: number;
  fileSha256: string;
}

export interface ReleaseTransactionIdentity {
  transactionId: string;
  sourceId: `sha256:${string}`;
  revision: string;
  version: string;
}

export interface ReleaseAssetEvidence {
  channel: ReleaseAssetChannel;
  fileName: string;
  fileSize: number;
  fileSha256: string;
}

export interface ReleaseTransactionManifest extends ReleaseTransactionIdentity {
  schema: "mycellios-native-public-release/1";
  assets: ReleaseAssetEvidence[];
  releaseId: `sha256:${string}`;
}

export interface ReleaseCommitResult {
  transactionId: string;
  releaseId: string;
  previousTransactionId: string | null;
  alreadyCommitted: boolean;
}

export interface ReleaseRollbackResult {
  transactionId: string;
  activeTransactionId: string | null;
  alreadyRolledBack: boolean;
}

export interface ReleaseAbortResult {
  transactionId: string;
  alreadyAborted: boolean;
}

const RELEASE_SCHEMA = "mycellios-native-public-release/1";
const TRANSACTION_SEAL_SCHEMA = "mycellios-native-release-transaction-seal/1";
const COMMIT_SCHEMA = "mycellios-native-release-commit/1";
const ACTIVE_SCHEMA = "mycellios-native-release-active/1";
const ROLLBACK_SCHEMA = "mycellios-native-release-rollback/1";

const UPDATE_FILE = /^(?:RELEASES|latest\.json|mycellios-setup\.exe|mycellios-\d+\.\d+\.\d+-full\.nupkg)$/;
const DOWNLOAD_FILE = /^mycellios-(?:windows-x64\.exe|macos-(?:arm64|x64)\.dmg|linux-x64\.(?:deb|rpm))$/;
const SHA256 = /^[0-9a-f]{64}$/;
const MAX_RELEASE_FILE_SIZE = 1_500_000_000;
const MAX_RELEASE_CHUNKS = 2_048;
export const MAX_RELEASE_CHUNK_SIZE_BYTES = 64 * 1024 * 1024;
const ABANDONED_TRANSACTION_TTL_MS = 7 * 24 * 60 * 60 * 1_000;
const TRANSACTION_ID = /^[A-Za-z0-9][A-Za-z0-9_-]{15,127}$/;
const SOURCE_ID = /^sha256:[0-9a-f]{64}$/;
const REVISION = /^[0-9a-f]{40}$/;
const VERSION = /^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)$/;

export function parseReleaseTransactionIdentity(
  headers: Record<string, unknown>,
): ReleaseTransactionIdentity {
  return validateTransactionIdentity({
    transactionId: header(headers, "x-release-transaction-id"),
    sourceId: header(headers, "x-release-source-id"),
    revision: header(headers, "x-release-revision"),
    version: header(headers, "x-release-version"),
  });
}

export function buildReleaseTransactionManifest(
  input: ReleaseTransactionIdentity & { assets: ReleaseAssetEvidence[] },
): ReleaseTransactionManifest {
  const identity = validateTransactionIdentity(input);
  const assets = validateManifestAssets(identity.version, input.assets);
  const sealed = {
    schema: RELEASE_SCHEMA,
    transactionId: identity.transactionId,
    sourceId: identity.sourceId,
    revision: identity.revision,
    version: identity.version,
    assets,
  } as const;
  return {
    ...sealed,
    releaseId: `sha256:${digest(Buffer.from(canonicalJson(sealed)))}`,
  };
}

export function parseReleaseTransactionManifest(
  candidate: unknown,
): ReleaseTransactionManifest {
  assertPlainObject(candidate, "release_manifest");
  assertExactKeys(
    candidate,
    [
      "assets",
      "releaseId",
      "revision",
      "schema",
      "sourceId",
      "transactionId",
      "version",
    ],
    "release_manifest",
  );
  if (candidate.schema !== RELEASE_SCHEMA) {
    throw new Error("release_manifest_schema_invalid");
  }
  if (typeof candidate.releaseId !== "string" || !SOURCE_ID.test(candidate.releaseId)) {
    throw new Error("release_manifest_release_id_invalid");
  }
  if (!Array.isArray(candidate.assets)) {
    throw new Error("release_manifest_assets_invalid");
  }
  const identity = validateTransactionIdentity(candidate);
  const expected = buildReleaseTransactionManifest({
    ...identity,
    assets: candidate.assets,
  });
  if (canonicalJson(candidate) !== canonicalJson(expected)) {
    throw new Error("release_manifest_seal_invalid");
  }
  return expected;
}

export class NativeReleaseTransactionStore {
  readonly #storageRoot: string;
  readonly #legacyRoots: Partial<Record<ReleaseAssetChannel, string>>;
  readonly #expected: Omit<ReleaseTransactionIdentity, "transactionId">;
  #active: ActiveRelease | null = null;
  #mutation: Promise<void> = Promise.resolve();

  constructor(input: {
    storageRoot: string;
    legacyUpdatesRoot?: string | null;
    legacyDownloadsRoot?: string | null;
    sourceId: string;
    revision: string;
    version: string;
  }) {
    this.#storageRoot = resolve(input.storageRoot);
    this.#expected = validateExpectedRuntime(input);
    this.#legacyRoots = {
      ...(input.legacyUpdatesRoot
        ? { updates: resolve(input.legacyUpdatesRoot) }
        : {}),
      ...(input.legacyDownloadsRoot
        ? { downloads: resolve(input.legacyDownloadsRoot) }
        : {}),
    };
  }

  async initialize(): Promise<void> {
    await this.#exclusive(async () => {
      const pointerPath = resolveInside(this.#storageRoot, "active.json");
      let pointer: ActivePointer | null = null;
      try {
        pointer = parseActivePointer(JSON.parse(await readFile(pointerPath, "utf8")));
      } catch (error) {
        if (!isMissing(error)) {
          throw new Error(
            `release_active_pointer_invalid:${error instanceof Error ? error.message : String(error)}`,
          );
        }
      }
      this.#active = pointer?.transactionId
        ? await this.#loadCommittedRelease(
          pointer.transactionId,
          pointer.releaseId,
        )
        : null;
      await this.#garbageCollectAbandonedTransactions(
        pointer?.transactionId ?? null,
      );
    });
  }

  async storeChunk(input: {
    identity: ReleaseTransactionIdentity;
    channel: ReleaseAssetChannel;
    fileName: string;
    metadata: ReleaseChunkMetadata;
    body: Buffer;
  }): Promise<StoredReleaseChunk> {
    return this.#exclusive(async () => {
      const identity = this.#assertExpectedIdentity(input.identity);
      assertExpectedAsset(identity.version, input.channel, input.fileName);
      const transactionRoot = this.#transactionRoot(identity.transactionId);
      await ensureSafeDirectory(transactionRoot);
      await writeExclusiveOrMatch(
        resolveInside(transactionRoot, "seal.json"),
        {
          schema: TRANSACTION_SEAL_SCHEMA,
          ...identity,
        },
        "release_transaction_identity_conflict",
      );
      const assetSealPath = resolveInside(
        transactionRoot,
        `asset-seals/${input.channel}/${input.fileName}.json`,
      );
      await writeExclusiveOrMatch(
        assetSealPath,
        {
          channel: input.channel,
          fileName: input.fileName,
          chunkCount: input.metadata.chunkCount,
          fileSha256: input.metadata.fileSha256,
          fileSize: input.metadata.fileSize,
        },
        "release_asset_identity_conflict",
      );
      return storeReleaseChunk({
        root: resolveInside(transactionRoot, `assets/${input.channel}`),
        channel: input.channel,
        fileName: input.fileName,
        metadata: input.metadata,
        body: input.body,
      });
    });
  }

  async commit(candidate: unknown): Promise<ReleaseCommitResult> {
    return this.#exclusive(async () => {
      const manifest = parseReleaseTransactionManifest(candidate);
      this.#assertExpectedIdentity(manifest);
      const transactionRoot = this.#transactionRoot(manifest.transactionId);
      await this.#verifyTransactionSeal(transactionRoot, manifest);
      await this.#verifyExactAssets(transactionRoot, manifest);

      const commitPath = resolveInside(transactionRoot, "commit.json");
      let previousTransactionId = this.#active?.manifest.transactionId ?? null;
      let previousReleaseId = this.#active?.manifest.releaseId ?? null;
      const existingCommit = await readOptionalJson(commitPath);
      if (existingCommit !== null) {
        const parsed = parseCommitDocument(existingCommit);
        if (canonicalJson(parsed.manifest) !== canonicalJson(manifest)) {
          throw new Error("release_transaction_commit_conflict");
        }
        previousTransactionId = parsed.previousTransactionId;
        previousReleaseId = parsed.previousReleaseId;
        if (this.#active?.manifest.transactionId === manifest.transactionId) {
          return {
            transactionId: manifest.transactionId,
            releaseId: manifest.releaseId,
            previousTransactionId,
            alreadyCommitted: true,
          };
        }
        if (
          await readOptionalJson(
            resolveInside(transactionRoot, "rollback.json"),
          ) !== null
        ) {
          throw new Error("release_transaction_not_recommittable");
        }
        const currentReleaseId = this.#active?.manifest.releaseId ?? null;
        if (
          (this.#active?.manifest.transactionId ?? null)
            === parsed.previousTransactionId
          && currentReleaseId === parsed.previousReleaseId
        ) {
          await this.#promote({
            transactionId: manifest.transactionId,
            releaseId: manifest.releaseId,
          });
          this.#active = {
            manifest,
            previousTransactionId,
            previousReleaseId,
          };
          return {
            transactionId: manifest.transactionId,
            releaseId: manifest.releaseId,
            previousTransactionId,
            alreadyCommitted: false,
          };
        }
        throw new Error("release_transaction_not_recommittable");
      }
      const commitDocument: CommitDocument = {
        schema: COMMIT_SCHEMA,
        manifest,
        previousTransactionId,
        previousReleaseId,
      };
      await writeExclusiveOrMatch(
        commitPath,
        commitDocument,
        "release_transaction_commit_conflict",
      );
      await this.#promote({
        transactionId: manifest.transactionId,
        releaseId: manifest.releaseId,
      });
      this.#active = { manifest, previousTransactionId, previousReleaseId };
      return {
        transactionId: manifest.transactionId,
        releaseId: manifest.releaseId,
        previousTransactionId,
        alreadyCommitted: false,
      };
    });
  }

  async rollback(transactionId: string): Promise<ReleaseRollbackResult> {
    return this.#exclusive(async () => {
      validateTransactionId(transactionId);
      const transactionRoot = this.#transactionRoot(transactionId);
      const commit = parseCommitDocument(
        await readRequiredJson(
          resolveInside(transactionRoot, "commit.json"),
          "release_transaction_not_committed",
        ),
      );
      const activeId = this.#active?.manifest.transactionId ?? null;
      if (activeId !== transactionId) {
        if (activeId === commit.previousTransactionId) {
          await writeExclusiveOrMatch(
            resolveInside(transactionRoot, "rollback.json"),
            rollbackDocument(transactionId, this.#active),
            "release_rollback_receipt_conflict",
          );
          return {
            transactionId,
            activeTransactionId: activeId,
            alreadyRolledBack: true,
          };
        }
        throw new Error("release_rollback_active_transaction_mismatch");
      }

      const restored = commit.previousTransactionId === null
        ? null
        : await this.#loadCommittedRelease(
          commit.previousTransactionId,
          commit.previousReleaseId,
        );
      await writeExclusiveOrMatch(
        resolveInside(transactionRoot, "rollback.json"),
        rollbackDocument(transactionId, restored),
        "release_rollback_receipt_conflict",
      );
      await this.#promote({
        transactionId: restored?.manifest.transactionId ?? null,
        releaseId: restored?.manifest.releaseId ?? null,
      });
      this.#active = restored;
      return {
        transactionId,
        activeTransactionId: restored?.manifest.transactionId ?? null,
        alreadyRolledBack: false,
      };
    });
  }

  async abort(transactionId: string): Promise<ReleaseAbortResult> {
    return this.#exclusive(async () => {
      validateTransactionId(transactionId);
      const durablePointer = await readOptionalJson(
        resolveInside(this.#storageRoot, "active.json"),
      );
      const durableActiveId = durablePointer === null
        ? null
        : parseActivePointer(durablePointer).transactionId;
      if (
        sameTransactionId(this.#active?.manifest.transactionId, transactionId)
        || sameTransactionId(durableActiveId, transactionId)
      ) {
        throw new Error("release_abort_active_transaction");
      }

      const transactionRoot = this.#transactionRoot(transactionId);
      let details;
      try {
        details = await lstat(transactionRoot);
      } catch (error) {
        if (isMissing(error)) {
          return { transactionId, alreadyAborted: true };
        }
        throw error;
      }
      if (!details.isDirectory() || details.isSymbolicLink()) {
        throw new Error("release_transaction_directory_unsafe");
      }
      if (
        await pathEntryExists(resolveInside(transactionRoot, "commit.json"))
        || await pathEntryExists(resolveInside(transactionRoot, "rollback.json"))
      ) {
        throw new Error("release_abort_finalized_transaction");
      }

      await rm(transactionRoot, { recursive: true });
      await syncDirectory(dirname(transactionRoot));
      return { transactionId, alreadyAborted: false };
    });
  }

  async publicAssetPath(
    channel: ReleaseAssetChannel,
    fileName: string,
  ): Promise<string | null> {
    try {
      validateReleaseAssetName(channel, fileName);
    } catch {
      return null;
    }
    if (this.#active) {
      const evidence = this.#active.manifest.assets.find(
        (asset) => asset.channel === channel && asset.fileName === fileName,
      );
      if (!evidence) return null;
      return resolveInside(
        this.#transactionRoot(this.#active.manifest.transactionId),
        `assets/${channel}/${fileName}`,
      );
    }
    const legacyRoot = this.#legacyRoots[channel];
    return legacyRoot ? resolveInside(legacyRoot, fileName) : null;
  }

  #assertExpectedIdentity(
    candidate: ReleaseTransactionIdentity,
  ): ReleaseTransactionIdentity {
    const identity = validateTransactionIdentity(candidate);
    if (identity.sourceId !== this.#expected.sourceId) {
      throw new Error("release_transaction_source_id_mismatch");
    }
    if (identity.revision !== this.#expected.revision) {
      throw new Error("release_transaction_revision_mismatch");
    }
    if (identity.version !== this.#expected.version) {
      throw new Error("release_transaction_version_mismatch");
    }
    return identity;
  }

  #transactionRoot(transactionId: string): string {
    validateTransactionId(transactionId);
    return resolveInside(this.#storageRoot, `transactions/${transactionId}`);
  }

  async #verifyTransactionSeal(
    transactionRoot: string,
    manifest: ReleaseTransactionManifest,
  ): Promise<void> {
    const seal = await readRequiredJson(
      resolveInside(transactionRoot, "seal.json"),
      "release_transaction_seal_missing",
    );
    const expected = {
      schema: TRANSACTION_SEAL_SCHEMA,
      transactionId: manifest.transactionId,
      sourceId: manifest.sourceId,
      revision: manifest.revision,
      version: manifest.version,
    };
    if (canonicalJson(seal) !== canonicalJson(expected)) {
      throw new Error("release_transaction_seal_mismatch");
    }
  }

  async #verifyExactAssets(
    transactionRoot: string,
    manifest: ReleaseTransactionManifest,
  ): Promise<void> {
    const assetsRoot = resolveInside(transactionRoot, "assets");
    const actualPaths = await listRegularFiles(assetsRoot);
    const expectedPaths = manifest.assets
      .map((asset) => `${asset.channel}/${asset.fileName}`)
      .sort();
    if (canonicalJson(actualPaths) !== canonicalJson(expectedPaths)) {
      throw new Error("release_transaction_asset_set_mismatch");
    }
    const sealPaths = await listRegularFiles(
      resolveInside(transactionRoot, "asset-seals"),
    );
    const expectedSealPaths = manifest.assets
      .map((asset) => `${asset.channel}/${asset.fileName}.json`)
      .sort();
    if (canonicalJson(sealPaths) !== canonicalJson(expectedSealPaths)) {
      throw new Error("release_transaction_asset_seal_set_mismatch");
    }
    const observed = new Map<string, FileEvidence>();
    for (const asset of manifest.assets) {
      const path = resolveInside(
        assetsRoot,
        `${asset.channel}/${asset.fileName}`,
      );
      const evidence = await inspectFile(path);
      if (
        evidence.bytes !== asset.fileSize
        || evidence.sha256 !== asset.fileSha256
      ) {
        throw new Error(`release_transaction_asset_mismatch:${asset.fileName}`);
      }
      observed.set(`${asset.channel}/${asset.fileName}`, evidence);
      const seal = await readRequiredJson(
        resolveInside(
          transactionRoot,
          `asset-seals/${asset.channel}/${asset.fileName}.json`,
        ),
        "release_asset_seal_missing",
      );
      if (
        !isPlainObject(seal)
        || seal.channel !== asset.channel
        || seal.fileName !== asset.fileName
        || seal.fileSha256 !== asset.fileSha256
        || seal.fileSize !== asset.fileSize
      ) {
        throw new Error(`release_asset_seal_mismatch:${asset.fileName}`);
      }
    }
    await verifyFeedCoherence(manifest, assetsRoot, observed);
  }

  async #loadCommittedRelease(
    transactionId: string,
    releaseId: `sha256:${string}` | null,
  ): Promise<ActiveRelease> {
    if (releaseId === null) throw new Error("release_active_release_id_missing");
    const transactionRoot = this.#transactionRoot(transactionId);
    const commit = parseCommitDocument(
      await readRequiredJson(
        resolveInside(transactionRoot, "commit.json"),
        "release_active_commit_missing",
      ),
    );
    if (commit.manifest.releaseId !== releaseId) {
      throw new Error("release_active_release_id_mismatch");
    }
    await this.#verifyTransactionSeal(transactionRoot, commit.manifest);
    await this.#verifyExactAssets(transactionRoot, commit.manifest);
    return {
      manifest: commit.manifest,
      previousTransactionId: commit.previousTransactionId,
      previousReleaseId: commit.previousReleaseId,
    };
  }

  async #garbageCollectAbandonedTransactions(
    activeTransactionId: string | null,
  ): Promise<void> {
    const transactionsRoot = resolveInside(this.#storageRoot, "transactions");
    let rootDetails;
    try {
      rootDetails = await lstat(transactionsRoot);
    } catch (error) {
      if (isMissing(error)) return;
      throw error;
    }
    if (!rootDetails.isDirectory() || rootDetails.isSymbolicLink()) {
      throw new Error("release_transaction_directory_unsafe");
    }

    const cutoff = Date.now() - ABANDONED_TRANSACTION_TTL_MS;
    for (const entry of await readdir(transactionsRoot, { withFileTypes: true })) {
      if (
        !TRANSACTION_ID.test(entry.name)
        || !entry.isDirectory()
        || entry.isSymbolicLink()
        || sameTransactionId(entry.name, activeTransactionId)
      ) {
        continue;
      }
      const transactionRoot = resolveInside(transactionsRoot, entry.name);
      let details;
      try {
        details = await lstat(transactionRoot);
      } catch (error) {
        if (isMissing(error)) continue;
        throw error;
      }
      if (!details.isDirectory() || details.isSymbolicLink()) continue;
      if (
        await pathEntryExists(resolveInside(transactionRoot, "commit.json"))
        || await pathEntryExists(resolveInside(transactionRoot, "rollback.json"))
      ) {
        continue;
      }
      const newestMtime = await newestSafeTreeMtime(transactionRoot);
      if (newestMtime === null || newestMtime >= cutoff) continue;

      const pointer = await readOptionalJson(
        resolveInside(this.#storageRoot, "active.json"),
      );
      if (
        pointer !== null
        && sameTransactionId(
          parseActivePointer(pointer).transactionId,
          entry.name,
        )
      ) {
        continue;
      }
      if (
        await pathEntryExists(resolveInside(transactionRoot, "commit.json"))
        || await pathEntryExists(resolveInside(transactionRoot, "rollback.json"))
      ) {
        continue;
      }
      let rechecked;
      try {
        rechecked = await lstat(transactionRoot);
      } catch (error) {
        if (isMissing(error)) continue;
        throw error;
      }
      if (
        !rechecked.isDirectory()
        || rechecked.isSymbolicLink()
        || rechecked.dev !== details.dev
        || rechecked.ino !== details.ino
      ) {
        continue;
      }
      const finalMtime = await newestSafeTreeMtime(transactionRoot);
      if (finalMtime === null || finalMtime >= cutoff) continue;

      try {
        await rm(transactionRoot, { recursive: true });
      } catch (error) {
        if (isMissing(error)) continue;
        throw error;
      }
      await syncDirectory(transactionsRoot);
    }
  }

  async #promote(input: {
    transactionId: string | null;
    releaseId: `sha256:${string}` | null;
  }): Promise<void> {
    await ensureSafeDirectory(this.#storageRoot);
    const pointer: ActivePointer = {
      schema: ACTIVE_SCHEMA,
      transactionId: input.transactionId,
      releaseId: input.releaseId,
    };
    await durableReplaceBytes(
      resolveInside(this.#storageRoot, "active.json"),
      jsonBytes(pointer),
    );
  }

  #exclusive<T>(operation: () => Promise<T>): Promise<T> {
    const result = this.#mutation.then(operation);
    this.#mutation = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }
}

export function validateReleaseAssetName(channel: ReleaseAssetChannel, fileName: string): void {
  const allowed = channel === "updates" ? UPDATE_FILE : DOWNLOAD_FILE;
  if (!allowed.test(fileName)) throw new Error("release_asset_name_not_allowed");
}

export function parseReleaseChunkMetadata(headers: Record<string, unknown>): ReleaseChunkMetadata {
  const chunkIndex = integerHeader(headers, "x-chunk-index", 0, MAX_RELEASE_CHUNKS - 1);
  const chunkCount = integerHeader(headers, "x-chunk-count", 1, MAX_RELEASE_CHUNKS);
  const fileSize = integerHeader(headers, "x-file-size", 1, MAX_RELEASE_FILE_SIZE);
  const chunkSha256 = shaHeader(headers, "x-chunk-sha256");
  const fileSha256 = shaHeader(headers, "x-file-sha256");
  if (chunkIndex >= chunkCount) throw new Error("release_chunk_index_out_of_range");
  return { chunkIndex, chunkCount, chunkSha256, fileSha256, fileSize };
}

export async function storeReleaseChunk(input: {
  root: string;
  channel: ReleaseAssetChannel;
  fileName: string;
  metadata: ReleaseChunkMetadata;
  body: Buffer;
}): Promise<StoredReleaseChunk> {
  validateReleaseAssetName(input.channel, input.fileName);
  if (
    !Buffer.isBuffer(input.body)
    || input.body.length === 0
    || input.body.length > MAX_RELEASE_CHUNK_SIZE_BYTES
  ) {
    throw new Error("release_chunk_size_invalid");
  }
  if (digest(input.body) !== input.metadata.chunkSha256) {
    throw new Error("release_chunk_checksum_mismatch");
  }

  const root = resolve(input.root);
  const target = resolve(root, input.fileName);
  if (dirname(target) !== root) throw new Error("release_asset_path_invalid");
  await mkdir(root, { recursive: true });

  if (await fileMatches(target, input.metadata.fileSize, input.metadata.fileSha256)) {
    return result(input.metadata, true);
  }

  const staging = resolve(root, ".incoming", input.metadata.fileSha256);
  await ensureSafeDirectory(staging);
  const chunkPath = resolve(staging, `${String(input.metadata.chunkIndex).padStart(6, "0")}.part`);
  await durableReplaceBytes(chunkPath, input.body);

  for (let index = 0; index < input.metadata.chunkCount; index += 1) {
    try {
      await stat(resolve(staging, `${String(index).padStart(6, "0")}.part`));
    } catch {
      return result(input.metadata, false);
    }
  }

  const hash = createHash("sha256");
  let bytes = 0;
  await durableReplaceFile(target, async (handle) => {
    for (let index = 0; index < input.metadata.chunkCount; index += 1) {
      const chunk = await readFile(resolve(staging, `${String(index).padStart(6, "0")}.part`));
      await writeAll(handle, chunk);
      hash.update(chunk);
      bytes += chunk.length;
    }

    if (bytes !== input.metadata.fileSize || hash.digest("hex") !== input.metadata.fileSha256) {
      throw new Error("release_file_checksum_mismatch");
    }
  });
  await rm(staging, { recursive: true, force: true });
  await syncDirectory(dirname(staging));
  return result(input.metadata, true);
}

async function fileMatches(path: string, size: number, sha256: string): Promise<boolean> {
  try {
    if ((await stat(path)).size !== size) return false;
    const hash = createHash("sha256");
    for await (const chunk of createReadStream(path)) hash.update(chunk);
    return hash.digest("hex") === sha256;
  } catch {
    return false;
  }
}

function result(metadata: ReleaseChunkMetadata, complete: boolean): StoredReleaseChunk {
  return {
    complete,
    receivedChunk: metadata.chunkIndex,
    chunkCount: metadata.chunkCount,
    fileSha256: metadata.fileSha256,
  };
}

function digest(value: Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function header(headers: Record<string, unknown>, name: string): string {
  const value = headers[name];
  if (typeof value !== "string" || !value) throw new Error(`release_header_missing_${name}`);
  return value;
}

function integerHeader(
  headers: Record<string, unknown>,
  name: string,
  minimum: number,
  maximum: number,
): number {
  const value = Number(header(headers, name));
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`release_header_invalid_${name}`);
  }
  return value;
}

function shaHeader(headers: Record<string, unknown>, name: string): string {
  const value = header(headers, name).toLowerCase();
  if (!SHA256.test(value)) throw new Error(`release_header_invalid_${name}`);
  return value;
}

interface ActivePointer {
  schema: typeof ACTIVE_SCHEMA;
  transactionId: string | null;
  releaseId: `sha256:${string}` | null;
}

interface CommitDocument {
  schema: typeof COMMIT_SCHEMA;
  manifest: ReleaseTransactionManifest;
  previousTransactionId: string | null;
  previousReleaseId: `sha256:${string}` | null;
}

interface ActiveRelease {
  manifest: ReleaseTransactionManifest;
  previousTransactionId: string | null;
  previousReleaseId: `sha256:${string}` | null;
}

interface FileEvidence {
  bytes: number;
  sha256: string;
  sha1: string;
}

function validateExpectedRuntime(input: {
  sourceId: string;
  revision: string;
  version: string;
}): Omit<ReleaseTransactionIdentity, "transactionId"> {
  if (!SOURCE_ID.test(input.sourceId)) {
    throw new Error("release_runtime_source_id_invalid");
  }
  if (!REVISION.test(input.revision)) {
    throw new Error("release_runtime_revision_invalid");
  }
  if (!VERSION.test(input.version)) {
    throw new Error("release_runtime_version_invalid");
  }
  return {
    sourceId: input.sourceId as `sha256:${string}`,
    revision: input.revision,
    version: input.version,
  };
}

function validateTransactionIdentity(candidate: {
  transactionId?: unknown;
  sourceId?: unknown;
  revision?: unknown;
  version?: unknown;
}): ReleaseTransactionIdentity {
  if (typeof candidate.transactionId !== "string") {
    throw new Error("release_transaction_id_invalid");
  }
  validateTransactionId(candidate.transactionId);
  if (typeof candidate.sourceId !== "string" || !SOURCE_ID.test(candidate.sourceId)) {
    throw new Error("release_transaction_source_id_invalid");
  }
  if (typeof candidate.revision !== "string" || !REVISION.test(candidate.revision)) {
    throw new Error("release_transaction_revision_invalid");
  }
  if (typeof candidate.version !== "string" || !VERSION.test(candidate.version)) {
    throw new Error("release_transaction_version_invalid");
  }
  return {
    transactionId: candidate.transactionId,
    sourceId: candidate.sourceId as `sha256:${string}`,
    revision: candidate.revision,
    version: candidate.version,
  };
}

function validateTransactionId(value: string): void {
  if (!TRANSACTION_ID.test(value)) {
    throw new Error("release_transaction_id_invalid");
  }
}

export function expectedPublicReleaseAssets(version: string): Array<{
  channel: ReleaseAssetChannel;
  fileName: string;
}> {
  return [
    { channel: "downloads", fileName: "mycellios-linux-x64.deb" },
    { channel: "downloads", fileName: "mycellios-linux-x64.rpm" },
    { channel: "downloads", fileName: "mycellios-macos-arm64.dmg" },
    { channel: "downloads", fileName: "mycellios-macos-x64.dmg" },
    { channel: "downloads", fileName: "mycellios-windows-x64.exe" },
    { channel: "updates", fileName: "RELEASES" },
    { channel: "updates", fileName: "latest.json" },
    { channel: "updates", fileName: `mycellios-${version}-full.nupkg` },
    { channel: "updates", fileName: "mycellios-setup.exe" },
  ];
}

function validateManifestAssets(
  version: string,
  candidates: unknown[],
): ReleaseAssetEvidence[] {
  const expected = expectedPublicReleaseAssets(version);
  if (candidates.length !== expected.length) {
    throw new Error("release_manifest_asset_count_invalid");
  }
  const result = candidates.map((candidate, index) => {
    assertPlainObject(candidate, `release_manifest_asset_${index}`);
    assertExactKeys(
      candidate,
      ["channel", "fileName", "fileSha256", "fileSize"],
      `release_manifest_asset_${index}`,
    );
    const canonical = expected[index];
    if (
      !canonical
      || candidate.channel !== canonical.channel
      || candidate.fileName !== canonical.fileName
    ) {
      throw new Error("release_manifest_asset_order_or_name_invalid");
    }
    if (
      typeof candidate.fileSize !== "number"
      || !Number.isSafeInteger(candidate.fileSize)
      || candidate.fileSize < 1
      || candidate.fileSize > MAX_RELEASE_FILE_SIZE
    ) {
      throw new Error("release_manifest_asset_size_invalid");
    }
    if (
      typeof candidate.fileSha256 !== "string"
      || !SHA256.test(candidate.fileSha256)
    ) {
      throw new Error("release_manifest_asset_sha256_invalid");
    }
    return {
      channel: canonical.channel,
      fileName: canonical.fileName,
      fileSize: candidate.fileSize,
      fileSha256: candidate.fileSha256,
    };
  });
  return result;
}

function assertExpectedAsset(
  version: string,
  channel: ReleaseAssetChannel,
  fileName: string,
): void {
  if (!expectedPublicReleaseAssets(version).some(
    (asset) => asset.channel === channel && asset.fileName === fileName,
  )) {
    throw new Error("release_transaction_asset_not_expected");
  }
}

function rollbackDocument(
  transactionId: string,
  restored: ActiveRelease | null,
): {
  schema: typeof ROLLBACK_SCHEMA;
  transactionId: string;
  restoredTransactionId: string | null;
  restoredReleaseId: `sha256:${string}` | null;
} {
  return {
    schema: ROLLBACK_SCHEMA,
    transactionId,
    restoredTransactionId: restored?.manifest.transactionId ?? null,
    restoredReleaseId: restored?.manifest.releaseId ?? null,
  };
}

const DURABLE_TEMPORARY_SUFFIX = ".durable-tmp";

async function durableReplaceBytes(path: string, bytes: Buffer): Promise<void> {
  await durableReplaceFile(path, async (handle) => {
    await handle.writeFile(bytes);
  });
}

async function durableReplaceFile(
  path: string,
  writer: (handle: FileHandle) => Promise<void>,
): Promise<void> {
  const directory = dirname(path);
  await ensureSafeDirectory(directory);
  await removeDurableTemporaries(path);
  const temporary = join(
    directory,
    `.${basename(path)}.${randomUUID()}${DURABLE_TEMPORARY_SUFFIX}`,
  );
  let renamed = false;
  try {
    const handle = await open(temporary, "wx");
    try {
      await writer(handle);
      await handle.sync();
    } finally {
      await handle.close();
    }
    await rename(temporary, path);
    renamed = true;
    if (process.platform === "win32") {
      // Node cannot fsync directory handles on Windows (EPERM). Flushing the
      // renamed file is the strongest portable FlushFileBuffers barrier before
      // attempting the directory barrier below.
      await syncRegularFile(path);
    }
    await syncDirectory(directory);
  } finally {
    if (!renamed) await rm(temporary, { force: true });
  }
}

async function removeDurableTemporaries(path: string): Promise<void> {
  const directory = dirname(path);
  const prefix = `.${basename(path)}.`;
  let removed = false;
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (
      !entry.name.startsWith(prefix)
      || !entry.name.endsWith(DURABLE_TEMPORARY_SUFFIX)
    ) {
      continue;
    }
    if (entry.isDirectory()) {
      throw new Error("release_transaction_temporary_path_unsafe");
    }
    await rm(join(directory, entry.name), { force: true });
    removed = true;
  }
  if (removed) await syncDirectory(directory);
}

async function writeAll(handle: FileHandle, bytes: Buffer): Promise<void> {
  let offset = 0;
  while (offset < bytes.length) {
    const { bytesWritten } = await handle.write(
      bytes,
      offset,
      bytes.length - offset,
      null,
    );
    if (bytesWritten < 1) {
      throw new Error("release_file_write_incomplete");
    }
    offset += bytesWritten;
  }
}

async function syncRegularFile(path: string): Promise<void> {
  const handle = await open(path, "r+");
  try {
    await handle.sync();
  } finally {
    await handle.close();
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
      && typeof error.code === "string"
      && ["EINVAL", "ENOTSUP", "EPERM"].includes(error.code)
    ) {
      return;
    }
    throw error;
  } finally {
    await handle?.close();
  }
}

async function writeExclusiveOrMatch(
  path: string,
  value: unknown,
  conflictCode: string,
): Promise<void> {
  await ensureSafeDirectory(dirname(path));
  const bytes = jsonBytes(value);
  try {
    const existing = await readFile(path);
    if (!Buffer.from(existing).equals(bytes)) {
      throw new Error(conflictCode);
    }
    return;
  } catch (error) {
    if (!isMissing(error)) throw error;
  }
  await durableReplaceBytes(path, bytes);
  if (!Buffer.from(await readFile(path)).equals(bytes)) {
    throw new Error(conflictCode);
  }
}

async function ensureSafeDirectory(path: string): Promise<void> {
  const target = resolve(path);
  const missing: string[] = [];
  let current = target;
  while (true) {
    try {
      const details = await lstat(current);
      if (!details.isDirectory() || details.isSymbolicLink()) {
        throw new Error("release_transaction_directory_unsafe");
      }
      break;
    } catch (error) {
      if (!isMissing(error)) throw error;
      missing.push(current);
      const parent = dirname(current);
      if (parent === current) throw error;
      current = parent;
    }
  }
  for (const directory of missing.reverse()) {
    try {
      await mkdir(directory);
    } catch (error) {
      if (!isAlreadyExists(error)) throw error;
    }
    const details = await lstat(directory);
    if (!details.isDirectory() || details.isSymbolicLink()) {
      throw new Error("release_transaction_directory_unsafe");
    }
    await syncDirectory(dirname(directory));
  }
  await syncDirectory(target);
}

async function listRegularFiles(root: string): Promise<string[]> {
  const base = resolve(root);
  const output: string[] = [];
  const visit = async (directory: string): Promise<void> => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const absolute = join(directory, entry.name);
      if (entry.isSymbolicLink()) {
        throw new Error("release_transaction_symbolic_link_rejected");
      }
      if (entry.isDirectory()) await visit(absolute);
      else if (entry.isFile()) {
        const portable = relative(base, absolute).replaceAll("\\", "/");
        if (!portable || portable.startsWith("../")) {
          throw new Error("release_transaction_path_invalid");
        }
        output.push(portable);
      } else {
        throw new Error("release_transaction_special_file_rejected");
      }
    }
  };
  try {
    await visit(base);
  } catch (error) {
    if (isMissing(error)) return [];
    throw error;
  }
  return output.sort();
}

async function inspectFile(path: string): Promise<FileEvidence> {
  const details = await lstat(path);
  if (!details.isFile() || details.isSymbolicLink()) {
    throw new Error("release_transaction_asset_not_regular");
  }
  const sha256 = createHash("sha256");
  const sha1 = createHash("sha1");
  let bytes = 0;
  for await (const chunk of createReadStream(path)) {
    sha256.update(chunk);
    sha1.update(chunk);
    bytes += chunk.length;
  }
  return {
    bytes,
    sha256: sha256.digest("hex"),
    sha1: sha1.digest("hex"),
  };
}

async function verifyFeedCoherence(
  manifest: ReleaseTransactionManifest,
  assetsRoot: string,
  observed: Map<string, FileEvidence>,
): Promise<void> {
  const versionedName = `mycellios-${manifest.version}-full.nupkg`;
  const nupkg = observed.get(`updates/${versionedName}`);
  const setup = observed.get("updates/mycellios-setup.exe");
  const publicWindows = observed.get("downloads/mycellios-windows-x64.exe");
  if (!nupkg || !setup || !publicWindows) {
    throw new Error("release_transaction_windows_assets_missing");
  }
  if (
    setup.bytes !== publicWindows.bytes
    || setup.sha256 !== publicWindows.sha256
  ) {
    throw new Error("release_transaction_windows_installer_alias_mismatch");
  }

  const releases = parseSquirrelReleaseRecord(
    await readFile(resolveInside(assetsRoot, "updates/RELEASES")),
  );
  if (
    releases.sha1.toLowerCase() !== nupkg.sha1
    || releases.name !== versionedName
    || releases.size !== BigInt(nupkg.bytes)
  ) {
    throw new Error("release_transaction_releases_feed_mismatch");
  }

  let latest: unknown;
  try {
    latest = JSON.parse(
      await readFile(
        resolveInside(assetsRoot, "updates/latest.json"),
        "utf8",
      ),
    );
  } catch {
    throw new Error("release_transaction_latest_json_invalid");
  }
  assertPlainObject(latest, "release_latest_json");
  assertExactKeys(
    latest,
    ["files", "publishedAt", "schema", "version"],
    "release_latest_json",
  );
  if (
    latest.schema !== "mycellios-windows-update-feed/1"
    || latest.version !== manifest.version
  ) {
    throw new Error("release_transaction_latest_version_mismatch");
  }
  if (
    typeof latest.publishedAt !== "string"
    || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(latest.publishedAt)
    || new Date(latest.publishedAt).toISOString() !== latest.publishedAt
  ) {
    throw new Error("release_transaction_latest_published_at_invalid");
  }
  if (!Array.isArray(latest.files)) {
    throw new Error("release_transaction_latest_files_invalid");
  }
  const latestFiles = [
    ["RELEASES", observed.get("updates/RELEASES")],
    ["mycellios-setup.exe", setup],
    [versionedName, nupkg],
  ].map(([name, evidence]) => {
    if (typeof name !== "string" || !evidence || typeof evidence === "string") {
      throw new Error("release_transaction_latest_files_invalid");
    }
    return {
      name,
      bytes: evidence.bytes,
      sha256: evidence.sha256,
    };
  });
  if (canonicalJson(latest.files) !== canonicalJson(latestFiles)) {
    throw new Error("release_transaction_latest_files_mismatch");
  }
}

function parseSquirrelReleaseRecord(bytes: Buffer): {
  sha1: string;
  name: string;
  size: bigint;
} {
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw new Error("release_transaction_releases_feed_invalid_utf8");
  }
  if (
    text.startsWith("\uFEFF")
    || /[\u0000-\u0009\u000B\u000C\u000E-\u001F\u007F-\u009F]/u.test(text)
  ) {
    throw new Error("release_transaction_releases_feed_control_bytes");
  }
  if (text.endsWith("\r\n")) text = text.slice(0, -2);
  else if (text.endsWith("\n")) text = text.slice(0, -1);
  if (text.includes("\r") || text.includes("\n") || text.trim() !== text) {
    throw new Error("release_transaction_releases_feed_not_single_line");
  }
  const match = /^([0-9A-Fa-f]{40}) +([^ \t]+) +([0-9]+)$/.exec(text);
  if (!match) throw new Error("release_transaction_releases_feed_format_invalid");
  const [, sha1, name, sizeText] = match;
  if (
    name === undefined
    || sha1 === undefined
    || sizeText === undefined
    || name.includes("/")
    || name.includes("\\")
    || name === "."
    || name === ".."
  ) {
    throw new Error("release_transaction_releases_feed_name_invalid");
  }
  return {
    sha1,
    name,
    size: BigInt(sizeText),
  };
}

function parseCommitDocument(candidate: unknown): CommitDocument {
  assertPlainObject(candidate, "release_commit");
  assertExactKeys(
    candidate,
    ["manifest", "previousReleaseId", "previousTransactionId", "schema"],
    "release_commit",
  );
  if (candidate.schema !== COMMIT_SCHEMA) {
    throw new Error("release_commit_schema_invalid");
  }
  const manifest = parseReleaseTransactionManifest(candidate.manifest);
  validateNullableTransactionId(candidate.previousTransactionId);
  validateNullableReleaseId(candidate.previousReleaseId);
  if (
    (candidate.previousTransactionId === null)
    !== (candidate.previousReleaseId === null)
  ) {
    throw new Error("release_commit_previous_identity_invalid");
  }
  return {
    schema: COMMIT_SCHEMA,
    manifest,
    previousTransactionId: candidate.previousTransactionId,
    previousReleaseId: candidate.previousReleaseId,
  };
}

function parseActivePointer(candidate: unknown): ActivePointer {
  assertPlainObject(candidate, "release_active");
  assertExactKeys(
    candidate,
    ["releaseId", "schema", "transactionId"],
    "release_active",
  );
  if (candidate.schema !== ACTIVE_SCHEMA) {
    throw new Error("release_active_schema_invalid");
  }
  validateNullableTransactionId(candidate.transactionId);
  validateNullableReleaseId(candidate.releaseId);
  if ((candidate.transactionId === null) !== (candidate.releaseId === null)) {
    throw new Error("release_active_identity_invalid");
  }
  return {
    schema: ACTIVE_SCHEMA,
    transactionId: candidate.transactionId,
    releaseId: candidate.releaseId,
  };
}

function validateNullableTransactionId(value: unknown): asserts value is string | null {
  if (value === null) return;
  if (typeof value !== "string") throw new Error("release_transaction_id_invalid");
  validateTransactionId(value);
}

function validateNullableReleaseId(
  value: unknown,
): asserts value is `sha256:${string}` | null {
  if (value !== null && (typeof value !== "string" || !SOURCE_ID.test(value))) {
    throw new Error("release_id_invalid");
  }
}

async function readOptionalJson(path: string): Promise<unknown | null> {
  try {
    return JSON.parse(await readFile(path, "utf8"));
  } catch (error) {
    if (isMissing(error)) return null;
    throw error;
  }
}

async function readRequiredJson(path: string, missingCode: string): Promise<unknown> {
  const value = await readOptionalJson(path);
  if (value === null) throw new Error(missingCode);
  return value;
}

async function pathEntryExists(path: string): Promise<boolean> {
  try {
    await lstat(path);
    return true;
  } catch (error) {
    if (isMissing(error)) return false;
    throw error;
  }
}

async function newestSafeTreeMtime(root: string): Promise<number | null> {
  let details;
  try {
    details = await lstat(root);
  } catch (error) {
    if (isMissing(error)) return null;
    throw error;
  }
  if (!details.isDirectory() || details.isSymbolicLink()) return null;
  let newest = details.mtimeMs;
  let entries;
  try {
    entries = await readdir(root, { withFileTypes: true });
  } catch (error) {
    if (isMissing(error)) return null;
    throw error;
  }
  for (const entry of entries) {
    const path = join(root, entry.name);
    let child;
    try {
      child = await lstat(path);
    } catch (error) {
      if (isMissing(error)) return null;
      throw error;
    }
    if (child.isSymbolicLink()) return null;
    if (child.isDirectory()) {
      const nested = await newestSafeTreeMtime(path);
      if (nested === null) return null;
      newest = Math.max(newest, nested);
      continue;
    }
    if (!child.isFile()) return null;
    newest = Math.max(newest, child.mtimeMs);
  }
  return newest;
}

function sameTransactionId(
  left: string | null | undefined,
  right: string | null | undefined,
): boolean {
  if (left === null || left === undefined || right === null || right === undefined) {
    return false;
  }
  return process.platform === "win32"
    ? left.toLowerCase() === right.toLowerCase()
    : left === right;
}

function resolveInside(root: string, portable: string): string {
  const base = resolve(root);
  if (
    !portable
    || portable.includes("\\")
    || portable.startsWith("/")
    || /^[A-Za-z]:/.test(portable)
    || portable.split("/").some(
      (segment) => !segment || segment === "." || segment === "..",
    )
  ) {
    throw new Error("release_asset_path_invalid");
  }
  const absolute = resolve(base, ...portable.split("/"));
  if (!absolute.startsWith(`${base}${sep}`)) {
    throw new Error("release_asset_path_invalid");
  }
  return absolute;
}

function jsonBytes(value: unknown): Buffer {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
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

function assertPlainObject(
  value: unknown,
  label: string,
): asserts value is Record<string, unknown> {
  if (!isPlainObject(value)) throw new Error(`${label}_invalid`);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return (
    value !== null
    && typeof value === "object"
    && !Array.isArray(value)
    && Object.getPrototypeOf(value) === Object.prototype
  );
}

function assertExactKeys(
  value: Record<string, unknown>,
  expected: string[],
  label: string,
): void {
  if (
    canonicalJson(Object.keys(value).sort())
    !== canonicalJson([...expected].sort())
  ) {
    throw new Error(`${label}_fields_invalid`);
  }
}

function isMissing(error: unknown): boolean {
  return (
    isFileSystemError(error)
    && error.code === "ENOENT"
  );
}

function isAlreadyExists(error: unknown): boolean {
  return (
    isFileSystemError(error)
    && error.code === "EEXIST"
  );
}

function isFileSystemError(error: unknown): error is NodeJS.ErrnoException {
  return (
    error !== null
    && typeof error === "object"
    && "code" in error
    && typeof error.code === "string"
  );
}
