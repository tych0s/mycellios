import {
  createHash,
  createPublicKey,
  randomUUID,
  sign as signBytes,
  verify as verifyBytes,
  type KeyObject,
  type KeyLike,
} from "node:crypto";
import { createReadStream } from "node:fs";
import {
  mkdir,
  open,
  rename,
  rm,
  stat,
} from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";

export const ARTIFACT_SWARM_SCHEMA = "mycellios-artifact-swarm/2" as const;

const MAX_PACKAGES = 256;
const MAX_BLOBS_PER_PACKAGE = 16;
const MAX_CHUNKS_PER_BLOB = 65_536;
const MAX_PEER_CHUNKS = 1_000_000;
const QUARANTINE_STRIKES = 2;

export interface ArtifactChunkDescriptor {
  index: number;
  offset: number;
  sizeBytes: number;
  sha256: string;
}

export interface ArtifactBlobDescriptor {
  sha256: string;
  sizeBytes: number;
  chunks: ArtifactChunkDescriptor[];
}

export interface ArtifactPackageDescriptor {
  packageId: string;
  artifactIds: string[];
  layerStart: number;
  layerEnd: number;
  manifest: ArtifactBlobDescriptor;
  blobs: ArtifactBlobDescriptor[];
}

export interface UnsignedArtifactSwarmManifest {
  schema: typeof ARTIFACT_SWARM_SCHEMA;
  modelIdentity: string;
  distributionManifestId: string;
  sourceRevision: string;
  tensorAbi: string;
  packages: ArtifactPackageDescriptor[];
}

export interface SignedArtifactSwarmManifest extends UnsignedArtifactSwarmManifest {
  manifestId: string;
  publisherPublicKey: string;
  signature: string;
}

export interface ArtifactPeerAnnouncement {
  peerId: string;
  chunkIds: string[];
  expiresAt: number;
  rttMs: number;
  goodputMbps: number;
  activeTransfers: number;
  reliability: number;
}

export interface ArtifactChunkRequest {
  packageId: string;
  blobSha256: string;
  chunk: ArtifactChunkDescriptor;
  chunkId: string;
  sources: Array<{
    peerId: string;
    score: number;
    expiresAt: number;
  }>;
}

/**
 * Structural copy of the pinned-artifact contract used by the desktop
 * provisioner. Keeping it here prevents the model-fabric layer from depending
 * on a UI shell while still allowing the production downloader to use it
 * directly.
 */
export interface ArtifactSwarmPinnedArtifact {
  label: string;
  url: string;
  sha256: string;
  sizeBytes: number;
}

export interface ArtifactSwarmTransfer {
  bytesDownloaded: number;
  bytesTotal: number;
  bytesPerSecond: number | null;
  etaSeconds: number | null;
  resumed: boolean;
}

export interface ArtifactPeerChunkFetch {
  peerId: string;
  packageId: string;
  blobSha256: string;
  chunkId: string;
  offset: number;
  sizeBytes: number;
}

export interface ArtifactPeerChunkResponse {
  /**
   * A streaming body keeps malicious or unexpectedly large responses from
   * being buffered in RAM before the declared chunk limit is enforced.
   */
  body: AsyncIterable<Uint8Array>;
  contentLength?: number | undefined;
}

/**
 * Real transports (the native direct channel, HTTPS, or a local test peer)
 * implement this narrow byte interface. Peer inventory is deliberately not
 * coupled to transport: an announcement never proves possession.
 */
export interface ArtifactPeerChunkClient {
  fetchChunk(request: ArtifactPeerChunkFetch): Promise<ArtifactPeerChunkResponse>;
}

export type ArtifactOriginDownloader = (
  artifact: ArtifactSwarmPinnedArtifact,
  cacheRoot: string,
  onProgress: (progress: ArtifactSwarmTransfer) => void,
) => Promise<string>;

export interface ArtifactSwarmDownloaderOptions {
  registry: ArtifactSwarmRegistry;
  manifestId: string;
  requesterPeerId: string;
  peerClient: ArtifactPeerChunkClient;
  originDownloader: ArtifactOriginDownloader;
  /**
   * Optional explicit package mapping. Without it, the signed manifest must
   * contain exactly one package with a blob matching the pinned artifact.
   */
  packageIdForArtifact?:
    | ((artifact: ArtifactSwarmPinnedArtifact) => string | null | undefined)
    | undefined;
  maximumSourcesPerChunk?: number | undefined;
  maximumParallelChunks?: number | undefined;
  now?: (() => number) | undefined;
}

interface StoredPeer {
  announcement: ArtifactPeerAnnouncement;
  chunkIds: Set<string>;
  corruptionStrikes: number;
  quarantinedAt: number | null;
}

export function signArtifactSwarmManifest(
  manifest: UnsignedArtifactSwarmManifest,
  privateKey: KeyLike,
  publicKey: KeyLike,
): SignedArtifactSwarmManifest {
  const normalized = validateUnsignedManifest(manifest);
  const payload = canonicalBytes(normalized);
  const publicKeyObject =
    isPublicKeyObject(publicKey) ? publicKey : createPublicKey(publicKey);
  const exportedPublicKey = publicKeyObject.export({
    format: "der",
    type: "spki",
  }).toString("base64url");
  return {
    ...normalized,
    manifestId: sha256Identity(payload),
    publisherPublicKey: exportedPublicKey,
    signature: signBytes(null, payload, privateKey).toString("base64url"),
  };
}

export function verifyArtifactSwarmManifest(
  value: SignedArtifactSwarmManifest,
): SignedArtifactSwarmManifest {
  const record = strictRecord(value, [
    "schema",
    "modelIdentity",
    "distributionManifestId",
    "sourceRevision",
    "tensorAbi",
    "packages",
    "manifestId",
    "publisherPublicKey",
    "signature",
  ], "artifact_swarm_manifest_is_invalid");
  const normalized = validateUnsignedManifest({
    schema: record.schema as typeof ARTIFACT_SWARM_SCHEMA,
    modelIdentity: record.modelIdentity as string,
    distributionManifestId: record.distributionManifestId as string,
    sourceRevision: record.sourceRevision as string,
    tensorAbi: record.tensorAbi as string,
    packages: record.packages as ArtifactPackageDescriptor[],
  });
  const manifestId = sha256IdentityText(record.manifestId, "artifact_swarm_manifest_id_is_invalid");
  const publisherPublicKey = canonicalBase64url(
    record.publisherPublicKey,
    "artifact_swarm_publisher_key_is_invalid",
  );
  const signature = canonicalBase64url(
    record.signature,
    "artifact_swarm_signature_is_invalid",
  );
  const payload = canonicalBytes(normalized);
  if (manifestId !== sha256Identity(payload)) {
    throw new Error("artifact_swarm_manifest_identity_mismatch");
  }
  let publicKey;
  try {
    publicKey = createPublicKey({
      key: Buffer.from(publisherPublicKey, "base64url"),
      format: "der",
      type: "spki",
    });
  } catch {
    throw new Error("artifact_swarm_publisher_key_is_invalid");
  }
  if (publicKey.asymmetricKeyType !== "ed25519") {
    throw new Error("artifact_swarm_publisher_key_is_invalid");
  }
  if (!verifyBytes(null, payload, publicKey, Buffer.from(signature, "base64url"))) {
    throw new Error("artifact_swarm_signature_verification_failed");
  }
  return {
    ...normalized,
    manifestId,
    publisherPublicKey,
    signature,
  };
}

export class ArtifactSwarmRegistry {
  private readonly manifests = new Map<string, SignedArtifactSwarmManifest>();
  private readonly peers = new Map<string, StoredPeer>();
  private readonly trustedPublisherKeys: ReadonlySet<string>;

  constructor(options: { trustedPublisherKeys: readonly KeyLike[] }) {
    if (options.trustedPublisherKeys.length === 0) throw new Error("artifact_swarm_trusted_publisher_is_missing");
    this.trustedPublisherKeys = new Set(options.trustedPublisherKeys.map((key) => {
      const publicKey = isPublicKeyObject(key) ? key : createPublicKey(key);
      if (publicKey.asymmetricKeyType !== "ed25519") throw new Error("artifact_swarm_trusted_publisher_is_invalid");
      return publicKey.export({ format: "der", type: "spki" }).toString("base64url");
    }));
  }

  publish(value: SignedArtifactSwarmManifest): SignedArtifactSwarmManifest {
    const manifest = verifyArtifactSwarmManifest(value);
    if (!this.trustedPublisherKeys.has(manifest.publisherPublicKey)) {
      throw new Error("artifact_swarm_publisher_is_not_trusted");
    }
    const existing = this.manifests.get(manifest.manifestId);
    if (existing && canonicalJson(existing) !== canonicalJson(manifest)) {
      throw new Error("artifact_swarm_manifest_identity_collision");
    }
    this.manifests.set(manifest.manifestId, structuredClone(manifest));
    return structuredClone(manifest);
  }

  getManifest(manifestId: string): SignedArtifactSwarmManifest | null {
    const value = this.manifests.get(
      sha256IdentityText(manifestId, "artifact_swarm_manifest_id_is_invalid"),
    );
    return value ? structuredClone(value) : null;
  }

  announcePeer(value: ArtifactPeerAnnouncement, now = Date.now()): void {
    const announcement = validatePeerAnnouncement(value, now);
    const previous = this.peers.get(announcement.peerId);
    this.peers.set(announcement.peerId, {
      announcement,
      chunkIds: new Set(announcement.chunkIds),
      corruptionStrikes: previous?.corruptionStrikes ?? 0,
      quarantinedAt: previous?.quarantinedAt ?? null,
    });
  }

  planDownload(input: {
    manifestId: string;
    requesterPeerId: string;
    packageIds: string[];
    requesterChunkIds?: ReadonlySet<string>;
    maximumSourcesPerChunk?: number;
    now?: number;
  }): ArtifactChunkRequest[] {
    const now = input.now ?? Date.now();
    this.expirePeers(now);
    const manifest = this.manifests.get(
      sha256IdentityText(
        input.manifestId,
        "artifact_swarm_manifest_id_is_invalid",
      ),
    );
    if (!manifest) throw new Error("artifact_swarm_manifest_not_found");
    const requested = uniqueDigests(input.packageIds, "artifact_swarm_package_id_is_invalid");
    const order = new Map(requested.map((packageId, index) => [packageId, index]));
    const selected = manifest.packages
      .filter((entry) => order.has(entry.packageId))
      .sort((left, right) => order.get(left.packageId)! - order.get(right.packageId)!);
    if (selected.length !== requested.length) {
      throw new Error("artifact_swarm_package_not_found");
    }
    const maximumSources = boundedInteger(
      input.maximumSourcesPerChunk ?? 3,
      1,
      16,
      "artifact_swarm_source_limit_is_invalid",
    );
    const availablePeers = [...this.peers.values()].filter(
      (peer) =>
        peer.announcement.peerId !== input.requesterPeerId
        && peer.quarantinedAt === null
        && peer.announcement.expiresAt > now,
    );
    const requesterChunks = input.requesterChunkIds ?? new Set<string>();
    const requests = selected.flatMap((entry) => packageChunks(entry)
      .filter((candidate) => !requesterChunks.has(candidate.chunkId))
      .map((candidate) => {
        const sources = availablePeers
          .filter((peer) => peer.chunkIds.has(candidate.chunkId))
          .map((peer) => ({
            peerId: peer.announcement.peerId,
            score: peerScore(peer, candidate.chunk.sizeBytes),
            expiresAt: peer.announcement.expiresAt,
          }))
          .sort(
            (left, right) =>
              left.score - right.score || left.peerId.localeCompare(right.peerId),
          )
          .slice(0, maximumSources);
        return {
          ...candidate,
          sources,
          rarity: sources.length,
          packagePriority: order.get(entry.packageId)!,
        };
      }))
      .sort(
        (left, right) =>
          left.packagePriority - right.packagePriority
          || left.rarity - right.rarity
          || left.chunk.offset - right.chunk.offset
          || left.chunkId.localeCompare(right.chunkId),
      )
      .map(({ rarity: _rarity, packagePriority: _priority, ...request }) => request);
    return requests;
  }

  recordVerifiedChunk(peerId: string): void {
    const peer = this.peers.get(peerId);
    if (!peer || peer.quarantinedAt !== null) return;
    peer.announcement.reliability = Math.min(
      1,
      peer.announcement.reliability + 0.002,
    );
  }

  recordCorruptChunk(peerId: string, now = Date.now()): boolean {
    const peer = this.peers.get(peerId);
    if (!peer) return false;
    peer.corruptionStrikes += 1;
    peer.announcement.reliability = Math.max(
      0,
      peer.announcement.reliability * 0.5,
    );
    if (peer.corruptionStrikes >= QUARANTINE_STRIKES) {
      peer.quarantinedAt = now;
      return true;
    }
    return false;
  }

  peerState(peerId: string): {
    corruptionStrikes: number;
    quarantined: boolean;
    expiresAt: number;
  } | null {
    const peer = this.peers.get(peerId);
    return peer
      ? {
          corruptionStrikes: peer.corruptionStrikes,
          quarantined: peer.quarantinedAt !== null,
          expiresAt: peer.announcement.expiresAt,
        }
      : null;
  }

  expirePeers(now = Date.now()): number {
    let removed = 0;
    for (const [peerId, peer] of this.peers) {
      if (peer.announcement.expiresAt <= now) {
        this.peers.delete(peerId);
        removed += 1;
      }
    }
    return removed;
  }
}

/**
 * Build a production-compatible downloader that opportunistically consumes
 * verified chunks from Mycellios peers and retains the existing pinned origin
 * downloader as the authoritative fallback.
 */
export function createArtifactSwarmDownloader(
  options: ArtifactSwarmDownloaderOptions,
): ArtifactOriginDownloader {
  const maximumSources = boundedInteger(
    options.maximumSourcesPerChunk ?? 3,
    1,
    16,
    "artifact_swarm_source_limit_is_invalid",
  );
  const maximumParallel = boundedInteger(
    options.maximumParallelChunks ?? 4,
    1,
    32,
    "artifact_swarm_parallelism_is_invalid",
  );
  const requesterPeerId = text(
    options.requesterPeerId,
    256,
    "artifact_peer_id_is_invalid",
  );
  const now = options.now ?? Date.now;

  return async (artifact, cacheRoot, onProgress) => {
    const normalizedArtifact = validatePinnedArtifact(artifact);
    const cache = artifactCachePaths(normalizedArtifact, cacheRoot);
    await mkdir(cache.artifactDirectory, { recursive: true });

    if (await verifiedFile(cache.target, normalizedArtifact)) {
      onProgress(completedSwarmTransfer(normalizedArtifact.sizeBytes, false));
      return cache.target;
    }
    await quarantineInvalidArtifact(cache.target, cache.artifactDirectory);

    const selection = selectArtifactBlob(
      options.registry.getManifest(options.manifestId),
      normalizedArtifact,
      options.packageIdForArtifact?.(normalizedArtifact),
    );
    if (selection === null) {
      return verifiedOriginFallback(
        options.originDownloader,
        normalizedArtifact,
        cacheRoot,
        cache,
        onProgress,
        0,
        false,
      );
    }

    await mkdir(cache.chunkDirectory, { recursive: true });
    const present = await verifyPresentChunks(cache.chunkDirectory, selection.blob);
    const resumed = present.size > 0;
    let verifiedBytes = verifiedChunkBytes(selection.blob, present);
    const startedAt = now();
    onProgress(swarmTransfer(
      verifiedBytes,
      normalizedArtifact.sizeBytes,
      startedAt,
      now(),
      resumed,
    ));

    const plan = options.registry.planDownload({
      manifestId: options.manifestId,
      requesterPeerId,
      packageIds: [selection.packageId],
      requesterChunkIds: present,
      maximumSourcesPerChunk: maximumSources,
      now: now(),
    }).filter((request) => request.blobSha256 === selection.blob.sha256);

    let nextIndex = 0;
    let peerFailure: unknown = null;
    const workerCount = Math.min(maximumParallel, Math.max(1, plan.length));
    await Promise.all(Array.from({ length: workerCount }, async () => {
      while (true) {
        const requestIndex = nextIndex;
        nextIndex += 1;
        const request = plan[requestIndex];
        if (request === undefined) return;

        let stored = false;
        for (const source of request.sources) {
          const peerState = options.registry.peerState(source.peerId);
          if (
            peerState === null
            || peerState.quarantined
            || source.expiresAt <= now()
          ) {
            continue;
          }
          try {
            await fetchVerifiedPeerChunk(
              options.peerClient,
              source.peerId,
              request,
              cache.chunkDirectory,
            );
            options.registry.recordVerifiedChunk(source.peerId);
            present.add(request.chunkId);
            verifiedBytes += request.chunk.sizeBytes;
            onProgress(swarmTransfer(
              verifiedBytes,
              normalizedArtifact.sizeBytes,
              startedAt,
              now(),
              resumed,
            ));
            stored = true;
            break;
          } catch (error) {
            peerFailure ??= error;
            if (error instanceof ArtifactPeerCorruptionError) {
              options.registry.recordCorruptChunk(source.peerId, now());
            }
          }
        }
        if (!stored) peerFailure ??= new Error("artifact_swarm_chunk_has_no_verified_source");
      }
    }));

    if (present.size !== selection.blob.chunks.length) {
      return verifiedOriginFallback(
        options.originDownloader,
        normalizedArtifact,
        cacheRoot,
        cache,
        onProgress,
        verifiedBytes,
        resumed,
        peerFailure,
      );
    }

    try {
      await assembleVerifiedArtifact(
        cache.chunkDirectory,
        cache.target,
        normalizedArtifact,
        selection.blob,
      );
    } catch (error) {
      return verifiedOriginFallback(
        options.originDownloader,
        normalizedArtifact,
        cacheRoot,
        cache,
        onProgress,
        verifiedBytes,
        resumed,
        error,
      );
    }
    onProgress(completedSwarmTransfer(normalizedArtifact.sizeBytes, resumed));
    return cache.target;
  };
}

async function quarantineInvalidArtifact(target: string, artifactDirectory: string): Promise<void> {
  try {
    await stat(target);
  } catch {
    return;
  }
  const quarantine = resolve(
    artifactDirectory,
    `.quarantine-${basename(target)}-${randomUUID()}`,
  );
  assertDirectChild(artifactDirectory, quarantine, "artifact_swarm_cache_path_escaped");
  await rename(target, quarantine);
}

export function artifactChunkId(blobSha256: string, chunkIndex: number): string {
  const digest = sha256Text(blobSha256, "artifact_swarm_blob_digest_is_invalid");
  const index = boundedInteger(
    chunkIndex,
    0,
    MAX_CHUNKS_PER_BLOB - 1,
    "artifact_swarm_chunk_index_is_invalid",
  );
  return `${digest}:${index}`;
}

function validateUnsignedManifest(
  value: UnsignedArtifactSwarmManifest,
): UnsignedArtifactSwarmManifest {
  const record = strictRecord(value, [
    "schema",
    "modelIdentity",
    "distributionManifestId",
    "sourceRevision",
    "tensorAbi",
    "packages",
  ], "artifact_swarm_manifest_is_invalid");
  if (record.schema !== ARTIFACT_SWARM_SCHEMA) {
    throw new Error("artifact_swarm_schema_is_unsupported");
  }
  const modelIdentity = sha256IdentityText(
    record.modelIdentity,
    "artifact_swarm_model_identity_is_invalid",
  );
  const distributionManifestId = sha256IdentityText(
    record.distributionManifestId,
    "artifact_swarm_distribution_manifest_id_is_invalid",
  );
  const sourceRevision = text(
    record.sourceRevision,
    512,
    "artifact_swarm_revision_is_invalid",
  );
  const tensorAbi = text(
    record.tensorAbi,
    256,
    "artifact_swarm_tensor_abi_is_invalid",
  );
  if (!Array.isArray(record.packages) || record.packages.length < 1 || record.packages.length > MAX_PACKAGES) {
    throw new Error("artifact_swarm_packages_are_invalid");
  }
  const packages = record.packages.map((entry) => validatePackage(entry));
  const ids = new Set(packages.map((entry) => entry.packageId));
  if (ids.size !== packages.length) throw new Error("artifact_swarm_package_is_duplicated");
  const ranges = [...packages].sort((left, right) => left.layerStart - right.layerStart);
  for (let index = 1; index < ranges.length; index += 1) {
    if (ranges[index - 1]!.layerEnd > ranges[index]!.layerStart) {
      throw new Error("artifact_swarm_package_ranges_overlap");
    }
  }
  return {
    schema: ARTIFACT_SWARM_SCHEMA,
    modelIdentity,
    distributionManifestId,
    sourceRevision,
    tensorAbi,
    packages,
  };
}

function validatePackage(value: unknown): ArtifactPackageDescriptor {
  const record = strictRecord(value, [
    "packageId",
    "artifactIds",
    "layerStart",
    "layerEnd",
    "manifest",
    "blobs",
  ], "artifact_swarm_package_is_invalid");
  const packageId = sha256Text(
    record.packageId,
    "artifact_swarm_package_id_is_invalid",
  );
  if (
    !Array.isArray(record.artifactIds)
    || record.artifactIds.length < 1
    || record.artifactIds.length > 1_024
    || !record.artifactIds.every((value) => typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9._+/-]{0,255}$/.test(value))
  ) throw new Error("artifact_swarm_package_artifact_ids_are_invalid");
  const declaredArtifactIds = record.artifactIds as string[];
  if (new Set(declaredArtifactIds).size !== declaredArtifactIds.length) {
    throw new Error("artifact_swarm_package_artifact_ids_are_invalid");
  }
  const artifactIds = [...declaredArtifactIds].sort();
  if (artifactIds.some((value, index) => value !== declaredArtifactIds[index])) {
    throw new Error("artifact_swarm_package_artifact_ids_are_not_canonical");
  }
  const layerStart = boundedInteger(
    record.layerStart,
    0,
    1_000_000,
    "artifact_swarm_layer_range_is_invalid",
  );
  const layerEnd = boundedInteger(
    record.layerEnd,
    1,
    1_000_001,
    "artifact_swarm_layer_range_is_invalid",
  );
  if (layerEnd <= layerStart) throw new Error("artifact_swarm_layer_range_is_invalid");
  if (!Array.isArray(record.blobs) || record.blobs.length < 1 || record.blobs.length > MAX_BLOBS_PER_PACKAGE) {
    throw new Error("artifact_swarm_blobs_are_invalid");
  }
  const manifest = validateBlob(record.manifest);
  const blobs = record.blobs.map(validateBlob);
  const digests = new Set([manifest.sha256, ...blobs.map((blob) => blob.sha256)]);
  if (digests.size !== blobs.length + 1) {
    throw new Error("artifact_swarm_blob_is_duplicated");
  }
  return { packageId, artifactIds, layerStart, layerEnd, manifest, blobs };
}

export function selectArtifactSwarmPackages(
  manifest: SignedArtifactSwarmManifest,
  distributionManifestId: string,
  selectedArtifactIds: readonly string[],
): string[] {
  const verified = verifyArtifactSwarmManifest(manifest);
  if (verified.distributionManifestId !== distributionManifestId) {
    throw new Error("artifact_swarm_distribution_manifest_mismatch");
  }
  const requested = new Set(selectedArtifactIds);
  if (requested.size !== selectedArtifactIds.length || requested.size === 0) {
    throw new Error("artifact_swarm_selected_artifact_ids_are_invalid");
  }
  const packages = verified.packages.filter((entry) =>
    entry.artifactIds.some((artifactId) => requested.has(artifactId))
  );
  const covered = new Set(packages.flatMap(({ artifactIds }) => artifactIds));
  for (const artifactId of covered) {
    if (!requested.has(artifactId)) {
      throw new Error("artifact_swarm_package_contains_unassigned_artifact");
    }
  }
  for (const artifactId of requested) {
    if (!covered.has(artifactId)) throw new Error("artifact_swarm_selected_artifact_is_missing");
  }
  return packages.map(({ packageId }) => packageId);
}

function validateBlob(value: unknown): ArtifactBlobDescriptor {
  const record = strictRecord(value, [
    "sha256",
    "sizeBytes",
    "chunks",
  ], "artifact_swarm_blob_is_invalid");
  const sha256 = sha256Text(record.sha256, "artifact_swarm_blob_digest_is_invalid");
  const sizeBytes = boundedInteger(
    record.sizeBytes,
    1,
    Number.MAX_SAFE_INTEGER,
    "artifact_swarm_blob_size_is_invalid",
  );
  if (!Array.isArray(record.chunks) || record.chunks.length < 1 || record.chunks.length > MAX_CHUNKS_PER_BLOB) {
    throw new Error("artifact_swarm_chunks_are_invalid");
  }
  const chunks = record.chunks.map((chunk, expectedIndex) => {
    const chunkRecord = strictRecord(chunk, [
      "index",
      "offset",
      "sizeBytes",
      "sha256",
    ], "artifact_swarm_chunk_is_invalid");
    const index = boundedInteger(
      chunkRecord.index,
      0,
      MAX_CHUNKS_PER_BLOB - 1,
      "artifact_swarm_chunk_index_is_invalid",
    );
    const offset = boundedInteger(
      chunkRecord.offset,
      0,
      Number.MAX_SAFE_INTEGER,
      "artifact_swarm_chunk_offset_is_invalid",
    );
    const chunkSize = boundedInteger(
      chunkRecord.sizeBytes,
      1,
      Number.MAX_SAFE_INTEGER,
      "artifact_swarm_chunk_size_is_invalid",
    );
    if (index !== expectedIndex) throw new Error("artifact_swarm_chunks_are_not_ordered");
    return {
      index,
      offset,
      sizeBytes: chunkSize,
      sha256: sha256Text(
        chunkRecord.sha256,
        "artifact_swarm_chunk_digest_is_invalid",
      ),
    };
  });
  let expectedOffset = 0;
  for (const chunk of chunks) {
    if (chunk.offset !== expectedOffset) {
      throw new Error("artifact_swarm_chunks_are_not_contiguous");
    }
    expectedOffset += chunk.sizeBytes;
  }
  if (expectedOffset !== sizeBytes) {
    throw new Error("artifact_swarm_chunks_do_not_cover_blob");
  }
  return { sha256, sizeBytes, chunks };
}

function validatePeerAnnouncement(
  value: ArtifactPeerAnnouncement,
  now: number,
): ArtifactPeerAnnouncement {
  const record = strictRecord(value, [
    "peerId",
    "chunkIds",
    "expiresAt",
    "rttMs",
    "goodputMbps",
    "activeTransfers",
    "reliability",
  ], "artifact_peer_announcement_is_invalid");
  const peerId = text(record.peerId, 256, "artifact_peer_id_is_invalid");
  if (!Array.isArray(record.chunkIds) || record.chunkIds.length > MAX_PEER_CHUNKS) {
    throw new Error("artifact_peer_chunks_are_invalid");
  }
  const chunkIds = [...new Set(record.chunkIds.map((chunkId) =>
    parseChunkId(chunkId, "artifact_peer_chunk_id_is_invalid")
  ))].sort();
  const expiresAt = boundedInteger(
    record.expiresAt,
    1,
    Number.MAX_SAFE_INTEGER,
    "artifact_peer_expiry_is_invalid",
  );
  if (expiresAt <= now) throw new Error("artifact_peer_announcement_is_expired");
  return {
    peerId,
    chunkIds,
    expiresAt,
    rttMs: boundedNumber(record.rttMs, 0, 60_000, "artifact_peer_rtt_is_invalid"),
    goodputMbps: boundedNumber(
      record.goodputMbps,
      0.001,
      10_000_000,
      "artifact_peer_goodput_is_invalid",
    ),
    activeTransfers: boundedInteger(
      record.activeTransfers,
      0,
      100_000,
      "artifact_peer_transfer_count_is_invalid",
    ),
    reliability: boundedNumber(
      record.reliability,
      0,
      1,
      "artifact_peer_reliability_is_invalid",
    ),
  };
}

interface SelectedArtifactBlob {
  packageId: string;
  blob: ArtifactBlobDescriptor;
}

interface ArtifactCachePaths {
  artifactDirectory: string;
  chunkDirectory: string;
  target: string;
}

class ArtifactPeerCorruptionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ArtifactPeerCorruptionError";
  }
}

function validatePinnedArtifact(
  value: ArtifactSwarmPinnedArtifact,
): ArtifactSwarmPinnedArtifact {
  const label = text(value.label, 512, "artifact_swarm_artifact_label_is_invalid");
  const sha256 = sha256Text(
    value.sha256,
    "artifact_swarm_artifact_digest_is_invalid",
  );
  const sizeBytes = boundedInteger(
    value.sizeBytes,
    1,
    Number.MAX_SAFE_INTEGER,
    "artifact_swarm_artifact_size_is_invalid",
  );
  let source: URL;
  try {
    source = new URL(value.url);
  } catch {
    throw new Error("artifact_swarm_artifact_url_is_invalid");
  }
  if (
    source.protocol !== "https:"
    || source.hash !== `#sha256=${sha256}`
  ) {
    throw new Error("artifact_swarm_artifact_url_is_invalid");
  }
  return { label, url: source.toString(), sha256, sizeBytes };
}

function artifactCachePaths(
  artifact: ArtifactSwarmPinnedArtifact,
  cacheRoot: string,
): ArtifactCachePaths {
  const source = new URL(artifact.url);
  source.hash = "";
  let decodedName: string;
  try {
    decodedName = decodeURIComponent(basename(source.pathname));
  } catch {
    throw new Error("artifact_swarm_artifact_name_is_invalid");
  }
  if (!decodedName || decodedName === "." || decodedName === "..") {
    throw new Error("artifact_swarm_artifact_name_is_invalid");
  }
  const root = resolve(cacheRoot);
  const artifactDirectory = resolve(root, artifact.sha256);
  assertDirectChild(root, artifactDirectory, "artifact_swarm_cache_path_escaped");
  const target = resolve(artifactDirectory, decodedName);
  assertDirectChild(
    artifactDirectory,
    target,
    "artifact_swarm_cache_path_escaped",
  );
  const chunkDirectory = resolve(artifactDirectory, ".swarm-chunks");
  assertDirectChild(
    artifactDirectory,
    chunkDirectory,
    "artifact_swarm_cache_path_escaped",
  );
  return { artifactDirectory, chunkDirectory, target };
}

function selectArtifactBlob(
  manifest: SignedArtifactSwarmManifest | null,
  artifact: ArtifactSwarmPinnedArtifact,
  explicitPackageId: string | null | undefined,
): SelectedArtifactBlob | null {
  if (manifest === null) return null;
  const normalizedPackageId = explicitPackageId === null || explicitPackageId === undefined
    ? null
    : sha256Text(
      explicitPackageId,
      "artifact_swarm_package_id_is_invalid",
    );
  const matches = manifest.packages.flatMap((entry) => {
    if (normalizedPackageId !== null && entry.packageId !== normalizedPackageId) {
      return [];
    }
    return [entry.manifest, ...entry.blobs]
      .filter(
        (blob) =>
          blob.sha256 === artifact.sha256
          && blob.sizeBytes === artifact.sizeBytes,
      )
      .map((blob) => ({ packageId: entry.packageId, blob }));
  });
  return matches.length === 1 ? matches[0]! : null;
}

async function verifyPresentChunks(
  chunkDirectory: string,
  blob: ArtifactBlobDescriptor,
): Promise<Set<string>> {
  const present = new Set<string>();
  for (const chunk of blob.chunks) {
    const chunkId = artifactChunkId(blob.sha256, chunk.index);
    const path = chunkPath(chunkDirectory, chunk);
    if (await verifiedPath(path, chunk.sizeBytes, chunk.sha256)) {
      present.add(chunkId);
    } else {
      await rm(path, { force: true });
    }
  }
  return present;
}

function verifiedChunkBytes(
  blob: ArtifactBlobDescriptor,
  present: ReadonlySet<string>,
): number {
  return blob.chunks.reduce(
    (total, chunk) =>
      total + (
        present.has(artifactChunkId(blob.sha256, chunk.index))
          ? chunk.sizeBytes
          : 0
      ),
    0,
  );
}

async function fetchVerifiedPeerChunk(
  client: ArtifactPeerChunkClient,
  peerId: string,
  request: ArtifactChunkRequest,
  chunkDirectory: string,
): Promise<void> {
  const target = chunkPath(chunkDirectory, request.chunk);
  const partial = `${target}.partial`;
  assertDirectChild(chunkDirectory, partial, "artifact_swarm_cache_path_escaped");
  let partialBytes = 0;
  const digest = createHash("sha256");
  try {
    const metadata = await stat(partial);
    if (!metadata.isFile() || metadata.size >= request.chunk.sizeBytes) {
      await rm(partial, { force: true });
    } else {
      partialBytes = metadata.size;
      for await (const value of createReadStream(partial)) digest.update(value as Buffer);
    }
  } catch {
    partialBytes = 0;
  }
  const response = await client.fetchChunk({
    peerId,
    packageId: request.packageId,
    blobSha256: request.blobSha256,
    chunkId: request.chunkId,
    offset: request.chunk.offset + partialBytes,
    sizeBytes: request.chunk.sizeBytes - partialBytes,
  });
  if (
    response.contentLength !== undefined
    && (
      !Number.isSafeInteger(response.contentLength)
      || response.contentLength !== request.chunk.sizeBytes - partialBytes
    )
  ) {
    throw new ArtifactPeerCorruptionError(
      "artifact_swarm_peer_chunk_size_mismatch",
    );
  }
  if (
    response.body === null
    || typeof response.body !== "object"
    || !(Symbol.asyncIterator in response.body)
  ) {
    throw new ArtifactPeerCorruptionError(
      "artifact_swarm_peer_chunk_body_is_invalid",
    );
  }

  const writer = await open(partial, partialBytes === 0 ? "w" : "a");
  let bytesRead = partialBytes;
  try {
    for await (const value of response.body) {
      if (!(value instanceof Uint8Array)) {
        throw new ArtifactPeerCorruptionError(
          "artifact_swarm_peer_chunk_body_is_invalid",
        );
      }
      if (value.byteLength === 0) continue;
      bytesRead += value.byteLength;
      if (bytesRead > request.chunk.sizeBytes) {
        throw new ArtifactPeerCorruptionError(
          "artifact_swarm_peer_chunk_size_mismatch",
        );
      }
      digest.update(value);
      await writer.write(value);
    }
    await writer.sync();
  } catch (error) {
    await writer.close().catch(() => undefined);
    // A bounded prefix remains resumable by absolute artifact offset.
    throw error;
  }
  await writer.close();
  if (
    bytesRead !== request.chunk.sizeBytes
    || digest.digest("hex") !== request.chunk.sha256
  ) {
    await rm(partial, { force: true });
    throw new ArtifactPeerCorruptionError(
      "artifact_swarm_peer_chunk_digest_mismatch",
    );
  }

  if (await verifiedPath(target, request.chunk.sizeBytes, request.chunk.sha256)) {
    await rm(partial, { force: true });
    return;
  }
  await rm(target, { force: true });
  await rename(partial, target);
}

async function assembleVerifiedArtifact(
  chunkDirectory: string,
  target: string,
  artifact: ArtifactSwarmPinnedArtifact,
  blob: ArtifactBlobDescriptor,
): Promise<void> {
  const temporary = resolve(
    dirname(target),
    `.${basename(target)}.${randomUUID()}.assembling`,
  );
  assertDirectChild(
    dirname(target),
    temporary,
    "artifact_swarm_cache_path_escaped",
  );
  const writer = await open(temporary, "wx");
  const digest = createHash("sha256");
  let bytesWritten = 0;
  try {
    for (const chunk of blob.chunks) {
      const path = chunkPath(chunkDirectory, chunk);
      for await (const value of createReadStream(path)) {
        const bytes = value as Buffer;
        bytesWritten += bytes.byteLength;
        if (bytesWritten > artifact.sizeBytes) {
          throw new Error("artifact_swarm_assembled_size_mismatch");
        }
        digest.update(bytes);
        await writer.write(bytes);
      }
    }
    await writer.sync();
  } catch (error) {
    await writer.close().catch(() => undefined);
    await rm(temporary, { force: true });
    throw error;
  }
  await writer.close();
  if (
    bytesWritten !== artifact.sizeBytes
    || digest.digest("hex") !== artifact.sha256
  ) {
    await rm(temporary, { force: true });
    throw new Error("artifact_swarm_assembled_digest_mismatch");
  }
  await rm(target, { force: true });
  await rename(temporary, target);
}

async function verifiedOriginFallback(
  originDownloader: ArtifactOriginDownloader,
  artifact: ArtifactSwarmPinnedArtifact,
  cacheRoot: string,
  cache: ArtifactCachePaths,
  onProgress: (progress: ArtifactSwarmTransfer) => void,
  verifiedPeerBytes: number,
  resumed: boolean,
  peerFailure?: unknown,
): Promise<string> {
  try {
    const path = await originDownloader(artifact, cacheRoot, (progress) => {
      onProgress({
        ...progress,
        bytesDownloaded: Math.max(
          Math.min(artifact.sizeBytes, verifiedPeerBytes),
          progress.bytesDownloaded,
        ),
        resumed: resumed || progress.resumed,
      });
    });
    const candidate = resolve(path);
    if (
      dirname(candidate) !== cache.artifactDirectory
      || !(await verifiedFile(candidate, artifact))
    ) {
      throw new Error("artifact_swarm_origin_returned_unverified_artifact");
    }
    return candidate;
  } catch (error) {
    // Preserve the origin's actionable network/integrity error for callers,
    // while retaining the peer failure as diagnostic context where supported.
    if (
      peerFailure !== undefined
      && error instanceof Error
      && error.cause === undefined
    ) {
      Object.defineProperty(error, "cause", {
        configurable: true,
        value: peerFailure,
      });
    }
    throw error;
  }
}

function chunkPath(
  chunkDirectory: string,
  chunk: ArtifactChunkDescriptor,
): string {
  const target = resolve(
    chunkDirectory,
    `${chunk.index.toString().padStart(8, "0")}-${chunk.sha256}.chunk`,
  );
  assertDirectChild(
    chunkDirectory,
    target,
    "artifact_swarm_cache_path_escaped",
  );
  return target;
}

async function verifiedFile(
  path: string,
  artifact: ArtifactSwarmPinnedArtifact,
): Promise<boolean> {
  return verifiedPath(path, artifact.sizeBytes, artifact.sha256);
}

async function verifiedPath(
  path: string,
  expectedSize: number,
  expectedSha256: string,
): Promise<boolean> {
  let info;
  try {
    info = await stat(path);
  } catch {
    return false;
  }
  if (!info.isFile() || info.size !== expectedSize) return false;
  const digest = createHash("sha256");
  try {
    for await (const value of createReadStream(path)) {
      digest.update(value as Buffer);
    }
  } catch {
    return false;
  }
  return digest.digest("hex") === expectedSha256;
}

function swarmTransfer(
  bytesDownloaded: number,
  bytesTotal: number,
  startedAt: number,
  currentTime: number,
  resumed: boolean,
): ArtifactSwarmTransfer {
  const elapsedSeconds = Math.max(0, currentTime - startedAt) / 1_000;
  const bytesPerSecond = elapsedSeconds > 0 && bytesDownloaded > 0
    ? bytesDownloaded / elapsedSeconds
    : null;
  return {
    bytesDownloaded: Math.min(bytesTotal, bytesDownloaded),
    bytesTotal,
    bytesPerSecond,
    etaSeconds: bytesPerSecond === null
      ? null
      : Math.max(0, Math.ceil((bytesTotal - bytesDownloaded) / bytesPerSecond)),
    resumed,
  };
}

function completedSwarmTransfer(
  bytes: number,
  resumed: boolean,
): ArtifactSwarmTransfer {
  return {
    bytesDownloaded: bytes,
    bytesTotal: bytes,
    bytesPerSecond: null,
    etaSeconds: 0,
    resumed,
  };
}

function assertDirectChild(parent: string, child: string, code: string): void {
  if (dirname(resolve(child)) !== resolve(parent) || basename(child) === "") {
    throw new Error(code);
  }
}

function packageChunks(
  entry: ArtifactPackageDescriptor,
): Array<Omit<ArtifactChunkRequest, "sources">> {
  return [entry.manifest, ...entry.blobs].flatMap((blob) =>
    blob.chunks.map((chunk) => ({
      packageId: entry.packageId,
      blobSha256: blob.sha256,
      chunk,
      chunkId: artifactChunkId(blob.sha256, chunk.index),
    }))
  );
}

function peerScore(peer: StoredPeer, sizeBytes: number): number {
  const transferMs = (sizeBytes * 8) / (peer.announcement.goodputMbps * 1_000);
  const loadPenalty = peer.announcement.activeTransfers * 25;
  const reliabilityPenalty = (1 - peer.announcement.reliability) * 5_000;
  return roundScore(
    peer.announcement.rttMs + transferMs + loadPenalty + reliabilityPenalty,
  );
}

function canonicalBytes(value: unknown): Buffer {
  return Buffer.from(canonicalJson(value), "utf8");
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) {
    return `[${value.map((entry) => canonicalJson(entry)).join(",")}]`;
  }
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) =>
    `${JSON.stringify(key)}:${canonicalJson(record[key])}`
  ).join(",")}}`;
}

function sha256Identity(value: Uint8Array): string {
  return `sha256:${createHash("sha256").update(value).digest("hex")}`;
}

function sha256IdentityText(value: unknown, code: string): string {
  if (typeof value !== "string" || !/^sha256:[0-9a-f]{64}$/.test(value)) {
    throw new Error(code);
  }
  return value;
}

function sha256Text(value: unknown, code: string): string {
  if (typeof value !== "string") throw new Error(code);
  const normalized = value.startsWith("sha256:") ? value.slice(7) : value;
  if (!/^[0-9a-f]{64}$/.test(normalized)) throw new Error(code);
  return normalized;
}

function parseChunkId(value: unknown, code: string): string {
  if (typeof value !== "string") throw new Error(code);
  const match = /^([0-9a-f]{64}):([0-9]+)$/.exec(value);
  if (!match) throw new Error(code);
  return artifactChunkId(
    match[1]!,
    boundedInteger(
      Number(match[2]),
      0,
      MAX_CHUNKS_PER_BLOB - 1,
      code,
    ),
  );
}

function uniqueDigests(values: readonly string[], code: string): string[] {
  if (values.length < 1 || values.length > MAX_PACKAGES) throw new Error(code);
  const normalized = values.map((value) => sha256Text(value, code));
  if (new Set(normalized).size !== normalized.length) throw new Error(code);
  return normalized;
}

function canonicalBase64url(value: unknown, code: string): string {
  if (typeof value !== "string" || !/^[A-Za-z0-9_-]+$/.test(value)) {
    throw new Error(code);
  }
  const bytes = Buffer.from(value, "base64url");
  if (bytes.length === 0 || bytes.toString("base64url") !== value) throw new Error(code);
  return value;
}

function isPublicKeyObject(value: KeyLike): value is KeyObject {
  return (
    typeof value === "object"
    && value !== null
    && "type" in value
    && (value as { type?: unknown }).type === "public"
    && "export" in value
    && typeof (value as { export?: unknown }).export === "function"
  );
}

function strictRecord(
  value: unknown,
  keys: readonly string[],
  code: string,
): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(code);
  const record = value as Record<string, unknown>;
  const actual = Object.keys(record).sort();
  const expected = [...keys].sort();
  if (
    actual.length !== expected.length
    || actual.some((key, index) => key !== expected[index])
  ) {
    throw new Error(code);
  }
  return record;
}

function text(value: unknown, maximum: number, code: string): string {
  if (typeof value !== "string") throw new Error(code);
  const normalized = value.trim();
  if (
    !normalized
    || normalized.length > maximum
    || /[\u0000-\u001f\u007f]/.test(normalized)
  ) {
    throw new Error(code);
  }
  return normalized;
}

function boundedInteger(
  value: unknown,
  minimum: number,
  maximum: number,
  code: string,
): number {
  if (
    typeof value !== "number"
    || !Number.isSafeInteger(value)
    || value < minimum
    || value > maximum
  ) {
    throw new Error(code);
  }
  return value;
}

function boundedNumber(
  value: unknown,
  minimum: number,
  maximum: number,
  code: string,
): number {
  if (
    typeof value !== "number"
    || !Number.isFinite(value)
    || value < minimum
    || value > maximum
  ) {
    throw new Error(code);
  }
  return value;
}

function roundScore(value: number): number {
  return Math.round(value * 1_000) / 1_000;
}
