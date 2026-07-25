import {
  createHash,
  createPublicKey,
  sign as signBytes,
  verify as verifyBytes,
  type KeyObject,
  type KeyLike,
} from "node:crypto";

export const ARTIFACT_SWARM_SCHEMA = "mycellios-artifact-swarm/1" as const;

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
  layerStart: number;
  layerEnd: number;
  manifest: ArtifactBlobDescriptor;
  blobs: ArtifactBlobDescriptor[];
}

export interface UnsignedArtifactSwarmManifest {
  schema: typeof ARTIFACT_SWARM_SCHEMA;
  modelIdentity: string;
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

  publish(value: SignedArtifactSwarmManifest): SignedArtifactSwarmManifest {
    const manifest = verifyArtifactSwarmManifest(value);
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
    sourceRevision,
    tensorAbi,
    packages,
  };
}

function validatePackage(value: unknown): ArtifactPackageDescriptor {
  const record = strictRecord(value, [
    "packageId",
    "layerStart",
    "layerEnd",
    "manifest",
    "blobs",
  ], "artifact_swarm_package_is_invalid");
  const packageId = sha256Text(
    record.packageId,
    "artifact_swarm_package_id_is_invalid",
  );
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
  return { packageId, layerStart, layerEnd, manifest, blobs };
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
