import { generateKeyPairSync } from "node:crypto";
import { describe, expect, it } from "vitest";
import {
  ARTIFACT_SWARM_SCHEMA,
  ArtifactSwarmRegistry,
  artifactChunkId,
  selectArtifactSwarmPackages,
  signArtifactSwarmManifest,
  verifyArtifactSwarmManifest,
  type ArtifactBlobDescriptor,
  type UnsignedArtifactSwarmManifest,
} from "../src/model-fabric/artifact-swarm.js";

describe("native artifact swarm", () => {
  it("seals a strict manifest and rejects modified package contracts", () => {
    const keys = generateKeyPairSync("ed25519");
    const signed = signArtifactSwarmManifest(manifest(), keys.privateKey, keys.publicKey);
    expect(verifyArtifactSwarmManifest(signed)).toEqual(signed);

    const tampered = structuredClone(signed);
    tampered.sourceRevision = "commit-tampered";
    expect(() => verifyArtifactSwarmManifest(tampered)).toThrow(
      "artifact_swarm_manifest_identity_mismatch",
    );
  });

  it("rejects a valid self-signed manifest from an unpinned publisher", () => {
    const trusted = generateKeyPairSync("ed25519");
    const attacker = generateKeyPairSync("ed25519");
    const registry = new ArtifactSwarmRegistry({ trustedPublisherKeys: [trusted.publicKey] });
    expect(() => registry.publish(
      signArtifactSwarmManifest(manifest(), attacker.privateKey, attacker.publicKey),
    )).toThrow("artifact_swarm_publisher_is_not_trusted");
  });

  it("prioritizes the requested stage path, then rare chunks and measured peers", () => {
    const keys = generateKeyPairSync("ed25519");
    const registry = new ArtifactSwarmRegistry({ trustedPublisherKeys: [keys.publicKey] });
    const signed = registry.publish(
      signArtifactSwarmManifest(manifest(), keys.privateKey, keys.publicKey),
    );
    const first = signed.packages[0]!;
    const second = signed.packages[1]!;
    const firstChunk = artifactChunkId(first.blobs[0]!.sha256, 0);
    const secondRare = artifactChunkId(second.blobs[0]!.sha256, 1);
    const secondCommon = artifactChunkId(second.blobs[0]!.sha256, 0);
    const knownManifests = new Set([
      artifactChunkId(first.manifest.sha256, 0),
      artifactChunkId(second.manifest.sha256, 0),
    ]);
    const now = 50_000;
    registry.announcePeer(peer("fast", [firstChunk, secondCommon], now, {
      rttMs: 10,
      goodputMbps: 1_000,
    }), now);
    registry.announcePeer(peer("slow", [firstChunk, secondCommon, secondRare], now, {
      rttMs: 90,
      goodputMbps: 100,
    }), now);

    const plan = registry.planDownload({
      manifestId: signed.manifestId,
      requesterPeerId: "requester",
      packageIds: [second.packageId, first.packageId],
      requesterChunkIds: knownManifests,
      now,
    });
    expect(plan[0]?.packageId).toBe(second.packageId);
    expect(plan[0]?.chunkId).toBe(secondRare);
    const common = plan.find((request) => request.chunkId === secondCommon);
    expect(common?.sources.map((source) => source.peerId)).toEqual(["fast", "slow"]);
    const firstRequest = plan.find((request) => request.chunkId === firstChunk);
    expect(firstRequest?.sources[0]?.peerId).toBe("fast");
  });

  it("quarantines corrupt peers and expires stale inventories", () => {
    const keys = generateKeyPairSync("ed25519");
    const registry = new ArtifactSwarmRegistry({ trustedPublisherKeys: [keys.publicKey] });
    const signed = registry.publish(
      signArtifactSwarmManifest(manifest(), keys.privateKey, keys.publicKey),
    );
    const packageEntry = signed.packages[0]!;
    const chunkId = artifactChunkId(packageEntry.blobs[0]!.sha256, 0);
    const now = 100_000;
    registry.announcePeer(peer("bad", [chunkId], now), now);
    expect(registry.recordCorruptChunk("bad", now + 1)).toBe(false);
    expect(registry.recordCorruptChunk("bad", now + 2)).toBe(true);
    expect(registry.peerState("bad")).toMatchObject({
      corruptionStrikes: 2,
      quarantined: true,
    });
    expect(registry.planDownload({
      manifestId: signed.manifestId,
      requesterPeerId: "requester",
      packageIds: [packageEntry.packageId],
      now: now + 3,
    }).every((request) => request.sources.length === 0)).toBe(true);

    registry.announcePeer(peer("short-lived", [chunkId], now, {
      expiresAt: now + 10,
    }), now);
    expect(registry.expirePeers(now + 11)).toBe(1);
    expect(registry.peerState("short-lived")).toBeNull();
  });

  it("omits chunks already verified by the requester", () => {
    const keys = generateKeyPairSync("ed25519");
    const registry = new ArtifactSwarmRegistry({ trustedPublisherKeys: [keys.publicKey] });
    const signed = registry.publish(
      signArtifactSwarmManifest(manifest(), keys.privateKey, keys.publicKey),
    );
    const packageEntry = signed.packages[0]!;
    const alreadyPresent = artifactChunkId(packageEntry.manifest.sha256, 0);
    const plan = registry.planDownload({
      manifestId: signed.manifestId,
      requesterPeerId: "requester",
      packageIds: [packageEntry.packageId],
      requesterChunkIds: new Set([alreadyPresent]),
      now: 1,
    });
    expect(plan.some((request) => request.chunkId === alreadyPresent)).toBe(false);
  });

  it("maps only the selected distribution artifacts without widening package bytes", () => {
    const keys = generateKeyPairSync("ed25519");
    const signed = signArtifactSwarmManifest(manifest(), keys.privateKey, keys.publicKey);
    expect(selectArtifactSwarmPackages(
      signed,
      signed.distributionManifestId,
      ["layer-2-4", "output"],
    )).toEqual([signed.packages[1]!.packageId]);
    expect(() => selectArtifactSwarmPackages(
      signed,
      signed.distributionManifestId,
      ["layer-0-2"],
    )).toThrow("artifact_swarm_package_contains_unassigned_artifact");
  });
});

function manifest(): UnsignedArtifactSwarmManifest {
  return {
    schema: ARTIFACT_SWARM_SCHEMA,
    modelIdentity: identity(1, true),
    distributionManifestId: identity(2, true),
    sourceRevision: "commit-abc",
    tensorAbi: "mycellios-transformers-global-stage-tensors/1",
    packages: [
      {
        packageId: identity(10),
        artifactIds: ["input", "layer-0-2", "shared"],
        layerStart: 0,
        layerEnd: 2,
        manifest: blob(20, [200]),
        blobs: [blob(21, [1_024, 1_024])],
      },
      {
        packageId: identity(11),
        artifactIds: ["layer-2-4", "output"],
        layerStart: 2,
        layerEnd: 4,
        manifest: blob(22, [200]),
        blobs: [blob(23, [1_024, 1_024])],
      },
    ],
  };
}

function blob(seed: number, sizes: number[]): ArtifactBlobDescriptor {
  let offset = 0;
  return {
    sha256: identity(seed),
    sizeBytes: sizes.reduce((sum, size) => sum + size, 0),
    chunks: sizes.map((sizeBytes, index) => {
      const chunk = {
        index,
        offset,
        sizeBytes,
        sha256: identity(seed * 100 + index),
      };
      offset += sizeBytes;
      return chunk;
    }),
  };
}

function peer(
  peerId: string,
  chunkIds: string[],
  now: number,
  overrides: Partial<{
    expiresAt: number;
    rttMs: number;
    goodputMbps: number;
    activeTransfers: number;
    reliability: number;
  }> = {},
) {
  return {
    peerId,
    chunkIds,
    expiresAt: overrides.expiresAt ?? now + 60_000,
    rttMs: overrides.rttMs ?? 20,
    goodputMbps: overrides.goodputMbps ?? 500,
    activeTransfers: overrides.activeTransfers ?? 0,
    reliability: overrides.reliability ?? 0.99,
  };
}

function identity(seed: number, prefix = false): string {
  const digest = seed.toString(16).padStart(64, "0");
  return prefix ? `sha256:${digest}` : digest;
}
