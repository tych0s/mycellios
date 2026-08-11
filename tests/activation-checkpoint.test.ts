import { createHash, generateKeyPairSync } from "node:crypto";
import { describe, expect, it } from "vitest";
import {
  signActivationCheckpoint,
  verifyActivationCheckpoint,
  type ActivationCheckpointCompatibility,
} from "../src/contracts/activation-checkpoint.js";
import { ActivationCheckpointStore } from "../src/distribution/activation-checkpoint-store.js";

const { privateKey, publicKey } = generateKeyPairSync("ed25519");
const digest = (label: string) => `sha256:${createHash("sha256").update(label).digest("hex")}`;
const bytesDigest = (value: Uint8Array) =>
  `sha256:${createHash("sha256").update(value).digest("hex")}`;

function fixture(payload: Uint8Array, suffix = "a") {
  return signActivationCheckpoint({
    schema: "mycellios-activation-checkpoint/1",
    requestIdHash: digest(`request-${suffix}`),
    stageId: `stage-${suffix}`,
    nodeIdHash: digest(`node-${suffix}`),
    topologyGeneration: 7,
    topologyDigest: digest("topology"),
    engineDescriptorDigest: digest("engine"),
    artifactManifestDigest: digest("artifact"),
    configurationDigest: digest("configuration"),
    layerStart: 12,
    layerEnd: 24,
    committedPosition: 128,
    payloadDigest: bytesDigest(payload),
    bytes: payload.byteLength,
    createdAt: 1_000,
    expiresAt: 10_000,
    keyId: "stage-key",
  }, privateKey);
}

function expected(suffix = "a"): ActivationCheckpointCompatibility {
  return {
    keyId: "stage-key",
    requestIdHash: digest(`request-${suffix}`),
    nodeIdHash: digest(`node-${suffix}`),
    topologyGeneration: 7,
    topologyDigest: digest("topology"),
    engineDescriptorDigest: digest("engine"),
    artifactManifestDigest: digest("artifact"),
    configurationDigest: digest("configuration"),
    stageId: `stage-${suffix}`,
    layerStart: 12,
    layerEnd: 24,
    minimumCommittedPosition: 100,
  };
}

function payload(label: string): Buffer {
  return Buffer.from(label);
}

describe("bounded activation checkpoints", () => {
  it("resolves the pinned signing key lazily without accepting another key id", () => {
    const bytes = payload("lazy-key");
    const checkpoint = fixture(bytes);
    const resolver = (keyId: string) => keyId === "stage-key" ? publicKey : undefined;
    const store = new ActivationCheckpointStore(resolver, {
      maxEntries: 1, maxEntryBytes: 1_024, maxTotalBytes: 1_024,
    });
    expect(store.put(checkpoint, bytes, expected(), 2_000).checkpointId)
      .toBe(checkpoint.checkpointId);
    expect(() => store.put(checkpoint, bytes, { ...expected(), keyId: "unknown" }, 2_000))
      .toThrow(/keyId_is_incompatible|signature_is_invalid/);
  });

  it("seals engine, artifact, configuration and topology identity", () => {
    const bytes = payload("kv-pages");
    const checkpoint = fixture(bytes);
    expect(verifyActivationCheckpoint(
      checkpoint,
      bytes,
      new Map([["stage-key", publicKey]]),
      expected(),
      2_000,
    )).toEqual(checkpoint);
  });

  it.each([
    ["keyId", "other-stage-key"],
    ["requestIdHash", digest("other-request")],
    ["nodeIdHash", digest("other-node")],
    ["engineDescriptorDigest", digest("other-engine")],
    ["artifactManifestDigest", digest("other-artifact")],
    ["configurationDigest", digest("other-config")],
    ["topologyGeneration", 8],
    ["layerEnd", 25],
  ] as const)("rejects incompatible %s", (key, value) => {
    const bytes = payload("kv-pages");
    expect(() => verifyActivationCheckpoint(
      fixture(bytes), bytes, new Map([["stage-key", publicKey]]),
      { ...expected(), [key]: value }, 2_000,
    )).toThrow(/incompatible/);
  });

  it("rejects corruption, expiry, unknown keys and metadata tampering", () => {
    const bytes = payload("kv-pages");
    const checkpoint = fixture(bytes);
    expect(() => verifyActivationCheckpoint(
      checkpoint, payload("corrupt!"), new Map([["stage-key", publicKey]]), expected(), 2_000,
    )).toThrow(/payload_is_corrupt|size_is_invalid/);
    expect(() => verifyActivationCheckpoint(
      checkpoint, bytes, new Map([["stage-key", publicKey]]), expected(), 10_000,
    )).toThrow(/expired/);
    expect(() => verifyActivationCheckpoint(
      checkpoint, bytes, new Map(), expected(), 2_000,
    )).toThrow(/signature/);
    expect(() => verifyActivationCheckpoint(
      { ...checkpoint, committedPosition: 129 }, bytes,
      new Map([["stage-key", publicKey]]), expected(), 2_000,
    )).toThrow(/identity/);
  });

  it("bounds memory, evicts least-recently-used entries and returns copies", () => {
    const store = new ActivationCheckpointStore(new Map([["stage-key", publicKey]]), {
      maxEntries: 2,
      maxEntryBytes: 16,
      maxTotalBytes: 16,
    });
    const firstBytes = payload("first");
    const secondBytes = payload("second");
    const thirdBytes = payload("third");
    const first = fixture(firstBytes, "a");
    const second = fixture(secondBytes, "b");
    const third = fixture(thirdBytes, "c");
    store.put(first, firstBytes, expected("a"), 2_000);
    store.put(second, secondBytes, expected("b"), 2_001);
    const read = store.get(first.checkpointId, expected("a"), 2_002)!;
    read.payload.fill(0);
    store.put(third, thirdBytes, expected("c"), 2_003);
    expect(store.get(second.checkpointId, expected("b"), 2_004)).toBeNull();
    expect(store.get(first.checkpointId, expected("a"), 2_005)!.payload.equals(firstBytes)).toBe(true);
    expect(store.snapshot()).toEqual({ entries: 2, totalBytes: 10 });
  });
});
