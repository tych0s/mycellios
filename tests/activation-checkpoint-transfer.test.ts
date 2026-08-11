import { createHash, generateKeyPairSync } from "node:crypto";
import { describe, expect, it } from "vitest";
import {
  signActivationCheckpoint,
  type ActivationCheckpointCompatibility,
} from "../src/contracts/activation-checkpoint.js";
import { ActivationCheckpointStore } from "../src/distribution/activation-checkpoint-store.js";
import {
  ACTIVATION_CHECKPOINT_CHUNK_BYTES,
  ActivationCheckpointTransferAuthority,
  activationCheckpointChunks,
} from "../src/coordinator/activation-checkpoint-transfer.js";
import { activationCheckpointRequestIdHash } from "../src/contracts/activation-checkpoint-transfer.js";

const { privateKey, publicKey } = generateKeyPairSync("ed25519");
const digest = (value: string | Uint8Array) =>
  `sha256:${createHash("sha256").update(value).digest("hex")}`;

const expected: ActivationCheckpointCompatibility = {
  keyId: "stage-key",
  requestIdHash: activationCheckpointRequestIdHash(64),
  nodeIdHash: digest("node"),
  topologyGeneration: 9,
  topologyDigest: digest("topology"),
  engineDescriptorDigest: digest("engine"),
  artifactManifestDigest: digest("artifact"),
  configurationDigest: digest("configuration"),
  stageId: "stage-1",
  layerStart: 0,
  layerEnd: 16,
  minimumCommittedPosition: 64,
};

function fixture(payload: Uint8Array) {
  return signActivationCheckpoint({
    schema: "mycellios-activation-checkpoint/1",
    requestIdHash: expected.requestIdHash,
    stageId: expected.stageId,
    nodeIdHash: digest("node"),
    topologyGeneration: expected.topologyGeneration,
    topologyDigest: expected.topologyDigest,
    engineDescriptorDigest: expected.engineDescriptorDigest,
    artifactManifestDigest: expected.artifactManifestDigest,
    configurationDigest: expected.configurationDigest,
    layerStart: expected.layerStart,
    layerEnd: expected.layerEnd,
    committedPosition: 64,
    payloadDigest: digest(payload),
    bytes: payload.byteLength,
    createdAt: 1_000,
    expiresAt: 20_000,
    keyId: "stage-key",
  }, privateKey);
}

function authority() {
  const store = new ActivationCheckpointStore(new Map([["stage-key", publicKey]]), {
    maxEntries: 4,
    maxEntryBytes: 2 * ACTIVATION_CHECKPOINT_CHUNK_BYTES,
    maxTotalBytes: 4 * ACTIVATION_CHECKPOINT_CHUNK_BYTES,
  });
  return { store, transfer: new ActivationCheckpointTransferAuthority(store) };
}

describe("authenticated activation checkpoint transfer", () => {
  it("assembles ordered bounded chunks and admits only after full verification", () => {
    const payload = Buffer.alloc(ACTIVATION_CHECKPOINT_CHUNK_BYTES + 17, 7);
    const checkpoint = fixture(payload);
    const { store, transfer } = authority();
    const transferId = transfer.expect({
      workerId: "worker-a", workerSessionId: "session-a", expected,
      maximumBytes: payload.byteLength, expiresAt: 10_000,
    }, 2_000);
    const chunks = activationCheckpointChunks(transferId, checkpoint, payload);
    transfer.begin("worker-a", "session-a", {
      transferId, checkpoint, chunkCount: chunks.length,
    }, 2_000);
    for (const chunk of chunks) transfer.chunk("worker-a", "session-a", chunk, 2_001);
    expect(transfer.commit("worker-a", "session-a", {
      transferId, checkpointId: checkpoint.checkpointId,
    }, 2_002)).toEqual(checkpoint);
    expect(store.get(checkpoint.checkpointId, expected, 2_003)?.payload.equals(payload)).toBe(true);
  });

  it("rejects cross-session, reordered, incomplete and corrupt transfers", () => {
    const payload = Buffer.alloc(ACTIVATION_CHECKPOINT_CHUNK_BYTES + 3, 11);
    const checkpoint = fixture(payload);
    const { transfer } = authority();
    const transferId = transfer.expect({
      workerId: "worker-a", workerSessionId: "session-a", expected,
      maximumBytes: payload.byteLength, expiresAt: 10_000,
    }, 2_000);
    const chunks = activationCheckpointChunks(transferId, checkpoint, payload);
    expect(() => transfer.begin("worker-a", "session-b", {
      transferId, checkpoint, chunkCount: chunks.length,
    }, 2_000)).toThrow(/session/);
    transfer.begin("worker-a", "session-a", {
      transferId, checkpoint, chunkCount: chunks.length,
    }, 2_000);
    expect(() => transfer.chunk("worker-a", "session-a", chunks[1], 2_001)).toThrow(/order/);
    transfer.chunk("worker-a", "session-a", chunks[0], 2_001);
    expect(() => transfer.commit("worker-a", "session-a", {
      transferId, checkpointId: checkpoint.checkpointId,
    }, 2_002)).toThrow(/incomplete/);
    const corrupt = { ...chunks[1]!, data: Buffer.alloc(3, 12).toString("base64") };
    transfer.chunk("worker-a", "session-a", corrupt, 2_003);
    expect(() => transfer.commit("worker-a", "session-a", {
      transferId, checkpointId: checkpoint.checkpointId,
    }, 2_004)).toThrow(/payload_is_corrupt/);
  });

  it("expires expectations and zero-aborts every transfer from a dead session", () => {
    const { transfer } = authority();
    const one = transfer.expect({
      workerId: "worker-a", workerSessionId: "session-a", expected,
      maximumBytes: 32, expiresAt: 3_000,
    }, 2_000);
    transfer.expect({
      workerId: "worker-a", workerSessionId: "session-a", expected,
      maximumBytes: 32, expiresAt: 4_000,
    }, 2_000);
    expect(() => transfer.begin("worker-a", "session-a", {
      transferId: one, checkpoint: fixture(Buffer.alloc(32)), chunkCount: 1,
    }, 3_000)).toThrow(/expired/);
    expect(transfer.abortSession("worker-a", "session-a")).toBe(1);
  });

  it("binds restore acknowledgement to the target worker session and checkpoint", () => {
    const payload = Buffer.from("admitted-restore");
    const checkpoint = fixture(payload);
    const { store, transfer } = authority();
    store.put(checkpoint, payload, expected, 2_000);
    const prepared = transfer.prepareRestore({
      workerId: "worker-b",
      workerSessionId: "session-b",
      checkpointId: checkpoint.checkpointId,
      expected,
      expiresAt: 5_000,
    }, 2_001);
    expect(prepared.payload.equals(payload)).toBe(true);
    expect(() => transfer.completeRestore(
      "worker-b", "stale-session", prepared.transferId, checkpoint.checkpointId, 2_002,
    )).toThrow(/session/);
    expect(() => transfer.completeRestore(
      "worker-b", "session-b", prepared.transferId, digest("wrong"), 2_002,
    )).toThrow(/identity/);
    expect(transfer.completeRestore(
      "worker-b", "session-b", prepared.transferId, checkpoint.checkpointId, 2_002,
    )).toBe(checkpoint.checkpointId);

    const abandoned = transfer.prepareRestore({
      workerId: "worker-b", workerSessionId: "session-b",
      checkpointId: checkpoint.checkpointId, expected, expiresAt: 5_000,
    }, 2_003);
    expect(transfer.abortSession("worker-b", "session-b")).toBe(1);
    expect(() => transfer.completeRestore(
      "worker-b", "session-b", abandoned.transferId, checkpoint.checkpointId, 2_004,
    )).toThrow(/unknown/);
  });
});
