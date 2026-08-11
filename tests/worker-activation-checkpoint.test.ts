import { createHash, generateKeyPairSync } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import { signActivationCheckpoint } from "../src/contracts/activation-checkpoint.js";
import { workerConfigSchema } from "../src/contracts/schemas.js";
import { WorkerAgent } from "../src/worker/agent.js";
import {
  generateWorkerAdmissionCredential,
  workerAdmissionSigner,
} from "../src/worker/admission-credential.js";
import { activationCheckpointRequestIdHash } from "../src/contracts/activation-checkpoint-transfer.js";
import { activationCheckpointChunks } from "../src/contracts/activation-checkpoint-transfer.js";

const digest = (value: string | Uint8Array) =>
  `sha256:${createHash("sha256").update(value).digest("hex")}`;

const config = workerConfigSchema.parse({
  region: "test",
  offeredVramMb: 512,
  limits: { maxConcurrency: 1, pauseWhenForeground: false },
  adapter: { kind: "mock", developmentOnly: true, model: "test",
    tokensPerSecond: 1, ttftMs: 1, failureRate: 0 },
  deployment: { contextLimit: 1_024 },
});

describe("worker activation checkpoint publisher", () => {
  it("reassembles a bounded restore and mutates exactly one compatible warm process", async () => {
    const now = Date.now();
    const payload = Buffer.alloc(300_000, 19);
    const hash = digest("restore-identity");
    const expected = {
      keyId: "source-stage-key",
      requestIdHash: activationCheckpointRequestIdHash(31),
      nodeIdHash: digest("source-node"),
      topologyGeneration: 7,
      topologyDigest: hash,
      engineDescriptorDigest: hash,
      artifactManifestDigest: hash,
      configurationDigest: hash,
      stageId: "stage-warm",
      layerStart: 8,
      layerEnd: 16,
      minimumCommittedPosition: 31,
    };
    const { privateKey } = generateKeyPairSync("ed25519");
    const checkpoint = signActivationCheckpoint({
      schema: "mycellios-activation-checkpoint/1",
      requestIdHash: expected.requestIdHash,
      nodeIdHash: expected.nodeIdHash,
      topologyGeneration: expected.topologyGeneration,
      topologyDigest: hash,
      engineDescriptorDigest: hash,
      artifactManifestDigest: hash,
      configurationDigest: hash,
      stageId: expected.stageId,
      layerStart: 8,
      layerEnd: 16,
      committedPosition: 31,
      payloadDigest: digest(payload),
      bytes: payload.byteLength,
      createdAt: now,
      expiresAt: now + 20_000,
      keyId: expected.keyId,
    }, privateKey);
    let restoredPayload: Buffer | null = null;
    const restoreActivationCheckpoint = vi.fn(async (
      _requestId: number, value: Uint8Array,
    ) => { restoredPayload = Buffer.from(value); });
    const agent = new WorkerAgent(config, {
      coordinatorUrl: "http://127.0.0.1:9999",
      reconnect: false,
      logger: { info() {}, warn() {}, error() {} },
    });
    const sent: Array<{ type: string; payload: any }> = [];
    const harness = agent as unknown as {
      handleServerMessage(input: unknown): Promise<void>;
      sendMessage(type: string, payload: unknown): boolean;
      runtimeProcesses: Map<string, { restoreActivationCheckpoint: typeof restoreActivationCheckpoint }>;
      runtimeProcessStages: Map<string, string>;
    };
    harness.sendMessage = vi.fn((type, value) => {
      sent.push({ type, payload: value });
      return true;
    });
    harness.runtimeProcesses.set("warm-launch", { restoreActivationCheckpoint });
    harness.runtimeProcessStages.set("warm-launch", expected.stageId);
    const transferId = "fc95781e-6133-4b03-a0fa-728873c932fe";
    const chunks = activationCheckpointChunks(transferId, checkpoint, payload);
    await harness.handleServerMessage({
      v: 1,
      type: "runtime.checkpoint.restore.begin",
      payload: {
        transferId,
        targetLaunchRequestId: "warm-launch",
        targetStageRequestId: 31,
        expected,
        checkpoint,
        chunkCount: chunks.length,
        maximumBytes: payload.byteLength,
        expiresAt: now + 10_000,
      },
    });
    for (const chunk of chunks) {
      await harness.handleServerMessage({
        v: 1, type: "runtime.checkpoint.restore.chunk", payload: chunk,
      });
    }
    await harness.handleServerMessage({
      v: 1,
      type: "runtime.checkpoint.restore.commit",
      payload: { transferId, checkpointId: checkpoint.checkpointId },
    });
    expect(restoreActivationCheckpoint).toHaveBeenCalledOnce();
    expect(restoreActivationCheckpoint.mock.calls[0]).toEqual([
      31, expect.any(Buffer), 31, payload.byteLength,
    ]);
    expect(restoredPayload).not.toBeNull();
    expect(Buffer.from(restoredPayload as unknown as Uint8Array).equals(payload)).toBe(true);
    expect(sent).toContainEqual({
      type: "runtime.checkpoint.restored",
      payload: { transferId, checkpointId: checkpoint.checkpointId },
    });

    const rejectedTransferId = "5ca5f60a-11da-40c2-89e4-67d50bac760d";
    const rejectedChunks = activationCheckpointChunks(rejectedTransferId, checkpoint, payload);
    await harness.handleServerMessage({
      v: 1,
      type: "runtime.checkpoint.restore.begin",
      payload: {
        transferId: rejectedTransferId,
        targetLaunchRequestId: "warm-launch",
        targetStageRequestId: 31,
        expected,
        checkpoint,
        chunkCount: rejectedChunks.length,
        maximumBytes: payload.byteLength,
        expiresAt: now + 10_000,
      },
    });
    await harness.handleServerMessage({
      v: 1, type: "runtime.checkpoint.restore.chunk", payload: rejectedChunks[1],
    });
    expect(restoreActivationCheckpoint).toHaveBeenCalledOnce();
    expect(sent).toContainEqual({
      type: "runtime.checkpoint.restore.failed",
      payload: {
        transferId: rejectedTransferId,
        checkpointId: checkpoint.checkpointId,
        code: "checkpoint_restore_incompatible",
      },
    });
  });

  it("seals live capture output with the enrolled device key", async () => {
    const credential = generateWorkerAdmissionCredential();
    const signer = workerAdmissionSigner(credential);
    const keyId = digest(Buffer.from(signer.publicKey.spki, "base64url"));
    const now = Date.now();
    const payload = Buffer.from("live-python-kv");
    const hash = digest("identity");
    const expected = {
      keyId,
      requestIdHash: activationCheckpointRequestIdHash(9),
      nodeIdHash: digest("node-live"),
      topologyGeneration: 2,
      topologyDigest: hash,
      engineDescriptorDigest: hash,
      artifactManifestDigest: hash,
      configurationDigest: hash,
      stageId: "stage-live",
      layerStart: 4,
      layerEnd: 8,
      minimumCommittedPosition: 9,
    };
    const agent = new WorkerAgent(config, {
      coordinatorUrl: "http://127.0.0.1:9999",
      reconnect: false,
      admissionSigner: signer,
      activationCheckpointCapture: vi.fn(async () => ({ payload, committedPosition: 9 })),
      logger: { info() {}, warn() {}, error() {} },
    });
    const sent: Array<{ type: string; payload: any }> = [];
    const harness = agent as unknown as {
      handleServerMessage(input: unknown): Promise<void>;
      sendMessage(type: string, payload: unknown): boolean;
    };
    harness.sendMessage = vi.fn((type, value) => {
      sent.push({ type, payload: value });
      return true;
    });
    await harness.handleServerMessage({
      v: 1,
      type: "runtime.checkpoint.request",
      payload: {
        transferId: "af595536-6121-4aad-85e3-d315efb3a699",
        stageRequestId: 9,
        expected,
        maximumBytes: 1_024,
        expiresAt: now + 10_000,
      },
    });
    const begin = sent.find(({ type }) => type === "runtime.checkpoint.begin");
    expect(begin?.payload.checkpoint).toMatchObject({
      keyId,
      requestIdHash: expected.requestIdHash,
      nodeIdHash: expected.nodeIdHash,
      committedPosition: 9,
      payloadDigest: digest(payload),
    });
    expect(begin?.payload.checkpoint.signature).toMatch(/^[A-Za-z0-9_-]{86}$/);
  });

  it("publishes a compatible checkpoint in bounded ordered envelopes", async () => {
    const now = Date.now();
    const payload = Buffer.alloc(300_000, 7);
    const expected = {
      keyId: "stage-key",
      requestIdHash: activationCheckpointRequestIdHash(21),
      nodeIdHash: digest("node"),
      topologyGeneration: 4,
      topologyDigest: digest("topology"),
      engineDescriptorDigest: digest("engine"),
      artifactManifestDigest: digest("artifact"),
      configurationDigest: digest("configuration"),
      stageId: "stage-a",
      layerStart: 2,
      layerEnd: 9,
      minimumCommittedPosition: 21,
    };
    const { privateKey } = generateKeyPairSync("ed25519");
    const checkpoint = signActivationCheckpoint({
      schema: "mycellios-activation-checkpoint/1",
      requestIdHash: expected.requestIdHash,
      nodeIdHash: digest("node"),
      topologyGeneration: expected.topologyGeneration,
      topologyDigest: expected.topologyDigest,
      engineDescriptorDigest: expected.engineDescriptorDigest,
      artifactManifestDigest: expected.artifactManifestDigest,
      configurationDigest: expected.configurationDigest,
      stageId: expected.stageId,
      layerStart: expected.layerStart,
      layerEnd: expected.layerEnd,
      committedPosition: 21,
      payloadDigest: digest(payload),
      bytes: payload.byteLength,
      createdAt: now,
      expiresAt: now + 20_000,
      keyId: "stage-key",
    }, privateKey);
    const provider = vi.fn(async () => ({ checkpoint, payload }));
    const agent = new WorkerAgent(config, {
      coordinatorUrl: "http://127.0.0.1:9999",
      reconnect: false,
      activationCheckpointProvider: provider,
      logger: { info() {}, warn() {}, error() {} },
    });
    const sent: Array<{ type: string; payload: unknown }> = [];
    const harness = agent as unknown as {
      handleServerMessage(input: unknown): Promise<void>;
      sendMessage(type: string, payload: unknown): boolean;
    };
    harness.sendMessage = vi.fn((type, value) => {
      sent.push({ type, payload: value });
      return true;
    });
    const transferId = "6ae89c75-a41a-407d-9634-0d42d83bd965";
    await harness.handleServerMessage({
      v: 1,
      type: "runtime.checkpoint.request",
      payload: { transferId, stageRequestId: 21, expected, maximumBytes: payload.byteLength, expiresAt: now + 10_000 },
    });
    expect(provider).toHaveBeenCalledOnce();
    expect(sent.map(({ type }) => type)).toEqual([
      "runtime.checkpoint.begin",
      "runtime.checkpoint.chunk",
      "runtime.checkpoint.chunk",
      "runtime.checkpoint.commit",
    ]);
    expect(sent.at(-1)?.payload).toEqual({ transferId, checkpointId: checkpoint.checkpointId });
  });

  it("fails closed when no checkpoint provider is installed", async () => {
    const agent = new WorkerAgent(config, {
      coordinatorUrl: "http://127.0.0.1:9999",
      reconnect: false,
      logger: { info() {}, warn() {}, error() {} },
    });
    const sendMessage = vi.fn(() => true);
    const harness = agent as unknown as {
      handleServerMessage(input: unknown): Promise<void>;
      sendMessage: typeof sendMessage;
    };
    harness.sendMessage = sendMessage;
    const hash = digest("same");
    await harness.handleServerMessage({
      v: 1,
      type: "runtime.checkpoint.request",
      payload: {
        transferId: "7a07fc45-eb61-42e0-b553-d444262f4891",
        expected: {
          keyId: "stage-key",
          requestIdHash: activationCheckpointRequestIdHash(1),
          nodeIdHash: hash,
          topologyGeneration: 1,
          topologyDigest: hash,
          engineDescriptorDigest: hash,
          artifactManifestDigest: hash,
          configurationDigest: hash,
          stageId: "stage-a",
          layerStart: 0,
          layerEnd: 1,
        },
        maximumBytes: 1,
        stageRequestId: 1,
        expiresAt: Date.now() + 10_000,
      },
    });
    expect(sendMessage).toHaveBeenCalledWith("runtime.checkpoint.failed", {
      transferId: "7a07fc45-eb61-42e0-b553-d444262f4891",
      code: "checkpoint_provider_unavailable",
    });
  });
});
