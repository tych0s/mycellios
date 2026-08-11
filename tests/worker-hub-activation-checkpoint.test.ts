import { createHash, generateKeyPairSync } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import { signActivationCheckpoint } from "../src/contracts/activation-checkpoint.js";
import { ActivationCheckpointTransferAuthority, activationCheckpointChunks } from "../src/coordinator/activation-checkpoint-transfer.js";
import { WorkerHub } from "../src/coordinator/worker-hub.js";
import { ActivationCheckpointStore } from "../src/distribution/activation-checkpoint-store.js";
import type { MeshStore } from "../src/storage/store.js";
import { activationCheckpointRequestIdHash } from "../src/contracts/activation-checkpoint-transfer.js";

const { privateKey, publicKey } = generateKeyPairSync("ed25519");
const digest = (value: string | Uint8Array) =>
  `sha256:${createHash("sha256").update(value).digest("hex")}`;

describe("WorkerHub activation checkpoint channel", () => {
  it("binds a complete transfer to the authenticated websocket session", async () => {
    const now = Date.now();
    const payload = Buffer.from("bounded-live-kv");
    const expected = {
      keyId: "stage-key",
      requestIdHash: activationCheckpointRequestIdHash(12),
      nodeIdHash: digest("node"),
      topologyGeneration: 3,
      topologyDigest: digest("topology"),
      engineDescriptorDigest: digest("engine"),
      artifactManifestDigest: digest("artifact"),
      configurationDigest: digest("configuration"),
      stageId: "stage-1",
      layerStart: 0,
      layerEnd: 8,
      minimumCommittedPosition: 12,
    };
    const checkpoint = signActivationCheckpoint({
      schema: "mycellios-activation-checkpoint/1",
      requestIdHash: expected.requestIdHash,
      stageId: "stage-1",
      nodeIdHash: digest("node"),
      topologyGeneration: 3,
      topologyDigest: expected.topologyDigest,
      engineDescriptorDigest: expected.engineDescriptorDigest,
      artifactManifestDigest: expected.artifactManifestDigest,
      configurationDigest: expected.configurationDigest,
      layerStart: 0,
      layerEnd: 8,
      committedPosition: 12,
      payloadDigest: digest(payload),
      bytes: payload.byteLength,
      createdAt: now - 1_000,
      expiresAt: now + 20_000,
      keyId: "stage-key",
    }, privateKey);
    const store = new ActivationCheckpointStore(new Map([["stage-key", publicKey]]), {
      maxEntries: 2, maxEntryBytes: 1_024, maxTotalBytes: 2_048,
    });
    const transfers = new ActivationCheckpointTransferAuthority(store);
    const worker = {
      id: "worker-a",
      capabilities: { distributedExecutor: { protocol: "gdlp-worker-tunnel/2" } },
    };
    const hub = new WorkerHub({
      getWorker: vi.fn(() => worker),
      listWorkers: vi.fn(() => [worker]),
    } as unknown as MeshStore, { activationCheckpointTransfers: transfers });
    const sent: Array<{ type: string; payload: Record<string, unknown> }> = [];
    vi.spyOn(hub, "send").mockImplementation((_workerId, type, value) => {
      sent.push({ type, payload: value as Record<string, unknown> });
      return true;
    });
    const timer = setTimeout(() => undefined, 60_000);
    timer.unref();
    const state = {
      socket: { OPEN: 1, readyState: 1, close: vi.fn() },
      workerId: "worker-a",
      ready: true,
      helloTimer: timer,
      pending: false,
      messageWindowStartedAt: Date.now(),
      messagesInWindow: 0,
      sessionId: "authenticated-session-a",
    };
    const internal = hub as unknown as {
      connections: Map<string, typeof state>;
      allConnections: Set<typeof state>;
      handleRawMessage(connection: typeof state, raw: string): Promise<void>;
    };
    internal.connections.set("worker-a", state);
    internal.allConnections.add(state);
    const transferId = hub.requestActivationCheckpoint(
      "worker-a", 12, expected, payload.byteLength, now + 10_000, now,
    );
    expect(sent[0]).toMatchObject({
      type: "runtime.checkpoint.request",
      payload: { transferId, stageRequestId: 12, maximumBytes: payload.byteLength },
    });
    const chunks = activationCheckpointChunks(transferId, checkpoint, payload);
    for (const [type, value] of [
      ["runtime.checkpoint.begin", { transferId, checkpoint, chunkCount: chunks.length }],
      ...chunks.map((chunk) => ["runtime.checkpoint.chunk", chunk]),
      ["runtime.checkpoint.commit", { transferId, checkpointId: checkpoint.checkpointId }],
    ] as const) {
      await internal.handleRawMessage(state, JSON.stringify({
        v: 1, workerId: "worker-a", type, payload: value,
      }));
    }
    expect(sent.at(-1)).toEqual({
      type: "runtime.checkpoint.committed",
      payload: { transferId, checkpointId: checkpoint.checkpointId },
    });
    expect(store.get(checkpoint.checkpointId, expected, now + 100)?.payload.equals(payload)).toBe(true);
    const restored = vi.fn();
    hub.on("activationCheckpointRestored", restored);
    const restoration = hub.restoreActivationCheckpoint(
      "worker-a", "warm-launch", 12, checkpoint.checkpointId, expected,
      payload.byteLength, now + 10_000, now + 100,
    );
    await vi.waitFor(() => expect(sent.at(-1)?.type).toBe("runtime.checkpoint.restore.commit"));
    const restoreTransferId = sent.at(-1)?.payload.transferId as string;
    expect(sent.slice(-3).map(({ type }) => type)).toEqual([
      "runtime.checkpoint.restore.begin",
      "runtime.checkpoint.restore.chunk",
      "runtime.checkpoint.restore.commit",
    ]);
    await internal.handleRawMessage(state, JSON.stringify({
      v: 1,
      workerId: "worker-a",
      type: "runtime.checkpoint.restored",
      payload: { transferId: restoreTransferId, checkpointId: checkpoint.checkpointId },
    }));
    await expect(restoration).resolves.toEqual({
      transferId: restoreTransferId,
      checkpointId: checkpoint.checkpointId,
    });
    expect(restored).toHaveBeenCalledWith("worker-a", restoreTransferId, checkpoint.checkpointId);

    const failedRestoration = hub.restoreActivationCheckpoint(
      "worker-a", "warm-launch", 12, checkpoint.checkpointId, expected,
      payload.byteLength, now + 10_000, now + 200,
    );
    await vi.waitFor(() => expect(sent.at(-1)?.type).toBe("runtime.checkpoint.restore.commit"));
    const failedTransferId = sent.at(-1)?.payload.transferId as string;
    await internal.handleRawMessage(state, JSON.stringify({
      v: 1,
      workerId: "worker-a",
      type: "runtime.checkpoint.restore.failed",
      payload: {
        transferId: failedTransferId,
        checkpointId: checkpoint.checkpointId,
        code: "checkpoint_restore_failed",
      },
    }));
    await expect(failedRestoration).rejects.toThrow("checkpoint_restore_failed");
    hub.close();
  });
});
