import { randomUUID } from "node:crypto";
import {
  type ActivationCheckpoint,
  type ActivationCheckpointCompatibility,
} from "../contracts/activation-checkpoint.js";
import {
  ACTIVATION_CHECKPOINT_CHUNK_BYTES,
  activationCheckpointBeginSchema,
  activationCheckpointChunkSchema,
  activationCheckpointCommitSchema,
} from "../contracts/activation-checkpoint-transfer.js";
import { ActivationCheckpointStore } from "../distribution/activation-checkpoint-store.js";

export {
  ACTIVATION_CHECKPOINT_CHUNK_BYTES,
  MAX_ACTIVATION_CHECKPOINT_CHUNKS,
  activationCheckpointBeginSchema,
  activationCheckpointChunkSchema,
  activationCheckpointCommitSchema,
  activationCheckpointChunks,
} from "../contracts/activation-checkpoint-transfer.js";

interface ExpectedTransfer {
  workerId: string;
  workerSessionId: string;
  expected: ActivationCheckpointCompatibility;
  maximumBytes: number;
  expiresAt: number;
  checkpoint: ActivationCheckpoint | null;
  chunkCount: number;
  chunks: Buffer[];
  receivedBytes: number;
}

interface ExpectedRestore {
  workerId: string;
  workerSessionId: string;
  checkpointId: string;
  expiresAt: number;
}

export class ActivationCheckpointTransferAuthority {
  private readonly transfers = new Map<string, ExpectedTransfer>();
  private readonly restores = new Map<string, ExpectedRestore>();

  constructor(private readonly store: ActivationCheckpointStore) {}

  expect(input: {
    workerId: string;
    workerSessionId: string;
    expected: ActivationCheckpointCompatibility;
    maximumBytes: number;
    expiresAt: number;
  }, now = Date.now()): string {
    if (
      !input.workerId || !input.workerSessionId
      || !Number.isSafeInteger(input.maximumBytes) || input.maximumBytes < 1
      || !Number.isSafeInteger(input.expiresAt) || input.expiresAt <= now
    ) throw new Error("activation_checkpoint_transfer_expectation_is_invalid");
    const transferId = randomUUID();
    this.transfers.set(transferId, {
      ...input,
      expected: structuredClone(input.expected),
      checkpoint: null,
      chunkCount: 0,
      chunks: [],
      receivedBytes: 0,
    });
    return transferId;
  }

  begin(
    workerId: string,
    workerSessionId: string,
    value: unknown,
    now = Date.now(),
  ): void {
    const begin = activationCheckpointBeginSchema.parse(value);
    const transfer = this.requireTransfer(begin.transferId, workerId, workerSessionId, now);
    if (transfer.checkpoint) throw new Error("activation_checkpoint_transfer_already_started");
    if (
      begin.checkpoint.bytes > transfer.maximumBytes
      || begin.chunkCount !== Math.ceil(begin.checkpoint.bytes / ACTIVATION_CHECKPOINT_CHUNK_BYTES)
    ) throw new Error("activation_checkpoint_transfer_shape_is_invalid");
    transfer.checkpoint = begin.checkpoint;
    transfer.chunkCount = begin.chunkCount;
  }

  chunk(workerId: string, workerSessionId: string, value: unknown, now = Date.now()): void {
    const chunk = activationCheckpointChunkSchema.parse(value);
    const transfer = this.requireTransfer(chunk.transferId, workerId, workerSessionId, now);
    const checkpoint = transfer.checkpoint;
    if (!checkpoint || chunk.checkpointId !== checkpoint.checkpointId) {
      throw new Error("activation_checkpoint_transfer_identity_is_invalid");
    }
    if (chunk.index !== transfer.chunks.length || chunk.index >= transfer.chunkCount) {
      throw new Error("activation_checkpoint_transfer_chunk_order_is_invalid");
    }
    const bytes = Buffer.from(chunk.data, "base64");
    const expectedLength = chunk.index === transfer.chunkCount - 1
      ? checkpoint.bytes - ACTIVATION_CHECKPOINT_CHUNK_BYTES * chunk.index
      : ACTIVATION_CHECKPOINT_CHUNK_BYTES;
    if (bytes.byteLength !== expectedLength) {
      throw new Error("activation_checkpoint_transfer_chunk_size_is_invalid");
    }
    transfer.receivedBytes += bytes.byteLength;
    if (transfer.receivedBytes > transfer.maximumBytes) {
      throw new Error("activation_checkpoint_transfer_limit_exceeded");
    }
    transfer.chunks.push(bytes);
  }

  commit(
    workerId: string,
    workerSessionId: string,
    value: unknown,
    now = Date.now(),
  ): ActivationCheckpoint {
    const commit = activationCheckpointCommitSchema.parse(value);
    const transfer = this.requireTransfer(commit.transferId, workerId, workerSessionId, now);
    const checkpoint = transfer.checkpoint;
    if (
      !checkpoint || commit.checkpointId !== checkpoint.checkpointId
      || transfer.chunks.length !== transfer.chunkCount
      || transfer.receivedBytes !== checkpoint.bytes
    ) throw new Error("activation_checkpoint_transfer_is_incomplete");
    try {
      return this.store.put(
        checkpoint,
        Buffer.concat(transfer.chunks, transfer.receivedBytes),
        transfer.expected,
        now,
      );
    } finally {
      this.abort(commit.transferId);
    }
  }

  abort(transferId: string): boolean {
    const transfer = this.transfers.get(transferId);
    if (!transfer || !this.transfers.delete(transferId)) return false;
    for (const chunk of transfer.chunks) chunk.fill(0);
    return true;
  }

  abortSession(workerId: string, workerSessionId: string): number {
    let count = 0;
    for (const [transferId, transfer] of this.transfers) {
      if (transfer.workerId === workerId && transfer.workerSessionId === workerSessionId) {
        if (this.abort(transferId)) count += 1;
      }
    }
    for (const [transferId, restore] of this.restores) {
      if (restore.workerId === workerId && restore.workerSessionId === workerSessionId) {
        if (this.restores.delete(transferId)) count += 1;
      }
    }
    return count;
  }

  prepareRestore(input: {
    workerId: string;
    workerSessionId: string;
    checkpointId: string;
    expected: ActivationCheckpointCompatibility;
    expiresAt: number;
  }, now = Date.now()): {
    transferId: string;
    checkpoint: ActivationCheckpoint;
    payload: Buffer;
  } {
    if (!input.workerId || !input.workerSessionId || input.expiresAt <= now) {
      throw new Error("activation_checkpoint_restore_expectation_is_invalid");
    }
    const stored = this.store.get(input.checkpointId, input.expected, now);
    if (!stored) throw new Error("activation_checkpoint_restore_is_unavailable");
    const transferId = randomUUID();
    this.restores.set(transferId, {
      workerId: input.workerId,
      workerSessionId: input.workerSessionId,
      checkpointId: stored.checkpoint.checkpointId,
      expiresAt: input.expiresAt,
    });
    return { transferId, ...stored };
  }

  completeRestore(
    workerId: string,
    workerSessionId: string,
    transferId: string,
    checkpointId: string,
    now = Date.now(),
  ): string {
    const restore = this.requireRestore(transferId, workerId, workerSessionId, now);
    if (restore.checkpointId !== checkpointId) {
      throw new Error("activation_checkpoint_restore_identity_is_invalid");
    }
    this.restores.delete(transferId);
    return checkpointId;
  }

  failRestore(
    workerId: string,
    workerSessionId: string,
    transferId: string,
    checkpointId: string,
    now = Date.now(),
  ): void {
    const restore = this.requireRestore(transferId, workerId, workerSessionId, now);
    if (restore.checkpointId !== checkpointId) {
      throw new Error("activation_checkpoint_restore_identity_is_invalid");
    }
    this.restores.delete(transferId);
  }

  abortRestore(transferId: string): boolean {
    return this.restores.delete(transferId);
  }

  private requireRestore(
    transferId: string,
    workerId: string,
    workerSessionId: string,
    now: number,
  ): ExpectedRestore {
    const restore = this.restores.get(transferId);
    if (!restore) throw new Error("activation_checkpoint_restore_is_unknown");
    if (restore.expiresAt <= now) {
      this.restores.delete(transferId);
      throw new Error("activation_checkpoint_restore_is_expired");
    }
    if (restore.workerId !== workerId || restore.workerSessionId !== workerSessionId) {
      throw new Error("activation_checkpoint_restore_session_is_invalid");
    }
    return restore;
  }

  private requireTransfer(
    transferId: string,
    workerId: string,
    workerSessionId: string,
    now: number,
  ): ExpectedTransfer {
    const transfer = this.transfers.get(transferId);
    if (!transfer) throw new Error("activation_checkpoint_transfer_is_unknown");
    if (transfer.expiresAt <= now) {
      this.abort(transferId);
      throw new Error("activation_checkpoint_transfer_is_expired");
    }
    if (transfer.workerId !== workerId || transfer.workerSessionId !== workerSessionId) {
      throw new Error("activation_checkpoint_transfer_session_is_invalid");
    }
    return transfer;
  }
}
