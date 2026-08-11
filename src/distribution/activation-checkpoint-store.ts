import type { KeyLike } from "node:crypto";
import {
  verifyActivationCheckpoint,
  type ActivationCheckpoint,
  type ActivationCheckpointCompatibility,
} from "../contracts/activation-checkpoint.js";

export interface ActivationCheckpointStoreLimits {
  maxEntries: number;
  maxEntryBytes: number;
  maxTotalBytes: number;
}

interface StoredCheckpoint {
  checkpoint: ActivationCheckpoint;
  payload: Buffer;
  lastAccessedAt: number;
}

export class ActivationCheckpointStore {
  private readonly entries = new Map<string, StoredCheckpoint>();
  private totalBytes = 0;

  constructor(
    private readonly pinnedKeys: ReadonlyMap<string, KeyLike> | ((keyId: string) => KeyLike | undefined),
    private readonly limits: ActivationCheckpointStoreLimits,
  ) {
    if (
      !Number.isSafeInteger(limits.maxEntries) || limits.maxEntries < 1
      || !Number.isSafeInteger(limits.maxEntryBytes) || limits.maxEntryBytes < 1
      || !Number.isSafeInteger(limits.maxTotalBytes) || limits.maxTotalBytes < limits.maxEntryBytes
    ) throw new Error("activation_checkpoint_store_limits_are_invalid");
  }

  private keysFor(keyId: string): ReadonlyMap<string, KeyLike> {
    if (typeof this.pinnedKeys !== "function") return this.pinnedKeys;
    const key = this.pinnedKeys(keyId);
    return key ? new Map([[keyId, key]]) : new Map();
  }

  put(
    value: unknown,
    payload: Uint8Array,
    expected: ActivationCheckpointCompatibility,
    now = Date.now(),
  ): ActivationCheckpoint {
    if (payload.byteLength > this.limits.maxEntryBytes) {
      throw new Error("activation_checkpoint_store_entry_limit_exceeded");
    }
    const parsedKeyId = typeof value === "object" && value !== null
      && "keyId" in value && typeof value.keyId === "string" ? value.keyId : "";
    const checkpoint = verifyActivationCheckpoint(
      value, payload, this.keysFor(parsedKeyId), expected, now,
    );
    const existing = this.entries.get(checkpoint.checkpointId);
    if (existing) {
      if (!existing.payload.equals(payload)) throw new Error("activation_checkpoint_store_identity_conflict");
      existing.lastAccessedAt = now;
      return structuredClone(checkpoint);
    }
    while (
      this.entries.size >= this.limits.maxEntries
      || this.totalBytes + payload.byteLength > this.limits.maxTotalBytes
    ) {
      const victim = [...this.entries.values()].sort((a, b) =>
        a.lastAccessedAt - b.lastAccessedAt || a.checkpoint.checkpointId.localeCompare(b.checkpoint.checkpointId)
      )[0];
      if (!victim) throw new Error("activation_checkpoint_store_capacity_is_unavailable");
      this.delete(victim.checkpoint.checkpointId);
    }
    this.entries.set(checkpoint.checkpointId, {
      checkpoint,
      payload: Buffer.from(payload),
      lastAccessedAt: now,
    });
    this.totalBytes += payload.byteLength;
    return structuredClone(checkpoint);
  }

  get(
    checkpointId: string,
    expected: ActivationCheckpointCompatibility,
    now = Date.now(),
  ): { checkpoint: ActivationCheckpoint; payload: Buffer } | null {
    const stored = this.entries.get(checkpointId);
    if (!stored) return null;
    try {
      const checkpoint = verifyActivationCheckpoint(
        stored.checkpoint,
        stored.payload,
        this.keysFor(stored.checkpoint.keyId),
        expected,
        now,
      );
      stored.lastAccessedAt = now;
      return { checkpoint: structuredClone(checkpoint), payload: Buffer.from(stored.payload) };
    } catch (error) {
      this.delete(checkpointId);
      throw error;
    }
  }

  delete(checkpointId: string): boolean {
    const stored = this.entries.get(checkpointId);
    if (!stored || !this.entries.delete(checkpointId)) return false;
    this.totalBytes -= stored.payload.byteLength;
    stored.payload.fill(0);
    return true;
  }

  snapshot(): { entries: number; totalBytes: number } {
    return { entries: this.entries.size, totalBytes: this.totalBytes };
  }
}
