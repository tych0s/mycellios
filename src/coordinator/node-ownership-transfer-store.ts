import { createHash, randomBytes, randomUUID, timingSafeEqual } from "node:crypto";
import type { MeshDatabase } from "../storage/database.js";

export class NodeOwnershipTransferStore {
  constructor(private readonly database: MeshDatabase, private readonly now: () => number = Date.now) {}

  create(input: { nodeId: string; sourceAccountId: string; targetAccountId: string; expectedGeneration: number; expiresInSeconds: number }) {
    if (input.sourceAccountId === input.targetAccountId) throw new Error("node_transfer_target_is_current_owner");
    const ownership = this.database.getNodeOwnership("device", input.nodeId);
    if (!ownership || ownership.status !== "active" || ownership.accountId !== input.sourceAccountId) throw new Error("node_transfer_source_mismatch");
    if (ownership.generation !== input.expectedGeneration) throw new Error("node_transfer_generation_conflict");
    const id = randomUUID();
    const token = randomBytes(32).toString("base64url");
    const createdAt = this.now();
    const expiresAt = createdAt + input.expiresInSeconds * 1_000;
    this.database.transaction(() => {
      this.database.raw.prepare(
        `INSERT INTO node_ownership_transfers(id, identity_id, source_account_id, target_account_id,
          token_hash, expected_generation, expires_at, accepted_at, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)`,
      ).run(id, input.nodeId, input.sourceAccountId, input.targetAccountId, hash(token), input.expectedGeneration, expiresAt, createdAt);
      this.database.appendNodeIdentityEvent({ identityKind: "device", identityId: input.nodeId,
        generation: ownership.generation, actorKind: "account", actorId: input.sourceAccountId,
        eventType: "ownership.transfer_requested", details: { transferId: id, targetAccountId: input.targetAccountId, expiresAt } });
    });
    return { transferId: id, transferToken: token, nodeId: input.nodeId, targetAccountId: input.targetAccountId, expiresAt: new Date(expiresAt).toISOString() };
  }

  accept(input: { transferId: string; transferToken: string; targetAccountId: string; nodeId: string }) {
    return this.database.transaction(() => {
      const row = this.database.raw.prepare(
        `SELECT identity_id, source_account_id, target_account_id, token_hash, expected_generation, expires_at, accepted_at
         FROM node_ownership_transfers WHERE id = ?`,
      ).get(input.transferId) as { identity_id: string; source_account_id: string; target_account_id: string; token_hash: string; expected_generation: number; expires_at: number; accepted_at: number | null } | undefined;
      if (!row) throw new Error("node_transfer_unknown");
      if (row.accepted_at !== null) throw new Error("node_transfer_already_accepted");
      if (row.expires_at < this.now()) throw new Error("node_transfer_expired");
      if (row.identity_id !== input.nodeId || row.target_account_id !== input.targetAccountId) throw new Error("node_transfer_target_mismatch");
      if (!equalHash(row.token_hash, hash(input.transferToken))) throw new Error("node_transfer_token_mismatch");
      const ownership = this.database.getNodeOwnership("device", input.nodeId);
      if (!ownership || ownership.status !== "active" || ownership.accountId !== row.source_account_id
        || ownership.generation !== row.expected_generation) throw new Error("node_transfer_generation_conflict");
      const acceptedAt = this.now();
      const changed = this.database.raw.prepare(
        `UPDATE node_ownership SET account_id = ?, generation = generation + 1, updated_at = ?
         WHERE identity_kind = 'device' AND identity_id = ? AND account_id = ? AND generation = ? AND status = 'active'`,
      ).run(input.targetAccountId, acceptedAt, input.nodeId, row.source_account_id, row.expected_generation);
      if (Number(changed.changes) !== 1) throw new Error("node_transfer_generation_conflict");
      this.database.raw.prepare(`UPDATE node_ownership_transfers SET accepted_at = ? WHERE id = ? AND accepted_at IS NULL`).run(acceptedAt, input.transferId);
      const next = this.database.getNodeOwnership("device", input.nodeId)!;
      this.database.appendNodeIdentityEvent({ identityKind: "device", identityId: input.nodeId,
        generation: next.generation, actorKind: "account", actorId: input.targetAccountId,
        eventType: "ownership.transferred", details: { transferId: input.transferId, previousAccountId: row.source_account_id } });
      return { state: "transferred" as const, nodeId: input.nodeId, generation: next.generation };
    });
  }
}

function hash(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function equalHash(left: string, right: string): boolean {
  const leftBytes = Buffer.from(left, "hex");
  const rightBytes = Buffer.from(right, "hex");
  return leftBytes.length === rightBytes.length && timingSafeEqual(leftBytes, rightBytes);
}
