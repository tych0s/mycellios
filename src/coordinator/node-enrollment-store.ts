import { createHash, randomBytes, randomUUID } from "node:crypto";
import { z } from "zod";
import type { MeshDatabase } from "../storage/database.js";
import { nodeControlActorSchema } from "../contracts/node-control.js";

const sha256 = (value: string) =>
  `sha256:${createHash("sha256").update(value, "utf8").digest("hex")}` as const;

const rowSchema = z.object({
  id: z.string().uuid(),
  token_hash: z.string().regex(/^sha256:[a-f0-9]{64}$/),
  nonce_hash: z.string().regex(/^sha256:[a-f0-9]{64}$/),
  account_id: z.string().min(1),
  expires_at: z.number().int(),
  confirmed_at: z.number().int().nullable(),
  consumed_at: z.number().int().nullable(),
}).passthrough();

export interface IssuedNodeEnrollment {
  enrollmentId: string;
  enrollmentToken: string;
  nonce: string;
  expiresAt: string;
}

export type NodeEnrollmentConsumeResult =
  | { state: "consumed"; enrollmentId: string; accountId: string }
  | { state: "unknown" | "expired" | "unconfirmed" | "already-consumed" | "nonce-mismatch" | "ownership-conflict" | "credential-conflict" | "recovery-required" | "revoked" };

export class NodeEnrollmentStore {
  constructor(
    private readonly database: MeshDatabase,
    private readonly now: () => number = Date.now,
  ) {}

  issue(input: {
    accountId: string;
    actor: z.infer<typeof nodeControlActorSchema>;
    expiresInSeconds: number;
  }): IssuedNodeEnrollment {
    const actor = nodeControlActorSchema.parse(input.actor);
    if (!actor.scopes.includes("node:identity")) throw new Error("node_enrollment_scope_denied");
    if (actor.kind === "node") throw new Error("node_enrollment_actor_denied");
    if (!Number.isInteger(input.expiresInSeconds) || input.expiresInSeconds < 60 || input.expiresInSeconds > 900) {
      throw new Error("node_enrollment_ttl_invalid");
    }
    const id = randomUUID();
    const token = randomBytes(32).toString("base64url");
    const nonce = randomBytes(24).toString("base64url");
    const createdAt = this.now();
    const expiresAt = createdAt + input.expiresInSeconds * 1_000;
    this.database.transaction(() => {
      this.database.raw.prepare(
        `INSERT INTO node_enrollments(
           id, token_hash, nonce_hash, account_id, created_by_kind, created_by_id,
           expires_at, created_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(id, sha256(token), sha256(nonce), input.accountId, actor.kind, actor.id, expiresAt, createdAt);
      this.appendEvent(id, actor.kind, actor.id, "issued", { expiresAt });
    });
    return { enrollmentId: id, enrollmentToken: token, nonce, expiresAt: new Date(expiresAt).toISOString() };
  }

  confirm(input: { enrollmentId: string; accountId: string; actorId: string }): void {
    this.database.transaction(() => {
      const row = this.readById(input.enrollmentId);
      if (!row) throw new Error("node_enrollment_unknown");
      if (row.account_id !== input.accountId) throw new Error("node_enrollment_account_mismatch");
      if (row.expires_at <= this.now()) throw new Error("node_enrollment_expired");
      if (row.consumed_at !== null) throw new Error("node_enrollment_already_consumed");
      if (row.confirmed_at === null) {
        this.database.raw.prepare(
          "UPDATE node_enrollments SET confirmed_at = ?, confirmed_by = ? WHERE id = ? AND confirmed_at IS NULL",
        ).run(this.now(), input.actorId, input.enrollmentId);
        this.appendEvent(input.enrollmentId, "account", input.actorId, "confirmed", {});
      }
    });
  }

  consume(input: {
    enrollmentToken: string;
    nonce: string;
    identityKind: "device" | "cell";
    identityId: string;
    publicKeyFingerprint: `sha256:${string}`;
  }): NodeEnrollmentConsumeResult {
    return this.database.transaction(() => {
      const row = this.readByTokenHash(sha256(input.enrollmentToken));
      if (!row) return { state: "unknown" };
      if (row.consumed_at !== null) return { state: "already-consumed" };
      if (row.expires_at <= this.now()) return { state: "expired" };
      if (row.confirmed_at === null) return { state: "unconfirmed" };
      if (row.nonce_hash !== sha256(input.nonce)) return { state: "nonce-mismatch" };
      const ownership = this.database.getNodeOwnership(input.identityKind, input.identityId);
      if (ownership?.accountId !== undefined && ownership.accountId !== row.account_id) {
        return { state: "ownership-conflict" };
      }
      if (ownership?.status === "revoked") return { state: "revoked" };
      if (ownership && ownership.credentialFingerprint !== input.publicKeyFingerprint) {
        return { state: "recovery-required" };
      }
      const credentialOwner = this.database.getNodeOwnershipByCredentialFingerprint(input.publicKeyFingerprint);
      if (
        credentialOwner
        && (credentialOwner.identityKind !== input.identityKind || credentialOwner.identityId !== input.identityId)
      ) {
        return { state: "credential-conflict" };
      }
      const changed = this.database.raw.prepare(
        `UPDATE node_enrollments
         SET consumed_at = ?, identity_kind = ?, identity_id = ?, public_key_fingerprint = ?
         WHERE id = ? AND consumed_at IS NULL`,
      ).run(this.now(), input.identityKind, input.identityId, input.publicKeyFingerprint, row.id);
      if (Number(changed.changes) !== 1) return { state: "already-consumed" };
      if (!ownership) {
        const now = this.now();
        this.database.raw.prepare(
          `INSERT INTO node_ownership(
             identity_kind, identity_id, account_id, credential_fingerprint,
             status, generation, created_at, updated_at
           ) VALUES (?, ?, ?, ?, 'active', 1, ?, ?)`,
        ).run(input.identityKind, input.identityId, row.account_id, input.publicKeyFingerprint, now, now);
        this.database.appendNodeIdentityEvent({ identityKind: input.identityKind, identityId: input.identityId,
          generation: 1, actorKind: "account", actorId: row.account_id, eventType: "identity.enrolled",
          details: { credentialFingerprint: input.publicKeyFingerprint, enrollmentId: row.id } });
      }
      this.appendEvent(row.id, input.identityKind, input.identityId, "consumed", {
        publicKeyFingerprint: input.publicKeyFingerprint,
      });
      return { state: "consumed", enrollmentId: row.id, accountId: row.account_id };
    });
  }

  inspectStoredSecrets(enrollmentId: string): { tokenHash: string; nonceHash: string } | null {
    const row = this.readById(enrollmentId);
    return row ? { tokenHash: row.token_hash, nonceHash: row.nonce_hash } : null;
  }

  audit(enrollmentId: string): Array<{ sequence: number; eventType: string; actorKind: string; actorId: string }> {
    return (this.database.raw.prepare(
      `SELECT sequence, event_type, actor_kind, actor_id
       FROM node_enrollment_events WHERE enrollment_id = ? ORDER BY sequence`,
    ).all(enrollmentId) as Array<Record<string, unknown>>).map((row) => ({
      sequence: Number(row.sequence),
      eventType: String(row.event_type),
      actorKind: String(row.actor_kind),
      actorId: String(row.actor_id),
    }));
  }

  private readById(id: string): z.infer<typeof rowSchema> | null {
    const row = this.database.raw.prepare("SELECT * FROM node_enrollments WHERE id = ?").get(id);
    return row ? rowSchema.parse(row) : null;
  }

  private readByTokenHash(tokenHash: string): z.infer<typeof rowSchema> | null {
    const row = this.database.raw.prepare("SELECT * FROM node_enrollments WHERE token_hash = ?").get(tokenHash);
    return row ? rowSchema.parse(row) : null;
  }

  private appendEvent(
    enrollmentId: string,
    actorKind: string,
    actorId: string,
    eventType: string,
    details: Record<string, unknown>,
  ): void {
    this.database.raw.prepare(
      `INSERT INTO node_enrollment_events(
         enrollment_id, actor_kind, actor_id, event_type, details_json, occurred_at
       ) VALUES (?, ?, ?, ?, ?, ?)`,
    ).run(enrollmentId, actorKind, actorId, eventType, JSON.stringify(details), this.now());
  }
}
