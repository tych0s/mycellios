import { z } from "zod";
import {
  activationCheckpointCompatibilitySchema,
  activationCheckpointIdSchema,
} from "../contracts/activation-checkpoint-transfer.js";
import { canonicalEvidenceJson, sha256CanonicalEvidence } from "../core/json.js";
import type { MeshDatabase } from "../storage/database.js";

const recoveryInputSchema = z.object({
  modelId: z.string().min(1).max(256),
  reservationId: z.string().min(1).max(256).nullable(),
  recoveryGeneration: z.number().int().positive().safe(),
  sourceCheckpointId: activationCheckpointIdSchema,
  targetWorkerId: z.string().min(1).max(256),
  targetLaunchRequestId: z.string().min(1).max(256),
  targetStageRequestId: z.number().int().nonnegative().safe(),
  expected: activationCheckpointCompatibilitySchema,
  maximumBytes: z.number().int().positive().max(512 * 1024 * 1024),
  expiresAt: z.number().int().positive().safe(),
}).strict();

export type ActivationCheckpointRecoveryInput = z.infer<typeof recoveryInputSchema>;
export type ActivationCheckpointRecoveryStatus =
  | "prepared" | "restoring" | "restored" | "promoting" | "promoted" | "failed";

export interface ActivationCheckpointRecoveryRecord extends ActivationCheckpointRecoveryInput {
  id: string;
  operationKey: string;
  status: ActivationCheckpointRecoveryStatus;
  transferId: string | null;
  error: string | null;
  createdAt: number;
  updatedAt: number;
  restoredAt: number | null;
  promotedAt: number | null;
}

interface RecoveryRow {
  id: string;
  operation_key: string;
  model_id: string;
  reservation_id: string | null;
  recovery_generation: number;
  source_checkpoint_id: string;
  target_worker_id: string;
  target_launch_request_id: string;
  target_stage_request_id: number;
  compatibility_json: string;
  maximum_bytes: number;
  expires_at: number;
  status: ActivationCheckpointRecoveryStatus;
  transfer_id: string | null;
  error: string | null;
  created_at: number;
  updated_at: number;
  restored_at: number | null;
  promoted_at: number | null;
}

export interface ActivationCheckpointRestorer {
  restoreActivationCheckpoint(
    workerId: string,
    targetLaunchRequestId: string,
    targetStageRequestId: number,
    checkpointId: string,
    expected: ActivationCheckpointRecoveryInput["expected"],
    maximumBytes: number,
    expiresAt: number,
    now?: number,
  ): Promise<{ transferId: string; checkpointId: string }>;
}

/**
 * Durable fail-closed gate for activation-KV recovery promotion.
 *
 * The promotion callback is invoked only after the authenticated worker ACK
 * has resolved. Interrupted `restoring`/`promoting` rows become terminally
 * failed on initialize: an ambiguous physical mutation is never replayed.
 */
export class ActivationCheckpointRecoveryAuthority {
  constructor(
    private readonly database: MeshDatabase,
    private readonly restorer: ActivationCheckpointRestorer,
  ) {}

  initialize(now = Date.now()): number {
    return Number(this.database.raw.prepare(
      `UPDATE activation_checkpoint_recoveries
       SET status = 'failed', error = 'coordinator_restarted_during_recovery', updated_at = ?
       WHERE status IN ('restoring', 'promoting')`,
    ).run(now).changes);
  }

  prepare(value: unknown, now = Date.now()): ActivationCheckpointRecoveryRecord {
    const input = recoveryInputSchema.parse(value);
    if (input.expiresAt <= now) throw new Error("activation_checkpoint_recovery_is_expired");
    const identity = { schema: "mycellios-activation-checkpoint-recovery/1", ...input };
    const operationKey = sha256CanonicalEvidence(identity);
    const id = `acr_${operationKey.slice("sha256:".length, "sha256:".length + 32)}`;
    const compatibilityJson = canonicalEvidenceJson(input.expected);
    this.database.raw.prepare(
      `INSERT INTO activation_checkpoint_recoveries(
         id, operation_key, model_id, reservation_id, recovery_generation,
         source_checkpoint_id, target_worker_id, target_launch_request_id,
         target_stage_request_id, compatibility_json, maximum_bytes, expires_at,
         status, created_at, updated_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?, ?)
       ON CONFLICT(operation_key) DO NOTHING`,
    ).run(
      id, operationKey, input.modelId, input.reservationId, input.recoveryGeneration,
      input.sourceCheckpointId, input.targetWorkerId, input.targetLaunchRequestId,
      input.targetStageRequestId, compatibilityJson, input.maximumBytes, input.expiresAt,
      now, now,
    );
    return this.require(id);
  }

  get(id: string): ActivationCheckpointRecoveryRecord | null {
    const row = this.database.raw.prepare(
      "SELECT * FROM activation_checkpoint_recoveries WHERE id = ?",
    ).get(id) as RecoveryRow | undefined;
    return row ? mapRecovery(row) : null;
  }

  async restoreThenPromote(
    id: string,
    promote: (record: ActivationCheckpointRecoveryRecord) => Promise<void> | void,
    now = Date.now(),
  ): Promise<ActivationCheckpointRecoveryRecord> {
    const prepared = this.require(id);
    if (prepared.status === "promoted") return prepared;
    if (prepared.status !== "prepared") {
      throw new Error(`activation_checkpoint_recovery_is_not_prepared:${prepared.status}`);
    }
    if (prepared.expiresAt <= now) {
      this.fail(id, "activation_checkpoint_recovery_is_expired", now);
      throw new Error("activation_checkpoint_recovery_is_expired");
    }
    this.transition(id, "prepared", "restoring", now);
    try {
      const restored = await this.restorer.restoreActivationCheckpoint(
        prepared.targetWorkerId,
        prepared.targetLaunchRequestId,
        prepared.targetStageRequestId,
        prepared.sourceCheckpointId,
        prepared.expected,
        prepared.maximumBytes,
        prepared.expiresAt,
        now,
      );
      if (restored.checkpointId !== prepared.sourceCheckpointId) {
        throw new Error("activation_checkpoint_recovery_ack_identity_is_invalid");
      }
      const restoredAt = Date.now();
      const restoredChanges = this.database.raw.prepare(
        `UPDATE activation_checkpoint_recoveries
         SET status = 'restored', transfer_id = ?, restored_at = ?, updated_at = ?
         WHERE id = ? AND status = 'restoring'`,
      ).run(restored.transferId, restoredAt, restoredAt, id).changes;
      if (restoredChanges !== 1n) throw new Error("activation_checkpoint_recovery_restore_fence_failed");
      this.transition(id, "restored", "promoting", restoredAt);
      await promote(this.require(id));
      const promotedAt = Date.now();
      const promotedChanges = this.database.raw.prepare(
        `UPDATE activation_checkpoint_recoveries
         SET status = 'promoted', promoted_at = ?, updated_at = ?
         WHERE id = ? AND status = 'promoting' AND restored_at IS NOT NULL AND transfer_id IS NOT NULL`,
      ).run(promotedAt, promotedAt, id).changes;
      if (promotedChanges !== 1n) throw new Error("activation_checkpoint_recovery_promotion_fence_failed");
      return this.require(id);
    } catch (error) {
      this.fail(id, error instanceof Error ? error.message : String(error), Date.now());
      throw error;
    }
  }

  private transition(
    id: string,
    from: ActivationCheckpointRecoveryStatus,
    to: ActivationCheckpointRecoveryStatus,
    now: number,
  ): void {
    const changes = this.database.raw.prepare(
      `UPDATE activation_checkpoint_recoveries SET status = ?, updated_at = ?
       WHERE id = ? AND status = ?`,
    ).run(to, now, id, from).changes;
    if (changes !== 1n) throw new Error(`activation_checkpoint_recovery_transition_failed:${from}:${to}`);
  }

  private fail(id: string, error: string, now: number): void {
    this.database.raw.prepare(
      `UPDATE activation_checkpoint_recoveries
       SET status = 'failed', error = ?, updated_at = ?
       WHERE id = ? AND status != 'promoted'`,
    ).run(error.slice(0, 2_000), now, id);
  }

  private require(id: string): ActivationCheckpointRecoveryRecord {
    const record = this.get(id);
    if (!record) throw new Error(`activation_checkpoint_recovery_not_found:${id}`);
    return record;
  }
}

function mapRecovery(row: RecoveryRow): ActivationCheckpointRecoveryRecord {
  const expected = activationCheckpointCompatibilitySchema.parse(JSON.parse(row.compatibility_json));
  return {
    id: row.id,
    operationKey: row.operation_key,
    modelId: row.model_id,
    reservationId: row.reservation_id,
    recoveryGeneration: row.recovery_generation,
    sourceCheckpointId: row.source_checkpoint_id,
    targetWorkerId: row.target_worker_id,
    targetLaunchRequestId: row.target_launch_request_id,
    targetStageRequestId: row.target_stage_request_id,
    expected,
    maximumBytes: row.maximum_bytes,
    expiresAt: row.expires_at,
    status: row.status,
    transferId: row.transfer_id,
    error: row.error,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    restoredAt: row.restored_at,
    promotedAt: row.promoted_at,
  };
}
