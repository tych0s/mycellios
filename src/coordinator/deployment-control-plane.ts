import { createHash } from "node:crypto";
import { newId } from "../core/ids.js";
import type { MeshStore, StoredRequestedModel } from "../storage/store.js";

export type DeploymentDesiredState = "active" | "inactive";
export type DeploymentObservedState =
  | "inactive"
  | "waiting_capacity"
  | "preparing"
  | "canary"
  | "active"
  | "degraded"
  | "failed"
  | "stopping";
export type DeploymentOperationKind = "activate" | "deactivate" | "repair" | "replan";
export type DeploymentOperationStatus =
  | "pending"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "interrupted";
export type RouteReservationStatus = "prepared" | "committed" | "released" | "expired" | "failed";

export interface DeploymentState {
  modelId: string;
  desiredState: DeploymentDesiredState;
  observedState: DeploymentObservedState;
  generation: number;
  observedGeneration: number;
  retryCount: number;
  nextRetryAt: number | null;
  lastError: string | null;
  activeOperationId: string | null;
  controllerOwner: string | null;
  controllerLeaseUntil: number | null;
  createdAt: number;
  updatedAt: number;
}

export interface DeploymentOperation {
  id: string;
  modelId: string;
  generation: number;
  kind: DeploymentOperationKind;
  status: DeploymentOperationStatus;
  attempt: number;
  idempotencyKey: string;
  errorCode: string | null;
  errorMessage: string | null;
  metadata: Record<string, unknown>;
  startedAt: number;
  updatedAt: number;
  finishedAt: number | null;
}

export interface RouteStageReservation {
  nodeId: string;
  stageIndex: number;
  layerStart: number;
  layerEnd: number;
  memoryMiB: number;
  capacityMiB: number;
}

export interface RouteReservation {
  id: string;
  modelId: string;
  operationId: string;
  generation: number;
  status: RouteReservationStatus;
  routeDigest: string;
  stages: RouteStageReservation[];
  canary: Record<string, unknown> | null;
  expiresAt: number;
  committedAt: number | null;
  releasedAt: number | null;
  error: string | null;
  createdAt: number;
  updatedAt: number;
}

export interface DeploymentTimeline {
  state: DeploymentState;
  operations: DeploymentOperation[];
  reservations: RouteReservation[];
}

interface DeploymentStateRow {
  model_id: string;
  desired_state: DeploymentDesiredState;
  observed_state: DeploymentObservedState;
  generation: number;
  observed_generation: number;
  retry_count: number;
  next_retry_at: number | null;
  last_error: string | null;
  active_operation_id: string | null;
  controller_owner: string | null;
  controller_lease_until: number | null;
  created_at: number;
  updated_at: number;
}

interface DeploymentOperationRow {
  id: string;
  model_id: string;
  generation: number;
  kind: DeploymentOperationKind;
  status: DeploymentOperationStatus;
  attempt: number;
  idempotency_key: string;
  error_code: string | null;
  error_message: string | null;
  metadata_json: string;
  started_at: number;
  updated_at: number;
  finished_at: number | null;
}

interface RouteReservationRow {
  id: string;
  model_id: string;
  operation_id: string;
  generation: number;
  status: RouteReservationStatus;
  route_digest: string;
  stages_json: string;
  canary_json: string | null;
  expires_at: number;
  committed_at: number | null;
  released_at: number | null;
  error: string | null;
  created_at: number;
  updated_at: number;
}

/**
 * Durable desired/observed controller for model deployments.
 *
 * The coordinator may crash at any instruction boundary. Every mutating
 * operation therefore has an idempotency key, a controller lease and an
 * append-only operation row. Route capacity is prepared transactionally and
 * only becomes visible after the real inference canary commits it.
 */
export class DeploymentControlPlane {
  constructor(
    private readonly store: MeshStore,
    readonly ownerId = `coordinator-${process.pid}-${newId("ctl")}`,
  ) {}

  initialize(now = Date.now()): {
    models: number;
    interruptedOperations: number;
    expiredReservations: number;
  } {
    return this.store.database.transaction(() => {
      const models = this.store.listRequestedModels();
      for (const model of models) this.ensureModel(model, now);
      const interruptedOperations = Number(this.store.database.raw.prepare(
        `UPDATE deployment_operations
         SET status = 'interrupted', error_code = 'coordinator_restarted',
             error_message = 'The coordinator restarted while the operation was running.',
             updated_at = ?, finished_at = ?
         WHERE status IN ('pending', 'running')`,
      ).run(now, now).changes);
      if (interruptedOperations > 0) {
        this.store.database.raw.prepare(
          `UPDATE deployment_states
           SET observed_state = CASE
                 WHEN desired_state = 'active' THEN 'degraded'
                 ELSE 'inactive'
               END,
               next_retry_at = CASE WHEN desired_state = 'active' THEN ? ELSE NULL END,
               last_error = CASE
                 WHEN desired_state = 'active' THEN 'coordinator_restarted'
                 ELSE NULL
               END,
               active_operation_id = NULL,
               controller_owner = NULL,
               controller_lease_until = NULL,
               updated_at = ?
           WHERE active_operation_id IS NOT NULL`,
        ).run(now, now);
      }
      const expiredReservations = this.expireReservations(now);
      this.queueAll();
      return { models: models.length, interruptedOperations, expiredReservations };
    });
  }

  ensureModel(
    model: Pick<StoredRequestedModel, "id" | "autoActivate">,
    now = Date.now(),
  ): DeploymentState {
    const desiredState: DeploymentDesiredState = model.autoActivate ? "active" : "inactive";
    this.store.database.raw.prepare(
      `INSERT INTO deployment_states(
         model_id, desired_state, observed_state, generation, observed_generation,
         retry_count, created_at, updated_at
       ) VALUES (?, ?, 'inactive', 1, 0, 0, ?, ?)
       ON CONFLICT(model_id) DO NOTHING`,
    ).run(model.id, desiredState, now, now);
    this.queueState(model.id);
    return this.getState(model.id)!;
  }

  setDesiredState(
    modelId: string,
    desiredState: DeploymentDesiredState,
    now = Date.now(),
  ): DeploymentState {
    return this.store.database.transaction(() => {
      const model = this.store.getRequestedModel(modelId);
      if (!model) throw new Error(`deployment_model_not_found:${modelId}`);
      this.ensureModel(model, now);
      this.store.database.raw.prepare(
        `UPDATE deployment_states
         SET desired_state = ?,
             generation = generation + CASE WHEN desired_state = ? THEN 0 ELSE 1 END,
             retry_count = CASE WHEN desired_state = ? THEN retry_count ELSE 0 END,
             next_retry_at = CASE WHEN desired_state = ? THEN next_retry_at ELSE NULL END,
             last_error = CASE WHEN desired_state = ? THEN last_error ELSE NULL END,
             updated_at = ?
         WHERE model_id = ?`,
      ).run(
        desiredState,
        desiredState,
        desiredState,
        desiredState,
        desiredState,
        now,
        modelId,
      );
      this.queueState(modelId);
      return this.getState(modelId)!;
    });
  }

  rearmAfterRuntimeChange(
    modelId: string,
    reason: string,
    now = Date.now(),
  ): DeploymentState {
    return this.store.database.transaction(() => {
      const model = this.store.getRequestedModel(modelId);
      if (!model) throw new Error(`deployment_model_not_found:${modelId}`);
      const state = this.ensureModel(model, now);
      if (state.activeOperationId !== null) {
        throw new Error(`deployment_operation_is_active:${modelId}`);
      }
      this.store.database.raw.prepare(
        `UPDATE deployment_states
         SET generation = generation + 1,
             observed_state = 'failed',
             retry_count = 0,
             next_retry_at = NULL,
             last_error = ?,
             controller_owner = NULL,
             controller_lease_until = NULL,
             updated_at = ?
         WHERE model_id = ? AND active_operation_id IS NULL`,
      ).run(reason.slice(0, 2_000), now, modelId);
      this.queueState(modelId);
      return this.getState(modelId)!;
    });
  }

  getState(modelId: string): DeploymentState | null {
    const row = this.store.database.raw.prepare(
      "SELECT * FROM deployment_states WHERE model_id = ?",
    ).get(modelId) as DeploymentStateRow | undefined;
    return row ? mapDeploymentState(row) : null;
  }

  listStates(): DeploymentState[] {
    const rows = this.store.database.raw.prepare(
      "SELECT * FROM deployment_states ORDER BY created_at, model_id",
    ).all() as unknown as DeploymentStateRow[];
    return rows.map(mapDeploymentState);
  }

  listDueStates(now = Date.now()): DeploymentState[] {
    const rows = this.store.database.raw.prepare(
      `SELECT * FROM deployment_states
       WHERE (
         desired_state = 'active'
         AND observed_state != 'active'
         AND (next_retry_at IS NULL OR next_retry_at <= ?)
       ) OR (
         desired_state = 'inactive'
         AND observed_state != 'inactive'
       )
       ORDER BY updated_at, model_id`,
    ).all(now) as unknown as DeploymentStateRow[];
    return rows.map(mapDeploymentState);
  }

  claimOperation(
    modelId: string,
    kind: DeploymentOperationKind,
    options: {
      leaseMs?: number;
      metadata?: Record<string, unknown>;
      now?: number;
    } = {},
  ): DeploymentOperation | null {
    return this.store.database.transaction(() => {
      const now = options.now ?? Date.now();
      const model = this.store.getRequestedModel(modelId);
      if (!model) throw new Error(`deployment_model_not_found:${modelId}`);
      let state = this.ensureModel(model, now);
      const active = state.activeOperationId
        ? this.getOperation(state.activeOperationId)
        : null;
      if (
        active?.status === "running"
        && state.controllerOwner === this.ownerId
        && (state.controllerLeaseUntil ?? 0) > now
      ) {
        return active;
      }
      if (
        state.controllerOwner
        && state.controllerOwner !== this.ownerId
        && (state.controllerLeaseUntil ?? 0) > now
      ) {
        return null;
      }
      if (active?.status === "running") {
        this.finishOperationRow(
          active.id,
          "interrupted",
          "controller_lease_expired",
          "The previous controller lease expired before the operation completed.",
          now,
        );
      }
      state = this.getState(modelId)!;
      const attempt = state.retryCount + 1;
      const idempotencyKey = `${modelId}:${state.generation}:${kind}:${attempt}`;
      const existing = this.store.database.raw.prepare(
        "SELECT * FROM deployment_operations WHERE idempotency_key = ?",
      ).get(idempotencyKey) as DeploymentOperationRow | undefined;
      if (existing?.status === "running") return mapDeploymentOperation(existing);
      const operationId = newId("dop");
      this.store.database.raw.prepare(
        `INSERT INTO deployment_operations(
           id, model_id, generation, kind, status, attempt, idempotency_key,
           metadata_json, started_at, updated_at
         ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, ?)`,
      ).run(
        operationId,
        modelId,
        state.generation,
        kind,
        attempt,
        idempotencyKey,
        JSON.stringify(options.metadata ?? {}),
        now,
        now,
      );
      const observedState: DeploymentObservedState =
        kind === "deactivate" ? "stopping" : "preparing";
      this.store.database.raw.prepare(
        `UPDATE deployment_states
         SET observed_state = ?, active_operation_id = ?, controller_owner = ?,
             controller_lease_until = ?, next_retry_at = NULL, updated_at = ?
         WHERE model_id = ?`,
      ).run(
        observedState,
        operationId,
        this.ownerId,
        now + Math.max(1_000, options.leaseMs ?? 60_000),
        now,
        modelId,
      );
      this.queueOperation(operationId);
      this.queueState(modelId);
      return this.getOperation(operationId)!;
    });
  }

  renewOperation(operationId: string, leaseMs = 60_000, now = Date.now()): boolean {
    return this.store.database.transaction(() => {
      const operation = this.getOperation(operationId);
      if (!operation || operation.status !== "running") return false;
      const changed = Number(this.store.database.raw.prepare(
        `UPDATE deployment_states
         SET controller_lease_until = ?, updated_at = ?
         WHERE model_id = ? AND active_operation_id = ? AND controller_owner = ?`,
      ).run(now + Math.max(1_000, leaseMs), now, operation.modelId, operationId, this.ownerId).changes);
      if (changed === 1) this.queueState(operation.modelId);
      return changed === 1;
    });
  }

  markWaitingCapacity(modelId: string, message: string | null = null, now = Date.now()): void {
    this.store.database.transaction(() => {
      this.store.database.raw.prepare(
        `UPDATE deployment_states
         SET observed_state = 'waiting_capacity', last_error = ?,
             active_operation_id = NULL, controller_owner = NULL,
             controller_lease_until = NULL, updated_at = ?
         WHERE model_id = ? AND desired_state = 'active'`,
      ).run(message, now, modelId);
      this.queueState(modelId);
    });
  }

  adoptObservedState(
    modelId: string,
    observedState: "active" | "inactive" | "degraded",
    now = Date.now(),
  ): void {
    this.store.database.transaction(() => {
      const state = this.getState(modelId);
      if (!state) return;
      this.store.database.raw.prepare(
        `UPDATE deployment_states
         SET observed_state = ?,
             observed_generation = CASE
               WHEN ? IN ('active', 'inactive') THEN generation
               ELSE observed_generation
             END,
             next_retry_at = CASE WHEN ? = 'degraded' THEN ? ELSE NULL END,
             last_error = CASE WHEN ? = 'degraded' THEN last_error ELSE NULL END,
             updated_at = ?
         WHERE model_id = ? AND active_operation_id IS NULL`,
      ).run(
        observedState,
        observedState,
        observedState,
        now,
        observedState,
        now,
        modelId,
      );
      this.queueState(modelId);
    });
  }

  activeOperationForModel(modelId: string): DeploymentOperation | null {
    const state = this.getState(modelId);
    return state?.activeOperationId ? this.getOperation(state.activeOperationId) : null;
  }

  markCanary(operationId: string, now = Date.now()): boolean {
    return this.setOperationObservedState(operationId, "canary", now);
  }

  completeOperation(
    operationId: string,
    observedState: "active" | "inactive",
    metadata: Record<string, unknown> = {},
    now = Date.now(),
  ): boolean {
    return this.store.database.transaction(() => {
      const operation = this.getOperation(operationId);
      if (!operation || operation.status !== "running") return false;
      this.store.database.raw.prepare(
        `UPDATE deployment_operations
         SET status = 'succeeded', metadata_json = ?, updated_at = ?, finished_at = ?
         WHERE id = ? AND status = 'running'`,
      ).run(JSON.stringify({ ...operation.metadata, ...metadata }), now, now, operationId);
      this.store.database.raw.prepare(
        `UPDATE deployment_states
         SET observed_state = ?, observed_generation = generation,
             retry_count = 0, next_retry_at = NULL, last_error = NULL,
             active_operation_id = NULL, controller_owner = NULL,
             controller_lease_until = NULL, updated_at = ?
         WHERE model_id = ? AND active_operation_id = ?`,
      ).run(observedState, now, operation.modelId, operationId);
      this.queueOperation(operationId);
      this.queueState(operation.modelId);
      return true;
    });
  }

  failOperation(
    operationId: string,
    errorCode: string,
    errorMessage: string,
    options: { retryAt?: number | null; now?: number } = {},
  ): boolean {
    return this.store.database.transaction(() => {
      const now = options.now ?? Date.now();
      const operation = this.getOperation(operationId);
      if (!operation || operation.status !== "running") return false;
      this.finishOperationRow(operationId, "failed", errorCode, errorMessage, now);
      this.store.database.raw.prepare(
        `UPDATE deployment_states
         SET observed_state = 'failed', retry_count = retry_count + 1,
             next_retry_at = ?, last_error = ?, active_operation_id = NULL,
             controller_owner = NULL, controller_lease_until = NULL, updated_at = ?
         WHERE model_id = ? AND active_operation_id = ?`,
      ).run(options.retryAt ?? null, errorMessage.slice(0, 2_000), now, operation.modelId, operationId);
      this.failPreparedReservations(operationId, errorMessage, now);
      this.queueOperation(operationId);
      this.queueState(operation.modelId);
      return true;
    });
  }

  getOperation(operationId: string): DeploymentOperation | null {
    const row = this.store.database.raw.prepare(
      "SELECT * FROM deployment_operations WHERE id = ?",
    ).get(operationId) as DeploymentOperationRow | undefined;
    return row ? mapDeploymentOperation(row) : null;
  }

  listOperations(modelId: string, limit = 100): DeploymentOperation[] {
    const rows = this.store.database.raw.prepare(
      `SELECT * FROM deployment_operations
       WHERE model_id = ?
       ORDER BY started_at DESC, id DESC
       LIMIT ?`,
    ).all(modelId, limit) as unknown as DeploymentOperationRow[];
    return rows.map(mapDeploymentOperation).toReversed();
  }

  prepareRoute(
    operationId: string,
    stages: readonly RouteStageReservation[],
    ttlMs = 90_000,
    now = Date.now(),
  ): RouteReservation {
    if (stages.length === 0) throw new Error("route_reservation_requires_stages");
    return this.store.database.transaction(() => {
      this.expireReservations(now);
      const operation = this.getOperation(operationId);
      if (!operation || operation.status !== "running") {
        throw new Error(`deployment_operation_not_running:${operationId}`);
      }
      const normalized = [...stages]
        .map((stage) => ({
          ...stage,
          memoryMiB: Math.max(0, Math.ceil(stage.memoryMiB)),
          capacityMiB: Math.max(0, Math.floor(stage.capacityMiB)),
        }))
        .sort((left, right) => left.stageIndex - right.stageIndex || left.nodeId.localeCompare(right.nodeId));
      const requestedByNode = new Map<string, { requested: number; capacity: number }>();
      for (const stage of normalized) {
        if (!stage.nodeId || stage.stageIndex < 0 || stage.layerEnd <= stage.layerStart) {
          throw new Error("invalid_route_stage_reservation");
        }
        const current = requestedByNode.get(stage.nodeId) ?? {
          requested: 0,
          capacity: stage.capacityMiB,
        };
        current.requested += stage.memoryMiB;
        current.capacity = Math.min(current.capacity, stage.capacityMiB);
        requestedByNode.set(stage.nodeId, current);
      }
      for (const [nodeId, request] of requestedByNode) {
        const row = this.store.database.raw.prepare(
          `SELECT COALESCE(SUM(memory_mib), 0) AS leased
           FROM deployment_stage_leases
           WHERE node_id = ? AND status IN ('prepared', 'committed') AND expires_at > ?`,
        ).get(nodeId, now) as { leased: number };
        if (Number(row.leased) + request.requested > request.capacity) {
          throw new Error(
            `route_capacity_conflict:${nodeId}:requested=${request.requested}:`
            + `leased=${Number(row.leased)}:capacity=${request.capacity}`,
          );
        }
      }
      const reservationId = newId("rsv");
      const expiresAt = now + Math.max(1_000, ttlMs);
      const routeDigest = createHash("sha256")
        .update(JSON.stringify(normalized))
        .digest("hex");
      this.store.database.raw.prepare(
        `INSERT INTO route_reservations(
           id, model_id, operation_id, generation, status, route_digest,
           stages_json, expires_at, created_at, updated_at
         ) VALUES (?, ?, ?, ?, 'prepared', ?, ?, ?, ?, ?)`,
      ).run(
        reservationId,
        operation.modelId,
        operationId,
        operation.generation,
        routeDigest,
        JSON.stringify(normalized),
        expiresAt,
        now,
        now,
      );
      const insertLease = this.store.database.raw.prepare(
        `INSERT INTO deployment_stage_leases(
           id, reservation_id, model_id, node_id, stage_index, memory_mib,
           status, expires_at, created_at, updated_at
         ) VALUES (?, ?, ?, ?, ?, ?, 'prepared', ?, ?, ?)`,
      );
      for (const stage of normalized) {
        insertLease.run(
          newId("dls"),
          reservationId,
          operation.modelId,
          stage.nodeId,
          stage.stageIndex,
          stage.memoryMiB,
          expiresAt,
          now,
          now,
        );
      }
      this.queueReservation(reservationId);
      return this.getReservation(reservationId)!;
    });
  }

  commitRoute(
    reservationId: string,
    canary: Record<string, unknown>,
    now = Date.now(),
  ): RouteReservation {
    return this.store.database.transaction(() => {
      const reservation = this.getReservation(reservationId);
      if (!reservation) throw new Error(`route_reservation_not_found:${reservationId}`);
      if (reservation.status === "committed") return reservation;
      if (reservation.status !== "prepared") {
        throw new Error(`route_reservation_not_prepared:${reservationId}:${reservation.status}`);
      }
      if (reservation.expiresAt <= now) {
        this.expireReservations(now);
        throw new Error(`route_reservation_expired:${reservationId}`);
      }
      this.store.database.raw.prepare(
        `UPDATE route_reservations
         SET status = 'committed', canary_json = ?, committed_at = ?, updated_at = ?
         WHERE id = ? AND status = 'prepared'`,
      ).run(JSON.stringify(canary), now, now, reservationId);
      this.store.database.raw.prepare(
        `UPDATE deployment_stage_leases
         SET status = 'committed', updated_at = ?
         WHERE reservation_id = ? AND status = 'prepared'`,
      ).run(now, reservationId);
      this.completeOperation(
        reservation.operationId,
        "active",
        { routeReservationId: reservationId, routeDigest: reservation.routeDigest, canary },
        now,
      );
      this.queueReservation(reservationId);
      return this.getReservation(reservationId)!;
    });
  }

  renewRoute(reservationId: string, ttlMs = 90_000, now = Date.now()): boolean {
    return this.store.database.transaction(() => {
      const expiresAt = now + Math.max(1_000, ttlMs);
      const changed = Number(this.store.database.raw.prepare(
        `UPDATE route_reservations
         SET expires_at = ?, updated_at = ?
         WHERE id = ? AND status IN ('prepared', 'committed') AND expires_at > ?`,
      ).run(expiresAt, now, reservationId, now).changes);
      if (changed === 0) return false;
      this.store.database.raw.prepare(
        `UPDATE deployment_stage_leases
         SET expires_at = ?, updated_at = ?
         WHERE reservation_id = ? AND status IN ('prepared', 'committed')`,
      ).run(expiresAt, now, reservationId);
      this.queueReservation(reservationId);
      return true;
    });
  }

  renewCommittedRoutesForModel(
    modelId: string,
    ttlMs = 90_000,
    now = Date.now(),
  ): number {
    const rows = this.store.database.raw.prepare(
      `SELECT id FROM route_reservations
       WHERE model_id = ? AND status = 'committed' AND expires_at > ?`,
    ).all(modelId, now) as unknown as Array<{ id: string }>;
    let renewed = 0;
    for (const row of rows) {
      if (this.renewRoute(row.id, ttlMs, now)) renewed += 1;
    }
    return renewed;
  }

  releaseRoutesForModel(modelId: string, now = Date.now()): number {
    return this.store.database.transaction(() => {
      const rows = this.store.database.raw.prepare(
        `SELECT id FROM route_reservations
         WHERE model_id = ? AND status IN ('prepared', 'committed')`,
      ).all(modelId) as unknown as Array<{ id: string }>;
      this.store.database.raw.prepare(
        `UPDATE route_reservations
         SET status = 'released', released_at = ?, updated_at = ?
         WHERE model_id = ? AND status IN ('prepared', 'committed')`,
      ).run(now, now, modelId);
      this.store.database.raw.prepare(
        `UPDATE deployment_stage_leases
         SET status = 'released', updated_at = ?
         WHERE model_id = ? AND status IN ('prepared', 'committed')`,
      ).run(now, modelId);
      for (const row of rows) this.queueReservation(row.id);
      return rows.length;
    });
  }

  getReservation(reservationId: string): RouteReservation | null {
    const row = this.store.database.raw.prepare(
      "SELECT * FROM route_reservations WHERE id = ?",
    ).get(reservationId) as RouteReservationRow | undefined;
    return row ? mapRouteReservation(row) : null;
  }

  listReservations(modelId: string, limit = 100): RouteReservation[] {
    const rows = this.store.database.raw.prepare(
      `SELECT * FROM route_reservations
       WHERE model_id = ?
       ORDER BY created_at DESC, id DESC
       LIMIT ?`,
    ).all(modelId, limit) as unknown as RouteReservationRow[];
    return rows.map(mapRouteReservation).toReversed();
  }

  timeline(modelId: string): DeploymentTimeline | null {
    const state = this.getState(modelId);
    if (!state) return null;
    return {
      state,
      operations: this.listOperations(modelId),
      reservations: this.listReservations(modelId),
    };
  }

  removeModelState(modelId: string): void {
    this.store.database.transaction(() => {
      this.releaseRoutesForModel(modelId);
      this.store.database.raw.prepare(
        "DELETE FROM deployment_states WHERE model_id = ?",
      ).run(modelId);
      this.store.database.enqueueRemoteChange("deployment_states", modelId, "delete", null);
    });
  }

  private setOperationObservedState(
    operationId: string,
    observedState: DeploymentObservedState,
    now: number,
  ): boolean {
    return this.store.database.transaction(() => {
      const operation = this.getOperation(operationId);
      if (!operation || operation.status !== "running") return false;
      const changed = Number(this.store.database.raw.prepare(
        `UPDATE deployment_states
         SET observed_state = ?, updated_at = ?
         WHERE model_id = ? AND active_operation_id = ? AND controller_owner = ?`,
      ).run(observedState, now, operation.modelId, operationId, this.ownerId).changes);
      if (changed === 1) {
        this.store.database.raw.prepare(
          "UPDATE deployment_operations SET updated_at = ? WHERE id = ?",
        ).run(now, operationId);
        this.queueOperation(operationId);
        this.queueState(operation.modelId);
      }
      return changed === 1;
    });
  }

  private finishOperationRow(
    operationId: string,
    status: Extract<DeploymentOperationStatus, "failed" | "interrupted">,
    errorCode: string,
    errorMessage: string,
    now: number,
  ): void {
    this.store.database.raw.prepare(
      `UPDATE deployment_operations
       SET status = ?, error_code = ?, error_message = ?, updated_at = ?, finished_at = ?
       WHERE id = ? AND status IN ('pending', 'running')`,
    ).run(status, errorCode, errorMessage.slice(0, 2_000), now, now, operationId);
  }

  private failPreparedReservations(operationId: string, error: string, now: number): void {
    const rows = this.store.database.raw.prepare(
      `SELECT id FROM route_reservations
       WHERE operation_id = ? AND status = 'prepared'`,
    ).all(operationId) as unknown as Array<{ id: string }>;
    this.store.database.raw.prepare(
      `UPDATE route_reservations
       SET status = 'failed', error = ?, released_at = ?, updated_at = ?
       WHERE operation_id = ? AND status = 'prepared'`,
    ).run(error.slice(0, 2_000), now, now, operationId);
    this.store.database.raw.prepare(
      `UPDATE deployment_stage_leases
       SET status = 'released', updated_at = ?
       WHERE reservation_id IN (
         SELECT id FROM route_reservations WHERE operation_id = ?
       ) AND status = 'prepared'`,
    ).run(now, operationId);
    for (const row of rows) this.queueReservation(row.id);
  }

  private expireReservations(now: number): number {
    const rows = this.store.database.raw.prepare(
      `SELECT id FROM route_reservations
       WHERE status IN ('prepared', 'committed') AND expires_at <= ?`,
    ).all(now) as unknown as Array<{ id: string }>;
    if (rows.length === 0) return 0;
    this.store.database.raw.prepare(
      `UPDATE route_reservations
       SET status = 'expired', released_at = ?, updated_at = ?
       WHERE status IN ('prepared', 'committed') AND expires_at <= ?`,
    ).run(now, now, now);
    this.store.database.raw.prepare(
      `UPDATE deployment_stage_leases
       SET status = 'expired', updated_at = ?
       WHERE status IN ('prepared', 'committed') AND expires_at <= ?`,
    ).run(now, now);
    for (const row of rows) this.queueReservation(row.id);
    return rows.length;
  }

  private queueAll(): void {
    for (const state of this.listStates()) this.queueState(state.modelId);
    const operations = this.store.database.raw.prepare(
      "SELECT id FROM deployment_operations",
    ).all() as unknown as Array<{ id: string }>;
    for (const operation of operations) this.queueOperation(operation.id);
    const reservations = this.store.database.raw.prepare(
      "SELECT id FROM route_reservations",
    ).all() as unknown as Array<{ id: string }>;
    for (const reservation of reservations) this.queueReservation(reservation.id);
  }

  private queueState(modelId: string): void {
    const state = this.getState(modelId);
    if (!state) return;
    this.store.database.enqueueRemoteChange("deployment_states", modelId, "upsert", {
      model_id: state.modelId,
      desired_state: state.desiredState,
      observed_state: state.observedState,
      generation: state.generation,
      observed_generation: state.observedGeneration,
      retry_count: state.retryCount,
      next_retry_at: state.nextRetryAt,
      last_error: state.lastError,
      active_operation_id: state.activeOperationId,
      controller_owner: state.controllerOwner,
      controller_lease_until: state.controllerLeaseUntil,
      created_at: state.createdAt,
      updated_at: state.updatedAt,
    });
  }

  private queueOperation(operationId: string): void {
    const operation = this.getOperation(operationId);
    if (!operation) return;
    this.store.database.enqueueRemoteChange("deployment_operations", operationId, "upsert", {
      id: operation.id,
      model_id: operation.modelId,
      generation: operation.generation,
      kind: operation.kind,
      status: operation.status,
      attempt: operation.attempt,
      idempotency_key: operation.idempotencyKey,
      error_code: operation.errorCode,
      error_message: operation.errorMessage,
      metadata: operation.metadata,
      started_at: operation.startedAt,
      updated_at: operation.updatedAt,
      finished_at: operation.finishedAt,
    });
  }

  private queueReservation(reservationId: string): void {
    const reservation = this.getReservation(reservationId);
    if (!reservation) return;
    this.store.database.enqueueRemoteChange("route_reservations", reservationId, "upsert", {
      id: reservation.id,
      model_id: reservation.modelId,
      operation_id: reservation.operationId,
      generation: reservation.generation,
      status: reservation.status,
      route_digest: reservation.routeDigest,
      stages: reservation.stages,
      canary: reservation.canary,
      expires_at: reservation.expiresAt,
      committed_at: reservation.committedAt,
      released_at: reservation.releasedAt,
      error: reservation.error,
      created_at: reservation.createdAt,
      updated_at: reservation.updatedAt,
    });
    const leases = this.store.database.raw.prepare(
      "SELECT id FROM deployment_stage_leases WHERE reservation_id = ?",
    ).all(reservationId) as unknown as Array<{ id: string }>;
    for (const lease of leases) this.queueStageLease(lease.id);
  }

  private queueStageLease(leaseId: string): void {
    const row = this.store.database.raw.prepare(
      "SELECT * FROM deployment_stage_leases WHERE id = ?",
    ).get(leaseId) as
      | {
          id: string;
          reservation_id: string;
          model_id: string;
          node_id: string;
          stage_index: number;
          memory_mib: number;
          status: string;
          expires_at: number;
          created_at: number;
          updated_at: number;
        }
      | undefined;
    if (!row) return;
    this.store.database.enqueueRemoteChange("deployment_stage_leases", leaseId, "upsert", {
      id: row.id,
      reservation_id: row.reservation_id,
      model_id: row.model_id,
      node_id: row.node_id,
      stage_index: Number(row.stage_index),
      memory_mib: Number(row.memory_mib),
      status: row.status,
      expires_at: Number(row.expires_at),
      created_at: Number(row.created_at),
      updated_at: Number(row.updated_at),
    });
  }
}

function mapDeploymentState(row: DeploymentStateRow): DeploymentState {
  return {
    modelId: row.model_id,
    desiredState: row.desired_state,
    observedState: row.observed_state,
    generation: Number(row.generation),
    observedGeneration: Number(row.observed_generation),
    retryCount: Number(row.retry_count),
    nextRetryAt: row.next_retry_at === null ? null : Number(row.next_retry_at),
    lastError: row.last_error,
    activeOperationId: row.active_operation_id,
    controllerOwner: row.controller_owner,
    controllerLeaseUntil:
      row.controller_lease_until === null ? null : Number(row.controller_lease_until),
    createdAt: Number(row.created_at),
    updatedAt: Number(row.updated_at),
  };
}

function mapDeploymentOperation(row: DeploymentOperationRow): DeploymentOperation {
  return {
    id: row.id,
    modelId: row.model_id,
    generation: Number(row.generation),
    kind: row.kind,
    status: row.status,
    attempt: Number(row.attempt),
    idempotencyKey: row.idempotency_key,
    errorCode: row.error_code,
    errorMessage: row.error_message,
    metadata: JSON.parse(row.metadata_json) as Record<string, unknown>,
    startedAt: Number(row.started_at),
    updatedAt: Number(row.updated_at),
    finishedAt: row.finished_at === null ? null : Number(row.finished_at),
  };
}

function mapRouteReservation(row: RouteReservationRow): RouteReservation {
  return {
    id: row.id,
    modelId: row.model_id,
    operationId: row.operation_id,
    generation: Number(row.generation),
    status: row.status,
    routeDigest: row.route_digest,
    stages: JSON.parse(row.stages_json) as RouteStageReservation[],
    canary: row.canary_json
      ? JSON.parse(row.canary_json) as Record<string, unknown>
      : null,
    expiresAt: Number(row.expires_at),
    committedAt: row.committed_at === null ? null : Number(row.committed_at),
    releasedAt: row.released_at === null ? null : Number(row.released_at),
    error: row.error,
    createdAt: Number(row.created_at),
    updatedAt: Number(row.updated_at),
  };
}
