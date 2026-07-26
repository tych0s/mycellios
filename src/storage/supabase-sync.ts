import type { MeshStore } from "./store.js";
import type { BenchmarkRun } from "../benchlab/types.js";
import { parseBenchmarkRun } from "../benchlab/history.js";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

const TABLE_PRIMARY_KEYS = Object.freeze({
  workers: "id",
  jobs: "id",
  sessions: "id",
  idempotency_keys: "idempotency_key",
  requested_models: "id",
  benchmark_runs: "run_id",
  inference_conversations: "id",
  inference_messages: "id",
  activation_events: "id",
  deployment_states: "model_id",
  deployment_operations: "id",
  route_reservations: "id",
  deployment_stage_leases: "id",
} as const);

type SyncedTable = keyof typeof TABLE_PRIMARY_KEYS;

export interface SupabasePersistenceOptions {
  url: string;
  serviceRoleKey: string;
  required?: boolean;
  flushIntervalMs?: number;
  fetchImpl?: typeof fetch;
}

export interface SupabasePersistenceStatus {
  configured: true;
  connected: boolean;
  required: boolean;
  pendingChanges: number;
  pendingArtifacts: number;
  lastSuccessfulSyncAt: string | null;
  lastError: string | null;
}

export class SupabasePersistence {
  private readonly baseUrl: URL;
  private readonly fetchImpl: typeof fetch;
  private readonly required: boolean;
  private readonly flushIntervalMs: number;
  private timer: NodeJS.Timeout | null = null;
  private flushPromise: Promise<void> | null = null;
  private connected = false;
  private lastSuccessfulSyncAt: string | null = null;
  private lastError: string | null = null;
  private closed = false;

  constructor(
    private readonly store: MeshStore,
    private readonly options: SupabasePersistenceOptions,
  ) {
    this.baseUrl = new URL(options.url);
    if (this.baseUrl.protocol !== "https:" && this.baseUrl.hostname !== "127.0.0.1") {
      throw new Error("Supabase persistence requires HTTPS outside loopback.");
    }
    this.fetchImpl = options.fetchImpl ?? fetch;
    this.required = options.required ?? false;
    this.flushIntervalMs = options.flushIntervalMs ?? 1_000;
  }

  async initialize(): Promise<void> {
    try {
      await this.assertReachable();
      await this.restoreRemoteState();
      this.store.queueAllForRemotePersistence();
      await this.flush();
    } catch (error) {
      this.connected = false;
      this.lastError = errorText(error);
      if (this.required) throw error;
    }
    if (!this.closed) {
      this.timer = setInterval(() => void this.flush(), this.flushIntervalMs);
      this.timer.unref();
    }
  }

  status(): SupabasePersistenceStatus {
    return {
      configured: true,
      connected: this.connected,
      required: this.required,
      pendingChanges: this.store.database.pendingRemoteChangeCount(),
      pendingArtifacts: this.store.database.pendingArtifactBackupCount(),
      lastSuccessfulSyncAt: this.lastSuccessfulSyncAt,
      lastError: this.lastError,
    };
  }

  flush(): Promise<void> {
    if (this.closed) return Promise.resolve();
    if (this.flushPromise) return this.flushPromise;
    this.flushPromise = this.flushPendingChanges()
      .then(() => this.flushPendingArtifacts())
      .finally(() => {
      this.flushPromise = null;
      });
    return this.flushPromise;
  }

  async close(): Promise<void> {
    this.closed = true;
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    await this.flushPromise?.catch(() => undefined);
    await this.flushPendingChanges().catch((error) => {
      if (this.required) throw error;
    });
    await this.flushPendingArtifacts().catch((error) => {
      if (this.required) throw error;
    });
  }

  registerArtifactBackup(input: {
    id: string;
    localPath: string;
    storagePath: string;
    contentType: string;
    sha256: string;
    sizeBytes: number;
    metadata?: Record<string, unknown>;
  }): void {
    this.store.database.enqueueArtifactBackup({
      ...input,
      metadata: input.metadata ?? {},
    });
    void this.flush();
  }

  async readBenchmarkRuns(): Promise<BenchmarkRun[]> {
    const rows = await this.readTable("benchmark_runs");
    return rows
      .map((row) => parseBenchmarkRun(row.document))
      .filter((document): document is BenchmarkRun => document !== null)
      .sort((left, right) => left.finishedAt.localeCompare(right.finishedAt));
  }

  private async assertReachable(): Promise<void> {
    const response = await this.request("workers?select=id&limit=1", { method: "GET" });
    if (!response.ok) throw await responseError(response, "Supabase persistence health check failed");
    this.connected = true;
    this.lastError = null;
  }

  private async restoreRemoteState(): Promise<void> {
    const [
      workers,
      requestedModels,
      jobs,
      sessions,
      idempotencyKeys,
      inferenceConversations,
      inferenceMessages,
      activationEvents,
      deploymentStates,
      deploymentOperations,
      routeReservations,
      deploymentStageLeases,
    ] = await Promise.all([
      this.readTable("workers"),
      this.readTable("requested_models"),
      this.readTable("jobs"),
      this.readTable("sessions"),
      this.readTable("idempotency_keys"),
      this.readTable("inference_conversations"),
      this.readTable("inference_messages"),
      this.readTable("activation_events"),
      this.readTable("deployment_states"),
      this.readTable("deployment_operations"),
      this.readTable("route_reservations"),
      this.readTable("deployment_stage_leases"),
    ]);
    this.store.database.transaction(() => {
      for (const row of workers) this.restoreWorker(row);
      for (const row of requestedModels) this.restoreRequestedModel(row);
      for (const row of jobs) this.restoreJob(row);
      for (const row of sessions) this.restoreSession(row);
      for (const row of idempotencyKeys) this.restoreIdempotencyKey(row);
      for (const row of inferenceConversations) this.restoreInferenceConversation(row);
      for (const row of inferenceMessages) this.restoreInferenceMessage(row);
      for (const row of activationEvents) this.restoreActivationEvent(row);
      for (const row of deploymentOperations) this.restoreDeploymentOperation(row);
      for (const row of routeReservations) this.restoreRouteReservation(row);
      for (const row of deploymentStageLeases) this.restoreDeploymentStageLease(row);
      for (const row of deploymentStates) this.restoreDeploymentState(row);
    });
  }

  private async readTable(table: SyncedTable): Promise<Array<Record<string, unknown>>> {
    const rows: Array<Record<string, unknown>> = [];
    for (let offset = 0; ; offset += 1_000) {
      const response = await this.request(`${table}?select=*`, {
        method: "GET",
        headers: { Range: `${offset}-${offset + 999}` },
      });
      if (!response.ok) throw await responseError(response, `Could not read ${table} from Supabase`);
      const body = await response.json() as unknown;
      if (!Array.isArray(body)) throw new Error(`Supabase returned a non-array payload for ${table}.`);
      rows.push(...body.filter(isRecord));
      if (body.length < 1_000) return rows;
    }
  }

  private async flushPendingChanges(): Promise<void> {
    let processed = 0;
    while (!this.closed || processed === 0) {
      const pending = this.store.database.listPendingRemoteChanges(100);
      if (pending.length === 0) break;
      for (const change of pending) {
        try {
          if (!isSyncedTable(change.tableName)) {
            throw new Error(`Unsupported Supabase persistence table: ${change.tableName}`);
          }
          const primaryKey = TABLE_PRIMARY_KEYS[change.tableName];
          const response = change.operation === "delete"
            ? await this.request(
                `${change.tableName}?${primaryKey}=eq.${encodeURIComponent(change.recordKey)}`,
                { method: "DELETE" },
              )
            : await this.request(change.tableName, {
                method: "POST",
                headers: { Prefer: "resolution=merge-duplicates,return=minimal" },
                body: change.payloadJson ?? "{}",
              });
          if (!response.ok) {
            throw await responseError(response, `Could not persist ${change.tableName}/${change.recordKey}`);
          }
          this.store.database.markRemoteChangeSynced(change.id);
          this.connected = true;
          this.lastSuccessfulSyncAt = new Date().toISOString();
          this.lastError = null;
        } catch (error) {
          const message = errorText(error);
          this.store.database.markRemoteChangeFailed(change.id, message);
          this.connected = false;
          this.lastError = message;
          if (this.required && change.attempts >= 4) throw error;
          return;
        }
        processed += 1;
      }
      if (pending.length < 100) break;
    }
  }

  private async flushPendingArtifacts(): Promise<void> {
    const pending = this.store.database.listPendingArtifactBackups(10);
    for (const artifact of pending) {
      try {
        const body = await readFile(artifact.localPath);
        if (body.length !== artifact.sizeBytes) {
          throw new Error(`Artifact size changed before backup: ${artifact.id}`);
        }
        const digest = createHash("sha256").update(body).digest("hex");
        if (digest !== artifact.sha256) {
          throw new Error(`Artifact digest changed before backup: ${artifact.id}`);
        }
        const upload = await this.fetchImpl(
          new URL(`/storage/v1/object/mycellios-artifacts/${artifact.storagePath}`, this.baseUrl),
          {
            method: "POST",
            signal: AbortSignal.timeout(5 * 60_000),
            headers: {
              apikey: this.options.serviceRoleKey,
              authorization: `Bearer ${this.options.serviceRoleKey}`,
              "content-type": artifact.contentType,
              "x-upsert": "true",
            },
            body,
          },
        );
        if (!upload.ok) throw await responseError(upload, `Could not back up artifact ${artifact.id}`);
        const metadata = await this.request("artifacts", {
          method: "POST",
          headers: { Prefer: "resolution=merge-duplicates,return=minimal" },
          body: JSON.stringify({
            id: artifact.id,
            kind: JSON.parse(artifact.metadataJson).kind ?? "runtime",
            storage_bucket: "mycellios-artifacts",
            storage_path: artifact.storagePath,
            sha256: artifact.sha256,
            size_bytes: artifact.sizeBytes,
            metadata: JSON.parse(artifact.metadataJson),
          }),
        });
        if (!metadata.ok) throw await responseError(metadata, `Could not persist artifact metadata ${artifact.id}`);
        this.store.database.markArtifactBackupSynced(artifact.id);
        this.connected = true;
        this.lastSuccessfulSyncAt = new Date().toISOString();
        this.lastError = null;
      } catch (error) {
        const message = errorText(error);
        this.store.database.markArtifactBackupFailed(artifact.id, message);
        this.connected = false;
        this.lastError = message;
        if (this.required && artifact.attempts >= 4) throw error;
        return;
      }
    }
  }

  private request(path: string, init: RequestInit): Promise<Response> {
    return this.fetchImpl(new URL(`/rest/v1/${path}`, this.baseUrl), {
      ...init,
      signal: AbortSignal.timeout(15_000),
      headers: {
        apikey: this.options.serviceRoleKey,
        authorization: `Bearer ${this.options.serviceRoleKey}`,
        "content-type": "application/json",
        ...(init.headers ?? {}),
      },
    });
  }

  private restoreWorker(row: Record<string, unknown>): void {
    if (typeof row.id !== "string" || typeof row.updated_at !== "number") return;
    const existing = this.store.database.raw.prepare(
      "SELECT updated_at FROM workers WHERE id = ?",
    ).get(row.id) as { updated_at: number } | undefined;
    if (existing && Number(existing.updated_at) >= row.updated_at) return;
    this.store.database.raw.prepare(
      `INSERT INTO workers(
         id, status, capabilities_json, reliability, jobs_completed, last_seen_at,
         created_at, updated_at, deregistered, identity_kind, identity_id
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         status=excluded.status, capabilities_json=excluded.capabilities_json,
         reliability=excluded.reliability, jobs_completed=excluded.jobs_completed,
         last_seen_at=excluded.last_seen_at, updated_at=excluded.updated_at,
         deregistered=excluded.deregistered, identity_kind=excluded.identity_kind,
         identity_id=excluded.identity_id`,
    ).run(
      row.id,
      stringValue(row.status, "offline"),
      JSON.stringify(row.capabilities_json ?? {}),
      numberValue(row.reliability, 0.95),
      numberValue(row.jobs_completed, 0),
      numberValue(row.last_seen_at, 0),
      numberValue(row.created_at, row.updated_at),
      row.updated_at,
      row.deregistered === true ? 1 : 0,
      nullableString(row.identity_kind),
      nullableString(row.identity_id),
    );
  }

  private restoreRequestedModel(row: Record<string, unknown>): void {
    if (typeof row.id !== "string" || typeof row.updated_at !== "number") return;
    const existing = this.store.database.raw.prepare(
      "SELECT updated_at FROM requested_models WHERE id = ?",
    ).get(row.id) as { updated_at: number } | undefined;
    if (existing && Number(existing.updated_at) >= row.updated_at) return;
    this.store.database.raw.prepare(
      `INSERT INTO requested_models(
         id, source, revision, context_tokens, minimum_nodes, auto_activate,
         profile_json, profile_error, activation_requested_at, activation_error,
         created_at, updated_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         source=excluded.source, revision=excluded.revision,
         context_tokens=excluded.context_tokens, minimum_nodes=excluded.minimum_nodes,
         auto_activate=excluded.auto_activate, profile_json=excluded.profile_json,
         profile_error=excluded.profile_error,
         activation_requested_at=excluded.activation_requested_at,
         activation_error=excluded.activation_error, updated_at=excluded.updated_at`,
    ).run(
      row.id,
      stringValue(row.source, row.id),
      nullableString(row.revision),
      numberValue(row.context_tokens, 4096),
      numberValue(row.minimum_nodes, 1),
      row.auto_activate === false ? 0 : 1,
      row.profile_json == null ? null : JSON.stringify(row.profile_json),
      nullableString(row.profile_error),
      nullableNumber(row.activation_requested_at),
      nullableString(row.activation_error),
      numberValue(row.created_at, row.updated_at),
      row.updated_at,
    );
  }

  private restoreJob(row: Record<string, unknown>): void {
    if (typeof row.id !== "string" || typeof row.updated_at !== "number") return;
    const existing = this.store.database.raw.prepare(
      "SELECT updated_at FROM jobs WHERE id = ?",
    ).get(row.id) as { updated_at: number } | undefined;
    if (existing && Number(existing.updated_at) >= row.updated_at) return;
    this.store.database.raw.prepare(
      `INSERT INTO jobs(
         id, session_id, model, workload_class, status, worker_id, deployment_id,
         model_digest, lease_id, input_tokens, output_tokens, failure_code,
         deadline_at, created_at, updated_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         session_id=excluded.session_id, model=excluded.model,
         workload_class=excluded.workload_class, status=excluded.status,
         worker_id=excluded.worker_id, deployment_id=excluded.deployment_id,
         model_digest=excluded.model_digest, lease_id=excluded.lease_id,
         input_tokens=excluded.input_tokens, output_tokens=excluded.output_tokens,
         failure_code=excluded.failure_code, deadline_at=excluded.deadline_at,
         updated_at=excluded.updated_at`,
    ).run(
      row.id,
      stringValue(row.session_id, row.id),
      stringValue(row.model, "unknown"),
      stringValue(row.workload_class, "interactive"),
      stringValue(row.status, "failed"),
      nullableString(row.worker_id),
      nullableString(row.deployment_id),
      nullableString(row.model_digest),
      nullableString(row.lease_id),
      numberValue(row.input_tokens, 0),
      numberValue(row.output_tokens, 0),
      nullableString(row.failure_code),
      numberValue(row.deadline_at, row.updated_at),
      numberValue(row.created_at, row.updated_at),
      row.updated_at,
    );
  }

  private restoreSession(row: Record<string, unknown>): void {
    if (typeof row.id !== "string") return;
    const remoteLastUsed = numberValue(row.last_used_at, 0);
    const existing = this.store.database.raw.prepare(
      "SELECT last_used_at FROM sessions WHERE id = ?",
    ).get(row.id) as { last_used_at: number } | undefined;
    if (existing && Number(existing.last_used_at) >= remoteLastUsed) return;
    this.store.database.raw.prepare(
      `INSERT INTO sessions(id, model, route_json, expires_at, last_used_at, version)
       VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         model=excluded.model, route_json=excluded.route_json,
         expires_at=excluded.expires_at, last_used_at=excluded.last_used_at,
         version=excluded.version`,
    ).run(
      row.id,
      stringValue(row.model, "unknown"),
      JSON.stringify(row.route_json ?? {}),
      numberValue(row.expires_at, 0),
      remoteLastUsed,
      numberValue(row.version, 1),
    );
  }

  private restoreIdempotencyKey(row: Record<string, unknown>): void {
    if (typeof row.idempotency_key !== "string" || typeof row.job_id !== "string") return;
    this.store.database.raw.prepare(
      `INSERT OR IGNORE INTO idempotency_keys(
         idempotency_key, request_hash, job_id, created_at
       ) VALUES (?, ?, ?, ?)`,
    ).run(
      row.idempotency_key,
      stringValue(row.request_hash, ""),
      row.job_id,
      numberValue(row.created_at, Date.now()),
    );
  }

  private restoreInferenceConversation(row: Record<string, unknown>): void {
    if (typeof row.id !== "string" || typeof row.session_id !== "string") return;
    const createdAt = timestampValue(row.created_at, Date.now());
    const updatedAt = timestampValue(row.updated_at, createdAt);
    const existing = this.store.database.raw.prepare(
      "SELECT updated_at FROM inference_conversations WHERE id = ?",
    ).get(row.id) as { updated_at: number } | undefined;
    if (existing && Number(existing.updated_at) >= updatedAt) return;
    this.store.database.raw.prepare(
      `INSERT INTO inference_conversations(id, session_id, model, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         session_id=excluded.session_id, model=excluded.model,
         updated_at=excluded.updated_at`,
    ).run(
      row.id,
      row.session_id,
      stringValue(row.model, "unknown"),
      createdAt,
      updatedAt,
    );
  }

  private restoreInferenceMessage(row: Record<string, unknown>): void {
    if (
      typeof row.id !== "string"
      || typeof row.conversation_id !== "string"
      || typeof row.role !== "string"
      || typeof row.content !== "string"
    ) return;
    this.store.database.raw.prepare(
      `INSERT OR IGNORE INTO inference_messages(
         id, conversation_id, job_id, role, content, status, input_tokens,
         output_tokens, route_class, latency_ms, metadata_json, created_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      row.id,
      row.conversation_id,
      nullableString(row.job_id),
      row.role,
      row.content,
      stringValue(row.status, "completed"),
      nullableNumber(row.input_tokens),
      nullableNumber(row.output_tokens),
      nullableString(row.route_class),
      nullableNumber(row.latency_ms),
      JSON.stringify(row.metadata ?? {}),
      timestampValue(row.created_at, Date.now()),
    );
  }

  private restoreActivationEvent(row: Record<string, unknown>): void {
    if (
      typeof row.id !== "string"
      || typeof row.model_id !== "string"
      || typeof row.phase !== "string"
      || typeof row.state !== "string"
      || typeof row.message !== "string"
    ) return;
    this.store.database.raw.prepare(
      `INSERT OR IGNORE INTO activation_events(
         id, model_id, phase, state, message, node_id, process_id, device,
         details_json, occurred_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      row.id,
      row.model_id,
      row.phase,
      row.state,
      row.message,
      nullableString(row.node_id),
      nullableString(row.process_id),
      nullableString(row.device),
      row.details == null ? null : JSON.stringify(row.details),
      numberValue(row.occurred_at, Date.now()),
    );
  }

  private restoreDeploymentState(row: Record<string, unknown>): void {
    if (typeof row.model_id !== "string" || typeof row.updated_at !== "number") return;
    const existing = this.store.database.raw.prepare(
      "SELECT updated_at FROM deployment_states WHERE model_id = ?",
    ).get(row.model_id) as { updated_at: number } | undefined;
    if (existing && Number(existing.updated_at) >= row.updated_at) return;
    this.store.database.raw.prepare(
      `INSERT INTO deployment_states(
         model_id, desired_state, observed_state, generation, observed_generation,
         retry_count, next_retry_at, last_error, active_operation_id,
         controller_owner, controller_lease_until, created_at, updated_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(model_id) DO UPDATE SET
         desired_state=excluded.desired_state,
         observed_state=excluded.observed_state,
         generation=excluded.generation,
         observed_generation=excluded.observed_generation,
         retry_count=excluded.retry_count,
         next_retry_at=excluded.next_retry_at,
         last_error=excluded.last_error,
         active_operation_id=excluded.active_operation_id,
         controller_owner=excluded.controller_owner,
         controller_lease_until=excluded.controller_lease_until,
         updated_at=excluded.updated_at`,
    ).run(
      row.model_id,
      stringValue(row.desired_state, "inactive"),
      stringValue(row.observed_state, "inactive"),
      numberValue(row.generation, 1),
      numberValue(row.observed_generation, 0),
      numberValue(row.retry_count, 0),
      nullableNumber(row.next_retry_at),
      nullableString(row.last_error),
      nullableString(row.active_operation_id),
      nullableString(row.controller_owner),
      nullableNumber(row.controller_lease_until),
      numberValue(row.created_at, row.updated_at),
      row.updated_at,
    );
  }

  private restoreDeploymentOperation(row: Record<string, unknown>): void {
    if (
      typeof row.id !== "string"
      || typeof row.model_id !== "string"
      || typeof row.updated_at !== "number"
    ) return;
    const existing = this.store.database.raw.prepare(
      "SELECT updated_at FROM deployment_operations WHERE id = ?",
    ).get(row.id) as { updated_at: number } | undefined;
    if (existing && Number(existing.updated_at) >= row.updated_at) return;
    this.store.database.raw.prepare(
      `INSERT INTO deployment_operations(
         id, model_id, generation, kind, status, attempt, idempotency_key,
         error_code, error_message, metadata_json, started_at, updated_at, finished_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         generation=excluded.generation, kind=excluded.kind, status=excluded.status,
         attempt=excluded.attempt, idempotency_key=excluded.idempotency_key,
         error_code=excluded.error_code, error_message=excluded.error_message,
         metadata_json=excluded.metadata_json, updated_at=excluded.updated_at,
         finished_at=excluded.finished_at`,
    ).run(
      row.id,
      row.model_id,
      numberValue(row.generation, 1),
      stringValue(row.kind, "activate"),
      stringValue(row.status, "interrupted"),
      numberValue(row.attempt, 1),
      stringValue(row.idempotency_key, row.id),
      nullableString(row.error_code),
      nullableString(row.error_message),
      JSON.stringify(row.metadata ?? {}),
      numberValue(row.started_at, row.updated_at),
      row.updated_at,
      nullableNumber(row.finished_at),
    );
  }

  private restoreRouteReservation(row: Record<string, unknown>): void {
    if (
      typeof row.id !== "string"
      || typeof row.model_id !== "string"
      || typeof row.operation_id !== "string"
      || typeof row.updated_at !== "number"
    ) return;
    const existing = this.store.database.raw.prepare(
      "SELECT updated_at FROM route_reservations WHERE id = ?",
    ).get(row.id) as { updated_at: number } | undefined;
    if (existing && Number(existing.updated_at) >= row.updated_at) return;
    this.store.database.raw.prepare(
      `INSERT INTO route_reservations(
         id, model_id, operation_id, generation, status, route_digest,
         stages_json, canary_json, expires_at, committed_at, released_at,
         error, created_at, updated_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         status=excluded.status, route_digest=excluded.route_digest,
         stages_json=excluded.stages_json, canary_json=excluded.canary_json,
         expires_at=excluded.expires_at, committed_at=excluded.committed_at,
         released_at=excluded.released_at, error=excluded.error,
         updated_at=excluded.updated_at`,
    ).run(
      row.id,
      row.model_id,
      row.operation_id,
      numberValue(row.generation, 1),
      stringValue(row.status, "expired"),
      stringValue(row.route_digest, row.id),
      JSON.stringify(row.stages ?? []),
      row.canary == null ? null : JSON.stringify(row.canary),
      numberValue(row.expires_at, 0),
      nullableNumber(row.committed_at),
      nullableNumber(row.released_at),
      nullableString(row.error),
      numberValue(row.created_at, row.updated_at),
      row.updated_at,
    );
  }

  private restoreDeploymentStageLease(row: Record<string, unknown>): void {
    if (
      typeof row.id !== "string"
      || typeof row.reservation_id !== "string"
      || typeof row.model_id !== "string"
      || typeof row.node_id !== "string"
      || typeof row.updated_at !== "number"
    ) return;
    const existing = this.store.database.raw.prepare(
      "SELECT updated_at FROM deployment_stage_leases WHERE id = ?",
    ).get(row.id) as { updated_at: number } | undefined;
    if (existing && Number(existing.updated_at) >= row.updated_at) return;
    this.store.database.raw.prepare(
      `INSERT INTO deployment_stage_leases(
         id, reservation_id, model_id, node_id, stage_index, memory_mib,
         status, expires_at, created_at, updated_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         status=excluded.status, expires_at=excluded.expires_at,
         memory_mib=excluded.memory_mib, updated_at=excluded.updated_at`,
    ).run(
      row.id,
      row.reservation_id,
      row.model_id,
      row.node_id,
      numberValue(row.stage_index, 0),
      numberValue(row.memory_mib, 0),
      stringValue(row.status, "expired"),
      numberValue(row.expires_at, 0),
      numberValue(row.created_at, row.updated_at),
      row.updated_at,
    );
  }
}

function isSyncedTable(value: string): value is SyncedTable {
  return Object.hasOwn(TABLE_PRIMARY_KEYS, value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function stringValue(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

function nullableString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function numberValue(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function nullableNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function timestampValue(value: unknown, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

async function responseError(response: Response, prefix: string): Promise<Error> {
  const text = (await response.text()).slice(0, 1_000);
  return new Error(`${prefix} (HTTP ${response.status})${text ? `: ${text}` : ""}`);
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
