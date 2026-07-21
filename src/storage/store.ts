import type {
  CompletionMetrics,
  JobStatus,
  ScheduledRoute,
  WorkerCapabilities,
  WorkerRegistration,
  WorkerStatus,
} from "../contracts/types.js";
import { newId } from "../core/ids.js";
import { MeshDatabase } from "./database.js";

export interface StoredWorker {
  id: string;
  status: WorkerStatus;
  capabilities: WorkerCapabilities;
  reliability: number;
  jobsCompleted: number;
  lastSeenAt: number;
}

export interface StoredJob {
  id: string;
  sessionId: string;
  model: string;
  workloadClass: string;
  status: JobStatus;
  workerId: string | null;
  deploymentId: string | null;
  modelDigest: string | null;
  leaseId: string | null;
  inputTokens: number;
  outputTokens: number;
  failureCode: string | null;
  deadlineAt: number;
  createdAt: number;
  updatedAt: number;
}

interface WorkerRow {
  id: string;
  status: WorkerStatus;
  capabilities_json: string;
  reliability: number;
  jobs_completed: number;
  last_seen_at: number;
}

interface JobRow {
  id: string;
  session_id: string;
  model: string;
  workload_class: string;
  status: JobStatus;
  worker_id: string | null;
  deployment_id: string | null;
  model_digest: string | null;
  lease_id: string | null;
  input_tokens: number;
  output_tokens: number;
  failure_code: string | null;
  deadline_at: number;
  created_at: number;
  updated_at: number;
}

export class MeshStore {
  constructor(readonly database: MeshDatabase) {}

  registerWorker(registration: WorkerRegistration): StoredWorker {
    const workerId = newId("wrk");
    const now = Date.now();
    this.database.raw
      .prepare(
        `INSERT INTO workers(
           id, status, capabilities_json, last_seen_at, created_at, updated_at
         ) VALUES (?, 'offline', ?, ?, ?, ?)`,
      )
      .run(workerId, JSON.stringify(registration.capabilities), now, now, now);
    return this.getWorker(workerId)!;
  }

  getWorker(workerId: string): StoredWorker | null {
    const row = this.database.raw
      .prepare("SELECT * FROM workers WHERE id = ? AND deregistered = 0")
      .get(workerId) as
      | WorkerRow
      | undefined;
    return row ? this.mapWorker(row) : null;
  }

  listWorkers(): StoredWorker[] {
    const rows = this.database.raw
      .prepare("SELECT * FROM workers WHERE deregistered = 0")
      .all() as unknown as WorkerRow[];
    return rows.map((row) => this.mapWorker(row));
  }

  listSchedulableWorkers(now = Date.now(), staleAfterMs = 15_000): StoredWorker[] {
    const rows = this.database.raw
      .prepare(
        "SELECT * FROM workers WHERE deregistered = 0 AND status = 'online' AND last_seen_at >= ?",
      )
      .all(now - staleAfterMs) as unknown as WorkerRow[];
    return rows.map((row) => this.mapWorker(row));
  }

  updateWorkerHeartbeat(
    workerId: string,
    capabilities: WorkerCapabilities,
    status: WorkerStatus,
  ): void {
    const now = Date.now();
    this.database.raw
      .prepare(
        `UPDATE workers
         SET capabilities_json = ?, status = ?, last_seen_at = ?, updated_at = ?
         WHERE id = ?`,
      )
      .run(JSON.stringify(capabilities), status, now, now, workerId);
  }

  setWorkerStatus(workerId: string, status: WorkerStatus): void {
    this.database.raw
      .prepare("UPDATE workers SET status = ?, updated_at = ? WHERE id = ?")
      .run(status, Date.now(), workerId);
  }

  deregisterWorker(workerId: string): boolean {
    const result = this.database.raw
      .prepare(
        `UPDATE workers
         SET deregistered = 1, status = 'offline', updated_at = ?
         WHERE id = ? AND deregistered = 0`,
      )
      .run(Date.now(), workerId);
    return Number(result.changes) === 1;
  }

  deregisterOfflineWorkers(): number {
    const result = this.database.raw
      .prepare(
        `UPDATE workers
         SET deregistered = 1, updated_at = ?
         WHERE deregistered = 0 AND status IN ('offline', 'suspect')`,
      )
      .run(Date.now());
    return Number(result.changes);
  }

  markStaleWorkers(now = Date.now()): { suspect: number; offline: number } {
    const suspect = this.database.raw
      .prepare(
        `UPDATE workers SET status = 'suspect', updated_at = ?
         WHERE status = 'online' AND last_seen_at < ?`,
      )
      .run(now, now - 10_000).changes;
    const offline = this.database.raw
      .prepare(
        `UPDATE workers SET status = 'offline', updated_at = ?
         WHERE status IN ('online', 'suspect') AND last_seen_at < ?`,
      )
      .run(now, now - 15_000).changes;
    return { suspect: Number(suspect), offline: Number(offline) };
  }

  createJob(input: {
    id: string;
    sessionId: string;
    model: string;
    workloadClass: string;
    deadlineAt: number;
  }): StoredJob {
    const now = Date.now();
    this.database.raw
      .prepare(
        `INSERT INTO jobs(
           id, session_id, model, workload_class, status, deadline_at, created_at, updated_at
         ) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)`,
      )
      .run(
        input.id,
        input.sessionId,
        input.model,
        input.workloadClass,
        input.deadlineAt,
        now,
        now,
      );
    return this.getJob(input.id)!;
  }

  getIdempotentJob(idempotencyKey: string): { jobId: string; requestHash: string } | null {
    const row = this.database.raw
      .prepare("SELECT job_id, request_hash FROM idempotency_keys WHERE idempotency_key = ?")
      .get(idempotencyKey) as { job_id: string; request_hash: string } | undefined;
    return row ? { jobId: row.job_id, requestHash: row.request_hash } : null;
  }

  bindIdempotencyKey(idempotencyKey: string, requestHash: string, jobId: string): void {
    this.database.raw
      .prepare(
        `INSERT INTO idempotency_keys(idempotency_key, request_hash, job_id, created_at)
         VALUES (?, ?, ?, ?)`,
      )
      .run(idempotencyKey, requestHash, jobId, Date.now());
  }

  getJob(jobId: string): StoredJob | null {
    const row = this.database.raw.prepare("SELECT * FROM jobs WHERE id = ?").get(jobId) as
      | JobRow
      | undefined;
    return row ? this.mapJob(row) : null;
  }

  setJobRoute(jobId: string, route: ScheduledRoute, leaseId: string): void {
    const first = route.stages[0];
    if (!first) throw new Error("Cannot assign an empty route");
    this.database.raw
      .prepare(
        `UPDATE jobs
         SET status = 'leasing', worker_id = ?, deployment_id = ?, lease_id = ?,
             model_digest = ?, updated_at = ?
         WHERE id = ? AND status = 'queued'`,
      )
      .run(
        first.workerId,
        first.deploymentId,
        leaseId,
        first.modelDigest,
        Date.now(),
        jobId,
      );
  }

  setJobStatus(jobId: string, status: JobStatus, failureCode?: string): void {
    this.database.raw
      .prepare("UPDATE jobs SET status = ?, failure_code = ?, updated_at = ? WHERE id = ?")
      .run(status, failureCode ?? null, Date.now(), jobId);
  }

  requeueJob(jobId: string, failureCode?: string): void {
    this.database.raw
      .prepare(
        `UPDATE jobs
         SET status = 'queued', worker_id = NULL, deployment_id = NULL,
             model_digest = NULL, lease_id = NULL, failure_code = ?, updated_at = ?
         WHERE id = ? AND status NOT IN ('completed', 'cancelled', 'expired')`,
      )
      .run(failureCode ?? null, Date.now(), jobId);
  }

  completeJob(jobId: string, metrics: CompletionMetrics): void {
    this.database.transaction(() => {
      this.database.raw
        .prepare(
          `UPDATE jobs
           SET status = 'completed', input_tokens = ?, output_tokens = ?, updated_at = ?
           WHERE id = ?`,
        )
        .run(metrics.inputTokens, metrics.outputTokens, Date.now(), jobId);
      this.database.raw
        .prepare(
          `UPDATE workers SET jobs_completed = jobs_completed + 1, updated_at = ?
           WHERE id = (SELECT worker_id FROM jobs WHERE id = ?)`,
        )
        .run(Date.now(), jobId);
    });
  }

  countActiveJobs(workerId: string): number {
    const row = this.database.raw
      .prepare(
        `SELECT COUNT(*) AS count FROM jobs
         WHERE worker_id = ? AND status IN ('leasing', 'running', 'streaming')`,
      )
      .get(workerId) as { count: number };
    return Number(row.count);
  }

  listActiveJobsForWorker(workerId: string): StoredJob[] {
    const rows = this.database.raw
      .prepare(
        `SELECT * FROM jobs
         WHERE worker_id = ? AND status IN ('leasing', 'running', 'streaming')`,
      )
      .all(workerId) as unknown as JobRow[];
    return rows.map((row) => this.mapJob(row));
  }

  listNonterminalJobs(): StoredJob[] {
    const rows = this.database.raw
      .prepare(
        `SELECT * FROM jobs
         WHERE status IN ('queued', 'leasing', 'running', 'streaming')`,
      )
      .all() as unknown as JobRow[];
    return rows.map((row) => this.mapJob(row));
  }

  saveSession(sessionId: string, model: string, route: ScheduledRoute, ttlMs = 15 * 60_000): void {
    const now = Date.now();
    this.database.raw
      .prepare(
        `INSERT INTO sessions(id, model, route_json, expires_at, last_used_at)
         VALUES (?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           model = excluded.model,
           route_json = excluded.route_json,
           expires_at = excluded.expires_at,
           last_used_at = excluded.last_used_at,
           version = sessions.version + 1`,
      )
      .run(sessionId, model, JSON.stringify(route), now + ttlMs, now);
  }

  getSessionRoute(sessionId: string, model: string): ScheduledRoute | null {
    const row = this.database.raw
      .prepare(
        `SELECT route_json FROM sessions
         WHERE id = ? AND model = ? AND expires_at > ?`,
      )
      .get(sessionId, model, Date.now()) as { route_json: string } | undefined;
    return row ? (JSON.parse(row.route_json) as ScheduledRoute) : null;
  }

  listJobs(limit = 100): StoredJob[] {
    const rows = this.database.raw
      .prepare("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?")
      .all(limit) as unknown as JobRow[];
    return rows.map((row) => this.mapJob(row));
  }

  private mapWorker(row: WorkerRow): StoredWorker {
    return {
      id: row.id,
      status: row.status,
      capabilities: JSON.parse(row.capabilities_json) as WorkerCapabilities,
      reliability: Number(row.reliability),
      jobsCompleted: Number(row.jobs_completed),
      lastSeenAt: Number(row.last_seen_at),
    };
  }

  private mapJob(row: JobRow): StoredJob {
    return {
      id: row.id,
      sessionId: row.session_id,
      model: row.model,
      workloadClass: row.workload_class,
      status: row.status,
      workerId: row.worker_id,
      deploymentId: row.deployment_id,
      modelDigest: row.model_digest,
      leaseId: row.lease_id,
      inputTokens: Number(row.input_tokens),
      outputTokens: Number(row.output_tokens),
      failureCode: row.failure_code,
      deadlineAt: Number(row.deadline_at),
      createdAt: Number(row.created_at),
      updatedAt: Number(row.updated_at),
    };
  }
}
