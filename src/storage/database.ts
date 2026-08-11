import { mkdirSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";

const SCHEMA_VERSION = 26;

export interface PersistenceOutboxRow {
  id: number;
  tableName: string;
  recordKey: string;
  operation: "upsert" | "delete";
  payloadJson: string | null;
  attempts: number;
  lastError: string | null;
  createdAt: number;
  updatedAt: number;
}

export interface ArtifactBackupOutboxRow {
  id: string;
  localPath: string;
  storagePath: string;
  contentType: string;
  sha256: string;
  sizeBytes: number;
  metadataJson: string;
  attempts: number;
  lastError: string | null;
}

export interface WorkerAdmissionCredentialRow {
  identityKind: "device" | "cell" | "browser";
  identityId: string;
  algorithm: "ed25519" | "ecdsa-p256-sha256";
  publicKey: string;
  fingerprint: string;
  status: "active" | "revoked";
  protocolVersion: number;
  createdAt: number;
  updatedAt: number;
  lastSeenAt: number;
  revokedAt: number | null;
  revocationReason: string | null;
}

export interface NodeOwnershipRow {
  identityKind: "device" | "cell";
  identityId: string;
  accountId: string;
  credentialFingerprint: string;
  status: "active" | "revoked";
  generation: number;
  createdAt: number;
  updatedAt: number;
}

export type WorkerAdmissionResult =
  | { state: "enrolled" | "accepted"; credential: WorkerAdmissionCredentialRow }
  | { state: "revoked" | "key_mismatch" | "fingerprint_in_use"; credential: WorkerAdmissionCredentialRow };

export type WorkerCredentialRevocationResult =
  | { state: "revoked" | "already_revoked"; credential: WorkerAdmissionCredentialRow }
  | { state: "not_found" | "fingerprint_mismatch"; credential?: WorkerAdmissionCredentialRow };

export type WorkerCredentialRotationResult =
  | { state: "rotated"; credential: WorkerAdmissionCredentialRow }
  | {
    state: "not_found" | "revoked" | "fingerprint_mismatch" | "fingerprint_in_use";
    credential?: WorkerAdmissionCredentialRow;
  };
export type WorkerCredentialRecoveryResult =
  | { state: "recovered"; credential: WorkerAdmissionCredentialRow; generation: number }
  | { state: "not_found" | "owner_mismatch" | "fingerprint_in_use"; credential?: WorkerAdmissionCredentialRow };

export class MeshDatabase {
  readonly raw: DatabaseSync;
  private transactionDepth = 0;

  constructor(path: string) {
    if (path !== ":memory:") {
      mkdirSync(dirname(resolve(path)), { recursive: true });
    }
    this.raw = new DatabaseSync(path);
    this.raw.exec("PRAGMA foreign_keys = ON");
    this.raw.exec("PRAGMA busy_timeout = 5000");
    if (path !== ":memory:") this.raw.exec("PRAGMA journal_mode = WAL");
    this.migrate();
  }

  close(): void {
    this.raw.close();
  }

  transaction<T>(operation: () => T): T {
    if (this.transactionDepth > 0) return operation();
    this.raw.exec("BEGIN IMMEDIATE");
    this.transactionDepth += 1;
    try {
      const result = operation();
      this.raw.exec("COMMIT");
      return result;
    } catch (error) {
      this.raw.exec("ROLLBACK");
      throw error;
    } finally {
      this.transactionDepth -= 1;
    }
  }

  private migrate(): void {
    this.raw.exec(`
      CREATE TABLE IF NOT EXISTS schema_meta (version INTEGER NOT NULL);
      INSERT INTO schema_meta(version)
      SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_meta);
    `);
    const row = this.raw.prepare("SELECT version FROM schema_meta LIMIT 1").get() as {
      version: number;
    };
    const currentVersion = Number(row.version);
    if (currentVersion < 2) this.removeLegacyProductSchema();

    this.raw.exec(`
      CREATE TABLE IF NOT EXISTS workers (
        id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        capabilities_json TEXT NOT NULL,
        reliability REAL NOT NULL DEFAULT 0.95,
        jobs_completed INTEGER NOT NULL DEFAULT 0,
        last_seen_at INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
        ,deregistered INTEGER NOT NULL DEFAULT 0
        ,identity_kind TEXT
        ,identity_id TEXT
      );

      CREATE INDEX IF NOT EXISTS workers_status_seen
      ON workers(status, last_seen_at);

      CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        model TEXT NOT NULL,
        workload_class TEXT NOT NULL,
        status TEXT NOT NULL,
        worker_id TEXT REFERENCES workers(id),
        deployment_id TEXT,
        model_digest TEXT,
        lease_id TEXT,
        input_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        failure_code TEXT,
        deadline_at INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS jobs_status_created
      ON jobs(status, created_at);

      CREATE INDEX IF NOT EXISTS jobs_worker_status
      ON jobs(worker_id, status);

      CREATE TABLE IF NOT EXISTS idempotency_keys (
        idempotency_key TEXT PRIMARY KEY,
        request_hash TEXT NOT NULL,
        job_id TEXT NOT NULL REFERENCES jobs(id),
        created_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        model TEXT NOT NULL,
        route_json TEXT NOT NULL,
        expires_at INTEGER NOT NULL,
        last_used_at INTEGER NOT NULL,
        version INTEGER NOT NULL DEFAULT 1
      );

      CREATE INDEX IF NOT EXISTS sessions_expiry
      ON sessions(expires_at);

      CREATE TABLE IF NOT EXISTS requested_models (
        id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        revision TEXT,
        context_tokens INTEGER NOT NULL,
        minimum_nodes INTEGER NOT NULL,
        auto_activate INTEGER NOT NULL DEFAULT 1,
        profile_json TEXT,
        profile_error TEXT,
        activation_requested_at INTEGER,
        activation_error TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS persistence_outbox (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        table_name TEXT NOT NULL,
        record_key TEXT NOT NULL,
        operation TEXT NOT NULL CHECK(operation IN ('upsert', 'delete')),
        payload_json TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        UNIQUE(table_name, record_key)
      );

      CREATE INDEX IF NOT EXISTS persistence_outbox_updated
      ON persistence_outbox(updated_at, id);

      CREATE TABLE IF NOT EXISTS inference_conversations (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL UNIQUE,
        model TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS inference_messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES inference_conversations(id) ON DELETE CASCADE,
        job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        status TEXT NOT NULL,
        input_tokens INTEGER,
        output_tokens INTEGER,
        route_class TEXT,
        latency_ms INTEGER,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS inference_messages_conversation_created
      ON inference_messages(conversation_id, created_at);

      CREATE TABLE IF NOT EXISTS activation_events (
        id TEXT PRIMARY KEY,
        model_id TEXT NOT NULL,
        phase TEXT NOT NULL,
        state TEXT NOT NULL,
        message TEXT NOT NULL,
        node_id TEXT,
        process_id TEXT,
        device TEXT,
        details_json TEXT,
        occurred_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS activation_events_model_occurred
      ON activation_events(model_id, occurred_at DESC);

      CREATE TABLE IF NOT EXISTS deployment_states (
        model_id TEXT PRIMARY KEY REFERENCES requested_models(id) ON DELETE CASCADE,
        desired_state TEXT NOT NULL CHECK(desired_state IN ('active', 'inactive')),
        observed_state TEXT NOT NULL CHECK(observed_state IN (
          'inactive', 'waiting_capacity', 'preparing', 'canary', 'active',
          'degraded', 'failed', 'stopping'
        )),
        generation INTEGER NOT NULL DEFAULT 1,
        observed_generation INTEGER NOT NULL DEFAULT 0,
        retry_count INTEGER NOT NULL DEFAULT 0,
        next_retry_at INTEGER,
        last_error TEXT,
        active_operation_id TEXT,
        controller_owner TEXT,
        controller_lease_until INTEGER,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS deployment_states_reconcile
      ON deployment_states(desired_state, observed_state, next_retry_at, controller_lease_until);

      CREATE TABLE IF NOT EXISTS deployment_operations (
        id TEXT PRIMARY KEY,
        model_id TEXT NOT NULL REFERENCES requested_models(id) ON DELETE CASCADE,
        generation INTEGER NOT NULL,
        kind TEXT NOT NULL CHECK(kind IN ('activate', 'deactivate', 'repair', 'replan')),
        status TEXT NOT NULL CHECK(status IN (
          'pending', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted'
        )),
        attempt INTEGER NOT NULL DEFAULT 1,
        idempotency_key TEXT NOT NULL UNIQUE,
        error_code TEXT,
        error_message TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        started_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        finished_at INTEGER
      );

      CREATE INDEX IF NOT EXISTS deployment_operations_model_started
      ON deployment_operations(model_id, started_at DESC);

      CREATE TABLE IF NOT EXISTS route_reservations (
        id TEXT PRIMARY KEY,
        model_id TEXT NOT NULL REFERENCES requested_models(id) ON DELETE CASCADE,
        operation_id TEXT NOT NULL REFERENCES deployment_operations(id) ON DELETE CASCADE,
        generation INTEGER NOT NULL,
        status TEXT NOT NULL CHECK(status IN (
          'prepared', 'committed', 'released', 'expired', 'failed'
        )),
        route_digest TEXT NOT NULL,
        stages_json TEXT NOT NULL,
        canary_json TEXT,
        expires_at INTEGER NOT NULL,
        committed_at INTEGER,
        released_at INTEGER,
        error TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS route_reservations_model_status
      ON route_reservations(model_id, status, expires_at);

      CREATE TABLE IF NOT EXISTS deployment_stage_leases (
        id TEXT PRIMARY KEY,
        reservation_id TEXT NOT NULL REFERENCES route_reservations(id) ON DELETE CASCADE,
        model_id TEXT NOT NULL REFERENCES requested_models(id) ON DELETE CASCADE,
        node_id TEXT NOT NULL,
        stage_index INTEGER NOT NULL,
        memory_mib INTEGER NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('prepared', 'committed', 'released', 'expired')),
        expires_at INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        UNIQUE(reservation_id, node_id, stage_index)
      );

      CREATE INDEX IF NOT EXISTS deployment_stage_leases_node_status
      ON deployment_stage_leases(node_id, status, expires_at);

      CREATE TABLE IF NOT EXISTS artifact_backup_outbox (
        id TEXT PRIMARY KEY,
        local_path TEXT NOT NULL,
        storage_path TEXT NOT NULL,
        content_type TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS assistant_settings (
        id TEXT PRIMARY KEY,
        enabled INTEGER NOT NULL DEFAULT 1,
        model_id TEXT,
        system_prompt TEXT NOT NULL,
        welcome_message TEXT NOT NULL,
        suggestions_json TEXT NOT NULL,
        max_output_tokens INTEGER NOT NULL DEFAULT 512,
        temperature REAL NOT NULL DEFAULT 0.2,
        allow_device_control INTEGER NOT NULL DEFAULT 1,
        updated_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS api_accounts (
        user_id TEXT PRIMARY KEY,
        token_balance INTEGER NOT NULL DEFAULT 0 CHECK(token_balance >= 0),
        usd_micros INTEGER NOT NULL DEFAULT 0 CHECK(usd_micros >= 0),
        lifetime_input_tokens INTEGER NOT NULL DEFAULT 0,
        lifetime_output_tokens INTEGER NOT NULL DEFAULT 0,
        request_count INTEGER NOT NULL DEFAULT 0,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS api_keys (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES api_accounts(user_id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        prefix TEXT NOT NULL UNIQUE,
        secret_hash TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        last_used_at INTEGER,
        revoked_at INTEGER
      );

      CREATE INDEX IF NOT EXISTS api_keys_user_created
      ON api_keys(user_id, created_at DESC);

      CREATE TABLE IF NOT EXISTS api_usage (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES api_accounts(user_id) ON DELETE CASCADE,
        api_key_id TEXT REFERENCES api_keys(id) ON DELETE SET NULL,
        job_id TEXT,
        session_id TEXT,
        model TEXT NOT NULL,
        reserved_tokens INTEGER NOT NULL,
        input_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL CHECK(status IN ('pending', 'completed', 'failed')),
        error_code TEXT,
        created_at INTEGER NOT NULL,
        completed_at INTEGER
      );

      CREATE INDEX IF NOT EXISTS api_usage_user_created
      ON api_usage(user_id, created_at DESC);

      CREATE INDEX IF NOT EXISTS api_usage_user_status
      ON api_usage(user_id, status);

      CREATE UNIQUE INDEX IF NOT EXISTS api_usage_job
      ON api_usage(job_id)
      WHERE job_id IS NOT NULL;

      CREATE TABLE IF NOT EXISTS network_telemetry_history (
        captured_at INTEGER PRIMARY KEY,
        registered_nodes INTEGER NOT NULL,
        connected_nodes INTEGER NOT NULL,
        online_nodes INTEGER NOT NULL,
        browser_nodes INTEGER NOT NULL,
        active_models INTEGER NOT NULL,
        model_replicas INTEGER NOT NULL,
        model_pipelines INTEGER NOT NULL,
        offered_vram_mb INTEGER NOT NULL,
        free_vram_mb INTEGER NOT NULL,
        inflight_jobs INTEGER NOT NULL,
        running_jobs INTEGER NOT NULL,
        completed_jobs INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS network_telemetry_history_captured
      ON network_telemetry_history(captured_at DESC);

      CREATE TABLE IF NOT EXISTS diagnostic_events (
        id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        app_version TEXT NOT NULL,
        platform TEXT NOT NULL,
        arch TEXT NOT NULL,
        level TEXT NOT NULL CHECK(level IN ('info', 'warning', 'error')),
        source TEXT NOT NULL,
        event TEXT NOT NULL,
        message TEXT NOT NULL,
        details TEXT,
        occurred_at INTEGER NOT NULL,
        received_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS diagnostic_events_occurred
      ON diagnostic_events(occurred_at DESC);

      CREATE INDEX IF NOT EXISTS diagnostic_events_source_occurred
      ON diagnostic_events(source_id, occurred_at DESC);

      CREATE TABLE IF NOT EXISTS worker_admission_credentials (
        identity_kind TEXT NOT NULL CHECK(identity_kind IN ('device', 'cell', 'browser')),
        identity_id TEXT NOT NULL,
        algorithm TEXT NOT NULL CHECK(algorithm IN ('ed25519', 'ecdsa-p256-sha256')),
        public_key TEXT NOT NULL,
        fingerprint TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL CHECK(status IN ('active', 'revoked')),
        protocol_version INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        last_seen_at INTEGER NOT NULL,
        revoked_at INTEGER,
        revocation_reason TEXT,
        PRIMARY KEY(identity_kind, identity_id)
      );

      CREATE INDEX IF NOT EXISTS worker_admission_credentials_status
      ON worker_admission_credentials(status, last_seen_at);

      CREATE TABLE IF NOT EXISTS node_enrollments (
        id TEXT PRIMARY KEY,
        token_hash TEXT NOT NULL UNIQUE,
        nonce_hash TEXT NOT NULL,
        account_id TEXT NOT NULL,
        created_by_kind TEXT NOT NULL,
        created_by_id TEXT NOT NULL,
        expires_at INTEGER NOT NULL,
        confirmed_at INTEGER,
        confirmed_by TEXT,
        consumed_at INTEGER,
        identity_kind TEXT,
        identity_id TEXT,
        public_key_fingerprint TEXT,
        created_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS node_enrollments_account_created
      ON node_enrollments(account_id, created_at DESC);

      CREATE TABLE IF NOT EXISTS node_enrollment_events (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        enrollment_id TEXT NOT NULL REFERENCES node_enrollments(id) ON DELETE CASCADE,
        actor_kind TEXT NOT NULL,
        actor_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        details_json TEXT NOT NULL,
        occurred_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS node_enrollment_events_enrollment
      ON node_enrollment_events(enrollment_id, sequence);

      CREATE TABLE IF NOT EXISTS node_ownership (
        identity_kind TEXT NOT NULL CHECK(identity_kind IN ('device', 'cell')),
        identity_id TEXT NOT NULL,
        account_id TEXT NOT NULL,
        credential_fingerprint TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('active', 'revoked')),
        generation INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        PRIMARY KEY(identity_kind, identity_id),
        UNIQUE(credential_fingerprint)
      );

      CREATE TABLE IF NOT EXISTS economic_accounts (
        id TEXT PRIMARY KEY,
        owner_kind TEXT NOT NULL CHECK(owner_kind IN ('account', 'node', 'platform')),
        owner_id TEXT NOT NULL,
        asset TEXT NOT NULL CHECK(asset = 'MYC_MICROCREDITS'),
        created_at INTEGER NOT NULL,
        UNIQUE(owner_kind, owner_id, asset)
      );

      CREATE TABLE IF NOT EXISTS pricing_policies (
        id TEXT PRIMARY KEY,
        model_id TEXT NOT NULL,
        route_class TEXT NOT NULL,
        version INTEGER NOT NULL CHECK(version > 0),
        input_microunits_per_token INTEGER NOT NULL CHECK(input_microunits_per_token >= 0),
        output_microunits_per_token INTEGER NOT NULL CHECK(output_microunits_per_token >= 0),
        platform_fee_bps INTEGER NOT NULL CHECK(platform_fee_bps BETWEEN 0 AND 10000),
        effective_at INTEGER NOT NULL,
        retired_at INTEGER,
        UNIQUE(model_id, route_class, version)
      );

      CREATE TABLE IF NOT EXISTS economic_settlements (
        id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL UNIQUE,
        execution_receipt_id TEXT NOT NULL UNIQUE,
        contribution_evidence_id TEXT NOT NULL UNIQUE REFERENCES economic_contribution_evidence(id),
        pricing_policy_id TEXT NOT NULL REFERENCES pricing_policies(id),
        payer_account_id TEXT NOT NULL,
        input_tokens INTEGER NOT NULL CHECK(input_tokens >= 0),
        output_tokens INTEGER NOT NULL CHECK(output_tokens >= 0),
        gross_microunits INTEGER NOT NULL CHECK(gross_microunits >= 0),
        provider_microunits INTEGER NOT NULL CHECK(provider_microunits >= 0),
        platform_microunits INTEGER NOT NULL CHECK(platform_microunits >= 0),
        request_digest TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        CHECK(gross_microunits = provider_microunits + platform_microunits)
      );

      CREATE TABLE IF NOT EXISTS economic_ledger_entries (
        id TEXT PRIMARY KEY,
        settlement_id TEXT NOT NULL REFERENCES economic_settlements(id),
        account_id TEXT NOT NULL REFERENCES economic_accounts(id),
        category TEXT NOT NULL CHECK(category IN ('usage', 'work', 'platform_fee')),
        amount_microunits INTEGER NOT NULL CHECK(amount_microunits != 0),
        created_at INTEGER NOT NULL,
        UNIQUE(settlement_id, account_id, category)
      );

      CREATE INDEX IF NOT EXISTS economic_ledger_account_created
      ON economic_ledger_entries(account_id, created_at);

      CREATE TABLE IF NOT EXISTS economic_settlement_receipts (
        settlement_id TEXT PRIMARY KEY REFERENCES economic_settlements(id),
        receipt_id TEXT NOT NULL UNIQUE,
        key_id TEXT NOT NULL,
        receipt_json TEXT NOT NULL,
        created_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS execution_receipts (
        job_id TEXT PRIMARY KEY REFERENCES jobs(id),
        receipt_id TEXT NOT NULL UNIQUE,
        key_id TEXT NOT NULL,
        receipt_json TEXT NOT NULL,
        created_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS execution_topologies (
        job_id TEXT PRIMARY KEY REFERENCES execution_receipts(job_id),
        receipt_id TEXT NOT NULL UNIQUE REFERENCES execution_receipts(receipt_id),
        trace_digest TEXT NOT NULL UNIQUE,
        topology_json TEXT NOT NULL,
        created_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS model_certifications (
        certification_id TEXT PRIMARY KEY,
        model_family TEXT NOT NULL,
        decision TEXT NOT NULL CHECK(decision IN ('certified', 'revoked')),
        topology_kind TEXT NOT NULL,
        platform TEXT NOT NULL,
        backend TEXT NOT NULL,
        certification_json TEXT NOT NULL,
        reviewed_at INTEGER NOT NULL,
        expires_at INTEGER NOT NULL,
        created_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS model_certifications_family_reviewed
      ON model_certifications(model_family, reviewed_at DESC);

      CREATE TABLE IF NOT EXISTS economic_contribution_evidence (
        id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL UNIQUE,
        execution_receipt_id TEXT NOT NULL UNIQUE,
        trace_digest TEXT NOT NULL UNIQUE,
        evidence_json TEXT NOT NULL,
        created_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS economic_node_reputation (
        node_id TEXT PRIMARY KEY,
        verified_jobs INTEGER NOT NULL CHECK(verified_jobs >= 0),
        verified_stage_count INTEGER NOT NULL CHECK(verified_stage_count >= 0),
        verified_physical_boundaries INTEGER NOT NULL CHECK(verified_physical_boundaries >= 0),
        earned_microunits INTEGER NOT NULL CHECK(earned_microunits >= 0),
        last_evidence_id TEXT NOT NULL REFERENCES economic_contribution_evidence(id),
        updated_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS node_ownership_account
      ON node_ownership(account_id, status, updated_at DESC);

      CREATE TABLE IF NOT EXISTS node_ownership_transfers (
        id TEXT PRIMARY KEY,
        identity_id TEXT NOT NULL,
        source_account_id TEXT NOT NULL,
        target_account_id TEXT NOT NULL,
        token_hash TEXT NOT NULL UNIQUE,
        expected_generation INTEGER NOT NULL CHECK(expected_generation > 0),
        expires_at INTEGER NOT NULL,
        accepted_at INTEGER,
        created_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS node_ownership_transfers_target
      ON node_ownership_transfers(target_account_id, expires_at DESC);

      CREATE TABLE IF NOT EXISTS node_identity_events (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        identity_kind TEXT NOT NULL CHECK(identity_kind IN ('device', 'cell')),
        identity_id TEXT NOT NULL,
        generation INTEGER NOT NULL CHECK(generation > 0),
        actor_kind TEXT NOT NULL CHECK(actor_kind IN ('account', 'node', 'operator')),
        actor_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        details_json TEXT NOT NULL,
        previous_event_digest TEXT,
        event_digest TEXT NOT NULL UNIQUE,
        occurred_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS node_identity_events_identity_sequence
      ON node_identity_events(identity_kind, identity_id, sequence);

      CREATE TABLE IF NOT EXISTS node_commands (
        id TEXT PRIMARY KEY,
        node_id TEXT NOT NULL,
        generation INTEGER NOT NULL CHECK(generation > 0),
        nonce TEXT NOT NULL,
        command_json TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('queued', 'delivered', 'applied', 'rejected', 'expired')),
        issued_at INTEGER NOT NULL,
        expires_at INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        delivered_at INTEGER,
        completed_at INTEGER,
        UNIQUE(node_id, nonce)
      );

      CREATE INDEX IF NOT EXISTS node_commands_pending
      ON node_commands(node_id, generation, state, created_at);

      CREATE TABLE IF NOT EXISTS node_command_results (
        id TEXT PRIMARY KEY,
        command_id TEXT NOT NULL UNIQUE REFERENCES node_commands(id) ON DELETE CASCADE,
        node_id TEXT NOT NULL,
        generation INTEGER NOT NULL,
        result_digest TEXT NOT NULL,
        result_json TEXT NOT NULL,
        observed_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS node_control_events (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        node_id TEXT NOT NULL,
        generation INTEGER NOT NULL,
        actor_json TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload_digest TEXT NOT NULL,
        previous_event_digest TEXT,
        event_digest TEXT NOT NULL UNIQUE,
        occurred_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS node_control_events_node_sequence
      ON node_control_events(node_id, sequence);

      CREATE TABLE IF NOT EXISTS node_desired_states (
        node_id TEXT PRIMARY KEY,
        generation INTEGER NOT NULL CHECK(generation > 0),
        desired_json TEXT NOT NULL,
        updated_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS node_snapshots (
        node_id TEXT PRIMARY KEY,
        generation INTEGER NOT NULL CHECK(generation > 0),
        cursor_sequence INTEGER NOT NULL CHECK(cursor_sequence >= 0),
        snapshot_json TEXT NOT NULL,
        observed_at INTEGER NOT NULL,
        received_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS node_snapshot_history (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        node_id TEXT NOT NULL,
        generation INTEGER NOT NULL,
        cursor_sequence INTEGER NOT NULL,
        snapshot_digest TEXT NOT NULL,
        snapshot_json TEXT NOT NULL,
        received_at INTEGER NOT NULL,
        UNIQUE(node_id, snapshot_digest)
      );

      CREATE INDEX IF NOT EXISTS node_snapshot_history_node_sequence
      ON node_snapshot_history(node_id, sequence);

    `);
    if (currentVersion >= 2 && currentVersion < 3) {
      const columns = this.raw.prepare("PRAGMA table_info(workers)").all() as Array<{
        name: string;
      }>;
      if (!columns.some((column) => column.name === "deregistered")) {
        this.raw.exec("ALTER TABLE workers ADD COLUMN deregistered INTEGER NOT NULL DEFAULT 0");
      }
      this.raw.exec("UPDATE workers SET deregistered = 1");
    }
    if (currentVersion >= 4 && currentVersion < 5) {
      const columns = this.raw.prepare("PRAGMA table_info(requested_models)").all() as Array<{
        name: string;
      }>;
      if (!columns.some((column) => column.name === "activation_error")) {
        this.raw.exec("ALTER TABLE requested_models ADD COLUMN activation_error TEXT");
      }
    }
    if (currentVersion < 6) this.migrateWorkerIdentities();
    if (currentVersion >= 18 && currentVersion < 19) {
      const columns = this.raw.prepare("PRAGMA table_info(economic_settlements)").all() as Array<{ name: string }>;
      if (!columns.some((column) => column.name === "contribution_evidence_id")) {
        this.raw.exec("ALTER TABLE economic_settlements ADD COLUMN contribution_evidence_id TEXT REFERENCES economic_contribution_evidence(id)");
      }
      this.raw.exec(`
        CREATE UNIQUE INDEX IF NOT EXISTS economic_settlements_contribution_evidence_unique
        ON economic_settlements(contribution_evidence_id)
        WHERE contribution_evidence_id IS NOT NULL
      `);
    }
    this.raw.exec(`
      CREATE UNIQUE INDEX IF NOT EXISTS workers_identity_unique
      ON workers(identity_kind, identity_id)
      WHERE identity_kind IS NOT NULL AND identity_id IS NOT NULL;
      CREATE UNIQUE INDEX IF NOT EXISTS economic_settlements_contribution_evidence_unique
      ON economic_settlements(contribution_evidence_id)
      WHERE contribution_evidence_id IS NOT NULL;
    `);
    this.raw.prepare("UPDATE schema_meta SET version = ?").run(SCHEMA_VERSION);
  }

  admitWorkerCredential(input: {
    identityKind: WorkerAdmissionCredentialRow["identityKind"];
    identityId: string;
    algorithm: WorkerAdmissionCredentialRow["algorithm"];
    publicKey: string;
    fingerprint: string;
    protocolVersion: number;
  }): WorkerAdmissionResult {
    return this.transaction(() => {
      const existing = this.readWorkerAdmissionCredential(
        this.raw.prepare(
          `SELECT identity_kind, identity_id, algorithm, public_key, fingerprint,
                  status, protocol_version, created_at, updated_at, last_seen_at,
                  revoked_at, revocation_reason
           FROM worker_admission_credentials
           WHERE identity_kind = ? AND identity_id = ?`,
        ).get(input.identityKind, input.identityId),
      );
      if (existing) {
        if (existing.status === "revoked") {
          return { state: "revoked", credential: existing };
        }
        if (
          existing.algorithm !== input.algorithm
          || existing.publicKey !== input.publicKey
          || existing.fingerprint !== input.fingerprint
        ) {
          return { state: "key_mismatch", credential: existing };
        }
        const now = Date.now();
        this.raw.prepare(
          `UPDATE worker_admission_credentials
           SET protocol_version = ?, updated_at = ?, last_seen_at = ?
           WHERE identity_kind = ? AND identity_id = ?`,
        ).run(
          input.protocolVersion,
          now,
          now,
          input.identityKind,
          input.identityId,
        );
        return {
          state: "accepted",
          credential: {
            ...existing,
            protocolVersion: input.protocolVersion,
            updatedAt: now,
            lastSeenAt: now,
          },
        };
      }

      const reused = this.readWorkerAdmissionCredential(
        this.raw.prepare(
          `SELECT identity_kind, identity_id, algorithm, public_key, fingerprint,
                  status, protocol_version, created_at, updated_at, last_seen_at,
                  revoked_at, revocation_reason
           FROM worker_admission_credentials
           WHERE fingerprint = ?`,
        ).get(input.fingerprint),
      );
      if (reused) {
        return { state: "fingerprint_in_use", credential: reused };
      }

      const now = Date.now();
      this.raw.prepare(
        `INSERT INTO worker_admission_credentials(
           identity_kind, identity_id, algorithm, public_key, fingerprint,
           status, protocol_version, created_at, updated_at, last_seen_at
         ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)`,
      ).run(
        input.identityKind,
        input.identityId,
        input.algorithm,
        input.publicKey,
        input.fingerprint,
        input.protocolVersion,
        now,
        now,
        now,
      );
      return {
        state: "enrolled",
        credential: {
          ...input,
          status: "active",
          createdAt: now,
          updatedAt: now,
          lastSeenAt: now,
          revokedAt: null,
          revocationReason: null,
        },
      };
    });
  }

  listWorkerAdmissionCredentials(limit = 1_000): WorkerAdmissionCredentialRow[] {
    return this.raw.prepare(
      `SELECT identity_kind, identity_id, algorithm, public_key, fingerprint,
              status, protocol_version, created_at, updated_at, last_seen_at,
              revoked_at, revocation_reason
       FROM worker_admission_credentials
       ORDER BY last_seen_at DESC, identity_kind ASC, identity_id ASC
       LIMIT ?`,
    ).all(Math.max(1, Math.min(10_000, Math.trunc(limit))))
      .flatMap((row) => {
        const credential = this.readWorkerAdmissionCredential(row);
        return credential ? [credential] : [];
      });
  }

  getWorkerAdmissionCredential(
    identityKind: WorkerAdmissionCredentialRow["identityKind"],
    identityId: string,
  ): WorkerAdmissionCredentialRow | null {
    return this.readWorkerAdmissionCredential(
      this.raw.prepare(
        `SELECT identity_kind, identity_id, algorithm, public_key, fingerprint,
                status, protocol_version, created_at, updated_at, last_seen_at,
                revoked_at, revocation_reason
         FROM worker_admission_credentials
         WHERE identity_kind = ? AND identity_id = ?`,
      ).get(identityKind, identityId),
    );
  }

  getNodeOwnership(
    identityKind: NodeOwnershipRow["identityKind"],
    identityId: string,
  ): NodeOwnershipRow | null {
    const row = this.raw.prepare(
      `SELECT identity_kind, identity_id, account_id, credential_fingerprint,
              status, generation, created_at, updated_at
       FROM node_ownership WHERE identity_kind = ? AND identity_id = ?`,
    ).get(identityKind, identityId) as Record<string, unknown> | undefined;
    return row ? {
      identityKind: String(row.identity_kind) as NodeOwnershipRow["identityKind"],
      identityId: String(row.identity_id),
      accountId: String(row.account_id),
      credentialFingerprint: String(row.credential_fingerprint),
      status: String(row.status) as NodeOwnershipRow["status"],
      generation: Number(row.generation),
      createdAt: Number(row.created_at),
      updatedAt: Number(row.updated_at),
    } : null;
  }

  getNodeOwnershipByCredentialFingerprint(credentialFingerprint: string): NodeOwnershipRow | null {
    const row = this.raw.prepare(
      `SELECT identity_kind, identity_id, account_id, credential_fingerprint,
              status, generation, created_at, updated_at
       FROM node_ownership WHERE credential_fingerprint = ?`,
    ).get(credentialFingerprint) as Record<string, unknown> | undefined;
    return row ? {
      identityKind: String(row.identity_kind) as NodeOwnershipRow["identityKind"],
      identityId: String(row.identity_id),
      accountId: String(row.account_id),
      credentialFingerprint: String(row.credential_fingerprint),
      status: String(row.status) as NodeOwnershipRow["status"],
      generation: Number(row.generation),
      createdAt: Number(row.created_at),
      updatedAt: Number(row.updated_at),
    } : null;
  }

  appendNodeIdentityEvent(input: {
    identityKind: NodeOwnershipRow["identityKind"];
    identityId: string;
    generation: number;
    actorKind: "account" | "node" | "operator";
    actorId: string;
    eventType: string;
    details: Record<string, unknown>;
  }): string {
    const previous = this.raw.prepare(
      `SELECT event_digest FROM node_identity_events
       WHERE identity_kind = ? AND identity_id = ? ORDER BY sequence DESC LIMIT 1`,
    ).get(input.identityKind, input.identityId) as { event_digest: string } | undefined;
    const occurredAt = Date.now();
    const detailsJson = JSON.stringify(input.details);
    const eventDigest = nodeIdentityEventDigest({ identityKind: input.identityKind, identityId: input.identityId,
      generation: input.generation, actorKind: input.actorKind, actorId: input.actorId, eventType: input.eventType,
      detailsJson, previousEventDigest: previous?.event_digest ?? null, occurredAt });
    this.raw.prepare(
      `INSERT INTO node_identity_events(
         identity_kind, identity_id, generation, actor_kind, actor_id,
         event_type, details_json, previous_event_digest, event_digest, occurred_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(input.identityKind, input.identityId, input.generation, input.actorKind, input.actorId,
      input.eventType, detailsJson, previous?.event_digest ?? null, eventDigest, occurredAt);
    return eventDigest;
  }

  nodeIdentityEvents(identityKind: NodeOwnershipRow["identityKind"], identityId: string): Array<{
    sequence: number; generation: number; actorKind: string; actorId: string; eventType: string;
    details: Record<string, unknown>; previousEventDigest: string | null; eventDigest: string; occurredAt: string;
  }> {
    const rows = this.raw.prepare(
      `SELECT sequence, identity_kind, identity_id, generation, actor_kind, actor_id,
              event_type, details_json, previous_event_digest, event_digest, occurred_at
       FROM node_identity_events WHERE identity_kind = ? AND identity_id = ? ORDER BY sequence`,
    ).all(identityKind, identityId) as Array<Record<string, unknown>>;
    let previous: string | null = null;
    return rows.map((row) => {
      const previousEventDigest = row.previous_event_digest === null ? null : String(row.previous_event_digest);
      if (previousEventDigest !== previous) throw new Error("node_identity_event_chain_broken");
      const expected = nodeIdentityEventDigest({ identityKind: String(row.identity_kind), identityId: String(row.identity_id),
        generation: Number(row.generation), actorKind: String(row.actor_kind), actorId: String(row.actor_id),
        eventType: String(row.event_type), detailsJson: String(row.details_json), previousEventDigest,
        occurredAt: Number(row.occurred_at) });
      if (expected !== String(row.event_digest)) throw new Error("node_identity_event_digest_mismatch");
      previous = expected;
      return { sequence: Number(row.sequence), generation: Number(row.generation), actorKind: String(row.actor_kind),
        actorId: String(row.actor_id), eventType: String(row.event_type),
        details: JSON.parse(String(row.details_json)) as Record<string, unknown>, previousEventDigest,
        eventDigest: expected, occurredAt: new Date(Number(row.occurred_at)).toISOString() };
    });
  }

  revokeWorkerAdmissionCredential(input: {
    identityKind: WorkerAdmissionCredentialRow["identityKind"];
    identityId: string;
    expectedFingerprint: string;
    reason: string;
    actor?: { kind: "account" | "operator"; id: string };
  }): WorkerCredentialRevocationResult {
    return this.transaction(() => {
      const existing = this.getWorkerAdmissionCredential(input.identityKind, input.identityId);
      if (!existing) return { state: "not_found" };
      if (existing.fingerprint !== input.expectedFingerprint) {
        return { state: "fingerprint_mismatch", credential: existing };
      }
      if (existing.status === "revoked") {
        return { state: "already_revoked", credential: existing };
      }
      const now = Date.now();
      this.raw.prepare(
        `UPDATE worker_admission_credentials
         SET status = 'revoked', updated_at = ?, revoked_at = ?, revocation_reason = ?
         WHERE identity_kind = ? AND identity_id = ? AND fingerprint = ? AND status = 'active'`,
      ).run(
        now,
        now,
        input.reason,
        input.identityKind,
        input.identityId,
        input.expectedFingerprint,
      );
      this.raw.prepare(
        `UPDATE node_ownership SET status = 'revoked', generation = generation + 1,
             updated_at = ?
         WHERE identity_kind = ? AND identity_id = ? AND credential_fingerprint = ?`,
      ).run(
        now,
        input.identityKind === "browser" ? "cell" : input.identityKind,
        input.identityId,
        input.expectedFingerprint,
      );
      const ownership = this.getNodeOwnership(input.identityKind === "browser" ? "cell" : input.identityKind, input.identityId);
      if (ownership) this.appendNodeIdentityEvent({ identityKind: ownership.identityKind, identityId: ownership.identityId,
        generation: ownership.generation, actorKind: input.actor?.kind ?? "operator", actorId: input.actor?.id ?? "coordinator", eventType: "credential.revoked",
        details: { credentialFingerprint: existing.fingerprint, reason: input.reason } });
      return {
        state: "revoked",
        credential: {
          ...existing,
          status: "revoked",
          updatedAt: now,
          revokedAt: now,
          revocationReason: input.reason,
        },
      };
    });
  }

  rotateWorkerAdmissionCredential(input: {
    identityKind: WorkerAdmissionCredentialRow["identityKind"];
    identityId: string;
    expectedFingerprint: string;
    algorithm: WorkerAdmissionCredentialRow["algorithm"];
    publicKey: string;
    fingerprint: string;
    protocolVersion: number;
  }): WorkerCredentialRotationResult {
    return this.transaction(() => {
      const existing = this.getWorkerAdmissionCredential(input.identityKind, input.identityId);
      if (!existing) return { state: "not_found" };
      if (existing.status === "revoked") return { state: "revoked", credential: existing };
      if (existing.fingerprint !== input.expectedFingerprint) {
        return { state: "fingerprint_mismatch", credential: existing };
      }
      const reused = this.readWorkerAdmissionCredential(
        this.raw.prepare(
          `SELECT identity_kind, identity_id, algorithm, public_key, fingerprint,
                  status, protocol_version, created_at, updated_at, last_seen_at,
                  revoked_at, revocation_reason
           FROM worker_admission_credentials
           WHERE fingerprint = ?`,
        ).get(input.fingerprint),
      );
      if (reused && (
        reused.identityKind !== input.identityKind
        || reused.identityId !== input.identityId
      )) {
        return { state: "fingerprint_in_use", credential: reused };
      }
      const now = Date.now();
      this.raw.prepare(
        `UPDATE worker_admission_credentials
         SET algorithm = ?, public_key = ?, fingerprint = ?, protocol_version = ?,
             updated_at = ?, last_seen_at = ?, revoked_at = NULL,
             revocation_reason = NULL
         WHERE identity_kind = ? AND identity_id = ? AND fingerprint = ? AND status = 'active'`,
      ).run(
        input.algorithm,
        input.publicKey,
        input.fingerprint,
        input.protocolVersion,
        now,
        now,
        input.identityKind,
        input.identityId,
        input.expectedFingerprint,
      );
      this.raw.prepare(
        `UPDATE node_ownership
         SET credential_fingerprint = ?, generation = generation + 1,
             status = 'active', updated_at = ?
         WHERE identity_kind = ? AND identity_id = ?
           AND credential_fingerprint = ?`,
      ).run(
        input.fingerprint,
        now,
        input.identityKind === "browser" ? "cell" : input.identityKind,
        input.identityId,
        input.expectedFingerprint,
      );
      const ownership = input.identityKind === "browser" ? null : this.getNodeOwnership(input.identityKind, input.identityId);
      if (ownership) this.appendNodeIdentityEvent({ identityKind: ownership.identityKind, identityId: ownership.identityId,
        generation: ownership.generation, actorKind: "node", actorId: input.identityId, eventType: "credential.rotated",
        details: { previousFingerprint: input.expectedFingerprint, credentialFingerprint: input.fingerprint } });
      return {
        state: "rotated",
        credential: {
          ...existing,
          algorithm: input.algorithm,
          publicKey: input.publicKey,
          fingerprint: input.fingerprint,
          protocolVersion: input.protocolVersion,
          updatedAt: now,
          lastSeenAt: now,
          revokedAt: null,
          revocationReason: null,
        },
      };
    });
  }

  recoverWorkerAdmissionCredential(input: {
    identityKind: "device";
    identityId: string;
    accountId: string;
    algorithm: WorkerAdmissionCredentialRow["algorithm"];
    publicKey: string;
    fingerprint: string;
    protocolVersion: number;
  }): WorkerCredentialRecoveryResult {
    return this.transaction(() => {
      const ownership = this.getNodeOwnership(input.identityKind, input.identityId);
      const existing = this.getWorkerAdmissionCredential(input.identityKind, input.identityId);
      if (!ownership || !existing) return { state: "not_found" };
      if (ownership.accountId !== input.accountId) return { state: "owner_mismatch", credential: existing };
      const reused = this.readWorkerAdmissionCredential(this.raw.prepare(
        `SELECT identity_kind, identity_id, algorithm, public_key, fingerprint,
                status, protocol_version, created_at, updated_at, last_seen_at,
                revoked_at, revocation_reason
         FROM worker_admission_credentials WHERE fingerprint = ?`,
      ).get(input.fingerprint));
      if (reused && (reused.identityKind !== input.identityKind || reused.identityId !== input.identityId)) {
        return { state: "fingerprint_in_use", credential: reused };
      }
      const now = Date.now();
      this.raw.prepare(
        `UPDATE worker_admission_credentials SET
           algorithm = ?, public_key = ?, fingerprint = ?, status = 'active',
           protocol_version = ?, updated_at = ?, last_seen_at = ?,
           revoked_at = NULL, revocation_reason = NULL
         WHERE identity_kind = ? AND identity_id = ?`,
      ).run(input.algorithm, input.publicKey, input.fingerprint, input.protocolVersion, now, now, input.identityKind, input.identityId);
      this.raw.prepare(
        `UPDATE node_ownership SET credential_fingerprint = ?, status = 'active',
             generation = generation + 1, updated_at = ?
         WHERE identity_kind = ? AND identity_id = ? AND account_id = ?`,
      ).run(input.fingerprint, now, input.identityKind, input.identityId, input.accountId);
      const recovered = this.getWorkerAdmissionCredential(input.identityKind, input.identityId);
      const updatedOwnership = this.getNodeOwnership(input.identityKind, input.identityId);
      if (!recovered || !updatedOwnership) throw new Error("worker_recovery_persistence_failed");
      this.appendNodeIdentityEvent({ identityKind: input.identityKind, identityId: input.identityId,
        generation: updatedOwnership.generation, actorKind: "account", actorId: input.accountId,
        eventType: "credential.recovered", details: { previousFingerprint: existing.fingerprint, credentialFingerprint: input.fingerprint } });
      return { state: "recovered", credential: recovered, generation: updatedOwnership.generation };
    });
  }

  private readWorkerAdmissionCredential(value: unknown): WorkerAdmissionCredentialRow | null {
    if (!value || typeof value !== "object") return null;
    const row = value as {
      identity_kind: WorkerAdmissionCredentialRow["identityKind"];
      identity_id: string;
      algorithm: WorkerAdmissionCredentialRow["algorithm"];
      public_key: string;
      fingerprint: string;
      status: WorkerAdmissionCredentialRow["status"];
      protocol_version: number;
      created_at: number;
      updated_at: number;
      last_seen_at: number;
      revoked_at: number | null;
      revocation_reason: string | null;
    };
    return {
      identityKind: row.identity_kind,
      identityId: row.identity_id,
      algorithm: row.algorithm,
      publicKey: row.public_key,
      fingerprint: row.fingerprint,
      status: row.status,
      protocolVersion: Number(row.protocol_version),
      createdAt: Number(row.created_at),
      updatedAt: Number(row.updated_at),
      lastSeenAt: Number(row.last_seen_at),
      revokedAt: row.revoked_at === null ? null : Number(row.revoked_at),
      revocationReason: row.revocation_reason,
    };
  }

  enqueueRemoteChange(
    tableName: string,
    recordKey: string,
    operation: "upsert" | "delete",
    payload: Record<string, unknown> | null,
  ): void {
    const now = Date.now();
    this.raw.prepare(
      `INSERT INTO persistence_outbox(
         table_name, record_key, operation, payload_json, created_at, updated_at
       ) VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT(table_name, record_key) DO UPDATE SET
         operation = excluded.operation,
         payload_json = excluded.payload_json,
         attempts = 0,
         last_error = NULL,
         updated_at = excluded.updated_at`,
    ).run(
      tableName,
      recordKey,
      operation,
      payload ? JSON.stringify(payload) : null,
      now,
      now,
    );
  }

  listPendingRemoteChanges(limit = 100): PersistenceOutboxRow[] {
    const rows = this.raw.prepare(
      `SELECT id, table_name, record_key, operation, payload_json, attempts,
              last_error, created_at, updated_at
       FROM persistence_outbox
       ORDER BY updated_at, id
       LIMIT ?`,
    ).all(limit) as unknown as Array<{
      id: number;
      table_name: string;
      record_key: string;
      operation: "upsert" | "delete";
      payload_json: string | null;
      attempts: number;
      last_error: string | null;
      created_at: number;
      updated_at: number;
    }>;
    return rows.map((row) => ({
      id: Number(row.id),
      tableName: row.table_name,
      recordKey: row.record_key,
      operation: row.operation,
      payloadJson: row.payload_json,
      attempts: Number(row.attempts),
      lastError: row.last_error,
      createdAt: Number(row.created_at),
      updatedAt: Number(row.updated_at),
    }));
  }

  markRemoteChangeSynced(id: number): void {
    this.raw.prepare("DELETE FROM persistence_outbox WHERE id = ?").run(id);
  }

  markRemoteChangeFailed(id: number, error: string): void {
    this.raw.prepare(
      `UPDATE persistence_outbox
       SET attempts = attempts + 1, last_error = ?, updated_at = ?
       WHERE id = ?`,
    ).run(error.slice(0, 2_000), Date.now(), id);
  }

  pendingRemoteChangeCount(): number {
    const row = this.raw.prepare(
      "SELECT COUNT(*) AS count FROM persistence_outbox",
    ).get() as { count: number };
    return Number(row.count);
  }

  enqueueArtifactBackup(input: {
    id: string;
    localPath: string;
    storagePath: string;
    contentType: string;
    sha256: string;
    sizeBytes: number;
    metadata: Record<string, unknown>;
  }): void {
    const now = Date.now();
    this.raw.prepare(
      `INSERT INTO artifact_backup_outbox(
         id, local_path, storage_path, content_type, sha256, size_bytes,
         metadata_json, created_at, updated_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         local_path=excluded.local_path, storage_path=excluded.storage_path,
         content_type=excluded.content_type, sha256=excluded.sha256,
         size_bytes=excluded.size_bytes, metadata_json=excluded.metadata_json,
         attempts=0, last_error=NULL, updated_at=excluded.updated_at`,
    ).run(
      input.id,
      input.localPath,
      input.storagePath,
      input.contentType,
      input.sha256,
      input.sizeBytes,
      JSON.stringify(input.metadata),
      now,
      now,
    );
  }

  listPendingArtifactBackups(limit = 10): ArtifactBackupOutboxRow[] {
    const rows = this.raw.prepare(
      `SELECT id, local_path, storage_path, content_type, sha256, size_bytes,
              metadata_json, attempts, last_error
       FROM artifact_backup_outbox
       ORDER BY updated_at, id
       LIMIT ?`,
    ).all(limit) as unknown as Array<{
      id: string;
      local_path: string;
      storage_path: string;
      content_type: string;
      sha256: string;
      size_bytes: number;
      metadata_json: string;
      attempts: number;
      last_error: string | null;
    }>;
    return rows.map((row) => ({
      id: row.id,
      localPath: row.local_path,
      storagePath: row.storage_path,
      contentType: row.content_type,
      sha256: row.sha256,
      sizeBytes: Number(row.size_bytes),
      metadataJson: row.metadata_json,
      attempts: Number(row.attempts),
      lastError: row.last_error,
    }));
  }

  markArtifactBackupSynced(id: string): void {
    this.raw.prepare("DELETE FROM artifact_backup_outbox WHERE id = ?").run(id);
  }

  markArtifactBackupFailed(id: string, error: string): void {
    this.raw.prepare(
      `UPDATE artifact_backup_outbox
       SET attempts=attempts+1, last_error=?, updated_at=?
       WHERE id=?`,
    ).run(error.slice(0, 2_000), Date.now(), id);
  }

  pendingArtifactBackupCount(): number {
    const row = this.raw.prepare(
      "SELECT COUNT(*) AS count FROM artifact_backup_outbox",
    ).get() as { count: number };
    return Number(row.count);
  }

  private migrateWorkerIdentities(): void {
    const columns = this.raw.prepare("PRAGMA table_info(workers)").all() as Array<{
      name: string;
    }>;
    if (!columns.some((column) => column.name === "identity_kind")) {
      this.raw.exec("ALTER TABLE workers ADD COLUMN identity_kind TEXT");
    }
    if (!columns.some((column) => column.name === "identity_id")) {
      this.raw.exec("ALTER TABLE workers ADD COLUMN identity_id TEXT");
    }

    const rows = this.raw.prepare(
      `SELECT id, status, capabilities_json, deregistered, last_seen_at,
              identity_kind, identity_id
       FROM workers`,
    ).all() as Array<{
      id: string;
      status: string;
      capabilities_json: string;
      deregistered: number;
      last_seen_at: number;
      identity_kind: string | null;
      identity_id: string | null;
    }>;
    const grouped = new Map<string, Array<{
      row: (typeof rows)[number];
      kind: "device" | "cell";
      identityId: string;
    }>>();
    for (const row of rows) {
      let kind: "device" | "cell" = row.identity_kind === "cell" ? "cell" : "device";
      let identityId = row.identity_id;
      if (!identityId) {
        try {
          const capabilities = JSON.parse(row.capabilities_json) as {
            distributedExecutor?: { nodeId?: unknown };
          };
          const nodeId = capabilities.distributedExecutor?.nodeId;
          if (typeof nodeId === "string" && /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(nodeId)) {
            identityId = nodeId;
            kind = "device";
          }
        } catch {
          // Invalid legacy capability JSON stays anonymous instead of blocking startup.
        }
      }
      if (!identityId || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(identityId)) continue;
      const key = `${kind}:${identityId}`;
      const entries = grouped.get(key) ?? [];
      entries.push({ row, kind, identityId });
      grouped.set(key, entries);
    }

    for (const entries of grouped.values()) {
      entries.sort((left, right) => {
        const activeDifference = workerMigrationPriority(right.row) - workerMigrationPriority(left.row);
        return activeDifference || right.row.last_seen_at - left.row.last_seen_at;
      });
      const keeper = entries[0]!;
      for (const duplicate of entries.slice(1)) {
        this.raw.prepare(
          `UPDATE workers
           SET deregistered = 1, status = 'offline', identity_kind = NULL, identity_id = NULL,
               updated_at = ?
           WHERE id = ?`,
        ).run(Date.now(), duplicate.row.id);
      }
      this.raw.prepare(
        `UPDATE workers
         SET identity_kind = ?, identity_id = ?
         WHERE id = ?`,
      ).run(keeper.kind, keeper.identityId, keeper.row.id);
    }
  }

  private removeLegacyProductSchema(): void {
    // Product-level tables from the superseded prototype are intentionally not
    // carried into the inference-only runtime.
    this.raw.exec("PRAGMA foreign_keys = OFF");
    this.raw.exec(`
      DROP TABLE IF EXISTS work_proofs;
      DROP TABLE IF EXISTS ledger_entries;
      DROP TABLE IF EXISTS ledger_transactions;
      DROP TABLE IF EXISTS accounts;
      DROP TABLE IF EXISTS idempotency_keys;
      DROP TABLE IF EXISTS sessions;
      DROP TABLE IF EXISTS jobs;
      DROP TABLE IF EXISTS api_keys;
      DROP TABLE IF EXISTS workers;
      DROP TABLE IF EXISTS users;
      DROP TABLE IF EXISTS owners;
    `);
    this.raw.exec("PRAGMA foreign_keys = ON");
  }
}

function workerMigrationPriority(row: { status: string; deregistered: number }): number {
  if (row.deregistered) return 0;
  if (row.status === "online") return 3;
  if (row.status === "suspect" || row.status === "draining") return 2;
  return 1;
}

function nodeIdentityEventDigest(input: {
  identityKind: string; identityId: string; generation: number; actorKind: string; actorId: string;
  eventType: string; detailsJson: string; previousEventDigest: string | null; occurredAt: number;
}): string {
  return `sha256:${createHash("sha256").update(JSON.stringify({
    schema: "mycellios-node-identity-event/1", identityKind: input.identityKind, identityId: input.identityId,
    generation: input.generation, actorKind: input.actorKind, actorId: input.actorId, eventType: input.eventType,
    detailsJson: input.detailsJson, previousEventDigest: input.previousEventDigest, occurredAt: input.occurredAt,
  })).digest("hex")}`;
}
