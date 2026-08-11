import { mkdirSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";

const SCHEMA_VERSION = 31;

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

      CREATE TABLE IF NOT EXISTS activation_checkpoint_recoveries (
        id TEXT PRIMARY KEY,
        operation_key TEXT NOT NULL UNIQUE,
        model_id TEXT NOT NULL REFERENCES requested_models(id) ON DELETE CASCADE,
        reservation_id TEXT REFERENCES route_reservations(id) ON DELETE SET NULL,
        recovery_generation INTEGER NOT NULL,
        source_checkpoint_id TEXT NOT NULL,
        target_worker_id TEXT NOT NULL,
        target_launch_request_id TEXT NOT NULL,
        target_stage_request_id INTEGER NOT NULL,
        compatibility_json TEXT NOT NULL,
        maximum_bytes INTEGER NOT NULL,
        expires_at INTEGER NOT NULL,
        status TEXT NOT NULL CHECK(status IN (
          'prepared', 'restoring', 'restored', 'promoting', 'promoted', 'failed'
        )),
        transfer_id TEXT,
        error TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        restored_at INTEGER,
        promoted_at INTEGER
      );

      CREATE INDEX IF NOT EXISTS activation_checkpoint_recoveries_model_generation
      ON activation_checkpoint_recoveries(model_id, recovery_generation, created_at);

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

      CREATE TABLE IF NOT EXISTS billing_plans (
        plan_id TEXT NOT NULL,
        version INTEGER NOT NULL CHECK(version > 0),
        price_currency TEXT NOT NULL,
        price_micros INTEGER NOT NULL CHECK(price_micros > 0),
        included_tokens INTEGER NOT NULL CHECK(included_tokens > 0),
        status TEXT NOT NULL CHECK(status IN ('draft', 'active', 'retired')),
        created_at INTEGER NOT NULL,
        PRIMARY KEY(plan_id, version)
      );

      CREATE TABLE IF NOT EXISTS billing_subscriptions (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES api_accounts(user_id) ON DELETE CASCADE,
        plan_id TEXT NOT NULL,
        plan_version INTEGER NOT NULL,
        provider TEXT NOT NULL CHECK(provider IN ('stripe', 'stablecoin')),
        provider_subscription_id TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('active', 'past_due', 'cancelled')),
        period_start INTEGER NOT NULL,
        period_end INTEGER NOT NULL,
        status_changed_at INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        FOREIGN KEY(plan_id, plan_version) REFERENCES billing_plans(plan_id, version),
        UNIQUE(provider, provider_subscription_id)
      );

      CREATE INDEX IF NOT EXISTS billing_subscriptions_user_status
      ON billing_subscriptions(user_id, status, period_end);

      CREATE TABLE IF NOT EXISTS billing_events (
        id TEXT PRIMARY KEY,
        provider TEXT NOT NULL CHECK(provider IN ('stripe', 'stablecoin')),
        provider_event_id TEXT NOT NULL,
        event_type TEXT NOT NULL CHECK(event_type IN ('subscription_paid', 'topup_paid')),
        user_id TEXT NOT NULL REFERENCES api_accounts(user_id) ON DELETE CASCADE,
        event_digest TEXT NOT NULL,
        external_reference TEXT NOT NULL,
        amount_micros INTEGER NOT NULL CHECK(amount_micros > 0),
        currency TEXT NOT NULL,
        occurred_at INTEGER NOT NULL,
        processed_at INTEGER NOT NULL,
        UNIQUE(provider, provider_event_id)
      );

      CREATE INDEX IF NOT EXISTS billing_events_user_occurred
      ON billing_events(user_id, occurred_at DESC);

      CREATE TABLE IF NOT EXISTS billing_topup_quotes (
        quote_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES api_accounts(user_id) ON DELETE CASCADE,
        amount_micros INTEGER NOT NULL CHECK(amount_micros > 0),
        currency TEXT NOT NULL,
        token_amount INTEGER NOT NULL CHECK(token_amount > 0),
        status TEXT NOT NULL CHECK(status IN ('open', 'consumed')),
        consumed_event_key TEXT UNIQUE,
        expires_at INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        consumed_at INTEGER
      );

      CREATE INDEX IF NOT EXISTS billing_topup_quotes_user_status
      ON billing_topup_quotes(user_id, status, expires_at);

      CREATE TABLE IF NOT EXISTS billing_customers (
        provider TEXT NOT NULL CHECK(provider IN ('stripe')),
        user_id TEXT NOT NULL REFERENCES api_accounts(user_id) ON DELETE CASCADE,
        provider_customer_id TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        PRIMARY KEY(provider, user_id),
        UNIQUE(provider, provider_customer_id)
      );

      CREATE TABLE IF NOT EXISTS billing_checkout_operations (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES api_accounts(user_id) ON DELETE CASCADE,
        idempotency_key TEXT NOT NULL,
        kind TEXT NOT NULL CHECK(kind IN ('subscription', 'topup', 'portal')),
        request_digest TEXT NOT NULL,
        quote_id TEXT REFERENCES billing_topup_quotes(quote_id),
        status TEXT NOT NULL CHECK(status IN ('pending', 'completed')),
        provider_session_id TEXT,
        provider_url TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        UNIQUE(user_id, idempotency_key)
      );

      CREATE INDEX IF NOT EXISTS billing_checkout_operations_user_created
      ON billing_checkout_operations(user_id, created_at DESC);

      CREATE TABLE IF NOT EXISTS billing_stablecoin_intents (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES api_accounts(user_id) ON DELETE CASCADE,
        idempotency_key TEXT NOT NULL,
        request_digest TEXT NOT NULL,
        kind TEXT NOT NULL CHECK(kind IN ('subscription', 'topup')),
        quote_id TEXT REFERENCES billing_topup_quotes(quote_id),
        plan_id TEXT,
        plan_version INTEGER,
        subscription_id TEXT,
        period_duration_ms INTEGER,
        amount_micros INTEGER NOT NULL CHECK(amount_micros > 0),
        currency TEXT NOT NULL,
        chain_id TEXT NOT NULL,
        asset TEXT NOT NULL,
        asset_atomic_amount TEXT NOT NULL,
        recipient TEXT,
        provider_reference TEXT,
        checkout_url TEXT,
        status TEXT NOT NULL CHECK(status IN ('allocating', 'awaiting_payment', 'consumed')),
        consumed_event_key TEXT UNIQUE,
        expires_at INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        consumed_at INTEGER,
        UNIQUE(user_id, idempotency_key),
        UNIQUE(chain_id, provider_reference)
      );

      CREATE INDEX IF NOT EXISTS billing_stablecoin_intents_user_status
      ON billing_stablecoin_intents(user_id, status, expires_at);

      CREATE TABLE IF NOT EXISTS credit_grants (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES api_accounts(user_id) ON DELETE CASCADE,
        idempotency_key TEXT NOT NULL UNIQUE,
        source_type TEXT NOT NULL CHECK(source_type IN ('subscription', 'topup')),
        source_event_id TEXT NOT NULL REFERENCES billing_events(id),
        token_amount INTEGER NOT NULL CHECK(token_amount > 0),
        debt_repaid_tokens INTEGER NOT NULL DEFAULT 0 CHECK(debt_repaid_tokens >= 0),
        created_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS credit_grants_user_created
      ON credit_grants(user_id, created_at DESC);

      CREATE TABLE IF NOT EXISTS account_credit_debts (
        user_id TEXT PRIMARY KEY REFERENCES api_accounts(user_id) ON DELETE CASCADE,
        token_debt INTEGER NOT NULL DEFAULT 0 CHECK(token_debt >= 0),
        updated_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS billing_reversals (
        id TEXT PRIMARY KEY,
        provider TEXT NOT NULL CHECK(provider IN ('stripe', 'stablecoin')),
        provider_event_id TEXT NOT NULL,
        original_event_id TEXT NOT NULL REFERENCES billing_events(id),
        user_id TEXT NOT NULL REFERENCES api_accounts(user_id) ON DELETE CASCADE,
        reason TEXT NOT NULL CHECK(reason IN ('refund', 'dispute', 'stablecoin_reorg')),
        event_digest TEXT NOT NULL,
        token_amount INTEGER NOT NULL CHECK(token_amount > 0),
        recovered_tokens INTEGER NOT NULL CHECK(recovered_tokens >= 0),
        debt_tokens INTEGER NOT NULL CHECK(debt_tokens >= 0),
        occurred_at INTEGER NOT NULL,
        processed_at INTEGER NOT NULL,
        UNIQUE(provider, provider_event_id)
      );

      CREATE INDEX IF NOT EXISTS billing_reversals_original_event
      ON billing_reversals(original_event_id, processed_at);

      CREATE TABLE IF NOT EXISTS billing_subscription_events (
        id TEXT PRIMARY KEY,
        provider TEXT NOT NULL CHECK(provider IN ('stripe', 'stablecoin')),
        provider_event_id TEXT NOT NULL,
        provider_subscription_id TEXT NOT NULL,
        user_id TEXT NOT NULL REFERENCES api_accounts(user_id) ON DELETE CASCADE,
        status TEXT NOT NULL CHECK(status IN ('past_due', 'cancelled')),
        event_digest TEXT NOT NULL,
        occurred_at INTEGER NOT NULL,
        processed_at INTEGER NOT NULL,
        applied INTEGER NOT NULL CHECK(applied IN (0, 1)),
        UNIQUE(provider, provider_event_id)
      );

      CREATE INDEX IF NOT EXISTS billing_subscription_events_subscription
      ON billing_subscription_events(provider, provider_subscription_id, occurred_at);

      CREATE TABLE IF NOT EXISTS seller_payout_preferences (
        seller_id TEXT PRIMARY KEY,
        method TEXT NOT NULL CHECK(method IN ('stable', 'spore')),
        updated_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS verified_work_receipts (
        receipt_id TEXT PRIMARY KEY,
        seller_id TEXT NOT NULL,
        worker_id TEXT NOT NULL,
        job_id TEXT NOT NULL,
        stage_id TEXT NOT NULL,
        pricing_version TEXT NOT NULL,
        amount_usd_micros INTEGER NOT NULL CHECK(amount_usd_micros > 0),
        evidence_digest TEXT NOT NULL,
        accepted_at INTEGER NOT NULL,
        verifier_key_id TEXT NOT NULL,
        signature TEXT NOT NULL,
        recorded_at INTEGER NOT NULL,
        UNIQUE(job_id, stage_id)
      );

      CREATE INDEX IF NOT EXISTS verified_work_receipts_seller_accepted
      ON verified_work_receipts(seller_id, accepted_at DESC);

      CREATE TABLE IF NOT EXISTS seller_earnings (
        id TEXT PRIMARY KEY,
        seller_id TEXT NOT NULL,
        receipt_id TEXT NOT NULL UNIQUE REFERENCES verified_work_receipts(receipt_id),
        amount_usd_micros INTEGER NOT NULL CHECK(amount_usd_micros > 0),
        payout_method TEXT NOT NULL CHECK(payout_method IN ('stable', 'spore')),
        status TEXT NOT NULL CHECK(status IN ('available', 'batched', 'paid', 'reversed')),
        created_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS seller_earnings_seller_status
      ON seller_earnings(seller_id, status, created_at);

      CREATE TABLE IF NOT EXISTS seller_payout_destinations (
        id TEXT PRIMARY KEY,
        seller_id TEXT NOT NULL,
        payout_method TEXT NOT NULL CHECK(payout_method IN ('stable', 'spore')),
        destination_kind TEXT NOT NULL CHECK(destination_kind IN ('provider_account', 'wallet')),
        destination_reference TEXT NOT NULL,
        destination_fingerprint TEXT NOT NULL,
        verifier_key_id TEXT NOT NULL,
        attestation_signature TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('active', 'revoked')),
        verified_at INTEGER NOT NULL,
        expires_at INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        revoked_at INTEGER
      );

      CREATE UNIQUE INDEX IF NOT EXISTS seller_payout_destinations_active
      ON seller_payout_destinations(seller_id, payout_method) WHERE status = 'active';

      CREATE TABLE IF NOT EXISTS payout_batches (
        id TEXT PRIMARY KEY,
        seller_id TEXT NOT NULL,
        payout_method TEXT NOT NULL CHECK(payout_method IN ('stable', 'spore')),
        idempotency_key TEXT NOT NULL UNIQUE,
        request_digest TEXT NOT NULL,
        gross_usd_micros INTEGER NOT NULL CHECK(gross_usd_micros > 0),
        debt_offset_usd_micros INTEGER NOT NULL DEFAULT 0 CHECK(debt_offset_usd_micros >= 0),
        amount_usd_micros INTEGER NOT NULL CHECK(amount_usd_micros > 0),
        destination_id TEXT REFERENCES seller_payout_destinations(id),
        destination_reference TEXT,
        destination_fingerprint TEXT,
        status TEXT NOT NULL CHECK(status IN ('prepared', 'submitted', 'paid', 'cancelled')),
        dispatch_key TEXT,
        external_reference TEXT,
        settlement_reference TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        submitted_at INTEGER,
        paid_at INTEGER,
        cancelled_at INTEGER
      );

      CREATE INDEX IF NOT EXISTS payout_batches_seller_status
      ON payout_batches(seller_id, status, created_at);

      CREATE UNIQUE INDEX IF NOT EXISTS payout_batches_dispatch_key
      ON payout_batches(dispatch_key) WHERE dispatch_key IS NOT NULL;

      CREATE UNIQUE INDEX IF NOT EXISTS payout_batches_external_reference
      ON payout_batches(external_reference) WHERE external_reference IS NOT NULL;

      CREATE TABLE IF NOT EXISTS payout_dispatch_operations (
        id TEXT PRIMARY KEY,
        batch_id TEXT NOT NULL UNIQUE REFERENCES payout_batches(id) ON DELETE CASCADE,
        payout_method TEXT NOT NULL CHECK(payout_method IN ('stable', 'spore')),
        dispatch_key TEXT NOT NULL UNIQUE,
        state TEXT NOT NULL CHECK(state IN ('dispatching', 'uncertain', 'submitted', 'paid', 'rejected')),
        external_reference TEXT,
        settlement_reference TEXT,
        spore_quote_id TEXT,
        spore_quote_attestation_digest TEXT,
        last_error TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        reconciled_at INTEGER
      );

      CREATE INDEX IF NOT EXISTS payout_dispatch_operations_state
      ON payout_dispatch_operations(state, updated_at);

      CREATE TABLE IF NOT EXISTS payout_settlement_evidence (
        batch_id TEXT PRIMARY KEY REFERENCES payout_batches(id) ON DELETE CASCADE,
        schema_name TEXT NOT NULL CHECK(schema_name = 'mycellios.payout-settlement.v1'),
        external_reference TEXT NOT NULL,
        settlement_reference TEXT NOT NULL UNIQUE,
        paid_at INTEGER NOT NULL,
        issued_at INTEGER NOT NULL,
        verifier_key_id TEXT NOT NULL,
        signature TEXT NOT NULL,
        recorded_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS spore_conversion_quotes (
        quote_id TEXT PRIMARY KEY,
        batch_id TEXT NOT NULL REFERENCES payout_batches(id) ON DELETE CASCADE,
        seller_id TEXT NOT NULL,
        usd_micros INTEGER NOT NULL CHECK(usd_micros > 0),
        chain_id TEXT NOT NULL,
        asset_id TEXT NOT NULL,
        token_decimals INTEGER NOT NULL,
        token_atomic_amount TEXT NOT NULL,
        destination_fingerprint TEXT NOT NULL,
        issued_at INTEGER NOT NULL,
        expires_at INTEGER NOT NULL,
        oracle_key_id TEXT NOT NULL,
        signature TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('active', 'superseded')),
        recorded_at INTEGER NOT NULL,
        replaced_at INTEGER
      );

      CREATE UNIQUE INDEX IF NOT EXISTS spore_conversion_quotes_active_batch
      ON spore_conversion_quotes(batch_id) WHERE status = 'active';

      CREATE TABLE IF NOT EXISTS payout_batch_items (
        batch_id TEXT NOT NULL REFERENCES payout_batches(id) ON DELETE CASCADE,
        earning_id TEXT NOT NULL REFERENCES seller_earnings(id),
        amount_usd_micros INTEGER NOT NULL CHECK(amount_usd_micros > 0),
        PRIMARY KEY(batch_id, earning_id)
      );

      CREATE TABLE IF NOT EXISTS seller_earning_reversals (
        id TEXT PRIMARY KEY,
        reversal_id TEXT NOT NULL UNIQUE,
        earning_id TEXT NOT NULL UNIQUE REFERENCES seller_earnings(id),
        seller_id TEXT NOT NULL,
        reason TEXT NOT NULL CHECK(reason IN ('fraud', 'verification_error', 'buyer_dispute')),
        evidence_digest TEXT NOT NULL,
        verifier_key_id TEXT NOT NULL,
        signature TEXT NOT NULL,
        amount_usd_micros INTEGER NOT NULL CHECK(amount_usd_micros > 0),
        state TEXT NOT NULL CHECK(state IN ('applied', 'seller_debt', 'pending_payout')),
        reversed_at INTEGER NOT NULL,
        recorded_at INTEGER NOT NULL,
        resolved_at INTEGER
      );

      CREATE INDEX IF NOT EXISTS seller_earning_reversals_seller_state
      ON seller_earning_reversals(seller_id, state, recorded_at);

      CREATE TABLE IF NOT EXISTS seller_debts (
        seller_id TEXT PRIMARY KEY,
        amount_usd_micros INTEGER NOT NULL DEFAULT 0 CHECK(amount_usd_micros >= 0),
        updated_at INTEGER NOT NULL
      );

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

      CREATE TABLE IF NOT EXISTS runtime_link_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        from_node_id TEXT NOT NULL,
        to_node_id TEXT NOT NULL,
        measured_at INTEGER NOT NULL,
        rtt_ms REAL,
        goodput_mbps REAL,
        transport_mode TEXT NOT NULL DEFAULT 'relay'
          CHECK(transport_mode IN ('direct', 'relay')),
        CHECK(from_node_id <> to_node_id),
        CHECK((rtt_ms IS NULL AND goodput_mbps IS NULL)
          OR (rtt_ms > 0 AND goodput_mbps > 0))
      );

      CREATE INDEX IF NOT EXISTS runtime_link_samples_pair_time
      ON runtime_link_samples(from_node_id, to_node_id, measured_at DESC, id DESC);

      CREATE TABLE IF NOT EXISTS engine_runtime_activation_plans (
        model_id TEXT NOT NULL,
        activation_id TEXT NOT NULL,
        worker_id TEXT NOT NULL,
        plan_json TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        PRIMARY KEY(model_id, worker_id)
      );

      CREATE INDEX IF NOT EXISTS engine_runtime_activation_plans_activation
      ON engine_runtime_activation_plans(activation_id, model_id);

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

    if (currentVersion < 27) {
      const payoutColumns = this.raw.prepare("PRAGMA table_info(payout_batches)").all() as Array<{ name: string }>;
      const names = new Set(payoutColumns.map((column) => column.name));
      if (!names.has("destination_id")) this.raw.exec("ALTER TABLE payout_batches ADD COLUMN destination_id TEXT");
      if (!names.has("destination_reference")) this.raw.exec("ALTER TABLE payout_batches ADD COLUMN destination_reference TEXT");
      if (!names.has("destination_fingerprint")) this.raw.exec("ALTER TABLE payout_batches ADD COLUMN destination_fingerprint TEXT");
    }
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
    if (currentVersion < 17) {
      const columns = this.raw.prepare("PRAGMA table_info(credit_grants)").all() as Array<{
        name: string;
      }>;
      if (!columns.some((column) => column.name === "debt_repaid_tokens")) {
        this.raw.exec(
          "ALTER TABLE credit_grants ADD COLUMN debt_repaid_tokens INTEGER NOT NULL DEFAULT 0 CHECK(debt_repaid_tokens >= 0)",
        );
      }
    }
    if (currentVersion < 18) {
      const columns = this.raw.prepare("PRAGMA table_info(billing_subscriptions)").all() as Array<{
        name: string;
      }>;
      if (!columns.some((column) => column.name === "status_changed_at")) {
        this.raw.exec(
          "ALTER TABLE billing_subscriptions ADD COLUMN status_changed_at INTEGER NOT NULL DEFAULT 0",
        );
        this.raw.exec(
          "UPDATE billing_subscriptions SET status_changed_at = updated_at WHERE status_changed_at = 0",
        );
      }
    }
    if (currentVersion === 19) {
      this.raw.exec("PRAGMA foreign_keys = OFF");
      this.raw.exec(`
        ALTER TABLE payout_batch_items RENAME TO payout_batch_items_v19;
        CREATE TABLE payout_batch_items (
          batch_id TEXT NOT NULL REFERENCES payout_batches(id) ON DELETE CASCADE,
          earning_id TEXT NOT NULL REFERENCES seller_earnings(id),
          amount_usd_micros INTEGER NOT NULL CHECK(amount_usd_micros > 0),
          PRIMARY KEY(batch_id, earning_id)
        );
        INSERT INTO payout_batch_items(batch_id, earning_id, amount_usd_micros)
        SELECT batch_id, earning_id, amount_usd_micros FROM payout_batch_items_v19;
        DROP TABLE payout_batch_items_v19;
      `);
      this.raw.exec("PRAGMA foreign_keys = ON");
    }
    if (currentVersion < 22) {
      const columns = this.raw.prepare("PRAGMA table_info(payout_batches)").all() as Array<{
        name: string;
      }>;
      if (!columns.some((column) => column.name === "gross_usd_micros")) {
        this.raw.exec(
          "ALTER TABLE payout_batches ADD COLUMN gross_usd_micros INTEGER NOT NULL DEFAULT 0 CHECK(gross_usd_micros >= 0)",
        );
        this.raw.exec("UPDATE payout_batches SET gross_usd_micros = amount_usd_micros");
      }
      if (!columns.some((column) => column.name === "debt_offset_usd_micros")) {
        this.raw.exec(
          "ALTER TABLE payout_batches ADD COLUMN debt_offset_usd_micros INTEGER NOT NULL DEFAULT 0 CHECK(debt_offset_usd_micros >= 0)",
        );
      }
    }
    if (currentVersion === 29) {
      this.raw.exec("PRAGMA foreign_keys = OFF");
      this.raw.exec(`
        DROP INDEX IF EXISTS spore_conversion_quotes_active_batch;
        ALTER TABLE spore_conversion_quotes RENAME TO spore_conversion_quotes_v29;
        CREATE TABLE spore_conversion_quotes (
          quote_id TEXT PRIMARY KEY,
          batch_id TEXT NOT NULL REFERENCES payout_batches(id) ON DELETE CASCADE,
          seller_id TEXT NOT NULL,
          usd_micros INTEGER NOT NULL CHECK(usd_micros > 0),
          chain_id TEXT NOT NULL,
          asset_id TEXT NOT NULL,
          token_decimals INTEGER NOT NULL,
          token_atomic_amount TEXT NOT NULL,
          destination_fingerprint TEXT NOT NULL,
          issued_at INTEGER NOT NULL,
          expires_at INTEGER NOT NULL,
          oracle_key_id TEXT NOT NULL,
          signature TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('active', 'superseded')),
          recorded_at INTEGER NOT NULL,
          replaced_at INTEGER
        );
        INSERT INTO spore_conversion_quotes(
          quote_id, batch_id, seller_id, usd_micros, chain_id, asset_id,
          token_decimals, token_atomic_amount, destination_fingerprint,
          issued_at, expires_at, oracle_key_id, signature, status, recorded_at
        ) SELECT quote_id, batch_id, seller_id, usd_micros, chain_id, asset_id,
                 token_decimals, token_atomic_amount, destination_fingerprint,
                 issued_at, expires_at, oracle_key_id, signature, 'active', recorded_at
          FROM spore_conversion_quotes_v29;
        DROP TABLE spore_conversion_quotes_v29;
        CREATE UNIQUE INDEX spore_conversion_quotes_active_batch
        ON spore_conversion_quotes(batch_id) WHERE status = 'active';
      `);
      this.raw.exec("PRAGMA foreign_keys = ON");
    }
    if (currentVersion === 30) {
      this.raw.exec("ALTER TABLE payout_dispatch_operations ADD COLUMN spore_quote_id TEXT");
      this.raw.exec("ALTER TABLE payout_dispatch_operations ADD COLUMN spore_quote_attestation_digest TEXT");
    }
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

      CREATE TRIGGER IF NOT EXISTS payout_dispatch_spore_quote_insert_guard
      BEFORE INSERT ON payout_dispatch_operations
      WHEN (NEW.payout_method = 'spore' AND (
              NEW.spore_quote_id IS NULL OR NEW.spore_quote_attestation_digest IS NULL
            ))
        OR (NEW.payout_method = 'stable' AND (
              NEW.spore_quote_id IS NOT NULL OR NEW.spore_quote_attestation_digest IS NOT NULL
            ))
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_dispatch_spore_quote_audit');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_dispatch_time_insert_guard
      BEFORE INSERT ON payout_dispatch_operations
      WHEN NEW.created_at <= 0 OR NEW.updated_at < NEW.created_at
        OR (NEW.reconciled_at IS NOT NULL
          AND (NEW.reconciled_at < NEW.created_at OR NEW.reconciled_at > NEW.updated_at))
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_dispatch_time');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_dispatch_time_update_guard
      BEFORE UPDATE OF created_at, updated_at, reconciled_at
      ON payout_dispatch_operations
      WHEN NEW.created_at <> OLD.created_at
        OR NEW.updated_at < OLD.updated_at
        OR NEW.updated_at < NEW.created_at
        OR (NEW.reconciled_at IS NOT NULL
          AND (NEW.reconciled_at < NEW.created_at OR NEW.reconciled_at > NEW.updated_at))
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_dispatch_time');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_dispatch_batch_binding_insert_guard
      BEFORE INSERT ON payout_dispatch_operations
      WHEN NOT EXISTS (
        SELECT 1 FROM payout_batches batch
        WHERE batch.id = NEW.batch_id
          AND batch.payout_method = NEW.payout_method
          AND NEW.dispatch_key = 'payout:' || batch.id
      )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_dispatch_batch_binding');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_dispatch_economic_composition_insert_guard
      BEFORE INSERT ON payout_dispatch_operations
      WHEN NOT EXISTS (
        SELECT 1 FROM payout_batches batch
        WHERE batch.id = NEW.batch_id
          AND batch.gross_usd_micros = (
            SELECT COALESCE(SUM(item.amount_usd_micros), 0)
            FROM payout_batch_items item WHERE item.batch_id = batch.id
          )
          AND NOT EXISTS (
            SELECT 1
            FROM payout_batch_items item
            JOIN seller_earnings earning ON earning.id = item.earning_id
            WHERE item.batch_id = batch.id
              AND (earning.status <> 'batched'
                OR earning.seller_id <> batch.seller_id
                OR earning.payout_method <> batch.payout_method
                OR earning.amount_usd_micros <> item.amount_usd_micros)
          )
      )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_dispatch_economic_composition');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_dispatch_spore_quote_binding_insert_guard
      BEFORE INSERT ON payout_dispatch_operations
      WHEN NEW.payout_method = 'spore'
        AND NEW.spore_quote_id IS NOT NULL
        AND NEW.spore_quote_attestation_digest IS NOT NULL
        AND NOT EXISTS (
          SELECT 1 FROM spore_conversion_quotes quote
          WHERE quote.quote_id = NEW.spore_quote_id
            AND quote.batch_id = NEW.batch_id
            AND quote.status = 'active'
        )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_dispatch_spore_quote_binding');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_dispatch_spore_quote_update_guard
      BEFORE UPDATE OF payout_method, spore_quote_id, spore_quote_attestation_digest
      ON payout_dispatch_operations
      WHEN (NEW.payout_method = 'spore' AND (
              NEW.spore_quote_id IS NULL OR NEW.spore_quote_attestation_digest IS NULL
            ))
        OR (NEW.payout_method = 'stable' AND (
              NEW.spore_quote_id IS NOT NULL OR NEW.spore_quote_attestation_digest IS NOT NULL
            ))
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_dispatch_spore_quote_audit');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_dispatch_spore_quote_immutable_guard
      BEFORE UPDATE OF payout_method, spore_quote_id, spore_quote_attestation_digest
      ON payout_dispatch_operations
      WHEN OLD.payout_method IS NOT NEW.payout_method
        OR OLD.spore_quote_id IS NOT NEW.spore_quote_id
        OR OLD.spore_quote_attestation_digest IS NOT NEW.spore_quote_attestation_digest
      BEGIN
        SELECT RAISE(ABORT, 'immutable_payout_dispatch_spore_quote_audit');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_dispatch_provider_identity_immutable_guard
      BEFORE UPDATE OF batch_id, dispatch_key, external_reference, settlement_reference
      ON payout_dispatch_operations
      WHEN OLD.batch_id IS NOT NEW.batch_id
        OR OLD.dispatch_key IS NOT NEW.dispatch_key
        OR (OLD.external_reference IS NOT NULL
            AND OLD.external_reference IS NOT NEW.external_reference)
        OR (OLD.state IN ('paid', 'rejected')
            AND OLD.external_reference IS NOT NEW.external_reference)
        OR (OLD.settlement_reference IS NOT NULL
            AND OLD.settlement_reference IS NOT NEW.settlement_reference)
      BEGIN
        SELECT RAISE(ABORT, 'immutable_payout_dispatch_provider_identity');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_settlement_evidence_immutable_guard
      BEFORE UPDATE ON payout_settlement_evidence
      BEGIN
        SELECT RAISE(ABORT, 'immutable_payout_settlement_evidence');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_settlement_evidence_insert_guard
      BEFORE INSERT ON payout_settlement_evidence
      WHEN NEW.paid_at <= 0 OR NEW.issued_at <= 0 OR NEW.recorded_at <= 0
        OR NEW.paid_at > NEW.issued_at + 60000
        OR NEW.issued_at > NEW.recorded_at + 60000
        OR NOT EXISTS (
          SELECT 1
          FROM payout_batches batch
          JOIN payout_dispatch_operations operation ON operation.batch_id = batch.id
          WHERE batch.id = NEW.batch_id
            AND batch.status IN ('submitted', 'paid')
            AND operation.state IN ('submitted', 'uncertain', 'paid')
            AND batch.external_reference = NEW.external_reference
            AND operation.external_reference = NEW.external_reference
        )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_settlement_evidence_binding');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_dispatch_settlement_evidence_binding_guard
      BEFORE UPDATE OF state, external_reference, settlement_reference
      ON payout_dispatch_operations
      WHEN NEW.state = 'paid' AND EXISTS (
        SELECT 1 FROM payout_settlement_evidence evidence
        WHERE evidence.batch_id = NEW.batch_id
          AND (evidence.external_reference <> NEW.external_reference
            OR evidence.settlement_reference <> NEW.settlement_reference)
      )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_settlement_evidence_binding');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_batch_settlement_evidence_binding_guard
      BEFORE UPDATE OF status, external_reference, settlement_reference, paid_at
      ON payout_batches
      WHEN NEW.status = 'paid' AND EXISTS (
        SELECT 1 FROM payout_settlement_evidence evidence
        WHERE evidence.batch_id = NEW.id
          AND (evidence.external_reference <> NEW.external_reference
            OR evidence.settlement_reference <> NEW.settlement_reference
            OR evidence.paid_at <> NEW.paid_at)
      )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_settlement_evidence_binding');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_dispatch_state_transition_guard
      BEFORE UPDATE OF state ON payout_dispatch_operations
      WHEN NOT (
        OLD.state = NEW.state
        OR (OLD.state IN ('dispatching', 'uncertain', 'submitted')
            AND NEW.state IN ('uncertain', 'submitted', 'paid', 'rejected'))
      )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_dispatch_state_transition');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_dispatch_state_evidence_insert_guard
      BEFORE INSERT ON payout_dispatch_operations
      WHEN (NEW.state IN ('submitted', 'paid') AND NEW.external_reference IS NULL)
        OR (NEW.state = 'paid' AND NEW.settlement_reference IS NULL)
        OR (NEW.state <> 'paid' AND NEW.settlement_reference IS NOT NULL)
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_dispatch_state_evidence');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_dispatch_batch_projection_insert_guard
      BEFORE INSERT ON payout_dispatch_operations
      WHEN NOT EXISTS (
        SELECT 1 FROM payout_batches batch
        WHERE batch.id = NEW.batch_id AND (
          (NEW.state = 'dispatching'
            AND NEW.external_reference IS NULL
            AND batch.status = 'prepared')
          OR (NEW.state = 'uncertain' AND (
            (NEW.external_reference IS NULL AND batch.status = 'prepared')
            OR (NEW.external_reference IS NOT NULL
              AND batch.status = 'submitted'
              AND batch.dispatch_key = NEW.dispatch_key
              AND batch.external_reference = NEW.external_reference)
          ))
          OR (NEW.state = 'submitted'
            AND batch.status = 'submitted'
            AND batch.dispatch_key = NEW.dispatch_key
            AND batch.external_reference = NEW.external_reference)
          OR (NEW.state = 'paid'
            AND batch.status = 'paid'
            AND batch.dispatch_key = NEW.dispatch_key
            AND batch.external_reference = NEW.external_reference
            AND batch.settlement_reference = NEW.settlement_reference)
          OR (NEW.state = 'rejected' AND (
            (NEW.external_reference IS NULL AND batch.status IN ('prepared', 'cancelled'))
            OR (NEW.external_reference IS NOT NULL
              AND batch.status = 'submitted'
              AND batch.dispatch_key = NEW.dispatch_key
              AND batch.external_reference = NEW.external_reference)
          ))
        )
      )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_dispatch_batch_projection');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_dispatch_batch_projection_update_guard
      BEFORE UPDATE OF state, external_reference, settlement_reference
      ON payout_dispatch_operations
      WHEN NOT EXISTS (
        SELECT 1 FROM payout_batches batch
        WHERE batch.id = NEW.batch_id AND (
          (NEW.state = 'dispatching'
            AND NEW.external_reference IS NULL
            AND batch.status = 'prepared')
          OR (NEW.state = 'uncertain' AND (
            (NEW.external_reference IS NULL AND batch.status = 'prepared')
            OR (NEW.external_reference IS NOT NULL
              AND batch.status = 'submitted'
              AND batch.dispatch_key = NEW.dispatch_key
              AND batch.external_reference = NEW.external_reference)
          ))
          OR (NEW.state = 'submitted'
            AND batch.status = 'submitted'
            AND batch.dispatch_key = NEW.dispatch_key
            AND batch.external_reference = NEW.external_reference)
          OR (NEW.state = 'paid'
            AND batch.status = 'paid'
            AND batch.dispatch_key = NEW.dispatch_key
            AND batch.external_reference = NEW.external_reference
            AND batch.settlement_reference = NEW.settlement_reference)
          OR (NEW.state = 'rejected' AND (
            (NEW.external_reference IS NULL AND batch.status IN ('prepared', 'cancelled'))
            OR (NEW.external_reference IS NOT NULL
              AND batch.status = 'submitted'
              AND batch.dispatch_key = NEW.dispatch_key
              AND batch.external_reference = NEW.external_reference)
          ))
        )
      )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_dispatch_batch_projection');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_dispatch_state_evidence_update_guard
      BEFORE UPDATE OF state, external_reference, settlement_reference
      ON payout_dispatch_operations
      WHEN (NEW.state IN ('submitted', 'paid') AND NEW.external_reference IS NULL)
        OR (NEW.state = 'paid' AND NEW.settlement_reference IS NULL)
        OR (NEW.state <> 'paid' AND NEW.settlement_reference IS NOT NULL)
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_dispatch_state_evidence');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_dispatch_operation_delete_guard
      BEFORE DELETE ON payout_dispatch_operations
      BEGIN
        SELECT RAISE(ABORT, 'immutable_payout_dispatch_operation');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_settlement_evidence_delete_guard
      BEFORE DELETE ON payout_settlement_evidence
      BEGIN
        SELECT RAISE(ABORT, 'immutable_payout_settlement_evidence');
      END;

      CREATE TRIGGER IF NOT EXISTS spore_conversion_quote_content_immutable_guard
      BEFORE UPDATE OF quote_id, batch_id, seller_id, usd_micros, chain_id,
        asset_id, token_decimals, token_atomic_amount, destination_fingerprint,
        issued_at, expires_at, oracle_key_id, signature, recorded_at
      ON spore_conversion_quotes
      BEGIN
        SELECT RAISE(ABORT, 'immutable_spore_conversion_quote_content');
      END;

      CREATE TRIGGER IF NOT EXISTS spore_conversion_quote_batch_binding_insert_guard
      BEFORE INSERT ON spore_conversion_quotes
      WHEN NOT EXISTS (
        SELECT 1 FROM payout_batches batch
        WHERE batch.id = NEW.batch_id
          AND batch.payout_method = 'spore'
          AND batch.status = 'prepared'
          AND batch.seller_id = NEW.seller_id
          AND batch.amount_usd_micros = NEW.usd_micros
          AND batch.destination_fingerprint = NEW.destination_fingerprint
      )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_spore_conversion_quote_batch_binding');
      END;

      CREATE TRIGGER IF NOT EXISTS spore_conversion_quote_lifecycle_guard
      BEFORE UPDATE OF status, replaced_at ON spore_conversion_quotes
      WHEN NOT (
        (OLD.status = NEW.status AND OLD.replaced_at IS NEW.replaced_at)
        OR (OLD.status = 'active' AND NEW.status = 'superseded'
            AND OLD.replaced_at IS NULL AND NEW.replaced_at IS NOT NULL)
      )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_spore_conversion_quote_lifecycle');
      END;

      CREATE TRIGGER IF NOT EXISTS spore_conversion_quote_delete_guard
      BEFORE DELETE ON spore_conversion_quotes
      BEGIN
        SELECT RAISE(ABORT, 'immutable_spore_conversion_quote');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_batch_economic_snapshot_immutable_guard
      BEFORE UPDATE OF id, seller_id, payout_method, idempotency_key,
        request_digest, gross_usd_micros, debt_offset_usd_micros,
        amount_usd_micros, destination_id, destination_reference,
        destination_fingerprint, created_at
      ON payout_batches
      BEGIN
        SELECT RAISE(ABORT, 'immutable_payout_batch_economic_snapshot');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_batch_amount_equation_insert_guard
      BEFORE INSERT ON payout_batches
      WHEN NEW.debt_offset_usd_micros > NEW.gross_usd_micros
        OR NEW.amount_usd_micros <> NEW.gross_usd_micros - NEW.debt_offset_usd_micros
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_batch_amount_equation');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_batch_destination_binding_insert_guard
      BEFORE INSERT ON payout_batches
      WHEN (
        NEW.destination_id IS NULL
        AND (NEW.destination_reference IS NOT NULL OR NEW.destination_fingerprint IS NOT NULL)
      ) OR (
        NEW.destination_id IS NOT NULL
        AND NOT EXISTS (
          SELECT 1 FROM seller_payout_destinations destination
          WHERE destination.id = NEW.destination_id
            AND destination.seller_id = NEW.seller_id
            AND destination.payout_method = NEW.payout_method
            AND destination.destination_reference = NEW.destination_reference
            AND destination.destination_fingerprint = NEW.destination_fingerprint
            AND destination.status = 'active'
        )
      )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_batch_destination_binding');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_batch_state_transition_guard
      BEFORE UPDATE OF status ON payout_batches
      WHEN NOT (
        OLD.status = NEW.status
        OR (OLD.status = 'prepared' AND NEW.status IN ('submitted', 'cancelled'))
        OR (OLD.status = 'submitted' AND NEW.status = 'paid')
      )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_batch_state_transition');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_batch_dispatched_cancellation_guard
      BEFORE UPDATE OF status ON payout_batches
      WHEN NEW.status = 'cancelled'
        AND EXISTS (
          SELECT 1 FROM payout_dispatch_operations operation
          WHERE operation.batch_id = NEW.id
            AND NOT (
              operation.state = 'rejected'
              AND operation.external_reference IS NULL
              AND operation.settlement_reference IS NULL
            )
        )
      BEGIN
        SELECT RAISE(ABORT, 'payout_cancellation_unsafe');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_batch_state_evidence_insert_guard
      BEFORE INSERT ON payout_batches
      WHEN NOT (
        NEW.status = 'prepared'
        AND NEW.dispatch_key IS NULL
        AND NEW.external_reference IS NULL
        AND NEW.settlement_reference IS NULL
        AND NEW.submitted_at IS NULL
        AND NEW.paid_at IS NULL
        AND NEW.cancelled_at IS NULL
      )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_batch_state_evidence');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_batch_state_evidence_update_guard
      BEFORE UPDATE OF status, dispatch_key, external_reference,
        settlement_reference, submitted_at, paid_at, cancelled_at
      ON payout_batches
      WHEN NOT (
        (NEW.status = 'prepared'
          AND NEW.dispatch_key IS NULL AND NEW.external_reference IS NULL
          AND NEW.settlement_reference IS NULL AND NEW.submitted_at IS NULL
          AND NEW.paid_at IS NULL AND NEW.cancelled_at IS NULL)
        OR (NEW.status = 'submitted'
          AND NEW.dispatch_key IS NOT NULL AND NEW.external_reference IS NOT NULL
          AND NEW.settlement_reference IS NULL AND NEW.submitted_at IS NOT NULL
          AND NEW.paid_at IS NULL AND NEW.cancelled_at IS NULL)
        OR (NEW.status = 'paid'
          AND NEW.dispatch_key IS NOT NULL AND NEW.external_reference IS NOT NULL
          AND NEW.settlement_reference IS NOT NULL AND NEW.submitted_at IS NOT NULL
          AND NEW.paid_at IS NOT NULL AND NEW.cancelled_at IS NULL)
        OR (NEW.status = 'cancelled'
          AND NEW.dispatch_key IS NULL AND NEW.external_reference IS NULL
          AND NEW.settlement_reference IS NULL AND NEW.submitted_at IS NULL
          AND NEW.paid_at IS NULL AND NEW.cancelled_at IS NOT NULL)
      )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_batch_state_evidence');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_batch_provider_identity_immutable_guard
      BEFORE UPDATE OF dispatch_key, external_reference, settlement_reference
      ON payout_batches
      WHEN (OLD.dispatch_key IS NOT NULL AND OLD.dispatch_key IS NOT NEW.dispatch_key)
        OR (OLD.external_reference IS NOT NULL
            AND OLD.external_reference IS NOT NEW.external_reference)
        OR (OLD.settlement_reference IS NOT NULL
            AND OLD.settlement_reference IS NOT NEW.settlement_reference)
      BEGIN
        SELECT RAISE(ABORT, 'immutable_payout_batch_provider_identity');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_batch_time_insert_guard
      BEFORE INSERT ON payout_batches
      WHEN NEW.created_at <= 0
        OR NEW.updated_at < NEW.created_at
        OR (NEW.submitted_at IS NOT NULL AND (
          NEW.submitted_at < NEW.created_at OR NEW.submitted_at > NEW.updated_at
        ))
        OR (NEW.cancelled_at IS NOT NULL AND (
          NEW.cancelled_at < NEW.created_at OR NEW.cancelled_at > NEW.updated_at
        ))
        OR (NEW.paid_at IS NOT NULL AND NEW.paid_at <= 0)
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_batch_time');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_batch_time_update_guard
      BEFORE UPDATE OF created_at, updated_at, submitted_at, paid_at, cancelled_at
      ON payout_batches
      WHEN NEW.created_at <> OLD.created_at
        OR NEW.updated_at < OLD.updated_at
        OR NEW.updated_at < NEW.created_at
        OR (OLD.submitted_at IS NOT NULL AND NEW.submitted_at IS NOT OLD.submitted_at)
        OR (OLD.paid_at IS NOT NULL AND NEW.paid_at IS NOT OLD.paid_at)
        OR (OLD.cancelled_at IS NOT NULL AND NEW.cancelled_at IS NOT OLD.cancelled_at)
        OR (NEW.submitted_at IS NOT NULL AND (
          NEW.submitted_at < NEW.created_at OR NEW.submitted_at > NEW.updated_at
        ))
        OR (NEW.cancelled_at IS NOT NULL AND (
          NEW.cancelled_at < NEW.created_at OR NEW.cancelled_at > NEW.updated_at
        ))
        OR (NEW.paid_at IS NOT NULL AND NEW.paid_at <= 0)
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_batch_time');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_batch_item_insert_guard
      BEFORE INSERT ON payout_batch_items
      WHEN NOT EXISTS (
        SELECT 1
        FROM payout_batches b
        JOIN seller_earnings e ON e.id = NEW.earning_id
        WHERE b.id = NEW.batch_id
          AND b.status = 'prepared'
          AND e.status = 'available'
          AND e.seller_id = b.seller_id
          AND e.payout_method = b.payout_method
          AND e.amount_usd_micros = NEW.amount_usd_micros
          AND NOT EXISTS (
            SELECT 1
            FROM payout_batch_items existing_item
            JOIN payout_batches existing_batch ON existing_batch.id = existing_item.batch_id
            WHERE existing_item.earning_id = NEW.earning_id
              AND existing_batch.status <> 'cancelled'
          )
      )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_payout_batch_item');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_batch_item_update_guard
      BEFORE UPDATE ON payout_batch_items
      BEGIN
        SELECT RAISE(ABORT, 'immutable_payout_batch_item');
      END;

      CREATE TRIGGER IF NOT EXISTS payout_batch_item_delete_guard
      BEFORE DELETE ON payout_batch_items
      BEGIN
        SELECT RAISE(ABORT, 'immutable_payout_batch_item');
      END;

      CREATE TRIGGER IF NOT EXISTS seller_earning_status_transition_guard
      BEFORE UPDATE OF status ON seller_earnings
      WHEN NOT (
        OLD.status = NEW.status
        OR (OLD.status = 'available' AND NEW.status = 'reversed')
        OR (OLD.status = 'paid' AND NEW.status = 'reversed')
        OR (OLD.status = 'available' AND NEW.status = 'batched' AND EXISTS (
          SELECT 1 FROM payout_batch_items i
          JOIN payout_batches b ON b.id = i.batch_id
          WHERE i.earning_id = OLD.id AND b.status = 'prepared'
        ))
        OR (OLD.status = 'batched' AND NEW.status = 'available' AND NOT EXISTS (
          SELECT 1 FROM payout_batch_items i
          JOIN payout_batches b ON b.id = i.batch_id
          WHERE i.earning_id = OLD.id AND b.status <> 'cancelled'
        ))
        OR (OLD.status = 'batched' AND NEW.status = 'paid' AND EXISTS (
          SELECT 1 FROM payout_batch_items i
          JOIN payout_batches b ON b.id = i.batch_id
          WHERE i.earning_id = OLD.id AND b.status = 'paid'
        ))
      )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_seller_earning_status_transition');
      END;

      CREATE TRIGGER IF NOT EXISTS verified_work_receipt_immutable_guard
      BEFORE UPDATE ON verified_work_receipts
      BEGIN
        SELECT RAISE(ABORT, 'immutable_verified_work_receipt');
      END;

      CREATE TRIGGER IF NOT EXISTS verified_work_receipt_delete_guard
      BEFORE DELETE ON verified_work_receipts
      BEGIN
        SELECT RAISE(ABORT, 'immutable_verified_work_receipt');
      END;

      CREATE TRIGGER IF NOT EXISTS seller_payout_destination_insert_guard
      BEFORE INSERT ON seller_payout_destinations
      WHEN NEW.status <> 'active' OR NEW.revoked_at IS NOT NULL
      BEGIN
        SELECT RAISE(ABORT, 'invalid_seller_payout_destination_state');
      END;

      CREATE TRIGGER IF NOT EXISTS seller_payout_destination_content_immutable_guard
      BEFORE UPDATE OF id, seller_id, payout_method, destination_kind,
        destination_reference, destination_fingerprint, verifier_key_id,
        attestation_signature, verified_at, expires_at, created_at
      ON seller_payout_destinations
      BEGIN
        SELECT RAISE(ABORT, 'immutable_seller_payout_destination_content');
      END;

      CREATE TRIGGER IF NOT EXISTS seller_payout_destination_lifecycle_guard
      BEFORE UPDATE OF status, revoked_at ON seller_payout_destinations
      WHEN NOT (
        (OLD.status = NEW.status AND OLD.revoked_at IS NEW.revoked_at)
        OR (OLD.status = 'active' AND NEW.status = 'revoked'
            AND OLD.revoked_at IS NULL AND NEW.revoked_at IS NOT NULL)
      )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_seller_payout_destination_lifecycle');
      END;

      CREATE TRIGGER IF NOT EXISTS seller_payout_destination_delete_guard
      BEFORE DELETE ON seller_payout_destinations
      BEGIN
        SELECT RAISE(ABORT, 'immutable_seller_payout_destination');
      END;

      CREATE TRIGGER IF NOT EXISTS seller_earning_insert_binding_guard
      BEFORE INSERT ON seller_earnings
      WHEN NEW.status <> 'available' OR NOT EXISTS (
        SELECT 1 FROM verified_work_receipts receipt
        WHERE receipt.receipt_id = NEW.receipt_id
          AND receipt.seller_id = NEW.seller_id
          AND receipt.amount_usd_micros = NEW.amount_usd_micros
      )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_seller_earning_receipt_binding');
      END;

      CREATE TRIGGER IF NOT EXISTS seller_earning_economic_immutable_guard
      BEFORE UPDATE OF id, seller_id, receipt_id, amount_usd_micros,
        payout_method, created_at
      ON seller_earnings
      BEGIN
        SELECT RAISE(ABORT, 'immutable_seller_earning_economic_snapshot');
      END;

      CREATE TRIGGER IF NOT EXISTS seller_earning_delete_guard
      BEFORE DELETE ON seller_earnings
      BEGIN
        SELECT RAISE(ABORT, 'immutable_seller_earning');
      END;

      CREATE TRIGGER IF NOT EXISTS seller_earning_reversal_insert_guard
      BEFORE INSERT ON seller_earning_reversals
      WHEN (NEW.state = 'pending_payout' AND NEW.resolved_at IS NOT NULL)
        OR (NEW.state IN ('applied', 'seller_debt') AND NEW.resolved_at IS NULL)
      BEGIN
        SELECT RAISE(ABORT, 'invalid_seller_earning_reversal_state');
      END;

      CREATE TRIGGER IF NOT EXISTS seller_earning_reversal_content_immutable_guard
      BEFORE UPDATE OF id, reversal_id, earning_id, seller_id, reason,
        evidence_digest, verifier_key_id, signature, amount_usd_micros,
        reversed_at, recorded_at
      ON seller_earning_reversals
      BEGIN
        SELECT RAISE(ABORT, 'immutable_seller_earning_reversal_content');
      END;

      CREATE TRIGGER IF NOT EXISTS seller_earning_reversal_lifecycle_guard
      BEFORE UPDATE OF state, resolved_at ON seller_earning_reversals
      WHEN NOT (
        (OLD.state = NEW.state AND OLD.resolved_at IS NEW.resolved_at)
        OR (OLD.state = 'pending_payout' AND NEW.state = 'seller_debt'
            AND OLD.resolved_at IS NULL AND NEW.resolved_at IS NOT NULL)
      )
      BEGIN
        SELECT RAISE(ABORT, 'invalid_seller_earning_reversal_lifecycle');
      END;

      CREATE TRIGGER IF NOT EXISTS seller_earning_reversal_delete_guard
      BEFORE DELETE ON seller_earning_reversals
      BEGIN
        SELECT RAISE(ABORT, 'immutable_seller_earning_reversal');
      END;

      CREATE TRIGGER IF NOT EXISTS seller_debt_identity_time_guard
      BEFORE UPDATE OF seller_id, updated_at ON seller_debts
      WHEN OLD.seller_id IS NOT NEW.seller_id OR NEW.updated_at < OLD.updated_at
      BEGIN
        SELECT RAISE(ABORT, 'invalid_seller_debt_identity_time');
      END;

      CREATE TRIGGER IF NOT EXISTS seller_debt_delete_guard
      BEFORE DELETE ON seller_debts
      BEGIN
        SELECT RAISE(ABORT, 'immutable_seller_debt');
      END;
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
