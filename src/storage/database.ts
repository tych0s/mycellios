import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";

const SCHEMA_VERSION = 3;

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
    this.raw.prepare("UPDATE schema_meta SET version = ?").run(SCHEMA_VERSION);
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
