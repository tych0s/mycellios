import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import { MeshService } from "../src/coordinator/mesh-service.js";
import { WorkerHub } from "../src/coordinator/worker-hub.js";
import { Scheduler } from "../src/scheduler/scheduler.js";
import { MeshDatabase } from "../src/storage/database.js";
import { MeshStore } from "../src/storage/store.js";

describe("coordinator crash recovery", () => {
  const directories: string[] = [];
  afterEach(() => {
    for (const directory of directories.splice(0)) {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("fails orphaned inference jobs on restart", () => {
    const directory = mkdtempSync(join(tmpdir(), "gpu-mesh-recovery-"));
    directories.push(directory);
    const path = join(directory, "mesh.db");
    let database = new MeshDatabase(path);
    let store = new MeshStore(database);
    store.createJob({
      id: "job-orphan",
      sessionId: "session",
      model: "model",
      workloadClass: "interactive",
      deadlineAt: Date.now() + 60_000,
    });
    database.close();

    database = new MeshDatabase(path);
    store = new MeshStore(database);
    const hub = new WorkerHub(store);
    new MeshService(store, new Scheduler(store), hub, 10_000);

    expect(store.getJob("job-orphan")?.status).toBe("failed");
    expect(store.getJob("job-orphan")?.failureCode).toBe("coordinator_restarted");
    expect((database.raw.prepare(
      "SELECT COUNT(*) AS count FROM execution_receipts WHERE job_id = 'job-orphan'",
    ).get() as { count: number }).count).toBe(0);
    hub.close();
    database.close();
  });

  it("migrates a version 2 worker database without losing registered rows", () => {
    const directory = mkdtempSync(join(tmpdir(), "gpu-mesh-v2-migration-"));
    directories.push(directory);
    const path = join(directory, "mesh.db");
    const legacy = new DatabaseSync(path);
    legacy.exec(`
      CREATE TABLE schema_meta (version INTEGER NOT NULL);
      INSERT INTO schema_meta(version) VALUES (2);
      CREATE TABLE workers (
        id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        capabilities_json TEXT NOT NULL,
        reliability REAL NOT NULL DEFAULT 0.95,
        jobs_completed INTEGER NOT NULL DEFAULT 0,
        last_seen_at INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );
      INSERT INTO workers(
        id, status, capabilities_json, last_seen_at, created_at, updated_at
      ) VALUES ('wrk-existing', 'offline', '{}', 1, 1, 1);
    `);
    legacy.close();

    const migrated = new MeshDatabase(path);
    const version = migrated.raw.prepare("SELECT version FROM schema_meta").get() as {
      version: number;
    };
    const row = migrated.raw
      .prepare("SELECT id, deregistered FROM workers WHERE id = 'wrk-existing'")
      .get() as { id: string; deregistered: number };
    expect(version.version).toBe(26);
    expect(row).toEqual({ id: "wrk-existing", deregistered: 1 });
    migrated.close();
  });

  it("finishes an interrupted version 2 migration when deregistered already exists", () => {
    const directory = mkdtempSync(join(tmpdir(), "gpu-mesh-v2-partial-migration-"));
    directories.push(directory);
    const path = join(directory, "mesh.db");
    const legacy = new DatabaseSync(path);
    legacy.exec(`
      CREATE TABLE schema_meta (version INTEGER NOT NULL);
      INSERT INTO schema_meta(version) VALUES (2);
      CREATE TABLE workers (
        id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        capabilities_json TEXT NOT NULL,
        reliability REAL NOT NULL DEFAULT 0.95,
        jobs_completed INTEGER NOT NULL DEFAULT 0,
        last_seen_at INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        deregistered INTEGER NOT NULL DEFAULT 0
      );
      INSERT INTO workers(
        id, status, capabilities_json, last_seen_at, created_at, updated_at, deregistered
      ) VALUES ('wrk-partially-migrated', 'offline', '{}', 1, 1, 1, 0);
    `);
    legacy.close();

    const migrated = new MeshDatabase(path);
    const version = migrated.raw.prepare("SELECT version FROM schema_meta").get() as {
      version: number;
    };
    const row = migrated.raw
      .prepare("SELECT id, deregistered FROM workers WHERE id = 'wrk-partially-migrated'")
      .get() as { id: string; deregistered: number };
    expect(version.version).toBe(26);
    expect(row).toEqual({ id: "wrk-partially-migrated", deregistered: 1 });
    migrated.close();

    const reopened = new MeshDatabase(path);
    expect(
      (reopened.raw.prepare("SELECT version FROM schema_meta").get() as { version: number })
        .version,
    ).toBe(26);
    reopened.close();
  });

  it("adds durable activation failures when migrating a version 4 model catalog", () => {
    const directory = mkdtempSync(join(tmpdir(), "gpu-mesh-v4-model-migration-"));
    directories.push(directory);
    const path = join(directory, "mesh.db");
    const legacy = new DatabaseSync(path);
    legacy.exec(`
      CREATE TABLE schema_meta (version INTEGER NOT NULL);
      INSERT INTO schema_meta(version) VALUES (4);
      CREATE TABLE requested_models (
        id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        revision TEXT,
        context_tokens INTEGER NOT NULL,
        minimum_nodes INTEGER NOT NULL,
        auto_activate INTEGER NOT NULL DEFAULT 1,
        profile_json TEXT,
        profile_error TEXT,
        activation_requested_at INTEGER,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );
      INSERT INTO requested_models VALUES (
        'qwen-ui', 'Qwen/Qwen3-0.6B', NULL, 4096, 2, 1, NULL, NULL, NULL, 1, 1
      );
    `);
    legacy.close();

    const migrated = new MeshDatabase(path);
    const columns = migrated.raw.prepare("PRAGMA table_info(requested_models)").all() as Array<{ name: string }>;
    expect(columns.some((column) => column.name === "activation_error")).toBe(true);
    expect(
      (migrated.raw.prepare("SELECT version FROM schema_meta").get() as { version: number }).version,
    ).toBe(26);
    migrated.close();
  });

  it("adds physical contribution binding when migrating the version 18 economic ledger", () => {
    const directory = mkdtempSync(join(tmpdir(), "mycellios-v18-economy-migration-"));
    directories.push(directory);
    const path = join(directory, "mesh.db");
    const legacy = new DatabaseSync(path);
    legacy.exec(`
      CREATE TABLE schema_meta (version INTEGER NOT NULL);
      INSERT INTO schema_meta(version) VALUES (18);
      CREATE TABLE economic_settlements (
        id TEXT PRIMARY KEY, job_id TEXT NOT NULL UNIQUE, execution_receipt_id TEXT NOT NULL UNIQUE,
        pricing_policy_id TEXT NOT NULL, payer_account_id TEXT NOT NULL,
        input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,
        gross_microunits INTEGER NOT NULL, provider_microunits INTEGER NOT NULL,
        platform_microunits INTEGER NOT NULL, request_digest TEXT NOT NULL, created_at INTEGER NOT NULL
      );
    `);
    legacy.close();

    const migrated = new MeshDatabase(path);
    const columns = migrated.raw.prepare("PRAGMA table_info(economic_settlements)").all() as Array<{ name: string }>;
    expect(columns.some(({ name }) => name === "contribution_evidence_id")).toBe(true);
    expect((migrated.raw.prepare("SELECT version FROM schema_meta").get() as { version: number }).version).toBe(26);
    migrated.close();
  });

  it("backfills desktop identities and retires duplicate legacy rows", () => {
    const directory = mkdtempSync(join(tmpdir(), "gpu-mesh-v5-identity-migration-"));
    directories.push(directory);
    const path = join(directory, "mesh.db");
    const legacy = new DatabaseSync(path);
    legacy.exec(`
      CREATE TABLE schema_meta (version INTEGER NOT NULL);
      INSERT INTO schema_meta(version) VALUES (5);
      CREATE TABLE workers (
        id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        capabilities_json TEXT NOT NULL,
        reliability REAL NOT NULL DEFAULT 0.95,
        jobs_completed INTEGER NOT NULL DEFAULT 0,
        last_seen_at INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        deregistered INTEGER NOT NULL DEFAULT 0
      );
    `);
    const capabilities = JSON.stringify({
      distributedExecutor: { nodeId: "desktop-migrated", protocol: "gdlp-worker-tunnel/1" },
    });
    const insert = legacy.prepare(`
      INSERT INTO workers(
        id, status, capabilities_json, last_seen_at, created_at, updated_at, deregistered
      ) VALUES (?, ?, ?, ?, 1, ?, 0)
    `);
    insert.run("wrk-old", "offline", capabilities, 10, 10);
    insert.run("wrk-current", "online", capabilities, 20, 20);
    legacy.close();

    const migrated = new MeshDatabase(path);
    const rows = migrated.raw.prepare(`
      SELECT id, deregistered, identity_kind, identity_id
      FROM workers ORDER BY id
    `).all() as Array<{
      id: string;
      deregistered: number;
      identity_kind: string | null;
      identity_id: string | null;
    }>;
    expect(rows).toEqual([
      {
        id: "wrk-current",
        deregistered: 0,
        identity_kind: "device",
        identity_id: "desktop-migrated",
      },
      {
        id: "wrk-old",
        deregistered: 1,
        identity_kind: null,
        identity_id: null,
      },
    ]);
    migrated.close();
  });
});
