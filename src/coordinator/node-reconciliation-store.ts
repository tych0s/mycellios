import { createHash } from "node:crypto";
import {
  NODE_CONTROL_MAX_CLOCK_SKEW_MS,
  nodeDesiredStateSchema,
  nodeSnapshotSchema,
  type NodeDesiredState,
  type NodeEvent,
  type NodeSnapshot,
} from "../contracts/node-control.js";
import type { MeshDatabase } from "../storage/database.js";
import type { NodeCommandStore } from "./node-command-store.js";
import { DEFAULT_NODE_WORK_POLICY } from "../contracts/node-work-policy.js";

export interface NodeReconciliationResult {
  snapshot: NodeSnapshot;
  desiredState: NodeDesiredState;
  events: NodeEvent[];
  latestCursor: string;
}

export class NodeReconciliationStore {
  constructor(
    private readonly database: MeshDatabase,
    private readonly commands: NodeCommandStore,
    private readonly now: () => number = Date.now,
  ) {}

  reconcile(input: unknown): NodeReconciliationResult {
    const snapshot = nodeSnapshotSchema.parse(input);
    const observedAt = Date.parse(snapshot.observedAt);
    if (observedAt > this.now() + NODE_CONTROL_MAX_CLOCK_SKEW_MS) throw new Error("node_snapshot_observed_in_future");
    return this.database.transaction(() => {
      const cursorSequence = this.commands.cursorSequence(snapshot.nodeId, snapshot.cursor);
      const previous = this.database.raw.prepare(
        "SELECT generation, cursor_sequence, observed_at FROM node_snapshots WHERE node_id = ?",
      ).get(snapshot.nodeId) as { generation: number; cursor_sequence: number; observed_at: number } | undefined;
      if (previous) {
        if (snapshot.generation < previous.generation) throw new Error("node_snapshot_generation_downgrade");
        if (snapshot.generation === previous.generation && cursorSequence < previous.cursor_sequence) throw new Error("node_snapshot_cursor_regression");
        if (snapshot.generation === previous.generation && observedAt < previous.observed_at) throw new Error("node_snapshot_time_regression");
      }
      this.assertActiveCommands(snapshot);
      const desiredState = this.readOrSeedDesired(snapshot);
      const serialized = canonicalJson(snapshot);
      const receivedAt = this.now();
      this.database.raw.prepare(
        `INSERT INTO node_snapshots(node_id, generation, cursor_sequence, snapshot_json, observed_at, received_at)
         VALUES (?, ?, ?, ?, ?, ?)
         ON CONFLICT(node_id) DO UPDATE SET
           generation = excluded.generation,
           cursor_sequence = excluded.cursor_sequence,
           snapshot_json = excluded.snapshot_json,
           observed_at = excluded.observed_at,
           received_at = excluded.received_at`,
      ).run(snapshot.nodeId, snapshot.generation, cursorSequence, serialized, observedAt, receivedAt);
      const snapshotDigest = digest(serialized);
      this.database.raw.prepare(
        `INSERT OR IGNORE INTO node_snapshot_history(
           node_id, generation, cursor_sequence, snapshot_digest, snapshot_json, received_at
         ) VALUES (?, ?, ?, ?, ?, ?)`,
      ).run(snapshot.nodeId, snapshot.generation, cursorSequence, snapshotDigest, serialized, receivedAt);
      return {
        snapshot,
        desiredState,
        events: this.commands.eventsAfter(snapshot.nodeId, snapshot.cursor),
        latestCursor: this.commands.latestCursor(snapshot.nodeId),
      };
    });
  }

  desiredState(nodeId: string): NodeDesiredState | null {
    const row = this.database.raw.prepare(
      "SELECT desired_json FROM node_desired_states WHERE node_id = ?",
    ).get(nodeId) as { desired_json: string } | undefined;
    return row ? nodeDesiredStateSchema.parse(JSON.parse(row.desired_json)) : null;
  }

  latestSnapshot(nodeId: string): NodeSnapshot | null {
    const row = this.database.raw.prepare(
      "SELECT snapshot_json FROM node_snapshots WHERE node_id = ?",
    ).get(nodeId) as { snapshot_json: string } | undefined;
    return row ? nodeSnapshotSchema.parse(JSON.parse(row.snapshot_json)) : null;
  }

  private readOrSeedDesired(snapshot: NodeSnapshot): NodeDesiredState {
    const existing = this.desiredState(snapshot.nodeId);
    if (existing?.generation === snapshot.generation) return existing;
    if (existing && existing.generation > snapshot.generation) throw new Error("node_desired_state_generation_downgrade");
    const desired = nodeDesiredStateSchema.parse({
      schema: "mycellios-node-desired-state/1",
      nodeId: snapshot.nodeId,
      generation: snapshot.generation,
      contributionEnabled: snapshot.contributionEnabled,
      drain: snapshot.draining,
      updateChannel: snapshot.updateChannel,
      limits: snapshot.limits,
      policy: snapshot.policy ?? DEFAULT_NODE_WORK_POLICY,
      updatedAt: snapshot.observedAt,
      updatedBy: { kind: "node", id: snapshot.nodeId, scopes: ["node:read"] },
    });
    this.database.raw.prepare(
      `INSERT INTO node_desired_states(node_id, generation, desired_json, updated_at) VALUES (?, ?, ?, ?)
       ON CONFLICT(node_id) DO UPDATE SET
         generation = excluded.generation,
         desired_json = excluded.desired_json,
         updated_at = excluded.updated_at`,
    ).run(snapshot.nodeId, snapshot.generation, canonicalJson(desired), this.now());
    this.commands.replayDesiredCommands(snapshot.nodeId, snapshot.generation);
    return this.desiredState(snapshot.nodeId)!;
  }

  private assertActiveCommands(snapshot: NodeSnapshot): void {
    if (snapshot.activeCommandIds.length === 0) return;
    const placeholders = snapshot.activeCommandIds.map(() => "?").join(",");
    const rows = this.database.raw.prepare(
      `SELECT id, node_id, generation, state FROM node_commands WHERE id IN (${placeholders})`,
    ).all(...snapshot.activeCommandIds) as Array<{ id: string; node_id: string; generation: number; state: string }>;
    const valid = new Set(rows.filter((row) =>
      row.node_id === snapshot.nodeId
      && row.generation === snapshot.generation
      && (row.state === "queued" || row.state === "delivered")
    ).map(({ id }) => id));
    if (valid.size !== snapshot.activeCommandIds.length) throw new Error("node_snapshot_active_commands_invalid");
  }
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function digest(value: string): `sha256:${string}` {
  return `sha256:${createHash("sha256").update(value, "utf8").digest("hex")}`;
}
