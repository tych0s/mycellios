import { createHash } from "node:crypto";
import { z } from "zod";
import {
  nodeCommandResultSchema,
  nodeCommandSchema,
  nodeControlActorSchema,
  nodeDesiredStateSchema,
  nodeEventSchema,
  NODE_EVENT_ORIGIN_CURSOR,
  parseAuthorizedNodeCommand,
  type NodeCommand,
  type NodeCommandResult,
  type NodeEvent,
} from "../contracts/node-control.js";
import type { MeshDatabase } from "../storage/database.js";

const commandStateSchema = z.enum(["queued", "delivered", "applied", "rejected", "expired"]);
type CommandState = z.infer<typeof commandStateSchema>;

interface CommandRow {
  id: string;
  node_id: string;
  generation: number;
  nonce: string;
  command_json: string;
  state: CommandState;
  expires_at: number;
}

export interface DurableNodeCommand {
  command: NodeCommand;
  state: CommandState;
}

export class NodeCommandStore {
  constructor(
    private readonly database: MeshDatabase,
    private readonly now: () => number = Date.now,
  ) {}

  enqueue(input: unknown, minimumGeneration: number): DurableNodeCommand {
    const parsed = nodeCommandSchema.parse(input);
    return this.database.transaction(() => {
      const existing = this.readCommand(parsed.id);
      if (existing) {
        if (existing.command_json !== canonicalJson(parsed)) throw new Error("node_command_id_conflict");
        return this.toDurable(existing);
      }
      const consumedNonces = new Set(
        (this.database.raw.prepare("SELECT nonce FROM node_commands WHERE node_id = ?").all(parsed.nodeId) as Array<{ nonce: string }>).map(({ nonce }) => nonce),
      );
      const command = parseAuthorizedNodeCommand(parsed, {
        now: new Date(this.now()),
        minimumGeneration,
        consumedNonces,
        expectedNodeId: parsed.nodeId,
      });
      const createdAt = this.now();
      this.database.raw.prepare(
        `INSERT INTO node_commands(
           id, node_id, generation, nonce, command_json, state,
           issued_at, expires_at, created_at
         ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?)`,
      ).run(
        command.id,
        command.nodeId,
        command.generation,
        command.nonce,
        canonicalJson(command),
        Date.parse(command.issuedAt),
        Date.parse(command.expiresAt),
        createdAt,
      );
      this.updateDesiredState(command);
      this.appendEvent(command.nodeId, command.generation, command.actor, "command.queued", command);
      return { command, state: "queued" };
    });
  }

  pull(nodeId: string, generation: number, limit = 32): DurableNodeCommand[] {
    if (!Number.isSafeInteger(generation) || generation < 1) throw new Error("node_command_generation_invalid");
    return this.database.transaction(() => {
      this.expirePending(nodeId, generation);
      const rows = this.database.raw.prepare(
        `SELECT id, node_id, generation, nonce, command_json, state, expires_at
         FROM node_commands
         WHERE node_id = ? AND generation = ? AND state IN ('queued', 'delivered')
         ORDER BY created_at, id LIMIT ?`,
      ).all(nodeId, generation, Math.max(1, Math.min(256, Math.trunc(limit)))) as unknown as CommandRow[];
      const deliveredAt = this.now();
      for (const row of rows) {
        if (row.state !== "queued") continue;
        this.database.raw.prepare(
          "UPDATE node_commands SET state = 'delivered', delivered_at = ? WHERE id = ? AND state = 'queued'",
        ).run(deliveredAt, row.id);
        const command = nodeCommandSchema.parse(JSON.parse(row.command_json));
        this.appendEvent(nodeId, generation, command.actor, "command.delivered", { commandId: row.id });
        row.state = "delivered";
      }
      return rows.map((row) => this.toDurable(row));
    });
  }

  recordResult(input: unknown): NodeCommandResult {
    const result = nodeCommandResultSchema.parse(input);
    if (result.state === "accepted") throw new Error("node_command_result_not_terminal");
    return this.database.transaction(() => {
      const commandRow = this.readCommand(result.commandId);
      if (!commandRow) throw new Error("node_command_unknown");
      if (commandRow.node_id !== result.nodeId) throw new Error("node_command_result_wrong_node");
      if (commandRow.generation !== result.generation) throw new Error("node_command_result_wrong_generation");
      const json = canonicalJson(result);
      const existing = this.database.raw.prepare(
        "SELECT result_json FROM node_command_results WHERE command_id = ?",
      ).get(result.commandId) as { result_json: string } | undefined;
      if (existing) {
        if (existing.result_json !== json) throw new Error("node_command_result_conflict");
        return nodeCommandResultSchema.parse(JSON.parse(existing.result_json));
      }
      const finalState = commandStateSchema.parse(result.state);
      this.database.raw.prepare(
        `INSERT INTO node_command_results(
           id, command_id, node_id, generation, result_digest, result_json, observed_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
      ).run(result.id, result.commandId, result.nodeId, result.generation, result.resultDigest, json, Date.parse(result.observedAt));
      this.database.raw.prepare(
        `UPDATE node_commands SET state = ?, completed_at = ?
         WHERE id = ? AND state IN ('queued', 'delivered')`,
      ).run(finalState, this.now(), result.commandId);
      this.appendEvent(result.nodeId, result.generation, {
        kind: "node",
        id: result.nodeId,
        scopes: ["node:read"],
      }, `command.${finalState}`, result);
      return result;
    });
  }

  eventsAfter(nodeId: string, cursor: string | null, limit = 100): NodeEvent[] {
    const after = cursor === null ? 0 : this.cursorSequence(nodeId, cursor);
    const rows = this.database.raw.prepare(
      `SELECT sequence, node_id, generation, actor_json, event_type, payload_digest,
              previous_event_digest, event_digest, occurred_at
       FROM node_control_events WHERE node_id = ? AND sequence > ?
       ORDER BY sequence LIMIT ?`,
    ).all(nodeId, after, Math.max(1, Math.min(1_000, Math.trunc(limit)))) as Array<Record<string, unknown>>;
    return rows.map((row) => nodeEventSchema.parse({
      schema: "mycellios-node-event/1",
      cursor: formatCursor(Number(row.sequence), String(row.event_digest)),
      nodeId: String(row.node_id),
      generation: Number(row.generation),
      actor: JSON.parse(String(row.actor_json)),
      type: String(row.event_type),
      occurredAt: new Date(Number(row.occurred_at)).toISOString(),
      payloadDigest: String(row.payload_digest),
      previousEventDigest: row.previous_event_digest === null ? null : String(row.previous_event_digest),
    }));
  }

  cursorSequence(nodeId: string, cursor: string): number {
    return cursor === NODE_EVENT_ORIGIN_CURSOR ? 0 : this.validateCursor(nodeId, cursor);
  }

  latestCursor(nodeId: string): string {
    const row = this.database.raw.prepare(
      "SELECT sequence, event_digest FROM node_control_events WHERE node_id = ? ORDER BY sequence DESC LIMIT 1",
    ).get(nodeId) as { sequence: number; event_digest: string } | undefined;
    return row ? formatCursor(Number(row.sequence), row.event_digest) : NODE_EVENT_ORIGIN_CURSOR;
  }

  acknowledgedCommandIds(nodeId: string, generation: number, limit = 256): string[] {
    return (this.database.raw.prepare(
      `SELECT id FROM node_commands WHERE node_id = ? AND generation = ? AND state = 'applied'
       ORDER BY completed_at DESC, id DESC LIMIT ?`,
    ).all(nodeId, generation, Math.max(1, Math.min(1_000, Math.trunc(limit)))) as Array<{ id: string }>).map(({ id }) => id);
  }

  replayDesiredCommands(nodeId: string, generation: number): void {
    const rows = this.database.raw.prepare(
      `SELECT command_json FROM node_commands
       WHERE node_id = ? AND generation = ? AND state != 'expired'
       ORDER BY created_at, id`,
    ).all(nodeId, generation) as Array<{ command_json: string }>;
    for (const row of rows) this.updateDesiredState(nodeCommandSchema.parse(JSON.parse(row.command_json)));
  }

  private validateCursor(nodeId: string, cursor: string): number {
    const sequence = parseCursorSequence(cursor);
    const row = this.database.raw.prepare(
      "SELECT event_digest FROM node_control_events WHERE node_id = ? AND sequence = ?",
    ).get(nodeId, sequence) as { event_digest: string } | undefined;
    if (!row || formatCursor(sequence, row.event_digest) !== cursor) {
      throw new Error("node_event_cursor_unknown");
    }
    return sequence;
  }

  private expirePending(nodeId: string, generation: number): void {
    const rows = this.database.raw.prepare(
      `SELECT id, node_id, generation, nonce, command_json, state, expires_at
       FROM node_commands
       WHERE node_id = ? AND state IN ('queued', 'delivered')
         AND (expires_at <= ? OR generation < ?)`,
    ).all(nodeId, this.now(), generation) as unknown as CommandRow[];
    for (const row of rows) {
      this.database.raw.prepare(
        "UPDATE node_commands SET state = 'expired', completed_at = ? WHERE id = ? AND state IN ('queued', 'delivered')",
      ).run(this.now(), row.id);
      const command = nodeCommandSchema.parse(JSON.parse(row.command_json));
      this.appendEvent(nodeId, row.generation, command.actor, "command.expired", { commandId: row.id });
    }
  }

  private updateDesiredState(command: NodeCommand): void {
    const row = this.database.raw.prepare(
      "SELECT desired_json FROM node_desired_states WHERE node_id = ?",
    ).get(command.nodeId) as { desired_json: string } | undefined;
    if (!row) return;
    const current = nodeDesiredStateSchema.parse(JSON.parse(row.desired_json));
    if (current.generation !== command.generation) throw new Error("node_desired_state_generation_mismatch");
    const next = nodeDesiredStateSchema.parse({
      ...current,
      ...(command.type === "pause" ? { contributionEnabled: false }
        : command.type === "resume" ? { contributionEnabled: true, drain: false }
          : command.type === "drain" ? { drain: true }
            : command.type === "set-limits" ? (() => { const { version: _version, ...limits } = command.payload; return { limits }; })()
              : command.type === "set-policy" ? { policy: command.payload.policy }
              : command.type === "update" ? { updateChannel: command.payload.channel }
                : command.type === "revoke" || command.type === "uninstall" ? { contributionEnabled: false, drain: true }
                  : {}),
      updatedAt: new Date(this.now()).toISOString(),
      updatedBy: command.actor,
    });
    this.database.raw.prepare(
      "UPDATE node_desired_states SET desired_json = ?, updated_at = ? WHERE node_id = ? AND generation = ?",
    ).run(canonicalJson(next), this.now(), command.nodeId, command.generation);
  }

  private readCommand(id: string): CommandRow | null {
    const row = this.database.raw.prepare(
      `SELECT id, node_id, generation, nonce, command_json, state, expires_at
       FROM node_commands WHERE id = ?`,
    ).get(id) as CommandRow | undefined;
    return row ?? null;
  }

  private toDurable(row: CommandRow): DurableNodeCommand {
    return { command: nodeCommandSchema.parse(JSON.parse(row.command_json)), state: commandStateSchema.parse(row.state) };
  }

  private appendEvent(
    nodeId: string,
    generation: number,
    actorInput: unknown,
    eventType: string,
    payload: unknown,
  ): void {
    const actor = nodeControlActorSchema.parse(actorInput);
    const previous = this.database.raw.prepare(
      "SELECT event_digest FROM node_control_events WHERE node_id = ? ORDER BY sequence DESC LIMIT 1",
    ).get(nodeId) as { event_digest: string } | undefined;
    const occurredAt = this.now();
    const payloadDigest = digest(canonicalJson(payload));
    const eventDigest = digest(canonicalJson({
      nodeId,
      generation,
      actor,
      eventType,
      payloadDigest,
      previousEventDigest: previous?.event_digest ?? null,
      occurredAt,
    }));
    this.database.raw.prepare(
      `INSERT INTO node_control_events(
         node_id, generation, actor_json, event_type, payload_digest,
         previous_event_digest, event_digest, occurred_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(nodeId, generation, canonicalJson(actor), eventType, payloadDigest, previous?.event_digest ?? null, eventDigest, occurredAt);
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

function formatCursor(sequence: number, eventDigest: string): string {
  return `evt_${sequence}_${eventDigest.slice("sha256:".length, "sha256:".length + 16)}`;
}

function parseCursorSequence(cursor: string): number {
  const match = /^evt_([0-9]{1,20})_[a-f0-9]{16}$/.exec(cursor);
  if (!match) throw new Error("node_event_cursor_invalid");
  const sequence = Number(match[1]);
  if (!Number.isSafeInteger(sequence)) throw new Error("node_event_cursor_invalid");
  return sequence;
}
