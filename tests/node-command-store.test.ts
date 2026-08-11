import { randomUUID } from "node:crypto";
import { describe, expect, it } from "vitest";
import { NodeCommandStore } from "../src/coordinator/node-command-store.js";
import { MeshDatabase } from "../src/storage/database.js";

const actor = {
  kind: "account" as const,
  id: "account-1",
  scopes: ["node:read", "node:control"] as const,
};

function command(overrides: Record<string, unknown> = {}) {
  return {
    schema: "mycellios-node-command/1",
    id: randomUUID(),
    nodeId: "node-1",
    actor,
    generation: 3,
    issuedAt: "2026-08-10T12:00:00.000Z",
    expiresAt: "2026-08-10T12:05:00.000Z",
    nonce: `nonce_${randomUUID().replaceAll("-", "")}`,
    type: "pause",
    payload: { version: 1 },
    ...overrides,
  };
}

function result(commandId: string, overrides: Record<string, unknown> = {}) {
  return {
    schema: "mycellios-node-command-result/1",
    id: randomUUID(),
    commandId,
    nodeId: "node-1",
    generation: 3,
    state: "applied",
    observedAt: "2026-08-10T12:00:03.000Z",
    resultDigest: `sha256:${"a".repeat(64)}`,
    error: null,
    ...overrides,
  };
}

describe("NodeCommandStore", () => {
  it("redelivers after reconnect and records an applied result exactly once", () => {
    const database = new MeshDatabase(":memory:");
    const now = () => Date.parse("2026-08-10T12:00:01.000Z");
    const first = new NodeCommandStore(database, now);
    const input = command();

    expect(first.enqueue(input, 3).state).toBe("queued");
    expect(first.enqueue(input, 3).state).toBe("queued");
    expect(first.pull("node-1", 3)).toMatchObject([{ state: "delivered", command: { id: input.id } }]);

    const restarted = new NodeCommandStore(database, now);
    expect(restarted.pull("node-1", 3)).toMatchObject([{ state: "delivered", command: { id: input.id } }]);
    const applied = result(input.id);
    expect(restarted.recordResult(applied)).toEqual(applied);
    expect(restarted.recordResult(applied)).toEqual(applied);
    expect(restarted.pull("node-1", 3)).toEqual([]);
    expect(() => restarted.recordResult({ ...applied, resultDigest: `sha256:${"b".repeat(64)}` })).toThrow("node_command_result_conflict");

    const events = restarted.eventsAfter("node-1", null);
    expect(events.map(({ type }) => type)).toEqual([
      "command.queued",
      "command.delivered",
      "command.applied",
    ]);
    expect(events.at(-1)?.actor).toMatchObject({ kind: "node", id: "node-1" });
    expect(restarted.eventsAfter("node-1", events[1]!.cursor).map(({ type }) => type)).toEqual(["command.applied"]);
    const forgedCursor = `${events[1]!.cursor.slice(0, -1)}${events[1]!.cursor.endsWith("0") ? "1" : "0"}`;
    expect(() => restarted.eventsAfter("node-1", forgedCursor)).toThrow("node_event_cursor_unknown");
    database.close();
  });

  it("rejects nonce replay, stale generations and conflicting command ids", () => {
    const database = new MeshDatabase(":memory:");
    const store = new NodeCommandStore(database, () => Date.parse("2026-08-10T12:00:01.000Z"));
    const input = command();
    store.enqueue(input, 3);
    expect(() => store.enqueue({ ...input, type: "resume" }, 3)).toThrow("node_command_id_conflict");
    expect(() => store.enqueue(command({ nonce: input.nonce }), 3)).toThrow("node_command_replay");
    expect(() => store.enqueue(command({ generation: 2 }), 3)).toThrow("node_command_generation_downgrade");
    database.close();
  });

  it("expires commands before delivery and retires old generations", () => {
    let clock = Date.parse("2026-08-10T12:00:01.000Z");
    const database = new MeshDatabase(":memory:");
    const store = new NodeCommandStore(database, () => clock);
    const expiring = command({ expiresAt: "2026-08-10T12:00:02.000Z" });
    store.enqueue(expiring, 3);
    clock += 2_000;
    expect(store.pull("node-1", 3)).toEqual([]);

    const oldGeneration = command({ generation: 3 });
    store.enqueue(oldGeneration, 3);
    expect(store.pull("node-1", 4)).toEqual([]);
    expect(store.eventsAfter("node-1", null).filter(({ type }) => type === "command.expired")).toHaveLength(2);
    database.close();
  });

  it("requires terminal, matching results", () => {
    const database = new MeshDatabase(":memory:");
    const store = new NodeCommandStore(database, () => Date.parse("2026-08-10T12:00:01.000Z"));
    const input = command();
    store.enqueue(input, 3);
    expect(() => store.recordResult(result(input.id, { state: "accepted" }))).toThrow("node_command_result_not_terminal");
    expect(() => store.recordResult(result(input.id, { nodeId: "node-2" }))).toThrow("node_command_result_wrong_node");
    expect(() => store.recordResult(result(input.id, { generation: 4 }))).toThrow("node_command_result_wrong_generation");
    database.close();
  });
});
