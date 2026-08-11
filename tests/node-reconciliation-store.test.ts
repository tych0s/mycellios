import { randomUUID } from "node:crypto";
import { describe, expect, it } from "vitest";
import { NODE_EVENT_ORIGIN_CURSOR } from "../src/contracts/node-control.js";
import { NodeCommandStore } from "../src/coordinator/node-command-store.js";
import { NodeReconciliationStore } from "../src/coordinator/node-reconciliation-store.js";
import { MeshDatabase } from "../src/storage/database.js";

const observedAt = "2026-08-10T12:00:00.000Z";
const actor = { kind: "account" as const, id: "account-1", scopes: ["node:control"] as const };

function snapshot(overrides: Record<string, unknown> = {}) {
  return {
    schema: "mycellios-node-snapshot/1",
    nodeId: "node-1",
    generation: 3,
    cursor: NODE_EVENT_ORIGIN_CURSOR,
    observedAt,
    state: "ready",
    contributionEnabled: true,
    draining: false,
    updateChannel: "stable",
    limits: { maxConcurrency: 2, maxCpuPercent: 80, maxRamMiB: 8_192, maxVramMiB: 6_144, maxDiskMiB: 16_384, maxTemperatureC: 85 },
    activeCommandIds: [],
    build: { version: "1.0.0", sourceRevision: "a".repeat(40) },
    runtime: { ready: true, abi: "runtime-v1", backend: "cuda" },
    ...overrides,
  };
}

function pause() {
  return {
    schema: "mycellios-node-command/1",
    id: randomUUID(),
    nodeId: "node-1",
    actor,
    generation: 3,
    issuedAt: "2026-08-10T12:00:01.000Z",
    expiresAt: "2026-08-10T12:05:00.000Z",
    nonce: `nonce_${randomUUID().replaceAll("-", "")}`,
    type: "pause",
    payload: { version: 1 },
  };
}

describe("NodeReconciliationStore", () => {
  it("seeds desired state and replays every missing event after loss or restart", () => {
    const database = new MeshDatabase(":memory:");
    let clock = Date.parse(observedAt);
    const commands = new NodeCommandStore(database, () => clock);
    const first = new NodeReconciliationStore(database, commands, () => clock);
    expect(first.reconcile(snapshot())).toMatchObject({
      latestCursor: NODE_EVENT_ORIGIN_CURSOR,
      events: [],
      desiredState: { contributionEnabled: true, drain: false, updateChannel: "stable" },
    });

    clock += 1_000;
    const input = pause();
    commands.enqueue(input, 3);
    commands.pull("node-1", 3);
    const afterLoss = new NodeReconciliationStore(database, new NodeCommandStore(database, () => clock), () => clock)
      .reconcile(snapshot({ observedAt: new Date(clock).toISOString(), activeCommandIds: [input.id] }));
    expect(afterLoss.desiredState.contributionEnabled).toBe(false);
    expect(afterLoss.events.map(({ type }) => type)).toEqual(["command.queued", "command.delivered"]);
    expect(afterLoss.latestCursor).toBe(afterLoss.events.at(-1)?.cursor);

    const fromFirstEvent = first.reconcile(snapshot({
      observedAt: new Date(clock).toISOString(),
      cursor: afterLoss.events[0]!.cursor,
      activeCommandIds: [input.id],
    }));
    expect(fromFirstEvent.events.map(({ type }) => type)).toEqual(["command.delivered"]);
    database.close();
  });

  it("rejects forged, regressed and cross-generation observations and stores duplicates idempotently", () => {
    const database = new MeshDatabase(":memory:");
    let clock = Date.parse(observedAt);
    const commands = new NodeCommandStore(database, () => clock);
    const reconciliation = new NodeReconciliationStore(database, commands, () => clock);
    reconciliation.reconcile(snapshot());
    reconciliation.reconcile(snapshot());
    expect((database.raw.prepare("SELECT COUNT(*) AS count FROM node_snapshot_history").get() as { count: number }).count).toBe(1);

    clock += 1_000;
    const input = pause();
    commands.enqueue(input, 3);
    const latest = commands.latestCursor("node-1");
    const forged = `${latest.slice(0, -1)}${latest.endsWith("0") ? "1" : "0"}`;
    reconciliation.reconcile(snapshot({ cursor: latest, observedAt: new Date(clock).toISOString(), activeCommandIds: [input.id] }));
    expect(() => reconciliation.reconcile(snapshot({ observedAt: new Date(clock).toISOString(), activeCommandIds: [input.id] }))).toThrow("node_snapshot_cursor_regression");
    expect(() => reconciliation.reconcile(snapshot({ cursor: forged, observedAt: new Date(clock).toISOString() }))).toThrow(/node_event_cursor/);
    expect(() => reconciliation.reconcile(snapshot({ generation: 2, cursor: latest, observedAt: new Date(clock).toISOString() }))).toThrow("node_snapshot_generation_downgrade");
    expect(() => reconciliation.reconcile(snapshot({ cursor: latest, observedAt, activeCommandIds: [input.id] }))).toThrow("node_snapshot_time_regression");
    expect(() => reconciliation.reconcile(snapshot({ cursor: latest, observedAt: new Date(clock).toISOString(), activeCommandIds: [randomUUID()] }))).toThrow("node_snapshot_active_commands_invalid");
    database.close();
  });

  it("rotates desired state atomically when ownership advances to a new generation", () => {
    const database = new MeshDatabase(":memory:");
    const reconciliation = new NodeReconciliationStore(database, new NodeCommandStore(database), () => Date.parse(observedAt));
    reconciliation.reconcile(snapshot());
    const rotated = reconciliation.reconcile(snapshot({ generation: 4, contributionEnabled: false }));
    expect(rotated.desiredState).toMatchObject({ generation: 4, contributionEnabled: false });
    expect(reconciliation.desiredState("node-1")?.generation).toBe(4);
    database.close();
  });

  it("projects commands queued before the first snapshot into desired state", () => {
    const database = new MeshDatabase(":memory:");
    const commands = new NodeCommandStore(database, () => Date.parse("2026-08-10T12:00:01.000Z"));
    commands.enqueue(pause(), 3);
    const result = new NodeReconciliationStore(database, commands, () => Date.parse("2026-08-10T12:00:01.000Z"))
      .reconcile(snapshot({ observedAt: "2026-08-10T12:00:01.000Z" }));
    expect(result.desiredState.contributionEnabled).toBe(false);
    database.close();
  });

  it("projects schedule and model policy into durable desired state", () => {
    const database = new MeshDatabase(":memory:");
    const now = () => Date.parse("2026-08-10T12:00:01.000Z");
    const commands = new NodeCommandStore(database, now);
    const reconciliation = new NodeReconciliationStore(database, commands, now);
    reconciliation.reconcile(snapshot());
    const policy = { schedule: [{ days: [1, 2, 3, 4, 5], startMinuteUtc: 480, endMinuteUtc: 1080 }], modelAllowlist: ["org/model"] };
    commands.enqueue({ ...pause(), id: randomUUID(), nonce: `nonce_${randomUUID().replaceAll("-", "")}`,
      actor: { kind: "account", id: "account-1", scopes: ["node:limits"] }, type: "set-policy", payload: { version: 1, policy } }, 3);
    expect(reconciliation.desiredState("node-1")?.policy).toEqual(policy);
    database.close();
  });
});
