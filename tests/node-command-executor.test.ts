import { randomUUID } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  NodeCommandExecutionError,
  NodeCommandExecutionInterruptedError,
  NodeCommandExecutor,
} from "../src/node/command-executor.js";

const cleanup: string[] = [];
afterEach(async () => Promise.all(cleanup.splice(0).map((path) => rm(path, { recursive: true, force: true }))));

function command(overrides: Record<string, unknown> = {}) {
  return {
    schema: "mycellios-node-command/1",
    id: randomUUID(),
    nodeId: "node-1",
    actor: { kind: "account", id: "account-1", scopes: ["node:control"] },
    generation: 7,
    issuedAt: "2026-08-10T12:00:00.000Z",
    expiresAt: "2026-08-10T12:05:00.000Z",
    nonce: `nonce_${randomUUID().replaceAll("-", "")}`,
    type: "pause",
    payload: { version: 1 },
    ...overrides,
  };
}

describe("NodeCommandExecutor", () => {
  it("replays an interrupted declarative mutation and seals one durable result", async () => {
    const path = await journalPath();
    const input = command();
    let contributionEnabled = true;
    let attempts = 0;
    const first = new NodeCommandExecutor(path, "node-1", 7, clock);

    await expect(first.execute(input, async ({ operationKey }) => {
      expect(operationKey).toBe(`node-command:${input.id}`);
      attempts += 1;
      contributionEnabled = false;
      throw new NodeCommandExecutionInterruptedError();
    })).rejects.toThrow("node_command_execution_interrupted");
    expect(contributionEnabled).toBe(false);
    expect((await first.inspect()).records).toMatchObject([{ state: "applying", result: null }]);

    const restarted = new NodeCommandExecutor(path, "node-1", 7, clock);
    const result = await restarted.execute(input, async () => {
      attempts += 1;
      contributionEnabled = false;
      return { contributionEnabled };
    });
    expect(result.state).toBe("applied");
    expect(attempts).toBe(2);
    expect(await restarted.execute(input, async () => { throw new Error("must not rerun"); })).toEqual(result);
    expect((await restarted.inspect()).records).toMatchObject([{ state: "complete", result: { id: result.id } }]);
  });

  it("fails closed on replay, generation, node and corruption", async () => {
    const path = await journalPath();
    const executor = new NodeCommandExecutor(path, "node-1", 7, clock);
    const input = command();
    await executor.execute(input, async () => null);
    await expect(executor.execute({ ...input, type: "resume" }, async () => null)).rejects.toThrow("node_command_id_conflict");
    await expect(executor.execute(command({ nonce: input.nonce }), async () => null)).rejects.toThrow("node_command_replay");
    await expect(executor.execute(command({ generation: 6 }), async () => null)).rejects.toThrow("node_command_generation_downgrade");
    await expect(executor.execute(command({ nodeId: "node-2" }), async () => null)).rejects.toThrow("node_command_wrong_node");
    await writeFile(path, "{truncated", "utf8");
    await expect(executor.inspect()).rejects.toThrow("mycellios_node_command_journal_is_invalid");
  });

  it("returns durable expired and rejected results without retrying effects", async () => {
    const path = await journalPath();
    const executor = new NodeCommandExecutor(path, "node-1", 7, clock);
    let called = 0;
    const expired = await executor.execute(command({ expiresAt: "2026-08-10T11:59:59.000Z" }), async () => { called += 1; });
    expect(expired).toMatchObject({ state: "expired", error: { code: "node_command_expired" } });
    const rejectedInput = command();
    const rejected = await executor.execute(rejectedInput, async () => {
      called += 1;
      throw new NodeCommandExecutionError("node_update_unavailable", "No compatible update is staged.");
    });
    expect(rejected).toMatchObject({ state: "rejected", error: { code: "node_update_unavailable" } });
    expect(await executor.execute(rejectedInput, async () => { called += 1; })).toEqual(rejected);
    expect(called).toBe(1);
  });

  it("redacts secret-like handler errors", async () => {
    const executor = new NodeCommandExecutor(await journalPath(), "node-1", 7, clock);
    const result = await executor.execute(command(), async () => {
      throw new Error("authorization token abcdefghijklmnopqrstuvwxyz0123456789");
    });
    expect(result.error).toEqual({ code: "node_command_apply_failed", message: "The node could not apply the command." });
    expect(JSON.stringify(await executor.inspect())).not.toContain("abcdefghijklmnopqrstuvwxyz");
  });
});

const clock = () => Date.parse("2026-08-10T12:00:01.000Z");

async function journalPath(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), "mycellios-command-journal-"));
  cleanup.push(directory);
  return join(directory, "commands.json");
}
