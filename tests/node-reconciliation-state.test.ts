import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { NODE_EVENT_ORIGIN_CURSOR } from "../src/contracts/node-control.js";
import { NodeReconciliationStateStore } from "../src/node/reconciliation-state.js";

const cleanup: string[] = [];
afterEach(async () => Promise.all(cleanup.splice(0).map((path) => rm(path, { recursive: true, force: true }))));

describe("NodeReconciliationStateStore", () => {
  it("persists a cursor across restart and resets it on generation rotation", async () => {
    const directory = await mkdtemp(join(tmpdir(), "mycellios-reconcile-")); cleanup.push(directory);
    const path = join(directory, "state.json");
    const first = new NodeReconciliationStateStore(path, "node-1");
    expect((await first.load(3)).cursor).toBe(NODE_EVENT_ORIGIN_CURSOR);
    await first.save(3, "evt_12_aaaaaaaaaaaaaaaa");
    expect((await new NodeReconciliationStateStore(path, "node-1").load(3)).cursor).toBe("evt_12_aaaaaaaaaaaaaaaa");
    expect((await first.load(4)).cursor).toBe(NODE_EVENT_ORIGIN_CURSOR);
  });

  it("fails closed for corrupt or cross-node durable state", async () => {
    const directory = await mkdtemp(join(tmpdir(), "mycellios-reconcile-")); cleanup.push(directory);
    const path = join(directory, "state.json");
    await writeFile(path, "{}\n");
    await expect(new NodeReconciliationStateStore(path, "node-1").load(3)).rejects.toThrow("node_reconciliation_state_is_invalid");
  });
});
