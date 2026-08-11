import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { NodeControlStateStore } from "../src/node/control-state.js";

const cleanup: string[] = [];
afterEach(async () => Promise.all(cleanup.splice(0).map((path) => rm(path, { recursive: true, force: true }))));

describe("NodeControlStateStore", () => {
  it("persists pause/resume idempotently across store instances", async () => {
    const directory = await temporaryDirectory();
    const path = join(directory, "control.json");
    const first = new NodeControlStateStore(path);
    expect((await first.load()).contributionEnabled).toBe(true);
    expect((await first.load()).draining).toBe(false);
    expect(await first.setContributionEnabled(false)).toMatchObject({
      revision: 1,
      contributionEnabled: false,
    });
    expect((await first.setContributionEnabled(false)).revision).toBe(1);
    const restarted = new NodeControlStateStore(path);
    expect(await restarted.load()).toMatchObject({ revision: 1, contributionEnabled: false });
    expect(await restarted.setContributionEnabled(true)).toMatchObject({ revision: 2, contributionEnabled: true });
    expect(await restarted.setDraining(true)).toMatchObject({ revision: 3, draining: true });
    expect((await new NodeControlStateStore(path).load()).draining).toBe(true);
  });

  it("serializes concurrent mutations and fails closed on corruption", async () => {
    const directory = await temporaryDirectory();
    const path = join(directory, "control.json");
    const store = new NodeControlStateStore(path);
    await Promise.all([
      store.setContributionEnabled(false),
      store.setContributionEnabled(true),
      store.setContributionEnabled(false),
    ]);
    expect(await store.load()).toMatchObject({ revision: 3, contributionEnabled: false });
    await writeFile(path, "{truncated", "utf8");
    await expect(store.load()).rejects.toThrow("mycellios_node_control_state_is_invalid");
  });

  it("migrates the v1 contribution preference without inventing a drain", async () => {
    const directory = await temporaryDirectory();
    const path = join(directory, "control.json");
    await writeFile(path, JSON.stringify({
      schema: "mycellios-node-control-state/1",
      revision: 4,
      contributionEnabled: false,
      updatedAt: "2026-08-10T12:00:00.000Z",
    }), "utf8");
    expect(await new NodeControlStateStore(path).load()).toMatchObject({
      schema: "mycellios-node-control-state/2",
      revision: 4,
      contributionEnabled: false,
      draining: false,
    });
  });
});

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), "mycellios-node-control-"));
  cleanup.push(directory);
  return directory;
}
