import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { MycelliosNodeServiceHost } from "../src/node/service-host.js";

const cleanup: string[] = [];
afterEach(async () => Promise.all(cleanup.splice(0).map((path) => rm(path, { recursive: true, force: true }))));

describe("MycelliosNodeServiceHost", () => {
  it("rejects a second live instance and releases its lock idempotently", async () => {
    const directory = await temporaryDirectory();
    const first = new MycelliosNodeServiceHost(directory);
    const second = new MycelliosNodeServiceHost(directory);
    await first.acquire(3);
    await expect(second.acquire(3)).rejects.toThrow("mycellios_node_instance_is_already_running");
    await first.writeHealth("ready", 3);
    expect(JSON.parse(await readFile(first.healthPath, "utf8"))).toMatchObject({
      state: "ready",
      configRevision: 3,
    });
    await first.release(3);
    await first.release(3);
    await second.acquire(3);
    await second.release(3);
  });

  it("recovers a stale lock and redacts health errors", async () => {
    const directory = await temporaryDirectory();
    const host = new MycelliosNodeServiceHost(directory);
    await writeFile(host.lockPath, JSON.stringify({
      schema: "mycellios-node-lock/1",
      pid: 2_147_483_647,
      token: "a".repeat(32),
      startedAt: new Date().toISOString(),
    }));
    await host.acquire(1);
    await host.release(
      1,
      "failed",
      "token_abcdefghijklmnopqrstuvwxyz123456 at https://private.example/path /home/user/model",
    );
    const health = JSON.parse(await readFile(host.healthPath, "utf8"));
    expect(health.state).toBe("failed");
    expect(health.error).not.toContain("private.example");
    expect(health.error).not.toContain("/home/user");
    expect(health.error).not.toContain("abcdefghijklmnopqrstuvwxyz123456");
  });

  it("fails closed for an invalid lock whose owner cannot be proven stale", async () => {
    const directory = await temporaryDirectory();
    const host = new MycelliosNodeServiceHost(directory);
    await writeFile(host.lockPath, "truncated");
    await expect(host.acquire(1)).rejects.toThrow("mycellios_node_instance_lock_is_invalid");
  });
});

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), "mycellios-node-service-"));
  cleanup.push(directory);
  return directory;
}
