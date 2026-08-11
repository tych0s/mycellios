import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { generateKeyPairSync } from "node:crypto";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { MYCELLIOS_NODE_CONFIGURATION_SCHEMA } from "../src/contracts/node-configuration.js";
import { NodeConfigurationStore } from "../src/node/config-store.js";

const cleanup: string[] = [];
afterEach(async () => Promise.all(cleanup.splice(0).map((path) => rm(path, { recursive: true, force: true }))));

describe("NodeConfigurationStore", () => {
  it("writes atomically, preserves the previous revision and rolls back", async () => {
    const directory = await temporaryDirectory();
    const store = new NodeConfigurationStore(join(directory, "node.json"));
    await store.save(configuration(1));
    await store.save(configuration(2));
    expect((await store.load()).config.revision).toBe(2);
    expect(JSON.parse(await readFile(store.backupPath, "utf8")).revision).toBe(1);
    expect((await store.rollback()).revision).toBe(1);
    expect((await store.load()).config.revision).toBe(1);
  });

  it("recovers from a corrupt primary without treating it as a valid default", async () => {
    const directory = await temporaryDirectory();
    const store = new NodeConfigurationStore(join(directory, "node.json"));
    await store.save(configuration(1));
    await store.save(configuration(2));
    await writeFile(store.path, "{truncated", "utf8");
    const loaded = await store.load();
    expect(loaded.source).toBe("backup");
    expect(loaded.config.revision).toBe(1);
  });

  it("rejects unknown fields and migrates only the explicit legacy schema", async () => {
    const directory = await temporaryDirectory();
    const store = new NodeConfigurationStore(join(directory, "node.json"));
    await expect(store.save({ ...configuration(1), secretToken: "must-not-enter-config" }))
      .rejects.toThrow();
    await writeFile(store.path, JSON.stringify({
      schema: "mycellios-node-configuration/0",
      nodeId: "node-a",
      coordinatorUrl: "https://coordinator.example",
      credentialPath: "/var/lib/mycellios/identity.json",
      workerConfigPath: "/etc/mycellios/worker.json",
      pythonExecutable: "/opt/mycellios/python",
      pythonPath: "/opt/mycellios/runtime",
      cachePath: "/var/cache/mycellios",
      stagePort: 9_850,
    }), "utf8");
    await expect(store.load()).rejects.toThrow(
      "mycellios_node_legacy_configuration_migration_confirmation_required",
    );
    const loaded = await new NodeConfigurationStore(store.path, { allowLegacyMigration: true }).load();
    expect(loaded.migrated).toBe(true);
    expect(loaded.config).toMatchObject({
      schema: MYCELLIOS_NODE_CONFIGURATION_SCHEMA,
      revision: 1,
      updateChannel: "stable",
    });
    await new NodeConfigurationStore(store.path, { allowLegacyMigration: true }).save(loaded.config);
    await expect(new NodeConfigurationStore(store.path).load()).resolves.toMatchObject({
      migrated: false,
      source: "primary",
    });
  });

  it("accepts only canonical Ed25519 pins and HTTPS component feeds", async () => {
    const directory = await temporaryDirectory();
    const store = new NodeConfigurationStore(join(directory, "node.json"));
    const { publicKey } = generateKeyPairSync("ed25519");
    const spki = publicKey.export({ format: "der", type: "spki" }).toString("base64url");
    await expect(store.save({ ...configuration(1), componentUpdates: {
      feedUrl: "https://updates.example.test",
      pinnedKeys: { dev: [], stable: [{ keyId: "stable-1", spki }] },
    } })).resolves.toMatchObject({ componentUpdates: { pinnedKeys: { stable: [{ keyId: "stable-1" }] } } });
    await expect(store.save({ ...configuration(2), componentUpdates: {
      feedUrl: "http://updates.example.test?token=secret",
      pinnedKeys: { dev: [], stable: [{ keyId: "stable-1", spki }] },
    } })).rejects.toThrow();
    await expect(store.save({ ...configuration(2), componentUpdates: {
      feedUrl: "https://updates.example.test",
      pinnedKeys: { dev: [], stable: [{ keyId: "stable-1", spki: Buffer.from("not-a-key").toString("base64url") }] },
    } })).rejects.toThrow();
  });
});

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), "mycellios-node-config-"));
  cleanup.push(directory);
  return directory;
}

function configuration(revision: number) {
  return {
    schema: MYCELLIOS_NODE_CONFIGURATION_SCHEMA,
    revision,
    nodeId: "node-a",
    coordinator: {
      url: "https://coordinator.example",
      identityPath: "/var/lib/mycellios/identity.json",
    },
    worker: { configPath: "/etc/mycellios/worker.json" },
    runtime: {
      pythonExecutable: "/opt/mycellios/python",
      pythonPath: "/opt/mycellios/runtime",
      cachePath: "/var/cache/mycellios",
      stagePort: 9_850,
    },
    limits: {
      maxConcurrency: 2,
      maxCpuPercent: 90,
      maxRamMiB: 8_192,
      maxVramMiB: 6_144,
      maxDiskMiB: 32_768,
      maxTemperatureC: 85,
    },
    isolation: { mode: "linux-cgroup-v2" as const },
    updateChannel: "stable" as const,
  };
}
