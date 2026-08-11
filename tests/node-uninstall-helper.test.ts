import { randomUUID } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { NodeUninstallScheduler, verifyUninstallRequest } from "../src/node/uninstall-helper.js";
import { executeVerifiedUninstall } from "../src/node/uninstall-executor.js";
import type { NodeInstallationManifest } from "../src/contracts/node-uninstall.js";

const cleanup: string[] = [];
afterEach(async () => Promise.all(cleanup.splice(0).map((path) => rm(path, { recursive: true, force: true }))));

describe("privileged node uninstall lifecycle", () => {
  it("writes a minimal request, invokes without a shell and requires a digest-bound arm", async () => {
    const root = await temporaryDirectory();
    const manifestPath = join(root, "installation.json");
    const helperPath = join(root, "install", "node", "uninstall-main.js");
    const manifest = installation(root, helperPath);
    await writeFile(manifestPath, JSON.stringify(manifest), "utf8");
    const runner = vi.fn(async () => undefined);
    const requests = join(root, "state", "uninstall-requests");
    const scheduler = new NodeUninstallScheduler(manifestPath, requests, runner, () => new Date("2026-08-10T12:00:00.000Z"));
    const commandId = randomUUID();
    const receipt = await scheduler.schedule({ nodeId: "node-1", generation: 4, commandId,
      retain: { cache: true, logs: false, configuration: true, identity: true } });
    expect(receipt).toMatchObject({ state: "scheduled", retained: ["cache", "configuration", "identity"] });
    expect(runner).toHaveBeenCalledWith(manifest.nodeExecutable, [helperPath, "--request", expect.stringMatching(/\.json$/), "--request-digest", receipt.requestDigest]);
    const requestPath = join(requests, `${receipt.requestId}.json`);
    const verified = await verifyUninstallRequest({ requestPath, requestDigest: receipt.requestDigest, expectedRequestDirectory: requests });
    expect(verified.request).toMatchObject({ nodeId: "node-1", commandId, retain: { identity: true, logs: false } });
    expect(JSON.parse(await readFile(requestPath, "utf8"))).not.toHaveProperty("installRoot");
    await expect(scheduler.armAcknowledged(new Set([randomUUID()]))).resolves.toBe(0);
    await expect(scheduler.armAcknowledged(new Set([commandId]))).resolves.toBe(1);
    await expect(readFile(join(requests, `${receipt.requestId}.armed`), "utf8")).resolves.toBe(`${receipt.requestDigest}\n`);
  });

  it("derives deletion targets only from the pinned manifest and is receipt-idempotent", async () => {
    const root = await temporaryDirectory();
    const manifest = installation(root, join(root, "install", "node", "uninstall-main.js"));
    const request = { schema: "mycellios-node-uninstall-request/1" as const, id: randomUUID(), nodeId: "node-1", generation: 4,
      commandId: randomUUID(), manifestPath: join(root, "installation.json"), manifestDigest: `sha256:${"a".repeat(64)}` as const,
      retain: { cache: true, logs: false, configuration: true, identity: true }, createdAt: "2026-08-10T12:00:00.000Z" };
    const run = vi.fn(async () => undefined);
    const remove = vi.fn(async (_path: string) => undefined);
    const digest = `sha256:${"b".repeat(64)}` as const;
    const first = await executeVerifiedUninstall({ request, manifest, requestDigest: digest, run, remove,
      now: () => new Date("2026-08-10T12:00:01.000Z") });
    if (manifest.platform === "linux") {
      expect(run).toHaveBeenNthCalledWith(1, "systemctl", ["disable", "--now", "mycellios-node.service"], true);
      expect(run).toHaveBeenNthCalledWith(2, "systemctl", ["daemon-reload"]);
    } else if (manifest.platform === "darwin") {
      expect(run).toHaveBeenCalledWith("launchctl", ["bootout", "system/mycellios-node.service"], true);
    } else {
      expect(run).toHaveBeenNthCalledWith(1, "sc.exe", ["stop", "mycellios-node.service"], true);
      expect(run).toHaveBeenNthCalledWith(2, "sc.exe", ["delete", "mycellios-node.service"], true);
    }
    const removed = remove.mock.calls.map(([path]) => path);
    expect(removed).toEqual(expect.arrayContaining([manifest.logsPath, manifest.statePath, manifest.serviceDefinitionPath, manifest.installRoot]));
    expect(removed).not.toEqual(expect.arrayContaining([manifest.cachePath, manifest.configPath, manifest.identityPath]));
    expect(first.state).toBe("completed");
    run.mockClear(); remove.mockClear();
    await expect(executeVerifiedUninstall({ request, manifest, requestDigest: digest, run, remove })).resolves.toEqual(first);
    expect(run).not.toHaveBeenCalled(); expect(remove).not.toHaveBeenCalled();
  });

  it.each([
    ["KEEP-DATA", { cache: true, logs: true, configuration: true, identity: true }, ["cache", "logs", "configuration", "identity"]],
    ["DELETE-ALL", { cache: false, logs: false, configuration: false, identity: false }, []],
  ] as const)("executes the complete %s retention matrix", async (_choice, retain, expectedRetained) => {
    const root = await temporaryDirectory();
    const manifest = installation(root, join(root, "install", "node", "uninstall-main.js"));
    const request = {
      schema: "mycellios-node-uninstall-request/1" as const,
      id: randomUUID(), nodeId: "node-1", generation: 4, commandId: randomUUID(),
      manifestPath: join(root, "installation.json"),
      manifestDigest: `sha256:${"a".repeat(64)}` as const,
      retain,
      createdAt: "2026-08-10T12:00:00.000Z",
    };
    const remove = vi.fn(async (_path: string) => undefined);
    const receipt = await executeVerifiedUninstall({
      request, manifest, requestDigest: `sha256:${"c".repeat(64)}`,
      run: async () => undefined, remove,
    });
    const removed = new Set(remove.mock.calls.map(([path]) => path));
    const dataPaths = {
      cache: manifest.cachePath,
      logs: manifest.logsPath,
      configuration: manifest.configPath,
      identity: manifest.identityPath,
    };
    for (const [name, path] of Object.entries(dataPaths)) {
      expect(removed.has(path), name).toBe(!retain[name as keyof typeof retain]);
    }
    expect(removed.has(manifest.serviceDefinitionPath)).toBe(true);
    expect(removed.has(manifest.statePath)).toBe(true);
    expect(removed.has(manifest.installRoot)).toBe(true);
    expect(receipt.retained).toEqual(expectedRetained);
  });

  it("rejects request traversal and changed installation manifests", async () => {
    const root = await temporaryDirectory();
    const manifestPath = join(root, "installation.json");
    await writeFile(manifestPath, JSON.stringify(installation(root, join(root, "install", "node", "uninstall-main.js"))), "utf8");
    const scheduler = new NodeUninstallScheduler(manifestPath, join(root, "requests"), async () => undefined);
    const receipt = await scheduler.schedule({ nodeId: "node-1", generation: 1, commandId: randomUUID(), retain: { cache: true, logs: true, configuration: true, identity: true } });
    const requestPath = join(root, "requests", `${receipt.requestId}.json`);
    await expect(verifyUninstallRequest({ requestPath, requestDigest: receipt.requestDigest, expectedRequestDirectory: join(root, "other") }))
      .rejects.toThrow("node_uninstall_request_path_outside_root");
    await writeFile(manifestPath, `${JSON.stringify({ ...installation(root, join(root, "install", "node", "uninstall-main.js")), serviceName: "changed" })}\n`, "utf8");
    await expect(verifyUninstallRequest({ requestPath, requestDigest: receipt.requestDigest, expectedRequestDirectory: join(root, "requests") }))
      .rejects.toThrow("node_uninstall_manifest_digest_mismatch");
  });

  it("rejects retention and receipt paths that would be erased with the installation", async () => {
    const root = await temporaryDirectory();
    const manifestPath = join(root, "installation.json");
    const base = installation(root, join(root, "install", "node", "uninstall-main.js"));
    await writeFile(manifestPath, JSON.stringify({ ...base, cachePath: join(base.installRoot, "cache") }), "utf8");
    await expect(new NodeUninstallScheduler(manifestPath, join(root, "requests"), async () => undefined).schedule({
      nodeId: "node-1", generation: 1, commandId: randomUUID(), retain: { cache: true, logs: true, configuration: true, identity: true },
    })).rejects.toThrow("node_uninstall_data_path_inside_install_root");
    await writeFile(manifestPath, JSON.stringify({ ...base, receiptPath: join(base.statePath, "receipt.json") }), "utf8");
    await expect(new NodeUninstallScheduler(manifestPath, join(root, "requests"), async () => undefined).schedule({
      nodeId: "node-1", generation: 1, commandId: randomUUID(), retain: { cache: true, logs: true, configuration: true, identity: true },
    })).rejects.toThrow("node_uninstall_receipt_path_is_not_durable");
  });
});

async function temporaryDirectory(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "mycellios-uninstall-")); cleanup.push(root); return root;
}

function installation(root: string, helperEntrypoint: string): NodeInstallationManifest {
  return { schema: "mycellios-node-installation/1", platform: process.platform as "linux" | "darwin" | "win32", serviceName: "mycellios-node.service",
    installRoot: join(root, "install"), nodeExecutable: process.execPath, helperEntrypoint,
    serviceDefinitionPath: join(root, "service", "mycellios-node.service"), configPath: join(root, "config", "node.json"),
    identityPath: join(root, "identity", "node.key"), cachePath: join(root, "cache"), logsPath: join(root, "logs"),
    statePath: join(root, "state"), receiptPath: join(root, "receipts", "uninstall.json") };
}
