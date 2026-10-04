import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { bootstrapNodeInstallation } from "../src/node/installation-bootstrap.js";
import { inspectNodeInstallationResume, renewNodeInstallationEnrollment } from "../src/node/installation-resume.js";
import { resumeNodeInstallationService } from "../src/node/installation-service-resume.js";
import { NodeIdentityStore, type ProtectedSecretStore } from "../src/node/identity-store.js";
import type { ServiceCommandRunner } from "../src/node/service-registration.js";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "mycellios-resume-")); roots.push(root);
  const manifest = { schema: "mycellios-node-installation/1" as const, platform: "win32" as const,
    serviceName: "MycelliosNode", installRoot: join(root, "install"), nodeExecutable: join(root, "install", "bin", "node.exe"),
    helperEntrypoint: join(root, "install", "app", "node", "uninstall-main.js"),
    serviceDefinitionPath: join(root, "service", "MycelliosNode.xml"), configPath: join(root, "configuration", "node.json"),
    identityPath: join(root, "identity", "node.json"), cachePath: join(root, "cache"), logsPath: join(root, "logs"),
    statePath: join(root, "state"), receiptPath: join(root, "receipts", "uninstall.json") };
  const bundle = { schema: "mycellios-node-enrollment-bundle/1", coordinatorUrl: "https://coordinator.example/",
    enrollmentId: "4b5e61db-9e21-4268-b1a3-50dd0e818660", enrollmentToken: "t".repeat(43), nonce: "n".repeat(32),
    expiresAt: "2030-08-10T12:10:00.000Z" };
  const source = join(root, "source.json"); await writeFile(source, JSON.stringify(bundle));
  const result = await bootstrapNodeInstallation({ manifest, enrollmentSourcePath: source });
  const values = new Map<string, string>();
  const secrets: ProtectedSecretStore = { provider: "windows-dpapi-local-machine",
    get: async (key) => values.get(key) ?? null, set: async (key, value) => { values.set(key, value); },
    delete: async (key) => { values.delete(key); } };
  const identity = new NodeIdentityStore(manifest.identityPath, secrets);
  await identity.loadOrCreate(result.nodeId);
  return { root, manifest, bundle, source, result, identity, values };
}

describe("native installation continuation", () => {
  it("renews expired pairing while preserving configuration, node ID and protected key", async () => {
    const f = await fixture(); const before = await readFile(f.result.configPath, "utf8");
    const identityBefore = await readFile(f.manifest.identityPath, "utf8");
    const keyBefore = (await f.identity.loadOrCreate(f.result.nodeId)).publicKey;
    await writeFile(f.result.enrollmentPath, JSON.stringify({ ...f.bundle, expiresAt: "2020-01-01T00:00:00.000Z" }));
    const retained = await inspectNodeInstallationResume(f.manifest);
    const fresh = { ...f.bundle, enrollmentId: "5b5e61db-9e21-4268-b1a3-50dd0e818660", enrollmentToken: "z".repeat(43) };
    await writeFile(f.source, JSON.stringify(fresh));
    await renewNodeInstallationEnrollment(retained, f.source);
    expect(JSON.parse(await readFile(f.result.enrollmentPath, "utf8"))).toEqual(fresh);
    expect(await readFile(f.result.configPath, "utf8")).toBe(before);
    expect(await readFile(f.manifest.identityPath, "utf8")).toBe(identityBefore);
    expect((await f.identity.loadOrCreate(f.result.nodeId)).publicKey).toEqual(keyBefore);
    expect(f.values.size).toBe(1);
  });

  it("permits finishing accelerator setup after consumption without another pairing", async () => {
    const f = await fixture(); await rm(f.result.enrollmentPath);
    const retained = await inspectNodeInstallationResume(f.manifest);
    expect(retained.pending).toBe(false);
    await expect(renewNodeInstallationEnrollment(retained, f.source)).rejects.toThrow("pairing_already_consumed");
    expect((await f.identity.loadOrCreate(f.result.nodeId)).publicKey).toBeDefined();
  });

  it("rejects foreign coordinator and mismatched config, worker or identity without rewriting files", async () => {
    const f = await fixture(); const retained = await inspectNodeInstallationResume(f.manifest);
    const pending = await readFile(f.result.enrollmentPath, "utf8");
    await writeFile(f.source, JSON.stringify({ ...f.bundle, coordinatorUrl: "https://other.example/" }));
    await expect(renewNodeInstallationEnrollment(retained, f.source)).rejects.toThrow("coordinator_mismatch");
    expect(await readFile(f.result.enrollmentPath, "utf8")).toBe(pending);
    await expect(inspectNodeInstallationResume({ ...f.manifest, cachePath: join(f.root, "other") })).rejects.toThrow("manifest_mismatch");
    const worker = await readFile(f.result.workerConfigPath, "utf8");
    await writeFile(f.result.workerConfigPath, worker.replace(f.result.nodeId, "node-foreign"));
    await expect(inspectNodeInstallationResume(f.manifest)).rejects.toThrow("worker_mismatch");
    await writeFile(f.result.workerConfigPath, worker);
    const metadata = await readFile(f.manifest.identityPath, "utf8");
    await writeFile(f.manifest.identityPath, metadata.replace(f.result.nodeId, "node-foreign"));
    await expect(inspectNodeInstallationResume(f.manifest)).rejects.toThrow("identity_mismatch");
  });

  it("stops only the owned service before renewing and starting, and rejects a foreign service", async () => {
    const f = await fixture(); const retained = await inspectNodeInstallationResume(f.manifest);
    const calls: string[] = []; let stopped = false;
    const run: ServiceCommandRunner = async (exe, args) => {
      calls.push(exe === "powershell.exe" ? "cim" : args[0]!);
      if (args[0] === "query") return { code: 0, stdout: stopped ? "ESTADO : 1 STOPPED" : "ESTADO : 4 RUNNING" };
      if (exe === "powershell.exe") return { code: 0, stdout: JSON.stringify({ Name: f.manifest.serviceName, PathName: `"${join(dirname(f.manifest.serviceDefinitionPath), "MycelliosNode.exe")}"`, StartName: "NT AUTHORITY\\LocalService" }) };
      if (args[0] === "stop") stopped = true;
      return { code: 0, stdout: "" };
    };
    await resumeNodeInstallationService({ ...retained, run, pause: async () => {}, beforeStart: async () => { calls.push("renew"); } });
    expect(calls).toEqual(["query", "cim", "stop", "query", "renew", "start"]);
    calls.length = 0;
    const foreign: ServiceCommandRunner = async (exe, args) => {
      calls.push(exe === "powershell.exe" ? "cim" : args[0]!); return { code: 0, stdout: exe === "powershell.exe" ? JSON.stringify({ Name: f.manifest.serviceName, PathName: "foreign.exe", StartName: "LocalSystem" }) : "ESTADO : 4 RUNNING" };
    };
    await expect(resumeNodeInstallationService({ ...retained, run: foreign, beforeStart: async () => { calls.push("renew"); } }))
      .rejects.toThrow("ownership_mismatch");
    expect(calls).toEqual(["query", "cim"]);
  });

  it("restores an absent service with pending pairing protected before startup", async () => {
    const f = await fixture(); const retained = await inspectNodeInstallationResume(f.manifest);
    await mkdir(join(f.manifest.installRoot, "bin"), { recursive: true });
    await writeFile(join(f.manifest.installRoot, "bin", "MycelliosNode.exe"), "wrapper fixture");
    const calls: string[][] = [];
    const run: ServiceCommandRunner = async (exe, args) => {
      calls.push([exe, ...args]); return { code: exe === "sc.exe" && args[0] === "query" ? 1060 : 0, stdout: "" };
    };
    await resumeNodeInstallationService({ ...retained, run, beforeStart: async () => { calls.push(["renew"]); } });
    expect(calls[0]).toEqual(["sc.exe", "query", "MycelliosNode"]);
    expect(calls[1]).toEqual(["renew"]);
    expect(calls).toContainEqual(expect.arrayContaining(["icacls.exe", f.result.enrollmentPath, "*S-1-5-19:M"]));
    expect(calls).toContainEqual(["sc.exe", "start", "MycelliosNode"]);
  });

  it.each(["invalid-json", "wrong-name", "wrong-account"])("rejects %s CIM output before service mutation", async (failure) => {
    const f = await fixture();
    const calls: string[] = [];
    const run: ServiceCommandRunner = async (exe, args) => {
      calls.push(exe === "powershell.exe" ? "cim" : args[0]!);
      if (exe !== "powershell.exe") return { code: 0, stdout: "ESTADO : 4 RUNNING" };
      return { code: 0, stdout: failure === "invalid-json" ? "null" : JSON.stringify({
        Name: failure === "wrong-name" ? "another-service" : f.manifest.serviceName,
        PathName: `"${join(dirname(f.manifest.serviceDefinitionPath), "MycelliosNode.exe")}"`,
        StartName: failure === "wrong-account" ? "LocalSystem" : "NT AUTHORITY\\LocalService",
      }) };
    };
    await expect(resumeNodeInstallationService({ manifest: f.manifest, config: JSON.parse(await readFile(f.result.configPath, "utf8")),
      run, beforeStart: async () => { calls.push("renew"); } })).rejects.toThrow("ownership_mismatch");
    expect(calls).toEqual(["query", "cim"]);
  });
});
