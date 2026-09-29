import { lstat, readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { nodeInstallationManifestSchema } from "../contracts/node-uninstall.js";
import { NodeConfigurationStore } from "./config-store.js";
import { defaultNodeInstallationManifest } from "./default-installation.js";
import { NodeIdentityStore, WindowsDpapiSecretStore } from "./identity-store.js";
import { registerNativeNodeService } from "./service-registration.js";

if (process.platform !== "win32") throw new Error("node_restore_requires_windows");
const installRoot = resolve(import.meta.dirname, "../..");
const manifest = defaultNodeInstallationManifest({ platform: "win32", installRoot });
const saved = nodeInstallationManifestSchema.parse(JSON.parse(await readFile(join(manifest.statePath, "installation.json"), "utf8")) as unknown);
for (const key of Object.keys(manifest) as (keyof typeof manifest)[]) {
  if (saved[key] !== manifest[key]) throw new Error(`node_restore_manifest_mismatch:${key}`);
}
const config = (await new NodeConfigurationStore(manifest.configPath).load()).config;
if (config.coordinator.identityPath !== manifest.identityPath ||
    config.worker.configPath !== join(dirname(manifest.configPath), "worker.json") ||
    config.runtime.pythonExecutable !== join(installRoot, "runtime", "python.exe") ||
    config.runtime.pythonPath !== join(installRoot, "python") ||
    config.runtime.cachePath !== manifest.cachePath ||
    config.uninstall?.manifestPath !== join(manifest.statePath, "installation.json")) {
  throw new Error("node_restore_configuration_paths_mismatch");
}
await requireRegularFile(config.worker.configPath);
const identity = JSON.parse(await readFile(manifest.identityPath, "utf8")) as Record<string, unknown>;
if (identity.schema !== "mycellios-node-identity/1" || identity.identityId !== config.nodeId ||
    identity.provider !== "windows-dpapi-local-machine") throw new Error("node_restore_identity_mismatch");
const pendingEnrollment = await lstat(join(dirname(manifest.configPath), "enrollment.json")).then(() => true, (error: NodeJS.ErrnoException) => {
  if (error.code === "ENOENT") return false;
  throw error;
});
if (pendingEnrollment) throw new Error("node_restore_pending_enrollment");
const programData = process.env.PROGRAMDATA;
if (!programData) throw new Error("node_restore_programdata_missing");
await new NodeIdentityStore(manifest.identityPath,
  new WindowsDpapiSecretStore(join(programData, "mycellios", "protected-identity"))).loadOrCreate(config.nodeId);
const healthPath = join(dirname(manifest.configPath), "state", "health.json");
const previousPid = await readHealth(healthPath).then((health) => health?.pid);
await registerNativeNodeService({ manifest, config, replaceRetainedServiceFiles: true });
for (let attempt = 0; attempt < 120; attempt += 1) {
  const health = await readHealth(healthPath);
  if (health?.state === "failed") throw new Error(`node_restore_service_failed:${health.error ?? "unknown"}`);
  if (health?.state === "ready" && typeof health.pid === "number" && health.pid !== previousPid) {
    process.stdout.write(`${JSON.stringify({ status: "restored", nodeId: config.nodeId })}\n`);
    process.exit(0);
  }
  await delay(1_000);
}
throw new Error("node_restore_service_did_not_reach_ready");

async function requireRegularFile(path: string): Promise<void> {
  const item = await lstat(path);
  if (!item.isFile() || item.isSymbolicLink()) throw new Error("node_restore_file_is_unsafe");
}

async function readHealth(path: string): Promise<{ state?: string; pid?: number; error?: string } | null> {
  return await readFile(path, "utf8").then((value) => JSON.parse(value) as { state?: string; pid?: number; error?: string }, () => null);
}
