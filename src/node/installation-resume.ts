import { lstat, readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { nodeConfigurationSchema } from "../contracts/node-configuration.js";
import { nodeEnrollmentBundleSchema } from "../contracts/node-control.js";
import { nodeInstallationManifestSchema } from "../contracts/node-uninstall.js";
import { writeProtectedJson, type NodeInstallationBootstrapResult } from "./installation-bootstrap.js";

/** Inspect durable bootstrap output without rewriting configuration or identity. */
export async function inspectNodeInstallationResume(manifestInput: unknown) {
  const manifest = nodeInstallationManifestSchema.parse(manifestInput);
  const workerConfigPath = join(dirname(manifest.configPath), "worker.json");
  const enrollmentPath = join(dirname(manifest.configPath), "enrollment.json");
  const installationManifestPath = join(manifest.statePath, "installation.json");
  const saved = nodeInstallationManifestSchema.parse(await readProtectedJson(installationManifestPath));
  for (const key of Object.keys(manifest) as (keyof typeof manifest)[]) {
    if (saved[key] !== manifest[key]) throw new Error(`node_install_resume_manifest_mismatch:${key}`);
  }
  const config = nodeConfigurationSchema.parse(await readProtectedJson(manifest.configPath));
  const executable = manifest.platform === "win32" ? join(manifest.installRoot, "runtime", "python.exe")
    : join(manifest.installRoot, "runtime", "bin", "python3");
  if (config.coordinator.identityPath !== manifest.identityPath || config.worker.configPath !== workerConfigPath ||
      config.runtime.pythonExecutable !== executable || config.runtime.pythonPath !== join(manifest.installRoot, "python") ||
      config.runtime.cachePath !== manifest.cachePath || config.uninstall?.manifestPath !== installationManifestPath ||
      (manifest.platform === "win32" && (config.isolation.mode !== "windows-job-object" ||
        config.isolation.brokerExecutable !== join(manifest.installRoot, "bin", "mycellios-job-broker.exe")))) {
    throw new Error("node_install_resume_configuration_mismatch");
  }
  const worker = await readProtectedJson(workerConfigPath) as { instanceId?: string; adapter?: { kind?: string } };
  if (worker.instanceId !== config.nodeId || worker.adapter?.kind !== "mycellios-native") {
    throw new Error("node_install_resume_worker_mismatch");
  }
  const pending = await exists(enrollmentPath);
  if (await exists(manifest.identityPath)) {
    const identity = await readProtectedJson(manifest.identityPath) as { schema?: string; identityId?: string; provider?: string };
    if (identity.schema !== "mycellios-node-identity/1" || identity.identityId !== config.nodeId ||
        (manifest.platform === "win32" && identity.provider !== "windows-dpapi-local-machine")) {
      throw new Error("node_install_resume_identity_mismatch");
    }
  } else if (!pending) throw new Error("node_install_resume_identity_missing");
  if (pending) {
    const bundle = nodeEnrollmentBundleSchema.parse(await readProtectedJson(enrollmentPath, 4_096));
    if (coordinator(bundle.coordinatorUrl) !== coordinator(config.coordinator.url)) {
      throw new Error("node_install_resume_coordinator_mismatch");
    }
  }
  const result: NodeInstallationBootstrapResult = {
    nodeId: config.nodeId, configPath: manifest.configPath, workerConfigPath, enrollmentPath, installationManifestPath,
  };
  return { manifest, config, result, pending };
}

/** Caller must stop the service before atomically replacing its pending bundle. */
export async function renewNodeInstallationEnrollment(
  installation: Awaited<ReturnType<typeof inspectNodeInstallationResume>>, source: string,
): Promise<void> {
  // Never turn a consumed enrollment or an existing healthy node into a new identity.
  if (!installation.pending) throw new Error("node_install_resume_pairing_already_consumed");
  const bundle = nodeEnrollmentBundleSchema.parse(await readProtectedJson(resolve(source), 4_096));
  if (Date.parse(bundle.expiresAt) <= Date.now()) throw new Error("node_install_enrollment_bundle_expired");
  if (coordinator(bundle.coordinatorUrl) !== coordinator(installation.config.coordinator.url)) {
    throw new Error("node_install_resume_coordinator_mismatch");
  }
  await writeProtectedJson(installation.result.enrollmentPath, bundle);
}

async function readProtectedJson(path: string, maxBytes = 65_536): Promise<unknown> {
  const stats = await lstat(path);
  if (!stats.isFile() || stats.isSymbolicLink() || stats.size > maxBytes) throw new Error("node_install_resume_file_is_unsafe");
  return JSON.parse((await readFile(path, "utf8")).replace(/^\uFEFF/, "")) as unknown;
}

async function exists(path: string): Promise<boolean> {
  try { await lstat(path); return true; }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return false; throw error; }
}

function coordinator(value: string): string {
  const url = new URL(value);
  if (url.username || url.password || url.search || url.hash || url.pathname !== "/") {
    throw new Error("node_install_resume_coordinator_is_unsafe");
  }
  return url.toString();
}
