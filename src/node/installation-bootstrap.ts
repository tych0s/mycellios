import { randomBytes } from "node:crypto";
import { lstat, mkdir, open, readFile, rename, rm } from "node:fs/promises";
import { dirname, join, relative, resolve, sep } from "node:path";
import { nodeEnrollmentBundleSchema } from "../contracts/node-control.js";
import { nodeInstallationManifestSchema, type NodeInstallationManifest } from "../contracts/node-uninstall.js";
import { NodeConfigurationStore } from "./config-store.js";

export interface NodeInstallationBootstrapResult {
  nodeId: string;
  configPath: string;
  workerConfigPath: string;
  enrollmentPath: string;
  installationManifestPath: string;
}

/** Creates the protected, stable configuration consumed by the native service. */
export async function bootstrapNodeInstallation(input: {
  manifest: unknown;
  enrollmentSourcePath: string;
}): Promise<NodeInstallationBootstrapResult> {
  const manifest = nodeInstallationManifestSchema.parse(input.manifest);
  assertManifestLayout(manifest);
  const source = resolve(input.enrollmentSourcePath);
  const sourceStats = await lstat(source);
  if (!sourceStats.isFile() || sourceStats.isSymbolicLink() || sourceStats.size > 4_096) {
    throw new Error("node_install_enrollment_source_is_unsafe");
  }
  const bundle = nodeEnrollmentBundleSchema.parse(JSON.parse(await readFile(source, "utf8")));
  if (Date.parse(bundle.expiresAt) <= Date.now()) throw new Error("node_install_enrollment_bundle_expired");

  const configurationDirectory = dirname(manifest.configPath);
  const workerConfigPath = join(configurationDirectory, "worker.json");
  const enrollmentPath = join(configurationDirectory, "enrollment.json");
  const installationManifestPath = join(manifest.statePath, "installation.json");
  const nodeId = `node-${bundle.enrollmentId.replaceAll("-", "")}`;
  const pythonPath = join(manifest.installRoot, "runtime");
  const pythonExecutable = manifest.platform === "win32"
    ? join(pythonPath, "python.exe")
    : join(pythonPath, "bin", "python3");
  const isolation = manifest.platform === "win32"
    ? { mode: "windows-job-object" as const, brokerExecutable: join(manifest.installRoot, "bin", "mycellios-job-broker.exe") }
    : manifest.platform === "darwin"
      ? { mode: "macos-launchd-limits" as const }
      : { mode: "linux-cgroup-v2" as const };
  const nodeConfiguration = {
    schema: "mycellios-node-configuration/1" as const,
    revision: 1,
    nodeId,
    coordinator: { url: bundle.coordinatorUrl, identityPath: manifest.identityPath },
    worker: { configPath: workerConfigPath },
    runtime: { pythonExecutable, pythonPath, cachePath: manifest.cachePath, stagePort: 9_850 },
    limits: { maxConcurrency: 1, maxCpuPercent: 90, maxRamMiB: 8_192, maxVramMiB: 0, maxDiskMiB: 32_768, maxTemperatureC: 85 },
    isolation,
    updateChannel: "stable" as const,
    uninstall: { manifestPath: installationManifestPath },
  };
  const workerConfiguration = {
    region: "auto",
    instanceId: nodeId,
    capacityScope: "host" as const,
    offeredVramMb: 512,
    limits: { maxConcurrency: 1, pauseWhenForeground: true, maxTemperatureC: 85 },
    adapter: { kind: "mycellios-native" as const, model: "mycellios-native-control" as const },
    deployment: { contextLimit: 8_192 },
  };

  await mkdir(configurationDirectory, { recursive: true, mode: 0o700 });
  await mkdir(manifest.statePath, { recursive: true, mode: 0o700 });
  await assertAbsent(manifest.configPath);
  await assertAbsent(workerConfigPath);
  await assertAbsent(enrollmentPath);
  await assertAbsent(installationManifestPath);
  try {
    await new NodeConfigurationStore(manifest.configPath).save(nodeConfiguration);
    await writeProtectedJson(workerConfigPath, workerConfiguration);
    await writeProtectedJson(enrollmentPath, bundle);
    await writeProtectedJson(installationManifestPath, manifest);
  } catch (error) {
    await Promise.all([manifest.configPath, workerConfigPath, enrollmentPath, installationManifestPath].map((path) => rm(path, { force: true })));
    throw error;
  }
  return { nodeId, configPath: manifest.configPath, workerConfigPath, enrollmentPath, installationManifestPath };
}

function assertManifestLayout(manifest: NodeInstallationManifest): void {
  for (const path of [manifest.configPath, manifest.identityPath, manifest.cachePath, manifest.logsPath, manifest.statePath, manifest.receiptPath]) {
    const relation = relative(manifest.installRoot, path);
    if (relation === "" || (!relation.startsWith(`..${sep}`) && relation !== "..")) {
      throw new Error("node_install_data_path_inside_install_root");
    }
  }
}

async function assertAbsent(path: string): Promise<void> {
  try { await lstat(path); throw new Error("node_install_configuration_already_exists"); }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
}

async function writeProtectedJson(path: string, value: unknown): Promise<void> {
  const temporary = `${path}.tmp-${process.pid}-${randomBytes(8).toString("hex")}`;
  const file = await open(temporary, "wx", 0o600);
  try {
    await file.writeFile(`${JSON.stringify(value, null, 2)}\n`, "utf8");
    await file.sync();
    await file.close();
    await rename(temporary, path);
  } catch (error) {
    await file.close().catch(() => undefined);
    await rm(temporary, { force: true });
    throw error;
  }
}
