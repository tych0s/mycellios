import { randomBytes } from "node:crypto";
import { lstat, mkdir, open, readFile, rename, rm } from "node:fs/promises";
import { dirname, join, relative, resolve, sep } from "node:path";
import { isDeepStrictEqual } from "node:util";
import { z } from "zod";
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
  const progressPath = join(manifest.statePath, "bootstrap-progress.json");
  const progress = await readBootstrapProgress(progressPath);
  if (progress) {
    if (!isDeepStrictEqual(progress.manifest, manifest) ||
        new URL(progress.coordinatorUrl).toString() !== new URL(bundle.coordinatorUrl).toString()) {
      throw new Error("node_install_bootstrap_progress_mismatch");
    }
    // A service cannot create its key until bootstrap is complete. Never reset
    // resource settings from a journal belonging to a node that already ran.
    await assertAbsent(manifest.identityPath);
  }
  const nodeId = progress?.nodeId ?? `node-${bundle.enrollmentId.replaceAll("-", "")}`;
  const pythonPath = join(manifest.installRoot, "python");
  const runtimePath = join(manifest.installRoot, "runtime");
  const pythonExecutable = manifest.platform === "win32"
    ? join(runtimePath, "python.exe")
    : join(runtimePath, "bin", "python3");
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
  const output = [[manifest.configPath, nodeConfiguration], [workerConfigPath, workerConfiguration],
    [installationManifestPath, manifest]] as const;
  if (progress) {
    // Validate all surviving output before filling any missing file.
    for (const [path, expected] of output) {
      const existing = await readBootstrapJson(path);
      if (existing !== undefined && !isDeepStrictEqual(existing, expected)) {
        throw new Error("node_install_bootstrap_output_mismatch");
      }
    }
    const pending = await readBootstrapJson(enrollmentPath);
    if (pending !== undefined && new URL(nodeEnrollmentBundleSchema.parse(pending).coordinatorUrl).toString() !==
        new URL(bundle.coordinatorUrl).toString()) throw new Error("node_install_bootstrap_progress_mismatch");
  } else {
    for (const path of [...output.map(([path]) => path), enrollmentPath]) await assertAbsent(path);
    await writeProtectedJson(progressPath, { schema: "mycellios-node-bootstrap-progress/1", manifest, nodeId,
      coordinatorUrl: bundle.coordinatorUrl });
  }
  try {
    for (const [path, value] of output) {
      if (progress && await readBootstrapJson(path) !== undefined) continue;
      if (path === manifest.configPath) await new NodeConfigurationStore(path).save(value);
      else await writeProtectedJson(path, value);
    }
    await writeProtectedJson(enrollmentPath, bundle);
    await rm(progressPath);
  } catch (error) {
    if (!progress) await Promise.all([...output.map(([path]) => path), enrollmentPath, progressPath].map((path) => rm(path, { force: true })));
    throw error;
  }
  return { nodeId, configPath: manifest.configPath, workerConfigPath, enrollmentPath, installationManifestPath };
}

export async function hasNodeBootstrapProgress(manifest: NodeInstallationManifest): Promise<boolean> {
  return await readBootstrapProgress(join(manifest.statePath, "bootstrap-progress.json")) !== undefined;
}

async function readBootstrapProgress(path: string) {
  const value = await readBootstrapJson(path);
  if (value === undefined) return undefined;
  return z.object({ schema: z.literal("mycellios-node-bootstrap-progress/1"), manifest: nodeInstallationManifestSchema,
    nodeId: z.string().regex(/^node-[a-f0-9]{32}$/), coordinatorUrl: z.string().url() }).strict().parse(value);
}

async function readBootstrapJson(path: string): Promise<unknown | undefined> {
  let stats;
  try { stats = await lstat(path); }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined; throw error; }
  if (!stats.isFile() || stats.isSymbolicLink() || stats.size > 65_536) throw new Error("node_install_bootstrap_file_is_unsafe");
  return JSON.parse((await readFile(path, "utf8")).replace(/^\uFEFF/, "")) as unknown;
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

export async function writeProtectedJson(path: string, value: unknown): Promise<void> {
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
