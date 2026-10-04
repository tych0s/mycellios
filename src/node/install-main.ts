import { lstat, readFile, rm } from "node:fs/promises";
import { setTimeout as delay } from "node:timers/promises";
import { bootstrapNodeInstallation, hasNodeBootstrapProgress } from "./installation-bootstrap.js";
import { NodeConfigurationStore } from "./config-store.js";
import { nodeInstallationManifestSchema } from "../contracts/node-uninstall.js";
import { registerNativeNodeService, runServiceCommand } from "./service-registration.js";
import { defaultNodeInstallationManifest } from "./default-installation.js";
import { dirname, join, resolve } from "node:path";
import { prepareInstalledNodeAccelerator } from "./native-accelerator.js";
import { inspectNodeInstallationResume, renewNodeInstallationEnrollment } from "./installation-resume.js";
import { resumeNodeInstallationService } from "./installation-service-resume.js";

const enrollmentPath = argument("--enrollment");
const resumeInstalled = process.argv.includes("--resume-installed");
if (!enrollmentPath && !resumeInstalled) throw new Error("usage: install-main --enrollment <path> [--resume-installed] [--manifest <path>]");
const manifestPath = argument("--manifest");
const manifest = manifestPath
  ? nodeInstallationManifestSchema.parse(JSON.parse(await readFile(manifestPath, "utf8")) as unknown)
  : defaultNodeInstallationManifest({ platform: supportedPlatform(), installRoot: argument("--install-root") ?? resolve(import.meta.dirname, "../..") });
if (resumeInstalled && await hasNodeBootstrapProgress(manifest)) {
  if (!enrollmentPath) throw new Error("node_install_bootstrap_resume_requires_current_pairing");
  await bootstrapNodeInstallation({ manifest, enrollmentSourcePath: enrollmentPath });
}
const retained = resumeInstalled ? await inspectNodeInstallationResume(manifest) : undefined;
const result = retained?.result ?? await bootstrapNodeInstallation({ manifest, enrollmentSourcePath: enrollmentPath! });
const config = (await new NodeConfigurationStore(result.configPath).load()).config;
const previousPid = retained ? await readFile(join(dirname(result.configPath), "state", "health.json"), "utf8")
  .then((text) => (JSON.parse(text) as { pid?: number }).pid).catch(() => undefined) : undefined;
if (retained) {
  await resumeNodeInstallationService({ manifest, config, beforeStart: async () => {
    if (enrollmentPath) await renewNodeInstallationEnrollment(retained, enrollmentPath);
  } });
} else await registerNativeNodeService({ manifest, config });
if (enrollmentPath && resolve(enrollmentPath) !== resolve(result.enrollmentPath)) await rm(resolve(enrollmentPath), { force: true });
const firstPid = await waitForReady(result.configPath, result.enrollmentPath, previousPid);
const accelerator = await prepareInstalledNodeAccelerator({
  configPath: result.configPath,
  installRoot: manifest.installRoot,
  allowProvisioning: true,
  onProgress: (event) => {
    if (event.recordLog) process.stdout.write(`${JSON.stringify({ phase: event.phase, message: event.message })}\n`);
  },
});
let backend = "cpu";
if (accelerator.status === "gpu-ready") {
  if (manifest.platform === "win32") {
    await restartService(manifest.serviceName);
    await waitForReady(result.configPath, result.enrollmentPath, firstPid);
  }
  const diagnostics = JSON.parse(await readFile(join(dirname(result.configPath), "state", "diagnostics.json"), "utf8")) as {
    runtime?: { backend?: string };
  };
  backend = diagnostics.runtime?.backend ?? "cpu";
}
process.stdout.write(`${JSON.stringify({ status: "installed", nodeId: result.nodeId, backend })}\n`);

async function waitForReady(configPath: string, pendingEnrollmentPath: string, previousPid?: number): Promise<number> {
  const healthPath = join(dirname(configPath), "state", "health.json");
  for (let attempt = 0; attempt < 120; attempt += 1) {
    const health = await readFile(healthPath, "utf8").then((text) => JSON.parse(text) as {
      state?: string; pid?: number; error?: string;
    }).catch(() => null);
    if (health?.state === "failed" && health.pid !== previousPid) throw new Error(`node_install_service_failed:${health.error ?? "unknown"}`);
    if (health?.state === "ready" && typeof health.pid === "number" &&
        health.pid !== previousPid && await enrollmentConsumed(pendingEnrollmentPath)) {
      return health.pid;
    }
    await delay(1_000);
  }
  throw new Error("node_install_service_did_not_reach_ready_and_redeem_pairing");
}

async function enrollmentConsumed(path: string): Promise<boolean> {
  try { await lstat(path); return false; }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return true;
    throw error;
  }
}

async function restartService(serviceName: string): Promise<void> {
  const stopped = await runServiceCommand("sc.exe", ["stop", serviceName]);
  if (stopped.code !== 0) throw new Error("node_install_gpu_service_stop_failed");
  for (let attempt = 0; attempt < 60; attempt += 1) {
    const status = await runServiceCommand("sc.exe", ["query", serviceName]);
    if (status.code === 0 && /:\s*1\s+STOPPED\b/i.test(status.stdout)) break;
    if (attempt === 59) throw new Error("node_install_gpu_service_stop_timed_out");
    await delay(1_000);
  }
  const started = await runServiceCommand("sc.exe", ["start", serviceName]);
  if (started.code !== 0) throw new Error("node_install_gpu_service_restart_failed");
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function supportedPlatform(): "linux" | "darwin" | "win32" {
  if (process.platform === "linux" || process.platform === "darwin" || process.platform === "win32") return process.platform;
  throw new Error(`node_install_platform_is_unsupported:${process.platform}`);
}
