import { constants } from "node:fs";
import { access, readFile } from "node:fs/promises";
import { posix } from "node:path";
import type { NodeConfiguration } from "../contracts/node-configuration.js";

export interface NodeIsolationProbe {
  platform: NodeJS.Platform;
  parentPid: number;
  readText(path: string): Promise<string>;
  executable(path: string): Promise<boolean>;
}

export async function assertNodeOsIsolation(
  config: NodeConfiguration,
  probe: NodeIsolationProbe = systemIsolationProbe(),
): Promise<void> {
  const isolation = config.isolation;
  if (probe.platform === "win32") {
    if (isolation.mode !== "windows-job-object") throw new Error("node_isolation_mode_platform_mismatch");
    if (!await probe.executable(isolation.brokerExecutable)) {
      throw new Error("node_windows_job_broker_is_unavailable");
    }
    return;
  }
  if (probe.platform === "linux") {
    if (isolation.mode !== "linux-cgroup-v2") throw new Error("node_isolation_mode_platform_mismatch");
    await assertLinuxCgroup(config, probe);
    return;
  }
  if (probe.platform === "darwin") {
    if (isolation.mode !== "macos-launchd-limits") throw new Error("node_isolation_mode_platform_mismatch");
    if (probe.parentPid !== 1) throw new Error("node_macos_launchd_containment_is_unverified");
    return;
  }
  throw new Error(`node_os_isolation_is_unsupported:${probe.platform}`);
}

async function assertLinuxCgroup(
  config: NodeConfiguration,
  probe: NodeIsolationProbe,
): Promise<void> {
  const membership = await probe.readText("/proc/self/cgroup");
  const relative = membership.split("\n")
    .map((line) => line.match(/^0::(.+)$/)?.[1])
    .find((value): value is string => Boolean(value));
  if (!relative || relative === "/") throw new Error("node_linux_dedicated_cgroup_is_missing");
  const root = posix.join("/sys/fs/cgroup", relative);
  const [memoryText, cpuText, pidsText] = await Promise.all([
    probe.readText(posix.join(root, "memory.max")),
    probe.readText(posix.join(root, "cpu.max")),
    probe.readText(posix.join(root, "pids.max")),
  ]);
  const memoryMax = boundedNumber(memoryText, "node_linux_memory_limit_is_unbounded");
  if (memoryMax > config.limits.maxRamMiB * 1024 * 1024) {
    throw new Error("node_linux_memory_limit_exceeds_configuration");
  }
  const [quotaText, periodText] = cpuText.trim().split(/\s+/);
  const quota = boundedNumber(quotaText ?? "", "node_linux_cpu_limit_is_unbounded");
  const period = boundedNumber(periodText ?? "", "node_linux_cpu_period_is_invalid");
  if ((quota / period) * 100 > config.limits.maxCpuPercent + 0.001) {
    throw new Error("node_linux_cpu_limit_exceeds_configuration");
  }
  boundedNumber(pidsText, "node_linux_process_limit_is_unbounded");
}

function boundedNumber(value: string, error: string): number {
  const normalized = value.trim();
  const parsed = Number(normalized);
  if (normalized === "max" || !Number.isSafeInteger(parsed) || parsed < 1) throw new Error(error);
  return parsed;
}

function systemIsolationProbe(): NodeIsolationProbe {
  return {
    platform: process.platform,
    parentPid: process.ppid,
    readText: (path) => readFile(path, "utf8"),
    executable: async (path) => {
      try { await access(path, constants.X_OK); return true; } catch { return false; }
    },
  };
}
