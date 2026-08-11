import { describe, expect, it } from "vitest";
import type { NodeConfiguration } from "../src/contracts/node-configuration.js";
import { assertNodeOsIsolation, type NodeIsolationProbe } from "../src/node/os-isolation.js";

describe("node OS isolation", () => {
  it("requires an executable Windows Job Object broker", async () => {
    await expect(assertNodeOsIsolation(configuration({
      mode: "windows-job-object",
      brokerExecutable: "C:\\Program Files\\mycellios\\job-broker.exe",
    }), probe("win32", {}, false))).rejects.toThrow("node_windows_job_broker_is_unavailable");
    await expect(assertNodeOsIsolation(configuration({
      mode: "windows-job-object",
      brokerExecutable: "C:\\Program Files\\mycellios\\job-broker.exe",
    }), probe("win32", {}, true))).resolves.toBeUndefined();
  });

  it("proves bounded Linux cgroup v2 CPU, memory and process containment", async () => {
    const files = {
      "/proc/self/cgroup": "0::/system.slice/mycellios-node.service\n",
      "/sys/fs/cgroup/system.slice/mycellios-node.service/memory.max": String(4_096 * 1024 * 1024),
      "/sys/fs/cgroup/system.slice/mycellios-node.service/cpu.max": "80000 100000",
      "/sys/fs/cgroup/system.slice/mycellios-node.service/pids.max": "256",
    };
    await expect(assertNodeOsIsolation(configuration({ mode: "linux-cgroup-v2" }), probe("linux", files)))
      .resolves.toBeUndefined();
    await expect(assertNodeOsIsolation(configuration({ mode: "linux-cgroup-v2" }), probe("linux", {
      ...files,
      "/sys/fs/cgroup/system.slice/mycellios-node.service/memory.max": "max",
    }))).rejects.toThrow("node_linux_memory_limit_is_unbounded");
  });

  it("requires macOS to be owned by launchd", async () => {
    await expect(assertNodeOsIsolation(configuration({ mode: "macos-launchd-limits" }), {
      ...probe("darwin", {}), parentPid: 200,
    })).rejects.toThrow("node_macos_launchd_containment_is_unverified");
    await expect(assertNodeOsIsolation(configuration({ mode: "macos-launchd-limits" }), {
      ...probe("darwin", {}), parentPid: 1,
    })).resolves.toBeUndefined();
  });
});

function probe(
  platform: NodeJS.Platform,
  files: Record<string, string>,
  executable = false,
): NodeIsolationProbe {
  return {
    platform,
    parentPid: 1,
    readText: async (path) => {
      const value = files[path];
      if (value === undefined) throw new Error(`missing fixture:${path}`);
      return value;
    },
    executable: async () => executable,
  };
}

function configuration(isolation: NodeConfiguration["isolation"]): NodeConfiguration {
  return {
    schema: "mycellios-node-configuration/1",
    revision: 1,
    nodeId: "node-a",
    coordinator: { url: "https://coordinator.example", identityPath: "/var/lib/mycellios/identity.json" },
    worker: { configPath: "/etc/mycellios/worker.json" },
    runtime: {
      pythonExecutable: "/opt/mycellios/python",
      pythonPath: "/opt/mycellios/runtime",
      cachePath: "/var/cache/mycellios",
      stagePort: 9_850,
    },
    limits: {
      maxConcurrency: 2,
      maxCpuPercent: 80,
      maxRamMiB: 4_096,
      maxVramMiB: 6_144,
      maxDiskMiB: 32_768,
      maxTemperatureC: 85,
    },
    isolation,
    updateChannel: "stable",
  };
}
