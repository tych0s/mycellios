import { describe, expect, it } from "vitest";
import type { NodeConfiguration } from "../src/contracts/node-configuration.js";
import {
  linuxSystemdServiceDefinition,
  macOsLaunchdServiceDefinition,
} from "../src/node/service-definition.js";

describe("mycellios-node service definitions", () => {
  it("renders a bounded systemd cgroup with group-wide teardown", () => {
    const definition = linuxSystemdServiceDefinition({
      config: configuration({ mode: "linux-cgroup-v2" }),
      configPath: "/etc/mycellios/node.json",
      nodeExecutable: "/opt/mycellios/node",
      serviceEntrypoint: "/opt/mycellios/dist/node/main.js",
    });
    expect(definition).toContain("KillMode=control-group");
    expect(definition).toContain("CPUQuota=80%");
    expect(definition).toContain("MemoryMax=4096M");
    expect(definition).toContain("TasksMax=256");
    expect(definition).toContain("NoNewPrivileges=true");
    expect(definition).not.toContain("privateKeyPkcs8");
  });

  it("renders launchd process-group and hard resident/process limits", () => {
    const definition = macOsLaunchdServiceDefinition({
      config: configuration({ mode: "macos-launchd-limits" }),
      configPath: "/Library/Application Support/mycellios/node.json",
      nodeExecutable: "/opt/mycellios/node",
      serviceEntrypoint: "/opt/mycellios/dist/node/main.js",
    });
    expect(definition).toContain("<key>AbandonProcessGroup</key><false/>");
    expect(definition).toContain("<key>ResidentSetSize</key><integer>4294967296</integer>");
    expect(definition).toContain("<key>NumberOfProcesses</key><integer>256</integer>");
    expect(definition).toContain("/Library/Application Support/mycellios/node.json");
  });

  it("rejects platform mismatch and newline injection", () => {
    expect(() => linuxSystemdServiceDefinition({
      config: configuration({ mode: "macos-launchd-limits" }),
      configPath: "/etc/mycellios/node.json",
      nodeExecutable: "/opt/mycellios/node",
      serviceEntrypoint: "/opt/mycellios/dist/node/main.js",
    })).toThrow("node_service_definition_isolation_mode_mismatch");
    expect(() => linuxSystemdServiceDefinition({
      config: configuration({ mode: "linux-cgroup-v2" }),
      configPath: "/etc/mycellios/node.json\nExecStart=/bin/sh",
      nodeExecutable: "/opt/mycellios/node",
      serviceEntrypoint: "/opt/mycellios/dist/node/main.js",
    })).toThrow("node_service_path_is_invalid");
  });
});

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
