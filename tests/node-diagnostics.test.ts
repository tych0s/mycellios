import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { NodeConfiguration } from "../src/contracts/node-configuration.js";
import type { PhysicalProbeV1 } from "../src/distribution/physical-probe.js";
import { buildNodeDiagnosticSnapshot, writeNodeDiagnosticSnapshot } from "../src/node/diagnostics.js";

const cleanup: string[] = [];
afterEach(async () => Promise.all(cleanup.splice(0).map((path) => rm(path, { recursive: true, force: true }))));

describe("node diagnostics", () => {
  it("exports useful build/runtime/hardware state without identity, paths or payloads", async () => {
    const snapshot = buildNodeDiagnosticSnapshot({
      config: configuration(),
      version: "0.2.77",
      state: "ready",
      physicalProbe: probe(),
      errorCodes: [
        "runtime_stage_failed /home/alice/model token=secret_abcdefghijklmnopqrstuvwxyz prompt='private request'",
      ],
      now: new Date("2026-08-09T00:00:00.000Z"),
    });
    const serialized = JSON.stringify(snapshot);
    expect(snapshot.errors).toEqual(["runtime_stage_failed"]);
    expect(snapshot.host.nodeFingerprint).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(serialized).not.toContain("node-private-name");
    expect(serialized).not.toContain("/home/alice");
    expect(serialized).not.toContain("private request");
    expect(serialized).not.toContain("secret_abcdefghijklmnopqrstuvwxyz");
    expect(serialized).not.toContain("identity.json");

    const directory = await mkdtemp(join(tmpdir(), "mycellios-node-diagnostics-"));
    cleanup.push(directory);
    const path = join(directory, "diagnostics.json");
    await writeNodeDiagnosticSnapshot(path, snapshot);
    expect(JSON.parse(await readFile(path, "utf8"))).toEqual(snapshot);
  });
});

function configuration(): NodeConfiguration {
  return {
    schema: "mycellios-node-configuration/1",
    revision: 2,
    nodeId: "node-private-name",
    coordinator: { url: "https://private.example", identityPath: "/var/lib/mycellios/identity.json" },
    worker: { configPath: "/etc/mycellios/worker.json" },
    runtime: {
      pythonExecutable: "/opt/mycellios/python",
      pythonPath: "/opt/mycellios/runtime",
      cachePath: "/var/cache/mycellios",
      stagePort: 9_850,
    },
    limits: {
      maxConcurrency: 2,
      maxCpuPercent: 90,
      maxRamMiB: 8_192,
      maxVramMiB: 6_144,
      maxDiskMiB: 32_768,
      maxTemperatureC: 85,
    },
    isolation: { mode: "linux-cgroup-v2" },
    updateChannel: "stable",
  };
}

function probe(): PhysicalProbeV1 {
  return {
    schema: "gdlp-physical-probe/1",
    nonce: "diagnostic-probe",
    host: {
      fingerprintSha256: `sha256:${"1".repeat(64)}`,
      fingerprintSource: "linux-machine-id",
      platform: "linux",
      architecture: "x86_64",
      kernelRelease: "6.8.0",
      pythonVersion: "3.12.13",
    },
    runtime: {
      torchVersion: "2.13.0",
      cudaVersion: "12.6",
      rocmVersion: null,
      cudaApiAvailable: true,
      distributedAvailable: true,
      ncclAvailable: true,
      ncclVersion: "2.27.3",
    },
    devices: [{
      index: 0,
      name: "NVIDIA RTX 4090",
      totalMemoryBytes: 24_576 * 1024 * 1024,
      freeMemoryBytes: 20_000 * 1024 * 1024,
      runtimeTotalMemoryBytes: 24_576 * 1024 * 1024,
      capability: [8, 9],
      uuidSha256: `sha256:${"2".repeat(64)}`,
      fingerprintSha256: `sha256:${"3".repeat(64)}`,
    }],
  };
}
