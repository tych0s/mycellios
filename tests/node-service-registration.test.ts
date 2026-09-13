import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { registerNativeNodeService, type ServiceCommandRunner } from "../src/node/service-registration.js";
import type { NodeConfiguration } from "../src/contracts/node-configuration.js";
import type { NodeInstallationManifest } from "../src/contracts/node-uninstall.js";

const cleanup: string[] = [];
afterEach(async () => Promise.all(cleanup.splice(0).map((path) => rm(path, { recursive: true, force: true }))));

describe("native service registration", () => {
  it("installs and starts a bounded systemd service under the dedicated account", async () => {
    const root = await temporaryDirectory(); const calls: string[][] = [];
    const manifest = installation(root, "linux");
    await registerNativeNodeService({ manifest, config: configuration(manifest, { mode: "linux-cgroup-v2" }), run: runner(calls) });
    const unit = await readFile(manifest.serviceDefinitionPath, "utf8");
    expect(unit).toContain("User=mycellios"); expect(unit).toContain(`ReadWritePaths=${JSON.stringify(join(root, "config"))}`);
    expect(calls).toContainEqual(["/bin/systemctl", "enable", "--now", manifest.serviceName]);
  });

  it("registers launchd and SCM with deterministic service identities and no enrollment secret", async () => {
    for (const platform of ["darwin", "win32"] as const) {
      const root = await temporaryDirectory(); const calls: string[][] = [];
      const manifest = installation(root, platform);
      const isolation = platform === "darwin" ? { mode: "macos-launchd-limits" as const }
        : { mode: "windows-job-object" as const, brokerExecutable: join(root, "install", "bin", "broker.exe") };
      await registerNativeNodeService({ manifest, config: configuration(manifest, isolation), run: runner(calls) });
      const definition = await readFile(manifest.serviceDefinitionPath, "utf8");
      expect(definition).not.toContain("enrollmentToken");
      if (platform === "darwin") expect(calls.some((call) => call[0] === "/bin/launchctl" && call[1] === "kickstart")).toBe(true);
      else {
        expect(calls).toContainEqual(expect.arrayContaining(["sc.exe", "create", manifest.serviceName, "binPath="]));
        expect(calls.flat().join(" ")).toContain("NT AUTHORITY\\LocalService");
      }
    }
  });

  it("fails closed when the service manager rejects registration", async () => {
    const root = await temporaryDirectory(); const manifest = installation(root, "linux");
    const run: ServiceCommandRunner = async (executable, arguments_) => ({ code: executable === "/bin/systemctl" && arguments_[0] === "enable" ? 1 : 0, stdout: "" });
    await expect(registerNativeNodeService({ manifest, config: configuration(manifest, { mode: "linux-cgroup-v2" }), run }))
      .rejects.toThrow("node_service_start_failed");
  });
});

function runner(calls: string[][]): ServiceCommandRunner { return async (executable, arguments_) => { calls.push([executable, ...arguments_]); return { code: 0, stdout: "" }; }; }
function installation(root: string, platform: "linux" | "darwin" | "win32"): NodeInstallationManifest {
  return { schema: "mycellios-node-installation/1", platform, serviceName: platform === "darwin" ? "io.mycellios.node" : "mycellios-node",
    installRoot: join(root, "install"), nodeExecutable: join(root, "install", "bin", platform === "win32" ? "node.exe" : "node"),
    helperEntrypoint: join(root, "install", "app", "node", "uninstall-main.js"), serviceDefinitionPath: join(root, "service", platform === "darwin" ? "io.mycellios.node.plist" : "mycellios-node.service"),
    configPath: join(root, "config", "node.json"), identityPath: join(root, "identity", "node.json"), cachePath: join(root, "cache"), logsPath: join(root, "logs"), statePath: join(root, "state"), receiptPath: join(root, "receipt", "uninstall.json") };
}
function configuration(manifest: NodeInstallationManifest, isolation: NodeConfiguration["isolation"]): NodeConfiguration {
  return { schema: "mycellios-node-configuration/1", revision: 1, nodeId: "node-a", coordinator: { url: "https://coordinator.example", identityPath: manifest.identityPath }, worker: { configPath: join(manifest.configPath, "..", "worker.json") },
    runtime: { pythonExecutable: join(manifest.installRoot, "runtime", "python"), pythonPath: join(manifest.installRoot, "runtime"), cachePath: manifest.cachePath, stagePort: 9850 },
    limits: { maxConcurrency: 1, maxCpuPercent: 90, maxRamMiB: 8192, maxVramMiB: 0, maxDiskMiB: 32768, maxTemperatureC: 85 }, isolation, updateChannel: "stable", uninstall: { manifestPath: join(manifest.statePath, "installation.json") } };
}
async function temporaryDirectory() { const root = await mkdtemp(join(tmpdir(), "mycellios-service-registration-")); cleanup.push(root); return root; }
