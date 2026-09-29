import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
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
      if (platform === "win32") {
        await mkdir(join(root, "install", "bin"), { recursive: true });
        await writeFile(join(root, "install", "bin", "MycelliosNode.exe"), "wrapper");
      }
      const isolation = platform === "darwin" ? { mode: "macos-launchd-limits" as const }
        : { mode: "windows-job-object" as const, brokerExecutable: join(root, "install", "bin", "broker.exe") };
      await registerNativeNodeService({ manifest, config: configuration(manifest, isolation), run: runner(calls) });
      const definition = await readFile(manifest.serviceDefinitionPath, "utf8");
      expect(definition).not.toContain("enrollmentToken");
      if (platform === "darwin") expect(calls.some((call) => call[0] === "/bin/launchctl" && call[1] === "kickstart")).toBe(true);
      else {
        expect(calls).toContainEqual(expect.arrayContaining(["sc.exe", "create", manifest.serviceName, "binPath=", expect.stringContaining("MycelliosNode.exe")]));
        expect(calls.flat().join(" ")).toContain("NT AUTHORITY\\LocalService");
        expect(definition).toContain("<service>");
        expect(definition).toContain("<executable>");
        expect(definition).toContain("<logpath>");
        expect(definition).toContain("service-supervisor.js");
        expect(calls.some((call) => call[0] === "icacls.exe" && call[1]?.endsWith("protected-identity")
          && call.includes("*S-1-5-19:(OI)(CI)M"))).toBe(true);
      }
    }
  });

  it("fails closed when the service manager rejects registration", async () => {
    const root = await temporaryDirectory(); const manifest = installation(root, "linux");
    const run: ServiceCommandRunner = async (executable, arguments_) => ({ code: executable === "/bin/systemctl" && arguments_[0] === "enable" ? 1 : 0, stdout: "" });
    await expect(registerNativeNodeService({ manifest, config: configuration(manifest, { mode: "linux-cgroup-v2" }), run }))
      .rejects.toThrow("node_service_start_failed");
  });

  it("restores retained Windows service files only after SCM confirms the service is absent", async () => {
    const root = await temporaryDirectory(); const manifest = installation(root, "win32");
    const config = configuration(manifest, { mode: "windows-job-object", brokerExecutable: join(root, "install", "bin", "broker.exe") });
    await mkdir(join(root, "install", "bin"), { recursive: true });
    await mkdir(join(root, "service"), { recursive: true });
    await writeFile(join(root, "install", "bin", "MycelliosNode.exe"), "new-wrapper");
    await writeFile(join(root, "service", "MycelliosNode.exe"), "old-wrapper");
    await writeFile(manifest.serviceDefinitionPath, "old-definition");
    const running: ServiceCommandRunner = async (executable, arguments_) => ({
      code: executable === "sc.exe" && arguments_[0] === "query" ? 0 : 0, stdout: "",
    });
    await expect(registerNativeNodeService({ manifest, config, run: running, replaceRetainedServiceFiles: true }))
      .rejects.toThrow("node_service_must_be_absent_for_restore");
    expect(await readFile(join(root, "service", "MycelliosNode.exe"), "utf8")).toBe("old-wrapper");
    const calls: string[][] = [];
    const absent: ServiceCommandRunner = async (executable, arguments_) => {
      calls.push([executable, ...arguments_]);
      return { code: executable === "sc.exe" && arguments_[0] === "query" ? 1060 : 0, stdout: "" };
    };
    await registerNativeNodeService({ manifest, config, run: absent, replaceRetainedServiceFiles: true });
    expect(await readFile(join(root, "service", "MycelliosNode.exe"), "utf8")).toBe("new-wrapper");
    expect(await readFile(manifest.serviceDefinitionPath, "utf8")).toContain("service-supervisor.js");
    expect(calls).toContainEqual(expect.arrayContaining(["sc.exe", "create", manifest.serviceName, "binPath="]));
    expect(calls.flat().join(" ")).toContain("NT AUTHORITY\\LocalService");
    expect(calls.some((call) => call[0] === "icacls.exe" && call[1]?.endsWith("enrollment.json"))).toBe(false);
  });
});

function runner(calls: string[][]): ServiceCommandRunner { return async (executable, arguments_) => { calls.push([executable, ...arguments_]); return { code: 0, stdout: "" }; }; }
function installation(root: string, platform: "linux" | "darwin" | "win32"): NodeInstallationManifest {
  return { schema: "mycellios-node-installation/1", platform, serviceName: platform === "darwin" ? "io.mycellios.node" : "mycellios-node",
    installRoot: join(root, "install"), nodeExecutable: join(root, "install", "bin", platform === "win32" ? "node.exe" : "node"),
    helperEntrypoint: join(root, "install", "app", "node", "uninstall-main.js"), serviceDefinitionPath: join(root, "service", platform === "darwin" ? "io.mycellios.node.plist" : platform === "win32" ? "MycelliosNode.xml" : "mycellios-node.service"),
    configPath: join(root, "config", "node.json"), identityPath: join(root, "identity", "node.json"), cachePath: join(root, "cache"), logsPath: join(root, "logs"), statePath: join(root, "state"), receiptPath: join(root, "receipt", "uninstall.json") };
}
function configuration(manifest: NodeInstallationManifest, isolation: NodeConfiguration["isolation"]): NodeConfiguration {
  return { schema: "mycellios-node-configuration/1", revision: 1, nodeId: "node-a", coordinator: { url: "https://coordinator.example", identityPath: manifest.identityPath }, worker: { configPath: join(manifest.configPath, "..", "worker.json") },
    runtime: { pythonExecutable: join(manifest.installRoot, "runtime", "python"), pythonPath: join(manifest.installRoot, "runtime"), cachePath: manifest.cachePath, stagePort: 9850 },
    limits: { maxConcurrency: 1, maxCpuPercent: 90, maxRamMiB: 8192, maxVramMiB: 0, maxDiskMiB: 32768, maxTemperatureC: 85 }, isolation, updateChannel: "stable", uninstall: { manifestPath: join(manifest.statePath, "installation.json") } };
}
async function temporaryDirectory() { const root = await mkdtemp(join(tmpdir(), "mycellios-service-registration-")); cleanup.push(root); return root; }
