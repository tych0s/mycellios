import { access, chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { bootstrapNodeInstallation } from "../src/node/installation-bootstrap.js";

const cleanup: string[] = [];
afterEach(async () => Promise.all(cleanup.splice(0).map((path) => rm(path, { recursive: true, force: true }))));

describe("native installation bootstrap", () => {
  it("creates protected service configuration without copying the enrollment secret into durable config", async () => {
    const root = await temporaryDirectory();
    const enrollment = join(root, "pairing.mycellios-enrollment");
    await writeFile(enrollment, JSON.stringify(bundle()), { mode: 0o600 });
    const manifest = installation(root, "linux");
    const result = await bootstrapNodeInstallation({ manifest, enrollmentSourcePath: enrollment });
    const node = JSON.parse(await readFile(result.configPath, "utf8"));
    const worker = JSON.parse(await readFile(result.workerConfigPath, "utf8"));
    expect(node).toMatchObject({ nodeId: result.nodeId, coordinator: { url: bundle().coordinatorUrl },
      isolation: { mode: "linux-cgroup-v2" }, uninstall: { manifestPath: join(manifest.statePath, "installation.json") } });
    expect(worker).toMatchObject({ instanceId: result.nodeId, adapter: { kind: "mycellios-native" } });
    expect(`${JSON.stringify(node)}${JSON.stringify(worker)}`).not.toContain(bundle().enrollmentToken);
    expect(await readFile(result.enrollmentPath, "utf8")).toContain(bundle().enrollmentToken);
    expect(JSON.parse(await readFile(result.installationManifestPath, "utf8"))).toEqual(manifest);
    if (process.platform !== "win32") expect((await import("node:fs/promises")).stat(result.enrollmentPath).then((value) => value.mode & 0o777)).resolves.toBe(0o600);
  });

  it("is fail-closed on reinstall and rolls back partial output", async () => {
    const root = await temporaryDirectory();
    const enrollment = join(root, "pairing.json");
    await writeFile(enrollment, JSON.stringify(bundle()), { mode: 0o600 });
    const manifest = installation(root, "darwin");
    await bootstrapNodeInstallation({ manifest, enrollmentSourcePath: enrollment });
    await expect(bootstrapNodeInstallation({ manifest, enrollmentSourcePath: enrollment }))
      .rejects.toThrow("node_install_configuration_already_exists");
  });

  it("selects the Windows broker and rejects layouts that erase durable state", async () => {
    const root = await temporaryDirectory();
    const enrollment = join(root, "pairing.json");
    await writeFile(enrollment, JSON.stringify(bundle()), { mode: 0o600 });
    const windows = installation(root, "win32");
    const result = await bootstrapNodeInstallation({ manifest: windows, enrollmentSourcePath: enrollment });
    expect(JSON.parse(await readFile(result.configPath, "utf8")).isolation).toEqual({
      mode: "windows-job-object", brokerExecutable: join(windows.installRoot, "bin", "mycellios-job-broker.exe"),
    });
    const secondRoot = await temporaryDirectory();
    const secondEnrollment = join(secondRoot, "pairing.json");
    await writeFile(secondEnrollment, JSON.stringify(bundle()), { mode: 0o600 });
    const unsafe = installation(secondRoot, "linux");
    unsafe.configPath = join(unsafe.installRoot, "node.json");
    await expect(bootstrapNodeInstallation({ manifest: unsafe, enrollmentSourcePath: secondEnrollment }))
      .rejects.toThrow("node_install_data_path_inside_install_root");
  });
});

function bundle() {
  return { schema: "mycellios-node-enrollment-bundle/1", coordinatorUrl: "https://coordinator.example",
    enrollmentId: "4b5e61db-9e21-4268-b1a3-50dd0e818660", enrollmentToken: "t".repeat(43), nonce: "n".repeat(32),
    expiresAt: "2030-08-10T12:10:00.000Z" };
}

function installation(root: string, platform: "linux" | "darwin" | "win32") {
  return { schema: "mycellios-node-installation/1" as const, platform, serviceName: "mycellios-node.service",
    installRoot: join(root, "install"), nodeExecutable: join(root, "install", "bin", "node"),
    helperEntrypoint: join(root, "install", "node", "uninstall-main.js"), serviceDefinitionPath: join(root, "service", "mycellios-node.service"),
    configPath: join(root, "config", "node.json"), identityPath: join(root, "identity", "node.key"), cachePath: join(root, "cache"),
    logsPath: join(root, "logs"), statePath: join(root, "state"), receiptPath: join(root, "receipts", "uninstall.json") };
}

async function temporaryDirectory(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "mycellios-install-bootstrap-")); cleanup.push(root); return root;
}
