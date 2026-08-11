import { spawn } from "node:child_process";
import { mkdir, open, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import type { NodeConfiguration } from "../contracts/node-configuration.js";
import type { NodeInstallationManifest } from "../contracts/node-uninstall.js";
import { linuxSystemdServiceDefinition, macOsLaunchdServiceDefinition } from "./service-definition.js";

export interface ServiceCommandResult { code: number; stdout: string }
export type ServiceCommandRunner = (executable: string, arguments_: readonly string[]) => Promise<ServiceCommandResult>;

export async function registerNativeNodeService(input: {
  manifest: NodeInstallationManifest;
  config: NodeConfiguration;
  run?: ServiceCommandRunner;
}): Promise<void> {
  const run = input.run ?? runServiceCommand;
  const { manifest, config } = input;
  const serviceEntrypoint = join(dirname(manifest.helperEntrypoint), "main.js");
  if (manifest.platform === "linux") {
    const definition = linuxSystemdServiceDefinition({ config, configPath: manifest.configPath,
      nodeExecutable: manifest.nodeExecutable, serviceEntrypoint });
    await writeNewServiceDefinition(manifest.serviceDefinitionPath, definition);
    const user = await run("/usr/sbin/useradd", ["--system", "--user-group", "--home-dir", "/nonexistent", "--shell", "/usr/sbin/nologin", "mycellios"]);
    if (user.code !== 0 && user.code !== 9) throw new Error("node_service_user_creation_failed");
    await requireSuccess(run, "/bin/chown", ["-R", "mycellios:mycellios", dirname(manifest.configPath), dirname(manifest.identityPath), manifest.cachePath, manifest.logsPath, manifest.statePath], "node_service_permissions_failed");
    await requireSuccess(run, "/bin/systemctl", ["daemon-reload"], "node_service_manager_reload_failed");
    await requireSuccess(run, "/bin/systemctl", ["enable", "--now", manifest.serviceName], "node_service_start_failed");
    return;
  }
  if (manifest.platform === "darwin") {
    const definition = macOsLaunchdServiceDefinition({ config, configPath: manifest.configPath,
      nodeExecutable: manifest.nodeExecutable, serviceEntrypoint, label: manifest.serviceName });
    await writeNewServiceDefinition(manifest.serviceDefinitionPath, definition);
    await requireSuccess(run, "/usr/bin/chown", ["-R", "root:wheel", dirname(manifest.configPath), dirname(manifest.identityPath), manifest.cachePath, manifest.logsPath, manifest.statePath], "node_service_permissions_failed");
    await requireSuccess(run, "/bin/launchctl", ["bootstrap", "system", manifest.serviceDefinitionPath], "node_service_registration_failed");
    await requireSuccess(run, "/bin/launchctl", ["enable", `system/${manifest.serviceName}`], "node_service_enable_failed");
    await requireSuccess(run, "/bin/launchctl", ["kickstart", "-k", `system/${manifest.serviceName}`], "node_service_start_failed");
    return;
  }
  if (manifest.platform === "win32") {
    const binPath = windowsServiceBinPath(manifest.nodeExecutable, serviceEntrypoint, manifest.configPath);
    await writeNewServiceDefinition(manifest.serviceDefinitionPath, `${JSON.stringify({ schema: "mycellios-windows-service-definition/1", serviceName: manifest.serviceName, binPath }, null, 2)}\n`);
    for (const path of [dirname(manifest.configPath), dirname(manifest.identityPath), manifest.cachePath, manifest.logsPath, manifest.statePath]) {
      await requireSuccess(run, "icacls.exe", [path, "/inheritance:r", "/grant:r", "NT AUTHORITY\\SYSTEM:(OI)(CI)F", "/grant:r", "NT AUTHORITY\\LOCAL SERVICE:(OI)(CI)M"], "node_service_permissions_failed");
    }
    await requireSuccess(run, "sc.exe", ["create", manifest.serviceName, "binPath=", binPath, "start=", "auto", "obj=", "NT AUTHORITY\\LocalService"], "node_service_registration_failed");
    await requireSuccess(run, "sc.exe", ["failure", manifest.serviceName, "reset=", "86400", "actions=", "restart/5000/restart/15000/none/0"], "node_service_recovery_policy_failed");
    await requireSuccess(run, "sc.exe", ["start", manifest.serviceName], "node_service_start_failed");
    return;
  }
  const exhaustive: never = manifest.platform;
  throw new Error(`node_service_platform_is_unsupported:${exhaustive}`);
}

function windowsServiceBinPath(nodeExecutable: string, entrypoint: string, configPath: string): string {
  for (const value of [nodeExecutable, entrypoint, configPath]) if (/[\r\n\0"]/.test(value)) throw new Error("node_service_path_is_invalid");
  return `"${nodeExecutable}" "${entrypoint}" --config "${configPath}"`;
}

async function writeNewServiceDefinition(path: string, value: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const file = await open(path, "wx", 0o644);
  try { await file.writeFile(value, "utf8"); await file.sync(); }
  finally { await file.close(); }
}

async function requireSuccess(run: ServiceCommandRunner, executable: string, arguments_: readonly string[], error: string): Promise<void> {
  const result = await run(executable, arguments_);
  if (result.code !== 0) throw new Error(error);
}

export async function runServiceCommand(executable: string, arguments_: readonly string[]): Promise<ServiceCommandResult> {
  return await new Promise((resolve, reject) => {
    const child = spawn(executable, [...arguments_], { shell: false, windowsHide: true, stdio: ["ignore", "pipe", "ignore"] });
    const chunks: Buffer[] = [];
    child.stdout.on("data", (chunk: Buffer) => chunks.push(Buffer.from(chunk)));
    child.once("error", (error) => reject(new Error("node_service_manager_is_unavailable", { cause: error })));
    child.once("close", (code) => resolve({ code: code ?? -1, stdout: Buffer.concat(chunks).toString("utf8") }));
  });
}
