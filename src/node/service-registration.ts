import { spawn } from "node:child_process";
import { constants, copyFile, mkdir, open, writeFile } from "node:fs/promises";
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
  replaceRetainedServiceFiles?: boolean;
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
    if (input.replaceRetainedServiceFiles) {
      const existing = await run("sc.exe", ["query", manifest.serviceName]);
      if (existing.code !== 1060) throw new Error("node_service_must_be_absent_for_restore");
    }
    const supervisedEntrypoint = join(dirname(manifest.helperEntrypoint), "service-supervisor.js");
    const serviceDirectory = dirname(manifest.serviceDefinitionPath);
    const wrapperSource = join(manifest.installRoot, "bin", "MycelliosNode.exe");
    const wrapperTarget = join(serviceDirectory, "MycelliosNode.exe");
    if (manifest.serviceDefinitionPath !== join(serviceDirectory, "MycelliosNode.xml")) {
      throw new Error("node_service_definition_path_is_invalid");
    }
    const protectedIdentity = join(dirname(dirname(manifest.configPath)), "protected-identity");
    const writable = [dirname(manifest.configPath), dirname(manifest.identityPath), protectedIdentity,
      manifest.cachePath, manifest.logsPath, manifest.statePath];
    for (const path of [...writable, serviceDirectory]) await mkdir(path, { recursive: true });
    for (const path of writable) {
      await requireSuccess(run, "icacls.exe", [path, "/inheritance:r", "/grant:r", "*S-1-5-18:(OI)(CI)F", "/grant:r", "*S-1-5-32-544:(OI)(CI)F", "/grant:r", "*S-1-5-19:(OI)(CI)M"], "node_service_permissions_failed");
    }
    const protectedFiles = [manifest.configPath, config.worker.configPath,
      ...(!input.replaceRetainedServiceFiles ? [join(dirname(manifest.configPath), "enrollment.json")] : []),
      join(manifest.statePath, "installation.json")];
    for (const path of protectedFiles) {
      await requireSuccess(run, "icacls.exe", [path, "/inheritance:r", "/grant:r", "*S-1-5-18:F", "/grant:r", "*S-1-5-32-544:F", "/grant:r", "*S-1-5-19:M"], "node_service_permissions_failed");
    }
    await requireSuccess(run, "icacls.exe", [serviceDirectory, "/inheritance:r", "/grant:r", "*S-1-5-18:(OI)(CI)F", "/grant:r", "*S-1-5-32-544:(OI)(CI)F", "/grant:r", "*S-1-5-19:(OI)(CI)RX"], "node_service_permissions_failed");
    await copyFile(wrapperSource, wrapperTarget, input.replaceRetainedServiceFiles ? 0 : constants.COPYFILE_EXCL);
    const definition = windowsServiceDefinition({ manifest, serviceEntrypoint: supervisedEntrypoint });
    if (input.replaceRetainedServiceFiles) await writeFile(manifest.serviceDefinitionPath, definition, "utf8");
    else await writeNewServiceDefinition(manifest.serviceDefinitionPath, definition);
    await requireSuccess(run, "sc.exe", ["create", manifest.serviceName, "binPath=", `"${wrapperTarget}"`, "start=", "auto", "obj=", "NT AUTHORITY\\LocalService"], "node_service_registration_failed");
    await requireSuccess(run, "sc.exe", ["failure", manifest.serviceName, "reset=", "86400", "actions=", "restart/5000/restart/15000/none/0"], "node_service_recovery_policy_failed");
    await requireSuccess(run, "sc.exe", ["start", manifest.serviceName], "node_service_start_failed");
    return;
  }
  const exhaustive: never = manifest.platform;
  throw new Error(`node_service_platform_is_unsupported:${exhaustive}`);
}

function windowsServiceDefinition(input: { manifest: NodeInstallationManifest; serviceEntrypoint: string }): string {
  const { manifest, serviceEntrypoint } = input;
  for (const value of [manifest.nodeExecutable, serviceEntrypoint, manifest.configPath, manifest.logsPath]) {
    if (/[\r\n\0"]/.test(value)) throw new Error("node_service_path_is_invalid");
  }
  const escaped = (value: string) => value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
  return ["<service>", `  <id>${escaped(manifest.serviceName)}</id>`,
    "  <name>Mycellios Node</name>", "  <description>Mycellios native inference node</description>",
    `  <executable>${escaped(manifest.nodeExecutable)}</executable>`,
    `  <arguments>${escaped(`"${serviceEntrypoint}" --config "${manifest.configPath}"`)}</arguments>`,
    `  <workingdirectory>${escaped(manifest.installRoot)}</workingdirectory>`,
    `  <logpath>${escaped(manifest.logsPath)}</logpath>`,
    "  <log mode=\"roll\" />", "</service>", ""].join("\n");
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
