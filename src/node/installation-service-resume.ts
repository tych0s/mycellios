import { dirname, join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import type { NodeConfiguration } from "../contracts/node-configuration.js";
import type { NodeInstallationManifest } from "../contracts/node-uninstall.js";
import { registerNativeNodeService, runServiceCommand, type ServiceCommandRunner } from "./service-registration.js";

/** Resume only the service owned by the validated native installation. */
export async function resumeNodeInstallationService(input: {
  manifest: NodeInstallationManifest; config: NodeConfiguration;
  beforeStart: () => Promise<void>; run?: ServiceCommandRunner; pause?: () => Promise<void>;
}): Promise<void> {
  if (input.manifest.platform !== "win32") throw new Error("node_install_resume_requires_windows");
  const run = input.run ?? runServiceCommand;
  const pause = input.pause ?? (() => delay(1_000));
  const query = () => run("sc.exe", ["query", input.manifest.serviceName]);
  const status = await query();
  if (status.code === 1060) {
    await input.beforeStart();
    await registerNativeNodeService({ ...input, run, replaceRetainedServiceFiles: true });
    return;
  }
  if (status.code !== 0) throw new Error("node_install_resume_service_query_failed");
  // SCM display labels are localized. CIM property names remain invariant.
  const nameLiteral = input.manifest.serviceName.replaceAll("'", "''");
  const script = `$ErrorActionPreference='Stop'; $service=Get-CimInstance -ClassName Win32_Service | Where-Object { $_.Name -eq '${nameLiteral}' }; if(@($service).Count -ne 1){throw 'service_not_unique'}; $service | Select-Object Name,PathName,StartName | ConvertTo-Json -Compress`;
  const configuration = await run("powershell.exe", ["-NoProfile", "-NonInteractive", "-EncodedCommand", Buffer.from(script, "utf16le").toString("base64")]);
  const expectedExecutable = join(dirname(input.manifest.serviceDefinitionPath), "MycelliosNode.exe");
  let service: { Name?: unknown; PathName?: unknown; StartName?: unknown } = {};
  try { service = JSON.parse(configuration.stdout.replace(/^\uFEFF/, "")); } catch { /* reject invalid manager output */ }
  const binary = typeof service?.PathName === "string" ? service.PathName.trim().replace(/^"(.*)"$/, "$1") : undefined;
  const account = typeof service?.StartName === "string" ? service.StartName.toLowerCase() : undefined;
  if (configuration.code !== 0 || service?.Name !== input.manifest.serviceName || binary !== expectedExecutable || account !== "nt authority\\localservice") {
    throw new Error("node_install_resume_service_ownership_mismatch");
  }
  if (!/:\s*1\s+STOPPED\b/i.test(status.stdout)) {
    const stopped = await run("sc.exe", ["stop", input.manifest.serviceName]);
    if (stopped.code !== 0 && stopped.code !== 1062) throw new Error("node_install_resume_service_stop_failed");
    let stoppedNow = false;
    for (let attempt = 0; attempt < 60; attempt += 1) {
      const latest = await query();
      if (latest.code === 0 && /:\s*1\s+STOPPED\b/i.test(latest.stdout)) { stoppedNow = true; break; }
      await pause();
    }
    if (!stoppedNow) throw new Error("node_install_resume_service_stop_timed_out");
  }
  await input.beforeStart();
  const started = await run("sc.exe", ["start", input.manifest.serviceName]);
  if (started.code !== 0) throw new Error("node_install_resume_service_start_failed");
}
