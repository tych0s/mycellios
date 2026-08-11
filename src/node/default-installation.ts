import { join as posixJoin } from "node:path/posix";
import { join as windowsJoin } from "node:path/win32";
import { nodeInstallationManifestSchema, type NodeInstallationManifest } from "../contracts/node-uninstall.js";

export function defaultNodeInstallationManifest(input: {
  platform: "linux" | "darwin" | "win32";
  installRoot?: string;
  programData?: string;
}): NodeInstallationManifest {
  if (input.platform === "linux") {
    const root = input.installRoot ?? "/opt/mycellios";
    return nodeInstallationManifestSchema.parse({ schema: "mycellios-node-installation/1", platform: "linux", serviceName: "mycellios-node",
      installRoot: root, nodeExecutable: posixJoin(root, "bin/node"), helperEntrypoint: posixJoin(root, "app/node/uninstall-main.js"),
      serviceDefinitionPath: "/etc/systemd/system/mycellios-node.service", configPath: "/etc/mycellios/node.json",
      identityPath: "/var/lib/mycellios/identity/node.json", cachePath: "/var/cache/mycellios", logsPath: "/var/log/mycellios",
      statePath: "/var/lib/mycellios/state", receiptPath: "/var/lib/mycellios/receipts/uninstall.json" });
  }
  if (input.platform === "darwin") {
    const root = input.installRoot ?? "/Library/Application Support/Mycellios/Product";
    const data = "/Library/Application Support/Mycellios";
    return nodeInstallationManifestSchema.parse({ schema: "mycellios-node-installation/1", platform: "darwin", serviceName: "io.mycellios.node",
      installRoot: root, nodeExecutable: posixJoin(root, "bin/node"), helperEntrypoint: posixJoin(root, "app/node/uninstall-main.js"),
      serviceDefinitionPath: "/Library/LaunchDaemons/io.mycellios.node.plist", configPath: posixJoin(data, "Configuration/node.json"),
      identityPath: posixJoin(data, "Identity/node.json"), cachePath: "/Library/Caches/io.mycellios.node", logsPath: "/Library/Logs/Mycellios",
      statePath: posixJoin(data, "State"), receiptPath: posixJoin(data, "Receipts/uninstall.json") });
  }
  const root = input.installRoot ?? "C:\\Program Files\\Mycellios";
  const data = input.programData ?? process.env.PROGRAMDATA ?? "C:\\ProgramData";
  return nodeInstallationManifestSchema.parse({ schema: "mycellios-node-installation/1", platform: "win32", serviceName: "MycelliosNode",
    installRoot: root, nodeExecutable: windowsJoin(root, "bin\\node.exe"), helperEntrypoint: windowsJoin(root, "app\\node\\uninstall-main.js"),
    serviceDefinitionPath: windowsJoin(data, "Mycellios\\Service\\definition.json"), configPath: windowsJoin(data, "Mycellios\\Configuration\\node.json"),
    identityPath: windowsJoin(data, "Mycellios\\Identity\\node.json"), cachePath: windowsJoin(data, "Mycellios\\Cache"), logsPath: windowsJoin(data, "Mycellios\\Logs"),
    statePath: windowsJoin(data, "Mycellios\\State"), receiptPath: windowsJoin(data, "Mycellios\\Receipts\\uninstall.json") });
}
