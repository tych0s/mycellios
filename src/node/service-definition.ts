import { dirname } from "node:path";
import type { NodeConfiguration } from "../contracts/node-configuration.js";

export function linuxSystemdServiceDefinition(input: {
  config: NodeConfiguration;
  configPath: string;
  nodeExecutable: string;
  serviceEntrypoint: string;
  serviceUser?: string;
}): string {
  if (input.config.isolation.mode !== "linux-cgroup-v2") {
    throw new Error("node_service_definition_isolation_mode_mismatch");
  }
  const executable = servicePath(input.nodeExecutable);
  const entrypoint = servicePath(input.serviceEntrypoint);
  const configPath = servicePath(input.configPath ?? "");
  const user = serviceIdentity(input.serviceUser ?? "mycellios");
  const writable = [
    dirname(input.configPath),
    dirname(input.config.coordinator.identityPath),
    input.config.runtime.cachePath,
  ].map(servicePath);
  return [
    "[Unit]",
    "Description=mycellios node",
    "After=network-online.target",
    "Wants=network-online.target",
    "",
    "[Service]",
    "Type=simple",
    `User=${user}`,
    `Group=${user}`,
    `ExecStart=${systemdQuote(executable)} ${systemdQuote(entrypoint)} --config ${systemdQuote(configPath)}`,
    "Restart=on-failure",
    "RestartSec=5s",
    "KillMode=control-group",
    "OOMPolicy=stop",
    `CPUQuota=${input.config.limits.maxCpuPercent}%`,
    `MemoryMax=${input.config.limits.maxRamMiB}M`,
    "TasksMax=256",
    "NoNewPrivileges=true",
    "PrivateTmp=true",
    "ProtectSystem=strict",
    "ProtectHome=true",
    `ReadWritePaths=${writable.map(systemdQuote).join(" ")}`,
    "",
    "[Install]",
    "WantedBy=multi-user.target",
    "",
  ].join("\n");
}

export function macOsLaunchdServiceDefinition(input: {
  config: NodeConfiguration;
  configPath: string;
  nodeExecutable: string;
  serviceEntrypoint: string;
  label?: string;
}): string {
  if (input.config.isolation.mode !== "macos-launchd-limits") {
    throw new Error("node_service_definition_isolation_mode_mismatch");
  }
  const label = plistText(input.label ?? "io.mycellios.node");
  const arguments_ = [
    servicePath(input.nodeExecutable),
    servicePath(input.serviceEntrypoint),
    "--config",
    servicePath(input.configPath ?? ""),
  ];
  const residentBytes = input.config.limits.maxRamMiB * 1024 * 1024;
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
    '<plist version="1.0">',
    "<dict>",
    "  <key>Label</key>", `  <string>${label}</string>`,
    "  <key>ProgramArguments</key>", "  <array>",
    ...arguments_.map((value) => `    <string>${plistText(value)}</string>`),
    "  </array>",
    "  <key>KeepAlive</key><true/>",
    "  <key>RunAtLoad</key><true/>",
    "  <key>ProcessType</key><string>Background</string>",
    "  <key>AbandonProcessGroup</key><false/>",
    "  <key>ThrottleInterval</key><integer>5</integer>",
    "  <key>HardResourceLimits</key>",
    "  <dict>",
    `    <key>ResidentSetSize</key><integer>${residentBytes}</integer>`,
    "    <key>NumberOfProcesses</key><integer>256</integer>",
    "  </dict>",
    "</dict>",
    "</plist>",
    "",
  ].join("\n");
}

function servicePath(value: string): string {
  if (!value || /[\r\n\0]/.test(value) || !/^(?:[A-Za-z]:[\\/]|\/)/.test(value)) {
    throw new Error("node_service_path_is_invalid");
  }
  return value;
}

function serviceIdentity(value: string): string {
  if (!/^[a-z_][a-z0-9_-]{0,31}$/i.test(value)) throw new Error("node_service_identity_is_invalid");
  return value;
}

function systemdQuote(value: string): string {
  return `"${value.replaceAll("\\", "\\\\").replaceAll('"', '\\"')}"`;
}

function plistText(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}
