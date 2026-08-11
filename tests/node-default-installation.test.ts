import { describe, expect, it } from "vitest";
import { defaultNodeInstallationManifest } from "../src/node/default-installation.js";

describe("default native installation layouts", () => {
  it("uses durable data roots outside the replaceable product on every platform", () => {
    const fixtures = [
      defaultNodeInstallationManifest({ platform: "linux" }),
      defaultNodeInstallationManifest({ platform: "darwin" }),
      defaultNodeInstallationManifest({ platform: "win32", programData: "C:\\ProgramData" }),
    ];
    for (const manifest of fixtures) {
      expect(manifest.configPath.startsWith(manifest.installRoot)).toBe(false);
      expect(manifest.identityPath.startsWith(manifest.installRoot)).toBe(false);
      expect(manifest.statePath.startsWith(manifest.installRoot)).toBe(false);
      expect(manifest.receiptPath.startsWith(manifest.installRoot)).toBe(false);
      expect(manifest.nodeExecutable.startsWith(manifest.installRoot)).toBe(true);
      expect(manifest.helperEntrypoint.startsWith(manifest.installRoot)).toBe(true);
    }
  });

  it("derives canonical manager and service identities", () => {
    expect(defaultNodeInstallationManifest({ platform: "linux" })).toMatchObject({ serviceName: "mycellios-node", serviceDefinitionPath: "/etc/systemd/system/mycellios-node.service" });
    expect(defaultNodeInstallationManifest({ platform: "darwin" })).toMatchObject({ serviceName: "io.mycellios.node", serviceDefinitionPath: "/Library/LaunchDaemons/io.mycellios.node.plist" });
    expect(defaultNodeInstallationManifest({ platform: "win32", programData: "C:\\Data" })).toMatchObject({ serviceName: "MycelliosNode", serviceDefinitionPath: "C:\\Data\\Mycellios\\Service\\definition.json" });
  });
});
