import { describe, expect, it } from "vitest";
import {
  DEFAULT_DESKTOP_SETTINGS,
  desktopSettingsRequireMigration,
  sanitizeDesktopSettings,
} from "../src/desktop/settings.js";

describe("native-only desktop settings", () => {
  it("migrates legacy local model runtime settings without retaining an external runtime route", () => {
    const stored = {
      ...DEFAULT_DESKTOP_SETTINGS,
      remoteCoordinatorUrl: "https://mycellios.com/",
      contributionEnabled: true,
      adapterMode: "local-model-runtime",
      modelName: "qwen3:0.6b",
      adapterBaseUrl: "http://127.0.0.1:11434",
      modelDigest: "sha256:legacy",
    };

    const migrated = sanitizeDesktopSettings(stored);

    expect(migrated).toEqual({
      ...DEFAULT_DESKTOP_SETTINGS,
      contributionEnabled: true,
    });
    expect(migrated).not.toHaveProperty("adapterMode");
    expect(migrated).not.toHaveProperty("modelName");
    expect(migrated).not.toHaveProperty("adapterBaseUrl");
    expect(migrated).not.toHaveProperty("modelDigest");
    expect(desktopSettingsRequireMigration(stored, migrated)).toBe(true);
  });

  it("keeps current settings stable and clamps the offered memory budget", () => {
    const current = {
      ...DEFAULT_DESKTOP_SETTINGS,
      coordinatorMode: "local" as const,
      computeMode: "gpu-only" as const,
      offeredVramMb: 999_999,
    };

    const sanitized = sanitizeDesktopSettings(current);

    expect(sanitized.offeredVramMb).toBe(262_144);
    expect(sanitized.computeMode).toBe("gpu-only");
    expect(desktopSettingsRequireMigration(sanitized, sanitized)).toBe(false);
  });

  it("continues to reject an insecure non-loopback coordinator", () => {
    expect(() => sanitizeDesktopSettings({
      ...DEFAULT_DESKTOP_SETTINGS,
      remoteCoordinatorUrl: "http://network.example.test",
    })).toThrow("Remote coordinators must use HTTPS/WSS");
  });
});
