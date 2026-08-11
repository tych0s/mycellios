import { describe, expect, it } from "vitest";
import { detectInstallerPlatform, nodePresentation, uninstallRetentionChoice } from "./Contribute";

describe("native node contribution entry", () => {
  it.each([
    ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Win32", "/downloads/windows"],
    ["Mozilla/5.0 (Macintosh; Intel Mac OS X)", "MacIntel", "/downloads/macos-arm64"],
    ["Mozilla/5.0 (X11; Linux x86_64)", "Linux x86_64", "/downloads/linux-deb"],
  ])("selects a declared installer without probing localhost", (agent, platform, expected) => {
    expect(detectInstallerPlatform(agent, platform).path).toBe(expected);
  });

  it.each([
    ["loading", "Loading", "paused"],
    ["pairing", "Pairing", "paused"],
    ["canary", "Canary", "paused"],
    ["ready", "Ready", "online"],
    ["reconnecting", "Reconnecting", "paused"],
    ["degraded", "Degraded", "paused"],
    ["failed", "Failed", "error"],
    ["revoked", "Revoked", "error"],
  ] as const)("maps observed %s to an explicit product state", (state, label, tone) => {
    const observed = {
      state,
      build: { version: "1.2.3", sourceRevision: "a".repeat(40) },
      runtime: { ready: state === "ready", abi: "mycellios-distribution-runtime/4", backend: "cuda" },
      observedAt: "2026-08-11T00:00:00.000Z",
    };
    expect(nodePresentation({ connected: true, status: "active", observed } as never)).toMatchObject({ label, tone });
  });

  it("shows reconnecting when a previously observed node is offline", () => {
    const observed = { state: "ready", build: { version: "1.2.3" } };
    expect(nodePresentation({ connected: false, status: "active", observed } as never)).toMatchObject({ label: "Reconnecting", tone: "paused" });
  });

  it("maps only explicit uninstall retention choices", () => {
    expect(uninstallRetentionChoice("KEEP-DATA")).toEqual({ cache: true, logs: true, configuration: true, identity: true });
    expect(uninstallRetentionChoice("DELETE-ALL")).toEqual({ cache: false, logs: false, configuration: false, identity: false });
    expect(uninstallRetentionChoice("keep-data")).toBeNull();
    expect(uninstallRetentionChoice(null)).toBeNull();
  });
});
