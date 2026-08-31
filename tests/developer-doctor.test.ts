import { describe, expect, it } from "vitest";
import { evaluateDeveloperEnvironment } from "../scripts/developer-doctor.mjs";

describe("developer doctor", () => {
  it("accepts the complete supported toolchain", () => {
    const result = evaluateDeveloperEnvironment(input());
    expect(result.ok).toBe(true);
    expect(result.fullRuntimeReady).toBe(true);
  });

  it("fails required setup but reports Python as an optional runtime gap", () => {
    const result = evaluateDeveloperEnvironment({ ...input(), nodeVersion: "v22.0.0", pythonVersion: null });
    expect(result.ok).toBe(false);
    expect(result.fullRuntimeReady).toBe(false);
    expect(result.checks.find((check) => check.id === "python")?.required).toBe(false);
  });

  it("rejects an alternative package-manager lock", () => {
    const result = evaluateDeveloperEnvironment({ ...input(), hasAlternativeLock: true });
    expect(result.ok).toBe(false);
  });

  it("reports platform-specific physical capabilities without changing readiness", () => {
    const result = evaluateDeveloperEnvironment({ ...input(), platform: "linux", powershellVersion: null });
    expect(result.capabilities).toMatchObject({ controlPlane: true, pythonRuntime: true, portableTwoHostPreflight: true, windowsPhysicalScripts: false });
  });
});

function input() { return { nodeVersion: "v24.4.0", npmVersion: "10.9.8", pythonVersion: "Python 3.12.9", hasLockfile: true, hasAlternativeLock: false, gitVersion: "git version 2.45.0", powershellVersion: "7.4.0", platform: "win32" as const }; }
