import { describe, expect, it } from "vitest";
import { resolve } from "node:path";
import { evaluateDeveloperEnvironment, inspectDeveloperEnvironment } from "../scripts/developer-doctor.mjs";

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

  it("probes npm's real CLI through Node on Windows, including paths with spaces", () => {
    const calls: Array<[string, string[]]> = [];
    const cli = "C:/Program Files/nodejs/node_modules/npm/bin/npm-cli.js";
    const result = inspectDeveloperEnvironment(process.cwd(), {
      platform: "win32", environment: { npm_execpath: cli },
      nodeExecutable: "C:/Program Files/nodejs/node.exe",
      commandVersion: (command, args) => {
        calls.push([command, args]);
        return args[0] === cli ? "10.9.8" : null;
      },
    });
    expect(result.checks.find((check) => check.id === "npm")).toMatchObject({ ok: true, detail: "10.9.8" });
    expect(calls[0]).toEqual(["C:/Program Files/nodejs/node.exe", [cli, "--version"]]);
    expect(calls.some(([command]) => command === "npm.cmd")).toBe(false);
  });

  it("uses a constant Windows shell command when invoked outside npm", () => {
    const calls: Array<[string, string[]]> = [];
    inspectDeveloperEnvironment(process.cwd(), {
      platform: "win32", environment: {},
      commandVersion: (command, args) => { calls.push([command, args]); return null; },
    });
    expect(calls[0]).toEqual(["cmd.exe", ["/d", "/s", "/c", "npm --version"]]);
  });

  it("prefers the supported managed Python over an incompatible system interpreter", () => {
    const managed = resolve("managed-python", "python.exe");
    const result = inspectDeveloperEnvironment(process.cwd(), {
      platform: "win32", environment: {},
      commandVersion: (command, args) => {
        if (command === "python") return "Python 3.14.0";
        if (command === "uv" && args[0] === "python") return managed;
        if (command === managed) return "Python 3.12.13";
        return null;
      },
    });
    expect(result.checks.find((check) => check.id === "python")).toMatchObject({ ok: true, detail: "Python 3.12.13" });
  });

  it("discovers an explicitly selected Mycellios runtime before system Python", () => {
    const runtime = resolve("runtime with spaces");
    const calls: string[] = [];
    const result = inspectDeveloperEnvironment(process.cwd(), {
      platform: "win32", environment: { MYCELLIOS_DESKTOP_RUNTIME_ROOT: runtime },
      commandVersion: (command) => {
        calls.push(command);
        return command === resolve(runtime, "python.exe") ? "Python 3.12.13" : null;
      },
    });
    expect(result.capabilities.pythonRuntime).toBe(true);
    expect(calls).not.toContain("python");
  });

  it.each([false, true])("finds a Windows venv without Python on PATH (configured root: %s)", (configured) => {
    const workspace = process.cwd();
    const runtime = configured ? resolve("runtime with spaces") : resolve(workspace, "runtime", "distribution-venv");
    const venvPython = resolve(runtime, "Scripts", "python.exe");
    const calls: string[] = [];
    const result = inspectDeveloperEnvironment(workspace, {
      platform: "win32", environment: configured ? { MYCELLIOS_DESKTOP_RUNTIME_ROOT: runtime } : {},
      commandVersion: (command) => {
        calls.push(command);
        return command === venvPython ? "Python 3.12.13" : null;
      },
    });
    expect(result.checks.find((check) => check.id === "python")).toMatchObject({ ok: true, detail: "Python 3.12.13" });
    expect(result.capabilities.pythonRuntime).toBe(true);
    expect(calls.filter((command) => ["py", "python", "uv"].includes(command))).toEqual([]);
  });
});

function input() { return { nodeVersion: "v24.4.0", npmVersion: "10.9.8", pythonVersion: "Python 3.12.9", hasLockfile: true, hasAlternativeLock: false, gitVersion: "git version 2.45.0", powershellVersion: "7.4.0", platform: "win32" as const }; }
