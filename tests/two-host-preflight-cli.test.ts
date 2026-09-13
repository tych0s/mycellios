import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { runTwoHostPreflight } from "../src/distribution/two-host-preflight-cli.js";

const probe = vi.hoisted(() => vi.fn());
vi.mock("node:child_process", async (importOriginal) => ({
  ...await importOriginal<typeof import("node:child_process")>(), execFileSync: probe,
}));

describe("two-host configured interpreter preflight", () => {
  const roots: string[] = [];
  afterEach(() => {
    vi.restoreAllMocks();
    probe.mockReset();
    for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
  });

  it.each(["Python 3.12.13", "Python 3.13.1", "unavailable"])(
    "checks only the configured executable, with a deadline: %s", async (version) => {
      const root = mkdtempSync(join(tmpdir(), "mycellios-preflight-test-"));
      roots.push(root);
      const config = JSON.parse(readFileSync("config/auto-distribute.two-host.example.json", "utf8"));
      const executable = join(root, "prepared runtime", "python.exe");
      config.runtime.pythonExecutable = executable;
      const path = join(root, "config.json");
      writeFileSync(path, JSON.stringify(config));
      if (version === "unavailable") probe.mockImplementation(() => { throw new Error("ENOENT"); });
      else probe.mockReturnValue(version);
      const output = vi.spyOn(process.stdout, "write").mockImplementation(() => true);
      expect(await runTwoHostPreflight(["--config", path, "--json"], {})).toBe(2);
      const report = JSON.parse(String(output.mock.calls[0]![0]));
      expect(report.capabilities.python312).toBe(version === "Python 3.12.13");
      expect(report.dryRun).toBe(true);
      expect(probe).toHaveBeenCalledTimes(1);
      expect(probe).toHaveBeenCalledWith(executable, ["--version"], expect.objectContaining({
        timeout: 5_000, windowsHide: true,
      }));
    },
  );
});
