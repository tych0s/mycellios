import {
  mkdirSync,
  mkdtempSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { afterEach, describe, expect, it } from "vitest";

const roots: string[] = [];
const script = resolve("scripts/resolve-distribution-python.ps1");

afterEach(() => {
  for (const root of roots.splice(0)) {
    rmSync(root, { recursive: true, force: true });
  }
});

describe.skipIf(process.platform !== "win32")(
  "distribution Python resolver",
  () => {
    it("prefers the sealed standalone Python at the runtime root", () => {
      const root = fixture();
      const standalone = fakePython(root, "python.exe");
      fakePython(root, "Scripts/python.exe");

      expect(resolvePython(root)).toBe(standalone);
    });

    it("accepts a conventional Windows venv when no standalone exists", () => {
      const root = fixture();
      const conventional = fakePython(root, "Scripts/python.exe");

      expect(resolvePython(root)).toBe(conventional);
    });

    it("fails with one actionable error when the runtime is absent", () => {
      const root = fixture();
      const result = runResolver(root);

      expect(result.status).not.toBe(0);
      expect(result.stderr).toContain(
        "Distribution runtime is missing. Run npm run desktop:runtime first.",
      );
    });
  },
);

function fixture(): string {
  const root = mkdtempSync(join(tmpdir(), "mycellios-python-resolver-"));
  roots.push(root);
  return root;
}

function fakePython(root: string, portablePath: string): string {
  const path = join(
    root,
    "runtime",
    "distribution-venv",
    ...portablePath.split("/"),
  );
  mkdirSync(resolve(path, ".."), { recursive: true });
  writeFileSync(path, "");
  return resolve(path);
}

function resolvePython(root: string): string {
  const result = runResolver(root);
  expect(result.status).toBe(0);
  return result.stdout.trim();
}

function runResolver(root: string) {
  return spawnSync(
    "powershell.exe",
    [
      "-NoProfile",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      script,
      "-WorkspacePath",
      root,
    ],
    {
      encoding: "utf8",
      windowsHide: true,
    },
  );
}
