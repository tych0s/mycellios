import {
  existsSync,
  mkdtempSync,
  writeFileSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";
import { createExecutorWorkspace } from "../src/distribution/executor-workspace.js";

describe("executor workspace", () => {
  it("allocates unique contained workspaces and removes only each lease", () => {
    const root = mkdtempSync(join(tmpdir(), "mycellios-workspace-test-"));
    try {
      const first = createExecutorWorkspace({
        root,
        launchId: "launch-a",
        processId: "process-a",
      });
      const second = createExecutorWorkspace({
        root,
        launchId: "launch-a",
        processId: "process-a",
      });
      expect(first.path).not.toBe(second.path);
      expect(relative(root, first.path)).not.toMatch(/^\.\./u);
      expect(relative(root, second.path)).not.toMatch(/^\.\./u);

      writeFileSync(join(first.path, "owned.tmp"), "one", "utf8");
      writeFileSync(join(second.path, "owned.tmp"), "two", "utf8");
      expect(first.measure(16)).toEqual({
        bytes: 3,
        entries: 1,
        entryLimitExceeded: false,
      });
      first.cleanup();
      expect(existsSync(first.path)).toBe(false);
      expect(existsSync(second.path)).toBe(true);

      first.cleanup();
      second.cleanup();
      expect(existsSync(second.path)).toBe(false);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("hashes hostile-looking identities instead of treating them as paths", () => {
    const root = mkdtempSync(join(tmpdir(), "mycellios-workspace-path-test-"));
    try {
      const lease = createExecutorWorkspace({
        root,
        launchId: "../../../outside",
        processId: "..\\..\\outside",
      });
      expect(relative(root, lease.path)).not.toMatch(/^\.\./u);
      expect(lease.path).toContain("exec-");
      lease.cleanup();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
