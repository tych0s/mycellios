import { EventEmitter } from "node:events";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { terminateProcessTree } from "../src/distribution/process-tree.js";

const mocks = vi.hoisted(() => ({ spawn: vi.fn() }));
vi.mock("node:child_process", () => ({ spawn: mocks.spawn }));
vi.mock("node:fs", () => ({ existsSync: () => true }));

describe("Windows process tree helper outcomes", () => {
  beforeEach(() => {
    vi.stubGlobal("process", new Proxy(process, {
      get(target, property) {
        if (property === "platform") return "win32";
        if (property === "env") return { ...target.env, SystemRoot: "C:\\Windows" };
        return Reflect.get(target, property);
      },
    }));
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it.each(["failure", "error", "timeout", "success"])(
    "falls back to direct force termination only when taskkill does not succeed: %s",
    async (outcome) => {
      vi.useFakeTimers();
      const helper = Object.assign(new EventEmitter(), { kill: vi.fn(() => true) });
      mocks.spawn.mockReturnValue(helper);
      const root = { pid: 42_424, kill: vi.fn(() => true) };
      const pending = terminateProcessTree(root, true, 100);
      if (outcome === "timeout") await vi.advanceTimersByTimeAsync(100);
      else if (outcome === "error") helper.emit("error", new Error("spawn failed"));
      else helper.emit("close", outcome === "success" ? 0 : 1);
      await pending;
      if (outcome === "success") expect(root.kill).not.toHaveBeenCalled();
      else expect(root.kill).toHaveBeenCalledWith("SIGKILL");
      if (outcome === "timeout") expect(helper.kill).toHaveBeenCalledWith("SIGKILL");
    },
  );
});
