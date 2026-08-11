import { afterEach, describe, expect, it, vi } from "vitest";
import { terminateProcessTree } from "../src/distribution/process-tree.js";

describe("process tree bounded termination", () => {
  afterEach(() => vi.restoreAllMocks());

  it.runIf(process.platform !== "win32")("escalates a supervised process group to SIGKILL", async () => {
    const kill = vi.spyOn(process, "kill").mockImplementation(() => true);
    const direct = vi.fn(() => true);
    await terminateProcessTree({ pid: 42_424, kill: direct }, true, 100);
    expect(kill).toHaveBeenCalledWith(-42_424, "SIGKILL");
    expect(direct).not.toHaveBeenCalled();
  });

  it.runIf(process.platform !== "win32")("uses graceful SIGTERM before forced cleanup", async () => {
    const kill = vi.spyOn(process, "kill").mockImplementation(() => true);
    await terminateProcessTree({ pid: 42_425, kill: () => true }, false, 100);
    expect(kill).toHaveBeenCalledWith(-42_425, "SIGTERM");
  });
});
