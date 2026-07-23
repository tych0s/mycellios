import { describe, expect, it } from "vitest";
import {
  automaticUpdateRetryDelayMs,
  canInstallAutomaticUpdate,
  summarizeAutomaticUpdateError,
} from "../src/desktop/update-recovery.js";

describe("automatic desktop update recovery", () => {
  it("installs a downloaded repair when the node is idle", () => {
    expect(canInstallAutomaticUpdate({
      updateReady: true,
      quitting: false,
      activeJobs: 0,
      activeStages: 0,
    })).toBe(true);
  });

  it("never interrupts active inference or a distributed stage", () => {
    expect(canInstallAutomaticUpdate({
      updateReady: true,
      quitting: false,
      activeJobs: 1,
      activeStages: 0,
    })).toBe(false);
    expect(canInstallAutomaticUpdate({
      updateReady: true,
      quitting: false,
      activeJobs: 0,
      activeStages: 1,
    })).toBe(false);
  });

  it("does nothing before a download is ready or while quitting", () => {
    expect(canInstallAutomaticUpdate({
      updateReady: false,
      quitting: false,
      activeJobs: 0,
      activeStages: 0,
    })).toBe(false);
    expect(canInstallAutomaticUpdate({
      updateReady: true,
      quitting: true,
      activeJobs: 0,
      activeStages: 0,
    })).toBe(false);
  });

  it("retries failed checks forever with a bounded backoff", () => {
    expect(automaticUpdateRetryDelayMs(0)).toBe(30_000);
    expect(automaticUpdateRetryDelayMs(1)).toBe(120_000);
    expect(automaticUpdateRetryDelayMs(2)).toBe(300_000);
    expect(automaticUpdateRetryDelayMs(3)).toBe(900_000);
    expect(automaticUpdateRetryDelayMs(4)).toBe(1_800_000);
    expect(automaticUpdateRetryDelayMs(400)).toBe(1_800_000);
    expect(automaticUpdateRetryDelayMs(Number.NaN)).toBe(30_000);
  });

  it("keeps updater errors useful and bounded", () => {
    expect(summarizeAutomaticUpdateError(
      "System.AggregateException: failed ---> System.OutOfMemoryException: no memory\n at RegexRunner",
    )).toBe("The Windows updater ran out of memory while checking the feed.");
    expect(summarizeAutomaticUpdateError(`network failed ${"x".repeat(500)}`)).toHaveLength(320);
  });
});
