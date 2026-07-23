import { describe, expect, it } from "vitest";
import { canInstallAutomaticUpdate } from "../src/desktop/update-recovery.js";

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
});
