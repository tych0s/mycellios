import { describe, expect, it } from "vitest";
import {
  AUTOMATIC_UPDATE_CHECK_INTERVAL_MS,
  automaticUpdateRetryDelayMs,
  canInstallAutomaticUpdate,
  summarizeAutomaticUpdateError,
  trackActiveStage,
} from "../src/update/update-recovery.js";

describe("automatic desktop update recovery", () => {
  it("checks for unattended repairs at least every fifteen minutes", () => {
    expect(AUTOMATIC_UPDATE_CHECK_INTERVAL_MS).toBe(15 * 60_000);
  });

  it("installs a downloaded repair when the node is idle", () => {
    expect(canInstallAutomaticUpdate({
      updateReady: true,
      quitting: false,
      activeJobs: 0,
      activeStages: 0,
      componentUpdateBusy: false,
    })).toBe(true);
  });

  it("never interrupts active inference or a distributed stage", () => {
    expect(canInstallAutomaticUpdate({
      updateReady: true,
      quitting: false,
      activeJobs: 1,
      activeStages: 0,
      componentUpdateBusy: false,
    })).toBe(false);
    expect(canInstallAutomaticUpdate({
      updateReady: true,
      quitting: false,
      activeJobs: 0,
      activeStages: 1,
      componentUpdateBusy: false,
    })).toBe(false);
    expect(canInstallAutomaticUpdate({
      updateReady: true,
      quitting: false,
      activeJobs: 0,
      activeStages: 0,
      componentUpdateBusy: true,
    })).toBe(false);
  });

  it("does nothing before a download is ready or while quitting", () => {
    expect(canInstallAutomaticUpdate({
      updateReady: false,
      quitting: false,
      activeJobs: 0,
      activeStages: 0,
      componentUpdateBusy: false,
    })).toBe(false);
    expect(canInstallAutomaticUpdate({
      updateReady: true,
      quitting: true,
      activeJobs: 0,
      activeStages: 0,
      componentUpdateBusy: false,
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

  it("releases a stage that exits before readiness and wakes the updater", async () => {
    let resolveExit!: () => void;
    const handle = {
      exited: new Promise<void>((resolve) => {
        resolveExit = resolve;
      }),
    };
    const activeStages = new Set<typeof handle>();
    let idleNotifications = 0;

    trackActiveStage(activeStages, handle, () => {
      idleNotifications += 1;
    });
    expect(activeStages.size).toBe(1);

    resolveExit();
    await handle.exited;
    await Promise.resolve();

    expect(activeStages.size).toBe(0);
    expect(idleNotifications).toBe(1);
  });

  it("releases rejected and concurrent stage exits exactly when the set becomes idle", async () => {
    let resolveFirst!: () => void;
    let rejectSecond!: (reason: Error) => void;
    const first = {
      exited: new Promise<void>((resolve) => {
        resolveFirst = resolve;
      }),
    };
    const second = {
      exited: new Promise<void>((_resolve, reject) => {
        rejectSecond = reject;
      }),
    };
    const activeStages = new Set<typeof first | typeof second>();
    let idleNotifications = 0;
    const onIdle = () => {
      idleNotifications += 1;
    };

    trackActiveStage(activeStages, first, onIdle);
    trackActiveStage(activeStages, second, onIdle);
    resolveFirst();
    await first.exited;
    await Promise.resolve();
    expect(activeStages.size).toBe(1);
    expect(idleNotifications).toBe(0);

    rejectSecond(new Error("startup failed"));
    await second.exited.catch(() => undefined);
    await Promise.resolve();
    expect(activeStages.size).toBe(0);
    expect(idleNotifications).toBe(1);
  });
});
