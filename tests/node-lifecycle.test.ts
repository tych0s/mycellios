import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it, vi } from "vitest";
import { NodeLifecycleController } from "../src/node/lifecycle.js";

async function fixture(isProcessAlive: (pid: number) => boolean = () => true) {
  const directory = await mkdtemp(path.join(tmpdir(), "mycellios-lifecycle-"));
  const hooks = {
    start: vi.fn().mockResolvedValue(undefined),
    drain: vi.fn().mockResolvedValue(undefined),
    stop: vi.fn().mockResolvedValue(undefined),
  };
  const controller = new NodeLifecycleController(
    path.join(directory, "state.json"),
    path.join(directory, "instance.lock"),
    hooks,
    () => new Date("2026-08-09T19:00:00.000Z"),
    4242,
    isProcessAlive,
  );
  return { controller, directory, hooks };
}

describe("NodeLifecycleController", () => {
  it("starts only once and persists a running generation", async () => {
    const { controller, hooks } = await fixture();
    const first = await controller.ensureStarted();
    const repeated = await controller.ensureStarted();

    expect(first.started).toBe(true);
    expect(repeated.started).toBe(false);
    expect(first.state).toMatchObject({ status: "running", generation: 1 });
    expect(hooks.start).toHaveBeenCalledTimes(1);
  });

  it("prevents a second live process from starting the runtime", async () => {
    const { controller, directory, hooks } = await fixture(() => true);
    await writeFile(path.join(directory, "instance.lock"), JSON.stringify({
      instanceId: "2bd12ac7-f92c-4619-9ca4-26b4df8e052e",
      pid: 99,
      acquiredAt: "2026-08-09T18:00:00.000Z",
    }));
    const result = await controller.ensureStarted();
    expect(result.started).toBe(false);
    expect(hooks.start).not.toHaveBeenCalled();
  });

  it("recovers a stale lock before starting", async () => {
    const { controller, directory, hooks } = await fixture(() => false);
    await writeFile(path.join(directory, "instance.lock"), JSON.stringify({
      instanceId: "2bd12ac7-f92c-4619-9ca4-26b4df8e052e",
      pid: 99,
      acquiredAt: "2026-08-09T18:00:00.000Z",
    }));
    const result = await controller.ensureStarted();
    expect(result.started).toBe(true);
    expect(hooks.start).toHaveBeenCalledTimes(1);
  });

  it("drains, stops, releases the lock and can restart", async () => {
    const { controller, directory, hooks } = await fixture();
    await controller.ensureStarted();
    const stopped = await controller.drainAndStop();
    const restarted = await controller.ensureStarted();

    expect(stopped).toMatchObject({ status: "stopped", instanceId: null });
    expect(restarted.state.generation).toBe(2);
    expect(hooks.drain).toHaveBeenCalledTimes(1);
    expect(hooks.stop).toHaveBeenCalledTimes(1);
    expect(JSON.parse(await readFile(path.join(directory, "state.json"), "utf8"))).toMatchObject({
      status: "running",
      generation: 2,
    });
  });

  it("fails closed on corrupt locks and records sanitized start failure", async () => {
    const corrupt = await fixture();
    await writeFile(path.join(corrupt.directory, "instance.lock"), "not-json");
    await expect(corrupt.controller.ensureStarted()).rejects.toThrow("node_lifecycle_lock_corrupt");

    const failed = await fixture();
    failed.hooks.start.mockRejectedValue(new Error("sensitive runtime details"));
    await expect(failed.controller.ensureStarted()).rejects.toThrow("node_runtime_start_failed");
    expect(await failed.controller.readState()).toMatchObject({
      status: "failed",
      failureCode: "runtime_start_failed",
    });
    expect(failed.hooks.stop).toHaveBeenCalledTimes(1);
  });

  it("keeps ownership when stop fails so another runtime cannot start", async () => {
    const current = await fixture(() => true);
    await current.controller.ensureStarted();
    current.hooks.stop.mockRejectedValue(new Error("still alive"));
    await expect(current.controller.drainAndStop()).rejects.toThrow("node_runtime_stop_failed");

    const contenderHooks = {
      start: vi.fn().mockResolvedValue(undefined),
      drain: vi.fn().mockResolvedValue(undefined),
      stop: vi.fn().mockResolvedValue(undefined),
    };
    const contender = new NodeLifecycleController(
      path.join(current.directory, "state.json"),
      path.join(current.directory, "instance.lock"),
      contenderHooks,
      () => new Date("2026-08-09T19:01:00.000Z"),
      9898,
      () => true,
    );
    expect((await contender.ensureStarted()).started).toBe(false);
    expect(contenderHooks.start).not.toHaveBeenCalled();
  });
});
