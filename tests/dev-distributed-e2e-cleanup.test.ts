import { EventEmitter } from "node:events";
import { existsSync } from "node:fs";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  createIdempotentDevelopmentCleanup,
  installDevelopmentSignalHandlers,
} from "../scripts/dev-distributed-e2e.js";

describe("distributed development gate cleanup", () => {
  it("drains every resource and removes its temporary root exactly once", async () => {
    const temporaryRoot = await mkdtemp(join(tmpdir(), "mycellios-dev-e2e-cleanup-test-"));
    await writeFile(join(temporaryRoot, "state.json"), "{}");
    const events: string[] = [];
    let executions = 0;
    const cleanup = createIdempotentDevelopmentCleanup([
      {
        name: "abort",
        run: () => {
          executions += 1;
          events.push("abort");
        },
      },
      {
        name: "stages",
        run: async () => {
          events.push("stages");
        },
      },
      {
        name: "temporary-root",
        run: async () => {
          events.push("temporary-root");
          await rm(temporaryRoot, { recursive: true, force: true });
        },
      },
    ]);

    await Promise.all([cleanup(), cleanup(), cleanup()]);

    expect(executions).toBe(1);
    expect(events).toEqual(["abort", "stages", "temporary-root"]);
    expect(existsSync(temporaryRoot)).toBe(false);
  });

  it("continues draining after one cleanup step fails", async () => {
    const temporaryRoot = await mkdtemp(join(tmpdir(), "mycellios-dev-e2e-cleanup-test-"));
    let finalStepRuns = 0;
    const cleanup = createIdempotentDevelopmentCleanup([
      {
        name: "broken-stage",
        run: () => {
          throw new Error("stage stop failed");
        },
      },
      {
        name: "temporary-root",
        run: async () => {
          finalStepRuns += 1;
          await rm(temporaryRoot, { recursive: true, force: true });
        },
      },
    ]);

    await expect(cleanup()).rejects.toThrow("development_gate_cleanup_failed");
    await expect(cleanup()).rejects.toThrow("development_gate_cleanup_failed");
    expect(finalStepRuns).toBe(1);
    expect(existsSync(temporaryRoot)).toBe(false);
  });

  it("turns the first SIGINT or SIGTERM into one idempotent cleanup", async () => {
    const source = new EventEmitter();
    const binding = installDevelopmentSignalHandlers(source);
    let cleanupRuns = 0;
    const cleanup = createIdempotentDevelopmentCleanup([{
      name: "all-resources",
      run: () => {
        cleanupRuns += 1;
      },
    }]);
    const afterSignal = binding.signal.then(() => cleanup());

    source.emit("SIGINT");
    source.emit("SIGTERM");
    await afterSignal;
    await cleanup();

    await expect(binding.signal).resolves.toBe("SIGINT");
    expect(cleanupRuns).toBe(1);
    expect(source.listenerCount("SIGINT")).toBe(0);
    expect(source.listenerCount("SIGTERM")).toBe(0);
  });
});
