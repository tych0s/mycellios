import { describe, expect, it } from "vitest";
import { workerConfigSchema } from "../src/contracts/schemas.js";
import type { WorkerCapabilities } from "../src/contracts/types.js";
import { WorkerAgent } from "../src/worker/agent.js";

const config = workerConfigSchema.parse({
  region: "test",
  offeredVramMb: 2_048,
  limits: { maxConcurrency: 2, pauseWhenForeground: false },
  adapter: {
    kind: "mock",
    developmentOnly: true,
    model: "drain-test",
    tokensPerSecond: 10,
    ttftMs: 1,
    failureRate: 0,
  },
  deployment: { contextLimit: 2_048 },
});

interface DrainHarness {
  capabilities: WorkerCapabilities | null;
  activeJobs: Map<string, AbortController>;
  activeEvidenceChallenges: Set<string>;
  activeWorkCount: number;
  buildCapabilities(): Promise<WorkerCapabilities>;
  acceptsNewWork(): boolean;
  runRuntimeOperation<T>(operation: () => Promise<T>): Promise<T>;
}

describe("WorkerAgent runtime update drain", () => {
  it("rejects new work without aborting active leases and restores capacity", async () => {
    const agent = new WorkerAgent(config, {
      coordinatorUrl: "http://127.0.0.1:9999",
      reconnect: false,
      logger: { info() {}, warn() {}, error() {} },
    });
    const harness = agent as unknown as DrainHarness;
    harness.capabilities = await harness.buildCapabilities();
    const active = new AbortController();
    harness.activeJobs.set("existing-job", active);

    const release = await agent.beginRuntimeUpdateDrain();

    expect(harness.acceptsNewWork()).toBe(false);
    expect(active.signal.aborted).toBe(false);
    expect(harness.activeJobs.has("existing-job")).toBe(true);
    expect(harness.activeWorkCount).toBe(1);
    expect(harness.capabilities?.deployments[0]?.freeSlots).toBe(0);

    harness.activeJobs.delete("existing-job");
    await release();
    await release();

    expect(harness.acceptsNewWork()).toBe(true);
    expect(harness.capabilities?.deployments[0]?.freeSlots).toBe(2);
  });

  it("keeps runtime preparation and evidence work inside the update drain", async () => {
    const agent = new WorkerAgent(config, {
      coordinatorUrl: "http://127.0.0.1:9999",
      reconnect: false,
      logger: { info() {}, warn() {}, error() {} },
    });
    const harness = agent as unknown as DrainHarness;
    harness.capabilities = await harness.buildCapabilities();

    let finishPreparation!: () => void;
    const preparation = harness.runRuntimeOperation(
      () => new Promise<void>((resolve) => {
        finishPreparation = resolve;
      }),
    );
    harness.activeEvidenceChallenges.add("evidence-in-flight");
    expect(harness.activeWorkCount).toBe(2);

    const release = await agent.beginRuntimeUpdateDrain();
    expect(harness.acceptsNewWork()).toBe(false);
    expect(harness.activeWorkCount).toBe(2);

    finishPreparation();
    await preparation;
    expect(harness.activeWorkCount).toBe(1);
    harness.activeEvidenceChallenges.delete("evidence-in-flight");
    expect(harness.activeWorkCount).toBe(0);

    await release();
    expect(harness.acceptsNewWork()).toBe(true);
  });
});
