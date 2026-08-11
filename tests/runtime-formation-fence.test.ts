import { describe, expect, it, vi } from "vitest";
import { workerConfigSchema } from "../src/contracts/schemas.js";
import { WorkerAgent } from "../src/worker/agent.js";
import { normalizeExecutorIsolationPolicy } from "../src/distribution/process-environment.js";

describe("worker runtime formation fencing", () => {
  it("serializes prepares, increments generation, and revokes older launch authority immediately", async () => {
    const agent = new WorkerAgent(workerConfigSchema.parse({
      region: "test",
      offeredVramMb: 512,
      limits: { maxConcurrency: 1, pauseWhenForeground: false },
      adapter: { kind: "mock", developmentOnly: true, model: "test",
        tokensPerSecond: 1, ttftMs: 1, failureRate: 0 },
      deployment: { contextLimit: 1_024 },
    }), {
      coordinatorUrl: "http://127.0.0.1:9999",
      reconnect: false,
      hardwareProbe: async () => ({ hostname: "node", platform: process.platform,
        ramMb: 1_024, gpus: [] }),
      logger: { info() {}, warn() {}, error() {} },
    });
    let releaseFirst!: () => void;
    const firstBlocked = new Promise<void>((resolve) => { releaseFirst = resolve; });
    const calls: Array<{ requestId: string; generation: number }> = [];
    const harness = agent as unknown as {
      handleServerMessage(input: unknown): Promise<void>;
      prepareDistributedRuntime(requestId: string, input: unknown, generation: number): Promise<void>;
      authorizedRuntimeProcesses: Map<string, string>;
      preparedRuntimeProcesses: Map<string, unknown>;
      assertCurrentRuntimePreparation(generation: number): void;
    };
    harness.prepareDistributedRuntime = vi.fn(async (requestId, _input, generation) => {
      calls.push({ requestId, generation });
      if (requestId === "prepare-old") await firstBlocked;
    });
    harness.authorizedRuntimeProcesses.set("old-process", "old-authority");
    harness.preparedRuntimeProcesses.set("old-process", {});

    const oldPrepare = harness.handleServerMessage({
      v: 1, type: "runtime.prepare",
      payload: { requestId: "prepare-old", description: {} },
    });
    await Promise.resolve();
    const newPrepare = harness.handleServerMessage({
      v: 1, type: "runtime.prepare",
      payload: { requestId: "prepare-new", description: {} },
    });

    expect(harness.authorizedRuntimeProcesses.size).toBe(0);
    expect(harness.preparedRuntimeProcesses.size).toBe(0);
    expect(calls).toEqual([{ requestId: "prepare-old", generation: 1 }]);
    releaseFirst();
    await Promise.all([oldPrepare, newPrepare]);
    expect(calls).toEqual([
      { requestId: "prepare-old", generation: 1 },
      { requestId: "prepare-new", generation: 2 },
    ]);
    expect(() => harness.assertCurrentRuntimePreparation(1))
      .toThrow("distributed_runtime_preparation_superseded");
    expect(() => harness.assertCurrentRuntimePreparation(2)).not.toThrow();
  });

  it("rejects a start replayed with another launch or pipeline identity", async () => {
    const started = vi.fn();
    const agent = new WorkerAgent(workerConfigSchema.parse({
      region: "test", offeredVramMb: 512,
      limits: { maxConcurrency: 1, pauseWhenForeground: false },
      adapter: { kind: "mock", developmentOnly: true, model: "test",
        tokensPerSecond: 1, ttftMs: 1, failureRate: 0 },
      deployment: { contextLimit: 1_024 },
    }), {
      coordinatorUrl: "http://127.0.0.1:9999", reconnect: false,
      hardwareProbe: async () => ({ hostname: "node", platform: process.platform,
        ramMb: 1_024, gpus: [] }),
      distributedExecutor: { nodeId: "node-a", stageHost: "127.0.0.1", stagePort: 9_850,
        launchAgent: { id: "fake", start: started } },
      logger: { info() {}, warn() {}, error() {} },
    } as never);
    const processDescription = {
      processId: "process-a", anchor: { memberId: "node-a" },
      isolation: normalizeExecutorIsolationPolicy(),
    };
    const messages: Array<{ type: string; payload: unknown }> = [];
    const harness = agent as unknown as {
      startDistributedRuntime(requestId: string, input: unknown): Promise<void>;
      preparedRuntimeFormation: { launchId: string; pipelineId: string; deploymentGeneration: number } | null;
      authorizedRuntimeProcesses: Map<string, string>;
      preparedRuntimeProcesses: Map<string, unknown>;
      sendMessage(type: string, payload: unknown): void;
    };
    harness.preparedRuntimeFormation = {
      launchId: "launch-current", pipelineId: "pipeline-current", deploymentGeneration: 7,
    };
    harness.authorizedRuntimeProcesses.set("process-a", JSON.stringify(processDescription));
    harness.preparedRuntimeProcesses.set("process-a", processDescription);
    harness.sendMessage = (type, payload) => messages.push({ type, payload });

    await harness.startDistributedRuntime("start-replayed", {
      launchId: "launch-current", pipelineId: "pipeline-current", deploymentGeneration: 6,
      nodeId: "node-a",
      process: processDescription,
    });
    expect(started).not.toHaveBeenCalled();
    expect(messages).toEqual([expect.objectContaining({
      type: "runtime.exited",
      payload: expect.objectContaining({
        exit: expect.objectContaining({ error: "distributed_launch_formation_identity_mismatch" }),
      }),
    })]);
  });
});
