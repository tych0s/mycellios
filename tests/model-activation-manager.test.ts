import { describe, expect, it } from "vitest";
import {
  AutomaticModelActivationManager,
  type AutomaticModelRunner,
} from "../src/coordinator/model-activation-manager.js";
import { parseAutoDistributionConfig } from "../src/distribution/auto-distribute.js";
import type { AutoDistributionConfig } from "../src/distribution/auto-distribute.js";
import type { StoredRequestedModel } from "../src/storage/store.js";

describe("automatic model activation manager", () => {
  it("reserves a healthy executor pool and translates an interface request into a launch", async () => {
    const launchedConfigs: AutoDistributionConfig[] = [];
    const runner: AutomaticModelRunner = async (config, signal) => {
      launchedConfigs.push(config);
      await new Promise<void>((resolve) => {
        signal.addEventListener("abort", () => resolve(), { once: true });
        if (signal.aborted) resolve();
      });
    };
    const manager = new AutomaticModelActivationManager(baseConfig(), process.cwd(), process.env, runner);
    await manager.initialize();

    expect(manager.capacityNodesForModel("qwen-ui")).toEqual([
      { id: "node-a", availableVramMiB: 3_840 },
      { id: "node-b", availableVramMiB: 3_840 },
    ]);

    const running = manager.activate(requestedModel());
    await Promise.resolve();
    expect(manager.isManaging("qwen-ui")).toBe(true);
    expect(manager.capacityNodesForModel("qwen-ui")).toHaveLength(2);
    expect(manager.capacityNodesForModel("another-model")).toHaveLength(0);
    expect(launchedConfigs[0]?.model).toEqual({
      source: "Qwen/Qwen3-0.6B",
      revision: null,
      publicName: "qwen-ui",
    });
    expect(launchedConfigs[0]?.workload.contextTokens).toBe(8_192);
    expect(launchedConfigs[0]?.distribution.minimumStages).toBe(2);

    await manager.close();
    await running;
    expect(manager.isBusy()).toBe(false);
  });

  it("does not start a second model while the executor pool is occupied", async () => {
    const runner: AutomaticModelRunner = async (_config, signal) => {
      await new Promise<void>((resolve) => signal.addEventListener("abort", () => resolve(), { once: true }));
    };
    const manager = new AutomaticModelActivationManager(baseConfig(), process.cwd(), process.env, runner);
    await manager.initialize();
    void manager.activate(requestedModel());

    await expect(manager.activate({ ...requestedModel(), id: "second" })).rejects.toThrow(
      "automatic_activation_busy:qwen-ui",
    );
    await manager.close();
  });
});

function baseConfig() {
  return parseAutoDistributionConfig({
    schema: "gdlp-auto-distribute/1",
    model: { source: "Qwen/Qwen3-0.6B", revision: null, publicName: "base" },
    nodes: [
      {
        id: "node-a",
        region: "test",
        endpoint: { host: "127.0.0.1", port: 24_101 },
        memoryMiB: 4_096,
        reserveMiB: 256,
        agent: { kind: "local" },
      },
      {
        id: "node-b",
        region: "test",
        endpoint: { host: "127.0.0.1", port: 24_102 },
        memoryMiB: 4_096,
        reserveMiB: 256,
        agent: { kind: "local" },
      },
    ],
    distribution: { minimumStages: 2, maximumStages: 2, allowLossyActivation: false },
    workload: {
      promptTokens: 32,
      outputTokens: 8,
      contextTokens: 4_096,
      concurrentSequences: 1,
      minRouteAvailability: 0.9,
      batchWindowMs: 2,
      p95: true,
    },
    runtime: {
      pythonExecutable: "python",
      pythonPath: "python",
      hfHome: "runtime/hf-cache",
      apiEndpoint: { host: "127.0.0.1", port: 8_192 },
      apiAdvertiseHost: "127.0.0.1",
      returnEndpoint: { host: "127.0.0.1", port: 30_192 },
      returnBindHost: "127.0.0.1",
      threadsPerStage: 1,
      connectTimeoutSeconds: 30,
      readinessTimeoutMs: 60_000,
      maxOutputTokens: 256,
    },
    canary: { prompt: "OK", maxTokens: 1, timeoutMs: 30_000 },
  });
}

function requestedModel(): StoredRequestedModel {
  return {
    id: "qwen-ui",
    source: "Qwen/Qwen3-0.6B",
    revision: null,
    contextTokens: 8_192,
    minimumNodes: 2,
    autoActivate: true,
    profile: null,
    profileError: null,
    activationRequestedAt: Date.now(),
    activationError: null,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}
