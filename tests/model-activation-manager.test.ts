import { describe, expect, it } from "vitest";
import {
  AutomaticModelActivationManager,
  DynamicModelActivationManager,
  plannedStageArtifactBytes,
  runWithReservationHeartbeat,
  type AutomaticModelRunner,
} from "../src/coordinator/model-activation-manager.js";
import { parseAutoDistributionConfig } from "../src/distribution/auto-distribute.js";
import type { AutoDistributionConfig } from "../src/distribution/auto-distribute.js";
import type { StoredRequestedModel } from "../src/storage/store.js";

describe("automatic model activation manager", () => {
  it("rejects activation while shutdown drains and after it has completed", async () => {
    let launches = 0;
    let finishDrain!: () => void;
    const runner: AutomaticModelRunner = async (_config, signal) => {
      launches += 1;
      if (launches > 1) throw new Error("unexpected_second_launch");
      await new Promise<void>((resolve) => {
        signal.addEventListener("abort", () => { finishDrain = resolve; }, { once: true });
      });
    };
    const manager = new AutomaticModelActivationManager(baseConfig(), process.cwd(), process.env, runner);
    await manager.initialize();
    const running = manager.activate(requestedModel());
    const closing = manager.close();
    try {
      await expect(manager.activate({ ...requestedModel(), id: "another-model" })).rejects.toThrow("automatic_activation_manager_closed");
      expect(manager.capacityNodesForModel(requestedModel().id)).toEqual([]);
    } finally {
      finishDrain();
      await closing;
      await running;
    }
    await expect(manager.activate(requestedModel())).rejects.toThrow("automatic_activation_manager_closed");
    expect(launches).toBe(1);
    expect(manager.isBusy()).toBe(false);
  });

  it("does not restart a closed dynamic manager from a later reconciliation tick", async () => {
    const phases: string[] = [];
    const manager = new DynamicModelActivationManager({
      snapshot: () => ({ capacityNodes: [{ id: "node-a", availableVramMiB: 4096 }], config: null }),
      resolveManagedAgent: () => undefined,
      onProgress: (_modelId, event) => phases.push(event.phase),
    });
    await manager.initialize();
    await manager.close();
    await manager.refresh();
    await expect(manager.activate(requestedModel())).rejects.toThrow("automatic_activation_manager_closed");
    expect(phases).toEqual([]);
    expect(manager.capacityNodesForModel(requestedModel().id)).toEqual([]);
    expect(manager.isBusy()).toBe(false);
    await manager.close();
  });

  it("keeps a prepared route alive and fails closed when renewal is rejected", async () => {
    let renewals = 0;
    const completed = await runWithReservationHeartbeat(
      new Promise<string>((resolve) => setTimeout(() => resolve("ready"), 230)),
      () => { renewals += 1; return true; },
      100,
    );
    expect(completed).toBe("ready");
    expect(renewals).toBeGreaterThanOrEqual(2);

    let aborted: Error | null = null;
    await expect(runWithReservationHeartbeat(
      new Promise<never>(() => undefined),
      () => false,
      100,
      (error) => { aborted = error; },
    )).rejects.toThrow("route_reservation_heartbeat_rejected");
    expect(aborted).toMatchObject({ message: "route_reservation_heartbeat_rejected" });
  });
  it("accounts exact planned artifact bytes without double-counting tied endpoints", () => {
    const model = {
      id: "tiny",
      layers: [0, 1, 2, 3].map((index) => ({
        index,
        weightBytes: 100 + index,
        activationElements: 8,
        kvBytesPerToken: 4,
        decodeMsAtUnit: 1,
        prefillMsPerTokenAtUnit: 1,
      })),
      embeddingBytes: 50,
      lmHeadBytes: 50,
      tiedEmbeddingAndHead: true,
      runtimeOverheadBytesPerStage: 10,
      embeddingDecodeMsAtUnit: 1,
      lmHeadDecodeMsAtUnit: 1,
      embeddingPrefillMsPerTokenAtUnit: 1,
      lmHeadPrefillMsPerTokenAtUnit: 1,
    };
    expect(plannedStageArtifactBytes(model, 0, 2)).toBe(50 + 100 + 101);
    expect(plannedStageArtifactBytes(model, 2, 4)).toBe(102 + 103 + 50);
    expect(plannedStageArtifactBytes(model, 0, 4)).toBe(50 + 100 + 101 + 102 + 103);
  });

  it("replaces stale progress when the verified executor topology is not ready", async () => {
    const persisted = [{
      phase: "active",
      message: "A previous activation completed.",
      at: "2026-07-24T08:18:45.682Z",
      state: "completed" as const,
    }];
    const emitted: Array<{ phase: string; state: string; message: string }> = [];
    const manager = new DynamicModelActivationManager({
      snapshot: () => ({
        capacityNodes: [
          { id: "node-a", availableVramMiB: 4_096 },
          { id: "node-b", availableVramMiB: 4_096 },
        ],
        config: null,
        readinessDetails: [
          "node-a: runtime performance verified.",
          "node-b: connected; waiting for a verified runtime performance profile.",
        ],
      }),
      resolveManagedAgent: () => undefined,
      loadProgress: () => persisted,
      onProgress: (_modelId, event) => emitted.push(event),
    });
    await manager.initialize();
    expect(manager.activationProgressForModel("qwen-ui")).toEqual(persisted);

    await expect(manager.activate(requestedModel())).rejects.toThrow(
      "distributed_activation_requires_two_connected_shard_executors",
    );

    expect(manager.activationProgressForModel("qwen-ui")).toEqual([
      expect.objectContaining({ phase: "queued", state: "completed" }),
      expect.objectContaining({
        phase: "failed",
        state: "failed",
        message: "The connected PCs have not finished preparing a verified two-node execution route.",
        details: [
          "node-a: runtime performance verified.",
          "node-b: connected; waiting for a verified runtime performance profile.",
        ],
      }),
    ]);
    expect(emitted.map((event) => event.phase)).toEqual(["queued", "failed"]);
  });

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
