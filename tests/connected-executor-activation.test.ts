import { describe, expect, it } from "vitest";
import { buildConnectedExecutorActivationSnapshot } from "../src/coordinator/connected-executor-activation.js";
import { parseAutoDistributionConfig } from "../src/distribution/auto-distribute.js";
import type { StoredWorker } from "../src/storage/store.js";

describe("connected executor activation", () => {
  it("uses only connected unique desktop executors as model capacity", () => {
    const workers = [
      worker("worker-a", "node-a", "192.168.1.10", 9_850, 4_096, 3_500, 12),
      worker("worker-b", "node-b", "192.168.1.11", 9_851, 6_144, 5_000, 20),
      worker("worker-duplicate", "node-a", "192.168.1.12", 9_852, 8_192, 8_000, 5),
      worker("worker-offline", "node-offline", "192.168.1.13", 9_853, 8_192, 8_000, 5),
    ];
    const snapshot = buildConnectedExecutorActivationSnapshot(
      baseConfig(),
      workers,
      new Set(["worker-a", "worker-b", "worker-duplicate"]),
    );

    expect(snapshot.capacityNodes).toEqual([
      { id: "node-a", availableVramMiB: 3_500 },
      { id: "node-b", availableVramMiB: 5_000 },
    ]);
    expect(snapshot.config?.nodes.map((node) => ({
      id: node.id,
      endpoint: node.endpoint,
      memoryMiB: node.memoryMiB,
      agent: node.agent,
    }))).toEqual([
      {
        id: "node-a",
        endpoint: { host: "192.168.1.10", port: 9_850 },
        memoryMiB: 4_096,
        agent: { kind: "managed" },
      },
      {
        id: "node-b",
        endpoint: { host: "192.168.1.11", port: 9_851 },
        memoryMiB: 6_144,
        agent: { kind: "managed" },
      },
    ]);
    expect(snapshot.config?.runtime.apiAdvertiseHost).toBe("192.168.1.10");
    expect(snapshot.config?.runtime.returnEndpoint.host).toBe("192.168.1.10");
    expect(snapshot.config?.links).toHaveLength(2);
    expect(snapshot.config?.links.every((link) => link.availability === 1)).toBe(true);
  });

  it("reports capacity but withholds a launch topology until two executors connect", () => {
    const snapshot = buildConnectedExecutorActivationSnapshot(
      baseConfig(),
      [worker("worker-a", "node-a", "127.0.0.1", 9_850, 4_096, 3_500, 10)],
      new Set(["worker-a"]),
    );
    expect(snapshot.capacityNodes).toEqual([{ id: "node-a", availableVramMiB: 3_500 }]);
    expect(snapshot.config).toBeNull();
  });
});

function worker(
  id: string,
  nodeId: string,
  stageHost: string,
  stagePort: number,
  offeredVramMb: number,
  freeOfferedVramMb: number,
  coordinatorRttMs: number,
): StoredWorker {
  return {
    id,
    status: "online",
    reliability: 0.99,
    jobsCompleted: 0,
    lastSeenAt: Date.now(),
    identityKind: "device",
    identityId: nodeId,
    capabilities: {
      region: "test",
      agentVersion: "0.2.16",
      gpus: [{
        id: `${id}-gpu`,
        vendor: "test",
        model: "test",
        physicalVramMb: offeredVramMb,
        offeredVramMb,
        freeOfferedVramMb,
        powerW: 80,
      }],
      limits: { maxConcurrency: 1, pauseWhenForeground: false },
      deployments: [],
      network: { coordinatorRttMs, uplinkMbps: 100, downlinkMbps: 100 },
      distributedExecutor: {
        protocol: "gdlp-worker-tunnel/1",
        nodeId,
        stageHost,
        stagePort,
        runtime: "python-safetensors",
      },
    },
  } as StoredWorker;
}

function baseConfig() {
  return parseAutoDistributionConfig({
    schema: "gdlp-auto-distribute/1",
    model: { source: "Qwen/Qwen3-0.6B", revision: null, publicName: "base" },
    nodes: [
      { id: "slot-a", region: "test", endpoint: { host: "127.0.0.1", port: 22_101 }, memoryMiB: 6_144, reserveMiB: 512, agent: { kind: "local" } },
      { id: "slot-b", region: "test", endpoint: { host: "127.0.0.1", port: 22_102 }, memoryMiB: 6_144, reserveMiB: 512, agent: { kind: "local" } },
    ],
    distribution: { minimumStages: 2, maximumStages: 2, allowLossyActivation: false },
    workload: { promptTokens: 128, outputTokens: 128, contextTokens: 4_096, concurrentSequences: 1, minRouteAvailability: 0.9, batchWindowMs: 2, p95: true },
    runtime: {
      pythonExecutable: "python",
      pythonPath: "python",
      hfHome: "runtime/hf-cache",
      apiEndpoint: { host: "0.0.0.0", port: 9_860 },
      apiAdvertiseHost: "127.0.0.1",
      returnEndpoint: { host: "127.0.0.1", port: 9_861 },
      returnBindHost: "0.0.0.0",
      threadsPerStage: 1,
      connectTimeoutSeconds: 300,
      readinessTimeoutMs: 600_000,
      maxOutputTokens: 2_048,
    },
    canary: { prompt: "OK", maxTokens: 1, timeoutMs: 30_000 },
    coordinator: { url: "http://127.0.0.1:8787", region: "test", maxConcurrency: 1 },
  });
}
