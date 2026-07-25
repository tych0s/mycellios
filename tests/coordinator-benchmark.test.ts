import { describe, expect, it } from "vitest";
import {
  buildCoordinatorBenchmarkTelemetrySnapshot,
  buildCoordinatorBenchmarkInventory,
  coordinatorBenchmarkActivations,
  coordinatorBenchmarkModelIdentity,
  detectNewActiveModels,
  runCoordinatorModelSuite,
} from "../src/benchlab/coordinator-suite.js";
import type { RunIdentity } from "../src/benchlab/history.js";
import type { StoredWorker } from "../src/storage/store.js";

const IDENTITY: RunIdentity = {
  runId: "auto-run-1",
  version: "0.2.19",
  label: "v0.2.19",
  gitCommit: "0123456789abcdef",
  gitBranch: "main",
  gitDirty: false,
  build: {
    release: "0.2.19",
    releaseSource: "override",
    revision: "0123456789abcdef",
    revisionSource: "git",
  },
};

describe("automatic coordinator benchmark", () => {
  it("detects each real model start once and detects it again after a stop", () => {
    const observed = new Set<string>();
    expect(detectNewActiveModels(new Set(["qwen"]), observed)).toEqual(["qwen"]);
    expect(detectNewActiveModels(new Set(["qwen"]), observed)).toEqual([]);
    expect(detectNewActiveModels(new Set(), observed)).toEqual([]);
    expect(detectNewActiveModels(new Set(["qwen"]), observed)).toEqual(["qwen"]);
  });

  it("propagates one real deployment digest and fails closed on conflicts", () => {
    const workers = distributedWorkers();
    const connected = new Set(workers.map((worker) => worker.id));
    const identity = coordinatorBenchmarkModelIdentity(
      { id: "qwen3-0.6b", source: "Qwen/Qwen3-0.6B", revision: "rev-1" },
      workers,
      connected,
    );
    expect(identity.digest).toBe("sha256:model");

    const conflicting = structuredClone(workers);
    conflicting[1]!.capabilities.deployments = [{
      ...conflicting[0]!.capabilities.deployments[0]!,
      deploymentId: "qwen-conflict",
      modelDigest: "sha256:other",
    }];
    expect(coordinatorBenchmarkModelIdentity(
      { id: "qwen3-0.6b", source: "Qwen/Qwen3-0.6B", revision: "rev-1" },
      conflicting,
      connected,
    ).digest).toBeNull();
  });

  it("seals a concrete activation from deployment generation and stage ranges", () => {
    const workers = distributedWorkers();
    const connected = new Set(workers.map((worker) => worker.id));
    const [activation] = coordinatorBenchmarkActivations(
      new Set(["qwen3-0.6b"]),
      workers,
      connected,
    );
    expect(activation).toMatchObject({
      modelId: "qwen3-0.6b",
      modelDigest: "sha256:model",
      participants: [{
        workerId: "worker-a",
        deploymentId: "qwen-pipeline",
        nodeIds: ["node-a", "node-b"],
      }],
    });
    expect(activation?.activationId).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(activation?.topologyDigest).toMatch(/^sha256:[0-9a-f]{64}$/);

    const replacement = structuredClone(workers);
    replacement[0]!.capabilities.deployments[0]!.deploymentId = "qwen-pipeline-2";
    expect(coordinatorBenchmarkActivations(
      new Set(["qwen3-0.6b"]),
      replacement,
      connected,
    )[0]?.activationId).not.toBe(activation?.activationId);
  });

  it("samples live power and free memory without using configured limits", () => {
    const workers = distributedWorkers();
    const snapshot = buildCoordinatorBenchmarkTelemetrySnapshot(
      workers,
      new Set(workers.map((worker) => worker.id)),
      10_000,
    );
    expect(snapshot.atMs).toBe(10_000);
    expect(snapshot.nodes).toEqual([
      expect.objectContaining({
        nodeId: "node-a",
        powerWatts: 70,
        offeredMemoryGb: 4,
        freeOfferedMemoryGb: 4,
      }),
      expect.objectContaining({
        nodeId: "node-b",
        powerWatts: 80,
        offeredMemoryGb: 6,
        freeOfferedMemoryGb: 6,
      }),
    ]);
  });

  it("records real coordinator inference with routed nodes, VRAM, power and latency", async () => {
    const workers = distributedWorkers();
    const connected = new Set(workers.map((worker) => worker.id));
    const fetchImpl = async () => new Response(JSON.stringify({
      id: "job-real",
      model: "qwen3-0.6b",
      choices: [{ message: { content: "respuesta física real" } }],
      usage: { prompt_tokens: 18, completion_tokens: 24, total_tokens: 42 },
      x_network: {
        route_class: "pipeline",
        affinity_hit: false,
        reused_kv_tokens: 0,
        ttft_ms: 420,
        active_ms: 6_000,
      },
    }), { status: 200, headers: { "content-type": "application/json" } });

    const run = await runCoordinatorModelSuite(IDENTITY, {
      cwd: ".",
      coordinatorUrl: "http://127.0.0.1:4180",
      model: {
        id: "qwen3-0.6b",
        source: "Qwen/Qwen3-0.6B",
        revision: "rev-1",
        digest: "sha256:rev-1",
      },
      inventory: (routed) =>
        buildCoordinatorBenchmarkInventory("qwen3-0.6b", workers, connected, routed),
      resolveWorkerId: () => "worker-a",
      fetchImpl,
      samples: 3,
      outputTokens: 24,
      trigger: "automatic-model-start",
    });

    const measurement = run.measurements[0]!;
    expect(run.trigger).toBe("automatic-model-start");
    expect(run.triggerModelId).toBe("qwen3-0.6b");
    expect(measurement.evidence).toBe("physical");
    expect(measurement.inventory.selectedDevices).toBe(2);
    expect(measurement.inventory.offeredMemoryGb).toBe(10);
    expect(measurement.inventory.physicalMemoryGb).toBe(12);
    expect(measurement.inventory.observedPowerWatts).toBe(150);
    expect(measurement.metrics.tokensPerSecond).toBe(4);
    expect(measurement.metrics.ttftMsP95).toBe(420);
    expect(measurement.workload.successfulRequests).toBe(3);
    expect(measurement.workload.routeClasses).toEqual(["pipeline"]);
    expect(measurement.model.digest).toBe("sha256:rev-1");
    expect(measurement.scenarioFingerprint).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(measurement.notes.join(" ")).toContain("no hay datos simulados");
  });

  it("keeps a failed real test without manufacturing performance numbers", async () => {
    const run = await runCoordinatorModelSuite(IDENTITY, {
      cwd: ".",
      coordinatorUrl: "http://127.0.0.1:4180",
      model: { id: "missing", source: "missing", revision: null, digest: null },
      inventory: () => ({
        totalDevices: 0,
        connectedDevices: 0,
        selectedDevices: 0,
        profiles: [],
      }),
      fetchImpl: async () => new Response(JSON.stringify({
        error: { message: "No model route is active" },
      }), { status: 503 }),
      samples: 2,
      warmupSamples: 0,
      retryDelayMs: 0,
    });

    const measurement = run.measurements[0]!;
    expect(run.status).toBe("failed");
    expect(measurement.metrics.tokensPerSecond).toBeNull();
    expect(measurement.metrics.ttftMsP95).toBeNull();
    expect(measurement.workload.successfulRequests).toBe(0);
    expect(measurement.notes.join(" ")).toContain("No model route is active");
    expect(measurement.comparison.baselineRunId).toBeNull();
  });
});

function distributedWorkers(): StoredWorker[] {
  return [
    worker("worker-a", "node-a", "NVIDIA RTX A", 4_096, 6_144, 70, true),
    worker("worker-b", "node-b", "AMD Radeon B", 6_144, 6_144, 80, false),
  ];
}

function worker(
  id: string,
  nodeId: string,
  gpuModel: string,
  offeredVramMb: number,
  physicalVramMb: number,
  powerW: number,
  root: boolean,
): StoredWorker {
  return {
    id,
    status: "online",
    reliability: 1,
    jobsCompleted: 0,
    lastSeenAt: Date.now(),
    identityKind: "device",
    identityId: id,
    capabilities: {
      region: "local",
      agentVersion: "0.2.19",
      gpus: [{
        id: `${id}-gpu`,
        vendor: gpuModel.startsWith("NVIDIA") ? "nvidia" : "amd",
        model: gpuModel,
        physicalVramMb,
        offeredVramMb,
        freeOfferedVramMb: offeredVramMb,
        utilizationPct: 55,
        temperatureC: 62,
        powerW,
      }],
      limits: {
        maxConcurrency: 1,
        maxPowerW: 100,
        pauseWhenForeground: false,
      },
      deployments: root ? [{
        deploymentId: "qwen-pipeline",
        model: "qwen3-0.6b",
        modelDigest: "sha256:model",
        mode: "replica",
        adapter: "local-model-runtime",
        peakVramMb: 8_192,
        contextLimit: 4_096,
        maxConcurrency: 1,
        freeSlots: 1,
        tokensPerSecond: 4,
        throughputSource: "measured",
        ttftMs: 420,
        dataLocality: "local",
        execution: {
          deviceType: "gpu",
          backend: "cuda",
          deviceName: gpuModel,
          precision: "bf16",
          fallback: false,
          stages: [
            {
              nodeId: "node-a",
              stageIndex: 0,
              layerStart: 0,
              layerEnd: 14,
              deviceType: "gpu",
              backend: "cuda",
              deviceName: "NVIDIA RTX A",
              precision: "bf16",
              fallback: false,
            },
            {
              nodeId: "node-b",
              stageIndex: 1,
              layerStart: 14,
              layerEnd: 28,
              deviceType: "gpu",
              backend: "rocm",
              deviceName: "AMD Radeon B",
              precision: "bf16",
              fallback: false,
            },
          ],
        },
      }] : [],
      network: { coordinatorRttMs: 3, uplinkMbps: 500, downlinkMbps: 500 },
      distributedExecutor: {
        protocol: "gdlp-worker-tunnel/2",
        nodeId,
        stageHost: "127.0.0.1",
        stagePort: 9_000,
        runtime: "python-safetensors",
        computeMode: "gpu-only",
        cpuEligible: false,
      },
    },
  };
}
