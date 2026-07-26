import { describe, expect, it } from "vitest";
import { buildConnectedExecutorActivationSnapshot } from "../src/coordinator/connected-executor-activation.js";
import { parseAutoDistributionConfig } from "../src/distribution/auto-distribute.js";
import type { StoredWorker } from "../src/storage/store.js";
import {
  createCoordinatorRuntimePerformanceEvidence,
  sealRuntimePerformanceProfile,
  type RuntimePerformanceProfile,
} from "../src/performance/runtime-profile.js";

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
      measuredLinks("node-a", "node-b"),
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
    expect(snapshot.config?.runtime.stagePythonExecutable).toBe("python");
    expect(snapshot.config?.runtime.returnEndpoint.host).toBe("192.168.1.10");
    expect(snapshot.config?.links).toHaveLength(2);
    expect(snapshot.config?.links.every((link) => link.availability === 1)).toBe(true);
    expect(snapshot.config?.links[0]).toMatchObject({
      oneWayLatencyMs: 10,
      jitterP95Ms: 2,
      bandwidthMbps: 80,
    });
    expect(snapshot.config?.nodes[0]).toMatchObject({
      decodeScale: 0.5,
      prefillScale: 0.5,
      codecScale: 0.5,
    });
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

  it("waits for verified GPU capacity instead of distributing a permanent CPU fallback", () => {
    const gpuWorker = worker("worker-gpu", "node-gpu", "node-gpu.relay", 9_850, 4_096, 3_500, 10);
    const cpuWorker = worker("worker-cpu", "node-cpu", "node-cpu.relay", 9_850, 4_096, 3_500, 10);
    cpuWorker.capabilities.gpus[0]!.id = "cpu-memory";
    cpuWorker.capabilities.gpus[0]!.vendor = "cpu";
    cpuWorker.capabilities.gpus[0]!.model = "Intel CPU · system-memory fallback";
    cpuWorker.capabilities.gpus[0]!.physicalVramMb = 0;
    cpuWorker.capabilities.gpus[0]!.sharedMemoryMb = 4_096;

    const snapshot = buildConnectedExecutorActivationSnapshot(
      baseConfig(),
      [gpuWorker, cpuWorker],
      new Set([gpuWorker.id, cpuWorker.id]),
      measuredLinks("node-gpu", "node-cpu"),
    );

    expect(snapshot.capacityNodes).toEqual([{ id: "node-gpu", availableVramMiB: 3_500 }]);
    expect(snapshot.config).toBeNull();
  });

  it("accepts CPU capacity only when the desktop explicitly authorizes it", () => {
    const gpuWorker = worker("worker-gpu", "node-gpu", "node-gpu.relay", 9_850, 4_096, 3_500, 10);
    const cpuWorker = worker("worker-cpu", "node-cpu", "node-cpu.relay", 9_850, 4_096, 3_500, 10);
    Object.assign(cpuWorker.capabilities.gpus[0]!, {
      id: "cpu-memory",
      vendor: "cpu",
      model: "Intel CPU · system-memory",
      physicalVramMb: 0,
      sharedMemoryMb: 4_096,
    });
    Object.assign(cpuWorker.capabilities.distributedExecutor!, {
      computeMode: "cpu-only",
      cpuEligible: true,
      performanceEvidence: performanceEvidence(
        cpuWorker.id,
        "node-cpu",
        performanceProfile("cpu", "Intel CPU"),
      ),
    });

    const snapshot = buildConnectedExecutorActivationSnapshot(
      baseConfig(),
      [gpuWorker, cpuWorker],
      new Set([gpuWorker.id, cpuWorker.id]),
      measuredLinks("node-gpu", "node-cpu"),
    );

    expect(snapshot.capacityNodes.map((node) => node.id)).toEqual(["node-gpu", "node-cpu"]);
    expect(snapshot.config?.nodes.map((node) => node.id)).toEqual(["node-gpu", "node-cpu"]);
  });

  it("withholds a launch topology until the current transport has measured both directions", () => {
    const workers = [
      worker("worker-a", "node-a", "node-a.relay", 9_850, 4_096, 3_500, 10),
      worker("worker-b", "node-b", "node-b.relay", 9_850, 4_096, 3_500, 10),
    ];
    const withoutEvidence = buildConnectedExecutorActivationSnapshot(
      baseConfig(),
      workers,
      new Set(workers.map((entry) => entry.id)),
    );
    expect(withoutEvidence.capacityNodes).toHaveLength(2);
    expect(withoutEvidence.config).toBeNull();

    const oneDirection = buildConnectedExecutorActivationSnapshot(
      baseConfig(),
      workers,
      new Set(workers.map((entry) => entry.id)),
      measuredLinks("node-a", "node-b").slice(0, 1),
    );
    expect(oneDirection.config).toBeNull();
  });

  it("keeps capacity visible but rejects stale or low-confidence node profiles", () => {
    const first = worker("worker-a", "node-a", "node-a.relay", 9_850, 4_096, 3_500, 10);
    const second = worker("worker-b", "node-b", "node-b.relay", 9_850, 4_096, 3_500, 10);
    second.capabilities.distributedExecutor!.performanceEvidence =
      performanceEvidence(second.id, "node-b", sealRuntimePerformanceProfile({
        ...profileInput("cuda", "NVIDIA Test GPU"),
        measuredAt: "2020-01-01T00:00:00.000Z",
      }));
    const stale = buildConnectedExecutorActivationSnapshot(
      baseConfig(),
      [first, second],
      new Set([first.id, second.id]),
      measuredLinks("node-a", "node-b"),
    );
    expect(stale.capacityNodes).toHaveLength(2);
    expect(stale.config).toBeNull();

    second.capabilities.distributedExecutor!.performanceEvidence =
      performanceEvidence(second.id, "node-b", sealRuntimePerformanceProfile({
        ...profileInput("cuda", "NVIDIA Test GPU"),
        activationCodec: {
          ...series("GB/s", 1.8, 2, 2.1),
          confidenceHalfWidthPct: 25,
        },
      }));
    const noisy = buildConnectedExecutorActivationSnapshot(
      baseConfig(),
      [first, second],
      new Set([first.id, second.id]),
      measuredLinks("node-a", "node-b"),
    );
    expect(noisy.config).toBeNull();
  });

  it("does not optimize a node without a matching physical profile", () => {
    const first = worker("worker-a", "node-a", "node-a.relay", 9_850, 4_096, 3_500, 10);
    const second = worker("worker-b", "node-b", "node-b.relay", 9_850, 4_096, 3_500, 10);
    delete second.capabilities.distributedExecutor!.performanceEvidence;
    const missing = buildConnectedExecutorActivationSnapshot(
      baseConfig(),
      [first, second],
      new Set([first.id, second.id]),
      measuredLinks("node-a", "node-b"),
    );
    expect(missing.capacityNodes).toHaveLength(2);
    expect(missing.config).toBeNull();

    second.capabilities.distributedExecutor!.performanceEvidence =
      performanceEvidence(
        second.id,
        "node-b",
        performanceProfile("cuda", "A different physical GPU"),
      );
    const mismatched = buildConnectedExecutorActivationSnapshot(
      baseConfig(),
      [first, second],
      new Set([first.id, second.id]),
      measuredLinks("node-a", "node-b"),
    );
    expect(mismatched.config).toBeNull();
  });

  it("never schedules CPU capacity from a GPU-only desktop", () => {
    const cpuWorker = worker("worker-cpu", "node-cpu", "node-cpu.relay", 9_850, 4_096, 3_500, 10);
    cpuWorker.capabilities.gpus[0]!.vendor = "cpu";
    Object.assign(cpuWorker.capabilities.distributedExecutor!, {
      computeMode: "gpu-only",
      cpuEligible: true,
    });

    const snapshot = buildConnectedExecutorActivationSnapshot(
      baseConfig(),
      [cpuWorker],
      new Set([cpuWorker.id]),
    );

    expect(snapshot.capacityNodes).toEqual([]);
    expect(snapshot.config).toBeNull();
  });

  it("does not mix legacy direct-LAN executors into a relay topology", () => {
    const relayWorker = worker("worker-relay", "node-relay", "node-relay.relay", 9_850, 4_096, 3_500, 10);
    const legacyWorker = worker("worker-legacy", "node-legacy", "192.168.1.20", 9_850, 4_096, 3_500, 10);
    legacyWorker.capabilities.distributedExecutor!.protocol = "gdlp-worker-tunnel/1";
    const snapshot = buildConnectedExecutorActivationSnapshot(
      baseConfig(),
      [relayWorker, legacyWorker],
      new Set([relayWorker.id, legacyWorker.id]),
    );
    expect(snapshot.capacityNodes).toEqual([{ id: "node-relay", availableVramMiB: 3_500 }]);
    expect(snapshot.config).toBeNull();
  });

  // Los tokens por segundo dependen del MODELO. Tomar el máximo sobre todos los
  // despliegues del worker comparaba medidas incomparables: el nodo que aún
  // conservaba la medida de un modelo pequeño parecía el más rápido y se llevaba
  // más capas del modelo grande. El sesgo premiaba justo al que midió con la
  // carga más ligera.
  it("scales decode only by throughput measured for the model being planned", () => {
    const slowOnThisModel = worker("worker-a", "node-a", "192.168.1.10", 9_850, 4_096, 3_500, 12);
    const fastOnThisModel = worker("worker-b", "node-b", "192.168.1.11", 9_851, 6_144, 5_000, 20);
    // node-a es LENTO en "base" (10 tok/s) pero conserva una medida antigua de un
    // modelo diminuto a 1.000 tok/s. Esa segunda medida no debe contar aquí.
    slowOnThisModel.capabilities.deployments = [
      deployment("dep-a-base", "base", 10),
      deployment("dep-a-tiny", "modelo-diminuto-de-otra-activacion", 1_000),
    ];
    fastOnThisModel.capabilities.deployments = [deployment("dep-b-base", "base", 20)];

    const snapshot = buildConnectedExecutorActivationSnapshot(
      baseConfig(),
      [slowOnThisModel, fastOnThisModel],
      new Set(["worker-a", "worker-b"]),
      measuredLinks("node-a", "node-b"),
    );

    // `decodeScale` es un multiplicador de COSTE normalizado al más rápido, así
    // que el rápido vale 1 y el que va a la mitad vale 2. Con el máximo sobre
    // todos los despliegues salía al revés: node-a 1 y node-b 20 (tope).
    const scaleById = new Map(
      (snapshot.config?.nodes ?? []).map((node) => [node.id, node.decodeScale]),
    );
    expect(scaleById.get("node-a")).toBeCloseTo(2, 5);
    expect(scaleById.get("node-b")).toBeCloseTo(1, 5);
  });

  // Sin ninguna medida DEL MODELO planificado, el reparto vuelve a declararse no
  // medido en vez de colarse por informado usando la medida de otro modelo.
  it("falls back to unmeasured when only other models were measured", () => {
    const workerA = worker("worker-a", "node-a", "192.168.1.10", 9_850, 4_096, 3_500, 12);
    const workerB = worker("worker-b", "node-b", "192.168.1.11", 9_851, 6_144, 5_000, 20);
    workerA.capabilities.deployments = [deployment("dep-a-otro", "otro-modelo", 10)];
    workerB.capabilities.deployments = [deployment("dep-b-otro", "otro-modelo", 20)];

    const snapshot = buildConnectedExecutorActivationSnapshot(
      baseConfig(),
      [workerA, workerB],
      new Set(["worker-a", "worker-b"]),
      measuredLinks("node-a", "node-b"),
    );

    // Cae al perfil de rendimiento, que es el mismo 0,5 del primer test.
    expect(snapshot.config?.nodes.every((node) => node.decodeScale === 0.5)).toBe(true);
  });
});

function deployment(deploymentId: string, model: string, tokensPerSecond: number) {
  return {
    deploymentId,
    model,
    modelDigest: `digest-${deploymentId}`,
    mode: "replica" as const,
    adapter: "mycellios-pipeline" as const,
    peakVramMb: 2_600,
    contextLimit: 4_096,
    maxConcurrency: 1,
    freeSlots: 1,
    tokensPerSecond,
    throughputSource: "measured" as const,
    ttftMs: 800,
    dataLocality: "local" as const,
  };
}

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
        vendor: "nvidia",
        model: "NVIDIA Test GPU",
        physicalVramMb: offeredVramMb,
        offeredVramMb,
        freeOfferedVramMb,
        powerW: 80,
      }],
      limits: { maxConcurrency: 1, pauseWhenForeground: false },
      deployments: [],
      network: { coordinatorRttMs, uplinkMbps: 100, downlinkMbps: 100 },
      distributedExecutor: {
        protocol: "gdlp-worker-tunnel/2",
        nodeId,
        stageHost,
        stagePort,
        runtime: "python-safetensors",
        computeMode: "automatic",
        cpuEligible: false,
        acceleration: {
          schema: "mycellios-accelerator-diagnostics/1",
          appVersion: "0.2.99",
          state: "gpu-ready",
          backend: "cuda",
          deviceName: "NVIDIA Test GPU",
          gpuVendor: "nvidia",
          gpuModel: "NVIDIA Test GPU",
          phase: "ready",
          progressPct: 100,
          issueCode: null,
          issueSummary: null,
          retryable: false,
          retryAttempt: 0,
          nextRetryAt: null,
          updatedAt: new Date().toISOString(),
          recentEvents: [],
        },
        performanceEvidence: performanceEvidence(
          id,
          nodeId,
          performanceProfile("cuda", "NVIDIA Test GPU"),
        ),
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

function performanceProfile(
  backend: "cuda" | "cpu",
  deviceName: string,
) {
  return sealRuntimePerformanceProfile(profileInput(backend, deviceName));
}

function performanceEvidence(
  workerId: string,
  nodeId: string,
  profile: RuntimePerformanceProfile,
) {
  const measuredAt = Date.parse(profile.measuredAt);
  return createCoordinatorRuntimePerformanceEvidence({
    challengeId: `challenge-${workerId}`,
    nonce: Buffer.alloc(32, workerId.length).toString("base64url"),
    workerId,
    sessionId: `session-${workerId}`,
    nodeId,
    issuedAt: new Date(measuredAt - 1_000).toISOString(),
    expiresAt: new Date(measuredAt + 60_000).toISOString(),
    observedAt: new Date(measuredAt + 1_000).toISOString(),
    profile,
  });
}

function profileInput(
  backend: "cuda" | "cpu",
  deviceName: string,
) {
  return {
    measuredAt: new Date().toISOString(),
    backend,
    deviceName,
    precision: backend === "cpu" ? "float32" as const : "float16" as const,
    source: "physical-microbenchmark" as const,
    activationCodecId: "fp16" as const,
    decodeMemory: series("GB/s", 350, 400, 430),
    prefillCompute: series("TFLOP/s", 18, 20, 21),
    activationCodec: series("GB/s", 1.8, 2, 2.1),
  };
}

function series(
  unit: "GB/s" | "TFLOP/s",
  p5: number,
  p50: number,
  p95: number,
) {
  return {
    unit,
    warmupSamples: 2,
    samples: 7,
    p5,
    p50,
    p95,
    confidenceHalfWidthPct: 5,
  };
}

function measuredLinks(fromNodeId: string, toNodeId: string) {
  const measuredAt = Date.now();
  return [
    {
      fromNodeId,
      toNodeId,
      measuredAt,
      rttP50Ms: 20,
      rttP95Ms: 24,
      goodputMbpsP50: 80,
      successfulSamples: 5,
      failedSamples: 0,
      availability: 1,
    },
    {
      fromNodeId: toNodeId,
      toNodeId: fromNodeId,
      measuredAt,
      rttP50Ms: 22,
      rttP95Ms: 28,
      goodputMbpsP50: 75,
      successfulSamples: 5,
      failedSamples: 0,
      availability: 1,
    },
  ];
}
