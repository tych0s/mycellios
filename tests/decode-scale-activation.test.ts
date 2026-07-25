import { describe, expect, it } from "vitest";
import { buildConnectedExecutorActivationSnapshot } from "../src/coordinator/connected-executor-activation.js";
import {
  AUTO_DISTRIBUTE_SCHEMA,
  parseAutoDistributionConfig,
} from "../src/distribution/auto-distribute.js";
import type { StoredWorker } from "../src/storage/store.js";

/**
 * PRUEBA DE ACTIVACIÓN de `decodeScale`.
 *
 * La suite entera pasaba con `decodeScale: 1` cableado y sigue pasando sin él,
 * porque los fixtures existentes traen `deployments: []` y nunca ejercitan la
 * rama. Sin este fichero, el segundo cero podría "arreglarse" sin que ningún
 * test notara la diferencia — que es exactamente lo que pasó con el cero del
 * RTT hasta que se buscó a propósito el escenario discriminante.
 */
describe("activación de decodeScale", () => {
  it("un nodo el doble de rápido recibe la mitad de escala de decode", () => {
    const snapshot = buildConnectedExecutorActivationSnapshot(
      baseConfig(),
      [
        workerWithThroughput("w-fast", "node-fast", 9_850, 100, "measured"),
        workerWithThroughput("w-slow", "node-slow", 9_851, 25, "measured"),
      ],
      new Set(["w-fast", "w-slow"]),
    );

    const scales = new Map(
      (snapshot.config?.nodes ?? []).map((node) => [node.id, node.decodeScale]),
    );
    // La aserción que falla con el código viejo: allí ambos valían 1.
    expect(scales.get("node-fast")).toBeCloseTo(1);
    expect(scales.get("node-slow")).toBeCloseTo(4);
    expect(scales.get("node-fast")).not.toBe(scales.get("node-slow"));
  });

  it("throughput por defecto NO mueve la escala: no es telemetría", () => {
    const snapshot = buildConnectedExecutorActivationSnapshot(
      baseConfig(),
      [
        workerWithThroughput("w-a", "node-a", 9_850, 100, "default"),
        workerWithThroughput("w-b", "node-b", 9_851, 25, "default"),
      ],
      new Set(["w-a", "w-b"]),
    );
    for (const node of snapshot.config?.nodes ?? []) {
      expect(node.decodeScale).toBe(1);
    }
  });

  it("sin deployments se comporta como antes del cambio", () => {
    // Degradación segura: los fixtures y despliegues existentes no cambian.
    const snapshot = buildConnectedExecutorActivationSnapshot(
      baseConfig(),
      [
        workerWithThroughput("w-a", "node-a", 9_850, 0, undefined, true),
        workerWithThroughput("w-b", "node-b", 9_851, 0, undefined, true),
      ],
      new Set(["w-a", "w-b"]),
    );
    for (const node of snapshot.config?.nodes ?? []) {
      expect(node.decodeScale).toBe(1);
    }
  });
});

function workerWithThroughput(
  id: string,
  nodeId: string,
  stagePort: number,
  tokensPerSecond: number,
  throughputSource: "measured" | "default" | undefined,
  omitDeployment = false,
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
        physicalVramMb: 8_192,
        offeredVramMb: 8_192,
        freeOfferedVramMb: 8_000,
        powerW: 80,
      }],
      limits: { maxConcurrency: 1, pauseWhenForeground: false },
      deployments: omitDeployment
        ? []
        : [{
            deploymentId: `${id}-dep`,
            model: "test-model",
            adapter: "transformers",
            peakVramMb: 4_096,
            contextLimit: 4_096,
            maxConcurrency: 1,
            freeSlots: 1,
            tokensPerSecond,
            ...(throughputSource ? { throughputSource } : {}),
            ttftMs: 100,
            dataLocality: "local",
          }],
      network: { coordinatorRttMs: 10, uplinkMbps: 100, downlinkMbps: 100 },
      distributedExecutor: {
        protocol: "gdlp-worker-tunnel/2",
        nodeId,
        stageHost: "127.0.0.1",
        stagePort,
        runtime: "python-safetensors",
      },
    },
  } as unknown as StoredWorker;
}

function baseConfig() {
  return parseAutoDistributionConfig({
    schema: AUTO_DISTRIBUTE_SCHEMA,
    model: { source: "local-model", revision: null, publicName: "scale-test" },
    nodes: [
      {
        id: "seed-a",
        region: "test",
        endpoint: { host: "127.0.0.1", port: 21_001 },
        memoryMiB: 8_192,
        reserveMiB: 256,
        decodeScale: 1,
        prefillScale: 1,
        codecScale: 1,
        powerWatts: 65,
        availability: 0.999,
        agent: { kind: "local" },
      },
      {
        id: "seed-b",
        region: "test",
        endpoint: { host: "127.0.0.1", port: 21_002 },
        memoryMiB: 8_192,
        reserveMiB: 256,
        decodeScale: 1,
        prefillScale: 1,
        codecScale: 1,
        powerWatts: 65,
        availability: 0.999,
        agent: { kind: "local" },
      },
    ],
    links: [],
    distribution: { minimumStages: 2, maximumStages: 2, allowLossyActivation: false },
    workload: {
      promptTokens: 32,
      outputTokens: 16,
      contextTokens: 128,
      concurrentSequences: 1,
      minRouteAvailability: 0.9,
      batchWindowMs: 1,
      p95: false,
    },
    runtime: {
      pythonExecutable: "runtime/distribution-venv/Scripts/python.exe",
      pythonPath: "python",
      hfHome: "runtime/hf-cache",
      apiEndpoint: { host: "127.0.0.1", port: 8_088 },
      apiAdvertiseHost: "127.0.0.1",
      returnEndpoint: { host: "127.0.0.1", port: 30_000 },
      returnBindHost: "127.0.0.1",
      threadsPerStage: 1,
      connectTimeoutSeconds: 60,
    },
  });
}
