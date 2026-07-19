import type {
  ExecutionMode,
  WorkerRegistration,
} from "../src/contracts/types.js";
import type { MeshStore, StoredWorker } from "../src/storage/store.js";

export function addWorker(
  store: MeshStore,
  input: {
    id: string;
    model?: string;
    region?: string;
    offeredVramMb?: number;
    peakVramMb?: number;
    tokensPerSecond?: number;
    ttftMs?: number;
    contextLimit?: number;
    modelDigest?: string;
    mode?: ExecutionMode;
    stage?: { index: number; total: number; layerStart: number; layerEnd: number };
    llmfit?: {
      fitLevel: string;
      bestQuant?: string;
      estimatedTokensPerSecond?: number;
      measuredTokensPerSecond?: number;
      memoryRequiredMb?: number;
    };
  },
): StoredWorker {
  const mode = input.mode ?? "replica";
  const registration: WorkerRegistration = {
    capabilities: {
      region: input.region ?? "es-mad",
      agentVersion: "test",
      gpus: [
        {
          id: "gpu-0",
          vendor: "test",
          model: "Synthetic GPU",
          physicalVramMb: input.offeredVramMb ?? 8_192,
          offeredVramMb: input.offeredVramMb ?? 8_192,
          freeOfferedVramMb: input.offeredVramMb ?? 8_192,
        },
      ],
      limits: { maxConcurrency: 2, pauseWhenForeground: true },
      deployments: [
        {
          deploymentId: `dep-${input.id}`,
          model: input.model ?? "distributed-small",
          modelDigest: input.modelDigest ?? `sha256:${input.id}`,
          mode,
          adapter: "mock",
          peakVramMb: input.peakVramMb ?? 3_000,
          contextLimit: input.contextLimit ?? 8_192,
          maxConcurrency: 2,
          freeSlots: 2,
          tokensPerSecond: input.tokensPerSecond ?? 10,
          ttftMs: input.ttftMs ?? 1_000,
          dataLocality: "local",
          ...(input.stage ? { stage: input.stage } : {}),
        },
      ],
      network: { coordinatorRttMs: 20, uplinkMbps: 100, downlinkMbps: 100 },
      ...(input.llmfit
        ? {
            llmfit: {
              source: "llmfit" as const,
              scope: "host" as const,
              backend: "test",
              cpuName: "Synthetic CPU",
              cpuCores: 8,
              totalRamMb: 16_384,
              availableRamMb: 8_192,
              gpuCount: 1,
              gpus: [
                {
                  name: "Synthetic GPU",
                  backend: "test",
                  vramMb: input.offeredVramMb ?? 8_192,
                  unifiedMemory: false,
                },
              ],
              model: {
                deploymentId: `dep-${input.id}`,
                requestedModel: input.model ?? "distributed-small",
                resolvedModel: input.model ?? "distributed-small",
                fitLevel: input.llmfit.fitLevel,
                runMode: "GPU",
                ...(input.llmfit.bestQuant
                  ? { bestQuant: input.llmfit.bestQuant }
                  : {}),
                ...(input.llmfit.estimatedTokensPerSecond
                  ? {
                      estimatedTokensPerSecond:
                        input.llmfit.estimatedTokensPerSecond,
                    }
                  : {}),
                ...(input.llmfit.measuredTokensPerSecond
                  ? { measuredTokensPerSecond: input.llmfit.measuredTokensPerSecond }
                  : {}),
                ...(input.llmfit.memoryRequiredMb !== undefined
                  ? { memoryRequiredMb: input.llmfit.memoryRequiredMb }
                  : {}),
              },
            },
          }
        : {}),
    },
  };
  const worker = store.registerWorker(registration);
  store.updateWorkerHeartbeat(worker.id, registration.capabilities, "online");
  return store.getWorker(worker.id)!;
}
