import type {
  ExecutionMode,
  ModelDeployment,
  WorkerCapabilities,
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
    maxConcurrency?: number;
    contextLimit?: number;
    modelDigest?: string;
    mode?: ExecutionMode;
    stage?: { index: number; total: number; layerStart: number; layerEnd: number };
    internalPipeline?: { stageCount: number; boundaries: number[] };
    execution?: ModelDeployment["execution"];
    distributedExecutor?: WorkerCapabilities["distributedExecutor"];
    identity?: WorkerRegistration["identity"];
  },
): StoredWorker {
  const mode = input.mode ?? "replica";
  const registration: WorkerRegistration = {
    ...(input.identity ? { identity: input.identity } : {}),
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
      limits: {
        maxConcurrency: input.maxConcurrency ?? 2,
        pauseWhenForeground: true,
      },
      deployments: [
        {
          deploymentId: `dep-${input.id}`,
          model: input.model ?? "distributed-small",
          modelDigest: input.modelDigest ?? `sha256:${input.id}`,
          mode,
          adapter: "mock",
          peakVramMb: input.peakVramMb ?? 3_000,
          contextLimit: input.contextLimit ?? 8_192,
          maxConcurrency: input.maxConcurrency ?? 2,
          freeSlots: input.maxConcurrency ?? 2,
          tokensPerSecond: input.tokensPerSecond ?? 10,
          ttftMs: input.ttftMs ?? 1_000,
          dataLocality: "local",
          ...(input.stage ? { stage: input.stage } : {}),
          ...(input.internalPipeline ? { internalPipeline: input.internalPipeline } : {}),
          ...(input.execution ? { execution: input.execution } : {}),
        },
      ],
      network: { coordinatorRttMs: 20, uplinkMbps: 100, downlinkMbps: 100 },
      ...(input.distributedExecutor ? { distributedExecutor: input.distributedExecutor } : {}),
    },
  };
  const worker = store.registerWorker(registration);
  store.updateWorkerHeartbeat(worker.id, registration.capabilities, "online");
  return store.getWorker(worker.id)!;
}
