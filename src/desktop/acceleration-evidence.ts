import type {
  DashboardWorker,
  DesktopAccelerationStatus,
} from "./contracts.js";

export interface VerifiedAccelerationUsage {
  cpuStages: number;
  gpuStages: number;
}

/**
 * Count only execution telemetry published after a runtime loaded its model
 * and passed the distributed canary. Detected hardware and spawned processes
 * are deliberately not considered proof of active compute.
 */
export function readVerifiedAccelerationUsage(input: {
  workers: DashboardWorker[];
  runtimeNodeId: string | null;
  localWorkerId: string | null;
  contributionConnected: boolean;
}): VerifiedAccelerationUsage {
  if (!input.contributionConnected || !input.runtimeNodeId) {
    return { cpuStages: 0, gpuStages: 0 };
  }

  let cpuStages = 0;
  let gpuStages = 0;
  for (const worker of input.workers) {
    if (!worker.connected || worker.status === "offline") continue;
    for (const deployment of worker.deployments) {
      const execution = deployment.execution;
      if (!execution) continue;
      const stages = execution.stages ?? [];
      if (stages.length > 0) {
        for (const stage of stages) {
          if (stage.nodeId !== input.runtimeNodeId) continue;
          if (stage.deviceType === "gpu") gpuStages += 1;
          else cpuStages += 1;
        }
        continue;
      }

      // A direct (non-pipeline) runtime has no per-stage list. It is local
      // evidence only when it belongs to this exact desktop worker.
      if (worker.id !== input.localWorkerId) continue;
      if (execution.deviceType === "cpu" || execution.deviceType === "mixed") cpuStages += 1;
      if (execution.deviceType === "gpu" || execution.deviceType === "mixed") gpuStages += 1;
    }
  }
  return { cpuStages, gpuStages };
}

export function applyVerifiedAccelerationUsage(
  current: DesktopAccelerationStatus,
  usage: VerifiedAccelerationUsage,
): DesktopAccelerationStatus {
  return {
    ...current,
    cpu: {
      ...current.cpu,
      activeStages: usage.cpuStages,
      state: current.cpu.state === "unavailable"
        ? "unavailable"
        : usage.cpuStages > 0
          ? "active"
          : "ready",
      message: usage.cpuStages > 0
        ? `${usage.cpuStages} canary-verified model stage${usage.cpuStages === 1 ? " is" : "s are"} active on CPU.`
        : current.cpu.message,
    },
    gpu: {
      ...current.gpu,
      activeStages: usage.gpuStages,
    },
  };
}

const GPU_RETRY_DELAYS_MS = [30_000, 120_000, 300_000] as const;

export function gpuPreparationRetryDelayMs(attempt: number): number {
  const safeAttempt = Number.isFinite(attempt) ? Math.max(0, Math.floor(attempt)) : 0;
  return GPU_RETRY_DELAYS_MS[Math.min(safeAttempt, GPU_RETRY_DELAYS_MS.length - 1)]!;
}

/** Null means a cheap/recoverable condition can keep retrying with backoff. */
export function gpuPreparationAutomaticRetryLimit(issueCode: string): number | null {
  if (issueCode === "network" || issueCode === "disk-space") return null;
  if (issueCode === "integrity") return 1;
  return 2;
}
