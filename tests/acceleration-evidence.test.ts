import { describe, expect, it } from "vitest";
import {
  applyVerifiedAccelerationUsage,
  gpuPreparationAutomaticRetryLimit,
  gpuPreparationRetryDelayMs,
  readVerifiedAccelerationUsage,
} from "../src/desktop/acceleration-evidence.js";
import { createInitialAccelerationStatus } from "../src/desktop/acceleration-progress.js";
import type { DashboardWorker } from "../src/desktop/contracts.js";

function worker(input: {
  id: string;
  executionNodeId?: string;
  execution?: DashboardWorker["deployments"][number]["execution"];
}): DashboardWorker {
  return {
    id: input.id,
    kind: "desktop",
    status: "online",
    connected: true,
    region: "test",
    offeredVramMb: 4_096,
    reliability: 1,
    jobsCompleted: 0,
    lastSeenAt: new Date(0).toISOString(),
    gpus: [],
    deployments: input.execution
      ? [{
          deploymentId: `deployment-${input.id}`,
          model: "model",
          modelDigest: "sha256:model",
          mode: "replica",
          adapter: "transformers-qwen3-v1",
          peakVramMb: 1_024,
          contextLimit: 2_048,
          freeSlots: 1,
          tokensPerSecond: 1,
          ttftMs: 1,
          execution: input.execution,
        }]
      : [],
    ...(input.executionNodeId ? { executionNodeId: input.executionNodeId } : {}),
  };
}

describe("verified desktop acceleration evidence", () => {
  it("does not claim active compute from detection or an unverified deployment", () => {
    const usage = readVerifiedAccelerationUsage({
      workers: [worker({ id: "desktop" })],
      runtimeNodeId: "runtime-node",
      localWorkerId: "desktop",
      contributionConnected: true,
    });
    expect(usage).toEqual({ cpuStages: 0, gpuStages: 0 });
  });

  it("counts only canary-published stages assigned to this runtime node", () => {
    const source = worker({
      id: "cell",
      execution: {
        deviceType: "mixed",
        backend: "cuda",
        deviceName: "mixed topology",
        precision: "float16",
        fallback: false,
        stages: [
          { nodeId: "runtime-node", stageIndex: 0, layerStart: 0, layerEnd: 8, deviceType: "cpu", backend: "cpu", deviceName: "CPU", precision: "float32", fallback: false },
          { nodeId: "runtime-node", stageIndex: 1, layerStart: 8, layerEnd: 16, deviceType: "gpu", backend: "cuda", deviceName: "RTX", precision: "float16", fallback: false },
          { nodeId: "someone-else", stageIndex: 2, layerStart: 16, layerEnd: 24, deviceType: "gpu", backend: "cuda", deviceName: "Other RTX", precision: "float16", fallback: false },
        ],
      },
    });
    expect(readVerifiedAccelerationUsage({
      workers: [source],
      runtimeNodeId: "runtime-node",
      localWorkerId: "desktop",
      contributionConnected: true,
    })).toEqual({ cpuStages: 1, gpuStages: 1 });
  });

  it("does not keep ACTIVE lit from stale execution telemetry on an offline worker", () => {
    const source = worker({
      id: "stale-cell",
      execution: {
        deviceType: "gpu",
        backend: "cuda",
        deviceName: "RTX",
        precision: "float16",
        fallback: false,
        stages: [
          { nodeId: "runtime-node", stageIndex: 0, layerStart: 0, layerEnd: 8, deviceType: "gpu", backend: "cuda", deviceName: "RTX", precision: "float16", fallback: false },
        ],
      },
    });
    source.connected = false;
    source.status = "offline";
    expect(readVerifiedAccelerationUsage({
      workers: [source],
      runtimeNodeId: "runtime-node",
      localWorkerId: "desktop",
      contributionConnected: true,
    })).toEqual({ cpuStages: 0, gpuStages: 0 });
  });

  it("clears active claims when contribution is no longer connected", () => {
    const current = createInitialAccelerationStatus();
    current.cpu.state = "ready";
    current.cpu.activeStages = 2;
    current.gpu.activeStages = 1;
    const usage = readVerifiedAccelerationUsage({
      workers: [], runtimeNodeId: "runtime-node", localWorkerId: "desktop", contributionConnected: false,
    });
    const next = applyVerifiedAccelerationUsage(current, usage);
    expect(next.cpu).toMatchObject({ state: "ready", activeStages: 0 });
    expect(next.gpu.activeStages).toBe(0);
  });

  it("uses bounded automatic retry backoff", () => {
    expect([0, 1, 2, 9].map(gpuPreparationRetryDelayMs)).toEqual([30_000, 120_000, 300_000, 300_000]);
  });

  it("limits expensive repeated failures while network failures keep resumable backoff", () => {
    expect(gpuPreparationAutomaticRetryLimit("integrity")).toBe(1);
    expect(gpuPreparationAutomaticRetryLimit("install")).toBe(2);
    expect(gpuPreparationAutomaticRetryLimit("physical-probe")).toBe(2);
    expect(gpuPreparationAutomaticRetryLimit("runtime-error")).toBe(2);
    expect(gpuPreparationAutomaticRetryLimit("network")).toBeNull();
    expect(gpuPreparationAutomaticRetryLimit("disk-space")).toBeNull();
  });
});
