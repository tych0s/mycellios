import { describe, expect, it } from "vitest";
import {
  modelHasGpuFallback,
  verifiedGpuCapacityCanRepairModel,
} from "../src/coordinator/connected-executor-activation.js";
import type { StoredRequestedModel, StoredWorker } from "../src/storage/store.js";

describe("automatic GPU repair", () => {
  it("detects a published mixed route from verified fallback telemetry", () => {
    const cell = worker("cell", "cell-node", "cpu", 0, 0);
    cell.capabilities.deployments = [{
      deploymentId: "model-pipeline",
      model: "model",
      modelDigest: "sha256:test",
      mode: "replica",
      adapter: "openai-compatible",
      peakVramMb: 512,
      contextLimit: 4_096,
      maxConcurrency: 1,
      freeSlots: 1,
      tokensPerSecond: 1,
      ttftMs: 1,
      dataLocality: "local",
      execution: {
        deviceType: "mixed",
        backend: "rocm",
        deviceName: "GPU + CPU",
        precision: "mixed",
        fallback: true,
        stages: [{
          nodeId: "node-a",
          stageIndex: 0,
          layerStart: 0,
          layerEnd: 1,
          deviceType: "cpu",
          backend: "cpu",
          deviceName: "CPU",
          precision: "float32",
          fallback: true,
          fallbackReason: "accelerator probe failed",
        }],
      },
    }];
    expect(modelHasGpuFallback("model", [cell], new Set([cell.id]))).toBe(true);
    expect(modelHasGpuFallback("other", [cell], new Set([cell.id]))).toBe(false);
  });

  it("waits for enough re-verified physical GPU capacity", () => {
    const model = requestedModel();
    const first = worker("worker-a", "node-a", "amd", 4_096, 3_000);
    const second = worker("worker-b", "node-b", "nvidia", 4_096, 3_000);
    const connected = new Set([first.id, second.id]);
    expect(verifiedGpuCapacityCanRepairModel(model, [first, second], connected)).toBe(true);

    second.capabilities.gpus[0]!.vendor = "cpu";
    second.capabilities.gpus[0]!.physicalVramMb = 0;
    expect(verifiedGpuCapacityCanRepairModel(model, [first, second], connected)).toBe(false);

    second.capabilities.gpus[0]!.vendor = "nvidia";
    second.capabilities.gpus[0]!.physicalVramMb = 4_096;
    second.capabilities.distributedExecutor!.computeMode = "cpu-only";
    expect(verifiedGpuCapacityCanRepairModel(model, [first, second], connected)).toBe(false);
  });
});

function requestedModel(): StoredRequestedModel {
  return {
    id: "model",
    source: "org/model",
    revision: null,
    contextTokens: 4_096,
    minimumNodes: 2,
    autoActivate: true,
    profile: { minimumStageVramMiB: 1_000, requiredVramMiB: 5_000 },
    profileError: null,
    activationRequestedAt: Date.now(),
    activationError: null,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

function worker(
  id: string,
  nodeId: string,
  vendor: string,
  physicalVramMb: number,
  freeOfferedVramMb: number,
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
      agentVersion: "test",
      gpus: [{
        id: `${id}-gpu`,
        vendor,
        model: vendor === "cpu" ? "CPU" : "GPU",
        physicalVramMb,
        offeredVramMb: 4_096,
        freeOfferedVramMb,
      }],
      limits: { maxConcurrency: 1, pauseWhenForeground: false },
      deployments: [],
      network: { coordinatorRttMs: 1, uplinkMbps: 100, downlinkMbps: 100 },
      distributedExecutor: {
        protocol: "gdlp-worker-tunnel/2",
        nodeId,
        stageHost: `${nodeId}.relay`,
        stagePort: 9_850,
        runtime: "python-safetensors",
        computeMode: "automatic",
        cpuEligible: false,
      },
    },
  };
}
