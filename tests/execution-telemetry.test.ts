import { describe, expect, it } from "vitest";
import { deploymentSchema } from "../src/contracts/schemas.js";

const baseDeployment = {
  deploymentId: "dep-qwen",
  model: "qwen3-0.6b",
  modelDigest: "sha256:qwen",
  mode: "replica" as const,
  adapter: "openai-compatible" as const,
  peakVramMb: 2_600,
  contextLimit: 4_096,
  maxConcurrency: 1,
  freeSlots: 1,
  tokensPerSecond: 4.5,
  ttftMs: 800,
  dataLocality: "external" as const,
};

describe("effective execution telemetry", () => {
  it("accepts a mixed pipeline only when every effective stage device is reported", () => {
    const result = deploymentSchema.safeParse({
      ...baseDeployment,
      execution: {
        deviceType: "mixed",
        backend: "cuda",
        deviceName: "2-node distributed pipeline",
        precision: "float32",
        fallback: true,
        fallbackReason: "ROCm is unavailable on one stage",
        stages: [
          {
            nodeId: "desktop-amd",
            stageIndex: 0,
            layerStart: 0,
            layerEnd: 14,
            deviceType: "cpu",
            backend: "cpu",
            deviceName: "AMD Ryzen AI 9 HX 370",
            precision: "float32",
            fallback: true,
            fallbackReason: "ROCm is unavailable",
          },
          {
            nodeId: "desktop-nvidia",
            stageIndex: 1,
            layerStart: 14,
            layerEnd: 28,
            deviceType: "gpu",
            backend: "cuda",
            deviceName: "NVIDIA GeForce RTX 2060",
            precision: "float16",
            fallback: false,
          },
        ],
      },
    });

    expect(result.success).toBe(true);
  });

  it("rejects inferred or contradictory device claims", () => {
    const contradictoryStage = {
      ...baseDeployment,
      execution: {
        deviceType: "gpu",
        backend: "cuda",
        deviceName: "Claimed GPU",
        precision: "float16",
        fallback: false,
        stages: [{
          nodeId: "desktop-test",
          stageIndex: 0,
          layerStart: 0,
          layerEnd: 28,
          deviceType: "cpu",
          backend: "cpu",
          deviceName: "Actual CPU",
          precision: "float32",
          fallback: true,
          fallbackReason: "CUDA unavailable",
        }],
      },
    };
    const invalidBackend = {
      ...baseDeployment,
      execution: {
        deviceType: "cpu",
        backend: "cuda",
        deviceName: "CPU",
        precision: "float32",
        fallback: false,
      },
    };

    expect(deploymentSchema.safeParse(contradictoryStage).success).toBe(false);
    expect(deploymentSchema.safeParse(invalidBackend).success).toBe(false);
  });
});
