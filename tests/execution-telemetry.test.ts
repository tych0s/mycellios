import { describe, expect, it } from "vitest";
import { createCoordinatorDeploymentCanaryEvidence } from "../src/contracts/deployment-canary.js";
import { deploymentSchema } from "../src/contracts/schemas.js";

const MODEL_DIGEST = `sha256:${"f".repeat(64)}`;
const ACTIVATION_ID = "execution-telemetry-activation";
const EVIDENCE_NOW = Date.now();
const CANARY_EVIDENCE = createCoordinatorDeploymentCanaryEvidence({
  challengeId: "challenge-execution-telemetry",
  nonce: Buffer.alloc(32, 3).toString("base64url"),
  workerId: "worker-execution-telemetry",
  sessionId: "session-execution-telemetry",
  issuedAt: new Date(EVIDENCE_NOW - 1_000).toISOString(),
  expiresAt: new Date(EVIDENCE_NOW + 60_000).toISOString(),
  model: "qwen3-0.6b",
  modelDigest: MODEL_DIGEST,
  activationId: ACTIVATION_ID,
  promptDigest: `sha256:${"e".repeat(64)}`,
  maxOutputTokens: 9,
  observedAt: new Date(EVIDENCE_NOW).toISOString(),
  warmupSamples: 1,
  samples: [0, 1, 2].map((index) => ({
    sampleId: `sample-${index}`,
    outputTokens: 9,
    activeMs: 2_000,
    ttftMs: 800,
    completed: true as const,
  })),
});

const baseDeployment = {
  deploymentId: "dep-qwen",
  model: "qwen3-0.6b",
  modelDigest: MODEL_DIGEST,
  activationId: ACTIVATION_ID,
  mode: "replica" as const,
  adapter: "mycellios-pipeline" as const,
  peakVramMb: 2_600,
  contextLimit: 4_096,
  maxConcurrency: 1,
  freeSlots: 1,
  tokensPerSecond: 4.5,
  throughputSource: "measured" as const,
  ttftMs: 800,
  verificationState: "verified" as const,
  canaryEvidence: CANARY_EVIDENCE,
  dataLocality: "local" as const,
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
