import { describe, expect, it } from "vitest";
import {
  deploymentCanaryEvidenceSchema,
  deploymentMetricsFromCanaryEvidence,
  sealDeploymentCanaryEvidence,
} from "../src/contracts/deployment-canary.js";

const MODEL_DIGEST = `sha256:${"d".repeat(64)}`;
const ACTIVATION_ID = "activation-17";

describe("sealed deployment canary evidence", () => {
  it("derives throughput and median TTFT from completed physical samples", () => {
    const evidence = fixture();

    expect(deploymentCanaryEvidenceSchema.parse(evidence)).toEqual(evidence);
    expect(deploymentMetricsFromCanaryEvidence(evidence, {
      model: "qwen",
      modelDigest: MODEL_DIGEST,
      activationId: ACTIVATION_ID,
      now: Date.parse(evidence.measuredAt) + 1_000,
    })).toMatchObject({
      tokensPerSecond: 20,
      ttftMs: 300,
      evidenceId: evidence.evidenceId,
      activationId: ACTIVATION_ID,
    });
  });

  it("rejects metric mutation and any model or activation mismatch", () => {
    const evidence = fixture();
    expect(() => deploymentCanaryEvidenceSchema.parse({
      ...evidence,
      samples: evidence.samples.map((sample, index) =>
        index === 0 ? { ...sample, activeMs: 1 } : sample
      ),
    })).toThrow(/deployment_canary_seal_is_invalid/);
    expect(() => deploymentMetricsFromCanaryEvidence(evidence, {
      model: "other",
      modelDigest: MODEL_DIGEST,
      activationId: ACTIVATION_ID,
      now: Date.parse(evidence.measuredAt),
    })).toThrow("deployment_canary_model_mismatch");
    expect(() => deploymentMetricsFromCanaryEvidence(evidence, {
      model: "qwen",
      modelDigest: MODEL_DIGEST,
      activationId: "other-activation",
      now: Date.parse(evidence.measuredAt),
    })).toThrow("deployment_canary_activation_mismatch");
  });
});

function fixture() {
  return sealDeploymentCanaryEvidence({
    model: "qwen",
    modelDigest: MODEL_DIGEST,
    activationId: ACTIVATION_ID,
    promptDigest: `sha256:${"e".repeat(64)}`,
    maxOutputTokens: 20,
    measuredAt: "2026-07-25T16:00:00.000Z",
    warmupSamples: 2,
    samples: [
      {
        sampleId: "sample-a",
        outputTokens: 20,
        activeMs: 1_000,
        ttftMs: 300,
        completed: true,
      },
      {
        sampleId: "sample-b",
        outputTokens: 20,
        activeMs: 1_000,
        ttftMs: 280,
        completed: true,
      },
      {
        sampleId: "sample-c",
        outputTokens: 20,
        activeMs: 1_000,
        ttftMs: 320,
        completed: true,
      },
    ],
  });
}
