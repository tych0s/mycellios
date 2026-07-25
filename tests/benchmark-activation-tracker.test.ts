import { describe, expect, it } from "vitest";
import {
  sealBenchmarkActivation,
  StableBenchmarkActivationTracker,
  type BenchmarkActivationParticipant,
} from "../src/benchlab/activation-tracker.js";

describe("stable benchmark activation tracker", () => {
  it("waits for repeated identical READY observations", () => {
    const activation = sealBenchmarkActivation("qwen", [participant()]);
    const tracker = new StableBenchmarkActivationTracker(3, 6_000);

    expect(tracker.observe([activation], 1_000)).toEqual([]);
    expect(tracker.observe([activation], 6_000)).toEqual([]);
    expect(tracker.observe([activation], 11_000)).toEqual([activation]);
    expect(tracker.observe([activation], 16_000)).toEqual([]);
  });

  it("triggers the same model again when deployment generation or topology changes", () => {
    const first = sealBenchmarkActivation("qwen", [participant()]);
    const replacement = sealBenchmarkActivation("qwen", [{
      ...participant(),
      deploymentId: "deployment-2",
      stageRanges: [{
        nodeId: "node-a",
        stageIndex: 0,
        layerStart: 0,
        layerEnd: 12,
      }, {
        nodeId: "node-b",
        stageIndex: 1,
        layerStart: 12,
        layerEnd: 24,
      }],
    }]);
    const tracker = new StableBenchmarkActivationTracker(2);

    expect(tracker.observe([first], 1_000)).toEqual([]);
    expect(tracker.observe([first], 2_000)).toEqual([first]);
    expect(tracker.observe([replacement], 3_000)).toEqual([]);
    expect(tracker.observe([replacement], 4_000)).toEqual([replacement]);
    expect(replacement.activationId).not.toBe(first.activationId);
  });

  it("resets stability after an observation gap or disappearance", () => {
    const activation = sealBenchmarkActivation("qwen", [participant()]);
    const tracker = new StableBenchmarkActivationTracker(2, 5_000);

    expect(tracker.observe([activation], 1_000)).toEqual([]);
    expect(tracker.observe([activation], 10_000)).toEqual([]);
    expect(tracker.observe([], 11_000)).toEqual([]);
    expect(tracker.observe([activation], 12_000)).toEqual([]);
    expect(tracker.observe([activation], 13_000)).toEqual([activation]);
  });

  it("fails closed on inconsistent model digests", () => {
    expect(() => sealBenchmarkActivation("qwen", [
      participant(),
      { ...participant(), workerId: "worker-b", modelDigest: "sha256:other" },
    ])).toThrow("benchmark_activation_model_digest_is_inconsistent");
  });

  it("detects a tampered activation seal", () => {
    const activation = sealBenchmarkActivation("qwen", [participant()]);
    const tracker = new StableBenchmarkActivationTracker(1);
    expect(() => tracker.observe([{
      ...activation,
      topologyDigest: "sha256:tampered",
    }], 1_000)).toThrow("benchmark_activation_seal_is_invalid");
  });
});

function participant(): BenchmarkActivationParticipant {
  return {
    workerId: "worker-a",
    agentVersion: "0.3.0",
    deploymentId: "deployment-1",
    modelDigest: "sha256:model",
    nodeIds: ["node-a"],
    stageRanges: [{
      nodeId: "node-a",
      stageIndex: 0,
      layerStart: 0,
      layerEnd: 24,
    }],
  };
}
