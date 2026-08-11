import { describe, expect, it } from "vitest";

import { buildModelDistributionManifest } from "../src/contracts/model-distribution-manifest.js";
import { selectModelArtifactsForAssignment } from "../src/model-fabric/model-artifact-selection.js";

const sha = (character: string) => `sha256:${character.repeat(64)}` as const;
const artifact = (
  id: string,
  role: "shared" | "input" | "output" | "layer-range",
  layerRange: { start: number; end: number } | null,
  sizeBytes: number,
) => ({
  id, role, layerRange, expertRange: null,
  path: `model/${id}.safetensors`, sha256: sha(id === "input" ? "a" : id === "output" ? "b" : id.startsWith("layer") ? "c" : "d"),
  sizeBytes, sourceUrl: `https://models.example.test/${id}`,
});

const manifest = buildModelDistributionManifest({
  model: { id: "Qwen/Qwen3", revision: "1".repeat(40), architecture: "Qwen3ForCausalLM" },
  tokenizer: { id: "Qwen/Qwen3", revision: "1".repeat(40) },
  format: "safetensors", quantization: { scheme: "none", bits: null, groupSize: null },
  tensorAbi: "mycellios-tensor/1", layerCount: 28, expertCount: null,
  license: { spdx: "Apache-2.0", sourceUrl: "https://models.example.test/license" },
  artifacts: [
    artifact("shared", "shared", null, 100), artifact("input", "input", null, 200),
    artifact("layer-0-14", "layer-range", { start: 0, end: 14 }, 1_400),
    artifact("layer-14-28", "layer-range", { start: 14, end: 28 }, 1_400),
    artifact("output", "output", null, 300),
  ],
});

describe("model artifact selection", () => {
  it("returns only shared, explicitly owned boundaries and the assigned range", () => {
    const selected = selectModelArtifactsForAssignment(manifest, {
      layerStart: 14, layerEnd: 28, expertStart: null, expertEnd: null,
      includeInput: false, includeOutput: true,
    });
    expect(selected.artifactIds).toEqual(["layer-14-28", "output", "shared"]);
    expect(selected.totalBytes).toBe(1_800);
    expect(selected.artifactIds).not.toContain("input");
    expect(selected.artifactIds).not.toContain("layer-0-14");
  });

  it("rejects assignments that would widen or partially slice a package", () => {
    expect(() => selectModelArtifactsForAssignment(manifest, {
      layerStart: 7, layerEnd: 21, expertStart: null, expertEnd: null,
      includeInput: false, includeOutput: false,
    })).toThrow("model_artifact_layer_assignment_not_package_aligned");
  });
});
