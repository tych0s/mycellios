import { describe, expect, it } from "vitest";

import {
  buildModelDistributionManifest,
  parseModelDistributionManifest,
  type ModelDistributionManifestBuildInput,
} from "../src/contracts/model-distribution-manifest.js";

const digest = (character: string) => `sha256:${character.repeat(64)}` as const;
const artifact = (
  id: string,
  role: "shared" | "input" | "output" | "layer-range" | "expert-range",
  layerRange: { start: number; end: number } | null = null,
) => ({
  id,
  role,
  path: `model/${id}.safetensors`,
  sha256: digest(id === "input" ? "a" : id === "output" ? "b" : "c"),
  sizeBytes: 1_024,
  sourceUrl: `https://models.example.test/revisions/${"1".repeat(40)}/${id}`,
  layerRange,
  expertRange: null,
});

function fixture(): ModelDistributionManifestBuildInput {
  return {
    model: { id: "Qwen/Qwen3-0.6B", revision: "1".repeat(40), architecture: "Qwen3ForCausalLM" },
    tokenizer: { id: "Qwen/Qwen3-0.6B", revision: "1".repeat(40) },
    format: "safetensors",
    quantization: { scheme: "none", bits: null, groupSize: null },
    tensorAbi: "mycellios-tensor/1",
    layerCount: 28,
    expertCount: null,
    license: { spdx: "Apache-2.0", sourceUrl: "https://models.example.test/licenses/apache-2.0" },
    artifacts: [
      artifact("output", "output"),
      artifact("layers-14-28", "layer-range", { start: 14, end: 28 }),
      artifact("input", "input"),
      artifact("layers-0-14", "layer-range", { start: 0, end: 14 }),
      artifact("shared", "shared"),
    ],
  };
}

describe("model distribution manifest", () => {
  it("builds a canonical content-addressed immutable distribution", () => {
    const manifest = buildModelDistributionManifest(fixture());
    expect(manifest.manifestId).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(manifest.artifacts.map(({ id }) => id)).toEqual([
      "input", "layers-0-14", "layers-14-28", "output", "shared",
    ]);
    expect(parseModelDistributionManifest(JSON.parse(JSON.stringify(manifest)))).toEqual(manifest);
  });

  it("rejects mutable revisions, unsafe paths and non-HTTPS sources", () => {
    expect(() => buildModelDistributionManifest({ ...fixture(), model: { ...fixture().model, revision: "main" } })).toThrow();
    expect(() => buildModelDistributionManifest({ ...fixture(), artifacts: [{ ...fixture().artifacts[0]!, path: "../escape" }, ...fixture().artifacts.slice(1)] })).toThrow("model_distribution_artifact_path_is_unsafe");
    expect(() => buildModelDistributionManifest({ ...fixture(), artifacts: [{ ...fixture().artifacts[0]!, sourceUrl: "http://models.example.test/file" }, ...fixture().artifacts.slice(1)] })).toThrow("model_distribution_artifact_source_is_unsafe");
  });

  it("rejects layer gaps, overlaps, duplicate ownership and role mismatches", () => {
    const input = fixture();
    expect(() => buildModelDistributionManifest({ ...input, artifacts: input.artifacts.map((entry) => entry.id === "layers-14-28" ? { ...entry, layerRange: { start: 15, end: 28 } } : entry) })).toThrow("model_distribution_layer_ownership_has_gap_or_overlap");
    expect(() => buildModelDistributionManifest({ ...input, artifacts: [...input.artifacts, { ...input.artifacts[0]!, id: "output-copy" }] })).toThrow("model_distribution_output_artifact_is_not_unique");
    expect(() => buildModelDistributionManifest({ ...input, artifacts: input.artifacts.map((entry) => entry.id === "shared" ? { ...entry, layerRange: { start: 0, end: 1 } } : entry) })).toThrow("model_distribution_layer_range_role_mismatch");
  });

  it("detects tampering and unknown fields", () => {
    const manifest = buildModelDistributionManifest(fixture());
    expect(() => parseModelDistributionManifest({ ...manifest, layerCount: 27 })).toThrow();
    expect(() => parseModelDistributionManifest({ ...manifest, future: true })).toThrow();
  });
});
