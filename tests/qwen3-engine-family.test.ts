import { describe, expect, it } from "vitest";

import {
  NativeEngineFamilyRegistry,
} from "../src/distribution/engine-family-registry.js";
import {
  QWEN3_DENSE_IMPLEMENTATION_CONTRACT,
  QWEN3_DENSE_NATIVE_ADAPTER,
  buildQwen3DenseEngineDescriptor,
} from "../src/distribution/engine-families/qwen3-dense/descriptor.js";

const digest = (character: string) => `sha256:${character.repeat(64)}` as const;

describe("Qwen3 dense engine-family adapter", () => {
  it("binds the existing native adapter to the shared certified seam", () => {
    const descriptor = buildQwen3DenseEngineDescriptor({
      version: "1.0.0",
      implementationDigest: digest("1"),
      tensorAbi: "mycellios-tensor-v1",
      modelId: "Qwen/Qwen3-0.6B",
      revision: "a".repeat(40),
      manifestDigest: digest("2"),
      targets: [
        {
          platform: "linux",
          arch: "x64",
          backend: "cuda",
          runtimeAbi: "cuda-12",
          quantizations: ["bf16"],
          context: { minTokens: 1, maxTokens: 32_768 },
          graphMode: "optional",
          minDriver: null,
        },
      ],
    });

    expect(descriptor.model.nativeAdapterId).toBe(
      QWEN3_DENSE_NATIVE_ADAPTER,
    );
    expect(descriptor.model.implementationContract).toBe(
      QWEN3_DENSE_IMPLEMENTATION_CONTRACT,
    );
    expect(descriptor.ownership).toMatchObject({
      embeddings: "head",
      outputHead: "tail",
      tokenizer: "root",
    });

    const registry = new NativeEngineFamilyRegistry();
    expect(registry.registerDescriptor(descriptor)).toMatch(
      /^sha256:[0-9a-f]{64}$/,
    );
  });
});

