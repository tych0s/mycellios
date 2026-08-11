import { describe, expect, it } from "vitest";

import {
  DRAFT_STRATEGY_DESCRIPTOR_SCHEMA,
  ENGINE_CERTIFICATION_SCHEMA,
  ENGINE_FAMILY_DESCRIPTOR_SCHEMA,
  draftStrategyDescriptorSchema,
  engineFamilyDescriptorSchema,
  sealEngineCertification,
} from "../src/contracts/engine-family.js";
import {
  EngineFamilyResolutionError,
  NativeEngineFamilyRegistry,
} from "../src/distribution/engine-family-registry.js";

const digest = (character: string) => `sha256:${character.repeat(64)}` as const;
const revision = "a".repeat(40);

function descriptor() {
  return {
    schema: ENGINE_FAMILY_DESCRIPTOR_SCHEMA,
    familyId: "qwen3-dense",
    version: "1.0.0",
    implementationDigest: digest("1"),
    tensorAbi: "mycellios-tensor-v1",
    model: {
      modelId: "Qwen/Qwen3-8B",
      revision,
      manifestDigest: digest("2"),
      nativeAdapterId: "transformers-qwen3-v1",
      implementationContract: "mycellios-selective-qwen3-v1",
    },
    ownership: {
      embeddings: "head" as const,
      outputHead: "tail" as const,
      tokenizer: "root" as const,
      sharedTensorIds: ["rotary-embedding"],
    },
    operations: [
      "prepare",
      "load",
      "prefill",
      "verify",
      "decode",
      "reset",
      "checkpoint",
      "rollback",
      "health",
      "receipt-observe",
      "unload",
    ] as const,
    targets: [
      {
        platform: "linux" as const,
        arch: "x64" as const,
        backend: "cuda" as const,
        runtimeAbi: "cuda-12",
        quantizations: ["bf16", "q4_k_m"],
        context: { minTokens: 1, maxTokens: 32_768 },
        graphMode: "optional" as const,
        minDriver: "550.54",
      },
    ],
    boundaryConstraints: [
      {
        role: "privacy" as const,
        trust: "private-network" as const,
        requiredBackends: ["cuda" as const],
        requiredFailureDomain: "distinct-host" as const,
      },
    ],
  };
}

function certification(descriptorDigest: string) {
  return sealEngineCertification({
    schema: ENGINE_CERTIFICATION_SCHEMA,
    descriptorDigest,
    sourceId: digest("4"),
    artifactManifestDigest: digest("2"),
    status: "certified" as const,
    validFrom: "2026-08-01T00:00:00.000Z",
    expiresAt: "2026-09-01T00:00:00.000Z",
    targets: descriptor().targets,
    hardwareClasses: ["rtx-4090"],
    parity: [
      {
        kind: "tokens" as const,
        evidenceDigest: digest("5"),
        evidenceClass: "lan" as const,
      },
    ],
  });
}

function samplingCertification(descriptorDigest: string) {
  const { certificationId: _certificationId, ...base } = certification(descriptorDigest);
  return sealEngineCertification({
    ...base,
    sourceId: digest("6"),
    parity: [{
      kind: "sampling-distribution" as const,
      evidenceDigest: digest("7"),
      evidenceClass: "lan" as const,
      samplerSchema: "mycellios-rejection-sampler-v1",
      seedCount: 64,
      sampleCount: 16_384,
      maximumTotalVariationDistance: 0.005,
      minimumGoodnessOfFitPValue: 0.05,
    }],
  });
}

function request(descriptorDigest: `sha256:${string}`) {
  return {
    descriptorDigest,
    modelId: "Qwen/Qwen3-8B",
    revision,
    manifestDigest: digest("2"),
    tensorAbi: "mycellios-tensor-v1",
    platform: "linux" as const,
    arch: "x64" as const,
    backend: "cuda" as const,
    runtimeAbi: "cuda-12",
    quantization: "q4_k_m",
    contextTokens: 8_192,
    hardwareClass: "rtx-4090",
    generationMode: "greedy" as const,
    now: new Date("2026-08-10T00:00:00.000Z"),
  };
}

describe("native engine family contracts", () => {
  it("requires the complete stage lifecycle and strict fields", () => {
    expect(engineFamilyDescriptorSchema.parse(descriptor()).familyId).toBe(
      "qwen3-dense",
    );
    expect(() =>
      engineFamilyDescriptorSchema.parse({
        ...descriptor(),
        operations: descriptor().operations.filter(
          (operation) => operation !== "rollback",
        ),
      }),
    ).toThrow(/engine_required_operation_is_missing:rollback/);
    expect(() =>
      engineFamilyDescriptorSchema.parse({ ...descriptor(), externalDaemon: "shard" }),
    ).toThrow();
  });

  it("validates bounded draft strategy compatibility", () => {
    const strategy = draftStrategyDescriptorSchema.parse({
      schema: DRAFT_STRATEGY_DESCRIPTOR_SCHEMA,
      strategyId: "qwen3-ngram",
      version: "1.0.0",
      kind: "ngram",
      componentDigest: digest("6"),
      targetDescriptorDigest: digest("7"),
      tokenizerDigest: digest("8"),
      vocabularyDigest: digest("9"),
      resource: {
        minimumRamBytes: 0,
        minimumVramBytes: 0,
        allowedBackends: ["cpu"],
      },
      limits: {
        minDraftTokens: 1,
        maxDraftTokens: 16,
        maxInflightWaves: 8,
      },
      evidenceDigest: digest("a"),
    });
    expect(strategy.kind).toBe("ngram");
  });

  it("resolves only a current certified exact target", () => {
    const registry = new NativeEngineFamilyRegistry();
    const descriptorDigest = registry.registerDescriptor(descriptor());
    registry.registerCertification(certification(descriptorDigest));

    const resolved = registry.resolve(request(descriptorDigest));
    expect(resolved.descriptor.familyId).toBe("qwen3-dense");
    expect(resolved.target.backend).toBe("cuda");

    expect(() => registry.resolve({
      ...request(descriptorDigest),
      generationMode: "sampling",
    })).toThrowError(new EngineFamilyResolutionError(
      "engine_compatible_certification_is_missing",
    ));

    expect(() =>
      registry.resolve({ ...request(descriptorDigest), contextTokens: 65_536 }),
    ).toThrowError(
      new EngineFamilyResolutionError("engine_descriptor_target_mismatch"),
    );
    expect(() =>
      registry.resolve({ ...request(descriptorDigest), hardwareClass: "rtx-3090" }),
    ).toThrowError(
      new EngineFamilyResolutionError(
        "engine_compatible_certification_is_missing",
      ),
    );
    expect(() =>
      registry.resolve({
        ...request(descriptorDigest),
        now: new Date("2026-10-01T00:00:00.000Z"),
      }),
    ).toThrowError(
      new EngineFamilyResolutionError(
        "engine_compatible_certification_is_missing",
      ),
    );
  });

  it("rejects certification before its descriptor exists", () => {
    const registry = new NativeEngineFamilyRegistry();
    expect(() => registry.registerCertification(certification(digest("7"))))
      .toThrowError(
        new EngineFamilyResolutionError(
          "engine_certification_descriptor_is_unknown",
        ),
      );
  });

  it("requires a separate statistically bounded sampling certification", () => {
    const registry = new NativeEngineFamilyRegistry();
    const descriptorDigest = registry.registerDescriptor(descriptor());
    registry.registerCertification(certification(descriptorDigest));
    expect(() => registry.resolve({
      ...request(descriptorDigest),
      generationMode: "sampling",
    })).toThrowError(new EngineFamilyResolutionError(
      "engine_compatible_certification_is_missing",
    ));

    registry.registerCertification(samplingCertification(descriptorDigest));
    expect(registry.resolve({
      ...request(descriptorDigest),
      generationMode: "sampling",
    }).certification.parity[0]?.kind).toBe("sampling-distribution");
    expect(() => sealEngineCertification({
      ...(() => {
        const { certificationId: _id, ...content } = samplingCertification(descriptorDigest);
        return content;
      })(),
      parity: [{
        kind: "sampling-distribution",
        evidenceDigest: digest("7"),
        evidenceClass: "lan",
        samplerSchema: "mycellios-rejection-sampler-v1",
        seedCount: 64,
        sampleCount: 1_024,
        maximumTotalVariationDistance: 0.005,
        minimumGoodnessOfFitPValue: 0.05,
      }],
    })).toThrow();
  });

  it("rejects a certification whose content changed after sealing", () => {
    const registry = new NativeEngineFamilyRegistry();
    const descriptorDigest = registry.registerDescriptor(descriptor());
    expect(() => registry.registerCertification({
      ...certification(descriptorDigest),
      expiresAt: "2026-12-01T00:00:00.000Z",
    })).toThrowError(
      new EngineFamilyResolutionError("engine_certification_identity_mismatch"),
    );
  });
});
