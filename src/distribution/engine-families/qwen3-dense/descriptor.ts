import {
  ENGINE_FAMILY_DESCRIPTOR_SCHEMA,
  engineFamilyDescriptorSchema,
  type EngineFamilyDescriptor,
  type EngineTarget,
} from "../../../contracts/engine-family.js";

export const QWEN3_DENSE_ENGINE_FAMILY = "qwen3-dense" as const;
export const QWEN3_DENSE_NATIVE_ADAPTER = "transformers-qwen3-v1" as const;
export const QWEN3_DENSE_IMPLEMENTATION_CONTRACT =
  "mycellios-selective-qwen3-v1" as const;

export interface Qwen3DenseEngineDescriptorInput {
  version: string;
  implementationDigest: `sha256:${string}`;
  tensorAbi: string;
  modelId: string;
  revision: string;
  manifestDigest: `sha256:${string}`;
  targets: EngineTarget[];
}

/**
 * Adapts the existing certified Qwen3 selective loader to the common engine
 * seam. It is descriptor-only: runtime selection still requires a separate,
 * current EngineCertification in the fail-closed registry.
 */
export function buildQwen3DenseEngineDescriptor(
  input: Qwen3DenseEngineDescriptorInput,
): EngineFamilyDescriptor {
  return engineFamilyDescriptorSchema.parse({
    schema: ENGINE_FAMILY_DESCRIPTOR_SCHEMA,
    familyId: QWEN3_DENSE_ENGINE_FAMILY,
    version: input.version,
    implementationDigest: input.implementationDigest,
    tensorAbi: input.tensorAbi,
    model: {
      modelId: input.modelId,
      revision: input.revision,
      manifestDigest: input.manifestDigest,
      nativeAdapterId: QWEN3_DENSE_NATIVE_ADAPTER,
      implementationContract: QWEN3_DENSE_IMPLEMENTATION_CONTRACT,
    },
    ownership: {
      embeddings: "head",
      outputHead: "tail",
      tokenizer: "root",
      sharedTensorIds: ["token-embedding", "final-norm", "output-head"],
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
    ],
    targets: input.targets,
    boundaryConstraints: [
      {
        role: "head",
        trust: "account",
        requiredBackends: [],
        requiredFailureDomain: "any",
      },
      {
        role: "tail",
        trust: "account",
        requiredBackends: [],
        requiredFailureDomain: "any",
      },
      {
        role: "privacy",
        trust: "private-network",
        requiredBackends: [],
        requiredFailureDomain: "any",
      },
    ],
  });
}

