import { describe, expect, it } from "vitest";
import {
  ENGINE_ACTIVATION_AUTHORITY_SCHEMA,
  buildEngineRuntimeActivationPlans,
  engineActivationAuthoritySchema,
} from "../src/coordinator/engine-activation-authority.js";
import {
  ENGINE_CERTIFICATION_SCHEMA,
  ENGINE_FAMILY_DESCRIPTOR_SCHEMA,
  sealEngineCertification,
} from "../src/contracts/engine-family.js";
import { sha256CanonicalEvidence } from "../src/core/json.js";
import type { AutoDistributionRunResult } from "../src/distribution/auto-distribute.js";
import type { StoredWorker } from "../src/storage/store.js";

const digest = (value: string) => `sha256:${value.repeat(64)}` as const;
const revision = "1".repeat(40);
const artifact = digest("a");

function authority() {
  const descriptor = {
    schema: ENGINE_FAMILY_DESCRIPTOR_SCHEMA,
    familyId: "qwen3-dense", version: "1.0.0", implementationDigest: digest("1"),
    tensorAbi: "mycellios-tensor-v1",
    model: { modelId: "Qwen/Qwen3-Test", revision, manifestDigest: artifact,
      nativeAdapterId: "transformers-qwen3-v1",
      implementationContract: "mycellios-selective-qwen3-v1" },
    ownership: { embeddings: "head" as const, outputHead: "tail" as const,
      tokenizer: "root" as const, sharedTensorIds: [] },
    operations: ["prepare", "load", "prefill", "verify", "decode", "reset",
      "checkpoint", "rollback", "health", "receipt-observe", "unload"] as const,
    targets: [{ platform: "linux" as const, arch: "x64" as const,
      backend: "cuda" as const, runtimeAbi: "cuda-12", quantizations: ["bf16"],
      context: { minTokens: 1, maxTokens: 8_192 }, graphMode: "optional" as const,
      minDriver: null }],
    boundaryConstraints: [],
  };
  return {
    schema: ENGINE_ACTIVATION_AUTHORITY_SCHEMA,
    descriptor,
    certification: sealEngineCertification({
      schema: ENGINE_CERTIFICATION_SCHEMA,
      descriptorDigest: sha256CanonicalEvidence(descriptor), sourceId: digest("3"),
      artifactManifestDigest: artifact, status: "certified" as const,
      validFrom: "2026-08-01T00:00:00.000Z", expiresAt: "2026-09-01T00:00:00.000Z",
      targets: descriptor.targets, hardwareClasses: ["nvidia-test-gpu"],
      parity: [{ kind: "tokens" as const, evidenceDigest: digest("4"), evidenceClass: "lan" as const }],
    }),
    probe: { kind: "qwen3-dense-v1" as const, quantization: "bf16",
      attentionHeads: 8, kvHeads: 2, headDim: 64,
      referenceDecodeMsPerToken: 8, referencePrefillMsPerToken: 0.4,
      minimumSamples: 7 },
  };
}

function result(): AutoDistributionRunResult {
  const layers = Array.from({ length: 4 }, (_, index) => ({
    index, weightBytes: 1_024, activationElements: 512, kvBytesPerToken: 512,
    decodeMsAtUnit: 1, prefillMsPerTokenAtUnit: 1,
  }));
  return {
    profile: { schema: "gdlp-model-profile/1",
      source: { model: "Qwen/Qwen3-Test", revision, snapshotCommit: revision,
        snapshotIdentityUint64Hex: "0000000000000001", artifactIdentity: artifact,
        canonicalSource: "Qwen/Qwen3-Test", canonicalRevision: revision, format: "safetensors" },
      inspection: { architecture: "Qwen3ForCausalLM", calibrationRequired: false },
      compatibility: { selectiveSafetensors: true, requiresAdapter: true,
        adapterId: "transformers-qwen3-v1", reasons: [] },
      model: { id: "qwen-public", layers, embeddingBytes: 1, lmHeadBytes: 1,
        runtimeOverheadBytesPerStage: 1, embeddingDecodeMsAtUnit: 1,
        lmHeadDecodeMsAtUnit: 1, embeddingPrefillMsPerTokenAtUnit: 1,
        lmHeadPrefillMsPerTokenAtUnit: 1 } },
    request: { workload: { contextTokens: 4_096 } },
    manifest: { hiddenSize: 512, plans: { decode: { stages: [
      { index: 0, layerStart: 0, layerEnd: 2, members: [{ nodeId: "node-a" }] },
      { index: 1, layerStart: 2, layerEnd: 4, members: [{ nodeId: "node-b" }] },
    ] } } }, canaryText: "OK", canaryMetrics: { completionTokens: 1, ttftMs: 1,
      tpotMs: 1, pipelineMs: 2, measuredTokensPerSecond: 1 },
  } as unknown as AutoDistributionRunResult;
}

function worker(id: string, nodeId: string, stageIndex: number): StoredWorker {
  const layerStart = stageIndex * 2;
  return { id, status: "online", reliability: 1, jobsCompleted: 0, lastSeenAt: Date.now(),
    identityKind: "device", identityId: id,
    capabilities: { region: "test", agentVersion: "1", gpus: [{ id: "gpu-0",
      vendor: "NVIDIA", model: "Test GPU", physicalVramMb: 16_384,
      offeredVramMb: 16_384, freeOfferedVramMb: 16_384 }],
      limits: { maxConcurrency: 1, pauseWhenForeground: false },
      network: { coordinatorRttMs: 1, uplinkMbps: 1_000, downlinkMbps: 1_000 },
      distributedExecutor: { protocol: "gdlp-worker-tunnel/2", nodeId,
        stageHost: "127.0.0.1", stagePort: 9000 + stageIndex, runtime: "python-safetensors",
        physicalIdentity: { schema: "gdlp-worker-physical-identity/1", provider: "generic",
          providerMachineFingerprintSha256: digest("5"), hostFingerprintSha256: digest("6"),
          gpuFingerprintsSha256: [digest("7")], attestedAt: "2026-08-10T12:00:00Z" },
        performanceEvidence: { profile: { backend: "cuda" } } as never },
      deployments: [{ deploymentId: `deployment-${id}`, model: "qwen-public",
        modelDigest: artifact, activationId: "activation-1", mode: "pipeline",
        adapter: "mycellios-pipeline", peakVramMb: 1, contextLimit: 4_096,
        maxConcurrency: 1, freeSlots: 1, tokensPerSecond: 1,
        throughputSource: "measured", ttftMs: 1, verificationState: "verified",
        dataLocality: "local", stage: { index: stageIndex, total: 2,
          layerStart, layerEnd: layerStart + 2 }, execution: { deviceType: "gpu",
          backend: "cuda", deviceName: "Test GPU", precision: "bf16", fallback: false,
          stages: [{ nodeId, stageIndex, layerStart, layerEnd: layerStart + 2,
            deviceType: "gpu", backend: "cuda", deviceName: "Test GPU",
            precision: "bf16", fallback: false }] } }] } };
}

describe("engine activation authority", () => {
  it("accepts only canonical draft tokenizer and vocabulary fingerprints", () => {
    const current = authority();
    expect(engineActivationAuthoritySchema.parse({
      ...current,
      draftCompatibility: {
        tokenizerDigest: digest("8"),
        vocabularyDigest: digest("9"),
      },
    }).draftCompatibility).toEqual({
      tokenizerDigest: digest("8"),
      vocabularyDigest: digest("9"),
    });
    expect(() => engineActivationAuthoritySchema.parse({
      ...current,
      draftCompatibility: {
        tokenizerDigest: "tokenizer-latest",
        vocabularyDigest: digest("9"),
      },
    })).toThrow();
  });

  it("derives exact per-stage physical challenges from certified activation evidence", () => {
    const plans = buildEngineRuntimeActivationPlans(authority(), "qwen-public", "activation-1", "reservation-1",
      result(), [worker("worker-a", "node-a", 0), worker("worker-b", "node-b", 1)],
      Date.parse("2026-08-10T12:00:00Z"));
    expect(plans).toHaveLength(2);
    expect(plans.map((plan) => plan.request.requiredRoles)).toEqual([["head"], ["tail"]]);
    expect(plans.every((plan) => plan.request.expectedKvBytesPerToken === 1_024)).toBe(true);
  });

  it("fails closed without exact verified stage evidence", () => {
    const missing = worker("worker-a", "node-a", 0);
    delete missing.capabilities.distributedExecutor!.performanceEvidence;
    expect(() => buildEngineRuntimeActivationPlans(authority(), "qwen-public", "activation-1", "reservation-1",
      result(), [missing, worker("worker-b", "node-b", 1)],
      Date.parse("2026-08-10T12:00:00Z")))
      .toThrow("engine_activation_stage_evidence_is_missing:node-a:0");
  });
});
