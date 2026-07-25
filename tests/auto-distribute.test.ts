import { describe, expect, it } from "vitest";
import {
  AUTO_DISTRIBUTE_SCHEMA,
  buildCellWorkerConfig,
  compileAutoDistribution,
  parseAutoDistributionConfig,
  type AutoDistributionConfig,
  type CompiledModelProfile,
} from "../src/distribution/auto-distribute.js";
import { deploymentMetricsFromCanaryEvidence } from "../src/contracts/deployment-canary.js";
import type { DistributedModelProfile } from "../src/distribution/types.js";

const MIB = 1024 * 1024;
const SNAPSHOT_HEX = "0000000000003039";
const ARTIFACT_IDENTITY = `sha256:${SNAPSHOT_HEX}${"a".repeat(48)}`;

describe("automatic compatible-model distribution", () => {
  it("forces a real multi-stage route and calculates layer boundaries", () => {
    const config = parseAutoDistributionConfig(configFixture());
    const compiled = compileAutoDistribution(config, profileFixture());

    expect(compiled.profile.compatibility.adapterId).toBe("transformers-llama-v1");
    expect(compiled.boundaries[0]).toBe(0);
    expect(compiled.boundaries.at(-1)).toBe(8);
    expect(compiled.boundaries).toHaveLength(3);
    expect(compiled.manifest.plans.decode.stages).toHaveLength(2);
    expect(compiled.manifest.plans.decode.stages.map((stage) => stage.anchor.memberId)).toEqual([
      "node-a",
      "node-b",
    ]);
    expect(compiled.launch.runtimeModel).toMatchObject({
      source: "local-compatible-model",
      revision: null,
      snapshotIdentity: "12345",
      artifactIdentity: ARTIFACT_IDENTITY,
    });
    expect(compiled.launch.launchOrder.at(-1)?.kind).toBe("root-engine");
  });

  it("rejects a model.layers checkpoint when no adapter certified its architecture", () => {
    const profile = profileFixture();
    profile.compatibility = {
      selectiveSafetensors: false,
      requiresAdapter: true,
      adapterId: null,
      reasons: ["no certified selective-stage adapter for model_type='mistral'"],
    };

    expect(() => compileAutoDistribution(parseAutoDistributionConfig(configFixture()), profile))
      .toThrow(/model_is_not_automatically_distributable:no certified selective-stage adapter/);
  });

  it("rejects duplicate nodes and impossible minimum stage counts at config load", () => {
    const value = configFixture();
    value.nodes[1]!.id = "node-a";
    value.distribution.minimumStages = 3;

    expect(() => parseAutoDistributionConfig(value)).toThrow();
  });

  it("seals the activation canary instead of configuring measured TPS manually", () => {
    const input = configFixture();
    input.coordinator = {
      url: "http://127.0.0.1:8080",
      region: "test-lan",
      maxConcurrency: 1,
    };
    const config = parseAutoDistributionConfig(input);
    const compilation = compileAutoDistribution(config, profileFixture());
    const worker = buildCellWorkerConfig(
      config,
      compilation,
      "http://127.0.0.1:8088",
      [1, 2, 3].map((index) => ({
        sampleId: `canary-${index}`,
        outputTokens: 8,
        activeMs: 1_000,
        ttftMs: 250 + index,
        completed: true as const,
      })),
      "pipeline-activation-123",
    );

    expect(worker.deployment.tokensPerSecond).toBeUndefined();
    expect(worker.deployment.ttftMs).toBeUndefined();
    expect(worker.deployment.canaryEvidence).toMatchObject({
      model: config.model.publicName,
      modelDigest: ARTIFACT_IDENTITY,
      activationId: "pipeline-activation-123",
      warmupSamples: 1,
    });
    expect(deploymentMetricsFromCanaryEvidence(
      worker.deployment.canaryEvidence!,
      {
        model: config.model.publicName,
        modelDigest: ARTIFACT_IDENTITY,
        activationId: "pipeline-activation-123",
      },
    )).toMatchObject({
      tokensPerSecond: 8,
      ttftMs: 252,
    });
  });
});

function configFixture(): AutoDistributionConfig {
  return {
    schema: AUTO_DISTRIBUTE_SCHEMA,
    model: {
      source: "local-compatible-model",
      revision: null,
      publicName: "automatic-test-model",
    },
    nodes: [
      {
        id: "node-a",
        region: "test-lan",
        endpoint: { host: "127.0.0.1", port: 21_001 },
        memoryMiB: 2_048,
        reserveMiB: 256,
        decodeScale: 1,
        prefillScale: 1,
        codecScale: 1,
        powerWatts: 65,
        availability: 0.999,
        agent: { kind: "local" },
      },
      {
        id: "node-b",
        region: "test-lan",
        endpoint: { host: "127.0.0.1", port: 21_002 },
        memoryMiB: 2_048,
        reserveMiB: 256,
        decodeScale: 1.1,
        prefillScale: 1.1,
        codecScale: 1,
        powerWatts: 65,
        availability: 0.999,
        agent: { kind: "local" },
      },
    ],
    links: [],
    distribution: {
      minimumStages: 2,
      maximumStages: 2,
      allowLossyActivation: false,
    },
    workload: {
      promptTokens: 32,
      outputTokens: 16,
      contextTokens: 128,
      concurrentSequences: 1,
      minRouteAvailability: 0.9,
      batchWindowMs: 1,
      p95: false,
    },
    runtime: {
      pythonExecutable: "runtime/distribution-venv/Scripts/python.exe",
      pythonPath: "python",
      hfHome: "runtime/hf-cache",
      apiEndpoint: { host: "127.0.0.1", port: 8_088 },
      apiAdvertiseHost: "127.0.0.1",
      returnEndpoint: { host: "127.0.0.1", port: 30_000 },
      returnBindHost: "127.0.0.1",
      threadsPerStage: 1,
      connectTimeoutSeconds: 60,
      readinessTimeoutMs: 120_000,
      maxOutputTokens: 128,
    },
    canary: {
      prompt: "Reply only OK",
      maxTokens: 8,
      timeoutMs: 120_000,
    },
  };
}

function profileFixture(): CompiledModelProfile {
  return {
    schema: "gdlp-model-profile/1",
    source: {
      model: "local-compatible-model",
      revision: null,
      snapshotCommit: null,
      snapshotIdentityUint64Hex: SNAPSHOT_HEX,
      artifactIdentity: ARTIFACT_IDENTITY,
      canonicalSource: `content-addressed://${ARTIFACT_IDENTITY}`,
      canonicalRevision: ARTIFACT_IDENTITY,
      format: "safetensors",
    },
    inspection: {
      architecture: "LlamaForCausalLM",
      calibrationRequired: true,
    },
    compatibility: {
      selectiveSafetensors: true,
      requiresAdapter: false,
      adapterId: "transformers-llama-v1",
      reasons: [],
    },
    model: modelProfile(),
  };
}

function modelProfile(): DistributedModelProfile {
  return {
    id: "local-compatible-model",
    layers: Array.from({ length: 8 }, (_, index) => ({
      index,
      weightBytes: 32 * MIB,
      activationElements: 512,
      kvBytesPerToken: 128,
      decodeMsAtUnit: 1,
      prefillMsPerTokenAtUnit: 0.1,
    })),
    embeddingBytes: 16 * MIB,
    lmHeadBytes: 16 * MIB,
    tiedEmbeddingAndHead: false,
    runtimeOverheadBytesPerStage: 256 * MIB,
    embeddingDecodeMsAtUnit: 0,
    lmHeadDecodeMsAtUnit: 0,
    embeddingPrefillMsPerTokenAtUnit: 0,
    lmHeadPrefillMsPerTokenAtUnit: 0,
  };
}
