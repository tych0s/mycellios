import { describe, expect, it } from "vitest";
import {
  compilePythonLaunchDescription,
  type PythonLaunchCompilerOptions,
} from "../src/distribution/python-launcher.js";
import {
  buildRuntimePipelineManifest,
  validateRuntimePipelineManifest,
  type RuntimePipelineManifestV2,
  type RuntimePlanRequest,
} from "../src/distribution/runtime-manifest.js";

const MIB = 1024 * 1024;

function request(usableVramBytes: number, macroWave = true): RuntimePlanRequest {
  const nodes = [0, 1].map((index) => ({
    id: `mw-${index}`,
    region: `home-${index}`,
    memoryBytes: usableVramBytes,
    reserveBytes: 0,
    decodeScale: 1,
    prefillScale: 1,
    codecScale: 1,
    batchGain: 0,
    maxBatchSpeedup: 1,
    powerWatts: 60,
    availability: 0.999,
    endpoint: { host: `mw-${index}.internal`, port: 24_000 + index },
    backend: {
      engine: "python-transformers",
      modelFormats: ["safetensors"],
      executionModes: ["layer-range", "ram-backed-moe-stage"],
    },
    capabilities: {
      deviceKinds: ["gpu"],
      computeApis: ["cuda"],
      weightDtypes: ["fp16"],
      activationCodecs: ["fp16" as const],
      features: [
        "layer-range",
        "kv-reuse",
        "ram-authoritative-routed-experts",
        "device-expert-cache",
        "authoritative-router",
        "predictive-prefetch-only",
        "meta-model-construction",
        "no-full-model-materialization",
      ],
    },
    ramVram: {
      usableRamBytes: 80 * MIB,
      usableVramBytes,
      ramBandwidthGBps: 25,
      pcieBandwidthGBps: 12,
      residentKind: "layers" as const,
    },
  }));
  return {
    model: {
      id: "macro-wave-runtime-fixture",
      layers: Array.from({ length: 4 }, (_, index) => ({
        index,
        weightBytes: 30 * MIB,
        activationElements: 256,
        kvBytesPerToken: 128,
        decodeMsAtUnit: 1,
        prefillMsPerTokenAtUnit: 0.1,
        macroWave: {
          activeWeightBytesPerWave: 12 * MIB,
          largestTransferUnitBytes: 8 * MIB,
          expertWorkspaceBytesPerPosition: 256,
        },
        expertParallel: {
          expertWeightBytes: 26 * MIB,
        },
      })),
      embeddingBytes: MIB,
      lmHeadBytes: MIB,
      runtimeOverheadBytesPerStage: 4 * MIB,
      embeddingDecodeMsAtUnit: 0.1,
      lmHeadDecodeMsAtUnit: 0.1,
      embeddingPrefillMsPerTokenAtUnit: 0.02,
      lmHeadPrefillMsPerTokenAtUnit: 0.02,
    },
    modelRevision: "sha256:macro-wave-runtime-r1",
    tokenizerId: "macro-wave-tokenizer-r1",
    topology: {
      nodes,
      links: [
        {
          from: "mw-0",
          to: "mw-1",
          oneWayLatencyMs: 3,
          jitterP95Ms: 0.2,
          bandwidthMbps: 1_000,
          lossRate: 0,
          availability: 0.999,
        },
        {
          from: "mw-1",
          to: "mw-0",
          oneWayLatencyMs: 3,
          jitterP95Ms: 0.2,
          bandwidthMbps: 1_000,
          lossRate: 0,
          availability: 0.999,
        },
      ],
    },
    workload: {
      promptTokens: 8,
      outputTokens: 4,
      contextTokens: 32,
      concurrentSequences: 1,
      maxStages: 2,
      maxQualityLoss: 0,
      minRouteAvailability: 0.9,
      batchWindowMs: 0,
      p95: true,
    },
    ...(macroWave
      ? {
          planner: {
            kind: "macro-wave" as const,
            options: {
              candidateMicroBatchSizes: [1],
              candidatePrefillChunks: [8],
              waveTokens: 1,
              expectedCommittedTokensPerWave: 1,
            },
          },
        }
      : {}),
  };
}

const launchOptions = {
  apiEndpoint: { host: "0.0.0.0", port: 8_081 },
  returnEndpoint: { host: "root.internal", port: 30_000 },
  returnBindHost: "0.0.0.0",
};

const ARTIFACT_IDENTITY = `sha256:0123456789abcdef${"0".repeat(48)}`;
const PIPELINE_SNAPSHOT_IDENTITY = BigInt("0x0123456789abcdef").toString();

function ramBackedLaunchOptions(
  manifest: RuntimePipelineManifestV2,
): PythonLaunchCompilerOptions {
  return {
    ...launchOptions,
    runtimeModel: {
      source: "C:\\models\\macro-wave-moe",
      revision: null,
      snapshotIdentity: PIPELINE_SNAPSHOT_IDENTITY,
    },
    ramBackedMoeStages: Object.fromEntries(
      manifest.plans.prefill.stages
        .filter((stage) => stage.macroWave?.memoryMode === "ram-backed")
        .map((stage) => {
          const execution = stage.macroWave!;
          return [
            stage.stageId,
            {
              schema: "gdlp-local-safetensors-moe-stage/1" as const,
              snapshotPath: "C:\\models\\macro-wave-moe",
              artifactIdentity: ARTIFACT_IDENTITY,
              adapterId: "transformers-qwen3-moe-v1" as const,
              layerStart: stage.layerStart,
              layerEnd: stage.layerEnd,
              totalLayers: manifest.totalLayers,
              device: "cuda:0",
              pinMemory: false as const,
              allowCpuFallback: false as const,
              cache: {
                schema: "gdlp-predictive-expert-cache/1" as const,
                capacityBytes:
                  execution.requirements.weightBufferBytes +
                  execution.cachePolicy.capacityBytes,
                prefetchReserveBytes:
                  execution.workingSet.largestTransferUnitBytes,
                pcieBandwidthGbytesPerSecond: 12,
                hotnessDecay: 0.95,
                minPrefetchConfidence: 0.6,
              },
            },
          ];
        }),
    ),
  };
}

describe("MacroWave runtime contract integration", () => {
  it("keeps default planning unchanged and preserves an explicit resident contract", () => {
    const standard = buildRuntimePipelineManifest(request(80 * MIB, false));
    expect(standard.plans.decode.transport).toBe("persistent-tcp");
    expect(standard.plans.decode.macroWave).toBeUndefined();
    expect(standard.plans.decode.stages.every((stage) => !stage.macroWave)).toBe(true);

    const manifest = buildRuntimePipelineManifest(request(80 * MIB));
    expect(manifest.plans.decode.transport).toBe("persistent-tcp-macro-wave-v1");
    expect(manifest.plans.decode.macroWave?.schema).toBe("gdlp-macro-wave-plan/1");
    expect(manifest.plans.decode.macroWave?.routeKind).toBe("resident-baseline");
    expect(manifest.plans.decode.stages.every(
      (stage) => stage.macroWave?.memoryMode === "resident",
    )).toBe(true);

    const launch = compilePythonLaunchDescription(manifest, launchOptions);
    expect(launch.route.phaseSettings[1].macroWave).toEqual(
      manifest.plans.decode.macroWave,
    );
    expect(launch.launchOrder.map((process) => process.macroWave)).toEqual(
      [...manifest.plans.decode.stages].reverse().slice(0, -1)
        .map((stage) => stage.macroWave)
        .concat(manifest.plans.decode.stages[0]!.macroWave),
    );
    for (const process of launch.launchOrder) {
      const contract = process.macroWave!;
      expect(argumentValue(
        process.command.args,
        "--dense-host-ram-budget-bytes",
      )).toBe(String(contract.budgets.hostRamBytes));
      expect(argumentValue(
        process.command.args,
        "--dense-vram-budget-bytes",
      )).toBe(String(contract.budgets.vramBytes));
      expect(argumentValue(
        process.command.args,
        "--dense-activation-reserve-bytes",
      )).toBe(String(contract.requirements.activationBufferBytes));
      expect(process.command.args).toContain("--dense-require-full-residency");
    }
  });

  it("seals budgets and compiles certified RAM-backed stages", () => {
    const manifest = buildRuntimePipelineManifest(request(40 * MIB));
    expect(manifest.plans.decode.macroWave?.routeKind).toBe("macro-wave");
    expect(manifest.plans.decode.stages.some(
      (stage) => stage.macroWave?.memoryMode === "ram-backed",
    )).toBe(true);
    const ramStage = manifest.plans.decode.stages.find(
      (stage) => stage.macroWave?.memoryMode === "ram-backed",
    )!;
    const ramExecution = ramStage.macroWave!;
    expect(ramStage.macroWave?.ramArtifact).toMatchObject({
      schema: "gdlp-local-safetensors-moe-stage/1",
      expertExecutionMode: "serial-exact",
      largestExpertBytes: ramExecution.workingSet.largestTransferUnitBytes,
      weightBufferCopies: 2,
      fullModelMaterialization: false,
    });
    expect(
      ramExecution.requirements.boundedPinnedStagingReserveBytes,
    ).toBe(ramExecution.workingSet.largestExpertBytes * 2);
    expect(ramExecution.requirements.residentStreamingTransientBytes).toBe(
      ramExecution.requirements.residentParameterBudgetBytes,
    );
    expect(ramExecution.requirements.hostRamPeakUpperBoundBytes).toBe(
      ramExecution.workingSet.totalRoutedExpertBytes +
        ramExecution.requirements.boundedPinnedStagingReserveBytes +
        ramExecution.requirements.residentStreamingTransientBytes,
    );
    expect(ramExecution.requirements.hostRamBytes).toBe(
      ramExecution.requirements.hostRamPeakUpperBoundBytes,
    );
    expect(() => compilePythonLaunchDescription(manifest, launchOptions)).toThrow(
      `python_ram_backed_moe_stage_binding_is_missing:${ramStage.stageId}`,
    );
    const launch = compilePythonLaunchDescription(
      manifest,
      ramBackedLaunchOptions(manifest),
    );
    const ramProcess = launch.launchOrder.find(
      (process) => process.stageId === ramStage.stageId,
    )!;
    expect(ramProcess.command.args).toContain(
      "--ram-moe-resident-parameter-budget-bytes",
    );
    expect(ramProcess.command.args).toContain("--ram-moe-total-routed-expert-bytes");
    expect(ramProcess.command.args).toContain("--ram-moe-largest-expert-bytes");
    expect(ramProcess.command.args).toContain(
      "--ram-moe-resident-streaming-transient-bytes",
    );
    expect(ramProcess.command.args).toContain(
      "--ram-moe-bounded-pinned-staging-reserve-bytes",
    );
    expect(ramProcess.command.args).toContain(
      "--ram-moe-host-ram-peak-upper-bound-bytes",
    );
    expect(ramProcess.command.args).not.toContain("--model-artifact-identity");

    const tampered = structuredClone(manifest);
    const contract = tampered.plans.decode.stages[0]!.macroWave!;
    contract.requirements.vramBytes = contract.budgets.vramBytes + 1;
    expect(() => validateRuntimePipelineManifest(tampered)).toThrow(
      "runtime_macro_wave_vram_budget_exceeded",
    );

    const wrongSchema = structuredClone(manifest);
    (wrongSchema.plans.prefill.stages[0]!.macroWave as { schema: string }).schema =
      "gdlp-macro-wave-stage/2";
    expect(() => validateRuntimePipelineManifest(wrongSchema)).toThrow(
      "runtime_macro_wave_stage_schema_is_not_supported",
    );
  });

  it("binds RAM artifacts and byte-bounded cache fail-closed before runner release", () => {
    const manifest = buildRuntimePipelineManifest(request(40 * MIB));
    const stage = manifest.plans.prefill.stages.find(
      (candidate) => candidate.macroWave?.memoryMode === "ram-backed",
    )!;
    const options = ramBackedLaunchOptions(manifest);

    const relative = structuredClone(options);
    relative.ramBackedMoeStages![stage.stageId]!.snapshotPath = "org/model";
    expect(() => compilePythonLaunchDescription(manifest, relative)).toThrow(
      `python_ram_backed_moe_snapshot_path_is_invalid:${stage.stageId}`,
    );

    const wrongIdentity = structuredClone(options);
    wrongIdentity.ramBackedMoeStages![stage.stageId]!.artifactIdentity =
      `sha256:fedcba9876543210${"0".repeat(48)}`;
    expect(() => compilePythonLaunchDescription(manifest, wrongIdentity)).toThrow(
      `python_ram_backed_moe_pipeline_identity_mismatch:${stage.stageId}`,
    );

    const wrongRange = structuredClone(options);
    wrongRange.ramBackedMoeStages![stage.stageId]!.layerEnd -= 1;
    expect(() => compilePythonLaunchDescription(manifest, wrongRange)).toThrow(
      `python_ram_backed_moe_stage_range_mismatch:${stage.stageId}`,
    );

    const wrongCapacity = structuredClone(options);
    wrongCapacity.ramBackedMoeStages![stage.stageId]!.cache.capacityBytes += 1;
    expect(() => compilePythonLaunchDescription(manifest, wrongCapacity)).toThrow(
      `python_ram_backed_moe_cache_budget_mismatch:${stage.stageId}`,
    );

    const wrongResidentBudget = structuredClone(options);
    wrongResidentBudget.ramBackedMoeStages![stage.stageId]!
      .residentParameterBudgetBytes =
      stage.macroWave!.requirements.residentParameterBudgetBytes + 1;
    expect(() =>
      compilePythonLaunchDescription(manifest, wrongResidentBudget),
    ).toThrow(
      `python_ram_backed_moe_resident_parameter_budget_mismatch:${stage.stageId}`,
    );

    const cpuFallback = structuredClone(options) as unknown as {
      ramBackedMoeStages: Record<string, { allowCpuFallback: boolean }>;
    };
    cpuFallback.ramBackedMoeStages[stage.stageId]!.allowCpuFallback = true;
    expect(() =>
      compilePythonLaunchDescription(
        manifest,
        cpuFallback as unknown as PythonLaunchCompilerOptions,
      ),
    ).toThrow(`python_ram_backed_moe_cpu_fallback_is_forbidden:${stage.stageId}`);
  });

  it("rejects missing executable backend declarations and RAM artifact tampering", () => {
    const manifest = buildRuntimePipelineManifest(request(40 * MIB));
    const stage = manifest.plans.prefill.stages.find(
      (candidate) => candidate.macroWave?.memoryMode === "ram-backed",
    )!;

    const missingCapability = structuredClone(manifest);
    missingCapability.plans.prefill.stages[stage.index]!.members[0]!.capabilities.features =
      missingCapability.plans.prefill.stages[stage.index]!.members[0]!.capabilities.features
        .filter((feature) => feature !== "no-full-model-materialization");
    expect(() => validateRuntimePipelineManifest(missingCapability)).toThrow(
      "runtime_macro_wave_ram_backend_is_not_declared",
    );

    const mismatchedTile = structuredClone(manifest);
    mismatchedTile.plans.decode.stages[stage.index]!.macroWave!.ramArtifact!
      .largestExpertBytes += 1;
    expect(() => validateRuntimePipelineManifest(mismatchedTile)).toThrow(
      "runtime_macro_wave_ram_artifact_budget_mismatch",
    );
  });

  it("rejects speculation that can exceed the sealed one-position wave", () => {
    const input = request(80 * MIB);
    input.speculation = {
      mode: "adaptive",
      controller: "acceptance-adaptive",
      defaultStrategyId: "ngram-5",
      fallbackStrategyId: "autoregressive",
      acceptanceWindowTokens: 32,
      strategies: [
        {
          id: "ngram-5",
          kind: "ngram",
          maxDraftTokens: 5,
          minAcceptanceRate: 0.5,
          maxWasteRatio: 0.5,
          priority: 1,
        },
        {
          id: "autoregressive",
          kind: "autoregressive",
          maxDraftTokens: 1,
          minAcceptanceRate: 1,
          maxWasteRatio: 0,
          priority: 0,
        },
      ],
    };
    expect(() => buildRuntimePipelineManifest(input)).toThrow(
      "runtime_macro_wave_speculation_exceeds_wave",
    );
  });

  it("launches a multi-position sealed wave only when the ngram VERIFY width fits", () => {
    const input = request(80 * MIB);
    if (input.planner?.kind !== "macro-wave") throw new Error("missing fixture planner");
    input.planner.options = {
      ...input.planner.options,
      waveTokens: 6,
      expectedCommittedTokensPerWave: 4,
    };
    input.speculation = {
      mode: "adaptive",
      controller: "acceptance-adaptive",
      defaultStrategyId: "ngram-5",
      fallbackStrategyId: "autoregressive",
      acceptanceWindowTokens: 32,
      strategies: [
        {
          id: "ngram-5",
          kind: "ngram",
          maxDraftTokens: 5,
          minAcceptanceRate: 0.5,
          maxWasteRatio: 0.5,
          priority: 1,
        },
        {
          id: "autoregressive",
          kind: "autoregressive",
          maxDraftTokens: 1,
          minAcceptanceRate: 1,
          maxWasteRatio: 0,
          priority: 0,
        },
      ],
    };
    const manifest = buildRuntimePipelineManifest(input);
    const launch = compilePythonLaunchDescription(
      manifest,
      ramBackedLaunchOptions(manifest),
    );

    for (const process of launch.launchOrder.filter(
      (candidate) => candidate.kind !== "cell-member",
    )) {
      const waveIndex = process.command.args.indexOf("--sealed-wave-tokens");
      const prefillIndex = process.command.args.indexOf(
        "--max-prefill-chunk-tokens",
      );
      expect(process.command.args[waveIndex + 1]).toBe("6");
      expect(process.command.args[prefillIndex + 1]).toBe("8");
    }
  });
});

function argumentValue(args: string[], name: string): string | undefined {
  const index = args.indexOf(name);
  return index < 0 ? undefined : args[index + 1];
}
