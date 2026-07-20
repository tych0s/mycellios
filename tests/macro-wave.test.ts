import { describe, expect, it } from "vitest";
import { MacroWaveRamVramPlanner } from "../src/distribution/planners.js";
import type {
  ComputeNodeProfile,
  DistributedModelProfile,
  DistributionTopology,
  DistributionWorkload,
} from "../src/distribution/types.js";

const MIB = 1024 ** 2;
const GIB = 1024 ** 3;

function model(
  layers: number,
  weightBytes: number,
  macroWave?: {
    activeWeightBytesPerWave?: number;
    largestTransferUnitBytes: number;
    expertWorkspaceBytesPerPosition?: number;
  },
  expertWeightBytes?: number,
  geometry?: { expertCount: number; expertsPerToken: number },
): DistributedModelProfile {
  return {
    id: "macro-wave-fixture",
    layers: Array.from({ length: layers }, (_, index) => ({
      index,
      weightBytes,
      activationElements: 4_096,
      kvBytesPerToken: 256,
      decodeMsAtUnit: 2,
      prefillMsPerTokenAtUnit: 0.05,
      ...(macroWave
        ? {
            macroWave: {
              expertWorkspaceBytesPerPosition: 256,
              ...macroWave,
            },
          }
        : {}),
      ...(expertWeightBytes !== undefined
        ? { expertParallel: { expertWeightBytes, ...geometry } }
        : {}),
    })),
    embeddingBytes: 0,
    lmHeadBytes: 0,
    runtimeOverheadBytesPerStage: 128 * MIB,
    embeddingDecodeMsAtUnit: 0.1,
    lmHeadDecodeMsAtUnit: 0.2,
    embeddingPrefillMsPerTokenAtUnit: 0.001,
    lmHeadPrefillMsPerTokenAtUnit: 0.002,
  };
}

function node(
  id: string,
  overrides: Partial<NonNullable<ComputeNodeProfile["ramVram"]>> = {},
): ComputeNodeProfile {
  return {
    id,
    region: "test-cell",
    memoryBytes: 4 * GIB,
    reserveBytes: 0,
    decodeScale: 1,
    prefillScale: 1,
    codecScale: 1,
    batchGain: 0.5,
    maxBatchSpeedup: 2,
    powerWatts: 100,
    availability: 0.999,
    ramVram: {
      usableRamBytes: 32 * GIB,
      usableVramBytes: 3_500 * MIB,
      ramBandwidthGBps: 12.5,
      pcieBandwidthGBps: 8,
      ...overrides,
    },
  };
}

function fullMesh(nodes: ComputeNodeProfile[], oneWayLatencyMs: number): DistributionTopology {
  return {
    nodes,
    links: nodes.flatMap((from) =>
      nodes
        .filter((to) => to.id !== from.id)
        .map((to) => ({
          from: from.id,
          to: to.id,
          oneWayLatencyMs,
          jitterP95Ms: 0,
          bandwidthMbps: 1_000,
          lossRate: 0,
          availability: 0.999,
        })),
    ),
  };
}

function workload(maxStages: number): DistributionWorkload {
  return {
    promptTokens: 64,
    outputTokens: 16,
    contextTokens: 128,
    concurrentSequences: 1,
    maxStages,
    maxQualityLoss: 0,
    minRouteAvailability: 0.9,
    batchWindowMs: 0,
    p95: false,
  };
}

describe("MacroWave RAM+VRAM planner", () => {
  it("uses complete header tensor telemetry and falls back safely when it is partial", () => {
    const planner = new MacroWaveRamVramPlanner({
      candidateMicroBatchSizes: [1],
      candidatePrefillChunks: [32],
    });
    const exact = model(
      1,
      3 * GIB,
      {
        activeWeightBytesPerWave: 256 * MIB,
        largestTransferUnitBytes: 256 * MIB,
      },
      2 * GIB,
      { expertCount: 8, expertsPerToken: 1 },
    );
    exact.layers[0]!.largestResidentTensorBytes = 64 * MIB;
    exact.largestEmbeddingTensorBytes = 0;
    exact.largestLmHeadTensorBytes = 0;
    const topology = fullMesh(
      [node("exact", { usableVramBytes: 2 * GIB })],
      0,
    );

    const exactCost = planner.evaluate(exact, topology, workload(1)).selected!
      .stages[0]!.ramBacked;
    expect(exactCost.residentParameterBudgetBytes).toBe(1 * GIB);
    expect(exactCost.residentStreamingTransientBytes).toBe(64 * MIB);
    expect(exactCost.boundedPinnedStagingReserveBytes).toBe(512 * MIB);
    expect(exactCost.hostRamPeakUpperBoundBytes).toBe(
      2 * GIB + 512 * MIB + 64 * MIB,
    );

    const partial = structuredClone(exact);
    delete partial.layers[0]!.largestResidentTensorBytes;
    const fallback = planner.evaluate(partial, topology, workload(1)).selected!
      .stages[0]!.ramBacked;
    expect(fallback.residentStreamingTransientBytes).toBe(1 * GIB);
    expect(fallback.hostRamPeakUpperBoundBytes).toBe(
      2 * GIB + 512 * MIB + 1 * GIB,
    );

    const missingEndpoint = structuredClone(exact);
    delete missingEndpoint.largestEmbeddingTensorBytes;
    const endpointFallback = planner.evaluate(
      missingEndpoint,
      topology,
      workload(1),
    ).selected!.stages[0]!.ramBacked;
    expect(endpointFallback.residentStreamingTransientBytes).toBe(1 * GIB);
  });

  it("builds contiguous RAM-backed stages when a resident route cannot cover the model", () => {
    const planner = new MacroWaveRamVramPlanner({
      candidateMicroBatchSizes: [1],
      candidatePrefillChunks: [32],
      waveTokens: 4,
      expectedCommittedTokensPerWave: 3,
      verificationScalePerExtraToken: 0.5,
    });
    const topology = fullMesh([node("a"), node("b"), node("c")], 10);
    const result = planner.evaluate(
      model(6, 2 * GIB, {
        activeWeightBytesPerWave: 384 * MIB,
        largestTransferUnitBytes: 384 * MIB,
      }, 1_536 * MIB),
      topology,
      workload(2),
    );

    expect(result.feasible).toBe(true);
    expect(result.selected?.kind).toBe("macro-wave");
    expect(result.alternatives.resident.feasible).toBe(false);
    expect(result.plan?.stages).toHaveLength(2);
    expect(result.plan?.stages[0]?.layerStart).toBe(0);
    expect(result.plan?.stages.at(-1)?.layerEnd).toBe(6);
    expect(result.plan?.stages[1]?.layerStart).toBe(result.plan?.stages[0]?.layerEnd);
    expect(result.selected?.stages.some((stage) => stage.selectedMode === "ram-backed")).toBe(
      true,
    );
    const streamed = result.selected!.stages.find((stage) => stage.selectedMode === "ram-backed")!;
    expect(streamed.resident.reason).toBe("resident_vram_capacity_exceeded");
    expect(streamed.ramBacked.feasible).toBe(true);
    expect(streamed.ramBacked.ramReadMsPerWave).toBeGreaterThan(0);
    expect(streamed.ramBacked.pcieTransferMsPerWave).toBeGreaterThan(0);
    expect(streamed.outgoing.rttMs).toBeGreaterThanOrEqual(0);
  });

  it("shards only profiled expert weights while keeping replicated weights resident", () => {
    const planner = new MacroWaveRamVramPlanner({
      candidateMicroBatchSizes: [1],
      candidatePrefillChunks: [64],
    });
    const profiledModel = model(1, 3 * GIB, undefined, 2 * GIB);
    const fullLayer = planner.evaluate(
      profiledModel,
      fullMesh([node("layer", { usableVramBytes: 4 * GIB, residentKind: "layers" })], 0),
      workload(1),
    );
    const expertParallel = planner.evaluate(
      profiledModel,
      fullMesh(
        [
          node("ep", {
            residentKind: "expert-shard",
            usableVramBytes: 2 * GIB,
            expertShard: { worldSize: 4, fraction: 0.25 },
          }),
        ],
        0,
      ),
      workload(1),
    );
    const unshardedTight = planner.evaluate(
      profiledModel,
      fullMesh([node("tight", { usableVramBytes: 2 * GIB, residentKind: "layers" })], 0),
      workload(1),
    );

    const fullCost = fullLayer.selected!.stages[0]!.resident;
    const shardCost = expertParallel.selected!.stages[0]!.resident;
    expect(expertParallel.selected?.kind).toBe("resident-baseline");
    expect(expertParallel.selected?.stages[0]?.selectedMode).toBe("ep-resident");
    expect(shardCost.shardableExpertWeightBytes).toBe(2 * GIB);
    expect(shardCost.replicatedWeightBytes).toBe(1 * GIB);
    expect(shardCost.residentWeightBytes).toBe(1.5 * GIB);
    expect(shardCost.vramRequiredBytes).toBeLessThan(fullCost.vramRequiredBytes);
    expect(unshardedTight.alternatives.resident.feasible).toBe(false);
    expect(expertParallel.selected?.ramWeightBytesPerOutputToken).toBe(0);
  });

  it("uses measured RTT to choose one RAM-backed macro-stage over two resident stages", () => {
    const planner = new MacroWaveRamVramPlanner({
      candidateMicroBatchSizes: [1],
      candidatePrefillChunks: [64],
      waveTokens: 4,
      expectedCommittedTokensPerWave: 4,
      verificationScalePerExtraToken: 0.25,
    });
    const nodes = [
      node("a", { usableVramBytes: 2_500 * MIB }),
      node("b", { usableVramBytes: 2_500 * MIB }),
    ];
    const routedModel = model(2, 2 * GIB, {
      activeWeightBytesPerWave: 256 * MIB,
      largestTransferUnitBytes: 128 * MIB,
    }, 1_536 * MIB, { expertCount: 12, expertsPerToken: 2 });

    const lowRtt = planner.evaluate(routedModel, fullMesh(nodes, 1), workload(2));
    const highRtt = planner.evaluate(routedModel, fullMesh(nodes, 5_000), workload(2));

    expect(lowRtt.selected?.kind).toBe("resident-baseline");
    expect(lowRtt.selected?.stages).toHaveLength(2);
    expect(highRtt.selected?.kind).toBe("macro-wave");
    expect(highRtt.selected?.stages).toHaveLength(1);
    expect(highRtt.alternatives.resident.stages[0]?.outgoing.rttMs).toBe(10_000);
    expect(highRtt.selected!.tpotMs).toBeLessThan(highRtt.alternatives.resident.tpotMs);
  });

  it("prices PCIe and RAM bandwidth instead of treating host-backed weights as free", () => {
    const planner = new MacroWaveRamVramPlanner({
      candidateMicroBatchSizes: [1],
      candidatePrefillChunks: [64],
      waveTokens: 2,
      expectedCommittedTokensPerWave: 2,
    });
    const routedModel = model(2, 2 * GIB, {
      activeWeightBytesPerWave: 256 * MIB,
      largestTransferUnitBytes: 128 * MIB,
    }, 1_536 * MIB, { expertCount: 12, expertsPerToken: 2 });
    const fast = planner.evaluate(
      routedModel,
      fullMesh([node("fast", { pcieBandwidthGBps: 8, usableVramBytes: 1_792 * MIB })], 0),
      workload(1),
    );
    const slow = planner.evaluate(
      routedModel,
      fullMesh([node("slow", { pcieBandwidthGBps: 0.5, usableVramBytes: 1_792 * MIB })], 0),
      workload(1),
    );

    expect(fast.selected?.kind).toBe("macro-wave");
    expect(slow.selected?.kind).toBe("macro-wave");
    expect(slow.selected!.stages[0]!.ramBacked.pcieTransferMsPerWave).toBeGreaterThan(
      fast.selected!.stages[0]!.ramBacked.pcieTransferMsPerWave,
    );
    expect(slow.selected!.tpotMs).toBeGreaterThan(fast.selected!.tpotMs);
  });

  it("fails closed on EP labels without per-layer expert telemetry", () => {
    const planner = new MacroWaveRamVramPlanner({
      candidateMicroBatchSizes: [1],
      candidatePrefillChunks: [32],
      weightBufferCopies: 2,
    });
    const result = planner.evaluate(
      model(1, 512 * MIB, {
        activeWeightBytesPerWave: 64 * MIB,
        largestTransferUnitBytes: 64 * MIB,
      }),
      fullMesh(
        [
          node("ep-without-layer-telemetry", {
            residentKind: "expert-shard",
            expertShard: { worldSize: 4, fraction: 0.25 },
          }),
        ],
        0,
      ),
      workload(1),
    );

    expect(result.feasible).toBe(false);
    expect(result.alternatives.resident.feasible).toBe(false);
    expect(result.alternatives.resident.rejectionBreakdown).toContainEqual({
      reason: "missing_expert_weight_telemetry:ep-without-layer-telemetry",
      count: 1,
    });
    expect(result.alternatives.macroWave.rejectionBreakdown).toContainEqual({
      reason: "missing_ram_backed_expert_telemetry:ep-without-layer-telemetry",
      count: 1,
    });

    const missingCellTelemetry = planner.evaluate(
      model(1, 512 * MIB, undefined, 384 * MIB),
      fullMesh([node("ep-without-cell-telemetry", { residentKind: "expert-shard" })], 0),
      workload(1),
    );
    expect(missingCellTelemetry.feasible).toBe(false);
    expect(missingCellTelemetry.alternatives.resident.rejectionBreakdown).toContainEqual({
      reason: "missing_expert_shard_profile:ep-without-cell-telemetry",
      count: 1,
    });
  });

  it("reserves explicit compute and prefetch weight buffers in VRAM", () => {
    const routedModel = model(1, 4 * GIB, {
      activeWeightBytesPerWave: 512 * MIB,
      largestTransferUnitBytes: 512 * MIB,
    }, 4 * GIB);
    const topology = fullMesh(
      [node("buffer-tight", { usableVramBytes: 900 * MIB })],
      0,
    );
    expect(
      () =>
        new MacroWaveRamVramPlanner({
          candidateMicroBatchSizes: [1],
          candidatePrefillChunks: [32],
          weightBufferCopies: 1,
        }),
    ).toThrow(
      "weightBufferCopies must be 2 for serial-exact ping-pong execution",
    );
    const doubleBuffer = new MacroWaveRamVramPlanner({
      candidateMicroBatchSizes: [1],
      candidatePrefillChunks: [32],
    }).evaluate(routedModel, topology, workload(1));

    expect(doubleBuffer.feasible).toBe(false);
    expect(doubleBuffer.alternatives.macroWave.rejectionBreakdown).toContainEqual({
      reason: "ram_backed_weight_buffers_vram_exceeded:buffer-tight",
      count: 1,
    });
  });

  it("fails closed when a routed RAM stage has no sealed expert workspace", () => {
    const routed = model(
      1,
      2 * GIB,
      { largestTransferUnitBytes: 128 * MIB },
      1_536 * MIB,
      { expertCount: 12, expertsPerToken: 2 },
    );
    delete routed.layers[0]!.macroWave!.expertWorkspaceBytesPerPosition;
    const result = new MacroWaveRamVramPlanner({
      candidateMicroBatchSizes: [1],
      candidatePrefillChunks: [32],
    }).evaluate(
      routed,
      fullMesh([node("workspace-missing", { usableVramBytes: 1 * GIB })], 0),
      workload(1),
    );

    expect(result.alternatives.macroWave.feasible).toBe(false);
    expect(result.alternatives.macroWave.rejectionBreakdown).toContainEqual({
      reason: "missing_ram_backed_expert_workspace:workspace-missing",
      count: 1,
    });
  });

  it("moves the WAN RTT crossover only from an explicit expected cache hit rate", () => {
    const nodes = [
      node("a", { usableVramBytes: 2_500 * MIB }),
      node("b", { usableVramBytes: 2_500 * MIB }),
    ];
    const routedModel = model(2, 2 * GIB, {
      activeWeightBytesPerWave: 256 * MIB,
      largestTransferUnitBytes: 128 * MIB,
    }, 1_536 * MIB, { expertCount: 12, expertsPerToken: 2 });
    const candidateRtts = [0, 2, 5, 10, 20, 40, 80, 160, 320, 640, 1_280, 2_560];
    const crossover = (expectedWeightCacheHitRate: number): number | undefined =>
      candidateRtts.find((oneWayLatencyMs) => {
        const result = new MacroWaveRamVramPlanner({
          candidateMicroBatchSizes: [1],
          candidatePrefillChunks: [64],
          waveTokens: 4,
          expectedCommittedTokensPerWave: 4,
          verificationScalePerExtraToken: 0.25,
          expectedWeightCacheHitRate,
        }).evaluate(routedModel, fullMesh(nodes, oneWayLatencyMs), workload(2));
        return result.selected?.kind === "macro-wave";
      });

    const coldCrossover = crossover(0);
    const warmCrossover = crossover(0.75);
    expect(coldCrossover).toBeDefined();
    expect(warmCrossover).toBeDefined();
    expect(warmCrossover!).toBeLessThan(coldCrossover!);

    const measured = new MacroWaveRamVramPlanner({
      candidateMicroBatchSizes: [1],
      candidatePrefillChunks: [64],
      expectedWeightCacheHitRate: 0.75,
    }).evaluate(routedModel, fullMesh([nodes[0]!], 0), workload(1));
    const mode = measured.selected!.stages[0]!.ramBacked;
    expect(mode.expectedWeightCacheHitRate).toBe(0.75);
    expect(mode.expectedCacheMissWeightBytesPerWave).toBe(
      mode.activeWeightBytesPerWave * 0.25,
    );
  });

  it("fails closed with capacity reasons when neither resident nor RAM-backed execution fits", () => {
    const planner = new MacroWaveRamVramPlanner({
      candidateMicroBatchSizes: [1],
      candidatePrefillChunks: [32],
    });
    const tooSmall = node("tiny", {
      usableRamBytes: 1 * GIB,
      usableVramBytes: 512 * MIB,
    });
    const result = planner.evaluate(
      model(
        1,
        4 * GIB,
        {
          activeWeightBytesPerWave: 512 * MIB,
          largestTransferUnitBytes: 512 * MIB,
        },
        3_584 * MIB,
      ),
      fullMesh([tooSmall], 0),
      workload(1),
    );

    expect(result.feasible).toBe(false);
    expect(result.plan).toBeNull();
    expect(result.reason).toContain("no_contiguous_route");
    expect(
      result.alternatives.macroWave.rejectionBreakdown.some((entry) =>
        entry.reason.startsWith("resident_vram_capacity_exceeded:tiny"),
      ),
    ).toBe(true);
    expect(
      result.alternatives.macroWave.rejectionBreakdown.some((entry) =>
        entry.reason.startsWith("ram_capacity_exceeded:tiny"),
      ),
    ).toBe(true);
  });

  it("derives the sealed decode union, reserves the larger prefill buffer, and prices cold prefill", () => {
    const planner = new MacroWaveRamVramPlanner({
      candidateMicroBatchSizes: [1],
      candidatePrefillChunks: [64],
      waveTokens: 4,
      expectedCommittedTokensPerWave: 3,
    });
    const routedModel = model(
      1,
      2 * GIB,
      { largestTransferUnitBytes: 128 * MIB },
      1_536 * MIB,
      { expertCount: 12, expertsPerToken: 2 },
    );
    const result = planner.evaluate(
      routedModel,
      fullMesh([node("geometry", { usableVramBytes: 1 * GIB })], 0),
      workload(1),
    );

    expect(result.selected?.kind).toBe("macro-wave");
    const mode = result.selected!.stages[0]!.ramBacked;
    expect(mode.activeWeightBytesPerWave).toBe(8 * 128 * MIB);
    expect(mode.prefillActiveWeightBytesPerChunk).toBe(1_536 * MIB);
    expect(mode.prefillLoadMsPerChunk).toBeGreaterThan(mode.loadMsPerWave);
    expect(mode.expertWorkspaceBytes).toBe(256 * 64);
    expect(mode.activationBufferBytes).toBe(4_096 * 2 * 64 * 2 + 256 * 64);
  });

  it("falls back to every routed expert for legacy working-set telemetry", () => {
    const result = new MacroWaveRamVramPlanner({
      candidateMicroBatchSizes: [1],
      candidatePrefillChunks: [32],
      waveTokens: 4,
    }).evaluate(
      model(
        1,
        2 * GIB,
        {
          activeWeightBytesPerWave: 128 * MIB,
          largestTransferUnitBytes: 128 * MIB,
        },
        1_536 * MIB,
      ),
      fullMesh([node("legacy", { usableVramBytes: 1 * GIB })], 0),
      workload(1),
    );

    expect(result.selected!.stages[0]!.ramBacked.activeWeightBytesPerWave).toBe(
      1_536 * MIB,
    );
  });

  it("accepts an explicitly dense prefix inside a mixed RAM-backed stage", () => {
    const mixed = model(2, 2 * GIB);
    mixed.layers[0]!.expertParallel = { expertWeightBytes: 0 };
    mixed.layers[1]!.expertParallel = {
      expertWeightBytes: 1_536 * MIB,
      expertCount: 12,
      expertsPerToken: 2,
    };
    mixed.layers[1]!.macroWave = {
      largestTransferUnitBytes: 128 * MIB,
      expertWorkspaceBytesPerPosition: 256,
    };
    const result = new MacroWaveRamVramPlanner({
      candidateMicroBatchSizes: [1],
      candidatePrefillChunks: [32],
      waveTokens: 4,
    }).evaluate(
      mixed,
      fullMesh([node("mixed", { usableVramBytes: 3 * GIB })], 0),
      workload(1),
    );

    expect(result.selected?.kind).toBe("macro-wave");
    expect(result.selected?.stages[0]?.selectedMode).toBe("ram-backed");
    expect(result.alternatives.macroWave.rejectionBreakdown).not.toContainEqual(
      expect.objectContaining({ reason: expect.stringContaining("missing_ram_backed") }),
    );
  });

  it("rejects nodes that do not publish the complete RAM/VRAM hierarchy", () => {
    const { ramVram: _omitted, ...missing } = node("missing");
    const result = new MacroWaveRamVramPlanner({
      candidateMicroBatchSizes: [1],
      candidatePrefillChunks: [32],
    }).evaluate(model(1, 128 * MIB), fullMesh([missing], 0), workload(1));

    expect(result.feasible).toBe(false);
    expect(result.alternatives.resident.reason).toBe("no_profiled_ram_vram_nodes");
    expect(result.alternatives.resident.rejectionBreakdown).toContainEqual({
      reason: "missing_ram_vram_profile:missing",
      count: 1,
    });
  });
});
