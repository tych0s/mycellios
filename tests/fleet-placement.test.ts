import { describe, expect, it } from "vitest";
import { evaluateDistributionPlan } from "../src/distribution/cost-model.js";
import {
  comparePlannerWithExhaustiveOracle,
  planNativeReplicaChains,
  requirePlannerWithinOracleTolerance,
  selectNativeChain,
  engineNodeIsEligible,
  type NativeCompleteChain,
  type NativeFleetNodeState,
} from "../src/distribution/fleet-placement.js";
import {
  DEFAULT_SEARCH_OPTIONS,
} from "../src/distribution/planners.js";
import type {
  ComputeNodeProfile,
  DistributedModelProfile,
  DistributionPlan,
  DistributionTopology,
  DistributionWorkload,
  SearchOptions,
} from "../src/distribution/types.js";

describe("native two-phase fleet placement", () => {
  it("forms several short complete chains instead of one long 20-node chain", () => {
    const topology = completeTopology(20);
    const placement = planNativeReplicaChains(
      model(4),
      topology,
      workload(2),
      states(topology),
      { workloadClass: "interactive", sessionId: null, kvMissPenaltyMs: 0 },
      { desiredReplicas: 4, searchOptions: narrowSearch() },
    );

    expect(placement.complete).toBe(true);
    expect(placement.chains).toHaveLength(4);
    expect(placement.chains.every((chain) => chain.plan.stages.length === 1)).toBe(true);
    expect(new Set(placement.chains.flatMap((chain) => chain.nodeIds)).size).toBe(4);
    expect(new Set(placement.chains.flatMap((chain) => chain.failureDomainIds)).size).toBe(4);
  });

  it("never publishes a requested replica when independent failure domains run out", () => {
    const topology = completeTopology(4);
    const nodeStates = states(topology).map((state) => ({
      ...state,
      failureDomainId: "one-home",
    }));
    const placement = planNativeReplicaChains(
      model(4),
      topology,
      workload(2),
      nodeStates,
      { workloadClass: "interactive", sessionId: null, kvMissPenaltyMs: 0 },
      { desiredReplicas: 2, searchOptions: narrowSearch() },
    );

    expect(placement.complete).toBe(false);
    expect(placement.chains).toHaveLength(1);
    expect(placement.reason).toBe("independent_complete_chains_unavailable:1:2");
  });

  it("charges physical boundaries, not the raw number of local stages", () => {
    const topology = completeTopology(4);
    const nodeStates = states(topology).map((state, index) => ({
      ...state,
      physicalHostId: index < 2 ? "host-a" : "host-b",
    }));
    const stateByNode = new Map(nodeStates.map((state) => [state.nodeId, state]));
    const plan: DistributionPlan = {
      algorithm: "test",
      codec: "fp16",
      microBatchSize: 1,
      prefillChunkTokens: 16,
      stages: [
        { nodeId: "node-0", layerStart: 0, layerEnd: 1 },
        { nodeId: "node-1", layerStart: 1, layerEnd: 2 },
        { nodeId: "node-2", layerStart: 2, layerEnd: 3 },
        { nodeId: "node-3", layerStart: 3, layerEnd: 4 },
      ],
    };
    const profile = model(4);
    const activeWorkload = workload(4);
    const metrics = evaluateDistributionPlan(profile, topology, activeWorkload, plan);
    const placement = planNativeReplicaChains(
      profile,
      topology,
      activeWorkload,
      nodeStates,
      { workloadClass: "interactive", sessionId: null, kvMissPenaltyMs: 0 },
      { desiredReplicas: 1, searchOptions: narrowSearch() },
    );

    expect(metrics.feasible).toBe(true);
    // The public planner may choose fewer stages; score the explicit four-stage
    // chain through the selector to exercise the same production scorer.
    const explicit: NativeCompleteChain = {
      chainId: "explicit",
      plan,
      metrics,
      score: placement.chains[0]!.score,
      nodeIds: plan.stages.map((stage) => stage.nodeId),
      failureDomainIds: ["domain-0", "domain-1", "domain-2", "domain-3"],
    };
    const selected = selectNativeChain(
      [explicit],
      { workloadClass: "interactive", sessionId: null, kvMissPenaltyMs: 0 },
      topology,
      [...stateByNode.values()],
    );
    expect(selected?.score.physicalBoundaryCount).toBe(2);
  });

  it("removes a WAN loop boundary by preferring a feasible local full-model node", () => {
    const topology = completeTopology(3);
    topology.nodes = topology.nodes.map((node, index) => ({
      ...node,
      maxStageLayers: index === 0 ? 4 : 2,
      decodeScale: index === 0 ? 1 : 0.1,
      prefillScale: index === 0 ? 1 : 0.1,
    }));
    topology.links = topology.links.map((link) => ({
      ...link,
      oneWayLatencyMs: 100,
      jitterP95Ms: 10,
    }));
    const placement = planNativeReplicaChains(
      model(4), topology, workload(3), states(topology),
      { workloadClass: "interactive", sessionId: null, kvMissPenaltyMs: 0 },
      { desiredReplicas: 1, searchOptions: narrowSearch() },
    );

    expect(placement.complete).toBe(true);
    expect(placement.chains[0]!.nodeIds).toEqual(["node-0"]);
    expect(placement.chains[0]!.score.physicalBoundaryCount).toBe(0);
  });

  it("uses a fat node only when its saved hops outweigh measured compute cost", () => {
    const topology = completeTopology(3);
    topology.nodes = topology.nodes.map((node, index) => ({
      ...node,
      maxStageLayers: index === 0 ? 4 : 2,
      decodeScale: index === 0 ? 20 : 0.1,
      prefillScale: index === 0 ? 20 : 0.1,
    }));
    topology.links = topology.links.map((link) => ({
      ...link,
      oneWayLatencyMs: 0.1,
      jitterP95Ms: 0,
    }));
    const placement = planNativeReplicaChains(
      model(4), topology, workload(3), states(topology),
      { workloadClass: "interactive", sessionId: null, kvMissPenaltyMs: 0 },
      { desiredReplicas: 1, searchOptions: narrowSearch() },
    );

    expect(placement.complete).toBe(true);
    expect(placement.chains[0]!.nodeIds).toEqual(["node-1", "node-2"]);
    expect(placement.chains[0]!.score.physicalBoundaryCount).toBe(2);
  });

  it("uses path sums for interactive routing and the bottleneck for throughput", () => {
    const topology = completeTopology(4);
    const baseStates = states(topology);
    const adjusted = baseStates.map((state) => ({
      ...state,
      observedP95ServiceMs:
        state.nodeId === "node-0" || state.nodeId === "node-1"
          ? 20
          : state.nodeId === "node-2"
            ? 5
            : 30,
    }));
    const profile = model(4);
    const activeWorkload = workload(2);
    const planA = twoStagePlan("node-0", "node-1");
    const planB = twoStagePlan("node-2", "node-3");
    const chains = [
      chain("a", planA, evaluateDistributionPlan(profile, topology, activeWorkload, planA)),
      chain("b", planB, evaluateDistributionPlan(profile, topology, activeWorkload, planB)),
    ];

    expect(selectNativeChain(
      chains,
      { workloadClass: "interactive", sessionId: null, kvMissPenaltyMs: 0 },
      topology,
      adjusted,
    )?.chainId).toBe("b");
    expect(selectNativeChain(
      chains,
      { workloadClass: "throughput", sessionId: null, kvMissPenaltyMs: 0 },
      topology,
      adjusted,
    )?.chainId).toBe("a");
  });

  it("keeps the fast planner inside an exhaustive small-instance oracle gate", () => {
    const result = comparePlannerWithExhaustiveOracle(
      model(4),
      completeTopology(3),
      workload(3),
      narrowSearch(),
      0.01,
    );
    expect(result.withinTolerance).toBe(true);
    expect(result.relativeGap).toBeLessThanOrEqual(0.01);
    expect(() => requirePlannerWithinOracleTolerance(result)).not.toThrow();
    expect(() => requirePlannerWithinOracleTolerance({
      ...result,
      withinTolerance: false,
      relativeGap: 0.25,
    })).toThrow("native_planner_oracle_regression:0.250000");
  });

  it("fails closed on stale or incompatible engine capability evidence", () => {
    const state = states(completeTopology(1))[0]!;
    const descriptorDigest = `sha256:${"a".repeat(64)}` as const;
    const certificationId = `sha256:${"d".repeat(64)}` as const;
    const artifactManifestDigest = `sha256:${"e".repeat(64)}` as const;
    const capability = {
      profileId: `sha256:${"f".repeat(64)}` as const,
      descriptorDigest,
      certificationId,
      artifactManifestDigest,
      backend: "cuda",
      runtimeAbi: "cuda-12",
      quantizations: ["bf16"],
      maxContextTokens: 32_768,
      maxLayerCount: 16,
      kvBytesPerToken: 2_048,
      maxKvTokens: 32_768,
      decodeScale: 0.5,
      prefillScale: 0.75,
      fastKernel: true,
      graphMode: "available" as const,
      roles: ["head", "middle", "tail"] as const,
      validUntilMs: 2_000,
    };
    const requirement = {
      descriptorDigest,
      certificationId,
      artifactManifestDigest,
      backend: "cuda",
      runtimeAbi: "cuda-12",
      quantization: "bf16",
      contextTokens: 8_192,
      requiredRoles: ["head", "tail"] as const,
      requireFastKernel: true,
      requireGraph: true,
      nowMs: 1_000,
    };

    expect(engineNodeIsEligible({ ...state, engineCapability: capability }, requirement))
      .toBe(true);
    expect(engineNodeIsEligible({ ...state, engineCapability: capability }, {
      ...requirement,
      nowMs: 2_000,
    })).toBe(false);
    expect(engineNodeIsEligible({ ...state, engineCapability: capability }, {
      ...requirement,
      quantization: "q4_k_m",
    })).toBe(false);
    expect(engineNodeIsEligible({ ...state, engineCapability: capability }, {
      ...requirement,
      requiredRoles: ["draft"],
    })).toBe(false);
    expect(engineNodeIsEligible(state, requirement)).toBe(false);
  });

  it("never assigns more layers than the certified engine capacity", () => {
    const topology = completeTopology(4);
    const descriptorDigest = `sha256:${"b".repeat(64)}` as const;
    const certificationId = `sha256:${"d".repeat(64)}` as const;
    const artifactManifestDigest = `sha256:${"e".repeat(64)}` as const;
    const nodeStates = states(topology).map((state) => ({
      ...state,
      engineCapability: {
        profileId: `sha256:${"f".repeat(64)}` as const,
        descriptorDigest,
        certificationId,
        artifactManifestDigest,
        backend: "cuda",
        runtimeAbi: "cuda-12",
        quantizations: ["bf16"],
        maxContextTokens: 32_768,
        maxLayerCount: 2,
        kvBytesPerToken: 2_048,
        maxKvTokens: 32_768,
        decodeScale: 0.5,
        prefillScale: 0.75,
        fastKernel: true,
        graphMode: "available" as const,
        roles: ["head", "middle", "tail"] as const,
        validUntilMs: 2_000,
      },
    }));
    const placement = planNativeReplicaChains(
      model(6),
      topology,
      workload(4),
      nodeStates,
      {
        workloadClass: "interactive",
        sessionId: null,
        kvMissPenaltyMs: 0,
        engine: {
          descriptorDigest,
          certificationId,
          artifactManifestDigest,
          backend: "cuda",
          runtimeAbi: "cuda-12",
          quantization: "bf16",
          contextTokens: 128,
          requiredRoles: ["head", "middle", "tail"],
          requireFastKernel: true,
          requireGraph: true,
          nowMs: 1_000,
        },
      },
      { desiredReplicas: 1, searchOptions: narrowSearch() },
    );

    expect(placement.complete).toBe(true);
    expect(placement.chains[0]!.plan.stages).toHaveLength(3);
    expect(placement.chains[0]!.plan.stages.every(
      (stage) => stage.layerEnd - stage.layerStart <= 2,
    )).toBe(true);
  });

  it("rejects a manually supplied route beyond the engine layer ceiling", () => {
    const topology = completeTopology(1);
    topology.nodes[0] = { ...topology.nodes[0]!, maxStageLayers: 2 };
    const profile = model(3);
    const activeWorkload = workload(1);
    const metrics = evaluateDistributionPlan(profile, topology, activeWorkload, {
      algorithm: "external",
      codec: "fp16",
      microBatchSize: 1,
      prefillChunkTokens: 16,
      stages: [{ nodeId: "node-0", layerStart: 0, layerEnd: 3 }],
    });

    expect(metrics.feasible).toBe(false);
    expect(metrics.infeasibleReason).toBe("engine_layer_capacity_exceeded:node-0");
  });

  it("forms a chain from specialized head, middle and tail capabilities", () => {
    const topology = completeTopology(3);
    const descriptorDigest = `sha256:${"c".repeat(64)}` as const;
    const certificationId = `sha256:${"d".repeat(64)}` as const;
    const artifactManifestDigest = `sha256:${"e".repeat(64)}` as const;
    const roleByNode = ["head", "middle", "tail"] as const;
    const nodeStates = states(topology).map((state, index) => ({
      ...state,
      engineCapability: {
        profileId: `sha256:${"f".repeat(64)}` as const,
        descriptorDigest,
        certificationId,
        artifactManifestDigest,
        backend: "cuda",
        runtimeAbi: "cuda-12",
        quantizations: ["bf16"],
        maxContextTokens: 8_192,
        maxLayerCount: 1,
        kvBytesPerToken: 2_048,
        maxKvTokens: 8_192,
        decodeScale: 0.5,
        prefillScale: 0.75,
        fastKernel: true,
        graphMode: "available" as const,
        roles: [roleByNode[index]!],
        validUntilMs: 2_000,
      },
    }));
    const placement = planNativeReplicaChains(
      model(3),
      topology,
      workload(3),
      nodeStates,
      {
        workloadClass: "interactive",
        sessionId: null,
        kvMissPenaltyMs: 0,
        engine: {
          descriptorDigest,
          certificationId,
          artifactManifestDigest,
          backend: "cuda",
          runtimeAbi: "cuda-12",
          quantization: "bf16",
          contextTokens: 128,
          requiredRoles: ["head", "middle", "tail"],
          requireFastKernel: true,
          requireGraph: true,
          nowMs: 1_000,
        },
      },
      { desiredReplicas: 1, searchOptions: narrowSearch() },
    );

    expect(placement.complete).toBe(true);
    expect(placement.chains[0]!.nodeIds).toEqual(["node-0", "node-1", "node-2"]);
  });

  it("rejects a route whose node cannot own its pipeline position", () => {
    const topology = completeTopology(1);
    topology.nodes[0] = { ...topology.nodes[0]!, stageRoles: ["middle"] };
    const metrics = evaluateDistributionPlan(model(1), topology, workload(1), {
      algorithm: "external",
      codec: "fp16",
      microBatchSize: 1,
      prefillChunkTokens: 16,
      stages: [{ nodeId: "node-0", layerStart: 0, layerEnd: 1 }],
    });

    expect(metrics.feasible).toBe(false);
    expect(metrics.infeasibleReason).toBe("engine_stage_role_unsupported:node-0");
  });
});

function chain(
  chainId: string,
  plan: DistributionPlan,
  metrics: ReturnType<typeof evaluateDistributionPlan>,
): NativeCompleteChain {
  return {
    chainId,
    plan,
    metrics,
    score: {
      total: 0,
      computeMs: 0,
      queueMs: 0,
      p95Ms: 0,
      kvPenaltyMs: 0,
      physicalBoundaryMs: 0,
      physicalBoundaryCount: 0,
      failureRisk: 0,
    },
    nodeIds: plan.stages.map((stage) => stage.nodeId),
    failureDomainIds: plan.stages.map((stage) => `domain-${stage.nodeId}`),
  };
}

function twoStagePlan(first: string, second: string): DistributionPlan {
  return {
    algorithm: "test",
    codec: "fp16",
    microBatchSize: 1,
    prefillChunkTokens: 16,
    stages: [
      { nodeId: first, layerStart: 0, layerEnd: 2 },
      { nodeId: second, layerStart: 2, layerEnd: 4 },
    ],
  };
}

function model(layerCount: number): DistributedModelProfile {
  return {
    id: "native-test-model",
    layers: Array.from({ length: layerCount }, (_, index) => ({
      index,
      weightBytes: 64 * 1024 * 1024,
      activationElements: 1_024,
      kvBytesPerToken: 32,
      decodeMsAtUnit: 1,
      prefillMsPerTokenAtUnit: 0.04,
    })),
    embeddingBytes: 16 * 1024 * 1024,
    lmHeadBytes: 16 * 1024 * 1024,
    runtimeOverheadBytesPerStage: 8 * 1024 * 1024,
    embeddingDecodeMsAtUnit: 0.2,
    lmHeadDecodeMsAtUnit: 0.2,
    embeddingPrefillMsPerTokenAtUnit: 0.01,
    lmHeadPrefillMsPerTokenAtUnit: 0.01,
  };
}

function workload(maxStages: number): DistributionWorkload {
  return {
    promptTokens: 32,
    outputTokens: 16,
    contextTokens: 128,
    concurrentSequences: 1,
    maxStages,
    maxQualityLoss: 0,
    minRouteAvailability: 0.9,
    batchWindowMs: 0,
    p95: true,
  };
}

function completeTopology(count: number): DistributionTopology {
  const nodes = Array.from({ length: count }, (_, index): ComputeNodeProfile => ({
    id: `node-${index}`,
    region: `region-${index % 3}`,
    memoryBytes: 2 * 1024 * 1024 * 1024,
    reserveBytes: 128 * 1024 * 1024,
    decodeScale: 1 + index * 0.001,
    prefillScale: 1 + index * 0.001,
    codecScale: 1,
    batchGain: 0,
    maxBatchSpeedup: 1,
    powerWatts: 100,
    availability: 1,
  }));
  return {
    nodes,
    links: nodes.flatMap((from) =>
      nodes
        .filter((to) => to.id !== from.id)
        .map((to) => ({
          from: from.id,
          to: to.id,
          oneWayLatencyMs: 5,
          jitterP95Ms: 1,
          bandwidthMbps: 1_000,
          lossRate: 0,
          availability: 1,
        })),
    ),
  };
}

function states(topology: DistributionTopology): NativeFleetNodeState[] {
  return topology.nodes.map((node, index) => ({
    nodeId: node.id,
    physicalHostId: `host-${index}`,
    failureDomainId: `domain-${index}`,
    queuedRequests: 0,
    activeRequests: 0,
    capacity: 4,
    freeSlots: 4,
    observedP95ServiceMs: 10,
    kvSessionIds: [],
  }));
}

function narrowSearch(): SearchOptions {
  return {
    ...DEFAULT_SEARCH_OPTIONS,
    beamWidth: 64,
    candidateCodecs: ["fp16"],
    candidateMicroBatchSizes: [1],
    candidatePrefillChunks: [16],
  };
}
