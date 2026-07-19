import type {
  ComputeNodeProfile,
  DirectedLinkProfile,
  DistributedModelProfile,
  DistributionTopology,
  DistributionWorkload,
  LayerProfile,
} from "./types.js";

const MIB = 1024 * 1024;

export interface DistributionScenario {
  id: string;
  description: string;
  model: DistributedModelProfile;
  topology: DistributionTopology;
  workload: DistributionWorkload;
}

export function fixedDistributionScenarios(): DistributionScenario[] {
  const interactive: DistributionWorkload = {
    promptTokens: 256,
    outputTokens: 128,
    contextTokens: 512,
    concurrentSequences: 1,
    maxStages: 8,
    maxQualityLoss: 0.005,
    minRouteAvailability: 0.88,
    batchWindowMs: 0.35,
    p95: true,
  };
  const smallModel = syntheticModel("synthetic-135m", 12, 48 * MIB, 768, 2_048, 1.15);
  const qwenProxy = syntheticModel("qwen-0.5b-proxy", 28, 32 * MIB, 1_024, 4_096, 1.6);
  const mediumProxy = syntheticModel("dense-1.5b-proxy", 24, 72 * MIB, 2_048, 8_192, 2.25);

  const lanNodes = [
    node("lan-a", "home", 384, 0.55, 0.62, 85, 0.998),
    node("lan-b", "home", 384, 0.72, 0.8, 105, 0.997),
    node("lan-c-large-slow", "home", 640, 1.75, 1.6, 145, 0.995),
    node("lan-d", "home", 320, 0.68, 0.75, 95, 0.996),
  ];
  const wifiNodes = Array.from({ length: 8 }, (_, index) =>
    node(
      `wifi-${index}`,
      index < 5 ? "madrid" : "toledo",
      [384, 512, 384, 640, 320, 768, 384, 512][index]!,
      [0.65, 0.9, 0.58, 1.5, 0.72, 1.85, 0.8, 1.1][index]!,
      [0.72, 0.95, 0.66, 1.35, 0.78, 1.7, 0.9, 1.2][index]!,
      [90, 120, 85, 150, 95, 175, 110, 135][index]!,
      0.995 - index * 0.0004,
    ),
  );
  const regionalNodes = Array.from({ length: 12 }, (_, index) =>
    node(
      `regional-${index}`,
      ["madrid", "madrid", "madrid", "madrid", "valencia", "valencia", "bilbao", "bilbao", "lisboa", "paris", "paris", "madrid"][index]!,
      [768, 512, 1_024, 640, 768, 512, 1_024, 640, 768, 1_024, 512, 384][index]!,
      [0.58, 0.72, 1.2, 0.64, 0.8, 1.05, 0.7, 1.4, 0.9, 0.62, 1.1, 0.68][index]!,
      [0.64, 0.8, 1.08, 0.7, 0.9, 1.12, 0.78, 1.3, 0.96, 0.69, 1.16, 0.75][index]!,
      [110, 95, 165, 105, 125, 115, 155, 145, 135, 170, 120, 90][index]!,
      0.996,
    ),
  );
  const manyNodes = Array.from({ length: 32 }, (_, index) =>
    node(
      `fleet-${index}`,
      `cell-${Math.floor(index / 8)}`,
      [384, 512, 768, 1_024][index % 4]!,
      0.5 + ((index * 37) % 140) / 100,
      0.55 + ((index * 29) % 135) / 100,
      75 + ((index * 23) % 110),
      0.994 + ((index % 5) * 0.001),
    ),
  );

  return [
    {
      id: "lan-small-4",
      description: "Modelo pequeño forzado a repartirse en cuatro equipos LAN heterogéneos",
      model: smallModel,
      topology: completeTopology(lanNodes, (left, right) => ({
        oneWayLatencyMs: 0.35 + Math.abs(hashNumber(left.id) - hashNumber(right.id)) % 0.6,
        jitterP95Ms: 0.25,
        bandwidthMbps: 940,
        lossRate: 0.0002,
      })),
      workload: interactive,
    },
    {
      id: "wifi-qwen-proxy-8",
      description: "Proxy Qwen 0.5B sobre Wi-Fi y dos ciudades",
      model: qwenProxy,
      topology: completeTopology(wifiNodes, (left, right) => {
        const sameRegion = left.region === right.region;
        return {
          oneWayLatencyMs: sameRegion ? 1.8 + (hashPair(left.id, right.id) % 30) / 10 : 9 + (hashPair(left.id, right.id) % 80) / 10,
          jitterP95Ms: sameRegion ? 1.2 : 4.5,
          bandwidthMbps: sameRegion ? 180 + (hashPair(left.id, right.id) % 320) : 70 + (hashPair(left.id, right.id) % 90),
          lossRate: sameRegion ? 0.002 : 0.006,
        };
      }),
      workload: interactive,
    },
    {
      id: "regional-medium-12",
      description: "Modelo de 1.5B repartido regionalmente con enlaces asimétricos",
      model: mediumProxy,
      topology: completeTopology(regionalNodes, (left, right) => {
        const sameRegion = left.region === right.region;
        return {
          oneWayLatencyMs: sameRegion ? 2.5 : 11 + (hashPair(left.region, right.region) % 180) / 10,
          jitterP95Ms: sameRegion ? 1.5 : 5,
          bandwidthMbps: sameRegion ? 300 : 80 + (hashPair(left.id, right.id) % 120),
          lossRate: sameRegion ? 0.0015 : 0.005,
        };
      }),
      workload: { ...interactive, promptTokens: 512, contextTokens: 1_024, maxStages: 7 },
    },
    {
      id: "many-machines-select-few",
      description: "Treinta y dos nodos; el planificador debe ignorar los que empeoran la ruta",
      model: mediumProxy,
      topology: completeTopology(manyNodes, (left, right) => {
        const sameCell = left.region === right.region;
        return {
          oneWayLatencyMs: sameCell ? 1.2 + (hashPair(left.id, right.id) % 20) / 10 : 14 + (hashPair(left.id, right.id) % 210) / 10,
          jitterP95Ms: sameCell ? 0.8 : 6,
          bandwidthMbps: sameCell ? 600 : 60 + (hashPair(left.id, right.id) % 100),
          lossRate: sameCell ? 0.001 : 0.008,
        };
      }),
      workload: { ...interactive, maxStages: 6 },
    },
    {
      id: "concurrent-recurrent-pipeline",
      description: "Ocho conversaciones intercaladas para medir batching y ciclo de pipeline",
      model: qwenProxy,
      topology: completeTopology(wifiNodes, (left, right) => ({
        oneWayLatencyMs: left.region === right.region ? 1.5 : 9,
        jitterP95Ms: left.region === right.region ? 0.8 : 3.5,
        bandwidthMbps: left.region === right.region ? 400 : 120,
        lossRate: left.region === right.region ? 0.001 : 0.004,
      })),
      workload: {
        ...interactive,
        promptTokens: 192,
        contextTokens: 384,
        concurrentSequences: 8,
        maxStages: 7,
        batchWindowMs: 0.5,
      },
    },
  ];
}

export function randomDistributionScenario(seed: number, index: number): DistributionScenario {
  const random = mulberry32((seed ^ Math.imul(index + 1, 0x9e3779b1)) >>> 0);
  const nodeCount = 8 + Math.floor(random() * 9);
  const layerCount = 16 + 4 * Math.floor(random() * 4);
  const layerWeightMiB = 34 + Math.floor(random() * 35);
  const model = syntheticModel(
    `random-model-${index}`,
    layerCount,
    layerWeightMiB * MIB,
    random() < 0.5 ? 1_024 : 2_048,
    random() < 0.5 ? 4_096 : 8_192,
    1.2 + random() * 1.4,
  );
  const nodes = Array.from({ length: nodeCount }, (_, nodeIndex) =>
    node(
      `r${index}-n${nodeIndex}`,
      `region-${Math.floor(random() * 3)}`,
      [320, 384, 512, 640, 768, 1_024][Math.floor(random() * 6)]!,
      0.48 + random() * 1.7,
      0.52 + random() * 1.55,
      70 + random() * 130,
      0.992 + random() * 0.0075,
    ),
  );
  const topology = completeTopology(nodes, (left, right) => {
    const sameRegion = left.region === right.region;
    return {
      oneWayLatencyMs: sameRegion ? 0.8 + random() * 4 : 8 + random() * 30,
      jitterP95Ms: sameRegion ? 0.4 + random() * 1.8 : 2 + random() * 8,
      bandwidthMbps: sameRegion ? 180 + random() * 820 : 40 + random() * 180,
      lossRate: sameRegion ? random() * 0.003 : 0.001 + random() * 0.009,
    };
  });
  const concurrentSequences = [1, 2, 4, 8][Math.floor(random() * 4)]!;
  return {
    id: `random-${index}`,
    description: "Escenario Monte Carlo heterogéneo",
    model,
    topology,
    workload: {
      promptTokens: [128, 256, 512][Math.floor(random() * 3)]!,
      outputTokens: [64, 128, 192][Math.floor(random() * 3)]!,
      contextTokens: concurrentSequences >= 8 ? 256 : [256, 512, 1_024][Math.floor(random() * 3)]!,
      concurrentSequences,
      maxStages: Math.min(8, nodeCount),
      maxQualityLoss: 0.005,
      minRouteAvailability: 0.82,
      batchWindowMs: 0.5,
      p95: true,
    },
  };
}

export function syntheticModel(
  id: string,
  layerCount: number,
  meanWeightBytes: number,
  activationElements: number,
  kvBytesPerToken: number,
  meanDecodeMs: number,
): DistributedModelProfile {
  const layers: LayerProfile[] = Array.from({ length: layerCount }, (_, index) => {
    const wave = 1 + 0.08 * Math.sin((index + 1) * 1.7);
    return {
      index,
      weightBytes: Math.round(meanWeightBytes * wave),
      activationElements,
      kvBytesPerToken,
      decodeMsAtUnit: meanDecodeMs * (0.93 + (index % 5) * 0.035),
      prefillMsPerTokenAtUnit: (meanDecodeMs / 22) * (0.9 + (index % 4) * 0.05),
    };
  });
  const tiedEndpointBytes = Math.round(meanWeightBytes * 1.25);
  return {
    id,
    layers,
    embeddingBytes: tiedEndpointBytes,
    // Tied weights occupy one allocation on one process, but first and last
    // pipeline stages must each retain the matrix when they are different PCs.
    lmHeadBytes: tiedEndpointBytes,
    tiedEmbeddingAndHead: true,
    runtimeOverheadBytesPerStage: 24 * MIB,
    embeddingDecodeMsAtUnit: meanDecodeMs * 0.22,
    lmHeadDecodeMsAtUnit: meanDecodeMs * 0.65,
    embeddingPrefillMsPerTokenAtUnit: meanDecodeMs / 90,
    lmHeadPrefillMsPerTokenAtUnit: 0,
  };
}

export function completeTopology(
  nodes: ComputeNodeProfile[],
  values: (
    left: ComputeNodeProfile,
    right: ComputeNodeProfile,
  ) => Omit<DirectedLinkProfile, "from" | "to">,
): DistributionTopology {
  const links: DirectedLinkProfile[] = [];
  for (const left of nodes) {
    for (const right of nodes) {
      if (left.id === right.id) continue;
      links.push({ from: left.id, to: right.id, ...values(left, right) });
    }
  }
  return { nodes, links };
}

function node(
  id: string,
  region: string,
  memoryMiB: number,
  decodeScale: number,
  prefillScale: number,
  powerWatts: number,
  availability: number,
): ComputeNodeProfile {
  return {
    id,
    region,
    memoryBytes: memoryMiB * MIB,
    reserveBytes: 32 * MIB,
    decodeScale,
    prefillScale,
    codecScale: 0.8 + decodeScale * 0.2,
    batchGain: 0.22,
    maxBatchSpeedup: 2.8,
    powerWatts,
    availability,
  };
}

function hashNumber(value: string): number {
  let hash = 2166136261;
  for (const character of value) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function hashPair(left: string, right: string): number {
  return hashNumber(`${left}\u0000${right}`);
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}
