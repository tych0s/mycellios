import { beforeAll, describe, expect, it } from "vitest";
import {
  GLM45_AIR_Q4KM,
  IDEAL_SLOTS_DISCLAIMER,
  THEORETICAL_PHYSICS_KIND,
  activationBoundaryBytes,
  activeWeightReadBytesPerToken,
  allocateLayersWaterFill,
  approximateActiveFlopsPerToken,
  defaultWanSensitivityMatrix,
  estimateWanSensitivity,
  kvBytesForContext,
  kvBytesPerContextToken,
  kvLimitedConversationSlots,
  routeSuccessProbability,
  simulateCuratedTenNodeRoute,
  simulateWanSuite,
  type WanSummary,
} from "../src/simulator/wan-physics.js";

describe("WAN theoretical physics v1", () => {
  let canonicalSuite: WanSummary[];

  beforeAll(() => {
    canonicalSuite = simulateWanSuite([10, 100, 1_000], {}, 123_456_789);
  }, 30_000);

  it("labels every result as theoretical and ideal, not a benchmark or guarantee", () => {
    for (const summary of canonicalSuite) {
      expect(summary.kind).toBe(THEORETICAL_PHYSICS_KIND);
      expect(summary.kind).toBe("theoretical_physics_v1_not_benchmark");
      expect(summary.idealSlotsSemantics).toBe(IDEAL_SLOTS_DISCLAIMER);
    }
  });

  it("reproduces the shared-RNG 10/100/1000 fleet snapshots", () => {
    const ten = canonicalSuite[0]!;
    expect(ten).toMatchObject({
      nodes: 10,
      trials: 1_000,
      probabilityAnyRoute: 0,
      routes: { p50: 0, p95: 0 },
      stages: { p50: 0, p95: 0 },
      tokensPerSecond: { p50: 0, slowPathP95: 0 },
      ttftMs: { p50: 0, p95: 0 },
      concurrency: { p50: 0, p95: 0 },
      incrementalPowerKw: { p50: 0, p95: 0 },
    });

    const hundred = canonicalSuite[1]!;
    expect(hundred).toMatchObject({
      nodes: 100,
      trials: 1_000,
      probabilityAnyRoute: 0.82,
      routes: { p50: 1, p95: 2 },
      stages: { p50: 8, p95: 15 },
      concurrency: { p50: 9, p95: 23 },
    });
    expect(hundred.tokensPerSecond.p50).toBeCloseTo(2.939957335366745, 12);
    expect(hundred.tokensPerSecond.slowPathP95).toBeCloseTo(
      1.8204846194168316,
      12,
    );
    expect(hundred.ttftMs.p50).toBeCloseTo(12_167.776447600353, 9);
    expect(hundred.ttftMs.p95).toBeCloseTo(16_679.847084669092, 9);
    expect(hundred.incrementalPowerKw.p50).toBeCloseTo(1.044635735400254, 12);
    expect(hundred.incrementalPowerKw.p95).toBeCloseTo(2.337060912726913, 12);

    const thousand = canonicalSuite[2]!;
    expect(thousand).toMatchObject({
      nodes: 1_000,
      trials: 1_000,
      probabilityAnyRoute: 1,
      routes: { p50: 26, p95: 29 },
      stages: { p50: 8, p95: 16 },
      concurrency: { p50: 230, p95: 261 },
    });
    expect(thousand.tokensPerSecond.p50).toBeCloseTo(2.939364443842158, 12);
    expect(thousand.tokensPerSecond.slowPathP95).toBeCloseTo(
      1.435567939194832,
      12,
    );
    expect(thousand.ttftMs.p50).toBeCloseTo(11_212.177771954513, 9);
    expect(thousand.ttftMs.p95).toBeCloseTo(17_371.311972447963, 9);
    expect(thousand.incrementalPowerKw.p50).toBeCloseTo(
      26.896750137080446,
      12,
    );
    expect(thousand.incrementalPowerKw.p95).toBeCloseTo(
      30.237400569662917,
      12,
    );
  });

  it("is reproducible for the same seed and preserves percentile direction", () => {
    const first = simulateWanSuite([10, 100], { trials: 25 }, 42);
    const second = simulateWanSuite([10, 100], { trials: 25 }, 42);
    expect(first).toEqual(second);
    for (const summary of canonicalSuite) {
      expect(summary.tokensPerSecond.slowPathP95).toBeLessThanOrEqual(
        summary.tokensPerSecond.p50,
      );
      expect(summary.ttftMs.p95).toBeGreaterThanOrEqual(summary.ttftMs.p50);
      expect(summary.routes.p95).toBeGreaterThanOrEqual(summary.routes.p50);
    }
  });

  it("reproduces the viable curated ten-node regional route", () => {
    const curated = simulateCuratedTenNodeRoute();
    expect(curated).toMatchObject({
      nodes: 10,
      trials: 10_000,
      probabilityAnyRoute: 1,
      routes: { p50: 1, p95: 1 },
      stages: { p50: 10, p95: 10 },
      concurrency: { p50: 10, p95: 10 },
    });
    expect(curated.tokensPerSecond.p50).toBeCloseTo(3.0247193508415973, 12);
    expect(curated.tokensPerSecond.slowPathP95).toBeCloseTo(
      2.4521446490264767,
      12,
    );
    expect(curated.ttftMs.p50).toBeCloseTo(10_493.37470023294, 9);
    expect(curated.ttftMs.p95).toBeCloseTo(12_711.036241659087, 9);
    expect(curated.incrementalPowerKw.p50).toBeCloseTo(
      1.2198119124055375,
      12,
    );
    expect(curated.incrementalPowerKw.p95).toBeCloseTo(
      1.2791925706653855,
      12,
    );
  }, 30_000);

  it("water-fills exactly 46 layers without exceeding a node capacity", () => {
    const nodes = [
      { layerCapacity: 3, decodeMsPerLayer: 3.5 },
      { layerCapacity: 3, decodeMsPerLayer: 3.5 },
      { layerCapacity: 3, decodeMsPerLayer: 3.5 },
      { layerCapacity: 3, decodeMsPerLayer: 3.5 },
      { layerCapacity: 6, decodeMsPerLayer: 2.5 },
      { layerCapacity: 6, decodeMsPerLayer: 2.5 },
      { layerCapacity: 6, decodeMsPerLayer: 2.5 },
      { layerCapacity: 8, decodeMsPerLayer: 2 },
      { layerCapacity: 8, decodeMsPerLayer: 2 },
      { layerCapacity: 12, decodeMsPerLayer: 1.5 },
    ];
    const allocation = allocateLayersWaterFill(nodes);
    expect(allocation).not.toBeNull();
    expect(allocation!.reduce((total, layers) => total + layers, 0)).toBe(46);
    allocation!.forEach((layers, index) => {
      expect(layers).toBeLessThanOrEqual(nodes[index]!.layerCapacity);
    });
    expect(
      allocateLayersWaterFill([
        { layerCapacity: 20, decodeMsPerLayer: 1 },
        { layerCapacity: 25, decodeMsPerLayer: 1 },
      ]),
    ).toBeNull();
  });

  it("makes RTT, lower bandwidth and larger activations hurt throughput", () => {
    const baseline = estimateWanSensitivity({
      stages: 10,
      computeMs: 110,
      oneWayHopMs: 15,
      bandwidthMbps: 100,
      activationBytes: activationBoundaryBytes(1),
    });
    const doubledRtt = estimateWanSensitivity({
      stages: 10,
      computeMs: 110,
      oneWayHopMs: 30,
      bandwidthMbps: 100,
      activationBytes: activationBoundaryBytes(1),
    });
    const tenthBandwidth = estimateWanSensitivity({
      stages: 10,
      computeMs: 110,
      oneWayHopMs: 15,
      bandwidthMbps: 10,
      activationBytes: activationBoundaryBytes(1),
    });
    const bf16Activation = estimateWanSensitivity({
      stages: 10,
      computeMs: 110,
      oneWayHopMs: 15,
      bandwidthMbps: 100,
      activationBytes: activationBoundaryBytes(2),
    });
    expect(doubledRtt.tokensPerSecond).toBeLessThan(baseline.tokensPerSecond);
    expect(tenthBandwidth.tokensPerSecond).toBeLessThan(baseline.tokensPerSecond);
    expect(bf16Activation.tokensPerSecond).toBeLessThan(baseline.tokensPerSecond);

    const matrix = defaultWanSensitivityMatrix().map((point) =>
      point.tokensPerSecond,
    );
    expect(matrix).toHaveLength(9);
    expect(matrix[0]).toBeCloseTo(8.9313, 4);
    expect(matrix[1]).toBeCloseTo(5.8151, 4);
    expect(matrix[2]).toBeCloseTo(2.618, 4);
    expect(matrix[3]).toBeCloseTo(6.1246, 4);
    expect(matrix[8]).toBeCloseTo(1.0254, 4);
  });

  it("exposes activation, active-weight, KV and route-success physics", () => {
    expect(GLM45_AIR_Q4KM).toMatchObject({
      layers: 46,
      hiddenSize: 4_096,
      weightBytes: 73e9,
      totalParams: 106e9,
      activeParams: 12e9,
    });
    expect(activationBoundaryBytes(1)).toBe(4_096);
    expect(activationBoundaryBytes(2)).toBe(8_192);
    expect(activeWeightReadBytesPerToken()).toBeCloseTo(
      8_264_150_943.396226,
      5,
    );
    expect(approximateActiveFlopsPerToken()).toBe(24e9);
    expect(kvBytesPerContextToken()).toBe(188_416);
    expect(kvBytesForContext(2_000)).toBe(376_832_000);
    expect(kvBytesForContext(20_000)).toBe(3_768_320_000);
    expect(kvBytesForContext(128_000)).toBe(24_117_248_000);
    expect(kvBytesForContext(2_000, 1)).toBe(188_416_000);
    expect(kvLimitedConversationSlots(8e9, 20_000)).toBeLessThanOrEqual(
      kvLimitedConversationSlots(8e9, 2_000),
    );
    expect(routeSuccessProbability(10, 0.02)).toBeCloseTo(
      (1 - 0.02) ** 10,
      15,
    );
  });
});

