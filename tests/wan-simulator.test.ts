import { describe, expect, it } from "vitest";
import {
  DEFAULT_WAN_OPTIONS,
  DEFAULT_WAN_ROUTES,
  estimateDefaultWanRoutes,
  estimateFleet,
  estimateWanRoute,
  WAN_PROJECTION_KIND,
} from "../src/simulator/wan-model.js";

describe("calibratable WAN inference model", () => {
  it("keeps the short regional routes interactive and rejects the 4 GB chain", () => {
    const results = estimateDefaultWanRoutes();
    expect(
      results.find((route) => route.id === "regional-fiber")?.decodeTokensPerSecond.slowPathP95,
    )
      .toBeGreaterThanOrEqual(4);
    expect(results.find((route) => route.id === "four-gb-long")?.loaded.recommendedInteractiveChats)
      .toBe(0);
  });

  it("shows that speculation improves a single WAN conversation", () => {
    const route = DEFAULT_WAN_ROUTES.find((candidate) => candidate.id === "regional-fiber")!;
    const optimized = estimateWanRoute(route);
    const baseline = estimateWanRoute(route, {
      ...DEFAULT_WAN_OPTIONS,
      useSpeculativeDecoding: false,
    });
    expect(optimized.decodeTokensPerSecond.p50).toBeGreaterThan(
      baseline.decodeTokensPerSecond.p50,
    );
  });

  it("groups many low-memory contributors into at most eight WAN-visible stages", () => {
    for (const route of estimateDefaultWanRoutes()) {
      expect(route.provenance).toBe(WAN_PROJECTION_KIND);
      expect(route.virtualStages).toBe(route.stages);
      expect(route.virtualStages).toBeLessThanOrEqual(8);
      expect(route.physicalNodes).toBeGreaterThanOrEqual(route.virtualStages);
    }
    const lowMemory = estimateDefaultWanRoutes().find(
      (route) => route.id === "four-gb-long",
    )!;
    expect(lowMemory.physicalNodes).toBe(24);
    expect(lowMemory.virtualStages).toBe(6);
  });

  it("falls back to autoregression when speculation misses break-even", () => {
    const route = DEFAULT_WAN_ROUTES.find(
      (candidate) => candidate.id === "regional-fiber",
    )!;
    const harmful = {
      ...route,
      acceptedTokensPerRound: 1.01,
      draftMsPerToken: 10_000,
    };
    const adaptive = estimateWanRoute(harmful);
    const baseline = estimateWanRoute(harmful, {
      ...DEFAULT_WAN_OPTIONS,
      useSpeculativeDecoding: false,
    });
    expect(adaptive.speculation.requested).toBe(true);
    expect(adaptive.speculation.appliedP50).toBe(false);
    expect(adaptive.decodeTokensPerSecond.p50).toBe(
      baseline.decodeTokensPerSecond.p50,
    );
  });

  it("only enables activation compression when it pays for its codec cost", () => {
    const lan = estimateWanRoute(DEFAULT_WAN_ROUTES[0]!);
    const constrained = estimateWanRoute(
      DEFAULT_WAN_ROUTES.find((route) => route.id === "four-gb-long")!,
    );
    expect(lan.activationCodec.decodeApplied).toBe(false);
    expect(constrained.activationCodec.decodeApplied).toBe(true);
    const withoutCompression = estimateWanRoute(
      DEFAULT_WAN_ROUTES.find((route) => route.id === "four-gb-long")!,
      { ...DEFAULT_WAN_OPTIONS, activationCompressionRatio: 1 },
    );
    expect(constrained.responseSeconds.p50).toBeLessThanOrEqual(
      withoutCompression.responseSeconds.p50,
    );
  });

  it("reports the slow-path throughput no faster than the median", () => {
    for (const route of estimateDefaultWanRoutes()) {
      expect(route.decodeTokensPerSecond.slowPathP95).toBeLessThanOrEqual(
        route.decodeTokensPerSecond.p50,
      );
      expect(route.ttftMs.p95).toBeGreaterThanOrEqual(route.ttftMs.p50);
    }
  });

  it("makes additional latency and stages hurt per-user speed", () => {
    const fast = estimateWanRoute(DEFAULT_WAN_ROUTES[0]!);
    const long = estimateWanRoute(
      DEFAULT_WAN_ROUTES.find((route) => route.id === "four-gb-long")!,
    );
    expect(fast.decodeTokensPerSecond.p50).toBeGreaterThan(long.decodeTokensPerSecond.p50);
    expect(fast.reliability.requestSuccessPct).toBeGreaterThan(long.reliability.requestSuccessPct);
  });

  it("scales fleet capacity without pretending ten random domestic nodes form GLM", () => {
    const regional = estimateWanRoute(
      DEFAULT_WAN_ROUTES.find((route) => route.id === "regional-fiber")!,
    );
    const ten = estimateFleet(10, regional);
    const thousand = estimateFleet(1_000, regional);
    expect(ten.interactiveRegionalRoutes).toBe(0);
    expect(thousand.interactiveRegionalRoutes).toBeGreaterThan(10);
    expect(thousand.interactiveGlmChatsAtTarget).toBeGreaterThan(100);
  });
});
