import { describe, expect, it } from "vitest";
import { simulateNetwork } from "../src/simulator/model.js";

describe("1,000-node planning simulator", () => {
  it("is reproducible for a fixed seed", () => {
    const options = { nodes: 1_000, users: 1_000, seed: 42, scenario: "normal" as const };
    expect(simulateNetwork(options)).toEqual(simulateNetwork(options));
  });

  it("shows why a large-model demand spike creates a queue", () => {
    const normal = simulateNetwork({ nodes: 1_000, users: 1_000, seed: 42, scenario: "normal" });
    const spike = simulateNetwork({
      nodes: 1_000,
      users: 1_000,
      seed: 42,
      scenario: "large_model_spike",
    });
    expect(normal.inventory.onlineNodes).toBeGreaterThan(600);
    expect(normal.inventory.glmRoutes).toBeGreaterThan(10);
    expect(normal.network.acceptanceRate).toBeGreaterThan(0.7);
    expect(spike.network.queued).toBeGreaterThan(normal.network.queued);
  });

  it("lets 4 GB nodes contribute to the small-model pool", () => {
    const result = simulateNetwork({ nodes: 1_000, users: 1_000, seed: 9, scenario: "normal" });
    expect(result.byTier.T0.total).toBe(500);
    expect(result.byTier.T0.active).toBeGreaterThan(0);
  });
});
