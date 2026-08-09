import { describe, expect, it } from "vitest";
import { buildLandVectors, dot, MAX_FILAMENTS, Mycelium, slerp } from "./mycelium-growth";

const land = buildLandVectors();

/** Distance from a point to the nearest land dot, in radians of arc. */
function distanceToLand(point: { x: number; y: number; z: number }): number {
  let best = -2;
  for (const candidate of land) best = Math.max(best, dot(candidate, point));
  return Math.acos(Math.min(1, Math.max(-1, best)));
}

describe("hero globe mycelium growth", () => {
  it("rasterizes a land grid of unit vectors", () => {
    expect(land.length).toBeGreaterThan(3000);
    for (const point of land.slice(0, 200)) {
      const length = Math.sqrt(point.x ** 2 + point.y ** 2 + point.z ** 2);
      expect(length).toBeCloseTo(1, 6);
    }
  });

  it("starts already colonised so the hero is never an empty planet", () => {
    const mycelium = new Mycelium(land);
    for (let i = 0; i < 34; i += 1) mycelium.advance(6.5 * 0.9);
    expect(mycelium.filaments.length).toBeGreaterThan(20);
  });

  it("keeps extending the network over time rather than settling", () => {
    const mycelium = new Mycelium(land);
    const step = () => mycelium.advance(1 / 60);

    for (let i = 0; i < 60 * 10; i += 1) step();
    const afterTenSeconds = mycelium.filaments.length;

    for (let i = 0; i < 60 * 60; i += 1) step();
    const afterSeventySeconds = mycelium.filaments.length;

    expect(afterTenSeconds).toBeGreaterThan(0);
    expect(afterSeventySeconds).toBeGreaterThan(afterTenSeconds);
  });

  it("grows slowly: a filament takes seconds, not frames, to extend", () => {
    const mycelium = new Mycelium(land);
    // Seeds wait a randomized moment before the first hop, so let them sprout.
    for (let i = 0; i < 60 * 3; i += 1) mycelium.advance(1 / 60);

    expect(mycelium.filaments.length).toBeGreaterThan(0);
    for (const filament of mycelium.filaments) {
      // Even the fastest permitted filament needs >4s of wall clock to finish.
      expect(1 / filament.speed).toBeGreaterThan(4);
    }
  });

  it("retires old hyphae so memory stays bounded during a long session", () => {
    const mycelium = new Mycelium(land);
    for (let i = 0; i < 60 * 60 * 10; i += 1) mycelium.advance(1 / 60);
    expect(mycelium.filaments.length).toBeLessThanOrEqual(MAX_FILAMENTS);
  });

  it("connects land to land, so filaments never cross open ocean", () => {
    const mycelium = new Mycelium(land);
    for (let i = 0; i < 60 * 45; i += 1) mycelium.advance(1 / 60);

    expect(mycelium.filaments.length).toBeGreaterThan(10);
    for (const filament of mycelium.filaments) {
      // Endpoints are land dots, and the hop is short enough to read as local
      // spread rather than a line flung across the globe.
      expect(distanceToLand(filament.from)).toBeLessThan(1e-6);
      expect(distanceToLand(filament.to)).toBeLessThan(1e-6);
      const span = Math.acos(Math.min(1, Math.max(-1, dot(filament.from, filament.to))));
      expect(span).toBeLessThanOrEqual(0.2 + 1e-6);
    }
  });

  it("interpolates along the sphere so arcs hug the surface", () => {
    const a = land[0]!;
    const b = land[400]!;
    const middle = slerp(a, b, 0.5);
    const length = Math.sqrt(middle.x ** 2 + middle.y ** 2 + middle.z ** 2);
    expect(length).toBeCloseTo(1, 6);
  });

  it("is deterministic, so every visitor sees the same colony", () => {
    const runOnce = () => {
      const mycelium = new Mycelium(land);
      for (let i = 0; i < 600; i += 1) mycelium.advance(1 / 60);
      return mycelium.filaments.map((f) => `${f.to.x.toFixed(5)},${f.to.y.toFixed(5)}`).join("|");
    };
    expect(runOnce()).toBe(runOnce());
  });
});
