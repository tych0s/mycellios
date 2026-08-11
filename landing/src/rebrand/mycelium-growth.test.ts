import { describe, expect, it } from "vitest";
import {
  buildLandVectors,
  dot,
  FADE_SECONDS,
  filamentHeading,
  filamentPoint,
  FRUIT_SECONDS,
  hairPoint,
  MAX_FILAMENTS,
  Mycelium,
  slerp,
} from "./mycelium-growth";

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
    // Live hyphae are capped exactly; the dying ones on top of that are bounded
    // by how many can be mid-fade at once, which is small and self-limiting.
    const live = mycelium.filaments.filter((filament) => filament.vitality >= 1);
    expect(live.length).toBeLessThanOrEqual(MAX_FILAMENTS);
    expect(mycelium.filaments.length).toBeLessThan(MAX_FILAMENTS * 1.5);
  });

  /*
   * The two things a visitor said were wrong with this globe: the filaments
   * were straight lines and the nodes were plain dots, so the whole thing read
   * as a node graph rather than as a mycelium. The renderer draws the curve and
   * the soft tips, but the geometry that makes them possible lives here.
   */
  describe("filaments read as hyphae rather than as graph edges", () => {
    it("bows every filament off the straight great-circle path", () => {
      const mycelium = new Mycelium(land);
      for (let i = 0; i < 60 * 60; i += 1) mycelium.advance(1 / 60);
      expect(mycelium.filaments.length).toBeGreaterThan(10);

      for (const filament of mycelium.filaments) {
        // Measured at the middle, where the bow is widest, against the straight
        // path the old renderer drew.
        const bowed = filamentPoint(filament, 0.5);
        const straight = slerp(filament.from, filament.to, 0.5);
        const offset = Math.acos(Math.min(1, Math.max(-1, dot(bowed, straight))));
        const span = Math.acos(Math.min(1, Math.max(-1, dot(filament.from, filament.to))));
        // A visible curve, but never so wide the hop doubles back on itself.
        expect(offset).toBeGreaterThan(span * 0.05);
        expect(offset).toBeLessThan(span * 0.45);
      }
    });

    /*
     * The bow has to leave the great circle, and checking "it moved away from
     * slerp" does not prove that.
     *
     * The first implementation displaced along `axis × P`, which is the
     * *forward* tangent — so every sample slid further along the very same
     * great circle. Distance-from-slerp was happily non-zero, every test
     * passed, and the filaments rasterised dead straight. The only measurement
     * that catches it is the component along the circle's own axis, which is
     * exactly zero for any point on the circle no matter how far along it has
     * been slid.
     */
    it("bends out of the great-circle plane rather than sliding along it", () => {
      const mycelium = new Mycelium(land);
      for (let i = 0; i < 60 * 60; i += 1) mycelium.advance(1 / 60);

      for (const filament of mycelium.filaments) {
        const axis = {
          x: filament.from.y * filament.to.z - filament.from.z * filament.to.y,
          y: filament.from.z * filament.to.x - filament.from.x * filament.to.z,
          z: filament.from.x * filament.to.y - filament.from.y * filament.to.x,
        };
        const length = Math.hypot(axis.x, axis.y, axis.z);
        if (length < 1e-9) continue;
        const unit = { x: axis.x / length, y: axis.y / length, z: axis.z / length };
        // Out-of-plane displacement at the widest point of the bow.
        const offPlane = Math.abs(dot(filamentPoint(filament, 0.5), unit));
        const span = Math.acos(Math.min(1, Math.max(-1, dot(filament.from, filament.to))));
        expect(offPlane).toBeGreaterThan(span * 0.04);
      }
    });

    /*
     * Curving each filament is not enough on its own. With hops chosen purely
     * by proximity, every junction turned a corner at an angle unrelated to the
     * one before it, and a chain of corners reads as a routing diagram however
     * gracefully each individual segment is drawn. Growth has to carry
     * momentum, so this measures the turn at the joins rather than the shape of
     * any one filament.
     */
    it("keeps heading across junctions instead of zigzagging", () => {
      const mycelium = new Mycelium(land);
      for (let i = 0; i < 60 * 120; i += 1) mycelium.advance(1 / 60);

      // Filaments that continue one another: this one's start is that one's end.
      let joins = 0;
      let aligned = 0;
      for (const first of mycelium.filaments) {
        for (const second of mycelium.filaments) {
          if (first === second || second.from !== first.to) continue;
          const heading = filamentHeading(first);
          const next = filamentHeading({ ...second, bow: 0, wobble: 0 });
          if (Math.hypot(heading.x, heading.y, heading.z) < 1e-9) continue;
          joins += 1;
          // cos of the turn: >0 means the colony carried on rather than
          // doubling back on the hop it just finished.
          if (dot(heading, next) > 0) aligned += 1;
        }
      }

      expect(joins).toBeGreaterThan(20);
      // Momentum, not a rule: branches deliberately leave at an angle, so this
      // is a strong majority rather than unanimity.
      expect(aligned / joins).toBeGreaterThan(0.7);
    });

    it("keeps both endpoints exact, so junctions still meet", () => {
      const mycelium = new Mycelium(land);
      for (let i = 0; i < 60 * 60; i += 1) mycelium.advance(1 / 60);

      for (const filament of mycelium.filaments) {
        // A bow that moved the ends would tear the network apart at every fork.
        expect(dot(filamentPoint(filament, 0), filament.from)).toBeCloseTo(1, 9);
        expect(dot(filamentPoint(filament, 1), filament.to)).toBeCloseTo(1, 9);
      }
    });

    it("stays on the surface of the sphere all the way along", () => {
      const mycelium = new Mycelium(land);
      for (let i = 0; i < 60 * 60; i += 1) mycelium.advance(1 / 60);

      for (const filament of mycelium.filaments.slice(0, 20)) {
        for (let step = 0; step <= 10; step += 1) {
          const point = filamentPoint(filament, step / 10);
          expect(Math.hypot(point.x, point.y, point.z)).toBeCloseTo(1, 9);
        }
      }
    });

    it("grows side hairs that trail the advancing tip", () => {
      const mycelium = new Mycelium(land);
      for (let i = 0; i < 60 * 90; i += 1) mycelium.advance(1 / 60);

      const withHairs = mycelium.filaments.filter((filament) => filament.hairs.length > 0);
      // The texture has to be common enough to read as fuzz along the network.
      expect(withHairs.length).toBeGreaterThan(mycelium.filaments.length * 0.4);

      for (const filament of withHairs) {
        for (const hair of filament.hairs) {
          // A hair never grows ahead of the parent that carries it.
          if (filament.progress < hair.at) expect(hair.progress).toBe(0);
          // …and it leaves the filament, rather than lying along it.
          if (hair.progress > 0.5) {
            const root = filamentPoint(filament, hair.at);
            const end = hairPoint(filament, hair, hair.progress);
            expect(Math.acos(Math.min(1, Math.max(-1, dot(root, end))))).toBeGreaterThan(0);
            expect(Math.hypot(end.x, end.y, end.z)).toBeCloseTo(1, 9);
          }
        }
      }
    });

    it("fades hyphae out instead of deleting them between frames", () => {
      const mycelium = new Mycelium(land);
      // Long enough that retirement is well underway.
      for (let i = 0; i < 60 * 60 * 6; i += 1) mycelium.advance(1 / 60);

      const dying = mycelium.filaments.filter((f) => f.vitality > 0 && f.vitality < 1);
      expect(dying.length).toBeGreaterThan(0);

      // Nothing may jump to invisible: over one frame the dimmest survivor can
      // only lose one frame's worth of the fade.
      const before = dying.map((filament) => filament.vitality);
      mycelium.advance(1 / 60);
      dying.forEach((filament, index) => {
        const lost = before[index]! - filament.vitality;
        expect(lost).toBeLessThanOrEqual(1 / 60 / FADE_SECONDS + 1e-9);
      });
    });
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

  /*
   * Fruiting is the point of the hero: a network that only ever spreads is a
   * graph, and every distributed-compute landing draws a graph. These guard the
   * three ways it could stop meaning that — never fruiting, fruiting so often
   * the globe turns into a mushroom farm, or leaking mushrooms across a long
   * session until the frame budget goes.
   */
  describe("fruiting bodies", () => {
    it("fruits from the colony once it is established", () => {
      const mycelium = new Mycelium(land);
      expect(mycelium.fruits).toHaveLength(0);

      for (let i = 0; i < 60 * 300; i += 1) mycelium.advance(1 / 60);
      expect(mycelium.fruits.length).toBeGreaterThan(0);
    });

    it("fruits only from hubs, on land, at the end of a completed hop", () => {
      const mycelium = new Mycelium(land);
      for (let i = 0; i < 60 * 300; i += 1) mycelium.advance(1 / 60);

      expect(mycelium.fruits.length).toBeGreaterThan(0);
      for (const fruit of mycelium.fruits) {
        expect(fruit.source.hub).toBe(true);
        expect(fruit.source.progress).toBe(1);
        expect(fruit.at).toBe(fruit.source.to);
        expect(distanceToLand(fruit.at)).toBeLessThan(1e-6);
      }
    });

    /*
     * Fruiting has to land in a band, and both edges of it are failures that
     * actually shipped.
     *
     * Too few is what a visitor complained about: at a 0.16 hub rate and a
     * 0.28 roll the colony carried seven bodies and, because a sphere hides
     * half of itself, put two in front of the viewer. One bronze mushroom
     * among four thousand land dots does not read as this brand's motif — it
     * reads as a rendering artefact.
     *
     * Too many silts the continents up. Land dots sit about 11px apart on a
     * desktop hero and a mature cap is about 19px across, so once bodies
     * outnumber roughly a third of the hubs their caps start overlapping and
     * the landmasses fill in as solid bronze. That end was tested directly by
     * drawing one body per land point: the globe stops being a globe and the
     * frame rate falls to 16fps.
     *
     * Averaged over seeds because a single colony's count swings by several on
     * nothing more than which hops happened to complete before the cutoff.
     */
    it("keeps mushrooms frequent enough to read, and rare enough to stay events", () => {
      const runs = 8;
      let filaments = 0;
      let fruits = 0;
      let facing = 0;
      for (let seed = 0; seed < runs; seed += 1) {
        const mycelium = new Mycelium(land, seed * 977 + 13);
        for (let i = 0; i < 60 * 600; i += 1) mycelium.advance(1 / 60);
        filaments += mycelium.filaments.length;
        fruits += mycelium.fruits.length;
        // The hemisphere an orthographic camera at +z actually shows.
        facing += mycelium.fruits.filter((fruit) => fruit.at.z > 0.16).length;
      }

      expect(filaments / runs).toBeGreaterThan(50);
      // Enough on screen that fruiting is obviously something this network
      // does repeatedly, rather than one body a visitor may never scroll past.
      expect(facing / runs).toBeGreaterThan(6);
      // …and still a small minority of the colony's own filaments.
      expect(fruits / runs).toBeLessThan((filaments / runs) * 0.2);
    });

    it("takes a long time to open, so nothing pops onto the globe", () => {
      const mycelium = new Mycelium(land);
      for (let i = 0; i < 60 * 300; i += 1) mycelium.advance(1 / 60);

      for (const fruit of mycelium.fruits) {
        expect(1 / fruit.speed).toBeGreaterThan(FRUIT_SECONDS * 0.6);
        expect(fruit.maturity).toBeLessThanOrEqual(1);
      }
    });

    it("retires a mushroom with the filament it grew on", () => {
      const mycelium = new Mycelium(land);
      for (let i = 0; i < 60 * 60 * 10; i += 1) mycelium.advance(1 / 60);

      // Nothing is left standing on a hop the colony has already forgotten.
      const live = new Set(mycelium.filaments);
      for (const fruit of mycelium.fruits) expect(live.has(fruit.source)).toBe(true);
      expect(mycelium.fruits.length).toBeLessThanOrEqual(MAX_FILAMENTS);
    });

    it("does not fruit faster on a faster display", () => {
      // Fruiting is rolled once per completed hop, never per frame, so the
      // rate is set by the colony rather than by the refresh rate. A per-frame
      // roll would show up here as a ~5x difference.
      //
      // Measured across seeds rather than on one colony. A single run lands
      // around twenty hubs and four to nine mushrooms, so its ratio moves by
      // ±0.2 on nothing more than which hops happened to complete — a finer
      // timestep lands them at slightly different moments and so draws the
      // shared RNG in a different order. Comparing two single runs measures
      // that noise, not the frame-rate independence being claimed here.
      const fruitsPerHub = (step: number, seconds: number) => {
        let fruits = 0;
        let hubs = 0;
        for (const seed of [0x9e3779b9, 1, 2, 3, 4, 5, 6, 7]) {
          const mycelium = new Mycelium(land, seed);
          for (let i = 0; i < seconds / step; i += 1) mycelium.advance(step);
          fruits += mycelium.fruits.length;
          hubs += mycelium.filaments.filter((filament) => filament.hub).length;
        }
        expect(hubs).toBeGreaterThan(40);
        return fruits / hubs;
      };

      const slow = fruitsPerHub(1 / 30, 300);
      const fast = fruitsPerHub(1 / 144, 300);
      expect(Math.abs(fast - slow)).toBeLessThan(0.1);
    });
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
