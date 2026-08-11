import { describe, expect, it } from "vitest";
import { growFruitingBody } from "./fruiting-body";

/*
 * These assert the *silhouette*, not the drawing. The renderer can restyle a
 * fruiting body freely, but if any of these stop holding it has stopped being
 * a mushroom: a body that opens its cap before it has a stalk, or one whose
 * cap is wider than the one next to it at a later stage, reads as a glitch on
 * a page whose entire premise is that the network grows.
 */
describe("fruiting body", () => {
  it("is nothing at all before it starts", () => {
    const body = growFruitingBody({ maturity: 0 });
    expect(body.height).toBe(0);
    expect(body.capRadius).toBe(0);
    expect(body.gills).toHaveLength(0);
  });

  it("grows a stalk before it opens a cap", () => {
    const young = growFruitingBody({ maturity: 0.2 });
    const open = growFruitingBody({ maturity: 1 });

    // A button: real height, but no underside on show yet.
    expect(young.height).toBeGreaterThan(0.2);
    expect(young.gills).toHaveLength(0);
    // …and the cap is still narrow relative to the body.
    expect(young.capRadius / young.height).toBeLessThan(open.capRadius / open.height);
    expect(open.gills.length).toBeGreaterThan(0);
  });

  it("only ever gets taller and wider as it matures", () => {
    let previousHeight = -1;
    let previousRadius = -1;
    for (let step = 0; step <= 20; step += 1) {
      const body = growFruitingBody({ maturity: step / 20 });
      expect(body.height).toBeGreaterThanOrEqual(previousHeight);
      expect(body.capRadius).toBeGreaterThanOrEqual(previousRadius);
      previousHeight = body.height;
      previousRadius = body.capRadius;
    }
  });

  it("clamps past the ends, so a caller can hand it a raw scroll value", () => {
    expect(growFruitingBody({ maturity: -3 }).height).toBe(0);
    const beyond = growFruitingBody({ maturity: 4 });
    const mature = growFruitingBody({ maturity: 1 });
    expect(beyond.height).toBeCloseTo(mature.height, 10);
    expect(beyond.capRadius).toBeCloseTo(mature.capRadius, 10);
  });

  it("caps the stalk it grew: the cap sits on the apex, not beside it", () => {
    const body = growFruitingBody({ maturity: 1, lean: 0.8 });
    const stalkTop = body.stipe[body.stipe.length - 1]!;

    expect(stalkTop.x).toBeCloseTo(body.apex.x, 10);
    expect(stalkTop.y).toBeCloseTo(body.apex.y, 10);

    // The two rim ends hang *below* the apex and the crown rises *above* it:
    // that is the join between stalk and cap, and it is what stops the shape
    // reading as two sprites stacked on one another.
    const left = body.cap[0]!;
    const right = body.cap[body.cap.length - 1]!;
    const crown = body.cap.reduce((highest, point) => (point.y > highest.y ? point : highest));

    expect(left.y).toBeLessThan(body.apex.y);
    expect(right.y).toBeLessThan(body.apex.y);
    expect(crown.y).toBeGreaterThan(body.apex.y);
    // The rim is level and centred on the apex, so the cap is not askew.
    expect(left.y).toBeCloseTo(right.y, 10);
    expect((left.x + right.x) / 2).toBeCloseTo(body.apex.x, 10);
  });

  it("leans off the vertical without shearing the stalk off its base", () => {
    const straight = growFruitingBody({ maturity: 1, lean: 0 });
    const leaning = growFruitingBody({ maturity: 1, lean: 1 });

    expect(straight.apex.x).toBeCloseTo(0, 10);
    expect(leaning.apex.x).toBeGreaterThan(0.1);
    // Both stay rooted where they were planted.
    expect(leaning.stipe[0]).toEqual({ x: 0, y: 0 });
    expect(straight.stipe[0]).toEqual({ x: 0, y: 0 });
  });

  it("hangs its gills under the cap and inside the rim", () => {
    const body = growFruitingBody({ maturity: 1, gillCount: 9 });
    expect(body.gills).toHaveLength(9);

    for (const [hub, rim] of body.gills) {
      expect(Math.abs(hub.x - body.apex.x)).toBeLessThan(body.capRadius * 0.5);
      expect(rim.y).toBeLessThanOrEqual(hub.y + 1e-9);
      expect(Math.abs(rim.x - body.apex.x)).toBeLessThan(body.capRadius);
    }
  });

  /*
   * Both of these shipped to the dev URL and were caught by looking at it.
   * They are the difference between a mushroom's underside and a teepee.
   */
  it("gives every gill a visible length, including the middle one", () => {
    // An odd count puts one gill at dead centre; that is the one facing the
    // viewer, and it used to collapse to a zero-length segment that a round
    // cap painted as a dot on the crown.
    for (const gillCount of [5, 7, 9]) {
      const body = growFruitingBody({ maturity: 1, gillCount });
      const shortest = Math.min(
        ...body.gills.map(([a, b]) => Math.hypot(b.x - a.x, b.y - a.y)),
      );
      expect(shortest).toBeGreaterThan(body.capRadius * 0.2);
    }
  });

  it("spreads the gill roots across the stalk instead of one shared point", () => {
    const body = growFruitingBody({ maturity: 1, gillCount: 9 });
    const roots = new Set(body.gills.map(([hub]) => hub.x.toFixed(6)));
    // Nine lines out of a single pixel read as an apex, not as an underside.
    expect(roots.size).toBe(9);
  });

  /*
   * The cap is filled by walking `cap` and then `underside`. If the underside
   * ever stops being concave, or stops meeting the rim, that fill closes on a
   * straight chord and paints a bronze crescent — which is exactly what shipped
   * to the globe and read as a lens rather than a mushroom.
   */
  it("closes the cap on a hollow, so the rim overhangs instead of filling a lens", () => {
    const body = growFruitingBody({ maturity: 1, lean: 0.3, gillCount: 7 });
    const rimLeft = body.cap[0]!;
    const rimRight = body.cap[body.cap.length - 1]!;

    // The underside runs back the other way: it starts at the right rim and
    // ends at the left one, so outline + underside is a single closed loop.
    expect(body.underside[0]!.x).toBeCloseTo(rimRight.x, 10);
    expect(body.underside[0]!.y).toBeCloseTo(rimRight.y, 10);
    const last = body.underside[body.underside.length - 1]!;
    expect(last.x).toBeCloseTo(rimLeft.x, 10);
    expect(last.y).toBeCloseTo(rimLeft.y, 10);

    // It bows up into the dome, and never as far as the crown — that gap is
    // the solid bronze of the cap itself.
    const crown = body.cap.reduce((high, p) => (p.y > high.y ? p : high));
    const deepest = body.underside.reduce((high, p) => (p.y > high.y ? p : high));
    expect(deepest.y).toBeGreaterThan(rimLeft.y);
    expect(deepest.y).toBeLessThan(crown.y);
  });

  /*
   * The one that matters, and the one every outline-based assertion above
   * missed: the cap can satisfy all of them and still *fill* as a crescent,
   * because what gets painted is the polygon, not the curve. So rasterise it.
   *
   * Down the middle of a mature cap there must be a continuous run of solid
   * bronze between the crown and the hollow's ceiling. When the hollow was
   * tuned too deep that run vanished, the fill collapsed onto its two edges,
   * and the globe drew the bronze lens the whole underside was added to fix.
   */
  it("fills as a solid dome, not as a crescent", () => {
    for (const maturity of [0.5, 0.75, 1]) {
      const body = growFruitingBody({ maturity, gillCount: 7 });
      const polygon = [...body.cap, ...body.underside];
      const inside = (px: number, py: number) => {
        let hit = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
          const a = polygon[i]!;
          const b = polygon[j]!;
          if (a.y > py !== b.y > py && px < ((b.x - a.x) * (py - a.y)) / (b.y - a.y) + a.x) hit = !hit;
        }
        return hit;
      };

      const crown = body.cap.reduce((high, p) => (p.y > high.y ? p : high));
      const ceiling = body.underside.reduce((high, p) => (p.y > high.y ? p : high));
      // Sample the vertical centre line between the hollow's ceiling and the
      // crown: every sample should land in bronze.
      const samples = 12;
      let solid = 0;
      for (let i = 1; i < samples; i += 1) {
        const y = ceiling.y + ((crown.y - ceiling.y) * i) / samples;
        if (inside(body.apex.x, y)) solid += 1;
      }
      expect(solid).toBe(samples - 1);
      // …and that band has to be a real thickness, not a sliver.
      expect(crown.y - ceiling.y).toBeGreaterThan(body.capRadius * 0.25);
    }
  });

  /*
   * The cap has to *cover* the stalk, not just sit above it. On the globe a
   * mature body is about 22px tall and its stipe is drawn 3.3px wide, so a
   * crown clearing the apex by only 2.4px let the stalk's own stroke stand
   * proud of the dome — a nail through a thimble, right where the eye reads
   * the silhouette. Stated as a ratio so it survives any render size.
   */
  it("covers the top of its own stalk with the crown", () => {
    for (const maturity of [0.5, 0.75, 1]) {
      const body = growFruitingBody({ maturity });
      const crown = body.cap.reduce((high, p) => (p.y > high.y ? p : high));
      // The renderer draws the stipe at ~0.15 of the body's height; the crown
      // has to clear the apex by more than half of that, or the round end of
      // the stalk pokes out through the top of the cap.
      expect(crown.y - body.apex.y).toBeGreaterThan(body.height * 0.15 * 0.5);
    }
  });

  it("keeps the gills inside the hollow they hang in", () => {
    const body = growFruitingBody({ maturity: 1, gillCount: 9 });
    const deepest = body.underside.reduce((high, p) => (p.y > high.y ? p : high));
    // A gill poking out through the top of the cap would draw a dark scratch
    // across the bronze instead of shading its underside.
    for (const [hub, rim] of body.gills) {
      expect(hub.y).toBeLessThanOrEqual(deepest.y + 1e-9);
      expect(rim.y).toBeLessThanOrEqual(deepest.y + 1e-9);
    }
  });

  it("is a pure function of its inputs, so two visitors see the same organism", () => {
    const shape = (body: ReturnType<typeof growFruitingBody>) =>
      body.cap.map((point) => `${point.x.toFixed(9)},${point.y.toFixed(9)}`).join("|");
    const options = { maturity: 0.62, lean: -0.3, gillCount: 5 };
    expect(shape(growFruitingBody(options))).toBe(shape(growFruitingBody(options)));
  });
});
