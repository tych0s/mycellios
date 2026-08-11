/*
 * The fruiting body: the only new shape in the brand's visual language.
 *
 * A mycelium is invisible until it fruits. That is the whole reason this
 * exists: the page already draws the network — filaments spreading over the
 * globe, a colony branching down the page — but a network drawn as lines says
 * "graph", and every distributed-systems landing draws a graph. What no one
 * else can draw is the moment the network produces something. Here that is a
 * mushroom, and it means the same thing in both places it appears: a hub that
 * has finished its work.
 *
 * The geometry is generated, not drawn as an asset, for the same reason the
 * globe is: it has to grow. A fruiting body is a button first — short stipe, a
 * closed dome — and only opens its cap once it is tall. Handing the renderer a
 * finished silhouette to scale would produce a mushroom-shaped balloon, which
 * is the tell that separates a generated organism from a sticker.
 *
 * Everything is emitted in a local frame: origin at the base of the stipe, +y
 * away from the surface the body grew out of, +x to its right, and a mature
 * body exactly 1 unit tall. The two consumers then place it in very different
 * spaces — the hero maps +y onto the sphere's normal in screen space, the page
 * backdrop maps it onto document pixels — and neither has to know how a
 * mushroom is shaped.
 */

export interface Point {
  x: number;
  y: number;
}

export interface FruitingBody {
  /** Base of the stipe → apex, as a sampled curve. */
  stipe: Point[];
  /** Cap silhouette: left rim → over the crown → right rim. */
  cap: Point[];
  /**
   * The hollow under the cap: right rim → up under the crown → left rim.
   *
   * `cap` alone is an open curve, so filling it closes rim to rim on a straight
   * chord and paints a lens — a bronze crescent with no underside, which is
   * what a cap looks like with its most characteristic feature deleted. Walking
   * this back after `cap` closes the outline on a concave sweep instead, so the
   * rim becomes an overhanging edge with shadow beneath it. That overhang is
   * the whole silhouette: it is what separates a mushroom from a lollipop, and
   * it survives at 12px where the gills inside it no longer resolve.
   */
  underside: Point[];
  /** Underside, from the stipe apex out to the rim. Empty while still a button. */
  gills: Array<[Point, Point]>;
  /** Where the stipe meets the cap: the point the renderer can mark as a node. */
  apex: Point;
  /** Current height in local units (1 at full maturity). */
  height: number;
  /** Current half-width of the cap, in the same units. */
  capRadius: number;
}

const STIPE_SAMPLES = 6;
const CAP_SAMPLES = 16;
/* The cap only starts opening once the stipe is a third of the way up, so the
   two phases overlap instead of the body inflating uniformly. */
const OPEN_START = 0.3;
/* Below this the cap is still a closed button and has no visible underside. */
const GILLS_VISIBLE = 0.12;

function clamp01(value: number): number {
  return value < 0 ? 0 : value > 1 ? 1 : value;
}

/** Quadratic Bézier: the stipe is a curve, never a straight stick. */
function quadratic(a: Point, control: Point, b: Point, t: number): Point {
  const inverse = 1 - t;
  const wa = inverse * inverse;
  const wc = 2 * inverse * t;
  const wb = t * t;
  return { x: a.x * wa + control.x * wc + b.x * wb, y: a.y * wa + control.y * wc + b.y * wb };
}

/**
 * Builds a fruiting body at a given maturity.
 *
 * `maturity` 0 is nothing, 1 is a fully open cap; values outside are clamped so
 * a caller can drive this straight from a scroll position or an age in seconds
 * without guarding the ends. `lean` tilts the body off the surface normal
 * (-1…1) — real ones are never plumb, and a row of vertical mushrooms reads as
 * a bar chart. `gillCount` is the number of radial lines under the cap; those
 * are what make the shape read as a mushroom rather than as a pin or a tree.
 */
export function growFruitingBody(options: {
  maturity: number;
  lean?: number;
  gillCount?: number;
}): FruitingBody {
  const maturity = clamp01(options.maturity);
  const lean = options.lean ?? 0;
  const gillCount = Math.max(0, Math.round(options.gillCount ?? 7));

  // The stipe rises quickly and then eases; the cap lags and opens late.
  const height = maturity ** 0.62;
  const open = clamp01((maturity - OPEN_START) / (1 - OPEN_START));

  /*
   * A young cap is narrow and domed, a mature one broader and flatter — that
   * silhouette change is most of what reads as growth.
   *
   * The absolute proportions are the part that took measuring. The ratio that
   * decides whether this reads as a mushroom or as a letter T is depth against
   * width: a cap 1.2 wide and 0.19 deep is a bar balanced on a line no matter
   * how the exponent is tuned, because at any real render size the curve is
   * under a pixel. These numbers give a mature cap 0.68 wide and 0.6 deep —
   * close to square — so the dome is still a dome at the 12px the globe draws
   * it at, and the rim reaches down far enough to enclose the gills instead of
   * skimming past them.
   */
  const capRadius = height * (0.26 + 0.42 * open);
  const capDepth = height * (0.6 - 0.05 * open);
  /*
   * Rim droop: the cap hangs over the gills and lifts as it opens. It is most
   * of the cap's depth, which is what turns a dome into a bell with an
   * underside — the shape that has somewhere for gills to hang.
   *
   * It must stay well under `capDepth`, though, and "well" is measured in
   * pixels rather than in ratios. The crown sits at `apex.y - droop +
   * capDepth`, so what is left over — `capDepth - droop` — is the only cover
   * the cap gives the top of the stalk. At 0.92 that leftover was 2.4px on
   * the globe while the stipe itself is 3.3px wide, so the stalk's own round
   * end stood proud of the crown: a nail through a thimble, at the exact spot
   * the eye goes to read the silhouette. At 0.75 the cover is 4.5px, clear of
   * the stalk, and the rim still hangs 7.6px below the apex — more than
   * enough overhang for the hollow to read.
   */
  const droop = capDepth * (0.75 - 0.12 * open);

  const apex: Point = { x: lean * height * 0.2, y: height };
  const base: Point = { x: 0, y: 0 };
  // Control point sits inside the lean, so the stipe bows rather than shears.
  const control: Point = { x: lean * height * 0.02, y: height * 0.55 };

  const stipe: Point[] = [];
  for (let i = 0; i <= STIPE_SAMPLES; i += 1) stipe.push(quadratic(base, control, apex, i / STIPE_SAMPLES));

  /*
   * The cap is a half-ellipse with its vertical term raised to 1.1. Slightly
   * above 1 rather than below: with a cap this deep, an exponent under 1 pulls
   * the shoulders out into a flare and the crown flat, which is an umbrella.
   * Just over 1 keeps the shoulders tucked under the crown, so the outline
   * closes onto the rim the way a real cap does.
   */
  const cap: Point[] = [];
  for (let i = 0; i <= CAP_SAMPLES; i += 1) {
    const angle = Math.PI - (i / CAP_SAMPLES) * Math.PI;
    cap.push({
      x: apex.x + capRadius * Math.cos(angle),
      y: apex.y - droop + capDepth * Math.sin(angle) ** 1.1,
    });
  }

  /*
   * The underside, walked back from the right rim to the left one.
   *
   * It is a shallower arc than the cap and it bows *upward* into the dome, so
   * outline-then-underside encloses a crescent of solid cap with a hollow bitten
   * out of its base. The depth is a fraction of the cap's: too deep and the cap
   * becomes a ring, too shallow and the rim stops overhanging and the shape
   * flattens back into a lens.
   */
  const underside: Point[] = [];
  /*
   * The hollow is a third of the cap's depth, no more.
   *
   * Both numbers below were measured by rasterising the filled polygon rather
   * than chosen by eye, because the failure mode is invisible in the outline:
   * at 0.46 of the depth with a flatter exponent the ceiling rises so close to
   * the crown that the solid part thins to its two edges, and the fill paints
   * the exact bronze crescent this was meant to remove. A third of the depth
   * leaves a dome with real thickness above a cup that is still clearly a cup.
   */
  const hollow = capDepth * 0.3;
  /* Height of the hollow's ceiling above rim level, at a fraction `u` of the
     way out to the rim. The gills hang from this curve, so they share it with
     the outline rather than being fitted to it by eye. */
  const ceilingAt = (u: number) => hollow * (1 - u * u) ** 0.75;
  for (let i = 0; i <= CAP_SAMPLES; i += 1) {
    const angle = (i / CAP_SAMPLES) * Math.PI;
    underside.push({
      x: apex.x + capRadius * Math.cos(angle),
      y: apex.y - droop + ceilingAt(Math.cos(angle)),
    });
  }

  const gills: Array<[Point, Point]> = [];
  if (open > GILLS_VISIBLE && gillCount > 0) {
    /*
     * Gills attach around the stalk, not at a point on it. Running every one
     * out of a single hub is what draws a teepee: nine lines sharing one pixel
     * read as an apex, and with round caps that shared pixel also blots into a
     * visible dot. Spreading the inner ends across the top of the stalk — the
     * projection of a ring seen from the side — gives the fan somewhere to
     * start and leaves the crown clean.
     */
    const inner = capRadius * 0.2;
    for (let i = 1; i <= gillCount; i += 1) {
      // -1 … 1 across the underside, never quite reaching the rim.
      const across = -1 + (2 * i) / (gillCount + 1);
      /*
       * Every gill hangs from the hollow's own ceiling down to rim level.
       *
       * The fan reads as a fan because the *ceiling* curves, not because the
       * gills are fitted to a curve of their own: the inner ends ride
       * `ceilingAt`, which is highest over the stalk and drops away towards
       * the rim, so the outer gills start lower and the whole set splays.
       * Hanging them from a flat height instead put the inner ends *above*
       * the ceiling — outside the shape, drawn as dark scratches across the
       * solid bronze of the crown rather than as shading inside the cup.
       * Sharing `ceilingAt` with the outline means the fan keeps sitting in
       * the hollow whatever the cap's proportions do.
       */
      const rimU = 0.93;
      const innerU = (inner / capRadius) * Math.abs(across);
      const clearance = hollow * 0.06;
      gills.push([
        {
          x: apex.x + inner * across,
          y: apex.y - droop + ceilingAt(innerU) - clearance,
        },
        {
          /*
           * Out to the rim and all the way down to rim level.
           *
           * The outer gills get their length from how far they travel across
           * the cap, but the centre one barely moves in x — almost all of its
           * length is the drop. Stopping it short of the rim is what made it
           * the shortest of the fan and tripped the visible-length check: the
           * gill facing the viewer was the one hardest to see. Ending every
           * gill at rim level makes each one span the full depth of the cup.
           */
          x: apex.x + capRadius * rimU * across,
          y: apex.y - droop,
        },
      ]);
    }
  }

  return { stipe, cap, underside, gills, apex, height, capRadius };
}
