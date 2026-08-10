/*
 * The branching model behind the page-long mycelium.
 *
 * This is deliberately not `mycelium-growth.ts`: that one grows over the
 * surface of a sphere for the hero globe, in latitude/longitude, and none of
 * its geometry survives being flattened. This grows down a tall rectangle in
 * page pixels, which is the only shape a scrolling page has.
 *
 * Two properties matter more than the drawing itself:
 *
 *  - It is deterministic. The seed is fixed, so the colony is identical on
 *    every load and every reload. A background that re-rolls its own shape on
 *    each visit reads as noise rather than as part of the brand, and it makes
 *    a visual regression impossible to review.
 *  - It is generated once per size, not per frame. The renderer only decides
 *    how much of a precomputed tree is currently visible, so a scroll frame
 *    costs a walk over an array and nothing else.
 *
 * Growth runs downward because the reader does. Filaments are pushed away from
 * the horizontal centre as they extend: the middle of the page is where the
 * text lives, and a decorative background that crosses a paragraph stops being
 * decorative.
 */

export interface Hypha {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  /** Quadratic control point: each hypha bows instead of reading as a ruler line. */
  cx: number;
  cy: number;
  /** Branch generation: 0 is a founding filament, deeper is thinner and fainter. */
  gen: number;
  /** True where this segment ends at a fork — the renderer marks those points. */
  fork: boolean;
}

/** Fixed seed: same colony every load. See the note at the top of this file. */
const SEED = 0x5eed_1a9f;
const MAX_GEN = 5;
const STEP_PX = 30;

/** mulberry32 — small, fast, and stable across engines, which `Math.random` is not. */
function makeRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b_79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296;
  };
}

interface Tip {
  x: number;
  y: number;
  /** Radians from straight down; positive bends right. */
  angle: number;
  gen: number;
}

/**
 * Grows a colony down a `width` × `height` rectangle.
 *
 * Coordinates come back in real pixels rather than normalised units, so the
 * caller regenerates on resize instead of stretching a fixed shape — a colony
 * scaled from 1440px to a phone would read as smeared, not as smaller.
 */
export function growField(width: number, height: number): Hypha[] {
  const random = makeRandom(SEED);
  const hyphae: Hypha[] = [];
  // Segment scale belongs to the viewport, not to the document. The old
  // height/42 calculation turned a long page into a handful of 150px strokes.
  const step = Math.max(24, Math.min(34, STEP_PX * (width / 1440 + 0.45)));
  const maxHyphae = Math.ceil((height / step) * 26);
  const centre = width / 2;

  // Two founding filaments, one per margin, and that is the whole colony.
  //
  // An earlier pass seeded a fresh cluster of roots every few hundred pixels.
  // Because each of those runs was short, the result was a chain of separate
  // tufts with bare paper between them: the network appeared to restart as the
  // reader scrolled instead of continuing. Continuity has to come from the
  // geometry, not from the density — so each margin gets exactly one trunk that
  // descends the entire span, and everything else hangs off it.
  const tips: Tip[] = [];
  tips.push({ x: width * 0.08, y: -step * 2, angle: 0.16, gen: 0 });
  tips.push({ x: width * 0.92, y: -step * 2, angle: -0.16, gen: 0 });

  while (tips.length > 0 && hyphae.length < maxHyphae) {
    const tip = tips.shift();
    if (!tip) break;

    let { x, y, angle } = tip;
    const { gen } = tip;
    // A trunk lives until it runs out of page; branches are short and feathery.
    // Both halves of that rule matter. Unlimited life everywhere would fill the
    // margins into a solid mat, and short life everywhere breaks the colony into
    // the disconnected tufts this replaced.
    const life =
      gen === 0
        ? Math.ceil((height - y) / (step * 0.55)) + 8
        : Math.round((7 - gen * 1.2) * (0.7 + random() * 0.6));

    for (let i = 0; i < life && hyphae.length < maxHyphae; i += 1) {
      // Wander, then bend away from the centre column where the copy sits.
      angle += (random() - 0.5) * 0.34;
      const fromCentre = (x - centre) / (width / 2); // -1 … 1
      // The centre column is not merely discouraged, it is fenced off: the push
      // outward grows the deeper a filament strays, so nothing ever wanders
      // across the copy the way a single long diagonal used to.
      if (Math.abs(fromCentre) < 0.62) {
        const intrusion = (0.62 - Math.abs(fromCentre)) / 0.62;
        angle += Math.sign(fromCentre || 1) * (0.14 + intrusion * 0.42);
      }
      // And bend back in at the very edge, so filaments graze the margin
      // rather than pinning themselves flat against it.
      if (Math.abs(fromCentre) > 0.93) angle -= Math.sign(fromCentre) * 0.3;
      angle = Math.max(-1.05, Math.min(1.05, angle));

      const length = step * (0.5 + random() * 0.5);
      const nx = x + Math.sin(angle) * length;
      const ny = y + Math.cos(angle) * length;

      // Trunks branch at a steady rate; sub-branches branch far less. A uniform
      // rate compounds — each generation doubles the last — and the colony comes
      // out as dense knots separated by bare stretches, which reads as the same
      // jumpiness the single-trunk change was meant to remove. Damping by
      // generation keeps the filament count roughly even down the whole span.
      const forkChance = gen === 0 ? 0.34 : 0.3 / (gen + 1);
      const forking = gen < MAX_GEN && random() < forkChance && hyphae.length < maxHyphae - 6;
      const bow = (random() - 0.5) * length * 0.42;
      hyphae.push({
        x1: x,
        y1: y,
        x2: nx,
        y2: ny,
        cx: (x + nx) * 0.5 + Math.cos(angle) * bow,
        cy: (y + ny) * 0.5 - Math.sin(angle) * bow,
        gen,
        fork: forking,
      });

      x = nx;
      y = ny;
      if (y > height) break;

      if (forking) {
        // Real hyphae fork at a narrow angle and often throw two branches at
        // once. Wide near-perpendicular splits were what made the old field
        // look like shattering rather than growth.
        const side = random() < 0.5 ? -1 : 1;
        tips.push({ x, y, angle: angle + side * (0.26 + random() * 0.3), gen: gen + 1 });
        if (gen === 0 && random() < 0.28) {
          tips.push({ x, y, angle: angle - side * (0.22 + random() * 0.26), gen: gen + 1 });
        }
      }
    }
  }

  // Rendering stops at the growth frontier, so document order must be spatial
  // rather than breadth-first branch order.
  return hyphae.sort((a, b) => a.y1 - b.y1);
}
