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
  /** Branch generation: 0 is a founding filament, deeper is thinner and fainter. */
  gen: number;
  /** True where this segment ends at a fork — the renderer marks those points. */
  fork: boolean;
}

/** Fixed seed: same colony every load. See the note at the top of this file. */
const SEED = 0x5eed_1a9f;
const MAX_HYPHAE = 320; // ~4 kB of geometry; a full redraw stays well under a frame
const MAX_GEN = 4;
const STEPS_PER_HEIGHT = 30; // segment length ≈ field height / 30

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
  const step = height / STEPS_PER_HEIGHT;
  const centre = width / 2;

  // Four founding filaments, entering from both margins and staggered down the
  // field so the colony never reads as two symmetrical halves.
  const tips: Tip[] = [
    { x: width * 0.08, y: -step * 0.5, angle: 0.22, gen: 0 },
    { x: width * 0.93, y: -step * 0.5, angle: -0.24, gen: 0 },
    { x: width * 0.2, y: height * 0.16, angle: 0.1, gen: 1 },
    { x: width * 0.82, y: height * 0.27, angle: -0.12, gen: 1 },
  ];

  while (tips.length > 0 && hyphae.length < MAX_HYPHAE) {
    const tip = tips.shift();
    if (!tip) break;

    let { x, y, angle } = tip;
    const { gen } = tip;
    // Deeper generations are shorter-lived, so the colony thins out downward
    // instead of doubling forever.
    const life = Math.round((10 - gen * 1.7) * (0.7 + random() * 0.6));

    for (let i = 0; i < life && hyphae.length < MAX_HYPHAE; i += 1) {
      // Wander, then bend away from the centre column where the copy sits.
      angle += (random() - 0.5) * 0.34;
      const fromCentre = (x - centre) / (width / 2); // -1 … 1
      if (Math.abs(fromCentre) < 0.55) angle += Math.sign(fromCentre || 1) * 0.14;
      // And bend back in at the very edge, so filaments graze the margin
      // rather than pinning themselves flat against it.
      if (Math.abs(fromCentre) > 0.93) angle -= Math.sign(fromCentre) * 0.3;
      angle = Math.max(-1.05, Math.min(1.05, angle));

      const length = step * (0.72 + random() * 0.66);
      const nx = x + Math.sin(angle) * length;
      const ny = y + Math.cos(angle) * length;

      const forking = gen < MAX_GEN && random() < 0.19 && hyphae.length < MAX_HYPHAE - 6;
      hyphae.push({ x1: x, y1: y, x2: nx, y2: ny, gen, fork: forking });

      x = nx;
      y = ny;
      if (y > height) break;

      if (forking) {
        const side = random() < 0.5 ? -1 : 1;
        tips.push({ x, y, angle: angle + side * (0.42 + random() * 0.42), gen: gen + 1 });
      }
    }
  }

  return hyphae;
}
