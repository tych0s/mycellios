/*
 * One deterministic RNG for every generated visual on the landing.
 *
 * All of them — the globe colony, the page-long field, the fruiting bodies —
 * need the same guarantee: identical geometry on every load, on every engine.
 * `Math.random` gives neither, and three private copies of the same generator
 * gave the first without making it obvious the three were meant to match.
 *
 * mulberry32: 32-bit state, no dependencies, stable output across V8, JSC and
 * SpiderMonkey, which is what makes a visual regression reviewable at all.
 */
export function makeRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b_79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296;
  };
}
