import { useEffect, useRef } from "react";
import {
  FLARE_TAPER,
  makeSporeField,
  refitSporeField,
  riseSpore,
  SPECIES,
  SPECIES_PROFILE,
  sporeAlpha,
  sporeCount,
  sporeDrawOrder,
  sporeExtent,
  sporeX,
  type Species,
  type Spore,
} from "./spore-drift";

/*
 * Spores drifting through the band from the local address to the three doors.
 *
 * WHY SPORES HERE AND NOTHING ELSE.
 *
 * There is a real hole in the page at this point: the local-address band and the
 * doors are two short sections with a rule between them and a run of empty paper
 * in each. Every other empty stretch on this page was answered with a fruiting
 * body — the hero, the exchange section's bottom rule, the questions column —
 * and a fourth one here would be the page saying the same thing for the fourth
 * time. It also would not be true to the place: what is above a colony is not
 * another mushroom, it is what the colony released. So this band gets the
 * release.
 *
 * It is the only decoration in either section. No body, no filament of its own,
 * no silhouette — the layer is motes and nothing else, which is what makes it
 * read as air rather than as another illustration.
 *
 * WHY IT IS NOT THE HERO'S FIELD PAINTED A SECOND TIME.
 *
 * The hero's spores are one sprite rising in a line against a dark stage. Ivory
 * paper is the opposite problem: a single soft shape repeated a hundred times
 * over it is a halftone screen, and the same gesture twice is weaker than either
 * once. So this field has three species drawn from `SPECIES_PROFILE` — a plain
 * mote, a defocused halo with a faint ring, and a small hard glint — painted far
 * stratum first so the near motes land on top of the far ones. `spore-drift.ts`
 * owns why; this file only has to paint them in that order.
 *
 * WHY CANVAS2D AND NOT THE WEBGL RENDERER'S OWN SPORE SYSTEM.
 *
 * `<mushroom-stage>` has a spore shader, and reaching for it here was the first
 * idea. It is the wrong one: the element's spores exist to fall around a body,
 * so using them means either mounting a fourth WebGL context with an invisible
 * mushroom in it — 137KB and a context, for a dozen dots — or teaching the
 * renderer a bodyless variant that nothing else would ever want. A dozen soft
 * radial sprites is a few hundred microseconds of Canvas2D per frame and no new
 * bytes at all, which is the correct price for decoration.
 *
 * WHY IT COVERS TWO WHOLE SECTIONS RATHER THAN JUST THE GAPS.
 *
 * A layer sized to the empty parts would need to know how tall they are, and
 * they are `padding-block: clamp(...)` on two different sections — a different
 * height at every viewport. So the layer is the whole band, and the sections'
 * own content covers what should not show: the doors plate, the terminal card
 * and the hardware switch all paint opaque paper at z-index 2 and hide every
 * mote behind them, leaving exactly the empty stretches and the margins beside
 * the copy. The gaps define themselves; nothing here has to be measured.
 *
 * WHY ONE LAYER AND NOT ONE PER SECTION.
 *
 * This started inside `.rb-doors` alone and stopped at the hairline above it —
 * a horizon halfway up a column of drifting spores, which air does not have.
 * Two instances would not fix it: two fields are two seeds, two densities agreed
 * on only by accident, and two independent recyclings, so the rule would still
 * be visible as the line where the picture changes. One field over a wrapper
 * that spans both sections is the only arrangement in which a mote can cross
 * that rule, and it is why `.rb-drift-band` exists in the markup.
 *
 * REDUCED MOTION GETS THE PICTURE, NOT A BLANK.
 *
 * Ambient movement is precisely what the setting exists to stop, so the loop
 * never starts. But the field is *drawn* once — a still frame of suspended
 * spores, which is a photograph rather than an animation, and is the same band
 * inhabited by the same organism. Removing it entirely would take something
 * away from the visitor who asked for less motion rather than less design.
 */

/*
 * Below this the layer is not drawn at all.
 *
 * The same 700px the hero organism, the ground colony and the preload script
 * use, for the same reason: at that width the copy occupies the full column, so
 * every mote would sit behind text being read rather than in a margin beside
 * it. A field that has to be faded to nothing to be acceptable is a field that
 * should not be there.
 */
const NARROW = "(max-width: 700px)";

/*
 * The two colours a mote is mixed from.
 *
 * `SPORE_RGB` is the accent — bronze on paper, per the brand tokens. `CORE_RGB`
 * is the warm near-white a backlit spore has at its centre. Each stop in
 * `SPECIES_PROFILE` carries a `tint` that mixes between them, so the species
 * differ in colour temperature as well as in shape: the glint is almost all
 * core, the halo's ring is pure bronze.
 */
const SPORE_RGB = [173, 122, 72] as const;
const CORE_RGB = [255, 244, 230] as const;

/*
 * Cap on the backing-store scale.
 *
 * The layer is a section-sized canvas of soft blobs with no edge anywhere in
 * it, so a 3× backing store on a phone-class GPU buys literally nothing
 * visible and costs nine times the fill. 1.25 is what the page's other ambient
 * canvas settled on.
 */
const MAX_DPR = 1.25;

/*
 * The sprite is drawn once per species at this size and then scaled down to
 * whatever the mote's diameter is. It has to be comfortably larger than the
 * biggest mote on screen (a near halo is about 37px) or every spore in the field
 * is a magnified low-resolution sprite — and the flare's arms are one pixel wide
 * at the waist, which is exactly the feature that dies first when a sprite is
 * undersampled. 128 is four times the largest disc and costs one 64KB canvas per
 * species, once, for the life of the page.
 */
const SPRITE = 128;

/** `stop.tint` mixed between the bronze and the warm core, as a CSS colour. */
function tintedRGB(tint: number): string {
  const mix = (from: number, to: number) => Math.round(from + (to - from) * tint);
  return [
    mix(SPORE_RGB[0], CORE_RGB[0]),
    mix(SPORE_RGB[1], CORE_RGB[1]),
    mix(SPORE_RGB[2], CORE_RGB[2]),
  ].join(",");
}

/**
 * One species' sprite, drawn once into an offscreen canvas and then blitted.
 *
 * The stops come from `SPECIES_PROFILE` rather than being written here, because
 * the coverage ceiling the layer is held to is a statement about those numbers
 * and a `CanvasGradient` cannot be measured. Every species is a radial falloff
 * that reaches zero at its rim and has no hard stop anywhere in it — a mote with
 * an edge is a dot, and a field of dots is a halftone screen, which is the exact
 * printed texture this brand was taken off.
 *
 * The sprite is drawn at full opacity and the per-mote alpha is applied by the
 * caller through `globalAlpha`, so one sprite per species serves the whole
 * field at every ink and every size.
 *
 * The canvas is `extent` times the disc so a species whose flare runs past its
 * own rim has somewhere to put it; the disc stays centred and keeps its own
 * radius, so nothing about a mote's size or ink changes because the box grew.
 */
function makeSprite(species: Species): HTMLCanvasElement {
  const sprite = document.createElement("canvas");
  sprite.width = sprite.height = SPRITE;
  const context = sprite.getContext("2d");
  if (!context) return sprite;
  const profile = SPECIES_PROFILE[species];
  const mid = SPRITE / 2;
  /* The disc occupies 1/extent of the sprite; the rest is room for the arms. */
  const radius = mid / sporeExtent(species);

  const gradient = context.createRadialGradient(mid, mid, 0, mid, mid, radius);
  for (const stop of profile.stops) {
    gradient.addColorStop(stop.offset, `rgba(${tintedRGB(stop.tint)},${stop.alpha})`);
  }
  context.fillStyle = gradient;
  context.fillRect(0, 0, SPRITE, SPRITE);

  /*
   * The flare, painted after the disc and only where a species declares one.
   *
   * Each arm is a triangle from the centre out to `reach`, filled with a
   * gradient that reaches zero alpha at the tip — so the arm tapers in width and
   * in ink at the same time and has no visible end. `lighter` composites it
   * additively, which is what makes the crossing point at the centre read as the
   * brightest part of the spark rather than as four overlapping strokes.
   *
   * The arms are drawn at half-offsets of π so a four-armed flare is an upright
   * cross rather than a saltire: a diagonal star is a decorative asset, an
   * upright one is a highlight.
   */
  const flare = profile.flare;
  if (flare) {
    context.globalCompositeOperation = "lighter";
    const reach = radius * flare.reach;
    const half = radius * flare.width;
    for (let arm = 0; arm < flare.arms; arm += 1) {
      const angle = (arm / flare.arms) * Math.PI * 2;
      context.save();
      context.translate(mid, mid);
      context.rotate(angle);
      /* Straight from `FLARE_TAPER`, which is the curve `flareInk` integrates —
         a gradient written by hand here would drift away from the measured cost
         of the arms with nothing to catch it. Its tints are near-bronze
         throughout, because an arm is a thin streak and a thin streak of
         near-white over ivory paper is nothing at all: the same
         annulus-around-a-hole failure the radial stops are kept low for, in the
         one place it would have been easiest to miss, since a flare is
         *supposed* to be the bright part. */
      const taper = context.createLinearGradient(0, 0, reach, 0);
      for (const stop of FLARE_TAPER) {
        taper.addColorStop(stop.offset, `rgba(${tintedRGB(stop.tint)},${flare.alpha * stop.alpha})`);
      }
      context.fillStyle = taper;
      context.beginPath();
      context.moveTo(0, -half);
      context.lineTo(reach, 0);
      context.lineTo(0, half);
      context.closePath();
      context.fill();
      context.restore();
    }
    context.globalCompositeOperation = "source-over";
  }
  return sprite;
}

export function SporeDrift() {
  const hostRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    if (window.matchMedia(NARROW).matches) return;

    const canvas = document.createElement("canvas");
    canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block;";
    const context = canvas.getContext("2d");
    if (!context) return;
    host.appendChild(canvas);

    /* One sprite per species, built once for the life of the layer. */
    const sprites = Object.fromEntries(SPECIES.map((species) => [species, makeSprite(species)])) as Record<
      Species,
      HTMLCanvasElement
    >;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let field: Spore[] = [];
    /* Indices into `field`, far stratum first. Recomputed only when the field's
       length changes: sorting a couple of hundred motes every frame is real work
       for an order that cannot change, since `depth` is fixed at birth. */
    let order: number[] = [];
    let width = 0;
    let height = 0;
    let frame = 0;
    let last = 0;
    let clock = 0;
    /* Off-screen frames are skipped rather than throttled: this is a decoration
       three sections deep, and there is no reason to composite it while the
       reader is somewhere else on the page. */
    let onScreen = false;

    const draw = () => {
      context.clearRect(0, 0, width, height);
      /* Far motes first, so a near one lands on top of the haze behind it rather
         than under it — which is the whole of the depth illusion, and it is free
         once the order is precomputed. */
      for (const index of order) {
        const spore = field[index];
        /* `order` is a permutation of `field`'s own indices, so this is only the
           path a stale order would take — one frame between a resize and the
           relayout. Skipping is the right answer there; throwing would take the
           page down for a decoration. */
        if (!spore) continue;
        const alpha = sporeAlpha(spore, clock);
        if (alpha <= 0.002) continue;
        context.globalAlpha = alpha;
        const x = sporeX(spore, clock) * width;
        const y = spore.y * height;
        /* `spore.size` is the diameter of the *disc*, and the sprite is that
           disc plus whatever room its flare needed. Painting the sprite at
           `size` would shrink the disc by the flare's reach and make a glint
           half the mote it is meant to be, so the box is scaled up by the same
           factor the sprite was and the disc lands at exactly `size`. */
        const box = spore.size * sporeExtent(spore.species);
        context.drawImage(sprites[spore.species], x - box / 2, y - box / 2, box, box);
      }
      context.globalAlpha = 1;
    };

    const layout = () => {
      const rect = host.getBoundingClientRect();
      const nextWidth = Math.max(2, Math.round(rect.width));
      const nextHeight = Math.max(2, Math.round(rect.height));
      if (nextWidth === width && nextHeight === height) return;
      width = nextWidth;
      height = nextHeight;
      const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      /* setTransform, not scale: the context keeps its transform across resizes
         and a second `scale` would multiply into the first, shrinking the field
         into the top-left corner on every reflow. */
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      const count = sporeCount(width, height);
      field = field.length ? refitSporeField(field, count) : makeSporeField(count);
      order = sporeDrawOrder(field);
      draw();
    };

    const tick = (now: number) => {
      frame = 0;
      /* Clamped: a background tab suspends rAF, and the first frame back would
         otherwise advance the field by however many seconds the reader spent
         elsewhere — every mote jumping at once. */
      const delta = last ? Math.min(0.05, (now - last) / 1000) : 0;
      last = now;
      clock += delta;
      field = field.map((spore) => riseSpore(spore, delta));
      draw();
      if (onScreen) frame = window.requestAnimationFrame(tick);
    };

    const observer = new IntersectionObserver((entries) => {
      onScreen = entries.some((entry) => entry.isIntersecting);
      if (reduced || !onScreen) {
        if (frame) window.cancelAnimationFrame(frame);
        frame = 0;
        last = 0;
        return;
      }
      if (!frame) frame = window.requestAnimationFrame(tick);
    });
    observer.observe(host);

    const resize = new ResizeObserver(layout);
    resize.observe(host);
    layout();

    return () => {
      observer.disconnect();
      resize.disconnect();
      if (frame) window.cancelAnimationFrame(frame);
      canvas.remove();
    };
  }, []);

  /* `rb-decor` opts this out of the stacking rule that turns every section
     child into a positioned content layer at z-index 2; see
     mycelium-network.css. It positions itself, below the content. */
  return <div className="rb-spore-drift rb-decor" ref={hostRef} aria-hidden="true" />;
}
