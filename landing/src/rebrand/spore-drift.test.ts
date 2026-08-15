import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import {
  alphaAtRadius,
  AREA_PER_SPORE,
  EDGE,
  FIELD_INK_CEILING,
  fieldInk,
  flareInk,
  FLARE_TAPER,
  MAX_INK,
  MAX_SPORES,
  MEAN_INK_CEILING,
  meanCoverage,
  MIN_INK,
  MIN_SPORES,
  makeSporeField,
  pickSpecies,
  refitSporeField,
  riseSpore,
  SPECIES,
  SPECIES_MIX,
  SPECIES_PROFILE,
  SPORE_SEED,
  sporeAlpha,
  sporeCount,
  sporeDrawOrder,
  sporeExtent,
  sporeFade,
  sporeMeanInk,
  sporeShimmer,
  sporeTotalInk,
  sporeX,
  TWINKLE_MAX,
  TWINKLE_MIN,
  type Spore,
} from "./spore-drift";

/*
 * The drifting spores over the local strip and the three doors.
 *
 * Every way this can go wrong is silent, which is why it is tested at all. Ink
 * that creeps up past the page's hairline turns air into a dot screen over the
 * heading; a wrap at the box edge instead of past it pops every mote in and out
 * of existence in plain view; a resize that rebuilds the field teleports the
 * whole layer whenever a mobile URL bar collapses; a field seeded from the clock
 * is a different picture on every load and cannot be reviewed. None of them
 * throw and none of them fail a render test, so the numbers are asserted here.
 */

const read = (path: string) => readFile(new URL(path, import.meta.url), "utf8");

/*
 * Source with its prose removed.
 *
 * Every "this file must not mention X" assertion below has to read the code and
 * not the comments, because the comments are where the reasoning lives and the
 * reasoning names the thing being ruled out — the note in SporeDrift.tsx
 * explaining why it does *not* reach for `<mushroom-stage>`'s spore shader
 * contains the string a naive match would trip on. Stripping first keeps the
 * explanation and the assertion from being in each other's way.
 */
const code = (source: string) => source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");

describe("the field is the same picture on every load", () => {
  it("generates identical geometry from the same seed", () => {
    expect(makeSporeField(20)).toEqual(makeSporeField(20));
  });

  it("does not share a seed with the page's other two fields", async () => {
    /* Three fields drawn from one seed put their motes in correlated columns,
       and on a page that runs all three at once that correlation is visible. */
    const network = await read("./vendor/mycelium-network.js");
    const colony = await read("./HeroGroundColony.tsx");
    expect(network).not.toContain(String(SPORE_SEED));
    expect(colony).not.toContain(String(SPORE_SEED));
    expect(network).toContain("20260811");
    expect(network).toContain("20260817");
  });
});

/** The layer's own opacity, as the browser will apply it to every mote. */
const layerOpacity = async () => {
  const css = await read("./rebrand.css");
  const layer = css.slice(css.indexOf(".rb-spore-drift { opacity"));
  const value = layer.match(/opacity:\s*(\.\d+|\d+(?:\.\d+)?)/)![1];
  return Number(value);
};

/** `--rb-line`, the hairline every plate on this page is ruled with. */
const hairline = async () => {
  const tokens = await read("../brand-tokens.css");
  return Number(tokens.match(/--rb-line:\s*rgba\([^)]*?([\d.]+)\)/)![1]);
};

describe("the layer stays quieter than the rules the page is drawn with", () => {
  it("bounds what the whole field puts on the paper, not just what one mote does", async () => {
    /*
     * The rule that governs, and the one a per-mote ceiling cannot express.
     *
     * The band was asked for more spores and brighter ones, and both were given:
     * the density went up by a fifth and the ink range up by a quarter. Every
     * per-mote assertion in this file would have passed either change alone and
     * both together, because a ceiling on one mote cannot see how many there
     * are — doubling the count while halving each mote's ink passes it exactly
     * as well as the honest version does, and puts the same bronze on the paper.
     *
     * So the governing figure is the fraction of the band the field darkens:
     * each mote's mean coverage over its own disc, summed, over the area of the
     * box. That is what a reader perceives as "how strong is this layer", and it
     * is measured at the field's peak — the fade and the shimmer only subtract.
     */
    const opacity = await layerOpacity();
    for (const [width, height] of [
      [1200, 1000],
      [1440, 1150],
      [1920, 1300],
      [2560, 1400],
    ] as const) {
      const field = makeSporeField(sporeCount(width, height));
      expect(fieldInk(field, width, height) * opacity).toBeLessThanOrEqual(FIELD_INK_CEILING);
    }
  });

  it("holds the field's weight steady across viewports rather than letting it drift", () => {
    /* The density is derived from the area precisely so that a 27" display gets
       the same *picture* as a laptop rather than the same count. If the total
       ink moved with the viewport, that derivation would not be doing its job
       and the ceiling above would be a statement about one screen size. */
    const measure = (width: number, height: number) =>
      fieldInk(makeSporeField(sporeCount(width, height)), width, height);
    const laptop = measure(1440, 1150);
    for (const [width, height] of [
      [1200, 1000],
      [1920, 1300],
      [2560, 1400],
      [3840, 1500],
    ] as const) {
      expect(measure(width, height)).toBeGreaterThan(laptop * 0.8);
      expect(measure(width, height)).toBeLessThan(laptop * 1.2);
    }
  });

  it("keeps any single mote from becoming a blot, at a hairline's own order of weight", async () => {
    /*
     * The local rule under the field-wide one.
     *
     * `--rb-line` is not a threshold this can be held under — a mote is a point
     * and a hairline is a line across a 1200px plate, so they were never the same
     * quantity, and pinning one to the other is what kept this layer invisible
     * for two revisions. What it is good for is *scale*: a piece of scenery whose
     * individual marks average several times the page's own rules is a dot screen
     * whatever the field total says. So the ceiling is asserted as a ratio against
     * the token rather than as an inequality, which keeps the two files moving
     * together without pretending they measure the same thing.
     */
    const line = await hairline();
    const opacity = await layerOpacity();
    expect(line).toBeGreaterThan(0);
    expect(opacity).toBeGreaterThan(0);
    expect(MEAN_INK_CEILING).toBeGreaterThan(line * 0.8);
    expect(MEAN_INK_CEILING).toBeLessThan(line * 1.5);

    const field = makeSporeField(MAX_SPORES);
    for (const spore of field) expect(sporeMeanInk(spore) * opacity).toBeLessThanOrEqual(MEAN_INK_CEILING);
  });

  it("counts a doubled field as twice the ink, which is the point of measuring it this way", () => {
    /* The property that makes `fieldInk` the right quantity: it is linear in the
       count and in the ink, so neither knob can be turned up behind the other's
       back. A test that only ever saw the shipped numbers would not notice if it
       were measuring a mean instead of a total. */
    const field = makeSporeField(200);
    expect(fieldInk([...field, ...field], 1440, 1150)).toBeCloseTo(fieldInk(field, 1440, 1150) * 2, 10);
    expect(fieldInk(field, 1440, 1150)).toBeGreaterThan(0);
    /* And it is a fraction of the box, so the same field in a bigger box is
       lighter — which is why the count has to come from the area. */
    expect(fieldInk(field, 2880, 1150)).toBeCloseTo(fieldInk(field, 1440, 1150) / 2, 10);
    expect(fieldInk(field, 0, 0)).toBe(0);
  });

  it("does not compound three separate dimmers into a field nobody can see", async () => {
    /*
     * The regression this whole file was rewritten around.
     *
     * The layer shipped invisible: a peak ink of 0.26, times a sprite core stop
     * of 0.62, times a layer opacity of 0.58, times the falloff's own average —
     * about 2% ink over `--rb-paper`, which measured off a screenshot of the
     * live section came back within two levels of the background across the
     * whole band. Every one of those factors was individually defended in a
     * comment as "barely anything". None of them failed a test.
     *
     * So the weight is asserted where it lands — on the paper — rather than
     * factor by factor, and it is asserted from below as well as from above.
     * `--rb-line` is a visible hairline on this page; a field whose near motes
     * cover a reasonable fraction of that is a field that is actually there.
     */
    const line = await hairline();
    const opacity = await layerOpacity();
    const field = makeSporeField(MAX_SPORES);
    const near = field.filter((spore) => spore.depth > 0.75);
    expect(near.length).toBeGreaterThan(4);

    const strongest = Math.max(...near.map((spore) => sporeMeanInk(spore) * opacity));
    expect(strongest).toBeGreaterThan(line * 0.35);

    /* And the brightest single pixel in the field has to be visible against the
       paper it sits on, which is the check the old peak-under-hairline rule
       replaced with its opposite. Ivory is rgb(238,233,226); a bronze at 30%+
       over it is a difference of ~20 levels, which the eye reads easily. */
    const peakPixel = Math.max(...field.map((spore) => spore.ink)) * opacity;
    expect(peakPixel).toBeGreaterThan(0.2);

    /* The floor from below on the field as a whole, and the half of the rule the
       brightening request is actually about: a band that darkens less than a
       twentieth of a percent of its own paper is a band nobody notices is
       decorated. Paired with FIELD_INK_CEILING above, this brackets the layer
       rather than only capping it — which is what the invisible version was
       missing. */
    const band = makeSporeField(sporeCount(1440, 1150));
    expect(fieldInk(band, 1440, 1150) * opacity).toBeGreaterThan(0.0005);
  });

  it("never draws a mote so faint it is a smudge rather than a spore", () => {
    /* The other half of the same rule, and the one the earlier version of this
       whole line of work got wrong: subordination is low contrast, not
       invisibility. A field nobody can see is not a subtler field. */
    expect(MIN_INK).toBeGreaterThan(0.1);
    const field = makeSporeField(MAX_SPORES);
    for (const spore of field) {
      expect(spore.ink).toBeGreaterThan(0);
      expect(spore.ink).toBeLessThanOrEqual(MAX_INK);
    }
  });

  it("measures coverage over the disc and not along a line", () => {
    /* A plain average of the stops would flatter every soft sprite enormously: a
       ring at 0.9 of the radius covers nine times the paper of one at 0.1, so
       the weighting is the difference between a real coverage figure and one
       that says a halo is as heavy as its own bright centre. */
    const flat = (stops: readonly { offset: number; alpha: number }[]) => {
      let sum = 0;
      for (let i = 0; i < 4096; i += 1) sum += alphaAtRadius(stops as never, (i + 0.5) / 4096);
      return sum / 4096;
    };
    for (const species of SPECIES) {
      const stops = SPECIES_PROFILE[species].stops;
      expect(meanCoverage(stops)).toBeLessThan(flat(stops));
      expect(meanCoverage(stops)).toBeGreaterThan(0);
    }
    /* Stable under sampling: a figure that moves with the sample count is not a
       property of the sprite. */
    const mote = SPECIES_PROFILE.mote.stops;
    expect(meanCoverage(mote, 512)).toBeCloseTo(meanCoverage(mote, 8192), 3);
  });
});

describe("the count comes from the area, not from a guess", () => {
  it("fills the visible band at every desktop width", () => {
    /*
     * The band is roughly 1,000–1,400px tall across the viewport heights the two
     * clamps cover. What a reader actually sees of this layer is only the part
     * the plates do not cover — the doors grid, the terminal card and the
     * hardware switch all paint opaque paper over it — which is a little under
     * half the band's height. That visible fraction is what has to be filled, so
     * it is what is asserted; counting the whole band would let the field thin
     * out to four visible motes while the total still looked healthy.
     *
     * The floor is 20 rather than the 8 it was: "fill this section with spores"
     * is the request, and a dozen motes over 1400px of paper is a scatter.
     *
     * There is no upper bound on the *count* here, because a count is the wrong
     * thing to bound — 60 visible motes on a laptop and 130 on a 27" display are
     * the same picture, which is the entire point of deriving the count from the
     * area. What has to stay in range is the spacing, and that is asserted on its
     * own below.
     */
    const VISIBLE = 0.4;
    for (const width of [960, 1200, 1440, 1920, 2560]) {
      for (const height of [1000, 1150, 1400]) {
        expect(sporeCount(width, height) * VISIBLE).toBeGreaterThan(20);
      }
    }
  });

  it("puts the motes the same distance apart at every viewport", () => {
    /*
     * The figure that actually decides how the layer reads, and the reason the
     * constant is an area rather than a count: mean spacing is √(area/count), so
     * a density-derived count holds it fixed while a fixed count would let it run
     * from 55px on a laptop to 105px on a 27" display — a filled band at one size
     * and a thin scatter at the other, which is the failure this test exists to
     * catch.
     *
     * 65–90px is the window the rendered field was tuned in: below it the motes
     * start to touch at the near stratum, above it the paper opens up between
     * them and the band reads as dusted rather than inhabited. It was 70–95 and
     * came down with the density, because the band was asked for more spores in
     * it — what matters is the width of the window, not where it sits.
     */
    for (const width of [960, 1200, 1440, 1920, 2560, 3840]) {
      for (const height of [1000, 1150, 1400, 1500]) {
        const spacing = Math.sqrt((width * height) / sporeCount(width, height));
        expect(spacing).toBeGreaterThan(65);
        expect(spacing).toBeLessThan(90);
      }
    }
  });

  it("leaves the density in charge at every viewport anyone has", () => {
    /* The clamps are runaway guards, not working limits. If either one binds at
       a real size the density stops governing and every display gets the same
       field — which is the failure the area-based count exists to prevent. */
    for (const width of [960, 1200, 1440, 1920, 2560, 3840]) {
      for (const height of [1000, 1150, 1400]) {
        const raw = Math.round((width * height) / AREA_PER_SPORE);
        expect(raw).toBeGreaterThan(MIN_SPORES);
        expect(raw).toBeLessThan(MAX_SPORES);
        expect(sporeCount(width, height)).toBe(raw);
      }
    }
  });

  it("scales with the box instead of thinning out on a large display", () => {
    const small = sporeCount(1000, 1150);
    const large = sporeCount(2400, 1150);
    expect(large).toBeGreaterThan(small);
    expect(sporeCount(2400, 1150)).toBe(Math.min(MAX_SPORES, Math.round((2400 * 1150) / AREA_PER_SPORE)));
  });

  it("never asks for a mote in a box with no area", () => {
    expect(sporeCount(0, 0)).toBe(MIN_SPORES);
    expect(sporeCount(-100, 700)).toBe(MIN_SPORES);
  });
});

describe("no part of the width is left empty", () => {
  it("stratifies the lanes so the field cannot clump", () => {
    const count = 24;
    const field = makeSporeField(count);
    const lanes = field.map((spore) => spore.lane);

    /* One mote per equal column, in order. A uniform draw reliably leaves a
       quarter of a 24-mote field's width bare, and over a 1200px band that
       empty quarter is the most visible thing in the layer. */
    for (let i = 0; i < count; i += 1) {
      expect(lanes[i]!).toBeGreaterThanOrEqual(i / count);
      expect(lanes[i]!).toBeLessThan((i + 1) / count);
    }
    expect(Math.min(...lanes)).toBeLessThan(0.05);
    expect(Math.max(...lanes)).toBeGreaterThan(0.95);
  });

  it("keeps the sway small enough that a mote stays in its own part of the band", () => {
    /* The sway is life, not travel. A mote that crosses several lanes turns the
       field into drifting weather and undoes the stratification above. */
    const field = makeSporeField(24);
    for (const spore of field) {
      const samples = Array.from({ length: 64 }, (_, i) => sporeX(spore, i * 0.5));
      const excursion = Math.max(...samples) - Math.min(...samples);
      expect(excursion).toBeLessThan(2 / 24);
      /* And it has to actually move: a zero-amplitude column of rising dots is
         a progress bar, not spores in air. */
      expect(excursion).toBeGreaterThan(0.004);
    }
  });

  it("curves rather than swinging, so no mote retraces its own path", () => {
    /*
     * The clearest single difference between this field and the hero's, and the
     * reason `sporeX` has two harmonics.
     *
     * A single sine is a pendulum: the mote is at the same x every period, for
     * as long as it is on screen. In a field this dense a hundred pendulums read
     * as a mechanism even at different periods. Asserted as a comparison against
     * that exact alternative rather than against a number picked to pass: the
     * one-harmonic version repeats to within a hair, the two-harmonic one does
     * not come back to where it was.
     */
    const spore = makeSporeField(12)[0]!;
    const period = (2 * Math.PI) / spore.swayRate;
    const single = (t: number) => spore.lane + Math.sin(t * spore.swayRate + spore.phase) * spore.sway;

    let repeat = 0;
    let drift = 0;
    for (let cycle = 1; cycle <= 6; cycle += 1) {
      const t = cycle * period;
      repeat = Math.max(repeat, Math.abs(single(t) - single(0)));
      drift = Math.max(drift, Math.abs(sporeX(spore, t) - sporeX(spore, 0)));
    }
    expect(repeat).toBeLessThan(1e-9);
    expect(drift).toBeGreaterThan(spore.sway * 0.2);
  });
});

describe("the field is a colony and not one shape repeated", () => {
  it("draws three species, in a mix mostly of plain motes", () => {
    const field = makeSporeField(300);
    const share = (species: string) => field.filter((spore) => spore.species === species).length / field.length;
    for (const species of SPECIES) expect(share(species)).toBeGreaterThan(0.05);
    /* Mostly plain motes: the halo and the glint are what a reader notices, and a
       field where a third of the motes are sparks is a christmas light rather
       than a spore release. */
    expect(share("mote")).toBeGreaterThan(0.5);
    expect(share("glint")).toBeLessThan(0.25);
  });

  it("assigns a species from the roll alone, so a resize cannot shift the mix", () => {
    /* Cumulative thresholds against one draw of the RNG rather than a counter:
       an index-based mix would give a mote a different shape after a refit
       changed the count. */
    expect(pickSpecies(0)).toBe("mote");
    expect(pickSpecies(0.999_999)).toBe("glint");
    for (const entry of SPECIES_MIX) expect(SPECIES).toContain(entry.species);
    expect(SPECIES_MIX.at(-1)!.upTo).toBe(1);
    /* Every roll lands somewhere: a gap in the thresholds would silently fall
       through to the last species. */
    for (let i = 0; i < 1000; i += 1) expect(SPECIES).toContain(pickSpecies(i / 1000));
  });

  it("gives each species a falloff that reaches zero at the rim", () => {
    /* A sprite with any coverage at its own edge draws a disc with a visible
       boundary, and a field of those is the halftone screen this brand was taken
       off the back of. */
    for (const species of SPECIES) {
      const stops = SPECIES_PROFILE[species].stops;
      expect(stops.at(-1)!.offset).toBe(1);
      expect(stops.at(-1)!.alpha).toBe(0);
      expect(alphaAtRadius(stops, 1)).toBe(0);
      expect(alphaAtRadius(stops, 0.999)).toBeLessThan(0.06);
      /* Monotonic offsets, or the interpolation walks backwards. */
      for (let i = 1; i < stops.length; i += 1) expect(stops[i]!.offset).toBeGreaterThan(stops[i - 1]!.offset);
      /* No stop brighter than the mote's own ink, which `MAX_INK` is the ceiling
         on — a species multiplying past 1 would make that ceiling a lie. */
      for (const stop of stops) expect(stop.alpha).toBeLessThanOrEqual(1);
      expect(SPECIES_PROFILE[species].ink).toBeLessThanOrEqual(1);
    }
  });

  it("makes the halo a ring rather than a brighter mote", () => {
    /* The ring is the whole species: coverage has to rise again away from the
       centre, which is what reads as a spore off the plane of focus. */
    const stops = SPECIES_PROFILE.halo.stops;
    const trough = alphaAtRadius(stops, 0.36);
    expect(alphaAtRadius(stops, 0.58)).toBeGreaterThan(trough);
    /* And it is larger and lighter per unit area than a plain mote, or it would
       simply be a heavier one. */
    expect(SPECIES_PROFILE.halo.size).toBeGreaterThan(SPECIES_PROFILE.mote.size);
    expect(meanCoverage(stops)).toBeLessThan(meanCoverage(SPECIES_PROFILE.mote.stops));
  });

  it("makes the glint small and bright rather than a small mote", () => {
    expect(SPECIES_PROFILE.glint.size).toBeLessThan(SPECIES_PROFILE.mote.size * 0.7);
    expect(meanCoverage(SPECIES_PROFILE.glint.stops)).toBeGreaterThan(meanCoverage(SPECIES_PROFILE.mote.stops));
  });

  it("keeps the warm core a hint rather than the hero's half-white mix", async () => {
    /*
     * The hero's shader mixes its accent halfway to #fff2e4 and adds it, which is
     * right on a dark stage and wrong here: `--rb-paper` is near-white, so a
     * near-white centre is *less* ink than the paper under it and the mote comes
     * out as a bronze annulus around a hole. The tint has to stay low enough that
     * bronze carries the shape.
     */
    const tokens = await read("../brand-tokens.css");
    expect(tokens).toContain("--rb-paper");
    for (const species of SPECIES) {
      const stops = SPECIES_PROFILE[species].stops;
      /* Only the centre may carry warmth, and never as much as the hero's 0.5
         except on the glint, which is small enough to read as a highlight. */
      const limit = species === "glint" ? 0.7 : 0.4;
      for (const stop of stops) expect(stop.tint).toBeLessThanOrEqual(limit);
      /* Outside the core it is the accent and nothing else. */
      expect(alphaAtRadius(stops, 0.5)).toBeGreaterThan(0);
      expect(stops.filter((stop) => stop.offset >= 0.5).every((stop) => stop.tint === 0)).toBe(true);
    }
  });

  it("gives every species internal structure, so a mote is not a blur", () => {
    /*
     * The request that produced this was "more detail, like they are shining",
     * and the failure it named is real: three radial gradients are three sizes of
     * the same soft blur, and a soft blur at 12px is a speck of dust. What makes
     * a small shape read as an object is a *step* in its falloff — somewhere the
     * ink drops fast enough that the eye finds a boundary — and no amount of
     * extra ink substitutes for it.
     *
     * So every species has to have a shoulder, measured as a *rate*: somewhere
     * in the falloff, alpha has to drop by more than 2.5 core-alphas per unit of
     * radius. A profile that decays smoothly from centre to rim passes every ink
     * assertion in this file and renders as a thumbprint.
     */
    for (const species of SPECIES) {
      const stops = SPECIES_PROFILE[species].stops;
      const core = stops[0]!.alpha;
      let steepest = 0;
      for (let i = 1; i < stops.length; i += 1) {
        const from = stops[i - 1]!;
        const to = stops[i]!;
        const span = to.offset - from.offset;
        if (span <= 0) continue;
        steepest = Math.max(steepest, (from.alpha - to.alpha) / span / core);
      }
      expect(steepest).toBeGreaterThan(2.5);
    }
  });

  it("makes the halo's ring legible as a ring rather than as a bump", () => {
    /* The species exists to read as a spore out of focus, which needs two
       visible rings — so the gap between the core and the ring has to be a real
       trough. If the ring is not at least three times the gap, the whole shape
       collapses back into one soft falloff and this species stops differing from
       the mote at all. */
    const stops = SPECIES_PROFILE.halo.stops;
    const gap = alphaAtRadius(stops, 0.34);
    const ring = alphaAtRadius(stops, 0.56);
    expect(ring).toBeGreaterThan(gap * 3);
    /* And the ring must not out-ink the core, or it is a donut rather than a
       spore. */
    expect(ring).toBeLessThan(alphaAtRadius(stops, 0));
  });

  it("puts a flare only on the glint, and keeps it a catch of light", () => {
    /*
     * The flare is what actually answers "shining": a disc cannot read as lit
     * however bright it is, because what the eye recognises as a highlight is the
     * streak, not the point. It belongs to exactly one species — arms on the bulk
     * of the field would be a sky full of stars, which is a different picture
     * from a spore release.
     */
    expect(SPECIES_PROFILE.mote.flare).toBeUndefined();
    expect(SPECIES_PROFILE.halo.flare).toBeUndefined();
    const flare = SPECIES_PROFILE.glint.flare!;
    expect(flare).toBeDefined();
    /* Upright cross, not a starburst: past four arms it stops being a highlight
       and becomes a sparkle asset. */
    expect(flare.arms).toBe(4);
    /* The arms have to run past the disc — a spike that stopped at the rim is
       just a lumpy dot — but not so far that a glint becomes the biggest thing in
       the field. The disc is 0.52 of a mote, so 1.85 radii of arm still lands
       inside a plain mote's own footprint. */
    expect(flare.reach).toBeGreaterThan(1.2);
    expect(flare.reach).toBeLessThan(2.5);
    /*
     * Thin, but bounded from *below* as well, and that lower bound is the one
     * that was learned the hard way. `width` is a fraction of the radius and the
     * radius here is about 5px, so the first value — 0.03 — asked for an arm a
     * fifth of a pixel wide at its waist, tapering from there. It measured as ink
     * in `flareInk` and rendered as literally nothing: geometry below a pixel is
     * not faint, it is absent, and no ceiling in this file could tell the
     * difference. Anything under about a tenth of the radius is back in that
     * regime.
     *
     * The upper bound is the original concern and still holds: a wide bright
     * cross is a stock-pack sparkle, and what is wanted is a catch of light.
     */
    expect(flare.width).toBeGreaterThan(0.1);
    expect(flare.width).toBeLessThan(0.3);
  });

  it("counts the flare's ink rather than letting it ride for free", () => {
    /*
     * Ink the ceiling cannot see is ink that grows. The flare is drawn additively
     * and outside the disc, so it is precisely the kind of decoration that gets
     * added to, brightened, and widened over successive passes without a single
     * assertion in this file ever noticing — until the band is a field of crosses.
     */
    expect(flareInk(undefined)).toBe(0);
    const flare = SPECIES_PROFILE.glint.flare!;
    expect(flareInk(flare)).toBeGreaterThan(0);
    /* The arms cost about what the disc does, which is the honest figure for a
       shape that is mostly flare — but they must not run away with the species:
       past about 1.5× the disc the glint has become a cross with a dot in the
       middle rather than a spore catching light. */
    const disc = meanCoverage(SPECIES_PROFILE.glint.stops);
    expect(flareInk(flare)).toBeLessThan(disc * 1.5);
    /* And it has to scale with the geometry, or it is a constant wearing the
       costume of a measurement. */
    expect(flareInk({ ...flare, alpha: flare.alpha * 2 })).toBeCloseTo(flareInk(flare) * 2, 10);
    expect(flareInk({ ...flare, reach: flare.reach * 2 })).toBeCloseTo(flareInk(flare) * 2, 10);
    expect(flareInk({ ...flare, arms: flare.arms * 2 })).toBeCloseTo(flareInk(flare) * 2, 10);
  });

  it("integrates the taper the painter actually draws", () => {
    /*
     * The arms taper in width *and* in alpha at the same time, and the first
     * version of `flareInk` multiplied the two means as though they were
     * independent — it overstated the flare by about two thirds. That is the
     * benign direction of the error, but a measurement that is wrong in a safe
     * direction is still a measurement nobody can tune against.
     *
     * So the taper is data, shared with the painter, and the integral is over the
     * curve rather than over an idealisation of it. Sanity: a flare whose alpha
     * curve is flat at 1 must cost strictly more than the real one, which decays.
     */
    const flare = SPECIES_PROFILE.glint.flare!;
    expect(FLARE_TAPER[0]!.alpha).toBe(1);
    expect(FLARE_TAPER[FLARE_TAPER.length - 1]!.alpha).toBe(0);
    /* Fades to nothing at the tip, so an arm has no visible end — the same rule
       every radial profile in this file is held to. */
    expect(alphaAtRadius(FLARE_TAPER, 0.99)).toBeLessThan(0.05);
    /* And it stays bronze along its length: a pale streak on ivory is invisible,
       which is precisely how the first flare shipped as nothing at all. */
    for (const stop of FLARE_TAPER) expect(stop.tint).toBeLessThan(0.25);
    /* The integral converges — a sample count that changed the answer would mean
       the figure the ceiling is checked against depends on an argument nobody
       passes. */
    expect(flareInk(flare, 256)).toBeCloseTo(flareInk(flare, 4096), 4);
  });

  it("charges the flare to the page's weight but not to the blot rule", () => {
    /*
     * Two ceilings, two questions, and the flare belongs to only one of them.
     *
     * MEAN_INK_CEILING asks "is this one mote a dot on the page" — a question
     * about how concentrated a mark is over the area it occupies. Folding four
     * thin arms that run to twice the radius into an average over the *disc*
     * charges a spread-out mark at a concentrated mark's rate, and it briefly
     * made the glint measure as the heaviest species in a field where it is the
     * lightest thing. FIELD_INK_CEILING asks "how much bronze is on this paper",
     * and there every drawn pixel counts, arms included.
     */
    const field = makeSporeField(400);
    const glint = field.find((spore) => spore.species === "glint")!;
    const mote = field.find((spore) => spore.species === "mote")!;
    expect(sporeMeanInk(glint)).toBeLessThan(sporeTotalInk(glint));
    /* An unflared species has nothing to separate the two measures. */
    expect(sporeMeanInk(mote)).toBeCloseTo(sporeTotalInk(mote), 12);
    /* The field's weight counts the arms: dropping them has to make the number
       fall, or `fieldInk` is reading the blot measure and the arms are free. */
    const withArms = fieldInk(field, 1440, 1150);
    const discsOnly = field.reduce((sum, spore) => {
      const radius = spore.size / 2;
      return sum + sporeMeanInk(spore) * Math.PI * radius * radius;
    }, 0) / (1440 * 1150);
    expect(withArms).toBeGreaterThan(discsOnly);
  });

  it("gives a flared sprite room to draw its arms without clipping them", () => {
    /*
     * The sprite is a square canvas and the arms run past the disc's rim, so the
     * box has to grow by the same factor or every arm ends in a hard vertical
     * edge at the canvas boundary — four clipped stubs, which is both the
     * opposite of the intended shape and the kind of artefact nobody can explain
     * later. `sporeExtent` is that factor, and it is never below 1 so an
     * unflared species keeps painting at exactly its own diameter.
     */
    expect(sporeExtent("mote")).toBe(1);
    expect(sporeExtent("halo")).toBe(1);
    expect(sporeExtent("glint")).toBe(SPECIES_PROFILE.glint.flare!.reach);
    for (const species of SPECIES) expect(sporeExtent(species)).toBeGreaterThanOrEqual(1);
  });
});

describe("depth is what lets the field be dense without being a texture", () => {
  it("ties size, speed, sway and ink to one axis", () => {
    /* A near mote has to be bigger, faster and stronger *together*. Any one of
       these varying independently makes the layer a flat plane of dots with
       random attributes rather than air with volume. */
    const field = makeSporeField(400).filter((spore) => spore.species === "mote");
    const near = field.filter((spore) => spore.depth > 0.8);
    const far = field.filter((spore) => spore.depth < 0.2);
    expect(near.length).toBeGreaterThan(3);
    expect(far.length).toBeGreaterThan(3);

    const mean = (list: Spore[], pick: (spore: Spore) => number) =>
      list.reduce((sum, spore) => sum + pick(spore), 0) / list.length;
    expect(mean(near, (s) => s.size)).toBeGreaterThan(mean(far, (s) => s.size));
    expect(mean(near, (s) => s.rise)).toBeGreaterThan(mean(far, (s) => s.rise));
    expect(mean(near, (s) => s.sway)).toBeGreaterThan(mean(far, (s) => s.sway));
    expect(mean(near, (s) => s.ink)).toBeGreaterThan(mean(far, (s) => s.ink));
  });

  it("crowds the far stratum rather than spreading the depths evenly", () => {
    /* Squared draw. A flat distribution puts as many 18px motes in the band as
       6px ones, which is the dot-screen failure arriving by another door: most of
       what fills a dense field has to be small and faint. */
    const field = makeSporeField(400);
    const far = field.filter((spore) => spore.depth < 0.5).length / field.length;
    expect(far).toBeGreaterThan(0.6);
  });

  it("paints far motes first so a near one lands on top of the haze", () => {
    const field = makeSporeField(120);
    const order = sporeDrawOrder(field);
    expect(order).toHaveLength(field.length);
    /* Every mote exactly once — a sort that dropped or duplicated an index would
       simply lose motes from the field. */
    expect(new Set(order).size).toBe(field.length);
    for (let i = 1; i < order.length; i += 1) {
      expect(field[order[i]!]!.depth).toBeGreaterThanOrEqual(field[order[i - 1]!]!.depth);
    }
  });

  it("keeps the field's own order intact, because a refit depends on the prefix", () => {
    /* The draw order is a separate list rather than a sorted field for exactly
       this reason: sorting in place would make "the first N motes" a different
       set after every resize, and `refitSporeField`'s whole contract is that the
       prefix survives. */
    const field = makeSporeField(40);
    sporeDrawOrder(field);
    expect(field).toEqual(makeSporeField(40));
  });
});

describe("the shimmer keeps a dense field from reading as a fixed pattern", () => {
  it("only ever dims a mote, never brightens it past its ink", () => {
    /* `MAX_INK` is the ceiling on the brightest pixel in the field. A shimmer
       centred on 1 instead of bounded above by it would put every assertion
       about the layer's weight out by the shimmer's amplitude. */
    const field = makeSporeField(60);
    for (const spore of field) {
      for (let t = 0; t < 40; t += 0.37) {
        const shimmer = sporeShimmer(spore, t);
        expect(shimmer).toBeLessThanOrEqual(1 + 1e-9);
        expect(shimmer).toBeGreaterThanOrEqual(1 - TWINKLE_MAX - 1e-9);
      }
      expect(sporeAlpha(spore, 0)).toBeLessThanOrEqual(spore.ink);
    }
  });

  it("breathes slowly enough to be air and hard enough to be seen", () => {
    const field = makeSporeField(MAX_SPORES);
    for (const spore of field) {
      expect(spore.twinkle).toBeGreaterThanOrEqual(TWINKLE_MIN);
      expect(spore.twinkle).toBeLessThanOrEqual(TWINKLE_MAX);
      /* Seven to twenty seconds a cycle: slower than anything a reader tracks
         deliberately, faster than never. Past a third of the ink it blinks,
         which is an event rather than air — that bound is TWINKLE_MAX. */
      const seconds = 1 / spore.twinkleRate;
      expect(seconds).toBeGreaterThan(6);
      expect(seconds).toBeLessThan(21);
    }
    expect(TWINKLE_MAX).toBeLessThan(0.34);
  });

  it("does not breathe in unison", () => {
    /* The point is that motes come forward and fall back independently. A field
       sharing one phase pulses as a single object, which is a heartbeat. */
    const field = makeSporeField(80);
    const at = field.map((spore) => sporeShimmer(spore, 3.1));
    const spread = Math.max(...at) - Math.min(...at);
    expect(spread).toBeGreaterThan(0.15);
  });
});

describe("the rise is slow enough to read as air", () => {
  it("moves at the speed a reader sees, which is pixels and not fractions", () => {
    /*
     * The assertion that had to change when the band did, and the reason it is
     * now written in pixels.
     *
     * `rise` is a fraction of the box per second, and the box grew from one
     * section to two. Held at the old fractions, every mote would have sped up by
     * half without a line of the module changing — and this test, phrased as
     * "seconds to cross", would have gone on passing, because a mote crossing a
     * band half again as tall in the same time is exactly what it was checking
     * for. What a reader perceives is neither of those: it is pixels per second.
     *
     * 6–19px/s is a mote that drifts. Below about 4 the field looks frozen within
     * a visit; above about 25 it is snow, and the eye follows it instead of the
     * copy beside it.
     */
    const BAND = 1150;
    const field = makeSporeField(MAX_SPORES);
    for (const spore of field) {
      const pixelsPerSecond = spore.rise * BAND;
      expect(pixelsPerSecond).toBeGreaterThan(4);
      expect(pixelsPerSecond).toBeLessThan(25);
    }
  });

  it("takes at least a minute to cross the whole band", () => {
    const field = makeSporeField(MAX_SPORES);
    for (const spore of field) {
      const seconds = 1 / spore.rise;
      expect(seconds).toBeGreaterThan(55);
      /* And not so slow that the field looks frozen — a mote that takes five
         minutes to cross has not moved within any one reader's visit. */
      expect(seconds).toBeLessThan(180);
    }
  });

  it("rises rather than falls", () => {
    const spore = makeSporeField(4)[0]!;
    /* y grows downward, so rising is y decreasing. Getting this backwards is
       falling snow, which is a different weather system entirely. */
    expect(riseSpore(spore, 1).y).toBeLessThan(spore.y);
  });

  it("cannot be jumped forward by a suspended tab", () => {
    const spore = makeSporeField(4)[0]!;
    /* The caller clamps the delta; this only has to not run backwards on a
       negative one, which a clock adjustment can produce. */
    expect(riseSpore(spore, -10).y).toBe(spore.y);
  });
});

describe("the recycling happens where nobody can see it", () => {
  it("wraps past the edge and not on it", () => {
    const field = makeSporeField(8);
    const spore: Spore = { ...field[0]!, y: -EDGE + 1e-9, rise: 1 };
    const wrapped = riseSpore(spore, 1);
    expect(wrapped.y).toBeCloseTo(1 + EDGE, 6);
  });

  it("has already faded a mote to nothing by the time it is moved", () => {
    /* This is the assertion that matters. If the fade were a linear ramp to the
       box edge, a mote at the recycling line would still be at 18% ink and
       every wrap would be a dot blinking out on the top border. */
    expect(sporeFade(-EDGE)).toBe(0);
    expect(sporeFade(1 + EDGE)).toBe(0);
    expect(sporeAlpha({ ...makeSporeField(4)[0]!, y: -EDGE })).toBe(0);
    expect(sporeAlpha({ ...makeSporeField(4)[0]!, y: 1 + EDGE })).toBe(0);

    /* Faint well before it: a mote is under a tenth of its ink for the last few
       percent of its travel, so the disappearance is never an event. */
    expect(sporeFade(-EDGE + 0.01)).toBeLessThan(0.1);
    expect(sporeFade(1 + EDGE - 0.01)).toBeLessThan(0.1);
  });

  it("spends most of its travel at full strength rather than fading throughout", () => {
    /* A field that ramps everywhere reads as blinking, so the shape of the fade
       matters and not only its endpoints. Measured against the obvious
       alternative rather than against a number picked to pass: a linear triangle
       to the same endpoints is above 40% for exactly 60% of the travel, and the
       half-sine holds it for 73.75% — the whole reason it is a sine. */
    const samples = 4000;
    const fraction = (curve: (t: number) => number) => {
      let strong = 0;
      for (let i = 0; i <= samples; i += 1) if (curve(i / samples) > 0.4) strong += 1;
      return strong / samples;
    };
    const travel = (t: number) => sporeFade(-EDGE + t * (1 + 2 * EDGE));
    const triangle = (t: number) => 1 - Math.abs(2 * t - 1);

    expect(fraction(triangle)).toBeCloseTo(0.6, 2);
    expect(fraction(travel)).toBeCloseTo(0.7375, 2);
    expect(fraction(travel)).toBeGreaterThan(fraction(triangle));
  });

  it("starts the field already in mid-drift instead of as a row", () => {
    const field = makeSporeField(MAX_SPORES);
    const ys = field.map((spore) => spore.y);
    expect(Math.min(...ys)).toBeLessThan(0.2);
    expect(Math.max(...ys)).toBeGreaterThan(0.8);
    /* Spread over the whole travel band, including the parts outside the box,
       so nothing is bunched against an edge on the first frame. */
    expect(Math.max(...ys)).toBeLessThanOrEqual(1 + EDGE);
    expect(Math.min(...ys)).toBeGreaterThanOrEqual(-EDGE);
  });
});

describe("a resize does not restart the picture", () => {
  it("leaves the surviving motes exactly where they were", () => {
    /* A phone's URL bar collapsing fires a resize nobody asked for. Rebuilding
       the field on it would teleport every mote at once. */
    const field = makeSporeField(20).map((spore, i) => ({ ...spore, y: 0.5 + i * 0.01 }));

    const grown = refitSporeField(field, 30);
    expect(grown).toHaveLength(30);
    expect(grown.slice(0, 20)).toEqual(field);

    const shrunk = refitSporeField(field, 12);
    expect(shrunk).toHaveLength(12);
    expect(shrunk).toEqual(field.slice(0, 12));

    expect(refitSporeField(field, 20)).toEqual(field);
  });

  it("stratifies the motes it adds against the new count, not the old one", () => {
    const grown = refitSporeField(makeSporeField(10), 40);
    for (let i = 10; i < 40; i += 1) {
      expect(grown[i]!.lane).toBeGreaterThanOrEqual(i / 40);
      expect(grown[i]!.lane).toBeLessThan((i + 1) / 40);
    }
  });

  it("returns copies rather than the caller's own objects", () => {
    const field = makeSporeField(6);
    const refit = refitSporeField(field, 6);
    expect(refit[0]).not.toBe(field[0]);
    expect(refit[0]).toEqual(field[0]);
  });
});

describe("the layer is mounted where the empty band actually is", () => {
  it("spans the local strip and the doors as one field, not one per section", async () => {
    /*
     * The arrangement the whole `.rb-drift-band` wrapper exists for.
     *
     * Mounted inside `.rb-doors` alone, the field stopped at the hairline above
     * it — a horizon halfway up a column of drifting spores. Two instances would
     * not have fixed it: two seeds, two recyclings and two densities agreed on
     * only by accident still leave the rule visible as the line where the picture
     * changes. Exactly one layer, on a wrapper that contains both sections, is
     * the only arrangement in which a mote can cross it.
     */
    const page = await read("./RebrandLanding.tsx");
    expect(page.match(/<SporeDrift \/>/g)).toHaveLength(1);
    const band = page.slice(page.indexOf('className="rb-drift-band"'), page.indexOf('className="rb-proof"'));
    expect(band).toContain("<SporeDrift />");
    expect(band).toContain("<LocalStrip />");
    expect(band).toContain('className="rb-doors"');
    /* Last child, and that is load-bearing: the layer and the two sections all
       paint in the same auto/0 level, where tree order decides. Mounted before
       them it would go under two opaque section backgrounds and be invisible —
       the same failure as the layer opacity that used to be on it. */
    expect(band.indexOf("<SporeDrift />")).toBeGreaterThan(band.indexOf('className="rb-doors"'));
  });

  it("is the only decoration in that band — no fourth fruiting body", async () => {
    const page = await read("./RebrandLanding.tsx");
    const band = page.slice(page.indexOf('className="rb-drift-band"'), page.indexOf('className="rb-proof"'));
    /* The page already stands a body in the hero, at the exchange rule and in
       the questions column. The user asked for spores here and only spores. */
    expect(band).not.toContain("Mushroom");
    expect(band).not.toContain("Fruiting");
    expect(band).not.toContain("Colony");
    expect(band).not.toContain("mushroom-stage");
  });

  it("costs no bytes and no WebGL context", async () => {
    const component = code(await read("./SporeDrift.tsx"));
    /* Reaching for `<mushroom-stage>`'s spore shader would mean a fourth WebGL
       context with an invisible mushroom in it — 137KB for a dozen dots. */
    expect(component).not.toContain("loadMushroomStage");
    expect(component).not.toContain("mushroom-stage");
    expect(component).not.toContain("three");
    expect(component).not.toContain("webgl");
    expect(component).toContain('getContext("2d")');
  });

  it("builds one sprite per species from the profile rather than inline gradients", async () => {
    const component = code(await read("./SporeDrift.tsx"));
    /* The stops are the thing MEAN_INK_CEILING is a statement about, so they have
       to come from the module the ceiling is asserted in. A gradient written by
       hand here would drift away from the measured coverage silently — which is
       exactly how the 0.62 core stop ended up compounding the layer into
       invisibility. */
    expect(component).toContain("SPECIES_PROFILE[species]");
    expect(component).toContain("of profile.stops");
    expect(component).toContain("addColorStop(stop.offset");
    /* The flare is drawn from the profile too, and its geometry is read rather
       than written here — `flareInk` has to be able to account for the ink those
       arms lay down, and it can only do that if the numbers it integrates are
       the numbers the painter uses. */
    expect(component).toContain("profile.flare");
    expect(component).toContain("flare.arms");
    expect(component).toContain("flare.reach");
    /* One sprite per species, not one per mote: a couple of hundred canvases
       would be a real cost for three distinct pictures. */
    expect(component).toContain("SPECIES.map((species) => [species, makeSprite(species)])");
    /* And the per-mote alpha rides on globalAlpha, which is what lets those three
       sprites serve every ink and every size in the field. */
    expect(component).toContain("context.globalAlpha = alpha;");
  });

  it("scales the sprite box by the flare's reach at both ends", async () => {
    /*
     * `sporeExtent` has to be used twice and it is easy to use once. Applied only
     * when building the sprite, every glint paints its disc at 1/1.85 of the size
     * the field thinks it has — a species that silently shrank by half. Applied
     * only when drawing, the arms are clipped at the canvas edge instead. Both
     * mistakes render something plausible, which is what makes them worth
     * pinning.
     */
    const component = code(await read("./SporeDrift.tsx"));
    const sprite = component.slice(component.indexOf("function makeSprite"), component.indexOf("export function SporeDrift"));
    expect(sprite).toContain("sporeExtent(species)");
    const drawBody = component.slice(component.indexOf("const draw ="), component.indexOf("const layout ="));
    expect(drawBody).toContain("sporeExtent(spore.species)");
    /* The disc, not the box, is what `size` means everywhere else in the module,
       so the draw has to centre the enlarged box on the mote. */
    expect(drawBody).toContain("x - box / 2");
    expect(drawBody).toContain("y - box / 2");
  });

  it("paints in the module's depth order rather than the field's own", async () => {
    const component = code(await read("./SporeDrift.tsx"));
    expect(component).toContain("sporeDrawOrder(field)");
    expect(component).toContain("for (const index of order)");
    /* Recomputed on layout, not per frame: `depth` is fixed at birth, so sorting
       a few hundred motes every frame would be work for an order that cannot
       change. */
    const layout = component.slice(component.indexOf("const layout ="), component.indexOf("const tick ="));
    expect(layout).toContain("sporeDrawOrder(field)");
    const drawBody = component.slice(component.indexOf("const draw ="), component.indexOf("const layout ="));
    expect(drawBody).not.toContain("sporeDrawOrder");
  });

  it("advances the shimmer from the same clock as the sway", async () => {
    const component = code(await read("./SporeDrift.tsx"));
    /* `sporeAlpha` takes the clock now. Calling it without one would pin every
       mote to its phase at t=0 — a field that sways but never breathes, which is
       the failure that looks like nothing at all rather than like a bug. */
    expect(component).toContain("sporeAlpha(spore, clock)");
    expect(component).toContain("sporeX(spore, clock)");
  });
});

describe("the layer paints under the content and inside the section", () => {
  it("is positioned by the band rather than by the document or one section", async () => {
    const css = await read("./rebrand.css");
    const band = css.match(/\.rb-drift-band \{[^}]*\}/)![0];
    /* Without a containing block the absolute layer would size itself to the
       page, so the field would cover every section at once. On the wrapper and
       not on either section, because the field spans both. */
    expect(band).toContain("position:relative");
    /* And without clipping, a mote would leave the band into the catalogue above
       or the evidence below — different surface colours, where it reads as an
       artefact. Inside the band it crosses the rule freely, which is the point.
       `clip` and not `hidden`: `hidden` makes this a scroll container. */
    expect(band).toContain("overflow:clip");
    expect(band).not.toContain("overflow:hidden");
    /* Nothing else. A wrapper that painted a background or added padding would
       be changing what the two sections look like, and they still own that. */
    expect(band).not.toContain("background");
    expect(band).not.toContain("padding");
    expect(band).not.toContain("z-index");

    const layer = css.match(/\.rb-spore-drift \{ position[^}]*\}/)![0];
    expect(layer).toContain("position:absolute");
    expect(layer).toContain("inset:0");
    expect(layer).toContain("pointer-events:none");
    /* Above the section's ivory background, below the content wrappers and the
       mycelium sheet — which is where the page's one stacking rule puts it. */
    expect(layer).toMatch(/z-index:0/);
  });

  it("opts out of the stacking rule that would pull it into the content layer", async () => {
    const component = await read("./SporeDrift.tsx");
    expect(component).toContain("rb-spore-drift rb-decor");
    const stacking = await read("./mycelium-network.css");
    /* The rule is more specific than a single class, so without `rb-decor` it
       would win over this layer's own z-index no matter what it declared. */
    expect(stacking).toContain(":not(.rb-decor)");
  });

  it("does not drop the wrapped sections out of the page's content layer", async () => {
    /*
     * The quiet casualty of introducing the wrapper, and the reason this test
     * exists at all.
     *
     * The content-above-the-colony rule was `.rb-page > section …`. Wrapping the
     * local strip and the doors in `.rb-drift-band` took both of them out of
     * that child relationship, so their headings, cards and door plates silently
     * fell back to the auto level and the mycelium sheet at z-index 1 began
     * painting filaments across the copy. Nothing throws, nothing fails to
     * render, and it is two sections of the page.
     *
     * A descendant combinator restores them. The page nests no section inside
     * another, so it matches exactly the same set plus whatever any future
     * wrapper holds — which is the behaviour that was meant: a layout wrapper
     * should not be able to change what paints above what.
     */
    const stacking = await read("./mycelium-network.css");
    expect(stacking).toContain(".rb-page section:not(.rb-hero):not(.rb-story) > *:not(.rb-decor)");
    expect(stacking).not.toContain(".rb-page > section:not(");
  });

  it("is hidden from pointers and from screen readers", async () => {
    const component = await read("./SporeDrift.tsx");
    expect(component).toContain('aria-hidden="true"');
  });
});

describe("reduced motion gets a still frame, not a blank band", () => {
  it("never starts the loop but still draws the field once", async () => {
    const component = await read("./SporeDrift.tsx");
    expect(component).toContain('matchMedia("(prefers-reduced-motion: reduce)")');
    /* `layout()` calls `draw()` unconditionally and is called on mount, so the
        suspended field is painted; the observer's `reduced` branch is what
        stops it ever advancing. Removing the draw from layout would leave these
        visitors an empty band, which takes the design away from the person who
        asked for less motion rather than less design. */
    const layout = component.slice(component.indexOf("const layout ="), component.indexOf("const tick ="));
    expect(layout).toContain("draw();");
    const observerBody = component.slice(component.indexOf("const observer = new IntersectionObserver"));
    expect(observerBody).toMatch(/if \(reduced \|\| !onScreen\)/);
    expect(observerBody).toMatch(/cancelAnimationFrame/);
  });

  it("skips the layer entirely on the phone widths the rest of the page skips", async () => {
    const component = await read("./SporeDrift.tsx");
    const css = await read("./rebrand.css");
    /* The same 700 the hero organism, the ground colony and the build's preload
       script use. Below it the copy fills the column, so a mote is behind text
       being read rather than in a margin beside it. */
    expect(component).toContain('const NARROW = "(max-width: 700px)"');
    expect(component).toContain("if (window.matchMedia(NARROW).matches) return;");
    expect(css).toContain("@media(max-width:700px)");
  });

  it("stops drawing when the section is off screen", async () => {
    const component = await read("./SporeDrift.tsx");
    /* A decoration three sections deep has no business compositing while the
       reader is somewhere else, and the rAF chain has to stop rather than
       merely skip work. */
    expect(component).toContain("if (onScreen) frame = window.requestAnimationFrame(tick);");
  });
});
