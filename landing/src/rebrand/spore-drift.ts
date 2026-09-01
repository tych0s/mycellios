import { makeRandom } from "./seeded-random";

/*
 * The spore drift that fills the band from the local strip through the three
 * doors.
 *
 * WHY THE MOTION IS A MODULE AND NOT SIXTY LINES INSIDE THE COMPONENT.
 *
 * Everything that can go wrong with a field of drifting motes is silent. Ink
 * that creeps up past the page's own hairline turns ambient decoration into a
 * dot screen over the heading; a rise speed a factor of three too high turns it
 * into falling snow; a field generated from `Math.random` is a different picture
 * on every load, so nobody can review it; and a wrap that fires at y = 0 instead
 * of past the edge makes every mote vanish and reappear at the top border in
 * plain sight. None of those throw. So the rules live here, as pure functions
 * over numbers, and the canvas in `SporeDrift.tsx` only paints what they say.
 *
 * WHY THIS BAND GETS SPORES AND NOT A BODY.
 *
 * The page already carries three fruiting bodies (hero, exchange edge, the
 * questions column) and one is enough per beat. What is actually empty here is
 * the run of paper from "just a local address" through "three doors" — and a
 * fourth mushroom standing in it would be the page repeating itself. Spores are
 * the other half of the same organism and they are what an empty stretch of air
 * above a colony actually contains: the release, not another body.
 *
 * WHY IT IS ONE FIELD OVER TWO SECTIONS.
 *
 * The layer started inside `.rb-doors` and stopped dead at the hairline between
 * that section and the local strip above it — a horizon halfway up a column of
 * drifting spores, which is the one thing air cannot have. Weather does not
 * observe a section boundary. So the two sections are wrapped in
 * `.rb-drift-band` and the field is mounted on the wrapper: the sections keep
 * their own surfaces and their own rule, and the motes cross it. Nothing in this
 * module knows about either section — it is given a box and fills it, which is
 * why growing the box was a change to the markup and to two numbers here (the
 * rise rates, which are fractions of a box that is now half again as tall) and
 * to nothing else.
 *
 * WHY THE FIELD IS NOT THE HERO'S FIELD REPEATED.
 *
 * The hero's motes are one sprite rising in a straight line with a single sine
 * of sway, seen against a dark stage where anything luminous reads. Copying that
 * onto ivory paper gives a column of identical dots — the same gesture twice,
 * and the weaker of the two. So this field is built out of three things the hero
 * does not have:
 *
 *   · Three species (`SPECIES_PROFILE`) instead of one sprite — a soft mote, a
 *     ringed halo that reads as a spore slightly out of focus, and a small hard
 *     glint. A field of one shape is a texture; a field of three is a colony.
 *   · Depth strata (`depth`) — size, rise, sway and ink all come off the same
 *     0→1 axis, so a near mote is bigger, faster, stronger *together*. That is
 *     what makes the layer read as air with volume rather than as a flat plane
 *     of dots, and it is the reason the field can be dense without becoming a
 *     screen: most of what fills it is far, small and faint.
 *   · Curved travel and a slow shimmer — two sway harmonics rather than one, so
 *     no mote retraces its own path, and a per-mote breathing of alpha that
 *     never brightens past its own ink. The hero rises; this one drifts.
 *
 * SUBORDINATION IS LOW CONTRAST, NOT ABSENCE — AND IT IS AN AVERAGE, NOT A PEAK.
 *
 * The previous version of this file capped a mote's *peak* pixel under
 * `--rb-line`'s 16% and then multiplied that peak by a 0.62 sprite stop and a
 * 0.58 layer opacity. Those three factors compounded to about 2% ink over paper,
 * which is not a subtle field — it is no field at all, and it is what the layer
 * actually shipped as. The rule that replaced it moved the ceiling onto a mote's
 * mean coverage over its own footprint (`MEAN_INK_CEILING`), because what makes
 * a mark on a page is the ink a shape covers rather than its brightest pixel.
 *
 * That was the right quantity on the wrong subject, and this version finishes
 * the correction. A ceiling on one mote cannot see the field: raising the
 * density and the ink together — which is precisely what "more of them and
 * brighter" asks for — passes a per-mote rule untouched while putting half again
 * as much bronze on the paper. So the governing rule is now
 * `FIELD_INK_CEILING`, on the fraction of the band the whole field darkens
 * (`fieldInk`), and `MEAN_INK_CEILING` stays underneath it as the local rule
 * that keeps any single mote from becoming a blot. Two rules, each about the
 * thing it can actually see.
 */

/* One seed for the whole field, so the picture is identical on every load and
   on every engine — which is the only thing that makes it reviewable. Distinct
   from the network's 20260811 and the hero colony's 20260817: three fields
   sharing a seed would put their motes in correlated columns. */
export const SPORE_SEED = 20_260_812;

/*
 * Density, not count.
 *
 * The layer covers the whole drift band — the local strip and the doors
 * together, around 1,100px tall on a laptop and half as wide again on a 27"
 * display. A fixed count would read as a thin scatter at one size and as weather
 * at another, so the count comes from the area and the *density* is the constant
 * being tuned.
 *
 * One mote per 5,600px² is a mote every ~75px in both directions, at every
 * viewport — which is the figure that actually decides how the layer reads, and
 * the reason the constant is an area and not a count. At 1440×1150 that is 295
 * motes in the band and about half of them in the part a reader ever sees:
 * everything the doors plate and the local strip's two cards do not cover. It
 * was 6,500 — an 81px spacing over the doors alone — and the request was for
 * more of them, so the paper between motes came down by a fifth. It can be this
 * dense without becoming a texture because the depth strata keep most of those
 * motes small, slow and faint, and because the field's total ink is bounded
 * directly by FIELD_INK_CEILING below.
 *
 * The clamps are runaway guards, not working limits: they must stay clear of the
 * counts the density actually asks for, or the density stops governing and every
 * wide display gets the same field as every other. A 4K desktop is a real
 * viewport with a band this tall and asks for around 960, so the ceiling sits
 * above that and only catches a box no layout produces.
 */
export const AREA_PER_SPORE = 5_600;
export const MIN_SPORES = 18;
export const MAX_SPORES = 1_800;

/*
 * The ink range a mote's core is drawn at.
 *
 * MAX_INK is the alpha of the brightest single pixel in the field — the centre
 * of the nearest mote — before the layer's own opacity. It is deliberately well
 * above `--rb-line`: a hairline is 16% ink drawn continuously across a 1200px
 * plate, and a 24px soft blob whose centre touches 44% covers a tiny fraction of
 * that while being visible at all, which is the entire point. Against ivory
 * (#eee9e2) a bronze at 44% is a difference of about thirty levels — a spore you
 * can see, not a suggestion of one. What keeps the layer subordinate is
 * MEAN_INK_CEILING below, which governs coverage rather than a peak.
 *
 * MIN_INK is the other half of the same rule, and it is the one that decides
 * whether a *dense* field reads as filled or as a sparse handful of dots over
 * nothing: subordination is low contrast, not invisibility. At 12% — the first
 * value here — the whole far stratum rendered as paper, so the count said 155
 * and the screen showed thirty.
 *
 * Both ends came up — 0.44/0.20 to 0.56/0.26 — because the field was asked to
 * be brighter, and on ivory paper brightness is bronze rather than white: the
 * one thing that must *not* move for it is the tint, since mixing toward the
 * near-white core makes a mote lighter than the paper it covers and it comes out
 * as an annulus around a hole. Ink is the honest knob. 0.56 against #eee9e2 is a
 * difference of about forty levels at the centre of a near mote, and 0.26 keeps
 * the farthest one a mark rather than a rumour.
 *
 * They came up again — to 0.62/0.3 — when the species gained their internal
 * structure, and that is a consequence rather than a second opinion. Putting a
 * shoulder in a falloff means the ink outside the nucleus *drops*: the three
 * profiles lost about a fifth of their mean coverage on the day they stopped
 * being smooth blurs, so holding the peak constant would have quietly made the
 * whole band lighter than the version this one is meant to improve on. The
 * measured field ink is the thing being held steady; these two are the dial that
 * holds it.
 */
export const MAX_INK = 0.62;
export const MIN_INK = 0.3;

/*
 * The ceiling on a mote's *mean* coverage over its own footprint.
 *
 * A mote is a point and `--rb-line` is a line across a 1200px plate, so the two
 * were never the same quantity — pinning one to the other was a proxy, and it
 * was the proxy that stood between this layer and being visible. What a page is
 * actually spared by a rule like this is *total* ink, which is what
 * FIELD_INK_CEILING below governs directly. This one stays because a single mote
 * that averages more than a hairline is a dot on the page whatever the field
 * around it does, and 0.19 is where a 20px near mote is a spore rather than a
 * blot. The paired test asserts it against `--rb-line` as a ratio rather than as
 * an inequality, so the two numbers still move together.
 */
export const MEAN_INK_CEILING = 0.19;

/*
 * The ceiling on what the whole field lays down, as a fraction of the band.
 *
 * The rule that replaces "quieter than a hairline", and the one that is actually
 * about the page rather than about one sprite. Sum each mote's mean ink over its
 * own disc, divide by the area of the band: that is the fraction of the paper
 * this layer darkens, and it is the only figure that stays meaningful when both
 * the density and the ink go up at once — which is exactly what was asked for
 * here, and exactly the pair of changes a per-mote ceiling cannot see.
 *
 * The field measures about 0.22% at every viewport (the density holds it flat,
 * which is the point of deriving the count from the area). 0.4% is the ceiling:
 * roughly twice the current weight, so the layer has room to be tuned, and far
 * below the ~1% where a full-width band of soft bronze stops reading as air over
 * paper and starts reading as a tint on the paper itself.
 */
export const FIELD_INK_CEILING = 0.004;

/*
 * How far outside the box a mote travels before it is recycled.
 *
 * The wrap has to happen where nobody can see it. Recycling at y = 0 would pop
 * every mote out of existence exactly on the top edge and back in on the
 * bottom one, in full view; and it would do it to a mote that `sporeFade` has
 * only faded to sin(π·0.06) ≈ 18% rather than to nothing. So the field is a
 * band taller than the box at both ends, and the fade reaches zero at the
 * recycling line itself.
 */
export const EDGE = 0.06;

/*
 * How much of its own ink a mote breathes away at the bottom of its shimmer,
 * and how slowly.
 *
 * The shimmer is what keeps a dense field from reading as a fixed pattern of
 * dots that happens to be sliding upward: motes come forward and fall back
 * independently, so the eye never locks onto the arrangement. It is bounded at
 * both ends for the usual two reasons — 0 is a static field, and anything past
 * about a third starts to blink, which is an event rather than air.
 *
 * The rate is in cycles per second: 0.05–0.15Hz is a seven-to-twenty second
 * breath, slower than anything a reader tracks deliberately.
 */
export const TWINKLE_MIN = 0.1;
export const TWINKLE_MAX = 0.32;
export const TWINKLE_RATE_MIN = 0.05;
export const TWINKLE_RATE_MAX = 0.15;

/**
 * One stop of a species' radial falloff.
 *
 * `offset` is the fraction of the sprite radius, `alpha` is that ring's opacity
 * as a fraction of the mote's ink, and `tint` mixes the two brand colours: 1 is
 * the warm near-white a backlit spore has at its centre, 0 is the bronze. The
 * stops live here rather than in the component because MEAN_INK_CEILING is a
 * statement about them — the coverage rule cannot be checked from a
 * `CanvasGradient`.
 *
 * WHY THE TINTS ARE LOW, WHICH IS THE OPPOSITE OF THE HERO'S.
 *
 * The hero's spore shader mixes its accent halfway to #fff2e4 and adds it, and
 * that is right for the hero: it is a lit object on a dark stage, where a warm
 * near-white core is the brightest thing in the frame. This layer is on ivory
 * paper — `--rb-paper` is rgb(238,233,226) — and there a near-white centre is
 * *less* ink than the paper it covers, so copying the hero's mix produces a mote
 * with a hole in it: a faint bronze annulus around a centre nobody can see. The
 * warmth is worth keeping because it is what stops the field looking like brown
 * dots, but it belongs as a hint at the very middle of a bronze-dominant
 * falloff, which is what these tints are. The one species that carries real
 * warmth is the glint, and only because it is small enough to read as a
 * highlight rather than as a gap.
 */
export type SporeStop = { readonly offset: number; readonly alpha: number; readonly tint: number };

/**
 * The star flare on a species that is meant to read as lit.
 *
 * A radial falloff can only ever be a disc, and a disc — however bright its
 * centre — is a dot. What the eye actually reads as *shining* is the streak: the
 * flare a bright point throws across a lens or a wet eyelash, which is why every
 * drawn star since the invention of drawing has arms on it. So the glint gets
 * arms, and they are the reason it stops looking like a small mote.
 *
 * `arms` is how many (4 is a cross, the diffraction figure of a round aperture),
 * `reach` how far out they run as a fraction of the sprite radius, `width` the
 * half-width at the centre where an arm is thickest, and `alpha` the arm's
 * opacity at the centre as a fraction of the mote's ink. Each arm is a spike that
 * tapers to a point *and* to zero alpha at the same time, so it has no tip and no
 * edge — the same rule the radial stops are held to, for the same reason.
 *
 * Data rather than drawing code, again because `flareInk` has to be able to
 * measure it: a flare that is only a path in the component is ink the field's
 * ceiling cannot see.
 */
export type SporeFlare = {
  readonly arms: number;
  readonly reach: number;
  readonly width: number;
  readonly alpha: number;
};

export const SPECIES = ["mote", "halo", "glint"] as const;
export type Species = (typeof SPECIES)[number];

/*
 * The three shapes in the field.
 *
 * `size` and `ink` are multipliers on what the depth axis already decided, which
 * is what makes a species a variation on one organism rather than three
 * unrelated decorations:
 *
 *   mote  — the plain spore, and the bulk of the field. A dense nucleus inside a
 *           softer body, because a spore under any magnification is a cell with
 *           something in it and a pure falloff is a thumbprint.
 *   halo  — larger and softer, with a faint ring outside a dim core: a spore
 *           just off the plane of focus. Its ink is pulled *down* because it
 *           covers half again as much paper as a mote at the same depth.
 *   glint — half the size, the full ink, and the only species with a flare. This
 *           is the one that makes the field read as lit rather than as printed;
 *           there are few of them for exactly that reason.
 *
 * WHY THE SHAPES GAINED STRUCTURE.
 *
 * The field as first shipped was three radial gradients, and at a hundred
 * percent zoom on real paper that is three sizes of the same soft blur — the
 * band read as specks of dust rather than as spores catching light. The fix is
 * not more ink (the field's weight is already where it should be) but internal
 * *structure*: a step in the falloff where a nucleus ends, and arms on the
 * species that is supposed to shine. Detail is what survives being small;
 * brightness alone just makes a bigger smudge.
 *
 * No `ink` is above 1: `MAX_INK` is the ceiling on the brightest pixel in the
 * field, and a species that multiplied past it would make that ceiling a lie.
 */
export const SPECIES_PROFILE: Record<
  Species,
  {
    readonly size: number;
    readonly ink: number;
    readonly stops: readonly SporeStop[];
    readonly flare?: SporeFlare;
  }
> = {
  mote: {
    size: 1,
    ink: 1,
    stops: [
      /* Nucleus out to 0.18, then a fast shoulder: the ramp from 0.9 to 0.44
         across a tenth of the radius is what gives the mote a discernible body
         inside its own glow instead of one continuous blur. It is a steep ramp
         and not a step, because a step is an edge and an edge is a dot. */
      { offset: 0, alpha: 1, tint: 0.34 },
      { offset: 0.18, alpha: 0.9, tint: 0.16 },
      { offset: 0.28, alpha: 0.44, tint: 0.04 },
      { offset: 0.58, alpha: 0.2, tint: 0 },
      { offset: 1, alpha: 0, tint: 0 },
    ],
  },
  halo: {
    size: 1.5,
    ink: 0.82,
    stops: [
      { offset: 0, alpha: 0.92, tint: 0.4 },
      { offset: 0.12, alpha: 0.62, tint: 0.16 },
      /* The gap. Pulled lower than it was so the ring outside it reads as a
         separate thing rather than as a bump in a falloff — the whole point of
         this species is that a reader can see two rings in it. */
      { offset: 0.34, alpha: 0.08, tint: 0 },
      /* The ring: coverage rises again away from the centre. A hard stop here
         would draw a circle, so both sides of it are ramps. */
      /* Pure bronze, like every stop this far out. The ring briefly carried a
         little warmth here and the paired test was right to reject it: at 0.56
         of the radius a warm stop is a pale ring on pale paper, which is the
         annulus-around-a-hole failure arriving on the one species whose whole
         point is that its ring is legible. */
      { offset: 0.56, alpha: 0.38, tint: 0 },
      { offset: 0.72, alpha: 0.1, tint: 0 },
      { offset: 1, alpha: 0, tint: 0 },
    ],
  },
  glint: {
    /* 0.6 rather than 0.52. The disc is still the smallest of the three — that
       is what makes it a spark and not a mote — but at 0.52 a far glint was a
       4px disc, and 4px of soft falloff cannot hold a nucleus, a shoulder and
       four arms. This is the smallest size at which the species' own detail
       survives being drawn. */
    size: 0.6,
    ink: 1,
    stops: [
      /* A hard core and then a cliff. This species is described as a *hard*
         glint and its falloff was the softest of the three — 0.82 still standing
         at a quarter of the radius and no step anywhere. A point of light has a
         sharp middle; that is what distinguishes it from a small mote, and no
         amount of flare rescues a soft centre. */
      { offset: 0, alpha: 1, tint: 0.62 },
      { offset: 0.22, alpha: 0.92, tint: 0.3 },
      { offset: 0.34, alpha: 0.4, tint: 0.08 },
      { offset: 0.62, alpha: 0.16, tint: 0 },
      { offset: 1, alpha: 0, tint: 0 },
    ],
    /*
     * Four arms reaching well past the disc — the flare is most of what a glint
     * is, and a spike that stopped at the rim would just be a lumpy dot.
     *
     * `width` is a fraction of the *radius*, and the radius here is small: a near
     * glint is a 10px disc, so 5px of radius. The first value was 0.03, which is
     * a fifth of a pixel at the waist and tapers from there — geometry too thin
     * for any rasteriser to put ink in, and it rendered as literally nothing
     * while every ink assertion passed. 0.18 is a hairline of about a pixel at
     * the near stratum and half of one at the far, which is the thinnest an arm
     * can be and still exist.
     *
     * `alpha` is 0.62 rather than the 0.34 it started at, and that is not the
     * layer getting louder: it is a per-arm figure that then tapers, so what
     * lands on the paper along most of an arm is a fraction of it. At 0.34 the
     * arms measured as ink in `flareInk` and rendered as nothing on ivory —
     * which is the same trap the radial stops fell into once already. The
     * restraint that keeps this a catch of light rather than a stock-pack
     * sparkle is the *width* and the fact that only a fifth of the field
     * carries one.
     */
    flare: { arms: 4, reach: 2.2, width: 0.18, alpha: 0.62 },
  },
};

/*
 * How the field is made up.
 *
 * Cumulative thresholds against one draw of the RNG, so the mix is a property of
 * the seed and not of a counter that a resize could shift. Mostly plain motes:
 * the halo and the glint are what a reader notices, and a field where a third of
 * the motes are sparks is a christmas light, not a spore release.
 *
 * The glint's share went from 14% to 20% for the same reason the ink came up: it
 * is the species that makes the field read as *lit* rather than as printed, and
 * a fifth of the motes is where the band picks up a sparkle without any one part
 * of it becoming a constellation. It is also the cheapest brightness available
 * here — a glint is half a mote's diameter, so raising its share adds sparkle at
 * about a quarter the ink of raising anything else.
 */
export const SPECIES_MIX: readonly { readonly species: Species; readonly upTo: number }[] = [
  { species: "mote", upTo: 0.58 },
  { species: "halo", upTo: 0.8 },
  { species: "glint", upTo: 1 },
];

export type Spore = {
  /** Which of the three shapes this mote is drawn as. */
  readonly species: Species;
  /** Column centre, 0→1 across the box. Stratified; see `makeSporeField`. */
  readonly lane: number;
  /** 0 at the top edge, 1 at the bottom. Rising means this decreases. */
  readonly y: number;
  /** 0 is the far stratum, 1 the near one. Drives size, rise, sway and ink. */
  readonly depth: number;
  /** Sprite diameter in CSS pixels. A mote is a mote at any viewport. */
  readonly size: number;
  /** Fractions of the box height per second. */
  readonly rise: number;
  /** Sway amplitude, fractions of the box width. */
  readonly sway: number;
  /** Sway frequency, so the field does not breathe sideways in unison. */
  readonly swayRate: number;
  /** Sway phase, so no two motes cross the box in step. */
  readonly phase: number;
  /** Peak alpha of this mote's core, before the layer's own opacity. */
  readonly ink: number;
  /** Fraction of its ink this mote breathes away at the bottom of a cycle. */
  readonly twinkle: number;
  /** Cycles per second of that breath. */
  readonly twinkleRate: number;
};

/** The species a draw of the RNG in [0,1) lands on. */
export function pickSpecies(roll: number): Species {
  for (const entry of SPECIES_MIX) if (roll < entry.upTo) return entry.species;
  /* The last threshold is 1 and a roll is always below it, so this is only the
     path a malformed mix would take. It falls back to the plain mote rather than
     asserting: a decoration has no business throwing, and one species drawn for
     the whole field is a visible bug that leads straight back here. */
  return "mote";
}

/** A species' opacity at a fraction of its radius, interpolated between stops. */
export function alphaAtRadius(stops: readonly SporeStop[], radius: number): number {
  const first = stops[0];
  if (!first) return 0;
  if (radius <= 0) return first.alpha;
  if (radius >= 1) return 0;
  for (let i = 1; i < stops.length; i += 1) {
    const from = stops[i - 1];
    const to = stops[i];
    if (!from || !to || radius > to.offset) continue;
    const span = to.offset - from.offset;
    const t = span <= 0 ? 0 : (radius - from.offset) / span;
    return from.alpha + (to.alpha - from.alpha) * t;
  }
  return 0;
}

/**
 * A species' mean opacity over its own footprint.
 *
 * The area-weighted average of `alphaAtRadius` over the unit disc — the 2r is
 * the weight, because a ring at 0.9 of the radius is nine times the paper of a
 * ring at 0.1 and a plain average of the stops would flatter every soft sprite
 * enormously. This is the quantity MEAN_INK_CEILING is a ceiling on, and it is
 * why the stops are data in this file rather than a gradient in the component.
 */
export function meanCoverage(stops: readonly SporeStop[], samples = 4096): number {
  let sum = 0;
  for (let i = 0; i < samples; i += 1) {
    const radius = (i + 0.5) / samples;
    sum += alphaAtRadius(stops, radius) * 2 * radius;
  }
  return sum / samples;
}

/**
 * How far a species is drawn past its own disc, as a multiple of the diameter.
 *
 * The glint's arms reach 1.85 radii and the sprite is a square canvas, so
 * painting it into a box the size of the disc would clip the flare into a
 * four-sided stub with a hard edge on every arm — the exact opposite of the
 * shape being asked for, and something that would be visible immediately and
 * explicable by nobody. Everything about a mote's size, ink and spacing stays a
 * statement about the *disc*; this is only the box the painter needs.
 */
export function sporeExtent(species: Species): number {
  return Math.max(1, SPECIES_PROFILE[species].flare?.reach ?? 1);
}

/**
 * How an arm's ink falls off along its own length.
 *
 * Shared data rather than a gradient written in the painter, for the same reason
 * `SPECIES_PROFILE.stops` is: `flareInk` integrates this curve to decide what the
 * arms cost, and a taper that lived only in the component would drift away from
 * the measured figure silently. `offset` is the fraction of `reach`; `tint` is
 * near-bronze throughout, because an arm is a thin streak and a thin streak of
 * near-white over ivory paper is nothing at all.
 */
export const FLARE_TAPER: readonly SporeStop[] = [
  { offset: 0, alpha: 1, tint: 0.22 },
  { offset: 0.4, alpha: 0.45, tint: 0.05 },
  { offset: 1, alpha: 0, tint: 0 },
];

/**
 * The ink a flare lays down, in the same units as `meanCoverage`.
 *
 * Integrated rather than approximated in closed form, because the arms taper in
 * two ways at once — the width shrinks linearly to a point while the alpha
 * follows `FLARE_TAPER` — and the first version of this function multiplied the
 * two means as though they were independent, which overstated the arms by about
 * two thirds. Same shape of integral as `meanCoverage`: walk the arm, take the
 * ink of each slice, normalise by the disc's own area so the two are summable.
 *
 * It is measured at all because the arms are drawn additively and mostly
 * *outside* the disc: they are exactly the kind of decoration that gets
 * brightened and widened over successive passes with no assertion noticing,
 * until the band is a field of crosses.
 */
export function flareInk(flare: SporeFlare | undefined, samples = 1024): number {
  if (!flare) return 0;
  let arm = 0;
  for (let i = 0; i < samples; i += 1) {
    const t = (i + 0.5) / samples;
    /* Slice of one arm: full width (2×) at this point, times its ink there. */
    arm += 2 * flare.width * (1 - t) * flare.alpha * alphaAtRadius(FLARE_TAPER, t);
  }
  return (flare.arms * (arm / samples) * flare.reach) / Math.PI;
}

/**
 * The ink a mote lays down over its own disc — the *blot* measure.
 *
 * Deliberately not counting the flare, which is the correction to a mistake this
 * function briefly shipped with. MEAN_INK_CEILING asks "is this one mote a dot
 * on the page", and that is a question about density over the area a mote
 * occupies. A flare's arms are thin and run to twice the radius, so folding them
 * into an average over the *disc* charges a spread-out mark at the concentrated
 * mark's rate — it made the glint read as the heaviest species in the field when
 * it is the lightest thing in it. The page-weight question is a different one
 * and `fieldInk` below answers it, with the flare fully counted.
 */
export function sporeMeanInk(spore: Spore): number {
  return spore.ink * meanCoverage(SPECIES_PROFILE[spore.species].stops);
}

/** Everything a mote puts on the paper, disc and flare, per unit of disc area. */
export function sporeTotalInk(spore: Spore): number {
  const profile = SPECIES_PROFILE[spore.species];
  return spore.ink * (meanCoverage(profile.stops) + flareInk(profile.flare));
}

/**
 * The fraction of a box's paper the whole field darkens.
 *
 * Each mote's mean ink times the area of its own disc, summed, over the area of
 * the box. This is the quantity FIELD_INK_CEILING is a ceiling on, and it is the
 * only one that catches the failure a per-mote rule cannot: doubling the count
 * while halving each mote's ink passes every per-mote assertion and puts exactly
 * the same amount of bronze on the paper.
 *
 * Peak ink rather than any one frame's: the fade and the shimmer only ever
 * subtract, so this is the field at its heaviest, which is what a ceiling should
 * be measured against.
 */
export function fieldInk(field: readonly Spore[], width: number, height: number): number {
  const area = Math.max(0, width) * Math.max(0, height);
  if (area <= 0) return 0;
  let ink = 0;
  for (const spore of field) {
    const radius = spore.size / 2;
    /* Total, not the blot measure: the arms are real ink on real paper, and the
       whole purpose of this figure is that nothing the layer draws escapes it. */
    ink += sporeTotalInk(spore) * Math.PI * radius * radius;
  }
  return ink / area;
}

/** How many motes a box of this size carries. */
export function sporeCount(width: number, height: number): number {
  const area = Math.max(0, width) * Math.max(0, height);
  return Math.min(MAX_SPORES, Math.max(MIN_SPORES, Math.round(area / AREA_PER_SPORE)));
}

/**
 * Builds the field.
 *
 * The x positions are stratified — one mote per equal column, jittered inside
 * it — rather than drawn uniformly. Uniform draws clump: a 40-mote uniform
 * field reliably leaves a quarter of the width empty and puts three motes
 * within a few pixels of each other, and over a 1200px band that empty quarter
 * is the most visible thing in the layer. Stratification is the same randomness
 * with the clumping removed, and it is what the hero's own spores do.
 *
 * Everything else about a mote hangs off `depth`. That is the whole reason the
 * field can be this dense and still read as air: a mote drawn small is also
 * drawn slow and faint, so density arrives as distance rather than as clutter.
 */
export function makeSporeField(count: number, seed: number = SPORE_SEED): Spore[] {
  const random = makeRandom(seed);
  return Array.from({ length: count }, (_, index) => {
    const species = pickSpecies(random());
    const profile = SPECIES_PROFILE[species];
    /*
     * Biased toward the far stratum, but not squared.
     *
     * A flat distribution puts as many 20px motes in the band as 7px ones, which
     * is the dot-screen failure arriving by a different door. Squaring it is the
     * opposite mistake and the one the first draft made: it pushed most of the
     * field to depth < 0.3, where a mote is 8px at 15% ink, and the rendered band
     * came back as a sparse handful of visible dots over a haze nobody could
     * see — a filled field on paper and an empty one on screen. 1.55 keeps the
     * far stratum the crowded one while leaving a real middle.
     */
    const depth = random() ** 1.55;
    return {
      species,
      lane: (index + 0.18 + random() * 0.64) / count,
      /* Spread over the taller-than-the-box band, so the field is already in
         mid-drift on the first frame instead of starting as a row. */
      y: -EDGE + random() * (1 + 2 * EDGE),
      depth,
      /*
       * The floor matters more than the ceiling.
       *
       * `size` is a diameter in CSS pixels, and it is the diameter of a shape
       * with no edge — so a 5px mote is a 2.5px radius of falloff that peaks at
       * one pixel and is gone by the second. It rounds to nothing on any display.
       * 7.5px is where a soft radial blob is still a *shape* at the far end of
       * the field, and the near end stays under 25px, past which a single mote is
       * a smudge on the paper rather than a spore.
       */
      size: (7.5 + depth * 12) * profile.size,
      /*
       * Slow, and slower than the hero's.
       *
       * The hero's motes rise at 0.011–0.023 of the viewport per second, in a
       * scene the visitor is looking at. This band is beside copy the visitor is
       * reading, and a mote that crosses it in fifteen seconds is something that
       * keeps catching the eye.
       *
       * The rate is a fraction of the box per second, and the box just grew from
       * one section to two — so these numbers came *down* when the band grew,
       * which looks backwards and is not. What a reader perceives is pixels per
       * second, not fractions: leaving 0.009–0.026 alone across a 1,100px band
       * instead of a 700px one would have sped every mote up by half without a
       * single line of the file changing. At 0.006–0.017 of the taller band a
       * mote travels the same 7–19px a second it did before and takes between one
       * and three minutes to cross the whole thing, which is what a spore in
       * still air looks like. Tied to depth so the near stratum runs faster than
       * the far one — the parallax that gives the layer its volume.
       */
      rise: 0.006 + depth * 0.011,
      sway: 0.004 + depth * 0.009,
      swayRate: 0.26 + random() * 0.2,
      phase: random() * Math.PI * 2,
      ink: (MIN_INK + depth * (MAX_INK - MIN_INK)) * profile.ink,
      twinkle: TWINKLE_MIN + random() * (TWINKLE_MAX - TWINKLE_MIN),
      twinkleRate: TWINKLE_RATE_MIN + random() * (TWINKLE_RATE_MAX - TWINKLE_RATE_MIN),
    };
  });
}

/**
 * Refits an existing field to a new count without disturbing the motes that
 * survive.
 *
 * A resize must not rebuild the field. Rebuilding teleports every mote to a new
 * position, so dragging a window edge — or a phone rotating, or a mobile
 * browser's URL bar collapsing, which fires a resize for a change nobody asked
 * for — makes the whole layer jump. Keeping the prefix means a resize adds or
 * removes motes at the end and leaves the picture otherwise where it was.
 */
export function refitSporeField(field: readonly Spore[], count: number, seed: number = SPORE_SEED): Spore[] {
  /* Copies in every branch, including the one where nothing changes. A function
     that returns fresh objects at two counts and the caller's own at the third
     has a contract nobody can hold in their head, and the no-op branch is
     exactly the one a future caller will reach for when it decides it may keep
     the result. Uniformity costs one allocation per resize. */
  if (count === field.length) return field.map((spore) => ({ ...spore }));
  if (count < field.length) return field.slice(0, count).map((spore) => ({ ...spore }));
  /* The new motes come from a field generated at the target count, so their
     lanes stratify against that count rather than against the old one — take
     them from the tail, which is the part the surviving prefix does not cover. */
  return [...field.map((spore) => ({ ...spore })), ...makeSporeField(count, seed).slice(field.length)];
}

/** One frame of rise, recycled past the far edge. */
export function riseSpore(spore: Spore, deltaSeconds: number): Spore {
  const y = spore.y - spore.rise * Math.max(0, deltaSeconds);
  return { ...spore, y: y < -EDGE ? 1 + EDGE : y };
}

/**
 * Where the mote is across the box, as a fraction of the width.
 *
 * Two harmonics, not one. A single sine is a pendulum: the mote retraces the
 * same arc for as long as it is on screen, and in a field this dense a hundred
 * synchronised pendulums of different periods still read as a mechanism. Adding
 * a second, faster, smaller term at an incommensurable phase makes the path a
 * slowly precessing figure that never closes — which is what a mote in still air
 * actually does, and is the clearest difference between this field and the
 * hero's.
 */
export function sporeX(spore: Spore, seconds: number): number {
  const base = Math.sin(seconds * spore.swayRate + spore.phase);
  const detail = Math.sin(seconds * spore.swayRate * 1.87 + spore.phase * 2.3);
  return spore.lane + (base + detail * 0.42) * spore.sway;
}

/**
 * The fade that hides the recycling.
 *
 * Zero at both ends of the travel band and full in the middle, so a mote is
 * already invisible by the time it reaches the line where it is moved. Not a
 * linear ramp: a half-sine spends most of the travel near full opacity and
 * collapses quickly at the ends, which is what keeps the field from looking
 * like it is fading in and out everywhere at once.
 */
export function sporeFade(y: number): number {
  const t = (y + EDGE) / (1 + 2 * EDGE);
  if (t <= 0 || t >= 1) return 0;
  return Math.sin(Math.PI * t);
}

/**
 * The slow breath on one mote's ink.
 *
 * Bounded in [1 − twinkle, 1] rather than centred on 1: a mote may only ever be
 * dimmer than its ink, never brighter, or `MAX_INK` would stop being the
 * ceiling on the brightest pixel in the field and every assertion about the
 * layer's weight would be off by the shimmer's amplitude.
 */
export function sporeShimmer(spore: Spore, seconds: number): number {
  const wave = 0.5 + 0.5 * Math.sin(seconds * spore.twinkleRate * Math.PI * 2 + spore.phase * 1.7);
  return 1 - spore.twinkle + spore.twinkle * wave;
}

/** Final alpha for one mote on one frame. */
export function sporeAlpha(spore: Spore, seconds = 0): number {
  return spore.ink * sporeFade(spore.y) * sporeShimmer(spore, seconds);
}

/**
 * Draw order: far motes first.
 *
 * Returns indices into the field rather than a sorted copy, because the field's
 * own order is load-bearing — `refitSporeField` keeps a prefix, and a sorted
 * field would make "the first N motes" a different set after every resize. The
 * order only has to be recomputed when the field's length changes.
 */
export function sporeDrawOrder(field: readonly Spore[]): number[] {
  return field
    .map((spore, index) => ({ depth: spore.depth, index }))
    .sort((a, b) => a.depth - b.depth)
    .map((entry) => entry.index);
}
