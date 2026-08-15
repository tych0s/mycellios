import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { growFruitingBody, type Point } from "./fruiting-body";
import { ORIGIN_FRACTION } from "./GutterFruiting";

/*
 * The fruiting body at the exchange section's bottom edge.
 *
 * Everything that can go wrong with it is geometric and silent: a view box that
 * no longer contains the specimen clips the cap, a dash pattern shorter than
 * its path leaves a permanent gap in the outline, and an origin constant that
 * has drifted from the stylesheet leaves the organism hovering above the rule
 * it is supposed to grow out of. None of those throw, none of them fail a
 * render test, and every one of them looks like a styling mistake rather than a
 * stale number — so the numbers are checked against the generator itself here
 * instead of being trusted.
 */

const read = (path: string) => readFile(new URL(path, import.meta.url), "utf8");

/*
 * Source with the comments taken out.
 *
 * Anything below that asserts a declaration exists — or, worse, that one does
 * *not* exist — has to read the code and not the prose around it. The rules in
 * this stylesheet carry long notes that quote the exact values they rejected and
 * the exact properties they must never use, so a naive match finds the argument
 * against a thing and reports it as the thing. Three of these tests failed that
 * way on their first run: the note explaining that there is deliberately no
 * `scaleX(-1)` contains "scaleX(-1)", and the note giving the support
 * launcher's `right:24px` looks exactly like this element's own offset.
 */
const code = (source: string) => source.replace(/\/\*[\s\S]*?\*\//g, "");

const SCALE = 78;
const px = (point: Point) => ({ x: point.x * SCALE, y: -point.y * SCALE });
const length = (points: readonly Point[]) => {
  let total = 0;
  for (let i = 1; i < points.length; i += 1) {
    const a = px(points[i - 1]!);
    const b = px(points[i]!);
    total += Math.hypot(b.x - a.x, b.y - a.y);
  }
  return total;
};

/* The same specimen the component grows. If its parameters change there, this
   fails loudly rather than the drawing quietly going out of its box. */
const body = growFruitingBody({ maturity: 0.68, lean: -0.42, gillCount: 0 });

describe("the specimen fits the box that is drawn around it", () => {
  it("grows the pose the constants were fitted to", async () => {
    const component = await read("./GutterFruiting.tsx");
    // Every measured number below assumes this specimen. Changing it without
    // refitting the box is the failure these tests exist to catch, so the pose
    // is asserted rather than inferred.
    expect(component).toMatch(/MATURITY = 0\.68/);
    expect(component).toMatch(/LEAN = -0\.42/);
    expect(component).toContain("gillCount: 0");
  });

  it("keeps every stroke inside the view box", async () => {
    const component = await read("./GutterFruiting.tsx");
    const box = component.match(/VIEW_LEFT = (-?\d+)[\s\S]*?VIEW_WIDTH = (\d+)[\s\S]*?VIEW_TOP = (-?\d+)/);
    expect(box).not.toBeNull();
    const [left, width, top] = box!.slice(1).map(Number) as [number, number, number];
    const stipeWidth = Number(component.match(/STIPE_WIDTH = (\d+)/)![1]);

    const points = [...body.stipe, ...body.cap, ...body.underside].map(px);
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);

    // Half the stalk's width, because a stroke is painted outside its path and
    // the stalk is by far the widest one here — a margin sized for the cap's
    // 1.4px rim would shave the stalk's flank off against the clip.
    const margin = stipeWidth / 2;
    expect(Math.min(...xs)).toBeGreaterThanOrEqual(left + margin);
    expect(Math.max(...xs)).toBeLessThanOrEqual(left + width - margin);
    expect(Math.min(...ys)).toBeGreaterThanOrEqual(top + margin);
    // The base of the stalk is the origin, y=0, and the root runs below it.
    expect(Math.max(...ys)).toBeCloseTo(0, 5);
  });

  it("puts the base of the stalk where the stylesheet expects the ground", async () => {
    const css = await read("./rebrand.css");
    // The wrapper's bottom edge is the section's rule; the svg is scaled up so
    // the origin lands on it and the root runs underneath. If the two ever
    // disagree the organism floats above the line or is buried in it.
    const rule = css.match(/\.rb-gutter-fruiting svg \{[^}]*\}/)?.[0] ?? "";
    const scaled = Number(rule.match(/height:([\d.]+)%/)?.[1]);
    expect(scaled).toBeCloseTo(100 / ORIGIN_FRACTION, 0);

    const wrapper = css.match(/\.rb-gutter-fruiting \{[^}]*\}/)?.[0] ?? "";
    const ratio = wrapper.match(/aspect-ratio:(\d+)\/(\d+)/);
    const w = Number(ratio?.[1]);
    const h = Number(ratio?.[2]);
    // The wrapper is the above-ground body only, so its ratio is the view box's
    // width against the part of it above the origin — not the whole box.
    const component = await read("./GutterFruiting.tsx");
    const width = Number(component.match(/VIEW_WIDTH = (\d+)/)![1]);
    const above = -Number(component.match(/VIEW_TOP = (-?\d+)/)![1]);
    expect(w / h).toBeCloseTo(width / above, 2);
  });
});

describe("it is a mushroom rather than a parasol", () => {
  it("closes the cap into one filled silhouette instead of stroking two arcs", async () => {
    const component = await read("./GutterFruiting.tsx");
    const css = await read("./rebrand.css");
    // This is the whole design. An arc over a line, with or without ribs under
    // it, is an umbrella — the rim only reads as an overhang when there is
    // solid cap above it. So `cap` and `underside` are one closed path, and it
    // is filled.
    expect(component).toContain("capSilhouette(");
    expect(component).toMatch(/path\(underside\)\.replace\(\/\^M\/, "L"\)/);
    expect(component).toMatch(/Z`/);
    const cap = css.match(/\.rb-gutter-fruiting \.rb-gutter-cap \{[^}]*\}/)?.[0] ?? "";
    // Filled with *something* — the paint may be a flat colour or a ramp, but
    // it must not be `none`, which would leave the outline it was drawn to
    // replace. What matters is that the cap has mass, not which paint gives it.
    expect(cap).toMatch(/fill:(?!none)/);
  });

  it("draws no gills, which are the other half of the parasol", async () => {
    const component = await read("./GutterFruiting.tsx");
    expect(component).toContain("gillCount: 0");
    expect(component).not.toContain("body.gills.map");
  });

  it("gives the stalk real width, and swells it towards the ground", async () => {
    const component = await read("./GutterFruiting.tsx");
    const width = Number(component.match(/STIPE_WIDTH = (\d+)/)![1]);

    /* Measured against the *mature* cap, not this specimen's. `capRadius` grows
       with maturity, so a ratio taken from a young pose tightens as the pose
       gets younger — the bound would fail on a body that had simply been tuned
       to look less open, which is a proportion this test has no business
       policing. The species' full span is what "a tenth of the cap" means. */
    const span = growFruitingBody({ maturity: 1 }).capRadius * 2 * SCALE;
    expect(width / span).toBeGreaterThan(0.08);
    expect(width / span).toBeLessThan(0.2);

    // A constant-width stalk is a dowel; a real one is widest where it meets
    // the ground. The outline is built from those two widths, so the base must
    // stay the broader of the pair.
    const [base, top] = component
      .match(/stipeOutline\(\[\.\.\.body\.stipe\], STIPE_WIDTH \* ([\d.]+), STIPE_WIDTH \* ([\d.]+)\)/)!
      .slice(1)
      .map(Number) as [number, number];
    expect(base).toBeGreaterThan(top);
  });

  it("carries its transparency on the group and never on the shapes", async () => {
    const component = await read("./GutterFruiting.tsx");
    const css = await read("./rebrand.css");
    /* The defect this pins is the one that made the first version look drawn
       rather than grown: the stalk runs behind the cap, so per-shape alpha
       composites the overlap twice and the stalk shows through the cap as a
       dark vertical stripe. Opaque shapes inside a group with a single opacity
       is what makes the body occlude itself like a solid.

       It is checked here rather than left to review because it is invisible in
       the geometry and invisible in a passing render test — the shapes are
       correct either way, and only the compositing is wrong. */
    expect(component).toContain('<g className="rb-gutter-body"');
    const block = css.slice(css.indexOf(".rb-gutter-fruiting {"), css.indexOf(".rb-pay-head {"));
    expect(block).toMatch(/\.rb-gutter-body \{[^}]*opacity:/);

    // No fill inside the body may be translucent: not an rgba() with alpha, and
    // not a fill-opacity that rests below 1 once the growth has finished.
    for (const klass of ["rb-gutter-stipe", "rb-gutter-cap"]) {
      const rule = block.match(new RegExp(`\\.${klass} \\{[^}]*fill:[^;]*;`))?.[0] ?? "";
      expect(rule).not.toMatch(/rgba\([^)]*,\s*0?\.\d+\s*\)/);
    }
  });

  it("stays texture beside the pricing table rather than a second subject", async () => {
    const css = await read("./rebrand.css");
    const block = css.slice(css.indexOf(".rb-gutter-fruiting {"), css.indexOf(".rb-pay-head {"));
    /* The cap is allowed a shallow crown-to-rim ramp so it reads as a body with
       a near side, but the section's subject is the pricing rail. Blur, shadow
       and glow are what turn a decoration into a focal point, so those stay
       out; and the group's opacity is what keeps the whole thing quiet. */
    expect(block).not.toContain("filter:");
    expect(block).not.toContain("drop-shadow");
    const groupAlpha = Number(block.match(/\.rb-gutter-body \{[^}]*opacity:([\d.]+)/)?.[1]);
    expect(groupAlpha).toBeGreaterThan(0);
    expect(groupAlpha).toBeLessThanOrEqual(0.55);
  });

  it("keeps material depth in the mobile SVG instead of reverting to a flat cut-out", async () => {
    const component = await read("./GutterFruiting.tsx");
    const css = await read("./rebrand.css");

    expect(component).toContain('<radialGradient id="rbGutterCap"');
    expect(component).toContain('<linearGradient id="rbGutterStipe"');
    expect(component).toContain('filter="url(#rbGutterDepth)"');
    const phone = css.slice(css.indexOf("@media(max-width:700px){ .rb-gutter-fruiting"));
    expect(phone).toMatch(/\.rb-gutter-body \{ opacity:\.9;/);
  });
});

describe("the mark arrives grown rather than popped", () => {
  it("gives every dashed stroke a pattern at least as long as its own path", async () => {
    const css = await read("./rebrand.css");
    const dashOf = (klass: string) =>
      Number(css.match(new RegExp(`\\.rb-gutter-fruiting \\.${klass} \\{[^}]*--len:(\\d+)`))?.[1]);

    // The rim is the only stroked part of the body left; the stalk and cap are
    // fills so the body can be one solid, and fills cannot be dashed.
    expect(dashOf("rb-gutter-rim")).toBeGreaterThanOrEqual(length(body.underside));
  });

  it("fades the fills in rather than dashing them, since a fill cannot be dashed", async () => {
    const css = await read("./rebrand.css");
    // Left on the dash rule these would simply appear at full strength on the
    // first frame — a button floating over a stalk that has not risen yet.
    expect(css).toMatch(/\.rb-gutter-fruiting \.rb-gutter-stipe,\s*\.rb-gutter-fruiting \.rb-gutter-cap \{[^}]*fill-opacity:0/);
    expect(css).toMatch(/is-visible \.rb-gutter-stipe \{[^}]*animation:rbFruitFill/);
    expect(css).toMatch(/is-visible \.rb-gutter-cap \{[^}]*animation:rbFruitFill/);
  });

  it("finishes drawn rather than invisible when motion is reduced", async () => {
    const css = await read("./rebrand.css");
    const reduced = css.slice(css.indexOf("@media(prefers-reduced-motion:reduce)"));
    // The blanket .01ms rule shortens each animation but leaves `animation-delay`
    // alone, so without this the rim sits invisible for most of a second and
    // then snaps in — and if the animations were dropped without writing their
    // end states, the outline and the cap would never appear at all.
    expect(reduced).toMatch(/\.rb-gutter-fruiting path \{[^}]*animation:none/);
    expect(reduced).toMatch(/\.rb-gutter-fruiting path \{[^}]*stroke-dashoffset:0/);
    // Both fills, or the body finishes as a cap with no stalk under it.
    expect(reduced).toMatch(
      /\.rb-gutter-fruiting \.rb-gutter-stipe,\s*\.rb-gutter-fruiting \.rb-gutter-cap \{[^}]*fill-opacity:1/,
    );
  });
});

describe("it stays scenery", () => {
  it("is placed by the section, not by the page's content stacking rule", async () => {
    const component = await read("./GutterFruiting.tsx");
    const layers = await read("./mycelium-network.css");
    // That selector is more specific than any single class, so a decorative
    // child without the opt-out is forced back into the flow no matter what its
    // own rules say — the element would simply appear in the middle of the copy.
    expect(component).toContain("rb-decor");
    expect(layers).toContain(":not(.rb-hero):not(.rb-story) > *:not(.rb-decor)");
  });

  it("is inert to pointers and to screen readers", async () => {
    const component = await read("./GutterFruiting.tsx");
    const css = await read("./rebrand.css");
    expect(component).toMatch(/className="rb-gutter-fruiting[^"]*" aria-hidden="true"/);
    expect(css).toMatch(/\.rb-gutter-fruiting \{[^}]*pointer-events:none/);
  });

  it("grows from the shared generator so it stays the same species", async () => {
    const component = await read("./GutterFruiting.tsx");
    // A hand-drawn path here would silently become a different organism the
    // first time the other three are re-tuned.
    expect(component).toContain("growFruitingBody({");
    expect(component).not.toMatch(/const \w+Path = "M/);
  });
});

describe("it stays inside the empty band it was given", () => {
  it("is never taller than the section padding it stands in, at any viewport", async () => {
    /* Comment-stripped: the placement note in this rule quotes the *stage*
       variant's `height:clamp(88px,11.5vh,134px)` while explaining the offset,
       and `height:[^;]+` would otherwise pick that up instead of the flat body's
       own height and then fail to parse its fractional vh term. */
    const css = code(await read("./rebrand.css"));
    /* Both are clamps, and the guarantee has to hold across the whole range —
       not just at the one width someone happened to check. Comparing term by
       term is what makes it hold everywhere: min, preferred and max of the
       body must each stay under the matching term of the padding, or the cap
       rises into the last line of copy at some viewport height nobody tested. */
    const terms = (declaration: string) =>
      declaration.match(/clamp\((\d+)px,(\d+)vh,(\d+)px\)/)!.slice(1).map(Number);

    const pay = css.match(/\.rb-pay \{[^}]*\}/)![0];
    const mark = css.match(/\.rb-gutter-fruiting \{[^}]*\}/)![0];
    const padding = terms(pay.match(/padding-block:[^;]+/)![0]);
    const height = terms(mark.match(/height:[^;]+/)![0]);

    height.forEach((term, index) => expect(term).toBeLessThan(padding[index]!));
  });

  it("shrinks under the section's own narrow-viewport padding too", async () => {
    const css = await read("./rebrand.css");
    // Below 700px `.rb-pay` drops to a flat 80px, which is under the desktop
    // body's 78–116px range — so the body needs its own smaller value there.
    const phone = css.slice(css.indexOf("@media(max-width:700px)"));
    expect(phone).toMatch(/\.rb-models,\.rb-pay[^{]*\{[^}]*padding-block:80px/);
    const shrunk = Number(
      css.match(/@media\(max-width:700px\)\{ \.rb-gutter-fruiting[^}]*height:(\d+)px/)?.[1],
    );
    expect(shrunk).toBeLessThan(80);
  });
});

/*
 * Which side of the page it stands on.
 *
 * Moving it to the right margin put it into the one region of this page that
 * something else already owns: the support assistant's launcher is
 * `position:fixed` in the bottom-right corner of the *viewport*, so it is not
 * enough for the body to be inside the section's empty padding — it has to miss
 * a box that floats over every section at some scroll position. Both boxes are
 * measured from their own stylesheets here rather than assumed, because the
 * symptom is a mushroom cap sliding behind a chat button mid-scroll, which
 * nobody would read as decoration.
 */
describe("it stands clear of everything else in that corner", () => {
  const assistant = () => read("../support-assistant.css");

  it("is anchored to the right edge on desktop", async () => {
    const css = code(await read("./rebrand.css"));
    const wrapper = css.match(/\.rb-gutter-fruiting \{[^}]*\}/)![0];
    /* The whole point of the change. `left` here would put it back where it
       was, and both would fight. */
    expect(wrapper).toMatch(/[^-]right:\d+px/);
    expect(wrapper).not.toMatch(/[^-]left:/);
  });

  it("clears the fixed support launcher at every desktop viewport", async () => {
    const css = code(await read("./rebrand.css"));
    const support = code(await assistant());

    /* The launcher's own numbers, from its own stylesheet. */
    const root = support.match(/\.support-assistant \{[^}]*\}/)![0];
    const launcherRight = Number(root.match(/right:\s*(\d+)px/)![1]);
    const launcher = support.match(/\.support-assistant-launcher \{[^}]*\}/)![0];
    /* It is a flex row of a 23px icon plus a two-line label, so its rendered
       width is not in the CSS. `min-width` is the floor; the measured width in
       the shipped screenshot is 147px. Use the larger, since a wider launcher is
       what would reach the body. */
    const launcherWidth = Math.max(147, Number(launcher.match(/min-width:\s*(\d+)px/)![1]));
    const launcherInnerEdge = launcherRight + launcherWidth;

    const wrapper = css.match(/\.rb-gutter-fruiting \{[^}]*\}/)![0];
    const bodyRight = Number(wrapper.match(/[^-]right:(\d+)px/)![1]);

    /* The body starts further in than the launcher ends. Both are measured from
       the same edge, so this is the whole collision test in one comparison. */
    expect(bodyRight).toBeGreaterThan(launcherInnerEdge);
  });

  it("keeps the body inside the page rather than off the right edge", async () => {
    const css = code(await read("./rebrand.css"));
    const wrapper = css.match(/\.rb-gutter-fruiting \{[^}]*\}/)![0];
    const bodyRight = Number(wrapper.match(/[^-]right:(\d+)px/)![1]);
    const stage = css.match(/\.rb-gutter-stage \{[^}]*\}/)![0];

    /* Widest the rendered box ever gets: the top of its height clamp against
       its aspect ratio. If the offset ever grew past the point where the far
       flank crosses into the content column's centre, the body would be
       standing in the copy rather than beside it. */
    const maxHeight = Number(stage.match(/clamp\(\d+px,[\d.]+vh,(\d+)px\)/)![1]);
    const aspect = Number(stage.match(/aspect-ratio:([\d.]+)/)![1]);
    const farFlank = bodyRight + maxHeight * aspect;

    /* Half of the narrowest desktop viewport this rule applies to (701px), so
       the body can never reach the middle of the page. */
    expect(farFlank).toBeLessThan(701 / 2);
  });

  it("goes back to the left edge on phones, where there is no room to clear it", async () => {
    const css = code(await read("./rebrand.css"));
    const support = code(await assistant());

    /* Under 620px the launcher collapses to an icon — but the viewport collapses
       further, so there is no longer a gap to the left of it that fits a body.
       The left edge is empty at this width, and the desktop `right` has to be
       actively cancelled or the two rules would both apply. */
    const phone = support.slice(support.indexOf("@media (max-width: 620px)"));
    expect(phone).toMatch(/\.support-assistant-launcher \{[^}]*width:\s*56px/);

    const rule = css.match(/@media\(max-width:700px\)\{ \.rb-gutter-fruiting[^}]*\}/)![0];
    expect(rule).toContain("left:4px");
    expect(rule).toContain("right:auto");
  });

  it("is moved rather than mirrored, so the page keeps one light source", async () => {
    /*
     * The subtle one. A `scaleX(-1)` would have been the cheap way to face the
     * body into the page, and it would also have flipped its shading: the
     * scene's key light is at negative x, so every specimen on this page is lit
     * from the left. A mirrored one would be lit from the right and the page
     * would have two suns — permanent, and invisible in a diff.
     */
    const css = code(await read("./rebrand.css"));
    const wrapper = css.match(/\.rb-gutter-fruiting \{[^}]*\}/)![0];
    const stage = css.match(/\.rb-gutter-stage \{[^}]*\}/g)!.join("");
    expect(wrapper).not.toMatch(/scaleX|rotateY/);
    expect(stage).not.toMatch(/scaleX|rotateY/);
  });

  it("keeps the key light on the left, which is what forbids the mirror", async () => {
    /* Read from the renderer, so that moving the light would fail here and
       force the placement argument above to be revisited rather than silently
       becoming false. */
    const renderer = await read("./vendor/mushroom-stage.js");
    const key = renderer.match(/const key = new THREE\.DirectionalLight[^\n]*/)![0];
    const x = Number(key.match(/key\.position\.set\((-?[\d.]+)/)![1]);
    expect(x).toBeLessThan(0);
  });
});

/*
 * The rendered specimen — the one desktop actually gets.
 *
 * Two things here fail silently and expensively. The first is drift in the
 * hero: its look lives in tuned constants that were lifted into a shared table
 * so a second body could be grown from them, and nothing about editing the
 * second body's numbers would tell you that you had nudged the first. The
 * second is the framing, which carries a claim — the stem is *cut* by the
 * section's rule rather than standing on it — that is true only for a
 * particular camera against a particular box ratio, and is a positioning bug
 * rather than an error when it goes stale.
 */
describe("the rendered specimen", () => {
  it("keeps the hero on the reference's own constants", async () => {
    /* The whole safety property of the VARIANTS table. Parameterising a look
       that lives in a hundred tuned numbers is only safe if the original
       numbers survive verbatim, so they are pinned as literals here: tuning
       the gutter body cannot quietly move the hero. */
    const { VARIANTS } = await import("./vendor/mushroom-stage.js");
    expect(VARIANTS.hero.capScale).toEqual([1.1, 0.86, 1.1]);
    expect(VARIANTS.hero.stemBend).toBe(-0.075);
    expect(VARIANTS.hero.gillFactor).toBe(1);
    expect(VARIANTS.hero.camera).toEqual({ x: 0.30, y: -1.02, z: 5.15, fov: 26, target: [0, 0.36, 0] });
    expect(VARIANTS.hero.tune).toEqual({ halo: 1, glow: 1, shell: 1, bloom: 0.68, threshold: 0.42 });
    expect(VARIANTS.hero.capTop[0]).toEqual([0, 0.955]);
    expect(VARIANTS.hero.capTop.at(-1)).toEqual([1.08, 0.302]);
    expect(VARIANTS.hero.capUnder.at(-1)).toEqual([1.08, 0.302]);
    expect(VARIANTS.hero.stem[1]).toEqual([0.078, 0.548]);
    expect(VARIANTS.hero.stem.at(-1)).toEqual([0, -1.18]);
  });

  it("grows a different organism rather than the hero scaled down", async () => {
    /* Two identical bodies on one page is wallpaper. The difference has to be
       in the silhouette — which is all anyone reads at 110px in a margin — not
       in the size, so it is measured as a proportion. The hero's cap is wider
       than it is tall; this one's is the other way round, which is what a cap
       that has not opened yet looks like. */
    const { VARIANTS } = await import("./vendor/mushroom-stage.js");
    const shape = (v: (typeof VARIANTS)["hero"]) => {
      const rim = v.capUnder.at(-1)![0] * v.capScale[0];
      const crown = v.capTop[0]![1] * v.capScale[1];
      const junction = v.capUnder[0]![1] * v.capScale[1];
      return (crown - junction) / (rim * 2); // cap depth against cap span
    };
    expect(shape(VARIANTS.hero)).toBeLessThan(0.5);
    expect(shape(VARIANTS.gutter)).toBeGreaterThan(0.5);
    // And it stands on a visibly thinner stem, not merely a shorter one.
    const waist = (v: (typeof VARIANTS)["hero"]) => v.stem[1]![0];
    expect(waist(VARIANTS.gutter)).toBeLessThan(waist(VARIANTS.hero) * 0.7);
  });

  it("frames the body so the section's rule cuts the stem", async () => {
    /*
     * The claim the framing makes, checked against the renderer's own camera
     * rather than trusted. A body that stops just above the rule is an object
     * resting on a line; the stem has to run past the bottom edge and be
     * clipped there, which is what growing out of a surface looks like.
     *
     * Projected through the real camera at the real box ratio, so a change to
     * either the profile table or the CSS aspect fails here instead of showing
     * up as a mushroom hovering over the border.
     */
    const [{ VARIANTS }, { PerspectiveCamera, Vector3 }] = await Promise.all([
      import("./vendor/mushroom-stage.js"),
      import("three"),
    ]);
    const css = await read("./rebrand.css");
    const aspect = Number(css.match(/\.rb-gutter-stage \{[^}]*aspect-ratio:([\d.]+)/)![1]);

    const V = VARIANTS.gutter;
    const camera = new PerspectiveCamera(V.camera.fov, aspect, 0.1, 60);
    /* The settled pose from the tick loop, which drifts 0.04 right of where
       boot() places it — that drift is part of the framing, so it is included
       rather than measured against the boot position the eye never sees. */
    camera.position.set(V.camera.x + 0.04, V.camera.y, V.camera.z);
    camera.lookAt(new Vector3(...V.camera.target));
    camera.updateMatrixWorld();
    camera.updateProjectionMatrix();

    const [sx, sy, sz] = V.capScale;
    let left = 1, right = 0, top = 1, bottom = 0;
    const add = (x: number, y: number, z: number) => {
      const q = new Vector3(x, y, z).project(camera);
      const fx = (q.x + 1) / 2;
      const fy = 1 - (q.y + 1) / 2;
      left = Math.min(left, fx); right = Math.max(right, fx);
      top = Math.min(top, fy); bottom = Math.max(bottom, fy);
    };
    for (const [r, y] of V.capTop) for (const s of [-1, 1]) { add(r * sx * s, y * sy, 0); add(0, y * sy, r * sz * s); }
    for (const [r, y] of V.stem) {
      const k = Math.max(0, 0.56 - y) / 1.74;
      const bend = V.stemBend * Math.pow(k, 1.75);
      for (const s of [-1, 1]) add(r * s + bend, y, 0);
    }

    // Cut, not resting: the foot of the stem is past the wrapper's bottom edge.
    expect(bottom).toBeGreaterThan(1);
    // But only just — a body mostly below the rule is a body that is buried.
    expect(bottom).toBeLessThan(1.2);
    // The crown stays in frame, with room for the sway the tick loop adds.
    expect(top).toBeGreaterThan(0.01);
    expect(top).toBeLessThan(0.2);
    // And it does not touch the sides, which would read as a clipping bug.
    expect(left).toBeGreaterThan(0.02);
    expect(right).toBeLessThan(0.98);
  });

  it("stays subordinate to the pricing rail by emitting less, not by going flat", async () => {
    /*
     * The correction this specimen exists to make. The first version of this
     * body was a flat drawing, on the reasoning that a decoration must not
     * compete with the pricing table — but what prevents competition is low
     * contrast, not absence of volume, and a flat organism next to a rendered
     * one just reads as the worse of the two. So the check is that every
     * emissive term is well under the hero's while the geometry stays real.
     */
    const { VARIANTS } = await import("./vendor/mushroom-stage.js");
    const { tune } = VARIANTS.gutter;
    expect(tune.halo).toBe(0);
    expect(tune.glow).toBeLessThan(0.5);
    expect(tune.shell).toBeLessThan(0.5);
    expect(tune.bloom).toBeLessThan(VARIANTS.hero.tune.bloom * 0.5);
    // A higher bloom threshold means less of the body passes into the glow.
    expect(tune.threshold).toBeGreaterThan(VARIANTS.hero.tune.threshold);
    // Still a real body: the profile is geometry, not a silhouette to fill.
    expect(VARIANTS.gutter.capTop.length).toBeGreaterThan(8);
  });

  it("mounts the quiet configuration of the renderer", async () => {
    const component = await read("./GutterFruiting.tsx");
    expect(component).toContain('setAttribute("variant", "gutter")');
    /* Spores drift upward across whatever is above them, which here is the
       pricing table, and they are the hero's signature besides. Motion is
       halved because this sits in the corner of the eye. */
    expect(component).toContain('setAttribute("spores", "off")');
    expect(component).toContain('reduceMotion ? "off" : "calm"');
  });

  it("uses the same rendered organism on phones with a bounded quality profile", async () => {
    /*
     * Visual identity stays invariant: phones and reduced-motion visitors keep
     * the real gutter geometry. Only renderer cost and motion change.
     */
    const component = await read("./GutterFruiting.tsx");
    expect(component).toContain('matchMedia("(max-width: 700px)")');
    expect(component).toContain('setAttribute("quality", "mobile")');
    expect(component).toContain('matchMedia("(prefers-reduced-motion: reduce)")');
    expect(component).toMatch(/return <GutterFruitingStage \/>/);
    expect(code(component)).not.toContain("wantsMushroom()");
    // And the module is the hero's, so the chunk is fetched once for the page.
    expect(component).toContain('from "./HeroMushroom"');
    expect(component).toContain("loadMushroomStage()");
  });

  it("positions the rendered organism in the empty left corner on phones", async () => {
    const css = await read("./rebrand.css");
    const phone = css.match(/@media\(max-width:700px\)\{[^\n]*\.rb-gutter-stage \{[^}]*\}/)?.[0] ?? "";
    expect(phone).toContain("right:auto");
    expect(phone).toContain("left:4px");
    expect(phone).toContain("height:88px");
  });

  it("waits until the section is near before paying for a context", async () => {
    /* The bytes are shared with the hero, but a context that renders every
       frame behind three screens of scroll is GPU spent on nothing. */
    const component = await read("./GutterFruiting.tsx");
    expect(component).toContain("IntersectionObserver");
    expect(component).toContain("observer.disconnect()");
  });

  it("fades the rendered body in rather than letting it pop", async () => {
    /* Shader compilation cannot be removed, so it is covered: the host is
       transparent until the element's first drawn frame. A boot that never
       arrives (no WebGL, a lost context) must not leave it invisible forever,
       hence the failsafe. */
    const [component, css] = await Promise.all([read("./GutterFruiting.tsx"), read("./rebrand.css")]);
    expect(component).toContain("mushroom-ready");
    expect(component).toMatch(/setTimeout\(\(\) => setShown\(true\), 4000\)/);
    expect(css).toMatch(/\.rb-gutter-stage \{[^}]*opacity:0/);
    expect(css).toMatch(/\.rb-gutter-stage\.is-grown \{[^}]*opacity:1/);
  });

  it("is never taller than the section padding it stands in, at any viewport", async () => {
    /* Same guarantee as the flat body and the same reasoning: term by term, so
       it holds at every viewport height at once rather than at the two someone
       happened to check. This specimen is taller, so it is the one at risk. */
    const css = await read("./rebrand.css");
    const terms = (declaration: string) =>
      declaration.match(/clamp\(([\d.]+)px,([\d.]+)vh,([\d.]+)px\)/)!.slice(1).map(Number);
    const padding = terms(css.match(/\.rb-pay \{[^}]*\}/)![0].match(/padding-block:[^;]+/)![0]);
    const height = terms(css.match(/\.rb-gutter-stage \{[^}]*height:[^;]+/)![0]);
    height.forEach((term, index) => expect(term).toBeLessThan(padding[index]!));
  });
});
