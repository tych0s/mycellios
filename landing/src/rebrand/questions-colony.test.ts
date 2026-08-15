import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

/*
 * The three fruiting bodies in the questions column.
 *
 * What can go wrong here is the same class of thing that can go wrong with the
 * gutter specimen — silent and geometric — plus one failure the single-body
 * specimens could not have: the three can quietly become the same mushroom
 * three times. Nothing throws when that happens. The page just stops having a
 * colony in it and starts having a repeated ornament, which is exactly the
 * mistake the whole arrangement exists to avoid, and which looks like a design
 * choice rather than a regression.
 *
 * So the ages are checked as proportions, the framing is solved through the
 * real camera against all three bodies at once, and the context budget — the
 * technical reason this is one element and not three — is asserted rather than
 * left as a comment someone can delete.
 */

const read = (path: string) => readFile(new URL(path, import.meta.url), "utf8");

describe("the colony is three different ages of one species", () => {
  it("holds three bodies", async () => {
    const { VARIANTS } = await import("./vendor/mushroom-stage.js");
    expect(VARIANTS.colony.colony).toHaveLength(3);
  });

  it("gives each body its own silhouette rather than one body at three sizes", async () => {
    /*
     * The load-bearing check. Three copies of a profile at different `scale`
     * values would pass any test that only looked at rendered size, and would
     * read as wallpaper — so the comparison deliberately ignores `scale` and
     * measures the shape of the untransformed profile.
     *
     * The proportion used is stem thickness against cap span, because that is
     * what actually separates ages in this organism: a button is a knob on a
     * thick short stalk, and an old cap is a wide thin plate on a wire. It has
     * to be monotonic — youngest thickest — or the three are not a sequence.
     */
    const { VARIANTS } = await import("./vendor/mushroom-stage.js");
    const bodies = VARIANTS.colony.colony;
    const span = (b: (typeof bodies)[number]) => b.capTop.at(-1)![0] * 2;
    const stemR = (b: (typeof bodies)[number]) => b.stem[1]![0];
    const ratio = bodies.map((b) => stemR(b) / span(b));

    // Oldest to youngest, each one's stem proportionally thicker than the last.
    expect(ratio[0]!).toBeLessThan(ratio[1]!);
    expect(ratio[1]!).toBeLessThan(ratio[2]!);
    // And the separation is a real one, not a rounding difference.
    expect(ratio[2]!).toBeGreaterThan(ratio[0]! * 3);
    // No two bodies share a cap span, so none is another's profile reused.
    expect(new Set(bodies.map(span)).size).toBe(3);
  });

  it("makes the oldest cap turn up at the brim and the youngest stay closed", async () => {
    /*
     * Age read a second, independent way, because the stem ratio alone would
     * still pass if all three caps were the same dome. `brim` is how far the
     * rim sits above the lowest point of the margin: positive and large is a
     * cap that has gone past flat and turned up, which is what an old one does;
     * near zero is a knob whose margin is still tucked against the stem.
     */
    const { VARIANTS } = await import("./vendor/mushroom-stage.js");
    const brim = (b: (typeof VARIANTS.colony.colony)[number]) => {
      const rim = b.capTop.at(-1)![1];
      const low = Math.min(...b.capTop.map((p) => p[1]));
      return rim - low;
    };
    const bodies = VARIANTS.colony.colony;
    const [old, middle, young] = [bodies[0]!, bodies[1]!, bodies[2]!];
    expect(brim(old)).toBeGreaterThan(brim(middle));
    expect(brim(middle)).toBeGreaterThan(brim(young));
    // The old one's upturn is pronounced enough to see, not a numerical wobble.
    expect(brim(old)).toBeGreaterThan(0.08);

    /* And the crowns run the other way: a closed knob is tall relative to its
       span, an opened plate is low. Two orthogonal measures of the same
       sequence, so one of them drifting cannot pass unnoticed. */
    const height = (b: (typeof VARIANTS.colony.colony)[number]) =>
      b.capTop[0]![1] / (b.capTop.at(-1)![0] * 2);
    expect(height(young)).toBeGreaterThan(height(middle));
    expect(height(middle)).toBeGreaterThan(height(old));
  });

  it("differs from the two specimens already on the page", async () => {
    /* Five bodies across the page, five silhouettes. A colony member that
       duplicated the hero or the gutter would make the page repeat itself
       across sections instead of within one. */
    const { VARIANTS } = await import("./vendor/mushroom-stage.js");
    const key = (b: { capTop: readonly (readonly number[])[]; stem: readonly (readonly number[])[] }) =>
      JSON.stringify([b.capTop, b.stem]);
    const all = [VARIANTS.hero, VARIANTS.gutter, ...VARIANTS.colony.colony].map(key);
    // The colony's own top-level profile is intentionally its middle member's,
    // so it is excluded here; these are the five drawn bodies.
    expect(new Set(all).size).toBe(5);
  });

  it("stands the bodies on an arc rather than in a row", async () => {
    /*
     * Three bodies at the same z with the same rotation is a row of ornaments;
     * what makes a patch of ground read as one place is that the front body
     * overlaps the back one and no two present the same profile. Both are
     * placement, so both are checked here rather than trusted to the eye.
     */
    const { VARIANTS } = await import("./vendor/mushroom-stage.js");
    const bodies = VARIANTS.colony.colony;
    const z = bodies.map((b) => b.at[2]);
    expect(new Set(z).size).toBe(3);
    // Real depth, not three bodies on almost the same plane.
    expect(Math.max(...z) - Math.min(...z)).toBeGreaterThan(0.5);
    // Each turned a different way, so no two show the same face.
    const rot = bodies.map((b) => b.rotY);
    expect(new Set(rot).size).toBe(3);
    for (let i = 0; i < rot.length; i += 1)
      for (let j = i + 1; j < rot.length; j += 1)
        expect(Math.abs(rot[i]! - rot[j]!)).toBeGreaterThan(0.3);
    // The youngest is the small one in front — where the youngest one is.
    const young = bodies[2]!;
    expect(young.scale).toBeLessThan(0.8);
    expect(young.at[2]).toBe(Math.max(...z));
  });
});

describe("the colony is framed by solving, not by eye", () => {
  it("cuts every stem with the bottom edge and keeps every crown in frame", async () => {
    /*
     * The same claim the gutter specimen makes, checked the same way, but over
     * all three bodies at once and across the whole rotation the tick loop
     * sways the group through — not just its resting pose. A framing that only
     * holds in the middle of the sway is a framing that clips a cap for half of
     * every cycle.
     *
     * Projected through the renderer's real camera at the real CSS aspect, so a
     * change to any profile table, any placement, or the stylesheet's
     * `aspect-ratio` fails here rather than showing up as a clipped mushroom.
     */
    const [{ VARIANTS }, { Euler, Matrix4, PerspectiveCamera, Vector3 }] = await Promise.all([
      import("./vendor/mushroom-stage.js"),
      import("three"),
    ]);
    const css = await read("./questions.css");
    const aspect = Number(css.match(/\.rb-quest-colony \{[^}]*aspect-ratio:\s*([\d.]+)/)![1]);
    expect(aspect).toBeGreaterThan(0); // the selector still exists

    const V = VARIANTS.colony;
    const camera = new PerspectiveCamera(V.camera.fov, aspect, 0.1, 60);
    /* The settled pose: boot() places the camera, then the loop drifts it 0.04
       right. That drift is part of the framing the eye actually sees. */
    camera.position.set(V.camera.x + 0.04, V.camera.y, V.camera.z);
    camera.lookAt(new Vector3(...V.camera.target));
    camera.updateMatrixWorld();
    camera.updateProjectionMatrix();

    /* Every point the renderer builds: each lathe row revolved, the stem bend
       applied exactly as the vertex loop applies it, then the body's own
       transform. The cap is measured at the fresnel shell's 1.016, which is the
       outermost surface and therefore the one that clips first. */
    const worldPoints: InstanceType<typeof Vector3>[] = [];
    for (const B of V.colony) {
      const local: InstanceType<typeof Vector3>[] = [];
      const push = (r: number, y: number, stem: boolean) => {
        for (let s = 0; s < 24; s += 1) {
          const a = (s / 24) * Math.PI * 2;
          let x = Math.cos(a) * r;
          let z = -Math.sin(a) * r;
          if (stem) {
            const k = Math.max(0, 0.56 - y) / 1.74;
            const off = B.stemBend * Math.pow(k, 1.75);
            x += off;
            z += off * 0.35;
          }
          local.push(new Vector3(x, y, z));
        }
      };
      const [sx, sy] = B.capScale;
      for (const [r, y] of B.capTop) push(r * sx * 1.016, y * sy, false);
      for (const [r, y] of B.stem) push(r, y, true);

      const place = new Matrix4()
        .makeTranslation(B.at[0], B.at[1], B.at[2])
        .multiply(new Matrix4().makeRotationY(B.rotY))
        .multiply(new Matrix4().makeScale(B.scale, B.scale, B.scale));
      for (const p of local) worldPoints.push(p.applyMatrix4(place));
    }

    /* Every pose the loop can put the group in, not just the resting one: the
       scroll progress both rotates it 0.22 and lifts it 0.14, the pointer adds
       ±0.17 of yaw (halved by `calm`), and the breath adds ±0.0125 of lift. The
       lift is the term that matters here — it raises the feet toward the bottom
       edge, so a framing checked only at rest is a framing that floats the
       moment the reader scrolls. */
    const poses: { gx: number; gy: number; py: number }[] = [];
    for (const g of [0, 0.5, 1])
      for (const px of [-1, 0, 1])
        for (const br of [-1, 1])
          poses.push({
            gx: 0.17 + px * 0.05 * 0.5 + 0.0075,
            gy: 0.18 + px * 0.17 * 0.5 + g * 0.22,
            py: g * 0.14 + br * 0.0125,
          });

    let left = 1;
    let right = 0;
    let top = 1;
    let bottom = 0;
    for (const pose of poses) {
      const g = new Matrix4()
        .makeTranslation(0, pose.py, 0)
        .multiply(new Matrix4().makeRotationFromEuler(new Euler(pose.gx, pose.gy, -0.03, "XYZ")));
      for (const p of worldPoints) {
        const q = p.clone().applyMatrix4(g).project(camera);
        const fx = (q.x + 1) / 2;
        const fy = 1 - (q.y + 1) / 2;
        left = Math.min(left, fx);
        right = Math.max(right, fx);
        top = Math.min(top, fy);
        bottom = Math.max(bottom, fy);
      }
    }

    // Not buried — a group mostly below the edge is a group you cannot see.
    expect(bottom).toBeLessThan(1.34);
    // Crowns clear of the top, with headroom for the sway.
    expect(top).toBeGreaterThan(0.04);
    expect(top).toBeLessThan(0.22);
    // Inside the box on both sides, at every point of the sway.
    expect(left).toBeGreaterThan(0.015);
    expect(right).toBeLessThan(0.985);
    // And filling it, rather than a small group adrift in a large empty block.
    expect(right - left).toBeGreaterThan(0.8);
    // Centred: a group hugging one side reads as a mistake, not a composition.
    expect(Math.abs(left - (1 - right))).toBeLessThan(0.04);
  });

  it("buries every stem's tip below the bottom edge, one body at a time", async () => {
    /*
     * THE ONE THAT MATTERS, and the one the first version got wrong.
     *
     * The check above takes a single worst case over the whole group, so it
     * passes as soon as *one* stem is cut — and that is exactly what happened:
     * each profile kept its own foot depth, and after `scale` and `at[1]` the
     * button's tip projected at 0.989 of the box at rest and 0.911 once
     * scrolled. Its lathe closes on the axis, so what was showing was a rounded
     * shaded cone hanging in mid-air, and three bodies read as floating rather
     * than growing out of anything. The aggregate assertion never noticed,
     * because the other two stems were past the edge and hid it.
     *
     * So this measures the *tip* of each body separately — the cone below the
     * last side row, which is the part that gives away that a stem ended — and
     * requires its highest point to be below the edge in every pose. Aggregate
     * over bodies is the mistake; per-body is the fix.
     */
    const [{ VARIANTS }, { Euler, Matrix4, PerspectiveCamera, Vector3 }] = await Promise.all([
      import("./vendor/mushroom-stage.js"),
      import("three"),
    ]);
    const css = await read("./questions.css");
    const aspect = Number(css.match(/\.rb-quest-colony \{[^}]*aspect-ratio:\s*([\d.]+)/)![1]);

    const V = VARIANTS.colony;
    const camera = new PerspectiveCamera(V.camera.fov, aspect, 0.1, 60);
    camera.position.set(V.camera.x + 0.04, V.camera.y, V.camera.z);
    camera.lookAt(new Vector3(...V.camera.target));
    camera.updateMatrixWorld();
    camera.updateProjectionMatrix();

    const poses: { gx: number; gy: number; py: number }[] = [];
    for (const g of [0, 0.5, 1])
      for (const px of [-1, 0, 1])
        for (const br of [-1, 1])
          poses.push({
            gx: 0.17 + px * 0.05 * 0.5 + 0.0075,
            gy: 0.18 + px * 0.17 * 0.5 + g * 0.22,
            py: g * 0.14 + br * 0.0125,
          });

    const grounds: number[] = [];
    for (const B of V.colony) {
      const stem = B.stem;
      const shoulder = stem.at(-2)!;
      const point = stem.at(-1)!;
      // The tip cone: the last side ring, and the ring where the lathe closes.
      const tip: InstanceType<typeof Vector3>[] = [];
      for (const [r, y] of [shoulder, [0, point[1]] as const]) {
        const k = Math.max(0, 0.56 - y) / 1.74;
        const off = B.stemBend * Math.pow(k, 1.75);
        for (let s = 0; s < 24; s += 1) {
          const a = (s / 24) * Math.PI * 2;
          tip.push(new Vector3(Math.cos(a) * r + off, y, -Math.sin(a) * r + off * 0.35));
        }
      }
      const place = new Matrix4()
        .makeTranslation(B.at[0], B.at[1], B.at[2])
        .multiply(new Matrix4().makeRotationY(B.rotY))
        .multiply(new Matrix4().makeScale(B.scale, B.scale, B.scale));

      let highest = Infinity;
      for (const pose of poses) {
        const g = new Matrix4()
          .makeTranslation(0, pose.py, 0)
          .multiply(new Matrix4().makeRotationFromEuler(new Euler(pose.gx, pose.gy, -0.03, "XYZ")));
        for (const p of tip) {
          const q = p.clone().applyMatrix4(place).applyMatrix4(g).project(camera);
          highest = Math.min(highest, 1 - (q.y + 1) / 2);
        }
      }
      /* Below the edge with room to spare, in the worst pose. Not `> 1.0`: a tip
         that merely touches the edge is a mushroom standing on a line, which is
         the same picture as one floating just above it. */
      expect(highest).toBeGreaterThan(1.05);

      // Where this body's ground actually is, in world space.
      grounds.push(B.at[1] + shoulder[1] * B.scale);
    }

    /* And one patch of ground, not three. The bodies are at three scales and
       three heights, so equal *profile* depths would be three different ground
       planes — the depths have to be solved per body against a common world y,
       which is what this asserts. */
    expect(Math.max(...grounds) - Math.min(...grounds)).toBeLessThan(0.02);
  });
});

describe("the colony stays subordinate to the answers beside it", () => {
  it("emits less than the gutter specimen, which already emitted less than the hero", async () => {
    /*
     * This block's neighbour is six rows of prose being *read*, which is a
     * harder neighbour than a pricing table being scanned — so it is the
     * quietest body on the page. Subordination is low contrast, not flatness:
     * the geometry stays real and the emission comes down, which is the
     * correction this whole line of work started by getting wrong.
     */
    const { VARIANTS } = await import("./vendor/mushroom-stage.js");
    const { tune } = VARIANTS.colony;
    expect(tune.halo).toBe(0);
    expect(tune.glow).toBeLessThan(VARIANTS.gutter.tune.glow);
    expect(tune.shell).toBeLessThan(VARIANTS.gutter.tune.shell);
    expect(tune.bloom).toBeLessThan(VARIANTS.gutter.tune.bloom);
    expect(tune.threshold).toBeGreaterThan(VARIANTS.gutter.tune.threshold);
    // Still real bodies: profiles are geometry, not silhouettes to fill.
    for (const b of VARIANTS.colony.colony) expect(b.capTop.length).toBeGreaterThan(8);
  });

  it("mounts one element for three bodies", async () => {
    /*
     * The technical reason the colony exists as a renderer feature instead of
     * three React components. Three elements would be three WebGL contexts,
     * which with the hero and the gutter makes five on one page — close enough
     * to the browser limit that scrolling far enough would drop the hero's.
     * Asserted rather than left in a comment, because the cheap refactor is
     * exactly the one that reintroduces it.
     */
    const component = await read("./QuestionsColony.tsx");
    expect(component.match(/createElement\("mushroom-stage"\)/g)).toHaveLength(1);
    expect(component).toContain('setAttribute("variant", "colony")');
    // Quiet configuration: no spores drifting across the answers, calm motion.
    expect(component).toContain('setAttribute("spores", "off")');
    expect(component).toContain('setAttribute("motion", "calm")');
    // Decoration, so it is out of the accessibility tree.
    expect(component).toContain('aria-hidden="true"');
  });

  it("builds nothing for visitors the renderer excludes", async () => {
    /* The same gate the hero and the gutter use. A phone must not pay for a
       133KB chunk to draw something its stylesheet then hides. */
    const component = await read("./QuestionsColony.tsx");
    expect(component).toContain("wantsMushroom()");
    expect(component).toMatch(/if \(!host \|\| !wantsMushroom\(\)\) return;/);
  });

  it("waits until the column is nearly in view before building", async () => {
    /* Three bodies is more geometry than one, and this section is most of a
       page down, so none of it is built at boot. */
    const component = await read("./QuestionsColony.tsx");
    expect(component).toContain("IntersectionObserver");
    expect(component).toContain("observer.disconnect()");
    expect(component).toMatch(/rootMargin: "100% 0px"/);
  });

  it("arrives grown rather than popped, and reveals itself if the renderer never reports", async () => {
    const [component, css] = await Promise.all([read("./QuestionsColony.tsx"), read("./questions.css")]);
    expect(component).toContain("mushroom-ready");
    expect(component).toContain("is-grown");
    // A lost context must not leave a permanently invisible block.
    expect(component).toMatch(/setTimeout\(\(\) => setShown\(true\), 4000\)/);
    expect(css).toMatch(/\.rb-quest-colony \{ opacity: 0; transition: opacity [\d.]+s/);
    expect(css).toMatch(/\.rb-quest-colony\.is-grown \{ opacity: 1; \}/);
  });
});

describe("the colony fits the space the column has spare", () => {
  it("is shorter than the left column's empty run at every width it applies to", async () => {
    /*
     * The size guarantee, checked against the stylesheet rather than asserted in
     * a comment — the last time a claim like this was made by eye every term of
     * it was false.
     *
     * The height is now derived: one definite axis (the column's width) and the
     * aspect the framing was solved at. That is the fix for a real bug — the
     * first version gave the block a width, a `height: clamp()` *and* an
     * `aspect-ratio`, and with both axes definite CSS ignores the ratio, so the
     * box shipped anywhere from 1.31 to 3.35 and the group sat in a band of
     * empty paper. The cost of deriving it is that the height is no longer a
     * literal to read off, so it has to be computed the way the browser will.
     */
    const css = await read("./questions.css");
    const block = css.slice(css.indexOf(".rb-quest-colony {"));
    const aspect = Number(block.match(/aspect-ratio:\s*([\d.]+)/)![1]);
    /* A height declaration would silently win over the ratio, so there must be
       none — read from the declarations only, since the comment above them
       explains the bug by quoting the very thing being forbidden. */
    const declarations = block.slice(0, block.indexOf("}")).replace(/\/\*[\s\S]*?\*\//g, "");
    expect(declarations).not.toMatch(/(^|[^-])height:/);
    expect(declarations).toMatch(/position: absolute/);
    expect(declarations).toMatch(/bottom: 0/);
    expect(declarations).toMatch(/left: var\(--rb-quest-colony-left, max\(/);

    expect(aspect).toBeGreaterThan(1);

    // Stems run past the bottom edge on purpose, so the block must clip.
    expect(css).toMatch(/\.rb-quest-colony \{[^}]*overflow: hidden/);
    // It is decoration in a text column: it must not eat clicks.
    expect(css).toMatch(/\.rb-quest-colony \{[^}]*pointer-events: none/);
  });

  it("disappears in the one-column layout instead of splitting question from answer", async () => {
    /* Below 1000px the grid collapses, and the colony would land between a
       question and its answers — the one path a reader's eye has to take. */
    const css = await read("./questions.css");
    const collapsed = css.slice(css.indexOf("@media (max-width: 1000px)"));
    expect(collapsed).toMatch(/\.rb-quest-colony \{ display: none; \}/);
  });

  it("stays planted under the heading column instead of sliding with scroll", async () => {
    /* The colony used to travel left as the sticky heading released. The travel
       is gone: it was motion the reader never asked for, in the margin of a
       block they are reading, and every pixel of it was another chance to end
       up on the column it had to stay off. What is left is a fixed offset. */
    const [css, component] = await Promise.all([read("./questions.css"), read("./QuestionsColony.tsx")]);
    const section = css.match(/\.rb-quest \{[^}]*\}/)?.[0] ?? "";
    const colony = css.match(/\.rb-quest-colony \{[^}]*\}/)?.[0] ?? "";

    expect(section).toContain("overflow: clip");
    // The only transform is the constant vertical planting depth.
    expect(colony).toContain("translate3d(0, 49%, 0)");
    expect(component).toContain('querySelector<HTMLElement>(".rb-quest-head")');
    expect(component).toContain("headingBox.left - originBox.left + headingBox.width / 2");
    expect(component).toContain("startCenter - host.offsetWidth / 2");

    // No scroll-driven horizontal machinery survives, in any spelling.
    expect(component).not.toContain("maxShift");
    expect(component).not.toContain("shiftLeft");
    expect(component).not.toContain("destinationCenter");
    // `progress`/`release` only as identifiers — the words may still appear in
    // the comment explaining why the travel was removed.
    expect(component).not.toMatch(/\b(const|let|var)\s+(progress|release)\b/);
    expect(component).not.toMatch(/host\.style\.transform\s*=/);

    // And nothing recomputes on scroll, so reading the section costs no frames.
    expect(component).not.toContain('addEventListener("scroll"');
  });

  it("measures its offset in the same box the browser resolves `left` against", async () => {
    /* THE 1720px BUG.
     *
     * `left` on an absolutely positioned box is resolved against its
     * offsetParent, which here is the centred shell — not the full-bleed
     * section. The first version measured the heading in section space and
     * wrote the result as `left`, so it rendered too far right by exactly the
     * shell's inset. That inset is zero while the shell still fills the
     * viewport, so at 1440px the page looked right and the bug shipped; at
     * 1720px the inset is 212px and the caps landed on top of the answers.
     *
     * Both endpoints must therefore be expressed in the offsetParent's space,
     * and nothing may subtract `sectionBox.left` on its own. */
    const component = await read("./QuestionsColony.tsx");
    expect(component).toContain("host.offsetParent ?? section).getBoundingClientRect()");

    // Nothing is measured against the full-bleed section any more: with the
    // travel removed, the only geometry left is heading-vs-offsetParent.
    expect(component).not.toContain("sectionBox");
    expect(component).toContain("headingBox.left - originBox.left");

    // The arithmetic the browser confirmed at 1720px: heading at 260 in a shell
    // inset 212 must place the 468px colony centred on 48 in shell space, which
    // is x 260 on screen — the heading column — and clear of the FAQs at 828.
    const shellLeft = 212;
    const headingLeft = 260;
    const headingWidth = 468;
    const colonyWidth = 468;
    const startCenter = headingLeft - shellLeft + headingWidth / 2;
    const screenLeft = shellLeft + (startCenter - colonyWidth / 2);
    expect(screenLeft).toBe(headingLeft);
    expect(screenLeft + colonyWidth).toBeLessThanOrEqual(828);
  });

  it("stops the sticky heading before it descends onto the standing bodies", async () => {
    /* Removing the slide removed the thing that used to save this: the caps
       were leaving as the heading came down, so the crossing was brief. Planted
       bodies do not leave, and the heading's own descent put 147px of paragraph
       on top of them at 1720px. The sticky container therefore reserves the
       colony's visible height, and the reservation is cancelled with a negative
       margin so the reserved band costs the grid no extra height. */
    const css = await read("./questions.css");
    const head = css.match(/\.rb-quest-head \{[^}]*\}/)?.[0] ?? "";
    expect(head).toContain("padding-bottom: calc(var(--rb-quest-colony-visible) + 28px)");
    /* A sticky box stops against its margin box, so cancelling the reservation
       with a negative bottom margin silently restores the full travel. The
       first attempt at this fix did exactly that and measured unchanged. The
       check runs on the declarations only — the comment above this rule
       discusses the negative margin by name. */
    const declarations = head.replace(/\/\*[\s\S]*?\*\//g, "");
    expect(declarations).not.toMatch(/margin-bottom:\s*-/);
    expect(declarations).not.toMatch(/\*\s*-1\)/);
    expect(declarations).toContain("padding-bottom: calc(var(--rb-quest-colony-visible) + 28px)");

    // Both the reservation and the box itself derive from one declared size,
    // so a change to the colony's width cannot desync them.
    expect(css).toContain("--rb-quest-colony-w: min(39vw, 468px)");
    expect(css).toContain("--rb-quest-colony-visible: calc(var(--rb-quest-colony-w) / 1.62 * 0.51)");
    const colony = css.match(/\.rb-quest-colony \{[^}]*\}/)?.[0] ?? "";
    expect(colony).toContain("width: var(--rb-quest-colony-w)");
    expect(colony).toContain("aspect-ratio: 1.62");

    // The reserved band must actually exceed the measured 147px crossing at the
    // width where it was observed: 468px wide / 1.62 * 0.51 + 28 ≈ 175px.
    const reserved = (468 / 1.62) * 0.51 + 28;
    expect(reserved).toBeGreaterThan(147);
  });

  it("pins the FAQs to the second explicit grid column", async () => {
    const css = await read("./questions.css");
    expect(css).toMatch(/\.rb-quest-head \{[^}]*grid-column: 1/);
    expect(css).toMatch(/\.rb-quest-list \{[^}]*grid-column: 2/);
  });

  it("keeps the heading and FAQs in front when the copy descends across the colony", async () => {
    const css = await read("./questions.css");
    expect(css).toMatch(/\.rb-quest-head \{[^}]*z-index: 2/);
    expect(css).toMatch(/\.rb-quest-list \{[^}]*z-index: 2/);
    expect(css).toMatch(/\.rb-quest-colony \{[^}]*z-index: 0/);
  });

  it("has nothing left to gate behind reduced motion but the fade", async () => {
    /* The colony no longer moves for anyone, so there is no travel to shorten
       for a reduced-motion reader and no media query needed in the component:
       the still version *is* the reduced-motion version. The one animation that
       remains is the reveal fade, and the stylesheet already drops it. */
    const [component, css] = await Promise.all([read("./QuestionsColony.tsx"), read("./questions.css")]);
    expect(component).not.toContain("prefers-reduced-motion");
    expect(component).not.toContain("reducedMotion");

    const guard = css.slice(css.indexOf("@media (prefers-reduced-motion: reduce)"));
    expect(guard).toMatch(/\.rb-quest-colony \{\s*transition: none;\s*\}/);
  });

  it("sits beside the sticky heading while owning the section ground", async () => {
    const section = await read("./QuestionsSection.tsx");
    const headStart = section.indexOf('className="rb-quest-head');
    const headEnd = section.indexOf("</div>", headStart);
    const head = section.slice(headStart, headEnd);
    const colony = head.indexOf("<QuestionsColony />");
    expect(colony).toBe(-1);
    expect(section.indexOf("<QuestionsColony />")).toBeGreaterThan(headEnd);
  });

  it("keeps the hero and the gutter as single bodies", async () => {
    /* The colony is additive. A variant with no `colony` key builds exactly one
       body at the identity transform, which is what the reference always did —
       so the two specimens already on the page cannot have been changed by it. */
    const { VARIANTS } = await import("./vendor/mushroom-stage.js");
    expect(VARIANTS.hero.colony).toBeUndefined();
    expect(VARIANTS.gutter.colony).toBeUndefined();
    const renderer = await read("./vendor/mushroom-stage.js");
    expect(renderer).toContain("const members = V.colony || [V];");
    // Placement defaults to the identity for a lone body.
    expect(renderer).toContain("root.position.set(...(B.at || [0, 0, 0]));");
    expect(renderer).toContain("root.scale.setScalar(B.scale || 1);");
    expect(renderer).toContain("root.rotation.y = B.rotY || 0;");
  });
});
