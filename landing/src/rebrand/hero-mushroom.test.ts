import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

/*
 * The hero renderer is the largest asset on the page, and the two things that
 * make it feel late are structural rather than visual: whether its download can
 * start before the main bundle finishes, and whether it pops in once it does.
 * Both are easy to undo by accident — moving the prefetch, or dropping the
 * `.is-grown` class — and neither would fail a rendering test, so they are
 * pinned here.
 */

const read = (path: string) => readFile(new URL(path, import.meta.url), "utf8");

describe("the hero mushroom is not left downloading behind the bundle", () => {
  it("starts its chunk from the page bootstrap, not only from the effect", async () => {
    const main = await read("../main.tsx");
    // A dynamic import inside an effect cannot begin until the entry bundle has
    // been fetched, parsed and rendered; that serial waterfall is exactly the
    // one-to-two second gap this guards against.
    expect(main).toContain("loadMushroomStage()");
    expect(main).toContain("wantsMushroom()");
  });

  it("gates the prefetch on the page actually rendering the hero", async () => {
    const main = await read("../main.tsx");
    // The same mutually exclusive route flags choose the lazy page and derive
    // `isLanding`; a panel or studio route must never pay for this chunk.
    expect(main).toContain("const isLanding = !isCreate && !isEarn && !isSpore && !isNetwork && !isPanel");
    expect(main).toMatch(/if \(isLanding && wantsMushroom\(\)\)/);
  });

  it("keeps the renderer in its own chunk instead of inlining it into the entry", async () => {
    const component = await read("./HeroMushroom.tsx");
    // The prefetch must stay a dynamic import. A static `import ... from` here
    // would fold ~190KB gzipped of three.js into the entry bundle and delay the
    // headline itself — trading a late mushroom for a late page.
    expect(component).toMatch(/import\("\.\/vendor\/mushroom-stage\.js"\)/);
    expect(component).not.toMatch(/^import .* from "\.\/vendor\/mushroom-stage\.js"/m);
  });

  it("shares one promise so the bootstrap and the component do not fetch twice", async () => {
    const component = await read("./HeroMushroom.tsx");
    expect(component).toContain("stagePromise ??=");
  });
});

describe("the arrival is faded rather than popped", () => {
  it("waits for the first drawn frame before showing the host", async () => {
    const component = await read("./HeroMushroom.tsx");
    const vendor = await read("./vendor/mushroom-stage.js");
    // `_ready` precedes any pixels, so the event has to come from the frame
    // loop; a fade started on `_ready` would still reveal an empty canvas.
    expect(vendor).toContain("mushroom-ready");
    expect(component).toContain('addEventListener("mushroom-ready"');
    expect(component).toContain("is-grown");
  });

  it("still reveals the host if the renderer never reports ready", async () => {
    const component = await read("./HeroMushroom.tsx");
    // No WebGL or a lost context would otherwise strand the hero at opacity 0,
    // taking the shade plate with it.
    expect(component).toContain("setTimeout(show, 4000)");
  });

  it("shows only the real renderer and transitions it in quickly", async () => {
    const css = await read("./rebrand.css");
    expect(css).toMatch(/\.rb-hero-mushroom mushroom-stage \{[^}]*opacity:0/);
    expect(css).toMatch(/\.rb-hero-mushroom\.is-grown mushroom-stage \{[^}]*opacity:1/);
    expect(css).not.toContain("rb-hero-mushroom-placeholder");
    expect(css).toMatch(/\.rb-hero-mushroom mushroom-stage \{[^}]*transition:opacity \.22s/);
  });

  it("keeps the horizontal centring in both states", async () => {
    const css = await read("./rebrand.css");
    // The element is centred with translateX(-50%). A grown state that dropped
    // it would slide the organism half its own width across the hero.
    const base = css.match(/\n\.rb-hero-mushroom \{[^}]*\}/)?.[0] ?? "";
    const grown = css.match(/\.rb-hero-mushroom\.is-grown mushroom-stage \{[^}]*\}/)?.[0] ?? "";
    expect(base).toContain("translateX(-50%)");
    expect(base).toContain("left:50%");
    expect(grown).toContain("transform:none");
  });

  it("keeps the planted end of the stem solid at the viewport edge", async () => {
    const css = await read("./rebrand.css");
    const base = css.match(/\n\.rb-hero-mushroom \{[^}]*\}/)?.[0] ?? "";

    // The hero already clips the organism at the viewport boundary. A mask on
    // the host erased the last 12% of the stem before it reached that boundary,
    // making its base look airbrushed away instead of planted below the fold.
    expect(base).not.toContain("mask-image");
    expect(base).toContain("bottom:-8px");
  });

  it("starts the mycelium below the mushroom instead of on its visible stem", async () => {
    const renderer = await read("./vendor/mycelium-network.js");
    const component = await read("./MyceliumNetwork.tsx");
    const mushroom = await read("./HeroMushroom.tsx");

    expect(renderer).toContain("b.bottom - r.top");
    expect(renderer).not.toMatch(/b\.height \* 0\.9\d/);
    expect(mushroom).toContain('id="rb-hero-mushroom"');
    expect(component).toContain('"#rb-hero-mushroom"');
  });

  it("anchors the colony to the hero floor on phones, where the organism is mid-hero", async () => {
    const component = await read("./MyceliumNetwork.tsx");
    const page = await read("./RebrandLanding.tsx");

    // Behind the globe, the organism's base is the middle of the hero —
    // anchoring there would start the colony in the middle of the copy. The
    // phone anchor is the hero's floor, so the network begins right after
    // the hero exactly like the desktop.
    expect(component).toContain('mobile ? "#rb-hero-base" : "#rb-hero-mushroom"');
    expect(page).toContain('id="rb-hero-base"');
  });

  it("keeps the grown colony out of the page reading column", async () => {
    const renderer = await read("./vendor/mycelium-network.js");

    // Stacking the canvas behind transparent wrappers is not an exclusion:
    // filaments still show through copy. The renderer itself must reject
    // centre segments, nodes and travelling pulses after leaving the hero.
    expect(renderer).toContain("clearsReadingZone");
    expect(renderer).toContain("x0 <= railInset && x1 <= railInset");
    // Point marks and travelling glints are still excluded outright: unlike a
    // filament, omitting one leaves nothing dangling.
    expect(renderer).toContain("cx <= railInset || cx >= W - railInset");
    expect(renderer).toContain("x > railInset && x < this.vw - railInset");
  });

  it("keeps the colony sparse and ambient after its visible root flare", async () => {
    const renderer = await read("./vendor/mycelium-network.js");
    const component = await read("./MyceliumNetwork.tsx");

    expect(component).toContain('density", "0.58"');
    expect(renderer).toContain("const ink = dim ? 1.08 : 0.88");
    expect(renderer).toContain("Math.min(140, Math.max(96");
    expect(renderer).toContain("slice(0, 4)");
    expect(renderer).not.toContain("SURGE_SECONDS");
    expect(renderer).not.toContain("armSurge");
  });

  it("lets organic runners cross the story section to stay connected", async () => {
    const renderer = await read("./vendor/mycelium-network.js");

    expect(renderer).toContain("const railInset = W * 0.32");
    expect(renderer).not.toContain("const leftSpine = []");
    expect(renderer).toContain("clearsReadingZone");
  });

  it("grows extra trunks on narrow pages so the colony starts right after the hero", async () => {
    const renderer = await read("./vendor/mycelium-network.js");

    // Measured against the built geometry: five trunks on a phone leave the
    // first viewport and a half below the origin nearly empty — branches
    // accumulate per pixel travelled, so the network only read as present
    // from mid-page. The extras are interleaved, not appended: growth is
    // depth-first and capped, so appended trunks starve before drawing a
    // segment. The larger budget lets the colony still reach the footer.
    expect(renderer).toContain("trunkDefs.splice(");
    expect(renderer).toContain("W < 700 ? 13500 : 9000");
  });

  it("never severs a filament at the reading column", async () => {
    const renderer = await read("./vendor/mycelium-network.js");

    // Two separate ways to look severed, both fixed here. Discarding a segment
    // mid-path leaves the rest floating with nothing attached; and stepping the
    // ink abruptly between two adjacent segments reads as a cut even when the
    // path is unbroken. So nothing is dropped, and the veil is a 0..1 ramp.
    expect(renderer).toContain("c: centreVeil(x0, cx, cy)");
    expect(renderer).not.toContain("if (clearsReadingZone(x0, cx, cy)) {");
    expect(renderer).toContain("const veil = 1 - s.c * (1 - floor)");
    expect(renderer).toContain("if (s.d === 0 && s.c < 1)");
    // A boolean crossing flag is exactly the regression this guards against.
    expect(renderer).not.toContain("s.c ? (s.d === 0 ? 0.34 : 0.16) : 1");
  });

  it("keeps centre crossings at ghost ink, not as lines across the page", async () => {
    const renderer = await read("./vendor/mycelium-network.js");

    // The complaint that pinned this: with the old floors a trunk kept a third
    // of its ink at full depth, and the steered verticals read as orange lines
    // spanning every band. The colony belongs to the margins — what crosses the
    // copy survives only as a ghost that keeps the path unbroken.
    expect(renderer).toContain("s.d === 0 ? 0.08 : 0.03");
    // sqrt pushes the veil outward from the column's centre line, so a segment
    // only somewhat inside the column already loses most of its ink.
    expect(renderer).toContain("Math.sqrt(lateral)");
  });

  it("uses a lighter first-frame renderer budget", async () => {
    const vendor = await read("./vendor/mushroom-stage.js");

    expect(vendor).toContain("lowPower ? 1.15 : 1.4");
    expect(vendor).toContain("lowPower ? 56 : 80");
    expect(vendor).toContain("lowPower ? 44 : 64");
    expect(vendor).toContain("samples: lowPower ? 0 : 2");
    expect(vendor).toContain("if (!this._firstFramePainted)");
    expect(vendor.indexOf("if (!this._firstFramePainted)")).toBeLessThan(vendor.indexOf("r.setRenderTarget(this.rtScene)"));
  });
});

describe("the browser learns about the chunk while it is still parsing the HTML", () => {
  it("injects the preload at build time rather than waiting for the entry bundle", async () => {
    const config = await read("../../../vite.landing.config.ts");
    // Even a prefetch on the first line of main.tsx cannot run until ~200KB of
    // entry bundle has been downloaded and parsed. This makes the chunk visible
    // to the preload scanner instead, so both large fetches start together.
    expect(config).toContain("mycellios:preload-hero-renderer");
    expect(config).toContain('rel="modulepreload"');
    expect(config).toContain('fetchPriority="high"');
    expect(config).toContain("mushroom-stage-");
  });

  it("makes the preload route- and width-conditional in script rather than trusting `media`", async () => {
    const config = await read("../../../vite.landing.config.ts");
    // `media` on `modulepreload` is not reliably honoured across browsers, and
    // one that ignored it would hand every phone the entire renderer. An inline
    // check cannot be ignored.
    // Scoped to the emitted link, not the whole file — the comment above the
    // plugin says the word `media` precisely to explain why it is not used.
    const link = config.slice(config.indexOf('l.rel="modulepreload"'));
    expect(link.slice(0, 200)).not.toMatch(/\bmedia\s*=/);
    const mobileGate = config.indexOf('window.matchMedia("(max-width: 700px)").matches)return;');
    expect(mobileGate).toBeGreaterThan(0);
    expect(mobileGate).toBeLessThan(config.indexOf('l.rel="modulepreload"'));
    expect(config).not.toContain('matchMedia("(prefers-reduced-motion: reduce)")');
  });

  it("keeps three.js tree-shakeable", async () => {
    const vendor = await read("./vendor/mushroom-stage.js");
    // A namespace import retains the whole library: the chunk was 190KB gzipped
    // that way and is 137KB with named bindings. The namespace object is
    // rebuilt locally so the ~500 `THREE.Foo` call sites stay as the upstream
    // reference wrote them.
    expect(vendor).not.toMatch(/import \* as \w+ from ["']three["']/);
    expect(vendor).toMatch(/import \{[^}]+\} from ["']three["']/);
  });
});

describe("the mobile hero stays free of the 3D renderer", () => {
  it("keeps the organism and its renderer off phones entirely", async () => {
    const component = await read("./HeroMushroom.tsx");
    const css = await read("./rebrand.css");
    const config = await read("../../../vite.landing.config.ts");
    expect(component).toContain('matchMedia("(prefers-reduced-motion: reduce)")');
    /*
     * The phone experiment is settled: behind the globe the specimen read as
     * a transparent ghost, and every earlier framing fought the copy. So the
     * mobile hero is planet + text, and the bootstrap and resize listener
     * keep the large chunk and WebGL context off the phone.
     */
    expect(component).toMatch(/wantsMushroom\(\)[\s\S]*matchMedia\("\(max-width: 700px\)"\)/);
    expect(component).toContain("if (wantsMushroom()) start();");
    expect(component).toContain("else stop(true);");
    expect(component).toContain('mobile.addEventListener("change", syncWidth)');
    const phone = css.slice(css.indexOf("@media(max-width:700px)"));
    expect(phone).toMatch(/\.rb-hero-mushroom \{ display:none/);
    // Nothing passes behind the input on a phone, so the field keeps the
    // theme's translucent fill — the opaque plate it once needed for
    // legibility over the cap must stay gone.
    expect(phone).not.toContain("background:rgba(11,15,12,.92)");
    expect(css).not.toContain("rb-hero-mushroom-placeholder");
    expect(config).toContain('window.matchMedia("(max-width: 700px)").matches)return;');
  });

  it("keeps the same 3D geometry but disables motion for reduced motion", async () => {
    const component = await read("./HeroMushroom.tsx");
    expect(component).toContain('reduceMotion ? "off" : "full"');
    expect(component).toContain('reduceMotion ? "off" : "on"');
  });
});

/*
 * The phone still has to contain the species.
 *
 * Keeping the standing organism off a phone is settled and correct, but the
 * first screen was then a planet and some text on a page named after a fungus.
 * The ground colony is the answer that costs nothing: five SVG paths already in
 * the markup, drawn small along the hero's floor, under everything.
 *
 * What can go wrong is silent and geometric — a body that grows back to desktop
 * size, or one that drifts under the SCROLL cue or the support assistant's
 * fixed launcher — so the sizes and the two keep-clear bands are measured here
 * rather than eyeballed once.
 */
describe("the phone hero still grows something", () => {
  /** The `@media(max-width:700px)` block that carries the hero overrides. */
  const phoneBlock = async () => {
    const css = await read("./rebrand.css");
    return css.slice(css.indexOf(".rb-hero-mushroom { display:none"));
  };

  /** Every phone specimen as `{ id, left%, size px }`, in declaration order. */
  const phoneSpecimens = async () => {
    const block = await phoneBlock();
    return [...block.matchAll(/\.rb-ground-mushroom-([a-e]) \{ left:([\d.]+)%;[^}]*--rb-mini-size:([\d.]+)px/g)]
      .map((match) => ({ id: String(match[1]), left: Number(match[2]), size: Number(match[3]) }));
  };

  it("no longer hides the colony on phones", async () => {
    const block = await phoneBlock();
    // The one line this whole arrangement replaces. It hid markup that was
    // being rendered anyway, so the phone paid for the colony and drew none.
    expect(block).not.toMatch(/\.rb-ground-colony \{ display:none/);
  });

  it("keeps all five specimens rather than planting a token one", async () => {
    const specimens = await phoneSpecimens();
    // A colony is a colony because there is more than one age of body in it;
    // one mushroom in a corner is an icon.
    expect(specimens).toHaveLength(5);
  });

  it("plants undergrowth rather than the desktop bodies at phone width", async () => {
    const css = await read("./rebrand.css");
    const specimens = await phoneSpecimens();
    const desktop = [...css.matchAll(/^\.rb-ground-mushroom-[a-e] \{[^}]*--rb-mini-size:([\d.]+)px/gm)].map((m) => Number(m[1]));
    const largestDesktop = Math.max(...desktop);

    /*
     * The four bodies around the subject are undergrowth and stay that way:
     * well under the desktop colony's largest, and never so small they read as
     * smudges. The centre specimen is deliberately exempt — it is the phone's
     * standing organism, so it is allowed to exceed the desktop *colony*
     * (whose largest is undergrowth too; the desktop's real subject is the
     * WebGL stage, which phones do not load). What still binds it is the
     * cap-versus-prompt clearance measured in the keep-clear test below.
     */
    expect(desktop).toHaveLength(5);
    const centre = specimens.find((s) => s.left === 50)!;
    for (const specimen of specimens) {
      if (specimen !== centre) expect(specimen.size).toBeLessThan(largestDesktop * 0.7);
      expect(specimen.size).toBeGreaterThanOrEqual(18);
    }
    // Still a range of ages, not five copies of one size.
    expect(new Set(specimens.map((s) => s.size)).size).toBe(5);
  });

  it("gives the phone colony a subject in the middle instead of an even border", async () => {
    const specimens = await phoneSpecimens();
    const centre = specimens.find((s) => s.left === 50);

    /*
     * The complaint this pins: five similar bodies spread along an edge read
     * as a border print rather than as a colony. The desktop hero solves it
     * with the standing organism holding the middle, and this is the phone's
     * version of the same composition — so there has to *be* a body at the
     * centre, and it has to be decisively the largest rather than merely
     * joint-largest with a neighbour.
     */
    expect(centre).toBeDefined();
    const others = specimens.filter((s) => s !== centre).map((s) => s.size);
    expect(centre!.size).toBeGreaterThan(Math.max(...others) * 1.5);
  });

  it("puts a long-stemmed specimen in the middle so the cue has stem to cross", async () => {
    const specimens = await phoneSpecimens();
    const centre = specimens.find((s) => s.left === 50)!;
    const colony = await read("./HeroGroundColony.tsx");

    /*
     * Which specimen stands in the centre is load-bearing, not cosmetic. The
     * cue crosses the stem below the cap, so the centre body needs a cap that
     * sits high in its own box and a stem long enough to carry the label.
     *
     * `HeroGroundColony.tsx` declares the specimens in `a..e` order, so index
     * 3 is `-d`, the parasol: its cap path starts at y18 of a 140 viewBox.
     * The flat specimen `-c` hangs its cap at y38 and would put the gills
     * exactly where the cue goes. Reading the cap's y out of the component
     * keeps this honest if the geometry is ever redrawn.
     */
    const kinds = [...colony.matchAll(/kind: "(\w+)"/g)].map((m) => String(m[1]));
    const centreKind = kinds[centre.id.charCodeAt(0) - 97] ?? "";
    expect(centreKind).toBe("parasol");

    const capTops = new Map<string, number>(
      [...colony.matchAll(/\{kind === "(\w+)" &&[\s\S]*?rb-mini-cap" d="M\d+ ([\d.]+)/g)]
        .map((m) => [String(m[1]), Number(m[2])] as [string, number]),
    );
    const centreCap = capTops.get(centreKind);
    expect(centreCap).toBeDefined();
    // The centre cap must ride higher in its box than every alternative.
    for (const [kind, y] of capTops) {
      if (kind !== centreKind) expect(centreCap!).toBeLessThanOrEqual(y);
    }
  });

  it("keeps the cue and the launcher's corner clear of the bodies", async () => {
    const css = await read("./rebrand.css");
    const block = await phoneBlock();
    const specimens = await phoneSpecimens();
    const VIEWPORT = 390;

    /*
     * The cue shares the centre specimen's column by design, and now sits on
     * its stem below the cap — the arrangement the desktop hero already uses.
     * So the clearance runs the other way from before: the cue's top must stay
     * under the cap's lower edge, or the label lands on the gills.
     *
     * `.rb-ground-mushroom` is `bottom:-5px` with a box 1.4x its width. SVG y
     * grows downward, so the parasol's cap bottom at y54 of a 140 viewBox
     * hangs `size * 1.4 * (1 - 54/140) - 5` above the floor — measured from
     * the floor up, which is the direction `bottom` counts in. The cue is its
     * own `bottom` plus ~37px of label, gap and arrow.
     */
    const cueBottom = Number(block.match(/\.rb-scroll-cue \{ bottom:(\d+)px/)![1]);
    const centre = specimens.find((s) => s.left === 50)!;
    const bodyBox = Number(css.match(/\.rb-ground-mushroom \{[^}]*height:calc\(var\(--rb-mini-size\) \* ([\d.]+)\)/)![1]);
    const bodyBottom = Number(css.match(/\.rb-ground-mushroom \{[^}]*bottom:(-?\d+)px/)![1]);
    const CUE_HEIGHT = 37;
    const CAP_BOTTOM_FRACTION = 54 / 140;
    const gillsAbove = (size: number) => size * bodyBox * (1 - CAP_BOTTOM_FRACTION) + bodyBottom;
    expect(cueBottom + CUE_HEIGHT).toBeLessThan(gillsAbove(centre.size));
    // And the cue still clears the floor rather than sinking into it.
    expect(cueBottom).toBeGreaterThan(0);

    /*
     * The short-phone tier re-cuts the centre body alone: a 740pt screen runs
     * ~100px poorer than an 844pt one and spends the difference in this exact
     * band, so the cap has to stop shorter to stay clear of the prompt input.
     * The cue does not move — it is measured from the floor, not the cap.
     */
    const shortBlock = css.slice(css.indexOf("@media(max-width:700px) and (max-height:800px)"));
    const shortCentre = Number(shortBlock.match(/\.rb-ground-mushroom-d \{ --rb-mini-size:(\d+)px/)![1]);
    expect(cueBottom + CUE_HEIGHT).toBeLessThan(gillsAbove(shortCentre));
    // A smaller cut of the same composition, never a larger one.
    expect(shortCentre).toBeLessThan(centre.size);
    // Still decisively the subject at this size too.
    expect(shortCentre).toBeGreaterThan(Math.max(...specimens.filter((s) => s !== centre).map((s) => s.size)) * 1.5);

    /* support-assistant.css: `right:12px` and a 56px launcher below 620px. */
    const LAUNCHER_FROM = VIEWPORT - 12 - 56;
    for (const specimen of specimens) {
      // The specimens are centred on their `left` too, so the body spans half
      // its own width either side of it.
      const at = (specimen.left / 100) * VIEWPORT;
      expect(at + specimen.size / 2).toBeLessThanOrEqual(LAUNCHER_FROM);
      // And nothing runs off the left edge either.
      expect(at - specimen.size / 2).toBeGreaterThanOrEqual(0);
    }
  });

  it("still keeps the 190KB renderer off the phone", async () => {
    const component = await read("./HeroMushroom.tsx");
    const colony = await read("./HeroGroundColony.tsx");
    // The colony is inline SVG and must stay that way: the reason a phone can
    // afford mushrooms at all is that these are paths, not a WebGL context.
    expect(colony).not.toContain("mushroom-stage");
    expect(colony).not.toContain("loadMushroomStage");
    expect(component).toContain("if (wantsMushroom()) start();");
  });

  it("holds the specimens still under reduced motion", async () => {
    const css = await read("./rebrand.css");
    const reduced = css.slice(css.indexOf("@media(prefers-reduced-motion:reduce)"));
    // The sway is an infinite alternating loop, so the blanket .01ms rule
    // would leave it flicking rather than stopping it.
    expect(reduced).toContain(".rb-ground-mushroom svg { animation:none; }");
  });
});
