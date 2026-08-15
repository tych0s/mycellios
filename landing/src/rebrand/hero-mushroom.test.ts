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
    // Derived from the rendered element, not from a second paraphrase of the
    // router — otherwise a phone or a studio route pays for a chunk it never
    // draws. Both must be present on the same condition.
    expect(main).toMatch(/page\.type === RebrandLanding && wantsMushroom\(\)/);
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
    expect(component).toMatch(/setTimeout\(\(\) => setShown\(true\)/);
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
    expect(component).toContain('origin-from", "#rb-hero-mushroom"');
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

  it("makes the preload route-conditional in script rather than trusting `media`", async () => {
    const config = await read("../../../vite.landing.config.ts");
    // `media` on `modulepreload` is not reliably honoured across browsers, and
    // one that ignored it would hand every phone the entire renderer. An inline
    // check cannot be ignored.
    // Scoped to the emitted link, not the whole file — the comment above the
    // plugin says the word `media` precisely to explain why it is not used.
    const link = config.slice(config.indexOf('l.rel="modulepreload"'));
    expect(link.slice(0, 200)).not.toMatch(/\bmedia\s*=/);
    expect(config).not.toContain('matchMedia("(max-width: 700px)")');
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

describe("every visitor sees the same organism", () => {
  it("uses the hero WebGL geometry on phones with a bounded quality profile", async () => {
    const component = await read("./HeroMushroom.tsx");
    const css = await read("./rebrand.css");
    const config = await read("../../../vite.landing.config.ts");
    const vendor = await read("./vendor/mushroom-stage.js");
    expect(component).toContain('matchMedia("(prefers-reduced-motion: reduce)")');
    expect(component).toContain('matchMedia("(max-width: 700px)")');
    const phone = css.slice(css.indexOf("@media(max-width:700px)"));
    expect(phone).toMatch(/\.rb-hero-mushroom \{ display:block/);
    expect(component).toContain('setAttribute("variant", "hero")');
    expect(component).toContain('setAttribute("quality", "mobile")');
    expect(component).not.toContain("rb-hero-mushroom-mobile");
    expect(vendor).toContain("mobileQuality ? 1");
    expect(vendor).toContain("samples: lowPower ? 0 : 2");
    expect(phone).toMatch(/\.rb-hero-prompt input,\.rb-hero-prompt button \{ background:rgba\(11,15,12,\.92\)/);
    expect(css).not.toContain("rb-hero-mushroom-placeholder");
    expect(config).not.toContain('matchMedia("(max-width: 700px)")');
  });

  it("keeps the same 3D geometry but disables motion for reduced motion", async () => {
    const component = await read("./HeroMushroom.tsx");
    expect(component).toContain('reduceMotion ? "off" : mobile ? "calm" : "full"');
    expect(component).toContain('mobile || reduceMotion ? "off" : "on"');
  });
});
