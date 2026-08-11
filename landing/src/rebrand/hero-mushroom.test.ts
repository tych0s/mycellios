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

  it("starts the host hidden and transitions it in", async () => {
    const css = await read("./rebrand.css");
    const base = css.match(/\n\.rb-hero-mushroom \{[^}]*\}/)?.[0] ?? "";
    expect(base).toMatch(/opacity:0/);
    expect(base).toMatch(/transition:[^;]*opacity/);
    expect(css).toMatch(/\.rb-hero-mushroom\.is-grown \{[^}]*opacity:1/);
  });

  it("keeps the horizontal centring in both states", async () => {
    const css = await read("./rebrand.css");
    // The element is centred with translateX(-50%). A grown state that dropped
    // it would slide the organism half its own width across the hero.
    const base = css.match(/\n\.rb-hero-mushroom \{[^}]*\}/)?.[0] ?? "";
    const grown = css.match(/\.rb-hero-mushroom\.is-grown \{[^}]*\}/)?.[0] ?? "";
    expect(base).toContain("translateX(-50%)");
    expect(grown).toContain("translateX(-50%)");
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
    expect(config).toContain("mushroom-stage-");
  });

  it("makes the preload conditional in script rather than trusting `media`", async () => {
    const config = await read("../../../vite.landing.config.ts");
    // `media` on `modulepreload` is not reliably honoured across browsers, and
    // one that ignored it would hand every phone the entire renderer. An inline
    // check cannot be ignored.
    // Scoped to the emitted link, not the whole file — the comment above the
    // plugin says the word `media` precisely to explain why it is not used.
    const link = config.slice(config.indexOf('l.rel="modulepreload"'));
    expect(link.slice(0, 200)).not.toMatch(/\bmedia\s*=/);
    expect(config).toContain('matchMedia("(max-width: 700px)")');
    expect(config).toContain('matchMedia("(prefers-reduced-motion: reduce)")');
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

describe("visitors who will not see it do not pay for it", () => {
  it("skips reduced-motion and phone widths at the same 700px the CSS hides it at", async () => {
    const component = await read("./HeroMushroom.tsx");
    const css = await read("./rebrand.css");
    const config = await read("../../../vite.landing.config.ts");
    expect(component).toContain('matchMedia("(prefers-reduced-motion: reduce)")');
    expect(component).toContain('matchMedia("(max-width: 700px)")');
    // The same breakpoint must hide `.rb-hero-mushroom`, or there is a band of
    // widths painting a shade plate with no organism behind it — and it must
    // gate the preload too, or the phone downloads what the CSS then hides.
    const phone = css.slice(css.indexOf("@media(max-width:700px)"));
    expect(phone).toMatch(/\.rb-hero-mushroom \{ display:none/);
    expect(config).toContain('matchMedia("(max-width: 700px)")');
  });
});
