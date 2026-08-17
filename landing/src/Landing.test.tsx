import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { RebrandLanding } from "./rebrand/RebrandLanding.js";

/*
 * The landing renders to a static string, so these tests read it the way a
 * crawler would: the copy and the anchors have to be in the markup itself, not
 * assembled later by a scroll handler. That matters most for the scroll story,
 * whose animated diagram is aria-hidden — the four beats must still be present
 * as real text for anyone who never triggers the animation.
 */
describe("Landing", () => {
  it("uses the approved editorial, interface, and technical font roles", async () => {
    const css = await import("node:fs/promises").then(({ readFile }) =>
      readFile(new URL("./rebrand/rebrand.css", import.meta.url), "utf8"),
    );

    expect(css).toContain('--rb-font-display:"Fraunces",serif');
    expect(css).toContain('--rb-font-body:"Geist",sans-serif');
    expect(css).toContain('--rb-font-mono:"IBM Plex Mono",monospace');
    expect(css).not.toContain("family=Manrope");
  });

  it("opens the hero on the headline and ends it at the prompt", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    // The headline names the open-compute category and the distributed-AI job
    // it serves. Asserted with the line breaks in place: the three-line setting
    // is what keeps the headline clear of the hero organism, so a two-line
    // regression should fail here rather than silently overlap the artwork.
    expect(html).toContain("The open compute<br/>layer<br/>");
    expect(html).toContain("<em>for distributed AI.</em>");
    expect(html.match(/<h1/g)).toHaveLength(1);
    expect(html).toContain('href="/join"');
    // Nothing frames the headline and nothing follows the prompt: the status
    // pill, the public-testing footnote and "See how it works" are all gone.
    expect(html).not.toContain("rb-status-pill");
    expect(html).not.toContain("Physical multi-node validation underway");
    expect(html).not.toContain("No account or invitation required");
    expect(html).not.toContain("See how it works");
    expect(html).not.toContain("rb-hero-footnote");
  });

  it("closes the full-height hero with a scroll cue pointing at the story", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);
    const hero = html.slice(html.indexOf('class="rb-hero"'), html.indexOf('id="how-it-works"'));

    expect(hero).toContain("rb-scroll-cue");
    expect(hero).toContain('href="#how-it-works"');
    expect(hero).toContain(">Scroll</span>");
  });

  it("plants a varied five-specimen colony along the hero ground", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);
    const hero = html.slice(html.indexOf('class="rb-hero"'), html.indexOf('id="how-it-works"'));

    expect(hero).toContain('class="rb-ground-colony"');
    expect(hero.match(/class="rb-ground-mushroom /g)).toHaveLength(5);
    expect(hero).toContain('aria-hidden="true"');
  });

  it("links the blog from the header navigation, not only the footer", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    // The blog is served by the coordinator at /blog, so it must be a real
    // navigation, not an in-page anchor.
    const header = html.slice(0, html.indexOf("</header>"));
    expect(header).toContain('href="/blog"');
    expect(header).toContain(">Blog</a>");
  });

  it("closes the header with the social marks and Login instead of a worker CTA", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);
    const header = html.slice(0, html.indexOf("</header>"));

    expect(header).toContain('aria-label="mycellios on GitHub"');
    expect(header).toContain('href="https://x.com/mycellios"');
    expect(header).toContain('href="https://t.me/mycellios"');
    expect(header).toContain('href="/network?view=overview">Login</a>');
    expect(header).toContain(">Login</a>");
    // The old "Find your worker" pill is gone from every header surface, mobile
    // burger menu included.
    expect(header).not.toContain("Find your worker");
  });

  it("renders the hero prompt input as the primary entry point", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    expect(html).toContain('id="rb-hero-prompt-input"');
    expect(html).toContain('placeholder="Ask the impossible…"');
    expect(html).toContain("Send this prompt to the network");
    expect(html).toContain("Ask the network");
    // The globe canvas replaces the old mushroom photo and static SVG mesh.
    expect(html).toContain('class="rb-globe"');
    expect(html).not.toContain("rb-hero-image");
    expect(html).not.toContain("rb-active-card");
  });

  it("tells the whole product story once, in four scrubbed beats", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    expect(html).toContain('id="how-it-works"');
    expect(html).toContain("One question");
    expect(html).toContain("The model splits");
    expect(html).toContain("Machines take a part");
    expect(html).toContain("One answer returns");
    // The animated diagram is decorative, so the same four beats also exist as
    // plain text for reduced-motion readers and for crawlers.
    expect(html).toContain("rb-story-fallback");
    // The rail under the diagram names the four steps rather than showing four
    // anonymous marks, and each device card states which slice of the model it
    // holds — both are what make the split concrete rather than decorative.
    expect(html).toContain("Split model");
    expect(html).toContain("Stream result");
    expect(html).toContain("layers 0–6");
    expect(html).toContain("layers 14–20");
  });

  it("leads with subscriptions, keeps token access secondary and gates earnings on proof", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    expect(html).toContain('id="payments"');
    // Buyer and seller each have to be named: a payment section that only says
    // "you get paid" is an earnings pitch, not an exchange.
    expect(html).toContain("Subscribe for ongoing access.");
    expect(html).toContain("Machines earn from verified work.");
    expect(html).toContain("Choose Mycellios Go");
    expect(html).toContain('href="/account"');
    expect(html).not.toContain("$0.0240");
    expect(html).toContain("Sell your idle machine");
    // The split is the same memory split the diagram above shows, so the three
    // layer ranges reappear here as shares of one payment.
    expect(html).toContain("layers 0–6");
    expect(html).toContain("52%");
    expect(html).toContain("13%");
    expect(html).toContain("35%");
    expect(html).toContain("receipt signed");
    expect(html).toContain("signed receipts");
    // Usage credits must not be confused with the future public asset, and the
    // route attribution must not read as a guaranteed payout.
    expect(html).toContain("not subscription pricing or a guaranteed payout");
    expect(html).toContain("Usage tokens are access credits, not $SPORE");
    expect(html).toContain("contributor payouts and $SPORE are not live yet");
  });

  it("renders the doors, evidence and install sections without crashing", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    expect(html).toContain('id="doors"');
    expect(html).toContain("Three doors.");
    expect(html).toContain('id="evidence"');
    expect(html).toContain("We are building");
    expect(html).toContain("Multi-node · GPU · LAN");
    expect(html).toContain('id="install"');
    expect(html).toContain("Download for");
  });

  it("gives the staggered lists the index their entrance animation needs", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    // The evidence rows and the download shelf start at opacity:0 and are
    // revealed by a delay computed from `--i`. If the index stops being
    // rendered, every row still animates — all at once, on top of each other —
    // so this guards the stagger, not the visibility.
    expect(html.match(/rb-board-row[^>]*--i:3/)).not.toBeNull();
    expect(html.match(/rb-get-shelf[\s\S]*?--i:3/)).not.toBeNull();
  });

  it("shows the verified August 2026 open-model shortlist without claiming native support", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    expect(html).toContain('id="models"');
    // The benchmark curve is gone: "which models can I run" is the question a
    // visitor actually has at this point in the page.
    expect(html).not.toContain('id="economics"');
    expect(html).not.toContain("≈13,000×");
    expect(html).not.toContain("ARC-AGI-1");
    // Canonical references verified against official model cards.
    expect(html).toContain("openai/gpt-oss-20b");
    expect(html).toContain("google/gemma-4-12B");
    expect(html).toContain("Qwen/Qwen3.5-27B");
    expect(html).toContain("mistralai/Mistral-Small-4-119B-2603");
    expect(html).toContain("deepseek-ai/DeepSeek-V4-Flash");
    expect(html).toContain("zai-org/GLM-5.2");
    expect(html).toContain("moonshotai/Kimi-K3");
    expect(html).toContain("August 2026 shortlist");
    expect(html).toContain("Shortlist ≠ installed support");
    // Runtime support remains a separate, registry-backed fact.
    expect(html).toContain("Native adapters today");
    expect(html).toContain("Qwen3-MoE");
  });

  it("answers 'do I have to join anything' in a band between the models and the doors", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    // The endpoint is the whole claim, and it must be the real default from
    // src/core/config.ts rather than a pretty placeholder.
    expect(html).toContain("http://127.0.0.1:8787/v1/chat/completions");
    expect(html).toContain("No account");
    expect(html).toContain("No API key");
    // The contribution switch ships off, so the page may not imply otherwise.
    expect(html).toContain("Share your hardware");
    expect(html).toContain("Off until you turn it on.");
    expect(html).toContain('role="switch"');
    expect(html).toContain('aria-checked="false"');
    // And the storage line stays honest: the conversation is written to a local
    // database file, so "we store nothing" is never claimed.
    expect(html).toContain("nothing is uploaded unless you point the app at a hosted coordinator");
    // It sits between the catalogue and the three doors, in that order.
    expect(html.indexOf("rb-local")).toBeGreaterThan(html.indexOf('id="models"'));
    expect(html.indexOf("rb-local")).toBeLessThan(html.indexOf('id="doors"'));
  });

  it("answers the objections before asking for the install, not after", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    expect(html).toContain('id="questions"');
    // The section is only worth having if it takes the uncomfortable questions,
    // so the two that cost conversions are the ones asserted here: what runs on
    // my machine, and who can read my prompt.
    expect(html).toContain("What actually runs on my machine if I join?");
    expect(html).toContain("Who can see what I send?");
    expect(html).toContain("Is there a token? Can I earn today?");
    // And each answer stays at the repo's position rather than at the flattering
    // one: contribution ships off (DEFAULT_DESKTOP_SETTINGS), stages are not yet
    // end-to-end encrypted, and nothing pays out.
    expect(html).toContain("Contribution ships switched off");
    expect(html).toContain("stages are not end-to-end encrypted yet");
    expect(html).toContain("no token, no staking, no payouts");
    // Native <details>, so every answer is in the markup and opens without
    // JavaScript — a crawler and a reader get the same page.
    expect(html).toContain("<details");
    // It lands between the evidence board and the download ask.
    expect(html.indexOf('id="questions"')).toBeGreaterThan(html.indexOf('id="evidence"'));
    expect(html.indexOf('id="questions"')).toBeLessThan(html.indexOf('id="install"'));
  });

  it("preserves the honesty disclaimers around models and downloads", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    // Published scale is not runtime evidence, and shortlist presence is not a
    // support promise — neither may read as a measured, shipped capability.
    expect(html).toContain("not a memory or speed promise");
    expect(html).toContain("Shortlist ≠ installed support");
    expect(html).toContain("only after an adapter and physical evidence ship");
    expect(html).toContain("VISION · NOT LIVE YET · no active token, no financial promise");
    // Unsigned early builds are still disclosed at the download itself.
    expect(html).toContain("Code signing is rolling out");
  });

  it("drops every static illustration in favour of browser-drawn visuals", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    expect(html).not.toContain(".webp");
    expect(html).not.toContain("mycelium-story");
    expect(html).not.toContain("efficiency-curve-ai");
    expect(html).not.toContain("mycellios-live-network");
  });

  it("closes the page on the fruiting body, drawn into the served markup", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);
    const closing = html.slice(html.indexOf('class="rb-closing"'));

    // The mark is real geometry in the HTML, not a canvas the client fills in
    // after hydration: this is the last thing on the page and it should be
    // painted on the first frame.
    expect(closing).toContain("rb-fruit-mark");
    expect(closing).toContain("rb-fruit-cap");
    expect(closing).toContain("rb-fruit-stipe");
    expect(closing).toContain("rb-fruit-under");
    expect(closing).toContain("rb-fruit-cap-paint");
    expect(closing).toContain('d="M');
    // Decorative only — the panel already says "Many machines. One model.",
    // and announcing the same thing twice is noise on a screen reader.
    expect(closing).toContain('aria-hidden="true"');
    expect(closing).toContain("Many machines.");
    // It stands above the wordmark and the final ask, not after them.
    expect(closing.indexOf("rb-fruit-mark")).toBeLessThan(closing.indexOf('href="/join"'));
  });

  it("keeps the support assistant off the landing", () => {
    // The launcher floated over every section at every scroll position; the
    // assistant lives in the panel now, not on the public page.
    const html = renderToStaticMarkup(<RebrandLanding />);

    expect(html).not.toContain("support-assistant");
    expect(html).not.toContain("Need help?");
  });

  it("ships the mobile menu as a dark grouped sheet, not the ivory link stack", async () => {
    const read = (path: string) => import("node:fs/promises").then(({ readFile }) =>
      readFile(new URL(path, import.meta.url), "utf8"),
    );
    const page = await read("./rebrand/RebrandLanding.tsx");
    const css = await read("./rebrand/rebrand.css");

    // Structure: groups with a mono kicker for the $SPORE cluster, the login
    // as a CTA, and a backdrop that dims the page and closes on tap.
    expect(page).toContain("rb-mobile-group");
    expect(page).toContain("rb-mobile-kicker");
    expect(page).toContain("rb-mobile-cta");
    expect(page).toContain("rb-mobile-backdrop");
    // Skin: dark forest panel on the hero's register, bronze accents, and a
    // per-row arrow affordance. The ivory box must not come back.
    expect(css).toContain("#0e1511");
    expect(css).toContain("rbMobileNavIn");
    expect(css).toContain(".rb-header-inner > nav:not(.rb-mobile-nav),.rb-desktop-cta { display:none; }");
    expect(css).toContain(".rb-header-inner > .rb-mobile-nav { position:absolute;");
    expect(css).toMatch(/\.rb-mobile-nav a::after \{ content:"→"/);
    expect(css).not.toMatch(/\.rb-mobile-nav \{[^}]*background:#f8f5f0/);
  });
});
