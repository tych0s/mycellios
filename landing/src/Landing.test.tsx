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
  it("opens the hero on the headline and ends it at the prompt", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    // The headline names the category explicitly and states the single-machine
    // limit the network removes.
    expect(html).toContain("The open compute layer");
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
    expect(html).toContain("layers A–F");
    expect(html).toContain("layers O–Z");
  });

  it("states both sides of the exchange and the proof that gates the payout", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    expect(html).toContain('id="payments"');
    // Buyer and seller each have to be named: a payment section that only says
    // "you get paid" is an earnings pitch, not an exchange.
    expect(html).toContain("Buyers pay for answers.");
    expect(html).toContain("Machines get paid for work.");
    expect(html).toContain("Buy compute");
    expect(html).toContain("Sell your idle machine");
    // The split is the same memory split the diagram above shows, so the three
    // layer ranges reappear here as shares of one payment.
    expect(html).toContain("layers A–F");
    expect(html).toContain("52%");
    expect(html).toContain("13%");
    expect(html).toContain("35%");
    expect(html).toContain("receipt signed");
    expect(html).toContain("settled in USDC");
    // Nothing here may read as a promise: the split is illustrative and no
    // payout is live.
    expect(html).toContain("not a live rate");
    expect(html).toContain("no token is live and no payout is promised yet");
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

  it("shows open models of every size and how many machines each one needs", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    expect(html).toContain('id="models"');
    // The benchmark curve is gone: "which models can I run" is the question a
    // visitor actually has at this point in the page.
    expect(html).not.toContain('id="economics"');
    expect(html).not.toContain("≈13,000×");
    expect(html).not.toContain("ARC-AGI-1");
    // Real Hugging Face references, not invented names — the catalogue is a live
    // query against the hub, so the rows have to point at checkpoints that exist.
    expect(html).toContain("Qwen/Qwen3-0.6B");
    expect(html).toContain("zai-org/GLM-4.5-Air");
    expect(html).toContain("Qwen/Qwen3-235B-A22B");
    // The run mode is the claim: a model too big for one box still runs.
    expect(html).toContain("One machine");
    expect(html).toContain("8+ machines");
    // Support is architecture-based, so the adapter families are named rather
    // than a fixed model list being implied.
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

    // Sizes are arithmetic on parameter counts, and the split is still in
    // physical testing — neither may read as a measured, shipped capability.
    expect(html).toContain("BF16 weight estimates, not measured runtimes");
    expect(html).toContain("the multi-machine split is in physical testing");
    expect(html).toContain("Model support is architecture-based");
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

  it("renders the network-only support assistant entry point", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    expect(html).toContain('aria-label="Open mycellios assistant"');
    expect(html).toContain("Need help?");
    expect(html).toContain("Assistant offline");
  });
});
