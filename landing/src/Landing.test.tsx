import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { RebrandLanding } from "./rebrand/RebrandLanding.js";

describe("Merged landing (rebrand design + legacy content)", () => {
  it("opens the hero on the headline and ends it at the prompt", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    expect(html).toContain("Intelligence");
    expect(html.match(/<h1/g)).toHaveLength(1);
    expect(html).toContain('href="/join"');
    expect(html).toContain('href="#how-it-works"');
    // Nothing frames the headline and nothing follows the prompt: the status
    // pill, the public-testing footnote and "See how it works" are all gone.
    expect(html).not.toContain("rb-status-pill");
    expect(html).not.toContain("Physical multi-node validation underway");
    expect(html).not.toContain("No account or invitation required");
    expect(html).not.toContain("See how it works");
    expect(html).not.toContain("rb-hero-footnote");
  });

  it("closes the full-height hero with a scroll cue pointing at the explainer", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);
    const hero = html.slice(html.indexOf('class="rb-hero"'), html.indexOf('id="explainer"'));

    expect(hero).toContain("rb-scroll-cue");
    expect(hero).toContain('href="#explainer"');
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
    expect(header).toContain(">Login</a>");
    // The old "Find your worker" pill is gone from every header surface, mobile
    // burger menu included — the hardware section is still reachable by scroll.
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

  it("renders the explainer, evidence, and install sections without crashing", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    expect(html).toContain('id="explainer"');
    expect(html).toContain("Several devices.");
    expect(html).toContain("Choose a stage in the system explanation");
    expect(html).toContain("Playing this stage");
    expect(html).toContain("Play this stage");
    expect(html).toContain("MYCELLIOS CELL");
    expect(html).toContain("RTX 4090");
    expect(html).toContain("Answer assembled");
    expect(html).toContain("Pause automatic demonstration");
    expect(html).toContain("Your app sends one familiar request");
    expect(html).toContain('id="evidence"');
    expect(html).toContain("We are building");
    expect(html).toContain("Multi-node · GPU · LAN");
    expect(html).toContain('id="install"');
    expect(html).toContain("NETWORK READY");
  });

  it("preserves the efficiency claims and honesty disclaimers", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    expect(html).toContain('id="acceleration"');
    expect(html).toContain("≈13,000×");
    expect(html).toContain("$0.231 / task");
    expect(html).toContain("87.5% ARC-AGI-1");
    expect(html).toContain("DIRECTION · NOT A PERFORMANCE FORECAST");
    expect(html).toContain('id="roadmap"');
    expect(html).toContain("VISION · NOT LIVE YET");
    expect(html).toContain("No active token · no financial promise");
  });

  it("renders the network-only support assistant entry point", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    expect(html).toContain('aria-label="Abrir asistente de mycellios"');
    expect(html).toContain("¿Necesitas ayuda?");
    expect(html).toContain("Asistente sin conexión");
  });
});
