import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { RebrandLanding } from "./rebrand/RebrandLanding.js";

describe("Merged landing (rebrand design + legacy content)", () => {
  it("renders the hero with the status pill and public-testing footnote", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    expect(html).toContain("Intelligence");
    expect(html.match(/<h1/g)).toHaveLength(1);
    expect(html).toContain("Physical multi-node validation underway");
    expect(html).toContain("No account or invitation required during public testing.");
    expect(html).toContain('href="/join"');
    expect(html).toContain('href="#how-it-works"');
  });

  it("renders the explainer, evidence, and install sections without crashing", () => {
    const html = renderToStaticMarkup(<RebrandLanding />);

    expect(html).toContain('id="explainer"');
    expect(html).toContain("Several devices.");
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
