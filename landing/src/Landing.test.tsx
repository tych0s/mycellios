import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Landing } from "./Landing.js";

describe("Landing navigation", () => {
  it("keeps the blog visible in both the primary navigation and footer", () => {
    const html = renderToStaticMarkup(<Landing />);

    expect(html.match(/href="\/blog"/g)).toHaveLength(2);
    expect(html).toContain(
      'class="nav-blog-link" href="/blog" aria-label="Read the Mycellios blog"',
    );
    expect(html).toContain('<a href="/blog">Blog</a>');
  });

  it("places the short visual explanation before the first idea section", () => {
    const html = renderToStaticMarkup(<Landing />);
    const explainer = html.indexOf('id="explainer"');
    const idea = html.indexOf('id="vision"');

    expect(explainer).toBeGreaterThan(-1);
    expect(idea).toBeGreaterThan(-1);
    expect(explainer).toBeLessThan(idea);
    expect(html).toContain("Several devices. One AI model.");
    expect(html).toContain('href="#explainer"');
  });

  it("renders the network-only support assistant entry point", () => {
    const html = renderToStaticMarkup(<Landing />);

    expect(html).toContain('aria-label="Abrir asistente de mycellios"');
    expect(html).toContain("¿Necesitas ayuda?");
    expect(html).toContain("Asistente sin conexión");
  });
});
