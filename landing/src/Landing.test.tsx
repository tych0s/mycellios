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
});
