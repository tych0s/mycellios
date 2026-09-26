import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  canonicalUrl,
  INDEXABLE_SEO_PATHS,
  LANDING_SEO_PATHS,
  SEO_PAGES,
  seoPageForPath,
  type SeoPath,
} from "../landing/src/seo.js";

const read = (path: string): string => readFileSync(new URL(`../${path}`, import.meta.url), "utf8");

describe("landing SEO contract", () => {
  it("uses unique titles and canonical URLs for every route", () => {
    const paths = Object.keys(SEO_PAGES) as SeoPath[];
    const titles = paths.map((path) => SEO_PAGES[path].title);
    const canonicals = paths.map((path) => canonicalUrl(SEO_PAGES[path]));

    expect(new Set(titles).size).toBe(titles.length);
    expect(new Set(canonicals).size).toBe(canonicals.length);
    expect(SEO_PAGES["/admin"].robots).toContain("noindex");
    expect(SEO_PAGES["/account"].robots).toContain("noindex");
    expect(SEO_PAGES["/dashboard"].robots).toContain("noindex");
    expect(LANDING_SEO_PATHS).toContain("/account");
    expect(LANDING_SEO_PATHS).toContain("/dashboard");
    for (const path of ["/spore", "/spore/treasury", "/spore/data"] as const) {
      expect(seoPageForPath(path)).toBe(SEO_PAGES[path]);
      expect(SEO_PAGES[path].robots).toContain("noindex");
      expect("structuredData" in SEO_PAGES[path]).toBe(false);
      expect(LANDING_SEO_PATHS).toContain(path);
    }
    for (const path of INDEXABLE_SEO_PATHS) {
      expect(SEO_PAGES[path].robots).toMatch(/^index,/);
      expect(SEO_PAGES[path].structuredData).toBeDefined();
    }
  });

  it("publishes exactly the indexable canonical URLs in the sitemap", () => {
    const sitemap = read("landing/public/sitemap.xml");
    const urls = [...sitemap.matchAll(/<loc>(.*?)<\/loc>/g)].map((match) => match[1]);
    const expected = [
      ...INDEXABLE_SEO_PATHS.map((path) => canonicalUrl(SEO_PAGES[path])),
      `${canonicalUrl(SEO_PAGES["/"])}blog`,
    ];

    expect(urls).toEqual(expected);
    expect(sitemap).not.toContain(canonicalUrl(SEO_PAGES["/admin"]));
    expect(sitemap).not.toContain("https://mycellios.com/");
  });

  it("keeps crawlers away from operational endpoints without hiding admin noindex", () => {
    const robots = read("landing/public/robots.txt");

    expect(robots).toContain("Sitemap: https://www.mycellios.com/sitemap.xml");
    expect(robots).toContain("Disallow: /internal/");
    expect(robots).toContain("Disallow: /mobile/v1/");
    expect(robots).toContain("Disallow: /public/v1/");
    expect(robots).toContain("Disallow: /v1/");
    expect(robots).not.toContain("Disallow: /admin");
  });

  it("ships universal browser-worker metadata before JavaScript runs", () => {
    const mobileHtml = read("src/mobile/index.html");
    const mobile = SEO_PAGES["/browser/"];

    expect(mobileHtml).toContain(`<title>${mobile.title}</title>`);
    expect(mobileHtml).toContain(`content="${mobile.description}"`);
    expect(mobileHtml).toContain(`href="${canonicalUrl(mobile)}"`);
    expect(mobileHtml).toContain('id="seo-structured-data"');
  });
});
