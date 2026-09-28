import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
  canonicalUrl,
  INDEXABLE_SEO_PATHS,
  LANDING_SEO_PATHS,
  SEO_PAGES,
  seoPageForPath,
  type SeoPath,
} from "../landing/src/seo.js";

const outputRoot = join(process.cwd(), "landing-dist");
const templatePath = join(outputRoot, "index.html");
const routeOutputPaths: Record<(typeof LANDING_SEO_PATHS)[number], string> = {
  "/": "index.html",
  "/network": "network/index.html",
  "/create": "create/index.html",
  "/earn": "earn/index.html",
  "/spore": "spore/index.html",
  "/spore/treasury": "spore/treasury/index.html",
  "/spore/data": "spore/data/index.html",
  "/downloads": "downloads/index.html",
  "/admin": "admin/index.html",
  "/dashboard": "dashboard/index.html",
  "/account": "account/index.html",
};

function escapeAttribute(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function replaceRequired(source: string, pattern: RegExp, replacement: string, label: string): string {
  if (!pattern.test(source)) throw new Error(`Landing SEO template is missing ${label}`);
  return source.replace(pattern, replacement);
}

function metaTag(attribute: "name" | "property", key: string, content: string): string {
  return `<meta ${attribute}="${key}" content="${escapeAttribute(content)}">`;
}

function renderStaticSeoContent(page: (typeof SEO_PAGES)[SeoPath]): string {
  const content = page.staticContent ?? {
    heading: page.title,
    paragraphs: [page.description],
    links: [{ href: "/", label: "Return to Mycellios" }],
  };
  const paragraphs = content.paragraphs.map((paragraph) => `<p>${escapeAttribute(paragraph)}</p>`).join("");
  const links = content.links
    .map(({ href, label }) => `<li><a href="${escapeAttribute(href)}">${escapeAttribute(label)}</a></li>`)
    .join("");

  return `<main class="seo-static-fallback"><h1>${escapeAttribute(content.heading)}</h1>${paragraphs}<nav aria-label="Related Mycellios pages"><ul>${links}</ul></nav></main>`;
}

function renderSeoDocument(template: string, path: (typeof LANDING_SEO_PATHS)[number]): string {
  const page = seoPageForPath(path);
  const url = canonicalUrl(page);
  let html = template;

  html = replaceRequired(html, /<title>[\s\S]*?<\/title>/i, `<title>${escapeAttribute(page.title)}</title>`, "title");
  html = replaceRequired(
    html,
    /<meta\b[^>]*\bname=["']description["'][^>]*>/i,
    metaTag("name", "description", page.description),
    "description",
  );
  html = replaceRequired(
    html,
    /<meta\b[^>]*\bname=["']robots["'][^>]*>/i,
    metaTag("name", "robots", page.robots),
    "robots",
  );
  html = replaceRequired(
    html,
    /<meta\b[^>]*\bproperty=["']og:url["'][^>]*>/i,
    metaTag("property", "og:url", url),
    "Open Graph URL",
  );
  html = replaceRequired(
    html,
    /<meta\b[^>]*\bproperty=["']og:title["'][^>]*>/i,
    metaTag("property", "og:title", page.title),
    "Open Graph title",
  );
  html = replaceRequired(
    html,
    /<meta\b[^>]*\bproperty=["']og:description["'][^>]*>/i,
    metaTag("property", "og:description", page.description),
    "Open Graph description",
  );
  html = replaceRequired(
    html,
    /<meta\b[^>]*\bname=["']twitter:title["'][^>]*>/i,
    metaTag("name", "twitter:title", page.title),
    "Twitter title",
  );
  html = replaceRequired(
    html,
    /<meta\b[^>]*\bname=["']twitter:description["'][^>]*>/i,
    metaTag("name", "twitter:description", page.description),
    "Twitter description",
  );
  html = replaceRequired(
    html,
    /<link\b[^>]*\brel=["']canonical["'][^>]*>/i,
    `<link rel="canonical" href="${escapeAttribute(url)}">`,
    "canonical URL",
  );
  html = replaceRequired(
    html,
    /<div\s+id=["']root["']><\/div>/i,
    `<div id="root">${renderStaticSeoContent(page)}</div>`,
    "static page content",
  );

  const structuredDataPattern =
    /<script\b[^>]*\bid=["']seo-structured-data["'][^>]*>[\s\S]*?<\/script>/i;
  if (page.structuredData) {
    const json = JSON.stringify(page.structuredData).replaceAll("<", "\\u003c");
    html = replaceRequired(
      html,
      structuredDataPattern,
      `<script id="seo-structured-data" type="application/ld+json">${json}</script>`,
      "structured data",
    );
  } else {
    html = replaceRequired(html, structuredDataPattern, "", "structured data");
  }

  return html;
}

function countMatches(source: string, pattern: RegExp): number {
  return [...source.matchAll(pattern)].length;
}

function verifyDocument(path: (typeof LANDING_SEO_PATHS)[number], html: string): void {
  const page = seoPageForPath(path);
  const url = canonicalUrl(page);
  const expected = [
    `<title>${escapeAttribute(page.title)}</title>`,
    metaTag("name", "description", page.description),
    metaTag("name", "robots", page.robots),
    metaTag("property", "og:url", url),
    `<link rel="canonical" href="${escapeAttribute(url)}">`,
  ];
  for (const fragment of expected) {
    if (!html.includes(fragment)) throw new Error(`Generated SEO document for ${path} is missing ${fragment}`);
  }
  const pageContent = page.staticContent ?? { heading: page.title, paragraphs: [page.description] };
  const staticBody = html.match(/<main\s+class=["']seo-static-fallback["']>([\s\S]*?)<\/main>/i)?.[1];
  if (!staticBody || countMatches(staticBody, /<h1\b/gi) !== 1) {
    throw new Error(`Generated SEO document for ${path} must include exactly one static H1`);
  }
  if (!staticBody.includes(`<h1>${escapeAttribute(pageContent.heading)}</h1>`)) {
    throw new Error(`Generated SEO document for ${path} has the wrong static H1`);
  }
  for (const paragraph of pageContent.paragraphs) {
    if (!staticBody.includes(`<p>${escapeAttribute(paragraph)}</p>`)) {
      throw new Error(`Generated SEO document for ${path} is missing static page content`);
    }
  }
  if (countMatches(html, /<link\b[^>]*\brel=["']canonical["'][^>]*>/gi) !== 1) {
    throw new Error(`Generated SEO document for ${path} must contain exactly one canonical URL`);
  }
  const structuredData = html.match(
    /<script\b[^>]*\bid=["']seo-structured-data["'][^>]*>([\s\S]*?)<\/script>/i,
  )?.[1];
  if (page.structuredData) {
    if (!structuredData) throw new Error(`Generated SEO document for ${path} is missing structured data`);
    JSON.parse(structuredData);
  } else if (structuredData) {
    throw new Error(`Non-indexable SEO document for ${path} must not expose structured data`);
  }
}

function verifySitemap(): void {
  const sitemap = readFileSync(join(outputRoot, "sitemap.xml"), "utf8");
  const actual = [...sitemap.matchAll(/<loc>(.*?)<\/loc>/g)].map((match) => match[1]);
  const expected = [
    ...INDEXABLE_SEO_PATHS.map((path) => canonicalUrl(SEO_PAGES[path])),
    `${canonicalUrl(SEO_PAGES["/"])}blog`,
  ];
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`Sitemap routes do not match the SEO contract.\nExpected: ${expected.join(", ")}\nActual: ${actual.join(", ")}`);
  }
  if (sitemap.includes(canonicalUrl(SEO_PAGES["/admin"]))) {
    throw new Error("The admin route must not appear in the sitemap");
  }
}

function verifyRouteContract(): void {
  const titles = new Set<string>();
  const canonicals = new Set<string>();
  for (const path of Object.keys(SEO_PAGES) as SeoPath[]) {
    const page = seoPageForPath(path);
    const url = canonicalUrl(page);
    if (titles.has(page.title)) throw new Error(`Duplicate SEO title: ${page.title}`);
    if (canonicals.has(url)) throw new Error(`Duplicate canonical URL: ${url}`);
    titles.add(page.title);
    canonicals.add(url);
  }
}

verifyRouteContract();
const template = readFileSync(templatePath, "utf8");
for (const path of LANDING_SEO_PATHS) {
  const outputPath = join(outputRoot, routeOutputPaths[path]);
  const html = renderSeoDocument(template, path);
  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, html);
  verifyDocument(path, html);
}
verifySitemap();

console.log(`Generated and verified ${LANDING_SEO_PATHS.length} route-specific landing documents.`);
