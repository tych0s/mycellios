import Fastify from "fastify";
import { createHmac } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ContentHubClient,
  registerContentHubRoutes,
  type BlogPost,
  type PaginatedPosts,
} from "../src/coordinator/content-hub.js";
import { loadCoordinatorConfig } from "../src/core/config.js";

describe("Content Hub blog integration", () => {
  const apps: Array<ReturnType<typeof Fastify>> = [];

  afterEach(async () => {
    vi.restoreAllMocks();
    for (const app of apps.splice(0).reverse()) await app.close();
  });

  it("loads the Content Hub endpoint and prioritizes the Mycellios webhook secret", () => {
    const config = loadCoordinatorConfig({
      GPU_MESH_DB: ":memory:",
      CONTENT_HUB_API_URL: "https://content.example.com",
      MYCELLIOS_PUBLICATION_WEBHOOK_SECRET: "site-specific-secret",
      PUBLICATION_WEBHOOK_SECRET: "legacy-secret",
    });

    expect(config.contentHubApiUrl).toBe("https://content.example.com");
    expect(config.publicationWebhookSecret).toBe("site-specific-secret");
  });

  it("validates and caches Content Hub responses, with bounded stale-if-error", async () => {
    let now = 1_000;
    const payload = postsPage();
    const request = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(Response.json(payload))
      .mockRejectedValueOnce(new Error("offline"))
      .mockRejectedValueOnce(new Error("still offline"));
    const client = new ContentHubClient({
      baseUrl: "https://content.example.com/",
      fetch: request,
      cacheTtlMs: 60_000,
      staleIfErrorMs: 120_000,
      now: () => now,
    });

    await expect(client.getPosts()).resolves.toEqual(payload);
    await expect(client.getPosts()).resolves.toEqual(payload);
    expect(request).toHaveBeenCalledTimes(1);

    now += 60_001;
    await expect(client.getPosts()).resolves.toEqual(payload);
    expect(request).toHaveBeenCalledTimes(2);

    now += 120_001;
    await expect(client.getPosts()).rejects.toThrow("still offline");
    expect(request).toHaveBeenCalledTimes(3);
  });

  it("server-renders list and article routes without trusting metadata as HTML", async () => {
    const post = postFixture({
      title: `Research </script><script>alert("metadata")</script>`,
      excerpt: "A validated excerpt with <strong>markup</strong> that must remain plain text.",
      contentHtml: "<p><strong>Trusted article HTML</strong> from Content Hub.</p>",
      seo: {
        title: "Distributed intelligence research | mycellios",
        description:
          "A practical field note about coordinating heterogeneous machines as one distributed AI inference system.",
        canonicalUrl: "https://untrusted.example/research-note",
        noIndex: false,
      },
    });
    const request = vi.fn<typeof fetch>(async (url) => {
      const href = String(url);
      return href.includes("/posts/research-note?")
        ? Response.json(post)
        : Response.json(postsPage({ items: [post] }));
    });
    const app = Fastify();
    apps.push(app);
    await registerContentHubRoutes(app, {
      client: new ContentHubClient({
        baseUrl: "https://content.example.com",
        fetch: request,
      }),
    });

    const listing = await app.inject({ method: "GET", url: "/blog" });
    expect(listing.statusCode).toBe(200);
    expect(listing.headers["content-type"]).toContain("text/html");
    expect(listing.body).toContain(`href="/blog.css?v=20260723"`);
    expect(listing.body).toContain(`class="blog-feed"`);
    expect(listing.body).toContain(
      `<a href="/blog" aria-current="page">Blog</a>`,
    );
    expect(listing.body).toContain("&lt;strong&gt;markup&lt;/strong&gt;");
    expect(listing.body).not.toContain("<strong>markup</strong>");

    const trailingSlash = await app.inject({
      method: "GET",
      url: "/blog/?page=2",
    });
    expect(trailingSlash.statusCode).toBe(308);
    expect(trailingSlash.headers.location).toBe("/blog?page=2");

    const article = await app.inject({ method: "GET", url: "/blog/research-note" });
    expect(article.statusCode).toBe(200);
    expect(article.body).toContain(`class="blog-article"`);
    expect(article.body).toContain(`class="blog-prose"`);
    expect(article.body).toContain("<p><strong>Trusted article HTML</strong> from Content Hub.</p>");
    expect(article.body).toContain("Research &lt;/script&gt;&lt;script&gt;");
    expect(article.body).not.toContain(`</script><script>alert("metadata")</script>`);
    expect(article.body).toContain(`"@type":"BlogPosting"`);
    expect(article.body).toContain("\\u003c/script\\u003e");
    expect(article.body).toContain(
      `<link rel="canonical" href="https://www.mycellios.com/blog/research-note"`,
    );
    expect(article.body).not.toContain("https://untrusted.example/research-note");
    expect(article.body).toContain(`property="og:type" content="article"`);

    const articleTrailingSlash = await app.inject({
      method: "GET",
      url: "/blog/research-note/?utm_source=content-hub",
    });
    expect(articleTrailingSlash.statusCode).toBe(308);
    expect(articleTrailingSlash.headers.location).toBe(
      "/blog/research-note?utm_source=content-hub",
    );
  });

  it("verifies signed webhooks, deduplicates events and invalidates the cache", async () => {
    const secret = "test-publication-secret";
    const request = vi.fn<typeof fetch>().mockImplementation(async () =>
      Response.json(postsPage()),
    );
    const client = new ContentHubClient({
      baseUrl: "https://content.example.com",
      fetch: request,
    });
    const app = Fastify();
    apps.push(app);
    await registerContentHubRoutes(app, {
      client,
      publicationWebhookSecret: secret,
    });

    expect((await app.inject({ method: "GET", url: "/blog" })).statusCode).toBe(200);
    expect((await app.inject({ method: "GET", url: "/blog" })).statusCode).toBe(200);
    expect(request).toHaveBeenCalledTimes(1);

    const event = {
      id: "ed3aeeeb-a741-48f1-9e13-d67aacbc4ab9",
      type: "post.published",
      occurredAt: "2026-07-23T10:30:00.000Z",
      site: "mycellios",
      locale: "en",
      slug: "research-note",
      url: "https://www.mycellios.com/blog/research-note",
    } as const;
    const body = JSON.stringify(event);
    const signature = createHmac("sha256", secret).update(body).digest("hex");
    const headers = {
      "content-type": "application/json",
      "x-content-hub-event": event.type,
      "x-content-hub-idempotency-key": event.id,
      "x-content-hub-signature": `sha256=${signature}`,
    };

    const accepted = await app.inject({
      method: "POST",
      url: "/api/content-hub/webhook",
      headers,
      payload: body,
    });
    expect(accepted.statusCode).toBe(202);
    expect(accepted.json()).toEqual({ accepted: true, duplicate: false });

    expect((await app.inject({ method: "GET", url: "/blog" })).statusCode).toBe(200);
    expect(request).toHaveBeenCalledTimes(2);

    const replay = await app.inject({
      method: "POST",
      url: "/api/content-hub/webhook",
      headers,
      payload: body,
    });
    expect(replay.statusCode).toBe(200);
    expect(replay.json()).toEqual({ accepted: true, duplicate: true });

    expect((await app.inject({ method: "GET", url: "/blog" })).statusCode).toBe(200);
    expect(request).toHaveBeenCalledTimes(2);

    const rejected = await app.inject({
      method: "POST",
      url: "/api/content-hub/webhook",
      headers: { ...headers, "x-content-hub-signature": `sha256=${"0".repeat(64)}` },
      payload: body,
    });
    expect(rejected.statusCode).toBe(401);
  });

  it("builds a dynamic sitemap and falls back to the static sitemap", async () => {
    const post = postFixture();
    const dynamicApp = Fastify();
    apps.push(dynamicApp);
    await registerContentHubRoutes(dynamicApp, {
      client: new ContentHubClient({
        baseUrl: "https://content.example.com",
        fetch: vi.fn<typeof fetch>().mockResolvedValue(
          Response.json(postsPage({ items: [post] })),
        ),
      }),
    });

    const dynamicSitemap = await dynamicApp.inject({
      method: "GET",
      url: "/sitemap.xml",
    });
    expect(dynamicSitemap.statusCode).toBe(200);
    expect(dynamicSitemap.headers["content-type"]).toContain("application/xml");
    for (const path of ["/", "/network", "/join", "/downloads", "/mobile/", "/blog"]) {
      expect(dynamicSitemap.body).toContain(
        `<loc>https://www.mycellios.com${path}</loc>`,
      );
    }
    expect(dynamicSitemap.body).toContain(
      "<loc>https://www.mycellios.com/blog/research-note</loc>",
    );
    expect(dynamicSitemap.body).toContain(`<lastmod>${post.updatedAt}</lastmod>`);

    const fallbackPath = resolve("landing/public/sitemap.xml");
    const fallbackApp = Fastify();
    apps.push(fallbackApp);
    await registerContentHubRoutes(fallbackApp, {
      client: new ContentHubClient({
        baseUrl: "https://content.example.com",
        fetch: vi.fn<typeof fetch>().mockRejectedValue(new Error("offline")),
      }),
      fallbackSitemapPath: fallbackPath,
    });
    const fallback = await fallbackApp.inject({
      method: "GET",
      url: "/sitemap.xml",
    });
    expect(fallback.statusCode).toBe(200);
    expect(fallback.body).toBe(readFileSync(fallbackPath, "utf8"));
  });

  it("returns HTML fallbacks for missing content or an unconfigured hub", async () => {
    const app = Fastify();
    apps.push(app);
    await registerContentHubRoutes(app, { client: null });

    const unavailable = await app.inject({ method: "GET", url: "/blog" });
    expect(unavailable.statusCode).toBe(503);
    expect(unavailable.headers["cache-control"]).toBe("no-store");
    expect(unavailable.body).toContain("Research is temporarily unavailable");
    expect(unavailable.body).toContain(`name="robots" content="noindex, nofollow"`);

    const missing = Fastify();
    apps.push(missing);
    await registerContentHubRoutes(missing, {
      client: new ContentHubClient({
        baseUrl: "https://content.example.com",
        fetch: vi.fn<typeof fetch>().mockResolvedValue(
          new Response(null, { status: 404 }),
        ),
      }),
    });
    const response = await missing.inject({
      method: "GET",
      url: "/blog/does-not-exist",
    });
    expect(response.statusCode).toBe(404);
    expect(response.body).toContain("Article not found");
  });
});

function postsPage(
  overrides: Partial<PaginatedPosts> = {},
): PaginatedPosts {
  return {
    items: [],
    page: 1,
    pageSize: 12,
    total: overrides.items?.length ?? 0,
    ...overrides,
  };
}

function postFixture(overrides: Partial<BlogPost> = {}): BlogPost {
  return {
    id: "71fa879f-ac31-4505-9781-e543e84298a4",
    site: "mycellios",
    locale: "en",
    slug: "research-note",
    title: "A practical note about distributed intelligence",
    excerpt:
      "What we learned while coordinating heterogeneous machines as one inference system.",
    category: "Engineering",
    author: {
      name: "mycellios research",
      role: "Engineering team",
    },
    image: {
      url: "https://cdn.example.com/research-note.webp",
      alt: "A network of connected computers",
      width: 1_600,
      height: 900,
    },
    publishedAt: "2026-07-23T09:00:00.000Z",
    updatedAt: "2026-07-23T10:00:00.000Z",
    contentHtml: "<p>Validated and sanitized article content.</p>",
    readingMinutes: 6,
    seo: {
      title: "Distributed intelligence research | mycellios",
      description:
        "A practical field note about coordinating heterogeneous machines as one distributed AI inference system.",
      canonicalUrl: "https://www.mycellios.com/blog/research-note",
      noIndex: false,
    },
    alternateUrls: {},
    ...overrides,
  };
}
