import type { FastifyInstance, FastifyReply } from "fastify";
import { createHmac, timingSafeEqual } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { z } from "zod";

const SITE_SLUG = "mycellios";
const SITE_ORIGIN = "https://www.mycellios.com";
const BLOG_PATH = "/blog";
const BLOG_LOCALE = "en";
const BLOG_ASSET_VERSION = "20260816";
const DEFAULT_CACHE_TTL_MS = 60_000;
const DEFAULT_STALE_IF_ERROR_MS = 24 * 60 * 60 * 1_000;
const DEFAULT_REQUEST_TIMEOUT_MS = 5_000;
const WEBHOOK_REPLAY_TTL_MS = 24 * 60 * 60 * 1_000;
const MAX_WEBHOOK_EVENTS = 1_000;
const MAX_SITEMAP_PAGES = 200;
const PUBLIC_PRODUCT_SITEMAP_ROUTES = [
  { path: "/", changeFrequency: "weekly", priority: "1.0" },
  { path: "/network", changeFrequency: "daily", priority: "0.9" },
  { path: "/join", changeFrequency: "weekly", priority: "0.8" },
  { path: "/downloads", changeFrequency: "weekly", priority: "0.9" },
  { path: "/mobile/", changeFrequency: "weekly", priority: "0.7" },
] as const;

const localeSchema = z.string().regex(/^[a-z]{2}(?:-[A-Z]{2})?$/);

const blogAuthorSchema = z.object({
  name: z.string().min(1),
  url: z.string().url().optional(),
  role: z.string().min(1).optional(),
  avatarUrl: z.string().url().optional(),
});

const blogImageSchema = z.object({
  url: z.string().url(),
  alt: z.string().min(1),
  width: z.number().int().positive().optional(),
  height: z.number().int().positive().optional(),
});

const blogPostSummarySchema = z.object({
  id: z.string().uuid(),
  site: z.literal(SITE_SLUG),
  locale: localeSchema,
  slug: z.string().min(1).max(220),
  title: z.string().min(1),
  excerpt: z.string().min(1),
  category: z.string().min(1),
  author: blogAuthorSchema,
  image: blogImageSchema.nullable(),
  publishedAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
});

const seoMetadataSchema = z.object({
  title: z.string().min(10).max(70),
  description: z.string().min(40).max(180),
  canonicalUrl: z.string().url(),
  noIndex: z.boolean().default(false),
});

const blogPostSchema = blogPostSummarySchema.extend({
  contentHtml: z.string().min(1),
  readingMinutes: z.number().int().positive(),
  seo: seoMetadataSchema,
  alternateUrls: z.record(localeSchema, z.string().url()).default({}),
});

const paginatedPostsSchema = z.object({
  items: z.array(blogPostSummarySchema),
  page: z.number().int().positive(),
  pageSize: z.number().int().positive(),
  total: z.number().int().nonnegative(),
});

const publicationEventSchema = z.object({
  id: z.string().uuid(),
  type: z.enum(["post.published", "post.updated", "post.archived"]),
  occurredAt: z.string().datetime(),
  site: z.literal(SITE_SLUG),
  locale: localeSchema,
  slug: z.string().min(1).max(220),
  url: z.string().url(),
});

const blogListQuerySchema = z.object({
  page: z.coerce.number().int().positive().max(10_000).default(1),
});

const blogPostParamsSchema = z.object({
  slug: z.string().regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/).max(220),
});

export type BlogPostSummary = z.infer<typeof blogPostSummarySchema>;
export type BlogPost = z.infer<typeof blogPostSchema>;
export type PaginatedPosts = z.infer<typeof paginatedPostsSchema>;

type CacheEntry = {
  value: unknown;
  freshUntil: number;
  staleUntil: number;
};

export type ContentHubClientOptions = {
  baseUrl: string;
  fetch?: typeof globalThis.fetch;
  cacheTtlMs?: number;
  staleIfErrorMs?: number;
  timeoutMs?: number;
  now?: () => number;
};

export class ContentHubClient {
  readonly #baseUrl: string;
  readonly #fetch: typeof globalThis.fetch;
  readonly #cacheTtlMs: number;
  readonly #staleIfErrorMs: number;
  readonly #timeoutMs: number;
  readonly #now: () => number;
  readonly #cache = new Map<string, CacheEntry>();

  constructor(options: ContentHubClientOptions) {
    const baseUrl = new URL(options.baseUrl);
    this.#baseUrl = baseUrl.href.replace(/\/$/, "");
    this.#fetch = options.fetch ?? globalThis.fetch;
    this.#cacheTtlMs = options.cacheTtlMs ?? DEFAULT_CACHE_TTL_MS;
    this.#staleIfErrorMs = options.staleIfErrorMs ?? DEFAULT_STALE_IF_ERROR_MS;
    this.#timeoutMs = options.timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
    this.#now = options.now ?? Date.now;
  }

  async getPosts(page = 1, pageSize = 12): Promise<PaginatedPosts> {
    const query = new URLSearchParams({
      locale: BLOG_LOCALE,
      page: String(page),
      pageSize: String(pageSize),
    });
    const url = `${this.#baseUrl}/api/v1/sites/${SITE_SLUG}/posts?${query.toString()}`;
    return this.#readThrough(url, async () => {
      const response = await this.#fetch(url, {
        signal: AbortSignal.timeout(this.#timeoutMs),
      });
      if (!response.ok) {
        throw new ContentHubError("Could not load published posts", response.status);
      }
      return paginatedPostsSchema.parse(await response.json());
    });
  }

  async getPost(slug: string): Promise<BlogPost | null> {
    const query = new URLSearchParams({ locale: BLOG_LOCALE });
    const url = `${this.#baseUrl}/api/v1/sites/${SITE_SLUG}/posts/${encodeURIComponent(slug)}?${query.toString()}`;
    return this.#readThrough(url, async () => {
      const response = await this.#fetch(url, {
        signal: AbortSignal.timeout(this.#timeoutMs),
      });
      if (response.status === 404) return null;
      if (!response.ok) {
        throw new ContentHubError("Could not load the published post", response.status);
      }
      return blogPostSchema.parse(await response.json());
    });
  }

  async getAllPostsForSitemap(): Promise<readonly BlogPostSummary[]> {
    const pageSize = 50;
    const firstPage = await this.getPosts(1, pageSize);
    const pageCount = Math.ceil(firstPage.total / pageSize);
    if (pageCount > MAX_SITEMAP_PAGES) {
      throw new ContentHubError("Content Hub sitemap exceeds its safe page limit", 502);
    }
    const pages = await Promise.all(
      Array.from({ length: Math.max(0, pageCount - 1) }, (_, index) =>
        this.getPosts(index + 2, pageSize),
      ),
    );
    return [firstPage, ...pages].flatMap((page) => page.items);
  }

  invalidate(): void {
    this.#cache.clear();
  }

  async #readThrough<T>(key: string, load: () => Promise<T>): Promise<T> {
    const cached = this.#cache.get(key);
    const now = this.#now();
    if (cached && cached.freshUntil > now) return cached.value as T;
    try {
      const value = await load();
      const loadedAt = this.#now();
      this.#cache.set(key, {
        value,
        freshUntil: loadedAt + this.#cacheTtlMs,
        staleUntil: loadedAt + this.#cacheTtlMs + this.#staleIfErrorMs,
      });
      return value;
    } catch (error) {
      if (cached && cached.staleUntil > now) return cached.value as T;
      throw error;
    }
  }
}

export class ContentHubError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ContentHubError";
  }
}

export type ContentHubRoutesOptions = {
  client: ContentHubClient | null;
  publicationWebhookSecret?: string | undefined;
  fallbackSitemapPath?: string | undefined;
  now?: () => number;
};

export async function registerContentHubRoutes(
  app: FastifyInstance,
  options: ContentHubRoutesOptions,
): Promise<void> {
  const fallbackSitemap = loadFallbackSitemap(options.fallbackSitemapPath);
  const webhookEvents = new WebhookEventRegistry(options.now);

  app.get("/blog/", async (request, reply) => {
    const query = request.url.includes("?")
      ? request.url.slice(request.url.indexOf("?"))
      : "";
    return reply
      .code(308)
      .header("Location", `${BLOG_PATH}${query}`)
      .send();
  });

  app.get("/blog", async (request, reply) => {
    const query = blogListQuerySchema.safeParse(request.query);
    if (!query.success) {
      return sendBlogHtml(
        reply,
        renderBlogStatePage(
          "Invalid page",
          "The requested blog page does not exist.",
          { noIndex: true },
        ),
        400,
      );
    }
    if (!options.client) {
      return sendBlogHtml(reply, renderBlogUnavailablePage(), 503);
    }
    try {
      const posts = await options.client.getPosts(query.data.page);
      if (query.data.page > 1 && posts.items.length === 0) {
        return sendBlogHtml(
          reply,
          renderBlogStatePage("Page not found", "There are no articles on this page.", {
            noIndex: true,
          }),
          404,
        );
      }
      return sendBlogHtml(reply, renderBlogIndex(posts), 200);
    } catch (error) {
      request.log.warn({ err: error }, "Content Hub blog listing failed");
      return sendBlogHtml(reply, renderBlogUnavailablePage(), 503);
    }
  });

  app.get("/blog/:slug/", async (request, reply) => {
    const params = blogPostParamsSchema.safeParse(request.params);
    if (!params.success) {
      return sendBlogHtml(
        reply,
        renderBlogStatePage("Article not found", "This article is not available.", {
          noIndex: true,
        }),
        404,
      );
    }
    const query = request.url.includes("?")
      ? request.url.slice(request.url.indexOf("?"))
      : "";
    return reply
      .code(308)
      .header(
        "Location",
        `${BLOG_PATH}/${encodeURIComponent(params.data.slug)}${query}`,
      )
      .send();
  });

  app.get("/blog/:slug", async (request, reply) => {
    const params = blogPostParamsSchema.safeParse(request.params);
    if (!params.success) {
      return sendBlogHtml(
        reply,
        renderBlogStatePage("Article not found", "This article is not available.", {
          noIndex: true,
        }),
        404,
      );
    }
    if (!options.client) {
      return sendBlogHtml(reply, renderBlogUnavailablePage(), 503);
    }
    try {
      const post = await options.client.getPost(params.data.slug);
      if (!post) {
        return sendBlogHtml(
          reply,
          renderBlogStatePage("Article not found", "This article is not available.", {
            noIndex: true,
          }),
          404,
        );
      }
      return sendBlogHtml(reply, renderBlogPost(post), 200);
    } catch (error) {
      request.log.warn({ err: error }, "Content Hub blog article failed");
      return sendBlogHtml(reply, renderBlogUnavailablePage(), 503);
    }
  });

  app.get("/sitemap.xml", async (request, reply) => {
    reply.type("application/xml; charset=utf-8");
    reply.header("Cache-Control", "public, max-age=60, stale-if-error=86400");
    if (!options.client) return reply.send(fallbackSitemap);
    try {
      const posts = await options.client.getAllPostsForSitemap();
      return reply.send(renderSitemap(posts));
    } catch (error) {
      request.log.warn({ err: error }, "Content Hub sitemap refresh failed");
      return reply.send(fallbackSitemap);
    }
  });

  await app.register(async (webhookApp) => {
    webhookApp.removeContentTypeParser("application/json");
    webhookApp.addContentTypeParser(
      "application/json",
      { parseAs: "buffer", bodyLimit: 64 * 1_024 },
      (_request, body, done) => done(null, body),
    );
    webhookApp.post("/api/content-hub/webhook", async (request, reply) => {
      reply.header("Cache-Control", "no-store");
      if (!options.publicationWebhookSecret) {
        return reply.code(503).send({ error: { code: "webhook_not_configured" } });
      }
      if (!Buffer.isBuffer(request.body)) {
        return reply.code(400).send({ error: { code: "invalid_webhook_body" } });
      }
      const signature = firstHeader(request.headers["x-content-hub-signature"]);
      if (
        !signature
        || !verifyWebhookSignature(
          request.body,
          signature,
          options.publicationWebhookSecret,
        )
      ) {
        return reply.code(401).send({ error: { code: "invalid_webhook_signature" } });
      }

      let input: unknown;
      try {
        input = JSON.parse(request.body.toString("utf8")) as unknown;
      } catch {
        return reply.code(400).send({ error: { code: "invalid_webhook_body" } });
      }
      const parsed = publicationEventSchema.safeParse(input);
      if (!parsed.success) {
        return reply.code(400).send({
          error: { code: "invalid_publication_event", details: parsed.error.issues },
        });
      }
      const idempotencyKey = firstHeader(
        request.headers["x-content-hub-idempotency-key"],
      );
      const eventType = firstHeader(request.headers["x-content-hub-event"]);
      if (
        idempotencyKey !== parsed.data.id
        || eventType !== parsed.data.type
      ) {
        return reply.code(400).send({ error: { code: "inconsistent_webhook_headers" } });
      }
      if (!webhookEvents.accept(parsed.data.id)) {
        return reply.send({ accepted: true, duplicate: true });
      }
      options.client?.invalidate();
      return reply.code(202).send({ accepted: true, duplicate: false });
    });
  });
}

class WebhookEventRegistry {
  readonly #events = new Map<string, number>();
  readonly #now: () => number;

  constructor(now: (() => number) | undefined) {
    this.#now = now ?? Date.now;
  }

  accept(id: string): boolean {
    const now = this.#now();
    for (const [eventId, expiresAt] of this.#events) {
      if (expiresAt <= now) this.#events.delete(eventId);
    }
    if (this.#events.has(id)) return false;
    this.#events.set(id, now + WEBHOOK_REPLAY_TTL_MS);
    while (this.#events.size > MAX_WEBHOOK_EVENTS) {
      const oldestId = this.#events.keys().next().value as string | undefined;
      if (!oldestId) break;
      this.#events.delete(oldestId);
    }
    return true;
  }
}

function verifyWebhookSignature(
  body: Buffer,
  signatureHeader: string,
  secret: string,
): boolean {
  const match = /^sha256=([0-9a-f]{64})$/i.exec(signatureHeader);
  if (!match?.[1]) return false;
  const received = Buffer.from(match[1], "hex");
  const expected = createHmac("sha256", secret).update(body).digest();
  return received.length === expected.length && timingSafeEqual(received, expected);
}

function firstHeader(value: string | readonly string[] | undefined): string | undefined {
  return typeof value === "string" ? value : value?.[0];
}

function sendBlogHtml(
  reply: FastifyReply,
  html: string,
  statusCode: number,
): FastifyReply {
  reply.type("text/html; charset=utf-8");
  reply.header(
    "Cache-Control",
    statusCode >= 200 && statusCode < 300
      ? "public, max-age=60, stale-if-error=86400"
      : "no-store",
  );
  return reply.code(statusCode).send(html);
}

function renderBlogIndex(posts: PaginatedPosts): string {
  const pageCount = Math.max(1, Math.ceil(posts.total / posts.pageSize));
  const canonical = posts.page === 1
    ? `${SITE_ORIGIN}${BLOG_PATH}`
    : `${SITE_ORIGIN}${BLOG_PATH}?page=${posts.page}`;
  const entries = posts.items.length > 0
    ? posts.items.map(renderBlogEntry).join("")
    : `<section class="blog-state"><div><p class="blog-hero__eyebrow">Field notes</p><h1>No articles yet</h1><p>Our first research note is being prepared.</p></div></section>`;
  const pagination = renderPagination(posts.page, pageCount);
  return renderDocument({
    title: "Research & field notes | mycellios",
    description:
      "Technical research, product updates and field notes from the team building mycellios.",
    canonical,
    body: `
      ${renderNavigation()}
      <main class="blog-main">
        <header class="blog-hero">
          <p class="blog-hero__eyebrow">Research &amp; field notes</p>
          <h1>Building distributed intelligence, one practical breakthrough at a time.</h1>
          <p>Architecture notes, experiments and product updates from the mycellios team.</p>
        </header>
        <section class="blog-feed" aria-label="Latest articles">${entries}</section>
        ${pagination}
      </main>
      ${renderFooter()}
    `,
  });
}

function renderBlogEntry(post: BlogPostSummary): string {
  const articleUrl = `${BLOG_PATH}/${encodeURIComponent(post.slug)}`;
  const image = post.image
    ? `<a class="blog-entry__media" href="${articleUrl}" aria-label="${escapeHtml(post.title)}">
        <img src="${escapeHtml(post.image.url)}" alt="${escapeHtml(post.image.alt)}" loading="lazy"${renderImageDimensions(post.image)} />
      </a>`
    : "";
  return `
    <article class="blog-entry">
      ${image}
      <div class="blog-entry__body">
        <p class="blog-entry__meta"><span>${escapeHtml(post.category)}</span><time datetime="${escapeHtml(post.publishedAt)}">${formatDate(post.publishedAt)}</time></p>
        <h2 class="blog-entry__title"><a href="${articleUrl}">${escapeHtml(post.title)}</a></h2>
        <p>${escapeHtml(post.excerpt)}</p>
        <a href="${articleUrl}" aria-label="Read ${escapeHtml(post.title)}">Read article <span aria-hidden="true">→</span></a>
      </div>
    </article>
  `;
}

function renderBlogPost(post: BlogPost): string {
  const canonical = `${SITE_ORIGIN}${BLOG_PATH}/${encodeURIComponent(post.slug)}`;
  const jsonLd = safeJson({
    "@context": "https://schema.org",
    "@type": "BlogPosting",
    headline: post.title,
    description: post.excerpt,
    datePublished: post.publishedAt,
    dateModified: post.updatedAt,
    mainEntityOfPage: canonical,
    author: {
      "@type": post.author.url ? "Person" : "Organization",
      name: post.author.name,
      ...(post.author.url ? { url: post.author.url } : {}),
    },
    publisher: {
      "@type": "Organization",
      name: "mycellios",
      url: SITE_ORIGIN,
    },
    ...(post.image ? { image: [post.image.url] } : {}),
  });
  const alternates = Object.entries(post.alternateUrls)
    .map(
      ([locale, url]) =>
        `<link rel="alternate" hreflang="${escapeHtml(locale)}" href="${escapeHtml(url)}" />`,
    )
    .join("\n");
  const image = post.image
    ? `<div class="blog-article__image"><img src="${escapeHtml(post.image.url)}" alt="${escapeHtml(post.image.alt)}"${renderImageDimensions(post.image)} /></div>`
    : "";
  return renderDocument({
    title: post.seo.title,
    description: post.seo.description,
    canonical,
    noIndex: post.seo.noIndex,
    ogType: "article",
    image: post.image?.url,
    extraHead: `${alternates}<script type="application/ld+json">${jsonLd}</script>`,
    body: `
      ${renderNavigation()}
      <main class="blog-main">
        <article class="blog-article">
          <header class="blog-article__header">
            <p class="blog-hero__eyebrow">${escapeHtml(post.category)}</p>
            <h1>${escapeHtml(post.title)}</h1>
            <p>${escapeHtml(post.excerpt)}</p>
            <div class="blog-article__meta">
              <span>By ${escapeHtml(post.author.name)}</span>
              <time datetime="${escapeHtml(post.publishedAt)}">${formatDate(post.publishedAt)}</time>
              <span>${post.readingMinutes} min read</span>
            </div>
          </header>
          ${image}
          <div class="blog-prose">${post.contentHtml}</div>
        </article>
      </main>
      ${renderFooter()}
    `,
  });
}

function renderBlogUnavailablePage(): string {
  return renderBlogStatePage(
    "Research is temporarily unavailable",
    "The connection to our publishing service timed out. Please try again shortly.",
    { noIndex: true },
  );
}

function renderBlogStatePage(
  title: string,
  message: string,
  options: { noIndex: boolean },
): string {
  return renderDocument({
    title: `${title} | mycellios`,
    description: message,
    canonical: `${SITE_ORIGIN}${BLOG_PATH}`,
    noIndex: options.noIndex,
    body: `
      ${renderNavigation()}
      <main class="blog-main">
        <section class="blog-state">
          <div>
            <p class="blog-hero__eyebrow">mycellios journal</p>
            <h1>${escapeHtml(title)}</h1>
            <p>${escapeHtml(message)}</p>
            <a href="${BLOG_PATH}">Back to the journal</a>
          </div>
        </section>
      </main>
      ${renderFooter()}
    `,
  });
}

type DocumentOptions = {
  title: string;
  description: string;
  canonical: string;
  body: string;
  noIndex?: boolean | undefined;
  ogType?: "website" | "article" | undefined;
  image?: string | undefined;
  extraHead?: string | undefined;
};

function renderDocument(options: DocumentOptions): string {
  const image = options.image
    ? `<meta property="og:image" content="${escapeHtml(options.image)}" />
    <meta name="twitter:card" content="summary_large_image" />`
    : `<meta name="twitter:card" content="summary" />`;
  return `<!doctype html>
<html lang="${BLOG_LOCALE}">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
    <meta name="theme-color" content="#07080b" />
    <meta name="color-scheme" content="dark" />
    <title>${escapeHtml(options.title)}</title>
    <meta name="description" content="${escapeHtml(options.description)}" />
    ${options.noIndex ? `<meta name="robots" content="noindex, nofollow" />` : ""}
    <link rel="canonical" href="${escapeHtml(options.canonical)}" />
    <link rel="stylesheet" href="/blog.css?v=${BLOG_ASSET_VERSION}" />
    <link rel="icon" href="/assets/brand/favicon.png" type="image/png" sizes="32x32" />
    <link rel="shortcut icon" href="/assets/brand/favicon.png" type="image/png" />
    <link rel="apple-touch-icon" href="/assets/brand/app-icon.png" />
    <meta property="og:type" content="${options.ogType ?? "website"}" />
    <meta property="og:site_name" content="mycellios" />
    <meta property="og:url" content="${escapeHtml(options.canonical)}" />
    <meta property="og:title" content="${escapeHtml(options.title)}" />
    <meta property="og:description" content="${escapeHtml(options.description)}" />
    ${image}
    ${options.extraHead ?? ""}
  </head>
  <body class="blog-site">
    ${options.body}
  </body>
</html>`;
}

function renderNavigation(): string {
  return `
    <header class="rb-header blog-header">
      <div class="rb-header-inner rb-shell">
        <a class="rb-brand" href="/" aria-label="mycellios, home"><img src="/assets/logos/logo.png" alt="" width="38" height="38" /><span>mycellios</span></a>
        <nav aria-label="Main navigation">
          <a href="/network?view=inference">Chat</a><a href="/create">Create</a><a href="/earn">Earn</a><a href="/spore">$ SPORE</a><a href="/network">Live network</a><a class="active" href="/blog" aria-current="page">Blog</a>
        </nav>
        <div class="rb-header-actions rb-desktop-cta"><a class="rb-social" href="https://github.com/nodecodex-org/mycellios" target="_blank" rel="noreferrer" aria-label="mycellios on GitHub">GH</a><a class="rb-social" href="https://x.com/mycellios" target="_blank" rel="noreferrer" aria-label="mycellios on X">X</a><a class="rb-pill rb-pill-ghost" href="/network?view=overview">Login</a></div>
      </div>
    </header>
  `;
}

function renderFooter(): string {
  return `
    <footer class="blog-footer">
      <p>Many machines. One model.</p>
      <a href="/">mycellios home</a>
    </footer>
  `;
}

function renderPagination(currentPage: number, pageCount: number): string {
  if (pageCount <= 1) return "";
  const previous = currentPage > 1
    ? `<a rel="prev" href="${pageHref(currentPage - 1)}">← Newer</a>`
    : `<span aria-hidden="true"></span>`;
  const next = currentPage < pageCount
    ? `<a rel="next" href="${pageHref(currentPage + 1)}">Older →</a>`
    : `<span aria-hidden="true"></span>`;
  return `
    <nav class="blog-pagination" aria-label="Article pages">
      ${previous}
      <span>Page ${currentPage} of ${pageCount}</span>
      ${next}
    </nav>
  `;
}

function pageHref(page: number): string {
  return page === 1 ? BLOG_PATH : `${BLOG_PATH}?page=${page}`;
}

function renderImageDimensions(image: {
  width?: number | undefined;
  height?: number | undefined;
}): string {
  return `${image.width ? ` width="${image.width}"` : ""}${image.height ? ` height="${image.height}"` : ""}`;
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat("en", {
    dateStyle: "long",
    timeZone: "UTC",
  }).format(new Date(value));
}

function renderSitemap(posts: readonly BlogPostSummary[]): string {
  const urls = [
    ...PUBLIC_PRODUCT_SITEMAP_ROUTES.map((route) =>
      sitemapEntry(
        `${SITE_ORIGIN}${route.path}`,
        undefined,
        route.changeFrequency,
        route.priority,
      ),
    ),
    sitemapEntry(`${SITE_ORIGIN}${BLOG_PATH}`, undefined, "daily", "0.8"),
    ...posts.map((post) =>
      sitemapEntry(
        `${SITE_ORIGIN}${BLOG_PATH}/${encodeURIComponent(post.slug)}`,
        post.updatedAt,
        "weekly",
        "0.7",
      ),
    ),
  ];
  return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.join("\n")}
</urlset>`;
}

function sitemapEntry(
  location: string,
  lastModified: string | undefined,
  changeFrequency: string,
  priority: string,
): string {
  return `  <url>
    <loc>${escapeXml(location)}</loc>
    ${lastModified ? `<lastmod>${escapeXml(lastModified)}</lastmod>\n    ` : ""}<changefreq>${changeFrequency}</changefreq>
    <priority>${priority}</priority>
  </url>`;
}

function loadFallbackSitemap(path: string | undefined): string {
  if (path && existsSync(path)) {
    try {
      return readFileSync(path, "utf8");
    } catch {
      // The built-in fallback below keeps discovery available during deploys.
    }
  }
  return renderSitemap([]);
}

function safeJson(value: unknown): string {
  return JSON.stringify(value)
    .replaceAll("<", "\\u003c")
    .replaceAll(">", "\\u003e")
    .replaceAll("&", "\\u0026");
}

function escapeXml(value: string): string {
  return escapeHtml(value).replaceAll("&#39;", "&apos;");
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
