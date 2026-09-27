import type { FastifyInstance } from "fastify";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

const REPOSITORY_DOCS = "https://github.com/tych0s/mycellios/blob/main/docs";
const CANONICAL_LANDING_HOST = "www.mycellios.com";
const CANONICAL_LANDING_PATHS = new Set([
  "/",
  "/network",
  "/create",
  "/earn",
  "/spore",
  "/spore/treasury",
  "/spore/data",
  "/downloads",
  "/admin",
  "/dashboard",
  "/account",
  "/browser/",
]);

export function canonicalLandingRedirectLocation(
  method: string,
  hostname: string,
  requestUrl: string,
): string | null {
  if (!(["GET", "HEAD"].includes(method.toUpperCase())) || hostname.toLowerCase() !== "mycellios.com") return null;

  const queryIndex = requestUrl.indexOf("?");
  const pathname = queryIndex === -1 ? requestUrl : requestUrl.slice(0, queryIndex);
  const query = queryIndex === -1 ? "" : requestUrl.slice(queryIndex);
  const normalizedPath = pathname.length > 1 ? pathname.replace(/\/+$/, "") : "/";
  const canonicalPath = normalizedPath === "/browser" || normalizedPath === "/mobile"
    ? "/browser/"
    : normalizedPath;
  const isLandingPath = CANONICAL_LANDING_PATHS.has(canonicalPath)
    || canonicalPath === "/blog"
    || canonicalPath.startsWith("/blog/");

  return isLandingPath ? `https://${CANONICAL_LANDING_HOST}${canonicalPath}${query}` : null;
}

export function registerCanonicalLandingHostRedirect(app: FastifyInstance): void {
  app.addHook("onRequest", async (request, reply) => {
    const location = canonicalLandingRedirectLocation(request.method, request.hostname, request.url);
    if (location) return reply.redirect(location, 308);
  });
}

export function registerLandingEntryRoutes(app: FastifyInstance, assetsRoot: string): void {
  const routeDocuments = {
    "/network": "network/index.html",
    "/dashboard": "dashboard/index.html",
    "/create": "create/index.html",
    "/earn": "earn/index.html",
    "/account": "account/index.html",
    "/spore": "spore/index.html",
    "/spore/treasury": "spore/treasury/index.html",
    "/spore/data": "spore/data/index.html",
    "/admin": "admin/index.html",
    "/downloads": "downloads/index.html",
  } as const;
  for (const [path, documentName] of Object.entries(routeDocuments)) {
    const document = existsSync(resolve(assetsRoot, documentName)) ? documentName : "index.html";
    app.get(path, async (_request, reply) => reply.sendFile(document));
  }

  const redirects = [
    ["/join", "/earn"], ["/join/", "/earn"],
    ["/docs", `${REPOSITORY_DOCS}/README.md`], ["/docs/", `${REPOSITORY_DOCS}/README.md`],
    ["/docs/protocol", `${REPOSITORY_DOCS}/ARCHITECTURE.md`], ["/docs/protocol/", `${REPOSITORY_DOCS}/ARCHITECTURE.md`],
    ["/docs/downloads", "/downloads"], ["/docs/network", "/network"], ["/docs/blog", "/blog"],
    ["/docs/downloads/", "/downloads"], ["/docs/network/", "/network"], ["/docs/blog/", "/blog"],
  ] as const;
  for (const [path, destination] of redirects) {
    app.get(path, async (_request, reply) => reply.redirect(destination));
  }
  for (const path of ["/dashboard", "/account", "/spore", "/spore/treasury", "/spore/data"]) {
    app.get(`${path}/`, async (request, reply) => {
      const queryIndex = request.url.indexOf("?");
      return reply.redirect(`${path}${queryIndex === -1 ? "" : request.url.slice(queryIndex)}`);
    });
  }
}
