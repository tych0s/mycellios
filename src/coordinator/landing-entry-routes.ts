import type { FastifyInstance } from "fastify";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

export function registerLandingEntryRoutes(app: FastifyInstance, assetsRoot: string): void {
  const routeDocuments = {
    "/network": "network/index.html",
    "/dashboard": "index.html",
    "/create": "create/index.html",
    "/earn": "earn/index.html",
    "/account": "index.html",
    "/spore": "index.html",
    "/spore/treasury": "index.html",
    "/spore/data": "index.html",
    "/admin": "admin/index.html",
    "/downloads": "downloads/index.html",
  } as const;
  for (const [path, documentName] of Object.entries(routeDocuments)) {
    const document = existsSync(resolve(assetsRoot, documentName)) ? documentName : "index.html";
    app.get(path, async (_request, reply) => reply.sendFile(document));
  }

  const redirects = [
    ["/join", "/earn"], ["/join/", "/earn"], ["/docs", "/docs/"],
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
