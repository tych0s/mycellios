import react from "@vitejs/plugin-react";
import { createRequire } from "node:module";
import { resolve, sep } from "node:path";
import { defineConfig, loadEnv } from "vite";
import { buildNativeSourceProvenance } from "./scripts/native-build-provenance.mjs";

/*
 * Where `node_modules` actually lives.
 *
 * A git worktree shares one install with the repo it was cut from, so inside
 * `.state/worktrees/<branch>` there is no local `node_modules` and every import
 * resolves to an ancestor directory — outside Vite's root. Vite serves those
 * through `/@fs/`, and its default allow list only covers the project root, so
 * anything reached as a plain asset rather than as a module gets a 403. That is
 * how `@fontsource-variable/manrope` fails: its `index.css` is in the module
 * graph and loads fine, but the `.woff2` it points at is a raw file request and
 * is denied — the page renders in fallback fonts with no error of its own.
 *
 * Resolving a known package and cutting the path at its `node_modules` segment
 * finds the real directory wherever it is, so this stays correct both in the
 * repo proper and in any worktree cut from it.
 *
 * Cut at the *first* segment, not the last. pnpm resolves through a nested
 * store — `<root>/node_modules/.pnpm/vite@8.2.1_.../node_modules/vite` — so
 * the last segment is vite's own private directory, which contains none of
 * the other packages. Allowing that instead of the store root let the font
 * 403 come straight back, silently, in exactly the way this comment exists to
 * describe.
 */
const installedModulesDir = (() => {
  try {
    const entry = createRequire(import.meta.url).resolve("vite/package.json");
    const marker = `${sep}node_modules${sep}`;
    const at = entry.indexOf(marker);
    return at === -1 ? null : entry.slice(0, at + marker.length - 1);
  } catch {
    return null;
  }
})();
const servableRoots = [
  resolve(import.meta.dirname),
  ...(installedModulesDir ? [installedModulesDir] : []),
];
const landingBuildProvenance = buildNativeSourceProvenance(import.meta.dirname);
const landingBuildIdentity = {
  schema: landingBuildProvenance.schema,
  version: landingBuildProvenance.version,
  sourceId: landingBuildProvenance.sourceId,
};

const LOCAL_UI_PORT = Number(process.env.PORT || 4_174);
const LOCAL_UI_HOST = process.env.HOST || "127.0.0.1";
const landingEnv = loadEnv(process.env.NODE_ENV || "development", resolve(import.meta.dirname, "landing"), "");
const apiTarget = process.env.MYCELLIOS_UI_API_TARGET?.trim()
  || landingEnv.MYCELLIOS_UI_API_TARGET?.trim()
  || "https://www.mycellios.com";
const identityTarget = process.env.MYCELLIOS_UI_IDENTITY_TARGET?.trim()
  || landingEnv.MYCELLIOS_UI_IDENTITY_TARGET?.trim()
  || "https://www.mycellios.com";
const productionProxy = (websocket = false) => ({
  target: apiTarget,
  changeOrigin: true,
  secure: apiTarget.startsWith("https://"),
  ...(websocket ? { ws: true } : {}),
});
const identityProxy = () => ({
  target: identityTarget,
  changeOrigin: true,
  secure: identityTarget.startsWith("https://"),
});

/*
 * Let the browser start fetching the hero renderer while it is still parsing
 * the HTML.
 *
 * The mushroom chunk is the biggest asset on the page and it is reached through
 * a dynamic `import()`, so the browser cannot know it exists until it has
 * downloaded and parsed the ~200KB entry bundle. Even with the import kicked
 * off at the top of the bootstrap, that is a serial hop the preload scanner
 * could have skipped entirely.
 *
 * This injects the preload from a tiny inline script in `<head>`, which runs
 * during HTML parse — before the entry bundle has even finished downloading —
 * so the two large fetches overlap from the very start.
 *
 * It is a script rather than a plain `<link rel="modulepreload" media="...">`
 * because the preload has to be *conditional* and `media` on `modulepreload` is
 * not reliably honoured across browsers. Every landing viewport now draws the
 * same organism, so the preload is route-conditional but no longer width- or
 * motion-conditional; mobile quality is controlled inside the renderer.
 */
function preloadHeroRenderer() {
  let file: string | null = null;
  return {
    name: "mycellios:preload-hero-renderer",
    apply: "build" as const,
    generateBundle(_options: unknown, bundle: Record<string, unknown>) {
      file = Object.keys(bundle).find((name) => /mushroom-stage-[^/]*\.js$/.test(name)) ?? null;
    },
    transformIndexHtml(html: string) {
      if (!file) return html;
      const script =
        `<script>(function(){` +
        `if(location.pathname!=="/"&&location.pathname!=="/rebrand"&&location.pathname!=="/rebrand/")return;` +
        `var l=document.createElement("link");` +
        `l.rel="modulepreload";l.crossOrigin="";l.fetchPriority="high";l.href="/${file}";` +
        `document.head.appendChild(l);` +
        `})();</script>`;
      return html.replace("</head>", `  ${script}\n  </head>`);
    },
  };
}

export default defineConfig({
  root: "landing",
  define: {
    __MYCELLIOS_BUILD_IDENTITY__: JSON.stringify(landingBuildIdentity),
  },
  plugins: [react(), preloadHeroRenderer()],
  publicDir: "public",
  server: {
    host: LOCAL_UI_HOST,
    port: LOCAL_UI_PORT,
    strictPort: true,
    allowedHosts: ["www.mycellios.com"],
    fs: { allow: servableRoots },
    proxy: {
      "/public/v1/auth-config": identityProxy(),
      "/v1/auth": identityProxy(),
      "/mobile/v1": productionProxy(true),
      "/local": productionProxy(),
      "/public": productionProxy(),
      "/v1": productionProxy(),
    },
  },
  preview: {
    host: LOCAL_UI_HOST,
    port: LOCAL_UI_PORT,
    strictPort: true,
    allowedHosts: ["www.mycellios.com"],
    proxy: {
      "/public/v1/auth-config": identityProxy(),
      "/v1/auth": identityProxy(),
      "/mobile/v1": productionProxy(true),
      "/local": productionProxy(),
      "/public": productionProxy(),
      "/v1": productionProxy(),
    },
  },
  build: {
    outDir: resolve(import.meta.dirname, "landing-dist"),
    emptyOutDir: true,
    sourcemap: true,
  },
});
