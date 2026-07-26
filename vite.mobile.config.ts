import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";
import { buildNativeSourceProvenance } from "./scripts/native-build-provenance.mjs";

const workspaceRoot = import.meta.dirname;
const mobileRoot = resolve(workspaceRoot, "src/mobile");
const mobileBuildProvenance = buildNativeSourceProvenance(workspaceRoot);
const mobileBuildIdentity = {
  schema: mobileBuildProvenance.schema,
  version: mobileBuildProvenance.version,
  sourceId: mobileBuildProvenance.sourceId,
};

function mobileAssets(): Plugin {
  const packageVersion = JSON.parse(
    readFileSync(resolve(workspaceRoot, "package.json"), "utf8"),
  ) as { version: string };
  return {
    name: "mycellios-mobile-assets",
    generateBundle() {
      this.emitFile({
        type: "asset",
        fileName: "manifest.webmanifest",
        source: JSON.stringify(
          {
            name: "mycellios mobile worker",
            short_name: "mycellios",
            description: "Contribute verified compute power to the mycellios network from this device.",
            start_url: "./",
            scope: "./",
            display: "standalone",
            orientation: "portrait",
            background_color: "#11172f",
            theme_color: "#11172f",
            icons: [
              {
                src: "./mycellios-app-icon-v2.png",
                sizes: "1024x1024",
                type: "image/png",
                purpose: "any maskable",
              },
            ],
          },
          null,
          2,
        ),
      });
      this.emitFile({
        type: "asset",
        fileName: "mycellios-app-icon-v2.png",
        source: readFileSync(resolve(workspaceRoot, "build/icons/app-icon-v2.png")),
      });
      this.emitFile({
        type: "asset",
        fileName: "sw.js",
        source: readFileSync(resolve(mobileRoot, "sw.js"), "utf8").replace(
          "__MYCELLIOS_VERSION__",
          packageVersion.version,
        ),
      });
    },
  };
}

export default defineConfig({
  root: mobileRoot,
  base: "./",
  publicDir: false,
  define: {
    __MYCELLIOS_BUILD_IDENTITY__: JSON.stringify(mobileBuildIdentity),
  },
  plugins: [react(), mobileAssets()],
  build: {
    outDir: resolve(workspaceRoot, "mobile-dist"),
    emptyOutDir: true,
    sourcemap: true,
    target: "es2022",
  },
});
