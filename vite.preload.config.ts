import { builtinModules } from "node:module";
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    sourcemap: true,
    rollupOptions: {
      external: ["electron", ...builtinModules, ...builtinModules.map((module) => `node:${module}`)],
      output: {
        format: "cjs",
        entryFileNames: "[name].cjs",
        chunkFileNames: "[name].cjs",
        assetFileNames: "[name].[ext]",
      },
    },
  },
});
