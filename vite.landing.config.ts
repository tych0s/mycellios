import react from "@vitejs/plugin-react";
import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  root: "landing",
  plugins: [react()],
  publicDir: "public",
  server: {
    host: "127.0.0.1",
    port: 4174,
  },
  preview: {
    host: "127.0.0.1",
    port: 4174,
  },
  build: {
    outDir: resolve(import.meta.dirname, "landing-dist"),
    emptyOutDir: true,
    sourcemap: true,
  },
});
