import react from "@vitejs/plugin-react";
import { resolve } from "node:path";
import { defineConfig } from "vite";

const LOCAL_UI_PORT = 4_174;
const apiTarget = process.env.MYCELLIOS_UI_API_TARGET?.trim()
  || "https://www.mycellios.com";
const productionProxy = () => ({
  target: apiTarget,
  changeOrigin: true,
  secure: apiTarget.startsWith("https://"),
});

export default defineConfig({
  root: "landing",
  plugins: [react()],
  publicDir: "public",
  server: {
    host: "127.0.0.1",
    port: LOCAL_UI_PORT,
    strictPort: true,
    proxy: {
      "/public": productionProxy(),
      "/v1": productionProxy(),
    },
  },
  preview: {
    host: "127.0.0.1",
    port: LOCAL_UI_PORT,
    strictPort: true,
    proxy: {
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
