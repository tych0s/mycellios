import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  root: "src/renderer",
  plugins: [react()],
  build: {
    // Forge copies renderer assets from the repository-level .vite directory.
    // Because this Vite project has a nested root, make the output path explicit.
    outDir: "../../.vite/renderer/main_window",
    emptyOutDir: true,
    sourcemap: true,
  },
});
