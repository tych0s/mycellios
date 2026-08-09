import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts", "landing/**/*.test.ts", "landing/**/*.test.tsx"],
    exclude: ["dist/**", "node_modules/**"],
    testTimeout: 60_000,
    hookTimeout: 30_000,
  },
});
