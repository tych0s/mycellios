import { describe, expect, it } from "vitest";
import { createCoordinator } from "../src/coordinator/server.js";

describe("public auth configuration", () => {
  it("serves the browser origin while retaining the internal Supabase origin", async () => {
    const runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 0,
      databasePath: ":memory:",
      requestTimeoutMs: 10_000,
      supabaseUrl: "http://supabase-internal:8443",
      publicSupabaseUrl: "https://auth.example",
      supabaseAnonKey: "public-anon",
    }, { logger: false });
    try {
      const response = await runtime.app.inject({ method: "GET", url: "/public/v1/auth-config" });
      expect(response.statusCode).toBe(200);
      expect(response.json()).toMatchObject({ enabled: true, url: "https://auth.example", anonKey: "public-anon" });
      expect(response.body).not.toContain("supabase-internal");
    } finally {
      await runtime.close();
    }
  });
});
