import { afterEach, describe, expect, it } from "vitest";
import {
  createCoordinator,
  type CoordinatorRuntime,
} from "../src/coordinator/server.js";
import { SupabaseAuthService } from "../src/coordinator/supabase-auth.js";

const runtimes: CoordinatorRuntime[] = [];

afterEach(async () => {
  await Promise.all(runtimes.splice(0).map((runtime) => runtime.close()));
});

describe("remote diagnostics API", () => {
  it("requires network ingestion and administrator read credentials", async () => {
    const runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 0,
      databasePath: ":memory:",
      requestTimeoutMs: 1_000,
      networkToken: "network-secret",
      modelAdminToken: "admin-secret",
    }, { logger: false });
    runtimes.push(runtime);
    const payload = {
      events: [{
        ...diagnosticEvent(),
        message: "runtime failed token=server-visible",
        details: JSON.stringify({ password: "server-visible", reason: "timeout" }),
      }],
    };

    const anonymousWrite = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/diagnostics",
      payload,
    });
    expect(anonymousWrite.statusCode).toBe(401);

    const accepted = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/diagnostics",
      headers: { authorization: "Bearer network-secret" },
      payload,
    });
    expect(accepted.statusCode).toBe(201);
    expect(accepted.json()).toEqual({ accepted: 1, duplicates: 0 });

    const duplicate = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/diagnostics",
      headers: { authorization: "Bearer network-secret" },
      payload,
    });
    expect(duplicate.statusCode).toBe(200);
    expect(duplicate.json()).toEqual({ accepted: 0, duplicates: 1 });

    const anonymousRead = await runtime.app.inject({
      method: "GET",
      url: "/public/v1/admin/diagnostics",
    });
    expect(anonymousRead.statusCode).toBe(401);

    const read = await runtime.app.inject({
      method: "GET",
      url: "/public/v1/admin/diagnostics?level=error&limit=10",
      headers: { authorization: "Bearer admin-secret" },
    });
    expect(read.statusCode).toBe(200);
    expect(read.json()).toMatchObject({
      events: [{
        id: diagnosticEvent().id,
        event: "runtime-start-failed",
        level: "error",
        message: "runtime failed token=[REDACTED]",
      }],
    });
    expect(read.body).not.toContain("server-visible");
  });

  it("rejects oversized or malformed diagnostic payloads", async () => {
    const runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 0,
      databasePath: ":memory:",
      requestTimeoutMs: 1_000,
      networkToken: "network-secret",
    }, { logger: false });
    runtimes.push(runtime);

    const response = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/diagnostics",
      headers: { authorization: "Bearer network-secret" },
      payload: {
        events: [{
          ...diagnosticEvent(),
          message: "x".repeat(1_201),
        }],
      },
    });

    expect(response.statusCode).toBe(400);
  });

  it("does not expose diagnostics to an operator account", async () => {
    const auth = new SupabaseAuthService(
      "https://accounts.example.test",
      "service-role",
      (async (input) => {
        const url = String(input);
        if (url.includes("/auth/v1/user")) {
          return Response.json({
            id: "73b6d6c3-ec87-4f67-b98e-9ffca4fb5576",
            email: "operator@example.test",
          });
        }
        if (url.includes("/rest/v1/network_members")) {
          return Response.json([{ role: "operator" }]);
        }
        return new Response(null, { status: 404 });
      }) as typeof fetch,
    );
    const runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 0,
      databasePath: ":memory:",
      requestTimeoutMs: 1_000,
    }, { logger: false, supabaseAuthService: auth });
    runtimes.push(runtime);

    const response = await runtime.app.inject({
      method: "GET",
      url: "/public/v1/admin/diagnostics",
      headers: { authorization: "Bearer account-session" },
      remoteAddress: "203.0.113.44",
    });

    expect(response.statusCode).toBe(403);
    expect(response.json()).toMatchObject({
      error: { code: "insufficient_network_role" },
    });
  });
});

function diagnosticEvent() {
  return {
    id: `diag_${"a".repeat(64)}`,
    sourceId: "3d7e5bea-c718-4e82-8f05-dfbda14257fc",
    appVersion: "0.2.54",
    platform: "win32",
    arch: "x64",
    level: "error",
    source: "runtime",
    event: "runtime-start-failed",
    message: "CUDA runtime could not start.",
    // Keep the fixture inside the retention window. A fixed historical date
    // eventually gets pruned after the first insert and makes the retry look
    // like a new event instead of exercising idempotency.
    occurredAt: new Date().toISOString(),
  };
}
