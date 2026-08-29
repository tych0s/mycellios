import { generateKeyPairSync } from "node:crypto";
import { afterEach, describe, expect, it } from "vitest";
import { createCoordinator, type CoordinatorRuntime } from "../src/coordinator/server.js";
import { SupabaseAuthService } from "../src/coordinator/supabase-auth.js";

const runtimes: CoordinatorRuntime[] = [];
afterEach(async () => {
  await Promise.all(runtimes.splice(0).map((runtime) => runtime.close()));
});

describe("node enrollment API", () => {
  it("requires the owning account, confirmation and one atomic redemption", async () => {
    const auth = new SupabaseAuthService(
      "https://auth.example.test",
      "service-role",
      (async (input) => {
        const url = String(input);
        if (url.includes("/auth/v1/user")) {
          return Response.json({ id: "account-1", email: "owner@example.test" });
        }
        if (url.includes("/rest/v1/network_members")) return Response.json([{ role: "viewer" }]);
        return new Response(null, { status: 404 });
      }) as typeof fetch,
    );
    const runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 0,
      databasePath: ":memory:",
      requestTimeoutMs: 1_000,
      apiAccessEnabled: false,
      networkToken: "network-secret",
    }, { logger: false, supabaseAuthService: auth });
    runtimes.push(runtime);

    const created = await runtime.app.inject({
      method: "POST",
      url: "/v1/nodes/enrollments",
      headers: { authorization: "Bearer account-session" },
      payload: {
        schema: "mycellios-node-enrollment-create/1",
        accountId: "account-1",
        requestedBy: { kind: "account", id: "account-1", scopes: ["node:identity"] },
        expiresInSeconds: 120,
      },
    });
    expect(created.statusCode).toBe(201);
    const enrollment = created.json() as {
      enrollmentId: string;
      enrollmentToken: string;
      nonce: string;
    };
    const publicKey = generateKeyPairSync("ed25519").publicKey.export({ format: "der", type: "spki" }).toString("base64url");
    const redemption = {
      schema: "mycellios-node-enrollment-redeem/1",
      enrollmentToken: enrollment.enrollmentToken,
      identity: { kind: "device", id: "node-1" },
      publicKey: { algorithm: "ed25519", spki: publicKey },
      protocol: { min: 1, max: 1 },
      registrationDigest: `sha256:${"a".repeat(64)}`,
      nonce: enrollment.nonce,
    };
    const premature = await runtime.app.inject({ method: "POST", url: "/internal/v1/nodes/enrollments/redeem", payload: redemption });
    expect(premature.statusCode).toBe(401);
    expect(premature.json()).toMatchObject({ error: { code: "node_enrollment_unconfirmed" } });

    const confirmed = await runtime.app.inject({
      method: "POST",
      url: `/v1/nodes/enrollments/${enrollment.enrollmentId}/confirm`,
      headers: { authorization: "Bearer account-session" },
    });
    expect(confirmed.statusCode).toBe(204);
    const [first, second] = await Promise.all([
      runtime.app.inject({ method: "POST", url: "/internal/v1/nodes/enrollments/redeem", payload: redemption }),
      runtime.app.inject({ method: "POST", url: "/internal/v1/nodes/enrollments/redeem", payload: redemption }),
    ]);
    expect([first.statusCode, second.statusCode].sort()).toEqual([200, 409]);
    expect((first.statusCode === 200 ? first : second).json()).toMatchObject({
      accountId: "account-1",
      identity: { kind: "device", id: "node-1" },
    });
  });
});
