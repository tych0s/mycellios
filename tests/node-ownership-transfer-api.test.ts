import { afterEach, describe, expect, it } from "vitest";
import { createCoordinator, type CoordinatorRuntime } from "../src/coordinator/server.js";
import { SupabaseAuthService } from "../src/coordinator/supabase-auth.js";

const runtimes: CoordinatorRuntime[] = [];
afterEach(async () => Promise.all(runtimes.splice(0).map((runtime) => runtime.close())));

describe("node ownership transfer API", () => {
  it("requires recent AAL2 from both accounts and rotates inventory generation", async () => {
    const ownerToken = token("owner-a", 0);
    const targetToken = token("owner-b", 0);
    const staleOwnerToken = token("owner-a", -301);
    const auth = new SupabaseAuthService("https://accounts.example.test", "service-role", (async (input, init) => {
      if (String(input).includes("/auth/v1/user")) {
        const authorization = new Headers(init?.headers).get("authorization") ?? "";
        return Response.json({ id: authorization.includes(ownerToken) || authorization.includes(staleOwnerToken) ? "owner-a" : "owner-b" });
      }
      if (String(input).includes("/rest/v1/network_members")) return Response.json([{ role: "owner" }]);
      return new Response(null, { status: 404 });
    }) as typeof fetch);
    const runtime = await createCoordinator({ host: "127.0.0.1", port: 0, databasePath: ":memory:", requestTimeoutMs: 1_000, apiAccessEnabled: true }, { logger: false, supabaseAuthService: auth });
    runtimes.push(runtime);
    runtime.database.raw.prepare(`INSERT INTO node_ownership(identity_kind, identity_id, account_id, credential_fingerprint, status, generation, created_at, updated_at) VALUES ('device', 'node-transfer-api', 'owner-a', ?, 'active', 1, 1, 1)`).run(`sha256:${"a".repeat(64)}`);

    const payload = { targetAccountId: "owner-b", expectedGeneration: 1, confirmation: "node-transfer-api", expiresInSeconds: 900 };
    const stale = await runtime.app.inject({ method: "POST", url: "/v1/nodes/node-transfer-api/ownership-transfers", headers: { authorization: `Bearer ${staleOwnerToken}` }, payload });
    expect(stale.statusCode).toBe(403);
    const created = await runtime.app.inject({ method: "POST", url: "/v1/nodes/node-transfer-api/ownership-transfers", headers: { authorization: `Bearer ${ownerToken}` }, payload });
    expect(created.statusCode).toBe(201);
    const transfer = created.json<{ transferId: string; transferToken: string }>();

    const accepted = await runtime.app.inject({ method: "POST", url: `/v1/node-ownership-transfers/${transfer.transferId}/accept`, headers: { authorization: `Bearer ${targetToken}` }, payload: { nodeId: "node-transfer-api", transferToken: transfer.transferToken, confirmation: "node-transfer-api" } });
    expect(accepted.statusCode).toBe(200);
    expect(accepted.json()).toEqual({ state: "transferred", nodeId: "node-transfer-api", generation: 2 });
    expect((await runtime.app.inject({ method: "GET", url: "/v1/nodes", headers: { authorization: `Bearer ${ownerToken}` } })).json()).toMatchObject({ data: [] });
    expect((await runtime.app.inject({ method: "GET", url: "/v1/nodes", headers: { authorization: `Bearer ${targetToken}` } })).json()).toMatchObject({ data: [{ nodeId: "node-transfer-api", generation: 2 }] });
  });
});

function token(subject: string, offsetSeconds: number): string {
  return `header.${Buffer.from(JSON.stringify({ sub: subject, aal: "aal2", auth_time: Math.floor(Date.now() / 1_000) + offsetSeconds })).toString("base64url")}.signature`;
}
