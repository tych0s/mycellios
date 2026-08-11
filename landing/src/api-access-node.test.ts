import { afterEach, describe, expect, it, vi } from "vitest";
import { acceptNodeOwnershipTransfer, createNodeEnrollmentBundle, createNodeOwnershipTransfer, enqueueOwnedNodeCommand, loadOwnedNodes, revokeOwnedNode } from "./api-access";

afterEach(() => vi.unstubAllGlobals());

describe("account node control API", () => {
  it("loads only the authenticated account inventory", async () => {
    const request = vi.fn(async () => Response.json({ data: [{ nodeId: "node-1", generation: 3, status: "active", commands: [] }] }));
    vi.stubGlobal("fetch", request);
    await expect(loadOwnedNodes("account-token")).resolves.toMatchObject([{ nodeId: "node-1", generation: 3 }]);
    expect(request).toHaveBeenCalledWith("/v1/nodes", expect.objectContaining({ headers: expect.objectContaining({ authorization: "Bearer account-token" }) }));
  });

  it("builds a scoped, expiring command from the signed-in account", async () => {
    let body: Record<string, unknown> | null = null;
    vi.stubGlobal("fetch", vi.fn(async (_path, init?: RequestInit) => {
      body = JSON.parse(String(init?.body));
      return Response.json({ command: body, state: "queued" });
    }));
    await enqueueOwnedNodeCommand("account-token", "account-1", { nodeId: "node-1", generation: 3 }, {
      type: "set-limits", payload: { version: 1, maxConcurrency: 2, maxCpuPercent: 80, maxRamMiB: 8192,
        maxVramMiB: 6144, maxDiskMiB: 16384, maxTemperatureC: 85 },
    });
    expect(body).toMatchObject({ nodeId: "node-1", generation: 3, type: "set-limits",
      actor: { kind: "account", id: "account-1", scopes: ["node:limits"] } });
    expect(Date.parse(String(body!.expiresAt)) - Date.parse(String(body!.issuedAt))).toBe(300_000);
  });

  it("sends exact identity and credential CAS evidence for owner revocation", async () => {
    let body: Record<string, unknown> | null = null;
    const request = vi.fn(async (_path, init?: RequestInit) => {
      body = JSON.parse(String(init?.body));
      return Response.json({ state: "revoked", identityId: "node-1", generation: 4, disconnected: 1, affectedLeases: 2 });
    });
    vi.stubGlobal("fetch", request);
    await expect(revokeOwnedNode("aal2-token", {
      nodeId: "node-1", credentialFingerprint: `sha256:${"a".repeat(64)}`,
    }, "owner retired the machine", "node-1")).resolves.toMatchObject({ state: "revoked", generation: 4 });
    expect(request).toHaveBeenCalledWith("/v1/nodes/node-1/revoke", expect.objectContaining({
      method: "POST", headers: expect.objectContaining({ authorization: "Bearer aal2-token" }),
    }));
    expect(body).toEqual({ expectedFingerprint: `sha256:${"a".repeat(64)}`, reason: "owner retired the machine", confirmation: "node-1" });
  });

  it("surfaces the server security code when recent AAL2 is missing", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({ error: { code: "recent_aal2_reauthentication_required" } }, { status: 403 })));
    await expect(revokeOwnedNode("stale-token", {
      nodeId: "node-1", credentialFingerprint: `sha256:${"b".repeat(64)}`,
    }, "owner retired the machine", "node-1")).rejects.toThrow("recent_aal2_reauthentication_required");
  });

  it("sends generation CAS and exact confirmation for two-party ownership transfer", async () => {
    const request = vi.fn()
      .mockResolvedValueOnce(Response.json({ transferId: "11111111-1111-4111-8111-111111111111", transferToken: "t".repeat(43), nodeId: "node-1", targetAccountId: "account-2", expiresAt: "2030-01-01T00:00:00.000Z" }, { status: 201 }))
      .mockResolvedValueOnce(Response.json({ state: "transferred", nodeId: "node-1", generation: 4 }));
    vi.stubGlobal("fetch", request);
    await createNodeOwnershipTransfer("owner-aal2", { nodeId: "node-1", generation: 3 }, "account-2", "node-1");
    expect(JSON.parse(String(request.mock.calls[0]?.[1]?.body))).toEqual({ targetAccountId: "account-2", expectedGeneration: 3, confirmation: "node-1", expiresInSeconds: 900 });
    await acceptNodeOwnershipTransfer("target-aal2", { transferId: "11111111-1111-4111-8111-111111111111", transferToken: "t".repeat(43), nodeId: "node-1", confirmation: "node-1" });
    expect(request).toHaveBeenNthCalledWith(2, "/v1/node-ownership-transfers/11111111-1111-4111-8111-111111111111/accept", expect.objectContaining({ headers: expect.objectContaining({ authorization: "Bearer target-aal2" }) }));
  });

  it("issues and confirms a one-time enrollment before returning the pairing bundle", async () => {
    const request = vi.fn()
      .mockResolvedValueOnce(Response.json({ enrollmentId: "4b5e61db-9e21-4268-b1a3-50dd0e818660",
        enrollmentToken: "t".repeat(43), nonce: "n".repeat(32), expiresAt: "2030-08-10T12:15:00.000Z" }, { status: 201 }))
      .mockResolvedValueOnce(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", request);
    await expect(createNodeEnrollmentBundle("account-token", "account-1", "https://coordinator.example/network?view=contribute"))
      .resolves.toMatchObject({ schema: "mycellios-node-enrollment-bundle/1", coordinatorUrl: "https://coordinator.example",
        enrollmentToken: "t".repeat(43) });
    expect(request).toHaveBeenNthCalledWith(1, "/v1/nodes/enrollments", expect.objectContaining({
      method: "POST", body: expect.stringContaining('"expiresInSeconds":900'),
    }));
    expect(request).toHaveBeenNthCalledWith(2, "/v1/nodes/enrollments/4b5e61db-9e21-4268-b1a3-50dd0e818660/confirm",
      expect.objectContaining({ method: "POST" }));
  });
});
