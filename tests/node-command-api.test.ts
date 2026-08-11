import { randomUUID } from "node:crypto";
import { afterEach, describe, expect, it } from "vitest";
import type { WorkerCapabilities } from "../src/contracts/types.js";
import { createCoordinator, type CoordinatorRuntime } from "../src/coordinator/server.js";
import { SupabaseAuthService } from "../src/coordinator/supabase-auth.js";
import { workerRegistrationDigest } from "../src/core/worker-admission-digest.js";
import { generateWorkerAdmissionCredential, workerAdmissionSigner } from "../src/worker/admission-credential.js";

const runtimes: CoordinatorRuntime[] = [];
afterEach(async () => Promise.all(runtimes.splice(0).map((runtime) => runtime.close())));

describe("node command API", () => {
  it("binds account ownership, worker session, durable pull, result and event cursor", async () => {
    const accountId = "73b6d6c3-ec87-4f67-b98e-9ffca4fb5576";
    const auth = new SupabaseAuthService(
      "https://accounts.example.test",
      "service-role",
      (async (input) => String(input).includes("/auth/v1/user")
        ? Response.json({ id: accountId, email: "owner@example.test" })
        : String(input).includes("/rest/v1/network_members")
          ? Response.json([{ role: "owner" }])
          : new Response(null, { status: 404 })) as typeof fetch,
    );
    const runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 0,
      databasePath: ":memory:",
      requestTimeoutMs: 1_000,
      apiAccessEnabled: true,
      networkToken: "network-secret",
    }, { logger: false, supabaseAuthService: auth });
    runtimes.push(runtime);
    const accountHeaders = { authorization: "Bearer account-session" };
    const signer = workerAdmissionSigner(generateWorkerAdmissionCredential());
    const identity = { kind: "device" as const, id: "node-command-api" };
    const protocol = { min: 1 as const, max: 1 as const };
    const capabilities = workerCapabilities();
    const registrationDigest = workerRegistrationDigest({ identity, capabilities, protocol });

    const issuedResponse = await runtime.app.inject({
      method: "POST",
      url: "/v1/nodes/enrollments",
      headers: accountHeaders,
      payload: {
        schema: "mycellios-node-enrollment-create/1",
        accountId,
        requestedBy: { kind: "account", id: accountId, scopes: ["node:identity"] },
        expiresInSeconds: 120,
      },
    });
    expect(issuedResponse.statusCode).toBe(201);
    const issued = issuedResponse.json<{ enrollmentId: string; enrollmentToken: string; nonce: string }>();
    expect((await runtime.app.inject({ method: "POST", url: `/v1/nodes/enrollments/${issued.enrollmentId}/confirm`, headers: accountHeaders })).statusCode).toBe(204);

    const redeemed = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/nodes/enrollments/redeem",
      payload: {
        schema: "mycellios-node-enrollment-redeem/1",
        enrollmentToken: issued.enrollmentToken,
        identity,
        publicKey: signer.publicKey,
        protocol,
        registrationDigest,
        nonce: issued.nonce,
      },
    });
    expect(redeemed.statusCode).toBe(200);

    const challenge = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/workers/admission-challenge",
      headers: { authorization: "Bearer network-secret" },
      payload: { identity, publicKey: signer.publicKey, protocol, registrationDigest },
    });
    const challengeBody = challenge.json<{ challengeId: string; signingPayload: string }>();
    const registered = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/workers/register",
      headers: { authorization: "Bearer network-secret" },
      payload: {
        identity,
        capabilities,
        protocol,
        admission: {
          challengeId: challengeBody.challengeId,
          publicKey: signer.publicKey,
          protocol,
          registrationDigest,
          signature: signer.sign(challengeBody.signingPayload),
        },
      },
    });
    expect(registered.statusCode).toBe(201);
    const session = registered.json<{ workerSessionToken: string; nodeGeneration: number }>();
    expect(session.nodeGeneration).toBe(1);

    const now = Date.now();
    const commandId = randomUUID();
    const command = {
      schema: "mycellios-node-command/1",
      id: commandId,
      nodeId: identity.id,
      actor: { kind: "account", id: accountId, scopes: ["node:control"] },
      generation: 1,
      issuedAt: new Date(now).toISOString(),
      expiresAt: new Date(now + 60_000).toISOString(),
      nonce: `nonce_${randomUUID().replaceAll("-", "")}`,
      type: "pause",
      payload: { version: 1 },
    };
    const queued = await runtime.app.inject({ method: "POST", url: `/v1/nodes/${identity.id}/commands`, headers: accountHeaders, payload: command });
    expect(queued.statusCode).toBe(202);

    const workerHeaders = { authorization: `Bearer ${session.workerSessionToken}` };
    const crossNode = await runtime.app.inject({ method: "GET", url: "/internal/v1/nodes/other-node/commands?generation=1", headers: workerHeaders });
    expect(crossNode.statusCode).toBe(403);
    expect(crossNode.json()).toMatchObject({ error: { code: "worker_session_wrong_node" } });
    const pulled = await runtime.app.inject({ method: "GET", url: `/internal/v1/nodes/${identity.id}/commands?generation=1`, headers: workerHeaders });
    expect(pulled.statusCode).toBe(200);
    expect(pulled.json()).toMatchObject({ data: [{ state: "delivered", command: { id: commandId } }] });

    const result = {
      schema: "mycellios-node-command-result/1",
      id: randomUUID(),
      commandId,
      nodeId: identity.id,
      generation: 1,
      state: "applied",
      observedAt: new Date(now + 1_000).toISOString(),
      resultDigest: `sha256:${"a".repeat(64)}`,
      error: null,
    };
    expect((await runtime.app.inject({ method: "POST", url: `/internal/v1/nodes/${identity.id}/commands/results`, headers: workerHeaders, payload: result })).statusCode).toBe(200);
    expect((await runtime.app.inject({ method: "GET", url: `/internal/v1/nodes/${identity.id}/commands?generation=1`, headers: workerHeaders })).json()).toEqual({ object: "list", data: [] });

    const events = await runtime.app.inject({ method: "GET", url: `/v1/nodes/${identity.id}/events`, headers: accountHeaders });
    expect(events.statusCode).toBe(200);
    expect(events.json()).toMatchObject({ data: [
      { type: "command.queued", actor: { kind: "account" } },
      { type: "command.delivered" },
      { type: "command.applied", actor: { kind: "node" } },
    ] });

    const snapshot = {
      schema: "mycellios-node-snapshot/1", nodeId: identity.id, generation: 1,
      cursor: "evt_0_0000000000000000", observedAt: new Date(now + 2_000).toISOString(), state: "ready",
      contributionEnabled: true, draining: false, updateChannel: "stable",
      limits: { maxConcurrency: 1, maxCpuPercent: 80, maxRamMiB: 8192, maxVramMiB: 0, maxDiskMiB: 16384, maxTemperatureC: 85 },
      activeCommandIds: [], build: { version: "1.0.0", sourceRevision: "a".repeat(40) },
      runtime: { ready: true, abi: "mycellios-distribution-runtime/4", backend: "cuda" },
      diagnostics: {
        capturedAt: new Date(now + 2_000).toISOString(), configRevision: 3,
        capacity: { acceleratorCount: 1, primaryAccelerator: { name: "Test GPU", totalMemoryMiB: 24_576, offeredMemoryMiB: 20_480 } },
        resources: { cpuPercent: 21, ramMiB: 2_048, diskMiB: 4_096, temperatureC: 62, healthy: true, violation: null },
        incident: null,
      },
    };
    const reconciled = await runtime.app.inject({ method: "POST", url: `/internal/v1/nodes/${identity.id}/snapshot`, headers: workerHeaders, payload: snapshot });
    expect(reconciled.statusCode).toBe(200);
    expect(reconciled.json()).toMatchObject({
      desiredState: { contributionEnabled: false, generation: 1 },
      events: [{ type: "command.queued" }, { type: "command.delivered" }, { type: "command.applied" }],
    });
    const accountNodes = await runtime.app.inject({ method: "GET", url: "/v1/nodes", headers: accountHeaders });
    expect(accountNodes.statusCode).toBe(200);
    expect(accountNodes.json()).toMatchObject({ data: [{ nodeId: identity.id, generation: 1, status: "active",
      connected: false, observed: { state: "ready", contributionEnabled: true,
        diagnostics: { capacity: { acceleratorCount: 1, primaryAccelerator: { name: "Test GPU" } }, resources: { healthy: true }, incident: null } },
      desired: { contributionEnabled: false }, commands: [{ type: "pause", state: "applied" }] }] });
    const uninstallCommand = { ...command, id: randomUUID(), nonce: `nonce_${randomUUID().replaceAll("-", "")}`,
      actor: { kind: "account", id: accountId, scopes: ["node:identity"] }, type: "uninstall",
      payload: { version: 1, retain: { cache: true, logs: true, configuration: true, identity: true } } };
    const staleUninstall = await runtime.app.inject({ method: "POST", url: `/v1/nodes/${identity.id}/commands`, headers: accountHeaders, payload: uninstallCommand });
    expect(staleUninstall.statusCode).toBe(403);
    expect(staleUninstall.json()).toMatchObject({ error: { code: "recent_aal2_reauthentication_required" } });
    const aal2 = `header.${Buffer.from(JSON.stringify({ aal: "aal2", auth_time: Math.floor(Date.now() / 1_000) })).toString("base64url")}.signature`;
    expect((await runtime.app.inject({ method: "POST", url: `/v1/nodes/${identity.id}/commands`,
      headers: { authorization: `Bearer ${aal2}` }, payload: uninstallCommand })).statusCode).toBe(202);
    const wrongNodeSnapshot = await runtime.app.inject({ method: "POST", url: "/internal/v1/nodes/other-node/snapshot", headers: workerHeaders, payload: { ...snapshot, nodeId: "other-node" } });
    expect(wrongNodeSnapshot.statusCode).toBe(403);

    const replacement = workerAdmissionSigner(generateWorkerAdmissionCredential());
    const recoveryChallenge = await runtime.app.inject({
      method: "POST", url: `/v1/nodes/${identity.id}/credential-recovery-challenge`, headers: accountHeaders,
      payload: { identity, nextPublicKey: replacement.publicKey, protocol },
    });
    expect(recoveryChallenge.statusCode).toBe(201);
    const recoveryChallengeBody = recoveryChallenge.json<{ challengeId: string; signingPayload: string }>();
    const recovered = await runtime.app.inject({
      method: "POST", url: `/v1/nodes/${identity.id}/credential-recovery`, headers: accountHeaders,
      payload: { identity, proof: { challengeId: recoveryChallengeBody.challengeId, nextSignature: replacement.sign(recoveryChallengeBody.signingPayload) } },
    });
    expect(recovered.statusCode).toBe(200);
    expect(recovered.json()).toMatchObject({ generation: 2 });
    const staleSession = await runtime.app.inject({ method: "GET", url: `/internal/v1/nodes/${identity.id}/commands?generation=1`, headers: workerHeaders });
    expect(staleSession.statusCode).toBe(401);
    expect(staleSession.json()).toMatchObject({ error: { code: "invalid_worker_session_token" } });
    const identityEvents = await runtime.app.inject({ method: "GET", url: `/v1/nodes/${identity.id}/identity-events`, headers: accountHeaders });
    expect(identityEvents.statusCode).toBe(200);
    expect(identityEvents.json()).toMatchObject({ data: [
      { eventType: "identity.enrolled", generation: 1, actorKind: "account", previousEventDigest: null },
      { eventType: "credential.recovered", generation: 2, actorKind: "account" },
    ] });
    const recoveryFingerprint = recovered.json<{ credentialFingerprint: string }>().credentialFingerprint;
    const revocationPayload = { expectedFingerprint: recoveryFingerprint, reason: "owner removed a lost node",
      confirmation: identity.id };
    const withoutReauth = await runtime.app.inject({ method: "POST", url: `/v1/nodes/${identity.id}/revoke`, headers: accountHeaders, payload: revocationPayload });
    expect(withoutReauth.statusCode).toBe(403);
    expect(withoutReauth.json()).toMatchObject({ error: { code: "recent_aal2_reauthentication_required" } });
    const revoked = await runtime.app.inject({ method: "POST", url: `/v1/nodes/${identity.id}/revoke`,
      headers: { authorization: `Bearer ${aal2}` }, payload: revocationPayload });
    expect(revoked.statusCode).toBe(200);
    expect(revoked.json()).toMatchObject({ state: "revoked", identityId: identity.id, generation: 3, disconnected: 1, affectedLeases: 0 });
    const afterRevocation = await runtime.app.inject({ method: "GET", url: `/v1/nodes/${identity.id}/identity-events`, headers: accountHeaders });
    expect(afterRevocation.json()).toMatchObject({ data: [
      { eventType: "identity.enrolled", generation: 1 },
      { eventType: "credential.recovered", generation: 2 },
      { eventType: "credential.revoked", generation: 3, actorKind: "account", actorId: accountId },
    ] });
  });
});

function workerCapabilities(): WorkerCapabilities {
  return {
    region: "test",
    agentVersion: "test",
    gpus: [{ id: "gpu-0", vendor: "unknown", model: "CPU fallback", physicalVramMb: 0, offeredVramMb: 512, freeOfferedVramMb: 512 }],
    deployments: [],
    limits: { maxConcurrency: 1, maxTemperatureC: 80, maxPowerW: 100, pauseWhenForeground: false },
    network: { coordinatorRttMs: 1, uplinkMbps: 1, downlinkMbps: 1 },
  };
}
