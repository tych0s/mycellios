import { createHash } from "node:crypto";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { WorkerCapabilities } from "../src/contracts/types.js";
import { createCoordinator, type CoordinatorRuntime } from "../src/coordinator/server.js";
import { SupabaseAuthService } from "../src/coordinator/supabase-auth.js";
import { workerRegistrationDigest } from "../src/core/worker-admission-digest.js";
import {
  generateWorkerAdmissionCredential,
  workerAdmissionSigner,
} from "../src/worker/admission-credential.js";

const runtimes: CoordinatorRuntime[] = [];

afterEach(async () => {
  await Promise.all(runtimes.splice(0).map((runtime) => runtime.close()));
  vi.unstubAllGlobals();
});

describe("public coordinator security boundaries", () => {
  it("blocks anonymous internal expert uploads while worker ingress stays public", async () => {
    const runtime = await coordinator();
    const body = Buffer.alloc(12, 1);
    const digest = createHash("sha256").update(body).digest("hex");

    const upload = await runtime.app.inject({
      method: "PUT",
      url: `/internal/v1/mobile/experts/weights/${digest}`,
      headers: { "content-type": "application/octet-stream" },
      payload: body,
    });
    expect(upload.statusCode).toBe(503);
    expect(upload.json()).toMatchObject({
      error: { code: "internal_administration_not_configured" },
    });

    const registration = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/workers/register",
      payload: validWorkerRegistration(),
    });
    expect(registration.statusCode).toBe(201);
  });

  it("requires the dedicated internal bearer token before parsing expert data", async () => {
    const runtime = await coordinator("internal-secret");
    const body = Buffer.alloc(12, 1);
    const digest = createHash("sha256").update(body).digest("hex");
    const missing = await runtime.app.inject({
      method: "PUT",
      url: `/internal/v1/mobile/experts/weights/${digest}`,
      headers: { "content-type": "application/octet-stream" },
      payload: body,
    });
    expect(missing.statusCode).toBe(401);

    const accepted = await runtime.app.inject({
      method: "PUT",
      url: `/internal/v1/mobile/experts/weights/${digest}`,
      headers: {
        authorization: "Bearer internal-secret",
        "content-type": "application/octet-stream",
      },
      payload: body,
    });
    expect(accepted.statusCode).toBe(201);
  });

  it("keeps expert administration usable when network and internal tokens differ", async () => {
    const runtime = await coordinator("internal-secret", "network-secret");
    const body = Buffer.alloc(12, 1);
    const digest = createHash("sha256").update(body).digest("hex");

    const upload = await runtime.app.inject({
      method: "PUT",
      url: `/internal/v1/mobile/experts/weights/${digest}`,
      headers: {
        authorization: "Bearer internal-secret",
        "content-type": "application/octet-stream",
      },
      payload: body,
    });
    expect(upload.statusCode).toBe(201);

    const wrongWorkerToken = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/workers/register",
      headers: { authorization: "Bearer internal-secret" },
      payload: validWorkerRegistration(),
    });
    expect(wrongWorkerToken.statusCode).toBe(401);

    const worker = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/workers/register",
      headers: { authorization: "Bearer network-secret" },
      payload: validWorkerRegistration(),
    });
    expect(worker.statusCode).toBe(201);
  });

  it("requires and verifies one-shot signed admission for remote workers", async () => {
    const runtime = await coordinator(undefined, "network-secret");
    const registration = validWorkerRegistration();
    const protocol = { min: 1, max: 1 };
    const unsigned = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/workers/register",
      headers: { authorization: "Bearer network-secret" },
      remoteAddress: "203.0.113.44",
      payload: { ...registration, protocol },
    });
    expect(unsigned.statusCode).toBe(401);
    expect(unsigned.json()).toMatchObject({
      error: { code: "worker_admission_required" },
    });

    const signer = workerAdmissionSigner(generateWorkerAdmissionCredential());
    const registrationDigest = workerRegistrationDigest({
      identity: registration.identity!,
      capabilities: registration.capabilities,
      protocol,
    });
    const challenge = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/workers/admission-challenge",
      headers: { authorization: "Bearer network-secret" },
      remoteAddress: "203.0.113.44",
      payload: {
        identity: registration.identity,
        publicKey: signer.publicKey,
        protocol,
        registrationDigest,
      },
    });
    expect(challenge.statusCode).toBe(200);
    const issued = challenge.json() as {
      challengeId: string;
      signingPayload: string;
    };
    const proof = {
      challengeId: issued.challengeId,
      publicKey: signer.publicKey,
      protocol,
      registrationDigest,
      signature: signer.sign(issued.signingPayload),
    };
    const accepted = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/workers/register",
      headers: { authorization: "Bearer network-secret" },
      remoteAddress: "203.0.113.44",
      payload: { ...registration, protocol, admission: proof },
    });
    expect(accepted.statusCode).toBe(201);
    expect(accepted.json()).toMatchObject({
      protocolVersion: 1,
      enrollment: "enrolled",
      credentialFingerprint: expect.stringMatching(/^sha256:[a-f0-9]{64}$/),
    });

    const replay = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/workers/register",
      headers: { authorization: "Bearer network-secret" },
      remoteAddress: "203.0.113.44",
      payload: { ...registration, protocol, admission: proof },
    });
    expect(replay.statusCode).toBe(401);
    expect(replay.json()).toMatchObject({
      error: { code: "worker_admission_challenge_unknown" },
    });

    const credentialFingerprint = accepted.json<{
      credentialFingerprint: string;
    }>().credentialFingerprint;
    const credentials = await runtime.app.inject({
      method: "GET",
      url: "/public/v1/worker-credentials",
    });
    expect(credentials.statusCode).toBe(200);
    expect(credentials.json()).toMatchObject({
      data: [expect.objectContaining({
        identityKind: "device",
        identityId: registration.identity.id,
        fingerprint: credentialFingerprint,
        status: "active",
      })],
    });
    const revoked = await runtime.app.inject({
      method: "POST",
      url: `/public/v1/worker-credentials/device/${registration.identity.id}/revoke`,
      payload: {
        expectedFingerprint: credentialFingerprint,
        reason: "security regression test",
      },
    });
    expect(revoked.statusCode).toBe(200);
    expect(revoked.json()).toMatchObject({
      state: "revoked",
      disconnected: 1,
      credential: {
        fingerprint: credentialFingerprint,
        status: "revoked",
        revocationReason: "security regression test",
      },
    });
  });

  it("rejects worker protocol downgrade or unsupported upgrade before enrollment", async () => {
    const runtime = await coordinator(undefined, "network-secret");
    const signer = workerAdmissionSigner(generateWorkerAdmissionCredential());
    const response = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/workers/admission-challenge",
      headers: { authorization: "Bearer network-secret" },
      remoteAddress: "203.0.113.45",
      payload: {
        identity: { kind: "device", id: "future-worker" },
        publicKey: signer.publicKey,
        protocol: { min: 2, max: 3 },
        registrationDigest: `sha256:${"a".repeat(64)}`,
      },
    });
    expect(response.statusCode).toBe(426);
    expect(response.json()).toMatchObject({
      error: { code: "worker_protocol_incompatible" },
    });
  });

  it("rejects anonymous remote worker administration before revealing worker state", async () => {
    const runtime = await coordinator(undefined, undefined, "admin-secret");

    const remove = await runtime.app.inject({
      method: "DELETE",
      url: "/public/v1/workers/non-existent",
      remoteAddress: "203.0.113.10",
    });
    expect(remove.statusCode).toBe(401);
    expect(remove.json()).toMatchObject({
      error: { code: "invalid_model_admin_token" },
    });

    const clear = await runtime.app.inject({
      method: "POST",
      url: "/public/v1/workers/clear-offline",
      remoteAddress: "203.0.113.10",
    });
    expect(clear.statusCode).toBe(401);
  });

  it("accepts the administrative bearer token for remote worker cleanup", async () => {
    const runtime = await coordinator(undefined, undefined, "admin-secret");
    const registration = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/workers/register",
      payload: validWorkerRegistration(),
    });
    expect(registration.statusCode).toBe(201);

    const clear = await runtime.app.inject({
      method: "POST",
      url: "/public/v1/workers/clear-offline",
      headers: { authorization: "Bearer admin-secret" },
      remoteAddress: "203.0.113.10",
    });
    expect(clear.statusCode).toBe(200);
    expect(clear.json()).toEqual({ removed: 1 });
  });

  it("keeps loopback worker administration available when no remote admin is configured", async () => {
    const runtime = await coordinator();
    const registration = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/workers/register",
      payload: validWorkerRegistration(),
    });
    expect(registration.statusCode).toBe(201);

    const clear = await runtime.app.inject({
      method: "POST",
      url: "/public/v1/workers/clear-offline",
    });
    expect(clear.statusCode).toBe(200);
    expect(clear.json()).toEqual({ removed: 1 });
  });

  it("fails closed for remote worker administration when no authority is configured", async () => {
    const runtime = await coordinator();
    const response = await runtime.app.inject({
      method: "POST",
      url: "/public/v1/workers/clear-offline",
      remoteAddress: "203.0.113.10",
    });
    expect(response.statusCode).toBe(503);
    expect(response.json()).toMatchObject({
      error: { code: "model_administration_not_configured" },
    });
  });

  it("bounds public Hugging Face catalog amplification per remote client", async () => {
    const fetchMock = vi.fn<typeof fetch>(async () => Response.json([]));
    vi.stubGlobal("fetch", fetchMock);
    const runtime = await coordinator();
    const request = {
      method: "GET" as const,
      url: "/public/v1/huggingface-models?q=qwen&limit=10",
      headers: { "user-agent": "catalog-rate-test" },
      remoteAddress: "203.0.113.20",
    };

    for (let index = 0; index < 120; index += 1) {
      const response = await runtime.app.inject({
        ...request,
        headers: {
          ...request.headers,
          "x-forwarded-for": `198.51.100.${(index % 200) + 1}`,
        },
      });
      expect(response.statusCode).toBe(200);
    }
    const limited = await runtime.app.inject({
      ...request,
      headers: {
        ...request.headers,
        "x-forwarded-for": "198.51.100.250",
      },
    });
    expect(limited.statusCode).toBe(429);
    expect(limited.json()).toMatchObject({
      error: { code: "catalog_rate_limited" },
    });
    expect(fetchMock).toHaveBeenCalledTimes(120);
  });

  it("refuses anonymous worker eviction from the public network", async () => {
    // Regression: `DELETE /public/v1/workers/:id` and `clear-offline` carried no
    // guard at all, and the network-token hook only covers `/internal/v1/` and
    // `/v1/`. Anyone who loaded the public panel could empty the whole network.
    const runtime = await coordinator(undefined, undefined, "admin-secret");
    const registration = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/workers/register",
      payload: validWorkerRegistration(),
    });
    expect(registration.statusCode).toBe(201);
    const workerId = registration.json().workerId;

    const anonymousDelete = await runtime.app.inject({
      method: "DELETE",
      url: `/public/v1/workers/${workerId}`,
    });
    expect(anonymousDelete.statusCode).toBe(401);

    const anonymousClear = await runtime.app.inject({
      method: "POST",
      url: "/public/v1/workers/clear-offline",
    });
    expect(anonymousClear.statusCode).toBe(401);

    // The worker must still be there: a rejected request may not have side effects.
    const snapshot = await runtime.app.inject({ method: "GET", url: "/public/v1/snapshot" });
    expect(snapshot.json().workers).toHaveLength(1);

    const authorized = await runtime.app.inject({
      method: "DELETE",
      url: `/public/v1/workers/${workerId}`,
      headers: { authorization: "Bearer admin-secret" },
    });
    expect(authorized.statusCode).toBe(200);
    expect(authorized.json()).toMatchObject({ removed: true });
  });

  it("publishes the executor isolation truth for the node inspector", async () => {
    const runtime = await coordinator();
    const payload = validWorkerRegistration();
    payload.capabilities.distributedExecutor = {
      protocol: "gdlp-worker-tunnel/2",
      nodeId: "public-worker-node",
      stageHost: "public-worker-node.relay",
      stagePort: 9_850,
      runtime: "python-safetensors",
      isolation: {
        schema: "mycellios-executor-isolation-capability/1",
        launchPolicySchema: "gdlp-executor-isolation/4",
        environment: "filtered",
        workspace: "private-temp-watchdog",
        processTree: "best-effort",
        resourceLimits: "workspace-watchdog-only",
        osSandbox: "not-enforced",
        hardResourceQuotas: "not-enforced",
        killOnClose: "not-enforced",
        maxWorkspaceBytes: 512 * 1024 * 1024,
        maxWorkspaceEntries: 10_000,
        workspaceCheckIntervalMs: 1_000,
      },
    };

    const registration = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/workers/register",
      payload,
    });
    expect(registration.statusCode).toBe(201);

    const snapshot = await runtime.app.inject({ method: "GET", url: "/public/v1/snapshot" });
    expect(snapshot.statusCode).toBe(200);
    expect(snapshot.json().workers[0].isolation).toEqual(
      payload.capabilities.distributedExecutor.isolation,
    );
  });

  it("does not treat a proxied request as local just because the socket is loopback", async () => {
    // Regression: Fastify runs without `trustProxy`, so a reverse proxy
    // terminating on 127.0.0.1 makes every remote request look like loopback.
    // The `!expected && isLoopbackAddress(request.ip)` shortcut therefore handed
    // model administration to the whole internet.
    const runtime = await coordinator();

    const direct = await runtime.app.inject({
      method: "POST",
      url: "/public/v1/requested-models",
      payload: { source: "acme/model", contextTokens: 4_096 },
    });
    expect(direct.statusCode).not.toBe(503);

    for (const header of ["x-forwarded-for", "x-real-ip", "forwarded", "x-forwarded-host"]) {
      const proxied = await runtime.app.inject({
        method: "POST",
        url: "/public/v1/requested-models",
        headers: { [header]: "203.0.113.9" },
        payload: { source: "acme/model", contextTokens: 4_096 },
      });
      expect(proxied.statusCode, `header ${header} must not grant local trust`).toBe(503);
      expect(proxied.json()).toMatchObject({
        error: { code: "model_administration_not_configured" },
      });
    }
  });

  it("keeps host-only benchmark routes closed behind a reverse proxy", async () => {
    const runtime = await coordinator();
    const local = await runtime.app.inject({ method: "GET", url: "/local/v1/benchmarks" });
    expect(local.statusCode).toBe(200);

    const proxied = await runtime.app.inject({
      method: "GET",
      url: "/local/v1/benchmarks",
      headers: { "x-forwarded-for": "203.0.113.9" },
    });
    expect(proxied.statusCode).toBe(403);
    expect(proxied.json()).toMatchObject({ error: { code: "local_access_required" } });
  });

  it("sends anti-iframe headers independently of the reverse proxy", async () => {
    const runtime = await coordinator();
    const response = await runtime.app.inject({ method: "GET", url: "/health" });
    expect(response.headers["x-frame-options"]).toBe("DENY");
    expect(response.headers["content-security-policy"]).toContain("frame-ancestors 'none'");
    expect(response.headers["x-content-type-options"]).toBe("nosniff");
  });

  it("rate limits every HTTP route at the coordinator boundary", async () => {
    const runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 0,
      databasePath: ":memory:",
      requestTimeoutMs: 1_000,
    }, { logger: false, globalRateLimitMax: 2 });
    runtimes.push(runtime);

    expect((await runtime.app.inject({ method: "GET", url: "/health" })).statusCode).toBe(200);
    expect((await runtime.app.inject({ method: "GET", url: "/health" })).statusCode).toBe(200);
    const limited = await runtime.app.inject({ method: "GET", url: "/health" });
    expect(limited.statusCode).toBe(429);
    expect(limited.json()).toMatchObject({ statusCode: 429, error: "Too Many Requests" });
    expect(Number(limited.headers["retry-after"])).toBeGreaterThan(0);
  });

  it("publishes one exact coordinator build identity on health and public snapshot", async () => {
    const buildIdentity = {
      schema: "mycellios-native-build-provenance/1" as const,
      version: "0.2.19",
      sourceId: `sha256:${"a".repeat(64)}` as const,
    };
    const runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 0,
      databasePath: ":memory:",
      requestTimeoutMs: 1_000,
    }, {
      logger: false,
      buildIdentity,
      runtimeMetadata: {
        root: process.cwd(),
        version: buildIdentity.version,
        revision: "b".repeat(40),
        buildIdentity,
      },
    });
    runtimes.push(runtime);

    const health = await runtime.app.inject({ method: "GET", url: "/health" });
    const snapshot = await runtime.app.inject({ method: "GET", url: "/public/v1/snapshot" });

    expect(health.statusCode).toBe(200);
    expect(snapshot.statusCode).toBe(200);
    expect(health.json().version).toBe(buildIdentity.version);
    expect(health.json().revision).toBe("b".repeat(40));
    expect(health.json().buildIdentity).toEqual(buildIdentity);
    expect(snapshot.json().version).toBe(buildIdentity.version);
    expect(snapshot.json().buildIdentity).toEqual(buildIdentity);
  });

  it("rejects development mock deployments on the production coordinator boundary", async () => {
    const runtime = await coordinator();
    const payload = validWorkerRegistration();
    (payload.capabilities.deployments as unknown as Array<Record<string, unknown>>).push({
      deploymentId: "dep-mock",
      model: "fixture",
      modelDigest: "sha256:fixture",
      mode: "replica",
      adapter: "mock",
      peakVramMb: 512,
      contextLimit: 4_096,
      maxConcurrency: 1,
      freeSlots: 1,
      tokensPerSecond: 20,
      ttftMs: 10,
      dataLocality: "local",
    });
    const response = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/workers/register",
      payload,
    });
    expect(response.statusCode).toBe(400);
    expect(response.json()).toMatchObject({
      error: { code: "development_adapter_not_allowed" },
    });
  });

  it("protects the public API with account sessions or revocable API keys and supports CORS", async () => {
    const auth = new SupabaseAuthService(
      "https://accounts.example.test",
      "service-role",
      (async (input) => {
        const url = String(input);
        if (url.includes("/auth/v1/user")) {
          return Response.json({
            id: "73b6d6c3-ec87-4f67-b98e-9ffca4fb5576",
            email: "user@example.test",
          });
        }
        if (url.includes("/rest/v1/network_members")) {
          return Response.json([{ role: "viewer" }]);
        }
        return new Response(null, { status: 404 });
      }) as typeof fetch,
    );
    const runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 0,
      databasePath: ":memory:",
      requestTimeoutMs: 1_000,
      apiAccessEnabled: true,
      apiStarterTokens: 100,
    }, { logger: false, supabaseAuthService: auth });
    runtimes.push(runtime);

    const preflight = await runtime.app.inject({
      method: "OPTIONS",
      url: "/v1/chat/completions",
    });
    expect(preflight.statusCode).toBe(204);
    expect(preflight.headers["access-control-allow-origin"]).toBe("*");

    const anonymous = await runtime.app.inject({ method: "GET", url: "/v1/models" });
    expect(anonymous.statusCode).toBe(401);
    expect(anonymous.json()).toMatchObject({ error: { code: "authentication_required" } });

    const account = await runtime.app.inject({
      method: "GET",
      url: "/v1/account",
      headers: { authorization: "Bearer account-session" },
    });
    expect(account.statusCode).toBe(200);
    expect(account.json()).toMatchObject({ token_balance: 100 });

    const created = await runtime.app.inject({
      method: "POST",
      url: "/v1/api-keys",
      headers: { authorization: "Bearer account-session" },
      payload: { name: "Test client" },
    });
    expect(created.statusCode).toBe(201);
    const key = created.json() as { id: string; secret: string };
    expect(key.secret).toMatch(/^myc_live_/);

    const models = await runtime.app.inject({
      method: "GET",
      url: "/v1/models",
      headers: { authorization: `Bearer ${key.secret}` },
    });
    expect(models.statusCode).toBe(200);
    expect(models.json()).toEqual({ object: "list", data: [] });

    const revoked = await runtime.app.inject({
      method: "DELETE",
      url: `/v1/api-keys/${key.id}`,
      headers: { authorization: "Bearer account-session" },
    });
    expect(revoked.statusCode).toBe(204);
    const rejected = await runtime.app.inject({
      method: "GET",
      url: "/v1/models",
      headers: { authorization: `Bearer ${key.secret}` },
    });
    expect(rejected.statusCode).toBe(401);
    expect(rejected.json()).toMatchObject({ error: { code: "invalid_api_key" } });
  });

  it("exposes bounded desktop repair diagnostics without requiring machine access", async () => {
    const runtime = await coordinator();
    const payload = validWorkerRegistration();
    payload.capabilities.distributedExecutor = {
      protocol: "gdlp-worker-tunnel/2",
      nodeId: "desktop-public-worker",
      stageHost: "desktop-public-worker.relay",
      stagePort: 9_850,
      runtime: "python-safetensors",
      computeMode: "automatic",
      cpuEligible: true,
      acceleration: {
        schema: "mycellios-accelerator-diagnostics/1",
        appVersion: "0.2.26",
        state: "gpu-fallback",
        backend: "cpu",
        deviceName: "NVIDIA GeForce RTX 2060",
        gpuVendor: "nvidia",
        gpuModel: "NVIDIA GeForce RTX 2060",
        phase: "physical-probe",
        progressPct: 95,
        issueCode: "physical-probe",
        issueSummary: "CUDA driver probe failed",
        retryable: true,
        retryAttempt: 3,
        nextRetryAt: "2026-07-23T10:30:00.000Z",
        updatedAt: "2026-07-23T10:00:00.000Z",
        recentEvents: [],
      },
    };
    const registration = await runtime.app.inject({
      method: "POST",
      url: "/internal/v1/workers/register",
      payload,
    });
    expect(registration.statusCode).toBe(201);

    const snapshot = await runtime.app.inject({ method: "GET", url: "/public/v1/snapshot" });
    expect(snapshot.statusCode).toBe(200);
    expect(snapshot.json().workers[0]).toMatchObject({
      agentVersion: "test",
      acceleration: {
        schema: "mycellios-accelerator-diagnostics/1",
        appVersion: "0.2.26",
        issueCode: "physical-probe",
        retryAttempt: 3,
      },
    });
  });
});

async function coordinator(
  internalToken?: string,
  networkToken?: string,
  modelAdminToken?: string,
): Promise<CoordinatorRuntime> {
  const runtime = await createCoordinator({
    host: "127.0.0.1",
    port: 0,
    databasePath: ":memory:",
    requestTimeoutMs: 1_000,
    ...(internalToken ? { internalToken } : {}),
    ...(networkToken ? { networkToken } : {}),
    ...(modelAdminToken ? { modelAdminToken } : {}),
  }, { logger: false });
  runtimes.push(runtime);
  return runtime;
}

function validWorkerRegistration(): {
  identity: { kind: "device"; id: string };
  capabilities: WorkerCapabilities;
} {
  return {
    identity: { kind: "device", id: "public-worker" },
    capabilities: {
      region: "test",
      agentVersion: "test",
      gpus: [{
        id: "gpu-0",
        vendor: "unknown",
        model: "CPU fallback",
        physicalVramMb: 0,
        offeredVramMb: 512,
        freeOfferedVramMb: 512,
      }],
      deployments: [],
      limits: {
        maxConcurrency: 1,
        maxTemperatureC: 80,
        maxPowerW: 100,
        pauseWhenForeground: false,
      },
      network: { coordinatorRttMs: 1, uplinkMbps: 1, downlinkMbps: 1 },
    },
  };
}
