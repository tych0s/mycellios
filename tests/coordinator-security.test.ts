import { createHash } from "node:crypto";
import { afterEach, describe, expect, it } from "vitest";
import { createCoordinator, type CoordinatorRuntime } from "../src/coordinator/server.js";
import { SupabaseAuthService } from "../src/coordinator/supabase-auth.js";

const runtimes: CoordinatorRuntime[] = [];

afterEach(async () => {
  await Promise.all(runtimes.splice(0).map((runtime) => runtime.close()));
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

  it("sends anti-iframe headers independently of the reverse proxy", async () => {
    const runtime = await coordinator();
    const response = await runtime.app.inject({ method: "GET", url: "/health" });
    expect(response.headers["x-frame-options"]).toBe("DENY");
    expect(response.headers["content-security-policy"]).toContain("frame-ancestors 'none'");
    expect(response.headers["x-content-type-options"]).toBe("nosniff");
  });

  it("protects the public API with account sessions or revocable API keys and supports CORS", async () => {
    const auth = new SupabaseAuthService(
      "https://accounts.example.test",
      "service-role",
      (async (input) => {
        const url = String(input);
        if (url.includes("/auth/v1/user")) {
          return Response.json({ id: "73b6d6c3-ec87-4f67-b98e-9ffca4fb5576", email: "user@example.test" });
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

async function coordinator(internalToken?: string, networkToken?: string): Promise<CoordinatorRuntime> {
  const runtime = await createCoordinator({
    host: "127.0.0.1",
    port: 0,
    databasePath: ":memory:",
    requestTimeoutMs: 1_000,
    ...(internalToken ? { internalToken } : {}),
    ...(networkToken ? { networkToken } : {}),
  }, { logger: false });
  runtimes.push(runtime);
  return runtime;
}

function validWorkerRegistration(): {
  identity: { kind: "device"; id: string };
  capabilities: {
    region: string;
    agentVersion: string;
    gpus: Array<{
      id: string;
      vendor: string;
      model: string;
      physicalVramMb: number;
      offeredVramMb: number;
      freeOfferedVramMb: number;
    }>;
    deployments: never[];
    limits: {
      maxConcurrency: number;
      maxTemperatureC: number;
      maxPowerW: number;
      pauseWhenForeground: boolean;
    };
    network: { coordinatorRttMs: number; uplinkMbps: number; downlinkMbps: number };
    distributedExecutor?: Record<string, unknown>;
  };
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
