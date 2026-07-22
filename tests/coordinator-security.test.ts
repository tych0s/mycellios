import { createHash } from "node:crypto";
import { afterEach, describe, expect, it } from "vitest";
import { createCoordinator, type CoordinatorRuntime } from "../src/coordinator/server.js";

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

function validWorkerRegistration() {
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
