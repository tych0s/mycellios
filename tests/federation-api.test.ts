import { afterEach, describe, expect, it } from "vitest";
import {
  createCoordinator,
  type CoordinatorRuntime,
} from "../src/coordinator/server.js";

let runtime: CoordinatorRuntime | null = null;

afterEach(async () => {
  await runtime?.close();
  runtime = null;
});

describe("federation admin API", () => {
  it("is protected, persists controls and never returns provider secrets", async () => {
    runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 8787,
      databasePath: ":memory:",
      requestTimeoutMs: 5_000,
      modelAdminToken: "admin-secret",
      apiAccessEnabled: false,
      publicApiBaseUrl: "https://www.mycellios.com/v1",
      federation: {
        enabled: true,
        external-runtime-aInferenceUrl: "http://127.0.0.1:19337",
        external-runtime-aManagementUrl: "http://127.0.0.1:13131",
        aiHordeBaseUrl: "http://127.0.0.1:19999",
        aiHordeApiKey: "horde-secret",
        peer-runtimeBaseUrl: "http://127.0.0.1:18000/v1",
        chutesBaseUrl: "http://127.0.0.1:18001/v1",
        chutesApiKey: "chutes-secret",
        akashMlBaseUrl: "http://127.0.0.1:18002/v1",
        akashMlApiKey: "akash-secret",
      },
    });

    expect((await runtime.app.inject({
      method: "GET",
      url: "/public/v1/admin/federation",
    })).statusCode).toBe(401);
    const preflight = await runtime.app.inject({
      method: "OPTIONS",
      url: "/public/v1/admin/federation/settings",
      headers: {
        origin: "app://mycellios",
        "access-control-request-method": "PUT",
      },
    });
    expect(preflight.statusCode).toBe(204);
    expect(preflight.headers["access-control-allow-methods"]).toContain("PUT");

    const updated = await runtime.app.inject({
      method: "PUT",
      url: "/public/v1/admin/federation/settings",
      headers: { authorization: "Bearer admin-secret" },
      payload: {
        dailyBudgetUsd: 2,
        monthlyBudgetUsd: 20,
        autoscalingEnabled: false,
        maxRentals: 2,
      },
    });
    expect(updated.statusCode).toBe(200);
    expect(updated.json().settings).toEqual(expect.objectContaining({
      dailyBudgetUsd: 2,
      monthlyBudgetUsd: 20,
      maxRentals: 2,
    }));

    const network = await runtime.app.inject({
      method: "PUT",
      url: "/public/v1/admin/federation/networks/chutes",
      headers: { authorization: "Bearer admin-secret" },
      payload: {
        enabled: true,
        priority: 42,
        dailyBudgetUsd: 1,
        monthlyBudgetUsd: 10,
      },
    });
    expect(network.statusCode).toBe(200);
    expect(network.json().settings).toEqual(expect.objectContaining({
      id: "chutes",
      enabled: true,
      priority: 42,
    }));

    const admin = await runtime.app.inject({
      method: "GET",
      url: "/public/v1/admin/federation",
      headers: { authorization: "Bearer admin-secret" },
    });
    expect(admin.statusCode).toBe(200);
    const raw = admin.body;
    expect(raw).not.toContain("horde-secret");
    expect(raw).not.toContain("chutes-secret");
    expect(raw).not.toContain("akash-secret");
    expect(admin.json().networks.find(
      (entry: { id: string }) => entry.id === "chutes",
    )).toEqual(expect.objectContaining({ configured: true }));

    const snapshot = await runtime.app.inject({
      method: "GET",
      url: "/public/v1/snapshot",
    });
    expect(snapshot.statusCode).toBe(200);
    expect(snapshot.json()).toEqual(expect.objectContaining({
      federation: expect.any(Object),
      federatedNetworks: expect.any(Array),
      federatedNodes: expect.any(Array),
      workers: [],
    }));
  });

  it("requires explicit confirmation for emergency stop", async () => {
    runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 8787,
      databasePath: ":memory:",
      requestTimeoutMs: 5_000,
      modelAdminToken: "admin-secret",
      apiAccessEnabled: false,
      federation: {
        enabled: true,
        external-runtime-aInferenceUrl: "http://127.0.0.1:19337",
        external-runtime-aManagementUrl: "http://127.0.0.1:13131",
        aiHordeBaseUrl: "http://127.0.0.1:19999",
        peer-runtimeBaseUrl: "http://127.0.0.1:18000/v1",
        chutesBaseUrl: "http://127.0.0.1:18001/v1",
        akashMlBaseUrl: "http://127.0.0.1:18002/v1",
      },
    });
    const rejected = await runtime.app.inject({
      method: "POST",
      url: "/public/v1/admin/federation/emergency-stop",
      headers: { authorization: "Bearer admin-secret" },
      payload: { confirm: false },
    });
    expect(rejected.statusCode).toBe(400);

    const stopped = await runtime.app.inject({
      method: "POST",
      url: "/public/v1/admin/federation/emergency-stop",
      headers: { authorization: "Bearer admin-secret" },
      payload: { confirm: true },
    });
    expect(stopped.statusCode).toBe(200);
    expect(stopped.json().settings.enabled).toBe(false);
  });
});
