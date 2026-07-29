import { afterEach, describe, expect, it } from "vitest";
import { FleetContributionController } from "../src/coordinator/fleet-contribution-control.js";
import { createCoordinator, type CoordinatorRuntime } from "../src/coordinator/server.js";
import { workerConfigSchema } from "../src/contracts/schemas.js";
import type { WorkerCapabilities } from "../src/contracts/types.js";
import type { StoredWorker } from "../src/storage/store.js";
import { WorkerAgent } from "../src/worker/agent.js";

function capabilities(controlReady: boolean): WorkerCapabilities {
  return {
    region: "test",
    agentVersion: controlReady ? "0.2.54" : "0.2.53",
    ...(controlReady
      ? {
          administration: {
            contributionControl: "mycellios-contribution-control/1",
          },
        }
      : {}),
    gpus: [{
      id: "gpu-0",
      vendor: "test",
      model: "test",
      physicalVramMb: 1_024,
      offeredVramMb: 512,
      freeOfferedVramMb: 512,
    }],
    limits: { maxConcurrency: 1, pauseWhenForeground: false },
    deployments: [],
    network: { coordinatorRttMs: 1, uplinkMbps: 100, downlinkMbps: 100 },
  };
}

function worker(
  id: string,
  status: StoredWorker["status"],
  controlReady = true,
): StoredWorker {
  return {
    id,
    status,
    capabilities: capabilities(controlReady),
    reliability: 1,
    jobsCompleted: 0,
    lastSeenAt: Date.now(),
    identityKind: "device",
    identityId: id,
  };
}

describe("fleet contribution controller", () => {
  it("separates confirmed desktop states from legacy and offline nodes", () => {
    const workers = [
      worker("active", "online"),
      worker("paused", "draining"),
      worker("legacy", "online", false),
      worker("offline", "offline"),
      { ...worker("cell", "online"), identityKind: "cell" as const },
    ];
    const connected = new Set(["active", "paused", "legacy", "cell"]);
    const controller = new FleetContributionController(
      { listWorkers: () => workers },
      { isConnected: (id) => connected.has(id), send: () => false },
    );

    expect(controller.status().summary).toEqual({
      desktopNodes: 4,
      connected: 3,
      controlReady: 2,
      contributing: 1,
      paused: 1,
      unsupported: 1,
      offline: 1,
    });
  });

  it("targets only compatible connected desktops and waits for their acknowledgement", async () => {
    const workers = [
      worker("active", "online"),
      worker("paused", "draining"),
      worker("legacy", "online", false),
    ];
    const connected = new Set(workers.map(({ id }) => id));
    let controller: FleetContributionController;
    const sent: string[] = [];
    controller = new FleetContributionController(
      { listWorkers: () => workers },
      {
        isConnected: (id) => connected.has(id),
        send: (workerId, type, payload) => {
          sent.push(`${workerId}:${type}`);
          const command = payload as { commandId: string; enabled: boolean };
          const target = workers.find(({ id }) => id === workerId)!;
          target.status = "draining";
          queueMicrotask(() => {
            controller.acknowledge(workerId, {
              commandId: command.commandId,
              enabled: command.enabled,
              changed: workerId === "active",
              applied: true,
            });
          });
          return true;
        },
      },
    );

    const response = await controller.setAll(false);

    expect(sent).toEqual([
      "active:contribution.set",
      "paused:contribution.set",
    ]);
    expect(response.results).toEqual([
      { workerId: "active", state: "applied" },
      { workerId: "paused", state: "unchanged" },
    ]);
    expect(response.status.summary.paused).toBe(2);
  });

  it("reports a missing acknowledgement instead of assuming success", async () => {
    const target = worker("silent", "online");
    const controller = new FleetContributionController(
      { listWorkers: () => [target] },
      {
        isConnected: () => true,
        send: () => true,
      },
      10,
    );

    const response = await controller.setAll(false);

    expect(response.results).toEqual([{ workerId: "silent", state: "timeout" }]);
    expect(response.status.summary.contributing).toBe(1);
  });
});

describe("fleet contribution administration API", () => {
  let runtime: CoordinatorRuntime | null = null;
  let agent: WorkerAgent | null = null;
  let agentRun: Promise<void> | null = null;

  afterEach(async () => {
    await agent?.stop();
    await agentRun;
    agent = null;
    agentRun = null;
    await runtime?.close();
    runtime = null;
  });

  it("requires administrator authorization for status and commands", async () => {
    runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 0,
      databasePath: ":memory:",
      requestTimeoutMs: 1_000,
      modelAdminToken: "model-admin-secret",
    });

    const anonymous = await runtime.app.inject({
      method: "GET",
      url: "/public/v1/admin/fleet-contribution",
    });
    const status = await runtime.app.inject({
      method: "GET",
      url: "/public/v1/admin/fleet-contribution",
      headers: { authorization: "Bearer model-admin-secret" },
    });
    const command = await runtime.app.inject({
      method: "POST",
      url: "/public/v1/admin/fleet-contribution",
      headers: { authorization: "Bearer model-admin-secret" },
      payload: { enabled: false },
    });

    expect(anonymous.statusCode).toBe(401);
    expect(status.statusCode).toBe(200);
    expect(status.json().summary.controlReady).toBe(0);
    expect(command.statusCode).toBe(200);
    expect(command.json()).toMatchObject({ enabled: false, results: [] });
  });

  it("pauses and reactivates a live compatible worker end to end", async () => {
    const remoteChanges: boolean[] = [];
    runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 0,
      databasePath: ":memory:",
      requestTimeoutMs: 5_000,
      modelAdminToken: "model-admin-secret",
      allowDevelopmentAdapters: true,
    });
    const address = await runtime.app.listen({ host: "127.0.0.1", port: 0 });
    agent = new WorkerAgent(
      workerConfigSchema.parse({
        region: "test",
        offeredVramMb: 512,
        limits: { maxConcurrency: 1, pauseWhenForeground: false },
        adapter: {
          kind: "mock",
          developmentOnly: true,
          model: "fleet-control-test",
          tokensPerSecond: 10,
          ttftMs: 1,
          failureRate: 0,
        },
        deployment: {
          modelDigest: "sha256:fleet-control-test",
          contextLimit: 1_024,
        },
      }),
      {
        coordinatorUrl: address,
        reconnect: false,
        heartbeatIntervalMs: 50,
        contributionControl: {
          initialEnabled: true,
          onRemoteChange: (enabled) => {
            remoteChanges.push(enabled);
          },
        },
        hardwareProbe: async () => ({
          hostname: "fleet-control-worker",
          platform: process.platform,
          ramMb: 4_096,
          gpus: [],
        }),
        logger: { info() {}, warn() {}, error() {} },
      },
    );
    agentRun = agent.start();
    await waitUntil(
      () => runtime?.store.listWorkers()[0]?.status === "online",
      3_000,
    );

    const paused = await runtime.app.inject({
      method: "POST",
      url: "/public/v1/admin/fleet-contribution",
      headers: { authorization: "Bearer model-admin-secret" },
      payload: { enabled: false },
    });
    expect(paused.statusCode).toBe(200);
    expect(paused.json().results).toEqual([
      expect.objectContaining({ state: "applied" }),
    ]);
    expect(agent.isContributionEnabled).toBe(false);
    expect(paused.json().status.summary.paused).toBe(1);

    const activated = await runtime.app.inject({
      method: "POST",
      url: "/public/v1/admin/fleet-contribution",
      headers: { authorization: "Bearer model-admin-secret" },
      payload: { enabled: true },
    });
    expect(activated.statusCode).toBe(200);
    expect(activated.json().results).toEqual([
      expect.objectContaining({ state: "applied" }),
    ]);
    expect(agent.isContributionEnabled).toBe(true);
    expect(activated.json().status.summary.contributing).toBe(1);
    expect(remoteChanges).toEqual([false, true]);
  });
});

async function waitUntil(predicate: () => boolean, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error("Timed out waiting for worker state");
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}
