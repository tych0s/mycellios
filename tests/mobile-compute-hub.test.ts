import type { AddressInfo } from "node:net";
import { resolve } from "node:path";
import WebSocket from "ws";
import { afterEach, describe, expect, it } from "vitest";
import type { CoordinatorRuntime } from "../src/coordinator/server.js";
import { createCoordinator } from "../src/coordinator/server.js";

const runtimes: CoordinatorRuntime[] = [];

afterEach(async () => {
  await Promise.all(runtimes.splice(0).map((runtime) => runtime.close()));
});

describe("mobile compute hub", () => {
  it("dispatches real matrix work and independently verifies the result", async () => {
    const runtime = await createCoordinator(
      {
        host: "127.0.0.1",
        port: 0,
        databasePath: ":memory:",
        requestTimeoutMs: 10_000,
      },
      { logger: false },
    );
    runtimes.push(runtime);
    await runtime.app.listen({ host: "127.0.0.1", port: 0 });
    const address = runtime.app.server.address() as AddressInfo;

    const registration = await runtime.app.inject({
      method: "POST",
      url: "/mobile/v1/register",
      payload: registrationPayload(),
    });
    expect(registration.statusCode).toBe(201);
    const credentials = registration.json<{ workerId: string; token: string }>();
    const socket = new WebSocket(
      `ws://127.0.0.1:${address.port}/mobile/v1/connect?workerId=${credentials.workerId}&token=${credentials.token}`,
    );

    const ready = await nextMessage(socket);
    expect(ready.type).toBe("server.ready");
    socket.send(JSON.stringify({ v: 1, type: "work.request", payload: {} }));
    const offered = await nextMessage(socket);
    expect(offered.type).toBe("compute.offer");
    const task = offered.payload as {
      taskId: string;
      leaseId: string;
      size: number;
      seed: number;
    };
    expect(task.size).toBe(64);
    socket.send(
      JSON.stringify({
        v: 1,
        type: "compute.accept",
        payload: { taskId: task.taskId, leaseId: task.leaseId },
      }),
    );
    socket.send(
      JSON.stringify({
        v: 1,
        type: "compute.result",
        payload: {
          taskId: task.taskId,
          leaseId: task.leaseId,
          backend: "cpu",
          durationMs: 12,
          estimatedGflops: 0.5,
          samples: independentSamples(task.size, task.seed),
        },
      }),
    );
    const verified = await nextMessage(socket);
    expect(verified.type).toBe("compute.verified");

    const workers = await runtime.app.inject({ method: "GET", url: "/mobile/v1/workers" });
    expect(workers.json<{ data: Array<{ verifiedTasks: number; status: string }> }>().data[0]).toMatchObject({
      verifiedTasks: 1,
      status: "online",
    });
    const dashboard = await runtime.app.inject({ method: "GET", url: "/internal/v1/workers" });
    expect(dashboard.json<{ data: Array<{ id: string; deployments: unknown[] }> }>().data).toContainEqual(
      expect.objectContaining({ id: credentials.workerId, deployments: [] }),
    );
    socket.close();
  });

  it("requires the optional invitation token and serves the built PWA", async () => {
    const runtime = await createCoordinator(
      {
        host: "127.0.0.1",
        port: 0,
        databasePath: ":memory:",
        requestTimeoutMs: 10_000,
        mobileJoinToken: "invite-test",
        mobileAssetsPath: resolve("tests/fixtures/mobile-assets"),
      },
      { logger: false },
    );
    runtimes.push(runtime);
    const rejected = await runtime.app.inject({
      method: "POST",
      url: "/mobile/v1/register",
      payload: registrationPayload(),
    });
    expect(rejected.statusCode).toBe(401);
    const accepted = await runtime.app.inject({
      method: "POST",
      url: "/mobile/v1/register",
      payload: { ...registrationPayload(), joinToken: "invite-test" },
    });
    expect(accepted.statusCode).toBe(201);

    const page = await runtime.app.inject({ method: "GET", url: "/mobile/" });
    expect(page.statusCode).toBe(200);
    expect(page.body).toContain("Aporta potencia a la red");
    const manifest = await runtime.app.inject({ method: "GET", url: "/mobile/manifest.webmanifest" });
    expect(manifest.statusCode).toBe(200);
    expect(manifest.json<{ short_name: string }>().short_name).toBe("mycellios");
  });
});

function registrationPayload() {
  return {
    clientId: "mobile-client-test",
    name: "Test phone",
    region: "es-mad",
    platform: "vitest",
    backend: "cpu",
    performanceLevel: "balanced",
    capabilities: {
      webgpu: false,
      wasm: true,
      hardwareConcurrency: 8,
      deviceMemoryGb: 8,
    },
    benchmark: { durationMs: 10, estimatedGflops: 0.4, matrixSize: 48 },
  };
}

function nextMessage(socket: WebSocket): Promise<{ type: string; payload: unknown }> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("Timed out waiting for WebSocket message")), 5_000);
    socket.once("message", (raw) => {
      clearTimeout(timer);
      resolve(JSON.parse(raw.toString()) as { type: string; payload: unknown });
    });
    socket.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
  });
}

function independentSamples(size: number, seed: number): number[] {
  const positions: Array<readonly [number, number]> = [
    [0, 0],
    [Math.floor(size / 2), Math.floor(size / 3)],
    [size - 1, size - 1],
    [Math.floor(size / 3), size - 2],
  ];
  return positions.map(([row, column]) => {
    let result = 0;
    for (let k = 0; k < size; k += 1) {
      const left = ((row * 3 + k * 5 + seed) % 31 - 15) / 16;
      const right = ((k * 7 + column * 11 + seed * 3) % 29 - 14) / 16;
      result += left * right;
    }
    return result;
  });
}
