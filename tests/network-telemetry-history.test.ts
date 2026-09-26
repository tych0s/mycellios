import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  createCoordinator,
  NETWORK_TELEMETRY_INTERVAL_MS,
  type CoordinatorRuntime,
} from "../src/coordinator/server.js";
import { MeshDatabase } from "../src/storage/database.js";
import { MeshStore, type StoredNetworkTelemetrySample } from "../src/storage/store.js";

describe("network telemetry history", () => {
  const directories: string[] = [];
  const runtimes: CoordinatorRuntime[] = [];

  afterEach(async () => {
    await Promise.all(runtimes.splice(0).map((runtime) => runtime.close()));
    for (const directory of directories.splice(0)) {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("persists ten-minute network samples across coordinator restarts", () => {
    const directory = mkdtempSync(join(tmpdir(), "mycellios-network-history-"));
    directories.push(directory);
    const path = join(directory, "mesh.db");
    let database = new MeshDatabase(path);
    let store = new MeshStore(database);
    const sample = telemetrySample({
      capturedAt: 1_800_000,
      connectedNodes: 4,
      onlineNodes: 3,
      offeredVramMb: 98_304,
      freeVramMb: 65_536,
    });
    store.recordNetworkTelemetrySample(sample);
    database.close();

    database = new MeshDatabase(path);
    store = new MeshStore(database);
    expect(store.listNetworkTelemetrySamples(0)).toEqual([sample]);
    database.close();
  });

  it("serves bounded public history and records a current sample", async () => {
    const runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 0,
      databasePath: ":memory:",
      requestTimeoutMs: 1_000,
    }, { logger: false });
    runtimes.push(runtime);
    const now = Date.now();
    runtime.store.recordNetworkTelemetrySample(telemetrySample({
      capturedAt: now - 8 * 24 * 60 * 60_000,
      connectedNodes: 99,
    }));
    runtime.store.recordNetworkTelemetrySample(telemetrySample({
      capturedAt: now - 60 * 60_000,
      connectedNodes: 2,
      offeredVramMb: 24_576,
    }));

    const response = await runtime.app.inject({
      method: "GET",
      url: "/public/v1/history?range=7d",
    });
    expect(response.statusCode).toBe(200);
    const history = response.json() as {
      intervalMinutes: number;
      retentionDays: number;
      range: string;
      samples: Array<{ capturedAt: string; connectedNodes: number }>;
    };
    expect(history).toMatchObject({
      intervalMinutes: NETWORK_TELEMETRY_INTERVAL_MS / 60_000,
      retentionDays: 90,
      range: "7d",
    });
    expect(history.samples.some((sample) => sample.connectedNodes === 99)).toBe(false);
    expect(history.samples.some((sample) => sample.connectedNodes === 2)).toBe(true);
    expect(history.samples.at(-1)?.capturedAt).toBe(
      new Date(Math.floor(now / NETWORK_TELEMETRY_INTERVAL_MS) * NETWORK_TELEMETRY_INTERVAL_MS)
        .toISOString(),
    );
  });

  it("counts completed work beyond the 100 jobs shown in a snapshot", async () => {
    const runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 0,
      databasePath: ":memory:",
      requestTimeoutMs: 1_000,
    }, { logger: false });
    runtimes.push(runtime);
    for (let index = 0; index < 101; index += 1) {
      const id = `completed-${index}`;
      runtime.store.createJob({
        id,
        sessionId: "history-test",
        model: "test-model",
        workloadClass: "interactive",
        deadlineAt: Date.now() + 60_000,
      });
      runtime.store.setJobStatus(id, "completed");
    }

    const response = await runtime.app.inject({ method: "GET", url: "/public/v1/snapshot" });
    expect(response.statusCode).toBe(200);
    expect(response.json()).toMatchObject({ summary: { completedJobs: 101 } });
    expect(runtime.store.listJobs(100)).toHaveLength(100);
  });
});

function telemetrySample(
  overrides: Partial<StoredNetworkTelemetrySample>,
): StoredNetworkTelemetrySample {
  return {
    capturedAt: 0,
    registeredNodes: 4,
    connectedNodes: 0,
    onlineNodes: 0,
    browserNodes: 0,
    activeModels: 0,
    modelReplicas: 0,
    modelPipelines: 0,
    offeredVramMb: 0,
    freeVramMb: 0,
    inflightJobs: 0,
    runningJobs: 0,
    completedJobs: 0,
    ...overrides,
  };
}
