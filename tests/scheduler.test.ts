import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { ChatCompletionRequest } from "../src/contracts/types.js";
import { Scheduler } from "../src/scheduler/scheduler.js";
import { MeshDatabase } from "../src/storage/database.js";
import { MeshStore } from "../src/storage/store.js";
import { addWorker } from "./helpers.js";

const request: ChatCompletionRequest = {
  model: "distributed-small",
  messages: [{ role: "user", content: "hola" }],
  max_tokens: 200,
  workload_class: "interactive",
  preferred_region: "es-mad",
  deadline_ms: 60_000,
};

describe("multi-objective scheduler", () => {
  let database: MeshDatabase;
  let store: MeshStore;
  let scheduler: Scheduler;

  beforeEach(() => {
    database = new MeshDatabase(":memory:");
    store = new MeshStore(database);
    scheduler = new Scheduler(store);
  });

  afterEach(() => database.close());

  it("selects an eligible regional replica and enforces the offered VRAM budget", () => {
    const tooSmall = addWorker(store, {
      id: "small",
      offeredVramMb: 4_096,
      peakVramMb: 3_600,
      tokensPerSecond: 100,
    });
    const eligible = addWorker(store, {
      id: "eligible",
      offeredVramMb: 4_096,
      peakVramMb: 3_000,
      tokensPerSecond: 20,
    });
    const route = scheduler.selectRoute(request, "session", {
      connectedWorkerIds: new Set([tooSmall.id, eligible.id]),
    });
    expect(route?.stages[0]?.workerId).toBe(eligible.id);
  });

  it("reuses a healthy session route", () => {
    const first = addWorker(store, { id: "first", tokensPerSecond: 10 });
    const second = addWorker(store, { id: "second", tokensPerSecond: 30 });
    const pinned = {
      routeClass: "replica" as const,
      model: request.model,
      region: first.capabilities.region,
      stages: [
        {
          workerId: first.id,
          deploymentId: first.capabilities.deployments[0]!.deploymentId,
          modelDigest: first.capabilities.deployments[0]!.modelDigest,
          stageIndex: 0,
          score: 1,
        },
      ],
      score: 1,
      affinityHit: false,
    };
    store.saveSession("session", request.model, pinned);
    const route = scheduler.selectRoute(request, "session", {
      connectedWorkerIds: new Set([first.id, second.id]),
    });
    expect(route?.stages[0]?.workerId).toBe(first.id);
    expect(route?.affinityHit).toBe(true);
  });

  it("reports a distributed cell as a pipeline instead of a whole-model replica", () => {
    const cell = addWorker(store, {
      id: "distributed-cell",
      internalPipeline: { stageCount: 2, boundaries: [0, 14, 28] },
    });
    const models = scheduler.listAvailableModels({ connectedWorkerIds: new Set([cell.id]) });
    expect(models).toEqual([{ id: "distributed-small", replicas: 0, pipelines: 1 }]);
    expect(scheduler.selectRoute(request, "cell-route")?.routeClass).toBe("replica");
  });

  it("withdraws an internal pipeline while one of its physical stages is disconnected", () => {
    const physical = addWorker(store, {
      id: "physical-stage",
      model: "unrelated-model",
      distributedExecutor: {
        protocol: "gdlp-worker-tunnel/2",
        nodeId: "desktop-stage",
        stageHost: "desktop-stage.relay",
        stagePort: 43110,
        runtime: "python-safetensors",
        computeMode: "gpu-only",
        cpuEligible: false,
      },
    });
    const cell = addWorker(store, {
      id: "dependent-cell",
      internalPipeline: { stageCount: 2, boundaries: [0, 14, 28] },
      execution: {
        deviceType: "gpu",
        backend: "cuda",
        deviceName: "Distributed GPU pipeline",
        precision: "float16",
        fallback: false,
        stages: [{
          nodeId: "desktop-stage",
          stageIndex: 0,
          layerStart: 0,
          layerEnd: 28,
          deviceType: "gpu",
          backend: "cuda",
          deviceName: "Physical GPU",
          precision: "float16",
          fallback: false,
        }],
      },
    });

    expect(scheduler.listAvailableModels({
      connectedWorkerIds: new Set([physical.id, cell.id]),
    }).find((model) => model.id === "distributed-small")).toEqual({
      id: "distributed-small",
      replicas: 0,
      pipelines: 1,
    });
    expect(scheduler.listAvailableModels({
      connectedWorkerIds: new Set([cell.id]),
    }).find((model) => model.id === "distributed-small")).toBeUndefined();
  });

  it("preplans disjoint standbys with the exact primary model revision", () => {
    const primary = addWorker(store, {
      id: "primary",
      modelDigest: "sha256:revision-a",
      tokensPerSecond: 50,
    });
    const incompatible = addWorker(store, {
      id: "incompatible",
      modelDigest: "sha256:revision-b",
      tokensPerSecond: 40,
    });
    const standbyOne = addWorker(store, {
      id: "standby-one",
      modelDigest: "sha256:revision-a",
      tokensPerSecond: 30,
    });
    const standbyTwo = addWorker(store, {
      id: "standby-two",
      modelDigest: "sha256:revision-a",
      tokensPerSecond: 20,
    });
    const workers = [primary, incompatible, standbyOne, standbyTwo];

    const plan = scheduler.selectRoutePlan(request, "route-plan-session", {
      connectedWorkerIds: new Set(workers.map((worker) => worker.id)),
      maxStandbyRoutes: 2,
    });

    expect(plan?.primary.stages[0]?.workerId).toBe(primary.id);
    expect(plan?.standbys.map((route) => route.stages[0]?.workerId)).toEqual([
      standbyOne.id,
      standbyTwo.id,
    ]);
    expect(plan?.standbys.every(
      (route) => route.stages[0]?.modelDigest === plan.primary.stages[0]?.modelDigest,
    )).toBe(true);
    const plannedWorkers = [plan!.primary, ...plan!.standbys]
      .flatMap((route) => route.stages.map((stage) => stage.workerId));
    expect(new Set(plannedWorkers).size).toBe(plannedWorkers.length);
  });

  it("constructs a complete regional pipeline without reusing a worker", () => {
    const workers = [0, 1, 2].map((index) =>
      addWorker(store, {
        id: `stage-${index}`,
        model: "large-model",
        mode: "pipeline",
        offeredVramMb: 12_288,
        peakVramMb: 7_000,
        stage: { index, total: 3, layerStart: index * 10, layerEnd: index * 10 + 9 },
      }),
    );
    const route = scheduler.selectRoute(
      { ...request, model: "large-model" },
      "pipeline-session",
      { connectedWorkerIds: new Set(workers.map((worker) => worker.id)), allowPipeline: true },
    );
    expect(route?.routeClass).toBe("pipeline");
    expect(route?.stages).toHaveLength(3);
    expect(new Set(route?.stages.map((stage) => stage.workerId)).size).toBe(3);
  });

  it("fails closed for a physical pipeline until every forward and return link is measured", () => {
    const observations = [
      measuredLink("node-0", "node-1", 14, 800),
      measuredLink("node-1", "node-2", 18, 700),
    ];
    scheduler = new Scheduler(store, {
      runtimeLinkObservations: () => observations,
      strictRuntimeLinks: true,
    });
    const workers = [0, 1, 2].map((index) =>
      addWorker(store, {
        id: `measured-stage-${index}`,
        model: "measured-model",
        mode: "pipeline",
        stage: { index, total: 3, layerStart: index * 10, layerEnd: index * 10 + 10 },
        distributedExecutor: {
          protocol: "gdlp-worker-tunnel/2",
          nodeId: `node-${index}`,
          stageHost: `node-${index}.relay`,
          stagePort: 43_100 + index,
          runtime: "python-safetensors",
          computeMode: "gpu-only",
          cpuEligible: false,
        },
      }),
    );
    const routeInput = { ...request, model: "measured-model" };
    const options = {
      connectedWorkerIds: new Set(workers.map((worker) => worker.id)),
      allowPipeline: true,
    };

    expect(scheduler.selectRoute(routeInput, "missing-return", options)).toBeNull();
    observations.push(measuredLink("node-2", "node-0", 16, 750));
    expect(scheduler.selectRoute(routeInput, "complete-cycle", options)?.stages).toHaveLength(3);
  });

  it("routes interactive and batch traffic with different latency objectives", () => {
    const lowTtft = addWorker(store, {
      id: "low-ttft",
      ttftMs: 100,
      tokensPerSecond: 10,
    });
    const highThroughput = addWorker(store, {
      id: "high-throughput",
      ttftMs: 5_000,
      tokensPerSecond: 100,
    });
    const connectedWorkerIds = new Set([lowTtft.id, highThroughput.id]);

    expect(scheduler.selectRoute(
      { ...request, workload_class: "interactive" },
      "interactive-objective",
      { connectedWorkerIds },
    )?.stages[0]?.workerId).toBe(lowTtft.id);
    expect(scheduler.selectRoute(
      { ...request, workload_class: "batch" },
      "throughput-objective",
      { connectedWorkerIds },
    )?.stages[0]?.workerId).toBe(highThroughput.id);
  });

  it("drops KV affinity when its route is saturated and a free route exists", () => {
    const pinned = addWorker(store, {
      id: "affinity-saturated",
      maxConcurrency: 4,
      tokensPerSecond: 50,
    });
    const available = addWorker(store, {
      id: "affinity-free",
      maxConcurrency: 4,
      tokensPerSecond: 20,
    });
    const pinnedRoute = scheduler.selectRoute(request, "saturation-session", {
      connectedWorkerIds: new Set([pinned.id]),
    })!;
    store.saveSession("saturation-session", request.model, pinnedRoute);
    for (let index = 0; index < 3; index += 1) {
      const job = store.createJob({
        id: `saturated-${index}`,
        sessionId: `saturated-session-${index}`,
        model: request.model,
        workloadClass: "interactive",
        deadlineAt: Date.now() + 60_000,
      });
      store.setJobRoute(job.id, pinnedRoute, `lease-${index}`);
    }

    const selected = scheduler.selectRoute(request, "saturation-session", {
      connectedWorkerIds: new Set([pinned.id, available.id]),
    });
    expect(selected?.stages[0]?.workerId).toBe(available.id);
    expect(selected?.affinityHit).toBe(false);
  });

  it("does not select stale or disconnected workers", () => {
    const worker = addWorker(store, { id: "stale" });
    const route = scheduler.selectRoute(request, "session", {
      connectedWorkerIds: new Set(),
      now: Date.now(),
    });
    expect(worker.status).toBe("online");
    expect(route).toBeNull();
  });

  it("rejects a request that cannot fit in the deployment context window", () => {
    const worker = addWorker(store, { id: "short-context", contextLimit: 128 });
    const route = scheduler.selectRoute(
      { ...request, max_tokens: 120, messages: [{ role: "user", content: "x".repeat(100) }] },
      "context-session",
      { connectedWorkerIds: new Set([worker.id]) },
    );
    expect(route).toBeNull();
  });
});

function measuredLink(
  fromNodeId: string,
  toNodeId: string,
  rttP95Ms: number,
  goodputMbpsP50: number,
) {
  return {
    fromNodeId,
    toNodeId,
    measuredAt: Date.now(),
    rttP50Ms: rttP95Ms * 0.75,
    rttP95Ms,
    goodputMbpsP50,
    successfulSamples: 7,
    failedSamples: 0,
    availability: 1,
  };
}
