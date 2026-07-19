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

  it("uses llmfit whole-model fit as a soft replica ranking signal", () => {
    const marginal = addWorker(store, {
      id: "marginal",
      tokensPerSecond: 20,
      llmfit: { fitLevel: "Marginal", bestQuant: "Q4_K_M" },
    });
    const perfect = addWorker(store, {
      id: "perfect",
      tokensPerSecond: 20,
      llmfit: { fitLevel: "Perfect", bestQuant: "Q8_0" },
    });
    const route = scheduler.selectRoute(request, "llmfit-ranking", {
      connectedWorkerIds: new Set([marginal.id, perfect.id]),
    });
    expect(route?.stages[0]?.workerId).toBe(perfect.id);
    expect(route!.score).toBeLessThan(
      scheduler.scoreWorker(marginal, marginal.capabilities.deployments[0]!, request),
    );
  });

  it("summarizes linked llmfit advice in the application model catalog", () => {
    const measured = addWorker(store, {
      id: "measured",
      llmfit: {
        fitLevel: "Good",
        bestQuant: "Q5_K_M",
        estimatedTokensPerSecond: 30,
        measuredTokensPerSecond: 24,
        memoryRequiredMb: 5_900,
      },
    });
    const perfect = addWorker(store, {
      id: "catalog-perfect",
      llmfit: {
        fitLevel: "Perfect",
        bestQuant: "Q8_0",
        estimatedTokensPerSecond: 40,
        memoryRequiredMb: 7_000,
      },
    });
    const models = scheduler.listAvailableModels({
      connectedWorkerIds: new Set([measured.id, perfect.id]),
    });
    expect(models[0]?.llmfit).toEqual({
      advisedReplicas: 2,
      bestFit: "Perfect",
      quantizations: ["Q5_K_M", "Q8_0"],
      maxEstimatedTokensPerSecond: 40,
      maxMeasuredTokensPerSecond: 24,
      minMemoryRequiredMb: 5_900,
    });
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
        ...(index === 0 ? { llmfit: { fitLevel: "Too Tight" } } : {}),
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
