import { describe, expect, it } from "vitest";
import { DeploymentControlPlane } from "../src/coordinator/deployment-control-plane.js";
import { MeshDatabase } from "../src/storage/database.js";
import { MeshStore } from "../src/storage/store.js";

describe("durable deployment control plane", () => {
  it("tracks desired and observed generations and claims operations idempotently", () => {
    const database = new MeshDatabase(":memory:");
    const store = new MeshStore(database);
    const model = addModel(store, "qwen");
    const controller = new DeploymentControlPlane(store, "controller-a");

    expect(controller.initialize(1_000)).toEqual({
      models: 1,
      interruptedOperations: 0,
      expiredReservations: 0,
    });
    expect(controller.getState(model.id)).toMatchObject({
      desiredState: "active",
      observedState: "inactive",
      generation: 1,
      observedGeneration: 0,
    });

    const operation = controller.claimOperation(model.id, "activate", {
      now: 2_000,
      leaseMs: 30_000,
    });
    const duplicate = controller.claimOperation(model.id, "activate", {
      now: 2_100,
      leaseMs: 30_000,
    });
    expect(operation).not.toBeNull();
    expect(duplicate?.id).toBe(operation?.id);
    expect(controller.listOperations(model.id)).toHaveLength(1);
    expect(controller.getState(model.id)).toMatchObject({
      observedState: "preparing",
      activeOperationId: operation?.id,
      controllerOwner: "controller-a",
    });

    expect(controller.completeOperation(operation!.id, "active", {}, 3_000)).toBe(true);
    expect(controller.getState(model.id)).toMatchObject({
      desiredState: "active",
      observedState: "active",
      observedGeneration: 1,
      retryCount: 0,
      activeOperationId: null,
    });

    expect(controller.setDesiredState(model.id, "inactive", 4_000)).toMatchObject({
      desiredState: "inactive",
      generation: 2,
      observedGeneration: 1,
    });
    database.close();
  });

  it("reserves stage capacity atomically and publishes only after a canary commit", () => {
    const database = new MeshDatabase(":memory:");
    const store = new MeshStore(database);
    addModel(store, "model-a");
    addModel(store, "model-b");
    const controller = new DeploymentControlPlane(store, "controller-a");
    controller.initialize(1_000);
    const first = controller.claimOperation("model-a", "activate", { now: 2_000 })!;
    const second = controller.claimOperation("model-b", "activate", { now: 2_000 })!;

    const reservation = controller.prepareRoute(first.id, [{
      nodeId: "node-a",
      stageIndex: 0,
      layerStart: 0,
      layerEnd: 12,
      memoryMiB: 700,
      capacityMiB: 1_000,
    }], 60_000, 2_100);
    expect(reservation.status).toBe("prepared");
    expect(controller.getState("model-a")?.observedState).toBe("preparing");

    expect(() => controller.prepareRoute(second.id, [{
      nodeId: "node-a",
      stageIndex: 0,
      layerStart: 0,
      layerEnd: 8,
      memoryMiB: 400,
      capacityMiB: 1_000,
    }], 60_000, 2_200)).toThrow("route_capacity_conflict:node-a");

    const committed = controller.commitRoute(reservation.id, {
      passed: true,
      ttftMs: 120,
      tokensPerSecond: 8.5,
    }, 2_300);
    expect(committed.status).toBe("committed");
    expect(controller.getState("model-a")).toMatchObject({
      observedState: "active",
      observedGeneration: 1,
      activeOperationId: null,
    });
    expect(controller.getOperation(first.id)?.status).toBe("succeeded");

    expect(controller.releaseRoutesForModel("model-a", 2_400)).toBe(1);
    const replacement = controller.prepareRoute(second.id, [{
      nodeId: "node-a",
      stageIndex: 0,
      layerStart: 0,
      layerEnd: 8,
      memoryMiB: 400,
      capacityMiB: 1_000,
    }], 60_000, 2_500);
    expect(replacement.status).toBe("prepared");
    database.close();
  });

  it("recovers interrupted operations and schedules active deployments for repair", () => {
    const database = new MeshDatabase(":memory:");
    const store = new MeshStore(database);
    addModel(store, "qwen");
    const firstController = new DeploymentControlPlane(store, "controller-before-crash");
    firstController.initialize(1_000);
    const operation = firstController.claimOperation("qwen", "activate", {
      now: 2_000,
      leaseMs: 60_000,
    })!;
    firstController.prepareRoute(operation.id, [{
      nodeId: "node-a",
      stageIndex: 0,
      layerStart: 0,
      layerEnd: 10,
      memoryMiB: 512,
      capacityMiB: 1_024,
    }], 1_000, 2_000);

    const recovered = new DeploymentControlPlane(store, "controller-after-crash");
    expect(recovered.initialize(4_000)).toEqual({
      models: 1,
      interruptedOperations: 1,
      expiredReservations: 1,
    });
    expect(recovered.getOperation(operation.id)).toMatchObject({
      status: "interrupted",
      errorCode: "coordinator_restarted",
    });
    expect(recovered.getState("qwen")).toMatchObject({
      desiredState: "active",
      observedState: "degraded",
      nextRetryAt: 4_000,
      activeOperationId: null,
    });
    expect(recovered.listDueStates(4_000).map((state) => state.modelId)).toEqual(["qwen"]);
    database.close();
  });

  it("persists retry scheduling rather than keeping it only in process memory", () => {
    const database = new MeshDatabase(":memory:");
    const store = new MeshStore(database);
    addModel(store, "qwen");
    const controller = new DeploymentControlPlane(store, "controller-a");
    controller.initialize(1_000);
    const operation = controller.claimOperation("qwen", "activate", { now: 2_000 })!;

    expect(controller.failOperation(
      operation.id,
      "worker_disconnected",
      "node-a disconnected",
      { retryAt: 12_000, now: 3_000 },
    )).toBe(true);
    expect(controller.getState("qwen")).toMatchObject({
      observedState: "failed",
      retryCount: 1,
      nextRetryAt: 12_000,
      lastError: "node-a disconnected",
    });
    expect(controller.listDueStates(11_999)).toEqual([]);
    expect(controller.listDueStates(12_000)).toHaveLength(1);
    database.close();
  });
});

function addModel(store: MeshStore, id: string) {
  return store.upsertRequestedModel({
    id,
    source: `test/${id}`,
    revision: null,
    contextTokens: 4_096,
    minimumNodes: 1,
    autoActivate: true,
  });
}
