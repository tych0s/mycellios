import { afterEach, describe, expect, it, vi } from "vitest";
import type { ModelActivationManager } from "../src/coordinator/model-activation-manager.js";
import type { ModelActivationProgressEvent } from "../src/coordinator/model-catalog.js";
import {
  automaticActivationFailureIsTransient,
  createCoordinator,
  nextAutomaticActivationRetry,
  type CoordinatorRuntime,
} from "../src/coordinator/server.js";
import type { StoredRequestedModel } from "../src/storage/store.js";
import { addWorker } from "./helpers.js";

describe("requested model API activation flow", () => {
  let runtime: CoordinatorRuntime | null = null;

  afterEach(async () => {
    vi.unstubAllGlobals();
    await runtime?.close();
    runtime = null;
  });

  it("profiles an interface request, records its deficit-free activation, and hands it to the executor", async () => {
    const manager = new FakeActivationManager();
    vi.stubGlobal("fetch", async (input: string | URL | Request) => {
      const url = String(input);
      if (url.endsWith("config.json")) {
        return Response.json({
          model_type: "qwen3",
          architectures: ["Qwen3ForCausalLM"],
          num_hidden_layers: 28,
          hidden_size: 1_024,
          num_attention_heads: 16,
          num_key_value_heads: 8,
          head_dim: 128,
        });
      }
      if (url.endsWith("model.safetensors.index.json")) {
        return Response.json({ metadata: { total_size: 1_200_000_000 } });
      }
      return new Response(null, { status: 404 });
    });
    runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 8_787,
      databasePath: ":memory:",
      requestTimeoutMs: 30_000,
    }, { activationManager: manager });

    const response = await runtime.app.inject({
      method: "POST",
      url: "/public/v1/requested-models",
      payload: {
        id: "qwen-ui",
        source: "Qwen/Qwen3-0.6B",
        revision: null,
        contextTokens: 4_096,
        minimumNodes: 2,
        autoActivate: true,
      },
    });

    expect(response.statusCode).toBe(201);
    expect(response.json().model.status).toBe("activating");
    expect(manager.activated.map((model) => model.id)).toEqual(["qwen-ui"]);
    const stored = runtime.store.getRequestedModel("qwen-ui")!;
    expect(stored.activationRequestedAt).not.toBeNull();
    expect(stored.activationError).toBeNull();
  });

  it("returns an actionable authorization error for protected model changes", async () => {
    runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 8_787,
      databasePath: ":memory:",
      requestTimeoutMs: 30_000,
      modelAdminToken: "model-admin-secret",
    });

    const response = await runtime.app.inject({
      method: "POST",
      url: "/public/v1/requested-models",
      payload: {
        id: "qwen-ui",
        source: "Qwen/Qwen3-0.6B",
        revision: null,
        contextTokens: 4_096,
        minimumNodes: 2,
        autoActivate: true,
      },
    });

    expect(response.statusCode).toBe(401);
    expect(response.json()).toEqual({
      error: {
        code: "invalid_model_admin_token",
        message: "Sign in with an authorized Mycellios account or enter the administrator token.",
      },
    });
  });

  it("recovers a previously stuck transient activation and exposes the retry in the live log", async () => {
    const manager = new FakeActivationManager();
    runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 8_787,
      databasePath: ":memory:",
      requestTimeoutMs: 30_000,
    }, {
      activationManager: manager,
      automaticActivationRetryDelaysMs: [0],
    });
    runtime.store.upsertRequestedModel({
      id: "recover-me",
      source: "Qwen/Qwen3-0.6B",
      revision: null,
      contextTokens: 4_096,
      minimumNodes: 2,
      autoActivate: true,
    });
    runtime.store.setRequestedModelProfile("recover-me", {
      schema: "mycellios-hub-model-capacity/1",
      compatible: true,
      adapterId: "transformers-qwen3-v1",
      requiredVramMiB: 2_200,
      minimumStageVramMiB: 512,
      minimumNodes: 2,
    }, null);
    runtime.store.setRequestedModelActivationError(
      "recover-me",
      "managed_launch_agent_is_unavailable:desktop-a",
    );

    const response = await runtime.app.inject({
      method: "GET",
      url: "/public/v1/snapshot",
    });

    expect(response.statusCode).toBe(200);
    expect(manager.activated.map((model) => model.id)).toEqual(["recover-me"]);
    expect(runtime.store.getRequestedModel("recover-me")?.activationError).toBeNull();
    expect(runtime.store.getRequestedModel("recover-me")?.activationRequestedAt).not.toBeNull();
    const model = response.json().requestedModels.find(
      (entry: { id: string }) => entry.id === "recover-me",
    );
    expect(model.status).toBe("activating");
    expect(model.message).toContain("Automatic retry 1 of 1");
    expect(model.activationProgress).toEqual(expect.arrayContaining([
      expect.objectContaining({ phase: "retrying", state: "running" }),
    ]));
  });

  it("moves a live transient launch failure into retry instead of leaving it red", async () => {
    const manager = new FakeActivationManager([
      "managed_launch_agent_is_unavailable:desktop-a",
    ]);
    manager.progress.push({
      phase: "profiling",
      message: "Existing activation event.",
      at: "2026-01-01T00:00:00.000Z",
      state: "completed",
    });
    runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 8_787,
      databasePath: ":memory:",
      requestTimeoutMs: 30_000,
    }, {
      activationManager: manager,
      automaticActivationRetryDelaysMs: [0],
    });
    runtime.store.upsertRequestedModel({
      id: "retry-live",
      source: "Qwen/Qwen3-0.6B",
      revision: null,
      contextTokens: 4_096,
      minimumNodes: 2,
      autoActivate: true,
    });
    runtime.store.setRequestedModelProfile("retry-live", {
      schema: "mycellios-hub-model-capacity/1",
      compatible: true,
      adapterId: "transformers-qwen3-v1",
      requiredVramMiB: 2_200,
      minimumStageVramMiB: 512,
      minimumNodes: 2,
    }, null);

    await runtime.app.inject({ method: "GET", url: "/public/v1/snapshot" });
    await vi.waitFor(() => {
      expect(runtime?.store.getRequestedModel("retry-live")?.activationRequestedAt).toBeNull();
    });
    const retryResponse = await runtime.app.inject({
      method: "GET",
      url: "/public/v1/snapshot",
    });

    expect(manager.activated.map((model) => model.id)).toEqual(["retry-live", "retry-live"]);
    expect(runtime.store.getRequestedModel("retry-live")?.activationError).toBeNull();
    const model = retryResponse.json().requestedModels.find(
      (entry: { id: string }) => entry.id === "retry-live",
    );
    expect(model.status).toBe("activating");
    expect(model.activationProgress).toEqual(expect.arrayContaining([
      expect.objectContaining({ phase: "retrying", state: "running" }),
    ]));
    const timestamps = model.activationProgress.map(
      (event: ModelActivationProgressEvent) => Date.parse(event.at),
    );
    expect(timestamps).toEqual([...timestamps].sort((left, right) => left - right));
  });

  it("rebuilds a managed cell immediately when its inference proxy fails before token zero", async () => {
    const manager = new FakeActivationManager();
    runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 8_787,
      databasePath: ":memory:",
      requestTimeoutMs: 30_000,
    }, { activationManager: manager });
    const requested = requestedModel(runtime, "repair-adapter");
    const cell = addWorker(runtime.store, {
      id: "cell-route",
      model: requested.id,
      identity: { kind: "cell", id: "cell-route" },
    });
    void manager.activate(requested);

    runtime.service.emit("degraded", {
      jobId: "job-adapter-failed",
      model: requested.id,
      code: "adapter_error",
      workerId: cell.id,
    });

    await vi.waitFor(() => {
      expect(manager.deactivated).toEqual([requested.id]);
      expect(manager.activated.map((model) => model.id)).toEqual([
        requested.id,
        requested.id,
      ]);
    });
  });

  it("unpublishes and rebuilds a managed cell when one physical executor disconnects", async () => {
    const manager = new FakeActivationManager();
    runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 8_787,
      databasePath: ":memory:",
      requestTimeoutMs: 30_000,
    }, { activationManager: manager });
    const requested = requestedModel(runtime, "repair-disconnect");
    const executor = addWorker(runtime.store, {
      id: "executor-route",
      identity: { kind: "device", id: "executor-route" },
      distributedExecutor: {
        protocol: "gdlp-worker-tunnel/2",
        nodeId: "desktop-route",
        stageHost: "127.0.0.1",
        stagePort: 9_860,
        runtime: "python-safetensors",
        computeMode: "automatic",
        cpuEligible: false,
      },
    });
    addWorker(runtime.store, {
      id: "cell-route",
      model: requested.id,
      identity: { kind: "cell", id: "cell-route" },
      execution: {
        deviceType: "gpu",
        backend: "cuda",
        deviceName: "Synthetic GPU",
        precision: "float16",
        fallback: false,
        stages: [{
          nodeId: "desktop-route",
          stageIndex: 0,
          layerStart: 0,
          layerEnd: 14,
          deviceType: "gpu",
          backend: "cuda",
          deviceName: "Synthetic GPU",
          precision: "float16",
          fallback: false,
        }],
      },
    });
    void manager.activate(requested);

    runtime.hub.emit("disconnect", executor.id);

    await vi.waitFor(() => {
      expect(manager.deactivated).toEqual([requested.id]);
      expect(manager.activated.map((model) => model.id)).toEqual([
        requested.id,
        requested.id,
      ]);
    });
  });
});

describe("automatic activation recovery", () => {
  it.each([
    "distributed_worker_disconnected:wrk-test",
    "managed_launch_agent_is_unavailable:desktop-test",
    "gpu_only_runtime_not_ready",
    "launch_process_exited:stage-1:gpu_model_stage_unavailable_after_retries:cuda:out of memory",
    "launch_readiness_timeout:stage-1:600000",
  ])("retries transient distributed failure %s", (message) => {
    expect(automaticActivationFailureIsTransient(message)).toBe(true);
  });

  it("keeps deterministic model incompatibility as an actionable failure", () => {
    expect(automaticActivationFailureIsTransient("unsupported_model_architecture:qwen-next")).toBe(false);
  });

  it("uses bounded progressive retries and stops after the configured limit", () => {
    const first = nextAutomaticActivationRetry(0, "temporary", 1_000, [5_000, 15_000]);
    expect(first).toMatchObject({
      retryCount: 0,
      nextAttemptAt: 6_000,
      launching: false,
    });
    const second = nextAutomaticActivationRetry(1, "temporary", 1_000, [5_000, 15_000]);
    expect(second?.nextAttemptAt).toBe(16_000);
    expect(nextAutomaticActivationRetry(2, "temporary", 1_000, [5_000, 15_000])).toBeNull();
    expect(automaticActivationFailureIsTransient(
      "automatic_activation_retries_exhausted:2:managed_launch_agent_is_unavailable:desktop-test",
    )).toBe(false);
  });
});

class FakeActivationManager implements ModelActivationManager {
  readonly activated: StoredRequestedModel[] = [];
  readonly deactivated: string[] = [];
  readonly progress: ModelActivationProgressEvent[] = [];
  private managing: string | null = null;

  constructor(private readonly failures: string[] = []) {}

  async initialize(): Promise<void> {}
  async refresh(): Promise<void> {}
  capacityNodesForModel(modelId: string) {
    if (this.managing !== null && this.managing !== modelId) return [];
    return [
      { id: "executor-a", availableVramMiB: 4_096 },
      { id: "executor-b", availableVramMiB: 4_096 },
    ];
  }
  isManaging(modelId: string): boolean { return this.managing === modelId; }
  isBusy(): boolean { return this.managing !== null; }
  activate(model: StoredRequestedModel): Promise<void> {
    this.activated.push(model);
    const failure = this.failures.shift();
    if (failure) return Promise.reject(new Error(failure));
    this.managing = model.id;
    return new Promise<void>(() => undefined);
  }
  async deactivate(modelId: string): Promise<boolean> {
    if (this.managing !== modelId) return false;
    this.deactivated.push(modelId);
    this.managing = null;
    return true;
  }
  activationProgressForModel(): readonly ModelActivationProgressEvent[] {
    return this.progress;
  }
  async close(): Promise<void> { this.managing = null; }
}

function requestedModel(runtime: CoordinatorRuntime, id: string): StoredRequestedModel {
  runtime.store.upsertRequestedModel({
    id,
    source: "Qwen/Qwen3-0.6B",
    revision: null,
    contextTokens: 4_096,
    minimumNodes: 2,
    autoActivate: true,
  });
  runtime.store.setRequestedModelProfile(id, {
    schema: "mycellios-hub-model-capacity/1",
    compatible: true,
    adapterId: "transformers-qwen3-v1",
    requiredVramMiB: 2_200,
    minimumStageVramMiB: 512,
    minimumNodes: 2,
  }, null);
  return runtime.store.getRequestedModel(id)!;
}
