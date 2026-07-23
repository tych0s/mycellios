import { afterEach, describe, expect, it, vi } from "vitest";
import type { ModelActivationManager } from "../src/coordinator/model-activation-manager.js";
import {
  automaticActivationFailureIsTransient,
  createCoordinator,
  nextAutomaticActivationRetry,
  type CoordinatorRuntime,
} from "../src/coordinator/server.js";
import type { StoredRequestedModel } from "../src/storage/store.js";

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
        message: "The network administrator token is missing or invalid.",
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
    this.managing = null;
    return true;
  }
  async close(): Promise<void> { this.managing = null; }
}
