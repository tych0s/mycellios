import { describe, expect, it } from "vitest";
import {
  inspectHubModelCapacity,
  requestedModelCapacityViews,
  shouldQueueAutomaticActivation,
  type HubModelCapacityProfile,
} from "../src/coordinator/model-catalog.js";
import type { StoredRequestedModel, StoredWorker } from "../src/storage/store.js";

describe("requested model capacity catalog", () => {
  it("profiles a compatible Hub checkpoint from metadata without downloading weights", async () => {
    const requests: string[] = [];
    const profile = await inspectHubModelCapacity(
      {
        source: "Qwen/Qwen3-0.6B",
        revision: null,
        contextTokens: 4_096,
        minimumNodes: 2,
      },
      async (input) => {
        const url = String(input);
        requests.push(url);
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
          return Response.json({ metadata: { total_size: 1_200_000_000 }, weight_map: {} });
        }
        return new Response(null, { status: 404 });
      },
    );

    expect(profile.adapterId).toBe("transformers-qwen3-v1");
    expect(profile.compatible).toBe(true);
    expect(profile.weightBytes).toBe(1_200_000_000);
    expect(profile.requiredVramMiB).toBeGreaterThan(1_200_000_000 / 1024 / 1024);
    expect(profile.minimumNodes).toBe(2);
    expect(requests).toHaveLength(2);
  });

  it("marks a familiar but uncertified architecture as incompatible", async () => {
    const profile = await inspectHubModelCapacity(
      { source: "org/mistral-model", revision: "main", contextTokens: 2_048, minimumNodes: 2 },
      async (input) => String(input).endsWith("config.json")
        ? Response.json({
            model_type: "mistral",
            architectures: ["MistralForCausalLM"],
            num_hidden_layers: 8,
            hidden_size: 512,
            num_attention_heads: 8,
          })
        : Response.json({ metadata: { total_size: 500_000_000 } }),
    );

    expect(profile.compatible).toBe(false);
    expect(profile.adapterId).toBeNull();
    expect(profile.incompatibilityReason).toContain("mistral");
  });

  it("reports exact missing memory and node counts, then queues activation at zero deficit", () => {
    const request = requestedModel(profileFixture(6_000, 1_000));
    const waiting = requestedModelCapacityViews({
      requests: [request],
      workers: [worker("a", 1_500, 1_000), worker("b", 2_048, 0)],
      connectedWorkerIds: new Set(["a", "b"]),
      activeModelIds: new Set(),
    })[0]!;

    expect(waiting.status).toBe("waiting_capacity");
    expect(waiting.availableNodes).toBe(1);
    expect(waiting.missingNodes).toBe(1);
    expect(waiting.availableVramMiB).toBe(2_048);
    expect(waiting.missingVramMiB).toBe(3_952);
    expect(waiting.message).toContain("3952 MiB");
    expect(waiting.message).toContain("1 more compatible node");

    const ready = requestedModelCapacityViews({
      requests: [request],
      workers: [worker("a", 4_096, 0), worker("b", 4_096, 0)],
      connectedWorkerIds: new Set(["a", "b"]),
      activeModelIds: new Set(),
    })[0]!;
    expect(ready.status).toBe("activating");
    expect(ready.missingNodes).toBe(0);
    expect(ready.missingVramMiB).toBe(0);
    expect(shouldQueueAutomaticActivation(ready)).toBe(true);

    const active = requestedModelCapacityViews({
      requests: [{ ...request, activationRequestedAt: Date.now() }],
      workers: [worker("a", 4_096, 0), worker("b", 4_096, 0)],
      connectedWorkerIds: new Set(["a", "b"]),
      activeModelIds: new Set([request.id]),
    })[0]!;
    expect(active.status).toBe("active");
  });
});

function profileFixture(requiredVramMiB: number, minimumStageVramMiB: number): HubModelCapacityProfile {
  return {
    schema: "mycellios-hub-model-capacity/1",
    adapterId: "transformers-qwen3-v1",
    compatible: true,
    incompatibilityReason: null,
    architecture: "Qwen3ForCausalLM",
    modelType: "qwen3",
    totalLayers: 28,
    hiddenSize: 1_024,
    weightBytes: 1_200_000_000,
    requiredVramMiB,
    minimumStageVramMiB,
    minimumNodes: 2,
    contextTokens: 4_096,
    source: "Qwen/Qwen3-0.6B",
    revision: "main",
  };
}

function requestedModel(profile: HubModelCapacityProfile): StoredRequestedModel {
  return {
    id: "qwen3-auto",
    source: profile.source,
    revision: null,
    contextTokens: profile.contextTokens,
    minimumNodes: profile.minimumNodes,
    autoActivate: true,
    profile: profile as unknown as Record<string, unknown>,
    profileError: null,
    activationRequestedAt: null,
    activationError: null,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

function worker(id: string, offeredVramMb: number, reservedVramMb: number): StoredWorker {
  return {
    id,
    status: "online",
    capabilities: {
      region: "test",
      agentVersion: "test",
      gpus: [{
        id: `${id}-gpu`,
        vendor: "test",
        model: "test",
        physicalVramMb: offeredVramMb,
        offeredVramMb,
        freeOfferedVramMb: offeredVramMb,
      }],
      limits: { maxConcurrency: 1, pauseWhenForeground: false },
      deployments: reservedVramMb > 0 ? [{
        deploymentId: `${id}-deployment`,
        model: "other-model",
        modelDigest: "sha256:test",
        mode: "replica",
        adapter: "mock",
        peakVramMb: reservedVramMb,
        contextLimit: 4_096,
        maxConcurrency: 1,
        freeSlots: 1,
        tokensPerSecond: 1,
        ttftMs: 1,
        dataLocality: "local",
      }] : [],
      network: { coordinatorRttMs: 1, uplinkMbps: 100, downlinkMbps: 100 },
    },
    reliability: 1,
    jobsCompleted: 0,
    lastSeenAt: Date.now(),
  };
}
