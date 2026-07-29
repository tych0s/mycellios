import { describe, expect, it } from "vitest";
import {
  inspectHubModelCapacity,
  requestedModelCapacityViews,
  searchHubModelCatalog,
  shouldQueueAutomaticActivation,
  type HubModelCapacityProfile,
} from "../src/coordinator/model-catalog.js";
import {
  MODEL_ADAPTER_EVIDENCE_SCOPE,
  MODEL_ADAPTER_REGISTRY_ID,
  modelAdapterRegistry,
  resolveModelAdapterContract,
} from "../src/contracts/model-adapter-registry.js";
import type { StoredRequestedModel, StoredWorker } from "../src/storage/store.js";

describe("requested model capacity catalog", () => {
  it("uses the one identity-sealed software registry without claiming physical evidence", () => {
    const registry = modelAdapterRegistry();
    expect(registry.registryId).toBe(MODEL_ADAPTER_REGISTRY_ID);
    expect(registry.evidenceScope).toBe("software-contract-only");
    expect(registry.adapters).toHaveLength(4);
    expect(resolveModelAdapterContract("qwen3", "Qwen3ForCausalLM")).toMatchObject({
      id: "transformers-qwen3-v1",
      adapterContractId: expect.stringMatching(/^sha256:[0-9a-f]{64}$/),
    });
    expect(resolveModelAdapterContract("mistral", "MistralForCausalLM")).toBeNull();
  });

  it("searches and paginates the public Hub catalog without hiding unsupported models", async () => {
    const results = await searchHubModelCatalog("qwen", async (input) => {
      const url = new URL(String(input));
      expect(url.searchParams.get("search")).toBe("qwen");
      expect(url.searchParams.get("pipeline_tag")).toBe("text-generation");
      expect(url.searchParams.getAll("expand")).toContain("safetensors");
      expect(url.searchParams.getAll("expand")).toContain("config");
      expect(url.searchParams.get("limit")).toBe("50");
      return Response.json([
        {
          id: "org/unsupported-popular",
          author: "org",
          downloads: 9_000_000,
          likes: 10,
          gated: false,
          private: false,
          pipeline_tag: "text-generation",
          tags: ["safetensors"],
          config: { model_type: "mistral", architectures: ["MistralForCausalLM"] },
        },
        {
          id: "Qwen/Qwen3-0.6B",
          author: "Qwen",
          downloads: 1_000_000,
          likes: 100,
          gated: false,
          private: false,
          pipeline_tag: "text-generation",
          tags: ["transformers", "safetensors"],
          safetensors: { parameters: { BF16: 751_632_384 }, total: 751_632_384 },
          config: { model_type: "qwen3", architectures: ["Qwen3ForCausalLM"] },
        },
      ], { headers: { link: '<https://huggingface.co/api/models?cursor=next-page-token>; rel="next"' } });
    });

    expect(results.data.map((model) => model.id)).toEqual(["org/unsupported-popular", "Qwen/Qwen3-0.6B"]);
    expect(results.nextCursor).toBe("next-page-token");
    expect(results.data[1]?.compatible).toBe(true);
    expect(results.data[1]?.adapterId).toBe("transformers-qwen3-v1");
    expect(results.data[1]?.parameterCount).toBe(751_632_384);
    expect(results.data[1]?.estimatedMemoryMiB).toBeGreaterThan(1_500);
    expect(results.data[0]?.compatible).toBe(false);
    expect(results.data[0]?.compatibilityReason).toContain("registered native");
    expect(results.data[1]?.adapterRegistryId).toBe(MODEL_ADAPTER_REGISTRY_ID);
  });

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
    expect(profile.adapterRegistryId).toBe(MODEL_ADAPTER_REGISTRY_ID);
    expect(profile.adapterContractId).toMatch(/^sha256:[0-9a-f]{64}$/);
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

    const oneRealExecutor = requestedModelCapacityViews({
      requests: [request],
      workers: [],
      connectedWorkerIds: new Set(),
      activeModelIds: new Set(),
      executionNodesForModel: () => [{ id: "desktop-only", availableVramMiB: 16_384 }],
      activationAvailable: true,
    })[0]!;
    expect(oneRealExecutor.status).toBe("waiting_capacity");
    expect(oneRealExecutor.availableNodes).toBe(1);
    expect(oneRealExecutor.missingNodes).toBe(1);
    expect(shouldQueueAutomaticActivation(oneRealExecutor)).toBe(false);

    const active = requestedModelCapacityViews({
      requests: [{
        ...request,
        activationRequestedAt: Date.now(),
        activationError: "an older activation failed",
      }],
      workers: [worker("a", 4_096, 0), worker("b", 4_096, 0)],
      connectedWorkerIds: new Set(["a", "b"]),
      activeModelIds: new Set([request.id]),
      activationIncidentForModel: () => ({
        schema: "mycellios-activation-incident/1",
        code: "unknown",
        scope: "model",
        title: "Stale failure",
        summary: "Stale failure",
        remedy: "Stale failure",
        steps: [],
        retryable: false,
        automatic: false,
        repairState: "manual_required",
        automaticAction: "none",
        attempt: 0,
        maximumAttempts: 5,
        nextRetryAt: null,
        nodeId: null,
        stageId: null,
        processExitCode: null,
      }),
    })[0]!;
    expect(active.status).toBe("active");
    expect(active.activationIncident).toBeNull();
  });

  it("fails closed instead of activating a profile from another adapter registry", () => {
    const staleProfile = {
      ...profileFixture(2_000, 500),
      adapterRegistryId: `sha256:${"0".repeat(64)}`,
    };
    const view = requestedModelCapacityViews({
      requests: [requestedModel(staleProfile)],
      workers: [worker("a", 4_096, 0), worker("b", 4_096, 0)],
      connectedWorkerIds: new Set(["a", "b"]),
      activeModelIds: new Set(),
    })[0]!;

    expect(view.status).toBe("profiling");
    expect(view.compatible).toBeNull();
    expect(shouldQueueAutomaticActivation(view)).toBe(false);
  });
});

function profileFixture(requiredVramMiB: number, minimumStageVramMiB: number): HubModelCapacityProfile {
  const adapter = resolveModelAdapterContract("qwen3", "Qwen3ForCausalLM")!;
  return {
    schema: "mycellios-hub-model-capacity/1",
    adapterId: adapter.id,
    adapterContractId: adapter.adapterContractId,
    adapterRegistryId: MODEL_ADAPTER_REGISTRY_ID,
    adapterEvidenceScope: MODEL_ADAPTER_EVIDENCE_SCOPE,
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
    identityKind: "device",
    identityId: id,
  };
}
