import { afterEach, describe, expect, it } from "vitest";
import type {
  FederatedInferenceEvent,
  FederatedInferenceInput,
  FederatedProviderAdapter,
} from "../src/contracts/federation.js";
import type { ChatCompletionRequest } from "../src/contracts/types.js";
import { FederationManager } from "../src/coordinator/federation-manager.js";
import { MeshDatabase } from "../src/storage/database.js";
import { MeshStore } from "../src/storage/store.js";

const resources: Array<{ manager: FederationManager; database: MeshDatabase }> = [];

afterEach(async () => {
  for (const resource of resources.splice(0)) {
    await resource.manager.close();
    resource.database.close();
  }
});

describe("FederationManager", () => {
  it("keeps discovery unroutable until a real generation completes", async () => {
    const adapter = new FakeCommunityAdapter();
    const { manager } = createManager(adapter);

    await manager.discover("external-runtime-a");

    expect(manager.verifiedModels()).toEqual([]);
    expect(manager.nodes()).toEqual([
      expect.objectContaining({
        networkId: "external-runtime-a",
        scope: "logical",
        routable: false,
      }),
    ]);
    expect(manager.nodes()[0]?.id).not.toContain(adapter.rawNodeId);

    const network = await manager.probe("external-runtime-a");

    expect(adapter.lastRequest?.max_tokens).toBe(16);
    expect(network.actualState).toBe("ready");
    expect(manager.verifiedModels().map((model) => model.canonicalId)).toEqual([
      "test/model",
    ]);
    expect(manager.nodes()[0]?.routable).toBe(true);
  });

  it("streams one verified route and persists its real attempt", async () => {
    const adapter = new FakeCommunityAdapter();
    const { manager, database } = createManager(adapter);
    await manager.probe("external-runtime-a");

    const handle = manager.submit(chatRequest("test/model"), "session-1", "idem-1");
    const events = [];
    for await (const event of handle.events) events.push(event);

    expect(events.map((event) => event.type)).toEqual([
      "accepted",
      "token",
      "completed",
    ]);
    expect(database.raw.prepare(
      "SELECT result, provider, output_tokens FROM federated_route_attempts",
    ).get()).toEqual({
      result: "completed",
      provider: "external-runtime-a",
      output_tokens: 1,
    });
    expect(database.raw.prepare(
      "SELECT status FROM jobs WHERE id = ?",
    ).get(handle.jobId)).toEqual({ status: "completed" });
  });

  it("opens the circuit after three pre-token canary failures", async () => {
    const adapter = new FakeCommunityAdapter();
    adapter.failInference = true;
    const { manager } = createManager(adapter);

    await manager.probe("external-runtime-a");
    await manager.probe("external-runtime-a");
    const network = await manager.probe("external-runtime-a");

    expect(network.actualState).toBe("circuit-open");
    expect(network.consecutivePretokenFailures).toBe(3);
    expect(network.retryAt).not.toBeNull();
  });

  it("reserves provider spend atomically and hard-blocks the limit", () => {
    const database = new MeshDatabase(":memory:");
    const store = new MeshStore(database);
    const now = Date.UTC(2026, 6, 29, 12);

    expect(store.reserveProviderSpend({
      id: "r1",
      provider: "chutes",
      requestId: "q1",
      maximumUsd: 0.7,
      dailyBudgetUsd: 1,
      monthlyBudgetUsd: 1,
      now,
    })).toBe(true);
    expect(store.reserveProviderSpend({
      id: "r2",
      provider: "chutes",
      requestId: "q2",
      maximumUsd: 0.4,
      dailyBudgetUsd: 1,
      monthlyBudgetUsd: 1,
      now,
    })).toBe(false);
    database.close();
  });
});

class FakeCommunityAdapter implements FederatedProviderAdapter {
  readonly id = "external-runtime-a" as const;
  readonly class = "community" as const;
  readonly configured = true;
  readonly rawNodeId = "upstream-secret-node-123";
  failInference = false;
  lastRequest: ChatCompletionRequest | null = null;

  async discover() {
    return {
      models: [{
        canonicalId: "test/model",
        externalId: "upstream/test-model",
        displayName: "Test model",
      }],
      nodes: [{
        externalId: this.rawNodeId,
        scope: "logical" as const,
        models: ["test/model"],
        status: "online" as const,
      }],
    };
  }

  async *infer(input: FederatedInferenceInput): AsyncIterable<FederatedInferenceEvent> {
    this.lastRequest = input.request;
    if (this.failInference) throw new Error("fake_pretoken_failure");
    yield { type: "token", text: "OK", index: 0, at: Date.now() };
    yield {
      type: "completed",
      text: "OK",
      inputTokens: 3,
      outputTokens: 1,
      finishReason: "stop",
    };
  }

  estimateMaximumCostUsd() {
    return 0;
  }
}

function createManager(adapter: FederatedProviderAdapter) {
  const database = new MeshDatabase(":memory:");
  const store = new MeshStore(database);
  const manager = new FederationManager(store, [adapter], true);
  resources.push({ manager, database });
  return { manager, database, store };
}

function chatRequest(model: string): ChatCompletionRequest {
  return {
    model,
    messages: [{ role: "user", content: "hello" }],
    stream: true,
    max_tokens: 32,
  };
}
