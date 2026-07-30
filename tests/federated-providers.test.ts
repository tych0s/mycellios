import { afterEach, describe, expect, it, vi } from "vitest";
import type { FederationRuntimeConfig } from "../src/core/config.js";
import { createFederatedProviderAdapters } from "../src/coordinator/federated-providers.js";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("federated provider adapters", () => {
  it("discovers external runtime A models when its optional management console is unavailable", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({
        data: [{ id: "Qwen/Qwen2.5-3B-Instruct-GGUF:q4_k_m" }],
      }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }))
      .mockRejectedValueOnce(new TypeError("management console unavailable"));
    vi.stubGlobal("fetch", fetchMock);

    const adapter = createFederatedProviderAdapters(config())[0];
    const discovered = await adapter!.discover(new AbortController().signal);

    expect(discovered.models).toEqual([
      expect.objectContaining({
        canonicalId: "Qwen/Qwen2.5-3B-Instruct-GGUF:q4_k_m",
      }),
    ]);
    expect(discovered.nodes).toEqual([
      expect.objectContaining({
        externalId: "external-runtime-a-aggregate",
        scope: "aggregate",
        individuallySelectable: false,
      }),
    ]);
  });
});

function config(): FederationRuntimeConfig {
  return {
    enabled: true,
    external-runtime-aInferenceUrl: "http://127.0.0.1:9337",
    external-runtime-aManagementUrl: "http://127.0.0.1:3131",
    aiHordeBaseUrl: "https://aihorde.net/api",
    peer-runtimeBaseUrl: "http://127.0.0.1:8000/v1",
    chutesBaseUrl: "https://llm.chutes.ai/v1",
    akashMlBaseUrl: "https://api.akashml.com/v1",
  };
}
