import { describe, expect, it } from "vitest";
import { workerCapabilitiesSchema, workerConfigSchema } from "../src/contracts/schemas.js";
import {
  llmfitHardwareFallback,
  probeLlmfit,
  type LlmfitRunner,
} from "../src/worker/llmfit.js";

describe("llmfit worker integration", () => {
  it("normalizes system and configured-model advice", async () => {
    const calls: string[][] = [];
    const runner: LlmfitRunner = async (_executable, arguments_) => {
      calls.push(arguments_);
      if (arguments_.includes("system")) return JSON.stringify(systemFixture());
      return JSON.stringify({
        models: [
          {
            name: "HuggingFaceTB/SmolLM2-135M-Instruct",
            local-model-runtime_name: "smollm2:135m",
            fit_level: "Perfect",
            run_mode: "GPU",
            runtime: "external GGUF runtime",
            best_quant: "Q8_0",
            estimated_tps: 981.3,
            measured_tps: 812.4,
            memory_required_gb: 0.82,
            usable_context: 8192,
          },
        ],
      });
    };

    const result = await probeLlmfit(
      {
        executable: "llmfit",
        arguments: [],
        timeoutMs: 5_000,
        model: "smollm2:135m",
        maxContext: 8_192,
      },
      runner,
    );

    expect(calls).toEqual([
      ["--json", "system"],
      ["--max-context", "8192", "--json", "info", "smollm2:135m"],
    ]);
    expect(result.warnings).toEqual([]);
    expect(result.advisory).toMatchObject({
      source: "llmfit",
      backend: "Vulkan",
      totalRamMb: 31_867,
      gpuCount: 1,
      gpus: [{ name: "AMD Radeon(TM) 890M Graphics", vramMb: 16_384 }],
      model: {
        requestedModel: "smollm2:135m",
        resolvedModel: "HuggingFaceTB/SmolLM2-135M-Instruct",
        measuredTokensPerSecond: 812.4,
        memoryRequiredMb: 840,
      },
    });
    expect(workerCapabilitiesSchema.shape.llmfit.safeParse(result.advisory).success).toBe(true);
  });

  it("keeps hardware advice when model inspection fails", async () => {
    const runner: LlmfitRunner = async (_executable, arguments_) => {
      if (arguments_.includes("system")) return JSON.stringify(systemFixture());
      throw new Error("model not found");
    };
    const result = await probeLlmfit(
      {
        executable: "llmfit",
        arguments: [],
        timeoutMs: 5_000,
        model: "missing/model",
      },
      runner,
    );
    expect(result.advisory.model).toBeUndefined();
    expect(result.warnings[0]).toMatch(/model inspection failed.*model not found/);
  });

  it("can replace only an unidentified native GPU probe", async () => {
    const result = await probeLlmfit(
      { executable: "llmfit", arguments: [], timeoutMs: 5_000 },
      async () => JSON.stringify(systemFixture()),
    );
    expect(llmfitHardwareFallback(result.advisory)).toMatchObject({
      vendor: "amd",
      model: "AMD Radeon(TM) 890M Graphics",
      physicalVramMb: 16_384,
    });
  });

  it("defaults to advisory-only behavior", () => {
    const config = workerConfigSchema.parse({
      region: "test",
      offeredVramMb: 4_096,
      limits: { maxConcurrency: 1, pauseWhenForeground: false },
      adapter: {
        kind: "mock",
        model: "test-model",
        tokensPerSecond: 20,
        ttftMs: 100,
        failureRate: 0,
      },
      deployment: { contextLimit: 8_192 },
      llmfit: { enabled: true },
    });
    expect(config.llmfit).toMatchObject({
      enabled: true,
      required: false,
      executable: "llmfit",
      applyPerformanceEstimate: false,
    });
  });

  it("rejects invalid system JSON", async () => {
    await expect(
      probeLlmfit(
        { executable: "llmfit", arguments: [], timeoutMs: 5_000 },
        async () => "not-json",
      ),
    ).rejects.toThrow(/invalid JSON for system/);
  });
});

function systemFixture() {
  return {
    system: {
      available_ram_gb: 2.29,
      backend: "Vulkan",
      cpu_cores: 24,
      cpu_name: "AMD Ryzen AI 9 HX 370 w/ Radeon 890M",
      gpu_count: 1,
      gpu_name: "AMD Radeon(TM) 890M Graphics",
      gpu_vram_gb: 16,
      gpus: [
        {
          backend: "Vulkan",
          count: 1,
          memory_bandwidth_gbps: null,
          name: "AMD Radeon(TM) 890M Graphics",
          unified_memory: false,
          vram_gb: 16,
        },
      ],
      has_gpu: true,
      total_ram_gb: 31.12,
      unified_memory: false,
    },
  };
}
