import { describe, expect, it } from "vitest";
import { workerConfigSchema } from "../src/contracts/schemas.js";
import type { LaunchAgent } from "../src/distribution/launch-supervisor.js";
import {
  parsePhysicalRuntimePerformanceProfile,
  probeRuntimePerformanceProfile,
  type RuntimeProfileCommandRunner,
} from "../src/performance/runtime-profile-probe.js";
import {
  sealRuntimePerformanceProfile,
  type RuntimePerformanceProfileInput,
} from "../src/performance/runtime-profile.js";
import { WorkerAgent } from "../src/worker/agent.js";

describe("physical runtime performance probe", () => {
  it("runs the native module in the selected runtime and seals its current evidence", async () => {
    const measuredAt = "2026-07-25T12:00:00.000Z";
    const calls: Array<{
      executable: string;
      arguments_: readonly string[];
      pythonPath: string | undefined;
    }> = [];
    const runner: RuntimeProfileCommandRunner = async (
      executable,
      arguments_,
      options,
    ) => {
      calls.push({
        executable,
        arguments_,
        pythonPath: options.env?.PYTHONPATH,
      });
      return {
        code: 0,
        stdout: JSON.stringify(input(measuredAt)),
        stderr: "",
      };
    };
    const profile = await probeRuntimePerformanceProfile({
      pythonExecutable: "C:\\mycellios\\python.exe",
      pythonPath: ["C:\\mycellios\\python"],
      backend: "cuda",
      device: "cuda:0",
      precision: "float16",
      expectedDeviceName: "NVIDIA RTX 2060",
      commandRunner: runner,
      now: () => Date.parse(measuredAt),
    });

    expect(profile.profileId).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(profile.source).toBe("physical-microbenchmark");
    expect(profile.activationCodecId).toBe("fp16");
    expect(calls).toHaveLength(1);
    expect(calls[0]?.arguments_).toEqual(expect.arrayContaining([
      "distributed_runtime.runtime_profile",
      "--backend",
      "cuda",
      "--device",
      "cuda:0",
      "--samples",
      "9",
    ]));
    expect(calls[0]?.pythonPath).toContain("mycellios");
  });

  it("fails closed for non-physical, wrong-backend or structurally unknown output", () => {
    const expected = { backend: "cuda" as const, precision: "float16" as const };
    expect(() => parsePhysicalRuntimePerformanceProfile(JSON.stringify({
      ...input(),
      source: "runtime-calibration",
    }), expected)).toThrow("runtime_performance_probe_source_is_not_physical");
    expect(() => parsePhysicalRuntimePerformanceProfile(JSON.stringify({
      ...input(),
      backend: "rocm",
    }), expected)).toThrow("runtime_performance_probe_backend_does_not_match");
    expect(() => parsePhysicalRuntimePerformanceProfile(JSON.stringify({
      ...input(),
      invented: true,
    }), expected)).toThrow();
  });

  it("publishes the sealed profile in worker capabilities only when it matches capacity", async () => {
    const profile = sealRuntimePerformanceProfile(input());
    const launchAgent: LaunchAgent = {
      id: "profile-test",
      async start() {
        throw new Error("not used");
      },
    };
    const agent = new WorkerAgent(workerConfigSchema.parse({
      region: "test",
      offeredVramMb: 6_144,
      limits: { maxConcurrency: 1, pauseWhenForeground: false },
      adapter: {
        kind: "mock",
        model: "profile-test",
        tokensPerSecond: 10,
        ttftMs: 10,
        failureRate: 0,
      },
      deployment: { contextLimit: 2_048 },
    }), {
      coordinatorUrl: "http://127.0.0.1:9999",
      reconnect: false,
      hardwareProbe: async () => ({
        hostname: "profile-node",
        platform: "win32",
        ramMb: 16_384,
        gpus: [{
          id: "gpu-0",
          vendor: "nvidia",
          model: "NVIDIA RTX 2060",
          physicalVramMb: 6_144,
        }],
      }),
      verifiedGpuRuntime: {
        status: "gpu-ready",
        backend: "cuda",
        deviceName: "NVIDIA RTX 2060",
      },
      distributedExecutor: {
        nodeId: "profile-node",
        stageHost: "profile-node.relay",
        stagePort: 9_850,
        launchAgent,
        computeMode: "automatic",
        cpuEligible: false,
      },
      runtimePerformanceProfileProbe: async () => profile,
      logger: { info() {}, warn() {}, error() {} },
    });

    const capabilities = await (agent as unknown as {
      buildCapabilities(): Promise<{
        distributedExecutor?: { performanceProfile?: typeof profile };
      }>;
    }).buildCapabilities();
    expect(capabilities.distributedExecutor?.performanceProfile).toEqual(profile);
  });

  it.skip("requires the packaged Mycellios runtime and a physical accelerator", () => {
    // RUN_PHYSICAL_RUNTIME_PROFILE_TESTS is intentionally handled by the
    // hardware campaign, never replaced with synthetic GPU timings in CI.
  });
});

function input(
  measuredAt = "2026-07-25T12:00:00.000Z",
): RuntimePerformanceProfileInput {
  return {
    measuredAt,
    backend: "cuda",
    deviceName: "NVIDIA RTX 2060",
    precision: "float16",
    source: "physical-microbenchmark",
    activationCodecId: "fp16",
    decodeMemory: series("GB/s", 400),
    prefillCompute: series("TFLOP/s", 20),
    activationCodec: series("GB/s", 2),
  };
}

function series(unit: "GB/s" | "TFLOP/s", median: number) {
  return {
    unit,
    warmupSamples: 2,
    samples: 7,
    p5: median * 0.9,
    p50: median,
    p95: median * 1.1,
    confidenceHalfWidthPct: 5,
  };
}
