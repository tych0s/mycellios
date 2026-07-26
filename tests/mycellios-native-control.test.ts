import { describe, expect, it } from "vitest";
import { createAdapter } from "../src/adapters/factory.js";
import { workerConfigSchema } from "../src/contracts/schemas.js";
import { WorkerAgent } from "../src/worker/agent.js";

const nativeControlConfig = workerConfigSchema.parse({
  region: "test",
  offeredVramMb: 4_096,
  limits: {
    maxConcurrency: 1,
    pauseWhenForeground: false,
  },
  adapter: {
    kind: "mycellios-native",
    model: "mycellios-native-control",
  },
  deployment: { contextLimit: 8_192 },
});

describe("Mycellios native control worker", () => {
  it("reports no loaded model and cannot synthesize inference", async () => {
    const adapter = createAdapter(nativeControlConfig);

    await expect(adapter.probe()).resolves.toMatchObject({
      kind: "mycellios-native",
      models: [],
      streaming: false,
    });
    await expect(adapter.metrics()).resolves.toEqual({
      ready: false,
      activeJobs: 0,
      loadedModels: [],
    });
    const generation = adapter.generate({
      jobId: "job",
      request: {
        model: "not-deployed",
        messages: [{ role: "user", content: "hello" }],
      },
    }, new AbortController().signal);
    await expect(generation[Symbol.asyncIterator]().next()).rejects.toMatchObject({
      code: "native_deployment_required",
      retryable: false,
    });
  });

  it("refuses to advertise the control adapter as a deployment", () => {
    expect(() => new WorkerAgent(nativeControlConfig, {
      coordinatorUrl: "http://127.0.0.1:4180",
      advertiseDeployment: true,
    })).toThrow("mycellios_native_control_must_not_advertise_an_inference_deployment");

    expect(() => new WorkerAgent(nativeControlConfig, {
      coordinatorUrl: "http://127.0.0.1:4180",
      advertiseDeployment: false,
    })).not.toThrow();
  });

  it("publishes physical capacity with an empty deployment list", async () => {
    const agent = new WorkerAgent(nativeControlConfig, {
      coordinatorUrl: "http://127.0.0.1:4180",
      advertiseDeployment: false,
      hardwareProbe: async () => ({
        hostname: "native-node",
        platform: "linux",
        ramMb: 16_384,
        gpus: [{
          id: "gpu-0",
          vendor: "nvidia",
          model: "NVIDIA RTX",
          physicalVramMb: 8_192,
        }],
      }),
      logger: { info() {}, warn() {}, error() {} },
    });

    const capabilities = await (agent as unknown as {
      buildCapabilities(): Promise<{
        deployments: unknown[];
        gpus: Array<{ vendor: string; sharedMemoryMb?: number; offeredVramMb: number }>;
      }>;
    }).buildCapabilities();

    expect(capabilities.deployments).toEqual([]);
    // Without a verified GPU runtime, detected VRAM remains unclaimed and the
    // node exposes only its bounded CPU-memory capacity.
    expect(capabilities.gpus[0]).toMatchObject({
      vendor: "cpu",
      sharedMemoryMb: 4_096,
      offeredVramMb: 4_096,
    });
  });
});
