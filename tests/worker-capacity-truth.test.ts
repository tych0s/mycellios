import { describe, expect, it } from "vitest";
import { workerConfigSchema } from "../src/contracts/schemas.js";
import { WorkerAgent } from "../src/worker/agent.js";

const hardwareProbe = async () => ({
  hostname: "capacity-node",
  platform: "linux" as const,
  ramMb: 16_384,
  gpus: [{
    id: "gpu-0",
    vendor: "nvidia",
    model: "NVIDIA GeForce RTX 4090",
    physicalVramMb: 24_576,
  }],
});

const config = workerConfigSchema.parse({
  region: "test",
  offeredVramMb: 24_576,
  limits: { maxConcurrency: 1, pauseWhenForeground: false },
  adapter: {
    kind: "mock",
    model: "capacity-test",
    tokensPerSecond: 10,
    ttftMs: 10,
    failureRate: 0,
  },
  deployment: { contextLimit: 2_048 },
});

type CapabilityHarness = { buildCapabilities(): Promise<{ gpus: Array<{
  vendor: string;
  model: string;
  physicalVramMb: number;
  sharedMemoryMb?: number;
  offeredVramMb: number;
}>; deployments: Array<{ peakVramMb: number }> }> };

type MutableCapabilityHarness = CapabilityHarness & {
  capabilities: Awaited<ReturnType<CapabilityHarness["buildCapabilities"]>> | null;
  sendHeartbeat(): Promise<void>;
};

describe("worker capacity truth", () => {
  it("announces bounded CPU RAM while no verified GPU backend exists", async () => {
    const agent = new WorkerAgent(config, {
      coordinatorUrl: "http://127.0.0.1:9999",
      reconnect: false,
      hardwareProbe,
      logger: { info() {}, warn() {}, error() {} },
    });
    const capabilities = await (agent as unknown as CapabilityHarness).buildCapabilities();
    expect(capabilities.gpus[0]).toMatchObject({
      vendor: "cpu",
      physicalVramMb: 0,
      sharedMemoryMb: 4_096,
      offeredVramMb: 4_096,
    });
  });

  it("announces VRAM only after matching gpu-ready evidence", async () => {
    const agent = new WorkerAgent(config, {
      coordinatorUrl: "http://127.0.0.1:9999",
      reconnect: false,
      hardwareProbe,
      verifiedGpuRuntime: {
        status: "gpu-ready",
        backend: "cuda",
        deviceName: "NVIDIA GeForce RTX 4090",
      },
      logger: { info() {}, warn() {}, error() {} },
    });
    const capabilities = await (agent as unknown as CapabilityHarness).buildCapabilities();
    expect(capabilities.gpus[0]).toMatchObject({
      vendor: "nvidia",
      physicalVramMb: 24_576,
      offeredVramMb: 24_576,
    });
  });

  it("switches published capacity in place without restarting active work", async () => {
    const agent = new WorkerAgent(config, {
      coordinatorUrl: "http://127.0.0.1:9999",
      reconnect: false,
      hardwareProbe,
      logger: { info() {}, warn() {}, error() {} },
    });
    const harness = agent as unknown as MutableCapabilityHarness;
    harness.capabilities = await harness.buildCapabilities();
    harness.sendHeartbeat = async () => undefined;

    await agent.refreshRuntimeCapacity({
      status: "gpu-ready",
      backend: "cuda",
      deviceName: "GeForce RTX 4090",
    });
    expect(harness.capabilities?.gpus[0]).toMatchObject({
      vendor: "nvidia",
      physicalVramMb: 24_576,
      offeredVramMb: 24_576,
    });

    await agent.refreshRuntimeCapacity(undefined);
    expect(harness.capabilities?.gpus[0]).toMatchObject({
      vendor: "cpu",
      physicalVramMb: 0,
      sharedMemoryMb: 4_096,
      offeredVramMb: 4_096,
    });
  });

  it("keeps the newest runtime evidence when capacity probes overlap", async () => {
    type Hardware = Awaited<ReturnType<typeof hardwareProbe>>;
    let delayed = false;
    const pending: Array<(hardware: Hardware) => void> = [];
    const delayedHardwareProbe = async (): Promise<Hardware> => {
      if (!delayed) return hardwareProbe();
      return new Promise<Hardware>((resolve) => pending.push(resolve));
    };
    const agent = new WorkerAgent(config, {
      coordinatorUrl: "http://127.0.0.1:9999",
      reconnect: false,
      hardwareProbe: delayedHardwareProbe,
      logger: { info() {}, warn() {}, error() {} },
    });
    const harness = agent as unknown as MutableCapabilityHarness;
    harness.capabilities = await harness.buildCapabilities();
    harness.sendHeartbeat = async () => undefined;
    delayed = true;

    const staleGpuRefresh = agent.refreshRuntimeCapacity({
      status: "gpu-ready",
      backend: "cuda",
      deviceName: "GeForce RTX 4090",
    });
    const newestCpuRefresh = agent.refreshRuntimeCapacity(undefined);
    expect(pending).toHaveLength(2);
    const hardware = await hardwareProbe();
    pending[1]!(hardware);
    await newestCpuRefresh;
    pending[0]!(hardware);
    await staleGpuRefresh;

    expect(harness.capabilities?.gpus[0]).toMatchObject({
      vendor: "cpu",
      offeredVramMb: 4_096,
    });
  });
});
