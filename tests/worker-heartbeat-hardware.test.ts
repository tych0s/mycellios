import { describe, expect, it, vi } from "vitest";
import { workerConfigSchema } from "../src/contracts/schemas.js";
import type { WorkerCapabilities } from "../src/contracts/types.js";
import { WorkerAgent } from "../src/worker/agent.js";
import type { HardwareProbe } from "../src/worker/hardware.js";
import { createHeartbeatHardwareSampler } from "../src/worker/heartbeat-hardware.js";

describe("worker heartbeat during hardware telemetry stalls", () => {
  it("recovers after an unavailable OS telemetry sample without inventing one", async () => {
    vi.useFakeTimers();
    try {
      const hardware: HardwareProbe = {
        hostname: "recovered-probe", platform: "win32", ramMb: 16_384, gpus: [],
      };
      const probe = vi.fn().mockRejectedValueOnce(new Error("OS probe unavailable"))
        .mockResolvedValue(hardware);
      const sample = createHeartbeatHardwareSampler(probe);
      expect(sample()).toBeNull();
      await vi.advanceTimersByTimeAsync(0);
      expect(sample()).toBeNull();
      await vi.advanceTimersByTimeAsync(0);
      expect(sample()).toEqual(hardware);
    } finally {
      vi.useRealTimers();
    }
  });
  it("keeps sending liveness while one telemetry probe is unresolved", async () => {
    vi.useFakeTimers();
    const hardware: HardwareProbe = {
      hostname: "heartbeat-test", platform: "win32", ramMb: 16_384,
      gpus: [{ id: "gpu-0", vendor: "nvidia", model: "RTX test", physicalVramMb: 6_144 }],
    };
    let resolveProbe!: (value: HardwareProbe) => void;
    const pendingProbe = new Promise<HardwareProbe>((resolve) => { resolveProbe = resolve; });
    const probe = vi.fn().mockResolvedValueOnce(hardware).mockReturnValue(pendingProbe);
    const agent = new WorkerAgent(workerConfigSchema.parse({
      region: "test", capacityScope: "host", offeredVramMb: 4_096,
      limits: { maxConcurrency: 1, pauseWhenForeground: false },
      adapter: { kind: "mycellios-native", model: "mycellios-native-control" },
      deployment: { contextLimit: 2_048 },
    }), {
      coordinatorUrl: "http://127.0.0.1:9999", advertiseDeployment: false,
      verifiedGpuRuntime: { status: "gpu-ready", backend: "cuda", deviceName: "RTX test" },
      hardwareProbe: probe, logger: { info() {}, warn() {}, error() {} },
    });
    const harness = agent as unknown as {
      buildCapabilities(): Promise<WorkerCapabilities>;
      capabilities: WorkerCapabilities;
      registeredWorkerId: string;
      socket: unknown;
      sendHeartbeat(): Promise<void>;
    };
    const sent: Array<{ type: string; payload: { heartbeat: { gpus: unknown[] } } }> = [];
    try {
      harness.capabilities = await harness.buildCapabilities();
      harness.registeredWorkerId = "worker-heartbeat-test";
      harness.socket = { readyState: 1, bufferedAmount: 0,
        send: (value: string) => { sent.push(JSON.parse(value)); } };
      const first = harness.sendHeartbeat();
      await vi.advanceTimersByTimeAsync(1_000);
      expect(sent).toHaveLength(1);
      await harness.sendHeartbeat();
      expect(sent).toHaveLength(2);
      expect(probe).toHaveBeenCalledTimes(2); // Initial capacity plus one live sample.
      resolveProbe({ ...hardware, gpus: [{ ...hardware.gpus[0]!, utilizationPct: 67 }] });
      await first;
      await vi.advanceTimersByTimeAsync(0);
      await harness.sendHeartbeat();
      expect(sent[2]?.payload.heartbeat.gpus[0]).toMatchObject({ utilizationPct: 67 });
      expect(sent.every((message) => message.type === "worker.heartbeat")).toBe(true);
    } finally {
      resolveProbe(hardware);
      vi.useRealTimers();
    }
  });
});
