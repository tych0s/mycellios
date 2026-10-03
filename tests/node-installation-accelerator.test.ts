import { describe, expect, it, vi } from "vitest";
import { prepareInstalledNodeAccelerator } from "../src/node/native-accelerator.js";
import type { AcceleratorProgressEvent, AcceleratorProgressIssueCode, AcceleratorRuntimeResult } from "../src/node/accelerator-runtime.js";

const cpu: AcceleratorRuntimeResult = { status: "cpu-ready", requestedBackend: "cpu", effectiveBackend: "cpu",
  deviceType: "cpu", runtimeRoot: "/fixture", pythonExecutable: "/fixture/python", pythonPathAdditions: [],
  pathAdditions: [], launchDevice: "cpu", deviceName: "CPU", precision: "float32", torchVersion: "fixture" };

describe("installer accelerator completion", () => {
  it.each(["network", "install", "disk-space", "integrity", "physical-probe"] as const)(
    "keeps %s GPU failures pending while CPU remains available", async (code) => {
      const prepare = async (input: Parameters<typeof prepareInstalledNodeAccelerator>[0]) => {
        input.onProgress?.(event(code)); return { ...cpu, status: "gpu-fallback" as const, requestedBackend: "cuda" as const };
      };
      await expect(prepareInstalledNodeAccelerator({ configPath: "/config", installRoot: "/install", allowProvisioning: true }, prepare))
        .rejects.toThrow(`node_install_gpu_setup_pending:${code}`);
    },
  );

  it("allows CPU-only and unsupported GPU nodes and forwards progress", async () => {
    const onProgress = vi.fn();
    const input = { configPath: "/config", installRoot: "/install", allowProvisioning: true, onProgress };
    await expect(prepareInstalledNodeAccelerator(input, async () => cpu)).resolves.toEqual(cpu);
    const prepare = async (options: Parameters<typeof prepareInstalledNodeAccelerator>[0]) => {
      options.onProgress?.(event("unsupported-gpu"));
      return { ...cpu, status: "gpu-fallback" as const, requestedBackend: "cuda" as const };
    };
    expect((await prepareInstalledNodeAccelerator(input, prepare)).status).toBe("gpu-fallback");
    expect(onProgress).toHaveBeenCalledOnce();
  });
});

function event(code: AcceleratorProgressIssueCode): AcceleratorProgressEvent {
  return { sequence: 1, at: new Date().toISOString(), phase: "fallback", backend: "cuda", gpuVendor: "nvidia",
    gpuModel: "fixture", cpuAvailable: true, percent: 100, message: "fixture", recordLog: true,
    issue: { code, message: "fixture", retryable: true, action: "retry" } };
}
