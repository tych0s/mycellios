import { describe, expect, it, vi } from "vitest";
import {
  MobileExecutionCancelledError,
  mobileCpuFallbackLabel,
  mobileGpuAdapterLabel,
  requestMobileGpuAdapter,
  runWithMobileBackendFallback,
} from "../src/mobile/backend-state.js";

describe("mobile backend state", () => {
  it("uses a real compatibility adapter for preview when available", async () => {
    const adapter = { info: {} } as never;
    const requestAdapter = vi.fn().mockResolvedValue(adapter);

    await expect(requestMobileGpuAdapter({ requestAdapter })).resolves.toBe(adapter);
    expect(requestAdapter).toHaveBeenCalledTimes(1);
    expect(requestAdapter).toHaveBeenCalledWith({ featureLevel: "compatibility" });
  });

  it("falls back to a high-performance probe when compatibility is rejected", async () => {
    const adapter = { info: {} } as never;
    const requestAdapter = vi.fn()
      .mockRejectedValueOnce(new TypeError("unknown option"))
      .mockResolvedValueOnce(adapter);

    await expect(requestMobileGpuAdapter({ requestAdapter })).resolves.toBe(adapter);
    expect(requestAdapter).toHaveBeenNthCalledWith(2, { powerPreference: "high-performance" });
  });

  it("reports no adapter only after both real probes fail", async () => {
    const requestAdapter = vi.fn().mockResolvedValue(null);

    await expect(requestMobileGpuAdapter({ requestAdapter })).resolves.toBeNull();
    expect(requestAdapter).toHaveBeenCalledTimes(2);
  });

  it("runs CPU and reports the effective backend when a GPU kernel fails", async () => {
    const onGpuFailure = vi.fn();
    const execution = await runWithMobileBackendFallback({
      backend: "webgpu",
      runGpu: async () => { throw new Error("device lost"); },
      runCpu: async () => "cpu-result",
      onGpuFailure,
    });

    expect(execution).toEqual({ backend: "cpu", value: "cpu-result" });
    expect(onGpuFailure).toHaveBeenCalledWith(expect.objectContaining({ message: "device lost" }));
  });

  it("does not invoke CPU when the GPU canary succeeds", async () => {
    const runCpu = vi.fn(async () => "cpu-result");
    await expect(runWithMobileBackendFallback({
      backend: "webgpu",
      runGpu: async () => "gpu-result",
      runCpu,
      onGpuFailure: vi.fn(),
    })).resolves.toEqual({ backend: "webgpu", value: "gpu-result" });
    expect(runCpu).not.toHaveBeenCalled();
  });

  it("does not touch the GPU path when CPU is already effective", async () => {
    const runGpu = vi.fn(async () => "gpu-result");
    await expect(runWithMobileBackendFallback({
      backend: "cpu",
      runGpu,
      runCpu: async () => "cpu-result",
      onGpuFailure: vi.fn(),
    })).resolves.toEqual({ backend: "cpu", value: "cpu-result" });
    expect(runGpu).not.toHaveBeenCalled();
  });

  it("never continues on CPU when contribution is cancelled during GPU work", async () => {
    const controller = new AbortController();
    const runCpu = vi.fn(async () => "cpu-result");
    const onGpuFailure = vi.fn();

    await expect(runWithMobileBackendFallback({
      backend: "webgpu",
      runGpu: async () => {
        controller.abort(new MobileExecutionCancelledError("page hidden"));
        throw new Error("device destroyed");
      },
      runCpu,
      onGpuFailure,
      signal: controller.signal,
    })).rejects.toBeInstanceOf(MobileExecutionCancelledError);

    expect(runCpu).not.toHaveBeenCalled();
    expect(onGpuFailure).not.toHaveBeenCalled();
  });

  it("discards a completed CPU result if contribution was stopped meanwhile", async () => {
    const controller = new AbortController();

    await expect(runWithMobileBackendFallback({
      backend: "cpu",
      runGpu: vi.fn(async () => "gpu-result"),
      runCpu: async () => {
        controller.abort(new MobileExecutionCancelledError("user stopped contribution"));
        return "cpu-result";
      },
      onGpuFailure: vi.fn(),
      signal: controller.signal,
    })).rejects.toBeInstanceOf(MobileExecutionCancelledError);
  });

  it("keeps the detected adapter visible while clearly labelling CPU fallback", () => {
    expect(mobileGpuAdapterLabel({
      vendor: "apple",
      architecture: "m3",
      device: "",
      description: "",
    })).toBe("apple · m3");
    expect(mobileCpuFallbackLabel("Apple M3 GPU")).toBe("CPU fallback · Apple M3 GPU");
  });
});
