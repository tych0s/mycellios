import { describe, expect, it } from "vitest";
import { selectDesktopHardwareGpu, selectWorkerCapacityHardware } from "../src/desktop/hardware-selection.js";
import { mergeHardwareGpuProbes, selectHardwareGpu, windowsGpuPhysicalVramMb } from "../src/worker/hardware.js";

describe("desktop hardware selection", () => {
  it("prefers a certified discrete adapter over the first integrated adapter", () => {
    const selected = selectDesktopHardwareGpu([
      {
        id: "gpu-0",
        vendor: "intel",
        model: "Intel UHD Graphics",
        physicalVramMb: 128,
        sharedMemoryMb: 8_192,
      },
      {
        id: "gpu-1",
        vendor: "amd",
        model: "AMD Radeon RX 9070 XT",
        physicalVramMb: 16_384,
      },
    ], {
      platform: "win32",
      arch: "x64",
      osRelease: "10.0.26100",
      cpuModel: "AMD Ryzen 9",
    });

    expect(selected?.id).toBe("gpu-1");
  });

  it("falls back deterministically when no native provider is certified", () => {
    const selected = selectDesktopHardwareGpu([
      { id: "gpu-0", vendor: "intel", model: "Intel Arc", physicalVramMb: 8_192 },
      { id: "gpu-1", vendor: "unknown", model: "Virtual display", physicalVramMb: 0 },
    ], {
      platform: "linux",
      arch: "x64",
      osRelease: "6.8.0",
    });

    expect(selected?.id).toBe("gpu-0");
  });

  it("advertises bounded real system memory when GPU memory is unknown", () => {
    const hardware = {
      hostname: "linux-node",
      platform: "linux" as const,
      ramMb: 16_384,
      gpus: [{ id: "gpu-0", vendor: "amd", model: "Radeon", physicalVramMb: 0 }],
    };
    const capacity = selectWorkerCapacityHardware(hardware, hardware.gpus[0], "Ryzen 9");
    expect(capacity).toMatchObject({
      id: "cpu-memory",
      vendor: "cpu",
      physicalVramMb: 0,
      sharedMemoryMb: 4_096,
    });
  });

  it("keeps a large detected GPU out of capacity until its runtime is physically ready", () => {
    const gpu = { id: "gpu-0", vendor: "nvidia", model: "NVIDIA GeForce RTX 4090", physicalVramMb: 24_576 };
    const hardware = {
      hostname: "linux-node",
      platform: "linux" as const,
      ramMb: 16_384,
      gpus: [gpu],
    };

    expect(selectWorkerCapacityHardware(hardware, gpu, "Ryzen 9")).toMatchObject({
      id: "cpu-memory",
      sharedMemoryMb: 4_096,
    });
    expect(selectWorkerCapacityHardware(hardware, gpu, "Ryzen 9", {
      status: "gpu-ready",
      backend: "cuda",
      deviceName: "NVIDIA GeForce RTX 4090",
    })).toBe(gpu);
  });

  it("rejects gpu-ready evidence for a different physical adapter", () => {
    const gpu = { id: "gpu-0", vendor: "amd", model: "AMD Radeon RX 9070 XT", physicalVramMb: 16_384 };
    const hardware = {
      hostname: "windows-node",
      platform: "win32" as const,
      ramMb: 32_768,
      gpus: [gpu],
    };
    expect(selectWorkerCapacityHardware(hardware, gpu, "Ryzen 9", {
      status: "gpu-ready",
      backend: "rocm",
      deviceName: "AMD Radeon 890M Graphics",
    })).toMatchObject({ id: "cpu-memory", sharedMemoryMb: 8_192 });
  });
});

describe("Windows VRAM parsing", () => {
  it("uses the 64-bit driver value for GPUs above four GiB", () => {
    expect(windowsGpuPhysicalVramMb({
      AdapterRAM: 4_294_967_295,
      DedicatedBytes: 16 * 1_024 ** 3,
    })).toBe(16_384);
  });

  it("retains AdapterRAM as a compatibility fallback", () => {
    expect(windowsGpuPhysicalVramMb({ AdapterRAM: 2 * 1_024 ** 3 })).toBe(2_048);
  });
});

describe("worker GPU advertisement", () => {
  it("keeps the runtime-selected GPU when the OS reports another adapter first", () => {
    const gpus = [
      { id: "gpu-0", vendor: "intel", model: "Intel UHD", physicalVramMb: 128 },
      { id: "gpu-1", vendor: "amd", model: "AMD Radeon RX 9070 XT", physicalVramMb: 16_384 },
    ];
    expect(selectHardwareGpu(gpus, {
      vendor: "AMD",
      model: "amd radeon rx 9070 xt",
    })?.id).toBe("gpu-1");
  });

  it("does not trust a regenerated synthetic id without matching GPU identity", () => {
    const gpus = [
      { id: "gpu-0", vendor: "intel", model: "Intel UHD", physicalVramMb: 128 },
      { id: "gpu-1", vendor: "amd", model: "AMD Radeon RX 9070 XT", physicalVramMb: 16_384 },
    ];
    expect(selectHardwareGpu(gpus, {
      id: "gpu-0",
      vendor: "amd",
      model: "AMD Radeon RX 9070 XT",
    })?.id).toBe("gpu-1");
  });

  it("does not borrow live telemetry when the selected adapter disappears", () => {
    const gpus = [
      { id: "gpu-0", vendor: "intel", model: "Intel UHD", physicalVramMb: 128 },
    ];
    expect(selectHardwareGpu(gpus, {
      id: "gpu-0",
      vendor: "nvidia",
      model: "NVIDIA GeForce RTX 4090",
    })).toBeUndefined();
  });

  it("merges NVIDIA telemetry without hiding other vendors", () => {
    const merged = mergeHardwareGpuProbes([
      { id: "native-0", vendor: "intel", model: "Intel UHD", physicalVramMb: 128 },
      { id: "native-1", vendor: "nvidia", model: "NVIDIA GeForce RTX 4090", physicalVramMb: 0 },
      { id: "native-2", vendor: "amd", model: "AMD Radeon RX 9070 XT", physicalVramMb: 16_384 },
    ], [
      { id: "smi-0", vendor: "nvidia", model: "NVIDIA GeForce RTX 4090", runtimeDeviceIndex: 0, physicalVramMb: 24_564, utilizationPct: 12 },
    ]);
    expect(merged.map((gpu) => gpu.vendor)).toEqual(["intel", "nvidia", "amd"]);
    expect(merged[1]).toMatchObject({ physicalVramMb: 24_564, utilizationPct: 12 });
    expect(merged.map((gpu) => gpu.runtimeDeviceIndex)).toEqual([undefined, 0, undefined]);
  });
});
