import { describe, expect, it } from "vitest";
import type { PhysicalProbeV1 } from "../src/distribution/physical-probe.js";
import {
  buildPhysicalIdentity,
  createHeadlessRuntime,
  loadHeadlessWorkerEnvironment,
} from "../src/worker/headless-runtime.js";

const machineId = "5ab6bf90-3142-4eb0-a4f6-16defe36dca1";
const probe: PhysicalProbeV1 = {
  schema: "gdlp-physical-probe/1",
  nonce: "fixture-nonce-0001",
  host: {
    fingerprintSha256: `sha256:${"1".repeat(64)}`,
    fingerprintSource: "linux-machine-id",
    platform: "linux",
    architecture: "x86_64",
    kernelRelease: "6.8.0",
    pythonVersion: "3.12.13",
  },
  runtime: {
    torchVersion: "2.13.0+cu126",
    cudaVersion: "12.6",
    rocmVersion: null,
    cudaApiAvailable: true,
    distributedAvailable: true,
    ncclAvailable: true,
    ncclVersion: "2.27.3",
  },
  devices: [{
    index: 0,
    name: "NVIDIA GeForce RTX 4090",
    totalMemoryBytes: 24_576 * 1024 * 1024,
    freeMemoryBytes: 23_000 * 1024 * 1024,
    runtimeTotalMemoryBytes: 24_576 * 1024 * 1024,
    capability: [8, 9],
    uuidSha256: `sha256:${"2".repeat(64)}`,
    fingerprintSha256: `sha256:${"3".repeat(64)}`,
  }],
};

describe("headless GpuCloud worker", () => {
  it("derives a private stable node id and requires authenticated HTTPS externally", () => {
    expect(() => loadHeadlessWorkerEnvironment({
      GPU_CLOUD_MACHINE_ID: machineId,
      GPU_MESH_COORDINATOR: "https://www.mycellios.com",
    }, "/opt/mycellios")).toThrow(
      "headless_worker_external_coordinator_requires_32_character_network_token",
    );

    const environment = loadHeadlessWorkerEnvironment({
      GPU_CLOUD_MACHINE_ID: machineId,
      GPU_MESH_COORDINATOR: "https://www.mycellios.com",
      MYCELLIOS_NETWORK_TOKEN: "n".repeat(32),
    }, "/opt/mycellios");
    expect(environment.nodeId).toMatch(/^gpu_cloud-[0-9a-f]{32}$/);
    expect(environment.nodeId).not.toContain(machineId);
    expect(environment.provider).toBe("gpu_cloud");
    expect(environment.configPath).toBe(
      "/opt/mycellios/config/worker.gpu_cloud.example.json",
    );
  });

  it("allows explicit insecure transport only for loopback development", () => {
    const local = loadHeadlessWorkerEnvironment({
      MYCELLIOS_NODE_ID: "local-headless",
      GPU_MESH_COORDINATOR: "http://127.0.0.1:8787",
      MYCELLIOS_ALLOW_INSECURE_COORDINATOR: "true",
    }, "/workspace");
    expect(local.networkToken).toBeUndefined();

    expect(() => loadHeadlessWorkerEnvironment({
      MYCELLIOS_NODE_ID: "external-headless",
      GPU_MESH_COORDINATOR: "http://example.com",
      MYCELLIOS_ALLOW_INSECURE_COORDINATOR: "true",
      MYCELLIOS_NETWORK_TOKEN: "n".repeat(32),
    }, "/workspace")).toThrow("headless_worker_external_coordinator_requires_https");
  });

  it("publishes only hashed provider and GPU identity", () => {
    const identity = buildPhysicalIdentity(
      { provider: "gpu_cloud", providerMachineId: machineId },
      probe,
      "2026-07-24T00:00:00.000Z",
    );
    expect(identity.providerMachineFingerprintSha256).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(identity.hostFingerprintSha256).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(identity.hostFingerprintSha256).not.toBe(probe.host.fingerprintSha256);
    expect(identity.gpuFingerprintsSha256).toHaveLength(1);
    expect(identity.gpuFingerprintsSha256[0]).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(identity.gpuFingerprintsSha256[0]).not.toBe(
      probe.devices[0]!.fingerprintSha256,
    );
    expect(JSON.stringify(identity)).not.toContain(machineId);
    expect(JSON.stringify(identity)).not.toContain(probe.devices[0]!.uuidSha256!);
  });

  it("creates a gpu-only relay executor after the physical CUDA probe", async () => {
    const environment = loadHeadlessWorkerEnvironment({
      GPU_CLOUD_MACHINE_ID: machineId,
      GPU_MESH_COORDINATOR: "https://www.mycellios.com",
      MYCELLIOS_NETWORK_TOKEN: "n".repeat(32),
      MYCELLIOS_VERSION: "0.2.99",
    }, "/opt/mycellios");
    const runtime = await createHeadlessRuntime(
      environment,
      { collect: async () => structuredClone(probe) },
      () => new Date("2026-07-24T00:00:00.000Z"),
    );
    expect(runtime.executor).toMatchObject({
      nodeId: environment.nodeId,
      stageHost: `${environment.nodeId}.relay`,
      stagePort: 9_850,
      computeMode: "gpu-only",
      cpuEligible: false,
    });
    expect(runtime.verifiedGpuRuntime).toEqual({
      status: "gpu-ready",
      backend: "cuda",
      deviceName: "NVIDIA GeForce RTX 4090",
    });
    expect(runtime.acceleration).toMatchObject({
      appVersion: "0.2.99",
      state: "gpu-ready",
      phase: "ready",
    });
  });

  it("fails closed when the container cannot prove a CUDA device", async () => {
    const environment = loadHeadlessWorkerEnvironment({
      GPU_CLOUD_MACHINE_ID: machineId,
      GPU_MESH_COORDINATOR: "https://www.mycellios.com",
      MYCELLIOS_NETWORK_TOKEN: "n".repeat(32),
    }, "/opt/mycellios");
    const cpuProbe = structuredClone(probe);
    cpuProbe.runtime.cudaApiAvailable = false;
    cpuProbe.runtime.ncclAvailable = false;
    cpuProbe.devices = [];
    await expect(createHeadlessRuntime(
      environment,
      { collect: async () => cpuProbe },
    )).rejects.toThrow("headless_worker_requires_verified_cuda_device");
  });
});
