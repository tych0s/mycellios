import { describe, expect, it } from "vitest";
import { resolve } from "node:path";
import type { PhysicalProbeV1 } from "../src/distribution/physical-probe.js";
import type {
  LaunchAgent,
  LaunchAgentStartRequest,
  LaunchProcessHandle,
} from "../src/distribution/launch-supervisor.js";
import type {
  PythonLaunchProcess,
  PythonPipelineLaunchDescription,
} from "../src/distribution/python-launcher.js";
import {
  buildPhysicalIdentity,
  createHeadlessRuntime,
  HeadlessStageLaunchAgent,
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
      "headless_worker_external_coordinator_requires_scoped_credential_or_32_character_network_token",
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
      resolve("/opt/mycellios", "./config/worker.gpu_cloud.example.json"),
    );
    const scoped = loadHeadlessWorkerEnvironment({
      GPU_CLOUD_MACHINE_ID: machineId,
      GPU_MESH_COORDINATOR: "https://www.mycellios.com",
      MYCELLIOS_WORKER_CREDENTIAL_PATH: "/var/lib/mycellios/worker-credential.json",
    }, "/opt/mycellios");
    expect(scoped.networkToken).toBeUndefined();
    expect(scoped.workerCredentialPath).toBe(
      resolve("/opt/mycellios", "/var/lib/mycellios/worker-credential.json"),
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
    expect(runtime.executor.launchAgent).toBeInstanceOf(HeadlessStageLaunchAgent);
  });

  it("prepares this node's assigned artifacts before delegating process start", async () => {
    const calls: string[] = [];
    const base: LaunchAgent = {
      id: "base",
      async start(_request: LaunchAgentStartRequest, _signal: AbortSignal): Promise<LaunchProcessHandle> {
        calls.push("start");
        return {
          ready: Promise.resolve(),
          exited: Promise.resolve({ code: 0, signal: null }),
          stop: async () => undefined,
        };
      },
    };
    const localProcess = {
      processId: "stage-1-node-a",
      anchor: { memberId: "node-a" },
      stageIndex: 1,
      layerStart: 4,
      layerEnd: 8,
    } as PythonLaunchProcess;
    const description = {
      launchOrder: [localProcess, {
        ...localProcess,
        processId: "stage-0-node-b",
        anchor: { memberId: "node-b" },
        stageIndex: 0,
        layerStart: 0,
        layerEnd: 4,
      }],
    } as PythonPipelineLaunchDescription;
    const progress: number[] = [];
    const agent = new HeadlessStageLaunchAgent(
      base,
      {
        nodeId: "node-a",
        pythonExecutable: "/opt/mycellios/python",
        pythonPath: "/opt/mycellios/runtime",
        cachePath: "/var/cache/mycellios",
      },
      async (received, options) => {
        calls.push("prepare");
        expect(received).toBe(description);
        expect(options).toMatchObject({
          nodeId: "node-a",
          pythonExecutable: "/opt/mycellios/python",
          cacheDirectory: "/var/cache/mycellios",
          environment: {
            PYTHONPATH: "/opt/mycellios/runtime",
            HF_HOME: "/var/cache/mycellios",
          },
        });
        options.onProgress?.({
          stageIndex: 1,
          layerStart: 4,
          layerEnd: 8,
          state: "ready",
          packageId: "a".repeat(64),
          weightsSizeBytes: 4096,
        });
        return [localProcess];
      },
    );
    const prepared = await agent.prepareRuntime(description, "node-a", (event) => {
      progress.push(event.weightsSizeBytes ?? 0);
    });
    await agent.start({
      launchId: "launch-1",
      pipelineId: "pipeline-1",
      deploymentGeneration: 0,
      nodeId: "node-a",
      process: prepared[0]!,
    }, new AbortController().signal);
    expect(prepared).toEqual([localProcess]);
    expect(progress).toEqual([4096]);
    expect(calls).toEqual(["prepare", "start"]);
    await expect(agent.prepareRuntime(description, "node-b")).rejects.toThrow(
      "headless_stage_artifact_node_mismatch",
    );
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
