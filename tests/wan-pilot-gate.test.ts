import { describe, expect, it } from "vitest";
import { verifyWanPilotSnapshot } from "../src/distribution/wan-pilot-gate.js";

const digest = (character: string) => `sha256:${character.repeat(64)}`;

function identity(
  provider: "gpu_cloud" | "generic",
  providerFingerprint: string,
  hostFingerprint: string,
  gpuFingerprint: string,
) {
  return {
    schema: "gdlp-worker-physical-identity/1",
    provider,
    providerMachineFingerprintSha256: providerFingerprint,
    hostFingerprintSha256: hostFingerprint,
    gpuFingerprintsSha256: [gpuFingerprint],
    attestedAt: "2026-07-24T00:00:00.000Z",
  };
}

function snapshot() {
  return {
    capturedAt: "2026-07-24T00:05:00.000Z",
    models: [{ id: "Qwen/Qwen3-0.6B" }],
    jobs: [{
      model: "Qwen/Qwen3-0.6B",
      status: "completed",
      workerId: "pipeline-cell",
    }],
    workers: [
      {
        id: "gpu_cloud-worker-a",
        status: "online",
        connected: true,
        executionNodeId: "gpu_cloud-a",
        physicalIdentity: identity("gpu_cloud", digest("1"), digest("2"), digest("3")),
        deployments: [],
      },
      {
        id: "gpu_cloud-worker-b",
        status: "online",
        connected: true,
        executionNodeId: "gpu_cloud-b",
        physicalIdentity: identity("gpu_cloud", digest("4"), digest("5"), digest("6")),
        deployments: [],
      },
      {
        id: "pipeline-cell",
        status: "online",
        connected: true,
        deployments: [{
          model: "Qwen/Qwen3-0.6B",
          internalPipeline: {
            stageCount: 2,
            boundaries: [0, 14, 28],
          },
          execution: {
            fallback: false,
            stages: [
              {
                nodeId: "gpu_cloud-a",
                stageIndex: 0,
                layerStart: 0,
                layerEnd: 14,
                deviceType: "gpu",
                backend: "cuda",
                fallback: false,
              },
              {
                nodeId: "gpu_cloud-b",
                stageIndex: 1,
                layerStart: 14,
                layerEnd: 28,
                deviceType: "gpu",
                backend: "cuda",
                fallback: false,
              },
            ],
          },
        }],
      },
    ],
  };
}

describe("WAN physical pipeline gate", () => {
  it("accepts a completed model crossing two unique attested GpuCloud GPUs", () => {
    const report = verifyWanPilotSnapshot(snapshot(), {
      model: "Qwen/Qwen3-0.6B",
    });
    expect(report).toMatchObject({
      passed: true,
      stageCount: 2,
      physicalNodeCount: 2,
      gpu_cloudNodeCount: 2,
      gpuFingerprintCount: 2,
    });
    expect(report.nodes.map((node) => node.nodeId)).toEqual(["gpu_cloud-a", "gpu_cloud-b"]);
  });

  it("rejects duplicate physical machines hidden behind different worker ids", () => {
    const value = snapshot();
    value.workers[1]!.physicalIdentity =
      structuredClone(value.workers[0]!.physicalIdentity!);
    expect(() => verifyWanPilotSnapshot(value, {
      model: "Qwen/Qwen3-0.6B",
    })).toThrow("wan_pilot_physical_machine_fingerprints_are_not_unique");
  });

  it("rejects fallback and unattested stage execution", () => {
    const fallback = snapshot();
    fallback.workers[2]!.deployments[0]!.execution!.fallback = true;
    expect(() => verifyWanPilotSnapshot(fallback, {
      model: "Qwen/Qwen3-0.6B",
    })).toThrow("wan_pilot_fallback_execution_is_not_evidence");

    const unattested = snapshot();
    delete unattested.workers[1]!.physicalIdentity;
    expect(() => verifyWanPilotSnapshot(unattested, {
      model: "Qwen/Qwen3-0.6B",
    })).toThrow("wan_pilot_stage_worker_is_not_physically_attested:gpu_cloud-b");
  });

  it("rejects a model with no completed inference or non-contiguous layers", () => {
    const noJob = snapshot();
    noJob.jobs = [];
    expect(() => verifyWanPilotSnapshot(noJob, {
      model: "Qwen/Qwen3-0.6B",
    })).toThrow("wan_pilot_pipeline_has_no_completed_job");

    const wrongWorker = snapshot();
    wrongWorker.jobs[0]!.workerId = "gpu_cloud-worker-a";
    expect(() => verifyWanPilotSnapshot(wrongWorker, {
      model: "Qwen/Qwen3-0.6B",
    })).toThrow("wan_pilot_pipeline_has_no_completed_job");

    const brokenLayers = snapshot();
    brokenLayers.workers[2]!.deployments[0]!.execution!.stages![1]!.layerStart = 15;
    expect(() => verifyWanPilotSnapshot(brokenLayers, {
      model: "Qwen/Qwen3-0.6B",
    })).toThrow("wan_pilot_layer_ranges_do_not_match_boundaries");
  });
});
