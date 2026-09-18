import { describe, expect, it } from "vitest";
import { isAvailableInferenceWorker, selectInferenceModel } from "./inference-selection";
import { inferenceModelOption } from "./Panel";

describe("chat model selection and observed capacity", () => {
  const models = [{ id: "busy", freeSlots: 0 }, { id: "available", freeSlots: 1 }];
  it("selects available capacity automatically but honors a deliberate model choice", () => {
    expect(selectInferenceModel(models, "")?.id).toBe("available");
    expect(selectInferenceModel(models, "busy")?.id).toBe("busy");
  });
  it("never substitutes a different model when the selected model disappears", () => {
    expect(selectInferenceModel(models, "disconnected")).toBeNull();
    expect(selectInferenceModel([], "")).toBeNull();
  });
  it("excludes disconnected, draining, quarantined, and explicitly unverified workers", () => {
    const online = { connected: true, status: "online" };
    expect(isAvailableInferenceWorker(online)).toBe(true);
    for (const unavailable of [{ connected: false }, { status: "draining" }, { status: "suspect" }, { quarantined: true }, { observedCapability: { eligibility: { serving: false } } }]) {
      expect(isAvailableInferenceWorker({ ...online, ...unavailable })).toBe(false);
    }
  });
  it("counts only eligible native deployments for the selected model", () => {
    const native = { deploymentId: "native", model: "model", mode: "replica", adapter: "mycellios-pipeline", freeSlots: 2, tokensPerSecond: 0 };
    const peer = { id: "a", kind: "desktop" as const, status: "online" as const, connected: true, offeredVramMb: 1_024, quarantined: false, quarantineExpiresAt: null, region: "test", reliability: 1, jobsCompleted: 0, lastSeenAt: new Date().toISOString(), gpus: [], deployments: [native] };
    const snapshot = { capturedAt: new Date().toISOString(), version: "test", buildIdentity: null, summary: { registered: 4, connected: 3, online: 3, mobile: 0, offeredVramMb: 4_096, completedJobs: 0 }, models: [], requestedModels: [], jobs: [], workers: [peer, { ...peer, id: "offline", connected: false }, { ...peer, id: "legacy", deployments: [{ ...native, adapter: "external-runtime", freeSlots: 100 }] }, { ...peer, id: "different", deployments: [{ ...native, model: "other", freeSlots: 100 }] }] };
    expect(inferenceModelOption(snapshot, { id: "model", replicas: 1, pipelines: 0 })).toMatchObject({ nativeRuntime: true, nodeCount: 1, freeSlots: 2, peerMemoryMb: 1_024 });
    expect(inferenceModelOption({ ...snapshot, workers: [] }, { id: "model", replicas: 1, pipelines: 0 })).toMatchObject({ nativeRuntime: false, nodeCount: 0, freeSlots: 0 });
  });
});
