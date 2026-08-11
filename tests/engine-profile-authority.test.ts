import { describe, expect, it } from "vitest";
import {
  sealEngineRuntimeProfile,
  type EngineRuntimeProfileInput,
} from "../src/contracts/engine-runtime-profile.js";
import { publishCoordinatorEngineRuntimeProfile } from "../src/coordinator/engine-profile-authority.js";
import type { MeshStore, StoredWorker } from "../src/storage/store.js";

describe("coordinator engine profile authority", () => {
  it("persists a profile bound to current physical and canary evidence", () => {
    const worker = workerFixture();
    let persisted: StoredWorker["capabilities"] | null = null;
    const store = {
      getWorker: () => worker,
      listEngineRuntimeActivationPlans: () => [activationPlan()],
      updateWorkerHeartbeat: (
        _workerId: string,
        capabilities: StoredWorker["capabilities"],
      ) => {
        persisted = capabilities;
      },
    } as unknown as MeshStore;
    const profile = profileFixture();

    publishCoordinatorEngineRuntimeProfile(store, worker.id, profile);

    expect(persisted!.distributedExecutor!.engineProfiles).toEqual([profile]);
  });

  it("rejects profiles detached from the current session or artifact", () => {
    const worker = workerFixture();
    const store = {
      getWorker: () => worker,
      listEngineRuntimeActivationPlans: () => [activationPlan()],
      updateWorkerHeartbeat: () => undefined,
    } as unknown as MeshStore;
    expect(() => publishCoordinatorEngineRuntimeProfile(store, worker.id, {
      ...profileFixture(),
      sessionId: "session-attacker",
    })).toThrow("engine_runtime_profile_seal_is_invalid");
    const wrongArtifact = sealEngineRuntimeProfile({
      ...profileInput(),
      artifactManifestDigest: digest("9"),
    });
    expect(() => publishCoordinatorEngineRuntimeProfile(
      store,
      worker.id,
      wrongArtifact,
    )).toThrow("engine_runtime_profile_evidence_binding_is_invalid");
  });
});

function workerFixture(): StoredWorker {
  return {
    id: "worker-a",
    status: "online",
    reliability: 1,
    jobsCompleted: 0,
    lastSeenAt: Date.now(),
    identityKind: "device",
    identityId: "node-a",
    capabilities: {
      region: "test",
      agentVersion: "0.2.99",
      buildIdentity: {
        schema: "mycellios-native-build-provenance/1",
        version: "0.2.99",
        sourceId: digest("d"),
      },
      gpus: [],
      limits: { maxConcurrency: 1, pauseWhenForeground: false },
      network: { coordinatorRttMs: 1, uplinkMbps: 100, downlinkMbps: 100 },
      deployments: [{
        deploymentId: "deployment-a",
        model: "Qwen/Qwen3-8B",
        modelDigest: digest("c"),
        activationId: "activation-a",
        mode: "pipeline",
        adapter: "mycellios-pipeline",
        peakVramMb: 16_384,
        contextLimit: 8_192,
        maxConcurrency: 1,
        freeSlots: 1,
        tokensPerSecond: 20,
        throughputSource: "measured",
        ttftMs: 100,
        verificationState: "verified",
        canaryEvidence: { evidenceId: digest("6") } as never,
        dataLocality: "local",
        stage: { index: 0, total: 1, layerStart: 0, layerEnd: 16 },
      }],
      distributedExecutor: {
        protocol: "gdlp-worker-tunnel/2",
        nodeId: "node-a",
        stageHost: "127.0.0.1",
        stagePort: 9_001,
        runtime: "python-safetensors",
        computeMode: "gpu-only",
        cpuEligible: false,
        physicalIdentity: {
          schema: "gdlp-worker-physical-identity/1",
          provider: "generic",
          providerMachineFingerprintSha256: digest("8"),
          hostFingerprintSha256: digest("e"),
          gpuFingerprintsSha256: [digest("9")],
          attestedAt: "2026-08-10T12:00:00.000Z",
        },
        performanceEvidence: {
          evidenceId: digest("7"),
          workerId: "worker-a",
          sessionId: "session-a",
          nodeId: "node-a",
          profile: { backend: "cuda" },
        } as never,
      },
    },
  };
}

function activationPlan() {
  const input = profileInput();
  return {
    modelId: input.modelId,
    workerId: input.workerId,
    evidence: { canaryEvidenceId: input.evidence.deploymentCanaryEvidenceId },
    request: {
      descriptorDigest: input.descriptorDigest,
      certificationId: input.certificationId,
      artifactManifestDigest: input.artifactManifestDigest,
      modelId: input.modelId,
      modelRevision: input.modelRevision,
      backend: input.backend,
      runtimeAbi: input.runtimeAbi,
      quantization: input.quantization,
      contextTokens: input.capacity.contextTokens,
      expectedLayerStart: 0,
      expectedLayerEnd: input.capacity.maxLayerCount,
      expectedKvBytesPerToken: input.capacity.kvBytesPerToken,
      requiredRoles: ["head"],
    },
  } as never;
}

function profileFixture() {
  return sealEngineRuntimeProfile(profileInput());
}

function profileInput(): EngineRuntimeProfileInput {
  return {
    descriptorDigest: digest("a"),
    certificationId: digest("b"),
    artifactManifestDigest: digest("c"),
    sourceId: digest("d"),
    hardwareFingerprintSha256: digest("e"),
    workerId: "worker-a",
    sessionId: "session-a",
    nodeId: "node-a",
    modelId: "Qwen/Qwen3-8B",
    modelRevision: "1".repeat(40),
    backend: "cuda" as const,
    runtimeAbi: "cuda-12",
    quantization: "bf16",
    measuredAt: "2026-08-10T12:00:00.000Z",
    expiresAt: "2026-08-11T12:00:00.000Z",
    samples: 21,
    confidenceHalfWidthPct: 8,
    capacity: {
      contextTokens: 8_192,
      maxLayerCount: 16,
      kvBytesPerToken: 2_048,
      maxKvTokens: 16_384,
      usableMemoryBytes: 16 * 1024 * 1024 * 1024,
    },
    costs: {
      decodeMsPerTokenP50: 4,
      decodeMsPerTokenP95: 5,
      prefillMsPerTokenP50: 0.2,
      prefillMsPerTokenP95: 0.3,
      verifyMsPerTokenP50: 3,
      verifyMsPerTokenP95: 4,
      decodeScale: 0.5,
      prefillScale: 0.75,
    },
    features: {
      fastKernel: true,
      graphMode: "available" as const,
      roles: ["head", "middle", "tail"],
    },
    evidence: {
      deploymentCanaryEvidenceId: digest("6"),
      runtimePerformanceEvidenceId: digest("7"),
    },
  };
}

function digest(seed: string): `sha256:${string}` {
  return `sha256:${seed.repeat(64)}`;
}
