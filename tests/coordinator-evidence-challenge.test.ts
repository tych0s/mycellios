import { describe, expect, it } from "vitest";
import {
  createCoordinatorDeploymentCanaryEvidence,
} from "../src/contracts/deployment-canary.js";
import type {
  WorkerCapabilities,
  WorkerEnvelope,
} from "../src/contracts/types.js";
import {
  mergeCurrentSessionEvidence,
} from "../src/coordinator/evidence-authority.js";
import { WorkerHub, type EngineRuntimeChallengeRequest } from "../src/coordinator/worker-hub.js";
import type { MeshStore, StoredWorker } from "../src/storage/store.js";
import {
  sealRuntimePerformanceProfile,
} from "../src/performance/runtime-profile.js";
import { Scheduler } from "../src/scheduler/scheduler.js";

describe("coordinator evidence authority", () => {
  it("keeps a pipeline unschedulable until it observes every challenged sample", () => {
    const worker = storedWorker(pipelineCapabilities());
    const { hub, socket, state, current } = harness(worker);
    const scheduler = new Scheduler({
      listSchedulableWorkers: () => [current.worker],
      countActiveJobs: () => 0,
      getSessionRoute: () => null,
    } as unknown as MeshStore);

    expect(scheduler.listAvailableModels()).toEqual([]);
    issueChallenges(hub, state, worker.capabilities);
    const challenge = socket.sent.find(
      (message) => message.type === "evidence.challenge"
        && (message.payload as { kind?: string }).kind === "deployment-canary",
    )!.payload as Record<string, unknown>;

    for (let sampleIndex = 0; sampleIndex < 3; sampleIndex += 1) {
      deliver(hub, state, "evidence.canary.started", {
        ...binding(challenge),
        sampleIndex,
      });
      deliver(hub, state, "evidence.canary.token", {
        ...binding(challenge),
        sampleIndex,
        index: 0,
        text: "measured answer",
      });
      deliver(hub, state, "evidence.canary.complete", {
        ...binding(challenge),
        sampleIndex,
        outputTokens: 4,
        finishReason: "stop",
      });
    }

    const deployment = current.worker.capabilities.deployments[0]!;
    expect(deployment).toMatchObject({
      verificationState: "verified",
      throughputSource: "measured",
      canaryEvidence: {
        challengeId: challenge.challengeId,
        nonce: challenge.nonce,
        workerId: worker.id,
        sessionId: challenge.sessionId,
      },
    });
    expect(deployment.tokensPerSecond).toBeGreaterThan(0);
    expect(scheduler.listAvailableModels()).toEqual([
      { id: "qwen-test", replicas: 1, pipelines: 0 },
    ]);

    deliver(hub, state, "evidence.canary.complete", {
      ...binding(challenge),
      sampleIndex: 2,
      outputTokens: 4,
      finishReason: "stop",
    });
    expect(socket.closed.at(-1)).toMatchObject({
      code: 4403,
      reason: "invalid or replayed evidence response",
    });
    hub.close();
  });

  it("publishes a runtime profile only after the directed current-session challenge", () => {
    const capabilities = executorCapabilities();
    const worker = storedWorker(capabilities);
    const { hub, socket, state, current } = harness(worker);
    issueChallenges(hub, state, capabilities);
    const challenge = socket.sent.find(
      (message) => message.type === "evidence.challenge"
        && (message.payload as { kind?: string }).kind === "runtime-performance",
    )!.payload as Record<string, unknown>;
    const profile = sealRuntimePerformanceProfile({
      measuredAt: new Date().toISOString(),
      backend: "cuda",
      deviceName: "NVIDIA Test GPU",
      precision: "float16",
      source: "physical-microbenchmark",
      activationCodecId: "fp16",
      decodeMemory: series("GB/s", 400),
      prefillCompute: series("TFLOP/s", 20),
      activationCodec: series("GB/s", 2),
    });

    deliver(hub, state, "evidence.runtime.complete", {
      ...binding(challenge),
      profile,
    });
    expect(current.worker.capabilities.distributedExecutor?.performanceEvidence)
      .toMatchObject({
        challengeId: challenge.challengeId,
        workerId: worker.id,
        sessionId: challenge.sessionId,
        nodeId: "node-test",
        profile: { profileId: profile.profileId },
      });

    const nextSessionCapabilities = current.worker.capabilities;
    const nextState = {
      ...state,
      sessionId: "session-reconnected",
    };
    const merged = (
      hub as unknown as {
        applyHeartbeat(
          connection: typeof state,
          payload: {
            heartbeat: {
              draining: boolean;
              pausedReason: null;
              activeLeases: string[];
              gpus: Array<{ id: string; freeOfferedVramMb: number }>;
              deployments: Array<{ deploymentId: string; freeSlots: number }>;
              network: { coordinatorRttMs: number; uplinkMbps: number };
            };
            capabilities: WorkerCapabilities;
            metrics: { ready: boolean; activeJobs: number; loadedModels: string[] };
          },
        ): void;
      }
    );
    merged.applyHeartbeat(nextState, {
      heartbeat: {
        draining: false,
        pausedReason: null,
        activeLeases: [],
        gpus: [{ id: "gpu-0", freeOfferedVramMb: 4_096 }],
        deployments: [],
        network: { coordinatorRttMs: 1, uplinkMbps: 100 },
      },
      capabilities: nextSessionCapabilities,
      metrics: { ready: true, activeJobs: 0, loadedModels: [] },
    });
    expect(current.worker.capabilities.distributedExecutor?.performanceEvidence)
      .toBeUndefined();
    hub.close();
  });

  it("revokes coordinator observations when the executable source changes", () => {
    const current = pipelineCapabilities();
    current.buildIdentity = buildIdentity("a");
    current.deployments[0] = {
      ...current.deployments[0]!,
      verificationState: "verified",
      throughputSource: "measured",
      tokensPerSecond: 12,
      ttftMs: 250,
      canaryEvidence: deploymentEvidence(),
    };
    const incoming = structuredClone(current);
    incoming.buildIdentity = buildIdentity("b");

    const merged = mergeCurrentSessionEvidence(
      "worker-test",
      "session-test",
      incoming,
      current,
    );

    expect(merged.deployments[0]).toMatchObject({
      verificationState: "pending",
      throughputSource: "default",
      tokensPerSecond: 1,
      ttftMs: 60_000,
    });
    expect(merged.deployments[0]?.canaryEvidence).toBeUndefined();
  });

  it("challenges the loaded engine and publishes a bound runtime profile", () => {
    const capabilities = executorCapabilities();
    capabilities.buildIdentity = buildIdentity("d");
    capabilities.distributedExecutor!.physicalIdentity = {
      schema: "gdlp-worker-physical-identity/1",
      provider: "generic",
      providerMachineFingerprintSha256: `sha256:${"8".repeat(64)}`,
      hostFingerprintSha256: `sha256:${"e".repeat(64)}`,
      gpuFingerprintsSha256: [`sha256:${"9".repeat(64)}`],
      attestedAt: new Date().toISOString(),
    };
    capabilities.deployments = [{
      ...pipelineCapabilities().deployments[0]!,
      mode: "pipeline",
      throughputSource: "measured",
      verificationState: "verified",
      tokensPerSecond: 20,
      ttftMs: 100,
      canaryEvidence: deploymentEvidence(),
      stage: { index: 0, total: 1, layerStart: 0, layerEnd: 16 },
      contextLimit: 8_192,
    }];
    const worker = storedWorker(capabilities);
    const { hub, socket, state, current } = harness(worker);
    issueChallenges(hub, state, capabilities);
    const runtimeChallenge = socket.sent.find(
      (message) => message.type === "evidence.challenge"
        && (message.payload as { kind?: string }).kind === "runtime-performance",
    )!.payload as Record<string, unknown>;
    const profile = sealRuntimePerformanceProfile({
      measuredAt: new Date().toISOString(),
      backend: "cuda",
      deviceName: "NVIDIA Test GPU",
      precision: "float16",
      source: "physical-microbenchmark",
      activationCodecId: "fp16",
      decodeMemory: series("GB/s", 400),
      prefillCompute: series("TFLOP/s", 20),
      activationCodec: series("GB/s", 2),
    });
    deliver(hub, state, "evidence.runtime.complete", {
      ...binding(runtimeChallenge),
      profile,
    });

    const engineRequest = {
      descriptorDigest: `sha256:${"a".repeat(64)}`,
      probeKind: "qwen3-dense-v1",
      certificationId: `sha256:${"b".repeat(64)}`,
      artifactManifestDigest: `sha256:${"a".repeat(64)}`,
      modelId: "qwen-test",
      modelRevision: "1".repeat(40),
      backend: "cuda",
      runtimeAbi: "cuda-12",
      quantization: "bf16",
      contextTokens: 8_192,
      expectedLayerStart: 0,
      expectedLayerEnd: 16,
      expectedKvBytesPerToken: 32_768,
      expectedLayerWeightBytes: 256 * 1024 * 1024,
      referenceDecodeMsPerToken: 8,
      referencePrefillMsPerToken: 0.4,
      hiddenSize: 4_096,
      attentionHeads: 32,
      kvHeads: 4,
      headDim: 128,
      requiredRoles: ["head", "tail"],
      minimumSamples: 7,
    } satisfies EngineRuntimeChallengeRequest;
    current.plans = [{
      workerId: worker.id,
      modelId: "qwen-test",
      request: engineRequest,
      evidence: {
        canaryEvidenceId: current.worker.capabilities.deployments[0]!.canaryEvidence!.evidenceId,
      },
    }];
    const challengeId = hub.startEngineRuntimeProfileChallenge(worker.id, engineRequest);
    expect(challengeId).toMatch(/^challenge-/);
    const engineChallenge = socket.sent.find(
      (message) => message.type === "evidence.challenge"
        && (message.payload as { kind?: string }).kind === "engine-runtime",
    )!.payload as Record<string, unknown>;
    deliver(hub, state, "evidence.engine-runtime.complete", {
      ...binding(engineChallenge),
      measurement: {
        measuredAt: new Date().toISOString(),
        samples: 21,
        confidenceHalfWidthPct: 8,
        capacity: {
          contextTokens: 8_192,
          maxLayerCount: 16,
          kvBytesPerToken: 32_768,
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
          prefillScale: 0.5,
        },
        features: {
          fastKernel: true,
          graphMode: "available",
          roles: ["head", "middle", "tail"],
        },
      },
    });

    expect(current.worker.capabilities.distributedExecutor?.engineProfiles)
      .toEqual([expect.objectContaining({
        descriptorDigest: `sha256:${"a".repeat(64)}`,
        certificationId: `sha256:${"b".repeat(64)}`,
        artifactManifestDigest: `sha256:${"a".repeat(64)}`,
        workerId: worker.id,
        sessionId: "session-test",
      })]);
    hub.close();
  });
});

function buildIdentity(seed: string): NonNullable<WorkerCapabilities["buildIdentity"]> {
  return {
    schema: "mycellios-native-build-provenance/1",
    version: "0.2.99",
    sourceId: `sha256:${seed.repeat(64)}`,
  };
}

function deploymentEvidence() {
  const observedAt = Date.now();
  return createCoordinatorDeploymentCanaryEvidence({
    challengeId: "challenge-build-change",
    nonce: Buffer.alloc(32, 7).toString("base64url"),
    workerId: "worker-test",
    sessionId: "session-test",
    issuedAt: new Date(observedAt - 1_000).toISOString(),
    expiresAt: new Date(observedAt + 60_000).toISOString(),
    model: "qwen-test",
    modelDigest: `sha256:${"a".repeat(64)}`,
    activationId: "activation-test",
    promptDigest: `sha256:${"c".repeat(64)}`,
    maxOutputTokens: 4,
    observedAt: new Date(observedAt).toISOString(),
    warmupSamples: 1,
    samples: [0, 1, 2].map((sampleIndex) => ({
      sampleId: `sample-${sampleIndex}`,
      outputTokens: 4,
      activeMs: 1_000,
      ttftMs: 250,
      completed: true as const,
    })),
  });
}

function pipelineCapabilities(): WorkerCapabilities {
  return {
    region: "test",
    agentVersion: "0.2.99",
    gpus: [{
      id: "cell",
      vendor: "mycellios",
      model: "cell",
      physicalVramMb: 0,
      offeredVramMb: 4_096,
      freeOfferedVramMb: 4_096,
    }],
    limits: { maxConcurrency: 1, pauseWhenForeground: false },
    deployments: [{
      deploymentId: "dep-qwen-test",
      model: "qwen-test",
      modelDigest: `sha256:${"a".repeat(64)}`,
      activationId: "activation-test",
      mode: "replica",
      adapter: "mycellios-pipeline",
      peakVramMb: 2_048,
      contextLimit: 4_096,
      maxConcurrency: 1,
      freeSlots: 1,
      tokensPerSecond: 1,
      throughputSource: "default",
      ttftMs: 60_000,
      verificationState: "pending",
      dataLocality: "local",
    }],
    network: { coordinatorRttMs: 1, uplinkMbps: 100, downlinkMbps: 100 },
  };
}

function executorCapabilities(): WorkerCapabilities {
  const capabilities = pipelineCapabilities();
  capabilities.deployments = [];
  capabilities.gpus = [{
    id: "gpu-0",
    vendor: "nvidia",
    model: "NVIDIA Test GPU",
    physicalVramMb: 8_192,
    offeredVramMb: 4_096,
    freeOfferedVramMb: 4_096,
  }];
  capabilities.distributedExecutor = {
    protocol: "gdlp-worker-tunnel/2",
    nodeId: "node-test",
    stageHost: "node-test.relay",
    stagePort: 9_850,
    runtime: "python-safetensors",
    computeMode: "automatic",
    cpuEligible: false,
    acceleration: {
      schema: "mycellios-accelerator-diagnostics/1",
      appVersion: "0.2.99",
      state: "gpu-ready",
      backend: "cuda",
      deviceName: "NVIDIA Test GPU",
      gpuVendor: "nvidia",
      gpuModel: "NVIDIA Test GPU",
      phase: "ready",
      progressPct: 100,
      issueCode: null,
      issueSummary: null,
      retryable: false,
      retryAttempt: 0,
      nextRetryAt: null,
      updatedAt: new Date().toISOString(),
      recentEvents: [],
    },
  };
  return capabilities;
}

function storedWorker(capabilities: WorkerCapabilities): StoredWorker {
  return {
    id: "worker-test",
    status: "online",
    capabilities: structuredClone(capabilities),
    reliability: 1,
    jobsCompleted: 0,
    lastSeenAt: Date.now(),
    identityKind: "device",
    identityId: "node-test",
  };
}

function harness(initial: StoredWorker) {
  const current = { worker: initial, plans: [] as Array<Record<string, unknown>> };
  const store = {
    getWorker: (workerId: string) => workerId === current.worker.id ? current.worker : null,
    updateWorkerHeartbeat: (
      workerId: string,
      capabilities: WorkerCapabilities,
      status: StoredWorker["status"],
    ) => {
      if (workerId !== current.worker.id) return;
      current.worker = { ...current.worker, capabilities, status, lastSeenAt: Date.now() };
    },
    listWorkers: () => [current.worker],
    listEngineRuntimeActivationPlans: () => current.plans,
    setWorkerStatus: () => undefined,
  } as unknown as MeshStore;
  const hub = new WorkerHub(store);
  const socket = {
    OPEN: 1,
    readyState: 1,
    bufferedAmount: 0,
    sent: [] as Array<{ type: string; payload: unknown }>,
    closed: [] as Array<{ code: number; reason: string }>,
    send(serialized: string) {
      this.sent.push(JSON.parse(serialized) as { type: string; payload: unknown });
    },
    close(code: number, reason: string) {
      this.closed.push({ code, reason });
    },
  };
  const state = {
    socket,
    workerId: initial.id,
    ready: true,
    helloTimer: setTimeout(() => undefined, 60_000),
    pending: false,
    messageWindowStartedAt: Date.now(),
    messagesInWindow: 0,
    sessionId: "session-test",
  };
  (hub as unknown as { connections: Map<string, unknown> })
    .connections.set(initial.id, state);
  return { hub, socket, state, current };
}

function issueChallenges(
  hub: WorkerHub,
  state: ReturnType<typeof harness>["state"],
  capabilities: WorkerCapabilities,
): void {
  (
    hub as unknown as {
      ensureEvidenceChallenges(
        connection: typeof state,
        claims: WorkerCapabilities,
      ): void;
    }
  ).ensureEvidenceChallenges(state, capabilities);
}

function deliver(
  hub: WorkerHub,
  state: ReturnType<typeof harness>["state"],
  type: string,
  payload: Record<string, unknown>,
): void {
  (
    hub as unknown as {
      handleEvidenceEnvelope(
        connection: typeof state,
        envelope: WorkerEnvelope,
      ): void;
    }
  ).handleEvidenceEnvelope(state, {
    v: 1,
    workerId: state.workerId,
    type,
    payload,
  });
}

function binding(challenge: Record<string, unknown>) {
  return {
    challengeId: challenge.challengeId,
    nonce: challenge.nonce,
    sessionId: challenge.sessionId,
  };
}

function series(unit: "GB/s" | "TFLOP/s", median: number) {
  return {
    unit,
    warmupSamples: 2,
    samples: 7,
    p5: median * 0.9,
    p50: median,
    p95: median * 1.1,
    confidenceHalfWidthPct: 5,
  };
}
