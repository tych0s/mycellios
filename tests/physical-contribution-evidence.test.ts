import { describe, expect, it } from "vitest";

import { createCoordinatorDeploymentCanaryEvidence } from "../src/contracts/deployment-canary.js";
import type { NetworkExecutionTrace, WorkerPhysicalIdentity } from "../src/contracts/types.js";
import { PhysicalContributionEvidenceAuthority } from "../src/economy/physical-contribution-evidence.js";
import { MeshDatabase } from "../src/storage/database.js";
import type { StoredWorker } from "../src/storage/store.js";

const sha = (character: string) => `sha256:${character.repeat(64)}`;

describe("physical contribution evidence", () => {
  it("certifies an exact physical route idempotently", () => {
    const database = new MeshDatabase(":memory:");
    const authority = new PhysicalContributionEvidenceAuthority(database);
    const workers = [worker("worker-a", "node-a", "a", "dep-a"), worker("worker-b", "node-b", "b", "dep-b")];
    const trace = physicalTrace();
    const input = { jobId: trace.jobId, executionReceiptId: sha("e"), trace, workers, createdAt: 2_100 };
    const evidence = authority.certify(input);

    expect(evidence).toMatchObject({
      schema: "mycellios-physical-contribution-evidence/1",
      jobId: "job-physical",
      executionReceiptId: sha("e"),
      physicalBoundaryCount: 1,
      participants: [
        { nodeId: "node-a", workerId: "worker-a", stageCount: 1 },
        { nodeId: "node-b", workerId: "worker-b", stageCount: 1 },
      ],
    });
    expect(authority.certify(input)).toEqual(evidence);
    expect(authority.read(evidence.id)).toEqual(evidence);
    database.close();
  });

  it("rejects replay conflicts, collocated identities and unmeasured boundaries", () => {
    const database = new MeshDatabase(":memory:");
    const authority = new PhysicalContributionEvidenceAuthority(database);
    const trace = physicalTrace();
    const workers = [worker("worker-a", "node-a", "a", "dep-a"), worker("worker-b", "node-b", "b", "dep-b")];
    authority.certify({ jobId: trace.jobId, executionReceiptId: sha("e"), trace, workers, createdAt: 2_100 });
    expect(() => authority.certify({
      jobId: trace.jobId, executionReceiptId: sha("e"),
      trace: { ...trace, observedUntil: 2_001, durationMs: 1_001 }, workers, createdAt: 2_100,
    })).toThrow("economic_contribution_evidence_replay_conflict");

    const duplicate = worker("worker-b", "node-b", "b", "dep-b");
    duplicate.capabilities.distributedExecutor!.physicalIdentity = structuredClone(
      workers[0]!.capabilities.distributedExecutor!.physicalIdentity,
    );
    expect(() => new PhysicalContributionEvidenceAuthority(new MeshDatabase(":memory:")).certify({
      jobId: trace.jobId, executionReceiptId: sha("f"), trace,
      workers: [workers[0]!, duplicate], createdAt: 2_100,
    })).toThrow("economic_contribution_collocated_identity_detected");

    expect(() => new PhysicalContributionEvidenceAuthority(new MeshDatabase(":memory:")).certify({
      jobId: trace.jobId, executionReceiptId: sha("f"),
      trace: { ...trace, boundaries: [{ ...trace.boundaries[0]!, transport: "unobserved", streamId: null, countersExclusive: null, bytesSourceToDestination: null, bytesDestinationToSource: null, connectRttMs: null, streamCreatedAt: null, streamConnectedAt: null, streamEndedAt: null, observedOverlapMs: null }] },
      workers, createdAt: 2_100,
    })).toThrow("economic_contribution_physical_boundary_is_unverified");
    database.close();
  });
});

function worker(workerId: string, nodeId: string, marker: string, deploymentId: string): StoredWorker {
  const identity: WorkerPhysicalIdentity = {
    schema: "gdlp-worker-physical-identity/1",
    provider: "generic",
    providerMachineFingerprintSha256: sha(marker),
    hostFingerprintSha256: sha(marker === "a" ? "c" : "d"),
    gpuFingerprintsSha256: [sha(marker === "a" ? "1" : "2")],
    attestedAt: new Date(500).toISOString(),
  };
  const canary = createCoordinatorDeploymentCanaryEvidence({
    challengeId: `challenge-${workerId}`,
    nonce: Buffer.alloc(32, marker.charCodeAt(0)).toString("base64url"),
    workerId,
    sessionId: `session-${workerId}`,
    issuedAt: new Date(500).toISOString(),
    expiresAt: new Date(2_500).toISOString(),
    model: "qwen3",
    modelDigest: sha(marker),
    activationId: `activation-${workerId}`,
    promptDigest: sha("9"),
    maxOutputTokens: 8,
    observedAt: new Date(900).toISOString(),
    warmupSamples: 1,
    samples: [0, 1, 2].map((index) => ({ sampleId: `sample-${index}`, outputTokens: 8, activeMs: 10, ttftMs: 1, completed: true as const })),
  });
  return {
    id: workerId,
    capabilities: {
      region: "test", agentVersion: "test", gpus: [],
      limits: { maxConcurrent: 1, maxQueueDepth: 1 },
      network: { coordinatorRttMs: 1, uplinkMbps: 1_000, downlinkMbps: 1_000 },
      deployments: [{
        deploymentId, model: "qwen3", modelDigest: sha(marker), activationId: `activation-${workerId}`,
        mode: "replica", adapter: "python", peakVramMb: 1, contextLimit: 1_024,
        maxConcurrency: 1, freeSlots: 1, tokensPerSecond: 10, ttftMs: 1,
        verificationState: "verified", canaryEvidence: canary, dataLocality: "local",
        execution: { deviceType: "gpu", backend: "cuda", precision: "bf16", deviceName: "GPU" },
      }],
      distributedExecutor: { protocol: "gdlp-worker-tunnel/2", nodeId, stageHost: "127.0.0.1", stagePort: 1, runtime: "python-safetensors", physicalIdentity: identity },
    },
  } as unknown as StoredWorker;
}

function physicalTrace(): NetworkExecutionTrace {
  return {
    schema: "mycellios-network-execution-trace/1", jobId: "job-physical", attempt: 1,
    observedFrom: 1_000, observedUntil: 2_000, durationMs: 1_000,
    routeClass: "pipeline", affinityHit: false, routeDecision: null,
    selectedRoute: [
      { routeStageIndex: 0, workerId: "worker-a", deploymentId: "dep-a", modelDigest: sha("a"), stageIndex: 0 },
      { routeStageIndex: 1, workerId: "worker-b", deploymentId: "dep-b", modelDigest: sha("b"), stageIndex: 1 },
    ],
    stages: [
      stage(0, 0, "worker-a", "node-a", "dep-a", "a", 0, 10),
      stage(1, 1, "worker-b", "node-b", "dep-b", "b", 10, 20),
    ],
    physicalBoundaryCount: 1,
    boundaries: [{
      boundaryIndex: 0, fromStageIndex: 0, toStageIndex: 1,
      sourceNodeId: "node-a", destinationNodeId: "node-b", physicalBoundary: true,
      transport: "direct", streamId: "stream-a-b", bytesSourceToDestination: 128,
      bytesDestinationToSource: 16, countersExclusive: true, connectRttMs: 2,
      streamCreatedAt: 800, streamConnectedAt: 900, streamEndedAt: null, observedOverlapMs: 1_000,
    }],
  };
}

function stage(routeStageIndex: number, stageIndex: number, workerId: string, nodeId: string, deploymentId: string, marker: string, layerStart: number, layerEnd: number) {
  return {
    routeStageIndex, stageIndex, nodeId, workerId, deploymentId,
    deploymentOwnerWorkerId: workerId, modelDigest: sha(marker), layerStart, layerEnd,
    deviceType: "gpu" as const, backend: "cuda" as const, precision: "bf16", deviceName: "GPU",
    startedAt: null, endedAt: null, durationMs: null,
  };
}
