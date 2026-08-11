import { generateKeyPairSync } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { sha256CanonicalEvidence } from "../src/core/json.js";
import { ExecutionReceiptStore } from "../src/coordinator/execution-receipt-store.js";
import { MeshDatabase } from "../src/storage/database.js";
import { MeshStore } from "../src/storage/store.js";
import type { NetworkExecutionTrace } from "../src/contracts/types.js";

const sha = (character: string) => `sha256:${character.repeat(64)}` as const;

describe("durable redacted execution topology", () => {
  it("persists a receipt-bound projection across restart without internal identities", () => {
    const directory = mkdtempSync(join(tmpdir(), "mycellios-topology-"));
    const path = join(directory, "mesh.db");
    const keys = generateKeyPairSync("ed25519");
    let database = new MeshDatabase(path);
    const jobs = new MeshStore(database);
    jobs.createJob({ id: "job-topology", sessionId: "session", model: "qwen", workloadClass: "interactive", deadlineAt: 5_000 });
    const trace = fixtureTrace();
    const receipts = new ExecutionReceiptStore(database, { keyId: "key", privateKey: keys.privateKey });
    const body = { schema: "mycellios-execution-receipt/1" as const, jobId: trace.jobId,
      modelIdHash: sha("a"), routeClass: "pipeline" as const,
      metrics: { inputTokens: 2, outputTokens: 1, ttftMs: 10, activeMs: 20 },
      networkTraceDigest: sha256CanonicalEvidence(trace), recovery: { mode: "none" as const, attempts: 1, replayedTokenEvents: 0 },
      privacy: { trust: "default" as const, boundary: "trusted-edges" as const, pinnedIdentityHashes: [] }, completedAt: 1_000 };
    const receipt = receipts.record(body, { trace, regionForWorker: (workerId) => workerId === "private-worker-a" ? "Madrid" : "10.0.0.7/private" });
    expect(receipts.record(body, { trace, regionForWorker: (workerId) => workerId === "private-worker-a" ? "Madrid" : "10.0.0.7/private" })).toEqual(receipt);
    const topology = receipts.topologyForJob(trace.jobId)!;
    expect(topology).toMatchObject({ receiptId: receipt.receiptId, traceDigest: receipt.networkTraceDigest,
      classification: "physical", stages: [{ alias: "stage-0", region: "Madrid" }, { alias: "stage-1", region: "undisclosed" }],
      boundaries: [{ transport: "direct", physicalBoundary: true }] });
    const serialized = JSON.stringify(topology);
    for (const secret of ["private-worker-a", "private-worker-b", "private-node", "private-deployment", "10.0.0.7"]) expect(serialized).not.toContain(secret);
    database.close();
    database = new MeshDatabase(path);
    expect(new ExecutionReceiptStore(database, { keyId: "key", privateKey: keys.privateKey }).topologyForJob(trace.jobId)).toEqual(topology);
    database.close();
    rmSync(directory, { recursive: true, force: true });
  });

  it("rejects a trace that is not the one sealed by the receipt", () => {
    const database = new MeshDatabase(":memory:");
    new MeshStore(database).createJob({ id: "job-topology", sessionId: "session", model: "qwen", workloadClass: "interactive", deadlineAt: 5_000 });
    const keys = generateKeyPairSync("ed25519");
    const receipts = new ExecutionReceiptStore(database, { keyId: "key", privateKey: keys.privateKey });
    const trace = fixtureTrace();
    expect(() => receipts.record({ schema: "mycellios-execution-receipt/1", jobId: trace.jobId,
      modelIdHash: sha("a"), routeClass: "pipeline", metrics: { inputTokens: 1, outputTokens: 1, ttftMs: 1, activeMs: 1 },
      networkTraceDigest: sha("f"), recovery: { mode: "none", attempts: 1, replayedTokenEvents: 0 },
      privacy: { trust: "default", boundary: "trusted-edges", pinnedIdentityHashes: [] }, completedAt: 1 },
    { trace, regionForWorker: () => null })).toThrow("execution_topology_trace_digest_mismatch");
    expect(receipts.forJob(trace.jobId)).toBeNull();
    database.close();
  });
});

function fixtureTrace(): NetworkExecutionTrace {
  const stage = (stageIndex: number, workerId: string) => ({ routeStageIndex: stageIndex, stageIndex,
    nodeId: `private-node-${stageIndex}`, workerId, deploymentId: `private-deployment-${stageIndex}`,
    deploymentOwnerWorkerId: workerId, modelDigest: sha("b"), layerStart: stageIndex * 10,
    layerEnd: (stageIndex + 1) * 10, deviceType: "gpu" as const, backend: "cuda" as const,
    precision: "bf16", deviceName: "private gpu", startedAt: null, endedAt: null, durationMs: null });
  return { schema: "mycellios-network-execution-trace/1", jobId: "job-topology", attempt: 1,
    observedFrom: 100, observedUntil: 200, durationMs: 100, routeClass: "pipeline", affinityHit: false,
    routeDecision: null, selectedRoute: [], stages: [stage(0, "private-worker-a"), stage(1, "private-worker-b")],
    physicalBoundaryCount: 1, boundaries: [{ boundaryIndex: 0, fromStageIndex: 0, toStageIndex: 1,
      sourceNodeId: "private-node-0", destinationNodeId: "private-node-1", physicalBoundary: true,
      transport: "direct", streamId: "private-stream", bytesSourceToDestination: 10,
      bytesDestinationToSource: 2, countersExclusive: true, connectRttMs: 5, streamCreatedAt: 100,
      streamConnectedAt: 105, streamEndedAt: 200, observedOverlapMs: 95 }] };
}
