import { EventEmitter } from "node:events";
import { afterEach, describe, expect, it } from "vitest";
import { createCoordinatorDeploymentCanaryEvidence } from "../src/contracts/deployment-canary.js";
import type {
  CompletionResult,
  ModelDeployment,
  ScheduledRoute,
  WorkerEnvelope,
} from "../src/contracts/types.js";
import { MeshService } from "../src/coordinator/mesh-service.js";
import type {
  RuntimeTransportSnapshot,
  WorkerHub,
} from "../src/coordinator/worker-hub.js";
import { Scheduler } from "../src/scheduler/scheduler.js";
import { MeshDatabase } from "../src/storage/database.js";
import { MeshStore, type StoredWorker } from "../src/storage/store.js";
import {
  buildNetworkExecutionTrace,
  directedBoundaryKey,
  parseNetworkExecutionTrace,
} from "../src/telemetry/network-execution-trace.js";

describe("physical network execution trace", () => {
  it("attributes exact directed stream deltas to an exclusive request", () => {
    const start = transport({ bytesForward: 100, bytesReturn: 20 });
    const end = transport({ bytesForward: 540, bytesReturn: 84 });
    const trace = buildNetworkExecutionTrace({
      jobId: "job-one",
      attempt: 1,
      route: route(),
      workers: workers(),
      observedFrom: 1_000,
      observedUntil: 2_000,
      startTransports: [start],
      endTransports: [end],
    });

    expect(trace).toMatchObject({
      schema: "mycellios-network-execution-trace/1",
      jobId: "job-one",
      durationMs: 1_000,
      physicalBoundaryCount: 1,
      stages: [
        {
          nodeId: "node-a",
          workerId: "worker-a",
          deploymentId: "dep-a",
          layerStart: 0,
          layerEnd: 14,
          backend: "cuda",
          precision: "bf16",
          startedAt: null,
        },
        {
          nodeId: "node-b",
          workerId: "worker-b",
          layerStart: 14,
          layerEnd: 28,
          backend: "rocm",
        },
      ],
      boundaries: [{
        transport: "direct",
        streamId: "stream-ab",
        bytesSourceToDestination: 440,
        bytesDestinationToSource: 64,
        countersExclusive: true,
        connectRttMs: 8,
      }],
    });
    expect(parseNetworkExecutionTrace(trace)).toEqual(trace);
  });

  it("keeps shared counters null instead of attributing another job's traffic", () => {
    const trace = buildNetworkExecutionTrace({
      jobId: "job-shared",
      attempt: 1,
      route: route(),
      workers: workers(),
      observedFrom: 1_000,
      observedUntil: 2_000,
      startTransports: [transport({ bytesForward: 100, bytesReturn: 20 })],
      endTransports: [transport({ bytesForward: 540, bytesReturn: 84 })],
      contendedBoundaryKeys: new Set([directedBoundaryKey("node-a", "node-b")]),
    });

    expect(trace.boundaries[0]).toMatchObject({
      transport: "direct",
      countersExclusive: false,
      bytesSourceToDestination: null,
      bytesDestinationToSource: null,
    });
  });

  it("does not turn an unchanged direct telemetry counter into a measured zero", () => {
    const zero = transport({ bytesForward: 0, bytesReturn: 0 });
    const trace = buildNetworkExecutionTrace({
      jobId: "job-direct-without-fresh-counter",
      attempt: 1,
      route: route(),
      workers: workers(),
      observedFrom: 1_000,
      observedUntil: 2_000,
      startTransports: [zero],
      endTransports: [zero],
    });
    expect(trace.boundaries[0]).toMatchObject({
      transport: "direct",
      countersExclusive: false,
      bytesSourceToDestination: null,
      bytesDestinationToSource: null,
    });
  });

  it("marks an ambiguous pair unobserved and rejects malformed evidence", () => {
    const duplicate = {
      ...transport({ bytesForward: 20, bytesReturn: 10 }),
      streamId: "stream-ab-duplicate",
    };
    const trace = buildNetworkExecutionTrace({
      jobId: "job-ambiguous",
      attempt: 1,
      route: route(),
      workers: workers(),
      observedFrom: 1_000,
      observedUntil: 2_000,
      startTransports: [],
      endTransports: [
        transport({ bytesForward: 40, bytesReturn: 20 }),
        duplicate,
      ],
    });
    expect(trace.boundaries[0]).toMatchObject({
      physicalBoundary: true,
      transport: "unobserved",
      streamId: null,
      connectRttMs: null,
    });
    expect(parseNetworkExecutionTrace({
      ...trace,
      boundaries: [{
        ...trace.boundaries[0],
        transport: "direct",
        streamId: null,
      }],
    })).toBeNull();
  });

  it("does not attach a stream when another deployment owns the same node pair", () => {
    const otherDeployment = {
      ...rootDeployment(),
      deploymentId: "dep-c",
    };
    const trace = buildNetworkExecutionTrace({
      jobId: "job-deployment-ambiguous",
      attempt: 1,
      route: route(),
      workers: [...workers(), worker("worker-c", "node-c", otherDeployment)],
      observedFrom: 1_000,
      observedUntil: 2_000,
      startTransports: [transport({ bytesForward: 10, bytesReturn: 2 })],
      endTransports: [transport({ bytesForward: 100, bytesReturn: 20 })],
    });
    expect(trace.boundaries[0]).toMatchObject({
      physicalBoundary: true,
      transport: "unobserved",
      streamId: null,
      bytesSourceToDestination: null,
    });
  });
});

describe("MeshService route evidence", () => {
  const databases: MeshDatabase[] = [];
  afterEach(() => {
    for (const database of databases.splice(0)) database.close();
  });

  it("attaches the successful deployment route and interval-scoped transport to completion", async () => {
    const database = new MeshDatabase(":memory:");
    databases.push(database);
    const store = new MeshStore(database);
    for (const worker of workers()) {
      store.registerWorker({ capabilities: worker.capabilities });
    }
    const registered = store.listWorkers();
    const workerA = registered.find(
      (worker) => worker.capabilities.distributedExecutor?.nodeId === "node-a",
    )!;
    const workerB = registered.find(
      (worker) => worker.capabilities.distributedExecutor?.nodeId === "node-b",
    )!;
    // The scheduler-selected deployment must use the IDs assigned by storage.
    workerA.capabilities.deployments[0]!.deploymentId = `dep-${workerA.id}`;
    workerA.capabilities.deployments[0]!.modelDigest = `sha256:${"a".repeat(64)}`;
    const observedAt = Date.now();
    Object.assign(workerA.capabilities.deployments[0]!, {
      activationId: "activation-network-trace",
      verificationState: "verified",
      throughputSource: "measured",
      tokensPerSecond: 10,
      ttftMs: 10,
      canaryEvidence: createCoordinatorDeploymentCanaryEvidence({
        challengeId: "challenge-network-trace",
        nonce: Buffer.alloc(32, 6).toString("base64url"),
        workerId: workerA.id,
        sessionId: "session-network-trace",
        issuedAt: new Date(observedAt - 1_000).toISOString(),
        expiresAt: new Date(observedAt + 60_000).toISOString(),
        model: "distributed-small",
        modelDigest: `sha256:${"a".repeat(64)}`,
        activationId: "activation-network-trace",
        promptDigest: `sha256:${"c".repeat(64)}`,
        maxOutputTokens: 10,
        observedAt: new Date(observedAt).toISOString(),
        warmupSamples: 1,
        samples: [0, 1, 2].map((index) => ({
          sampleId: `sample-${index}`,
          outputTokens: 10,
          activeMs: 1_000,
          ttftMs: 10,
          completed: true as const,
        })),
      }),
    });
    store.updateWorkerHeartbeat(workerA.id, workerA.capabilities, "online");
    store.updateWorkerHeartbeat(workerB.id, workerB.capabilities, "online");

    const hub = new TraceHub();
    hub.connected.add(workerA.id);
    hub.connected.add(workerB.id);
    hub.snapshots = [transport({ bytesForward: 10, bytesReturn: 5 })];
    const service = new MeshService(
      store,
      new Scheduler(store),
      hub as unknown as WorkerHub,
      30_000,
    );
    const handle = service.submit({
      model: "distributed-small",
      messages: [{ role: "user", content: "ruta física" }],
      max_tokens: 8,
    });
    const offer = hub.sent.find((message) => message.type === "lease.offer")!;
    const payload = offer.payload as { leaseId: string };
    hub.workerMessage(envelope(workerA.id, "lease.accept", {
      jobId: handle.jobId,
      leaseId: payload.leaseId,
    }));
    hub.workerMessage(envelope(workerA.id, "task.token", {
      jobId: handle.jobId,
      leaseId: payload.leaseId,
      index: 0,
      text: "ok",
    }));
    hub.snapshots = [transport({ bytesForward: 210, bytesReturn: 45 })];
    const completion: CompletionResult = {
      jobId: handle.jobId,
      leaseId: payload.leaseId,
      text: "ok",
      finishReason: "stop",
      metrics: {
        inputTokens: 5,
        outputTokens: 1,
        ttftMs: 10,
        activeMs: 30,
      },
    };
    hub.workerMessage(envelope(workerA.id, "task.complete", completion));

    const events = [];
    for await (const event of handle.events) events.push(event);
    const completed = events.find((event) => event.type === "completed");
    expect(completed).toMatchObject({
      type: "completed",
      result: {
        networkTrace: {
          jobId: handle.jobId,
          stages: [
            { nodeId: "node-a", workerId: workerA.id },
            { nodeId: "node-b", workerId: workerB.id },
          ],
          boundaries: [{
            transport: "direct",
            bytesSourceToDestination: 200,
            bytesDestinationToSource: 40,
          }],
        },
      },
    });
  });
});

class TraceHub extends EventEmitter {
  readonly connected = new Set<string>();
  readonly sent: Array<{ workerId: string; type: string; payload: unknown }> = [];
  snapshots: RuntimeTransportSnapshot[] = [];

  connectedWorkerIds(): ReadonlySet<string> {
    return new Set(this.connected);
  }

  isConnected(workerId: string): boolean {
    return this.connected.has(workerId);
  }

  send(workerId: string, type: string, payload: unknown): boolean {
    this.sent.push({ workerId, type, payload });
    return this.isConnected(workerId);
  }

  runtimeTransportSnapshot(): RuntimeTransportSnapshot[] {
    return this.snapshots.map((snapshot) => ({ ...snapshot }));
  }

  workerMessage(message: WorkerEnvelope): void {
    this.emit("envelope", message);
  }
}

function route(): ScheduledRoute {
  return {
    routeClass: "replica",
    model: "distributed-small",
    region: "local",
    stages: [{
      workerId: "worker-a",
      deploymentId: "dep-a",
      modelDigest: "sha256:model",
      stageIndex: 0,
      score: 1,
    }],
    score: 1,
    affinityHit: false,
  };
}

function workers(): StoredWorker[] {
  return [
    worker("worker-a", "node-a", rootDeployment()),
    worker("worker-b", "node-b", null),
  ];
}

function worker(
  id: string,
  nodeId: string,
  deployment: ModelDeployment | null,
): StoredWorker {
  return {
    id,
    status: "online",
    reliability: 1,
    jobsCompleted: 0,
    lastSeenAt: 1_000,
    identityKind: "device",
    identityId: nodeId,
    capabilities: {
      region: "local",
      agentVersion: "test",
      gpus: [{
        id: `${nodeId}-gpu`,
        vendor: "test",
        model: nodeId,
        physicalVramMb: 8_192,
        offeredVramMb: 8_192,
        freeOfferedVramMb: 8_192,
      }],
      limits: { maxConcurrency: 1, pauseWhenForeground: false },
      deployments: deployment ? [deployment] : [],
      network: { coordinatorRttMs: 3, uplinkMbps: 500, downlinkMbps: 500 },
      distributedExecutor: {
        protocol: "gdlp-worker-tunnel/2",
        nodeId,
        stageHost: "127.0.0.1",
        stagePort: 9_000,
        runtime: "python-safetensors",
        computeMode: "gpu-only",
        cpuEligible: false,
      },
    },
  };
}

function rootDeployment(): ModelDeployment {
  return {
    deploymentId: "dep-a",
    model: "distributed-small",
    modelDigest: "sha256:model",
    mode: "replica",
    adapter: "mycellios-pipeline",
    peakVramMb: 4_096,
    contextLimit: 4_096,
    maxConcurrency: 1,
    freeSlots: 1,
    tokensPerSecond: 10,
    ttftMs: 10,
    dataLocality: "local",
    execution: {
      deviceType: "gpu",
      backend: "cuda",
      deviceName: "distributed",
      precision: "bf16",
      fallback: false,
      stages: [
        {
          nodeId: "node-a",
          stageIndex: 0,
          layerStart: 0,
          layerEnd: 14,
          deviceType: "gpu",
          backend: "cuda",
          deviceName: "GPU A",
          precision: "bf16",
          fallback: false,
        },
        {
          nodeId: "node-b",
          stageIndex: 1,
          layerStart: 14,
          layerEnd: 28,
          deviceType: "gpu",
          backend: "rocm",
          deviceName: "GPU B",
          precision: "bf16",
          fallback: false,
        },
      ],
    },
  };
}

function transport(input: {
  bytesForward: number;
  bytesReturn: number;
}): RuntimeTransportSnapshot {
  return {
    streamId: "stream-ab",
    sourceNodeId: "node-a",
    destinationNodeId: "node-b",
    targetPort: 9_001,
    mode: "direct",
    state: "active",
    bytesSourceToDestination: input.bytesForward,
    bytesDestinationToSource: input.bytesReturn,
    createdAt: 500,
    connectedAt: 600,
    endedAt: null,
    connectRttMs: 8,
  };
}

function envelope(workerId: string, type: string, payload: unknown): WorkerEnvelope {
  return { v: 1, workerId, type, payload };
}
