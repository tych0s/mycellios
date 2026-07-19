import { EventEmitter } from "node:events";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type {
  ChatCompletionRequest,
  JobPayload,
  WorkerEnvelope,
} from "../src/contracts/types.js";
import { MeshService } from "../src/coordinator/mesh-service.js";
import type { WorkerHub } from "../src/coordinator/worker-hub.js";
import { inputHashForRequest } from "../src/core/request.js";
import { Scheduler } from "../src/scheduler/scheduler.js";
import { MeshDatabase } from "../src/storage/database.js";
import { MeshStore } from "../src/storage/store.js";
import { addWorker } from "./helpers.js";

interface SentMessage {
  workerId: string;
  type: string;
  payload: unknown;
}

class FakeWorkerHub extends EventEmitter {
  readonly connected = new Set<string>();
  readonly sent: SentMessage[] = [];

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

  workerMessage(envelope: WorkerEnvelope): void {
    this.emit("envelope", envelope);
  }
}

describe("active-route recovery policy", () => {
  let database: MeshDatabase;
  let store: MeshStore;
  let hub: FakeWorkerHub;
  let service: MeshService;

  beforeEach(() => {
    database = new MeshDatabase(":memory:");
    store = new MeshStore(database);
    hub = new FakeWorkerHub();
  });

  afterEach(() => database.close());

  it("replays the immutable prompt checkpoint on an exact-revision standby before token zero", () => {
    const primary = addWorker(store, {
      id: "primary",
      modelDigest: "sha256:revision-a",
      tokensPerSecond: 50,
    });
    const incompatible = addWorker(store, {
      id: "incompatible",
      modelDigest: "sha256:revision-b",
      tokensPerSecond: 40,
    });
    const standby = addWorker(store, {
      id: "standby",
      modelDigest: "sha256:revision-a",
      tokensPerSecond: 30,
    });
    hub.connected.add(primary.id);
    hub.connected.add(incompatible.id);
    hub.connected.add(standby.id);
    service = new MeshService(
      store,
      new Scheduler(store),
      hub as unknown as WorkerHub,
      30_000,
    );
    const request: ChatCompletionRequest = {
      model: "distributed-small",
      messages: [{ role: "user", content: "prompt original" }],
      max_tokens: 32,
      seed: 7,
    };

    const handle = service.submit(request, "failover-session");
    request.messages[0]!.content = "mutated after submit";
    const firstOffer = leaseOffers(hub)[0]!;
    expect(firstOffer.workerId).toBe(primary.id);

    hub.workerMessage({
      v: 1,
      type: "lease.reject",
      workerId: primary.id,
      payload: {
        jobId: handle.jobId,
        leaseId: firstOffer.payload.leaseId,
        reason: "worker_busy",
      },
    });

    const offers = leaseOffers(hub);
    expect(offers).toHaveLength(2);
    expect(offers[1]!.workerId).toBe(standby.id);
    expect(offers[1]!.workerId).not.toBe(incompatible.id);
    expect(offers[1]!.payload.request.messages[0]?.content).toBe("prompt original");
    expect(inputHashForRequest(offers[1]!.payload.request)).toBe(
      inputHashForRequest(firstOffer.payload.request),
    );
    expect(hub.sent).toContainEqual({
      workerId: primary.id,
      type: "task.cancel",
      payload: { jobId: handle.jobId },
    });
    expect(store.getJob(handle.jobId)?.workerId).toBe(standby.id);

    expect(service.cancel(handle.jobId)).toBe(true);
  });

  it("does not restart after an empty token event because streaming has already begun", () => {
    const primary = addWorker(store, {
      id: "primary",
      modelDigest: "sha256:revision-a",
      tokensPerSecond: 50,
    });
    const standby = addWorker(store, {
      id: "standby",
      modelDigest: "sha256:revision-a",
      tokensPerSecond: 30,
    });
    hub.connected.add(primary.id);
    hub.connected.add(standby.id);
    service = new MeshService(
      store,
      new Scheduler(store),
      hub as unknown as WorkerHub,
      30_000,
    );
    const handle = service.submit({
      model: "distributed-small",
      messages: [{ role: "user", content: "hola" }],
      max_tokens: 32,
    });
    const firstOffer = leaseOffers(hub)[0]!;

    hub.workerMessage({
      v: 1,
      type: "lease.accept",
      workerId: primary.id,
      payload: { jobId: handle.jobId, leaseId: firstOffer.payload.leaseId },
    });
    hub.workerMessage({
      v: 1,
      type: "task.token",
      workerId: primary.id,
      payload: {
        jobId: handle.jobId,
        leaseId: firstOffer.payload.leaseId,
        index: 0,
        text: "",
      },
    });
    hub.workerMessage({
      v: 1,
      type: "task.fail",
      workerId: primary.id,
      payload: {
        jobId: handle.jobId,
        leaseId: firstOffer.payload.leaseId,
        code: "gpu_lost",
      },
    });

    expect(leaseOffers(hub)).toHaveLength(1);
    expect(store.getJob(handle.jobId)?.status).toBe("failed");
    expect(store.getJob(handle.jobId)?.failureCode).toBe("gpu_lost");
  });
});

function leaseOffers(hub: FakeWorkerHub): Array<{
  workerId: string;
  payload: JobPayload;
}> {
  return hub.sent
    .filter((message) => message.type === "lease.offer")
    .map((message) => ({ workerId: message.workerId, payload: message.payload as JobPayload }));
}
