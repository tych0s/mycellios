import { describe, expect, it } from "vitest";
import type { ChatCompletionRequest } from "../src/contracts/types.js";
import type { FederationManager } from "../src/coordinator/federation-manager.js";
import type {
  JobHandle,
  JobStreamEvent,
  MeshService,
} from "../src/coordinator/mesh-service.js";
import { UnifiedInferenceRouter } from "../src/coordinator/unified-inference-router.js";
import type { MeshStore } from "../src/storage/store.js";

describe("UnifiedInferenceRouter", () => {
  it("keeps exact model ids exact when it selects federation", async () => {
    const native = fakeNative([], false);
    const federation = fakeFederation();
    const router = new UnifiedInferenceRouter(
      native as unknown as MeshService,
      federation as unknown as FederationManager,
      fakeStore([]),
    );
    const request = chat("exact/model");

    const handle = router.submit(request, "s1");
    for await (const _event of handle.events) {
      // Drain the fake handle.
    }

    expect(federation.requests).toEqual([
      expect.objectContaining({ model: "exact/model" }),
    ]);
  });

  it("resolves aliases to a native model but never rewrites an exact id", async () => {
    const native = fakeNative([
      { type: "completed", result: completion("native-job") },
    ], true);
    const router = new UnifiedInferenceRouter(
      native as unknown as MeshService,
      fakeFederation() as unknown as FederationManager,
      fakeStore(["fast-7b", "quality-70b"]),
    );

    const alias = router.submit(chat("mycellios-quality"), "s1");
    for await (const _event of alias.events) {
      // Drain.
    }
    const exact = router.submit(chat("requested-exact"), "s2");
    for await (const _event of exact.events) {
      // Drain.
    }

    expect(native.requests[0]?.model).toBe("quality-70b");
    expect(native.requests[1]?.model).toBe("requested-exact");
  });

  it("does not change route after the first token", async () => {
    const native = fakeNative([
      { type: "accepted", jobId: "native-job", sessionId: "s1", route: route() },
      { type: "token", token: { index: 0, text: "partial" } },
      { type: "failed", code: "worker_disconnected", message: "gone" },
    ], true);
    const federation = fakeFederation();
    const router = new UnifiedInferenceRouter(
      native as unknown as MeshService,
      federation as unknown as FederationManager,
      fakeStore([]),
    );

    const handle = router.submit(chat("exact/model"), "s1");
    const events: JobStreamEvent[] = [];
    for await (const event of handle.events) events.push(event);

    expect(events.map((event) => event.type)).toEqual([
      "accepted",
      "token",
      "failed",
    ]);
    expect(federation.requests).toEqual([]);
  });

  it("counts native attempts against the three-attempt pre-token limit", async () => {
    const native = fakeNative([
      { type: "accepted", jobId: "native-job", sessionId: "s1", route: route() },
      {
        type: "progress",
        phase: "recovering",
        message: "retry",
        attempt: 2,
      },
      { type: "failed", code: "worker_disconnected", message: "gone" },
    ], true);
    const federation = fakeFederation();
    const router = new UnifiedInferenceRouter(
      native as unknown as MeshService,
      federation as unknown as FederationManager,
      fakeStore([]),
    );

    const handle = router.submit(chat("exact/model"), "s1");
    for await (const _event of handle.events) {
      // Drain.
    }

    expect(federation.maximumAttempts).toEqual([1]);
  });
});

function fakeNative(events: JobStreamEvent[], capacity: boolean) {
  return {
    requests: [] as ChatCompletionRequest[],
    hasCapacity: () => capacity,
    cancelMatchingActiveSession: () => null,
    cancel: () => true,
    submit(request: ChatCompletionRequest): JobHandle {
      this.requests.push(request);
      return {
        jobId: "native-job",
        sessionId: request.session_id ?? "s1",
        events: iterable(events),
      };
    },
  };
}
function fakeFederation() {
  return {
    requests: [] as ChatCompletionRequest[],
    maximumAttempts: [] as number[],
    hasCapacity: () => true,
    cancel: () => true,
    submit(
      request: ChatCompletionRequest,
      sessionId?: string,
      _idempotencyKey?: string,
      maximumAttempts = 3,
    ): JobHandle {
      this.requests.push(request);
      this.maximumAttempts.push(maximumAttempts);
      return {
        jobId: "federated-job",
        sessionId: sessionId ?? "s1",
        events: iterable([
          { type: "completed", result: completion("federated-job") },
        ]),
      };
    },
  };
}

function fakeStore(models: string[]): MeshStore {
  return {
    listWorkers: () => models.length === 0 ? [] : [{
      id: "worker",
      status: "online",
      capabilities: {
        deployments: models.map((model) => ({
          model,
          verificationState: "verified",
        })),
      },
    }],
  } as unknown as MeshStore;
}

async function* iterable(events: JobStreamEvent[]): AsyncIterable<JobStreamEvent> {
  for (const event of events) yield event;
}

function chat(model: string): ChatCompletionRequest {
  return {
    model,
    messages: [{ role: "user", content: "hello" }],
  };
}

function route() {
  return {
    routeClass: "replica" as const,
    model: "exact/model",
    region: "test",
    stages: [{
      workerId: "worker",
      deploymentId: "deployment",
      modelDigest: "sha256:test",
      stageIndex: 0,
      score: 1,
    }],
    score: 1,
    affinityHit: false,
  };
}

function completion(jobId: string) {
  return {
    jobId,
    leaseId: "lease",
    text: "ok",
    finishReason: "stop" as const,
    metrics: {
      inputTokens: 1,
      outputTokens: 1,
      ttftMs: 1,
      activeMs: 1,
    },
  };
}
