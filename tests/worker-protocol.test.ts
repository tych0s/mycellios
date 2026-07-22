import { describe, expect, it } from "vitest";
import {
  parseWorkerEnvelope,
  taskCompleteEnvelopeSchema,
  taskTokenEnvelopeSchema,
  runtimeExitedEnvelopeSchema,
  runtimePreparedEnvelopeSchema,
  runtimeStreamDataEnvelopeSchema,
  runtimeStreamOpenEnvelopeSchema,
  workerGoodbyeEnvelopeSchema,
  workerEnvelopeSchema,
  workerHeartbeatEnvelopeSchema,
  workerHelloEnvelopeSchema,
} from "../src/contracts/worker-protocol.js";

const baseEnvelope = {
  v: 1 as const,
  workerId: "wrk-test",
};

describe("worker protocol schemas", () => {
  it("accepts a well-formed hello and rejects unknown envelope or payload fields", () => {
    const valid = {
      ...baseEnvelope,
      type: "worker.hello" as const,
      payload: {},
    };

    expect(workerHelloEnvelopeSchema.safeParse(valid).success).toBe(true);
    expect(workerHelloEnvelopeSchema.safeParse({ ...valid, unexpected: true }).success).toBe(false);
    expect(
      workerHelloEnvelopeSchema.safeParse({
        ...valid,
        payload: { ...valid.payload, unexpected: true },
      }).success,
    ).toBe(false);
  });

  it("rejects null, arrays, unknown message types, and invalid field types", () => {
    expect(parseWorkerEnvelope(null)).toBeNull();
    expect(parseWorkerEnvelope([])).toBeNull();
    expect(
      parseWorkerEnvelope({
        ...baseEnvelope,
        type: "worker.unknown",
        payload: {},
      }),
    ).toBeNull();
    expect(
      workerEnvelopeSchema.safeParse({
        ...baseEnvelope,
        workerId: 42,
        type: "worker.hello",
        payload: {},
      }).success,
    ).toBe(false);
  });

  it("accepts only an explicit bounded worker goodbye", () => {
    const goodbye = {
      ...baseEnvelope,
      type: "worker.goodbye" as const,
      payload: { reason: "user_requested" as const },
    };
    expect(workerGoodbyeEnvelopeSchema.safeParse(goodbye).success).toBe(true);
    expect(workerGoodbyeEnvelopeSchema.safeParse({
      ...goodbye,
      payload: { reason: "network_error" },
    }).success).toBe(false);
  });

  it("enforces UTF-8 byte limits for token chunks and completion output", () => {
    const token = {
      ...baseEnvelope,
      type: "task.token" as const,
      payload: { jobId: "job-1", leaseId: "lease-1", index: 0, text: "ok" },
    };
    expect(taskTokenEnvelopeSchema.safeParse(token).success).toBe(true);
    expect(
      taskTokenEnvelopeSchema.safeParse({
        ...token,
        payload: { ...token.payload, text: "€".repeat(22_000) },
      }).success,
    ).toBe(false);

    const completion = {
      ...baseEnvelope,
      type: "task.complete" as const,
      payload: {
        jobId: "job-1",
        leaseId: "lease-1",
        text: "done",
        finishReason: "stop" as const,
        metrics: {
          inputTokens: 10,
          outputTokens: 1,
          ttftMs: 20,
          activeMs: 100,
        },
      },
    };
    expect(taskCompleteEnvelopeSchema.safeParse(completion).success).toBe(true);
    expect(
      taskCompleteEnvelopeSchema.safeParse({
        ...completion,
        payload: { ...completion.payload, text: "x".repeat(2 * 1024 * 1024 + 1) },
      }).success,
    ).toBe(false);
  });

  it("validates a strict heartbeat including bounded capabilities and metrics", () => {
    const heartbeat = {
      ...baseEnvelope,
      type: "worker.heartbeat" as const,
      payload: {
        heartbeat: {
          draining: false,
          pausedReason: null,
          activeLeases: [],
          gpus: [{ id: "gpu-0", freeOfferedVramMb: 4_096 }],
          deployments: [{ deploymentId: "dep-0", freeSlots: 1 }],
          network: { coordinatorRttMs: 10, uplinkMbps: 100 },
        },
        capabilities: {
          region: "es-mad",
          agentVersion: "0.1.0",
          gpus: [
            {
              id: "gpu-0",
              vendor: "test",
              model: "test-gpu",
              physicalVramMb: 8_192,
              offeredVramMb: 4_096,
              freeOfferedVramMb: 4_096,
            },
          ],
          limits: { maxConcurrency: 1, pauseWhenForeground: true },
          deployments: [
            {
              deploymentId: "dep-0",
              model: "distributed-small",
              modelDigest: "sha256:model",
              mode: "replica" as const,
              adapter: "mock" as const,
              peakVramMb: 3_000,
              contextLimit: 8_192,
              maxConcurrency: 1,
              freeSlots: 1,
              tokensPerSecond: 10,
              ttftMs: 100,
              dataLocality: "local" as const,
            },
          ],
          network: { coordinatorRttMs: 10, uplinkMbps: 100, downlinkMbps: 100 },
          distributedExecutor: {
            protocol: "gdlp-worker-tunnel/2" as const,
            nodeId: "desktop-test",
            stageHost: "192.168.1.20",
            stagePort: 9_850,
            runtime: "python-safetensors" as const,
          },
        },
        metrics: { ready: true, activeJobs: 0, loadedModels: ["distributed-small"] },
      },
    };

    expect(workerHeartbeatEnvelopeSchema.safeParse(heartbeat).success).toBe(true);
    expect(workerHeartbeatEnvelopeSchema.safeParse({
      ...heartbeat,
      payload: {
        ...heartbeat.payload,
        heartbeat: { ...heartbeat.payload.heartbeat, deployments: [] },
        capabilities: { ...heartbeat.payload.capabilities, deployments: [] },
        metrics: { ...heartbeat.payload.metrics, loadedModels: [] },
      },
    }).success).toBe(true);
    expect(
      workerHeartbeatEnvelopeSchema.safeParse({
        ...heartbeat,
        payload: {
          ...heartbeat.payload,
          metrics: { ...heartbeat.payload.metrics, secret: "must-not-pass" },
        },
      }).success,
    ).toBe(false);
  });

  it("accepts bounded shard-runtime lifecycle responses", () => {
    expect(runtimePreparedEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.prepared",
      payload: { requestId: "prepare-1", ok: true },
    }).success).toBe(true);
    expect(runtimeExitedEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.exited",
      payload: {
        requestId: "run-1",
        exit: { code: 0, signal: null },
        output: { stdout: "ready", stderr: "", stdoutTruncated: false, stderrTruncated: false },
      },
    }).success).toBe(true);
    expect(runtimeStreamOpenEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.stream.open",
      payload: { streamId: "stream-1", destinationNodeId: "desktop-b", targetPort: 9_850 },
    }).success).toBe(true);
    expect(runtimeStreamDataEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.stream.data",
      payload: { streamId: "stream-1", sequence: 0, data: Buffer.from("hello").toString("base64") },
    }).success).toBe(true);
    expect(runtimeStreamDataEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.stream.data",
      payload: { streamId: "stream-1", sequence: 0, data: Buffer.alloc(48 * 1024 + 1).toString("base64") },
    }).success).toBe(false);
  });
});
