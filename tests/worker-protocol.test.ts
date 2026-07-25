import { describe, expect, it } from "vitest";
import {
  parseWorkerEnvelope,
  taskCompleteEnvelopeSchema,
  taskTokenEnvelopeSchema,
  runtimeExitedEnvelopeSchema,
  runtimePreparedEnvelopeSchema,
  runtimeReadyEnvelopeSchema,
  runtimeLinkProbePingEnvelopeSchema,
  runtimeLinkProbePongEnvelopeSchema,
  runtimeLinkProbeResultEnvelopeSchema,
  runtimeDirectClosedEnvelopeSchema,
  runtimeDirectEstablishedEnvelopeSchema,
  runtimeDirectFallbackEnvelopeSchema,
  runtimeDirectReadyEnvelopeSchema,
  runtimeDirectTelemetryEnvelopeSchema,
  runtimeStreamAckEnvelopeSchema,
  runtimeStreamDataEnvelopeSchema,
  runtimeStreamOpenEnvelopeSchema,
  runtimeStreamResumeEnvelopeSchema,
  workerGoodbyeEnvelopeSchema,
  workerEnvelopeSchema,
  workerHeartbeatEnvelopeSchema,
  workerHelloEnvelopeSchema,
} from "../src/contracts/worker-protocol.js";
import {
  createCoordinatorRuntimePerformanceEvidence,
  sealRuntimePerformanceProfile,
} from "../src/performance/runtime-profile.js";

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

    const legacyHeartbeat = workerHeartbeatEnvelopeSchema.parse(heartbeat);
    expect(legacyHeartbeat.payload.capabilities.distributedExecutor).toMatchObject({
      computeMode: "automatic",
      cpuEligible: false,
    });
    const physicalProfile = sealRuntimePerformanceProfile({
      measuredAt: new Date().toISOString(),
      backend: "cuda",
      deviceName: "NVIDIA test-gpu",
      precision: "float16",
      source: "physical-microbenchmark",
      activationCodecId: "fp16",
      decodeMemory: profileSeries("GB/s", 400),
      prefillCompute: profileSeries("TFLOP/s", 20),
      activationCodec: profileSeries("GB/s", 2),
    });
    const measuredAt = Date.parse(physicalProfile.measuredAt);
    const performanceEvidence = createCoordinatorRuntimePerformanceEvidence({
      challengeId: "challenge-profile",
      nonce: Buffer.alloc(32, 8).toString("base64url"),
      workerId: "desktop-worker",
      sessionId: "session-profile",
      nodeId: "desktop-test",
      issuedAt: new Date(measuredAt - 1_000).toISOString(),
      expiresAt: new Date(measuredAt + 60_000).toISOString(),
      observedAt: new Date(measuredAt + 1_000).toISOString(),
      profile: physicalProfile,
    });
    const calibratedHeartbeat = workerHeartbeatEnvelopeSchema.parse({
      ...heartbeat,
      payload: {
        ...heartbeat.payload,
        capabilities: {
          ...heartbeat.payload.capabilities,
          distributedExecutor: {
            ...heartbeat.payload.capabilities.distributedExecutor,
            performanceEvidence,
          },
        },
      },
    });
    expect(
      calibratedHeartbeat.payload.capabilities.distributedExecutor?.performanceEvidence,
    ).toEqual(performanceEvidence);
    expect(workerHeartbeatEnvelopeSchema.safeParse({
      ...heartbeat,
      payload: {
        ...heartbeat.payload,
        capabilities: {
          ...heartbeat.payload.capabilities,
          distributedExecutor: {
            ...heartbeat.payload.capabilities.distributedExecutor,
            performanceEvidence: {
              ...performanceEvidence,
              profile: {
                ...physicalProfile,
                decodeMemory: { ...physicalProfile.decodeMemory, p50: 999 },
              },
            },
          },
        },
      },
    }).success).toBe(false);

    const cpuAuthorizedHeartbeat = workerHeartbeatEnvelopeSchema.parse({
      ...heartbeat,
      payload: {
        ...heartbeat.payload,
        capabilities: {
          ...heartbeat.payload.capabilities,
          distributedExecutor: {
            ...heartbeat.payload.capabilities.distributedExecutor,
            computeMode: "cpu-only",
            cpuEligible: true,
            acceleration: {
              schema: "mycellios-accelerator-diagnostics/1",
              appVersion: "0.2.26",
              state: "gpu-fallback",
              backend: "cpu",
              deviceName: "NVIDIA GeForce RTX 2060",
              gpuVendor: "nvidia",
              gpuModel: "NVIDIA GeForce RTX 2060",
              phase: "physical-probe",
              progressPct: 95,
              issueCode: "physical-probe",
              issueSummary: "CUDA probe failed",
              retryable: true,
              retryAttempt: 2,
              nextRetryAt: "2026-07-23T10:30:00.000Z",
              updatedAt: "2026-07-23T10:00:00.000Z",
              recentEvents: [],
            },
          },
        },
      },
    });
    expect(cpuAuthorizedHeartbeat.payload.capabilities.distributedExecutor).toMatchObject({
      computeMode: "cpu-only",
      cpuEligible: true,
      acceleration: {
        schema: "mycellios-accelerator-diagnostics/1",
        issueCode: "physical-probe",
        retryAttempt: 2,
      },
    });
    const nvidiaOrdinalHeartbeat = workerHeartbeatEnvelopeSchema.parse({
      ...heartbeat,
      payload: {
        ...heartbeat.payload,
        capabilities: {
          ...heartbeat.payload.capabilities,
          gpus: [{
            ...heartbeat.payload.capabilities.gpus[0],
            runtimeDeviceIndex: 0,
          }],
        },
      },
    });
    expect(nvidiaOrdinalHeartbeat.payload.capabilities.gpus[0]).not.toHaveProperty("runtimeDeviceIndex");
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
    expect(runtimeReadyEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.ready",
      payload: {
        requestId: "run-1",
        output: {
          stdout: "root online",
          stderr: '{"execution":{"backend":"cuda"}}',
          stdoutTruncated: false,
          stderrTruncated: false,
        },
      },
    }).success).toBe(true);
    // gdlp-worker-tunnel/2 remains rolling-upgrade compatible with workers
    // released before readiness output was introduced.
    expect(runtimeReadyEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.ready",
      payload: { requestId: "run-legacy" },
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
    const recoveryToken = "recovery_token_0123456789";
    expect(runtimeStreamOpenEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.stream.open",
      payload: {
        streamId: "stream-recovery",
        destinationNodeId: "desktop-b",
        targetPort: 9_850,
        generation: 0,
        recoveryToken,
      },
    }).success).toBe(true);
    expect(runtimeStreamDataEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.stream.data",
      payload: {
        streamId: "stream-recovery",
        sequence: 0,
        generation: 0,
        recoveryToken,
        offset: 0,
        data: Buffer.from("hello").toString("base64"),
      },
    }).success).toBe(true);
    // Recovery metadata is atomic. A partially upgraded or truncated envelope
    // must never be interpreted as either a legacy or a recoverable chunk.
    expect(runtimeStreamDataEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.stream.data",
      payload: {
        streamId: "stream-recovery",
        sequence: 0,
        generation: 0,
        data: Buffer.from("hello").toString("base64"),
      },
    }).success).toBe(false);
    expect(runtimeStreamAckEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.stream.ack",
      payload: {
        streamId: "stream-recovery",
        generation: 0,
        recoveryToken,
        acknowledgedOffset: 5,
      },
    }).success).toBe(true);
    expect(runtimeStreamResumeEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.stream.resume",
      payload: {
        streamId: "stream-recovery",
        generation: 0,
        recoveryToken,
        sendOffset: 5,
        acknowledgedOffset: 0,
        receiveOffset: 0,
        bufferedFromOffset: 0,
      },
    }).success).toBe(true);
    expect(runtimeStreamResumeEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.stream.resume",
      payload: {
        streamId: "stream-recovery",
        generation: 0,
        recoveryToken,
        sendOffset: 5,
        acknowledgedOffset: 6,
        receiveOffset: 0,
        bufferedFromOffset: 0,
      },
    }).success).toBe(false);
    expect(runtimeDirectReadyEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.direct.ready",
      payload: { streamId: "stream-direct", connectionId: "stream-direct" },
    }).success).toBe(true);
    expect(runtimeDirectEstablishedEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.direct.established",
      payload: {
        streamId: "stream-direct",
        connectionId: "stream-direct",
        connectRttMs: 2.4,
      },
    }).success).toBe(true);
    expect(runtimeDirectFallbackEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.direct.fallback",
      payload: {
        streamId: "stream-direct",
        connectionId: "stream-direct",
        reason: "candidate_unreachable",
      },
    }).success).toBe(true);
    expect(runtimeDirectClosedEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.direct.closed",
      payload: {
        streamId: "stream-direct",
        connectionId: "stream-direct",
        bytesTx: 12,
        bytesRx: 8,
        secret: "must_not_cross_the_protocol",
      },
    }).success).toBe(false);
    expect(runtimeDirectTelemetryEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.direct.telemetry",
      payload: {
        streamId: "stream-direct",
        connectionId: "stream-direct",
        bytesTx: 12,
        bytesRx: 8,
      },
    }).success).toBe(true);
    expect(runtimeStreamDataEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.stream.data",
      payload: { streamId: "stream-1", sequence: 0, data: Buffer.alloc(48 * 1024 + 1).toString("base64") },
    }).success).toBe(false);
    const probeData = Buffer.alloc(16 * 1024, 7).toString("base64");
    expect(runtimeLinkProbePingEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.link.probe.ping",
      payload: {
        probeId: "probe-1",
        destinationNodeId: "desktop-b",
        data: probeData,
      },
    }).success).toBe(true);
    expect(runtimeLinkProbePongEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.link.probe.pong",
      payload: { probeId: "probe-1", data: probeData },
    }).success).toBe(true);
    expect(runtimeLinkProbeResultEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.link.probe.result",
      payload: {
        probeId: "probe-1",
        destinationNodeId: "desktop-b",
        rttMs: 42.5,
        goodputMbps: 6.2,
      },
    }).success).toBe(true);
    expect(runtimeLinkProbeResultEnvelopeSchema.safeParse({
      ...baseEnvelope,
      type: "runtime.link.probe.result",
      payload: {
        probeId: "probe-1",
        destinationNodeId: "desktop-b",
        rttMs: null,
        goodputMbps: null,
      },
    }).success).toBe(true);
  });
});

function profileSeries(unit: "GB/s" | "TFLOP/s", median: number) {
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
