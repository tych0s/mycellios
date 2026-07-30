import { z } from "zod";
import { deploymentSchema, gpuSchema, workerCapabilitiesSchema } from "./schemas.js";
import { evidenceResponseBindingSchema } from "./evidence-challenge.js";
import { runtimePerformanceProfileSchema } from "../performance/runtime-profile.js";

const MAX_IDENTIFIER_LENGTH = 256;
const MAX_TOKEN_CHUNK_BYTES = 64 * 1024;
const MAX_COMPLETION_BYTES = 2 * 1024 * 1024;
export const MAX_RUNTIME_STREAM_CHUNK_BYTES = 48 * 1024;
const MAX_RUNTIME_LINK_RTT_MS = 60_000;

const identifierSchema = z.string().min(1).max(MAX_IDENTIFIER_LENGTH);
const boundedText = (maximum: number) =>
  z.string().refine((value) => Buffer.byteLength(value, "utf8") <= maximum, {
    message: `Text exceeds the ${maximum} byte protocol limit`,
  });

const strictDeploymentSchema = deploymentSchema.strict();
const strictWireGpuSchema = gpuSchema
  .extend({
    // v0.2.34 briefly exposed the NVIDIA runtime ordinal in capabilities.
    // Accept it during the rolling upgrade, but remove it before capabilities
    // reach storage, scheduling, or public snapshots.
    runtimeDeviceIndex: z.number().int().nonnegative().max(1_024).optional(),
  })
  .strict()
  .transform((gpu) => {
    const { runtimeDeviceIndex, ...capability } = gpu;
    void runtimeDeviceIndex;
    return capability;
  });
const strictCapabilitiesSchema = workerCapabilitiesSchema
  .safeExtend({
    gpus: z.array(strictWireGpuSchema).min(1).max(16),
    limits: z
      .object({
        maxConcurrency: z.number().int().positive().max(1_024),
        maxPowerW: z.number().positive().max(100_000).optional(),
        maxTemperatureC: z.number().min(-100).max(500).optional(),
        pauseWhenForeground: z.boolean(),
      })
      .strict(),
    deployments: z.array(strictDeploymentSchema).max(256),
    network: z
      .object({
        coordinatorRttMs: z.number().nonnegative().max(3_600_000),
        uplinkMbps: z.number().nonnegative().max(10_000_000),
        downlinkMbps: z.number().nonnegative().max(10_000_000),
      })
      .strict(),
  })
  .strict();

const heartbeatSchema = z
  .object({
    draining: z.boolean(),
    pausedReason: z.string().max(300).nullable().optional(),
    activeLeases: z.array(identifierSchema).max(1_024),
    gpus: z
      .array(
        z
          .object({
            id: identifierSchema,
            freeOfferedVramMb: z.number().int().nonnegative().max(10_000_000),
            utilizationPct: z.number().min(0).max(100).optional(),
            temperatureC: z.number().min(-100).max(500).optional(),
            powerW: z.number().nonnegative().max(100_000).optional(),
          })
          .strict(),
      )
      .max(16),
    deployments: z
      .array(
        z
          .object({
            deploymentId: identifierSchema,
            freeSlots: z.number().int().nonnegative().max(1_024),
          })
          .strict(),
      )
      .max(256),
    network: z
      .object({
        coordinatorRttMs: z.number().nonnegative().max(3_600_000),
        uplinkMbps: z.number().nonnegative().max(10_000_000),
      })
      .strict(),
  })
  .strict();

const completionMetricsSchema = z
  .object({
    inputTokens: z.number().int().nonnegative().max(100_000_000),
    outputTokens: z.number().int().nonnegative().max(100_000_000),
    ttftMs: z.number().int().nonnegative().max(3_600_000),
    activeMs: z.number().int().positive().max(86_400_000),
    reusedKvTokens: z.number().int().nonnegative().max(100_000_000).optional(),
    energyWh: z.number().nonnegative().max(1_000_000).optional(),
  })
  .strict();

const envelopeFields = {
  v: z.literal(1),
  workerId: identifierSchema,
};

function envelopeSchema<TType extends string, TPayload extends z.ZodType>(
  type: TType,
  payload: TPayload,
) {
  return z
    .object({
      ...envelopeFields,
      type: z.literal(type),
      payload,
    })
    .strict();
}

export const workerHelloEnvelopeSchema = envelopeSchema(
  "worker.hello",
  z.object({}).strict(),
);

export const workerHeartbeatEnvelopeSchema = envelopeSchema(
  "worker.heartbeat",
  z
    .object({
      heartbeat: heartbeatSchema,
      capabilities: strictCapabilitiesSchema,
      metrics: z
        .object({
          ready: z.boolean(),
          activeJobs: z.number().int().nonnegative().max(1_024),
          loadedModels: z.array(z.string().min(1).max(512)).max(256),
        })
        .strict(),
    })
    .strict(),
);

export const workerGoodbyeEnvelopeSchema = envelopeSchema(
  "worker.goodbye",
  z.object({ reason: z.enum(["user_requested", "shutdown"]) }).strict(),
);

export const contributionAckEnvelopeSchema = envelopeSchema(
  "contribution.ack",
  z
    .object({
      commandId: identifierSchema,
      enabled: z.boolean(),
      changed: z.boolean(),
      applied: z.boolean(),
      error: boundedText(1_024).optional(),
    })
    .strict(),
);

export const leaseAcceptEnvelopeSchema = envelopeSchema(
  "lease.accept",
  z.object({ jobId: identifierSchema, leaseId: identifierSchema }).strict(),
);

export const leaseRejectEnvelopeSchema = envelopeSchema(
  "lease.reject",
  z
    .object({
      jobId: identifierSchema,
      leaseId: identifierSchema,
      reason: z.string().min(1).max(300),
    })
    .strict(),
);

export const taskTokenEnvelopeSchema = envelopeSchema(
  "task.token",
  z
    .object({
      jobId: identifierSchema,
      leaseId: identifierSchema,
      index: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
      text: boundedText(MAX_TOKEN_CHUNK_BYTES),
    })
    .strict(),
);

export const taskCompleteEnvelopeSchema = envelopeSchema(
  "task.complete",
  z
    .object({
      jobId: identifierSchema,
      leaseId: identifierSchema,
      text: boundedText(MAX_COMPLETION_BYTES),
      finishReason: z.enum(["stop", "length", "cancelled", "error"]),
      metrics: completionMetricsSchema,
    })
    .strict(),
);

export const taskFailEnvelopeSchema = envelopeSchema(
  "task.fail",
  z
    .object({
      jobId: identifierSchema,
      leaseId: identifierSchema,
      code: z.string().min(1).max(128),
      message: boundedText(1_024),
    })
    .strict(),
);

const runtimeRequestIdSchema = z.string().min(1).max(MAX_IDENTIFIER_LENGTH);
const runtimeStreamIdSchema = z.string().min(1).max(MAX_IDENTIFIER_LENGTH);
const runtimeStreamRecoveryTokenSchema = z.string()
  .min(16)
  .max(128)
  .regex(/^[A-Za-z0-9_-]+$/);
const runtimeStreamGenerationSchema = z.number().int().nonnegative().max(1_000_000);
const runtimeStreamOffsetSchema = z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER);
const runtimeStreamDataSchema = z.string().min(1).max(Math.ceil(MAX_RUNTIME_STREAM_CHUNK_BYTES / 3) * 4)
  .refine((value) => /^[A-Za-z0-9+/]+={0,2}$/.test(value), "Runtime stream data must be base64")
  .refine(
    (value) => Buffer.from(value, "base64").byteLength <= MAX_RUNTIME_STREAM_CHUNK_BYTES,
    `Runtime stream chunks cannot exceed ${MAX_RUNTIME_STREAM_CHUNK_BYTES} bytes`,
  );
export const runtimePreparedEnvelopeSchema = envelopeSchema(
  "runtime.prepared",
  z.object({ requestId: runtimeRequestIdSchema, ok: z.boolean(), error: z.string().max(2_048).optional() }).strict(),
);
export const runtimeReadyEnvelopeSchema = envelopeSchema(
  "runtime.ready",
  z.object({
    requestId: runtimeRequestIdSchema,
    output: z.object({
      stdout: boundedText(256 * 1024),
      stderr: boundedText(256 * 1024),
      stdoutTruncated: z.boolean(),
      stderrTruncated: z.boolean(),
    }).strict().optional(),
  }).strict(),
);
export const runtimeExitedEnvelopeSchema = envelopeSchema(
  "runtime.exited",
  z.object({
    requestId: runtimeRequestIdSchema,
    exit: z.object({ code: z.number().int().nullable(), signal: z.string().nullable(), error: z.string().max(2_048).optional() }).strict(),
    output: z.object({ stdout: boundedText(256 * 1024), stderr: boundedText(256 * 1024), stdoutTruncated: z.boolean(), stderrTruncated: z.boolean() }).strict(),
  }).strict(),
);

export const runtimeStreamOpenEnvelopeSchema = envelopeSchema(
  "runtime.stream.open",
  z.union([
    z.object({
      streamId: runtimeStreamIdSchema,
      destinationNodeId: identifierSchema,
      targetPort: z.number().int().min(1).max(65_535),
    }).strict(),
    z.object({
      streamId: runtimeStreamIdSchema,
      destinationNodeId: identifierSchema,
      targetPort: z.number().int().min(1).max(65_535),
      generation: runtimeStreamGenerationSchema,
      recoveryToken: runtimeStreamRecoveryTokenSchema,
    }).strict(),
  ]),
);

const canaryResponseBinding = evidenceResponseBindingSchema.extend({
  sampleIndex: z.number().int().min(0).max(15),
}).strict();

export const evidenceCanaryStartedEnvelopeSchema = envelopeSchema(
  "evidence.canary.started",
  canaryResponseBinding,
);

export const evidenceCanaryTokenEnvelopeSchema = envelopeSchema(
  "evidence.canary.token",
  canaryResponseBinding.extend({
    index: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
    text: boundedText(MAX_TOKEN_CHUNK_BYTES),
  }).strict(),
);

export const evidenceCanaryCompleteEnvelopeSchema = envelopeSchema(
  "evidence.canary.complete",
  canaryResponseBinding.extend({
    outputTokens: z.number().int().min(1).max(32_768),
    finishReason: z.enum(["stop", "length"]),
  }).strict(),
);

export const evidenceRuntimeCompleteEnvelopeSchema = envelopeSchema(
  "evidence.runtime.complete",
  evidenceResponseBindingSchema.extend({
    profile: runtimePerformanceProfileSchema,
  }).strict(),
);

export const evidenceChallengeFailedEnvelopeSchema = envelopeSchema(
  "evidence.challenge.failed",
  evidenceResponseBindingSchema.extend({
    reason: boundedText(512),
  }).strict(),
);
export const runtimePrepareProgressEnvelopeSchema = envelopeSchema(
  "runtime.prepare.progress",
  z.object({
    requestId: runtimeRequestIdSchema,
    stageIndex: z.number().int().nonnegative().max(255),
    layerStart: z.number().int().nonnegative().max(1_000_000),
    layerEnd: z.number().int().positive().max(1_000_000),
    state: z.enum(["preparing", "ready"]),
    packageId: z.string().length(64).regex(/^[0-9a-f]+$/).optional(),
    weightsSizeBytes: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER).optional(),
  }).strict(),
);
export const runtimeStreamOpenedEnvelopeSchema = envelopeSchema(
  "runtime.stream.opened",
  z.union([
    z.object({ streamId: runtimeStreamIdSchema }).strict(),
    z.object({
      streamId: runtimeStreamIdSchema,
      generation: runtimeStreamGenerationSchema,
      recoveryToken: runtimeStreamRecoveryTokenSchema,
    }).strict(),
  ]),
);
export const runtimeStreamDataEnvelopeSchema = envelopeSchema(
  "runtime.stream.data",
  z.union([
    z.object({
      streamId: runtimeStreamIdSchema,
      sequence: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
      data: runtimeStreamDataSchema,
    }).strict(),
    z.object({
      streamId: runtimeStreamIdSchema,
      sequence: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
      generation: runtimeStreamGenerationSchema,
      recoveryToken: runtimeStreamRecoveryTokenSchema,
      offset: runtimeStreamOffsetSchema,
      data: runtimeStreamDataSchema,
    }).strict(),
  ]),
);
export const runtimeStreamEndEnvelopeSchema = envelopeSchema(
  "runtime.stream.end",
  z.union([
    z.object({ streamId: runtimeStreamIdSchema }).strict(),
    z.object({
      streamId: runtimeStreamIdSchema,
      generation: runtimeStreamGenerationSchema,
      recoveryToken: runtimeStreamRecoveryTokenSchema,
      finalOffset: runtimeStreamOffsetSchema,
    }).strict(),
  ]),
);
export const runtimeStreamErrorEnvelopeSchema = envelopeSchema(
  "runtime.stream.error",
  z.union([
    z.object({ streamId: runtimeStreamIdSchema, message: boundedText(1_024) }).strict(),
    z.object({
      streamId: runtimeStreamIdSchema,
      generation: runtimeStreamGenerationSchema,
      recoveryToken: runtimeStreamRecoveryTokenSchema,
      message: boundedText(1_024),
    }).strict(),
  ]),
);

export const runtimeStreamAckEnvelopeSchema = envelopeSchema(
  "runtime.stream.ack",
  z.object({
    streamId: runtimeStreamIdSchema,
    generation: runtimeStreamGenerationSchema,
    recoveryToken: runtimeStreamRecoveryTokenSchema,
    acknowledgedOffset: runtimeStreamOffsetSchema,
  }).strict(),
);

export const runtimeStreamResumeEnvelopeSchema = envelopeSchema(
  "runtime.stream.resume",
  z.object({
    streamId: runtimeStreamIdSchema,
    generation: runtimeStreamGenerationSchema,
    recoveryToken: runtimeStreamRecoveryTokenSchema,
    sendOffset: runtimeStreamOffsetSchema,
    acknowledgedOffset: runtimeStreamOffsetSchema,
    receiveOffset: runtimeStreamOffsetSchema,
    bufferedFromOffset: runtimeStreamOffsetSchema,
  }).strict().superRefine((value, context) => {
    if (value.acknowledgedOffset > value.sendOffset) {
      context.addIssue({
        code: "custom",
        message: "Acknowledged offset cannot exceed sent offset",
        path: ["acknowledgedOffset"],
      });
    }
    if (
      value.bufferedFromOffset > value.acknowledgedOffset
      || value.bufferedFromOffset > value.sendOffset
    ) {
      context.addIssue({
        code: "custom",
        message: "Replay buffer cannot start after the acknowledged offset",
        path: ["bufferedFromOffset"],
      });
    }
  }),
);

export const runtimeDirectReadyEnvelopeSchema = envelopeSchema(
  "runtime.direct.ready",
  z.object({
    streamId: runtimeStreamIdSchema,
    connectionId: runtimeStreamIdSchema,
  }).strict(),
);

export const runtimeDirectFallbackEnvelopeSchema = envelopeSchema(
  "runtime.direct.fallback",
  z.object({
    streamId: runtimeStreamIdSchema,
    connectionId: runtimeStreamIdSchema,
    reason: boundedText(256),
  }).strict(),
);

export const runtimeDirectEstablishedEnvelopeSchema = envelopeSchema(
  "runtime.direct.established",
  z.object({
    streamId: runtimeStreamIdSchema,
    connectionId: runtimeStreamIdSchema,
    connectRttMs: z.number().positive().max(60_000),
  }).strict(),
);

export const runtimeDirectCommittedEnvelopeSchema = envelopeSchema(
  "runtime.direct.committed",
  z.object({
    streamId: runtimeStreamIdSchema,
    connectionId: runtimeStreamIdSchema,
  }).strict(),
);

export const runtimeDirectClosedEnvelopeSchema = envelopeSchema(
  "runtime.direct.closed",
  z.object({
    streamId: runtimeStreamIdSchema,
    connectionId: runtimeStreamIdSchema,
    bytesTx: runtimeStreamOffsetSchema,
    bytesRx: runtimeStreamOffsetSchema,
    reason: boundedText(256).optional(),
  }).strict(),
);

export const runtimeDirectTelemetryEnvelopeSchema = envelopeSchema(
  "runtime.direct.telemetry",
  z.object({
    streamId: runtimeStreamIdSchema,
    connectionId: runtimeStreamIdSchema,
    bytesTx: runtimeStreamOffsetSchema,
    bytesRx: runtimeStreamOffsetSchema,
  }).strict(),
);

export const runtimeLinkProbePingEnvelopeSchema = envelopeSchema(
  "runtime.link.probe.ping",
  z.object({
    probeId: runtimeRequestIdSchema,
    destinationNodeId: identifierSchema,
    data: runtimeStreamDataSchema,
  }).strict(),
);

export const runtimeLinkProbePongEnvelopeSchema = envelopeSchema(
  "runtime.link.probe.pong",
  z.object({
    probeId: runtimeRequestIdSchema,
    data: runtimeStreamDataSchema,
  }).strict(),
);

export const runtimeLinkProbeResultEnvelopeSchema = envelopeSchema(
  "runtime.link.probe.result",
  z.object({
    probeId: runtimeRequestIdSchema,
    destinationNodeId: identifierSchema,
    rttMs: z.number().positive().max(MAX_RUNTIME_LINK_RTT_MS).nullable(),
    goodputMbps: z.number().positive().max(10_000_000).nullable(),
  }).strict(),
);

export const workerEnvelopeSchema = z.discriminatedUnion("type", [
  workerHelloEnvelopeSchema,
  workerHeartbeatEnvelopeSchema,
  workerGoodbyeEnvelopeSchema,
  contributionAckEnvelopeSchema,
  leaseAcceptEnvelopeSchema,
  leaseRejectEnvelopeSchema,
  taskTokenEnvelopeSchema,
  taskCompleteEnvelopeSchema,
  taskFailEnvelopeSchema,
  evidenceCanaryStartedEnvelopeSchema,
  evidenceCanaryTokenEnvelopeSchema,
  evidenceCanaryCompleteEnvelopeSchema,
  evidenceRuntimeCompleteEnvelopeSchema,
  evidenceChallengeFailedEnvelopeSchema,
  runtimePrepareProgressEnvelopeSchema,
  runtimePreparedEnvelopeSchema,
  runtimeReadyEnvelopeSchema,
  runtimeExitedEnvelopeSchema,
  runtimeStreamOpenEnvelopeSchema,
  runtimeStreamOpenedEnvelopeSchema,
  runtimeStreamDataEnvelopeSchema,
  runtimeStreamEndEnvelopeSchema,
  runtimeStreamErrorEnvelopeSchema,
  runtimeStreamAckEnvelopeSchema,
  runtimeStreamResumeEnvelopeSchema,
  runtimeDirectReadyEnvelopeSchema,
  runtimeDirectFallbackEnvelopeSchema,
  runtimeDirectEstablishedEnvelopeSchema,
  runtimeDirectCommittedEnvelopeSchema,
  runtimeDirectClosedEnvelopeSchema,
  runtimeDirectTelemetryEnvelopeSchema,
  runtimeLinkProbePingEnvelopeSchema,
  runtimeLinkProbePongEnvelopeSchema,
  runtimeLinkProbeResultEnvelopeSchema,
]);

export type ValidatedWorkerEnvelope = z.infer<typeof workerEnvelopeSchema>;
export type WorkerHeartbeatPayload = z.infer<
  typeof workerHeartbeatEnvelopeSchema
>["payload"];

export function parseWorkerEnvelope(input: unknown): ValidatedWorkerEnvelope | null {
  const result = workerEnvelopeSchema.safeParse(input);
  return result.success ? result.data : null;
}

export function workerEnvelopeValidationIssues(input: unknown): string[] {
  const result = workerEnvelopeSchema.safeParse(input);
  if (result.success) return [];
  return result.error.issues.slice(0, 8).map((issue) => {
    const path = issue.path.length > 0 ? issue.path.join(".") : "envelope";
    return `${path}: ${issue.message}`;
  });
}
