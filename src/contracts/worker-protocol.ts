import { z } from "zod";
import { deploymentSchema, gpuSchema, workerCapabilitiesSchema } from "./schemas.js";

const MAX_IDENTIFIER_LENGTH = 256;
const MAX_TOKEN_CHUNK_BYTES = 64 * 1024;
const MAX_COMPLETION_BYTES = 2 * 1024 * 1024;

const identifierSchema = z.string().min(1).max(MAX_IDENTIFIER_LENGTH);
const boundedText = (maximum: number) =>
  z.string().refine((value) => Buffer.byteLength(value, "utf8") <= maximum, {
    message: `Text exceeds the ${maximum} byte protocol limit`,
  });

const strictDeploymentSchema = deploymentSchema.strict();
const strictCapabilitiesSchema = workerCapabilitiesSchema
  .extend({
    gpus: z.array(gpuSchema.strict()).min(1).max(16),
    limits: z
      .object({
        maxConcurrency: z.number().int().positive().max(1_024),
        maxPowerW: z.number().positive().max(100_000).optional(),
        maxTemperatureC: z.number().min(-100).max(500).optional(),
        pauseWhenForeground: z.boolean(),
      })
      .strict(),
    deployments: z.array(strictDeploymentSchema).min(1).max(256),
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

export const workerEnvelopeSchema = z.discriminatedUnion("type", [
  workerHelloEnvelopeSchema,
  workerHeartbeatEnvelopeSchema,
  leaseAcceptEnvelopeSchema,
  leaseRejectEnvelopeSchema,
  taskTokenEnvelopeSchema,
  taskCompleteEnvelopeSchema,
  taskFailEnvelopeSchema,
]);

export type ValidatedWorkerEnvelope = z.infer<typeof workerEnvelopeSchema>;
export type WorkerHeartbeatPayload = z.infer<
  typeof workerHeartbeatEnvelopeSchema
>["payload"];

export function parseWorkerEnvelope(input: unknown): ValidatedWorkerEnvelope | null {
  const result = workerEnvelopeSchema.safeParse(input);
  return result.success ? result.data : null;
}
