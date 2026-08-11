import { z } from "zod";

export const REQUEST_RECOVERY_TRACE_SCHEMA = "mycellios-request-recovery-trace/1" as const;

const digest = z.string().regex(/^sha256:[a-f0-9]{64}$/);

export const recoveryEdgeHealthSchema = z.object({
  attempt: z.number().int().positive().max(1_000_000),
  topologyGeneration: z.number().int().positive().max(1_000_000),
  streamId: z.string().min(1).max(512),
  sourceNodeId: z.string().min(1).max(512).nullable(),
  destinationNodeId: z.string().min(1).max(512),
  transport: z.enum(["direct", "relay"]),
  state: z.enum(["negotiating", "active", "suspended", "closed"]),
  rttMs: z.number().nonnegative().finite().nullable(),
  lastProgressAt: z.number().int().nonnegative(),
  deadlineAt: z.number().int().nonnegative().nullable(),
  observedAt: z.number().int().nonnegative(),
  reasonCode: z.enum([
    "link_drop",
    "progress_frozen",
    "progress_stalled",
    "process_crash",
    "out_of_memory",
  ]),
}).strict().superRefine((edge, context) => {
  if (edge.attempt !== edge.topologyGeneration) {
    context.addIssue({
      code: "custom",
      message: "edge health attempt must match topology generation",
      path: ["topologyGeneration"],
    });
  }
});

export const recoveryAttemptTraceSchema = z.object({
  attempt: z.number().int().positive().max(1_000_000),
  topologyGeneration: z.number().int().positive().max(1_000_000),
  routeDigest: digest,
  leaseId: z.string().min(1).max(128),
  recoveryMode: z.enum([
    "initial",
    "full_retry",
    "exact_replay",
    "checkpoint_resume",
    "topology_rebuild",
  ]),
  startedAt: z.number().int().nonnegative(),
  endedAt: z.number().int().nonnegative(),
  outcome: z.enum(["failed", "succeeded"]),
  reason: z.string().min(1).max(128).nullable(),
  restoredTokenCount: z.number().int().nonnegative(),
  kvCheckpointId: digest.nullable(),
  kvContinuationProofId: digest.nullable().optional(),
  edgeHealth: recoveryEdgeHealthSchema.nullable().optional(),
}).strict().superRefine((attempt, context) => {
  if (attempt.endedAt < attempt.startedAt) {
    context.addIssue({ code: "custom", message: "attempt interval is invalid", path: ["endedAt"] });
  }
  if ((attempt.outcome === "failed") !== (attempt.reason !== null)) {
    context.addIssue({
      code: "custom",
      message: "only failed attempts carry a reason",
      path: ["reason"],
    });
  }
  if (attempt.edgeHealth && attempt.outcome !== "failed") {
    context.addIssue({
      code: "custom",
      message: "only failed attempts can carry edge health evidence",
      path: ["edgeHealth"],
    });
  }
  if (
    attempt.edgeHealth
    && (
      attempt.edgeHealth.attempt !== attempt.attempt
      || attempt.edgeHealth.topologyGeneration !== attempt.topologyGeneration
    )
  ) {
    context.addIssue({
      code: "custom",
      message: "edge health identity must match the recovery attempt",
      path: ["edgeHealth"],
    });
  }
  if (
    attempt.recoveryMode === "full_retry"
    && (
      attempt.restoredTokenCount !== 0
      || attempt.kvCheckpointId !== null
      || attempt.kvContinuationProofId != null
    )
  ) {
    context.addIssue({
      code: "custom",
      message: "full retry cannot claim committed tokens or restored KV",
      path: ["recoveryMode"],
    });
  }
  if (
    attempt.recoveryMode === "initial"
    && (
      attempt.restoredTokenCount !== 0
      || attempt.kvCheckpointId !== null
      || attempt.kvContinuationProofId != null
    )
  ) {
    context.addIssue({
      code: "custom",
      message: "initial execution cannot claim restored state",
      path: ["recoveryMode"],
    });
  }
  if (
    attempt.recoveryMode === "checkpoint_resume"
    && (attempt.kvCheckpointId === null || attempt.kvContinuationProofId == null)
  ) {
    context.addIssue({
      code: "custom",
      message: "checkpoint resume requires a bound KV checkpoint and continuation proof",
      path: ["kvContinuationProofId"],
    });
  }
  if (
    attempt.recoveryMode === "exact_replay"
    && (
      attempt.restoredTokenCount === 0
      || attempt.kvCheckpointId !== null
      || attempt.kvContinuationProofId != null
    )
  ) {
    context.addIssue({
      code: "custom",
      message: "exact replay requires a token prefix and cannot claim restored KV",
      path: ["recoveryMode"],
    });
  }
});

export const requestRecoveryTraceSchema = z.object({
  schema: z.literal(REQUEST_RECOVERY_TRACE_SCHEMA),
  requestHash: digest,
  finalMode: z.enum(["initial", "full_retry", "exact_replay", "checkpoint_resume", "topology_rebuild"]),
  attempts: z.array(recoveryAttemptTraceSchema).min(1).max(1_000_000),
}).strict().superRefine((trace, context) => {
  for (const [index, attempt] of trace.attempts.entries()) {
    if (attempt.attempt !== index + 1 || attempt.topologyGeneration !== index + 1) {
      context.addIssue({
        code: "custom",
        message: "attempt and topology generations must be contiguous",
        path: ["attempts", index],
      });
    }
    if (index === 0 && attempt.recoveryMode !== "initial") {
      context.addIssue({ code: "custom", message: "first attempt must be initial", path: ["attempts", 0] });
    }
    if (index > 0 && attempt.recoveryMode === "initial") {
      context.addIssue({ code: "custom", message: "later attempts cannot be initial", path: ["attempts", index] });
    }
    if (index < trace.attempts.length - 1 && attempt.outcome !== "failed") {
      context.addIssue({ code: "custom", message: "only the final attempt may succeed", path: ["attempts", index] });
    }
  }
  const finalAttempt = trace.attempts.at(-1);
  if (!finalAttempt || finalAttempt.outcome !== "succeeded" || finalAttempt.recoveryMode !== trace.finalMode) {
    context.addIssue({ code: "custom", message: "final recovery mode is inconsistent", path: ["finalMode"] });
  }
});

export type RecoveryAttemptTrace = z.infer<typeof recoveryAttemptTraceSchema>;
export type RecoveryEdgeHealth = z.infer<typeof recoveryEdgeHealthSchema>;
export type RequestRecoveryTrace = z.infer<typeof requestRecoveryTraceSchema>;
