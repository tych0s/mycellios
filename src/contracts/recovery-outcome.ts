import { z } from "zod";

export const RECOVERY_EVENT_SCHEMA = "mycellios-recovery-event/1" as const;

export const recoveryRoleSchema = z.enum([
  "root", "head", "middle", "tail", "direct", "relay", "coordinator",
]);

export const recoveryFailureClassSchema = z.enum([
  "process-exit",
  "health-failed",
  "direct-link-lost",
  "relay-link-lost",
  "coordinator-lost",
  "generation-superseded",
  "checkpoint-corrupt",
  "identity-mismatch",
  "retry-exhausted",
]);

export const recoveryCheckpointKindSchema = z.enum([
  "none",
  "stream-offset",
  "visible-token-prefix",
  "activation-kv",
]);

export const recoveryReplayScopeSchema = z.enum([
  "none",
  "stream-offset",
  "visible-token-prefix",
  "full-request",
]);

export const recoveryEventSchema = z.object({
  schema: z.literal(RECOVERY_EVENT_SCHEMA),
  generation: z.number().int().nonnegative().safe(),
  role: recoveryRoleSchema,
  failureClass: recoveryFailureClassSchema,
  outcome: z.enum(["resume", "exact-replay", "safe-retry", "terminal"]),
  checkpointKind: recoveryCheckpointKindSchema,
  replayScope: recoveryReplayScopeSchema,
  downtimeMs: z.number().nonnegative().finite(),
  replayedTokens: z.number().int().nonnegative().safe(),
  discardedWaves: z.number().int().nonnegative().safe(),
  discardedBytes: z.number().int().nonnegative().safe(),
}).strict().superRefine((event, context) => {
  if (event.outcome === "resume" && event.replayScope === "full-request") {
    context.addIssue({ code: "custom", message: "recovery_resume_scope_is_invalid", path: ["replayScope"] });
  }
  if (event.outcome === "terminal" && event.replayScope !== "none") {
    context.addIssue({ code: "custom", message: "recovery_terminal_cannot_replay", path: ["replayScope"] });
  }
  if (event.replayScope === "none" && event.replayedTokens !== 0) {
    context.addIssue({ code: "custom", message: "recovery_replayed_tokens_without_scope", path: ["replayedTokens"] });
  }
});

export const recoveryFailureObservationSchema = z.object({
  generation: z.number().int().nonnegative().safe(),
  activeGeneration: z.number().int().nonnegative().safe(),
  role: recoveryRoleSchema,
  failureClass: recoveryFailureClassSchema,
  checkpointKind: recoveryCheckpointKindSchema,
  checkpointCompatible: z.boolean(),
  compatibleStandbyAvailable: z.boolean(),
  visibleTokens: z.number().int().nonnegative().safe(),
  downtimeMs: z.number().nonnegative().finite(),
  discardedWaves: z.number().int().nonnegative().safe(),
  discardedBytes: z.number().int().nonnegative().safe(),
}).strict().refine(
  ({ generation, activeGeneration }) => generation <= activeGeneration,
  { message: "recovery_failure_generation_is_from_the_future", path: ["generation"] },
);

export type RecoveryEvent = z.infer<typeof recoveryEventSchema>;
export type RecoveryFailureObservation = z.infer<typeof recoveryFailureObservationSchema>;

/**
 * Pure fail-closed recovery classifier shared by the runtime lifecycle and the
 * signed receipt boundary. It never infers checkpoint compatibility from the
 * failure role: compatibility must already be established by the relevant
 * identity/hash verifier.
 */
export function classifyRecoveryFailure(value: unknown): RecoveryEvent {
  const input = recoveryFailureObservationSchema.parse(value);
  const terminal = (
    failureClass: RecoveryEvent["failureClass"] = input.failureClass,
  ): RecoveryEvent => recoveryEventSchema.parse({
    schema: RECOVERY_EVENT_SCHEMA,
    generation: input.generation,
    role: input.role,
    failureClass,
    outcome: "terminal",
    checkpointKind: input.checkpointKind,
    replayScope: "none",
    downtimeMs: input.downtimeMs,
    replayedTokens: 0,
    discardedWaves: input.discardedWaves,
    discardedBytes: input.discardedBytes,
  });

  if (input.generation < input.activeGeneration) {
    return terminal("generation-superseded");
  }
  if (
    input.failureClass === "checkpoint-corrupt"
    || input.failureClass === "identity-mismatch"
    || input.failureClass === "retry-exhausted"
    || !input.checkpointCompatible
  ) return terminal();

  let outcome: RecoveryEvent["outcome"];
  let replayScope: RecoveryEvent["replayScope"];
  let replayedTokens = 0;
  if (input.checkpointKind === "stream-offset") {
    outcome = "resume";
    replayScope = "stream-offset";
  } else if (input.checkpointKind === "activation-kv" && input.compatibleStandbyAvailable) {
    outcome = "resume";
    replayScope = "none";
  } else if (
    input.checkpointKind === "visible-token-prefix"
    && input.compatibleStandbyAvailable
    && input.visibleTokens > 0
  ) {
    outcome = "exact-replay";
    replayScope = "visible-token-prefix";
    replayedTokens = input.visibleTokens;
  } else if (input.visibleTokens === 0 && input.compatibleStandbyAvailable) {
    outcome = "safe-retry";
    replayScope = "full-request";
  } else {
    return terminal();
  }

  return recoveryEventSchema.parse({
    schema: RECOVERY_EVENT_SCHEMA,
    generation: input.generation,
    role: input.role,
    failureClass: input.failureClass,
    outcome,
    checkpointKind: input.checkpointKind,
    replayScope,
    downtimeMs: input.downtimeMs,
    replayedTokens,
    discardedWaves: input.discardedWaves,
    discardedBytes: input.discardedBytes,
  });
}
