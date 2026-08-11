import { z } from "zod";
import {
  classifyRecoveryFailure,
  recoveryEventSchema,
  type RecoveryEvent,
} from "./recovery-outcome.js";

export const RUNTIME_LINK_FAILURE_SCHEMA = "mycellios-runtime-link-failure/1" as const;

export const runtimeLinkFailureEvidenceSchema = z.object({
  schema: z.literal(RUNTIME_LINK_FAILURE_SCHEMA),
  streamId: z.string().min(1).max(128),
  sourceNodeId: z.string().min(1).max(256).nullable(),
  destinationNodeId: z.string().min(1).max(256).nullable(),
  generation: z.number().int().nonnegative().safe(),
  role: z.enum(["direct", "relay", "coordinator"]),
  failureClass: z.enum(["direct-link-lost", "relay-link-lost", "coordinator-lost"]),
  transportMode: z.enum(["direct", "relay"]),
  checkpointKind: z.enum(["none", "stream-offset"]),
  sourceOffset: z.number().int().nonnegative().safe(),
  destinationOffset: z.number().int().nonnegative().safe(),
  observedAt: z.number().int().nonnegative().safe(),
  reason: z.string().min(1).max(256),
}).strict().superRefine((evidence, context) => {
  const roleBindingIsValid = evidence.role === "direct"
    ? evidence.failureClass === "direct-link-lost"
      && evidence.transportMode === "direct"
      && evidence.checkpointKind === "none"
    : evidence.role === "relay"
      ? evidence.failureClass === "relay-link-lost"
        && evidence.transportMode === "relay"
        && evidence.checkpointKind === "stream-offset"
      : evidence.failureClass === "coordinator-lost"
        && evidence.transportMode === "relay";
  if (!roleBindingIsValid) {
    context.addIssue({ code: "custom", message: "runtime_link_failure_role_binding_is_invalid" });
  }
});

export type RuntimeLinkFailureEvidence = z.infer<typeof runtimeLinkFailureEvidenceSchema>;

export function classifyRuntimeLinkFailure(
  value: unknown,
  activeGeneration: number,
  now = Date.now(),
): RecoveryEvent {
  const evidence = runtimeLinkFailureEvidenceSchema.parse(value);
  if (!Number.isSafeInteger(activeGeneration) || activeGeneration < 0 || now < evidence.observedAt) {
    throw new Error("runtime_link_failure_classification_context_is_invalid");
  }
  return recoveryEventSchema.parse(classifyRecoveryFailure({
    generation: evidence.generation,
    activeGeneration,
    role: evidence.role,
    failureClass: evidence.failureClass,
    checkpointKind: evidence.checkpointKind,
    checkpointCompatible: evidence.checkpointKind === "stream-offset",
    compatibleStandbyAvailable: false,
    visibleTokens: 0,
    downtimeMs: now - evidence.observedAt,
    discardedWaves: 0,
    discardedBytes: Math.max(evidence.sourceOffset, evidence.destinationOffset),
  }));
}

export function runtimeLinkRecoveryEventsForReceipt(
  values: readonly unknown[],
  activeGeneration: number,
  receiptObservedAt: number,
): RecoveryEvent[] {
  if (values.length > 256) throw new Error("runtime_link_failure_receipt_limit_exceeded");
  const evidence = values.map((value) => runtimeLinkFailureEvidenceSchema.parse(value));
  const identities = new Set<string>();
  for (const item of evidence) {
    const identity = `${item.streamId}:${item.generation}:${item.role}:${item.observedAt}`;
    if (identities.has(identity)) throw new Error("runtime_link_failure_receipt_duplicate");
    identities.add(identity);
  }
  return evidence
    .sort((left, right) => left.observedAt - right.observedAt || left.streamId.localeCompare(right.streamId))
    .map((item) => classifyRuntimeLinkFailure(item, activeGeneration, receiptObservedAt));
}
