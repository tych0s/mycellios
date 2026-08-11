import { z } from "zod";

export const RUNTIME_RECOVERY_EVIDENCE_SCHEMA = "mycellios-runtime-recovery/1" as const;

const digest = z.string().regex(/^sha256:[a-f0-9]{64}$/);

/** Worker-authored evidence for recovery performed inside one leased runtime. */
export const runtimeRecoveryEvidenceSchema = z.object({
  schema: z.literal(RUNTIME_RECOVERY_EVIDENCE_SCHEMA),
  mode: z.literal("topology_rebuild"),
  attempts: z.number().int().min(2).max(1_000_000),
  topologyGeneration: z.number().int().min(2).max(1_000_000),
  restoredTokenCount: z.number().int().nonnegative().max(100_000_000),
  checkpointKind: z.enum(["immutable_prompt", "visible_token_prefix"]),
  checkpointId: digest,
  recoveryIdentity: digest,
}).strict().superRefine((evidence, context) => {
  if (evidence.topologyGeneration !== evidence.attempts) {
    context.addIssue({
      code: "custom",
      message: "topology generation must match the completed runtime attempt",
      path: ["topologyGeneration"],
    });
  }
  if (
    (evidence.checkpointKind === "visible_token_prefix")
    !== (evidence.restoredTokenCount > 0)
  ) {
    context.addIssue({
      code: "custom",
      message: "checkpoint kind must match whether a visible token prefix was restored",
      path: ["checkpointKind"],
    });
  }
});

export type RuntimeRecoveryEvidence = z.infer<typeof runtimeRecoveryEvidenceSchema>;
