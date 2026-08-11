import { z } from "zod";
import { engineRuntimeChallengeRequestSchema } from "./evidence-challenge.js";

export const ENGINE_RUNTIME_ACTIVATION_PLAN_SCHEMA =
  "mycellios-engine-runtime-activation-plan/1" as const;

const identifierSchema = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/);

export const engineRuntimeActivationPlanSchema = z.object({
  schema: z.literal(ENGINE_RUNTIME_ACTIVATION_PLAN_SCHEMA),
  modelId: z.string().trim().min(1).max(512),
  activationId: identifierSchema,
  routeReservationId: identifierSchema,
  workerId: identifierSchema,
  request: engineRuntimeChallengeRequestSchema,
  evidence: z.object({
    routeManifestDigest: z.string().regex(/^sha256:[0-9a-f]{64}$/),
    canaryEvidenceId: z.string().regex(/^sha256:[0-9a-f]{64}$/),
  }).strict(),
  createdAt: z.string().datetime({ offset: true }),
}).strict().superRefine((plan, context) => {
  if (plan.modelId !== plan.request.modelId) {
    context.addIssue({
      code: "custom",
      message: "engine_runtime_activation_model_binding_is_invalid",
      path: ["request", "modelId"],
    });
  }
});

export type EngineRuntimeActivationPlan = z.infer<
  typeof engineRuntimeActivationPlanSchema
>;
