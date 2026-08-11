import { z } from "zod";
import {
  engineCertificationSchema,
  engineCertificationIdentity,
  engineFamilyDescriptorSchema,
} from "../contracts/engine-family.js";
import {
  ENGINE_RUNTIME_ACTIVATION_PLAN_SCHEMA,
  engineRuntimeActivationPlanSchema,
  type EngineRuntimeActivationPlan,
} from "../contracts/engine-runtime-activation.js";
import { sha256CanonicalEvidence } from "../core/json.js";
import type { AutoDistributionRunResult } from "../distribution/auto-distribute.js";
import type { StoredWorker } from "../storage/store.js";

export const ENGINE_ACTIVATION_AUTHORITY_SCHEMA =
  "mycellios-engine-activation-authority/1" as const;

export const engineActivationAuthoritySchema = z.object({
  schema: z.literal(ENGINE_ACTIVATION_AUTHORITY_SCHEMA),
  descriptor: engineFamilyDescriptorSchema,
  certification: engineCertificationSchema,
  draftCompatibility: z.object({
    tokenizerDigest: z.string().regex(/^sha256:[0-9a-f]{64}$/),
    vocabularyDigest: z.string().regex(/^sha256:[0-9a-f]{64}$/),
  }).strict().optional(),
  probe: z.object({
    kind: z.literal("qwen3-dense-v1"),
    quantization: z.string().trim().min(1).max(64),
    attentionHeads: z.number().int().positive().max(100_000),
    kvHeads: z.number().int().positive().max(100_000),
    headDim: z.number().int().positive().max(100_000),
    referenceDecodeMsPerToken: z.number().positive().finite().max(1_000_000_000),
    referencePrefillMsPerToken: z.number().positive().finite().max(1_000_000_000),
    minimumSamples: z.number().int().min(7).max(100_000),
  }).strict(),
}).strict().superRefine((authority, context) => {
  const descriptorDigest = sha256CanonicalEvidence(authority.descriptor);
  if (
    authority.certification.descriptorDigest !== descriptorDigest
    || authority.certification.artifactManifestDigest
      !== authority.descriptor.model.manifestDigest
  ) {
    context.addIssue({
      code: "custom",
      message: "engine_activation_authority_certification_binding_is_invalid",
      path: ["certification"],
    });
  }
  if (
    authority.certification.certificationId
      !== engineCertificationIdentity(authority.certification)
  ) {
    context.addIssue({
      code: "custom",
      message: "engine_activation_authority_certification_identity_mismatch",
      path: ["certification", "certificationId"],
    });
  }
  if (authority.certification.status !== "certified") {
    context.addIssue({
      code: "custom",
      message: "engine_activation_authority_is_not_certified",
      path: ["certification", "status"],
    });
  }
  if (
    authority.probe.attentionHeads % authority.probe.kvHeads !== 0
    || authority.probe.attentionHeads * authority.probe.headDim <= 0
  ) {
    context.addIssue({
      code: "custom",
      message: "engine_activation_authority_qwen_shape_is_invalid",
      path: ["probe"],
    });
  }
});

export type EngineActivationAuthority = z.infer<
  typeof engineActivationAuthoritySchema
>;

export function buildEngineRuntimeActivationPlans(
  authorityValue: unknown,
  modelId: string,
  activationId: string,
  routeReservationId: string,
  result: AutoDistributionRunResult,
  workers: readonly StoredWorker[],
  now = Date.now(),
): EngineRuntimeActivationPlan[] {
  const authority = engineActivationAuthoritySchema.parse(authorityValue);
  if (
    !Number.isFinite(now)
    || Date.parse(authority.certification.validFrom) > now
    || Date.parse(authority.certification.expiresAt) <= now
  ) throw new Error("engine_activation_authority_certification_is_not_current");
  if (
    authority.descriptor.model.revision !== result.profile.source.snapshotCommit
    || authority.descriptor.model.manifestDigest !== result.profile.source.artifactIdentity
    || authority.descriptor.model.modelId !== result.profile.source.model
    || result.manifest.hiddenSize
      !== authority.probe.attentionHeads * authority.probe.headDim
  ) throw new Error("engine_activation_authority_model_binding_is_invalid");

  const descriptorDigest = sha256CanonicalEvidence(authority.descriptor) as `sha256:${string}`;
  const workerByNode = new Map(workers.flatMap((worker) => {
    const nodeId = worker.capabilities.distributedExecutor?.nodeId;
    return nodeId ? [[nodeId, worker] as const] : [];
  }));
  const createdAt = new Date(now).toISOString();
  const routeManifestDigest = sha256CanonicalEvidence(result.manifest) as `sha256:${string}`;
  const canaryEvidenceId = sha256CanonicalEvidence({
    activationId,
    routeManifestDigest,
    canaryText: result.canaryText,
    canaryMetrics: result.canaryMetrics,
  }) as `sha256:${string}`;
  const plans: EngineRuntimeActivationPlan[] = [];
  for (const stage of result.manifest.plans.decode.stages) {
    for (const member of stage.members) {
      const worker = workerByNode.get(member.nodeId);
      const executor = worker?.capabilities.distributedExecutor;
      const performance = executor?.performanceEvidence;
      const physical = executor?.physicalIdentity;
      if (!worker || !executor || !performance || !physical) {
        throw new Error(`engine_activation_stage_evidence_is_missing:${member.nodeId}:${stage.index}`);
      }
      const backend = performance.profile.backend;
      const contextTokens = result.request.workload.contextTokens;
      const target = authority.descriptor.targets.find((candidate) =>
        candidate.backend === backend
        && candidate.quantizations.includes(authority.probe.quantization)
        && candidate.context.minTokens <= contextTokens
        && candidate.context.maxTokens >= contextTokens
      );
      const certifiedTarget = authority.certification.targets.find((candidate) =>
        target
        && candidate.platform === target.platform
        && candidate.arch === target.arch
        && candidate.backend === target.backend
        && candidate.runtimeAbi === target.runtimeAbi
        && candidate.quantizations.includes(authority.probe.quantization)
      );
      if (!target || !certifiedTarget) {
        throw new Error(`engine_activation_certified_target_is_missing:${member.nodeId}`);
      }
      const hardwareClasses = worker.capabilities.gpus.map((gpu) =>
        `${gpu.vendor}-${gpu.model}`.toLowerCase().replace(/[^a-z0-9._:-]+/g, "-")
          .replace(/^-+|-+$/g, "").slice(0, 256)
      );
      if (!hardwareClasses.some((hardwareClass) =>
        authority.certification.hardwareClasses.includes(hardwareClass)
      )) throw new Error(`engine_activation_hardware_is_not_certified:${member.nodeId}`);
      const layers = result.profile.model.layers.slice(stage.layerStart, stage.layerEnd);
      if (layers.length !== stage.layerEnd - stage.layerStart) {
        throw new Error("engine_activation_stage_layer_range_is_invalid");
      }
      const roles: Array<"head" | "middle" | "tail"> = [];
      if (stage.index === 0) roles.push("head");
      if (stage.index > 0 && stage.index < result.manifest.plans.decode.stages.length - 1) {
        roles.push("middle");
      }
      if (stage.index === result.manifest.plans.decode.stages.length - 1) roles.push("tail");
      plans.push(engineRuntimeActivationPlanSchema.parse({
        schema: ENGINE_RUNTIME_ACTIVATION_PLAN_SCHEMA,
        modelId,
        activationId,
        routeReservationId,
        workerId: worker.id,
        createdAt,
        evidence: { routeManifestDigest, canaryEvidenceId },
        request: {
          probeKind: authority.probe.kind,
          descriptorDigest,
          certificationId: authority.certification.certificationId,
          artifactManifestDigest: authority.certification.artifactManifestDigest,
          modelId,
          modelRevision: authority.descriptor.model.revision,
          backend,
          runtimeAbi: target.runtimeAbi,
          quantization: authority.probe.quantization,
          contextTokens,
          expectedLayerStart: stage.layerStart,
          expectedLayerEnd: stage.layerEnd,
          expectedKvBytesPerToken: layers.reduce((sum, layer) => sum + layer.kvBytesPerToken, 0),
          expectedLayerWeightBytes: layers.reduce((sum, layer) => sum + layer.weightBytes, 0),
          referenceDecodeMsPerToken: authority.probe.referenceDecodeMsPerToken,
          referencePrefillMsPerToken: authority.probe.referencePrefillMsPerToken,
          hiddenSize: result.manifest.hiddenSize,
          attentionHeads: authority.probe.attentionHeads,
          kvHeads: authority.probe.kvHeads,
          headDim: authority.probe.headDim,
          requiredRoles: roles,
          minimumSamples: authority.probe.minimumSamples,
        },
      }));
    }
  }
  return plans;
}
