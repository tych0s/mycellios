import {
  engineCertificationSchema,
  engineCertificationIdentity,
  engineFamilyDescriptorSchema,
  type EngineCertification,
  type EngineFamilyDescriptor,
  type EngineTarget,
} from "../contracts/engine-family.js";
import { sha256CanonicalEvidence } from "../core/json.js";

export interface EngineResolutionRequest {
  descriptorDigest: `sha256:${string}`;
  modelId: string;
  revision: string;
  manifestDigest: `sha256:${string}`;
  tensorAbi: string;
  platform: EngineTarget["platform"];
  arch: EngineTarget["arch"];
  backend: EngineTarget["backend"];
  runtimeAbi: string;
  quantization: string;
  contextTokens: number;
  hardwareClass: string;
  generationMode: "greedy" | "sampling";
  now: Date;
}

export interface ResolvedEngineFamily {
  descriptor: EngineFamilyDescriptor;
  certification: EngineCertification;
  target: EngineTarget;
}

export class EngineFamilyResolutionError extends Error {
  constructor(readonly code: string) {
    super(code);
    this.name = "EngineFamilyResolutionError";
  }
}

export class NativeEngineFamilyRegistry {
  readonly #descriptors = new Map<string, EngineFamilyDescriptor>();
  readonly #certifications = new Map<string, EngineCertification[]>();

  registerDescriptor(value: unknown): `sha256:${string}` {
    const descriptor = engineFamilyDescriptorSchema.parse(value);
    const digest = sha256CanonicalEvidence(descriptor) as `sha256:${string}`;
    const existing = this.#descriptors.get(digest);
    if (existing && sha256CanonicalEvidence(existing) !== digest) {
      throw new EngineFamilyResolutionError("engine_descriptor_digest_collision");
    }
    this.#descriptors.set(digest, descriptor);
    return digest;
  }

  registerCertification(value: unknown): void {
    const certification = engineCertificationSchema.parse(value);
    if (certification.certificationId !== engineCertificationIdentity(certification)) {
      throw new EngineFamilyResolutionError("engine_certification_identity_mismatch");
    }
    if (!this.#descriptors.has(certification.descriptorDigest)) {
      throw new EngineFamilyResolutionError(
        "engine_certification_descriptor_is_unknown",
      );
    }
    const certifications =
      this.#certifications.get(certification.descriptorDigest) ?? [];
    const retained = certifications.filter(
      (candidate) => candidate.certificationId !== certification.certificationId,
    );
    retained.push(certification);
    this.#certifications.set(certification.descriptorDigest, retained);
  }

  resolve(request: EngineResolutionRequest): ResolvedEngineFamily {
    if (!Number.isSafeInteger(request.contextTokens) || request.contextTokens <= 0) {
      throw new EngineFamilyResolutionError("engine_context_tokens_are_invalid");
    }
    if (Number.isNaN(request.now.getTime())) {
      throw new EngineFamilyResolutionError("engine_resolution_time_is_invalid");
    }

    const descriptor = this.#descriptors.get(request.descriptorDigest);
    if (!descriptor) {
      throw new EngineFamilyResolutionError("engine_descriptor_is_unknown");
    }
    if (
      descriptor.model.modelId !== request.modelId
      || descriptor.model.revision !== request.revision
      || descriptor.model.manifestDigest !== request.manifestDigest
      || descriptor.tensorAbi !== request.tensorAbi
    ) {
      throw new EngineFamilyResolutionError("engine_descriptor_model_mismatch");
    }

    const descriptorTarget = findTarget(descriptor.targets, request);
    if (!descriptorTarget) {
      throw new EngineFamilyResolutionError("engine_descriptor_target_mismatch");
    }

    const certification = (this.#certifications.get(request.descriptorDigest) ?? [])
      .filter((candidate) => candidate.status === "certified")
      .filter(
        (candidate) =>
          Date.parse(candidate.validFrom) <= request.now.getTime()
          && Date.parse(candidate.expiresAt) > request.now.getTime(),
      )
      .filter((candidate) => candidate.hardwareClasses.includes(request.hardwareClass))
      .filter((candidate) => candidate.parity.some((evidence) =>
        request.generationMode === "greedy"
          ? evidence.kind === "tokens"
          : evidence.kind === "sampling-distribution"
      ))
      .find((candidate) => findTarget(candidate.targets, request));
    if (!certification) {
      throw new EngineFamilyResolutionError(
        "engine_compatible_certification_is_missing",
      );
    }

    return { descriptor, certification, target: descriptorTarget };
  }
}

function findTarget(
  targets: EngineTarget[],
  request: EngineResolutionRequest,
): EngineTarget | undefined {
  return targets.find(
    (target) =>
      target.platform === request.platform
      && target.arch === request.arch
      && target.backend === request.backend
      && target.runtimeAbi === request.runtimeAbi
      && target.quantizations.includes(request.quantization)
      && target.context.minTokens <= request.contextTokens
      && target.context.maxTokens >= request.contextTokens,
  );
}
