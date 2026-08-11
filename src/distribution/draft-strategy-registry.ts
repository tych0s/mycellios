import type { KeyLike } from "node:crypto";

import {
  draftStrategyDescriptorIdentity,
  draftStrategyDescriptorSchema,
  verifyDraftStrategyCertification,
  type DraftStrategyCertification,
  type DraftStrategyDescriptor,
} from "../contracts/engine-family.js";

export interface DraftStrategyResolutionRequest {
  descriptorDigest: `sha256:${string}`;
  targetDescriptorDigest: `sha256:${string}`;
  tokenizerDigest: `sha256:${string}`;
  vocabularyDigest: `sha256:${string}`;
  backend: DraftStrategyDescriptor["resource"]["allowedBackends"][number];
  availableRamBytes: number;
  availableVramBytes: number;
  requestedDraftTokens: number;
  requestedInflightWaves: number;
  now: Date;
}

export interface ResolvedDraftStrategy {
  descriptor: DraftStrategyDescriptor;
  certification: DraftStrategyCertification;
}

const verifiedResolutions = new WeakSet<object>();

export function assertVerifiedDraftStrategyResolution(
  value: unknown,
): asserts value is ResolvedDraftStrategy {
  if (typeof value !== "object" || value === null || !verifiedResolutions.has(value)) {
    throw new DraftStrategyResolutionError(
      "draft_strategy_resolution_was_not_issued_by_registry",
    );
  }
}

export class DraftStrategyResolutionError extends Error {
  constructor(readonly code: string) {
    super(code);
    this.name = "DraftStrategyResolutionError";
  }
}

/**
 * Coordinator-side authority for native draft components. Descriptors are
 * inert until a pinned publisher certifies their exact digest. Runtime code
 * receives only the result of this fail-closed resolution.
 */
export class NativeDraftStrategyRegistry {
  readonly #publisherKeys: ReadonlyMap<string, KeyLike>;
  readonly #descriptors = new Map<string, DraftStrategyDescriptor>();
  readonly #certifications = new Map<string, DraftStrategyCertification[]>();

  constructor(publisherKeys: ReadonlyMap<string, KeyLike>) {
    this.#publisherKeys = new Map(publisherKeys);
  }

  registerDescriptor(value: unknown): `sha256:${string}` {
    const descriptor = draftStrategyDescriptorSchema.parse(value);
    const digest = draftStrategyDescriptorIdentity(descriptor);
    this.#descriptors.set(digest, descriptor);
    return digest;
  }

  registerCertification(value: unknown): void {
    let certification: DraftStrategyCertification;
    try {
      certification = verifyDraftStrategyCertification(
        value,
        this.#publisherKeys,
      );
    } catch (error) {
      throw new DraftStrategyResolutionError(
        error instanceof Error ? error.message : "draft_strategy_certification_is_invalid",
      );
    }
    if (!this.#descriptors.has(certification.descriptorDigest)) {
      throw new DraftStrategyResolutionError(
        "draft_strategy_certification_descriptor_is_unknown",
      );
    }
    const current = this.#certifications.get(certification.descriptorDigest) ?? [];
    this.#certifications.set(
      certification.descriptorDigest,
      [
        ...current.filter(
          (candidate) => candidate.certificationId !== certification.certificationId,
        ),
        certification,
      ],
    );
  }

  resolve(request: DraftStrategyResolutionRequest): ResolvedDraftStrategy {
    validateRequest(request);
    const descriptor = this.#descriptors.get(request.descriptorDigest);
    if (!descriptor) {
      throw new DraftStrategyResolutionError("draft_strategy_descriptor_is_unknown");
    }
    if (
      descriptor.targetDescriptorDigest !== request.targetDescriptorDigest
      || descriptor.tokenizerDigest !== request.tokenizerDigest
      || descriptor.vocabularyDigest !== request.vocabularyDigest
    ) {
      throw new DraftStrategyResolutionError("draft_strategy_target_is_incompatible");
    }
    if (!descriptor.resource.allowedBackends.includes(request.backend)) {
      throw new DraftStrategyResolutionError("draft_strategy_backend_is_incompatible");
    }
    if (
      request.availableRamBytes < descriptor.resource.minimumRamBytes
      || request.availableVramBytes < descriptor.resource.minimumVramBytes
    ) {
      throw new DraftStrategyResolutionError("draft_strategy_resources_are_insufficient");
    }
    if (
      request.requestedDraftTokens < descriptor.limits.minDraftTokens
      || request.requestedDraftTokens > descriptor.limits.maxDraftTokens
      || request.requestedInflightWaves > descriptor.limits.maxInflightWaves
    ) {
      throw new DraftStrategyResolutionError("draft_strategy_limits_are_exceeded");
    }

    const now = request.now.getTime();
    const active = (this.#certifications.get(request.descriptorDigest) ?? []).filter(
      (candidate) =>
        Date.parse(candidate.validFrom) <= now
        && Date.parse(candidate.expiresAt) > now,
    );
    if (active.some((candidate) => candidate.status === "revoked")) {
      throw new DraftStrategyResolutionError("draft_strategy_is_revoked");
    }
    const certification = active
      .filter((candidate) => candidate.status === "certified")
      .sort((left, right) => Date.parse(right.validFrom) - Date.parse(left.validFrom))[0];
    if (!certification) {
      throw new DraftStrategyResolutionError(
        "draft_strategy_compatible_certification_is_missing",
      );
    }
    const resolution: ResolvedDraftStrategy = { descriptor, certification };
    verifiedResolutions.add(resolution);
    return resolution;
  }
}

function validateRequest(request: DraftStrategyResolutionRequest): void {
  if (Number.isNaN(request.now.getTime())) {
    throw new DraftStrategyResolutionError("draft_strategy_resolution_time_is_invalid");
  }
  for (const value of [
    request.availableRamBytes,
    request.availableVramBytes,
    request.requestedDraftTokens,
    request.requestedInflightWaves,
  ]) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new DraftStrategyResolutionError("draft_strategy_request_is_invalid");
    }
  }
  if (request.requestedDraftTokens === 0 || request.requestedInflightWaves === 0) {
    throw new DraftStrategyResolutionError("draft_strategy_request_is_invalid");
  }
}
