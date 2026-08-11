import {
  engineRuntimeProfileSchema,
  type EngineRuntimeProfile,
} from "../contracts/engine-runtime-profile.js";
import type { MeshStore } from "../storage/store.js";

/**
 * Publishes only profiles whose two physical observations already belong to
 * this coordinator's current authenticated worker session. Worker claims can
 * never call this path successfully by merely embedding a sealed document.
 */
export function publishCoordinatorEngineRuntimeProfile(
  store: MeshStore,
  workerId: string,
  value: unknown,
): EngineRuntimeProfile {
  const profile = engineRuntimeProfileSchema.parse(value);
  const worker = store.getWorker(workerId);
  const executor = worker?.capabilities.distributedExecutor;
  if (!worker || !executor || profile.workerId !== workerId || profile.nodeId !== executor.nodeId) {
    throw new Error("engine_runtime_profile_worker_binding_is_invalid");
  }
  const build = worker.capabilities.buildIdentity;
  const physical = executor.physicalIdentity;
  const performance = executor.performanceEvidence;
  const activationPlan = store.listEngineRuntimeActivationPlans().find(
    (candidate) => candidate.workerId === workerId
      && candidate.modelId === profile.modelId
      && candidate.evidence.canaryEvidenceId === profile.evidence.deploymentCanaryEvidenceId
      && requestMatchesProfile(candidate.request, profile),
  );
  if (
    !build
    || build.sourceId !== profile.sourceId
    || !physical
    || physical.hostFingerprintSha256 !== profile.hardwareFingerprintSha256
    || !performance
    || performance.evidenceId !== profile.evidence.runtimePerformanceEvidenceId
    || performance.sessionId !== profile.sessionId
    || performance.workerId !== profile.workerId
    || performance.nodeId !== profile.nodeId
    || performance.profile.backend !== profile.backend
    || !activationPlan
  ) {
    throw new Error("engine_runtime_profile_evidence_binding_is_invalid");
  }

  const capabilities = structuredClone(worker.capabilities);
  const current = capabilities.distributedExecutor!.engineProfiles ?? [];
  capabilities.distributedExecutor!.engineProfiles = [
    ...current.filter((candidate) => !sameProfileSlot(candidate, profile)),
    profile,
  ].sort((left, right) => left.profileId.localeCompare(right.profileId)).slice(-128);
  store.updateWorkerHeartbeat(worker.id, capabilities, worker.status);
  return profile;
}

function requestMatchesProfile(
  request: import("./worker-hub.js").EngineRuntimeChallengeRequest,
  profile: EngineRuntimeProfile,
): boolean {
  return request.descriptorDigest === profile.descriptorDigest
    && request.certificationId === profile.certificationId
    && request.artifactManifestDigest === profile.artifactManifestDigest
    && request.modelId === profile.modelId
    && request.modelRevision === profile.modelRevision
    && request.backend === profile.backend
    && request.runtimeAbi === profile.runtimeAbi
    && request.quantization === profile.quantization
    && profile.capacity.contextTokens >= request.contextTokens
    && profile.capacity.maxKvTokens >= request.contextTokens
    && profile.capacity.maxLayerCount >= request.expectedLayerEnd - request.expectedLayerStart
    && profile.capacity.kvBytesPerToken === request.expectedKvBytesPerToken
    && request.requiredRoles.every((role) => profile.features.roles.includes(role));
}

function sameProfileSlot(left: EngineRuntimeProfile, right: EngineRuntimeProfile): boolean {
  return left.descriptorDigest === right.descriptorDigest
    && left.certificationId === right.certificationId
    && left.artifactManifestDigest === right.artifactManifestDigest
    && left.backend === right.backend
    && left.runtimeAbi === right.runtimeAbi
    && left.quantization === right.quantization
    && left.capacity.contextTokens === right.capacity.contextTokens;
}
