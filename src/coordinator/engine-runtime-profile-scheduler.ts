import {
  requireEligibleEngineRuntimeProfile,
  type EngineRuntimeProfile,
} from "../contracts/engine-runtime-profile.js";
import type { EngineRuntimeChallengeRequest } from "../contracts/evidence-challenge.js";
import type { EngineRuntimeActivationPlan } from "../contracts/engine-runtime-activation.js";
import type { DeploymentControlPlane } from "./deployment-control-plane.js";
import type { StoredWorker } from "../storage/store.js";

export const ENGINE_RUNTIME_PROFILE_RENEWAL_LEAD_MS = 60 * 60_000;

export interface CertifiedEngineRuntimeChallengePlan {
  workerId: string;
  request: EngineRuntimeChallengeRequest;
}

/**
 * Decides whether the coordinator must issue an initial or renewal challenge.
 * The plan is deliberately an authority-side input: worker heartbeat metadata
 * is used only for the already sealed profiles, never to choose descriptor,
 * certification, artifact, model shape, or reference costs.
 */
export function engineRuntimeProfileNeedsChallenge(
  profiles: readonly EngineRuntimeProfile[],
  request: EngineRuntimeChallengeRequest,
  now = Date.now(),
  renewalLeadMs = ENGINE_RUNTIME_PROFILE_RENEWAL_LEAD_MS,
): boolean {
  if (!Number.isFinite(now) || !Number.isFinite(renewalLeadMs) || renewalLeadMs < 0) {
    throw new Error("engine_runtime_profile_renewal_policy_is_invalid");
  }
  const layerCount = request.expectedLayerEnd - request.expectedLayerStart;
  const matching = profiles.filter((profile) => {
    if (
      profile.modelId !== request.modelId
      || profile.modelRevision !== request.modelRevision
      || profile.capacity.maxLayerCount < layerCount
      || profile.capacity.kvBytesPerToken !== request.expectedKvBytesPerToken
      || !request.requiredRoles.every((role) => profile.features.roles.includes(role))
    ) return false;
    try {
      requireEligibleEngineRuntimeProfile(profile, {
        nowMs: now,
        descriptorDigest: request.descriptorDigest,
        certificationId: request.certificationId,
        artifactManifestDigest: request.artifactManifestDigest,
        backend: request.backend,
        runtimeAbi: request.runtimeAbi,
        quantization: request.quantization,
        contextTokens: request.contextTokens,
      });
      return true;
    } catch {
      // A sealed profile can still be unusable because it is stale, noisy,
      // future-dated, or no longer matches the certified launch authority.
      // Such evidence must trigger renewal instead of suppressing probes until
      // its nominal expiresAt.
      return false;
    }
  });
  return !matching.some((profile) => Date.parse(profile.expiresAt) > now + renewalLeadMs);
}

/**
 * A bootstrap route is publishable only when every stage has a fresh profile
 * sealed against the exact activation request. This deliberately reuses the
 * renewal matcher so serving and renewal cannot drift into separate policies.
 */
export function engineRuntimeActivationPlansReady(
  plans: readonly EngineRuntimeActivationPlan[],
  workers: readonly StoredWorker[],
  now = Date.now(),
): boolean {
  const profilesByWorker = new Map(workers.map((worker) => [
    worker.id,
    worker.capabilities.distributedExecutor?.engineProfiles ?? [],
  ]));
  return plans.length > 0 && plans.every((plan) => !engineRuntimeProfileNeedsChallenge(
    (profilesByWorker.get(plan.workerId) ?? []).filter((profile) =>
      profile.workerId === plan.workerId
      && profile.evidence.deploymentCanaryEvidenceId === plan.evidence.canaryEvidenceId
    ),
    plan.request,
    now,
    0,
  ));
}

export function activeEngineRuntimeActivationPlans(
  plans: readonly EngineRuntimeActivationPlan[],
  controller: Pick<DeploymentControlPlane, "getState" | "listReservations">,
): EngineRuntimeActivationPlan[] {
  return plans.filter((plan) => {
    const state = controller.getState(plan.modelId);
    const reservation = controller.listReservations(plan.modelId)
      .find((candidate) => candidate.id === plan.routeReservationId);
    return state?.desiredState === "active"
      && state.observedState === "active"
      && reservation?.status === "committed";
  });
}
