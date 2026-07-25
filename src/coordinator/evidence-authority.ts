import type { WorkerCapabilities } from "../contracts/types.js";

/**
 * Worker registration and heartbeat documents are claims, not authority.
 * Remove every field that could affect measured routing before persistence.
 */
export function stripWorkerDeclaredEvidence(
  input: WorkerCapabilities,
): WorkerCapabilities {
  const capabilities = structuredClone(input);
  capabilities.deployments = capabilities.deployments.map((deployment) => {
    if (deployment.adapter !== "mycellios-pipeline") return deployment;
    const {
      canaryEvidence: _canaryEvidence,
      ...claim
    } = deployment;
    return {
      ...claim,
      verificationState: "pending",
      throughputSource: "default",
      tokensPerSecond: 1,
      ttftMs: 60_000,
    };
  });
  if (capabilities.distributedExecutor) {
    const {
      performanceEvidence: _performanceEvidence,
      ...executor
    } = capabilities.distributedExecutor;
    capabilities.distributedExecutor = executor;
  }
  return capabilities;
}

/**
 * A normal heartbeat cannot overwrite coordinator observations, but an
 * artifact, activation, node, device or authenticated-session change revokes
 * them immediately.
 */
export function mergeCurrentSessionEvidence(
  workerId: string,
  sessionId: string,
  incoming: WorkerCapabilities,
  current: WorkerCapabilities | null,
): WorkerCapabilities {
  const sanitized = stripWorkerDeclaredEvidence(incoming);
  if (!current) return sanitized;
  sanitized.deployments = sanitized.deployments.map((claim) => {
    if (claim.adapter !== "mycellios-pipeline") return claim;
    const verified = current.deployments.find(
      (candidate) =>
        candidate.deploymentId === claim.deploymentId
        && candidate.model === claim.model
        && candidate.modelDigest === claim.modelDigest
        && candidate.activationId === claim.activationId
        && candidate.verificationState === "verified"
        && candidate.canaryEvidence?.workerId === workerId
        && candidate.canaryEvidence.sessionId === sessionId,
    );
    return verified
      ? {
          ...claim,
          verificationState: "verified",
          throughputSource: "measured",
          tokensPerSecond: verified.tokensPerSecond,
          ttftMs: verified.ttftMs,
          canaryEvidence: structuredClone(verified.canaryEvidence!),
        }
      : claim;
  });

  const incomingExecutor = sanitized.distributedExecutor;
  const currentExecutor = current.distributedExecutor;
  const evidence = currentExecutor?.performanceEvidence;
  if (
    incomingExecutor
    && currentExecutor
    && evidence
    && incomingExecutor.nodeId === currentExecutor.nodeId
    && evidence.workerId === workerId
    && evidence.sessionId === sessionId
    && evidence.nodeId === incomingExecutor.nodeId
    && runtimeIdentityMatches(incomingExecutor, evidence.profile)
  ) {
    incomingExecutor.performanceEvidence = structuredClone(evidence);
  }
  return sanitized;
}

function runtimeIdentityMatches(
  executor: NonNullable<WorkerCapabilities["distributedExecutor"]>,
  profile: NonNullable<
    NonNullable<WorkerCapabilities["distributedExecutor"]>["performanceEvidence"]
  >["profile"],
): boolean {
  if (profile.backend === "cpu") {
    return executor.cpuEligible === true && executor.computeMode !== "gpu-only";
  }
  return executor.computeMode !== "cpu-only"
    && executor.acceleration?.state === "gpu-ready"
    && executor.acceleration.backend === profile.backend
    && normalize(executor.acceleration.deviceName ?? "") === normalize(profile.deviceName);
}

function normalize(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}
