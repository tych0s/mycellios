import { createHash } from "node:crypto";

export interface BenchmarkActivationParticipant {
  workerId: string;
  agentVersion: string;
  deploymentId: string;
  modelDigest: string;
  nodeIds: string[];
  stageRanges: Array<{
    nodeId: string;
    stageIndex: number;
    layerStart: number;
    layerEnd: number;
  }>;
}

export interface BenchmarkActivation {
  activationId: string;
  modelId: string;
  modelDigest: string;
  topologyDigest: string;
  participants: BenchmarkActivationParticipant[];
}

interface TrackedActivation {
  activation: BenchmarkActivation;
  consecutiveObservations: number;
  lastSeenAt: number;
  emitted: boolean;
}

/**
 * Emits a model activation only after it has remained identical for several
 * coordinator observations. A new generation/topology has a different sealed
 * activation ID and therefore triggers its own campaign even when modelId did
 * not change.
 */
export class StableBenchmarkActivationTracker {
  private readonly tracked = new Map<string, TrackedActivation>();

  constructor(
    private readonly requiredConsecutiveObservations = 3,
    private readonly maximumObservationGapMs = 8_000,
  ) {
    if (!Number.isInteger(requiredConsecutiveObservations) || requiredConsecutiveObservations < 1) {
      throw new Error("benchmark_activation_observations_must_be_positive");
    }
    if (!Number.isFinite(maximumObservationGapMs) || maximumObservationGapMs <= 0) {
      throw new Error("benchmark_activation_observation_gap_must_be_positive");
    }
  }

  observe(
    activations: readonly BenchmarkActivation[],
    now = Date.now(),
  ): BenchmarkActivation[] {
    const currentIds = new Set(activations.map((activation) => activation.activationId));
    for (const activationId of this.tracked.keys()) {
      if (!currentIds.has(activationId)) this.tracked.delete(activationId);
    }

    const ready: BenchmarkActivation[] = [];
    for (const activation of activations) {
      validateBenchmarkActivation(activation);
      const previous = this.tracked.get(activation.activationId);
      const consecutiveObservations = previous
        && now >= previous.lastSeenAt
        && now - previous.lastSeenAt <= this.maximumObservationGapMs
          ? previous.consecutiveObservations + 1
          : 1;
      const next: TrackedActivation = {
        activation,
        consecutiveObservations,
        lastSeenAt: now,
        emitted: previous?.emitted ?? false,
      };
      if (
        !next.emitted
        && next.consecutiveObservations >= this.requiredConsecutiveObservations
      ) {
        next.emitted = true;
        ready.push(activation);
      }
      this.tracked.set(activation.activationId, next);
    }
    return ready;
  }
}

export function sealBenchmarkActivation(
  modelId: string,
  participants: readonly BenchmarkActivationParticipant[],
): BenchmarkActivation {
  const normalizedModelId = requiredText(modelId, "benchmark_activation_model_id_is_invalid");
  if (participants.length === 0) throw new Error("benchmark_activation_has_no_participants");
  const normalizedParticipants = participants
    .map(normalizeParticipant)
    .sort((left, right) =>
      left.workerId.localeCompare(right.workerId)
      || left.deploymentId.localeCompare(right.deploymentId)
    );
  const modelDigests = [...new Set(normalizedParticipants.map((item) => item.modelDigest))];
  if (modelDigests.length !== 1) {
    throw new Error("benchmark_activation_model_digest_is_inconsistent");
  }
  const topologyDocument = normalizedParticipants.map((participant) => ({
    workerId: participant.workerId,
    deploymentId: participant.deploymentId,
    stageRanges: participant.stageRanges,
  }));
  const topologyDigest = digest(topologyDocument);
  const activationId = digest({
    modelId: normalizedModelId,
    modelDigest: modelDigests[0],
    participants: normalizedParticipants,
    topologyDigest,
  });
  return {
    activationId,
    modelId: normalizedModelId,
    modelDigest: modelDigests[0]!,
    topologyDigest,
    participants: normalizedParticipants,
  };
}

function normalizeParticipant(
  participant: BenchmarkActivationParticipant,
): BenchmarkActivationParticipant {
  const stageRanges = participant.stageRanges
    .map((stage) => {
      const nodeId = requiredText(stage.nodeId, "benchmark_activation_stage_node_is_invalid");
      if (
        !Number.isInteger(stage.stageIndex)
        || stage.stageIndex < 0
        || !Number.isInteger(stage.layerStart)
        || stage.layerStart < 0
        || !Number.isInteger(stage.layerEnd)
        || stage.layerEnd <= stage.layerStart
      ) {
        throw new Error("benchmark_activation_stage_range_is_invalid");
      }
      return { ...stage, nodeId };
    })
    .sort((left, right) =>
      left.stageIndex - right.stageIndex
      || left.nodeId.localeCompare(right.nodeId)
      || left.layerStart - right.layerStart
    );
  const nodeIds = [...new Set([
    ...participant.nodeIds.map((nodeId) =>
      requiredText(nodeId, "benchmark_activation_node_id_is_invalid")
    ),
    ...stageRanges.map((stage) => stage.nodeId),
  ])].sort();
  if (nodeIds.length === 0) throw new Error("benchmark_activation_participant_has_no_nodes");
  return {
    workerId: requiredText(
      participant.workerId,
      "benchmark_activation_worker_id_is_invalid",
    ),
    agentVersion: requiredText(
      participant.agentVersion,
      "benchmark_activation_agent_version_is_invalid",
    ),
    deploymentId: requiredText(
      participant.deploymentId,
      "benchmark_activation_deployment_id_is_invalid",
    ),
    modelDigest: requiredText(
      participant.modelDigest,
      "benchmark_activation_model_digest_is_invalid",
    ),
    nodeIds,
    stageRanges,
  };
}

function validateBenchmarkActivation(activation: BenchmarkActivation): void {
  const sealed = sealBenchmarkActivation(activation.modelId, activation.participants);
  if (
    activation.activationId !== sealed.activationId
    || activation.modelDigest !== sealed.modelDigest
    || activation.topologyDigest !== sealed.topologyDigest
  ) {
    throw new Error("benchmark_activation_seal_is_invalid");
  }
}

function digest(value: unknown): string {
  return `sha256:${createHash("sha256").update(JSON.stringify(value)).digest("hex")}`;
}

function requiredText(value: string, code: string): string {
  const normalized = value.trim();
  if (!normalized) throw new Error(code);
  return normalized;
}
