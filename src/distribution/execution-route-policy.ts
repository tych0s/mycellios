export type ExecutionRouteKind =
  | "local-complete"
  | "remote-replica"
  | "distributed-pipeline"
  | "unavailable";

export type ExecutionRouteReasonCode =
  | "selected_best_service"
  | "selected_kv_affinity"
  | "selected_fallback_after_failure"
  | "candidate_not_ready"
  | "candidate_slo_exceeded"
  | "candidate_evidence_missing"
  | "candidate_capacity_exhausted"
  | "candidate_context_exceeded"
  | "candidate_trust_rejected"
  | "candidate_boundary_trust_rejected"
  | "candidate_boundary_pin_rejected"
  | "candidate_residency_rejected"
  | "candidate_failure_domain_rejected"
  | "candidate_cost_exceeded"
  | "no_candidate_routes";

export interface ExecutionRouteCandidate<T> {
  id: string;
  kind: Exclude<ExecutionRouteKind, "unavailable">;
  route: T;
  /** Lower is better; callers normalize compute, queue, network, cost and SLO. */
  score: number;
  nodeCount: number;
  meetsSlo: boolean;
  eligible: boolean;
  rejectionReasons: ExecutionRouteReasonCode[];
}

export interface ExecutionRouteSelection<T> {
  candidateId: string;
  kind: Exclude<ExecutionRouteKind, "unavailable">;
  route: T;
  score: number;
  nodeCount: number;
  reason: "selected_best_service" | "selected_kv_affinity" | "selected_fallback_after_failure";
}

export interface ExecutionRouteEvaluation {
  candidateId: string;
  kind: Exclude<ExecutionRouteKind, "unavailable">;
  eligible: boolean;
  score: number;
  nodeCount: number;
  reasons: ExecutionRouteReasonCode[];
}

export interface ExecutionRouteDecision<T> {
  recommendation: ExecutionRouteSelection<T> | null;
  selected: ExecutionRouteSelection<T> | null;
  selectedKind: ExecutionRouteKind;
  fallbacks: ExecutionRouteSelection<T>[];
  reasons: ExecutionRouteReasonCode[];
  evaluations: ExecutionRouteEvaluation[];
}

export interface ExecutionRoutePolicyInput<T> {
  candidates: readonly ExecutionRouteCandidate<T>[];
  affinityCandidateId?: string | undefined;
  /** Maximum additive normalized score tolerated to keep an eligible KV route. */
  maxAffinityScorePenalty?: number | undefined;
}

const KIND_ORDER: Record<Exclude<ExecutionRouteKind, "unavailable">, number> = {
  "local-complete": 0,
  "remote-replica": 1,
  "distributed-pipeline": 2,
};

/**
 * Pure, deterministic execution-route policy. It does not read clocks, stores,
 * probes or queues: callers must snapshot and normalize that evidence first.
 */
export function decideExecutionRoute<T>(
  input: ExecutionRoutePolicyInput<T>,
): ExecutionRouteDecision<T> {
  const affinityPenalty = input.maxAffinityScorePenalty ?? 0.1;
  if (!Number.isFinite(affinityPenalty) || affinityPenalty < 0) {
    throw new RangeError("maxAffinityScorePenalty must be a finite non-negative number");
  }
  const ids = new Set<string>();
  const evaluations: ExecutionRouteEvaluation[] = input.candidates.map((candidate) => {
    validateCandidate(candidate, ids);
    const reasons = [...candidate.rejectionReasons];
    if (!candidate.meetsSlo && !reasons.includes("candidate_slo_exceeded")) {
      reasons.push("candidate_slo_exceeded");
    }
    const eligible = candidate.eligible && candidate.meetsSlo && reasons.length === 0;
    if (!eligible && reasons.length === 0) reasons.push("candidate_not_ready");
    return {
      candidateId: candidate.id,
      kind: candidate.kind,
      eligible,
      score: candidate.score,
      nodeCount: candidate.nodeCount,
      reasons,
    };
  });
  const evaluationById = new Map(evaluations.map((item) => [item.candidateId, item]));
  const eligible = input.candidates
    .filter((candidate) => evaluationById.get(candidate.id)?.eligible)
    .sort(compareCandidates);

  if (eligible.length === 0) {
    const reasons = uniqueReasons(evaluations.flatMap((item) => item.reasons));
    if (reasons.length === 0) reasons.push("no_candidate_routes");
    return {
      recommendation: null,
      selected: null,
      selectedKind: "unavailable",
      fallbacks: [],
      reasons,
      evaluations,
    };
  }

  const recommended = eligible[0]!;
  const affinity = input.affinityCandidateId
    ? eligible.find((candidate) => candidate.id === input.affinityCandidateId)
    : undefined;
  const chosen = affinity && affinity.score <= recommended.score + affinityPenalty
    ? affinity
    : recommended;
  const recommendation = selection(recommended, "selected_best_service");
  const selectedReason = chosen === affinity && chosen !== recommended
    ? "selected_kv_affinity"
    : "selected_best_service";
  return {
    recommendation,
    selected: selection(chosen, selectedReason),
    selectedKind: chosen.kind,
    fallbacks: eligible
      .filter((candidate) => candidate !== chosen)
      .map((candidate) => selection(candidate, "selected_best_service")),
    reasons: [selectedReason],
    evaluations,
  };
}

function validateCandidate<T>(candidate: ExecutionRouteCandidate<T>, ids: Set<string>): void {
  if (!candidate.id || ids.has(candidate.id)) {
    throw new Error(candidate.id ? `duplicate execution route candidate: ${candidate.id}` : "execution route candidate id is required");
  }
  ids.add(candidate.id);
  if (!Number.isFinite(candidate.score) || candidate.score < 0) {
    throw new RangeError(`execution route candidate ${candidate.id} has an invalid score`);
  }
  if (!Number.isInteger(candidate.nodeCount) || candidate.nodeCount < 1) {
    throw new RangeError(`execution route candidate ${candidate.id} has an invalid nodeCount`);
  }
}

function compareCandidates<T>(
  left: ExecutionRouteCandidate<T>,
  right: ExecutionRouteCandidate<T>,
): number {
  return left.score - right.score
    || left.nodeCount - right.nodeCount
    || KIND_ORDER[left.kind] - KIND_ORDER[right.kind]
    || left.id.localeCompare(right.id);
}

function selection<T>(
  candidate: ExecutionRouteCandidate<T>,
  reason: ExecutionRouteSelection<T>["reason"],
): ExecutionRouteSelection<T> {
  return {
    candidateId: candidate.id,
    kind: candidate.kind,
    route: candidate.route,
    score: candidate.score,
    nodeCount: candidate.nodeCount,
    reason,
  };
}

function uniqueReasons(reasons: ExecutionRouteReasonCode[]): ExecutionRouteReasonCode[] {
  return [...new Set(reasons)].sort();
}
