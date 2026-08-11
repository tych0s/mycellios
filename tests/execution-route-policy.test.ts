import { describe, expect, it } from "vitest";
import {
  decideExecutionRoute,
  type ExecutionRouteCandidate,
} from "../src/distribution/execution-route-policy.js";

type Route = { id: string };

function candidate(
  id: string,
  kind: ExecutionRouteCandidate<Route>["kind"],
  score: number,
  overrides: Partial<ExecutionRouteCandidate<Route>> = {},
): ExecutionRouteCandidate<Route> {
  return {
    id,
    kind,
    route: { id },
    score,
    nodeCount: kind === "distributed-pipeline" ? 2 : 1,
    meetsSlo: true,
    eligible: true,
    rejectionReasons: [],
    ...overrides,
  };
}

describe("ExecutionRoutePolicy", () => {
  it("keeps recommendation, selected KV route, fallbacks and reasons separate", () => {
    const decision = decideExecutionRoute({
      candidates: [
        candidate("local", "local-complete", 0.2),
        candidate("affinity", "remote-replica", 0.25),
        candidate("split", "distributed-pipeline", 0.3),
      ],
      affinityCandidateId: "affinity",
      maxAffinityScorePenalty: 0.1,
    });

    expect(decision.recommendation?.candidateId).toBe("local");
    expect(decision.selected).toMatchObject({
      candidateId: "affinity",
      kind: "remote-replica",
      reason: "selected_kv_affinity",
    });
    expect(decision.fallbacks.map((fallback) => fallback.candidateId)).toEqual([
      "local",
      "split",
    ]);
    expect(decision.reasons).toEqual(["selected_kv_affinity"]);
  });

  it("prefers the smallest physical cell when service scores tie", () => {
    const decision = decideExecutionRoute({
      candidates: [
        candidate("split", "distributed-pipeline", 0.2, { nodeCount: 3 }),
        candidate("replica", "remote-replica", 0.2),
        candidate("local", "local-complete", 0.2),
      ],
    });
    expect(decision.selected?.candidateId).toBe("local");
    expect(decision.fallbacks.map((fallback) => fallback.candidateId)).toEqual([
      "replica",
      "split",
    ]);
  });

  it("excludes an SLO-breaking node instead of forcing a larger split", () => {
    const decision = decideExecutionRoute({
      candidates: [
        candidate("small-cell", "distributed-pipeline", 0.4, { nodeCount: 2 }),
        candidate("slow-extra-node", "distributed-pipeline", 0.3, {
          nodeCount: 3,
          meetsSlo: false,
        }),
      ],
    });
    expect(decision.selected?.candidateId).toBe("small-cell");
    expect(decision.evaluations.find((item) => item.candidateId === "slow-extra-node")).toMatchObject({
      eligible: false,
      reasons: ["candidate_slo_exceeded"],
    });
  });

  it("returns unavailable with stable rejection reasons", () => {
    const decision = decideExecutionRoute({
      candidates: [candidate("uncertified", "remote-replica", 0.1, {
        eligible: false,
        rejectionReasons: ["candidate_evidence_missing"],
      })],
    });
    expect(decision.selectedKind).toBe("unavailable");
    expect(decision.selected).toBeNull();
    expect(decision.reasons).toEqual(["candidate_evidence_missing"]);
  });

  it("reports an empty fleet without inventing a route", () => {
    expect(decideExecutionRoute<Route>({ candidates: [] })).toMatchObject({
      selectedKind: "unavailable",
      reasons: ["no_candidate_routes"],
    });
  });
});
