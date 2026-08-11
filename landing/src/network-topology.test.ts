import { describe, expect, it } from "vitest";
import { topologyEvidenceLabel, verifiedNetworkRoutes, type NetworkJobTopology } from "./network-topology";

const digest = `sha256:${"a".repeat(64)}`;
const topology: NetworkJobTopology = {
  traceDigest: digest, receiptTraceDigest: digest, classification: "physical",
  stages: [{ stageIndex: 0, alias: "stage-0", region: "Madrid" }, { stageIndex: 1, alias: "stage-1", region: "Nuremberg" }],
  boundaries: [{ boundaryIndex: 0, fromStageIndex: 0, toStageIndex: 1,
    transport: "direct", physicalBoundary: true }],
};

describe("receipt-bound network topology", () => {
  it("materializes only the stages and boundary sealed by the receipt", () => {
    expect(verifiedNetworkRoutes(topology)).toEqual([{ fromAlias: "stage-0", toAlias: "stage-1", fromRegion: "Madrid", toRegion: "Nuremberg",
      transport: "direct", physicalBoundary: true, classification: "physical" }]);
    expect(topologyEvidenceLabel(topology)).toBe("1 verified physical route.");
  });

  it("rejects digest mismatch and ambiguous topology instead of drawing a route", () => {
    expect(verifiedNetworkRoutes({ ...topology, receiptTraceDigest: `sha256:${"b".repeat(64)}` })).toEqual([]);
    expect(topologyEvidenceLabel({ ...topology, receiptTraceDigest: `sha256:${"b".repeat(64)}` })).toContain("digest mismatch");
    expect(verifiedNetworkRoutes({ ...topology, stages: [...topology.stages, topology.stages[0]!] })).toEqual([]);
  });

  it("labels a sealed single-stage execution without inventing a network edge", () => {
    const local = { ...topology, classification: "loopback" as const, stages: [topology.stages[0]!], boundaries: [] };
    expect(verifiedNetworkRoutes(local)).toEqual([]);
    expect(topologyEvidenceLabel(local)).toContain("single-stage");
  });
});
