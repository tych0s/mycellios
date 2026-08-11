import type { NetworkExecutionTrace } from "../../src/contracts/types";

export type NetworkEvidenceClass = "physical" | "simulated" | "loopback" | "unverified";

export interface NetworkJobTopology {
  receiptId?: string;
  traceDigest: string;
  receiptTraceDigest: string;
  classification: NetworkEvidenceClass;
  stages: Array<{ stageIndex: number; alias: string; region: string }>;
  boundaries: Array<{
    boundaryIndex: number;
    fromStageIndex: number;
    toStageIndex: number;
    transport: "direct" | "relay" | "local" | "unobserved";
    physicalBoundary: boolean | null;
  }>;
}

export interface VerifiedNetworkRoute {
  fromAlias: string;
  toAlias: string;
  fromRegion: string;
  toRegion: string;
  transport: NetworkJobTopology["boundaries"][number]["transport"];
  physicalBoundary: boolean | null;
  classification: NetworkEvidenceClass;
}

/**
 * Builds only routes sealed by the same trace digest as the execution receipt.
 * Missing stages, duplicate stage indexes and unbound boundaries fail closed.
 */
export function verifiedNetworkRoutes(topology: NetworkJobTopology | null | undefined): VerifiedNetworkRoute[] {
  if (!topology || topology.traceDigest !== topology.receiptTraceDigest) return [];
  if (!/^sha256:[a-f0-9]{64}$/.test(topology.traceDigest)) return [];
  const stages = new Map<number, { alias: string; region: string }>();
  for (const stage of topology.stages) {
    if (!Number.isSafeInteger(stage.stageIndex) || stage.stageIndex < 0 || !/^stage-[0-9]+$/.test(stage.alias)
      || !stage.region || stages.has(stage.stageIndex)) return [];
    stages.set(stage.stageIndex, { alias: stage.alias, region: stage.region });
  }
  const routes: VerifiedNetworkRoute[] = [];
  const indexes = new Set<number>();
  for (const boundary of topology.boundaries) {
    if (!Number.isSafeInteger(boundary.boundaryIndex) || boundary.boundaryIndex < 0 || indexes.has(boundary.boundaryIndex)) return [];
    indexes.add(boundary.boundaryIndex);
    const from = stages.get(boundary.fromStageIndex);
    const to = stages.get(boundary.toStageIndex);
    if (!from || !to || from.alias === to.alias) return [];
    routes.push({ fromAlias: from.alias, toAlias: to.alias, fromRegion: from.region, toRegion: to.region, transport: boundary.transport,
      physicalBoundary: boundary.physicalBoundary, classification: topology.classification });
  }
  return routes;
}

export function topologyEvidenceLabel(topology: NetworkJobTopology | null | undefined): string {
  if (!topology) return "No execution topology was published for this job.";
  if (topology.traceDigest !== topology.receiptTraceDigest) return "Topology rejected: execution receipt digest mismatch.";
  const routes = verifiedNetworkRoutes(topology);
  if (topology.boundaries.length > 0 && routes.length === 0) return "Topology rejected: incomplete or ambiguous stage boundary.";
  if (routes.length === 0) return `Verified ${topology.classification} single-stage execution; no network boundary.`;
  return `${routes.length} verified ${topology.classification} route${routes.length === 1 ? "" : "s"}.`;
}

export interface ObservedTopologyNode { id: string; executionNodeId?: string }
export interface ObservedTopologyLink {
  sourceNodeId: string;
  destinationNodeId: string;
  evidenceClass: "physical" | "declared" | "simulated";
  transport: "direct" | "relay" | "local" | "unobserved";
}

/** Snapshot traces lack the durable receipt binding required by P6-B. */
export function projectObservedTopology(
  _workers: readonly ObservedTopologyNode[],
  _traces: readonly NetworkExecutionTrace[],
): ObservedTopologyLink[] {
  return [];
}
