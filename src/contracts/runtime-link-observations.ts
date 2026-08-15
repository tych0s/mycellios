export interface RuntimeLinkObservation {
  fromNodeId: string;
  toNodeId: string;
  measuredAt: number;
  validUntil: number;
  rttP50Ms: number;
  rttP95Ms: number;
  jitterP95Ms: number;
  goodputMbpsP50: number;
  successfulSamples: number;
  failedSamples: number;
  availability: number;
  confidence: number;
  transportMode?: "direct" | "relay";
}

export interface RuntimeLinkSample {
  fromNodeId: string;
  toNodeId: string;
  measuredAt: number;
  rttMs: number | null;
  goodputMbps: number | null;
  transportMode: "direct" | "relay";
}

export function selectPreferredRuntimeLinkObservation(
  observations: readonly RuntimeLinkObservation[],
  minimumDirectAvailability = 0.9,
): RuntimeLinkObservation | undefined {
  if (!Number.isFinite(minimumDirectAvailability) || minimumDirectAvailability <= 0 || minimumDirectAvailability > 1) {
    throw new Error("runtime_link_direct_availability_policy_is_invalid");
  }
  const direct = observations.find((observation) => observation.transportMode === "direct");
  const relay = observations.find((observation) => observation.transportMode === "relay" || observation.transportMode === undefined);
  return direct && direct.availability >= minimumDirectAvailability ? direct : relay ?? direct;
}
