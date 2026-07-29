export type FleetContributionNodeState =
  | "contributing"
  | "paused"
  | "unavailable"
  | "unsupported"
  | "offline";

export interface FleetContributionNode {
  workerId: string;
  agentVersion: string;
  connected: boolean;
  controlReady: boolean;
  state: FleetContributionNodeState;
}

export interface FleetContributionStatus {
  capturedAt: string;
  summary: {
    desktopNodes: number;
    connected: number;
    controlReady: number;
    contributing: number;
    paused: number;
    unsupported: number;
    offline: number;
  };
  nodes: FleetContributionNode[];
}

export type FleetContributionCommandResultState =
  | "applied"
  | "unchanged"
  | "failed"
  | "timeout"
  | "unreachable";

export interface FleetContributionCommandResult {
  workerId: string;
  state: FleetContributionCommandResultState;
  error?: string;
}

export interface FleetContributionCommandResponse {
  commandId: string;
  enabled: boolean;
  issuedAt: string;
  results: FleetContributionCommandResult[];
  status: FleetContributionStatus;
}
