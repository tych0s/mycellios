/** Engine-visible evidence for one runtime transport stream. */
export interface RuntimeTransportSnapshot {
  streamId: string;
  sourceNodeId: string | null;
  destinationNodeId: string;
  targetPort: number;
  mode: "direct" | "relay";
  state: "negotiating" | "active" | "suspended" | "closed";
  bytesSourceToDestination: number;
  bytesDestinationToSource: number;
  createdAt: number;
  connectedAt: number | null;
  endedAt: number | null;
  connectRttMs: number | null;
}

export interface RuntimeProxyHandle {
  host: "127.0.0.1";
  port: number;
  close(): Promise<void>;
}
