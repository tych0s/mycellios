import type { NodeCommand } from "../../src/contracts/node-control";
import type {
  DesktopAccelerationStatus,
  DesktopComponentUpdateStatus,
  DesktopUpdateStatus,
} from "../../src/contracts/control-api";

export type OperationalStateId =
  | "pairing"
  | "downloading"
  | "canarying"
  | "ready"
  | "paused"
  | "draining"
  | "reconnecting"
  | "degraded"
  | "failed"
  | "revoked"
  | "updating"
  | "rollback";

export interface OperationalState {
  id: OperationalStateId;
  label: string;
  summary: string;
  nextAction: string;
  tone: "neutral" | "progress" | "success" | "warning" | "danger";
  busy: boolean;
}

const STATES: Record<OperationalStateId, OperationalState> = {
  pairing: { id: "pairing", label: "Pairing", summary: "Waiting for the one-time enrollment and signed node identity.", nextAction: "Complete pairing on the node", tone: "progress", busy: true },
  downloading: { id: "downloading", label: "Downloading", summary: "Verified runtime bytes are being transferred and checked before use.", nextAction: "Keep the node online", tone: "progress", busy: true },
  canarying: { id: "canarying", label: "Running canary", summary: "The runtime is proving real inference before capacity is published.", nextAction: "Wait for canary evidence", tone: "progress", busy: true },
  ready: { id: "ready", label: "Ready", summary: "The signed runtime is healthy and may accept verified work.", nextAction: "Run inference", tone: "success", busy: false },
  paused: { id: "paused", label: "Paused", summary: "Runtimes remain installed, but this node is accepting no new work.", nextAction: "Resume contribution", tone: "neutral", busy: false },
  draining: { id: "draining", label: "Draining", summary: "New leases are blocked while active work finishes safely.", nextAction: "Wait for active work to finish", tone: "warning", busy: true },
  reconnecting: { id: "reconnecting", label: "Reconnecting", summary: "The node is restoring its authenticated coordinator session.", nextAction: "Check network connectivity", tone: "progress", busy: true },
  degraded: { id: "degraded", label: "Degraded", summary: "The node remains known, but one or more required checks need attention.", nextAction: "Open diagnostics", tone: "warning", busy: false },
  failed: { id: "failed", label: "Failed", summary: "The node could not reach a safe operational state and is not serving work.", nextAction: "Open diagnostics and retry", tone: "danger", busy: false },
  revoked: { id: "revoked", label: "Revoked", summary: "This identity can no longer register, receive leases or execute commands.", nextAction: "Create a new enrollment", tone: "danger", busy: false },
  updating: { id: "updating", label: "Updating", summary: "The node is draining and activating a compatible signed component set.", nextAction: "Keep the node online", tone: "progress", busy: true },
  rollback: { id: "rollback", label: "Rolling back", summary: "The previous verified runtime is being restored before restart.", nextAction: "Keep the node online", tone: "warning", busy: true },
};

export function operationalState(id: OperationalStateId): OperationalState {
  return STATES[id];
}

type CommandState = "queued" | "delivered" | "applied" | "rejected" | "expired";

export interface NodeOperationalInput {
  status: "active" | "revoked";
  connected: boolean;
  observed: null | {
    state: "pairing" | "canary" | "ready" | "reconnecting" | "degraded" | "failed" | "revoked";
    contributionEnabled: boolean;
    draining: boolean;
  };
  latestCommand: null | { type: NodeCommand["type"]; state: CommandState };
}

export function nodeOperationalState(input: NodeOperationalInput): OperationalState {
  if (input.status === "revoked" || input.observed?.state === "revoked") return STATES.revoked;
  const commandPending = input.latestCommand && (input.latestCommand.state === "queued" || input.latestCommand.state === "delivered");
  if (commandPending && input.latestCommand?.type === "rollback") return STATES.rollback;
  if (commandPending && input.latestCommand?.type === "update") return STATES.updating;
  if (!input.observed || input.observed.state === "pairing") return STATES.pairing;
  if (input.observed.state === "canary") return STATES.canarying;
  if (input.observed.state === "failed") return STATES.failed;
  if (input.observed.state === "degraded") return STATES.degraded;
  if (input.observed.state === "reconnecting" || !input.connected) return STATES.reconnecting;
  if (input.observed.draining) return STATES.draining;
  if (!input.observed.contributionEnabled) return STATES.paused;
  return STATES.ready;
}

export interface LocalOperationalInput {
  update: DesktopUpdateStatus;
  acceleration: DesktopAccelerationStatus;
  contribution: "paused" | "connecting" | "connected" | "error";
}

function componentState(update: DesktopUpdateStatus): DesktopComponentUpdateStatus["state"] | null {
  return update.components?.state ?? null;
}

export function localOperationalState(input: LocalOperationalInput): OperationalState {
  const component = componentState(input.update);
  if (input.update.state === "downloading" || component === "downloading" || input.acceleration.preparation.phase === "downloading") return STATES.downloading;
  if (component === "activating" || component === "waiting-idle" || input.update.state === "ready") return STATES.updating;
  if (["verifying-package", "physical-probe", "activating"].includes(input.acceleration.preparation.phase)) return STATES.canarying;
  if (input.update.state === "error" || component === "error" || input.acceleration.state === "error") return STATES.failed;
  if (input.contribution === "error") return STATES.degraded;
  if (input.contribution === "connecting") return STATES.reconnecting;
  if (input.contribution === "paused") return STATES.paused;
  return STATES.ready;
}
