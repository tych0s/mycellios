import { randomUUID } from "node:crypto";
import type {
  FleetContributionCommandResponse,
  FleetContributionCommandResult,
  FleetContributionStatus,
} from "../contracts/fleet-contribution.js";
import type { StoredWorker } from "../storage/store.js";

const CONTRIBUTION_CONTROL_PROTOCOL = "mycellios-contribution-control/1";

interface FleetContributionStore {
  listWorkers(): StoredWorker[];
}

interface FleetContributionHub {
  isConnected(workerId: string): boolean;
  send(workerId: string, type: string, payload: unknown): boolean;
}

interface ContributionAcknowledgement {
  commandId: string;
  enabled: boolean;
  changed: boolean;
  applied: boolean;
  error?: string | undefined;
}

interface PendingAcknowledgement {
  workerId: string;
  enabled: boolean;
  timer: NodeJS.Timeout;
  resolve: (result: FleetContributionCommandResult) => void;
}

export class FleetContributionController {
  private readonly pending = new Map<string, PendingAcknowledgement>();

  constructor(
    private readonly store: FleetContributionStore,
    private readonly hub: FleetContributionHub,
    private readonly acknowledgementTimeoutMs = 3_000,
  ) {}

  status(): FleetContributionStatus {
    const nodes = this.desktopWorkers().map((worker) => {
      const connected = this.hub.isConnected(worker.id);
      const controlReady =
        worker.capabilities.administration?.contributionControl
          === CONTRIBUTION_CONTROL_PROTOCOL;
      const state = !connected
        ? "offline" as const
        : !controlReady
          ? "unsupported" as const
          : worker.status === "online"
            ? "contributing" as const
            : worker.status === "draining"
              ? "paused" as const
              : "unavailable" as const;
      return {
        workerId: worker.id,
        agentVersion: worker.capabilities.agentVersion,
        connected,
        controlReady,
        state,
      };
    });
    return {
      capturedAt: new Date().toISOString(),
      summary: {
        desktopNodes: nodes.length,
        connected: nodes.filter((node) => node.connected).length,
        controlReady: nodes.filter((node) => node.connected && node.controlReady).length,
        contributing: nodes.filter((node) => node.state === "contributing").length,
        paused: nodes.filter((node) => node.state === "paused").length,
        unsupported: nodes.filter((node) => node.state === "unsupported").length,
        offline: nodes.filter((node) => node.state === "offline").length,
      },
      nodes,
    };
  }

  async setAll(enabled: boolean): Promise<FleetContributionCommandResponse> {
    const commandId = randomUUID();
    const issuedAt = Date.now();
    const targets = this.desktopWorkers().filter(
      (worker) =>
        this.hub.isConnected(worker.id)
        && worker.capabilities.administration?.contributionControl
          === CONTRIBUTION_CONTROL_PROTOCOL,
    );
    const results = await Promise.all(
      targets.map((worker) => this.sendCommand(worker.id, commandId, enabled, issuedAt)),
    );
    return {
      commandId,
      enabled,
      issuedAt: new Date(issuedAt).toISOString(),
      results,
      status: this.status(),
    };
  }

  acknowledge(workerId: string, acknowledgement: ContributionAcknowledgement): boolean {
    const key = this.pendingKey(acknowledgement.commandId, workerId);
    const pending = this.pending.get(key);
    if (!pending || pending.enabled !== acknowledgement.enabled) return false;
    clearTimeout(pending.timer);
    this.pending.delete(key);
    pending.resolve({
      workerId,
      state: acknowledgement.applied
        ? acknowledgement.changed ? "applied" : "unchanged"
        : "failed",
      ...(acknowledgement.error ? { error: acknowledgement.error } : {}),
    });
    return true;
  }

  close(): void {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.resolve({
        workerId: pending.workerId,
        state: "failed",
        error: "coordinator_shutting_down",
      });
    }
    this.pending.clear();
  }

  private desktopWorkers(): StoredWorker[] {
    return this.store
      .listWorkers()
      .filter((worker) => worker.identityKind !== "cell");
  }

  private sendCommand(
    workerId: string,
    commandId: string,
    enabled: boolean,
    issuedAt: number,
  ): Promise<FleetContributionCommandResult> {
    return new Promise((resolve) => {
      const key = this.pendingKey(commandId, workerId);
      const timer = setTimeout(() => {
        this.pending.delete(key);
        resolve({ workerId, state: "timeout" });
      }, this.acknowledgementTimeoutMs);
      timer.unref();
      this.pending.set(key, { workerId, enabled, timer, resolve });
      if (!this.hub.send(workerId, "contribution.set", { commandId, enabled, issuedAt })) {
        clearTimeout(timer);
        this.pending.delete(key);
        resolve({ workerId, state: "unreachable" });
      }
    });
  }

  private pendingKey(commandId: string, workerId: string): string {
    return `${commandId}:${workerId}`;
  }
}
