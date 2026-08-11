import type { NodeConfiguration } from "../contracts/node-configuration.js";
import type { NodeCommandApplication } from "./command-executor.js";
import { NodeCommandExecutionError } from "./command-executor.js";
import type { NodeConfigurationStore } from "./config-store.js";
import type { NodeControlStateStore } from "./control-state.js";
import type { NodeComponentLifecycleApi } from "./component-lifecycle.js";
import type { NodeUninstallLifecycleApi } from "./uninstall-helper.js";

export interface NodeCommandAgent {
  setContributionEnabled(enabled: boolean): Promise<boolean>;
  beginRuntimeUpdateDrain(): Promise<() => Promise<void>>;
  waitForIdle(timeoutMs: number): Promise<void>;
}

export class NodeCommandApplicator {
  private drainRelease: (() => Promise<void>) | null = null;
  private restartAfterAck: string | null = null;
  private stopAfterAck: { commandId: string; requestId: string; requestDigest: string } | null = null;

  constructor(
    private readonly agent: NodeCommandAgent,
    private readonly controlStore: NodeControlStateStore,
    private readonly configurationStore: NodeConfigurationStore,
    private readonly configuration: NodeConfiguration,
    private readonly components?: NodeComponentLifecycleApi,
    private readonly uninstall?: NodeUninstallLifecycleApi,
  ) {}

  async restoreDrain(): Promise<void> {
    await this.reconcileDrain(true);
  }

  async reconcileDrain(draining: boolean): Promise<void> {
    if (draining) {
      await this.controlStore.setDraining(true);
      this.drainRelease ??= await this.agent.beginRuntimeUpdateDrain();
      return;
    }
    if (this.drainRelease) {
      await this.drainRelease();
      this.drainRelease = null;
    }
    await this.controlStore.setDraining(false);
  }

  async apply({ command, operationKey }: NodeCommandApplication): Promise<unknown> {
    switch (command.type) {
      case "pause": {
        await this.controlStore.setContributionEnabled(false);
        await this.agent.setContributionEnabled(false);
        return { operationKey, contributionEnabled: false };
      }
      case "resume": {
        await this.reconcileDrain(false);
        await this.controlStore.setContributionEnabled(true);
        await this.agent.setContributionEnabled(true);
        return { operationKey, contributionEnabled: true, draining: false };
      }
      case "drain": {
        await this.reconcileDrain(true);
        await this.agent.waitForIdle(command.payload.deadlineMs);
        return { operationKey, draining: true, activeWork: 0 };
      }
      case "set-limits": {
        const { version: _version, ...limits } = command.payload;
        const next = await this.configurationStore.save({
          ...this.configuration,
          revision: this.configuration.revision + 1,
          limits,
        });
        this.restartAfterAck = command.id;
        return { operationKey, configRevision: next.revision, restartRequired: true };
      }
      case "set-policy": {
        const next = await this.configurationStore.save({
          ...this.configuration,
          revision: this.configuration.revision + 1,
          policy: command.payload.policy,
        });
        this.restartAfterAck = command.id;
        return { operationKey, configRevision: next.revision, restartRequired: true, policy: next.policy };
      }
      case "update": {
        if (!this.components) throw new NodeCommandExecutionError("node_update_not_configured", "Component update trust is not configured.");
        await this.controlStore.setDraining(true);
        this.drainRelease ??= await this.agent.beginRuntimeUpdateDrain();
        await this.agent.waitForIdle(60_000);
        const channelChanged = command.payload.channel !== this.configuration.updateChannel;
        let configRevision = this.configuration.revision;
        if (channelChanged) {
          configRevision = (await this.configurationStore.save({
            ...this.configuration,
            revision: this.configuration.revision + 1,
            updateChannel: command.payload.channel,
          })).revision;
        }
        let result;
        try {
          result = await this.components.update(
            command.payload.channel,
            command.payload.manifestId as `sha256:${string}` | undefined,
          );
        } catch (error) {
          if (channelChanged) await this.configurationStore.rollback().catch(() => undefined);
          throw error;
        }
        if (result.state === "waiting-idle") {
          if (channelChanged) await this.configurationStore.rollback().catch(() => undefined);
          throw new NodeCommandExecutionError("node_update_waiting_idle", "The node did not become idle before activation.");
        }
        if (result.state === "applied" || channelChanged) this.restartAfterAck = command.id;
        return { operationKey, ...result, configRevision, restartRequired: result.state === "applied" || channelChanged };
      }
      case "rollback": {
        if (!this.components) throw new NodeCommandExecutionError("node_rollback_not_configured", "Component update trust is not configured.");
        await this.controlStore.setDraining(true);
        this.drainRelease ??= await this.agent.beginRuntimeUpdateDrain();
        await this.agent.waitForIdle(60_000);
        const result = await this.components.rollback(command.payload.componentIds);
        this.restartAfterAck = command.id;
        return { operationKey, ...result, restartRequired: true };
      }
      case "revoke":
        throw new NodeCommandExecutionError("node_revocation_requires_reauth", "Identity revocation must be confirmed by the coordinator before local shutdown.");
      case "uninstall": {
        if (!this.uninstall) throw new NodeCommandExecutionError("node_uninstall_requires_service_helper", "Uninstall requires the privileged platform service helper.");
        await this.reconcileDrain(true);
        await this.agent.waitForIdle(60_000);
        await this.controlStore.setContributionEnabled(false);
        await this.agent.setContributionEnabled(false);
        const receipt = await this.uninstall.schedule({
          nodeId: command.nodeId,
          generation: command.generation,
          commandId: command.id,
          retain: command.payload.retain,
        });
        this.stopAfterAck = { commandId: command.id, requestId: receipt.requestId, requestDigest: receipt.requestDigest };
        return { operationKey, ...receipt, stopAfterAck: true };
      }
    }
  }

  takeRestartAfterAck(commandId: string): boolean {
    if (this.restartAfterAck !== commandId) return false;
    this.restartAfterAck = null;
    return true;
  }

  async armStopAfterAck(commandId: string): Promise<boolean> {
    if (this.stopAfterAck?.commandId !== commandId || !this.uninstall) return false;
    await this.uninstall.arm(this.stopAfterAck.requestId, this.stopAfterAck.requestDigest);
    this.stopAfterAck = null;
    return true;
  }
}
