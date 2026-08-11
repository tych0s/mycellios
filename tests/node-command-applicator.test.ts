import { randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { nodeCommandSchema, type NodeCommand } from "../src/contracts/node-control.js";
import { NodeCommandApplicator, type NodeCommandAgent } from "../src/node/command-applicator.js";
import { NodeConfigurationStore } from "../src/node/config-store.js";
import { NodeControlStateStore } from "../src/node/control-state.js";
import type { NodeComponentLifecycleApi } from "../src/node/component-lifecycle.js";
import type { NodeUninstallLifecycleApi } from "../src/node/uninstall-helper.js";
import type { NodeUninstallReceipt } from "../src/contracts/node-uninstall.js";

const cleanup: string[] = [];
afterEach(async () => Promise.all(cleanup.splice(0).map((path) => rm(path, { recursive: true, force: true }))));

describe("NodeCommandApplicator", () => {
  it("persists pause, drain and resume while keeping drain idempotent", async () => {
    const fixture = await setup();
    await fixture.applicator.apply(application(command("pause", { version: 1 })));
    expect(await fixture.control.load()).toMatchObject({ contributionEnabled: false, draining: false });
    expect(fixture.agent.setContributionEnabled).toHaveBeenLastCalledWith(false);

    const drain = command("drain", { version: 1, deadlineMs: 5_000 });
    await fixture.applicator.apply(application(drain));
    await fixture.applicator.apply(application(drain));
    expect(fixture.agent.beginRuntimeUpdateDrain).toHaveBeenCalledTimes(1);
    expect(fixture.agent.waitForIdle).toHaveBeenCalledTimes(2);
    expect((await fixture.control.load()).draining).toBe(true);

    await fixture.applicator.apply(application(command("resume", { version: 1 })));
    expect(fixture.release).toHaveBeenCalledTimes(1);
    expect(await fixture.control.load()).toMatchObject({ contributionEnabled: true, draining: false });
    expect(fixture.agent.setContributionEnabled).toHaveBeenLastCalledWith(true);
  });

  it("persists desired limits atomically and reports the required restart", async () => {
    const fixture = await setup();
    const output = await fixture.applicator.apply(application(command("set-limits", {
      version: 1,
      maxConcurrency: 2,
      maxCpuPercent: 70,
      maxRamMiB: 4_096,
      maxVramMiB: 2_048,
      maxDiskMiB: 8_192,
      maxTemperatureC: 80,
    })));
    expect(output).toMatchObject({ configRevision: 2, restartRequired: true });
    expect((await fixture.configStore.load()).config).toMatchObject({
      revision: 2,
      limits: { maxConcurrency: 2, maxCpuPercent: 70, maxRamMiB: 4_096 },
    });
  });

  it("persists schedule and exact model policy atomically before restarting", async () => {
    const fixture = await setup();
    const policy = { schedule: [{ days: [1, 2, 3, 4, 5], startMinuteUtc: 480, endMinuteUtc: 1080 }], modelAllowlist: ["org/model"] };
    const change = command("set-policy", { version: 1, policy });
    await expect(fixture.applicator.apply(application(change))).resolves.toMatchObject({ configRevision: 2, restartRequired: true, policy });
    expect((await fixture.configStore.load()).config.policy).toEqual(policy);
    expect(fixture.applicator.takeRestartAfterAck(change.id)).toBe(true);
  });

  it("drains, activates components, changes channel and requests restart only after ACK", async () => {
    const lifecycle: NodeComponentLifecycleApi = {
      update: vi.fn(async () => ({ state: "applied", manifest: { manifestId: `sha256:${"a".repeat(64)}` }, changedComponents: ["python-product"] }) as never),
      rollback: vi.fn(async () => ({ state: "rolled-back" as const, changedComponents: ["python-product"], activeComponentRoots: { "python-product": "/previous" } })),
    };
    const fixture = await setup(lifecycle);
    const update = command("update", { version: 1, channel: "dev", manifestId: `sha256:${"a".repeat(64)}` });
    expect(await fixture.applicator.apply(application(update))).toMatchObject({ state: "applied", restartRequired: true, configRevision: 2 });
    expect(lifecycle.update).toHaveBeenCalledWith("dev", `sha256:${"a".repeat(64)}`);
    expect((await fixture.configStore.load()).config.updateChannel).toBe("dev");
    expect(fixture.applicator.takeRestartAfterAck(update.id)).toBe(true);
    expect(fixture.applicator.takeRestartAfterAck(update.id)).toBe(false);

    const rollback = command("rollback", { version: 1, componentIds: ["python-product"] });
    expect(await fixture.applicator.apply(application(rollback))).toMatchObject({ state: "rolled-back", restartRequired: true });
    expect(fixture.applicator.takeRestartAfterAck(rollback.id)).toBe(true);
  });

  it("rejects lifecycle operations until their privileged helpers are integrated", async () => {
    const fixture = await setup();
    await expect(fixture.applicator.apply(application(command("update", { version: 1, channel: "stable" })))).rejects.toMatchObject({ code: "node_update_not_configured" });
    await expect(fixture.applicator.apply(application(command("uninstall", { version: 1, retain: { cache: true, logs: true, configuration: true, identity: true } })))).rejects.toMatchObject({ code: "node_uninstall_requires_service_helper" });
  });

  it("drains and schedules uninstall, then arms it only after the result ACK", async () => {
    const uninstall: NodeUninstallLifecycleApi = {
      schedule: vi.fn(async (): Promise<NodeUninstallReceipt> => ({ schema: "mycellios-node-uninstall-receipt/1", requestId: randomUUID(),
        requestDigest: `sha256:${"a".repeat(64)}`, state: "scheduled", platform: "linux", retained: ["identity"],
        createdAt: "2026-08-10T12:00:00.000Z" })),
      arm: vi.fn(async () => undefined),
    };
    const fixture = await setup(undefined, uninstall);
    const uninstallCommand = command("uninstall", { version: 1, retain: { cache: false, logs: false, configuration: false, identity: true } });
    const result = await fixture.applicator.apply(application(uninstallCommand));
    expect(result).toMatchObject({ state: "scheduled", stopAfterAck: true, retained: ["identity"] });
    expect(fixture.agent.waitForIdle).toHaveBeenCalledWith(60_000);
    expect(fixture.agent.setContributionEnabled).toHaveBeenLastCalledWith(false);
    expect(uninstall.arm).not.toHaveBeenCalled();
    expect(await fixture.applicator.armStopAfterAck(uninstallCommand.id)).toBe(true);
    expect(uninstall.arm).toHaveBeenCalledWith(expect.any(String), `sha256:${"a".repeat(64)}`);
    expect(await fixture.applicator.armStopAfterAck(uninstallCommand.id)).toBe(false);
  });
});

async function setup(lifecycle?: NodeComponentLifecycleApi, uninstall?: NodeUninstallLifecycleApi) {
  const directory = await mkdtemp(join(tmpdir(), "mycellios-command-applicator-"));
  cleanup.push(directory);
  const configStore = new NodeConfigurationStore(join(directory, "node.json"));
  const config = await configStore.save({
    schema: "mycellios-node-configuration/1",
    revision: 1,
    nodeId: "node-1",
    coordinator: { url: "https://coordinator.example.test", identityPath: join(directory, "identity.json") },
    worker: { configPath: join(directory, "worker.json") },
    runtime: { pythonExecutable: "/usr/bin/python3", pythonPath: join(directory, "python"), cachePath: join(directory, "cache"), stagePort: 9_850 },
    limits: { maxConcurrency: 1, maxCpuPercent: 90, maxRamMiB: 8_192, maxVramMiB: 0, maxDiskMiB: 32_768, maxTemperatureC: 85 },
    isolation: { mode: "linux-cgroup-v2" },
    updateChannel: "stable",
  });
  const control = new NodeControlStateStore(join(directory, "control.json"));
  const release = vi.fn(async () => undefined);
  const agent: NodeCommandAgent = {
    setContributionEnabled: vi.fn(async () => true),
    beginRuntimeUpdateDrain: vi.fn(async () => release),
    waitForIdle: vi.fn(async () => undefined),
  };
  return { configStore, control, agent, release, applicator: new NodeCommandApplicator(agent, control, configStore, config, lifecycle, uninstall) };
}

function command(type: string, payload: unknown): NodeCommand {
  return nodeCommandSchema.parse({
    schema: "mycellios-node-command/1",
    id: randomUUID(),
    nodeId: "node-1",
    actor: { kind: "account", id: "account-1", scopes: type === "set-limits" || type === "set-policy" ? ["node:limits"] : type === "update" || type === "rollback" ? ["node:update"] : type === "uninstall" ? ["node:identity"] : ["node:control"] },
    generation: 3,
    issuedAt: "2026-08-10T12:00:00.000Z",
    expiresAt: "2026-08-10T12:05:00.000Z",
    nonce: `nonce_${randomUUID().replaceAll("-", "")}`,
    type,
    payload,
  });
}

function application(commandValue: NodeCommand) {
  return { command: commandValue, operationKey: `node-command:${commandValue.id}` };
}
