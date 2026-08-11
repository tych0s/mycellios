import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdir, open, readFile, rename, rm } from "node:fs/promises";
import { promisify } from "node:util";
import { dirname, join } from "node:path";
import { z } from "zod";
import type { ComponentUpdateChannel, PinnedComponentUpdateKey } from "../contracts/component-update-manifest.js";
import type { NodeConfiguration } from "../contracts/node-configuration.js";
import { WORKER_PROTOCOL_MAX, WORKER_PROTOCOL_MIN } from "../contracts/worker-admission.js";
import {
  ComponentUpdateManager,
  readComponentInstallState,
  resolveActiveComponentRoot,
  rollbackInstalledComponents,
  rollbackFailedComponentActivation,
  type ComponentRollbackResult,
  type ComponentUpdateApplyResult,
} from "../update/component-update-manager.js";
import { MYCELLIOS_RUNTIME_ABI } from "../update/runtime-compatibility.js";

const execFileAsync = promisify(execFile);
const pendingActivationSchema = z.object({
  schema: z.literal("mycellios-node-component-activation/1"),
  channel: z.enum(["dev", "stable"]),
  changedComponents: z.array(z.string().min(1)).min(1),
  targetManifestIds: z.record(z.string(), z.string().regex(/^sha256:[a-f0-9]{64}$/)),
  requestedAt: z.string().datetime(),
}).strict();

export interface NodeComponentLifecycleApi {
  update(channel: ComponentUpdateChannel, manifestId?: `sha256:${string}`): Promise<ComponentUpdateApplyResult>;
  rollback(componentIds: readonly string[]): Promise<ComponentRollbackResult>;
}

export class NodeComponentLifecycle implements NodeComponentLifecycleApi {
  readonly storageRoot: string;

  constructor(
    private readonly config: NodeConfiguration,
    private readonly bootstrapVersion: string,
    private readonly isIdle: () => boolean | Promise<boolean>,
    private readonly acquireActivationLease: () => Promise<(() => void | Promise<void>) | null>,
    private readonly fetchImpl: typeof fetch = fetch,
    private readonly canary: (componentRoot: string) => Promise<void> = (root) =>
      canaryPythonProduct(config.runtime.pythonExecutable, root),
  ) {
    this.storageRoot = componentStorageRoot(config);
  }

  async update(
    channel: ComponentUpdateChannel,
    manifestId?: `sha256:${string}`,
  ): Promise<ComponentUpdateApplyResult> {
    const trust = requireUpdateTrust(this.config, channel);
    return new ComponentUpdateManager({
      storageRoot: this.storageRoot,
      feedBaseUrl: trust.feedUrl,
      channel,
      pinnedKeys: trust.pinnedKeys,
      bootstrapVersion: this.bootstrapVersion,
      workerProtocol: { min: WORKER_PROTOCOL_MIN, max: WORKER_PROTOCOL_MAX },
      runtimeAbi: MYCELLIOS_RUNTIME_ABI,
      managedComponentIds: ["python-product"],
      ...(manifestId ? { expectedManifestId: manifestId } : {}),
      fetch: this.fetchImpl,
      isIdle: this.isIdle,
      acquireActivationLease: this.acquireActivationLease,
      onPrepare: async ({ stagedComponentRoots }) => {
        const pythonProduct = stagedComponentRoots["python-product"];
        if (pythonProduct) await this.canary(pythonProduct);
      },
      // The pointer is consumed only by the next service process. The caller
      // requests that restart after its durable command result is ACKed.
      onActivate: async ({ manifest, changedComponents }) => {
        const targetManifestIds: Record<string, `sha256:${string}`> = {};
        for (const id of changedComponents) targetManifestIds[id] = manifest.manifestId as `sha256:${string}`;
        await this.writePending(channel, changedComponents, targetManifestIds);
      },
    }).checkAndApply();
  }

  async rollback(componentIds: readonly string[]): Promise<ComponentRollbackResult> {
    const trust = requireUpdateTrust(this.config, this.config.updateChannel);
    const before = await readComponentInstallState(this.storageRoot);
    const targetManifestIds: Record<string, `sha256:${string}`> = {};
    for (const id of componentIds) {
      const previous = before.previous[id];
      if (!previous) throw new Error(`component_rollback_previous_missing:${id}`);
      targetManifestIds[id] = previous.manifestId as `sha256:${string}`;
    }
    return rollbackInstalledComponents({
      storageRoot: this.storageRoot,
      componentIds,
      expectedChannel: this.config.updateChannel,
      pinnedKeys: trust.pinnedKeys,
      onPrepare: async ({ stagedComponentRoots }) => {
        const pythonProduct = stagedComponentRoots["python-product"];
        if (pythonProduct) await this.canary(pythonProduct);
        await this.writePending(this.config.updateChannel, componentIds, targetManifestIds);
      },
    });
  }

  async pendingActivation(): Promise<z.infer<typeof pendingActivationSchema> | null> {
    try {
      return pendingActivationSchema.parse(JSON.parse(await readFile(this.pendingPath(), "utf8")));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
      throw new Error("node_component_pending_activation_invalid", { cause: error });
    }
  }

  async markActivationHealthy(): Promise<void> {
    await rm(this.pendingPath(), { force: true });
  }

  async pendingActivationIsCurrent(): Promise<boolean> {
    const pending = await this.pendingActivation();
    if (!pending) return false;
    const state = await readComponentInstallState(this.storageRoot);
    return pending.changedComponents.every((id) =>
      state.active[id]?.manifestId === pending.targetManifestIds[id]
    );
  }

  async rollbackFailedActivation(): Promise<{ state: "rolled-back-failed-activation"; changedComponents: string[] }> {
    const pending = await this.pendingActivation();
    if (!pending) throw new Error("node_component_pending_activation_missing");
    const result = await rollbackFailedComponentActivation({
      storageRoot: this.storageRoot,
      targetManifestIds: pending.targetManifestIds as Record<string, `sha256:${string}`>,
      expectedChannel: pending.channel,
      pinnedKeys: requireUpdateTrust(this.config, pending.channel).pinnedKeys,
      onPrepare: async ({ previousComponentRoots }) => {
        const pythonProduct = previousComponentRoots["python-product"];
        if (pythonProduct) await this.canary(pythonProduct);
      },
    });
    await this.markActivationHealthy();
    return result;
  }

  private pendingPath(): string {
    return join(this.storageRoot, "pending-activation.json");
  }

  private async writePending(
    channel: ComponentUpdateChannel,
    changedComponents: readonly string[],
    targetManifestIds: Readonly<Record<string, `sha256:${string}`>>,
  ): Promise<void> {
    if (changedComponents.length === 0) return;
    const document = pendingActivationSchema.parse({
      schema: "mycellios-node-component-activation/1",
      channel,
      changedComponents: [...changedComponents],
      targetManifestIds,
      requestedAt: new Date().toISOString(),
    });
    const path = this.pendingPath();
    await mkdir(dirname(path), { recursive: true, mode: 0o700 });
    const temporary = `${path}.tmp-${randomUUID()}`;
    try {
      const file = await open(temporary, "wx", 0o600);
      try { await file.writeFile(`${JSON.stringify(document, null, 2)}\n`, "utf8"); await file.sync(); }
      finally { await file.close(); }
      await rename(temporary, path);
    } finally {
      await rm(temporary, { force: true });
    }
  }
}

export async function resolveNodePythonProduct(config: NodeConfiguration): Promise<string | null> {
  if (!config.componentUpdates) return null;
  const keys = config.componentUpdates.pinnedKeys[config.updateChannel];
  if (keys.length === 0) return null;
  return resolveActiveComponentRoot(
    componentStorageRoot(config),
    "python-product",
    config.updateChannel,
    keys,
  );
}

export async function canaryPythonProduct(pythonExecutable: string, componentRoot: string): Promise<void> {
  try {
    const { stdout } = await execFileAsync(pythonExecutable, [
      "-I",
      "-c",
      "import sys; sys.path.insert(0, sys.argv[1]); import distributed_runtime; print('MYCELLIOS_COMPONENT_CANARY_OK')",
      componentRoot,
    ], {
      timeout: 30_000,
      maxBuffer: 64 * 1024,
      windowsHide: true,
      env: {
        PYTHONIOENCODING: "utf-8",
        PYTHONNOUSERSITE: "1",
        ...(process.platform === "win32" && process.env.SystemRoot ? { SystemRoot: process.env.SystemRoot } : {}),
      },
    });
    if (stdout.trim() !== "MYCELLIOS_COMPONENT_CANARY_OK") throw new Error("marker_missing");
  } catch (error) {
    throw new Error("node_python_product_canary_failed", { cause: error });
  }
}

function componentStorageRoot(config: NodeConfiguration): string {
  return join(config.runtime.cachePath, "component-updates");
}

function requireUpdateTrust(
  config: NodeConfiguration,
  channel: ComponentUpdateChannel,
): { feedUrl: string; pinnedKeys: readonly PinnedComponentUpdateKey[] } {
  const updates = config.componentUpdates;
  const pinnedKeys = updates?.pinnedKeys[channel] ?? [];
  if (!updates || pinnedKeys.length === 0) throw new Error(`node_update_trust_not_configured:${channel}`);
  return { feedUrl: updates.feedUrl, pinnedKeys };
}
