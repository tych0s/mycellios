import { readFile } from "node:fs/promises";
import { isAbsolute, resolve } from "node:path";
import type { StoredRequestedModel } from "../storage/store.js";
import {
  compileAutoDistribution,
  parseAutoDistributionConfig,
  profileCompatibleModel,
  runAutoDistribution,
  writeAutoDistributionArtifacts,
  type AutoDistributionConfig,
} from "../distribution/auto-distribute.js";
import { HttpLaunchAgent } from "../distribution/launch-agent-rpc.js";
import type { ModelExecutionCapacityNode } from "./model-catalog.js";

export type AutomaticModelRunner = (
  config: AutoDistributionConfig,
  signal: AbortSignal,
) => Promise<void>;

export interface ModelActivationManager {
  initialize(): Promise<void>;
  refresh(): Promise<void>;
  capacityNodesForModel(modelId: string): readonly ModelExecutionCapacityNode[];
  isManaging(modelId: string): boolean;
  isBusy(): boolean;
  activate(model: StoredRequestedModel): Promise<void>;
  deactivate(modelId: string): Promise<boolean>;
  close(): Promise<void>;
}

export class AutomaticModelActivationManager implements ModelActivationManager {
  private readonly healthyNodeIds = new Set<string>();
  private activeModelId: string | null = null;
  private activeAbort: AbortController | null = null;
  private activePromise: Promise<void> | null = null;
  private refreshPromise: Promise<void> | null = null;

  constructor(
    private readonly baseConfig: AutoDistributionConfig,
    private readonly cwd = process.cwd(),
    private readonly environment: NodeJS.ProcessEnv = process.env,
    private readonly runner: AutomaticModelRunner = defaultAutomaticModelRunner,
  ) {}

  static async fromFile(
    configPath: string,
    cwd = process.cwd(),
    environment: NodeJS.ProcessEnv = process.env,
    runner?: AutomaticModelRunner,
  ): Promise<AutomaticModelActivationManager> {
    const absolutePath = isAbsolute(configPath) ? configPath : resolve(cwd, configPath);
    const value = JSON.parse(await readFile(absolutePath, "utf8")) as unknown;
    return new AutomaticModelActivationManager(
      parseAutoDistributionConfig(value),
      cwd,
      environment,
      runner,
    );
  }

  async initialize(): Promise<void> {
    await this.refresh();
  }

  async refresh(): Promise<void> {
    if (this.refreshPromise) return this.refreshPromise;
    this.refreshPromise = this.refreshNodeHealth().finally(() => {
      this.refreshPromise = null;
    });
    return this.refreshPromise;
  }

  capacityNodesForModel(modelId: string): readonly ModelExecutionCapacityNode[] {
    if (this.activeModelId !== null && this.activeModelId !== modelId) return [];
    return this.baseConfig.nodes
      .filter((node) => this.healthyNodeIds.has(node.id))
      .map((node) => ({
        id: node.id,
        availableVramMiB: Math.max(0, node.memoryMiB - node.reserveMiB),
      }));
  }

  isManaging(modelId: string): boolean {
    return this.activeModelId === modelId && this.activePromise !== null;
  }

  isBusy(): boolean {
    return this.activePromise !== null;
  }

  activate(model: StoredRequestedModel): Promise<void> {
    if (this.isManaging(model.id)) return this.activePromise!;
    if (this.activePromise) {
      return Promise.reject(new Error(`automatic_activation_busy:${this.activeModelId}`));
    }
    const config = this.configFor(model);
    const controller = new AbortController();
    this.activeModelId = model.id;
    this.activeAbort = controller;
    const running = this.runner(config, controller.signal);
    this.activePromise = running.finally(() => {
      if (this.activePromise === running || this.activeModelId === model.id) {
        this.activePromise = null;
        this.activeModelId = null;
        this.activeAbort = null;
      }
    });
    return this.activePromise;
  }

  async deactivate(modelId: string): Promise<boolean> {
    if (!this.isManaging(modelId)) return false;
    const running = this.activePromise;
    this.activeAbort?.abort(new Error(`automatic_model_deactivated:${modelId}`));
    if (running) await running.catch(() => undefined);
    return true;
  }

  async close(): Promise<void> {
    if (this.activeModelId) await this.deactivate(this.activeModelId);
  }

  private configFor(model: StoredRequestedModel): AutoDistributionConfig {
    const minimumStages = model.minimumNodes;
    return parseAutoDistributionConfig({
      ...structuredClone(this.baseConfig),
      model: {
        source: model.source,
        revision: model.revision,
        publicName: model.id,
      },
      distribution: {
        ...this.baseConfig.distribution,
        minimumStages,
        maximumStages: Math.max(
          minimumStages,
          Math.min(this.baseConfig.distribution.maximumStages, this.baseConfig.nodes.length),
        ),
      },
      workload: {
        ...this.baseConfig.workload,
        contextTokens: model.contextTokens,
      },
      artifactsDirectory: `runtime/auto-distribute/${model.id}`,
    });
  }

  private async refreshNodeHealth(): Promise<void> {
    const checks = await Promise.all(this.baseConfig.nodes.map(async (node) => {
      if (node.agent.kind === "local") return { id: node.id, healthy: true };
      try {
        const token = node.agent.authTokenEnv
          ? this.environment[node.agent.authTokenEnv]?.trim()
          : undefined;
        if (node.agent.authTokenEnv && !token) return { id: node.id, healthy: false };
        const health = await new HttpLaunchAgent({
          endpoint: node.agent.endpoint,
          id: `activation-health-${node.id}`,
          requestTimeoutMs: node.agent.requestTimeoutMs,
          ...(token ? { authToken: token } : {}),
        }).health();
        return {
          id: node.id,
          healthy: health.nodeId === null || health.nodeId === node.id,
        };
      } catch {
        return { id: node.id, healthy: false };
      }
    }));
    this.healthyNodeIds.clear();
    for (const check of checks) if (check.healthy) this.healthyNodeIds.add(check.id);
  }
}

async function defaultAutomaticModelRunner(
  config: AutoDistributionConfig,
  signal: AbortSignal,
): Promise<void> {
  const profile = await profileCompatibleModel(config);
  if (signal.aborted) return;
  const compilation = compileAutoDistribution(config, profile);
  await writeAutoDistributionArtifacts(config, compilation);
  if (signal.aborted) return;
  await runAutoDistribution(config, compilation, process.cwd(), process.env, signal);
}
