import { readFile } from "node:fs/promises";
import { isAbsolute, resolve } from "node:path";
import type { NativeBuildIdentity } from "../contracts/build-identity.js";
import type { StoredRequestedModel } from "../storage/store.js";
import {
  compileAutoDistribution,
  parseAutoDistributionConfig,
  profileCompatibleModel,
  runAutoDistribution,
  writeAutoDistributionArtifacts,
  type AutoDistributionConfig,
  type AutoDistributionProgressEvent,
  type AutoDistributionRunResult,
} from "../distribution/auto-distribute.js";
import { HttpLaunchAgent } from "../distribution/launch-agent-rpc.js";
import {
  bindPythonLaunchDeploymentGeneration,
  bindPythonLaunchDraftStrategyAuthority,
  pythonLaunchRequiresDraftStrategyAuthority,
  type PythonPipelineLaunchDescription,
} from "../distribution/python-launcher.js";
import type { ResolvedDraftStrategy } from "../distribution/draft-strategy-registry.js";
import type { ModelActivationProgressEvent, ModelExecutionCapacityNode } from "./model-catalog.js";

export type AutomaticModelRunner = (
  config: AutoDistributionConfig,
  signal: AbortSignal,
) => Promise<void>;

export interface ModelActivationManager {
  initialize(): Promise<void>;
  refresh(): Promise<void>;
  capacityNodesForModel(modelId: string): readonly ModelExecutionCapacityNode[];
  activationProgressForModel?(modelId: string): readonly ModelActivationProgressEvent[];
  isManaging(modelId: string): boolean;
  isBusy(): boolean;
  activate(model: StoredRequestedModel): Promise<void>;
  deactivate(modelId: string): Promise<boolean>;
  close(): Promise<void>;
}

export interface DynamicActivationSnapshot {
  capacityNodes: readonly ModelExecutionCapacityNode[];
  config: AutoDistributionConfig | null;
  readinessDetails?: readonly string[];
}

export interface DynamicActivationRouteStage {
  nodeId: string;
  stageIndex: number;
  layerStart: number;
  layerEnd: number;
  memoryMiB: number;
  capacityMiB: number;
  artifactBytes: number;
}

export interface DynamicActivationPreparedRoute {
  id: string;
  generation: number;
}

export interface DynamicModelActivationManagerOptions {
  snapshot(): DynamicActivationSnapshot;
  resolveManagedAgent(
    nodeId: string,
    launch: import("../distribution/python-launcher.js").PythonPipelineLaunchDescription,
  ): import("../distribution/launch-supervisor.js").LaunchAgent | undefined;
  /** Resolve only pinned, verified strategy grants; raw catalog data is forbidden. */
  resolveDraftStrategyAuthority?(
    launch: PythonPipelineLaunchDescription,
  ): ResolvedDraftStrategy | null | Promise<ResolvedDraftStrategy | null>;
  cwd?: string;
  environment?: NodeJS.ProcessEnv;
  workerBuildIdentity?: NativeBuildIdentity;
  workerAgentVersion?: string;
  loadProgress?(modelId: string): readonly ModelActivationProgressEvent[];
  onProgress?(modelId: string, event: ModelActivationProgressEvent): void;
  onPlanPrepared?(
    modelId: string,
    stages: readonly DynamicActivationRouteStage[],
  ): DynamicActivationPreparedRoute | Promise<DynamicActivationPreparedRoute>;
  onPlanHeartbeat?(modelId: string, reservationId: string): boolean | Promise<boolean>;
  reservationHeartbeatMs?: number;
  onActivated?(
    modelId: string,
    reservationId: string | null,
    result: AutoDistributionRunResult,
  ): void | Promise<void>;
  onStopped?(modelId: string): void | Promise<void>;
}

export class AutomaticModelActivationManager implements ModelActivationManager {
  private closed = false;
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
    if (this.closed) return [];
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
    if (this.closed) return Promise.reject(new Error("automatic_activation_manager_closed"));
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
    this.closed = true;
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
      artifactsDirectory: modelArtifactsDirectory(this.baseConfig, model.id),
    });
  }

  private async refreshNodeHealth(): Promise<void> {
    const checks = await Promise.all(this.baseConfig.nodes.map(async (node) => {
      if (node.agent.kind === "local") return { id: node.id, healthy: true };
      if (node.agent.kind === "managed") return { id: node.id, healthy: true };
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

/** Activation manager backed by the shard executors currently connected to the coordinator. */
export class DynamicModelActivationManager implements ModelActivationManager {
  private closed = false;
  private current: DynamicActivationSnapshot = { capacityNodes: [], config: null };
  private activeModelId: string | null = null;
  private activeAbort: AbortController | null = null;
  private activePromise: Promise<void> | null = null;
  private readonly progress = new Map<string, ModelActivationProgressEvent[]>();

  constructor(private readonly options: DynamicModelActivationManagerOptions) {}

  async initialize(): Promise<void> { await this.refresh(); }

  async refresh(): Promise<void> {
    this.current = this.options.snapshot();
  }

  capacityNodesForModel(modelId: string): readonly ModelExecutionCapacityNode[] {
    if (this.closed) return [];
    if (this.activeModelId !== null && this.activeModelId !== modelId) return [];
    return this.current.capacityNodes.map((node) => ({ ...node }));
  }

  isManaging(modelId: string): boolean {
    return this.activeModelId === modelId && this.activePromise !== null;
  }

  isBusy(): boolean { return this.activePromise !== null; }

  activationProgressForModel(modelId: string): readonly ModelActivationProgressEvent[] {
    const cached = this.progress.get(modelId);
    if (cached) return cached.map((event) => ({ ...event }));
    const restored = this.options.loadProgress?.(modelId).map((event) => ({ ...event })) ?? [];
    if (restored.length > 0) this.progress.set(modelId, restored.slice(-100));
    return restored;
  }

  activate(model: StoredRequestedModel): Promise<void> {
    if (this.closed) return Promise.reject(new Error("automatic_activation_manager_closed"));
    if (this.isManaging(model.id)) return this.activePromise!;
    if (this.activePromise) return Promise.reject(new Error(`automatic_activation_busy:${this.activeModelId}`));
    this.progress.set(model.id, []);
    this.appendProgress(model.id, "queued", "Activation accepted by the coordinator.");
    const base = this.current.config;
    if (!base) {
      const error = new Error("distributed_activation_requires_two_connected_shard_executors");
      this.appendProgress(
        model.id,
        "failed",
        "The connected PCs have not finished preparing a verified two-node execution route.",
        "failed",
        {
          details: this.current.readinessDetails ?? [
            "Two connected shard executors must finish runtime and reciprocal-link verification.",
          ],
        },
      );
      return Promise.reject(error);
    }
    let config = parseAutoDistributionConfig({
      ...structuredClone(base),
      model: { source: model.source, revision: model.revision, publicName: model.id },
      distribution: {
        ...base.distribution,
        minimumStages: model.minimumNodes,
        maximumStages: Math.max(model.minimumNodes, Math.min(base.distribution.maximumStages, base.nodes.length)),
      },
      workload: { ...base.workload, contextTokens: model.contextTokens },
      artifactsDirectory: modelArtifactsDirectory(base, model.id),
    });
    const controller = new AbortController();
    this.activeModelId = model.id;
    this.activeAbort = controller;
    const cwd = this.options.cwd ?? process.cwd();
    const environment = this.options.environment ?? process.env;
    const running = (async () => {
      this.appendProgress(model.id, "profiling", "Reading the model and preparing its execution profile.");
      const profile = await profileCompatibleModel(config, cwd, environment);
      this.appendProgress(
        model.id,
        "profile_ready",
        "Model profile ready. Calculating the layer distribution.",
        "running",
        {
          details: [
            `Model: ${profile.source.model}`,
            `Architecture: ${profile.inspection.architecture ?? "not reported"}`,
            `Layers detected: ${profile.model.layers.length}`,
            `Snapshot: ${profile.source.snapshotCommit ?? profile.source.revision ?? "default revision"}`,
          ],
        },
      );
      if (controller.signal.aborted) return;
      let compilation = compileAutoDistribution(config, profile);
      const rootHost = compilation.manifest.plans.decode.stages[0]?.anchor.endpoint.host;
      if (
        rootHost
        && (config.runtime.apiAdvertiseHost !== rootHost
          || config.runtime.returnEndpoint.host !== rootHost)
      ) {
        config = parseAutoDistributionConfig({
          ...structuredClone(config),
          runtime: {
            ...config.runtime,
            apiAdvertiseHost: rootHost,
            returnEndpoint: { ...config.runtime.returnEndpoint, host: rootHost },
          },
        });
        compilation = compileAutoDistribution(config, profile);
      }
      await writeAutoDistributionArtifacts(config, compilation, cwd);
      const reservationStages = routeStagesForReservation(config, compilation);
      const preparedRoute = this.options.onPlanPrepared
        ? await this.options.onPlanPrepared(
            model.id,
            reservationStages,
          )
        : null;
      const routeReservationId = preparedRoute?.id ?? null;
      if (preparedRoute !== null) {
        compilation = {
          ...compilation,
          launch: bindPythonLaunchDeploymentGeneration(
            compilation.launch,
            preparedRoute.generation,
          ),
        };
      }
      if (pythonLaunchRequiresDraftStrategyAuthority(compilation.launch)) {
        const authority = await this.options.resolveDraftStrategyAuthority?.(
          compilation.launch,
        );
        if (!authority) {
          throw new Error("distributed_activation_draft_strategy_authority_is_required");
        }
        compilation = {
          ...compilation,
          launch: bindPythonLaunchDraftStrategyAuthority(
            compilation.launch,
            authority,
          ),
        };
      }
      await writeAutoDistributionArtifacts(config, compilation, cwd);
      this.appendProgress(
        model.id,
        "plan_ready",
        `Distribution plan ready: ${compilation.manifest.plans.decode.stages.length} stages across ${config.nodes.length} nodes.`,
        "running",
        {
          details: compilation.manifest.plans.decode.stages.map((stage, index, stages) => (
            `Stage ${index + 1}/${stages.length}: layers ${stage.layerStart}–`
            + `${Math.max(stage.layerStart, stage.layerEnd - 1)} of ${profile.model.layers.length} · `
            + `node ${stage.anchor.memberId} · `
            + `${reservationStages
              .filter((reservation) => reservation.stageIndex === stage.index)
              .reduce((total, reservation) => total + (reservation.artifactBytes ?? 0), 0)} artifact bytes`
          )),
        },
      );
      if (controller.signal.aborted) return;
      const activation = runAutoDistribution(config, compilation, cwd, environment, controller.signal, {
        resolveManagedAgent: (nodeId, launch) => this.options.resolveManagedAgent(nodeId, launch),
        onProgress: (event) => this.appendRuntimeProgress(model.id, event),
        ...(this.options.workerBuildIdentity
          ? { workerBuildIdentity: this.options.workerBuildIdentity }
          : {}),
        ...(this.options.workerAgentVersion
          ? { workerAgentVersion: this.options.workerAgentVersion }
          : {}),
        ...(this.options.onActivated
          ? {
              onActivated: (result: AutoDistributionRunResult) =>
                this.options.onActivated!(model.id, routeReservationId, result),
            }
          : {}),
      });
      await (routeReservationId && this.options.onPlanHeartbeat
        ? runWithReservationHeartbeat(
            activation,
            () => this.options.onPlanHeartbeat!(model.id, routeReservationId),
            this.options.reservationHeartbeatMs ?? 30_000,
            (error) => controller.abort(error),
          )
        : activation);
    })().catch((error: unknown) => {
      this.failProgress(model.id, error instanceof Error ? error.message : String(error));
      throw error;
    });
    this.activePromise = running.finally(async () => {
      await this.options.onStopped?.(model.id);
      this.activePromise = null;
      this.activeModelId = null;
      this.activeAbort = null;
    });
    return this.activePromise;
  }

  private appendRuntimeProgress(modelId: string, event: AutoDistributionProgressEvent): void {
    this.appendProgress(
      modelId,
      event.phase,
      event.message,
      event.phase === "active" ? "completed" : event.phase === "failed" ? "failed" : "running",
      {
        ...(event.nodeId ? { nodeId: event.nodeId } : {}),
        ...(event.processId ? { processId: event.processId } : {}),
        ...(event.device ? { device: event.device } : {}),
        ...(event.details ? { details: [...event.details] } : {}),
      },
    );
  }

  private appendProgress(
    modelId: string,
    phase: string,
    message: string,
    state: ModelActivationProgressEvent["state"] = "running",
    context: Pick<ModelActivationProgressEvent, "nodeId" | "processId" | "device" | "details"> = {},
  ): void {
    const events = this.progress.get(modelId) ?? [];
    const previous = events.at(-1);
    if (previous?.state === "running") previous.state = "completed";
    if (previous?.phase === phase && previous.message === message) return;
    const event = { phase, message, at: new Date().toISOString(), state, ...context };
    events.push(event);
    this.progress.set(modelId, events.slice(-100));
    this.options.onProgress?.(modelId, { ...event });
  }

  private failProgress(modelId: string, message: string): void {
    const events = this.progress.get(modelId) ?? [];
    const previous = events.at(-1);
    if (previous?.phase === "failed") return;
    if (previous?.state === "running") previous.state = "failed";
    const event = {
      phase: "failed",
      message,
      at: new Date().toISOString(),
      state: "failed" as const,
    };
    events.push(event);
    this.progress.set(modelId, events.slice(-100));
    this.options.onProgress?.(modelId, { ...event });
  }

  async deactivate(modelId: string): Promise<boolean> {
    if (!this.isManaging(modelId)) return false;
    const running = this.activePromise;
    this.activeAbort?.abort(new Error(`automatic_model_deactivated:${modelId}`));
    if (running) await running.catch(() => undefined);
    return true;
  }

  async close(): Promise<void> {
    this.closed = true;
    if (this.activeModelId) await this.deactivate(this.activeModelId);
  }
}

export async function runWithReservationHeartbeat<T>(
  task: Promise<T>,
  renew: () => boolean | Promise<boolean>,
  intervalMs: number,
  onFailure: (error: Error) => void = () => undefined,
): Promise<T> {
  if (!Number.isSafeInteger(intervalMs) || intervalMs < 100) {
    throw new Error("route_reservation_heartbeat_interval_is_invalid");
  }
  let rejectFailure!: (error: Error) => void;
  const failure = new Promise<never>((_resolve, reject) => { rejectFailure = reject; });
  let renewing = false;
  const timer = setInterval(() => {
    if (renewing) return;
    renewing = true;
    void Promise.resolve(renew()).then(
      (renewed) => {
        renewing = false;
        if (renewed) return;
        const error = new Error("route_reservation_heartbeat_rejected");
        onFailure(error);
        rejectFailure(error);
      },
      (cause: unknown) => {
        renewing = false;
        const error = cause instanceof Error
          ? cause
          : new Error(`route_reservation_heartbeat_failed:${String(cause)}`);
        onFailure(error);
        rejectFailure(error);
      },
    );
  }, intervalMs);
  timer.unref?.();
  try {
    return await Promise.race([task, failure]);
  } finally {
    clearInterval(timer);
  }
}

function routeStagesForReservation(
  config: AutoDistributionConfig,
  compilation: ReturnType<typeof compileAutoDistribution>,
): DynamicActivationRouteStage[] {
  const nodes = new Map(config.nodes.map((node) => [node.id, node]));
  const model = compilation.profile.model;
  return compilation.manifest.plans.decode.stages.flatMap((stage) =>
    stage.members.map((member, memberIndex, members) => {
      const node = nodes.get(member.nodeId);
      if (!node) throw new Error(`distribution_plan_references_unknown_node:${member.nodeId}`);
      const stageArtifactBytes = plannedStageArtifactBytes(
        model,
        stage.layerStart,
        stage.layerEnd,
      );
      const assignedTotal = members.reduce(
        (total, candidate) => total + candidate.assignedMemoryBytes,
        0,
      );
      const assignedBefore = members.slice(0, memberIndex).reduce(
        (total, candidate) => total + candidate.assignedMemoryBytes,
        0,
      );
      const assignedThrough = assignedBefore + member.assignedMemoryBytes;
      return {
        nodeId: member.nodeId,
        stageIndex: stage.index,
        layerStart: stage.layerStart,
        layerEnd: stage.layerEnd,
        memoryMiB: Math.ceil(member.assignedMemoryBytes / (1024 * 1024)),
        capacityMiB: Math.max(0, node.memoryMiB - node.reserveMiB),
        artifactBytes: assignedTotal === 0
          ? memberIndex === 0 ? stageArtifactBytes : 0
          : Math.floor(stageArtifactBytes * assignedThrough / assignedTotal)
            - Math.floor(stageArtifactBytes * assignedBefore / assignedTotal),
      };
    })
  );
}

export function plannedStageArtifactBytes(
  model: import("../distribution/types.js").DistributedModelProfile,
  layerStart: number,
  layerEnd: number,
): number {
  if (
    !Number.isInteger(layerStart)
    || !Number.isInteger(layerEnd)
    || layerStart < 0
    || layerEnd <= layerStart
    || layerEnd > model.layers.length
  ) {
    throw new Error("planned_stage_artifact_range_is_invalid");
  }
  let bytes = model.layers
    .slice(layerStart, layerEnd)
    .reduce((total, layer) => total + layer.weightBytes, 0);
  if (layerStart === 0) bytes += model.embeddingBytes;
  if (layerEnd === model.layers.length) {
    if (!(model.tiedEmbeddingAndHead && layerStart === 0)) bytes += model.lmHeadBytes;
  }
  if (!Number.isSafeInteger(bytes) || bytes < 0) {
    throw new Error("planned_stage_artifact_bytes_are_invalid");
  }
  return bytes;
}

function modelArtifactsDirectory(base: AutoDistributionConfig, modelId: string): string {
  const configuredRoot = base.artifactsDirectory?.trim();
  return configuredRoot
    ? resolve(configuredRoot, modelId)
    : `runtime/auto-distribute/${modelId}`;
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
