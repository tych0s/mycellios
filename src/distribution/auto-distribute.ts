import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { isAbsolute, resolve } from "node:path";
import { z } from "zod";
import { workerConfigSchema, type WorkerConfig } from "../contracts/schemas.js";
import { WorkerAgent } from "../worker/agent.js";
import { evaluateDistributionPlan, stageMemoryBytes } from "./cost-model.js";
import { HttpLaunchAgent } from "./launch-agent-rpc.js";
import {
  LocalProcessAgent,
  PythonLaunchSupervisor,
  type LaunchAgent,
} from "./launch-supervisor.js";
import {
  compilePythonLaunchDescription,
  type PythonPipelineLaunchDescription,
} from "./python-launcher.js";
import {
  buildRuntimePipelineManifest,
  type RuntimePipelineManifestV2,
  type RuntimePlanRequest,
  type RuntimeTopology,
} from "./runtime-manifest.js";
import type {
  DirectedLinkProfile,
  ComputeNodeProfile,
  DistributedModelProfile,
  DistributionPlan,
  DistributionWorkload,
  StagePlacement,
} from "./types.js";

export const AUTO_DISTRIBUTE_SCHEMA = "gdlp-auto-distribute/1";
export const MODEL_PROFILE_SCHEMA = "gdlp-model-profile/1";
const MIB = 1024 * 1024;
const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const ENVIRONMENT_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;

const endpointSchema = z
  .object({
    host: z.string().min(1).max(253),
    port: z.number().int().min(1).max(65_535),
  })
  .strict();

const agentSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("local") }).strict(),
  z.object({ kind: z.literal("managed") }).strict(),
  z
    .object({
      kind: z.literal("http"),
      endpoint: z.url(),
      authTokenEnv: z.string().regex(ENVIRONMENT_NAME).optional(),
      requestTimeoutMs: z.number().int().min(1_000).max(300_000).default(30_000),
    })
    .strict(),
]);

const nodeSchema = z
  .object({
    id: z.string().regex(IDENTIFIER),
    region: z.string().min(1).max(128),
    endpoint: endpointSchema,
    memoryMiB: z.number().int().min(512),
    reserveMiB: z.number().int().nonnegative().default(256),
    decodeScale: z.number().positive().finite().default(1),
    prefillScale: z.number().positive().finite().default(1),
    codecScale: z.number().positive().finite().default(1),
    powerWatts: z.number().positive().finite().default(65),
    availability: z.number().positive().max(1).default(0.999),
    agent: agentSchema,
  })
  .strict()
  .refine((node) => node.reserveMiB < node.memoryMiB, {
    message: "reserveMiB must be smaller than memoryMiB",
    path: ["reserveMiB"],
  });

const linkSchema = z
  .object({
    from: z.string().regex(IDENTIFIER),
    to: z.string().regex(IDENTIFIER),
    oneWayLatencyMs: z.number().nonnegative().finite(),
    jitterP95Ms: z.number().nonnegative().finite().default(0),
    bandwidthMbps: z.number().positive().finite(),
    lossRate: z.number().min(0).max(0.9).default(0),
    availability: z.number().positive().max(1).default(0.999),
  })
  .strict();

export const autoDistributionConfigSchema = z
  .object({
    schema: z.literal(AUTO_DISTRIBUTE_SCHEMA),
    model: z
      .object({
        source: z.string().min(1).max(4_096),
        revision: z.string().min(1).max(512).nullable().default(null),
        publicName: z.string().regex(IDENTIFIER),
      })
      .strict(),
    nodes: z.array(nodeSchema).min(2).max(64),
    links: z.array(linkSchema).max(4_032).default([]),
    distribution: z
      .object({
        minimumStages: z.number().int().min(2).max(8).default(2),
        maximumStages: z.number().int().min(2).max(8).default(4),
        allowLossyActivation: z.boolean().default(false),
      })
      .strict()
      .default({ minimumStages: 2, maximumStages: 4, allowLossyActivation: false }),
    workload: z
      .object({
        promptTokens: z.number().int().positive().default(128),
        outputTokens: z.number().int().positive().default(128),
        contextTokens: z.number().int().positive().default(4_096),
        concurrentSequences: z.number().int().positive().default(1),
        minRouteAvailability: z.number().positive().max(1).default(0.9),
        batchWindowMs: z.number().nonnegative().finite().default(2),
        p95: z.boolean().default(true),
      })
      .strict()
      .default({
        promptTokens: 128,
        outputTokens: 128,
        contextTokens: 4_096,
        concurrentSequences: 1,
        minRouteAvailability: 0.9,
        batchWindowMs: 2,
        p95: true,
      }),
    runtime: z
      .object({
        pythonExecutable: z.string().min(1).max(4_096),
        stagePythonExecutable: z.string().min(1).max(4_096).optional(),
        pythonPath: z.string().min(1).max(4_096).default("python"),
        hfHome: z.string().min(1).max(4_096).default("runtime/hf-cache"),
        apiEndpoint: endpointSchema,
        apiAdvertiseHost: z.string().min(1).max(253).default("127.0.0.1"),
        returnEndpoint: endpointSchema,
        returnBindHost: z.string().min(1).max(253).default("0.0.0.0"),
        threadsPerStage: z.number().int().positive().max(1_024).default(1),
        connectTimeoutSeconds: z.number().positive().max(86_400).default(300),
        readinessTimeoutMs: z.number().int().positive().max(86_400_000).default(600_000),
        maxOutputTokens: z.number().int().positive().max(32_768).default(2_048),
      })
      .strict(),
    canary: z
      .object({
        prompt: z.string().min(1).max(100_000).default("Reply with only OK. /no_think"),
        maxTokens: z.number().int().positive().max(512).default(16),
        timeoutMs: z.number().int().min(1_000).max(900_000).default(300_000),
      })
      .strict()
      .default({
        prompt: "Reply with only OK. /no_think",
        maxTokens: 16,
        timeoutMs: 300_000,
      }),
    coordinator: z
      .object({
        url: z.url(),
        region: z.string().min(1).max(128),
        networkTokenEnv: z.string().regex(ENVIRONMENT_NAME).optional(),
        maxConcurrency: z.number().int().positive().max(1_024).default(1),
      })
      .strict()
      .optional(),
    artifactsDirectory: z.string().min(1).max(4_096).optional(),
  })
  .strict()
  .superRefine((config, context) => {
    const ids = new Set<string>();
    const endpoints = new Set<string>();
    for (const [index, node] of config.nodes.entries()) {
      if (ids.has(node.id)) {
        context.addIssue({ code: "custom", message: "duplicate node id", path: ["nodes", index, "id"] });
      }
      ids.add(node.id);
      const endpoint = `${node.endpoint.host}:${node.endpoint.port}`;
      if (endpoints.has(endpoint)) {
        context.addIssue({ code: "custom", message: "duplicate stage endpoint", path: ["nodes", index, "endpoint"] });
      }
      endpoints.add(endpoint);
    }
    if (config.distribution.minimumStages > config.distribution.maximumStages) {
      context.addIssue({
        code: "custom",
        message: "minimumStages cannot exceed maximumStages",
        path: ["distribution", "minimumStages"],
      });
    }
    if (config.distribution.minimumStages > Math.min(config.nodes.length, 8)) {
      context.addIssue({
        code: "custom",
        message: "minimumStages exceeds available nodes",
        path: ["distribution", "minimumStages"],
      });
    }
    for (const [index, link] of config.links.entries()) {
      if (!ids.has(link.from) || !ids.has(link.to) || link.from === link.to) {
        context.addIssue({
          code: "custom",
          message: "link must reference two different configured nodes",
          path: ["links", index],
        });
      }
    }
  });

export type AutoDistributionConfig = z.infer<typeof autoDistributionConfigSchema>;

export interface CompiledModelProfile {
  schema: typeof MODEL_PROFILE_SCHEMA;
  source: {
    model: string;
    revision: string | null;
    snapshotCommit: string | null;
    snapshotIdentityUint64Hex: string;
    artifactIdentity: string;
    canonicalSource: string;
    canonicalRevision: string | null;
    format: string;
  };
  inspection: { architecture: string | null; calibrationRequired: boolean };
  compatibility: {
    selectiveSafetensors: boolean;
    requiresAdapter: boolean;
    adapterId: string | null;
    reasons: string[];
  };
  model: DistributedModelProfile;
  [key: string]: unknown;
}

export interface AutoDistributionCompilation {
  profile: CompiledModelProfile;
  request: RuntimePlanRequest;
  manifest: RuntimePipelineManifestV2;
  launch: PythonPipelineLaunchDescription;
  boundaries: number[];
}

export interface AutoDistributionRunResult extends AutoDistributionCompilation {
  health: Record<string, unknown>;
  canaryText: string;
  canaryMetrics: AutoDistributionCanaryMetrics;
  workerId: string | null;
}

export interface AutoDistributionCanaryMetrics {
  completionTokens: number;
  ttftMs: number;
  tpotMs: number;
  pipelineMs: number;
  measuredTokensPerSecond: number;
}

export interface AutoDistributionRunOptions {
  resolveManagedAgent?: (
    nodeId: string,
    launch: PythonPipelineLaunchDescription,
  ) => LaunchAgent | undefined;
}

export function parseAutoDistributionConfig(value: unknown): AutoDistributionConfig {
  return autoDistributionConfigSchema.parse(value);
}

/** Profile a normal HF/local checkpoint; no pre-split artifact is accepted or required. */
export async function profileCompatibleModel(
  config: AutoDistributionConfig,
  cwd = process.cwd(),
  environment: NodeJS.ProcessEnv = process.env,
): Promise<CompiledModelProfile> {
  const python = absoluteFrom(cwd, config.runtime.pythonExecutable);
  const pythonPath = absoluteFrom(cwd, config.runtime.pythonPath);
  const hfHome = absoluteFrom(cwd, config.runtime.hfHome);
  const args = ["-m", "distributed_runtime.profile", config.model.source];
  if (config.model.revision !== null) args.push("--revision", config.model.revision);
  const stdout = await runCaptured(python, args, cwd, {
    ...environment,
    PYTHONPATH: pythonPath,
    HF_HOME: hfHome,
    TOKENIZERS_PARALLELISM: "false",
  });
  return validateCompiledProfile(JSON.parse(stdout) as unknown);
}

export function compileAutoDistribution(
  configValue: AutoDistributionConfig,
  profileValue: CompiledModelProfile,
): AutoDistributionCompilation {
  const config = parseAutoDistributionConfig(configValue);
  const profile = validateCompiledProfile(profileValue);
  assertCompatibleProfile(profile);
  const topology = runtimeTopology(config);
  const workload = distributionWorkload(config);
  const request: RuntimePlanRequest = {
    model: structuredClone(profile.model),
    modelRevision: profile.source.artifactIdentity,
    tokenizerId: profile.source.artifactIdentity,
    topology,
    workload,
    allowLossyActivation: config.distribution.allowLossyActivation,
  };
  let manifest = buildRuntimePipelineManifest(request);
  if (manifest.plans.decode.stages.length < config.distribution.minimumStages) {
    const forced = exactStagePlan(request, config.distribution.minimumStages);
    request.phasePlans = { prefill: forced, decode: forced };
    manifest = buildRuntimePipelineManifest(request);
  }
  if (manifest.plans.decode.stages.length < 2) {
    throw new Error("automatic_distribution_requires_at_least_two_stages");
  }
  const cwd = process.cwd();
  const snapshotIdentity = BigInt(`0x${profile.source.snapshotIdentityUint64Hex}`).toString(10);
  const launch = compilePythonLaunchDescription(manifest, {
    apiEndpoint: config.runtime.apiEndpoint,
    returnEndpoint: config.runtime.returnEndpoint,
    returnBindHost: config.runtime.returnBindHost,
    runtimeModel: {
      source: config.model.source,
      revision: profile.source.snapshotCommit ?? config.model.revision,
      snapshotIdentity,
      artifactIdentity: profile.source.artifactIdentity,
      canonicalSource: profile.source.canonicalSource,
      canonicalRevision: profile.source.canonicalRevision,
    },
    publicModelName: config.model.publicName,
      pythonExecutable: config.runtime.stagePythonExecutable ?? absoluteFrom(cwd, config.runtime.pythonExecutable),
    threadsPerStage: config.runtime.threadsPerStage,
    connectTimeoutSeconds: config.runtime.connectTimeoutSeconds,
    batchWindowMs: config.workload.batchWindowMs,
    maxOutputTokens: config.runtime.maxOutputTokens,
  });
  return {
    profile,
    request,
    manifest,
    launch,
    boundaries: stageBoundaries(manifest),
  };
}

export async function writeAutoDistributionArtifacts(
  config: AutoDistributionConfig,
  compilation: AutoDistributionCompilation,
  cwd = process.cwd(),
): Promise<string> {
  const directory = resolve(
    cwd,
    config.artifactsDirectory ?? `runtime/auto-distribute/${config.model.publicName}`,
  );
  await mkdir(directory, { recursive: true });
  await Promise.all([
    writeJson(resolve(directory, "profile.json"), compilation.profile),
    writeJson(resolve(directory, "runtime-manifest.json"), compilation.manifest),
    writeJson(resolve(directory, "python-launch.json"), compilation.launch),
  ]);
  return directory;
}

export async function runAutoDistribution(
  configValue: AutoDistributionConfig,
  compilation: AutoDistributionCompilation,
  cwd = process.cwd(),
  environment: NodeJS.ProcessEnv = process.env,
  shutdownSignal?: AbortSignal,
  options: AutoDistributionRunOptions = {},
): Promise<AutoDistributionRunResult> {
  const config = parseAutoDistributionConfig(configValue);
  const agents = await createLaunchAgents(config, cwd, environment, compilation.launch, options);
  const supervisor = new PythonLaunchSupervisor(compilation.launch, {
    resolveAgent: (nodeId) => agents.get(nodeId),
    readinessTimeoutMs: config.runtime.readinessTimeoutMs,
  });
  let worker: WorkerAgent | null = null;
  let workerPromise: Promise<void> | null = null;
  let runtimeProxy: { host: string; port: number; close(): Promise<void> } | null = null;
  try {
    await supervisor.start();
    const rootProcess = compilation.launch.launchOrder.find((process) => process.kind === "root-engine");
    if (!rootProcess) throw new Error("distributed_root_process_is_missing");
    const rootAgent = agents.get(rootProcess.anchor.memberId);
    runtimeProxy = rootAgent?.createRuntimeProxy
      ? await rootAgent.createRuntimeProxy(rootProcess.apiEndpoint.port)
      : null;
    const apiBaseUrl = runtimeProxy
      ? `http://${runtimeProxy.host}:${runtimeProxy.port}`
      : `http://${config.runtime.apiAdvertiseHost}:${config.runtime.apiEndpoint.port}`;
    const health = await verifyRootHealth(apiBaseUrl, config, compilation);
    const canary = await runCanary(apiBaseUrl, config);
    if (config.coordinator) {
      const workerConfig = buildCellWorkerConfig(config, compilation, apiBaseUrl, canary.metrics);
      const token = optionalSecret(environment, config.coordinator.networkTokenEnv);
      worker = new WorkerAgent(workerConfig, {
        coordinatorUrl: config.coordinator.url,
        ...(token ? { networkToken: token } : {}),
        identity: {
          kind: "cell",
          id: automaticCellIdentity(config, compilation),
        },
      });
      workerPromise = worker.start();
      await waitForWorkerRegistration(worker, workerPromise, 30_000);
    }
    const result: AutoDistributionRunResult = {
      ...compilation,
      health,
      canaryText: canary.text,
      canaryMetrics: canary.metrics,
      workerId: worker?.workerId ?? null,
    };
    await writeRuntimeStatus(config, result, cwd, "running");
    await waitForShutdown(supervisor, workerPromise, shutdownSignal);
    return result;
  } catch (error) {
    await writeRuntimeFailure(config, error, supervisor.snapshot(), cwd).catch(() => undefined);
    throw error;
  } finally {
    if (worker) await worker.stop().catch(() => undefined);
    await supervisor.stop("auto_distribute_shutdown").catch(() => undefined);
    await runtimeProxy?.close().catch(() => undefined);
    await Promise.all(
      [...new Set(agents.values())].map((agent) => Promise.resolve(agent.close?.()).catch(() => undefined)),
    );
  }
}

function automaticCellIdentity(
  config: AutoDistributionConfig,
  compilation: AutoDistributionCompilation,
): string {
  const topology = config.nodes.map((node) => node.id).sort().join(",");
  const digest = createHash("sha256")
    .update(`${config.model.publicName}\n${compilation.profile.source.artifactIdentity}\n${topology}`)
    .digest("hex")
    .slice(0, 32);
  return `cell-${digest}`;
}

function validateCompiledProfile(value: unknown): CompiledModelProfile {
  if (!isRecord(value) || value.schema !== MODEL_PROFILE_SCHEMA) {
    throw new Error("unsupported_model_profile_schema");
  }
  const source = value.source;
  const compatibility = value.compatibility;
  const inspection = value.inspection;
  if (!isRecord(source) || !isRecord(compatibility) || !isRecord(inspection) || !isRecord(value.model)) {
    throw new Error("model_profile_is_incomplete");
  }
  if (
    typeof source.model !== "string" ||
    (source.revision !== null && typeof source.revision !== "string") ||
    (source.snapshotCommit !== null && typeof source.snapshotCommit !== "string") ||
    typeof source.snapshotIdentityUint64Hex !== "string" ||
    !/^[0-9a-f]{16}$/.test(source.snapshotIdentityUint64Hex) ||
    typeof source.artifactIdentity !== "string" ||
    !/^sha256:[0-9a-f]{64}$/.test(source.artifactIdentity) ||
    typeof source.canonicalSource !== "string" ||
    (source.canonicalRevision !== null && typeof source.canonicalRevision !== "string") ||
    typeof source.format !== "string" ||
    typeof compatibility.selectiveSafetensors !== "boolean" ||
    typeof compatibility.requiresAdapter !== "boolean" ||
    (compatibility.adapterId !== null && typeof compatibility.adapterId !== "string") ||
    !Array.isArray(compatibility.reasons) ||
    !compatibility.reasons.every((reason) => typeof reason === "string")
  ) {
    throw new Error("model_profile_fields_are_invalid");
  }
  return value as unknown as CompiledModelProfile;
}

function assertCompatibleProfile(profile: CompiledModelProfile): void {
  if (
    profile.source.format !== "safetensors" ||
    !profile.compatibility.selectiveSafetensors ||
    profile.compatibility.adapterId === null
  ) {
    throw new Error(
      `model_is_not_automatically_distributable:${profile.compatibility.reasons.join(";") || "unsupported_format_or_architecture"}`,
    );
  }
}

function runtimeTopology(config: AutoDistributionConfig): RuntimeTopology {
  const nodes: RuntimeTopology["nodes"] = config.nodes.map((node) => ({
    id: node.id,
    region: node.region,
    memoryBytes: node.memoryMiB * MIB,
    reserveBytes: node.reserveMiB * MIB,
    decodeScale: node.decodeScale,
    prefillScale: node.prefillScale,
    codecScale: node.codecScale,
    batchGain: 0.1,
    maxBatchSpeedup: 1.5,
    powerWatts: node.powerWatts,
    availability: node.availability,
    endpoint: { ...node.endpoint },
    backend: {
      engine: "python-transformers",
      version: "1",
      modelFormats: ["safetensors"],
      executionModes: ["layer-range"],
    },
    capabilities: {
      deviceKinds: ["cpu"],
      computeApis: ["torch"],
      weightDtypes: ["fp16", "bf16", "fp32"],
      activationCodecs: ["fp16", "int8"],
      features: ["layer-range", "kv-reuse", "kv-transfer"],
    },
  }));
  const overrides = new Map(config.links.map((link) => [`${link.from}\0${link.to}`, link]));
  const links: DirectedLinkProfile[] = [];
  for (const from of nodes) {
    for (const to of nodes) {
      if (from.id === to.id) continue;
      const override = overrides.get(`${from.id}\0${to.id}`);
      if (override) {
        links.push({ ...override });
      } else {
        const sameRegion = from.region === to.region;
        links.push({
          from: from.id,
          to: to.id,
          oneWayLatencyMs: sameRegion ? 1 : 35,
          jitterP95Ms: sameRegion ? 0.25 : 5,
          bandwidthMbps: sameRegion ? 1_000 : 100,
          lossRate: 0,
          availability: 0.999,
        });
      }
    }
  }
  return { nodes, links };
}

function distributionWorkload(config: AutoDistributionConfig): DistributionWorkload {
  return {
    ...config.workload,
    maxStages: Math.min(config.distribution.maximumStages, config.nodes.length, 8),
    maxQualityLoss: config.distribution.allowLossyActivation ? 1 : 0,
  };
}

/** Dynamic programming chooses exact contiguous boundaries without pre-split weights. */
function exactStagePlan(request: RuntimePlanRequest, count: number): DistributionPlan {
  const defaultManifest = buildRuntimePipelineManifest(request);
  const preferred = defaultManifest.plans.decode.stages.map((stage) => stage.anchor.memberId);
  const remaining = request.topology.nodes
    .filter((node) => !preferred.includes(node.id))
    .sort((left, right) => left.decodeScale - right.decodeScale || right.memoryBytes - left.memoryBytes)
    .map((node) => node.id);
  const orders = uniqueOrders([
    [...preferred, ...remaining].slice(0, count),
    request.topology.nodes.map((node) => node.id).slice(0, count),
    request.topology.nodes
      .slice()
      .sort((left, right) => left.decodeScale - right.decodeScale || right.memoryBytes - left.memoryBytes)
      .map((node) => node.id)
      .slice(0, count),
    request.topology.nodes
      .slice()
      .sort((left, right) => right.memoryBytes - right.reserveBytes - (left.memoryBytes - left.reserveBytes))
      .map((node) => node.id)
      .slice(0, count),
  ]);
  let best: { plan: DistributionPlan; score: number } | null = null;
  for (const order of orders) {
    const stages = balancedExactPlacement(request, order);
    if (!stages) continue;
    const plan: DistributionPlan = {
      algorithm: "auto-balanced-exact-stages",
      codec: "fp16",
      microBatchSize: Math.max(1, Math.min(1, request.workload.concurrentSequences)),
      prefillChunkTokens: Math.max(1, Math.min(64, request.workload.promptTokens)),
      stages,
    };
    const metrics = evaluateDistributionPlan(
      request.model,
      { nodes: planningNodes(request.topology), links: request.topology.links },
      request.workload,
      plan,
    );
    if (!metrics.feasible) continue;
    const score = metrics.tpotMs + metrics.ttftMs * 0.05;
    if (!best || score < best.score) best = { plan, score };
  }
  if (!best) throw new Error(`no_feasible_automatic_${count}_stage_pipeline`);
  return best.plan;
}

function planningNodes(topology: RuntimeTopology): ComputeNodeProfile[] {
  return topology.nodes.map((node) => ({
    id: node.id,
    region: node.region,
    memoryBytes: node.memoryBytes,
    reserveBytes: node.reserveBytes,
    decodeScale: node.decodeScale,
    prefillScale: node.prefillScale,
    codecScale: node.codecScale,
    batchGain: node.batchGain,
    maxBatchSpeedup: node.maxBatchSpeedup,
    powerWatts: node.powerWatts,
    availability: node.availability,
    ...(node.ramVram ? { ramVram: structuredClone(node.ramVram) } : {}),
  }));
}

function balancedExactPlacement(
  request: RuntimePlanRequest,
  order: string[],
): StagePlacement[] | null {
  const nodeById = new Map(request.topology.nodes.map((node) => [node.id, node]));
  const layers = request.model.layers.length;
  const stages = order.length;
  if (stages < 2 || stages > layers) return null;
  type Cell = { score: number; previous: number };
  const dp: Array<Map<number, Cell>> = Array.from({ length: stages + 1 }, () => new Map());
  dp[0]!.set(0, { score: 0, previous: -1 });
  for (let stageIndex = 0; stageIndex < stages; stageIndex += 1) {
    const node = nodeById.get(order[stageIndex]!);
    if (!node) return null;
    for (const [start, state] of dp[stageIndex]!) {
      const remainingStages = stages - stageIndex - 1;
      const maximumEnd = layers - remainingStages;
      for (let end = start + 1; end <= maximumEnd; end += 1) {
        const placement = { nodeId: node.id, layerStart: start, layerEnd: end };
        const memory = stageMemoryBytes(
          request.model,
          placement,
          request.workload,
          stageIndex === 0,
          stageIndex === stages - 1,
        );
        if (memory > node.memoryBytes - node.reserveBytes) continue;
        const work = request.model.layers
          .slice(start, end)
          .reduce((sum, layer) => sum + layer.decodeMsAtUnit, 0) * node.decodeScale;
        const score = Math.max(state.score, work);
        const existing = dp[stageIndex + 1]!.get(end);
        if (!existing || score < existing.score) {
          dp[stageIndex + 1]!.set(end, { score, previous: start });
        }
      }
    }
  }
  if (!dp[stages]!.has(layers)) return null;
  const result: StagePlacement[] = [];
  let end = layers;
  for (let stageIndex = stages; stageIndex > 0; stageIndex -= 1) {
    const cell = dp[stageIndex]!.get(end)!;
    result.unshift({
      nodeId: order[stageIndex - 1]!,
      layerStart: cell.previous,
      layerEnd: end,
    });
    end = cell.previous;
  }
  return result;
}

function stageBoundaries(manifest: RuntimePipelineManifestV2): number[] {
  const stages = manifest.plans.decode.stages;
  return [stages[0]!.layerStart, ...stages.map((stage) => stage.layerEnd)];
}

async function createLaunchAgents(
  config: AutoDistributionConfig,
  cwd: string,
  environment: NodeJS.ProcessEnv,
  launch: PythonPipelineLaunchDescription,
  options: AutoDistributionRunOptions,
): Promise<Map<string, LaunchAgent>> {
  const agents = new Map<string, LaunchAgent>();
  const local = new LocalProcessAgent({
    id: "auto-distribute-local",
    cwd,
    env: {
      PYTHONPATH: absoluteFrom(cwd, config.runtime.pythonPath),
      HF_HOME: absoluteFrom(cwd, config.runtime.hfHome),
      TOKENIZERS_PARALLELISM: "false",
    },
  });
  for (const node of config.nodes) {
    if (node.agent.kind === "local") {
      agents.set(node.id, local);
      continue;
    }
    if (node.agent.kind === "managed") {
      const agent = options.resolveManagedAgent?.(node.id, launch);
      if (!agent) throw new Error(`managed_launch_agent_is_unavailable:${node.id}`);
      agents.set(node.id, agent);
      continue;
    }
    const token = optionalSecret(environment, node.agent.authTokenEnv);
    const agent = new HttpLaunchAgent({
      endpoint: node.agent.endpoint,
      id: `auto-${node.id}`,
      requestTimeoutMs: node.agent.requestTimeoutMs,
      ...(token ? { authToken: token } : {}),
    });
    const health = await agent.health();
    if (health.nodeId !== null && health.nodeId !== node.id) {
      throw new Error(`launch_agent_node_mismatch:${node.id}:${health.nodeId}`);
    }
    agents.set(node.id, agent);
  }
  return agents;
}

async function verifyRootHealth(
  apiBaseUrl: string,
  config: AutoDistributionConfig,
  compilation: AutoDistributionCompilation,
): Promise<Record<string, unknown>> {
  const response = await fetch(`${apiBaseUrl}/health`, {
    signal: AbortSignal.timeout(30_000),
  });
  if (!response.ok) throw new Error(`distributed_root_health_http_${response.status}`);
  const value = (await response.json()) as unknown;
  if (!isRecord(value)) throw new Error("distributed_root_health_is_invalid");
  if (
    value.status !== "ready" ||
    value.model !== config.model.publicName ||
    value.artifact_identity !== compilation.profile.source.artifactIdentity ||
    value.stages !== compilation.boundaries.length - 1 ||
    JSON.stringify(value.boundaries) !== JSON.stringify(compilation.boundaries)
  ) {
    throw new Error("distributed_root_health_identity_mismatch");
  }
  return value;
}

async function runCanary(
  apiBaseUrl: string,
  config: AutoDistributionConfig,
): Promise<{ text: string; metrics: AutoDistributionCanaryMetrics }> {
  const response = await fetch(`${apiBaseUrl}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: config.model.publicName,
      messages: [{ role: "user", content: config.canary.prompt }],
      temperature: 0,
      max_tokens: config.canary.maxTokens,
      stream: false,
    }),
    signal: AbortSignal.timeout(config.canary.timeoutMs),
  });
  if (!response.ok) throw new Error(`distributed_canary_http_${response.status}`);
  const value = (await response.json()) as unknown;
  const text = isRecord(value) && Array.isArray(value.choices) && isRecord(value.choices[0]) &&
    isRecord(value.choices[0].message) && typeof value.choices[0].message.content === "string"
    ? value.choices[0].message.content.trim()
    : "";
  if (!text) throw new Error("distributed_canary_returned_empty_text");
  const usage = isRecord(value) ? value.usage : null;
  const distribution = isRecord(value) ? value.distribution_metrics : null;
  if (
    !isRecord(usage) ||
    !Number.isInteger(usage.completion_tokens) ||
    (usage.completion_tokens as number) < 1 ||
    !isRecord(distribution) ||
    typeof distribution.ttft_ms !== "number" ||
    !Number.isFinite(distribution.ttft_ms) ||
    distribution.ttft_ms < 0 ||
    typeof distribution.tpot_ms !== "number" ||
    !Number.isFinite(distribution.tpot_ms) ||
    distribution.tpot_ms < 0 ||
    typeof distribution.pipeline_ms !== "number" ||
    !Number.isFinite(distribution.pipeline_ms) ||
    distribution.pipeline_ms <= 0
  ) {
    throw new Error("distributed_canary_metrics_are_invalid");
  }
  const completionTokens = usage.completion_tokens as number;
  return {
    text,
    metrics: {
      completionTokens,
      ttftMs: distribution.ttft_ms,
      tpotMs: distribution.tpot_ms,
      pipelineMs: distribution.pipeline_ms,
      measuredTokensPerSecond:
        distribution.tpot_ms > 0
          ? 1_000 / distribution.tpot_ms
          : (completionTokens * 1_000) / distribution.pipeline_ms,
    },
  };
}

function buildCellWorkerConfig(
  config: AutoDistributionConfig,
  compilation: AutoDistributionCompilation,
  apiBaseUrl: string,
  canary: AutoDistributionCanaryMetrics,
): WorkerConfig {
  const stages = compilation.manifest.plans.decode.stages;
  const peakMiB = Math.max(512, Math.ceil(stages.reduce((sum, stage) => sum + stage.memoryBytes, 0) / MIB));
  const offeredMiB = Math.max(
    peakMiB,
    Math.floor(config.nodes.reduce((sum, node) => sum + node.memoryMiB - node.reserveMiB, 0)),
  );
  return workerConfigSchema.parse({
    region: config.coordinator!.region,
    capacityScope: "cell",
    offeredVramMb: offeredMiB,
    limits: {
      maxConcurrency: config.coordinator!.maxConcurrency,
      pauseWhenForeground: false,
    },
    adapter: {
      kind: "openai-compatible",
      model: config.model.publicName,
      baseUrl: apiBaseUrl,
      apiPathPrefix: "v1",
      requestTemperature: 0,
      allowedHosts: [],
    },
    deployment: {
      modelDigest: compilation.profile.source.artifactIdentity,
      peakVramMb: peakMiB,
      contextLimit: config.workload.contextTokens,
      tokensPerSecond: Math.max(0.001, canary.measuredTokensPerSecond),
      ttftMs: canary.ttftMs,
      internalPipeline: {
        stageCount: stages.length,
        boundaries: [...compilation.boundaries],
      },
    },
  });
}

async function waitForWorkerRegistration(
  worker: WorkerAgent,
  startPromise: Promise<void>,
  timeoutMs: number,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!worker.workerId) {
    if (Date.now() >= deadline) throw new Error("coordinator_worker_registration_timeout");
    await Promise.race([
      new Promise<void>((resolveDelay) => setTimeout(resolveDelay, 100)),
      startPromise.then(() => {
        throw new Error("coordinator_worker_stopped_before_registration");
      }),
    ]);
  }
}

async function waitForShutdown(
  supervisor: PythonLaunchSupervisor,
  workerPromise: Promise<void> | null,
  shutdownSignal?: AbortSignal,
): Promise<void> {
  let finishSignal: (() => void) | null = null;
  const signal = new Promise<void>((resolveSignal) => {
    finishSignal = resolveSignal;
  });
  const finish = () => finishSignal?.();
  process.once("SIGINT", finish);
  process.once("SIGTERM", finish);
  shutdownSignal?.addEventListener("abort", finish, { once: true });
  if (shutdownSignal?.aborted) finish();
  const terminal = supervisor.waitForTerminal().then((snapshot) => {
    if (snapshot.state === "failed") throw new Error(snapshot.failure ?? "distributed_pipeline_failed");
  });
  try {
    await Promise.race([
      signal,
      terminal,
      ...(workerPromise
        ? [workerPromise.then(() => { throw new Error("coordinator_worker_stopped"); })]
        : []),
    ]);
  } finally {
    process.removeListener("SIGINT", finish);
    process.removeListener("SIGTERM", finish);
    shutdownSignal?.removeEventListener("abort", finish);
  }
}

async function writeRuntimeStatus(
  config: AutoDistributionConfig,
  result: AutoDistributionRunResult,
  cwd: string,
  state: "running",
): Promise<void> {
  const directory = resolve(cwd, config.artifactsDirectory ?? `runtime/auto-distribute/${config.model.publicName}`);
  await mkdir(directory, { recursive: true });
  await writeJson(resolve(directory, "status.json"), {
    schema: "gdlp-auto-distribute-status/1",
    state,
    model: config.model.publicName,
    adapterId: result.profile.compatibility.adapterId,
    pipelineId: result.manifest.pipelineId,
    boundaries: result.boundaries,
    workerId: result.workerId,
    canaryText: result.canaryText,
    canaryMetrics: result.canaryMetrics,
    startedAt: new Date().toISOString(),
  });
}

async function writeRuntimeFailure(
  config: AutoDistributionConfig,
  error: unknown,
  supervisor: ReturnType<PythonLaunchSupervisor["snapshot"]>,
  cwd: string,
): Promise<void> {
  const directory = resolve(cwd, config.artifactsDirectory ?? `runtime/auto-distribute/${config.model.publicName}`);
  await mkdir(directory, { recursive: true });
  await writeJson(resolve(directory, "status.json"), {
    schema: "gdlp-auto-distribute-status/1",
    state: "failed",
    error: error instanceof Error ? error.message : String(error),
    supervisor,
    failedAt: new Date().toISOString(),
  });
}

function optionalSecret(environment: NodeJS.ProcessEnv, name: string | undefined): string | undefined {
  if (!name) return undefined;
  const value = environment[name]?.trim();
  if (!value) throw new Error(`required_secret_environment_is_missing:${name}`);
  return value;
}

function uniqueOrders(orders: string[][]): string[][] {
  const seen = new Set<string>();
  return orders.filter((order) => {
    const key = order.join("\0");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function absoluteFrom(cwd: string, path: string): string {
  return isAbsolute(path) ? path : resolve(cwd, path);
}

async function runCaptured(
  executable: string,
  args: string[],
  cwd: string,
  env: NodeJS.ProcessEnv,
): Promise<string> {
  return new Promise<string>((resolveRun, rejectRun) => {
    const child = spawn(executable, args, {
      cwd,
      env,
      shell: false,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
    child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
    child.once("error", rejectRun);
    child.once("close", (code) => {
      if (code !== 0) {
        rejectRun(new Error(`model_profiler_failed:${code}:${Buffer.concat(stderr).toString("utf8").trim()}`));
        return;
      }
      resolveRun(Buffer.concat(stdout).toString("utf8"));
    });
  });
}

async function writeJson(path: string, value: unknown): Promise<void> {
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function autoDistributionDigest(compilation: AutoDistributionCompilation): string {
  return `sha256:${createHash("sha256").update(JSON.stringify(compilation.launch)).digest("hex")}`;
}
