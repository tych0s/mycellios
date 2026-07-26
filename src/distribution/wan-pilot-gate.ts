import { z } from "zod";

const digest = z.string().regex(/^sha256:[0-9a-f]{64}$/);
const physicalIdentitySchema = z.object({
  schema: z.literal("gdlp-worker-physical-identity/1"),
  provider: z.enum(["gpu_cloud", "generic"]),
  providerMachineFingerprintSha256: digest,
  hostFingerprintSha256: digest,
  gpuFingerprintsSha256: z.array(digest).min(1).max(64),
  attestedAt: z.iso.datetime(),
}).strict();
const stageSchema = z.object({
  nodeId: z.string().min(1).max(128),
  stageIndex: z.number().int().nonnegative(),
  layerStart: z.number().int().nonnegative(),
  layerEnd: z.number().int().positive(),
  deviceType: z.enum(["cpu", "gpu"]),
  backend: z.string().min(1),
  fallback: z.boolean(),
}).passthrough();
const deploymentSchema = z.object({
  model: z.string().min(1),
  internalPipeline: z.object({
    stageCount: z.number().int().min(2).max(64),
    boundaries: z.array(z.number().int().nonnegative()).min(3).max(65),
  }).strict().optional(),
  execution: z.object({
    fallback: z.boolean(),
    stages: z.array(stageSchema).min(1).max(64).optional(),
  }).passthrough().optional(),
}).passthrough();
const workerSchema = z.object({
  id: z.string().min(1),
  status: z.string().min(1),
  connected: z.boolean(),
  executionNodeId: z.string().min(1).max(128).optional(),
  physicalIdentity: physicalIdentitySchema.optional(),
  deployments: z.array(deploymentSchema),
}).passthrough();
const snapshotSchema = z.object({
  capturedAt: z.iso.datetime(),
  workers: z.array(workerSchema),
  models: z.array(z.object({ id: z.string().min(1) }).passthrough()),
  jobs: z.array(z.object({
    model: z.string().min(1),
    status: z.string().min(1),
    workerId: z.string().min(1).nullable(),
  }).passthrough()),
}).passthrough();

export interface WanPilotGateOptions {
  model: string;
  minimumPhysicalNodes?: number;
  minimumGpuCloudNodes?: number;
}

export interface WanPilotGateReport {
  schema: "gdlp-wan-pilot-gate-report/1";
  scope: "wan-layer-pipeline-functional/1";
  passed: true;
  capturedAt: string;
  model: string;
  stageCount: number;
  physicalNodeCount: number;
  gpu_cloudNodeCount: number;
  gpuFingerprintCount: number;
  nodes: Array<{
    nodeId: string;
    provider: "gpu_cloud" | "generic";
    stageIndexes: number[];
  }>;
  limitations: string[];
}

export function verifyWanPilotSnapshot(
  value: unknown,
  options: WanPilotGateOptions,
): WanPilotGateReport {
  const snapshot = snapshotSchema.parse(value);
  const model = normalizedModel(options.model);
  const minimumPhysicalNodes = boundedMinimum(
    options.minimumPhysicalNodes ?? 2,
    "wan_pilot_minimum_physical_nodes_is_invalid",
  );
  const minimumGpuCloudNodes = boundedMinimum(
    options.minimumGpuCloudNodes ?? 2,
    "wan_pilot_minimum_gpu_cloud_nodes_is_invalid",
  );
  if (!snapshot.models.some((entry) => entry.id === model)) {
    throw new Error("wan_pilot_model_is_not_active");
  }
  const candidates = snapshot.workers.flatMap((worker) =>
    worker.deployments
      .filter((deployment) =>
        deployment.model === model
        && deployment.internalPipeline
        && deployment.execution?.stages
      )
      .map((deployment) => ({ worker, deployment })),
  );
  if (candidates.length !== 1) {
    throw new Error(
      candidates.length === 0
        ? "wan_pilot_pipeline_execution_evidence_is_missing"
        : "wan_pilot_pipeline_execution_evidence_is_ambiguous",
    );
  }
  const candidate = candidates[0]!;
  if (!snapshot.jobs.some((job) =>
    job.model === model
    && job.status === "completed"
    && job.workerId === candidate.worker.id
  )) {
    throw new Error("wan_pilot_pipeline_has_no_completed_job");
  }
  const deployment = candidate.deployment;
  const pipeline = deployment.internalPipeline!;
  const stages = [...deployment.execution!.stages!]
    .sort((left, right) => left.stageIndex - right.stageIndex);
  if (deployment.execution!.fallback || stages.some((stage) => stage.fallback)) {
    throw new Error("wan_pilot_fallback_execution_is_not_evidence");
  }
  if (stages.length !== pipeline.stageCount) {
    throw new Error("wan_pilot_stage_count_does_not_match_pipeline");
  }
  if (stages.some((stage) => stage.deviceType !== "gpu" || stage.backend === "cpu")) {
    throw new Error("wan_pilot_requires_gpu_execution_for_every_stage");
  }
  assertContiguousStages(stages, pipeline.boundaries);

  const executionWorkers = new Map(
    snapshot.workers
      .filter((worker) => worker.executionNodeId)
      .map((worker) => [worker.executionNodeId!, worker]),
  );
  const nodes = new Map<string, {
    nodeId: string;
    provider: "gpu_cloud" | "generic";
    providerFingerprint: string;
    hostFingerprint: string;
    gpuFingerprints: string[];
    stageIndexes: number[];
  }>();
  for (const stage of stages) {
    const worker = executionWorkers.get(stage.nodeId);
    if (!worker || !worker.connected || worker.status !== "online") {
      throw new Error(`wan_pilot_stage_worker_is_not_online:${stage.nodeId}`);
    }
    const identity = worker.physicalIdentity;
    if (!identity) {
      throw new Error(`wan_pilot_stage_worker_is_not_physically_attested:${stage.nodeId}`);
    }
    const existing = nodes.get(stage.nodeId);
    if (existing) {
      existing.stageIndexes.push(stage.stageIndex);
      continue;
    }
    nodes.set(stage.nodeId, {
      nodeId: stage.nodeId,
      provider: identity.provider,
      providerFingerprint: identity.providerMachineFingerprintSha256,
      hostFingerprint: identity.hostFingerprintSha256,
      gpuFingerprints: [...identity.gpuFingerprintsSha256],
      stageIndexes: [stage.stageIndex],
    });
  }
  const physicalNodes = [...nodes.values()];
  const providerFingerprints = new Set(physicalNodes.map((node) => node.providerFingerprint));
  const hostFingerprints = new Set(physicalNodes.map((node) => node.hostFingerprint));
  const gpuFingerprints = new Set(physicalNodes.flatMap((node) => node.gpuFingerprints));
  if (
    physicalNodes.length < minimumPhysicalNodes
    || providerFingerprints.size < minimumPhysicalNodes
    || hostFingerprints.size < minimumPhysicalNodes
    || gpuFingerprints.size < minimumPhysicalNodes
  ) {
    throw new Error("wan_pilot_physical_machine_fingerprints_are_not_unique");
  }
  const gpu_cloudNodeCount = physicalNodes.filter((node) => node.provider === "gpu_cloud").length;
  if (gpu_cloudNodeCount < minimumGpuCloudNodes) {
    throw new Error("wan_pilot_has_too_few_gpu_cloud_nodes");
  }

  return {
    schema: "gdlp-wan-pilot-gate-report/1",
    scope: "wan-layer-pipeline-functional/1",
    passed: true,
    capturedAt: snapshot.capturedAt,
    model,
    stageCount: stages.length,
    physicalNodeCount: physicalNodes.length,
    gpu_cloudNodeCount,
    gpuFingerprintCount: gpuFingerprints.size,
    nodes: physicalNodes.map(({ nodeId, provider, stageIndexes }) => ({
      nodeId,
      provider,
      stageIndexes: [...stageIndexes].sort((left, right) => left - right),
    })),
    limitations: [
      "This gate proves a completed, attested multi-machine GPU layer pipeline; it does not prove cross-host NCCL tensor parallelism.",
      "Coordinator-relayed WebSocket transport is functional evidence, not production throughput evidence.",
      "Provider-bound machine fingerprints detect duplicate pilot replicas but are not tamper-proof remote attestation.",
      "Token-level parity remains the responsibility of the separate sealed physical GPU campaign.",
    ],
  };
}

function assertContiguousStages(
  stages: Array<z.infer<typeof stageSchema>>,
  boundaries: number[],
): void {
  if (boundaries.length !== stages.length + 1) {
    throw new Error("wan_pilot_boundaries_do_not_match_stages");
  }
  const stageIndexes = new Set<number>();
  for (const [index, stage] of stages.entries()) {
    if (stageIndexes.has(stage.stageIndex) || stage.stageIndex !== index) {
      throw new Error("wan_pilot_stage_indexes_are_not_contiguous");
    }
    stageIndexes.add(stage.stageIndex);
    if (
      stage.layerStart !== boundaries[index]
      || stage.layerEnd !== boundaries[index + 1]
      || stage.layerEnd <= stage.layerStart
    ) {
      throw new Error("wan_pilot_layer_ranges_do_not_match_boundaries");
    }
  }
}

function normalizedModel(value: string): string {
  const normalized = value?.trim();
  if (!normalized) throw new Error("wan_pilot_model_is_required");
  return normalized;
}

function boundedMinimum(value: number, error: string): number {
  if (!Number.isSafeInteger(value) || value < 2 || value > 64) throw new Error(error);
  return value;
}
