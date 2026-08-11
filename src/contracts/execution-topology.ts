import { z } from "zod";
import { sha256CanonicalEvidence } from "../core/json.js";
import type { NetworkExecutionTrace } from "./types.js";
import type { ExecutionReceipt } from "./execution-receipt.js";

export const EXECUTION_TOPOLOGY_SCHEMA = "mycellios-execution-topology/1" as const;
const sha256 = z.string().regex(/^sha256:[a-f0-9]{64}$/);

export const executionTopologySchema = z.object({
  schema: z.literal(EXECUTION_TOPOLOGY_SCHEMA),
  jobId: z.string().min(1).max(256),
  receiptId: sha256,
  traceDigest: sha256,
  classification: z.enum(["physical", "loopback", "unverified"]),
  stages: z.array(z.object({
    stageIndex: z.number().int().nonnegative(),
    alias: z.string().regex(/^stage-[0-9]+$/),
    region: z.string().min(1).max(64),
    deviceType: z.enum(["cpu", "gpu"]),
    backend: z.enum(["cpu", "cuda", "rocm", "directml", "mps", "xpu", "vulkan", "webgpu"]),
  }).strict()).max(256),
  boundaries: z.array(z.object({
    boundaryIndex: z.number().int().nonnegative(),
    fromStageIndex: z.number().int().nonnegative(),
    toStageIndex: z.number().int().nonnegative(),
    transport: z.enum(["direct", "relay", "local", "unobserved"]),
    physicalBoundary: z.boolean().nullable(),
  }).strict()).max(255),
  createdAt: z.number().int().nonnegative().safe(),
}).strict();

export type ExecutionTopology = z.infer<typeof executionTopologySchema>;

export function redactExecutionTopology(input: {
  trace: NetworkExecutionTrace;
  receipt: ExecutionReceipt;
  regionForWorker: (workerId: string) => string | null;
}): ExecutionTopology {
  if (input.trace.jobId !== input.receipt.jobId) throw new Error("execution_topology_job_mismatch");
  if (sha256CanonicalEvidence(input.trace) !== input.receipt.networkTraceDigest) throw new Error("execution_topology_trace_digest_mismatch");
  const stages = input.trace.stages.map((stage) => ({
    stageIndex: stage.stageIndex,
    alias: `stage-${stage.stageIndex}`,
    region: safeRegion(input.regionForWorker(stage.workerId ?? stage.deploymentOwnerWorkerId)),
    deviceType: stage.deviceType,
    backend: stage.backend,
  }));
  const indexes = new Set(stages.map((stage) => stage.stageIndex));
  if (indexes.size !== stages.length) throw new Error("execution_topology_stage_index_is_ambiguous");
  for (const boundary of input.trace.boundaries) {
    if (!indexes.has(boundary.fromStageIndex) || !indexes.has(boundary.toStageIndex)) {
      throw new Error("execution_topology_boundary_stage_is_missing");
    }
  }
  const classification = input.trace.boundaries.length === 0 || input.trace.boundaries.every((boundary) => boundary.transport === "local")
    ? "loopback" as const
    : input.trace.boundaries.every((boundary) => boundary.physicalBoundary === true && boundary.transport !== "unobserved")
      ? "physical" as const : "unverified" as const;
  return executionTopologySchema.parse({ schema: EXECUTION_TOPOLOGY_SCHEMA, jobId: input.receipt.jobId,
    receiptId: input.receipt.receiptId, traceDigest: input.receipt.networkTraceDigest, classification, stages,
    boundaries: input.trace.boundaries.map(({ boundaryIndex, fromStageIndex, toStageIndex, transport, physicalBoundary }) =>
      ({ boundaryIndex, fromStageIndex, toStageIndex, transport, physicalBoundary })), createdAt: input.receipt.completedAt });
}

function safeRegion(value: string | null): string {
  const region = value?.trim() ?? "";
  if (!region || region.length > 64 || !/^[\p{L}\p{N} ._-]+$/u.test(region)
    || /(?:https?:\/\/|[\\/]|(?:\d{1,3}\.){3}\d{1,3})/i.test(region)) return "undisclosed";
  return region;
}
