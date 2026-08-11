import { z } from "zod";

import {
  DEPLOYMENT_CANARY_MAXIMUM_AGE_MS,
  validateDeploymentCanaryEvidence,
} from "../contracts/deployment-canary.js";
import type { NetworkExecutionTrace, WorkerPhysicalIdentity } from "../contracts/types.js";
import { sha256CanonicalEvidence } from "../core/json.js";
import { MeshDatabase } from "../storage/database.js";
import type { StoredWorker } from "../storage/store.js";
import { networkExecutionTraceSchema } from "../telemetry/network-execution-trace.js";

const sha256Schema = z.string().regex(/^sha256:[0-9a-f]{64}$/);
const identifierSchema = z.string().min(1).max(256);
const participantSchema = z.object({
  nodeId: identifierSchema,
  workerId: identifierSchema,
  physicalIdentityDigest: sha256Schema,
  canaryEvidenceIds: z.array(sha256Schema).min(1).max(64),
  stageCount: z.number().int().positive(),
}).strict();

export const physicalContributionEvidenceSchema = z.object({
  schema: z.literal("mycellios-physical-contribution-evidence/1"),
  id: sha256Schema,
  jobId: identifierSchema,
  executionReceiptId: sha256Schema,
  traceDigest: sha256Schema,
  participants: z.array(participantSchema).min(1).max(256),
  physicalBoundaryCount: z.number().int().nonnegative(),
  observedFrom: z.number().int().nonnegative(),
  observedUntil: z.number().int().nonnegative(),
  createdAt: z.number().int().nonnegative(),
}).strict();

export type PhysicalContributionEvidence = z.infer<typeof physicalContributionEvidenceSchema>;

/**
 * Converts coordinator-observed execution into durable economic evidence.
 * Hardware identity is already redacted by the worker contract; only its
 * content digest leaves this authority. Any ambiguity fails closed.
 */
export class PhysicalContributionEvidenceAuthority {
  constructor(private readonly database: MeshDatabase) {}

  certify(input: {
    jobId: string;
    executionReceiptId: string;
    trace: NetworkExecutionTrace;
    workers: readonly StoredWorker[];
    createdAt: number;
  }): PhysicalContributionEvidence {
    const trace = networkExecutionTraceSchema.parse(input.trace);
    const jobId = identifierSchema.parse(input.jobId);
    const executionReceiptId = sha256Schema.parse(input.executionReceiptId);
    const createdAt = z.number().int().nonnegative().parse(input.createdAt);
    if (trace.jobId !== jobId) throw new Error("economic_contribution_job_trace_mismatch");
    if (trace.observedUntil > createdAt) throw new Error("economic_contribution_created_before_observation");

    const workersById = new Map(input.workers.map((worker) => [worker.id, worker]));
    const participants = new Map<string, {
      nodeId: string; workerId: string; physicalIdentityDigest: string;
      canaryEvidenceIds: Set<string>; stageCount: number;
    }>();
    const machineFingerprints = new Map<string, string>();
    const hostFingerprints = new Map<string, string>();
    const gpuFingerprints = new Map<string, string>();

    for (const stage of trace.stages) {
      if (!stage.nodeId || !stage.workerId || stage.deviceType !== "gpu") {
        throw new Error("economic_contribution_stage_is_not_physical_gpu");
      }
      const worker = workersById.get(stage.workerId);
      if (!worker || worker.capabilities.distributedExecutor?.nodeId !== stage.nodeId) {
        throw new Error("economic_contribution_stage_worker_is_ambiguous");
      }
      const identity = worker.capabilities.distributedExecutor.physicalIdentity;
      if (!identity) throw new Error("economic_contribution_physical_identity_is_missing");
      assertDistinctPhysicalIdentity(stage.nodeId, identity, machineFingerprints, hostFingerprints, gpuFingerprints);

      const selected = trace.selectedRoute[stage.routeStageIndex];
      const owner = selected ? workersById.get(selected.workerId) : undefined;
      const deployment = owner?.capabilities.deployments.find((candidate) =>
        candidate.deploymentId === stage.deploymentId && candidate.modelDigest === stage.modelDigest
      );
      const canary = deployment?.canaryEvidence;
      if (!owner || deployment?.verificationState !== "verified" || !canary) {
        throw new Error("economic_contribution_verified_canary_is_missing");
      }
      validateDeploymentCanaryEvidence(canary);
      if (canary.workerId !== owner.id || canary.modelDigest !== stage.modelDigest) {
        throw new Error("economic_contribution_canary_binding_is_invalid");
      }
      const canaryAt = Date.parse(canary.observedAt);
      if (canaryAt > trace.observedUntil || trace.observedUntil - canaryAt > DEPLOYMENT_CANARY_MAXIMUM_AGE_MS) {
        throw new Error("economic_contribution_canary_is_stale");
      }

      const digest = sha256CanonicalEvidence(identity);
      const existing = participants.get(stage.nodeId);
      if (existing && (existing.workerId !== stage.workerId || existing.physicalIdentityDigest !== digest)) {
        throw new Error("economic_contribution_node_identity_changed_within_trace");
      }
      const participant = existing ?? {
        nodeId: stage.nodeId,
        workerId: stage.workerId,
        physicalIdentityDigest: digest,
        canaryEvidenceIds: new Set<string>(),
        stageCount: 0,
      };
      participant.canaryEvidenceIds.add(canary.evidenceId);
      participant.stageCount += 1;
      participants.set(stage.nodeId, participant);
    }

    for (const boundary of trace.boundaries.filter(({ physicalBoundary }) => physicalBoundary)) {
      if (
        !["direct", "relay"].includes(boundary.transport)
        || boundary.countersExclusive !== true
        || boundary.bytesSourceToDestination === null
        || boundary.bytesDestinationToSource === null
        || boundary.bytesSourceToDestination + boundary.bytesDestinationToSource <= 0
      ) throw new Error("economic_contribution_physical_boundary_is_unverified");
    }

    const body = {
      schema: "mycellios-physical-contribution-evidence/1" as const,
      jobId,
      executionReceiptId,
      traceDigest: sha256CanonicalEvidence(trace),
      participants: [...participants.values()].map((participant) => ({
        ...participant,
        canaryEvidenceIds: [...participant.canaryEvidenceIds].sort(),
      })).sort((left, right) => left.nodeId.localeCompare(right.nodeId)),
      physicalBoundaryCount: trace.physicalBoundaryCount ?? 0,
      observedFrom: trace.observedFrom,
      observedUntil: trace.observedUntil,
      createdAt,
    };
    const evidence = physicalContributionEvidenceSchema.parse({
      ...body,
      id: sha256CanonicalEvidence(body),
    });
    return this.database.transaction(() => {
      const existing = this.database.raw.prepare(
        "SELECT evidence_json FROM economic_contribution_evidence WHERE job_id = ? OR execution_receipt_id = ?",
      ).get(jobId, executionReceiptId) as { evidence_json: string } | undefined;
      if (existing) {
        const replay = physicalContributionEvidenceSchema.parse(JSON.parse(existing.evidence_json));
        if (replay.id !== evidence.id) throw new Error("economic_contribution_evidence_replay_conflict");
        return replay;
      }
      this.database.raw.prepare(`
        INSERT INTO economic_contribution_evidence(id, job_id, execution_receipt_id, trace_digest, evidence_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
      `).run(evidence.id, jobId, executionReceiptId, evidence.traceDigest, JSON.stringify(evidence), createdAt);
      return evidence;
    });
  }

  read(id: string): PhysicalContributionEvidence | null {
    const row = this.database.raw.prepare(
      "SELECT evidence_json FROM economic_contribution_evidence WHERE id = ?",
    ).get(sha256Schema.parse(id)) as { evidence_json: string } | undefined;
    return row ? physicalContributionEvidenceSchema.parse(JSON.parse(row.evidence_json)) : null;
  }
}

function assertDistinctPhysicalIdentity(
  nodeId: string,
  identity: WorkerPhysicalIdentity,
  machines: Map<string, string>,
  hosts: Map<string, string>,
  gpus: Map<string, string>,
): void {
  const claims = [
    [identity.providerMachineFingerprintSha256, machines],
    [identity.hostFingerprintSha256, hosts],
    ...identity.gpuFingerprintsSha256.map((fingerprint) => [fingerprint, gpus] as const),
  ] as const;
  for (const [fingerprint, owners] of claims) {
    if (!/^sha256:[0-9a-f]{64}$/.test(fingerprint)) throw new Error("economic_contribution_physical_fingerprint_is_invalid");
    const owner = owners.get(fingerprint);
    if (owner && owner !== nodeId) throw new Error("economic_contribution_collocated_identity_detected");
    owners.set(fingerprint, nodeId);
  }
}
