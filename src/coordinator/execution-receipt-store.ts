import { createPublicKey, type KeyLike } from "node:crypto";
import { executionReceiptSchema, signExecutionReceipt, type ExecutionReceipt, type ExecutionReceiptBody } from "../contracts/execution-receipt.js";
import type { MeshDatabase } from "../storage/database.js";
import { executionTopologySchema, redactExecutionTopology, type ExecutionTopology } from "../contracts/execution-topology.js";
import type { NetworkExecutionTrace } from "../contracts/types.js";

export class ExecutionReceiptStore {
  constructor(private readonly database: MeshDatabase, private readonly signing: { keyId: string; privateKey: KeyLike }) {}

  verificationKey(): { keyId: string; spki: string } {
    const publicKey = createPublicKey(this.signing.privateKey);
    if (publicKey.asymmetricKeyType !== "ed25519") throw new Error("execution_receipt_public_key_is_not_ed25519");
    return { keyId: this.signing.keyId, spki: publicKey.export({ format: "der", type: "spki" }).toString("base64url") };
  }

  record(body: ExecutionReceiptBody, topology?: { trace: NetworkExecutionTrace; regionForWorker: (workerId: string) => string | null }): ExecutionReceipt {
    const receipt = signExecutionReceipt(body, this.signing);
    return this.database.transaction(() => {
      const existing = this.forJob(body.jobId);
      if (existing) {
        if (existing.receiptId !== receipt.receiptId) throw new Error("execution_receipt_job_conflict");
        if (topology) this.recordTopology(receipt, topology);
        return existing;
      }
      this.database.raw.prepare(`INSERT INTO execution_receipts(job_id, receipt_id, key_id, receipt_json, created_at) VALUES (?, ?, ?, ?, ?)`)
        .run(body.jobId, receipt.receiptId, receipt.keyId, JSON.stringify(receipt), body.completedAt);
      if (topology) this.recordTopology(receipt, topology);
      return receipt;
    });
  }

  forJob(jobId: string): ExecutionReceipt | null {
    const row = this.database.raw.prepare("SELECT receipt_json FROM execution_receipts WHERE job_id = ?").get(jobId) as { receipt_json: string } | undefined;
    return row ? executionReceiptSchema.parse(JSON.parse(row.receipt_json)) : null;
  }

  topologyForJob(jobId: string): ExecutionTopology | null {
    const row = this.database.raw.prepare("SELECT topology_json FROM execution_topologies WHERE job_id = ?").get(jobId) as { topology_json: string } | undefined;
    return row ? executionTopologySchema.parse(JSON.parse(row.topology_json)) : null;
  }

  listRecent(limit = 20): ExecutionReceipt[] {
    const bounded = Math.max(1, Math.min(100, Math.trunc(limit)));
    const rows = this.database.raw.prepare("SELECT receipt_json FROM execution_receipts ORDER BY created_at DESC, job_id DESC LIMIT ?")
      .all(bounded) as Array<{ receipt_json: string }>;
    return rows.map((row) => executionReceiptSchema.parse(JSON.parse(row.receipt_json)));
  }

  private recordTopology(receipt: ExecutionReceipt, input: { trace: NetworkExecutionTrace; regionForWorker: (workerId: string) => string | null }): void {
    const topology = redactExecutionTopology({ receipt, ...input });
    const existing = this.topologyForJob(receipt.jobId);
    if (existing) {
      if (JSON.stringify(existing) !== JSON.stringify(topology)) throw new Error("execution_topology_job_conflict");
      return;
    }
    this.database.raw.prepare(`INSERT INTO execution_topologies(job_id, receipt_id, trace_digest, topology_json, created_at) VALUES (?, ?, ?, ?, ?)`)
      .run(receipt.jobId, receipt.receiptId, receipt.networkTraceDigest, JSON.stringify(topology), receipt.completedAt);
  }
}
