import { generateKeyPairSync } from "node:crypto";
import { describe, expect, it } from "vitest";
import { signExecutionReceipt, verifyExecutionReceipt } from "../src/contracts/execution-receipt.js";
import { ExecutionReceiptStore } from "../src/coordinator/execution-receipt-store.js";
import { MeshDatabase } from "../src/storage/database.js";
import { MeshStore } from "../src/storage/store.js";

const sha = (value: string) => `sha256:${value.repeat(64)}` as const;

describe("execution receipt", () => {
  it("signs a redacted execution/recovery identity and rejects tampering", () => {
    const keys = generateKeyPairSync("ed25519");
    const receipt = signExecutionReceipt({
      schema: "mycellios-execution-receipt/1",
      jobId: "job-1",
      modelIdHash: sha("a"),
      routeClass: "pipeline",
      metrics: { inputTokens: 10, outputTokens: 4, ttftMs: 20, activeMs: 80 },
      networkTraceDigest: sha("b"),
      recovery: { mode: "deterministic-prefix-replay", attempts: 2, replayedTokenEvents: 3 },
      privacy: { trust: "trusted-only", boundary: "pinned-edges", pinnedIdentityHashes: [sha("c")] },
      completedAt: 1_000,
    }, { keyId: "execution-key-1", privateKey: keys.privateKey });
    const pinned = { keyId: "execution-key-1", spki: keys.publicKey.export({ format: "der", type: "spki" }).toString("base64url") };
    expect(verifyExecutionReceipt(receipt, pinned)).toEqual(receipt);
    for (const field of ["prompt", "messages", "tokenText", "workerId", "nodeId", "ipAddress", "path"]) {
      expect(receipt).not.toHaveProperty(field);
    }
    expect(JSON.stringify(receipt)).not.toContain("private prompt contents");
    expect(() => verifyExecutionReceipt({ ...receipt, recovery: { ...receipt.recovery, mode: "none" } }, pinned))
      .toThrow("execution_receipt_identity_mismatch");
    expect(() => verifyExecutionReceipt({ ...receipt, privacy: { ...receipt.privacy, trust: "default" } }, pinned))
      .toThrow("execution_receipt_identity_mismatch");
    expect(() => verifyExecutionReceipt(receipt, { ...pinned, keyId: "other" })).toThrow("execution_receipt_key_id_mismatch");
  });

  it("persists identical replay idempotently and rejects a changed job receipt", () => {
    const keys = generateKeyPairSync("ed25519");
    const database = new MeshDatabase(":memory:");
    const jobs = new MeshStore(database);
    jobs.createJob({ id: "job-durable", sessionId: "session", model: "qwen", workloadClass: "interactive", deadlineAt: 2_000 });
    const store = new ExecutionReceiptStore(database, { keyId: "key", privateKey: keys.privateKey });
    const body = { schema: "mycellios-execution-receipt/1" as const, jobId: "job-durable", modelIdHash: sha("a"), routeClass: "replica" as const,
      metrics: { inputTokens: 2, outputTokens: 1, ttftMs: 10, activeMs: 20 }, networkTraceDigest: sha("b"),
      recovery: { mode: "none" as const, attempts: 1, replayedTokenEvents: 0 },
      privacy: { trust: "default" as const, boundary: "trusted-edges" as const, pinnedIdentityHashes: [] }, completedAt: 1_000 };
    expect(store.record(body)).toEqual(store.record(body));
    expect(store.forJob("job-durable")).toEqual(store.record(body));
    expect(() => store.record({ ...body, metrics: { ...body.metrics, outputTokens: 2 } })).toThrow("execution_receipt_job_conflict");
    database.close();
  });
});
