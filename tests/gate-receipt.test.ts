import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { GATE_RECEIPT_SCHEMA, gateReceiptSchema, type GateReceipt } from "../src/contracts/gate-receipt.js";
import { readGateReceipt, receiptMatches, writeGateReceiptAtomic } from "../src/program/gate-receipt-store.js";

const digest = `sha256:${"a".repeat(64)}` as const;
const receipt: GateReceipt = { schema: GATE_RECEIPT_SCHEMA, gate: "G0_BASELINE", phase: 0, status: "pass", sourceSha: "b".repeat(40), sourceIdentity: digest, inputDigest: digest, createdAt: "2026-08-09T00:00:00.000Z", target: "local", checks: [{ id: "baseline", kind: "automatic", status: "pass", detail: "ok", evidence: ["test"] }], hardware: [], blocker: null };

describe("gate receipt", () => {
  it("rejects unknown fields and inconsistent decisions", () => {
    expect(() => gateReceiptSchema.parse({ ...receipt, unexpected: true })).toThrow();
    expect(() => gateReceiptSchema.parse({ ...receipt, status: "blocked" })).toThrow();
  });

  it("writes atomically and validates on read", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "mycellios-gate-"));
    const file = path.join(directory, "nested", "receipt.json");
    await writeGateReceiptAtomic(file, receipt);
    expect(await readGateReceipt(file)).toEqual(receipt);
    expect((await readFile(file, "utf8")).endsWith("\n")).toBe(true);
  });

  it("invalidates stale SHA and digest", () => {
    expect(receiptMatches(receipt, receipt.sourceSha, digest)).toBe(true);
    expect(receiptMatches(receipt, "c".repeat(40), digest)).toBe(false);
    expect(receiptMatches(receipt, receipt.sourceSha, `sha256:${"d".repeat(64)}`)).toBe(false);
  });
});
