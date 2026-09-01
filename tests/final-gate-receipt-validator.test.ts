import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { GATE_RECEIPT_SCHEMA } from "../src/contracts/gate-receipt.js";
import { validateFinalGateReceiptSet } from "../src/program/final-gate-receipt-validator.js";

const program = JSON.parse(readFileSync("config/final-web-native-program.json", "utf8")) as {
  gates: Array<{ id: string; phase: number; checks: Array<{ id: string; kind: "automatic" | "physical" | "external" }> }>;
};
const sourceSha = "a".repeat(40), sourceIdentity = `sha256:${"b".repeat(64)}`, artifact = `sha256:${"c".repeat(64)}`;

function fixture() {
  const evidence = program.gates.flatMap((gate) => gate.checks.filter((check) => check.kind !== "automatic").map((check) => ({
    gate: gate.id, check: check.id, kind: check.kind, status: "pass" as const, detail: "verified candidate evidence",
    observedAt: "2026-08-11T00:00:00.000Z", sourceSha, sourceIdentity, artifactDigests: [artifact], evidence: [`receipt:${gate.id}:${check.id}`],
  })));
  const receipts = program.gates.map((gate) => ({
    schema: GATE_RECEIPT_SCHEMA, gate: gate.id, phase: gate.phase, status: "pass", sourceSha, sourceIdentity,
    inputDigest: `sha256:${"d".repeat(64)}`, createdAt: "2026-08-11T00:00:00.000Z", target: "release",
    checks: gate.checks.map((check) => {
      const imported = evidence.find((entry) => entry.gate === gate.id && entry.check === check.id);
      return { ...check, status: "pass", detail: "verified",
        evidence: imported ? [`source-sha:${sourceSha}`, `source-identity:${sourceIdentity}`, artifact, ...imported.evidence] : ["automatic:test"] };
    }), hardware: [], blocker: null,
  }));
  return { receipts, importedEvidence: { schema: "mycellios-gate-imported-evidence/1", evidence } };
}

describe("final gate receipt validator", () => {
  it("accepts one coherent passing G0-G7 candidate", () => {
    const value = fixture();
    expect(validateFinalGateReceiptSet({ ...value, program: { schema: "mycellios-gate-program/1", ...program }, sourceSha, sourceIdentity, requirePassing: true }))
      .toMatchObject({ gates: 8, importedClaims: value.importedEvidence.evidence.length, artifactDigests: 1 });
  });

  it("rejects a receipt or imported claim from another source tree", () => {
    const value = fixture();
    expect(() => validateFinalGateReceiptSet({ ...value, receipts: value.receipts.map((receipt, index) => index === 3 ? { ...receipt, sourceIdentity: `sha256:${"e".repeat(64)}` } : receipt),
      program: { schema: "mycellios-gate-program/1", ...program }, sourceSha, sourceIdentity })).toThrow("source_identity_mismatch");
    expect(() => validateFinalGateReceiptSet({ ...value, importedEvidence: { ...value.importedEvidence, evidence: value.importedEvidence.evidence.map((entry, index) => index === 2 ? { ...entry, sourceIdentity: `sha256:${"e".repeat(64)}` } : entry) },
      program: { schema: "mycellios-gate-program/1", ...program }, sourceSha, sourceIdentity })).toThrow("imported_source_identity_mismatch");
  });

  it("rejects dropped evidence and a release matrix that omits an earlier artifact", () => {
    const value = fixture();
    const physical = value.importedEvidence.evidence.find((entry) => entry.kind === "physical")!;
    const dropped = value.receipts.map((receipt) => receipt.gate !== physical.gate ? receipt : { ...receipt,
      checks: receipt.checks.map((check) => check.id === physical.check ? { ...check, evidence: ["dropped"] } : check) });
    expect(() => validateFinalGateReceiptSet({ ...value, receipts: dropped, program: { schema: "mycellios-gate-program/1", ...program }, sourceSha, sourceIdentity }))
      .toThrow("dropped_imported_evidence");

    const other = `sha256:${"f".repeat(64)}`;
    const importedEvidence = { ...value.importedEvidence, evidence: value.importedEvidence.evidence.map((entry) =>
      entry.gate === "G3_AUTOMATIC_EXECUTION" && entry.check === "two_host_exact_execution" ? { ...entry, artifactDigests: [other] } : entry) };
    const receipts = value.receipts.map((receipt) => receipt.gate !== "G3_AUTOMATIC_EXECUTION" ? receipt : { ...receipt,
      checks: receipt.checks.map((check) => check.id !== "two_host_exact_execution" ? check : { ...check, evidence: [
        `source-sha:${sourceSha}`, `source-identity:${sourceIdentity}`, other, `receipt:${receipt.gate}:${check.id}`,
      ] }) });
    expect(() => validateFinalGateReceiptSet({ receipts, importedEvidence, program: { schema: "mycellios-gate-program/1", ...program }, sourceSha, sourceIdentity, requirePassing: true }))
      .toThrow("release_matrix_omits_artifacts");
  });
});
