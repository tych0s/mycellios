import { describe, expect, it } from "vitest";
import { importedGateEvidenceResult, parseImportedGateEvidence } from "../scripts/gate-program.js";

const base = { gate: "G1_ELECTRON_ZERO", check: "native_lifecycle_matrix", kind: "physical" as const,
  observedAt: "2026-08-10T00:00:00.000Z", detail: "physical matrix", evidence: ["receipt.json"],
  sourceIdentity: `sha256:${"d".repeat(64)}` };

describe("imported gate evidence identity", () => {
  it("rejects a passing receipt from another source SHA", () => {
    const imported = parseImportedGateEvidence({ schema: "mycellios-gate-imported-evidence/1", evidence: [{ ...base,
      status: "pass", sourceSha: "a".repeat(40), artifactDigests: [`sha256:${"b".repeat(64)}`] }] }).evidence[0];
    expect(importedGateEvidenceResult({ id: base.check, kind: "physical" }, imported, "c".repeat(40))).toMatchObject({
      status: "blocked", detail: expect.stringContaining("not current source"),
    });
  });

  it("requires artifact digests for passing physical evidence", () => {
    expect(() => parseImportedGateEvidence({ schema: "mycellios-gate-imported-evidence/1", evidence: [{ ...base,
      status: "pass", sourceSha: "a".repeat(40), artifactDigests: [] }] })).toThrow("Passing physical evidence must bind artifact digests");
  });

  it("blocks evidence from a different source identity even when the SHA matches", () => {
    const imported = parseImportedGateEvidence({ schema: "mycellios-gate-imported-evidence/1", evidence: [{ ...base,
      status: "pass", sourceSha: "a".repeat(40), artifactDigests: [`sha256:${"b".repeat(64)}`] }] }).evidence[0];
    expect(importedGateEvidenceResult({ id: base.check, kind: "physical" }, imported, "a".repeat(40), `sha256:${"e".repeat(64)}`))
      .toMatchObject({ status: "blocked", detail: expect.stringContaining("source identity") });
  });

  it("accepts an explicit unbound blocker without turning it into a pass", () => {
    const imported = parseImportedGateEvidence({ schema: "mycellios-gate-imported-evidence/1", evidence: [{ ...base,
      status: "blocked", sourceSha: null, sourceIdentity: null, artifactDigests: [] }] }).evidence[0];
    expect(importedGateEvidenceResult({ id: base.check, kind: "physical" }, imported, "c".repeat(40))).toMatchObject({ status: "blocked" });
  });
});
