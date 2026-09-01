import { z } from "zod";
import { gateReceiptSchema } from "../contracts/gate-receipt.js";
import { importedGateEvidenceDocumentSchema } from "./physical-gate-evidence.js";

const programSchema = z.object({
  schema: z.literal("mycellios-gate-program/1"),
  gates: z.array(z.object({
    id: z.string(), phase: z.number().int(), inputs: z.array(z.string()),
    checks: z.array(z.object({ id: z.string(), kind: z.enum(["automatic", "physical", "external"]) }).strict()),
  }).strict()),
}).strict();

const ARTIFACT_BOUND_EXTERNAL_CHECKS = new Set([
  "G0_BASELINE/native_builders",
  "G1_ELECTRON_ZERO/external_rollback",
  "G7_FINAL_RELEASE/pre_environment_and_rollback",
  "G7_FINAL_RELEASE/release_authorization",
]);

export interface FinalGateReceiptValidationInput {
  receipts: unknown[];
  program: unknown;
  importedEvidence: unknown;
  sourceSha: string;
  sourceIdentity: string;
  requirePassing?: boolean;
}

export function validateFinalGateReceiptSet(input: FinalGateReceiptValidationInput): {
  gates: number; checks: number; importedClaims: number; artifactDigests: number;
} {
  const program = programSchema.parse(input.program);
  const receipts = input.receipts.map((receipt) => gateReceiptSchema.parse(receipt));
  const imported = importedGateEvidenceDocumentSchema.parse(input.importedEvidence).evidence;
  const receiptByGate = new Map(receipts.map((receipt) => [receipt.gate, receipt]));
  if (receiptByGate.size !== receipts.length) throw new Error("final_gate_receipts_duplicate_gate");
  if (receipts.length !== program.gates.length) throw new Error("final_gate_receipts_incomplete");
  const targets = new Set(receipts.map((receipt) => receipt.target));
  if (targets.size !== 1) throw new Error("final_gate_receipts_mixed_targets");
  if (input.requirePassing && targets.has("local")) throw new Error("final_gate_receipts_release_cannot_be_local");

  const claimedArtifacts = new Set<string>();
  let checkCount = 0;
  let importedClaims = 0;
  for (const gate of program.gates) {
    const receipt = receiptByGate.get(gate.id);
    if (!receipt) throw new Error(`final_gate_receipt_missing:${gate.id}`);
    if (receipt.phase !== gate.phase) throw new Error(`final_gate_receipt_phase_mismatch:${gate.id}`);
    if (receipt.sourceSha !== input.sourceSha) throw new Error(`final_gate_receipt_source_sha_mismatch:${gate.id}`);
    if (receipt.sourceIdentity !== input.sourceIdentity) throw new Error(`final_gate_receipt_source_identity_mismatch:${gate.id}`);
    if (input.requirePassing && receipt.status !== "pass") throw new Error(`final_gate_receipt_not_passing:${gate.id}:${receipt.status}`);
    const expected = new Map(gate.checks.map((check) => [check.id, check.kind]));
    const actual = new Map(receipt.checks.map((check) => [check.id, check.kind]));
    if (expected.size !== gate.checks.length || actual.size !== receipt.checks.length) throw new Error(`final_gate_receipt_duplicate_check:${gate.id}`);
    if (expected.size !== actual.size || [...expected].some(([id, kind]) => actual.get(id) !== kind)) {
      throw new Error(`final_gate_receipt_check_set_mismatch:${gate.id}`);
    }
    checkCount += receipt.checks.length;
    for (const check of receipt.checks) {
      if (check.kind === "automatic" || check.status === "blocked") continue;
      const key = `${gate.id}/${check.id}`;
      const evidence = imported.find((entry) => entry.gate === gate.id && entry.check === check.id && entry.kind === check.kind);
      if (!evidence) throw new Error(`final_gate_imported_evidence_missing:${key}`);
      if (evidence.status !== check.status) throw new Error(`final_gate_imported_status_mismatch:${key}`);
      if (evidence.sourceSha !== input.sourceSha) throw new Error(`final_gate_imported_source_sha_mismatch:${key}`);
      if (evidence.sourceIdentity !== input.sourceIdentity) throw new Error(`final_gate_imported_source_identity_mismatch:${key}`);
      const expectedRows = [`source-sha:${input.sourceSha}`, `source-identity:${input.sourceIdentity}`,
        ...evidence.artifactDigests, ...evidence.evidence];
      if (expectedRows.some((row) => !check.evidence.includes(row))) throw new Error(`final_gate_receipt_dropped_imported_evidence:${key}`);
      if (check.status === "pass" && (check.kind === "physical" || ARTIFACT_BOUND_EXTERNAL_CHECKS.has(key)) && evidence.artifactDigests.length === 0) {
        throw new Error(`final_gate_claim_has_no_artifact_digest:${key}`);
      }
      if (check.status === "pass") for (const digest of evidence.artifactDigests) claimedArtifacts.add(digest);
      importedClaims += 1;
    }
  }

  if (input.requirePassing) {
    const release = imported.find((entry) => entry.gate === "G7_FINAL_RELEASE" && entry.check === "release_physical_matrix" && entry.kind === "physical");
    if (!release || release.status !== "pass") throw new Error("final_gate_release_matrix_missing");
    const releaseArtifacts = new Set(release.artifactDigests);
    const omitted = [...claimedArtifacts].filter((digest) => !releaseArtifacts.has(digest));
    if (omitted.length > 0) throw new Error(`final_gate_release_matrix_omits_artifacts:${omitted.join(",")}`);
  }
  return { gates: receipts.length, checks: checkCount, importedClaims, artifactDigests: claimedArtifacts.size };
}
