import { z } from "zod";
import { verifyPhysicalGateEvidence, type PhysicalGateEvidence } from "../contracts/physical-gate-evidence.js";

export const physicalEvidenceTrustSchema = z.object({ schema: z.literal("mycellios-physical-evidence-trust/1"),
  keys: z.array(z.object({ keyId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/), publicKeySpki: z.string().min(40).max(256).regex(/^[A-Za-z0-9_-]+$/),
    status: z.enum(["active", "revoked"]) }).strict()).max(32).refine((keys) => new Set(keys.map((key) => key.keyId)).size === keys.length) }).strict();

export const importedGateEvidenceDocumentSchema = z.object({ schema: z.literal("mycellios-gate-imported-evidence/1"), evidence: z.array(z.object({
  gate: z.string(), check: z.string(), kind: z.enum(["physical", "external"]), status: z.enum(["pass", "fail", "blocked"]), detail: z.string().min(1),
  observedAt: z.string().datetime(), sourceSha: z.string().regex(/^[a-f0-9]{40}$/).nullable(),
  sourceIdentity: z.string().regex(/^sha256:[a-f0-9]{64}$/).nullable(),
  artifactDigests: z.array(z.string().regex(/^sha256:[a-f0-9]{64}$/)), evidence: z.array(z.string().min(1)).min(1),
}).strict().superRefine((entry, context) => {
  if (entry.status !== "blocked" && (entry.sourceSha === null || entry.sourceIdentity === null)) {
    context.addIssue({ code: "custom", path: ["sourceIdentity"], message: "Pass/fail evidence must bind source SHA and source identity" });
  }
})) }).strict();

type ImportedDocument = z.infer<typeof importedGateEvidenceDocumentSchema>;

export function importPhysicalGateEvidence(input: { receipts: unknown[]; trust: unknown; sourceSha: string; sourceIdentity: string; existing: unknown }): ImportedDocument {
  if (!/^[a-f0-9]{40}$/.test(input.sourceSha)) throw new Error("physical_gate_import_source_sha_is_invalid");
  if (!/^sha256:[a-f0-9]{64}$/.test(input.sourceIdentity)) throw new Error("physical_gate_import_source_identity_is_invalid");
  const trust = physicalEvidenceTrustSchema.parse(input.trust);
  const trustedKeys = new Map(trust.keys.filter((key) => key.status === "active").map((key) => [key.keyId, key.publicKeySpki]));
  const receipts = input.receipts.map((receipt) => verifyPhysicalGateEvidence(receipt, trustedKeys));
  if (receipts.length === 0) throw new Error("physical_gate_import_requires_receipts");
  for (const receipt of receipts) if (receipt.sourceSha !== input.sourceSha) throw new Error("physical_gate_import_source_sha_mismatch");
  for (const receipt of receipts) if (receipt.sourceId !== input.sourceIdentity) throw new Error("physical_gate_import_source_identity_mismatch");
  const existing = importedGateEvidenceDocumentSchema.parse(input.existing);
  const groups = new Map<string, PhysicalGateEvidence[]>();
  for (const receipt of receipts) { const key = `${receipt.gate}:${receipt.check}`; const group = groups.get(key) ?? []; group.push(receipt); groups.set(key, group); }
  const replacements = [...groups.values()].map(buildImportEntry);
  const replaced = new Set(replacements.map((entry) => `${entry.gate}:${entry.check}:${entry.kind}`));
  return importedGateEvidenceDocumentSchema.parse({ schema: existing.schema,
    evidence: [...existing.evidence.filter((entry) => !replaced.has(`${entry.gate}:${entry.check}:${entry.kind}`)), ...replacements]
      .sort((left, right) => `${left.gate}:${left.check}`.localeCompare(`${right.gate}:${right.check}`)) });
}

function buildImportEntry(receipts: PhysicalGateEvidence[]) {
  const first = receipts[0]!;
  if (receipts.some((receipt) => receipt.gate !== first.gate || receipt.check !== first.check || receipt.sourceSha !== first.sourceSha || receipt.sourceId !== first.sourceId)) {
    throw new Error("physical_gate_receipts_are_not_one_candidate");
  }
  const failed = receipts.some((receipt) => receipt.status === "fail");
  if (!failed) {
    const missing = missingCoverage(first.check, receipts);
    if (missing.length > 0) throw new Error(`physical_gate_coverage_is_incomplete:${first.check}:${missing.join(",")}`);
  }
  const artifactDigests = [...new Set(receipts.flatMap((receipt) => receipt.artifactDigests))].sort();
  return { gate: first.gate, check: first.check, kind: "physical" as const, status: failed ? "fail" as const : "pass" as const,
    detail: failed ? `${receipts.length} signed physical receipt(s) include a failed assertion`
      : `${receipts.length} signed physical receipt(s) satisfy ${first.check} coverage`,
    observedAt: receipts.map((receipt) => receipt.completedAt).sort().at(-1)!, sourceSha: first.sourceSha,
    sourceIdentity: first.sourceId, artifactDigests,
    evidence: receipts.map((receipt) => `physical-receipt:${receipt.id}:${receipt.receiptDigest}`).sort() };
}

function missingCoverage(check: string, receipts: PhysicalGateEvidence[]): string[] {
  const platforms = new Set(receipts.flatMap((receipt) => receipt.coverage.platforms.map((entry) => `${entry.os}-${entry.arch}`)));
  const scopes = new Set(receipts.flatMap((receipt) => receipt.coverage.networkScopes));
  const scenarios = new Set(receipts.flatMap((receipt) => receipt.coverage.scenarios));
  const models = new Set(receipts.flatMap((receipt) => receipt.coverage.models.map((model) => model.toLowerCase())));
  const hardware = new Set(receipts.flatMap((receipt) => receipt.hardware.map((entry) => entry.fingerprint)));
  const missing: string[] = [];
  const requireValues = (label: string, actual: Set<string>, required: string[]) => { for (const value of required) if (!actual.has(value)) missing.push(`${label}:${value}`); };
  if (check === "physical_hardware") { if (hardware.size === 0) missing.push("hardware:any"); }
  if (check === "native_lifecycle_matrix" || check === "release_physical_matrix") {
    requireValues("platform", platforms, ["linux-x64", "win32-x64", "darwin-arm64"]);
    requireValues("scenario", scenarios, ["install", "pairing", "canary", "web-close", "reboot", "ready", "update", "rollback", "revoke", "uninstall-preserve", "uninstall-purge"]);
  }
  if (check === "qwen3_physical_certification") {
    if (![...models].some((model) => model.includes("qwen3"))) missing.push("model:qwen3");
    requireValues("scenario", scenarios, ["canary", "inference"]);
  }
  if (check === "two_host_exact_execution") {
    if (hardware.size < 2) missing.push("hardware:two-distinct"); requireValues("network", scopes, ["lan"]);
    requireValues("scenario", scenarios, ["local-complete", "split"]);
  }
  if (check === "physical_conveyor_ab") {
    if (hardware.size < 2) missing.push("hardware:two-distinct"); requireValues("network", scopes, ["same-host", "lan", "multi-site"]);
    requireValues("scenario", scenarios, ["direct", "relay", "w1", "wgt1", "interleaved-ab"]);
  }
  if (check === "physical_fault_campaign") requireValues("scenario", scenarios,
    ["sigkill", "freeze", "packet-loss", "relay-loss", "corrupt-artifact", "coordinator-restart", "power-loss"]);
  return missing;
}
