import { createHash, createPublicKey, sign, verify, type KeyObject } from "node:crypto";
import { z } from "zod";
import { canonicalEvidenceJson } from "../core/json.js";

export const PHYSICAL_GATE_EVIDENCE_SCHEMA = "mycellios-physical-gate-evidence/1" as const;
const sha256 = z.string().regex(/^sha256:[a-f0-9]{64}$/);
const platform = z.object({ os: z.enum(["linux", "darwin", "win32"]), arch: z.enum(["x64", "arm64"]) }).strict();

const physicalGateEvidenceUnsignedBase = z.object({
  schema: z.literal(PHYSICAL_GATE_EVIDENCE_SCHEMA), id: z.string().uuid(),
  gate: z.enum(["G0_BASELINE", "G1_ELECTRON_ZERO", "G2_CERTIFIED_ARTIFACTS", "G3_AUTOMATIC_EXECUTION", "G4_WAN_CONVEYOR", "G5_RESILIENCE_SECURITY", "G7_FINAL_RELEASE"]),
  check: z.enum(["physical_hardware", "native_lifecycle_matrix", "qwen3_physical_certification", "two_host_exact_execution", "physical_conveyor_ab", "physical_fault_campaign", "release_physical_matrix"]),
  campaign: z.enum(["hardware-inventory", "native-lifecycle", "model-certification", "two-host-execution", "wan-conveyor-ab", "fault-recovery", "release-matrix"]),
  status: z.enum(["pass", "fail"]), sourceSha: z.string().regex(/^[a-f0-9]{40}$/), sourceId: sha256,
  artifactDigests: z.array(sha256).min(1).max(64).refine((values) => new Set(values).size === values.length),
  startedAt: z.string().datetime({ offset: true }), completedAt: z.string().datetime({ offset: true }),
  hardware: z.array(z.object({ id: z.string().min(1).max(128), fingerprint: sha256,
    platform, backend: z.enum(["cpu", "cuda", "rocm", "mps", "directml"]), driverVersion: z.string().min(1).max(128).nullable() }).strict()).min(1).max(64),
  coverage: z.object({ platforms: z.array(platform).max(8), networkScopes: z.array(z.enum(["same-host", "lan", "multi-site", "public-relay"])).max(8),
    scenarios: z.array(z.string().regex(/^[a-z0-9][a-z0-9._-]{0,127}$/)).max(64), models: z.array(z.string().min(1).max(256)).max(32) }).strict(),
  measurements: z.array(z.object({ name: z.string().regex(/^[a-z][a-z0-9._-]{0,127}$/), value: z.number().finite(), unit: z.string().min(1).max(64),
    aggregation: z.enum(["sample", "min", "max", "mean", "median", "p50", "p95", "p99"]), evidenceClass: z.literal("hardware-physical") }).strict()).max(512),
  assertions: z.array(z.object({ id: z.string().regex(/^[a-z0-9][a-z0-9._-]{0,127}$/), status: z.enum(["pass", "fail"]),
    detail: z.string().min(1).max(2_048), evidence: z.array(z.string().min(1).max(2_048)).min(1).max(32) }).strict()).min(1).max(128),
}).strict();

function refineReceipt(receipt: z.infer<typeof physicalGateEvidenceUnsignedBase>, context: z.RefinementCtx): void {
  const expected = PHYSICAL_CHECK_CAMPAIGNS[receipt.check];
  if (!expected || expected.gate !== receipt.gate || expected.campaign !== receipt.campaign) {
    context.addIssue({ code: "custom", path: ["check"], message: "Physical check, gate and campaign do not match" });
  }
  const expectedStatus = receipt.assertions.some((assertion) => assertion.status === "fail") ? "fail" : "pass";
  if (receipt.status !== expectedStatus) context.addIssue({ code: "custom", path: ["status"], message: `Expected ${expectedStatus} from assertions` });
  if (Date.parse(receipt.completedAt) < Date.parse(receipt.startedAt)) context.addIssue({ code: "custom", path: ["completedAt"], message: "Completion cannot precede start" });
}

const PHYSICAL_CHECK_CAMPAIGNS: Record<string, { gate: string; campaign: string }> = {
  physical_hardware: { gate: "G0_BASELINE", campaign: "hardware-inventory" },
  native_lifecycle_matrix: { gate: "G1_ELECTRON_ZERO", campaign: "native-lifecycle" },
  qwen3_physical_certification: { gate: "G2_CERTIFIED_ARTIFACTS", campaign: "model-certification" },
  two_host_exact_execution: { gate: "G3_AUTOMATIC_EXECUTION", campaign: "two-host-execution" },
  physical_conveyor_ab: { gate: "G4_WAN_CONVEYOR", campaign: "wan-conveyor-ab" },
  physical_fault_campaign: { gate: "G5_RESILIENCE_SECURITY", campaign: "fault-recovery" },
  release_physical_matrix: { gate: "G7_FINAL_RELEASE", campaign: "release-matrix" },
};

export const physicalGateEvidenceUnsignedSchema = physicalGateEvidenceUnsignedBase.superRefine(refineReceipt);

export const physicalGateEvidenceSchema = physicalGateEvidenceUnsignedBase.extend({ receiptDigest: sha256,
  signature: z.object({ keyId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/), algorithm: z.literal("ed25519"),
    publicKeySpki: z.string().min(40).max(256).regex(/^[A-Za-z0-9_-]+$/), value: z.string().min(64).max(256).regex(/^[A-Za-z0-9_-]+$/) }).strict() }).strict().superRefine(refineReceipt);

export type PhysicalGateEvidenceUnsigned = z.infer<typeof physicalGateEvidenceUnsignedSchema>;
export type PhysicalGateEvidence = z.infer<typeof physicalGateEvidenceSchema>;

export function sealPhysicalGateEvidence(unsignedValue: unknown, keyId: string, privateKey: KeyObject): PhysicalGateEvidence {
  const unsigned = physicalGateEvidenceUnsignedSchema.parse(unsignedValue);
  const receiptDigest = digest(unsigned); const publicKey = createPublicKey(privateKey);
  const publicKeySpki = Buffer.from(publicKey.export({ format: "der", type: "spki" })).toString("base64url");
  const value = sign(null, Buffer.from(signingPayload(unsigned, receiptDigest)), privateKey).toString("base64url");
  return physicalGateEvidenceSchema.parse({ ...unsigned, receiptDigest, signature: { keyId, algorithm: "ed25519", publicKeySpki, value } });
}

export function verifyPhysicalGateEvidence(value: unknown, trustedKeys: ReadonlyMap<string, string>): PhysicalGateEvidence {
  const receipt = physicalGateEvidenceSchema.parse(value); const { receiptDigest, signature, ...unsigned } = receipt;
  if (receiptDigest !== digest(unsigned)) throw new Error("physical_gate_receipt_digest_mismatch");
  const trusted = trustedKeys.get(signature.keyId);
  if (!trusted || trusted !== signature.publicKeySpki) throw new Error("physical_gate_receipt_signer_is_untrusted");
  const bytes = Buffer.from(signature.publicKeySpki, "base64url");
  if (bytes.toString("base64url") !== signature.publicKeySpki) throw new Error("physical_gate_receipt_public_key_is_not_canonical");
  const key = createPublicKey({ key: bytes, format: "der", type: "spki" });
  if (key.asymmetricKeyType !== "ed25519" || !verify(null, Buffer.from(signingPayload(unsigned, receiptDigest)), key, Buffer.from(signature.value, "base64url"))) {
    throw new Error("physical_gate_receipt_signature_is_invalid");
  }
  return receipt;
}

function digest(unsigned: PhysicalGateEvidenceUnsigned): `sha256:${string}` { return `sha256:${createHash("sha256").update(canonicalEvidenceJson(unsigned)).digest("hex")}`; }
function signingPayload(unsigned: PhysicalGateEvidenceUnsigned, receiptDigest: string): string { return canonicalEvidenceJson({ ...unsigned, receiptDigest }); }
