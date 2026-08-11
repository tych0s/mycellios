import { createPublicKey, generateKeyPairSync, randomUUID, type KeyObject } from "node:crypto";
import { describe, expect, it } from "vitest";
import { sealPhysicalGateEvidence, verifyPhysicalGateEvidence, type PhysicalGateEvidenceUnsigned } from "../src/contracts/physical-gate-evidence.js";
import { importPhysicalGateEvidence } from "../src/program/physical-gate-evidence.js";

const privateKey = generateKeyPairSync("ed25519").privateKey;
const publicKeySpki = generatePublic(privateKey);
const trust = { schema: "mycellios-physical-evidence-trust/1", keys: [{ keyId: "physical-lab-1", publicKeySpki, status: "active" }] };
const sourceSha = "a".repeat(40), sourceIdentity = `sha256:${"c".repeat(64)}`, artifact = `sha256:${"b".repeat(64)}`;
const empty = { schema: "mycellios-gate-imported-evidence/1", evidence: [] };

describe("signed physical gate evidence", () => {
  it("verifies an exact trusted receipt and rejects tampering or an unknown signer", () => {
    const receipt = sealPhysicalGateEvidence(twoHost(), "physical-lab-1", privateKey);
    expect(verifyPhysicalGateEvidence(receipt, new Map([["physical-lab-1", publicKeySpki]]))).toEqual(receipt);
    expect(() => verifyPhysicalGateEvidence({ ...receipt, completedAt: "2026-08-10T13:00:01.000Z" }, new Map([["physical-lab-1", publicKeySpki]])))
      .toThrow("physical_gate_receipt_digest_mismatch");
    expect(() => verifyPhysicalGateEvidence(receipt, new Map())).toThrow("physical_gate_receipt_signer_is_untrusted");
  });

  it("imports complete two-host coverage bound to the exact candidate", () => {
    const receipt = sealPhysicalGateEvidence(twoHost(), "physical-lab-1", privateKey);
    const imported = importPhysicalGateEvidence({ receipts: [receipt], trust, sourceSha, sourceIdentity, existing: empty });
    expect(imported.evidence).toEqual([expect.objectContaining({ gate: "G3_AUTOMATIC_EXECUTION", check: "two_host_exact_execution",
      status: "pass", sourceSha, sourceIdentity, artifactDigests: [artifact], evidence: [`physical-receipt:${receipt.id}:${receipt.receiptDigest}`] })]);
  });

  it("rejects incomplete coverage, mixed SHA and unsigned trust", () => {
    const incomplete = sealPhysicalGateEvidence({ ...twoHost(), coverage: { platforms: [{ os: "linux", arch: "x64" }], networkScopes: [], scenarios: ["split"], models: [] } }, "physical-lab-1", privateKey);
    expect(() => importPhysicalGateEvidence({ receipts: [incomplete], trust, sourceSha, sourceIdentity, existing: empty })).toThrow("physical_gate_coverage_is_incomplete");
    const foreign = sealPhysicalGateEvidence({ ...twoHost(), sourceSha: "c".repeat(40) }, "physical-lab-1", privateKey);
    expect(() => importPhysicalGateEvidence({ receipts: [foreign], trust, sourceSha, sourceIdentity, existing: empty })).toThrow("physical_gate_import_source_sha_mismatch");
    expect(() => importPhysicalGateEvidence({ receipts: [incomplete], trust: { ...trust, keys: [] }, sourceSha, sourceIdentity, existing: empty })).toThrow("signer_is_untrusted");
    expect(() => importPhysicalGateEvidence({ receipts: [incomplete], trust: { ...trust, keys: trust.keys.map((key) => ({ ...key, status: "revoked" as const })) }, sourceSha, sourceIdentity, existing: empty }))
      .toThrow("signer_is_untrusted");
  });

  it("rejects a signed receipt from another source tree even when the commit SHA matches", () => {
    const receipt = sealPhysicalGateEvidence({ ...twoHost(), sourceId: `sha256:${"f".repeat(64)}` }, "physical-lab-1", privateKey);
    expect(() => importPhysicalGateEvidence({ receipts: [receipt], trust, sourceSha, sourceIdentity, existing: empty }))
      .toThrow("physical_gate_import_source_identity_mismatch");
  });

  it("rejects receipts that mix independently built candidates", () => {
    const first = sealPhysicalGateEvidence(twoHost(), "physical-lab-1", privateKey);
    const second = sealPhysicalGateEvidence({ ...twoHost(), sourceId: `sha256:${"f".repeat(64)}` }, "physical-lab-1", privateKey);
    expect(() => importPhysicalGateEvidence({ receipts: [first, second], trust, sourceSha, sourceIdentity, existing: empty }))
      .toThrow("physical_gate_import_source_identity_mismatch");
  });

  it("imports a signed failure immediately without pretending matrix coverage", () => {
    const failed = sealPhysicalGateEvidence({ ...twoHost(), status: "fail", coverage: { platforms: [], networkScopes: [], scenarios: [], models: [] },
      assertions: [{ id: "execution", status: "fail", detail: "physical output diverged", evidence: ["raw/output.json"] }] }, "physical-lab-1", privateKey);
    expect(importPhysicalGateEvidence({ receipts: [failed], trust, sourceSha, sourceIdentity, existing: empty }).evidence[0]).toMatchObject({ status: "fail" });
  });
});

function twoHost(): PhysicalGateEvidenceUnsigned {
  return { schema: "mycellios-physical-gate-evidence/1", id: randomUUID(), gate: "G3_AUTOMATIC_EXECUTION", check: "two_host_exact_execution", campaign: "two-host-execution",
    status: "pass", sourceSha, sourceId: sourceIdentity, artifactDigests: [artifact], startedAt: "2026-08-10T12:00:00.000Z", completedAt: "2026-08-10T12:05:00.000Z",
    hardware: [hardware("host-a", "d"), hardware("host-b", "e")], coverage: { platforms: [{ os: "linux", arch: "x64" }], networkScopes: ["lan"], scenarios: ["local-complete", "split"], models: ["Qwen/Qwen3"] },
    measurements: [{ name: "tokens-per-second", value: 12.5, unit: "tokens/s", aggregation: "median", evidenceClass: "hardware-physical" }],
    assertions: [{ id: "exact-output", status: "pass", detail: "Output matched the local reference", evidence: ["raw/exact-output.json"] }] };
}
function hardware(id: string, digest: string) { return { id, fingerprint: `sha256:${digest.repeat(64)}` as const, platform: { os: "linux" as const, arch: "x64" as const }, backend: "cuda" as const, driverVersion: "600.1" }; }
function generatePublic(key: KeyObject): string {
  return Buffer.from(createPublicKey(key).export({ format: "der", type: "spki" })).toString("base64url");
}
