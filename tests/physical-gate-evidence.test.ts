import { createPublicKey, generateKeyPairSync, randomUUID, type KeyObject } from "node:crypto";
import { describe, expect, it } from "vitest";
import { sealPhysicalGateEvidence, verifyPhysicalGateEvidence, type PhysicalGateEvidenceUnsigned } from "../src/contracts/physical-gate-evidence.js";
import { importPhysicalGateEvidence } from "../src/program/physical-gate-evidence.js";
import {
  ACTIVATION_INTEGRITY_EVIDENCE_SCHEMA,
  ACTIVATION_SKETCH_SCHEMA,
  activationSeedCommitment,
  compareActivationSketches,
} from "../src/contracts/activation-integrity.js";

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

  it("binds valid activation integrity and rejects a forged verdict before signing", () => {
    const activationIntegrity = activationEvidence();
    const receipt = sealPhysicalGateEvidence(
      { ...twoHost(), activationIntegrity: [activationIntegrity] },
      "physical-lab-1",
      privateKey,
    );
    expect(receipt.activationIntegrity).toEqual([activationIntegrity]);
    expect(() => sealPhysicalGateEvidence({
      ...twoHost(),
      activationIntegrity: [{
        ...activationIntegrity,
        verdict: { ...activationIntegrity.verdict, cosine: 0.5 },
      }],
    }, "physical-lab-1", privateKey)).toThrow("activation_integrity_verdict_mismatch");
  });

  it("rejects incomplete coverage, mixed SHA and unsigned trust", () => {
    expect(() => sealPhysicalGateEvidence({ ...twoHost(), coverage: { platforms: [{ os: "linux", arch: "x64" }], networkScopes: [], scenarios: ["split"], models: [] } }, "physical-lab-1", privateKey))
      .toThrow("Two-host evidence cannot use same-host or loopback scope");
    const valid = sealPhysicalGateEvidence(twoHost(), "physical-lab-1", privateKey);
    const foreign = sealPhysicalGateEvidence({ ...twoHost(), sourceSha: "c".repeat(40) }, "physical-lab-1", privateKey);
    expect(() => importPhysicalGateEvidence({ receipts: [foreign], trust, sourceSha, sourceIdentity, existing: empty })).toThrow("physical_gate_import_source_sha_mismatch");
    expect(() => importPhysicalGateEvidence({ receipts: [valid], trust: { ...trust, keys: [] }, sourceSha, sourceIdentity, existing: empty })).toThrow("signer_is_untrusted");
    expect(() => importPhysicalGateEvidence({ receipts: [valid], trust: { ...trust, keys: trust.keys.map((key) => ({ ...key, status: "revoked" as const })) }, sourceSha, sourceIdentity, existing: empty }))
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
    measurements: [measurement("ttft-ms", 125, "ms"), measurement("tpot-ms", 80, "ms"), measurement("tokens-per-second", 12.5, "tokens/s"), measurement("peak-memory-mib", 7900, "MiB")],
    assertions: [assertion("exact-output"), assertion("non-empty-ranges"), assertion("build-model-match"), assertion("cleanup-complete")] };
}
function measurement(name: string, value: number, unit: string) { return { name, value, unit, aggregation: "median" as const, evidenceClass: "hardware-physical" as const }; }
function assertion(id: string) { return { id, status: "pass" as const, detail: `${id} verified`, evidence: [`raw/${id}.json`] }; }
function hardware(id: string, digest: string) { return { id, fingerprint: `sha256:${digest.repeat(64)}` as const, platform: { os: "linux" as const, arch: "x64" as const }, backend: "cuda" as const, driverVersion: "600.1" }; }
function activationEvidence() {
  const seed = "01".repeat(16);
  const base = {
    schema: ACTIVATION_SKETCH_SCHEMA,
    seed,
    seedCommitment: activationSeedCommitment(seed),
    elementCount: 4096,
    sampleCount: 4,
    norm: 10,
  };
  const trusted = { ...base, projection: [1, 2, 3, 4] };
  const suspect = { ...base, projection: [1.00001, 2.00002, 3.00003, 4.00004], norm: 10.0001 };
  const verdict = compareActivationSketches(suspect, trusted);
  if (!verdict.passed) throw new Error("activation evidence fixture must pass");
  return {
    schema: ACTIVATION_INTEGRITY_EVIDENCE_SCHEMA,
    challengeId: "challenge-1",
    suspectStageId: "stage-1",
    trustedStageId: "trusted-stage-1",
    suspect,
    trusted,
    verdict: { passed: true as const, cosine: verdict.cosine, relativeNorm: verdict.relativeNorm },
  };
}
function generatePublic(key: KeyObject): string {
  return Buffer.from(createPublicKey(key).export({ format: "der", type: "spki" })).toString("base64url");
}
