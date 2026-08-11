import { generateKeyPairSync } from "node:crypto";
import { describe, expect, it } from "vitest";

import {
  assertProductiveModelCertification,
  buildModelCertification,
  signModelCertification,
  verifyModelCertification,
} from "../src/contracts/model-certification.js";
import { ModelCertificationRegistry } from "../src/coordinator/model-certification-registry.js";
import { MeshDatabase } from "../src/storage/database.js";

const sha = (character: string) => `sha256:${character.repeat(64)}` as const;

function fixture(decision: "certified" | "revoked" = "certified") {
  const keys = generateKeyPairSync("ed25519");
  const unsigned = buildModelCertification({
    decision,
    modelFamily: "qwen3-dense",
    distributionManifestId: sha("a"),
    adapterContractId: sha("b"),
    componentManifestId: sha("c"),
    topology: { kind: "local-complete", stageCount: 1, tensorParallelDegree: 1 },
    hardware: {
      platform: "linux",
      arch: "x64",
      backend: "cuda",
      deviceFamily: "nvidia-sm89",
      minimumMemoryBytes: 8_000_000_000,
      driverFingerprint: sha("d"),
    },
    context: {
      maximumTokens: 8_192,
      codecs: ["fp16-activations/1"],
      workerProtocol: { min: 8, max: 8 },
      tensorAbi: "mycellios-tensor/1",
    },
    evidence: {
      class: "physical",
      receiptId: sha("e"),
      sourceRevision: "f".repeat(40),
      measuredAt: "2026-08-10T12:00:00.000Z",
    },
    review: { reviewerId: "release-operator-1", reviewedAt: "2026-08-10T13:00:00.000Z" },
    expiresAt: "2027-08-10T13:00:00.000Z",
  });
  const certification = signModelCertification(unsigned, { keyId: "model-certification-2026", privateKey: keys.privateKey });
  return {
    certification,
    pinnedKey: {
      keyId: "model-certification-2026",
      spki: keys.publicKey.export({ format: "der", type: "spki" }).toString("base64url"),
    },
  };
}

describe("model certification", () => {
  it("publishes a verified certification durably and idempotently", () => {
    const data = fixture();
    const database = new MeshDatabase(":memory:");
    const registry = new ModelCertificationRegistry(database, [data.pinnedKey]);
    expect(registry.publish(data.certification, new Date("2026-08-11T00:00:00.000Z")).alreadyPublished).toBe(false);
    expect(registry.publish(data.certification, new Date("2026-08-11T00:00:00.000Z")).alreadyPublished).toBe(true);
    expect(registry.list()).toEqual([data.certification]);
    expect(() => new ModelCertificationRegistry(database, [{ ...data.pinnedKey, keyId: "other" }])
      .publish(data.certification, new Date("2026-08-11T00:00:00.000Z"))).toThrow("model_certification_key_is_not_pinned");
    database.close();
  });
  it("verifies a physically evidenced, reviewed certification against a pinned key", () => {
    const data = fixture();
    const verified = verifyModelCertification(data.certification, {
      pinnedKey: data.pinnedKey,
      expectedDistributionManifestId: sha("a"),
      now: new Date("2026-08-11T00:00:00.000Z"),
    });
    expect(() => assertProductiveModelCertification(verified)).not.toThrow();
  });

  it("rejects heuristic evidence, invalid topology and expiry", () => {
    const data = fixture();
    expect(() => buildModelCertification({ ...data.certification, evidence: { ...data.certification.evidence, class: "automatic" as never } })).toThrow();
    expect(() => buildModelCertification({ ...data.certification, topology: { ...data.certification.topology, stageCount: 2 } })).toThrow("model_certification_topology_stage_count_is_invalid");
    expect(() => verifyModelCertification(data.certification, { pinnedKey: data.pinnedKey, now: new Date("2028-01-01T00:00:00.000Z") })).toThrow("model_certification_expired");
  });

  it("rejects tampering, the wrong distribution and revoked decisions", () => {
    const data = fixture();
    expect(() => verifyModelCertification({ ...data.certification, context: { ...data.certification.context, maximumTokens: 16_384 } }, { pinnedKey: data.pinnedKey })).toThrow("model_certification_identity_mismatch");
    expect(() => verifyModelCertification(data.certification, { pinnedKey: data.pinnedKey, expectedDistributionManifestId: sha("9") })).toThrow("model_certification_distribution_mismatch");
    expect(() => assertProductiveModelCertification(fixture("revoked").certification)).toThrow("model_certification_is_not_productive");
  });
});
