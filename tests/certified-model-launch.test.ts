import { generateKeyPairSync } from "node:crypto";
import { describe, expect, it } from "vitest";

import { buildModelCertification, signModelCertification } from "../src/contracts/model-certification.js";
import { buildModelDistributionManifest } from "../src/contracts/model-distribution-manifest.js";
import { authorizeCertifiedModelLaunch } from "../src/model-fabric/certified-model-launch.js";

const sha = (character: string) => `sha256:${character.repeat(64)}` as const;
const distribution = buildModelDistributionManifest({
  model: { id: "Qwen/Qwen3", revision: "1".repeat(40), architecture: "Qwen3ForCausalLM" }, tokenizer: { id: "Qwen/Qwen3", revision: "1".repeat(40) },
  format: "safetensors", quantization: { scheme: "none", bits: null, groupSize: null }, tensorAbi: "mycellios-tensor/1", layerCount: 1, expertCount: null,
  license: { spdx: "Apache-2.0", sourceUrl: "https://models.example.test/license" },
  artifacts: [
    { id: "input", role: "input", path: "input", sha256: sha("1"), sizeBytes: 1, sourceUrl: "https://models.example.test/input", layerRange: null, expertRange: null },
    { id: "layer", role: "layer-range", path: "layer", sha256: sha("2"), sizeBytes: 1, sourceUrl: "https://models.example.test/layer", layerRange: { start: 0, end: 1 }, expertRange: null },
    { id: "output", role: "output", path: "output", sha256: sha("3"), sizeBytes: 1, sourceUrl: "https://models.example.test/output", layerRange: null, expertRange: null },
  ],
});

function fixture() {
  const keys = generateKeyPairSync("ed25519");
  const topology = { kind: "local-complete" as const, stageCount: 1, tensorParallelDegree: 1 };
  const hardware = { platform: "linux" as const, arch: "x64" as const, backend: "cuda" as const, deviceFamily: "sm89", minimumMemoryBytes: 8_000, driverFingerprint: sha("4") };
  const certification = signModelCertification(buildModelCertification({
    decision: "certified", modelFamily: "qwen3-dense", distributionManifestId: distribution.manifestId,
    adapterContractId: sha("5"), componentManifestId: sha("6"), topology, hardware,
    context: { maximumTokens: 8_192, codecs: ["fp16/1"], workerProtocol: { min: 8, max: 8 }, tensorAbi: distribution.tensorAbi },
    evidence: { class: "physical", receiptId: sha("7"), sourceRevision: "8".repeat(40), measuredAt: "2026-08-10T10:00:00.000Z" },
    review: { reviewerId: "operator", reviewedAt: "2026-08-10T11:00:00.000Z" }, expiresAt: "2027-08-10T11:00:00.000Z",
  }), { keyId: "cert-key", privateKey: keys.privateKey });
  return {
    certification, topology, hardware,
    pinnedCertificationKey: { keyId: "cert-key", spki: keys.publicKey.export({ format: "der", type: "spki" }).toString("base64url") },
  };
}

describe("certified model launch", () => {
  it("authorizes only the exact certified distribution and environment", () => {
    const data = fixture();
    expect(authorizeCertifiedModelLaunch({
      distributionManifest: distribution, certification: data.certification, pinnedCertificationKey: data.pinnedCertificationKey,
      expectedAdapterContractId: sha("5"), expectedComponentManifestId: sha("6"), now: new Date("2026-08-11T00:00:00Z"),
      environment: { topology: data.topology, hardware: { ...data.hardware, availableMemoryBytes: 9_000 }, contextTokens: 4_096, codec: "fp16/1", workerProtocol: 8, tensorAbi: distribution.tensorAbi },
    })).toMatchObject({ distributionManifestId: distribution.manifestId });
  });

  it("rejects hardware, codec and ABI mismatches before launch", () => {
    const data = fixture();
    const base = { distributionManifest: distribution, certification: data.certification, pinnedCertificationKey: data.pinnedCertificationKey, expectedAdapterContractId: sha("5"), expectedComponentManifestId: sha("6"), environment: { topology: data.topology, hardware: { ...data.hardware, availableMemoryBytes: 9_000 }, contextTokens: 4_096, codec: "fp16/1", workerProtocol: 8, tensorAbi: distribution.tensorAbi } };
    expect(() => authorizeCertifiedModelLaunch({ ...base, environment: { ...base.environment, codec: "int8/1" } })).toThrow("certified_model_codec_mismatch");
    expect(() => authorizeCertifiedModelLaunch({ ...base, environment: { ...base.environment, tensorAbi: "other/1" } })).toThrow("certified_model_tensor_abi_mismatch");
    expect(() => authorizeCertifiedModelLaunch({ ...base, environment: { ...base.environment, hardware: { ...base.environment.hardware, availableMemoryBytes: 7_999 } } })).toThrow("certified_model_hardware_mismatch");
  });
});
