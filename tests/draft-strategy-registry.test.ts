import { generateKeyPairSync } from "node:crypto";
import { describe, expect, it } from "vitest";

import {
  DRAFT_STRATEGY_CERTIFICATION_SCHEMA,
  DRAFT_STRATEGY_DESCRIPTOR_SCHEMA,
  signDraftStrategyCertification,
} from "../src/contracts/engine-family.js";
import {
  DraftStrategyResolutionError,
  NativeDraftStrategyRegistry,
} from "../src/distribution/draft-strategy-registry.js";

const digest = (character: string) => `sha256:${character.repeat(64)}` as const;
const keys = generateKeyPairSync("ed25519");

function descriptor() {
  return {
    schema: DRAFT_STRATEGY_DESCRIPTOR_SCHEMA,
    strategyId: "qwen3-ngram",
    version: "1.0.0",
    kind: "ngram" as const,
    componentDigest: digest("6"),
    targetDescriptorDigest: digest("7"),
    tokenizerDigest: digest("8"),
    vocabularyDigest: digest("9"),
    resource: {
      minimumRamBytes: 1024,
      minimumVramBytes: 2048,
      allowedBackends: ["cpu" as const, "cuda" as const],
    },
    limits: { minDraftTokens: 1, maxDraftTokens: 16, maxInflightWaves: 8 },
    evidenceDigest: digest("a"),
  };
}

function certification(
  descriptorDigest: `sha256:${string}`,
  status: "candidate" | "certified" | "revoked" = "certified",
) {
  return signDraftStrategyCertification(
    {
      schema: DRAFT_STRATEGY_CERTIFICATION_SCHEMA,
      descriptorDigest,
      sourceId: digest("b"),
      status,
      validFrom: "2026-08-01T00:00:00.000Z",
      expiresAt: "2026-09-01T00:00:00.000Z",
      publisherKeyId: "release-2026",
    },
    keys.privateKey,
  );
}

function request(descriptorDigest: `sha256:${string}`) {
  return {
    descriptorDigest,
    targetDescriptorDigest: digest("7"),
    tokenizerDigest: digest("8"),
    vocabularyDigest: digest("9"),
    backend: "cuda" as const,
    availableRamBytes: 4096,
    availableVramBytes: 8192,
    requestedDraftTokens: 8,
    requestedInflightWaves: 4,
    now: new Date("2026-08-10T00:00:00.000Z"),
  };
}

function registry() {
  return new NativeDraftStrategyRegistry(
    new Map([["release-2026", keys.publicKey]]),
  );
}

describe("native draft strategy registry", () => {
  it("activates an exact compatible strategy only after signed certification", () => {
    const authority = registry();
    const descriptorDigest = authority.registerDescriptor(descriptor());
    authority.registerCertification(certification(descriptorDigest));
    const resolved = authority.resolve(request(descriptorDigest));
    expect(resolved.descriptor.kind).toBe("ngram");
    expect(resolved.certification.status).toBe("certified");
  });

  it("rejects unknown publisher keys and signature tampering", () => {
    const authority = registry();
    const descriptorDigest = authority.registerDescriptor(descriptor());
    const signed = certification(descriptorDigest);
    expect(() => authority.registerCertification({ ...signed, sourceId: digest("c") }))
      .toThrowError(new DraftStrategyResolutionError(
        "draft_strategy_certification_identity_mismatch",
      ));
    const changedSignature = `${signed.signature.startsWith("A") ? "B" : "A"}${
      signed.signature.slice(1)
    }`;
    expect(() => authority.registerCertification({
      ...signed,
      signature: changedSignature,
    })).toThrowError(new DraftStrategyResolutionError(
      "draft_strategy_certification_signature_is_invalid",
    ));
    const { certificationId: _id, signature: _signature, ...content } = signed;
    const unknown = signDraftStrategyCertification(
      { ...content, publisherKeyId: "other" },
      keys.privateKey,
    );
    expect(() => authority.registerCertification(unknown)).toThrowError(
      new DraftStrategyResolutionError(
        "draft_strategy_certification_key_is_unknown:other",
      ),
    );
  });

  it("rejects certification before its descriptor exists", () => {
    expect(() => registry().registerCertification(certification(digest("d"))))
      .toThrowError(new DraftStrategyResolutionError(
        "draft_strategy_certification_descriptor_is_unknown",
      ));
  });

  it("fails closed for candidate, expired, and revoked authorities", () => {
    const candidateAuthority = registry();
    const descriptorDigest = candidateAuthority.registerDescriptor(descriptor());
    candidateAuthority.registerCertification(certification(descriptorDigest, "candidate"));
    expect(() => candidateAuthority.resolve(request(descriptorDigest))).toThrowError(
      new DraftStrategyResolutionError(
        "draft_strategy_compatible_certification_is_missing",
      ),
    );

    const revokedAuthority = registry();
    revokedAuthority.registerDescriptor(descriptor());
    revokedAuthority.registerCertification(certification(descriptorDigest));
    revokedAuthority.registerCertification(certification(descriptorDigest, "revoked"));
    expect(() => revokedAuthority.resolve(request(descriptorDigest))).toThrowError(
      new DraftStrategyResolutionError("draft_strategy_is_revoked"),
    );
    expect(() => revokedAuthority.resolve({
      ...request(descriptorDigest),
      now: new Date("2026-10-01T00:00:00.000Z"),
    })).toThrowError(new DraftStrategyResolutionError(
      "draft_strategy_compatible_certification_is_missing",
    ));
  });

  it.each([
    [{ targetDescriptorDigest: digest("c") }, "draft_strategy_target_is_incompatible"],
    [{ tokenizerDigest: digest("c") }, "draft_strategy_target_is_incompatible"],
    [{ vocabularyDigest: digest("c") }, "draft_strategy_target_is_incompatible"],
    [{ backend: "rocm" as const }, "draft_strategy_backend_is_incompatible"],
    [{ availableVramBytes: 1024 }, "draft_strategy_resources_are_insufficient"],
    [{ requestedDraftTokens: 17 }, "draft_strategy_limits_are_exceeded"],
    [{ requestedInflightWaves: 9 }, "draft_strategy_limits_are_exceeded"],
  ])("rejects incompatible request %j", (override, code) => {
    const authority = registry();
    const descriptorDigest = authority.registerDescriptor(descriptor());
    authority.registerCertification(certification(descriptorDigest));
    expect(() => authority.resolve({ ...request(descriptorDigest), ...override }))
      .toThrowError(new DraftStrategyResolutionError(code));
  });
});
