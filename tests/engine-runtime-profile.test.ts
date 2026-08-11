import { describe, expect, it } from "vitest";
import {
  engineRuntimeProfileSchema,
  requireEligibleEngineRuntimeProfile,
  sealEngineRuntimeProfile,
  type EngineRuntimeProfileInput,
} from "../src/contracts/engine-runtime-profile.js";
import { nativeEngineNodeCapabilityFromProfile } from "../src/distribution/fleet-placement.js";

describe("certified engine runtime profiles", () => {
  it("seals physical capacity and derives a planner capability", () => {
    const profile = sealEngineRuntimeProfile(profileInput());
    const capability = nativeEngineNodeCapabilityFromProfile(profile, {
      nowMs: Date.parse("2026-08-10T12:30:00.000Z"),
      descriptorDigest: digest("a"),
      certificationId: digest("b"),
      artifactManifestDigest: digest("c"),
      backend: "cuda",
      runtimeAbi: "cuda-12",
      quantization: "bf16",
      contextTokens: 8_192,
    });

    expect(capability).toMatchObject({
      profileId: profile.profileId,
      descriptorDigest: digest("a"),
      maxContextTokens: 8_192,
      maxLayerCount: 16,
      kvBytesPerToken: 2_048,
      maxKvTokens: 16_384,
      decodeScale: 0.5,
      prefillScale: 0.75,
      roles: ["head", "middle", "tail"],
    });
  });

  it("rejects tampering, stale evidence, noise and identity reuse", () => {
    const profile = sealEngineRuntimeProfile(profileInput());
    expect(engineRuntimeProfileSchema.safeParse({
      ...profile,
      capacity: { ...profile.capacity, maxLayerCount: 32 },
    }).success).toBe(false);
    expect(() => requireEligibleEngineRuntimeProfile(profile, {
      nowMs: Date.parse("2026-08-12T12:00:00.000Z"),
    })).toThrow("engine_runtime_profile_is_stale");
    expect(() => requireEligibleEngineRuntimeProfile(profile, {
      nowMs: Date.parse("2026-08-10T12:30:00.000Z"),
      maximumConfidenceHalfWidthPct: 5,
    })).toThrow("engine_runtime_profile_confidence_is_too_low");
    expect(() => requireEligibleEngineRuntimeProfile(profile, {
      nowMs: Date.parse("2026-08-10T12:30:00.000Z"),
      artifactManifestDigest: digest("f"),
    })).toThrow("engine_runtime_profile_artifactManifestDigest_mismatch");
    expect(() => requireEligibleEngineRuntimeProfile(profile, {
      nowMs: Date.parse("2026-08-10T12:30:00.000Z"),
      contextTokens: 16_385,
    })).toThrow("engine_runtime_profile_context_is_insufficient");
  });

  it("rejects inconsistent KV and latency distributions", () => {
    expect(() => sealEngineRuntimeProfile({
      ...profileInput(),
      capacity: {
        ...profileInput().capacity,
        maxKvTokens: 4_096,
      },
    })).toThrow("engine_runtime_profile_kv_capacity_is_insufficient");
    expect(() => sealEngineRuntimeProfile({
      ...profileInput(),
      costs: {
        ...profileInput().costs,
        verifyMsPerTokenP50: 6,
        verifyMsPerTokenP95: 5,
      },
    })).toThrow("engine_runtime_profile_cost_distribution_is_invalid");
  });
});

function profileInput(): EngineRuntimeProfileInput {
  return {
    descriptorDigest: digest("a"),
    certificationId: digest("b"),
    artifactManifestDigest: digest("c"),
    sourceId: digest("d"),
    hardwareFingerprintSha256: digest("e"),
    workerId: "worker-a",
    sessionId: "session-a",
    nodeId: "node-a",
    modelId: "Qwen/Qwen3-8B",
    modelRevision: "1".repeat(40),
    backend: "cuda",
    runtimeAbi: "cuda-12",
    quantization: "bf16",
    measuredAt: "2026-08-10T12:00:00.000Z",
    expiresAt: "2026-08-11T12:00:00.000Z",
    samples: 21,
    confidenceHalfWidthPct: 8,
    capacity: {
      contextTokens: 8_192,
      maxLayerCount: 16,
      kvBytesPerToken: 2_048,
      maxKvTokens: 16_384,
      usableMemoryBytes: 16 * 1024 * 1024 * 1024,
    },
    costs: {
      decodeMsPerTokenP50: 4,
      decodeMsPerTokenP95: 5,
      prefillMsPerTokenP50: 0.2,
      prefillMsPerTokenP95: 0.3,
      verifyMsPerTokenP50: 3,
      verifyMsPerTokenP95: 4,
      decodeScale: 0.5,
      prefillScale: 0.75,
    },
    features: {
      fastKernel: true,
      graphMode: "available",
      roles: ["head", "middle", "tail"],
    },
    evidence: {
      deploymentCanaryEvidenceId: digest("6"),
      runtimePerformanceEvidenceId: digest("7"),
    },
  };
}

function digest(seed: string): `sha256:${string}` {
  return `sha256:${seed.repeat(64)}`;
}
