import { describe, expect, it } from "vitest";
import {
  plannerScalesFromProfile,
  sealRuntimePerformanceProfile,
  validateRuntimePerformanceProfile,
  type RuntimePerformanceProfileInput,
} from "../src/performance/runtime-profile.js";
import { activationCodec } from "../src/distribution/codecs.js";

describe("sealed runtime performance profile", () => {
  it("converts physical measurements into comparable planner scales", () => {
    const profile = sealRuntimePerformanceProfile(input());
    validateRuntimePerformanceProfile(profile);

    expect(plannerScalesFromProfile(profile, {
      now: Date.parse("2026-07-26T00:00:00.000Z"),
    })).toEqual({
      decodeScale: 0.5,
      prefillScale: 0.5,
      codecScale: 0.5,
      profileId: profile.profileId,
      measuredAt: "2026-07-25T12:00:00.000Z",
    });
    const codec = activationCodec("fp16");
    expect(codec.encodeGbps / plannerScalesFromProfile(profile, {
      now: Date.parse("2026-07-26T00:00:00.000Z"),
    }).codecScale).toBe(2);
  });

  it("rejects tampering instead of silently planning with false speed", () => {
    const profile = sealRuntimePerformanceProfile(input());
    expect(() => validateRuntimePerformanceProfile({
      ...profile,
      decodeMemory: { ...profile.decodeMemory, p50: 410 },
    })).toThrow("runtime_performance_profile_seal_is_invalid");
  });

  it("rejects stale and noisy profiles", () => {
    const profile = sealRuntimePerformanceProfile(input());
    expect(() => plannerScalesFromProfile(profile, {
      now: Date.parse("2026-08-10T00:00:00.000Z"),
    })).toThrow("runtime_performance_profile_is_stale");

    const noisy = sealRuntimePerformanceProfile({
      ...input(),
      prefillCompute: {
        ...input().prefillCompute,
        confidenceHalfWidthPct: 25,
      },
    });
    expect(() => plannerScalesFromProfile(noisy, {
      now: Date.parse("2026-07-26T00:00:00.000Z"),
    })).toThrow("runtime_performance_profile_prefill_confidence_is_too_low");
  });

  it("rejects a non-physical calibration by default", () => {
    const profile = sealRuntimePerformanceProfile({
      ...input(),
      source: "runtime-calibration",
    });
    expect(() => plannerScalesFromProfile(profile, {
      now: Date.parse("2026-07-26T00:00:00.000Z"),
    })).toThrow("runtime_performance_profile_source_is_not_physical");
  });

  it("requires enough samples and coherent percentiles", () => {
    expect(() => sealRuntimePerformanceProfile({
      ...input(),
      activationCodec: { ...input().activationCodec, samples: 2 },
    })).toThrow("runtime_performance_profile_codec_sample_count_is_invalid");
    expect(() => sealRuntimePerformanceProfile({
      ...input(),
      decodeMemory: { ...input().decodeMemory, p5: 500, p50: 400 },
    })).toThrow("runtime_performance_profile_decode_distribution_is_invalid");
  });
});

function input(): RuntimePerformanceProfileInput {
  return {
    measuredAt: "2026-07-25T12:00:00.000Z",
    backend: "cuda",
    deviceName: "NVIDIA RTX 2060",
    precision: "float16",
    source: "physical-microbenchmark",
    activationCodecId: "fp16",
    decodeMemory: series("GB/s", 350, 400, 430),
    prefillCompute: series("TFLOP/s", 18, 20, 21),
    activationCodec: series("GB/s", 1.8, 2, 2.1),
  };
}

function series(
  unit: "GB/s" | "TFLOP/s",
  p5: number,
  p50: number,
  p95: number,
) {
  return {
    unit,
    warmupSamples: 2,
    samples: 7,
    p5,
    p50,
    p95,
    confidenceHalfWidthPct: 5,
  };
}
