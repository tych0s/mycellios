import { describe, expect, it } from "vitest";
import type { EngineRuntimeChallenge } from "../src/contracts/evidence-challenge.js";
import { probeQwen3EngineRuntimeProfile } from "../src/performance/engine-runtime-profile-probe.js";

describe("packaged Qwen3 engine runtime probe", () => {
  it("passes the exact challenged shape and accepts strict physical output", async () => {
    let invoked: readonly string[] = [];
    const measurement = physicalMeasurement();
    const result = await probeQwen3EngineRuntimeProfile(challenge(), {
      pythonExecutable: "python-runtime",
      pythonPath: ["/native/product"],
      device: "cuda:0",
      precision: "float16",
      commandRunner: async (_executable, arguments_) => {
        invoked = arguments_;
        return { code: 0, stdout: JSON.stringify(measurement), stderr: "" };
      },
    });

    expect(result).toEqual(measurement);
    expect(invoked).toEqual(expect.arrayContaining([
      "distributed_runtime.engine_runtime_profile",
      "--hidden-size",
      "4096",
      "--layer-count",
      "16",
      "--kv-bytes-per-token",
      "32768",
    ]));
  });

  it("fails closed on contaminated or mismatched probe output", async () => {
    await expect(probeQwen3EngineRuntimeProfile(challenge(), {
      pythonExecutable: "python-runtime",
      pythonPath: ["/native/product"],
      device: "cuda:0",
      precision: "float16",
      commandRunner: async () => ({
        code: 0,
        stdout: `${JSON.stringify(physicalMeasurement())}\ntrailing-noise`,
        stderr: "",
      }),
    })).rejects.toThrow("engine_runtime_probe_output_is_not_json");
    await expect(probeQwen3EngineRuntimeProfile(challenge(), {
      pythonExecutable: "python-runtime",
      pythonPath: ["/native/product"],
      device: "cuda:0",
      precision: "float16",
      commandRunner: async () => ({
        code: 0,
        stdout: JSON.stringify({
          ...physicalMeasurement(),
          capacity: { ...physicalMeasurement().capacity, kvBytesPerToken: 16_384 },
        }),
        stderr: "",
      }),
    })).rejects.toThrow("engine_runtime_probe_output_does_not_match_challenge");
  });
});

function challenge(): EngineRuntimeChallenge {
  const now = Date.now();
  return {
    schema: "mycellios-evidence-challenge/1",
    kind: "engine-runtime",
    challengeId: "challenge-qwen3",
    nonce: Buffer.alloc(32, 5).toString("base64url"),
    sessionId: "session-qwen3",
    workerId: "worker-qwen3",
    issuedAt: new Date(now - 1_000).toISOString(),
    expiresAt: new Date(now + 60_000).toISOString(),
    nodeId: "node-qwen3",
    probeKind: "qwen3-dense-v1",
    descriptorDigest: digest("a"),
    certificationId: digest("b"),
    artifactManifestDigest: digest("c"),
    modelId: "Qwen/Qwen3-8B",
    modelRevision: "1".repeat(40),
    backend: "cuda",
    runtimeAbi: "cuda-12",
    quantization: "bf16",
    contextTokens: 8_192,
    expectedLayerStart: 0,
    expectedLayerEnd: 16,
    expectedKvBytesPerToken: 32_768,
    expectedLayerWeightBytes: 256 * 1024 * 1024,
    referenceDecodeMsPerToken: 8,
    referencePrefillMsPerToken: 0.4,
    hiddenSize: 4_096,
    attentionHeads: 32,
    kvHeads: 4,
    headDim: 128,
    requiredRoles: ["head", "tail"],
    minimumSamples: 7,
  };
}

function physicalMeasurement() {
  return {
    measuredAt: new Date().toISOString(),
    samples: 7,
    confidenceHalfWidthPct: 8,
    capacity: {
      contextTokens: 8_192,
      maxLayerCount: 16,
      kvBytesPerToken: 32_768,
      maxKvTokens: 8_192,
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
      prefillScale: 0.5,
    },
    features: {
      fastKernel: true,
      graphMode: "available" as const,
      roles: ["head", "middle", "tail"] as Array<"head" | "middle" | "tail">,
    },
  };
}

function digest(seed: string): `sha256:${string}` {
  return `sha256:${seed.repeat(64)}`;
}
