import { describe, expect, it } from "vitest";
import {
  ACTIVATION_INTEGRITY_EVIDENCE_SCHEMA,
  ACTIVATION_SKETCH_SCHEMA,
  activationIntegrityEvidenceSchema,
  activationProjectionIndices,
  activationSeedCommitment,
  compareActivationSketches,
} from "../src/contracts/activation-integrity.js";

const seed = "01".repeat(16);

function sketch(projection: number[], overrides: Record<string, unknown> = {}) {
  return {
    schema: ACTIVATION_SKETCH_SCHEMA,
    seed,
    seedCommitment: activationSeedCommitment(seed),
    elementCount: 4096,
    sampleCount: projection.length,
    norm: 10,
    projection,
    ...overrides,
  };
}

describe("activation integrity", () => {
  it("pins the Python commitment and projection-index protocol", () => {
    expect(activationSeedCommitment(seed)).toBe(
      "sha256:d515beb3ef8010b8cca47253e5c7e1225acd5fc89fca5cf582cf736505de9ecb",
    );
    expect(activationProjectionIndices(seed, 4096, 8)).toEqual([
      1092, 523, 3972, 541, 863, 406, 2898, 2956,
    ]);
  });

  it("accepts bounded heterogeneous drift and binds the recompute verdict", () => {
    const trusted = sketch(Array.from({ length: 32 }, (_, index) => index + 1));
    const suspect = sketch(
      trusted.projection.map((value) => value * 1.00001),
      { norm: trusted.norm * 1.00001 },
    );
    const verdict = compareActivationSketches(suspect, trusted);
    expect(verdict.passed).toBe(true);
    expect(activationIntegrityEvidenceSchema.parse({
      schema: ACTIVATION_INTEGRITY_EVIDENCE_SCHEMA,
      challengeId: "challenge-1",
      suspectStageId: "stage-1",
      trustedStageId: "trusted-stage-1",
      suspect,
      trusted,
      verdict: { passed: true, cosine: verdict.cosine, relativeNorm: verdict.relativeNorm },
    })).toBeTruthy();
  });

  it("fails closed for forgery, malformed commitments and forged verdicts", () => {
    const trusted = sketch([1, 2, 3, 4]);
    const wrong = sketch([1, -2, 3, -4]);
    expect(compareActivationSketches(wrong, trusted)).toMatchObject({ passed: false });
    expect(compareActivationSketches(
      { ...trusted, seedCommitment: `sha256:${"0".repeat(64)}` },
      trusted,
    )).toMatchObject({ passed: false });
    expect(() => activationIntegrityEvidenceSchema.parse({
      schema: ACTIVATION_INTEGRITY_EVIDENCE_SCHEMA,
      challengeId: "challenge-1",
      suspectStageId: "stage-1",
      trustedStageId: "trusted-stage-1",
      suspect: trusted,
      trusted,
      verdict: { passed: true, cosine: 0.5, relativeNorm: 0 },
    })).toThrow("activation_integrity_verdict_mismatch");
  });
});
