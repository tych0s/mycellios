import { createHash, timingSafeEqual } from "node:crypto";
import { z } from "zod";

export const ACTIVATION_SKETCH_SCHEMA = "mycellios-activation-sketch/1" as const;
export const ACTIVATION_INTEGRITY_EVIDENCE_SCHEMA =
  "mycellios-activation-integrity-evidence/1" as const;
export const DEFAULT_ACTIVATION_SAMPLE_COUNT = 256;
export const MAX_ACTIVATION_SAMPLE_COUNT = 1_024;
export const MAX_ACTIVATION_ELEMENT_COUNT = 2 ** 40;
export const DEFAULT_ACTIVATION_COSINE_THRESHOLD = 0.99;
export const DEFAULT_ACTIVATION_RELATIVE_NORM_THRESHOLD = 0.05;

const SEED_PATTERN = /^[0-9a-f]{32}$/;
const DIGEST_PATTERN = /^sha256:[0-9a-f]{64}$/;
const INDEX_DOMAIN = Buffer.from("mycellios-activation-index/1\0", "utf8");
const COMMITMENT_DOMAIN = Buffer.from("mycellios-activation-seed/1\0", "utf8");
const identifierSchema = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/);

export const activationSketchSchema = z.object({
  schema: z.literal(ACTIVATION_SKETCH_SCHEMA),
  seed: z.string().regex(SEED_PATTERN),
  seedCommitment: z.string().regex(DIGEST_PATTERN),
  elementCount: z.number().int().min(1).max(MAX_ACTIVATION_ELEMENT_COUNT),
  sampleCount: z.number().int().min(1).max(MAX_ACTIVATION_SAMPLE_COUNT),
  norm: z.number().nonnegative().finite(),
  projection: z.array(z.number().finite()).min(1).max(MAX_ACTIVATION_SAMPLE_COUNT),
}).strict().superRefine((sketch, context) => {
  if (sketch.sampleCount > sketch.elementCount) {
    context.addIssue({ code: "custom", path: ["sampleCount"], message: "activation_sample_count_is_invalid" });
  }
  if (sketch.projection.length !== sketch.sampleCount) {
    context.addIssue({ code: "custom", path: ["projection"], message: "activation_sketch_projection_is_invalid" });
  }
  if (!verifyActivationSeedCommitment(sketch.seed, sketch.seedCommitment)) {
    context.addIssue({ code: "custom", path: ["seedCommitment"], message: "activation_sketch_commitment_is_invalid" });
  }
});

export type ActivationSketch = z.infer<typeof activationSketchSchema>;

export interface ActivationSketchVerdict {
  passed: boolean;
  cosine: number;
  relativeNorm: number;
  error?: string;
}

export const activationIntegrityEvidenceSchema = z.object({
  schema: z.literal(ACTIVATION_INTEGRITY_EVIDENCE_SCHEMA),
  challengeId: identifierSchema,
  suspectStageId: identifierSchema,
  trustedStageId: identifierSchema,
  suspect: activationSketchSchema,
  trusted: activationSketchSchema,
  verdict: z.object({
    passed: z.literal(true),
    cosine: z.number().min(-1).max(1).finite(),
    relativeNorm: z.number().min(0).max(1).finite(),
  }).strict(),
}).strict().superRefine((evidence, context) => {
  if (evidence.suspectStageId === evidence.trustedStageId) {
    context.addIssue({ code: "custom", path: ["trustedStageId"], message: "activation_integrity_trusted_stage_must_be_distinct" });
  }
  const actual = compareActivationSketches(evidence.suspect, evidence.trusted);
  if (!actual.passed) {
    context.addIssue({ code: "custom", path: ["verdict"], message: actual.error ?? "activation_sketch_diverged" });
    return;
  }
  if (
    Math.abs(actual.cosine - evidence.verdict.cosine) > 1e-12
    || Math.abs(actual.relativeNorm - evidence.verdict.relativeNorm) > 1e-12
  ) {
    context.addIssue({ code: "custom", path: ["verdict"], message: "activation_integrity_verdict_mismatch" });
  }
});

export type ActivationIntegrityEvidence = z.infer<typeof activationIntegrityEvidenceSchema>;

export function activationSeedCommitment(seed: string): `sha256:${string}` {
  const seedBytes = parseSeed(seed);
  return `sha256:${createHash("sha256").update(COMMITMENT_DOMAIN).update(seedBytes).digest("hex")}`;
}

export function verifyActivationSeedCommitment(seed: string, commitment: string): boolean {
  if (!DIGEST_PATTERN.test(commitment)) return false;
  try {
    const expected = Buffer.from(activationSeedCommitment(seed), "utf8");
    const actual = Buffer.from(commitment, "utf8");
    return expected.length === actual.length && timingSafeEqual(expected, actual);
  } catch {
    return false;
  }
}

export function activationProjectionIndices(
  seed: string,
  elementCount: number,
  sampleCount = DEFAULT_ACTIVATION_SAMPLE_COUNT,
): number[] {
  const seedBytes = parseSeed(seed);
  if (!Number.isSafeInteger(elementCount) || elementCount < 1 || elementCount > MAX_ACTIVATION_ELEMENT_COUNT) {
    throw new Error("activation_element_count_is_invalid");
  }
  if (!Number.isSafeInteger(sampleCount) || sampleCount < 1 || sampleCount > Math.min(MAX_ACTIVATION_SAMPLE_COUNT, elementCount)) {
    throw new Error("activation_sample_count_is_invalid");
  }
  const indexes: number[] = [];
  const seen = new Set<number>();
  for (let counter = 0n; indexes.length < sampleCount; counter += 1n) {
    const counterBytes = Buffer.alloc(8);
    counterBytes.writeBigUInt64BE(counter);
    const digest = createHash("sha256")
      .update(INDEX_DOMAIN)
      .update(seedBytes)
      .update(counterBytes)
      .digest();
    const index = Number(digest.readBigUInt64BE(0) % BigInt(elementCount));
    if (!seen.has(index)) {
      seen.add(index);
      indexes.push(index);
    }
  }
  return indexes;
}

export function compareActivationSketches(
  suspectValue: unknown,
  trustedValue: unknown,
  options: { cosineThreshold?: number; relativeNormThreshold?: number } = {},
): ActivationSketchVerdict {
  try {
    const suspect = activationSketchSchema.parse(suspectValue);
    const trusted = activationSketchSchema.parse(trustedValue);
    const cosineThreshold = validThreshold(
      options.cosineThreshold ?? DEFAULT_ACTIVATION_COSINE_THRESHOLD,
      "activation_cosine_threshold_is_invalid",
    );
    const normThreshold = validThreshold(
      options.relativeNormThreshold ?? DEFAULT_ACTIVATION_RELATIVE_NORM_THRESHOLD,
      "activation_norm_threshold_is_invalid",
    );
    if (suspect.seed !== trusted.seed) throw new Error("activation_sketch_seed_mismatch");
    if (suspect.seedCommitment !== trusted.seedCommitment) throw new Error("activation_sketch_commitment_mismatch");
    if (suspect.elementCount !== trusted.elementCount || suspect.sampleCount !== trusted.sampleCount) {
      throw new Error("activation_sketch_shape_mismatch");
    }
    let dot = 0;
    let magnitudeSuspect = 0;
    let magnitudeTrusted = 0;
    for (let index = 0; index < suspect.sampleCount; index += 1) {
      const left = suspect.projection[index]!;
      const right = trusted.projection[index]!;
      dot += left * right;
      magnitudeSuspect += left * left;
      magnitudeTrusted += right * right;
    }
    magnitudeSuspect = Math.sqrt(magnitudeSuspect);
    magnitudeTrusted = Math.sqrt(magnitudeTrusted);
    const cosine = magnitudeSuspect === 0 || magnitudeTrusted === 0
      ? (magnitudeSuspect === magnitudeTrusted ? 1 : 0)
      : Math.max(-1, Math.min(1, dot / (magnitudeSuspect * magnitudeTrusted)));
    const relativeNorm = Math.abs(suspect.norm - trusted.norm)
      / Math.max(suspect.norm, trusted.norm, Number.MIN_VALUE);
    const passed = cosine >= cosineThreshold && relativeNorm < normThreshold;
    return { passed, cosine, relativeNorm, ...(passed ? {} : { error: "activation_sketch_diverged" }) };
  } catch (error) {
    return {
      passed: false,
      cosine: 0,
      relativeNorm: 1,
      error: error instanceof Error && error.message ? error.message : "activation_sketch_is_malformed",
    };
  }
}

function parseSeed(seed: string): Buffer {
  if (!SEED_PATTERN.test(seed)) throw new Error("activation_seed_is_invalid");
  return Buffer.from(seed, "hex");
}

function validThreshold(value: number, error: string): number {
  if (!Number.isFinite(value) || value <= 0 || value > 1) throw new Error(error);
  return value;
}
