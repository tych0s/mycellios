import { createHash } from "node:crypto";
import { z } from "zod";

import { sha256CanonicalEvidence } from "../core/json.js";

export const MODEL_CERTIFICATION_HARNESS_SCHEMA =
  "mycellios-model-certification-harness/1" as const;

const sha256Schema = z.string().regex(/^sha256:[0-9a-f]{64}$/);
const revisionSchema = z.string().regex(/^[0-9a-f]{40}$/);
const identifierSchema = z.string().min(1).max(256).regex(/^[A-Za-z0-9][A-Za-z0-9._+:/-]*$/);
const tokenSchema = z.number().int().nonnegative().max(4_294_967_295);
const promptCaseSchema = z.object({
  id: identifierSchema,
  prompt: z.string().min(1).max(1_000_000),
  seed: z.number().int().nonnegative().safe(),
  maximumTokens: z.number().int().positive().max(10_000_000),
  referenceTokens: z.array(tokenSchema).max(10_000_000),
  candidateTokens: z.array(tokenSchema).max(10_000_000),
}).strict();

const harnessInputSchema = z.object({
  distributionManifestId: sha256Schema,
  componentManifestId: sha256Schema,
  adapterContractId: sha256Schema,
  sourceRevision: revisionSchema,
  evidenceClass: z.enum(["automatic", "physical"]),
  measuredAt: z.string().datetime({ offset: true }),
  hardware: z.object({
    fingerprint: sha256Schema.nullable(),
    backend: z.enum(["cpu", "cuda", "rocm", "metal", "vulkan"]),
    deviceFamily: identifierSchema,
    memoryBytes: z.number().int().positive().safe(),
  }).strict(),
  codecs: z.array(identifierSchema).min(1).max(32),
  cases: z.array(promptCaseSchema).min(1).max(10_000),
  performance: z.object({
    ttftMs: z.number().nonnegative().finite(),
    tpotMs: z.number().nonnegative().finite(),
    peakMemoryBytes: z.number().int().positive().safe(),
  }).strict(),
}).strict().superRefine((input, context) => {
  if (input.evidenceClass === "physical" && input.hardware.fingerprint === null) {
    context.addIssue({ code: "custom", message: "model_certification_physical_hardware_fingerprint_is_required", path: ["hardware", "fingerprint"] });
  }
  if (new Set(input.codecs).size !== input.codecs.length) context.addIssue({ code: "custom", message: "model_certification_harness_codecs_are_duplicated", path: ["codecs"] });
  if (new Set(input.cases.map(({ id }) => id)).size !== input.cases.length) context.addIssue({ code: "custom", message: "model_certification_harness_cases_are_duplicated", path: ["cases"] });
});

export type ModelCertificationHarnessInput = z.infer<typeof harnessInputSchema>;

export interface ModelCertificationHarnessReceipt {
  schema: typeof MODEL_CERTIFICATION_HARNESS_SCHEMA;
  classification: "software-candidate" | "physical-candidate";
  distributionManifestId: string;
  componentManifestId: string;
  adapterContractId: string;
  sourceRevision: string;
  measuredAt: string;
  hardware: ModelCertificationHarnessInput["hardware"];
  codecs: string[];
  cases: Array<{
    id: string;
    promptSha256: string;
    seed: number;
    maximumTokens: number;
    referenceTokenCount: number;
    candidateTokenCount: number;
    firstMismatchIndex: number | null;
    exact: boolean;
  }>;
  performance: ModelCertificationHarnessInput["performance"];
  exact: boolean;
  receiptId: string;
}

export function runModelCertificationHarness(value: unknown): ModelCertificationHarnessReceipt {
  const input = harnessInputSchema.parse(value);
  const cases = [...input.cases]
    .sort((left, right) => left.id.localeCompare(right.id))
    .map((entry) => {
      const firstMismatchIndex = mismatchIndex(entry.referenceTokens, entry.candidateTokens);
      return {
        id: entry.id,
        promptSha256: `sha256:${createHash("sha256").update(entry.prompt).digest("hex")}`,
        seed: entry.seed,
        maximumTokens: entry.maximumTokens,
        referenceTokenCount: entry.referenceTokens.length,
        candidateTokenCount: entry.candidateTokens.length,
        firstMismatchIndex,
        exact: firstMismatchIndex === null,
      };
    });
  const body = {
    schema: MODEL_CERTIFICATION_HARNESS_SCHEMA,
    classification: input.evidenceClass === "physical" ? "physical-candidate" : "software-candidate",
    distributionManifestId: input.distributionManifestId,
    componentManifestId: input.componentManifestId,
    adapterContractId: input.adapterContractId,
    sourceRevision: input.sourceRevision,
    measuredAt: input.measuredAt,
    hardware: input.hardware,
    codecs: [...input.codecs].sort(),
    cases,
    performance: input.performance,
    exact: cases.every(({ exact }) => exact),
  } as const;
  return { ...body, receiptId: sha256CanonicalEvidence(body) };
}

function mismatchIndex(reference: readonly number[], candidate: readonly number[]): number | null {
  const length = Math.max(reference.length, candidate.length);
  for (let index = 0; index < length; index += 1) {
    if (reference[index] !== candidate[index]) return index;
  }
  return null;
}
