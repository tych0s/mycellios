import { z } from "zod";

import { sha256CanonicalEvidence } from "../core/json.js";

export const MODEL_DISTRIBUTION_MANIFEST_SCHEMA =
  "mycellios-model-distribution/1" as const;

const sha256Schema = z.string().regex(/^sha256:[0-9a-f]{64}$/);
const revisionSchema = z.string().regex(/^[0-9a-f]{40}$/);
const identifierSchema = z.string().min(1).max(256).regex(/^[A-Za-z0-9][A-Za-z0-9._+/-]*$/);
const artifactPathSchema = z.string().min(1).max(1_024).refine(isSafeRelativePath, {
  message: "model_distribution_artifact_path_is_unsafe",
});
const sourceUrlSchema = z.string().url().max(4_096).refine((value) => {
  const url = new URL(value);
  return url.protocol === "https:" && url.username === "" && url.password === "";
}, { message: "model_distribution_artifact_source_is_unsafe" });
const rangeSchema = z.object({
  start: z.number().int().nonnegative().safe(),
  end: z.number().int().positive().safe(),
}).strict().refine(({ start, end }) => start < end, {
  message: "model_distribution_artifact_range_is_invalid",
});

const artifactSchema = z.object({
  id: identifierSchema,
  role: z.enum(["shared", "input", "output", "layer-range", "expert-range"]),
  path: artifactPathSchema,
  sha256: sha256Schema,
  sizeBytes: z.number().int().positive().safe(),
  sourceUrl: sourceUrlSchema,
  layerRange: rangeSchema.nullable(),
  expertRange: rangeSchema.nullable(),
}).strict().superRefine((artifact, context) => {
  if ((artifact.role === "layer-range") !== (artifact.layerRange !== null)) {
    context.addIssue({ code: "custom", message: "model_distribution_layer_range_role_mismatch" });
  }
  if ((artifact.role === "expert-range") !== (artifact.expertRange !== null)) {
    context.addIssue({ code: "custom", message: "model_distribution_expert_range_role_mismatch" });
  }
});

const unsignedManifestSchema = z.object({
  schema: z.literal(MODEL_DISTRIBUTION_MANIFEST_SCHEMA),
  model: z.object({
    id: identifierSchema,
    revision: revisionSchema,
    architecture: identifierSchema,
  }).strict(),
  tokenizer: z.object({
    id: identifierSchema,
    revision: revisionSchema,
  }).strict(),
  format: z.enum(["safetensors", "gguf", "mycellios-packed"]),
  quantization: z.object({
    scheme: z.enum(["none", "gptq", "awq", "gguf", "fp8", "int8"]),
    bits: z.number().int().min(1).max(64).nullable(),
    groupSize: z.number().int().positive().max(65_536).nullable(),
  }).strict(),
  tensorAbi: identifierSchema,
  layerCount: z.number().int().positive().max(1_000_000),
  expertCount: z.number().int().positive().max(1_000_000).nullable(),
  license: z.object({
    spdx: z.string().min(1).max(128).regex(/^[A-Za-z0-9][A-Za-z0-9-.+]*$/),
    sourceUrl: sourceUrlSchema,
  }).strict(),
  artifacts: z.array(artifactSchema).min(3).max(100_000),
}).strict().superRefine(validateArtifactOwnership);

export const modelDistributionManifestSchema = unsignedManifestSchema.extend({
  manifestId: sha256Schema,
}).strict();

export type UnsignedModelDistributionManifest = z.infer<typeof unsignedManifestSchema>;
export type ModelDistributionManifest = z.infer<typeof modelDistributionManifestSchema>;
export type ModelDistributionManifestBuildInput = Omit<
  UnsignedModelDistributionManifest,
  "schema"
>;

export function buildModelDistributionManifest(
  input: ModelDistributionManifestBuildInput,
): ModelDistributionManifest {
  const unsigned = unsignedManifestSchema.parse({
    schema: MODEL_DISTRIBUTION_MANIFEST_SCHEMA,
    ...input,
    artifacts: [...input.artifacts].sort((left, right) => left.id.localeCompare(right.id)),
  });
  return modelDistributionManifestSchema.parse({
    ...unsigned,
    manifestId: sha256CanonicalEvidence(unsigned),
  });
}

export function parseModelDistributionManifest(value: unknown): ModelDistributionManifest {
  const manifest = modelDistributionManifestSchema.parse(value);
  const { manifestId, ...unsigned } = manifest;
  if (sha256CanonicalEvidence(unsigned) !== manifestId) {
    throw new Error("model_distribution_manifest_identity_mismatch");
  }
  return manifest;
}

function validateArtifactOwnership(
  manifest: z.infer<typeof unsignedManifestSchema>,
  context: z.RefinementCtx,
): void {
  const ids = new Set<string>();
  const paths = new Set<string>();
  for (let index = 0; index < manifest.artifacts.length; index += 1) {
    const artifact = manifest.artifacts[index]!;
    if (ids.has(artifact.id)) context.addIssue({ code: "custom", message: "model_distribution_artifact_id_is_duplicated", path: ["artifacts", index, "id"] });
    if (paths.has(artifact.path)) context.addIssue({ code: "custom", message: "model_distribution_artifact_path_is_duplicated", path: ["artifacts", index, "path"] });
    ids.add(artifact.id);
    paths.add(artifact.path);
    if (index > 0 && manifest.artifacts[index - 1]!.id.localeCompare(artifact.id) > 0) {
      context.addIssue({ code: "custom", message: "model_distribution_artifacts_are_not_canonical", path: ["artifacts", index] });
    }
  }
  for (const role of ["input", "output"] as const) {
    if (manifest.artifacts.filter((artifact) => artifact.role === role).length !== 1) {
      context.addIssue({ code: "custom", message: `model_distribution_${role}_artifact_is_not_unique`, path: ["artifacts"] });
    }
  }
  validateCoverage(manifest.artifacts.flatMap((artifact) => artifact.layerRange ? [artifact.layerRange] : []), manifest.layerCount, "layer", context);
  const expertRanges = manifest.artifacts.flatMap((artifact) => artifact.expertRange ? [artifact.expertRange] : []);
  if (manifest.expertCount === null) {
    if (expertRanges.length > 0) context.addIssue({ code: "custom", message: "model_distribution_dense_model_has_expert_artifacts", path: ["artifacts"] });
  } else {
    validateCoverage(expertRanges, manifest.expertCount, "expert", context);
  }
}

function validateCoverage(
  ranges: Array<{ start: number; end: number }>,
  expectedEnd: number,
  kind: "layer" | "expert",
  context: z.RefinementCtx,
): void {
  const ordered = [...ranges].sort((left, right) => left.start - right.start || left.end - right.end);
  let cursor = 0;
  for (const range of ordered) {
    if (range.start !== cursor) {
      context.addIssue({ code: "custom", message: `model_distribution_${kind}_ownership_has_gap_or_overlap`, path: ["artifacts"] });
      return;
    }
    cursor = range.end;
  }
  if (cursor !== expectedEnd) context.addIssue({ code: "custom", message: `model_distribution_${kind}_ownership_is_incomplete`, path: ["artifacts"] });
}

function isSafeRelativePath(value: string): boolean {
  return !value.startsWith("/")
    && !value.startsWith("\\")
    && !/^[A-Za-z]:/.test(value)
    && !value.includes("\\")
    && value.split("/").every((part) => part !== "" && part !== "." && part !== "..");
}
