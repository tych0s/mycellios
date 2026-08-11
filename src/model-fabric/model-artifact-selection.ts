import {
  parseModelDistributionManifest,
  type ModelDistributionManifest,
} from "../contracts/model-distribution-manifest.js";

export interface ModelArtifactAssignment {
  layerStart: number;
  layerEnd: number;
  expertStart: number | null;
  expertEnd: number | null;
  includeInput: boolean;
  includeOutput: boolean;
}

export interface SelectedModelArtifacts {
  manifestId: string;
  artifactIds: string[];
  totalBytes: number;
}

/**
 * Converts a certified model distribution into the exact byte allowlist for a
 * stage. Shared tensors are always present; input/output ownership is explicit;
 * range artifacts may never be sliced implicitly or widened to adjacent layers.
 */
export function selectModelArtifactsForAssignment(
  value: unknown,
  assignment: ModelArtifactAssignment,
): SelectedModelArtifacts {
  const manifest = parseModelDistributionManifest(value);
  validateAssignment(manifest, assignment);
  const selected = manifest.artifacts.filter((artifact) => {
    if (artifact.role === "shared") return true;
    if (artifact.role === "input") return assignment.includeInput;
    if (artifact.role === "output") return assignment.includeOutput;
    if (artifact.layerRange !== null) {
      return artifact.layerRange.start >= assignment.layerStart
        && artifact.layerRange.end <= assignment.layerEnd;
    }
    if (artifact.expertRange !== null) {
      return assignment.expertStart !== null
        && assignment.expertEnd !== null
        && artifact.expertRange.start >= assignment.expertStart
        && artifact.expertRange.end <= assignment.expertEnd;
    }
    return false;
  });
  assertExactCoverage(
    selected.flatMap((artifact) => artifact.layerRange ? [artifact.layerRange] : []),
    assignment.layerStart,
    assignment.layerEnd,
    "layer",
  );
  if (assignment.expertStart !== null && assignment.expertEnd !== null) {
    assertExactCoverage(
      selected.flatMap((artifact) => artifact.expertRange ? [artifact.expertRange] : []),
      assignment.expertStart,
      assignment.expertEnd,
      "expert",
    );
  }
  return {
    manifestId: manifest.manifestId,
    artifactIds: selected.map(({ id }) => id),
    totalBytes: selected.reduce((sum, artifact) => sum + artifact.sizeBytes, 0),
  };
}

function validateAssignment(
  manifest: ModelDistributionManifest,
  assignment: ModelArtifactAssignment,
): void {
  if (!Number.isSafeInteger(assignment.layerStart)
    || !Number.isSafeInteger(assignment.layerEnd)
    || assignment.layerStart < 0
    || assignment.layerEnd <= assignment.layerStart
    || assignment.layerEnd > manifest.layerCount) {
    throw new Error("model_artifact_layer_assignment_is_invalid");
  }
  const hasExperts = assignment.expertStart !== null || assignment.expertEnd !== null;
  if (hasExperts) {
    if (manifest.expertCount === null
      || assignment.expertStart === null
      || assignment.expertEnd === null
      || !Number.isSafeInteger(assignment.expertStart)
      || !Number.isSafeInteger(assignment.expertEnd)
      || assignment.expertStart < 0
      || assignment.expertEnd <= assignment.expertStart
      || assignment.expertEnd > manifest.expertCount) {
      throw new Error("model_artifact_expert_assignment_is_invalid");
    }
  } else if (manifest.expertCount !== null) {
    throw new Error("model_artifact_expert_assignment_is_required");
  }
}

function assertExactCoverage(
  ranges: Array<{ start: number; end: number }>,
  start: number,
  end: number,
  kind: "layer" | "expert",
): void {
  const ordered = [...ranges].sort((left, right) => left.start - right.start);
  let cursor = start;
  for (const range of ordered) {
    if (range.start !== cursor) throw new Error(`model_artifact_${kind}_assignment_not_package_aligned`);
    cursor = range.end;
  }
  if (cursor !== end) throw new Error(`model_artifact_${kind}_assignment_not_package_aligned`);
}
