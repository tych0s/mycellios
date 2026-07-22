import type { ComputeMode } from "../contracts/types.js";

export interface DesktopExecutorPolicy {
  computeMode: ComputeMode;
  cpuEligible: boolean;
}

export function normalizeComputeMode(value: unknown): ComputeMode {
  return value === "gpu-only" || value === "cpu-only" ? value : "automatic";
}

export function desktopExecutorPolicy(
  computeMode: ComputeMode,
  gpuPreparationContinuing: boolean,
): DesktopExecutorPolicy {
  return {
    computeMode,
    cpuEligible: computeMode === "cpu-only"
      || (computeMode === "automatic" && !gpuPreparationContinuing),
  };
}
