import {
  selectRuntimeCapacityHardware,
  type HardwareProbe,
  type VerifiedGpuRuntimeEvidence,
} from "../worker/hardware.js";
import { selectAcceleratorPack } from "./accelerator-runtime.js";

export interface DesktopHardwareSelectionInput {
  platform: NodeJS.Platform;
  arch: string;
  osRelease: string;
  cpuModel?: string | undefined;
}

export type DesktopHardwareGpu = HardwareProbe["gpus"][number];

/**
 * Prefer the adapter for which this build has a verified native provider.
 * OS adapter ordering is not stable and commonly puts an integrated GPU
 * before a usable discrete accelerator.
 */
export function selectDesktopHardwareGpu(
  gpus: readonly DesktopHardwareGpu[],
  input: DesktopHardwareSelectionInput,
): DesktopHardwareGpu | undefined {
  return gpus.find((candidate) => selectAcceleratorPack({
    platform: input.platform,
    arch: input.arch,
    osRelease: input.osRelease,
    gpuVendor: candidate.vendor,
    gpuModel: candidate.model,
    cpuModel: input.cpuModel,
  }) !== null) ?? gpus[0];
}

/** Use measured accelerator memory, or an explicit fraction of real system RAM for CPU fallback. */
export function selectWorkerCapacityHardware(
  hardware: HardwareProbe,
  selected: DesktopHardwareGpu | undefined,
  cpuModel?: string | undefined,
  runtime?: VerifiedGpuRuntimeEvidence | undefined,
): DesktopHardwareGpu {
  return selectRuntimeCapacityHardware(hardware, selected, runtime, cpuModel);
}
