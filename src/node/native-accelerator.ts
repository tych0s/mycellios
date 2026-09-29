import { release } from "node:os";
import { delimiter, dirname, join } from "node:path";
import { probeHardware } from "../worker/hardware.js";
import type { HeadlessWorkerEnvironment } from "../worker/headless-runtime.js";
import {
  prepareAcceleratorRuntime,
  type AcceleratorProgressEvent,
  type AcceleratorRuntimeResult,
} from "./accelerator-runtime.js";
import { selectDesktopHardwareGpu } from "./hardware-selection.js";

export async function prepareNativeNodeAccelerator(input: {
  configPath: string;
  installRoot: string;
  allowProvisioning: boolean;
  onProgress?: (event: AcceleratorProgressEvent) => void;
}): Promise<AcceleratorRuntimeResult> {
  const hardware = await probeHardware();
  const selected = selectDesktopHardwareGpu(hardware.gpus, {
    platform: process.platform,
    arch: process.arch,
    osRelease: release(),
  });
  return prepareAcceleratorRuntime({
    baseRuntimeRoot: join(input.installRoot, "runtime"),
    userDataPath: join(dirname(dirname(input.configPath)), "Runtime"),
    hardware: {
      platform: process.platform,
      arch: process.arch,
      osRelease: release(),
      gpuVendor: selected?.vendor,
      gpuModel: selected?.model,
      gpuDeviceIndex: selected?.runtimeDeviceIndex,
    },
    allowProvisioning: input.allowProvisioning,
    onProgress: input.onProgress,
  });
}

export function activateNativeNodeAccelerator(
  environment: HeadlessWorkerEnvironment,
  runtime: AcceleratorRuntimeResult,
): HeadlessWorkerEnvironment {
  if (runtime.status !== "gpu-ready") return environment;
  process.env.PATH = [...runtime.pathAdditions, process.env.PATH].filter(Boolean).join(delimiter);
  return { ...environment, pythonExecutable: runtime.pythonExecutable };
}
