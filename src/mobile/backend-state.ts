/// <reference types="@webgpu/types" />

export type EffectiveMobileBackend = "webgpu" | "cpu";

export interface MobileBackendExecution<T> {
  backend: EffectiveMobileBackend;
  value: T;
}

export class MobileExecutionCancelledError extends Error {
  override readonly name = "MobileExecutionCancelledError";

  constructor(message = "Mobile compute was cancelled") {
    super(message);
  }
}

export function throwIfMobileExecutionCancelled(signal?: AbortSignal): void {
  if (!signal?.aborted) return;
  throw signal.reason instanceof MobileExecutionCancelledError
    ? signal.reason
    : new MobileExecutionCancelledError();
}

export function isMobileExecutionCancelled(error: unknown): boolean {
  return error instanceof MobileExecutionCancelledError;
}

export async function requestMobileGpuAdapter(
  gpu: Pick<GPU, "requestAdapter">,
): Promise<GPUAdapter | null> {
  try {
    const compatible = await gpu.requestAdapter({ featureLevel: "compatibility" });
    if (compatible) return compatible;
  } catch {
    // Older implementations can reject an option they do not yet recognize.
  }

  try {
    return await gpu.requestAdapter({ powerPreference: "high-performance" });
  } catch {
    return null;
  }
}

export function mobileGpuAdapterLabel(
  info: Pick<GPUAdapterInfo, "vendor" | "architecture" | "device" | "description">,
): string {
  if (info.description.trim()) return info.description.trim();
  const identifiers = [info.vendor, info.architecture, info.device]
    .map((value) => value.trim())
    .filter((value, index, values) => value.length > 0 && values.indexOf(value) === index);
  return identifiers.join(" · ") || "WebGPU adapter";
}

export function mobileCpuFallbackLabel(detectedGpuLabel: string | null): string {
  return detectedGpuLabel
    ? `CPU fallback · ${detectedGpuLabel}`
    : "Universal CPU fallback";
}

export async function runWithMobileBackendFallback<T>(input: {
  backend: EffectiveMobileBackend;
  runGpu: () => Promise<T>;
  runCpu: () => Promise<T>;
  onGpuFailure: (error: unknown) => void;
  signal?: AbortSignal;
}): Promise<MobileBackendExecution<T>> {
  throwIfMobileExecutionCancelled(input.signal);
  if (input.backend === "webgpu") {
    try {
      const value = await input.runGpu();
      throwIfMobileExecutionCancelled(input.signal);
      return { backend: "webgpu", value };
    } catch (error) {
      throwIfMobileExecutionCancelled(input.signal);
      input.onGpuFailure(error);
    }
  }
  throwIfMobileExecutionCancelled(input.signal);
  const value = await input.runCpu();
  throwIfMobileExecutionCancelled(input.signal);
  return { backend: "cpu", value };
}
