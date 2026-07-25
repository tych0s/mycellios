import { execFile } from "node:child_process";
import { hostname, platform, totalmem } from "node:os";
import { promisify } from "node:util";
import type { GpuCapability } from "../contracts/types.js";

const execFileAsync = promisify(execFile);

export interface HardwareProbe {
  hostname: string;
  platform: NodeJS.Platform;
  ramMb: number;
  gpus: HardwareGpu[];
}

export interface HardwareGpu extends Omit<GpuCapability, "offeredVramMb" | "freeOfferedVramMb"> {
  /** Ordinal reported by the vendor runtime itself, never inferred from OS display ordering. */
  runtimeDeviceIndex?: number | undefined;
}

export interface VerifiedGpuRuntimeEvidence {
  status: "gpu-ready";
  backend: "cuda" | "rocm" | "mps" | "xpu";
  deviceName: string;
}

/** Compare stable model identity while ignoring vendor/UI-only decorations. */
export function gpuModelsMatch(left: string, right: string): boolean {
  const exactLeft = left.trim().toLowerCase();
  const exactRight = right.trim().toLowerCase();
  if (!exactLeft || !exactRight) return false;
  if (exactLeft === exactRight) return true;
  const leftKey = normalizedGpuIdentity(left);
  const rightKey = normalizedGpuIdentity(right);
  return leftKey.length >= 3 && leftKey === rightKey;
}

/**
 * Publish GPU memory only after a physical runtime probe confirms the selected
 * adapter. Until then, publish a conservative fraction of real system RAM.
 */
export function selectRuntimeCapacityHardware(
  hardware: HardwareProbe,
  selected: HardwareGpu | undefined,
  runtime: VerifiedGpuRuntimeEvidence | undefined,
  cpuModel?: string | undefined,
): HardwareGpu {
  const selectedBudget = selected
    ? selected.physicalVramMb + (selected.sharedMemoryMb ?? 0)
    : 0;
  if (selected && selectedBudget >= 512 && runtimeMatchesGpu(runtime, selected)) return selected;
  return cpuMemoryCapacityHardware(hardware, cpuModel);
}

export function selectHardwareGpu(
  gpus: readonly HardwareProbe["gpus"][number][],
  preferred?: { id?: string | undefined; vendor: string; model: string } | undefined,
): HardwareProbe["gpus"][number] | undefined {
  if (!preferred) return gpus[0];
  if (preferred.id) {
    const exact = gpus.find((gpu) => gpu.id === preferred.id);
    if (
      exact
      && exact.vendor.toLowerCase() === preferred.vendor.toLowerCase()
      && exact.model.toLowerCase() === preferred.model.toLowerCase()
    ) return exact;
  }
  return gpus.find((gpu) =>
    gpu.vendor.toLowerCase() === preferred.vendor.toLowerCase()
    && gpu.model.toLowerCase() === preferred.model.toLowerCase()
  );
}

export async function probeHardware(): Promise<HardwareProbe> {
  const base = {
    hostname: hostname(),
    platform: platform(),
    ramMb: Math.floor(totalmem() / 1024 / 1024),
  };
  const nvidia = await probeNvidia();
  const windows = platform() === "win32" ? await probeWindowsGpu(base.ramMb) : [];
  const mac = platform() === "darwin" ? await probeMacGpu(base.ramMb) : [];
  const linux = platform() === "linux" ? await probeLinuxGpu(base.ramMb) : [];
  const native = windows.length > 0 ? windows : mac.length > 0 ? mac : linux;
  const gpus = mergeHardwareGpuProbes(native, nvidia);
  if (gpus.length > 0) return { ...base, gpus };
  return {
    ...base,
    gpus: [
      {
        id: "gpu-0",
        vendor: "unknown",
        model: "Unidentified GPU or shared-memory accelerator",
        physicalVramMb: 0,
      },
    ],
  };
}

export function mergeHardwareGpuProbes(
  native: readonly HardwareProbe["gpus"][number][],
  nvidia: readonly HardwareProbe["gpus"][number][],
): HardwareProbe["gpus"] {
  const remainingNvidia = [...nvidia];
  const merged = native.map((gpu) => {
    if (gpu.vendor !== "nvidia") return gpu;
    const index = remainingNvidia.findIndex((candidate) =>
      candidate.vendor === "nvidia" && normalizedGpuModel(candidate.model) === normalizedGpuModel(gpu.model)
    );
    if (index < 0) return gpu;
    const measured = remainingNvidia.splice(index, 1)[0]!;
    return { ...gpu, ...measured };
  });
  merged.push(...remainingNvidia);
  return merged.map((gpu, index) => {
    return {
      ...gpu,
      id: `gpu-${index}`,
    };
  });
}

async function probeMacGpu(systemRamMb: number): Promise<HardwareProbe["gpus"]> {
  try {
    const { stdout } = await execFileAsync("system_profiler", [
      "-json",
      "SPDisplaysDataType",
    ]);
    const payload = JSON.parse(stdout) as { SPDisplaysDataType?: Array<Record<string, unknown>> };
    return (payload.SPDisplaysDataType ?? []).map((device, index) => {
      const model = stringValue(device.sppci_model) ?? stringValue(device._name) ?? "Apple GPU";
      const vendorName = stringValue(device.spdisplays_vendor) ?? model;
      const vendor = classifyVendor(vendorName);
      const unifiedMemory = vendor === "apple" || /apple\s+m\d|apple\s+silicon/i.test(model);
      return {
        id: `gpu-${index}`,
        vendor,
        model,
        physicalVramMb: unifiedMemory ? 0 : parseMemoryMb(device.spdisplays_vram),
        ...(unifiedMemory
          ? {
              unifiedMemory: true,
              sharedMemoryMb: systemRamMb,
            }
          : {}),
      };
    });
  } catch {
    return [];
  }
}

async function probeLinuxGpu(systemRamMb: number): Promise<HardwareProbe["gpus"]> {
  try {
    const { stdout } = await execFileAsync("lspci", ["-mm"]);
    return stdout
      .trim()
      .split("\n")
      .filter((line) => /"(?:VGA compatible controller|3D controller|Display controller)"/i.test(line))
      .map((line, index) => {
        const fields = [...line.matchAll(/"([^"]*)"/g)].map((match) => match[1] ?? "");
        const vendorName = fields[1] ?? "unknown";
        const model = fields[2] ?? "Linux GPU";
        const vendor = classifyVendor(`${vendorName} ${model}`);
        const unifiedMemory =
          vendor === "intel" ||
          (vendor === "amd" && /(integrated|apu|radeon\(tm\).*graphics)/i.test(model));
        return {
          id: `gpu-${index}`,
          vendor,
          model,
          physicalVramMb: 0,
          ...(unifiedMemory
            ? {
                unifiedMemory: true,
                sharedMemoryMb: Math.floor(systemRamMb / 2),
              }
            : {}),
        };
      });
  } catch {
    return [];
  }
}

async function probeNvidia(): Promise<HardwareProbe["gpus"]> {
  try {
    const { stdout } = await execFileAsync("nvidia-smi", [
      "--query-gpu=index,name,memory.total,utilization.gpu,temperature.gpu,power.draw",
      "--format=csv,noheader,nounits",
    ]);
    return stdout
      .trim()
      .split("\n")
      .filter(Boolean)
      .map((line) => {
        const [id, model, memory, utilization, temperature, power] = line
          .split(",")
          .map((field) => field.trim());
        return {
          id: `gpu-${id ?? "0"}`,
          vendor: "nvidia",
          model: model ?? "NVIDIA GPU",
          runtimeDeviceIndex: Number(id ?? 0),
          physicalVramMb: Number(memory ?? 0),
          utilizationPct: Number(utilization ?? 0),
          temperatureC: Number(temperature ?? 0),
          powerW: Number(power ?? 0),
        };
      });
  } catch {
    return [];
  }
}

async function probeWindowsGpu(systemRamMb: number): Promise<HardwareProbe["gpus"]> {
  // Win32_VideoController.AdapterRAM is a uint32 and wraps/truncates modern
  // cards above 4 GiB. Prefer the driver's 64-bit registry value and retain
  // AdapterRAM only as a compatibility fallback for older drivers.
  const script = [
    "$class = 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}'",
    "$registry = @(Get-ChildItem -LiteralPath $class -ErrorAction SilentlyContinue | ForEach-Object {",
    "  $p = Get-ItemProperty -LiteralPath $_.PSPath -ErrorAction SilentlyContinue",
    "  $raw = $p.'HardwareInformation.qwMemorySize'",
    "  $bytes = if ($raw -is [byte[]] -and $raw.Length -ge 8) { [BitConverter]::ToUInt64($raw, 0) } elseif ($null -ne $raw) { [uint64]$raw } else { [uint64]0 }",
    "  $name = if ($p.DriverDesc) { [string]$p.DriverDesc } elseif ($p.'HardwareInformation.AdapterString') { [string]$p.'HardwareInformation.AdapterString' } else { '' }",
    "  $matchingId = if ($p.MatchingDeviceId) { [string]$p.MatchingDeviceId } else { '' }",
    "  if ($name -and $bytes -gt 0) { [pscustomobject]@{ Name = $name; MatchingDeviceId = $matchingId; DedicatedBytes = $bytes } }",
    "})",
    "$devices = @(Get-CimInstance Win32_VideoController | ForEach-Object {",
    "  $deviceName = [string]$_.Name",
    "  $pnpId = [string]$_.PNPDeviceID",
    "  $match = $registry | Where-Object { $vendorDevice = [regex]::Match($_.MatchingDeviceId, '^PCI\\\\VEN_[^&]+&DEV_[^&]+').Value; ($vendorDevice -and $pnpId.StartsWith($vendorDevice, [StringComparison]::OrdinalIgnoreCase)) -or (-not $vendorDevice -and $_.Name -eq $deviceName) } | Sort-Object DedicatedBytes -Descending | Select-Object -First 1",
    "  [pscustomobject]@{ Name = $deviceName; PNPDeviceID = $pnpId; AdapterRAM = $_.AdapterRAM; DedicatedBytes = if ($match) { [uint64]$match.DedicatedBytes } else { [uint64]0 } }",
    "})",
    "$engineSamples = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine -ErrorAction SilentlyContinue | ForEach-Object { [pscustomobject]@{ Name = [string]$_.Name; UtilizationPercentage = [double]$_.UtilizationPercentage } })",
    "[pscustomobject]@{ Devices = $devices; EngineSamples = $engineSamples } | ConvertTo-Json -Compress -Depth 3",
  ].join("; ");
  try {
    const { stdout } = await execFileAsync("powershell.exe", [
      "-NoProfile",
      "-NonInteractive",
      "-Command",
      script,
    ]);
    const raw = JSON.parse(stdout.trim()) as {
      Devices?: WindowsGpuDevice | WindowsGpuDevice[];
      EngineSamples?: WindowsGpuEngineSample | WindowsGpuEngineSample[];
    };
    const devices = raw.Devices === undefined
      ? []
      : Array.isArray(raw.Devices)
        ? raw.Devices
        : [raw.Devices];
    const engineSamples = raw.EngineSamples === undefined
      ? []
      : Array.isArray(raw.EngineSamples)
        ? raw.EngineSamples
        : [raw.EngineSamples];
    // The Windows counter identifies adapters with a DirectX LUID that
    // Win32_VideoController does not expose. Assign aggregate utilization only
    // when there is a single physical adapter, avoiding telemetry leakage
    // between GPUs on hybrid systems. NVIDIA remains covered by nvidia-smi.
    const utilizationPct = devices.length === 1
      ? windowsGpuEngineUtilizationPct(engineSamples)
      : undefined;
    return devices.map((device, index) => {
      const model = device.Name ?? "Windows GPU";
      const vendor = classifyVendor(model);
      const physicalVramMb = windowsGpuPhysicalVramMb(device);
      const unifiedMemory = isLikelyUnifiedMemory(vendor, model, physicalVramMb);
      return {
        id: `gpu-${index}`,
        vendor,
        model,
        physicalVramMb,
        ...(utilizationPct === undefined ? {} : { utilizationPct }),
        ...(unifiedMemory
          ? {
              unifiedMemory: true,
              // Windows does not expose the Task Manager shared-GPU limit via
              // Win32_VideoController. Half of system RAM is the documented
              // upper budget on common UMA systems; the user's offered quota
              // and runtime benchmark remain the tighter limits.
              sharedMemoryMb: Math.floor(systemRamMb / 2),
            }
          : {}),
      };
    });
  } catch {
    return [];
  }
}

interface WindowsGpuDevice {
  Name?: string | undefined;
  AdapterRAM?: number | undefined;
  DedicatedBytes?: number | undefined;
}

export interface WindowsGpuEngineSample {
  Name?: string | undefined;
  UtilizationPercentage?: number | undefined;
}

/**
 * Windows reports one utilization row per process and GPU engine. Task
 * Manager-style device load is the busiest engine after summing its processes,
 * not the sum of every engine (which can exceed 100%).
 */
export function windowsGpuEngineUtilizationPct(
  samples: readonly WindowsGpuEngineSample[],
): number | undefined {
  const engineTotals = new Map<string, number>();
  for (const sample of samples) {
    const name = sample.Name?.trim();
    const value = Number(sample.UtilizationPercentage);
    if (!name || !Number.isFinite(value) || value < 0) continue;
    const engine = name.match(/luid_[^_]+_[^_]+_phys_\d+_eng_\d+_engtype_.+$/i)?.[0];
    if (!engine) continue;
    const key = engine.toLowerCase();
    engineTotals.set(key, (engineTotals.get(key) ?? 0) + value);
  }
  if (engineTotals.size === 0) return undefined;
  return Math.min(100, Math.max(0, ...engineTotals.values()));
}

export function windowsGpuPhysicalVramMb(device: {
  AdapterRAM?: number | undefined;
  DedicatedBytes?: number | undefined;
}): number {
  const dedicatedBytes = Number(device.DedicatedBytes ?? 0);
  const adapterBytes = Number(device.AdapterRAM ?? 0);
  const bytes = Number.isFinite(dedicatedBytes) && dedicatedBytes > 0
    ? dedicatedBytes
    : Number.isFinite(adapterBytes) && adapterBytes > 0
      ? adapterBytes
      : 0;
  return Math.max(0, Math.floor(bytes / 1024 / 1024));
}

function isLikelyUnifiedMemory(vendor: string, model: string, physicalVramMb: number): boolean {
  if (vendor === "apple" || vendor === "intel") return true;
  if (vendor !== "amd" || physicalVramMb > 2_048) return false;
  return /(radeon\(tm\).*graphics|radeon.*\b[6-9]\d0m\b|vega)/i.test(model);
}

function classifyVendor(model: string): string {
  const normalized = model.toLowerCase();
  if (normalized.includes("nvidia")) return "nvidia";
  if (normalized.includes("amd") || normalized.includes("radeon")) return "amd";
  if (normalized.includes("intel")) return "intel";
  if (normalized.includes("apple")) return "apple";
  return "unknown";
}

function normalizedGpuModel(model: string): string {
  return model.toLowerCase().replace(/\b(?:nvidia|corporation|inc\.?|amd|advanced micro devices)\b/g, "").replace(/[^a-z0-9]+/g, "");
}

function normalizedGpuIdentity(model: string): string {
  return model
    .toLowerCase()
    .replace(/\(tm\)|\(r\)/g, " ")
    .replace(/\b(?:advanced micro devices|nvidia|amd|ati|intel|apple|radeon|geforce|graphics|graphic|gpu|display|adapter|series|corporation|inc)\b/g, " ")
    .replace(/[^a-z0-9]+/g, "");
}

function runtimeMatchesGpu(
  runtime: VerifiedGpuRuntimeEvidence | undefined,
  gpu: HardwareGpu,
): boolean {
  if (!runtime || runtime.status !== "gpu-ready") return false;
  const vendor = gpu.vendor.toLowerCase();
  if (runtime.backend === "mps") {
    return vendor === "apple";
  }
  const expectedVendor = runtime.backend === "xpu"
    ? "intel"
    : runtime.backend === "cuda"
      ? "nvidia"
      : "amd";
  return vendor === expectedVendor && gpuModelsMatch(runtime.deviceName, gpu.model);
}

function cpuMemoryCapacityHardware(
  hardware: HardwareProbe,
  cpuModel?: string | undefined,
): HardwareGpu {
  const cpuMemoryBudget = Math.max(512, Math.floor(hardware.ramMb / 4));
  return {
    id: "cpu-memory",
    vendor: "cpu",
    model: `${cpuModel?.trim() || "CPU"} · system-memory fallback`,
    physicalVramMb: 0,
    sharedMemoryMb: cpuMemoryBudget,
    unifiedMemory: true,
  };
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function parseMemoryMb(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) return Math.max(0, Math.floor(value));
  if (typeof value !== "string") return 0;
  const match = value.match(/([\d.]+)\s*(GB|MB)/i);
  if (!match) return 0;
  const amount = Number(match[1]);
  if (!Number.isFinite(amount)) return 0;
  return Math.round(amount * (match[2]?.toUpperCase() === "GB" ? 1_024 : 1));
}
