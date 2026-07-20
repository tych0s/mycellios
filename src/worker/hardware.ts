import { execFile } from "node:child_process";
import { hostname, platform, totalmem } from "node:os";
import { promisify } from "node:util";
import type { GpuCapability } from "../contracts/types.js";

const execFileAsync = promisify(execFile);

export interface HardwareProbe {
  hostname: string;
  platform: NodeJS.Platform;
  ramMb: number;
  gpus: Array<Omit<GpuCapability, "offeredVramMb" | "freeOfferedVramMb">>;
}

export async function probeHardware(): Promise<HardwareProbe> {
  const base = {
    hostname: hostname(),
    platform: platform(),
    ramMb: Math.floor(totalmem() / 1024 / 1024),
  };
  const nvidia = await probeNvidia();
  if (nvidia.length > 0) return { ...base, gpus: nvidia };
  const windows = platform() === "win32" ? await probeWindowsGpu(base.ramMb) : [];
  if (windows.length > 0) return { ...base, gpus: windows };
  const mac = platform() === "darwin" ? await probeMacGpu(base.ramMb) : [];
  if (mac.length > 0) return { ...base, gpus: mac };
  const linux = platform() === "linux" ? await probeLinuxGpu(base.ramMb) : [];
  if (linux.length > 0) return { ...base, gpus: linux };
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
  const script =
    "Get-CimInstance Win32_VideoController | " +
    "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress";
  try {
    const { stdout } = await execFileAsync("powershell.exe", [
      "-NoProfile",
      "-NonInteractive",
      "-Command",
      script,
    ]);
    const raw = JSON.parse(stdout.trim()) as
      | { Name?: string; AdapterRAM?: number }
      | Array<{ Name?: string; AdapterRAM?: number }>;
    const devices = Array.isArray(raw) ? raw : [raw];
    return devices.map((device, index) => {
      const model = device.Name ?? "Windows GPU";
      const vendor = classifyVendor(model);
      const physicalVramMb = Math.floor(Number(device.AdapterRAM ?? 0) / 1024 / 1024);
      const unifiedMemory = isLikelyUnifiedMemory(vendor, model, physicalVramMb);
      return {
        id: `gpu-${index}`,
        vendor,
        model,
        physicalVramMb,
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
