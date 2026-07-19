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
