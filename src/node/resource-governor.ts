import { opendir, stat } from "node:fs/promises";
import { resolve } from "node:path";
import type { NodeConfiguration } from "../contracts/node-configuration.js";
import { probeHardware, type HardwareProbe } from "../worker/hardware.js";

export interface NodeResourceObservation {
  cpuPercent: number;
  ramMiB: number;
  diskMiB: number;
  temperatureC: number | null;
}

export type NodeResourceLimitCode =
  | "node_cpu_limit_exceeded"
  | "node_ram_limit_exceeded"
  | "node_disk_limit_exceeded"
  | "node_temperature_limit_exceeded"
  | "node_resource_probe_failed";

export class NodeResourceGovernor {
  private timer: NodeJS.Timeout | null = null;
  private evaluating = false;
  private stopped = false;

  constructor(
    private readonly limits: NodeConfiguration["limits"],
    private readonly observe: () => Promise<NodeResourceObservation>,
    private readonly onBreach: (
      code: NodeResourceLimitCode,
      observation: NodeResourceObservation | null,
    ) => Promise<void> | void,
    private readonly intervalMs = 5_000,
  ) {
    if (!Number.isSafeInteger(intervalMs) || intervalMs < 100) {
      throw new Error("node_resource_governor_interval_is_invalid");
    }
  }

  start(): void {
    if (this.timer || this.stopped) return;
    this.timer = setInterval(() => void this.tick(), this.intervalMs);
    this.timer.unref();
  }

  async evaluate(): Promise<NodeResourceLimitCode | null> {
    const observation = await this.observe();
    const code = breachedLimit(this.limits, observation);
    if (code) await this.onBreach(code, observation);
    return code;
  }

  stop(): void {
    this.stopped = true;
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  private async tick(): Promise<void> {
    if (this.evaluating || this.stopped) return;
    this.evaluating = true;
    try {
      const code = await this.evaluate();
      if (code) this.stop();
    } catch {
      await this.onBreach("node_resource_probe_failed", null);
      this.stop();
    } finally {
      this.evaluating = false;
    }
  }
}

export function breachedLimit(
  limits: NodeConfiguration["limits"],
  observation: NodeResourceObservation,
): NodeResourceLimitCode | null {
  if (observation.temperatureC !== null && observation.temperatureC > limits.maxTemperatureC) {
    return "node_temperature_limit_exceeded";
  }
  if (observation.ramMiB > limits.maxRamMiB) return "node_ram_limit_exceeded";
  if (observation.diskMiB > limits.maxDiskMiB) return "node_disk_limit_exceeded";
  if (observation.cpuPercent > limits.maxCpuPercent) return "node_cpu_limit_exceeded";
  return null;
}

export function createNodeResourceObserver(
  cachePath: string,
  hardwareProbe: () => Promise<HardwareProbe> = probeHardware,
): () => Promise<NodeResourceObservation> {
  const root = resolve(cachePath);
  let previousCpu = process.cpuUsage();
  let previousTime = process.hrtime.bigint();
  return async () => {
    const [diskBytes, hardware] = await Promise.all([
      directoryBytes(root),
      hardwareProbe(),
    ]);
    const cpu = process.cpuUsage();
    const now = process.hrtime.bigint();
    const elapsedMicros = Math.max(1, Number(now - previousTime) / 1_000);
    const usedMicros = Math.max(
      0,
      cpu.user + cpu.system - previousCpu.user - previousCpu.system,
    );
    previousCpu = cpu;
    previousTime = now;
    const temperatures = hardware.gpus
      .map((gpu) => gpu.temperatureC)
      .filter((value): value is number => value !== undefined && Number.isFinite(value));
    return {
      cpuPercent: Math.max(0, Math.min(100, (usedMicros / elapsedMicros) * 100)),
      ramMiB: process.memoryUsage().rss / (1024 * 1024),
      diskMiB: diskBytes / (1024 * 1024),
      temperatureC: temperatures.length > 0 ? Math.max(...temperatures) : null,
    };
  };
}

async function directoryBytes(path: string): Promise<number> {
  let total = 0;
  let directory;
  try { directory = await opendir(path); } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return 0;
    throw error;
  }
  for await (const entry of directory) {
    const child = resolve(path, entry.name);
    if (entry.isSymbolicLink()) continue;
    if (entry.isDirectory()) total += await directoryBytes(child);
    else if (entry.isFile()) total += (await stat(child)).size;
  }
  return total;
}
