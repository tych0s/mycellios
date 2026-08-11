import { createHash } from "node:crypto";
import { open, rename, rm } from "node:fs/promises";
import { z } from "zod";
import type { NodeConfiguration } from "../contracts/node-configuration.js";
import type { PhysicalProbeV1 } from "../distribution/physical-probe.js";

export const nodeDiagnosticSnapshotSchema = z.object({
  schema: z.literal("mycellios-node-diagnostics/1"),
  generatedAt: z.string().datetime(),
  build: z.object({ version: z.string().min(1).max(64) }).strict(),
  host: z.object({
    platform: z.string().min(1).max(32),
    architecture: z.string().min(1).max(32),
    nodeFingerprint: z.string().regex(/^sha256:[0-9a-f]{64}$/),
  }).strict(),
  service: z.object({
    state: z.enum(["starting", "ready", "stopping", "stopped", "failed"]),
    configRevision: z.number().int().positive(),
    updateChannel: z.enum(["dev", "stable"]),
  }).strict(),
  runtime: z.object({
    backend: z.enum(["cuda", "rocm"]),
    pythonVersion: z.string().min(1).max(64),
    cudaVersion: z.string().max(64).nullable(),
    rocmVersion: z.string().max(64).nullable(),
  }).strict(),
  hardware: z.object({
    gpuCount: z.number().int().nonnegative(),
    devices: z.array(z.object({
      index: z.number().int().nonnegative(),
      name: z.string().min(1).max(200),
      totalMemoryMiB: z.number().int().nonnegative(),
      capability: z.tuple([z.number().int(), z.number().int()]).nullable(),
    }).strict()).max(16),
  }).strict(),
  limits: z.object({
    maxConcurrency: z.number().int().positive(),
    maxCpuPercent: z.number().int().min(1).max(100),
    maxRamMiB: z.number().int().positive(),
    maxVramMiB: z.number().int().nonnegative(),
    maxDiskMiB: z.number().int().positive(),
    maxTemperatureC: z.number().int().positive(),
  }).strict(),
  errors: z.array(z.string().regex(/^[a-zA-Z0-9_.:-]{1,120}$/)).max(50),
}).strict();

export type NodeDiagnosticSnapshot = z.infer<typeof nodeDiagnosticSnapshotSchema>;

export function buildNodeDiagnosticSnapshot(input: {
  config: NodeConfiguration;
  version: string;
  state: NodeDiagnosticSnapshot["service"]["state"];
  physicalProbe: PhysicalProbeV1;
  errorCodes?: readonly string[];
  now?: Date;
}): NodeDiagnosticSnapshot {
  const backend = input.physicalProbe.runtime.rocmVersion ? "rocm" as const : "cuda" as const;
  return nodeDiagnosticSnapshotSchema.parse({
    schema: "mycellios-node-diagnostics/1",
    generatedAt: (input.now ?? new Date()).toISOString(),
    build: { version: input.version },
    host: {
      platform: input.physicalProbe.host.platform,
      architecture: input.physicalProbe.host.architecture,
      nodeFingerprint: `sha256:${createHash("sha256")
        .update(`mycellios-node-diagnostic/1\0${input.config.nodeId}`)
        .digest("hex")}`,
    },
    service: {
      state: input.state,
      configRevision: input.config.revision,
      updateChannel: input.config.updateChannel,
    },
    runtime: {
      backend,
      pythonVersion: input.physicalProbe.host.pythonVersion,
      cudaVersion: input.physicalProbe.runtime.cudaVersion,
      rocmVersion: input.physicalProbe.runtime.rocmVersion,
    },
    hardware: {
      gpuCount: input.physicalProbe.devices.length,
      devices: input.physicalProbe.devices.map((device) => ({
        index: device.index,
        name: device.name,
        totalMemoryMiB: Math.floor(device.totalMemoryBytes / (1024 * 1024)),
        capability: device.capability,
      })),
    },
    limits: input.config.limits,
    errors: (input.errorCodes ?? []).map(safeErrorCode),
  });
}

export async function writeNodeDiagnosticSnapshot(
  path: string,
  snapshot: NodeDiagnosticSnapshot,
): Promise<void> {
  const parsed = nodeDiagnosticSnapshotSchema.parse(snapshot);
  const temporary = `${path}.tmp-${process.pid}`;
  try {
    const file = await open(temporary, "w", 0o600);
    try {
      await file.writeFile(`${JSON.stringify(parsed, null, 2)}\n`, "utf8");
      await file.sync();
    } finally {
      await file.close();
    }
    await rename(temporary, path);
  } finally {
    await rm(temporary, { force: true });
  }
}

function safeErrorCode(value: string): string {
  const candidate = value.match(/^[a-zA-Z0-9_.:-]{1,120}/)?.[0];
  if (
    !candidate
    || /(authorization|cookie|credential|password|secret|token|api[-_]?key)/i.test(candidate)
    || /[A-Za-z0-9_-]{32,}/.test(candidate)
  ) return "unclassified_error";
  return candidate;
}
