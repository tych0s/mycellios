import { execFile } from "node:child_process";
import { z } from "zod";
import type { LlmfitAdvisory, LlmfitModelAdvisory } from "../contracts/types.js";

const MAX_OUTPUT_BYTES = 4 * 1024 * 1024;

const gpuSchema = z.object({
  name: z.string().min(1),
  backend: z.string().min(1).optional(),
  vram_gb: z.number().nonnegative().nullable().optional(),
  unified_memory: z.boolean().optional(),
});

const systemSchema = z.object({
  total_ram_gb: z.number().nonnegative(),
  available_ram_gb: z.number().nonnegative(),
  cpu_cores: z.number().int().nonnegative(),
  cpu_name: z.string().nullable().optional(),
  has_gpu: z.boolean(),
  gpu_vram_gb: z.number().nonnegative().nullable().optional(),
  gpu_name: z.string().nullable().optional(),
  gpu_count: z.number().int().nonnegative(),
  backend: z.string().min(1),
  unified_memory: z.boolean().optional(),
  gpus: z.array(gpuSchema).optional(),
});

const systemEnvelopeSchema = z.object({ system: systemSchema });

const modelSchema = z.object({
  name: z.string().min(1),
  local-model-runtime_name: z.string().min(1).nullable().optional(),
  fit_level: z.string().min(1),
  run_mode: z.string().min(1),
  runtime: z.string().min(1).nullable().optional(),
  best_quant: z.string().min(1).nullable().optional(),
  estimated_tps: z.number().positive().nullable().optional(),
  measured_tps: z.number().positive().nullable().optional(),
  memory_required_gb: z.number().nonnegative().nullable().optional(),
  usable_context: z.number().int().positive().nullable().optional(),
});

const modelEnvelopeSchema = z.object({ models: z.array(modelSchema) });

export interface LlmfitProbeOptions {
  executable: string;
  arguments: string[];
  timeoutMs: number;
  model?: string | undefined;
  maxContext?: number | undefined;
}

export interface LlmfitProbeResult {
  advisory: LlmfitAdvisory;
  warnings: string[];
}

export type LlmfitRunner = (
  executable: string,
  arguments_: string[],
  timeoutMs: number,
) => Promise<string>;

export async function probeLlmfit(
  options: LlmfitProbeOptions,
  runner: LlmfitRunner = runLlmfit,
): Promise<LlmfitProbeResult> {
  const systemOutput = await runner(
    options.executable,
    [...options.arguments, "--json", "system"],
    options.timeoutMs,
  );
  const system = systemEnvelopeSchema.parse(parseJson(systemOutput, "system")).system;
  const advisory = systemAdvisory(system);
  const warnings: string[] = [];

  if (options.model) {
    const modelArguments = [...options.arguments];
    if (options.maxContext !== undefined) {
      modelArguments.push("--max-context", String(options.maxContext));
    }
    modelArguments.push("--json", "info", options.model);
    try {
      const modelOutput = await runner(options.executable, modelArguments, options.timeoutMs);
      const envelope = modelEnvelopeSchema.parse(parseJson(modelOutput, "model info"));
      const model = selectModel(envelope.models, options.model);
      if (model) advisory.model = modelAdvisory(model, options.model);
      else warnings.push(`llmfit did not return a model matching ${options.model}`);
    } catch (error) {
      warnings.push(`llmfit model inspection failed: ${errorText(error)}`);
    }
  }

  return { advisory, warnings };
}

export function llmfitHardwareFallback(
  advisory: LlmfitAdvisory,
): {
  id: string;
  vendor: string;
  model: string;
  physicalVramMb: number;
  sharedMemoryMb?: number | undefined;
  unifiedMemory?: boolean | undefined;
} | null {
  const gpu = advisory.gpus[0];
  if (!gpu) return null;
  return {
    id: "gpu-0",
    vendor: classifyVendor(gpu.name),
    model: gpu.name,
    physicalVramMb: gpu.unifiedMemory ? 0 : gpu.vramMb,
    ...(gpu.unifiedMemory
      ? { sharedMemoryMb: gpu.vramMb, unifiedMemory: true }
      : {}),
  };
}

function systemAdvisory(system: z.infer<typeof systemSchema>): LlmfitAdvisory {
  const gpus = (system.gpus ?? []).map((gpu) => ({
    name: gpu.name,
    backend: gpu.backend ?? system.backend,
    vramMb: gibToMib(gpu.vram_gb ?? 0),
    unifiedMemory: gpu.unified_memory ?? system.unified_memory ?? false,
  }));
  if (gpus.length === 0 && system.has_gpu && system.gpu_name) {
    gpus.push({
      name: system.gpu_name,
      backend: system.backend,
      vramMb: gibToMib(system.gpu_vram_gb ?? 0),
      unifiedMemory: system.unified_memory ?? false,
    });
  }
  return {
    source: "llmfit",
    scope: "host",
    backend: system.backend,
    cpuName: system.cpu_name ?? "",
    cpuCores: system.cpu_cores,
    totalRamMb: gibToMib(system.total_ram_gb),
    availableRamMb: gibToMib(system.available_ram_gb),
    gpuCount: system.gpu_count,
    gpus,
  };
}

function modelAdvisory(
  model: z.infer<typeof modelSchema>,
  requestedModel: string,
): LlmfitModelAdvisory {
  return {
    requestedModel,
    resolvedModel: model.name,
    fitLevel: model.fit_level,
    runMode: model.run_mode,
    ...(model.runtime ? { runtime: model.runtime } : {}),
    ...(model.best_quant ? { bestQuant: model.best_quant } : {}),
    ...(model.estimated_tps
      ? { estimatedTokensPerSecond: model.estimated_tps }
      : {}),
    ...(model.measured_tps
      ? { measuredTokensPerSecond: model.measured_tps }
      : {}),
    ...(model.memory_required_gb !== null && model.memory_required_gb !== undefined
      ? { memoryRequiredMb: gibToMib(model.memory_required_gb) }
      : {}),
    ...(model.usable_context ? { usableContext: model.usable_context } : {}),
  };
}

function selectModel(
  models: Array<z.infer<typeof modelSchema>>,
  requestedModel: string,
): z.infer<typeof modelSchema> | undefined {
  const requested = requestedModel.trim().toLowerCase();
  return (
    models.find((model) => model.name.toLowerCase() === requested) ??
    models.find((model) => model.local-model-runtime_name?.toLowerCase() === requested) ??
    (models.length === 1 ? models[0] : undefined)
  );
}

function parseJson(output: string, label: string): unknown {
  try {
    return JSON.parse(output) as unknown;
  } catch {
    throw new Error(`llmfit returned invalid JSON for ${label}`);
  }
}

function gibToMib(value: number): number {
  return Math.max(0, Math.round(value * 1_024));
}

function classifyVendor(model: string): string {
  const normalized = model.toLowerCase();
  if (normalized.includes("nvidia")) return "nvidia";
  if (normalized.includes("amd") || normalized.includes("radeon")) return "amd";
  if (normalized.includes("intel")) return "intel";
  if (normalized.includes("apple")) return "apple";
  if (normalized.includes("ascend")) return "huawei";
  return "unknown";
}

function runLlmfit(
  executable: string,
  arguments_: string[],
  timeoutMs: number,
): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(
      executable,
      arguments_,
      {
        encoding: "utf8",
        timeout: timeoutMs,
        maxBuffer: MAX_OUTPUT_BYTES,
        windowsHide: true,
      },
      (error, stdout, stderr) => {
        if (!error) {
          resolve(stdout);
          return;
        }
        const detail = stderr.trim().slice(0, 500);
        reject(
          new Error(
            detail.length > 0
              ? `llmfit command failed: ${detail}`
              : `llmfit command failed: ${error.message}`,
          ),
        );
      },
    );
  });
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
