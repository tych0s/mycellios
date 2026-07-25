import { execFile } from "node:child_process";
import {
  runtimePerformanceProfileInputSchema,
  sealRuntimePerformanceProfile,
  type RuntimePerformanceProfile,
  type RuntimePerformanceProfileInput,
} from "./runtime-profile.js";

export interface RuntimeProfileCommandOptions {
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  timeoutMs: number;
}

export interface RuntimeProfileCommandResult {
  code: number;
  stdout: string;
  stderr: string;
}

export type RuntimeProfileCommandRunner = (
  executable: string,
  arguments_: readonly string[],
  options: RuntimeProfileCommandOptions,
) => Promise<RuntimeProfileCommandResult>;

export interface ProbeRuntimePerformanceProfileOptions {
  pythonExecutable: string;
  pythonPath: readonly string[];
  pathAdditions?: readonly string[];
  backend: RuntimePerformanceProfileInput["backend"];
  device: string;
  precision: RuntimePerformanceProfileInput["precision"];
  expectedDeviceName?: string;
  cwd?: string;
  timeoutMs?: number;
  warmupSamples?: number;
  samples?: number;
  threads?: number;
  env?: NodeJS.ProcessEnv;
  commandRunner?: RuntimeProfileCommandRunner;
  now?: () => number;
}

const MAXIMUM_OUTPUT_BYTES = 256 * 1024;

/**
 * Execute the packaged physical calibration in the exact Python environment
 * that launches Mycellios stages. There is intentionally no fallback result:
 * unavailable hardware, a backend mismatch, or malformed evidence rejects.
 */
export async function probeRuntimePerformanceProfile(
  options: ProbeRuntimePerformanceProfileOptions,
): Promise<RuntimePerformanceProfile> {
  const warmupSamples = boundedInteger(options.warmupSamples ?? 2, 1, 1_000);
  const samples = boundedInteger(options.samples ?? 9, 7, 10_000);
  const threads = boundedInteger(options.threads ?? 1, 1, 256);
  const timeoutMs = boundedInteger(options.timeoutMs ?? 180_000, 1_000, 3_600_000);
  const executable = options.pythonExecutable.trim();
  if (!executable) throw new Error("runtime_performance_probe_python_is_required");
  if (!options.device.trim()) throw new Error("runtime_performance_probe_device_is_required");
  if (options.pythonPath.length === 0 || options.pythonPath.some((entry) => !entry.trim())) {
    throw new Error("runtime_performance_probe_python_path_is_required");
  }
  const separator = process.platform === "win32" ? ";" : ":";
  const runner = options.commandRunner ?? runRuntimeProfileCommand;
  const startedAt = (options.now ?? Date.now)();
  const result = await runner(executable, [
    "-m",
    "distributed_runtime.runtime_profile",
    "--backend",
    options.backend,
    "--device",
    options.device,
    "--precision",
    options.precision,
    "--warmup-samples",
    String(warmupSamples),
    "--samples",
    String(samples),
    "--threads",
    String(threads),
  ], {
    ...(options.cwd ? { cwd: options.cwd } : {}),
    timeoutMs,
    env: {
      ...process.env,
      ...options.env,
      PYTHONPATH: options.pythonPath.join(separator),
      PATH: [
        ...(options.pathAdditions ?? []),
        process.env.PATH ?? "",
      ].filter(Boolean).join(separator),
      TOKENIZERS_PARALLELISM: "false",
    },
  });
  if (result.code !== 0) {
    throw new Error(
      `runtime_performance_probe_failed:${result.code}:${boundedDiagnostic(result.stderr)}`,
    );
  }
  const finishedAt = (options.now ?? Date.now)();
  const profile = parsePhysicalRuntimePerformanceProfile(result.stdout, {
    backend: options.backend,
    precision: options.precision,
    ...(options.expectedDeviceName
      ? { expectedDeviceName: options.expectedDeviceName }
      : {}),
  });
  const measuredAt = Date.parse(profile.measuredAt);
  if (measuredAt < startedAt - 60_000 || measuredAt > finishedAt + 60_000) {
    throw new Error("runtime_performance_probe_timestamp_is_not_current");
  }
  return profile;
}

export function parsePhysicalRuntimePerformanceProfile(
  serialized: string,
  expected: {
    backend: RuntimePerformanceProfileInput["backend"];
    precision: RuntimePerformanceProfileInput["precision"];
    expectedDeviceName?: string;
  },
): RuntimePerformanceProfile {
  if (Buffer.byteLength(serialized, "utf8") > MAXIMUM_OUTPUT_BYTES) {
    throw new Error("runtime_performance_probe_output_is_too_large");
  }
  let decoded: unknown;
  try {
    decoded = JSON.parse(serialized.trim());
  } catch {
    throw new Error("runtime_performance_probe_output_is_not_json");
  }
  const input = runtimePerformanceProfileInputSchema.parse(decoded);
  if (input.source !== "physical-microbenchmark") {
    throw new Error("runtime_performance_probe_source_is_not_physical");
  }
  if (input.backend !== expected.backend) {
    throw new Error("runtime_performance_probe_backend_does_not_match");
  }
  if (input.precision !== expected.precision) {
    throw new Error("runtime_performance_probe_precision_does_not_match");
  }
  if (input.activationCodecId !== "fp16") {
    throw new Error("runtime_performance_probe_codec_does_not_match");
  }
  if (
    expected.expectedDeviceName
    && normalizedDeviceName(input.deviceName)
      !== normalizedDeviceName(expected.expectedDeviceName)
  ) {
    throw new Error("runtime_performance_probe_device_does_not_match");
  }
  return sealRuntimePerformanceProfile(input);
}

export const runRuntimeProfileCommand: RuntimeProfileCommandRunner = (
  executable,
  arguments_,
  options,
) => new Promise((resolve, reject) => {
  execFile(executable, [...arguments_], {
    shell: false,
    windowsHide: true,
    timeout: options.timeoutMs,
    maxBuffer: MAXIMUM_OUTPUT_BYTES,
    encoding: "utf8",
    ...(options.cwd ? { cwd: options.cwd } : {}),
    ...(options.env ? { env: options.env } : {}),
  }, (error, stdout, stderr) => {
    if (error && typeof error.code !== "number") {
      reject(new Error(`runtime_performance_probe_process_error:${error.message}`));
      return;
    }
    resolve({
      code: typeof error?.code === "number" ? error.code : 0,
      stdout,
      stderr,
    });
  });
});

function normalizedDeviceName(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

function boundedInteger(value: number, minimum: number, maximum: number): number {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new Error("runtime_performance_probe_option_is_invalid");
  }
  return value;
}

function boundedDiagnostic(value: string): string {
  return value.trim().replace(/\s+/g, " ").slice(-1_024) || "no_stderr";
}
