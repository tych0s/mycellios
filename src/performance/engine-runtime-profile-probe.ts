import type { EngineRuntimeChallenge } from "../contracts/evidence-challenge.js";
import {
  engineRuntimeMeasurementSchema,
  type EngineRuntimeMeasurement,
} from "../contracts/engine-runtime-profile.js";
import {
  runRuntimeProfileCommand,
  type RuntimeProfileCommandRunner,
} from "./runtime-profile-probe.js";

export interface ProbeQwen3EngineRuntimeProfileOptions {
  pythonExecutable: string;
  pythonPath: readonly string[];
  pathAdditions?: readonly string[];
  device: string;
  precision: "float16" | "float32";
  cwd?: string;
  timeoutMs?: number;
  warmupSamples?: number;
  threads?: number;
  env?: NodeJS.ProcessEnv;
  commandRunner?: RuntimeProfileCommandRunner;
}

const MAXIMUM_OUTPUT_BYTES = 256 * 1024;

export async function probeQwen3EngineRuntimeProfile(
  challenge: EngineRuntimeChallenge,
  options: ProbeQwen3EngineRuntimeProfileOptions,
): Promise<EngineRuntimeMeasurement> {
  if (challenge.probeKind !== "qwen3-dense-v1") {
    throw new Error("engine_runtime_probe_kind_is_unsupported");
  }
  const executable = options.pythonExecutable.trim();
  if (!executable || !options.device.trim() || options.pythonPath.some((entry) => !entry.trim())) {
    throw new Error("engine_runtime_probe_runtime_is_invalid");
  }
  const separator = process.platform === "win32" ? ";" : ":";
  const runner = options.commandRunner ?? runRuntimeProfileCommand;
  const result = await runner(executable, [
    "-m",
    "distributed_runtime.engine_runtime_profile",
    "--probe-kind",
    challenge.probeKind,
    "--backend",
    challenge.backend,
    "--device",
    options.device,
    "--precision",
    options.precision,
    "--hidden-size",
    String(challenge.hiddenSize),
    "--attention-heads",
    String(challenge.attentionHeads),
    "--kv-heads",
    String(challenge.kvHeads),
    "--head-dim",
    String(challenge.headDim),
    "--layer-count",
    String(challenge.expectedLayerEnd - challenge.expectedLayerStart),
    "--context-tokens",
    String(challenge.contextTokens),
    "--kv-bytes-per-token",
    String(challenge.expectedKvBytesPerToken),
    "--layer-weight-bytes",
    String(challenge.expectedLayerWeightBytes),
    "--reference-decode-ms-per-token",
    String(challenge.referenceDecodeMsPerToken),
    "--reference-prefill-ms-per-token",
    String(challenge.referencePrefillMsPerToken),
    "--warmup-samples",
    String(boundedInteger(options.warmupSamples ?? 2, 1, 1_000)),
    "--samples",
    String(challenge.minimumSamples),
    "--threads",
    String(boundedInteger(options.threads ?? 1, 1, 256)),
  ], {
    ...(options.cwd ? { cwd: options.cwd } : {}),
    timeoutMs: boundedInteger(options.timeoutMs ?? 300_000, 1_000, 3_600_000),
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
    throw new Error(`engine_runtime_probe_failed:${result.code}:${boundedText(result.stderr)}`);
  }
  if (Buffer.byteLength(result.stdout, "utf8") > MAXIMUM_OUTPUT_BYTES) {
    throw new Error("engine_runtime_probe_output_is_too_large");
  }
  const measurement = engineRuntimeMeasurementSchema.parse(parseFinalJson(result.stdout));
  if (
    measurement.samples < challenge.minimumSamples
    || measurement.capacity.contextTokens !== challenge.contextTokens
    || measurement.capacity.maxKvTokens < challenge.contextTokens
    || measurement.capacity.kvBytesPerToken !== challenge.expectedKvBytesPerToken
    || measurement.capacity.maxLayerCount
      < challenge.expectedLayerEnd - challenge.expectedLayerStart
  ) {
    throw new Error("engine_runtime_probe_output_does_not_match_challenge");
  }
  return measurement;
}

function parseFinalJson(serialized: string): unknown {
  const lines = serialized.trim().split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const final = lines.at(-1);
  if (!final) throw new Error("engine_runtime_probe_output_is_not_json");
  try {
    return JSON.parse(final);
  } catch {
    throw new Error("engine_runtime_probe_output_is_not_json");
  }
}

function boundedInteger(value: number, minimum: number, maximum: number): number {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new Error("engine_runtime_probe_option_is_invalid");
  }
  return value;
}

function boundedText(value: string): string {
  return value.trim().replace(/\s+/g, " ").slice(-1_024) || "no_stderr";
}
