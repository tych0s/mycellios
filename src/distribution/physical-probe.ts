import { execFile } from "node:child_process";
import { buildIsolatedProcessEnvironment } from "./process-environment.js";

export const PHYSICAL_PROBE_SCHEMA = "gdlp-physical-probe/1" as const;
export const PHYSICAL_PROBE_REQUEST_SCHEMA =
  "gdlp-physical-probe-request/1" as const;

const NONCE_PATTERN = /^[A-Za-z0-9._:-]{16,128}$/;
const DIGEST_PATTERN = /^sha256:[0-9a-f]{64}$/;

export interface PhysicalProbeHostV1 {
  fingerprintSha256: string;
  fingerprintSource: string;
  platform: string;
  architecture: string;
  kernelRelease: string;
  pythonVersion: string;
}

export interface PhysicalProbeRuntimeV1 {
  torchVersion: string;
  cudaVersion: string | null;
  rocmVersion: string | null;
  cudaApiAvailable: boolean;
  distributedAvailable: boolean;
  ncclAvailable: boolean;
  ncclVersion: string | null;
}

export interface PhysicalProbeDeviceV1 {
  index: number;
  name: string;
  totalMemoryBytes: number;
  freeMemoryBytes: number;
  runtimeTotalMemoryBytes: number;
  capability: [number, number];
  uuidSha256: string | null;
  fingerprintSha256: string;
}

export interface PhysicalProbeV1 {
  schema: typeof PHYSICAL_PROBE_SCHEMA;
  nonce: string;
  host: PhysicalProbeHostV1;
  runtime: PhysicalProbeRuntimeV1;
  devices: PhysicalProbeDeviceV1[];
}

export interface PhysicalProbeCollector {
  collect(nonce: string, signal?: AbortSignal): Promise<PhysicalProbeV1>;
}

export interface PythonPhysicalProbeOptions {
  pythonExecutable?: string;
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  timeoutMs?: number;
  maxOutputBytes?: number;
}

export class PythonPhysicalProbe implements PhysicalProbeCollector {
  private readonly pythonExecutable: string;
  private readonly cwd: string | undefined;
  private readonly env: NodeJS.ProcessEnv | undefined;
  private readonly timeoutMs: number;
  private readonly maxOutputBytes: number;

  constructor(options: PythonPhysicalProbeOptions = {}) {
    this.pythonExecutable = normalizedText(
      options.pythonExecutable ?? "python",
      "physical_probe_python_executable_is_invalid",
    );
    this.cwd = optionalText(options.cwd, "physical_probe_cwd_is_invalid");
    this.env = options.env;
    this.timeoutMs = boundedInteger(
      options.timeoutMs ?? 30_000,
      1,
      300_000,
      "physical_probe_timeout_is_invalid",
    );
    this.maxOutputBytes = boundedInteger(
      options.maxOutputBytes ?? 1024 * 1024,
      1_024,
      16 * 1024 * 1024,
      "physical_probe_output_limit_is_invalid",
    );
  }

  collect(nonce: string, signal?: AbortSignal): Promise<PhysicalProbeV1> {
    assertPhysicalProbeNonce(nonce);
    if (signal?.aborted) return Promise.reject(abortError(signal));
    return new Promise((resolve, reject) => {
      execFile(
        this.pythonExecutable,
        ["-m", "distributed_runtime.physical_probe", "--nonce", nonce],
        {
          shell: false,
          windowsHide: true,
          timeout: this.timeoutMs,
          maxBuffer: this.maxOutputBytes,
          encoding: "utf8",
          ...(this.cwd === undefined ? {} : { cwd: this.cwd }),
          env: buildIsolatedProcessEnvironment(
            this.env === undefined ? {} : { overrides: this.env },
          ),
          ...(signal === undefined ? {} : { signal }),
        },
        (error, stdout, stderr) => {
          if (error) {
            reject(
              new Error(
                `physical_probe_process_failed:${error.message}:${String(stderr).slice(0, 1_024)}`,
              ),
            );
            return;
          }
          let value: unknown;
          try {
            value = JSON.parse(String(stdout));
          } catch {
            reject(new Error("physical_probe_output_is_not_json"));
            return;
          }
          try {
            validatePhysicalProbe(value, nonce);
            resolve(structuredClone(value));
          } catch (validationError) {
            reject(validationError);
          }
        },
      );
    });
  }
}

export function assertPhysicalProbeNonce(value: unknown): asserts value is string {
  if (typeof value !== "string" || !NONCE_PATTERN.test(value)) {
    throw new Error("physical_probe_nonce_is_invalid");
  }
}

export function validatePhysicalProbe(
  value: unknown,
  expectedNonce?: string,
): asserts value is PhysicalProbeV1 {
  record(value, "physical_probe");
  exactKeys(value, ["schema", "nonce", "host", "runtime", "devices"], "physical_probe");
  if (value.schema !== PHYSICAL_PROBE_SCHEMA) throw new Error("physical_probe_schema_is_invalid");
  assertPhysicalProbeNonce(value.nonce);
  if (expectedNonce !== undefined && value.nonce !== expectedNonce) {
    throw new Error("physical_probe_nonce_does_not_match");
  }
  validateHost(value.host);
  validateRuntime(value.runtime);
  if (!Array.isArray(value.devices) || value.devices.length > 64) {
    throw new Error("physical_probe_devices_are_invalid");
  }
  const indices = new Set<number>();
  const fingerprints = new Set<string>();
  for (const device of value.devices) {
    validateDevice(device);
    if (indices.has(device.index) || fingerprints.has(device.fingerprintSha256)) {
      throw new Error("physical_probe_devices_are_not_unique");
    }
    indices.add(device.index);
    fingerprints.add(device.fingerprintSha256);
  }
  if (!value.runtime.cudaApiAvailable && value.devices.length !== 0) {
    throw new Error("physical_probe_devices_require_cuda_api");
  }
  if (value.runtime.ncclAvailable && !value.runtime.distributedAvailable) {
    throw new Error("physical_probe_nccl_requires_distributed_runtime");
  }
}

function validateHost(value: unknown): asserts value is PhysicalProbeHostV1 {
  record(value, "physical_probe_host");
  exactKeys(
    value,
    [
      "fingerprintSha256",
      "fingerprintSource",
      "platform",
      "architecture",
      "kernelRelease",
      "pythonVersion",
    ],
    "physical_probe_host",
  );
  digest(value.fingerprintSha256, "physical_probe_host_fingerprint");
  for (const key of [
    "fingerprintSource",
    "platform",
    "architecture",
    "kernelRelease",
    "pythonVersion",
  ] as const) {
    normalizedText(value[key], `physical_probe_host_${key}_is_invalid`);
  }
}

function validateRuntime(value: unknown): asserts value is PhysicalProbeRuntimeV1 {
  record(value, "physical_probe_runtime");
  exactKeys(
    value,
    [
      "torchVersion",
      "cudaVersion",
      "rocmVersion",
      "cudaApiAvailable",
      "distributedAvailable",
      "ncclAvailable",
      "ncclVersion",
    ],
    "physical_probe_runtime",
  );
  normalizedText(value.torchVersion, "physical_probe_torch_version_is_invalid");
  for (const key of ["cudaVersion", "rocmVersion", "ncclVersion"] as const) {
    if (value[key] !== null) normalizedText(value[key], `physical_probe_${key}_is_invalid`);
  }
  for (const key of [
    "cudaApiAvailable",
    "distributedAvailable",
    "ncclAvailable",
  ] as const) {
    if (typeof value[key] !== "boolean") throw new Error(`physical_probe_${key}_is_invalid`);
  }
}

function validateDevice(value: unknown): asserts value is PhysicalProbeDeviceV1 {
  record(value, "physical_probe_device");
  exactKeys(
    value,
    [
      "index",
      "name",
      "totalMemoryBytes",
      "freeMemoryBytes",
      "runtimeTotalMemoryBytes",
      "capability",
      "uuidSha256",
      "fingerprintSha256",
    ],
    "physical_probe_device",
  );
  integer(value.index, 0, 63, "physical_probe_device_index_is_invalid");
  normalizedText(value.name, "physical_probe_device_name_is_invalid");
  integer(
    value.totalMemoryBytes,
    1,
    Number.MAX_SAFE_INTEGER,
    "physical_probe_device_memory_is_invalid",
  );
  integer(
    value.freeMemoryBytes,
    0,
    value.totalMemoryBytes,
    "physical_probe_device_free_memory_is_invalid",
  );
  integer(
    value.runtimeTotalMemoryBytes,
    1,
    Number.MAX_SAFE_INTEGER,
    "physical_probe_device_runtime_memory_is_invalid",
  );
  if (!Array.isArray(value.capability) || value.capability.length !== 2) {
    throw new Error("physical_probe_device_capability_is_invalid");
  }
  integer(value.capability[0], 0, 100, "physical_probe_device_capability_is_invalid");
  integer(value.capability[1], 0, 100, "physical_probe_device_capability_is_invalid");
  if (value.uuidSha256 !== null) digest(value.uuidSha256, "physical_probe_device_uuid");
  digest(value.fingerprintSha256, "physical_probe_device_fingerprint");
}

function record(value: unknown, name: string): asserts value is Record<string, any> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${name}_is_invalid`);
  }
}

function exactKeys(value: Record<string, any>, expected: string[], name: string): void {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (actual.length !== wanted.length || actual.some((key, index) => key !== wanted[index])) {
    throw new Error(`${name}_keys_are_invalid`);
  }
}

function digest(value: unknown, name: string): asserts value is string {
  if (typeof value !== "string" || !DIGEST_PATTERN.test(value)) {
    throw new Error(`${name}_is_invalid`);
  }
}

function integer(value: unknown, minimum: number, maximum: number, name: string): asserts value is number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum || (value as number) > maximum) {
    throw new Error(name);
  }
}

function normalizedText(value: unknown, name: string): string {
  if (typeof value !== "string" || !value.trim() || /[\0\r\n]/.test(value)) {
    throw new Error(name);
  }
  return value;
}

function optionalText(value: unknown, name: string): string | undefined {
  return value === undefined ? undefined : normalizedText(value, name);
}

function boundedInteger(value: unknown, min: number, max: number, name: string): number {
  integer(value, min, max, name);
  return value;
}

function abortError(signal: AbortSignal): Error {
  return signal.reason instanceof Error ? signal.reason : new Error("physical_probe_aborted");
}
