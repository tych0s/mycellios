import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream, type Dirent, type Stats } from "node:fs";
import { lstat, mkdir, readFile, readdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { canonicalEvidenceJson } from "../core/json.js";
import { MODEL_ADAPTER_REGISTRY_ID } from "../contracts/model-adapter-registry.js";
import {
  validatePythonLaunchDescription,
  type PythonNativeGgufStageConfiguration,
  type PythonLaunchProcess,
  type PythonPipelineLaunchDescription,
} from "../distribution/python-launcher.js";

const SHA256 = /^[0-9a-f]{64}$/;
const SHA256_IDENTITY = /^sha256:[0-9a-f]{64}$/;
const MAX_COMPILER_OUTPUT_BYTES = 2 * 1024 * 1024;
const STAGE_ARTIFACT_MANIFEST = "mycellios-stage.json";
const MAX_NATIVE_MANIFEST_BYTES = 16 * 1024 * 1024;
const NATIVE_STAGE_MANIFEST = "native-stage.json";
const NATIVE_STAGE_WEIGHTS = "native-stage.gguf";
const NATIVE_STAGE_CONFIG = "config.json";
const NATIVE_STAGE_ENTRIES = [
  NATIVE_STAGE_MANIFEST,
  NATIVE_STAGE_WEIGHTS,
  NATIVE_STAGE_CONFIG,
] as const;
const NATIVE_EXECUTABLE_GGML_TYPES = [
  0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 30,
] as const;

export type NativeGgufPackageVerifier = (
  binding: Readonly<PythonNativeGgufStageConfiguration>,
  signal?: AbortSignal,
) => void | Promise<void>;

export interface StageArtifactPreparationOptions {
  nodeId: string;
  pythonExecutable: string;
  cacheDirectory: string;
  cwd?: string;
  environment?: NodeJS.ProcessEnv;
  signal?: AbortSignal;
  /** Deterministic seam for unit tests; production always uses spawn(). */
  compilerRunner?: (executable: string, args: readonly string[]) => Promise<string>;
  /**
   * Deterministic seam for the content-addressed cache import. Production
   * invokes the native Python cache CLI. Keeping this separate prevents a
   * compiler test double from accidentally masquerading as verified storage.
   */
  artifactCacheRunner?: (
    executable: string,
    args: readonly string[],
  ) => Promise<string>;
  /**
   * Host-local verification seam for an already sealed native package. The
   * default implementation is entirely in-process and never invokes Python.
   */
  nativePackageVerifier?: NativeGgufPackageVerifier;
  onProgress?: (event: {
    stageIndex: number;
    layerStart: number;
    layerEnd: number;
    state: "preparing" | "ready";
    packageId?: string;
    weightsSizeBytes?: number;
  }) => void;
}

interface StageArtifactCompilerResult {
  destination: string;
  package_id: string;
  artifact_identity: string;
  model_identity: string;
  layer_start: number;
  layer_end: number;
  total_layers: number;
  weights_size_bytes: number;
}

interface StageArtifactCacheResult {
  cache_root: string;
  package_directory: string;
  package_id: string;
  artifact_identity: string;
  manifest_sha256: string;
  downloaded_bytes: number;
  resumed_bytes: number;
  materialized: boolean;
}

/**
 * Prepare only the physical ranges assigned to this node and return trusted
 * host-local launch commands. The coordinator's signed launch description is
 * not mutated and remains the authorization input for runtime.start.
 */
export async function prepareNodeStageArtifacts(
  description: PythonPipelineLaunchDescription,
  options: StageArtifactPreparationOptions,
): Promise<PythonLaunchProcess[]> {
  // This independently authenticates every native binding against the exact
  // compiler-derived argv before any host-local filesystem work is attempted.
  validatePythonLaunchDescription(description);
  const local = description.launchOrder.filter(
    (process) => process.anchor.memberId === options.nodeId,
  );
  if (local.length === 0) throw new Error("stage_artifact_plan_has_no_local_process");
  let cacheRoot: string | null = null;
  const preparedByRange = new Map<string, StageArtifactCompilerResult>();
  let preparedModelIdentity: string | null = null;
  const result: PythonLaunchProcess[] = [];
  for (const process of local) {
    if (process.kind !== "cell-member" && process.nativeGguf !== null) {
      await (options.nativePackageVerifier ?? verifyNativeGgufStagePackage)(
        structuredClone(process.nativeGguf),
        options.signal,
      );
      options.onProgress?.({
        stageIndex: process.stageIndex,
        layerStart: process.layerStart,
        layerEnd: process.layerEnd,
        state: "ready",
        packageId: process.nativeGguf.packageId,
      });
      // A first-class GGUF process is already materialized and sealed. In
      // particular, --model and --stage-package-identity must stay byte-exact.
      result.push(process);
      continue;
    }

    if (cacheRoot === null) {
      cacheRoot = resolve(options.cacheDirectory, "native-stages");
      await mkdir(cacheRoot, { recursive: true });
    }
    const rangeKey = `${process.layerStart}:${process.layerEnd}:${process.totalLayers}`;
    let prepared = preparedByRange.get(rangeKey);
    if (!prepared) {
      options.onProgress?.({
        stageIndex: process.stageIndex,
        layerStart: process.layerStart,
        layerEnd: process.layerEnd,
        state: "preparing",
      });
      const cacheKey = createHash("sha256")
        .update("mycellios-native-stage-cache/2\0")
        .update(MODEL_ADAPTER_REGISTRY_ID)
        .update("\0")
        .update(description.runtimeModel.artifactIdentity ?? description.modelIdentity.revision)
        .update("\0")
        .update(rangeKey)
        .digest("hex");
      const destination = resolve(cacheRoot, cacheKey);
      const args = [
        "-u",
        "-m",
        "distributed_runtime.stage_artifact",
        description.runtimeModel.source,
        destination,
        "--layer-start",
        String(process.layerStart),
        "--layer-end",
        String(process.layerEnd),
        "--reuse-verified",
      ];
      if (description.runtimeModel.revision !== null) {
        args.push("--revision", description.runtimeModel.revision);
      }
      prepared = parseCompilerResult(await (
        options.compilerRunner
          ? options.compilerRunner(options.pythonExecutable, args)
          : runCompiler(options.pythonExecutable, args, options)
      ));
      if (
        prepared.layer_start !== process.layerStart
        || prepared.layer_end !== process.layerEnd
        || prepared.total_layers !== process.totalLayers
      ) {
        throw new Error("stage_artifact_compiler_returned_wrong_range");
      }
      if (
        preparedModelIdentity !== null
        && prepared.model_identity !== preparedModelIdentity
      ) {
        throw new Error("stage_artifact_compiler_returned_inconsistent_model_identity");
      }
      if (
        description.runtimeModel.artifactIdentity !== undefined
        && prepared.model_identity !== description.runtimeModel.artifactIdentity
      ) {
        throw new Error("stage_artifact_compiler_returned_wrong_model_identity");
      }
      const cacheResult = parseArtifactCacheResult(await (
        options.artifactCacheRunner
          ? options.artifactCacheRunner(
              options.pythonExecutable,
              stageArtifactCacheArgs(
                resolve(options.cacheDirectory, "stage-cas"),
                prepared,
              ),
            )
          : runCompiler(
              options.pythonExecutable,
              stageArtifactCacheArgs(
                resolve(options.cacheDirectory, "stage-cas"),
                prepared,
              ),
              options,
            )
      ));
      if (
        cacheResult.package_id !== prepared.package_id
        || cacheResult.artifact_identity !== prepared.artifact_identity
      ) {
        throw new Error("stage_artifact_cache_returned_wrong_identity");
      }
      prepared = {
        ...prepared,
        destination: cacheResult.package_directory,
      };
      preparedModelIdentity = prepared.model_identity;
      preparedByRange.set(rangeKey, prepared);
      options.onProgress?.({
        stageIndex: process.stageIndex,
        layerStart: process.layerStart,
        layerEnd: process.layerEnd,
        state: "ready",
        packageId: prepared.package_id,
        weightsSizeBytes: prepared.weights_size_bytes,
      });
    }
    result.push(rewriteProcessForStageArtifact(process, prepared));
  }
  return result;
}

function stageArtifactCacheArgs(
  cacheRoot: string,
  artifact: StageArtifactCompilerResult,
): string[] {
  return [
    "-u",
    "-m",
    "distributed_runtime.stage_artifact_cache",
    "acquire",
    "--cache-root",
    cacheRoot,
    "--manifest",
    resolve(artifact.destination, STAGE_ARTIFACT_MANIFEST),
    "--expected-package-id",
    artifact.package_id,
  ];
}

/**
 * Verify the small native package envelope with Node primitives. The native
 * runtime repeats deeper GGUF/tensor validation immediately before loading.
 */
export async function verifyNativeGgufStagePackage(
  binding: Readonly<PythonNativeGgufStageConfiguration>,
  signal?: AbortSignal,
): Promise<void> {
  if (signal?.aborted) throw abortError(signal);
  const root = resolve(binding.packagePath);
  let entries: Dirent[];
  try {
    entries = await readdir(root, { withFileTypes: true });
  } catch {
    throw new Error("native_stage_package_path_is_invalid");
  }
  const entryNames = entries.map((entry) => entry.name).sort();
  if (
    canonicalEvidenceJson(entryNames)
      !== canonicalEvidenceJson([...NATIVE_STAGE_ENTRIES].sort())
    || entries.some((entry) => entry.isSymbolicLink())
  ) {
    throw new Error("native_stage_package_contains_unsealed_entries");
  }

  const manifestPath = join(root, NATIVE_STAGE_MANIFEST);
  const manifestStat = await checkedRegularFile(manifestPath, NATIVE_STAGE_MANIFEST);
  if (manifestStat.size > MAX_NATIVE_MANIFEST_BYTES) {
    throw new Error("native_stage_package_manifest_is_too_large");
  }
  let document: unknown;
  try {
    document = JSON.parse(await readFile(manifestPath, "utf8"));
  } catch {
    throw new Error("native_stage_package_manifest_is_invalid");
  }
  assertExactRecord(
    document,
    ["schema", "model", "stage", "files", "tensors", "execution", "packageId"],
    "native_stage_package_manifest_is_invalid",
  );
  if (document.schema !== "gdlp-native-gguf-stage/1") {
    throw new Error("native_stage_package_schema_is_invalid");
  }
  if (typeof document.packageId !== "string" || !SHA256.test(document.packageId)) {
    throw new Error("native_stage_package_id_is_invalid");
  }
  const withoutPackageId = { ...document };
  delete withoutPackageId.packageId;
  const manifestPackageId = createHash("sha256")
    .update(canonicalEvidenceJson(withoutPackageId))
    .digest("hex");
  if (
    manifestPackageId !== document.packageId
    || document.packageId !== binding.packageId
  ) {
    throw new Error("native_stage_package_identity_mismatch");
  }

  assertExactRecord(
    document.model,
    ["source", "revision", "architecture", "sourceGgufSha256"],
    "native_stage_package_model_is_invalid",
  );
  if (
    document.model.source !== binding.modelSource
    || document.model.revision !== binding.modelRevision
  ) {
    throw new Error("native_stage_package_model_coordinates_mismatch");
  }
  if (
    (document.model.architecture !== "llama" && document.model.architecture !== "qwen3")
    || typeof document.model.sourceGgufSha256 !== "string"
    || !SHA256.test(document.model.sourceGgufSha256)
  ) {
    throw new Error("native_stage_package_model_is_invalid");
  }

  assertExactRecord(
    document.stage,
    ["layerStart", "layerEnd", "totalLayers", "first", "last"],
    "native_stage_package_range_is_invalid",
  );
  if (
    document.stage.layerStart !== binding.layerStart
    || document.stage.layerEnd !== binding.layerEnd
    || document.stage.totalLayers !== binding.totalLayers
    || document.stage.first !== (binding.layerStart === 0)
    || document.stage.last !== (binding.layerEnd === binding.totalLayers)
  ) {
    throw new Error("native_stage_package_range_mismatch");
  }

  assertExactRecord(
    document.execution,
    [
      "engine",
      "externalRuntimeRequired",
      "materialization",
      "tensorLayout",
      "executableGgmlTypes",
    ],
    "native_stage_package_execution_is_invalid",
  );
  const expectedLayout = document.model.architecture === "llama"
    ? "llama-rope-qk-permuted"
    : "huggingface-row-major";
  if (
    document.execution.engine !== "mycellios-native-gguf"
    || document.execution.externalRuntimeRequired !== false
    || document.execution.materialization !== "stage-only-dequantize-to-torch"
    || document.execution.tensorLayout !== expectedLayout
    || canonicalEvidenceJson(document.execution.executableGgmlTypes)
      !== canonicalEvidenceJson(NATIVE_EXECUTABLE_GGML_TYPES)
  ) {
    throw new Error("native_stage_package_execution_is_invalid");
  }
  validateNativeTensorRecords(document.tensors);

  assertExactRecord(
    document.files,
    [NATIVE_STAGE_WEIGHTS, NATIVE_STAGE_CONFIG],
    "native_stage_package_files_are_invalid",
  );
  await verifyNativeFileRecord(
    root,
    NATIVE_STAGE_WEIGHTS,
    document.files[NATIVE_STAGE_WEIGHTS],
    signal,
  );
  const configFile = await verifyNativeFileRecord(
    root,
    NATIVE_STAGE_CONFIG,
    document.files[NATIVE_STAGE_CONFIG],
    signal,
  );
  const modelIdentity = `sha256:${createHash("sha256")
    .update(canonicalEvidenceJson({
      schema: "gdlp-native-gguf-model-identity/1",
      sourceGgufSha256: document.model.sourceGgufSha256,
      configSha256: configFile.sha256,
    }))
    .digest("hex")}`;
  if (!SHA256_IDENTITY.test(modelIdentity) || modelIdentity !== binding.modelIdentity) {
    throw new Error("native_stage_package_model_identity_mismatch");
  }
}

interface NativeFileRecord {
  sizeBytes: number;
  sha256: string;
}

async function verifyNativeFileRecord(
  root: string,
  name: string,
  value: unknown,
  signal?: AbortSignal,
): Promise<NativeFileRecord> {
  assertExactRecord(
    value,
    ["sizeBytes", "sha256"],
    `native_stage_package_file_record_is_invalid:${name}`,
  );
  if (
    !Number.isSafeInteger(value.sizeBytes)
    || Number(value.sizeBytes) < 1
    || typeof value.sha256 !== "string"
    || !SHA256.test(value.sha256)
  ) {
    throw new Error(`native_stage_package_file_record_is_invalid:${name}`);
  }
  const path = join(root, name);
  const file = await checkedRegularFile(path, name);
  if (file.size !== value.sizeBytes) {
    throw new Error(`native_stage_package_file_size_mismatch:${name}`);
  }
  if (await sha256File(path, signal) !== value.sha256) {
    throw new Error(`native_stage_package_file_digest_mismatch:${name}`);
  }
  return {
    sizeBytes: value.sizeBytes as number,
    sha256: value.sha256,
  };
}

async function checkedRegularFile(path: string, name: string): Promise<Stats> {
  let result: Stats;
  try {
    result = await lstat(path);
  } catch {
    throw new Error(`native_stage_package_file_is_invalid:${name}`);
  }
  if (!result.isFile() || result.isSymbolicLink()) {
    throw new Error(`native_stage_package_file_is_invalid:${name}`);
  }
  return result;
}

async function sha256File(path: string, signal?: AbortSignal): Promise<string> {
  if (signal?.aborted) throw abortError(signal);
  const digest = createHash("sha256");
  const stream = createReadStream(path);
  try {
    for await (const chunk of stream) {
      if (signal?.aborted) throw abortError(signal);
      digest.update(chunk);
    }
  } finally {
    stream.destroy();
  }
  return digest.digest("hex");
}

function validateNativeTensorRecords(value: unknown): void {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error("native_stage_package_tensors_are_invalid");
  }
  for (const tensor of value) {
    assertExactRecord(
      tensor,
      ["name", "dimensions", "ggmlType", "sizeBytes"],
      "native_stage_package_tensors_are_invalid",
    );
    if (
      typeof tensor.name !== "string"
      || tensor.name.length === 0
      || !Array.isArray(tensor.dimensions)
      || tensor.dimensions.length === 0
      || tensor.dimensions.some(
        (dimension) => !Number.isSafeInteger(dimension) || Number(dimension) < 1,
      )
      || !Number.isSafeInteger(tensor.ggmlType)
      || !NATIVE_EXECUTABLE_GGML_TYPES.includes(
        tensor.ggmlType as (typeof NATIVE_EXECUTABLE_GGML_TYPES)[number],
      )
      || !Number.isSafeInteger(tensor.sizeBytes)
      || Number(tensor.sizeBytes) < 1
    ) {
      throw new Error("native_stage_package_tensors_are_invalid");
    }
  }
}

function assertExactRecord(
  value: unknown,
  keys: readonly string[],
  error: string,
): asserts value is Record<string, unknown> {
  if (
    value === null
    || typeof value !== "object"
    || Array.isArray(value)
    || Object.keys(value).length !== keys.length
    || keys.some((key) => !Object.prototype.hasOwnProperty.call(value, key))
  ) {
    throw new Error(error);
  }
}

function rewriteProcessForStageArtifact(
  process: PythonLaunchProcess,
  artifact: StageArtifactCompilerResult,
): PythonLaunchProcess {
  const args = [...process.command.args];
  replaceFlagValue(args, "--model", artifact.destination);
  removeFlag(args, "--revision");
  replaceFlagValue(args, "--model-artifact-identity", artifact.model_identity);
  setFlagValue(args, "--stage-package-identity", artifact.artifact_identity);
  return {
    ...process,
    command: { ...process.command, args },
  } as PythonLaunchProcess;
}

function parseCompilerResult(stdout: string): StageArtifactCompilerResult {
  let value: unknown;
  try {
    value = JSON.parse(stdout);
  } catch {
    throw new Error("stage_artifact_compiler_returned_invalid_json");
  }
  if (!value || typeof value !== "object") {
    throw new Error("stage_artifact_compiler_returned_invalid_document");
  }
  const candidate = value as Record<string, unknown>;
  const packageId = candidate.package_id;
  const artifactIdentity = candidate.artifact_identity;
  const modelIdentity = candidate.model_identity;
  const destination = candidate.destination;
  if (
    typeof packageId !== "string"
    || !SHA256.test(packageId)
    || artifactIdentity !== `sha256:${packageId}`
    || typeof modelIdentity !== "string"
    || !/^sha256:[0-9a-f]{64}$/.test(modelIdentity)
    || typeof destination !== "string"
  ) {
    throw new Error("stage_artifact_compiler_returned_invalid_identity");
  }
  const integers = [
    "layer_start",
    "layer_end",
    "total_layers",
    "weights_size_bytes",
  ] as const;
  for (const key of integers) {
    if (!Number.isSafeInteger(candidate[key]) || Number(candidate[key]) < 0) {
      throw new Error(`stage_artifact_compiler_returned_invalid_${key}`);
    }
  }
  return candidate as unknown as StageArtifactCompilerResult;
}

function parseArtifactCacheResult(stdout: string): StageArtifactCacheResult {
  let value: unknown;
  try {
    value = JSON.parse(stdout);
  } catch {
    throw new Error("stage_artifact_cache_returned_invalid_json");
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("stage_artifact_cache_returned_invalid_document");
  }
  const candidate = value as Record<string, unknown>;
  const packageId = candidate.package_id;
  if (
    typeof packageId !== "string"
    || !SHA256.test(packageId)
    || candidate.artifact_identity !== `sha256:${packageId}`
    || typeof candidate.cache_root !== "string"
    || candidate.cache_root.length === 0
    || typeof candidate.package_directory !== "string"
    || candidate.package_directory.length === 0
    || typeof candidate.manifest_sha256 !== "string"
    || !SHA256.test(candidate.manifest_sha256)
    || typeof candidate.materialized !== "boolean"
  ) {
    throw new Error("stage_artifact_cache_returned_invalid_identity");
  }
  for (const key of ["downloaded_bytes", "resumed_bytes"] as const) {
    if (!Number.isSafeInteger(candidate[key]) || Number(candidate[key]) < 0) {
      throw new Error(`stage_artifact_cache_returned_invalid_${key}`);
    }
  }
  return candidate as unknown as StageArtifactCacheResult;
}

async function runCompiler(
  executable: string,
  args: string[],
  options: StageArtifactPreparationOptions,
): Promise<string> {
  if (options.signal?.aborted) throw abortError(options.signal);
  return await new Promise<string>((resolvePromise, reject) => {
    const child = spawn(executable, args, {
      shell: false,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
      ...(options.cwd ? { cwd: options.cwd } : {}),
      env: { ...process.env, ...options.environment },
    });
    let stdout: Buffer<ArrayBufferLike> = Buffer.alloc(0);
    let stderr: Buffer<ArrayBufferLike> = Buffer.alloc(0);
    let overflow = false;
    const append = (
      current: Buffer<ArrayBufferLike>,
      chunk: Buffer<ArrayBufferLike>,
    ): Buffer<ArrayBufferLike> => {
      const next = Buffer.concat([current, chunk]);
      if (next.byteLength <= MAX_COMPILER_OUTPUT_BYTES) return next;
      overflow = true;
      return next.subarray(0, MAX_COMPILER_OUTPUT_BYTES);
    };
    child.stdout.on("data", (chunk: Buffer<ArrayBufferLike>) => { stdout = append(stdout, chunk); });
    child.stderr.on("data", (chunk: Buffer<ArrayBufferLike>) => { stderr = append(stderr, chunk); });
    const abort = () => child.kill();
    options.signal?.addEventListener("abort", abort, { once: true });
    child.once("error", reject);
    child.once("close", (code) => {
      options.signal?.removeEventListener("abort", abort);
      if (options.signal?.aborted) return reject(abortError(options.signal));
      if (overflow) return reject(new Error("stage_artifact_compiler_output_limit_exceeded"));
      if (code !== 0) {
        const detail = stderr.toString("utf8").trim();
        return reject(new Error(
          detail ? `stage_artifact_preparation_failed:${detail}` : `stage_artifact_compiler_exited:${code}`,
        ));
      }
      resolvePromise(stdout.toString("utf8").trim());
    });
  });
}

function replaceFlagValue(args: string[], flag: string, value: string): void {
  const index = args.lastIndexOf(flag);
  if (index < 0 || index + 1 >= args.length) {
    throw new Error(`stage_artifact_launch_flag_missing:${flag}`);
  }
  args[index + 1] = value;
}

function setFlagValue(args: string[], flag: string, value: string): void {
  const index = args.lastIndexOf(flag);
  if (index < 0) {
    args.push(flag, value);
    return;
  }
  if (index + 1 >= args.length) {
    throw new Error(`stage_artifact_launch_flag_missing:${flag}`);
  }
  args[index + 1] = value;
}

function removeFlag(args: string[], flag: string): void {
  for (let index = args.length - 2; index >= 0; index -= 1) {
    if (args[index] !== flag) continue;
    args.splice(index, 2);
  }
}

function abortError(signal: AbortSignal): Error {
  return signal.reason instanceof Error
    ? signal.reason
    : new Error("stage_artifact_preparation_cancelled");
}
