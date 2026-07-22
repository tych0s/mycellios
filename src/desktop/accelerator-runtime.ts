import { spawn } from "node:child_process";
import {
  cp,
  mkdir,
  readFile,
  rename,
  rm,
  stat,
  statfs,
  writeFile,
} from "node:fs/promises";
import { existsSync, readdirSync } from "node:fs";
import { basename, dirname, isAbsolute, join, relative, resolve } from "node:path";
import { randomUUID } from "node:crypto";

export const PORTABLE_RUNTIME_SCHEMA = "mycellios-distribution-runtime/2" as const;
export const ACCELERATOR_RUNTIME_SCHEMA = "mycellios-accelerator-runtime/1" as const;

const GIB = 1024 ** 3;
const PROBE_MARKER = "MYCELLIOS_RUNTIME_PROBE=";
const ACCELERATOR_DIRECTORY = "accelerator-runtimes-v1";
const ACCELERATOR_MANIFEST = "accelerator-runtime.json";
const PORTABLE_MANIFEST = "runtime-manifest.json";

export type AcceleratorBackend = "cuda" | "rocm";
export type EffectiveRuntimeBackend = "cpu" | AcceleratorBackend;
export type AcceleratorRuntimeStatus = "cpu-ready" | "gpu-ready" | "gpu-fallback";

export interface PortableRuntimeManifest {
  schema: typeof PORTABLE_RUNTIME_SCHEMA;
  platform: string;
  arch: string;
  pythonVersion: string;
  pythonAbi: string;
  executable: string;
  torchVersion: string;
  transformersVersion: string;
  accelerateVersion: string;
  safetensorsVersion: string;
  backend: "cpu";
}

export interface AcceleratorHardware {
  platform?: NodeJS.Platform | undefined;
  arch?: string | undefined;
  gpuVendor?: string | undefined;
  gpuModel?: string | undefined;
  cpuModel?: string | undefined;
}

export interface AcceleratorProbe {
  backend: AcceleratorBackend;
  device: string;
  deviceName: string;
  torchVersion: string;
  pythonVersion: string;
  cudaVersion: string | null;
  hipVersion: string | null;
  precision: "float16";
  operation: "fp16_matmul";
  checksum: number;
  totalMemoryBytes: number;
  allocatedBytes: number;
  reservedBytes: number;
}

export interface AcceleratorRuntimeResult {
  status: AcceleratorRuntimeStatus;
  requestedBackend: EffectiveRuntimeBackend;
  effectiveBackend: EffectiveRuntimeBackend;
  deviceType: "cpu" | "gpu";
  runtimeRoot: string;
  pythonExecutable: string;
  pythonPathAdditions: string[];
  pathAdditions: string[];
  deviceName: string;
  precision: "float32" | "float16";
  torchVersion: string;
  probe?: AcceleratorProbe | undefined;
  fallbackReason?: string | undefined;
}

export interface RuntimeCommandOptions {
  cwd?: string | undefined;
  env?: NodeJS.ProcessEnv | undefined;
}

export interface RuntimeCommandResult {
  code: number;
  stdout: string;
  stderr: string;
}

export type RuntimeCommandRunner = (
  executable: string,
  args: readonly string[],
  options?: RuntimeCommandOptions,
) => Promise<RuntimeCommandResult>;

export interface PrepareAcceleratorRuntimeOptions {
  baseRuntimeRoot: string;
  userDataPath: string;
  hardware: AcceleratorHardware;
  /**
   * Provisioning is deliberately opt-in. Importing this module, running tests,
   * and producing an installer can therefore never download multi-gigabyte
   * vendor runtimes. The packaged desktop enables it from its runtime startup.
   */
  allowProvisioning?: boolean | undefined;
  preferredBackend?: "auto" | EffectiveRuntimeBackend | undefined;
  commandRunner?: RuntimeCommandRunner | undefined;
  onStatus?: ((status: string) => void) | undefined;
}

interface AcceleratorPack {
  id: string;
  backend: AcceleratorBackend;
  torchVersion: string;
  minimumFreeBytes: number;
  installGroups: readonly (readonly string[])[];
}

interface AcceleratorRuntimeManifest {
  schema: typeof ACCELERATOR_RUNTIME_SCHEMA;
  packId: string;
  backend: AcceleratorBackend;
  baseRuntimeSchema: typeof PORTABLE_RUNTIME_SCHEMA;
  pythonVersion: string;
  torchVersion: string;
  packageUrls: string[];
  probe: AcceleratorProbe;
  environment: {
    pythonPathAdditions: string[];
    pathAdditions: string[];
  };
}

const CUDA_TORCH_URL =
  "https://download-r2.pytorch.org/whl/cu126/" +
  "torch-2.13.0%2Bcu126-cp312-cp312-win_amd64.whl" +
  "#sha256=380081ea098bf2b9e727aa85205d94790d884d17c62df3bb00a4f6a1047010a2";
const AMD_ROCM_BASE = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1";
const AMD_SDK_URLS = [
  `${AMD_ROCM_BASE}/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl#sha256=f68989d48df71cbfc3cb68bf705dc37c0f56e9666feddb59a1a0f5ff7539fe1c`,
  `${AMD_ROCM_BASE}/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl#sha256=19e6ee67e13432b7c1e8a4077df795dcb1239546ae314700c4b4d97e5b4b8f63`,
  `${AMD_ROCM_BASE}/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl#sha256=c7fe0b0731af8896093ff69e11496830d3cb6a4aed73e895c60b7cbdc200be92`,
  `${AMD_ROCM_BASE}/rocm-7.2.1.tar.gz#sha256=9084902eaa69213a00a90784ad89e6e5fe73c702df0cc6cc3a70d777c7a6142b`,
] as const;
const AMD_TORCH_URL =
  `${AMD_ROCM_BASE}/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl` +
  "#sha256=e88bf270163b48f7f27f7ea3db5ffb3be4ba107301933022bcb3c6ddedfeeabb";

export const WINDOWS_ACCELERATOR_PACKS: Readonly<Record<AcceleratorBackend, AcceleratorPack>> = {
  cuda: {
    id: "win-x64-py312-torch213-cu126-v1",
    backend: "cuda",
    torchVersion: "2.13.0+cu126",
    // The compressed torch wheel alone is 2.42 GiB. Keep enough room for the
    // wheel cache, its extracted DLLs, the copied base runtime, and rollback.
    minimumFreeBytes: 8 * GIB,
    installGroups: [[CUDA_TORCH_URL]],
  },
  rocm: {
    id: "win-x64-py312-torch291-rocm721-v1",
    backend: "rocm",
    torchVersion: "2.9.1+rocm7.2.1",
    // AMD's official Windows install is roughly 2 GiB compressed before the
    // portable Python base and extracted libraries are counted.
    minimumFreeBytes: 8 * GIB,
    installGroups: [AMD_SDK_URLS, [AMD_TORCH_URL]],
  },
};

/** Select only combinations for which this release has a pinned Windows pack. */
export function selectAcceleratorPack(
  hardware: AcceleratorHardware,
  preferredBackend: "auto" | EffectiveRuntimeBackend = "auto",
): AcceleratorPack | null {
  const platform = hardware.platform ?? process.platform;
  const arch = hardware.arch ?? process.arch;
  if (preferredBackend === "cpu" || platform !== "win32" || arch !== "x64") return null;
  if (preferredBackend === "cuda" || preferredBackend === "rocm") {
    return WINDOWS_ACCELERATOR_PACKS[preferredBackend];
  }
  const vendor = (hardware.gpuVendor ?? "").trim().toLowerCase();
  const gpu = (hardware.gpuModel ?? "").toLowerCase();
  const cpu = (hardware.cpuModel ?? "").toLowerCase();
  if (vendor === "nvidia" || gpu.includes("nvidia") || /\b(?:rtx|gtx)\b/.test(gpu)) {
    return WINDOWS_ACCELERATOR_PACKS.cuda;
  }
  if (
    vendor === "amd" &&
    (/(?:radeon.*(?:890m|8050s|8060s))/.test(gpu) ||
      /ryzen ai (?:max\+? )?(?:9 )?(?:hx )?(?:365|370|375|385|390|395|465|470|475)/.test(cpu))
  ) {
    return WINDOWS_ACCELERATOR_PACKS.rocm;
  }
  return null;
}

export async function prepareAcceleratorRuntime(
  options: PrepareAcceleratorRuntimeOptions,
): Promise<AcceleratorRuntimeResult> {
  const baseRoot = resolve(options.baseRuntimeRoot);
  const baseManifest = await readPortableRuntimeManifest(baseRoot);
  const basePython = runtimePythonExecutable(baseRoot, baseManifest.executable);
  await requireFile(basePython, "portable CPU Python executable");
  const baseEnvironment = runtimeEnvironmentAdditions(baseRoot);
  const preferred = options.preferredBackend ?? "auto";
  const pack = selectAcceleratorPack(options.hardware, preferred);
  if (pack === null) {
    return cpuResult(baseRoot, basePython, baseManifest, baseEnvironment);
  }

  const acceleratorRoot = resolve(options.userDataPath, ACCELERATOR_DIRECTORY);
  const target = resolve(acceleratorRoot, pack.id);
  assertDirectChild(acceleratorRoot, target);
  const runner = options.commandRunner ?? runRuntimeCommand;
  await mkdir(acceleratorRoot, { recursive: true });

  let cachedFailure: string | undefined;
  if (existsSync(target)) {
    try {
      const cached = await readAcceleratorManifest(target, pack);
      const additions = environmentFromManifest(target, cached);
      const python = runtimePythonExecutable(target, baseManifest.executable);
      const probe = await probeAcceleratorRuntime(
        python,
        pack.backend,
        additions,
        runner,
      );
      validatePackProbe(probe, pack);
      return gpuResult(target, python, pack, additions, probe);
    } catch (error) {
      cachedFailure = shortError(error);
    }
  }

  if (options.allowProvisioning !== true) {
    return cpuFallbackResult(
      baseRoot,
      basePython,
      baseManifest,
      baseEnvironment,
      pack.backend,
      cachedFailure
        ? `Cached ${pack.backend.toUpperCase()} runtime is invalid: ${cachedFailure}`
        : `${pack.backend.toUpperCase()} runtime is not installed yet`,
    );
  }

  const staging = resolve(acceleratorRoot, `${pack.id}.staging-${randomUUID()}`);
  assertDirectChild(acceleratorRoot, staging);
  options.onStatus?.(`Preparing ${pack.backend.toUpperCase()} runtime`);
  try {
    await requireFreeSpace(acceleratorRoot, pack.minimumFreeBytes);
    if (existsSync(target)) await rm(target, { recursive: true, force: true });
    await rm(staging, { recursive: true, force: true });
    await cp(baseRoot, staging, { recursive: true, force: false, errorOnExist: true });
    const python = runtimePythonExecutable(staging, baseManifest.executable);
    const initialEnvironment = runtimeEnvironmentAdditions(staging);
    await runChecked(
      runner,
      python,
      ["-m", "pip", "uninstall", "--yes", "torch"],
      commandEnvironment(initialEnvironment),
      "remove the CPU torch wheel",
    );
    for (const [index, urls] of pack.installGroups.entries()) {
      options.onStatus?.(
        `Installing ${pack.backend.toUpperCase()} runtime (${index + 1}/${pack.installGroups.length})`,
      );
      await runChecked(
        runner,
        python,
        [
          "-m",
          "pip",
          "install",
          "--disable-pip-version-check",
          "--no-input",
          "--no-cache-dir",
          "--no-deps",
          ...urls,
        ],
        commandEnvironment(initialEnvironment),
        `install ${pack.backend.toUpperCase()} runtime group ${index + 1}`,
      );
    }
    const additions = runtimeEnvironmentAdditions(staging, true);
    options.onStatus?.(`Verifying ${pack.backend.toUpperCase()} on a real FP16 operation`);
    const probe = await probeAcceleratorRuntime(
      python,
      pack.backend,
      additions,
      runner,
    );
    validatePackProbe(probe, pack);
    const manifest = acceleratorManifest(pack, baseManifest, staging, additions, probe);
    await writeFile(
      join(staging, ACCELERATOR_MANIFEST),
      `${JSON.stringify(manifest, null, 2)}\n`,
      "utf8",
    );
    await rename(staging, target);
    const targetAdditions = environmentFromManifest(target, manifest);
    return gpuResult(
      target,
      runtimePythonExecutable(target, baseManifest.executable),
      pack,
      targetAdditions,
      probe,
    );
  } catch (error) {
    await rm(staging, { recursive: true, force: true }).catch(() => undefined);
    return cpuFallbackResult(
      baseRoot,
      basePython,
      baseManifest,
      baseEnvironment,
      pack.backend,
      shortError(error),
    );
  }
}

export async function readPortableRuntimeManifest(
  runtimeRoot: string,
): Promise<PortableRuntimeManifest> {
  const path = join(resolve(runtimeRoot), PORTABLE_MANIFEST);
  let raw: unknown;
  try {
    raw = JSON.parse(await readFile(path, "utf8"));
  } catch (error) {
    throw new Error(`portable runtime v2 manifest is missing or invalid: ${shortError(error)}`);
  }
  if (!isRecord(raw) || raw.schema !== PORTABLE_RUNTIME_SCHEMA) {
    throw new Error("portable runtime does not use the v2 manifest schema");
  }
  for (const key of [
    "platform",
    "arch",
    "pythonVersion",
    "pythonAbi",
    "executable",
    "torchVersion",
    "transformersVersion",
    "accelerateVersion",
    "safetensorsVersion",
  ] as const) {
    if (typeof raw[key] !== "string" || !raw[key].trim()) {
      throw new Error(`portable runtime manifest has invalid ${key}`);
    }
  }
  if (
    raw.pythonAbi !== "cp312" ||
    typeof raw.pythonVersion !== "string" ||
    !raw.pythonVersion.startsWith("3.12.")
  ) {
    throw new Error("portable runtime must use CPython 3.12");
  }
  if (raw.backend !== "cpu") throw new Error("portable base runtime must be CPU-only");
  return raw as unknown as PortableRuntimeManifest;
}

export async function probeAcceleratorRuntime(
  pythonExecutable: string,
  expectedBackend: AcceleratorBackend,
  additions: Pick<AcceleratorRuntimeResult, "pythonPathAdditions" | "pathAdditions">,
  runner: RuntimeCommandRunner = runRuntimeCommand,
): Promise<AcceleratorProbe> {
  const result = await runner(
    pythonExecutable,
    ["-c", acceleratorProbeScript(), expectedBackend],
    { env: commandEnvironment(additions) },
  );
  if (result.code !== 0) {
    throw new Error(
      `accelerator probe failed (${result.code}): ${tail(result.stderr || result.stdout)}`,
    );
  }
  const marker = result.stdout
    .split(/\r?\n/)
    .reverse()
    .find((line) => line.startsWith(PROBE_MARKER));
  if (!marker) throw new Error("accelerator probe did not emit verified metadata");
  let value: unknown;
  try {
    value = JSON.parse(marker.slice(PROBE_MARKER.length));
  } catch {
    throw new Error("accelerator probe metadata is not valid JSON");
  }
  return validateProbe(value, expectedBackend);
}

export function acceleratorRuntimeTargetPath(
  userDataPath: string,
  backend: AcceleratorBackend,
): string {
  return resolve(userDataPath, ACCELERATOR_DIRECTORY, WINDOWS_ACCELERATOR_PACKS[backend].id);
}

function acceleratorProbeScript(): string {
  return [
    "import json, math, sys, torch",
    "expected = sys.argv[1]",
    "hip = getattr(torch.version, 'hip', None)",
    "cuda = getattr(torch.version, 'cuda', None)",
    "actual = 'rocm' if hip else ('cuda' if cuda else 'cpu')",
    "assert actual == expected, f'expected {expected}, got {actual}'",
    "assert torch.cuda.is_available() and torch.cuda.device_count() > 0",
    "device = torch.device('cuda:0')",
    // Keep every operand exactly representable in FP16. arange(65536) in
    // float16 overflows above 65504, making an otherwise healthy GPU produce
    // an infinite/NaN checksum and forcing a false CPU fallback.
    "a = torch.full((256, 256), 0.5, device=device, dtype=torch.float16)",
    "b = torch.eye(256, device=device, dtype=torch.float16)",
    "c = a @ b",
    "torch.cuda.synchronize(device)",
    "checksum = float(c.float().sum().item())",
    "assert c.device.type == 'cuda' and c.dtype == torch.float16",
    "assert math.isfinite(checksum) and checksum != 0.0",
    "props = torch.cuda.get_device_properties(0)",
    "payload = {'backend': actual, 'device': str(device), 'device_name': str(props.name), 'torch_version': str(torch.__version__), 'python_version': '.'.join(map(str, sys.version_info[:3])), 'cuda_version': str(cuda) if cuda else None, 'hip_version': str(hip) if hip else None, 'precision': 'float16', 'operation': 'fp16_matmul', 'checksum': checksum, 'total_memory_bytes': int(props.total_memory), 'allocated_bytes': int(torch.cuda.memory_allocated(device)), 'reserved_bytes': int(torch.cuda.memory_reserved(device))}",
    `print('${PROBE_MARKER}' + json.dumps(payload, separators=(',', ':')), flush=True)`,
  ].join("\n");
}

function validateProbe(value: unknown, expectedBackend: AcceleratorBackend): AcceleratorProbe {
  if (!isRecord(value) || value.backend !== expectedBackend) {
    throw new Error(`accelerator probe did not confirm ${expectedBackend}`);
  }
  if (value.device !== "cuda:0" || value.precision !== "float16" || value.operation !== "fp16_matmul") {
    throw new Error("accelerator probe did not execute the certified FP16 CUDA/ROCm path");
  }
  for (const key of ["device_name", "torch_version", "python_version"] as const) {
    if (typeof value[key] !== "string" || !value[key].trim()) {
      throw new Error(`accelerator probe has invalid ${key}`);
    }
  }
  for (const key of ["checksum", "total_memory_bytes", "allocated_bytes", "reserved_bytes"] as const) {
    if (typeof value[key] !== "number" || !Number.isFinite(value[key]) || value[key] < 0) {
      throw new Error(`accelerator probe has invalid ${key}`);
    }
  }
  const checksum = value.checksum as number;
  const totalMemoryBytes = value.total_memory_bytes as number;
  const allocatedBytes = value.allocated_bytes as number;
  if (checksum === 0 || totalMemoryBytes <= 0 || allocatedBytes <= 0) {
    throw new Error("accelerator probe has no physical device allocation evidence");
  }
  if (expectedBackend === "cuda" && (typeof value.cuda_version !== "string" || value.hip_version !== null)) {
    throw new Error("accelerator probe CUDA build metadata is inconsistent");
  }
  if (expectedBackend === "rocm" && (typeof value.hip_version !== "string" || value.cuda_version !== null)) {
    throw new Error("accelerator probe ROCm build metadata is inconsistent");
  }
  return {
    backend: expectedBackend,
    device: "cuda:0",
    deviceName: value.device_name as string,
    torchVersion: value.torch_version as string,
    pythonVersion: value.python_version as string,
    cudaVersion: value.cuda_version as string | null,
    hipVersion: value.hip_version as string | null,
    precision: "float16",
    operation: "fp16_matmul",
    checksum,
    totalMemoryBytes,
    allocatedBytes,
    reservedBytes: value.reserved_bytes as number,
  };
}

function validatePackProbe(probe: AcceleratorProbe, pack: AcceleratorPack): void {
  if (probe.torchVersion !== pack.torchVersion) {
    throw new Error(
      `runtime reported torch ${probe.torchVersion}; expected ${pack.torchVersion}`,
    );
  }
  if (!probe.pythonVersion.startsWith("3.12.")) {
    throw new Error(`runtime reported Python ${probe.pythonVersion}; expected CPython 3.12`);
  }
  if (pack.backend === "cuda" && !/nvidia/i.test(probe.deviceName)) {
    throw new Error(`CUDA runtime reported an unexpected device: ${probe.deviceName}`);
  }
  if (pack.backend === "rocm" && !/(amd|radeon)/i.test(probe.deviceName)) {
    throw new Error(`ROCm runtime reported an unexpected device: ${probe.deviceName}`);
  }
}

function runtimePythonExecutable(root: string, manifestExecutable?: string): string {
  if (manifestExecutable) {
    if (isAbsolute(manifestExecutable) || manifestExecutable.includes("..")) {
      throw new Error("runtime manifest executable must be relative");
    }
    return resolve(root, manifestExecutable);
  }
  const portable = join(root, "python.exe");
  return existsSync(portable) ? portable : join(root, "Scripts", "python.exe");
}

function runtimeEnvironmentAdditions(
  root: string,
  discoverLibraries = false,
): Pick<AcceleratorRuntimeResult, "pythonPathAdditions" | "pathAdditions"> {
  const sitePackages = join(root, "Lib", "site-packages");
  const paths = [root, join(root, "Scripts"), join(sitePackages, "torch", "lib")]
    .filter((path) => existsSync(path));
  if (discoverLibraries && existsSync(sitePackages)) {
    paths.push(...discoverDllDirectories(sitePackages));
  }
  return {
    pythonPathAdditions: existsSync(sitePackages) ? [sitePackages] : [],
    pathAdditions: unique(paths),
  };
}

function discoverDllDirectories(root: string): string[] {
  const found: string[] = [];
  const visit = (directory: string, depth: number): void => {
    if (depth > 7 || found.length >= 64) return;
    let entries;
    try {
      entries = readdirSync(directory, { withFileTypes: true });
    } catch {
      return;
    }
    let hasDll = false;
    for (const entry of entries) {
      if (entry.isFile() && entry.name.toLowerCase().endsWith(".dll")) hasDll = true;
    }
    if (hasDll) found.push(directory);
    for (const entry of entries) {
      if (entry.isDirectory() && entry.name !== "__pycache__") {
        visit(join(directory, entry.name), depth + 1);
      }
    }
  };
  visit(root, 0);
  return found;
}

function commandEnvironment(
  additions: Pick<AcceleratorRuntimeResult, "pythonPathAdditions" | "pathAdditions">,
): NodeJS.ProcessEnv {
  return {
    ...process.env,
    PATH: [...additions.pathAdditions, process.env.PATH].filter(Boolean).join(";"),
    PYTHONPATH: [...additions.pythonPathAdditions, process.env.PYTHONPATH]
      .filter(Boolean)
      .join(";"),
    PYTHONNOUSERSITE: "1",
  };
}

function acceleratorManifest(
  pack: AcceleratorPack,
  base: PortableRuntimeManifest,
  root: string,
  additions: Pick<AcceleratorRuntimeResult, "pythonPathAdditions" | "pathAdditions">,
  probe: AcceleratorProbe,
): AcceleratorRuntimeManifest {
  return {
    schema: ACCELERATOR_RUNTIME_SCHEMA,
    packId: pack.id,
    backend: pack.backend,
    baseRuntimeSchema: PORTABLE_RUNTIME_SCHEMA,
    pythonVersion: base.pythonVersion,
    torchVersion: probe.torchVersion,
    packageUrls: pack.installGroups.flatMap((group) => [...group]),
    probe,
    environment: {
      pythonPathAdditions: additions.pythonPathAdditions.map((path) => safeRelative(root, path)),
      pathAdditions: additions.pathAdditions.map((path) => safeRelative(root, path)),
    },
  };
}

async function readAcceleratorManifest(
  root: string,
  pack: AcceleratorPack,
): Promise<AcceleratorRuntimeManifest> {
  const raw = JSON.parse(await readFile(join(root, ACCELERATOR_MANIFEST), "utf8")) as unknown;
  const expectedPackageUrls = pack.installGroups.flatMap((group) => [...group]);
  if (
    !isRecord(raw) ||
    raw.schema !== ACCELERATOR_RUNTIME_SCHEMA ||
    raw.packId !== pack.id ||
    raw.backend !== pack.backend ||
    raw.baseRuntimeSchema !== PORTABLE_RUNTIME_SCHEMA ||
    raw.torchVersion !== pack.torchVersion ||
    !stringArray(raw.packageUrls) ||
    raw.packageUrls.length !== expectedPackageUrls.length ||
    raw.packageUrls.some((url, index) => url !== expectedPackageUrls[index]) ||
    !isRecord(raw.environment) ||
    !stringArray(raw.environment.pythonPathAdditions) ||
    !stringArray(raw.environment.pathAdditions)
  ) {
    throw new Error("accelerator runtime manifest does not match the selected pack");
  }
  return raw as unknown as AcceleratorRuntimeManifest;
}

function environmentFromManifest(
  root: string,
  manifest: AcceleratorRuntimeManifest,
): Pick<AcceleratorRuntimeResult, "pythonPathAdditions" | "pathAdditions"> {
  return {
    pythonPathAdditions: manifest.environment.pythonPathAdditions.map((path) => safeResolve(root, path)),
    pathAdditions: manifest.environment.pathAdditions.map((path) => safeResolve(root, path)),
  };
}

function cpuResult(
  root: string,
  python: string,
  manifest: PortableRuntimeManifest,
  additions: Pick<AcceleratorRuntimeResult, "pythonPathAdditions" | "pathAdditions">,
): AcceleratorRuntimeResult {
  return {
    status: "cpu-ready",
    requestedBackend: "cpu",
    effectiveBackend: "cpu",
    deviceType: "cpu",
    runtimeRoot: root,
    pythonExecutable: python,
    ...additions,
    deviceName: "CPU",
    precision: "float32",
    torchVersion: manifest.torchVersion,
  };
}

function cpuFallbackResult(
  root: string,
  python: string,
  manifest: PortableRuntimeManifest,
  additions: Pick<AcceleratorRuntimeResult, "pythonPathAdditions" | "pathAdditions">,
  requestedBackend: AcceleratorBackend,
  fallbackReason: string,
): AcceleratorRuntimeResult {
  return {
    ...cpuResult(root, python, manifest, additions),
    status: "gpu-fallback",
    requestedBackend,
    fallbackReason,
  };
}

function gpuResult(
  root: string,
  python: string,
  pack: AcceleratorPack,
  additions: Pick<AcceleratorRuntimeResult, "pythonPathAdditions" | "pathAdditions">,
  probe: AcceleratorProbe,
): AcceleratorRuntimeResult {
  return {
    status: "gpu-ready",
    requestedBackend: pack.backend,
    effectiveBackend: pack.backend,
    deviceType: "gpu",
    runtimeRoot: root,
    pythonExecutable: python,
    ...additions,
    deviceName: probe.deviceName,
    precision: "float16",
    torchVersion: probe.torchVersion,
    probe,
  };
}

async function runChecked(
  runner: RuntimeCommandRunner,
  executable: string,
  args: readonly string[],
  env: NodeJS.ProcessEnv,
  action: string,
): Promise<void> {
  const result = await runner(executable, args, { env });
  if (result.code !== 0) {
    throw new Error(`${action} failed (${result.code}): ${tail(result.stderr || result.stdout)}`);
  }
}

export function runRuntimeCommand(
  executable: string,
  args: readonly string[],
  options: RuntimeCommandOptions = {},
): Promise<RuntimeCommandResult> {
  return new Promise((resolveResult, reject) => {
    const child = spawn(executable, [...args], {
      cwd: options.cwd,
      env: options.env,
      shell: false,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer) => { stdout = tail(stdout + chunk.toString("utf8")); });
    child.stderr.on("data", (chunk: Buffer) => { stderr = tail(stderr + chunk.toString("utf8")); });
    child.once("error", reject);
    child.once("close", (code) => resolveResult({ code: code ?? -1, stdout, stderr }));
  });
}

async function requireFreeSpace(path: string, requiredBytes: number): Promise<void> {
  const info = await statfs(path);
  const available = Number(info.bavail) * Number(info.bsize);
  if (!Number.isFinite(available) || available < requiredBytes) {
    throw new Error(
      `accelerator runtime needs ${Math.ceil(requiredBytes / GIB)} GiB free; ` +
      `${Math.max(0, Math.floor(available / GIB))} GiB is available`,
    );
  }
}

async function requireFile(path: string, label: string): Promise<void> {
  try {
    const info = await stat(path);
    if (!info.isFile()) throw new Error("not a file");
  } catch {
    throw new Error(`${label} is missing: ${path}`);
  }
}

function assertDirectChild(parent: string, child: string): void {
  if (dirname(child) !== resolve(parent) || basename(child) === "") {
    throw new Error("accelerator runtime target escaped its managed directory");
  }
}

function safeRelative(root: string, path: string): string {
  const value = relative(resolve(root), resolve(path));
  if (!value || value === ".") return ".";
  if (value.startsWith("..") || isAbsolute(value)) {
    throw new Error("runtime environment path escaped its root");
  }
  return value;
}

function safeResolve(root: string, value: string): string {
  if (isAbsolute(value)) throw new Error("runtime manifest path must be relative");
  const result = resolve(root, value);
  const back = relative(resolve(root), result);
  if (back.startsWith("..") || isAbsolute(back)) {
    throw new Error("runtime manifest path escaped its root");
  }
  return result;
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values.map((value) => resolve(value)))];
}

function tail(value: string, limit = 64 * 1024): string {
  return value.length <= limit ? value : value.slice(-limit);
}

function shortError(error: unknown): string {
  const value = error instanceof Error ? error.message : String(error);
  return tail(value.trim() || "unknown runtime error", 1_000);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}
