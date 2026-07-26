import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import {
  cp,
  mkdir,
  open,
  readdir,
  readFile,
  rename,
  rm,
  stat,
  statfs,
  writeFile,
} from "node:fs/promises";
import { createReadStream, existsSync, readdirSync } from "node:fs";
import { basename, delimiter, dirname, isAbsolute, join, relative, resolve } from "node:path";
import {
  createArtifactSwarmDownloader,
  type ArtifactSwarmDownloaderOptions,
} from "../model-fabric/artifact-swarm.js";
import { gpuModelsMatch } from "../worker/hardware.js";

export const PORTABLE_RUNTIME_SCHEMA = "mycellios-distribution-runtime/4" as const;
export const ACCELERATOR_RUNTIME_SCHEMA = "mycellios-accelerator-runtime/1" as const;

const GIB = 1024 ** 3;
const PROBE_MARKER = "MYCELLIOS_RUNTIME_PROBE=";
export const PORTABLE_RUNTIME_PROBE_MARKER = "MYCELLIOS_PORTABLE_RUNTIME_PROBE=";
const ACCELERATOR_DIRECTORY = "accelerator-runtimes-v1";
const ACCELERATOR_MANIFEST = "accelerator-runtime.json";
const WINDOWS_CUDA_RUNTIME_DIRECTORY = "mcg";
const WINDOWS_CUDA_RUNTIME_TARGET = "cuda";
// Keep provisioning paths short enough for Python wheels on Windows systems
// where Win32 long-path support is not enabled. Some PyTorch headers are more
// than 120 characters below site-packages, so repeating the full pack id and a
// UUID in the staging directory can cross MAX_PATH.
const ACCELERATOR_STAGING_PREFIX = "stg-";
const PORTABLE_MANIFEST = "runtime-manifest.json";

export type AcceleratorBackend = "cuda" | "rocm" | "mps" | "xpu";
export type EffectiveRuntimeBackend = "cpu" | AcceleratorBackend;
export type AcceleratorRuntimeStatus = "cpu-ready" | "gpu-ready" | "gpu-fallback";

export type AcceleratorProgressPhase =
  | "detecting"
  | "checking-cache"
  | "checking-prerequisites"
  | "copying-base"
  | "downloading"
  | "verifying-package"
  | "installing"
  | "physical-probe"
  | "activating"
  | "ready"
  | "fallback"
  | "blocked";

export type AcceleratorProgressIssueCode =
  | "no-gpu"
  | "unsupported-platform"
  | "unsupported-gpu"
  | "installation-required"
  | "cache-invalid"
  | "disk-space"
  | "network"
  | "integrity"
  | "install"
  | "physical-probe";

export interface AcceleratorProgressIssue {
  code: AcceleratorProgressIssueCode;
  message: string;
  retryable: boolean;
  action: string;
  requiredBytes?: number | undefined;
  availableBytes?: number | undefined;
}

export interface AcceleratorDownloadProgress {
  artifact: string;
  artifactIndex: number;
  artifactCount: number;
  bytesDownloaded: number;
  bytesTotal: number;
  aggregateDownloaded: number;
  aggregateTotal: number;
  bytesPerSecond: number | null;
  etaSeconds: number | null;
  resumed: boolean;
}

export interface AcceleratorProgressEvent {
  sequence: number;
  at: string;
  phase: AcceleratorProgressPhase;
  backend: AcceleratorBackend | null;
  gpuVendor: string | null;
  gpuModel: string | null;
  cpuAvailable: boolean;
  percent: number;
  message: string;
  /** False for high-frequency byte ticks that should update UI but not append a log row. */
  recordLog: boolean;
  download?: AcceleratorDownloadProgress | undefined;
  issue?: AcceleratorProgressIssue | undefined;
}

export interface AcceleratorPackageArtifact {
  label: string;
  /** Retain the hash fragment so old content-addressed manifests invalidate safely. */
  url: string;
  sha256: string;
  sizeBytes: number;
}

export interface AcceleratorArtifactTransfer {
  bytesDownloaded: number;
  bytesTotal: number;
  bytesPerSecond: number | null;
  etaSeconds: number | null;
  resumed: boolean;
}

export type AcceleratorArtifactDownloader = (
  artifact: AcceleratorPackageArtifact,
  cacheRoot: string,
  onProgress: (progress: AcceleratorArtifactTransfer) => void,
) => Promise<string>;

export type AcceleratorArtifactSwarmOptions = Omit<
  ArtifactSwarmDownloaderOptions,
  "originDownloader"
>;

export interface PortableRuntimeManifest {
  schema: typeof PORTABLE_RUNTIME_SCHEMA;
  platform: string;
  arch: string;
  pythonVersion: string;
  pythonAbi: string;
  executable: string;
  pythonArtifact: PortablePythonArtifactManifest;
  wheelLock: PortableRuntimeWheelLockManifest;
  torchVersion: string;
  transformersVersion: string;
  accelerateVersion: string;
  safetensorsVersion: string;
  aiohttpVersion: string;
  sentencepieceVersion: string;
  numpyVersion: string;
  backend: "cpu";
  bundledAccelerators: string[];
}

export interface PortableRuntimeWheelLockManifest {
  path: string;
  sha256: string;
}

export interface PortablePythonArtifactManifest {
  source: "astral-sh/python-build-standalone";
  release: "20260510";
  version: "3.12.13";
  flavor: "install_only_stripped";
  filename: string;
  url: string;
  size: number;
  sha256: string;
}

export interface AcceleratorHardware {
  platform?: NodeJS.Platform | undefined;
  arch?: string | undefined;
  osRelease?: string | undefined;
  gpuVendor?: string | undefined;
  gpuModel?: string | undefined;
  gpuDeviceIndex?: number | undefined;
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
  /** Exact Torch device request used by the distributed stage launcher. */
  launchDevice: "cpu" | "mps" | `cuda:${number}` | `xpu:${number}`;
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
  artifactDownloader?: AcceleratorArtifactDownloader | undefined;
  /**
   * Optional native peer source. Every peer chunk is verified against a
   * signed Mycellios manifest; the pinned HTTPS downloader remains the
   * authoritative fallback when no peer can provide all required bytes.
   */
  artifactSwarm?: AcceleratorArtifactSwarmOptions | undefined;
  onStatus?: ((status: string) => void) | undefined;
  onProgress?: ((event: AcceleratorProgressEvent) => void) | undefined;
}

interface AcceleratorPack {
  id: string;
  backend: AcceleratorBackend;
  platform: NodeJS.Platform;
  arch: string;
  strategy: "bundled" | "pinned-download";
  launchDevice: "cuda" | "mps" | "xpu";
  torchVersion: string;
  minimumFreeBytes: number;
  installGroups: readonly (readonly AcceleratorPackageArtifact[])[];
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

const CUDA_TORCH = artifact(
  "PyTorch CUDA 12.6",
  "https://download-r2.pytorch.org/whl/cu126/" +
    "torch-2.13.0%2Bcu126-cp312-cp312-win_amd64.whl",
  "380081ea098bf2b9e727aa85205d94790d884d17c62df3bb00a4f6a1047010a2",
  2_594_590_371,
);
const AMD_ROCM_BASE = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1";
const AMD_SDK_ARTIFACTS = [
  artifact("ROCm SDK core", `${AMD_ROCM_BASE}/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl`, "f68989d48df71cbfc3cb68bf705dc37c0f56e9666feddb59a1a0f5ff7539fe1c", 644_793_492),
  artifact("ROCm SDK development files", `${AMD_ROCM_BASE}/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl`, "19e6ee67e13432b7c1e8a4077df795dcb1239546ae314700c4b4d97e5b4b8f63", 232_840_013),
  artifact("ROCm SDK libraries", `${AMD_ROCM_BASE}/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl`, "c7fe0b0731af8896093ff69e11496830d3cb6a4aed73e895c60b7cbdc200be92", 489_964_648),
  // PyTorch imports `rocm_sdk` during its own module initialization. AMD
  // distributes that Python bootstrap separately from the three binary SDK
  // wheels, so it must be installed explicitly when dependencies are pinned
  // and `pip --no-deps` is used.
  artifact("ROCm SDK bootstrap", `${AMD_ROCM_BASE}/rocm-7.2.1.tar.gz`, "9084902eaa69213a00a90784ad89e6e5fe73c702df0cc6cc3a70d777c7a6142b", 15_940),
] as const;
const AMD_TORCH = artifact(
  "PyTorch ROCm 7.2.1",
  `${AMD_ROCM_BASE}/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl`,
  "e88bf270163b48f7f27f7ea3db5ffb3be4ba107301933022bcb3c6ddedfeeabb",
  821_065_907,
);

const CERTIFIED_PORTABLE_PYTHON = {
  "win32/x64": {
    filename: "cpython-3.12.13+20260510-x86_64-pc-windows-msvc-install_only_stripped.tar.gz",
    size: 21_921_642,
    sha256: "24168aff2e7d93784c6a436124c4ebb79b076a4e289bde4902c08333507b71d0",
  },
  "linux/x64": {
    filename: "cpython-3.12.13+20260510-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz",
    size: 34_075_380,
    sha256: "d480f5d5878910ecbae212bf23bd7c25d7b209eb8cf5e98823c977384d272e88",
  },
  "darwin/arm64": {
    filename: "cpython-3.12.13+20260510-aarch64-apple-darwin-install_only_stripped.tar.gz",
    size: 24_942_229,
    sha256: "55bc1a5edbc8ac4da0081f4f5731ed2d1ed10c57cb37a820b2a0dbc7cad742e9",
  },
  "darwin/x64": {
    filename: "cpython-3.12.13+20260510-x86_64-apple-darwin-install_only_stripped.tar.gz",
    size: 24_639_521,
    sha256: "6bab7fa97d4f2ddba86da0e05acff66c53b5edaca1df8edcf00ddca785a9c59b",
  },
} as const;

const CERTIFIED_PORTABLE_RUNTIME = {
  "win32/x64": { executable: "python.exe", torchVersion: "2.13.0+cpu", transformersVersion: "5.14.1", bundledAccelerators: [] },
  "linux/x64": { executable: "bin/python3", torchVersion: "2.13.0+cpu", transformersVersion: "5.14.1", bundledAccelerators: [] },
  "darwin/arm64": { executable: "bin/python3", torchVersion: "2.11.0", transformersVersion: "5.14.1", bundledAccelerators: ["mps"] },
  "darwin/x64": { executable: "bin/python3", torchVersion: "2.2.2", transformersVersion: "4.57.3", bundledAccelerators: [] },
} as const;

const CERTIFIED_PORTABLE_WHEEL_LOCKS = {
  "win32/x64": {
    path: "scripts/wheel-locks/win32-x64-cp312.txt",
    sha256: "2a5de3d3f2e3ba4ebac91e1efe068159e81263d3ccd6a3d93333078e753c6c65",
  },
  "linux/x64": {
    path: "scripts/wheel-locks/linux-x64-cp312.txt",
    sha256: "090355535c96e7202ae761a18dadc5d66c9e98493a148eba16ae3c3e7c95c59d",
  },
  "darwin/arm64": {
    path: "scripts/wheel-locks/darwin-arm64-cp312.txt",
    sha256: "75c84302540edc1b7f6c1620c3474b7ccbf431dd6df881717f74ebf419c56348",
  },
  "darwin/x64": {
    path: "scripts/wheel-locks/darwin-x64-cp312.txt",
    sha256: "808a7fd6894862abf2ed704eef036bf0b6290c6cce4f618682e831d67bf2b5eb",
  },
} as const;

const CERTIFIED_PORTABLE_PACKAGES = {
  accelerateVersion: "1.14.0",
  safetensorsVersion: "0.8.0",
  aiohttpVersion: "3.14.1",
  sentencepieceVersion: "0.2.2",
  numpyVersion: "1.26.4",
} as const;

export const WINDOWS_ACCELERATOR_PACKS: Readonly<Record<"cuda" | "rocm", AcceleratorPack>> = {
  cuda: {
    id: "win-x64-py312-torch213-cu126-v1",
    backend: "cuda",
    platform: "win32",
    arch: "x64",
    strategy: "pinned-download",
    launchDevice: "cuda",
    torchVersion: "2.13.0+cu126",
    // The compressed torch wheel alone is 2.42 GiB. Keep enough room for the
    // wheel cache, its extracted DLLs, the copied base runtime, and rollback.
    minimumFreeBytes: 8 * GIB,
    installGroups: [[CUDA_TORCH]],
  },
  rocm: {
    id: "win-x64-py312-torch291-rocm721-v2",
    backend: "rocm",
    platform: "win32",
    arch: "x64",
    strategy: "pinned-download",
    launchDevice: "cuda",
    torchVersion: "2.9.1+rocm7.2.1",
    // AMD's official Windows install is roughly 2 GiB compressed before the
    // portable Python base and extracted libraries are counted.
    minimumFreeBytes: 8 * GIB,
    installGroups: [AMD_SDK_ARTIFACTS, [AMD_TORCH]],
  },
};

/** Apple ships Metal with macOS, so the native arm64 PyTorch wheel needs no device-side download. */
export const MACOS_MPS_PACK: Readonly<AcceleratorPack> = {
  id: "macos-arm64-py312-torch211-mps-v1",
  backend: "mps",
  platform: "darwin",
  arch: "arm64",
  strategy: "bundled",
  launchDevice: "mps",
  torchVersion: "2.11.0",
  minimumFreeBytes: 0,
  installGroups: [],
};

const CERTIFIED_ACCELERATOR_PACKS: readonly AcceleratorPack[] = [
  WINDOWS_ACCELERATOR_PACKS.cuda,
  WINDOWS_ACCELERATOR_PACKS.rocm,
  MACOS_MPS_PACK,
];

/** Select only combinations for which this release has a certified native provider. */
export function selectAcceleratorPack(
  hardware: AcceleratorHardware,
  preferredBackend: "auto" | EffectiveRuntimeBackend = "auto",
): AcceleratorPack | null {
  const platform = hardware.platform ?? process.platform;
  const arch = hardware.arch ?? process.arch;
  const vendor = (hardware.gpuVendor ?? "").trim().toLowerCase();
  const gpu = (hardware.gpuModel ?? "").toLowerCase();
  const cpu = (hardware.cpuModel ?? "").toLowerCase();
  if (preferredBackend === "cpu") return null;
  if (preferredBackend !== "auto") {
    const selected = CERTIFIED_ACCELERATOR_PACKS.find((pack) =>
      pack.backend === preferredBackend && pack.platform === platform && pack.arch === arch
    ) ?? null;
    if (!selected) return null;
    if (selected.backend === "rocm") {
      return vendor === "amd"
        && isWindows11Release(hardware.osRelease)
        && isCertifiedWindowsRadeon(gpu)
        ? selected
        : null;
    }
    if (selected.backend === "cuda") {
      return vendor === "nvidia" || gpu.includes("nvidia") || /\b(?:rtx|gtx)\b/.test(gpu)
        ? selected
        : null;
    }
    if (selected.backend === "mps") {
      return vendor === "apple" || /apple\s+m\d|apple\s+silicon/.test(gpu) || /apple\s+m\d|apple\s+silicon/.test(cpu)
        ? selected
        : null;
    }
    return null;
  }
  if (platform === "darwin" && arch === "arm64" && (
    vendor === "apple" || /apple\s+m\d|apple\s+silicon/.test(gpu) || /apple\s+m\d|apple\s+silicon/.test(cpu)
  )) {
    return MACOS_MPS_PACK;
  }
  if (platform !== "win32" || arch !== "x64") return null;
  if (vendor === "nvidia" || gpu.includes("nvidia") || /\b(?:rtx|gtx)\b/.test(gpu)) {
    return WINDOWS_ACCELERATOR_PACKS.cuda;
  }
  if (
    vendor === "amd" &&
    isWindows11Release(hardware.osRelease) &&
    isCertifiedWindowsRadeon(gpu)
  ) {
    return WINDOWS_ACCELERATOR_PACKS.rocm;
  }
  return null;
}

export async function prepareAcceleratorRuntime(
  options: PrepareAcceleratorRuntimeOptions,
): Promise<AcceleratorRuntimeResult> {
  let selectedBackend: AcceleratorBackend | null = null;
  let sequence = 0;
  let lastPercent = 0;
  let lastLegacyStatus = "";
  const emit = (
    phase: AcceleratorProgressPhase,
    percent: number,
    message: string,
    extra: Pick<AcceleratorProgressEvent, "download" | "issue"> & { recordLog?: boolean } = {},
  ): void => {
    lastPercent = Math.max(lastPercent, Math.min(100, Math.max(0, percent)));
    const event: AcceleratorProgressEvent = {
      sequence: ++sequence,
      at: new Date().toISOString(),
      phase,
      backend: selectedBackend,
      gpuVendor: normalizedHardwareValue(options.hardware.gpuVendor),
      gpuModel: normalizedHardwareValue(options.hardware.gpuModel),
      cpuAvailable: true,
      percent: lastPercent,
      message,
      recordLog: extra.recordLog ?? true,
      ...extra,
    };
    try {
      options.onProgress?.(event);
    } catch {
      // UI telemetry must never make runtime provisioning fail.
    }
    if (message !== lastLegacyStatus) {
      lastLegacyStatus = message;
      try {
        options.onStatus?.(message);
      } catch {
        // Retain the same failure isolation for the legacy callback.
      }
    }
  };

  const baseRoot = resolve(options.baseRuntimeRoot);
  const baseManifest = await readPortableRuntimeManifest(baseRoot);
  const basePython = runtimePythonExecutable(baseRoot, baseManifest.executable);
  await requireFile(basePython, "portable CPU Python executable");
  const baseEnvironment = runtimeEnvironmentAdditions(baseRoot);
  const preferred = options.preferredBackend ?? "auto";
  const pack = selectAcceleratorPack(options.hardware, preferred);
  if (pack === null) {
    if (preferred !== "cpu") {
      const issue = unsupportedHardwareIssue(options.hardware);
      emit("fallback", 100, `${issueMessage(issue)} CPU remains available.`, { issue });
    }
    return cpuResult(baseRoot, basePython, baseManifest, baseEnvironment);
  }
  selectedBackend = pack.backend;
  emit(
    "detecting",
    2,
    `${pack.backend.toUpperCase()} setup selected for ${options.hardware.gpuModel?.trim() || "the detected GPU"}. CPU remains available.`,
  );

  const runner = options.commandRunner ?? runRuntimeCommand;
  const gpuDeviceIndex = normalizeGpuDeviceIndex(options.hardware.gpuDeviceIndex);
  const expectedGpuModel = options.hardware.gpuModel?.trim() || undefined;
  if (pack.strategy === "bundled") {
    try {
      emit("checking-prerequisites", 20, `Checking the bundled ${pack.backend.toUpperCase()} provider.`);
      emit("physical-probe", 70, `Verifying ${pack.backend.toUpperCase()} with a real FP16 operation.`);
      const probe = await probeAcceleratorRuntime(basePython, pack.backend, baseEnvironment, runner, gpuDeviceIndex, expectedGpuModel);
      validatePackProbe(probe, pack, expectedGpuModel);
      emit("ready", 100, `${probe.deviceName} passed the physical FP16 probe and is ready for new stages.`);
      return gpuResult(baseRoot, basePython, pack, baseEnvironment, probe);
    } catch (error) {
      const issue = progressIssue(error, "physical-probe");
      emit("fallback", lastPercent, `${shortError(error)} CPU remains available.`, { issue });
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

  const acceleratorRoot = acceleratorRuntimeRootPath(
    options.userDataPath,
    pack,
    pack.platform,
  );
  const target = resolve(
    acceleratorRoot,
    acceleratorRuntimeTargetName(pack, pack.platform),
  );
  assertDirectChild(acceleratorRoot, target);
  const downloader = options.artifactDownloader
    ?? (
      options.artifactSwarm
        ? createArtifactSwarmDownloader({
            ...options.artifactSwarm,
            originDownloader: downloadPinnedArtifact,
          })
        : downloadPinnedArtifact
    );
  await mkdir(acceleratorRoot, { recursive: true });

  let cachedFailure: string | undefined;
  emit("checking-cache", 5, `Checking the cached ${pack.backend.toUpperCase()} runtime.`);
  if (existsSync(target)) {
    try {
      const cached = await readAcceleratorManifest(target, pack);
      const additions = environmentFromManifest(target, cached);
      const python = runtimePythonExecutable(target, baseManifest.executable);
      emit("physical-probe", 7, `Verifying cached ${pack.backend.toUpperCase()} with a real FP16 operation.`);
      const probe = await probeAcceleratorRuntime(
        python,
        pack.backend,
        additions,
        runner,
        gpuDeviceIndex,
        expectedGpuModel,
      );
      validatePackProbe(probe, pack, expectedGpuModel);
      emit("ready", 100, `${probe.deviceName} passed the physical FP16 probe.`);
      return gpuResult(target, python, pack, additions, probe);
    } catch (error) {
      cachedFailure = shortError(error);
      emit("checking-cache", 6, `The cached ${pack.backend.toUpperCase()} runtime is invalid and will be rebuilt.`);
    }
  }

  if (options.allowProvisioning !== true) {
    const issue: AcceleratorProgressIssue = cachedFailure
      ? {
          code: "cache-invalid",
          message: `Cached ${pack.backend.toUpperCase()} runtime is invalid.`,
          retryable: true,
          action: "Enable contribution or retry GPU setup to rebuild the cached runtime.",
        }
      : {
          code: "installation-required",
          message: `${pack.backend.toUpperCase()} runtime is not installed yet.`,
          retryable: true,
          action: "Enable contribution to install the verified GPU runtime automatically.",
        };
    emit(
      "fallback",
      100,
      cachedFailure
        ? `Cached ${pack.backend.toUpperCase()} runtime is invalid. CPU remains available.`
        : `${pack.backend.toUpperCase()} runtime is not installed yet. CPU remains available.`,
      { issue },
    );
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

  const staging = resolve(
    acceleratorRoot,
    `${ACCELERATOR_STAGING_PREFIX}${randomUUID().replaceAll("-", "").slice(0, 12)}`,
  );
  assertDirectChild(acceleratorRoot, staging);
  let failurePhase: AcceleratorProgressPhase = "checking-prerequisites";
  try {
    await removeAbandonedAcceleratorStaging(acceleratorRoot, pack.id);
    const legacyAcceleratorRoot = resolve(options.userDataPath, ACCELERATOR_DIRECTORY);
    if (
      legacyAcceleratorRoot !== acceleratorRoot
      && existsSync(legacyAcceleratorRoot)
    ) {
      await removeAbandonedAcceleratorStaging(legacyAcceleratorRoot, pack.id);
    }
    emit("checking-prerequisites", 8, `Checking free space for the ${pack.backend.toUpperCase()} runtime.`);
    const availableBytes = await requireFreeSpace(acceleratorRoot, pack.minimumFreeBytes);
    emit(
      "checking-prerequisites",
      10,
      `${formatGiB(availableBytes)} GiB is available; ${formatGiB(pack.minimumFreeBytes)} GiB is required.`,
    );

    const artifacts = pack.installGroups.flatMap((group) => [...group]);
    const aggregateTotal = artifacts.reduce((sum, artifact) => sum + artifact.sizeBytes, 0);
    // Keep the content-addressed package cache in its existing managed
    // location. Windows CUDA only shortens the installed runtime root, so
    // upgrades reuse the already verified multi-gigabyte wheel.
    const cacheRoot = resolve(options.userDataPath, ACCELERATOR_DIRECTORY, "package-cache");
    await mkdir(cacheRoot, { recursive: true });
    const downloadedPaths = new Map<string, string>();
    let completedBytes = 0;
    failurePhase = "downloading";
    for (const [artifactIndex, packageArtifact] of artifacts.entries()) {
      const artifactNumber = artifactIndex + 1;
      emit(
        "downloading",
        15 + 55 * (completedBytes / aggregateTotal),
        `Downloading ${packageArtifact.label} (${artifactNumber}/${artifacts.length}).`,
      );
      const localPath = await downloader(packageArtifact, cacheRoot, (transfer) => {
        const aggregateDownloaded = Math.min(
          aggregateTotal,
          completedBytes + Math.min(packageArtifact.sizeBytes, transfer.bytesDownloaded),
        );
        emit(
          "downloading",
          15 + 55 * (aggregateDownloaded / aggregateTotal),
          `Downloading ${packageArtifact.label} (${artifactNumber}/${artifacts.length}).`,
          {
            recordLog: false,
            download: {
              artifact: packageArtifact.label,
              artifactIndex: artifactNumber,
              artifactCount: artifacts.length,
              bytesDownloaded: transfer.bytesDownloaded,
              bytesTotal: transfer.bytesTotal,
              aggregateDownloaded,
              aggregateTotal,
              bytesPerSecond: transfer.bytesPerSecond,
              etaSeconds: transfer.bytesPerSecond === null
                ? transfer.etaSeconds
                : etaSeconds(aggregateTotal - aggregateDownloaded, transfer.bytesPerSecond),
              resumed: transfer.resumed,
            },
          },
        );
      });
      failurePhase = "verifying-package";
      const verifiedLocalPath = await verifyDownloadedArtifactPath(
        localPath,
        cacheRoot,
        packageArtifact,
      );
      downloadedPaths.set(packageArtifact.sha256, verifiedLocalPath);
      completedBytes += packageArtifact.sizeBytes;
      emit(
        "verifying-package",
        15 + 55 * (completedBytes / aggregateTotal),
        `${packageArtifact.label} passed exact size and SHA-256 verification.`,
      );
      failurePhase = "downloading";
    }

    failurePhase = "copying-base";
    emit("copying-base", 72, "Creating an isolated GPU runtime while CPU remains available.");
    if (existsSync(target)) await rm(target, { recursive: true, force: true });
    await rm(staging, { recursive: true, force: true });
    await cp(baseRoot, staging, { recursive: true, force: false, errorOnExist: true });
    const python = runtimePythonExecutable(staging, baseManifest.executable);
    const initialEnvironment = runtimeEnvironmentAdditions(staging);
    failurePhase = "installing";
    // The certified portable base deliberately omits wheel RECORD files. A
    // regular uninstall/reinstall therefore cannot remove its CPU-only torch
    // package and pip aborts with `uninstall-no-record-file`. The accelerator
    // runtime is an isolated copy, so overwrite that package in place and rely
    // on the mandatory physical probe below before activating the runtime.
    emit("installing", 74, "Replacing CPU-only PyTorch inside the isolated GPU runtime.");
    for (const [index, group] of pack.installGroups.entries()) {
      const installPercent = 76 + 16 * (index / pack.installGroups.length);
      emit(
        "installing",
        installPercent,
        `Installing ${pack.backend.toUpperCase()} runtime (${index + 1}/${pack.installGroups.length}).`,
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
          "--no-index",
          "--no-deps",
          "--ignore-installed",
          // ROCm's signed SDK bootstrap is distributed as a source archive.
          // Reuse the certified runtime's pinned setuptools instead of asking
          // pip to create an isolated build environment that would require an
          // unpinned network dependency.
          "--no-build-isolation",
          ...group.map((packageArtifact) => requiredMapValue(downloadedPaths, packageArtifact.sha256)),
        ],
        commandEnvironment(initialEnvironment),
        `install ${pack.backend.toUpperCase()} runtime group ${index + 1}`,
      );
    }
    const additions = runtimeEnvironmentAdditions(staging, true);
    failurePhase = "physical-probe";
    emit("physical-probe", 95, `Verifying ${pack.backend.toUpperCase()} on a real FP16 operation.`);
    const probe = await probeAcceleratorRuntime(
      python,
      pack.backend,
      additions,
      runner,
      gpuDeviceIndex,
      expectedGpuModel,
    );
    validatePackProbe(probe, pack, expectedGpuModel);
    failurePhase = "activating";
    emit("activating", 99, `Activating the verified runtime for ${probe.deviceName}.`);
    const manifest = acceleratorManifest(pack, baseManifest, staging, additions, probe);
    await writeFile(
      join(staging, ACCELERATOR_MANIFEST),
      `${JSON.stringify(manifest, null, 2)}\n`,
      "utf8",
    );
    await rename(staging, target);
    const targetAdditions = environmentFromManifest(target, manifest);
    emit("ready", 100, `${probe.deviceName} passed the physical FP16 probe and is ready for new stages.`);
    return gpuResult(
      target,
      runtimePythonExecutable(target, baseManifest.executable),
      pack,
      targetAdditions,
      probe,
    );
  } catch (error) {
    await rm(staging, { recursive: true, force: true }).catch(() => undefined);
    const issue = progressIssue(error, failurePhase);
    emit(
      issue.retryable ? "fallback" : "blocked",
      lastPercent,
      `${shortError(error)} CPU remains available.`,
      { issue },
    );
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

async function removeAbandonedAcceleratorStaging(
  acceleratorRoot: string,
  packId: string,
): Promise<void> {
  const legacyPrefix = `${packId}.staging-`;
  const entries = await readdir(acceleratorRoot, { withFileTypes: true });
  await Promise.all(entries
    .filter((entry) => entry.isDirectory() && (
      entry.name.startsWith(legacyPrefix)
      || entry.name.startsWith(ACCELERATOR_STAGING_PREFIX)
    ))
    .map(async (entry) => {
      const candidate = resolve(acceleratorRoot, entry.name);
      assertDirectChild(acceleratorRoot, candidate);
      await rm(candidate, { recursive: true, force: true });
    }));
}

export async function readPortableRuntimeManifest(
  runtimeRoot: string,
  expected?: { platform: string; arch: string },
): Promise<PortableRuntimeManifest> {
  const path = join(resolve(runtimeRoot), PORTABLE_MANIFEST);
  let raw: unknown;
  try {
    raw = JSON.parse(await readFile(path, "utf8"));
  } catch (error) {
    throw new Error(`portable runtime v4 manifest is missing or invalid: ${shortError(error)}`);
  }
  if (!isRecord(raw) || raw.schema !== PORTABLE_RUNTIME_SCHEMA) {
    throw new Error("portable runtime does not use the v4 manifest schema");
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
    "aiohttpVersion",
    "sentencepieceVersion",
    "numpyVersion",
  ] as const) {
    if (typeof raw[key] !== "string" || !raw[key].trim()) {
      throw new Error(`portable runtime manifest has invalid ${key}`);
    }
  }
  if (raw.pythonAbi !== "cp312" || raw.pythonVersion !== "3.12.13") {
    throw new Error("portable runtime must use the certified CPython 3.12.13 build");
  }
  const platform = raw.platform as string;
  const arch = raw.arch as string;
  const certified = CERTIFIED_PORTABLE_RUNTIME[
    `${platform}/${arch}` as keyof typeof CERTIFIED_PORTABLE_RUNTIME
  ];
  if (!certified) {
    throw new Error(`portable runtime has no certified base for ${platform}/${arch}`);
  }
  if (
    raw.executable !== certified.executable ||
    raw.torchVersion !== certified.torchVersion ||
    raw.transformersVersion !== certified.transformersVersion ||
    !stringArray(raw.bundledAccelerators) ||
    raw.bundledAccelerators.length !== certified.bundledAccelerators.length ||
    raw.bundledAccelerators.some(
      (value, index) => value !== certified.bundledAccelerators[index],
    ) ||
    Object.entries(CERTIFIED_PORTABLE_PACKAGES).some(([key, version]) => raw[key] !== version)
  ) {
    throw new Error(`portable runtime package versions do not match ${platform}/${arch}`);
  }
  validatePortablePythonArtifact(raw.pythonArtifact, platform, arch);
  validatePortableWheelLock(raw.wheelLock, platform, arch);
  if (raw.backend !== "cpu") throw new Error("portable base runtime must be CPU-only");
  if (expected && (raw.platform !== expected.platform || raw.arch !== expected.arch)) {
    throw new Error(
      `portable runtime targets ${String(raw.platform)}/${String(raw.arch)}; ` +
        `this device requires ${expected.platform}/${expected.arch}`,
    );
  }
  return raw as unknown as PortableRuntimeManifest;
}

export async function verifyPortableRuntimeInstallation(
  runtimeRoot: string,
  expected?: { platform: string; arch: string },
  runner: RuntimeCommandRunner = runRuntimeCommand,
): Promise<PortableRuntimeManifest> {
  const root = resolve(runtimeRoot);
  const manifest = await readPortableRuntimeManifest(root, expected);
  const python = runtimePythonExecutable(root, manifest.executable);
  await requireFile(python, "portable CPU Python executable");
  const result = await runner(
    python,
    ["-I", "-c", portableRuntimeProbeScript()],
    {
      env: {
        ...process.env,
        PYTHONNOUSERSITE: "1",
      },
    },
  );
  if (result.code !== 0) {
    throw new Error(
      `portable runtime health check failed (${result.code}): ${tail(result.stderr || result.stdout).trim()}`,
    );
  }
  const marker = result.stdout
    .split(/\r?\n/)
    .reverse()
    .find((line) => line.startsWith(PORTABLE_RUNTIME_PROBE_MARKER));
  if (!marker) throw new Error("portable runtime health check did not emit verified metadata");
  let value: unknown;
  try {
    value = JSON.parse(marker.slice(PORTABLE_RUNTIME_PROBE_MARKER.length));
  } catch {
    throw new Error("portable runtime health check metadata is not valid JSON");
  }
  if (
    !isRecord(value)
    || !sameResolvedPath(value.prefix, root)
    || !sameResolvedPath(value.basePrefix, root)
    || value.pythonVersion !== manifest.pythonVersion
    || value.torchVersion !== manifest.torchVersion
    || value.transformersVersion !== manifest.transformersVersion
    || value.accelerateVersion !== manifest.accelerateVersion
    || value.safetensorsVersion !== manifest.safetensorsVersion
    || value.aiohttpVersion !== manifest.aiohttpVersion
    || value.sentencepieceVersion !== manifest.sentencepieceVersion
    || value.numpyVersion !== manifest.numpyVersion
  ) {
    throw new Error("portable runtime health check does not match its certified manifest");
  }
  return manifest;
}

function portableRuntimeProbeScript(): string {
  return [
    "import encodings, importlib.metadata, importlib.util, json, sys",
    "packages = ('torch', 'transformers', 'accelerate', 'safetensors', 'aiohttp', 'sentencepiece', 'numpy')",
    "missing = [name for name in packages if importlib.util.find_spec(name) is None]",
    "assert not missing, f'missing runtime packages: {missing}'",
    "versions = {name: importlib.metadata.version(name) for name in packages}",
    "payload = {'prefix': sys.prefix, 'basePrefix': sys.base_prefix, 'pythonVersion': '.'.join(map(str, sys.version_info[:3])), 'torchVersion': versions['torch'], 'transformersVersion': versions['transformers'], 'accelerateVersion': versions['accelerate'], 'safetensorsVersion': versions['safetensors'], 'aiohttpVersion': versions['aiohttp'], 'sentencepieceVersion': versions['sentencepiece'], 'numpyVersion': versions['numpy']}",
    `print('${PORTABLE_RUNTIME_PROBE_MARKER}' + json.dumps(payload, separators=(',', ':')), flush=True)`,
  ].join("\n");
}

function validatePortableWheelLock(value: unknown, platform: string, arch: string): void {
  if (!isRecord(value)) throw new Error("portable runtime has no sealed wheel lock");
  const expected = CERTIFIED_PORTABLE_WHEEL_LOCKS[
    `${platform}/${arch}` as keyof typeof CERTIFIED_PORTABLE_WHEEL_LOCKS
  ];
  if (!expected) throw new Error(`portable runtime has no certified wheel lock for ${platform}/${arch}`);
  if (
    Object.keys(value).sort().join(",") !== "path,sha256"
    || value.path !== expected.path
    || value.sha256 !== expected.sha256
  ) {
    throw new Error(`portable runtime wheel lock does not match ${platform}/${arch}`);
  }
}

function validatePortablePythonArtifact(value: unknown, platform: string, arch: string): void {
  if (!isRecord(value)) throw new Error("portable runtime has no Python artifact provenance");
  const expected = CERTIFIED_PORTABLE_PYTHON[`${platform}/${arch}` as keyof typeof CERTIFIED_PORTABLE_PYTHON];
  if (!expected) throw new Error(`portable runtime has no certified Python build for ${platform}/${arch}`);
  const expectedUrl =
    `https://github.com/astral-sh/python-build-standalone/releases/download/20260510/${expected.filename}`;
  if (
    value.source !== "astral-sh/python-build-standalone" ||
    value.release !== "20260510" ||
    value.version !== "3.12.13" ||
    value.flavor !== "install_only_stripped" ||
    value.filename !== expected.filename ||
    value.url !== expectedUrl ||
    value.size !== expected.size ||
    value.sha256 !== expected.sha256
  ) {
    throw new Error(`portable runtime Python provenance does not match ${platform}/${arch}`);
  }
}

export async function probeAcceleratorRuntime(
  pythonExecutable: string,
  expectedBackend: AcceleratorBackend,
  additions: Pick<AcceleratorRuntimeResult, "pythonPathAdditions" | "pathAdditions">,
  runner: RuntimeCommandRunner = runRuntimeCommand,
  deviceIndex?: number | null | undefined,
  expectedDeviceName?: string | undefined,
): Promise<AcceleratorProbe> {
  const normalizedDeviceIndex = normalizeGpuDeviceIndex(deviceIndex);
  const requestedDeviceIndex = normalizedDeviceIndex ?? (expectedDeviceName?.trim() ? -1 : 0);
  const result = await runner(
    pythonExecutable,
    ["-c", acceleratorProbeScript(), expectedBackend, String(requestedDeviceIndex), expectedDeviceName?.trim() ?? ""],
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
  return validateProbe(
    value,
    expectedBackend,
    normalizedDeviceIndex ?? (expectedDeviceName?.trim() ? null : 0),
    expectedDeviceName,
  );
}

export function acceleratorRuntimeTargetPath(
  userDataPath: string,
  backend: AcceleratorBackend,
  platform = process.platform,
): string {
  const pack = CERTIFIED_ACCELERATOR_PACKS.find((candidate) => candidate.backend === backend);
  if (!pack) throw new Error(`No certified ${backend} runtime pack is registered`);
  return resolve(
    acceleratorRuntimeRootPath(userDataPath, pack, platform),
    acceleratorRuntimeTargetName(pack, platform),
  );
}

function acceleratorRuntimeRootPath(
  userDataPath: string,
  pack: AcceleratorPack,
  platform: string,
): string {
  if (platform === "win32" && pack.backend === "cuda") {
    // PyTorch ships nested headers and license trees that can exceed MAX_PATH
    // even with a short staging name. A compact per-user sibling remains
    // writable without elevation and leaves enough room for the full wheel.
    return resolve(dirname(resolve(userDataPath)), WINDOWS_CUDA_RUNTIME_DIRECTORY);
  }
  return resolve(userDataPath, ACCELERATOR_DIRECTORY);
}

function acceleratorRuntimeTargetName(
  pack: AcceleratorPack,
  platform: string,
): string {
  return platform === "win32" && pack.backend === "cuda"
    ? WINDOWS_CUDA_RUNTIME_TARGET
    : pack.id;
}

function acceleratorProbeScript(): string {
  return [
    "import json, math, sys, torch",
    "expected = sys.argv[1]",
    "requested_index = int(sys.argv[2])",
    "expected_device_name = sys.argv[3].strip()",
    "def identity_key(value):",
    "    value = value.lower().replace('(tm)', ' ').replace('(r)', ' ')",
    "    generic = {'advanced', 'micro', 'devices', 'nvidia', 'amd', 'ati', 'intel', 'apple', 'radeon', 'geforce', 'graphics', 'graphic', 'gpu', 'display', 'adapter', 'series', 'corporation', 'inc'}",
    "    return ''.join(token for token in ''.join(ch if ch.isalnum() else ' ' for ch in value).split() if token not in generic)",
    "hip = getattr(torch.version, 'hip', None)",
    "cuda = getattr(torch.version, 'cuda', None)",
    "if expected in ('cuda', 'rocm'):",
    "    actual = 'rocm' if hip else ('cuda' if cuda else 'cpu')",
    "    assert actual == expected, f'expected {expected}, got {actual}'",
    "    assert torch.cuda.is_available() and torch.cuda.device_count() > 0",
    "    device_names = [str(torch.cuda.get_device_properties(i).name) for i in range(torch.cuda.device_count())]",
    "    expected_key = identity_key(expected_device_name)",
    "    matches = [i for i, name in enumerate(device_names) if expected_key and identity_key(name) == expected_key]",
    "    if expected_key:",
    "        assert matches, f'selected GPU {expected_device_name!r} is absent from runtime devices: {device_names!r}'",
    "        device_index = requested_index if requested_index in matches else (matches[0] if len(matches) == 1 else -1)",
    "        assert device_index >= 0, f'selected GPU identity is ambiguous across runtime devices: {device_names!r}'",
    "    else:",
    "        device_index = requested_index",
    "    assert 0 <= device_index < torch.cuda.device_count(), f'CUDA/ROCm device {device_index} does not exist'",
    "    device = torch.device(f'cuda:{device_index}')",
    "    synchronize = lambda: torch.cuda.synchronize(device)",
    "    props = torch.cuda.get_device_properties(device_index)",
    "    device_name = str(props.name)",
    "    total_memory = int(props.total_memory)",
    "    allocated_memory = lambda: int(torch.cuda.memory_allocated(device))",
    "    reserved_memory = lambda: int(torch.cuda.memory_reserved(device))",
    "elif expected == 'mps':",
    "    actual = 'mps'",
    "    backend = getattr(torch.backends, 'mps', None)",
    "    assert backend is not None and backend.is_built() and backend.is_available(), 'MPS is not available'",
    "    device = torch.device('mps')",
    "    synchronize = lambda: torch.mps.synchronize()",
    "    get_name = getattr(backend, 'get_name', None)",
    "    device_name = str(get_name()) if callable(get_name) else 'Apple GPU'",
    "    total_memory = int(torch.mps.recommended_max_memory())",
    "    allocated_memory = lambda: int(torch.mps.current_allocated_memory())",
    "    reserved_memory = lambda: int(torch.mps.driver_allocated_memory())",
    "elif expected == 'xpu':",
    "    actual = 'xpu'",
    "    xpu = getattr(torch, 'xpu', None)",
    "    assert xpu is not None and xpu.is_available() and xpu.device_count() > 0, 'XPU is not available'",
    "    device_names = [str(xpu.get_device_properties(i).name) for i in range(xpu.device_count())]",
    "    expected_key = identity_key(expected_device_name)",
    "    matches = [i for i, name in enumerate(device_names) if expected_key and identity_key(name) == expected_key]",
    "    if expected_key:",
    "        assert matches, f'selected XPU {expected_device_name!r} is absent from runtime devices: {device_names!r}'",
    "        device_index = requested_index if requested_index in matches else (matches[0] if len(matches) == 1 else -1)",
    "        assert device_index >= 0, f'selected XPU identity is ambiguous across runtime devices: {device_names!r}'",
    "    else:",
    "        device_index = requested_index",
    "    assert 0 <= device_index < xpu.device_count(), f'XPU device {device_index} does not exist'",
    "    device = torch.device(f'xpu:{device_index}')",
    "    synchronize = lambda: xpu.synchronize(device)",
    "    props = xpu.get_device_properties(device_index)",
    "    device_name = str(props.name)",
    "    total_memory = int(props.total_memory)",
    "    allocated_memory = lambda: int(xpu.memory_allocated(device))",
    "    reserved_memory = lambda: int(xpu.memory_reserved(device))",
    "else:",
    "    raise AssertionError(f'unsupported accelerator backend: {expected}')",
    // Keep every operand exactly representable in FP16. arange(65536) in
    // float16 overflows above 65504, making an otherwise healthy GPU produce
    // an infinite/NaN checksum and forcing a false CPU fallback.
    "a = torch.full((256, 256), 0.5, device=device, dtype=torch.float16)",
    "b = torch.eye(256, device=device, dtype=torch.float16)",
    "c = a @ b",
    "synchronize()",
    "checksum = float(c.float().sum().item())",
    "assert c.device.type == device.type and c.dtype == torch.float16",
    "assert math.isfinite(checksum) and checksum != 0.0",
    "payload = {'backend': actual, 'device': str(device), 'device_name': device_name, 'torch_version': str(torch.__version__), 'python_version': '.'.join(map(str, sys.version_info[:3])), 'cuda_version': str(cuda) if cuda else None, 'hip_version': str(hip) if hip else None, 'precision': 'float16', 'operation': 'fp16_matmul', 'checksum': checksum, 'total_memory_bytes': total_memory, 'allocated_bytes': allocated_memory(), 'reserved_bytes': reserved_memory()}",
    `print('${PROBE_MARKER}' + json.dumps(payload, separators=(',', ':')), flush=True)`,
  ].join("\n");
}

function validateProbe(
  value: unknown,
  expectedBackend: AcceleratorBackend,
  deviceIndex: number | null,
  expectedDeviceName?: string | undefined,
): AcceleratorProbe {
  if (!isRecord(value) || value.backend !== expectedBackend) {
    throw new Error(`accelerator probe did not confirm ${expectedBackend}`);
  }
  const devicePrefix = expectedBackend === "cuda" || expectedBackend === "rocm"
    ? "cuda"
    : expectedBackend === "xpu"
      ? "xpu"
      : "mps";
  const expectedDevice = deviceIndex === null || devicePrefix === "mps"
    ? null
    : `${devicePrefix}:${deviceIndex}`;
  const validDevice = devicePrefix === "mps"
    ? value.device === "mps"
    : typeof value.device === "string"
      && new RegExp(`^${devicePrefix}:\\d+$`).test(value.device)
      && (expectedDevice === null || value.device === expectedDevice);
  if (!validDevice || value.precision !== "float16" || value.operation !== "fp16_matmul") {
    throw new Error(`accelerator probe did not execute the certified FP16 ${expectedBackend.toUpperCase()} path`);
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
  if (
    expectedDeviceName?.trim()
    && expectedBackend !== "mps"
    && !gpuModelsMatch(value.device_name as string, expectedDeviceName)
  ) {
    throw new Error(`accelerator probe ran on ${value.device_name as string}; expected ${expectedDeviceName}`);
  }
  if ((expectedBackend === "mps" || expectedBackend === "xpu") && (value.cuda_version !== null || value.hip_version !== null)) {
    throw new Error(`accelerator probe ${expectedBackend.toUpperCase()} build metadata is inconsistent`);
  }
  return {
    backend: expectedBackend,
    device: value.device as string,
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

function validatePackProbe(
  probe: AcceleratorProbe,
  pack: AcceleratorPack,
  expectedDeviceName?: string | undefined,
): void {
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
  if (pack.backend === "mps" && !/(apple|m\d)/i.test(probe.deviceName)) {
    throw new Error(`MPS runtime reported an unexpected device: ${probe.deviceName}`);
  }
  if (pack.backend === "xpu" && !/intel/i.test(probe.deviceName)) {
    throw new Error(`XPU runtime reported an unexpected device: ${probe.deviceName}`);
  }
  if (expectedDeviceName?.trim() && pack.backend !== "mps" && !gpuModelsMatch(probe.deviceName, expectedDeviceName)) {
    throw new Error(`runtime verified ${probe.deviceName}; selected adapter is ${expectedDeviceName}`);
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
  const sitePackages = [
    join(root, "Lib", "site-packages"),
    join(root, "lib", "python3.12", "site-packages"),
  ].find((path) => existsSync(path));
  const paths = [
    root,
    join(root, "Scripts"),
    join(root, "bin"),
    ...(sitePackages ? [join(sitePackages, "torch", "lib")] : []),
  ]
    .filter((path) => existsSync(path));
  if (discoverLibraries && sitePackages) {
    paths.push(...discoverDllDirectories(sitePackages));
  }
  return {
    pythonPathAdditions: sitePackages ? [sitePackages] : [],
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
    PATH: [...additions.pathAdditions, process.env.PATH].filter(Boolean).join(delimiter),
    PYTHONPATH: [...additions.pythonPathAdditions, process.env.PYTHONPATH]
      .filter(Boolean)
      .join(delimiter),
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
    packageUrls: pack.installGroups.flatMap((group) => group.map((packageArtifact) => packageArtifact.url)),
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
  const expectedPackageUrls = pack.installGroups.flatMap((group) => group.map((packageArtifact) => packageArtifact.url));
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
    launchDevice: "cpu",
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
    launchDevice: probe.device as "mps" | `cuda:${number}` | `xpu:${number}`,
    deviceName: probe.deviceName,
    precision: "float16",
    torchVersion: probe.torchVersion,
    probe,
  };
}

function normalizeGpuDeviceIndex(value: number | null | undefined): number | null {
  return Number.isInteger(value) && (value as number) >= 0 && (value as number) <= 1_024
    ? value as number
    : null;
}

/**
 * Download a pinned package into a SHA-addressed cache. Partial files remain in
 * place so a normal app restart can continue with an HTTP Range request.
 */
export async function downloadPinnedArtifact(
  packageArtifact: AcceleratorPackageArtifact,
  cacheRoot: string,
  onProgress: (progress: AcceleratorArtifactTransfer) => void,
): Promise<string> {
  validateArtifact(packageArtifact);
  const source = new URL(packageArtifact.url);
  if (source.protocol !== "https:") {
    throw setupError("network", false, "GPU runtime packages must use HTTPS.", "Reinstall or update mycellios; this package source is invalid.");
  }
  source.hash = "";
  const decodedName = decodeURIComponent(basename(source.pathname));
  if (!decodedName || decodedName === "." || decodedName === "..") {
    throw setupError("network", false, "GPU runtime package has an invalid file name.", "Reinstall or update mycellios.");
  }

  const root = resolve(cacheRoot);
  const artifactDirectory = resolve(root, packageArtifact.sha256);
  assertDirectChild(root, artifactDirectory);
  await mkdir(artifactDirectory, { recursive: true });
  const target = resolve(artifactDirectory, decodedName);
  assertDirectChild(artifactDirectory, target);
  const partial = `${target}.part`;

  if (existsSync(target)) {
    try {
      await verifyArtifactFile(target, packageArtifact);
      onProgress(completedTransfer(packageArtifact.sizeBytes, false));
      return target;
    } catch {
      await rm(target, { force: true });
    }
  }

  let offset = await fileSize(partial);
  if (offset > packageArtifact.sizeBytes) {
    await rm(partial, { force: true });
    offset = 0;
  }
  if (offset === packageArtifact.sizeBytes) {
    try {
      await verifyArtifactFile(partial, packageArtifact);
      await rename(partial, target);
      onProgress(completedTransfer(packageArtifact.sizeBytes, true));
      return target;
    } catch {
      await rm(partial, { force: true });
      offset = 0;
    }
  }

  const resumed = offset > 0;
  onProgress({
    bytesDownloaded: offset,
    bytesTotal: packageArtifact.sizeBytes,
    bytesPerSecond: null,
    etaSeconds: null,
    resumed,
  });

  let response: Response;
  try {
    response = await fetch(source, {
      cache: "no-store",
      redirect: "follow",
      ...(offset > 0 ? { headers: { range: `bytes=${offset}-` } } : {}),
    });
  } catch (error) {
    throw setupError(
      "network",
      true,
      `Could not download ${packageArtifact.label}: ${shortError(error)}`,
      "Check the internet connection; the verified download can resume on retry.",
    );
  }
  if (response.url && new URL(response.url).protocol !== "https:") {
    throw setupError("network", false, "GPU runtime download redirected away from HTTPS.", "Retry after updating mycellios.");
  }
  if (!(response.status === 200 || response.status === 206) || response.body === null) {
    throw setupError(
      "network",
      true,
      `Could not download ${packageArtifact.label}: HTTP ${response.status}.`,
      "Check the internet connection; the verified download can resume on retry.",
    );
  }
  if (response.status === 206) {
    const rangeStart = contentRangeStart(response.headers.get("content-range"));
    if (rangeStart !== offset) {
      throw setupError(
        "network",
        true,
        `The package server resumed ${packageArtifact.label} at an unexpected byte.`,
        "Retry the download; mycellios will validate every byte before installation.",
      );
    }
  } else if (offset > 0) {
    // Some CDNs ignore Range. Restart this artifact safely instead of appending
    // a complete response to a partial wheel.
    offset = 0;
  }

  const writer = await open(partial, offset > 0 ? "a" : "w");
  const reader = response.body.getReader();
  let downloaded = offset;
  const startedAt = Date.now();
  const startedBytes = offset;
  let sampledAt = startedAt;
  let sampledBytes = offset;
  let smoothedBytesPerSecond: number | null = null;
  let lastProgressAt = startedAt;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value.byteLength === 0) continue;
      if (downloaded + value.byteLength > packageArtifact.sizeBytes) {
        throw setupError(
          "integrity",
          true,
          `${packageArtifact.label} exceeded its pinned size.`,
          "Retry the download. If this repeats, update mycellios before continuing.",
        );
      }
      await writer.write(value);
      downloaded += value.byteLength;
      const now = Date.now();
      const sampleSeconds = (now - sampledAt) / 1_000;
      if (sampleSeconds >= 0.25) {
        const instant = (downloaded - sampledBytes) / sampleSeconds;
        if (Number.isFinite(instant) && instant > 0) {
          smoothedBytesPerSecond = smoothedBytesPerSecond === null
            ? instant
            : smoothedBytesPerSecond * 0.72 + instant * 0.28;
        }
        sampledAt = now;
        sampledBytes = downloaded;
      } else if (smoothedBytesPerSecond === null && now > startedAt && downloaded > startedBytes) {
        smoothedBytesPerSecond = (downloaded - startedBytes) / ((now - startedAt) / 1_000);
      }
      // Body streams can yield thousands of tiny chunks. Four UI updates per
      // second are enough for a fluid progress bar without flooding IPC or
      // React. The exact final state is emitted after integrity verification.
      if (downloaded < packageArtifact.sizeBytes && now - lastProgressAt >= 250) {
        lastProgressAt = now;
        onProgress({
          bytesDownloaded: downloaded,
          bytesTotal: packageArtifact.sizeBytes,
          bytesPerSecond: finitePositive(smoothedBytesPerSecond),
          etaSeconds: etaSeconds(packageArtifact.sizeBytes - downloaded, smoothedBytesPerSecond),
          resumed,
        });
      }
    }
    await writer.sync();
  } finally {
    await writer.close();
  }

  if (downloaded !== packageArtifact.sizeBytes) {
    throw setupError(
      "network",
      true,
      `${packageArtifact.label} stopped at ${downloaded} of ${packageArtifact.sizeBytes} bytes.`,
      "Check the internet connection; the verified download can resume on retry.",
    );
  }
  try {
    await verifyArtifactFile(partial, packageArtifact);
  } catch (error) {
    await rm(partial, { force: true });
    throw error;
  }
  await rename(partial, target);
  onProgress(completedTransfer(packageArtifact.sizeBytes, resumed));
  return target;
}

class AcceleratorSetupError extends Error {
  constructor(
    message: string,
    readonly issue: AcceleratorProgressIssue,
  ) {
    super(message);
    this.name = "AcceleratorSetupError";
  }
}

function setupError(
  code: AcceleratorProgressIssueCode,
  retryable: boolean,
  message: string,
  action: string,
  details: Pick<AcceleratorProgressIssue, "requiredBytes" | "availableBytes"> = {},
): AcceleratorSetupError {
  return new AcceleratorSetupError(message, { code, message, retryable, action, ...details });
}

async function verifyArtifactFile(path: string, packageArtifact: AcceleratorPackageArtifact): Promise<void> {
  const size = await fileSize(path);
  if (size !== packageArtifact.sizeBytes) {
    throw setupError(
      "integrity",
      true,
      `${packageArtifact.label} has ${size} bytes; expected ${packageArtifact.sizeBytes}.`,
      "Retry the download. mycellios will resume or replace the invalid package safely.",
    );
  }
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
  const actual = hash.digest("hex");
  if (actual !== packageArtifact.sha256) {
    throw setupError(
      "integrity",
      true,
      `${packageArtifact.label} failed SHA-256 verification.`,
      "Retry the download. If this repeats, update mycellios before continuing.",
    );
  }
}

async function verifyDownloadedArtifactPath(
  localPath: string,
  cacheRoot: string,
  packageArtifact: AcceleratorPackageArtifact,
): Promise<string> {
  const root = resolve(cacheRoot);
  const artifactDirectory = resolve(root, packageArtifact.sha256);
  assertDirectChild(root, artifactDirectory);
  const candidate = resolve(localPath);
  if (dirname(candidate) !== artifactDirectory || basename(candidate) === "") {
    throw setupError(
      "integrity",
      true,
      `${packageArtifact.label} was returned outside its managed cache directory.`,
      "Retry the download. If this repeats, update mycellios before continuing.",
    );
  }
  await verifyArtifactFile(candidate, packageArtifact);
  return candidate;
}

async function fileSize(path: string): Promise<number> {
  try {
    const info = await stat(path);
    return info.isFile() ? info.size : 0;
  } catch {
    return 0;
  }
}

function contentRangeStart(value: string | null): number | null {
  const match = value?.match(/^bytes (\d+)-\d+\/\d+$/i);
  if (!match) return null;
  const parsed = Number(match[1]);
  return Number.isSafeInteger(parsed) && parsed >= 0 ? parsed : null;
}

function completedTransfer(bytes: number, resumed: boolean): AcceleratorArtifactTransfer {
  return {
    bytesDownloaded: bytes,
    bytesTotal: bytes,
    bytesPerSecond: null,
    etaSeconds: 0,
    resumed,
  };
}

function finitePositive(value: number | null): number | null {
  return value !== null && Number.isFinite(value) && value > 0 ? value : null;
}

function etaSeconds(remainingBytes: number, bytesPerSecond: number | null): number | null {
  const speed = finitePositive(bytesPerSecond);
  if (speed === null) return null;
  return Math.max(0, Math.ceil(remainingBytes / speed));
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

async function requireFreeSpace(path: string, requiredBytes: number): Promise<number> {
  const info = await statfs(path);
  const available = Number(info.bavail) * Number(info.bsize);
  if (!Number.isFinite(available) || available < requiredBytes) {
    const safeAvailable = Number.isFinite(available) ? Math.max(0, available) : 0;
    throw setupError(
      "disk-space",
      true,
      `accelerator runtime needs ${Math.ceil(requiredBytes / GIB)} GiB free; ` +
        `${Math.max(0, Math.floor(safeAvailable / GIB))} GiB is available`,
      "Free disk space, then retry GPU setup. CPU remains available.",
      { requiredBytes, availableBytes: safeAvailable },
    );
  }
  return available;
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

function sameResolvedPath(value: unknown, expected: string): boolean {
  if (typeof value !== "string" || !value.trim()) return false;
  const actual = resolve(value);
  const target = resolve(expected);
  return process.platform === "win32"
    ? actual.toLowerCase() === target.toLowerCase()
    : actual === target;
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values.map((value) => resolve(value)))];
}

function artifact(
  label: string,
  url: string,
  sha256: string,
  sizeBytes: number,
): AcceleratorPackageArtifact {
  return { label, url: `${url}#sha256=${sha256}`, sha256, sizeBytes };
}

function validateArtifact(packageArtifact: AcceleratorPackageArtifact): void {
  if (!packageArtifact.label.trim()) throw new Error("accelerator package label is empty");
  if (!/^[0-9a-f]{64}$/.test(packageArtifact.sha256)) {
    throw new Error(`accelerator package ${packageArtifact.label} has an invalid SHA-256 digest`);
  }
  if (!Number.isSafeInteger(packageArtifact.sizeBytes) || packageArtifact.sizeBytes <= 0) {
    throw new Error(`accelerator package ${packageArtifact.label} has an invalid pinned size`);
  }
  const source = new URL(packageArtifact.url);
  if (source.hash !== `#sha256=${packageArtifact.sha256}`) {
    throw new Error(`accelerator package ${packageArtifact.label} has inconsistent hash metadata`);
  }
}

function normalizedHardwareValue(value: string | undefined): string | null {
  const normalized = value?.trim();
  return normalized ? normalized : null;
}

function unsupportedHardwareIssue(hardware: AcceleratorHardware): AcceleratorProgressIssue {
  const platform = hardware.platform ?? process.platform;
  const arch = hardware.arch ?? process.arch;
  const vendor = normalizedHardwareValue(hardware.gpuVendor);
  const model = normalizedHardwareValue(hardware.gpuModel);
  if (
    (vendor === null || vendor.toLowerCase() === "unknown") &&
    (model === null || /(?:unidentified|unknown)/i.test(model))
  ) {
    return {
      code: "no-gpu",
      message: "No supported GPU or display driver was detected.",
      retryable: false,
      action: "Install the display driver or continue contributing with CPU.",
    };
  }
  if (platform !== "win32" && platform !== "darwin" && platform !== "linux") {
    return {
      code: "unsupported-platform",
      message: `No native accelerator provider is available for ${platform}/${arch}.`,
      retryable: false,
      action: "CPU contribution remains available. WebGPU tasks may still run from a compatible browser.",
    };
  }
  if (
    platform === "win32" &&
    vendor?.toLowerCase() === "amd" &&
    (isCertifiedWindowsRadeon((model ?? "").toLowerCase()) || /(?:890m|8050s|8060s)/i.test(model ?? "")) &&
    !isWindows11Release(hardware.osRelease)
  ) {
    return {
      code: "unsupported-platform",
      message: `${model ?? "This AMD GPU"} requires Windows 11 for the certified ROCm 7.2.1 provider.`,
      retryable: false,
      action: "Continue with CPU or the browser WebGPU worker, or upgrade this device to Windows 11.",
    };
  }
  return {
    code: "unsupported-gpu",
    message: `${model ?? vendor ?? "The detected GPU"} has no certified native provider for ${platform}/${arch} in this release.`,
    retryable: false,
    action: "CPU contribution remains available. A compatible WebGPU worker can still use this device for certified browser tasks.",
  };
}

function isWindows11Release(value: string | undefined): boolean {
  const parts = value?.trim().split(".").map((part) => Number(part)) ?? [];
  const major = parts[0];
  const build = parts[2];
  return Number.isInteger(major) && Number.isInteger(build) && (
    (major as number) > 10 || ((major as number) === 10 && (build as number) >= 22_000)
  );
}

function isCertifiedWindowsRadeon(model: string): boolean {
  return /\bradeon(?:\(tm\))?\s+(?:rx\s+(?:9070(?:\s+xt)?|9060\s+xt|7900\s+xtx|7700)|ai\s+pro\s+r9700|pro\s+w7900(?:\s+dual\s+slot)?|(?:890m|8050s|8060s)(?:\s+graphics)?)\b/i
    .test(model);
}

function issueMessage(issue: AcceleratorProgressIssue): string {
  return issue.message;
}

function progressIssue(
  error: unknown,
  phase: AcceleratorProgressPhase,
): AcceleratorProgressIssue {
  if (error instanceof AcceleratorSetupError) return error.issue;
  const message = shortError(error);
  if (/init_fs_encoding|no module named ['"]encodings['"]/i.test(message)) {
    return {
      code: "cache-invalid",
      message: "The bundled Python runtime is incomplete after an interrupted application update.",
      retryable: true,
      action: "Restart mycellios so it can rebuild the runtime automatically. The node will not advertise compute until verification passes.",
    };
  }
  if (phase === "physical-probe") {
    return {
      code: "physical-probe",
      message,
      retryable: true,
      action: "mycellios will revalidate the GPU automatically. Compatible operating-system driver updates can then recover the node without reinstalling mycellios.",
    };
  }
  if (phase === "downloading" || phase === "verifying-package") {
    return {
      code: phase === "verifying-package" ? "integrity" : "network",
      message,
      retryable: true,
      action: "Check the internet connection and retry; verified partial downloads are reusable.",
    };
  }
  return {
    code: "install",
    message,
    retryable: true,
    action: "mycellios will rebuild the isolated GPU runtime and retry automatically with expanding backoff.",
  };
}

function requiredMapValue(map: ReadonlyMap<string, string>, key: string): string {
  const value = map.get(key);
  if (!value) throw new Error(`verified accelerator package ${key} is missing from the local cache`);
  return value;
}

function formatGiB(bytes: number): string {
  return (bytes / GIB).toFixed(bytes >= 10 * GIB ? 0 : 1);
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
