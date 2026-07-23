import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ACCELERATOR_RUNTIME_SCHEMA,
  MACOS_MPS_PACK,
  PORTABLE_RUNTIME_PROBE_MARKER,
  PORTABLE_RUNTIME_SCHEMA,
  WINDOWS_ACCELERATOR_PACKS,
  acceleratorRuntimeTargetPath,
  downloadPinnedArtifact,
  prepareAcceleratorRuntime,
  probeAcceleratorRuntime,
  readPortableRuntimeManifest,
  selectAcceleratorPack,
  verifyPortableRuntimeInstallation,
  type RuntimeCommandRunner,
} from "../src/desktop/accelerator-runtime.js";

const temporaryDirectories: string[] = [];
const artifactRestorers: Array<() => void> = [];

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  for (const restore of artifactRestorers.splice(0)) restore();
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("desktop accelerator runtime", () => {
  it("pins every vendor package URL to an exact SHA-256 digest", () => {
    for (const pack of Object.values(WINDOWS_ACCELERATOR_PACKS)) {
      for (const artifact of pack.installGroups.flat()) {
        expect(artifact.url).toBe(`${artifact.url.split("#")[0]}#sha256=${artifact.sha256}`);
        expect(artifact.sha256).toMatch(/^[0-9a-f]{64}$/);
        expect(artifact.sizeBytes).toBeGreaterThan(0);
        expect(artifact.label).not.toBe("");
      }
    }
    expect(WINDOWS_ACCELERATOR_PACKS.cuda.installGroups.flat().reduce((sum, artifact) => sum + artifact.sizeBytes, 0)).toBe(2_594_590_371);
    expect(WINDOWS_ACCELERATOR_PACKS.rocm.installGroups.flat().reduce((sum, artifact) => sum + artifact.sizeBytes, 0)).toBe(2_188_680_000);
    expect(WINDOWS_ACCELERATOR_PACKS.rocm.installGroups.flat().map((artifact) => artifact.url.split("#")[0])).toEqual([
      "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
      "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
      "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
      "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz",
      "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
    ]);
  });

  it("selects certified native providers by platform, architecture and hardware", () => {
    expect(selectAcceleratorPack({
      platform: "win32",
      arch: "x64",
      gpuVendor: "nvidia",
      gpuModel: "NVIDIA GeForce RTX 2060",
    })?.backend).toBe("cuda");
    expect(selectAcceleratorPack({
      platform: "win32",
      arch: "x64",
      osRelease: "10.0.26100",
      gpuVendor: "amd",
      gpuModel: "AMD Radeon(TM) 890M Graphics",
      cpuModel: "AMD Ryzen AI 9 HX 370 w/ Radeon 890M",
    })?.backend).toBe("rocm");
    expect(selectAcceleratorPack({
      platform: "win32",
      arch: "x64",
      osRelease: "10.0.26100",
      gpuVendor: "amd",
      gpuModel: "AMD Radeon RX 9070 XT",
    })?.backend).toBe("rocm");
    expect(selectAcceleratorPack({
      platform: "win32",
      arch: "x64",
      osRelease: "10.0.19045",
      gpuVendor: "amd",
      gpuModel: "AMD Radeon RX 9070 XT",
    })).toBeNull();
    expect(selectAcceleratorPack({
      platform: "win32",
      arch: "x64",
      gpuVendor: "amd",
      gpuModel: "AMD Radeon RX 580",
    })).toBeNull();
    expect(selectAcceleratorPack({
      platform: "win32",
      arch: "x64",
      osRelease: "10.0.26100",
      gpuVendor: "amd",
      gpuModel: "AMD Radeon RX 580",
      cpuModel: "AMD Ryzen AI 9 HX 370 w/ Radeon 890M",
    })).toBeNull();
    expect(selectAcceleratorPack({
      platform: "win32",
      arch: "x64",
      osRelease: "10.0.26100",
      gpuVendor: "amd",
      gpuModel: "AMD Radeon RX 580",
      cpuModel: "AMD Ryzen AI 9 HX 370 w/ Radeon 890M",
    }, "rocm")).toBeNull();
    expect(selectAcceleratorPack({
      platform: "linux",
      arch: "x64",
      gpuVendor: "nvidia",
      gpuModel: "NVIDIA GeForce RTX 2060",
    })).toBeNull();
    expect(selectAcceleratorPack({
      platform: "darwin",
      arch: "arm64",
      gpuVendor: "apple",
      gpuModel: "Apple M4 Pro",
    })).toBe(MACOS_MPS_PACK);
    expect(selectAcceleratorPack({
      platform: "darwin",
      arch: "x64",
      gpuVendor: "intel",
      gpuModel: "Intel Iris Plus",
    })).toBeNull();
  });

  it("accepts only an allocated FP16 operation from the requested backend", async () => {
    const runner: RuntimeCommandRunner = vi.fn(async () => ({
      code: 0,
      stderr: "",
      stdout: "warning\nMYCELLIOS_RUNTIME_PROBE=" + JSON.stringify({
        backend: "cuda",
        device: "cuda:0",
        device_name: "NVIDIA GeForce RTX 2060",
        torch_version: "2.13.0+cu126",
        python_version: "3.12.13",
        cuda_version: "12.6",
        hip_version: null,
        precision: "float16",
        operation: "fp16_matmul",
        checksum: 32767.5,
        total_memory_bytes: 6_442_450_944,
        allocated_bytes: 786_432,
        reserved_bytes: 2_097_152,
      }) + "\n",
    }));
    const probe = await probeAcceleratorRuntime(
      "python.exe",
      "cuda",
      { pythonPathAdditions: [], pathAdditions: [] },
      runner,
    );
    expect(probe).toMatchObject({
      backend: "cuda",
      deviceName: "NVIDIA GeForce RTX 2060",
      precision: "float16",
      allocatedBytes: 786_432,
    });
    const probeScript = vi.mocked(runner).mock.calls[0]?.[1][1];
    expect(probeScript).toContain("torch.full((256, 256), 0.5");
    expect(probeScript).not.toContain("torch.arange(65536");
  });

  it("probes and launches the exact selected GPU on multi-adapter systems", async () => {
    const runner: RuntimeCommandRunner = vi.fn(async (_command, args) => ({
      code: 0,
      stderr: "",
      stdout: "MYCELLIOS_RUNTIME_PROBE=" + JSON.stringify({
        backend: "cuda",
        device: `cuda:${args[3]}`,
        device_name: "NVIDIA GeForce RTX 4090",
        torch_version: "2.13.0+cu126",
        python_version: "3.12.13",
        cuda_version: "12.6",
        hip_version: null,
        precision: "float16",
        operation: "fp16_matmul",
        checksum: 32768,
        total_memory_bytes: 24_564 * 1_024 ** 2,
        allocated_bytes: 786_432,
        reserved_bytes: 2_097_152,
      }) + "\n",
    }));

    const probe = await probeAcceleratorRuntime(
      "python.exe",
      "cuda",
      { pythonPathAdditions: [], pathAdditions: [] },
      runner,
      2,
      "NVIDIA GeForce RTX 4090",
    );

    expect(probe.device).toBe("cuda:2");
    expect(vi.mocked(runner).mock.calls[0]?.[1].slice(-3)).toEqual([
      "cuda",
      "2",
      "NVIDIA GeForce RTX 4090",
    ]);
  });

  it("rejects a successful probe from a different selected GPU", async () => {
    const runner: RuntimeCommandRunner = vi.fn(async () => ({
      code: 0,
      stderr: "",
      stdout: "MYCELLIOS_RUNTIME_PROBE=" + JSON.stringify({
        backend: "rocm",
        device: "cuda:1",
        device_name: "AMD Radeon 890M Graphics",
        torch_version: "2.9.1+rocm7.2.1",
        python_version: "3.12.13",
        cuda_version: null,
        hip_version: "7.2.1",
        precision: "float16",
        operation: "fp16_matmul",
        checksum: 32768,
        total_memory_bytes: 8_589_934_592,
        allocated_bytes: 786_432,
        reserved_bytes: 2_097_152,
      }) + "\n",
    }));

    await expect(probeAcceleratorRuntime(
      "python.exe",
      "rocm",
      { pythonPathAdditions: [], pathAdditions: [] },
      runner,
      undefined,
      "AMD Radeon RX 9070 XT",
    )).rejects.toThrow(/expected AMD Radeon RX 9070 XT/);
    expect(vi.mocked(runner).mock.calls[0]?.[1].slice(-3)).toEqual([
      "rocm",
      "-1",
      "AMD Radeon RX 9070 XT",
    ]);
  });

  it("accepts a physically allocated MPS FP16 operation", async () => {
    const runner: RuntimeCommandRunner = vi.fn(async () => mpsProbeResult());
    const probe = await probeAcceleratorRuntime(
      "bin/python3",
      "mps",
      { pythonPathAdditions: [], pathAdditions: [] },
      runner,
    );
    expect(probe).toMatchObject({
      backend: "mps",
      device: "mps",
      deviceName: "Apple M4 Pro",
      allocatedBytes: 786_432,
    });
    const probeScript = vi.mocked(runner).mock.calls[0]?.[1][1];
    expect(probeScript).toContain("backend.is_available()");
    expect(probeScript).toContain("torch.device('mps')");
  });

  it("uses bundled Apple MPS without downloading or mutating the CPU runtime", async () => {
    const root = temporaryRoot();
    const base = createMpsBaseRuntime(root);
    const runner: RuntimeCommandRunner = vi.fn(async () => mpsProbeResult());
    const downloader = vi.fn();
    const result = await prepareAcceleratorRuntime({
      baseRuntimeRoot: base,
      userDataPath: join(root, "user-data"),
      hardware: {
        platform: "darwin",
        arch: "arm64",
        gpuVendor: "apple",
        gpuModel: "Apple M4 Pro",
      },
      allowProvisioning: false,
      commandRunner: runner,
      artifactDownloader: downloader,
    });
    expect(result).toMatchObject({
      status: "gpu-ready",
      effectiveBackend: "mps",
      launchDevice: "mps",
      runtimeRoot: base,
    });
    expect(downloader).not.toHaveBeenCalled();
    expect(runner).toHaveBeenCalledOnce();
  });

  it("rejects a portable runtime built for another platform", async () => {
    const root = temporaryRoot();
    const base = createBaseRuntime(root);
    await expect(readPortableRuntimeManifest(base, { platform: "linux", arch: "x64" }))
      .rejects.toThrow("this device requires linux/x64");
  });

  it("rejects portable Python provenance that is not the certified artifact", async () => {
    const root = temporaryRoot();
    const base = createBaseRuntime(root);
    const manifestPath = join(base, "runtime-manifest.json");
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as {
      pythonArtifact: { sha256: string };
    };
    manifest.pythonArtifact.sha256 = "0".repeat(64);
    writeFileSync(manifestPath, JSON.stringify(manifest), "utf8");

    await expect(readPortableRuntimeManifest(base, { platform: "win32", arch: "x64" }))
      .rejects.toThrow("Python provenance does not match win32/x64");
  });

  it("rejects a stale v3 base runtime with different dependency versions", async () => {
    const root = temporaryRoot();
    const base = createBaseRuntime(root);
    const manifestPath = join(base, "runtime-manifest.json");
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as {
      transformersVersion: string;
    };
    manifest.transformersVersion = "5.13.0";
    writeFileSync(manifestPath, JSON.stringify(manifest), "utf8");

    await expect(readPortableRuntimeManifest(base, { platform: "win32", arch: "x64" }))
      .rejects.toThrow("package versions do not match win32/x64");
  });

  it("executes the portable Python before accepting an extracted runtime", async () => {
    const root = temporaryRoot();
    const base = createBaseRuntime(root);
    const runner: RuntimeCommandRunner = vi.fn(async () => ({
      code: 0,
      stderr: "",
      stdout: `${PORTABLE_RUNTIME_PROBE_MARKER}${JSON.stringify({
        prefix: base,
        basePrefix: base,
        pythonVersion: "3.12.13",
        torchVersion: "2.13.0+cpu",
        transformersVersion: "5.14.1",
        accelerateVersion: "1.14.0",
        safetensorsVersion: "0.8.0",
        aiohttpVersion: "3.14.1",
        sentencepieceVersion: "0.2.2",
        numpyVersion: "1.26.4",
      })}\n`,
    }));

    await expect(verifyPortableRuntimeInstallation(
      base,
      { platform: "win32", arch: "x64" },
      runner,
    )).resolves.toMatchObject({ pythonVersion: "3.12.13", backend: "cpu" });
    expect(runner).toHaveBeenCalledOnce();
    expect(vi.mocked(runner).mock.calls[0]?.[1].slice(0, 2)).toEqual(["-I", "-c"]);
    expect(vi.mocked(runner).mock.calls[0]?.[1][2]).toContain("import encodings");
  });

  it("rejects the exact incomplete extraction that cannot import encodings", async () => {
    const root = temporaryRoot();
    const base = createBaseRuntime(root);
    const runner: RuntimeCommandRunner = vi.fn(async () => ({
      code: 1,
      stdout: "",
      stderr: "Fatal Python error: init_fs_encoding\nModuleNotFoundError: No module named 'encodings'",
    }));

    await expect(verifyPortableRuntimeInstallation(
      base,
      { platform: "win32", arch: "x64" },
      runner,
    )).rejects.toThrow("No module named 'encodings'");
  });

  it("accepts the certified CPU runtime for macOS Intel", async () => {
    const root = temporaryRoot();
    const base = createMacIntelBaseRuntime(root);

    await expect(readPortableRuntimeManifest(base, { platform: "darwin", arch: "x64" }))
      .resolves.toMatchObject({ platform: "darwin", arch: "x64", torchVersion: "2.2.2" });
  });

  it("falls back explicitly without downloading when provisioning is disabled", async () => {
    const root = temporaryRoot();
    const base = createBaseRuntime(root);
    const runner = vi.fn<RuntimeCommandRunner>();
    const result = await prepareAcceleratorRuntime({
      baseRuntimeRoot: base,
      userDataPath: join(root, "user-data"),
      hardware: {
        platform: "win32",
        arch: "x64",
        gpuVendor: "nvidia",
        gpuModel: "NVIDIA GeForce RTX 2060",
      },
      allowProvisioning: false,
      commandRunner: runner,
    });
    expect(result).toMatchObject({
      status: "gpu-fallback",
      requestedBackend: "cuda",
      effectiveBackend: "cpu",
      deviceType: "cpu",
      precision: "float32",
    });
    expect(result.fallbackReason).toContain("not installed");
    expect(runner).not.toHaveBeenCalled();
  });

  it("keeps the explicit CPU path immediate and free of accelerator side effects", async () => {
    const root = temporaryRoot();
    const base = createBaseRuntime(root);
    const runner = vi.fn<RuntimeCommandRunner>();
    const artifactDownloader = vi.fn();
    const onProgress = vi.fn();
    const result = await prepareAcceleratorRuntime({
      baseRuntimeRoot: base,
      userDataPath: join(root, "user-data"),
      hardware: {
        platform: "win32",
        arch: "x64",
        gpuVendor: "nvidia",
        gpuModel: "NVIDIA GeForce RTX 2060",
      },
      preferredBackend: "cpu",
      allowProvisioning: true,
      commandRunner: runner,
      artifactDownloader,
      onProgress,
    });
    expect(result).toMatchObject({ status: "cpu-ready", effectiveBackend: "cpu" });
    expect(runner).not.toHaveBeenCalled();
    expect(artifactDownloader).not.toHaveBeenCalled();
    expect(onProgress).not.toHaveBeenCalled();
  });

  it("reuses a cached runtime only after probing it again", async () => {
    const root = temporaryRoot();
    const base = createBaseRuntime(root);
    const userData = join(root, "user-data");
    const target = acceleratorRuntimeTargetPath(userData, "rocm");
    mkdirSync(join(target, "Lib", "site-packages", "torch", "lib"), { recursive: true });
    writeFileSync(join(target, "python.exe"), "test", "utf8");
    writeFileSync(join(target, "accelerator-runtime.json"), JSON.stringify({
      schema: ACCELERATOR_RUNTIME_SCHEMA,
      packId: "win-x64-py312-torch291-rocm721-v2",
      backend: "rocm",
      baseRuntimeSchema: PORTABLE_RUNTIME_SCHEMA,
      pythonVersion: "3.12.13",
      torchVersion: "2.9.1+rocm7.2.1",
      packageUrls: WINDOWS_ACCELERATOR_PACKS.rocm.installGroups.flat().map((artifact) => artifact.url),
      probe: {},
      environment: {
        pythonPathAdditions: ["Lib/site-packages"],
        pathAdditions: [".", "Lib/site-packages/torch/lib"],
      },
    }), "utf8");
    const runner: RuntimeCommandRunner = vi.fn(async () => ({
      code: 0,
      stderr: "",
      stdout: "MYCELLIOS_RUNTIME_PROBE=" + JSON.stringify({
        backend: "rocm",
        device: "cuda:0",
        device_name: "AMD Radeon(TM) 890M Graphics",
        torch_version: "2.9.1+rocm7.2.1",
        python_version: "3.12.13",
        cuda_version: null,
        hip_version: "7.2.53211",
        precision: "float16",
        operation: "fp16_matmul",
        checksum: 32767.5,
        total_memory_bytes: 8_589_934_592,
        allocated_bytes: 786_432,
        reserved_bytes: 2_097_152,
      }) + "\n",
    }));
    const result = await prepareAcceleratorRuntime({
      baseRuntimeRoot: base,
      userDataPath: userData,
      hardware: {
        platform: "win32",
        arch: "x64",
        osRelease: "10.0.26100",
        gpuVendor: "amd",
        gpuModel: "AMD Radeon(TM) 890M Graphics",
      },
      allowProvisioning: false,
      commandRunner: runner,
    });
    expect(result).toMatchObject({
      status: "gpu-ready",
      effectiveBackend: "rocm",
      deviceType: "gpu",
      precision: "float16",
      deviceName: "AMD Radeon(TM) 890M Graphics",
    });
    expect(runner).toHaveBeenCalledOnce();
  });

  it("emits monotonic structured progress and installs verified local artifacts", async () => {
    const root = temporaryRoot();
    const base = createBaseRuntime(root);
    const abandonedStaging = join(
      root,
      "user-data",
      "accelerator-runtimes-v1",
      `${WINDOWS_ACCELERATOR_PACKS.cuda.id}.staging-abandoned`,
    );
    mkdirSync(abandonedStaging, { recursive: true });
    writeFileSync(join(abandonedStaging, "incomplete.txt"), "interrupted", "utf8");
    const artifactBytes = Buffer.from("verified-test-cuda-wheel");
    replaceCudaArtifact(artifactBytes);
    const events: import("../src/desktop/accelerator-runtime.js").AcceleratorProgressEvent[] = [];
    const downloadedPaths: string[] = [];
    const artifactDownloader = vi.fn(async (artifact, cacheRoot, onProgress) => {
      onProgress({
        bytesDownloaded: Math.floor(artifact.sizeBytes / 2),
        bytesTotal: artifact.sizeBytes,
        bytesPerSecond: 32 * 1024 * 1024,
        etaSeconds: 39,
        resumed: false,
      });
      onProgress({
        bytesDownloaded: artifact.sizeBytes,
        bytesTotal: artifact.sizeBytes,
        bytesPerSecond: 32 * 1024 * 1024,
        etaSeconds: 0,
        resumed: false,
      });
      const path = join(cacheRoot, artifact.sha256, "torch.whl");
      mkdirSync(join(cacheRoot, artifact.sha256), { recursive: true });
      writeFileSync(path, artifactBytes);
      downloadedPaths.push(path);
      return path;
    });
    const runner: RuntimeCommandRunner = vi.fn(async (_executable, args) => {
      if (args[0] === "-c") return cudaProbeResult();
      return { code: 0, stdout: "", stderr: "" };
    });

    const result = await prepareAcceleratorRuntime({
      baseRuntimeRoot: base,
      userDataPath: join(root, "user-data"),
      hardware: {
        platform: "win32",
        arch: "x64",
        gpuVendor: "nvidia",
        gpuModel: "NVIDIA GeForce RTX 2060",
      },
      allowProvisioning: true,
      commandRunner: runner,
      artifactDownloader,
      onProgress: (event) => events.push(event),
    });

    expect(result.status).toBe("gpu-ready");
    expect(events[0]).toMatchObject({
      phase: "detecting",
      backend: "cuda",
      gpuModel: "NVIDIA GeForce RTX 2060",
      cpuAvailable: true,
    });
    expect(events.at(-1)).toMatchObject({ phase: "ready", percent: 100, recordLog: true });
    expect(events.map((event) => event.percent)).toEqual(
      [...events.map((event) => event.percent)].sort((left, right) => left - right),
    );
    const byteTick = events.find(
      (event) => event.download?.bytesDownloaded === Math.floor(artifactBytes.length / 2),
    );
    expect(byteTick).toMatchObject({ phase: "downloading", recordLog: false });
    expect(byteTick?.download).toMatchObject({
      aggregateTotal: artifactBytes.length,
      bytesPerSecond: 32 * 1024 * 1024,
      etaSeconds: 1,
    });
    expect(events.map((event) => event.phase)).toEqual(expect.arrayContaining([
      "checking-prerequisites",
      "downloading",
      "verifying-package",
      "copying-base",
      "installing",
      "physical-probe",
      "activating",
      "ready",
    ]));
    const pipInstall = vi.mocked(runner).mock.calls.find((call) => call[1].includes("install"));
    expect(pipInstall?.[1]).toEqual(expect.arrayContaining(downloadedPaths));
    expect(pipInstall?.[1]).toEqual(expect.arrayContaining([
      "--no-index",
      "--no-deps",
      "--no-build-isolation",
      "--ignore-installed",
    ]));
    expect(pipInstall?.[1].some((argument) => argument.startsWith("https://"))).toBe(false);
    expect(vi.mocked(runner).mock.calls.some((call) => call[1].includes("uninstall"))).toBe(false);
    expect(existsSync(abandonedStaging)).toBe(false);
  });

  it("rejects an injected downloader path outside the artifact cache", async () => {
    const root = temporaryRoot();
    const base = createBaseRuntime(root);
    const artifactBytes = Buffer.from("verified-test-cuda-wheel");
    replaceCudaArtifact(artifactBytes);
    const externalPath = join(root, "external.whl");
    writeFileSync(externalPath, artifactBytes);
    const events: import("../src/desktop/accelerator-runtime.js").AcceleratorProgressEvent[] = [];
    const runner = vi.fn<RuntimeCommandRunner>();

    const result = await prepareAcceleratorRuntime({
      baseRuntimeRoot: base,
      userDataPath: join(root, "user-data"),
      hardware: cudaHardware(),
      allowProvisioning: true,
      commandRunner: runner,
      artifactDownloader: vi.fn(async () => externalPath),
      onProgress: (event) => events.push(event),
    });

    expect(result).toMatchObject({ status: "gpu-fallback", effectiveBackend: "cpu" });
    expect(result.fallbackReason).toContain("outside its managed cache directory");
    expect(events.at(-1)?.issue?.code).toBe("integrity");
    expect(runner).not.toHaveBeenCalled();
  });

  it("rejects a missing file returned by an injected downloader", async () => {
    const root = temporaryRoot();
    const base = createBaseRuntime(root);
    const artifactBytes = Buffer.from("verified-test-cuda-wheel");
    const artifact = replaceCudaArtifact(artifactBytes);
    const missingPath = join(
      root,
      "user-data",
      "accelerator-runtimes-v1",
      "package-cache",
      artifact.sha256,
      "missing.whl",
    );
    const runner = vi.fn<RuntimeCommandRunner>();

    const result = await prepareAcceleratorRuntime({
      baseRuntimeRoot: base,
      userDataPath: join(root, "user-data"),
      hardware: cudaHardware(),
      allowProvisioning: true,
      commandRunner: runner,
      artifactDownloader: vi.fn(async () => missingPath),
    });

    expect(result.status).toBe("gpu-fallback");
    expect(result.fallbackReason).toContain(`has 0 bytes; expected ${artifactBytes.length}`);
    expect(runner).not.toHaveBeenCalled();
  });

  it("rejects an exact-size file with the wrong hash from an injected downloader", async () => {
    const root = temporaryRoot();
    const base = createBaseRuntime(root);
    const artifactBytes = Buffer.from("verified-test-cuda-wheel");
    replaceCudaArtifact(artifactBytes);
    const runner = vi.fn<RuntimeCommandRunner>();

    const result = await prepareAcceleratorRuntime({
      baseRuntimeRoot: base,
      userDataPath: join(root, "user-data"),
      hardware: cudaHardware(),
      allowProvisioning: true,
      commandRunner: runner,
      artifactDownloader: vi.fn(async (artifact, cacheRoot) => {
        const path = join(cacheRoot, artifact.sha256, "tampered.whl");
        mkdirSync(join(cacheRoot, artifact.sha256), { recursive: true });
        writeFileSync(path, Buffer.alloc(artifactBytes.length, 0x78));
        return path;
      }),
    });

    expect(result.status).toBe("gpu-fallback");
    expect(result.fallbackReason).toContain("failed SHA-256 verification");
    expect(runner).not.toHaveBeenCalled();
  });

  it("reports an actionable installation requirement while CPU remains available", async () => {
    const root = temporaryRoot();
    const base = createBaseRuntime(root);
    const events: import("../src/desktop/accelerator-runtime.js").AcceleratorProgressEvent[] = [];
    const result = await prepareAcceleratorRuntime({
      baseRuntimeRoot: base,
      userDataPath: join(root, "user-data"),
      hardware: {
        platform: "win32",
        arch: "x64",
        gpuVendor: "nvidia",
        gpuModel: "NVIDIA GeForce RTX 2060",
      },
      allowProvisioning: false,
      onProgress: (event) => events.push(event),
    });
    expect(result.status).toBe("gpu-fallback");
    expect(events.at(-1)).toMatchObject({
      phase: "fallback",
      cpuAvailable: true,
      issue: {
        code: "installation-required",
        retryable: true,
      },
    });
    expect(events.at(-1)?.issue?.action).toContain("Enable contribution");
  });

  it("does not download the Windows ROCm pack on Windows 10", async () => {
    const root = temporaryRoot();
    const base = createBaseRuntime(root);
    const events: import("../src/desktop/accelerator-runtime.js").AcceleratorProgressEvent[] = [];
    const downloader = vi.fn();

    const result = await prepareAcceleratorRuntime({
      baseRuntimeRoot: base,
      userDataPath: join(root, "user-data"),
      hardware: {
        platform: "win32",
        arch: "x64",
        osRelease: "10.0.19045",
        gpuVendor: "amd",
        gpuModel: "AMD Radeon RX 9070 XT",
      },
      allowProvisioning: true,
      artifactDownloader: downloader,
      onProgress: (event) => events.push(event),
    });

    expect(result).toMatchObject({ status: "cpu-ready", effectiveBackend: "cpu" });
    expect(downloader).not.toHaveBeenCalled();
    expect(events.at(-1)?.issue).toMatchObject({
      code: "unsupported-platform",
      retryable: false,
    });
    expect(events.at(-1)?.issue?.message).toContain("requires Windows 11");
  });

  it("resumes a partial HTTPS artifact and validates its exact SHA-256", async () => {
    const root = temporaryRoot();
    const cacheRoot = join(root, "cache");
    const bytes = Buffer.from("mycellios-verified-package");
    const sha256 = createHash("sha256").update(bytes).digest("hex");
    const artifact = {
      label: "Test wheel",
      url: `https://packages.example/test.whl#sha256=${sha256}`,
      sha256,
      sizeBytes: bytes.length,
    };
    const partial = join(cacheRoot, sha256, "test.whl.part");
    const prefixLength = 9;
    mkdirSync(join(cacheRoot, sha256), { recursive: true });
    writeFileSync(partial, bytes.subarray(0, prefixLength));
    const fetchMock = vi.fn(async (_input: string | URL | Request, init?: RequestInit) => {
      expect(init?.headers).toEqual({ range: `bytes=${prefixLength}-` });
      return new Response(bytes.subarray(prefixLength), {
        status: 206,
        headers: {
          "content-range": `bytes ${prefixLength}-${bytes.length - 1}/${bytes.length}`,
        },
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    const transfers: import("../src/desktop/accelerator-runtime.js").AcceleratorArtifactTransfer[] = [];

    const target = await downloadPinnedArtifact(artifact, cacheRoot, (transfer) => transfers.push(transfer));

    expect(readFileSync(target)).toEqual(bytes);
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(transfers[0]).toMatchObject({ bytesDownloaded: prefixLength, resumed: true });
    expect(transfers.at(-1)).toMatchObject({
      bytesDownloaded: bytes.length,
      bytesTotal: bytes.length,
      etaSeconds: 0,
      resumed: true,
    });
  });

  it("throttles body-chunk progress to about four updates per second", async () => {
    const root = temporaryRoot();
    const cacheRoot = join(root, "cache");
    const bytes = Buffer.from("abcdefghijklmnopqrst");
    const sha256 = createHash("sha256").update(bytes).digest("hex");
    const response = new Response(new ReadableStream<Uint8Array>({
      start(controller) {
        for (let index = 0; index < bytes.length; index += 2) {
          controller.enqueue(bytes.subarray(index, index + 2));
        }
        controller.close();
      },
    }), { status: 200 });
    vi.stubGlobal("fetch", vi.fn(async () => response));
    let now = 0;
    vi.spyOn(Date, "now").mockImplementation(() => {
      now += 50;
      return now;
    });
    const transfers: import("../src/desktop/accelerator-runtime.js").AcceleratorArtifactTransfer[] = [];

    await downloadPinnedArtifact({
      label: "Chunked wheel",
      url: `https://packages.example/chunked.whl#sha256=${sha256}`,
      sha256,
      sizeBytes: bytes.length,
    }, cacheRoot, (transfer) => transfers.push(transfer));

    expect(transfers.map((transfer) => transfer.bytesDownloaded)).toEqual([0, 10, bytes.length]);
    expect(transfers.at(-1)).toMatchObject({
      bytesDownloaded: bytes.length,
      bytesTotal: bytes.length,
      etaSeconds: 0,
    });
  });

  it("rejects and removes a package whose bytes do not match its pinned digest", async () => {
    const root = temporaryRoot();
    const cacheRoot = join(root, "cache");
    const expected = Buffer.from("expected");
    const received = Buffer.from("tampered");
    const sha256 = createHash("sha256").update(expected).digest("hex");
    const artifact = {
      label: "Tampered wheel",
      url: `https://packages.example/tampered.whl#sha256=${sha256}`,
      sha256,
      sizeBytes: received.length,
    };
    vi.stubGlobal("fetch", vi.fn(async () => new Response(received, { status: 200 })));

    await expect(downloadPinnedArtifact(artifact, cacheRoot, () => undefined))
      .rejects.toThrow("failed SHA-256 verification");
    expect(() => readFileSync(join(cacheRoot, sha256, "tampered.whl.part"))).toThrow();
  });

  it("rejects an encoded package name that escapes its managed cache directory", async () => {
    const root = temporaryRoot();
    const sha256 = "a".repeat(64);
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(downloadPinnedArtifact({
      label: "Escaping wheel",
      url: `https://packages.example/%2E%2E%2Fevil.whl#sha256=${sha256}`,
      sha256,
      sizeBytes: 10,
    }, join(root, "cache"), () => undefined)).rejects.toThrow("escaped its managed directory");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

function temporaryRoot(): string {
  const root = mkdtempSync(join(tmpdir(), "mycellios-accelerator-runtime-"));
  temporaryDirectories.push(root);
  return root;
}

function createBaseRuntime(root: string): string {
  const base = join(root, "base-runtime");
  mkdirSync(join(base, "Lib", "site-packages"), { recursive: true });
  writeFileSync(join(base, "python.exe"), "test", "utf8");
  writeFileSync(join(base, "runtime-manifest.json"), JSON.stringify({
    schema: PORTABLE_RUNTIME_SCHEMA,
    platform: "win32",
    arch: "x64",
    pythonVersion: "3.12.13",
    pythonAbi: "cp312",
    executable: "python.exe",
    pythonArtifact: portablePythonArtifact("win32/x64"),
    torchVersion: "2.13.0+cpu",
    transformersVersion: "5.14.1",
    accelerateVersion: "1.14.0",
    safetensorsVersion: "0.8.0",
    aiohttpVersion: "3.14.1",
    sentencepieceVersion: "0.2.2",
    numpyVersion: "1.26.4",
    backend: "cpu",
  }), "utf8");
  return base;
}

function createMpsBaseRuntime(root: string): string {
  const base = join(root, "mps-base-runtime");
  mkdirSync(join(base, "lib", "python3.12", "site-packages"), { recursive: true });
  mkdirSync(join(base, "bin"), { recursive: true });
  writeFileSync(join(base, "bin", "python3"), "test", "utf8");
  writeFileSync(join(base, "runtime-manifest.json"), JSON.stringify({
    schema: PORTABLE_RUNTIME_SCHEMA,
    platform: "darwin",
    arch: "arm64",
    pythonVersion: "3.12.13",
    pythonAbi: "cp312",
    executable: "bin/python3",
    pythonArtifact: portablePythonArtifact("darwin/arm64"),
    torchVersion: "2.13.0",
    transformersVersion: "5.14.1",
    accelerateVersion: "1.14.0",
    safetensorsVersion: "0.8.0",
    aiohttpVersion: "3.14.1",
    sentencepieceVersion: "0.2.2",
    numpyVersion: "1.26.4",
    backend: "cpu",
  }), "utf8");
  return base;
}

function createMacIntelBaseRuntime(root: string): string {
  const base = join(root, "mac-intel-base-runtime");
  mkdirSync(join(base, "lib", "python3.12", "site-packages"), { recursive: true });
  mkdirSync(join(base, "bin"), { recursive: true });
  writeFileSync(join(base, "bin", "python3"), "test", "utf8");
  writeFileSync(join(base, "runtime-manifest.json"), JSON.stringify({
    schema: PORTABLE_RUNTIME_SCHEMA,
    platform: "darwin",
    arch: "x64",
    pythonVersion: "3.12.13",
    pythonAbi: "cp312",
    executable: "bin/python3",
    pythonArtifact: portablePythonArtifact("darwin/x64"),
    torchVersion: "2.2.2",
    transformersVersion: "4.57.3",
    accelerateVersion: "1.14.0",
    safetensorsVersion: "0.8.0",
    aiohttpVersion: "3.14.1",
    sentencepieceVersion: "0.2.2",
    numpyVersion: "1.26.4",
    backend: "cpu",
  }), "utf8");
  return base;
}

function portablePythonArtifact(platform: "win32/x64" | "darwin/arm64" | "darwin/x64") {
  const metadata = platform === "win32/x64"
    ? {
        filename: "cpython-3.12.13+20260510-x86_64-pc-windows-msvc-install_only_stripped.tar.gz",
        size: 21_921_642,
        sha256: "24168aff2e7d93784c6a436124c4ebb79b076a4e289bde4902c08333507b71d0",
      }
    : platform === "darwin/arm64"
      ? {
          filename: "cpython-3.12.13+20260510-aarch64-apple-darwin-install_only_stripped.tar.gz",
          size: 24_942_229,
          sha256: "55bc1a5edbc8ac4da0081f4f5731ed2d1ed10c57cb37a820b2a0dbc7cad742e9",
        }
      : {
          filename: "cpython-3.12.13+20260510-x86_64-apple-darwin-install_only_stripped.tar.gz",
          size: 24_639_521,
          sha256: "6bab7fa97d4f2ddba86da0e05acff66c53b5edaca1df8edcf00ddca785a9c59b",
        };
  return {
    source: "astral-sh/python-build-standalone",
    release: "20260510",
    version: "3.12.13",
    flavor: "install_only_stripped",
    filename: metadata.filename,
    url: `https://github.com/astral-sh/python-build-standalone/releases/download/20260510/${metadata.filename}`,
    size: metadata.size,
    sha256: metadata.sha256,
  };
}

function cudaHardware() {
  return {
    platform: "win32" as const,
    arch: "x64",
    gpuVendor: "nvidia",
    gpuModel: "NVIDIA GeForce RTX 2060",
  };
}

function replaceCudaArtifact(bytes: Buffer) {
  const packageArtifact = WINDOWS_ACCELERATOR_PACKS.cuda.installGroups[0]?.[0];
  if (!packageArtifact) throw new Error("CUDA test artifact is missing");
  const previous = { ...packageArtifact };
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  Object.assign(packageArtifact, {
    label: "Test CUDA wheel",
    url: `https://packages.example/test.whl#sha256=${sha256}`,
    sha256,
    sizeBytes: bytes.length,
  });
  artifactRestorers.push(() => Object.assign(packageArtifact, previous));
  return packageArtifact;
}

function cudaProbeResult() {
  return {
    code: 0,
    stderr: "",
    stdout: "MYCELLIOS_RUNTIME_PROBE=" + JSON.stringify({
      backend: "cuda",
      device: "cuda:0",
      device_name: "NVIDIA GeForce RTX 2060",
      torch_version: "2.13.0+cu126",
      python_version: "3.12.13",
      cuda_version: "12.6",
      hip_version: null,
      precision: "float16",
      operation: "fp16_matmul",
      checksum: 32_768,
      total_memory_bytes: 6_442_450_944,
      allocated_bytes: 786_432,
      reserved_bytes: 2_097_152,
    }) + "\n",
  };
}

function mpsProbeResult() {
  return {
    code: 0,
    stderr: "",
    stdout: "MYCELLIOS_RUNTIME_PROBE=" + JSON.stringify({
      backend: "mps",
      device: "mps",
      device_name: "Apple M4 Pro",
      torch_version: "2.13.0",
      python_version: "3.12.13",
      cuda_version: null,
      hip_version: null,
      precision: "float16",
      operation: "fp16_matmul",
      checksum: 32_768,
      total_memory_bytes: 12_884_901_888,
      allocated_bytes: 786_432,
      reserved_bytes: 2_097_152,
    }) + "\n",
  };
}
