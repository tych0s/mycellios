import {
  mkdirSync,
  mkdtempSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ACCELERATOR_RUNTIME_SCHEMA,
  PORTABLE_RUNTIME_SCHEMA,
  WINDOWS_ACCELERATOR_PACKS,
  acceleratorRuntimeTargetPath,
  prepareAcceleratorRuntime,
  probeAcceleratorRuntime,
  selectAcceleratorPack,
  type RuntimeCommandRunner,
} from "../src/desktop/accelerator-runtime.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("desktop accelerator runtime", () => {
  it("pins every vendor package URL to an exact SHA-256 digest", () => {
    for (const pack of Object.values(WINDOWS_ACCELERATOR_PACKS)) {
      for (const url of pack.installGroups.flat()) {
        expect(url).toMatch(/#sha256=[0-9a-f]{64}$/);
      }
    }
  });

  it("selects only pinned Windows accelerator packs", () => {
    expect(selectAcceleratorPack({
      platform: "win32",
      arch: "x64",
      gpuVendor: "nvidia",
      gpuModel: "NVIDIA GeForce RTX 2060",
    })?.backend).toBe("cuda");
    expect(selectAcceleratorPack({
      platform: "win32",
      arch: "x64",
      gpuVendor: "amd",
      gpuModel: "AMD Radeon(TM) 890M Graphics",
      cpuModel: "AMD Ryzen AI 9 HX 370 w/ Radeon 890M",
    })?.backend).toBe("rocm");
    expect(selectAcceleratorPack({
      platform: "win32",
      arch: "x64",
      gpuVendor: "amd",
      gpuModel: "AMD Radeon RX 580",
    })).toBeNull();
    expect(selectAcceleratorPack({
      platform: "linux",
      arch: "x64",
      gpuVendor: "nvidia",
      gpuModel: "NVIDIA GeForce RTX 2060",
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

  it("reuses a cached runtime only after probing it again", async () => {
    const root = temporaryRoot();
    const base = createBaseRuntime(root);
    const userData = join(root, "user-data");
    const target = acceleratorRuntimeTargetPath(userData, "rocm");
    mkdirSync(join(target, "Lib", "site-packages", "torch", "lib"), { recursive: true });
    writeFileSync(join(target, "python.exe"), "test", "utf8");
    writeFileSync(join(target, "accelerator-runtime.json"), JSON.stringify({
      schema: ACCELERATOR_RUNTIME_SCHEMA,
      packId: "win-x64-py312-torch291-rocm721-v1",
      backend: "rocm",
      baseRuntimeSchema: PORTABLE_RUNTIME_SCHEMA,
      pythonVersion: "3.12.13",
      torchVersion: "2.9.1+rocm7.2.1",
      packageUrls: WINDOWS_ACCELERATOR_PACKS.rocm.installGroups.flat(),
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
    torchVersion: "2.13.0+cpu",
    transformersVersion: "5.14.1",
    accelerateVersion: "1.14.0",
    safetensorsVersion: "0.8.0",
    backend: "cpu",
  }), "utf8");
  return base;
}
