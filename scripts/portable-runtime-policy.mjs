export const PORTABLE_RUNTIME_SCHEMA = "mycellios-distribution-runtime/3";
export const PORTABLE_PYTHON_MINOR = "3.12";
export const PORTABLE_PYTHON_VERSION = "3.12.13";
export const PORTABLE_PYTHON_RELEASE = "20260510";
export const PORTABLE_PYTHON_SOURCE = "astral-sh/python-build-standalone";
export const PORTABLE_PYTHON_FLAVOR = "install_only_stripped";
export const PORTABLE_PYTHON_PROVENANCE_SCHEMA = "mycellios-python-standalone/1";
export const PORTABLE_PYTHON_PROVENANCE_FILE = "python-artifact.json";

export const DISTRIBUTION_PACKAGE_VERSIONS = Object.freeze({
  numpy: "1.26.4",
  aiohttp: "3.14.1",
  accelerate: "1.14.0",
  transformers: "5.14.1",
  safetensors: "0.8.0",
  sentencepiece: "0.2.2",
});
const MACOS_X64_PACKAGE_VERSIONS = Object.freeze({
  ...DISTRIBUTION_PACKAGE_VERSIONS,
  // Transformers 5 requires PyTorch >=2.4 and disables the last official
  // macOS Intel wheel. 4.57.3 retains Qwen3 support with PyTorch 2.2.2.
  transformers: "4.57.3",
});

const CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu";
const PYPI_INDEX = "https://pypi.org/simple";
const PYTHON_RELEASE_URL =
  `https://github.com/${PORTABLE_PYTHON_SOURCE}/releases/download/${PORTABLE_PYTHON_RELEASE}`;

function pythonArtifact(filename, size, sha256) {
  return Object.freeze({
    source: PORTABLE_PYTHON_SOURCE,
    release: PORTABLE_PYTHON_RELEASE,
    version: PORTABLE_PYTHON_VERSION,
    flavor: PORTABLE_PYTHON_FLAVOR,
    filename,
    url: `${PYTHON_RELEASE_URL}/${filename}`,
    size,
    sha256,
  });
}

const WINDOWS_X64_PYTHON = pythonArtifact(
  "cpython-3.12.13+20260510-x86_64-pc-windows-msvc-install_only_stripped.tar.gz",
  21_921_642,
  "24168aff2e7d93784c6a436124c4ebb79b076a4e289bde4902c08333507b71d0",
);
const LINUX_X64_PYTHON = pythonArtifact(
  "cpython-3.12.13+20260510-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz",
  34_075_380,
  "d480f5d5878910ecbae212bf23bd7c25d7b209eb8cf5e98823c977384d272e88",
);
const MACOS_ARM64_PYTHON = pythonArtifact(
  "cpython-3.12.13+20260510-aarch64-apple-darwin-install_only_stripped.tar.gz",
  24_942_229,
  "55bc1a5edbc8ac4da0081f4f5731ed2d1ed10c57cb37a820b2a0dbc7cad742e9",
);
const MACOS_X64_PYTHON = pythonArtifact(
  "cpython-3.12.13+20260510-x86_64-apple-darwin-install_only_stripped.tar.gz",
  24_639_521,
  "6bab7fa97d4f2ddba86da0e05acff66c53b5edaca1df8edcf00ddca785a9c59b",
);

/**
 * Return the release policy for a platform-native Python runtime.
 */
export function portableRuntimeSpec(platform, arch) {
  if (platform === "win32" && arch === "x64") {
    return Object.freeze({
      supported: true,
      platform,
      arch,
      pythonVersion: PORTABLE_PYTHON_VERSION,
      pythonExecutable: "python.exe",
      pythonArtifact: WINDOWS_X64_PYTHON,
      packageVersions: DISTRIBUTION_PACKAGE_VERSIONS,
      torchRequirement: "torch==2.13.0+cpu",
      torchVersion: "2.13.0+cpu",
      torchIndex: CPU_TORCH_INDEX,
      bundledAccelerators: Object.freeze([]),
    });
  }
  if (platform === "linux" && arch === "x64") {
    return Object.freeze({
      supported: true,
      platform,
      arch,
      pythonVersion: PORTABLE_PYTHON_VERSION,
      pythonExecutable: "bin/python3",
      pythonArtifact: LINUX_X64_PYTHON,
      packageVersions: DISTRIBUTION_PACKAGE_VERSIONS,
      torchRequirement: "torch==2.13.0+cpu",
      torchVersion: "2.13.0+cpu",
      torchIndex: CPU_TORCH_INDEX,
      bundledAccelerators: Object.freeze([]),
    });
  }
  if (platform === "darwin" && arch === "arm64") {
    return Object.freeze({
      supported: true,
      platform,
      arch,
      pythonVersion: PORTABLE_PYTHON_VERSION,
      pythonExecutable: "bin/python3",
      pythonArtifact: MACOS_ARM64_PYTHON,
      packageVersions: DISTRIBUTION_PACKAGE_VERSIONS,
      // The official macOS arm64 wheel contains the MPS backend. It is built
      // into the signed application runtime; the client performs no pip install.
      torchRequirement: "torch==2.13.0",
      torchVersion: "2.13.0",
      torchIndex: PYPI_INDEX,
      bundledAccelerators: Object.freeze(["mps"]),
    });
  }
  if (platform === "darwin" && arch === "x64") {
    return Object.freeze({
      supported: true,
      platform,
      arch,
      pythonVersion: PORTABLE_PYTHON_VERSION,
      pythonExecutable: "bin/python3",
      pythonArtifact: MACOS_X64_PYTHON,
      packageVersions: MACOS_X64_PACKAGE_VERSIONS,
      // Intel Macs receive the same self-contained inference stack, but the
      // release makes no native Metal/MPS claim for this architecture.
      // PyTorch 2.2.2 is the final official macOS x86_64 wheel supporting
      // CPython 3.12. Newer PyPI releases publish macOS arm64 only.
      torchRequirement: "torch==2.2.2",
      torchVersion: "2.2.2",
      torchIndex: PYPI_INDEX,
      bundledAccelerators: Object.freeze([]),
    });
  }
  return Object.freeze({
    supported: false,
    platform,
    arch,
    reason: `No certified portable Python runtime is published for ${platform}/${arch}.`,
  });
}

export function normalizePythonMachine(value) {
  const machine = String(value ?? "").trim().toLowerCase();
  if (machine === "amd64" || machine === "x86_64") return "x64";
  if (machine === "arm64" || machine === "aarch64") return "arm64";
  return machine;
}

export function matchesPinnedPythonArtifact(actual, expected) {
  return Boolean(
    actual &&
    expected &&
    actual.source === expected.source &&
    actual.release === expected.release &&
    actual.version === expected.version &&
    actual.flavor === expected.flavor &&
    actual.filename === expected.filename &&
    actual.url === expected.url &&
    actual.size === expected.size &&
    actual.sha256 === expected.sha256
  );
}
