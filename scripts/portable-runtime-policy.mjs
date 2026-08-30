export const PORTABLE_RUNTIME_SCHEMA = "mycellios-distribution-runtime/4";
export const PORTABLE_PYTHON_MINOR = "3.12";
export const PORTABLE_PYTHON_VERSION = "3.12.13";
export const PORTABLE_PYTHON_RELEASE = "20260510";
export const PORTABLE_PYTHON_SOURCE = "astral-sh/python-build-standalone";
export const PORTABLE_PYTHON_FLAVOR = "install_only_stripped";
export const PORTABLE_PYTHON_PROVENANCE_SCHEMA = "mycellios-python-standalone/1";
export const PORTABLE_PYTHON_PROVENANCE_FILE = "python-artifact.json";

export const DISTRIBUTION_PACKAGE_VERSIONS = Object.freeze({
  numpy: "1.26.4",
  aiohttp: "3.14.3",
  accelerate: "1.14.0",
  transformers: "5.14.1",
  safetensors: "0.8.0",
  sentencepiece: "0.2.2",
});

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

const WINDOWS_X64_WHEEL_LOCK = wheelLock(
  "win32-x64-cp312.txt",
  "de504739e05a89229e437ec5d09a45dc00f2080db042d11519204926e1a90040",
);
const LINUX_X64_WHEEL_LOCK = wheelLock(
  "linux-x64-cp312.txt",
  "f3d9a819ebcc82c33da8ccad858eecf2d9ac4c35d5e8ad4b1da6702bfeb9748a",
);
const MACOS_ARM64_WHEEL_LOCK = wheelLock(
  "darwin-arm64-cp312.txt",
  "1adf58ca36108ca3c7e26c303a8c6c802bf25825e87e7d1d9f6ab2d86de3371a",
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
      wheelLock: WINDOWS_X64_WHEEL_LOCK,
      packageVersions: DISTRIBUTION_PACKAGE_VERSIONS,
      torchVersion: "2.13.0+cpu",
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
      wheelLock: LINUX_X64_WHEEL_LOCK,
      packageVersions: DISTRIBUTION_PACKAGE_VERSIONS,
      torchVersion: "2.13.0+cpu",
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
      wheelLock: MACOS_ARM64_WHEEL_LOCK,
      packageVersions: DISTRIBUTION_PACKAGE_VERSIONS,
      // The official macOS arm64 wheel contains the MPS backend. It is built
      // into the signed application runtime; the client performs no pip install.
      torchVersion: "2.11.0",
      bundledAccelerators: Object.freeze(["mps"]),
    });
  }
  if (platform === "darwin" && arch === "x64") {
    // macOS Intel RETIRADO el 26-07-2026, y no por decision nuestra.
    //
    // PyTorch dejo de publicar ruedas para macOS x86_64 en la version 2.3.0
    // (24-04-2024). La ultima con soporte fue la 2.2.2 (27-03-2024), asi que
    // esta plataforma quedaba clavada a marzo de 2024 mientras el resto va por
    // la 2.13.0.
    //
    // Eso arrastraba el resto: `transformers` 5 exige torch >=2.4, de modo que
    // Intel tenia que quedarse en la 4.57.3 — una segunda pila sellada, con su
    // propio lock de ruedas y su propio conjunto de versiones. Y aun asi NO
    // FUNCIONABA: `kv_arena.py` importa `DYNAMIC_LAYER_TYPE_MAPPING` de
    // `transformers.cache_utils` a nivel de modulo, simbolo que solo existe en
    // la 5. Verificado en CI: el runtime ni se importa.
    //
    // O sea que no se retira soporte que funcionaba; se deja de ofrecer una
    // descarga que no podia arrancar. Y como no hay CUDA ni MPS ahi, aunque
    // alguien lo arreglara seria un nodo de solo CPU.
    return Object.freeze({
      supported: false,
      platform,
      arch,
      reason:
        "PyTorch published its last macOS x86_64 wheel in 2.2.2 (March 2024); "
        + "transformers 5 requires torch >=2.4, so this architecture cannot run "
        + "the sealed inference stack.",
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

function wheelLock(filename, sha256) {
  return Object.freeze({
    schema: "mycellios-python-wheel-lock/1",
    path: `scripts/wheel-locks/${filename}`,
    sha256,
  });
}
