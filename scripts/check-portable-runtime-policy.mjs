import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  DISTRIBUTION_PACKAGE_VERSIONS,
  PORTABLE_PYTHON_FLAVOR,
  PORTABLE_PYTHON_RELEASE,
  PORTABLE_PYTHON_SOURCE,
  PORTABLE_PYTHON_VERSION,
  PORTABLE_RUNTIME_SCHEMA,
  portableRuntimeSpec,
} from "./portable-runtime-policy.mjs";
import {
  PORTABLE_RUNTIME_WHEEL_LOCK_SCHEMA,
  readPortableRuntimeWheelLock,
} from "./portable-runtime-wheel-lock.mjs";

const expected = [
  ["win32", "x64", true, "2.13.0+cpu", [], "cpython-3.12.13+20260510-x86_64-pc-windows-msvc-install_only_stripped.tar.gz", 21_921_642, "24168aff2e7d93784c6a436124c4ebb79b076a4e289bde4902c08333507b71d0", "scripts/wheel-locks/win32-x64-cp312.txt", "2a5de3d3f2e3ba4ebac91e1efe068159e81263d3ccd6a3d93333078e753c6c65"],
  ["linux", "x64", true, "2.13.0+cpu", [], "cpython-3.12.13+20260510-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz", 34_075_380, "d480f5d5878910ecbae212bf23bd7c25d7b209eb8cf5e98823c977384d272e88", "scripts/wheel-locks/linux-x64-cp312.txt", "090355535c96e7202ae761a18dadc5d66c9e98493a148eba16ae3c3e7c95c59d"],
  ["darwin", "arm64", true, "2.11.0", ["mps"], "cpython-3.12.13+20260510-aarch64-apple-darwin-install_only_stripped.tar.gz", 24_942_229, "55bc1a5edbc8ac4da0081f4f5731ed2d1ed10c57cb37a820b2a0dbc7cad742e9", "scripts/wheel-locks/darwin-arm64-cp312.txt", "75c84302540edc1b7f6c1620c3474b7ccbf431dd6df881717f74ebf419c56348"],
  ["darwin", "x64", true, "2.2.2", [], "cpython-3.12.13+20260510-x86_64-apple-darwin-install_only_stripped.tar.gz", 24_639_521, "6bab7fa97d4f2ddba86da0e05acff66c53b5edaca1df8edcf00ddca785a9c59b", "scripts/wheel-locks/darwin-x64-cp312.txt", "808a7fd6894862abf2ed704eef036bf0b6290c6cce4f618682e831d67bf2b5eb"],
  ["linux", "arm64", false, undefined, undefined, undefined, undefined, undefined, undefined, undefined],
];

if (PORTABLE_RUNTIME_SCHEMA !== "mycellios-distribution-runtime/4") {
  throw new Error(`Unexpected portable runtime schema: ${PORTABLE_RUNTIME_SCHEMA}.`);
}

const workspace = resolve(import.meta.dirname, "..");
for (const [
  platform,
  arch,
  supported,
  torchVersion,
  accelerators,
  filename,
  size,
  sha256,
  wheelLockPath,
  wheelLockSha256,
] of expected) {
  const spec = portableRuntimeSpec(platform, arch);
  if (spec.supported !== supported) {
    throw new Error(`Unexpected runtime policy for ${platform}/${arch}: ${JSON.stringify(spec)}.`);
  }
  if (supported) {
    if (spec.torchVersion !== torchVersion) {
      throw new Error(`Unexpected torch version for ${platform}/${arch}: ${spec.torchVersion}.`);
    }
    if (JSON.stringify(spec.bundledAccelerators) !== JSON.stringify(accelerators)) {
      throw new Error(`Unexpected accelerators for ${platform}/${arch}: ${spec.bundledAccelerators}.`);
    }
    for (const [name, defaultVersion] of Object.entries(DISTRIBUTION_PACKAGE_VERSIONS)) {
      const expectedVersion = platform === "darwin" && arch === "x64" && name === "transformers"
        ? "4.57.3"
        : defaultVersion;
      if (spec.packageVersions?.[name] !== expectedVersion) {
        throw new Error(
          `Unexpected ${name} version for ${platform}/${arch}: ${spec.packageVersions?.[name]}.`,
        );
      }
    }
    const artifact = spec.pythonArtifact;
    const expectedUrl = `https://github.com/${PORTABLE_PYTHON_SOURCE}/releases/download/${PORTABLE_PYTHON_RELEASE}/${filename}`;
    if (
      spec.pythonVersion !== PORTABLE_PYTHON_VERSION ||
      artifact?.source !== PORTABLE_PYTHON_SOURCE ||
      artifact?.release !== PORTABLE_PYTHON_RELEASE ||
      artifact?.version !== PORTABLE_PYTHON_VERSION ||
      artifact?.flavor !== PORTABLE_PYTHON_FLAVOR ||
      artifact?.filename !== filename ||
      artifact?.url !== expectedUrl ||
      artifact?.size !== size ||
      artifact?.sha256 !== sha256 ||
      spec.wheelLock?.schema !== PORTABLE_RUNTIME_WHEEL_LOCK_SCHEMA ||
      spec.wheelLock?.path !== wheelLockPath ||
      spec.wheelLock?.sha256 !== wheelLockSha256
    ) {
      throw new Error(`Unexpected Python artifact for ${platform}/${arch}: ${JSON.stringify(artifact)}.`);
    }
    const wheelLock = readPortableRuntimeWheelLock(workspace, spec);
    if (
      wheelLock.platform !== platform ||
      wheelLock.arch !== arch ||
      wheelLock.path !== wheelLockPath ||
      wheelLock.sha256 !== wheelLockSha256
    ) {
      throw new Error(
        `Unexpected wheel lock for ${platform}/${arch}: ${JSON.stringify(wheelLock)}.`,
      );
    }
  }
}

const requirementsPath = resolve(import.meta.dirname, "..", "python", "requirements-distribution.txt");
const requirements = new Map(
  readFileSync(requirementsPath, "utf8")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"))
    .map((line) => {
      const match = /^([A-Za-z0-9_.-]+)==([^\s]+)$/.exec(line);
      if (!match) throw new Error(`Distribution requirement is not exactly pinned: ${line}.`);
      return [match[1].toLowerCase(), match[2]];
    }),
);
for (const [name, version] of Object.entries(DISTRIBUTION_PACKAGE_VERSIONS)) {
  if (requirements.get(name) !== version) {
    throw new Error(`Expected ${name}==${version} in ${requirementsPath}.`);
  }
}
if (requirements.size !== Object.keys(DISTRIBUTION_PACKAGE_VERSIONS).length) {
  throw new Error("The distribution requirements and portable-runtime policy have drifted.");
}

process.stdout.write("Portable runtime policy verified for Windows x64, Linux x64, macOS arm64, and macOS Intel x64.\n");
