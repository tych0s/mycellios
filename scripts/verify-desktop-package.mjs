import { existsSync, readFileSync } from "node:fs";
import { join, posix, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { extractFile, listPackage } from "@electron/asar";
import {
  PORTABLE_PYTHON_PROVENANCE_FILE,
  PORTABLE_PYTHON_PROVENANCE_SCHEMA,
  PORTABLE_RUNTIME_SCHEMA,
  matchesPinnedPythonArtifact,
  portableRuntimeSpec,
} from "./portable-runtime-policy.mjs";
import { assertNativePythonProductMatchesSource } from "./native-python-product-policy.mjs";
import {
  NATIVE_BUILD_PROVENANCE_FILE,
  assertNativeSourceProvenanceMatches,
} from "./native-build-provenance.mjs";
import { readPortableRuntimeWheelLock } from "./portable-runtime-wheel-lock.mjs";

function readArgument(name, fallback) {
  const prefix = `--${name}=`;
  const value = process.argv.slice(2).find((argument) => argument.startsWith(prefix));
  return value ? value.slice(prefix.length) : fallback;
}

const platform = readArgument("platform", process.platform);
const arch = readArgument("arch", process.arch);
const bundleDirectory = resolve("out", `mycellios-${platform}-${arch}`);
const asarPath =
  platform === "darwin"
    ? resolve(bundleDirectory, "mycellios.app", "Contents", "Resources", "app.asar")
    : resolve(bundleDirectory, "resources", "app.asar");
const resourcesDirectory =
  platform === "darwin"
    ? resolve(bundleDirectory, "mycellios.app", "Contents", "Resources")
    : resolve(bundleDirectory, "resources");
const runtimeArchive = resolve(resourcesDirectory, "distribution-runtime.tar.gz");
const packagedPythonSource = resolve(resourcesDirectory, "python");

if (!existsSync(asarPath)) {
  throw new Error(`No existe el paquete esperado: ${asarPath}`);
}

const entries = new Set(listPackage(asarPath).map((entry) => entry.replaceAll("\\", "/")));
const requiredEntries = [
  "/.vite/build/main.cjs",
  "/.vite/build/preload.cjs",
  "/.vite/renderer/main_window/index.html",
  `/${NATIVE_BUILD_PROVENANCE_FILE}`,
];

for (const entry of requiredEntries) {
  if (!entries.has(entry)) throw new Error(`Falta ${entry} dentro de ${asarPath}`);
}

const packagedMetadata = JSON.parse(extractFile(asarPath, "package.json").toString("utf8"));
const sourceMetadata = JSON.parse(readFileSync(resolve("package.json"), "utf8"));
if (packagedMetadata.version !== sourceMetadata.version) {
  throw new Error(
    `La versión empaquetada ${packagedMetadata.version} no coincide con package.json ${sourceMetadata.version}.`,
  );
}
if (packagedMetadata.main !== ".vite/build/main.cjs") {
  throw new Error(`El entrypoint empaquetado es incorrecto: ${packagedMetadata.main}`);
}

const mainBundle = extractFile(asarPath, join(".vite", "build", "main.cjs")).toString("utf8");
const forbiddenRuntimeImports = [
  "fastify",
  "@fastify/websocket",
  "zod",
  "ws",
  "electron-squirrel-startup",
];

for (const dependency of forbiddenRuntimeImports) {
  const escaped = dependency.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const externalImport = new RegExp(`require\\([\"']${escaped}[\"']\\)`);
  if (externalImport.test(mainBundle)) {
    throw new Error(`Dependencia de runtime no incluida en el bundle: ${dependency}`);
  }
}

const packagedProvenance = JSON.parse(
  extractFile(asarPath, NATIVE_BUILD_PROVENANCE_FILE).toString("utf8"),
);
assertNativeSourceProvenanceMatches(resolve("."), packagedProvenance);

for (const forbiddenProductRoute of [
  "local model runtimeAdapter",
  "OpenAICompatibleAdapter",
  "probeLlmfit",
  "apiKeyEnv",
  "allowedHosts",
]) {
  if (mainBundle.includes(forbiddenProductRoute)) {
    throw new Error(
      `El bundle incluye una ruta de inferencia externa retirada: ${forbiddenProductRoute}.`,
    );
  }
}

if (
  !mainBundle.includes("mycellios-pipeline")
  || !mainBundle.includes("mycellios_pipeline_endpoint_not_local")
) {
  throw new Error(
    "El bundle no contiene la frontera fail-closed de la pipeline nativa Mycellios.",
  );
}

const expectedUpdateFeed = "https://www.mycellios.com/updates/win32/x64/";
if (platform === "win32") {
  const trayIconPath = resolve(resourcesDirectory, "icons", "app-icon-v2.ico");
  const jobBrokerPath = resolve(
    resourcesDirectory,
    "windows-job-broker",
    "mycellios-job-broker.exe",
  );
  if (!existsSync(trayIconPath)) {
    throw new Error(`El paquete no contiene el icono de bandeja de Windows: ${trayIconPath}`);
  }
  if (!existsSync(jobBrokerPath)) {
    throw new Error(
      `El paquete no contiene el broker Job Object de Windows: ${jobBrokerPath}`,
    );
  }
  const jobBrokerProbe = spawnSync(jobBrokerPath, ["--probe"], {
    encoding: "utf8",
    shell: false,
    windowsHide: true,
    timeout: 5_000,
  });
  if (
    jobBrokerProbe.error
    || jobBrokerProbe.status !== 0
    || jobBrokerProbe.stdout.trim() !== "mycellios-windows-job-broker/1:ready"
  ) {
    throw jobBrokerProbe.error
      ?? new Error(
        jobBrokerProbe.stderr?.trim()
          || "El broker Job Object empaquetado no supera su canary.",
      );
  }
  if (!mainBundle.includes(expectedUpdateFeed)) {
    throw new Error(`El paquete no contiene el canal de actualización: ${expectedUpdateFeed}`);
  }
  if (!mainBundle.includes("quitAndInstall")) {
    throw new Error("El paquete no contiene la instalación de la actualización descargada.");
  }
}

const runtimeSpec = portableRuntimeSpec(platform, arch);
if (runtimeSpec.supported) {
  const wheelLock = readPortableRuntimeWheelLock(resolve("."), runtimeSpec);
  if (!existsSync(runtimeArchive)) {
    throw new Error(`El paquete ${platform}/${arch} no contiene ${runtimeArchive}.`);
  }
  if (!existsSync(packagedPythonSource)) {
    throw new Error(
      `El paquete ${platform}/${arch} no contiene el runtime Mycellios en ${packagedPythonSource}.`,
    );
  }
  assertNativePythonProductMatchesSource(
    resolve("python"),
    packagedPythonSource,
  );
  const archiveEntries = tar(["-tzf", runtimeArchive])
    .split(/\r?\n/)
    .map((entry) => entry.trim())
    .filter(Boolean);
  for (const entry of archiveEntries) {
    const portableEntry = entry.replaceAll("\\", "/");
    const normalized = posix.normalize(portableEntry.replace(/^\.\//, ""));
    if (
      portableEntry.startsWith("/") ||
      /^[A-Za-z]:/.test(portableEntry) ||
      normalized === ".." ||
      normalized.startsWith("../")
    ) {
      throw new Error(`El runtime contiene una ruta insegura: ${entry}.`);
    }
  }
  const manifestText = tarEntry(runtimeArchive, ["./runtime-manifest.json", "runtime-manifest.json"]);
  const manifest = JSON.parse(manifestText);
  if (
    manifest.schema !== PORTABLE_RUNTIME_SCHEMA ||
    manifest.platform !== platform ||
    manifest.arch !== arch ||
    manifest.pythonVersion !== runtimeSpec.pythonVersion ||
    manifest.pythonAbi !== "cp312" ||
    manifest.executable !== runtimeSpec.pythonExecutable ||
    !matchesPinnedPythonArtifact(manifest.pythonArtifact, runtimeSpec.pythonArtifact) ||
    manifest.wheelLock?.path !== wheelLock.path ||
    manifest.wheelLock?.sha256 !== wheelLock.sha256 ||
    Object.keys(manifest.wheelLock ?? {}).sort().join(",") !== "path,sha256" ||
    manifest.torchVersion !== runtimeSpec.torchVersion ||
    manifest.transformersVersion !== runtimeSpec.packageVersions.transformers ||
    manifest.accelerateVersion !== runtimeSpec.packageVersions.accelerate ||
    manifest.safetensorsVersion !== runtimeSpec.packageVersions.safetensors ||
    manifest.aiohttpVersion !== runtimeSpec.packageVersions.aiohttp ||
    manifest.sentencepieceVersion !== runtimeSpec.packageVersions.sentencepiece ||
    manifest.numpyVersion !== runtimeSpec.packageVersions.numpy ||
    manifest.backend !== "cpu" ||
    JSON.stringify(manifest.bundledAccelerators ?? []) !== JSON.stringify(runtimeSpec.bundledAccelerators)
  ) {
    throw new Error(
      `El manifiesto del runtime no coincide con ${platform}/${arch}: ${JSON.stringify(manifest)}.`,
    );
  }
  const provenance = JSON.parse(tarEntry(runtimeArchive, [
    `./${PORTABLE_PYTHON_PROVENANCE_FILE}`,
    PORTABLE_PYTHON_PROVENANCE_FILE,
  ]));
  if (
    provenance.schema !== PORTABLE_PYTHON_PROVENANCE_SCHEMA ||
    provenance.platform !== platform ||
    provenance.arch !== arch ||
    !matchesPinnedPythonArtifact(provenance.artifact, runtimeSpec.pythonArtifact)
  ) {
    throw new Error(
      `La procedencia del Python empaquetado no coincide con ${platform}/${arch}: ${JSON.stringify(provenance)}.`,
    );
  }
  const canaryVerification = spawnSync(process.execPath, [
    resolve(import.meta.dirname, "verify-portable-runtime-archive.mjs"),
    `--archive=${runtimeArchive}`,
    `--platform=${platform}`,
    `--arch=${arch}`,
    `--python-source=${packagedPythonSource}`,
  ], {
    encoding: "utf8",
    maxBuffer: 32 * 1024 * 1024,
    shell: false,
    windowsHide: true,
  });
  if (canaryVerification.error) throw canaryVerification.error;
  if (canaryVerification.status !== 0) {
    throw new Error(
      canaryVerification.stderr?.trim() ||
        `El canary del runtime instalado terminó con ${canaryVerification.status ?? "estado desconocido"}.`,
    );
  }
  process.stdout.write(canaryVerification.stdout);
}

const rendererHtml = extractFile(
  asarPath,
  join(".vite", "renderer", "main_window", "index.html"),
).toString("utf8");
if (!rendererHtml.includes('<div id="root"></div>') || !rendererHtml.includes('type="module"')) {
  throw new Error("El renderer empaquetado no contiene el arranque React esperado.");
}

console.log(`Paquete mycellios verificado: ${platform}/${arch}`);
console.log(`ASAR: ${asarPath}`);

function tar(args) {
  const result = spawnSync("tar", args, {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
    shell: false,
    windowsHide: true,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(result.stderr?.trim() || `tar exited with ${result.status ?? "no status"}.`);
  }
  return result.stdout;
}

function tarEntry(archive, candidates) {
  for (const candidate of candidates) {
    const result = spawnSync("tar", ["-xOf", archive, candidate], {
      encoding: "utf8",
      maxBuffer: 4 * 1024 * 1024,
      shell: false,
      windowsHide: true,
    });
    if (result.status === 0 && result.stdout.trim()) return result.stdout;
  }
  throw new Error(`El runtime ${archive} no contiene runtime-manifest.json.`);
}
