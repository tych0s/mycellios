import { existsSync } from "node:fs";
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

if (!existsSync(asarPath)) {
  throw new Error(`No existe el paquete esperado: ${asarPath}`);
}

const entries = new Set(listPackage(asarPath).map((entry) => entry.replaceAll("\\", "/")));
const requiredEntries = [
  "/.vite/build/main.cjs",
  "/.vite/build/preload.cjs",
  "/.vite/renderer/main_window/index.html",
];

for (const entry of requiredEntries) {
  if (!entries.has(entry)) throw new Error(`Falta ${entry} dentro de ${asarPath}`);
}

const packagedMetadata = JSON.parse(extractFile(asarPath, "package.json").toString("utf8"));
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

const expectedUpdateFeed = "https://www.mycellios.com/updates/win32/x64/";
if (platform === "win32") {
  const trayIconPath = resolve(resourcesDirectory, "icons", "app-icon-v2.ico");
  if (!existsSync(trayIconPath)) {
    throw new Error(`El paquete no contiene el icono de bandeja de Windows: ${trayIconPath}`);
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
  if (!existsSync(runtimeArchive)) {
    throw new Error(`El paquete ${platform}/${arch} no contiene ${runtimeArchive}.`);
  }
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
