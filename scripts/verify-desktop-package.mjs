import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { extractFile, listPackage } from "@electron/asar";

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

const expectedUpdateFeed = "https://mycellios.com/updates/win32/x64/";
if (platform === "win32") {
  if (!mainBundle.includes(expectedUpdateFeed)) {
    throw new Error(`El paquete no contiene el canal de actualización: ${expectedUpdateFeed}`);
  }
  if (!mainBundle.includes("quitAndInstall")) {
    throw new Error("El paquete no contiene la instalación de la actualización descargada.");
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
