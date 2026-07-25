import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import {
  lstat,
  readFile,
  readdir,
} from "node:fs/promises";
import { resolve } from "node:path";
import { TextDecoder } from "node:util";

const MAX_MANIFEST_BYTES = 16n * 1024n * 1024n;
const MAX_TAR_LIST_BYTES = 32 * 1024 * 1024;
const MAX_TAR_STDERR_BYTES = 1024 * 1024;
const TAR_TIMEOUT_MS = 5 * 60 * 1000;

const REQUIRED_NUPKG_ENTRIES = Object.freeze({
  appAsar: "lib/net45/resources/app.asar",
  runtimeArchive: "lib/net45/resources/distribution-runtime.tar.gz",
  pythonManifest:
    "lib/net45/resources/python/mycellios-native-python-manifest.json",
});

try {
  const root = parseRootArgument(process.argv.slice(2));
  await verifyWindowsRelease(root);
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`Verificación del release Windows fallida: ${message}`);
  process.exitCode = 1;
}

async function verifyWindowsRelease(workspaceRoot) {
  await requireDirectory(workspaceRoot, "La raíz indicada");

  const packageJsonPath = resolve(workspaceRoot, "package.json");
  const packageJsonFile = await requireRegularFile(
    packageJsonPath,
    "package.json",
  );
  if (packageJsonFile.size > MAX_MANIFEST_BYTES) {
    throw new Error(
      `package.json es demasiado grande para ser válido: ${packageJsonFile.size} bytes.`,
    );
  }

  const packageJsonBytes = await readFile(packageJsonPath);
  const packageMetadata = parseJsonObject(packageJsonBytes, packageJsonPath);
  const version = requireSafeVersion(packageMetadata.version);

  const makerDirectory = resolve(
    workspaceRoot,
    "out",
    "make",
    "squirrel.windows",
    "x64",
  );
  await requireDirectory(
    makerDirectory,
    "El directorio de artefactos Squirrel Windows x64",
  );

  const setupName = "mycellios-setup.exe";
  const nupkgName = `mycellios-${version}-full.nupkg`;
  const setupPath = resolve(makerDirectory, setupName);
  const nupkgPath = resolve(makerDirectory, nupkgName);
  const releasesPath = resolve(makerDirectory, "RELEASES");

  const artifactNames = await readdir(makerDirectory);
  const nupkgNames = artifactNames.filter((name) =>
    name.toLowerCase().endsWith(".nupkg"),
  );
  if (nupkgNames.length !== 1 || nupkgNames[0] !== nupkgName) {
    throw new Error(
      `Se esperaba únicamente ${nupkgName} en ${makerDirectory}; encontrados: ${
        nupkgNames.length > 0 ? nupkgNames.join(", ") : "ninguno"
      }.`,
    );
  }

  const [setupFile, nupkgFile, releasesFile] = await Promise.all([
    requireRegularFile(setupPath, setupName, { nonEmpty: true }),
    requireRegularFile(nupkgPath, nupkgName, { nonEmpty: true }),
    requireRegularFile(releasesPath, "RELEASES", { nonEmpty: true }),
  ]);

  if (releasesFile.size > MAX_MANIFEST_BYTES) {
    throw new Error(
      `RELEASES es demasiado grande para ser válido: ${releasesFile.size} bytes.`,
    );
  }

  const releaseRecord = parseReleases(
    await readFile(releasesPath),
    releasesPath,
  );
  if (releaseRecord.name !== nupkgName) {
    throw new Error(
      `RELEASES anuncia ${releaseRecord.name}, pero package.json ${version} exige ${nupkgName}.`,
    );
  }
  if (releaseRecord.size !== nupkgFile.size) {
    throw new Error(
      `RELEASES anuncia ${releaseRecord.size} bytes para ${nupkgName}, pero el archivo tiene ${nupkgFile.size} bytes.`,
    );
  }

  const [setupDigest, nupkgDigest] = await Promise.all([
    hashFile(setupPath, ["sha256"]),
    hashFile(nupkgPath, ["sha1", "sha256"]),
  ]);
  assertHashedExpectedSize(setupPath, setupFile, setupDigest);
  assertHashedExpectedSize(nupkgPath, nupkgFile, nupkgDigest);

  if (nupkgDigest.hashes.sha1.toUpperCase() !== releaseRecord.sha1) {
    throw new Error(
      `El SHA-1 real de ${nupkgName} es ${nupkgDigest.hashes.sha1.toUpperCase()}, pero RELEASES anuncia ${releaseRecord.sha1}.`,
    );
  }

  const archiveEntries = listNupkgSafely(nupkgPath);
  for (const requiredEntry of Object.values(REQUIRED_NUPKG_ENTRIES)) {
    const entry = archiveEntries.get(requiredEntry.toLowerCase());
    if (!entry || entry.path !== requiredEntry || entry.directory) {
      throw new Error(
        `El NUPKG no contiene el archivo exacto requerido ${requiredEntry}.`,
      );
    }
  }

  const currentResources = resolve(
    workspaceRoot,
    "out",
    "mycellios-win32-x64",
    "resources",
  );
  const currentAppAsarPath = resolve(currentResources, "app.asar");
  const currentRuntimePath = resolve(
    currentResources,
    "distribution-runtime.tar.gz",
  );
  const currentManifestPath = resolve(
    currentResources,
    "python",
    "mycellios-native-python-manifest.json",
  );
  const buildManifestPath = resolve(
    workspaceRoot,
    "build",
    "python",
    "mycellios-native-python-manifest.json",
  );

  const [
    currentAppAsarFile,
    currentRuntimeFile,
    currentManifestFile,
    buildManifestFile,
  ] = await Promise.all([
    requireRegularFile(currentAppAsarPath, "app.asar actual", {
      nonEmpty: true,
    }),
    requireRegularFile(currentRuntimePath, "runtime distribuido actual", {
      nonEmpty: true,
    }),
    requireRegularFile(currentManifestPath, "manifiesto Python empaquetado", {
      nonEmpty: true,
    }),
    requireRegularFile(buildManifestPath, "manifiesto Python de build", {
      nonEmpty: true,
    }),
  ]);

  for (const [label, file] of [
    ["manifiesto Python empaquetado", currentManifestFile],
    ["manifiesto Python de build", buildManifestFile],
  ]) {
    if (file.size > MAX_MANIFEST_BYTES) {
      throw new Error(
        `El ${label} es demasiado grande para ser válido: ${file.size} bytes.`,
      );
    }
  }

  const [buildManifestBytes, currentManifestBytes, nupkgManifest] =
    await Promise.all([
      readFile(buildManifestPath),
      readFile(currentManifestPath),
      readTarEntry(nupkgPath, REQUIRED_NUPKG_ENTRIES.pythonManifest, {
        captureLimit: Number(MAX_MANIFEST_BYTES),
      }),
    ]);

  parseJsonObject(buildManifestBytes, buildManifestPath);
  parseJsonObject(currentManifestBytes, currentManifestPath);
  parseJsonObject(
    nupkgManifest.buffer,
    `${nupkgName}:${REQUIRED_NUPKG_ENTRIES.pythonManifest}`,
  );

  assertBufferMatches(
    "El manifiesto Python incluido en el NUPKG",
    nupkgManifest.buffer,
    "el manifiesto Python de build",
    buildManifestBytes,
  );
  assertBufferMatches(
    "El manifiesto Python del paquete Windows actual",
    currentManifestBytes,
    "el manifiesto Python de build",
    buildManifestBytes,
  );

  const [currentAppAsarDigest, nupkgAppAsar] = await Promise.all([
    hashFile(currentAppAsarPath, ["sha256"]),
    hashTarEntry(nupkgPath, REQUIRED_NUPKG_ENTRIES.appAsar),
  ]);
  assertHashedExpectedSize(
    currentAppAsarPath,
    currentAppAsarFile,
    currentAppAsarDigest,
  );
  assertEntryMatchesFile(
    "app.asar",
    nupkgAppAsar,
    currentAppAsarDigest,
  );

  const [currentRuntimeDigest, nupkgRuntime] = await Promise.all([
    hashFile(currentRuntimePath, ["sha256"]),
    hashTarEntry(nupkgPath, REQUIRED_NUPKG_ENTRIES.runtimeArchive),
  ]);
  assertHashedExpectedSize(
    currentRuntimePath,
    currentRuntimeFile,
    currentRuntimeDigest,
  );
  assertEntryMatchesFile(
    "distribution-runtime.tar.gz",
    nupkgRuntime,
    currentRuntimeDigest,
  );

  console.log("Release Squirrel Windows verificado correctamente.");
  console.log(`Versión: ${version}`);
  console.log(
    `Setup: ${setupName} | ${setupFile.size} bytes | SHA-256 ${setupDigest.hashes.sha256}`,
  );
  console.log(
    `NUPKG: ${nupkgName} | ${nupkgFile.size} bytes | SHA-1 ${nupkgDigest.hashes.sha1.toUpperCase()} | SHA-256 ${nupkgDigest.hashes.sha256}`,
  );
  console.log(
    `RELEASES: nombre, tamaño y SHA-1 coinciden con ${nupkgName}.`,
  );
  console.log(
    "Contenido: app.asar, distribution-runtime.tar.gz y manifiesto Python coinciden byte a byte con el paquete Windows actual.",
  );
  console.log(
    "Authenticode: no se comprueba en este verificador; debe validarse por separado.",
  );
}

function parseRootArgument(args) {
  let requestedRoot;

  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    let value;

    if (argument === "--root") {
      if (index + 1 >= args.length) {
        throw new Error("Falta el valor de --root.");
      }
      value = args[index + 1];
      index += 1;
    } else if (argument.startsWith("--root=")) {
      value = argument.slice("--root=".length);
    } else {
      throw new Error(
        `Argumento desconocido ${argument}. Uso: node scripts/verify-windows-release-artifacts.mjs [--root <directorio>].`,
      );
    }

    if (requestedRoot !== undefined) {
      throw new Error("--root sólo puede indicarse una vez.");
    }
    if (typeof value !== "string" || value.trim() === "") {
      throw new Error("--root no puede estar vacío.");
    }
    requestedRoot = value;
  }

  return resolve(requestedRoot ?? resolve(import.meta.dirname, ".."));
}

async function requireDirectory(path, label) {
  let stats;
  try {
    stats = await lstat(path, { bigint: true });
  } catch (error) {
    if (error?.code === "ENOENT") {
      throw new Error(`${label} no existe: ${path}.`);
    }
    throw new Error(`${label} no se pudo inspeccionar (${path}): ${error.message}`);
  }
  if (!stats.isDirectory()) {
    throw new Error(`${label} no es un directorio real: ${path}.`);
  }
  return stats;
}

async function requireRegularFile(path, label, options = {}) {
  let stats;
  try {
    stats = await lstat(path, { bigint: true });
  } catch (error) {
    if (error?.code === "ENOENT") {
      throw new Error(`Falta ${label}: ${path}.`);
    }
    throw new Error(`No se pudo inspeccionar ${label} (${path}): ${error.message}`);
  }
  if (!stats.isFile()) {
    throw new Error(`${label} no es un archivo regular: ${path}.`);
  }
  if (options.nonEmpty && stats.size === 0n) {
    throw new Error(`${label} está vacío: ${path}.`);
  }
  return stats;
}

function requireSafeVersion(value) {
  if (
    typeof value !== "string" ||
    !/^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/.test(
      value,
    ) ||
    value.includes("..")
  ) {
    throw new Error(
      `package.json contiene una versión no válida para un artefacto Squirrel: ${JSON.stringify(value)}.`,
    );
  }
  return value;
}

function parseJsonObject(bytes, source) {
  let value;
  try {
    const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    value = JSON.parse(text);
  } catch (error) {
    throw new Error(`${source} no contiene JSON UTF-8 válido: ${error.message}`);
  }
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${source} debe contener un objeto JSON.`);
  }
  return value;
}

function parseReleases(bytes, releasesPath) {
  let text;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (error) {
    throw new Error(`${releasesPath} no es UTF-8 válido: ${error.message}`);
  }

  if (
    text.startsWith("\uFEFF") ||
    /[\u0000-\u0009\u000B\u000C\u000E-\u001F\u007F-\u009F]/u.test(text)
  ) {
    throw new Error(`${releasesPath} contiene bytes no permitidos.`);
  }
  if (text.endsWith("\r\n")) {
    text = text.slice(0, -2);
  } else if (text.endsWith("\n")) {
    text = text.slice(0, -1);
  }
  if (text.includes("\r") || text.includes("\n") || text.trim() !== text) {
    throw new Error(
      `${releasesPath} debe contener exactamente una línea Squirrel sin espacios sobrantes.`,
    );
  }

  const match = /^([0-9A-Fa-f]{40}) +([^ \t]+) +([0-9]+)$/.exec(text);
  if (!match) {
    throw new Error(
      `${releasesPath} no tiene el formato esperado: <SHA-1> <NUPKG> <tamaño>.`,
    );
  }

  const [, sha1, name, sizeText] = match;
  if (
    name.includes("/") ||
    name.includes("\\") ||
    name === "." ||
    name === ".."
  ) {
    throw new Error(`${releasesPath} anuncia un nombre inseguro: ${name}.`);
  }

  return {
    sha1: sha1.toUpperCase(),
    name,
    size: BigInt(sizeText),
  };
}

function listNupkgSafely(nupkgPath) {
  const result = spawnSync("tar", ["-tf", nupkgPath], {
    encoding: "utf8",
    maxBuffer: MAX_TAR_LIST_BYTES,
    shell: false,
    timeout: TAR_TIMEOUT_MS,
    windowsHide: true,
  });
  if (result.error) {
    throw new Error(
      `No se pudo listar ${nupkgPath} con tar: ${result.error.message}`,
    );
  }
  if (result.status !== 0) {
    throw new Error(
      `tar no pudo listar ${nupkgPath}: ${
        result.stderr?.trim() || `salida ${result.status ?? "desconocida"}`
      }`,
    );
  }
  if (!result.stdout) {
    throw new Error(`tar devolvió una lista vacía para ${nupkgPath}.`);
  }

  const lines = result.stdout.split("\n");
  if (lines.at(-1) === "") lines.pop();
  const entries = new Map();

  for (let rawEntry of lines) {
    if (rawEntry.endsWith("\r")) rawEntry = rawEntry.slice(0, -1);
    const entry = validateArchivePath(rawEntry);
    const key = entry.path.toLowerCase();
    if (entries.has(key)) {
      throw new Error(
        `El NUPKG contiene rutas duplicadas o incompatibles con Windows: ${entries.get(key).raw} y ${rawEntry}.`,
      );
    }
    entries.set(key, { ...entry, raw: rawEntry });
  }

  if (entries.size === 0) {
    throw new Error(`El NUPKG ${nupkgPath} no contiene entradas.`);
  }
  return entries;
}

function validateArchivePath(rawEntry) {
  if (
    typeof rawEntry !== "string" ||
    rawEntry.length === 0 ||
    /[\u0000-\u001F\u007F-\u009F\uFFFD]/u.test(rawEntry)
  ) {
    throw new Error("El NUPKG contiene una ruta vacía o con caracteres de control.");
  }
  if (rawEntry.includes("\\")) {
    throw new Error(
      `El NUPKG contiene una ruta con separadores Windows no permitidos: ${rawEntry}.`,
    );
  }

  const directory = rawEntry.endsWith("/");
  const path = directory ? rawEntry.slice(0, -1) : rawEntry;
  if (
    path.length === 0 ||
    path.startsWith("/") ||
    /^[A-Za-z]:/.test(path) ||
    path.includes("//")
  ) {
    throw new Error(`El NUPKG contiene una ruta absoluta o inválida: ${rawEntry}.`);
  }

  for (const segment of path.split("/")) {
    const windowsStem = segment.split(".", 1)[0].toUpperCase();
    if (
      segment === "." ||
      segment === ".." ||
      segment.endsWith(".") ||
      segment.endsWith(" ") ||
      /[<>:"|?*]/u.test(segment) ||
      /^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$/u.test(windowsStem)
    ) {
      throw new Error(`El NUPKG contiene una ruta insegura: ${rawEntry}.`);
    }
  }

  return { path, directory };
}

function hashFile(path, algorithms) {
  return new Promise((resolvePromise, rejectPromise) => {
    const hashes = Object.fromEntries(
      algorithms.map((algorithm) => [algorithm, createHash(algorithm)]),
    );
    let bytes = 0n;
    const input = createReadStream(path);

    input.on("data", (chunk) => {
      bytes += BigInt(chunk.length);
      for (const hash of Object.values(hashes)) hash.update(chunk);
    });
    input.once("error", (error) => {
      rejectPromise(new Error(`No se pudo leer ${path}: ${error.message}`));
    });
    input.once("end", () => {
      resolvePromise({
        bytes,
        hashes: Object.fromEntries(
          Object.entries(hashes).map(([algorithm, hash]) => [
            algorithm,
            hash.digest("hex"),
          ]),
        ),
      });
    });
  });
}

function hashTarEntry(nupkgPath, entryName) {
  return streamTarEntry(nupkgPath, entryName);
}

function readTarEntry(nupkgPath, entryName, options) {
  return streamTarEntry(nupkgPath, entryName, options);
}

function streamTarEntry(nupkgPath, entryName, options = {}) {
  return new Promise((resolvePromise, rejectPromise) => {
    const hash = createHash("sha256");
    const captured = [];
    const captureLimit = options.captureLimit ?? 0;
    let bytes = 0n;
    let stderr = "";
    let failure;
    let settled = false;

    const child = spawn("tar", ["-xOf", nupkgPath, entryName], {
      shell: false,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });

    const timer = setTimeout(() => {
      failure = new Error(
        `tar superó ${TAR_TIMEOUT_MS / 1000} segundos leyendo ${entryName}.`,
      );
      child.kill();
    }, TAR_TIMEOUT_MS);
    timer.unref();

    const rejectOnce = (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      rejectPromise(error);
    };

    child.stdout.on("data", (chunk) => {
      bytes += BigInt(chunk.length);
      hash.update(chunk);
      if (captureLimit > 0) {
        if (bytes > BigInt(captureLimit)) {
          failure = new Error(
            `${entryName} supera el límite seguro de ${captureLimit} bytes.`,
          );
          child.kill();
          return;
        }
        captured.push(chunk);
      }
    });

    child.stderr.on("data", (chunk) => {
      if (Buffer.byteLength(stderr) < MAX_TAR_STDERR_BYTES) {
        stderr += chunk.toString("utf8");
      } else if (!failure) {
        failure = new Error(
          `tar produjo demasiada salida de error leyendo ${entryName}.`,
        );
        child.kill();
      }
    });

    child.once("error", (error) => {
      rejectOnce(
        new Error(`No se pudo ejecutar tar para ${entryName}: ${error.message}`),
      );
    });

    child.once("close", (code, signal) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (failure) {
        rejectPromise(failure);
        return;
      }
      if (code !== 0) {
        rejectPromise(
          new Error(
            `tar no pudo leer ${entryName} de ${nupkgPath}: ${
              stderr.trim() ||
              `salida ${code ?? "desconocida"}${signal ? `, señal ${signal}` : ""}`
            }`,
          ),
        );
        return;
      }
      if (bytes === 0n) {
        rejectPromise(
          new Error(`La entrada requerida ${entryName} está vacía en el NUPKG.`),
        );
        return;
      }
      resolvePromise({
        bytes,
        sha256: hash.digest("hex"),
        buffer: captureLimit > 0 ? Buffer.concat(captured) : undefined,
      });
    });
  });
}

function assertHashedExpectedSize(path, stats, digest) {
  if (digest.bytes !== stats.size) {
    throw new Error(
      `${path} cambió mientras se verificaba: se esperaban ${stats.size} bytes y se leyeron ${digest.bytes}.`,
    );
  }
}

function assertBufferMatches(leftLabel, left, rightLabel, right) {
  if (left.equals(right)) return;
  throw new Error(
    `${leftLabel} no coincide byte a byte con ${rightLabel} (SHA-256 ${sha256(left)} frente a ${sha256(right)}). El release está obsoleto y debe reconstruirse.`,
  );
}

function assertEntryMatchesFile(label, nupkgEntry, currentFile) {
  if (
    nupkgEntry.bytes === currentFile.bytes &&
    nupkgEntry.sha256 === currentFile.hashes.sha256
  ) {
    return;
  }
  throw new Error(
    `${label} dentro del NUPKG no coincide con el paquete Windows actual: NUPKG ${nupkgEntry.bytes} bytes/SHA-256 ${nupkgEntry.sha256}, paquete actual ${currentFile.bytes} bytes/SHA-256 ${currentFile.hashes.sha256}. El release está obsoleto y debe reconstruirse.`,
  );
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
