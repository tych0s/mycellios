#!/usr/bin/env node
/**
 * Mantiene la version de escritorio POR DELANTE de todo lo ya publicado.
 *
 * POR QUE EXISTE. La version de release sale de `package.json` y de ningun otro
 * sitio: `desktop-build.yml` la lee, exige que `package-lock.json` coincida, y
 * crea el tag `v<version>` apuntando al commit exacto. Todo el job de
 * publicacion esta construido alrededor de la INMUTABILIDAD — se niega a reusar
 * un tag que apunte a otro commit y se niega a mutar los assets de una release
 * existente.
 *
 * Ese diseño es correcto, pero le faltaba una pieza: NADIE SUBIA LA VERSION.
 *
 * Lo que paso (26-07-2026). Las releases v0.2.20 a v0.2.51 se publicaron con un
 * proceso anterior que autoincrementaba el numero, mientras `package.json`
 * llevaba en 0.2.19 desde siempre — verificado: el `package.json` DENTRO de los
 * tags v0.2.20, v0.2.44 y v0.2.51 dice 0.2.19 en los tres. Cuando el control
 * estricto entro (8 horas despues de la ultima publicacion), el resultado fue un
 * ATASCO PERMANENTE: `package.json` pide el tag v0.2.19, ese tag ya existe
 * apuntando a otro commit, y el job muere con
 *
 *     Refusing to reuse v0.2.19: it targets c2e66f4, not <HEAD>
 *
 * en CADA push a main. Y el sintoma que lo delata no aparece hasta el final del
 * pipeline, despues de construir instaladores para tres plataformas.
 *
 * QUE HACE ESTE GUION.
 *
 *   --check   Falla si `package.json` no va estrictamente por delante de todos
 *             los tags `vX.Y.Z` existentes, e imprime el comando exacto que lo
 *             arregla. Corre al principio del pipeline, no al final.
 *
 *   --bump    Escribe la version siguiente en `package.json` y en los DOS sitios
 *             donde vive dentro de `package-lock.json`.
 *
 * LA COMPARACION ES NUMERICA, NO DE TEXTO. Es la trampa concreta de este caso:
 * en orden alfabetico "0.2.19" > "0.2.51" porque '1' < '5' se decide en el
 * segundo caracter. Comparar como cadenas habria dado el atasco por bueno.
 */

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const WORKSPACE = resolve(import.meta.dirname, "..");
const PACKAGE_JSON = resolve(WORKSPACE, "package.json");
const PACKAGE_LOCK = resolve(WORKSPACE, "package-lock.json");
const SEMVER = /^(\d+)\.(\d+)\.(\d+)$/u;

/** Convierte "0.2.51" en [0, 2, 51]. Lanza si no es canonica. */
export function parseVersion(value) {
  const match = SEMVER.exec(String(value ?? "").trim());
  if (!match) {
    throw new Error(`Version is not canonical MAJOR.MINOR.PATCH: ${value}.`);
  }
  return [Number(match[1]), Number(match[2]), Number(match[3])];
}

/** Negativo si a < b, cero si iguales, positivo si a > b. */
export function compareVersions(a, b) {
  const left = parseVersion(a);
  const right = parseVersion(b);
  for (let index = 0; index < 3; index += 1) {
    if (left[index] !== right[index]) return left[index] - right[index];
  }
  return 0;
}

/** La mayor de una lista de versiones, o null si viene vacia. */
export function highestVersion(versions) {
  let highest = null;
  for (const candidate of versions) {
    if (highest === null || compareVersions(candidate, highest) > 0) {
      highest = candidate;
    }
  }
  return highest;
}

/** Sube el parche: "0.2.51" -> "0.2.52". */
export function nextPatch(version) {
  const [major, minor, patch] = parseVersion(version);
  return `${major}.${minor}.${patch + 1}`;
}

/**
 * Versiones ya publicadas, leidas de los tags `vX.Y.Z` del repositorio.
 *
 * Se usan los TAGS y no la API de releases porque el tag es lo que hace fallar
 * al job de publicacion: un tag que existe apuntando a otro commit basta para
 * atascarlo, exista o no la release asociada.
 */
export function publishedVersions(cwd = WORKSPACE) {
  let output = "";
  try {
    output = execFileSync("git", ["tag", "--list", "v[0-9]*.[0-9]*.[0-9]*"], {
      cwd,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    });
  } catch {
    // Sin git, o sin tags: el llamante decide. No inventamos un historial.
    return null;
  }
  const versions = [];
  for (const line of output.split(/\r?\n/u)) {
    const name = line.trim();
    if (!name.startsWith("v")) continue;
    const candidate = name.slice(1);
    if (SEMVER.test(candidate)) versions.push(candidate);
  }
  return versions;
}

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

/** Escribe conservando el salto de linea final que usa npm. */
function writeJson(path, value) {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

export function currentVersion() {
  const declared = readJson(PACKAGE_JSON).version;
  const lock = readJson(PACKAGE_LOCK);
  const lockRoot = lock.version;
  const lockSelf = lock.packages?.[""]?.version;
  if (lockRoot !== declared || lockSelf !== declared) {
    throw new Error(
      `package-lock.json is out of step: root ${lockRoot}, packages[""] ${lockSelf}, `
        + `package.json ${declared}. Run: npm run version:bump`,
    );
  }
  return declared;
}

/** La siguiente version libre: por delante de lo declarado y de todo tag. */
export function resolveNextVersion(declared, published) {
  const ceiling = highestVersion([declared, ...(published ?? [])]);
  return nextPatch(ceiling);
}

function bump() {
  const declared = currentVersion();
  const next = resolveNextVersion(declared, publishedVersions());

  const pkg = readJson(PACKAGE_JSON);
  pkg.version = next;
  writeJson(PACKAGE_JSON, pkg);

  const lock = readJson(PACKAGE_LOCK);
  lock.version = next;
  if (lock.packages?.[""]) lock.packages[""].version = next;
  writeJson(PACKAGE_LOCK, lock);

  console.log(`Desktop release version ${declared} -> ${next}.`);
}

function check() {
  const declared = currentVersion();
  const published = publishedVersions();
  if (published === null) {
    console.log(
      `Desktop release version ${declared}; no git tags readable, nothing to compare.`,
    );
    return;
  }
  const taken = published.filter(
    (candidate) => compareVersions(candidate, declared) >= 0,
  );
  if (taken.length > 0) {
    const highest = highestVersion(taken);
    throw new Error(
      `Desktop release version ${declared} is not ahead of published tag v${highest}.\n`
        + `The publish job creates tag v${declared}, which already exists on another `
        + `commit, so it will refuse to reuse it and every push to main will fail at `
        + `the very end of the pipeline.\n`
        + `Fix it before merging:  npm run version:bump   `
        + `(would select ${resolveNextVersion(declared, published)})`,
    );
  }
  console.log(
    `Desktop release version ${declared} is ahead of all `
      + `${published.length} published tags.`,
  );
}

// Se ejecuta solo como guion; importarlo desde un test no dispara nada.
if (process.argv[1] && resolve(process.argv[1]) === resolve(import.meta.filename)) {
  const mode = process.argv[2] ?? "--check";
  try {
    if (mode === "--bump") {
      bump();
    } else if (mode === "--check") {
      check();
    } else {
      throw new Error(`Unknown mode: ${mode}. Use --check or --bump.`);
    }
  } catch (error) {
    // Sin volcado de pila: quien lea esto en un log de CI necesita el motivo y
    // el comando que lo arregla, no la ruta interna del guion.
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}
