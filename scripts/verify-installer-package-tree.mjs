import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { createReadStream, existsSync } from "node:fs";
import {
  lstat,
  readFile,
  readlink,
  readdir,
} from "node:fs/promises";
import { basename, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";


export async function assertExactDmgInstallerPayload(
  expectedAppRoot,
  mountedDmgRoot,
) {
  const expectedApp = resolve(expectedAppRoot);
  const dmgRoot = resolve(mountedDmgRoot);
  const appName = basename(expectedApp);
  if (!appName.endsWith(".app")) {
    throw new Error(`Expected macOS application must end in .app: ${expectedApp}.`);
  }
  if (expectedApp === dmgRoot) {
    throw new Error("Expected application and mounted DMG roots must differ.");
  }
  const rootStats = await lstat(dmgRoot);
  if (!rootStats.isDirectory() || rootStats.isSymbolicLink()) {
    throw new Error(`Mounted DMG root is not a real directory: ${dmgRoot}.`);
  }

  const rootEntries = new Map();
  const caseInsensitiveNames = new Map();
  const entries = await readdir(dmgRoot, { withFileTypes: true });
  entries.sort((left, right) =>
    left.name < right.name ? -1 : left.name > right.name ? 1 : 0
  );
  for (const entry of entries) {
    const caseKey = entry.name.toLowerCase();
    const collision = caseInsensitiveNames.get(caseKey);
    if (collision !== undefined) {
      throw new Error(
        `Mounted DMG has case-insensitive name collision: ${collision} and ${entry.name}.`,
      );
    }
    caseInsensitiveNames.set(caseKey, entry.name);
    const absolute = resolve(dmgRoot, entry.name);
    const stats = await lstat(absolute);
    if (stats.isDirectory() && !stats.isSymbolicLink()) {
      rootEntries.set(entry.name, {
        kind: "directory",
        mode: portablePermissionMode(stats.mode),
      });
    } else if (stats.isSymbolicLink()) {
      rootEntries.set(entry.name, {
        kind: "symlink",
        mode: portablePermissionMode(stats.mode),
        target: (await readlink(absolute)).replaceAll("\\", "/"),
      });
    } else if (stats.isFile()) {
      rootEntries.set(entry.name, {
        kind: "file",
        mode: portablePermissionMode(stats.mode),
      });
    } else {
      rootEntries.set(entry.name, { kind: "special" });
    }
  }
  assertExactDmgRootInventory(rootEntries, appName);
  await assertInstallerPackageTreeMatches(
    expectedApp,
    resolve(dmgRoot, appName),
  );
}

export function assertExactDmgRootInventory(rootEntries, appName) {
  if (
    typeof appName !== "string"
    || !appName.endsWith(".app")
    || appName.includes("/")
    || appName.includes("\\")
  ) {
    throw new Error(`Invalid expected DMG application name: ${appName}.`);
  }
  const expectedNames = ["Applications", appName].sort();
  const actualNames = [...rootEntries.keys()].sort();
  if (canonicalJson(actualNames) !== canonicalJson(expectedNames)) {
    const missing = expectedNames.filter((name) => !rootEntries.has(name));
    const extra = actualNames.filter((name) => !expectedNames.includes(name));
    throw new Error(
      `DMG root inventory differs; missing: ${missing.join(", ") || "none"}; `
        + `extra: ${extra.join(", ") || "none"}.`,
    );
  }
  const application = rootEntries.get(appName);
  if (application?.kind !== "directory") {
    throw new Error(`DMG application is not a real directory: ${appName}.`);
  }
  const applicationsAlias = rootEntries.get("Applications");
  if (
    applicationsAlias?.kind !== "symlink"
    || applicationsAlias.target !== "/Applications"
  ) {
    throw new Error(
      "DMG Applications entry must be the exact /Applications symlink.",
    );
  }
}

export async function assertDebianControlArchiveSafe(controlRootValue) {
  const controlRoot = resolve(controlRootValue);
  const entries = await collectTreeEvidence(controlRoot);
  const names = [...entries.keys()];
  const allowedNames = new Set(["control", "md5sums"]);
  const extra = names.filter((name) => !allowedNames.has(name));
  if (!entries.has("control") || extra.length > 0) {
    throw new Error(
      `DEB control inventory is not permitted; required: control; extra: ${
        extra.join(", ") || "none"
      }.`,
    );
  }
  for (const name of names) {
    const entry = entries.get(name);
    if (entry?.kind !== "file") {
      throw new Error(`DEB control entry must be a regular file: ${name}.`);
    }
    if (entry.mode !== null && entry.mode !== 0o644) {
      throw new Error(
        `DEB control entry has unexpected permissions at ${name}: ${
          entry.mode.toString(8)
        } instead of 644.`,
      );
    }
  }
}

const EXPECTED_DEBIAN_CONTROL_VALUES = Object.freeze({
  Package: "mycellios",
  Section: "utils",
  Priority: "optional",
  Depends: [
    "libgtk-3-0",
    "libnotify4",
    "libnss3",
    "xdg-utils",
    "libatspi2.0-0",
    "libdrm2",
    "libgbm1",
    "libxcb-dri3-0",
    "kde-cli-tools | kde-runtime | trash-cli | libglib2.0-bin | gvfs-bin",
  ].join(", "),
  Recommends: "pulseaudio | libasound2",
  Suggests: [
    "gir1.2-gnomekeyring-1.0",
    "libgnome-keyring0",
    "lsb-release",
  ].join(", "),
  Maintainer: "mycellios",
  Description:
    "Control and contribute compute to the mycellios distributed inference network.\n"
    + " mycellios: heterogeneous distributed LLM inference runtime and scheduler.",
});

export async function assertDebianPackageMetadata({
  controlRoot,
  payloadRoot,
  sourceApp,
  expectedVersion,
  expectedArch,
}) {
  await assertDebianControlArchiveSafe(controlRoot);
  assertExpectedPackageIdentity(expectedVersion, expectedArch);
  const controlEntries = await collectTreeEvidence(controlRoot);
  if (!controlEntries.has("md5sums")) {
    throw new Error("DEB control archive must contain canonical md5sums.");
  }
  const expectedFields = await expectedDebianControlFields({
    sourceApp,
    expectedVersion,
    expectedArch,
  });
  const controlText = await readFile(resolve(controlRoot, "control"), "utf8");
  const actualFields = parseDebianControl(controlText);
  assertExactStringMap("DEB control fields", expectedFields, actualFields);
  await assertDebianMd5Sums(
    resolve(controlRoot, "md5sums"),
    payloadRoot,
  );
}

export async function expectedDebianControlFields({
  sourceApp,
  expectedVersion,
  expectedArch,
}) {
  assertExpectedPackageIdentity(expectedVersion, expectedArch);
  const installedSize = await sourceApplicationInstalledSize(sourceApp);
  return new Map(Object.entries({
    Package: EXPECTED_DEBIAN_CONTROL_VALUES.Package,
    Version: expectedVersion,
    Section: EXPECTED_DEBIAN_CONTROL_VALUES.Section,
    Priority: EXPECTED_DEBIAN_CONTROL_VALUES.Priority,
    Architecture: expectedArch,
    Depends: EXPECTED_DEBIAN_CONTROL_VALUES.Depends,
    Recommends: EXPECTED_DEBIAN_CONTROL_VALUES.Recommends,
    Suggests: EXPECTED_DEBIAN_CONTROL_VALUES.Suggests,
    "Installed-Size": installedSize.toString(),
    Maintainer: EXPECTED_DEBIAN_CONTROL_VALUES.Maintainer,
    Description: EXPECTED_DEBIAN_CONTROL_VALUES.Description,
  }));
}

export function parseDebianControl(text) {
  if (
    typeof text !== "string"
    || text.includes("\0")
    || text.includes("\uFFFD")
    || text.includes("\r")
  ) {
    throw new Error("DEB control file is not canonical UTF-8 with LF endings.");
  }
  const fields = new Map();
  let currentName;
  const lines = text.endsWith("\n")
    ? text.slice(0, -1).split("\n")
    : text.split("\n");
  for (const line of lines) {
    if (/^[ \t]/u.test(line)) {
      if (currentName === undefined) {
        throw new Error("DEB control continuation has no field.");
      }
      fields.set(currentName, `${fields.get(currentName)}\n${line}`);
      continue;
    }
    const match = /^([A-Za-z0-9][A-Za-z0-9-]*):(?: (.*))?$/u.exec(line);
    if (!match) {
      throw new Error(`DEB control line is not canonical: ${line}.`);
    }
    if (fields.has(match[1])) {
      throw new Error(`DEB control field is duplicated: ${match[1]}.`);
    }
    currentName = match[1];
    fields.set(currentName, match[2] ?? "");
  }
  return fields;
}

export async function assertDebianMd5Sums(md5PathValue, payloadRootValue) {
  const md5Path = resolve(md5PathValue);
  const payloadRoot = resolve(payloadRootValue);
  const text = await readFile(md5Path, "utf8");
  if (text.includes("\0") || text.includes("\r") || text.includes("\uFFFD")) {
    throw new Error("DEB md5sums is not canonical UTF-8 with LF endings.");
  }
  const declared = new Map();
  const caseInsensitivePaths = new Map();
  const lines = text.endsWith("\n")
    ? text.slice(0, -1).split("\n")
    : text.split("\n");
  for (const line of lines) {
    const match = /^([0-9a-f]{32})  ([^\0\r\n]+)$/u.exec(line);
    if (!match) {
      throw new Error(`DEB md5sums line is not canonical: ${line}.`);
    }
    const path = match[2];
    if (
      path.startsWith("/")
      || path.startsWith("./")
      || path === ".."
      || path.startsWith("../")
      || path.includes("\\")
    ) {
      throw new Error(`DEB md5sums path is unsafe: ${path}.`);
    }
    if (declared.has(path)) {
      throw new Error(`DEB md5sums path is duplicated: ${path}.`);
    }
    const caseKey = path.toLowerCase();
    const collision = caseInsensitivePaths.get(caseKey);
    if (collision !== undefined) {
      throw new Error(
        `DEB md5sums has case-insensitive collision: ${collision} and ${path}.`,
      );
    }
    caseInsensitivePaths.set(caseKey, path);
    declared.set(path, match[1]);
  }

  const payloadEntries = await collectTreeEvidence(payloadRoot);
  const expectedFiles = [...payloadEntries.entries()]
    .filter(([, entry]) => entry.kind === "file")
    .map(([path]) => path)
    .sort();
  const declaredPaths = [...declared.keys()].sort();
  if (canonicalJson(expectedFiles) !== canonicalJson(declaredPaths)) {
    const missing = expectedFiles.filter((path) => !declared.has(path));
    const extra = declaredPaths.filter((path) => !payloadEntries.has(path));
    throw new Error(
      `DEB md5sums file set differs; missing: ${missing.join(", ") || "none"}; `
        + `extra: ${extra.join(", ") || "none"}.`,
    );
  }
  for (const path of expectedFiles) {
    const digest = await hashFile(resolve(payloadRoot, ...path.split("/")), "md5");
    if (declared.get(path) !== digest) {
      throw new Error(`DEB md5sums digest differs at ${path}.`);
    }
  }
}

export async function assertRpmPackageMetadata({
  rpmPackage,
  expectedRoot,
  expectedVersion,
  expectedArch,
}) {
  assertExpectedPackageIdentity(expectedVersion, expectedArch);
  const headerFormat = [
    "%{NAME}",
    "%{VERSION}",
    "%{RELEASE}",
    "%{ARCH}",
    "%{FILEDIGESTALGO}",
  ].join("\t") + "\n";
  const fileFormat = [
    "%{FILENAMES}",
    "%{FILESIZES}",
    "%{FILEMODES:octal}",
    "%{FILEUSERNAME}",
    "%{FILEGROUPNAME}",
    "%{FILECAPS}",
    "%{FILEFLAGS}",
    "%{FILELINKTOS}",
    "%{FILEDIGESTS}",
  ].join("\\t");
  const headerOutput = queryRpm(rpmPackage, headerFormat);
  const fileOutput = queryRpm(rpmPackage, `[${fileFormat}\\n]`);
  const expectedEntries = expectedRpmOwnedEntries(
    await collectTreeEvidence(expectedRoot),
  );
  assertRpmMetadataEvidence({
    headerOutput,
    fileOutput,
    expectedEntries,
    expectedVersion,
    expectedArch,
  });
}

export function assertRpmMetadataEvidence({
  headerOutput,
  fileOutput,
  expectedEntries,
  expectedVersion,
  expectedArch,
}) {
  const headerParts = headerOutput.trimEnd().split("\t");
  if (headerParts.length !== 5) {
    throw new Error("RPM identity query did not return five exact fields.");
  }
  const [name, version, release, arch, digestAlgorithm] = headerParts;
  if (
    name !== "mycellios"
    || version !== expectedVersion
    || release !== "1"
    || arch !== expectedArch
    || digestAlgorithm !== "8"
  ) {
    throw new Error(
      `RPM identity differs: ${canonicalJson({
        name,
        version,
        release,
        arch,
        digestAlgorithm,
      })}.`,
    );
  }

  const rows = new Map();
  const caseInsensitivePaths = new Map();
  for (const line of fileOutput.trimEnd().split("\n")) {
    if (!line) continue;
    const values = line.split("\t");
    if (values.length !== 9) {
      throw new Error(`RPM file metadata row is malformed: ${line}.`);
    }
    const [
      path,
      sizeText,
      modeText,
      user,
      group,
      capabilities,
      flagsText,
      linkTarget,
      digest,
    ] = values;
    if (!path.startsWith("/") || path.includes("\0") || path.includes("\\")) {
      throw new Error(`RPM metadata path is unsafe: ${path}.`);
    }
    if (rows.has(path)) {
      throw new Error(`RPM metadata path is duplicated: ${path}.`);
    }
    const caseKey = path.toLowerCase();
    const collision = caseInsensitivePaths.get(caseKey);
    if (collision !== undefined) {
      throw new Error(
        `RPM metadata has case-insensitive collision: ${collision} and ${path}.`,
      );
    }
    caseInsensitivePaths.set(caseKey, path);
    const size = parseExactDecimal(sizeText, `RPM file size at ${path}`);
    const mode = parseExactOctal(modeText, `RPM file mode at ${path}`);
    const flags = parseExactDecimal(flagsText, `RPM file flags at ${path}`);
    rows.set(path, {
      size,
      mode,
      user,
      group,
      capabilities,
      flags,
      linkTarget,
      digest,
    });
  }
  const expectedPaths = [...expectedEntries.keys()].sort();
  const actualPaths = [...rows.keys()].sort();
  if (canonicalJson(expectedPaths) !== canonicalJson(actualPaths)) {
    const missing = expectedPaths.filter((path) => !rows.has(path));
    const extra = actualPaths.filter((path) => !expectedEntries.has(path));
    throw new Error(
      `RPM owned path set differs; missing: ${missing.join(", ") || "none"}; `
        + `extra: ${extra.join(", ") || "none"}.`,
    );
  }
  for (const path of expectedPaths) {
    const actual = rows.get(path);
    const expected = expectedEntries.get(path);
    const kind = rpmModeKind(actual.mode);
    if (
      kind !== expected.kind
      || (actual.mode & 0o7777) !== expected.mode
    ) {
      throw new Error(
        `RPM type or permissions differ at ${path}: ${kind} ${
          (actual.mode & 0o7777).toString(8)
        } instead of ${expected.kind} ${expected.mode.toString(8)}.`,
      );
    }
    // El mensaje dice CUAL de las cuatro condiciones ha fallado y con que valor.
    // Sin eso, "privileged metadata is not permitted" no distingue un fichero
    // que se instala como un usuario que no es root —peligroso— de un simple
    // marcador `%doc` o `%license`, que en RPM es semantico y perfectamente
    // normal en /usr/share/doc. Son cosas muy distintas y merecen mirarse
    // distinto; con el mensaje mudo hacian falta vueltas de CI para saber cual.
    const offending = [
      actual.user !== "root" ? `user=${actual.user}` : null,
      actual.group !== "root" ? `group=${actual.group}` : null,
      actual.flags !== 0 ? `flags=${actual.flags}` : null,
      isEmptyRpmValue(actual.capabilities)
        ? null
        : `capabilities=${actual.capabilities}`,
    ].filter((entry) => entry !== null);
    if (offending.length > 0) {
      throw new Error(
        `RPM privileged metadata is not permitted at ${path}: ${
          offending.join(", ")
        }.`,
      );
    }
    if (expected.kind === "file") {
      if (
        actual.size !== expected.bytes
        || actual.digest !== expected.sha256
        || !isEmptyRpmValue(actual.linkTarget)
      ) {
        throw new Error(`RPM file metadata differs at ${path}.`);
      }
    } else if (expected.kind === "symlink") {
      if (
        actual.size !== Buffer.byteLength(expected.target)
        || actual.linkTarget !== expected.target
        || !isEmptyRpmValue(actual.digest)
      ) {
        throw new Error(`RPM symlink metadata differs at ${path}.`);
      }
    } else if (
      !isEmptyRpmValue(actual.linkTarget)
      || !isEmptyRpmValue(actual.digest)
    ) {
      throw new Error(`RPM directory metadata differs at ${path}.`);
    }
  }
}

export async function assertInstallerPackageTreeMatches(
  expectedRoot,
  actualRoot,
) {
  const expected = resolve(expectedRoot);
  const actual = resolve(actualRoot);
  if (expected === actual) {
    throw new Error("Expected and installed package trees must be different.");
  }
  const [expectedRootStats, actualRootStats] = await Promise.all([
    lstat(expected),
    lstat(actual),
  ]);
  const expectedRootMode = portablePermissionMode(expectedRootStats.mode);
  const actualRootMode = portablePermissionMode(actualRootStats.mode);
  if (expectedRootMode !== actualRootMode) {
    throw new Error(
      `Installer tree root permissions differ: ${
        actualRootMode?.toString(8) ?? "unavailable"
      } instead of ${expectedRootMode?.toString(8) ?? "unavailable"}.`,
    );
  }
  const [expectedEntries, actualEntries] = await Promise.all([
    collectTreeEvidence(expected),
    collectTreeEvidence(actual),
  ]);
  const expectedPaths = [...expectedEntries.keys()];
  const actualPaths = [...actualEntries.keys()];
  if (JSON.stringify(actualPaths) !== JSON.stringify(expectedPaths)) {
    const missing = expectedPaths.filter((path) => !actualEntries.has(path));
    const extra = actualPaths.filter((path) => !expectedEntries.has(path));
    throw new Error(
      `Installer tree file set differs; missing: ${
        missing.join(", ") || "none"
      }; extra: ${extra.join(", ") || "none"}.`,
    );
  }
  // Se acumulan TODAS las diferencias antes de fallar, en vez de abortar en la
  // primera. Con el aborto temprano, cada diferencia costaba una vuelta entera
  // de CI —construir los instaladores de tres plataformas— para enterarse de la
  // siguiente, y las diferencias de este arbol vienen en racimo: son
  // divergencias sistematicas entre como copia el empaquetador y como reconstruye
  // el workflow, no fallos independientes. Enterarse de una cada diez minutos
  // convierte un diagnostico de una tarde en uno de varios dias.
  //
  // No se relaja nada: sigue fallando si hay una sola diferencia, y el primer
  // mensaje es identico al de antes para no romper a quien lo busque en un log.
  const differences = [];
  for (const path of expectedPaths) {
    const left = expectedEntries.get(path);
    const right = actualEntries.get(path);
    if (canonicalJson(left) !== canonicalJson(right)) {
      differences.push(
        `Installer tree entry differs at ${path}: ${
          canonicalJson(right)
        } instead of ${canonicalJson(left)}.`,
      );
    }
  }
  if (differences.length > 0) {
    throw new Error(
      differences.length === 1
        ? differences[0]
        : `${differences.length} installer tree entries differ:\n  ${
          differences.join("\n  ")
        }`,
    );
  }
}

export async function collectTreeEvidence(rootValue) {
  const root = resolve(rootValue);
  const rootStats = await lstat(root);
  if (!rootStats.isDirectory() || rootStats.isSymbolicLink()) {
    throw new Error(`Package tree root is not a real directory: ${root}.`);
  }
  const evidence = new Map();
  const caseInsensitivePaths = new Map();
  await visit(root, root, evidence, caseInsensitivePaths);
  return new Map(
    [...evidence.entries()].sort(([left], [right]) =>
      left < right ? -1 : left > right ? 1 : 0
    ),
  );
}

async function visit(root, directory, evidence, caseInsensitivePaths) {
  const entries = await readdir(directory, { withFileTypes: true });
  entries.sort((left, right) =>
    left.name < right.name ? -1 : left.name > right.name ? 1 : 0
  );
  for (const entry of entries) {
    const absolute = resolve(directory, entry.name);
    const portable = portablePath(root, absolute);
    const caseKey = portable.toLowerCase();
    const collision = caseInsensitivePaths.get(caseKey);
    if (collision !== undefined) {
      throw new Error(
        `Package tree has case-insensitive path collision: ${collision} and ${portable}.`,
      );
    }
    caseInsensitivePaths.set(caseKey, portable);
    const stats = await lstat(absolute);
    if (stats.isDirectory() && !stats.isSymbolicLink()) {
      evidence.set(portable, {
        kind: "directory",
        mode: portablePermissionMode(stats.mode),
      });
      await visit(root, absolute, evidence, caseInsensitivePaths);
      continue;
    }
    if (stats.isFile() && !stats.isSymbolicLink()) {
      evidence.set(portable, {
        kind: "file",
        mode: portablePermissionMode(stats.mode),
        bytes: stats.size,
        sha256: await sha256File(absolute),
      });
      continue;
    }
    if (stats.isSymbolicLink()) {
      const target = await readlink(absolute);
      assertSafeRelativeSymlink(root, absolute, target);
      evidence.set(portable, {
        kind: "symlink",
        mode: portablePermissionMode(stats.mode),
        target: target.replaceAll("\\", "/"),
      });
      continue;
    }
    throw new Error(`Package tree contains a special file: ${absolute}.`);
  }
}

function assertSafeRelativeSymlink(root, absolute, target) {
  if (
    !target
    || target.includes("\0")
    || target.startsWith("/")
    || /^[A-Za-z]:/u.test(target)
  ) {
    throw new Error(`Package tree contains an unsafe symlink: ${absolute}.`);
  }
  const resolvedTarget = resolve(absolute, "..", target);
  if (
    resolvedTarget !== root
    && !resolvedTarget.startsWith(`${root}${sep}`)
  ) {
    throw new Error(`Package tree symlink escapes its root: ${absolute}.`);
  }
}

function portablePath(root, absolute) {
  const portable = relative(root, absolute).replaceAll("\\", "/");
  if (
    !portable
    || portable.startsWith("../")
    || portable === ".."
    || portable.startsWith("/")
    || /^[A-Za-z]:/u.test(portable)
  ) {
    throw new Error(`Package tree path escaped its root: ${absolute}.`);
  }
  return portable;
}

export function portablePermissionMode(mode, platform = process.platform) {
  if (!Number.isInteger(mode) || mode < 0) {
    throw new Error(`Filesystem mode must be a non-negative integer: ${mode}.`);
  }
  return platform === "win32" ? null : mode & 0o7777;
}

async function sourceApplicationInstalledSize(rootValue) {
  const root = resolve(rootValue);
  const seenInodes = new Set();
  let total = 0n;
  const visitSize = async (path) => {
    const stats = await lstat(path, { bigint: true });
    const inode = stats.ino.toString();
    if (seenInodes.has(inode)) return;
    seenInodes.add(inode);
    total += stats.size;
    if (!stats.isDirectory() || stats.isSymbolicLink()) return;
    const names = await readdir(path);
    names.sort();
    for (const name of names) {
      await visitSize(resolve(path, name));
    }
  };
  await visitSize(root);
  return (total + 1023n) / 1024n;
}

function assertExpectedPackageIdentity(version, arch) {
  if (
    typeof version !== "string"
    || !/^[0-9]+\.[0-9]+\.[0-9]+$/u.test(version)
  ) {
    throw new Error(`Expected installer version is not canonical: ${version}.`);
  }
  if (
    typeof arch !== "string"
    || !/^[A-Za-z0-9_.+-]+$/u.test(arch)
  ) {
    throw new Error(`Expected installer architecture is not canonical: ${arch}.`);
  }
}

function assertExactStringMap(label, expected, actual) {
  const expectedKeys = [...expected.keys()].sort();
  const actualKeys = [...actual.keys()].sort();
  if (canonicalJson(expectedKeys) !== canonicalJson(actualKeys)) {
    const missing = expectedKeys.filter((key) => !actual.has(key));
    const extra = actualKeys.filter((key) => !expected.has(key));
    throw new Error(
      `${label} differ; missing: ${missing.join(", ") || "none"}; `
        + `extra: ${extra.join(", ") || "none"}.`,
    );
  }
  for (const key of expectedKeys) {
    if (actual.get(key) !== expected.get(key)) {
      throw new Error(`${label} differ at ${key}.`);
    }
  }
}

function queryRpm(rpmPathValue, queryFormat) {
  const rpmPath = resolve(rpmPathValue);
  const result = spawnSync(
    "rpm",
    ["-qp", "--qf", queryFormat, rpmPath],
    {
      encoding: "utf8",
      maxBuffer: 128 * 1024 * 1024,
      shell: false,
      timeout: 60_000,
      windowsHide: true,
    },
  );
  if (
    result.error
    || result.status !== 0
    || result.signal !== null
    || typeof result.stdout !== "string"
  ) {
    throw new Error(
      `RPM metadata query failed: ${
        result.error?.message
          ?? result.stderr?.trim()
          ?? `status ${result.status}, signal ${result.signal}`
      }.`,
    );
  }
  return result.stdout;
}

function expectedRpmOwnedEntries(treeEntries) {
  const owned = new Map();
  for (const [path, entry] of treeEntries) {
    const isOwned =
      path === "usr/bin/mycellios"
      || path === "usr/lib/mycellios"
      || path.startsWith("usr/lib/mycellios/")
      || path === "usr/share/applications/mycellios.desktop"
      || path === "usr/share/doc/mycellios"
      || path.startsWith("usr/share/doc/mycellios/")
      || path === "usr/share/pixmaps/mycellios.png";
    if (!isOwned) continue;
    if (entry.mode === null) {
      throw new Error(`RPM expected mode is unavailable at ${path}.`);
    }
    owned.set(`/${path}`, entry);
  }
  for (const required of [
    "/usr/bin/mycellios",
    "/usr/lib/mycellios",
    "/usr/share/applications/mycellios.desktop",
    "/usr/share/doc/mycellios",
    "/usr/share/pixmaps/mycellios.png",
  ]) {
    if (!owned.has(required)) {
      throw new Error(`RPM expected payload is missing ${required}.`);
    }
  }
  return owned;
}

function parseExactDecimal(value, label) {
  if (!/^(?:0|[1-9][0-9]*)$/u.test(value)) {
    throw new Error(`${label} is not canonical decimal: ${value}.`);
  }
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) {
    throw new Error(`${label} exceeds safe integer range: ${value}.`);
  }
  return parsed;
}

function parseExactOctal(value, label) {
  if (!/^[0-7]+$/u.test(value)) {
    throw new Error(`${label} is not canonical octal: ${value}.`);
  }
  const parsed = Number.parseInt(value, 8);
  if (!Number.isSafeInteger(parsed)) {
    throw new Error(`${label} exceeds safe integer range: ${value}.`);
  }
  return parsed;
}

function rpmModeKind(mode) {
  const fileType = mode & 0o170000;
  if (fileType === 0o100000) return "file";
  if (fileType === 0o040000) return "directory";
  if (fileType === 0o120000) return "symlink";
  return "special";
}

function isEmptyRpmValue(value) {
  return value === "" || value === "(none)";
}

function sha256File(path) {
  return hashFile(path, "sha256");
}

function hashFile(path, algorithm) {
  return new Promise((resolvePromise, rejectPromise) => {
    const hash = createHash(algorithm);
    const stream = createReadStream(path);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.once("error", rejectPromise);
    stream.once("end", () => resolvePromise(hash.digest("hex")));
  });
}

function parseCommandLine(argumentsValue) {
  const supported = new Set([
    "actual",
    "deb-control-root",
    "deb-payload-root",
    "deb-source-app",
    "dmg-root",
    "expected",
    "expected-app",
    "expected-arch",
    "expected-version",
    "rpm-expected-root",
    "rpm-package",
  ]);
  const values = new Map();
  for (const argument of argumentsValue) {
    const match = /^--([a-z-]+)=(.+)$/u.exec(argument);
    if (!match || !supported.has(match[1])) {
      throw new Error(`Unsupported installer verification argument: ${argument}.`);
    }
    if (values.has(match[1])) {
      throw new Error(`Argument --${match[1]} may only be supplied once.`);
    }
    values.set(match[1], match[2]);
  }
  const pathFor = (name) => {
    const value = values.get(name);
    if (!value) return undefined;
    const path = resolve(value);
    if (!existsSync(path)) {
      throw new Error(`Package tree does not exist: ${path}.`);
    }
    return path;
  };
  const valueFor = (name) => {
    const value = values.get(name);
    if (!value) {
      throw new Error(`Argument --${name} must be non-empty.`);
    }
    return value;
  };
  if (values.has("expected") || values.has("actual")) {
    assertArgumentSet(values, ["actual", "expected"]);
    return {
      kind: "tree",
      actual: pathFor("actual"),
      expected: pathFor("expected"),
    };
  }
  if (values.has("dmg-root") || values.has("expected-app")) {
    assertArgumentSet(values, ["dmg-root", "expected-app"]);
    return {
      kind: "dmg",
      dmgRoot: pathFor("dmg-root"),
      expectedApp: pathFor("expected-app"),
    };
  }
  if (values.has("deb-control-root")) {
    assertArgumentSet(values, [
      "deb-control-root",
      "deb-payload-root",
      "deb-source-app",
      "expected-arch",
      "expected-version",
    ]);
    return {
      kind: "deb",
      controlRoot: pathFor("deb-control-root"),
      payloadRoot: pathFor("deb-payload-root"),
      sourceApp: pathFor("deb-source-app"),
      expectedArch: valueFor("expected-arch"),
      expectedVersion: valueFor("expected-version"),
    };
  }
  if (values.has("rpm-package") || values.has("rpm-expected-root")) {
    assertArgumentSet(values, [
      "expected-arch",
      "expected-version",
      "rpm-expected-root",
      "rpm-package",
    ]);
    return {
      kind: "rpm",
      rpmPackage: pathFor("rpm-package"),
      expectedRoot: pathFor("rpm-expected-root"),
      expectedArch: valueFor("expected-arch"),
      expectedVersion: valueFor("expected-version"),
    };
  }
  throw new Error("No installer verification mode was selected.");
}

function assertArgumentSet(values, requiredNames) {
  const required = new Set(requiredNames);
  const missing = requiredNames.filter((name) => !values.has(name));
  const extra = [...values.keys()].filter((name) => !required.has(name));
  if (missing.length > 0 || extra.length > 0) {
    throw new Error(
      `Installer verification arguments differ; missing: ${
        missing.map((name) => `--${name}`).join(", ") || "none"
      }; extra: ${
        extra.map((name) => `--${name}`).join(", ") || "none"
      }.`,
    );
  }
}

function canonicalJson(value) {
  if (Array.isArray(value)) {
    return `[${value.map((entry) => canonicalJson(entry)).join(",")}]`;
  }
  if (value !== null && typeof value === "object") {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

// PUNTO DE ENTRADA. Va AL FINAL a propósito, y no es cuestión de estilo.
//
// Este bloque usa `await` de nivel superior, así que se EJECUTA en el punto
// del fichero donde esté escrito. Arriba se ejecutaba antes de que los
// `const` del módulo estuviesen inicializados: las funciones se elevan, pero
// `const EXPECTED_DEBIAN_CONTROL_VALUES` no, y cualquier camino que lo tocara
// moría con «Cannot access ... before initialization». Sólo reventaba la rama
// de `deb`, así que las de `tree` y `dmg` lo tapaban.
//
// Al final del módulo, todo lo que el punto de entrada usa ya existe.
if (
  process.argv[1]
  && resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  try {
    const command = parseCommandLine(process.argv.slice(2));
    if (command.kind === "tree") {
      await assertInstallerPackageTreeMatches(
        command.expected,
        command.actual,
      );
      console.log(`Installer package tree verified: ${command.actual}`);
    } else if (command.kind === "dmg") {
      await assertExactDmgInstallerPayload(
        command.expectedApp,
        command.dmgRoot,
      );
      console.log(`Exact DMG payload verified: ${command.dmgRoot}`);
    } else if (command.kind === "deb") {
      await assertDebianPackageMetadata(command);
      console.log(`Exact DEB metadata verified: ${command.controlRoot}`);
    } else {
      await assertRpmPackageMetadata(command);
      console.log(`Exact RPM metadata verified: ${command.rpmPackage}`);
    }
  } catch (error) {
    console.error(
      `Installer package tree verification failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
    process.exitCode = 1;
  }
}
