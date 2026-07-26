import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  chmodSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import {
  assertInstallerPackageTreeMatches,
  assertDebianControlArchiveSafe,
  assertDebianPackageMetadata,
  assertExactDmgInstallerPayload,
  assertExactDmgRootInventory,
  assertRpmMetadataEvidence,
  collectTreeEvidence,
  expectedDebianControlFields,
  portablePermissionMode,
} from "../scripts/verify-installer-package-tree.mjs";

const temporaryRoots: string[] = [];

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) {
    rmSync(root, { recursive: true, force: true });
  }
});

describe("installer package tree binding", () => {
  it("accepts two independently materialized byte-identical trees", async () => {
    const [expected, actual] = packageTreePair();
    await expect(
      assertInstallerPackageTreeMatches(expected, actual),
    ).resolves.toBeUndefined();
    const evidence = await collectTreeEvidence(actual);
    expect(evidence.get("resources/app.asar")).toMatchObject({
      kind: "file",
      bytes: 11,
      sha256: expect.stringMatching(/^[0-9a-f]{64}$/),
    });
  });

  // Abortar en la primera diferencia hacia que cada una costase una vuelta
  // entera de CI —construir instaladores de tres plataformas— para enterarse de
  // la siguiente. Y las diferencias de este arbol vienen en racimo: son
  // divergencias sistematicas entre como copia el empaquetador y como
  // reconstruye el workflow, no fallos independientes.
  it("reports every differing entry at once instead of stopping at the first", async () => {
    const [expected, actual] = packageTreePair();
    writeFileSync(join(actual, "resources", "app.asar"), "otros bytes", "utf8");
    writeFileSync(join(actual, "resources", "runtime.bin"), "distinto", "utf8");

    const error = await assertInstallerPackageTreeMatches(expected, actual)
      .then(() => null, (reason: unknown) => reason as Error);

    expect(error).toBeInstanceOf(Error);
    expect(error!.message).toContain("2 installer tree entries differ");
    expect(error!.message).toContain("resources/app.asar");
    expect(error!.message).toContain("resources/runtime.bin");
  });

  // Una sola diferencia tiene que seguir dando EXACTAMENTE el mensaje de antes:
  // hay logs de CI donde se busca esa cadena, y cambiarla los rompe.
  it("keeps the single-difference message unchanged", async () => {
    const [expected, actual] = packageTreePair();
    writeFileSync(join(actual, "resources", "app.asar"), "otros bytes", "utf8");

    const error = await assertInstallerPackageTreeMatches(expected, actual)
      .then(() => null, (reason: unknown) => reason as Error);

    expect(error).toBeInstanceOf(Error);
    expect(error!.message).toMatch(
      /^Installer tree entry differs at resources\/app\.asar: /,
    );
    expect(error!.message).not.toContain("entries differ");
  });

  it("rejects a stale byte, missing file, extra file or complete permission-mode mismatch", async () => {
    {
      const [expected, actual] = packageTreePair();
      writeFileSync(join(actual, "resources", "app.asar"), "stale bytes", "utf8");
      await expect(
        assertInstallerPackageTreeMatches(expected, actual),
      ).rejects.toThrow("entry differs");
    }
    {
      const [expected, actual] = packageTreePair();
      rmSync(join(actual, "resources", "runtime.bin"));
      await expect(
        assertInstallerPackageTreeMatches(expected, actual),
      ).rejects.toThrow("file set differs");
    }
    {
      const [expected, actual] = packageTreePair();
      writeFileSync(join(actual, "unexpected"), "x", "utf8");
      await expect(
        assertInstallerPackageTreeMatches(expected, actual),
      ).rejects.toThrow("file set differs");
    }
    if (process.platform !== "win32") {
      const [expected, actual] = packageTreePair();
      chmodSync(join(actual, "mycellios"), 0o4755);
      await expect(
        assertInstallerPackageTreeMatches(expected, actual),
      ).rejects.toThrow("entry differs");

      const [expectedDirectory, actualDirectory] = packageTreePair();
      chmodSync(join(actualDirectory, "resources"), 0o777);
      await expect(
        assertInstallerPackageTreeMatches(expectedDirectory, actualDirectory),
      ).rejects.toThrow("entry differs");

      const [expectedRoot, actualRoot] = packageTreePair();
      chmodSync(actualRoot, 0o755);
      chmodSync(expectedRoot, 0o700);
      await expect(
        assertInstallerPackageTreeMatches(expectedRoot, actualRoot),
      ).rejects.toThrow("root permissions differ");
    }
  });

  it("retains all POSIX permission and special bits portably", () => {
    expect(portablePermissionMode(0o104755, "linux")).toBe(0o4755);
    expect(portablePermissionMode(0o102777, "darwin")).toBe(0o2777);
    expect(portablePermissionMode(0o40777, "linux")).toBe(0o777);
    expect(portablePermissionMode(0o104755, "win32")).toBeNull();
    expect(() => portablePermissionMode(-1, "linux")).toThrow(
      "non-negative integer",
    );
  });

  it("accepts only the application and the exact /Applications alias in a DMG", () => {
    const exact = new Map([
      ["Applications", {
        kind: "symlink" as const,
        mode: 0o777,
        target: "/Applications",
      }],
      ["mycellios.app", { kind: "directory" as const, mode: 0o755 }],
    ]);
    expect(() =>
      assertExactDmgRootInventory(exact, "mycellios.app")
    ).not.toThrow();

    const withHiddenPayload = new Map(exact);
    withHiddenPayload.set(".background", {
      kind: "directory",
      mode: 0o755,
    });
    expect(() =>
      assertExactDmgRootInventory(withHiddenPayload, "mycellios.app")
    ).toThrow("extra: .background");

    const redirectedAlias = new Map(exact);
    redirectedAlias.set("Applications", {
      kind: "symlink",
      mode: 0o777,
      target: "/tmp/Applications",
    });
    expect(() =>
      assertExactDmgRootInventory(redirectedAlias, "mycellios.app")
    ).toThrow("exact /Applications symlink");
  });

  it("accepts only inert DEB control metadata and rejects maintainer scripts", async () => {
    const safe = temporaryDirectory("mycellios-deb-control-safe-");
    writeFileSync(join(safe, "control"), "Package: mycellios\n", "utf8");
    writeFileSync(join(safe, "md5sums"), "abc usr/bin/mycellios\n", "utf8");
    if (process.platform !== "win32") {
      chmodSync(join(safe, "control"), 0o644);
      chmodSync(join(safe, "md5sums"), 0o644);
    }
    await expect(assertDebianControlArchiveSafe(safe)).resolves.toBeUndefined();

    const scripted = temporaryDirectory("mycellios-deb-control-scripted-");
    writeFileSync(join(scripted, "control"), "Package: mycellios\n", "utf8");
    writeFileSync(join(scripted, "postinst"), "#!/bin/sh\nexit 0\n", "utf8");
    if (process.platform !== "win32") {
      chmodSync(join(scripted, "control"), 0o644);
      chmodSync(join(scripted, "postinst"), 0o755);
    }
    await expect(assertDebianControlArchiveSafe(scripted)).rejects.toThrow(
      "extra: postinst",
    );
  });

  it("binds canonical DEB fields and recalculated payload checksums", async () => {
    const sourceApp = temporaryDirectory("mycellios-deb-source-");
    const payload = temporaryDirectory("mycellios-deb-payload-");
    const control = temporaryDirectory("mycellios-deb-exact-control-");
    materializePackageTree(sourceApp);
    materializePackageTree(payload);
    const fields = await expectedDebianControlFields({
      sourceApp,
      expectedVersion: "1.2.3",
      expectedArch: "amd64",
    });
    writeFileSync(
      join(control, "control"),
      `${[...fields].map(([name, value]) => `${name}: ${value}`).join("\n")}\n`,
      "utf8",
    );
    writeFileSync(
      join(control, "md5sums"),
      canonicalMd5Sums(payload),
      "utf8",
    );
    if (process.platform !== "win32") {
      chmodSync(join(control, "control"), 0o644);
      chmodSync(join(control, "md5sums"), 0o644);
    }
    await expect(assertDebianPackageMetadata({
      controlRoot: control,
      payloadRoot: payload,
      sourceApp,
      expectedVersion: "1.2.3",
      expectedArch: "amd64",
    })).resolves.toBeUndefined();

    writeFileSync(
      join(control, "md5sums"),
      canonicalMd5Sums(payload).replace(
        /^[0-9a-f]{32}/u,
        "0".repeat(32),
      ),
      "utf8",
    );
    await expect(assertDebianPackageMetadata({
      controlRoot: control,
      payloadRoot: payload,
      sourceApp,
      expectedVersion: "1.2.3",
      expectedArch: "amd64",
    })).rejects.toThrow("digest differs");
  });

  it("rejects RPM capabilities, flags, owners and header drift", () => {
    const digest = "a".repeat(64);
    const expected = new Map([
      ["/usr/lib/mycellios/mycellios", {
        kind: "file" as const,
        mode: 0o755,
        bytes: 10,
        sha256: digest,
      }],
    ]);
    const exactRow = [
      "/usr/lib/mycellios/mycellios",
      "10",
      "0100755",
      "root",
      "root",
      "(none)",
      "0",
      "(none)",
      digest,
    ].join("\t");
    const exact = {
      headerOutput: "mycellios\t1.2.3\t1\tx86_64\t8\n",
      fileOutput: `${exactRow}\n`,
      expectedEntries: expected,
      expectedVersion: "1.2.3",
      expectedArch: "x86_64",
    };
    expect(() => assertRpmMetadataEvidence(exact)).not.toThrow();
    expect(() => assertRpmMetadataEvidence({
      ...exact,
      fileOutput: `${exactRow.replace("(none)\t0", "cap_sys_admin=ep\t0")}\n`,
    })).toThrow("privileged metadata");
    expect(() => assertRpmMetadataEvidence({
      ...exact,
      fileOutput: `${exactRow.replace("\t0\t(none)", "\t64\t(none)")}\n`,
    })).toThrow("privileged metadata");
    expect(() => assertRpmMetadataEvidence({
      ...exact,
      headerOutput: "mycellios\t9.9.9\t1\tx86_64\t8\n",
    })).toThrow("identity differs");

    // `%doc` (2) es un marcador SEMANTICO del `%files`, no un permiso. El
    // fichero de copyright que genera el propio empaquetador lo lleva —CI
    // fallaba con `flags=2` en /usr/share/doc/mycellios/copyright—, asi que
    // rechazarlo tumbaba la verificacion del RPM entera por algo normal.
    expect(() => assertRpmMetadataEvidence({
      ...exact,
      fileOutput: `${exactRow.replace("\t0\t(none)", "\t2\t(none)")}\n`,
    })).not.toThrow();
    // Pero un flag informativo NO puede colar uno que si importa:
    // %doc|%ghost = 2|64 = 66 sigue rechazandose. Sin la mascara, permitir
    // `%doc` habria abierto la puerta a cualquier combinacion que lo incluyera.
    expect(() => assertRpmMetadataEvidence({
      ...exact,
      fileOutput: `${exactRow.replace("\t0\t(none)", "\t66\t(none)")}\n`,
    })).toThrow("privileged metadata");
  });

  it.runIf(process.platform !== "win32")(
    "binds the complete application tree inside an exact two-entry DMG root",
    async () => {
      const expectedContainer = temporaryDirectory(
        "mycellios-dmg-expected-",
      );
      const mounted = temporaryDirectory("mycellios-dmg-mounted-");
      const expectedApp = join(expectedContainer, "mycellios.app");
      const mountedApp = join(mounted, "mycellios.app");
      materializePackageTree(expectedApp);
      materializePackageTree(mountedApp);
      symlinkSync("/Applications", join(mounted, "Applications"));
      await expect(
        assertExactDmgInstallerPayload(expectedApp, mounted),
      ).resolves.toBeUndefined();
    },
  );

  it.runIf(process.platform !== "win32")(
    "rejects symlinks that escape the installed application root",
    async () => {
      const root = temporaryDirectory("mycellios-installer-symlink-");
      symlinkSync("../../outside", join(root, "escape"));
      await expect(collectTreeEvidence(root)).rejects.toThrow("escapes");
    },
  );

  // El punto de entrada del guion usa `await` de nivel superior, así que se
  // ejecuta donde esté escrito. Estuvo ARRIBA del fichero, antes de que los
  // `const` del módulo se inicializaran: las funciones se elevan, `const
  // EXPECTED_DEBIAN_CONTROL_VALUES` no. Resultado, «Cannot access ... before
  // initialization» y el instalador de Linux en rojo.
  //
  // Este test tiene que INVOCAR el guion como proceso: importarlo no basta,
  // porque el bloque de entrada comprueba `process.argv[1]` y no se ejecuta al
  // importar. Por eso ni el typecheck ni los tests de las funciones exportadas
  // lo vieron — sólo CI, y sólo en la rama de `deb`.
  it("runs its entry point after the module constants are initialised", () => {
    const controlRoot = temporaryDirectory("mycellios-installer-entry-");
    writeFileSync(join(controlRoot, "control"), "Package: mycellios\n");
    writeFileSync(join(controlRoot, "md5sums"), "");
    const script = fileURLToPath(
      new URL("../scripts/verify-installer-package-tree.mjs", import.meta.url),
    );
    const result = spawnSync(process.execPath, [
      script,
      `--deb-control-root=${controlRoot}`,
      `--deb-payload-root=${temporaryDirectory("mycellios-installer-payload-")}`,
      `--deb-source-app=${temporaryDirectory("mycellios-installer-app-")}`,
      "--expected-version=1.2.3",
      "--expected-arch=amd64",
    ], { encoding: "utf8" });

    const output = `${result.stdout ?? ""}${result.stderr ?? ""}`;
    expect(output).not.toContain("before initialization");
    // Y llega hasta la comparación real de campos, que es la prueba de que pasó
    // de la constante en vez de morir antes por otro motivo.
    expect(output).toContain("DEB control fields differ");
  });
});

function packageTreePair(): [string, string] {
  const expected = temporaryDirectory("mycellios-package-expected-");
  const actual = temporaryDirectory("mycellios-package-actual-");
  for (const root of [expected, actual]) materializePackageTree(root);
  return [expected, actual];
}

function materializePackageTree(root: string): void {
  mkdirSync(join(root, "resources"), { recursive: true });
  writeFileSync(join(root, "resources", "app.asar"), "native asar", "utf8");
  writeFileSync(join(root, "resources", "runtime.bin"), "runtime", "utf8");
  writeFileSync(join(root, "mycellios"), "executable", "utf8");
  if (process.platform !== "win32") chmodSync(join(root, "mycellios"), 0o755);
}

function canonicalMd5Sums(root: string): string {
  const paths = [
    "mycellios",
    "resources/app.asar",
    "resources/runtime.bin",
  ];
  return `${paths.map((path) => {
    const digest = createHash("md5")
      .update(readFileSync(join(root, ...path.split("/"))))
      .digest("hex");
    return `${digest}  ${path}`;
  }).join("\n")}\n`;
}

function temporaryDirectory(prefix: string): string {
  const root = mkdtempSync(join(tmpdir(), prefix));
  temporaryRoots.push(root);
  return root;
}
