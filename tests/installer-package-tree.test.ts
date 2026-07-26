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
