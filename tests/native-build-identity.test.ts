import {
  mkdtempSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  buildNativeSourceProvenance,
} from "../scripts/native-build-provenance.mjs";
import {
  readNativeBuildIdentity,
  readNativeRuntimeBuildMetadata,
} from "../src/core/native-build-identity.js";

const temporaryRoots: string[] = [];

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) {
    rmSync(root, { recursive: true, force: true });
  }
});

describe("native runtime build identity", () => {
  it("reads only a self-consistent packaged provenance document", () => {
    const provenance = buildNativeSourceProvenance(resolve("."));
    const path = temporaryProvenance(provenance);
    expect(readNativeBuildIdentity(path, provenance.version)).toEqual({
      schema: provenance.schema,
      version: provenance.version,
      sourceId: provenance.sourceId,
    });
  });

  it("rejects a forged source identity and a mismatched runtime version", () => {
    const provenance = buildNativeSourceProvenance(resolve("."));
    const forged = structuredClone(provenance);
    forged.sourceId = `sha256:${"0".repeat(64)}`;
    expect(() =>
      readNativeBuildIdentity(temporaryProvenance(forged)),
    ).toThrow("not self-consistent");

    expect(() =>
      readNativeBuildIdentity(
        temporaryProvenance(provenance),
        `${provenance.version}-other`,
      ),
    ).toThrow("does not match runtime");
  });

  it("rejects reordered or duplicate source evidence", () => {
    const provenance = buildNativeSourceProvenance(resolve("."));
    const duplicate = structuredClone(provenance);
    duplicate.files.splice(1, 0, structuredClone(duplicate.files[0]!));
    expect(() =>
      readNativeBuildIdentity(temporaryProvenance(duplicate)),
    ).toThrow("not unique and strictly sorted");
  });

  it("binds package version, exact revision and provenance to one explicit root", () => {
    const provenance = buildNativeSourceProvenance(resolve("."));
    const root = temporaryRuntimeRoot(provenance);
    writeFileSync(join(root, "REVISION"), `${"a".repeat(40)}\n`, "utf8");

    expect(readNativeRuntimeBuildMetadata(root)).toEqual({
      root: resolve(root),
      version: provenance.version,
      revision: "a".repeat(40),
      buildIdentity: {
        schema: provenance.schema,
        version: provenance.version,
        sourceId: provenance.sourceId,
      },
    });
  });

  it("allows an unsealed source checkout but fails closed on malformed release evidence", () => {
    const root = temporaryRuntimeRoot(null);
    expect(readNativeRuntimeBuildMetadata(root)).toMatchObject({
      version: "1.2.3",
      revision: null,
      buildIdentity: null,
    });

    writeFileSync(join(root, "REVISION"), "not-a-revision\n", "utf8");
    expect(() => readNativeRuntimeBuildMetadata(root)).toThrow("exact Git SHA");
  });
});

function temporaryProvenance(value: unknown): string {
  const root = mkdtempSync(join(tmpdir(), "mycellios-build-identity-"));
  temporaryRoots.push(root);
  const path = join(root, "provenance.json");
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  return path;
}

function temporaryRuntimeRoot(
  provenance: ReturnType<typeof buildNativeSourceProvenance> | null,
): string {
  const root = mkdtempSync(join(tmpdir(), "mycellios-runtime-identity-"));
  temporaryRoots.push(root);
  writeFileSync(
    join(root, "package.json"),
    `${JSON.stringify({ name: "fixture", version: provenance?.version ?? "1.2.3" })}\n`,
    "utf8",
  );
  if (provenance) {
    writeFileSync(
      join(root, "mycellios-native-build-provenance.json"),
      `${JSON.stringify(provenance, null, 2)}\n`,
      "utf8",
    );
  }
  return root;
}
