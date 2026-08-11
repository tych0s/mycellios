import {
  chmodSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { gunzipSync, gzipSync } from "node:zlib";
import { afterEach, describe, expect, it } from "vitest";

import { canonicalEvidenceJson } from "../src/core/json.js";
import {
  buildComponentFilesPackage,
  computeFilesManifestSha256,
  extractComponentFilesPackage,
  inspectComponentFilesPackage,
  verifyComponentFilesPackage,
  type ComponentFilesPackageDocument,
} from "../src/update/component-files.js";

const roots: string[] = [];

afterEach(() => {
  for (const root of roots.splice(0)) {
    rmSync(root, { recursive: true, force: true });
  }
});

describe("json-gzip-v1 component files", () => {
  it("builds a deterministic canonical package and logical files manifest", () => {
    const source = fixtureSource();

    const first = buildComponentFilesPackage(source, {
      executablePaths: ["bin/run"],
    });
    const second = buildComponentFilesPackage(source, {
      executablePaths: ["bin/run"],
    });

    expect(first.packageBytes.equals(second.packageBytes)).toBe(true);
    expect(first.artifactSha256).toBe(second.artifactSha256);
    expect(first.filesManifestSha256).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(first.files).toEqual([
      expect.objectContaining({ path: "README.txt", mode: "0644" }),
      expect.objectContaining({ path: "bin/run", mode: "0755" }),
      expect.objectContaining({ path: "nested/empty.dat", mode: "0644" }),
    ]);
    expect(
      computeFilesManifestSha256([...first.files].reverse()),
    ).toBe(first.filesManifestSha256);

    const inspected = inspectComponentFilesPackage(first.packageBytes);
    expect(inspected).toMatchObject({
      format: "json-gzip-v1",
      artifactSha256: first.artifactSha256,
      filesManifestSha256: first.filesManifestSha256,
      fileCount: 3,
      totalFileBytes: 20,
      files: first.files,
    });
  });

  it("verifies the signed-manifest identity and extracts only to new staging", () => {
    const root = temporaryRoot();
    const source = fixtureSource(root);
    const built = buildComponentFilesPackage(source, {
      executablePaths: ["bin/run"],
    });
    const staging = join(root, "staging");

    const extracted = extractComponentFilesPackage(
      built.packageBytes,
      staging,
      { expectedFilesManifestSha256: built.filesManifestSha256 },
    );

    expect(extracted.stagingDirectory).toBe(staging);
    expect(readFileSync(join(staging, "README.txt"), "utf8")).toBe("component\n");
    expect(readFileSync(join(staging, "bin", "run"), "utf8")).toBe("#!/bin/sh\n");
    expect(readFileSync(join(staging, "nested", "empty.dat"))).toHaveLength(0);
    if (process.platform !== "win32") {
      expect(lstatSync(join(staging, "bin", "run")).mode & 0o777).toBe(0o755);
      expect(lstatSync(join(staging, "README.txt")).mode & 0o777).toBe(0o644);
    }

    expect(() =>
      extractComponentFilesPackage(built.packageBytes, staging),
    ).toThrow("component_files_staging_directory_already_exists");
    expect(() =>
      verifyComponentFilesPackage(built.packageBytes, {
        expectedFilesManifestSha256: `sha256:${"0".repeat(64)}`,
      }),
    ).toThrow("component_files_expected_manifest_sha256_mismatch");
  });

  it.each([
    "/absolute",
    "../escape",
    "child/../escape",
    "ambiguous\\path",
    "C:/drive",
    "file:stream",
    "NUL.txt",
    "trailing.",
  ])("rejects unsafe portable path %s", (unsafePath) => {
    const built = buildComponentFilesPackage(fixtureSource());
    const document = decode(built.packageBytes);
    document.files[0]!.path = unsafePath;

    expect(() => inspectComponentFilesPackage(encode(document))).toThrow(
      "component_files_path_is_unsafe",
    );
  });

  it("rejects duplicate, case-ambiguous and file-prefix-conflicting paths", () => {
    const built = buildComponentFilesPackage(fixtureSource());
    const duplicate = decode(built.packageBytes);
    duplicate.files = [duplicate.files[0]!, { ...duplicate.files[0]! }];
    expect(() => inspectComponentFilesPackage(encode(duplicate))).toThrow();

    const ambiguous = decode(built.packageBytes);
    ambiguous.files = [
      ambiguous.files[0]!,
      { ...ambiguous.files[0]!, path: ambiguous.files[0]!.path.toLowerCase() },
    ].sort((left, right) => left.path < right.path ? -1 : 1);
    expect(() => inspectComponentFilesPackage(encode(ambiguous))).toThrow(
      "component_files_path_is_ambiguous",
    );

    const conflict = decode(built.packageBytes);
    conflict.files = [
      conflict.files[0]!,
      { ...conflict.files[0]!, path: `${conflict.files[0]!.path}/child` },
    ];
    expect(() => inspectComponentFilesPackage(encode(conflict))).toThrow(
      "component_files_path_conflicts_with_file",
    );

    const foldedConflict = decode(built.packageBytes);
    foldedConflict.files = [
      { ...foldedConflict.files[0]!, path: "Directory" },
      { ...foldedConflict.files[0]!, path: "directory/child" },
    ];
    expect(() => inspectComponentFilesPackage(encode(foldedConflict))).toThrow(
      "component_files_path_is_ambiguous",
    );
  });

  it("rejects wrong byte counts, hashes, base64 and unknown entry types", () => {
    const built = buildComponentFilesPackage(fixtureSource());

    const wrongBytes = decode(built.packageBytes);
    wrongBytes.files[0]!.bytes += 1;
    expect(() => inspectComponentFilesPackage(encode(wrongBytes))).toThrow(
      "component_files_byte_count_mismatch",
    );

    const wrongHash = decode(built.packageBytes);
    wrongHash.files[0]!.sha256 = `sha256:${"0".repeat(64)}`;
    expect(() => inspectComponentFilesPackage(encode(wrongHash))).toThrow(
      "component_files_sha256_mismatch",
    );

    const wrongBase64 = decode(built.packageBytes);
    wrongBase64.files[0]!.base64 = `${wrongBase64.files[0]!.base64.slice(0, -1)}!`;
    expect(() => inspectComponentFilesPackage(encode(wrongBase64))).toThrow(
      "component_files_base64_is_invalid",
    );

    const unknownType = decode(built.packageBytes) as unknown as {
      files: Array<Record<string, unknown>>;
    };
    unknownType.files[0]!.type = "symlink";
    expect(() => inspectComponentFilesPackage(encode(unknownType))).toThrow(
      "component_files_entry_shape_is_invalid",
    );
  });

  it("enforces file, individual, total, compressed and document limits", () => {
    const built = buildComponentFilesPackage(fixtureSource());

    expect(() =>
      inspectComponentFilesPackage(built.packageBytes, { maxFiles: 2 }),
    ).toThrow("component_files_file_limit_exceeded");
    expect(() =>
      inspectComponentFilesPackage(built.packageBytes, { maxFileBytes: 5 }),
    ).toThrow("component_files_individual_size_limit_exceeded");
    expect(() =>
      inspectComponentFilesPackage(built.packageBytes, {
        maxTotalFileBytes: 16,
      }),
    ).toThrow("component_files_total_size_limit_exceeded");
    expect(() =>
      inspectComponentFilesPackage(built.packageBytes, {
        maxCompressedBytes: built.packageBytes.length - 1,
      }),
    ).toThrow("component_files_compressed_size_limit_exceeded");
    expect(() =>
      inspectComponentFilesPackage(built.packageBytes, {
        maxDocumentBytes: gunzipSync(built.packageBytes).length - 1,
      }),
    ).toThrow("component_files_package_decompression_failed");
  });

  it("rejects non-canonical JSON and never creates staging for invalid input", () => {
    const root = temporaryRoot();
    const built = buildComponentFilesPackage(fixtureSource(root));
    const document = decode(built.packageBytes);
    const nonCanonical = gzipSync(
      Buffer.from(JSON.stringify(document, null, 2), "utf8"),
    );
    const staging = join(root, "must-not-exist");

    expect(() =>
      extractComponentFilesPackage(nonCanonical, staging),
    ).toThrow("component_files_package_json_is_not_canonical");
    expect(() => lstatSync(staging)).toThrow();
  });

  it("rejects symbolic links while building instead of following them", () => {
    const root = temporaryRoot();
    const source = join(root, "source-with-link");
    const real = join(root, "real");
    mkdirSync(source);
    mkdirSync(real);
    writeFileSync(join(real, "payload.txt"), "outside");
    symlinkSync(real, join(source, "linked"), "junction");

    expect(() => buildComponentFilesPackage(source)).toThrow(
      "component_files_symbolic_link_is_forbidden",
    );
  });
});

function fixtureSource(parent = temporaryRoot()): string {
  const source = join(parent, `source-${Math.random().toString(16).slice(2)}`);
  mkdirSync(join(source, "bin"), { recursive: true });
  mkdirSync(join(source, "nested"));
  writeFileSync(join(source, "README.txt"), "component\n");
  writeFileSync(join(source, "bin", "run"), "#!/bin/sh\n");
  writeFileSync(join(source, "nested", "empty.dat"), Buffer.alloc(0));
  if (process.platform !== "win32") {
    chmodSync(join(source, "bin", "run"), 0o755);
  }
  return source;
}

function temporaryRoot(): string {
  const root = mkdtempSync(join(tmpdir(), "mycellios-component-files-"));
  roots.push(root);
  return root;
}

function decode(bytes: Uint8Array): ComponentFilesPackageDocument {
  return JSON.parse(gunzipSync(bytes).toString("utf8")) as
    ComponentFilesPackageDocument;
}

function encode(value: unknown): Buffer {
  return gzipSync(Buffer.from(canonicalEvidenceJson(value), "utf8"), {
    level: 9,
  });
}
