import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  NATIVE_BUILD_PROVENANCE_SCHEMA,
  assertNativeSourceProvenanceMatches,
  buildNativeSourceProvenance,
  verifyNativeBuildProvenanceDocument,
} from "../scripts/native-build-provenance.mjs";

describe("native desktop source provenance", () => {
  it("seals every application build input deterministically", () => {
    const provenance = buildNativeSourceProvenance(resolve("."));
    expect(provenance.schema).toBe(NATIVE_BUILD_PROVENANCE_SCHEMA);
    expect(provenance.sourceId).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(provenance.files.map(({ path }) => path)).toEqual(
      expect.arrayContaining([
        ".gitattributes",
        "forge.config.ts",
        "package.json",
        "python/requirements-distribution.txt",
        "src/desktop/main.ts",
        "src/distribution/python-launcher.ts",
        "landing/src/main.tsx",
        "python/distributed_runtime/draft_model.py",
        ".github/workflows/desktop-build.yml",
      ]),
    );
    expect(
      provenance.files.some(({ path }) =>
        path.startsWith("landing-dist/")
        || path.startsWith("mobile-dist/")
        || path.startsWith("out/")
        || path.includes("/.vite/")
      ),
    ).toBe(false);
    expect(() =>
      assertNativeSourceProvenanceMatches(resolve("."), provenance),
    ).not.toThrow();
  });

  it("rejects a packaged provenance document with one stale source digest", () => {
    const provenance = buildNativeSourceProvenance(resolve("."));
    const stale = structuredClone(provenance);
    stale.files[0]!.sha256 = "0".repeat(64);
    expect(() =>
      assertNativeSourceProvenanceMatches(resolve("."), stale),
    ).toThrow("does not seal");
  });

  it("rejects duplicate paths and unknown fields before trusting sourceId", () => {
    const provenance = buildNativeSourceProvenance(resolve("."));
    const duplicate = structuredClone(provenance);
    duplicate.files.splice(1, 0, structuredClone(duplicate.files[0]!));
    expect(() =>
      verifyNativeBuildProvenanceDocument(duplicate),
    ).toThrow("unique and strictly sorted");

    const extended = {
      ...structuredClone(provenance),
      revision: "untrusted",
    };
    expect(() =>
      verifyNativeBuildProvenanceDocument(extended),
    ).toThrow("unexpected or missing fields");
  });
});
