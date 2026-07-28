import {
  mkdtempSync,
  readFileSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  prepareCoordinatorRuntimeIdentity,
} from "../scripts/prepare-coordinator-runtime-identity.mjs";
import {
  verifyNativeBuildProvenanceDocument,
} from "../scripts/native-build-provenance.mjs";

const temporaryRoots: string[] = [];

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) {
    rmSync(root, { recursive: true, force: true });
  }
});

describe("coordinator runtime identity", () => {
  it("writes an exact revision and source provenance for the deployed tree", () => {
    const runtime = mkdtempSync(join(tmpdir(), "mycellios-runtime-identity-"));
    temporaryRoots.push(runtime);
    const revision = "a".repeat(40);

    const identity = prepareCoordinatorRuntimeIdentity(
      resolve("."),
      runtime,
      revision,
    );
    const provenance = verifyNativeBuildProvenanceDocument(
      JSON.parse(
        readFileSync(
          join(runtime, "mycellios-native-build-provenance.json"),
          "utf8",
        ),
      ),
    );

    expect(readFileSync(join(runtime, "REVISION"), "utf8")).toBe(
      `${revision}\n`,
    );
    expect(identity).toEqual({
      revision,
      sourceId: provenance.sourceId,
      version: provenance.version,
    });
  });

  it("rejects missing or abbreviated revisions", () => {
    const runtime = mkdtempSync(join(tmpdir(), "mycellios-runtime-identity-"));
    temporaryRoots.push(runtime);

    expect(() =>
      prepareCoordinatorRuntimeIdentity(resolve("."), runtime, "abc123"),
    ).toThrow("exact 40-character Git SHA");
  });
});
