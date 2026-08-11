import { describe, expect, it } from "vitest";
import { generateWixSource } from "../scripts/build-node-msi.mjs";

describe("native Windows MSI source", () => {
  it("binds every staged file, a per-machine identity and pairing association", () => {
    const source = generateWixSource({ stagedRoot: "C:\\staged", version: "0.2.77", sourceRevision: "a".repeat(40), files: [
      { path: "bin/node.exe", bytes: 1, sha256: "b".repeat(64) }, { path: "app/node/install-main.js", bytes: 1, sha256: "c".repeat(64) },
      { path: "install.ps1", bytes: 1, sha256: "d".repeat(64) },
    ] });
    expect(source).toContain('InstallScope="perMachine"'); expect(source).toContain(".mycellios-enrollment");
    expect(source).toContain("[INSTALLFOLDER]install.ps1"); expect(source).toContain("%1"); expect(source).toContain("MYCELLIOSSOURCEREVISION");
    expect(source.match(/<File /g)).toHaveLength(3); expect(source).not.toContain("enrollmentToken");
  });

  it("is deterministic for identical source identity", () => {
    const input = { stagedRoot: "C:\\staged", version: "0.2.77", sourceRevision: "a".repeat(40), files: [{ path: "install.ps1", bytes: 1, sha256: "d".repeat(64) }] };
    expect(generateWixSource(input)).toBe(generateWixSource(input));
  });
});
