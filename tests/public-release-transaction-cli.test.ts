import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { preparePublicReleaseTransaction } from "../src/coordinator/public-release-transaction-cli.js";
import { parseReleaseTransactionManifest } from "../src/coordinator/release-upload.js";

const roots: string[] = [];

afterEach(() => {
  for (const root of roots.splice(0)) {
    rmSync(root, { recursive: true, force: true });
  }
});

describe("public release transaction CLI", () => {
  it("hashes the exact four-asset native node release set in canonical order", async () => {
    const root = fixture();
    writeChecksums(root);
    const outputPath = join(root, "manifest.json");

    const manifest = await preparePublicReleaseTransaction({
      assetsRoot: root,
      outputPath,
      transactionId: "release-1234567890abcdef",
      sourceId: `sha256:${"a".repeat(64)}`,
      revision: "b".repeat(40),
      version: "0.2.19",
    });

    expect(manifest.assets).toHaveLength(4);
    expect(manifest.assets.map(({ channel, fileName }) => `${channel}/${fileName}`))
      .toEqual([
        "downloads/mycellios-node-linux-x64.tar.gz",
        "downloads/mycellios-node-macos-arm64.tar.gz",
        "downloads/mycellios-node-windows-x64.zip",
        "updates/mycellios-node-latest.json",
      ]);
    expect(
      parseReleaseTransactionManifest(
        JSON.parse(readFileSync(outputPath, "utf8")),
      ),
    ).toEqual(manifest);
  });

  it("fails closed on a missing or unexpected release asset", async () => {
    const root = fixture();
    rmSync(join(root, "mycellios-node-linux-x64.tar.gz"));
    writeFileSync(join(root, "unexpected.exe"), "unexpected");

    await expect(preparePublicReleaseTransaction({
      assetsRoot: root,
      outputPath: join(root, "manifest.json"),
      transactionId: "release-1234567890abcdef",
      sourceId: `sha256:${"a".repeat(64)}`,
      revision: "b".repeat(40),
      version: "0.2.19",
    })).rejects.toThrow("public_release_assets_contain_unexpected_files");
  });

  it("rejects a human checksum list that does not match the sealed assets", async () => {
    const root = fixture();
    writeFileSync(join(root, "sha256sums.txt"), `${"0".repeat(64)}  mycellios-node-windows-x64.zip\n`);

    await expect(preparePublicReleaseTransaction({
      assetsRoot: root,
      outputPath: join(root, "manifest.json"),
      transactionId: "release-1234567890abcdef",
      sourceId: `sha256:${"a".repeat(64)}`,
      revision: "b".repeat(40),
      version: "0.2.19",
    })).rejects.toThrow("public_release_checksums_do_not_match_assets");
  });
});

function fixture(): string {
  const root = mkdtempSync(join(tmpdir(), "mycellios-public-release-"));
  roots.push(root);
  const names = [
    "mycellios-node-latest.json",
    "mycellios-node-windows-x64.zip",
    "mycellios-node-macos-arm64.tar.gz",
    "mycellios-node-linux-x64.tar.gz",
  ];
  for (const name of names) writeFileSync(join(root, name), `fixture:${name}`);
  return root;
}

function writeChecksums(root: string): void {
  const names = [
    "mycellios-node-latest.json",
    "mycellios-node-linux-x64.tar.gz",
    "mycellios-node-macos-arm64.tar.gz",
    "mycellios-node-windows-x64.zip",
  ];
  const content = names.map((name) => {
    const bytes = readFileSync(join(root, name));
    return `${createHash("sha256").update(bytes).digest("hex")}  ${name}`;
  }).join("\n") + "\n";
  writeFileSync(join(root, "sha256sums.txt"), content);
}
