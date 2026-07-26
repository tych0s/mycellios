import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildNativePythonProductManifest,
} from "../scripts/native-python-product-policy.mjs";
import {
  assertNativePythonManifest,
  parseNuspec,
  parseReleases,
  readZipEvidence,
  validateArchivePath,
} from "../scripts/verify-windows-release-artifacts.mjs";

describe("Windows release verifier policy", () => {
  it("accepts only the exact native Python manifest", () => {
    const manifest = buildNativePythonProductManifest(resolve("python"));
    expect(() =>
      assertNativePythonManifest(manifest, "fixture"),
    ).not.toThrow();

    const extra = structuredClone(manifest);
    extra.files.push({
      path: "distributed_runtime/unlisted.py",
      bytes: 1,
      sha256: "0".repeat(64),
    });
    expect(() =>
      assertNativePythonManifest(extra, "fixture"),
    ).toThrow("allowlist");

    const corrupt = structuredClone(manifest);
    corrupt.files[0]!.sha256 = "invalid";
    expect(() =>
      assertNativePythonManifest(corrupt, "fixture"),
    ).toThrow("evidencia");

    const wrongPolicy = {
      ...structuredClone(manifest),
      policyId: `sha256:${"f".repeat(64)}`,
    };
    expect(() =>
      assertNativePythonManifest(wrongPolicy, "fixture"),
    ).toThrow("allowlist");

    expect(() =>
      assertNativePythonManifest(
        { ...structuredClone(manifest), untrusted: true },
        "fixture",
      ),
    ).toThrow("allowlist");
  });

  it("parses one canonical Squirrel RELEASES record and rejects ambiguity", () => {
    const record = parseReleases(
      Buffer.from(`${"a".repeat(40)} mycellios-0.2.19-full.nupkg 123\n`),
      "RELEASES",
    );
    expect(record).toEqual({
      sha1: "A".repeat(40),
      name: "mycellios-0.2.19-full.nupkg",
      size: 123n,
    });

    for (const invalid of [
      `${"a".repeat(40)} ../escape.nupkg 123\n`,
      `${"a".repeat(40)} package.nupkg 123\nextra\n`,
      ` ${"a".repeat(40)} package.nupkg 123\n`,
      `${"a".repeat(40)} package.nupkg 123\r`,
    ]) {
      expect(() =>
        parseReleases(Buffer.from(invalid), "RELEASES"),
      ).toThrow();
    }
  });

  it("accepts one strict NUSPEC identity and rejects XML expansion or duplicates", () => {
    expect(
      parseNuspec(
        Buffer.from(
          '<?xml version="1.0"?><package><metadata><id>mycellios</id><version>0.2.19</version></metadata></package>',
        ),
        "fixture.nuspec",
      ),
    ).toEqual({ id: "mycellios", version: "0.2.19" });

    expect(() =>
      parseNuspec(
        Buffer.from(
          '<!DOCTYPE x [<!ENTITY leak "x">]><package><metadata><id>&leak;</id><version>1</version></metadata></package>',
        ),
        "fixture.nuspec",
      ),
    ).toThrow("no permitidas");
    expect(() =>
      parseNuspec(
        Buffer.from(
          "<package><metadata><id>mycellios</id><id>other</id><version>1</version></metadata></package>",
        ),
        "fixture.nuspec",
      ),
    ).toThrow("únicos");
  });

  it("rejects unsafe archive paths before extraction", () => {
    expect(validateArchivePath("lib/net45/resources/app.asar")).toEqual({
      path: "lib/net45/resources/app.asar",
      directory: false,
    });
    expect(validateArchivePath("lib/net45/resources/")).toEqual({
      path: "lib/net45/resources",
      directory: true,
    });
    for (const unsafe of [
      "../app.asar",
      "/absolute/app.asar",
      "C:/app.asar",
      "lib\\app.asar",
      "lib//app.asar",
      "lib/CON",
      "lib/trailing.",
    ]) {
      expect(() => validateArchivePath(unsafe)).toThrow();
    }
  });

  it("reads only the exact embedded Setup ZIP entry set", async () => {
    const releases = Buffer.from("release evidence\n");
    const nupkg = Buffer.from("sealed nupkg bytes");
    const valid = zipStored([
      ["RELEASES", releases],
      ["mycellios-0.2.19-full.nupkg", nupkg],
    ]);
    const expected = new Set([
      "RELEASES",
      "mycellios-0.2.19-full.nupkg",
    ]);
    const evidence = await readZipEvidence(valid, expected);
    expect(evidence.get("RELEASES")?.buffer).toEqual(releases);
    expect(evidence.get("mycellios-0.2.19-full.nupkg")?.bytes).toBe(
      BigInt(nupkg.length),
    );

    const unexpected = zipStored([
      ["RELEASES", releases],
      ["mycellios-0.2.19-full.nupkg", nupkg],
      ["unexpected.dll", Buffer.from("x")],
    ]);
    await expect(readZipEvidence(unexpected, expected)).rejects.toThrow(
      "entrada inesperada",
    );

    const duplicate = zipStored([
      ["RELEASES", releases],
      ["RELEASES", releases],
      ["mycellios-0.2.19-full.nupkg", nupkg],
    ]);
    await expect(readZipEvidence(duplicate, expected)).rejects.toThrow(
      "entrada inesperada",
    );
  });
});

function zipStored(entries: Array<readonly [string, Buffer]>): Buffer {
  const localParts: Buffer[] = [];
  const centralParts: Buffer[] = [];
  let localOffset = 0;

  for (const [name, data] of entries) {
    const fileName = Buffer.from(name, "utf8");
    const crc = crc32(data);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(0, 6);
    local.writeUInt16LE(0, 8);
    local.writeUInt32LE(0, 10);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(data.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(fileName.length, 26);
    local.writeUInt16LE(0, 28);
    localParts.push(local, fileName, data);

    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(20, 4);
    central.writeUInt16LE(20, 6);
    central.writeUInt16LE(0, 8);
    central.writeUInt16LE(0, 10);
    central.writeUInt32LE(0, 12);
    central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(data.length, 20);
    central.writeUInt32LE(data.length, 24);
    central.writeUInt16LE(fileName.length, 28);
    central.writeUInt16LE(0, 30);
    central.writeUInt16LE(0, 32);
    central.writeUInt16LE(0, 34);
    central.writeUInt16LE(0, 36);
    central.writeUInt32LE(0, 38);
    central.writeUInt32LE(localOffset, 42);
    centralParts.push(central, fileName);
    localOffset += local.length + fileName.length + data.length;
  }

  const centralDirectory = Buffer.concat(centralParts);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(0, 4);
  end.writeUInt16LE(0, 6);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(centralDirectory.length, 12);
  end.writeUInt32LE(localOffset, 16);
  end.writeUInt16LE(0, 20);
  return Buffer.concat([...localParts, centralDirectory, end]);
}

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}
