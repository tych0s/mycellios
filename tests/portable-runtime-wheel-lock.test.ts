import { createHash } from "node:crypto";
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it } from "vitest";
import {
  parsePortableRuntimeWheelLock,
  readPortableRuntimeWheelLock,
  verifyPortableRuntimeWheelhouse,
} from "../scripts/portable-runtime-wheel-lock.mjs";

const temporaryDirectories: string[] = [];
const WORKSPACE = resolve(import.meta.dirname, "..");

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("portable runtime wheel lock", () => {
  it("verifies every supported platform lock and its exact source digest", () => {
    for (const target of [
      ["win32", "x64", "2.13.0+cpu", "win32-x64-cp312.txt", "2a5de3d3f2e3ba4ebac91e1efe068159e81263d3ccd6a3d93333078e753c6c65"],
      ["linux", "x64", "2.13.0+cpu", "linux-x64-cp312.txt", "090355535c96e7202ae761a18dadc5d66c9e98493a148eba16ae3c3e7c95c59d"],
      ["darwin", "arm64", "2.11.0", "darwin-arm64-cp312.txt", "75c84302540edc1b7f6c1620c3474b7ccbf431dd6df881717f74ebf419c56348"],
    ] as const) {
      const [platform, arch, torchVersion, filename, sha256] = target;
      const lock = readPortableRuntimeWheelLock(WORKSPACE, {
        supported: true,
        platform,
        arch,
        torchVersion,
        packageVersions: {
          numpy: "1.26.4",
          aiohttp: "3.14.1",
          accelerate: "1.14.0",
          transformers: "5.14.1",
          safetensors: "0.8.0",
          sentencepiece: "0.2.2",
        },
        wheelLock: {
          schema: "mycellios-python-wheel-lock/1",
          path: `scripts/wheel-locks/${filename}`,
          sha256,
        },
      });
      expect(lock).toMatchObject({ platform, arch, sha256 });
      expect(lock.artifacts.some((artifact) =>
        artifact.name === "torch" && artifact.version === torchVersion
      )).toBe(true);
    }
  });

  it("fails closed when a reviewed lock file changes without updating policy", () => {
    const source = readFileSync(
      resolve(WORKSPACE, "scripts", "wheel-locks", "win32-x64-cp312.txt"),
      "utf8",
    );
    const root = temporaryRoot();
    const path = join(root, "scripts", "wheel-locks", "win32-x64-cp312.txt");
    mkdirSync(resolve(path, ".."), { recursive: true });
    writeFileSync(path, source.replace("accelerate-1.14.0", "accelerate-1.14.1"), "utf8");

    expect(() => readPortableRuntimeWheelLock(root, {
      supported: true,
      platform: "win32",
      arch: "x64",
      torchVersion: "2.13.0+cpu",
      packageVersions: {},
      wheelLock: {
        schema: "mycellios-python-wheel-lock/1",
        path: "scripts/wheel-locks/win32-x64-cp312.txt",
        sha256: createHash("sha256").update(source).digest("hex"),
      },
    })).toThrow("has SHA-256");
  });

  it("rejects a downloaded wheel whose bytes do not match the sealed artifact hash", async () => {
    const source = [
      "# mycellios-python-wheel-lock/1",
      "# target=win32/x64/cp312",
      "# Regenerate deliberately from a reviewed pip JSON report; never edit hashes by hand.",
      `torch @ https://files.pythonhosted.org/packages/test/torch-1.0.0-py3-none-any.whl --hash=sha256:${"a".repeat(64)} # version=1.0.0`,
      "",
    ].join("\n");
    const lock = parsePortableRuntimeWheelLock(source, {
      supported: true,
      platform: "win32",
      arch: "x64",
      torchVersion: "1.0.0",
      packageVersions: {},
      wheelLock: {
        schema: "mycellios-python-wheel-lock/1",
        path: "scripts/wheel-locks/test.txt",
        sha256: createHash("sha256").update(source).digest("hex"),
      },
    });
    const wheelhouse = temporaryRoot();
    writeFileSync(
      join(wheelhouse, "torch-1.0.0-py3-none-any.whl"),
      "tampered wheel",
      "utf8",
    );

    await expect(verifyPortableRuntimeWheelhouse(wheelhouse, lock))
      .rejects.toThrow("has SHA-256");
  });

  it("accepts only the exact sealed wheelhouse and rejects additional files", async () => {
    const wheel = Buffer.from("reviewed wheel bytes");
    const wheelSha256 = createHash("sha256").update(wheel).digest("hex");
    const source = [
      "# mycellios-python-wheel-lock/1",
      "# target=win32/x64/cp312",
      "# Regenerate deliberately from a reviewed pip JSON report; never edit hashes by hand.",
      `torch @ https://files.pythonhosted.org/packages/test/torch-1.0.0-py3-none-any.whl --hash=sha256:${wheelSha256} # version=1.0.0`,
      "",
    ].join("\n");
    const lock = parsePortableRuntimeWheelLock(source, {
      supported: true,
      platform: "win32",
      arch: "x64",
      torchVersion: "1.0.0",
      packageVersions: {},
      wheelLock: {
        schema: "mycellios-python-wheel-lock/1",
        path: "scripts/wheel-locks/test.txt",
        sha256: createHash("sha256").update(source).digest("hex"),
      },
    });
    const wheelhouse = temporaryRoot();
    writeFileSync(join(wheelhouse, "torch-1.0.0-py3-none-any.whl"), wheel);

    await expect(verifyPortableRuntimeWheelhouse(wheelhouse, lock))
      .resolves.toBe(true);

    writeFileSync(join(wheelhouse, "unexpected.whl"), wheel);
    await expect(verifyPortableRuntimeWheelhouse(wheelhouse, lock))
      .rejects.toThrow("contains 2 entries; expected 1");
  });
});

function temporaryRoot(): string {
  const root = mkdtempSync(join(tmpdir(), "mycellios-wheel-lock-"));
  temporaryDirectories.push(root);
  return root;
}
