import {
  cpSync,
  mkdirSync,
  mkdtempSync,
  readlinkSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  assertNoEscapingSymlinks,
  normalizeCopiedInternalAbsoluteSymlinks,
  samePath,
} from "../scripts/portable-runtime-filesystem.mjs";

const temporaryRoots: string[] = [];

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) {
    rmSync(root, { recursive: true, force: true });
  }
});

describe.skipIf(process.platform === "win32")("portable runtime symlinks", () => {
  it("rewrites copied internal absolute links as relocatable relative links", () => {
    const root = temporaryRoot();
    const source = join(root, "source");
    const copied = join(root, "copied");
    mkdirSync(join(source, "bin"), { recursive: true });
    writeFileSync(join(source, "bin", "2to3-3.12"), "#!/usr/bin/env python3\n", "utf8");
    symlinkSync(join(source, "bin", "2to3-3.12"), join(source, "bin", "2to3"));

    assertNoEscapingSymlinks(source);
    cpSync(source, copied, { recursive: true, verbatimSymlinks: true });
    normalizeCopiedInternalAbsoluteSymlinks(copied, source);
    assertNoEscapingSymlinks(copied);

    const copiedLink = join(copied, "bin", "2to3");
    const target = readlinkSync(copiedLink);
    expect(isAbsolute(target)).toBe(false);
    expect(resolve(dirname(copiedLink), target)).toBe(join(copied, "bin", "2to3-3.12"));
  });

  it("still rejects links whose target escapes the runtime", () => {
    const root = temporaryRoot();
    const runtime = join(root, "runtime");
    const outside = join(root, "outside.txt");
    mkdirSync(runtime, { recursive: true });
    writeFileSync(outside, "outside", "utf8");
    symlinkSync(outside, join(runtime, "escape"));

    expect(() => assertNoEscapingSymlinks(runtime)).toThrow(/escaping symlink/);
  });

  it("compares canonical paths through filesystem aliases", () => {
    const root = temporaryRoot();
    const runtime = join(root, "runtime");
    const alias = join(root, "runtime-alias");
    mkdirSync(runtime, { recursive: true });
    symlinkSync(runtime, alias, "dir");

    expect(samePath(runtime, alias)).toBe(true);
  });
});

function temporaryRoot() {
  const root = mkdtempSync(join(tmpdir(), "mycellios-runtime-links-"));
  temporaryRoots.push(root);
  return root;
}
