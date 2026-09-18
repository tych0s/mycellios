import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import { collectModelArtifactCache } from "../src/model-fabric/model-artifact-cache.js";

const cleanup: string[] = [];
afterEach(async () => Promise.all(cleanup.splice(0).map((path) => rm(path, { recursive: true, force: true }))));

async function temporaryDirectory(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "mycellios-model-cache-"));
  cleanup.push(root);
  return root;
}

async function artifact(root: string, character: string, bytes: number): Promise<string> {
  const digest = character.repeat(64);
  const directory = join(root, digest);
  await mkdir(directory);
  await writeFile(join(directory, "artifact.bin"), Buffer.alloc(bytes, character));
  return digest;
}

describe("model artifact cache", () => {
  it.each([false, true])("rejects directory links without touching their target (nested=%s)", async (nested) => {
    const root = await mkdtemp(join(tmpdir(), "mycellios-model-cache-link-"));
    try {
      const cache = join(root, "cache");
      const outside = join(root, "outside");
      await mkdir(cache);
      await mkdir(outside);
      await writeFile(join(outside, "keep.bin"), "preserve");
      const digestPath = join(cache, "a".repeat(64));
      if (nested) await mkdir(digestPath);
      await symlink(outside, nested ? join(digestPath, "linked") : digestPath,
        process.platform === "win32" ? "junction" : "dir");
      await expect(collectModelArtifactCache(cache, 0, {
        active: [], previous: [], inUse: [], resumable: [],
      })).rejects.toThrow("model_artifact_cache_entry_is_unsafe");
      await expect(readFile(join(outside, "keep.bin"), "utf8")).resolves.toBe("preserve");
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("collects oldest unprotected content while preserving active, previous, in-use and resumable roots", async () => {
    const root = await temporaryDirectory();
    const active = await artifact(root, "a", 10);
    const previous = await artifact(root, "b", 10);
    const inUse = await artifact(root, "c", 10);
    const resumable = await artifact(root, "d", 10);
    const victim = await artifact(root, "e", 20);
    const result = await collectModelArtifactCache(root, 40, {
      active: [active], previous: [previous], inUse: [inUse], resumable: [resumable],
    });
    expect(result.removedDigests).toEqual([victim]);
    await expect(readFile(join(root, active, "artifact.bin"))).resolves.toHaveLength(10);
    await expect(readFile(join(root, resumable, "artifact.bin"))).resolves.toHaveLength(10);
  });

  it("fails closed when protected bytes exceed quota", async () => {
    const root = await temporaryDirectory();
    const active = await artifact(root, "f", 20);
    await expect(collectModelArtifactCache(root, 10, {
      active: [active], previous: [], inUse: [], resumable: [],
    })).rejects.toThrow("model_artifact_cache_quota_blocked_by_protected_artifacts");
  });

  it("accounts for nested artifact directories using native paths", async () => {
    const root = await temporaryDirectory();
    const victim = await artifact(root, "a", 10);
    await mkdir(join(root, victim, "weights"));
    await writeFile(join(root, victim, "weights", "part.bin"), Buffer.alloc(20));
    await expect(collectModelArtifactCache(root, 0, {
      active: [], previous: [], inUse: [], resumable: [],
    })).resolves.toEqual({ bytesBefore: 30, bytesAfter: 0, removedDigests: [victim] });
  });

});
