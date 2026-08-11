import { lstat, readdir, rm, stat } from "node:fs/promises";
import { resolve } from "node:path";

export interface ModelArtifactCacheProtection {
  active: readonly string[];
  previous: readonly string[];
  inUse: readonly string[];
  resumable: readonly string[];
}

export interface ModelArtifactCacheGcResult {
  bytesBefore: number;
  bytesAfter: number;
  removedDigests: string[];
}

/** Quota GC for content-addressed model artifacts. Protected generations and
 * resumable downloads are immutable roots and can never be selected as victims.
 */
export async function collectModelArtifactCache(
  cacheRoot: string,
  quotaBytes: number,
  protection: ModelArtifactCacheProtection,
): Promise<ModelArtifactCacheGcResult> {
  if (!Number.isSafeInteger(quotaBytes) || quotaBytes < 0) throw new Error("model_artifact_cache_quota_is_invalid");
  const root = resolve(cacheRoot);
  const protectedDigests = new Set(
    [...protection.active, ...protection.previous, ...protection.inUse, ...protection.resumable]
      .map(validateDigest),
  );
  const entries: Array<{ digest: string; path: string; bytes: number; mtimeMs: number }> = [];
  for (const entry of await readdir(root, { withFileTypes: true })) {
    if (!/^[0-9a-f]{64}$/.test(entry.name)) continue;
    if (entry.isSymbolicLink() || !entry.isDirectory()) throw new Error("model_artifact_cache_entry_is_unsafe");
    const path = resolve(root, entry.name);
    if (!path.startsWith(`${root}/`)) throw new Error("model_artifact_cache_path_escaped");
    const metadata = await stat(path);
    entries.push({ digest: entry.name, path, bytes: await directoryBytes(path), mtimeMs: metadata.mtimeMs });
  }
  const bytesBefore = entries.reduce((sum, entry) => sum + entry.bytes, 0);
  let bytesAfter = bytesBefore;
  const removedDigests: string[] = [];
  for (const entry of entries
    .filter(({ digest }) => !protectedDigests.has(digest))
    .sort((left, right) => left.mtimeMs - right.mtimeMs || left.digest.localeCompare(right.digest))) {
    if (bytesAfter <= quotaBytes) break;
    await rm(entry.path, { recursive: true, force: true });
    bytesAfter -= entry.bytes;
    removedDigests.push(entry.digest);
  }
  if (bytesAfter > quotaBytes) throw new Error("model_artifact_cache_quota_blocked_by_protected_artifacts");
  return { bytesBefore, bytesAfter, removedDigests };
}

async function directoryBytes(root: string): Promise<number> {
  let total = 0;
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const path = resolve(root, entry.name);
    if (!path.startsWith(`${root}/`) || entry.isSymbolicLink()) throw new Error("model_artifact_cache_entry_is_unsafe");
    if (entry.isDirectory()) total += await directoryBytes(path);
    else if (entry.isFile()) total += (await lstat(path)).size;
    else throw new Error("model_artifact_cache_entry_is_unsafe");
  }
  return total;
}

function validateDigest(value: string): string {
  const digest = value.startsWith("sha256:") ? value.slice(7) : value;
  if (!/^[0-9a-f]{64}$/.test(digest)) throw new Error("model_artifact_cache_digest_is_invalid");
  return digest;
}
