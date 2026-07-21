import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, open, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

export type ReleaseAssetChannel = "updates" | "downloads";

export interface ReleaseChunkMetadata {
  chunkIndex: number;
  chunkCount: number;
  chunkSha256: string;
  fileSha256: string;
  fileSize: number;
}

export interface StoredReleaseChunk {
  complete: boolean;
  receivedChunk: number;
  chunkCount: number;
  fileSha256: string;
}

const UPDATE_FILE = /^(?:RELEASES|latest\.json|mycellios-setup\.exe|mycellios-\d+\.\d+\.\d+-full\.nupkg)$/;
const DOWNLOAD_FILE = /^mycellios-(?:windows-x64\.exe|macos-(?:arm64|x64)\.dmg|linux-x64\.(?:deb|rpm))$/;
const SHA256 = /^[0-9a-f]{64}$/;
const MAX_RELEASE_FILE_SIZE = 1_500_000_000;
const MAX_RELEASE_CHUNKS = 2_048;
const MAX_CHUNK_SIZE = 64 * 1024 * 1024;

export function validateReleaseAssetName(channel: ReleaseAssetChannel, fileName: string): void {
  const allowed = channel === "updates" ? UPDATE_FILE : DOWNLOAD_FILE;
  if (!allowed.test(fileName)) throw new Error("release_asset_name_not_allowed");
}

export function parseReleaseChunkMetadata(headers: Record<string, unknown>): ReleaseChunkMetadata {
  const chunkIndex = integerHeader(headers, "x-chunk-index", 0, MAX_RELEASE_CHUNKS - 1);
  const chunkCount = integerHeader(headers, "x-chunk-count", 1, MAX_RELEASE_CHUNKS);
  const fileSize = integerHeader(headers, "x-file-size", 1, MAX_RELEASE_FILE_SIZE);
  const chunkSha256 = shaHeader(headers, "x-chunk-sha256");
  const fileSha256 = shaHeader(headers, "x-file-sha256");
  if (chunkIndex >= chunkCount) throw new Error("release_chunk_index_out_of_range");
  return { chunkIndex, chunkCount, chunkSha256, fileSha256, fileSize };
}

export async function storeReleaseChunk(input: {
  root: string;
  channel: ReleaseAssetChannel;
  fileName: string;
  metadata: ReleaseChunkMetadata;
  body: Buffer;
}): Promise<StoredReleaseChunk> {
  validateReleaseAssetName(input.channel, input.fileName);
  if (!Buffer.isBuffer(input.body) || input.body.length === 0 || input.body.length > MAX_CHUNK_SIZE) {
    throw new Error("release_chunk_size_invalid");
  }
  if (digest(input.body) !== input.metadata.chunkSha256) {
    throw new Error("release_chunk_checksum_mismatch");
  }

  const root = resolve(input.root);
  const target = resolve(root, input.fileName);
  if (dirname(target) !== root) throw new Error("release_asset_path_invalid");
  await mkdir(root, { recursive: true });

  if (await fileMatches(target, input.metadata.fileSize, input.metadata.fileSha256)) {
    return result(input.metadata, true);
  }

  const staging = resolve(root, ".incoming", input.metadata.fileSha256);
  await mkdir(staging, { recursive: true });
  const chunkPath = resolve(staging, `${String(input.metadata.chunkIndex).padStart(6, "0")}.part`);
  const temporaryChunk = `${chunkPath}.${randomUUID()}.tmp`;
  await writeFile(temporaryChunk, input.body, { flag: "wx" });
  await rename(temporaryChunk, chunkPath);

  for (let index = 0; index < input.metadata.chunkCount; index += 1) {
    try {
      await stat(resolve(staging, `${String(index).padStart(6, "0")}.part`));
    } catch {
      return result(input.metadata, false);
    }
  }

  const temporaryTarget = `${target}.${input.metadata.fileSha256}.tmp`;
  const handle = await open(temporaryTarget, "w");
  const hash = createHash("sha256");
  let bytes = 0;
  try {
    for (let index = 0; index < input.metadata.chunkCount; index += 1) {
      const chunk = await readFile(resolve(staging, `${String(index).padStart(6, "0")}.part`));
      await handle.write(chunk);
      hash.update(chunk);
      bytes += chunk.length;
    }
  } finally {
    await handle.close();
  }

  if (bytes !== input.metadata.fileSize || hash.digest("hex") !== input.metadata.fileSha256) {
    await rm(temporaryTarget, { force: true });
    throw new Error("release_file_checksum_mismatch");
  }
  await rename(temporaryTarget, target);
  await rm(staging, { recursive: true, force: true });
  return result(input.metadata, true);
}

async function fileMatches(path: string, size: number, sha256: string): Promise<boolean> {
  try {
    if ((await stat(path)).size !== size) return false;
    const hash = createHash("sha256");
    for await (const chunk of createReadStream(path)) hash.update(chunk);
    return hash.digest("hex") === sha256;
  } catch {
    return false;
  }
}

function result(metadata: ReleaseChunkMetadata, complete: boolean): StoredReleaseChunk {
  return {
    complete,
    receivedChunk: metadata.chunkIndex,
    chunkCount: metadata.chunkCount,
    fileSha256: metadata.fileSha256,
  };
}

function digest(value: Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function header(headers: Record<string, unknown>, name: string): string {
  const value = headers[name];
  if (typeof value !== "string" || !value) throw new Error(`release_header_missing_${name}`);
  return value;
}

function integerHeader(
  headers: Record<string, unknown>,
  name: string,
  minimum: number,
  maximum: number,
): number {
  const value = Number(header(headers, name));
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`release_header_invalid_${name}`);
  }
  return value;
}

function shaHeader(headers: Record<string, unknown>, name: string): string {
  const value = header(headers, name).toLowerCase();
  if (!SHA256.test(value)) throw new Error(`release_header_invalid_${name}`);
  return value;
}
