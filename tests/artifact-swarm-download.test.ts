import { createHash, generateKeyPairSync } from "node:crypto";
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { mkdir } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import { basename, join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ARTIFACT_SWARM_SCHEMA,
  ArtifactSwarmRegistry,
  artifactChunkId,
  createArtifactSwarmDownloader,
  signArtifactSwarmManifest,
  type ArtifactBlobDescriptor,
  type ArtifactPeerChunkClient,
  type ArtifactSwarmPinnedArtifact,
  type ArtifactSwarmTransfer,
} from "../src/model-fabric/artifact-swarm.js";

const temporaryDirectories: string[] = [];
const servers: Server[] = [];

afterEach(async () => {
  await Promise.all(servers.splice(0).map((server) =>
    new Promise<void>((resolve) => server.close(() => resolve()))
  ));
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("native artifact swarm downloader", () => {
  it("streams real peer bytes, verifies every chunk, and atomically assembles the artifact", async () => {
    const fixture = createFixture(Buffer.from("mycellios-native-peer-artifact"), [7, 9, 14]);
    const peer = await startPeerServer((peerId, chunkId) => {
      expect(peerId).toBe("fast-peer");
      return fixture.chunkBytes.get(chunkId) ?? null;
    });
    announce(fixture, "fast-peer");
    const origin = vi.fn(async () => {
      throw new Error("origin_must_not_be_used");
    });
    const progress: ArtifactSwarmTransfer[] = [];
    const downloader = createArtifactSwarmDownloader({
      registry: fixture.registry,
      manifestId: fixture.manifestId,
      requesterPeerId: "requester",
      peerClient: peer.client,
      originDownloader: origin,
      maximumParallelChunks: 2,
    });

    const target = await downloader(
      fixture.artifact,
      fixture.cacheRoot,
      (event) => progress.push(event),
    );

    expect(readFileSync(target)).toEqual(fixture.bytes);
    expect(origin).not.toHaveBeenCalled();
    expect(progress.at(-1)).toMatchObject({
      bytesDownloaded: fixture.bytes.length,
      bytesTotal: fixture.bytes.length,
      etaSeconds: 0,
    });
    expect(
      readdirSync(join(fixture.cacheRoot, fixture.artifact.sha256))
        .some((entry) => entry.endsWith(".assembling") || entry.endsWith(".tmp")),
    ).toBe(false);
  });

  it("keeps verified chunks after an interrupted peer and resumes without downloading them again", async () => {
    const fixture = createFixture(Buffer.from("resume-real-peer-download"), [6, 7, 12]);
    const interruptedChunk = artifactChunkId(
      fixture.blob.sha256,
      fixture.blob.chunks[1]!.index,
    );
    let interruptOnce = true;
    const requests = new Map<string, number>();
    const peer = await startPeerServer((_peerId, chunkId) => {
      requests.set(chunkId, (requests.get(chunkId) ?? 0) + 1);
      if (chunkId === interruptedChunk && interruptOnce) {
        interruptOnce = false;
        return "interrupt";
      }
      return fixture.chunkBytes.get(chunkId) ?? null;
    });
    announce(fixture, "resume-peer");
    const origin = vi.fn(async () => {
      throw new Error("origin_temporarily_offline");
    });
    const progress: ArtifactSwarmTransfer[] = [];
    const downloader = createArtifactSwarmDownloader({
      registry: fixture.registry,
      manifestId: fixture.manifestId,
      requesterPeerId: "requester",
      peerClient: peer.client,
      originDownloader: origin,
      maximumParallelChunks: 1,
    });

    await expect(
      downloader(fixture.artifact, fixture.cacheRoot, () => undefined),
    ).rejects.toThrow("origin_temporarily_offline");
    const firstChunkId = artifactChunkId(
      fixture.blob.sha256,
      fixture.blob.chunks[0]!.index,
    );
    expect(requests.get(firstChunkId)).toBe(1);

    const target = await downloader(
      fixture.artifact,
      fixture.cacheRoot,
      (event) => progress.push(event),
    );

    expect(readFileSync(target)).toEqual(fixture.bytes);
    expect(requests.get(firstChunkId)).toBe(1);
    expect(progress.some((event) => event.resumed)).toBe(true);
  });

  it("penalizes and quarantines a corrupt peer while accepting verified bytes from a healthy peer", async () => {
    const fixture = createFixture(Buffer.from("corrupt-peer-cannot-poison-cache"), [9, 9, 14]);
    const peer = await startPeerServer((peerId, chunkId) => {
      const expected = fixture.chunkBytes.get(chunkId);
      if (expected === undefined) return null;
      return peerId === "bad-peer"
        ? Buffer.alloc(expected.length, 0x78)
        : expected;
    });
    announce(fixture, "bad-peer", {
      rttMs: 1,
      goodputMbps: 10_000,
      reliability: 1,
    });
    announce(fixture, "good-peer", {
      rttMs: 20,
      goodputMbps: 1_000,
      reliability: 0.99,
    });
    const origin = vi.fn(async () => {
      throw new Error("origin_must_not_be_used");
    });
    const downloader = createArtifactSwarmDownloader({
      registry: fixture.registry,
      manifestId: fixture.manifestId,
      requesterPeerId: "requester",
      peerClient: peer.client,
      originDownloader: origin,
      maximumParallelChunks: 1,
    });

    const target = await downloader(
      fixture.artifact,
      fixture.cacheRoot,
      () => undefined,
    );

    expect(readFileSync(target)).toEqual(fixture.bytes);
    expect(fixture.registry.peerState("bad-peer")).toMatchObject({
      corruptionStrikes: 2,
      quarantined: true,
    });
    expect(origin).not.toHaveBeenCalled();
  });

  it("falls back to the pinned origin when no peer proves the advertised bytes", async () => {
    const fixture = createFixture(Buffer.from("verified-origin-fallback"), [8, 8, 8]);
    const originServer = await startByteServer(fixture.bytes);
    const origin = vi.fn(async (
      artifact: ArtifactSwarmPinnedArtifact,
      cacheRoot: string,
      onProgress: (event: ArtifactSwarmTransfer) => void,
    ) => {
      const response = await fetch(originServer.url);
      expect(response.ok).toBe(true);
      const bytes = Buffer.from(await response.arrayBuffer());
      const directory = join(cacheRoot, artifact.sha256);
      await mkdir(directory, { recursive: true });
      const target = join(directory, basename(new URL(artifact.url).pathname));
      const temporary = `${target}.origin.tmp`;
      writeFileSync(temporary, bytes);
      renameSync(temporary, target);
      onProgress({
        bytesDownloaded: bytes.length,
        bytesTotal: artifact.sizeBytes,
        bytesPerSecond: null,
        etaSeconds: 0,
        resumed: false,
      });
      return target;
    });
    const unavailableClient: ArtifactPeerChunkClient = {
      async fetchChunk() {
        throw new Error("peer_unavailable");
      },
    };
    const downloader = createArtifactSwarmDownloader({
      registry: fixture.registry,
      manifestId: fixture.manifestId,
      requesterPeerId: "requester",
      peerClient: unavailableClient,
      originDownloader: origin,
    });

    const target = await downloader(
      fixture.artifact,
      fixture.cacheRoot,
      () => undefined,
    );

    expect(readFileSync(target)).toEqual(fixture.bytes);
    expect(origin).toHaveBeenCalledOnce();
    expect(existsSync(`${target}.origin.tmp`)).toBe(false);
  });
});

function createFixture(bytes: Buffer, chunkSizes: number[]) {
  expect(chunkSizes.reduce((sum, size) => sum + size, 0)).toBe(bytes.length);
  const blob = blobFromBytes(bytes, chunkSizes);
  const packageManifest = blobFromBytes(Buffer.from("sealed-package-contract"), [23]);
  const packageId = digest(Buffer.from("native-artifact-package"));
  const keys = generateKeyPairSync("ed25519");
  const registry = new ArtifactSwarmRegistry();
  const signed = registry.publish(signArtifactSwarmManifest({
    schema: ARTIFACT_SWARM_SCHEMA,
    modelIdentity: `sha256:${digest(Buffer.from("native-artifact-model"))}`,
    sourceRevision: "mycellios-native-runtime-1",
    tensorAbi: "mycellios-native-artifact-bytes/1",
    packages: [{
      packageId,
      layerStart: 0,
      layerEnd: 1,
      manifest: packageManifest,
      blobs: [blob],
    }],
  }, keys.privateKey, keys.publicKey));
  const cacheRoot = temporaryRoot();
  const artifact: ArtifactSwarmPinnedArtifact = {
    label: "Native runtime wheel",
    url: `https://origin.example/runtime.whl#sha256=${blob.sha256}`,
    sha256: blob.sha256,
    sizeBytes: blob.sizeBytes,
  };
  const chunkBytes = new Map<string, Buffer>();
  for (const chunk of blob.chunks) {
    chunkBytes.set(
      artifactChunkId(blob.sha256, chunk.index),
      bytes.subarray(chunk.offset, chunk.offset + chunk.sizeBytes),
    );
  }
  return {
    artifact,
    blob,
    bytes,
    cacheRoot,
    chunkBytes,
    manifestId: signed.manifestId,
    packageId,
    registry,
  };
}

function announce(
  fixture: ReturnType<typeof createFixture>,
  peerId: string,
  overrides: Partial<{
    rttMs: number;
    goodputMbps: number;
    reliability: number;
  }> = {},
): void {
  fixture.registry.announcePeer({
    peerId,
    chunkIds: fixture.blob.chunks.map((chunk) =>
      artifactChunkId(fixture.blob.sha256, chunk.index)
    ),
    expiresAt: Date.now() + 60_000,
    rttMs: overrides.rttMs ?? 5,
    goodputMbps: overrides.goodputMbps ?? 1_000,
    activeTransfers: 0,
    reliability: overrides.reliability ?? 0.99,
  });
}

function blobFromBytes(bytes: Buffer, sizes: number[]): ArtifactBlobDescriptor {
  let offset = 0;
  return {
    sha256: digest(bytes),
    sizeBytes: bytes.length,
    chunks: sizes.map((sizeBytes, index) => {
      const value = bytes.subarray(offset, offset + sizeBytes);
      const chunk = {
        index,
        offset,
        sizeBytes,
        sha256: digest(value),
      };
      offset += sizeBytes;
      return chunk;
    }),
  };
}

async function startPeerServer(
  provide: (
    peerId: string,
    chunkId: string,
  ) => Buffer | "interrupt" | null,
): Promise<{ url: string; client: ArtifactPeerChunkClient }> {
  const server = createServer((request, response) => {
    const match = /^\/peer\/([^/]+)\/chunk\/([^/]+)$/.exec(request.url ?? "");
    if (!match) {
      response.writeHead(404).end();
      return;
    }
    const peerId = decodeURIComponent(match[1]!);
    const chunkId = decodeURIComponent(match[2]!);
    const value = provide(peerId, chunkId);
    if (value === "interrupt") {
      request.socket.destroy();
      return;
    }
    if (value === null) {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, {
      "content-length": value.length,
      "content-type": "application/octet-stream",
    });
    response.end(value);
  });
  const url = await listen(server);
  return {
    url,
    client: {
      async fetchChunk(request) {
        const response = await fetch(
          `${url}/peer/${encodeURIComponent(request.peerId)}/chunk/${encodeURIComponent(request.chunkId)}`,
        );
        if (!response.ok || response.body === null) {
          throw new Error(`peer_http_${response.status}`);
        }
        return {
          body: webBody(response.body),
          contentLength: Number(response.headers.get("content-length")),
        };
      },
    },
  };
}

async function startByteServer(bytes: Buffer): Promise<{ url: string }> {
  const server = createServer((_request, response) => {
    response.writeHead(200, {
      "content-length": bytes.length,
      "content-type": "application/octet-stream",
    });
    response.end(bytes);
  });
  return { url: await listen(server) };
}

async function listen(server: Server): Promise<string> {
  servers.push(server);
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (address === null || typeof address === "string") {
    throw new Error("test_server_address_is_invalid");
  }
  return `http://127.0.0.1:${address.port}`;
}

function webBody(
  body: ReadableStream<Uint8Array>,
): AsyncIterable<Uint8Array> {
  return {
    async *[Symbol.asyncIterator]() {
      const reader = body.getReader();
      try {
        while (true) {
          const result = await reader.read();
          if (result.done) return;
          yield result.value;
        }
      } finally {
        reader.releaseLock();
      }
    },
  };
}

function temporaryRoot(): string {
  const root = mkdtempSync(join(tmpdir(), "mycellios-artifact-swarm-"));
  temporaryDirectories.push(root);
  return root;
}

function digest(value: Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}
