import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { validateGitHubReleaseClaims } from "../src/coordinator/github-oidc.js";
import type { CoordinatorRuntime } from "../src/coordinator/server.js";
import { createCoordinator } from "../src/coordinator/server.js";
import { storeReleaseChunk } from "../src/coordinator/release-upload.js";

const directories: string[] = [];
const runtimes: CoordinatorRuntime[] = [];

afterEach(async () => {
  await Promise.all(runtimes.splice(0).map((runtime) => runtime.close()));
  for (const directory of directories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

describe("release uploads", () => {
  it("assembles checksum-verified chunks atomically", async () => {
    const root = temporaryDirectory();
    const content = Buffer.from("automatic desktop update");
    const first = content.subarray(0, 10);
    const second = content.subarray(10);
    const fileSha256 = sha256(content);

    const pending = await storeReleaseChunk({
      root,
      channel: "updates",
      fileName: "latest.json",
      metadata: {
        chunkIndex: 0,
        chunkCount: 2,
        chunkSha256: sha256(first),
        fileSha256,
        fileSize: content.length,
      },
      body: first,
    });
    expect(pending.complete).toBe(false);

    const published = await storeReleaseChunk({
      root,
      channel: "updates",
      fileName: "latest.json",
      metadata: {
        chunkIndex: 1,
        chunkCount: 2,
        chunkSha256: sha256(second),
        fileSha256,
        fileSize: content.length,
      },
      body: second,
    });
    expect(published.complete).toBe(true);
    expect(readFileSync(join(root, "latest.json"))).toEqual(content);
  });

  it("accepts authenticated Actions uploads even when the mesh uses another token", async () => {
    const root = temporaryDirectory();
    const updates = join(root, "updates");
    const landing = join(root, "landing");
    const downloads = join(root, "downloads");
    mkdirSync(downloads);
    const content = Buffer.from('{"version":"0.2.12"}\n');
    const digest = sha256(content);
    const runtime = await createCoordinator(
      {
        host: "127.0.0.1",
        port: 0,
        databasePath: ":memory:",
        requestTimeoutMs: 1_000,
        networkToken: "mesh-token",
        desktopUpdatesPath: updates,
        landingAssetsPath: landing,
        releaseDownloadsPath: downloads,
      },
      {
        logger: false,
        releaseTokenVerifier: async (token) => {
          expect(token).toBe("actions-token");
        },
      },
    );
    runtimes.push(runtime);

    const response = await runtime.app.inject({
      method: "PUT",
      url: "/internal/v1/releases/updates/latest.json",
      headers: {
        authorization: "Bearer actions-token",
        "content-type": "application/octet-stream",
        "x-chunk-index": "0",
        "x-chunk-count": "1",
        "x-chunk-sha256": digest,
        "x-file-sha256": digest,
        "x-file-size": String(content.length),
      },
      payload: content,
    });

    expect(response.statusCode).toBe(201);
    expect(response.json()).toMatchObject({ complete: true, fileSha256: digest });
    expect(readFileSync(join(updates, "latest.json"))).toEqual(content);

    const installer = Buffer.from("verified installer");
    const installerDigest = sha256(installer);
    const downloadResponse = await runtime.app.inject({
      method: "PUT",
      url: "/internal/v1/releases/downloads/mycellios-linux-x64.deb",
      headers: {
        authorization: "Bearer actions-token",
        "content-type": "application/octet-stream",
        "x-chunk-index": "0",
        "x-chunk-count": "1",
        "x-chunk-sha256": installerDigest,
        "x-file-sha256": installerDigest,
        "x-file-size": String(installer.length),
      },
      payload: installer,
    });
    expect(downloadResponse.statusCode).toBe(201);
    expect(readFileSync(join(downloads, "mycellios-linux-x64.deb"))).toEqual(installer);
    const publicDownload = await runtime.app.inject({
      method: "GET",
      url: "/downloads/mycellios-linux-x64.deb",
    });
    expect(publicDownload.statusCode).toBe(200);
    expect(publicDownload.rawPayload).toEqual(installer);
  });

  it("restricts OIDC claims to this repository, workflow and release refs", () => {
    expect(validateGitHubReleaseClaims({
      repository: "tych0s/mycellios",
      ref: "refs/heads/main",
      sha: "a".repeat(40),
      workflow_ref: "tych0s/mycellios/.github/workflows/desktop-build.yml@refs/heads/main",
      event_name: "push",
    })).toMatchObject({ ref: "refs/heads/main", sha: "a".repeat(40) });

    expect(() => validateGitHubReleaseClaims({
      repository: "attacker/fork",
      ref: "refs/heads/main",
      sha: "a".repeat(40),
      workflow_ref: "attacker/fork/.github/workflows/desktop-build.yml@refs/heads/main",
      event_name: "push",
    })).toThrow("release_repository_not_allowed");

    expect(validateGitHubReleaseClaims({
      repository: "tych0s/mycellios",
      ref: "refs/heads/main",
      sha: "b".repeat(40),
      workflow_ref:
        "tych0s/mycellios/.github/workflows/publish-existing-release.yml@refs/heads/main",
      event_name: "workflow_dispatch",
    })).toMatchObject({ ref: "refs/heads/main", sha: "b".repeat(40) });
  });
});

function temporaryDirectory(): string {
  const directory = mkdtempSync(join(tmpdir(), "mycellios-release-upload-"));
  directories.push(directory);
  return directory;
}

function sha256(value: Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}
