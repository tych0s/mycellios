import {
  generateKeyPairSync,
} from "node:crypto";
import {
  mkdirSync,
  mkdtempSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  buildComponentUpdateManifest,
  signComponentUpdateManifest,
} from "../src/contracts/component-update-manifest.js";
import {
  createCoordinator,
  type CoordinatorRuntime,
} from "../src/coordinator/server.js";
import { SupabaseAuthService } from "../src/coordinator/supabase-auth.js";
import type { NativeRuntimeBuildMetadata } from "../src/core/native-build-identity.js";
import { buildComponentFilesPackage } from "../src/update/component-files.js";

const directories: string[] = [];
const runtimes: CoordinatorRuntime[] = [];

afterEach(async () => {
  await Promise.all(runtimes.splice(0).map((runtime) => runtime.close()));
  for (const directory of directories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("component update API", () => {
  it("requires recent MFA for account-backed artifact and channel publication", async () => {
    const auth = new SupabaseAuthService(
      "https://accounts.example.test",
      "service-role",
      (async (input) => String(input).includes("/auth/v1/user")
        ? Response.json({ id: "release-owner", email: "owner@example.test" })
        : String(input).includes("/rest/v1/network_members")
          ? Response.json([{ role: "owner" }])
          : new Response(null, { status: 404 })) as typeof fetch,
    );
    const runtime = await createCoordinator({
      host: "127.0.0.1", port: 0, databasePath: ":memory:", requestTimeoutMs: 1_000,
    }, { logger: false, supabaseAuthService: auth });
    runtimes.push(runtime);
    const token = (authTime: number) => `header.${Buffer.from(JSON.stringify({ aal: "aal2", auth_time: authTime })).toString("base64url")}.signature`;
    const stale = await runtime.app.inject({
      method: "PUT", url: `/public/v1/admin/component-updates/artifacts/${"a".repeat(64)}`,
      headers: { authorization: `Bearer ${token(Math.floor(Date.now() / 1_000) - 301)}`, "content-type": "application/octet-stream" }, payload: Buffer.from("artifact"),
    });
    expect(stale.statusCode).toBe(403);
    expect(stale.json()).toMatchObject({ error: { code: "recent_aal2_reauthentication_required" } });
    const staleChannel = await runtime.app.inject({
      method: "PUT", url: "/public/v1/admin/component-updates/stable/linux/x64/manifest",
      headers: { authorization: `Bearer ${token(Math.floor(Date.now() / 1_000) - 301)}`, "content-type": "application/json" }, payload: {},
    });
    expect(staleChannel.statusCode).toBe(403);
    expect(staleChannel.json()).toMatchObject({ error: { code: "recent_aal2_reauthentication_required" } });
    const recent = await runtime.app.inject({
      method: "PUT", url: `/public/v1/admin/component-updates/artifacts/${"a".repeat(64)}`,
      headers: { authorization: `Bearer ${token(Math.floor(Date.now() / 1_000))}`, "content-type": "application/octet-stream" }, payload: Buffer.from("artifact"),
    });
    expect(recent.statusCode).toBe(503);
    expect(recent.json()).toMatchObject({ error: { code: "component_updates_not_configured" } });
  });

  it("publishes an authenticated artifact and exposes one coherent public manifest", async () => {
    const root = mkdtempSync(join(tmpdir(), "mycellios-component-api-"));
    directories.push(root);
    const source = join(root, "source");
    mkdirSync(source);
    writeFileSync(join(source, "runtime.py"), "READY = True\n");
    const artifact = buildComponentFilesPackage(source);
    const { privateKey, publicKey } = generateKeyPairSync("ed25519");
    const keyId = "mycellios-dev-api-test";
    const manifest = signComponentUpdateManifest(
      buildComponentUpdateManifest({
        channel: "dev",
        sequence: 1,
        revision: "a".repeat(40),
        provenance: {
          baseRevision: "a".repeat(40),
          sourceTreeDirty: false,
          sourceTreeDigest: artifact.filesManifestSha256,
        },
        sourceId: `sha256:${"b".repeat(64)}`,
        compatibility: {
          workerProtocol: { min: 1, max: 1 },
          runtimeAbi: "mycellios-distribution-runtime/4",
          minBootstrapVersion: "0.2.70",
        },
        components: [
          {
            id: "python-product",
            version: "0.2.70-dev.1",
            platform: "win32",
            arch: "x64",
            artifact: {
              url:
                "https://updates.example.test/updates/v1/artifacts/"
                + artifact.artifactSha256.slice("sha256:".length),
              sha256: artifact.artifactSha256,
              bytes: artifact.packageBytes.length,
              format: "json-gzip-v1",
              filesManifestSha256: artifact.filesManifestSha256,
            },
            requirements: {
              backend: "any",
              driver: null,
              runtimeAbi: "mycellios-distribution-runtime/4",
              workerProtocol: { min: 3, max: 4 },
              dependencies: [],
            },
            restartScope: "runtime",
          },
        ],
      }),
      { keyId, privateKey },
    );
    const runtime = await createCoordinator(
      {
        host: "127.0.0.1",
        port: 0,
        databasePath: ":memory:",
        requestTimeoutMs: 1_000,
        modelAdminToken: "admin-test-token",
        componentUpdatesPath: join(root, "component-store"),
        componentUpdatePinnedKeys: {
          dev: [{
            keyId,
            spki: publicKey
              .export({ format: "der", type: "spki" })
              .toString("base64url"),
          }],
        },
      },
      {
        logger: false,
        runtimeMetadata: runtimeMetadata(root),
      },
    );
    runtimes.push(runtime);
    const authorization = { authorization: "Bearer admin-test-token" };
    const digest = artifact.artifactSha256.slice("sha256:".length);

    const unauthorizedUpload = await runtime.app.inject({
      method: "PUT",
      url: `/public/v1/admin/component-updates/artifacts/${digest}`,
      headers: {
        "content-type": "application/octet-stream",
      },
      payload: artifact.packageBytes,
    });
    expect(unauthorizedUpload.statusCode).toBe(401);

    const artifactUpload = await runtime.app.inject({
      method: "PUT",
      url: `/public/v1/admin/component-updates/artifacts/${digest}`,
      headers: {
        ...authorization,
        "content-type": "application/octet-stream",
      },
      payload: artifact.packageBytes,
    });
    expect(artifactUpload.statusCode).toBe(201);

    const manifestUpload = await runtime.app.inject({
      method: "PUT",
      url: "/public/v1/admin/component-updates/dev/win32/x64/manifest",
      headers: {
        ...authorization,
        "content-type": "application/json",
      },
      payload: manifest,
    });
    expect(manifestUpload.statusCode).toBe(201);

    const published = await runtime.app.inject({
      method: "GET",
      url: "/updates/v1/dev/win32/x64/manifest.json",
    });
    expect(published.statusCode).toBe(200);
    expect(published.headers["cache-control"]).toContain("no-store");
    expect(published.json()).toMatchObject({
      manifestId: manifest.manifestId,
      sequence: 1,
    });

    const downloaded = await runtime.app.inject({
      method: "GET",
      url: `/updates/v1/artifacts/${digest}`,
    });
    expect(downloaded.statusCode).toBe(200);
    expect(downloaded.rawPayload).toEqual(artifact.packageBytes);
    expect(downloaded.headers["cache-control"]).toContain("immutable");
  });

  it("does not publish a manifest before its referenced artifact exists", async () => {
    const root = mkdtempSync(join(tmpdir(), "mycellios-component-api-"));
    directories.push(root);
    const { privateKey, publicKey } = generateKeyPairSync("ed25519");
    const keyId = "mycellios-dev-api-test";
    const manifest = signComponentUpdateManifest(
      buildComponentUpdateManifest({
        channel: "dev",
        sequence: 1,
        revision: "a".repeat(40),
        provenance: {
          baseRevision: "a".repeat(40),
          sourceTreeDirty: false,
          sourceTreeDigest: `sha256:${"d".repeat(64)}`,
        },
        sourceId: `sha256:${"b".repeat(64)}`,
        compatibility: {
          workerProtocol: { min: 1, max: 1 },
          runtimeAbi: "mycellios-distribution-runtime/4",
          minBootstrapVersion: "0.2.70",
        },
        components: [
          {
            id: "python-product",
            version: "0.2.70-dev.1",
            platform: "win32",
            arch: "x64",
            artifact: {
              url: "https://updates.example.test/missing",
              sha256: `sha256:${"c".repeat(64)}`,
              bytes: 100,
              format: "json-gzip-v1",
              filesManifestSha256: `sha256:${"d".repeat(64)}`,
            },
            requirements: {
              backend: "any",
              driver: null,
              runtimeAbi: "mycellios-distribution-runtime/4",
              workerProtocol: { min: 3, max: 4 },
              dependencies: [],
            },
            restartScope: "runtime",
          },
        ],
      }),
      { keyId, privateKey },
    );
    const runtime = await createCoordinator(
      {
        host: "127.0.0.1",
        port: 0,
        databasePath: ":memory:",
        requestTimeoutMs: 1_000,
        modelAdminToken: "admin-test-token",
        componentUpdatesPath: join(root, "component-store"),
        componentUpdatePinnedKeys: {
          dev: [{
            keyId,
            spki: publicKey
              .export({ format: "der", type: "spki" })
              .toString("base64url"),
          }],
        },
      },
      {
        logger: false,
        runtimeMetadata: runtimeMetadata(root),
      },
    );
    runtimes.push(runtime);

    const response = await runtime.app.inject({
      method: "PUT",
      url: "/public/v1/admin/component-updates/dev/win32/x64/manifest",
      headers: {
        authorization: "Bearer admin-test-token",
        "content-type": "application/json",
      },
      payload: manifest,
    });
    expect(response.statusCode).toBe(404);
    expect(response.json()).toMatchObject({
      error: { code: "component_release_manifest_artifact_missing" },
    });
    expect(
      (
        await runtime.app.inject({
          method: "GET",
          url: "/updates/v1/dev/win32/x64/manifest.json",
        })
      ).statusCode,
    ).toBe(404);
  });
});

function runtimeMetadata(root: string): NativeRuntimeBuildMetadata {
  return {
    root,
    version: "0.2.70",
    revision: null,
    buildIdentity: null,
  };
}
