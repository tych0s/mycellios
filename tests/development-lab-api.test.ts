import { generateKeyPairSync } from "node:crypto";
import { type AddressInfo } from "node:net";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  buildComponentUpdateManifest,
  signComponentUpdateManifest,
} from "../src/contracts/component-update-manifest.js";
import {
  COMPONENT_UPDATE_BOOTSTRAP_MIN_VERSION,
  MYCELLIOS_RUNTIME_ABI,
} from "../src/update/runtime-compatibility.js";
import {
  COMPONENT_UPDATE_PUBLIC_KEY_SCHEMA,
  PREPARED_COMPONENT_RELEASE_SCHEMA,
  publishComponentRelease,
  type ComponentUpdateEnrollment,
  type PreparedComponentRelease,
} from "../src/update/component-update-publisher.js";
import { buildComponentFilesPackage } from "../src/update/component-files.js";
import { ComponentUpdateManager } from "../src/update/component-update-manager.js";
import {
  createCoordinator,
  type CoordinatorRuntime,
} from "../src/coordinator/server.js";

const roots: string[] = [];
const runtimes: CoordinatorRuntime[] = [];

afterEach(async () => {
  await Promise.all(runtimes.splice(0).map((runtime) => runtime.close()));
  await Promise.all(
    roots.splice(0).map((root) => rm(root, { recursive: true, force: true })),
  );
});

describe("development lab HTTP API", () => {
  it("creates, redeems once, publishes and applies an isolated signed runtime", async () => {
    const root = await mkdtemp(join(tmpdir(), "mycellios-dev-lab-api-"));
    roots.push(root);
    const adminToken = "development-lab-admin-token";
    const runtime = await createCoordinator({
      host: "127.0.0.1",
      port: 0,
      databasePath: join(root, "coordinator.db"),
      requestTimeoutMs: 120_000,
      componentUpdatesPath: join(root, "component-updates"),
      modelAdminToken: adminToken,
    });
    runtimes.push(runtime);
    await runtime.app.listen({ host: "127.0.0.1", port: 0 });
    const address = runtime.app.server.address() as AddressInfo;
    const coordinator = `http://127.0.0.1:${address.port}`;
    const { privateKey, publicKey } = generateKeyPairSync("ed25519");
    const enrollment: ComponentUpdateEnrollment = {
      schema: COMPONENT_UPDATE_PUBLIC_KEY_SCHEMA,
      channel: "dev",
      keyId: "mycellios-dev-two-home-test",
      spki: publicKey
        .export({ format: "der", type: "spki" })
        .toString("base64url"),
    };

    const unauthorized = await fetch(
      `${coordinator}/public/v1/admin/development-labs`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          keyId: enrollment.keyId,
          spki: enrollment.spki,
        }),
      },
    );
    expect(unauthorized.status).toBe(401);

    const createResponse = await fetch(
      `${coordinator}/public/v1/admin/development-labs`,
      {
        method: "POST",
        headers: {
          authorization: `Bearer ${adminToken}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          name: "Physical two-home candidate",
          keyId: enrollment.keyId,
          spki: enrollment.spki,
          invitationTtlSeconds: 600,
        }),
      },
    );
    expect(createResponse.status).toBe(201);
    const created = await createResponse.json() as {
      data: {
        labId: string;
        feedBaseUrl: string;
        invitation: { token: string; expiresAt: string };
      };
    };
    expect(created.data.feedBaseUrl).toBe(
      `${coordinator}/development-labs/${created.data.labId}/`,
    );

    const redeemInvitation = () => fetch(`${coordinator}/development-labs/v1/join`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        labId: created.data.labId,
        token: created.data.invitation.token,
      }),
    });
    const joined = await redeemInvitation();
    expect(joined.status).toBe(200);
    await expect(joined.json()).resolves.toMatchObject({
      data: {
        labId: created.data.labId,
        keyId: enrollment.keyId,
        spki: enrollment.spki,
        feedBaseUrl: created.data.feedBaseUrl,
      },
    });
    expect((await redeemInvitation()).status).toBe(410);

    const prepared = await preparedRelease({
      root,
      feedBaseUrl: created.data.feedBaseUrl,
      enrollment,
      privateKey,
    });
    const published = await publishComponentRelease({
      releaseDirectory: prepared.releaseDirectory,
      coordinatorUrl: created.data.feedBaseUrl,
      target: prepared.target,
      adminToken,
    });
    expect(published.commitPhase).toBe("readback-verified");

    let preparedRoot: string | null = null;
    let activatedRoot: string | null = null;
    const manager = new ComponentUpdateManager({
      storageRoot: join(root, "receiver"),
      feedBaseUrl: created.data.feedBaseUrl,
      channel: "dev",
      pinnedKeys: [{
        keyId: enrollment.keyId,
        spki: enrollment.spki,
      }],
      bootstrapVersion: "0.2.70",
      workerProtocol: { min: 1, max: 1 },
      runtimeAbi: MYCELLIOS_RUNTIME_ABI,
      platform: prepared.target.platform,
      arch: prepared.target.arch,
      isIdle: () => true,
      onPrepare: async ({ stagedComponentRoots }) => {
        preparedRoot = stagedComponentRoots["python-product"] ?? null;
      },
      onActivate: async ({ stagedComponentRoots }) => {
        activatedRoot = stagedComponentRoots["python-product"] ?? null;
      },
    });
    await expect(manager.checkAndApply()).resolves.toMatchObject({
      state: "applied",
      changedComponents: ["python-product"],
    });
    expect(preparedRoot).toBeTruthy();
    expect(activatedRoot).toBe(preparedRoot);

    const revoked = await fetch(
      `${created.data.feedBaseUrl}public/v1/admin/revoke`,
      {
        method: "POST",
        headers: { authorization: `Bearer ${adminToken}` },
      },
    );
    expect(revoked.status).toBe(200);
    expect(await fetch(
      `${created.data.feedBaseUrl}updates/v1/dev/`
      + `${prepared.target.platform}/${prepared.target.arch}/manifest.json`,
    ).then((response) => response.status)).toBe(410);
  }, 30_000);
});

async function preparedRelease(input: {
  root: string;
  feedBaseUrl: string;
  enrollment: ComponentUpdateEnrollment;
  privateKey: ReturnType<typeof generateKeyPairSync>["privateKey"];
}): Promise<{
  releaseDirectory: string;
  target: PreparedComponentRelease["target"];
}> {
  const source = join(input.root, "python-source");
  const releaseDirectory = join(input.root, "prepared-release");
  await mkdir(source, { recursive: true });
  await mkdir(releaseDirectory, { recursive: true });
  await writeFile(join(source, "runtime.py"), "READY = True\n");
  const artifact = buildComponentFilesPackage(source);
  const platform = process.platform as "win32" | "linux" | "darwin";
  const arch = process.arch as "x64" | "arm64";
  const target = { channel: "dev" as const, platform, arch };
  const manifest = signComponentUpdateManifest(
    buildComponentUpdateManifest({
      channel: "dev",
      sequence: 1,
      revision: "c".repeat(40),
      provenance: {
        baseRevision: "c".repeat(40),
        sourceTreeDirty: true,
        sourceTreeDigest: artifact.filesManifestSha256,
      },
      sourceId: `sha256:${"d".repeat(64)}`,
      compatibility: {
        workerProtocol: { min: 1, max: 1 },
        runtimeAbi: MYCELLIOS_RUNTIME_ABI,
        minBootstrapVersion: COMPONENT_UPDATE_BOOTSTRAP_MIN_VERSION,
      },
      components: [{
        id: "python-product",
        version: "0.2.70-dev.1",
        platform,
        arch,
        artifact: {
          url:
            `${input.feedBaseUrl}updates/v1/artifacts/`
            + artifact.artifactSha256.slice("sha256:".length),
          sha256: artifact.artifactSha256,
          bytes: artifact.packageBytes.length,
          format: artifact.format,
          filesManifestSha256: artifact.filesManifestSha256,
        },
        requirements: {
          backend: "any",
          driver: null,
          runtimeAbi: "mycellios-distribution-runtime/4",
          workerProtocol: { min: 1, max: 1 },
          dependencies: [],
        },
        restartScope: "runtime",
      }],
    }),
    { keyId: input.enrollment.keyId, privateKey: input.privateKey },
  );
  const release: PreparedComponentRelease = {
    schema: PREPARED_COMPONENT_RELEASE_SCHEMA,
    target,
    artifactFile: "artifact.blob",
    manifestFile: "manifest.json",
    enrollmentFile: "enrollment.json",
    manifestId: manifest.manifestId,
    sequence: manifest.sequence,
    provenance: {
      baseRevision: "c".repeat(40),
      sourceTreeDirty: true,
      releaseRevision: "c".repeat(40),
      filesManifestSha256: artifact.filesManifestSha256,
    },
  };
  await Promise.all([
    writeFile(join(releaseDirectory, release.artifactFile), artifact.packageBytes),
    writeFile(join(releaseDirectory, release.manifestFile), JSON.stringify(manifest)),
    writeFile(
      join(releaseDirectory, release.enrollmentFile),
      JSON.stringify(input.enrollment),
    ),
    writeFile(join(releaseDirectory, "release.json"), JSON.stringify(release)),
  ]);
  return { releaseDirectory, target };
}
