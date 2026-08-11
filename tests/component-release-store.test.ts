import {
  createHash,
  generateKeyPairSync,
  type KeyObject,
} from "node:crypto";
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  buildComponentUpdateManifest,
  signComponentUpdateManifest,
  verifyComponentUpdateManifest,
  type ComponentUpdateManifest,
} from "../src/contracts/component-update-manifest.js";
import {
  ComponentReleaseStore,
  type ComponentReleaseTarget,
} from "../src/coordinator/component-release-store.js";

const roots: string[] = [];
const TARGET = {
  channel: "dev",
  platform: "linux",
  arch: "x64",
} as const satisfies ComponentReleaseTarget;

afterEach(() => {
  for (const root of roots.splice(0)) {
    rmSync(root, { recursive: true, force: true });
  }
});

function temporaryRoot(): string {
  const root = mkdtempSync(join(tmpdir(), "mycellios-component-releases-"));
  roots.push(root);
  return root;
}

function digest(body: Uint8Array): `sha256:${string}` {
  return `sha256:${createHash("sha256").update(body).digest("hex")}`;
}

function keys(): {
  privateKey: KeyObject;
  keyId: string;
  spki: string;
} {
  const pair = generateKeyPairSync("ed25519");
  return {
    privateKey: pair.privateKey,
    keyId: "dev-release-key",
    spki: pair.publicKey
      .export({ format: "der", type: "spki" })
      .toString("base64url"),
  };
}

function manifest(
  body: Uint8Array,
  signing: ReturnType<typeof keys>,
  input: {
    sequence?: number;
    revision?: string;
    channel?: "dev" | "stable";
    platform?: "any" | "win32" | "linux" | "darwin";
    arch?: "any" | "x64" | "arm64";
  } = {},
): ComponentUpdateManifest {
  return signComponentUpdateManifest(
    buildComponentUpdateManifest({
      channel: input.channel ?? "dev",
      sequence: input.sequence ?? 1,
      revision: input.revision ?? "a".repeat(40),
      provenance: {
        baseRevision: input.revision ?? "a".repeat(40),
        sourceTreeDirty: false,
        sourceTreeDigest: digest(body),
      },
      sourceId: `sha256:${"b".repeat(64)}`,
      compatibility: {
        workerProtocol: { min: 3, max: 4 },
        runtimeAbi: "mycellios-distribution-runtime/4",
        minBootstrapVersion: "1.0.0",
      },
      components: [
        {
          id: "worker-runtime",
          version: "0.2.70-dev.1",
          platform: input.platform ?? "linux",
          arch: input.arch ?? "x64",
          artifact: {
            url: "https://updates.example.test/worker-runtime.json.gz",
            sha256: digest(body),
            bytes: body.byteLength,
            format: "json-gzip-v1",
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
    {
      keyId: signing.keyId,
      privateKey: signing.privateKey,
    },
  );
}

function pinnedStore(
  root: string,
  signing: ReturnType<typeof keys>,
): ComponentReleaseStore {
  return new ComponentReleaseStore({
    root,
    verification: {
      pinnedKeys: {
        dev: { keyId: signing.keyId, spki: signing.spki },
      },
    },
  });
}

describe("ComponentReleaseStore", () => {
  it("stores immutable artifacts by content identity using streamed staging", async () => {
    const root = temporaryRoot();
    const store = pinnedStore(root, keys());
    const first = Buffer.from("component ");
    const second = Buffer.from("runtime");
    const body = Buffer.concat([first, second]);
    async function* chunks(): AsyncGenerator<Uint8Array> {
      yield first;
      yield second;
    }

    await expect(
      store.putArtifact({
        sha256: digest(body),
        bytes: body.length,
        body: chunks(),
      }),
    ).resolves.toMatchObject({ alreadyStored: false });
    await expect(
      store.putArtifact({
        sha256: digest(body),
        bytes: body.length,
        body,
      }),
    ).resolves.toMatchObject({ alreadyStored: true });
    await expect(
      store.readArtifact({ sha256: digest(body), bytes: body.length }),
    ).resolves.toEqual(body);

    const artifactDirectory = join(
      root,
      "artifacts",
      "sha256",
      digest(body).slice(7, 9),
    );
    expect(readdirSync(artifactDirectory)).toEqual([
      `${digest(body).slice(7)}.blob`,
    ]);
  });

  it("rejects artifact byte and hash mismatches without publishing partial data", async () => {
    const root = temporaryRoot();
    const store = pinnedStore(root, keys());
    const body = Buffer.from("wrong content");
    const expected = digest(Buffer.from("expected content"));

    await expect(
      store.putArtifact({
        sha256: expected,
        bytes: body.length,
        body,
      }),
    ).rejects.toMatchObject({
      code: "component_release_artifact_hash_mismatch",
    });
    await expect(
      store.readArtifact({ sha256: expected }),
    ).rejects.toMatchObject({
      code: "component_release_artifact_missing",
    });
  });

  it("atomically publishes and safely reads a signed target manifest", async () => {
    const root = temporaryRoot();
    const signing = keys();
    const store = pinnedStore(root, signing);
    const body = Buffer.from("verified runtime payload");
    const release = manifest(body, signing);
    await store.putArtifact({
      sha256: digest(body),
      bytes: body.length,
      body,
    });

    await expect(
      store.publishManifest({ target: TARGET, manifest: release }),
    ).resolves.toMatchObject({
      alreadyPublished: false,
      previousManifestId: null,
    });
    await expect(store.readPublishedManifest(TARGET)).resolves.toEqual(release);
    await expect(
      store.publishManifest({ target: TARGET, manifest: release }),
    ).resolves.toMatchObject({
      alreadyPublished: true,
      previousManifestId: release.manifestId,
    });

    const channelDirectory = join(root, "channels", "dev", "linux", "x64");
    expect(readdirSync(channelDirectory)).toEqual(["manifest.json"]);
    expect(
      JSON.parse(readFileSync(join(channelDirectory, "manifest.json"), "utf8")),
    ).toEqual(release);
  });

  it("garbage-collects old unreferenced blobs without deleting the public manifest artifact", async () => {
    const root = temporaryRoot();
    const signing = keys();
    const store = pinnedStore(root, signing);
    const referencedBody = Buffer.from("referenced");
    await store.putArtifact({
      sha256: digest(referencedBody),
      bytes: referencedBody.length,
      body: referencedBody,
    });
    await store.publishManifest({
      target: TARGET,
      manifest: manifest(referencedBody, signing),
    });
    for (const value of ["old-a", "old-b", "old-c"]) {
      const body = Buffer.from(value);
      await store.putArtifact({
        sha256: digest(body),
        bytes: body.length,
        body,
      });
    }

    await expect(store.collectGarbage({
      retainUnreferenced: 1,
      minimumAgeMs: 0,
    })).resolves.toMatchObject({
      referencedArtifacts: 1,
      retainedUnreferencedArtifacts: 1,
      removedArtifacts: 2,
    });
    await expect(store.readArtifact({
      sha256: digest(referencedBody),
    })).resolves.toEqual(referencedBody);
  });

  it("does not replace a good manifest when a referenced artifact is absent or corrupt", async () => {
    const root = temporaryRoot();
    const signing = keys();
    const store = pinnedStore(root, signing);
    const firstBody = Buffer.from("first payload");
    const first = manifest(firstBody, signing, { sequence: 1 });
    await store.putArtifact({
      sha256: digest(firstBody),
      bytes: firstBody.length,
      body: firstBody,
    });
    await store.publishManifest({ target: TARGET, manifest: first });

    const missingBody = Buffer.from("not uploaded");
    const second = manifest(missingBody, signing, {
      sequence: 2,
      revision: "c".repeat(40),
    });
    await expect(
      store.publishManifest({ target: TARGET, manifest: second }),
    ).rejects.toMatchObject({
      code: "component_release_manifest_artifact_missing",
    });
    await expect(store.readPublishedManifest(TARGET)).resolves.toEqual(first);

    const firstArtifactPath = join(
      root,
      "artifacts",
      "sha256",
      digest(firstBody).slice(7, 9),
      `${digest(firstBody).slice(7)}.blob`,
    );
    writeFileSync(firstArtifactPath, Buffer.alloc(firstBody.length, 0x78));
    await expect(store.readPublishedManifest(TARGET)).rejects.toMatchObject({
      code: "component_release_manifest_artifact_invalid",
    });
  });

  it("enforces signatures, target isolation, monotonic sequences, and equivocation rejection", async () => {
    const root = temporaryRoot();
    const trusted = keys();
    const untrusted = keys();
    const store = pinnedStore(root, trusted);
    const body = Buffer.from("shared artifact");
    await store.putArtifact({
      sha256: digest(body),
      bytes: body.length,
      body,
    });
    const current = manifest(body, trusted, { sequence: 2 });
    await store.publishManifest({ target: TARGET, manifest: current });

    await expect(
      store.publishManifest({
        target: TARGET,
        manifest: manifest(body, untrusted, { sequence: 3 }),
      }),
    ).rejects.toMatchObject({
      code: "component_release_manifest_signature_rejected",
    });
    await expect(
      store.publishManifest({
        target: TARGET,
        manifest: manifest(body, trusted, {
          sequence: 1,
          revision: "c".repeat(40),
        }),
      }),
    ).rejects.toMatchObject({
      code: "component_release_manifest_rollback_rejected",
    });
    await expect(
      store.publishManifest({
        target: TARGET,
        manifest: manifest(body, trusted, {
          sequence: 2,
          revision: "d".repeat(40),
        }),
      }),
    ).rejects.toMatchObject({
      code: "component_release_manifest_sequence_conflict",
    });
    await expect(
      store.publishManifest({
        target: TARGET,
        manifest: manifest(body, trusted, {
          sequence: 3,
          platform: "win32",
        }),
      }),
    ).rejects.toMatchObject({
      code: "component_release_manifest_target_empty",
    });
    await expect(
      store.publishManifest({
        target: { ...TARGET, channel: "stable" },
        manifest: current,
      }),
    ).rejects.toMatchObject({
      code: "component_release_manifest_channel_mismatch",
    });
  });

  it("supports an injected verifier without accepting its failures", async () => {
    const root = temporaryRoot();
    const signing = keys();
    const body = Buffer.from("callback verified");
    const release = manifest(body, signing);
    const verifier = vi.fn(
      (candidate: ComponentUpdateManifest) => {
        verifyComponentUpdateManifest(candidate, {
          pinnedKey: { keyId: signing.keyId, spki: signing.spki },
          expectedChannel: "dev",
        });
      },
    );
    const store = new ComponentReleaseStore({
      root,
      verification: { verifyManifest: verifier },
    });
    await store.putArtifact({
      sha256: digest(body),
      bytes: body.length,
      body,
    });
    await store.publishManifest({ target: TARGET, manifest: release });
    expect(verifier).toHaveBeenCalledWith(
      release,
      expect.objectContaining({ target: TARGET, minimumSequence: null }),
    );

    const rejecting = new ComponentReleaseStore({
      root: temporaryRoot(),
      verification: {
        verifyManifest: () => {
          throw new Error("untrusted");
        },
      },
    });
    await expect(
      rejecting.publishManifest({ target: TARGET, manifest: release }),
    ).rejects.toMatchObject({
      code: "component_release_manifest_signature_rejected",
    });
  });

  it("returns typed errors for malformed identities and corrupted published state", async () => {
    const root = temporaryRoot();
    const signing = keys();
    const store = pinnedStore(root, signing);

    await expect(
      store.readArtifact({ sha256: "../../escape" }),
    ).rejects.toMatchObject({
      code: "component_release_artifact_identity_invalid",
    });
    await expect(store.readPublishedManifest(TARGET)).resolves.toBeNull();

    const channelDirectory = join(root, "channels", "dev", "linux", "x64");
    const channelFile = join(channelDirectory, "manifest.json");
    await store.putArtifact({
      sha256: digest(Buffer.from("seed")),
      bytes: 4,
      body: Buffer.from("seed"),
    });
    mkdirSync(channelDirectory, { recursive: true });
    writeFileSync(channelFile, "{not-json");
    await expect(store.readPublishedManifest(TARGET)).rejects.toMatchObject({
      code: "component_release_manifest_corrupt",
    });
  });
});
