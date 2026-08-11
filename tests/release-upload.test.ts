import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  utimesSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  validateGitHubReleaseClaims,
  type GitHubReleaseClaims,
} from "../src/coordinator/github-oidc.js";
import type { NativeRuntimeBuildMetadata } from "../src/core/native-build-identity.js";
import type { CoordinatorRuntime } from "../src/coordinator/server.js";
import { createCoordinator } from "../src/coordinator/server.js";
import {
  buildReleaseTransactionManifest,
  NativeReleaseTransactionStore,
  storeReleaseChunk,
  type ReleaseAssetChannel,
  type ReleaseAssetEvidence,
  type ReleaseTransactionIdentity,
} from "../src/coordinator/release-upload.js";

const directories: string[] = [];
const runtimes: CoordinatorRuntime[] = [];
const RUNTIME_REVISION = "a".repeat(40);
const RUNTIME_SOURCE_ID = `sha256:${"1".repeat(64)}` as const;
const RELEASE_VERSION = "0.2.19";

afterEach(async () => {
  await Promise.all(runtimes.splice(0).map((runtime) => runtime.close()));
  for (const directory of directories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

describe("release uploads", () => {
  it("assembles checksum-verified chunks atomically", async () => {
    const root = temporaryDirectory();
    const content = Buffer.from("automatic native node update");
    const first = content.subarray(0, 10);
    const second = content.subarray(10);
    const fileSha256 = sha256(content);

    const pending = await storeReleaseChunk({
      root,
      channel: "updates",
      fileName: "mycellios-node-latest.json",
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
      fileName: "mycellios-node-latest.json",
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
    expect(readFileSync(join(root, "mycellios-node-latest.json"))).toEqual(content);
  });

  it("publishes the exact four assets in one authenticated commit and rolls back idempotently", async () => {
    const root = temporaryDirectory();
    const updates = join(root, "updates");
    const landing = join(root, "landing");
    const downloads = join(root, "downloads");
    mkdirSync(updates);
    mkdirSync(landing);
    mkdirSync(downloads);
    const previousFeed = Buffer.from("previous node feed");
    const previousLinuxPackage = Buffer.from("previous Linux node package");
    writeFileSync(join(updates, "mycellios-node-latest.json"), previousFeed);
    writeFileSync(join(downloads, "mycellios-node-linux-x64.tar.gz"), previousLinuxPackage);
    writeFileSync(join(landing, "index.html"), "<main>landing</main>");
    const runtime = await createCoordinator(
      {
        host: "127.0.0.1",
        port: 0,
        databasePath: ":memory:",
        requestTimeoutMs: 1_000,
        networkToken: "mesh-token",
        nodeUpdatesPath: updates,
        landingAssetsPath: landing,
        releaseDownloadsPath: downloads,
      },
      {
        logger: false,
        runtimeMetadata: releaseRuntimeMetadata(root, RUNTIME_REVISION),
        releaseTransactionRoot: join(root, "release-transactions"),
        releaseTokenVerifier: async (token) => {
          expect(token).toBe("actions-token");
          return releaseClaims(RUNTIME_REVISION);
        },
      },
    );
    runtimes.push(runtime);

    expect((await runtime.app.inject({
      method: "GET",
      url: "/updates/node/mycellios-node-latest.json",
    })).rawPayload).toEqual(previousFeed);
    expect((await runtime.app.inject({
      method: "GET",
      url: "/downloads/mycellios-node-linux-x64.tar.gz",
    })).rawPayload).toEqual(previousLinuxPackage);

    const transaction = releaseFixture("transaction-release-0001");
    for (const asset of transaction.assets) {
      const response = await uploadAsset(
        runtime,
        transaction.identity,
        asset.channel,
        asset.fileName,
        asset.content,
      );
      expect(response.statusCode).toBe(201);
    }

    // Completed chunks remain private until the one exact-set commit.
    expect((await runtime.app.inject({
      method: "GET",
      url: "/updates/node/mycellios-node-latest.json",
    })).rawPayload).toEqual(previousFeed);
    expect((await runtime.app.inject({
      method: "GET",
      url: "/downloads/mycellios-node-linux-x64.tar.gz",
    })).rawPayload).toEqual(previousLinuxPackage);

    const commit = await runtime.app.inject({
      method: "POST",
      url: `/internal/v1/releases/transactions/${transaction.identity.transactionId}/commit`,
      headers: { authorization: "Bearer actions-token" },
      payload: transaction.manifest,
    });
    expect(commit.statusCode).toBe(201);
    expect(commit.json()).toMatchObject({
      transactionId: transaction.identity.transactionId,
      releaseId: transaction.manifest.releaseId,
      previousTransactionId: null,
      alreadyCommitted: false,
    });
    const repeatedCommit = await runtime.app.inject({
      method: "POST",
      url: `/internal/v1/releases/transactions/${transaction.identity.transactionId}/commit`,
      headers: { authorization: "Bearer actions-token" },
      payload: transaction.manifest,
    });
    expect(repeatedCommit.statusCode).toBe(201);
    expect(repeatedCommit.json()).toMatchObject({ alreadyCommitted: true });
    expect((await runtime.app.inject({
      method: "GET",
      url: "/updates/node/mycellios-node-latest.json",
    })).rawPayload).toEqual(transaction.byPath.get("updates/mycellios-node-latest.json"));
    expect((await runtime.app.inject({
      method: "GET",
      url: "/downloads/mycellios-node-linux-x64.tar.gz",
    })).rawPayload).toEqual(transaction.byPath.get("downloads/mycellios-node-linux-x64.tar.gz"));

    const deb = transaction.byPath.get("downloads/mycellios-node-linux-x64.tar.gz")!;
    const partial = await runtime.app.inject({
      method: "GET",
      url: "/downloads/mycellios-node-linux-x64.tar.gz",
      headers: { range: "bytes=4-8" },
    });
    expect(partial.statusCode).toBe(206);
    expect(partial.headers["accept-ranges"]).toBe("bytes");
    expect(partial.headers["content-range"]).toBe(`bytes 4-8/${deb.length}`);
    expect(partial.headers["content-length"]).toBe("5");
    expect(partial.rawPayload).toEqual(deb.subarray(4, 9));

    const mismatchedIfRange = await runtime.app.inject({
      method: "GET",
      url: "/downloads/mycellios-node-linux-x64.tar.gz",
      headers: {
        range: "bytes=4-8",
        "if-range": "\"different-release\"",
      },
    });
    expect(mismatchedIfRange.statusCode).toBe(200);
    expect(mismatchedIfRange.rawPayload).toEqual(deb);

    const unsatisfiable = await runtime.app.inject({
      method: "GET",
      url: "/downloads/mycellios-node-linux-x64.tar.gz",
      headers: { range: `bytes=${deb.length}-` },
    });
    expect(unsatisfiable.statusCode).toBe(416);
    expect(unsatisfiable.headers["content-range"]).toBe(`bytes */${deb.length}`);

    const head = await runtime.app.inject({
      method: "HEAD",
      url: "/downloads/mycellios-node-linux-x64.tar.gz",
    });
    expect(head.statusCode).toBe(200);
    expect(head.headers["accept-ranges"]).toBe("bytes");
    expect(head.headers["content-length"]).toBe(String(deb.length));
    expect(head.rawPayload).toHaveLength(0);

    const unauthorizedRollback = await runtime.app.inject({
      method: "POST",
      url: `/internal/v1/releases/transactions/${transaction.identity.transactionId}/rollback`,
      headers: { authorization: "Bearer wrong-token" },
    });
    expect(unauthorizedRollback.statusCode).toBe(401);
    expect((await runtime.app.inject({
      method: "GET",
      url: "/downloads/mycellios-node-linux-x64.tar.gz",
    })).rawPayload).toEqual(transaction.byPath.get("downloads/mycellios-node-linux-x64.tar.gz"));

    const rollback = await runtime.app.inject({
      method: "POST",
      url: `/internal/v1/releases/transactions/${transaction.identity.transactionId}/rollback`,
      headers: { authorization: "Bearer actions-token" },
    });
    expect(rollback.statusCode).toBe(200);
    expect(rollback.json()).toEqual({
      transactionId: transaction.identity.transactionId,
      activeTransactionId: null,
      alreadyRolledBack: false,
    });
    expect((await runtime.app.inject({
      method: "GET",
      url: "/updates/node/mycellios-node-latest.json",
    })).rawPayload).toEqual(previousFeed);
    expect((await runtime.app.inject({
      method: "GET",
      url: "/downloads/mycellios-node-linux-x64.tar.gz",
    })).rawPayload).toEqual(previousLinuxPackage);

    const repeated = await runtime.app.inject({
      method: "POST",
      url: `/internal/v1/releases/transactions/${transaction.identity.transactionId}/rollback`,
      headers: { authorization: "Bearer actions-token" },
    });
    expect(repeated.statusCode).toBe(200);
    expect(repeated.json()).toMatchObject({ alreadyRolledBack: true });
  });

  it("rejects source drift and an incomplete four-file set without changing public assets", async () => {
    const root = temporaryDirectory();
    const updates = join(root, "updates");
    const downloads = join(root, "downloads");
    mkdirSync(updates);
    mkdirSync(downloads);
    const previousFeed = Buffer.from("previous node feed");
    writeFileSync(join(updates, "mycellios-node-latest.json"), previousFeed);
    const runtime = await createCoordinator(
      {
        host: "127.0.0.1",
        port: 0,
        databasePath: ":memory:",
        requestTimeoutMs: 1_000,
        nodeUpdatesPath: updates,
        releaseDownloadsPath: downloads,
      },
      {
        logger: false,
        runtimeMetadata: releaseRuntimeMetadata(root, RUNTIME_REVISION),
        releaseTransactionRoot: join(root, "release-transactions"),
        releaseTokenVerifier: async () => releaseClaims(RUNTIME_REVISION),
      },
    );
    runtimes.push(runtime);
    const transaction = releaseFixture("transaction-release-0002");
    const wrongIdentity = {
      ...transaction.identity,
      sourceId: `sha256:${"2".repeat(64)}` as const,
    };
    const rejectedSource = await uploadAsset(
      runtime,
      wrongIdentity,
      transaction.assets[0]!.channel,
      transaction.assets[0]!.fileName,
      transaction.assets[0]!.content,
    );
    expect(rejectedSource.statusCode).toBe(400);
    expect(rejectedSource.json().error.message).toBe(
      "release_transaction_source_id_mismatch",
    );

    for (const asset of transaction.assets.slice(0, -1)) {
      expect((await uploadAsset(
        runtime,
        transaction.identity,
        asset.channel,
        asset.fileName,
        asset.content,
      )).statusCode).toBe(201);
    }
    const commit = await runtime.app.inject({
      method: "POST",
      url: `/internal/v1/releases/transactions/${transaction.identity.transactionId}/commit`,
      headers: { authorization: "Bearer actions-token" },
      payload: transaction.manifest,
    });
    expect(commit.statusCode).toBe(409);
    expect(commit.json().error.message).toContain(
      "release_transaction_asset_set_mismatch",
    );
    expect((await runtime.app.inject({
      method: "GET",
      url: "/updates/node/mycellios-node-latest.json",
    })).rawPayload).toEqual(previousFeed);
  });

  it("authenticates and idempotently aborts only the exact uncommitted transaction", async () => {
    const root = temporaryDirectory();
    const updates = join(root, "updates");
    const downloads = join(root, "downloads");
    const storageRoot = join(root, "release-transactions");
    mkdirSync(updates);
    mkdirSync(downloads);
    const runtime = await createCoordinator(
      {
        host: "127.0.0.1",
        port: 0,
        databasePath: ":memory:",
        requestTimeoutMs: 1_000,
        nodeUpdatesPath: updates,
        releaseDownloadsPath: downloads,
      },
      {
        logger: false,
        runtimeMetadata: releaseRuntimeMetadata(root, RUNTIME_REVISION),
        releaseTransactionRoot: storageRoot,
        releaseTokenVerifier: async () => releaseClaims(RUNTIME_REVISION),
      },
    );
    runtimes.push(runtime);

    const transaction = releaseFixture("transaction-release-abort-0001");
    const asset = transaction.assets[0]!;
    expect((await uploadAsset(
      runtime,
      transaction.identity,
      asset.channel,
      asset.fileName,
      asset.content,
    )).statusCode).toBe(201);
    const transactionRoot = join(
      storageRoot,
      "transactions",
      transaction.identity.transactionId,
    );
    const siblingRoot = join(
      storageRoot,
      "transactions",
      "transaction-release-abort-sibling",
    );
    mkdirSync(siblingRoot, { recursive: true });
    writeFileSync(join(siblingRoot, "sentinel"), "keep");

    const unauthorized = await runtime.app.inject({
      method: "POST",
      url: `/internal/v1/releases/transactions/${transaction.identity.transactionId}/abort`,
    });
    expect(unauthorized.statusCode).toBe(401);
    expect(existsSync(transactionRoot)).toBe(true);

    const aborted = await runtime.app.inject({
      method: "POST",
      url: `/internal/v1/releases/transactions/${transaction.identity.transactionId}/abort`,
      headers: { authorization: "Bearer actions-token" },
    });
    expect(aborted.statusCode).toBe(200);
    expect(aborted.json()).toEqual({
      transactionId: transaction.identity.transactionId,
      alreadyAborted: false,
    });
    expect(existsSync(transactionRoot)).toBe(false);
    expect(readFileSync(join(siblingRoot, "sentinel"), "utf8")).toBe("keep");

    const repeated = await runtime.app.inject({
      method: "POST",
      url: `/internal/v1/releases/transactions/${transaction.identity.transactionId}/abort`,
      headers: { authorization: "Bearer actions-token" },
    });
    expect(repeated.statusCode).toBe(200);
    expect(repeated.json()).toEqual({
      transactionId: transaction.identity.transactionId,
      alreadyAborted: true,
    });

    const commit = await runtime.app.inject({
      method: "POST",
      url: `/internal/v1/releases/transactions/${transaction.identity.transactionId}/commit`,
      headers: { authorization: "Bearer actions-token" },
      payload: transaction.manifest,
    });
    expect(commit.statusCode).toBe(409);
  });

  it("never aborts an active or finalized transaction", async () => {
    const root = temporaryDirectory();
    const storageRoot = join(root, "release-transactions");
    const transaction = releaseFixture("transaction-release-abort-0002");
    const store = transactionStore(storageRoot);
    await store.initialize();
    await storeTransaction(store, transaction);
    await store.commit(transaction.manifest);

    await expect(store.abort(transaction.identity.transactionId))
      .rejects.toThrow("release_abort_active_transaction");
    expect(existsSync(join(
      storageRoot,
      "transactions",
      transaction.identity.transactionId,
    ))).toBe(true);

    await store.rollback(transaction.identity.transactionId);
    await expect(store.abort(transaction.identity.transactionId))
      .rejects.toThrow("release_abort_finalized_transaction");

    const rollbackOnlyId = "transaction-release-abort-rollback-only";
    const rollbackOnlyRoot = join(storageRoot, "transactions", rollbackOnlyId);
    mkdirSync(rollbackOnlyRoot, { recursive: true });
    writeFileSync(join(rollbackOnlyRoot, "rollback.json"), "{}\n");
    await expect(store.abort(rollbackOnlyId))
      .rejects.toThrow("release_abort_finalized_transaction");
    expect(existsSync(rollbackOnlyRoot)).toBe(true);
  });

  it("restores a previously committed release across revisions", async () => {
    const root = temporaryDirectory();
    const storageRoot = join(root, "release-transactions");
    const transaction = releaseFixture("transaction-release-0003");
    const first = new NativeReleaseTransactionStore({
      storageRoot,
      sourceId: RUNTIME_SOURCE_ID,
      revision: RUNTIME_REVISION,
      version: RELEASE_VERSION,
    });
    await first.initialize();
    for (const asset of transaction.assets) {
      await first.storeChunk({
        identity: transaction.identity,
        channel: asset.channel,
        fileName: asset.fileName,
        metadata: {
          chunkIndex: 0,
          chunkCount: 1,
          chunkSha256: asset.fileSha256,
          fileSha256: asset.fileSha256,
          fileSize: asset.fileSize,
        },
        body: asset.content,
      });
    }
    await first.commit(transaction.manifest);
    const replacement = releaseFixture("transaction-release-0004", "\r\n");
    for (const asset of replacement.assets) {
      await first.storeChunk({
        identity: replacement.identity,
        channel: asset.channel,
        fileName: asset.fileName,
        metadata: {
          chunkIndex: 0,
          chunkCount: 1,
          chunkSha256: asset.fileSha256,
          fileSha256: asset.fileSha256,
          fileSize: asset.fileSize,
        },
        body: asset.content,
      });
    }
    expect(await first.commit(replacement.manifest)).toMatchObject({
      previousTransactionId: transaction.identity.transactionId,
    });
    expect(await first.rollback(replacement.identity.transactionId)).toMatchObject({
      activeTransactionId: transaction.identity.transactionId,
      alreadyRolledBack: false,
    });

    const restarted = new NativeReleaseTransactionStore({
      storageRoot,
      sourceId: `sha256:${"3".repeat(64)}`,
      revision: "b".repeat(40),
      version: "0.2.20",
    });
    await restarted.initialize();
    const publicSetup = await restarted.publicAssetPath(
      "updates",
      "mycellios-node-latest.json",
    );
    expect(publicSetup).not.toBeNull();
    expect(readFileSync(publicSetup!)).toEqual(
      transaction.byPath.get("updates/mycellios-node-latest.json"),
    );
  });

  it("reconstructs a missing rollback receipt before allowing another mutation", async () => {
    const root = temporaryDirectory();
    const storageRoot = join(root, "release-transactions");
    const transaction = releaseFixture("transaction-release-crash-gap");
    const first = transactionStore(storageRoot);
    await first.initialize();
    await storeTransaction(first, transaction);
    await first.commit(transaction.manifest);
    await first.rollback(transaction.identity.transactionId);

    const rollbackPath = join(
      storageRoot,
      "transactions",
      transaction.identity.transactionId,
      "rollback.json",
    );
    rmSync(rollbackPath);

    const restarted = transactionStore(storageRoot);
    await restarted.initialize();
    expect(await restarted.rollback(transaction.identity.transactionId))
      .toMatchObject({
        activeTransactionId: null,
        alreadyRolledBack: true,
      });
    expect(JSON.parse(readFileSync(rollbackPath, "utf8"))).toMatchObject({
      transactionId: transaction.identity.transactionId,
      restoredTransactionId: null,
      restoredReleaseId: null,
    });
    await expect(restarted.commit(transaction.manifest))
      .rejects.toThrow("release_transaction_not_recommittable");
  });

  it("serializes the final asset upload before the transaction commit", async () => {
    const root = temporaryDirectory();
    const store = transactionStore(join(root, "release-transactions"));
    await store.initialize();
    const transaction = releaseFixture("transaction-release-serialized");
    await storeTransaction(
      store,
      { ...transaction, assets: transaction.assets.slice(0, -1) },
    );
    const finalAsset = transaction.assets.at(-1)!;
    const upload = store.storeChunk({
      identity: transaction.identity,
      channel: finalAsset.channel,
      fileName: finalAsset.fileName,
      metadata: {
        chunkIndex: 0,
        chunkCount: 1,
        chunkSha256: finalAsset.fileSha256,
        fileSha256: finalAsset.fileSha256,
        fileSize: finalAsset.fileSize,
      },
      body: finalAsset.content,
    });
    const commit = store.commit(transaction.manifest);

    await expect(upload).resolves.toMatchObject({ complete: true });
    await expect(commit).resolves.toMatchObject({
      transactionId: transaction.identity.transactionId,
      releaseId: transaction.manifest.releaseId,
    });
  });

  it("cleans abandoned atomic temporaries and reloads only durable final state", async () => {
    const root = temporaryDirectory();
    const storageRoot = join(root, "release-transactions");
    const transaction = releaseFixture("transaction-release-durable");
    const transactionRoot = join(
      storageRoot,
      "transactions",
      transaction.identity.transactionId,
    );
    const downloadsRoot = join(transactionRoot, "assets", "downloads");
    mkdirSync(downloadsRoot, { recursive: true });
    writeFileSync(
      join(storageRoot, ".active.json.abandoned.durable-tmp"),
      "abandoned active pointer",
    );
    writeFileSync(
      join(transactionRoot, ".seal.json.abandoned.durable-tmp"),
      "abandoned transaction seal",
    );
    writeFileSync(
      join(transactionRoot, ".commit.json.abandoned.durable-tmp"),
      "abandoned commit",
    );
    writeFileSync(
      join(downloadsRoot, ".mycellios-node-linux-x64.tar.gz.abandoned.durable-tmp"),
      "abandoned asset",
    );

    const store = transactionStore(storageRoot);
    await store.initialize();
    await storeTransaction(store, transaction);
    await store.commit(transaction.manifest);

    expect(temporaryDurabilityFiles(storageRoot)).toEqual([]);
    expect(JSON.parse(readFileSync(join(storageRoot, "active.json"), "utf8")))
      .toEqual({
        schema: "mycellios-native-release-active/1",
        transactionId: transaction.identity.transactionId,
        releaseId: transaction.manifest.releaseId,
      });

    const restarted = transactionStore(storageRoot);
    await restarted.initialize();
    const publicDeb = await restarted.publicAssetPath(
      "downloads",
      "mycellios-node-linux-x64.tar.gz",
    );
    expect(publicDeb).not.toBeNull();
    expect(readFileSync(publicDeb!)).toEqual(
      transaction.byPath.get("downloads/mycellios-node-linux-x64.tar.gz"),
    );

    await restarted.rollback(transaction.identity.transactionId);
    const afterRollback = transactionStore(storageRoot);
    await afterRollback.initialize();
    expect(await afterRollback.publicAssetPath(
      "downloads",
      "mycellios-node-linux-x64.tar.gz",
    )).toBeNull();
    expect(JSON.parse(readFileSync(join(storageRoot, "active.json"), "utf8")))
      .toEqual({
        schema: "mycellios-native-release-active/1",
        transactionId: null,
        releaseId: null,
      });
    expect(temporaryDurabilityFiles(storageRoot)).toEqual([]);
  });

  it("garbage-collects only old unfinished transaction directories on initialize", async () => {
    const root = temporaryDirectory();
    const storageRoot = join(root, "release-transactions");
    const active = releaseFixture("transaction-release-gc-active");
    const preparing = transactionStore(storageRoot);
    await preparing.initialize();
    await storeTransaction(preparing, active);
    await preparing.commit(active.manifest);

    const transactionsRoot = join(storageRoot, "transactions");
    const oldPartialRoot = join(
      transactionsRoot,
      "transaction-release-gc-old-partial",
    );
    const recentPartialRoot = join(
      transactionsRoot,
      "transaction-release-gc-recent",
    );
    const committedRoot = join(
      transactionsRoot,
      "transaction-release-gc-committed",
    );
    const invalidSiblingRoot = join(transactionsRoot, "operator-notes!");
    const outsideSiblingRoot = join(storageRoot, "transaction-release-gc-sibling");
    const validNamedFile = join(
      transactionsRoot,
      "transaction-release-gc-regular-file",
    );

    mkdirSync(join(oldPartialRoot, "assets", ".incoming"), { recursive: true });
    writeFileSync(
      join(oldPartialRoot, "assets", ".incoming", "000000.part"),
      "old partial",
    );
    mkdirSync(recentPartialRoot, { recursive: true });
    writeFileSync(join(recentPartialRoot, "seal.json"), "recent");
    mkdirSync(committedRoot, { recursive: true });
    writeFileSync(join(committedRoot, "commit.json"), "{}\n");
    mkdirSync(invalidSiblingRoot, { recursive: true });
    writeFileSync(join(invalidSiblingRoot, "sentinel"), "keep invalid sibling");
    mkdirSync(outsideSiblingRoot, { recursive: true });
    writeFileSync(join(outsideSiblingRoot, "sentinel"), "keep outside sibling");
    writeFileSync(validNamedFile, "keep regular file");

    const old = new Date(Date.now() - 8 * 24 * 60 * 60 * 1_000);
    ageTree(oldPartialRoot, old);
    ageTree(committedRoot, old);
    ageTree(invalidSiblingRoot, old);
    ageTree(outsideSiblingRoot, old);
    utimesSync(validNamedFile, old, old);
    ageTree(
      join(transactionsRoot, active.identity.transactionId),
      old,
    );

    const restarted = transactionStore(storageRoot);
    await restarted.initialize();

    expect(existsSync(oldPartialRoot)).toBe(false);
    expect(existsSync(recentPartialRoot)).toBe(true);
    expect(existsSync(committedRoot)).toBe(true);
    expect(existsSync(invalidSiblingRoot)).toBe(true);
    expect(existsSync(outsideSiblingRoot)).toBe(true);
    expect(existsSync(validNamedFile)).toBe(true);
    expect(existsSync(join(
      transactionsRoot,
      active.identity.transactionId,
    ))).toBe(true);
    expect(await restarted.publicAssetPath(
      "downloads",
      "mycellios-node-linux-x64.tar.gz",
    )).not.toBeNull();
  });

  it("restricts OIDC claims to this repository, workflow and release refs", () => {
    expect(validateGitHubReleaseClaims({
      repository: "tych0s/mycellios",
      ref: "refs/heads/main",
      sha: "a".repeat(40),
      workflow_ref: "tych0s/mycellios/.github/workflows/node-build.yml@refs/heads/main",
      event_name: "push",
      environment: "production",
    })).toMatchObject({
      ref: "refs/heads/main",
      sha: "a".repeat(40),
      eventName: "push",
    });

    expect(() => validateGitHubReleaseClaims({
      repository: "attacker/fork",
      ref: "refs/heads/main",
      sha: "a".repeat(40),
      workflow_ref: "attacker/fork/.github/workflows/node-build.yml@refs/heads/main",
      event_name: "push",
      environment: "production",
    })).toThrow("release_repository_not_allowed");

    expect(validateGitHubReleaseClaims({
      repository: "tych0s/mycellios",
      ref: "refs/heads/main",
      sha: "b".repeat(40),
      workflow_ref:
        "tych0s/mycellios/.github/workflows/publish-existing-release.yml@refs/heads/main",
      event_name: "workflow_dispatch",
      environment: "production",
    })).toMatchObject({
      ref: "refs/heads/main",
      sha: "b".repeat(40),
      eventName: "workflow_dispatch",
    });

    expect(() => validateGitHubReleaseClaims({
      repository: "tych0s/mycellios",
      ref: "refs/tags/v0.2.19",
      sha: "b".repeat(40),
      workflow_ref: "tych0s/mycellios/.github/workflows/node-build.yml@refs/heads/main",
      event_name: "push",
      environment: "production",
    })).toThrow("release_ref_not_allowed");

    expect(() => validateGitHubReleaseClaims({
      repository: "tych0s/mycellios",
      ref: "refs/heads/main",
      sha: "b".repeat(40),
      workflow_ref:
        "tych0s/mycellios/.github/workflows/node-build.yml@refs/heads/main-evil",
      event_name: "push",
      environment: "production",
    })).toThrow("release_workflow_not_allowed");

    expect(() => validateGitHubReleaseClaims({
      repository: "tych0s/mycellios",
      ref: "refs/heads/main",
      sha: "b".repeat(40),
      workflow_ref: "tych0s/mycellios/.github/workflows/node-build.yml@refs/heads/main",
      event_name: "workflow_dispatch",
      environment: "production",
    })).toThrow("release_event_not_allowed");

    expect(() => validateGitHubReleaseClaims({
      repository: "tych0s/mycellios",
      ref: "refs/heads/main",
      sha: "b".repeat(40),
      workflow_ref:
        "tych0s/mycellios/.github/workflows/publish-existing-release.yml@refs/heads/main",
      event_name: "push",
      environment: "production",
    })).toThrow("release_event_not_allowed");

    expect(() => validateGitHubReleaseClaims({
      repository: "tych0s/mycellios",
      ref: "refs/heads/main",
      sha: "c".repeat(40),
      workflow_ref: "tych0s/mycellios/.github/workflows/node-build.yml@refs/heads/main",
      event_name: "push",
      environment: "attestation",
    })).toThrow("release_environment_not_allowed");
  });

  it("rejects a valid Actions token when its SHA is not the running revision", async () => {
    const root = temporaryDirectory();
    const content = Buffer.from('{"version":"0.2.19"}\n');
    const digest = sha256(content);
    const runtime = await createCoordinator(
      {
        host: "127.0.0.1",
        port: 0,
        databasePath: ":memory:",
        requestTimeoutMs: 1_000,
        nodeUpdatesPath: join(root, "updates"),
      },
      {
        logger: false,
        runtimeMetadata: releaseRuntimeMetadata(root, RUNTIME_REVISION),
        releaseTokenVerifier: async () => releaseClaims("b".repeat(40)),
      },
    );
    runtimes.push(runtime);

    const response = await uploadLatest(runtime, content, digest);

    expect(response.statusCode).toBe(409);
    expect(response.json()).toEqual({
      error: { code: "release_upload_revision_mismatch" },
    });
  });

  it("rejects release uploads when the running revision is not sealed", async () => {
    const root = temporaryDirectory();
    const content = Buffer.from('{"version":"0.2.19"}\n');
    const digest = sha256(content);
    const runtime = await createCoordinator(
      {
        host: "127.0.0.1",
        port: 0,
        databasePath: ":memory:",
        requestTimeoutMs: 1_000,
        nodeUpdatesPath: join(root, "updates"),
      },
      {
        logger: false,
        runtimeMetadata: releaseRuntimeMetadata(root, null),
        releaseTokenVerifier: async () => releaseClaims(RUNTIME_REVISION),
      },
    );
    runtimes.push(runtime);

    const response = await uploadLatest(runtime, content, digest);

    expect(response.statusCode).toBe(503);
    expect(response.json()).toEqual({
      error: { code: "release_upload_runtime_revision_unavailable" },
    });
  });

  it("authenticates release chunks before reading them and enforces the sealed chunk limit", async () => {
    const root = temporaryDirectory();
    const runtime = await createCoordinator(
      {
        host: "127.0.0.1",
        port: 0,
        databasePath: ":memory:",
        requestTimeoutMs: 1_000,
        nodeUpdatesPath: join(root, "updates"),
      },
      {
        logger: false,
        runtimeMetadata: releaseRuntimeMetadata(root, RUNTIME_REVISION),
        releaseTokenVerifier: async () => releaseClaims(RUNTIME_REVISION),
        releaseChunkBodyLimitBytes: 1_024,
      },
    );
    runtimes.push(runtime);
    const oversized = Buffer.alloc(1_025);

    const unauthenticated = await runtime.app.inject({
      method: "PUT",
      url: "/internal/v1/releases/updates/mycellios-node-latest.json",
      headers: { "content-type": "application/octet-stream" },
      payload: oversized,
    });
    expect(unauthenticated.statusCode).toBe(401);
    expect(unauthenticated.json()).toEqual({
      error: { code: "release_upload_token_missing" },
    });

    const authenticated = await runtime.app.inject({
      method: "PUT",
      url: "/internal/v1/releases/updates/mycellios-node-latest.json",
      headers: {
        authorization: "Bearer actions-token",
        "content-type": "application/octet-stream",
      },
      payload: oversized,
    });
    expect(authenticated.statusCode).toBe(413);
  });
});

function temporaryDirectory(): string {
  const directory = mkdtempSync(join(tmpdir(), "mycellios-release-upload-"));
  directories.push(directory);
  return directory;
}

function temporaryDurabilityFiles(root: string): string[] {
  const output: string[] = [];
  const visit = (directory: string): void => {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const absolute = join(directory, entry.name);
      if (entry.isDirectory()) visit(absolute);
      else if (entry.name.endsWith(".durable-tmp")) output.push(absolute);
    }
  };
  visit(root);
  return output.sort();
}

function ageTree(root: string, timestamp: Date): void {
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    const path = join(root, entry.name);
    if (entry.isDirectory()) ageTree(path, timestamp);
    else utimesSync(path, timestamp, timestamp);
  }
  utimesSync(root, timestamp, timestamp);
}

function sha256(value: Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function releaseRuntimeMetadata(
  root: string,
  revision: string | null,
): NativeRuntimeBuildMetadata {
  return {
    root,
    version: RELEASE_VERSION,
    revision,
    buildIdentity: {
      schema: "mycellios-native-build-provenance/1",
      version: RELEASE_VERSION,
      sourceId: RUNTIME_SOURCE_ID,
    },
  };
}

function releaseClaims(sha: string): GitHubReleaseClaims {
  return {
    repository: "tych0s/mycellios",
    ref: "refs/heads/main",
    sha,
    workflowRef:
      "tych0s/mycellios/.github/workflows/node-build.yml@refs/heads/main",
    eventName: "push",
    environment: "production",
  };
}

function transactionStore(storageRoot: string): NativeReleaseTransactionStore {
  return new NativeReleaseTransactionStore({
    storageRoot,
    sourceId: RUNTIME_SOURCE_ID,
    revision: RUNTIME_REVISION,
    version: RELEASE_VERSION,
  });
}

async function storeTransaction(
  store: NativeReleaseTransactionStore,
  transaction: ReturnType<typeof releaseFixture>,
): Promise<void> {
  for (const asset of transaction.assets) {
    await store.storeChunk({
      identity: transaction.identity,
      channel: asset.channel,
      fileName: asset.fileName,
      metadata: {
        chunkIndex: 0,
        chunkCount: 1,
        chunkSha256: asset.fileSha256,
        fileSha256: asset.fileSha256,
        fileSize: asset.fileSize,
      },
      body: asset.content,
    });
  }
}

async function uploadLatest(
  runtime: CoordinatorRuntime,
  content: Buffer,
  digest: string,
) {
  return runtime.app.inject({
    method: "PUT",
    url: "/internal/v1/releases/updates/mycellios-node-latest.json",
    headers: {
      authorization: "Bearer actions-token",
      "content-type": "application/octet-stream",
      "x-chunk-index": "0",
      "x-chunk-count": "1",
      "x-chunk-sha256": digest,
      "x-file-sha256": digest,
      "x-file-size": String(content.length),
      "x-release-transaction-id": "transaction-latest-0001",
      "x-release-source-id": RUNTIME_SOURCE_ID,
      "x-release-revision": RUNTIME_REVISION,
      "x-release-version": RELEASE_VERSION,
    },
    payload: content,
  });
}

function releaseFixture(
  transactionId: string,
  _lineEnding: "\n" | "\r\n" = "\n",
): {
  identity: ReleaseTransactionIdentity;
  assets: Array<ReleaseAssetEvidence & { content: Buffer }>;
  byPath: Map<string, Buffer>;
  manifest: ReturnType<typeof buildReleaseTransactionManifest>;
} {
  const identity: ReleaseTransactionIdentity = {
    transactionId,
    sourceId: RUNTIME_SOURCE_ID,
    revision: RUNTIME_REVISION,
    version: RELEASE_VERSION,
  };
  const linux = Buffer.from("sealed Linux node package");
  const macos = Buffer.from("sealed macOS node package");
  const windows = Buffer.from("sealed Windows node package");
  const packageEvidence = [
    { target: "linux-x64", name: "mycellios-node-linux-x64.tar.gz", content: linux },
    { target: "macos-arm64", name: "mycellios-node-macos-arm64.tar.gz", content: macos },
    { target: "windows-x64", name: "mycellios-node-windows-x64.zip", content: windows },
  ];
  const latest = Buffer.from(`${JSON.stringify({
    schema: "mycellios-node-update-feed/1",
    version: RELEASE_VERSION,
    publishedAt: "2026-07-25T00:00:00.000Z",
    packages: packageEvidence.map(({ target, name, content }) => ({
      target,
      name,
      bytes: content.length,
      sha256: sha256(content),
    })),
  }, null, 2)}\n`);
  const contents = new Map<string, Buffer>([
    ["downloads/mycellios-node-linux-x64.tar.gz", linux],
    ["downloads/mycellios-node-macos-arm64.tar.gz", macos],
    ["downloads/mycellios-node-windows-x64.zip", windows],
    ["updates/mycellios-node-latest.json", latest],
  ]);
  const assets = [...contents.entries()].map(([path, content]) => {
    const [channel, fileName] = path.split("/", 2) as [
      ReleaseAssetChannel,
      string,
    ];
    return {
      channel,
      fileName,
      fileSize: content.length,
      fileSha256: sha256(content),
      content,
    };
  });
  return {
    identity,
    assets,
    byPath: contents,
    manifest: buildReleaseTransactionManifest({
      ...identity,
      assets: assets.map(({ content: _content, ...evidence }) => evidence),
    }),
  };
}

async function uploadAsset(
  runtime: CoordinatorRuntime,
  identity: ReleaseTransactionIdentity,
  channel: ReleaseAssetChannel,
  fileName: string,
  content: Buffer,
) {
  const digest = sha256(content);
  return runtime.app.inject({
    method: "PUT",
    url: `/internal/v1/releases/${channel}/${fileName}`,
    headers: {
      authorization: "Bearer actions-token",
      "content-type": "application/octet-stream",
      "x-chunk-index": "0",
      "x-chunk-count": "1",
      "x-chunk-sha256": digest,
      "x-file-sha256": digest,
      "x-file-size": String(content.length),
      "x-release-transaction-id": identity.transactionId,
      "x-release-source-id": identity.sourceId,
      "x-release-revision": identity.revision,
      "x-release-version": identity.version,
    },
    payload: content,
  });
}
