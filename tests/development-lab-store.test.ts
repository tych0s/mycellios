import { spawnSync } from "node:child_process";
import { generateKeyPairSync, randomUUID } from "node:crypto";
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  stat,
  utimes,
  writeFile,
} from "node:fs/promises";
import { hostname, tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  DevelopmentLabStore,
  developmentLabStateContainsRawToken,
} from "../src/coordinator/development-lab-store.js";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(
    roots.splice(0).map((root) => rm(root, { recursive: true, force: true })),
  );
});

describe("development lab store", () => {
  it("persists only an invitation hash and consumes the token once", async () => {
    const root = await temporaryRoot();
    let now = new Date("2026-07-31T08:00:00.000Z");
    const store = new DevelopmentLabStore({ root, now: () => now });
    const key = signingKey();
    const created = await store.createLab({
      name: "Two homes",
      ...key,
      invitationTtlSeconds: 600,
    });

    expect(created.token).toMatch(/^[A-Za-z0-9_-]{43}$/);
    expect(
      await developmentLabStateContainsRawToken(root, created.token),
    ).toBe(false);
    await expect(
      store.redeemInvitation(created.lab.labId, created.token),
    ).resolves.toMatchObject({
      labId: created.lab.labId,
      status: "active",
    });
    await expect(
      store.redeemInvitation(created.lab.labId, created.token),
    ).rejects.toMatchObject({
      code: "development_lab_invitation_used",
    });

    const reloaded = new DevelopmentLabStore({ root, now: () => now });
    await expect(reloaded.getLab(created.lab.labId)).resolves.toMatchObject({
      labId: created.lab.labId,
      name: "Two homes",
    });
  });

  it("allows only one redemption across concurrent store instances", async () => {
    const root = await temporaryRoot();
    const first = new DevelopmentLabStore({ root });
    const second = new DevelopmentLabStore({ root });
    const created = await first.createLab({
      name: "Concurrent homes",
      ...signingKey(),
    });

    await expect(second.getLab(created.lab.labId)).resolves.toMatchObject({
      status: "active",
    });
    const results = await Promise.allSettled([
      first.redeemInvitation(created.lab.labId, created.token),
      second.redeemInvitation(created.lab.labId, created.token),
    ]);

    expect(results.filter((result) => result.status === "fulfilled")).toHaveLength(
      1,
    );
    const rejected = results.find((result) => result.status === "rejected");
    expect(rejected).toMatchObject({
      status: "rejected",
      reason: {
        code: "development_lab_invitation_used",
      },
    });
    const reloaded = new DevelopmentLabStore({ root });
    await expect(
      reloaded.redeemInvitation(created.lab.labId, created.token),
    ).rejects.toMatchObject({
      code: "development_lab_invitation_used",
    });
  });

  it("reloads under the lock so a stale instance cannot undo revocation", async () => {
    const root = await temporaryRoot();
    const owner = new DevelopmentLabStore({ root });
    const stale = new DevelopmentLabStore({ root });
    const created = await owner.createLab(signingKey());

    await expect(stale.getLab(created.lab.labId)).resolves.toMatchObject({
      status: "active",
    });
    await owner.revokeLab(created.lab.labId);

    await expect(
      stale.createInvitation(created.lab.labId),
    ).rejects.toMatchObject({
      code: "development_lab_revoked",
    });
    await expect(stale.getLab(created.lab.labId)).resolves.toMatchObject({
      status: "revoked",
    });
    const reloaded = new DevelopmentLabStore({ root });
    await expect(reloaded.getLab(created.lab.labId)).resolves.toMatchObject({
      status: "revoked",
    });
  });

  it("recovers an abandoned stale lock before loading state", async () => {
    const root = await temporaryRoot();
    const stateRoot = join(root, "development-labs");
    const lockPath = join(stateRoot, "state.lock");
    await mkdir(stateRoot, { recursive: true });
    await writeFile(lockPath, JSON.stringify({
      ownerId: randomUUID(),
      host: hostname(),
      pid: definitelyDeadPid(),
      acquiredAt: "2026-07-31T08:00:00.000Z",
    }));
    const old = new Date(Date.now() - 1_000);
    await utimes(lockPath, old, old);

    const store = new DevelopmentLabStore({
      root,
      lockStaleMs: 10,
      lockTimeoutMs: 1_000,
      lockRetryMs: 5,
    });
    await expect(store.initialize()).resolves.toBeUndefined();
    await expect(stat(lockPath)).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("fails closed instead of recovering a stale-looking live-process lock", async () => {
    const root = await temporaryRoot();
    const stateRoot = join(root, "development-labs");
    const lockPath = join(stateRoot, "state.lock");
    const encoded = JSON.stringify({
      ownerId: randomUUID(),
      host: hostname(),
      pid: process.pid,
      acquiredAt: "2026-07-31T08:00:00.000Z",
    });
    await mkdir(stateRoot, { recursive: true });
    await writeFile(lockPath, encoded);
    const old = new Date(Date.now() - 1_000);
    await utimes(lockPath, old, old);

    const store = new DevelopmentLabStore({
      root,
      lockStaleMs: 10,
      lockTimeoutMs: 50,
      lockRetryMs: 5,
    });
    await expect(store.initialize()).rejects.toMatchObject({
      code: "development_lab_io_failed",
    });
    await expect(readFile(lockPath, "utf8")).resolves.toBe(encoded);
  });

  it("rejects expired invitations without accepting a later clock", async () => {
    const root = await temporaryRoot();
    let now = new Date("2026-07-31T08:00:00.000Z");
    const store = new DevelopmentLabStore({ root, now: () => now });
    const created = await store.createLab({
      ...signingKey(),
      invitationTtlSeconds: 60,
    });
    now = new Date("2026-07-31T08:01:00.000Z");

    await expect(
      store.redeemInvitation(created.lab.labId, created.token),
    ).rejects.toMatchObject({
      code: "development_lab_invitation_expired",
    });
  });

  it("revokes invitation and release-store access together", async () => {
    const root = await temporaryRoot();
    const store = new DevelopmentLabStore({ root });
    const created = await store.createLab(signingKey());
    await store.revokeLab(created.lab.labId);

    await expect(
      store.createInvitation(created.lab.labId),
    ).rejects.toMatchObject({
      code: "development_lab_revoked",
    });
    await expect(
      store.releaseStore(created.lab.labId),
    ).rejects.toMatchObject({
      code: "development_lab_revoked",
    });
  });
});

function signingKey(): { keyId: string; spki: string } {
  const { publicKey } = generateKeyPairSync("ed25519");
  return {
    keyId: "mycellios-dev-lab-test",
    spki: publicKey
      .export({ format: "der", type: "spki" })
      .toString("base64url"),
  };
}

async function temporaryRoot(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "mycellios-development-lab-"));
  roots.push(root);
  return root;
}

function definitelyDeadPid(): number {
  const completed = spawnSync(process.execPath, [
    "-e",
    "process.exit(0)",
  ]);
  if (!completed.pid) {
    throw new Error("dead_pid_fixture_missing");
  }
  try {
    process.kill(completed.pid, 0);
  } catch (error) {
    if (
      error
      && typeof error === "object"
      && "code" in error
      && error.code === "ESRCH"
    ) {
      return completed.pid;
    }
    throw error;
  }
  throw new Error("dead_pid_fixture_still_alive");
}
