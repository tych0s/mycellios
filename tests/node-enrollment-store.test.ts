import { describe, expect, it } from "vitest";
import { NodeEnrollmentStore } from "../src/coordinator/node-enrollment-store.js";
import { MeshDatabase } from "../src/storage/database.js";

const actor = {
  kind: "account" as const,
  id: "account-1",
  scopes: ["node:identity" as const],
};

describe("NodeEnrollmentStore", () => {
  it("stores only hashes, requires account confirmation and consumes once atomically", async () => {
    const database = new MeshDatabase(":memory:");
    const store = new NodeEnrollmentStore(database, () => Date.parse("2026-08-09T20:00:00Z"));
    const issued = store.issue({ accountId: "account-1", actor, expiresInSeconds: 120 });
    const stored = store.inspectStoredSecrets(issued.enrollmentId)!;
    expect(stored.tokenHash).not.toContain(issued.enrollmentToken);
    expect(stored.nonceHash).not.toContain(issued.nonce);
    expect(JSON.stringify(database.raw.prepare("SELECT * FROM node_enrollments").get())).not.toContain(issued.enrollmentToken);

    const redeem = () => store.consume({
      enrollmentToken: issued.enrollmentToken,
      nonce: issued.nonce,
      identityKind: "device",
      identityId: "node-1",
      publicKeyFingerprint: `sha256:${"a".repeat(64)}`,
    });
    expect(redeem().state).toBe("unconfirmed");
    store.confirm({ enrollmentId: issued.enrollmentId, accountId: "account-1", actorId: "account-1" });
    const results = await Promise.all([Promise.resolve().then(redeem), Promise.resolve().then(redeem)]);
    expect(results.map(({ state }) => state).sort()).toEqual(["already-consumed", "consumed"]);
    expect(database.getNodeOwnership("device", "node-1")).toMatchObject({
      accountId: "account-1",
      credentialFingerprint: `sha256:${"a".repeat(64)}`,
      status: "active",
      generation: 1,
    });
    expect(store.audit(issued.enrollmentId).map(({ eventType }) => eventType)).toEqual(["issued", "confirmed", "consumed"]);
    expect(database.raw.prepare("SELECT event_type, generation, actor_kind FROM node_identity_events WHERE identity_id = 'node-1'").get())
      .toMatchObject({ event_type: "identity.enrolled", generation: 1, actor_kind: "account" });
    expect(database.nodeIdentityEvents("device", "node-1")).toMatchObject([{
      eventType: "identity.enrolled", generation: 1, previousEventDigest: null,
    }]);
    database.raw.prepare("UPDATE node_identity_events SET details_json = '{}' WHERE identity_id = 'node-1'").run();
    expect(() => database.nodeIdentityEvents("device", "node-1")).toThrow("node_identity_event_digest_mismatch");
    database.close();
  });

  it("prevents cross-account takeover and requires recovery for a replacement key", () => {
    const database = new MeshDatabase(":memory:");
    const store = new NodeEnrollmentStore(database, () => Date.parse("2026-08-09T20:00:00Z"));
    const enroll = (accountId: string, fingerprint: `sha256:${string}`) => {
      const enrollmentActor = { ...actor, id: accountId };
      const issued = store.issue({ accountId, actor: enrollmentActor, expiresInSeconds: 120 });
      store.confirm({ enrollmentId: issued.enrollmentId, accountId, actorId: accountId });
      return store.consume({
        enrollmentToken: issued.enrollmentToken,
        nonce: issued.nonce,
        identityKind: "device",
        identityId: "node-owned",
        publicKeyFingerprint: fingerprint,
      });
    };

    expect(enroll("account-1", `sha256:${"a".repeat(64)}`).state).toBe("consumed");
    expect(enroll("account-1", `sha256:${"b".repeat(64)}`).state).toBe("recovery-required");
    expect(enroll("account-2", `sha256:${"a".repeat(64)}`).state).toBe("ownership-conflict");
    expect(database.getNodeOwnership("device", "node-owned")).toMatchObject({
      accountId: "account-1",
      credentialFingerprint: `sha256:${"a".repeat(64)}`,
      status: "active",
      generation: 1,
    });
    database.close();
  });

  it("rejects expiration, wrong nonce and account mismatch", () => {
    let clock = Date.parse("2026-08-09T20:00:00Z");
    const database = new MeshDatabase(":memory:");
    const store = new NodeEnrollmentStore(database, () => clock);
    const first = store.issue({ accountId: "account-1", actor, expiresInSeconds: 60 });
    expect(() => store.confirm({ enrollmentId: first.enrollmentId, accountId: "account-2", actorId: "account-2" })).toThrow("node_enrollment_account_mismatch");
    store.confirm({ enrollmentId: first.enrollmentId, accountId: "account-1", actorId: "account-1" });
    expect(store.consume({ enrollmentToken: first.enrollmentToken, nonce: "wrong", identityKind: "device", identityId: "node-1", publicKeyFingerprint: `sha256:${"a".repeat(64)}` }).state).toBe("nonce-mismatch");
    const second = store.issue({ accountId: "account-1", actor, expiresInSeconds: 60 });
    store.confirm({ enrollmentId: second.enrollmentId, accountId: "account-1", actorId: "account-1" });
    clock += 60_001;
    expect(store.consume({ enrollmentToken: second.enrollmentToken, nonce: second.nonce, identityKind: "device", identityId: "node-2", publicKeyFingerprint: `sha256:${"b".repeat(64)}` }).state).toBe("expired");
    database.close();
  });

  it("rejects credential reuse across identities without leaking a database constraint", () => {
    const database = new MeshDatabase(":memory:");
    const store = new NodeEnrollmentStore(database, () => Date.parse("2026-08-09T20:00:00Z"));
    const enroll = (identityId: string) => {
      const issued = store.issue({ accountId: "account-1", actor, expiresInSeconds: 120 });
      store.confirm({ enrollmentId: issued.enrollmentId, accountId: "account-1", actorId: "account-1" });
      return store.consume({ enrollmentToken: issued.enrollmentToken, nonce: issued.nonce,
        identityKind: "device", identityId, publicKeyFingerprint: `sha256:${"c".repeat(64)}` });
    };
    expect(enroll("node-first").state).toBe("consumed");
    expect(enroll("node-second").state).toBe("credential-conflict");
    expect(database.getNodeOwnership("device", "node-second")).toBeNull();
    database.close();
  });
});
