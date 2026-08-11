import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { ProtectedSecretStore, SecretCommandRunner } from "../src/node/identity-store.js";
import {
  LinuxSecretToolStore,
  LinuxSystemdCredentialStore,
  MacOsKeychainSecretStore,
  NodeIdentityStore,
  WindowsDpapiSecretStore,
} from "../src/node/identity-store.js";
import { generateWorkerAdmissionCredential, workerAdmissionSigner } from "../src/worker/admission-credential.js";

const cleanup: string[] = [];
afterEach(async () => Promise.all(cleanup.splice(0).map((path) => rm(path, { recursive: true, force: true }))));

describe("NodeIdentityStore", () => {
  it("keeps private key material exclusively in the protected provider", async () => {
    const directory = await temporaryDirectory();
    const secrets = new MemorySecretStore();
    const path = join(directory, "identity.json");
    const store = new NodeIdentityStore(path, secrets);
    const signer = await store.loadOrCreate("node-a");
    const metadata = await readFile(path, "utf8");
    expect(metadata).toContain(signer.publicKey.spki);
    expect(metadata).not.toContain("privateKeyPkcs8");
    expect([...secrets.values.values()][0]).toContain("privateKeyPkcs8");

    const restored = await store.loadOrCreate("node-a");
    expect(restored.publicKey).toEqual(signer.publicKey);
    expect(restored.sign("challenge")).toBe(signer.sign("challenge"));
  });

  it("fails closed for missing secret, provider mismatch or identity takeover", async () => {
    const directory = await temporaryDirectory();
    const path = join(directory, "identity.json");
    const first = new MemorySecretStore();
    await new NodeIdentityStore(path, first).loadOrCreate("node-a");
    first.values.clear();
    await expect(new NodeIdentityStore(path, first).loadOrCreate("node-a"))
      .rejects.toThrow("node_identity_protected_secret_is_missing");
    await expect(new NodeIdentityStore(path, {
      provider: "another-provider",
      get: (reference) => first.get(reference),
      set: (reference, value) => first.set(reference, value),
      delete: (reference) => first.delete(reference),
    }).loadOrCreate("node-a"))
      .rejects.toThrow("node_identity_provider_mismatch");
    await expect(new NodeIdentityStore(path, first).loadOrCreate("node-b"))
      .rejects.toThrow("node_identity_id_mismatch");
  });

  it("migrates a legacy plaintext credential into protected storage without rotating it", async () => {
    const directory = await temporaryDirectory();
    const path = join(directory, "identity.json");
    const credential = generateWorkerAdmissionCredential();
    const expected = workerAdmissionSigner(credential);
    await writeFile(path, JSON.stringify(credential), { mode: 0o600 });
    const secrets = new MemorySecretStore();

    await expect(new NodeIdentityStore(path, secrets).loadOrCreate("node-a"))
      .rejects.toThrow("node_identity_legacy_migration_confirmation_required");
    expect(secrets.values.size).toBe(0);

    const migrated = await new NodeIdentityStore(path, secrets, { allowLegacyMigration: true }).loadOrCreate("node-a");
    expect(migrated.publicKey).toEqual(expected.publicKey);
    expect(migrated.sign("proof")).toBe(expected.sign("proof"));
    const metadata = await readFile(path, "utf8");
    expect(metadata).not.toContain("privateKeyPkcs8");
    expect([...secrets.values.values()][0]).toContain(credential.privateKeyPkcs8);
  });

  it("never mutates a corrupt legacy identity even when migration is confirmed", async () => {
    const directory = await temporaryDirectory();
    const path = join(directory, "identity.json");
    await writeFile(path, JSON.stringify({ privateKeyPkcs8: "foreign-or-corrupt" }), { mode: 0o600 });
    const secrets = new MemorySecretStore();
    await expect(new NodeIdentityStore(path, secrets, { allowLegacyMigration: true }).loadOrCreate("node-a"))
      .rejects.toThrow("node_identity_metadata_is_invalid");
    expect(secrets.values.size).toBe(0);
    expect(await readFile(path, "utf8")).toContain("foreign-or-corrupt");
  });

  it("passes protected values only through stdin on Linux, Windows and macOS", async () => {
    const credential = generateWorkerAdmissionCredential();
    const secret = JSON.stringify(credential);
    const calls: Array<{ executable: string; arguments_: readonly string[]; stdin?: string }> = [];
    const runner: SecretCommandRunner = async (executable, arguments_, stdin) => {
      calls.push({ executable, arguments_: [...arguments_], ...(stdin === undefined ? {} : { stdin }) });
      return { code: 0, stdout: secret };
    };
    const directory = await temporaryDirectory();
    await new LinuxSecretToolStore(runner).set("node-a", secret);
    await new WindowsDpapiSecretStore(directory, runner).set("node-a", secret);
    await new MacOsKeychainSecretStore(runner).set("node-a", secret);

    expect(calls).toHaveLength(3);
    for (const call of calls) {
      expect([call.executable, ...call.arguments_].join(" ")).not.toContain(secret);
      expect(call.stdin).toContain(credential.privateKeyPkcs8);
    }
    expect(calls[1]?.arguments_.join(" ")).toContain("DataProtectionScope]::LocalMachine");
    expect(calls[2]).toMatchObject({ executable: "/usr/bin/security", arguments_: ["-i"] });
  });

  it("uses systemd-creds instead of a graphical secret service for a Linux daemon", async () => {
    const directory = await temporaryDirectory();
    const secret = JSON.stringify(generateWorkerAdmissionCredential());
    const calls: Array<{ executable: string; arguments_: readonly string[]; stdin?: string }> = [];
    const runner: SecretCommandRunner = async (executable, arguments_, stdin) => {
      calls.push({ executable, arguments_: [...arguments_], ...(stdin === undefined ? {} : { stdin }) });
      if (arguments_[0] === "encrypt") await writeFile(String(arguments_[3]), "encrypted", { mode: 0o600 });
      return { code: 0, stdout: arguments_[0] === "decrypt" ? secret : "" };
    };
    const store = new LinuxSystemdCredentialStore(directory, runner);
    await store.set("node-a", secret);
    expect(await store.get("node-a")).toBe(secret);
    expect(calls.map((call) => call.executable)).toEqual(["systemd-creds", "systemd-creds"]);
    expect(calls[0]?.arguments_).toEqual(["encrypt", "--name=node-a", "-", expect.stringContaining("node-a.cred.tmp-")]);
    expect(calls[0]?.stdin).toBe(secret);
    expect(calls.flatMap((call) => call.arguments_).join(" ")).not.toContain(secret);
  });

  it("loads and deletes platform secrets without accepting path-like references", async () => {
    const secret = JSON.stringify(generateWorkerAdmissionCredential());
    const runner: SecretCommandRunner = async (_executable, arguments_) => ({
      code: 0,
      stdout: arguments_.includes("find-generic-password") ? `${secret}\n` : secret,
    });
    const directory = await temporaryDirectory();
    expect(await new WindowsDpapiSecretStore(directory, runner).get("node-a")).toBe(secret);
    expect(await new MacOsKeychainSecretStore(runner).get("node-a")).toBe(secret);
    await expect(new WindowsDpapiSecretStore(directory, runner).get("../escape"))
      .rejects.toThrow("node_identity_secret_reference_is_invalid");
    await expect(new MacOsKeychainSecretStore(runner).delete("../escape"))
      .rejects.toThrow("node_identity_secret_reference_is_invalid");
  });
});

class MemorySecretStore implements ProtectedSecretStore {
  readonly provider = "test-protected-store";
  readonly values = new Map<string, string>();
  async get(reference: string) { return this.values.get(reference) ?? null; }
  async set(reference: string, value: string) { this.values.set(reference, value); }
  async delete(reference: string) { this.values.delete(reference); }
}

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), "mycellios-node-identity-"));
  cleanup.push(directory);
  return directory;
}
