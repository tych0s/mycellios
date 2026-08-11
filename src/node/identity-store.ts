import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { mkdir, open, readFile, rename, rm } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { z } from "zod";
import {
  generateWorkerAdmissionCredential,
  parseWorkerAdmissionCredential,
  workerAdmissionSigner,
  type WorkerAdmissionCredential,
  type WorkerAdmissionSigner,
} from "../worker/admission-credential.js";

const metadataSchema = z.object({
  schema: z.literal("mycellios-node-identity/1"),
  identityId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
  provider: z.string().min(1).max(100),
  secretReference: z.string().regex(/^[A-Za-z0-9._:-]{1,200}$/),
  algorithm: z.literal("ed25519"),
  publicKeySpki: z.string().min(40).max(256),
  createdAt: z.string().datetime(),
}).strict();

export interface ProtectedSecretStore {
  readonly provider: string;
  get(reference: string): Promise<string | null>;
  set(reference: string, value: string): Promise<void>;
  delete(reference: string): Promise<void>;
}

export interface SecretCommandResult {
  code: number;
  stdout: string;
}

export type SecretCommandRunner = (
  executable: string,
  arguments_: readonly string[],
  stdin?: string,
) => Promise<SecretCommandResult>;

export class NodeIdentityStore {
  readonly metadataPath: string;

  constructor(
    metadataPath: string,
    private readonly secrets: ProtectedSecretStore,
    private readonly options: { allowLegacyMigration?: boolean } = {},
  ) {
    this.metadataPath = resolve(metadataPath);
  }

  async loadOrCreate(identityId: string): Promise<WorkerAdmissionSigner> {
    const existing = await this.readMetadata(identityId);
    if (existing) {
      if (existing.identityId !== identityId) throw new Error("node_identity_id_mismatch");
      if (existing.provider !== this.secrets.provider) throw new Error("node_identity_provider_mismatch");
      const value = await this.secrets.get(existing.secretReference);
      if (!value) throw new Error("node_identity_protected_secret_is_missing");
      const credential = parseWorkerAdmissionCredential(JSON.parse(value) as unknown);
      if (credential.publicKeySpki !== existing.publicKeySpki) {
        throw new Error("node_identity_public_key_mismatch");
      }
      return workerAdmissionSigner(credential);
    }

    const credential = generateWorkerAdmissionCredential();
    const reference = `node-${identityId}-${randomBytes(12).toString("hex")}`;
    await this.secrets.set(reference, JSON.stringify(credential));
    try {
      await this.writeMetadata({
        schema: "mycellios-node-identity/1",
        identityId,
        provider: this.secrets.provider,
        secretReference: reference,
        algorithm: "ed25519",
        publicKeySpki: credential.publicKeySpki,
        createdAt: new Date().toISOString(),
      });
    } catch (error) {
      await this.secrets.delete(reference).catch(() => undefined);
      throw error;
    }
    return workerAdmissionSigner(credential);
  }

  private async readMetadata(identityId: string): Promise<z.infer<typeof metadataSchema> | null> {
    let document: unknown;
    try { document = JSON.parse(await readFile(this.metadataPath, "utf8")); } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
      throw new Error("node_identity_metadata_is_invalid", { cause: error });
    }
    const metadata = metadataSchema.safeParse(document);
    if (metadata.success) return metadata.data;

    let legacy: WorkerAdmissionCredential;
    try { legacy = parseWorkerAdmissionCredential(document); } catch (error) {
      throw new Error("node_identity_metadata_is_invalid", { cause: error });
    }
    if (this.options.allowLegacyMigration !== true) {
      throw new Error("node_identity_legacy_migration_confirmation_required");
    }
    const reference = `node-${identityId}-${randomBytes(12).toString("hex")}`;
    await this.secrets.set(reference, JSON.stringify(legacy));
    const migrated = metadataSchema.parse({
      schema: "mycellios-node-identity/1",
      identityId,
      provider: this.secrets.provider,
      secretReference: reference,
      algorithm: "ed25519",
      publicKeySpki: legacy.publicKeySpki,
      createdAt: new Date().toISOString(),
    });
    try {
      await this.writeMetadata(migrated);
    } catch (error) {
      await this.secrets.delete(reference).catch(() => undefined);
      throw error;
    }
    return migrated;
  }

  private async writeMetadata(value: z.infer<typeof metadataSchema>): Promise<void> {
    await mkdir(dirname(this.metadataPath), { recursive: true, mode: 0o700 });
    const temporary = `${this.metadataPath}.tmp-${process.pid}-${randomBytes(6).toString("hex")}`;
    try {
      const file = await open(temporary, "wx", 0o600);
      try {
        await file.writeFile(`${JSON.stringify(value, null, 2)}\n`, "utf8");
        await file.sync();
      } finally {
        await file.close();
      }
      await rename(temporary, this.metadataPath);
    } finally {
      await rm(temporary, { force: true });
    }
  }
}

export class LinuxSecretToolStore implements ProtectedSecretStore {
  readonly provider = "linux-secret-service";

  constructor(private readonly run: SecretCommandRunner = runSecretCommand) {}

  async get(reference: string): Promise<string | null> {
    const result = await this.run("secret-tool", ["lookup", "application", "mycellios", "identity", reference]);
    if (result.code === 1) return null;
    if (result.code !== 0) throw new Error("node_identity_secret_service_lookup_failed");
    return result.stdout.replace(/\r?\n$/, "");
  }

  async set(reference: string, value: string): Promise<void> {
    const result = await this.run("secret-tool", [
      "store",
      "--label=mycellios node identity",
      "application",
      "mycellios",
      "identity",
      reference,
    ], value);
    if (result.code !== 0) throw new Error("node_identity_secret_service_store_failed");
  }

  async delete(reference: string): Promise<void> {
    const result = await this.run("secret-tool", ["clear", "application", "mycellios", "identity", reference]);
    if (result.code !== 0 && result.code !== 1) {
      throw new Error("node_identity_secret_service_delete_failed");
    }
  }
}

/** Persistent headless identity storage encrypted by systemd-creds (TPM2/host key). */
export class LinuxSystemdCredentialStore implements ProtectedSecretStore {
  readonly provider = "linux-systemd-creds";
  private readonly root: string;

  constructor(root: string, private readonly run: SecretCommandRunner = runSecretCommand) {
    this.root = resolve(root);
  }

  async get(reference: string): Promise<string | null> {
    const path = this.secretPath(reference);
    try { await readFile(path); } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
      throw error;
    }
    const result = await this.run("systemd-creds", ["decrypt", `--name=${reference}`, path, "-"]);
    if (result.code !== 0) throw new Error("node_identity_systemd_creds_decrypt_failed");
    return result.stdout;
  }

  async set(reference: string, value: string): Promise<void> {
    const path = this.secretPath(reference);
    await mkdir(this.root, { recursive: true, mode: 0o700 });
    const temporary = `${path}.tmp-${process.pid}-${randomBytes(6).toString("hex")}`;
    try {
      const result = await this.run("systemd-creds", ["encrypt", `--name=${reference}`, "-", temporary], value);
      if (result.code !== 0) throw new Error("node_identity_systemd_creds_encrypt_failed");
      await (await import("node:fs/promises")).chmod(temporary, 0o600);
      await rename(temporary, path);
    } finally {
      await rm(temporary, { force: true });
    }
  }

  async delete(reference: string): Promise<void> { await rm(this.secretPath(reference), { force: true }); }

  private secretPath(reference: string): string {
    assertSecretReference(reference);
    return join(this.root, `${reference}.cred`);
  }
}

const WINDOWS_DPAPI_SCRIPT = [
  "$ErrorActionPreference='Stop'",
  "$request=[Console]::In.ReadToEnd()|ConvertFrom-Json",
  "if($request.operation -eq 'protect') {",
  "  $plain=[Text.Encoding]::UTF8.GetBytes([string]$request.value)",
  "  $cipher=[Security.Cryptography.ProtectedData]::Protect($plain,$null,[Security.Cryptography.DataProtectionScope]::LocalMachine)",
  "  [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName([string]$request.path))|Out-Null",
  "  $temporary=([string]$request.path)+'.tmp-'+[Guid]::NewGuid().ToString('N')",
  "  [IO.File]::WriteAllBytes($temporary,$cipher)",
  "  [IO.File]::Move($temporary,[string]$request.path,$true)",
  "} elseif($request.operation -eq 'unprotect') {",
  "  if(-not [IO.File]::Exists([string]$request.path)) { exit 3 }",
  "  $cipher=[IO.File]::ReadAllBytes([string]$request.path)",
  "  $plain=[Security.Cryptography.ProtectedData]::Unprotect($cipher,$null,[Security.Cryptography.DataProtectionScope]::LocalMachine)",
  "  [Console]::Out.Write([Text.Encoding]::UTF8.GetString($plain))",
  "} else { throw 'unsupported operation' }",
].join(";");

export class WindowsDpapiSecretStore implements ProtectedSecretStore {
  readonly provider = "windows-dpapi-local-machine";
  private readonly root: string;

  constructor(
    root: string,
    private readonly run: SecretCommandRunner = runSecretCommand,
  ) {
    this.root = resolve(root);
  }

  async get(reference: string): Promise<string | null> {
    const result = await this.invoke("unprotect", reference);
    if (result.code === 3) return null;
    if (result.code !== 0) throw new Error("node_identity_dpapi_unprotect_failed");
    return result.stdout;
  }

  async set(reference: string, value: string): Promise<void> {
    const result = await this.invoke("protect", reference, value);
    if (result.code !== 0) throw new Error("node_identity_dpapi_protect_failed");
  }

  async delete(reference: string): Promise<void> {
    await rm(this.secretPath(reference), { force: true });
  }

  private async invoke(
    operation: "protect" | "unprotect",
    reference: string,
    value?: string,
  ): Promise<SecretCommandResult> {
    return await this.run(
      "powershell.exe",
      ["-NoLogo", "-NoProfile", "-NonInteractive", "-Command", WINDOWS_DPAPI_SCRIPT],
      JSON.stringify({ operation, path: this.secretPath(reference), ...(value === undefined ? {} : { value }) }),
    );
  }

  private secretPath(reference: string): string {
    assertSecretReference(reference);
    return join(this.root, `${reference}.dpapi`);
  }
}

export class MacOsKeychainSecretStore implements ProtectedSecretStore {
  readonly provider = "macos-keychain";

  constructor(private readonly run: SecretCommandRunner = runSecretCommand) {}

  async get(reference: string): Promise<string | null> {
    assertSecretReference(reference);
    const result = await this.run("/usr/bin/security", [
      "find-generic-password", "-k", "/Library/Keychains/System.keychain", "-a", reference, "-s", "mycellios.node.identity", "-w",
    ]);
    if (result.code === 44) return null;
    if (result.code !== 0) throw new Error("node_identity_keychain_lookup_failed");
    return result.stdout.replace(/\r?\n$/, "");
  }

  async set(reference: string, value: string): Promise<void> {
    assertSecretReference(reference);
    const command = [
      "add-generic-password -U",
      "-k '/Library/Keychains/System.keychain'",
      `-a ${shellQuote(reference)}`,
      "-s 'mycellios.node.identity'",
      `-w ${shellQuote(value)}`,
    ].join(" ");
    const result = await this.run("/usr/bin/security", ["-i"], `${command}\nquit\n`);
    if (result.code !== 0) throw new Error("node_identity_keychain_store_failed");
  }

  async delete(reference: string): Promise<void> {
    assertSecretReference(reference);
    const result = await this.run("/usr/bin/security", [
      "delete-generic-password", "-k", "/Library/Keychains/System.keychain", "-a", reference, "-s", "mycellios.node.identity",
    ]);
    if (result.code !== 0 && result.code !== 44) {
      throw new Error("node_identity_keychain_delete_failed");
    }
  }
}

export async function runSecretCommand(
  executable: string,
  arguments_: readonly string[],
  stdin?: string,
): Promise<SecretCommandResult> {
  return await new Promise((resolve, reject) => {
    const child = spawn(executable, [...arguments_], {
      shell: false,
      stdio: ["pipe", "pipe", "ignore"],
      windowsHide: true,
    });
    const chunks: Buffer[] = [];
    child.stdout.on("data", (chunk: Buffer) => chunks.push(Buffer.from(chunk)));
    child.once("error", (error) => reject(new Error("node_identity_protected_store_unavailable", { cause: error })));
    child.once("close", (code) => resolve({ code: code ?? -1, stdout: Buffer.concat(chunks).toString("utf8") }));
    child.stdin.end(stdin ?? "");
  });
}

function assertSecretReference(reference: string): void {
  if (!/^[A-Za-z0-9._:-]{1,200}$/.test(reference)) {
    throw new Error("node_identity_secret_reference_is_invalid");
  }
}

function shellQuote(value: string): string {
  if (value.includes("\0") || value.includes("\n") || value.includes("\r")) {
    throw new Error("node_identity_keychain_value_is_invalid");
  }
  return `'${value.replaceAll("'", "'\\''")}'`;
}

export function serializeCredentialForProtectedStore(
  credential: WorkerAdmissionCredential,
): string {
  return JSON.stringify(parseWorkerAdmissionCredential(credential));
}
