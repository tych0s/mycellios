import {
  createHash,
  createPublicKey,
  randomBytes,
  randomUUID,
  timingSafeEqual,
} from "node:crypto";
import { constants } from "node:fs";
import {
  access,
  link,
  mkdir,
  open,
  readFile,
  rename,
  rm,
  stat,
  unlink,
  utimes,
  writeFile,
} from "node:fs/promises";
import {
  dirname,
  join,
  parse as parsePath,
  resolve,
} from "node:path";
import { hostname } from "node:os";
import { z } from "zod";

import {
  developmentLabCreateRequestSchema,
  developmentLabIdSchema,
  developmentLabInvitationTokenSchema,
  developmentLabPublicKeySchema,
  developmentLabSummarySchema,
  type DevelopmentLabCreateRequest,
  type DevelopmentLabSummary,
} from "../contracts/development-lab.js";
import {
  verifyComponentUpdateManifest,
} from "../contracts/component-update-manifest.js";
import {
  ComponentReleaseStore,
} from "./component-release-store.js";

const STATE_SCHEMA = "mycellios-development-labs/1" as const;
const DEFAULT_INVITATION_TTL_SECONDS = 24 * 60 * 60;
const MAX_INVITATIONS_PER_LAB = 16;
const DEFAULT_LOCK_TIMEOUT_MS = 15_000;
const DEFAULT_LOCK_STALE_MS = 10_000;
const DEFAULT_LOCK_RETRY_MS = 50;

const storedInvitationSchema = z.object({
  tokenHash: z.string().regex(/^[0-9a-f]{64}$/),
  createdAt: z.string().datetime({ offset: true }),
  expiresAt: z.string().datetime({ offset: true }),
  redeemedAt: z.string().datetime({ offset: true }).nullable(),
}).strict();

const storedLabSchema = developmentLabSummarySchema.extend({
  invitations: z.array(storedInvitationSchema).max(MAX_INVITATIONS_PER_LAB),
}).strict();

const persistedStateSchema = z.object({
  schema: z.literal(STATE_SCHEMA),
  labs: z.array(storedLabSchema),
}).strict();

type StoredInvitation = z.infer<typeof storedInvitationSchema>;
type StoredLab = z.infer<typeof storedLabSchema>;
type PersistedState = z.infer<typeof persistedStateSchema>;

export type DevelopmentLabStoreErrorCode =
  | "development_labs_not_configured"
  | "development_lab_state_invalid"
  | "development_lab_not_found"
  | "development_lab_revoked"
  | "development_lab_invitation_invalid"
  | "development_lab_invitation_expired"
  | "development_lab_invitation_used"
  | "development_lab_signing_key_invalid"
  | "development_lab_io_failed";

export class DevelopmentLabStoreError extends Error {
  override readonly name = "DevelopmentLabStoreError";

  constructor(
    readonly code: DevelopmentLabStoreErrorCode,
    options?: { cause?: unknown },
  ) {
    super(code, options);
  }
}

export interface DevelopmentLabInvitationResult {
  lab: DevelopmentLabSummary;
  token: string;
  expiresAt: string;
}

export interface DevelopmentLabStoreOptions {
  root: string;
  now?: (() => Date) | undefined;
  lockTimeoutMs?: number | undefined;
  lockStaleMs?: number | undefined;
  lockRetryMs?: number | undefined;
}

interface StateLockOwner {
  ownerId: string;
  host: string;
  pid: number;
  acquiredAt: string;
  encoded: string;
}

/**
 * Persistent control-plane state for private development channels.
 *
 * Raw invitation tokens are returned once and never written to disk. Every lab
 * owns an isolated ComponentReleaseStore whose verifier resolves the current
 * lab key at publication time, so adding a dev lab does not mutate the stable
 * desktop trust root.
 */
export class DevelopmentLabStore {
  readonly #root: string;
  readonly #statePath: string;
  readonly #lockPath: string;
  readonly #now: () => Date;
  readonly #lockTimeoutMs: number;
  readonly #lockStaleMs: number;
  readonly #lockRetryMs: number;
  readonly #releaseStores = new Map<string, ComponentReleaseStore>();
  #state: PersistedState | null = null;
  #mutation: Promise<void> = Promise.resolve();

  constructor(options: DevelopmentLabStoreOptions) {
    if (typeof options?.root !== "string" || options.root.trim() === "") {
      throw new DevelopmentLabStoreError("development_labs_not_configured");
    }
    const root = resolve(options.root, "development-labs");
    if (root === parsePath(root).root) {
      throw new DevelopmentLabStoreError("development_labs_not_configured");
    }
    this.#root = root;
    this.#statePath = join(root, "state.json");
    this.#lockPath = join(root, "state.lock");
    this.#now = options.now ?? (() => new Date());
    this.#lockTimeoutMs = positiveInteger(
      options.lockTimeoutMs,
      DEFAULT_LOCK_TIMEOUT_MS,
    );
    this.#lockStaleMs = positiveInteger(
      options.lockStaleMs,
      DEFAULT_LOCK_STALE_MS,
    );
    this.#lockRetryMs = positiveInteger(
      options.lockRetryMs,
      DEFAULT_LOCK_RETRY_MS,
    );
  }

  async initialize(): Promise<void> {
    await this.#exclusive(async () => undefined);
  }

  async createLab(
    input: DevelopmentLabCreateRequest,
  ): Promise<DevelopmentLabInvitationResult> {
    const request = developmentLabCreateRequestSchema.parse(input);
    assertEd25519Key(request.keyId, request.spki);
    return this.#exclusive(async () => {
      const now = this.#now();
      const labId = `lab_${randomBytes(16).toString("base64url")}`;
      developmentLabIdSchema.parse(labId);
      const timestamp = now.toISOString();
      const lab: StoredLab = {
        labId,
        name: request.name ?? "Mycellios development lab",
        keyId: request.keyId,
        spki: request.spki,
        status: "active",
        createdAt: timestamp,
        updatedAt: timestamp,
        revokedAt: null,
        invitations: [],
      };
      const invitation = createStoredInvitation(
        now,
        request.invitationTtlSeconds ?? DEFAULT_INVITATION_TTL_SECONDS,
      );
      lab.invitations.push(invitation.stored);
      this.#state!.labs.push(lab);
      await this.#persist();
      return {
        lab: publicLab(lab),
        token: invitation.token,
        expiresAt: invitation.stored.expiresAt,
      };
    });
  }

  async createInvitation(
    labId: string,
    ttlSeconds = DEFAULT_INVITATION_TTL_SECONDS,
  ): Promise<DevelopmentLabInvitationResult> {
    const parsedLabId = developmentLabIdSchema.parse(labId);
    if (
      !Number.isInteger(ttlSeconds)
      || ttlSeconds < 60
      || ttlSeconds > 7 * 24 * 60 * 60
    ) {
      throw new DevelopmentLabStoreError(
        "development_lab_invitation_invalid",
      );
    }
    return this.#exclusive(async () => {
      const lab = this.#requiredLab(parsedLabId);
      assertLabActive(lab);
      const invitation = createStoredInvitation(this.#now(), ttlSeconds);
      lab.invitations = [
        ...lab.invitations
          .filter((candidate) => candidate.redeemedAt === null)
          .slice(-(MAX_INVITATIONS_PER_LAB - 1)),
        invitation.stored,
      ];
      lab.updatedAt = this.#now().toISOString();
      await this.#persist();
      return {
        lab: publicLab(lab),
        token: invitation.token,
        expiresAt: invitation.stored.expiresAt,
      };
    });
  }

  async redeemInvitation(
    labId: string,
    token: string,
  ): Promise<DevelopmentLabSummary> {
    const parsedLabId = developmentLabIdSchema.parse(labId);
    const parsedToken = developmentLabInvitationTokenSchema.parse(token);
    return this.#exclusive(async () => {
      const lab = this.#requiredLab(parsedLabId);
      assertLabActive(lab);
      const tokenHash = hashInvitationToken(parsedToken);
      const invitation = lab.invitations.find((candidate) =>
        hashesEqual(candidate.tokenHash, tokenHash)
      );
      if (!invitation) {
        throw new DevelopmentLabStoreError(
          "development_lab_invitation_invalid",
        );
      }
      if (invitation.redeemedAt !== null) {
        throw new DevelopmentLabStoreError(
          "development_lab_invitation_used",
        );
      }
      const now = this.#now();
      if (Date.parse(invitation.expiresAt) <= now.getTime()) {
        throw new DevelopmentLabStoreError(
          "development_lab_invitation_expired",
        );
      }
      invitation.redeemedAt = now.toISOString();
      lab.updatedAt = invitation.redeemedAt;
      await this.#persist();
      return publicLab(lab);
    });
  }

  async revokeLab(labId: string): Promise<DevelopmentLabSummary> {
    const parsedLabId = developmentLabIdSchema.parse(labId);
    return this.#exclusive(async () => {
      const lab = this.#requiredLab(parsedLabId);
      if (lab.status !== "revoked") {
        const now = this.#now().toISOString();
        lab.status = "revoked";
        lab.revokedAt = now;
        lab.updatedAt = now;
        await this.#persist();
      }
      return publicLab(lab);
    });
  }

  async getLab(labId: string): Promise<DevelopmentLabSummary | null> {
    const parsedLabId = developmentLabIdSchema.parse(labId);
    return this.#exclusive(async () => {
      const lab = this.#state!.labs.find(
        (candidate) => candidate.labId === parsedLabId,
      );
      return lab ? publicLab(lab) : null;
    });
  }

  async releaseStore(labId: string): Promise<ComponentReleaseStore> {
    const parsedLabId = developmentLabIdSchema.parse(labId);
    return this.#exclusive(async () => {
      const lab = this.#requiredLab(parsedLabId);
      assertLabActive(lab);
      const existing = this.#releaseStores.get(parsedLabId);
      if (existing) return existing;
      const store = new ComponentReleaseStore({
        root: join(this.#root, "labs", parsedLabId, "releases"),
        verification: {
          verifyManifest: async (manifest, context) => {
            const current = await this.getLab(parsedLabId);
            if (!current) {
              throw new DevelopmentLabStoreError(
                "development_lab_not_found",
              );
            }
            if (current.status !== "active") {
              throw new DevelopmentLabStoreError(
                "development_lab_revoked",
              );
            }
            if (
              manifest.channel !== "dev"
              || context.target.channel !== "dev"
            ) {
              throw new DevelopmentLabStoreError(
                "development_lab_signing_key_invalid",
              );
            }
            verifyComponentUpdateManifest(manifest, {
              pinnedKey: {
                keyId: current.keyId,
                spki: current.spki,
              },
              expectedChannel: "dev",
              ...(context.minimumSequence === null
                ? {}
                : { minimumSequence: context.minimumSequence }),
            });
          },
        },
      });
      this.#releaseStores.set(parsedLabId, store);
      return store;
    });
  }

  async #load(): Promise<void> {
    try {
      await mkdir(this.#root, { recursive: true });
      let decoded: unknown;
      try {
        decoded = JSON.parse(await readFile(this.#statePath, "utf8"));
      } catch (error) {
        if (isMissing(error)) {
          this.#state = { schema: STATE_SCHEMA, labs: [] };
          await this.#persist();
          return;
        }
        throw error;
      }
      const state = persistedStateSchema.parse(decoded);
      const ids = new Set<string>();
      for (const lab of state.labs) {
        if (ids.has(lab.labId)) {
          throw new DevelopmentLabStoreError(
            "development_lab_state_invalid",
          );
        }
        ids.add(lab.labId);
        assertEd25519Key(lab.keyId, lab.spki);
      }
      this.#state = state;
    } catch (error) {
      if (error instanceof DevelopmentLabStoreError) throw error;
      if (error instanceof z.ZodError) {
        throw new DevelopmentLabStoreError(
          "development_lab_state_invalid",
          { cause: error },
        );
      }
      throw new DevelopmentLabStoreError(
        "development_lab_io_failed",
        { cause: error },
      );
    }
  }

  async #persist(): Promise<void> {
    if (!this.#state) {
      throw new DevelopmentLabStoreError("development_lab_state_invalid");
    }
    const bytes = Buffer.from(
      `${JSON.stringify(this.#state, null, 2)}\n`,
      "utf8",
    );
    const staging = join(
      dirname(this.#statePath),
      `.state.${randomUUID()}.tmp`,
    );
    try {
      await mkdir(dirname(this.#statePath), { recursive: true });
      await writeFile(staging, bytes, { flag: "wx", mode: 0o600 });
      const file = await open(staging, "r+");
      try {
        await file.sync();
      } finally {
        await file.close();
      }
      await rename(staging, this.#statePath);
      await syncDirectory(dirname(this.#statePath));
    } catch (error) {
      await rm(staging, { force: true }).catch(() => undefined);
      throw new DevelopmentLabStoreError(
        "development_lab_io_failed",
        { cause: error },
      );
    }
  }

  #requiredLab(labId: string): StoredLab {
    const lab = this.#state!.labs.find(
      (candidate) => candidate.labId === labId,
    );
    if (!lab) {
      throw new DevelopmentLabStoreError("development_lab_not_found");
    }
    return lab;
  }

  #exclusive<T>(operation: () => Promise<T>): Promise<T> {
    const guarded = async () => this.#withStateLock(async () => {
      await this.#load();
      return operation();
    });
    const run = this.#mutation.then(guarded, guarded);
    this.#mutation = run.then(() => undefined, () => undefined);
    return run;
  }

  async #withStateLock<T>(operation: () => Promise<T>): Promise<T> {
    const owner = await this.#acquireStateLock();
    const heartbeat = setInterval(() => {
      void this.#refreshStateLock(owner).catch(() => undefined);
    }, Math.max(250, Math.floor(this.#lockStaleMs / 3)));
    heartbeat.unref();
    try {
      return await operation();
    } finally {
      clearInterval(heartbeat);
      await this.#releaseStateLock(owner);
    }
  }

  async #acquireStateLock(): Promise<StateLockOwner> {
    await mkdir(this.#root, { recursive: true });
    const startedAt = Date.now();
    while (true) {
      const owner = createStateLockOwner();
      let handle: Awaited<ReturnType<typeof open>> | null = null;
      try {
        handle = await open(this.#lockPath, "wx", 0o600);
        await handle.writeFile(owner.encoded, "utf8");
        await handle.sync();
        await handle.close();
        return owner;
      } catch (error) {
        await handle?.close().catch(() => undefined);
        if (!isAlreadyExists(error)) {
          await this.#removeOwnedStateLock(owner).catch(() => undefined);
          throw new DevelopmentLabStoreError(
            "development_lab_io_failed",
            { cause: error },
          );
        }
        if (await this.#recoverStaleStateLock()) {
          continue;
        }
        if (Date.now() - startedAt >= this.#lockTimeoutMs) {
          throw new DevelopmentLabStoreError(
            "development_lab_io_failed",
            { cause: new Error("development_lab_state_lock_timeout") },
          );
        }
        await delay(this.#lockRetryMs);
      }
    }
  }

  async #refreshStateLock(owner: StateLockOwner): Promise<void> {
    try {
      if (await readFile(this.#lockPath, "utf8") !== owner.encoded) return;
      const now = new Date();
      await utimes(this.#lockPath, now, now);
    } catch (error) {
      if (!isMissing(error)) throw error;
    }
  }

  async #releaseStateLock(owner: StateLockOwner): Promise<void> {
    try {
      await this.#removeOwnedStateLock(owner);
    } catch (error) {
      throw new DevelopmentLabStoreError(
        "development_lab_io_failed",
        { cause: error },
      );
    }
  }

  async #removeOwnedStateLock(owner: StateLockOwner): Promise<boolean> {
    return compareAndUnlink(
      this.#lockPath,
      (encoded) => encoded === owner.encoded,
    );
  }

  async #recoverStaleStateLock(): Promise<boolean> {
    try {
      return await compareAndUnlink(
        this.#lockPath,
        (encoded, modifiedAt) =>
          Date.now() - modifiedAt >= this.#lockStaleMs
          && lockOwnerProcessDefinitelyDead(encoded),
      );
    } catch (error) {
      throw new DevelopmentLabStoreError(
        "development_lab_io_failed",
        { cause: error },
      );
    }
  }
}

function positiveInteger(
  value: number | undefined,
  fallback: number,
): number {
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new DevelopmentLabStoreError("development_labs_not_configured");
  }
  return value;
}

function createStateLockOwner(): StateLockOwner {
  const owner = {
    ownerId: randomUUID(),
    host: hostname(),
    pid: process.pid,
    acquiredAt: new Date().toISOString(),
  };
  return {
    ...owner,
    encoded: JSON.stringify(owner),
  };
}

function lockOwnerProcessDefinitelyDead(encoded: string): boolean {
  let candidate: unknown;
  try {
    candidate = JSON.parse(encoded);
  } catch {
    return false;
  }
  if (
    !candidate
    || typeof candidate !== "object"
    || !("host" in candidate)
    || candidate.host !== hostname()
    || !("pid" in candidate)
    || typeof candidate.pid !== "number"
    || !Number.isSafeInteger(candidate.pid)
    || candidate.pid <= 0
  ) {
    return false;
  }
  try {
    process.kill(candidate.pid, 0);
    return false;
  } catch (error) {
    return Boolean(
      error
      && typeof error === "object"
      && "code" in error
      && error.code === "ESRCH",
    );
  }
}

/**
 * Removes a lock only while a hard-link still identifies the same filesystem
 * object and its immutable owner payload matches. The hard-link prevents a
 * stale contender from mistaking a replacement lock for the one it observed.
 */
async function compareAndUnlink(
  lockPath: string,
  matches: (encoded: string, modifiedAt: number) => boolean,
): Promise<boolean> {
  const claimPath = `${lockPath}.${randomUUID()}.claim`;
  try {
    await link(lockPath, claimPath);
  } catch (error) {
    if (isMissing(error)) return true;
    throw error;
  }
  try {
    const [encoded, claimed] = await Promise.all([
      readFile(claimPath, "utf8"),
      stat(claimPath),
    ]);
    if (!matches(encoded, claimed.mtimeMs)) return false;
    const [current, currentEncoded] = await Promise.all([
      stat(lockPath),
      readFile(lockPath, "utf8"),
    ]);
    if (
      claimed.dev !== current.dev
      || claimed.ino !== current.ino
      || currentEncoded !== encoded
    ) {
      return false;
    }
    await unlink(lockPath);
    return true;
  } catch (error) {
    if (isMissing(error)) return true;
    throw error;
  } finally {
    await rm(claimPath, { force: true }).catch(() => undefined);
  }
}

function createStoredInvitation(
  now: Date,
  ttlSeconds: number,
): { token: string; stored: StoredInvitation } {
  const token = randomBytes(32).toString("base64url");
  developmentLabInvitationTokenSchema.parse(token);
  return {
    token,
    stored: {
      tokenHash: hashInvitationToken(token),
      createdAt: now.toISOString(),
      expiresAt: new Date(
        now.getTime() + ttlSeconds * 1_000,
      ).toISOString(),
      redeemedAt: null,
    },
  };
}

function hashInvitationToken(token: string): string {
  return createHash("sha256").update(token, "utf8").digest("hex");
}

function hashesEqual(left: string, right: string): boolean {
  const leftBytes = Buffer.from(left, "hex");
  const rightBytes = Buffer.from(right, "hex");
  return leftBytes.length === rightBytes.length
    && timingSafeEqual(leftBytes, rightBytes);
}

function publicLab(lab: StoredLab): DevelopmentLabSummary {
  return developmentLabSummarySchema.parse({
    labId: lab.labId,
    name: lab.name,
    keyId: lab.keyId,
    spki: lab.spki,
    status: lab.status,
    createdAt: lab.createdAt,
    updatedAt: lab.updatedAt,
    revokedAt: lab.revokedAt,
  });
}

function assertLabActive(lab: StoredLab): void {
  if (lab.status !== "active") {
    throw new DevelopmentLabStoreError("development_lab_revoked");
  }
}

function assertEd25519Key(keyId: string, spki: string): void {
  try {
    developmentLabPublicKeySchema.parse({ keyId, spki });
    const encoded = Buffer.from(spki, "base64url");
    if (
      encoded.length === 0
      || encoded.length > 128
      || encoded.toString("base64url") !== spki
    ) {
      throw new Error("non-canonical SPKI");
    }
    const key = createPublicKey({
      key: encoded,
      format: "der",
      type: "spki",
    });
    const canonical = key.export({ format: "der", type: "spki" });
    if (
      key.asymmetricKeyType !== "ed25519"
      || !Buffer.isBuffer(canonical)
      || !canonical.equals(encoded)
    ) {
      throw new Error("non-Ed25519 SPKI");
    }
  } catch (error) {
    throw new DevelopmentLabStoreError(
      "development_lab_signing_key_invalid",
      { cause: error },
    );
  }
}

function isMissing(error: unknown): boolean {
  return Boolean(
    error
    && typeof error === "object"
    && "code" in error
    && error.code === "ENOENT",
  );
}

function isAlreadyExists(error: unknown): boolean {
  return Boolean(
    error
    && typeof error === "object"
    && "code" in error
    && error.code === "EEXIST",
  );
}

async function delay(milliseconds: number): Promise<void> {
  await new Promise<void>((resolveDelay) => {
    setTimeout(resolveDelay, milliseconds);
  });
}

async function syncDirectory(path: string): Promise<void> {
  let handle: Awaited<ReturnType<typeof open>> | null = null;
  try {
    handle = await open(path, "r");
    await handle.sync();
  } catch (error) {
    if (
      process.platform === "win32"
      && error
      && typeof error === "object"
      && "code" in error
      && ["EINVAL", "ENOTSUP", "EPERM"].includes(String(error.code))
    ) {
      return;
    }
    throw error;
  } finally {
    await handle?.close();
  }
}

export async function developmentLabStateContainsRawToken(
  root: string,
  token: string,
): Promise<boolean> {
  const path = resolve(root, "development-labs", "state.json");
  try {
    await access(path, constants.R_OK);
    return (await readFile(path, "utf8")).includes(token);
  } catch (error) {
    if (isMissing(error)) return false;
    throw error;
  }
}
