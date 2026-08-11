import { createHash, randomUUID } from "node:crypto";
import {
  lstat,
  mkdir,
  open,
  readFile,
  readdir,
  rename,
  rm,
} from "node:fs/promises";
import { dirname, join, relative, resolve, sep } from "node:path";
import { z } from "zod";

import {
  verifyComponentUpdateManifest,
  type ComponentUpdateChannel,
  type ComponentUpdateComponent,
  type ComponentUpdateManifest,
  type PinnedComponentUpdateKey,
} from "../contracts/component-update-manifest.js";
import {
  COMPONENT_FILES_POLICY,
  MANAGED_COMPONENT_POLICIES,
  MANAGED_COMPONENT_RESTART_SCOPES,
  MAX_COMPONENT_ARTIFACT_BYTES,
} from "../contracts/component-update-policy.js";
import {
  extractComponentFilesPackage,
  inspectComponentFilesPackage,
  type ComponentFilesLimits,
} from "./component-files.js";

export const COMPONENT_INSTALL_STATE_SCHEMA =
  "mycellios-component-install-state/2" as const;
const LEGACY_COMPONENT_INSTALL_STATE_SCHEMA =
  "mycellios-component-install-state/1" as const;

const sha256IdentitySchema = z.string().regex(/^sha256:[0-9a-f]{64}$/);
const installedComponentSchema = z
  .object({
    id: z
      .string()
      .min(1)
      .max(128)
      .regex(/^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$/),
    channel: z.enum(["dev", "stable"]),
    version: z.string().min(1).max(128),
    revision: z.string().min(7).max(128).optional(),
    sourceTreeDirty: z.boolean().optional(),
    sourceTreeDigest: sha256IdentitySchema.optional(),
    installId: z.string().uuid().optional(),
    keyId: z.string().min(1).max(128).optional(),
    trustKeySha256: sha256IdentitySchema.optional(),
    artifactSha256: sha256IdentitySchema,
    filesManifestSha256: sha256IdentitySchema,
    manifestId: sha256IdentitySchema,
    activatedAt: z.string().datetime({ offset: true }),
  })
  .strict();
const acceptedChannelSchema = z
  .object({
    sequence: z.number().int().positive().safe(),
    manifestId: sha256IdentitySchema,
  })
  .strict();
const legacyInstalledComponentSchema = z
  .object({
    id: z
      .string()
      .min(1)
      .max(128)
      .regex(/^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$/),
    version: z.string().min(1).max(128),
    artifactSha256: sha256IdentitySchema,
    filesManifestSha256: sha256IdentitySchema,
    manifestId: sha256IdentitySchema,
    activatedAt: z.string().datetime({ offset: true }),
  })
  .strict();
const legacyComponentInstallStateSchema = z
  .object({
    schema: z.literal(LEGACY_COMPONENT_INSTALL_STATE_SCHEMA),
    channels: z
      .object({
        dev: acceptedChannelSchema.nullable(),
        stable: acceptedChannelSchema.nullable(),
      })
      .strict(),
    active: z.record(z.string(), legacyInstalledComponentSchema),
    previous: z.record(z.string(), legacyInstalledComponentSchema),
  })
  .strict();
const rejectedManifestSchema = z
  .object({
    manifestId: sha256IdentitySchema,
    environment: sha256IdentitySchema,
    attempts: z.number().int().positive().safe(),
    reason: z.string().min(1).max(256),
    rejectedAt: z.string().datetime({ offset: true }),
    retryAfter: z.string().datetime({ offset: true }).nullable(),
  })
  .strict();
const componentInUseLeaseSchema = z.object({
  schema: z.literal("mycellios-component-in-use/1"),
  token: z.string().uuid(),
  pid: z.number().int().positive(),
  acquiredAt: z.string().datetime({ offset: true }),
  components: z.record(z.string(), installedComponentSchema),
}).strict();
const componentInstallStateSchema = z
  .object({
    schema: z.literal(COMPONENT_INSTALL_STATE_SCHEMA),
    channels: z
      .object({
        dev: acceptedChannelSchema.nullable(),
        stable: acceptedChannelSchema.nullable(),
      })
      .strict(),
    active: z.record(z.string(), installedComponentSchema),
    previous: z.record(z.string(), installedComponentSchema),
    rejected: z
      .object({
        dev: rejectedManifestSchema.nullable(),
        stable: rejectedManifestSchema.nullable(),
      })
      .strict()
      .default({ dev: null, stable: null }),
  })
  .strict();

export type InstalledComponent = z.infer<typeof installedComponentSchema>;
export type ComponentInstallState = z.infer<typeof componentInstallStateSchema>;

export type ComponentUpdateApplyState =
  | "up-to-date"
  | "waiting-idle"
  | "applied";

export interface ComponentUpdateApplyResult {
  state: ComponentUpdateApplyState;
  manifest: ComponentUpdateManifest;
  changedComponents: string[];
}

export interface ComponentUpdateManagerOptions {
  storageRoot: string;
  feedBaseUrl: string;
  channel: ComponentUpdateChannel;
  pinnedKey?: PinnedComponentUpdateKey;
  pinnedKeys?: readonly PinnedComponentUpdateKey[];
  bootstrapVersion: string;
  workerProtocol: { min: number; max: number };
  runtimeAbi: string;
  /** A command may pin one exact signed manifest instead of accepting latest. */
  expectedManifestId?: `sha256:${string}`;
  platform?: NodeJS.Platform;
  arch?: NodeJS.Architecture;
  managedComponentIds?: readonly string[];
  maxArtifactBytes?: number;
  fetch?: typeof fetch;
  signal?: AbortSignal;
  shouldContinue?: () => boolean;
  isIdle: () => boolean | Promise<boolean>;
  /**
   * Atomically prevents new leases while activation is prepared. Returning
   * null means a race made the node busy again; the update remains staged.
   */
  acquireActivationLease?: () => Promise<
    (() => void | Promise<void>) | null
  >;
  /**
   * Runs canaries against the staged paths while the active pointer still
   * references the previous, already verified component.
   */
  onPrepare?: (result: {
    manifest: ComponentUpdateManifest;
    changedComponents: string[];
    stagedComponentRoots: Readonly<Record<string, string>>;
  }) => Promise<void>;
  /**
   * Called after a successfully canaried pointer has changed. The node hook
   * schedules the affected service/runtime restart after durable ACK. Throwing
   * restores the previous pointer before onRollback is called.
   */
  onActivate: (result: {
    manifest: ComponentUpdateManifest;
    changedComponents: string[];
    stagedComponentRoots: Readonly<Record<string, string>>;
  }) => Promise<void>;
  onRollback?: (result: {
    manifest: ComponentUpdateManifest;
    changedComponents: string[];
    previousComponentRoots: Readonly<Record<string, string | null>>;
    cause: unknown;
  }) => Promise<void>;
}

const EMPTY_STATE: ComponentInstallState = {
  schema: COMPONENT_INSTALL_STATE_SCHEMA,
  channels: { dev: null, stable: null },
  active: {},
  previous: {},
  rejected: { dev: null, stable: null },
};

/**
 * Installs small, signed product components side-by-side. It never mutates the
 * bundled service resources and changes the active pointer only while the
 * caller reports the node idle.
 */
export class ComponentUpdateManager {
  readonly storageRoot: string;
  private readonly options: ComponentUpdateManagerOptions;
  private readonly fetchImpl: typeof fetch;
  private readonly managedComponentIds: ReadonlySet<string>;
  private readonly maxArtifactBytes: number;
  private readonly pinnedKeys: readonly PinnedComponentUpdateKey[];
  private running: Promise<ComponentUpdateApplyResult> | null = null;

  constructor(options: ComponentUpdateManagerOptions) {
    this.storageRoot = resolve(options.storageRoot);
    this.options = options;
    this.fetchImpl = options.fetch ?? fetch;
    this.managedComponentIds = new Set(
      options.managedComponentIds ?? ["python-product"],
    );
    this.pinnedKeys = options.pinnedKeys
      ?? (options.pinnedKey ? [options.pinnedKey] : []);
    if (this.pinnedKeys.length === 0) {
      throw new Error("component_update_pinned_key_missing");
    }
    this.maxArtifactBytes =
      options.maxArtifactBytes ?? MAX_COMPONENT_ARTIFACT_BYTES;
    if (this.maxArtifactBytes > MAX_COMPONENT_ARTIFACT_BYTES) {
      throw new Error("component_update_max_artifact_bytes_exceeds_policy");
    }
    assertFeedBaseUrl(options.feedBaseUrl);
    if (
      !Number.isSafeInteger(this.maxArtifactBytes)
      || this.maxArtifactBytes < 1
    ) {
      throw new Error("component_update_max_artifact_bytes_invalid");
    }
  }

  checkAndApply(): Promise<ComponentUpdateApplyResult> {
    this.running ??= this.checkAndApplyOnce().finally(() => {
      this.running = null;
    });
    return this.running;
  }

  private async checkAndApplyOnce(): Promise<ComponentUpdateApplyResult> {
    await ensureSafeUpdateDirectory(this.storageRoot, this.storageRoot);
    const before = await readComponentInstallState(this.storageRoot);
    const accepted = before.channels[this.options.channel];
    const manifest = await this.fetchManifest(accepted?.sequence);
    this.assertCurrent();
    if (
      this.options.expectedManifestId
      && manifest.manifestId !== this.options.expectedManifestId
    ) throw new Error("component_update_manifest_id_mismatch");
    const rejection = before.rejected[this.options.channel];
    if (
      rejection?.manifestId === manifest.manifestId
      && rejection.environment === this.environmentFingerprint()
      && (
        rejection.retryAfter === null
        || Date.parse(rejection.retryAfter) > Date.now()
      )
    ) {
      throw new Error(
        `component_update_manifest_rejected:${rejection.reason}`,
      );
    }
    try {
    if (
      accepted
      && accepted.sequence === manifest.sequence
      && accepted.manifestId !== manifest.manifestId
    ) {
      throw new Error("component_update_manifest_sequence_equivocation");
    }
    assertManifestCompatibility(manifest, this.options);
    const selected = selectApplicableComponents(
      manifest.components,
      this.options.platform ?? process.platform,
      this.options.arch ?? process.arch,
    );
    for (const component of selected) {
      if (component.requirements.runtimeAbi !== this.options.runtimeAbi) {
        throw new Error(
          `component_update_component_runtime_abi_incompatible:${component.id}`,
        );
      }
      const componentProtocol = component.requirements.workerProtocol;
      if (
        this.options.workerProtocol.max < componentProtocol.min
        || componentProtocol.max < this.options.workerProtocol.min
      ) {
        throw new Error(
          `component_update_component_worker_protocol_incompatible:${component.id}`,
        );
      }
      if (!this.managedComponentIds.has(component.id)) {
        throw new Error(`component_update_component_not_managed:${component.id}`);
      }
      if (
        component.artifact.format !== "json-gzip-v1"
        || !component.artifact.filesManifestSha256
      ) {
        throw new Error(
          `component_update_component_format_not_supported:${component.id}`,
        );
      }
      const expectedRestartScope =
        MANAGED_COMPONENT_RESTART_SCOPES[
          component.id as keyof typeof MANAGED_COMPONENT_RESTART_SCOPES
        ];
      if (
        !expectedRestartScope
      ) {
        throw new Error(`component_update_policy_missing:${component.id}`);
      }
      if (component.restartScope !== expectedRestartScope
      ) {
        throw new Error(
          `component_update_restart_scope_invalid:${component.id}`,
        );
      }
      assertManagedComponentPolicy(
        component,
        this.options.platform ?? process.platform,
        this.options.arch ?? process.arch,
      );
    }

    const manifestTrustKeySha256 =
      this.manifestTrustFingerprint(manifest);
    const changed: ComponentUpdateComponent[] = [];
    const validCurrentComponents = new Set<string>();
    for (const component of selected) {
      const current = before.active[component.id];
      const componentRoot = current?.channel === manifest.channel
        ? installedComponentRoot(this.storageRoot, current)
        : null;
      const currentTrusted = current !== undefined && this.pinnedKeys.some((key) =>
        key.keyId === current.keyId
        && componentTrustFingerprint(key.spki) === current.trustKeySha256
      );
      const currentUsable =
        current?.channel === manifest.channel
        && currentTrusted
        && componentRoot !== null
        && await isSafeDirectory(componentRoot)
        && await verifyInstalledComponentRoot(this.storageRoot, current, componentRoot);
      if (currentUsable) validCurrentComponents.add(component.id);
      const currentMatchesTarget = currentUsable
        && current.keyId === manifest.keyId
        && current.trustKeySha256 === manifestTrustKeySha256
        && current.artifactSha256 === component.artifact.sha256
        && current.filesManifestSha256 === component.artifact.filesManifestSha256;
      if (!currentMatchesTarget) changed.push(component);
    }
    assertResolvedComponentDependencies(selected, before, validCurrentComponents);
    if (
      accepted?.manifestId === manifest.manifestId
      && changed.length === 0
    ) {
      if (before.rejected[this.options.channel]) {
        const cleaned = cloneState(before);
        cleaned.rejected[this.options.channel] = null;
        await writeComponentInstallState(this.storageRoot, cleaned);
        await this.collectGarbage(cleaned).catch(() => undefined);
      } else {
        await this.collectGarbage(before).catch(() => undefined);
      }
      return {
        state: "up-to-date",
        manifest,
        changedComponents: [],
      };
    }

    if (changed.length === 0) {
      this.assertCurrent();
      const metadataOnly = cloneState(before);
      metadataOnly.rejected[this.options.channel] = null;
      metadataOnly.channels[this.options.channel] = {
        sequence: manifest.sequence,
        manifestId: manifest.manifestId,
      };
      for (const component of selected) {
        const current = metadataOnly.active[component.id]!;
        metadataOnly.active[component.id] = {
          ...current,
          version: component.version,
          revision: manifest.revision,
          sourceTreeDirty: manifest.provenance.sourceTreeDirty,
          sourceTreeDigest: manifest.provenance.sourceTreeDigest,
          keyId: manifest.keyId,
          trustKeySha256: this.manifestTrustFingerprint(manifest),
          manifestId: manifest.manifestId,
        };
      }
      await writeComponentInstallState(this.storageRoot, metadataOnly);
      await this.collectGarbage(metadataOnly).catch(() => undefined);
      return {
        state: "up-to-date",
        manifest,
        changedComponents: [],
      };
    }

    const staged = new Map<string, {
      installed: InstalledComponent;
      root: string;
    }>();
    const referencedComponentRoots = new Set(
      [...Object.values(before.active), ...Object.values(before.previous)]
        .map((component) =>
          installedComponentRoot(this.storageRoot, component),
        ),
    );
    for (const root of await readComponentInUseRoots(this.storageRoot)) {
      referencedComponentRoots.add(root);
    }
    for (const component of changed) {
      staged.set(
        component.id,
        await this.stageComponent(
          component,
          manifest,
          referencedComponentRoots,
        ),
      );
    }
    if (!(await this.options.isIdle())) {
      return {
        state: "waiting-idle",
        manifest,
        changedComponents: changed.map(({ id }) => id),
      };
    }
    this.assertCurrent();

    const releaseActivationLease =
      await this.options.acquireActivationLease?.()
      ?? (this.options.acquireActivationLease ? null : async () => undefined);
    if (!releaseActivationLease) {
      return {
        state: "waiting-idle",
        manifest,
        changedComponents: changed.map(({ id }) => id),
      };
    }
    const changedComponents = changed.map(({ id }) => id);
    const stagedComponentRoots = Object.fromEntries(
      [...staged].map(([id, value]) => [id, value.root]),
    );
    const previousComponentRoots = Object.fromEntries(
      changedComponents.map((id) => [
        id,
        before.active[id] && validCurrentComponents.has(id)
          ? installedComponentRoot(this.storageRoot, before.active[id])
          : null,
      ]),
    );
    try {
      try {
        this.assertCurrent();
        await this.options.onPrepare?.({
          manifest,
          changedComponents,
          stagedComponentRoots,
        });
      } catch (cause) {
        throw new Error("component_update_pre_activation_canary_failed", {
          cause,
        });
      }

      this.assertCurrent();
      const after = cloneState(before);
      after.rejected[this.options.channel] = null;
      after.channels[this.options.channel] = {
        sequence: manifest.sequence,
        manifestId: manifest.manifestId,
      };
      for (const component of changed) {
        const previous = before.active[component.id];
        if (previous && validCurrentComponents.has(component.id)) {
          after.previous[component.id] = previous;
        } else {
          delete after.previous[component.id];
        }
        after.active[component.id] = staged.get(component.id)!.installed;
      }
      for (const component of selected) {
        if (staged.has(component.id)) continue;
        const current = after.active[component.id]!;
        after.active[component.id] = {
          ...current,
          version: component.version,
          revision: manifest.revision,
          sourceTreeDirty: manifest.provenance.sourceTreeDirty,
          sourceTreeDigest: manifest.provenance.sourceTreeDigest,
          keyId: manifest.keyId,
          trustKeySha256: this.manifestTrustFingerprint(manifest),
          manifestId: manifest.manifestId,
        };
      }
      try {
        this.assertCurrent();
        await this.options.onActivate({
          manifest,
          changedComponents,
          stagedComponentRoots,
        });
        this.assertCurrent();
        // Commit the active pointer only after the restarted runtime completed
        // its coordinator handshake. A crash before this write therefore
        // leaves the previously verified component authoritative on startup.
        await writeComponentInstallState(this.storageRoot, after);
        await this.collectGarbage(after).catch(() => undefined);
      } catch (cause) {
        const rollbackErrors: unknown[] = [cause];
        // The pointer is normally still `before`; writing it again also covers
        // an I/O failure whose atomic rename completed before reporting error.
        try {
          await writeComponentInstallState(this.storageRoot, before);
        } catch (stateRollbackError) {
          rollbackErrors.push(stateRollbackError);
        }
        try {
          await this.options.onRollback?.({
            manifest,
            changedComponents,
            previousComponentRoots,
            cause,
          });
        } catch (rollbackError) {
          rollbackErrors.push(rollbackError);
        }
        if (rollbackErrors.length > 1) {
          throw new AggregateError(
            rollbackErrors,
            "component_update_activation_and_rollback_failed",
          );
        }
        throw new Error("component_update_activation_failed_and_rolled_back", {
          cause,
        });
      }
      return {
        state: "applied",
        manifest,
        changedComponents,
      };
    } finally {
      await releaseActivationLease();
    }
    } catch (error) {
      if (!isUpdateCancellation(error)) {
        await this.recordManifestFailure(manifest, error);
      }
      throw error;
    }
  }

  private async recordManifestFailure(
    manifest: ComponentUpdateManifest,
    error: unknown,
  ): Promise<void> {
    const state = await readComponentInstallState(this.storageRoot);
    const accepted = state.channels[this.options.channel];
    if (
      accepted
      && (
        accepted.sequence > manifest.sequence
        || accepted.manifestId === manifest.manifestId
      )
    ) return;
    const previous = state.rejected[this.options.channel];
    const attempts =
      previous?.manifestId === manifest.manifestId
        ? previous.attempts + 1
        : 1;
    const now = Date.now();
    const deterministic = isDeterministicManifestFailure(error);
    const delayMs = Math.min(
      15 * 60_000,
      60_000 * (2 ** Math.min(attempts - 1, 4)),
    );
    state.rejected[this.options.channel] = {
      manifestId: manifest.manifestId,
      environment: this.environmentFingerprint(),
      attempts,
      reason: updateErrorCode(error).slice(0, 256),
      rejectedAt: new Date(now).toISOString(),
      retryAfter: deterministic
        ? null
        : new Date(now + delayMs).toISOString(),
    };
    await writeComponentInstallState(this.storageRoot, state);
  }

  private environmentFingerprint(): `sha256:${string}` {
    return `sha256:${createHash("sha256")
      .update(
        [
          this.options.bootstrapVersion,
          this.options.workerProtocol.min,
          this.options.workerProtocol.max,
          this.options.runtimeAbi,
          this.options.platform ?? process.platform,
          this.options.arch ?? process.arch,
        ].join("\n"),
      )
      .digest("hex")}`;
  }

  private manifestTrustFingerprint(
    manifest: ComponentUpdateManifest,
  ): `sha256:${string}` {
    const key = this.pinnedKeys.find(
      (candidate) => candidate.keyId === manifest.keyId,
    );
    if (!key) throw new Error("component_update_manifest_key_not_pinned");
    return componentTrustFingerprint(key.spki);
  }

  private assertCurrent(): void {
    if (this.options.shouldContinue?.() === false) {
      throw new Error("component_update_check_superseded");
    }
  }

  private async fetchManifest(
    minimumSequence: number | undefined,
  ): Promise<ComponentUpdateManifest> {
    const platform = this.options.platform ?? process.platform;
    const architecture = this.options.arch ?? process.arch;
    const url = new URL(
      `updates/v1/${this.options.channel}/${platform}/${architecture}/manifest.json`,
      ensureTrailingSlash(this.options.feedBaseUrl),
    );
    const response = await this.fetchImpl(url, {
      headers: { accept: "application/json" },
      redirect: "error",
      signal: this.requestSignal(15_000),
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`component_update_manifest_http_${response.status}`);
    }
    const length = Number(response.headers.get("content-length"));
    if (Number.isFinite(length) && length > 2 * 1024 * 1024) {
      throw new Error("component_update_manifest_too_large");
    }
    const bytes = await readBoundedResponse(
      response,
      2 * 1024 * 1024,
      "component_update_manifest_too_large",
    );
    let candidate: unknown;
    try {
      candidate = JSON.parse(bytes.toString("utf8")) as unknown;
    } catch {
      throw new Error("component_update_manifest_json_invalid");
    }
    const manifestKeyId =
      candidate
      && typeof candidate === "object"
      && !Array.isArray(candidate)
      && typeof (candidate as { keyId?: unknown }).keyId === "string"
        ? (candidate as { keyId: string }).keyId
        : "";
    const pinnedKey = this.pinnedKeys.find(
      (candidateKey) => candidateKey.keyId === manifestKeyId,
    );
    if (!pinnedKey) throw new Error("component_update_manifest_key_not_pinned");
    return verifyComponentUpdateManifest(candidate, {
      pinnedKey,
      expectedChannel: this.options.channel,
      ...(minimumSequence === undefined ? {} : { minimumSequence }),
    });
  }

  private async stageComponent(
    component: ComponentUpdateComponent,
    manifest: ComponentUpdateManifest,
    referencedComponentRoots: ReadonlySet<string>,
  ): Promise<{ installed: InstalledComponent; root: string }> {
    if (component.artifact.bytes > this.maxArtifactBytes) {
      throw new Error(`component_update_artifact_too_large:${component.id}`);
    }
    const digest = component.artifact.sha256.slice("sha256:".length);
    const artifactPath = join(this.storageRoot, "artifacts", digest);
    await ensureSafeUpdateDirectory(this.storageRoot, dirname(artifactPath));
    const bytes = await this.loadArtifact(component, artifactPath);
    const limits = this.componentFileLimits(component);
    const inspected = inspectComponentFilesPackage(bytes, limits);
    if (
      inspected.filesManifestSha256
      !== component.artifact.filesManifestSha256
    ) {
      throw new Error(
        `component_update_files_manifest_mismatch:${component.id}`,
      );
    }

    const componentDirectory = join(
      this.storageRoot,
      "components",
      component.id,
    );
    await ensureSafeUpdateDirectory(this.storageRoot, componentDirectory);
    const candidatePattern = new RegExp(
      `^${digest}\\.([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-`
      + `[89ab][0-9a-f]{3}-[0-9a-f]{12})$`,
    );
    for (
      const entry of await readdir(
        componentDirectory,
        { withFileTypes: true },
      )
    ) {
      const match = candidatePattern.exec(entry.name);
      if (!match) continue;
      const candidateRoot = join(componentDirectory, entry.name);
      if (referencedComponentRoots.has(candidateRoot)) continue;
      if (
        entry.isDirectory()
        && !entry.isSymbolicLink()
        && await this.installedComponentIsComplete(component, candidateRoot)
      ) {
        return {
          root: candidateRoot,
          installed: {
            id: component.id,
            channel: manifest.channel,
            version: component.version,
            revision: manifest.revision,
            sourceTreeDirty: manifest.provenance.sourceTreeDirty,
            sourceTreeDigest: manifest.provenance.sourceTreeDigest,
            installId: match[1]!,
            keyId: manifest.keyId,
            trustKeySha256: this.manifestTrustFingerprint(manifest),
            artifactSha256: component.artifact.sha256,
            filesManifestSha256: inspected.filesManifestSha256,
            manifestId: manifest.manifestId,
            activatedAt: new Date().toISOString(),
          },
        };
      }
      if (entry.isSymbolicLink()) {
        await rm(candidateRoot, { force: true });
      } else if (entry.isDirectory()) {
        await rm(candidateRoot, { recursive: true, force: true });
      }
    }
    const installId = randomUUID();
    const componentRoot = join(
      componentDirectory,
      `${digest}.${installId}`,
    );
    const staging = `${componentRoot}.staging-${randomUUID()}`;
    try {
      extractComponentFilesPackage(bytes, staging, { limits });
      await rename(staging, componentRoot);
    } finally {
      await rm(staging, { recursive: true, force: true });
    }
    if (!(await this.installedComponentIsComplete(component, componentRoot))) {
      throw new Error(`component_update_staged_component_incomplete:${component.id}`);
    }
    return {
      root: componentRoot,
      installed: {
        id: component.id,
        channel: manifest.channel,
        version: component.version,
        revision: manifest.revision,
        sourceTreeDirty: manifest.provenance.sourceTreeDirty,
        sourceTreeDigest: manifest.provenance.sourceTreeDigest,
        installId,
        keyId: manifest.keyId,
        trustKeySha256: this.manifestTrustFingerprint(manifest),
        artifactSha256: component.artifact.sha256,
        filesManifestSha256: inspected.filesManifestSha256,
        manifestId: manifest.manifestId,
        activatedAt: new Date().toISOString(),
      },
    };
  }

  private async installedComponentIsComplete(
    component: ComponentUpdateComponent,
    componentRoot: string,
  ): Promise<boolean> {
    return verifyInstalledComponentRoot(
      this.storageRoot,
      {
        artifactSha256: component.artifact.sha256,
        filesManifestSha256: component.artifact.filesManifestSha256!,
      },
      componentRoot,
      component.artifact.bytes,
    );
  }

  private async loadArtifact(
    component: ComponentUpdateComponent,
    artifactPath: string,
  ): Promise<Buffer> {
    try {
      const cached = await readFile(artifactPath);
      assertArtifactBytes(component, cached);
      return cached;
    } catch (error) {
      if (!isMissing(error)) {
        // A truncated or tampered immutable cache entry is quarantined by
        // deletion and fetched again from the signed URL.
        await rm(artifactPath, { force: true });
      }
    }
    const response = await this.fetchImpl(component.artifact.url, {
      headers: { accept: "application/octet-stream" },
      redirect: "error",
      signal: this.requestSignal(60_000),
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(
        `component_update_artifact_http_${response.status}:${component.id}`,
      );
    }
    const contentLength = response.headers.get("content-length");
    const declaredLength =
      contentLength === null ? null : Number(contentLength);
    if (
      declaredLength !== null
      && Number.isFinite(declaredLength)
      && declaredLength !== component.artifact.bytes
    ) {
      throw new Error(`component_update_artifact_size_mismatch:${component.id}`);
    }
    const bytes = await readBoundedResponse(
      response,
      component.artifact.bytes,
      `component_update_artifact_size_mismatch:${component.id}`,
    );
    assertArtifactBytes(component, bytes);
    await mkdir(dirname(artifactPath), { recursive: true });
    await atomicWrite(artifactPath, bytes);
    return bytes;
  }

  private componentFileLimits(
    component: ComponentUpdateComponent,
  ): Partial<ComponentFilesLimits> {
    return {
      ...COMPONENT_FILES_POLICY,
      maxCompressedBytes: Math.min(
        component.artifact.bytes,
        this.maxArtifactBytes,
      ),
    };
  }

  private requestSignal(timeoutMs: number): AbortSignal {
    const timeout = AbortSignal.timeout(timeoutMs);
    return this.options.signal
      ? AbortSignal.any([timeout, this.options.signal])
      : timeout;
  }

  private async collectGarbage(
    state: ComponentInstallState,
  ): Promise<void> {
    const inUseRoots = await readComponentInUseRoots(this.storageRoot);
    const referenced = new Set(
      [...Object.values(state.active), ...Object.values(state.previous)]
        .map((component) =>
          installedComponentRoot(this.storageRoot, component),
        ),
    );
    for (const root of inUseRoots) referenced.add(root);
    const referencedDigests = new Set(
      [...Object.values(state.active), ...Object.values(state.previous)]
        .map(({ artifactSha256 }) =>
          artifactSha256.slice("sha256:".length),
        ),
    );
    for (const root of inUseRoots) {
      const digest = root.split(sep).at(-1)?.split(".")[0];
      if (digest && /^[0-9a-f]{64}$/.test(digest)) referencedDigests.add(digest);
    }
    const artifactsRoot = join(this.storageRoot, "artifacts");
    await ensureSafeUpdateDirectory(this.storageRoot, artifactsRoot);
    for (const entry of await readdir(artifactsRoot, { withFileTypes: true })) {
      if (!/^[0-9a-f]{64}$/.test(entry.name)) continue;
      const path = join(artifactsRoot, entry.name);
      if (entry.isSymbolicLink() || !entry.isFile()) {
        throw new Error("component_update_artifact_cache_path_unsafe");
      }
      if (!referencedDigests.has(entry.name)) await rm(path, { force: true });
    }

    const componentsRoot = join(this.storageRoot, "components");
    await ensureSafeUpdateDirectory(this.storageRoot, componentsRoot);
    for (
      const componentEntry of await readdir(
        componentsRoot,
        { withFileTypes: true },
      )
    ) {
      const componentPath = join(componentsRoot, componentEntry.name);
      if (componentEntry.isSymbolicLink() || !componentEntry.isDirectory()) {
        throw new Error("component_update_component_cache_path_unsafe");
      }
      for (
        const digestEntry of await readdir(
          componentPath,
          { withFileTypes: true },
        )
      ) {
        const path = join(componentPath, digestEntry.name);
        if (referenced.has(path)) continue;
        if (digestEntry.isSymbolicLink()) {
          await rm(path, { force: true });
          continue;
        }
        if (digestEntry.isDirectory()) {
          await rm(path, { recursive: true, force: true });
        } else if (digestEntry.isFile()) {
          await rm(path, { force: true });
        }
      }
    }
  }
}

export interface ComponentInUseLease {
  roots: Readonly<Record<string, string>>;
  release(): Promise<void>;
}

export async function acquireComponentInUseLease(
  storageRoot: string,
  componentIds: readonly string[],
): Promise<ComponentInUseLease> {
  const root = resolve(storageRoot);
  const ids = [...new Set(componentIds)].sort();
  if (ids.length === 0 || ids.length !== componentIds.length) {
    throw new Error("component_in_use_ids_invalid");
  }
  const state = await readComponentInstallState(root);
  const components: Record<string, InstalledComponent> = {};
  const roots: Record<string, string> = {};
  for (const id of ids) {
    const component = state.active[id];
    if (!component) throw new Error(`component_in_use_active_missing:${id}`);
    const componentRoot = installedComponentRoot(root, component);
    if (!(await isSafeDirectory(componentRoot)) || !(await verifyInstalledComponentRoot(root, component, componentRoot))) {
      throw new Error(`component_in_use_active_invalid:${id}`);
    }
    components[id] = component;
    roots[id] = componentRoot;
  }
  const token = randomUUID();
  const directory = join(root, "in-use");
  await ensureSafeUpdateDirectory(root, directory);
  const path = join(directory, `${token}.json`);
  await atomicWrite(path, Buffer.from(`${JSON.stringify(componentInUseLeaseSchema.parse({
    schema: "mycellios-component-in-use/1",
    token,
    pid: process.pid,
    acquiredAt: new Date().toISOString(),
    components,
  }), null, 2)}\n`, "utf8"));
  let released = false;
  return {
    roots,
    release: async () => {
      if (released) return;
      released = true;
      await rm(path, { force: true });
    },
  };
}

export async function readComponentInUseRoots(
  storageRoot: string,
  processAlive: (pid: number) => boolean = componentLeaseProcessAlive,
): Promise<Set<string>> {
  const root = resolve(storageRoot);
  const directory = join(root, "in-use");
  await ensureSafeUpdateDirectory(root, directory);
  const roots = new Set<string>();
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (!entry.name.endsWith(".json")) continue;
    const path = join(directory, entry.name);
    if (entry.isSymbolicLink() || !entry.isFile()) throw new Error("component_in_use_lease_path_unsafe");
    let lease: z.infer<typeof componentInUseLeaseSchema>;
    try { lease = componentInUseLeaseSchema.parse(JSON.parse(await readFile(path, "utf8"))); }
    catch (error) { throw new Error("component_in_use_lease_invalid", { cause: error }); }
    if (`${lease.token}.json` !== entry.name) throw new Error("component_in_use_lease_name_mismatch");
    if (!processAlive(lease.pid)) { await rm(path, { force: true }); continue; }
    for (const component of Object.values(lease.components)) {
      const componentRoot = installedComponentRoot(root, component);
      if (!(await isSafeDirectory(componentRoot)) || !(await verifyInstalledComponentRoot(root, component, componentRoot))) {
        throw new Error(`component_in_use_root_invalid:${component.id}`);
      }
      roots.add(componentRoot);
    }
  }
  return roots;
}

function componentLeaseProcessAlive(pid: number): boolean {
  try { process.kill(pid, 0); return true; }
  catch (error) { return (error as NodeJS.ErrnoException).code === "EPERM"; }
}

export async function readComponentInstallState(
  storageRoot: string,
): Promise<ComponentInstallState> {
  const root = resolve(storageRoot);
  await ensureSafeUpdateDirectory(root, root);
  const statePath = join(root, "state.json");
  let serialized: string;
  try {
    await assertSafeRegularFile(statePath);
    serialized = await readFile(statePath, "utf8");
  } catch (error) {
    if (isMissing(error)) {
      return await recoverLastGoodComponentState(root)
        ?? cloneState(EMPTY_STATE);
    }
    throw error;
  }
  let decoded: unknown;
  try {
    decoded = JSON.parse(serialized) as unknown;
  } catch {
    await quarantineInvalidComponentState(statePath);
    return await recoverLastGoodComponentState(root)
      ?? cloneState(EMPTY_STATE);
  }
  const current = componentInstallStateSchema.safeParse(decoded);
  if (current.success) return current.data;
  const legacy = legacyComponentInstallStateSchema.safeParse(decoded);
  if (legacy.success) {
    try {
      const migrated = migrateLegacyComponentInstallState(legacy.data);
      await writeComponentInstallState(root, migrated);
      return migrated;
    } catch {
      await quarantineInvalidComponentState(statePath);
      return await recoverLastGoodComponentState(root)
        ?? cloneState(EMPTY_STATE);
    }
  }
  await quarantineInvalidComponentState(statePath);
  return await recoverLastGoodComponentState(root)
    ?? cloneState(EMPTY_STATE);
}

export async function resolveActiveComponentRoot(
  storageRoot: string,
  componentId: string,
  expectedChannel?: ComponentUpdateChannel,
  expectedPinnedKeys?: readonly PinnedComponentUpdateKey[],
): Promise<string | null> {
  const state = await readComponentInstallState(storageRoot);
  const active = state.active[componentId];
  if (!active) return null;
  if (expectedChannel && active.channel !== expectedChannel) return null;
  if (
    expectedPinnedKeys
    && !expectedPinnedKeys.some(
      (key) =>
        key.keyId === active.keyId
        && componentTrustFingerprint(key.spki) === active.trustKeySha256,
    )
  ) return null;
  const componentRoot = installedComponentRoot(storageRoot, active);
  await ensureSafeUpdateDirectory(
    resolve(storageRoot),
    dirname(componentRoot),
  );
  if (
    !(await isSafeDirectory(componentRoot))
    || !(await verifyInstalledComponentRoot(
      storageRoot,
      active,
      componentRoot,
    ))
  ) {
    await quarantineInvalidComponentRoot(componentRoot);
    return null;
  }
  return componentRoot;
}

export interface ComponentRollbackResult {
  state: "rolled-back";
  changedComponents: string[];
  activeComponentRoots: Readonly<Record<string, string>>;
}

/**
 * Atomically promotes the last verified side-by-side roots. The callback must
 * canary those roots before the pointer changes; a failure leaves state intact.
 */
export async function rollbackInstalledComponents(options: {
  storageRoot: string;
  componentIds: readonly string[];
  expectedChannel: ComponentUpdateChannel;
  pinnedKeys: readonly PinnedComponentUpdateKey[];
  onPrepare: (result: {
    changedComponents: string[];
    stagedComponentRoots: Readonly<Record<string, string>>;
  }) => Promise<void>;
}): Promise<ComponentRollbackResult> {
  const ids = [...new Set(options.componentIds)];
  if (ids.length === 0 || ids.length !== options.componentIds.length) {
    throw new Error("component_rollback_component_ids_invalid");
  }
  const storageRoot = resolve(options.storageRoot);
  const before = await readComponentInstallState(storageRoot);
  const roots: Record<string, string> = {};
  for (const id of ids) {
    const active = before.active[id];
    const previous = before.previous[id];
    if (!active || !previous) throw new Error(`component_rollback_previous_missing:${id}`);
    if (previous.channel !== options.expectedChannel) {
      throw new Error(`component_rollback_channel_mismatch:${id}`);
    }
    const trusted = options.pinnedKeys.some((key) =>
      key.keyId === previous.keyId
      && componentTrustFingerprint(key.spki) === previous.trustKeySha256
    );
    if (!trusted) throw new Error(`component_rollback_key_not_trusted:${id}`);
    const root = installedComponentRoot(storageRoot, previous);
    if (!(await isSafeDirectory(root)) || !(await verifyInstalledComponentRoot(storageRoot, previous, root))) {
      throw new Error(`component_rollback_previous_invalid:${id}`);
    }
    roots[id] = root;
  }
  await options.onPrepare({ changedComponents: ids, stagedComponentRoots: roots });
  const after = cloneState(before);
  for (const id of ids) {
    const active = before.active[id]!;
    after.active[id] = before.previous[id]!;
    after.previous[id] = active;
  }
  await writeComponentInstallState(storageRoot, after);
  return { state: "rolled-back", changedComponents: ids, activeComponentRoots: roots };
}

/**
 * Boot-health rollback differs from a user rollback: a failed candidate must
 * never become `previous`. If no last-good component exists, remove the active
 * pointer so the bootstrap falls back to its bundled runtime.
 */
export async function rollbackFailedComponentActivation(options: {
  storageRoot: string;
  targetManifestIds: Readonly<Record<string, `sha256:${string}`>>;
  expectedChannel: ComponentUpdateChannel;
  pinnedKeys: readonly PinnedComponentUpdateKey[];
  onPrepare: (result: {
    changedComponents: string[];
    previousComponentRoots: Readonly<Record<string, string | null>>;
  }) => Promise<void>;
}): Promise<{ state: "rolled-back-failed-activation"; changedComponents: string[] }> {
  const ids = Object.keys(options.targetManifestIds).sort();
  if (ids.length === 0) throw new Error("component_failed_activation_targets_missing");
  const storageRoot = resolve(options.storageRoot);
  const before = await readComponentInstallState(storageRoot);
  const roots: Record<string, string | null> = {};
  for (const id of ids) {
    const active = before.active[id];
    if (!active || active.manifestId !== options.targetManifestIds[id]) {
      throw new Error(`component_failed_activation_target_mismatch:${id}`);
    }
    const previous = before.previous[id];
    if (!previous) {
      roots[id] = null;
      continue;
    }
    if (previous.channel !== options.expectedChannel) throw new Error(`component_rollback_channel_mismatch:${id}`);
    const trusted = options.pinnedKeys.some((key) =>
      key.keyId === previous.keyId
      && componentTrustFingerprint(key.spki) === previous.trustKeySha256
    );
    if (!trusted) throw new Error(`component_rollback_key_not_trusted:${id}`);
    const root = installedComponentRoot(storageRoot, previous);
    if (!(await isSafeDirectory(root)) || !(await verifyInstalledComponentRoot(storageRoot, previous, root))) {
      throw new Error(`component_rollback_previous_invalid:${id}`);
    }
    roots[id] = root;
  }
  await options.onPrepare({ changedComponents: ids, previousComponentRoots: roots });
  const after = cloneState(before);
  for (const id of ids) {
    const previous = before.previous[id];
    if (previous) after.active[id] = previous;
    else delete after.active[id];
    delete after.previous[id];
  }
  await writeComponentInstallState(storageRoot, after);
  return { state: "rolled-back-failed-activation", changedComponents: ids };
}

function componentTrustFingerprint(spki: string): `sha256:${string}` {
  return `sha256:${createHash("sha256")
    .update(Buffer.from(spki, "base64url"))
    .digest("hex")}`;
}

function installedComponentRoot(
  storageRoot: string,
  component: InstalledComponent,
): string {
  const root = resolve(storageRoot);
  const digest = component.artifactSha256.slice("sha256:".length);
  const componentRoot = resolve(
    root,
    "components",
    component.id,
    component.installId ? `${digest}.${component.installId}` : digest,
  );
  if (!componentRoot.startsWith(`${root}${sep}`)) {
    throw new Error("component_update_active_path_escaped_root");
  }
  return componentRoot;
}

async function verifyInstalledComponentRoot(
  storageRoot: string,
  seal: Pick<
    InstalledComponent,
    "artifactSha256" | "filesManifestSha256"
  >,
  componentRoot: string,
  expectedArtifactBytes?: number,
): Promise<boolean> {
  try {
    const root = resolve(storageRoot);
    const digest = seal.artifactSha256.slice("sha256:".length);
    const artifactPath = join(root, "artifacts", digest);
    await ensureSafeUpdateDirectory(root, dirname(artifactPath));
    const artifactMetadata = await lstat(artifactPath);
    if (
      !artifactMetadata.isFile()
      || artifactMetadata.isSymbolicLink()
      || artifactMetadata.size < 1
      || artifactMetadata.size > MAX_COMPONENT_ARTIFACT_BYTES
      || (
        expectedArtifactBytes !== undefined
        && artifactMetadata.size !== expectedArtifactBytes
      )
    ) return false;
    const artifact = await readFile(artifactPath);
    if (
      `sha256:${createHash("sha256").update(artifact).digest("hex")}`
      !== seal.artifactSha256
    ) return false;
    const inspected = inspectComponentFilesPackage(
      artifact,
      COMPONENT_FILES_POLICY,
    );
    if (inspected.filesManifestSha256 !== seal.filesManifestSha256) {
      return false;
    }
    const observed = await listComponentFiles(componentRoot);
    if (
      observed.length !== inspected.files.length
      || observed.some((path, index) => path !== inspected.files[index]?.path)
    ) return false;
    for (const expected of inspected.files) {
      const path = resolve(componentRoot, ...expected.path.split("/"));
      const metadata = await lstat(path);
      if (!metadata.isFile() || metadata.isSymbolicLink()) return false;
      const bytes = await readFile(path);
      if (
        bytes.length !== expected.bytes
        || `sha256:${createHash("sha256").update(bytes).digest("hex")}`
          !== expected.sha256
      ) return false;
    }
    return true;
  } catch {
    return false;
  }
}

async function quarantineInvalidComponentRoot(path: string): Promise<void> {
  try {
    await rename(path, `${path}.invalid-${randomUUID()}`);
  } catch (error) {
    if (!isMissing(error)) throw error;
  }
}

export function assertManifestCompatibility(
  manifest: ComponentUpdateManifest,
  current: Pick<
    ComponentUpdateManagerOptions,
    "bootstrapVersion" | "workerProtocol" | "runtimeAbi"
  >,
): void {
  const protocol = manifest.compatibility.workerProtocol;
  if (
    current.workerProtocol.max < protocol.min
    || protocol.max < current.workerProtocol.min
  ) {
    throw new Error("component_update_worker_protocol_incompatible");
  }
  if (manifest.compatibility.runtimeAbi !== current.runtimeAbi) {
    throw new Error("component_update_runtime_abi_incompatible");
  }
  if (
    compareVersions(
      current.bootstrapVersion,
      manifest.compatibility.minBootstrapVersion,
    ) < 0
  ) {
    throw new Error("component_update_bootstrap_too_old");
  }
}

export function assertManagedComponentPolicy(
  component: ComponentUpdateComponent,
  platform: NodeJS.Platform,
  architecture: NodeJS.Architecture,
): void {
  const policy = MANAGED_COMPONENT_POLICIES[
    component.id as keyof typeof MANAGED_COMPONENT_POLICIES
  ];
  if (!policy) throw new Error(`component_update_policy_missing:${component.id}`);
  const target = `${platform}/${architecture}`;
  if (!(policy.targets as readonly string[]).includes(target)) {
    throw new Error(`component_update_target_not_certified:${component.id}:${target}`);
  }
  if (component.requirements.backend !== policy.backend) {
    throw new Error(`component_update_backend_policy_mismatch:${component.id}`);
  }
  if (policy.driverApi === null) {
    if (component.requirements.driver !== null) {
      throw new Error(`component_update_driver_policy_mismatch:${component.id}`);
    }
  } else if (component.requirements.driver?.api !== policy.driverApi) {
    throw new Error(`component_update_driver_policy_mismatch:${component.id}`);
  }
  const dependencyIds = component.requirements.dependencies.map(({ id }) => id);
  if (JSON.stringify(dependencyIds) !== JSON.stringify(policy.dependencies)) {
    throw new Error(`component_update_dependency_policy_mismatch:${component.id}`);
  }
}

export function assertResolvedComponentDependencies(
  selected: readonly ComponentUpdateComponent[],
  state: ComponentInstallState,
  validCurrentComponents: ReadonlySet<string>,
): void {
  const targets = new Map(selected.map((component) => [component.id, component.version]));
  for (const component of selected) {
    for (const dependency of component.requirements.dependencies) {
      const selectedVersion = targets.get(dependency.id);
      const current = state.active[dependency.id];
      const version = selectedVersion
        ?? (current && validCurrentComponents.has(dependency.id) ? current.version : null);
      if (!version) {
        throw new Error(`component_update_dependency_missing:${component.id}:${dependency.id}`);
      }
      if (compareVersions(version, dependency.minVersion) < 0) {
        throw new Error(`component_update_dependency_too_old:${component.id}:${dependency.id}`);
      }
      if (
        dependency.maxVersionExclusive
        && compareVersions(version, dependency.maxVersionExclusive) >= 0
      ) {
        throw new Error(`component_update_dependency_too_new:${component.id}:${dependency.id}`);
      }
    }
  }
}

export function selectApplicableComponents(
  components: readonly ComponentUpdateComponent[],
  platform: NodeJS.Platform,
  architecture: NodeJS.Architecture,
): ComponentUpdateComponent[] {
  const candidates = components.filter(
    (component) =>
      (component.platform === "any" || component.platform === platform)
      && (component.arch === "any" || component.arch === architecture),
  );
  const selected = new Map<string, {
    score: number;
    component: ComponentUpdateComponent;
  }>();
  for (const component of candidates) {
    const score =
      (component.platform === platform ? 2 : 0)
      + (component.arch === architecture ? 1 : 0);
    const existing = selected.get(component.id);
    if (!existing || score > existing.score) {
      selected.set(component.id, { score, component });
      continue;
    }
    if (score === existing.score) {
      throw new Error(
        `component_update_component_target_is_ambiguous:${component.id}`,
      );
    }
  }
  return [...selected.values()]
    .map(({ component }) => component)
    .sort((left, right) => left.id.localeCompare(right.id));
}

function assertArtifactBytes(
  component: ComponentUpdateComponent,
  bytes: Buffer,
): void {
  if (bytes.length !== component.artifact.bytes) {
    throw new Error(`component_update_artifact_size_mismatch:${component.id}`);
  }
  const identity = `sha256:${createHash("sha256").update(bytes).digest("hex")}`;
  if (identity !== component.artifact.sha256) {
    throw new Error(`component_update_artifact_digest_mismatch:${component.id}`);
  }
}

async function writeComponentInstallState(
  storageRoot: string,
  state: ComponentInstallState,
): Promise<void> {
  const parsed = componentInstallStateSchema.parse(state);
  const root = resolve(storageRoot);
  const path = join(root, "state.json");
  await ensureSafeUpdateDirectory(root, dirname(path));
  await atomicWrite(
    path,
    Buffer.from(`${JSON.stringify(parsed, null, 2)}\n`, "utf8"),
  );
  await atomicWrite(
    join(root, "state.last-good.json"),
    Buffer.from(`${JSON.stringify(parsed, null, 2)}\n`, "utf8"),
  );
}

async function atomicWrite(path: string, bytes: Buffer): Promise<void> {
  const temporary = `${path}.tmp-${randomUUID()}`;
  let handle: Awaited<ReturnType<typeof open>> | null = null;
  try {
    handle = await open(temporary, "wx");
    await handle.writeFile(bytes);
    await handle.sync();
    await handle.close();
    handle = null;
    await rename(temporary, path);
    await syncUpdateDirectory(dirname(path));
  } finally {
    await handle?.close();
    await rm(temporary, { force: true });
  }
}

async function isSafeDirectory(path: string): Promise<boolean> {
  try {
    const details = await lstat(path);
    return details.isDirectory() && !details.isSymbolicLink();
  } catch (error) {
    if (isMissing(error)) return false;
    throw error;
  }
}

async function assertSafeRegularFile(path: string): Promise<void> {
  const details = await lstat(path);
  if (!details.isFile() || details.isSymbolicLink()) {
    throw new Error("component_update_state_path_unsafe");
  }
}

function migrateLegacyComponentInstallState(
  legacy: z.infer<typeof legacyComponentInstallStateSchema>,
): ComponentInstallState {
  const configuredChannels = (["dev", "stable"] as const).filter(
    (channel) => legacy.channels[channel] !== null,
  );
  const inferChannel = (
    component: z.infer<typeof legacyInstalledComponentSchema>,
    fallback?: ComponentUpdateChannel,
  ): ComponentUpdateChannel => {
    const exact = configuredChannels.filter(
      (channel) =>
        legacy.channels[channel]?.manifestId === component.manifestId,
    );
    if (exact.length === 1) return exact[0]!;
    if (fallback) return fallback;
    if (configuredChannels.length === 1) return configuredChannels[0]!;
    throw new Error("component_update_legacy_channel_ambiguous");
  };
  const active: ComponentInstallState["active"] = {};
  for (const [id, component] of Object.entries(legacy.active)) {
    active[id] = {
      ...component,
      channel: inferChannel(component),
    };
  }
  const previous: ComponentInstallState["previous"] = {};
  for (const [id, component] of Object.entries(legacy.previous)) {
    previous[id] = {
      ...component,
      channel: inferChannel(component, active[id]?.channel),
    };
  }
  return componentInstallStateSchema.parse({
    schema: COMPONENT_INSTALL_STATE_SCHEMA,
    channels: legacy.channels,
    active,
    previous,
    rejected: { dev: null, stable: null },
  });
}

async function quarantineInvalidComponentState(path: string): Promise<void> {
  const quarantined = `${path}.invalid-${randomUUID()}`;
  try {
    await rename(path, quarantined);
  } catch (error) {
    if (!isMissing(error)) throw error;
  }
}

async function recoverLastGoodComponentState(
  root: string,
): Promise<ComponentInstallState | null> {
  const backupPath = join(root, "state.last-good.json");
  try {
    await assertSafeRegularFile(backupPath);
    const decoded: unknown = JSON.parse(await readFile(backupPath, "utf8"));
    const recovered = componentInstallStateSchema.parse(decoded);
    await atomicWrite(
      join(root, "state.json"),
      Buffer.from(`${JSON.stringify(recovered, null, 2)}\n`, "utf8"),
    );
    return recovered;
  } catch (error) {
    if (isMissing(error) || error instanceof SyntaxError || error instanceof z.ZodError) {
      return null;
    }
    throw error;
  }
}

async function syncUpdateDirectory(path: string): Promise<void> {
  let handle: Awaited<ReturnType<typeof open>> | null = null;
  try {
    handle = await open(path, "r");
    await handle.sync();
  } catch (error) {
    if (
      process.platform === "win32"
      && isFileSystemError(error)
      && ["EINVAL", "ENOTSUP", "EPERM"].includes(error.code ?? "")
    ) return;
    throw error;
  } finally {
    await handle?.close();
  }
}

async function ensureSafeUpdateDirectory(
  rootValue: string,
  targetValue: string,
): Promise<void> {
  const root = resolve(rootValue);
  const target = resolve(targetValue);
  if (target !== root && !target.startsWith(`${root}${sep}`)) {
    throw new Error("component_update_storage_path_escaped");
  }
  await mkdir(root, { recursive: true });
  if (!(await isSafeDirectory(root))) {
    throw new Error("component_update_storage_path_unsafe");
  }
  const portable = relative(root, target);
  if (!portable) return;
  let current = root;
  for (const segment of portable.split(/[\\/]/u)) {
    if (!segment || segment === "." || segment === "..") {
      throw new Error("component_update_storage_path_unsafe");
    }
    current = resolve(current, segment);
    try {
      await mkdir(current);
    } catch (error) {
      if (!isAlreadyExists(error)) throw error;
    }
    if (!(await isSafeDirectory(current))) {
      throw new Error("component_update_storage_path_unsafe");
    }
  }
}

async function listComponentFiles(root: string): Promise<string[]> {
  const output: string[] = [];
  const visit = async (directory: string): Promise<void> => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const absolute = join(directory, entry.name);
      if (entry.isSymbolicLink()) {
        throw new Error("component_update_installed_symlink_rejected");
      }
      if (entry.isDirectory()) {
        await visit(absolute);
      } else if (entry.isFile()) {
        output.push(
          absolute.slice(root.length + 1).replaceAll("\\", "/"),
        );
      } else {
        throw new Error("component_update_installed_special_file_rejected");
      }
    }
  };
  await visit(root);
  return output.sort();
}

function cloneState(state: ComponentInstallState): ComponentInstallState {
  return componentInstallStateSchema.parse(structuredClone(state));
}

function compareVersions(left: string, right: string): number {
  const leftParts = versionCore(left);
  const rightParts = versionCore(right);
  for (let index = 0; index < 3; index += 1) {
    const delta = leftParts[index]! - rightParts[index]!;
    if (delta !== 0) return delta < 0 ? -1 : 1;
  }
  return 0;
}

function versionCore(value: string): [number, number, number] {
  const match = /^(\d+)\.(\d+)\.(\d+)/.exec(value);
  if (!match) throw new Error("component_update_bootstrap_version_invalid");
  return [Number(match[1]), Number(match[2]), Number(match[3])];
}

function assertFeedBaseUrl(raw: string): void {
  let url: URL;
  try {
    url = new URL(ensureTrailingSlash(raw));
  } catch {
    throw new Error("component_update_feed_url_invalid");
  }
  const host = url.hostname.toLowerCase();
  const loopback = host === "localhost" || host === "127.0.0.1"
    || host === "::1" || host === "[::1]";
  if (url.protocol !== "https:" && !(url.protocol === "http:" && loopback)) {
    throw new Error("component_update_feed_requires_https");
  }
  if (url.username || url.password || url.hash || url.search) {
    throw new Error("component_update_feed_url_invalid");
  }
}

function ensureTrailingSlash(value: string): string {
  return value.endsWith("/") ? value : `${value}/`;
}

async function readBoundedResponse(
  response: Response,
  maximumBytes: number,
  exceededCode: string,
): Promise<Buffer> {
  if (!response.body) throw new Error("component_update_response_body_missing");
  const reader = response.body.getReader();
  const chunks: Buffer[] = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maximumBytes) {
        await reader.cancel(exceededCode).catch(() => undefined);
        throw new Error(exceededCode);
      }
      chunks.push(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
  }
  return Buffer.concat(chunks, total);
}

function isMissing(error: unknown): error is NodeJS.ErrnoException {
  return (
    error !== null
    && typeof error === "object"
    && "code" in error
    && (error as NodeJS.ErrnoException).code === "ENOENT"
  );
}

function isAlreadyExists(error: unknown): error is NodeJS.ErrnoException {
  return (
    error !== null
    && typeof error === "object"
    && "code" in error
    && (error as NodeJS.ErrnoException).code === "EEXIST"
  );
}

function isFileSystemError(error: unknown): error is NodeJS.ErrnoException {
  return error instanceof Error && "code" in error;
}

function isUpdateCancellation(error: unknown): boolean {
  if (error instanceof AggregateError) {
    return error.errors.some((candidate) => isUpdateCancellation(candidate));
  }
  if (!(error instanceof Error)) return false;
  if (
    error.name === "AbortError"
    || error.message.includes("component_update_check_superseded")
  ) return true;
  return error.cause !== undefined && isUpdateCancellation(error.cause);
}

function isDeterministicManifestFailure(error: unknown): boolean {
  const codes = collectUpdateErrorCodes(error);
  return codes.some((code) =>
    code.startsWith("component_update_manifest_sequence_equivocation")
    || code.startsWith("component_update_worker_protocol_incompatible")
    || code.startsWith("component_update_runtime_abi_incompatible")
    || code.startsWith("component_update_bootstrap_too_old")
    || code.startsWith("component_update_component_not_managed")
    || code.startsWith("component_update_component_format_not_supported")
    || code.startsWith("component_update_component_target_is_ambiguous")
    || code.startsWith("component_update_restart_scope_invalid")
    || code.startsWith("component_update_max_artifact_bytes_exceeds_policy")
    || code.startsWith("component_update_artifact_size_mismatch")
    || code.startsWith("component_update_artifact_digest_mismatch")
    || code.startsWith("component_update_files_manifest_mismatch")
    || code.startsWith("component_update_pre_activation_canary_failed")
    || code.startsWith("component_update_staged_component_incomplete")
    || code.startsWith("component_files_")
    || code.startsWith("component_update_storage_path_")
    || code.startsWith("component_update_installed_")
  );
}

function updateErrorCode(error: unknown): string {
  return collectUpdateErrorCodes(error)[0] ?? "component_update_failed";
}

function collectUpdateErrorCodes(error: unknown): string[] {
  if (error instanceof AggregateError) {
    return error.errors.flatMap((candidate) =>
      collectUpdateErrorCodes(candidate),
    );
  }
  if (!(error instanceof Error)) return [];
  const own = error.message.trim() || error.name;
  return [
    own,
    ...(error.cause === undefined
      ? []
      : collectUpdateErrorCodes(error.cause)),
  ];
}
