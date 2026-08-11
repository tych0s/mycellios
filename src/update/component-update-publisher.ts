import {
  createHash,
  createPrivateKey,
  createPublicKey,
  generateKeyPairSync,
  type KeyObject,
} from "node:crypto";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import {
  chmod,
  mkdir,
  readFile,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import {
  basename,
  dirname,
  join,
  resolve,
  sep,
} from "node:path";

import {
  buildComponentUpdateManifest,
  signComponentUpdateManifest,
  verifyComponentUpdateManifest,
  type ComponentUpdateChannel,
  type ComponentUpdateManifest,
} from "../contracts/component-update-manifest.js";
import {
  COMPONENT_FILES_POLICY,
  MAX_COMPONENT_ARTIFACT_BYTES,
} from "../contracts/component-update-policy.js";
import {
  WORKER_PROTOCOL_MAX,
  WORKER_PROTOCOL_MIN,
} from "../contracts/worker-admission.js";
import { buildComponentFilesPackage } from "./component-files.js";
import { STABLE_COMPONENT_UPDATE_KEYS } from "./component-trust.js";
import {
  COMPONENT_UPDATE_BOOTSTRAP_MIN_VERSION,
  MYCELLIOS_RUNTIME_ABI,
} from "./runtime-compatibility.js";

export const COMPONENT_UPDATE_PUBLIC_KEY_SCHEMA =
  "mycellios-component-update-public-key/1" as const;
export const PREPARED_COMPONENT_RELEASE_SCHEMA =
  "mycellios-component-dev-release/1" as const;
export const COMPONENT_PUBLICATION_EVIDENCE_SCHEMA =
  "mycellios-component-publication-evidence/1" as const;

export type SupportedComponentPlatform = "win32" | "linux" | "darwin";
export type SupportedComponentArchitecture = "x64" | "arm64";

export interface ComponentUpdateTarget {
  channel: ComponentUpdateChannel;
  platform: SupportedComponentPlatform;
  arch: SupportedComponentArchitecture;
}

export interface ComponentUpdateEnrollment {
  schema: typeof COMPONENT_UPDATE_PUBLIC_KEY_SCHEMA;
  channel: ComponentUpdateChannel;
  keyId: string;
  spki: string;
}

/**
 * The private key is intentionally accepted only as an in-process object.
 * Public results from this module never contain private key bytes.
 */
export interface ComponentSigningIdentity {
  privateKey: KeyObject | string | Buffer;
  enrollment: ComponentUpdateEnrollment;
}

export interface CreateDevelopmentSigningIdentityOptions {
  localRoot: string;
  keyId?: string | undefined;
  outputDirectory?: string | undefined;
  persistToFiles?: boolean | undefined;
  storeIdentity?:
    | ((
      identity: ComponentSigningIdentity,
      signal: AbortSignal | undefined,
    ) => void | Promise<void>)
    | undefined;
  signal?: AbortSignal | undefined;
}

export interface DevelopmentSigningIdentityResult {
  created: true;
  channel: "dev";
  keyId: string;
  privateKeyFile: string | null;
  publicKeyFile: string | null;
  desktopEnrollment: ComponentUpdateEnrollment;
  warning: string;
}

export interface PreparedComponentRelease {
  schema: typeof PREPARED_COMPONENT_RELEASE_SCHEMA;
  target: ComponentUpdateTarget;
  artifactFile: string;
  manifestFile: string;
  enrollmentFile: string;
  manifestId: string;
  sequence: number;
  provenance: {
    baseRevision: string;
    sourceTreeDirty: boolean;
    releaseRevision: string;
    filesManifestSha256: string;
  };
}

export interface PrepareComponentReleaseOptions {
  workspace: string;
  localRoot: string;
  feedUrl: string;
  target: ComponentUpdateTarget;
  identity?: ComponentSigningIdentity | undefined;
  loadIdentity?:
    | ((
      signal: AbortSignal | undefined,
    ) => ComponentSigningIdentity | Promise<ComponentSigningIdentity>)
    | undefined;
  keyDirectory?: string | undefined;
  privateKeyFile?: string | undefined;
  publicKeyFile?: string | undefined;
  keyId?: string | undefined;
  outputDirectory?: string | undefined;
  sequence?: number | undefined;
  revision?: string | undefined;
  nodeExecutable?: string | undefined;
  gitExecutable?: string | undefined;
  processEnvironment: NodeJS.ProcessEnv;
  signal?: AbortSignal | undefined;
}

export interface PreparedComponentReleaseResult {
  releaseDirectory: string;
  manifestId: string;
  sequence: number;
  artifactBytes: number;
  artifactSha256: string;
  filesManifestSha256: string;
  baseRevision: string;
  sourceTreeDirty: boolean;
  releaseRevision: string;
  enrollment: ComponentUpdateEnrollment;
}

export type ComponentPublicationCommitPhase =
  | "none"
  | "artifact-staged"
  | "manifest-outcome-unknown"
  | "manifest-committed"
  | "readback-verified";

export interface ComponentPublicationReconciliationEvidence {
  schema: typeof COMPONENT_PUBLICATION_EVIDENCE_SCHEMA;
  safeToRetry: true;
  coordinator: string;
  target: ComponentUpdateTarget;
  manifestId: string;
  sequence: number;
  artifactSha256: string;
  artifactBytes: number;
  artifactStatus: number | null;
  manifestStatus: number | null;
  publicManifestVerified: boolean;
  artifactReadbackVerified: boolean;
}

export interface PublishComponentReleaseOptions {
  releaseDirectory: string;
  coordinatorUrl: string;
  target: ComponentUpdateTarget;
  adminToken?: string | undefined;
  adminTokenFile?: string | undefined;
  signal?: AbortSignal | undefined;
  timeouts?: {
    artifactUploadMs?: number | undefined;
    manifestCommitMs?: number | undefined;
    manifestReadbackMs?: number | undefined;
    artifactReadbackMs?: number | undefined;
  } | undefined;
}

export interface PublishedComponentReleaseResult {
  coordinator: string;
  manifestId: string;
  sequence: number;
  artifactStatus: number;
  /** Null when the PUT response was lost but public readback proved the commit. */
  manifestStatus: number | null;
  publicVerification: true;
  commitPhase: "readback-verified";
  reconciliation: ComponentPublicationReconciliationEvidence & {
    artifactStatus: number;
    manifestStatus: number | null;
    publicManifestVerified: true;
    artifactReadbackVerified: true;
  };
}

export interface ShipComponentReleaseOptions
  extends PrepareComponentReleaseOptions {
  coordinatorUrl?: string | undefined;
  adminToken?: string | undefined;
  adminTokenFile?: string | undefined;
  publishTimeouts?: PublishComponentReleaseOptions["timeouts"];
}

export interface ShippedComponentReleaseResult {
  prepared: PreparedComponentReleaseResult;
  published: PublishedComponentReleaseResult;
}

export class ComponentPublicationError extends Error {
  override readonly name = "ComponentPublicationError";

  constructor(
    message: string,
    readonly commitPhase: ComponentPublicationCommitPhase,
    readonly reconciliation: ComponentPublicationReconciliationEvidence,
    options?: { cause?: unknown },
  ) {
    super(message, options);
  }
}

export async function createDevelopmentSigningIdentity(
  options: CreateDevelopmentSigningIdentityOptions,
): Promise<DevelopmentSigningIdentityResult> {
  assertNotAborted(options.signal);
  const localRoot = resolveManagedRoot(options.localRoot);
  const keyId = options.keyId ?? "mycellios-dev-local-1";
  assertKeyId(keyId);
  const persistToFiles =
    options.persistToFiles
    ?? (
      options.outputDirectory !== undefined
      || options.storeIdentity === undefined
    );
  const directory = resolve(
    options.outputDirectory ?? join(localRoot, "keys", "dev"),
  );
  if (persistToFiles) assertManagedLocalOutput(directory, localRoot);
  const privateKeyPath = persistToFiles
    ? join(directory, "private.pem")
    : null;
  const publicKeyPath = persistToFiles
    ? join(directory, "public.json")
    : null;
  if (
    (privateKeyPath && existsSync(privateKeyPath))
    || (publicKeyPath && existsSync(publicKeyPath))
  ) {
    throw new Error(
      `Signing material already exists at ${directory}; refusing to overwrite it.`,
    );
  }

  const { privateKey, publicKey } = generateKeyPairSync("ed25519");
  const enrollment: ComponentUpdateEnrollment = {
    schema: COMPONENT_UPDATE_PUBLIC_KEY_SCHEMA,
    channel: "dev",
    keyId,
    spki: publicKey
      .export({ format: "der", type: "spki" })
      .toString("base64url"),
  };
  const identity = validateSigningIdentity({ privateKey, enrollment });
  assertNotAborted(options.signal);
  await options.storeIdentity?.(identity, options.signal);
  assertNotAborted(options.signal);

  if (privateKeyPath && publicKeyPath) {
    const privatePem = privateKey.export({
      format: "pem",
      type: "pkcs8",
    });
    await mkdir(directory, { recursive: true });
    assertNotAborted(options.signal);
    await writeFile(privateKeyPath, privatePem, {
      flag: "wx",
      mode: 0o600,
    });
    await chmod(privateKeyPath, 0o600).catch(() => undefined);
    try {
      assertNotAborted(options.signal);
      await writeFile(
        publicKeyPath,
        `${JSON.stringify(enrollment, null, 2)}\n`,
        { flag: "wx" },
      );
    } catch (error) {
      await rm(privateKeyPath, { force: true }).catch(() => undefined);
      throw error;
    }
  }

  return {
    created: true,
    channel: "dev",
    keyId,
    privateKeyFile: privateKeyPath,
    publicKeyFile: publicKeyPath,
    desktopEnrollment: enrollment,
    warning: "Keep the signing identity only on the signing computer.",
  };
}

export async function prepareComponentRelease(
  options: PrepareComponentReleaseOptions,
): Promise<PreparedComponentReleaseResult> {
  assertNotAborted(options.signal);
  const workspace = resolve(options.workspace);
  const localRoot = resolveManagedRoot(options.localRoot);
  const target = validateTarget(options.target);
  const feedBaseUrl = validateComponentFeedUrl(options.feedUrl);
  const identity = await resolveSigningIdentity(options, localRoot, target);
  assertChannelEnrollmentAllowed(identity.enrollment);
  if (options.keyId && options.keyId !== identity.enrollment.keyId) {
    throw new Error("The requested key ID does not match the enrollment.");
  }

  await preparePythonProductSource({
    workspace,
    nodeExecutable: options.nodeExecutable ?? process.execPath,
    processEnvironment: options.processEnvironment,
    signal: options.signal,
  });
  assertNotAborted(options.signal);
  const built = buildComponentFilesPackage(
    resolve(workspace, "build", "python"),
    { limits: COMPONENT_FILES_POLICY },
  );
  assertNotAborted(options.signal);
  if (built.packageBytes.length > MAX_COMPONENT_ARTIFACT_BYTES) {
    throw new Error("The component artifact exceeds the shared client limit.");
  }
  const sourceState = await gitSourceState({
    workspace,
    gitExecutable: options.gitExecutable ?? "git",
    processEnvironment: options.processEnvironment,
    signal: options.signal,
  });
  if (target.channel === "stable" && sourceState.dirty) {
    throw new Error(
      "Stable component publication requires a clean, committed source tree.",
    );
  }
  const releaseRevision = sourceState.dirty
    ? createHash("sha256")
      .update(
        `${sourceState.baseRevision}\n${built.filesManifestSha256}\n`,
      )
      .digest("hex")
      .slice(0, 40)
    : sourceState.baseRevision;
  if (options.revision && options.revision !== releaseRevision) {
    throw new Error(
      "The requested revision does not match the exact prepared source state.",
    );
  }
  const packageVersion = await workspaceVersion(workspace, options.signal);
  const keyDirectory = resolve(
    options.keyDirectory ?? join(localRoot, "keys", target.channel),
  );
  assertManagedLocalOutput(keyDirectory, localRoot);
  const sequence = options.sequence === undefined
    ? await nextSequence(keyDirectory, options.signal)
    : positiveSafeInteger(options.sequence, "sequence");
  const artifactDigest = built.artifactSha256.slice("sha256:".length);
  const sourceId = `sha256:${createHash("sha256")
    .update(
      `${sourceState.baseRevision}\n${sourceState.dirty ? "dirty" : "clean"}\n`
      + `${releaseRevision}\n${built.filesManifestSha256}\n${MYCELLIOS_RUNTIME_ABI}\n`,
    )
    .digest("hex")}` as const;
  const unsigned = buildComponentUpdateManifest({
    channel: target.channel,
    sequence,
    revision: sourceState.baseRevision,
    provenance: {
      baseRevision: sourceState.baseRevision,
      sourceTreeDirty: sourceState.dirty,
      sourceTreeDigest: built.filesManifestSha256,
    },
    sourceId,
    compatibility: {
      workerProtocol: {
        min: WORKER_PROTOCOL_MIN,
        max: WORKER_PROTOCOL_MAX,
      },
      runtimeAbi: MYCELLIOS_RUNTIME_ABI,
      minBootstrapVersion: COMPONENT_UPDATE_BOOTSTRAP_MIN_VERSION,
    },
    components: [
      {
        id: "python-product",
        version:
          `${packageVersion}-${target.channel}.${sequence}.${releaseRevision.slice(0, 10)}`,
        platform: target.platform,
        arch: target.arch,
        artifact: {
          url: new URL(
            `updates/v1/artifacts/${artifactDigest}`,
            trailingSlash(feedBaseUrl),
          ).toString(),
          sha256: built.artifactSha256,
          bytes: built.packageBytes.length,
          format: built.format,
          filesManifestSha256: built.filesManifestSha256,
        },
        requirements: {
          backend: "any",
          driver: null,
          runtimeAbi: MYCELLIOS_RUNTIME_ABI,
          workerProtocol: { min: WORKER_PROTOCOL_MIN, max: WORKER_PROTOCOL_MAX },
          dependencies: [],
        },
        restartScope: "runtime",
      },
    ],
  });
  const manifest = signComponentUpdateManifest(unsigned, {
    keyId: identity.enrollment.keyId,
    privateKey: identity.privateKey,
  });
  const releaseDirectory = resolve(
    options.outputDirectory
      ?? join(
        localRoot,
        "releases",
        target.channel,
        manifest.manifestId.slice("sha256:".length),
      ),
  );
  assertManagedLocalOutput(releaseDirectory, localRoot);
  await mkdir(releaseDirectory, { recursive: true });
  assertNotAborted(options.signal);
  const artifactFile = "artifact.blob";
  const manifestFile = "manifest.json";
  const enrollmentFile = "enrollment.json";
  await writeSameOrCreate(
    join(releaseDirectory, artifactFile),
    built.packageBytes,
  );
  await writeSameOrCreate(
    join(releaseDirectory, manifestFile),
    Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`, "utf8"),
  );
  await writeSameOrCreate(
    join(releaseDirectory, enrollmentFile),
    Buffer.from(`${JSON.stringify(identity.enrollment, null, 2)}\n`, "utf8"),
  );
  const release: PreparedComponentRelease = {
    schema: PREPARED_COMPONENT_RELEASE_SCHEMA,
    target,
    artifactFile,
    manifestFile,
    enrollmentFile,
    manifestId: manifest.manifestId,
    sequence,
    provenance: {
      baseRevision: sourceState.baseRevision,
      sourceTreeDirty: sourceState.dirty,
      releaseRevision,
      filesManifestSha256: built.filesManifestSha256,
    },
  };
  await writeSameOrCreate(
    join(releaseDirectory, "release.json"),
    Buffer.from(`${JSON.stringify(release, null, 2)}\n`, "utf8"),
  );
  assertNotAborted(options.signal);
  return {
    releaseDirectory,
    manifestId: manifest.manifestId,
    sequence,
    artifactBytes: built.packageBytes.length,
    artifactSha256: built.artifactSha256,
    filesManifestSha256: built.filesManifestSha256,
    baseRevision: sourceState.baseRevision,
    sourceTreeDirty: sourceState.dirty,
    releaseRevision,
    enrollment: identity.enrollment,
  };
}

export async function publishComponentRelease(
  options: PublishComponentReleaseOptions,
): Promise<PublishedComponentReleaseResult> {
  assertNotAborted(options.signal);
  const releaseDirectory = resolve(options.releaseDirectory);
  const expectedTarget = validateTarget(options.target);
  const release = await readPreparedComponentRelease(
    releaseDirectory,
    options.signal,
  );
  if (!sameTarget(release.target, expectedTarget)) {
    throw new Error("Prepared release target does not match the requested target.");
  }
  const manifest = JSON.parse(
    await readFile(
      resolveReleaseFile(releaseDirectory, release.manifestFile),
      "utf8",
    ),
  ) as ComponentUpdateManifest;
  const enrollment = await readComponentUpdateEnrollment(
    resolveReleaseFile(releaseDirectory, release.enrollmentFile),
    release.target.channel,
    options.signal,
  );
  assertChannelEnrollmentAllowed(enrollment);
  verifyComponentUpdateManifest(manifest, {
    pinnedKey: {
      keyId: enrollment.keyId,
      spki: enrollment.spki,
    },
    expectedChannel: release.target.channel,
  });
  if (
    manifest.manifestId !== release.manifestId
    || manifest.sequence !== release.sequence
  ) {
    throw new Error("Prepared release metadata does not match its manifest.");
  }
  const artifact = await readFile(
    resolveReleaseFile(releaseDirectory, release.artifactFile),
  );
  assertNotAborted(options.signal);
  const component = manifest.components[0];
  const artifactDigest = component?.artifact.sha256.slice("sha256:".length);
  if (!component || !artifactDigest || !/^[0-9a-f]{64}$/.test(artifactDigest)) {
    throw new Error("Prepared release artifact identity is invalid.");
  }
  if (
    artifact.length !== component.artifact.bytes
    || `sha256:${createHash("sha256").update(artifact).digest("hex")}`
      !== component.artifact.sha256
  ) {
    throw new Error("Prepared release artifact failed digest verification.");
  }
  const coordinator = validateComponentFeedUrl(options.coordinatorUrl);
  const authorization = await resolveAdminAuthorization({
    coordinator,
    adminToken: options.adminToken,
    adminTokenFile: options.adminTokenFile,
    signal: options.signal,
  });
  const timeouts = normalizePublicationTimeouts(options.timeouts);
  let commitPhase: ComponentPublicationCommitPhase = "none";
  const evidence: ComponentPublicationReconciliationEvidence = {
    schema: COMPONENT_PUBLICATION_EVIDENCE_SCHEMA,
    safeToRetry: true,
    coordinator,
    target: release.target,
    manifestId: manifest.manifestId,
    sequence: manifest.sequence,
    artifactSha256: component.artifact.sha256,
    artifactBytes: component.artifact.bytes,
    artifactStatus: null,
    manifestStatus: null,
    publicManifestVerified: false,
    artifactReadbackVerified: false,
  };
  let manifestRequestStarted = false;
  let manifestResponseReceived = false;

  try {
    const artifactResponse = await fetch(
      new URL(
        `public/v1/admin/component-updates/artifacts/${artifactDigest}`,
        trailingSlash(coordinator),
      ),
      {
        method: "PUT",
        headers: {
          "content-type": "application/octet-stream",
          ...(authorization ? { authorization } : {}),
        },
        body: new Uint8Array(artifact),
        redirect: "error",
        signal: requestSignal(options.signal, timeouts.artifactUploadMs),
      },
    );
    evidence.artifactStatus = artifactResponse.status;
    await assertResponse(
      artifactResponse,
      "Component artifact publication failed",
    );
    commitPhase = "artifact-staged";

    manifestRequestStarted = true;
    const manifestResponse = await fetch(
      new URL(
        `public/v1/admin/component-updates/${release.target.channel}/`
        + `${release.target.platform}/${release.target.arch}/manifest`,
        trailingSlash(coordinator),
      ),
      {
        method: "PUT",
        headers: {
          "content-type": "application/json",
          ...(authorization ? { authorization } : {}),
        },
        body: JSON.stringify(manifest),
        redirect: "error",
        signal: requestSignal(options.signal, timeouts.manifestCommitMs),
      },
    );
    manifestResponseReceived = true;
    evidence.manifestStatus = manifestResponse.status;
    await assertResponse(
      manifestResponse,
      "Component manifest publication failed",
    );
    commitPhase = "manifest-committed";

    const publicResponse = await fetch(
      new URL(
        `updates/v1/${release.target.channel}/${release.target.platform}/`
        + `${release.target.arch}/manifest.json`,
        trailingSlash(coordinator),
      ),
      {
        headers: { accept: "application/json" },
        cache: "no-store",
        redirect: "error",
        signal: requestSignal(options.signal, timeouts.manifestReadbackMs),
      },
    );
    await assertResponse(
      publicResponse,
      "Published manifest could not be read back",
    );
    const observed = verifyComponentUpdateManifest(
      await publicResponse.json(),
      {
        pinnedKey: {
          keyId: enrollment.keyId,
          spki: enrollment.spki,
        },
        expectedChannel: release.target.channel,
        minimumSequence: manifest.sequence,
      },
    );
    if (
      observed.manifestId !== manifest.manifestId
      || observed.sequence !== manifest.sequence
    ) {
      throw new Error("Public component manifest readback did not match.");
    }
    evidence.publicManifestVerified = true;

    const artifactReadback = await fetch(component.artifact.url, {
      headers: { accept: "application/octet-stream" },
      cache: "no-store",
      redirect: "error",
      signal: requestSignal(options.signal, timeouts.artifactReadbackMs),
    });
    await assertResponse(
      artifactReadback,
      "Published artifact URL could not be read back",
    );
    const observedArtifact = await readBoundedResponse(
      artifactReadback,
      component.artifact.bytes,
      options.signal,
    );
    const observedArtifactIdentity = `sha256:${createHash("sha256")
      .update(observedArtifact)
      .digest("hex")}`;
    if (
      observedArtifact.length !== component.artifact.bytes
      || observedArtifactIdentity !== component.artifact.sha256
    ) {
      throw new Error("Published artifact readback failed digest verification.");
    }
    evidence.artifactReadbackVerified = true;
    commitPhase = "readback-verified";
    return {
      coordinator,
      manifestId: manifest.manifestId,
      sequence: manifest.sequence,
      artifactStatus: evidence.artifactStatus!,
      manifestStatus: evidence.manifestStatus!,
      publicVerification: true,
      commitPhase,
      reconciliation: evidence as PublishedComponentReleaseResult["reconciliation"],
    };
  } catch (error) {
    if (error instanceof ComponentPublicationError) throw error;
    if (
      commitPhase === "artifact-staged"
      && manifestRequestStarted
      && !manifestResponseReceived
    ) {
      const reconciled = await reconcileAmbiguousManifestPublication({
        coordinator,
        release,
        manifest,
        enrollment,
        component,
        evidence,
        timeouts,
      });
      if (reconciled === "verified") {
        commitPhase = "readback-verified";
        return {
          coordinator,
          manifestId: manifest.manifestId,
          sequence: manifest.sequence,
          artifactStatus: evidence.artifactStatus!,
          manifestStatus: evidence.manifestStatus,
          publicVerification: true,
          commitPhase,
          reconciliation:
            evidence as PublishedComponentReleaseResult["reconciliation"],
        };
      }
      if (reconciled === "committed-readback-incomplete") {
        commitPhase = "manifest-committed";
      } else if (reconciled === "unknown") {
        commitPhase = "manifest-outcome-unknown";
      }
    }
    throw new ComponentPublicationError(
      error instanceof Error ? error.message : "Component publication failed.",
      commitPhase,
      { ...evidence, target: { ...evidence.target } },
      { cause: error },
    );
  }
}

async function reconcileAmbiguousManifestPublication(input: {
  coordinator: string;
  release: PreparedComponentRelease;
  manifest: ComponentUpdateManifest;
  enrollment: ComponentUpdateEnrollment;
  component: ComponentUpdateManifest["components"][number];
  evidence: ComponentPublicationReconciliationEvidence;
  timeouts: ReturnType<typeof normalizePublicationTimeouts>;
}): Promise<
  | "verified"
  | "not-committed"
  | "committed-readback-incomplete"
  | "unknown"
> {
  try {
    const publicResponse = await fetch(
      new URL(
        `updates/v1/${input.release.target.channel}/`
        + `${input.release.target.platform}/${input.release.target.arch}/`
        + "manifest.json",
        trailingSlash(input.coordinator),
      ),
      {
        headers: { accept: "application/json" },
        cache: "no-store",
        redirect: "error",
        // Deliberately independent from the cancelled publication signal. This
        // is a bounded, read-only check of whether the server committed.
        signal: requestSignal(undefined, input.timeouts.manifestReadbackMs),
      },
    );
    if (publicResponse.status === 404) return "not-committed";
    if (!publicResponse.ok) return "unknown";
    const observed = verifyComponentUpdateManifest(
      await publicResponse.json(),
      {
        pinnedKey: {
          keyId: input.enrollment.keyId,
          spki: input.enrollment.spki,
        },
        expectedChannel: input.release.target.channel,
      },
    );
    if (
      observed.manifestId !== input.manifest.manifestId
      || observed.sequence !== input.manifest.sequence
    ) {
      return observed.sequence < input.manifest.sequence
        ? "not-committed"
        : "unknown";
    }
    input.evidence.publicManifestVerified = true;
    try {
      const artifactReadback = await fetch(input.component.artifact.url, {
        headers: { accept: "application/octet-stream" },
        cache: "no-store",
        redirect: "error",
        signal: requestSignal(undefined, input.timeouts.artifactReadbackMs),
      });
      if (!artifactReadback.ok) return "committed-readback-incomplete";
      const observedArtifact = await readBoundedResponse(
        artifactReadback,
        input.component.artifact.bytes,
      );
      const identity = `sha256:${createHash("sha256")
        .update(observedArtifact)
        .digest("hex")}`;
      if (
        observedArtifact.length !== input.component.artifact.bytes
        || identity !== input.component.artifact.sha256
      ) {
        return "committed-readback-incomplete";
      }
      input.evidence.artifactReadbackVerified = true;
      return "verified";
    } catch {
      return "committed-readback-incomplete";
    }
  } catch {
    return "unknown";
  }
}

export async function shipComponentRelease(
  options: ShipComponentReleaseOptions,
): Promise<ShippedComponentReleaseResult> {
  const prepared = await prepareComponentRelease(options);
  assertNotAborted(options.signal);
  const published = await publishComponentRelease({
    releaseDirectory: prepared.releaseDirectory,
    coordinatorUrl: options.coordinatorUrl ?? options.feedUrl,
    target: options.target,
    adminToken: options.adminToken,
    adminTokenFile: options.adminTokenFile,
    signal: options.signal,
    timeouts: options.publishTimeouts,
  });
  return { prepared, published };
}

export async function readPreparedComponentRelease(
  releaseDirectory: string,
  signal?: AbortSignal,
): Promise<PreparedComponentRelease> {
  assertNotAborted(signal);
  const value = JSON.parse(
    await readFile(resolve(releaseDirectory, "release.json"), "utf8"),
  ) as unknown;
  assertNotAborted(signal);
  return parsePreparedComponentRelease(value);
}

export async function readComponentUpdateEnrollment(
  path: string,
  expectedChannel: ComponentUpdateChannel,
  signal?: AbortSignal,
): Promise<ComponentUpdateEnrollment> {
  assertNotAborted(signal);
  const value = JSON.parse(await readFile(path, "utf8")) as unknown;
  assertNotAborted(signal);
  return validateComponentUpdateEnrollment(
    value,
    expectedChannel,
    path,
  );
}

export function validateComponentUpdateEnrollment(
  value: unknown,
  expectedChannel: ComponentUpdateChannel,
  source = "component enrollment",
): ComponentUpdateEnrollment {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Invalid component public key document: ${source}`);
  }
  const candidate = value as Record<string, unknown>;
  if (
    candidate.schema !== COMPONENT_UPDATE_PUBLIC_KEY_SCHEMA
    || candidate.channel !== expectedChannel
    || typeof candidate.keyId !== "string"
    || typeof candidate.spki !== "string"
  ) {
    throw new Error(`Invalid component public key document: ${source}`);
  }
  assertKeyId(candidate.keyId);
  if (!/^[A-Za-z0-9_-]+$/.test(candidate.spki)) {
    throw new Error(`Invalid component public key SPKI: ${source}`);
  }
  const encoded = Buffer.from(candidate.spki, "base64url");
  if (
    encoded.length > 128
    || encoded.toString("base64url") !== candidate.spki
  ) {
    throw new Error(`Invalid component public key SPKI: ${source}`);
  }
  let publicKey: KeyObject;
  try {
    publicKey = createPublicKey({
      key: encoded,
      format: "der",
      type: "spki",
    });
  } catch (error) {
    throw new Error(`Invalid component public key SPKI: ${source}`, {
      cause: error,
    });
  }
  const canonical = publicKey.export({ format: "der", type: "spki" });
  if (
    publicKey.asymmetricKeyType !== "ed25519"
    || !Buffer.isBuffer(canonical)
    || !canonical.equals(encoded)
  ) {
    throw new Error(`Invalid component public key SPKI: ${source}`);
  }
  return {
    schema: COMPONENT_UPDATE_PUBLIC_KEY_SCHEMA,
    channel: expectedChannel,
    keyId: candidate.keyId,
    spki: candidate.spki,
  };
}

function validateSigningIdentity(
  identity: ComponentSigningIdentity,
): ComponentSigningIdentity {
  const enrollment = validateComponentUpdateEnrollment(
    identity.enrollment,
    identity.enrollment.channel,
  );
  const privateKey =
    typeof identity.privateKey === "string"
    || Buffer.isBuffer(identity.privateKey)
      ? createPrivateKey(identity.privateKey)
      : identity.privateKey;
  if (privateKey.type !== "private") {
    throw new Error("The component signing identity needs a private key.");
  }
  if (privateKey.asymmetricKeyType !== "ed25519") {
    throw new Error("The component signing key must be Ed25519.");
  }
  const derivedSpki = createPublicKey(privateKey)
    .export({ format: "der", type: "spki" })
    .toString("base64url");
  if (derivedSpki !== enrollment.spki) {
    throw new Error("The signing identity does not match its enrollment.");
  }
  return { privateKey, enrollment };
}

async function resolveSigningIdentity(
  options: PrepareComponentReleaseOptions,
  localRoot: string,
  target: ComponentUpdateTarget,
): Promise<ComponentSigningIdentity> {
  const sourceCount = Number(options.identity !== undefined)
    + Number(options.loadIdentity !== undefined);
  if (sourceCount > 1) {
    throw new Error("Provide only one in-memory component signing identity.");
  }
  if (options.identity) {
    return identityForTarget(
      validateSigningIdentity(options.identity),
      target,
    );
  }
  if (options.loadIdentity) {
    assertNotAborted(options.signal);
    const identity = await options.loadIdentity(options.signal);
    assertNotAborted(options.signal);
    return identityForTarget(validateSigningIdentity(identity), target);
  }

  const keyDirectory = resolve(
    options.keyDirectory ?? join(localRoot, "keys", target.channel),
  );
  assertManagedLocalOutput(keyDirectory, localRoot);
  const privateKeyPath = resolve(
    options.privateKeyFile ?? join(keyDirectory, "private.pem"),
  );
  const publicKeyPath = resolve(
    options.publicKeyFile ?? join(keyDirectory, "public.json"),
  );
  const enrollment = await readComponentUpdateEnrollment(
    publicKeyPath,
    target.channel,
    options.signal,
  );
  assertNotAborted(options.signal);
  const privateKey = createPrivateKey(await readFile(privateKeyPath));
  return identityForTarget(
    validateSigningIdentity({ privateKey, enrollment }),
    target,
  );
}

function identityForTarget(
  identity: ComponentSigningIdentity,
  target: ComponentUpdateTarget,
): ComponentSigningIdentity {
  if (identity.enrollment.channel !== target.channel) {
    throw new Error(
      "The signing identity channel does not match the release target.",
    );
  }
  return identity;
}

function assertChannelEnrollmentAllowed(
  enrollment: ComponentUpdateEnrollment,
): void {
  if (enrollment.channel === "dev") return;
  if (
    STABLE_COMPONENT_UPDATE_KEYS.some(
      (key) =>
        key.keyId === enrollment.keyId
        && key.spki === enrollment.spki,
    )
  ) return;
  throw new Error(
    "Stable component publication is disabled until its public key is embedded in a signed desktop bootstrap.",
  );
}

function parsePreparedComponentRelease(
  value: unknown,
): PreparedComponentRelease {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Prepared component release is invalid.");
  }
  const candidate = value as Partial<PreparedComponentRelease>;
  if (
    candidate.schema !== PREPARED_COMPONENT_RELEASE_SCHEMA
    || !candidate.target
    || !["dev", "stable"].includes(candidate.target.channel)
    || !["win32", "linux", "darwin"].includes(candidate.target.platform)
    || !["x64", "arm64"].includes(candidate.target.arch)
    || typeof candidate.artifactFile !== "string"
    || typeof candidate.manifestFile !== "string"
    || typeof candidate.enrollmentFile !== "string"
    || typeof candidate.manifestId !== "string"
    || !/^sha256:[0-9a-f]{64}$/.test(candidate.manifestId)
    || !Number.isSafeInteger(candidate.sequence)
    || (candidate.sequence ?? 0) < 1
    || !candidate.provenance
    || !/^[0-9a-f]{40}$/.test(candidate.provenance.baseRevision)
    || typeof candidate.provenance.sourceTreeDirty !== "boolean"
    || !/^[0-9a-f]{40}$/.test(candidate.provenance.releaseRevision)
    || !/^sha256:[0-9a-f]{64}$/.test(
      candidate.provenance.filesManifestSha256,
    )
  ) {
    throw new Error("Prepared component release is invalid.");
  }
  return {
    ...candidate,
    target: validateTarget(candidate.target),
  } as PreparedComponentRelease;
}

function validateTarget(target: ComponentUpdateTarget): ComponentUpdateTarget {
  if (
    !target
    || !["dev", "stable"].includes(target.channel)
    || !["win32", "linux", "darwin"].includes(target.platform)
    || !["x64", "arm64"].includes(target.arch)
  ) {
    throw new Error("Component update target is invalid.");
  }
  return { ...target };
}

function sameTarget(
  left: ComponentUpdateTarget,
  right: ComponentUpdateTarget,
): boolean {
  return (
    left.channel === right.channel
    && left.platform === right.platform
    && left.arch === right.arch
  );
}

async function preparePythonProductSource(options: {
  workspace: string;
  nodeExecutable: string;
  processEnvironment: NodeJS.ProcessEnv;
  signal?: AbortSignal | undefined;
}): Promise<void> {
  await runFixedProcess(
    options.nodeExecutable,
    [resolve(
      options.workspace,
      "scripts",
      "prepare-packaged-python-source.mjs",
    )],
    {
      cwd: options.workspace,
      signal: options.signal,
      timeoutMs: 120_000,
      label: "Python product preparation",
      environment: options.processEnvironment,
    },
  );
}

async function gitSourceState(options: {
  workspace: string;
  gitExecutable: string;
  processEnvironment: NodeJS.ProcessEnv;
  signal?: AbortSignal | undefined;
}): Promise<{ baseRevision: string; dirty: boolean }> {
  const revisionResult = await runFixedProcess(
    options.gitExecutable,
    ["rev-parse", "HEAD"],
    {
      cwd: options.workspace,
      signal: options.signal,
      timeoutMs: 30_000,
      label: "Git revision inspection",
      environment: options.processEnvironment,
    },
  );
  const baseRevision = revisionResult.stdout.trim().toLowerCase();
  if (!/^[0-9a-f]{40}$/.test(baseRevision)) {
    throw new Error("Could not resolve the current Git revision.");
  }
  const statusResult = await runFixedProcess(
    options.gitExecutable,
    ["status", "--porcelain=v1", "--untracked-files=all"],
    {
      cwd: options.workspace,
      signal: options.signal,
      timeoutMs: 30_000,
      label: "Git source inspection",
      environment: options.processEnvironment,
    },
  );
  return {
    baseRevision,
    dirty: statusResult.stdout.trim().length > 0,
  };
}

async function runFixedProcess(
  command: string,
  args: readonly string[],
  options: {
    cwd: string;
    signal?: AbortSignal | undefined;
    timeoutMs: number;
    label: string;
    environment: NodeJS.ProcessEnv;
  },
): Promise<{ stdout: string; stderr: string }> {
  assertNotAborted(options.signal);
  const signal = requestSignal(options.signal, options.timeoutMs);
  const child = spawn(command, [...args], {
    cwd: options.cwd,
    windowsHide: true,
    shell: false,
    signal,
    stdio: ["ignore", "pipe", "pipe"],
    env: options.environment,
  });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk: Buffer) => {
    stdout = appendBounded(stdout, chunk.toString("utf8"));
  });
  child.stderr.on("data", (chunk: Buffer) => {
    stderr = appendBounded(stderr, chunk.toString("utf8"));
  });
  await new Promise<void>((resolvePromise, rejectPromise) => {
    let settled = false;
    const settle = (callback: () => void) => {
      if (settled) return;
      settled = true;
      callback();
    };
    child.once("error", (error) => {
      settle(() => rejectPromise(error));
    });
    child.once("close", (code) => {
      settle(() => {
        if (code === 0) {
          resolvePromise();
          return;
        }
        rejectPromise(new Error(
          stderr.trim()
          || stdout.trim()
          || `${options.label} exited with ${code ?? "no status"}.`,
        ));
      });
    });
  });
  assertNotAborted(options.signal);
  return { stdout, stderr };
}

async function nextSequence(
  keyDirectory: string,
  signal?: AbortSignal,
): Promise<number> {
  assertNotAborted(signal);
  const path = join(keyDirectory, "sequence.txt");
  let previous = 0;
  try {
    previous = positiveSafeInteger(
      (await readFile(path, "utf8")).trim(),
      "stored sequence",
    );
  } catch (error) {
    if (!isMissing(error)) throw error;
  }
  assertNotAborted(signal);
  const next = Math.max(Date.now(), previous + 1);
  await atomicWrite(path, Buffer.from(`${next}\n`, "utf8"));
  assertNotAborted(signal);
  return next;
}

async function workspaceVersion(
  workspace: string,
  signal?: AbortSignal,
): Promise<string> {
  assertNotAborted(signal);
  const candidate = JSON.parse(
    await readFile(resolve(workspace, "package.json"), "utf8"),
  ) as { version?: unknown };
  if (
    typeof candidate.version !== "string"
    || !/^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)/.test(
      candidate.version,
    )
  ) {
    throw new Error("package.json does not contain a valid version.");
  }
  return candidate.version;
}

async function resolveAdminAuthorization(options: {
  coordinator: string;
  adminToken?: string | undefined;
  adminTokenFile?: string | undefined;
  signal?: AbortSignal | undefined;
}): Promise<string | null> {
  if (options.adminToken && options.adminTokenFile) {
    throw new Error("Provide an admin token or token file, not both.");
  }
  assertNotAborted(options.signal);
  const token = options.adminTokenFile
    ? (await readFile(resolve(options.adminTokenFile), "utf8")).trim()
    : options.adminToken?.trim();
  if (token) return `Bearer ${token}`;
  const host = new URL(options.coordinator).hostname.toLowerCase();
  if (["localhost", "127.0.0.1", "::1"].includes(host)) return null;
  throw new Error(
    "Remote component publication requires an explicit admin token or token file.",
  );
}

function normalizePublicationTimeouts(
  timeouts: PublishComponentReleaseOptions["timeouts"],
): {
  artifactUploadMs: number;
  manifestCommitMs: number;
  manifestReadbackMs: number;
  artifactReadbackMs: number;
} {
  return {
    artifactUploadMs: timeoutValue(
      timeouts?.artifactUploadMs,
      90_000,
      "artifact upload timeout",
    ),
    manifestCommitMs: timeoutValue(
      timeouts?.manifestCommitMs,
      30_000,
      "manifest commit timeout",
    ),
    manifestReadbackMs: timeoutValue(
      timeouts?.manifestReadbackMs,
      15_000,
      "manifest readback timeout",
    ),
    artifactReadbackMs: timeoutValue(
      timeouts?.artifactReadbackMs,
      90_000,
      "artifact readback timeout",
    ),
  };
}

function timeoutValue(
  value: number | undefined,
  fallback: number,
  label: string,
): number {
  if (value === undefined) return fallback;
  return positiveSafeInteger(value, label);
}

function requestSignal(
  signal: AbortSignal | undefined,
  timeoutMs: number,
): AbortSignal {
  const timeout = AbortSignal.timeout(timeoutMs);
  return signal ? AbortSignal.any([signal, timeout]) : timeout;
}

function resolveReleaseFile(root: string, portable: string): string {
  if (
    !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(portable)
    || basename(portable) !== portable
  ) {
    throw new Error("Prepared release filename is unsafe.");
  }
  const path = resolve(root, portable);
  if (dirname(path) !== resolve(root)) {
    throw new Error("Prepared release path escaped its directory.");
  }
  return path;
}

async function assertResponse(
  response: Response,
  message: string,
): Promise<void> {
  if (response.ok) return;
  const body = (await response.text()).slice(0, 2_000);
  throw new Error(`${message}: HTTP ${response.status}${body ? ` ${body}` : ""}`);
}

async function readBoundedResponse(
  response: Response,
  maximumBytes: number,
  signal?: AbortSignal,
): Promise<Buffer> {
  const declared = Number(response.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > maximumBytes) {
    throw new Error("Published artifact readback exceeded its signed size.");
  }
  if (!response.body) return Buffer.alloc(0);
  const reader = response.body.getReader();
  const chunks: Buffer[] = [];
  let total = 0;
  try {
    while (true) {
      assertNotAborted(signal);
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maximumBytes) {
        await reader.cancel().catch(() => undefined);
        throw new Error("Published artifact readback exceeded its signed size.");
      }
      chunks.push(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
  }
  return Buffer.concat(chunks, total);
}

async function writeSameOrCreate(path: string, bytes: Buffer): Promise<void> {
  try {
    const existing = await readFile(path);
    if (!existing.equals(bytes)) {
      throw new Error(`Refusing to overwrite different release data: ${path}`);
    }
    return;
  } catch (error) {
    if (!isMissing(error)) throw error;
  }
  await writeFile(path, bytes, { flag: "wx" });
}

async function atomicWrite(path: string, bytes: Buffer): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${process.pid}-${Date.now()}`;
  try {
    await writeFile(temporary, bytes, { flag: "wx" });
    await rename(temporary, path);
  } finally {
    await rm(temporary, { force: true });
  }
}

function validateComponentFeedUrl(raw: string): string {
  const url = new URL(trailingSlash(raw));
  const host = url.hostname.toLowerCase();
  const loopback = ["localhost", "127.0.0.1", "::1"].includes(host);
  if (
    (url.protocol !== "https:" && !(url.protocol === "http:" && loopback))
    || url.username
    || url.password
    || url.hash
  ) {
    throw new Error(
      "Component feeds require HTTPS, except for an explicit loopback development coordinator.",
    );
  }
  return url.toString().replace(/\/$/, "");
}

function resolveManagedRoot(path: string): string {
  const root = resolve(path);
  if (root === dirname(root)) {
    throw new Error("Component update local root is too broad.");
  }
  return root;
}

function assertManagedLocalOutput(path: string, localRoot: string): void {
  const local = resolve(localRoot);
  const target = resolve(path);
  if (target !== local && !target.startsWith(`${local}${sep}`)) {
    throw new Error(`Component update output must stay under ${local}.`);
  }
}

function assertKeyId(value: string): void {
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value)) {
    throw new Error("Component update key ID is invalid.");
  }
}

function positiveSafeInteger(value: number, label: string): number;
function positiveSafeInteger(value: string, label: string): number;
function positiveSafeInteger(value: number | string, label: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive safe integer.`);
  }
  return parsed;
}

function trailingSlash(value: string): string {
  return value.endsWith("/") ? value : `${value}/`;
}

function isMissing(error: unknown): error is NodeJS.ErrnoException {
  return (
    error !== null
    && typeof error === "object"
    && "code" in error
    && (error as NodeJS.ErrnoException).code === "ENOENT"
  );
}

function assertNotAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return;
  if (signal.reason instanceof Error) throw signal.reason;
  throw new DOMException("The component publication was aborted.", "AbortError");
}

function appendBounded(current: string, addition: string): string {
  const combined = current + addition;
  return combined.length <= 64 * 1_024
    ? combined
    : combined.slice(-(64 * 1_024));
}
