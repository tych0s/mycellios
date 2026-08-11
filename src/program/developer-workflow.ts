import { createHash, createPublicKey } from "node:crypto";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { join, resolve } from "node:path";

import type {
  DesktopDeveloperWorkflowStatus,
} from "../contracts/control-api.js";
import {
  createDevelopmentSigningIdentity,
  ComponentPublicationError,
  prepareComponentRelease,
  publishComponentRelease,
  type ComponentSigningIdentity,
  type ComponentUpdateEnrollment,
  type PreparedComponentReleaseResult,
} from "../update/component-update-publisher.js";
import {
  createCoordinator,
  type CoordinatorRuntime,
} from "../coordinator/server.js";
import {
  executeDistributedDevelopmentGate,
  type DevelopmentGateEvent,
} from "./distributed-development-gate.js";

const DEV_UPDATE_PORT = 8_791;
const DEV_UPDATE_FEED_URL = `http://127.0.0.1:${DEV_UPDATE_PORT}`;
const MAX_VISIBLE_LOG_LINES = 24;
const MAX_VISIBLE_LOG_LINE_LENGTH = 500;

type PublicKeyDocument = ComponentUpdateEnrollment & { channel: "dev" };

export interface DeveloperWorkflowOptions {
  workspace: string;
  sourceMode: boolean;
  runtimeExecutable?: string | undefined;
  environment?: NodeJS.ProcessEnv | undefined;
  loadSigningIdentity?:
    | (() => ComponentSigningIdentity | null | Promise<ComponentSigningIdentity | null>)
    | undefined;
  storeSigningIdentity?:
    | ((identity: ComponentSigningIdentity) => void | Promise<void>)
    | undefined;
  onStatus?: ((status: DesktopDeveloperWorkflowStatus) => void) | undefined;
}

export interface EnrolledDevelopmentChannel {
  feedUrl: string;
  keyId: string;
  spki: string;
  adminToken?: string | undefined;
}

export class DeveloperWorkflowController {
  readonly #workspace: string;
  readonly #runtimeExecutable: string;
  readonly #environment: NodeJS.ProcessEnv;
  readonly #loadSigningIdentity:
    | (() => ComponentSigningIdentity | null | Promise<ComponentSigningIdentity | null>)
    | undefined;
  readonly #storeSigningIdentity:
    | ((identity: ComponentSigningIdentity) => void | Promise<void>)
    | undefined;
  readonly #onStatus:
    | ((status: DesktopDeveloperWorkflowStatus) => void)
    | undefined;
  readonly #distributedGatePath: string;
  readonly #localRoot: string;
  #status: DesktopDeveloperWorkflowStatus;
  #coordinator: CoordinatorRuntime | null = null;
  #identity: ComponentSigningIdentity | null = null;
  #initializing: Promise<PublicKeyDocument> | null = null;
  #publication: Promise<void> | null = null;
  #publicationAbortController: AbortController | null = null;
  #publicationCancellationRequested = false;
  #preparedPublication: PreparedComponentReleaseResult | null = null;
  #gateAbortController: AbortController | null = null;
  #gateCompletion: Promise<void> | null = null;
  #gateCancellationRequested = false;
  #publicationFeedUrl = DEV_UPDATE_FEED_URL;
  #publicationCoordinatorUrl = DEV_UPDATE_FEED_URL;
  #publicationAdminToken: string | undefined;

  constructor(options: DeveloperWorkflowOptions) {
    this.#workspace = resolve(options.workspace);
    this.#runtimeExecutable =
      options.runtimeExecutable ?? process.execPath;
    this.#environment = options.environment ?? process.env;
    this.#loadSigningIdentity = options.loadSigningIdentity;
    this.#storeSigningIdentity = options.storeSigningIdentity;
    this.#onStatus = options.onStatus;
    this.#distributedGatePath = join(
      this.#workspace,
      "scripts",
      "dev-distributed-e2e.ts",
    );
    this.#localRoot = join(
      this.#workspace,
      ".codex-runtime",
      "component-updates",
    );
    const sourceAvailable =
      options.sourceMode
      && existsSync(this.#distributedGatePath)
      && existsSync(join(
        this.#workspace,
        "scripts",
        "prepare-packaged-python-source.mjs",
      ))
      && existsSync(join(this.#workspace, "package.json"));
    this.#status = createInitialDeveloperWorkflowStatus(
      sourceAvailable,
      sourceAvailable
        ? null
        : "The local lab is available only from a Mycellios source checkout.",
    );
  }

  get feedUrl(): string {
    return this.#publicationFeedUrl;
  }

  get usesLocalFeed(): boolean {
    return this.#publicationFeedUrl === DEV_UPDATE_FEED_URL;
  }

  get status(): DesktopDeveloperWorkflowStatus {
    return structuredClone(this.#status);
  }

  get hasActiveWork(): boolean {
    return this.#coordinator !== null
      || this.#initializing !== null
      || this.#publication !== null
      || this.#gateCompletion !== null;
  }

  setEnrolled(enrolled: boolean): void {
    if (this.#status.channel.enrolled === enrolled) return;
    this.#patch({
      channel: { ...this.#status.channel, enrolled },
    });
  }

  async signingEnrollment(): Promise<PublicKeyDocument> {
    this.#assertAvailable();
    return this.#ensureDevelopmentKey();
  }

  configureEnrolledChannel(
    input: EnrolledDevelopmentChannel | null,
  ): void {
    if (this.hasActiveWork) {
      throw new Error(
        "Wait for the current development operation before changing channels.",
      );
    }
    if (input === null) {
      this.#publicationFeedUrl = DEV_UPDATE_FEED_URL;
      this.#publicationCoordinatorUrl = DEV_UPDATE_FEED_URL;
      this.#publicationAdminToken = undefined;
      return;
    }
    const feedUrl = validateEnrolledDevelopmentFeed(input.feedUrl);
    this.#publicationFeedUrl = feedUrl;
    this.#publicationCoordinatorUrl = feedUrl;
    this.#publicationAdminToken = input.adminToken?.trim() || undefined;
    this.#patch({
      channel: {
        state: "ready",
        feedUrl,
        keyId: input.keyId,
        fingerprint: developmentKeyFingerprint(input.spki),
        enrolled: true,
        message: feedUrl === DEV_UPDATE_FEED_URL
          ? "Local signed channel ready. No terminal is required."
          : "Remote HTTPS development lab ready. No terminal is required.",
      },
    });
  }

  setPublicationAdminToken(adminToken: string | undefined): void {
    this.#publicationAdminToken = adminToken?.trim() || undefined;
  }

  async initialize(): Promise<PublicKeyDocument> {
    this.#assertAvailable();
    if (this.#gateCompletion) {
      throw new Error(
        "Wait for the distributed test to finish before changing the dev channel.",
      );
    }
    if (this.#initializing) return this.#initializing;
    const operation = this.#initializeInternal();
    this.#initializing = operation;
    try {
      return await operation;
    } finally {
      if (this.#initializing === operation) this.#initializing = null;
    }
  }

  async publish(): Promise<void> {
    this.#assertAvailable();
    if (this.#gateCompletion) {
      throw new Error(
        "Wait for the distributed test to finish before publishing a runtime.",
      );
    }
    if (this.#publication) return this.#publication;
    this.#publicationCancellationRequested = false;
    this.#publicationAbortController = new AbortController();
    const operation = this.#publishInternal();
    this.#publication = operation;
    try {
      await operation;
    } finally {
      if (this.#publication === operation) this.#publication = null;
      this.#publicationAbortController = null;
      this.#publicationCancellationRequested = false;
      this.#preparedPublication = null;
    }
  }

  async cancelPublication(): Promise<void> {
    const abortController = this.#publicationAbortController;
    if (!abortController) return;
    this.#publicationCancellationRequested = true;
    this.#patch({
      publication: {
        ...this.#status.publication,
        state: "cancelling",
        message: "Cancelling before commit and checking the public channel…",
      },
    });
    abortController.abort(
      new Error("development_publication_cancelled_from_panel"),
    );
    const completed = await settlesWithin(
      this.#publication ?? Promise.resolve(),
      120_000,
    );
    if (!completed) {
      this.#patch({
        publication: {
          ...this.#status.publication,
          state: "failed",
          message:
            "Cancellation timed out before publication state could be reconciled.",
          completedAt: new Date().toISOString(),
        },
      });
    }
  }

  startDistributedGate(): DesktopDeveloperWorkflowStatus {
    this.#assertAvailable();
    if (this.#initializing || this.#publication) {
      throw new Error(
        "Wait for the current dev-channel operation to finish.",
      );
    }
    if (this.#gateAbortController) return this.status;
    this.#gateCancellationRequested = false;
    const startedAt = new Date().toISOString();
    this.#patch({
      distributedGate: {
        state: "running",
        phase: "starting",
        message: "Starting the private coordinator and two source workers…",
        startedAt,
        completedAt: null,
        log: [],
      },
    });
    const abortController = new AbortController();
    this.#gateAbortController = abortController;
    const completion = executeDistributedDevelopmentGate(
      ["--timeout", "600"],
      this.#environment,
      {
        workspace: this.#workspace,
        signal: abortController.signal,
        installSignalHandlers: false,
        writeOutput: false,
        onEvent: (event) => this.#recordGateEvent(event),
      },
    ).then(
      (result) => {
        if (!result) {
          throw new Error("The distributed gate returned no evidence.");
        }
        if (this.#gateCancellationRequested) {
          this.#finishGate(
            "cancelled",
            "The distributed test was cancelled and its temporary resources were drained.",
          );
          return;
        }
        this.#finishGate(
          "passed",
          `${result.signedWorkers} signed logical workers completed a real distributed generation.`,
        );
      },
      (error: unknown) => {
        const cleanupFailed = errorText(error).includes(
          "development_gate_cleanup_failed",
        );
        this.#finishGate(
          this.#gateCancellationRequested && !cleanupFailed
            ? "cancelled"
            : "failed",
          this.#gateCancellationRequested && !cleanupFailed
            ? "The distributed test was cancelled."
            : `The distributed test could not start: ${errorText(error)}`,
        );
      },
    ).finally(() => {
      this.#gateAbortController = null;
      this.#gateCompletion = null;
      this.#gateCancellationRequested = false;
    });
    this.#gateCompletion = completion;
    return this.status;
  }

  async cancelDistributedGate(): Promise<void> {
    const abortController = this.#gateAbortController;
    if (!abortController) return;
    this.#gateCancellationRequested = true;
    this.#patch({
      distributedGate: {
        ...this.#status.distributedGate,
        state: "cancelling",
        message: "Cancelling safely and draining temporary resources…",
      },
    });
    abortController.abort(new Error("development_gate_cancelled_from_panel"));
    const completed = await settlesWithin(
      this.#gateCompletion ?? Promise.resolve(),
      60_000,
    );
    if (!completed) {
      this.#finishGate(
        "failed",
        "Cancellation timed out before cleanup could be proven.",
      );
    }
  }

  async stop(): Promise<void> {
    await this.cancelDistributedGate();
    await this.cancelPublication();
    await this.#publication?.catch(() => undefined);
    const runtime = this.#coordinator;
    if (!runtime) {
      this.#patch({
        channel: {
          ...this.#status.channel,
          state: "stopped",
          message: this.usesLocalFeed
            ? "The local signed channel is stopped."
            : "This computer left the remote development lab.",
        },
      });
      return;
    }
    this.#patch({
      channel: {
        ...this.#status.channel,
        state: "stopping",
        message: "Stopping the local signed channel…",
      },
    });
    this.#coordinator = null;
    await runtime.close();
    this.#patch({
      channel: {
        ...this.#status.channel,
        state: "stopped",
        message: "The local signed channel is stopped.",
      },
    });
  }

  async #initializeInternal(): Promise<PublicKeyDocument> {
    this.#publicationFeedUrl = DEV_UPDATE_FEED_URL;
    this.#publicationCoordinatorUrl = DEV_UPDATE_FEED_URL;
    this.#publicationAdminToken = undefined;
    this.#patch({
      channel: {
        ...this.#status.channel,
        state: "starting",
        feedUrl: DEV_UPDATE_FEED_URL,
        message: "Preparing the signing identity and local channel…",
      },
    });
    try {
      const enrollment = await this.#ensureDevelopmentKey();
      await this.#startCoordinator(enrollment);
      this.#patch({
        channel: {
          state: "ready",
          feedUrl: DEV_UPDATE_FEED_URL,
          keyId: enrollment.keyId,
          fingerprint: developmentKeyFingerprint(enrollment.spki),
          enrolled: this.#status.channel.enrolled,
          message: "Local signed channel ready. No terminal is required.",
        },
      });
      return enrollment;
    } catch (error) {
      this.#patch({
        channel: {
          ...this.#status.channel,
          state: "error",
          message: errorText(error),
        },
      });
      throw error;
    }
  }

  async #publishInternal(): Promise<void> {
    if (
      this.#status.channel.state !== "ready"
      || (
        this.#publicationFeedUrl === DEV_UPDATE_FEED_URL
        && !this.#coordinator
      )
    ) {
      await this.initialize();
    }
    const signal = this.#publicationAbortController?.signal;
    const identity = await this.#ensureDevelopmentIdentity();
    const feedUrl = this.#publicationFeedUrl;
    const coordinatorUrl = this.#publicationCoordinatorUrl;
    const adminToken = this.#publicationAdminToken;
    if (
      this.#status.channel.keyId !== identity.enrollment.keyId
      || this.#status.channel.fingerprint
        !== developmentKeyFingerprint(identity.enrollment.spki)
    ) {
      throw new Error(
        "The enrolled channel does not match this computer's development signing identity.",
      );
    }
    this.#patch({
      publication: {
        state: "running",
        message: "Preparing and signing the current Python runtime…",
        revision: null,
        completedAt: null,
      },
    });
    try {
      const prepared = await prepareComponentRelease({
        workspace: this.#workspace,
        localRoot: this.#localRoot,
        feedUrl,
        target: developmentTarget(),
        identity,
        keyDirectory: join(this.#localRoot, "keys", "dev"),
        nodeExecutable: this.#runtimeExecutable,
        processEnvironment: { ...this.#environment },
        signal,
      });
      this.#preparedPublication = prepared;
      this.#patch({
        publication: {
          state: "running",
          message:
            "Publishing the signed artifact and verifying public readback…",
          revision: prepared.releaseRevision,
          completedAt: null,
        },
      });
      await publishComponentRelease({
        releaseDirectory: prepared.releaseDirectory,
        coordinatorUrl,
        target: developmentTarget(),
        ...(adminToken ? { adminToken } : {}),
        signal,
      });
      this.#patch({
        publication: {
          state: "passed",
          message:
            "The signed runtime was published and read back successfully.",
          revision: prepared.releaseRevision,
          completedAt: new Date().toISOString(),
        },
      });
    } catch (error) {
      if (this.#publicationCancellationRequested) {
        if (
          error instanceof ComponentPublicationError
          && error.commitPhase === "manifest-outcome-unknown"
        ) {
          this.#patch({
            publication: {
              state: "failed",
              message:
                "Cancellation could not prove whether the manifest committed; Mycellios did not report a false cancelled state.",
              revision: this.#preparedPublication?.releaseRevision ?? null,
              completedAt: new Date().toISOString(),
            },
          });
          return;
        }
        const committed =
          error instanceof ComponentPublicationError
          && (
            error.commitPhase === "manifest-committed"
            || error.commitPhase === "readback-verified"
          );
        if (committed && this.#preparedPublication) {
          this.#patch({
            publication: {
              state: "running",
              message:
                "The manifest may already be public; reconciling it before reporting the result…",
              revision: this.#preparedPublication.releaseRevision,
              completedAt: null,
            },
          });
          await publishComponentRelease({
            releaseDirectory: this.#preparedPublication.releaseDirectory,
            coordinatorUrl,
            target: developmentTarget(),
            ...(adminToken ? { adminToken } : {}),
          });
          this.#patch({
            publication: {
              state: "passed",
              message:
                "Cancellation arrived after commit; the published runtime was reconciled and verified.",
              revision: this.#preparedPublication.releaseRevision,
              completedAt: new Date().toISOString(),
            },
          });
          return;
        }
        this.#patch({
          publication: {
            state: "cancelled",
            message:
              "Runtime publication was cancelled before a manifest commit.",
            revision: this.#preparedPublication?.releaseRevision ?? null,
            completedAt: new Date().toISOString(),
          },
        });
        return;
      }
      this.#patch({
        publication: {
          state: "failed",
          message: errorText(error),
          revision: null,
          completedAt: new Date().toISOString(),
        },
      });
      throw error;
    }
  }

  async #ensureDevelopmentIdentity(): Promise<ComponentSigningIdentity> {
    if (this.#identity) return this.#identity;
    const stored = await this.#loadSigningIdentity?.();
    if (stored) {
      this.#identity = stored;
      return stored;
    }
    if (!this.#storeSigningIdentity) {
      throw new Error(
        "Secure operating-system storage is unavailable for the development signing identity.",
      );
    }
    let generated: ComponentSigningIdentity | null = null;
    await createDevelopmentSigningIdentity({
      localRoot: this.#localRoot,
      persistToFiles: false,
      storeIdentity: async (identity) => {
        await this.#storeSigningIdentity!(identity);
        generated = identity;
      },
    });
    if (!generated) {
      throw new Error("The development signing identity was not created.");
    }
    this.#identity = generated;
    return generated;
  }

  async #ensureDevelopmentKey(): Promise<PublicKeyDocument> {
    const identity = await this.#ensureDevelopmentIdentity();
    if (identity.enrollment.channel !== "dev") {
      throw new Error("The development signing identity has the wrong channel.");
    }
    return identity.enrollment as PublicKeyDocument;
  }

  async #startCoordinator(
    enrollment: PublicKeyDocument,
  ): Promise<void> {
    if (this.#coordinator) return;
    const runtime = await createCoordinator({
      host: "127.0.0.1",
      port: DEV_UPDATE_PORT,
      databasePath: ":memory:",
      requestTimeoutMs: 120_000,
      componentUpdatesPath: join(this.#localRoot, "store"),
      componentUpdatePinnedKeys: {
        dev: [{ keyId: enrollment.keyId, spki: enrollment.spki }],
      },
    });
    try {
      await runtime.app.listen({
        host: "127.0.0.1",
        port: DEV_UPDATE_PORT,
      });
      this.#coordinator = runtime;
    } catch (error) {
      await runtime.close().catch(() => undefined);
      throw error;
    }
    this.#patch({
      channel: {
        state: "ready",
        feedUrl: DEV_UPDATE_FEED_URL,
        keyId: enrollment.keyId,
        fingerprint: developmentKeyFingerprint(enrollment.spki),
        enrolled: this.#status.channel.enrolled,
        message: "Local signed channel ready.",
      },
    });
  }

  #recordGateEvent(event: DevelopmentGateEvent): void {
    const line = `[${event.phase}] ${event.message}`
      .trim()
      .slice(0, MAX_VISIBLE_LOG_LINE_LENGTH);
    if (!line) return;
    this.#patch({
      distributedGate: {
        ...this.#status.distributedGate,
        phase: event.phase,
        message: event.message,
        log: [
          ...this.#status.distributedGate.log,
          line,
        ].slice(-MAX_VISIBLE_LOG_LINES),
      },
    });
  }

  #finishGate(
    state: "passed" | "failed" | "cancelled",
    message: string,
  ): void {
    this.#patch({
      distributedGate: {
        ...this.#status.distributedGate,
        state,
        phase: state === "passed" ? "complete" : state,
        message,
        completedAt: new Date().toISOString(),
      },
    });
  }

  #patch(
    patch: Partial<DesktopDeveloperWorkflowStatus>,
  ): void {
    this.#status = { ...this.#status, ...patch };
    this.#onStatus?.(this.status);
  }

  #assertAvailable(): void {
    if (!this.#status.available) {
      throw new Error(
        this.#status.unavailableReason
        ?? "The local developer workflow is unavailable.",
      );
    }
  }
}

export function createInitialDeveloperWorkflowStatus(
  available: boolean,
  unavailableReason: string | null,
): DesktopDeveloperWorkflowStatus {
  return {
    available,
    unavailableReason,
    channel: {
      state: "stopped",
      feedUrl: null,
      keyId: null,
      fingerprint: null,
      enrolled: false,
      message: available
        ? "Set up the local signed channel from this panel."
        : unavailableReason ?? "Local developer tools are unavailable.",
    },
    publication: {
      state: "idle",
      message: "No runtime change has been published in this session.",
      revision: null,
      completedAt: null,
    },
    distributedGate: {
      state: "idle",
      phase: null,
      message: "The distributed local test has not run in this session.",
      startedAt: null,
      completedAt: null,
      log: [],
    },
  };
}

export async function readDevelopmentEnrollment(
  path: string,
): Promise<PublicKeyDocument> {
  const value = JSON.parse(await readFile(path, "utf8")) as Record<
    string,
    unknown
  >;
  if (
    value.schema !== "mycellios-component-update-public-key/1"
    || value.channel !== "dev"
    || typeof value.keyId !== "string"
    || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value.keyId)
    || typeof value.spki !== "string"
    || !/^[A-Za-z0-9_-]+$/.test(value.spki)
  ) {
    throw new Error("The local development enrollment is invalid.");
  }
  const encoded = Buffer.from(value.spki, "base64url");
  if (encoded.toString("base64url") !== value.spki) {
    throw new Error("The local development enrollment is not canonical.");
  }
  const key = createPublicKey({
    key: encoded,
    format: "der",
    type: "spki",
  });
  const canonical = key.export({ format: "der", type: "spki" });
  if (
    key.asymmetricKeyType !== "ed25519"
    || encoded.length > 128
    || !Buffer.isBuffer(canonical)
    || !canonical.equals(encoded)
  ) {
    throw new Error("The local development enrollment is not Ed25519.");
  }
  return value as unknown as PublicKeyDocument;
}

async function settlesWithin(
  promise: Promise<unknown>,
  milliseconds: number,
): Promise<boolean> {
  let timeout: NodeJS.Timeout | null = null;
  try {
    return await Promise.race([
      promise.then(() => true, () => true),
      new Promise<boolean>((resolvePromise) => {
        timeout = setTimeout(() => resolvePromise(false), milliseconds);
        timeout.unref();
      }),
    ]);
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function developmentTarget(): {
  channel: "dev";
  platform: "win32" | "linux" | "darwin";
  arch: "x64" | "arm64";
} {
  if (!["win32", "linux", "darwin"].includes(process.platform)) {
    throw new Error(`Unsupported development platform: ${process.platform}.`);
  }
  if (!["x64", "arm64"].includes(process.arch)) {
    throw new Error(`Unsupported development architecture: ${process.arch}.`);
  }
  return {
    channel: "dev",
    platform: process.platform as "win32" | "linux" | "darwin",
    arch: process.arch as "x64" | "arm64",
  };
}

function developmentKeyFingerprint(spki: string): string {
  return createHash("sha256")
    .update(Buffer.from(spki, "base64url"))
    .digest("hex")
    .match(/.{1,4}/g)!
    .join(":");
}

function validateEnrolledDevelopmentFeed(raw: string): string {
  if (typeof raw !== "string" || raw.length === 0 || raw.length > 4_096) {
    throw new Error("The enrolled development feed URL is invalid.");
  }
  const url = new URL(raw.endsWith("/") ? raw : `${raw}/`);
  const loopback = new Set(["localhost", "127.0.0.1", "::1", "[::1]"])
    .has(url.hostname.toLowerCase());
  if (
    (url.protocol !== "https:" && !(url.protocol === "http:" && loopback))
    || url.username
    || url.password
    || url.search
    || url.hash
  ) {
    throw new Error(
      "Remote development feeds require HTTPS without embedded credentials.",
    );
  }
  return url.toString().replace(/\/+$/, "");
}
