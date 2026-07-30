import websocket from "@fastify/websocket";
import staticFiles from "@fastify/static";
import Fastify, { type FastifyInstance, type FastifyReply, type FastifyRequest } from "fastify";
import { createHash, timingSafeEqual } from "node:crypto";
import {
  createReadStream,
  existsSync,
  lstatSync,
  readFileSync,
  readdirSync,
  statSync,
} from "node:fs";
import { dirname, resolve, sep } from "node:path";
import { z, ZodError } from "zod";
import {
  StableBenchmarkActivationTracker,
  type BenchmarkActivation,
} from "../benchlab/activation-tracker.js";
import { loadBenchmarkRuns } from "../benchlab/history.js";
import {
  buildCoordinatorBenchmarkTelemetrySnapshot,
  buildCoordinatorBenchmarkInventory,
  coordinatorBenchmarkActivations,
  coordinatorBenchmarkModelIdentity,
  runAndPersistCoordinatorSuite,
  type CoordinatorBenchmarkModel,
} from "../benchlab/coordinator-suite.js";
import {
  chatCompletionRequestSchema,
  workerRegistrationSchema,
} from "../contracts/schemas.js";
import {
  redactDiagnosticDetails,
  redactDiagnosticText,
  remoteDiagnosticBatchSchema,
  remoteDiagnosticQuerySchema,
} from "../contracts/remote-diagnostics.js";
import {
  WORKER_PROTOCOL_MAX,
  WORKER_PROTOCOL_MIN,
  workerAdmissionChallengeRequestSchema,
  workerAdmissionIdentitySchema,
  workerCredentialRotationChallengeRequestSchema,
  workerCredentialRotationProofSchema,
} from "../contracts/worker-admission.js";
import { contributionAckEnvelopeSchema } from "../contracts/worker-protocol.js";
import {
  type NativeBuildIdentity,
} from "../contracts/build-identity.js";
import type { ChatCompletionRequest } from "../contracts/types.js";
import {
  assertCoordinatorNetworkSecurity,
  type CoordinatorConfig,
} from "../core/config.js";
import {
  readNativeRuntimeBuildMetadata,
  type NativeRuntimeBuildMetadata,
} from "../core/native-build-identity.js";
import { workerRegistrationDigest } from "../core/worker-admission-digest.js";
import { Scheduler } from "../scheduler/scheduler.js";
import { MeshDatabase } from "../storage/database.js";
import {
  MeshStore,
  type StoredNetworkTelemetrySample,
  type StoredRequestedModel,
  type StoredWorker,
} from "../storage/store.js";
import { SupabasePersistence } from "../storage/supabase-sync.js";
import {
  buildSupportAssistantMessages,
  resolveSupportAssistantModel,
  supportAssistantChatRequestSchema,
  supportAssistantSettingsUpdateSchema,
} from "../support/assistant.js";
import { MeshService, MeshServiceError, type JobStreamEvent } from "./mesh-service.js";
import { MobileComputeHub, type MobileWorkerSnapshot } from "./mobile-compute-hub.js";
import {
  verifyGitHubReleaseUploadToken,
  type GitHubReleaseClaims,
} from "./github-oidc.js";
import type { ModelActivationManager } from "./model-activation-manager.js";
import {
  inspectHubModelCapacity,
  requestedModelCapacityViews,
  searchHubModelCatalog,
  shouldQueueAutomaticActivation,
  type ModelActivationProgressEvent,
} from "./model-catalog.js";
import {
  MAX_RELEASE_CHUNK_SIZE_BYTES,
  NativeReleaseTransactionStore,
  parseReleaseChunkMetadata,
  parseReleaseTransactionIdentity,
  validateReleaseAssetName,
  type ReleaseAssetChannel,
} from "./release-upload.js";
import { WorkerHub } from "./worker-hub.js";
import {
  modelHasGpuFallback,
  verifiedGpuCapacityCanRepairModel,
} from "./connected-executor-activation.js";
import {
  ContentHubClient,
  registerContentHubRoutes,
} from "./content-hub.js";
import { SupabaseAuthService } from "./supabase-auth.js";
import {
  DeploymentControlPlane,
  type DeploymentOperation,
} from "./deployment-control-plane.js";
import { stripWorkerDeclaredEvidence } from "./evidence-authority.js";
import {
  activationFailureIsTransient,
  classifyActivationIncident,
  type ActivationIncident,
} from "./activation-incident.js";
import type { AuthenticatedNetworkUser } from "./supabase-auth.js";
import {
  API_KEY_PREFIX,
  ApiAccessError,
  ApiAccessManager,
  apiAccountJson,
  apiKeyJson,
  apiUsageJson,
  type ApiKeyPrincipal,
} from "./api-access.js";
import {
  WorkerAdmissionAuthority,
  WorkerAdmissionError,
  admissionCredentialSummary,
  selectWorkerProtocolVersion,
} from "./worker-admission.js";
import {
  WORKER_SESSION_TOKEN_PREFIX,
  issueWorkerSessionToken,
  verifyWorkerSessionToken,
} from "./worker-session-token.js";
import { FleetContributionController } from "./fleet-contribution-control.js";

export function automaticActivationFailureIsTransient(message: string): boolean {
  return activationFailureIsTransient(message);
}

export const DEFAULT_AUTOMATIC_ACTIVATION_RETRY_DELAYS_MS = [
  5_000,
  15_000,
  30_000,
  120_000,
  300_000,
] as const;

export const NETWORK_TELEMETRY_INTERVAL_MS = 10 * 60_000;
export const NETWORK_TELEMETRY_RETENTION_DAYS = 90;

const SIGNED_WORKER_ADMISSION_PATHS = new Set([
  "/internal/v1/workers/admission-challenge",
  "/internal/v1/workers/credential-rotation-challenge",
  "/internal/v1/workers/credential-rotation",
  "/internal/v1/workers/register",
]);
const WORKER_CONNECT_PATH = "/internal/v1/workers/connect";

const networkHistoryRanges = {
  "24h": 24,
  "7d": 7 * 24,
  "30d": 30 * 24,
  "90d": NETWORK_TELEMETRY_RETENTION_DAYS * 24,
} as const;

export interface AutomaticActivationRetryState {
  retryCount: number;
  nextAttemptAt: number;
  lastError: string;
  updatedAt: number;
  launching: boolean;
}

export function nextAutomaticActivationRetry(
  retriesStarted: number,
  message: string,
  now = Date.now(),
  delays: readonly number[] = DEFAULT_AUTOMATIC_ACTIVATION_RETRY_DELAYS_MS,
): AutomaticActivationRetryState | null {
  const delay = delays[retriesStarted];
  if (delay === undefined) return null;
  return {
    retryCount: retriesStarted,
    nextAttemptAt: now + Math.max(0, delay),
    lastError: message,
    updatedAt: now,
    launching: false,
  };
}

export interface CoordinatorRuntime {
  app: FastifyInstance;
  database: MeshDatabase;
  store: MeshStore;
  scheduler: Scheduler;
  hub: WorkerHub;
  mobileHub: MobileComputeHub;
  service: MeshService;
  apiAccess: ApiAccessManager;
  persistence: SupabasePersistence | null;
  deploymentController: DeploymentControlPlane;
  close(): Promise<void>;
}

export interface CoordinatorActivationContext {
  store: MeshStore;
  hub: WorkerHub;
  deploymentController: DeploymentControlPlane;
}

export async function createCoordinator(
  config: CoordinatorConfig,
  options: {
    logger?: boolean;
    activationManager?: ModelActivationManager;
    activationManagerFactory?: (context: CoordinatorActivationContext) => ModelActivationManager;
    releaseTokenVerifier?: (token: string) => Promise<GitHubReleaseClaims>;
    mobileDisconnectedRetentionMs?: number;
    automaticActivationRetryDelaysMs?: readonly number[];
    buildIdentity?: NativeBuildIdentity | null;
    runtimeMetadata?: NativeRuntimeBuildMetadata;
    benchmarkStorageRoot?: string;
    releaseTransactionRoot?: string;
    releaseChunkBodyLimitBytes?: number;
    supabaseAuthService?: SupabaseAuthService;
  } = {},
): Promise<CoordinatorRuntime> {
  assertCoordinatorNetworkSecurity(config);
  const runtimeMetadata = options.runtimeMetadata
    ?? readNativeRuntimeBuildMetadata(resolve(import.meta.dirname, "../.."));
  const coordinatorBuildIdentity = options.buildIdentity === undefined
    ? runtimeMetadata.buildIdentity
    : options.buildIdentity;
  if (
    coordinatorBuildIdentity
    && coordinatorBuildIdentity.version !== runtimeMetadata.version
  ) {
    throw new Error(
      `coordinator_build_version_mismatch:${coordinatorBuildIdentity.version}:${runtimeMetadata.version}`,
    );
  }
  const runtimeVersion = runtimeMetadata.version;
  const runtimeRevision = runtimeMetadata.revision;
  const app = Fastify({ logger: options.logger ?? false, bodyLimit: 2 * 1024 * 1024 });
  const apiAccessEnabled = config.apiAccessEnabled ?? false;
  app.addHook("onRequest", async (request, reply) => {
    // The public UI and mobile worker must never be embeddable as drive-by
    // compute. Apply the policy to static and dynamic responses so it remains
    // true even when a reverse proxy does not add security headers.
    reply.header("Content-Security-Policy", "frame-ancestors 'none'; object-src 'none'; base-uri 'self'");
    reply.header("X-Frame-Options", "DENY");
    reply.header("X-Content-Type-Options", "nosniff");
    reply.header("Referrer-Policy", "strict-origin-when-cross-origin");
    const path = request.url.split("?", 1)[0] ?? request.url;
    if (path.startsWith("/v1/")) {
      reply.header("Access-Control-Allow-Origin", "*");
      reply.header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS");
      reply.header(
        "Access-Control-Allow-Headers",
        "Authorization,Content-Type,Idempotency-Key",
      );
      reply.header(
        "Access-Control-Expose-Headers",
        "X-Network-Request-Id,X-Network-Session-Id,X-RateLimit-Limit,"
        + "X-RateLimit-Remaining,X-RateLimit-Reset,X-Token-Balance",
      );
      reply.header("Vary", "Origin");
      if (request.method === "OPTIONS") return reply.code(204).send();
    }
  });
  const internalToken = config.internalToken ?? config.modelAdminToken ?? config.networkToken;
  app.addHook("onRequest", async (request, reply) => {
    const path = request.url.split("?", 1)[0] ?? request.url;
    if (!path.startsWith("/internal/v1/mobile/experts/")) return;
    if (!internalToken) {
      return reply.code(503).send({
        error: {
          code: "internal_administration_not_configured",
          message: "Mobile expert administration requires MYCELLIOS_INTERNAL_TOKEN.",
        },
      });
    }
    const received = parseBearerToken(request.headers.authorization);
    if (!received || !constantTimeEqual(received, internalToken)) {
      return reply.code(401).send({ error: { code: "invalid_internal_token" } });
    }
  });
  if (config.networkToken) {
    const expectedToken = config.networkToken;
    app.addHook("onRequest", async (request, reply) => {
      const path = request.url.split("?", 1)[0] ?? request.url;
      if (
        !path.startsWith("/internal/v1/")
        && !(path.startsWith("/v1/") && !apiAccessEnabled)
      ) return;
      if (path.startsWith("/internal/v1/releases/")) return;
      if (path === "/v1/auth/me") return;
      const received = parseBearerToken(request.headers.authorization);
      // Remote public workers prove possession of a stable device key on these
      // routes. Registration returns a narrowly scoped session token for the
      // worker WebSocket; the global network secret never leaves the server.
      if (SIGNED_WORKER_ADMISSION_PATHS.has(path)) {
        if (!received || constantTimeEqual(received, expectedToken)) return;
        return reply.code(401).send({ error: { code: "invalid_network_token" } });
      }
      // Mobile expert administration has its own stronger control-plane
      // credential above. Requiring both secrets in one Authorization header
      // would make the route impossible to use when the tokens differ.
      if (path.startsWith("/internal/v1/mobile/experts/")) return;
      if (
        path === WORKER_CONNECT_PATH
        && received?.startsWith(`${WORKER_SESSION_TOKEN_PREFIX}.`)
      ) return;
      if (!received || !constantTimeEqual(received, expectedToken)) {
        return reply.code(401).send({ error: { code: "invalid_network_token" } });
      }
    });
  }
  const supabaseAuth = options.supabaseAuthService ?? (config.supabaseUrl && config.supabaseServiceRoleKey
    ? new SupabaseAuthService(config.supabaseUrl, config.supabaseServiceRoleKey)
    : null);
  const authorizeAdministrativeMutation = async (
    request: FastifyRequest,
    reply: FastifyReply,
    allowedRoles: readonly NonNullable<AuthenticatedNetworkUser["role"]>[] =
      ["owner", "admin", "operator"],
  ): Promise<boolean> => {
    const expected = config.modelAdminToken;
    if (!expected && isTrustedLocalRequest(request)) return true;
    const legacyHeader = request.headers["x-mycellios-admin-token"];
    const legacyToken = typeof legacyHeader === "string"
      ? legacyHeader.trim()
      : Array.isArray(legacyHeader) ? legacyHeader[0]?.trim() : undefined;
    const bearer = parseBearerToken(request.headers.authorization);
    if (expected && (
      (legacyToken && constantTimeEqual(legacyToken, expected))
      || (bearer && constantTimeEqual(bearer, expected))
    )) return true;
    if (bearer && supabaseAuth) {
      try {
        const user = await supabaseAuth.authenticate(bearer);
        if (user?.role && allowedRoles.includes(user.role)) return true;
        if (user) {
          void reply.code(403).send({
            error: {
              code: "insufficient_network_role",
              message: "Your Mycellios account does not have permission to administer the shared network.",
            },
          });
          return false;
        }
      } catch (error) {
        app.log.warn(
          { error: error instanceof Error ? error.message : String(error) },
          "Supabase authorization lookup failed",
        );
      }
    }
    if (!expected && !supabaseAuth) {
      void reply.code(503).send({
        error: {
          code: "model_administration_not_configured",
          message: "Remote network administration requires Supabase Auth or MYCELLIOS_MODEL_ADMIN_TOKEN.",
        },
      });
      return false;
    }
    void reply.code(401).send({
      error: {
        code: "invalid_model_admin_token",
        message: "Sign in with an authorized Mycellios account or enter the administrator token.",
      },
    });
    return false;
  };
  app.addContentTypeParser(
    "application/octet-stream",
    { parseAs: "buffer", bodyLimit: 512 * 1024 * 1024 },
    (_request, body, done) => done(null, body),
  );
  const database = new MeshDatabase(config.databasePath);
  const workerAdmission = new WorkerAdmissionAuthority(database);
  const workerSessionPrincipals = new WeakMap<FastifyRequest, string>();
  if (config.networkToken) {
    app.addHook("onRequest", async (request, reply) => {
      const path = request.url.split("?", 1)[0] ?? request.url;
      if (path !== WORKER_CONNECT_PATH) return;
      const token = parseBearerToken(request.headers.authorization);
      if (token && constantTimeEqual(token, config.networkToken!)) return;
      const claims = token ? verifyWorkerSessionToken(config.networkToken!, token) : null;
      const credential = claims
        ? database.getWorkerAdmissionCredential(claims.identityKind, claims.identityId)
        : null;
      if (
        !claims
        || !credential
        || credential.status !== "active"
        || credential.fingerprint !== claims.credentialFingerprint
      ) {
        return reply.code(401).send({
          error: { code: "invalid_worker_session_token" },
        });
      }
      workerSessionPrincipals.set(request, claims.workerId);
    });
  }
  const apiAccess = new ApiAccessManager(database, {
    starterTokens: config.apiStarterTokens ?? 25_000,
    requestsPerMinute: config.apiRequestsPerMinute ?? 30,
    maxConcurrent: config.apiMaxConcurrent ?? 2,
    maxActiveKeys: config.apiMaxActiveKeys ?? 10,
  });
  type ApiRequestPrincipal =
    | { kind: "system" }
    | {
        kind: "user";
        userId: string;
        role: AuthenticatedNetworkUser["role"];
      }
    | ApiKeyPrincipal;
  const apiPrincipals = new WeakMap<FastifyRequest, ApiRequestPrincipal>();
  const principalFor = (request: FastifyRequest): ApiRequestPrincipal => {
    const principal = apiPrincipals.get(request);
    if (!principal) {
      throw new ApiAccessError(
        "authentication_required",
        "Sign in or provide a Mycellios API key.",
        401,
      );
    }
    return principal;
  };
  app.addHook("preHandler", async (request, reply) => {
    if (!apiAccessEnabled || request.method === "OPTIONS") return;
    const path = request.url.split("?", 1)[0] ?? request.url;
    if (!path.startsWith("/v1/") || path === "/v1/auth/me") return;
    const token = parseBearerToken(request.headers.authorization);
    if (!token) {
      return reply.code(401).send({
        error: {
          code: "authentication_required",
          message: "Sign in or provide a Mycellios API key.",
        },
      });
    }
    if (config.networkToken && constantTimeEqual(token, config.networkToken)) {
      apiPrincipals.set(request, { kind: "system" });
      return;
    }
    if (token.startsWith(API_KEY_PREFIX)) {
      const principal = apiAccess.authenticateKey(token);
      if (!principal) {
        return reply.code(401).send({
          error: { code: "invalid_api_key", message: "The Mycellios API key is invalid or revoked." },
        });
      }
      apiPrincipals.set(request, principal);
      return;
    }
    if (!supabaseAuth) {
      return reply.code(503).send({
        error: {
          code: "account_authentication_unavailable",
          message: "Account authentication is not configured on this coordinator.",
        },
      });
    }
    try {
      const user = await supabaseAuth.authenticate(token);
      if (!user) {
        return reply.code(401).send({
          error: { code: "invalid_access_token", message: "The account session is invalid or expired." },
        });
      }
      apiAccess.getOrCreateAccount(user.id);
      apiPrincipals.set(request, { kind: "user", userId: user.id, role: user.role });
    } catch (error) {
      app.log.warn(
        { error: error instanceof Error ? error.message : String(error) },
        "Account authentication lookup failed",
      );
      return reply.code(503).send({
        error: {
          code: "account_authentication_unavailable",
          message: "The account service could not validate this request.",
        },
      });
    }
  });
  const store = new MeshStore(database);
  const persistence = config.supabaseUrl && config.supabaseServiceRoleKey
    ? new SupabasePersistence(store, {
        url: config.supabaseUrl,
        serviceRoleKey: config.supabaseServiceRoleKey,
        required: config.supabasePersistenceRequired ?? false,
      })
    : null;
  await persistence?.initialize();
  const deploymentController = new DeploymentControlPlane(store);
  deploymentController.initialize();
  await app.register(websocket, { options: { maxPayload: 10 * 1024 * 1024 } });
  const mobileAssetsPath = resolveMobileAssetsPath(
    config.mobileAssetsPath,
    runtimeMetadata.root,
  );
  if (mobileAssetsPath) {
    await app.register(staticFiles, {
      root: mobileAssetsPath,
      prefix: "/mobile/",
      decorateReply: false,
      index: "index.html",
      cacheControl: false,
      setHeaders: setPublicAssetCacheHeaders,
    });
    app.get("/mobile", async (_request, reply) => reply.redirect("/mobile/"));
  }
  const desktopUpdatesPath = resolveDesktopUpdatesPath(
    config.desktopUpdatesPath,
    runtimeMetadata.root,
  );
  const releaseDownloadsPath = resolveReleaseDownloadsPath(
    config.releaseDownloadsPath,
    config.landingAssetsPath,
    runtimeMetadata.root,
  );
  const publicAssetVersion = runtimeVersion;
  const uploadUpdatesRoot = releaseAssetRoot(
    "updates",
    config.desktopUpdatesPath,
    config.releaseDownloadsPath,
    config.landingAssetsPath,
    runtimeMetadata.root,
  );
  const uploadDownloadsRoot = releaseAssetRoot(
    "downloads",
    config.desktopUpdatesPath,
    config.releaseDownloadsPath,
    config.landingAssetsPath,
    runtimeMetadata.root,
  );
  const releaseTransactions = coordinatorBuildIdentity && runtimeRevision
    ? new NativeReleaseTransactionStore({
      storageRoot: options.releaseTransactionRoot
        ?? releaseTransactionStorageRoot(
          uploadUpdatesRoot,
          uploadDownloadsRoot,
          runtimeMetadata.root,
        ),
      legacyUpdatesRoot: desktopUpdatesPath,
      legacyDownloadsRoot: releaseDownloadsPath,
      sourceId: coordinatorBuildIdentity.sourceId,
      revision: runtimeRevision,
      version: runtimeVersion,
    })
    : null;
  await releaseTransactions?.initialize();
  app.get("/updates/win32/x64/:fileName", async (request, reply) => {
    const { fileName } = z.object({
      fileName: z.string().min(1).max(160),
    }).parse(request.params);
    return serveReleaseAsset(
      request,
      reply,
      await releasePublicAssetPath(
        releaseTransactions,
        "updates",
        fileName,
        desktopUpdatesPath,
      ),
    );
  });
  app.get("/downloads/", async (_request, reply) => reply.redirect("/downloads"));
  app.get("/downloads/:fileName", async (request, reply) => {
    const { fileName } = z.object({
      fileName: z.string().min(1).max(160),
    }).parse(request.params);
    return serveReleaseAsset(
      request,
      reply,
      await releasePublicAssetPath(
        releaseTransactions,
        "downloads",
        fileName,
        releaseDownloadsPath,
      ),
    );
  });
  app.get("/downloads/windows", async (_request, reply) => {
    reply.header("Cache-Control", "no-cache, no-store, must-revalidate");
    return reply.redirect(`/updates/win32/x64/mycellios-setup.exe?v=${publicAssetVersion}`);
  });
  const hub = new WorkerHub(store);
  const fleetContribution = new FleetContributionController(store, hub);
  hub.attach(app, {
    authorizedWorkerId: (request) => workerSessionPrincipals.get(request) ?? null,
  });
  const scheduler = new Scheduler(store, {
    runtimeLinkObservations: () => hub.runtimeLinkObservations(),
    strictRuntimeLinks: true,
  });
  const mobileHub = new MobileComputeHub({
    joinToken: config.mobileJoinToken,
    expertArtifactsPath: config.mobileExpertArtifactsPath,
    disconnectedRetentionMs: options.mobileDisconnectedRetentionMs,
    admissionAuthority: workerAdmission,
    isTrustedLocalRequest,
    ...(persistence
      ? { onArtifactStored: (artifact: Parameters<SupabasePersistence["registerArtifactBackup"]>[0]) =>
          persistence.registerArtifactBackup(artifact) }
      : {}),
  });
  mobileHub.attach(app);
  if (persistence && config.mobileExpertArtifactsPath) {
    queueExistingMobileArtifacts(persistence, config.mobileExpertArtifactsPath);
  }
  const service = new MeshService(store, scheduler, hub, config.requestTimeoutMs);
  const supportAssistantRateLimits = new Map<string, SupportAssistantRateState>();
  const publicCatalogRateLimits = new Map<string, PublicCatalogRateState>();
  let activeSupportAssistantRequests = 0;
  let activePublicCatalogRequests = 0;
  const activationManager = options.activationManager ?? options.activationManagerFactory?.({
    store,
    hub,
    deploymentController,
  });
  await activationManager?.initialize();
  const benchmarkWorkspace = runtimeMetadata.root;
  const benchmarkStorageRoot =
    options.benchmarkStorageRoot?.trim()
    || process.env.MYCELLIOS_BENCHMARK_ROOT?.trim()
    || resolve(runtimeMetadata.root, "benchmarks");
  const benchmarkHistoryDirectory = resolve(benchmarkStorageRoot, "history");
  for (const run of loadBenchmarkRuns(benchmarkWorkspace, benchmarkHistoryDirectory)) {
    store.queueBenchmarkRun(run);
  }
  void persistence?.flush();
  type PersistedBenchmark = Awaited<ReturnType<typeof runAndPersistCoordinatorSuite>>;
  interface QueuedAutomaticBenchmark {
    model: CoordinatorBenchmarkModel;
    activation: BenchmarkActivation;
  }
  let benchmarkRunInFlight: Promise<PersistedBenchmark> | null = null;
  const automaticBenchmarkQueue: QueuedAutomaticBenchmark[] = [];
  const queuedAutomaticBenchmarkActivations = new Set<string>();
  const benchmarkActivationTracker = new StableBenchmarkActivationTracker(3, 12_000);
  const automaticBenchmarkRetryAttempts = new Map<string, number>();
  const automaticBenchmarkRetryTimers = new Set<NodeJS.Timeout>();
  const benchmarkTargetForModel = (modelId: string): CoordinatorBenchmarkModel => {
    const requested = store.getRequestedModel(modelId);
    const connectedWorkerIds = hub.connectedWorkerIds();
    return coordinatorBenchmarkModelIdentity(
      {
        id: modelId,
        source: requested?.source ?? modelId,
        revision: requested?.revision ?? null,
      },
      store.listWorkers(),
      connectedWorkerIds,
    );
  };
  const benchmarkableActiveModelIds = (): string[] => {
    const connectedWorkerIds = hub.connectedWorkerIds();
    const realDeploymentModels = new Set(
      store.listWorkers()
        .filter((worker) => connectedWorkerIds.has(worker.id))
        .flatMap((worker) => worker.capabilities.deployments)
        .filter((deployment) => deployment.adapter === "mycellios-pipeline")
        .map((deployment) => deployment.model),
    );
    return scheduler
      .listAvailableModels({ connectedWorkerIds })
      .map((model) => model.id)
      .filter((modelId) => realDeploymentModels.has(modelId));
  };
  const currentBenchmarkActivations = (): BenchmarkActivation[] =>
    coordinatorBenchmarkActivations(
      new Set(benchmarkableActiveModelIds()),
      store.listWorkers(),
      hub.connectedWorkerIds(),
      coordinatorBuildIdentity,
    );
  const startCoordinatorBenchmark = (
    model: CoordinatorBenchmarkModel,
    trigger: "automatic-model-start" | "manual",
    metadata: {
      label?: string;
      activation?: BenchmarkActivation;
    } = {},
  ): Promise<PersistedBenchmark> => runAndPersistCoordinatorSuite({
    cwd: benchmarkWorkspace,
    ...(benchmarkHistoryDirectory ? { historyDirectory: benchmarkHistoryDirectory } : {}),
    coordinatorUrl: `http://127.0.0.1:${config.port}`,
    model,
    inventory: (routedWorkerIds) => buildCoordinatorBenchmarkInventory(
      model.id,
      store.listWorkers(),
      hub.connectedWorkerIds(),
      routedWorkerIds,
    ),
    telemetrySnapshot: () => buildCoordinatorBenchmarkTelemetrySnapshot(
      store.listWorkers(),
      hub.connectedWorkerIds(),
    ),
    resolveWorkerId: (jobId) => store.getJob(jobId)?.workerId ?? null,
    ...(config.networkToken ? { networkToken: config.networkToken } : {}),
    buildIdentity: coordinatorBuildIdentity,
    ...(metadata.label ? { label: metadata.label } : {}),
    version: runtimeVersion,
    ...(metadata.activation ? { activation: metadata.activation } : {}),
    ...(metadata.activation
      ? {
          validateActivation: (
            activation: BenchmarkActivation,
            routedWorkerIds: ReadonlySet<string>,
          ) => {
            const current = currentBenchmarkActivations().find(
              (candidate) => candidate.activationId === activation.activationId,
            );
            if (!current) {
              throw new Error(
                `benchmark_activation_drifted:${activation.activationId}`,
              );
            }
            const currentParticipantIds = new Set(
              current.participants.map((participant) => participant.workerId),
            );
            for (const workerId of routedWorkerIds) {
              if (!currentParticipantIds.has(workerId)) {
                throw new Error(
                  `benchmark_routed_worker_drifted:${workerId}:${activation.activationId}`,
                );
              }
            }
          },
        }
      : {}),
    trigger,
  }).then((result) => {
    store.queueBenchmarkRun(result.run);
    void persistence?.flush();
    return result;
  });
  const drainAutomaticBenchmarkQueue = (): void => {
    if (benchmarkRunInFlight) return;
    const queued = automaticBenchmarkQueue.shift();
    if (!queued) return;
    const { model, activation } = queued;
    benchmarkRunInFlight = startCoordinatorBenchmark(
      model,
      "automatic-model-start",
      { activation },
    );
    void benchmarkRunInFlight
      .then(({ run }) => {
        app.log.info(
          { modelId: model.id, benchmarkRunId: run.runId, status: run.status },
          "automatic model-start benchmark saved",
        );
        if (run.status === "failed") {
          scheduleAutomaticBenchmarkRetry(queued);
        } else {
          automaticBenchmarkRetryAttempts.delete(activation.activationId);
        }
      })
      .catch((error: unknown) => {
        app.log.error(
          { modelId: model.id, error: error instanceof Error ? error.message : String(error) },
          "automatic model-start benchmark could not be saved",
        );
        scheduleAutomaticBenchmarkRetry(queued);
      })
      .finally(() => {
        queuedAutomaticBenchmarkActivations.delete(activation.activationId);
        benchmarkRunInFlight = null;
        drainAutomaticBenchmarkQueue();
      });
  };
  const queueAutomaticBenchmark = (activation: BenchmarkActivation): void => {
    if (queuedAutomaticBenchmarkActivations.has(activation.activationId)) return;
    const model = benchmarkTargetForModel(activation.modelId);
    if (model.digest !== activation.modelDigest) {
      app.log.warn(
        {
          modelId: activation.modelId,
          activationId: activation.activationId,
          observedDigest: model.digest,
          activationDigest: activation.modelDigest,
        },
        "automatic benchmark skipped because model identity changed before queueing",
      );
      return;
    }
    queuedAutomaticBenchmarkActivations.add(activation.activationId);
    automaticBenchmarkQueue.push({ model, activation });
    drainAutomaticBenchmarkQueue();
  };
  function scheduleAutomaticBenchmarkRetry(queued: QueuedAutomaticBenchmark): void {
    const attempts = automaticBenchmarkRetryAttempts.get(
      queued.activation.activationId,
    ) ?? 0;
    if (attempts >= 2) return;
    automaticBenchmarkRetryAttempts.set(queued.activation.activationId, attempts + 1);
    const delayMs = attempts === 0 ? 30_000 : 120_000;
    const timer = setTimeout(() => {
      automaticBenchmarkRetryTimers.delete(timer);
      const current = currentBenchmarkActivations().find(
        (activation) => activation.activationId === queued.activation.activationId,
      );
      if (current) queueAutomaticBenchmark(current);
    }, delayMs);
    timer.unref();
    automaticBenchmarkRetryTimers.add(timer);
  }
  const automaticRepairState = new Map<string, { attempts: number; nextAttemptAt: number }>();
  const automaticRepairInFlight = new Set<string>();
  const inferenceRouteFailures = new Map<string, { count: number; lastAt: number }>();
  const automaticActivationRetryDelaysMs = (
    options.automaticActivationRetryDelaysMs
    ?? DEFAULT_AUTOMATIC_ACTIVATION_RETRY_DELAYS_MS
  ).map((delay) => Math.max(0, Math.round(delay)));
  const automaticActivationRetryState = new Map<string, AutomaticActivationRetryState>(
    deploymentController.listStates()
      .filter((state) => state.nextRetryAt !== null && state.lastError !== null)
      .map((state) => [
        state.modelId,
        {
          retryCount: Math.max(0, state.retryCount - 1),
          nextAttemptAt: state.nextRetryAt!,
          lastError: state.lastError!,
          updatedAt: state.updatedAt,
          launching: false,
        },
      ]),
  );
  const automaticActivationRetryProgressForModel = (
    modelId: string,
  ): readonly ModelActivationProgressEvent[] => {
    const retry = automaticActivationRetryState.get(modelId);
    if (!retry) return [];
    const incident = classifyActivationIncident({
      message: retry.lastError,
      retryCount: retry.retryCount,
      retryLaunching: retry.launching,
      nextRetryAt: retry.nextAttemptAt,
      maximumAttempts: automaticActivationRetryDelaysMs.length,
    });
    const retryNumber = retry.launching ? retry.retryCount : retry.retryCount + 1;
    const secondsRemaining = Math.max(0, Math.ceil((retry.nextAttemptAt - Date.now()) / 1_000));
    const waitMessage = secondsRemaining > 0
      ? `Automatic retry ${retryNumber} of ${automaticActivationRetryDelaysMs.length} starts in ${secondsRemaining}s.`
      : `Automatic retry ${retryNumber} of ${automaticActivationRetryDelaysMs.length} is ready and waiting for healthy capacity and a free activation slot.`;
    return [{
      phase: retry.launching ? "retrying" : "retry_wait",
      message: retry.launching
        ? `Automatic retry ${retryNumber} of ${automaticActivationRetryDelaysMs.length} is starting.`
        : `${incident.title}. ${waitMessage}`,
      at: new Date(retry.updatedAt).toISOString(),
      state: "running",
      details: [retry.lastError],
    }];
  };
  const activationProgressForModel = (
    modelId: string,
  ): readonly ModelActivationProgressEvent[] => {
    const managerProgress = activationManager?.activationProgressForModel?.(modelId) ?? [];
    const retryProgress = automaticActivationRetryProgressForModel(modelId);
    if (retryProgress.length === 0) return managerProgress;
    return [...managerProgress, ...retryProgress]
      .map((event, index) => ({ event, index, at: Date.parse(event.at) }))
      .sort((left, right) => {
        const leftAt = Number.isFinite(left.at) ? left.at : Number.MAX_SAFE_INTEGER;
        const rightAt = Number.isFinite(right.at) ? right.at : Number.MAX_SAFE_INTEGER;
        return leftAt - rightAt || left.index - right.index;
      })
      .map(({ event }) => event);
  };
  const activationStatusMessageForModel = (modelId: string): string | null => {
    const retry = automaticActivationRetryState.get(modelId);
    if (!retry) return null;
    const incident = classifyActivationIncident({
      message: retry.lastError,
      retryCount: retry.retryCount,
      retryLaunching: retry.launching,
      nextRetryAt: retry.nextAttemptAt,
      maximumAttempts: automaticActivationRetryDelaysMs.length,
    });
    const retryNumber = retry.launching ? retry.retryCount : retry.retryCount + 1;
    if (retry.launching) {
      return `Automatic retry ${retryNumber} of ${automaticActivationRetryDelaysMs.length} is starting.`;
    }
    const secondsRemaining = Math.max(0, Math.ceil((retry.nextAttemptAt - Date.now()) / 1_000));
    return secondsRemaining > 0
      ? `${incident.title}. Automatic retry ${retryNumber} of ${automaticActivationRetryDelaysMs.length} starts in ${secondsRemaining}s.`
      : `Automatic retry ${retryNumber} of ${automaticActivationRetryDelaysMs.length} is ready and waiting for healthy capacity and a free activation slot.`;
  };
  const activationIncidentForModel = (
    modelId: string,
  ): ActivationIncident | null => {
    const retry = automaticActivationRetryState.get(modelId);
    const stored = store.getRequestedModel(modelId);
    const message = retry?.lastError ?? stored?.activationError;
    if (!message) return null;
    return classifyActivationIncident({
      message,
      ...(retry
        ? {
            retryCount: retry.retryCount,
            retryLaunching: retry.launching,
          }
        : {}),
      nextRetryAt: retry?.nextAttemptAt ?? null,
      maximumAttempts: automaticActivationRetryDelaysMs.length,
    });
  };
  const handleRequestedModelActivationFailure = (
    modelId: string,
    error: unknown,
    operation: DeploymentOperation | null = deploymentController.activeOperationForModel(modelId),
  ): void => {
    if (!store.getRequestedModel(modelId)) return;
    const message = error instanceof Error ? error.message : String(error);
    if (!automaticActivationFailureIsTransient(message)) {
      automaticActivationRetryState.delete(modelId);
      store.setRequestedModelActivationError(modelId, message);
      if (operation) {
        deploymentController.failOperation(
          operation.id,
          "activation_failed",
          message,
          { retryAt: null },
        );
      }
      return;
    }
    const previous = automaticActivationRetryState.get(modelId);
    const retriesStarted = previous?.launching
      ? previous.retryCount
      : previous?.retryCount ?? 0;
    const nextRetry = nextAutomaticActivationRetry(
      retriesStarted,
      message,
      Date.now(),
      automaticActivationRetryDelaysMs,
    );
    if (!nextRetry) {
      automaticActivationRetryState.delete(modelId);
      const exhausted = `automatic_activation_retries_exhausted:${retriesStarted}:${message}`;
      store.setRequestedModelActivationError(modelId, exhausted);
      if (operation) {
        deploymentController.failOperation(
          operation.id,
          "activation_retries_exhausted",
          exhausted,
          { retryAt: null },
        );
      }
      return;
    }
    automaticActivationRetryState.set(modelId, nextRetry);
    store.setRequestedModelActivation(modelId, false);
    if (operation) {
      deploymentController.failOperation(
        operation.id,
        "transient_activation_failure",
        message,
        { retryAt: nextRetry.nextAttemptAt },
      );
    }
  };
  const launchRequestedModel = (model: StoredRequestedModel): boolean => {
    if (!activationManager || activationManager.isManaging(model.id) || activationManager.isBusy()) {
      return false;
    }
    deploymentController.ensureModel(model);
    deploymentController.setDesiredState(
      model.id,
      model.autoActivate ? "active" : "inactive",
    );
    const state = deploymentController.getState(model.id);
    const operation = deploymentController.claimOperation(
      model.id,
      state?.observedState === "active" || state?.observedState === "degraded"
        ? "repair"
        : "activate",
      { leaseMs: 90_000 },
    );
    if (!operation) return false;
    try {
      void activationManager.activate(model)
        .then(() => {
          if (deploymentController.getState(model.id)?.observedState === "active") {
            automaticActivationRetryState.delete(model.id);
          }
        })
        .catch((error: unknown) => {
          handleRequestedModelActivationFailure(model.id, error, operation);
        });
      return true;
    } catch (error) {
      handleRequestedModelActivationFailure(model.id, error, operation);
      return true;
    }
  };
  const reconcileAutomaticGpuRepair = (
    model: StoredRequestedModel,
    activeModelIds: ReadonlySet<string>,
    workers: readonly StoredWorker[],
    connectedWorkerIds: ReadonlySet<string>,
  ) => {
    if (!activationManager || !model.autoActivate || !activeModelIds.has(model.id)) return;
    const degraded = modelHasGpuFallback(model.id, workers, connectedWorkerIds);
    if (!degraded) {
      automaticRepairState.delete(model.id);
      return;
    }
    if (
      automaticRepairInFlight.has(model.id)
      || !activationManager.isManaging(model.id)
      || !verifiedGpuCapacityCanRepairModel(model, workers, connectedWorkerIds)
    ) return;
    const now = Date.now();
    const previous = automaticRepairState.get(model.id) ?? { attempts: 0, nextAttemptAt: 0 };
    if (now < previous.nextAttemptAt) return;
    const repairDelays = [30_000, 120_000, 300_000] as const;
    const delay = repairDelays[Math.min(previous.attempts, repairDelays.length - 1)]!;
    automaticRepairState.set(model.id, {
      attempts: previous.attempts + 1,
      nextAttemptAt: now + delay,
    });
    automaticRepairInFlight.add(model.id);
    void (async () => {
      // Keep the CPU fallback online until verified GPU capacity is ready, then
      // recycle the managed topology. The normal activation path runs a fresh
      // health check and real inference canary before publishing it again.
      const stopped = await activationManager.deactivate(model.id);
      if (!stopped) return;
      deploymentController.releaseRoutesForModel(model.id);
      deploymentController.adoptObservedState(model.id, "degraded");
      const latest = store.getRequestedModel(model.id);
      if (!latest?.autoActivate) return;
      store.setRequestedModelActivation(model.id, true);
      launchRequestedModel(store.getRequestedModel(model.id)!);
    })().catch((error: unknown) => {
      app.log.warn({ modelId: model.id, error }, "automatic GPU repair could not be started");
    }).finally(() => {
      automaticRepairInFlight.delete(model.id);
    });
  };
  const startInferenceRouteRepair = (modelId: string, reason: string): void => {
    const requested = store.getRequestedModel(modelId);
    if (
      !activationManager ||
      !requested?.autoActivate ||
      automaticRepairInFlight.has(modelId) ||
      !activationManager.isManaging(modelId)
    ) return;
    automaticRepairInFlight.add(modelId);
    void (async () => {
      app.log.warn({ modelId, reason }, "rebuilding an unresponsive distributed model route");
      const stopped = await activationManager.deactivate(modelId);
      if (!stopped) return;
      deploymentController.releaseRoutesForModel(modelId);
      deploymentController.adoptObservedState(modelId, "degraded");
      const latest = store.getRequestedModel(modelId);
      if (!latest?.autoActivate) return;
      store.setRequestedModelActivation(modelId, true);
      launchRequestedModel(store.getRequestedModel(modelId)!);
    })().catch((error: unknown) => {
      app.log.warn(
        { modelId, reason, error: error instanceof Error ? error.message : String(error) },
        "automatic inference route repair could not be started",
      );
    }).finally(() => {
      automaticRepairInFlight.delete(modelId);
    });
  };
  const modelsDependingOnExecutor = (workerId: string): string[] => {
    const nodeId = store.getWorker(workerId)?.capabilities.distributedExecutor?.nodeId;
    if (!nodeId) return [];
    return [...new Set(
      store.listWorkers()
        .filter((worker) => worker.identityKind === "cell")
        .flatMap((worker) => worker.capabilities.deployments)
        .filter((deployment) => deployment.execution?.stages?.some(
          (stage) => stage.nodeId === nodeId,
        ))
        .map((deployment) => deployment.model),
    )];
  };
  const reconcileRequestedModels = () => {
    const workers = store.listWorkers();
    const connectedWorkerIds = hub.connectedWorkerIds();
    const now = Date.now();
    const activeModelIds = new Set(
      scheduler
        .listAvailableModels({ connectedWorkerIds })
        .map((model) => model.id),
    );
    const benchmarkableRoutes = new Set(benchmarkableActiveModelIds());
    const benchmarkableModelIds = new Set(
      [...activeModelIds].filter((modelId) => benchmarkableRoutes.has(modelId)),
    );
    const activations = coordinatorBenchmarkActivations(
      benchmarkableModelIds,
      workers,
      connectedWorkerIds,
    );
    for (const activation of benchmarkActivationTracker.observe(activations, now)) {
      queueAutomaticBenchmark(activation);
    }
    let requests = store.listRequestedModels();
    for (const request of requests) {
      deploymentController.ensureModel(request, now);
      deploymentController.setDesiredState(
        request.id,
        request.autoActivate ? "active" : "inactive",
        now,
      );
      const operation = deploymentController.activeOperationForModel(request.id);
      if (operation && activationManager?.isManaging(request.id)) {
        deploymentController.renewOperation(operation.id, 90_000, now);
      }
    }
    for (const request of requests) {
      if (
        request.autoActivate
        && request.activationError
        && automaticActivationFailureIsTransient(request.activationError)
      ) {
        if (!automaticActivationRetryState.has(request.id)) {
          const retry = nextAutomaticActivationRetry(
            0,
            request.activationError,
            now,
            automaticActivationRetryDelaysMs,
          );
          if (retry) automaticActivationRetryState.set(request.id, retry);
        }
        store.clearRequestedModelActivationError(request.id);
      }
    }
    requests = store.listRequestedModels();
    const views = requestedModelCapacityViews({
      requests,
      workers,
      connectedWorkerIds,
      activeModelIds,
      ...(activationManager
        ? { executionNodesForModel: (modelId: string) => activationManager.capacityNodesForModel(modelId) }
        : {}),
      ...(activationManager
        ? {
            activationProgressForModel,
            activationStatusMessageForModel,
          }
        : {}),
      activationIncidentForModel,
      activationAvailable: activationManager !== undefined,
    });
    for (const view of views) {
      const stored = requests.find((request) => request.id === view.id)!;
      if (view.status === "active") {
        automaticActivationRetryState.delete(view.id);
        if (stored.activationError) store.clearRequestedModelActivationError(view.id);
        const operation = deploymentController.activeOperationForModel(view.id);
        if (operation) {
          deploymentController.completeOperation(
            operation.id,
            "active",
            { publication: "scheduler-observed-active" },
            now,
          );
        } else {
          deploymentController.adoptObservedState(view.id, "active", now);
        }
        deploymentController.renewCommittedRoutesForModel(view.id, 90_000, now);
      } else if (
        view.status === "waiting_capacity"
        && !deploymentController.activeOperationForModel(view.id)
      ) {
        deploymentController.markWaitingCapacity(view.id, view.message, now);
      }
      if (shouldQueueAutomaticActivation(view)) {
        const retry = automaticActivationRetryState.get(view.id);
        if (retry) {
          if (
            retry.launching
            || Date.now() < retry.nextAttemptAt
            || !activationManager
            || activationManager.isManaging(view.id)
            || activationManager.isBusy()
          ) {
            continue;
          }
          const launchingRetry: AutomaticActivationRetryState = {
            ...retry,
            retryCount: retry.retryCount + 1,
            nextAttemptAt: Date.now(),
            updatedAt: Date.now(),
            launching: true,
          };
          automaticActivationRetryState.set(view.id, launchingRetry);
          store.setRequestedModelActivation(view.id, true);
          if (!launchRequestedModel(store.getRequestedModel(view.id)!)) {
            automaticActivationRetryState.set(view.id, retry);
            store.setRequestedModelActivation(view.id, false);
          }
          continue;
        }
        store.setRequestedModelActivation(view.id, true);
        launchRequestedModel(store.getRequestedModel(view.id)!);
      } else if (
        view.status === "activating" &&
        stored.activationRequestedAt !== null &&
        activationManager &&
        !activationManager.isManaging(view.id) &&
        !activationManager.isBusy()
      ) {
        launchRequestedModel(stored);
      } else if (
        stored.activationRequestedAt !== null &&
        (view.status === "waiting_capacity" || view.status === "incompatible" || view.status === "failed")
      ) {
        store.setRequestedModelActivation(view.id, false);
      }
    }
    for (const model of requests) {
      reconcileAutomaticGpuRepair(model, activeModelIds, workers, connectedWorkerIds);
    }
  };
  hub.on("envelope", (envelope) => {
    if (envelope.type === "contribution.ack") {
      const acknowledgement = contributionAckEnvelopeSchema.parse(envelope);
      fleetContribution.acknowledge(
        acknowledgement.workerId,
        acknowledgement.payload,
      );
    }
    if (envelope.type === "worker.heartbeat") queueMicrotask(reconcileRequestedModels);
  });
  hub.on("disconnect", (workerId) => {
    for (const modelId of modelsDependingOnExecutor(workerId)) {
      startInferenceRouteRepair(modelId, `pipeline_stage_disconnected:${workerId}`);
    }
    queueMicrotask(reconcileRequestedModels);
  });
  service.on("healthy", ({ model }) => {
    inferenceRouteFailures.delete(model);
  });
  service.on("degraded", ({ model, code, jobId, workerId }) => {
    const now = Date.now();
    const previous = inferenceRouteFailures.get(model);
    const count = previous && now - previous.lastAt <= 5 * 60_000
      ? previous.count + 1
      : 1;
    inferenceRouteFailures.set(model, { count, lastAt: now });
    app.log.warn(
      { modelId: model, code, jobId, workerId, consecutiveFailures: count },
      "distributed inference route degraded before its first token",
    );
    const routeWorker = workerId ? store.getWorker(workerId) : null;
    const definitelyBrokenCellRoute =
      code === "adapter_error" && routeWorker?.identityKind === "cell";
    if (definitelyBrokenCellRoute || count >= 2) {
      inferenceRouteFailures.delete(model);
      startInferenceRouteRepair(model, `${code}:${jobId}`);
    }
  });
  const staleTimer = setInterval(() => {
    store.markStaleWorkers();
    hub.closeStaleConnections();
    mobileHub.expireDisconnectedWorkers();
    void activationManager?.refresh();
    reconcileRequestedModels();
  }, 5_000);
  staleTimer.unref();
  const captureNetworkTelemetry = (now = Date.now()) => {
    reconcileRequestedModels();
    const snapshot = publicSnapshot(store, scheduler, hub, mobileHub, activationManager, {
      activationProgressForModel,
      activationStatusMessageForModel,
      activationIncidentForModel,
    }, runtimeVersion, coordinatorBuildIdentity);
    const capturedAt = Math.floor(now / NETWORK_TELEMETRY_INTERVAL_MS)
      * NETWORK_TELEMETRY_INTERVAL_MS;
    store.recordNetworkTelemetrySample(networkTelemetrySample(snapshot, capturedAt));
    store.pruneNetworkTelemetrySamples(
      now - NETWORK_TELEMETRY_RETENTION_DAYS * 24 * 60 * 60_000,
    );
    return snapshot;
  };
  captureNetworkTelemetry();
  const networkTelemetryTimer = setInterval(
    () => captureNetworkTelemetry(),
    NETWORK_TELEMETRY_INTERVAL_MS,
  );
  networkTelemetryTimer.unref();

  app.get("/health", async () => {
    const workers = store.listWorkers();
    const mobileWorkers = mobileHub.listWorkers();
    return {
      status: "ok",
      version: runtimeVersion,
      revision: runtimeRevision,
      buildIdentity: coordinatorBuildIdentity,
      workers: {
        registered: workers.length + mobileWorkers.length,
        connected: hub.connectedWorkerIds().size + mobileHub.connectedCount(),
        online:
          workers.filter((worker) => worker.status === "online").length +
          mobileHub.onlineCount(),
        mobile: mobileWorkers.length,
      },
      mobilePwa: mobileAssetsPath ? "/mobile/" : null,
      landing: config.landingAssetsPath ? "/" : null,
      desktopUpdates: desktopUpdatesPath ? "/updates/win32/x64/" : null,
      downloads: releaseDownloadsPath ? "/downloads/" : null,
      features: {
        distributedActivation: activationManager !== undefined,
        apiAccess: apiAccessEnabled,
      },
      persistence: persistence?.status() ?? {
        configured: false,
        connected: false,
        required: false,
        pendingChanges: database.pendingRemoteChangeCount(),
        pendingArtifacts: database.pendingArtifactBackupCount(),
        lastSuccessfulSyncAt: null,
        lastError: null,
      },
    };
  });

  app.get("/public/v1/auth-config", async () => ({
    enabled: Boolean(config.supabaseUrl && config.supabaseAnonKey),
    apiAccessEnabled,
    publicApiBaseUrl: config.publicApiBaseUrl ?? "/v1",
    starterTokens: apiAccess.limits.starterTokens,
    limits: {
      requestsPerMinute: apiAccess.limits.requestsPerMinute,
      maxConcurrent: apiAccess.limits.maxConcurrent,
      maxActiveKeys: apiAccess.limits.maxActiveKeys,
    },
    ...(config.supabaseUrl && config.supabaseAnonKey
      ? { url: config.supabaseUrl, anonKey: config.supabaseAnonKey }
      : {}),
  }));

  app.get("/v1/auth/me", async (request, reply) => {
    const token = parseBearerToken(request.headers.authorization);
    if (!token || !supabaseAuth) {
      return reply.code(401).send({ error: { code: "authentication_required" } });
    }
    const user = await supabaseAuth.authenticate(token);
    if (!user) return reply.code(401).send({ error: { code: "invalid_access_token" } });
    const account = apiAccess.getOrCreateAccount(user.id);
    return { user, account: apiAccountJson(account, apiAccess.limits) };
  });

  app.get("/v1/account", async (request, reply) => {
    const principal = principalFor(request);
    if (principal.kind === "system") {
      return reply.code(403).send({
        error: {
          code: "account_identity_required",
          message: "Use an account session or account API key to inspect a balance.",
        },
      });
    }
    return { object: "account", ...apiAccountJson(
      apiAccess.getOrCreateAccount(principal.userId),
      apiAccess.limits,
    ) };
  });

  app.get("/v1/account/usage", async (request) => {
    const principal = principalFor(request);
    if (principal.kind === "system") return { object: "list", data: [] };
    const query = z.object({
      limit: z.coerce.number().int().min(1).max(100).default(25),
    }).parse(request.query);
    return {
      object: "list",
      data: apiAccess.listUsage(principal.userId, query.limit).map(apiUsageJson),
    };
  });

  app.get("/v1/api-keys", async (request, reply) => {
    const principal = principalFor(request);
    if (principal.kind !== "user") {
      return reply.code(403).send({
        error: {
          code: "account_session_required",
          message: "API keys can only be managed from a signed-in account session.",
        },
      });
    }
    return {
      object: "list",
      data: apiAccess.listKeys(principal.userId).map(apiKeyJson),
    };
  });

  app.post("/v1/api-keys", async (request, reply) => {
    const principal = principalFor(request);
    if (principal.kind !== "user") {
      return reply.code(403).send({
        error: {
          code: "account_session_required",
          message: "API keys can only be managed from a signed-in account session.",
        },
      });
    }
    const body = z.object({
      name: z.string().trim().min(1).max(80),
    }).parse(request.body);
    const created = apiAccess.createKey(principal.userId, body.name);
    return reply.code(201).send({
      object: "api_key",
      ...apiKeyJson(created),
      secret: created.secret,
    });
  });

  app.delete("/v1/api-keys/:keyId", async (request, reply) => {
    const principal = principalFor(request);
    if (principal.kind !== "user") {
      return reply.code(403).send({
        error: {
          code: "account_session_required",
          message: "API keys can only be managed from a signed-in account session.",
        },
      });
    }
    const { keyId } = z.object({ keyId: z.string().min(1).max(100) }).parse(request.params);
    if (!apiAccess.revokeKey(principal.userId, keyId)) {
      return reply.code(404).send({ error: { code: "api_key_not_found" } });
    }
    return reply.code(204).send();
  });

  app.post("/v1/admin/accounts/:userId/tokens", async (request, reply) => {
    const principal = principalFor(request);
    if (
      principal.kind !== "user"
      || (principal.role !== "owner" && principal.role !== "admin")
    ) {
      return reply.code(403).send({
        error: {
          code: "insufficient_network_role",
          message: "Only a network owner or administrator can grant account tokens.",
        },
      });
    }
    const { userId } = z.object({ userId: z.string().uuid() }).parse(request.params);
    const { amount } = z.object({
      amount: z.number().int().positive().max(10_000_000),
    }).parse(request.body);
    return {
      object: "account",
      ...apiAccountJson(apiAccess.grantTokens(userId, amount), apiAccess.limits),
    };
  });

  app.get("/public/v1/snapshot", async () => {
    reconcileRequestedModels();
    return publicSnapshot(store, scheduler, hub, mobileHub, activationManager, {
      activationProgressForModel,
      activationStatusMessageForModel,
      activationIncidentForModel,
    }, runtimeVersion, coordinatorBuildIdentity);
  });

  app.post("/internal/v1/diagnostics", async (request, reply) => {
    const parsed = remoteDiagnosticBatchSchema.safeParse(request.body);
    if (!parsed.success) {
      return reply.code(400).send({
        error: {
          code: "invalid_request",
          message: "Request validation failed",
          details: parsed.error.issues,
        },
      });
    }
    const body = parsed.data;
    const events = body.events.map((event) => ({
      ...event,
      message: redactDiagnosticText(event.message),
      ...(event.details
        ? { details: redactDiagnosticDetails(event.details) }
        : {}),
    }));
    const result = store.appendDiagnosticEvents(events);
    void persistence?.flush();
    return reply.code(result.accepted > 0 ? 201 : 200).send(result);
  });

  app.get("/public/v1/admin/diagnostics", async (request, reply) => {
    if (!await authorizeAdministrativeMutation(request, reply, ["owner", "admin"])) return;
    const parsed = remoteDiagnosticQuerySchema.safeParse(request.query);
    if (!parsed.success) {
      return reply.code(400).send({
        error: {
          code: "invalid_request",
          message: "Request validation failed",
          details: parsed.error.issues,
        },
      });
    }
    const query = parsed.data;
    return {
      capturedAt: new Date().toISOString(),
      events: store.listDiagnosticEvents({
        limit: query.limit,
        ...(query.level ? { level: query.level } : {}),
        ...(query.sourceId ? { sourceId: query.sourceId } : {}),
        ...(query.since ? { since: query.since } : {}),
      }),
    };
  });

  const availableSupportAssistantModels = (): string[] =>
    benchmarkableActiveModelIds().sort((left, right) => left.localeCompare(right));

  const publicSupportAssistantConfig = () => {
    const settings = store.getSupportAssistantSettings();
    const availableModels = availableSupportAssistantModels();
    const selectedModel = resolveSupportAssistantModel(settings, availableModels);
    return {
      enabled: settings.enabled,
      available: settings.enabled && selectedModel !== null,
      provider: "mycellios-network" as const,
      configuredModel: settings.modelId,
      selectedModel,
      availableModels,
      welcomeMessage: settings.welcomeMessage,
      suggestions: settings.suggestions,
      allowDeviceControl: settings.allowDeviceControl,
      updatedAt: settings.updatedAt ? new Date(settings.updatedAt).toISOString() : null,
    };
  };

  app.get("/public/v1/assistant/config", async () => publicSupportAssistantConfig());

  app.get("/public/v1/admin/fleet-contribution", async (request, reply) => {
    if (!await authorizeAdministrativeMutation(request, reply)) return;
    return fleetContribution.status();
  });

  app.post("/public/v1/admin/fleet-contribution", async (request, reply) => {
    if (!await authorizeAdministrativeMutation(request, reply)) return;
    const body = z.object({ enabled: z.boolean() }).strict().parse(request.body);
    return fleetContribution.setAll(body.enabled);
  });

  app.get("/public/v1/admin/assistant", async (request, reply) => {
    if (!await authorizeAdministrativeMutation(request, reply)) return;
    return {
      settings: store.getSupportAssistantSettings(),
      runtime: publicSupportAssistantConfig(),
    };
  });

  app.put("/public/v1/admin/assistant", async (request, reply) => {
    if (!await authorizeAdministrativeMutation(request, reply)) return;
    const body = supportAssistantSettingsUpdateSchema.parse(request.body);
    const availableModels = availableSupportAssistantModels();
    if (body.modelId && !availableModels.includes(body.modelId)) {
      return reply.code(409).send({
        error: {
          code: "assistant_model_not_available",
          message: `${body.modelId} is not currently announced by a real mycellios inference node.`,
        },
      });
    }
    const settings = store.saveSupportAssistantSettings(body);
    return {
      settings,
      runtime: publicSupportAssistantConfig(),
    };
  });

  app.post("/public/v1/assistant/chat", async (request, reply) => {
    const body = supportAssistantChatRequestSchema.parse(request.body);
    if (activeSupportAssistantRequests >= 8) {
      return reply.code(429).send({
        error: {
          code: "assistant_capacity_limited",
          message: "All support assistant slots are busy. Try again shortly.",
        },
      });
    }
    const releaseRateLimit = claimSupportAssistantRequest(
      supportAssistantRateLimits,
      supportAssistantRateKey(request, body.session_id),
    );
    if (!releaseRateLimit) {
      return reply.code(429).send({
        error: {
          code: "assistant_rate_limited",
          message: "The mycellios assistant is already handling too many requests. Try again shortly.",
        },
      });
    }
    activeSupportAssistantRequests += 1;
    try {
      const settings = store.getSupportAssistantSettings();
      if (!settings.enabled) {
        return reply.code(503).send({
          error: {
            code: "assistant_disabled",
            message: "The mycellios assistant is disabled by the network administrator.",
          },
        });
      }
      const availableModels = availableSupportAssistantModels();
      const selectedModel = resolveSupportAssistantModel(settings, availableModels);
      if (!selectedModel) {
        return reply.code(503).send({
          error: {
            code: "assistant_model_unavailable",
            message: settings.modelId
              ? `The configured network model ${settings.modelId} is not connected right now.`
              : "No real mycellios network model is available for support right now.",
          },
        });
      }

      const snapshot = publicSnapshot(store, scheduler, hub, mobileHub, activationManager, {
        activationProgressForModel,
        activationStatusMessageForModel,
        activationIncidentForModel,
      }, runtimeVersion, coordinatorBuildIdentity);
      const parsed: ChatCompletionRequest = {
        model: selectedModel,
        messages: buildSupportAssistantMessages(
          settings,
          {
            registeredNodes: snapshot.summary.registered,
            connectedNodes: snapshot.summary.connected,
            offeredMemoryGb: snapshot.summary.offeredVramMb / 1_024,
            availableModels,
            requestedModels: snapshot.requestedModels.map((model) => ({
              id: model.id,
              status: model.status,
            })),
          },
          body.messages,
          { ...(body.page ? { page: body.page } : {}), ...(body.platform ? { platform: body.platform } : {}) },
        ),
        stream: true,
        max_tokens: settings.maxOutputTokens,
        temperature: settings.temperature,
        top_p: 1,
        workload_class: "interactive",
        session_id: `support-${body.session_id}`,
        deadline_ms: Math.min(config.requestTimeoutMs, 180_000),
      };

      const supersededJobId = service.cancelMatchingActiveSession(parsed, parsed.session_id);
      if (supersededJobId) {
        await new Promise<void>((resolveDelay) => setTimeout(resolveDelay, 1_000));
      }
      if (!service.hasCapacity(parsed, parsed.session_id)) {
        await waitForChatCapacity(service, parsed, parsed.session_id, 30_000);
      }
      const handle = service.submit(parsed, parsed.session_id);
      const conversationId = store.startInferenceConversation(
        handle.sessionId,
        parsed.model,
        parsed.messages,
      );
      reply.header("x-network-request-id", handle.jobId);
      reply.header("x-network-session-id", handle.sessionId);
      reply.header("x-mycellios-assistant-provider", "mycellios-network");
      reply.header("x-mycellios-assistant-model", selectedModel);
      reply.hijack();
      reply.raw.writeHead(200, {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache, no-transform",
        connection: "keep-alive",
        "x-accel-buffering": "no",
        "x-network-request-id": handle.jobId,
        "x-network-session-id": handle.sessionId,
        "x-mycellios-assistant-provider": "mycellios-network",
        "x-mycellios-assistant-model": selectedModel,
      });

      let finished = false;
      let streamedText = "";
      let streamedRouteClass = "replica";
      let streamedResult: Extract<JobStreamEvent, { type: "completed" }> | null = null;
      let streamedFailure: Extract<JobStreamEvent, { type: "failed" }> | null = null;
      reply.raw.once("close", () => {
        if (!finished) service.cancel(handle.jobId);
      });
      const heartbeatTimer = setInterval(() => {
        if (!reply.raw.destroyed && !reply.raw.writableEnded) {
          reply.raw.write(`: mycellios-assistant-heartbeat ${Date.now()}\n\n`);
        }
      }, 10_000);
      heartbeatTimer.unref();
      try {
        for await (const event of handle.events) {
          if (event.type === "accepted") streamedRouteClass = event.route.routeClass;
          if (event.type === "token") streamedText += event.token.text;
          if (event.type === "completed") streamedResult = event;
          if (event.type === "failed") streamedFailure = event;
          writeMycelliosEvent(reply.raw, event, parsed.model, handle.jobId);
        }
      } finally {
        clearInterval(heartbeatTimer);
      }
      finished = true;
      store.appendInferenceMessage({
        conversationId,
        jobId: handle.jobId,
        role: "assistant",
        content: streamedResult?.result.text ?? streamedText,
        status: streamedFailure ? "failed" : "completed",
        inputTokens: streamedResult?.result.metrics.inputTokens ?? null,
        outputTokens: streamedResult?.result.metrics.outputTokens ?? null,
        routeClass: streamedRouteClass,
        latencyMs: streamedResult?.result.metrics.activeMs ?? null,
        metadata: streamedFailure
          ? { failure_code: streamedFailure.code, failure_message: streamedFailure.message, assistant: true }
          : {
              finish_reason: streamedResult?.result.finishReason ?? null,
              assistant: true,
              provider: "mycellios-network",
            },
      });
      void persistence?.flush();
      reply.raw.write("data: [DONE]\n\n");
      reply.raw.end();
      return;
    } finally {
      activeSupportAssistantRequests = Math.max(0, activeSupportAssistantRequests - 1);
      releaseRateLimit();
    }
  });

  app.get("/public/v1/huggingface-models", async (request, reply) => {
    const { q, cursor, sort, limit } = huggingFaceModelSearchSchema.parse(request.query);
    if (activePublicCatalogRequests >= 8) {
      return reply.code(429).send({
        error: {
          code: "catalog_capacity_limited",
          message: "The public model catalog is busy. Try again shortly.",
        },
      });
    }
    const releaseRateLimit = claimPublicCatalogRequest(
      publicCatalogRateLimits,
      publicCatalogRateKey(request),
    );
    if (!releaseRateLimit) {
      return reply.code(429).send({
        error: {
          code: "catalog_rate_limited",
          message: "Too many model catalog requests. Try again shortly.",
        },
      });
    }
    activePublicCatalogRequests += 1;
    try {
      return await searchHubModelCatalog(q, fetch, { ...(cursor ? { cursor } : {}), sort, limit });
    } catch (error) {
      return reply.code(502).send({
        error: {
          code: "huggingface_catalog_unavailable",
          message: error instanceof Error ? error.message : String(error),
        },
      });
    } finally {
      activePublicCatalogRequests = Math.max(0, activePublicCatalogRequests - 1);
      releaseRateLimit();
    }
  });

  app.post("/public/v1/requested-models", async (request, reply) => {
    if (!await authorizeAdministrativeMutation(request, reply)) return;
    const body = requestedModelCreateSchema.parse(request.body);
    automaticActivationRetryState.delete(body.id);
    const stored = store.upsertRequestedModel({
      id: body.id,
      source: body.source,
      revision: body.revision,
      contextTokens: body.contextTokens,
      minimumNodes: body.minimumNodes,
      autoActivate: body.autoActivate,
    });
    deploymentController.ensureModel(stored);
    deploymentController.setDesiredState(
      stored.id,
      stored.autoActivate ? "active" : "inactive",
    );
    try {
      const profile = await inspectHubModelCapacity({
        source: stored.source,
        revision: stored.revision,
        contextTokens: stored.contextTokens,
        minimumNodes: stored.minimumNodes,
      });
      store.setRequestedModelProfile(stored.id, profile as unknown as Record<string, unknown>, null);
    } catch (error) {
      store.setRequestedModelProfile(
        stored.id,
        null,
        error instanceof Error ? error.message : String(error),
      );
    }
    reconcileRequestedModels();
    const snapshot = publicSnapshot(store, scheduler, hub, mobileHub, activationManager, {
      activationProgressForModel,
      activationStatusMessageForModel,
      activationIncidentForModel,
    }, runtimeVersion, coordinatorBuildIdentity);
    return reply.code(201).send({
      model: snapshot.requestedModels.find((model) => model.id === stored.id),
    });
  });

  app.delete("/public/v1/requested-models/:modelId", async (request, reply) => {
    if (!await authorizeAdministrativeMutation(request, reply)) return;
    const { modelId } = requestedModelParamsSchema.parse(request.params);
    if (!store.getRequestedModel(modelId)) {
      return reply.code(404).send({ error: { code: "requested_model_not_found" } });
    }
    deploymentController.setDesiredState(modelId, "inactive");
    const deactivation = deploymentController.claimOperation(modelId, "deactivate", {
      leaseMs: 60_000,
    });
    await activationManager?.deactivate(modelId);
    deploymentController.releaseRoutesForModel(modelId);
    if (deactivation) deploymentController.completeOperation(deactivation.id, "inactive");
    store.removeRequestedModel(modelId);
    automaticActivationRetryState.delete(modelId);
    automaticRepairState.delete(modelId);
    automaticRepairInFlight.delete(modelId);
    return { removed: true, modelId };
  });

  app.get("/public/v1/requested-models/:modelId/timeline", async (request, reply) => {
    const { modelId } = requestedModelParamsSchema.parse(request.params);
    const timeline = deploymentController.timeline(modelId);
    if (!timeline) {
      return reply.code(404).send({ error: { code: "requested_model_not_found" } });
    }
    return { data: timeline };
  });

  app.get("/internal/v1/model-activation-requests", async () => {
    reconcileRequestedModels();
    return {
      data: store.listRequestedModels()
        .filter((model) => model.activationRequestedAt !== null)
        .map((model) => ({
          id: model.id,
          source: model.source,
          revision: model.revision,
          contextTokens: model.contextTokens,
          minimumNodes: model.minimumNodes,
          profile: model.profile,
          requestedAt: new Date(model.activationRequestedAt!).toISOString(),
        })),
    };
  });

  const benchmarkHistoryResponse = async () => {
    const localRuns = loadBenchmarkRuns(benchmarkWorkspace, benchmarkHistoryDirectory);
    let remoteRuns: typeof localRuns = [];
    if (persistence) {
      try {
        remoteRuns = await persistence.readBenchmarkRuns();
      } catch (error) {
        app.log.warn(
          { error: error instanceof Error ? error.message : String(error) },
          "Supabase benchmark history could not be read; serving the local durable copy",
        );
      }
    }
    const byId = new Map([...remoteRuns, ...localRuns].map((run) => [run.runId, run]));
    return {
      runs: [...byId.values()]
        .sort((left, right) => right.finishedAt.localeCompare(left.finishedAt)),
    };
  };

  app.get("/public/v1/benchmarks", async () => benchmarkHistoryResponse());

  app.get("/local/v1/benchmarks", async (request, reply) => {
    if (!isTrustedLocalRequest(request)) {
      return reply.code(403).send({
        error: { code: "local_access_required", message: "Benchmark history is available on the coordinator host only." },
      });
    }
    return benchmarkHistoryResponse();
  });

  app.get("/public/v1/history", async (request, reply) => {
    reply.header("Access-Control-Allow-Origin", "*");
    const { range } = z.object({
      range: z.enum(["24h", "7d", "30d", "90d"]).default("24h"),
    }).parse(request.query);
    const now = Date.now();
    captureNetworkTelemetry(now);
    const since = now - networkHistoryRanges[range] * 60 * 60_000;
    return {
      capturedAt: new Date(now).toISOString(),
      intervalMinutes: NETWORK_TELEMETRY_INTERVAL_MS / 60_000,
      retentionDays: NETWORK_TELEMETRY_RETENTION_DAYS,
      range,
      samples: store.listNetworkTelemetrySamples(since).map((sample) => ({
        ...sample,
        capturedAt: new Date(sample.capturedAt).toISOString(),
      })),
    };
  });

  app.post("/local/v1/benchmarks/run", async (request, reply) => {
    if (!isTrustedLocalRequest(request)) {
      return reply.code(403).send({
        error: { code: "local_access_required", message: "Benchmarks can only be started on the coordinator host." },
      });
    }
    if (benchmarkRunInFlight) {
      return reply.code(409).send({
        error: { code: "benchmark_in_progress", message: "A real benchmark is already running." },
      });
    }
    const body = benchmarkRunRequestSchema.parse(request.body ?? {});
    if (body.version !== undefined && body.version !== runtimeVersion) {
      return reply.code(409).send({
        error: {
          code: "benchmark_version_mismatch",
          message:
            `La versión solicitada ${body.version} no coincide con el runtime real ${runtimeVersion}.`,
        },
      });
    }
    const activeModelIds = benchmarkableActiveModelIds();
    const modelId = body.model ?? activeModelIds[0];
    if (!modelId || !activeModelIds.includes(modelId)) {
      return reply.code(409).send({
        error: {
          code: "benchmark_model_unavailable",
          message: body.model
            ? `El modelo ${body.model} no está activo; no se ha generado ningún dato.`
            : "No hay ningún modelo activo para medir; no se ha generado ningún dato.",
        },
      });
    }
    const activation = currentBenchmarkActivations().find(
      (candidate) => candidate.modelId === modelId,
    );
    if (!activation) {
      return reply.code(409).send({
        error: {
          code: "benchmark_activation_unsealed",
          message:
            "El modelo está activo, pero su cohorte de build declarada no coincide de forma estable con el coordinador.",
        },
      });
    }
    benchmarkRunInFlight = startCoordinatorBenchmark(
      benchmarkTargetForModel(modelId),
      "manual",
      {
        ...(body.label ? { label: body.label } : {}),
        activation,
      },
    );
    try {
      const result = await benchmarkRunInFlight;
      return { run: result.run };
    } catch (error) {
      return reply.code(503).send({
        error: {
          code: "benchmark_unavailable",
          message: error instanceof Error ? error.message : String(error),
        },
      });
    } finally {
      benchmarkRunInFlight = null;
      drainAutomaticBenchmarkQueue();
    }
  });

  app.delete("/public/v1/workers/:workerId", async (request, reply) => {
    // Evicting a worker takes capacity out of the shared network, so it needs
    // the same authorization as any other mutation of shared state.
    if (!await authorizeAdministrativeMutation(request, reply)) return;
    const { workerId } = workerIdParamsSchema.parse(request.params);
    const removed = hub.removeWorker(workerId) || mobileHub.removeWorker(workerId);
    if (!removed) return reply.code(404).send({ error: { code: "worker_not_found" } });
    return { removed: true, workerId };
  });

  app.post("/public/v1/workers/clear-offline", async (request, reply) => {
    if (!await authorizeAdministrativeMutation(request, reply)) return;
    return {
      removed: store.deregisterOfflineWorkers() + mobileHub.removeOfflineWorkers(),
    };
  });

  app.post("/internal/v1/workers/admission-challenge", async (request, reply) => {
    const challenge = workerAdmissionChallengeRequestSchema.parse(request.body);
    try {
      return workerAdmission.issue(challenge);
    } catch (error) {
      if (!(error instanceof WorkerAdmissionError)) throw error;
      return reply.code(error.statusCode).send({
        error: { code: error.code, message: error.message },
      });
    }
  });

  app.post("/internal/v1/workers/credential-rotation-challenge", async (request, reply) => {
    const rotation = workerCredentialRotationChallengeRequestSchema.parse(request.body);
    try {
      return workerAdmission.issueRotation(rotation);
    } catch (error) {
      if (!(error instanceof WorkerAdmissionError)) throw error;
      return reply.code(error.statusCode).send({
        error: { code: error.code, message: error.message },
      });
    }
  });

  app.post("/internal/v1/workers/credential-rotation", async (request, reply) => {
    const rotation = z.object({
      identity: workerAdmissionIdentitySchema,
      proof: workerCredentialRotationProofSchema,
    }).strict().parse(request.body);
    try {
      return workerAdmission.verifyRotation(rotation);
    } catch (error) {
      if (!(error instanceof WorkerAdmissionError)) throw error;
      return reply.code(error.statusCode).send({
        error: { code: error.code, message: error.message },
      });
    }
  });

  app.post("/internal/v1/workers/register", async (request, reply) => {
    const registration = workerRegistrationSchema.parse(request.body);
    if (
      !config.allowDevelopmentAdapters
      && registration.capabilities.deployments.some(
        (deployment) => deployment.adapter === "mock",
      )
    ) {
      return reply.code(400).send({
        error: {
          code: "development_adapter_not_allowed",
          message: "Production coordinators accept native Mycellios deployments only.",
        },
      });
    }
    const protocol = registration.protocol ?? {
      min: WORKER_PROTOCOL_MIN,
      max: WORKER_PROTOCOL_MAX,
    };
    let admission:
      | {
        protocolVersion: number;
        credentialFingerprint?: string;
        enrollment: "enrolled" | "accepted" | "local-legacy";
      };
    if (registration.admission) {
      if (!registration.identity) {
        return reply.code(400).send({
          error: {
            code: "worker_admission_identity_required",
            message: "Signed worker admission requires a stable device identity.",
          },
        });
      }
      let verified;
      try {
        verified = workerAdmission.verify({
          identity: registration.identity,
          registrationDigest: workerRegistrationDigest({
            identity: registration.identity,
            capabilities: registration.capabilities,
            protocol,
          }),
          proof: registration.admission,
        });
      } catch (error) {
        if (!(error instanceof WorkerAdmissionError)) throw error;
        return reply.code(error.statusCode).send({
          error: { code: error.code, message: error.message },
        });
      }
      admission = {
        protocolVersion: verified.protocolVersion,
        credentialFingerprint: verified.credentialFingerprint,
        enrollment: verified.enrollment,
      };
    } else {
      if (!isTrustedLocalRequest(request)) {
        return reply.code(401).send({
          error: {
            code: "worker_admission_required",
            message: "Remote workers must prove a stable signed device identity.",
          },
        });
      }
      admission = {
        protocolVersion: selectWorkerProtocolVersion(protocol),
        enrollment: "local-legacy",
      };
    }
    const worker = store.registerWorker({
      ...(registration.identity ? { identity: registration.identity } : {}),
      capabilities: stripWorkerDeclaredEvidence(registration.capabilities),
    });
    const workerSessionToken = config.networkToken
      && registration.identity
      && admission.credentialFingerprint
      ? issueWorkerSessionToken(config.networkToken, {
          workerId: worker.id,
          identityKind: registration.identity.kind,
          identityId: registration.identity.id,
          credentialFingerprint: admission.credentialFingerprint,
        })
      : undefined;
    return reply.code(201).send({
      workerId: worker.id,
      protocolVersion: admission.protocolVersion,
      ...(admission.credentialFingerprint
        ? { credentialFingerprint: admission.credentialFingerprint }
        : {}),
      ...(workerSessionToken ? { workerSessionToken } : {}),
      enrollment: admission.enrollment,
    });
  });

  app.get("/internal/v1/workers", async () => ({
    data: [
      ...store.listWorkers().filter((worker) => storedWorkerIsVisible(worker, hub)).map((worker) => ({
      id: worker.id,
      status: worker.status,
      connected: hub.isConnected(worker.id),
      region: worker.capabilities.region,
      agentVersion: worker.capabilities.agentVersion,
      buildIdentity: worker.capabilities.buildIdentity,
      offeredVramMb: worker.capabilities.gpus.reduce(
        (sum, gpu) => sum + gpu.offeredVramMb,
        0,
      ),
      gpus: worker.capabilities.gpus,
      deployments: worker.capabilities.deployments,
      executionNodeId: worker.capabilities.distributedExecutor?.nodeId,
      computeMode: worker.capabilities.distributedExecutor?.computeMode,
      acceleration: worker.capabilities.distributedExecutor?.acceleration,
      isolation: worker.capabilities.distributedExecutor?.isolation,
      physicalIdentity: worker.capabilities.distributedExecutor?.physicalIdentity,
      reliability: worker.reliability,
      jobsCompleted: worker.jobsCompleted,
      lastSeenAt: new Date(worker.lastSeenAt).toISOString(),
      kind: storedWorkerKind(worker),
      })),
      ...mobileHub.listWorkers().map((worker) => ({
        ...mobileDashboardWorker(worker),
        kind: "browser" as const,
      })),
    ],
  }));

  app.get("/public/v1/worker-credentials", async (request, reply) => {
    if (!(await authorizeAdministrativeMutation(request, reply))) return;
    return {
      data: database.listWorkerAdmissionCredentials().map((credential) => ({
        ...admissionCredentialSummary(credential),
        createdAt: new Date(credential.createdAt).toISOString(),
        updatedAt: new Date(credential.updatedAt).toISOString(),
        lastSeenAt: new Date(credential.lastSeenAt).toISOString(),
        revokedAt: credential.revokedAt === null
          ? null
          : new Date(credential.revokedAt).toISOString(),
        revocationReason: credential.revocationReason,
      })),
    };
  });

  app.post(
    "/public/v1/worker-credentials/:identityKind/:identityId/revoke",
    async (request, reply) => {
      if (!(await authorizeAdministrativeMutation(request, reply))) return;
      const params = z.object({
        identityKind: z.enum(["device", "cell", "browser"]),
        identityId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/),
      }).strict().parse(request.params);
      const body = z.object({
        expectedFingerprint: z.string().regex(/^sha256:[a-f0-9]{64}$/),
        reason: z.string().trim().min(3).max(300),
      }).strict().parse(request.body);
      const result = database.revokeWorkerAdmissionCredential({
        ...params,
        ...body,
      });
      if (result.state === "not_found") {
        return reply.code(404).send({
          error: { code: "worker_credential_unknown" },
        });
      }
      if (result.state === "fingerprint_mismatch") {
        return reply.code(409).send({
          error: {
            code: "worker_credential_changed",
            message: "The worker credential changed before revocation was applied.",
          },
        });
      }
      if (!result.credential) throw new Error("worker_revocation_credential_missing");
      const revokedCredential = result.credential;
      let disconnected = 0;
      if (params.identityKind === "browser") {
        for (const worker of mobileHub.listWorkers()) {
          if (worker.clientId === params.identityId && mobileHub.removeWorker(worker.id)) {
            disconnected += 1;
          }
        }
      } else {
        for (const worker of store.listWorkers()) {
          if (
            worker.identityKind === params.identityKind
            && worker.identityId === params.identityId
            && hub.removeWorker(worker.id)
          ) {
            disconnected += 1;
          }
        }
      }
      return {
        state: result.state,
        disconnected,
        credential: {
          ...admissionCredentialSummary(revokedCredential),
          revokedAt: revokedCredential.revokedAt === null
            ? null
            : new Date(revokedCredential.revokedAt).toISOString(),
          revocationReason: revokedCredential.revocationReason,
        },
      };
    },
  );

  app.get("/v1/models", async () => {
    const models = scheduler.listAvailableModels({ connectedWorkerIds: hub.connectedWorkerIds() });
    return {
      object: "list",
      data: models.map((model) => ({
        id: model.id,
        object: "model",
        created: 0,
        owned_by: "mycellios",
        x_replicas: model.replicas,
        x_pipelines: model.pipelines,
      })),
    };
  });

  app.post("/v1/chat/completions", async (request, reply) => {
    const parsed = chatCompletionRequestSchema.parse(request.body) as ChatCompletionRequest;
    const principal: ApiRequestPrincipal = apiAccessEnabled
      ? principalFor(request)
      : { kind: "system" };
    const idempotencyKey = parseIdempotencyKey(request.headers["idempotency-key"]);
    const supersededJobId = service.cancelMatchingActiveSession(parsed, parsed.session_id);
    if (supersededJobId) {
      app.log.warn({
        supersededJobId,
        sessionId: parsed.session_id,
        model: parsed.model,
      }, "replacing an interrupted identical chat request after client reconnection");
      // Give the cell worker time to process task.cancel and publish its freed
      // slot before the replacement lease is offered on the same socket.
      await new Promise<void>((resolveDelay) => setTimeout(resolveDelay, 1_000));
    }
    if (parsed.stream && !service.hasCapacity(parsed, parsed.session_id)) {
      // A reconnect can race both route rebuilding and the automatic startup
      // benchmark. Keep the fetch pending while capacity returns instead of
      // making the installed desktop surface a transient 503.
      await waitForChatCapacity(service, parsed, parsed.session_id, 45_000);
    }
    const usage = principal.kind === "system"
      ? null
      : apiAccess.beginUsage(
          principal.userId,
          principal.kind === "api_key" ? principal.apiKeyId : null,
          parsed,
        );
    if (usage) {
      reply.header("x-ratelimit-limit", usage.rateLimit.limit);
      reply.header("x-ratelimit-remaining", usage.rateLimit.remaining);
      reply.header("x-ratelimit-reset", Math.ceil(usage.rateLimit.resetAt / 1_000));
      reply.header("x-token-balance", usage.remainingTokens);
    }
    let handle: ReturnType<MeshService["submit"]>;
    try {
      handle = service.submit(parsed, parsed.session_id, idempotencyKey);
    } catch (error) {
      if (usage) apiAccess.failUsage(usage.id, "submission_failed");
      throw error;
    }
    if (usage) apiAccess.attachJob(usage.id, handle.jobId, handle.sessionId);
    const conversationId = store.startInferenceConversation(
      handle.sessionId,
      parsed.model,
      parsed.messages,
    );
    reply.header("x-network-request-id", handle.jobId);
    reply.header("x-network-session-id", handle.sessionId);

    if (parsed.stream) {
      reply.hijack();
      reply.raw.writeHead(200, {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache, no-transform",
        connection: "keep-alive",
        "x-accel-buffering": "no",
        "x-network-request-id": handle.jobId,
        "x-network-session-id": handle.sessionId,
      });
      let finished = false;
      let streamedText = "";
      let streamedRouteClass = "replica";
      let streamedResult: Extract<JobStreamEvent, { type: "completed" }> | null = null;
      let streamedFailure: Extract<JobStreamEvent, { type: "failed" }> | null = null;
      let streamError: unknown = null;
      reply.raw.once("close", () => {
        if (!finished) service.cancel(handle.jobId);
      });
      const heartbeatTimer = setInterval(() => {
        if (!reply.raw.destroyed && !reply.raw.writableEnded) {
          reply.raw.write(`: mycellios-heartbeat ${Date.now()}\n\n`);
        }
      }, 10_000);
      heartbeatTimer.unref();
      try {
        for await (const event of handle.events) {
          if (event.type === "accepted") streamedRouteClass = event.route.routeClass;
          if (event.type === "token") streamedText += event.token.text;
          if (event.type === "completed") streamedResult = event;
          if (event.type === "failed") streamedFailure = event;
          writeMycelliosEvent(reply.raw, event, parsed.model, handle.jobId);
        }
      } catch (error) {
        streamError = error;
      } finally {
        clearInterval(heartbeatTimer);
      }
      finished = true;
      if (usage) {
        if (streamedResult) {
          const account = apiAccess.completeUsage(
            usage.id,
            streamedResult.result.metrics.inputTokens,
            streamedResult.result.metrics.outputTokens,
          );
          void account;
        } else {
          apiAccess.failUsage(
            usage.id,
            streamedFailure?.code ?? (streamError ? "stream_failed" : "missing_result"),
          );
        }
      }
      store.appendInferenceMessage({
        conversationId,
        jobId: handle.jobId,
        role: "assistant",
        content: streamedResult?.result.text ?? streamedText,
        status: streamedFailure || streamError || !streamedResult ? "failed" : "completed",
        inputTokens: streamedResult?.result.metrics.inputTokens ?? null,
        outputTokens: streamedResult?.result.metrics.outputTokens ?? null,
        routeClass: streamedRouteClass,
        latencyMs: streamedResult?.result.metrics.activeMs ?? null,
        metadata: streamedFailure || streamError
          ? {
              failure_code: streamedFailure?.code ?? "stream_failed",
              failure_message: streamedFailure?.message
                ?? (streamError instanceof Error ? streamError.message : String(streamError)),
            }
          : {
              finish_reason: streamedResult?.result.finishReason ?? null,
              execution_trace: streamedResult?.result.networkTrace ?? null,
            },
      });
      void persistence?.flush();
      if (streamError && !reply.raw.destroyed && !reply.raw.writableEnded) {
        reply.raw.write(`data: ${JSON.stringify({
          error: {
            code: "stream_failed",
            message: streamError instanceof Error ? streamError.message : String(streamError),
          },
        })}\n\n`);
      }
      reply.raw.write("data: [DONE]\n\n");
      reply.raw.end();
      return;
    }

    let result: Extract<JobStreamEvent, { type: "completed" }> | null = null;
    let routeClass = "replica";
    let affinityHit = false;
    try {
      for await (const event of handle.events) {
        if (event.type === "accepted") {
          routeClass = event.route.routeClass;
          affinityHit = event.route.affinityHit;
        }
        if (event.type === "failed") {
          throw new MeshServiceError(event.code, event.message, 502);
        }
        if (event.type === "completed") result = event;
      }
      if (!result) {
        throw new MeshServiceError("missing_result", "Worker stream ended without a result", 502);
      }
    } catch (error) {
      if (usage) {
        apiAccess.failUsage(
          usage.id,
          error instanceof MeshServiceError ? error.code : "inference_failed",
        );
      }
      throw error;
    }
    if (usage) {
      const account = apiAccess.completeUsage(
        usage.id,
        result.result.metrics.inputTokens,
        result.result.metrics.outputTokens,
      );
      if (account) reply.header("x-token-balance", account.tokenBalance);
    }
    reply.header("x-route-class", routeClass);
    store.appendInferenceMessage({
      conversationId,
      jobId: handle.jobId,
      role: "assistant",
      content: result.result.text,
      status: "completed",
      inputTokens: result.result.metrics.inputTokens,
      outputTokens: result.result.metrics.outputTokens,
      routeClass,
      latencyMs: result.result.metrics.activeMs,
      metadata: {
        finish_reason: result.result.finishReason,
        affinity_hit: affinityHit,
        ttft_ms: result.result.metrics.ttftMs,
        reused_kv_tokens: result.result.metrics.reusedKvTokens ?? 0,
        execution_trace: result.result.networkTrace ?? null,
      },
    });
    void persistence?.flush();
    return {
      id: handle.jobId,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1_000),
      model: parsed.model,
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: result.result.text },
          finish_reason: result.result.finishReason,
        },
      ],
      usage: {
        prompt_tokens: result.result.metrics.inputTokens,
        completion_tokens: result.result.metrics.outputTokens,
        total_tokens: result.result.metrics.inputTokens + result.result.metrics.outputTokens,
      },
      x_network: {
        session_id: handle.sessionId,
        route_class: routeClass,
        affinity_hit: affinityHit,
        reused_kv_tokens: result.result.metrics.reusedKvTokens ?? 0,
        ttft_ms: result.result.metrics.ttftMs,
        active_ms: result.result.metrics.activeMs,
        execution_trace: result.result.networkTrace ?? null,
      },
    };
  });

  app.get("/v1/requests/:jobId", async (request, reply) => {
    const { jobId } = jobIdParamsSchema.parse(request.params);
    const principal = apiAccessEnabled ? principalFor(request) : { kind: "system" as const };
    if (
      principal.kind !== "system"
      && !apiAccess.userOwnsJob(principal.userId, jobId)
    ) {
      return reply.code(404).send({ error: { code: "not_found" } });
    }
    const job = store.getJob(jobId);
    if (!job) return reply.code(404).send({ error: { code: "not_found" } });
    return {
      id: job.id,
      session_id: job.sessionId,
      model: job.model,
      status: job.status,
      input_tokens: job.inputTokens,
      output_tokens: job.outputTokens,
      failure_code: job.failureCode,
      deadline_at: new Date(job.deadlineAt).toISOString(),
      created_at: new Date(job.createdAt).toISOString(),
      updated_at: new Date(job.updatedAt).toISOString(),
    };
  });

  app.get("/v1/conversations/:sessionId/messages", async (request, reply) => {
    const { sessionId } = z.object({ sessionId: z.string().min(1).max(200) }).parse(request.params);
    const principal = apiAccessEnabled ? principalFor(request) : { kind: "system" as const };
    if (
      principal.kind !== "system"
      && !apiAccess.userOwnsSession(principal.userId, sessionId)
    ) {
      return reply.code(404).send({ error: { code: "not_found" } });
    }
    return {
      data: store.listInferenceMessages(sessionId).map((message) => ({
        id: message.id,
        job_id: message.jobId,
        role: message.role,
        content: message.content,
        status: message.status,
        input_tokens: message.inputTokens,
        output_tokens: message.outputTokens,
        route_class: message.routeClass,
        latency_ms: message.latencyMs,
        metadata: message.metadata,
        created_at: new Date(message.createdAt).toISOString(),
      })),
    };
  });

  app.post("/v1/requests/:jobId/cancel", async (request, reply) => {
    const { jobId } = jobIdParamsSchema.parse(request.params);
    const principal = apiAccessEnabled ? principalFor(request) : { kind: "system" as const };
    if (
      principal.kind !== "system"
      && !apiAccess.userOwnsJob(principal.userId, jobId)
    ) {
      return reply.code(404).send({ error: { code: "not_found_or_terminal" } });
    }
    if (!service.cancel(jobId)) {
      return reply.code(404).send({ error: { code: "not_found_or_terminal" } });
    }
    return reply.code(202).send({ id: jobId, status: "cancelled" });
  });

  const landingAssetsPath = resolveLandingAssetsPath(
    config.landingAssetsPath,
    runtimeMetadata.root,
  );
  const releaseTokenVerifier = options.releaseTokenVerifier ?? verifyGitHubReleaseUploadToken;
  const releaseChunkBodyLimitBytes = options.releaseChunkBodyLimitBytes
    ?? MAX_RELEASE_CHUNK_SIZE_BYTES;
  if (
    !Number.isSafeInteger(releaseChunkBodyLimitBytes)
    || releaseChunkBodyLimitBytes < 1
    || releaseChunkBodyLimitBytes > MAX_RELEASE_CHUNK_SIZE_BYTES
  ) {
    throw new Error("release_chunk_body_limit_invalid");
  }
  const authorizeReleaseMutation = async (
    request: FastifyRequest,
    reply: FastifyReply,
  ): Promise<GitHubReleaseClaims | null> => {
    const authorization = parseBearerToken(request.headers.authorization);
    if (!authorization) {
      void reply.code(401).send({ error: { code: "release_upload_token_missing" } });
      return null;
    }
    let claims: GitHubReleaseClaims;
    try {
      claims = await releaseTokenVerifier(authorization);
    } catch {
      void reply.code(401).send({ error: { code: "release_upload_token_invalid" } });
      return null;
    }
    if (runtimeRevision === null) {
      void reply.code(503).send({
        error: { code: "release_upload_runtime_revision_unavailable" },
      });
      return null;
    }
    if (!coordinatorBuildIdentity || !releaseTransactions) {
      void reply.code(503).send({
        error: { code: "release_upload_runtime_source_unavailable" },
      });
      return null;
    }
    if (claims.sha !== runtimeRevision) {
      void reply.code(409).send({
        error: { code: "release_upload_revision_mismatch" },
      });
      return null;
    }
    return claims;
  };
  app.addHook("onRequest", async (request, reply) => {
    const path = request.url.split("?", 1)[0] ?? request.url;
    if (!path.startsWith("/internal/v1/releases/")) return;
    if (!await authorizeReleaseMutation(request, reply)) return reply;
  });
  app.put(
    "/internal/v1/releases/:channel/:fileName",
    { bodyLimit: releaseChunkBodyLimitBytes },
    async (request, reply) => {
    const params = z.object({
      channel: z.enum(["updates", "downloads"]),
      fileName: z.string().min(1).max(160),
    }).parse(request.params);
    if (!Buffer.isBuffer(request.body)) {
      return reply.code(400).send({ error: { code: "release_chunk_body_invalid" } });
    }
    try {
      const result = await releaseTransactions!.storeChunk({
        identity: parseReleaseTransactionIdentity(request.headers),
        channel: params.channel as ReleaseAssetChannel,
        fileName: params.fileName,
        metadata: parseReleaseChunkMetadata(request.headers),
        body: request.body,
      });
      return reply.code(result.complete ? 201 : 202).send(result);
    } catch (error) {
      return reply.code(400).send({
        error: {
          code: "release_chunk_rejected",
          message: error instanceof Error ? error.message : "release_chunk_rejected",
        },
      });
    }
    },
  );
  app.post(
    "/internal/v1/releases/transactions/:transactionId/commit",
    { bodyLimit: 64 * 1024 },
    async (request, reply) => {
      const { transactionId } = z.object({
        transactionId: z.string().min(16).max(128),
      }).parse(request.params);
      const bodyIdentity = z.object({
        transactionId: z.string(),
      }).passthrough().parse(request.body);
      if (bodyIdentity.transactionId !== transactionId) {
        return reply.code(400).send({
          error: { code: "release_transaction_path_mismatch" },
        });
      }
      try {
        return reply.code(201).send(
          await releaseTransactions!.commit(request.body),
        );
      } catch (error) {
        return reply.code(409).send({
          error: {
            code: "release_commit_rejected",
            message: error instanceof Error ? error.message : "release_commit_rejected",
          },
        });
      }
    },
  );
  app.post(
    "/internal/v1/releases/transactions/:transactionId/rollback",
    async (request, reply) => {
      const { transactionId } = z.object({
        transactionId: z.string().min(16).max(128),
      }).parse(request.params);
      try {
        return reply.send(await releaseTransactions!.rollback(transactionId));
      } catch (error) {
        return reply.code(409).send({
          error: {
            code: "release_rollback_rejected",
            message: error instanceof Error ? error.message : "release_rollback_rejected",
          },
        });
      }
    },
  );
  app.post(
    "/internal/v1/releases/transactions/:transactionId/abort",
    async (request, reply) => {
      const { transactionId } = z.object({
        transactionId: z.string().min(16).max(128),
      }).parse(request.params);
      try {
        return reply.send(await releaseTransactions!.abort(transactionId));
      } catch (error) {
        return reply.code(409).send({
          error: {
            code: "release_abort_rejected",
            message: error instanceof Error ? error.message : "release_abort_rejected",
          },
        });
      }
    },
  );
  app.get("/downloads/macos-arm64", async (_request, reply) => {
    reply.header("Cache-Control", "no-cache, no-store, must-revalidate");
    return reply.redirect(`/downloads/mycellios-macos-arm64.dmg?v=${publicAssetVersion}`);
  });
  app.get("/downloads/linux-deb", async (_request, reply) => {
    reply.header("Cache-Control", "no-cache, no-store, must-revalidate");
    return reply.redirect(`/downloads/mycellios-linux-x64.deb?v=${publicAssetVersion}`);
  });
  app.get("/downloads/linux-rpm", async (_request, reply) => {
    reply.header("Cache-Control", "no-cache, no-store, must-revalidate");
    return reply.redirect(`/downloads/mycellios-linux-x64.rpm?v=${publicAssetVersion}`);
  });
  const contentHubClient = config.contentHubApiUrl
    ? new ContentHubClient({ baseUrl: config.contentHubApiUrl })
    : null;
  await registerContentHubRoutes(app, {
    client: contentHubClient,
    publicationWebhookSecret: config.publicationWebhookSecret,
    ...(landingAssetsPath
      ? { fallbackSitemapPath: resolve(landingAssetsPath, "sitemap.xml") }
      : {}),
  });
  if (landingAssetsPath) {
    await app.register(staticFiles, {
      root: landingAssetsPath,
      prefix: "/",
      decorateReply: true,
      index: "index.html",
      cacheControl: false,
      setHeaders: setPublicAssetCacheHeaders,
    });
    const landingRouteDocuments = {
      "/network": "network/index.html",
      "/admin": "admin/index.html",
      "/join": "join/index.html",
      "/downloads": "downloads/index.html",
    } as const;
    for (const [path, routeDocument] of Object.entries(landingRouteDocuments)) {
      const document = existsSync(resolve(landingAssetsPath, routeDocument))
        ? routeDocument
        : "index.html";
      app.get(path, async (_request, reply) => reply.sendFile(document));
    }
  }

  app.setErrorHandler((error, _request, reply) => {
    if (error instanceof ZodError) {
      return reply.code(400).send({
        error: { code: "invalid_request", message: "Request validation failed", details: error.issues },
      });
    }
    if (error instanceof MeshServiceError) {
      return reply.code(error.statusCode).send({ error: { code: error.code, message: error.message } });
    }
    if (error instanceof ApiAccessError) {
      if (error.retryAfterSeconds !== undefined) {
        reply.header("retry-after", error.retryAfterSeconds);
      }
      return reply.code(error.statusCode).send({
        error: { code: error.code, message: error.message },
      });
    }
    if (error instanceof WorkerAdmissionError) {
      return reply.code(error.statusCode).send({
        error: { code: error.code, message: error.message },
      });
    }
    app.log.error(error);
    return reply.code(500).send({
      error: { code: "internal_error", message: "The coordinator could not process the request" },
    });
  });

  return {
    app,
    database,
    store,
    scheduler,
    hub,
    mobileHub,
    service,
    apiAccess,
    persistence,
    deploymentController,
    async close() {
      clearInterval(staleTimer);
      clearInterval(networkTelemetryTimer);
      for (const timer of automaticBenchmarkRetryTimers) clearTimeout(timer);
      automaticBenchmarkRetryTimers.clear();
      fleetContribution.close();
      hub.close();
      mobileHub.close();
      await activationManager?.close();
      await app.close();
      await persistence?.close();
      database.close();
    },
  };
}

function queueExistingMobileArtifacts(
  persistence: SupabasePersistence,
  directory: string,
): void {
  let files: string[];
  try {
    files = readdirSync(directory);
  } catch {
    return;
  }
  for (const file of files) {
    const path = resolve(directory, file);
    if (file.endsWith(".bin")) {
      const weightsHash = file.slice(0, -4);
      if (!/^[a-f0-9]{64}$/.test(weightsHash)) continue;
      const sizeBytes = statSync(path).size;
      persistence.registerArtifactBackup({
        id: `mobile-weight-${weightsHash}`,
        localPath: path,
        storagePath: `mobile-experts/weights/${file}`,
        contentType: "application/octet-stream",
        sha256: weightsHash,
        sizeBytes,
        metadata: { kind: "mobile-expert-weights", weightsHash },
      });
      continue;
    }
    if (!file.endsWith(".json")) continue;
    const artifactId = file.slice(0, -5);
    if (!/^[a-f0-9]{64}$/.test(artifactId)) continue;
    const body = readFileSync(path);
    persistence.registerArtifactBackup({
      id: `mobile-manifest-${artifactId}`,
      localPath: path,
      storagePath: `mobile-experts/manifests/${file}`,
      contentType: "application/json",
      sha256: createHash("sha256").update(body).digest("hex"),
      sizeBytes: body.length,
      metadata: { kind: "mobile-expert-manifest", artifactId },
    });
  }
}

const benchmarkRunRequestSchema = z.object({
  model: z.string().trim().min(1).max(200).optional(),
  version: z.string().trim().min(1).max(80).optional(),
  label: z.string().trim().min(1).max(160).optional(),
});

const huggingFaceModelSearchSchema = z.object({
  q: z.string().trim().max(80).default(""),
  cursor: z.string().trim().max(4_096).optional(),
  sort: z.enum(["downloads", "likes", "lastModified"]).default("downloads"),
  limit: z.coerce.number().int().min(10).max(100).default(50),
});

const requestedModelCreateSchema = z.object({
  id: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
  source: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
  revision: z.string().min(1).max(512).nullable().default(null),
  contextTokens: z.number().int().min(128).max(1_048_576).default(4_096),
  minimumNodes: z.number().int().min(2).max(8).default(2),
  autoActivate: z.boolean().default(true),
}).strict();

export function isLoopbackAddress(address: string): boolean {
  const normalized = address.toLowerCase();
  return normalized === "::1" || normalized === "127.0.0.1" || normalized.startsWith("127.") || normalized.startsWith("::ffff:127.");
}

const PROXY_FORWARD_HEADERS = [
  "x-forwarded-for",
  "x-forwarded-host",
  "x-forwarded-proto",
  "x-real-ip",
  "forwarded",
] as const;

/**
 * A loopback `request.ip` only proves the request came from this host when
 * nothing is proxying on our behalf. Fastify runs with `trustProxy` disabled,
 * so a reverse proxy terminating on 127.0.0.1 makes *every* remote request
 * look local. Any forwarding header means we cannot tell who is really
 * calling, so local-only routes must refuse rather than guess.
 */
export function isTrustedLocalRequest(request: FastifyRequest): boolean {
  if (!isLoopbackAddress(request.ip)) return false;
  return !PROXY_FORWARD_HEADERS.some((header) => request.headers[header] !== undefined);
}

function resolveMobileAssetsPath(
  configured: string | undefined,
  runtimeRoot: string,
): string | null {
  const candidates = [configured, resolve(runtimeRoot, "mobile-dist")].filter(
    (candidate): candidate is string => Boolean(candidate),
  );
  return candidates.find((candidate) => existsSync(resolve(candidate, "index.html"))) ?? null;
}

function setPublicAssetCacheHeaders(
  reply: FastifyReply,
  filePath: string,
): void {
  const normalized = filePath.replaceAll("\\", "/");
  if (
    normalized.endsWith("/index.html") ||
    normalized.endsWith("/blog.css") ||
    normalized.endsWith("/sw.js") ||
    normalized.endsWith("/manifest.webmanifest") ||
    normalized.includes("/downloads/") ||
    normalized.endsWith("/RELEASES") ||
    normalized.endsWith("-setup.exe") ||
    normalized.endsWith("/latest.json")
  ) {
    reply.header("Cache-Control", "no-cache, no-store, must-revalidate");
    return;
  }
  if (normalized.includes("/assets/")) {
    reply.header("Cache-Control", "public, max-age=31536000, immutable");
    return;
  }
  if (normalized.endsWith(".nupkg")) {
    reply.header("Cache-Control", "public, max-age=31536000, immutable");
    return;
  }
  reply.header("Cache-Control", "public, max-age=3600");
}

function resolveDesktopUpdatesPath(
  configured: string | undefined,
  runtimeRoot: string,
): string | null {
  const candidates = [configured, resolve(runtimeRoot, "updates", "win32", "x64")].filter(
    (candidate): candidate is string => Boolean(candidate),
  );
  return candidates.find((candidate) => existsSync(resolve(candidate, "RELEASES"))) ?? null;
}

function releaseAssetRoot(
  channel: ReleaseAssetChannel,
  configuredUpdates: string | undefined,
  configuredDownloads: string | undefined,
  configuredLanding: string | undefined,
  runtimeRoot: string,
): string {
  if (channel === "updates") {
    return resolve(configuredUpdates ?? resolve(runtimeRoot, "updates", "win32", "x64"));
  }
  if (configuredDownloads) return resolve(configuredDownloads);
  return resolve(
    configuredLanding ?? resolve(runtimeRoot, "landing-dist"),
    "downloads",
  );
}

function releaseTransactionStorageRoot(
  updatesRoot: string,
  downloadsRoot: string,
  runtimeRoot: string,
): string {
  let common = dirname(resolve(updatesRoot));
  const downloadsParent = dirname(resolve(downloadsRoot));
  while (
    downloadsParent !== common
    && !downloadsParent.startsWith(`${common}${sep}`)
  ) {
    const parent = dirname(common);
    if (parent === common) {
      return resolve(runtimeRoot, "release-transactions");
    }
    common = parent;
  }
  if (dirname(common) === common) {
    return resolve(runtimeRoot, "release-transactions");
  }
  return resolve(common, "release-transactions");
}

async function releasePublicAssetPath(
  transactions: NativeReleaseTransactionStore | null,
  channel: ReleaseAssetChannel,
  fileName: string,
  legacyRoot: string | null,
): Promise<string | null> {
  if (transactions) return transactions.publicAssetPath(channel, fileName);
  try {
    validateReleaseAssetName(channel, fileName);
  } catch {
    return null;
  }
  return legacyRoot ? resolve(legacyRoot, fileName) : null;
}

function serveReleaseAsset(
  request: FastifyRequest,
  reply: FastifyReply,
  filePath: string | null,
): FastifyReply {
  if (!filePath || !existsSync(filePath)) return reply.code(404).send();
  const details = lstatSync(filePath);
  if (!details.isFile() || details.isSymbolicLink()) {
    return reply.code(404).send();
  }
  setPublicAssetCacheHeaders(reply, filePath);
  const entityTag = `"${createHash("sha256")
    .update(`${filePath}\n${details.size}\n${details.mtimeMs}`)
    .digest("base64url")}"`;
  const lastModified = details.mtime.toUTCString();
  reply.header("Accept-Ranges", "bytes");
  reply.header("ETag", entityTag);
  reply.header("Last-Modified", lastModified);
  if (filePath.endsWith(".json")) reply.type("application/json; charset=utf-8");
  else if (filePath.endsWith(".dmg")) reply.type("application/x-apple-diskimage");
  else if (filePath.endsWith(".deb")) reply.type("application/vnd.debian.binary-package");
  else if (filePath.endsWith(".rpm")) reply.type("application/x-rpm");
  else reply.type("application/octet-stream");

  const rangeHeader = request.headers.range;
  const ifRangeHeader = request.headers["if-range"];
  if (
    typeof rangeHeader === "string"
    && (
      typeof ifRangeHeader !== "string"
      || ifRangeMatches(ifRangeHeader, entityTag, details.mtimeMs)
    )
  ) {
    const range = parseSingleByteRange(rangeHeader, details.size);
    if (!range) {
      reply.header("Content-Range", `bytes */${details.size}`);
      reply.header("Content-Length", 0);
      return reply.code(416).send();
    }
    reply.header("Content-Range", `bytes ${range.start}-${range.end}/${details.size}`);
    reply.header("Content-Length", range.end - range.start + 1);
    reply.code(206);
    return reply.send(createReadStream(filePath, range));
  }

  reply.header("Content-Length", details.size);
  return reply.send(createReadStream(filePath));
}

export function parseSingleByteRange(
  value: string,
  size: number,
): { start: number; end: number } | null {
  if (!Number.isSafeInteger(size) || size < 1) return null;
  const match = /^bytes=(\d*)-(\d*)$/i.exec(value.trim());
  if (!match || (!match[1] && !match[2])) return null;
  if (!match[1]) {
    const suffixLength = Number(match[2]);
    if (!Number.isSafeInteger(suffixLength) || suffixLength < 1) return null;
    return {
      start: Math.max(0, size - suffixLength),
      end: size - 1,
    };
  }
  const start = Number(match[1]);
  if (!Number.isSafeInteger(start) || start < 0 || start >= size) return null;
  const requestedEnd = match[2] ? Number(match[2]) : size - 1;
  if (
    !Number.isSafeInteger(requestedEnd)
    || requestedEnd < start
  ) return null;
  return {
    start,
    end: Math.min(requestedEnd, size - 1),
  };
}

function ifRangeMatches(
  value: string,
  entityTag: string,
  modifiedAtMs: number,
): boolean {
  const candidate = value.trim();
  if (candidate.startsWith("\"") || candidate.startsWith("W/")) {
    return candidate === entityTag;
  }
  const timestamp = Date.parse(candidate);
  return Number.isFinite(timestamp)
    && Math.floor(modifiedAtMs / 1_000) * 1_000 <= timestamp;
}

function resolveReleaseDownloadsPath(
  configuredDownloads: string | undefined,
  configuredLanding: string | undefined,
  runtimeRoot: string,
): string | null {
  const candidates = [
    configuredDownloads,
    configuredLanding ? resolve(configuredLanding, "downloads") : undefined,
    resolve(runtimeRoot, "landing-dist", "downloads"),
  ].filter((candidate): candidate is string => Boolean(candidate));
  return candidates.find((candidate) => existsSync(candidate)) ?? null;
}

function resolveLandingAssetsPath(
  configured: string | undefined,
  runtimeRoot: string,
): string | null {
  const candidates = [configured, resolve(runtimeRoot, "landing-dist")].filter(
    (candidate): candidate is string => Boolean(candidate),
  );
  return candidates.find((candidate) => existsSync(resolve(candidate, "index.html"))) ?? null;
}

function dashboardWorkers(store: MeshStore, hub: WorkerHub, mobileHub: MobileComputeHub) {
  return [
    ...store.listWorkers().filter((worker) => storedWorkerIsVisible(worker, hub)).map((worker) => ({
      id: worker.id,
      status: worker.status,
      connected: hub.isConnected(worker.id),
      region: worker.capabilities.region,
      agentVersion: worker.capabilities.agentVersion,
      buildIdentity: worker.capabilities.buildIdentity,
      offeredVramMb: worker.capabilities.gpus.reduce(
        (sum, gpu) => sum + gpu.offeredVramMb,
        0,
      ),
      gpus: worker.capabilities.gpus,
      deployments: worker.capabilities.deployments,
      executionNodeId: worker.capabilities.distributedExecutor?.nodeId,
      computeMode: worker.capabilities.distributedExecutor?.computeMode,
      acceleration: worker.capabilities.distributedExecutor?.acceleration,
      isolation: worker.capabilities.distributedExecutor?.isolation,
      physicalIdentity: worker.capabilities.distributedExecutor?.physicalIdentity,
      reliability: worker.reliability,
      jobsCompleted: worker.jobsCompleted,
      lastSeenAt: new Date(worker.lastSeenAt).toISOString(),
      kind: storedWorkerKind(worker),
    })),
    ...mobileHub.listWorkers().map((worker) => ({
      ...mobileDashboardWorker(worker),
      kind: "browser" as const,
    })),
  ];
}

function storedWorkerKind(worker: StoredWorker): "desktop" | "cell" {
  if (
    worker.identityKind === "cell" ||
    worker.capabilities.gpus.some((gpu) => gpu.vendor === "sidecar-cell")
  ) {
    return "cell";
  }
  return "desktop";
}

function storedWorkerIsVisible(worker: StoredWorker, hub: WorkerHub): boolean {
  return storedWorkerKind(worker) !== "cell" || hub.isConnected(worker.id);
}

function publicSnapshot(
  store: MeshStore,
  scheduler: Scheduler,
  hub: WorkerHub,
  mobileHub: MobileComputeHub,
  activationManager: ModelActivationManager | undefined,
  activationPresentation: {
    activationProgressForModel(modelId: string): readonly ModelActivationProgressEvent[];
    activationStatusMessageForModel(modelId: string): string | null;
    activationIncidentForModel(modelId: string): ActivationIncident | null;
  } | undefined,
  version: string,
  buildIdentity: NativeBuildIdentity | null = null,
) {
  const workers = dashboardWorkers(store, hub, mobileHub);
  const models = scheduler.listAvailableModels({ connectedWorkerIds: hub.connectedWorkerIds() });
  const requestedModels = requestedModelCapacityViews({
    requests: store.listRequestedModels(),
    workers: store.listWorkers(),
    connectedWorkerIds: hub.connectedWorkerIds(),
    activeModelIds: new Set(models.map((model) => model.id)),
    ...(activationManager
      ? { executionNodesForModel: (modelId: string) => activationManager.capacityNodesForModel(modelId) }
      : {}),
    ...(activationPresentation
      ? {
          activationProgressForModel: activationPresentation.activationProgressForModel,
          activationStatusMessageForModel: activationPresentation.activationStatusMessageForModel,
          activationIncidentForModel: activationPresentation.activationIncidentForModel,
        }
      : activationManager?.activationProgressForModel
        ? { activationProgressForModel: (modelId: string) => activationManager.activationProgressForModel!(modelId) }
        : {}),
    activationAvailable: activationManager !== undefined,
  });
  const jobs = store.listJobs(100).map((job) => ({
    id: job.id,
    model: job.model,
    status: job.status,
    workerId: job.workerId,
    inputTokens: job.inputTokens,
    outputTokens: job.outputTokens,
    failureCode: job.failureCode,
    createdAt: new Date(job.createdAt).toISOString(),
    updatedAt: new Date(job.updatedAt).toISOString(),
  }));
  return {
    capturedAt: new Date().toISOString(),
    version,
    buildIdentity,
    summary: {
      registered: workers.length,
      connected: workers.filter((worker) => worker.connected).length,
      online: workers.filter((worker) => worker.status === "online").length,
      mobile: workers.filter((worker) => worker.kind === "browser").length,
      offeredVramMb: workers.reduce((sum, worker) => sum + worker.offeredVramMb, 0),
      completedJobs: jobs.filter((job) => job.status === "completed").length,
    },
    workers,
    models: models.map((model) => ({
      id: model.id,
      replicas: model.replicas,
      pipelines: model.pipelines,
    })),
    requestedModels,
    jobs,
  };
}

function networkTelemetrySample(
  snapshot: ReturnType<typeof publicSnapshot>,
  capturedAt: number,
): StoredNetworkTelemetrySample {
  const onlineWorkers = snapshot.workers.filter(
    (worker) => worker.connected && worker.status === "online",
  );
  const offeredVramMb = onlineWorkers.reduce(
    (total, worker) => total + worker.offeredVramMb,
    0,
  );
  const freeVramMb = onlineWorkers.reduce(
    (total, worker) => total
      + worker.gpus.reduce((gpuTotal, gpu) => gpuTotal + gpu.freeOfferedVramMb, 0),
    0,
  );
  const inflightJobs = snapshot.jobs.filter(
    (job) => ["queued", "leasing", "running", "streaming"].includes(job.status),
  );
  return {
    capturedAt,
    registeredNodes: snapshot.summary.registered,
    connectedNodes: snapshot.summary.connected,
    onlineNodes: snapshot.summary.online,
    browserNodes: snapshot.summary.mobile,
    activeModels: snapshot.models.length,
    modelReplicas: snapshot.models.reduce((total, model) => total + model.replicas, 0),
    modelPipelines: snapshot.models.reduce((total, model) => total + model.pipelines, 0),
    offeredVramMb,
    freeVramMb: Math.min(offeredVramMb, freeVramMb),
    inflightJobs: inflightJobs.length,
    runningJobs: inflightJobs.filter(
      (job) => job.status === "running" || job.status === "streaming",
    ).length,
    completedJobs: snapshot.summary.completedJobs,
  };
}

function mobileDashboardWorker(worker: MobileWorkerSnapshot) {
  return {
    id: worker.id,
    status: worker.status,
    connected: worker.connected,
    region: worker.region,
    buildIdentity: worker.buildIdentity,
    offeredVramMb: 0,
    reliability:
      worker.completedTasks + worker.failedTasks === 0
        ? 1
        : worker.completedTasks / (worker.completedTasks + worker.failedTasks),
    jobsCompleted: worker.completedTasks,
    lastSeenAt: worker.lastSeenAt,
    gpus: [
      {
        id: `mobile-${worker.id}`,
        vendor: worker.backend === "webgpu" ? "WebGPU" : "Browser CPU",
        model: `${worker.name} · ${worker.backend.toUpperCase()}`,
        physicalVramMb: 0,
        sharedMemoryMb: worker.capabilities.deviceMemoryGb
          ? Math.round(worker.capabilities.deviceMemoryGb * 1_024)
          : undefined,
        offeredVramMb: 0,
        freeOfferedVramMb: 0,
        utilizationPct: worker.connected && worker.visible ? 100 : 0,
      },
    ],
    deployments: [],
    mobile: {
      platform: worker.platform,
      backend: worker.backend,
      performanceLevel: worker.performanceLevel,
      wakeLock: worker.wakeLock,
      estimatedGflops: worker.estimatedGflops,
      verifiedTasks: worker.verifiedTasks,
      residentExperts: worker.residentExperts,
    },
  };
}

async function waitForChatCapacity(
  service: MeshService,
  request: ChatCompletionRequest,
  sessionId: string | undefined,
  timeoutMs: number,
): Promise<void> {
  const deadline = Date.now() + Math.max(0, timeoutMs);
  while (!service.hasCapacity(request, sessionId) && Date.now() < deadline) {
    await new Promise<void>((resolveDelay) => setTimeout(resolveDelay, 250));
  }
}

function parseIdempotencyKey(received: string | string[] | undefined): string | undefined {
  const value = Array.isArray(received) ? received[0] : received;
  if (value === undefined) return undefined;
  if (!/^[\x21-\x7E]{1,128}$/.test(value)) {
    throw new MeshServiceError(
      "invalid_idempotency_key",
      "Idempotency-Key must contain 1-128 printable ASCII characters",
      400,
    );
  }
  return value;
}

interface SupportAssistantRateState {
  windowStartedAt: number;
  requests: number;
  active: number;
}

function supportAssistantRateKey(request: FastifyRequest, sessionId: string): string {
  const userAgent = request.headers["user-agent"] ?? "unknown";
  return createHash("sha256")
    .update(`${request.ip}\n${userAgent}\n${sessionId}`)
    .digest("hex")
    .slice(0, 24);
}

function claimSupportAssistantRequest(
  states: Map<string, SupportAssistantRateState>,
  key: string,
  now = Date.now(),
): (() => void) | null {
  const windowMs = 10 * 60_000;
  const previous = states.get(key);
  const state = !previous || now - previous.windowStartedAt >= windowMs
    ? { windowStartedAt: now, requests: 0, active: 0 }
    : previous;
  // A reconnect can overlap briefly with the abandoned socket. Two active
  // attempts let the new request supersede the stale job without opening an
  // unlimited parallel-inference path for one browser session.
  if (state.requests >= 24 || state.active >= 2) return null;
  state.requests += 1;
  state.active += 1;
  states.set(key, state);
  if (states.size > 2_000) {
    for (const [candidateKey, candidate] of states) {
      if (now - candidate.windowStartedAt >= windowMs && candidate.active === 0) {
        states.delete(candidateKey);
      }
    }
  }
  let released = false;
  return () => {
    if (released) return;
    released = true;
    state.active = Math.max(0, state.active - 1);
  };
}

interface PublicCatalogRateState {
  windowStartedAt: number;
  requests: number;
  active: number;
}

function publicCatalogRateKey(request: FastifyRequest): string {
  const userAgent = request.headers["user-agent"] ?? "unknown";
  return createHash("sha256")
    .update(`${request.ip}\n${userAgent}\nmodel-catalog`)
    .digest("hex")
    .slice(0, 24);
}

function claimPublicCatalogRequest(
  states: Map<string, PublicCatalogRateState>,
  key: string,
  now = Date.now(),
): (() => void) | null {
  const windowMs = 10 * 60_000;
  const previous = states.get(key);
  const state = !previous || now - previous.windowStartedAt >= windowMs
    ? { windowStartedAt: now, requests: 0, active: 0 }
    : previous;
  if (state.requests >= 120 || state.active >= 2) return null;
  state.requests += 1;
  state.active += 1;
  states.set(key, state);
  if (states.size > 2_000) {
    for (const [candidateKey, candidate] of states) {
      if (now - candidate.windowStartedAt >= windowMs && candidate.active === 0) {
        states.delete(candidateKey);
      }
    }
  }
  let released = false;
  return () => {
    if (released) return;
    released = true;
    state.active = Math.max(0, state.active - 1);
  };
}

const jobIdParamsSchema = z.object({ jobId: z.string().min(1).max(128) }).strict();
const workerIdParamsSchema = z.object({ workerId: z.string().min(1).max(256) }).strict();
const requestedModelParamsSchema = z.object({
  modelId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
}).strict();

function parseBearerToken(header: string | undefined): string | undefined {
  const match = /^Bearer (.+)$/i.exec(header ?? "");
  return match?.[1]?.trim() || undefined;
}

function constantTimeEqual(received: string, expected: string): boolean {
  const left = Buffer.from(received);
  const right = Buffer.from(expected);
  return left.length === right.length && timingSafeEqual(left, right);
}

function writeMycelliosEvent(
  stream: NodeJS.WritableStream,
  event: JobStreamEvent,
  model: string,
  jobId: string,
): void {
  if (event.type === "accepted") {
    stream.write(
      `data: ${JSON.stringify({
        id: jobId,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1_000),
        model,
        choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
        x_network: {
          session_id: event.sessionId,
          route_class: event.route.routeClass,
          affinity_hit: event.route.affinityHit,
        },
      })}\n\n`,
    );
  } else if (event.type === "progress") {
    stream.write(
      `data: ${JSON.stringify({
        id: jobId,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1_000),
        model,
        choices: [{ index: 0, delta: {}, finish_reason: null }],
        x_network: {
          phase: event.phase,
          status_message: event.message,
          attempt: event.attempt,
          ...(event.workerId ? { affected_worker_id: event.workerId } : {}),
          ...(event.nodeId ? { affected_node_id: event.nodeId } : {}),
        },
      })}\n\n`,
    );
  } else if (event.type === "token") {
    stream.write(
      `data: ${JSON.stringify({
        id: jobId,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1_000),
        model,
        choices: [{ index: 0, delta: { content: event.token.text }, finish_reason: null }],
        x_network: { token_index: event.token.index },
      })}\n\n`,
    );
  } else if (event.type === "completed") {
    stream.write(
      `data: ${JSON.stringify({
        id: jobId,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1_000),
        model,
        choices: [{ index: 0, delta: {}, finish_reason: event.result.finishReason }],
        usage: {
          prompt_tokens: event.result.metrics.inputTokens,
          completion_tokens: event.result.metrics.outputTokens,
          total_tokens: event.result.metrics.inputTokens + event.result.metrics.outputTokens,
        },
        x_network: {
          ttft_ms: event.result.metrics.ttftMs,
          active_ms: event.result.metrics.activeMs,
          reused_kv_tokens: event.result.metrics.reusedKvTokens ?? 0,
          execution_trace: event.result.networkTrace ?? null,
        },
      })}\n\n`,
    );
  } else if (event.type === "failed") {
    stream.write(`data: ${JSON.stringify({ error: { code: event.code, message: event.message } })}\n\n`);
  }
}
