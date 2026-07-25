import { createHash, randomBytes, randomUUID } from "node:crypto";
import type { FastifyInstance } from "fastify";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import type WebSocket from "ws";
import { z } from "zod";
import { nativeBuildIdentitySchema, type NativeBuildIdentity } from "../contracts/build-identity.js";

const LEVELS = ["low", "balanced", "maximum"] as const;
const BACKENDS = ["webgpu", "cpu"] as const;
const SHA256 = /^[a-f0-9]{64}$/;
const DEFAULT_DISCONNECTED_RETENTION_MS = 60_000;

const expertManifestSchema = z
  .object({
    modelId: z.string().min(1).max(200),
    modelDigest: z.string().min(1).max(200),
    layer: z.number().int().nonnegative().max(100_000),
    expert: z.number().int().nonnegative().max(100_000),
    contentId: z.string().min(1).max(300),
    weightsHash: z.string().regex(SHA256),
    hiddenSize: z.number().int().positive().max(16_384),
    intermediateSize: z.number().int().positive().max(65_536),
    dtype: z.literal("float32"),
    activation: z.literal("silu"),
    canaryInputBase64: z.string().min(4).max(512 * 1024),
    canaryOutputBase64: z.string().min(4).max(512 * 1024),
  })
  .strict();

const artifactParamsSchema = z.object({ artifactId: z.string().regex(SHA256) }).strict();
const weightsParamsSchema = z.object({ weightsHash: z.string().regex(SHA256) }).strict();
const expertActionSchema = z.object({ artifactId: z.string().regex(SHA256) }).strict();
const expertExecuteSchema = expertActionSchema.extend({
  rows: z.number().int().positive().max(4_096),
  hiddenSize: z.number().int().positive().max(16_384),
  activationsBase64: z.string().min(4).max(8 * 1024 * 1024),
}).strict();

export const mobileRegistrationSchema = z
  .object({
    clientId: z.string().min(8).max(128).regex(/^[A-Za-z0-9._:-]+$/),
    name: z.string().min(1).max(80),
    region: z.string().min(1).max(80).default("auto"),
    platform: z.string().min(1).max(120),
    buildIdentity: nativeBuildIdentitySchema.optional(),
    backend: z.enum(BACKENDS),
    performanceLevel: z.enum(LEVELS),
    joinToken: z.string().max(512).optional(),
    capabilities: z
      .object({
        webgpu: z.boolean(),
        wasm: z.boolean(),
        hardwareConcurrency: z.number().int().positive().max(1_024),
        deviceMemoryGb: z.number().positive().max(1_024).optional(),
        gpuDescription: z.string().min(1).max(200).optional(),
        maxBufferSize: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER).optional(),
      })
      .strict(),
    benchmark: z
      .object({
        durationMs: z.number().positive().max(600_000),
        estimatedGflops: z.number().nonnegative().max(10_000_000),
        matrixSize: z.number().int().min(8).max(2_048),
      })
      .strict(),
  })
  .strict();

const connectionQuerySchema = z
  .object({
    workerId: z.string().uuid(),
    token: z.string().min(32).max(256),
  })
  .strict();

const mobileMessageSchema = z.discriminatedUnion("type", [
  z
    .object({
      v: z.literal(1),
      type: z.literal("mobile.heartbeat"),
      payload: z
        .object({
          visible: z.boolean(),
          wakeLock: z.boolean(),
          backend: z.enum(BACKENDS),
          estimatedGflops: z.number().nonnegative().max(10_000_000),
        })
        .strict(),
    })
    .strict(),
  z
    .object({
      v: z.literal(1),
      type: z.literal("work.request"),
      payload: z.object({}).strict(),
    })
    .strict(),
  z
    .object({
      v: z.literal(1),
      type: z.literal("compute.accept"),
      payload: z.object({ taskId: z.string().uuid(), leaseId: z.string().uuid() }).strict(),
    })
    .strict(),
  z
    .object({
      v: z.literal(1),
      type: z.literal("compute.result"),
      payload: z
        .object({
          taskId: z.string().uuid(),
          leaseId: z.string().uuid(),
          backend: z.enum(BACKENDS),
          durationMs: z.number().positive().max(600_000),
          estimatedGflops: z.number().nonnegative().max(10_000_000),
          samples: z.array(z.number().finite()).length(4),
        })
        .strict(),
    })
    .strict(),
  z
    .object({
      v: z.literal(1),
      type: z.literal("compute.fail"),
      payload: z
        .object({
          taskId: z.string().uuid(),
          leaseId: z.string().uuid(),
          message: z.string().min(1).max(300),
        })
        .strict(),
    })
    .strict(),
  z.object({
    v: z.literal(1),
    type: z.literal("expert.ready"),
    payload: z.object({
      taskId: z.string().uuid(),
      leaseId: z.string().uuid(),
      artifactId: z.string().regex(SHA256),
      canaryOutputBase64: z.string().min(4).max(512 * 1024),
      backend: z.enum(BACKENDS),
      durationMs: z.number().nonnegative().max(600_000),
    }).strict(),
  }).strict(),
  z.object({
    v: z.literal(1),
    type: z.literal("expert.result"),
    payload: z.object({
      taskId: z.string().uuid(),
      leaseId: z.string().uuid(),
      artifactId: z.string().regex(SHA256),
      rows: z.number().int().positive().max(4_096),
      hiddenSize: z.number().int().positive().max(16_384),
      outputBase64: z.string().min(4).max(8 * 1024 * 1024),
      backend: z.enum(BACKENDS),
      durationMs: z.number().nonnegative().max(600_000),
    }).strict(),
  }).strict(),
  z.object({
    v: z.literal(1),
    type: z.literal("expert.fail"),
    payload: z.object({
      taskId: z.string().uuid(),
      leaseId: z.string().uuid(),
      artifactId: z.string().regex(SHA256),
      message: z.string().min(1).max(500),
    }).strict(),
  }).strict(),
]);

export type MobileRegistration = z.infer<typeof mobileRegistrationSchema>;
export type MobileBackend = (typeof BACKENDS)[number];
export type PerformanceLevel = (typeof LEVELS)[number];

export interface MobileWorkerSnapshot {
  id: string;
  clientId: string;
  name: string;
  region: string;
  platform: string;
  buildIdentity?: NativeBuildIdentity;
  backend: MobileBackend;
  performanceLevel: PerformanceLevel;
  connected: boolean;
  status: "online" | "suspect" | "offline";
  visible: boolean;
  wakeLock: boolean;
  estimatedGflops: number;
  completedTasks: number;
  failedTasks: number;
  verifiedTasks: number;
  lastSeenAt: string;
  capabilities: MobileRegistration["capabilities"];
  residentExperts: Array<{
    artifactId: string;
    modelId: string;
    modelDigest: string;
    layer: number;
    expert: number;
    contentId: string;
    bytes: number;
  }>;
}

interface MatrixTask {
  taskId: string;
  leaseId: string;
  operation: "matrix-multiply";
  size: number;
  seed: number;
  issuedAt: number;
  accepted: boolean;
}

interface MobileWorkerState {
  registration: MobileRegistration;
  id: string;
  token: string;
  socket: WebSocket | null;
  connected: boolean;
  disconnectedAt: number | null;
  visible: boolean;
  wakeLock: boolean;
  lastSeenAt: number;
  estimatedGflops: number;
  completedTasks: number;
  failedTasks: number;
  verifiedTasks: number;
  pendingTask: MatrixTask | null;
  pendingExpertTask: PendingExpertTask | null;
  residentExperts: Set<string>;
  messageWindowAt: number;
  messagesInWindow: number;
}

export interface MobileComputeHubOptions {
  joinToken?: string | undefined;
  taskTimeoutMs?: number | undefined;
  expertArtifactsPath?: string | undefined;
  disconnectedRetentionMs?: number | undefined;
  onArtifactStored?(artifact: {
    id: string;
    localPath: string;
    storagePath: string;
    contentType: string;
    sha256: string;
    sizeBytes: number;
    metadata: Record<string, unknown>;
  }): void | Promise<void>;
}

type ExpertManifest = z.infer<typeof expertManifestSchema> & { artifactId: string };

interface PendingExpertTask {
  kind: "load" | "execute";
  taskId: string;
  leaseId: string;
  artifactId: string;
  issuedAt: number;
  rows?: number;
  hiddenSize?: number;
  resolve: (value: ExpertExecutionResult | MobileWorkerState) => void;
  reject: (error: Error) => void;
  timer: NodeJS.Timeout;
}

export interface ExpertExecutionResult {
  workerId: string;
  replicaWorkerIds?: string[];
  outputBase64: string;
  rows: number;
  hiddenSize: number;
  backend: MobileBackend;
  durationMs: number;
}

export class MobileComputeHub {
  private readonly workers = new Map<string, MobileWorkerState>();
  private readonly workerByClient = new Map<string, string>();
  private readonly taskTimeoutMs: number;
  private readonly disconnectedRetentionMs: number;
  private readonly expertArtifactsPath: string;
  private readonly expertManifests = new Map<string, ExpertManifest>();

  constructor(private readonly options: MobileComputeHubOptions = {}) {
    this.taskTimeoutMs = options.taskTimeoutMs ?? 45_000;
    this.disconnectedRetentionMs = options.disconnectedRetentionMs ?? DEFAULT_DISCONNECTED_RETENTION_MS;
    this.expertArtifactsPath = resolve(options.expertArtifactsPath ?? "runtime/mobile-experts");
    mkdirSync(this.expertArtifactsPath, { recursive: true });
  }

  attach(app: FastifyInstance): void {
    app.post("/mobile/v1/register", async (request, reply) => {
      const registration = mobileRegistrationSchema.parse(request.body);
      if (this.options.joinToken && registration.joinToken !== this.options.joinToken) {
        return reply.code(401).send({ error: { code: "invalid_join_token" } });
      }
      const worker = this.register(registration);
      return reply.code(201).send({
        workerId: worker.id,
        token: worker.token,
        protocolVersion: 1,
        websocketPath: "/mobile/v1/connect",
      });
    });

    app.get("/mobile/v1/workers", async () => ({ data: this.listWorkers() }));

    app.put("/internal/v1/mobile/experts/weights/:weightsHash", async (request, reply) => {
      const { weightsHash } = weightsParamsSchema.parse(request.params);
      if (!Buffer.isBuffer(request.body)) {
        return reply.code(415).send({ error: { code: "binary_body_required" } });
      }
      const body = request.body;
      if (body.length < 12 || body.length > 512 * 1024 * 1024) {
        return reply.code(413).send({ error: { code: "invalid_weight_size" } });
      }
      if (createHash("sha256").update(body).digest("hex") !== weightsHash) {
        return reply.code(400).send({ error: { code: "weight_hash_mismatch" } });
      }
      const localPath = this.weightPath(weightsHash);
      writeFileSync(localPath, body);
      await this.options.onArtifactStored?.({
        id: `mobile-weight-${weightsHash}`,
        localPath,
        storagePath: `mobile-experts/weights/${weightsHash}.bin`,
        contentType: "application/octet-stream",
        sha256: weightsHash,
        sizeBytes: body.length,
        metadata: { kind: "mobile-expert-weights", weightsHash },
      });
      return reply.code(201).send({ weightsHash, bytes: body.length });
    });

    app.post("/internal/v1/mobile/experts/register", async (request, reply) => {
      const manifest = expertManifestSchema.parse(request.body);
      const weights = readFileSync(this.weightPath(manifest.weightsHash));
      const expectedBytes = (2 * manifest.intermediateSize * manifest.hiddenSize
        + manifest.hiddenSize * manifest.intermediateSize) * Float32Array.BYTES_PER_ELEMENT;
      if (weights.length !== expectedBytes) {
        return reply.code(400).send({ error: { code: "weight_shape_mismatch", expectedBytes } });
      }
      const artifactId = createHash("sha256")
        .update(JSON.stringify(manifest))
        .digest("hex");
      const stored = { ...manifest, artifactId };
      this.expertManifests.set(artifactId, stored);
      const manifestBody = Buffer.from(JSON.stringify(stored, null, 2), "utf8");
      const manifestLocalPath = this.manifestPath(artifactId);
      writeFileSync(manifestLocalPath, manifestBody);
      await this.options.onArtifactStored?.({
        id: `mobile-manifest-${artifactId}`,
        localPath: manifestLocalPath,
        storagePath: `mobile-experts/manifests/${artifactId}.json`,
        contentType: "application/json",
        sha256: createHash("sha256").update(manifestBody).digest("hex"),
        sizeBytes: manifestBody.length,
        metadata: {
          kind: "mobile-expert-manifest",
          artifactId,
          weightsHash: manifest.weightsHash,
        },
      });
      return reply.code(201).send({ artifactId });
    });

    app.get("/mobile/v1/experts/:artifactId/manifest", async (request, reply) => {
      const { artifactId } = artifactParamsSchema.parse(request.params);
      const manifest = this.expertManifests.get(artifactId) ?? this.loadManifest(artifactId);
      if (!manifest) return reply.code(404).send({ error: { code: "artifact_not_found" } });
      return manifest;
    });

    app.get("/mobile/v1/experts/weights/:weightsHash", async (request, reply) => {
      const { weightsHash } = weightsParamsSchema.parse(request.params);
      try {
        const body = readFileSync(this.weightPath(weightsHash));
        reply.type("application/octet-stream");
        return reply.send(body);
      } catch {
        return reply.code(404).send({ error: { code: "weights_not_found" } });
      }
    });

    app.post("/internal/v1/mobile/experts/prepare", async (request, reply) => {
      const { artifactId } = expertActionSchema.parse(request.body);
      try {
        const worker = await this.prepareExpert(artifactId);
        return { artifactId, workerId: worker.id, resident: true };
      } catch (error) {
        return reply.code(503).send({ error: { code: "mobile_expert_unavailable", message: errorText(error) } });
      }
    });

    app.post("/internal/v1/mobile/experts/execute", { bodyLimit: 10 * 1024 * 1024 }, async (request, reply) => {
      const input = expertExecuteSchema.parse(request.body);
      try {
        return await this.executeExpert(input);
      } catch (error) {
        return reply.code(503).send({ error: { code: "mobile_expert_failed", message: errorText(error) } });
      }
    });

    app.get("/mobile/v1/connect", { websocket: true }, (socket, request) => {
      const parsed = connectionQuerySchema.safeParse(request.query);
      if (!parsed.success) {
        socket.close(4400, "invalid connection parameters");
        return;
      }
      this.expireDisconnectedWorkers();
      const worker = this.workers.get(parsed.data.workerId);
      if (!worker || !constantTimeEqual(worker.token, parsed.data.token)) {
        socket.close(4401, "invalid worker credentials");
        return;
      }
      if (worker.socket && worker.socket !== socket) {
        worker.socket.close(4409, "superseded connection");
      }
      worker.socket = socket;
      worker.connected = true;
      worker.disconnectedAt = null;
      worker.visible = true;
      worker.lastSeenAt = Date.now();
      this.send(socket, "server.ready", {
        workerId: worker.id,
        verifiedTasks: worker.verifiedTasks,
        inferenceReady: worker.verifiedTasks > 0,
      });
      socket.on("message", (raw) => this.handleMessage(worker, raw.toString()));
      socket.on("close", (code, reason) => {
        this.disconnect(
          worker,
          socket,
          code === 1000 && reason.toString() === "user stopped contribution",
        );
      });
      socket.on("error", () => this.disconnect(worker, socket));
    });
  }

  register(registration: MobileRegistration): MobileWorkerState {
    this.expireDisconnectedWorkers();
    const previousId = this.workerByClient.get(registration.clientId);
    const previous = previousId ? this.workers.get(previousId) : undefined;
    if (previous?.socket) previous.socket.close(4409, "worker registered again");
    if (previousId) this.workers.delete(previousId);

    const worker: MobileWorkerState = {
      registration,
      id: randomUUID(),
      token: randomBytes(32).toString("base64url"),
      socket: null,
      connected: false,
      disconnectedAt: Date.now(),
      visible: true,
      wakeLock: false,
      lastSeenAt: Date.now(),
      estimatedGflops: registration.benchmark.estimatedGflops,
      completedTasks: previous?.completedTasks ?? 0,
      failedTasks: previous?.failedTasks ?? 0,
      verifiedTasks: previous?.verifiedTasks ?? 0,
      pendingTask: null,
      pendingExpertTask: null,
      residentExperts: new Set<string>(),
      messageWindowAt: Date.now(),
      messagesInWindow: 0,
    };
    this.workers.set(worker.id, worker);
    this.workerByClient.set(registration.clientId, worker.id);
    return worker;
  }

  listWorkers(): MobileWorkerSnapshot[] {
    this.expireDisconnectedWorkers();
    const now = Date.now();
    return [...this.workers.values()].filter((worker) => worker.connected).map((worker) => {
      const age = now - worker.lastSeenAt;
      const status = !worker.connected || age > 30_000
        ? "offline"
        : age > 15_000
          ? "suspect"
          : "online";
      return {
        id: worker.id,
        clientId: worker.registration.clientId,
        name: worker.registration.name,
        region: worker.registration.region,
        platform: worker.registration.platform,
        ...(worker.registration.buildIdentity
          ? { buildIdentity: worker.registration.buildIdentity }
          : {}),
        backend: worker.registration.backend,
        performanceLevel: worker.registration.performanceLevel,
        connected: worker.connected,
        status,
        visible: worker.visible,
        wakeLock: worker.wakeLock,
        estimatedGflops: worker.estimatedGflops,
        completedTasks: worker.completedTasks,
        failedTasks: worker.failedTasks,
        verifiedTasks: worker.verifiedTasks,
        lastSeenAt: new Date(worker.lastSeenAt).toISOString(),
        capabilities: worker.registration.capabilities,
        residentExperts: [...worker.residentExperts].flatMap((artifactId) => {
          const manifest = this.expertManifests.get(artifactId) ?? this.loadManifest(artifactId);
          return manifest ? [{
            artifactId,
            modelId: manifest.modelId,
            modelDigest: manifest.modelDigest,
            layer: manifest.layer,
            expert: manifest.expert,
            contentId: manifest.contentId,
            bytes: 3 * manifest.hiddenSize * manifest.intermediateSize * 4,
          }] : [];
        }),
      };
    });
  }

  connectedCount(): number {
    return this.listWorkers().filter((worker) => worker.connected).length;
  }

  onlineCount(): number {
    return this.listWorkers().filter((worker) => worker.status === "online").length;
  }

  removeWorker(workerId: string): boolean {
    const worker = this.workers.get(workerId);
    if (!worker) return false;
    this.workers.delete(workerId);
    if (this.workerByClient.get(worker.registration.clientId) === workerId) {
      this.workerByClient.delete(worker.registration.clientId);
    }
    worker.socket?.close(4000, "removed from mycellios panel");
    worker.socket = null;
    worker.connected = false;
    return true;
  }

  removeOfflineWorkers(): number {
    const offline = [...this.workers.values()].filter((worker) => !worker.connected);
    for (const worker of offline) this.removeWorker(worker.id);
    return offline.length;
  }

  expireDisconnectedWorkers(now = Date.now()): number {
    const expired = [...this.workers.values()].filter((worker) =>
      !worker.connected &&
      worker.disconnectedAt !== null &&
      now - worker.disconnectedAt >= this.disconnectedRetentionMs
    );
    for (const worker of expired) this.removeWorker(worker.id);
    return expired.length;
  }

  close(): void {
    for (const worker of this.workers.values()) {
      worker.socket?.close(1001, "coordinator shutting down");
      worker.socket = null;
      worker.connected = false;
    }
  }

  private handleMessage(worker: MobileWorkerState, raw: string): void {
    const now = Date.now();
    if (now - worker.messageWindowAt >= 1_000) {
      worker.messageWindowAt = now;
      worker.messagesInWindow = 0;
    }
    worker.messagesInWindow += 1;
    if (worker.messagesInWindow > 64 || Buffer.byteLength(raw, "utf8") > 10 * 1024 * 1024) {
      worker.socket?.close(4429, "message limit exceeded");
      return;
    }
    let decoded: unknown;
    try {
      decoded = JSON.parse(raw) as unknown;
    } catch {
      worker.socket?.close(4400, "invalid JSON");
      return;
    }
    const parsed = mobileMessageSchema.safeParse(decoded);
    if (!parsed.success) {
      worker.socket?.close(4400, "invalid message");
      return;
    }
    worker.lastSeenAt = now;
    const message = parsed.data;
    switch (message.type) {
      case "mobile.heartbeat":
        worker.visible = message.payload.visible;
        worker.wakeLock = message.payload.wakeLock;
        worker.estimatedGflops = message.payload.estimatedGflops;
        worker.registration.backend = message.payload.backend;
        break;
      case "work.request":
        this.offerWork(worker);
        break;
      case "compute.accept":
        if (this.matches(worker.pendingTask, message.payload)) worker.pendingTask.accepted = true;
        break;
      case "compute.result":
        this.completeWork(worker, message.payload);
        break;
      case "compute.fail":
        if (this.matches(worker.pendingTask, message.payload)) {
          worker.failedTasks += 1;
          worker.pendingTask = null;
        }
        break;
      case "expert.ready":
        this.completeExpertLoad(worker, message.payload);
        break;
      case "expert.result":
        this.completeExpertExecution(worker, message.payload);
        break;
      case "expert.fail":
        this.failExpertTask(worker, message.payload);
        break;
    }
  }

  private offerWork(worker: MobileWorkerState): void {
    if (!worker.socket || worker.socket.readyState !== worker.socket.OPEN || !worker.visible) return;
    // Matrix work is an admission check, not a permanent workload. Once one
    // result has been independently verified, this worker stays idle and is
    // reserved for real model-expert inference.
    if (worker.verifiedTasks > 0 || worker.pendingExpertTask) return;
    if (worker.pendingTask) {
      if (Date.now() - worker.pendingTask.issuedAt <= this.taskTimeoutMs) return;
      worker.failedTasks += 1;
      worker.pendingTask = null;
    }
    const size = matrixSize(worker.registration.performanceLevel, worker.registration.backend);
    const task: MatrixTask = {
      taskId: randomUUID(),
      leaseId: randomUUID(),
      operation: "matrix-multiply",
      size,
      seed: Math.floor(Math.random() * 1_000_000),
      issuedAt: Date.now(),
      accepted: false,
    };
    worker.pendingTask = task;
    this.send(worker.socket, "compute.offer", {
      taskId: task.taskId,
      leaseId: task.leaseId,
      operation: task.operation,
      size: task.size,
      seed: task.seed,
      deadlineAt: Date.now() + this.taskTimeoutMs,
    });
  }

  private completeWork(
    worker: MobileWorkerState,
    payload: Extract<z.infer<typeof mobileMessageSchema>, { type: "compute.result" }>["payload"],
  ): void {
    const task = worker.pendingTask;
    if (!this.matches(task, payload)) return;
    const expected = expectedMatrixSamples(task.size, task.seed);
    const verified = payload.samples.every(
      (value, index) => Math.abs(value - (expected[index] ?? Number.POSITIVE_INFINITY)) < 0.02,
    );
    if (!verified) {
      worker.failedTasks += 1;
      worker.pendingTask = null;
      this.send(worker.socket, "compute.rejected", {
        taskId: task.taskId,
        reason: "verification_failed",
      });
      return;
    }
    worker.completedTasks += 1;
    worker.verifiedTasks += 1;
    worker.estimatedGflops = payload.estimatedGflops;
    worker.registration.backend = payload.backend;
    worker.pendingTask = null;
    this.send(worker.socket, "compute.verified", {
      taskId: task.taskId,
      completedTasks: worker.completedTasks,
      verifiedTasks: worker.verifiedTasks,
      inferenceReady: true,
    });
  }

  private async prepareExpert(
    artifactId: string,
    excludedWorkerIds: ReadonlySet<string> = new Set(),
  ): Promise<MobileWorkerState> {
    const manifest = this.expertManifests.get(artifactId) ?? this.loadManifest(artifactId);
    if (!manifest) throw new Error(`expert artifact ${artifactId} is not registered`);
    const resident = [...this.workers.values()].find(
      (worker) => !excludedWorkerIds.has(worker.id)
        && this.usableForExpert(worker)
        && worker.residentExperts.has(artifactId),
    );
    if (resident) return resident;
    const worker = [...this.workers.values()].find(
      (candidate) => !excludedWorkerIds.has(candidate.id) && this.usableForExpert(candidate),
    );
    if (!worker) throw new Error("two distinct validated visible mobile workers are required for replicated expert consensus");
    this.cancelMatrixWork(worker);
    return await new Promise<MobileWorkerState>((resolvePromise, rejectPromise) => {
      const taskId = randomUUID();
      const leaseId = randomUUID();
      const timer = setTimeout(() => {
        if (worker.pendingExpertTask?.taskId !== taskId) return;
        worker.pendingExpertTask = null;
        worker.failedTasks += 1;
        rejectPromise(new Error("mobile expert load timed out"));
      }, this.taskTimeoutMs);
      timer.unref();
      worker.pendingExpertTask = {
        kind: "load",
        taskId,
        leaseId,
        artifactId,
        issuedAt: Date.now(),
        resolve: (value) => resolvePromise(value as MobileWorkerState),
        reject: rejectPromise,
        timer,
      };
      this.send(worker.socket, "expert.load", {
        taskId,
        leaseId,
        artifactId,
        manifestUrl: `/mobile/v1/experts/${artifactId}/manifest`,
        deadlineAt: Date.now() + this.taskTimeoutMs,
      });
    });
  }

  private async executeExpert(input: z.infer<typeof expertExecuteSchema>): Promise<ExpertExecutionResult> {
    const manifest = this.expertManifests.get(input.artifactId) ?? this.loadManifest(input.artifactId);
    if (!manifest) throw new Error(`expert artifact ${input.artifactId} is not registered`);
    if (input.hiddenSize !== manifest.hiddenSize) throw new Error("activation hidden size mismatch");
    const activations = decodeFloat32(input.activationsBase64);
    if (activations.length !== input.rows * input.hiddenSize || !allFinite(activations)) {
      throw new Error("activation payload does not match its declared shape");
    }
    const primary = await this.prepareExpert(input.artifactId);
    const replica = await this.prepareExpert(input.artifactId, new Set([primary.id]));
    let results: [ExpertExecutionResult, ExpertExecutionResult];
    try {
      results = await Promise.all([
        this.executeExpertOnWorker(primary, input),
        this.executeExpertOnWorker(replica, input),
      ]);
    } catch (error) {
      this.rejectExpertConsensus([primary, replica], input.artifactId, "replica_failed");
      throw error;
    }
    const [primaryResult, replicaResult] = results;
    const primaryOutput = decodeFloat32(primaryResult.outputBase64);
    const replicaOutput = decodeFloat32(replicaResult.outputBase64);
    if (!floatTensorsAgree(primaryOutput, replicaOutput)) {
      this.rejectExpertConsensus([primary, replica], input.artifactId, "replica_mismatch");
      throw new Error("mobile expert replicas returned different tensors");
    }
    for (const [worker, result] of [[primary, primaryResult], [replica, replicaResult]] as const) {
      worker.completedTasks += 1;
      worker.verifiedTasks += 1;
      worker.registration.backend = result.backend;
      this.send(worker.socket, "expert.verified", {
        artifactId: input.artifactId,
        phase: "consensus",
        replicas: 2,
        verifiedTasks: worker.verifiedTasks,
      });
    }
    return {
      ...primaryResult,
      replicaWorkerIds: [primary.id, replica.id],
    };
  }

  private async executeExpertOnWorker(
    worker: MobileWorkerState,
    input: z.infer<typeof expertExecuteSchema>,
  ): Promise<ExpertExecutionResult> {
    return await new Promise<ExpertExecutionResult>((resolvePromise, rejectPromise) => {
      const taskId = randomUUID();
      const leaseId = randomUUID();
      const timer = setTimeout(() => {
        if (worker.pendingExpertTask?.taskId !== taskId) return;
        worker.pendingExpertTask = null;
        worker.residentExperts.delete(input.artifactId);
        worker.failedTasks += 1;
        rejectPromise(new Error("mobile expert execution timed out"));
      }, this.taskTimeoutMs);
      timer.unref();
      worker.pendingExpertTask = {
        kind: "execute",
        taskId,
        leaseId,
        artifactId: input.artifactId,
        issuedAt: Date.now(),
        rows: input.rows,
        hiddenSize: input.hiddenSize,
        resolve: (value) => resolvePromise(value as ExpertExecutionResult),
        reject: rejectPromise,
        timer,
      };
      this.send(worker.socket, "expert.execute", {
        taskId,
        leaseId,
        artifactId: input.artifactId,
        rows: input.rows,
        hiddenSize: input.hiddenSize,
        activationsBase64: input.activationsBase64,
        deadlineAt: Date.now() + this.taskTimeoutMs,
      });
    });
  }

  private rejectExpertConsensus(
    workers: readonly MobileWorkerState[],
    artifactId: string,
    reason: string,
  ): void {
    for (const worker of workers) {
      const pending = worker.pendingExpertTask;
      if (pending?.artifactId === artifactId) {
        clearTimeout(pending.timer);
        worker.pendingExpertTask = null;
        pending.reject(new Error(`replicated expert consensus cancelled: ${reason}`));
      }
      if (worker.residentExperts.delete(artifactId)) worker.failedTasks += 1;
      this.send(worker.socket, "expert.rejected", { artifactId, reason });
    }
  }

  private completeExpertLoad(
    worker: MobileWorkerState,
    payload: Extract<z.infer<typeof mobileMessageSchema>, { type: "expert.ready" }>["payload"],
  ): void {
    const pending = worker.pendingExpertTask;
    if (!this.matchesExpert(pending, payload) || pending.kind !== "load") return;
    const manifest = this.expertManifests.get(pending.artifactId) ?? this.loadManifest(pending.artifactId);
    const expected = manifest ? decodeFloat32(manifest.canaryOutputBase64) : new Float32Array();
    const actual = decodeFloat32(payload.canaryOutputBase64);
    const verified = expected.length > 0 && actual.length === expected.length
      && [...actual].every((value, index) => Number.isFinite(value)
        && Math.abs(value - (expected[index] ?? Number.POSITIVE_INFINITY)) <= 2e-4);
    clearTimeout(pending.timer);
    worker.pendingExpertTask = null;
    if (!verified) {
      worker.failedTasks += 1;
      pending.reject(new Error("mobile expert canary verification failed"));
      this.send(worker.socket, "expert.rejected", { artifactId: pending.artifactId, reason: "canary_failed" });
      return;
    }
    worker.residentExperts.add(pending.artifactId);
    worker.registration.backend = payload.backend;
    pending.resolve(worker);
    this.send(worker.socket, "expert.verified", { artifactId: pending.artifactId, phase: "loaded" });
  }

  private completeExpertExecution(
    worker: MobileWorkerState,
    payload: Extract<z.infer<typeof mobileMessageSchema>, { type: "expert.result" }>["payload"],
  ): void {
    const pending = worker.pendingExpertTask;
    if (!this.matchesExpert(pending, payload) || pending.kind !== "execute") return;
    const output = decodeFloat32(payload.outputBase64);
    const valid = payload.rows === pending.rows && payload.hiddenSize === pending.hiddenSize
      && output.length === payload.rows * payload.hiddenSize && allFinite(output);
    clearTimeout(pending.timer);
    worker.pendingExpertTask = null;
    if (!valid) {
      worker.residentExperts.delete(pending.artifactId);
      worker.failedTasks += 1;
      pending.reject(new Error("mobile expert returned an invalid tensor"));
      return;
    }
    pending.resolve({
      workerId: worker.id,
      outputBase64: payload.outputBase64,
      rows: payload.rows,
      hiddenSize: payload.hiddenSize,
      backend: payload.backend,
      durationMs: payload.durationMs,
    });
  }

  private failExpertTask(
    worker: MobileWorkerState,
    payload: Extract<z.infer<typeof mobileMessageSchema>, { type: "expert.fail" }>["payload"],
  ): void {
    const pending = worker.pendingExpertTask;
    if (!this.matchesExpert(pending, payload)) return;
    clearTimeout(pending.timer);
    worker.pendingExpertTask = null;
    worker.residentExperts.delete(pending.artifactId);
    worker.failedTasks += 1;
    pending.reject(new Error(payload.message));
  }

  private usableForExpert(worker: MobileWorkerState): boolean {
    return Boolean(
      worker.connected && worker.visible && worker.socket
      && worker.socket.readyState === worker.socket.OPEN && worker.verifiedTasks > 0
      && !worker.pendingExpertTask,
    );
  }

  private cancelMatrixWork(worker: MobileWorkerState): void {
    if (!worker.pendingTask) return;
    this.send(worker.socket, "compute.cancel", { taskId: worker.pendingTask.taskId });
    worker.pendingTask = null;
  }

  private matchesExpert(
    task: PendingExpertTask | null,
    payload: { taskId: string; leaseId: string; artifactId: string },
  ): task is PendingExpertTask {
    return Boolean(task && task.taskId === payload.taskId && task.leaseId === payload.leaseId
      && task.artifactId === payload.artifactId);
  }

  private weightPath(weightsHash: string): string {
    return resolve(this.expertArtifactsPath, `${weightsHash}.bin`);
  }

  private manifestPath(artifactId: string): string {
    return resolve(this.expertArtifactsPath, `${artifactId}.json`);
  }

  private loadManifest(artifactId: string): ExpertManifest | null {
    try {
      const parsed = expertManifestSchema.extend({ artifactId: z.literal(artifactId) })
        .parse(JSON.parse(readFileSync(this.manifestPath(artifactId), "utf8")));
      this.expertManifests.set(artifactId, parsed);
      return parsed;
    } catch {
      return null;
    }
  }

  private matches(
    task: MatrixTask | null,
    payload: { taskId: string; leaseId: string },
  ): task is MatrixTask {
    return Boolean(task && task.taskId === payload.taskId && task.leaseId === payload.leaseId);
  }

  private disconnect(worker: MobileWorkerState, socket: WebSocket, voluntary = false): void {
    if (worker.socket !== socket) return;
    worker.socket = null;
    worker.connected = false;
    worker.disconnectedAt = Date.now();
    worker.visible = false;
    worker.wakeLock = false;
    worker.pendingTask = null;
    worker.residentExperts.clear();
    if (worker.pendingExpertTask) {
      clearTimeout(worker.pendingExpertTask.timer);
      worker.pendingExpertTask.reject(new Error("mobile worker disconnected"));
      worker.pendingExpertTask = null;
    }
    worker.lastSeenAt = worker.disconnectedAt;
    if (voluntary) {
      this.workers.delete(worker.id);
      if (this.workerByClient.get(worker.registration.clientId) === worker.id) {
        this.workerByClient.delete(worker.registration.clientId);
      }
    }
  }

  private send(socket: WebSocket | null, type: string, payload: unknown): void {
    if (!socket || socket.readyState !== socket.OPEN) return;
    socket.send(JSON.stringify({ v: 1, type, payload }));
  }
}

export function matrixValueA(row: number, column: number, seed: number): number {
  return ((row * 3 + column * 5 + seed) % 31 - 15) / 16;
}

export function matrixValueB(row: number, column: number, seed: number): number {
  return ((row * 7 + column * 11 + seed * 3) % 29 - 14) / 16;
}

export function matrixSamplePositions(size: number): Array<readonly [number, number]> {
  return [
    [0, 0],
    [Math.floor(size / 2), Math.floor(size / 3)],
    [size - 1, size - 1],
    [Math.floor(size / 3), Math.max(0, size - 2)],
  ];
}

export function expectedMatrixSamples(size: number, seed: number): number[] {
  return matrixSamplePositions(size).map(([row, column]) => {
    let sum = 0;
    for (let index = 0; index < size; index += 1) {
      sum += matrixValueA(row, index, seed) * matrixValueB(index, column, seed);
    }
    return sum;
  });
}

function matrixSize(level: PerformanceLevel, backend: MobileBackend): number {
  if (backend === "cpu") return level === "low" ? 48 : level === "balanced" ? 64 : 80;
  return level === "low" ? 96 : level === "balanced" ? 128 : 160;
}

function constantTimeEqual(left: string, right: string): boolean {
  if (left.length !== right.length) return false;
  let difference = 0;
  for (let index = 0; index < left.length; index += 1) {
    difference |= left.charCodeAt(index) ^ right.charCodeAt(index);
  }
  return difference === 0;
}

function decodeFloat32(value: string): Float32Array {
  const buffer = Buffer.from(value, "base64");
  if (buffer.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) return new Float32Array();
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
}

function allFinite(values: Float32Array): boolean {
  for (const value of values) if (!Number.isFinite(value)) return false;
  return true;
}

function floatTensorsAgree(left: Float32Array, right: Float32Array): boolean {
  if (left.length === 0 || left.length !== right.length) return false;
  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index] ?? Number.NaN;
    const rightValue = right[index] ?? Number.NaN;
    if (!Number.isFinite(leftValue) || !Number.isFinite(rightValue)) return false;
    const scale = Math.max(Math.abs(leftValue), Math.abs(rightValue));
    if (Math.abs(leftValue - rightValue) > 2e-4 + 2e-4 * scale) return false;
  }
  return true;
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
