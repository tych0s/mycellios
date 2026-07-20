import { randomBytes, randomUUID } from "node:crypto";
import type { FastifyInstance } from "fastify";
import type WebSocket from "ws";
import { z } from "zod";

const LEVELS = ["low", "balanced", "maximum"] as const;
const BACKENDS = ["webgpu", "cpu"] as const;

export const mobileRegistrationSchema = z
  .object({
    clientId: z.string().min(8).max(128).regex(/^[A-Za-z0-9._:-]+$/),
    name: z.string().min(1).max(80),
    region: z.string().min(1).max(80).default("auto"),
    platform: z.string().min(1).max(120),
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
  visible: boolean;
  wakeLock: boolean;
  lastSeenAt: number;
  estimatedGflops: number;
  completedTasks: number;
  failedTasks: number;
  verifiedTasks: number;
  pendingTask: MatrixTask | null;
  messageWindowAt: number;
  messagesInWindow: number;
}

export interface MobileComputeHubOptions {
  joinToken?: string | undefined;
  taskTimeoutMs?: number | undefined;
}

export class MobileComputeHub {
  private readonly workers = new Map<string, MobileWorkerState>();
  private readonly workerByClient = new Map<string, string>();
  private readonly taskTimeoutMs: number;

  constructor(private readonly options: MobileComputeHubOptions = {}) {
    this.taskTimeoutMs = options.taskTimeoutMs ?? 45_000;
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

    app.get("/mobile/v1/connect", { websocket: true }, (socket, request) => {
      const parsed = connectionQuerySchema.safeParse(request.query);
      if (!parsed.success) {
        socket.close(4400, "invalid connection parameters");
        return;
      }
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
      worker.visible = true;
      worker.lastSeenAt = Date.now();
      this.send(socket, "server.ready", { workerId: worker.id });
      socket.on("message", (raw) => this.handleMessage(worker, raw.toString()));
      socket.on("close", () => this.disconnect(worker, socket));
      socket.on("error", () => this.disconnect(worker, socket));
    });
  }

  register(registration: MobileRegistration): MobileWorkerState {
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
      visible: true,
      wakeLock: false,
      lastSeenAt: Date.now(),
      estimatedGflops: registration.benchmark.estimatedGflops,
      completedTasks: previous?.completedTasks ?? 0,
      failedTasks: previous?.failedTasks ?? 0,
      verifiedTasks: previous?.verifiedTasks ?? 0,
      pendingTask: null,
      messageWindowAt: Date.now(),
      messagesInWindow: 0,
    };
    this.workers.set(worker.id, worker);
    this.workerByClient.set(registration.clientId, worker.id);
    return worker;
  }

  listWorkers(): MobileWorkerSnapshot[] {
    const now = Date.now();
    return [...this.workers.values()].map((worker) => {
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
      };
    });
  }

  connectedCount(): number {
    return this.listWorkers().filter((worker) => worker.connected).length;
  }

  onlineCount(): number {
    return this.listWorkers().filter((worker) => worker.status === "online").length;
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
    if (worker.messagesInWindow > 64 || Buffer.byteLength(raw, "utf8") > 64 * 1024) {
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
    }
  }

  private offerWork(worker: MobileWorkerState): void {
    if (!worker.socket || worker.socket.readyState !== worker.socket.OPEN || !worker.visible) return;
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
    });
  }

  private matches(
    task: MatrixTask | null,
    payload: { taskId: string; leaseId: string },
  ): task is MatrixTask {
    return Boolean(task && task.taskId === payload.taskId && task.leaseId === payload.leaseId);
  }

  private disconnect(worker: MobileWorkerState, socket: WebSocket): void {
    if (worker.socket !== socket) return;
    worker.socket = null;
    worker.connected = false;
    worker.visible = false;
    worker.wakeLock = false;
    worker.pendingTask = null;
    worker.lastSeenAt = Date.now();
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
