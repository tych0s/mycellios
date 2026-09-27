import { createHash, randomUUID } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import type { FastifyInstance, FastifyRequest } from "fastify";
import { z } from "zod";
import {
  browserLayerExecuteSchema,
  browserLayerManifestInputSchema,
  browserLayerManifestSchema,
  tensorElements,
  type BrowserLayerManifest,
} from "../contracts/browser-layer.js";

const hashSchema = z.string().regex(/^[0-9a-f]{64}$/);
const actionSchema = z.object({ artifactId: hashSchema }).strict();
const lookupSchema = z.object({
  modelDigest: z.string().regex(/^sha256:[0-9a-f]{64}$/),
  layerStart: z.coerce.number().int().nonnegative().max(100_000),
  layerEnd: z.coerce.number().int().positive().max(100_001),
}).strict().refine((value) => value.layerStart < value.layerEnd);
const resetSchema = z.object({ artifactId: hashSchema, requestId: z.string().uuid() }).strict();
const resultSchema = z.object({
  taskId: z.string().uuid(), leaseId: z.string().uuid(), artifactId: hashSchema,
  requestId: z.string().uuid(), position: z.number().int().nonnegative(),
  tokens: z.number().int().positive(),
  outputBase64: z.string().min(4).max(8 * 1024 * 1024),
  stateBase64: z.record(z.string(), z.string().min(4).max(8 * 1024 * 1024)),
  backend: z.enum(["webgpu", "cpu"]),
  durationMs: z.number().nonnegative().max(600_000),
}).strict();
const readySchema = z.object({
  taskId: z.string().uuid(), leaseId: z.string().uuid(), artifactId: hashSchema,
  outputBase64: z.string().min(4).max(8 * 1024 * 1024),
  backend: z.enum(["webgpu", "cpu"]),
  durationMs: z.number().nonnegative().max(600_000),
}).strict();
const failSchema = z.object({
  taskId: z.string().uuid(), leaseId: z.string().uuid(), artifactId: hashSchema,
  message: z.string().min(1).max(500),
}).strict();

export interface BrowserLayerPeer {
  id: string;
  token: string;
  backend: "webgpu" | "cpu";
  connected: boolean;
  visible: boolean;
  validated: boolean;
  busy: boolean;
}

export interface BrowserLayerHubOptions {
  artifactDirectory: string;
  peers(): BrowserLayerPeer[];
  send(peerId: string, type: string, payload: unknown): void;
  onExecutionVerified?(peerId: string, layer: number, backend: "webgpu" | "cpu"): void;
  timeoutMs?: number;
  onArtifactStored?(artifact: {
    id: string; localPath: string; storagePath: string; contentType: string;
    sha256: string; sizeBytes: number; metadata: Record<string, unknown>;
  }): void | Promise<void>;
}

interface Pending {
  peerId: string;
  artifactId: string;
  leaseId: string;
  kind: "load" | "execute";
  resolve(value: unknown): void;
  reject(error: Error): void;
  timer: NodeJS.Timeout;
}

interface RequestOwner {
  artifactId: string;
  peerId: string;
  nextPosition: number;
  updatedAt: number;
}

/** Coordinator bridge from authenticated browser workers to model stages. */
export class BrowserLayerHub {
  private readonly manifests = new Map<string, BrowserLayerManifest>();
  private readonly resident = new Map<string, string>();
  private readonly requests = new Map<string, RequestOwner>();
  private readonly pending = new Map<string, Pending>();
  private readonly loading = new Map<string, Promise<BrowserLayerPeer>>();
  private readonly executing = new Set<string>();
  private readonly artifactDirectory: string;
  private readonly timeoutMs: number;

  constructor(private readonly options: BrowserLayerHubOptions) {
    this.artifactDirectory = resolve(options.artifactDirectory);
    this.timeoutMs = options.timeoutMs ?? 45_000;
    mkdirSync(this.artifactDirectory, { recursive: true });
  }

  attach(app: FastifyInstance): void {
    app.put("/internal/v1/mobile/layers/graphs/:graphSha256", {
      bodyLimit: 512 * 1024 * 1024,
    }, async (request, reply) => {
      const { graphSha256 } = z.object({ graphSha256: hashSchema }).parse(request.params);
      if (!Buffer.isBuffer(request.body) || request.body.length < 1) {
        return reply.code(415).send({ error: { code: "binary_graph_required" } });
      }
      const graph = request.body;
      if (createHash("sha256").update(graph).digest("hex") !== graphSha256) {
        return reply.code(400).send({ error: { code: "graph_hash_mismatch" } });
      }
      const localPath = this.graphPath(graphSha256);
      writeFileSync(localPath, graph);
      await this.options.onArtifactStored?.({
        id: `mobile-layer-graph-${graphSha256}`, localPath,
        storagePath: `mobile-experts/layers/graphs/${graphSha256}.onnx`,
        contentType: "application/octet-stream", sha256: graphSha256,
        sizeBytes: graph.length, metadata: { kind: "mobile-layer-graph", graphSha256 },
      });
      return reply.code(201).send({ graphSha256, bytes: graph.length });
    });
    app.post("/internal/v1/mobile/layers/register", async (request, reply) => {
      const input = browserLayerManifestInputSchema.parse(request.body);
      try {
        readFileSync(this.graphPath(input.graphSha256));
      } catch {
        return reply.code(404).send({ error: { code: "graph_not_found" } });
      }
      const artifactId = createHash("sha256").update(JSON.stringify(input)).digest("hex");
      const manifest = browserLayerManifestSchema.parse({ ...input, artifactId });
      this.manifests.set(artifactId, manifest);
      const localPath = this.manifestPath(artifactId);
      const body = Buffer.from(JSON.stringify(manifest), "utf8");
      writeFileSync(localPath, body);
      await this.options.onArtifactStored?.({
        id: `mobile-layer-manifest-${artifactId}`, localPath,
        storagePath: `mobile-experts/layers/manifests/${artifactId}.json`,
        contentType: "application/json",
        sha256: createHash("sha256").update(body).digest("hex"),
        sizeBytes: body.length,
        metadata: { kind: "mobile-layer-manifest", artifactId,
          graphSha256: manifest.graphSha256 },
      });
      // Registration is the explicit activation point for a model layer. The
      // pointer survives coordinator restarts and keeps older artifacts from
      // becoming ambiguous when an operator republishes the same layer.
      const activePath = this.activePath(manifest.modelDigest, manifest.layer);
      const activeBody = Buffer.from(artifactId, "utf8");
      writeFileSync(activePath, activeBody);
      await this.options.onArtifactStored?.({
        id: `mobile-layer-active-${manifest.modelDigest.slice(7)}-${manifest.layer}`,
        localPath: activePath,
        storagePath: `mobile-experts/layers/active/${manifest.modelDigest.slice(7)}-${manifest.layer}.active`,
        contentType: "text/plain", sha256: createHash("sha256").update(activeBody).digest("hex"),
        sizeBytes: activeBody.length,
        metadata: { kind: "mobile-layer-active", modelDigest: manifest.modelDigest,
          layer: manifest.layer, artifactId },
      });
      return reply.code(201).send({ artifactId });
    });
    app.get("/internal/v1/mobile/layers/lookup", async (request) => {
      const query = lookupSchema.parse(request.query);
      const ids = new Set<string>(this.manifests.keys());
      for (const entry of readdirSync(this.artifactDirectory)) {
        if (/^[0-9a-f]{64}\.json$/.test(entry)) ids.add(entry.slice(0, 64));
      }
      const matching = [...ids].map((id) => this.getManifest(id))
        .filter((manifest): manifest is BrowserLayerManifest => manifest !== null
          && manifest.modelDigest === query.modelDigest
          && manifest.layer >= query.layerStart && manifest.layer < query.layerEnd);
      const data = matching.filter((manifest) => {
        try {
          const active = readFileSync(this.activePath(manifest.modelDigest, manifest.layer), "utf8");
          return active === manifest.artifactId;
        } catch {
          return true;
        }
      })
        .map(({ artifactId, modelDigest, layer, maxContextTokens }) => ({
          artifactId, modelDigest, layer, maxContextTokens,
        }))
        .sort((left, right) => left.layer - right.layer
          || left.artifactId.localeCompare(right.artifactId));
      return { data };
    });
    app.get("/mobile/v1/layers/:artifactId/manifest", async (request, reply) => {
      const { artifactId } = actionSchema.parse(request.params);
      if (!this.authorized(request, artifactId)) return reply.code(401).send();
      const manifest = this.getManifest(artifactId);
      return manifest ?? reply.code(404).send({ error: { code: "layer_not_found" } });
    });
    app.get("/mobile/v1/layers/graphs/:graphSha256", async (request, reply) => {
      const { graphSha256 } = z.object({ graphSha256: hashSchema }).parse(request.params);
      const artifactId = [...this.resident.keys(), ...[...this.pending.values()]
        .filter((task) => task.kind === "load").map((task) => task.artifactId)]
        .find((id) => this.getManifest(id)?.graphSha256 === graphSha256);
      if (!artifactId || !this.authorized(request, artifactId)) return reply.code(401).send();
      try {
        reply.type("application/octet-stream");
        return reply.send(readFileSync(this.graphPath(graphSha256)));
      } catch {
        return reply.code(404).send({ error: { code: "graph_not_found" } });
      }
    });
    app.post("/internal/v1/mobile/layers/prepare", async (request, reply) => {
      const { artifactId } = actionSchema.parse(request.body);
      try {
        const peer = await this.prepare(artifactId);
        return { artifactId, workerId: peer.id, backend: peer.backend, resident: true };
      } catch (error) {
        return reply.code(503).send({ error: { code: "browser_layer_unavailable", message: errorText(error) } });
      }
    });
    app.post("/internal/v1/mobile/layers/execute", { bodyLimit: 12 * 1024 * 1024 },
      async (request, reply) => {
        const input = browserLayerExecuteSchema.parse(request.body);
        try {
          return await this.execute(input);
        } catch (error) {
          return reply.code(503).send({ error: { code: "browser_layer_failed", message: errorText(error) } });
        }
      });
    app.post("/internal/v1/mobile/layers/reset", async (request) => {
      const input = resetSchema.parse(request.body);
      this.reset(input.artifactId, input.requestId);
      return { reset: true };
    });
  }

  handleMessage(peerId: string, type: string, payload: unknown): void {
    if (type === "layer.fail") {
      const value = failSchema.safeParse(payload);
      if (!value.success) return;
      const pending = this.takePending(value.data.taskId, value.data.leaseId, peerId);
      pending?.reject(new Error(value.data.message));
      return;
    }
    if (type === "layer.ready") {
      const value = readySchema.safeParse(payload);
      if (!value.success) return;
      const pending = this.takePending(value.data.taskId, value.data.leaseId, peerId, "load");
      pending?.resolve(value.data);
      return;
    }
    if (type === "layer.result") {
      const value = resultSchema.safeParse(payload);
      if (!value.success) return;
      const pending = this.takePending(value.data.taskId, value.data.leaseId, peerId, "execute");
      pending?.resolve(value.data);
    }
  }

  disconnect(peerId: string): void {
    for (const [taskId, pending] of this.pending) {
      if (pending.peerId !== peerId) continue;
      clearTimeout(pending.timer);
      this.pending.delete(taskId);
      pending.reject(new Error("browser layer worker disconnected"));
    }
    for (const [artifactId, owner] of this.resident) {
      if (owner === peerId) this.resident.delete(artifactId);
    }
    for (const [requestId, owner] of this.requests) {
      if (owner.peerId === peerId) this.requests.delete(requestId);
    }
  }

  close(): void {
    for (const peer of this.options.peers()) this.disconnect(peer.id);
  }

  private authorized(request: FastifyRequest, artifactId: string): boolean {
    const authorization = request.headers.authorization;
    if (!authorization?.startsWith("Bearer ")) return false;
    const token = authorization.slice(7);
    return this.options.peers().some((peer) => peer.connected && peer.token === token
      && (this.resident.get(artifactId) === peer.id
        || [...this.pending.values()].some((task) => task.peerId === peer.id
          && task.artifactId === artifactId && task.kind === "load")));
  }

  private getManifest(artifactId: string): BrowserLayerManifest | null {
    const cached = this.manifests.get(artifactId);
    if (cached) return cached;
    try {
      const manifest = browserLayerManifestSchema.parse(
        JSON.parse(readFileSync(this.manifestPath(artifactId), "utf8")),
      );
      if (manifest.artifactId !== artifactId) return null;
      this.manifests.set(artifactId, manifest);
      return manifest;
    } catch {
      return null;
    }
  }

  private async prepare(artifactId: string): Promise<BrowserLayerPeer> {
    const loading = this.loading.get(artifactId);
    if (loading) return loading;
    const operation = this.load(artifactId);
    this.loading.set(artifactId, operation);
    try {
      return await operation;
    } finally {
      this.loading.delete(artifactId);
    }
  }

  private async load(artifactId: string): Promise<BrowserLayerPeer> {
    const manifest = this.getManifest(artifactId);
    if (!manifest) throw new Error("browser layer is not registered");
    const ownerId = this.resident.get(artifactId);
    const peers = this.options.peers().filter((peer) => peer.connected && peer.visible
      && peer.validated && !peer.busy
      && ![...this.pending.values()].some((task) => task.peerId === peer.id));
    const resident = peers.find((peer) => peer.id === ownerId);
    if (resident) return resident;
    const idle = peers.filter((peer) => ![...this.requests.values()]
      .some((request) => request.peerId === peer.id));
    const candidates = idle.sort((left, right) => {
      const leftScore = Number(left.backend === "webgpu") * 2
        + Number(![...this.resident.values()].includes(left.id));
      const rightScore = Number(right.backend === "webgpu") * 2
        + Number(![...this.resident.values()].includes(right.id));
      return rightScore - leftScore;
    });
    if (candidates.length === 0) throw new Error("no eligible browser worker for this model layer");
    const expected = decodeFloat32(manifest.canary.expectedBase64);
    let lastError: Error | null = null;
    for (const peer of candidates) {
      // Loading another artifact may evict this peer's prior graph. Never
      // retain a stale residency claim if its new canary fails.
      for (const [residentId, residentOwner] of this.resident) {
        if (residentOwner === peer.id && residentId !== artifactId) this.resident.delete(residentId);
      }
      try {
        const ready = readySchema.parse(await this.sendTask(peer, artifactId, "load", {
          artifactId,
          manifestUrl: `/mobile/v1/layers/${artifactId}/manifest`,
        }));
        if (ready.artifactId !== artifactId || ready.backend !== peer.backend) {
          throw new Error("browser layer backend or artifact changed during load");
        }
        const actual = decodeFloat32(ready.outputBase64);
        if (!sameTensor(actual, expected, 2e-4)) throw new Error("browser layer canary rejected");
        this.resident.set(artifactId, peer.id);
        return peer;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error("browser layer load failed");
      }
    }
    throw lastError ?? new Error("no eligible browser worker for this model layer");
  }

  private async execute(input: z.infer<typeof browserLayerExecuteSchema>): Promise<unknown> {
    if (Buffer.byteLength(JSON.stringify(input), "utf8") > 8 * 1024 * 1024) {
      throw new Error("browser layer activation exceeds the worker message limit");
    }
    for (const [requestId, owner] of this.requests) {
      if (Date.now() - owner.updatedAt > 5 * 60_000) this.reset(owner.artifactId, requestId);
    }
    const manifest = this.getManifest(input.artifactId);
    if (!manifest) throw new Error("browser layer is not registered");
    if (input.position + input.tokens > manifest.maxContextTokens) {
      throw new Error("browser layer context limit exceeded");
    }
    const hidden = decodeFloat32(input.hiddenBase64);
    if (hidden.length !== input.tokens * manifest.hidden.width || !allFinite(hidden)
      || Object.keys(input.auxiliaryBase64).sort().join("|")
        !== manifest.auxiliary.map((item) => item.name).sort().join("|")) {
      throw new Error("browser layer input tensor shape mismatch");
    }
    for (const item of manifest.auxiliary) {
      const values = decodeFloat32(input.auxiliaryBase64[item.name]!);
      if (values.length !== tensorElements(item.shape, input.tokens, input.position)
        || !allFinite(values)) throw new Error("browser layer auxiliary tensor shape mismatch");
    }
    if (this.executing.has(input.requestId)) {
      throw new Error("browser layer request already executing");
    }
    this.executing.add(input.requestId);
    try {
      return await this.executeLocked(input, manifest);
    } finally {
      this.executing.delete(input.requestId);
    }
  }

  private async executeLocked(input: z.infer<typeof browserLayerExecuteSchema>,
    manifest: BrowserLayerManifest): Promise<unknown> {
    const request = this.requests.get(input.requestId);
    if (request && (request.artifactId !== input.artifactId
      || request.nextPosition !== input.position)) {
      throw new Error("browser layer request sequence mismatch");
    }
    if (!request && input.position !== 0) throw new Error("browser layer request must start at zero");
    const peer = request
      ? this.options.peers().find((candidate) => candidate.id === request.peerId)
      : await this.prepare(input.artifactId);
    if (!peer?.connected || !peer.visible || !peer.validated || peer.busy) {
      this.reset(input.artifactId, input.requestId);
      throw new Error("assigned browser layer worker is unavailable");
    }
    const result = resultSchema.parse(await this.sendTask(peer, input.artifactId, "execute", input));
    if (result.artifactId !== input.artifactId || result.requestId !== input.requestId
      || result.position !== input.position || result.tokens !== input.tokens
      || result.backend !== peer.backend) {
      this.requests.delete(input.requestId);
      this.options.send(peer.id, "layer.reset", {
        artifactId: input.artifactId, requestId: input.requestId,
      });
      throw new Error("browser layer result does not match its lease");
    }
    const output = decodeFloat32(result.outputBase64);
    const stateNames = manifest.state.map((item) => item.name);
    const validState = Object.keys(result.stateBase64).sort().join("|")
      === stateNames.sort().join("|") && manifest.state.every((item) => {
        const values = decodeFloat32(result.stateBase64[item.name]!);
        return values.length === tensorElements(item.shape, input.tokens, input.tokens)
          && allFinite(values);
      });
    if (output.length !== input.tokens * manifest.hidden.width
      || !allFinite(output) || !validState) {
      this.requests.delete(input.requestId);
      this.options.send(peer.id, "layer.reset", {
        artifactId: input.artifactId, requestId: input.requestId,
      });
      throw new Error("browser layer result tensor is invalid");
    }
    this.requests.set(input.requestId, {
      artifactId: input.artifactId,
      peerId: peer.id,
      nextPosition: input.position + input.tokens,
      updatedAt: Date.now(),
    });
    this.options.onExecutionVerified?.(peer.id, manifest.layer, peer.backend);
    return { ...result, workerId: peer.id };
  }

  private reset(artifactId: string, requestId: string): void {
    const owner = this.requests.get(requestId);
    if (!owner || owner.artifactId !== artifactId) return;
    this.requests.delete(requestId);
    this.options.send(owner.peerId, "layer.reset", { artifactId, requestId });
  }

  private sendTask(peer: BrowserLayerPeer, artifactId: string,
    kind: "load" | "execute", payload: Record<string, unknown>): Promise<unknown> {
    if ([...this.pending.values()].some((task) => task.peerId === peer.id)) {
      return Promise.reject(new Error("browser layer worker is already executing"));
    }
    return new Promise((resolvePromise, rejectPromise) => {
      const taskId = randomUUID();
      const leaseId = randomUUID();
      const timer = setTimeout(() => {
        this.pending.delete(taskId);
        rejectPromise(new Error("browser layer task timed out"));
      }, this.timeoutMs);
      timer.unref();
      this.pending.set(taskId, {
        peerId: peer.id, artifactId, leaseId, kind,
        resolve: resolvePromise, reject: rejectPromise, timer,
      });
      this.options.send(peer.id, `layer.${kind === "load" ? "load" : "execute"}`, {
        ...payload, taskId, leaseId, deadlineAt: Date.now() + this.timeoutMs,
      });
    });
  }

  private takePending(taskId: string, leaseId: string, peerId: string,
    kind?: Pending["kind"]): Pending | null {
    const pending = this.pending.get(taskId);
    if (!pending || pending.leaseId !== leaseId || pending.peerId !== peerId
      || (kind && pending.kind !== kind)) return null;
    clearTimeout(pending.timer);
    this.pending.delete(taskId);
    return pending;
  }

  private graphPath(hash: string): string { return resolve(this.artifactDirectory, `${hash}.onnx`); }
  private manifestPath(id: string): string { return resolve(this.artifactDirectory, `${id}.json`); }
  private activePath(modelDigest: string, layer: number): string {
    return resolve(this.artifactDirectory, `${modelDigest.slice(7)}-${layer}.active`);
  }
}

function decodeFloat32(base64: string): Float32Array {
  const bytes = Buffer.from(base64, "base64");
  if (bytes.length % 4 !== 0) throw new Error("invalid float32 tensor length");
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

function allFinite(values: Float32Array): boolean {
  return values.every(Number.isFinite);
}

function sameTensor(actual: Float32Array, expected: Float32Array, tolerance: number): boolean {
  return actual.length === expected.length && allFinite(actual)
    && actual.every((value, index) => Math.abs(value - (expected[index] ?? Infinity)) <= tolerance);
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
