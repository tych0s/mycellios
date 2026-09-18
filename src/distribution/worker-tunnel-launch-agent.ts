import { createHash } from "node:crypto";
import type { WorkerEnvelope } from "../contracts/types.js";
import type { RuntimeProxyHandle } from "../contracts/runtime-transport.js";
import type {
  LaunchAgent,
  LaunchAgentStartRequest,
  LaunchCapturedOutput,
  LaunchProcessExit,
  LaunchProcessHandle,
  RuntimePreparationProgressEvent,
} from "./launch-supervisor.js";
import type { PythonPipelineLaunchDescription } from "./python-launcher.js";

interface Deferred<T> {
  promise: Promise<T>;
  resolve(value: T): void;
  reject(error: unknown): void;
}

interface PendingStart {
  ready: Deferred<void>;
  exited: Deferred<LaunchProcessExit>;
  output: LaunchCapturedOutput;
}

/** Minimal authenticated transport port required by the Engine launcher. */
export interface WorkerRuntimeTransport {
  on(event: "envelope", listener: (envelope: WorkerEnvelope) => void): unknown;
  on(event: "disconnect", listener: (workerId: string) => void): unknown;
  off(event: "envelope", listener: (envelope: WorkerEnvelope) => void): unknown;
  off(event: "disconnect", listener: (workerId: string) => void): unknown;
  send(workerId: string, type: string, payload: unknown): boolean;
  createRuntimeProxy(workerId: string, targetPort: number): Promise<RuntimeProxyHandle>;
}

/** Runs compiler-sealed Python stages through the already authenticated worker socket. */
export class WorkerTunnelLaunchAgent implements LaunchAgent {
  readonly id: string;
  private readonly pending = new Map<string, PendingStart>();
  private readonly preparations = new Map<string, Deferred<void>>();
  private prepared = false;
  private closed = false;
  private connectionGeneration = 0;
  private readonly preparationListeners = new Set<
    (event: RuntimePreparationProgressEvent) => void
  >();
  private readonly runtimeProxies = new Set<RuntimeProxyHandle>();
  private readonly onEnvelope = (envelope: WorkerEnvelope) => this.handleEnvelope(envelope);
  private readonly onDisconnect = (workerId: string) => {
    if (workerId !== this.workerId) return;
    this.connectionGeneration += 1;
    const error = new Error(`distributed_worker_disconnected:${workerId}`);
    for (const preparation of this.preparations.values()) preparation.reject(error);
    for (const start of this.pending.values()) {
      start.ready.reject(error);
      start.exited.reject(error);
    }
    this.preparations.clear();
    this.pending.clear();
    this.prepared = false;
  };

  constructor(
    private readonly hub: WorkerRuntimeTransport,
    private readonly workerId: string,
    private readonly nodeId: string,
    private readonly description: PythonPipelineLaunchDescription,
    private readonly timeoutMs = 3_600_000,
  ) {
    this.id = `worker-tunnel:${workerId}`;
    hub.on("envelope", this.onEnvelope);
    hub.on("disconnect", this.onDisconnect);
  }

  async start(request: LaunchAgentStartRequest, signal: AbortSignal): Promise<LaunchProcessHandle> {
    if (signal.aborted) throw abortError(signal);
    if (this.closed) throw new Error(`worker_tunnel_closed:${this.workerId}`);
    if (request.nodeId !== this.nodeId) throw new Error("worker_tunnel_launch_node_mismatch");
    const generation = this.connectionGeneration;
    await this.prepare(signal);
    if (signal.aborted) throw abortError(signal);
    if (this.closed) throw new Error(`worker_tunnel_closed:${this.workerId}`);
    if (generation !== this.connectionGeneration) throw new Error(`distributed_worker_disconnected:${this.workerId}`);
    const requestId = requestIdentity(request);
    if (this.pending.has(requestId)) throw new Error("worker_tunnel_duplicate_launch");
    const start: PendingStart = {
      ready: deferred<void>(),
      exited: deferred<LaunchProcessExit>(),
      output: { stdout: "", stderr: "", stdoutTruncated: false, stderrTruncated: false },
    };
    this.pending.set(requestId, start);
    if (!this.hub.send(this.workerId, "runtime.start", { requestId, request })) {
      this.pending.delete(requestId);
      throw new Error(`distributed_worker_not_connected:${this.workerId}`);
    }
    const abort = () => void this.stop(requestId, "launch_aborted");
    signal.addEventListener("abort", abort, { once: true });
    // Do not use an ignored `finally()` here. `finally()` returns a second
    // promise which rejects when the worker disconnects; leaving that derived
    // promise unobserved used to terminate the coordinator process even though
    // the supervisor was already handling the original rejection.
    void start.exited.promise.then(
      () => signal.removeEventListener("abort", abort),
      () => signal.removeEventListener("abort", abort),
    );
    return {
      ready: start.ready.promise,
      exited: start.exited.promise,
      stop: (reason: string) => this.stop(requestId, reason),
      output: () => ({ ...start.output }),
    };
  }

  async createRuntimeProxy(targetPort: number): Promise<RuntimeProxyHandle> {
    const proxy = await this.hub.createRuntimeProxy(this.workerId, targetPort);
    this.runtimeProxies.add(proxy);
    return {
      ...proxy,
      close: async () => {
        this.runtimeProxies.delete(proxy);
        await proxy.close();
      },
    };
  }

  subscribeRuntimePreparation(
    listener: (event: RuntimePreparationProgressEvent) => void,
  ): () => void {
    this.preparationListeners.add(listener);
    return () => this.preparationListeners.delete(listener);
  }

  async close(): Promise<void> {
    this.closed = true;
    this.hub.off("envelope", this.onEnvelope);
    this.hub.off("disconnect", this.onDisconnect);
    const error = new Error(`worker_tunnel_closed:${this.workerId}`);
    for (const preparation of this.preparations.values()) preparation.reject(error);
    for (const start of this.pending.values()) {
      start.ready.reject(error);
      start.exited.reject(error);
    }
    this.preparations.clear();
    this.preparationListeners.clear();
    this.pending.clear();
    this.prepared = false;
    const proxies = [...this.runtimeProxies];
    this.runtimeProxies.clear();
    await Promise.all(proxies.map((proxy) => proxy.close().catch(() => undefined)));
  }

  private async prepare(signal: AbortSignal): Promise<void> {
    if (this.prepared) return;
    const generation = this.connectionGeneration;
    const requestId = `prepare-${shortHash(JSON.stringify(this.description))}`;
    let preparation = this.preparations.get(requestId);
    if (!preparation) {
      preparation = deferred<void>();
      this.preparations.set(requestId, preparation);
      if (!this.hub.send(this.workerId, "runtime.prepare", { requestId, description: this.description })) {
        this.preparations.delete(requestId);
        throw new Error(`distributed_worker_not_connected:${this.workerId}`);
      }
    }
    await withTimeoutAndSignal(preparation.promise, this.timeoutMs, signal, "worker_tunnel_prepare_timeout");
    if (this.closed || generation !== this.connectionGeneration) return;
    this.prepared = true;
  }

  private async stop(requestId: string, reason: string): Promise<void> {
    const start = this.pending.get(requestId);
    if (!start) return;
    if (!this.hub.send(this.workerId, "runtime.stop", { requestId, reason })) {
      this.pending.delete(requestId);
      const error = new Error(`distributed_worker_not_connected:${this.workerId}`);
      start.ready.reject(error);
      start.exited.reject(error);
      await start.exited.promise.catch(() => undefined);
      return;
    }
    let timer: NodeJS.Timeout | null = null;
    const acknowledged = await Promise.race([
      start.exited.promise.then(() => true, () => true),
      new Promise<boolean>((resolve) => {
        timer = setTimeout(() => resolve(false), this.timeoutMs);
      }),
    ]);
    if (timer) clearTimeout(timer);
    if (acknowledged) return;
    if (this.pending.get(requestId) !== start) return;
    this.pending.delete(requestId);
    const error = new Error(`worker_tunnel_stop_timeout:${this.workerId}`);
    start.ready.reject(error);
    start.exited.reject(error);
    await start.exited.promise.catch(() => undefined);
  }

  private handleEnvelope(envelope: WorkerEnvelope): void {
    if (envelope.workerId !== this.workerId) return;
    const payload = envelope.payload as Record<string, unknown> | undefined;
    const requestId = typeof payload?.requestId === "string" ? payload.requestId : null;
    if (!requestId) return;
    if (!payload) return;
    if (envelope.type === "runtime.prepare.progress") {
      const event = {
        stageIndex: payload.stageIndex,
        layerStart: payload.layerStart,
        layerEnd: payload.layerEnd,
        state: payload.state,
        ...(typeof payload.packageId === "string" ? { packageId: payload.packageId } : {}),
        ...(typeof payload.weightsSizeBytes === "number"
          ? { weightsSizeBytes: payload.weightsSizeBytes }
          : {}),
        ...(typeof payload.downloadedBytes === "number"
          ? { downloadedBytes: payload.downloadedBytes }
          : {}),
        ...(typeof payload.resumedBytes === "number"
          ? { resumedBytes: payload.resumedBytes }
          : {}),
        ...(typeof payload.materialized === "boolean"
          ? { materialized: payload.materialized }
          : {}),
      };
      if (
        typeof event.stageIndex === "number"
        && typeof event.layerStart === "number"
        && typeof event.layerEnd === "number"
        && (event.state === "preparing" || event.state === "ready")
      ) {
        for (const listener of this.preparationListeners) listener(
          event as RuntimePreparationProgressEvent,
        );
      }
      return;
    }
    if (envelope.type === "runtime.prepared") {
      const preparation = this.preparations.get(requestId);
      if (!preparation) return;
      this.preparations.delete(requestId);
      payload.ok === true
        ? preparation.resolve()
        : preparation.reject(new Error(typeof payload.error === "string" ? payload.error : "worker_runtime_prepare_failed"));
      return;
    }
    const start = this.pending.get(requestId);
    if (!start) return;
    if (envelope.type === "runtime.ready") {
      const output = payload.output as LaunchCapturedOutput | undefined;
      if (output) Object.assign(start.output, output);
      start.ready.resolve();
      return;
    }
    if (envelope.type === "runtime.exited") {
      const exit = payload.exit as LaunchProcessExit;
      const output = payload.output as LaunchCapturedOutput;
      Object.assign(start.output, output);
      this.pending.delete(requestId);
      if (exit.error) start.ready.reject(new Error(exit.error));
      start.exited.resolve(exit);
    }
  }
}

function requestIdentity(request: LaunchAgentStartRequest): string {
  return `run-${shortHash(`${request.launchId}\0${request.process.processId}`)}`;
}

function shortHash(value: string): string {
  return createHash("sha256").update(value).digest("hex").slice(0, 32);
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

async function withTimeoutAndSignal<T>(promise: Promise<T>, timeoutMs: number, signal: AbortSignal, message: string): Promise<T> {
  if (signal.aborted) throw abortError(signal);
  let timer: NodeJS.Timeout | undefined;
  let abort: (() => void) | undefined;
  try {
    return await new Promise<T>((resolve, reject) => {
      timer = setTimeout(() => reject(new Error(message)), timeoutMs);
      abort = () => reject(abortError(signal));
      signal.addEventListener("abort", abort, { once: true });
      void promise.then(resolve, reject);
    });
  } finally {
    clearTimeout(timer);
    if (abort) signal.removeEventListener("abort", abort);
  }
}

function abortError(signal: AbortSignal): Error {
  return signal.reason instanceof Error ? signal.reason : new Error("distributed_launch_cancelled");
}
