import { createHash } from "node:crypto";
import type { WorkerEnvelope } from "../contracts/types.js";
import type { RuntimeProxyHandle, WorkerHub } from "../coordinator/worker-hub.js";
import type {
  LaunchAgent,
  LaunchAgentStartRequest,
  LaunchCapturedOutput,
  LaunchProcessExit,
  LaunchProcessHandle,
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

/** Runs compiler-sealed Python stages through the already authenticated worker socket. */
export class WorkerTunnelLaunchAgent implements LaunchAgent {
  readonly id: string;
  private readonly pending = new Map<string, PendingStart>();
  private readonly preparations = new Map<string, Deferred<void>>();
  private prepared = false;
  private readonly runtimeProxies = new Set<RuntimeProxyHandle>();
  private readonly onEnvelope = (envelope: WorkerEnvelope) => this.handleEnvelope(envelope);
  private readonly onDisconnect = (workerId: string) => {
    if (workerId !== this.workerId) return;
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
    private readonly hub: WorkerHub,
    private readonly workerId: string,
    private readonly nodeId: string,
    private readonly description: PythonPipelineLaunchDescription,
    private readonly timeoutMs = 30_000,
  ) {
    this.id = `worker-tunnel:${workerId}`;
    hub.on("envelope", this.onEnvelope);
    hub.on("disconnect", this.onDisconnect);
  }

  async start(request: LaunchAgentStartRequest, signal: AbortSignal): Promise<LaunchProcessHandle> {
    if (signal.aborted) throw abortError(signal);
    if (request.nodeId !== this.nodeId) throw new Error("worker_tunnel_launch_node_mismatch");
    await this.prepare(signal);
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
    void start.exited.promise.finally(() => signal.removeEventListener("abort", abort));
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

  async close(): Promise<void> {
    this.hub.off("envelope", this.onEnvelope);
    this.hub.off("disconnect", this.onDisconnect);
    const proxies = [...this.runtimeProxies];
    this.runtimeProxies.clear();
    await Promise.all(proxies.map((proxy) => proxy.close().catch(() => undefined)));
  }

  private async prepare(signal: AbortSignal): Promise<void> {
    if (this.prepared) return;
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
    this.prepared = true;
  }

  private async stop(requestId: string, reason: string): Promise<void> {
    this.hub.send(this.workerId, "runtime.stop", { requestId, reason });
    const start = this.pending.get(requestId);
    if (start) await start.exited.promise.catch(() => undefined);
  }

  private handleEnvelope(envelope: WorkerEnvelope): void {
    if (envelope.workerId !== this.workerId) return;
    const payload = envelope.payload as Record<string, unknown> | undefined;
    const requestId = typeof payload?.requestId === "string" ? payload.requestId : null;
    if (!requestId) return;
    if (!payload) return;
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
  return await new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(message)), timeoutMs);
    const abort = () => reject(abortError(signal));
    signal.addEventListener("abort", abort, { once: true });
    void promise.then(resolve, reject).finally(() => {
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
    });
  });
}

function abortError(signal: AbortSignal): Error {
  return signal.reason instanceof Error ? signal.reason : new Error("distributed_launch_cancelled");
}
