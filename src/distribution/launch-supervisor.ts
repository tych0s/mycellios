import { spawn, type ChildProcessByStdio } from "node:child_process";
import type { Readable } from "node:stream";
import { StringDecoder } from "node:string_decoder";
import {
  validatePythonLaunchDescription,
  type PythonLaunchProcess,
  type PythonPipelineLaunchDescription,
} from "./python-launcher.js";

export type LaunchSupervisorState =
  | "idle"
  | "starting"
  | "running"
  | "stopping"
  | "stopped"
  | "failed";

export type LaunchProcessState =
  | "pending"
  | "starting"
  | "ready"
  | "stopping"
  | "stopped"
  | "failed";

export interface LaunchProcessExit {
  code: number | null;
  signal: NodeJS.Signals | null;
  error?: string;
}

export interface LaunchCapturedOutput {
  stdout: string;
  stderr: string;
  stdoutTruncated: boolean;
  stderrTruncated: boolean;
}

export interface LaunchProcessHandle {
  ready: Promise<void>;
  exited: Promise<LaunchProcessExit>;
  stop(reason: string): Promise<void>;
  output?(): LaunchCapturedOutput;
}

export interface LaunchAgentStartRequest {
  launchId: string;
  pipelineId: string;
  nodeId: string;
  process: PythonLaunchProcess;
}

/** Implement this interface with an RPC client to launch on a remote node. */
export interface LaunchAgent {
  readonly id: string;
  start(request: LaunchAgentStartRequest, signal: AbortSignal): Promise<LaunchProcessHandle>;
  /** Optional coordinator-side loopback proxy for runtimes behind NAT. */
  createRuntimeProxy?(targetPort: number): Promise<{
    host: string;
    port: number;
    close(): Promise<void>;
  }>;
  close?(): void | Promise<void>;
}

export type LaunchAgentResolver = (
  nodeId: string,
  process: PythonLaunchProcess,
) => LaunchAgent | undefined;

export interface LaunchSupervisorOptions {
  resolveAgent: LaunchAgentResolver;
  readinessTimeoutMs?: number;
  maxTelemetryEvents?: number;
  now?: () => number;
}

export interface LaunchProcessSnapshot {
  processId: string;
  nodeId: string;
  kind: PythonLaunchProcess["kind"];
  stageIndex: number;
  state: LaunchProcessState;
  agentId?: string;
  startedAtMs?: number;
  readyAtMs?: number;
  stoppedAtMs?: number;
  exit?: LaunchProcessExit;
  error?: string;
  output?: LaunchCapturedOutput;
}

export type LaunchTelemetryEventType =
  | "supervisor_starting"
  | "process_starting"
  | "process_ready"
  | "process_exit"
  | "process_stop_starting"
  | "process_stopped"
  | "process_stop_failed"
  | "supervisor_running"
  | "supervisor_stopping"
  | "supervisor_cancelled"
  | "supervisor_failed"
  | "supervisor_stopped";

export interface LaunchTelemetryEvent {
  sequence: number;
  atMs: number;
  type: LaunchTelemetryEventType;
  processId?: string;
  nodeId?: string;
  detail?: string;
}

export interface LaunchSupervisorSnapshot {
  launchId: string;
  pipelineId: string;
  state: LaunchSupervisorState;
  failure: string | null;
  processes: LaunchProcessSnapshot[];
  telemetry: LaunchTelemetryEvent[];
  telemetryDropped: number;
}

interface MutableProcessRecord extends LaunchProcessSnapshot {
  launch: PythonLaunchProcess;
  handle?: LaunchProcessHandle;
}

export class LaunchCancelledError extends Error {
  constructor(message = "launch_cancelled") {
    super(message);
    this.name = "LaunchCancelledError";
  }
}

export class LaunchReadinessTimeoutError extends Error {
  constructor(readonly processId: string, readonly timeoutMs: number) {
    super(`launch_readiness_timeout:${processId}:${timeoutMs}`);
    this.name = "LaunchReadinessTimeoutError";
  }
}

export class LaunchProcessExitedError extends Error {
  constructor(readonly processId: string, readonly exit: LaunchProcessExit) {
    super(
      `launch_process_exited:${processId}:code=${exit.code ?? "null"}:signal=${exit.signal ?? "null"}`
        + (exit.error ? `:error=${exit.error}` : ""),
    );
    this.name = "LaunchProcessExitedError";
  }
}

export class PythonLaunchSupervisor {
  private readonly description: PythonPipelineLaunchDescription;
  private readonly resolveAgent: LaunchAgentResolver;
  private readonly readinessTimeoutMs: number;
  private readonly maxTelemetryEvents: number;
  private readonly now: () => number;
  private readonly processes: MutableProcessRecord[];
  private readonly telemetry: LaunchTelemetryEvent[] = [];
  private readonly listeners = new Set<(event: LaunchTelemetryEvent) => void>();
  private telemetryDropped = 0;
  private sequence = 0;
  private state: LaunchSupervisorState = "idle";
  private failure: string | null = null;
  private lifecycleController: AbortController | null = null;
  private startPromise: Promise<LaunchSupervisorSnapshot> | null = null;
  private stopPromise: Promise<LaunchSupervisorSnapshot> | null = null;
  private rollbackPromise: Promise<void> | null = null;
  private runningFailurePromise: Promise<void> | null = null;
  private terminal = deferred<LaunchSupervisorSnapshot>();

  constructor(descriptionValue: unknown, options: LaunchSupervisorOptions) {
    validatePythonLaunchDescription(descriptionValue);
    if (!options || typeof options.resolveAgent !== "function") {
      throw new Error("launch_agent_resolver_is_required");
    }
    this.description = structuredClone(descriptionValue);
    this.resolveAgent = options.resolveAgent;
    this.readinessTimeoutMs = boundedInteger(
      options.readinessTimeoutMs ?? 180_000,
      1,
      86_400_000,
      "launch_readiness_timeout_is_invalid",
    );
    this.maxTelemetryEvents = boundedInteger(
      options.maxTelemetryEvents ?? 1_000,
      1,
      1_000_000,
      "launch_telemetry_limit_is_invalid",
    );
    this.now = options.now ?? Date.now;
    this.processes = this.description.launchOrder.map((launch) => ({
      processId: launch.processId,
      nodeId: launch.anchor.memberId,
      kind: launch.kind,
      stageIndex: launch.stageIndex,
      state: "pending",
      launch,
    }));
  }

  start(signal?: AbortSignal): Promise<LaunchSupervisorSnapshot> {
    if (this.state === "running") return Promise.resolve(this.snapshot());
    if (this.startPromise) return this.startPromise;
    if (this.state !== "idle") {
      return Promise.reject(new Error(`launch_supervisor_cannot_start_from:${this.state}`));
    }
    this.startPromise = this.startInternal(signal);
    return this.startPromise;
  }

  stop(reason = "operator_stop"): Promise<LaunchSupervisorSnapshot> {
    if (this.stopPromise) return this.stopPromise;
    this.stopPromise = this.stopInternal(reason);
    return this.stopPromise;
  }

  waitForTerminal(): Promise<LaunchSupervisorSnapshot> {
    if (this.state === "stopped" || (this.state === "failed" && this.rollbackPromise === null)) {
      return Promise.resolve(this.snapshot());
    }
    return this.terminal.promise;
  }

  snapshot(): LaunchSupervisorSnapshot {
    return {
      launchId: this.description.launchId,
      pipelineId: this.description.pipelineId,
      state: this.state,
      failure: this.failure,
      processes: this.processes.map(({ launch: _launch, handle, ...record }) => ({
        ...structuredClone(record),
        ...safeHandleOutput(handle),
      })),
      telemetry: structuredClone(this.telemetry),
      telemetryDropped: this.telemetryDropped,
    };
  }

  subscribe(listener: (event: LaunchTelemetryEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private async startInternal(externalSignal?: AbortSignal): Promise<LaunchSupervisorSnapshot> {
    this.state = "starting";
    this.emit("supervisor_starting");
    const controller = new AbortController();
    this.lifecycleController = controller;
    const removeExternalAbort = linkAbortSignal(externalSignal, controller);
    try {
      for (const record of this.processes) {
        throwIfAborted(controller.signal);
        record.state = "starting";
        record.startedAtMs = this.now();
        this.emit("process_starting", record);
        const agent = this.resolveAgent(record.nodeId, record.launch);
        if (!agent) throw new Error(`launch_agent_not_found:${record.nodeId}`);
        record.agentId = agent.id;
        const deadline = this.now() + this.readinessTimeoutMs;
        const handle = await waitForOperation(
          agent.start(
            {
              launchId: this.description.launchId,
              pipelineId: this.description.pipelineId,
              nodeId: record.nodeId,
              process: structuredClone(record.launch),
            },
            controller.signal,
          ),
          controller.signal,
          remainingMs(deadline, this.now()),
          () => new LaunchReadinessTimeoutError(record.processId, this.readinessTimeoutMs),
        );
        validateHandle(handle, record.processId);
        record.handle = handle;
        await waitUntilReady(
          handle,
          record.processId,
          controller.signal,
          remainingMs(deadline, this.now()),
          this.readinessTimeoutMs,
        );
        throwIfAborted(controller.signal);
        record.state = "ready";
        record.readyAtMs = this.now();
        this.emit("process_ready", record);
        this.monitorReadyProcess(record);
      }
      throwIfAborted(controller.signal);
      this.state = "running";
      this.emit("supervisor_running");
      return this.snapshot();
    } catch (error) {
      const normalized = normalizeError(error);
      const cancelled = normalized instanceof LaunchCancelledError;
      const active = this.processes.find((record) => record.state === "starting");
      if (active) {
        if (cancelled && !active.handle) {
          active.state = "stopped";
          active.stoppedAtMs = this.now();
        } else if (!cancelled) {
          active.state = "failed";
          active.error = normalized.message;
        }
      }
      this.failure = cancelled ? null : normalized.message;
      this.state = cancelled ? "stopping" : "failed";
      this.emit(cancelled ? "supervisor_cancelled" : "supervisor_failed", undefined, normalized.message);
      if (!controller.signal.aborted) controller.abort(normalized);
      await this.rollback(cancelled ? "launch_cancelled" : "launch_failed");
      this.state = cancelled ? "stopped" : "failed";
      if (cancelled) this.emit("supervisor_stopped", undefined, "launch_cancelled");
      this.resolveTerminal();
      throw normalized;
    } finally {
      removeExternalAbort();
    }
  }

  private async stopInternal(reason: string): Promise<LaunchSupervisorSnapshot> {
    if (this.state === "stopped") return this.snapshot();
    if (this.state === "failed") {
      if (this.rollbackPromise) await this.rollbackPromise;
      return this.snapshot();
    }
    if (this.state === "idle") {
      this.state = "stopped";
      this.emit("supervisor_stopped", undefined, reason);
      this.resolveTerminal();
      return this.snapshot();
    }
    if (this.state === "starting") {
      this.lifecycleController?.abort(new LaunchCancelledError(reason));
      try {
        await this.startPromise;
      } catch {
        // startInternal owns rollback and the final stopped/failed transition.
      }
      return this.snapshot();
    }
    this.state = "stopping";
    this.emit("supervisor_stopping", undefined, reason);
    this.lifecycleController?.abort(new LaunchCancelledError(reason));
    await this.rollback(reason);
    this.state = "stopped";
    this.emit("supervisor_stopped", undefined, reason);
    this.resolveTerminal();
    return this.snapshot();
  }

  private monitorReadyProcess(record: MutableProcessRecord): void {
    const handle = record.handle!;
    void handle.exited.then(
      (exit) => this.onProcessExit(record, exit),
      (error) =>
        this.onProcessExit(record, {
          code: null,
          signal: null,
          error: normalizeError(error).message,
        }),
    );
  }

  private onProcessExit(record: MutableProcessRecord, exit: LaunchProcessExit): void {
    record.exit = structuredClone(exit);
    this.emit("process_exit", record, renderExit(exit));
    if (this.state !== "starting" && this.state !== "running") return;
    record.state = "failed";
    const error = new LaunchProcessExitedError(record.processId, exit);
    record.error = error.message;
    if (this.state === "starting") {
      this.lifecycleController?.abort(error);
      return;
    }
    if (!this.runningFailurePromise) {
      this.runningFailurePromise = this.failRunning(error);
    }
  }

  private async failRunning(error: Error): Promise<void> {
    this.failure = error.message;
    this.state = "failed";
    this.emit("supervisor_failed", undefined, error.message);
    this.lifecycleController?.abort(error);
    await this.rollback("process_exited_after_ready");
    this.resolveTerminal();
  }

  private rollback(reason: string): Promise<void> {
    if (this.rollbackPromise) return this.rollbackPromise;
    this.rollbackPromise = this.rollbackInternal(reason).finally(() => {
      this.rollbackPromise = null;
    });
    return this.rollbackPromise;
  }

  private async rollbackInternal(reason: string): Promise<void> {
    for (const record of [...this.processes].reverse()) {
      if (!record.handle) continue;
      if (record.state !== "failed") record.state = "stopping";
      this.emit("process_stop_starting", record, reason);
      try {
        await record.handle.stop(reason);
        if (record.state !== "failed") record.state = "stopped";
        record.stoppedAtMs = this.now();
        this.emit("process_stopped", record, reason);
      } catch (error) {
        const normalized = normalizeError(error);
        record.state = "failed";
        record.error = normalized.message;
        this.emit("process_stop_failed", record, normalized.message);
      }
    }
  }

  private emit(
    type: LaunchTelemetryEventType,
    process?: MutableProcessRecord,
    detail?: string,
  ): void {
    const event: LaunchTelemetryEvent = {
      sequence: this.sequence++,
      atMs: this.now(),
      type,
      ...(process
        ? { processId: process.processId, nodeId: process.nodeId }
        : {}),
      ...(detail ? { detail } : {}),
    };
    this.telemetry.push(event);
    while (this.telemetry.length > this.maxTelemetryEvents) {
      this.telemetry.shift();
      this.telemetryDropped += 1;
    }
    for (const listener of this.listeners) {
      try {
        listener(structuredClone(event));
      } catch {
        // Telemetry consumers cannot break process lifecycle.
      }
    }
  }

  private resolveTerminal(): void {
    this.terminal.resolve(this.snapshot());
  }
}

export interface LocalProcessReadinessObservation {
  process: PythonLaunchProcess;
  stream: "stdout" | "stderr";
  chunk: string;
  recentStdout: string;
  recentStderr: string;
}

export interface LocalProcessAgentOptions {
  id?: string;
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  maxOutputBytesPerStream?: number;
  stopGraceMs?: number;
  readyWhen?: (observation: LocalProcessReadinessObservation) => boolean;
}

/**
 * Explicit opt-in local executor. Remote deployments should inject an RPC
 * LaunchAgent instead. Commands use spawn(argv, shell=false), never a shell.
 */
export class LocalProcessAgent implements LaunchAgent {
  readonly id: string;
  private readonly cwd: string | undefined;
  private readonly env: NodeJS.ProcessEnv | undefined;
  private readonly maxOutputBytes: number;
  private readonly stopGraceMs: number;
  private readonly readyWhen: (observation: LocalProcessReadinessObservation) => boolean;

  constructor(options: LocalProcessAgentOptions = {}) {
    this.id = options.id ?? "local-process";
    this.cwd = options.cwd;
    this.env = options.env;
    this.maxOutputBytes = boundedInteger(
      options.maxOutputBytesPerStream ?? 64 * 1024,
      1_024,
      16 * 1024 * 1024,
      "local_process_output_limit_is_invalid",
    );
    this.stopGraceMs = boundedInteger(
      options.stopGraceMs ?? 5_000,
      1,
      300_000,
      "local_process_stop_grace_is_invalid",
    );
    this.readyWhen = options.readyWhen ?? defaultLocalReadiness;
  }

  async start(
    request: LaunchAgentStartRequest,
    signal: AbortSignal,
  ): Promise<LaunchProcessHandle> {
    if (signal.aborted) throw cancellationFromSignal(signal);
    const command = request.process.command;
    const spawnOptions = {
      shell: false,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"] as ["ignore", "pipe", "pipe"],
      ...(this.cwd ? { cwd: this.cwd } : {}),
      ...(this.env ? { env: { ...process.env, ...this.env } } : {}),
    };
    const child = spawn(command.executable, command.args, spawnOptions);
    const handle = new LocalProcessHandle(
      child,
      request.process,
      this.maxOutputBytes,
      this.stopGraceMs,
      this.readyWhen,
    );
    const abort = () => {
      void handle.stop("launch_aborted");
    };
    signal.addEventListener("abort", abort, { once: true });
    void handle.exited.finally(() => signal.removeEventListener("abort", abort));
    return handle;
  }
}

class LocalProcessHandle implements LaunchProcessHandle {
  readonly ready: Promise<void>;
  readonly exited: Promise<LaunchProcessExit>;
  private readonly readyDeferred = deferred<void>();
  private readonly exitDeferred = deferred<LaunchProcessExit>();
  private readonly stdout = new BoundedTextCapture();
  private readonly stderr = new BoundedTextCapture();
  private stopPromise: Promise<void> | null = null;
  private didExit = false;
  private didBecomeReady = false;

  constructor(
    private readonly child: ChildProcessByStdio<null, Readable, Readable>,
    private readonly process: PythonLaunchProcess,
    maxOutputBytes: number,
    private readonly stopGraceMs: number,
    private readonly readyWhen: (observation: LocalProcessReadinessObservation) => boolean,
  ) {
    this.stdout.limit = maxOutputBytes;
    this.stderr.limit = maxOutputBytes;
    this.ready = this.readyDeferred.promise;
    this.exited = this.exitDeferred.promise;
    const stdoutDecoder = new StringDecoder("utf8");
    const stderrDecoder = new StringDecoder("utf8");
    child.stdout.on("data", (chunk: Buffer) =>
      this.onOutput("stdout", stdoutDecoder.write(chunk)),
    );
    child.stderr.on("data", (chunk: Buffer) =>
      this.onOutput("stderr", stderrDecoder.write(chunk)),
    );
    child.once("error", (error) => {
      this.finish({ code: null, signal: null, error: error.message });
    });
    child.once("close", (code, signal) => {
      const stdoutTail = stdoutDecoder.end();
      const stderrTail = stderrDecoder.end();
      if (stdoutTail) this.onOutput("stdout", stdoutTail);
      if (stderrTail) this.onOutput("stderr", stderrTail);
      this.finish({ code, signal });
    });
  }

  output(): LaunchCapturedOutput {
    return {
      stdout: this.stdout.text,
      stderr: this.stderr.text,
      stdoutTruncated: this.stdout.truncated,
      stderrTruncated: this.stderr.truncated,
    };
  }

  stop(_reason: string): Promise<void> {
    if (this.stopPromise) return this.stopPromise;
    this.stopPromise = this.stopInternal();
    return this.stopPromise;
  }

  private async stopInternal(): Promise<void> {
    if (this.didExit) return;
    this.child.kill("SIGTERM");
    if (await settlesWithin(this.exited, this.stopGraceMs)) return;
    if (!this.didExit) this.child.kill("SIGKILL");
    if (!(await settlesWithin(this.exited, this.stopGraceMs))) {
      throw new Error(`local_process_did_not_exit:${this.process.processId}`);
    }
  }

  private onOutput(stream: "stdout" | "stderr", chunk: string): void {
    if (!chunk) return;
    (stream === "stdout" ? this.stdout : this.stderr).append(chunk);
    if (this.didBecomeReady || this.didExit) return;
    const observation: LocalProcessReadinessObservation = {
      process: this.process,
      stream,
      chunk,
      recentStdout: this.stdout.recent,
      recentStderr: this.stderr.recent,
    };
    let ready = false;
    try {
      ready = this.readyWhen(observation);
    } catch (error) {
      this.readyDeferred.reject(normalizeError(error));
      return;
    }
    if (ready) {
      this.didBecomeReady = true;
      this.readyDeferred.resolve();
    }
  }

  private finish(exit: LaunchProcessExit): void {
    if (this.didExit) return;
    this.didExit = true;
    if (!this.didBecomeReady) {
      this.readyDeferred.reject(new LaunchProcessExitedError(this.process.processId, exit));
    }
    this.exitDeferred.resolve(exit);
  }
}

class BoundedTextCapture {
  limit = 64 * 1024;
  text = "";
  recent = "";
  truncated = false;

  append(chunk: string): void {
    this.recent = `${this.recent}${chunk}`.slice(-8_192);
    const remaining = this.limit - Buffer.byteLength(this.text, "utf8");
    if (remaining <= 0) {
      this.truncated = true;
      return;
    }
    const buffer = Buffer.from(chunk, "utf8");
    if (buffer.length <= remaining) {
      this.text += chunk;
      return;
    }
    this.text += buffer.subarray(0, remaining).toString("utf8");
    this.truncated = true;
  }
}

function defaultLocalReadiness(observation: LocalProcessReadinessObservation): boolean {
  if (observation.process.kind === "cell-member") {
    // The member may start before its anchor and deliberately waits/retries.
    // This event proves argv/config parsing and process liveness; the anchor
    // stage itself does not become ready until every rank passes handshake,
    // shard hash validation, load and Gloo rendezvous.
    return observation.recentStderr.includes('"event": "joining_external_cell"');
  }
  if (observation.process.kind === "remote-stage") {
    return observation.recentStderr.includes("stage_ready");
  }
  return /Running on|======== Running/.test(
    `${observation.recentStdout}\n${observation.recentStderr}`,
  );
}

async function waitUntilReady(
  handle: LaunchProcessHandle,
  processId: string,
  signal: AbortSignal,
  timeoutMs: number,
  configuredTimeoutMs: number,
): Promise<void> {
  await waitForOperation(
    Promise.race([
      handle.ready,
      handle.exited.then((exit) => {
        throw new LaunchProcessExitedError(processId, exit);
      }),
    ]),
    signal,
    timeoutMs,
    () => new LaunchReadinessTimeoutError(processId, configuredTimeoutMs),
  );
}

function waitForOperation<T>(
  operation: Promise<T>,
  signal: AbortSignal,
  timeoutMs: number,
  timeoutError: () => Error,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const finish = (callback: () => void) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
      callback();
    };
    const abort = () => finish(() => reject(cancellationFromSignal(signal)));
    const timer = setTimeout(() => finish(() => reject(timeoutError())), Math.max(1, timeoutMs));
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) {
      abort();
      return;
    }
    operation.then(
      (value) => finish(() => resolve(value)),
      (error) => finish(() => reject(normalizeError(error))),
    );
  });
}

function validateHandle(handle: LaunchProcessHandle, processId: string): void {
  if (
    !handle ||
    typeof handle.stop !== "function" ||
    !(handle.ready instanceof Promise) ||
    !(handle.exited instanceof Promise)
  ) {
    throw new Error(`launch_agent_returned_invalid_handle:${processId}`);
  }
}

function safeHandleOutput(
  handle: LaunchProcessHandle | undefined,
): { output?: LaunchCapturedOutput } {
  if (!handle?.output) return {};
  try {
    return { output: structuredClone(handle.output()) };
  } catch {
    return {};
  }
}

function linkAbortSignal(external: AbortSignal | undefined, target: AbortController): () => void {
  if (!external) return () => undefined;
  const abort = () => target.abort(new LaunchCancelledError("external_abort"));
  external.addEventListener("abort", abort, { once: true });
  if (external.aborted) abort();
  return () => external.removeEventListener("abort", abort);
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) throw cancellationFromSignal(signal);
}

function cancellationFromSignal(signal: AbortSignal): Error {
  return signal.reason instanceof Error
    ? signal.reason
    : new LaunchCancelledError("launch_cancelled");
}

function remainingMs(deadline: number, now: number): number {
  return Math.max(1, deadline - now);
}

function renderExit(exit: LaunchProcessExit): string {
  return `code=${exit.code ?? "null"},signal=${exit.signal ?? "null"}${
    exit.error ? `,error=${exit.error}` : ""
  }`;
}

function normalizeError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}

function boundedInteger(value: unknown, min: number, max: number, error: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < min || (value as number) > max) {
    throw new Error(error);
  }
  return value as number;
}

function deferred<T>(): {
  promise: Promise<T>;
  resolve(value: T): void;
  reject(error: unknown): void;
} {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

async function settlesWithin<T>(promise: Promise<T>, timeoutMs: number): Promise<boolean> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise.then(() => true, () => true),
      new Promise<boolean>((resolve) => {
        timer = setTimeout(() => resolve(false), timeoutMs);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}
