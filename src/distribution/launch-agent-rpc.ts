import { createHash, timingSafeEqual } from "node:crypto";
import {
  createServer,
  type IncomingMessage,
  type Server,
  type ServerResponse,
} from "node:http";
import {
  LaunchCancelledError,
  LaunchProcessExitedError,
  type LaunchAgent,
  type LaunchAgentStartRequest,
  type LaunchCapturedOutput,
  type LaunchProcessExit,
  type LaunchProcessHandle,
} from "./launch-supervisor.js";
import {
  MYCELLIOS_CELL_MEMBER_MODULE,
  MYCELLIOS_SERVER_MODULE,
  MYCELLIOS_STAGE_MODULE,
  validatePythonLaunchDescription,
  type PythonLaunchProcess,
  type PythonPipelineLaunchDescription,
} from "./python-launcher.js";
import {
  PHYSICAL_PROBE_REQUEST_SCHEMA,
  assertPhysicalProbeNonce,
  validatePhysicalProbe,
  type PhysicalProbeCollector,
  type PhysicalProbeV1,
} from "./physical-probe.js";
import {
  nativeBuildIdentitySchema,
  type NativeBuildIdentity,
} from "../contracts/build-identity.js";
import { validateExecutorIsolationPolicy } from "./process-environment.js";

const START_SCHEMA = "gdlp-launch-agent-start/1";
const STOP_SCHEMA = "gdlp-launch-agent-stop/1";
const SNAPSHOT_SCHEMA = "gdlp-launch-agent-process/1";
const HEALTH_SCHEMA = "gdlp-launch-agent-health/3";
const ERROR_SCHEMA = "gdlp-launch-agent-error/1";

type RpcProcessState = "starting" | "ready" | "exited";

export interface LaunchAgentRpcProcessSnapshot {
  schema: typeof SNAPSHOT_SCHEMA;
  handleId: string;
  launchId: string;
  processId: string;
  state: RpcProcessState;
  ready: boolean;
  exit: LaunchProcessExit | null;
  output: LaunchCapturedOutput;
}

export interface LaunchAgentRpcHealth {
  schema: typeof HEALTH_SCHEMA;
  agentId: string;
  nodeId: string | null;
  buildIdentity: NativeBuildIdentity | null;
  /** Processes that are still starting or running. */
  activeProcesses: number;
  /** Completed idempotency records retained to prevent duplicate launches. */
  retainedTombstones: number;
}

export interface HttpLaunchAgentOptions {
  endpoint: string | URL;
  id?: string;
  /** Optional bearer credential for the launch-agent control plane. */
  authToken?: string;
  requestTimeoutMs?: number;
  /** Defaults to requestTimeoutMs, or 10 seconds when neither is provided. */
  healthTimeoutMs?: number;
  /** Defaults to requestTimeoutMs when explicitly set, otherwise 60 seconds. */
  physicalEvidenceTimeoutMs?: number;
  cleanupRequestTimeoutMs?: number;
  pollRequestTimeoutMs?: number;
  pollIntervalMs?: number;
  maxConsecutivePollErrors?: number;
  maxOutputBytesPerStream?: number;
}

export class LaunchAgentRpcTimeoutError extends Error {
  constructor(readonly operation: string, readonly timeoutMs: number) {
    super(`launch_agent_rpc_timeout:${operation}:${timeoutMs}`);
    this.name = "LaunchAgentRpcTimeoutError";
  }
}

export class LaunchAgentRpcHttpError extends Error {
  constructor(readonly status: number, message: string) {
    super(`launch_agent_rpc_http:${status}:${message}`);
    this.name = "LaunchAgentRpcHttpError";
  }
}

/** HTTP LaunchAgent client. It never interprets argv and never invokes a shell. */
export class HttpLaunchAgent implements LaunchAgent {
  readonly id: string;
  private readonly baseUrl: string;
  private readonly authToken: string | undefined;
  private readonly requestTimeoutMs: number;
  private readonly healthTimeoutMs: number;
  private readonly physicalEvidenceTimeoutMs: number;
  private readonly cleanupRequestTimeoutMs: number;
  private readonly pollRequestTimeoutMs: number;
  private readonly pollIntervalMs: number;
  private readonly maxConsecutivePollErrors: number;
  private readonly maxOutputBytes: number;

  constructor(options: HttpLaunchAgentOptions) {
    if (!options || typeof options !== "object") {
      throw new Error("launch_agent_rpc_options_are_required");
    }
    const endpoint = normalizeEndpoint(options.endpoint);
    this.baseUrl = endpoint.href.replace(/\/$/, "");
    this.id = options.id ?? defaultHttpAgentId(endpoint);
    assertIdentifier(this.id, "launch_agent_rpc_id");
    this.authToken = normalizeOptionalAuthToken(options.authToken);
    this.requestTimeoutMs = boundedInteger(
      options.requestTimeoutMs ?? 10_000,
      1,
      300_000,
      "launch_agent_rpc_request_timeout_is_invalid",
    );
    this.healthTimeoutMs = boundedInteger(
      options.healthTimeoutMs ?? options.requestTimeoutMs ?? 10_000,
      1,
      300_000,
      "launch_agent_rpc_health_timeout_is_invalid",
    );
    this.physicalEvidenceTimeoutMs = boundedInteger(
      options.physicalEvidenceTimeoutMs ?? options.requestTimeoutMs ?? 60_000,
      1,
      300_000,
      "launch_agent_rpc_physical_evidence_timeout_is_invalid",
    );
    // A cleanup starts after a failed/aborted fetch, when Undici may still be
    // discarding the old connection. Reusing a very small operation timeout
    // can abort the deterministic stop before it even reaches the daemon.
    this.cleanupRequestTimeoutMs = boundedInteger(
      options.cleanupRequestTimeoutMs ?? Math.max(this.requestTimeoutMs, 1_000),
      1,
      300_000,
      "launch_agent_rpc_cleanup_timeout_is_invalid",
    );
    this.pollRequestTimeoutMs = boundedInteger(
      options.pollRequestTimeoutMs ?? 5_000,
      1,
      300_000,
      "launch_agent_rpc_poll_timeout_is_invalid",
    );
    this.pollIntervalMs = boundedInteger(
      options.pollIntervalMs ?? 100,
      1,
      60_000,
      "launch_agent_rpc_poll_interval_is_invalid",
    );
    this.maxConsecutivePollErrors = boundedInteger(
      options.maxConsecutivePollErrors ?? 5,
      1,
      1_000,
      "launch_agent_rpc_poll_error_limit_is_invalid",
    );
    this.maxOutputBytes = boundedInteger(
      options.maxOutputBytesPerStream ?? 64 * 1024,
      1,
      16 * 1024 * 1024,
      "launch_agent_rpc_output_limit_is_invalid",
    );
  }

  async start(
    request: LaunchAgentStartRequest,
    signal: AbortSignal,
  ): Promise<LaunchProcessHandle> {
    validateLaunchAgentRpcStartRequest(request);
    if (signal.aborted) throw cancellationFromSignal(signal);
    const handleId = launchAgentRpcHandleId(request);
    let snapshot: LaunchAgentRpcProcessSnapshot;
    try {
      snapshot = await this.requestSnapshot(
        "/v1/processes",
        "POST",
        { schema: START_SCHEMA, request },
        signal,
        this.requestTimeoutMs,
        "start",
      );
    } catch (error) {
      // The request may have reached the daemon before the local timeout or
      // cancellation. The deterministic id makes orphan cleanup possible.
      if (
        signal.aborted ||
        error instanceof LaunchAgentRpcTimeoutError ||
        !(error instanceof LaunchAgentRpcHttpError) ||
        error.status >= 500
      ) {
        await this.bestEffortStop(handleId, "start_cancelled");
      }
      throw error;
    }
    if (
      snapshot.handleId !== handleId ||
      snapshot.launchId !== request.launchId ||
      snapshot.processId !== request.process.processId
    ) {
      await this.bestEffortStop(handleId, "start_identity_mismatch");
      throw new Error("launch_agent_rpc_start_returned_wrong_identity");
    }
    const handle = new HttpLaunchProcessHandle(this, snapshot);
    const abort = () => {
      void handle.stop("launch_aborted").catch(() => undefined);
    };
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
    void handle.exited.then(
      () => signal.removeEventListener("abort", abort),
      () => signal.removeEventListener("abort", abort),
    );
    return handle;
  }

  async poll(handleId: string): Promise<LaunchAgentRpcProcessSnapshot> {
    assertHandleId(handleId);
    return this.requestSnapshot(
      `/v1/processes/${handleId}`,
      "GET",
      undefined,
      undefined,
      this.pollRequestTimeoutMs,
      "poll",
    );
  }

  async health(signal?: AbortSignal): Promise<LaunchAgentRpcHealth> {
    const value = await this.requestJson(
      "/healthz",
      "GET",
      undefined,
      signal,
      this.healthTimeoutMs,
      "health",
    );
    return validateHealth(value);
  }

  async physicalEvidence(
    nonce: string,
    signal?: AbortSignal,
  ): Promise<PhysicalProbeV1> {
    assertPhysicalProbeNonce(nonce);
    const value = await this.requestJson(
      "/v1/physical-evidence",
      "POST",
      { schema: PHYSICAL_PROBE_REQUEST_SCHEMA, nonce },
      signal,
      this.physicalEvidenceTimeoutMs,
      "physical_evidence",
    );
    validatePhysicalProbe(value, nonce);
    return structuredClone(value);
  }

  async stopRemote(
    handleId: string,
    reason: string,
    timeoutMs = this.requestTimeoutMs,
  ): Promise<LaunchAgentRpcProcessSnapshot> {
    assertHandleId(handleId);
    assertReason(reason);
    return this.requestSnapshot(
      `/v1/processes/${handleId}/stop`,
      "POST",
      { schema: STOP_SCHEMA, reason },
      undefined,
      timeoutMs,
      "stop",
    );
  }

  pollDelay(): Promise<void> {
    return delay(this.pollIntervalMs);
  }

  pollErrorLimit(): number {
    return this.maxConsecutivePollErrors;
  }

  private async bestEffortStop(handleId: string, reason: string): Promise<void> {
    try {
      await this.stopRemote(handleId, reason, this.cleanupRequestTimeoutMs);
    } catch {
      // 404 means the cancelled start never reached the daemon. Network
      // failure is bounded by cleanupRequestTimeoutMs and cannot hide the root
      // error.
    }
  }

  private async requestSnapshot(
    path: string,
    method: "GET" | "POST",
    body: unknown,
    externalSignal: AbortSignal | undefined,
    timeoutMs: number,
    operation: string,
  ): Promise<LaunchAgentRpcProcessSnapshot> {
    const value = await this.requestJson(
      path,
      method,
      body,
      externalSignal,
      timeoutMs,
      operation,
    );
    return validateSnapshot(value, this.maxOutputBytes);
  }

  private async requestJson(
    path: string,
    method: "GET" | "POST",
    body: unknown,
    externalSignal: AbortSignal | undefined,
    timeoutMs: number,
    operation: string,
  ): Promise<unknown> {
    const controller = new AbortController();
    const removeAbort = linkAbortSignal(externalSignal, controller);
    const timer = setTimeout(
      () => controller.abort(new LaunchAgentRpcTimeoutError(operation, timeoutMs)),
      timeoutMs,
    );
    try {
      let response: Response;
      try {
        response = await fetch(`${this.baseUrl}${path}`, {
          method,
          headers: {
            accept: "application/json",
            ...(this.authToken === undefined
              ? {}
              : { authorization: `Bearer ${this.authToken}` }),
            ...(body === undefined ? {} : { "content-type": "application/json" }),
          },
          ...(body === undefined ? {} : { body: JSON.stringify(body) }),
          signal: controller.signal,
        });
      } catch (error) {
        if (controller.signal.aborted) {
          throw cancellationFromSignal(controller.signal);
        }
        throw normalizeError(error);
      }
      // JSON escaping can turn one UTF-8 byte into six response bytes (for
      // example U+0000 becomes "\\u0000"). Account for both captured streams
      // so every snapshot that the daemon is allowed to emit is readable.
      const responseLimit = this.maxOutputBytes * 12 + 128 * 1024;
      let text: string;
      try {
        text = await readResponseText(response, responseLimit);
      } catch (error) {
        // fetch() may resolve after the headers and only reject while the body
        // is being read. Preserve the configured timeout/cancellation reason
        // in that case instead of leaking a generic AbortError.
        if (controller.signal.aborted) {
          throw cancellationFromSignal(controller.signal);
        }
        throw normalizeError(error);
      }
      let value: unknown;
      try {
        value = text ? JSON.parse(text) : null;
      } catch {
        throw new Error("launch_agent_rpc_response_is_not_json");
      }
      if (!response.ok) {
        throw new LaunchAgentRpcHttpError(response.status, rpcErrorMessage(value));
      }
      return value;
    } finally {
      clearTimeout(timer);
      removeAbort();
    }
  }
}

class HttpLaunchProcessHandle implements LaunchProcessHandle {
  readonly ready: Promise<void>;
  readonly exited: Promise<LaunchProcessExit>;
  private readonly readyDeferred = deferred<void>();
  private readonly exitDeferred = deferred<LaunchProcessExit>();
  private latestOutput: LaunchCapturedOutput;
  private readySettled = false;
  private exitSettled = false;
  private remoteReadyObserved = false;
  private remoteExitObserved = false;
  private stopPromise: Promise<void> | null = null;

  constructor(
    private readonly agent: HttpLaunchAgent,
    initial: LaunchAgentRpcProcessSnapshot,
  ) {
    this.ready = this.readyDeferred.promise;
    this.exited = this.exitDeferred.promise;
    this.handleId = initial.handleId;
    this.launchId = initial.launchId;
    this.processId = initial.processId;
    this.latestOutput = structuredClone(initial.output);
    this.apply(initial);
    if (!this.exitSettled) void this.pollLoop();
  }

  private readonly handleId: string;
  private readonly launchId: string;
  private readonly processId: string;

  output(): LaunchCapturedOutput {
    return structuredClone(this.latestOutput);
  }

  stop(reason: string): Promise<void> {
    assertReason(reason);
    if (this.stopPromise) return this.stopPromise;
    const operation = this.stopInternal(reason);
    const coalesced = operation.catch((error) => {
      // A transport timeout does not prove that the process stopped. Permit a
      // later retry; the daemon endpoint itself is idempotent.
      if (this.stopPromise === coalesced) this.stopPromise = null;
      throw error;
    });
    this.stopPromise = coalesced;
    return coalesced;
  }

  private async stopInternal(reason: string): Promise<void> {
    if (this.remoteExitObserved) return;
    const snapshot = await this.agent.stopRemote(this.handleId, reason);
    this.apply(snapshot);
  }

  private async pollLoop(): Promise<void> {
    let failures = 0;
    while (!this.exitSettled) {
      await this.agent.pollDelay();
      if (this.exitSettled) return;
      try {
        this.apply(await this.agent.poll(this.handleId));
        failures = 0;
      } catch (error) {
        failures += 1;
        if (failures >= this.agent.pollErrorLimit()) {
          this.fail(normalizeError(error));
          // Losing the lifecycle channel must not silently leave an
          // unsupervised remote process running. This call coalesces with the
          // supervisor's rollback stop when one is present.
          void this.stop("poll_failed").catch(() => undefined);
          return;
        }
      }
    }
  }

  private apply(snapshot: LaunchAgentRpcProcessSnapshot): void {
    if (
      snapshot.handleId !== this.handleId ||
      snapshot.launchId !== this.launchId ||
      snapshot.processId !== this.processId
    ) {
      this.fail(new Error("launch_agent_rpc_poll_identity_mismatch"));
      void this.stop("poll_identity_mismatch").catch(() => undefined);
      return;
    }
    if (this.remoteReadyObserved && !snapshot.ready) {
      this.fail(new Error("launch_agent_rpc_poll_lifecycle_regressed"));
      void this.stop("poll_lifecycle_regressed").catch(() => undefined);
      return;
    }
    this.latestOutput = structuredClone(snapshot.output);
    if (snapshot.ready) this.remoteReadyObserved = true;
    if (snapshot.ready && !this.readySettled) {
      this.readySettled = true;
      this.readyDeferred.resolve();
    }
    if (snapshot.exit !== null) this.remoteExitObserved = true;
    if (snapshot.exit !== null && !this.exitSettled) {
      if (!this.readySettled) {
        this.readySettled = true;
        this.readyDeferred.reject(
          new LaunchProcessExitedError(this.processId, snapshot.exit),
        );
      }
      this.exitSettled = true;
      this.exitDeferred.resolve(structuredClone(snapshot.exit));
    }
  }

  private fail(error: Error): void {
    if (!this.readySettled) {
      this.readySettled = true;
      this.readyDeferred.reject(error);
    }
    if (!this.exitSettled) {
      this.exitSettled = true;
      this.exitDeferred.reject(error);
    }
  }
}

export interface LaunchAgentRpcServerOptions {
  agent: LaunchAgent;
  nodeId?: string;
  /** Exact sealed runtime source. Null keeps the daemon usable but unverified. */
  buildIdentity?: NativeBuildIdentity | null;
  /** Optional bearer credential required on every HTTP route. */
  authToken?: string;
  /**
   * Complete, compiler-sealed launches whose exact node-local processes may
   * be started. An empty or malformed list is rejected rather than treated as
   * an open control plane.
   */
  allowedLaunchDescriptions?: readonly unknown[];
  /** Fixed read-only host/GPU probe; never receives argv or a command string. */
  physicalProbe?: PhysicalProbeCollector;
  physicalProbeTimeoutMs?: number;
  maxBodyBytes?: number;
  maxOutputBytesPerStream?: number;
  startTimeoutMs?: number;
  stopTimeoutMs?: number;
  maxProcesses?: number;
}

export interface LaunchAgentRpcServerAddress {
  host: string;
  port: number;
  url: string;
}

interface ServerEntry {
  handleId: string;
  fingerprint: string;
  request: LaunchAgentStartRequest;
  controller: AbortController;
  startPromise: Promise<void>;
  handle: LaunchProcessHandle | null;
  state: RpcProcessState;
  ready: boolean;
  exit: LaunchProcessExit | null;
  startError: Error | null;
  startErrorStatus: number;
  stopPromise: Promise<void> | null;
}

/**
 * Small HTTP daemon around an injected LaunchAgent (normally LocalProcessAgent).
 * It retains one deterministic entry per logical process, so retries cannot
 * spawn a second child. The daemon never executes command strings itself.
 */
export class LaunchAgentRpcServer {
  private readonly agent: LaunchAgent;
  private readonly nodeId: string | undefined;
  private readonly buildIdentity: NativeBuildIdentity | null;
  private readonly authTokenDigest: Buffer | undefined;
  private readonly allowedStartFingerprints: ReadonlyMap<string, string> | undefined;
  private readonly physicalProbe: PhysicalProbeCollector | undefined;
  private readonly physicalProbeTimeoutMs: number;
  private readonly maxBodyBytes: number;
  private readonly maxOutputBytes: number;
  private readonly startTimeoutMs: number;
  private readonly stopTimeoutMs: number;
  private readonly maxProcesses: number;
  private readonly entries = new Map<string, ServerEntry>();
  private readonly server: Server;

  constructor(options: LaunchAgentRpcServerOptions) {
    if (!options || typeof options.agent?.start !== "function") {
      throw new Error("launch_agent_rpc_server_agent_is_required");
    }
    this.agent = options.agent;
    if (options.nodeId !== undefined) assertIdentifier(options.nodeId, "nodeId");
    this.nodeId = options.nodeId;
    this.buildIdentity = options.buildIdentity === undefined
      ? null
      : nativeBuildIdentitySchema.parse(options.buildIdentity);
    const authToken = normalizeOptionalAuthToken(options.authToken);
    this.authTokenDigest =
      authToken === undefined ? undefined : launchAgentAuthTokenDigest(authToken);
    this.allowedStartFingerprints = normalizeAllowedStartFingerprints(
      options.allowedLaunchDescriptions,
      this.nodeId,
    );
    if (
      options.physicalProbe !== undefined &&
      typeof options.physicalProbe.collect !== "function"
    ) {
      throw new Error("launch_agent_rpc_physical_probe_is_invalid");
    }
    this.physicalProbe = options.physicalProbe;
    this.physicalProbeTimeoutMs = boundedInteger(
      options.physicalProbeTimeoutMs ?? 30_000,
      1,
      300_000,
      "launch_agent_rpc_physical_probe_timeout_is_invalid",
    );
    this.maxBodyBytes = boundedInteger(
      options.maxBodyBytes ?? 2 * 1024 * 1024,
      1_024,
      32 * 1024 * 1024,
      "launch_agent_rpc_body_limit_is_invalid",
    );
    this.maxOutputBytes = boundedInteger(
      options.maxOutputBytesPerStream ?? 64 * 1024,
      1,
      16 * 1024 * 1024,
      "launch_agent_rpc_output_limit_is_invalid",
    );
    this.startTimeoutMs = boundedInteger(
      options.startTimeoutMs ?? 30_000,
      1,
      300_000,
      "launch_agent_rpc_start_timeout_is_invalid",
    );
    this.stopTimeoutMs = boundedInteger(
      options.stopTimeoutMs ?? 15_000,
      1,
      300_000,
      "launch_agent_rpc_stop_timeout_is_invalid",
    );
    this.maxProcesses = boundedInteger(
      options.maxProcesses ?? 1_024,
      1,
      1_000_000,
      "launch_agent_rpc_process_limit_is_invalid",
    );
    this.server = createServer((request, response) => {
      void this.route(request, response);
    });
    this.server.requestTimeout = 30_000;
    this.server.headersTimeout = 10_000;
    this.server.keepAliveTimeout = 5_000;
  }

  listen(port = 0, host = "127.0.0.1"): Promise<LaunchAgentRpcServerAddress> {
    if (!Number.isInteger(port) || port < 0 || port > 65_535) {
      return Promise.reject(new Error("launch_agent_rpc_listen_port_is_invalid"));
    }
    if (typeof host !== "string" || !host.trim()) {
      return Promise.reject(new Error("launch_agent_rpc_listen_host_is_invalid"));
    }
    const bindHost = host.trim();
    if (!isLoopbackBindHost(bindHost) && this.authTokenDigest === undefined) {
      return Promise.reject(
        new Error("launch_agent_rpc_non_loopback_requires_authentication"),
      );
    }
    if (!isLoopbackBindHost(bindHost) && this.allowedStartFingerprints === undefined) {
      return Promise.reject(
        new Error("launch_agent_rpc_non_loopback_requires_launch_allowlist"),
      );
    }
    return new Promise((resolve, reject) => {
      const onError = (error: Error) => reject(error);
      this.server.once("error", onError);
      this.server.listen(port, bindHost, () => {
        this.server.removeListener("error", onError);
        const address = this.server.address();
        if (!address || typeof address === "string") {
          reject(new Error("launch_agent_rpc_has_no_tcp_address"));
          return;
        }
        const advertisedHost =
          bindHost === "0.0.0.0"
            ? "127.0.0.1"
            : bindHost === "::"
              ? "::1"
              : bindHost;
        resolve({
          host: bindHost,
          port: address.port,
          url: `http://${formatUrlHost(advertisedHost)}:${address.port}`,
        });
      });
    });
  }

  async close(reason = "daemon_shutdown"): Promise<void> {
    for (const entry of this.entries.values()) {
      if (!entry.handle && entry.exit === null && !entry.controller.signal.aborted) {
        entry.controller.abort(new LaunchCancelledError(reason));
      }
    }
    await Promise.all(
      [...this.entries.values()].map((entry) =>
        entry.startPromise.catch(() => undefined),
      ),
    );
    const stops = [...this.entries.values()].map(async (entry) => {
      if (!entry.handle || entry.exit !== null) return;
      try {
        await operationWithTimeout(
          entry.handle.stop(reason),
          undefined,
          this.stopTimeoutMs,
          "daemon_close",
        );
      } catch {
        // Closing the listener must not hang on a broken injected agent.
      }
    });
    await Promise.all(stops);
    if (!this.server.listening) return;
    await new Promise<void>((resolve, reject) => {
      this.server.close((error) => (error ? reject(error) : resolve()));
      this.server.closeIdleConnections();
      this.server.closeAllConnections();
    });
  }

  private async route(request: IncomingMessage, response: ServerResponse): Promise<void> {
    try {
      if (!this.isAuthorized(request)) {
        throw new RpcRouteError(401, "launch_agent_rpc_unauthorized");
      }
      const url = new URL(request.url ?? "/", "http://launch-agent.invalid");
      if (url.search || url.hash) throw new RpcRouteError(400, "query_is_not_supported");
      if (request.method === "GET" && url.pathname === "/healthz") {
        const activeProcesses = [...this.entries.values()].filter(
          (entry) => entry.exit === null,
        ).length;
        sendJson(response, 200, {
          schema: HEALTH_SCHEMA,
          agentId: this.agent.id,
          nodeId: this.nodeId ?? null,
          buildIdentity: this.buildIdentity,
          activeProcesses,
          retainedTombstones: this.entries.size - activeProcesses,
        });
        return;
      }
      if (request.method === "POST" && url.pathname === "/v1/physical-evidence") {
        await this.physicalEvidenceRoute(request, response);
        return;
      }
      if (request.method === "POST" && url.pathname === "/v1/processes") {
        await this.startRoute(request, response);
        return;
      }
      const match = /^\/v1\/processes\/([a-f0-9]{48})(\/stop)?$/.exec(url.pathname);
      if (match) {
        const handleId = match[1]!;
        if (request.method === "GET" && !match[2]) {
          const entry = this.requireEntry(handleId);
          sendJson(response, 200, this.snapshot(entry));
          return;
        }
        if (request.method === "POST" && match[2] === "/stop") {
          await this.stopRoute(request, response, handleId);
          return;
        }
      }
      throw new RpcRouteError(404, "route_not_found");
    } catch (error) {
      const normalized = normalizeRouteError(error);
      if (normalized.status === 401) {
        response.setHeader("www-authenticate", 'Bearer realm="gdlp-launch-agent"');
      }
      sendJson(response, normalized.status, {
        schema: ERROR_SCHEMA,
        error: normalized.message,
      });
    }
  }

  private isAuthorized(request: IncomingMessage): boolean {
    if (this.authTokenDigest === undefined) return true;
    const header = request.headers.authorization;
    const match = /^Bearer ([\x21-\x7e]+)$/i.exec(header ?? "");
    const candidateDigest = launchAgentAuthTokenDigest(match?.[1] ?? "");
    return timingSafeEqual(candidateDigest, this.authTokenDigest);
  }

  private async physicalEvidenceRoute(
    request: IncomingMessage,
    response: ServerResponse,
  ): Promise<void> {
    if (this.physicalProbe === undefined) {
      throw new RpcRouteError(404, "physical_probe_is_not_configured");
    }
    requireJson(request);
    const value = await readJsonBody(request, Math.min(this.maxBodyBytes, 4_096));
    assertRecord(value, "physical_probe_request");
    assertExactKeys(
      value,
      ["schema", "nonce"],
      [],
      "physical_probe_request",
    );
    if (value.schema !== PHYSICAL_PROBE_REQUEST_SCHEMA) {
      throw new RpcRouteError(400, "physical_probe_request_schema_is_invalid");
    }
    try {
      assertPhysicalProbeNonce(value.nonce);
    } catch {
      throw new RpcRouteError(400, "physical_probe_nonce_is_invalid");
    }
    const nonce = value.nonce as string;
    const controller = new AbortController();
    let probe: PhysicalProbeV1;
    try {
      probe = await operationWithTimeout(
        Promise.resolve().then(() => this.physicalProbe!.collect(nonce, controller.signal)),
        undefined,
        this.physicalProbeTimeoutMs,
        "physical_probe",
      );
    } catch (error) {
      if (!controller.signal.aborted) controller.abort(error);
      throw error;
    }
    validatePhysicalProbe(probe, nonce);
    sendJson(response, 200, probe);
  }

  private async startRoute(
    request: IncomingMessage,
    response: ServerResponse,
  ): Promise<void> {
    requireJson(request);
    const value = await readJsonBody(request, this.maxBodyBytes);
    assertRecord(value, "launch_agent_rpc_start_envelope");
    assertExactKeys(value, ["schema", "request"], [], "launch_agent_rpc_start_envelope");
    if (value.schema !== START_SCHEMA) throw new RpcRouteError(400, "start_schema_is_invalid");
    validateLaunchAgentRpcStartRequest(value.request);
    const startRequest = structuredClone(value.request);
    if (this.nodeId !== undefined && startRequest.nodeId !== this.nodeId) {
      throw new RpcRouteError(422, "launch_node_does_not_match_daemon");
    }
    if (!this.isAllowedStart(startRequest)) {
      throw new RpcRouteError(403, "launch_agent_rpc_start_is_not_allowed");
    }
    const handleId = launchAgentRpcHandleId(startRequest);
    const fingerprint = stableJson(startRequest);
    let entry = this.entries.get(handleId);
    let created = false;
    if (entry) {
      if (entry.fingerprint !== fingerprint) {
        throw new RpcRouteError(409, "idempotency_key_conflict");
      }
    } else {
      this.pruneExitedEntriesForCapacity();
      if (this.entries.size >= this.maxProcesses) {
        throw new RpcRouteError(503, "launch_agent_process_capacity_reached");
      }
      created = true;
      entry = {
        handleId,
        fingerprint,
        request: startRequest,
        controller: new AbortController(),
        startPromise: Promise.resolve(),
        handle: null,
        state: "starting",
        ready: false,
        exit: null,
        startError: null,
        startErrorStatus: 500,
        stopPromise: null,
      };
      this.entries.set(handleId, entry);
      entry.startPromise = this.startEntry(entry);
    }
    try {
      await entry.startPromise;
    } catch {
      throw new RpcRouteError(
        entry.startErrorStatus,
        entry.startError?.message ?? "launch_agent_start_failed",
      );
    }
    sendJson(response, created ? 201 : 200, this.snapshot(entry));
  }

  private isAllowedStart(request: LaunchAgentStartRequest): boolean {
    if (this.allowedStartFingerprints === undefined) return true;
    const expected = this.allowedStartFingerprints.get(allowedStartKey(request));
    return expected !== undefined && expected === stableJson(request);
  }

  private async startEntry(entry: ServerEntry): Promise<void> {
    let returnedHandle: LaunchProcessHandle | null = null;
    try {
      const startOperation = this.agent.start(
        structuredClone(entry.request),
        entry.controller.signal,
      );
      // An injected agent is expected to honor AbortSignal, but a late handle
      // must still be stopped if it ignores cancellation or the start timeout.
      void startOperation.then(
        (lateHandle) => {
          queueMicrotask(() => {
            if (
              entry.handle === null &&
              (entry.controller.signal.aborted || entry.exit !== null)
            ) {
              void bestEffortStopHandle(lateHandle, "late_start_cancelled");
            }
          });
        },
        () => undefined,
      );
      const handle = await operationWithTimeout(
        startOperation,
        entry.controller.signal,
        this.startTimeoutMs,
        "server_start",
      );
      returnedHandle = handle;
      validateProcessHandle(handle, entry.request.process.processId);
      entry.handle = handle;
      void handle.ready.then(
        () => {
          if (entry.exit !== null) return;
          entry.ready = true;
          entry.state = "ready";
        },
        (error) => this.finishWithError(entry, normalizeError(error)),
      );
      void handle.exited.then(
        (exit) => this.finish(entry, sanitizeExit(exit)),
        (error) => this.finishWithError(entry, normalizeError(error)),
      );
    } catch (error) {
      const normalized = normalizeError(error);
      if (!entry.controller.signal.aborted) entry.controller.abort(normalized);
      if (returnedHandle && entry.handle === null) {
        void bestEffortStopHandle(returnedHandle, "invalid_start_handle");
      }
      entry.startError = normalized;
      entry.startErrorStatus =
        normalized instanceof LaunchAgentRpcTimeoutError ? 504 : 500;
      this.finishWithError(entry, normalized);
      throw normalized;
    }
  }

  private async stopRoute(
    request: IncomingMessage,
    response: ServerResponse,
    handleId: string,
  ): Promise<void> {
    requireJson(request);
    const value = await readJsonBody(request, this.maxBodyBytes);
    assertRecord(value, "launch_agent_rpc_stop_envelope");
    assertExactKeys(value, ["schema", "reason"], [], "launch_agent_rpc_stop_envelope");
    if (value.schema !== STOP_SCHEMA) throw new RpcRouteError(400, "stop_schema_is_invalid");
    assertReason(value.reason);
    const entry = this.requireEntry(handleId);
    if (!entry.handle && entry.exit === null) {
      entry.controller.abort(new LaunchCancelledError(`rpc_stop:${value.reason}`));
    }
    try {
      await entry.startPromise;
    } catch {
      sendJson(response, 200, this.snapshot(entry));
      return;
    }
    if (entry.exit === null && entry.handle) {
      entry.stopPromise ??= operationWithTimeout(
        entry.handle.stop(value.reason),
        undefined,
        this.stopTimeoutMs,
        "server_stop",
      );
      try {
        await entry.stopPromise;
      } catch (error) {
        entry.stopPromise = null;
        const normalized = normalizeError(error);
        throw new RpcRouteError(
          normalized instanceof LaunchAgentRpcTimeoutError ? 504 : 500,
          normalized.message,
        );
      }
    }
    sendJson(response, 200, this.snapshot(entry));
  }

  private requireEntry(handleId: string): ServerEntry {
    const entry = this.entries.get(handleId);
    if (!entry) throw new RpcRouteError(404, "launch_process_not_found");
    return entry;
  }

  private pruneExitedEntriesForCapacity(): void {
    if (this.entries.size < this.maxProcesses) return;
    // Completed entries are retained as idempotency tombstones while there is
    // room. Under pressure, evict the oldest completed tombstones but never an
    // active allocation, preventing a long-lived daemon from permanently
    // exhausting maxProcesses after enough successful launches.
    for (const [handleId, entry] of this.entries) {
      if (entry.exit === null) continue;
      this.entries.delete(handleId);
      if (this.entries.size < this.maxProcesses) return;
    }
  }

  private finish(entry: ServerEntry, exit: LaunchProcessExit): void {
    if (entry.exit !== null) return;
    entry.exit = exit;
    entry.state = "exited";
  }

  private finishWithError(entry: ServerEntry, error: Error): void {
    this.finish(entry, { code: null, signal: null, error: boundedError(error.message) });
    if (entry.handle) void bestEffortStopHandle(entry.handle, "launch_agent_error");
  }

  private snapshot(entry: ServerEntry): LaunchAgentRpcProcessSnapshot {
    return {
      schema: SNAPSHOT_SCHEMA,
      handleId: entry.handleId,
      launchId: entry.request.launchId,
      processId: entry.request.process.processId,
      state: entry.state,
      ready: entry.ready,
      exit: entry.exit === null ? null : structuredClone(entry.exit),
      output: boundedHandleOutput(entry.handle, this.maxOutputBytes),
    };
  }
}

export function launchAgentRpcHandleId(request: LaunchAgentStartRequest): string {
  validateLaunchAgentRpcStartRequest(request);
  return createHash("sha256")
    .update("gdlp-launch-agent-handle/1\0")
    .update(request.launchId)
    .update("\0")
    .update(request.process.processId)
    .digest("hex")
    .slice(0, 48);
}

/** Closed validation for the RPC boundary, including the executable argv. */
export function validateLaunchAgentRpcStartRequest(
  value: unknown,
): asserts value is LaunchAgentStartRequest {
  assertRecord(value, "launch_agent_start_request");
  assertExactKeys(
    value,
    ["launchId", "pipelineId", "nodeId", "process"],
    [],
    "launch_agent_start_request",
  );
  assertIdentifier(value.launchId, "launchId");
  assertIdentifier(value.pipelineId, "pipelineId");
  assertIdentifier(value.nodeId, "nodeId");
  validatePythonLaunchProcess(value.process, value.nodeId);
}

function validatePythonLaunchProcess(value: unknown, nodeId: string): asserts value is PythonLaunchProcess {
  assertRecord(value, "python_launch_process");
  if (!(["cell-member", "remote-stage", "root-engine"] as unknown[]).includes(value.kind)) {
    throw new Error("python_launch_process_kind_is_invalid");
  }
  const processKind = value.kind as PythonLaunchProcess["kind"];
  const common = [
    "kind",
    "launchIndex",
    "processId",
    "routeId",
    "phases",
    "logicalPlanIds",
    "stageId",
    "stageIndex",
    "layerStart",
    "layerEnd",
    "totalLayers",
    "codec",
    "sealedWaveTokens",
    "maxPrefillChunkTokens",
    "anchor",
    "members",
    "macroWave",
    "isolation",
    "command",
  ];
  const variant =
    value.kind === "cell-member"
      ? [
          "rank",
          "fixturePath",
          "pipelineSnapshotIdentity",
          "collectiveBackend",
          "computeDtype",
          "device",
          "cellAnchor",
          "worldSize",
          "operationTimeoutSeconds",
          "startupTimeoutSeconds",
        ]
      : value.kind === "remote-stage"
        ? ["downstream", "returnEndpoint", "cell", "nativeGguf"]
        : [
            "nativeGguf",
            "boundaries",
            "firstRemoteStage",
            "apiEndpoint",
            "returnBindHost",
            "returnEndpoint",
            "prefill",
            "decode",
          ];
  assertExactKeys(value, [...common, ...variant], [], "python_launch_process");
  validateExecutorIsolationPolicy(value.isolation);
  for (const name of ["launchIndex", "stageIndex", "layerStart", "layerEnd", "totalLayers"] as const) {
    assertInteger(value[name], 0, Number.MAX_SAFE_INTEGER, `python_launch_${name}`);
  }
  assertInteger(value.sealedWaveTokens, 1, 17, "python_launch_sealedWaveTokens");
  assertInteger(
    value.maxPrefillChunkTokens,
    1,
    2_147_483_647,
    "python_launch_maxPrefillChunkTokens",
  );
  if (
    (value.totalLayers as number) < 1 ||
    (value.layerStart as number) >= (value.layerEnd as number) ||
    (value.layerEnd as number) > (value.totalLayers as number)
  ) {
    throw new Error("python_launch_layer_range_is_invalid");
  }
  for (const name of ["processId", "routeId", "stageId"] as const) {
    assertIdentifier(value[name], name);
  }
  assertExactStringTuple(value.phases, ["prefill", "decode"], "python_launch_phases");
  assertStringArray(value.logicalPlanIds, "logicalPlanIds", 2, 2);
  assertIdentifier(value.codec, "codec");
  validateAnchor(value.anchor, "anchor");
  if ((value.anchor as { memberId: string }).memberId !== nodeId) {
    throw new Error("python_launch_anchor_does_not_match_node");
  }
  validateMembers(value.members);
  validateMacroWaveStage(value.macroWave);
  validateCommand(value.command, processKind);
  if (value.kind === "root-engine") validateRootEngineCommand(value.command);
  if (value.kind === "remote-stage") {
    validateSpeculativeTreeCommand(value.command, "remote_stage");
  }

  if (value.kind === "cell-member") {
    if (value.macroWave !== null) {
      throw new Error("cell_member_macro_wave_must_be_null");
    }
    validateCellMember(value);
  }
  if (value.kind === "remote-stage") validateRemoteStage(value);
  if (value.kind === "root-engine") validateRootEngine(value);
}

function validateCellMember(value: Record<string, unknown>): void {
  assertInteger(value.rank, 1, 1_000_000, "cell_member_rank");
  assertPath(value.fixturePath, "fixturePath");
  assertIdentifier(value.pipelineSnapshotIdentity, "pipelineSnapshotIdentity");
  if (value.collectiveBackend !== "gloo" && value.collectiveBackend !== "nccl") {
    throw new Error("cell_collective_backend_is_invalid");
  }
  if (!["float32", "float16", "bfloat16"].includes(String(value.computeDtype))) {
    throw new Error("cell_compute_dtype_is_invalid");
  }
  assertPath(value.device, "device");
  assertRecord(value.cellAnchor, "cellAnchor");
  assertExactKeys(value.cellAnchor, ["memberId", "controlEndpoint"], [], "cellAnchor");
  assertIdentifier(value.cellAnchor.memberId, "cellAnchor.memberId");
  validateEndpoint(value.cellAnchor.controlEndpoint, "cellAnchor.controlEndpoint");
  assertInteger(value.worldSize, 2, 1_000_000, "worldSize");
  if ((value.rank as number) >= (value.worldSize as number)) {
    throw new Error("cell_member_rank_is_outside_world");
  }
  assertPositiveFinite(value.operationTimeoutSeconds, "operationTimeoutSeconds");
  assertPositiveFinite(value.startupTimeoutSeconds, "startupTimeoutSeconds");
}

function validateRemoteStage(value: Record<string, unknown>): void {
  if (value.downstream !== null) validateDownstream(value.downstream, "downstream");
  validateEndpoint(value.returnEndpoint, "returnEndpoint");
  if (value.cell !== null) validateCell(value.cell);
  if (value.nativeGguf !== null) validateNativeGgufStage(value.nativeGguf, value);
}

function validateNativeGgufStage(
  value: unknown,
  launch: Record<string, unknown>,
): void {
  assertRecord(value, "nativeGguf");
  assertExactKeys(
    value,
    [
      "packagePath",
      "packageId",
      "modelIdentity",
      "modelSource",
      "modelRevision",
      "layerStart",
      "layerEnd",
      "totalLayers",
    ],
    [],
    "nativeGguf",
  );
  assertPath(value.packagePath, "nativeGguf.packagePath");
  assertSha256(value.packageId, "nativeGguf.packageId");
  if (
    typeof value.modelIdentity !== "string" ||
    !/^sha256:[0-9a-f]{64}$/.test(value.modelIdentity)
  ) {
    throw new Error("nativeGguf.modelIdentity_is_invalid");
  }
  assertPath(value.modelSource, "nativeGguf.modelSource");
  if (value.modelRevision !== null) {
    assertPath(value.modelRevision, "nativeGguf.modelRevision", 1_024);
  }
  assertInteger(value.layerStart, 0, Number.MAX_SAFE_INTEGER, "nativeGguf.layerStart");
  assertInteger(value.layerEnd, 1, Number.MAX_SAFE_INTEGER, "nativeGguf.layerEnd");
  assertInteger(value.totalLayers, 1, Number.MAX_SAFE_INTEGER, "nativeGguf.totalLayers");
  if (
    value.layerStart !== launch.layerStart ||
    value.layerEnd !== launch.layerEnd ||
    value.totalLayers !== launch.totalLayers
  ) {
    throw new Error("native_gguf_layer_range_does_not_match_launch");
  }
  if (
    launch.macroWave !== null ||
    (launch.kind === "remote-stage" && launch.cell !== null)
  ) {
    throw new Error("native_gguf_stage_has_conflicting_execution");
  }
  if (!Array.isArray(launch.members) || launch.members.length !== 1) {
    throw new Error("native_gguf_stage_requires_single_member");
  }
  validateNativeGgufCommand(launch.command, value);
}

function validateNativeGgufCommand(
  value: unknown,
  binding: Record<string, unknown>,
): void {
  assertRecord(value, "native_gguf_command");
  if (!Array.isArray(value.args)) {
    throw new Error("native_gguf_command_args_are_invalid");
  }
  assertExactCommandStringFlag(
    value.args,
    "--native-gguf-package",
    binding.packagePath,
    "native_gguf_package",
  );
  assertExactCommandStringFlag(
    value.args,
    "--native-gguf-package-id",
    binding.packageId,
    "native_gguf_package_id",
  );
  assertExactCommandStringFlag(
    value.args,
    "--stage-package-identity",
    `sha256:${String(binding.packageId)}`,
    "native_gguf_stage_package_identity",
  );
  assertExactCommandStringFlag(
    value.args,
    "--model-artifact-identity",
    binding.modelIdentity,
    "native_gguf_model_identity",
  );
  assertExactCommandStringFlag(
    value.args,
    "--model-canonical-source",
    binding.modelSource,
    "native_gguf_model_source",
  );
  const revisionPositions = value.args.flatMap(
    (argument, index) => argument === "--model-canonical-revision" ? [index] : [],
  );
  if (binding.modelRevision === null) {
    if (revisionPositions.length !== 0) {
      throw new Error("native_gguf_model_revision_flag_is_invalid");
    }
  } else if (
    revisionPositions.length !== 1 ||
    value.args[revisionPositions[0]! + 1] !== binding.modelRevision
  ) {
    throw new Error("native_gguf_model_revision_flag_is_invalid");
  }
  const pipelinePositions = value.args.flatMap(
    (argument, index) => argument === "--pipeline-snapshot-identity" ? [index] : [],
  );
  if (pipelinePositions.length !== 1) {
    throw new Error("native_gguf_pipeline_identity_flag_is_invalid");
  }
  assertUint64String(
    value.args[pipelinePositions[0]! + 1],
    "native_gguf_pipeline_identity",
  );
  for (const incompatible of [
    "--native_stage-package",
    "--ram-moe-artifact-schema",
    "--cell-fixture",
  ]) {
    if (value.args.includes(incompatible)) {
      throw new Error("native_gguf_command_backend_is_not_exclusive");
    }
  }
}

function validateRootEngine(value: Record<string, unknown>): void {
  if (value.stageIndex !== 0) throw new Error("root_engine_stage_index_is_invalid");
  if (value.nativeGguf !== null) validateNativeGgufStage(value.nativeGguf, value);
  assertIntegerArray(value.boundaries, "boundaries", 2);
  validateDownstream(value.firstRemoteStage, "firstRemoteStage");
  validateEndpoint(value.apiEndpoint, "apiEndpoint");
  assertPath(value.returnBindHost, "returnBindHost");
  validateEndpoint(value.returnEndpoint, "returnEndpoint");
  validatePrefill(value.prefill);
  validateDecode(value.decode);
  const prefill = value.prefill as Record<string, unknown>;
  const decode = value.decode as Record<string, unknown>;
  if (
    macroWavePlanExecutionIdentity(prefill.macroWave) !==
    macroWavePlanExecutionIdentity(decode.macroWave)
  ) {
    throw new Error("root_engine_macro_wave_phase_contracts_do_not_match");
  }
  if ((value.macroWave === null) !== (prefill.macroWave === null)) {
    throw new Error("root_engine_macro_wave_stage_contract_is_inconsistent");
  }
}

function macroWavePlanExecutionIdentity(value: unknown): string {
  if (value === null) return "null";
  const contract = value as Record<string, unknown>;
  return stableJson({
    schema: contract.schema,
    routeKind: contract.routeKind,
    waveTokens: contract.waveTokens,
    expectedCommittedTokensPerWave: contract.expectedCommittedTokensPerWave,
  });
}

function validateCommand(
  value: unknown,
  kind: PythonLaunchProcess["kind"],
): void {
  assertRecord(value, "command");
  assertExactKeys(value, ["executable", "args"], [], "command");
  assertPath(value.executable, "command.executable");
  const executableLeaf = value.executable
    .replaceAll("\\", "/")
    .split("/")
    .at(-1)!
    .toLowerCase();
  if (!/^python(?:3(?:\.\d+)?)?(?:\.exe)?$/.test(executableLeaf)) {
    throw new Error("command_executable_must_be_python");
  }
  if (
    !Array.isArray(value.args) ||
    value.args.length < 3 ||
    value.args.length > 2_048
  ) {
    throw new Error("command_args_are_invalid");
  }
  const expectedModule =
    kind === "cell-member"
      ? MYCELLIOS_CELL_MEMBER_MODULE
      : kind === "remote-stage"
        ? MYCELLIOS_STAGE_MODULE
        : MYCELLIOS_SERVER_MODULE;
  if (
    value.args[0] !== "-u" ||
    value.args[1] !== "-m" ||
    value.args[2] !== expectedModule
  ) {
    throw new Error(`command_entrypoint_must_be_mycellios_native:${kind}`);
  }
  for (const argument of value.args) {
    assertArgument(argument);
    const lowered = argument.toLowerCase();
    for (const [prefix, backend] of [
      ["--native_stage", "native_stage"],
      ["--external-gguf-runtime", "external GGUF runtime"],
      ["--local-model-runtime", "local-model-runtime"],
      ["--model-serving-runtime", "model-serving-runtime"],
    ] as const) {
      if (lowered.startsWith(prefix)) {
        throw new Error(`external_backend_command_is_not_allowed:${backend}`);
      }
    }
    for (const [moduleName, backend] of [
      ["distributed_runtime.native_stage", "native_stage"],
      ["distributed_runtime.external_gguf_runtime", "external GGUF runtime"],
    ] as const) {
      if (lowered === moduleName || lowered.startsWith(`${moduleName}_`)) {
        throw new Error(`external_backend_command_is_not_allowed:${backend}`);
      }
    }
  }
}

function validateRootEngineCommand(value: unknown): void {
  assertRecord(value, "root_engine_command");
  if (!Array.isArray(value.args)) throw new Error("root_engine_command_args_are_invalid");
  assertCommandIntegerFlag(
    value.args,
    "--prefill-inflight-chunks",
    1,
    64,
    "root_engine_prefill_inflight_chunks",
  );
  assertCommandIntegerFlag(
    value.args,
    "--prefill-inflight-bytes",
    1,
    1024 * 1024 * 1024,
    "root_engine_prefill_inflight_bytes",
  );
  const physicalSpeculativeTreeEnabled = validateSpeculativeTreeCommand(
    value,
    "root_engine",
  );
  const speculativeInflightWaves = assertOptionalCommandIntegerFlag(
    value.args,
    "--speculative-inflight-waves",
    1,
    16,
    "root_engine_speculative_inflight_waves",
  );
  const speculativeInflightBytes = assertOptionalCommandIntegerFlag(
    value.args,
    "--speculative-inflight-bytes",
    0,
    1024 * 1024 * 1024,
    "root_engine_speculative_inflight_bytes",
  );
  if (
    (speculativeInflightWaves === undefined)
    !== (speculativeInflightBytes === undefined)
  ) {
    throw new Error("root_engine_speculative_conveyor_flags_are_incomplete");
  }
  if (
    speculativeInflightWaves !== undefined
    && speculativeInflightBytes !== undefined
  ) {
    if (speculativeInflightWaves <= 1 || speculativeInflightBytes <= 0) {
      throw new Error(
        "root_engine_speculative_conveyor_limits_must_be_disabled_or_complete",
      );
    }
    const speculation = assertCommandStringFlag(
      value.args,
      "--speculation",
      "root_engine_speculation",
    );
    if (speculation !== "ngram" && speculation !== "draft-model") {
      throw new Error(
        "root_engine_speculative_conveyor_requires_linear_speculation",
      );
    }
    if (physicalSpeculativeTreeEnabled) {
      throw new Error(
        "root_engine_speculative_conveyor_cannot_use_physical_tree_limits",
      );
    }
    const maxActiveSequences = assertCommandIntegerFlag(
      value.args,
      "--max-active-sequences",
      1,
      1_000_000,
      "root_engine_max_active_sequences",
    );
    if (maxActiveSequences !== 1) {
      throw new Error(
        "root_engine_speculative_conveyor_requires_single_active_sequence",
      );
    }
  }
}

function validateSpeculativeTreeCommand(value: unknown, prefix: string): boolean {
  assertRecord(value, `${prefix}_command`);
  if (!Array.isArray(value.args)) throw new Error(`${prefix}_command_args_are_invalid`);
  const branches = assertCommandIntegerFlag(
    value.args,
    "--max-speculative-branches",
    0,
    64,
    `${prefix}_max_speculative_branches`,
  );
  const branchTokens = assertCommandIntegerFlag(
    value.args,
    "--max-speculative-branch-tokens",
    0,
    1_048_576,
    `${prefix}_max_speculative_branch_tokens`,
  );
  const kvBytes = assertCommandIntegerFlag(
    value.args,
    "--max-speculative-kv-bytes",
    0,
    2 ** 40,
    `${prefix}_max_speculative_kv_bytes`,
  );
  const enabled = [branches, branchTokens, kvBytes].map((limit) => limit > 0);
  if (enabled.some(Boolean) && !enabled.every(Boolean)) {
    throw new Error(`${prefix}_speculative_tree_limits_are_incomplete`);
  }
  return enabled.every(Boolean);
}

function assertCommandIntegerFlag(
  args: unknown[],
  flag: string,
  minimum: number,
  maximum: number,
  name: string,
): number {
  const positions = args.flatMap((argument, index) => argument === flag ? [index] : []);
  if (positions.length !== 1) throw new Error(`${name}_flag_is_invalid`);
  const raw = args[positions[0]! + 1];
  if (typeof raw !== "string" || !/^(?:0|[1-9][0-9]*)$/.test(raw)) {
    throw new Error(`${name}_is_invalid`);
  }
  const parsed = Number(raw);
  assertInteger(parsed, minimum, maximum, name);
  return parsed;
}

function assertOptionalCommandIntegerFlag(
  args: unknown[],
  flag: string,
  minimum: number,
  maximum: number,
  name: string,
): number | undefined {
  const positions = args.flatMap((argument, index) => argument === flag ? [index] : []);
  if (positions.length === 0) return undefined;
  if (positions.length !== 1) throw new Error(`${name}_flag_is_invalid`);
  const raw = args[positions[0]! + 1];
  if (typeof raw !== "string" || !/^(?:0|[1-9][0-9]*)$/.test(raw)) {
    throw new Error(`${name}_is_invalid`);
  }
  const parsed = Number(raw);
  assertInteger(parsed, minimum, maximum, name);
  return parsed;
}

function assertCommandStringFlag(
  args: unknown[],
  flag: string,
  name: string,
): string {
  const positions = args.flatMap((argument, index) => argument === flag ? [index] : []);
  if (positions.length !== 1) throw new Error(`${name}_flag_is_invalid`);
  const raw = args[positions[0]! + 1];
  if (typeof raw !== "string" || raw.length < 1 || raw.length > 256) {
    throw new Error(`${name}_is_invalid`);
  }
  return raw;
}

function assertExactCommandStringFlag(
  args: unknown[],
  flag: string,
  expected: unknown,
  name: string,
): void {
  const positions = args.flatMap((argument, index) => argument === flag ? [index] : []);
  if (
    positions.length !== 1 ||
    typeof expected !== "string" ||
    args[positions[0]! + 1] !== expected
  ) {
    throw new Error(`${name}_flag_is_invalid`);
  }
}

function validateAnchor(value: unknown, name: string): void {
  assertRecord(value, name);
  assertExactKeys(value, ["memberId", "endpoint"], [], name);
  assertIdentifier(value.memberId, `${name}.memberId`);
  validateEndpoint(value.endpoint, `${name}.endpoint`);
}

function validateMembers(value: unknown): void {
  if (!Array.isArray(value) || value.length < 1 || value.length > 1_024) {
    throw new Error("python_launch_members_are_invalid");
  }
  for (const member of value) {
    assertRecord(member, "member");
    assertExactKeys(
      member,
      [
        "nodeId",
        "endpoint",
        "backend",
        "capabilities",
        "assignedMemoryBytes",
        "memoryLimitBytes",
      ],
      [],
      "member",
    );
    assertIdentifier(member.nodeId, "member.nodeId");
    validateEndpoint(member.endpoint, "member.endpoint");
    assertRecord(member.backend, "member.backend");
    assertExactKeys(member.backend, ["engine", "modelFormats", "executionModes"], ["version"], "member.backend");
    assertPath(member.backend.engine, "member.backend.engine", 256);
    if (member.backend.version !== undefined) assertPath(member.backend.version, "member.backend.version", 256);
    assertBoundedStringArray(member.backend.modelFormats, "member.backend.modelFormats", 0, 256);
    assertBoundedStringArray(member.backend.executionModes, "member.backend.executionModes", 0, 256);
    assertRecord(member.capabilities, "member.capabilities");
    assertExactKeys(
      member.capabilities,
      ["deviceKinds", "computeApis", "weightDtypes", "activationCodecs", "features"],
      [],
      "member.capabilities",
    );
    for (const field of ["deviceKinds", "computeApis", "weightDtypes", "activationCodecs", "features"] as const) {
      assertBoundedStringArray(member.capabilities[field], `member.capabilities.${field}`, 0, 512);
    }
    assertInteger(member.assignedMemoryBytes, 0, Number.MAX_SAFE_INTEGER, "assignedMemoryBytes");
    assertInteger(member.memoryLimitBytes, 1, Number.MAX_SAFE_INTEGER, "memoryLimitBytes");
  }
}

function validateEndpoint(value: unknown, name: string): void {
  assertRecord(value, name);
  assertExactKeys(value, ["host", "port"], [], name);
  assertPath(value.host, `${name}.host`, 1_024);
  assertInteger(value.port, 1, 65_535, `${name}.port`);
}

function validateDownstream(value: unknown, name: string): void {
  assertRecord(value, name);
  assertExactKeys(
    value,
    ["stageId", "stageIndex", "layerEnd", "anchorMemberId", "endpoint"],
    [],
    name,
  );
  assertIdentifier(value.stageId, `${name}.stageId`);
  assertInteger(value.stageIndex, 0, 1_000_000, `${name}.stageIndex`);
  assertInteger(value.layerEnd, 1, 1_000_000, `${name}.layerEnd`);
  assertIdentifier(value.anchorMemberId, `${name}.anchorMemberId`);
  validateEndpoint(value.endpoint, `${name}.endpoint`);
}

function validateCell(value: unknown): void {
  assertRecord(value, "cell");
  assertExactKeys(
    value,
    [
      "mode",
      "engine",
      "collectiveBackend",
      "computeDtype",
      "fixture",
      "worldSize",
      "rankMemberIds",
      "rankWeights",
      "rankDevices",
      "operationTimeoutSeconds",
    ],
    ["external"],
    "cell",
  );
  if (value.mode !== "tensor-parallel-cell" || value.engine !== "python-torch") {
    throw new Error("cell_runtime_identity_is_invalid");
  }
  if (value.collectiveBackend !== "gloo" && value.collectiveBackend !== "nccl") {
    throw new Error("cell_collective_backend_is_invalid");
  }
  if (!["float32", "float16", "bfloat16"].includes(String(value.computeDtype))) {
    throw new Error("cell_compute_dtype_is_invalid");
  }
  assertInteger(value.worldSize, 2, 1_000_000, "cell.worldSize");
  assertStringArray(value.rankMemberIds, "cell.rankMemberIds", value.worldSize as number, value.worldSize as number);
  if (!Array.isArray(value.rankWeights) || value.rankWeights.length !== value.worldSize) {
    throw new Error("cell_rank_weights_are_invalid");
  }
  for (const weight of value.rankWeights) {
    assertPositiveFinite(weight, "cell.rankWeights");
  }
  assertStringArray(value.rankDevices, "cell.rankDevices", value.worldSize as number, value.worldSize as number);
  assertPositiveFinite(value.operationTimeoutSeconds, "cell.operationTimeoutSeconds");
  assertRecord(value.fixture, "cell.fixture");
  assertExactKeys(
    value.fixture,
    ["schema", "location", "path", "layerCount", "manifestSha256", "shardSha256", "rankMemory"],
    [],
    "cell.fixture",
  );
  if (!["gdlp-llama-cell-layer/1", "gdlp-llama-cell-stage/2"].includes(String(value.fixture.schema))) {
    throw new Error("cell_fixture_schema_is_invalid");
  }
  if (value.fixture.location !== "anchor-local" && value.fixture.location !== "member-local") {
    throw new Error("cell_fixture_location_is_invalid");
  }
  assertPath(value.fixture.path, "cell.fixture.path");
  assertInteger(value.fixture.layerCount, 1, 1_000_000, "cell.fixture.layerCount");
  assertSha256(value.fixture.manifestSha256, "cell.fixture.manifestSha256");
  assertStringArray(value.fixture.shardSha256, "cell.fixture.shardSha256", value.worldSize as number, value.worldSize as number);
  for (const digest of value.fixture.shardSha256 as unknown[]) assertSha256(digest, "cell.fixture.shardSha256");
  if (!Array.isArray(value.fixture.rankMemory) || value.fixture.rankMemory.length !== value.worldSize) {
    throw new Error("cell_rank_memory_is_invalid");
  }
  for (const memory of value.fixture.rankMemory) {
    assertRecord(memory, "cell.rankMemory");
    assertExactKeys(memory, ["fixedBytes", "kvBytesPerToken", "requiredBytes"], [], "cell.rankMemory");
    assertInteger(memory.fixedBytes, 0, Number.MAX_SAFE_INTEGER, "cell.rankMemory.fixedBytes");
    assertInteger(memory.kvBytesPerToken, 0, Number.MAX_SAFE_INTEGER, "cell.rankMemory.kvBytesPerToken");
    assertInteger(memory.requiredBytes, 0, Number.MAX_SAFE_INTEGER, "cell.rankMemory.requiredBytes");
  }
  if (value.external !== undefined) validateCellExternal(value.external, value.worldSize as number);
}

function validateCellExternal(value: unknown, worldSize: number): void {
  assertRecord(value, "cell.external");
  assertExactKeys(
    value,
    [
      "rankFixturePaths",
      "controlBindHost",
      "controlAdvertiseHost",
      "controlPort",
      "distributedAdvertiseHost",
      "distributedPort",
      "startupTimeoutSeconds",
    ],
    [],
    "cell.external",
  );
  assertBoundedStringArray(
    value.rankFixturePaths,
    "cell.external.rankFixturePaths",
    worldSize,
    worldSize,
    32_768,
  );
  for (const field of ["controlBindHost", "controlAdvertiseHost", "distributedAdvertiseHost"] as const) {
    assertPath(value[field], `cell.external.${field}`, 1_024);
  }
  assertInteger(value.controlPort, 1, 65_535, "cell.external.controlPort");
  assertInteger(value.distributedPort, 1, 65_535, "cell.external.distributedPort");
  assertPositiveFinite(value.startupTimeoutSeconds, "cell.external.startupTimeoutSeconds");
}

function validatePrefill(value: unknown): void {
  assertRecord(value, "prefill");
  assertExactKeys(
    value,
    ["phase", "planId", "activationCodec", "microBatchSize", "chunkTokens", "macroWave"],
    [],
    "prefill",
  );
  if (value.phase !== "prefill") throw new Error("prefill_phase_is_invalid");
  assertIdentifier(value.planId, "prefill.planId");
  assertIdentifier(value.activationCodec, "prefill.activationCodec");
  assertInteger(value.microBatchSize, 1, 1_000_000, "prefill.microBatchSize");
  assertInteger(value.chunkTokens, 1, 1_000_000, "prefill.chunkTokens");
  validateMacroWavePlan(value.macroWave, "prefill.macroWave");
}

function validateDecode(value: unknown): void {
  assertRecord(value, "decode");
  assertExactKeys(
    value,
    [
      "phase",
      "planId",
      "activationCodec",
      "microBatchSize",
      "directTokenReturnStage",
      "speculation",
      "macroWave",
    ],
    [],
    "decode",
  );
  if (value.phase !== "decode") throw new Error("decode_phase_is_invalid");
  assertIdentifier(value.planId, "decode.planId");
  assertIdentifier(value.activationCodec, "decode.activationCodec");
  assertInteger(value.microBatchSize, 1, 1_000_000, "decode.microBatchSize");
  assertInteger(value.directTokenReturnStage, 0, 1_000_000, "decode.directTokenReturnStage");
  validateSpeculation(value.speculation);
  validateMacroWavePlan(value.macroWave, "decode.macroWave");
}

function validateMacroWavePlan(value: unknown, name: string): void {
  if (value === null) return;
  assertRecord(value, name);
  assertExactKeys(
    value,
    [
      "schema",
      "routeKind",
      "waveTokens",
      "expectedCommittedTokensPerWave",
      "projection",
    ],
    [],
    name,
  );
  if (value.schema !== "gdlp-macro-wave-plan/1") {
    throw new Error(`${name}_schema_is_invalid`);
  }
  if (value.routeKind !== "resident-baseline" && value.routeKind !== "macro-wave") {
    throw new Error(`${name}_route_kind_is_invalid`);
  }
  assertInteger(value.waveTokens, 1, Number.MAX_SAFE_INTEGER, `${name}.waveTokens`);
  assertPositiveFinite(
    value.expectedCommittedTokensPerWave,
    `${name}.expectedCommittedTokensPerWave`,
  );
  if ((value.expectedCommittedTokensPerWave as number) > (value.waveTokens as number)) {
    throw new Error(`${name}_committed_tokens_exceed_wave`);
  }
  // The compiler currently emits only the exact one-position subset. Keep the
  // RPC boundary aligned with that executable ABI rather than accepting a
  // future contract that the Python process would silently ignore.
  if (value.waveTokens !== 1 || value.expectedCommittedTokensPerWave !== 1) {
    throw new Error(`${name}_is_not_executable_by_current_runtime`);
  }
  assertRecord(value.projection, `${name}.projection`);
  const projectionFields = [
    "ttftMs",
    "tpotMs",
    "responseTimeMs",
    "pathDecodeMs",
    "pipelineCycleMs",
    "tokensPerSecondPerSequence",
    "aggregateTokensPerSecond",
    "networkBytesPerOutputToken",
    "rawActiveWeightBytesPerOutputToken",
    "expectedWeightCacheMissBytesPerOutputToken",
    "routeAvailability",
  ];
  assertExactKeys(value.projection, projectionFields, [], `${name}.projection`);
  for (const field of projectionFields) {
    assertFiniteRange(
      value.projection[field],
      0,
      Number.MAX_VALUE,
      `${name}.projection.${field}`,
    );
  }
  assertFiniteRange(
    value.projection.routeAvailability,
    0,
    1,
    `${name}.projection.routeAvailability`,
  );
}

function validateMacroWaveStage(value: unknown): void {
  if (value === null) return;
  assertRecord(value, "macroWave");
  assertExactKeys(
    value,
    [
      "mode",
      "schema",
      "memoryMode",
      "residentKind",
      "budgets",
      "requirements",
      "workingSet",
      "cachePolicy",
    ],
    ["ramArtifact"],
    "macroWave",
  );
  if (value.mode !== "macro-wave-memory" || value.schema !== "gdlp-macro-wave-stage/1") {
    throw new Error("macro_wave_stage_identity_is_invalid");
  }
  if (value.memoryMode !== "resident" && value.memoryMode !== "ram-backed") {
    throw new Error("macro_wave_memory_mode_is_invalid");
  }
  if (value.residentKind !== "layers" && value.residentKind !== "expert-shard") {
    throw new Error("macro_wave_resident_kind_is_invalid");
  }
  assertRecord(value.budgets, "macroWave.budgets");
  assertExactKeys(
    value.budgets,
    ["hostRamBytes", "vramBytes"],
    [],
    "macroWave.budgets",
  );
  assertInteger(
    value.budgets.hostRamBytes,
    1,
    Number.MAX_SAFE_INTEGER,
    "macroWave.budgets.hostRamBytes",
  );
  assertInteger(
    value.budgets.vramBytes,
    1,
    Number.MAX_SAFE_INTEGER,
    "macroWave.budgets.vramBytes",
  );

  assertRecord(value.requirements, "macroWave.requirements");
  const requirementFields = [
    "fullStageStateBytes",
    "hostRamBytes",
    "vramBytes",
    "fixedVramBytes",
    "residentParameterBudgetBytes",
    "residentStreamingTransientBytes",
    "boundedPinnedStagingReserveBytes",
    "hostRamPeakUpperBoundBytes",
    "activationBufferBytes",
    "weightBufferBytes",
    "weightBufferCopies",
  ];
  assertExactKeys(
    value.requirements,
    requirementFields,
    [],
    "macroWave.requirements",
  );
  for (const field of requirementFields) {
    assertInteger(
      value.requirements[field],
      0,
      Number.MAX_SAFE_INTEGER,
      `macroWave.requirements.${field}`,
    );
  }
  if (
    (value.requirements.hostRamBytes as number) >
      (value.budgets.hostRamBytes as number) ||
    (value.requirements.vramBytes as number) > (value.budgets.vramBytes as number)
  ) {
    throw new Error("macro_wave_stage_budget_is_exceeded");
  }

  assertRecord(value.workingSet, "macroWave.workingSet");
  const workingSetFields = [
    "totalWeightBytes",
    "residentWeightBytes",
    "totalRoutedExpertBytes",
    "activeWeightBytesPerWave",
    "largestTransferUnitBytes",
    "largestExpertBytes",
  ];
  assertExactKeys(value.workingSet, workingSetFields, [], "macroWave.workingSet");
  for (const field of workingSetFields) {
    assertInteger(
      value.workingSet[field],
      0,
      Number.MAX_SAFE_INTEGER,
      `macroWave.workingSet.${field}`,
    );
  }
  if (
    (value.workingSet.residentWeightBytes as number) >
      (value.workingSet.totalWeightBytes as number) ||
    (value.workingSet.activeWeightBytesPerWave as number) >
      (value.workingSet.totalWeightBytes as number)
  ) {
    throw new Error("macro_wave_working_set_exceeds_total_weights");
  }

  assertRecord(value.cachePolicy, "macroWave.cachePolicy");
  assertExactKeys(
    value.cachePolicy,
    [
      "kind",
      "capacityBytes",
      "expectedHitRate",
      "expectedMissWeightBytesPerWave",
    ],
    [],
    "macroWave.cachePolicy",
  );
  if (
    value.cachePolicy.kind !== "full-resident" &&
    value.cachePolicy.kind !== "disabled" &&
    value.cachePolicy.kind !== "bounded-lru"
  ) {
    throw new Error("macro_wave_cache_kind_is_invalid");
  }
  assertInteger(
    value.cachePolicy.capacityBytes,
    0,
    Number.MAX_SAFE_INTEGER,
    "macroWave.cachePolicy.capacityBytes",
  );
  assertFiniteRange(
    value.cachePolicy.expectedHitRate,
    0,
    1,
    "macroWave.cachePolicy.expectedHitRate",
  );
  assertInteger(
    value.cachePolicy.expectedMissWeightBytesPerWave,
    0,
    Number.MAX_SAFE_INTEGER,
    "macroWave.cachePolicy.expectedMissWeightBytesPerWave",
  );

  const requirements = value.requirements;
  const workingSet = value.workingSet;
  const cachePolicy = value.cachePolicy;
  const hostRamBytes = requirements.hostRamBytes as number;
  const totalRoutedExpertBytes = workingSet.totalRoutedExpertBytes as number;
  const activeWeightBytesPerWave = workingSet.activeWeightBytesPerWave as number;
  const largestExpertBytes = workingSet.largestExpertBytes as number;
  const cacheCapacityBytes = cachePolicy.capacityBytes as number;
  const cacheExpectedHitRate = cachePolicy.expectedHitRate as number;
  if (value.memoryMode === "resident") {
    if (value.ramArtifact !== undefined) {
      throw new Error("macro_wave_resident_stage_cannot_bind_ram_artifact");
    }
    if (
      value.residentKind !== "layers" ||
      requirements.hostRamBytes !== 0 ||
      requirements.residentStreamingTransientBytes !== 0 ||
      requirements.boundedPinnedStagingReserveBytes !== 0 ||
      requirements.hostRamPeakUpperBoundBytes !== 0 ||
      requirements.weightBufferBytes !== 0 ||
      requirements.weightBufferCopies !== 0 ||
      requirements.residentParameterBudgetBytes !== workingSet.residentWeightBytes ||
      workingSet.totalRoutedExpertBytes !== 0 ||
      workingSet.activeWeightBytesPerWave !== 0 ||
      workingSet.largestTransferUnitBytes !== 0 ||
      workingSet.largestExpertBytes !== 0 ||
      workingSet.residentWeightBytes !== workingSet.totalWeightBytes ||
      cachePolicy.kind !== "full-resident" ||
      cachePolicy.capacityBytes !== workingSet.residentWeightBytes ||
      cachePolicy.expectedHitRate !== 1 ||
      cachePolicy.expectedMissWeightBytesPerWave !== 0 ||
      requirements.vramBytes !==
        (requirements.fixedVramBytes as number) +
          (requirements.activationBufferBytes as number) +
          (workingSet.residentWeightBytes as number)
    ) {
      throw new Error("macro_wave_resident_stage_contract_is_inconsistent");
    }
    return;
  }

  assertRecord(value.ramArtifact, "macroWave.ramArtifact");
  validateMacroWaveRamArtifact(value.ramArtifact);
  const artifact = value.ramArtifact;
  const requiredVram =
    (requirements.fixedVramBytes as number) +
    (requirements.activationBufferBytes as number) +
    (requirements.weightBufferBytes as number);
  if (
    value.residentKind !== "layers" ||
    requirements.vramBytes !== requiredVram ||
    hostRamBytes !== requirements.hostRamPeakUpperBoundBytes ||
    requirements.hostRamPeakUpperBoundBytes !==
      totalRoutedExpertBytes +
        (requirements.boundedPinnedStagingReserveBytes as number) +
        (requirements.residentStreamingTransientBytes as number) ||
    requirements.boundedPinnedStagingReserveBytes !== largestExpertBytes * 2 ||
    requirements.weightBufferCopies !== 2 ||
    requirements.residentParameterBudgetBytes !== workingSet.residentWeightBytes ||
    workingSet.totalWeightBytes !==
      (workingSet.residentWeightBytes as number) +
        (workingSet.totalRoutedExpertBytes as number) ||
    totalRoutedExpertBytes < 1 ||
    activeWeightBytesPerWave < 1 ||
    activeWeightBytesPerWave > totalRoutedExpertBytes ||
    largestExpertBytes < 1 ||
    workingSet.largestTransferUnitBytes !== workingSet.largestExpertBytes ||
    largestExpertBytes > activeWeightBytesPerWave ||
    requirements.weightBufferBytes !==
      (workingSet.largestExpertBytes as number) *
        (requirements.weightBufferCopies as number) ||
    artifact.largestExpertBytes !== workingSet.largestExpertBytes ||
    artifact.weightBufferCopies !== requirements.weightBufferCopies
  ) {
    throw new Error("macro_wave_ram_backed_stage_contract_is_inconsistent");
  }
  const expectedMiss = Math.ceil(
    activeWeightBytesPerWave * (1 - cacheExpectedHitRate),
  );
  if (cachePolicy.expectedMissWeightBytesPerWave !== expectedMiss) {
    throw new Error("macro_wave_cache_expectation_is_inconsistent");
  }
  if (cachePolicy.kind === "disabled") {
    if (cachePolicy.capacityBytes !== 0 || cachePolicy.expectedHitRate !== 0) {
      throw new Error("macro_wave_disabled_cache_is_inconsistent");
    }
  } else if (
    cachePolicy.kind !== "bounded-lru" ||
    cacheExpectedHitRate <= 0 ||
    cacheCapacityBytes < largestExpertBytes ||
    (requirements.vramBytes as number) + cacheCapacityBytes >
      (value.budgets.vramBytes as number)
  ) {
    throw new Error("macro_wave_bounded_cache_is_inconsistent");
  }
}

function validateMacroWaveRamArtifact(value: Record<string, unknown>): void {
  assertExactKeys(
    value,
    [
      "schema",
      "format",
      "locality",
      "loader",
      "weightEncoding",
      "sourceDtypes",
      "adapterIds",
      "expertExecutionMode",
      "largestExpertBytes",
      "weightBufferCopies",
      "fullModelMaterialization",
    ],
    [],
    "macroWave.ramArtifact",
  );
  if (
    value.schema !== "gdlp-local-safetensors-moe-stage/1" ||
    value.format !== "safetensors" ||
    value.locality !== "host-local-only" ||
    value.loader !== "selective-safetensors-ram-backed-moe" ||
    value.weightEncoding !== "floating-safetensors" ||
    value.expertExecutionMode !== "serial-exact" ||
    value.weightBufferCopies !== 2 ||
    value.fullModelMaterialization !== false ||
    !Array.isArray(value.sourceDtypes) ||
    value.sourceDtypes.join(",") !== "fp16,bf16,fp32" ||
    !Array.isArray(value.adapterIds) ||
    value.adapterIds.join(",") !==
      "transformers-qwen3-moe-v1,transformers-glm4-moe-v1"
  ) {
    throw new Error("macro_wave_ram_artifact_is_invalid");
  }
  assertInteger(
    value.largestExpertBytes,
    1,
    Number.MAX_SAFE_INTEGER,
    "macroWave.ramArtifact.largestExpertBytes",
  );
}

function validateSpeculation(value: unknown): void {
  assertRecord(value, "speculation");
  assertExactKeys(
    value,
    ["mode", "controller", "defaultStrategyId", "fallbackStrategyId", "acceptanceWindowTokens", "strategies"],
    [],
    "speculation",
  );
  if (value.mode !== "disabled" && value.mode !== "adaptive") throw new Error("speculation_mode_is_invalid");
  if (value.controller !== "fixed" && value.controller !== "acceptance-adaptive") throw new Error("speculation_controller_is_invalid");
  assertIdentifier(value.defaultStrategyId, "speculation.defaultStrategyId");
  assertIdentifier(value.fallbackStrategyId, "speculation.fallbackStrategyId");
  assertInteger(value.acceptanceWindowTokens, 1, 1_000_000, "speculation.acceptanceWindowTokens");
  if (!Array.isArray(value.strategies) || value.strategies.length < 1 || value.strategies.length > 1_024) {
    throw new Error("speculation_strategies_are_invalid");
  }
  for (const strategy of value.strategies) {
    assertRecord(strategy, "speculation.strategy");
    assertExactKeys(
      strategy,
      ["id", "kind", "maxDraftTokens", "minAcceptanceRate", "maxWasteRatio", "priority"],
      [
        "artifactId",
        "maxBranches",
        "maxBranchTokens",
        "maxKvBytes",
        "maxWaveTokens",
      ],
      "speculation.strategy",
    );
    assertIdentifier(strategy.id, "speculation.strategy.id");
    assertIdentifier(strategy.kind, "speculation.strategy.kind");
    assertInteger(strategy.maxDraftTokens, 0, 1_000_000, "speculation.strategy.maxDraftTokens");
    assertFiniteRange(strategy.minAcceptanceRate, 0, 1, "speculation.strategy.minAcceptanceRate");
    assertFiniteRange(strategy.maxWasteRatio, 0, 1, "speculation.strategy.maxWasteRatio");
    assertInteger(strategy.priority, 0, 1_000_000, "speculation.strategy.priority");
    if (strategy.artifactId !== undefined) assertIdentifier(strategy.artifactId, "speculation.strategy.artifactId");
    for (const [name, maximum] of [
      ["maxBranches", 64],
      ["maxBranchTokens", 1_048_576],
      ["maxKvBytes", 2 ** 40],
      ["maxWaveTokens", 17],
    ] as const) {
      if (strategy[name] !== undefined) {
        assertInteger(
          strategy[name],
          1,
          maximum,
          `speculation.strategy.${name}`,
        );
      }
    }
    const treeLimits = [
      strategy.maxBranches,
      strategy.maxBranchTokens,
      strategy.maxKvBytes,
      strategy.maxWaveTokens,
    ];
    if (strategy.kind === "draft-tree") {
      if (treeLimits.some((limit) => limit === undefined)) {
        throw new Error("speculation_strategy_tree_limits_are_missing");
      }
      if (
        Number(strategy.maxWaveTokens) !==
        Number(strategy.maxDraftTokens) + 1
      ) {
        throw new Error("speculation_strategy_tree_wave_is_invalid");
      }
    } else if (treeLimits.some((limit) => limit !== undefined)) {
      throw new Error("speculation_strategy_tree_limits_are_unexpected");
    }
  }
}

function validateSnapshot(value: unknown, maxOutputBytes: number): LaunchAgentRpcProcessSnapshot {
  assertRecord(value, "launch_agent_rpc_snapshot");
  assertExactKeys(
    value,
    ["schema", "handleId", "launchId", "processId", "state", "ready", "exit", "output"],
    [],
    "launch_agent_rpc_snapshot",
  );
  if (value.schema !== SNAPSHOT_SCHEMA) throw new Error("launch_agent_rpc_snapshot_schema_is_invalid");
  assertHandleId(value.handleId);
  assertIdentifier(value.launchId, "snapshot.launchId");
  assertIdentifier(value.processId, "snapshot.processId");
  if (!["starting", "ready", "exited"].includes(String(value.state))) throw new Error("launch_agent_rpc_snapshot_state_is_invalid");
  if (typeof value.ready !== "boolean") throw new Error("launch_agent_rpc_snapshot_ready_is_invalid");
  const exit = value.exit === null ? null : sanitizeExit(value.exit);
  if ((value.state === "exited") !== (exit !== null)) throw new Error("launch_agent_rpc_snapshot_exit_is_inconsistent");
  if (value.state === "starting" && value.ready) throw new Error("launch_agent_rpc_snapshot_ready_is_inconsistent");
  if (value.state === "ready" && !value.ready) throw new Error("launch_agent_rpc_snapshot_ready_is_inconsistent");
  const output = validateOutput(value.output, maxOutputBytes);
  return {
    schema: SNAPSHOT_SCHEMA,
    handleId: value.handleId,
    launchId: value.launchId,
    processId: value.processId,
    state: value.state as RpcProcessState,
    ready: value.ready,
    exit,
    output,
  };
}

function validateHealth(value: unknown): LaunchAgentRpcHealth {
  assertRecord(value, "launch_agent_rpc_health");
  assertExactKeys(
    value,
    ["schema", "agentId", "nodeId", "buildIdentity", "activeProcesses", "retainedTombstones"],
    [],
    "launch_agent_rpc_health",
  );
  if (value.schema !== HEALTH_SCHEMA) {
    throw new Error("launch_agent_rpc_health_schema_is_invalid");
  }
  assertIdentifier(value.agentId, "launch_agent_rpc_health_agent_id");
  if (value.nodeId !== null) {
    assertIdentifier(value.nodeId, "launch_agent_rpc_health_node_id");
  }
  const buildIdentity = value.buildIdentity === null
    ? null
    : nativeBuildIdentitySchema.parse(value.buildIdentity);
  if (
    !Number.isSafeInteger(value.activeProcesses) ||
    (value.activeProcesses as number) < 0
  ) {
    throw new Error("launch_agent_rpc_health_active_process_count_is_invalid");
  }
  if (
    !Number.isSafeInteger(value.retainedTombstones) ||
    (value.retainedTombstones as number) < 0
  ) {
    throw new Error("launch_agent_rpc_health_tombstone_count_is_invalid");
  }
  return {
    schema: HEALTH_SCHEMA,
    agentId: value.agentId as string,
    nodeId: value.nodeId as string | null,
    buildIdentity,
    activeProcesses: value.activeProcesses as number,
    retainedTombstones: value.retainedTombstones as number,
  };
}

function validateOutput(value: unknown, maxBytes: number): LaunchCapturedOutput {
  assertRecord(value, "launch_agent_rpc_output");
  assertExactKeys(value, ["stdout", "stderr", "stdoutTruncated", "stderrTruncated"], [], "launch_agent_rpc_output");
  if (typeof value.stdout !== "string" || Buffer.byteLength(value.stdout, "utf8") > maxBytes) throw new Error("launch_agent_rpc_stdout_is_invalid");
  if (typeof value.stderr !== "string" || Buffer.byteLength(value.stderr, "utf8") > maxBytes) throw new Error("launch_agent_rpc_stderr_is_invalid");
  if (typeof value.stdoutTruncated !== "boolean" || typeof value.stderrTruncated !== "boolean") throw new Error("launch_agent_rpc_output_flags_are_invalid");
  return {
    stdout: value.stdout,
    stderr: value.stderr,
    stdoutTruncated: value.stdoutTruncated,
    stderrTruncated: value.stderrTruncated,
  };
}

function boundedHandleOutput(
  handle: LaunchProcessHandle | null,
  maxBytes: number,
): LaunchCapturedOutput {
  let output: LaunchCapturedOutput | undefined;
  try {
    output = handle?.output?.();
  } catch {
    output = undefined;
  }
  const stdout = boundText(output?.stdout ?? "", maxBytes);
  const stderr = boundText(output?.stderr ?? "", maxBytes);
  return {
    stdout: stdout.text,
    stderr: stderr.text,
    stdoutTruncated: Boolean(output?.stdoutTruncated) || stdout.truncated,
    stderrTruncated: Boolean(output?.stderrTruncated) || stderr.truncated,
  };
}

function boundText(value: string, maxBytes: number): { text: string; truncated: boolean } {
  if (typeof value !== "string") return { text: "", truncated: true };
  const candidate = value.slice(0, maxBytes);
  const bytes = Buffer.from(candidate, "utf8");
  if (value.length === candidate.length && bytes.length <= maxBytes) {
    return { text: value, truncated: false };
  }
  let text = bytes.subarray(0, maxBytes).toString("utf8");
  while (Buffer.byteLength(text, "utf8") > maxBytes) text = text.slice(0, -1);
  return { text, truncated: true };
}

function sanitizeExit(value: unknown): LaunchProcessExit {
  assertRecord(value, "launch_process_exit");
  assertExactKeys(value, ["code", "signal"], ["error"], "launch_process_exit");
  if (value.code !== null && (!Number.isInteger(value.code) || Math.abs(value.code as number) > 2 ** 31)) {
    throw new Error("launch_process_exit_code_is_invalid");
  }
  if (value.signal !== null && (typeof value.signal !== "string" || !/^SIG[A-Z0-9]+$/.test(value.signal))) {
    throw new Error("launch_process_exit_signal_is_invalid");
  }
  if (value.error !== undefined && typeof value.error !== "string") throw new Error("launch_process_exit_error_is_invalid");
  return {
    code: value.code as number | null,
    signal: value.signal as NodeJS.Signals | null,
    ...(value.error === undefined ? {} : { error: boundedError(value.error) }),
  };
}

function validateProcessHandle(handle: LaunchProcessHandle, processId: string): void {
  if (
    !handle ||
    typeof handle.stop !== "function" ||
    !(handle.ready instanceof Promise) ||
    !(handle.exited instanceof Promise)
  ) {
    throw new Error(`launch_agent_returned_invalid_handle:${processId}`);
  }
}

async function bestEffortStopHandle(value: unknown, reason: string): Promise<void> {
  try {
    if (value === null || (typeof value !== "object" && typeof value !== "function")) return;
    const stop = (value as { stop?: unknown }).stop;
    if (typeof stop !== "function") return;
    await Promise.resolve(stop.call(value, reason));
  } catch {
    // The allocation is already in an error path and cleanup cannot replace
    // the original lifecycle failure.
  }
}

async function operationWithTimeout<T>(
  operation: Promise<T>,
  signal: AbortSignal | undefined,
  timeoutMs: number,
  name: string,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const finish = (callback: () => void) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
      callback();
    };
    const abort = () => finish(() => reject(cancellationFromSignal(signal!)));
    const timer = setTimeout(
      () => finish(() => reject(new LaunchAgentRpcTimeoutError(name, timeoutMs))),
      timeoutMs,
    );
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) {
      abort();
      return;
    }
    operation.then(
      (value) => finish(() => resolve(value)),
      (error) => finish(() => reject(normalizeError(error))),
    );
  });
}

async function readJsonBody(request: IncomingMessage, maxBytes: number): Promise<unknown> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of request) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    total += buffer.length;
    if (total > maxBytes) throw new RpcRouteError(413, "request_body_too_large");
    chunks.push(buffer);
  }
  if (total === 0) throw new RpcRouteError(400, "request_body_is_empty");
  try {
    return JSON.parse(Buffer.concat(chunks, total).toString("utf8"));
  } catch {
    throw new RpcRouteError(400, "request_body_is_not_json");
  }
}

async function readResponseText(response: Response, maxBytes: number): Promise<string> {
  if (!response.body) return "";
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  while (true) {
    const next = await reader.read();
    if (next.done) break;
    total += next.value.byteLength;
    if (total > maxBytes) {
      await reader.cancel();
      throw new Error("launch_agent_rpc_response_is_too_large");
    }
    chunks.push(next.value);
  }
  return Buffer.concat(chunks.map((chunk) => Buffer.from(chunk)), total).toString("utf8");
}

function requireJson(request: IncomingMessage): void {
  const type = request.headers["content-type"]?.split(";", 1)[0]?.trim().toLowerCase();
  if (type !== "application/json") throw new RpcRouteError(415, "content_type_must_be_json");
}

function sendJson(response: ServerResponse, status: number, value: unknown): void {
  if (response.headersSent || response.destroyed) return;
  const body = Buffer.from(JSON.stringify(value), "utf8");
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": body.length,
    "cache-control": "no-store",
  });
  response.end(body);
}

class RpcRouteError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

function normalizeRouteError(value: unknown): RpcRouteError {
  if (value instanceof RpcRouteError) return value;
  if (value instanceof LaunchAgentRpcTimeoutError) return new RpcRouteError(504, value.message);
  return new RpcRouteError(400, normalizeError(value).message);
}

function normalizeEndpoint(value: string | URL): URL {
  let endpoint: URL;
  try {
    endpoint = value instanceof URL ? new URL(value.href) : new URL(value);
  } catch {
    throw new Error("launch_agent_rpc_endpoint_is_invalid");
  }
  if (
    (endpoint.protocol !== "http:" && endpoint.protocol !== "https:") ||
    endpoint.username ||
    endpoint.password ||
    endpoint.search ||
    endpoint.hash
  ) {
    throw new Error("launch_agent_rpc_endpoint_is_invalid");
  }
  endpoint.pathname = endpoint.pathname.replace(/\/$/, "");
  return endpoint;
}

function defaultHttpAgentId(endpoint: URL): string {
  // A URL host can contain characters (notably IPv6 brackets) that are not in
  // the LaunchAgent identifier alphabet. A stable endpoint digest remains
  // readable as a type while supporting every valid HTTP(S) host form.
  return `http-launch-agent:${createHash("sha256")
    .update(endpoint.href)
    .digest("hex")
    .slice(0, 24)}`;
}

function normalizeOptionalAuthToken(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  if (
    typeof value !== "string" ||
    value.length < 32 ||
    value.length > 8_192 ||
    !/^[\x21-\x7e]+$/.test(value)
  ) {
    // Never include credential material in diagnostics.
    throw new Error("launch_agent_rpc_auth_token_is_invalid");
  }
  return value;
}

function launchAgentAuthTokenDigest(value: string): Buffer {
  return createHash("sha256")
    .update("gdlp-launch-agent-auth/1\0")
    .update(value, "utf8")
    .digest();
}

function isLoopbackBindHost(host: string): boolean {
  const normalized = host.toLowerCase();
  return (
    normalized === "::1" ||
    normalized.startsWith("127.") ||
    normalized.startsWith("::ffff:127.")
  );
}

function normalizeAllowedStartFingerprints(
  value: readonly unknown[] | undefined,
  nodeId: string | undefined,
): ReadonlyMap<string, string> | undefined {
  if (value === undefined) return undefined;
  if (!Array.isArray(value) || value.length < 1 || value.length > 1_024) {
    throw new Error("launch_agent_rpc_launch_allowlist_is_invalid");
  }
  if (nodeId === undefined) {
    throw new Error("launch_agent_rpc_launch_allowlist_requires_node_id");
  }

  const fingerprints = new Map<string, string>();
  for (const candidate of value) {
    validatePythonLaunchDescription(candidate);
    const description: PythonPipelineLaunchDescription = candidate;
    for (const process of description.launchOrder) {
      if (process.anchor.memberId !== nodeId) continue;
      const expected: LaunchAgentStartRequest = {
        launchId: description.launchId,
        pipelineId: description.pipelineId,
        nodeId,
        process: structuredClone(process),
      };
      const key = allowedStartKey(expected);
      const fingerprint = stableJson(expected);
      const previous = fingerprints.get(key);
      if (previous !== undefined && previous !== fingerprint) {
        throw new Error("launch_agent_rpc_launch_allowlist_conflicts");
      }
      fingerprints.set(key, fingerprint);
    }
  }
  if (fingerprints.size === 0) {
    throw new Error("launch_agent_rpc_launch_allowlist_has_no_process_for_node");
  }
  return fingerprints;
}

function allowedStartKey(request: LaunchAgentStartRequest): string {
  return `${request.launchId}\0${request.process.processId}`;
}

function stableJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  return `{${Object.keys(value as Record<string, unknown>)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${stableJson((value as Record<string, unknown>)[key])}`)
    .join(",")}}`;
}

function assertRecord(value: unknown, name: string): asserts value is Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${name}_must_be_an_object`);
  }
}

function assertExactKeys(
  value: Record<string, unknown>,
  required: readonly string[],
  optional: readonly string[],
  name: string,
): void {
  const actual = Object.keys(value).sort();
  const allowed = new Set([...required, ...optional]);
  if (required.some((key) => !Object.hasOwn(value, key)) || actual.some((key) => !allowed.has(key))) {
    throw new Error(`${name}_keys_are_invalid`);
  }
}

function assertIdentifier(value: unknown, name: string): asserts value is string {
  if (typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,255}$/.test(value)) {
    throw new Error(`${name}_is_invalid`);
  }
}

function assertUint64String(value: unknown, name: string): asserts value is string {
  if (typeof value !== "string" || !/^(?:0|[1-9]\d*)$/.test(value)) {
    throw new Error(`${name}_is_invalid`);
  }
  if (BigInt(value) > 18_446_744_073_709_551_615n) {
    throw new Error(`${name}_is_invalid`);
  }
}

function assertPath(value: unknown, name: string, maxLength = 32_768): asserts value is string {
  if (
    typeof value !== "string" ||
    value.length < 1 ||
    value.length > maxLength ||
    /[\0\r\n]/.test(value)
  ) {
    throw new Error(`${name}_is_invalid`);
  }
}

function assertArgument(value: unknown): asserts value is string {
  if (typeof value !== "string" || value.length > 32_768 || value.includes("\0")) {
    throw new Error("command_argument_is_invalid");
  }
}

function assertHandleId(value: unknown): asserts value is string {
  if (typeof value !== "string" || !/^[a-f0-9]{48}$/.test(value)) {
    throw new Error("launch_agent_rpc_handle_id_is_invalid");
  }
}

function assertReason(value: unknown): asserts value is string {
  if (typeof value !== "string" || value.length < 1 || value.length > 1_024 || /[\0\r\n]/.test(value)) {
    throw new Error("launch_agent_stop_reason_is_invalid");
  }
}

function assertInteger(value: unknown, min: number, max: number, name: string): void {
  if (!Number.isSafeInteger(value) || (value as number) < min || (value as number) > max) {
    throw new Error(`${name}_is_invalid`);
  }
}

function assertPositiveFinite(value: unknown, name: string): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${name}_is_invalid`);
  }
}

function assertFiniteRange(value: unknown, min: number, max: number, name: string): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) {
    throw new Error(`${name}_is_invalid`);
  }
}

function assertStringArray(
  value: unknown,
  name: string,
  minLength: number,
  maxLength: number,
): asserts value is string[] {
  if (!Array.isArray(value) || value.length < minLength || value.length > maxLength) {
    throw new Error(`${name}_is_invalid`);
  }
  for (const item of value) assertIdentifier(item, name);
}

function assertBoundedStringArray(
  value: unknown,
  name: string,
  minLength: number,
  maxLength: number,
  maxItemLength = 1_024,
): asserts value is string[] {
  if (!Array.isArray(value) || value.length < minLength || value.length > maxLength) {
    throw new Error(`${name}_is_invalid`);
  }
  for (const item of value) assertPath(item, name, maxItemLength);
}

function assertIntegerArray(value: unknown, name: string, minLength: number): void {
  if (!Array.isArray(value) || value.length < minLength || value.length > 1_000_000) {
    throw new Error(`${name}_is_invalid`);
  }
  for (const item of value) assertInteger(item, 0, Number.MAX_SAFE_INTEGER, name);
}

function assertExactStringTuple(value: unknown, expected: string[], name: string): void {
  if (!Array.isArray(value) || value.length !== expected.length || value.some((item, index) => item !== expected[index])) {
    throw new Error(`${name}_is_invalid`);
  }
}

function assertSha256(value: unknown, name: string): asserts value is string {
  if (typeof value !== "string" || !/^[a-f0-9]{64}$/.test(value)) {
    throw new Error(`${name}_is_invalid`);
  }
}

function boundedInteger(value: unknown, min: number, max: number, error: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < min || (value as number) > max) {
    throw new Error(error);
  }
  return value as number;
}

function boundedError(value: string): string {
  return value.slice(0, 4_096);
}

function formatUrlHost(host: string): string {
  return host.includes(":") && !host.startsWith("[") ? `[${host}]` : host;
}

function rpcErrorMessage(value: unknown): string {
  if (value && typeof value === "object" && "error" in value && typeof value.error === "string") {
    return boundedError(value.error);
  }
  return "request_failed";
}

function linkAbortSignal(
  source: AbortSignal | undefined,
  target: AbortController,
): () => void {
  if (!source) return () => undefined;
  const abort = () => target.abort(source.reason);
  source.addEventListener("abort", abort, { once: true });
  if (source.aborted) abort();
  return () => source.removeEventListener("abort", abort);
}

function cancellationFromSignal(signal: AbortSignal): Error {
  return signal.reason instanceof Error
    ? signal.reason
    : new LaunchCancelledError("launch_agent_rpc_cancelled");
}

function normalizeError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
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
