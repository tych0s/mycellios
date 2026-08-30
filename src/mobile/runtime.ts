import {
  isMobileExecutionCancelled,
  MobileExecutionCancelledError,
  mobileCpuFallbackLabel,
  mobileGpuAdapterLabel,
  requestMobileGpuAdapter,
  runWithMobileBackendFallback,
  throwIfMobileExecutionCancelled,
} from "./backend-state";
import type { NativeBuildIdentity } from "../contracts/build-identity";
import { trustedExpertManifestPath } from "./expert-manifest-url";

declare const __MYCELLIOS_BUILD_IDENTITY__: NativeBuildIdentity;

export type MobileWorkerBackend = "webgpu" | "cpu";
export type MobileWorkerLevel = "low" | "balanced" | "maximum";

type Backend = MobileWorkerBackend;
type Level = MobileWorkerLevel;

interface ComputeOffer {
  taskId: string;
  leaseId: string;
  operation: "matrix-multiply";
  size: number;
  seed: number;
  deadlineAt: number;
}

interface ComputeResult {
  samples: number[];
  durationMs: number;
  estimatedGflops: number;
}

interface ExpertManifest {
  artifactId: string;
  modelId: string;
  modelDigest: string;
  layer: number;
  expert: number;
  contentId: string;
  weightsHash: string;
  hiddenSize: number;
  intermediateSize: number;
  dtype: "float32";
  activation: "silu";
  canaryInputBase64: string;
  canaryOutputBase64: string;
}

interface ResidentExpert {
  manifest: ExpertManifest;
  gate: Float32Array;
  up: Float32Array;
  down: Float32Array;
}

interface RuntimeState {
  running: boolean;
  starting: boolean;
  backend: Backend;
  level: Level;
  socket: WebSocket | null;
  wakeLock: WakeLockSentinel | null;
  heartbeatTimer: number | null;
  reconnectTimer: number | null;
  workerId: string | null;
  token: string | null;
  estimatedGflops: number;
  verifiedTasks: number;
  gpu: GPUAdapter | null;
  device: GPUDevice | null;
  detectedGpuLabel: string | null;
  residentExperts: Map<string, ResidentExpert>;
  cancelledMatrixTasks: Set<string>;
  executionController: AbortController;
  resumePromise: Promise<void> | null;
}

export type MobileWorkerConnection = "online" | "offline" | "connecting";

export interface MobileWorkerSnapshot {
  running: boolean;
  starting: boolean;
  backend: MobileWorkerBackend;
  level: MobileWorkerLevel;
  connection: MobileWorkerConnection;
  connectionLabel: string;
  statusText: string;
  gpuLabel: string;
  estimatedGflops: number;
  verifiedTasks: number;
  wakeLockActive: boolean;
  activity: string[];
}

const state: RuntimeState = {
  running: false,
  starting: false,
  backend: "cpu",
  level: "balanced",
  socket: null,
  wakeLock: null,
  heartbeatTimer: null,
  reconnectTimer: null,
  workerId: null,
  token: null,
  estimatedGflops: 0,
  verifiedTasks: 0,
  gpu: null,
  device: null,
  detectedGpuLabel: null,
  residentExperts: new Map(),
  cancelledMatrixTasks: new Set(),
  executionController: new AbortController(),
  resumePromise: null,
};

const viewState: Pick<MobileWorkerSnapshot, "connection" | "connectionLabel" | "statusText" | "gpuLabel" | "activity"> = {
  connection: "offline",
  connectionLabel: "Disconnected",
  statusText: "Press the button and keep this screen open.",
  gpuLabel: "Detecting available hardware…",
  activity: ["Worker ready to start."],
};
const listeners = new Set<(snapshot: MobileWorkerSnapshot) => void>();
let initialized = false;
let backendProbeVersion = 0;

export function initializeMobileWorker(): void {
  if (initialized) return;
  initialized = true;
  document.addEventListener("visibilitychange", handleVisibilityChange);
  void detectBackendPreview();
}

export function subscribeMobileWorker(listener: (snapshot: MobileWorkerSnapshot) => void): () => void {
  listeners.add(listener);
  listener(currentSnapshot());
  return () => listeners.delete(listener);
}

export function getMobileWorkerSnapshot(): MobileWorkerSnapshot {
  return currentSnapshot();
}

export async function toggleMobileWorker(): Promise<void> {
  return state.running ? stop() : start();
}

export function setMobileWorkerLevel(level: MobileWorkerLevel): void {
  if (state.running || state.starting) return;
  state.level = level;
  emit();
}

function handleVisibilityChange(): void {
  if (!state.running && !state.starting) return;
  if (document.visibilityState === "visible") {
    if (state.running) void resumeVisibleWorker();
  } else {
    sendHeartbeat();
    cancelActiveExecutions("mycellios is no longer visible");
    if (state.heartbeatTimer !== null) window.clearInterval(state.heartbeatTimer);
    if (state.reconnectTimer !== null) window.clearTimeout(state.reconnectTimer);
    state.heartbeatTimer = null;
    state.reconnectTimer = null;
    const socket = state.socket;
    state.socket = null;
    socket?.close(1000, "page hidden");
    void state.wakeLock?.release().catch(() => undefined);
    state.wakeLock = null;
    renderWakeLock();
    setConnection("offline", "Paused");
    setStatus("Paused because mycellios is not visible.");
    addLog("Compute cancelled because this page is hidden.");
  }
}

async function start(): Promise<void> {
  if (state.starting || state.running) return;
  backendProbeVersion += 1;
  const signal = beginExecutionSession("a new contribution session started");
  state.starting = true;
  const clientIdPromise = persistentClientId();
  const persistentStoragePromise = requestPersistentBrowserStorage();
  setBusy(true);
  setConnection("connecting", "Preparing");
  setStatus("Preparing this device. No work is active until the network connects.");
  addLog("Checking the device compute engine.");
  try {
    await initializeComputeBackend(signal);
    assertExecutionAllowed(signal);
    setStatus(`Measuring the ${state.backend === "webgpu" ? "GPU" : "CPU"} with a real task…`);
    addLog(`${state.backend === "webgpu" ? `Validating ${state.detectedGpuLabel ?? "the WebGPU adapter"}` : "Validating the CPU fallback"} with a real matrix task.`);
    const { size: benchmarkSize, result: benchmark } = await runStartupBenchmark(73, signal);
    assertExecutionAllowed(signal);
    state.estimatedGflops = benchmark.estimatedGflops;
    renderPerformance();
    addLog(`${state.backend === "webgpu" ? "GPU" : "CPU"} validation passed: ${formatGflops(state.estimatedGflops)} GFLOPS.`);
    await acquireWakeLock(signal);
    const clientId = await abortable(clientIdPromise, signal);
    const persistentStorage = await abortable(persistentStoragePromise, signal);
    assertExecutionAllowed(signal);
    addLog(persistentStorage
      ? "Browser identity protected from automatic storage eviction."
      : "Browser identity will be retained while site storage remains available.");
    const credentials = await registerWorker(benchmarkSize, benchmark.durationMs, clientId, signal);
    assertExecutionAllowed(signal);
    state.workerId = credentials.workerId;
    state.token = credentials.token;
    state.running = true;
    setBusy(false);
    connect();
  } catch (error) {
    if (isMobileExecutionCancelled(error)) return;
    addLog(`Could not start: ${errorText(error)}`);
    await stop(false, true);
    setStatus(errorText(error));
    setConnection("offline", "Unavailable");
  } finally {
    state.starting = false;
    setBusy(false);
  }
}

async function stop(log = true, preserveView = false): Promise<void> {
  state.running = false;
  state.starting = false;
  cancelActiveExecutions("contribution stopped");
  if (state.heartbeatTimer !== null) window.clearInterval(state.heartbeatTimer);
  if (state.reconnectTimer !== null) window.clearTimeout(state.reconnectTimer);
  state.heartbeatTimer = null;
  state.reconnectTimer = null;
  state.socket?.close(1000, "user stopped contribution");
  state.socket = null;
  await state.wakeLock?.release().catch(() => undefined);
  state.wakeLock = null;
  state.residentExperts.clear();
  state.cancelledMatrixTasks.clear();
  renderWakeLock();
  setBusy(false);
  if (!preserveView) {
    setConnection("offline", "Disconnected");
    setStatus("Press the button and keep this screen open.");
  }
  if (log) addLog("Contribution stopped by the user.");
}

async function detectBackendPreview(): Promise<void> {
  const probeVersion = backendProbeVersion;
  if (!window.isSecureContext) {
    state.backend = "cpu";
    state.detectedGpuLabel = null;
    viewState.gpuLabel = "WebGPU requires HTTPS";
    emit();
    return;
  }
  if (!navigator.gpu) {
    state.backend = "cpu";
    state.detectedGpuLabel = null;
    viewState.gpuLabel = "Universal CPU fallback";
    emit();
    return;
  }
  viewState.gpuLabel = "Checking for a WebGPU adapter…";
  emit();
  const adapter = await requestMobileGpuAdapter(navigator.gpu);
  if (probeVersion !== backendProbeVersion || state.starting || state.running) return;
  if (!adapter) {
    state.backend = "cpu";
    state.detectedGpuLabel = null;
    viewState.gpuLabel = "No WebGPU adapter found · CPU available";
    emit();
    return;
  }
  const label = mobileGpuAdapterLabel(adapter.info);
  state.backend = "webgpu";
  state.detectedGpuLabel = label;
  viewState.gpuLabel = label;
  emit();
}

async function initializeComputeBackend(signal: AbortSignal): Promise<void> {
  assertExecutionAllowed(signal);
  if (!window.isSecureContext) {
    state.backend = "cpu";
    state.detectedGpuLabel = null;
    viewState.gpuLabel = "WebGPU requires HTTPS";
    emit();
    addLog("WebGPU is blocked outside HTTPS; the universal CPU engine is ready.");
    return;
  }
  if (!navigator.gpu) {
    state.backend = "cpu";
    state.detectedGpuLabel = null;
    viewState.gpuLabel = "WebGPU unavailable";
    emit();
    addLog("This browser does not expose WebGPU; the universal CPU engine is ready.");
    return;
  }
  try {
    addLog("Requesting a compatible GPU adapter from the browser.");
    const adapter = await abortable(requestMobileGpuAdapter(navigator.gpu), signal);
    if (!adapter) throw new Error("No WebGPU adapter found");
    const label = mobileGpuAdapterLabel(adapter.info);
    state.detectedGpuLabel = label;
    addLog(`GPU adapter detected: ${label}. Requesting a compute device.`);
    const device = await abortable(adapter.requestDevice(), signal);
    try {
      assertExecutionAllowed(signal);
    } catch (error) {
      device.destroy();
      throw error;
    }
    device.lost.then((info) => {
      if (state.device !== device) return;
      transitionToCpuFallback(
        `device lost: ${info.message || info.reason}`,
        device,
      );
      setStatus(state.running
        ? "The GPU became unavailable; contribution continues on CPU."
        : "The GPU became unavailable; setup continues on CPU.");
      if (state.running) requestWork();
    });
    state.backend = "webgpu";
    state.gpu = adapter;
    state.device = device;
    viewState.gpuLabel = label;
    emit();
    addLog(`WebGPU device ready: ${label}. No vendor runtime installation is required in this browser.`);
  } catch (error) {
    throwIfMobileExecutionCancelled(signal);
    state.backend = "cpu";
    state.gpu = null;
    state.device = null;
    viewState.gpuLabel = mobileCpuFallbackLabel(state.detectedGpuLabel);
    emit();
    addLog(`WebGPU unavailable; using CPU: ${errorText(error)}`);
  }
}

async function registerWorker(
  matrixSize: number,
  durationMs: number,
  clientId: string,
  signal: AbortSignal,
): Promise<{ workerId: string; token: string }> {
  const query = new URLSearchParams(window.location.search);
  const invitationFromUrl = query.get("join")?.trim();
  const joinToken = persistentJoinToken(invitationFromUrl);
  const protocol = { min: 1, max: 1 };
  const body = {
    clientId,
    name: mobileName(),
    region: query.get("region")?.trim() || "auto",
    platform: navigator.userAgent.slice(0, 120),
    buildIdentity: __MYCELLIOS_BUILD_IDENTITY__,
    backend: state.backend,
    performanceLevel: state.level,
    ...(joinToken ? { joinToken } : {}),
    capabilities: {
      webgpu: Boolean(state.device),
      wasm: typeof WebAssembly === "object",
      hardwareConcurrency: Math.max(1, navigator.hardwareConcurrency || 1),
      ...(deviceMemory() ? { deviceMemoryGb: deviceMemory() } : {}),
      ...(state.detectedGpuLabel ? { gpuDescription: state.detectedGpuLabel } : {}),
      ...(state.device ? { maxBufferSize: Number(state.device.limits.maxBufferSize) } : {}),
    },
    benchmark: {
      durationMs,
      estimatedGflops: state.estimatedGflops,
      matrixSize,
    },
    protocol,
  };
  const identity = { kind: "browser" as const, id: clientId };
  const admissionCredential = await persistentBrowserAdmissionCredential();
  const registrationDigest = await sha256CanonicalBrowserEvidence({
    schema: "mycellios-mobile-registration/1",
    identity,
    registration: body,
  });
  const challengeResponse = await abortable(fetch("/mobile/v1/admission-challenge", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      identity,
      publicKey: {
        algorithm: "ecdsa-p256-sha256",
        spki: admissionCredential.publicKeySpki,
      },
      protocol,
      registrationDigest,
      ...(joinToken ? { joinToken } : {}),
    }),
    signal,
  }), signal);
  if (!challengeResponse.ok) {
    throw new Error(
      `The coordinator rejected the device identity challenge (HTTP ${challengeResponse.status}).`,
    );
  }
  const challenge = await challengeResponse.json() as {
    challengeId?: unknown;
    signingPayload?: unknown;
  };
  if (
    typeof challenge.challengeId !== "string"
    || typeof challenge.signingPayload !== "string"
  ) {
    throw new Error("The coordinator returned an invalid device identity challenge.");
  }
  const signature = await signBrowserAdmissionPayload(
    admissionCredential.privateKey,
    challenge.signingPayload,
  );
  const response = await abortable(fetch("/mobile/v1/register", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      ...body,
      admission: {
        challengeId: challenge.challengeId,
        publicKey: {
          algorithm: "ecdsa-p256-sha256",
          spki: admissionCredential.publicKeySpki,
        },
        protocol,
        registrationDigest,
        signature,
      },
    }),
    signal,
  }), signal);
  if (!response.ok) {
    if (response.status === 401) throw new Error("The invitation link is invalid.");
    throw new Error(`The coordinator rejected registration (HTTP ${response.status}).`);
  }
  return (await response.json()) as { workerId: string; token: string };
}

function persistentJoinToken(invitationFromUrl: string | undefined): string | undefined {
  try {
    if (invitationFromUrl) localStorage.setItem("mycellios.joinToken", invitationFromUrl);
    return invitationFromUrl || localStorage.getItem("mycellios.joinToken")?.trim() || undefined;
  } catch {
    return invitationFromUrl;
  }
}

async function resumeVisibleWorker(): Promise<void> {
  if (state.resumePromise || !state.running || document.visibilityState !== "visible") return;
  const signal = state.executionController.signal;
  const resume = (async () => {
    setConnection("connecting", "Preparing");
    setStatus("Preparing the compute engine after returning to mycellios…");
    try {
      await initializeComputeBackend(signal);
      assertExecutionAllowed(signal);
      await acquireWakeLock(signal);
      assertExecutionAllowed(signal);
      connect();
    } catch (error) {
      if (isMobileExecutionCancelled(error)) return;
      state.running = false;
      setStatus(errorText(error));
      setConnection("offline", "Unavailable");
      addLog(`Could not resume: ${errorText(error)}`);
    }
  })();
  state.resumePromise = resume;
  try {
    await resume;
  } finally {
    if (state.resumePromise === resume) state.resumePromise = null;
  }
}

function connect(): void {
  if (!state.running || document.visibilityState !== "visible" || !state.workerId || !state.token) return;
  if (state.socket && state.socket.readyState <= WebSocket.OPEN) return;
  if (state.reconnectTimer !== null) window.clearTimeout(state.reconnectTimer);
  state.reconnectTimer = null;
  if (viewState.connectionLabel !== "Reconnecting") {
    setConnection("connecting", "Preparing");
    setStatus("Compute is ready; waiting for coordinator confirmation…");
  }
  const url = new URL("/mobile/v1/connect", window.location.origin);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.searchParams.set("workerId", state.workerId);
  url.searchParams.set("token", state.token);
  let socket: WebSocket;
  try {
    socket = new WebSocket(url);
  } catch (error) {
    state.running = false;
    cancelActiveExecutions("the coordinator channel could not be opened");
    setConnection("offline", "Unavailable");
    setStatus(errorText(error));
    addLog(`Could not open the coordinator channel: ${errorText(error)}`);
    return;
  }
  state.socket = socket;
  socket.addEventListener("open", () => {
    if (state.socket === socket) addLog("Secure coordinator channel opened; waiting for coordinator readiness.");
  });
  socket.addEventListener("message", (event) => {
    if (state.socket === socket) void handleServerMessage(String(event.data));
  });
  socket.addEventListener("close", () => {
    if (state.socket !== socket) return;
    state.socket = null;
    if (!state.running) return;
    if (document.visibilityState !== "visible") {
      setConnection("offline", "Paused");
      return;
    }
    setConnection("connecting", "Reconnecting");
    setStatus("Connection lost; retrying automatically…");
    state.reconnectTimer = window.setTimeout(connect, 2_000);
  });
  socket.addEventListener("error", () => {
    if (state.socket === socket) socket.close();
  });
}

async function handleServerMessage(raw: string): Promise<void> {
  let message: { type?: string; payload?: unknown };
  try {
    message = JSON.parse(raw) as { type?: string; payload?: unknown };
  } catch {
    return;
  }
  if (!state.running || document.visibilityState !== "visible") return;
  if (message.type !== "server.ready" && viewState.connection !== "online") return;
  if (message.type === "server.ready") {
    const payload = message.payload as { verifiedTasks?: number; inferenceReady?: boolean };
    state.verifiedTasks = payload.verifiedTasks ?? state.verifiedTasks;
    setConnection("online", "Connected");
    sendHeartbeat();
    if (state.heartbeatTimer !== null) window.clearInterval(state.heartbeatTimer);
    state.heartbeatTimer = window.setInterval(sendHeartbeat, 5_000);
    if (payload.inferenceReady || state.verifiedTasks > 0) {
      setStatus("Device validated and waiting for real inference work.");
      addLog("Device already validated · ready for real model inference.");
      emit();
    } else {
      setStatus("Connected. Running the one-time network admission check…");
      addLog("Browser worker connected · requesting its one-time admission check.");
      requestWork();
    }
    return;
  }
  if (message.type === "compute.offer") {
    const offer = message.payload as ComputeOffer;
    await executeOffer(offer);
    return;
  }
  if (message.type === "compute.cancel") {
    const payload = message.payload as { taskId?: string };
    if (payload.taskId) state.cancelledMatrixTasks.add(payload.taskId);
    return;
  }
  if (message.type === "expert.load") {
    await loadExpert(message.payload as {
      taskId: string; leaseId: string; artifactId: string; manifestUrl: string; deadlineAt: number;
    });
    return;
  }
  if (message.type === "expert.execute") {
    await executeExpert(message.payload as {
      taskId: string; leaseId: string; artifactId: string; rows: number;
      hiddenSize: number; activationsBase64: string; deadlineAt: number;
    });
    return;
  }
  if (message.type === "expert.verified") {
    const payload = message.payload as { phase?: string; replicas?: number; verifiedTasks?: number };
    if (payload.phase === "consensus") {
      state.verifiedTasks = payload.verifiedTasks ?? state.verifiedTasks + 1;
      emit();
      setStatus("Expert output matched an independent replica and was accepted by the model network.");
      addLog(`Replicated MoE consensus accepted · ${payload.replicas ?? 2} devices.`);
    }
    return;
  }
  if (message.type === "expert.rejected") {
    addLog("Expert weights or replicated output failed validation and were removed.");
  }
  if (message.type === "compute.verified") {
    const payload = message.payload as { verifiedTasks?: number; inferenceReady?: boolean };
    state.verifiedTasks = payload.verifiedTasks ?? state.verifiedTasks + 1;
    setStatus("Device validated and waiting for real inference work.");
    emit();
    addLog("Admission check verified · this device is ready for real model inference.");
    return;
  }
  if (message.type === "compute.rejected") {
    addLog("The result failed verification and was discarded.");
    window.setTimeout(requestWork, coolDownMs());
  }
}

async function executeOffer(offer: ComputeOffer): Promise<void> {
  if (!state.running || document.visibilityState !== "visible") return;
  if (offer.operation !== "matrix-multiply" || offer.deadlineAt <= Date.now()) return;
  const signal = state.executionController.signal;
  send("compute.accept", { taskId: offer.taskId, leaseId: offer.leaseId });
  setStatus(`Computing a ${offer.size}×${offer.size} matrix with ${state.backend.toUpperCase()}…`);
  addLog(`Job received: ${offer.size}×${offer.size} multiplication.`);
  try {
    const result = await runMatrixTask(offer.size, offer.seed, signal);
    assertExecutionAllowed(signal);
    if (state.cancelledMatrixTasks.delete(offer.taskId)) return;
    state.estimatedGflops = result.estimatedGflops;
    renderPerformance();
    send("compute.result", {
      taskId: offer.taskId,
      leaseId: offer.leaseId,
      backend: state.backend,
      ...result,
    });
    setStatus("Result sent; waiting for coordinator verification…");
  } catch (error) {
    if (isMobileExecutionCancelled(error)) return;
    send("compute.fail", {
      taskId: offer.taskId,
      leaseId: offer.leaseId,
      message: errorText(error).slice(0, 300),
    });
    addLog(`Task failed: ${errorText(error)}`);
    window.setTimeout(requestWork, coolDownMs());
  }
}

async function loadExpert(offer: {
  taskId: string;
  leaseId: string;
  artifactId: string;
  manifestUrl: string;
  deadlineAt: number;
}): Promise<void> {
  if (!state.running || document.visibilityState !== "visible" || offer.deadlineAt <= Date.now()) return;
  const signal = state.executionController.signal;
  try {
    setStatus("Downloading and verifying a real model expert…");
    const manifestPath = trustedExpertManifestPath(offer.artifactId, offer.manifestUrl);
    const manifestResponse = await abortable(fetch(
      manifestPath,
      { signal },
    ), signal);
    if (!manifestResponse.ok) throw new Error(`expert manifest HTTP ${manifestResponse.status}`);
    const manifest = (await abortable(manifestResponse.json(), signal)) as ExpertManifest;
    if (manifest.artifactId !== offer.artifactId || manifest.dtype !== "float32"
      || manifest.activation !== "silu") throw new Error("unsupported expert manifest");
    const weightsResponse = await abortable(fetch(
      new URL(`/mobile/v1/experts/weights/${manifest.weightsHash}`, window.location.origin),
      { signal },
    ), signal);
    if (!weightsResponse.ok) throw new Error(`expert weights HTTP ${weightsResponse.status}`);
    const bytes = await abortable(weightsResponse.arrayBuffer(), signal);
    const digest = hex(await abortable(crypto.subtle.digest("SHA-256", bytes), signal));
    if (digest !== manifest.weightsHash) throw new Error("expert weight hash mismatch");
    const expectedValues = 3 * manifest.hiddenSize * manifest.intermediateSize;
    if (bytes.byteLength !== expectedValues * Float32Array.BYTES_PER_ELEMENT) {
      throw new Error("expert weight shape mismatch");
    }
    const allWeights = new Float32Array(bytes);
    const projectionValues = manifest.hiddenSize * manifest.intermediateSize;
    const expert: ResidentExpert = {
      manifest,
      gate: allWeights.slice(0, projectionValues),
      up: allWeights.slice(projectionValues, 2 * projectionValues),
      down: allWeights.slice(2 * projectionValues),
    };
    const canaryInput = float32FromBase64(manifest.canaryInputBase64);
    if (canaryInput.length !== manifest.hiddenSize) throw new Error("invalid expert canary input");
    const started = performance.now();
    const canaryDevice = state.device;
    let computed = await computeSwiGlu(expert, canaryInput, 1, signal);
    const expectedCanary = float32FromBase64(manifest.canaryOutputBase64);
    if (!float32Close(computed.values, expectedCanary, 2e-4) && computed.backend === "webgpu") {
      assertExecutionAllowed(signal);
      transitionToCpuFallback("expert canary returned an invalid result", canaryDevice);
      computed = { values: await swiGluCpu(expert, canaryInput, 1, signal), backend: "cpu" };
    }
    if (!float32Close(computed.values, expectedCanary, 2e-4)) {
      throw new Error("expert canary verification failed");
    }
    state.residentExperts.set(offer.artifactId, expert);
    send("expert.ready", {
      taskId: offer.taskId,
      leaseId: offer.leaseId,
      artifactId: offer.artifactId,
      canaryOutputBase64: float32ToBase64(computed.values),
      backend: computed.backend,
      durationMs: Math.max(0, performance.now() - started),
    });
    setStatus(`Resident MoE expert ready · layer ${manifest.layer}, expert ${manifest.expert}.`);
    addLog(`Loaded ${manifest.modelId} L${manifest.layer}/E${manifest.expert}; SHA-256 verified.`);
  } catch (error) {
    state.residentExperts.delete(offer.artifactId);
    if (isMobileExecutionCancelled(error)) return;
    send("expert.fail", {
      taskId: offer.taskId,
      leaseId: offer.leaseId,
      artifactId: offer.artifactId,
      message: errorText(error).slice(0, 500),
    });
    addLog(`Expert load failed: ${errorText(error)}`);
  }
}

async function executeExpert(offer: {
  taskId: string;
  leaseId: string;
  artifactId: string;
  rows: number;
  hiddenSize: number;
  activationsBase64: string;
  deadlineAt: number;
}): Promise<void> {
  if (!state.running || document.visibilityState !== "visible" || offer.deadlineAt <= Date.now()) return;
  const signal = state.executionController.signal;
  const expert = state.residentExperts.get(offer.artifactId);
  try {
    if (!expert) throw new Error("expert is not resident");
    if (expert.manifest.hiddenSize !== offer.hiddenSize) throw new Error("activation shape mismatch");
    const activations = float32FromBase64(offer.activationsBase64);
    if (activations.length !== offer.rows * offer.hiddenSize) throw new Error("activation payload mismatch");
    setStatus(`Running real model expert L${expert.manifest.layer}/E${expert.manifest.expert}…`);
    const started = performance.now();
    const computed = await computeSwiGlu(expert, activations, offer.rows, signal);
    assertExecutionAllowed(signal);
    send("expert.result", {
      taskId: offer.taskId,
      leaseId: offer.leaseId,
      artifactId: offer.artifactId,
      rows: offer.rows,
      hiddenSize: offer.hiddenSize,
      outputBase64: float32ToBase64(computed.values),
      backend: computed.backend,
      durationMs: Math.max(0, performance.now() - started),
    });
    setStatus("Expert activation returned; awaiting network verification…");
  } catch (error) {
    if (isMobileExecutionCancelled(error)) return;
    state.residentExperts.delete(offer.artifactId);
    send("expert.fail", {
      taskId: offer.taskId,
      leaseId: offer.leaseId,
      artifactId: offer.artifactId,
      message: errorText(error).slice(0, 500),
    });
  }
}

async function computeSwiGlu(
  expert: ResidentExpert,
  activations: Float32Array,
  rows: number,
  signal: AbortSignal,
): Promise<{ values: Float32Array; backend: Backend }> {
  const device = state.device;
  const execution = await runWithMobileBackendFallback({
    backend: state.backend === "webgpu" && device ? "webgpu" : "cpu",
    runGpu: async () => {
      if (!device) throw new Error("WebGPU device is unavailable");
      return await swiGluWebGpu(device, expert, activations, rows, signal);
    },
    runCpu: () => swiGluCpu(expert, activations, rows, signal),
    onGpuFailure: (error) => transitionToCpuFallback(
      `expert kernel failed: ${errorText(error)}`,
      device,
    ),
    signal,
  });
  return { values: execution.value, backend: execution.backend };
}

async function swiGluCpu(
  expert: ResidentExpert,
  activations: Float32Array,
  rows: number,
  signal: AbortSignal,
): Promise<Float32Array> {
  assertExecutionAllowed(signal);
  const { hiddenSize, intermediateSize } = expert.manifest;
  const activated = new Float32Array(rows * intermediateSize);
  let operationsUntilYield = 16_384;
  for (let row = 0; row < rows; row += 1) {
    for (let intermediate = 0; intermediate < intermediateSize; intermediate += 1) {
      let gate = 0;
      let up = 0;
      const weightOffset = intermediate * hiddenSize;
      const inputOffset = row * hiddenSize;
      for (let hidden = 0; hidden < hiddenSize; hidden += 1) {
        const input = activations[inputOffset + hidden] ?? 0;
        gate += input * (expert.gate[weightOffset + hidden] ?? 0);
        up += input * (expert.up[weightOffset + hidden] ?? 0);
        operationsUntilYield -= 1;
        if (operationsUntilYield === 0) {
          operationsUntilYield = 16_384;
          await yieldToBrowser(signal);
        }
      }
      activated[row * intermediateSize + intermediate] = (gate / (1 + Math.exp(-gate))) * up;
    }
  }
  const output = new Float32Array(rows * hiddenSize);
  for (let row = 0; row < rows; row += 1) {
    for (let hidden = 0; hidden < hiddenSize; hidden += 1) {
      let sum = 0;
      const downOffset = hidden * intermediateSize;
      for (let intermediate = 0; intermediate < intermediateSize; intermediate += 1) {
        sum += (activated[row * intermediateSize + intermediate] ?? 0)
          * (expert.down[downOffset + intermediate] ?? 0);
        operationsUntilYield -= 1;
        if (operationsUntilYield === 0) {
          operationsUntilYield = 16_384;
          await yieldToBrowser(signal);
        }
      }
      output[row * hiddenSize + hidden] = sum;
    }
  }
  return output;
}

async function swiGluWebGpu(
  device: GPUDevice,
  expert: ResidentExpert,
  activations: Float32Array,
  rows: number,
  signal: AbortSignal,
): Promise<Float32Array> {
  assertExecutionAllowed(signal);
  const { hiddenSize, intermediateSize } = expert.manifest;
  const byteSizes = [activations.byteLength, expert.gate.byteLength, expert.up.byteLength,
    expert.down.byteLength, rows * intermediateSize * 4, rows * hiddenSize * 4];
  if (byteSizes.some((size) => size > Number(device.limits.maxStorageBufferBindingSize))) {
    throw new Error("expert tensor exceeds this GPU's storage binding limit");
  }
  const buffers: GPUBuffer[] = [];
  const storage = (label: string, values: Float32Array): GPUBuffer => {
    const buffer = device.createBuffer({
      label,
      size: Math.max(4, values.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, values);
    buffers.push(buffer);
    return buffer;
  };
  const input = storage("expert-input", activations);
  const gate = storage("expert-gate", expert.gate);
  const up = storage("expert-up", expert.up);
  const down = storage("expert-down", expert.down);
  const activated = device.createBuffer({
    label: "expert-activated",
    size: rows * intermediateSize * 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const output = device.createBuffer({
    label: "expert-output",
    size: rows * hiddenSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: "expert-readback",
    size: rows * hiddenSize * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  buffers.push(activated, output, readback);
  try {
    const firstModule = device.createShaderModule({ code: `
      @group(0) @binding(0) var<storage, read> x: array<f32>;
      @group(0) @binding(1) var<storage, read> gate: array<f32>;
      @group(0) @binding(2) var<storage, read> up: array<f32>;
      @group(0) @binding(3) var<storage, read_write> activated: array<f32>;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let index = id.x;
        if (index >= ${rows * intermediateSize}u) { return; }
        let row = index / ${intermediateSize}u;
        let neuron = index % ${intermediateSize}u;
        var gateValue = 0.0;
        var upValue = 0.0;
        for (var h = 0u; h < ${hiddenSize}u; h = h + 1u) {
          let value = x[row * ${hiddenSize}u + h];
          gateValue = gateValue + value * gate[neuron * ${hiddenSize}u + h];
          upValue = upValue + value * up[neuron * ${hiddenSize}u + h];
        }
        activated[index] = (gateValue / (1.0 + exp(-gateValue))) * upValue;
      }
    ` });
    const secondModule = device.createShaderModule({ code: `
      @group(0) @binding(0) var<storage, read> activated: array<f32>;
      @group(0) @binding(1) var<storage, read> down: array<f32>;
      @group(0) @binding(2) var<storage, read_write> output: array<f32>;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let index = id.x;
        if (index >= ${rows * hiddenSize}u) { return; }
        let row = index / ${hiddenSize}u;
        let hidden = index % ${hiddenSize}u;
        var sum = 0.0;
        for (var i = 0u; i < ${intermediateSize}u; i = i + 1u) {
          sum = sum + activated[row * ${intermediateSize}u + i] * down[hidden * ${intermediateSize}u + i];
        }
        output[index] = sum;
      }
    ` });
    const first = device.createComputePipeline({ layout: "auto", compute: { module: firstModule } });
    const second = device.createComputePipeline({ layout: "auto", compute: { module: secondModule } });
    const firstGroup = device.createBindGroup({ layout: first.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: gate } },
      { binding: 2, resource: { buffer: up } }, { binding: 3, resource: { buffer: activated } },
    ] });
    const secondGroup = device.createBindGroup({ layout: second.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: activated } }, { binding: 1, resource: { buffer: down } },
      { binding: 2, resource: { buffer: output } },
    ] });
    const encoder = device.createCommandEncoder();
    const firstPass = encoder.beginComputePass();
    firstPass.setPipeline(first); firstPass.setBindGroup(0, firstGroup);
    firstPass.dispatchWorkgroups(Math.ceil((rows * intermediateSize) / 64)); firstPass.end();
    const secondPass = encoder.beginComputePass();
    secondPass.setPipeline(second); secondPass.setBindGroup(0, secondGroup);
    secondPass.dispatchWorkgroups(Math.ceil((rows * hiddenSize) / 64)); secondPass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, rows * hiddenSize * 4);
    device.queue.submit([encoder.finish()]);
    await abortable(readback.mapAsync(GPUMapMode.READ), signal);
    assertExecutionAllowed(signal);
    return new Float32Array(readback.getMappedRange().slice(0));
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    buffers.forEach((buffer) => buffer.destroy());
  }
}

function float32FromBase64(value: string): Float32Array {
  const binary = atob(value);
  if (binary.length % 4 !== 0) return new Float32Array();
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return new Float32Array(bytes.buffer);
}

function float32ToBase64(values: Float32Array): string {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  let binary = "";
  const chunk = 0x8000;
  for (let index = 0; index < bytes.length; index += chunk) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunk));
  }
  return btoa(binary);
}

function hex(value: ArrayBuffer): string {
  return [...new Uint8Array(value)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function runMatrixTask(size: number, seed: number, signal: AbortSignal): Promise<ComputeResult> {
  const device = state.device;
  const execution = await runWithMobileBackendFallback({
    backend: state.backend === "webgpu" && device ? "webgpu" : "cpu",
    runGpu: async () => {
      if (!device) throw new Error("WebGPU device is unavailable");
      const result = await measureMatrixTask(size, () => multiplyWebGpu(device, size, seed, signal), signal);
      if (!matrixSamplesMatch(result.samples, size, seed)) throw new Error("GPU matrix canary mismatch");
      return result;
    },
    runCpu: () => measureMatrixTask(size, () => multiplyCpu(size, seed, signal), signal),
    onGpuFailure: (error) => transitionToCpuFallback(
      `matrix kernel failed: ${errorText(error)}`,
      device,
    ),
    signal,
  });
  return execution.value;
}

async function runStartupBenchmark(
  seed: number,
  signal: AbortSignal,
): Promise<{ size: number; result: ComputeResult }> {
  const device = state.device;
  const execution = await runWithMobileBackendFallback({
    backend: state.backend === "webgpu" && device ? "webgpu" : "cpu",
    runGpu: async () => {
      if (!device) throw new Error("WebGPU device is unavailable");
      const size = 96;
      const result = await measureMatrixTask(size, () => multiplyWebGpu(device, size, seed, signal), signal);
      if (!matrixSamplesMatch(result.samples, size, seed)) throw new Error("GPU matrix canary mismatch");
      return { size, result };
    },
    runCpu: async () => {
      const size = 48;
      return { size, result: await measureMatrixTask(size, () => multiplyCpu(size, seed, signal), signal) };
    },
    onGpuFailure: (error) => transitionToCpuFallback(
      `startup canary failed: ${errorText(error)}`,
      device,
    ),
    signal,
  });
  return execution.value;
}

async function measureMatrixTask(
  size: number,
  compute: () => Promise<number[]>,
  signal: AbortSignal,
): Promise<ComputeResult> {
  assertExecutionAllowed(signal);
  const started = performance.now();
  const samples = await compute();
  assertExecutionAllowed(signal);
  const durationMs = Math.max(0.01, performance.now() - started);
  const estimatedGflops = (2 * size * size * size) / durationMs / 1_000_000;
  return { samples, durationMs, estimatedGflops };
}

async function multiplyWebGpu(
  device: GPUDevice,
  size: number,
  seed: number,
  signal: AbortSignal,
): Promise<number[]> {
  assertExecutionAllowed(signal);
  const elementCount = size * size;
  const byteLength = elementCount * Float32Array.BYTES_PER_ELEMENT;
  if (byteLength > Number(device.limits.maxBufferSize)) throw new Error("The matrix does not fit on the GPU");
  const matrixA = new Float32Array(elementCount);
  const matrixB = new Float32Array(elementCount);
  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column < size; column += 1) {
      matrixA[row * size + column] = valueA(row, column, seed);
      matrixB[row * size + column] = valueB(row, column, seed);
    }
  }
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const aBuffer = device.createBuffer({ label: "matrix-a", size: byteLength, usage });
  const bBuffer = device.createBuffer({ label: "matrix-b", size: byteLength, usage });
  const output = device.createBuffer({
    label: "matrix-output",
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: "matrix-readback",
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    device.queue.writeBuffer(aBuffer, 0, matrixA);
    device.queue.writeBuffer(bBuffer, 0, matrixB);
    const module = device.createShaderModule({
      label: "mycellios-matmul",
      code: `
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> c: array<f32>;
        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          let row = id.y;
          let column = id.x;
          if (row >= ${size}u || column >= ${size}u) { return; }
          var sum = 0.0;
          for (var k = 0u; k < ${size}u; k = k + 1u) {
            sum = sum + a[row * ${size}u + k] * b[k * ${size}u + column];
          }
          c[row * ${size}u + column] = sum;
        }
      `,
    });
    const pipeline = device.createComputePipeline({
      label: "mycellios-matmul-pipeline",
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuffer } },
        { binding: 1, resource: { buffer: bBuffer } },
        { binding: 2, resource: { buffer: output } },
      ],
    });
    const commands = device.createCommandEncoder();
    const pass = commands.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(size / 8), Math.ceil(size / 8));
    pass.end();
    commands.copyBufferToBuffer(output, 0, readback, 0, byteLength);
    device.queue.submit([commands.finish()]);
    await abortable(readback.mapAsync(GPUMapMode.READ), signal);
    assertExecutionAllowed(signal);
    const values = new Float32Array(readback.getMappedRange());
    return samplePositions(size).map(([row, column]) => values[row * size + column] ?? Number.NaN);
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    aBuffer.destroy();
    bBuffer.destroy();
    output.destroy();
    readback.destroy();
  }
}

async function multiplyCpu(size: number, seed: number, signal: AbortSignal): Promise<number[]> {
  assertExecutionAllowed(signal);
  const positions = samplePositions(size);
  const wanted = new Map(positions.map(([row, column], index) => [`${row}:${column}`, index]));
  const samples = new Array<number>(positions.length).fill(0);
  let operationsUntilYield = 16_384;
  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column < size; column += 1) {
      let sum = 0;
      for (let k = 0; k < size; k += 1) {
        sum += valueA(row, k, seed) * valueB(k, column, seed);
        operationsUntilYield -= 1;
        if (operationsUntilYield === 0) {
          operationsUntilYield = 16_384;
          await yieldToBrowser(signal);
        }
      }
      const sampleIndex = wanted.get(`${row}:${column}`);
      if (sampleIndex !== undefined) samples[sampleIndex] = sum;
    }
    if (row % 8 === 0) await yieldToBrowser(signal);
  }
  return samples;
}

function valueA(row: number, column: number, seed: number): number {
  return ((row * 3 + column * 5 + seed) % 31 - 15) / 16;
}

function valueB(row: number, column: number, seed: number): number {
  return ((row * 7 + column * 11 + seed * 3) % 29 - 14) / 16;
}

function samplePositions(size: number): Array<readonly [number, number]> {
  return [[0, 0], [Math.floor(size / 2), Math.floor(size / 3)], [size - 1, size - 1], [Math.floor(size / 3), Math.max(0, size - 2)]];
}

function matrixSamplesMatch(samples: number[], size: number, seed: number): boolean {
  if (samples.length !== samplePositions(size).length) return false;
  return samplePositions(size).every(([row, column], sampleIndex) => {
    let expected = 0;
    for (let k = 0; k < size; k += 1) expected += valueA(row, k, seed) * valueB(k, column, seed);
    const actual = samples[sampleIndex];
    return actual !== undefined && Number.isFinite(actual) && Math.abs(actual - expected) < 0.02;
  });
}

function float32Close(actual: Float32Array, expected: Float32Array, tolerance: number): boolean {
  return actual.length > 0 && actual.length === expected.length
    && actual.every((value, index) => Number.isFinite(value)
      && Math.abs(value - (expected[index] ?? Number.POSITIVE_INFINITY)) <= tolerance);
}

async function acquireWakeLock(signal?: AbortSignal): Promise<void> {
  if (signal) assertExecutionAllowed(signal);
  if (!("wakeLock" in navigator) || document.visibilityState !== "visible") {
    renderWakeLock();
    return;
  }
  try {
    const wakeLock = signal
      ? await abortable(navigator.wakeLock.request("screen"), signal)
      : await navigator.wakeLock.request("screen");
    if (signal) {
      try {
        assertExecutionAllowed(signal);
      } catch (error) {
        await wakeLock.release().catch(() => undefined);
        throw error;
      }
    }
    state.wakeLock = wakeLock;
    wakeLock.addEventListener("release", renderWakeLock, { once: true });
  } catch (error) {
    if (isMobileExecutionCancelled(error)) throw error;
    if (signal) throwIfMobileExecutionCancelled(signal);
    state.wakeLock = null;
  }
  renderWakeLock();
}

function sendHeartbeat(): void {
  send("mobile.heartbeat", {
    visible: document.visibilityState === "visible",
    wakeLock: Boolean(state.wakeLock && !state.wakeLock.released),
    backend: state.backend,
    estimatedGflops: state.estimatedGflops,
  });
}

function requestWork(): void {
  if (!state.running || document.visibilityState !== "visible" || viewState.connection !== "online") return;
  send("work.request", {});
}

function send(type: string, payload: unknown): void {
  if (state.socket?.readyState !== WebSocket.OPEN) return;
  state.socket.send(JSON.stringify({ v: 1, type, payload }));
}

function renderPerformance(): void {
  emit();
}

function renderWakeLock(): void {
  emit();
}

function setConnection(kind: "online" | "offline" | "connecting", label: string): void {
  viewState.connection = kind;
  viewState.connectionLabel = label;
  emit();
}

function setBusy(_busy: boolean): void {
  emit();
}

function setStatus(text: string): void {
  viewState.statusText = text;
  emit();
}

function addLog(text: string): void {
  const entry = `${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} · ${text}`;
  viewState.activity = [entry, ...viewState.activity].slice(0, 50);
  emit();
}

function beginExecutionSession(reason: string): AbortSignal {
  state.executionController.abort(new MobileExecutionCancelledError(reason));
  state.executionController = new AbortController();
  return state.executionController.signal;
}

function cancelActiveExecutions(reason: string): void {
  state.executionController.abort(new MobileExecutionCancelledError(reason));
  state.executionController = new AbortController();
  const device = state.device;
  state.device = null;
  state.gpu = null;
  state.backend = "cpu";
  if (state.detectedGpuLabel) viewState.gpuLabel = mobileCpuFallbackLabel(state.detectedGpuLabel);
  device?.destroy();
}

function assertExecutionAllowed(signal: AbortSignal): void {
  throwIfMobileExecutionCancelled(signal);
  if ((!state.running && !state.starting) || document.visibilityState !== "visible") {
    throw new MobileExecutionCancelledError("Browser contribution is paused");
  }
}

function abortable<T>(promise: PromiseLike<T>, signal: AbortSignal): Promise<T> {
  throwIfMobileExecutionCancelled(signal);
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(new MobileExecutionCancelledError());
    signal.addEventListener("abort", onAbort, { once: true });
    Promise.resolve(promise).then(
      (value) => {
        signal.removeEventListener("abort", onAbort);
        try {
          throwIfMobileExecutionCancelled(signal);
          resolve(value);
        } catch (error) {
          reject(error);
        }
      },
      (error) => {
        signal.removeEventListener("abort", onAbort);
        if (signal.aborted) reject(new MobileExecutionCancelledError());
        else reject(error);
      },
    );
  });
}

async function yieldToBrowser(signal: AbortSignal): Promise<void> {
  await abortable(new Promise<void>((resolve) => window.setTimeout(resolve, 0)), signal);
  assertExecutionAllowed(signal);
}

function transitionToCpuFallback(reason: string, failedDevice: GPUDevice | null): void {
  if (failedDevice && state.device !== failedDevice) return;
  const device = state.device;
  state.device = null;
  state.gpu = null;
  state.backend = "cpu";
  viewState.gpuLabel = mobileCpuFallbackLabel(state.detectedGpuLabel);
  emit();
  sendHeartbeat();
  addLog(`GPU unavailable; continuing with CPU (${reason}).`);
  device?.destroy();
}

function coolDownMs(): number {
  return state.level === "low" ? 1_500 : state.level === "balanced" ? 600 : 150;
}

async function persistentClientId(): Promise<string> {
  const key = "mycellios.mobile.client-id";
  const legacy = safeLocalStorageGet(key);
  try {
    const database = await openIdentityDatabase();
    const stored = await readIdentityValue(database, key);
    const identity = stored || legacy || crypto.randomUUID();
    await writeIdentityValue(database, key, identity);
    database.close();
    safeLocalStorageSet(key, identity);
    return identity;
  } catch {
    const identity = legacy || crypto.randomUUID();
    safeLocalStorageSet(key, identity);
    return identity;
  }
}

interface BrowserAdmissionCredential {
  schema: "mycellios-browser-credential/1";
  publicKeySpki: string;
  privateKey: CryptoKey;
}

async function persistentBrowserAdmissionCredential(): Promise<BrowserAdmissionCredential> {
  if (!crypto.subtle) throw new Error("WebCrypto is required for secure network admission.");
  const database = await openIdentityDatabase();
  try {
    const stored = await readIdentityRecord(database, "mycellios.mobile.admission-key");
    if (isBrowserAdmissionCredential(stored)) return stored;
    const generated = await crypto.subtle.generateKey(
      { name: "ECDSA", namedCurve: "P-256" },
      true,
      ["sign", "verify"],
    ) as CryptoKeyPair;
    const publicKeySpki = base64UrlFromBytes(
      new Uint8Array(await crypto.subtle.exportKey("spki", generated.publicKey)),
    );
    const privatePkcs8 = await crypto.subtle.exportKey("pkcs8", generated.privateKey);
    const privateKey = await crypto.subtle.importKey(
      "pkcs8",
      privatePkcs8,
      { name: "ECDSA", namedCurve: "P-256" },
      false,
      ["sign"],
    );
    const credential: BrowserAdmissionCredential = {
      schema: "mycellios-browser-credential/1",
      publicKeySpki,
      privateKey,
    };
    await writeIdentityRecord(database, "mycellios.mobile.admission-key", credential);
    return credential;
  } finally {
    database.close();
  }
}

function isBrowserAdmissionCredential(value: unknown): value is BrowserAdmissionCredential {
  if (!value || typeof value !== "object") return false;
  const credential = value as Partial<BrowserAdmissionCredential>;
  return (
    credential.schema === "mycellios-browser-credential/1"
    && typeof credential.publicKeySpki === "string"
    && /^[A-Za-z0-9_-]{40,256}$/.test(credential.publicKeySpki)
    && typeof CryptoKey !== "undefined"
    && credential.privateKey instanceof CryptoKey
    && credential.privateKey.type === "private"
    && credential.privateKey.algorithm.name === "ECDSA"
  );
}

async function signBrowserAdmissionPayload(
  privateKey: CryptoKey,
  payload: string,
): Promise<string> {
  const signature = await crypto.subtle.sign(
    { name: "ECDSA", hash: "SHA-256" },
    privateKey,
    new TextEncoder().encode(payload),
  );
  return base64UrlFromBytes(new Uint8Array(signature));
}

async function sha256CanonicalBrowserEvidence(value: unknown): Promise<string> {
  const canonical = canonicalBrowserEvidence(value, "$", new Set<object>());
  const digest = new Uint8Array(
    await crypto.subtle.digest("SHA-256", new TextEncoder().encode(canonical)),
  );
  return `sha256:${[...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

function canonicalBrowserEvidence(
  value: unknown,
  path: string,
  ancestors: Set<object>,
): string {
  if (value === null) return "null";
  if (typeof value === "string" || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error(`non_finite_registration_value:${path}`);
    return JSON.stringify(value);
  }
  if (typeof value !== "object") {
    throw new Error(`unsupported_registration_value:${path}`);
  }
  if (ancestors.has(value)) throw new Error(`cyclic_registration_value:${path}`);
  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      return `[${value.map((item, index) =>
        canonicalBrowserEvidence(item, `${path}[${index}]`, ancestors)).join(",")}]`;
    }
    const record = value as Record<string, unknown>;
    const keys = Object.keys(record).filter((key) => record[key] !== undefined).sort();
    return `{${keys.map((key) =>
      `${JSON.stringify(key)}:${canonicalBrowserEvidence(
        record[key],
        `${path}.${key}`,
        ancestors,
      )}`).join(",")}}`;
  } finally {
    ancestors.delete(value);
  }
}

function base64UrlFromBytes(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}

async function requestPersistentBrowserStorage(): Promise<boolean> {
  try {
    if (!navigator.storage?.persist) return false;
    if (await navigator.storage.persisted()) return true;
    return await navigator.storage.persist();
  } catch {
    return false;
  }
}

function openIdentityDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (!globalThis.indexedDB) {
      reject(new Error("IndexedDB is unavailable"));
      return;
    }
    const request = indexedDB.open("mycellios-mobile-identity", 1);
    request.addEventListener("upgradeneeded", () => {
      if (!request.result.objectStoreNames.contains("identity")) {
        request.result.createObjectStore("identity");
      }
    });
    request.addEventListener("success", () => resolve(request.result));
    request.addEventListener("error", () => reject(request.error ?? new Error("Could not open identity storage")));
  });
}

function readIdentityValue(database: IDBDatabase, key: string): Promise<string | null> {
  return readIdentityRecord(database, key).then(
    (value) => typeof value === "string" ? value : null,
  );
}

function readIdentityRecord(database: IDBDatabase, key: string): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const request = database.transaction("identity", "readonly").objectStore("identity").get(key);
    request.addEventListener("success", () => resolve(request.result as unknown));
    request.addEventListener("error", () => reject(request.error ?? new Error("Could not read browser identity")));
  });
}

function writeIdentityValue(database: IDBDatabase, key: string, value: string): Promise<void> {
  return writeIdentityRecord(database, key, value);
}

function writeIdentityRecord(database: IDBDatabase, key: string, value: unknown): Promise<void> {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction("identity", "readwrite");
    transaction.objectStore("identity").put(value, key);
    transaction.addEventListener("complete", () => resolve());
    transaction.addEventListener("abort", () => reject(transaction.error ?? new Error("Could not save browser identity")));
    transaction.addEventListener("error", () => reject(transaction.error ?? new Error("Could not save browser identity")));
  });
}

function safeLocalStorageGet(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeLocalStorageSet(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // IndexedDB remains the primary identity store.
  }
}

function mobileName(): string {
  const platform =
    (navigator as Navigator & { userAgentData?: { platform?: string } }).userAgentData?.platform ||
    navigator.platform ||
    "Mobile";
  return `${platform} mobile`;
}

function deviceMemory(): number | undefined {
  return (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
}

function formatGflops(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "—";
  if (value < 1) return value.toFixed(2);
  if (value < 100) return value.toFixed(1);
  return Math.round(value).toLocaleString();
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function currentSnapshot(): MobileWorkerSnapshot {
  return {
    running: state.running,
    starting: state.starting,
    backend: state.backend,
    level: state.level,
    connection: viewState.connection,
    connectionLabel: viewState.connectionLabel,
    statusText: viewState.statusText,
    gpuLabel: viewState.gpuLabel,
    estimatedGflops: state.estimatedGflops,
    verifiedTasks: state.verifiedTasks,
    wakeLockActive: Boolean(state.wakeLock && !state.wakeLock.released),
    activity: [...viewState.activity],
  };
}

function emit(): void {
  const snapshot = currentSnapshot();
  for (const listener of listeners) listener(snapshot);
}
