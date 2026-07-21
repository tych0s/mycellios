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
  residentExperts: Map<string, ResidentExpert>;
  cancelledMatrixTasks: Set<string>;
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
  residentExperts: new Map(),
  cancelledMatrixTasks: new Set(),
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
  if (!state.running) return;
  if (document.visibilityState === "visible") {
    void acquireWakeLock();
    sendHeartbeat();
    requestWork();
  } else {
    sendHeartbeat();
    setStatus("Paused because mycellios is not visible.");
  }
}

async function start(): Promise<void> {
  if (state.starting || state.running) return;
  state.starting = true;
  setBusy(true);
  setConnection("connecting", "Preparing");
  addLog("Checking the device compute engine.");
  try {
    await initializeComputeBackend();
    setStatus(`Measuring the ${state.backend === "webgpu" ? "GPU" : "CPU"} with a real task…`);
    const benchmarkSize = state.backend === "webgpu" ? 96 : 48;
    const benchmark = await runMatrixTask(benchmarkSize, 73);
    state.estimatedGflops = benchmark.estimatedGflops;
    renderPerformance();
    addLog(`Benchmark complete: ${formatGflops(state.estimatedGflops)} GFLOPS.`);
    await acquireWakeLock();
    const credentials = await registerWorker(benchmarkSize, benchmark.durationMs);
    state.workerId = credentials.workerId;
    state.token = credentials.token;
    state.running = true;
    setBusy(false);
    connect();
  } catch (error) {
    addLog(`Could not start: ${errorText(error)}`);
    setStatus(errorText(error));
    setConnection("offline", "Unavailable");
    await stop(false);
  } finally {
    state.starting = false;
    setBusy(false);
  }
}

async function stop(log = true): Promise<void> {
  state.running = false;
  state.starting = false;
  if (state.heartbeatTimer !== null) window.clearInterval(state.heartbeatTimer);
  if (state.reconnectTimer !== null) window.clearTimeout(state.reconnectTimer);
  state.heartbeatTimer = null;
  state.reconnectTimer = null;
  state.socket?.close(1000, "user stopped contribution");
  state.socket = null;
  await state.wakeLock?.release().catch(() => undefined);
  state.wakeLock = null;
  state.device?.destroy();
  state.device = null;
  state.gpu = null;
  state.residentExperts.clear();
  state.cancelledMatrixTasks.clear();
  renderWakeLock();
  setBusy(false);
  setConnection("offline", "Disconnected");
  setStatus("Press the button and keep this screen open.");
  if (log) addLog("Contribution stopped by the user.");
}

async function detectBackendPreview(): Promise<void> {
  if (!navigator.gpu) {
    state.backend = "cpu";
    viewState.gpuLabel = "Universal CPU fallback";
    emit();
    return;
  }
  state.backend = "webgpu";
  viewState.gpuLabel = "GPU available";
  emit();
}

async function initializeComputeBackend(): Promise<void> {
  if (!window.isSecureContext) {
    state.backend = "cpu";
    viewState.gpuLabel = "WebGPU requires HTTPS";
    emit();
    return;
  }
  if (!navigator.gpu) {
    state.backend = "cpu";
    viewState.gpuLabel = "WebGPU unavailable";
    emit();
    return;
  }
  try {
    const compatibilityOptions = { featureLevel: "compatibility" } as GPURequestAdapterOptions;
    const adapter =
      (await navigator.gpu.requestAdapter(compatibilityOptions)) ??
      (await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }));
    if (!adapter) throw new Error("No WebGPU adapter found");
    const device = await adapter.requestDevice();
    device.lost.then((info) => {
      if (!state.running) return;
      addLog(`The GPU became unavailable: ${info.message || info.reason}.`);
      void stop(false);
    });
    state.backend = "webgpu";
    state.gpu = adapter;
    state.device = device;
    const info = adapter.info as GPUAdapterInfo & { description?: string };
    viewState.gpuLabel = info.description || info.device || info.architecture || "Mobile GPU";
    emit();
  } catch (error) {
    state.backend = "cpu";
    viewState.gpuLabel = "Browser fallback";
    emit();
    addLog(`WebGPU unavailable; using CPU: ${errorText(error)}`);
  }
}

async function registerWorker(matrixSize: number, durationMs: number): Promise<{ workerId: string; token: string }> {
  const query = new URLSearchParams(window.location.search);
  const invitationFromUrl = query.get("join")?.trim();
  const joinToken = persistentJoinToken(invitationFromUrl);
  const adapterInfo = state.gpu?.info as (GPUAdapterInfo & { description?: string }) | undefined;
  const body = {
    clientId: persistentClientId(),
    name: mobileName(),
    region: query.get("region")?.trim() || "auto",
    platform: navigator.userAgent.slice(0, 120),
    backend: state.backend,
    performanceLevel: state.level,
    ...(joinToken ? { joinToken } : {}),
    capabilities: {
      webgpu: Boolean(state.device),
      wasm: typeof WebAssembly === "object",
      hardwareConcurrency: Math.max(1, navigator.hardwareConcurrency || 1),
      ...(deviceMemory() ? { deviceMemoryGb: deviceMemory() } : {}),
      ...(adapterInfo?.description ? { gpuDescription: adapterInfo.description } : {}),
      ...(state.device ? { maxBufferSize: Number(state.device.limits.maxBufferSize) } : {}),
    },
    benchmark: {
      durationMs,
      estimatedGflops: state.estimatedGflops,
      matrixSize,
    },
  };
  const response = await fetch("/mobile/v1/register", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
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

function connect(): void {
  if (!state.running || !state.workerId || !state.token) return;
  setConnection("connecting", "Connecting");
  const url = new URL("/mobile/v1/connect", window.location.origin);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.searchParams.set("workerId", state.workerId);
  url.searchParams.set("token", state.token);
  const socket = new WebSocket(url);
  state.socket = socket;
  socket.addEventListener("open", () => addLog("Secure coordinator channel opened."));
  socket.addEventListener("message", (event) => void handleServerMessage(String(event.data)));
  socket.addEventListener("close", () => {
    if (state.socket === socket) state.socket = null;
    if (!state.running) return;
    setConnection("connecting", "Reconnecting");
    setStatus("Connection lost; retrying automatically…");
    state.reconnectTimer = window.setTimeout(connect, 2_000);
  });
  socket.addEventListener("error", () => socket.close());
}

async function handleServerMessage(raw: string): Promise<void> {
  let message: { type?: string; payload?: unknown };
  try {
    message = JSON.parse(raw) as { type?: string; payload?: unknown };
  } catch {
    return;
  }
  if (message.type === "server.ready") {
    setConnection("online", "Connected");
    setStatus("Contributing power. Keep mycellios visible.");
    addLog("Mobile worker registered and ready.");
    sendHeartbeat();
    if (state.heartbeatTimer !== null) window.clearInterval(state.heartbeatTimer);
    state.heartbeatTimer = window.setInterval(sendHeartbeat, 5_000);
    requestWork();
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
    const payload = message.payload as { phase?: string; verifiedTasks?: number };
    if (payload.phase === "executed") {
      state.verifiedTasks = payload.verifiedTasks ?? state.verifiedTasks + 1;
      emit();
      setStatus("Expert output verified and used by the model network.");
      addLog("Real MoE expert result accepted by the coordinator.");
    }
    return;
  }
  if (message.type === "expert.rejected") {
    addLog("Expert weights failed the network canary and were removed.");
  }
  if (message.type === "compute.verified") {
    const payload = message.payload as { verifiedTasks?: number };
    state.verifiedTasks = payload.verifiedTasks ?? state.verifiedTasks + 1;
    emit();
    addLog(`Result verified by the network · task ${state.verifiedTasks}.`);
    window.setTimeout(requestWork, coolDownMs());
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
  send("compute.accept", { taskId: offer.taskId, leaseId: offer.leaseId });
  setStatus(`Computing a ${offer.size}×${offer.size} matrix with ${state.backend.toUpperCase()}…`);
  addLog(`Job received: ${offer.size}×${offer.size} multiplication.`);
  try {
    const result = await runMatrixTask(offer.size, offer.seed);
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
  try {
    setStatus("Downloading and verifying a real model expert…");
    const manifestResponse = await fetch(new URL(offer.manifestUrl, window.location.origin));
    if (!manifestResponse.ok) throw new Error(`expert manifest HTTP ${manifestResponse.status}`);
    const manifest = (await manifestResponse.json()) as ExpertManifest;
    if (manifest.artifactId !== offer.artifactId || manifest.dtype !== "float32"
      || manifest.activation !== "silu") throw new Error("unsupported expert manifest");
    const weightsResponse = await fetch(
      new URL(`/mobile/v1/experts/weights/${manifest.weightsHash}`, window.location.origin),
    );
    if (!weightsResponse.ok) throw new Error(`expert weights HTTP ${weightsResponse.status}`);
    const bytes = await weightsResponse.arrayBuffer();
    const digest = hex(await crypto.subtle.digest("SHA-256", bytes));
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
    const computed = await computeSwiGlu(expert, canaryInput, 1);
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
  const expert = state.residentExperts.get(offer.artifactId);
  try {
    if (!expert) throw new Error("expert is not resident");
    if (expert.manifest.hiddenSize !== offer.hiddenSize) throw new Error("activation shape mismatch");
    const activations = float32FromBase64(offer.activationsBase64);
    if (activations.length !== offer.rows * offer.hiddenSize) throw new Error("activation payload mismatch");
    setStatus(`Running real model expert L${expert.manifest.layer}/E${expert.manifest.expert}…`);
    const started = performance.now();
    const computed = await computeSwiGlu(expert, activations, offer.rows);
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
): Promise<{ values: Float32Array; backend: Backend }> {
  if (state.device) {
    try {
      return { values: await swiGluWebGpu(state.device, expert, activations, rows), backend: "webgpu" };
    } catch (error) {
      addLog(`Expert WebGPU fallback to CPU: ${errorText(error)}`);
    }
  }
  return { values: await swiGluCpu(expert, activations, rows), backend: "cpu" };
}

async function swiGluCpu(
  expert: ResidentExpert,
  activations: Float32Array,
  rows: number,
): Promise<Float32Array> {
  const { hiddenSize, intermediateSize } = expert.manifest;
  const activated = new Float32Array(rows * intermediateSize);
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
      }
      activated[row * intermediateSize + intermediate] = (gate / (1 + Math.exp(-gate))) * up;
    }
    if (row % 4 === 0) await new Promise<void>((resolvePromise) => window.setTimeout(resolvePromise, 0));
  }
  const output = new Float32Array(rows * hiddenSize);
  for (let row = 0; row < rows; row += 1) {
    for (let hidden = 0; hidden < hiddenSize; hidden += 1) {
      let sum = 0;
      const downOffset = hidden * intermediateSize;
      for (let intermediate = 0; intermediate < intermediateSize; intermediate += 1) {
        sum += (activated[row * intermediateSize + intermediate] ?? 0)
          * (expert.down[downOffset + intermediate] ?? 0);
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
): Promise<Float32Array> {
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
    await readback.mapAsync(GPUMapMode.READ);
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

async function runMatrixTask(size: number, seed: number): Promise<ComputeResult> {
  const started = performance.now();
  const samples = state.backend === "webgpu" && state.device
    ? await multiplyWebGpu(state.device, size, seed)
    : await multiplyCpu(size, seed);
  const durationMs = Math.max(0.01, performance.now() - started);
  const estimatedGflops = (2 * size * size * size) / durationMs / 1_000_000;
  return { samples, durationMs, estimatedGflops };
}

async function multiplyWebGpu(device: GPUDevice, size: number, seed: number): Promise<number[]> {
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
    await readback.mapAsync(GPUMapMode.READ);
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

async function multiplyCpu(size: number, seed: number): Promise<number[]> {
  const positions = samplePositions(size);
  const wanted = new Map(positions.map(([row, column], index) => [`${row}:${column}`, index]));
  const samples = new Array<number>(positions.length).fill(0);
  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column < size; column += 1) {
      let sum = 0;
      for (let k = 0; k < size; k += 1) sum += valueA(row, k, seed) * valueB(k, column, seed);
      const sampleIndex = wanted.get(`${row}:${column}`);
      if (sampleIndex !== undefined) samples[sampleIndex] = sum;
    }
    if (row % 8 === 0) await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
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

async function acquireWakeLock(): Promise<void> {
  if (!("wakeLock" in navigator) || document.visibilityState !== "visible") {
    renderWakeLock();
    return;
  }
  try {
    state.wakeLock = await navigator.wakeLock.request("screen");
    state.wakeLock.addEventListener("release", renderWakeLock, { once: true });
  } catch {
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
  if (!state.running || document.visibilityState !== "visible") return;
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
  viewState.activity = [entry, ...viewState.activity].slice(0, 5);
  emit();
}

function coolDownMs(): number {
  return state.level === "low" ? 1_500 : state.level === "balanced" ? 600 : 150;
}

function persistentClientId(): string {
  const key = "mycellios.mobile.client-id";
  const stored = localStorage.getItem(key);
  if (stored) return stored;
  const created = crypto.randomUUID();
  localStorage.setItem(key, created);
  return created;
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
