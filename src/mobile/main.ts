import "./styles.css";

type Backend = "webgpu" | "cpu";
type Level = "low" | "balanced" | "maximum";

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
};

const elements = {
  toggle: required<HTMLButtonElement>("toggleButton"),
  badge: required<HTMLElement>("connectionBadge"),
  status: required<HTMLElement>("statusText"),
  backend: required<HTMLElement>("backendValue"),
  gpu: required<HTMLElement>("gpuValue"),
  gflops: required<HTMLElement>("gflopsValue"),
  tasks: required<HTMLElement>("tasksValue"),
  wake: required<HTMLElement>("wakeValue"),
  log: required<HTMLOListElement>("activityLog"),
  levels: [...document.querySelectorAll<HTMLButtonElement>("[data-level]")],
};

elements.toggle.addEventListener("click", () => void (state.running ? stop() : start()));
for (const button of elements.levels) {
  button.addEventListener("click", () => {
    if (state.running || state.starting) return;
    const level = button.dataset.level;
    if (level !== "low" && level !== "balanced" && level !== "maximum") return;
    state.level = level;
    elements.levels.forEach((candidate) => candidate.classList.toggle("active", candidate === button));
  });
}

document.addEventListener("visibilitychange", () => {
  if (!state.running) return;
  if (document.visibilityState === "visible") {
    void acquireWakeLock();
    sendHeartbeat();
    requestWork();
  } else {
    sendHeartbeat();
    setStatus("En pausa porque Mycellios no está visible.");
  }
});

if ("serviceWorker" in navigator && window.isSecureContext) {
  window.addEventListener("load", () => void navigator.serviceWorker.register("./sw.js"));
}

void detectBackendPreview();

async function start(): Promise<void> {
  if (state.starting || state.running) return;
  state.starting = true;
  setBusy(true);
  setConnection("connecting", "Preparando");
  addLog("Comprobando el motor de cálculo del dispositivo.");
  try {
    await initializeComputeBackend();
    setStatus(`Midiendo ${state.backend === "webgpu" ? "la GPU" : "la CPU"} con una tarea real…`);
    const benchmarkSize = state.backend === "webgpu" ? 96 : 48;
    const benchmark = await runMatrixTask(benchmarkSize, 73);
    state.estimatedGflops = benchmark.estimatedGflops;
    renderPerformance();
    addLog(`Benchmark terminado: ${formatGflops(state.estimatedGflops)} GFLOPS.`);
    await acquireWakeLock();
    const credentials = await registerWorker(benchmarkSize, benchmark.durationMs);
    state.workerId = credentials.workerId;
    state.token = credentials.token;
    state.running = true;
    document.body.classList.add("running");
    setBusy(false);
    elements.toggle.innerHTML = '<span class="power-symbol">⏻</span><span>Dejar de aportar</span>';
    elements.levels.forEach((button) => { button.disabled = true; });
    connect();
  } catch (error) {
    addLog(`No se pudo iniciar: ${errorText(error)}`);
    setStatus(errorText(error));
    setConnection("offline", "No disponible");
    await stop(false);
  } finally {
    state.starting = false;
    if (!state.running) setBusy(false);
  }
}

async function stop(log = true): Promise<void> {
  state.running = false;
  state.starting = false;
  document.body.classList.remove("running");
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
  renderWakeLock();
  setBusy(false);
  setConnection("offline", "Desconectado");
  setStatus("Pulsa el botón y mantén esta pantalla abierta.");
  elements.toggle.innerHTML = '<span class="power-symbol">⏻</span><span>Aportar potencia</span>';
  elements.levels.forEach((button) => { button.disabled = false; });
  if (log) addLog("Aportación detenida por el usuario.");
}

async function detectBackendPreview(): Promise<void> {
  if (!navigator.gpu) {
    elements.backend.textContent = "CPU";
    elements.gpu.textContent = "Fallback universal";
    return;
  }
  elements.backend.textContent = "WebGPU";
  elements.gpu.textContent = "GPU disponible";
}

async function initializeComputeBackend(): Promise<void> {
  if (!window.isSecureContext) {
    state.backend = "cpu";
    elements.backend.textContent = "CPU";
    elements.gpu.textContent = "WebGPU requiere HTTPS";
    return;
  }
  if (!navigator.gpu) {
    state.backend = "cpu";
    elements.backend.textContent = "CPU";
    elements.gpu.textContent = "WebGPU no disponible";
    return;
  }
  try {
    const compatibilityOptions = { featureLevel: "compatibility" } as GPURequestAdapterOptions;
    const adapter =
      (await navigator.gpu.requestAdapter(compatibilityOptions)) ??
      (await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }));
    if (!adapter) throw new Error("No hay adaptador WebGPU");
    const device = await adapter.requestDevice();
    device.lost.then((info) => {
      if (!state.running) return;
      addLog(`La GPU dejó de estar disponible: ${info.message || info.reason}.`);
      void stop(false);
    });
    state.backend = "webgpu";
    state.gpu = adapter;
    state.device = device;
    elements.backend.textContent = "WebGPU";
    const info = adapter.info as GPUAdapterInfo & { description?: string };
    elements.gpu.textContent = info.description || info.device || info.architecture || "GPU del móvil";
  } catch (error) {
    state.backend = "cpu";
    elements.backend.textContent = "CPU";
    elements.gpu.textContent = "Fallback del navegador";
    addLog(`WebGPU no disponible; se usará CPU: ${errorText(error)}`);
  }
}

async function registerWorker(matrixSize: number, durationMs: number): Promise<{ workerId: string; token: string }> {
  const query = new URLSearchParams(window.location.search);
  const joinToken = query.get("join")?.trim();
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
  const response = await fetch("./v1/register", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    if (response.status === 401) throw new Error("El enlace de invitación no es válido.");
    throw new Error(`El coordinador rechazó el registro (HTTP ${response.status}).`);
  }
  return (await response.json()) as { workerId: string; token: string };
}

function connect(): void {
  if (!state.running || !state.workerId || !state.token) return;
  setConnection("connecting", "Conectando");
  const url = new URL("./v1/connect", window.location.href);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.searchParams.set("workerId", state.workerId);
  url.searchParams.set("token", state.token);
  const socket = new WebSocket(url);
  state.socket = socket;
  socket.addEventListener("open", () => addLog("Canal seguro con el coordinador abierto."));
  socket.addEventListener("message", (event) => void handleServerMessage(String(event.data)));
  socket.addEventListener("close", () => {
    if (state.socket === socket) state.socket = null;
    if (!state.running) return;
    setConnection("connecting", "Reconectando");
    setStatus("Se perdió la conexión; reintentando automáticamente…");
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
    setConnection("online", "Conectado");
    setStatus("Aportando potencia. Mantén Mycellios visible.");
    addLog("Worker móvil registrado y listo para trabajar.");
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
  if (message.type === "compute.verified") {
    const payload = message.payload as { verifiedTasks?: number };
    state.verifiedTasks = payload.verifiedTasks ?? state.verifiedTasks + 1;
    elements.tasks.textContent = String(state.verifiedTasks);
    addLog(`Resultado verificado por la red · tarea ${state.verifiedTasks}.`);
    window.setTimeout(requestWork, coolDownMs());
    return;
  }
  if (message.type === "compute.rejected") {
    addLog("El resultado no superó la verificación y fue descartado.");
    window.setTimeout(requestWork, coolDownMs());
  }
}

async function executeOffer(offer: ComputeOffer): Promise<void> {
  if (!state.running || document.visibilityState !== "visible") return;
  if (offer.operation !== "matrix-multiply" || offer.deadlineAt <= Date.now()) return;
  send("compute.accept", { taskId: offer.taskId, leaseId: offer.leaseId });
  setStatus(`Calculando una matriz ${offer.size}×${offer.size} con ${state.backend.toUpperCase()}…`);
  addLog(`Trabajo recibido: multiplicación ${offer.size}×${offer.size}.`);
  try {
    const result = await runMatrixTask(offer.size, offer.seed);
    state.estimatedGflops = result.estimatedGflops;
    renderPerformance();
    send("compute.result", {
      taskId: offer.taskId,
      leaseId: offer.leaseId,
      backend: state.backend,
      ...result,
    });
    setStatus("Resultado enviado; esperando verificación del coordinador…");
  } catch (error) {
    send("compute.fail", {
      taskId: offer.taskId,
      leaseId: offer.leaseId,
      message: errorText(error).slice(0, 300),
    });
    addLog(`La tarea falló: ${errorText(error)}`);
    window.setTimeout(requestWork, coolDownMs());
  }
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
  if (byteLength > Number(device.limits.maxBufferSize)) throw new Error("La matriz no cabe en la GPU");
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
  elements.gflops.textContent = formatGflops(state.estimatedGflops);
}

function renderWakeLock(): void {
  const active = Boolean(state.wakeLock && !state.wakeLock.released);
  elements.wake.textContent = active ? "Activa" : "Normal";
}

function setConnection(kind: "online" | "offline" | "connecting", label: string): void {
  elements.badge.className = `badge ${kind}`;
  elements.badge.textContent = label;
}

function setBusy(busy: boolean): void {
  elements.toggle.disabled = busy;
  if (busy) elements.toggle.innerHTML = '<span class="power-symbol">◌</span><span>Preparando…</span>';
}

function setStatus(text: string): void {
  elements.status.textContent = text;
}

function addLog(text: string): void {
  const item = document.createElement("li");
  item.textContent = `${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} · ${text}`;
  elements.log.prepend(item);
  while (elements.log.children.length > 5) elements.log.lastElementChild?.remove();
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
    "Móvil";
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

function required<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing required element #${id}`);
  return element as T;
}
