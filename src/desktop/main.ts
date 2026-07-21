import { randomUUID } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { networkInterfaces } from "node:os";
import { delimiter, dirname, join } from "node:path";
import { spawn } from "node:child_process";
import type { AddressInfo } from "node:net";
import {
  app,
  autoUpdater,
  BrowserWindow,
  ipcMain,
  Menu,
  nativeImage,
  shell,
  Tray,
} from "electron";
import started from "electron-squirrel-startup";
import { workerConfigSchema, type WorkerConfig } from "../contracts/schemas.js";
import type { CoordinatorRuntime } from "../coordinator/server.js";
import { createCoordinator } from "../coordinator/server.js";
import { DynamicModelActivationManager } from "../coordinator/model-activation-manager.js";
import { parseAutoDistributionConfig } from "../distribution/auto-distribute.js";
import { LocalProcessAgent, type LaunchAgent } from "../distribution/launch-supervisor.js";
import type { PythonPipelineLaunchDescription } from "../distribution/python-launcher.js";
import { WorkerTunnelLaunchAgent } from "../distribution/worker-tunnel-launch-agent.js";
import type { StoredWorker } from "../storage/store.js";
import type { WorkerHub } from "../coordinator/worker-hub.js";
import { WorkerAgent, validateCoordinatorUrl } from "../worker/agent.js";
import { probeHardware, type HardwareProbe } from "../worker/hardware.js";
import type {
  ChatRequest,
  ChatResponse,
  ChatStreamUpdate,
  DashboardJob,
  DashboardModel,
  DashboardSnapshot,
  DashboardWorker,
  DesktopSettings,
  DesktopUpdateStatus,
} from "./contracts.js";
import type { HubCatalogModel } from "../contracts/types.js";
import { consumeChatCompletionStream } from "./chat-stream.js";

if (started) app.quit();

app.setName("mycellios");
if (process.platform === "win32") app.setAppUserModelId("app.mycellios.desktop.v2");

const PUBLIC_COORDINATOR_URL = "https://www.mycellios.com";
const LOCAL_DASHBOARD_COORDINATOR_URL = "http://127.0.0.1:4180";
const LOCAL_DASHBOARD_COORDINATOR_PORT = 4_180;

const DEFAULT_SETTINGS: DesktopSettings = {
  coordinatorMode: "remote",
  remoteCoordinatorUrl: PUBLIC_COORDINATOR_URL,
  remoteCoordinatorToken: "",
  contributionEnabled: false,
  launchAtLogin: false,
  closeToTray: true,
  onboardingComplete: false,
  region: "auto",
  offeredVramMb: 4_096,
  adapterMode: "connectivity-test",
  modelName: "mycellios-connectivity-check",
  adapterBaseUrl: "http://127.0.0.1:11434",
  modelDigest: "",
};

const UPDATE_FEED_URL = "https://www.mycellios.com/updates/win32/x64/";
const UPDATE_CHECK_INTERVAL_MS = 4 * 60 * 60 * 1_000;
const LEGACY_PUBLIC_COORDINATOR_URLS = new Set([
  "https://www.mycellios.com",
  "https://mycellios.com",
]);

let mainWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let coordinator: CoordinatorRuntime | null = null;
let coordinatorUrl = "";
let worker: WorkerAgent | null = null;
let distributedExecutor: Awaited<ReturnType<typeof createDesktopDistributedExecutor>> | null = null;
let distributionRuntimePromise: Promise<string> | null = null;
let hardwarePromise: Promise<HardwareProbe> | null = null;
let settings: DesktopSettings = DEFAULT_SETTINGS;
let isQuitting = false;
let runtimeError: string | null = null;
let updateCheckTimer: NodeJS.Timeout | null = null;
let updateCheckInFlight = false;
let updateStatus: DesktopUpdateStatus = {
  state: app.isPackaged ? "idle" : "development",
  currentVersion: app.getVersion(),
  availableVersion: null,
  message: app.isPackaged
    ? "Automatic updates are ready."
    : "Updates are disabled in development mode.",
  checkedAt: null,
};

function resourcePath(...segments: string[]): string {
  if (app.isPackaged) return join(process.resourcesPath, ...segments);
  return join(app.getAppPath(), ...segments);
}

function settingsPath(): string {
  return join(app.getPath("userData"), "desktop-settings.json");
}

function writeDesktopLog(event: string, details: unknown): void {
  try {
    appendFileSync(
      join(app.getPath("userData"), "mycellios.log"),
      `${new Date().toISOString()} ${event} ${JSON.stringify(details)}\n`,
      "utf8",
    );
  } catch {
    // Diagnostics must never prevent the app from starting.
  }
}

function setUpdateStatus(next: Partial<DesktopUpdateStatus>): void {
  updateStatus = { ...updateStatus, ...next };
  writeDesktopLog("update-status", updateStatus);
}

function configureAutomaticUpdates(): void {
  if (!app.isPackaged) return;
  if (process.platform !== "win32") {
    setUpdateStatus({
      state: "unsupported",
      message:
        process.platform === "darwin"
          ? "Automatic updates require a signed macOS release."
          : "Updates are delivered by your Linux package manager.",
    });
    return;
  }

  autoUpdater.setFeedURL({ url: UPDATE_FEED_URL });
  autoUpdater.on("checking-for-update", () => {
    setUpdateStatus({ state: "checking", message: "Checking for a new version…" });
  });
  autoUpdater.on("update-available", () => {
    setUpdateStatus({
      state: "downloading",
      message: "A new version is downloading in the background…",
    });
  });
  autoUpdater.on("update-not-available", () => {
    setUpdateStatus({
      state: "up-to-date",
      availableVersion: null,
      message: "mycellios is up to date.",
      checkedAt: new Date().toISOString(),
    });
  });
  autoUpdater.on("update-downloaded", (_event, _releaseNotes, releaseName) => {
    setUpdateStatus({
      state: "ready",
      availableVersion: releaseName || null,
      message: "Update ready. It will be installed automatically on the next restart.",
      checkedAt: new Date().toISOString(),
    });
  });
  autoUpdater.on("error", (error) => {
    setUpdateStatus({
      state: "error",
      message: `Could not check for updates: ${error.message}`,
      checkedAt: new Date().toISOString(),
    });
  });

  // Squirrel holds a file lock for a few seconds after first install.
  const initialDelayMs = process.argv.includes("--squirrel-firstrun") ? 15_000 : 10_000;
  setTimeout(() => void checkForUpdates(), initialDelayMs).unref();
  updateCheckTimer = setInterval(() => void checkForUpdates(), UPDATE_CHECK_INTERVAL_MS);
  updateCheckTimer.unref();
}

async function checkForUpdates(): Promise<DesktopUpdateStatus> {
  if (!app.isPackaged || process.platform !== "win32" || updateCheckInFlight) {
    return updateStatus;
  }
  updateCheckInFlight = true;
  try {
    await autoUpdater.checkForUpdates();
  } catch (error) {
    setUpdateStatus({
      state: "error",
      message: `Could not check for updates: ${errorText(error)}`,
      checkedAt: new Date().toISOString(),
    });
  } finally {
    updateCheckInFlight = false;
  }
  return updateStatus;
}

function loadSettings(): DesktopSettings {
  try {
    const stored = JSON.parse(readFileSync(settingsPath(), "utf8")) as Partial<DesktopSettings>;
    const storedCoordinatorUrl = stored.remoteCoordinatorUrl?.trim().replace(/\/+$/, "");
    const migrated = storedCoordinatorUrl && LEGACY_PUBLIC_COORDINATOR_URLS.has(storedCoordinatorUrl)
      ? { ...stored, remoteCoordinatorUrl: DEFAULT_SETTINGS.remoteCoordinatorUrl }
      : stored;
    const loaded = sanitizeSettings({ ...DEFAULT_SETTINGS, ...migrated });
    if (migrated !== stored) {
      writeFileSync(settingsPath(), `${JSON.stringify(loaded, null, 2)}\n`, "utf8");
    }
    return loaded;
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

function sanitizeSettings(input: DesktopSettings): DesktopSettings {
  const offeredVramMb = Number.isFinite(input.offeredVramMb)
    ? Math.max(512, Math.min(262_144, Math.round(input.offeredVramMb)))
    : DEFAULT_SETTINGS.offeredVramMb;
  const remoteCoordinatorUrl = input.remoteCoordinatorUrl.trim();
  const remoteCoordinatorToken = input.remoteCoordinatorToken.trim();
  if (input.coordinatorMode === "remote") validateCoordinatorUrl(remoteCoordinatorUrl);
  if (input.adapterMode === "local-model-runtime" && !input.modelDigest.trim()) {
    throw new Error("local model runtime requires a pinned model digest before contributing resources.");
  }
  return {
    coordinatorMode: input.coordinatorMode === "remote" ? "remote" : "local",
    remoteCoordinatorUrl,
    remoteCoordinatorToken,
    contributionEnabled: Boolean(input.contributionEnabled),
    launchAtLogin: Boolean(input.launchAtLogin),
    closeToTray: Boolean(input.closeToTray),
    onboardingComplete: Boolean(input.onboardingComplete),
    region: input.region.trim() || "auto",
    offeredVramMb,
    adapterMode: input.adapterMode === "local-model-runtime" ? "local-model-runtime" : "connectivity-test",
    modelName: input.modelName.trim() || DEFAULT_SETTINGS.modelName,
    adapterBaseUrl: input.adapterBaseUrl.trim() || DEFAULT_SETTINGS.adapterBaseUrl,
    modelDigest: input.modelDigest.trim(),
  };
}

function persistSettings(next: DesktopSettings): void {
  settings = sanitizeSettings(next);
  writeFileSync(settingsPath(), `${JSON.stringify(settings, null, 2)}\n`, "utf8");
  configureLaunchAtLogin(settings.launchAtLogin);
}

function configureLaunchAtLogin(enabled: boolean): void {
  if (process.platform !== "linux") {
    app.setLoginItemSettings({
      openAtLogin: enabled,
      ...(process.platform === "win32" ? { args: ["--hidden"] } : { openAsHidden: true }),
    });
    return;
  }

  const configHome = process.env.XDG_CONFIG_HOME?.trim() || join(app.getPath("home"), ".config");
  const autostartDirectory = join(configHome, "autostart");
  const desktopFile = join(autostartDirectory, "mycellios.desktop");
  if (!enabled) {
    rmSync(desktopFile, { force: true });
    return;
  }
  mkdirSync(autostartDirectory, { recursive: true });
  writeFileSync(
    desktopFile,
    [
      "[Desktop Entry]",
      "Type=Application",
      "Name=mycellios",
      `Exec=${escapeDesktopExec(process.execPath)} --hidden`,
      "Terminal=false",
      "X-GNOME-Autostart-enabled=true",
      "",
    ].join("\n"),
    "utf8",
  );
}

function escapeDesktopExec(value: string): string {
  return `"${value
    .replaceAll("\\", "\\\\")
    .replaceAll('"', '\\"')
    .replaceAll("`", "\\`")
    .replaceAll("$", "\\$")
    .replaceAll("%", "%%")}"`;
}

async function startCoordinatorIfNeeded(): Promise<void> {
  if (settings.coordinatorMode === "remote") {
    coordinatorUrl = normalizeHttpUrl(validateCoordinatorUrl(settings.remoteCoordinatorUrl));
    return;
  }
  if (await coordinatorIsReachable(LOCAL_DASHBOARD_COORDINATOR_URL)) {
    coordinatorUrl = LOCAL_DASHBOARD_COORDINATOR_URL;
    writeDesktopLog("local-coordinator-selected", { coordinatorUrl });
    return;
  }
  coordinator = await createCoordinator(
    {
      host: "127.0.0.1",
      port: LOCAL_DASHBOARD_COORDINATOR_PORT,
      databasePath: join(app.getPath("userData"), "mycellios.db"),
      requestTimeoutMs: 120_000,
      mobileAssetsPath: app.isPackaged
        ? join(process.resourcesPath, "mobile-dist")
        : join(app.getAppPath(), "mobile-dist"),
    },
    {
      logger: false,
      activationManagerFactory: ({ store, hub }) => new DynamicModelActivationManager({
        cwd: app.getAppPath(),
        snapshot: () => buildDesktopActivationSnapshot(store.listWorkers(), hub.connectedWorkerIds()),
        resolveManagedAgent: (nodeId, launch) => resolveDesktopTunnelAgent(store.listWorkers(), hub, nodeId, launch),
      }),
    },
  );
  await coordinator.app.listen({
    host: "127.0.0.1",
    port: LOCAL_DASHBOARD_COORDINATOR_PORT,
  });
  const address = coordinator.app.server.address() as AddressInfo;
  coordinatorUrl = `http://127.0.0.1:${address.port}`;
  writeDesktopLog("embedded-coordinator-started", { coordinatorUrl });
}

async function coordinatorIsReachable(baseUrl: string): Promise<boolean> {
  try {
    const response = await fetch(new URL("health", `${baseUrl}/`), {
      signal: AbortSignal.timeout(1_500),
      redirect: "error",
    });
    if (!response.ok) return false;
    const body = await response.json() as { status?: unknown };
    return body.status === "ok";
  } catch {
    return false;
  }
}

function normalizeHttpUrl(url: URL): string {
  const normalized = new URL(url);
  if (normalized.protocol === "ws:") normalized.protocol = "http:";
  if (normalized.protocol === "wss:") normalized.protocol = "https:";
  return normalized.toString().replace(/\/$/, "");
}

async function buildWorkerConfig(): Promise<WorkerConfig> {
  const hardware = await getHardware();
  const primary = hardware.gpus[0];
  const detectedBudget = primary
    ? primary.physicalVramMb + (primary.sharedMemoryMb ?? 0)
    : 0;
  if (detectedBudget > 0 && settings.offeredVramMb > detectedBudget) {
    throw new Error(
      `La cuota de ${settings.offeredVramMb} MB supera los ${detectedBudget} MB detectados.`,
    );
  }
  const adapter =
    settings.adapterMode === "local-model-runtime"
      ? {
          kind: "local-model-runtime" as const,
          model: settings.modelName,
          baseUrl: settings.adapterBaseUrl,
        }
      : {
          kind: "mock" as const,
          model: "mycellios-connectivity-check",
          tokensPerSecond: 20,
          ttftMs: 150,
          failureRate: 0,
        };
  return workerConfigSchema.parse({
    region: settings.region,
    capacityScope: "host",
    offeredVramMb: settings.offeredVramMb,
    limits: {
      maxConcurrency: 1,
      maxTemperatureC: 80,
      pauseWhenForeground: false,
    },
    adapter,
    deployment: {
      ...(settings.adapterMode === "local-model-runtime" ? { modelDigest: settings.modelDigest } : {}),
      contextLimit: 8_192,
    },
    llmfit: { enabled: false },
  });
}

async function startWorkerIfEnabled(): Promise<void> {
  if (!settings.contributionEnabled || worker) return;
  distributedExecutor ??= await createDesktopDistributedExecutor();
  const config = await buildWorkerConfig();
  const nextWorker = new WorkerAgent(config, {
    coordinatorUrl,
    ...(settings.coordinatorMode === "remote" && settings.remoteCoordinatorToken
      ? { networkToken: settings.remoteCoordinatorToken }
      : {}),
    reconnect: true,
    // The legacy connectivity option now means hardware-only standby. It
    // registers this physical PC but never advertises a fake model.
    advertiseDeployment: settings.adapterMode !== "connectivity-test",
    distributedExecutor,
    logger: {
      info: (message) => {
        console.info(`[agent] ${message}`);
        writeDesktopLog("worker-info", { message });
      },
      warn: (message) => {
        console.warn(`[agent] ${message}`);
        writeDesktopLog("worker-warning", { message });
      },
      error: (message) => {
        console.error(`[agent] ${message}`);
        writeDesktopLog("worker-error", { message });
      },
    },
  });
  worker = nextWorker;
  void nextWorker.start().catch((error: unknown) => {
    runtimeError = errorText(error);
    writeDesktopLog("worker-start-failed", { error: runtimeError });
    if (worker === nextWorker) worker = null;
  });
}

async function stopWorker(): Promise<void> {
  const activeWorker = worker;
  worker = null;
  if (activeWorker) await activeWorker.stop();
}

async function stopRuntime(): Promise<void> {
  await stopWorker();
  const activeCoordinator = coordinator;
  coordinator = null;
  if (activeCoordinator) await activeCoordinator.close();
  coordinatorUrl = "";
}

async function restartRuntime(): Promise<void> {
  runtimeError = null;
  await stopRuntime();
  await startCoordinatorIfNeeded();
  await startWorkerIfEnabled();
}

function getHardware(): Promise<HardwareProbe> {
  hardwarePromise ??= probeHardware();
  return hardwarePromise;
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (settings.coordinatorMode === "remote" && settings.remoteCoordinatorToken) {
    headers.set("authorization", `Bearer ${settings.remoteCoordinatorToken}`);
  }
  const response = await fetch(new URL(path, `${coordinatorUrl}/`), {
    ...init,
    headers,
    signal: AbortSignal.timeout(8_000),
    redirect: "error",
  });
  if (!response.ok) throw new Error(`The coordinator returned HTTP ${response.status}.`);
  return (await response.json()) as T;
}

async function readSnapshot(): Promise<DashboardSnapshot> {
  const localHardware = await getHardware();
  let connectionError: string | null = null;
  let health: DashboardSnapshot["health"] = null;
  let workers: DashboardWorker[] = [];
  let models: DashboardModel[] = [];
  let requestedModels: import("./contracts.js").RequestedModelCapacity[] = [];
  let jobs: DashboardJob[] = [];
  try {
    const [healthResult, publicResult] = await Promise.all([
      fetchJson<NonNullable<DashboardSnapshot["health"]>>("health"),
      fetchJson<{
        workers: DashboardWorker[];
        models: DashboardModel[];
        requestedModels: import("./contracts.js").RequestedModelCapacity[];
        jobs: DashboardJob[];
      }>("public/v1/snapshot"),
    ]);
    health = healthResult;
    workers = publicResult.workers;
    models = publicResult.models;
    requestedModels = publicResult.requestedModels;
    jobs = publicResult.jobs;
  } catch (error) {
    connectionError = errorText(error);
  }
  const localWorkerId = worker?.workerId ?? null;
  const localWorkerConnected = localWorkerId !== null
    && workers.some((candidate) => candidate.id === localWorkerId && candidate.connected);
  const contributionState: DashboardSnapshot["contribution"]["state"] =
    !settings.contributionEnabled
      ? "paused"
      : runtimeError
        ? "error"
        : localWorkerConnected
          ? "connected"
          : "connecting";
  return {
    capturedAt: new Date().toISOString(),
    coordinatorUrl,
    appVersion: app.getVersion(),
    platform: process.platform,
    connectionError,
    runtimeError,
    health,
    workers,
    models,
    requestedModels,
    jobs,
    localHardware,
    contribution: { state: contributionState, workerId: localWorkerId },
    settings,
    update: { ...updateStatus },
  };
}

async function sendChat(request: ChatRequest): Promise<ChatResponse> {
  const prompt = request.prompt.trim();
  if (!prompt) throw new Error("Escribe un mensaje antes de enviarlo.");
  const response = await fetchJson<{
    id: string;
    model: string;
    choices: Array<{ message: { content: string } }>;
    usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
    x_network: { route_class: string; affinity_hit: boolean; ttft_ms: number; active_ms: number };
  }>("v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: request.model,
      messages: [{ role: "user", content: prompt }],
      stream: false,
      max_tokens: Math.max(1, Math.min(2_048, request.maxTokens ?? 128)),
      temperature: 0,
      top_p: 1,
    }),
  });
  return {
    requestId: response.id,
    model: response.model,
    text: response.choices[0]?.message.content ?? "",
    promptTokens: response.usage.prompt_tokens,
    outputTokens: response.usage.completion_tokens,
    totalTokens: response.usage.total_tokens,
    routeClass: response.x_network.route_class,
    affinityHit: response.x_network.affinity_hit,
    ttftMs: response.x_network.ttft_ms,
    activeMs: response.x_network.active_ms,
  };
}

async function streamChat(request: ChatRequest, onUpdate: (update: ChatStreamUpdate) => void): Promise<ChatResponse> {
  const prompt = request.prompt.trim();
  if (!prompt) throw new Error("Escribe un mensaje antes de enviarlo.");
  const headers = new Headers({ accept: "text/event-stream", "content-type": "application/json" });
  if (settings.coordinatorMode === "remote" && settings.remoteCoordinatorToken) {
    headers.set("authorization", `Bearer ${settings.remoteCoordinatorToken}`);
  }
  const startedAt = Date.now();
  const response = await fetch(new URL("v1/chat/completions", `${coordinatorUrl}/`), {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: request.model,
      messages: [{ role: "user", content: prompt }],
      stream: true,
      max_tokens: Math.max(1, Math.min(2_048, request.maxTokens ?? 128)),
      temperature: 0,
      top_p: 1,
    }),
    signal: AbortSignal.timeout(3 * 60_000),
    redirect: "error",
  });
  return consumeChatCompletionStream(response, request.model, onUpdate, startedAt);
}

function registerIpc(): void {
  ipcMain.handle("dashboard:read", () => readSnapshot());
  ipcMain.handle("settings:save", async (_event, next: DesktopSettings) => {
    persistSettings(next);
    await restartRuntime();
    return readSnapshot();
  });
  ipcMain.handle("contribution:set", async (_event, enabled: boolean) => {
    persistSettings({ ...settings, contributionEnabled: Boolean(enabled) });
    if (enabled) await startWorkerIfEnabled();
    else if (worker) {
      const active = worker;
      worker = null;
      await active.stop();
    }
    return readSnapshot();
  });
  ipcMain.handle("chat:send", (_event, request: ChatRequest) => sendChat(request));
  ipcMain.handle("chat:stream", (event, streamId: string, request: ChatRequest) => streamChat(request, (update) => {
    if (!event.sender.isDestroyed()) event.sender.send("chat:stream:update", streamId, update);
  }));
  ipcMain.handle("workers:remove", async (_event, workerId: string) => {
    await fetchJson(`public/v1/workers/${encodeURIComponent(workerId)}`, { method: "DELETE" });
    return readSnapshot();
  });
  ipcMain.handle("workers:clear-offline", async () => {
    await fetchJson("public/v1/workers/clear-offline", { method: "POST" });
    return readSnapshot();
  });
  ipcMain.handle("models:search-hub", async (_event, query: string) => {
    const result = await fetchJson<{ data: HubCatalogModel[] }>(`public/v1/huggingface-models?q=${encodeURIComponent(query)}`);
    return result.data;
  });
  ipcMain.handle("models:request", async (_event, input: import("./contracts.js").RequestModelInput) => {
    await fetchJson("public/v1/requested-models", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(input),
    });
    return readSnapshot();
  });
  ipcMain.handle("models:remove-request", async (_event, modelId: string) => {
    await fetchJson(`public/v1/requested-models/${encodeURIComponent(modelId)}`, { method: "DELETE" });
    return readSnapshot();
  });
  ipcMain.handle("benchmarks:read", async () => {
    const result = await fetchJson<{ runs: import("../benchlab/types.js").BenchmarkRun[] }>("local/v1/benchmarks");
    return result.runs;
  });
  ipcMain.handle("benchmarks:run", async () => {
    const response = await fetch(new URL("local/v1/benchmarks/run", `${coordinatorUrl}/`), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{}",
      signal: AbortSignal.timeout(15 * 60_000),
      redirect: "error",
    });
    const body = await response.json() as {
      run?: import("../benchlab/types.js").BenchmarkRun;
      error?: { message?: string };
    };
    if (!response.ok || !body.run) throw new Error(body.error?.message ?? `The coordinator returned HTTP ${response.status}.`);
    return body.run;
  });
  ipcMain.handle("updates:check", () => checkForUpdates());
  ipcMain.handle("updates:install", () => {
    if (updateStatus.state !== "ready") {
      throw new Error("There is no downloaded update ready to install.");
    }
    isQuitting = true;
    autoUpdater.quitAndInstall();
  });
  ipcMain.handle("window:minimize", (event) => {
    BrowserWindow.fromWebContents(event.sender)?.minimize();
  });
  ipcMain.handle("window:toggle-maximize", (event) => {
    const window = BrowserWindow.fromWebContents(event.sender);
    if (!window) return false;
    if (window.isMaximized()) window.unmaximize();
    else window.maximize();
    return window.isMaximized();
  });
  ipcMain.handle("window:close", (event) => {
    BrowserWindow.fromWebContents(event.sender)?.close();
  });
}

function createWindow(): void {
  const icon = resourcePath("build", "icons", "app-icon-v2.png");
  mainWindow = new BrowserWindow({
    width: 1_520,
    height: 920,
    minWidth: 1_080,
    minHeight: 680,
    show: false,
    backgroundColor: "#050a0c",
    title: "mycellios",
    icon,
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "hidden",
    trafficLightPosition: { x: 18, y: 18 },
    webPreferences: {
      preload: join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
    },
  });

  mainWindow.once("ready-to-show", () => {
    const launchHidden =
      process.argv.includes("--hidden") ||
      (process.platform === "darwin" && app.getLoginItemSettings().wasOpenedAsHidden);
    if (!launchHidden) mainWindow?.show();
  });
  mainWindow.webContents.on("console-message", (_event, level, message, line, sourceId) => {
    if (level >= 2) writeDesktopLog("renderer-console", { level, message, line, sourceId });
  });
  mainWindow.webContents.on("did-fail-load", (_event, errorCode, description, url) => {
    writeDesktopLog("renderer-load-failed", { errorCode, description, url });
  });
  mainWindow.webContents.on("preload-error", (_event, preloadPath, error) => {
    writeDesktopLog("preload-error", { preloadPath, error: error.message });
  });
  mainWindow.webContents.on("render-process-gone", (_event, details) => {
    writeDesktopLog("renderer-gone", details);
  });
  mainWindow.on("close", (event) => {
    if (!isQuitting && settings.closeToTray) {
      event.preventDefault();
      mainWindow?.hide();
    }
  });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) void shell.openExternal(url);
    return { action: "deny" };
  });
  mainWindow.webContents.on("will-navigate", (event, url) => {
    const developmentUrl = MAIN_WINDOW_VITE_DEV_SERVER_URL;
    if (!developmentUrl || !url.startsWith(developmentUrl)) event.preventDefault();
  });

  if (MAIN_WINDOW_VITE_DEV_SERVER_URL) {
    void mainWindow.loadURL(MAIN_WINDOW_VITE_DEV_SERVER_URL);
  } else {
    void mainWindow.loadFile(
      join(__dirname, "..", "renderer", MAIN_WINDOW_VITE_NAME, "index.html"),
    );
  }
}

function createTray(): void {
  try {
    createTrayUnsafe();
  } catch (error) {
    tray?.destroy();
    tray = null;
    writeDesktopLog("tray-error", { error: errorText(error) });
  }
}

function createTrayUnsafe(): void {
  const trayImage = nativeImage.createFromPath(resourcePath("build", "icons", "app-icon-v2.png"));
  tray?.destroy();
  tray = new Tray(trayImage.resize({ width: 22, height: 22 }));
  tray.setToolTip("mycellios");
  tray.setContextMenu(
    Menu.buildFromTemplate([
      { label: "Open mycellios", click: () => mainWindow?.show() },
      {
        label: settings.contributionEnabled ? "Pause contribution" : "Enable contribution",
        click: () => {
          void (async () => {
            persistSettings({ ...settings, contributionEnabled: !settings.contributionEnabled });
            await restartRuntime();
            createTray();
          })();
        },
      },
      { type: "separator" },
      {
        label: "Salir",
        click: () => {
          isQuitting = true;
          app.quit();
        },
      },
    ]),
  );
  tray.on("double-click", () => mainWindow?.show());
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function createDesktopDistributedExecutor() {
  const runtimeRoot = await ensureDistributionRuntime();
  const pythonExecutable = distributionPythonExecutable(runtimeRoot);
  if (!existsSync(pythonExecutable)) {
    throw new Error("The packaged shard runtime is missing. Reinstall mycellios to contribute this device.");
  }
  const pythonPath = resourcePath("python");
  const hfHome = join(app.getPath("userData"), "model-shards");
  mkdirSync(hfHome, { recursive: true });
  const nodeId = persistentDistributedNodeId();
  return {
    nodeId,
    stageHost: preferredLanAddress(),
    stagePort: 9_850,
    launchAgent: new LocalProcessAgent({
      id: `desktop-shard-executor:${nodeId}`,
      cwd: app.getAppPath(),
      env: {
        PYTHONPATH: pythonPath,
        HF_HOME: hfHome,
        TOKENIZERS_PARALLELISM: "false",
        PATH: [dirname(pythonExecutable), process.env.PATH].filter(Boolean).join(delimiter),
      },
      maxOutputBytesPerStream: 256 * 1024,
    }),
  };
}

function distributionPythonExecutable(root = app.isPackaged
  ? join(app.getPath("userData"), "distribution-runtime-v1")
  : join(app.getAppPath(), "runtime", "distribution-venv")): string {
  return process.platform === "win32"
    ? join(root, "Scripts", "python.exe")
    : join(root, "bin", "python3");
}

function ensureDistributionRuntime(): Promise<string> {
  distributionRuntimePromise ??= ensureDistributionRuntimeOnce();
  return distributionRuntimePromise;
}

async function ensureDistributionRuntimeOnce(): Promise<string> {
  if (!app.isPackaged) return join(app.getAppPath(), "runtime", "distribution-venv");
  const root = join(app.getPath("userData"), "distribution-runtime-v1");
  if (existsSync(distributionPythonExecutable(root))) return root;
  const archive = resourcePath("distribution-runtime.tar.gz");
  if (!existsSync(archive)) throw new Error("The packaged shard runtime archive is missing. Reinstall mycellios.");
  mkdirSync(root, { recursive: true });
  await runProcess("tar", ["-xzf", archive, "-C", root]);
  if (!existsSync(distributionPythonExecutable(root))) {
    throw new Error("The shard runtime could not be extracted. Reinstall mycellios.");
  }
  return root;
}

function runProcess(executable: string, args: readonly string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(executable, [...args], { shell: false, windowsHide: true, stdio: ["ignore", "ignore", "pipe"] });
    let stderr = "";
    child.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString("utf8"); });
    child.once("error", reject);
    child.once("close", (code) => code === 0 ? resolve() : reject(new Error(stderr.trim() || `${executable} exited with ${code}`)));
  });
}

function persistentDistributedNodeId(): string {
  const path = join(app.getPath("userData"), "distributed-node-id.txt");
  try {
    const existing = readFileSync(path, "utf8").trim();
    if (/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(existing)) return existing;
  } catch {
    // First launch creates a stable executor identity below.
  }
  const nodeId = `desktop-${randomUUID()}`;
  writeFileSync(path, `${nodeId}\n`, "utf8");
  return nodeId;
}

function preferredLanAddress(): string {
  for (const addresses of Object.values(networkInterfaces())) {
    for (const address of addresses ?? []) {
      if (address.family === "IPv4" && !address.internal && !address.address.startsWith("169.254.")) {
        return address.address;
      }
    }
  }
  return "127.0.0.1";
}

function buildDesktopActivationSnapshot(
  workers: readonly StoredWorker[],
  connectedWorkerIds: ReadonlySet<string>,
): import("../coordinator/model-activation-manager.js").DynamicActivationSnapshot {
  const executors = workers
    .filter((worker) => connectedWorkerIds.has(worker.id) && worker.capabilities.distributedExecutor)
    .map((worker) => ({ worker, executor: worker.capabilities.distributedExecutor! }))
    .filter((entry, index, all) => all.findIndex((candidate) => candidate.executor.nodeId === entry.executor.nodeId) === index);
  const capacityNodes = executors.map(({ worker, executor }) => ({
    id: executor.nodeId,
    availableVramMiB: worker.capabilities.gpus.reduce((sum, gpu) => sum + gpu.freeOfferedVramMb, 0),
  }));
  if (executors.length < 2) return { capacityNodes, config: null };
  const nodes = executors.map(({ worker, executor }) => {
    const memoryMiB = worker.capabilities.gpus.reduce((sum, gpu) => sum + gpu.offeredVramMb, 0);
    const measuredPower = worker.capabilities.gpus.reduce((sum, gpu) => sum + (gpu.powerW ?? 0), 0);
    return {
      id: executor.nodeId,
      region: worker.capabilities.region,
      endpoint: { host: executor.stageHost, port: executor.stagePort },
      memoryMiB,
      reserveMiB: Math.min(256, Math.max(0, memoryMiB - 1)),
      decodeScale: 1,
      prefillScale: 1,
      codecScale: 1,
      powerWatts: measuredPower > 0 ? measuredPower : 1,
      availability: Math.max(0.01, Math.min(1, worker.reliability)),
      agent: { kind: "managed" as const },
    };
  });
  const links = executors.flatMap((from) => executors
    .filter((to) => to.executor.nodeId !== from.executor.nodeId)
    .map((to) => ({
      from: from.executor.nodeId,
      to: to.executor.nodeId,
      oneWayLatencyMs: Math.max(0.1, (from.worker.capabilities.network.coordinatorRttMs + to.worker.capabilities.network.coordinatorRttMs) / 2),
      jitterP95Ms: 0,
      bandwidthMbps: Math.max(1, Math.min(from.worker.capabilities.network.uplinkMbps, to.worker.capabilities.network.downlinkMbps)),
      lossRate: 0,
      availability: Math.max(0.01, Math.min(from.worker.reliability, to.worker.reliability)),
    })));
  const rootHost = nodes[0]!.endpoint.host;
  const config = parseAutoDistributionConfig({
    schema: "gdlp-auto-distribute/1",
    model: { source: "HuggingFaceTB/SmolLM2-135M-Instruct", revision: null, publicName: "pending-model" },
    nodes,
    links,
    distribution: { minimumStages: 2, maximumStages: Math.min(8, nodes.length), allowLossyActivation: false },
    workload: { promptTokens: 128, outputTokens: 128, contextTokens: 4_096, concurrentSequences: 1, minRouteAvailability: 0.9, batchWindowMs: 2, p95: true },
    runtime: {
      pythonExecutable: distributionPythonExecutable(),
      stagePythonExecutable: "python",
      pythonPath: resourcePath("python"),
      hfHome: join(app.getPath("userData"), "coordinator-model-cache"),
      apiEndpoint: { host: "0.0.0.0", port: 9_860 },
      apiAdvertiseHost: rootHost,
      returnEndpoint: { host: rootHost, port: 9_861 },
      returnBindHost: "0.0.0.0",
      threadsPerStage: 1,
      connectTimeoutSeconds: 300,
      readinessTimeoutMs: 600_000,
      maxOutputTokens: 2_048,
    },
    canary: { prompt: "Reply with only OK. /no_think", maxTokens: 16, timeoutMs: 300_000 },
    coordinator: { url: LOCAL_DASHBOARD_COORDINATOR_URL, region: settings.region, maxConcurrency: 1 },
  });
  return { capacityNodes, config };
}

function resolveDesktopTunnelAgent(
  workers: readonly StoredWorker[],
  hub: WorkerHub,
  nodeId: string,
  launch: PythonPipelineLaunchDescription,
): LaunchAgent | undefined {
  const worker = workers.find((candidate) => candidate.capabilities.distributedExecutor?.nodeId === nodeId);
  return worker ? new WorkerTunnelLaunchAgent(hub, worker.id, nodeId, launch) : undefined;
}

const hasSingleInstanceLock = app.requestSingleInstanceLock();
if (!hasSingleInstanceLock) app.quit();
else {
  app.on("second-instance", () => {
    if (!mainWindow) return;
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
  });
}

app.whenReady().then(async () => {
  settings = loadSettings();
  registerIpc();
  await restartRuntime().catch((error: unknown) => {
    runtimeError = errorText(error);
  });
  createWindow();
  createTray();
  configureAutomaticUpdates();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
  else mainWindow?.show();
});

app.on("before-quit", () => {
  isQuitting = true;
});

app.on("window-all-closed", () => {
  if (process.platform === "darwin" || settings.closeToTray) return;
  app.quit();
});

app.on("will-quit", () => {
  if (updateCheckTimer) clearInterval(updateCheckTimer);
  void stopRuntime();
});
