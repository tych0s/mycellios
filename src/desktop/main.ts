import { appendFileSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { AddressInfo } from "node:net";
import {
  app,
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
import { WorkerAgent, validateCoordinatorUrl } from "../worker/agent.js";
import { probeHardware, type HardwareProbe } from "../worker/hardware.js";
import type {
  ChatRequest,
  ChatResponse,
  DashboardModel,
  DashboardSnapshot,
  DashboardWorker,
  DesktopSettings,
} from "./contracts.js";

if (started) app.quit();

app.setName("mycellios");

const DEFAULT_SETTINGS: DesktopSettings = {
  coordinatorMode: "local",
  remoteCoordinatorUrl: "https://network.mycellios.app",
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

let mainWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let coordinator: CoordinatorRuntime | null = null;
let coordinatorUrl = "";
let worker: WorkerAgent | null = null;
let hardwarePromise: Promise<HardwareProbe> | null = null;
let settings: DesktopSettings = DEFAULT_SETTINGS;
let isQuitting = false;
let runtimeError: string | null = null;

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

function loadSettings(): DesktopSettings {
  try {
    const stored = JSON.parse(readFileSync(settingsPath(), "utf8")) as Partial<DesktopSettings>;
    return sanitizeSettings({ ...DEFAULT_SETTINGS, ...stored });
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

function sanitizeSettings(input: DesktopSettings): DesktopSettings {
  const offeredVramMb = Number.isFinite(input.offeredVramMb)
    ? Math.max(512, Math.min(262_144, Math.round(input.offeredVramMb)))
    : DEFAULT_SETTINGS.offeredVramMb;
  const remoteCoordinatorUrl = input.remoteCoordinatorUrl.trim();
  if (input.coordinatorMode === "remote") validateCoordinatorUrl(remoteCoordinatorUrl);
  if (input.adapterMode === "local-model-runtime" && !input.modelDigest.trim()) {
    throw new Error("local model runtime necesita un digest de modelo fijado antes de aportar recursos.");
  }
  return {
    coordinatorMode: input.coordinatorMode === "remote" ? "remote" : "local",
    remoteCoordinatorUrl,
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
  coordinator = await createCoordinator(
    {
      host: "127.0.0.1",
      port: 0,
      databasePath: join(app.getPath("userData"), "mycellios.db"),
      requestTimeoutMs: 120_000,
      mobileAssetsPath: app.isPackaged
        ? join(process.resourcesPath, "mobile-dist")
        : join(app.getAppPath(), "mobile-dist"),
    },
    { logger: false },
  );
  await coordinator.app.listen({ host: "127.0.0.1", port: 0 });
  const address = coordinator.app.server.address() as AddressInfo;
  coordinatorUrl = `http://127.0.0.1:${address.port}`;
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
  const config = await buildWorkerConfig();
  const nextWorker = new WorkerAgent(config, {
    coordinatorUrl,
    reconnect: true,
    logger: {
      info: (message) => console.info(`[agent] ${message}`),
      warn: (message) => console.warn(`[agent] ${message}`),
      error: (message) => console.error(`[agent] ${message}`),
    },
  });
  worker = nextWorker;
  void nextWorker.start().catch((error: unknown) => {
    runtimeError = errorText(error);
    if (worker === nextWorker) worker = null;
  });
}

async function stopRuntime(): Promise<void> {
  const activeWorker = worker;
  worker = null;
  if (activeWorker) await activeWorker.stop();
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
  const response = await fetch(new URL(path, `${coordinatorUrl}/`), {
    ...init,
    signal: AbortSignal.timeout(8_000),
    redirect: "error",
  });
  if (!response.ok) throw new Error(`El coordinador respondió HTTP ${response.status}.`);
  return (await response.json()) as T;
}

async function readSnapshot(): Promise<DashboardSnapshot> {
  const localHardware = await getHardware();
  let connectionError: string | null = null;
  let health: DashboardSnapshot["health"] = null;
  let workers: DashboardWorker[] = [];
  let models: DashboardModel[] = [];
  try {
    const [healthResult, workerResult, modelResult] = await Promise.all([
      fetchJson<NonNullable<DashboardSnapshot["health"]>>("health"),
      fetchJson<{ data: DashboardWorker[] }>("internal/v1/workers"),
      fetchJson<{ data: Array<{ id: string; x_replicas: number; x_pipelines: number }> }>(
        "v1/models",
      ),
    ]);
    health = healthResult;
    workers = workerResult.data;
    models = modelResult.data.map((model) => ({
      id: model.id,
      replicas: model.x_replicas,
      pipelines: model.x_pipelines,
    }));
  } catch (error) {
    connectionError = errorText(error);
  }
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
    localHardware,
    settings,
  };
}

async function sendChat(request: ChatRequest): Promise<ChatResponse> {
  const prompt = request.prompt.trim();
  if (!prompt) throw new Error("Escribe un mensaje antes de enviarlo.");
  const response = await fetchJson<{
    id: string;
    model: string;
    choices: Array<{ message: { content: string } }>;
    usage: { total_tokens: number };
    x_network: { route_class: string };
  }>("v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: request.model,
      messages: [{ role: "user", content: prompt }],
      stream: false,
      max_tokens: Math.max(1, Math.min(2_048, request.maxTokens ?? 256)),
    }),
  });
  return {
    requestId: response.id,
    model: response.model,
    text: response.choices[0]?.message.content ?? "",
    totalTokens: response.usage.total_tokens,
    routeClass: response.x_network.route_class,
  };
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
}

function createWindow(): void {
  const icon = resourcePath("build", "icons", "icon.png");
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
  const trayImage = nativeImage.createFromPath(resourcePath("build", "icons", "icon.png"));
  tray?.destroy();
  tray = new Tray(trayImage.resize({ width: 22, height: 22 }));
  tray.setToolTip("mycellios");
  tray.setContextMenu(
    Menu.buildFromTemplate([
      { label: "Abrir mycellios", click: () => mainWindow?.show() },
      {
        label: settings.contributionEnabled ? "Pausar contribución" : "Activar contribución",
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
  void stopRuntime();
});
