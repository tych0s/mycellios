import { randomUUID } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { delimiter, dirname, join } from "node:path";
import { cpus, release } from "node:os";
import { spawn } from "node:child_process";
import type { AddressInfo } from "node:net";
import {
  app,
  autoUpdater,
  BrowserWindow,
  ipcMain,
  Menu,
  nativeImage,
  safeStorage,
  shell,
  Tray,
} from "electron";
import started from "electron-squirrel-startup";
import { workerConfigSchema, type WorkerConfig } from "../contracts/schemas.js";
import type { CoordinatorRuntime } from "../coordinator/server.js";
import { createCoordinator } from "../coordinator/server.js";
import { DynamicModelActivationManager } from "../coordinator/model-activation-manager.js";
import { parseAutoDistributionConfig } from "../distribution/auto-distribute.js";
import {
  LocalProcessAgent,
  type LaunchAgent,
  type LaunchAgentStartRequest,
  type LaunchProcessExit,
  type LaunchProcessHandle,
} from "../distribution/launch-supervisor.js";
import type { PythonPipelineLaunchDescription } from "../distribution/python-launcher.js";
import { WorkerTunnelLaunchAgent } from "../distribution/worker-tunnel-launch-agent.js";
import type { StoredWorker } from "../storage/store.js";
import type { WorkerHub } from "../coordinator/worker-hub.js";
import { WorkerAgent, validateCoordinatorUrl } from "../worker/agent.js";
import {
  probeHardware,
  type HardwareProbe,
  type VerifiedGpuRuntimeEvidence,
} from "../worker/hardware.js";
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
import type {
  HubCatalogPage,
  HubCatalogSearchInput,
  WorkerAcceleratorDiagnostics,
} from "../contracts/types.js";
import {
  applyAccelerationProgress,
  appendAccelerationLog,
  beginAccelerationPreparation,
  createInitialAccelerationStatus,
  gpuPreparationIsContinuing,
  selectImmediateRuntime,
} from "./acceleration-progress.js";
import {
  applyVerifiedAccelerationUsage,
  gpuPreparationRetryDelayMs,
  readVerifiedAccelerationUsage,
} from "./acceleration-evidence.js";
import { consumeChatCompletionStreamWithRecovery } from "./chat-stream.js";
import { desktopExecutorPolicy, normalizeComputeMode } from "./compute-mode.js";
import {
  selectDesktopHardwareGpu,
  selectWorkerCapacityHardware,
  type DesktopHardwareGpu,
} from "./hardware-selection.js";
import {
  prepareAcceleratorRuntime,
  selectAcceleratorPack,
  verifyPortableRuntimeInstallation,
  type AcceleratorProgressEvent,
  type AcceleratorRuntimeResult,
} from "./accelerator-runtime.js";
import { buildWorkerAccelerationDiagnostics } from "./acceleration-diagnostics.js";
import {
  AUTOMATIC_UPDATE_CHECK_INTERVAL_MS,
  AUTOMATIC_UPDATE_GRACE_MS,
  AUTOMATIC_UPDATE_IDLE_RECHECK_MS,
  automaticUpdateRetryDelayMs,
  canInstallAutomaticUpdate,
  summarizeAutomaticUpdateError,
} from "./update-recovery.js";
import { SingleFlight } from "./single-flight.js";

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
  computeMode: "automatic",
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
const LEGACY_PUBLIC_COORDINATOR_URLS = new Set([
  "https://www.mycellios.com",
  "https://mycellios.com",
  "https://network.mycellios.app",
]);

let mainWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let coordinator: CoordinatorRuntime | null = null;
let coordinatorUrl = "";
let worker: WorkerAgent | null = null;
const workerStartFlight = new SingleFlight();
let distributedExecutor: Awaited<ReturnType<typeof createDesktopDistributedExecutor>> | null = null;
let distributionRuntimePromise: Promise<string> | null = null;
let cpuRuntimePromise: Promise<AcceleratorRuntimeResult> | null = null;
let acceleratorRuntimePromise: Promise<AcceleratorRuntimeResult> | null = null;
let resolvedAcceleratorRuntime: AcceleratorRuntimeResult | null = null;
let acceleratorRuntimeRoot: string | null = null;
let acceleratorRetryTimer: NodeJS.Timeout | null = null;
let acceleratorRetryAttempt = 0;
let acceleratorRetryIssueCode: string | null = null;
let acceleratorNextRetryAt: string | null = null;
// Keep expensive-failure budgets per code until a GPU really passes verification.
// Alternating through a transient network error must not reset the expanding
// backoff for another multi-gigabyte repair attempt.
const acceleratorRetryIssueAttempts = new Map<string, number>();
let accelerationDiagnosticsPublishTimer: NodeJS.Timeout | null = null;
const activeDistributedStages = new Set<LaunchProcessHandle>();
let hardwarePromise: Promise<HardwareProbe> | null = null;
let settings: DesktopSettings = DEFAULT_SETTINGS;
let modelAdminToken = "";
let isQuitting = false;
let runtimeError: string | null = null;
let accelerationStatus: DashboardSnapshot["acceleration"] = createInitialAccelerationStatus();
let updateCheckTimer: NodeJS.Timeout | null = null;
let updateRetryTimer: NodeJS.Timeout | null = null;
let updateRetryAttempt = 0;
let updateCheckInFlight = false;
let automaticUpdateInstallTimer: NodeJS.Timeout | null = null;
let automaticUpdateInstallInFlight = false;
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

function desktopIconPath(...segments: string[]): string {
  return app.isPackaged
    ? resourcePath("icons", ...segments)
    : resourcePath("build", "icons", ...segments);
}

function settingsPath(): string {
  return join(app.getPath("userData"), "desktop-settings.json");
}

function modelAdminTokenPath(): string {
  return join(app.getPath("userData"), "model-admin-token.enc");
}

function loadModelAdminToken(): string {
  if (!safeStorage.isEncryptionAvailable() || !existsSync(modelAdminTokenPath())) return "";
  try {
    return safeStorage.decryptString(readFileSync(modelAdminTokenPath())).trim();
  } catch (error) {
    writeDesktopLog("model-admin-token-load-failed", { error: errorText(error) });
    return "";
  }
}

function persistModelAdminToken(token: string): void {
  const next = token.trim();
  if (!next) return;
  if (!safeStorage.isEncryptionAvailable()) {
    throw new Error("Secure operating-system storage is unavailable. The administrator token was not saved.");
  }
  const target = modelAdminTokenPath();
  const temporary = `${target}.tmp`;
  writeFileSync(temporary, safeStorage.encryptString(next));
  renameSync(temporary, target);
  modelAdminToken = next;
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
    if (updateStatus.state === "ready") return;
    setUpdateStatus({ state: "checking", message: "Checking for a new version…" });
  });
  autoUpdater.on("update-available", () => {
    resetAutomaticUpdateRetry();
    if (updateStatus.state === "ready") return;
    setUpdateStatus({
      state: "downloading",
      message: "A new version is downloading in the background…",
    });
  });
  autoUpdater.on("update-not-available", () => {
    resetAutomaticUpdateRetry();
    if (updateStatus.state === "ready") return;
    setUpdateStatus({
      state: "up-to-date",
      availableVersion: null,
      message: "mycellios is up to date.",
      checkedAt: new Date().toISOString(),
    });
  });
  autoUpdater.on("update-downloaded", (_event, _releaseNotes, releaseName) => {
    resetAutomaticUpdateRetry();
    setUpdateStatus({
      state: "ready",
      availableVersion: releaseName || null,
      message: "Update ready. mycellios will restart automatically as soon as active work is idle.",
      checkedAt: new Date().toISOString(),
    });
    scheduleAutomaticUpdateInstall(AUTOMATIC_UPDATE_GRACE_MS);
  });
  autoUpdater.on("before-quit-for-update", () => {
    isQuitting = true;
  });
  autoUpdater.on("error", (error) => {
    if (updateStatus.state === "ready") return;
    const summary = summarizeAutomaticUpdateError(error.message);
    setUpdateStatus({
      state: "error",
      message: `Could not check for updates: ${summary}`,
      checkedAt: new Date().toISOString(),
    });
    scheduleAutomaticUpdateRetry(summary);
  });

  // Squirrel holds a file lock for a few seconds after first install.
  const initialDelayMs = process.argv.includes("--squirrel-firstrun") ? 15_000 : 10_000;
  setTimeout(() => void checkForUpdates(), initialDelayMs).unref();
  updateCheckTimer = setInterval(() => void checkForUpdates(), AUTOMATIC_UPDATE_CHECK_INTERVAL_MS);
  updateCheckTimer.unref();
}

async function checkForUpdates(): Promise<DesktopUpdateStatus> {
  if (!app.isPackaged || process.platform !== "win32" || updateCheckInFlight || updateStatus.state === "ready") {
    return updateStatus;
  }
  updateCheckInFlight = true;
  try {
    await autoUpdater.checkForUpdates();
  } catch (error) {
    const summary = summarizeAutomaticUpdateError(errorText(error));
    setUpdateStatus({
      state: "error",
      message: `Could not check for updates: ${summary}`,
      checkedAt: new Date().toISOString(),
    });
    scheduleAutomaticUpdateRetry(summary);
  } finally {
    updateCheckInFlight = false;
  }
  return updateStatus;
}

function resetAutomaticUpdateRetry(): void {
  updateRetryAttempt = 0;
  if (!updateRetryTimer) return;
  clearTimeout(updateRetryTimer);
  updateRetryTimer = null;
}

function scheduleAutomaticUpdateRetry(reason: string): void {
  if (
    process.platform !== "win32"
    || !app.isPackaged
    || updateStatus.state === "ready"
    || isQuitting
    || updateRetryTimer
  ) return;
  const attempt = updateRetryAttempt;
  const delayMs = automaticUpdateRetryDelayMs(attempt);
  updateRetryAttempt += 1;
  writeDesktopLog("automatic-update-retry-scheduled", {
    attempt: attempt + 1,
    delayMs,
    reason,
  });
  setUpdateStatus({
    message: `Update check will retry automatically in ${Math.ceil(delayMs / 60_000)} minute${delayMs > 60_000 ? "s" : ""}. ${reason}`,
  });
  updateRetryTimer = setTimeout(() => {
    updateRetryTimer = null;
    void checkForUpdates();
  }, delayMs);
  updateRetryTimer.unref();
}

function clearAutomaticUpdateInstallTimer(): void {
  if (!automaticUpdateInstallTimer) return;
  clearTimeout(automaticUpdateInstallTimer);
  automaticUpdateInstallTimer = null;
}

function scheduleAutomaticUpdateInstall(delayMs = AUTOMATIC_UPDATE_IDLE_RECHECK_MS): void {
  if (
    process.platform !== "win32"
    || !app.isPackaged
    || updateStatus.state !== "ready"
    || isQuitting
    || automaticUpdateInstallTimer
    || automaticUpdateInstallInFlight
  ) return;
  automaticUpdateInstallTimer = setTimeout(() => {
    automaticUpdateInstallTimer = null;
    void installDownloadedUpdate(false, "automatic-idle-repair");
  }, delayMs);
  automaticUpdateInstallTimer.unref();
}

async function installDownloadedUpdate(force: boolean, reason: string): Promise<void> {
  if (updateStatus.state !== "ready" || automaticUpdateInstallInFlight || isQuitting) return;
  const installable = canInstallAutomaticUpdate({
    updateReady: true,
    quitting: isQuitting,
    activeJobs: worker?.activeJobCount ?? 0,
    activeStages: activeDistributedStages.size,
  });
  if (!force && !installable) {
    setUpdateStatus({
      message: "Update ready. Waiting for active inference to finish before the automatic restart.",
    });
    scheduleAutomaticUpdateInstall();
    return;
  }

  clearAutomaticUpdateInstallTimer();
  automaticUpdateInstallInFlight = true;
  writeDesktopLog("automatic-update-installing", {
    reason,
    activeJobs: worker?.activeJobCount ?? 0,
    activeStages: activeDistributedStages.size,
    availableVersion: updateStatus.availableVersion,
  });
  isQuitting = true;
  try {
    await stopRuntime();
    autoUpdater.quitAndInstall();
  } catch (error) {
    isQuitting = false;
    automaticUpdateInstallInFlight = false;
    setUpdateStatus({
      state: "error",
      message: `Could not install the downloaded update automatically: ${errorText(error)}`,
      checkedAt: new Date().toISOString(),
    });
    await restartRuntime().catch((restartError: unknown) => {
      runtimeError = errorText(restartError);
      writeDesktopLog("runtime-restart-after-update-failed", { error: runtimeError });
    });
  }
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
    computeMode: normalizeComputeMode(input.computeMode),
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
      landingAssetsPath: app.isPackaged
        ? join(process.resourcesPath, "landing-dist")
        : join(app.getAppPath(), "landing-dist"),
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
    const body = await response.json() as {
      status?: unknown;
      features?: { distributedActivation?: unknown };
    };
    return body.status === "ok" && body.features?.distributedActivation === true;
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

async function buildWorkerConfig(
  primary: DesktopHardwareGpu | undefined,
): Promise<WorkerConfig> {
  const detectedBudget = primary
    ? primary.physicalVramMb + (primary.sharedMemoryMb ?? 0)
    : 0;
  // A fresh install must start on smaller and unified-memory devices without
  // asking the user to understand VRAM first. The stored preference remains
  // untouched and is capped only for the adapter actually advertised.
  // Keep the user's upper limit in config and let WorkerAgent clamp the live
  // capability to verified CPU/GPU memory. This allows the same connection to
  // grow from CPU RAM to GPU VRAM after the background physical probe passes.
  const offeredVramMb = primary?.id === "cpu-memory" || detectedBudget < 512
    ? settings.offeredVramMb
    : Math.min(settings.offeredVramMb, detectedBudget);
  if (offeredVramMb !== settings.offeredVramMb) {
    writeDesktopLog("worker-vram-auto-capped", {
      configuredMb: settings.offeredVramMb,
      offeredMb: offeredVramMb,
      gpu: primary?.model,
    });
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
    offeredVramMb,
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
  await workerStartFlight.run(initializeWorker);
}

async function initializeWorker(): Promise<void> {
  if (!settings.contributionEnabled || worker || isQuitting) return;
  writeDesktopLog("worker-start-requested", { coordinatorUrl, contributionEnabled: settings.contributionEnabled });
  try {
    distributedExecutor ??= await createDesktopDistributedExecutor();
    writeDesktopLog("distributed-executor-ready", {
      nodeId: distributedExecutor.nodeId,
      stageHost: distributedExecutor.stageHost,
      stagePort: distributedExecutor.stagePort,
    });
    if (settings.computeMode !== "cpu-only" && acceleratorRuntimeRoot) {
      startDesktopAcceleratorPreparation(acceleratorRuntimeRoot);
    }
  } catch (error) {
    writeDesktopLog("distributed-executor-failed", { error: errorText(error) });
    throw error;
  }
  let config: WorkerConfig;
  let preferredHardwareGpu: DesktopHardwareGpu | undefined;
  let capacityHardwareGpu: DesktopHardwareGpu | undefined;
  const verifiedGpuRuntime = currentVerifiedGpuRuntime();
  try {
    const hardware = await getHardware();
    preferredHardwareGpu = selectDesktopHardwareGpu(hardware.gpus, desktopHardwareSelectionInput());
    capacityHardwareGpu = selectWorkerCapacityHardware(
      hardware,
      preferredHardwareGpu,
      cpus()[0]?.model,
      verifiedGpuRuntime,
    );
    config = await buildWorkerConfig(capacityHardwareGpu);
  } catch (error) {
    writeDesktopLog("worker-config-failed", { error: errorText(error) });
    throw error;
  }
  const nextWorker = new WorkerAgent(config, {
    coordinatorUrl,
    agentVersion: app.getVersion(),
    ...(settings.coordinatorMode === "remote" && settings.remoteCoordinatorToken
      ? { networkToken: settings.remoteCoordinatorToken }
      : {}),
    reconnect: true,
    // The legacy connectivity option now means hardware-only standby. It
    // registers this physical PC but never advertises a fake model.
    advertiseDeployment: settings.adapterMode !== "connectivity-test",
    ...(preferredHardwareGpu
      ? { preferredHardwareGpu: { id: preferredHardwareGpu.id, vendor: preferredHardwareGpu.vendor, model: preferredHardwareGpu.model } }
      : {}),
    ...(verifiedGpuRuntime ? { verifiedGpuRuntime } : {}),
    distributedExecutor: {
      ...distributedExecutor,
      ...currentDesktopExecutorPolicy(),
      acceleration: currentAccelerationDiagnostics(),
    },
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
  // Close the small race where the background physical probe can finish
  // between hardware selection and WorkerAgent construction. The method
  // updates constructor state synchronously while capabilities are still null.
  await nextWorker.refreshRuntimeCapacity(
    currentVerifiedGpuRuntime(),
    currentDesktopExecutorPolicy(),
  );
  void nextWorker.start().catch((error: unknown) => {
    runtimeError = errorText(error);
    writeDesktopLog("worker-start-failed", { error: runtimeError });
    if (worker === nextWorker) worker = null;
  });
}

async function stopWorker(): Promise<void> {
  await workerStartFlight.wait().catch(() => undefined);
  const activeWorker = worker;
  worker = null;
  if (activeWorker) await activeWorker.stop();
}

async function stopRuntime(): Promise<void> {
  clearAcceleratorRetryTimer();
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
  if (settings.coordinatorMode === "remote" && settings.remoteCoordinatorToken && !headers.has("authorization")) {
    headers.set("authorization", `Bearer ${settings.remoteCoordinatorToken}`);
  }
  const response = await fetch(new URL(path, `${coordinatorUrl}/`), {
    ...init,
    headers,
    signal: AbortSignal.timeout(8_000),
    redirect: "error",
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { error?: { code?: string; message?: string } } | null;
    const message = body?.error?.message
      ?? (response.status === 401
        ? "The network administrator token is missing or invalid."
        : `The coordinator returned HTTP ${response.status}.`);
    throw new Error(message);
  }
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
  accelerationStatus = applyVerifiedAccelerationUsage(
    accelerationStatus,
    readVerifiedAccelerationUsage({
      workers,
      runtimeNodeId: distributedExecutor?.nodeId ?? null,
      localWorkerId,
      contributionConnected: contributionState === "connected",
    }),
  );
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
    acceleration: { ...accelerationStatus },
    modelAdminAuthorization: {
      configured: Boolean(modelAdminToken),
      encrypted: Boolean(modelAdminToken) && safeStorage.isEncryptionAvailable(),
    },
    settings,
    update: { ...updateStatus },
  };
}

async function sendChat(request: ChatRequest): Promise<ChatResponse> {
  const messages = normalizeDesktopChatMessages(request.messages);
  const response = await fetchJson<{
    id: string;
    model: string;
    choices: Array<{ message: { content: string } }>;
    usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
    x_network: {
      session_id: string;
      route_class: string;
      affinity_hit: boolean;
      reused_kv_tokens?: number;
      ttft_ms: number;
      active_ms: number;
    };
  }>("v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: request.model,
      messages,
      session_id: request.sessionId,
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
    sessionId: response.x_network.session_id,
    reusedKvTokens: response.x_network.reused_kv_tokens ?? 0,
    ttftMs: response.x_network.ttft_ms,
    activeMs: response.x_network.active_ms,
  };
}

async function streamChat(request: ChatRequest, onUpdate: (update: ChatStreamUpdate) => void): Promise<ChatResponse> {
  const messages = normalizeDesktopChatMessages(request.messages);
  return consumeChatCompletionStreamWithRecovery(
    async () => {
      const headers = new Headers({ accept: "text/event-stream", "content-type": "application/json" });
      if (settings.coordinatorMode === "remote" && settings.remoteCoordinatorToken) {
        headers.set("authorization", `Bearer ${settings.remoteCoordinatorToken}`);
      }
      return fetch(new URL("v1/chat/completions", `${coordinatorUrl}/`), {
        method: "POST",
        headers,
        body: JSON.stringify({
          model: request.model,
          messages,
          session_id: request.sessionId,
          stream: true,
          max_tokens: Math.max(1, Math.min(2_048, request.maxTokens ?? 128)),
          temperature: 0,
          top_p: 1,
        }),
        signal: AbortSignal.timeout(3 * 60_000),
        redirect: "error",
      });
    },
    request.model,
    onUpdate,
    { sessionId: request.sessionId },
  );
}

function normalizeDesktopChatMessages(messages: ChatRequest["messages"]): ChatRequest["messages"] {
  const normalized = messages.map((message) => ({ ...message }));
  const last = normalized.at(-1);
  if (!last || last.role !== "user" || !last.content.trim()) {
    throw new Error("Escribe un mensaje antes de enviarlo.");
  }
  return normalized;
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
    else {
      clearAcceleratorRetryTimer();
      if (worker) {
        const active = worker;
        worker = null;
        await active.stop();
      }
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
  ipcMain.handle("models:search-hub", async (_event, input: HubCatalogSearchInput) => {
    const parameters = new URLSearchParams({
      q: input.query,
      sort: input.sort ?? "downloads",
      limit: String(input.limit ?? 50),
    });
    if (input.cursor) parameters.set("cursor", input.cursor);
    return fetchJson<HubCatalogPage>(`public/v1/huggingface-models?${parameters.toString()}`);
  });
  ipcMain.handle("models:request", async (_event, input: import("./contracts.js").RequestModelInput, adminToken?: string) => {
    const providedToken = adminToken?.trim() ?? "";
    const token = providedToken || modelAdminToken;
    await fetchJson("public/v1/requested-models", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(token ? { authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(input),
    });
    if (providedToken && providedToken !== modelAdminToken) persistModelAdminToken(providedToken);
    return readSnapshot();
  });
  ipcMain.handle("models:remove-request", async (_event, modelId: string, adminToken?: string) => {
    const providedToken = adminToken?.trim() ?? "";
    const token = providedToken || modelAdminToken;
    await fetchJson(`public/v1/requested-models/${encodeURIComponent(modelId)}`, {
      method: "DELETE",
      ...(token ? { headers: { authorization: `Bearer ${token}` } } : {}),
    });
    if (providedToken && providedToken !== modelAdminToken) persistModelAdminToken(providedToken);
    return readSnapshot();
  });
  ipcMain.handle("benchmarks:read", async () => {
    const result = await fetchJson<{ runs: import("../benchlab/types.js").BenchmarkRun[] }>("public/v1/benchmarks");
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
    return installDownloadedUpdate(true, "user-requested");
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
  const icon = desktopIconPath("app-icon-v2.png");
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
  const trayIconPath = process.platform === "win32"
    ? desktopIconPath("app-icon-v2.ico")
    : desktopIconPath("32x32", "mycellios.png");
  const trayImage = nativeImage.createFromPath(trayIconPath);
  if (trayImage.isEmpty()) throw new Error(`Tray icon could not be loaded: ${trayIconPath}`);
  tray?.destroy();
  tray = new Tray(trayImage);
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

function desktopHardwareSelectionInput() {
  return {
    platform: process.platform,
    arch: process.arch,
    osRelease: release(),
    cpuModel: cpus()[0]?.model,
  };
}

function currentVerifiedGpuRuntime(): VerifiedGpuRuntimeEvidence | undefined {
  if (settings.computeMode === "cpu-only") return undefined;
  const runtime = resolvedAcceleratorRuntime;
  if (!runtime || runtime.status !== "gpu-ready") return undefined;
  const backend = runtime.effectiveBackend;
  if (backend === "cpu") return undefined;
  return {
    status: "gpu-ready",
    backend,
    deviceName: runtime.deviceName,
  };
}

function currentDesktopExecutorPolicy(): {
  computeMode: DesktopSettings["computeMode"];
  cpuEligible: boolean;
} {
  return desktopExecutorPolicy(
    settings.computeMode,
    gpuPreparationIsContinuing(accelerationStatus),
  );
}

function currentAccelerationDiagnostics(): WorkerAcceleratorDiagnostics {
  return buildWorkerAccelerationDiagnostics({
    appVersion: app.getVersion(),
    acceleration: accelerationStatus,
    retryAttempt: acceleratorRetryAttempt,
    nextRetryAt: acceleratorNextRetryAt,
  });
}

function scheduleAccelerationDiagnosticsPublish(delayMs = 1_500): void {
  if (!worker || accelerationDiagnosticsPublishTimer || isQuitting) return;
  accelerationDiagnosticsPublishTimer = setTimeout(() => {
    accelerationDiagnosticsPublishTimer = null;
    const activeWorker = worker;
    if (!activeWorker || isQuitting) return;
    const diagnostics = currentAccelerationDiagnostics();
    void activeWorker.refreshRuntimeDiagnostics(diagnostics).catch((error: unknown) => {
      writeDesktopLog("worker-acceleration-diagnostics-refresh-failed", {
        error: errorText(error),
      });
    });
  }, delayMs);
  accelerationDiagnosticsPublishTimer.unref();
}

function refreshPublishedRuntimeCapacity(): void {
  const activeWorker = worker;
  if (!activeWorker) return;
  const runtime = currentVerifiedGpuRuntime();
  const policy = currentDesktopExecutorPolicy();
  const diagnostics = currentAccelerationDiagnostics();
  void activeWorker.refreshRuntimeCapacity(runtime, policy, diagnostics).then(
    () => {
      writeDesktopLog("worker-runtime-capacity-refreshed", {
        backend: runtime?.backend ?? "cpu",
        deviceName: runtime?.deviceName ?? cpus()[0]?.model ?? "CPU",
        computeMode: policy.computeMode,
        cpuEligible: policy.cpuEligible,
      });
    },
    (error: unknown) => {
      writeDesktopLog("worker-runtime-capacity-refresh-failed", { error: errorText(error) });
    },
  );
}

function clearAcceleratorRetryTimer(): void {
  if (acceleratorRetryTimer) clearTimeout(acceleratorRetryTimer);
  acceleratorRetryTimer = null;
  acceleratorNextRetryAt = null;
  scheduleAccelerationDiagnosticsPublish();
}

function startDesktopAcceleratorPreparation(runtimeRoot: string): void {
  if (
    isQuitting
    || !settings.contributionEnabled
    || settings.computeMode === "cpu-only"
    || resolvedAcceleratorRuntime?.status === "gpu-ready"
    || acceleratorRuntimePromise !== null
  ) return;
  void prepareDesktopAcceleratorRuntime(runtimeRoot).then((runtime) => {
    writeDesktopLog("accelerator-runtime-ready", {
      status: runtime.status,
      requestedBackend: runtime.requestedBackend,
      effectiveBackend: runtime.effectiveBackend,
      deviceName: runtime.deviceName,
      precision: runtime.precision,
      fallbackReason: runtime.fallbackReason,
    });
    refreshPublishedRuntimeCapacity();
  }).catch((error: unknown) => {
    writeDesktopLog("accelerator-runtime-failed", { error: errorText(error) });
  });
}

function scheduleDesktopAcceleratorRetry(runtimeRoot: string, issueCode: string): void {
  if (
    isQuitting
    || !settings.contributionEnabled
    || settings.computeMode === "cpu-only"
    || acceleratorRetryTimer
  ) return;

  if (acceleratorRetryIssueCode !== issueCode) {
    acceleratorRetryIssueCode = issueCode;
    acceleratorRetryAttempt = 0;
  }

  const issueAttempts = acceleratorRetryIssueAttempts.get(issueCode) ?? 0;
  const delayMs = gpuPreparationRetryDelayMs(acceleratorRetryAttempt);
  acceleratorRetryAttempt += 1;
  acceleratorRetryIssueAttempts.set(issueCode, issueAttempts + 1);
  acceleratorNextRetryAt = new Date(Date.now() + delayMs).toISOString();
  const seconds = Math.ceil(delayMs / 1_000);
  accelerationStatus = appendAccelerationLog(accelerationStatus, {
    at: new Date().toISOString(),
    level: "warning",
    message: `GPU self-repair attempt ${issueAttempts + 1} will run automatically in ${seconds} seconds. The CPU runtime remains available.`,
  });
  scheduleAccelerationDiagnosticsPublish();
  // A failed GPU setup is often repaired by a newer certified pack or desktop
  // build. Check immediately on every failure so an unattended node does not
  // repeat a known-bad installer until the general update interval elapses.
  void checkForUpdates();
  acceleratorRetryTimer = setTimeout(() => {
    acceleratorRetryTimer = null;
    acceleratorNextRetryAt = null;
    scheduleAccelerationDiagnosticsPublish();
    startDesktopAcceleratorPreparation(runtimeRoot);
  }, delayMs);
  acceleratorRetryTimer.unref();
}

async function createDesktopDistributedExecutor() {
  const runtimeRoot = await ensureDistributionRuntime();
  acceleratorRuntimeRoot = runtimeRoot;
  const pythonExecutable = distributionPythonExecutable(runtimeRoot);
  if (!existsSync(pythonExecutable)) {
    throw new Error("The packaged shard runtime is missing. Reinstall mycellios to contribute this device.");
  }
  const nodeId = persistentDistributedNodeId();
  const launchAgent = new DesktopAcceleratedLaunchAgent(runtimeRoot, nodeId);
  // The small certified CPU runtime ships with the app and becomes available
  // before registration. GPU provisioning always happens beside it in an
  // isolated app-data directory, so a multi-gigabyte download never blocks
  // this device from accepting a compatible CPU stage.
  await prepareDesktopCpuRuntime(runtimeRoot);
  return {
    nodeId,
    // This is a globally unique logical route name, never a reachable LAN IP.
    // Protocol v2 rewrites every runtime connection onto the coordinator relay.
    stageHost: `${nodeId}.relay`,
    stagePort: 9_850,
    pythonExecutable,
    launchAgent,
  };
}

class DesktopAcceleratedLaunchAgent implements LaunchAgent {
  readonly id: string;

  constructor(
    private readonly baseRuntimeRoot: string,
    private readonly nodeId: string,
  ) {
    this.id = `desktop-shard-executor:${nodeId}`;
  }

  async start(
    request: LaunchAgentStartRequest,
    signal: AbortSignal,
  ): Promise<LaunchProcessHandle> {
    // Automatic distribution only selects this node as GPU capacity after the
    // physical accelerator probe has passed. A CPU runtime remains available
    // for explicitly CPU-only work, but a node that claimed GPU capacity must
    // not silently turn a distributed model into a permanent mixed pipeline.
    const runtime = settings.computeMode === "cpu-only"
      ? await prepareDesktopCpuRuntime(this.baseRuntimeRoot)
      : await selectImmediateRuntime(
          () => resolvedAcceleratorRuntime,
          () => prepareDesktopCpuRuntime(this.baseRuntimeRoot),
        );
    if (settings.computeMode === "gpu-only" && runtime.deviceType !== "gpu") {
      throw new Error("gpu_only_runtime_not_ready");
    }
    if (runtime.deviceType === "gpu") return this.startVerifiedGpuStage(request, signal, runtime);

    const cpuHandle = await this.launchWithRuntime(request, signal, runtime);
    this.observeHandle(cpuHandle, request, runtime, false);
    return cpuHandle;
  }

  private async startVerifiedGpuStage(
    request: LaunchAgentStartRequest,
    signal: AbortSignal,
    runtime: AcceleratorRuntimeResult,
  ): Promise<LaunchProcessHandle> {
    const maximumAttempts = 2;
    let lastError: unknown = new Error("gpu_model_stage_failed");
    for (let attempt = 1; attempt <= maximumAttempts; attempt += 1) {
      let handle: LaunchProcessHandle | null = null;
      try {
        handle = await this.launchWithRuntime(request, signal, runtime);
        await handle.ready;
        this.observeHandle(handle, request, runtime, true);
        return handle;
      } catch (error) {
        lastError = error;
        if (signal.aborted) {
          throw signal.reason instanceof Error ? signal.reason : new Error("distributed_launch_cancelled");
        }
        await handle?.stop("gpu_model_stage_failed").catch(() => undefined);
        if (attempt < maximumAttempts) {
          accelerationStatus = appendAccelerationLog(accelerationStatus, {
            at: new Date().toISOString(),
            level: "warning",
            message: `GPU model stage ${request.launchId} failed to load; retrying once on ${runtime.deviceName}.`,
          });
        }
      }
    }

    const reason = errorText(lastError);
    invalidateGpuRuntimeAfterStageFailure(runtime, request.launchId, reason);
    if (settings.computeMode === "automatic") {
      accelerationStatus = appendAccelerationLog(accelerationStatus, {
        at: new Date().toISOString(),
        level: "warning",
        message: `GPU retries were exhausted for ${request.launchId}; Automatic mode is using the authorized CPU runtime.`,
      });
      const cpuRuntime = await prepareDesktopCpuRuntime(this.baseRuntimeRoot);
      const cpuHandle = await this.launchWithRuntime(request, signal, cpuRuntime);
      this.observeHandle(cpuHandle, request, cpuRuntime, false);
      return cpuHandle;
    }
    throw new Error(`gpu_model_stage_unavailable_after_retries:${runtime.effectiveBackend}:${reason}`);
  }

  private async launchWithRuntime(
    request: LaunchAgentStartRequest,
    signal: AbortSignal,
    runtime: AcceleratorRuntimeResult,
  ): Promise<LaunchProcessHandle> {
    if (signal.aborted) throw signal.reason instanceof Error ? signal.reason : new Error("distributed_launch_cancelled");
    const pythonPath = resourcePath("python");
    const hfHome = join(app.getPath("userData"), "model-shards");
    mkdirSync(hfHome, { recursive: true });
    const executor = new LocalProcessAgent({
      id: this.id,
      cwd: app.isPackaged ? dirname(app.getAppPath()) : app.getAppPath(),
      env: {
        PYTHONPATH: [...runtime.pythonPathAdditions, pythonPath].join(delimiter),
        HF_HOME: hfHome,
        TOKENIZERS_PARALLELISM: "false",
        PATH: [...runtime.pathAdditions, dirname(runtime.pythonExecutable), process.env.PATH]
          .filter(Boolean)
          .join(delimiter),
      },
      maxOutputBytesPerStream: 256 * 1024,
    });
    const commandArgs = [...request.process.command.args];
    const deviceFlag = commandArgs.lastIndexOf("--device");
    if (deviceFlag >= 0 && deviceFlag + 1 < commandArgs.length) {
      // The preparation result owns this mapping. In particular, MPS/XPU must
      // never be rewritten to CUDA, and a failed accelerator probe must launch
      // explicitly on CPU instead of letting `auto` rediscover the failed GPU.
      commandArgs[deviceFlag + 1] = runtime.launchDevice;
    }
    const handle = await executor.start({
      ...request,
      process: {
        ...request.process,
        command: {
          ...request.process.command,
          executable: runtime.pythonExecutable,
          args: commandArgs,
        },
      },
    }, signal);
    activeDistributedStages.add(handle);
    const deviceType = runtime.deviceType;
    const setupInProgress = deviceType === "cpu" && gpuPreparationIsContinuing(accelerationStatus);
    accelerationStatus = appendAccelerationLog(accelerationStatus, {
      at: new Date().toISOString(),
      level: "info",
      message: deviceType === "gpu"
        ? `Model stage ${request.launchId} launched on ${runtime.deviceName}; waiting for model load and the distributed canary.`
        : setupInProgress
          ? `Model stage ${request.launchId} launched on CPU while GPU setup continues; waiting for model load and the distributed canary.`
          : `Model stage ${request.launchId} launched on CPU; waiting for model load and the distributed canary.`,
    });
    return handle;
  }

  private observeHandle(
    handle: LaunchProcessHandle,
    request: LaunchAgentStartRequest,
    runtime: AcceleratorRuntimeResult,
    readyAlready: boolean,
  ): void {
    const deviceType = runtime.deviceType;
    const recordReady = () => {
      accelerationStatus = appendAccelerationLog(accelerationStatus, {
        at: new Date().toISOString(),
        level: "info",
        message: `Model stage ${request.launchId} loaded on ${deviceType.toUpperCase()}; ACTIVE will appear only after coordinator canary evidence is published.`,
      });
    };
    if (readyAlready) recordReady();
    else void handle.ready.then(
      recordReady,
      (error: unknown) => {
        accelerationStatus = appendAccelerationLog(accelerationStatus, {
          at: new Date().toISOString(),
          level: "warning",
          message: `Model stage ${request.launchId} did not become ready on ${deviceType.toUpperCase()}: ${errorText(error)}`,
        });
      },
    );
    void handle.exited.then((exit) => {
      activeDistributedStages.delete(handle);
      if (updateStatus.state === "ready") scheduleAutomaticUpdateInstall();
      const unexpected = unexpectedRuntimeExit(exit);
      accelerationStatus = appendAccelerationLog(accelerationStatus, {
        at: new Date().toISOString(),
        level: unexpected ? "warning" : "info",
        message: unexpected
          ? `Model stage ${request.launchId} exited unexpectedly on ${deviceType.toUpperCase()} (code ${exit.code ?? "none"}${exit.signal ? `, ${exit.signal}` : ""}).`
          : `Model stage ${request.launchId} stopped on ${deviceType.toUpperCase()}.`,
      });
      if (deviceType === "gpu" && unexpected) {
        invalidateGpuRuntimeAfterStageFailure(
          runtime,
          request.launchId,
          exit.error ?? `process exited with code ${exit.code ?? "none"}${exit.signal ? ` and signal ${exit.signal}` : ""}`,
        );
      }
    });
  }
}

function unexpectedRuntimeExit(exit: LaunchProcessExit): boolean {
  if (exit.error) return true;
  if (exit.code !== null && exit.code !== 0) return true;
  return exit.signal !== null && exit.signal !== "SIGTERM" && exit.signal !== "SIGINT";
}

function invalidateGpuRuntimeAfterStageFailure(
  runtime: AcceleratorRuntimeResult,
  launchId: string,
  reason: string,
): void {
  if (runtime.deviceType !== "gpu" || resolvedAcceleratorRuntime !== runtime) return;
  resolvedAcceleratorRuntime = null;
  acceleratorRuntimePromise = null;
  clearAcceleratorRetryTimer();
  const message = `${runtime.deviceName} could not run model stage ${launchId}; GPU capacity was withdrawn until verification passes again.`;
  accelerationStatus = {
    ...accelerationStatus,
    state: "gpu-fallback",
    requestedBackend: runtime.effectiveBackend,
    effectiveBackend: "cpu",
    precision: "float32",
    message,
    cpu: {
      ...accelerationStatus.cpu,
      state: accelerationStatus.cpu.activeStages > 0 ? "active" : "ready",
      message: accelerationStatus.cpu.activeStages > 0
        ? accelerationStatus.cpu.message
        : "Certified CPU runtime remains ready for explicitly CPU-compatible work.",
    },
    gpu: {
      ...accelerationStatus.gpu,
      state: "fallback",
      activeStages: 0,
    },
    preparation: {
      ...accelerationStatus.preparation,
      phase: "fallback",
      progressPct: null,
      updatedAt: new Date().toISOString(),
      issue: {
        code: "gpu-model-stage",
        message: reason,
        action: "mycellios will verify the accelerator again automatically before this device rejoins GPU distribution.",
        retryable: true,
      },
    },
  };
  accelerationStatus = appendAccelerationLog(accelerationStatus, {
    at: new Date().toISOString(),
    level: "error",
    message: `${message} Reason: ${reason}`,
  });
  refreshPublishedRuntimeCapacity();
  if (acceleratorRuntimeRoot) scheduleDesktopAcceleratorRetry(acceleratorRuntimeRoot, "gpu-model-stage");
}

function prepareDesktopCpuRuntime(runtimeRoot: string): Promise<AcceleratorRuntimeResult> {
  if (cpuRuntimePromise) return cpuRuntimePromise;
  const cpuModel = cpus()[0]?.model;
  cpuRuntimePromise = prepareAcceleratorRuntime({
    baseRuntimeRoot: runtimeRoot,
    userDataPath: app.getPath("userData"),
    hardware: {
      platform: process.platform,
      arch: process.arch,
      cpuModel,
    },
    preferredBackend: "cpu",
    allowProvisioning: false,
  }).then((runtime) => {
    accelerationStatus = {
      ...accelerationStatus,
      effectiveBackend: "cpu",
      precision: "float32",
      cpu: {
        ...accelerationStatus.cpu,
        state: accelerationStatus.cpu.activeStages > 0 ? "active" : "ready",
        deviceName: cpuModel?.trim() || "CPU",
        message: accelerationStatus.cpu.activeStages > 0
          ? accelerationStatus.cpu.message
          : "Certified CPU runtime ready; this device can accept compatible work now.",
      },
    };
    accelerationStatus = appendAccelerationLog(accelerationStatus, {
      at: new Date().toISOString(),
      level: "success",
      message: "Certified CPU runtime ready; this device can accept compatible work now.",
    });
    return runtime;
  });
  return cpuRuntimePromise;
}

function prepareDesktopAcceleratorRuntime(runtimeRoot: string): Promise<AcceleratorRuntimeResult> {
  if (acceleratorRuntimePromise) return acceleratorRuntimePromise;
  if (!app.isPackaged && process.env.MYCELLIOS_PROVISION_ACCELERATOR !== "1") {
    acceleratorRuntimePromise = prepareDesktopCpuRuntime(runtimeRoot).then((developmentCpu) => {
      resolvedAcceleratorRuntime = developmentCpu;
      accelerationStatus = {
        ...accelerationStatus,
        state: "cpu-ready",
        requestedBackend: "cpu",
        effectiveBackend: "cpu",
        deviceName: developmentCpu.deviceName,
        precision: "float32",
        message: "Development mode uses the existing certified CPU runtime.",
        preparation: {
          ...accelerationStatus.preparation,
          phase: "fallback",
          progressPct: null,
          updatedAt: new Date().toISOString(),
        },
      };
      return developmentCpu;
    });
    return acceleratorRuntimePromise;
  }
  accelerationStatus = applyAccelerationProgress(accelerationStatus, {
    phase: "detecting",
    progressPct: 0,
    message: "Detecting the local GPU while the certified CPU runtime remains ready.",
  });
  const preparation = getHardware().then((hardware) => {
    const cpuModel = cpus()[0]?.model;
    // Win32_VideoController ordering is not stable and commonly puts an Intel
    // adapter or a virtual display ahead of a supported AMD accelerator. Pick
    // the first device for which this release actually has a certified pack;
    // otherwise the desktop could silently stay on CPU despite a usable GPU.
    const gpu = selectDesktopHardwareGpu(hardware.gpus, desktopHardwareSelectionInput());
    const selectedPack = selectAcceleratorPack({
      platform: process.platform,
      arch: process.arch,
      osRelease: release(),
      gpuVendor: gpu?.vendor,
      gpuModel: gpu?.model,
      cpuModel,
    });
    accelerationStatus = beginAccelerationPreparation(accelerationStatus, {
      vendor: gpu?.vendor ?? null,
      model: gpu?.model ?? null,
      backend: selectedPack?.backend ?? null,
      cpuName: cpuModel,
    });
    return prepareAcceleratorRuntime({
      baseRuntimeRoot: runtimeRoot,
      userDataPath: app.getPath("userData"),
      hardware: {
        platform: process.platform,
        arch: process.arch,
        osRelease: release(),
        gpuVendor: gpu?.vendor,
        gpuModel: gpu?.model,
        gpuDeviceIndex: gpu?.runtimeDeviceIndex,
        cpuModel,
      },
      allowProvisioning: app.isPackaged || process.env.MYCELLIOS_PROVISION_ACCELERATOR === "1",
      onProgress: applyDesktopAcceleratorProgress,
    });
  }).then((runtime) => {
    resolvedAcceleratorRuntime = runtime.status === "gpu-ready" ? runtime : null;
    const targetDevice = accelerationStatus.gpu.model;
    accelerationStatus = {
      ...accelerationStatus,
      state: runtime.status,
      requestedBackend: runtime.requestedBackend,
      effectiveBackend: runtime.effectiveBackend,
      deviceName: runtime.status === "gpu-ready" ? runtime.deviceName : targetDevice ?? runtime.deviceName,
      precision: runtime.precision,
      message: runtime.status === "gpu-ready"
        ? `${runtime.deviceName} passed the physical FP16 probe.`
        : runtime.status === "gpu-fallback"
          ? runtime.fallbackReason ?? "The accelerator probe failed; CPU fallback is active."
          : "The certified CPU runtime is ready.",
      cpu: {
        ...accelerationStatus.cpu,
        state: accelerationStatus.cpu.activeStages > 0 ? "active" : "ready",
      },
    };
    if (runtime.status === "gpu-ready") {
      acceleratorRetryAttempt = 0;
      acceleratorRetryIssueCode = null;
      acceleratorRetryIssueAttempts.clear();
      clearAcceleratorRetryTimer();
      refreshPublishedRuntimeCapacity();
    } else if (accelerationStatus.preparation.issue?.retryable) {
      scheduleDesktopAcceleratorRetry(runtimeRoot, accelerationStatus.preparation.issue.code);
    }
    return runtime;
  }, (error: unknown) => {
    const message = errorText(error);
    accelerationStatus = applyAccelerationProgress(accelerationStatus, {
      phase: "error",
      message,
      issue: {
        code: "runtime-error",
        message,
        action: "mycellios will keep retrying, rebuild damaged files and apply repair updates automatically.",
        retryable: true,
      },
      level: "error",
    });
    accelerationStatus = {
      ...accelerationStatus,
      state: "error",
      effectiveBackend: "cpu",
      precision: "float32",
    };
    resolvedAcceleratorRuntime = null;
    scheduleDesktopAcceleratorRetry(runtimeRoot, "runtime-error");
    throw error;
  });
  acceleratorRuntimePromise = preparation;
  void preparation.then(
    (runtime) => {
      if (runtime.status !== "gpu-ready" && acceleratorRuntimePromise === preparation) {
        acceleratorRuntimePromise = null;
      }
    },
    () => {
      if (acceleratorRuntimePromise === preparation) acceleratorRuntimePromise = null;
    },
  );
  return preparation;
}

function applyDesktopAcceleratorProgress(event: AcceleratorProgressEvent): void {
  accelerationStatus = {
    ...accelerationStatus,
    requestedBackend: event.backend ?? accelerationStatus.requestedBackend,
    deviceName: event.gpuModel ?? accelerationStatus.deviceName,
    gpu: {
      ...accelerationStatus.gpu,
      vendor: event.gpuVendor ?? accelerationStatus.gpu.vendor,
      model: event.gpuModel ?? accelerationStatus.gpu.model,
      backend: event.backend ?? accelerationStatus.gpu.backend,
    },
  };
  accelerationStatus = applyAccelerationProgress(accelerationStatus, {
    phase: event.phase,
    message: event.message,
    progressPct: event.percent,
    bytesCompleted: event.download?.aggregateDownloaded,
    bytesTotal: event.download?.aggregateTotal,
    bytesPerSecond: event.download?.bytesPerSecond ?? null,
    etaSeconds: event.download?.etaSeconds ?? null,
    currentArtifact: event.download?.artifact ?? null,
    artifactIndex: event.download?.artifactIndex ?? null,
    artifactCount: event.download?.artifactCount ?? null,
    issue: event.issue
      ? {
          ...event.issue,
          message: event.message,
        }
      : undefined,
    recordLog: event.recordLog,
    at: event.at,
  });
  if (event.recordLog) {
    writeDesktopLog("accelerator-runtime-progress", {
      phase: event.phase,
      percent: event.percent,
      backend: event.backend,
      gpuModel: event.gpuModel,
      message: event.message,
      issue: event.issue,
    });
  }
  scheduleAccelerationDiagnosticsPublish(event.recordLog ? 500 : 10_000);
}

function distributionPythonExecutable(root = app.isPackaged
  ? join(app.getPath("userData"), "distribution-runtime-v3")
  : join(app.getAppPath(), "runtime", "distribution-venv")): string {
  if (process.platform === "win32") {
    const portable = join(root, "python.exe");
    return app.isPackaged || existsSync(portable) ? portable : join(root, "Scripts", "python.exe");
  }
  return join(root, "bin", "python3");
}

function ensureDistributionRuntime(): Promise<string> {
  if (!distributionRuntimePromise) {
    distributionRuntimePromise = ensureDistributionRuntimeOnce().catch((error: unknown) => {
      distributionRuntimePromise = null;
      throw error;
    });
  }
  return distributionRuntimePromise;
}

async function ensureDistributionRuntimeOnce(): Promise<string> {
  if (!app.isPackaged) return join(app.getAppPath(), "runtime", "distribution-venv");
  const userData = app.getPath("userData");
  const root = join(userData, "distribution-runtime-v3");
  const staging = join(userData, "distribution-runtime-v3.staging");
  if (dirname(root) !== userData || dirname(staging) !== userData) {
    throw new Error("The shard runtime escaped its managed application directory.");
  }
  if (existsSync(root)) {
    try {
      await verifyPortableRuntimeInstallation(
        root,
        { platform: process.platform, arch: process.arch },
      );
      rmSync(staging, { recursive: true, force: true });
      return root;
    } catch (error) {
      writeDesktopLog("distribution-runtime-v3-invalid", { error: errorText(error) });
      rmSync(root, { recursive: true, force: true });
    }
  }
  const archive = resourcePath("distribution-runtime.tar.gz");
  if (!existsSync(archive)) throw new Error("The packaged shard runtime archive is missing. Reinstall mycellios.");
  rmSync(staging, { recursive: true, force: true });
  mkdirSync(staging, { recursive: true });
  try {
    // Never expose a half-extracted runtime as usable. If the application is
    // closed during extraction, only the staging directory is left behind and
    // the next launch safely starts it again.
    await runProcess("tar", ["-xzf", archive, "-C", staging]);
    await verifyPortableRuntimeInstallation(
      staging,
      { platform: process.platform, arch: process.arch },
    );
    renameSync(staging, root);
    return root;
  } catch (error) {
    rmSync(staging, { recursive: true, force: true });
    throw error;
  }
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

function buildDesktopActivationSnapshot(
  workers: readonly StoredWorker[],
  connectedWorkerIds: ReadonlySet<string>,
): import("../coordinator/model-activation-manager.js").DynamicActivationSnapshot {
  const executors = workers
    .filter((worker) => connectedWorkerIds.has(worker.id) && worker.capabilities.distributedExecutor)
    .map((worker) => ({ worker, executor: worker.capabilities.distributedExecutor! }))
    .filter(({ executor }) => executor.protocol === "gdlp-worker-tunnel/2")
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
      // The first physical launch may install a multi-gigabyte vendor runtime.
      // Subsequent launches reuse the content-addressed accelerator pack.
      readinessTimeoutMs: 3_600_000,
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
  const worker = workers.find((candidate) =>
    candidate.capabilities.distributedExecutor?.protocol === "gdlp-worker-tunnel/2"
    && candidate.capabilities.distributedExecutor.nodeId === nodeId
  );
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
  modelAdminToken = loadModelAdminToken();
  registerIpc();
  await restartRuntime().catch((error: unknown) => {
    runtimeError = errorText(error);
    writeDesktopLog("runtime-start-failed", { error: runtimeError });
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
  resetAutomaticUpdateRetry();
  clearAutomaticUpdateInstallTimer();
  if (accelerationDiagnosticsPublishTimer) {
    clearTimeout(accelerationDiagnosticsPublishTimer);
    accelerationDiagnosticsPublishTimer = null;
  }
  void stopRuntime();
});
