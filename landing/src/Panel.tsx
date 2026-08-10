import {
  Activity,
  ArrowRight,
  Bot,
  Boxes,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  CirclePower,
  Code2,
  Coins,
  Copy,
  Cpu,
  Download,
  ExternalLink,
  FileText,
  Gauge,
  GitFork,
  Globe2,
  HardDrive,
  History,
  Laptop,
  LayoutDashboard,
  LoaderCircle,
  LockKeyhole,
  LogOut,
  Maximize2,
  MemoryStick,
  Menu,
  MessageSquareText,
  Minus,
  Network,
  Paperclip,
  Pause,
  Play,
  Plus,
  Radio,
  RefreshCw,
  Search,
  Send,
  Server,
  Settings,
  Share2,
  ShieldCheck,
  SlidersHorizontal,
  ScrollText,
  Sparkles,
  Smartphone,
  SunMoon,
  Timer,
  Trash2,
  UserRound,
  WalletCards,
  Wifi,
  X,
  Zap,
} from "lucide-react";
import { useCallback, useEffect, useId, useMemo, useRef, useState, type CSSProperties, type FormEvent } from "react";
import { createPortal } from "react-dom";
import "@fontsource-variable/manrope";
import type {
  DashboardSnapshot,
  DashboardDeployment,
  ChatResponse,
  ChatStreamUpdate,
  DesktopAccelerationStatus,
  DesktopBridge,
  DesktopSettings,
  DesktopUpdateStatus,
  FleetContributionCommandResponse,
  FleetContributionStatus,
  HubCatalogModel,
  HubCatalogPage,
  HubCatalogSearchInput,
  HubCatalogSort,
  RequestModelInput,
  RequestedModelCapacity,
  SystemLogEntry,
  SystemLogSnapshot,
  WorkerCredentialSummary,
} from "../../src/desktop/contracts";
import { consumeChatCompletionStreamWithRecovery } from "../../src/desktop/chat-stream";
import type { ChatMessage, NetworkExecutionTrace } from "../../src/contracts/types";
import type { NativeBuildIdentity } from "../../src/contracts/build-identity";
import type { BenchmarkMeasurement, BenchmarkRun } from "../../src/benchlab/types";
import { Contribute } from "./Contribute";
import { SupportAssistant } from "./SupportAssistant";
const brandIcon = "/assets/logos/logo.png";
import {
  benchmarkRunSnapshot,
  benchmarkFilterValues,
  benchmarkComparisonNarrative,
  benchmarkTrendRuns,
  compareBenchmarkRuns,
  defaultBenchmarkBaseline,
  EMPTY_BENCHMARK_FILTERS,
  filterBenchmarkRuns,
  measurementBackends,
  type BenchmarkFilters,
  type BenchmarkRunComparison,
  type BenchmarkRunSnapshot,
} from "./benchmark-comparison";
import { isAdvertisedGpuSelected } from "./panel-hardware";
import { applySeoMetadata } from "./seo";
import {
  loadAuthConfig,
  loadNetworkIdentity,
  restoreAuthSession,
  signIn,
  signInWithMetaMask,
  signInWithX,
  signOut,
  signUp,
  type AuthSession,
  type NetworkIdentity,
  type PublicAuthConfig,
} from "./auth";
import { ApiAccessPanel } from "./ApiAccessPanel";
import {
  loadApiAccount,
  loadApiUsage,
  type ApiAccount,
  type ApiUsage,
} from "./api-access";
import "./panel.css";

const PUBLIC_COORDINATOR_URL = "https://www.mycellios.com";
import "./panel-downloads.css";
import "./panel-desktop.css";

type PanelView = "overview" | "history" | "nodes" | "models" | "jobs" | "tests" | "logs" | "inference" | "contribute" | "join" | "downloads" | "machine" | "settings" | "admin";
type PanelMode = "simple" | "developer";

interface PanelProps {
  desktopBridge?: DesktopBridge;
  mobileEntry?: boolean;
}

type AcceleratorProgressSnapshot = DesktopAccelerationStatus;

interface PublicGpu {
  id: string;
  vendor: string;
  model: string;
  physicalVramMb: number;
  sharedMemoryMb?: number;
  offeredVramMb: number;
  freeOfferedVramMb: number;
  utilizationPct?: number;
  temperatureC?: number;
  powerW?: number;
}

interface PublicDeployment {
  deploymentId: string;
  model: string;
  mode: string;
  adapter?: string;
  freeSlots: number;
  tokensPerSecond: number;
  throughputSource?: "measured" | "estimated" | "configured" | "default";
  ttftMs?: number;
  execution?: DashboardDeployment["execution"];
}

interface PublicWorker {
  id: string;
  kind: "desktop" | "browser" | "cell";
  status: "online" | "suspect" | "offline" | "draining";
  connected: boolean;
  region: string;
  offeredVramMb: number;
  reliability: number;
  jobsCompleted: number;
  lastSeenAt: string;
  gpus: PublicGpu[];
  deployments: PublicDeployment[];
  executionNodeId?: string;
  computeMode?: DesktopSettings["computeMode"];
  agentVersion?: string;
  buildIdentity?: NativeBuildIdentity;
  acceleration?: DashboardSnapshot["workers"][number]["acceleration"];
  isolation?: DashboardSnapshot["workers"][number]["isolation"];
  mobile?: {
    platform: string;
    backend: "webgpu" | "cpu";
    performanceLevel: string;
    wakeLock: boolean;
    estimatedGflops: number;
    verifiedTasks: number;
  };
}

interface PublicJob {
  id: string;
  model: string;
  status: string;
  workerId: string | null;
  inputTokens: number;
  outputTokens: number;
  failureCode: string | null;
  createdAt: string;
  updatedAt: string;
}

interface PublicSnapshot {
  capturedAt: string;
  version: string;
  buildIdentity: NativeBuildIdentity | null;
  summary: {
    registered: number;
    connected: number;
    online: number;
    mobile: number;
    offeredVramMb: number;
    completedJobs: number;
  };
  workers: PublicWorker[];
  models: Array<{ id: string; replicas: number; pipelines: number }>;
  requestedModels: RequestedModelCapacity[];
  jobs: PublicJob[];
}

type NetworkHistoryRange = "24h" | "7d" | "30d" | "90d";

interface NetworkTelemetrySample {
  capturedAt: string;
  registeredNodes: number;
  connectedNodes: number;
  onlineNodes: number;
  browserNodes: number;
  activeModels: number;
  modelReplicas: number;
  modelPipelines: number;
  offeredVramMb: number;
  freeVramMb: number;
  inflightJobs: number;
  runningJobs: number;
  completedJobs: number;
}

interface NetworkTelemetryHistory {
  capturedAt: string;
  intervalMinutes: number;
  retentionDays: number;
  range: NetworkHistoryRange;
  samples: NetworkTelemetrySample[];
}

const EMPTY: PublicSnapshot = {
  capturedAt: new Date(0).toISOString(),
  version: "—",
  buildIdentity: null,
  summary: { registered: 0, connected: 0, online: 0, mobile: 0, offeredVramMb: 0, completedJobs: 0 },
  workers: [],
  models: [],
  requestedModels: [],
  jobs: [],
};

const sharedNavItems: Array<{ id: PanelView; label: string; icon: typeof Network }> = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "history", label: "History", icon: History },
  { id: "nodes", label: "Nodes", icon: Server },
  { id: "jobs", label: "Tasks", icon: Activity },
  { id: "tests", label: "Tests", icon: Gauge },
  { id: "logs", label: "Logs", icon: ScrollText },
  { id: "models", label: "Models", icon: Boxes },
  { id: "inference", label: "Chat", icon: MessageSquareText },
  { id: "contribute", label: "This device", icon: Zap },
  { id: "downloads", label: "Downloads", icon: Download },
  { id: "admin", label: "Admin", icon: ShieldCheck },
];

const simplePanelViews = new Set<PanelView>(["overview", "history", "inference", "contribute", "downloads", "machine", "settings"]);
const developerPanelViews = new Set<PanelView>(["nodes", "jobs", "tests", "logs", "models"]);

function initialPanelMode(): PanelMode {
  const requested = new URLSearchParams(window.location.search).get("view") as PanelView | null;
  if (requested && developerPanelViews.has(requested)) return "developer";
  return window.localStorage.getItem("mycellios.panel-mode") === "developer" ? "developer" : "simple";
}

function initialView(desktop: boolean, mobileEntry: boolean): PanelView {
  if (desktop) return "overview";
  const requested = new URLSearchParams(window.location.search).get("view");
  if (sharedNavItems.some((item) => item.id === requested)) return requested as PanelView;
  if (mobileEntry || window.location.pathname.startsWith("/mobile")) return "contribute";
  if (window.location.pathname === "/join") return "join";
  if (window.location.pathname === "/downloads") return "downloads";
  if (window.location.pathname === "/admin") return "admin";
  return "overview";
}

export type PanelIssue = {
  source: "coordinator" | "runtime";
  message: string;
};

export function panelIssueForErrors(
  connectionError: string | null,
  runtimeError: string | null,
): PanelIssue | null {
  if (connectionError) return { source: "coordinator", message: connectionError };
  if (runtimeError) return { source: "runtime", message: runtimeError };
  return null;
}

function Panel({ desktopBridge, mobileEntry = false }: PanelProps = {}) {
  const desktop = desktopBridge !== undefined;
  const localBrowser = !desktop && ["localhost", "127.0.0.1", "::1", "[::1]"].includes(window.location.hostname);
  const localProductionProxy = localBrowser && window.location.port === "4174";
  const [panelMode, setPanelMode] = useState<PanelMode>(initialPanelMode);
  const navItems = useMemo(() => {
    const desktopItems = sharedNavItems.filter((item) => item.id !== "contribute");
    const allItems = !desktop
      ? sharedNavItems
      : [
        ...desktopItems.slice(0, 1),
        { id: "machine" as const, label: "This device", icon: Cpu },
        ...desktopItems.slice(1),
        { id: "settings" as const, label: "Settings", icon: Settings },
      ];
    return panelMode === "developer"
      ? allItems
      : allItems.filter((item) => simplePanelViews.has(item.id));
  }, [desktop, panelMode]);
  const [view, setView] = useState<PanelView>(() => initialView(desktop, mobileEntry));
  const [snapshot, setSnapshot] = useState<PublicSnapshot>(EMPTY);
  const [desktopSnapshot, setDesktopSnapshot] = useState<DashboardSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [issue, setIssue] = useState<PanelIssue | null>(null);
  const coordinatorError = issue?.source === "coordinator" ? issue.message : null;
  const error = issue?.message ?? null;
  const [menuOpen, setMenuOpen] = useState(false);
  const menuButtonRef = useRef<HTMLButtonElement>(null);
  const sidebarCloseButtonRef = useRef<HTMLButtonElement>(null);
  const [modelAdminToken, setModelAdminToken] = useState(() =>
    window.sessionStorage.getItem("mycellios-model-admin-token") ?? "",
  );
  const [authConfig, setAuthConfig] = useState<PublicAuthConfig>({ enabled: false });
  const [authSession, setAuthSession] = useState<AuthSession | null>(null);
  const [networkIdentity, setNetworkIdentity] = useState<NetworkIdentity | null>(null);
  const [apiAccount, setApiAccount] = useState<ApiAccount | null>(null);
  const [authOpen, setAuthOpen] = useState(false);
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    const saved = window.localStorage.getItem("mycellios.theme");
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
  });
  const canManageModels = networkIdentity?.role === "owner"
    || networkIdentity?.role === "admin"
    || networkIdentity?.role === "operator";
  const requiresModelAdminToken = desktop
    ? desktopSnapshot?.settings.coordinatorMode !== "local" && !desktopSnapshot?.modelAdminAuthorization.configured
    : (!localBrowser || localProductionProxy) && !canManageModels;
  const applyDesktopSnapshot = useCallback((next: DashboardSnapshot) => {
    setDesktopSnapshot(next);
    setSnapshot(desktopToPublicSnapshot(next));
    setIssue(panelIssueForErrors(next.connectionError, next.runtimeError));
  }, []);

  const refresh = useCallback(async () => {
    try {
      if (desktopBridge) {
        const next = await desktopBridge.getSnapshot();
        applyDesktopSnapshot(next);
        return;
      }
      const response = await fetch(`/public/v1/snapshot?view=${encodeURIComponent(view)}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setSnapshot(await response.json() as PublicSnapshot);
      setIssue(null);
    } catch (caught) {
      setIssue({
        source: "coordinator",
        message: caught instanceof Error ? caught.message : String(caught),
      });
    } finally {
      setLoading(false);
    }
  }, [applyDesktopSnapshot, desktopBridge, view]);

  useEffect(() => {
    let timer: number | undefined;
    const liveViews = new Set<PanelView>(["overview", "nodes", "models", "jobs", "logs", "inference", "machine"]);
    const intervalMs = liveViews.has(view) ? 5_000 : 15_000;
    const schedule = () => {
      if (document.visibilityState === "visible") timer = window.setInterval(() => void refresh(), intervalMs);
    };
    const handleVisibility = () => {
      if (timer !== undefined) window.clearInterval(timer);
      timer = undefined;
      if (document.visibilityState === "visible") {
        void refresh();
        schedule();
      }
    };
    void refresh();
    schedule();
    document.addEventListener("visibilitychange", handleVisibility);
    return () => {
      if (timer !== undefined) window.clearInterval(timer);
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  }, [refresh, view]);

  useEffect(() => {
    if (!menuOpen) return;

    const focusFrame = window.requestAnimationFrame(() => sidebarCloseButtonRef.current?.focus());
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setMenuOpen(false);
      menuButtonRef.current?.focus();
    };
    const closeBeyondCompactLayout = () => {
      if (!window.matchMedia("(max-width: 1440px)").matches) setMenuOpen(false);
    };

    window.addEventListener("keydown", closeOnEscape);
    window.addEventListener("resize", closeBeyondCompactLayout);
    return () => {
      window.cancelAnimationFrame(focusFrame);
      window.removeEventListener("keydown", closeOnEscape);
      window.removeEventListener("resize", closeBeyondCompactLayout);
    };
  }, [menuOpen]);

  useEffect(() => {
    if (desktop) return;
    let cancelled = false;
    void loadAuthConfig().then(async (config) => {
      if (cancelled) return;
      setAuthConfig(config);
      const session = await restoreAuthSession(config);
      if (cancelled || !session) return;
      setAuthSession(session);
      try {
        const identity = await loadNetworkIdentity(session.accessToken);
        if (!cancelled) {
          setNetworkIdentity(identity);
          if (config.apiAccessEnabled) {
            setApiAccount(await loadApiAccount(session.accessToken).catch(() => null));
          }
        }
      } catch {
        if (!cancelled) {
          setAuthSession(null);
          setNetworkIdentity(null);
          setApiAccount(null);
        }
      }
    });
    return () => { cancelled = true; };
  }, [desktop]);

  function changePanelMode(next: PanelMode) {
    setPanelMode(next);
    window.localStorage.setItem("mycellios.panel-mode", next);
    if (next === "simple" && developerPanelViews.has(view)) navigate("overview");
  }

  function toggleTheme() {
    setTheme((current) => {
      const next = current === "dark" ? "light" : "dark";
      window.localStorage.setItem("mycellios.theme", next);
      return next;
    });
  }

  function navigate(next: PanelView) {
    setView(next);
    setMenuOpen(false);
    if (!desktop) {
      const path = mobileEntry ? "/mobile/" : next === "join" ? "/join" : next === "downloads" ? "/downloads" : next === "admin" ? "/admin" : "/network";
      const query = mobileEntry
        ? next === "contribute" ? "" : `?view=${next}`
        : next === "overview" || next === "join" || next === "downloads" || next === "admin" ? "" : `?view=${next}`;
      window.history.replaceState({}, "", `${path}${query}`);
      applySeoMetadata(path);
    }
  }

  function closeNavigation() {
    setMenuOpen(false);
    window.requestAnimationFrame(() => menuButtonRef.current?.focus());
  }

  /**
   * Credenciales para las mutaciones de estado compartido.
   *
   * Expulsar un worker saca capacidad de la red, así que el coordinador exige
   * autorización para ello igual que para publicar un modelo. Antes esas dos
   * rutas no tenían guard y estas llamadas iban sin credencial; al cerrarlas,
   * mandarlas desnudas devuelve 401.
   *
   * La sesión de cuenta manda si existe; el token de administrador vale como
   * alternativa y también como refuerzo cuando hay sesión, que es el mismo
   * orden que usa `requestModel`.
   */
  function administrativeRequestHeaders(): Record<string, string> {
    const token = modelAdminToken.trim();
    if (authSession) {
      return {
        authorization: `Bearer ${authSession.accessToken}`,
        ...(token ? { "x-mycellios-admin-token": token } : {}),
      };
    }
    return token ? { authorization: `Bearer ${token}` } : {};
  }

  async function removeWorker(workerId: string) {
    if (desktopBridge) {
      const next = await desktopBridge.removeWorker(workerId);
      applyDesktopSnapshot(next);
      return;
    }
    const response = await fetch(`/public/v1/workers/${encodeURIComponent(workerId)}`, {
      method: "DELETE",
      headers: administrativeRequestHeaders(),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => null) as { error?: { message?: string } } | null;
      throw new Error(body?.error?.message ?? `HTTP ${response.status}`);
    }
    await refresh();
  }

  async function clearOfflineWorkers() {
    if (desktopBridge) {
      const next = await desktopBridge.clearOfflineWorkers();
      applyDesktopSnapshot(next);
      return;
    }
    const response = await fetch("/public/v1/workers/clear-offline", {
      method: "POST",
      headers: administrativeRequestHeaders(),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => null) as { error?: { message?: string } } | null;
      throw new Error(body?.error?.message ?? `HTTP ${response.status}`);
    }
    await refresh();
  }

  async function listWorkerCredentials(): Promise<WorkerCredentialSummary[]> {
    if (desktopBridge) {
      return desktopBridge.listWorkerCredentials(modelAdminToken.trim() || undefined);
    }
    const response = await fetch("/public/v1/worker-credentials", {
      cache: "no-store",
      headers: administrativeRequestHeaders(),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => null) as {
        error?: { message?: string };
      } | null;
      throw new Error(body?.error?.message ?? `HTTP ${response.status}`);
    }
    const body = await response.json() as { data: WorkerCredentialSummary[] };
    return body.data;
  }

  async function revokeWorkerCredential(
    credential: Pick<WorkerCredentialSummary, "identityKind" | "identityId" | "fingerprint">,
    reason: string,
  ): Promise<void> {
    if (desktopBridge) {
      await desktopBridge.revokeWorkerCredential(
        credential,
        reason,
        modelAdminToken.trim() || undefined,
      );
    } else {
      const response = await fetch(
        `/public/v1/worker-credentials/${encodeURIComponent(credential.identityKind)}/${encodeURIComponent(credential.identityId)}/revoke`,
        {
          method: "POST",
          headers: {
            "content-type": "application/json",
            ...administrativeRequestHeaders(),
          },
          body: JSON.stringify({
            expectedFingerprint: credential.fingerprint,
            reason,
          }),
        },
      );
      if (!response.ok) {
        const body = await response.json().catch(() => null) as {
          error?: { message?: string };
        } | null;
        throw new Error(body?.error?.message ?? `HTTP ${response.status}`);
      }
    }
    await refresh();
  }

  async function requestModel(input: RequestModelInput, adminToken: string) {
    const token = adminToken.trim();
    if (requiresModelAdminToken && !token) throw new Error("Enter the network administrator token.");
    if (token) {
      window.sessionStorage.setItem("mycellios-model-admin-token", token);
      setModelAdminToken(token);
    }
    if (desktopBridge) {
      applyDesktopSnapshot(await desktopBridge.requestModel(input, token));
      return;
    }
    const response = await fetch("/public/v1/requested-models", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(authSession
          ? {
              authorization: `Bearer ${authSession.accessToken}`,
              ...(token ? { "x-mycellios-admin-token": token } : {}),
            }
          : token ? { authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(input),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => null) as { error?: { message?: string } } | null;
      throw new Error(body?.error?.message ?? `HTTP ${response.status}`);
    }
    await refresh();
  }

  const searchHubModels = useCallback(async (input: HubCatalogSearchInput): Promise<HubCatalogPage> => {
    if (desktopBridge) return desktopBridge.searchHubModels(input);
    const parameters = new URLSearchParams({
      q: input.query,
      sort: input.sort ?? "downloads",
      limit: String(input.limit ?? 50),
    });
    if (input.cursor) parameters.set("cursor", input.cursor);
    const response = await fetch(`/public/v1/huggingface-models?${parameters.toString()}`, { cache: "no-store" });
    if (!response.ok) {
      const body = await response.json().catch(() => null) as { error?: { message?: string } } | null;
      throw new Error(body?.error?.message ?? `HTTP ${response.status}`);
    }
    return response.json() as Promise<HubCatalogPage>;
  }, [desktopBridge]);

  async function removeRequestedModel(modelId: string, adminToken: string) {
    const token = adminToken.trim();
    if (requiresModelAdminToken && !token) throw new Error("Enter the network administrator token.");
    if (desktopBridge) {
      applyDesktopSnapshot(await desktopBridge.removeRequestedModel(modelId, token));
      return;
    }
    const response = await fetch(`/public/v1/requested-models/${encodeURIComponent(modelId)}`, {
      method: "DELETE",
      ...((authSession || token)
        ? {
            headers: authSession
              ? {
                  authorization: `Bearer ${authSession.accessToken}`,
                  ...(token ? { "x-mycellios-admin-token": token } : {}),
                }
              : { authorization: `Bearer ${token}` },
          }
        : {}),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    await refresh();
  }

  async function sendPrompt(model: string, messages: ChatMessage[], sessionId: string, onUpdate?: (update: ChatStreamUpdate) => void): Promise<ChatResponse> {
    if (desktopBridge?.streamChat) return desktopBridge.streamChat({ model, messages, sessionId }, onUpdate ?? (() => undefined));
    if (desktopBridge) return desktopBridge.sendChat({ model, messages, sessionId });
    if (authConfig.apiAccessEnabled && !authSession) {
      throw new Error("Sign in to use your account's inference balance.");
    }
    const result = await consumeChatCompletionStreamWithRecovery(
      (_attempt, signal) => fetch("/v1/chat/completions", {
        method: "POST",
        headers: {
          accept: "text/event-stream",
          "content-type": "application/json",
          ...(authSession ? { authorization: `Bearer ${authSession.accessToken}` } : {}),
        },
        body: JSON.stringify({ model, messages, session_id: sessionId, stream: true, max_tokens: 128, temperature: 0, top_p: 1 }),
        signal,
      }),
      model,
      onUpdate ?? (() => undefined),
      { sessionId },
    );
    if (authSession && authConfig.apiAccessEnabled) {
      setApiAccount(await loadApiAccount(authSession.accessToken).catch(() => apiAccount));
    }
    return result;
  }

  const publicOrigin = desktopSnapshot?.coordinatorUrl
    ? desktopSnapshot.coordinatorUrl.replace(/\/$/, "")
    : PUBLIC_COORDINATOR_URL;
  const publicLink = (path: string) => desktop ? `${publicOrigin}${path}` : path;
  const configuredApiBase = authConfig.publicApiBaseUrl ?? "/v1";
  const apiBaseUrl = desktop
    ? `${publicOrigin}/v1`
    : configuredApiBase.startsWith("http")
      ? configuredApiBase
      : `${window.location.origin}${configuredApiBase.startsWith("/") ? "" : "/"}${configuredApiBase}`;
  const externalProps = desktop ? { target: "_blank", rel: "noreferrer" } as const : {};

  async function disconnectAccount() {
    if (authSession) await signOut(authConfig, authSession);
    setAuthSession(null);
    setNetworkIdentity(null);
    setApiAccount(null);
  }

  return (
    <div className="public-panel-viewport">
      <div className={`public-panel theme-${theme} panel-mode-${panelMode} ${desktop ? `desktop-panel platform-${desktopSnapshot?.platform ?? "loading"}` : ""}`}>
      <aside id="panel-navigation" className={menuOpen ? "panel-sidebar open" : "panel-sidebar"} aria-label="Primary navigation">
        <button ref={sidebarCloseButtonRef} className="panel-sidebar-close" aria-label="Close navigation" onClick={closeNavigation}><X size={18} /></button>
        <a className="panel-brand" href={publicLink("/")} {...externalProps}><img src={brandIcon} alt="" /><div><strong>mycellios</strong><span>network control</span></div></a>
        <nav>
          <span className="panel-nav-label">{panelMode === "developer" ? "DEVELOPER TOOLS" : "MYCELLIOS"}</span>
          {navItems.map(({ id, label, icon: Icon }) => (
            <button key={id} className={view === id ? "active" : ""} title={label} aria-label={label} onClick={() => navigate(id)}>
              <Icon size={18} /><span>{label}</span>{id === "nodes" && <b>{snapshot.summary.connected}</b>}
            </button>
          ))}
        </nav>
        <button
          type="button"
          className={`panel-mode-toggle ${panelMode === "developer" ? "active" : ""}`}
          aria-pressed={panelMode === "developer"}
          title={panelMode === "developer" ? "Switch to the simple panel" : "Show developer tools"}
          onClick={() => changePanelMode(panelMode === "developer" ? "simple" : "developer")}
        >
          <Code2 size={18} />
          <span><strong>Developer mode</strong><small>{panelMode === "developer" ? "Advanced tools visible" : "Simple panel active"}</small></span>
          <i aria-hidden="true" />
        </button>
        <div className={`panel-sidebar-status ${coordinatorError ? "degraded" : "healthy"}`}>
          <i />
          <div>
            <span>Network Status</span>
            <strong>{coordinatorError ? "Degraded" : "Healthy"}</strong>
            <small>{coordinatorError ? "Coordinator unavailable" : "Coordinator operational"}</small>
          </div>
        </div>
      </aside>
      {menuOpen && <button className="panel-menu-backdrop" aria-label="Close navigation overlay" onClick={closeNavigation} />}

      <div className="panel-main">
        <header className="panel-topbar">
          <button
            ref={menuButtonRef}
            className="panel-menu-button"
            aria-label={menuOpen ? "Close navigation" : "Show navigation labels"}
            aria-controls="panel-navigation"
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((current) => !current)}
          ><Menu /></button>
          <div className="panel-topbar-status"><span className={`panel-live-dot ${coordinatorError ? "degraded" : ""}`} /><div><strong>{coordinatorError ? "Network degraded" : "Network healthy"}</strong><small>{snapshot.capturedAt === EMPTY.capturedAt ? "Waiting for snapshot" : `Updated ${relativeTime(snapshot.capturedAt)}`}</small></div></div>
          <div className="panel-top-actions">
            {panelMode === "developer" && <div className="panel-balance-group" aria-label="Account balances">
              <span className="panel-balance-badge money" aria-label={`Balance: ${apiAccount?.usd_balance ?? "0.00"} dollars`} title="Available monetary balance">${apiAccount?.usd_balance ?? "0.00"}</span>
              <span className="panel-balance-badge tokens" aria-label={`Tokens: ${apiAccount?.token_balance ?? 0}`} title="Available inference tokens"><Coins size={14} />{formatCompactTokens(apiAccount?.token_balance ?? 0)} TOK</span>
            </div>}
            {panelMode === "developer" && <button className="panel-theme-indicator" type="button" onClick={toggleTheme} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`} title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}><SunMoon size={16} /></button>}
            {panelMode === "developer" && localProductionProxy && <span className="panel-environment-badge"><Globe2 size={13} /> Production via local</span>}
            {desktopSnapshot?.update.state === "ready" && <button className="panel-update-ready" onClick={() => void desktopBridge?.installUpdate()} title="Restart and install update"><Download size={15} /></button>}
            {panelMode === "developer" && desktopSnapshot && <span title={buildIdentityTitle(desktopSnapshot.buildIdentity)}>App {shortBuildIdentity(desktopSnapshot.buildIdentity)}</span>}
            {panelMode === "developer" && <span title={buildIdentityTitle(snapshot.buildIdentity)}>API {shortBuildIdentity(snapshot.buildIdentity)}</span>}
            {!desktop && authConfig.enabled && (
              networkIdentity
                ? <button className="panel-account-button signed-in" onClick={() => setAuthOpen(true)} title={networkIdentity.email ?? "Mycellios account"}><UserRound size={16} /><span>{networkIdentity.role ?? "member"}</span></button>
                : <button className="panel-account-button" onClick={() => setAuthOpen(true)}><UserRound size={16} /><span>Sign in</span></button>
            )}
            <JoinQuickMenu
              inviteUrl={desktop ? publicLink("/join") : `${window.location.origin}/join`}
              browserUrl={publicLink("/mobile/?autostart=1")}
              downloadsUrl={publicLink("/downloads")}
              external={desktop}
              onOpenJoin={() => navigate("join")}
            />
            {panelMode === "developer" && !desktop && <a href={mobileEntry ? "/mobile/" : "/network?view=contribute"}>This device <Zap size={15} /></a>}
            {desktopBridge && <div className="panel-window-controls"><button onClick={() => void desktopBridge.minimizeWindow()} aria-label="Minimize"><Minus size={16} /></button><button onClick={() => void desktopBridge.toggleMaximizeWindow()} aria-label="Maximize"><Maximize2 size={15} /></button><button className="close" onClick={() => void desktopBridge.closeWindow()} aria-label="Close"><X size={16} /></button></div>}
          </div>
        </header>

        {issue && <div className="panel-error" title={issue.message}><CircleAlert size={17} /> {issue.source === "coordinator" ? "Coordinator unavailable." : "This node could not join the network."} Retrying automatically.</div>}
        <main className="panel-content">
          {loading && snapshot.capturedAt === EMPTY.capturedAt ? <PanelLoading /> : (
            <>
              {view === "overview" && <Overview snapshot={snapshot} onNavigate={navigate} publicLink={publicLink} external={desktop} apiBaseUrl={apiBaseUrl} joinUrl={desktop ? publicLink("/join") : `${window.location.origin}/join`} localAcceleration={desktopSnapshot?.acceleration} localContributionState={desktopSnapshot?.contribution.state} localComputeMode={desktopSnapshot?.settings.computeMode} developerMode={panelMode === "developer"} />}
              {view === "history" && <NetworkHistory endpoint={desktop ? `${publicOrigin}/public/v1/history` : "/public/v1/history"} compact={panelMode === "simple"} />}
              {view === "nodes" && <Nodes snapshot={snapshot} onRemove={removeWorker} onClearOffline={clearOfflineWorkers} onListCredentials={listWorkerCredentials} onRevokeCredential={revokeWorkerCredential} />}
              {view === "models" && <Models snapshot={snapshot} onSearch={searchHubModels} onRequest={requestModel} onRemove={removeRequestedModel} adminToken={modelAdminToken} requiresAdminToken={requiresModelAdminToken} secureTokenStorage={desktop} />}
              {view === "jobs" && <Jobs snapshot={snapshot} />}
              {view === "tests" && <Tests bridge={desktopBridge} />}
              {view === "logs" && <SystemLogs snapshot={snapshot} bridge={desktopBridge} connectionError={coordinatorError} />}
              {view === "inference" && <Inference snapshot={snapshot} onSend={sendPrompt} onNavigate={navigate} developerMode={panelMode === "developer"} accountAuthenticated={desktop || !authConfig.apiAccessEnabled || authSession !== null} onSignIn={() => setAuthOpen(true)} apiAccessEnabled={authConfig.apiAccessEnabled ?? false} apiBaseUrl={apiBaseUrl} accessToken={authSession?.accessToken ?? null} apiAccount={apiAccount} />}
              {view === "contribute" && <Contribute />}
              {view === "join" && <JoinNetwork publicLink={publicLink} external={desktop} />}
              {view === "downloads" && <Downloads publicLink={publicLink} external={desktop} />}
              {view === "admin" && <AssistantAdmin
                apiOrigin={desktop ? publicOrigin : ""}
                adminToken={modelAdminToken}
                authSession={authSession}
                {...(desktopBridge ? { desktopBridge } : {})}
                onAdminTokenChange={(token) => {
                  setModelAdminToken(token);
                  if (token) window.sessionStorage.setItem("mycellios-model-admin-token", token);
                  else window.sessionStorage.removeItem("mycellios-model-admin-token");
                }}
              />}
              {view === "machine" && desktopSnapshot && desktopBridge && <Machine snapshot={desktopSnapshot} bridge={desktopBridge} onSnapshot={applyDesktopSnapshot} />}
              {view === "settings" && desktopSnapshot && desktopBridge && <DesktopSettingsView snapshot={desktopSnapshot} bridge={desktopBridge} onSnapshot={applyDesktopSnapshot} developerMode={panelMode === "developer"} />}
            </>
          )}
        </main>
      </div>
      {desktopSnapshot && desktopBridge && !desktopSnapshot.settings.onboardingComplete && <DesktopOnboarding snapshot={desktopSnapshot} bridge={desktopBridge} onSnapshot={applyDesktopSnapshot} />}
      {!desktop && authOpen && <AccountModal
        config={authConfig}
        session={authSession}
        identity={networkIdentity}
        account={apiAccount}
        onAuthenticated={(session, identity) => {
          setAuthSession(session);
          setNetworkIdentity(identity);
          if (authConfig.apiAccessEnabled) {
            void loadApiAccount(session.accessToken).then(setApiAccount);
          }
          setAuthOpen(false);
        }}
        onSignOut={() => void disconnectAccount().finally(() => setAuthOpen(false))}
        onClose={() => setAuthOpen(false)}
      />}
      </div>
      <SupportAssistant
        surface="panel"
        apiOrigin={desktop ? publicOrigin : ""}
        {...(desktopBridge ? { desktopBridge } : {})}
        onNavigate={(destination) => navigate(
          destination === "machine"
            ? desktop ? "machine" : "contribute"
            : destination,
        )}
        {...(desktopSnapshot && desktopBridge ? {
          deviceControl: {
            currentMode: desktopSnapshot.settings.computeMode,
            onChange: async (computeMode: DesktopSettings["computeMode"]) => {
              const next = await desktopBridge.saveSettings({ ...desktopSnapshot.settings, computeMode });
              applyDesktopSnapshot(next);
            },
          },
        } : {})}
      />
    </div>
  );
}

function JoinQuickMenu({
  inviteUrl,
  browserUrl,
  downloadsUrl,
  external,
  onOpenJoin,
}: {
  inviteUrl: string;
  browserUrl: string;
  downloadsUrl: string;
  external: boolean;
  onOpenJoin: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [copyState, setCopyState] = useState<"idle" | "copied" | "error">("idle");
  const menuId = useId();
  const rootRef = useRef<HTMLDivElement>(null);
  const copyResetRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const externalProps = external ? { target: "_blank", rel: "noreferrer" } as const : {};

  useEffect(() => {
    if (!open) return;
    const closeOnOutsidePress = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", closeOnOutsidePress);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsidePress);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  useEffect(() => () => {
    if (copyResetRef.current) clearTimeout(copyResetRef.current);
  }, []);

  async function copyInvite() {
    try {
      await navigator.clipboard.writeText(inviteUrl);
      setCopyState("copied");
    } catch {
      setCopyState("error");
    }
    if (copyResetRef.current) clearTimeout(copyResetRef.current);
    copyResetRef.current = setTimeout(() => setCopyState("idle"), 2200);
  }

  return (
    <div className={`panel-join-menu ${open ? "open" : ""}`} ref={rootRef}>
      <button
        type="button"
        className="panel-join-trigger"
        aria-label="Join"
        aria-haspopup="dialog"
        aria-controls={menuId}
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        <Share2 size={15} />
        <span>Join</span>
        <ChevronDown className="panel-join-chevron" size={13} />
      </button>
      {open && (
        <section className="panel-join-popover" id={menuId} role="dialog" aria-label="Join or invite">
          <div className="panel-join-popover-intro">
            <span>MESH ACCESS</span>
            <strong>Join or invite</strong>
            <p>Connect a node in seconds or share this public access with someone who wants to contribute compute.</p>
          </div>
          <div className="panel-join-invite">
            <div className="panel-join-invite-head">
              <span>PUBLIC JOIN LINK</span>
              <button type="button" className={copyState} onClick={() => void copyInvite()} aria-live="polite">
                {copyState === "copied" ? <Check size={14} /> : <Copy size={14} />}
                {copyState === "copied" ? "Copied" : copyState === "error" ? "Try again" : "Copy"}
              </button>
            </div>
            <code><span>$</span>{inviteUrl}</code>
            <small>Anyone with this link can choose a browser node or install the desktop client.</small>
          </div>
          <div className="panel-join-quick-actions">
            <button type="button" onClick={() => { setOpen(false); onOpenJoin(); }}>
              <Network size={16} />
              <span><strong>Open Join</strong><small>See every connection option</small></span>
              <ChevronRight size={15} />
            </button>
            <a href={browserUrl} {...externalProps}>
              <Smartphone size={16} />
              <span><strong>Browser node</strong><small>Connect without installing</small></span>
              <ExternalLink size={14} />
            </a>
            <a href={downloadsUrl} {...externalProps}>
              <Download size={16} />
              <span><strong>Install client</strong><small>Windows, macOS or Linux</small></span>
              <ChevronRight size={15} />
            </a>
          </div>
          <a className="panel-join-guide" href="https://github.com/tych0s/mycellios" target="_blank" rel="noreferrer">
            <GitFork size={14} /> Setup &amp; contribute on GitHub <ExternalLink size={12} />
          </a>
        </section>
      )}
    </div>
  );
}

function AccountModal({
  config,
  session,
  identity,
  account,
  onAuthenticated,
  onSignOut,
  onClose,
}: {
  config: PublicAuthConfig;
  session: AuthSession | null;
  identity: NetworkIdentity | null;
  account: ApiAccount | null;
  onAuthenticated: (session: AuthSession, identity: NetworkIdentity) => void;
  onSignOut: () => void;
  onClose: () => void;
}) {
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [providerBusy, setProviderBusy] = useState<"x" | "metamask" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [accountTab, setAccountTab] = useState<"account" | "usage">("account");
  const [usage, setUsage] = useState<ApiUsage[]>([]);
  const [usageBusy, setUsageBusy] = useState(false);

  useEffect(() => {
    if (!session || accountTab !== "usage") return;
    setUsageBusy(true);
    void loadApiUsage(session.accessToken).then(setUsage).catch((caught) => setError(errorText(caught))).finally(() => setUsageBusy(false));
  }, [accountTab, session]);

  const usageByModel = useMemo(() => {
    const totals = new Map<string, { requests: number; tokens: number }>();
    for (const entry of usage) {
      const current = totals.get(entry.model) ?? { requests: 0, tokens: 0 };
      totals.set(entry.model, { requests: current.requests + 1, tokens: current.tokens + entry.total_tokens });
    }
    return [...totals.entries()].sort((left, right) => right[1].tokens - left[1].tokens);
  }, [usage]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const nextSession = mode === "signin"
        ? await signIn(config, email.trim(), password)
        : await signUp(config, email.trim(), password);
      const nextIdentity = await loadNetworkIdentity(nextSession.accessToken);
      onAuthenticated(nextSession, nextIdentity);
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setBusy(false);
    }
  }

  function continueWithX() {
    setProviderBusy("x");
    setError(null);
    try {
      signInWithX(config);
    } catch (caught) {
      setProviderBusy(null);
      setError(errorText(caught));
    }
  }

  async function continueWithMetaMask() {
    setProviderBusy("metamask");
    setError(null);
    try {
      const nextSession = await signInWithMetaMask(config);
      const nextIdentity = await loadNetworkIdentity(nextSession.accessToken);
      onAuthenticated(nextSession, nextIdentity);
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setProviderBusy(null);
    }
  }

  return <div className="modal-backdrop account-modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) onClose(); }}>
    <section className="account-modal" role="dialog" aria-modal="true" aria-label="Mycellios account">
      <button className="account-modal-close" aria-label="Close" onClick={onClose}><X size={18} /></button>
      <div className="account-modal-brand"><img src={brandIcon} alt="" /><span>MYCELLIOS ID</span></div>
      {session && identity ? <>
        <div className="account-detail-tabs"><button className={accountTab === "account" ? "active" : ""} onClick={() => setAccountTab("account")}>Account</button><button className={accountTab === "usage" ? "active" : ""} onClick={() => setAccountTab("usage")}>Usage</button></div>
        {accountTab === "account" ? <><h2>Your network identity</h2><p>History and permissions follow this account across devices.</p><div className="account-identity-card"><UserRound /><span><strong>{identity.email ?? session.user.email ?? identity.id}</strong><small>{identity.role ?? "member"} · public network</small></span></div><div className="account-facts"><span><small>MEMBER SINCE</small><strong>{account ? new Date(account.created_at).toLocaleDateString() : "—"}</strong></span><span><small>PROMPTS SENT</small><strong>{account?.request_count ?? 0}</strong></span><span><small>AVAILABLE</small><strong>{formatCompactTokens(account?.token_balance ?? 0)} tokens</strong></span></div><button className="account-signout" onClick={onSignOut}><LogOut size={16} />Sign out</button></> : <><h2>Usage</h2><p>Real requests and tokens across chat and the API.</p><div className="account-usage-summary"><span><small>REQUESTS</small><strong>{account?.request_count ?? 0}</strong></span><span><small>INPUT TOKENS</small><strong>{formatCompactTokens(account?.lifetime_input_tokens ?? 0)}</strong></span><span><small>OUTPUT TOKENS</small><strong>{formatCompactTokens(account?.lifetime_output_tokens ?? 0)}</strong></span></div><div className="account-usage-models"><span>BY MODEL</span>{usageBusy ? <div className="account-usage-empty"><LoaderCircle className="spin" />Loading usage…</div> : usageByModel.length === 0 ? <div className="account-usage-empty">No usage recorded yet.</div> : usageByModel.map(([model, total]) => <div key={model}><strong>{model}</strong><span>{total.requests} request{total.requests === 1 ? "" : "s"}</span><b>{formatCompactTokens(total.tokens)} tokens</b></div>)}</div></>}
      </> : <>
        <h2>{mode === "signin" ? "Welcome back" : "Create your account"}</h2>
        <p>Use one identity for network access, history and administration.</p>
        <div className="account-provider-actions">
          <button type="button" className="account-provider-x" disabled={providerBusy !== null || busy} onClick={continueWithX}>
            {providerBusy === "x" ? <LoaderCircle className="spin" /> : <span aria-hidden="true">𝕏</span>}
            Continue with X
          </button>
          <div className="account-provider-divider"><i />or<i /></div>
          <button type="button" className="account-provider-wallet" disabled={providerBusy !== null || busy} onClick={() => void continueWithMetaMask()}>
            {providerBusy === "metamask" ? <LoaderCircle className="spin" /> : <WalletCards />}
            MetaMask
          </button>
        </div>
        <div className="account-mode-tabs"><button className={mode === "signin" ? "active" : ""} onClick={() => setMode("signin")}>Sign in</button><button className={mode === "signup" ? "active" : ""} onClick={() => setMode("signup")}>Create account</button></div>
        <form onSubmit={(event) => void submit(event)}>
          <label>Email<input type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} required /></label>
          <label>Password<input type="password" autoComplete={mode === "signin" ? "current-password" : "new-password"} minLength={8} value={password} onChange={(event) => setPassword(event.target.value)} required /></label>
          {error && <div className="account-auth-error"><CircleAlert size={16} />{error}</div>}
          <button className="account-submit" disabled={busy}>{busy ? <LoaderCircle className="spin" /> : <ShieldCheck />}{busy ? "Connecting…" : mode === "signin" ? "Sign in securely" : "Create account"}</button>
        </form>
      </>}
    </section>
  </div>;
}

interface AssistantAdminSettings {
  enabled: boolean;
  modelId: string | null;
  systemPrompt: string;
  welcomeMessage: string;
  suggestions: string[];
  maxOutputTokens: number;
  temperature: number;
  allowDeviceControl: boolean;
  updatedAt: number;
}

interface AssistantAdminRuntime {
  enabled: boolean;
  available: boolean;
  provider: "mycellios-network";
  configuredModel: string | null;
  selectedModel: string | null;
  availableModels: string[];
  welcomeMessage: string;
  suggestions: string[];
  allowDeviceControl: boolean;
  updatedAt: string | null;
}

interface AssistantAdminResponse {
  settings: AssistantAdminSettings;
  runtime: AssistantAdminRuntime;
}

function AssistantAdmin({
  apiOrigin,
  adminToken,
  authSession,
  desktopBridge,
  onAdminTokenChange,
}: {
  apiOrigin: string;
  adminToken: string;
  authSession: AuthSession | null;
  desktopBridge?: DesktopBridge;
  onAdminTokenChange: (token: string) => void;
}) {
  const [draftToken, setDraftToken] = useState(adminToken);
  const [settings, setSettings] = useState<AssistantAdminSettings | null>(null);
  const [runtime, setRuntime] = useState<AssistantAdminRuntime | null>(null);
  const [fleet, setFleet] = useState<FleetContributionStatus | null>(null);
  const [fleetAction, setFleetAction] = useState<"activate" | "pause" | null>(null);
  const [fleetConfirmation, setFleetConfirmation] = useState<"activate" | "pause" | null>(null);
  const [fleetResult, setFleetResult] = useState<FleetContributionCommandResponse | null>(null);
  const [fleetError, setFleetError] = useState<string | null>(null);
  const [suggestionsText, setSuggestionsText] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  function authorizationHeaders(token = draftToken): Record<string, string> {
    const trimmed = token.trim();
    if (authSession) {
      return {
        authorization: `Bearer ${authSession.accessToken}`,
        ...(trimmed ? { "x-mycellios-admin-token": trimmed } : {}),
      };
    }
    return trimmed ? { authorization: `Bearer ${trimmed}` } : {};
  }

  function applyResponse(response: AssistantAdminResponse) {
    setSettings(response.settings);
    setRuntime(response.runtime);
    setSuggestionsText(response.settings.suggestions.join("\n"));
  }

  async function requestFleetStatus(token = draftToken): Promise<FleetContributionStatus> {
    if (desktopBridge) {
      return desktopBridge.getFleetContributionAdmin(token.trim() || undefined);
    }
    return fetch(`${apiOrigin}/public/v1/admin/fleet-contribution`, {
      cache: "no-store",
      headers: authorizationHeaders(token),
    }).then(async (response) => {
      if (!response.ok) {
        const body = await response.json().catch(() => null) as { error?: { message?: string } } | null;
        throw new Error(body?.error?.message ?? `HTTP ${response.status}`);
      }
      return response.json() as Promise<FleetContributionStatus>;
    });
  }

  async function load(token = draftToken) {
    setLoading(true);
    setError(null);
    setSaved(false);
    try {
      const assistantRequest = desktopBridge
        ? desktopBridge.getSupportAssistantAdmin(token.trim() || undefined)
        : fetch(`${apiOrigin}/public/v1/admin/assistant`, {
            cache: "no-store",
            headers: authorizationHeaders(token),
          }).then(async (response) => {
            if (!response.ok) {
              const body = await response.json().catch(() => null) as { error?: { message?: string } } | null;
              throw new Error(body?.error?.message ?? `HTTP ${response.status}`);
            }
            return response.json() as Promise<AssistantAdminResponse>;
          });
      const [assistantResult, fleetStatus] = await Promise.all([
        assistantRequest,
        requestFleetStatus(token),
      ]);
      applyResponse(assistantResult);
      setFleet(fleetStatus);
      setFleetError(null);
      if (token.trim()) onAdminTokenChange(token.trim());
    } catch (caught) {
      setSettings(null);
      setRuntime(null);
      setFleet(null);
      setError(errorText(caught));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load(adminToken);
  }, [apiOrigin, authSession?.accessToken, desktopBridge]);

  useEffect(() => {
    if (!settings) return;
    const interval = window.setInterval(() => {
      void requestFleetStatus().then(setFleet).catch(() => undefined);
    }, 5_000);
    return () => window.clearInterval(interval);
  }, [settings, apiOrigin, authSession?.accessToken, desktopBridge, draftToken]);

  function update<K extends keyof AssistantAdminSettings>(key: K, value: AssistantAdminSettings[K]) {
    setSettings((current) => current ? { ...current, [key]: value } : current);
    setSaved(false);
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    if (!settings) return;
    const suggestions = suggestionsText
      .split("\n")
      .map((suggestion) => suggestion.trim())
      .filter(Boolean)
      .slice(0, 6);
    setSaving(true);
    setError(null);
    setSaved(false);
    try {
      const payload = {
        enabled: settings.enabled,
        modelId: settings.modelId,
        systemPrompt: settings.systemPrompt,
        welcomeMessage: settings.welcomeMessage,
        suggestions,
        maxOutputTokens: settings.maxOutputTokens,
        temperature: settings.temperature,
        allowDeviceControl: settings.allowDeviceControl,
      };
      const result = desktopBridge
        ? await desktopBridge.saveSupportAssistantAdmin(payload, draftToken.trim() || undefined)
        : await fetch(`${apiOrigin}/public/v1/admin/assistant`, {
            method: "PUT",
            headers: {
              "content-type": "application/json",
              ...authorizationHeaders(),
            },
            body: JSON.stringify(payload),
          }).then(async (response) => {
            if (!response.ok) {
              const body = await response.json().catch(() => null) as { error?: { message?: string } } | null;
              throw new Error(body?.error?.message ?? `HTTP ${response.status}`);
            }
            return response.json() as Promise<AssistantAdminResponse>;
          });
      applyResponse(result);
      if (draftToken.trim()) onAdminTokenChange(draftToken.trim());
      setSaved(true);
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setSaving(false);
    }
  }

  async function setFleetContribution(enabled: boolean) {
    const action = enabled ? "activate" : "pause";
    setFleetAction(action);
    setFleetError(null);
    setFleetResult(null);
    try {
      const result = desktopBridge
        ? await desktopBridge.setFleetContributionAdmin(
            enabled,
            draftToken.trim() || undefined,
          )
        : await fetch(`${apiOrigin}/public/v1/admin/fleet-contribution`, {
            method: "POST",
            headers: {
              "content-type": "application/json",
              ...authorizationHeaders(),
            },
            body: JSON.stringify({ enabled }),
          }).then(async (response) => {
            if (!response.ok) {
              const body = await response.json().catch(() => null) as { error?: { message?: string } } | null;
              throw new Error(body?.error?.message ?? `HTTP ${response.status}`);
            }
            return response.json() as Promise<FleetContributionCommandResponse>;
          });
      setFleet(result.status);
      setFleetResult(result);
      setFleetConfirmation(null);
      if (draftToken.trim()) onAdminTokenChange(draftToken.trim());
    } catch (caught) {
      setFleetError(errorText(caught));
    } finally {
      setFleetAction(null);
    }
  }

  const modelOptions = [
    { value: "", label: "Automatic · first available network model" },
    ...(runtime?.availableModels ?? []).map((model) => ({ value: model, label: model })),
  ];
  if (settings?.modelId && !modelOptions.some((option) => option.value === settings.modelId)) {
    modelOptions.push({ value: settings.modelId, label: `${settings.modelId} · currently offline` });
  }
  const confirmedFleetResults = fleetResult?.results.filter(
    (result) => result.state === "applied" || result.state === "unchanged",
  ).length ?? 0;
  const failedFleetResults = (fleetResult?.results.length ?? 0) - confirmedFleetResults;

  return (
    <section className="assistant-admin-page">
      <PageTitle
        eyebrow="NETWORK ADMINISTRATION"
        title="Network administration"
        copy="Control connected desktop contributors and the support assistant from one protected workspace."
        actions={<span className="assistant-network-only"><ShieldCheck size={15} /> Owner controls</span>}
      />

      <article className="assistant-admin-auth">
        <div className="assistant-admin-auth-copy">
          <ShieldCheck size={21} />
          <span><strong>Administrator authorization</strong><small>{authSession ? "Your signed-in network role is used first. A token can also be supplied if needed." : "Enter the network administrator token to read or change private network controls."}</small></span>
        </div>
        <div className="assistant-admin-auth-control">
          <input
            type="password"
            autoComplete="off"
            spellCheck={false}
            value={draftToken}
            onChange={(event) => setDraftToken(event.target.value)}
            placeholder="Administrator token"
            aria-label="Assistant administrator token"
          />
          <button type="button" disabled={loading || (!draftToken.trim() && !authSession)} onClick={() => void load()}>
            {loading ? <LoaderCircle className="spin" size={16} /> : <ShieldCheck size={16} />}Unlock
          </button>
        </div>
      </article>

      {error && <div className="assistant-admin-error"><CircleAlert size={17} /><span>{error}</span></div>}

      {loading && !settings ? (
        <div className="assistant-admin-loading"><LoaderCircle className="spin" /><strong>Loading private assistant configuration…</strong></div>
      ) : settings && runtime && fleet ? (
        <div className="assistant-admin-content">
          <article className="fleet-control-card" aria-live="polite">
            <div className="fleet-control-heading">
              <div className="assistant-admin-section-icon"><Server size={21} /></div>
              <div>
                <span>DESKTOP CONTRIBUTION</span>
                <strong>Control every reachable node</strong>
                <small>Only computers with Mycellios open and the compatible control channel can acknowledge these commands.</small>
              </div>
              <button
                type="button"
                className="fleet-refresh"
                disabled={fleetAction !== null}
                onClick={() => void requestFleetStatus().then(setFleet).catch((caught) => setFleetError(errorText(caught)))}
                aria-label="Refresh contribution status"
              >
                <RefreshCw size={16} />
              </button>
            </div>

            <div className="fleet-control-summary">
              <div><span>CONTRIBUTING</span><strong className="fleet-positive">{fleet.summary.contributing}</strong><small>Available to the scheduler</small></div>
              <div><span>PAUSED</span><strong>{fleet.summary.paused}</strong><small>Connected, accepting no work</small></div>
              <div><span>CONTROL READY</span><strong>{fleet.summary.controlReady}</strong><small>Can receive this command now</small></div>
              <div><span>OUT OF REACH</span><strong>{fleet.summary.offline + fleet.summary.unsupported}</strong><small>Offline or requires an update</small></div>
            </div>

            <div className="fleet-control-actions">
              <div>
                <strong>{fleet.summary.connected} of {fleet.summary.desktopNodes} desktop nodes connected</strong>
                <small>This controls Mycellios contribution, not the computer's physical power.</small>
              </div>
              <button
                type="button"
                className="fleet-action fleet-action-pause"
                disabled={fleetAction !== null || fleet.summary.controlReady === 0}
                onClick={() => setFleetConfirmation("pause")}
              >
                <Pause size={17} />Pause all
              </button>
              <button
                type="button"
                className="fleet-action fleet-action-activate"
                disabled={fleetAction !== null || fleet.summary.controlReady === 0}
                onClick={() => setFleetConfirmation("activate")}
              >
                <CirclePower size={17} />Activate all
              </button>
            </div>

            {fleetConfirmation && (
              <div className="fleet-confirmation" role="alert">
                <CircleAlert size={19} />
                <div>
                  <strong>{fleetConfirmation === "activate" ? "Activate" : "Pause"} {fleet.summary.controlReady} reachable nodes?</strong>
                  <small>{fleetConfirmation === "activate" ? "They will begin accepting compatible network work." : "Active work may be interrupted and no new work will be accepted."}</small>
                </div>
                <button type="button" className="secondary" onClick={() => setFleetConfirmation(null)}>Cancel</button>
                <button
                  type="button"
                  className={fleetConfirmation === "activate" ? "confirm-activate" : "confirm-pause"}
                  disabled={fleetAction !== null}
                  onClick={() => void setFleetContribution(fleetConfirmation === "activate")}
                >
                  {fleetAction ? <LoaderCircle className="spin" size={16} /> : <Check size={16} />}
                  Confirm
                </button>
              </div>
            )}

            {fleetError && <div className="fleet-control-message error"><CircleAlert size={16} />{fleetError}</div>}
            {fleetResult && (
              <div className={`fleet-control-message ${failedFleetResults > 0 ? "warning" : "success"}`}>
                {failedFleetResults > 0 ? <CircleAlert size={16} /> : <CheckCircle2 size={16} />}
                <span>
                  <strong>{confirmedFleetResults} of {fleetResult.results.length} nodes confirmed the command.</strong>
                  <small>{failedFleetResults > 0 ? `${failedFleetResults} did not acknowledge it; their state was not assumed.` : "Every targeted node reported its resulting state."}</small>
                </span>
              </div>
            )}
          </article>

          <form className="assistant-admin-form" onSubmit={(event) => void save(event)}>
          <div className="assistant-admin-status-grid">
            <article>
              <span>PUBLIC STATUS</span>
              <strong className={runtime.available ? "healthy" : "offline"}><i />{!runtime.enabled ? "Disabled" : runtime.available ? "Online" : "Waiting for a model"}</strong>
              <small>The widget never falls back to an external AI.</small>
            </article>
            <article>
              <span>MODEL IN USE</span>
              <strong>{runtime.selectedModel ?? "None connected"}</strong>
              <small>{runtime.configuredModel ? "Fixed by the administrator" : "Selected automatically from live models"}</small>
            </article>
            <article>
              <span>LIVE OPTIONS</span>
              <strong>{runtime.availableModels.length}</strong>
              <small>Real non-test models announced by connected nodes.</small>
            </article>
          </div>

          <article className="assistant-admin-section assistant-admin-primary">
            <div className="assistant-admin-section-heading">
              <div className="assistant-admin-section-icon"><Bot size={21} /></div>
              <span><strong>Assistant availability and model</strong><small>Choose Automatic to keep support online when network capacity changes.</small></span>
            </div>
            <div className="assistant-admin-field-grid">
              <label>
                <span>NETWORK MODEL</span>
                <AppSelect
                  ariaLabel="Support assistant network model"
                  value={settings.modelId ?? ""}
                  options={modelOptions}
                  onChange={(value) => update("modelId", value || null)}
                />
              </label>
              <div className="assistant-admin-toggle-stack">
                <Toggle label="Show assistant publicly" checked={settings.enabled} onChange={(value) => update("enabled", value)} />
                <Toggle label="Offer confirmed device actions" checked={settings.allowDeviceControl} onChange={(value) => update("allowDeviceControl", value)} />
              </div>
            </div>
          </article>

          <article className="assistant-admin-section">
            <div className="assistant-admin-section-heading">
              <div className="assistant-admin-section-icon"><MessageSquareText size={21} /></div>
              <span><strong>Conversation</strong><small>These texts are visible to users. One suggested question per line, up to six.</small></span>
            </div>
            <label className="assistant-admin-full-field">
              <span>WELCOME MESSAGE</span>
              <textarea rows={3} value={settings.welcomeMessage} onChange={(event) => update("welcomeMessage", event.target.value)} />
            </label>
            <label className="assistant-admin-full-field">
              <span>SUGGESTED QUESTIONS</span>
              <textarea rows={4} value={suggestionsText} onChange={(event) => { setSuggestionsText(event.target.value); setSaved(false); }} />
            </label>
          </article>

          <article className="assistant-admin-section">
            <div className="assistant-admin-section-heading">
              <div className="assistant-admin-section-icon"><Sparkles size={21} /></div>
              <span><strong>Project knowledge and behaviour</strong><small>The coordinator adds verified live network data separately on every request.</small></span>
            </div>
            <label className="assistant-admin-full-field">
              <span>PRIVATE SYSTEM INSTRUCTIONS</span>
              <textarea className="assistant-admin-system-prompt" rows={12} value={settings.systemPrompt} onChange={(event) => update("systemPrompt", event.target.value)} />
              <small>Never place passwords or private tokens here. The instructions are sent only to the selected network model for each support request.</small>
            </label>
            <div className="assistant-admin-number-grid">
              <label><span>MAXIMUM OUTPUT TOKENS</span><input type="number" min={64} max={4096} step={64} value={settings.maxOutputTokens} onChange={(event) => update("maxOutputTokens", Number(event.target.value))} /></label>
              <label><span>TEMPERATURE</span><input type="number" min={0} max={1.5} step={0.1} value={settings.temperature} onChange={(event) => update("temperature", Number(event.target.value))} /></label>
            </div>
          </article>

          <div className="assistant-admin-savebar">
            <span><ShieldCheck size={15} />Shared model activation remains protected by administrator authorization.</span>
            {saved && <b><Check size={15} />Saved</b>}
            <button type="submit" disabled={saving || settings.systemPrompt.trim().length < 20 || settings.welcomeMessage.trim().length < 10 || suggestionsText.trim().length < 3}>
              {saving ? <LoaderCircle className="spin" size={17} /> : <Check size={17} />}{saving ? "Saving…" : "Save assistant"}
            </button>
          </div>
          </form>
        </div>
      ) : (
        <article className="assistant-admin-locked">
          <ShieldCheck size={26} />
          <div><strong>Private configuration is locked</strong><p>Sign in with an owner, admin or operator account, or enter the administrator token above.</p></div>
        </article>
      )}
    </section>
  );
}

export function LiveNetworkTelemetry({ snapshot, compact = false }: { snapshot: PublicSnapshot; compact?: boolean }) {
  const onlineWorkers = snapshot.workers.filter((worker) => worker.connected && worker.status === "online");
  const activeVramMb = onlineWorkers.reduce((total, worker) => total + worker.offeredVramMb, 0);
  const activeGpus = onlineWorkers.flatMap((worker) => worker.gpus);
  const freeVramMb = activeGpus.reduce((total, gpu) => total + gpu.freeOfferedVramMb, 0);
  const freeRatio = activeVramMb > 0 ? Math.min(1, freeVramMb / activeVramMb) : 0;
  const replicas = snapshot.models.reduce((total, model) => total + model.replicas, 0);
  const pipelines = snapshot.models.reduce((total, model) => total + model.pipelines, 0);
  const inflightJobs = snapshot.jobs.filter((job) => ["queued", "leasing", "running", "streaming"].includes(job.status));
  const runningJobs = inflightJobs.filter((job) => job.status === "running" || job.status === "streaming").length;
  const queuedJobs = inflightJobs.length - runningJobs;
  const snapshotReady = snapshot.capturedAt !== EMPTY.capturedAt;

  return (
    <section className={`live-network-telemetry${compact ? " compact" : ""}`} aria-label="Current live network data" aria-live="polite">
      <div className="snapshot">
        <span><Radio size={12} />Snapshot</span>
        <strong><i />{snapshotReady ? "LIVE" : "WAITING"}</strong>
        <small>{snapshotReady ? `Updated ${relativeTime(snapshot.capturedAt)}` : "Waiting for coordinator"}</small>
      </div>
      {!compact && <div>
        <span><Network size={12} />Coordinator</span>
        <strong>mycellios</strong>
        <small>Public API · v{snapshot.version}</small>
      </div>}
      <div>
        <span><GitFork size={12} />Connected peers</span>
        <strong>{snapshot.summary.connected}</strong>
        <small>{snapshot.summary.online} online · {snapshot.summary.mobile} browser</small>
      </div>
      {!compact && <div>
        <span><Boxes size={12} />Active models</span>
        <strong>{snapshot.models.length}</strong>
        <small>{replicas} replicas · {pipelines} pipelines</small>
      </div>}
      {!compact && <div className="vram">
        <span><MemoryStick size={12} />Mesh VRAM</span>
        <strong>{formatMemory(activeVramMb)}</strong>
        <small>{formatMemory(freeVramMb)} free · {onlineWorkers.length} peers</small>
        <i className="live-telemetry-meter" aria-label={`${Math.round(freeRatio * 100)}% free`}><b style={{ width: `${freeRatio * 100}%` }} /></i>
      </div>}
      <div className="inflight">
        <span><Activity size={12} />Inflight</span>
        <strong>{inflightJobs.length}</strong>
        <small>{runningJobs} running · {queuedJobs} queued</small>
        <i className={inflightJobs.length > 0 ? "active" : ""} />
      </div>
    </section>
  );
}

const historyRangeLabels: Record<NetworkHistoryRange, string> = {
  "24h": "24 hours",
  "7d": "7 days",
  "30d": "30 days",
  "90d": "90 days",
};

export function NetworkHistory({ endpoint, compact = false }: { endpoint: string; compact?: boolean }) {
  const [range, setRange] = useState<NetworkHistoryRange>("24h");
  const [history, setHistory] = useState<NetworkTelemetryHistory | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const load = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const separator = endpoint.includes("?") ? "&" : "?";
      const response = await fetch(`${endpoint}${separator}range=${range}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setHistory(await response.json() as NetworkTelemetryHistory);
      setLoadError(null);
    } catch (caught) {
      setLoadError(errorText(caught));
    } finally {
      if (!silent) setLoading(false);
    }
  }, [endpoint, range]);

  useEffect(() => {
    void load();
    const timer = window.setInterval(() => void load(true), 60_000);
    return () => window.clearInterval(timer);
  }, [load]);

  return <section className="network-history-page">
    <PageTitle
      eyebrow="PERSISTENT TELEMETRY"
      title="Network history"
      copy="Real evolution of shared capacity, nodes, models and tasks. The coordinator stores one sample every ten minutes."
      actions={<button type="button" disabled={loading} onClick={() => void load()}>{loading ? <LoaderCircle className="spin" size={16} /> : <RefreshCw size={16} />}Refresh</button>}
    />
    <div className="network-history-toolbar">
      <div className="network-history-ranges" role="group" aria-label="History range">
        {(Object.keys(historyRangeLabels) as NetworkHistoryRange[]).map((value) => (
          <button
            type="button"
            className={range === value ? "active" : ""}
            aria-pressed={range === value}
            onClick={() => setRange(value)}
            key={value}
          >{historyRangeLabels[value]}</button>
        ))}
      </div>
      <span><History size={15} />Every {history?.intervalMinutes ?? 10} min · retained for {history?.retentionDays ?? 90} days</span>
    </div>
    {loadError && <div className="network-history-error"><CircleAlert size={17} /><span><strong>Could not load network history</strong>{loadError}</span></div>}
    {loading && !history ? <div className="network-history-loading"><LoaderCircle className="spin" /><span>Reading saved samples…</span></div> : history && <NetworkHistoryReport history={history} compact={compact} />}
  </section>;
}

export function NetworkHistoryReport({ history, compact = false }: { history: NetworkTelemetryHistory; compact?: boolean }) {
  const latest = history.samples.at(-1);
  const first = history.samples[0];
  if (!latest) {
    return <Empty icon={History} title="No samples yet" copy="The coordinator will save the first reading automatically and add another every ten minutes." />;
  }
  return <>
    <div className={`network-history-summary${compact ? " compact" : ""}`} aria-label="Latest saved sample">
      <HistorySummaryCard
        icon={MemoryStick}
        label="Active VRAM"
        value={formatMemory(latest.offeredVramMb)}
        detail={`${formatMemory(latest.freeVramMb)} free now`}
        delta={historyDelta(latest.offeredVramMb, first?.offeredVramMb)}
        tone="blue"
      />
      <HistorySummaryCard
        icon={Server}
        label="Connected nodes"
        value={String(latest.connectedNodes)}
        detail={`${latest.onlineNodes} online · ${latest.browserNodes} browser`}
        delta={historyDelta(latest.connectedNodes, first?.connectedNodes)}
        tone="green"
      />
      {!compact && <HistorySummaryCard
        icon={Boxes}
        label="Active models"
        value={String(latest.activeModels)}
        detail={`${latest.modelReplicas} replicas · ${latest.modelPipelines} pipelines`}
        delta={historyDelta(latest.activeModels, first?.activeModels)}
        tone="purple"
      />}
      {!compact && <HistorySummaryCard
        icon={Activity}
        label="Jobs in flight"
        value={String(latest.inflightJobs)}
        detail={`${latest.runningJobs} running`}
        delta={historyDelta(latest.inflightJobs, first?.inflightJobs)}
        tone="orange"
      />}
    </div>
    <div className="network-history-meta">
      <span><Radio />Latest sample <strong>{formatHistoryDate(latest.capturedAt)}</strong></span>
      <span><ShieldCheck />Coordinator data, no estimates</span>
      <span><History />{history.samples.length} samples in {historyRangeLabels[history.range]}</span>
    </div>
    <div className="network-history-charts">
      <NetworkHistoryChart
        title="Network capacity"
        eyebrow="ACTIVE VRAM"
        samples={history.samples}
        unit=""
        valueFormatter={formatMemory}
        series={[
          { key: "offeredVramMb", label: "Active total", tone: "blue" },
          { key: "freeVramMb", label: "Free", tone: "green" },
        ]}
      />
      <NetworkHistoryChart
        title="Node availability"
        eyebrow="PEERS"
        samples={history.samples}
        unit="nodes"
        valueFormatter={(value) => String(Math.round(value))}
        series={[
          { key: "connectedNodes", label: "Connected", tone: "blue" },
          { key: "onlineNodes", label: "Online", tone: "green" },
          { key: "browserNodes", label: "Browser", tone: "purple" },
        ]}
      />
      {!compact && <DetailedHistoryCharts history={history} />}
    </div>
    {compact ? (
      <PanelDisclosure icon={Activity} eyebrow="ON DEMAND" title="Detailed telemetry" summary="Model availability, inference load, and the raw sample table.">
        <div className="network-history-charts network-history-charts-detail"><DetailedHistoryCharts history={history} /></div>
        <NetworkHistoryTable samples={history.samples} />
      </PanelDisclosure>
    ) : <NetworkHistoryTable samples={history.samples} />}
  </>;
}

function DetailedHistoryCharts({ history }: { history: NetworkTelemetryHistory }) {
  return <>
    <NetworkHistoryChart
      title="Available models"
      eyebrow="MODEL FABRIC"
      samples={history.samples}
      unit="models"
      valueFormatter={(value) => String(Math.round(value))}
      series={[
        { key: "activeModels", label: "Models", tone: "purple" },
        { key: "modelReplicas", label: "Replicas", tone: "blue" },
        { key: "modelPipelines", label: "Pipelines", tone: "green" },
      ]}
    />
    <NetworkHistoryChart
      title="Inference load"
      eyebrow="ACTIVITY"
      samples={history.samples}
      unit="jobs"
      valueFormatter={(value) => String(Math.round(value))}
      series={[
        { key: "inflightJobs", label: "In flight", tone: "orange" },
        { key: "runningJobs", label: "Running", tone: "green" },
      ]}
    />
  </>;
}

type HistoryMetricKey =
  | "offeredVramMb"
  | "freeVramMb"
  | "connectedNodes"
  | "onlineNodes"
  | "browserNodes"
  | "activeModels"
  | "modelReplicas"
  | "modelPipelines"
  | "inflightJobs"
  | "runningJobs";

interface HistoryChartSeries {
  key: HistoryMetricKey;
  label: string;
  tone: "blue" | "green" | "purple" | "orange";
}

function NetworkHistoryChart({
  title,
  eyebrow,
  samples,
  series,
  unit,
  valueFormatter,
}: {
  title: string;
  eyebrow: string;
  samples: NetworkTelemetrySample[];
  series: HistoryChartSeries[];
  unit: string;
  valueFormatter: (value: number) => string;
}) {
  const points = downsampleNetworkHistory(samples, 720);
  const width = 760;
  const height = 250;
  const left = 52;
  const right = 24;
  const top = 24;
  const bottom = 44;
  const ceiling = Math.max(
    1,
    ...points.flatMap((sample) => series.map(({ key }) => sample[key])),
  );
  const x = (index: number) => points.length === 1
    ? (left + width - right) / 2
    : left + (index / Math.max(points.length - 1, 1)) * (width - left - right);
  const y = (value: number) => top + (1 - value / ceiling) * (height - top - bottom);
  const labelEvery = Math.max(1, Math.ceil(points.length / 5));
  return <article className="network-history-chart">
    <header><div><span>{eyebrow}</span><h2>{title}</h2></div><div className="network-history-legend">{series.map((item) => <span className={item.tone} key={item.key}><i />{item.label}</span>)}</div></header>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title}. ${samples.length} real samples.`}>
      {[0, 0.5, 1].map((ratio) => <g key={ratio}>
        <line className="history-grid-line" x1={left} y1={top + ratio * (height - top - bottom)} x2={width - right} y2={top + ratio * (height - top - bottom)} />
        <text x={left - 8} y={top + ratio * (height - top - bottom) + 4} textAnchor="end">{valueFormatter(ceiling * (1 - ratio))}</text>
      </g>)}
      {series.map((item) => <g className={`network-history-series ${item.tone}`} key={item.key}>
        {points.length > 1 && <polyline points={points.map((sample, index) => `${x(index)},${y(sample[item.key])}`).join(" ")} />}
        {points.length === 1 && <circle cx={x(0)} cy={y(points[0]![item.key])} r="5"><title>{`${item.label}: ${valueFormatter(points[0]![item.key])}`}</title></circle>}
      </g>)}
      {points.map((sample, index) => {
        if (index % labelEvery !== 0 && index !== points.length - 1) return null;
        return <text className="history-x-label" x={x(index)} y={height - 13} textAnchor="middle" key={sample.capturedAt}>{formatHistoryChartTime(sample.capturedAt)}</text>;
      })}
    </svg>
    <footer><span>{samples.length} samples · every 10 min</span>{series.map((item) => <strong key={item.key}>{item.label} {valueFormatter(points.at(-1)?.[item.key] ?? 0)} {unit && <small>{unit}</small>}</strong>)}</footer>
  </article>;
}

function HistorySummaryCard({
  icon: Icon,
  label,
  value,
  detail,
  delta,
  tone,
}: {
  icon: typeof Network;
  label: string;
  value: string;
  detail: string;
  delta: number;
  tone: "blue" | "green" | "purple" | "orange";
}) {
  return <article className={`network-history-summary-card ${tone}`}>
    <Icon />
    <span><small>{label}</small><strong>{value}</strong><em>{detail}</em></span>
    <b className={delta > 0 ? "positive" : delta < 0 ? "negative" : "neutral"}>{delta > 0 ? "+" : ""}{delta} since start</b>
  </article>;
}

function NetworkHistoryTable({ samples }: { samples: NetworkTelemetrySample[] }) {
  const visible = samples.slice(-24).reverse();
  return <article className="network-history-table-card">
    <header><div><span>EXACT SAMPLES</span><h2>Latest saved readings</h2></div><small>Showing the 24 most recent</small></header>
    <div className="network-history-table-scroll">
      <table>
        <thead><tr><th>Timestamp</th><th>Active VRAM</th><th>Free VRAM</th><th>Nodes</th><th>Online</th><th>Models</th><th>Inflight</th><th>Completed</th></tr></thead>
        <tbody>{visible.map((sample) => <tr key={sample.capturedAt}>
          <td><strong>{formatHistoryDate(sample.capturedAt)}</strong></td>
          <td>{formatMemory(sample.offeredVramMb)}</td>
          <td>{formatMemory(sample.freeVramMb)}</td>
          <td>{sample.connectedNodes}</td>
          <td>{sample.onlineNodes}</td>
          <td>{sample.activeModels}</td>
          <td>{sample.inflightJobs}</td>
          <td>{sample.completedJobs}</td>
        </tr>)}</tbody>
      </table>
    </div>
  </article>;
}

function downsampleNetworkHistory(
  samples: NetworkTelemetrySample[],
  maximum: number,
): NetworkTelemetrySample[] {
  if (samples.length <= maximum) return samples;
  const indexes = new Set<number>([0, samples.length - 1]);
  const step = (samples.length - 1) / (maximum - 1);
  for (let index = 1; index < maximum - 1; index += 1) {
    indexes.add(Math.round(index * step));
  }
  return [...indexes].sort((left, right) => left - right).map((index) => samples[index]!);
}

function historyDelta(current: number, initial: number | undefined): number {
  return initial === undefined ? 0 : Math.round(current - initial);
}

function formatHistoryDate(value: string): string {
  return new Intl.DateTimeFormat("en-US", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function formatHistoryChartTime(value: string): string {
  return new Intl.DateTimeFormat("en-US", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value)).replace(",", "");
}

function DashboardConnect({
  apiBaseUrl,
  joinUrl,
  downloadsUrl,
  browserUrl,
  external,
  onOpenJoin,
}: {
  apiBaseUrl: string;
  joinUrl: string;
  downloadsUrl: string;
  browserUrl: string;
  external: boolean;
  onOpenJoin: () => void;
}) {
  const titleId = useId();
  const [copied, setCopied] = useState<"api" | "join" | null>(null);
  const [copyError, setCopyError] = useState<"api" | "join" | null>(null);
  const resetRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const externalProps = external ? { target: "_blank", rel: "noreferrer" } as const : {};

  useEffect(() => () => {
    if (resetRef.current) clearTimeout(resetRef.current);
  }, []);

  async function copyValue(kind: "api" | "join", value: string) {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(kind);
      setCopyError(null);
    } catch {
      setCopied(null);
      setCopyError(kind);
    }
    if (resetRef.current) clearTimeout(resetRef.current);
    resetRef.current = setTimeout(() => {
      setCopied(null);
      setCopyError(null);
    }, 2200);
  }

  const copyLabel = (kind: "api" | "join") => copied === kind ? "Copied" : copyError === kind ? "Try again" : "Copy";

  return (
    <article className="dashboard-connect" aria-labelledby={titleId}>
      <header>
        <div><strong id={titleId}>Connect</strong><span>· join the public mycellios network</span></div>
        <a href="https://github.com/tych0s/mycellios#readme" target="_blank" rel="noreferrer">Full docs <ArrowRight size={13} /></a>
      </header>
      <div className="dashboard-connect-grid">
        <section>
          <span>1 · INSTALL</span>
          <a className="dashboard-connect-field install" href={downloadsUrl} {...externalProps}>
            <Download size={15} />
            <code>Download mycellios</code>
            <b>open</b>
          </a>
        </section>
        <section>
          <span>API ENDPOINT</span>
          <div className="dashboard-connect-field endpoint">
            <Globe2 size={15} />
            <code>{apiBaseUrl}</code>
            <button type="button" onClick={() => void copyValue("api", apiBaseUrl)} aria-label={`${copyLabel("api")} API endpoint`}>
              {copied === "api" ? <Check size={13} /> : <Copy size={13} />}
              <span>{copied === "api" ? "copied" : "configured target"}</span>
            </button>
          </div>
        </section>
        <section className="dashboard-connect-join">
          <span>2 · JOIN</span>
          <div className="dashboard-connect-command">
            <code><i>$</i>{joinUrl}</code>
            <button type="button" className={copyError === "join" ? "error" : copied === "join" ? "copied" : ""} onClick={() => void copyValue("join", joinUrl)}>
              {copied === "join" ? <Check size={14} /> : <Copy size={14} />}
              {copyLabel("join")}
            </button>
          </div>
          <p>Share the public link, connect in a browser, or install the desktop client. The configured API uses the OpenAI-compatible <code>/v1</code> interface.</p>
        </section>
      </div>
      <footer>
        <a href={browserUrl} {...externalProps}><Smartphone size={14} />Start a browser node <ExternalLink size={12} /></a>
        <button type="button" onClick={onOpenJoin}>View every option <ChevronRight size={13} /></button>
      </footer>
    </article>
  );
}

function Overview({ snapshot, onNavigate, publicLink, external, apiBaseUrl, joinUrl, localAcceleration, localContributionState, localComputeMode, developerMode }: { snapshot: PublicSnapshot; onNavigate: (view: PanelView) => void; publicLink: (path: string) => string; external: boolean; apiBaseUrl: string; joinUrl: string; localAcceleration: AcceleratorProgressSnapshot | undefined; localContributionState: DashboardSnapshot["contribution"]["state"] | undefined; localComputeMode: DesktopSettings["computeMode"] | undefined; developerMode: boolean }) {
  const active = snapshot.workers.filter((worker) => worker.connected);
  const operational = active.filter((worker) => worker.status === "online");
  const activeGpus = operational.flatMap((worker) => worker.gpus);
  const activeDeployments = operational.flatMap((worker) => worker.deployments);
  const telemetryGpus = activeGpus.filter((gpu) => gpu.utilizationPct !== undefined);
  const poweredGpus = activeGpus.filter((gpu) => (gpu.powerW ?? 0) > 0);
  const activeOfferedVramMb = activeGpus.reduce((total, gpu) => total + gpu.offeredVramMb, 0);
  const freeVramMb = Math.min(activeOfferedVramMb, activeGpus.reduce((total, gpu) => total + gpu.freeOfferedVramMb, 0));
  const usedVramMb = Math.max(0, activeOfferedVramMb - freeVramMb);
  const freeRatio = activeOfferedVramMb > 0 ? freeVramMb / activeOfferedVramMb : 0;
  const averageUtilization = telemetryGpus.length > 0
    ? telemetryGpus.reduce((total, gpu) => total + (gpu.utilizationPct ?? 0), 0) / telemetryGpus.length
    : null;
  const observedPowerW = poweredGpus.reduce((total, gpu) => total + (gpu.powerW ?? 0), 0);
  const measuredDeployments = activeDeployments.filter(isMeasuredDeployment);
  const measuredThroughput = measuredDeployments.reduce((total, deployment) => total + deployment.tokensPerSecond, 0);
  const execution = summarizeDeploymentExecution(activeDeployments);
  const nativeNodes = operational.filter((worker) => worker.kind === "desktop").length;
  const browserNodes = operational.filter((worker) => worker.kind === "browser").length;
  const cellNodes = operational.filter((worker) => worker.kind === "cell").length;
  const connectPanel = <DashboardConnect
    apiBaseUrl={apiBaseUrl}
    joinUrl={joinUrl}
    downloadsUrl={publicLink("/downloads")}
    browserUrl={publicLink("/mobile/?autostart=1")}
    external={external}
    onOpenJoin={() => onNavigate("join")}
  />;
  const recentActivity = <article className="activity-card">
    <div className="card-heading"><div><span>RECENT ACTIVITY</span><h2>Network tasks</h2></div>{developerMode && <button onClick={() => onNavigate("jobs")}>View all</button>}</div>
    <div className="activity-list">
      {snapshot.jobs.slice(0, developerMode ? 5 : 3).map((job) => <div key={job.id}><i className={job.status} /><div><strong>{job.model}</strong><span>{shortId(job.id)} · {relativeTime(job.updatedAt)}</span></div><b>{job.status}</b></div>)}
      {snapshot.jobs.length === 0 && <Empty icon={Activity} title="No tasks yet" copy="Tasks will appear here when you test a connected model." />}
    </div>
  </article>;
  return (
    <section className="overview-page">
      <PageTitle eyebrow="NETWORK CONTROL" title="Distributed AI network" copy="A live view of the capacity, models and tasks connected to your mycellios network." actions={developerMode ? <button onClick={() => onNavigate("nodes")}>Manage nodes</button> : undefined} />
      <aside className="overview-announcement" aria-label="Contribute to the public mycellios network">
        <div className="announcement-icon"><Network size={21} /></div>
        <div><strong>Welcome to the public mesh</strong><span>Run open models with shared community capacity. Contribute from this browser or install a local node; you choose the resources and can pause at any time.</span></div>
        <div className="overview-announcement-actions">
          <button onClick={() => onNavigate(external ? "machine" : "contribute")}>Contribute now <ArrowRight size={15} /></button>
          <a href="https://github.com/tych0s/mycellios" target="_blank" rel="noreferrer"><GitFork size={15} />GitHub</a>
        </div>
      </aside>
      <LiveNetworkTelemetry snapshot={snapshot} compact={!developerMode} />
      {localAcceleration && shouldShowAccelerationBanner(localAcceleration) && <AcceleratorCompactBanner acceleration={localAcceleration} contributionState={localContributionState} computeMode={localComputeMode} onOpen={() => onNavigate("machine")} />}
      {developerMode ? connectPanel : <PanelDisclosure icon={Share2} eyebrow="OPTIONAL" title="Connect another device" summary="Open setup links only when you are ready to expand the network.">{connectPanel}</PanelDisclosure>}
      <article className="global-capacity-card">
        <div className="global-capacity-main">
          <div className="global-capacity-heading"><span>GLOBAL NODE CAPACITY</span><b><i /> LIVE</b></div>
          <div className="execution-overview"><ExecutionBadge execution={execution} workers={snapshot.workers} large /><span><strong>Effective inference device</strong><small>Reported by the runtime after loading the model, never inferred from detected hardware.</small></span></div>
          <div className="global-capacity-total"><strong>{formatMemory(activeOfferedVramMb)}</strong><div><b>Active usable memory</b><small>Offered by {operational.length} operational node{operational.length === 1 ? "" : "s"}</small></div></div>
          <div className="global-capacity-bar" aria-label={`${Math.round(freeRatio * 100)}% of offered memory is free`}><i style={{ width: `${freeRatio * 100}%` }} /></div>
          <div className="global-capacity-legend"><span>{formatMemory(freeVramMb)} free now</span><span>{formatMemory(usedVramMb)} in use</span></div>
        </div>
        <div className={`global-capacity-metrics${developerMode ? "" : " compact"}`}>
          <CapacityMetric icon={Cpu} label="Runtime stages" value={execution.verified ? String(execution.unitCount) : "Unverified"} detail={execution.verified ? executionDetail(execution) : `${nativeNodes} native · ${browserNodes} browser · ${cellNodes} cells connected`} />
          {developerMode && <CapacityMetric icon={Gauge} label="Average GPU load" value={averageUtilization === null ? "No telemetry" : `${averageUtilization.toFixed(0)}%`} detail={`${telemetryGpus.length} reporting engine${telemetryGpus.length === 1 ? "" : "s"}`} />}
          <CapacityMetric icon={Zap} label="Measured inference capacity" value={measuredThroughput > 0 ? `${formatCompactNumber(measuredThroughput)} tok/s` : "Not measured"} detail={measuredDeployments.length > 0 ? `${measuredDeployments.length} runtime${measuredDeployments.length === 1 ? "" : "s"} with completed inference` : activeDeployments.length > 0 ? "Measured after the first real inference" : "Requires an active model"} />
          {developerMode && <CapacityMetric icon={Activity} label="Observed GPU draw" value={poweredGpus.length > 0 ? formatPower(observedPowerW) : "No telemetry"} detail={poweredGpus.length > 0 ? `${poweredGpus.length} physical reading${poweredGpus.length === 1 ? "" : "s"}` : telemetryGpus.length > 0 ? "GPU load is visible, but the driver does not expose watts" : "Never estimated"} />}
        </div>
      </article>
      {developerMode && <div className="panel-stat-grid">
        <Stat icon={Radio} label="Active nodes" value={String(snapshot.summary.connected)} detail={`${snapshot.summary.registered} registered`} />
        <Stat icon={HardDrive} label="Shared VRAM" value={formatMemory(activeOfferedVramMb)} detail={`${formatMemory(freeVramMb)} currently free`} tone="purple" />
        <Stat icon={Boxes} label="Available models" value={String(snapshot.models.length)} detail="Announced by live nodes" />
        <Stat icon={CheckCircle2} label="Tasks completed" value={String(snapshot.summary.completedJobs)} detail="Current coordinator history" />
      </div>}
      {developerMode ? <>
        <ConnectedPeersTable workers={snapshot.workers} />
        <div className="overview-grid overview-grid-interactive">
        <InteractiveMesh
          workers={active}
          registeredWorkers={snapshot.summary.registered}
          sharedVramMb={activeOfferedVramMb}
          publicLink={publicLink}
          external={external}
          developerMode={developerMode}
          onNavigate={onNavigate}
        />
        <NetworkModelCatalog snapshot={snapshot} developerMode={developerMode} onNavigate={onNavigate} />
          {recentActivity}
        </div>
      </> : <div className="overview-disclosure-stack">
        <PanelDisclosure icon={Network} eyebrow="NETWORK DETAIL" title="Nodes and topology" summary={`${operational.length} operational nodes. Open the full peer list and network map.`}>
          <ConnectedPeersTable workers={snapshot.workers} />
          <InteractiveMesh workers={active} registeredWorkers={snapshot.summary.registered} sharedVramMb={activeOfferedVramMb} publicLink={publicLink} external={external} developerMode={false} onNavigate={onNavigate} />
        </PanelDisclosure>
        <PanelDisclosure icon={Boxes} eyebrow="MODEL DETAIL" title="Available models" summary={`${snapshot.models.length} models announced by live nodes.`}>
          <NetworkModelCatalog snapshot={snapshot} developerMode={false} onNavigate={onNavigate} />
        </PanelDisclosure>
        <PanelDisclosure icon={Activity} eyebrow="RECENT DETAIL" title="Network activity" summary={`${snapshot.jobs.length} tasks in the current coordinator history.`}>{recentActivity}</PanelDisclosure>
      </div>}
    </section>
  );
}

type PeerSortKey = "id" | "hosted" | "version" | "vram" | "share" | "seen" | "role" | "status";
type PeerFilter = "all" | "serving" | "connected" | "offline";

interface ConnectedPeerRow {
  worker: PublicWorker;
  hosted: string;
  hostedCount: number;
  version: string;
  vramMb: number;
  sharePercent: number;
  seenAt: number;
  role: "Host" | "Worker" | "Browser";
  status: "serving" | "connected" | "attention" | "offline";
}

function workerPeerStatus(worker: PublicWorker): ConnectedPeerRow["status"] {
  if (!worker.connected || worker.status === "offline") return "offline";
  if (worker.status === "suspect" || worker.status === "draining") return "attention";
  return worker.deployments.length > 0 ? "serving" : "connected";
}

function workerPeerRole(worker: PublicWorker): ConnectedPeerRow["role"] {
  return worker.kind === "cell" ? "Host" : worker.kind === "browser" ? "Browser" : "Worker";
}

function ConnectedPeersTable({ workers }: { workers: PublicWorker[] }) {
  const pageSize = 10;
  const [filter, setFilter] = useState<PeerFilter>("all");
  const [sortKey, setSortKey] = useState<PeerSortKey>("status");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const [page, setPage] = useState(0);
  const connectedVramMb = workers.filter((worker) => worker.connected).reduce((total, worker) => total + worker.offeredVramMb, 0);
  const rows = useMemo<ConnectedPeerRow[]>(() => workers.map((worker) => {
    const models = [...new Set(worker.deployments.map((deployment) => deployment.model))];
    return {
      worker,
      hosted: models[0] ?? "—",
      hostedCount: models.length,
      version: worker.agentVersion ? `v${worker.agentVersion}` : "Legacy",
      vramMb: worker.offeredVramMb,
      sharePercent: connectedVramMb > 0 && worker.connected ? worker.offeredVramMb / connectedVramMb * 100 : 0,
      seenAt: Date.parse(worker.lastSeenAt) || 0,
      role: workerPeerRole(worker),
      status: workerPeerStatus(worker),
    };
  }), [connectedVramMb, workers]);
  const filteredRows = useMemo(() => {
    const filtered = rows.filter((row) => filter === "all"
      || filter === "serving" && row.status === "serving"
      || filter === "connected" && row.status === "connected"
      || filter === "offline" && (row.status === "offline" || row.status === "attention"));
    return filtered.sort((left, right) => {
      const leftValue = peerSortValue(left, sortKey);
      const rightValue = peerSortValue(right, sortKey);
      const comparison = typeof leftValue === "number" && typeof rightValue === "number"
        ? leftValue - rightValue
        : String(leftValue).localeCompare(String(rightValue));
      return sortDirection === "asc" ? comparison : -comparison;
    });
  }, [filter, rows, sortDirection, sortKey]);
  const pageCount = Math.max(1, Math.ceil(filteredRows.length / pageSize));
  const visiblePage = Math.min(page, pageCount - 1);
  const visibleRows = filteredRows.slice(visiblePage * pageSize, visiblePage * pageSize + pageSize);
  const rangeStart = filteredRows.length === 0 ? 0 : visiblePage * pageSize + 1;
  const rangeEnd = Math.min(filteredRows.length, (visiblePage + 1) * pageSize);

  useEffect(() => {
    if (page >= pageCount) setPage(pageCount - 1);
  }, [page, pageCount]);

  function toggleSort(nextKey: PeerSortKey) {
    if (sortKey === nextKey) {
      setSortDirection((current) => current === "asc" ? "desc" : "asc");
    } else {
      setSortKey(nextKey);
      setSortDirection("asc");
    }
    setPage(0);
  }

  const sortableHeader = (label: string, key: PeerSortKey) => (
    <button type="button" className={sortKey === key ? `active ${sortDirection}` : ""} onClick={() => toggleSort(key)}>
      {label}<ChevronDown size={11} />
    </button>
  );

  return (
    <article className="connected-peers-card" aria-labelledby="connected-peers-title">
      <header>
        <strong id="connected-peers-title">Connected peers</strong>
        <div className="connected-peers-controls">
          <span>{rangeStart}–{rangeEnd}</span>
          <div className="connected-peers-pagination" aria-label="Peer pages">
            <button type="button" aria-label="Previous peer page" disabled={visiblePage === 0} onClick={() => setPage((current) => Math.max(0, current - 1))}><ChevronLeft size={13} /></button>
            <button type="button" aria-label="Next peer page" disabled={visiblePage >= pageCount - 1} onClick={() => setPage((current) => Math.min(pageCount - 1, current + 1))}><ChevronRight size={13} /></button>
          </div>
          <span>· {filteredRows.length} total</span>
          <label><SlidersHorizontal size={12} /><span>Filter</span><select aria-label="Filter peers" value={filter} onChange={(event) => { setFilter(event.target.value as PeerFilter); setPage(0); }}><option value="all">All peers</option><option value="serving">Serving</option><option value="connected">Connected</option><option value="offline">Needs attention</option></select></label>
        </div>
      </header>
      <div className="connected-peers-scroll">
        <table>
          <thead><tr>
            <th aria-sort={sortKey === "id" ? sortDirection === "asc" ? "ascending" : "descending" : "none"}>{sortableHeader("ID", "id")}</th>
            <th aria-sort={sortKey === "hosted" ? sortDirection === "asc" ? "ascending" : "descending" : "none"}>{sortableHeader("Hosted", "hosted")}</th>
            <th aria-sort={sortKey === "version" ? sortDirection === "asc" ? "ascending" : "descending" : "none"}>{sortableHeader("Version", "version")}</th>
            <th aria-sort={sortKey === "vram" ? sortDirection === "asc" ? "ascending" : "descending" : "none"}>{sortableHeader("VRAM", "vram")}</th>
            <th aria-sort={sortKey === "share" ? sortDirection === "asc" ? "ascending" : "descending" : "none"}>{sortableHeader("Share", "share")}</th>
            <th aria-sort={sortKey === "seen" ? sortDirection === "asc" ? "ascending" : "descending" : "none"}>{sortableHeader("Last seen", "seen")}</th>
            <th aria-sort={sortKey === "role" ? sortDirection === "asc" ? "ascending" : "descending" : "none"}>{sortableHeader("Role", "role")}</th>
            <th aria-sort={sortKey === "status" ? sortDirection === "asc" ? "ascending" : "descending" : "none"}>{sortableHeader("Status", "status")}</th>
          </tr></thead>
          <tbody>
            {visibleRows.map((row) => (
              <tr key={row.worker.id}>
                <td><strong>{shortId(row.worker.id)}</strong><small>{workerLabel(row.worker)}</small></td>
                <td><code>{row.hosted}</code>{row.hostedCount > 1 && <small>+{row.hostedCount - 1} more</small>}</td>
                <td><code>{row.version}</code></td>
                <td><strong>{formatMemory(row.vramMb)}</strong></td>
                <td><div className="peer-share"><i><b style={{ width: `${Math.min(100, row.sharePercent)}%` }} /></i><span>{Math.round(row.sharePercent)}%</span></div></td>
                <td><code>{relativeTime(row.worker.lastSeenAt)}</code></td>
                <td><span className={`peer-role ${row.role.toLowerCase()}`}>{row.role}</span></td>
                <td><span className={`peer-state ${row.status}`}><i />{peerStatusLabel(row.status)}</span></td>
              </tr>
            ))}
            {visibleRows.length === 0 && <tr><td className="connected-peers-empty" colSpan={8}><Server size={18} /><span><strong>No peers in this view</strong><small>Change the filter or connect a new device.</small></span></td></tr>}
          </tbody>
        </table>
      </div>
    </article>
  );
}

function peerSortValue(row: ConnectedPeerRow, key: PeerSortKey): string | number {
  switch (key) {
    case "id": return row.worker.id;
    case "hosted": return row.hosted;
    case "version": return row.version;
    case "vram": return row.vramMb;
    case "share": return row.sharePercent;
    case "seen": return row.seenAt;
    case "role": return row.role;
    case "status": return { serving: 0, connected: 1, attention: 2, offline: 3 }[row.status];
  }
}

function peerStatusLabel(status: ConnectedPeerRow["status"]): string {
  if (status === "serving") return "Serving";
  if (status === "connected") return "Connected";
  if (status === "attention") return "Attention";
  return "Offline";
}

interface NetworkCatalogEntry {
  id: string;
  state: "active" | "queued";
  replicas: number;
  pipelines: number;
  request: RequestedModelCapacity | null;
}

function NetworkModelCatalog({ snapshot, developerMode, onNavigate }: {
  snapshot: PublicSnapshot;
  developerMode: boolean;
  onNavigate: (view: PanelView) => void;
}) {
  const [selectedModelId, setSelectedModelId] = useState("");
  const drawerTitleId = useId();
  const entries = useMemo<NetworkCatalogEntry[]>(() => {
    const active = snapshot.models.map((model) => ({
      id: model.id,
      state: "active" as const,
      replicas: model.replicas,
      pipelines: model.pipelines,
      request: snapshot.requestedModels.find((request) => request.id === model.id) ?? null,
    }));
    const activeIds = new Set(active.map((model) => model.id.toLowerCase()));
    const queued = snapshot.requestedModels
      .filter((request) => !activeIds.has(request.id.toLowerCase()))
      .map((request) => ({
        id: request.id,
        state: "queued" as const,
        replicas: 0,
        pipelines: 0,
        request,
      }));
    return [...active, ...queued];
  }, [snapshot.models, snapshot.requestedModels]);
  const selectedEntry = entries.find((entry) => entry.id === selectedModelId) ?? null;

  useEffect(() => {
    if (!selectedEntry) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSelectedModelId("");
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [selectedEntry]);

  return <>
    <article className="network-model-catalog">
      <header className="network-model-catalog-head">
        <div><span>MODEL & TOKEN CATALOG</span><h2>Available intelligence</h2><p>Inspect each model, its real token activity and the peers currently serving it.</p></div>
        <div><b><i /> LIVE DATA</b>{developerMode && <button type="button" onClick={() => onNavigate("models")}>Manage models</button>}</div>
      </header>
      <div className="network-model-table-wrap">
        <table className="network-model-table">
          <thead><tr><th>Model</th><th>Status</th><th>Availability</th><th>Peer memory</th><th>Measured speed</th><th>Processed tokens</th><th><span className="sr-only">Details</span></th></tr></thead>
          <tbody>{entries.map((entry) => {
            const telemetry = networkModelTelemetry(snapshot, entry.id);
            return <tr key={entry.id} className={selectedModelId === entry.id ? "selected" : ""}>
              <td><button type="button" className="network-model-name" onClick={() => setSelectedModelId(entry.id)}><i><Boxes /></i><span><strong>{entry.id}</strong><small>{networkModelRuntimeLabel(telemetry.deployments)}</small></span></button></td>
              <td><span className={`network-model-state ${entry.state}`}>{entry.state === "active" ? "Active" : modelRequestStatusLabel(entry.request?.status)}</span></td>
              <td><strong>{telemetry.peers.length} node{telemetry.peers.length === 1 ? "" : "s"}</strong><small>{entry.replicas} replicas · {entry.pipelines} pipelines</small></td>
              <td>{telemetry.peerMemoryMb > 0 ? formatMemory(telemetry.peerMemoryMb) : "Not reported"}</td>
              <td>{telemetry.measuredThroughput > 0 ? `${formatCompactNumber(telemetry.measuredThroughput)} tok/s` : "Not measured"}</td>
              <td>{telemetry.processedTokens > 0 ? formatCompactTokens(telemetry.processedTokens) : "No activity"}</td>
              <td><button type="button" className="network-model-open" onClick={() => setSelectedModelId(entry.id)} aria-label={`View details for ${entry.id}`}><ChevronRight /></button></td>
            </tr>;
          })}</tbody>
        </table>
        {entries.length === 0 && <div className="network-model-empty"><Boxes /><strong>No models are active yet</strong><span>Models and their token activity will appear here as soon as a node publishes one.</span>{developerMode && <button type="button" onClick={() => onNavigate("models")}>Open model catalog</button>}</div>}
      </div>
    </article>
    {selectedEntry && createPortal(
      <NetworkModelDrawer entry={selectedEntry} snapshot={snapshot} titleId={drawerTitleId} onClose={() => setSelectedModelId("")} />,
      document.body,
    )}
  </>;
}

function NetworkModelDrawer({ entry, snapshot, titleId, onClose }: {
  entry: NetworkCatalogEntry;
  snapshot: PublicSnapshot;
  titleId: string;
  onClose: () => void;
}) {
  const telemetry = networkModelTelemetry(snapshot, entry.id);
  const execution = modelExecutionSummary(snapshot, entry.id);
  const inputTokens = telemetry.jobs.reduce((total, job) => total + job.inputTokens, 0);
  const outputTokens = telemetry.jobs.reduce((total, job) => total + job.outputTokens, 0);
  const completedRequests = telemetry.jobs.filter((job) => job.status === "completed").length;
  return <div className="model-detail-backdrop" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) onClose(); }}>
    <aside className="model-detail-drawer" role="dialog" aria-modal="true" aria-labelledby={titleId}>
      <header className="model-detail-head">
        <div className="model-detail-icon"><Boxes /></div>
        <div><span>NETWORK MODEL</span><h2 id={titleId}>{entry.id}</h2><small>{networkModelRuntimeLabel(telemetry.deployments)}</small></div>
        <button type="button" onClick={onClose} aria-label="Close model details"><X /></button>
      </header>
      <div className="model-detail-badges"><span className={entry.state}><i />{entry.state === "active" ? "Active now" : modelRequestStatusLabel(entry.request?.status)}</span><span className={`execution ${execution.deviceType}`}><Cpu />{executionShortLabel(execution)}</span></div>
      <section className="model-detail-metrics" aria-label="Model telemetry">
        <ModelDetailMetric icon={Network} label="Availability" value={`${telemetry.peers.length} node${telemetry.peers.length === 1 ? "" : "s"}`} />
        <ModelDetailMetric icon={MemoryStick} label="Peer memory" value={telemetry.peerMemoryMb > 0 ? formatMemory(telemetry.peerMemoryMb) : "Not reported"} />
        <ModelDetailMetric icon={Coins} label="Processed tokens" value={telemetry.processedTokens > 0 ? formatCompactTokens(telemetry.processedTokens) : "No activity"} />
        <ModelDetailMetric icon={Gauge} label="Measured speed" value={telemetry.measuredThroughput > 0 ? `${formatCompactNumber(telemetry.measuredThroughput)} tok/s` : "Not measured"} />
      </section>
      <section className="model-detail-section">
        <div className="model-detail-section-title"><Activity /><h3>Token activity</h3></div>
        <div className="model-token-breakdown">
          <div><span>Input tokens</span><strong>{formatCompactTokens(inputTokens)}</strong></div>
          <div><span>Output tokens</span><strong>{formatCompactTokens(outputTokens)}</strong></div>
          <div><span>Completed requests</span><strong>{completedRequests}</strong></div>
        </div>
      </section>
      <section className="model-detail-section">
        <div className="model-detail-section-title"><Code2 /><h3>Runtime identity</h3></div>
        <div className="model-runtime-identity"><span><small>MODEL ID</small><code>{entry.id}</code></span><span><small>RUNTIME MODES</small><code>{networkModelRuntimeLabel(telemetry.deployments)}</code></span></div>
      </section>
      <section className="model-detail-section model-peer-section">
        <div className="model-detail-section-title"><Network /><h3>Active peers</h3><span>{telemetry.peers.length}</span></div>
        <div className="model-peer-table-wrap">
          <table className="model-peer-table">
            <thead><tr><th>Node</th><th>Region</th><th>Device</th><th>Memory</th><th>Speed</th></tr></thead>
            <tbody>{telemetry.peers.map(({ worker, deployments }) => {
              const measured = deployments.filter(isMeasuredDeployment).reduce((total, deployment) => total + deployment.tokensPerSecond, 0);
              return <tr key={worker.id}><td><strong><i />{shortId(worker.id)}</strong></td><td>{worker.region}</td><td>{worker.gpus[0]?.model ?? worker.kind}</td><td>{formatMemory(worker.offeredVramMb)}</td><td>{measured > 0 ? `${formatCompactNumber(measured)} tok/s` : "Not measured"}</td></tr>;
            })}</tbody>
          </table>
          {telemetry.peers.length === 0 && <div className="model-peer-empty"><Server /><span>No active peer is advertising this model right now.</span></div>}
        </div>
      </section>
    </aside>
  </div>;
}

function ModelDetailMetric({ icon: Icon, label, value }: { icon: typeof Network; label: string; value: string }) {
  return <div><Icon /><span><small>{label}</small><strong>{value}</strong></span></div>;
}

function networkModelTelemetry(snapshot: PublicSnapshot, modelId: string) {
  const normalizedModelId = modelId.toLowerCase();
  const peers = snapshot.workers.flatMap((worker) => {
    const deployments = worker.deployments.filter((deployment) => deployment.model.toLowerCase() === normalizedModelId);
    return deployments.length > 0 && worker.connected ? [{ worker, deployments }] : [];
  });
  const deployments = peers.flatMap((peer) => peer.deployments);
  const jobs = snapshot.jobs.filter((job) => job.model.toLowerCase() === normalizedModelId);
  return {
    peers,
    deployments,
    jobs,
    peerMemoryMb: peers.reduce((total, peer) => total + peer.worker.offeredVramMb, 0),
    measuredThroughput: deployments.filter(isMeasuredDeployment).reduce((total, deployment) => total + deployment.tokensPerSecond, 0),
    processedTokens: jobs.reduce((total, job) => total + job.inputTokens + job.outputTokens, 0),
  };
}

function networkModelRuntimeLabel(deployments: PublicDeployment[]): string {
  const labels = [...new Set(deployments.map((deployment) => deployment.adapter ?? deployment.mode).filter(Boolean))];
  return labels.length > 0 ? labels.join(" · ") : "Runtime not reported";
}

function modelRequestStatusLabel(status: RequestedModelCapacity["status"] | undefined): string {
  if (!status) return "Queued";
  const labels: Record<RequestedModelCapacity["status"], string> = {
    profiling: "Profiling",
    waiting_capacity: "Waiting for capacity",
    ready: "Ready",
    activating: "Activating",
    active: "Active",
    incompatible: "Incompatible",
    failed: "Needs attention",
  };
  return labels[status];
}

const MESH_OVERVIEW_POSITIONS = [
  { x: 10, y: 24 }, { x: 19, y: 70 }, { x: 31, y: 42 }, { x: 38, y: 82 },
  { x: 62, y: 20 }, { x: 70, y: 58 }, { x: 82, y: 30 }, { x: 89, y: 73 },
  { x: 59, y: 84 }, { x: 28, y: 14 }, { x: 75, y: 86 }, { x: 92, y: 48 },
  { x: 8, y: 51 }, { x: 47, y: 16 },
] as const;

const MESH_AMBIENT_SPORES = [
  { x: 15, y: 35 }, { x: 24, y: 84 }, { x: 36, y: 18 }, { x: 43, y: 67 },
  { x: 57, y: 31 }, { x: 66, y: 76 }, { x: 78, y: 16 }, { x: 88, y: 58 },
  { x: 94, y: 28 }, { x: 7, y: 77 },
] as const;

function InteractiveMesh({ workers, registeredWorkers, sharedVramMb, publicLink, external, developerMode, onNavigate }: {
  workers: PublicWorker[];
  registeredWorkers: number;
  sharedVramMb: number;
  publicLink: (path: string) => string;
  external: boolean;
  developerMode: boolean;
  onNavigate: (view: PanelView) => void;
}) {
  const [zoom, setZoom] = useState(1);
  const [selectedWorkerId, setSelectedWorkerId] = useState("");
  const [fullscreen, setFullscreen] = useState(false);
  const cardRef = useRef<HTMLElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const visibleWorkers = workers.slice(0, MESH_OVERVIEW_POSITIONS.length);
  const selectedWorker = visibleWorkers.find((worker) => worker.id === selectedWorkerId) ?? null;
  const selectedWorkerIndex = selectedWorker ? visibleWorkers.findIndex((worker) => worker.id === selectedWorker.id) : -1;
  const selectedWorkerPosition = selectedWorkerIndex >= 0 ? MESH_OVERVIEW_POSITIONS[selectedWorkerIndex] ?? { x: 50, y: 50 } : null;
  const visibleOfferedVramMb = visibleWorkers.reduce((total, worker) => total + worker.offeredVramMb, 0);
  const selectedSharePercent = selectedWorker && visibleOfferedVramMb > 0 ? selectedWorker.offeredVramMb / visibleOfferedVramMb * 100 : 0;
  const links = visibleWorkers.map((worker, index) => {
    const target = MESH_OVERVIEW_POSITIONS[index] ?? { x: 50, y: 50 };
    if (index < 2) return { worker, target, source: { x: 50, y: 50 } };
    const parentIndex = Math.floor((index - 2) / 2);
    return { worker, target, source: MESH_OVERVIEW_POSITIONS[parentIndex] ?? { x: 50, y: 50 } };
  });

  useEffect(() => {
    const updateFullscreen = () => setFullscreen(document.fullscreenElement === cardRef.current);
    document.addEventListener("fullscreenchange", updateFullscreen);
    return () => document.removeEventListener("fullscreenchange", updateFullscreen);
  }, []);

  function updatePointerGlow(event: React.PointerEvent<HTMLDivElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    event.currentTarget.style.setProperty("--mesh-pointer-x", `${((event.clientX - rect.left) / rect.width) * 100}%`);
    event.currentTarget.style.setProperty("--mesh-pointer-y", `${((event.clientY - rect.top) / rect.height) * 100}%`);
  }

  async function toggleFullscreen() {
    if (!cardRef.current) return;
    if (document.fullscreenElement === cardRef.current) await document.exitFullscreen();
    else await cardRef.current.requestFullscreen();
  }

  function resetView() {
    setZoom(1);
    setSelectedWorkerId("");
    stageRef.current?.style.setProperty("--mesh-pointer-x", "50%");
    stageRef.current?.style.setProperty("--mesh-pointer-y", "45%");
  }

  return (
    <article className={`network-canvas interactive-mesh${fullscreen ? " fullscreen" : ""}`} ref={cardRef}>
      <div className="canvas-head mesh-canvas-head">
        <div><span>LIVE TOPOLOGY</span><strong>Explore the network</strong></div>
        <div className="mesh-head-status" aria-label={`${visibleWorkers.length} live nodes and ${links.length} active links`}>
          <b><i /> LIVE</b>
          <span>{visibleWorkers.length} NODES</span>
          <span>{links.length} LINKS</span>
        </div>
        <div className="mesh-head-actions">
          {developerMode && <button className="canvas-link" onClick={() => onNavigate("nodes")}>Node details</button>}
          <button className="mesh-icon-button" onClick={resetView} title="Reset network view" aria-label="Reset network view"><RefreshCw /></button>
          <button className="mesh-fullscreen-button" onClick={() => void toggleFullscreen()}><Maximize2 />{fullscreen ? "Exit fullscreen" : "Fullscreen"}</button>
        </div>
      </div>
      <div
        className="interactive-mesh-stage"
        ref={stageRef}
        onPointerMove={updatePointerGlow}
        onPointerLeave={(event) => {
          event.currentTarget.style.setProperty("--mesh-pointer-x", "50%");
          event.currentTarget.style.setProperty("--mesh-pointer-y", "45%");
        }}
      >
        <div className="mesh-ambient mesh-ambient-one" />
        <div className="mesh-ambient mesh-ambient-two" />
        <div className="mesh-scene" style={{ "--mesh-zoom": zoom } as CSSProperties}>
          <div className="mesh-grid" />
          <div className="mesh-spores" aria-hidden="true">
            {MESH_AMBIENT_SPORES.map((spore, index) => <i key={`${spore.x}-${spore.y}`} style={{ "--spore-x": `${spore.x}%`, "--spore-y": `${spore.y}%`, "--spore-delay": `${index * -0.83}s` } as CSSProperties} />)}
          </div>
          <svg className="mesh-links" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
            <defs>
              <linearGradient id="mesh-link-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop className="mesh-link-start" offset="0%" stopColor="#a8794d" stopOpacity=".24" />
                <stop className="mesh-link-mid" offset="48%" stopColor="#d2a16c" stopOpacity=".78" />
                <stop className="mesh-link-end" offset="100%" stopColor="#8a5f36" stopOpacity=".28" />
              </linearGradient>
              <filter id="mesh-packet-glow" x="-200%" y="-200%" width="500%" height="500%">
                <feGaussianBlur stdDeviation=".8" result="blur" />
                <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
            </defs>
            {links.map(({ worker, source, target }, index) => {
              const path = `M ${source.x} ${source.y} L ${target.x} ${target.y}`;
              return <g key={worker.id} className={selectedWorkerId === worker.id ? "selected" : ""}>
                <path d={path} pathLength="1" />
                <circle className="mesh-packet" r=".62" filter="url(#mesh-packet-glow)">
                  <animateMotion dur={`${3.6 + (index % 4) * 0.7}s`} begin={`${index * -0.73}s`} path={path} repeatCount="indefinite" />
                </circle>
              </g>;
            })}
          </svg>
          <button className="mesh-core" onClick={resetView} aria-label="Focus mycellios coordinator">
            <span><img src={brandIcon} alt="" /></span>
            <strong>mycellios</strong>
            <small>coordinator</small>
            <i />
          </button>
          {visibleWorkers.map((worker, index) => {
            const position = MESH_OVERVIEW_POSITIONS[index] ?? { x: 50, y: 50 };
            const selected = selectedWorkerId === worker.id;
            return (
              <button
                className={`mesh-node ${worker.kind}${selected ? " selected" : ""}`}
                style={{ "--node-x": `${position.x}%`, "--node-y": `${position.y}%`, "--node-delay": `${index * -0.41}s` } as CSSProperties}
                key={worker.id}
                onClick={() => setSelectedWorkerId(selected ? "" : worker.id)}
                aria-pressed={selected}
                aria-label={`${workerLabel(worker)}, ${worker.gpus[0]?.model ?? worker.kind}`}
              >
                <span><WorkerKindIcon worker={worker} /><i /></span>
                <strong>{workerLabel(worker)}</strong>
                <small>{worker.gpus[0]?.model ?? worker.kind}</small>
              </button>
            );
          })}
        </div>
        {visibleWorkers.length === 0 && <div className="network-empty interactive-empty"><Wifi size={28} /><strong>The mesh is ready for its first node</strong><span>Every connected device becomes part of the live topology.</span><a href={publicLink("/mobile/")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}>Connect this device <ArrowRight size={13} /></a></div>}
        <div className="mesh-stage-legend"><b><i /> LIVE</b><span>{visibleWorkers.length} nodes</span><span>Encrypted peer links</span></div>
        {selectedWorker && selectedWorkerPosition && <MeshNodePopover worker={selectedWorker} position={selectedWorkerPosition} sharePercent={selectedSharePercent} onClose={() => setSelectedWorkerId("")} />}
        {visibleWorkers.length > 0 && <div className="mesh-zoom-controls" aria-label="Network zoom controls">
          <button onClick={() => setZoom((value) => Math.min(1.35, value + 0.12))} disabled={zoom >= 1.35} aria-label="Zoom in"><Plus /></button>
          <output>{Math.round(zoom * 100)}%</output>
          <button onClick={() => setZoom((value) => Math.max(0.76, value - 0.12))} disabled={zoom <= 0.76} aria-label="Zoom out"><Minus /></button>
        </div>}
      </div>
      <div className="network-summary mesh-network-summary">
        <div><span>Nodes online</span><strong>{workers.length} / {registeredWorkers}</strong></div>
        <div><span>Active links</span><strong>{links.length}</strong></div>
        <div><span>Shared VRAM</span><strong>{formatMemory(sharedVramMb)}</strong></div>
      </div>
    </article>
  );
}

export function MeshNodePopover({ worker, position, sharePercent, onClose }: {
  worker: PublicWorker;
  position: { x: number; y: number };
  sharePercent: number;
  onClose: () => void;
}) {
  const models = [...new Set(worker.deployments.map((deployment) => deployment.model))];
  const status = workerPeerStatus(worker);
  const role = workerPeerRole(worker);
  return <aside
    className={`mesh-node-popover ${position.y < 44 ? "below" : "above"}`}
    role="dialog"
    aria-label={`Node details for ${workerLabel(worker)}`}
    style={{
      "--mesh-popover-x": `${Math.max(18, Math.min(82, position.x))}%`,
      "--mesh-popover-y": `${position.y}%`,
    } as CSSProperties}
  >
    <button className="mesh-node-popover-close" onClick={onClose} aria-label="Close node details"><X /></button>
    <header>
      <div><strong>{workerLabel(worker)}</strong><small>{shortId(worker.id)}</small></div>
      <span className={`peer-state ${status}`}><i />{peerStatusLabel(status)}</span>
    </header>
    <div className="mesh-node-popover-meta">
      <span className={`peer-role ${role.toLowerCase()}`}>{role}</span>
      <span><Timer size={12} />Seen {relativeTime(worker.lastSeenAt)}</span>
    </div>
    <div className="mesh-node-popover-grid">
      <small><span><Boxes size={11} />Model</span><b title={models[0] ?? "No model hosted"}>{models[0] ?? "No model hosted"}</b></small>
      <small><span><Code2 size={11} />Version</span><b>{worker.agentVersion ? `v${worker.agentVersion}` : "Legacy"}</b></small>
      <small><span><MemoryStick size={11} />Shared VRAM</span><b>{formatMemory(worker.offeredVramMb)}</b></small>
      <small><span><Activity size={11} />Reliability</span><b>{Math.round(Math.max(0, Math.min(1, worker.reliability)) * 100)}%</b></small>
    </div>
    <footer>
      <span><Network size={12} />{Math.round(sharePercent)}% mesh share</span>
      <i />
      <span>{models.length} model{models.length === 1 ? "" : "s"}</span>
      <i />
      <span>{worker.region}</span>
    </footer>
  </aside>;
}

function Nodes({
  snapshot,
  onRemove,
  onClearOffline,
  onListCredentials,
  onRevokeCredential,
}: {
  snapshot: PublicSnapshot;
  onRemove: (workerId: string) => Promise<void>;
  onClearOffline: () => Promise<void>;
  onListCredentials: () => Promise<WorkerCredentialSummary[]>;
  onRevokeCredential: (
    credential: Pick<WorkerCredentialSummary, "identityKind" | "identityId" | "fingerprint">,
    reason: string,
  ) => Promise<void>;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [selectedWorkerId, setSelectedWorkerId] = useState("");
  const [credentials, setCredentials] = useState<WorkerCredentialSummary[] | null>(null);
  const [credentialError, setCredentialError] = useState<string | null>(null);
  const connectedWorkers = snapshot.workers.filter((worker) => worker.connected);
  const activeWorkers = snapshot.workers.filter((worker) => worker.connected && worker.status === "online");
  const physicalWorkers = snapshot.workers.filter((worker) => worker.kind !== "cell");
  const activePhysicalWorkers = activeWorkers.filter((worker) => worker.kind !== "cell");
  const connectedPhysicalWorkers = connectedWorkers.filter((worker) => worker.kind !== "cell");
  const installedWorkers = snapshot.workers.filter((worker) => worker.kind === "desktop");
  const browserWorkers = snapshot.workers.filter((worker) => worker.kind === "browser");
  const cellWorkers = snapshot.workers.filter((worker) => worker.kind === "cell");
  const activeGpus = activeWorkers.flatMap((worker) => worker.gpus);
  const activeDeployments = activeWorkers.flatMap((worker) => worker.deployments);
  const telemetryGpus = activeGpus.filter((gpu) => gpu.utilizationPct !== undefined);
  const poweredGpus = activeGpus.filter((gpu) => (gpu.powerW ?? 0) > 0);
  const activeOfferedVramMb = activeGpus.reduce((total, gpu) => total + gpu.offeredVramMb, 0);
  const freeVramMb = activeGpus.reduce((total, gpu) => total + gpu.freeOfferedVramMb, 0);
  const averageUtilization = telemetryGpus.length > 0 ? telemetryGpus.reduce((total, gpu) => total + (gpu.utilizationPct ?? 0), 0) / telemetryGpus.length : null;
  const observedPowerW = poweredGpus.reduce((total, gpu) => total + (gpu.powerW ?? 0), 0);
  const measuredDeployments = activeDeployments.filter(isMeasuredDeployment);
  const measuredThroughput = measuredDeployments.reduce((total, deployment) => total + deployment.tokensPerSecond, 0);
  const sortedWorkers = [...snapshot.workers].sort((left, right) => nodeStateOrder(left) - nodeStateOrder(right));
  const selectedWorker = snapshot.workers.find((worker) => worker.id === selectedWorkerId) ?? null;
  async function remove(workerId: string) {
    setBusy(workerId);
    setActionError(null);
    try {
      await onRemove(workerId);
    } catch (error) {
      setActionError(friendlyModelMutationError(error));
    } finally {
      setBusy(null);
    }
  }
  async function clearOffline() {
    setBusy("offline");
    setActionError(null);
    try {
      await onClearOffline();
    } catch (error) {
      setActionError(friendlyModelMutationError(error));
    } finally {
      setBusy(null);
    }
  }
  async function loadCredentials() {
    setBusy("credentials");
    setCredentialError(null);
    try {
      setCredentials(await onListCredentials());
    } catch (error) {
      setCredentialError(friendlyModelMutationError(error));
    } finally {
      setBusy(null);
    }
  }
  async function revokeCredential(credential: WorkerCredentialSummary) {
    if (
      !window.confirm(
        `Revoke access for ${credential.identityId}? This device will disconnect and cannot sign in again with this key.`,
      )
    ) return;
    setBusy(`credential:${credential.fingerprint}`);
    setCredentialError(null);
    try {
      await onRevokeCredential(
        credential,
        "Manually revoked from the network panel",
      );
      setCredentials(await onListCredentials());
    } catch (error) {
      setCredentialError(friendlyModelMutationError(error));
    } finally {
      setBusy(null);
    }
  }
  return (
    <section className="nodes-page">
      <PageTitle eyebrow="DISTRIBUTED NETWORK" title="Node network" copy="Physical devices, active browsers and compute cells connected to mycellios." actions={<button disabled={busy !== null} onClick={() => void clearOffline()}><Trash2 size={15} /> Clear offline</button>} />
      {actionError && <div className="inline-error" role="alert"><CircleAlert size={17} />{actionError}</div>}

      <div className="node-overview-grid compact">
        <NodeOverviewStat icon={Server} label="Active devices" value={`${activePhysicalWorkers.length} / ${physicalWorkers.length}`} detail={`${connectedPhysicalWorkers.length} connected now · ${cellWorkers.length} cells`} progress={physicalWorkers.length > 0 ? activePhysicalWorkers.length / physicalWorkers.length : 0} tone="green" />
        <NodeOverviewStat icon={MemoryStick} label="Offered memory" value={formatMemory(activeOfferedVramMb)} detail={`${formatMemory(freeVramMb)} available for models`} progress={activeOfferedVramMb > 0 ? freeVramMb / activeOfferedVramMb : 0} tone="blue" />
        <NodeOverviewStat icon={Zap} label="Measured inference capacity" value={measuredThroughput > 0 ? `${formatCompactNumber(measuredThroughput)} tok/s` : "Not measured"} detail={measuredDeployments.length > 0 ? `${measuredDeployments.length} runtime${measuredDeployments.length === 1 ? "" : "s"} with completed inference` : activeDeployments.length > 0 ? "Measured after the first real inference" : "Requires an active model"} tone="violet" />
      </div>

      <PanelDisclosure icon={Gauge} eyebrow="ON DEMAND" title="Physical telemetry" summary="GPU load, reporting coverage, and observed power.">
        <div className="node-overview-grid compact">
          <NodeOverviewStat icon={Gauge} label="Average GPU load" value={averageUtilization === null ? "No data" : `${averageUtilization.toFixed(0)}%`} detail={`${telemetryGpus.length} GPU${telemetryGpus.length === 1 ? "" : "s"} reporting telemetry`} progress={averageUtilization === null ? 0 : averageUtilization / 100} tone="purple" />
          <NodeOverviewStat icon={Activity} label="GPUs with telemetry" value={`${telemetryGpus.length} / ${activeGpus.length}`} detail={telemetryGpus.length > 0 ? `${telemetryGpus.length} GPU${telemetryGpus.length === 1 ? "" : "s"} sending physical load` : "No GPU is reporting this reading"} progress={activeGpus.length > 0 ? telemetryGpus.length / activeGpus.length : 0} tone="cyan" />
          <NodeOverviewStat icon={CirclePower} label="Observed GPU power" value={poweredGpus.length > 0 ? formatPower(observedPowerW) : "No data"} detail={poweredGpus.length > 0 ? "Sum of watts reported by GPUs" : telemetryGpus.length > 0 ? "Real load exists, but the driver exposes no wattage" : "Never estimated or invented"} tone="orange" />
        </div>
      </PanelDisclosure>

      <PanelDisclosure icon={Network} eyebrow="ON DEMAND" title="Live topology" summary={`${activeWorkers.length} active connections. Open the network map when you need routing context.`}>
      <section className="node-topology-card">
        <header className="node-topology-head">
          <div><span>LIVE TOPOLOGY</span><h2>Coordinator connections</h2><p>Each line represents a real session with mycellios.</p></div>
          <div className="node-topology-legend"><span><i className="active" />Active</span><span><i className="warning" />Needs attention</span><span><i className="offline" />Offline</span></div>
        </header>
        <div className={`node-topology-map${snapshot.workers.length === 0 ? " empty" : ""}`}>
          <div className="node-topology-grid" />
          <svg className="node-topology-links" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
            {sortedWorkers.map((worker, index) => {
              const position = topologyPosition(index, sortedWorkers.length);
              return <line key={worker.id} className={`${nodeVisualState(worker)}${selectedWorker?.id === worker.id ? " selected" : ""}`} x1="50" y1="52" x2={position.x} y2={position.y} vectorEffect="non-scaling-stroke" />;
            })}
          </svg>
          <div className="node-network-hub"><div><Network /></div><strong>mycellios</strong><span>{activeWorkers.length} active</span><i /></div>
          {sortedWorkers.map((worker, index) => {
            const position = topologyPosition(index, sortedWorkers.length);
            const label = workerLabel(worker);
            return <button
              type="button"
              key={worker.id}
              aria-label={`View node ${label}`}
              aria-pressed={selectedWorker?.id === worker.id}
              className={`topology-node ${nodeVisualState(worker)}${selectedWorker?.id === worker.id ? " selected" : ""}`}
              style={{ "--node-x": `${position.x}%`, "--node-y": `${position.y}%` } as CSSProperties}
              onClick={() => setSelectedWorkerId(worker.id)}
            ><div><WorkerKindIcon worker={worker} /><i /></div><span><strong>{label}</strong><small>{worker.region} · {nodeStatusLabel(worker)}</small></span></button>;
          })}
          {snapshot.workers.length === 0 && <div className="node-topology-empty"><Server /><strong>No nodes yet</strong><span>Connect a machine to see it appear on the network.</span></div>}
        </div>
      </section>
      </PanelDisclosure>

      <PanelDisclosure icon={ShieldCheck} eyebrow="ADMINISTRATION" title="Device trust" summary="Signed identities and access revocation.">
      <section className="device-trust-card">
        <header>
          <div>
            <span>DEVICE TRUST</span>
            <h2>Signed identities</h2>
            <p>Administrative inventory of enrolled keys. Revoking cuts the active session and blocks new registrations.</p>
          </div>
          <button
            type="button"
            disabled={busy !== null}
            onClick={() => void loadCredentials()}
          >
            {busy === "credentials" ? <LoaderCircle className="spin" /> : <ShieldCheck />}
            {credentials === null ? "Load identities" : "Refresh"}
          </button>
        </header>
        {credentialError && <div className="inline-error" role="alert"><CircleAlert size={17} />{credentialError}</div>}
        {credentials !== null && credentials.length === 0 && (
          <div className="device-trust-empty">
            <ShieldCheck />
            <strong>No enrolled credentials yet</strong>
            <span>Devices appear here after completing their first signed challenge.</span>
          </div>
        )}
        {credentials !== null && credentials.length > 0 && (
          <div className="device-trust-list">
            {credentials.map((credential) => (
              <article key={`${credential.identityKind}:${credential.identityId}`}>
                <div className="device-trust-identity">
                  <div><LockKeyhole /></div>
                  <span>
                    <small>{credentialKindLabel(credential.identityKind)}</small>
                    <strong>{credential.identityId}</strong>
                    <em title={credential.fingerprint}>{shortFingerprint(credential.fingerprint)}</em>
                  </span>
                </div>
                <div className="device-trust-meta">
                  <span><small>Protocol</small><strong>v{credential.protocolVersion}</strong></span>
                  <span><small>Last seen</small><strong>{relativeTime(credential.lastSeenAt)}</strong></span>
                  <span><small>Status</small><strong className={credential.status}>{credential.status === "active" ? "Active" : "Revoked"}</strong></span>
                </div>
                <div className="device-trust-action">
                  {credential.status === "active" ? (
                    <button
                      type="button"
                      disabled={busy !== null}
                      onClick={() => void revokeCredential(credential)}
                    >
                      {busy === `credential:${credential.fingerprint}`
                        ? <LoaderCircle className="spin" />
                        : <CircleAlert />}
                      Revoke access
                    </button>
                  ) : (
                    <span title={credential.revocationReason ?? undefined}>
                      {credential.revokedAt ? `Revoked ${relativeTime(credential.revokedAt)}` : "Revoked"}
                    </span>
                  )}
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
      </PanelDisclosure>

      <NodeInventorySection eyebrow="INSTALLED DEVICES" title="Physical devices" workers={installedWorkers} snapshot={snapshot} busy={busy} onRemove={remove} onSelect={setSelectedWorkerId} />
      {browserWorkers.length > 0 && <NodeInventorySection eyebrow="TEMPORARY CAPACITY" title="Active browsers" workers={browserWorkers} snapshot={snapshot} busy={busy} onRemove={remove} onSelect={setSelectedWorkerId} />}
      {cellWorkers.length > 0 && <NodeInventorySection eyebrow="DISTRIBUTED EXECUTORS" title="Compute cells" workers={cellWorkers} snapshot={snapshot} busy={busy} onRemove={remove} onSelect={setSelectedWorkerId} />}
      {snapshot.workers.length === 0 && <div className="node-card-grid"><div className="wide-empty"><Empty icon={Server} title="No registered nodes" copy="Open the mobile worker or install mycellios to add the first machine." /><a href="/mobile/">Connect a device <ArrowRight size={15} /></a></div></div>}
      {selectedWorker && <NodeDetailsDrawer worker={selectedWorker} snapshot={snapshot} onClose={() => setSelectedWorkerId("")} />}
    </section>
  );
}

function NodeInventorySection({ eyebrow, title, workers, snapshot, busy, onRemove, onSelect }: {
  eyebrow: string;
  title: string;
  workers: PublicWorker[];
  snapshot: PublicSnapshot;
  busy: string | null;
  onRemove: (workerId: string) => Promise<void>;
  onSelect: (workerId: string) => void;
}) {
  if (workers.length === 0) return null;
  return <>
    <div className="node-list-heading"><div><span>{eyebrow}</span><h2>{title}</h2></div><strong>{workers.length} registered</strong></div>
    <div className="node-card-grid compact">
      {workers.map((worker) => {
        const execution = workerExecutionSummary(snapshot, worker);
        const remoteRepair = worker.acceleration;
        const autoRepair = executionAutoRepairActive(execution, snapshot.workers)
          || Boolean(remoteRepair?.retryable && remoteRepair.state !== "gpu-ready");
        return <article className="node-card" key={worker.id}>
        <div className="node-card-head"><div className={`node-device ${worker.kind}`}><WorkerKindIcon worker={worker} /></div><div><span>{workerKindLabel(worker)}</span><h2>{workerLabel(worker)}</h2><small>{shortId(worker.id)}</small></div><span className={`node-status ${worker.status}`}><i />{nodeStatusLabel(worker)}</span></div>
        <div className={`node-runtime-state${autoRepair ? " repairing" : ""}`}><ExecutionBadge execution={execution} workers={snapshot.workers} /><span><strong>{remoteRepair?.retryable && remoteRepair.state !== "gpu-ready" ? `GPU auto-repair · ${remoteRepair.phase} · ${remoteRepair.progressPct ?? 0}%` : autoRepair ? "Service maintained on CPU · auto-repair active" : execution.verified ? executionDetail(execution) : "No active model is reporting execution on this node"}</strong><small>{remoteRepair?.retryable && remoteRepair.state !== "gpu-ready" ? remoteRepair.issueSummary ?? "mycellios is rebuilding and revalidating the local GPU runtime." : autoRepair ? "mycellios will revalidate the GPU and migrate this stage automatically." : execution.verified ? executionDeviceNames(execution) : "Detected GPU hardware is not counted as active compute."}</small></span></div>
        <div className="node-card-quick"><span><MemoryStick size={14} />{formatMemory(worker.offeredVramMb)} offered</span><span><Globe2 size={14} />{worker.region}</span></div>
        <div className="node-card-foot"><span>Seen {relativeTime(worker.lastSeenAt)}</span><div><button type="button" onClick={() => onSelect(worker.id)}>View details</button><button disabled={busy === worker.id} onClick={() => void onRemove(worker.id)}>{busy === worker.id ? <LoaderCircle className="spin" /> : <Trash2 />} Remove</button></div></div>
      </article>;})}
    </div>
  </>;
}

function NodeDetailsDrawer({ worker, snapshot, onClose }: { worker: PublicWorker; snapshot: PublicSnapshot; onClose: () => void }) {
  const drawerRef = useRef<HTMLElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeButtonRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = drawerRef.current?.querySelectorAll<HTMLElement>(
        'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
      if (!focusable || focusable.length === 0) return;
      const first = focusable.item(0);
      const last = focusable.item(focusable.length - 1);
      if (!first || !last) return;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      previouslyFocused?.focus();
    };
  }, [onClose]);
  return createPortal(<div className="panel-drawer-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <aside ref={drawerRef} className="panel-detail-drawer" role="dialog" aria-modal="true" aria-label={`Node details for ${workerLabel(worker)}`}>
      <header><span><small>NODE DETAILS</small><strong>{workerLabel(worker)}</strong></span><button ref={closeButtonRef} type="button" onClick={onClose} aria-label="Close node details"><X /></button></header>
      <NodeTopologyInspector worker={worker} snapshot={snapshot} />
    </aside>
  </div>, document.querySelector(".public-panel") ?? document.body);
}

function NodeOverviewStat({ icon: Icon, label, value, detail, progress, tone }: { icon: typeof Server; label: string; value: string; detail: string; progress?: number; tone: "green" | "blue" | "purple" | "violet" | "cyan" | "orange" }) {
  return <article className={`node-overview-stat ${tone}`}><div className="node-overview-icon"><Icon /></div><span>{label}</span><strong>{value}</strong><small>{detail}</small>{progress !== undefined && <div className="node-overview-progress"><i style={{ width: `${Math.max(0, Math.min(1, progress)) * 100}%` }} /></div>}</article>;
}

function NodeTopologyInspector({ worker, snapshot }: { worker: PublicWorker; snapshot: PublicSnapshot }) {
  const gpus = worker.gpus;
  const free = gpus.reduce((total, gpu) => total + gpu.freeOfferedVramMb, 0);
  const physicalVram = gpus.reduce((total, gpu) => total + gpu.physicalVramMb, 0);
  const utilizations = gpus.filter((gpu) => gpu.utilizationPct !== undefined);
  const utilization = utilizations.length > 0 ? utilizations.reduce((total, gpu) => total + (gpu.utilizationPct ?? 0), 0) / utilizations.length : null;
  const temperatures = gpus.map((gpu) => gpu.temperatureC).filter((value): value is number => Number.isFinite(value));
  const maximumTemperature = temperatures.length > 0 ? Math.max(...temperatures) : null;
  const powerReadings = gpus.map((gpu) => gpu.powerW).filter(
    (value): value is number => typeof value === "number" && Number.isFinite(value) && value > 0,
  );
  const observedPower = powerReadings.length > 0 ? powerReadings.reduce((total, value) => total + value, 0) : null;
  const measuredDeployments = worker.deployments.filter(isMeasuredDeployment);
  const throughput = measuredDeployments.reduce((total, deployment) => total + deployment.tokensPerSecond, 0);
  const execution = workerExecutionSummary(snapshot, worker);
  const kind = worker.kind === "cell" ? "Logical cell" : worker.kind === "browser" ? "Browser" : "Installed device";
  const computeMode = worker.computeMode ? computeModeLabel(worker.computeMode) : worker.kind === "cell" ? "Orchestrated" : "Not reported";
  return <div className="node-topology-inspector">
    <div className={`node-inspector-identity ${nodeVisualState(worker)}`}><div><WorkerKindIcon worker={worker} /></div><span><small>{worker.kind === "cell" ? "SELECTED CELL" : "SELECTED NODE"}</small><strong>{workerLabel(worker)}</strong><em>{worker.region} · {shortId(worker.id)}</em><ExecutionBadge execution={execution} workers={snapshot.workers} /></span></div>
    <div className="node-inspector-metrics">
      <Metric label="Status" value={nodeStatusLabel(worker)} />
      <Metric label="Type" value={kind} />
      <Metric label="Region" value={worker.region} />
      <Metric label="Last seen" value={relativeTimeEs(worker.lastSeenAt)} />
      <Metric label="Reliability" value={`${Math.round(Math.max(0, Math.min(1, worker.reliability)) * 100)}%`} />
      <Metric label="Version" value={worker.agentVersion ? `v${worker.agentVersion}` : "Legacy"} />
      <Metric label="Declared build" value={shortBuildIdentity(worker.buildIdentity)} />
      <Metric label={worker.kind === "cell" ? "Aggregated capacity" : "Physical VRAM"} value={physicalVram > 0 ? formatMemory(physicalVram) : "No data"} />
      <Metric label="Offered memory" value={formatMemory(worker.offeredVramMb)} />
      <Metric label="Free memory" value={formatMemory(free)} />
      <Metric label="Physical load" value={utilization === null ? "No data" : `${utilization.toFixed(0)}%`} />
      <Metric label="Max temperature" value={maximumTemperature === null ? "No data" : `${maximumTemperature.toFixed(0)} °C`} />
      <Metric label="GPU power" value={observedPower === null ? "No data" : formatPower(observedPower)} />
      <Metric label="Real compute" value={execution.verified ? executionShortLabel(execution) : "Unverified"} />
      <Metric label="GPU runtime" value={worker.acceleration ? remoteAccelerationLabel(worker.acceleration) : "No diagnostics"} />
      <Metric label="Mode" value={computeMode} />
      <Metric label="Measured inference" value={throughput > 0 ? `${formatCompactNumber(throughput)} tok/s` : "Not measured"} />
      <Metric label="Models" value={String(worker.deployments.length)} />
      <Metric label="Completed" value={String(worker.jobsCompleted)} />
    </div>
    {worker.acceleration && <RemoteAccelerationDiagnostics diagnostics={worker.acceleration} />}
    {worker.isolation && <ExecutorIsolationDiagnostics isolation={worker.isolation} />}
  </div>;
}

function ExecutorIsolationDiagnostics({
  isolation,
}: {
  isolation: NonNullable<PublicWorker["isolation"]>;
}) {
  const jobObject = isolation.killOnClose === "windows-job-object";
  return <details className="executor-isolation-diagnostics">
    <summary>
      <span><ShieldCheck />Executor isolation</span>
      <strong>{jobObject ? "PARTIAL · JOB OBJECT" : "PARTIAL · SEALED POLICY"}</strong>
      <ChevronDown />
    </summary>
    <div className="executor-isolation-body">
      <p className="executor-isolation-summary">
        <ShieldCheck />
        <span><strong>Active, verifiable controls.</strong> The process receives a filtered environment, a private temporary directory and a watchdog that stops the route when it exceeds its limits.{jobObject ? " Windows also places it in a Job Object before execution and closes the entire tree if the application exits." : ""}</span>
      </p>
      <div className="executor-isolation-grid">
        <Metric label="Environment" value="Filtered variables" />
        <Metric label="Private temp space" value={`Max ${formatMemory(isolation.maxWorkspaceBytes / (1024 * 1024))}`} />
        <Metric label="Temporary files" value={`Max ${isolation.maxWorkspaceEntries.toLocaleString("en-US")}`} />
        <Metric label="Watchdog" value={`Every ${isolation.workspaceCheckIntervalMs} ms`} />
        <Metric label="Process tree" value={jobObject ? "Windows Job Object" : "Best-effort close"} />
        <Metric label="On close" value={jobObject ? "Guaranteed termination" : "No OS guarantee"} />
        <Metric label="Policy" value={isolation.launchPolicySchema.replace("gdlp-executor-isolation/", "v")} />
      </div>
      <div className="executor-isolation-pending">
        <CircleAlert />
        <span><strong>Strong isolation still pending.</strong> This node does not yet announce an operating-system sandbox or hard CPU/RAM/GPU quotas{jobObject ? "." : " or a kill-on-close guarantee."}</span>
      </div>
    </div>
  </details>;
}

function RemoteAccelerationDiagnostics({
  diagnostics,
}: {
  diagnostics: NonNullable<PublicWorker["acceleration"]>;
}) {
  const repairing = diagnostics.retryable && diagnostics.state !== "gpu-ready";
  return <details className={`remote-accelerator-diagnostics${repairing ? " repairing" : diagnostics.state === "gpu-ready" ? " ready" : ""}`}>
    <summary>
      <span><Activity />{repairing ? "GPU self-repair" : "GPU diagnostics"}</span>
      <strong>{remoteAccelerationLabel(diagnostics)}</strong>
      <ChevronDown />
    </summary>
    <div className="remote-accelerator-body">
      <div className="remote-accelerator-grid">
        <Metric label="App" value={`v${diagnostics.appVersion}`} />
        <Metric label="Backend" value={(diagnostics.backend ?? "pending").toUpperCase()} />
        <Metric label="Phase" value={diagnostics.phase} />
        <Metric label="Attempt" value={String(diagnostics.retryAttempt)} />
      </div>
      {diagnostics.progressPct !== null && <div className="remote-accelerator-progress"><i style={{ width: `${diagnostics.progressPct}%` }} /></div>}
      {diagnostics.issueSummary && <p><CircleAlert />{diagnostics.issueSummary}</p>}
      {diagnostics.nextRetryAt && <small>Next automatic retry: {new Date(diagnostics.nextRetryAt).toLocaleString()}</small>}
      {diagnostics.recentEvents.length > 0 && <ol>{diagnostics.recentEvents.map((event, index) => <li className={event.level} key={`${event.at}-${index}`}><time>{new Date(event.at).toLocaleTimeString()}</time><span>{event.message}</span></li>)}</ol>}
    </div>
  </details>;
}

function remoteAccelerationLabel(diagnostics: NonNullable<PublicWorker["acceleration"]>): string {
  if (diagnostics.state === "gpu-ready") return `${diagnostics.gpuModel ?? diagnostics.deviceName ?? "GPU"} ready`;
  if (diagnostics.retryable) return `${diagnostics.phase} · retry ${diagnostics.retryAttempt}`;
  return diagnostics.issueCode ?? diagnostics.state;
}

function Models({ snapshot, onSearch, onRequest, onRemove, adminToken: initialAdminToken, requiresAdminToken, secureTokenStorage }: {
  snapshot: PublicSnapshot;
  onSearch: (input: HubCatalogSearchInput) => Promise<HubCatalogPage>;
  onRequest: (input: RequestModelInput, adminToken: string) => Promise<void>;
  onRemove: (modelId: string, adminToken: string) => Promise<void>;
  adminToken: string;
  requiresAdminToken: boolean;
  secureTokenStorage: boolean;
}) {
  const [source, setSource] = useState("");
  const [modelId, setModelId] = useState("");
  const [revision, setRevision] = useState("");
  const [contextTokens, setContextTokens] = useState(4096);
  const [minimumNodes, setMinimumNodes] = useState(2);
  const [autoActivate, setAutoActivate] = useState(true);
  const [adminToken, setAdminToken] = useState(initialAdminToken);
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [catalogQuery, setCatalogQuery] = useState("");
  const [catalogModels, setCatalogModels] = useState<HubCatalogModel[]>([]);
  const [catalogBusy, setCatalogBusy] = useState(true);
  const [catalogLoadingMore, setCatalogLoadingMore] = useState(false);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [catalogOpen, setCatalogOpen] = useState(true);
  const [catalogNextCursor, setCatalogNextCursor] = useState<string | null>(null);
  const [catalogSort, setCatalogSort] = useState<HubCatalogSort | "name" | "memory">("downloads");
  const [catalogStatus, setCatalogStatus] = useState<"all" | "compatible" | "unsupported" | "gated">("compatible");
  const [catalogFit, setCatalogFit] = useState<"all" | "fits" | "too-large">("all");
  const searchSequence = useRef(0);
  const selectedCatalogModel = catalogModels.find((model) => model.id === source) ?? null;
  const catalogRemoteSort: HubCatalogSort = catalogSort === "likes" || catalogSort === "lastModified" ? catalogSort : "downloads";
  const availableCatalogMemoryMiB = snapshot.workers
    .filter((worker) => worker.connected && worker.status === "online")
    .flatMap((worker) => worker.gpus)
    .reduce((total, gpu) => total + gpu.freeOfferedVramMb, 0);
  const visibleCatalogModels = useMemo(() => {
    const filtered = catalogModels.filter((model) => {
      if (catalogStatus === "compatible" && !model.compatible) return false;
      if (catalogStatus === "unsupported" && (model.compatible || model.gated)) return false;
      if (catalogStatus === "gated" && !model.gated) return false;
      const memory = estimatedHubMemoryMiB(model);
      if (catalogFit === "fits" && (memory === null || memory > availableCatalogMemoryMiB)) return false;
      if (catalogFit === "too-large" && (memory === null || memory <= availableCatalogMemoryMiB)) return false;
      return true;
    });
    if (catalogSort === "name") return filtered.slice().sort((left, right) => left.id.localeCompare(right.id));
    if (catalogSort === "memory") return filtered.slice().sort((left, right) => (estimatedHubMemoryMiB(left) ?? Number.MAX_SAFE_INTEGER) - (estimatedHubMemoryMiB(right) ?? Number.MAX_SAFE_INTEGER));
    return filtered;
  }, [availableCatalogMemoryMiB, catalogFit, catalogModels, catalogSort, catalogStatus]);

  useEffect(() => {
    const query = catalogQuery.trim();
    if (query.length === 1) {
      setCatalogModels([]);
      setCatalogNextCursor(null);
      setCatalogBusy(false);
      setCatalogLoadingMore(false);
      setCatalogError(null);
      return;
    }
    const sequence = ++searchSequence.current;
    setCatalogBusy(true);
    setCatalogLoadingMore(false);
    setCatalogError(null);
    const timer = window.setTimeout(() => {
      void onSearch({ query, sort: catalogRemoteSort, limit: 20 }).then(
        (page) => {
          if (searchSequence.current !== sequence) return;
          setCatalogModels(page.data);
          setCatalogNextCursor(page.nextCursor);
          setCatalogBusy(false);
        },
        (error: unknown) => {
          if (searchSequence.current !== sequence) return;
          setCatalogModels([]);
          setCatalogNextCursor(null);
          setCatalogError(errorText(error));
          setCatalogBusy(false);
        },
      );
    }, query ? 320 : 0);
    return () => window.clearTimeout(timer);
  }, [catalogQuery, catalogRemoteSort, onSearch]);

  async function loadMoreCatalog() {
    if (!catalogNextCursor || catalogLoadingMore) return;
    const sequence = searchSequence.current;
    setCatalogLoadingMore(true);
    setCatalogError(null);
    try {
      const page = await onSearch({ query: catalogQuery.trim(), cursor: catalogNextCursor, sort: catalogRemoteSort, limit: 20 });
      if (searchSequence.current !== sequence) return;
      setCatalogModels((current) => {
        const known = new Set(current.map((model) => model.id));
        return [...current, ...page.data.filter((model) => !known.has(model.id))];
      });
      setCatalogNextCursor(page.nextCursor);
    } catch (error) {
      if (searchSequence.current === sequence) setCatalogError(errorText(error));
    } finally {
      if (searchSequence.current === sequence) setCatalogLoadingMore(false);
    }
  }

  function selectCatalogModel(model: HubCatalogModel) {
    if (!model.compatible) return;
    setSource(model.id);
    setModelId(networkNameForHubModel(model.id));
    setRevision("");
    setFormError(null);
    setCatalogOpen(false);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setFormError(null);
    try {
      await onRequest({
        id: modelId.trim(),
        source: source.trim(),
        revision: revision.trim() || null,
        contextTokens,
        minimumNodes,
        autoActivate,
      }, adminToken);
    } catch (error) {
      setFormError(friendlyModelMutationError(error));
    } finally {
      setBusy(false);
    }
  }

  return <section className="models-page">
    <PageTitle eyebrow="MODEL FABRIC" title="Models" copy="Choose a compatible Hugging Face model. mycellios prepares its name, calculates capacity and distributes it automatically." />
    <form className="model-request-form" onSubmit={(event) => void submit(event)}>
      <div className="model-form-heading"><div><span>HUGGING FACE CATALOG</span><h2>Choose a model for the network</h2><p>Only certified architectures can be selected. No terminal commands or pre-split files.</p></div><Boxes size={24} /></div>

      <details className="model-advanced model-advanced-top">
        <summary><SlidersHorizontal size={15} /> Advanced settings or manual model ID <ChevronDown size={15} /></summary>
        <div className="model-form-grid">
          <label>HUGGING FACE MODEL<input value={source} onChange={(event) => { setSource(event.target.value); setModelId(networkNameForHubModel(event.target.value)); }} placeholder="Qwen/Qwen3-0.6B" required /></label>
          <label>NETWORK NAME <small>automatic</small><input value={modelId} onChange={(event) => setModelId(event.target.value)} placeholder="qwen3-0.6b" required pattern="[A-Za-z0-9][A-Za-z0-9._-]*" /></label>
          <label>REVISION <small>optional</small><input value={revision} onChange={(event) => setRevision(event.target.value)} placeholder="main or commit" /></label>
          <label>CONTEXT TOKENS<input type="number" min={128} max={1048576} value={contextTokens} onChange={(event) => setContextTokens(Number(event.target.value))} /></label>
          <label>MINIMUM NODES<AppSelect ariaLabel="Minimum nodes" value={String(minimumNodes)} onChange={(value) => setMinimumNodes(Number(value))} options={[2, 3, 4, 5, 6, 7, 8].map((count) => ({ value: String(count), label: String(count) }))} /></label>
        </div>
      </details>

      {(!source || catalogOpen) && <>
        <div className="hub-catalog-toolbar">
          <div className="hub-model-search"><Search size={17} /><input value={catalogQuery} onChange={(event) => setCatalogQuery(event.target.value)} placeholder="Search every Hugging Face text-generation model…" aria-label="Search Hugging Face models" />{catalogBusy && <LoaderCircle className="spin" />}</div>
          <label><span>AVAILABILITY</span><AppSelect ariaLabel="Filter by model availability" value={catalogStatus} onChange={(value) => setCatalogStatus(value as typeof catalogStatus)} options={[{ value: "compatible", label: "Ready to run" }, { value: "all", label: "All Hugging Face" }, { value: "unsupported", label: "Needs adapter" }, { value: "gated", label: "License required" }]} /></label>
          <label><span>NETWORK FIT</span><AppSelect ariaLabel="Filter by network capacity" value={catalogFit} onChange={(value) => setCatalogFit(value as typeof catalogFit)} options={[{ value: "all", label: "Any size" }, { value: "fits", label: "Fits now" }, { value: "too-large", label: "Needs capacity" }]} /></label>
          <label><span>SORT BY</span><AppSelect ariaLabel="Sort Hugging Face models" value={catalogSort} onChange={(value) => setCatalogSort(value as typeof catalogSort)} options={[{ value: "downloads", label: "Most downloaded" }, { value: "likes", label: "Most liked" }, { value: "lastModified", label: "Recently updated" }, { value: "name", label: "Name A–Z" }, { value: "memory", label: "Memory low–high" }]} /></label>
        </div>
        <div className="hub-catalog-heading"><span>{visibleCatalogModels.length} SHOWN · {catalogModels.length} LOADED FROM HUGGING FACE</span><a href="https://huggingface.co/models?pipeline_tag=text-generation" target="_blank" rel="noreferrer">Explore Hugging Face <ExternalLink size={12} /></a></div>
        {catalogError && <div className="hub-catalog-state error"><CircleAlert size={15} />Catalog unavailable. You can still enter a model manually below.</div>}
        {!catalogError && !catalogBusy && catalogModels.length === 0 && <div className="hub-catalog-state">{catalogQuery.trim().length === 1 ? "Type at least two characters to search." : "No matching models found."}</div>}
        {!catalogError && !catalogBusy && catalogModels.length > 0 && visibleCatalogModels.length === 0 && <div className="hub-catalog-state">No loaded models match these filters. Change a filter or load more results.</div>}
        {visibleCatalogModels.length > 0 && <div className="hub-model-table-wrap"><table className="hub-model-table">
          <thead><tr><th>Model</th><th>Architecture</th><th>Memory</th><th>Downloads</th><th>Likes</th><th>Updated</th><th>Status</th><th><span className="sr-only">Select</span></th></tr></thead>
          <tbody>{visibleCatalogModels.map((model) => {
          const estimatedMemoryMiB = estimatedHubMemoryMiB(model);
          const fits = estimatedMemoryMiB !== null && estimatedMemoryMiB <= availableCatalogMemoryMiB;
          const status = model.compatible ? "Certified" : model.gated ? "Gated" : "Not supported";
          return <tr key={model.id} className={`${source === model.id ? "selected" : ""} ${model.compatible ? "compatible" : "unsupported"}`} title={model.compatibilityReason ?? undefined}>
            <td><span className="hub-table-model"><i><Boxes /></i><span><strong>{model.id}</strong><small>{model.author}</small></span></span></td>
            <td><span className="hub-table-architecture">{model.architecture ?? model.modelType ?? "Unknown"}</span></td>
            <td><span className={`hub-table-memory ${fits ? "fits" : ""}`}><MemoryStick />{estimatedMemoryMiB === null ? "Unknown" : `≈ ${formatMemory(estimatedMemoryMiB)}`}</span></td>
            <td>{formatHubCount(model.downloads)}</td><td>{formatHubCount(model.likes)}</td><td>{formatHubDate(model.lastModified)}</td>
            <td><span className={`hub-status-pill ${model.compatible ? "certified" : model.gated ? "gated" : "unsupported"}`}>{status}</span></td>
            <td><button type="button" className="hub-table-select" disabled={!model.compatible} onClick={() => selectCatalogModel(model)}>{source === model.id ? <><CheckCircle2 /> Selected</> : "Select"}</button></td>
          </tr>;
        })}</tbody></table></div>}
        {catalogModels.length > 0 && <div className="hub-catalog-pagination"><span>Results are loaded directly from Hugging Face in pages of 20.</span>{catalogNextCursor ? <button type="button" onClick={() => void loadMoreCatalog()} disabled={catalogLoadingMore}>{catalogLoadingMore ? <LoaderCircle className="spin" /> : <Plus />}{catalogLoadingMore ? "Loading…" : "Load 20 more"}</button> : <em>End of results</em>}</div>}
      </>}

      {source && <div className="hub-selected-model"><CheckCircle2 /><span><small>SELECTED MODEL</small><strong>{source}</strong><em>It will be published as {modelId}{selectedCatalogModel && estimatedHubMemoryMiB(selectedCatalogModel) !== null ? ` · ≈ ${formatMemory(estimatedHubMemoryMiB(selectedCatalogModel)!)} required` : ""}</em></span><button type="button" onClick={() => setCatalogOpen(true)}>Change model</button></div>}

      {requiresAdminToken && <label className="model-admin-auth"><ShieldCheck /><span><strong>Network administrator authorization</strong><small>Required because preparing a model changes the shared network. {secureTokenStorage ? "Saved with encrypted operating-system storage after successful authorization." : "Kept only for this browser session."}</small></span><input type="password" autoComplete="off" spellCheck={false} value={adminToken} onChange={(event) => { setAdminToken(event.target.value); setFormError(null); }} placeholder="Administrator token" aria-label="Network administrator token" /></label>}

      <div className="model-submit-row">
        <label className="model-auto-toggle"><input type="checkbox" checked={autoActivate} onChange={(event) => setAutoActivate(event.target.checked)} /><span><strong>Activate automatically</strong><small>Start as soon as compatible capacity reaches the requirement.</small></span></label>
        <button className="model-submit" disabled={busy || !source.trim() || !modelId.trim() || (requiresAdminToken && !adminToken.trim())}>{busy ? <LoaderCircle className="spin" /> : <Play size={16} />}{busy ? "Inspecting model…" : "Calculate and prepare"}</button>
      </div>
      {formError && <div className="model-form-error"><CircleAlert size={16} />{formError}</div>}
    </form>

    <div className="model-request-list">
      {snapshot.requestedModels.map((model) => <RequestedModelCard key={model.id} model={model} onRemove={(modelId) => onRemove(modelId, adminToken)} />)}
    </div>

    <div className="panel-table model-active-table"><div className="panel-table-head"><span>Active model</span><span>Replicas</span><span>Pipelines</span><span>Effective device</span></div>
      {snapshot.models.map((model) => {
        const execution = modelExecutionSummary(snapshot, model.id);
        return <div className="panel-table-row model-execution-row" key={model.id}><strong><Boxes size={17} />{model.id}</strong><span>{model.replicas}</span><span>{model.pipelines}</span><ExecutionBadge execution={execution} workers={snapshot.workers} /></div>;
      })}
      {snapshot.models.length === 0 && <Empty icon={Boxes} title="No active models" copy="Choose a model above. It will remain queued with an exact capacity shortfall until the network can run it." />}
    </div>
  </section>;
}

function RequestedModelCard({ model, onRemove }: { model: RequestedModelCapacity; onRemove: (modelId: string) => Promise<void> }) {
  const [removing, setRemoving] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const required = model.requiredVramMiB ?? 0;
  const vramProgress = required > 0 ? model.availableVramMiB / required : 0;
  const nodeProgress = model.requiredNodes > 0 ? model.availableNodes / model.requiredNodes : 0;
  const progress = Math.min(100, Math.round(Math.min(vramProgress, nodeProgress) * 100));
  async function remove() {
    setRemoving(true);
    try { await onRemove(model.id); } finally { setRemoving(false); }
  }
  return <article className={`model-request-card ${model.status}`}>
    <div className="model-request-head"><div className="model-request-icon"><Boxes /></div><div><span>{model.source}</span><h2>{model.id}</h2><small>{model.adapterId ?? "Checking architecture"}</small></div><ModelStatus status={model.status} /></div>
    <p className="model-request-message">{model.message}</p>
    <div className="model-capacity-bar"><i style={{ width: `${progress}%` }} /></div>
    <div className="model-request-quick"><span><MemoryStick size={14} />{formatMemory(model.availableVramMiB)} available</span><span><Server size={14} />{model.availableNodes} / {model.requiredNodes} nodes</span><span><Activity size={14} />{model.activationProgress?.length ?? 0} events</span></div>
    <details className="model-request-details" onToggle={(event) => setDetailsOpen(event.currentTarget.open)}>
      <summary><Activity size={15} />Capacity and diagnostics<ChevronDown size={15} /></summary>
      {detailsOpen && <div className="model-request-details-body">
      <div className="model-capacity-grid">
        <Metric label="Estimated need" value={model.requiredVramMiB === null ? "Calculating" : formatMemory(model.requiredVramMiB)} />
        <Metric label="Available" value={formatMemory(model.availableVramMiB)} />
        <Metric label="Estimated shortfall" value={model.missingVramMiB === null ? "—" : formatMemory(model.missingVramMiB)} />
        <Metric label="Nodes" value={`${model.availableNodes} / ${model.requiredNodes}`} />
      </div>
      {model.activationIncident && <ActivationIncidentPanel incident={model.activationIncident} />}
      {(model.status === "activating" || model.status === "failed" || (model.activationProgress?.length ?? 0) > 0) && <ActivationProgressLog model={model} />}
      </div>}
    </details>
    <div className="model-request-foot"><span>{model.autoActivate ? <><Zap size={14} /> Auto-activation armed</> : "Manual activation"}</span><button disabled={removing} onClick={() => void remove()}>{removing ? <LoaderCircle className="spin" /> : <Trash2 />} Remove</button></div>
  </article>;
}

function ModelStatus({ status }: { status: RequestedModelCapacity["status"] }) {
  const labels: Record<RequestedModelCapacity["status"], string> = {
    profiling: "Profiling",
    waiting_capacity: "Waiting for capacity",
    ready: "Ready",
    activating: "Activation queued",
    active: "Active",
    incompatible: "Incompatible",
    failed: "Needs attention",
  };
  return <b className={`model-request-status ${status}`}>{status === "profiling" || status === "activating" ? <LoaderCircle className="spin" /> : status === "active" || status === "ready" ? <CheckCircle2 /> : <CircleAlert />}{labels[status]}</b>;
}

function ActivationIncidentPanel({
  incident,
}: {
  incident: NonNullable<RequestedModelCapacity["activationIncident"]>;
}) {
  const repairLabels: Record<typeof incident.repairState, string> = {
    scheduled: "Automatic retry scheduled",
    retrying: "Automatic recovery running",
    exhausted: "Safe retry limit reached",
    manual_required: "Verified change required",
  };
  const automatic = incident.repairState === "scheduled"
    || incident.repairState === "retrying";
  return <aside className={`activation-incident ${incident.repairState}`}>
    <header>
      <span>{automatic ? <RefreshCw className="spin-slow" /> : <ShieldCheck />}</span>
      <div><small>{automatic ? "AUTONOMOUS RECOVERY" : "FAIL-CLOSED DIAGNOSIS"}</small><strong>{incident.title}</strong></div>
      <b>{repairLabels[incident.repairState]}</b>
    </header>
    <p>{incident.summary}</p>
    <div className="activation-incident-grid">
      <span><small>Scope</small><strong>{incident.scope}</strong></span>
      <span><small>Equipment</small><strong>{incident.nodeId ? shortId(incident.nodeId) : "Not isolated"}</strong></span>
      <span><small>Stage</small><strong>{incident.stageId ? shortId(incident.stageId) : "—"}</strong></span>
      <span><small>Attempt</small><strong>{incident.attempt > 0 ? `${incident.attempt} / ${incident.maximumAttempts}` : "Not retrying"}</strong></span>
    </div>
    <div className="activation-incident-remedy"><Zap /><span><small>SAFE REMEDY</small><strong>{incident.remedy}</strong></span></div>
    <ol>{incident.steps.map((step, index) => <li key={step}><b>{String(index + 1).padStart(2, "0")}</b>{step}</li>)}</ol>
    {incident.nextRetryAt && <footer><Timer />Next bounded attempt: {new Date(incident.nextRetryAt).toLocaleString()}</footer>}
  </aside>;
}

function Jobs({ snapshot }: { snapshot: PublicSnapshot }) {
  const [visibleCount, setVisibleCount] = useState(20);
  const visibleJobs = snapshot.jobs.slice(0, visibleCount);
  return <section><PageTitle eyebrow="EXECUTION LOG" title="Network jobs" copy="Recent inference requests and their execution state." />
    <div className="panel-table jobs-table"><div className="panel-table-head"><span>Request</span><span>Model</span><span>Tokens</span><span>Status</span></div>
      {visibleJobs.map((job) => <div className="panel-table-row" key={job.id}><strong><Activity size={16} />{shortId(job.id)}<small>{relativeTime(job.createdAt)}</small></strong><span>{job.model}</span><span>{job.inputTokens + job.outputTokens}</span><b className={job.status}><i />{job.status}</b></div>)}
      {snapshot.jobs.length === 0 && <Empty icon={Activity} title="No jobs yet" copy="Use the inference console when a model becomes available." />}
    </div>
    {visibleCount < snapshot.jobs.length && <div className="panel-load-more"><span>Showing {visibleJobs.length} of {snapshot.jobs.length} tasks</span><button type="button" onClick={() => setVisibleCount((count) => count + 20)}><Plus size={15} />Load 20 more</button></div>}
  </section>;
}

type SystemLogFilter = "important" | "errors" | "all";

function SystemLogs({ snapshot, bridge, connectionError }: { snapshot: PublicSnapshot; bridge: DesktopBridge | undefined; connectionError: string | null }) {
  const [desktopLogs, setDesktopLogs] = useState<SystemLogSnapshot | null>(null);
  const [filter, setFilter] = useState<SystemLogFilter>("important");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(Boolean(bridge));
  const [loadError, setLoadError] = useState<string | null>(null);
  const [visibleCount, setVisibleCount] = useState(30);

  const load = useCallback(async (silent = false) => {
    if (!bridge) return;
    if (!silent) setLoading(true);
    try {
      setDesktopLogs(await bridge.getSystemLogs());
      setLoadError(null);
    } catch (caught) {
      if (!silent) setLoadError(errorText(caught));
    } finally {
      if (!silent) setLoading(false);
    }
  }, [bridge]);

  useEffect(() => {
    if (!bridge) return;
    void load();
    const timer = window.setInterval(() => void load(true), 5_000);
    return () => window.clearInterval(timer);
  }, [bridge, load]);

  const networkLogs = useMemo(
    () => networkSnapshotLogs(snapshot, connectionError),
    [connectionError, snapshot],
  );
  const logSnapshot = desktopLogs ?? networkLogs;
  const normalizedQuery = query.trim().toLowerCase();
  const visibleEntries = logSnapshot.entries.filter((entry) => {
    if (filter === "errors" && entry.level !== "error") return false;
    if (filter === "important" && !importantSystemLog(entry)) return false;
    if (!normalizedQuery) return true;
    return `${entry.event} ${entry.message} ${entry.details ?? ""} ${entry.source}`.toLowerCase().includes(normalizedQuery);
  });
  const warningCount = logSnapshot.entries.filter((entry) => entry.level === "warning").length;
  const errorCount = logSnapshot.entries.filter((entry) => entry.level === "error").length;
  const renderedEntries = visibleEntries.slice(0, visibleCount);

  useEffect(() => setVisibleCount(30), [filter, query]);

  return <section className="system-logs-page">
    <PageTitle
      eyebrow="SYSTEM DIAGNOSTICS"
      title="Important logs"
      copy={bridge ? "Real, sanitized events from the application, runtime, workers and updates." : "Important activity observed across the coordinator, nodes, models and tasks."}
      actions={<button disabled={loading} onClick={() => void load()}>{loading ? <LoaderCircle className="spin" size={16} /> : <RefreshCw size={16} />}Refresh</button>}
    />
    <div className="system-log-summary">
      <article><ScrollText /><span><small>LOADED ENTRIES</small><strong>{logSnapshot.entries.length}</strong><em>{bridge ? "Safe local log" : "Current network status"}</em></span></article>
      <article className="warning"><CircleAlert /><span><small>WARNINGS</small><strong>{warningCount}</strong><em>Retries and degradation</em></span></article>
      <article className="error"><X /><span><small>ERRORS</small><strong>{errorCount}</strong><em>Events requiring attention</em></span></article>
    </div>
    <div className="system-log-toolbar">
      <div className="system-log-filters" role="group" aria-label="Filter logs">
        <button className={filter === "important" ? "active" : ""} onClick={() => setFilter("important")}>Important</button>
        <button className={filter === "errors" ? "active" : ""} onClick={() => setFilter("errors")}>Errors only</button>
        <button className={filter === "all" ? "active" : ""} onClick={() => setFilter("all")}>All</button>
      </div>
      <label className="system-log-search"><Search /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search event, source, or message…" /></label>
    </div>
    {loadError && <div className="inline-error"><CircleAlert size={17} />Could not read local logs: {loadError}</div>}
    {logSnapshot.truncated && <div className="system-log-notice"><CircleAlert />Only the most recent entries are shown to keep the panel fast and safe.</div>}
    <div className="system-log-list" aria-live="polite">
      {loading && !desktopLogs ? <div className="system-log-loading"><LoaderCircle className="spin" /><span>Reading important events…</span></div> : renderedEntries.map((entry) => <SystemLogRow entry={entry} key={entry.id} />)}
      {!loading && visibleEntries.length === 0 && <Empty icon={ScrollText} title="No logs for this filter" copy={normalizedQuery ? "Try another search or show all levels." : "No events have been recorded at this level."} />}
    </div>
    {visibleCount < visibleEntries.length && <div className="panel-load-more"><span>Showing {renderedEntries.length} of {visibleEntries.length} matching events</span><button type="button" onClick={() => setVisibleCount((count) => count + 30)}><Plus size={15} />Load 30 more</button></div>}
    <footer className="system-log-safety"><ShieldCheck /><span><strong>Safe, limited read</strong><small>Credentials, tokens, passwords and secrets are redacted before reaching this screen.</small></span></footer>
  </section>;
}

function SystemLogRow({ entry }: { entry: SystemLogEntry }) {
  return <article className={`system-log-row ${entry.level}`}>
    <div className="system-log-row-marker"><i /></div>
    <div className="system-log-row-content">
      <header><time dateTime={entry.at}>{formatSystemLogTime(entry.at)}</time><span className={`system-log-level ${entry.level}`}>{systemLogLevelLabel(entry.level)}</span><span className="system-log-source">{systemLogSourceLabel(entry.source)}</span></header>
      <div><strong>{humanizeSystemLogEvent(entry.event)}</strong><p>{entry.message}</p></div>
      {entry.details && <details><summary>Technical details <ChevronDown /></summary><code>{entry.details}</code></details>}
    </div>
  </article>;
}

function networkSnapshotLogs(snapshot: PublicSnapshot, connectionError: string | null): SystemLogSnapshot {
  const entries: SystemLogEntry[] = [];
  const capturedAt = snapshot.capturedAt === EMPTY.capturedAt ? new Date().toISOString() : snapshot.capturedAt;
  if (connectionError) {
    entries.push({
      id: `network-error-${capturedAt}`,
      at: capturedAt,
      level: "error",
      source: "coordinator",
      event: "coordinator-unavailable",
      message: redactVisibleLogText(connectionError),
    });
  } else {
    entries.push({
      id: `network-snapshot-${capturedAt}`,
      at: capturedAt,
      level: "info",
      source: "coordinator",
      event: "coordinator-snapshot-ready",
      message: `${snapshot.summary.connected} connected nodes, ${snapshot.models.length} available models and ${snapshot.summary.completedJobs} completed tasks.`,
    });
  }
  for (const worker of snapshot.workers) {
    const state = nodeVisualState(worker);
    entries.push({
      id: `worker-${worker.id}-${worker.lastSeenAt}`,
      at: worker.lastSeenAt,
      level: state === "active" ? "info" : state === "warning" ? "warning" : "error",
      source: "worker",
      event: `worker-${state}`,
      message: `${workerLabel(worker)} · ${nodeStatusLabel(worker)} · ${worker.region}`,
      details: `id=${shortId(worker.id)} · memory=${formatMemory(worker.offeredVramMb)} · tasks=${worker.jobsCompleted}`,
    });
  }
  for (const model of snapshot.requestedModels) {
    entries.push({
      id: `model-${model.id}-${model.status}`,
      at: capturedAt,
      level: model.status === "failed" || model.status === "incompatible" ? "error" : model.status === "waiting_capacity" ? "warning" : "info",
      source: "runtime",
      event: `model-${model.status}`,
      message: redactVisibleLogText(model.message),
      details: model.id,
    });
  }
  for (const job of snapshot.jobs) {
    entries.push({
      id: `job-${job.id}-${job.updatedAt}`,
      at: job.updatedAt,
      level: job.status === "failed" ? "error" : job.failureCode ? "warning" : "info",
      source: "network",
      event: `job-${job.status}`,
      message: `${job.model} · ${job.inputTokens + job.outputTokens} tokens`,
      details: job.failureCode ? redactVisibleLogText(job.failureCode) : `request=${shortId(job.id)}`,
    });
  }
  return {
    capturedAt: new Date().toISOString(),
    entries: entries.sort((left, right) => Date.parse(right.at) - Date.parse(left.at)).slice(0, 300),
    truncated: entries.length > 300,
    source: "network-snapshot",
  };
}

function importantSystemLog(entry: SystemLogEntry): boolean {
  return entry.level !== "info" || /(start|ready|status|update|runtime|coordinator|connected|model|contribution)/i.test(entry.event);
}

function redactVisibleLogText(value: string): string {
  return value
    .replace(/(Bearer\s+)[A-Za-z0-9._~+/=-]+/gi, "$1[REDACTED]")
    .replace(/\b(authorization|credential|password|secret|token|api[-_ ]?key)\s*[:=]\s*["']?[^"',}\s]+/gi, "$1=[REDACTED]")
    .replace(/([?&](?:authorization|credential|password|secret|token|api[-_]?key)=)[^&\s]+/gi, "$1[REDACTED]");
}

function formatSystemLogTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown time";
  return new Intl.DateTimeFormat("en-US", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(date);
}

function systemLogLevelLabel(level: SystemLogEntry["level"]): string {
  if (level === "error") return "ERROR";
  if (level === "warning") return "WARNING";
  return "INFO";
}

function systemLogSourceLabel(source: SystemLogEntry["source"]): string {
  const labels: Record<SystemLogEntry["source"], string> = {
    desktop: "Application",
    coordinator: "Coordinator",
    worker: "Worker",
    runtime: "Runtime",
    renderer: "Interface",
    network: "Network",
  };
  return labels[source];
}

function humanizeSystemLogEvent(event: string): string {
  return event.replace(/[-_.]+/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function Tests({ bridge }: { bridge: DesktopBridge | undefined }) {
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [comparisonRunId, setComparisonRunId] = useState("");
  const [filters, setFilters] = useState<BenchmarkFilters>(EMPTY_BENCHMARK_FILTERS);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadRuns = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const next = bridge ? await bridge.getBenchmarkRuns() : await fetchBenchmarkRuns();
      setRuns(next);
      setSelectedRunId((current) => next.some((run) => run.runId === current) ? current : next[0]?.runId ?? "");
      setError(null);
    } catch (caught) {
      if (!silent) setError(errorText(caught));
    } finally {
      if (!silent) setLoading(false);
    }
  }, [bridge]);

  useEffect(() => {
    void loadRuns();
    const timer = window.setInterval(() => { void loadRuns(true); }, 5_000);
    return () => window.clearInterval(timer);
  }, [loadRuns]);

  async function runNow() {
    setRunning(true);
    setError(null);
    try {
      const run = bridge ? await bridge.runBenchmark() : await requestBenchmarkRun();
      setRuns((current) => [run, ...current.filter((item) => item.runId !== run.runId)]);
      setSelectedRunId(run.runId);
      setComparisonRunId("");
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setRunning(false);
    }
  }

  function selectRun(runId: string) {
    setSelectedRunId(runId);
    setComparisonRunId("");
  }

  function updateFilter<Key extends keyof BenchmarkFilters>(
    key: Key,
    value: BenchmarkFilters[Key],
  ) {
    const nextFilters = { ...filters, [key]: value };
    const nextRuns = filterBenchmarkRuns(runs, nextFilters);
    setFilters(nextFilters);
    setSelectedRunId((current) =>
      nextRuns.some((run) => run.runId === current) ? current : nextRuns[0]?.runId ?? ""
    );
    setComparisonRunId("");
  }

  const filterValues = benchmarkFilterValues(runs);
  const filteredRuns = filterBenchmarkRuns(runs, filters);
  const selected = filteredRuns.find((run) => run.runId === selectedRunId)
    ?? filteredRuns[0]
    ?? null;
  const selectedSnapshot = benchmarkRunSnapshot(selected);
  const trendRuns = benchmarkTrendRuns(selected, filteredRuns);
  const comparableMeasurements = trendRuns
    .flatMap((run) => run.measurements)
    .filter((measurement) =>
      !selectedSnapshot
      || measurement.scenarioFingerprint === selectedSnapshot.scenarioFingerprint
    );
  const bestThroughput = comparableMeasurements.reduce<number | null>((best, item) => {
    const value = item.metrics.tokensPerSecondP50 ?? item.metrics.tokensPerSecond;
    return value === null ? best : best === null ? value : Math.max(best, value);
  }, null);
  const selectedMeasurement = selectedSnapshot?.measurement ?? null;
  const selectedInventory = selectedMeasurement?.inventory ?? null;
  const selectedNodes = selectedInventory?.selectedDevices ?? 0;
  const selectedOfferedGb = selectedInventory?.offeredMemoryGb
    ?? nullableBenchmarkSum(selectedInventory?.profiles
      .filter((profile) => profile.kind === "gpu")
      .map((profile) => profile.offeredMemoryGb ?? profile.memoryGb) ?? []);
  const selectedPower = selectedMeasurement?.metrics.powerWattsP50
    ?? selectedInventory?.observedPowerWatts
    ?? nullableBenchmarkSum(selectedInventory?.profiles.map((profile) => profile.observedPowerWatts) ?? []);
  const selectedPowerLimit = selectedInventory?.powerLimitWatts
    ?? nullableBenchmarkSum(selectedInventory?.profiles.map((profile) => profile.powerLimitWatts) ?? []);
  const automaticBaseline = defaultBenchmarkBaseline(selected, filteredRuns);
  const comparisonRun = filteredRuns.find((run) => run.runId === comparisonRunId && run.runId !== selected?.runId)
    ?? automaticBaseline;
  const comparison = compareBenchmarkRuns(selected, comparisonRun);
  const runOptions = filteredRuns.map((run) => {
    const snapshot = benchmarkRunSnapshot(run);
    return {
      value: run.runId,
      label: `${run.trigger === "automatic-model-start" ? "Auto" : "Manual"} · ${snapshot?.measurement.model.label ?? run.triggerModelId ?? "model"} · v${run.version} · ${formatBenchmarkDate(run.finishedAt)}`,
    };
  });
  const comparisonOptions = [
    { value: "", label: "No comparable reference" },
    ...filteredRuns.filter((run) => run.runId !== selected?.runId).map((run) => {
      const snapshot = benchmarkRunSnapshot(run);
      const comparable = snapshot?.scenarioFingerprint === selectedSnapshot?.scenarioFingerprint
        && snapshot?.scenarioComparable === true;
      return {
        value: run.runId,
        label: `${comparable ? "Comparable" : "Different configuration"} · ${snapshot?.measurement.model.label ?? "model"} · ${snapshot ? snapshot.nodes : "No reading"} nodes · v${run.version}`,
      };
    }),
  ];

  return <section className="tests-page">
    <PageTitle eyebrow="AUTOMATED REAL TESTS" title="Does Mycellios really improve?" copy="Compare speed, latency and efficiency without mixing different models, capacity, routes or hardware." actions={<button disabled={running} onClick={() => void runNow()}>{running ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />}{running ? "Measuring…" : "Run test again"}</button>} />
    <div className="benchmark-auto-notice"><Activity size={19} /><span><strong>Automatic measurement with real evidence</strong>The bench warms the model, repeats requests and records its variation. A missing reading appears as “No reading”; it is never estimated to fill the screen.</span></div>
    {error && <div className="benchmark-error"><CircleAlert size={17} /><span><strong>Test could not run</strong>{error}</span></div>}
    <div className="panel-stat-grid benchmark-stats">
      <Stat icon={Activity} label="Pruebas visibles" value={String(filteredRuns.length)} detail={`${runs.length} guardadas en total`} />
      <Stat icon={Boxes} label="Selected model" value={selectedMeasurement?.model.label ?? "No reading"} detail={selected ? `v${selected.version} · ${formatBenchmarkDate(selected.finishedAt)}` : "No measurements"} />
      <Stat icon={Server} label="Nodes used" value={selectedInventory ? String(selectedNodes) : "No reading"} detail={selectedInventory ? `${selectedInventory.connectedDevices}/${selectedInventory.totalDevices} connected` : "No inventory"} />
      <Stat icon={Gauge} label="Best comparable P50" value={formatBenchmark(bestThroughput, " tok/s")} detail={selectedMeasurement ? `${trendRuns.length} test${trendRuns.length === 1 ? "" : "s"} with the same fingerprint` : "No scenario selected"} tone="purple" />
    </div>
    {selectedMeasurement && <div className="benchmark-capacity-strip">
      <BenchmarkMetric icon={MemoryStick} label="Offered memory" value={formatBenchmark(selectedOfferedGb, " GB")} detail={formatBenchmark(selectedInventory?.physicalMemoryGb ?? null, " physical GB")} />
      <BenchmarkMetric icon={Zap} label="P50 power" value={selectedPower === null ? "No reading" : formatPower(selectedPower)} detail={`P95 ${formatPowerReading(selectedMeasurement.metrics.powerWattsP95 ?? null)} · peak ${formatPowerReading(selectedMeasurement.metrics.powerWattsPeak ?? null)}`} />
      <BenchmarkMetric icon={Gauge} label="Velocidad P50" value={formatBenchmark(selectedMeasurement.metrics.tokensPerSecondP50 ?? selectedMeasurement.metrics.tokensPerSecond, " tok/s")} detail={`P5 ${formatBenchmark(selectedMeasurement.metrics.tokensPerSecondP5 ?? null, " tok/s")} · P95 ${formatBenchmark(selectedMeasurement.metrics.tokensPerSecondP95, " tok/s")}`} />
      <BenchmarkMetric icon={Timer} label="Primer token P95" value={formatBenchmark(selectedMeasurement.metrics.ttftMsP95, " ms")} detail={`P50 ${formatBenchmark(selectedMeasurement.metrics.ttftMsP50, " ms")}`} />
      <BenchmarkMetric icon={ShieldCheck} label="Statistical confidence" value={benchmarkStabilityLabel(selectedMeasurement)} detail={`CV ${formatBenchmark(selectedMeasurement.metrics.coefficientOfVariationPct ?? null, "%")} · ±${formatBenchmark(selectedMeasurement.metrics.confidenceHalfWidthPct ?? null, "%")}`} />
      <BenchmarkMetric icon={Activity} label="Measured energy" value={formatBenchmark(selectedMeasurement.metrics.energyWh ?? null, " Wh")} detail={`Coverage ${formatBenchmark(selectedMeasurement.metrics.energyCoveragePct ?? null, "%")}`} />
    </div>}
    {loading && runs.length === 0 ? <div className="benchmark-loading"><LoaderCircle className="spin" /><span>Loading real history…</span></div> : runs.length === 0 ? <Empty icon={Gauge} title="No real tests yet" copy="Start a model. The first measurement is saved automatically when a physical route is available." /> : <>
      <BenchmarkFiltersPanel
        filters={filters}
        values={filterValues}
        visibleCount={filteredRuns.length}
        onChange={updateFilter}
        onClear={() => {
          setFilters(EMPTY_BENCHMARK_FILTERS);
          setSelectedRunId(runs[0]?.runId ?? "");
          setComparisonRunId("");
        }}
      />
      {filteredRuns.length === 0 ? <Empty icon={Search} title="No tests match these filters" copy="Change one filter to see the saved real history again." /> : <>
      <div className="benchmark-toolbar">
        <div className="benchmark-toolbar-selects">
          <label>RESULTADO ACTUAL<AppSelect ariaLabel="Resultado actual" value={selected?.runId ?? ""} onChange={selectRun} options={runOptions} /></label>
          <label>COMPARAR CON<AppSelect ariaLabel="Resultado de referencia" value={comparisonRun?.runId ?? ""} onChange={setComparisonRunId} options={comparisonOptions} /></label>
        </div>
        {selected && <div className={`benchmark-run-state ${selected.status}`}><i />{benchmarkStatusLabel(selected.status)}<span>{selected.trigger === "automatic-model-start" ? "automatic start" : selected.suite === "physical-import" ? "physical campaign" : "manual test"}</span></div>}
      </div>
      <BenchmarkComparisonPanel comparison={comparison} />
      <div className="benchmark-chart-grid">
        <BenchmarkTrend runs={trendRuns} metric="tokensPerSecond" label="Velocidad P50 e intervalo P5–P95" unit="tok/s" tone="blue" />
        <BenchmarkTrend runs={trendRuns} metric="ttftMsP95" label="Primer token P95" unit="ms" tone="purple" />
        <BenchmarkTrend runs={trendRuns} metric="nodes" label="Nodes used" unit=" nodes" tone="green" />
        <BenchmarkTrend runs={trendRuns} metric="tokensPerSecondPerNode" label="Eficiencia por nodo" unit=" tok/s/nodo" tone="orange" />
      </div>
      <BenchmarkHistory runs={filteredRuns} selectedRunId={selected?.runId ?? ""} onSelect={selectRun} />
      {selected && <BenchmarkRunDetails run={selected} />}
      </>}
    </>}
  </section>;
}

function BenchmarkFiltersPanel({
  filters,
  values,
  visibleCount,
  onChange,
  onClear,
}: {
  filters: BenchmarkFilters;
  values: ReturnType<typeof benchmarkFilterValues>;
  visibleCount: number;
  onChange: (key: keyof BenchmarkFilters, value: string) => void;
  onClear: () => void;
}) {
  const activeFilters = Object.values(filters).filter(Boolean).length;
  return <section className="benchmark-filters" aria-label="Benchmark filters">
    <div className="benchmark-filters-heading">
      <div><SlidersHorizontal size={19} /><span><strong>Filter tests</strong><small>{visibleCount} visible result{visibleCount === 1 ? "" : "s"}</small></span></div>
      <button type="button" className="benchmark-clear-filters" disabled={activeFilters === 0} onClick={onClear}>Clear filters{activeFilters > 0 ? ` (${activeFilters})` : ""}</button>
    </div>
    <div className="benchmark-filter-grid">
      <label>MODEL<AppSelect ariaLabel="Filter by model" value={filters.model} onChange={(value) => onChange("model", value)} options={[{ value: "", label: "All models" }, ...values.models]} /></label>
      <label>VERSION<AppSelect ariaLabel="Filter by version" value={filters.version} onChange={(value) => onChange("version", value)} options={[{ value: "", label: "All versions" }, ...values.versions.map((version) => ({ value: version, label: `v${version}` }))]} /></label>
      <label>STATUS<AppSelect ariaLabel="Filter by status" value={filters.status} onChange={(value) => onChange("status", value)} options={[{ value: "", label: "All statuses" }, ...values.statuses.map((status) => ({ value: status, label: benchmarkStatusLabel(status) }))]} /></label>
      <label>BACKEND<AppSelect ariaLabel="Filter by backend" value={filters.backend} onChange={(value) => onChange("backend", value)} options={[{ value: "", label: "All backends" }, ...values.backends.map((backend) => ({ value: backend, label: backend === "__without_reading__" ? "No reading" : backend.toUpperCase() }))]} /></label>
      <label>EVIDENCE<AppSelect ariaLabel="Filter by evidence" value={filters.evidence} onChange={(value) => onChange("evidence", value)} options={[{ value: "", label: "All evidence" }, ...values.evidence.map((evidence) => ({ value: evidence, label: evidence === "physical" ? "Real physical" : "Local loopback" }))]} /></label>
    </div>
  </section>;
}

type BenchmarkTrendMetric = "tokensPerSecond" | "ttftMsP95" | "nodes" | "tokensPerSecondPerNode";

function BenchmarkTrend({ runs, metric, label, unit, tone }: { runs: BenchmarkRun[]; metric: BenchmarkTrendMetric; label: string; unit: string; tone: "blue" | "purple" | "green" | "orange" }) {
  const chronological = [...runs].reverse();
  const points = chronological.map((run) => {
    const snapshot = benchmarkRunSnapshot(run);
    if (!snapshot) return null;
    const value = benchmarkSnapshotMetric(snapshot, metric);
    if (value === null) return null;
    const low = metric === "tokensPerSecond" ? snapshot.tokensPerSecondP5 : null;
    const high = metric === "tokensPerSecond" ? snapshot.tokensPerSecondP95 : null;
    return { run, snapshot, value, low, high };
  }).filter((point): point is NonNullable<typeof point> => point !== null);
  const values = points.flatMap((point) => [
    point.value,
    point.low ?? point.value,
    point.high ?? point.value,
  ]);
  const width = 720; const height = 240; const left = 52; const right = 30; const top = 28; const bottom = 50;
  const ceiling = Math.max(...values, 1);
  const x = (index: number) => points.length === 1 ? (left + width - right) / 2 : left + (index / Math.max(points.length - 1, 1)) * (width - left - right);
  const y = (value: number) => top + (1 - value / ceiling) * (height - top - bottom);
  return <article className="benchmark-chart">
    <div><span>SAME FINGERPRINT · {points[0]?.snapshot.measurement.model.label ?? "NO DATA"}</span><h2>{label}</h2></div>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${label}. ${points.length} real tests.`}>
      <line className="benchmark-axis" x1={left} y1={height - bottom} x2={width - right} y2={height - bottom} />
      {[0, 0.5, 1].map((ratio) => <g key={ratio}><line className="benchmark-grid-line" x1={left} y1={top + ratio * (height - top - bottom)} x2={width - right} y2={top + ratio * (height - top - bottom)} /><text x={left - 8} y={top + ratio * (height - top - bottom) + 4} textAnchor="end">{formatBenchmark(ceiling * (1 - ratio), "")}</text></g>)}
      {metric === "tokensPerSecond" && points.length > 1 && points.every((point) => point.low !== null && point.high !== null) && <polygon
        className="benchmark-confidence-band"
        points={[
          ...points.map((point, index) => `${x(index)},${y(point.high ?? point.value)}`),
          ...[...points].reverse().map((point, reverseIndex) => `${x(points.length - 1 - reverseIndex)},${y(point.low ?? point.value)}`),
        ].join(" ")}
      />}
      <g className={`benchmark-series tone-${tone}`}>
        {points.length > 1 && <polyline points={points.map((point, index) => `${x(index)},${y(point.value)}`).join(" ")} />}
        {metric === "tokensPerSecond" && points.map((point, index) =>
          point.low === null || point.high === null ? null : <line
            className="benchmark-range"
            key={`range-${point.run.runId}`}
            x1={x(index)}
            x2={x(index)}
            y1={y(point.low)}
            y2={y(point.high)}
          />
        )}
        {points.map((point, index) => <circle key={point.run.runId} cx={x(index)} cy={y(point.value)} r="5"><title>{`v${point.run.version} · ${formatBenchmarkDate(point.run.finishedAt)}: ${formatBenchmark(point.value, ` ${unit}`)}${point.low === null || point.high === null ? " · interval: No reading" : ` · P5 ${formatBenchmark(point.low, ` ${unit}`)} · P95 ${formatBenchmark(point.high, ` ${unit}`)}`}`}</title></circle>)}
      </g>
      {points.map((point, index) => {
        const labelEvery = Math.max(1, Math.ceil(points.length / 6));
        if (index % labelEvery !== 0 && index !== points.length - 1) return null;
        return <text className="benchmark-x-label" key={point.run.runId} x={x(index)} textAnchor="middle">
          <tspan x={x(index)} y={height - 23}>v{point.run.version}</tspan>
          <tspan x={x(index)} y={height - 9}>{formatBenchmarkChartTime(point.run.finishedAt)}</tspan>
        </text>;
      })}
    </svg>
    <div className="benchmark-chart-summary"><span>{points.length} mediciones estrictamente comparables{metric === "tokensPerSecond" ? " · banda P5–P95" : ""}</span><strong>{formatBenchmark(points.at(-1)?.value ?? null, ` ${unit}`)}</strong></div>
  </article>;
}

function BenchmarkComparisonPanel({ comparison }: { comparison: BenchmarkRunComparison }) {
  const copy = benchmarkComparisonNarrative(comparison);
  return <article className={`benchmark-comparison ${comparison.mode} ${comparison.verdict}`}>
    <div className="benchmark-comparison-heading">
      <div><span>{copy.eyebrow}</span><h2>{copy.title}</h2><p>{copy.description}</p></div>
      {comparison.current && comparison.baseline && <div className="benchmark-comparison-versions">
        <span><small>REFERENCIA</small><strong>v{comparison.baseline.run.version}</strong><em>rev {benchmarkBuildRevision(comparison.baseline.run)}</em></span>
        <ArrowRight size={17} />
        <span><small>ACTUAL</small><strong>v{comparison.current.run.version}</strong><em>rev {benchmarkBuildRevision(comparison.current.run)}</em></span>
      </div>}
    </div>
    <div className="benchmark-comparison-grid">
      <BenchmarkComparisonMetric
        label="Velocidad"
        baseline={formatBenchmark(comparison.baseline?.tokensPerSecond ?? null, " tok/s")}
        current={formatBenchmark(comparison.current?.tokensPerSecond ?? null, " tok/s")}
        baselineDetail={formatBenchmarkRange(comparison.baseline)}
        currentDetail={formatBenchmarkRange(comparison.current)}
        delta={comparison.speedChangePct}
        claim={comparison.verdict === "faster-code-signal" || comparison.verdict === "slower-code-signal"}
      />
      <BenchmarkComparisonMetric
        label="Primer token P95"
        baseline={formatBenchmark(comparison.baseline?.ttftMsP95 ?? null, " ms")}
        current={formatBenchmark(comparison.current?.ttftMsP95 ?? null, " ms")}
        delta={comparison.latencyImprovementPct}
      />
      <BenchmarkComparisonMetric
        label="Nodes used"
        baseline={comparison.baseline ? String(comparison.baseline.nodes) : "No reading"}
        current={comparison.current ? String(comparison.current.nodes) : "No reading"}
        rawDelta={formatSignedBenchmark(comparison.nodesDelta, "")}
        neutral
      />
      <BenchmarkComparisonMetric
        label="Velocidad por nodo"
        baseline={formatBenchmark(comparison.baseline?.tokensPerSecondPerNode ?? null, " tok/s")}
        current={formatBenchmark(comparison.current?.tokensPerSecondPerNode ?? null, " tok/s")}
        delta={comparison.efficiencyPerNodeChangePct}
      />
      <BenchmarkComparisonMetric
        label="Velocidad por GB"
        baseline={formatBenchmark(comparison.baseline?.tokensPerSecondPerGb ?? null, " tok/s/GB")}
        current={formatBenchmark(comparison.current?.tokensPerSecondPerGb ?? null, " tok/s/GB")}
        delta={comparison.efficiencyPerGbChangePct}
      />
      <BenchmarkComparisonMetric
        label="Velocidad por vatio"
        baseline={formatBenchmark(comparison.baseline?.tokensPerSecondPerWatt ?? null, " tok/s/W")}
        current={formatBenchmark(comparison.current?.tokensPerSecondPerWatt ?? null, " tok/s/W")}
        delta={comparison.efficiencyPerWattChangePct}
      />
    </div>
  </article>;
}

function BenchmarkComparisonMetric({ label, baseline, current, baselineDetail, currentDetail, delta = null, rawDelta, neutral = false, claim = false }: {
  label: string;
  baseline: string;
  current: string;
  baselineDetail?: string;
  currentDetail?: string;
  delta?: number | null;
  rawDelta?: string;
  neutral?: boolean;
  claim?: boolean;
}) {
  const tone = delta === null && rawDelta === undefined
    ? "unavailable"
    : neutral || !claim
      ? "neutral"
      : (delta ?? 0) >= 0 ? "positive" : "negative";
  const deltaLabel = rawDelta ?? (delta === null
    ? "No comparison"
    : claim
      ? delta >= 0 ? `+${delta.toFixed(1)}% mejor` : `${delta.toFixed(1)}% peor`
      : `${delta > 0 ? "+" : ""}${delta.toFixed(1)}% observado`);
  return <div className="benchmark-comparison-metric">
    <small>{label}</small>
    <div><span><em>Antes</em><strong>{baseline}</strong>{baselineDetail && <small>{baselineDetail}</small>}</span><ArrowRight size={14} /><span><em>Ahora</em><strong>{current}</strong>{currentDetail && <small>{currentDetail}</small>}</span></div>
    <b className={tone}>{deltaLabel}</b>
  </div>;
}

function BenchmarkHistory({ runs, selectedRunId, onSelect }: { runs: BenchmarkRun[]; selectedRunId: string; onSelect: (runId: string) => void }) {
  return <article className="benchmark-history">
    <div className="card-heading"><div><span>EVOLUTION HISTORY</span><h2>All real tests</h2></div><small>Select a row to analyze it</small></div>
    <div className="benchmark-history-table">
      <div className="benchmark-history-head"><span>Version / build</span><span>Model / evidence</span><span>Nodes / topology</span><span>Capacity</span><span>P50 speed</span><span>Confidence</span><span>Status</span></div>
      {runs.map((run) => {
        const snapshot = benchmarkRunSnapshot(run);
        if (!snapshot) return null;
        return <button className={`benchmark-history-row ${run.runId === selectedRunId ? "selected" : ""}`} aria-pressed={run.runId === selectedRunId} key={run.runId} onClick={() => onSelect(run.runId)}>
          <span><strong>v{run.version}</strong><small>rev {benchmarkBuildRevision(run)} · src {shortSourceId(run.build.sourceId)} · {formatBenchmarkDate(run.finishedAt)}</small></span>
          <span><strong>{snapshot.measurement.model.label}</strong><small>{snapshot.measurement.evidence === "physical" ? "Real physical" : "Loopback"} · {formatBenchmarkBackend(snapshot.measurement)}</small></span>
          <span><strong>{snapshot.nodes}</strong><small>{formatTopologyDigest(snapshot.measurement.topology.digest)}</small></span>
          <span><strong>{formatBenchmark(snapshot.offeredMemoryGb, " GB")}</strong><small>{formatBenchmark(snapshot.measurement.inventory.physicalMemoryGb ?? null, " physical GB")}</small></span>
          <span><strong>{formatBenchmark(snapshot.tokensPerSecondP50, " tok/s")}</strong><small>{formatBenchmarkRange(snapshot)}</small></span>
          <span><strong>{benchmarkStabilityLabel(snapshot.measurement)}</strong><small>CV {formatBenchmark(snapshot.coefficientOfVariationPct, "%")} · TTFT {formatBenchmark(snapshot.ttftMsP95, " ms")}</small></span>
          <span><b className={run.status}><i />{benchmarkStatusLabel(run.status)}</b><ChevronRight size={15} /></span>
        </button>;
      })}
    </div>
  </article>;
}

function BenchmarkRunDetails({ run }: { run: BenchmarkRun }) {
  return <div className="benchmark-results">
    <div className="card-heading"><div><span>SELECTED RESULT</span><h2>{run.triggerModelId ?? run.label}</h2></div><small>v{run.version} · review {benchmarkBuildRevision(run)} · local source {shortSourceId(run.build.sourceId)} · node-declared sources {run.build.participantSourceIds.map(shortSourceId).join(", ") || "undeclared"} · {benchmarkDirtyLabel(run.gitDirty)}</small></div>
    <div className="benchmark-result-list">
      {run.measurements.map((measurement) => <BenchmarkResultCard measurement={measurement} run={run} key={measurement.id} />)}
    </div>
  </div>;
}

function BenchmarkResultCard({ measurement, run }: { measurement: BenchmarkMeasurement; run: BenchmarkRun }) {
  const delta = measurement.comparison.tokensPerSecondPct;
  const physicalMemory = measurement.inventory.physicalMemoryGb
    ?? nullableBenchmarkSum(measurement.inventory.profiles
      .filter((profile) => profile.kind === "gpu")
      .map((profile) => profile.physicalMemoryGb));
  const offeredMemory = measurement.inventory.offeredMemoryGb
    ?? nullableBenchmarkSum(measurement.inventory.profiles
      .filter((profile) => profile.kind === "gpu")
      .map((profile) => profile.offeredMemoryGb ?? profile.memoryGb));
  const power = measurement.metrics.powerWattsP50
    ?? measurement.inventory.observedPowerWatts
    ?? nullableBenchmarkSum(measurement.inventory.profiles.map((profile) => profile.observedPowerWatts));
  const powerLimit = measurement.inventory.powerLimitWatts
    ?? nullableBenchmarkSum(measurement.inventory.profiles.map((profile) => profile.powerLimitWatts));
  const temperaturePeak = measurement.metrics.temperatureCPeak
    ?? nullableBenchmarkMax(measurement.inventory.profiles.map((profile) => profile.temperatureC));
  return <article className="benchmark-result-card">
    <div className="benchmark-result-heading">
      <div><span className="benchmark-result-icon"><Gauge size={18} /></span><span><small>{measurement.evidence === "physical" ? "REAL PHYSICAL EVIDENCE" : "REAL LOCAL RUNTIME"}</small><strong>{measurement.model.label}</strong><em>{measurement.model.precision} · model rev {measurement.model.revision?.slice(0, 12) ?? "No reading"} · build rev {benchmarkBuildRevision(run)}</em></span></div>
      <b className={measurement.status}><i />{benchmarkStatusLabel(measurement.status)}</b>
    </div>
    <div className="benchmark-result-metrics">
      <BenchmarkValue label="Nodes used" value={String(measurement.inventory.selectedDevices)} detail={`${measurement.inventory.connectedDevices}/${measurement.inventory.totalDevices} connected`} />
      <BenchmarkValue label="Offered VRAM" value={formatBenchmark(offeredMemory, " GB")} detail={formatBenchmark(physicalMemory, " physical GB")} />
      <BenchmarkValue label="P50 power" value={formatPowerReading(power)} detail={`P95 ${formatPowerReading(measurement.metrics.powerWattsP95 ?? null)} · peak ${formatPowerReading(measurement.metrics.powerWattsPeak ?? null)}${powerLimit === null ? "" : ` · limit ${formatPower(powerLimit)}`}`} />
      <BenchmarkValue label="Tokens / segundo P50" value={formatBenchmark(measurement.metrics.tokensPerSecondP50 ?? measurement.metrics.tokensPerSecond, " tok/s")} detail={`P5 ${formatBenchmark(measurement.metrics.tokensPerSecondP5 ?? null, " tok/s")} · P95 ${formatBenchmark(measurement.metrics.tokensPerSecondP95, " tok/s")}`} accent />
      <BenchmarkValue label="Primer token" value={formatBenchmark(measurement.metrics.ttftMsP95, " ms")} detail={`P50 ${formatBenchmark(measurement.metrics.ttftMsP50, " ms")}`} />
      <BenchmarkValue label="Estabilidad" value={benchmarkStabilityLabel(measurement)} detail={`CV ${formatBenchmark(measurement.metrics.coefficientOfVariationPct ?? null, "%")} · IC ±${formatBenchmark(measurement.metrics.confidenceHalfWidthPct ?? null, "%")}`} tone={measurement.workload.statisticallyStable === false ? "negative" : measurement.workload.statisticallyStable === true ? "positive" : ""} />
      <BenchmarkValue label="Energy" value={formatBenchmark(measurement.metrics.energyWh ?? null, " Wh")} detail={`Coverage ${formatBenchmark(measurement.metrics.energyCoveragePct ?? null, "%")} · ${formatBenchmark(measurement.metrics.energyWhPerToken, " Wh/token")}`} />
      <BenchmarkValue label="Peak temperature" value={formatBenchmark(temperaturePeak, " °C")} detail={`Peak used memory ${formatBenchmark(measurement.metrics.usedMemoryGbPeak ?? null, " GB")}`} />
      <BenchmarkValue label="Saved change" value={delta === null ? "No comparison" : `${delta >= 0 ? "+" : ""}${delta.toFixed(2)}%`} detail={measurement.comparison.baselineRunId ? "Only valid with an identical fingerprint" : "No comparable run"} tone={delta === null ? "" : delta < 0 ? "negative" : "positive"} />
    </div>
    <div className="benchmark-node-list">
      {measurement.inventory.profiles.map((profile) => <span key={profile.nodeId ?? profile.label}><Server size={16} /><strong>{profile.label}</strong><small>{profile.backend ?? "No reading"} · {formatBenchmark(profile.offeredMemoryGb ?? profile.memoryGb, " GB")} · GPU {formatBenchmark(profile.utilizationPct ?? null, "%")} · {formatPowerReading(profile.observedPowerWatts ?? null)} · {formatBenchmark(profile.temperatureC ?? null, " °C")}</small></span>)}
    </div>
    <div className="benchmark-topology-summary">
      <div><Network size={18} /><span><small>OBSERVED TOPOLOGY</small><strong>{measurement.topology.stageCount === null ? "No reading" : `${measurement.topology.stageCount} stages`} · {formatTopologyDigest(measurement.topology.digest)}</strong></span></div>
      <dl>
        <div><dt>Route nodes</dt><dd>{measurement.topology.nodeIds.length > 0 ? measurement.topology.nodeIds.map(shortId).join(" → ") : "No reading"}</dd></div>
        <div><dt>Layer boundaries</dt><dd>{measurement.topology.boundaries.length > 0 ? measurement.topology.boundaries.join(" · ") : "No reading"}</dd></div>
        <div><dt>Route classes</dt><dd>{measurement.topology.routeClasses.length > 0 ? measurement.topology.routeClasses.join(" · ") : "No reading"}</dd></div>
      </dl>
    </div>
    <BenchmarkNetworkTrace traces={measurement.networkTraces ?? []} />
    <details className="benchmark-technical-details">
      <summary>View all metrics for this test</summary>
      <div>
        <BenchmarkValue label="Valid requests" value={`${formatCountReading(measurement.workload.successfulRequests)} / ${formatCountReading(measurement.workload.requests)}`} detail={`Success ${formatRate(measurement.metrics.requestSuccessRate ?? null)}`} />
        <BenchmarkValue label="Warmup" value={formatCountReading(measurement.workload.warmupRequests)} detail={`${formatCountReading(measurement.workload.recoveredFailures)} failures recovered`} />
        <BenchmarkValue label="Observed tokens" value={formatCountReading(measurement.workload.observedOutputTokens)} detail={`${measurement.workload.outputTokens} requested per call`} />
        <BenchmarkValue label="Total duration" value={formatBenchmark(measurement.workload.durationMs ?? null, " ms")} detail={`P95 latency ${formatBenchmark(measurement.metrics.latencyMsP95 ?? null, " ms")}`} />
        <BenchmarkValue label="Time per token" value={formatBenchmark(measurement.metrics.tpotMsP95, " ms")} detail={`P50 ${formatBenchmark(measurement.metrics.tpotMsP50, " ms")}`} />
        <BenchmarkValue label="Aggregate speed" value={formatBenchmark(measurement.metrics.aggregateTokensPerSecond, " tok/s")} detail={(measurement.workload.routeClasses ?? []).join(", ") || "No completed route"} />
        <BenchmarkValue label="Exactness" value={formatRate(measurement.metrics.exactnessRate ?? null)} detail={`Consistency ${formatRate(measurement.metrics.deterministicConsistencyRate ?? null)}`} />
        <BenchmarkValue label="Accepted speculation" value={formatRate(measurement.metrics.speculativeAcceptanceRate ?? null)} detail={`Inherited acceptance ${formatRate(measurement.metrics.acceptanceRate)}`} />
        <BenchmarkValue label="P50 GPU usage" value={formatBenchmark(measurement.metrics.utilizationPctP50 ?? null, "%")} detail={`P95 ${formatBenchmark(measurement.metrics.utilizationPctP95 ?? null, "%")}`} />
        <BenchmarkValue label="Campaign end" value={benchmarkCampaignStopLabel(measurement.workload.campaignStopReason)} detail={measurement.scenarioFingerprint ? measurement.scenarioFingerprint.slice(0, 19) : "No reading"} />
      </div>
      <ul>{measurement.notes.map((note) => <li key={note}>{note}</li>)}</ul>
    </details>
  </article>;
}

function BenchmarkNetworkTrace({ traces }: { traces: NetworkExecutionTrace[] }) {
  const trace = traces.at(-1) ?? null;
  if (!trace) {
    return <div className="benchmark-network-trace empty">
      <div><Wifi size={18} /><span><small>NETWORK TRACE PER REQUEST</small><strong>No reading</strong></span></div>
      <p>This run provided no physical evidence for stages, transport, RTT or bytes.</p>
    </div>;
  }
  return <div className="benchmark-network-trace">
    <div className="benchmark-network-trace-heading">
      <div><Wifi size={18} /><span><small>LATEST NETWORK TRACE · {traces.length} SAVED</small><strong>{trace.stages.length} observed stages · {trace.physicalBoundaryCount === null ? "boundaries not read" : `${trace.physicalBoundaryCount} physical boundaries`}</strong></span></div>
      <em>{trace.durationMs} ms · attempt {trace.attempt}</em>
    </div>
    <div className="benchmark-trace-stages">
      {trace.stages.length > 0
        ? trace.stages.map((stage, index) => <span key={`${stage.routeStageIndex}-${stage.stageIndex}-${index}`}>
            <b>{stage.stageIndex + 1}</b>
            <strong>{stage.nodeId ? shortId(stage.nodeId) : "Node: No reading"}</strong>
            <small>{stage.layerStart === null || stage.layerEnd === null ? "Layers: No reading" : `layers ${stage.layerStart}–${stage.layerEnd}`} · {stage.backend} · {stage.precision}</small>
            <em>{stage.workerId ? `worker ${shortId(stage.workerId)}` : "worker: No reading"} · deployment {shortId(stage.deploymentId)} · time: No reading</em>
          </span>)
        : <p>Physical stages were not observed by this runtime.</p>}
    </div>
    <div className="benchmark-trace-boundaries">
      {trace.boundaries.length > 0
        ? trace.boundaries.map((boundary) => <span key={boundary.boundaryIndex}>
            <strong>{networkTransportLabel(boundary.transport)}</strong>
            <small>{boundary.sourceNodeId ? shortId(boundary.sourceNodeId) : "No reading"} → {boundary.destinationNodeId ? shortId(boundary.destinationNodeId) : "No reading"}</small>
            <em>RTT {formatBenchmark(boundary.connectRttMs, " ms")} · bytes observados ida {formatTraceBytes(boundary.bytesSourceToDestination)} · vuelta {formatTraceBytes(boundary.bytesDestinationToSource)} · ventana {formatBenchmark(boundary.observedOverlapMs, " ms")}</em>
          </span>)
        : <p>No transport boundaries between stages.</p>}
    </div>
  </div>;
}

function networkTransportLabel(mode: NetworkExecutionTrace["boundaries"][number]["transport"]): string {
  if (mode === "direct") return "Direct";
  if (mode === "relay") return "Relay";
  if (mode === "local") return "Local";
  return "No reading";
}

function formatTraceBytes(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "No reading";
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(2)} MB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${value} B`;
}

function BenchmarkMetric({ icon: Icon, label, value, detail }: { icon: typeof Gauge; label: string; value: string; detail: string }) {
  return <div className="benchmark-capacity-metric"><Icon size={18} /><span><small>{label}</small><strong>{value}</strong><em>{detail}</em></span></div>;
}

function BenchmarkValue({ label, value, detail, accent = false, tone = "" }: { label: string; value: string; detail: string; accent?: boolean; tone?: string }) {
  return <div className={`benchmark-value ${accent ? "accent" : ""} ${tone}`}><small>{label}</small><strong>{value}</strong><em>{detail}</em></div>;
}

interface InferenceTurn {
  id: string;
  prompt: string;
  messageContent: string;
  attachments: InferenceAttachmentSummary[];
  response: ChatResponse;
}

interface InferencePendingTurn extends ChatStreamUpdate {
  prompt: string;
  attachments: InferenceAttachmentSummary[];
}

interface InferenceAttachmentSummary {
  id: string;
  name: string;
  size: number;
  kind: "pdf" | "docx" | "text";
  truncated: boolean;
}

interface InferenceAttachment extends InferenceAttachmentSummary {
  text: string;
}

const MAX_INFERENCE_FILES = 5;
const MAX_INFERENCE_FILE_BYTES = 10 * 1024 * 1024;
const MAX_INFERENCE_FILE_CHARS = 18_000;
const MAX_INFERENCE_TOTAL_CHARS = 36_000;
const INFERENCE_FILE_ACCEPT = [
  ".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml", ".toml",
  ".html", ".css", ".js", ".jsx", ".ts", ".tsx", ".py", ".java", ".c", ".cpp", ".h",
  ".sql", ".log", ".ini", ".env",
].join(",");

function Inference({ snapshot, onSend, onNavigate, developerMode, accountAuthenticated, onSignIn, apiAccessEnabled, apiBaseUrl, accessToken, apiAccount }: {
  snapshot: PublicSnapshot;
  onSend: (model: string, messages: ChatMessage[], sessionId: string, onUpdate?: (update: ChatStreamUpdate) => void) => Promise<ChatResponse>;
  onNavigate: (view: PanelView) => void;
  developerMode: boolean;
  accountAuthenticated: boolean;
  onSignIn: () => void;
  apiAccessEnabled: boolean;
  apiBaseUrl: string;
  accessToken: string | null;
  apiAccount: ApiAccount | null;
}) {
  const options = useMemo(() => snapshot.models.map((item) => inferenceModelOption(snapshot, item)), [snapshot]);
  const realModels = options.filter((item) => !item.legacyExternalRuntime);
  const [model, setModel] = useState("");
  // The landing hero hands off its prompt through sessionStorage so the visitor
  // lands in the chat with what they already typed, instead of an empty box.
  const [prompt, setPrompt] = useState(() => {
    const handoff = window.sessionStorage.getItem("mycellios.hero-prompt");
    if (handoff) window.sessionStorage.removeItem("mycellios.hero-prompt");
    return handoff ?? "";
  });
  const [attachments, setAttachments] = useState<InferenceAttachment[]>([]);
  const [attachmentError, setAttachmentError] = useState<string | null>(null);
  const [filesBusy, setFilesBusy] = useState(false);
  const [turns, setTurns] = useState<InferenceTurn[]>([]);
  const [sessionId, setSessionId] = useState(() => newInferenceSessionId());
  const [error, setError] = useState<string | null>(null);
  const [pendingTurn, setPendingTurn] = useState<InferencePendingTurn | null>(null);
  const [instructionsOpen, setInstructionsOpen] = useState(false);
  const [instructions, setInstructions] = useState(() => window.localStorage.getItem("mycellios.chat.instructions") ?? "");
  const outputRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const modelAvailable = realModels.length > 0;
  const inferenceAvailable = modelAvailable && accountAuthenticated;
  const selectedModel = realModels.some((item) => item.id === model) ? model : realModels[0]?.id ?? "";
  const selectedOption = realModels.find((item) => item.id === selectedModel) ?? null;

  function saveInstructions(next: string) {
    setInstructions(next);
    if (next.trim()) window.localStorage.setItem("mycellios.chat.instructions", next);
    else window.localStorage.removeItem("mycellios.chat.instructions");
  }

  useEffect(() => {
    if (!pendingTurn) return;
    const output = outputRef.current;
    if (output) output.scrollTop = output.scrollHeight;
  }, [pendingTurn?.text, pendingTurn?.outputTokens]);

  useEffect(() => {
    if (!pendingTurn) return;
    const timer = window.setInterval(() => {
      setPendingTurn((current) => current
        ? { ...current, elapsedMs: current.elapsedMs + 250 }
        : current);
    }, 250);
    return () => window.clearInterval(timer);
  }, [pendingTurn !== null]);

  async function addFiles(fileList: FileList | null) {
    if (!fileList || fileList.length === 0 || filesBusy) return;
    const availableSlots = Math.max(0, MAX_INFERENCE_FILES - attachments.length);
    if (availableSlots === 0) {
      setAttachmentError(`Puedes adjuntar hasta ${MAX_INFERENCE_FILES} archivos por mensaje.`);
      return;
    }
    setFilesBusy(true);
    setAttachmentError(null);
    try {
      const next: InferenceAttachment[] = [];
      let remainingChars = Math.max(0, MAX_INFERENCE_TOTAL_CHARS - attachments.reduce((total, attachment) => total + attachment.text.length, 0));
      for (const file of Array.from(fileList).slice(0, availableSlots)) {
        if (remainingChars === 0) throw new Error("Los documentos adjuntos ya ocupan todo el contexto permitido para este mensaje.");
        const attachment = await readInferenceAttachment(file, remainingChars);
        next.push(attachment);
        remainingChars -= attachment.text.length;
      }
      setAttachments((current) => [...current, ...next]);
      if (fileList.length > availableSlots) setAttachmentError(`Only ${availableSlots} files were added; the limit is ${MAX_INFERENCE_FILES}.`);
    } catch (caught) {
      setAttachmentError(errorText(caught));
    } finally {
      setFilesBusy(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function send() {
    const cleanPrompt = prompt.trim();
    const attachedFiles = attachments;
    if (!selectedModel || (!cleanPrompt && attachedFiles.length === 0) || pendingTurn || filesBusy) return;
    const displayPrompt = cleanPrompt || "Analiza los archivos adjuntos.";
    const messageContent = inferenceMessageWithAttachments(displayPrompt, attachedFiles);
    const attachmentSummaries = attachedFiles.map(({ id, name, size, kind, truncated }) => ({ id, name, size, kind, truncated }));
    setPendingTurn({
      prompt: displayPrompt,
      attachments: attachmentSummaries,
      requestId: "pending",
      model: selectedModel,
      delta: "",
      text: "",
      outputTokens: 0,
      routeClass: "waiting",
      affinityHit: false,
      sessionId,
      reusedKvTokens: 0,
      ttftMs: 0,
      elapsedMs: 0,
    });
    setPrompt("");
    setAttachments([]);
    setAttachmentError(null);
    setError(null);
    try {
      const messages: ChatMessage[] = [
        ...(instructions.trim() ? [{ role: "developer" as const, content: instructions.trim() }] : []),
        ...turns.flatMap((turn): ChatMessage[] => [
          { role: "user", content: turn.messageContent },
          { role: "assistant", content: turn.response.text },
        ]),
        { role: "user", content: messageContent },
      ];
      const response = await onSend(selectedModel, messages, sessionId, (update) => {
        setPendingTurn((current) => current ? { ...current, ...update } : current);
      });
      if (!response.text.trim()) throw new Error("The model finished without returning text.");
      setTurns((current) => [...current, { id: response.requestId, prompt: displayPrompt, messageContent, attachments: attachmentSummaries, response }]);
    } catch (caught) {
      setPrompt(cleanPrompt);
      setAttachments(attachedFiles);
      setError(errorText(caught));
    } finally {
      setPendingTurn(null);
    }
  }

  function resetConversation(nextModel?: string) {
    if (nextModel !== undefined) setModel(nextModel);
    setTurns([]);
    setPrompt("");
    setAttachments([]);
    setAttachmentError(null);
    setError(null);
    setPendingTurn(null);
    setSessionId(newInferenceSessionId());
  }

  const apiAccessPanel = <ApiAccessPanel enabled={apiAccessEnabled} apiBaseUrl={apiBaseUrl} accessToken={accessToken} account={apiAccount} availableModels={realModels.length} onSignIn={onSignIn} />;

  if (!modelAvailable) return <section className="inference-page inference-empty-page">
    <PageTitle eyebrow="REAL INFERENCE" title="Test a model" copy="Chat with a connected model when the network confirms a real runtime." />
    {developerMode ? apiAccessPanel : <PanelDisclosure icon={Code2} eyebrow="OPTIONAL" title="API access" summary="Keys, usage, and the OpenAI-compatible endpoint.">{apiAccessPanel}</PanelDisclosure>}
    <div className="inference-unavailable focused-empty">
      <div className="inference-unavailable-icon"><MessageSquareText /></div>
      <div className="inference-unavailable-copy"><span>NO AI MODELS AVAILABLE</span><h2>Connect a model to start chatting</h2><p>The network is healthy, but no verified inference runtime is available yet.</p>
        {developerMode && snapshot.requestedModels[0] && <div className="inference-request-state"><LoaderCircle className={snapshot.requestedModels[0].status === "active" ? "" : "spin"} /><span><strong>{snapshot.requestedModels[0].id}</strong>{snapshot.requestedModels[0].message}</span></div>}
        <div className="inference-unavailable-actions"><button className="primary-button" onClick={() => onNavigate("models")}><Boxes size={16} />View and activate models</button></div>
      </div>
    </div>
  </section>;

  return <section className="inference-page">
    <PageTitle eyebrow="REAL INFERENCE" title="Test a model" copy="Chat with a connected model and inspect the route, latency and tokens for every response." />
    {developerMode ? apiAccessPanel : <PanelDisclosure icon={Code2} eyebrow="OPTIONAL" title="API access" summary="Keys, usage, and the OpenAI-compatible endpoint.">{apiAccessPanel}</PanelDisclosure>}
    {modelAvailable && !accountAuthenticated && <div className="inference-unavailable account-required">
      <div className="inference-unavailable-icon"><LockKeyhole /></div>
      <div className="inference-unavailable-copy"><span>ACCOUNT REQUIRED</span><h2>The model is connected; sign in to use it</h2><p>Sending uses your account balance and limits. Chat remains visible so you can see how it works.</p><div className="inference-unavailable-actions"><button className="primary-button" onClick={onSignIn}><UserRound size={16} />Sign in</button></div></div>
    </div>}
    <div className={`inference-console${inferenceAvailable ? "" : " is-unavailable"}`} aria-disabled={!inferenceAvailable}>
      <div className="inference-toolbar">
        <div className="inference-toolbar-title"><MessageSquareText /><span><strong>Chat</strong><small>Real-time distributed inference</small></span></div>
        {developerMode && <div className="inference-toolbar-controls">
          <span className="inference-toolbar-stat"><Network /><b>{selectedOption?.nodeCount ?? 0}</b> node{selectedOption?.nodeCount === 1 ? "" : "s"}</span>
          <span className="inference-toolbar-stat"><MemoryStick /><b>{selectedOption && selectedOption.peerMemoryMb > 0 ? formatMemory(selectedOption.peerMemoryMb) : "—"}</b> memory</span>
        </div>}
      </div>
      {developerMode && <div className="inference-route-strip">
        {inferenceAvailable
          ? <div className="inference-model-summary"><ExecutionBadge execution={selectedOption?.execution ?? unknownExecutionSummary()} workers={snapshot.workers} large /><span className="inference-live"><i />AVAILABLE</span><span>{selectedOption?.routeLabel}</span><span>{selectedOption?.freeSlots ?? 0} free slot{selectedOption?.freeSlots === 1 ? "" : "s"}</span></div>
          : <div className="inference-model-summary inference-offline"><LockKeyhole size={14} /><span>{modelAvailable ? "SIGN IN" : "NO REAL CONNECTION"}</span></div>}
      </div>}
      <div className="inference-output" aria-live="polite" ref={outputRef}>
        {turns.length === 0 && !pendingTurn && !error && (inferenceAvailable
          ? <div className="inference-welcome"><Sparkles /><h2>Ask a question</h2><p>The response comes exclusively from the selected native Mycellios deployment.</p><div className="inference-suggestions">{["Summarize how this network works", "Explain an idea in three sentences", "Answer with a short proof"].map((suggestion) => <button key={suggestion} onClick={() => setPrompt(suggestion)}>{suggestion}</button>)}</div></div>
          : <div className="inference-welcome inference-welcome-locked"><LockKeyhole /><h2>{modelAvailable ? "Chat is waiting for your account" : "Chat is waiting for a model"}</h2><p>{modelAvailable ? "Sign in to activate sending with your token balance." : "You can write and send messages once the network confirms a real inference connection."}</p></div>)}
        {turns.map((turn) => <InferenceCompletedTurn turn={turn} compact={!developerMode} key={turn.id} />)}
        {pendingTurn && <InferenceStreamingTurn turn={pendingTurn} compact={!developerMode} />}
        {error && <div className="inference-error"><CircleAlert /><div><strong>Inference could not be completed</strong><span>{friendlyInferenceError(error)}</span></div></div>}
      </div>
      <div className="inference-input">
        {attachments.length > 0 && <div className="inference-attachments" aria-label="Attached files">{attachments.map((attachment) => <div key={attachment.id}><FileText /><span><strong>{attachment.name}</strong><small>{formatFileSize(attachment.size)} · {attachment.kind.toUpperCase()}{attachment.truncated ? " · truncated" : ""}</small></span><button type="button" onClick={() => setAttachments((current) => current.filter((item) => item.id !== attachment.id))} aria-label={`Remove ${attachment.name}`}><X /></button></div>)}</div>}
        {attachmentError && <div className="inference-attachment-error"><CircleAlert />{attachmentError}</div>}
        <div className="inference-composer-row">
          <input ref={fileInputRef} hidden type="file" multiple accept={INFERENCE_FILE_ACCEPT} onChange={(event) => void addFiles(event.target.files)} />
          <button type="button" className="inference-attach" disabled={!inferenceAvailable || pendingTurn !== null || filesBusy || attachments.length >= MAX_INFERENCE_FILES} onClick={() => fileInputRef.current?.click()} aria-label="Attach documents" title="PDF, DOCX, text, code, CSV or JSON">{filesBusy ? <LoaderCircle className="spin" /> : <Paperclip />}</button>
          <textarea aria-label="Message" value={prompt} disabled={!inferenceAvailable || pendingTurn !== null} onChange={(event) => setPrompt(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void send(); } }} placeholder={inferenceAvailable ? `Write to ${model ? selectedModel : "automatic mesh"}…` : modelAvailable ? "Sign in to start writing…" : "Connect a real model to start writing…"} />
        </div>
        <div className="inference-input-foot"><div className="inference-composer-tools">
          <InferenceModelPicker options={realModels} value={model} onChange={(nextModel) => resetConversation(nextModel)} />
          <div className={`inference-instructions${instructionsOpen ? " open" : ""}`}>
            <button type="button" aria-label="Conversation instructions" aria-expanded={instructionsOpen} onClick={() => setInstructionsOpen((current) => !current)}><SlidersHorizontal />{instructions.trim() && <i />}</button>
            {instructionsOpen && <div className="inference-instructions-popover"><header><strong>Conversation instructions</strong><span>Sent before each message.</span></header><textarea value={instructions} onChange={(event) => saveInstructions(event.target.value)} placeholder="For example: answer clearly and flag uncertainty." maxLength={2_000} /><div className="inference-instruction-presets">{["Direct answers", "Step by step", "Code first", "Short summary"].map((preset) => <button type="button" key={preset} onClick={() => saveInstructions(instructions.trim() ? `${instructions.trim()}\n${preset}.` : `${preset}.`)}>{preset}</button>)}</div><footer><button type="button" onClick={() => saveInstructions("")}>Clear</button><button type="button" onClick={() => setInstructionsOpen(false)}>Done</button></footer></div>}
          </div>
          <span>{inferenceAvailable ? <><Paperclip size={12} />Files · <kbd>Enter</kbd> send</> : <><LockKeyhole size={12} />{modelAvailable ? "Sign in to send" : "Connect a real model"}</>}</span>
        </div><div>{turns.length > 0 && <button className="inference-clear" onClick={() => resetConversation()}><Trash2 size={14} />New conversation</button>}<button className="inference-send" disabled={!inferenceAvailable || (!prompt.trim() && attachments.length === 0) || pendingTurn !== null || filesBusy} onClick={() => void send()}>{pendingTurn ? <LoaderCircle className="spin" /> : <Send />}<span>Send</span></button></div></div>
      </div>
    </div>
  </section>;
}

function InferenceStreamingTurn({ turn, compact = false }: { turn: InferencePendingTurn; compact?: boolean }) {
  const content = splitStreamingContent(turn.text);
  const waitingStatus = inferenceWaitingStatus(turn);
  return <div className="inference-turn pending">
    <InferenceUserMessage prompt={turn.prompt} attachments={turn.attachments} />
    {turn.text ? <div className="inference-message streaming"><img src={brandIcon} alt="" /><div>
      <div className="inference-stream-head"><span>{turn.model}</span><b><i />GENERATING LIVE</b></div>
      {turn.phase === "recovering" && <div className="inference-stream-recovery"><LoaderCircle className="spin" size={14} />{turn.statusMessage ?? "Reconnecting and recovering the response…"} <small>attempt {turn.attempt ?? 1} of {turn.maximumAttempts ?? 8}</small></div>}
      {content.reasoning !== null && <div className="inference-live-reasoning"><span>REASONING</span><p>{content.reasoning}{content.answer === "" && <i className="stream-cursor" />}</p></div>}
      {content.answer && <p>{content.answer}<i className="stream-cursor" /></p>}
      {!compact && <div className="inference-response-metrics live"><span><b>{formatDuration(turn.ttftMs)}</b>first token</span><span><b>{formatDuration(turn.elapsedMs)}</b>current time</span><span><b>{turn.outputTokens}</b>tokens received</span><span><b>{formatLiveThroughput(turn)}</b>tokens/s now</span><span><b>{turn.routeClass}</b>route</span>{turn.affinityHit && <span><b>AFFINITY</b>same route</span>}</div>}
    </div></div> : <div className={`inference-thinking ${turn.phase === "recovering" ? "recovering" : ""}`}><LoaderCircle className="spin" /><span><strong>{waitingStatus}</strong><small>{formatDuration(turn.elapsedMs)}{(turn.attempt ?? 1) > 1 ? ` · attempt ${turn.attempt} of ${turn.maximumAttempts ?? 8}` : ""}</small></span></div>}
  </div>;
}

function InferenceCompletedTurn({ turn, compact = false }: { turn: InferenceTurn; compact?: boolean }) {
  const content = splitThinkingContent(turn.response.text);
  const metrics = <div className="inference-response-metrics"><span><b>{formatDuration(turn.response.ttftMs)}</b>first token</span><span><b>{formatDuration(turn.response.activeMs)}</b>total time</span><span><b>{turn.response.outputTokens}</b>output tokens</span><span><b>{formatResponseThroughput(turn.response)}</b>tokens/s</span><span><b>{turn.response.routeClass}</b>route</span>{turn.response.reusedKvTokens > 0 ? <span><b>{turn.response.reusedKvTokens}</b>KV tokens reused</span> : turn.response.affinityHit ? <span><b>AFFINITY</b>same route</span> : null}</div>;
  return <div className="inference-turn">
    <InferenceUserMessage prompt={turn.prompt} attachments={turn.attachments} />
    {!compact && turn.response.activity && <InferenceActivitySummary activity={turn.response.activity} />}
    <div className="inference-message"><img src={brandIcon} alt="" /><div><span>{turn.response.model}</span>{content.reasoning && <details className="inference-reasoning"><summary>View model reasoning</summary><p>{content.reasoning}</p></details>}<p>{content.answer}</p>{!compact && metrics}</div></div>
    {compact && <details className="inference-evidence"><summary><Activity size={14} />Response details<ChevronDown size={14} /></summary>{turn.response.activity && <InferenceActivitySummary activity={turn.response.activity} />}{metrics}</details>}
  </div>;
}

function InferenceActivitySummary({ activity }: { activity: NonNullable<ChatResponse["activity"]> }) {
  const status = activity.goal?.status ?? "in_progress";
  const storageKey = `mycellios.remembered-activity.${activity.goal?.title ?? activity.model}`;
  const [remembered, setRemembered] = useState(() => window.localStorage.getItem(storageKey) !== null);
  function toggleRemembered() {
    if (remembered) window.localStorage.removeItem(storageKey);
    else window.localStorage.setItem(storageKey, JSON.stringify(activity));
    setRemembered(!remembered);
  }
  return <aside className="inference-activity-summary">
    <header><span><strong>Model {activity.model}</strong>{activity.effort && <><i />Effort {activity.effort}</>}</span>{activity.rememberable && <button type="button" aria-pressed={remembered} onClick={toggleRemembered}>{remembered ? <Check /> : <History />}{remembered ? "Remembered" : "Remember this"}</button>}</header>
    {activity.goal && <section><span className={`inference-activity-status ${status}`}>{status.replaceAll("_", " ")}</span><h3>{activity.goal.title}</h3><small>{activity.goal.tokens !== null ? `${activity.goal.tokens.toLocaleString()} tokens` : ""}{activity.goal.tokens !== null && activity.goal.elapsedMs !== null ? " · " : ""}{activity.goal.elapsedMs !== null ? formatDuration(activity.goal.elapsedMs) : ""}</small></section>}
    {activity.requestSummary && <section><span>Request summary</span><p>{activity.requestSummary}</p></section>}
    {activity.decision && <section className="inference-activity-decision"><span>{activity.goal?.status === "needs_decision" ? "Needs decision" : "Outcome"}</span><h3>{activity.decision.headline}</h3>{activity.decision.pending.length > 0 && <ul>{activity.decision.pending.map((item) => <li key={item}>{item}</li>)}</ul>}{activity.decision.userAction && <p><strong>Your action:</strong> {activity.decision.userAction}</p>}</section>}
  </aside>;
}

function InferenceUserMessage({ prompt, attachments }: { prompt: string; attachments: InferenceAttachmentSummary[] }) {
  return <div className="inference-user-message"><span>YOU</span><p>{prompt}</p>{attachments.length > 0 && <div className="inference-message-files">{attachments.map((attachment) => <span key={attachment.id}><FileText /><b>{attachment.name}</b><small>{formatFileSize(attachment.size)}</small></span>)}</div>}</div>;
}

type InferenceModelOption = ReturnType<typeof inferenceModelOption>;

function InferenceModelPicker({ options, value, onChange }: { options: InferenceModelOption[]; value: string; onChange: (value: string) => void }) {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const selected = options.find((option) => option.id === value) ?? null;
  const label = selected?.id ?? "Mesh — automatic";
  return <div className={`inference-model-picker${open ? " open" : ""}`} onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false); }}>
    <button ref={triggerRef} type="button" aria-label="Select chat model" aria-haspopup="listbox" aria-expanded={open} disabled={options.length === 0} onClick={() => setOpen((current) => !current)}>
      <Cpu /><span><small>MODEL</small><strong>{label}</strong></span><ChevronDown />
    </button>
    {open && <div className="inference-model-menu" role="listbox" aria-label="Available models">
      <button type="button" role="option" aria-selected={value === ""} className={value === "" ? "selected" : ""} onMouseDown={(event) => event.preventDefault()} onClick={() => { onChange(""); setOpen(false); triggerRef.current?.focus(); }}>
        <span><strong>Mesh — automatic</strong><small>The network chooses the best available route</small></span><b className="auto"><i />AUTO</b>{value === "" && <Check />}
      </button>
      {options.map((option) => <button type="button" role="option" aria-selected={option.id === value} className={option.id === value ? "selected" : ""} key={option.id} onMouseDown={(event) => event.preventDefault()} onClick={() => { onChange(option.id); setOpen(false); triggerRef.current?.focus(); }}>
        <span><strong>{option.id}</strong><small>{option.nodeCount} node{option.nodeCount === 1 ? "" : "s"} · {option.peerMemoryMb > 0 ? formatMemory(option.peerMemoryMb) : "memory not reported"}</small></span><b className={option.freeSlots > 0 ? "warm" : "ready"}><i />{option.freeSlots > 0 ? "WARM" : "READY"}</b>{option.id === value && <Check />}
      </button>)}
    </div>}
  </div>;
}

export async function readInferenceAttachment(file: File, remainingChars: number): Promise<InferenceAttachment> {
  if (file.size === 0) throw new Error(`${file.name} is empty.`);
  if (file.size > MAX_INFERENCE_FILE_BYTES) throw new Error(`${file.name} exceeds the ${formatFileSize(MAX_INFERENCE_FILE_BYTES)} limit.`);
  const extension = file.name.split(".").at(-1)?.toLowerCase() ?? "";
  let text = "";
  let kind: InferenceAttachment["kind"] = "text";
  if (extension === "pdf" || file.type === "application/pdf") {
    kind = "pdf";
    const pdfjs = await import("pdfjs-dist");
    pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();
    const document = await pdfjs.getDocument({ data: new Uint8Array(await file.arrayBuffer()) }).promise;
    const pages: string[] = [];
    for (let pageNumber = 1; pageNumber <= document.numPages; pageNumber += 1) {
      const page = await document.getPage(pageNumber);
      const content = await page.getTextContent();
      pages.push(`[Page ${pageNumber}]\n${content.items.map((item) => "str" in item ? item.str : "").join(" ")}`);
      if (pages.join("\n\n").length >= Math.min(MAX_INFERENCE_FILE_CHARS, remainingChars)) break;
    }
    text = pages.join("\n\n");
  } else if (extension === "docx" || file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document") {
    kind = "docx";
    const mammoth = await import("mammoth");
    const result = await mammoth.extractRawText({ arrayBuffer: await file.arrayBuffer() });
    text = result.value;
  } else if (file.type.startsWith("text/") || inferenceTextExtension(extension)) {
    text = await file.text();
  } else {
    throw new Error(`${file.name} is not supported. Use PDF, DOCX, text, Markdown, CSV, JSON or code files.`);
  }
  const normalized = text.replace(/\u0000/g, "").replace(/\r\n/g, "\n").trim();
  if (!normalized) throw new Error(`Could not extract text from ${file.name}. Scanned PDFs need OCR before attaching.`);
  const limit = Math.max(1, Math.min(MAX_INFERENCE_FILE_CHARS, remainingChars));
  const truncated = normalized.length > limit;
  return {
    id: crypto.randomUUID(),
    name: file.name.replace(/[\u0000-\u001f\u007f]/g, "").slice(0, 180) || "documento",
    size: file.size,
    kind,
    truncated,
    text: normalized.slice(0, limit),
  };
}

function inferenceTextExtension(extension: string): boolean {
  return new Set(["txt", "md", "csv", "json", "xml", "yaml", "yml", "toml", "html", "css", "js", "jsx", "ts", "tsx", "py", "java", "c", "cpp", "h", "sql", "log", "ini", "env"]).has(extension);
}

export function inferenceMessageWithAttachments(prompt: string, attachments: InferenceAttachment[]): string {
  if (attachments.length === 0) return prompt;
  const documents = attachments.map((attachment, index) => [
    `--- ARCHIVO ${index + 1}: ${attachment.name} (${formatFileSize(attachment.size)})${attachment.truncated ? " · CONTENIDO RECORTADO" : ""} ---`,
    attachment.text,
    `--- FIN DE ${attachment.name} ---`,
  ].join("\n")).join("\n\n");
  return `${prompt}\n\nEl usuario ha adjuntado los siguientes documentos como material de referencia. Analiza su contenido y distingue claramente lo que procede de los archivos.\n\n${documents}`;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(bytes < 10 * 1024 ? 1 : 0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function JoinNetwork({ publicLink, external }: { publicLink: (path: string) => string; external: boolean }) {
  return <section><PageTitle eyebrow="ZERO-CONFIG JOIN" title="Join the network" copy="No account, invitation or terminal. Choose an option, review the detected hardware and confirm participation." />
    <div className="join-grid"><a className="join-card featured" href={publicLink("/mobile/?autostart=1")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><div><Smartphone /><span>INSTANT</span></div><h2>Use this browser</h2><p>Works on modern Android, iPhone, tablet and desktop browsers. Keep the page visible while contributing.</p><ul><li><CheckCircle2 />WebGPU when available</li><li><CheckCircle2 />Automatic CPU fallback</li><li><CheckCircle2 />One explicit confirmation</li></ul><strong>Open and confirm <ArrowRight /></strong></a>
      <a className="join-card" href={publicLink("/downloads")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><div><Laptop /><span>WINDOWS · MACOS · LINUX</span></div><h2>Install the desktop agent</h2><p>Detects the hardware, prepares the certified runtime automatically and contributes on CPU while GPU setup finishes.</p><ul><li><CheckCircle2 />Windows CUDA/ROCm and Apple Silicon MPS when certified</li><li><CheckCircle2 />Automatic verified downloads</li><li><CheckCircle2 />Safe CPU fallback</li></ul><strong>Choose your computer <Download /></strong></a></div>
  </section>;
}

function Downloads({ publicLink, external }: { publicLink: (path: string) => string; external: boolean }) {
  return <section><PageTitle eyebrow="CLIENTS" title="Downloads" copy="Install the native client or enter immediately through the universal browser worker." />
    <div className="download-grid"><a className="download-card ready" href={publicLink("/downloads/windows")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Laptop />WINDOWS</span><h2>Windows 10/11</h2><p>x64 installer · automatic CPU runtime, CUDA verification and ROCm on certified Windows 11 hardware</p><strong>Download installer <Download /></strong></a><a className="download-card ready" href={publicLink("/downloads/macos-arm64")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Cpu />MACOS · APPLE SILICON</span><h2>Mac M1 or newer</h2><p>macOS 14+ arm64 DMG · bundled PyTorch runtime with automatic Metal / MPS verification</p><strong>Download DMG <Download /></strong></a><a className="download-card ready" href={publicLink("/downloads/linux-deb")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Cpu />LINUX · DEBIAN</span><h2>Ubuntu / Debian</h2><p>x64 DEB · automatic certified CPU runtime; compatible GPUs can contribute through the browser worker</p><strong>Download DEB <Download /></strong></a><a className="download-card ready" href={publicLink("/downloads/linux-rpm")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Cpu />LINUX · RPM</span><h2>Fedora / RHEL</h2><p>x64 RPM · automatic certified CPU runtime; compatible GPUs can contribute through the browser worker</p><strong>Download RPM <Download /></strong></a><a className="download-card ready" href={publicLink("/mobile/?autostart=1")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Globe2 />UNIVERSAL</span><h2>Browser worker</h2><p>Android, iOS and desktops · real WebGPU probe with automatic CPU fallback after confirmation</p><strong>Open and confirm <ExternalLink /></strong></a></div>
  </section>;
}

function Machine({ snapshot, bridge, onSnapshot }: { snapshot: DashboardSnapshot; bridge: DesktopBridge; onSnapshot: (snapshot: DashboardSnapshot) => void }) {
  const [busy, setBusy] = useState(false);
  const localWorker = snapshot.workers.find((worker) => worker.id === snapshot.contribution.workerId);
  const advertisedGpu = localWorker?.gpus[0];
  const contributionLabel = snapshot.contribution.state === "connected"
    ? "Connected"
    : snapshot.contribution.state === "connecting"
      ? "Connecting…"
      : snapshot.contribution.state === "error"
        ? "Connection error"
        : "Paused";
  async function toggleContribution() {
    setBusy(true);
    try { onSnapshot(await bridge.setContribution(!snapshot.settings.contributionEnabled)); }
    finally { setBusy(false); }
  }
  async function selectComputeMode(computeMode: DesktopSettings["computeMode"]) {
    if (computeMode === snapshot.settings.computeMode) return;
    setBusy(true);
    try { onSnapshot(await bridge.saveSettings({ ...snapshot.settings, computeMode })); }
    finally { setBusy(false); }
  }
  const computeModes: Array<{
    value: DesktopSettings["computeMode"];
    label: string;
    detail: string;
    Icon: typeof Cpu;
  }> = [
    { value: "automatic", label: "Automatic", detail: "Prefer the verified GPU; use CPU only when GPU execution is unavailable.", Icon: Sparkles },
    { value: "gpu-only", label: "GPU only", detail: "Wait for CUDA, ROCm or MPS. Never run model stages on CPU.", Icon: Zap },
    { value: "cpu-only", label: "CPU only", detail: "Do not prepare or use the GPU. Contribute with system memory and CPU.", Icon: Cpu },
  ];
  return <section className="content-page">
    <PageTitle eyebrow="NATIVE NODE" title="This device" copy="Hardware detected on this computer and the resources offered to mycellios." actions={<button disabled={busy} onClick={() => void toggleContribution()}><CirclePower size={16} />{snapshot.settings.contributionEnabled ? "Pause contribution" : "Start contributing"}</button>} />
    <div className="hardware-hero">
      <div className="device-orb"><Cpu size={42} /></div>
      <div><span className="eyebrow">{snapshot.platform.toUpperCase()}</span><h2>{snapshot.localHardware.hostname}</h2><p>{snapshot.localHardware.gpus.length} accelerator(s) · {formatMemory(snapshot.localHardware.ramMb)} RAM</p></div>
      <span className={`state-badge ${snapshot.contribution.state === "connected" ? "online" : snapshot.contribution.state}`}>{contributionLabel}</span>
    </div>
    <article className="compute-mode-panel">
      <div className="compute-mode-heading"><div><span>COMPUTE PREFERENCE</span><h2>Choose how this device contributes</h2><p>The coordinator will only use CPU when this device explicitly allows it.</p></div><b>{snapshot.settings.computeMode === "automatic" ? "RECOMMENDED" : "USER SELECTED"}</b></div>
      <div className="compute-mode-grid" role="radiogroup" aria-label="Compute preference">
        {computeModes.map(({ value, label, detail, Icon }) => <button type="button" role="radio" aria-checked={snapshot.settings.computeMode === value} className={snapshot.settings.computeMode === value ? "active" : ""} disabled={busy} onClick={() => void selectComputeMode(value)} key={value}><span><Icon /></span><strong>{label}</strong><small>{detail}</small>{snapshot.settings.computeMode === value && <CheckCircle2 />}</button>)}
      </div>
    </article>
    <AcceleratorProgressPanel acceleration={snapshot.acceleration} contributionState={snapshot.contribution.state} computeMode={snapshot.settings.computeMode} />
    <div className="cards-grid">
      {snapshot.localHardware.gpus.map((gpu) => {
        const selected = isAdvertisedGpuSelected(gpu, advertisedGpu);
        const offered = selected && advertisedGpu ? advertisedGpu.offeredVramMb : 0;
        const budget = gpu.physicalVramMb + (gpu.sharedMemoryMb ?? 0);
        return <article className={`hardware-card ${selected ? "selected" : ""}`} key={gpu.id}><div className="hardware-card-icon"><Gauge size={23} /></div><span className="card-kicker">{gpu.vendor} · {selected ? "SELECTED" : "DETECTED"}</span><h3>{gpu.model}</h3><div className="capacity-line"><span style={{ width: `${budget > 0 ? Math.min(100, offered / budget * 100) : 0}%` }} /></div><div className="hardware-metrics"><DesktopMetric label="Physical VRAM" value={formatMemory(gpu.physicalVramMb)} /><DesktopMetric label="Offered" value={selected ? formatMemory(offered) : "—"} /><DesktopMetric label="Utilization" value={`${gpu.utilizationPct ?? 0}%`} /><DesktopMetric label="Temperature" value={gpu.temperatureC === undefined ? "—" : `${gpu.temperatureC} °C`} /></div></article>;
      })}
      {snapshot.localHardware.gpus.length === 0 && <div className="wide-empty"><Empty icon={Cpu} title="No GPU detected" copy="This machine can register its real hardware, but it will not advertise a model until a compatible runtime is active." /></div>}
    </div>
  </section>;
}

function AcceleratorCompactBanner({ acceleration, contributionState, computeMode, onOpen }: { acceleration: AcceleratorProgressSnapshot; contributionState: DashboardSnapshot["contribution"]["state"] | undefined; computeMode: DesktopSettings["computeMode"] | undefined; onOpen: () => void }) {
  const cpuLabel = accelerationCpuLabel(acceleration, contributionState, computeMode);
  const cpuUnavailable = acceleration.cpu?.state === "unavailable";
  const cpuStandby = computeMode === "gpu-only" && (acceleration.cpu?.activeStages ?? 0) === 0;
  const cpuVisualState = (acceleration.cpu?.activeStages ?? 0) > 0 ? "active" : cpuUnavailable ? "unavailable" : contributionState === "error" ? "error" : "ready";
  const gpuLabel = accelerationGpuLabel(acceleration);
  const gpuPreparing = isGpuPreparing(acceleration.gpu?.state);
  const progress = visibleAccelerationProgress(acceleration);
  const gpuName = accelerationGpuName(acceleration);
  const detail = acceleration.preparation?.issue?.message
    ?? acceleration.message
    ?? "Preparing the local accelerator runtime.";
  return <article className={`accelerator-compact-banner gpu-${acceleration.gpu?.state ?? "checking"}${gpuPreparing ? "" : " progress-hidden"}`} aria-label="Local compute transition">
    <div className={`accelerator-compact-engine cpu ${cpuVisualState}`}>
      <span className="accelerator-compact-icon"><Cpu /></span>
      <span><small>{cpuUnavailable ? "NEEDS ATTENTION" : cpuStandby ? "STANDBY" : "AVAILABLE NOW"}</small><strong>{cpuLabel}</strong><em>{acceleration.cpu?.deviceName || "Local CPU runtime"}</em></span>
    </div>
    <ArrowRight className="accelerator-transition-arrow" aria-hidden="true" />
    <div className="accelerator-compact-gpu">
      <span className="accelerator-compact-icon">{gpuPreparing ? <LoaderCircle className="spin" /> : acceleration.preparation?.issue ? <CircleAlert /> : <Zap />}</span>
      <span className="accelerator-compact-copy" aria-live="polite" aria-atomic="true"><small>{gpuLabel}</small><strong>{gpuName}</strong><em title={detail}>{detail}</em></span>
      {progress !== null && <b>{progress}%</b>}
    </div>
    {gpuPreparing && <div className={`accelerator-compact-progress ${progress === null ? "indeterminate" : ""}`} role="progressbar" aria-label="GPU runtime preparation" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress ?? undefined}>
      <i style={progress === null ? undefined : { width: `${progress}%` }} />
    </div>}
    <button type="button" onClick={onOpen} aria-label="Open GPU setup details">View setup <ChevronRight /></button>
  </article>;
}

function ActivationProgressLog({ model }: { model: RequestedModelCapacity }) {
  const events = useMemo(() => activationProgressEventsForModel(model), [model]);
  const listRef = useRef<HTMLOListElement>(null);
  const currentIndex = Math.max(0, events.length - 1);
  const current = events[currentIndex] ?? events.at(-1)!;
  const nodeCount = new Set(events.flatMap((event) => event.nodeId ? [event.nodeId] : [])).size;
  const stageCount = activationStageCount(events);
  useEffect(() => {
    const list = listRef.current;
    if (!list) return;
    list.scrollTo({ top: list.scrollHeight, behavior: "smooth" });
  }, [events.length, current.at, current.message]);
  return <div className="model-activation-log" aria-live="polite">
    <div className="model-activation-log-head">
      <span><Activity size={14} /> ACTIVATION TIMELINE</span>
      <small>{events.length} events{stageCount > 0 ? ` · ${stageCount} stages` : ""}{nodeCount > 0 ? ` · ${nodeCount} nodes` : ""}</small>
    </div>
    <div className={`model-activation-current ${current.state}`}>
      <span>{current.state === "running" ? "NOW" : current.state === "failed" ? "ATTENTION" : "LATEST"}</span>
      <div><strong>{activationPhaseLabel(current.phase)}</strong><small>{current.message}</small></div>
      <b>{Math.min(currentIndex + 1, events.length)} / {events.length}</b>
    </div>
    <ol ref={listRef}>{events.map((event, index) => {
      const visibleDetails = (event.details ?? []).filter(activationDetailIsReadable);
      const technicalDetails = (event.details ?? []).filter((detail) => !activationDetailIsReadable(detail));
      return <li className={`${event.state}${index === currentIndex ? " current" : ""}`} key={`${event.phase}-${event.at}-${index}`}>
        <i>{event.state === "running" ? <LoaderCircle className="spin" /> : event.state === "failed" ? <CircleAlert /> : <CheckCircle2 />}</i>
        <span>
          <strong><b>{String(index + 1).padStart(2, "0")}</b>{activationPhaseLabel(event.phase)}</strong>
          <small>{event.message}</small>
          {(event.nodeId || event.device || event.processId) && <em>
            {event.nodeId && <>NODE · {shortId(event.nodeId)}</>}
            {event.nodeId && event.device && <> &nbsp;·&nbsp; </>}
            {event.device && <>DEVICE · {event.device.toUpperCase()}</>}
            {(event.nodeId || event.device) && event.processId && <> &nbsp;·&nbsp; </>}
            {event.processId && <>PROCESS · {shortId(event.processId)}</>}
          </em>}
          {visibleDetails.length > 0 && <ul className="model-activation-facts">
            {visibleDetails.map((detail, detailIndex) => <li key={`${detail}-${detailIndex}`}>{detail}</li>)}
          </ul>}
        </span>
        <time dateTime={event.at}>{formatActivationTime(event.at)}</time>
        {technicalDetails.length > 0 && <details className="model-activation-details"><summary>View raw runtime details</summary><div>{technicalDetails.map((detail, detailIndex) => <code key={`${detail}-${detailIndex}`}>{detail}</code>)}</div></details>}
      </li>;
    })}</ol>
  </div>;
}

export function orderActivationProgressEvents<T extends { at: string }>(events: readonly T[]): T[] {
  return events
    .map((event, index) => ({ event, index, timestamp: Date.parse(event.at) }))
    .sort((left, right) => {
      const leftAt = Number.isFinite(left.timestamp) ? left.timestamp : Number.MAX_SAFE_INTEGER;
      const rightAt = Number.isFinite(right.timestamp) ? right.timestamp : Number.MAX_SAFE_INTEGER;
      return leftAt - rightAt || left.index - right.index;
    })
    .map(({ event }) => event);
}

type ActivationProgressEvent = RequestedModelCapacity["activationProgress"][number];

export function normalizeActivationProgressEvents(
  events: readonly ActivationProgressEvent[],
): ActivationProgressEvent[] {
  const ordered = orderActivationProgressEvents(events);
  const lastIndex = ordered.length - 1;
  return ordered.map((event, index) =>
    index < lastIndex && event.state === "running"
      ? { ...event, state: "completed" }
      : event
  );
}

export function activationProgressEventsForModel(
  model: Pick<
    RequestedModelCapacity,
    "activationProgress" | "activationRequestedAt" | "message" | "status" | "updatedAt"
  >,
): ActivationProgressEvent[] {
  const ordered = normalizeActivationProgressEvents(model.activationProgress ?? []);
  const latest = ordered.at(-1);
  const fallbackAt = model.activationRequestedAt ?? model.updatedAt;
  const failureAt = latest && Date.parse(fallbackAt) < Date.parse(latest.at)
    ? latest.at
    : fallbackAt;

  if (model.status === "failed" && latest?.state !== "failed") {
    return normalizeActivationProgressEvents([
      ...ordered,
      {
        phase: "failed",
        message: friendlyActivationFailure(model.message),
        at: failureAt,
        state: "failed",
        details: fallbackActivationDetails(model.message),
      },
    ]);
  }

  if (ordered.length > 0) return ordered;
  return [{
    phase: "queued",
    message: "Waiting for the coordinator to begin activation.",
    at: fallbackAt,
    state: "running",
  }];
}

function activationStageCount(events: RequestedModelCapacity["activationProgress"]): number {
  let count = 0;
  for (const event of events) {
    for (const value of [event.message, ...(event.details ?? [])]) {
      const match = /(?:stage|stages)(?:\s+ready)?(?:\s*:)?\s*(?:\d+\s*(?:of|\/)\s*)?(\d+)/i.exec(value);
      if (/stages?\s+across/i.test(value)) {
        const planned = /(\d+)\s+stages?/i.exec(value)?.[1];
        if (planned) count = Math.max(count, Number(planned));
      } else if (match?.[1]) {
        count = Math.max(count, Number(match[1]));
      }
    }
  }
  return count;
}

function activationDetailIsReadable(detail: string): boolean {
  return detail.length <= 180
    && !detail.includes("launch_process_exited:")
    && !detail.includes("managed_launch_agent_")
    && !detail.includes("distributed_activation_")
    && !detail.includes("distributed_worker_disconnected:");
}

function friendlyActivationFailure(message: string): string {
  const exhausted = /automatic_activation_retries_exhausted:(\d+):/i.exec(message);
  if (exhausted && message.includes("distributed_activation_requires_two_connected_shard_executors")) {
    return `Automatic activation paused after ${exhausted[1]} attempts because the two-PC execution route was still not verified. Keep both PCs online, then run activation again.`;
  }
  if (message.includes("distributed_activation_requires_two_connected_shard_executors")) {
    return "Two verified shard executors are not ready yet. The model remains unpublished while Mycellios rebuilds the route.";
  }
  if (exhausted) {
    return "Automatic activation stopped after repeated temporary node disconnections. The model was not published.";
  }
  if (message.includes("3221225477")) {
    return "A Windows accelerator process crashed while loading the model stage. Capacity was available, but the runtime could not complete the load.";
  }
  return message.replace(/^Automatic activation failed:\s*/i, "") || "The model runtime stopped before activation completed.";
}

function fallbackActivationDetails(message: string): string[] {
  const exhaustedRetries = /automatic_activation_retries_exhausted:(\d+):/i.exec(message)?.[1];
  if (message.includes("distributed_activation_requires_two_connected_shard_executors")) {
    return exhaustedRetries
      ? [
          `Automatic retries attempted: ${exhaustedRetries}`,
          "Connected capacity is visible, but fewer than two nodes completed the required runtime and reciprocal-link evidence.",
          "Keep both PCs online, then run activation again.",
        ]
      : [
          "Connected capacity is visible, but fewer than two nodes have completed the required runtime and reciprocal-link evidence.",
          "Mycellios will retry after the verified executor topology changes.",
        ];
  }
  const details: string[] = [];
  const stage = /(?:stage-|process=)([a-z0-9-]+)/i.exec(message)?.[1];
  const code = /code=(\d+)/i.exec(message)?.[1];
  if (exhaustedRetries) details.push(`Automatic retries attempted: ${exhaustedRetries}`);
  if (stage) details.push(`Failed stage: ${stage}`);
  if (code === "3221225477") details.push("Windows exit code: 3221225477 · 0xC0000005 · native memory access violation");
  else if (code) details.push(`Process exit code: ${code}`);
  details.push("The capacity calculation passed; this failure happened during runtime startup.");
  return details;
}

function activationPhaseLabel(phase: string): string {
  return ({
    queued: "Request accepted",
    profiling: "Profiling model",
    profile_ready: "Profile complete",
    plan_ready: "Distribution calculated",
    preparing_nodes: "Preparing nodes",
    launching_stages: "Launching model stages",
    stage_loading: "Loading model stage",
    stage_ready: "Model stage ready",
    stages_ready: "Stages online",
    checking_health: "Checking pipeline",
    running_canary: "Testing real inference",
    publishing_model: "Publishing model",
    retry_wait: "Waiting to retry",
    retrying: "Retrying activation",
    active: "Activation complete",
    failed: "Activation failed",
  } as Record<string, string>)[phase] ?? phase.replaceAll("_", " ");
}

export function formatActivationTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  const day = date.toLocaleDateString([], {
    month: "short",
    day: "numeric",
    ...(date.getFullYear() === new Date().getFullYear() ? {} : { year: "numeric" }),
  });
  const time = date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  return `${day}\n${time}`;
}

function newInferenceSessionId(): string {
  const id = globalThis.crypto?.randomUUID?.();
  return id ? `chat-${id}` : `chat-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function AcceleratorProgressPanel({ acceleration, contributionState, computeMode }: { acceleration: AcceleratorProgressSnapshot; contributionState: DashboardSnapshot["contribution"]["state"]; computeMode: DesktopSettings["computeMode"] }) {
  const cpuActive = (acceleration.cpu?.activeStages ?? 0) > 0;
  const cpuUnavailable = acceleration.cpu?.state === "unavailable";
  const cpuStandby = computeMode === "gpu-only" && !cpuActive;
  const connectionError = contributionState === "error";
  const gpuPreparing = isGpuPreparing(acceleration.gpu?.state);
  const gpuActive = acceleration.gpu?.state === "ready" && (acceleration.gpu?.activeStages ?? 0) > 0;
  const progress = visibleAccelerationProgress(acceleration);
  const issue = acceleration.preparation?.issue ?? null;
  const log = (acceleration.preparation?.log ?? []).slice(-50);
  const gpuName = accelerationGpuName(acceleration);
  const backend = acceleration.gpu?.state === "not-detected" || acceleration.gpu?.state === "unsupported"
    ? "NOT AVAILABLE"
    : acceleration.gpu?.backend?.toUpperCase() ?? acceleration.requestedBackend?.toUpperCase() ?? "NOT SELECTED";
  const phase = acceleration.preparation?.phase || acceleration.gpu?.state || acceleration.state;
  const phaseLabel = humanizeAccelerationPhase(phase);
  const phaseMessage = issue?.message ?? acceleration.message;
  const panelTitle = accelerationPanelTitle(acceleration, contributionState, computeMode);
  const panelCopy = accelerationPanelCopy(acceleration, contributionState, computeMode);
  const showProgress = gpuPreparing || progress !== null;
  return <article className={`accelerator-progress-panel gpu-${acceleration.gpu?.state ?? "not-detected"}`}>
    <header className="accelerator-progress-header">
      <div className="accelerator-progress-heading">
        <span className="accelerator-runtime-icon">{gpuPreparing ? <LoaderCircle className="spin" /> : gpuActive ? <Zap /> : issue ? <CircleAlert /> : <Cpu />}</span>
        <div><span>LOCAL COMPUTE TRANSITION</span><h2>{panelTitle}</h2><p>{panelCopy}</p></div>
      </div>
      <span className={`accelerator-overall-badge ${gpuActive ? "gpu-active" : cpuActive ? "cpu-active" : cpuUnavailable ? "unavailable" : connectionError ? "connection-error" : "cpu-ready"}`}><i />{gpuActive ? "GPU ACTIVE" : cpuActive ? "CPU ACTIVE" : cpuUnavailable ? "CPU UNAVAILABLE" : cpuStandby ? "CPU STANDBY · GPU ONLY" : contributionState === "connected" ? "CPU READY · NODE ONLINE" : contributionState === "connecting" ? "CPU READY · CONNECTING" : connectionError ? "CPU READY · CONNECTION ERROR" : "CPU READY · CONTRIBUTION PAUSED"}</span>
    </header>

    <div className="accelerator-engine-grid">
      <section className={`accelerator-engine-card cpu ${cpuActive ? "active" : cpuUnavailable ? "unavailable" : connectionError ? "connection-error" : "ready"}`} aria-label="CPU runtime status">
        <div className="accelerator-engine-title"><span><Cpu /></span><div><small>CPU RUNTIME</small><strong>{accelerationCpuLabel(acceleration, contributionState, computeMode)}</strong></div><b>{acceleration.cpu?.precision?.toUpperCase() ?? "FP32"}</b></div>
        <h3>{acceleration.cpu?.deviceName || "Local CPU"}</h3>
        <p>{accelerationCpuCopy(acceleration, contributionState, computeMode)}</p>
        <div className="accelerator-engine-foot"><span><i />{cpuActive ? `${acceleration.cpu.activeStages} active stage${acceleration.cpu.activeStages === 1 ? "" : "s"}` : cpuUnavailable ? "Runtime unavailable" : cpuStandby ? "Disabled for model stages by GPU-only mode" : contributionState === "connected" ? "Ready for network work" : contributionState === "connecting" ? "Connecting to the network" : connectionError ? "Network connection needs attention" : "Ready when contribution resumes"}</span></div>
      </section>

      <section className={`accelerator-engine-card gpu ${acceleration.gpu?.state ?? "not-detected"}`} aria-label="GPU runtime status">
        <div className="accelerator-engine-title"><span>{gpuPreparing ? <LoaderCircle className="spin" /> : <Zap />}</span><div><small>GPU ACCELERATOR</small><strong aria-live="polite" aria-atomic="true">{accelerationGpuLabel(acceleration)}</strong></div><b>{backend}</b></div>
        <h3>{gpuName}</h3>
        <p>{acceleration.gpu?.vendor ? `${acceleration.gpu.vendor} · ` : ""}{gpuActive ? `${acceleration.gpu.activeStages} verified stage${acceleration.gpu.activeStages === 1 ? "" : "s"} running now.` : phaseMessage}</p>
        <div className="accelerator-engine-foot"><span><i />{gpuActive ? "Model weights verified on device" : gpuPreparing ? "Automatic setup in progress" : accelerationGpuLabel(acceleration)}</span></div>
      </section>
    </div>

    {showProgress && <section className="accelerator-preparation" aria-label="GPU runtime preparation progress">
      <div className="accelerator-preparation-head">
        <div role="status" aria-live="polite" aria-atomic="true"><span>CURRENT STEP</span><strong>{phaseLabel}</strong><small>{acceleration.preparation?.currentArtifact ? `Preparing ${acceleration.preparation.currentArtifact}` : phaseMessage}</small></div>
        <div className="accelerator-preparation-value"><strong>{progress === null ? "Working" : `${progress}%`}</strong><small>{formatAccelerationEta(acceleration.preparation?.etaSeconds ?? null, phase)}</small></div>
      </div>
      <div className={`accelerator-progress-track ${progress === null ? "indeterminate" : ""}`} role="progressbar" aria-label={`${phaseLabel} progress`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress ?? undefined}>
        <i style={progress === null ? undefined : { width: `${progress}%` }} />
      </div>
      <div className="accelerator-progress-meta">
        <span><small>TRANSFERRED</small><strong>{formatAccelerationTransfer(acceleration.preparation)}</strong></span>
        <span><small>SPEED</small><strong>{formatAccelerationRate(acceleration.preparation?.bytesPerSecond ?? null)}</strong></span>
        <span><small>PACKAGE</small><strong>{formatAccelerationArtifact(acceleration.preparation)}</strong></span>
        <span><small>ESTIMATED TIME</small><strong>{formatAccelerationEta(acceleration.preparation?.etaSeconds ?? null, phase)}</strong></span>
      </div>
    </section>}

    {issue && <section className="accelerator-requirement" role="alert">
      <span><CircleAlert /></span>
      <div><small>{issue.retryable ? "SETUP NEEDS ATTENTION" : "GPU REQUIREMENT"}</small><strong>{issue.message}</strong><p><b>What to do:</b> {issue.action}</p></div>
      {(issue.requiredBytes !== undefined || issue.availableBytes !== undefined) && <div className="accelerator-requirement-space">
        {issue.requiredBytes !== undefined && <span><small>REQUIRED</small><strong>{formatAccelerationBytes(issue.requiredBytes)}</strong></span>}
        {issue.availableBytes !== undefined && <span><small>AVAILABLE</small><strong>{formatAccelerationBytes(issue.availableBytes)}</strong></span>}
      </div>}
      {!issue.retryable && (issue.code === "unsupported-platform" || issue.code === "unsupported-gpu" || issue.code === "installation-required") && <a className="accelerator-browser-fallback" href={`${PUBLIC_COORDINATOR_URL}/mobile/?autostart=1`} target="_blank" rel="noreferrer"><Globe2 />Start WebGPU in your browser <ExternalLink /></a>}
    </section>}

    <details className="accelerator-log">
      <summary><Activity /> GPU setup log <span>{log.length} event{log.length === 1 ? "" : "s"}</span><ChevronDown /></summary>
      <div className="accelerator-log-body" aria-label="GPU setup event log">
        {log.length > 0 ? <ol>{log.map((entry, index) => <li className={entry.level} key={`${entry.at}-${index}`}><i /><time dateTime={entry.at}>{formatAccelerationLogTime(entry.at)}</time><span>{entry.message}</span></li>)}</ol> : <p>Waiting for the first GPU setup event. CPU availability is unaffected.</p>}
      </div>
    </details>

    <footer className="accelerator-evidence-note"><ShieldCheck /><span><strong>No guessed GPU status.</strong> GPU ACTIVE appears only after model weights load on this device and the distributed canary succeeds.</span></footer>
  </article>;
}

function DesktopSettingsView({ snapshot, bridge, onSnapshot, developerMode }: { snapshot: DashboardSnapshot; bridge: DesktopBridge; onSnapshot: (snapshot: DashboardSnapshot) => void; developerMode: boolean }) {
  const [settingsTab, setSettingsTab] = useState<"general" | "node" | "updates" | "developer">("general");
  const [draft, setDraft] = useState(snapshot.settings);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [checkingUpdate, setCheckingUpdate] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!dirty) setDraft(snapshot.settings);
  }, [dirty, snapshot.settings]);
  function update<K extends keyof DesktopSettings>(key: K, value: DesktopSettings[K]) {
    setDirty(true);
    setDraft((current) => ({ ...current, [key]: value }));
  }
  function updateCoordinatorMode(value: DesktopSettings["coordinatorMode"]) {
    setDirty(true);
    setDraft((current) => ({
      ...current,
      coordinatorMode: value,
      ...(value === "remote" && current.coordinatorMode !== "remote"
        ? { remoteCoordinatorUrl: PUBLIC_COORDINATOR_URL }
        : {}),
    }));
  }
  async function save() {
    setSaving(true); setError(null);
    try {
      const next = await bridge.saveSettings(draft);
      onSnapshot(next);
      setDraft(next.settings);
      setDirty(false);
    }
    catch (caught) { setError(errorText(caught)); }
    finally { setSaving(false); }
  }
  async function checkUpdate() {
    setCheckingUpdate(true); setError(null);
    try {
      const nextUpdate = await bridge.checkForUpdates();
      onSnapshot({ ...snapshot, update: nextUpdate });
    } catch (caught) { setError(errorText(caught)); }
    finally { setCheckingUpdate(false); }
  }
  return <section className="content-page settings-page">
    <PageTitle eyebrow={developerMode ? "DEVELOPER PREFERENCES" : "DESKTOP PREFERENCES"} title="Settings" copy={developerMode ? "Coordinator, runtime, contribution, background behavior and updates for this computer." : "The everyday application and update preferences for this computer."} />
    <div className={`settings-workspace show-${settingsTab}`}>
      <nav className="settings-local-nav" aria-label="Settings sections">
        <button className={settingsTab === "general" ? "active" : ""} onClick={() => setSettingsTab("general")}><UserRound />General</button>
        <button className={settingsTab === "node" ? "active" : ""} onClick={() => setSettingsTab("node")}><Cpu />Node</button>
        <button className={settingsTab === "updates" ? "active" : ""} onClick={() => setSettingsTab("updates")}><Download />Updates</button>
        {developerMode && <button className={settingsTab === "developer" ? "active" : ""} onClick={() => setSettingsTab("developer")}><Code2 />Developer</button>}
      </nav>
      <div className="settings-local-content">
    {error && <div className="inline-error"><CircleAlert size={17} />{error}</div>}
    <div className="settings-section" data-settings-group="general"><div><Globe2 size={20} /><div><h3>Coordinator</h3><p>Use the public network or host an isolated local coordinator.</p></div></div><div className="settings-fields"><label>Mode<AppSelect ariaLabel="Coordinator mode" value={draft.coordinatorMode} onChange={(value) => updateCoordinatorMode(value as DesktopSettings["coordinatorMode"])} options={[{ value: "remote", label: "Public mycellios network" }, { value: "local", label: "Local network on this machine" }]} /></label>{draft.coordinatorMode === "remote" && <label>Coordinator URL<input value={draft.remoteCoordinatorUrl} onChange={(event) => update("remoteCoordinatorUrl", event.target.value)} /></label>}<label>Region<input value={draft.region} onChange={(event) => update("region", event.target.value)} placeholder="auto" /></label></div></div>
    <div className="settings-section" data-settings-group="node">
      <div><Gauge size={20} /><div><h3>Contribution</h3><p>This computer always uses the built-in Mycellios runtime. Models appear only after a verified native deployment starts.</p></div></div>
      <div className="settings-fields">
        <label>Compute mode<AppSelect ariaLabel="Compute mode" value={draft.computeMode} onChange={(value) => update("computeMode", value as DesktopSettings["computeMode"])} options={[{ value: "automatic", label: "Automatic · GPU preferred" }, { value: "gpu-only", label: "GPU only · never CPU" }, { value: "cpu-only", label: "CPU only · GPU disabled" }]} /></label>
        <label>Offered memory (GB)<input type="number" min="1" step="1" value={draft.offeredVramMb / 1_024} onChange={(event) => update("offeredVramMb", Math.round(Number(event.target.value) * 1_024))} /></label>
        <div className="settings-native-runtime" role="status" aria-label="Contribution runtime">
          <ShieldCheck size={17} />
          <span><small>RUNTIME</small><strong>Mycellios native</strong><em>No external model service</em></span>
        </div>
      </div>
    </div>
    <div className="settings-section compact-settings" data-settings-group="general"><div><SlidersHorizontal size={20} /><div><h3>Application</h3><p>Startup and background contribution.</p></div></div><div className="toggle-list"><Toggle label="Start with the system" checked={draft.launchAtLogin} onChange={(value) => update("launchAtLogin", value)} /><Toggle label="Keep running in the tray" checked={draft.closeToTray} onChange={(value) => update("closeToTray", value)} /><Toggle label="Contribute resources" checked={draft.contributionEnabled} onChange={(value) => update("contributionEnabled", value)} /></div></div>
    <div className="settings-section" data-settings-group="updates"><div><Download size={20} /><div><h3>Automatic updates</h3><p>Desktop releases download in the background and install automatically as soon as active inference is idle.</p></div></div><div className="update-settings"><div><span className={`update-state ${snapshot.update.state}`}>{updateStateLabel(snapshot.update.state)}</span><strong>Version {snapshot.appVersion}</strong><p>{snapshot.update.message}</p><p title={buildIdentityTitle(snapshot.buildIdentity)}>Exact build: {shortBuildIdentity(snapshot.buildIdentity)}</p></div><div className="update-actions"><button className="secondary-button" disabled={checkingUpdate || snapshot.update.state === "checking" || snapshot.update.state === "downloading"} onClick={() => void checkUpdate()}><RefreshCw className={checkingUpdate ? "spin" : ""} size={15} />Check now</button>{snapshot.update.state === "ready" && <button className="primary-button" onClick={() => void bridge.installUpdate()}><Download size={15} />Restart now</button>}</div></div></div>
    <div className="settings-footer"><button className="primary-button" disabled={saving} onClick={() => void save()}>{saving ? <LoaderCircle className="spin" size={17} /> : <Check size={17} />}Save and reconnect</button></div>
      </div>
    </div>
  </section>;
}

function DesktopOnboarding({ snapshot, bridge, onSnapshot }: { snapshot: DashboardSnapshot; bridge: DesktopBridge; onSnapshot: (snapshot: DashboardSnapshot) => void }) {
  const [contribute, setContribute] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  async function complete() {
    setSaving(true); setError(null);
    try { onSnapshot(await bridge.saveSettings({ ...snapshot.settings, coordinatorMode: "remote", remoteCoordinatorUrl: PUBLIC_COORDINATOR_URL, remoteCoordinatorToken: "", contributionEnabled: contribute, onboardingComplete: true })); }
    catch (caught) { setError(errorText(caught)); }
    finally { setSaving(false); }
  }
  return <div className="modal-backdrop"><div className="onboarding-card"><div className="onboarding-brand"><img src={brandIcon} alt="" /><span>GET STARTED</span></div><h1>Connect this machine to mycellios</h1><p>The desktop agent and the web panel will show the same public network.</p><div className="mode-grid single-mode"><button className="selected"><Globe2 size={23} /><strong>Public mycellios network</strong><span>{PUBLIC_COORDINATOR_URL}</span><Check size={16} className="mode-check" /></button></div><label className="contribute-choice"><input type="checkbox" checked={contribute} onChange={(event) => setContribute(event.target.checked)} /><span><strong>Contribute this machine's resources</strong><small>Register real hardware only; models appear only when a verified runtime is active.</small></span></label>{error && <div className="inline-error"><CircleAlert size={17} />{error}</div>}<button className="primary-button onboarding-submit" disabled={saving} onClick={() => void complete()}>{saving ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={18} />}Enter mycellios <ChevronRight size={18} /></button><div className="onboarding-security"><ShieldCheck size={15} />Connected through HTTPS/WSS to the public coordinator.</div></div></div>;
}

function desktopToPublicSnapshot(snapshot: DashboardSnapshot): PublicSnapshot {
  return {
    capturedAt: snapshot.capturedAt,
    version: snapshot.health?.version ?? "—",
    buildIdentity: snapshot.health?.buildIdentity ?? null,
    summary: {
      registered: snapshot.health?.workers.registered ?? snapshot.workers.length,
      connected: snapshot.health?.workers.connected ?? snapshot.workers.filter((worker) => worker.connected).length,
      online: snapshot.health?.workers.online ?? snapshot.workers.filter((worker) => worker.status === "online").length,
      mobile: snapshot.workers.filter((worker) => worker.kind === "browser").length,
      offeredVramMb: snapshot.workers.reduce((sum, worker) => sum + worker.offeredVramMb, 0),
      completedJobs: snapshot.jobs.filter((job) => job.status === "completed").length,
    },
    workers: snapshot.workers,
    models: snapshot.models,
    requestedModels: snapshot.requestedModels,
    jobs: snapshot.jobs,
  };
}

function updateStateLabel(state: DesktopUpdateStatus["state"]): string {
  switch (state) {
    case "checking": return "Checking";
    case "downloading": return "Downloading";
    case "ready": return "Ready to install";
    case "up-to-date": return "Up to date";
    case "error": return "Check failed";
    case "unsupported": return "Managed externally";
    case "development": return "Development mode";
    default: return "Automatic";
  }
}

function DesktopMetric({ label, value }: { label: string; value: string }) { return <div className="metric"><span>{label}</span><strong>{value}</strong></div>; }

interface AppSelectOption {
  value: string;
  label: string;
}

function AppSelect({ ariaLabel, value, options, onChange, placeholder = "Select an option" }: { ariaLabel: string; value: string; options: AppSelectOption[]; onChange: (value: string) => void; placeholder?: string }) {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const selectedIndex = Math.max(0, options.findIndex((option) => option.value === value));
  const selected = options[selectedIndex];

  function moveSelection(direction: -1 | 1) {
    if (options.length === 0) return;
    const nextIndex = (selectedIndex + direction + options.length) % options.length;
    const next = options[nextIndex];
    if (next) onChange(next.value);
  }

  return <div className={`app-select${open ? " open" : ""}`} onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false); }}>
    <button
      ref={triggerRef}
      type="button"
      className="app-select-trigger"
      aria-label={ariaLabel}
      aria-haspopup="listbox"
      aria-expanded={open}
      disabled={options.length === 0}
      onClick={() => setOpen((current) => !current)}
      onKeyDown={(event) => {
        if (event.key === "Escape") { event.preventDefault(); setOpen(false); }
        if (event.key === "ArrowDown") { event.preventDefault(); setOpen(true); moveSelection(1); }
        if (event.key === "ArrowUp") { event.preventDefault(); setOpen(true); moveSelection(-1); }
      }}
    >
      <span>{selected?.label ?? placeholder}</span><ChevronDown size={16} />
    </button>
    {open && <div className="app-select-menu" role="listbox" aria-label={ariaLabel}>
      {options.map((option) => <button
        key={option.value}
        type="button"
        role="option"
        aria-selected={option.value === value}
        className={option.value === value ? "selected" : ""}
        tabIndex={-1}
        onMouseDown={(event) => event.preventDefault()}
        onClick={() => { onChange(option.value); setOpen(false); triggerRef.current?.focus(); }}
      ><span>{option.label}</span>{option.value === value && <Check size={15} />}</button>)}
    </div>}
  </div>;
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return <button type="button" className="toggle-row" role="switch" aria-checked={checked} onClick={() => onChange(!checked)}><span>{label}</span><i aria-hidden="true" /></button>;
}
function errorText(error: unknown) { return error instanceof Error ? error.message : String(error); }
function networkNameForHubModel(source: string): string {
  return (source.split("/").at(-1) ?? source)
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, "")
    .slice(0, 128);
}
function formatHubCount(value: number): string {
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function formatHubDate(value: string | null): string {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "Unknown";
  return new Intl.DateTimeFormat("en", { month: "short", day: "numeric", year: "numeric" }).format(date);
}

function estimatedHubMemoryMiB(model: HubCatalogModel): number | null {
  if (typeof model.estimatedMemoryMiB === "number" && Number.isFinite(model.estimatedMemoryMiB) && model.estimatedMemoryMiB > 0) {
    return model.estimatedMemoryMiB;
  }
  const name = model.id.split("/").at(-1) ?? model.id;
  const matches = [...name.matchAll(/(\d+(?:\.\d+)?)\s*b(?=$|[-_.])/gi)];
  const billions = Number(matches.at(-1)?.[1]);
  if (!Number.isFinite(billions) || billions <= 0) return null;
  const mib = 1_024 * 1_024;
  const weightBytes = billions * 1_000_000_000 * 2;
  const runtimeAndLoaderReserve = 2 * 256 * mib + Math.ceil(weightBytes * 0.1) + 2 * 64 * mib;
  return Math.ceil((weightBytes + runtimeAndLoaderReserve) / mib);
}

function friendlyModelMutationError(error: unknown): string {
  const message = errorText(error);
  if (/administrator token|admin token|invalid_model_admin_token|HTTP 401/i.test(message)) {
    return "The network administrator token is missing or invalid. Check it and try again.";
  }
  return message.replace(/^Error invoking remote method ['"]?models:[^:]+['"]?:\s*Error:\s*/i, "");
}

function PageTitle({ eyebrow, title, copy, actions }: { eyebrow: string; title: string; copy: string; actions?: React.ReactNode }) {
  return <div className="panel-page-title"><div><span>{eyebrow}</span><h1>{title}</h1><p>{copy}</p></div>{actions && <div className="page-actions">{actions}</div>}</div>;
}

function PanelDisclosure({ icon: Icon, eyebrow, title, summary, children, defaultOpen = false }: { icon: typeof Activity; eyebrow: string; title: string; summary: string; children: React.ReactNode; defaultOpen?: boolean }) {
  return <details className="panel-disclosure" open={defaultOpen}>
    <summary>
      <i className="panel-disclosure-icon"><Icon size={18} /></i>
      <span><small>{eyebrow}</small><strong>{title}</strong><em>{summary}</em></span>
      <ChevronDown className="panel-disclosure-chevron" size={18} aria-hidden="true" />
    </summary>
    <div className="panel-disclosure-body">{children}</div>
  </details>;
}

function Stat({ icon: Icon, label, value, detail, tone = "blue" }: { icon: typeof Radio; label: string; value: string; detail: string; tone?: "blue" | "purple" }) { return <article className={`panel-stat ${tone}`}><div><Icon /></div><span>{label}</span><strong>{value}</strong><small>{detail}</small><i className="stat-accent" /></article>; }
function CapacityMetric({ icon: Icon, label, value, detail }: { icon: typeof Cpu; label: string; value: string; detail: string }) { return <div className="global-capacity-metric"><i><Icon /></i><span><small>{label}</small><strong>{value}</strong><em>{detail}</em></span></div>; }
function Metric({ label, value }: { label: string; value: string }) { return <div><span>{label}</span><strong>{value}</strong></div>; }
function Empty({ icon: Icon, title, copy }: { icon: typeof Activity; title: string; copy: string }) { return <div className="panel-empty"><Icon /><strong>{title}</strong><p>{copy}</p></div>; }
function PanelLoading() { return <div className="panel-loading"><LoaderCircle className="spin" /><strong>Connecting to mycellios</strong><span>Loading the public network snapshot…</span></div>; }
function WorkerKindIcon({ worker }: { worker: PublicWorker }) {
  if (worker.kind === "browser") return <Smartphone />;
  if (worker.kind === "cell") return <Boxes />;
  return <Laptop />;
}
function workerKindLabel(worker: PublicWorker) {
  if (worker.kind === "browser") return "ACTIVE BROWSER SESSION";
  if (worker.kind === "cell") return "COMPUTE CELL";
  return "INSTALLED DEVICE";
}
function workerLabel(worker: PublicWorker) {
  if (worker.kind === "browser") return `Browser ${shortId(worker.id)}`;
  if (worker.kind === "cell") {
    return worker.deployments[0]?.model
      ?? worker.gpus[0]?.model.replace(/^Aggregate capacity exposed by /, "")
      ?? `Cell ${shortId(worker.id)}`;
  }
  return worker.gpus[0]?.model || `Desktop ${shortId(worker.id)}`;
}
function nodeVisualState(worker: PublicWorker): "active" | "warning" | "offline" { return worker.connected && worker.status === "online" ? "active" : worker.connected || worker.status === "suspect" || worker.status === "draining" ? "warning" : "offline"; }
function nodeStateOrder(worker: PublicWorker): number { return nodeVisualState(worker) === "active" ? 0 : nodeVisualState(worker) === "warning" ? 1 : 2; }
function nodeStatusLabel(worker: PublicWorker): string {
  if (worker.connected && worker.status === "online") return "Active";
  if (worker.status === "suspect") return "Unstable";
  if (worker.status === "draining") return "Draining";
  if (worker.connected) return "Connected";
  return "Offline";
}
function topologyPosition(index: number, total: number): { x: number; y: number } {
  if (total <= 1) return { x: 50, y: 17 };
  if (total <= 10) {
    const angle = -Math.PI / 2 + (index / total) * Math.PI * 2;
    return { x: 50 + Math.cos(angle) * 41, y: 52 + Math.sin(angle) * 36 };
  }
  const inner = index % 2 === 0;
  const slot = Math.floor(index / 2);
  const count = inner ? Math.ceil(total / 2) : Math.floor(total / 2);
  const angle = -Math.PI / 2 + (slot / count) * Math.PI * 2;
  return { x: 50 + Math.cos(angle) * (inner ? 28 : 43), y: 52 + Math.sin(angle) * (inner ? 25 : 38) };
}
function formatCompactNumber(value: number): string { return new Intl.NumberFormat("en-US", { maximumFractionDigits: value < 10 ? 1 : 0 }).format(value); }
function formatCompactTokens(value: number): string {
  if (value >= 1_000_000) return `${new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(value / 1_000_000)}M`;
  if (value >= 1_000) return `${new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(value / 1_000)}K`;
  return new Intl.NumberFormat("en-US").format(value);
}
function formatPower(watts: number): string { return watts >= 1_000 ? `${(watts / 1_000).toFixed(1)} kW` : `${Math.round(watts)} W`; }
function shortId(value: string) { return value.length <= 10 ? value : `${value.slice(0, 6)}…${value.slice(-4)}`; }
function shortFingerprint(value: string) {
  const digest = value.replace(/^sha256:/, "");
  return digest.length <= 20 ? digest : `${digest.slice(0, 10)}…${digest.slice(-8)}`;
}
function credentialKindLabel(kind: WorkerCredentialSummary["identityKind"]) {
  return kind === "browser"
    ? "NAVEGADOR PWA"
    : kind === "cell"
      ? "CELDA DISTRIBUIDA"
      : "DISPOSITIVO INSTALADO";
}
function formatMemory(value: number) { return value >= 1_024 ? `${(value / 1_024).toFixed(value >= 10_240 ? 0 : 1)} GB` : `${Math.round(value)} MB`; }
function shouldShowAccelerationBanner(acceleration: AcceleratorProgressSnapshot): boolean {
  return acceleration.state !== "idle"
    || acceleration.cpu?.state !== "unavailable"
    || acceleration.preparation?.phase !== "idle";
}

function isGpuPreparing(state: AcceleratorProgressSnapshot["gpu"]["state"] | undefined): boolean {
  return state === "checking" || state === "downloading" || state === "installing" || state === "verifying";
}

function accelerationProgress(acceleration: AcceleratorProgressSnapshot): number | null {
  const value = acceleration.preparation?.progressPct;
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(0, Math.min(100, Math.round(value)))
    : null;
}

function visibleAccelerationProgress(acceleration: AcceleratorProgressSnapshot): number | null {
  const progress = accelerationProgress(acceleration);
  const state = acceleration.gpu?.state;
  if (state === "fallback" || state === "error" || state === "unsupported" || state === "not-detected") return null;
  return progress;
}

function accelerationCpuLabel(acceleration: AcceleratorProgressSnapshot, contributionState?: DashboardSnapshot["contribution"]["state"], computeMode?: DesktopSettings["computeMode"]): string {
  if ((acceleration.cpu?.activeStages ?? 0) > 0) return "CPU ACTIVE";
  if (acceleration.cpu?.state === "unavailable") return "CPU UNAVAILABLE";
  if (computeMode === "gpu-only") return "CPU STANDBY · GPU ONLY";
  if (contributionState === "connected") return "CPU READY · NODE ONLINE";
  if (contributionState === "connecting") return "CPU READY · CONNECTING";
  if (contributionState === "error") return "CPU READY · CONNECTION ERROR";
  return "CPU READY · CONTRIBUTION PAUSED";
}

function accelerationCpuCopy(acceleration: AcceleratorProgressSnapshot, contributionState: DashboardSnapshot["contribution"]["state"], computeMode?: DesktopSettings["computeMode"]): string {
  if (acceleration.cpu?.state === "unavailable") return "The certified CPU runtime could not start. Open the runtime log, then restart or update mycellios.";
  if (computeMode === "gpu-only" && (acceleration.cpu?.activeStages ?? 0) === 0) return "The CPU runtime is installed but excluded from model execution because GPU-only mode is selected.";
  if (contributionState === "error") return "The CPU runtime is ready, but this worker is not connected to the network. mycellios will keep retrying the connection.";
  if (contributionState === "paused") return "The CPU runtime is installed and ready; contribution is currently paused.";
  return acceleration.cpu?.message || "The CPU runtime is available while GPU setup continues.";
}

function accelerationGpuLabel(acceleration: AcceleratorProgressSnapshot): string {
  const state = acceleration.gpu?.state;
  if (state === "ready" && (acceleration.gpu?.activeStages ?? 0) > 0) return "GPU ACTIVE";
  if (state === "ready") return "GPU READY";
  if (isGpuPreparing(state)) return "GPU PREPARING";
  if (state === "fallback") return "CPU FALLBACK";
  if (state === "unsupported") return "GPU NOT SUPPORTED YET";
  if (state === "error") return "GPU SETUP FAILED";
  if (state === "not-detected") return "NO GPU DETECTED";
  return "GPU CHECKING";
}

function accelerationGpuName(acceleration: AcceleratorProgressSnapshot): string {
  if (acceleration.gpu?.state === "not-detected") return "No physical GPU detected";
  if (acceleration.gpu?.state === "unsupported") return acceleration.gpu.model ?? "Unsupported GPU";
  return acceleration.gpu?.model ?? acceleration.deviceName ?? "GPU accelerator";
}

function accelerationPanelTitle(acceleration: AcceleratorProgressSnapshot, contributionState: DashboardSnapshot["contribution"]["state"], computeMode?: DesktopSettings["computeMode"]): string {
  const gpuState = acceleration.gpu?.state;
  if (acceleration.cpu?.state === "unavailable") return "The local compute runtime needs attention.";
  if (computeMode === "gpu-only" && (gpuState === "fallback" || gpuState === "error")) return "GPU-only mode is waiting for a verified accelerator.";
  if (contributionState === "error") return "CPU ready. Network connection needs attention.";
  if (contributionState === "paused" && isGpuPreparing(gpuState)) return "Contribution paused. GPU setup continues safely.";
  if (contributionState === "paused") return "CPU ready. Contribution is paused.";
  if (gpuState === "ready" && (acceleration.gpu?.activeStages ?? 0) > 0) return "GPU acceleration is active.";
  if (gpuState === "ready") return "GPU acceleration is ready for the next stage.";
  if (isGpuPreparing(gpuState)) return "CPU online. GPU acceleration is preparing automatically.";
  if (gpuState === "fallback" || gpuState === "error") return "CPU remains online. GPU setup needs attention.";
  if (gpuState === "unsupported") return "CPU online. This GPU is not supported yet.";
  return "CPU online. No compatible GPU was detected.";
}

function accelerationPanelCopy(acceleration: AcceleratorProgressSnapshot, contributionState: DashboardSnapshot["contribution"]["state"], computeMode?: DesktopSettings["computeMode"]): string {
  const gpuState = acceleration.gpu?.state;
  if (acceleration.cpu?.state === "unavailable") return "The certified CPU runtime is not available. Review the runtime log before contributing from this device.";
  if (computeMode === "gpu-only" && gpuState !== "ready") return "This node stays connected, but it will not execute CPU model stages while GPU-only mode is selected.";
  if (contributionState === "error") return "The CPU runtime remains ready, but the worker could not connect. mycellios will keep retrying the network connection.";
  if (contributionState === "paused" && isGpuPreparing(gpuState)) return "The CPU runtime stays ready and the current GPU setup can finish, but this device will not accept network work until contribution resumes.";
  if (contributionState === "paused") return "Installed runtimes remain ready, but this device will not accept network work until contribution resumes.";
  if (gpuState === "ready") return "The physical accelerator probe passed; model execution remains visible separately from hardware readiness.";
  if (isGpuPreparing(gpuState)) return "The CPU runtime stays available while mycellios installs and verifies the best runtime for this machine.";
  if (gpuState === "fallback" || gpuState === "error") return "CPU service is preserved; follow the requirement below to make GPU acceleration available.";
  if (gpuState === "unsupported") return "This release will continue safely on CPU until a certified accelerator pack is available.";
  return "The certified CPU runtime remains available for network work.";
}

function humanizeAccelerationPhase(value: string): string {
  const normalized = value.trim().replace(/[_-]+/g, " ");
  if (!normalized) return "Checking accelerator";
  return normalized.replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatAccelerationBytes(value: number): string {
  if (!Number.isFinite(value) || value < 0) return "—";
  if (value < 1_024) return `${Math.round(value)} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let amount = value;
  let unit = -1;
  do { amount /= 1_024; unit += 1; } while (amount >= 1_024 && unit < units.length - 1);
  return `${amount.toFixed(amount >= 100 ? 0 : amount >= 10 ? 1 : 2)} ${units[unit]}`;
}

function formatAccelerationTransfer(preparation: AcceleratorProgressSnapshot["preparation"]): string {
  const completed = preparation?.bytesCompleted;
  const total = preparation?.bytesTotal;
  if (completed === null || completed === undefined) return total === null || total === undefined ? "Calculating…" : `0 B / ${formatAccelerationBytes(total)}`;
  return total === null || total === undefined
    ? formatAccelerationBytes(completed)
    : `${formatAccelerationBytes(completed)} / ${formatAccelerationBytes(total)}`;
}

function formatAccelerationRate(value: number | null): string {
  return value !== null && Number.isFinite(value) && value > 0
    ? `${formatAccelerationBytes(value)}/s`
    : "Calculating…";
}

function formatAccelerationEta(value: number | null, phase = "downloading"): string {
  if (value === null || !Number.isFinite(value) || value < 0) {
    if (phase === "installing" || phase === "copying-base") return "Finishing installation…";
    if (phase === "physical-probe" || phase === "activating") return "Running verification…";
    if (phase === "verifying-package") return "Checking package…";
    return "Estimating…";
  }
  const seconds = Math.ceil(value);
  if (seconds < 60) return `${seconds}s remaining`;
  const minutes = Math.ceil(seconds / 60);
  if (minutes < 60) return `~${minutes}m remaining`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return `~${hours}h${rest > 0 ? ` ${rest}m` : ""} remaining`;
}

function formatAccelerationArtifact(preparation: AcceleratorProgressSnapshot["preparation"]): string {
  const index = preparation?.artifactIndex;
  const count = preparation?.artifactCount;
  if (index !== null && index !== undefined && count !== null && count !== undefined) return `${Math.max(0, index)} of ${Math.max(0, count)}`;
  return preparation?.currentArtifact ?? "Preparing…";
}

function formatAccelerationLogTime(value: string): string {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "—";
  return date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
function relativeTime(value: string) { const seconds = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 1_000)); if (seconds < 5) return "now"; if (seconds < 60) return `${seconds}s ago`; const minutes = Math.floor(seconds / 60); if (minutes < 60) return `${minutes}m ago`; return `${Math.floor(minutes / 60)}h ago`; }
function relativeTimeEs(value: string) { const seconds = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 1_000)); if (seconds < 5) return "now"; if (seconds < 60) return `${seconds} s ago`; const minutes = Math.floor(seconds / 60); if (minutes < 60) return `${minutes} min ago`; return `${Math.floor(minutes / 60)} h ago`; }

type ExecutionTelemetry = NonNullable<DashboardDeployment["execution"]>;
type ExecutionStageTelemetry = NonNullable<ExecutionTelemetry["stages"]>[number];

interface EffectiveExecutionSummary {
  verified: boolean;
  deviceType: "cpu" | "gpu" | "mixed" | "unknown";
  backends: ExecutionTelemetry["backend"][];
  deviceNames: string[];
  precisions: string[];
  fallback: boolean;
  fallbackReasons: string[];
  stages: ExecutionStageTelemetry[];
  unitCount: number;
}

function unknownExecutionSummary(): EffectiveExecutionSummary {
  return { verified: false, deviceType: "unknown", backends: [], deviceNames: [], precisions: [], fallback: false, fallbackReasons: [], stages: [], unitCount: 0 };
}

function summarizeDeploymentExecution(deployments: PublicDeployment[]): EffectiveExecutionSummary {
  const executions = deployments.map((deployment) => deployment.execution).filter((execution): execution is ExecutionTelemetry => execution !== undefined);
  if (executions.length === 0) return unknownExecutionSummary();
  const stages = executions.flatMap((execution) => execution.stages ?? []);
  if (stages.length > 0) return summarizeExecutionStages(stages, executions);
  const deviceTypes = new Set(executions.map((execution) => execution.deviceType));
  const hasCpu = deviceTypes.has("cpu") || deviceTypes.has("mixed");
  const hasGpu = deviceTypes.has("gpu") || deviceTypes.has("mixed");
  return {
    verified: true,
    deviceType: hasCpu && hasGpu ? "mixed" : hasGpu ? "gpu" : "cpu",
    backends: unique(executions.map((execution) => execution.backend)),
    deviceNames: unique(executions.map((execution) => execution.deviceName)),
    precisions: unique(executions.map((execution) => execution.precision)),
    fallback: executions.some((execution) => execution.fallback),
    fallbackReasons: unique(executions.flatMap((execution) => execution.fallbackReason ? [execution.fallbackReason] : [])),
    stages: [],
    unitCount: executions.length,
  };
}

function isMeasuredDeployment(
  deployment: PublicDeployment,
): deployment is PublicDeployment & { throughputSource: "measured" } {
  return deployment.throughputSource === "measured"
    && Number.isFinite(deployment.tokensPerSecond)
    && deployment.tokensPerSecond > 0;
}

function summarizeExecutionStages(stages: ExecutionStageTelemetry[], parents: ExecutionTelemetry[] = []): EffectiveExecutionSummary {
  const hasCpu = stages.some((stage) => stage.deviceType === "cpu");
  const hasGpu = stages.some((stage) => stage.deviceType === "gpu");
  return {
    verified: true,
    deviceType: hasCpu && hasGpu ? "mixed" : hasGpu ? "gpu" : "cpu",
    backends: unique(stages.map((stage) => stage.backend)),
    deviceNames: unique(stages.map((stage) => stage.deviceName)),
    precisions: unique(stages.map((stage) => stage.precision)),
    fallback: stages.some((stage) => stage.fallback) || parents.some((execution) => execution.fallback),
    fallbackReasons: unique([
      ...stages.flatMap((stage) => stage.fallbackReason ? [stage.fallbackReason] : []),
      ...parents.flatMap((execution) => execution.fallbackReason ? [execution.fallbackReason] : []),
    ]),
    stages: [...stages].sort((left, right) => left.stageIndex - right.stageIndex),
    unitCount: stages.length,
  };
}

function modelExecutionSummary(snapshot: PublicSnapshot, modelId: string): EffectiveExecutionSummary {
  return summarizeDeploymentExecution(snapshot.workers.flatMap((worker) => worker.deployments).filter((deployment) => deployment.model === modelId));
}

function workerExecutionSummary(snapshot: PublicSnapshot, worker: PublicWorker): EffectiveExecutionSummary {
  const runtimeNodeId = worker.executionNodeId ?? worker.id;
  const matchingStages = snapshot.workers
    .flatMap((candidate) => candidate.deployments)
    .flatMap((deployment) => deployment.execution?.stages ?? [])
    .filter((stage) => stage.nodeId === runtimeNodeId);
  if (matchingStages.length > 0) return summarizeExecutionStages(matchingStages);
  return summarizeDeploymentExecution(worker.deployments);
}

function unique<T>(values: T[]): T[] { return [...new Set(values)]; }

function executionShortLabel(execution: EffectiveExecutionSummary): string {
  if (!execution.verified) return "UNVERIFIED";
  if (execution.deviceType === "cpu") return execution.fallback ? "CPU FALLBACK" : "CPU ACTIVE";
  if (execution.deviceType === "gpu") return "GPU ACTIVE";
  return execution.fallback ? "MIXED · FALLBACK" : "HYBRID · AUTHORIZED";
}

function executionDetail(execution: EffectiveExecutionSummary): string {
  if (!execution.verified) return "No effective runtime telemetry";
  const units = `${execution.unitCount} ${execution.stages.length > 0 ? "stage" : "runtime"}${execution.unitCount === 1 ? "" : "s"}`;
  const backends = execution.backends.map((backend) => backend.toUpperCase()).join(" + ");
  const precision = execution.precisions.join(" + ");
  return [units, backends, precision].filter(Boolean).join(" · ");
}

function executionDeviceNames(execution: EffectiveExecutionSummary): string {
  return execution.deviceNames.length > 0 ? execution.deviceNames.join(" · ") : "Device name unavailable";
}

function ExecutionBadge({ execution, workers = [], large = false, interactive = true }: {
  execution: EffectiveExecutionSummary;
  workers?: PublicWorker[];
  large?: boolean;
  interactive?: boolean;
}) {
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  const dialogTitleId = useId();
  const Icon = execution.deviceType === "gpu" ? Zap : execution.deviceType === "mixed" ? Network : Cpu;
  const title = execution.fallbackReasons.length > 0 ? execution.fallbackReasons.join(" · ") : executionDetail(execution);
  const diagnosticsAvailable = interactive && execution.verified && (execution.stages.length > 0 || execution.fallbackReasons.length > 0);
  const autoRepair = executionAutoRepairActive(execution, workers);
  const className = `execution-badge ${execution.deviceType}${execution.fallback ? " fallback" : ""}${large ? " large" : ""}`;
  const contents = <><Icon /><span><strong>{executionShortLabel(execution)}</strong>{large && <small>{execution.verified ? execution.backends.map((backend) => backend.toUpperCase()).join(" + ") : "NO DEVICE CLAIM"}</small>}</span>{diagnosticsAvailable && <ChevronRight className="execution-badge-arrow" />}</>;

  useEffect(() => {
    if (!diagnosticsOpen) return;
    const previousOverflow = document.body.style.overflow;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setDiagnosticsOpen(false);
    };
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [diagnosticsOpen]);

  if (!diagnosticsAvailable) return <span className={className} title={title}>{contents}</span>;
  return <>
    <button type="button" className={className} title={title} aria-haspopup="dialog" aria-expanded={diagnosticsOpen} onClick={() => setDiagnosticsOpen(true)}>{contents}</button>
    {diagnosticsOpen && createPortal(
      <div className="execution-diagnostic-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setDiagnosticsOpen(false); }}>
        <section className="execution-diagnostic-dialog" role="dialog" aria-modal="true" aria-labelledby={dialogTitleId}>
          <header className="execution-diagnostic-head">
            <div><span>REAL TELEMETRY</span><h2 id={dialogTitleId}>Execution diagnostics</h2><p>Device and backend confirmed for each process after loading the model.</p></div>
            <button type="button" aria-label="Close diagnostics" onClick={() => setDiagnosticsOpen(false)}><X /></button>
          </header>
          <div className="execution-diagnostic-summary">
            <div><small>ESTADO</small><strong>{executionShortLabel(execution)}</strong></div>
            <div><small>BACKENDS</small><strong>{execution.backends.map((backend) => backend.toUpperCase()).join(" + ")}</strong></div>
            <div><small>ETAPAS</small><strong>{execution.unitCount}</strong></div>
            <div><small>FALLBACK</small><strong>{execution.fallback ? "Yes" : "No"}</strong></div>
          </div>
          {execution.fallbackReasons.length > 0 && <div className="execution-diagnostic-alert"><CircleAlert /><span><small>MOTIVO DEL FALLBACK</small><strong>{execution.fallbackReasons.join(" · ")}</strong></span></div>}
          {autoRepair && <div className="execution-diagnostic-repair"><RefreshCw /><span><small>AUTO-REPAIR ACTIVE</small><strong>CPU keeps the service running while mycellios revalidates the GPU. When enough verified GPU capacity is available, the coordinator rebuilds the route and runs a real canary before publishing it.</strong></span></div>}
          <div className="execution-diagnostic-stages">
            <div className="execution-diagnostic-section-title"><span>MODEL ROUTE</span><small>{execution.stages.length > 0 ? `${execution.stages.length} verified stages` : "Verified runtime"}</small></div>
            {execution.stages.length > 0 ? execution.stages.map((stage) => {
              const host = executionWorkerForNode(workers, stage.nodeId);
              const stageSummary = summarizeExecutionStages([stage]);
              return <article className={`execution-diagnostic-stage${stage.fallback ? " fallback" : ""}`} key={`${stage.nodeId}-${stage.stageIndex}`}>
                <div className="execution-diagnostic-stage-head"><span><b>{String(stage.stageIndex + 1).padStart(2, "0")}</b><span><small>ETAPA {stage.stageIndex + 1}</small><strong>Capas {stage.layerStart}–{stage.layerEnd}</strong></span></span><ExecutionBadge execution={stageSummary} interactive={false} /></div>
                <div className="execution-diagnostic-stage-grid">
                  <div><small>PHYSICAL DEVICE</small><strong>{host ? workerLabel(host) : "Device not identified"}</strong></div>
                  <div><small>DISPOSITIVO USADO</small><strong>{stage.deviceName}</strong></div>
                  <div><small>BACKEND</small><strong>{stage.backend.toUpperCase()} · {stage.precision}</strong></div>
                  <div><small>NODO</small><code title={stage.nodeId}>{shortId(stage.nodeId)}</code></div>
                  {host?.computeMode && <div><small>MODO</small><strong>{computeModeLabel(host.computeMode)}</strong></div>}
                </div>
                {stage.fallbackReason && <p><CircleAlert />{stage.fallbackReason}</p>}
              </article>;
            }) : <article className="execution-diagnostic-runtime"><strong>{executionDeviceNames(execution)}</strong><span>{executionDetail(execution)}</span></article>}
          </div>
          <footer className="execution-diagnostic-foot"><HardDrive /><span><strong>Complete log on the affected machine</strong><small>Windows: <code>%APPDATA%\mycellios\mycellios.log</code>. On macOS and Linux it is stored as <code>mycellios.log</code> inside the application data folder.</small></span></footer>
        </section>
      </div>,
      document.body,
    )}
  </>;
}

function ExecutionStages({ execution, workers = [], compact = false }: { execution: EffectiveExecutionSummary; workers?: PublicWorker[]; compact?: boolean }) {
  if (execution.stages.length === 0) return null;
  return <div className={`execution-stage-list${compact ? " compact" : ""}`}>
    {execution.stages.map((stage) => {
      const summary = summarizeExecutionStages([stage]);
      const host = executionWorkerForNode(workers, stage.nodeId);
      return <div className="execution-stage" key={`${stage.nodeId}-${stage.stageIndex}`}><span><small>STAGE {stage.stageIndex + 1} · LAYERS {stage.layerStart}–{stage.layerEnd}</small><strong>{stage.deviceName}</strong><em>{host ? `${workerLabel(host)} · ` : ""}{shortId(stage.nodeId)} · {stage.backend.toUpperCase()} · {stage.precision}</em></span><ExecutionBadge execution={summary} interactive={false} /></div>;
    })}
  </div>;
}

function executionWorkerForNode(workers: readonly PublicWorker[], nodeId: string): PublicWorker | undefined {
  return workers.find((worker) => (worker.executionNodeId ?? worker.id) === nodeId);
}

function executionAutoRepairActive(
  execution: EffectiveExecutionSummary,
  workers: readonly PublicWorker[],
): boolean {
  return execution.fallback && execution.stages.some((stage) => (
    executionWorkerForNode(workers, stage.nodeId)?.computeMode === "automatic"
  ));
}

function computeModeLabel(mode: NonNullable<PublicWorker["computeMode"]>): string {
  if (mode === "gpu-only") return "Solo GPU";
  if (mode === "cpu-only") return "Solo CPU";
  return "Automatic";
}

function inferenceModelOption(snapshot: PublicSnapshot, model: PublicSnapshot["models"][number]) {
  const deployments = snapshot.workers.flatMap((worker) => worker.deployments).filter((deployment) => deployment.model === model.id);
  const peers = snapshot.workers.filter((worker) => worker.connected && worker.deployments.some((deployment) => deployment.model === model.id));
  const adapters = deployments.map((deployment) => deployment.adapter).filter((adapter): adapter is NonNullable<PublicDeployment["adapter"]> => adapter !== undefined);
  const legacyExternalRuntime = adapters.some(
    (adapter) => adapter !== "mycellios-pipeline",
  );
  const freeSlots = deployments.reduce((total, deployment) => total + deployment.freeSlots, 0);
  const routeLabel = model.pipelines > 0
    ? `${model.pipelines} pipeline${model.pipelines === 1 ? "" : "s"}`
    : `${model.replicas} replica${model.replicas === 1 ? "" : "s"}`;
  return {
    ...model,
    legacyExternalRuntime,
    freeSlots,
    routeLabel,
    nodeCount: peers.length,
    peerMemoryMb: peers.reduce((total, worker) => total + worker.offeredVramMb, 0),
    execution: summarizeDeploymentExecution(deployments),
  };
}

function shortBuildIdentity(
  identity: NativeBuildIdentity | null | undefined,
): string {
  return shortSourceId(identity?.sourceId);
}

function shortSourceId(sourceId: string | null | undefined): string {
  return sourceId
    ? sourceId.slice("sha256:".length, "sha256:".length + 12)
    : "undeclared";
}

function buildIdentityTitle(
  identity: NativeBuildIdentity | null | undefined,
): string {
  return identity
    ? `Exact source ${identity.sourceId} · version ${identity.version}`
    : "This runtime has not published a sealed source identity.";
}

function formatDuration(milliseconds: number): string {
  if (!Number.isFinite(milliseconds) || milliseconds <= 0) return "—";
  return milliseconds < 1_000 ? `${Math.round(milliseconds)} ms` : `${(milliseconds / 1_000).toFixed(milliseconds < 10_000 ? 2 : 1)} s`;
}

function formatResponseThroughput(response: ChatResponse): string {
  if (response.activeMs <= 0 || response.outputTokens <= 0) return "—";
  return (response.outputTokens / (response.activeMs / 1_000)).toFixed(2);
}

function formatLiveThroughput(update: ChatStreamUpdate): string {
  if (update.elapsedMs <= 0 || update.outputTokens <= 0) return "—";
  return (update.outputTokens / (update.elapsedMs / 1_000)).toFixed(2);
}

function splitStreamingContent(text: string): { reasoning: string | null; answer: string } {
  const opening = /^\s*<think>/i.exec(text);
  if (!opening) return { reasoning: null, answer: text };
  const content = text.slice(opening[0].length);
  const closingIndex = content.toLowerCase().indexOf("</think>");
  if (closingIndex < 0) return { reasoning: content, answer: "" };
  return {
    reasoning: content.slice(0, closingIndex),
    answer: content.slice(closingIndex + "</think>".length).replace(/^\s+/, ""),
  };
}

function splitThinkingContent(text: string): { reasoning: string | null; answer: string } {
  const match = /^\s*<think>([\s\S]*?)<\/think>\s*/i.exec(text);
  if (!match) return { reasoning: null, answer: text.trim() };
  const reasoning = match[1]?.trim() || null;
  const answer = text.slice(match[0].length).trim();
  return { reasoning, answer: answer || "The model returned no final content." };
}

function friendlyInferenceError(message: string): string {
  const normalized = message.toLowerCase();
  if (normalized.includes("first token")) return "The route produced no token after retrying. The coordinator marked it degraded and will activate auto-repair.";
  if (normalized.includes("disconnected from the distributed pipeline")) return "A device in the distributed route disconnected. The request was safely cancelled while the network rebuilds the route.";
  if (normalized.includes("no_candidates") || normalized.includes("no candidate") || normalized.includes("unavailable")) return "The model is no longer available. Check its nodes and try again.";
  if (normalized.includes("timeout") || normalized.includes("deadline")) return "The network took too long to respond. The request was cancelled without inventing a response.";
  if (
    normalized.includes("401") ||
    normalized.includes("invalid_network_token") ||
    normalized.includes("invalid network token") ||
    normalized.includes("administrator token") ||
    normalized.includes("credential")
  ) return "The network requires a valid credential to use this model.";
  return message;
}

function inferenceWaitingStatus(update: ChatStreamUpdate): string {
  if (update.phase === "recovering") {
    if (update.statusMessage) return update.statusMessage;
    return update.affectedNodeId
      ? `Node ${shortId(update.affectedNodeId)} disconnected. Reconnecting the route and retrying…`
      : "A stage lost its connection. Reconnecting the route and retrying…";
  }
  if (update.phase === "connecting") return "Looking for an available route…";
  if (update.phase === "waiting_first_token") return "Route ready. Waiting for the first token…";
  return "Waiting for the model's first token…";
}

async function fetchBenchmarkRuns(): Promise<BenchmarkRun[]> {
  const response = await fetch("/public/v1/benchmarks", { cache: "no-store" });
  const body = await response.json() as { runs?: BenchmarkRun[]; error?: { message?: string } };
  if (!response.ok) throw new Error(body.error?.message ?? `HTTP ${response.status}`);
  return body.runs ?? [];
}

async function requestBenchmarkRun(): Promise<BenchmarkRun> {
  const response = await fetch("/local/v1/benchmarks/run", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{}",
  });
  const body = await response.json() as { run?: BenchmarkRun; error?: { message?: string } };
  if (!response.ok || !body.run) throw new Error(body.error?.message ?? `HTTP ${response.status}`);
  return body.run;
}

function benchmarkSnapshotMetric(snapshot: BenchmarkRunSnapshot, metric: BenchmarkTrendMetric): number | null {
  if (metric === "tokensPerSecond") return snapshot.tokensPerSecond;
  if (metric === "ttftMsP95") return snapshot.ttftMsP95;
  if (metric === "nodes") return snapshot.nodes;
  return snapshot.tokensPerSecondPerNode;
}

function formatSignedBenchmark(value: number | null, suffix: string): string {
  if (value === null || !Number.isFinite(value)) return "No comparison";
  return `${value > 0 ? "+" : ""}${value.toLocaleString("en-US", { maximumFractionDigits: 2 })}${suffix}`;
}

function formatBenchmark(value: number | null, suffix: string): string {
  if (value === null || !Number.isFinite(value)) return "No reading";
  return `${value.toLocaleString("en-US", { maximumFractionDigits: 2 })}${suffix}`;
}

function formatBenchmarkRange(snapshot: BenchmarkRunSnapshot | null | undefined): string {
  if (!snapshot || snapshot.tokensPerSecondP5 === null || snapshot.tokensPerSecondP95 === null) {
    return "P5–P95 No reading";
  }
  return `P5 ${formatBenchmark(snapshot.tokensPerSecondP5, " tok/s")} · P95 ${formatBenchmark(snapshot.tokensPerSecondP95, " tok/s")}`;
}

function formatPowerReading(value: number | null): string {
  return value === null || !Number.isFinite(value) ? "No reading" : formatPower(value);
}

function benchmarkStabilityLabel(measurement: BenchmarkMeasurement): string {
  if (measurement.workload.statisticallyStable === true) return "Stable";
  if (measurement.workload.statisticallyStable === false) return "Inconclusive";
  return "No reading";
}

function benchmarkBuildRevision(run: BenchmarkRun): string {
  const revision = run.build.revision ?? run.gitCommit;
  return revision?.trim() ? revision.trim().slice(0, 12) : "No reading";
}

function benchmarkDirtyLabel(dirty: boolean | null): string {
  if (dirty === true) return "code with local changes";
  if (dirty === false) return "clean build";
  return "build cleanliness: No reading";
}

function formatTopologyDigest(digest: string | null): string {
  if (!digest?.trim()) return "topology: No reading";
  const normalizedDigest = digest.trim().replace(/^sha256:/i, "");
  return `topology: ${normalizedDigest.slice(0, 12)}`;
}

function formatBenchmarkBackend(measurement: BenchmarkMeasurement): string {
  const backends = measurementBackends(measurement);
  return backends[0] === "__without_reading__"
    ? "Backend: No reading"
    : backends.map((backend) => backend.toUpperCase()).join(" + ");
}

function nullableBenchmarkMax(values: Array<number | null | undefined>): number | null {
  const present = values.filter((value): value is number =>
    value !== null && value !== undefined && Number.isFinite(value)
  );
  return present.length > 0 ? Math.max(...present) : null;
}

function formatCountReading(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? "No reading"
    : Math.round(value).toLocaleString("en-US");
}

function formatRate(value: number | null): string {
  return value === null || !Number.isFinite(value)
    ? "No reading"
    : `${(value * 100).toLocaleString("en-US", { maximumFractionDigits: 2 })}%`;
}

function benchmarkCampaignStopLabel(
  reason: BenchmarkMeasurement["workload"]["campaignStopReason"],
): string {
  if (reason === "confidence_reached") return "Confidence reached";
  if (reason === "maximum_samples") return "Maximum samples";
  if (reason === "insufficient_samples") return "Insufficient samples";
  return "No reading";
}

function nullableBenchmarkSum(values: Array<number | null | undefined>): number | null {
  const present = values.filter((value): value is number =>
    value !== null && value !== undefined && Number.isFinite(value)
  );
  return present.length > 0 ? present.reduce((sum, value) => sum + value, 0) : null;
}

function formatBenchmarkDate(value: string): string {
  return new Date(value).toLocaleString("en-US", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
}

function formatBenchmarkChartTime(value: string): string {
  return new Date(value).toLocaleString("en-US", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function benchmarkStatusLabel(status: BenchmarkRun["status"]): string {
  if (status === "baseline") return "Referencia";
  if (status === "passed") return "Valid";
  if (status === "regression") return "Regression";
  if (status === "inconclusive") return "Inconcluso";
  return "Fallido";
}

export { Panel };
