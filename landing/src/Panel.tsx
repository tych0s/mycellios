import {
  Activity,
  ArrowRight,
  Boxes,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  CirclePower,
  Cpu,
  Download,
  ExternalLink,
  Gauge,
  Globe2,
  HardDrive,
  House,
  Laptop,
  LayoutDashboard,
  LoaderCircle,
  Maximize2,
  MemoryStick,
  Menu,
  MessageSquareText,
  Minus,
  Network,
  Play,
  Radio,
  RefreshCw,
  Search,
  Send,
  Server,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Smartphone,
  Trash2,
  Wifi,
  X,
  Zap,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type FormEvent } from "react";
import "@fontsource-variable/manrope";
import type {
  DashboardSnapshot,
  ChatResponse,
  ChatStreamUpdate,
  DesktopBridge,
  DesktopSettings,
  DesktopUpdateStatus,
  HubCatalogModel,
  RequestModelInput,
  RequestedModelCapacity,
} from "../../src/desktop/contracts";
import { consumeChatCompletionStream } from "../../src/desktop/chat-stream";
import type { BenchmarkMeasurement, BenchmarkRun } from "../../src/benchlab/types";
import { Contribute } from "./Contribute";
import brandIcon from "./assets/mycellios-app-icon-v2.png";
import "./panel.css";

const PUBLIC_COORDINATOR_URL = "https://www.mycellios.com";
import "./panel-downloads.css";
import "./panel-desktop.css";

type PanelView = "overview" | "nodes" | "models" | "jobs" | "tests" | "inference" | "contribute" | "join" | "downloads" | "machine" | "settings";

interface PanelProps {
  desktopBridge?: DesktopBridge;
  mobileEntry?: boolean;
}

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
  ttftMs?: number;
}

interface PublicWorker {
  id: string;
  kind: "desktop" | "browser";
  status: "online" | "suspect" | "offline" | "draining";
  connected: boolean;
  region: string;
  offeredVramMb: number;
  reliability: number;
  jobsCompleted: number;
  lastSeenAt: string;
  gpus: PublicGpu[];
  deployments: PublicDeployment[];
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

const EMPTY: PublicSnapshot = {
  capturedAt: new Date(0).toISOString(),
  version: "—",
  summary: { registered: 0, connected: 0, online: 0, mobile: 0, offeredVramMb: 0, completedJobs: 0 },
  workers: [],
  models: [],
  requestedModels: [],
  jobs: [],
};

const sharedNavItems: Array<{ id: PanelView; label: string; icon: typeof Network }> = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "nodes", label: "Nodes", icon: Server },
  { id: "jobs", label: "Tasks", icon: Activity },
  { id: "tests", label: "Tests", icon: Gauge },
  { id: "models", label: "Models", icon: Boxes },
  { id: "inference", label: "Chat", icon: MessageSquareText },
  { id: "contribute", label: "This device", icon: Zap },
  { id: "downloads", label: "Downloads", icon: Download },
];

function initialView(desktop: boolean, mobileEntry: boolean): PanelView {
  if (desktop) return "overview";
  const requested = new URLSearchParams(window.location.search).get("view");
  if (sharedNavItems.some((item) => item.id === requested)) return requested as PanelView;
  if (mobileEntry || window.location.pathname.startsWith("/mobile")) return "contribute";
  if (window.location.pathname === "/join") return "join";
  if (window.location.pathname === "/downloads") return "downloads";
  return "overview";
}

function Panel({ desktopBridge, mobileEntry = false }: PanelProps = {}) {
  const desktop = desktopBridge !== undefined;
  const localBrowser = !desktop && ["localhost", "127.0.0.1", "::1", "[::1]"].includes(window.location.hostname);
  const navItems = useMemo(() => {
    if (!desktop) return sharedNavItems;
    const desktopItems = sharedNavItems.filter((item) => item.id !== "contribute");
    return [
        ...desktopItems.slice(0, 1),
        { id: "machine" as const, label: "This device", icon: Cpu },
        ...desktopItems.slice(1),
        { id: "settings" as const, label: "Settings", icon: Settings },
      ];
  }, [desktop]);
  const [view, setView] = useState<PanelView>(() => initialView(desktop, mobileEntry));
  const [snapshot, setSnapshot] = useState<PublicSnapshot>(EMPTY);
  const [desktopSnapshot, setDesktopSnapshot] = useState<DashboardSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [modelAdminToken, setModelAdminToken] = useState(() =>
    desktop ? "" : window.sessionStorage.getItem("mycellios-model-admin-token") ?? "",
  );
  const applyDesktopSnapshot = useCallback((next: DashboardSnapshot) => {
    setDesktopSnapshot(next);
    setSnapshot(desktopToPublicSnapshot(next));
    setError(next.connectionError ?? next.runtimeError);
  }, []);

  const refresh = useCallback(async () => {
    try {
      if (desktopBridge) {
        const next = await desktopBridge.getSnapshot();
        applyDesktopSnapshot(next);
        return;
      }
      const response = await fetch("/public/v1/snapshot", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setSnapshot(await response.json() as PublicSnapshot);
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setLoading(false);
    }
  }, [applyDesktopSnapshot, desktopBridge]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2_000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  function navigate(next: PanelView) {
    setView(next);
    setMenuOpen(false);
    if (!desktop) {
      const path = mobileEntry ? "/mobile/" : next === "join" ? "/join" : next === "downloads" ? "/downloads" : "/network";
      const query = mobileEntry
        ? next === "contribute" ? "" : `?view=${next}`
        : next === "overview" || next === "join" || next === "downloads" ? "" : `?view=${next}`;
      window.history.replaceState({}, "", `${path}${query}`);
    }
  }

  async function removeWorker(workerId: string) {
    if (desktopBridge) {
      const next = await desktopBridge.removeWorker(workerId);
      applyDesktopSnapshot(next);
      return;
    }
    const response = await fetch(`/public/v1/workers/${encodeURIComponent(workerId)}`, { method: "DELETE" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    await refresh();
  }

  async function clearOfflineWorkers() {
    if (desktopBridge) {
      const next = await desktopBridge.clearOfflineWorkers();
      applyDesktopSnapshot(next);
      return;
    }
    const response = await fetch("/public/v1/workers/clear-offline", { method: "POST" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    await refresh();
  }

  async function requestModel(input: RequestModelInput, adminToken: string) {
    if (desktopBridge) {
      applyDesktopSnapshot(await desktopBridge.requestModel(input));
      return;
    }
    const token = adminToken.trim();
    if (!token && !localBrowser) throw new Error("Enter the network administration token.");
    if (token) {
      window.sessionStorage.setItem("mycellios-model-admin-token", token);
      setModelAdminToken(token);
    }
    const response = await fetch("/public/v1/requested-models", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(token ? { authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(input),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => null) as { error?: { message?: string } } | null;
      throw new Error(body?.error?.message ?? `HTTP ${response.status}`);
    }
    await refresh();
  }

  const searchHubModels = useCallback(async (query: string): Promise<HubCatalogModel[]> => {
    if (desktopBridge) return desktopBridge.searchHubModels(query);
    const response = await fetch(`/public/v1/huggingface-models?q=${encodeURIComponent(query)}`, { cache: "no-store" });
    if (!response.ok) {
      const body = await response.json().catch(() => null) as { error?: { message?: string } } | null;
      throw new Error(body?.error?.message ?? `HTTP ${response.status}`);
    }
    const body = await response.json() as { data: HubCatalogModel[] };
    return body.data;
  }, [desktopBridge]);

  async function removeRequestedModel(modelId: string, adminToken: string) {
    if (desktopBridge) {
      applyDesktopSnapshot(await desktopBridge.removeRequestedModel(modelId));
      return;
    }
    const token = adminToken.trim();
    if (!token && !localBrowser) throw new Error("Enter the network administration token.");
    const response = await fetch(`/public/v1/requested-models/${encodeURIComponent(modelId)}`, {
      method: "DELETE",
      ...(token ? { headers: { authorization: `Bearer ${token}` } } : {}),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    await refresh();
  }

  async function sendPrompt(model: string, prompt: string, onUpdate?: (update: ChatStreamUpdate) => void): Promise<ChatResponse> {
    if (desktopBridge?.streamChat) return desktopBridge.streamChat({ model, prompt }, onUpdate ?? (() => undefined));
    if (desktopBridge) return desktopBridge.sendChat({ model, prompt });
    const startedAt = Date.now();
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { accept: "text/event-stream", "content-type": "application/json" },
      body: JSON.stringify({ model, messages: [{ role: "user", content: prompt }], stream: true, max_tokens: 128, temperature: 0, top_p: 1 }),
    });
    return consumeChatCompletionStream(response, model, onUpdate ?? (() => undefined), startedAt);
  }

  const publicOrigin = desktopSnapshot?.settings.coordinatorMode === "remote"
    ? desktopSnapshot.coordinatorUrl.replace(/\/$/, "")
    : PUBLIC_COORDINATOR_URL;
  const publicLink = (path: string) => desktop ? `${publicOrigin}${path}` : path;
  const externalProps = desktop ? { target: "_blank", rel: "noreferrer" } as const : {};

  return (
    <div className={`public-panel ${desktop ? `desktop-panel platform-${desktopSnapshot?.platform ?? "loading"}` : ""}`}>
      <aside className={menuOpen ? "panel-sidebar open" : "panel-sidebar"}>
        <button className="panel-sidebar-close" aria-label="Close navigation" onClick={() => setMenuOpen(false)}><X size={18} /></button>
        <a className="panel-brand" href={publicLink("/")} {...externalProps}><img src={brandIcon} alt="" /><div><strong>mycellios</strong><span>network control</span></div></a>
        <nav>
          <span className="panel-nav-label">CONTROL CENTER</span>
          {navItems.map(({ id, label, icon: Icon }) => (
            <button key={id} className={view === id ? "active" : ""} onClick={() => navigate(id)}>
              <Icon size={18} /><span>{label}</span>{id === "nodes" && <b>{snapshot.summary.connected}</b>}
            </button>
          ))}
        </nav>
        <div className="panel-sidebar-links">
          <a href={publicLink("/mobile/")} {...externalProps}><Smartphone size={17} /><span>Mobile app</span><ExternalLink size={13} /></a>
          <a href={publicLink("/")} {...externalProps}><House size={17} /><span>Landing</span><ExternalLink size={13} /></a>
        </div>
        <div className={`panel-sidebar-status ${error ? "degraded" : "healthy"}`}>
          <i />
          <div>
            <span>Network Status</span>
            <strong>{error ? "Degraded" : "Healthy"}</strong>
            <small>{error ? "Coordinator unavailable" : "All systems operational"}</small>
          </div>
        </div>
      </aside>
      {menuOpen && <button className="panel-menu-backdrop" aria-label="Close navigation overlay" onClick={() => setMenuOpen(false)} />}

      <div className="panel-main">
        <header className="panel-topbar">
          <button
            className="panel-menu-button"
            aria-label="Open menu"
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((current) => !current)}
          ><Menu /></button>
          <div className="panel-topbar-status"><span className={`panel-live-dot ${error ? "degraded" : ""}`} /><div><strong>{error ? "Network degraded" : "Network healthy"}</strong><small>{snapshot.capturedAt === EMPTY.capturedAt ? "Waiting for snapshot" : `Updated ${relativeTime(snapshot.capturedAt)}`}</small></div></div>
          <div className="panel-top-actions">
            {desktopSnapshot?.update.state === "ready" && <button className="panel-update-ready" onClick={() => void desktopBridge?.installUpdate()} title="Restart and install update"><Download size={15} /></button>}
            {desktopSnapshot && <span>v{desktopSnapshot.appVersion}</span>}
            <span>API {snapshot.version}</span>
            <button onClick={() => void refresh()} aria-label="Refresh"><RefreshCw className={loading ? "spin" : ""} size={17} /></button>
            {!desktop && <a href={mobileEntry ? "/mobile/" : "/network?view=contribute"}>This device <Zap size={15} /></a>}
            {desktopBridge && <div className="panel-window-controls"><button onClick={() => void desktopBridge.minimizeWindow()} aria-label="Minimize"><Minus size={16} /></button><button onClick={() => void desktopBridge.toggleMaximizeWindow()} aria-label="Maximize"><Maximize2 size={15} /></button><button className="close" onClick={() => void desktopBridge.closeWindow()} aria-label="Close"><X size={16} /></button></div>}
          </div>
        </header>

        {error && <div className="panel-error" title={error}><CircleAlert size={17} /> Coordinator unavailable. Retrying automatically.</div>}
        <main className="panel-content">
          {loading && snapshot.capturedAt === EMPTY.capturedAt ? <PanelLoading /> : (
            <>
              {view === "overview" && <Overview snapshot={snapshot} onNavigate={navigate} publicLink={publicLink} external={desktop} />}
              {view === "nodes" && <Nodes snapshot={snapshot} onRemove={removeWorker} onClearOffline={clearOfflineWorkers} />}
              {view === "models" && <Models snapshot={snapshot} onSearch={searchHubModels} onRequest={requestModel} onRemove={removeRequestedModel} adminToken={modelAdminToken} requiresAdminToken={!desktop && !localBrowser} />}
              {view === "jobs" && <Jobs snapshot={snapshot} />}
              {view === "tests" && <Tests bridge={desktopBridge} />}
              {view === "inference" && <Inference snapshot={snapshot} onSend={sendPrompt} onNavigate={navigate} />}
              {view === "contribute" && <Contribute />}
              {view === "join" && <JoinNetwork publicLink={publicLink} external={desktop} />}
              {view === "downloads" && <Downloads publicLink={publicLink} external={desktop} />}
              {view === "machine" && desktopSnapshot && desktopBridge && <Machine snapshot={desktopSnapshot} bridge={desktopBridge} onSnapshot={applyDesktopSnapshot} />}
              {view === "settings" && desktopSnapshot && desktopBridge && <DesktopSettingsView snapshot={desktopSnapshot} bridge={desktopBridge} onSnapshot={applyDesktopSnapshot} />}
            </>
          )}
        </main>
      </div>
      {desktopSnapshot && desktopBridge && !desktopSnapshot.settings.onboardingComplete && <DesktopOnboarding snapshot={desktopSnapshot} bridge={desktopBridge} onSnapshot={applyDesktopSnapshot} />}
    </div>
  );
}

function Overview({ snapshot, onNavigate, publicLink, external }: { snapshot: PublicSnapshot; onNavigate: (view: PanelView) => void; publicLink: (path: string) => string; external: boolean }) {
  const active = snapshot.workers.filter((worker) => worker.connected);
  return (
    <section className="overview-page">
      <PageTitle eyebrow="NETWORK CONTROL" title="Overview" copy="A live view of the capacity, models and tasks connected to your mycellios network." actions={<><button onClick={() => onNavigate("nodes")}>Manage nodes</button><a href={publicLink("/mobile/")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}>Add this device <ArrowRight size={16} /></a></>} />
      <div className="panel-stat-grid">
        <Stat icon={Radio} label="Active nodes" value={String(snapshot.summary.connected)} detail={`${snapshot.summary.registered} registered`} />
        <Stat icon={HardDrive} label="Shared VRAM" value={formatMemory(snapshot.summary.offeredVramMb)} detail="Verified offered capacity" tone="purple" />
        <Stat icon={Boxes} label="Available models" value={String(snapshot.models.length)} detail="Announced by live nodes" />
        <Stat icon={CheckCircle2} label="Tasks completed" value={String(snapshot.summary.completedJobs)} detail="Current coordinator history" />
      </div>
      <div className="overview-grid">
        <article className="activity-card">
          <div className="card-heading"><div><span>RECENT ACTIVITY</span><h2>Network tasks</h2></div><button onClick={() => onNavigate("jobs")}>View all</button></div>
          <div className="activity-list">
            {snapshot.jobs.slice(0, 5).map((job) => <div key={job.id}><i className={job.status} /><div><strong>{job.model}</strong><span>{shortId(job.id)} · {relativeTime(job.updatedAt)}</span></div><b>{job.status}</b></div>)}
            {snapshot.jobs.length === 0 && <Empty icon={Activity} title="No tasks yet" copy="Tasks will appear here when you test a connected model." />}
          </div>
        </article>
        <article className="network-canvas">
          <div className="canvas-head"><div><span>LIVE TOPOLOGY</span><strong>Network</strong></div><button className="canvas-link" onClick={() => onNavigate("nodes")}>View nodes</button></div>
          <div className="network-orbit">
            <div className="orbit-grid" />
            <div className="network-core"><img src={brandIcon} alt="" /><span>mycellios</span><small>coordinator</small></div>
            {active.slice(0, 14).map((worker, index) => {
              const angle = (index / Math.max(active.length, 1)) * Math.PI * 2 - Math.PI / 2;
              const radius = active.length < 5 ? 34 : 40;
              const style = { "--node-x": `${50 + Math.cos(angle) * radius}%`, "--node-y": `${50 + Math.sin(angle) * radius}%`, "--line-angle": `${angle}rad` } as CSSProperties;
              return <div className={`orbit-node ${worker.kind}`} style={style} key={worker.id}><i /><span>{workerLabel(worker)}</span><small>{worker.gpus[0]?.model ?? worker.kind}</small></div>;
            })}
            {active.length === 0 && <div className="network-empty"><Wifi size={28} /><strong>Waiting for the first node</strong><a href={publicLink("/mobile/")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}>Connect this device</a></div>}
          </div>
          <div className="network-summary">
            <div><span>Nodes online</span><strong>{snapshot.summary.connected} / {snapshot.summary.registered}</strong></div>
            <div><span>Shared VRAM</span><strong>{formatMemory(snapshot.summary.offeredVramMb)}</strong></div>
          </div>
        </article>
      </div>
      <div className="overview-announcement">
        <div className="announcement-icon"><Sparkles size={21} /></div>
        <div><strong>Public network validation</strong><span>Connect a device to help validate physical multi-node inference with real hardware.</span></div>
        <button onClick={() => onNavigate("join")}>Join network <ArrowRight size={16} /></button>
      </div>
    </section>
  );
}

function Nodes({ snapshot, onRemove, onClearOffline }: { snapshot: PublicSnapshot; onRemove: (workerId: string) => Promise<void>; onClearOffline: () => Promise<void> }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [selectedWorkerId, setSelectedWorkerId] = useState("");
  const connectedWorkers = snapshot.workers.filter((worker) => worker.connected);
  const activeWorkers = snapshot.workers.filter((worker) => worker.connected && worker.status === "online");
  const activeGpus = activeWorkers.flatMap((worker) => worker.gpus);
  const activeDeployments = activeWorkers.flatMap((worker) => worker.deployments);
  const telemetryGpus = activeGpus.filter((gpu) => gpu.utilizationPct !== undefined);
  const poweredGpus = activeGpus.filter((gpu) => (gpu.powerW ?? 0) > 0);
  const activeOfferedVramMb = activeGpus.reduce((total, gpu) => total + gpu.offeredVramMb, 0);
  const freeVramMb = activeGpus.reduce((total, gpu) => total + gpu.freeOfferedVramMb, 0);
  const averageUtilization = telemetryGpus.length > 0 ? telemetryGpus.reduce((total, gpu) => total + (gpu.utilizationPct ?? 0), 0) / telemetryGpus.length : null;
  const observedPowerW = poweredGpus.reduce((total, gpu) => total + (gpu.powerW ?? 0), 0);
  const activeThroughput = activeDeployments.reduce((total, deployment) => total + deployment.tokensPerSecond, 0);
  const sortedWorkers = [...snapshot.workers].sort((left, right) => nodeStateOrder(left) - nodeStateOrder(right));
  const selectedWorker = snapshot.workers.find((worker) => worker.id === selectedWorkerId) ?? activeWorkers[0] ?? snapshot.workers[0] ?? null;
  async function remove(workerId: string) {
    setBusy(workerId);
    try {
      await onRemove(workerId);
    } finally {
      setBusy(null);
    }
  }
  async function clearOffline() {
    setBusy("offline");
    try {
      await onClearOffline();
    } finally {
      setBusy(null);
    }
  }
  return (
    <section className="nodes-page">
      <PageTitle eyebrow="RED DISTRIBUIDA" title="Red de nodos" copy="Estado, capacidad y conexiones reales de todas las máquinas registradas en mycellios." actions={<button disabled={busy !== null} onClick={() => void clearOffline()}><Trash2 size={15} /> Limpiar inactivos</button>} />

      <div className="node-overview-grid">
        <NodeOverviewStat icon={Server} label="Nodos activos" value={`${activeWorkers.length} / ${snapshot.workers.length}`} detail={`${connectedWorkers.length} conectados ahora`} progress={snapshot.workers.length > 0 ? activeWorkers.length / snapshot.workers.length : 0} tone="green" />
        <NodeOverviewStat icon={MemoryStick} label="Capacidad activa" value={formatMemory(activeOfferedVramMb)} detail={`${formatMemory(freeVramMb)} libres`} progress={activeOfferedVramMb > 0 ? freeVramMb / activeOfferedVramMb : 0} tone="blue" />
        <NodeOverviewStat icon={Gauge} label="Carga GPU media" value={averageUtilization === null ? "Sin datos" : `${averageUtilization.toFixed(0)}%`} detail={`${telemetryGpus.length} GPU${telemetryGpus.length === 1 ? "" : "s"} con telemetría`} progress={averageUtilization === null ? 0 : averageUtilization / 100} tone="purple" />
        <NodeOverviewStat icon={Zap} label="Rendimiento anunciado" value={activeThroughput > 0 ? `${formatCompactNumber(activeThroughput)} tok/s` : "—"} detail={`${activeDeployments.length} despliegue${activeDeployments.length === 1 ? "" : "s"} activo${activeDeployments.length === 1 ? "" : "s"}`} tone="violet" />
        <NodeOverviewStat icon={Activity} label="Potencia observada" value={poweredGpus.length > 0 ? formatPower(observedPowerW) : "Sin datos"} detail={poweredGpus.length > 0 ? `${poweredGpus.length} lectura${poweredGpus.length === 1 ? "" : "s"} física${poweredGpus.length === 1 ? "" : "s"}` : "No se estima ni se inventa"} tone="cyan" />
      </div>

      <section className="node-topology-card">
        <header className="node-topology-head">
          <div><span>TOPOLOGÍA EN VIVO</span><h2>Conexiones con el coordinador</h2><p>Cada línea representa una sesión real con mycellios.</p></div>
          <div className="node-topology-legend"><span><i className="active" />Activo</span><span><i className="warning" />Atención</span><span><i className="offline" />Inactivo</span></div>
        </header>
        <div className={`node-topology-map${snapshot.workers.length === 0 ? " empty" : ""}`}>
          <div className="node-topology-grid" />
          <svg className="node-topology-links" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
            {sortedWorkers.map((worker, index) => {
              const position = topologyPosition(index, sortedWorkers.length);
              return <line key={worker.id} className={`${nodeVisualState(worker)}${selectedWorker?.id === worker.id ? " selected" : ""}`} x1="50" y1="52" x2={position.x} y2={position.y} vectorEffect="non-scaling-stroke" />;
            })}
          </svg>
          <div className="node-network-hub"><div><Network /></div><strong>mycellios</strong><span>{activeWorkers.length} activo{activeWorkers.length === 1 ? "" : "s"}</span><i /></div>
          {sortedWorkers.map((worker, index) => {
            const position = topologyPosition(index, sortedWorkers.length);
            const label = workerLabel(worker);
            return <button
              type="button"
              key={worker.id}
              aria-label={`Ver nodo ${label}`}
              aria-pressed={selectedWorker?.id === worker.id}
              className={`topology-node ${nodeVisualState(worker)}${selectedWorker?.id === worker.id ? " selected" : ""}`}
              style={{ "--node-x": `${position.x}%`, "--node-y": `${position.y}%` } as CSSProperties}
              onClick={() => setSelectedWorkerId(worker.id)}
            ><div>{worker.kind === "browser" ? <Smartphone /> : <Laptop />}<i /></div><span><strong>{label}</strong><small>{worker.region} · {nodeStatusLabel(worker)}</small></span></button>;
          })}
          {snapshot.workers.length === 0 && <div className="node-topology-empty"><Server /><strong>Aún no hay nodos</strong><span>Conecta una máquina para verla aparecer en la red.</span></div>}
        </div>
        {selectedWorker && <NodeTopologyInspector worker={selectedWorker} />}
      </section>

      <div className="node-list-heading"><div><span>INVENTARIO COMPLETO</span><h2>Detalle de nodos</h2></div><strong>{snapshot.workers.length} registrado{snapshot.workers.length === 1 ? "" : "s"}</strong></div>
      <div className="node-card-grid">
        {snapshot.workers.map((worker) => <article className="node-card" key={worker.id}>
          <div className="node-card-head"><div className={`node-device ${worker.kind}`}>{worker.kind === "browser" ? <Smartphone /> : <Laptop />}</div><div><span>{worker.kind === "browser" ? "NODO DE NAVEGADOR" : "AGENTE DE ESCRITORIO"}</span><h2>{workerLabel(worker)}</h2><small>{shortId(worker.id)}</small></div><span className={`node-status ${worker.status}`}><i />{nodeStatusLabel(worker)}</span></div>
          <div className="node-card-metrics"><Metric label="Hardware" value={worker.gpus[0]?.model ?? "Desconocido"} /><Metric label="Ofrecido" value={formatMemory(worker.offeredVramMb)} /><Metric label="Región" value={worker.region} /><Metric label="Completadas" value={String(worker.jobsCompleted)} /></div>
          <div className="node-card-foot"><span>Visto {relativeTimeEs(worker.lastSeenAt)}</span><button disabled={busy === worker.id} onClick={() => void remove(worker.id)}>{busy === worker.id ? <LoaderCircle className="spin" /> : <Trash2 />} Eliminar</button></div>
        </article>)}
        {snapshot.workers.length === 0 && <div className="wide-empty"><Empty icon={Server} title="No hay nodos registrados" copy="Abre el worker móvil o instala mycellios para añadir la primera máquina." /><a href="/mobile/">Conectar un dispositivo <ArrowRight size={15} /></a></div>}
      </div>
    </section>
  );
}

function NodeOverviewStat({ icon: Icon, label, value, detail, progress, tone }: { icon: typeof Server; label: string; value: string; detail: string; progress?: number; tone: "green" | "blue" | "purple" | "violet" | "cyan" }) {
  return <article className={`node-overview-stat ${tone}`}><div className="node-overview-icon"><Icon /></div><span>{label}</span><strong>{value}</strong><small>{detail}</small>{progress !== undefined && <div className="node-overview-progress"><i style={{ width: `${Math.max(0, Math.min(1, progress)) * 100}%` }} /></div>}</article>;
}

function NodeTopologyInspector({ worker }: { worker: PublicWorker }) {
  const gpus = worker.gpus;
  const free = gpus.reduce((total, gpu) => total + gpu.freeOfferedVramMb, 0);
  const utilizations = gpus.filter((gpu) => gpu.utilizationPct !== undefined);
  const utilization = utilizations.length > 0 ? utilizations.reduce((total, gpu) => total + (gpu.utilizationPct ?? 0), 0) / utilizations.length : null;
  const throughput = worker.deployments.reduce((total, deployment) => total + deployment.tokensPerSecond, 0);
  return <div className="node-topology-inspector">
    <div className={`node-inspector-identity ${nodeVisualState(worker)}`}><div>{worker.kind === "browser" ? <Smartphone /> : <Laptop />}</div><span><small>NODO SELECCIONADO</small><strong>{workerLabel(worker)}</strong><em>{worker.region} · {shortId(worker.id)}</em></span></div>
    <div className="node-inspector-metrics"><Metric label="Estado" value={nodeStatusLabel(worker)} /><Metric label="GPU libre" value={formatMemory(free)} /><Metric label="Carga" value={utilization === null ? "Sin datos" : `${utilization.toFixed(0)}%`} /><Metric label="Rend. anunciado" value={throughput > 0 ? `${formatCompactNumber(throughput)} tok/s` : "—"} /><Metric label="Fiabilidad" value={`${Math.round(worker.reliability * 100)}%`} /><Metric label="Modelos" value={String(worker.deployments.length)} /></div>
  </div>;
}

function Models({ snapshot, onSearch, onRequest, onRemove, adminToken: initialAdminToken, requiresAdminToken }: {
  snapshot: PublicSnapshot;
  onSearch: (query: string) => Promise<HubCatalogModel[]>;
  onRequest: (input: RequestModelInput, adminToken: string) => Promise<void>;
  onRemove: (modelId: string, adminToken: string) => Promise<void>;
  adminToken: string;
  requiresAdminToken: boolean;
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
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [catalogOpen, setCatalogOpen] = useState(true);
  const searchSequence = useRef(0);

  useEffect(() => {
    const query = catalogQuery.trim();
    if (query.length === 1) {
      setCatalogModels([]);
      setCatalogBusy(false);
      setCatalogError(null);
      return;
    }
    const sequence = ++searchSequence.current;
    setCatalogBusy(true);
    setCatalogError(null);
    const timer = window.setTimeout(() => {
      void onSearch(query).then(
        (models) => {
          if (searchSequence.current !== sequence) return;
          setCatalogModels(models);
          setCatalogBusy(false);
        },
        (error: unknown) => {
          if (searchSequence.current !== sequence) return;
          setCatalogModels([]);
          setCatalogError(errorText(error));
          setCatalogBusy(false);
        },
      );
    }, query ? 320 : 0);
    return () => window.clearTimeout(timer);
  }, [catalogQuery, onSearch]);

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
      setFormError(errorText(error));
    } finally {
      setBusy(false);
    }
  }

  return <section className="models-page">
    <PageTitle eyebrow="MODEL FABRIC" title="Models" copy="Choose a compatible Hugging Face model. mycellios prepares its name, calculates capacity and distributes it automatically." />
    <form className="model-request-form" onSubmit={(event) => void submit(event)}>
      <div className="model-form-heading"><div><span>HUGGING FACE CATALOG</span><h2>Choose a model for the network</h2><p>Only certified architectures can be selected. No terminal commands or pre-split files.</p></div><Boxes size={24} /></div>

      {(!source || catalogOpen) && <>
        <div className="hub-model-search"><Search size={17} /><input value={catalogQuery} onChange={(event) => setCatalogQuery(event.target.value)} placeholder="Search Qwen, Llama, GLM…" aria-label="Search Hugging Face models" />{catalogBusy && <LoaderCircle className="spin" />}</div>
        <div className="hub-catalog-heading"><span>{catalogQuery.trim() ? "SEARCH RESULTS" : "POPULAR COMPATIBLE MODELS"}</span><a href="https://huggingface.co/models?pipeline_tag=text-generation" target="_blank" rel="noreferrer">Explore Hugging Face <ExternalLink size={12} /></a></div>
        {catalogError && <div className="hub-catalog-state error"><CircleAlert size={15} />Catalog unavailable. You can still enter a model manually below.</div>}
        {!catalogError && !catalogBusy && catalogModels.length === 0 && <div className="hub-catalog-state">{catalogQuery.trim().length === 1 ? "Type at least two characters to search." : "No matching models found."}</div>}
        {catalogModels.length > 0 && <div className="hub-model-grid">{catalogModels.map((model) => <button type="button" key={model.id} className={`hub-model-card ${source === model.id ? "selected" : ""} ${model.compatible ? "compatible" : "unsupported"}`} aria-disabled={!model.compatible} onClick={() => selectCatalogModel(model)} title={model.compatibilityReason ?? `Select ${model.id}`}>
          <span className="hub-model-card-head"><i><Boxes /></i><strong>{model.id}</strong>{source === model.id && <CheckCircle2 />}</span>
          <small>{model.architecture ?? model.modelType ?? "Architecture unavailable"}</small>
          <span className="hub-model-card-meta"><b>{model.compatible ? "CERTIFIED" : model.gated ? "GATED" : "NOT SUPPORTED"}</b><em>{formatHubCount(model.downloads)} downloads</em></span>
        </button>)}</div>}
      </>}

      {source && <div className="hub-selected-model"><CheckCircle2 /><span><small>SELECTED MODEL</small><strong>{source}</strong><em>It will be published as {modelId}</em></span><button type="button" onClick={() => setCatalogOpen(true)}>Change model</button></div>}

      <details className="model-advanced">
        <summary><SlidersHorizontal size={15} /> Advanced settings or manual model ID <ChevronDown size={15} /></summary>
        <div className="model-form-grid">
          <label>HUGGING FACE MODEL<input value={source} onChange={(event) => { setSource(event.target.value); setModelId(networkNameForHubModel(event.target.value)); }} placeholder="Qwen/Qwen3-0.6B" required /></label>
          <label>NETWORK NAME <small>automatic</small><input value={modelId} onChange={(event) => setModelId(event.target.value)} placeholder="qwen3-0.6b" required pattern="[A-Za-z0-9][A-Za-z0-9._-]*" /></label>
          <label>REVISION <small>optional</small><input value={revision} onChange={(event) => setRevision(event.target.value)} placeholder="main or commit" /></label>
          <label>CONTEXT TOKENS<input type="number" min={128} max={1048576} value={contextTokens} onChange={(event) => setContextTokens(Number(event.target.value))} /></label>
          <label>MINIMUM NODES<AppSelect ariaLabel="Minimum nodes" value={String(minimumNodes)} onChange={(value) => setMinimumNodes(Number(value))} options={[2, 3, 4, 5, 6, 7, 8].map((count) => ({ value: String(count), label: String(count) }))} /></label>
          {requiresAdminToken && <label>NETWORK ADMIN TOKEN<input type="password" autoComplete="current-password" value={adminToken} onChange={(event) => setAdminToken(event.target.value)} placeholder="Required to deploy models" required /></label>}
        </div>
      </details>
      <div className="model-submit-row">
        <label className="model-auto-toggle"><input type="checkbox" checked={autoActivate} onChange={(event) => setAutoActivate(event.target.checked)} /><span><strong>Activate automatically</strong><small>Start as soon as compatible capacity reaches the requirement.</small></span></label>
        <button className="model-submit" disabled={busy || !source.trim() || !modelId.trim()}>{busy ? <LoaderCircle className="spin" /> : <Play size={16} />}{busy ? "Inspecting model…" : "Calculate and prepare"}</button>
      </div>
      {formError && <div className="model-form-error"><CircleAlert size={16} />{formError}</div>}
    </form>

    <div className="model-request-list">
      {snapshot.requestedModels.map((model) => <RequestedModelCard key={model.id} model={model} onRemove={(modelId) => onRemove(modelId, adminToken)} />)}
    </div>

    <div className="panel-table model-active-table"><div className="panel-table-head"><span>Active model</span><span>Replicas</span><span>Pipelines</span><span>Status</span></div>
      {snapshot.models.map((model) => <div className="panel-table-row" key={model.id}><strong><Boxes size={17} />{model.id}</strong><span>{model.replicas}</span><span>{model.pipelines}</span><b><i />Available</b></div>)}
      {snapshot.models.length === 0 && <Empty icon={Boxes} title="No active models" copy="Choose a model above. It will remain queued with an exact capacity shortfall until the network can run it." />}
    </div>
  </section>;
}

function RequestedModelCard({ model, onRemove }: { model: RequestedModelCapacity; onRemove: (modelId: string) => Promise<void> }) {
  const [removing, setRemoving] = useState(false);
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
    <div className="model-capacity-grid">
      <Metric label="Estimated need" value={model.requiredVramMiB === null ? "Calculating" : formatMemory(model.requiredVramMiB)} />
      <Metric label="Available" value={formatMemory(model.availableVramMiB)} />
      <Metric label="Estimated shortfall" value={model.missingVramMiB === null ? "—" : formatMemory(model.missingVramMiB)} />
      <Metric label="Nodes" value={`${model.availableNodes} / ${model.requiredNodes}`} />
    </div>
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

function Jobs({ snapshot }: { snapshot: PublicSnapshot }) {
  return <section><PageTitle eyebrow="EXECUTION LOG" title="Network jobs" copy="Recent inference requests and their execution state." />
    <div className="panel-table jobs-table"><div className="panel-table-head"><span>Request</span><span>Model</span><span>Tokens</span><span>Status</span></div>
      {snapshot.jobs.map((job) => <div className="panel-table-row" key={job.id}><strong><Activity size={16} />{shortId(job.id)}<small>{relativeTime(job.createdAt)}</small></strong><span>{job.model}</span><span>{job.inputTokens + job.outputTokens}</span><b className={job.status}><i />{job.status}</b></div>)}
      {snapshot.jobs.length === 0 && <Empty icon={Activity} title="No jobs yet" copy="Use the inference console when a model becomes available." />}
    </div></section>;
}

function Tests({ bridge }: { bridge: DesktopBridge | undefined }) {
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadRuns = useCallback(async () => {
    setLoading(true);
    try {
      const next = bridge ? await bridge.getBenchmarkRuns() : await fetchBenchmarkRuns();
      setRuns(next);
      setSelectedRunId((current) => next.some((run) => run.runId === current) ? current : next[0]?.runId ?? "");
      setError(null);
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setLoading(false);
    }
  }, [bridge]);

  useEffect(() => { void loadRuns(); }, [loadRuns]);

  async function runNow() {
    setRunning(true);
    setError(null);
    try {
      const run = bridge ? await bridge.runBenchmark() : await requestBenchmarkRun();
      setRuns((current) => [run, ...current.filter((item) => item.runId !== run.runId)]);
      setSelectedRunId(run.runId);
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setRunning(false);
    }
  }

  const selected = runs.find((run) => run.runId === selectedRunId) ?? runs[0] ?? null;
  const measurements = runs.flatMap((run) => run.measurements);
  const bestThroughput = measurements.reduce<number | null>((best, item) => {
    const value = item.metrics.tokensPerSecond;
    return value === null ? best : best === null ? value : Math.max(best, value);
  }, null);
  const latestDevices = selected?.measurements.reduce((best, item) => Math.max(best, item.inventory.connectedDevices), 0) ?? 0;

  return <section className="tests-page">
    <PageTitle eyebrow="REAL BENCHMARKS" title="Performance tests" copy="Real measurements from the active runtime, saved by version so improvements and regressions stay visible." actions={<button disabled={running} onClick={() => void runNow()}>{running ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />}{running ? "Running real test…" : "Run real test"}</button>} />
    {error && <div className="benchmark-error"><CircleAlert size={17} /><span><strong>Benchmark unavailable</strong>{error}</span></div>}
    <div className="panel-stat-grid benchmark-stats">
      <Stat icon={Activity} label="Recorded runs" value={String(runs.length)} detail="No simulated records" />
      <Stat icon={Gauge} label="Best throughput" value={formatBenchmark(bestThroughput, " tok/s")} detail="Measured per request" tone="purple" />
      <Stat icon={Server} label="Connected devices" value={String(latestDevices)} detail="In the selected run" />
      <Stat icon={Boxes} label="Selected version" value={selected ? `v${selected.version}` : "—"} detail={selected ? formatBenchmarkDate(selected.finishedAt) : "No measurements yet"} />
    </div>
    {loading && runs.length === 0 ? <div className="benchmark-loading"><LoaderCircle className="spin" /><span>Loading real benchmark history…</span></div> : runs.length === 0 ? <Empty icon={Gauge} title="No real tests recorded" copy="Start the active runtime and run the first measurement. Nothing is generated if the runtime is not ready." /> : <>
      <div className="benchmark-toolbar">
        <label>RUN<AppSelect ariaLabel="Benchmark run" value={selected?.runId ?? ""} onChange={setSelectedRunId} options={runs.map((run) => ({ value: run.runId, label: `v${run.version} · ${formatBenchmarkDate(run.finishedAt)} · ${benchmarkStatusLabel(run.status)}` }))} /></label>
        {selected && <div className={`benchmark-run-state ${selected.status}`}><i />{benchmarkStatusLabel(selected.status)}<span>{selected.suite === "physical-import" ? "physical campaign" : "active runtime"}</span></div>}
      </div>
      <div className="benchmark-chart-grid">
        <BenchmarkTrend runs={runs} metric="tokensPerSecond" label="Throughput by version" unit="tok/s" />
        <BenchmarkTrend runs={runs} metric="ttftMsP95" label="TTFT P95 by version" unit="ms" />
      </div>
      {selected && <BenchmarkRunDetails run={selected} />}
    </>}
  </section>;
}

function BenchmarkTrend({ runs, metric, label, unit }: { runs: BenchmarkRun[]; metric: "tokensPerSecond" | "ttftMsP95"; label: string; unit: string }) {
  const chronological = [...runs].reverse();
  const ids = Array.from(new Set(chronological.flatMap((run) => run.measurements.map((item) => item.id))));
  const values = chronological.flatMap((run) => run.measurements.map((item) => item.metrics[metric])).filter((value): value is number => value !== null);
  const width = 720; const height = 240; const left = 52; const right = 30; const top = 28; const bottom = 42;
  const ceiling = Math.max(...values, 1);
  const x = (index: number) => chronological.length === 1 ? (left + width - right) / 2 : left + (index / (chronological.length - 1)) * (width - left - right);
  const y = (value: number) => top + (1 - value / ceiling) * (height - top - bottom);
  return <article className="benchmark-chart">
    <div><span>VERSION TREND</span><h2>{label}</h2></div>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${label}. ${ids.length} measured test series.`}>
      <line className="benchmark-axis" x1={left} y1={height - bottom} x2={width - right} y2={height - bottom} />
      {[0, 0.5, 1].map((ratio) => <g key={ratio}><line className="benchmark-grid-line" x1={left} y1={top + ratio * (height - top - bottom)} x2={width - right} y2={top + ratio * (height - top - bottom)} /><text x={left - 8} y={top + ratio * (height - top - bottom) + 4} textAnchor="end">{formatBenchmark(ceiling * (1 - ratio), "")}</text></g>)}
      {ids.map((id, seriesIndex) => {
        const points = chronological.map((run, runIndex) => {
          const measurement = run.measurements.find((item) => item.id === id);
          const value = measurement?.metrics[metric] ?? null;
          return value === null ? null : { run, value, x: x(runIndex), y: y(value) };
        }).filter((point): point is NonNullable<typeof point> => point !== null);
        return <g className={`benchmark-series series-${seriesIndex % 4}`} key={id}>
          {points.length > 1 && <polyline points={points.map((point) => `${point.x},${point.y}`).join(" ")} />}
          {points.map((point) => <circle key={point.run.runId} cx={point.x} cy={point.y} r="5"><title>{`${id} · v${point.run.version}: ${formatBenchmark(point.value, ` ${unit}`)}`}</title></circle>)}
        </g>;
      })}
      {chronological.map((run, index) => <text className="benchmark-x-label" key={run.runId} x={x(index)} y={height - 14} textAnchor="middle">v{run.version}</text>)}
    </svg>
    <div className="benchmark-legend">{ids.map((id, index) => <span key={id}><i className={`series-${index % 4}`} />{id}</span>)}</div>
  </article>;
}

function BenchmarkRunDetails({ run }: { run: BenchmarkRun }) {
  return <div className="benchmark-results panel-table">
    <div className="card-heading"><div><span>SELECTED REAL RUN</span><h2>{run.label}</h2></div><small>{run.gitCommit ? run.gitCommit.slice(0, 8) : "uncommitted"}{run.gitDirty ? " · working tree changed" : ""}</small></div>
    <div className="benchmark-table-head"><span>Test / model</span><span>Devices</span><span>Evidence</span><span>Tokens/s</span><span>TTFT P95</span><span>Change</span><span>Status</span></div>
    {run.measurements.map((measurement) => <BenchmarkResultRow measurement={measurement} key={measurement.id} />)}
  </div>;
}

function BenchmarkResultRow({ measurement }: { measurement: BenchmarkMeasurement }) {
  const devices = measurement.inventory.profiles.map((profile) => `${profile.count}× ${profile.label}`).join(" · ");
  const delta = measurement.comparison.tokensPerSecondPct;
  return <div className="benchmark-table-row">
    <strong><Gauge size={16} /><span>{measurement.title}<small>{measurement.model.label} · {measurement.model.precision} · rev {measurement.model.revision?.slice(0, 8) ?? "—"}</small></span></strong>
    <span>{measurement.inventory.connectedDevices}/{measurement.inventory.totalDevices}<small>{devices || "No device profile"}</small></span>
    <span>{measurement.evidence}<small>{measurement.environment}</small></span>
    <span className="metric-value">{formatBenchmark(measurement.metrics.tokensPerSecond, " tok/s")}</span>
    <span className="metric-value">{formatBenchmark(measurement.metrics.ttftMsP95, " ms")}</span>
    <span className={delta !== null && delta < 0 ? "negative" : "positive"}>{delta === null ? "baseline" : `${delta >= 0 ? "+" : ""}${delta.toFixed(2)}%`}</span>
    <b className={measurement.status}><i />{benchmarkStatusLabel(measurement.status)}</b>
  </div>;
}

interface InferenceTurn {
  id: string;
  prompt: string;
  response: ChatResponse;
}

interface InferencePendingTurn extends ChatStreamUpdate {
  prompt: string;
}

function Inference({ snapshot, onSend, onNavigate }: {
  snapshot: PublicSnapshot;
  onSend: (model: string, prompt: string, onUpdate?: (update: ChatStreamUpdate) => void) => Promise<ChatResponse>;
  onNavigate: (view: PanelView) => void;
}) {
  const options = useMemo(() => snapshot.models.map((item) => inferenceModelOption(snapshot, item)), [snapshot]);
  const realModels = options.filter((item) => !item.connectivityOnly);
  const connectivityModel = options.find((item) => item.connectivityOnly) ?? null;
  const [model, setModel] = useState("");
  const [prompt, setPrompt] = useState("");
  const [turns, setTurns] = useState<InferenceTurn[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [pendingTurn, setPendingTurn] = useState<InferencePendingTurn | null>(null);
  const [diagnostic, setDiagnostic] = useState<"idle" | "running" | "ok" | "failed">("idle");
  const outputRef = useRef<HTMLDivElement>(null);
  const selectedModel = realModels.some((item) => item.id === model) ? model : realModels[0]?.id ?? "";
  const selectedOption = realModels.find((item) => item.id === selectedModel) ?? null;

  useEffect(() => {
    if (!pendingTurn) return;
    const output = outputRef.current;
    if (output) output.scrollTop = output.scrollHeight;
  }, [pendingTurn?.text, pendingTurn?.outputTokens]);

  async function send() {
    const cleanPrompt = prompt.trim();
    if (!selectedModel || !cleanPrompt || pendingTurn) return;
    setPendingTurn({
      prompt: cleanPrompt,
      requestId: "pending",
      model: selectedModel,
      delta: "",
      text: "",
      outputTokens: 0,
      routeClass: "waiting",
      affinityHit: false,
      ttftMs: 0,
      elapsedMs: 0,
    });
    setPrompt("");
    setError(null);
    try {
      const response = await onSend(selectedModel, cleanPrompt, (update) => {
        setPendingTurn((current) => current ? { ...current, ...update } : current);
      });
      if (!response.text.trim()) throw new Error("El modelo terminó sin devolver texto.");
      setTurns((current) => [...current, { id: response.requestId, prompt: cleanPrompt, response }]);
    } catch (caught) {
      setPrompt(cleanPrompt);
      setError(errorText(caught));
    } finally {
      setPendingTurn(null);
    }
  }

  async function checkConnection() {
    if (!connectivityModel || diagnostic === "running") return;
    setDiagnostic("running");
    try {
      await onSend(connectivityModel.id, "ping");
      setDiagnostic("ok");
    } catch {
      setDiagnostic("failed");
    }
  }

  return <section className="inference-page">
    <PageTitle eyebrow="INFERENCIA REAL" title="Probar un modelo" copy="Habla con un modelo conectado y comprueba la ruta, la latencia y los tokens de cada respuesta." />
    {realModels.length === 0 ? <div className="inference-unavailable">
      <div className="inference-unavailable-icon"><MessageSquareText /></div>
      <div className="inference-unavailable-copy"><span>NO HAY MODELOS DE IA DISPONIBLES</span><h2>Ahora mismo no se puede hacer una inferencia real</h2><p>La red no tiene ningún runtime de IA real conectado. Un PC puede aparecer como nodo disponible sin inventar un modelo ni respuestas.</p>
        {snapshot.requestedModels[0] && <div className="inference-request-state"><LoaderCircle className={snapshot.requestedModels[0].status === "active" ? "" : "spin"} /><span><strong>{snapshot.requestedModels[0].id}</strong>{snapshot.requestedModels[0].message}</span></div>}
        <div className="inference-unavailable-actions"><button className="primary-button" onClick={() => onNavigate("models")}><Boxes size={16} />Ver y activar modelos</button>{connectivityModel && <button className="secondary-button" disabled={diagnostic === "running"} onClick={() => void checkConnection()}>{diagnostic === "running" ? <LoaderCircle className="spin" size={16} /> : diagnostic === "ok" ? <CheckCircle2 size={16} /> : <Wifi size={16} />}{diagnostic === "idle" ? "Comprobar conexión" : diagnostic === "running" ? "Comprobando…" : diagnostic === "ok" ? "Conexión correcta" : "Reintentar conexión"}</button>}</div>
        {diagnostic === "failed" && <div className="inference-diagnostic-error"><CircleAlert size={15} />El coordinador no ha completado la prueba de conexión.</div>}
      </div>
    </div> : <div className="inference-console">
      <div className="inference-toolbar">
        <label><span>MODELO REAL</span><AppSelect ariaLabel="Modelo real" value={selectedModel} onChange={setModel} options={realModels.map((item) => ({ value: item.id, label: item.id }))} /></label>
        <div className="inference-model-summary"><span className="inference-live"><i />DISPONIBLE</span><span>{selectedOption?.routeLabel}</span><span>{selectedOption?.freeSlots ?? 0} hueco{selectedOption?.freeSlots === 1 ? "" : "s"} libre{selectedOption?.freeSlots === 1 ? "" : "s"}</span></div>
      </div>
      <div className="inference-output" aria-live="polite" ref={outputRef}>
        {turns.length === 0 && !pendingTurn && !error && <div className="inference-welcome"><Sparkles /><h2>Escribe una pregunta</h2><p>La respuesta vendrá del modelo seleccionado, no del adaptador de conectividad.</p><div className="inference-suggestions">{["Resume cómo funciona esta red", "Explica una idea en tres frases", "Responde con una prueba corta"].map((suggestion) => <button key={suggestion} onClick={() => setPrompt(suggestion)}>{suggestion}</button>)}</div></div>}
        {turns.map((turn) => <InferenceCompletedTurn turn={turn} key={turn.id} />)}
        {pendingTurn && <InferenceStreamingTurn turn={pendingTurn} />}
        {error && <div className="inference-error"><CircleAlert /><div><strong>No se pudo completar la inferencia</strong><span>{friendlyInferenceError(error)}</span></div></div>}
      </div>
      <div className="inference-input">
        <textarea aria-label="Mensaje" value={prompt} onChange={(event) => setPrompt(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void send(); } }} placeholder={`Escribe a ${selectedModel}…`} />
        <div className="inference-input-foot"><span><kbd>Enter</kbd> enviar · <kbd>Shift</kbd> + <kbd>Enter</kbd> nueva línea</span><div>{turns.length > 0 && <button className="inference-clear" onClick={() => { setTurns([]); setError(null); }}><Trash2 size={14} />Limpiar</button>}<button className="inference-send" disabled={!prompt.trim() || pendingTurn !== null} onClick={() => void send()}>{pendingTurn ? <LoaderCircle className="spin" /> : <Send />}<span>Enviar</span></button></div></div>
      </div>
    </div>}
  </section>;
}

function InferenceStreamingTurn({ turn }: { turn: InferencePendingTurn }) {
  const content = splitStreamingContent(turn.text);
  return <div className="inference-turn pending">
    <div className="inference-user-message"><span>TÚ</span><p>{turn.prompt}</p></div>
    {turn.text ? <div className="inference-message streaming"><img src={brandIcon} alt="" /><div>
      <div className="inference-stream-head"><span>{turn.model}</span><b><i />GENERANDO EN VIVO</b></div>
      {content.reasoning !== null && <div className="inference-live-reasoning"><span>RAZONAMIENTO</span><p>{content.reasoning}{content.answer === "" && <i className="stream-cursor" />}</p></div>}
      {content.answer && <p>{content.answer}<i className="stream-cursor" /></p>}
      <div className="inference-response-metrics live"><span><b>{formatDuration(turn.ttftMs)}</b>primer token</span><span><b>{formatDuration(turn.elapsedMs)}</b>tiempo actual</span><span><b>{turn.outputTokens}</b>tokens recibidos</span><span><b>{formatLiveThroughput(turn)}</b>tokens/s ahora</span><span><b>{turn.routeClass}</b>ruta</span></div>
    </div></div> : <div className="inference-thinking"><LoaderCircle className="spin" /><span>Esperando el primer token del modelo…</span></div>}
  </div>;
}

function InferenceCompletedTurn({ turn }: { turn: InferenceTurn }) {
  const content = splitThinkingContent(turn.response.text);
  return <div className="inference-turn">
    <div className="inference-user-message"><span>TÚ</span><p>{turn.prompt}</p></div>
    <div className="inference-message"><img src={brandIcon} alt="" /><div><span>{turn.response.model}</span>{content.reasoning && <details className="inference-reasoning"><summary>Ver razonamiento del modelo</summary><p>{content.reasoning}</p></details>}<p>{content.answer}</p><div className="inference-response-metrics"><span><b>{formatDuration(turn.response.ttftMs)}</b>primer token</span><span><b>{formatDuration(turn.response.activeMs)}</b>tiempo total</span><span><b>{turn.response.outputTokens}</b>tokens salida</span><span><b>{formatResponseThroughput(turn.response)}</b>tokens/s</span><span><b>{turn.response.routeClass}</b>ruta</span></div></div></div>
  </div>;
}

function JoinNetwork({ publicLink, external }: { publicLink: (path: string) => string; external: boolean }) {
  return <section><PageTitle eyebrow="ZERO-CONFIG JOIN" title="Add this device" copy="No account, invitation or terminal. Choose the fastest option for this device." />
    <div className="join-grid"><a className="join-card featured" href={publicLink("/mobile/")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><div><Smartphone /><span>INSTANT</span></div><h2>Use this browser</h2><p>Works on Android, iPhone, tablets and desktop browsers. Keep the page visible while contributing.</p><ul><li><CheckCircle2 />WebGPU when available</li><li><CheckCircle2 />CPU fallback everywhere</li><li><CheckCircle2 />Nothing to install</li></ul><strong>Open browser worker <ArrowRight /></strong></a>
      <a className="join-card" href={publicLink("/downloads")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><div><Laptop /><span>WINDOWS · MACOS</span></div><h2>Install the desktop agent</h2><p>Runs in the background and can expose native GPU runtimes such as local model runtime.</p><ul><li><CheckCircle2 />Windows 10/11</li><li><CheckCircle2 />Apple Silicon and Intel Macs</li><li><CheckCircle2 />Background contribution</li></ul><strong>Choose your computer <Download /></strong></a></div>
  </section>;
}

function Downloads({ publicLink, external }: { publicLink: (path: string) => string; external: boolean }) {
  return <section><PageTitle eyebrow="CLIENTS" title="Downloads" copy="Install the native client or enter immediately through the universal browser worker." />
    <div className="download-grid"><a className="download-card ready" href={publicLink("/downloads/windows")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Laptop />WINDOWS</span><h2>Windows 10/11</h2><p>x64 installer · Electron desktop agent with automatic updates</p><strong>Download installer <Download /></strong></a><a className="download-card ready" href={publicLink("/downloads/macos-arm64")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Cpu />MACOS · APPLE SILICON</span><h2>Mac M1–M4</h2><p>Unsigned test DMG · Apple Silicon arm64</p><strong>Download DMG <Download /></strong></a><a className="download-card ready" href={publicLink("/downloads/macos-x64")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Cpu />MACOS · INTEL</span><h2>Mac Intel</h2><p>Unsigned test DMG · Intel x64</p><strong>Download DMG <Download /></strong></a><a className="download-card ready" href={publicLink("/downloads/linux-deb")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Cpu />LINUX · DEBIAN</span><h2>Ubuntu / Debian</h2><p>x64 native desktop agent · DEB package</p><strong>Download DEB <Download /></strong></a><a className="download-card ready" href={publicLink("/downloads/linux-rpm")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Cpu />LINUX · RPM</span><h2>Fedora / RHEL</h2><p>x64 native desktop agent · RPM package</p><strong>Download RPM <Download /></strong></a><a className="download-card ready" href={publicLink("/mobile/")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Globe2 />UNIVERSAL</span><h2>Browser worker</h2><p>Android, iOS, macOS, Linux and Windows</p><strong>Open now <ExternalLink /></strong></a></div>
  </section>;
}

function Machine({ snapshot, bridge, onSnapshot }: { snapshot: DashboardSnapshot; bridge: DesktopBridge; onSnapshot: (snapshot: DashboardSnapshot) => void }) {
  const [busy, setBusy] = useState(false);
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
  return <section className="content-page">
    <PageTitle eyebrow="NATIVE NODE" title="This device" copy="Hardware detected on this computer and the resources offered to mycellios." actions={<button disabled={busy} onClick={() => void toggleContribution()}><CirclePower size={16} />{snapshot.settings.contributionEnabled ? "Pause contribution" : "Start contributing"}</button>} />
    <div className="hardware-hero">
      <div className="device-orb"><Cpu size={42} /></div>
      <div><span className="eyebrow">{snapshot.platform.toUpperCase()}</span><h2>{snapshot.localHardware.hostname}</h2><p>{snapshot.localHardware.gpus.length} accelerator(s) · {formatMemory(snapshot.localHardware.ramMb)} RAM</p></div>
      <span className={`state-badge ${snapshot.contribution.state === "connected" ? "online" : "paused"}`}>{contributionLabel}</span>
    </div>
    <div className="cards-grid">
      {snapshot.localHardware.gpus.map((gpu) => <article className="hardware-card" key={gpu.id}><div className="hardware-card-icon"><Gauge size={23} /></div><span className="card-kicker">{gpu.vendor}</span><h3>{gpu.model}</h3><div className="capacity-line"><span style={{ width: `${Math.min(100, snapshot.settings.offeredVramMb / Math.max(1, gpu.physicalVramMb + (gpu.sharedMemoryMb ?? 0)) * 100)}%` }} /></div><div className="hardware-metrics"><DesktopMetric label="Physical VRAM" value={formatMemory(gpu.physicalVramMb)} /><DesktopMetric label="Shared" value={formatMemory(gpu.sharedMemoryMb ?? 0)} /><DesktopMetric label="Utilization" value={`${gpu.utilizationPct ?? 0}%`} /><DesktopMetric label="Temperature" value={gpu.temperatureC === undefined ? "—" : `${gpu.temperatureC} °C`} /></div></article>)}
      {snapshot.localHardware.gpus.length === 0 && <div className="wide-empty"><Empty icon={Cpu} title="No GPU detected" copy="This machine can register its real hardware, but it will not advertise a model until a compatible runtime is active." /></div>}
    </div>
  </section>;
}

function DesktopSettingsView({ snapshot, bridge, onSnapshot }: { snapshot: DashboardSnapshot; bridge: DesktopBridge; onSnapshot: (snapshot: DashboardSnapshot) => void }) {
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
    <PageTitle eyebrow="DESKTOP PREFERENCES" title="Settings" copy="Native connection, contribution, background behavior and updates for this computer." />
    {error && <div className="inline-error"><CircleAlert size={17} />{error}</div>}
    <div className="settings-section"><div><Globe2 size={20} /><div><h3>Coordinator</h3><p>Use the public network or host an isolated local coordinator.</p></div></div><div className="settings-fields"><label>Mode<AppSelect ariaLabel="Coordinator mode" value={draft.coordinatorMode} onChange={(value) => updateCoordinatorMode(value as DesktopSettings["coordinatorMode"])} options={[{ value: "remote", label: "Public mycellios network" }, { value: "local", label: "Local network on this machine" }]} /></label>{draft.coordinatorMode === "remote" && <label>Coordinator URL<input value={draft.remoteCoordinatorUrl} onChange={(event) => update("remoteCoordinatorUrl", event.target.value)} /></label>}<label>Region<input value={draft.region} onChange={(event) => update("region", event.target.value)} placeholder="auto" /></label></div></div>
    <div className="settings-section"><div><Gauge size={20} /><div><h3>Contribution</h3><p>Register real hardware only, or expose a model through a verified local runtime.</p></div></div><div className="settings-fields"><label>Runtime<AppSelect ariaLabel="Contribution runtime" value={draft.adapterMode} onChange={(value) => update("adapterMode", value as DesktopSettings["adapterMode"])} options={[{ value: "connectivity-test", label: "Hardware only (no model)" }, { value: "local-model-runtime", label: "Local local model runtime" }]} /></label><label>Offered VRAM (MB)<input type="number" min="512" step="256" value={draft.offeredVramMb} onChange={(event) => update("offeredVramMb", Number(event.target.value))} /></label>{draft.adapterMode === "local-model-runtime" && <><label>local model runtime model<input value={draft.modelName} onChange={(event) => update("modelName", event.target.value)} /></label><label>local model runtime URL<input value={draft.adapterBaseUrl} onChange={(event) => update("adapterBaseUrl", event.target.value)} /></label><label>Pinned digest<input value={draft.modelDigest} onChange={(event) => update("modelDigest", event.target.value)} placeholder="sha256:…" /></label></>}</div></div>
    <div className="settings-section compact-settings"><div><SlidersHorizontal size={20} /><div><h3>Application</h3><p>Startup and background contribution.</p></div></div><div className="toggle-list"><Toggle label="Start with the system" checked={draft.launchAtLogin} onChange={(value) => update("launchAtLogin", value)} /><Toggle label="Keep running in the tray" checked={draft.closeToTray} onChange={(value) => update("closeToTray", value)} /><Toggle label="Contribute resources" checked={draft.contributionEnabled} onChange={(value) => update("contributionEnabled", value)} /></div></div>
    <div className="settings-section"><div><Download size={20} /><div><h3>Automatic updates</h3><p>Desktop releases are checked and downloaded in the background when supported.</p></div></div><div className="update-settings"><div><span className={`update-state ${snapshot.update.state}`}>{updateStateLabel(snapshot.update.state)}</span><strong>Version {snapshot.appVersion}</strong><p>{snapshot.update.message}</p></div><div className="update-actions"><button className="secondary-button" disabled={checkingUpdate || snapshot.update.state === "checking" || snapshot.update.state === "downloading"} onClick={() => void checkUpdate()}><RefreshCw className={checkingUpdate ? "spin" : ""} size={15} />Check now</button>{snapshot.update.state === "ready" && <button className="primary-button" onClick={() => void bridge.installUpdate()}><Download size={15} />Restart and update</button>}</div></div></div>
    <div className="settings-footer"><button className="primary-button" disabled={saving} onClick={() => void save()}>{saving ? <LoaderCircle className="spin" size={17} /> : <Check size={17} />}Save and reconnect</button></div>
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

function AppSelect({ ariaLabel, value, options, onChange }: { ariaLabel: string; value: string; options: AppSelectOption[]; onChange: (value: string) => void }) {
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
      <span>{selected?.label ?? "Selecciona una opción"}</span><ChevronDown size={16} />
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

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) { return <label className="toggle-row"><span>{label}</span><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><i /></label>; }
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

function PageTitle({ eyebrow, title, copy, actions }: { eyebrow: string; title: string; copy: string; actions?: React.ReactNode }) {
  return <div className="panel-page-title"><div><span>{eyebrow}</span><h1>{title}</h1><p>{copy}</p></div>{actions && <div className="page-actions">{actions}</div>}</div>;
}

function Stat({ icon: Icon, label, value, detail, tone = "blue" }: { icon: typeof Radio; label: string; value: string; detail: string; tone?: "blue" | "purple" }) { return <article className={`panel-stat ${tone}`}><div><Icon /></div><span>{label}</span><strong>{value}</strong><small>{detail}</small><i className="stat-accent" /></article>; }
function Metric({ label, value }: { label: string; value: string }) { return <div><span>{label}</span><strong>{value}</strong></div>; }
function Empty({ icon: Icon, title, copy }: { icon: typeof Activity; title: string; copy: string }) { return <div className="panel-empty"><Icon /><strong>{title}</strong><p>{copy}</p></div>; }
function PanelLoading() { return <div className="panel-loading"><LoaderCircle className="spin" /><strong>Connecting to mycellios</strong><span>Loading the public network snapshot…</span></div>; }
function workerLabel(worker: PublicWorker) { return worker.kind === "browser" ? `Browser ${shortId(worker.id)}` : worker.gpus[0]?.model || `Desktop ${shortId(worker.id)}`; }
function nodeVisualState(worker: PublicWorker): "active" | "warning" | "offline" { return worker.connected && worker.status === "online" ? "active" : worker.connected || worker.status === "suspect" || worker.status === "draining" ? "warning" : "offline"; }
function nodeStateOrder(worker: PublicWorker): number { return nodeVisualState(worker) === "active" ? 0 : nodeVisualState(worker) === "warning" ? 1 : 2; }
function nodeStatusLabel(worker: PublicWorker): string {
  if (worker.connected && worker.status === "online") return "Activo";
  if (worker.status === "suspect") return "Inestable";
  if (worker.status === "draining") return "Finalizando";
  if (worker.connected) return "Conectado";
  return "Inactivo";
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
function formatCompactNumber(value: number): string { return new Intl.NumberFormat("es-ES", { maximumFractionDigits: value < 10 ? 1 : 0 }).format(value); }
function formatPower(watts: number): string { return watts >= 1_000 ? `${(watts / 1_000).toFixed(1)} kW` : `${Math.round(watts)} W`; }
function shortId(value: string) { return value.length <= 10 ? value : `${value.slice(0, 6)}…${value.slice(-4)}`; }
function formatMemory(value: number) { return value >= 1_024 ? `${(value / 1_024).toFixed(value >= 10_240 ? 0 : 1)} GB` : `${Math.round(value)} MB`; }
function relativeTime(value: string) { const seconds = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 1_000)); if (seconds < 5) return "now"; if (seconds < 60) return `${seconds}s ago`; const minutes = Math.floor(seconds / 60); if (minutes < 60) return `${minutes}m ago`; return `${Math.floor(minutes / 60)}h ago`; }
function relativeTimeEs(value: string) { const seconds = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 1_000)); if (seconds < 5) return "ahora"; if (seconds < 60) return `hace ${seconds} s`; const minutes = Math.floor(seconds / 60); if (minutes < 60) return `hace ${minutes} min`; return `hace ${Math.floor(minutes / 60)} h`; }

function inferenceModelOption(snapshot: PublicSnapshot, model: PublicSnapshot["models"][number]) {
  const deployments = snapshot.workers.flatMap((worker) => worker.deployments).filter((deployment) => deployment.model === model.id);
  const adapters = deployments.map((deployment) => deployment.adapter).filter((adapter): adapter is NonNullable<PublicDeployment["adapter"]> => adapter !== undefined);
  const connectivityOnly = model.id === "mycellios-connectivity-check" || (adapters.length > 0 && adapters.every((adapter) => adapter === "mock"));
  const freeSlots = deployments.reduce((total, deployment) => total + deployment.freeSlots, 0);
  const routeLabel = model.pipelines > 0
    ? `${model.pipelines} pipeline${model.pipelines === 1 ? "" : "s"}`
    : `${model.replicas} réplica${model.replicas === 1 ? "" : "s"}`;
  return { ...model, connectivityOnly, freeSlots, routeLabel };
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
  return { reasoning, answer: answer || "El modelo no devolvió contenido final." };
}

function friendlyInferenceError(message: string): string {
  const normalized = message.toLowerCase();
  if (normalized.includes("no_candidates") || normalized.includes("no candidate") || normalized.includes("unavailable")) return "El modelo dejó de estar disponible. Comprueba el estado de sus nodos y vuelve a intentarlo.";
  if (normalized.includes("timeout") || normalized.includes("deadline")) return "La red tardó demasiado en responder. La petición se ha cancelado sin inventar una respuesta.";
  if (normalized.includes("401") || normalized.includes("token")) return "La red requiere una credencial válida para usar este modelo.";
  return message;
}

async function fetchBenchmarkRuns(): Promise<BenchmarkRun[]> {
  const response = await fetch("/local/v1/benchmarks", { cache: "no-store" });
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

function formatBenchmark(value: number | null, suffix: string): string {
  if (value === null || !Number.isFinite(value)) return "—";
  return `${value.toLocaleString("en-US", { maximumFractionDigits: 2 })}${suffix}`;
}

function formatBenchmarkDate(value: string): string {
  return new Date(value).toLocaleString("es-ES", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
}

function benchmarkStatusLabel(status: BenchmarkRun["status"]): string {
  if (status === "baseline") return "Baseline";
  if (status === "passed") return "Improved / stable";
  if (status === "regression") return "Regression";
  return "Failed";
}

export { Panel };
