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
  Coins,
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
  Milestone,
  Minus,
  Network,
  Play,
  Plus,
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
  DashboardDeployment,
  ChatResponse,
  ChatStreamUpdate,
  DesktopAccelerationStatus,
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
import { isAdvertisedGpuSelected } from "./panel-hardware";
import "./panel.css";

const PUBLIC_COORDINATOR_URL = "https://www.mycellios.com";
import "./panel-downloads.css";
import "./panel-desktop.css";

type PanelView = "overview" | "nodes" | "models" | "jobs" | "tests" | "roadmap" | "inference" | "contribute" | "join" | "downloads" | "machine" | "settings";

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
  { id: "roadmap", label: "Roadmap", icon: Milestone },
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
  const [contentScale, setContentScale] = useState(() => {
    const saved = Number(window.localStorage.getItem("mycellios.content-scale"));
    return Number.isFinite(saved) && saved >= 0.9 && saved <= 1.3 ? saved : 1;
  });
  const [modelAdminToken, setModelAdminToken] = useState(() =>
    window.sessionStorage.getItem("mycellios-model-admin-token") ?? "",
  );
  const requiresModelAdminToken = desktop
    ? desktopSnapshot?.settings.coordinatorMode !== "local" && !desktopSnapshot?.modelAdminAuthorization.configured
    : !localBrowser;
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

  function changeContentScale(delta: number) {
    setContentScale((current) => {
      const next = Math.min(1.3, Math.max(0.9, Math.round((current + delta) * 10) / 10));
      window.localStorage.setItem("mycellios.content-scale", String(next));
      return next;
    });
  }

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
    const token = adminToken.trim();
    if (requiresModelAdminToken && !token) throw new Error("Enter the network administrator token.");
    if (desktopBridge) {
      applyDesktopSnapshot(await desktopBridge.removeRequestedModel(modelId, token));
      return;
    }
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
            <div className="panel-zoom-controls" role="group" aria-label="Interface zoom">
              <button type="button" aria-label="Reduce interface size" disabled={contentScale <= 0.9} onClick={() => changeContentScale(-0.1)}><Minus size={14} /></button>
              <output aria-live="polite">{Math.round(contentScale * 100)}%</output>
              <button type="button" aria-label="Increase interface size" disabled={contentScale >= 1.3} onClick={() => changeContentScale(0.1)}><Plus size={14} /></button>
            </div>
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
          <div className="panel-content-scale" style={{ transform: `scale(${contentScale})`, width: `${100 / contentScale}%` } as CSSProperties}>
            {loading && snapshot.capturedAt === EMPTY.capturedAt ? <PanelLoading /> : (
              <>
              {view === "overview" && <Overview snapshot={snapshot} onNavigate={navigate} publicLink={publicLink} external={desktop} localAcceleration={desktopSnapshot?.acceleration} localContributionState={desktopSnapshot?.contribution.state} />}
              {view === "nodes" && <Nodes snapshot={snapshot} onRemove={removeWorker} onClearOffline={clearOfflineWorkers} />}
              {view === "models" && <Models snapshot={snapshot} onSearch={searchHubModels} onRequest={requestModel} onRemove={removeRequestedModel} adminToken={modelAdminToken} requiresAdminToken={requiresModelAdminToken} secureTokenStorage={desktop} />}
              {view === "jobs" && <Jobs snapshot={snapshot} />}
              {view === "tests" && <Tests bridge={desktopBridge} />}
              {view === "roadmap" && <Roadmap />}
              {view === "inference" && <Inference snapshot={snapshot} onSend={sendPrompt} onNavigate={navigate} />}
              {view === "contribute" && <Contribute />}
              {view === "join" && <JoinNetwork publicLink={publicLink} external={desktop} />}
              {view === "downloads" && <Downloads publicLink={publicLink} external={desktop} />}
              {view === "machine" && desktopSnapshot && desktopBridge && <Machine snapshot={desktopSnapshot} bridge={desktopBridge} onSnapshot={applyDesktopSnapshot} />}
              {view === "settings" && desktopSnapshot && desktopBridge && <DesktopSettingsView snapshot={desktopSnapshot} bridge={desktopBridge} onSnapshot={applyDesktopSnapshot} />}
              </>
            )}
          </div>
        </main>
      </div>
      {desktopSnapshot && desktopBridge && !desktopSnapshot.settings.onboardingComplete && <DesktopOnboarding snapshot={desktopSnapshot} bridge={desktopBridge} onSnapshot={applyDesktopSnapshot} />}
    </div>
  );
}

function Overview({ snapshot, onNavigate, publicLink, external, localAcceleration, localContributionState }: { snapshot: PublicSnapshot; onNavigate: (view: PanelView) => void; publicLink: (path: string) => string; external: boolean; localAcceleration: AcceleratorProgressSnapshot | undefined; localContributionState: DashboardSnapshot["contribution"]["state"] | undefined }) {
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
  const activeThroughput = activeDeployments.reduce((total, deployment) => total + deployment.tokensPerSecond, 0);
  const execution = summarizeDeploymentExecution(activeDeployments);
  const nativeNodes = operational.filter((worker) => worker.kind === "desktop").length;
  const browserNodes = operational.filter((worker) => worker.kind === "browser").length;
  const cellNodes = operational.filter((worker) => worker.kind === "cell").length;
  return (
    <section className="overview-page">
      <PageTitle eyebrow="NETWORK CONTROL" title="Overview" copy="A live view of the capacity, models and tasks connected to your mycellios network." actions={<><button onClick={() => onNavigate("nodes")}>Manage nodes</button><a href={publicLink("/mobile/")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}>Add this device <ArrowRight size={16} /></a></>} />
      {localAcceleration && shouldShowAccelerationBanner(localAcceleration) && <AcceleratorCompactBanner acceleration={localAcceleration} contributionState={localContributionState} onOpen={() => onNavigate("machine")} />}
      <article className="global-capacity-card">
        <div className="global-capacity-main">
          <div className="global-capacity-heading"><span>GLOBAL NODE CAPACITY</span><b><i /> LIVE</b></div>
          <div className="execution-overview"><ExecutionBadge execution={execution} large /><span><strong>Effective inference device</strong><small>Reported by the runtime after loading the model, never inferred from detected hardware.</small></span></div>
          <div className="global-capacity-total"><strong>{formatMemory(activeOfferedVramMb)}</strong><div><b>Active usable memory</b><small>Offered by {operational.length} operational node{operational.length === 1 ? "" : "s"}</small></div></div>
          <div className="global-capacity-bar" aria-label={`${Math.round(freeRatio * 100)}% of offered memory is free`}><i style={{ width: `${freeRatio * 100}%` }} /></div>
          <div className="global-capacity-legend"><span>{formatMemory(freeVramMb)} free now</span><span>{formatMemory(usedVramMb)} in use</span></div>
        </div>
        <div className="global-capacity-metrics">
          <CapacityMetric icon={Cpu} label="Runtime stages" value={execution.verified ? String(execution.unitCount) : "Unverified"} detail={execution.verified ? executionDetail(execution) : `${nativeNodes} native · ${browserNodes} browser · ${cellNodes} cells connected`} />
          <CapacityMetric icon={Gauge} label="Average GPU load" value={averageUtilization === null ? "No telemetry" : `${averageUtilization.toFixed(0)}%`} detail={`${telemetryGpus.length} reporting engine${telemetryGpus.length === 1 ? "" : "s"}`} />
          <CapacityMetric icon={Zap} label="Active throughput" value={activeThroughput > 0 ? `${formatCompactNumber(activeThroughput)} tok/s` : "No active runtime"} detail={`${activeDeployments.length} deployment${activeDeployments.length === 1 ? "" : "s"}`} />
          <CapacityMetric icon={Activity} label="Observed power" value={poweredGpus.length > 0 ? formatPower(observedPowerW) : "No telemetry"} detail={poweredGpus.length > 0 ? `${poweredGpus.length} physical reading${poweredGpus.length === 1 ? "" : "s"}` : "Never estimated"} />
        </div>
      </article>
      <div className="panel-stat-grid">
        <Stat icon={Radio} label="Active nodes" value={String(snapshot.summary.connected)} detail={`${snapshot.summary.registered} registered`} />
        <Stat icon={HardDrive} label="Shared VRAM" value={formatMemory(activeOfferedVramMb)} detail={`${formatMemory(freeVramMb)} currently free`} tone="purple" />
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
            <div><span>Shared VRAM</span><strong>{formatMemory(activeOfferedVramMb)}</strong></div>
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
      <PageTitle eyebrow="RED DISTRIBUIDA" title="Red de nodos" copy="Dispositivos físicos, navegadores activos y celdas de cómputo conectadas a mycellios." actions={<button disabled={busy !== null} onClick={() => void clearOffline()}><Trash2 size={15} /> Limpiar inactivos</button>} />

      <div className="node-overview-grid">
        <NodeOverviewStat icon={Server} label="Dispositivos activos" value={`${activePhysicalWorkers.length} / ${physicalWorkers.length}`} detail={`${connectedPhysicalWorkers.length} conectados ahora · ${cellWorkers.length} celdas`} progress={physicalWorkers.length > 0 ? activePhysicalWorkers.length / physicalWorkers.length : 0} tone="green" />
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
            ><div><WorkerKindIcon worker={worker} /><i /></div><span><strong>{label}</strong><small>{worker.region} · {nodeStatusLabel(worker)}</small></span></button>;
          })}
          {snapshot.workers.length === 0 && <div className="node-topology-empty"><Server /><strong>Aún no hay nodos</strong><span>Conecta una máquina para verla aparecer en la red.</span></div>}
        </div>
        {selectedWorker && <NodeTopologyInspector worker={selectedWorker} snapshot={snapshot} />}
      </section>

      <NodeInventorySection eyebrow="DISPOSITIVOS INSTALADOS" title="Equipos físicos" workers={installedWorkers} snapshot={snapshot} busy={busy} onRemove={remove} />
      {browserWorkers.length > 0 && <NodeInventorySection eyebrow="CAPACIDAD TEMPORAL" title="Navegadores activos" workers={browserWorkers} snapshot={snapshot} busy={busy} onRemove={remove} />}
      {cellWorkers.length > 0 && <NodeInventorySection eyebrow="EJECUTORES DISTRIBUIDOS" title="Celdas de cómputo" workers={cellWorkers} snapshot={snapshot} busy={busy} onRemove={remove} />}
      {snapshot.workers.length === 0 && <div className="node-card-grid"><div className="wide-empty"><Empty icon={Server} title="No hay nodos registrados" copy="Abre el worker móvil o instala mycellios para añadir la primera máquina." /><a href="/mobile/">Conectar un dispositivo <ArrowRight size={15} /></a></div></div>}
    </section>
  );
}

function NodeInventorySection({ eyebrow, title, workers, snapshot, busy, onRemove }: {
  eyebrow: string;
  title: string;
  workers: PublicWorker[];
  snapshot: PublicSnapshot;
  busy: string | null;
  onRemove: (workerId: string) => Promise<void>;
}) {
  if (workers.length === 0) return null;
  return <>
    <div className="node-list-heading"><div><span>{eyebrow}</span><h2>{title}</h2></div><strong>{workers.length} registrado{workers.length === 1 ? "" : "s"}</strong></div>
    <div className="node-card-grid">
      {workers.map((worker) => {
        const execution = workerExecutionSummary(snapshot, worker);
        return <article className="node-card" key={worker.id}>
        <div className="node-card-head"><div className={`node-device ${worker.kind}`}><WorkerKindIcon worker={worker} /></div><div><span>{workerKindLabel(worker)}</span><h2>{workerLabel(worker)}</h2><small>{shortId(worker.id)}</small></div><span className={`node-status ${worker.status}`}><i />{nodeStatusLabel(worker)}</span></div>
        <div className="node-runtime-state"><ExecutionBadge execution={execution} /><span><strong>{execution.verified ? executionDetail(execution) : "No active model is reporting execution on this node"}</strong><small>{execution.verified ? executionDeviceNames(execution) : "Detected GPU hardware is not counted as active compute."}</small></span></div>
        {execution.stages.length > 0 && <ExecutionStages execution={execution} compact />}
        <div className="node-card-metrics"><Metric label={worker.kind === "cell" ? "Capacidad" : "Hardware"} value={worker.gpus[0]?.model ?? "Desconocido"} /><Metric label="Ofrecido" value={formatMemory(worker.offeredVramMb)} /><Metric label="Región" value={worker.region} /><Metric label="Completadas" value={String(worker.jobsCompleted)} /></div>
        <div className="node-card-foot"><span>Visto {relativeTimeEs(worker.lastSeenAt)}</span><button disabled={busy === worker.id} onClick={() => void onRemove(worker.id)}>{busy === worker.id ? <LoaderCircle className="spin" /> : <Trash2 />} Eliminar</button></div>
      </article>;})}
    </div>
  </>;
}

function NodeOverviewStat({ icon: Icon, label, value, detail, progress, tone }: { icon: typeof Server; label: string; value: string; detail: string; progress?: number; tone: "green" | "blue" | "purple" | "violet" | "cyan" }) {
  return <article className={`node-overview-stat ${tone}`}><div className="node-overview-icon"><Icon /></div><span>{label}</span><strong>{value}</strong><small>{detail}</small>{progress !== undefined && <div className="node-overview-progress"><i style={{ width: `${Math.max(0, Math.min(1, progress)) * 100}%` }} /></div>}</article>;
}

function NodeTopologyInspector({ worker, snapshot }: { worker: PublicWorker; snapshot: PublicSnapshot }) {
  const gpus = worker.gpus;
  const free = gpus.reduce((total, gpu) => total + gpu.freeOfferedVramMb, 0);
  const utilizations = gpus.filter((gpu) => gpu.utilizationPct !== undefined);
  const utilization = utilizations.length > 0 ? utilizations.reduce((total, gpu) => total + (gpu.utilizationPct ?? 0), 0) / utilizations.length : null;
  const throughput = worker.deployments.reduce((total, deployment) => total + deployment.tokensPerSecond, 0);
  const execution = workerExecutionSummary(snapshot, worker);
  return <div className="node-topology-inspector">
    <div className={`node-inspector-identity ${nodeVisualState(worker)}`}><div><WorkerKindIcon worker={worker} /></div><span><small>{worker.kind === "cell" ? "CELDA SELECCIONADA" : "NODO SELECCIONADO"}</small><strong>{workerLabel(worker)}</strong><em>{worker.region} · {shortId(worker.id)}</em><ExecutionBadge execution={execution} /></span></div>
    <div className="node-inspector-metrics"><Metric label="Estado" value={nodeStatusLabel(worker)} /><Metric label="GPU libre" value={formatMemory(free)} /><Metric label="Carga física" value={utilization === null ? "Sin datos" : `${utilization.toFixed(0)}%`} /><Metric label="Rend. anunciado" value={throughput > 0 ? `${formatCompactNumber(throughput)} tok/s` : "—"} /><Metric label="Cómputo real" value={execution.verified ? executionShortLabel(execution) : "No verificado"} /><Metric label="Modelos" value={String(worker.deployments.length)} /></div>
  </div>;
}

function Models({ snapshot, onSearch, onRequest, onRemove, adminToken: initialAdminToken, requiresAdminToken, secureTokenStorage }: {
  snapshot: PublicSnapshot;
  onSearch: (query: string) => Promise<HubCatalogModel[]>;
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
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [catalogOpen, setCatalogOpen] = useState(true);
  const searchSequence = useRef(0);
  const selectedCatalogModel = catalogModels.find((model) => model.id === source) ?? null;

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
      setFormError(friendlyModelMutationError(error));
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
        {catalogModels.length > 0 && <div className="hub-model-grid">{catalogModels.map((model) => {
          const estimatedMemoryMiB = estimatedHubMemoryMiB(model);
          return <button type="button" key={model.id} className={`hub-model-card ${source === model.id ? "selected" : ""} ${model.compatible ? "compatible" : "unsupported"}`} aria-disabled={!model.compatible} onClick={() => selectCatalogModel(model)} title={model.compatibilityReason ?? `Select ${model.id}`}>
            <span className="hub-model-card-head"><i><Boxes /></i><strong>{model.id}</strong>{source === model.id && <CheckCircle2 />}</span>
            <small>{model.architecture ?? model.modelType ?? "Architecture unavailable"}</small>
            <span className="hub-model-card-memory"><MemoryStick /><span><b>{estimatedMemoryMiB === null ? "Calculating…" : `≈ ${formatMemory(estimatedMemoryMiB)}`}</b><em>BF16 · 4K context</em></span></span>
            <span className="hub-model-card-meta"><b>{model.compatible ? "CERTIFIED" : model.gated ? "GATED" : "NOT SUPPORTED"}</b><em>{formatHubCount(model.downloads)} downloads</em></span>
          </button>;
        })}</div>}
      </>}

      {source && <div className="hub-selected-model"><CheckCircle2 /><span><small>SELECTED MODEL</small><strong>{source}</strong><em>It will be published as {modelId}{selectedCatalogModel && estimatedHubMemoryMiB(selectedCatalogModel) !== null ? ` · ≈ ${formatMemory(estimatedHubMemoryMiB(selectedCatalogModel)!)} required` : ""}</em></span><button type="button" onClick={() => setCatalogOpen(true)}>Change model</button></div>}

      {requiresAdminToken && <label className="model-admin-auth"><ShieldCheck /><span><strong>Network administrator authorization</strong><small>Required because preparing a model changes the shared network. {secureTokenStorage ? "Saved with encrypted operating-system storage after successful authorization." : "Kept only for this browser session."}</small></span><input type="password" autoComplete="off" spellCheck={false} value={adminToken} onChange={(event) => { setAdminToken(event.target.value); setFormError(null); }} placeholder="Administrator token" aria-label="Network administrator token" /></label>}

      <details className="model-advanced">
        <summary><SlidersHorizontal size={15} /> Advanced settings or manual model ID <ChevronDown size={15} /></summary>
        <div className="model-form-grid">
          <label>HUGGING FACE MODEL<input value={source} onChange={(event) => { setSource(event.target.value); setModelId(networkNameForHubModel(event.target.value)); }} placeholder="Qwen/Qwen3-0.6B" required /></label>
          <label>NETWORK NAME <small>automatic</small><input value={modelId} onChange={(event) => setModelId(event.target.value)} placeholder="qwen3-0.6b" required pattern="[A-Za-z0-9][A-Za-z0-9._-]*" /></label>
          <label>REVISION <small>optional</small><input value={revision} onChange={(event) => setRevision(event.target.value)} placeholder="main or commit" /></label>
          <label>CONTEXT TOKENS<input type="number" min={128} max={1048576} value={contextTokens} onChange={(event) => setContextTokens(Number(event.target.value))} /></label>
          <label>MINIMUM NODES<AppSelect ariaLabel="Minimum nodes" value={String(minimumNodes)} onChange={(value) => setMinimumNodes(Number(value))} options={[2, 3, 4, 5, 6, 7, 8].map((count) => ({ value: String(count), label: String(count) }))} /></label>
        </div>
      </details>
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
        return <div className="panel-table-row model-execution-row" key={model.id}><strong><Boxes size={17} />{model.id}</strong><span>{model.replicas}</span><span>{model.pipelines}</span><ExecutionBadge execution={execution} /></div>;
      })}
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

function Roadmap() {
  return <section className="roadmap-page">
    <PageTitle eyebrow="FUTURE DIRECTION" title="Roadmap" copy="Dos iniciativas para que mycellios aprenda de la red y recompense el cómputo que realmente aporta valor." />
    <div className="roadmap-status-banner">
      <div><Milestone size={20} /><span><small>ESTADO ACTUAL</small><strong>Diseño · todavía no construido</strong></span></div>
      <p>La hoja de ruta marca la dirección del producto. Cada fase deberá demostrar seguridad, utilidad y resultados medibles antes de activarse en la red.</p>
    </div>
    <div className="roadmap-grid">
      <article className="roadmap-card improvement">
        <div className="roadmap-card-head">
          <span className="roadmap-icon"><Sparkles /></span>
          <span className="roadmap-phase"><i /> INICIATIVA 01</span>
        </div>
        <div className="roadmap-metric"><strong>0,1</strong><span>%<small>del cómputo útil</small></span></div>
        <span className="roadmap-kicker">AUTOMEJORA CONTINUA</span>
        <h2>Mejorar mycellios cada día</h2>
        <p>Una pequeña parte de la capacidad agregada se reservará para que agentes autónomos analicen la red, experimenten y propongan mejoras verificables.</p>
        <ol className="roadmap-steps">
          <li><b>01</b><span><strong>Observar</strong><small>Detectar fallos, regresiones y cuellos de botella reales.</small></span></li>
          <li><b>02</b><span><strong>Experimentar</strong><small>Probar cambios en entornos aislados y reproducibles.</small></span></li>
          <li><b>03</b><span><strong>Validar</strong><small>Conservar solo mejoras que superen tests y métricas.</small></span></li>
        </ol>
        <div className="roadmap-card-foot"><ShieldCheck size={16} /><span><strong>Con límites y trazabilidad</strong>El presupuesto no excederá el 0,1 % ni interferirá con las cargas de los usuarios.</span></div>
      </article>

      <article className="roadmap-card crypto">
        <div className="roadmap-card-head">
          <span className="roadmap-icon"><Coins /></span>
          <span className="roadmap-phase"><i /> INICIATIVA 02</span>
        </div>
        <div className="roadmap-metric word"><strong>CRYPTO</strong><span><small>incentivos de red</small></span></div>
        <span className="roadmap-kicker">TOKEN DE CÓMPUTO</span>
        <h2>Recompensar el trabajo útil</h2>
        <p>Un sistema de tokens reconocerá la computación aceptada por la red. Estar conectado no bastará: la recompensa dependerá de trabajo útil, calidad y fiabilidad.</p>
        <ol className="roadmap-steps">
          <li><b>01</b><span><strong>Créditos internos</strong><small>Contabilidad verificable, aún sin blockchain ni valor económico.</small></span></li>
          <li><b>02</b><span><strong>Testnet</strong><small>Pruebas anti-Sybil, antifraude y contratos auditables.</small></span></li>
          <li><b>03</b><span><strong>Lanzamiento evaluado</strong><small>Economía, seguridad y revisión legal antes de emitir.</small></span></li>
        </ol>
        <div className="roadmap-card-foot"><ShieldCheck size={16} /><span><strong>Sin promesas financieras</strong>No existe un token activo; primero deben validarse el modelo y su cumplimiento legal.</span></div>
      </article>
    </div>
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
        <div className="inference-model-summary"><ExecutionBadge execution={selectedOption?.execution ?? unknownExecutionSummary()} large /><span className="inference-live"><i />DISPONIBLE</span><span>{selectedOption?.routeLabel}</span><span>{selectedOption?.freeSlots ?? 0} hueco{selectedOption?.freeSlots === 1 ? "" : "s"} libre{selectedOption?.freeSlots === 1 ? "" : "s"}</span></div>
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
  return <section><PageTitle eyebrow="ZERO-CONFIG JOIN" title="Add this device" copy="No account, invitation or terminal. Choose an option, review the detected hardware and confirm participation." />
    <div className="join-grid"><a className="join-card featured" href={publicLink("/mobile/?autostart=1")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><div><Smartphone /><span>INSTANT</span></div><h2>Use this browser</h2><p>Works on modern Android, iPhone, tablet and desktop browsers. Keep the page visible while contributing.</p><ul><li><CheckCircle2 />WebGPU when available</li><li><CheckCircle2 />Automatic CPU fallback</li><li><CheckCircle2 />One explicit confirmation</li></ul><strong>Open and confirm <ArrowRight /></strong></a>
      <a className="join-card" href={publicLink("/downloads")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><div><Laptop /><span>WINDOWS · MACOS · LINUX</span></div><h2>Install the desktop agent</h2><p>Detects the hardware, prepares the certified runtime automatically and contributes on CPU while GPU setup finishes.</p><ul><li><CheckCircle2 />Windows CUDA/ROCm and Apple Silicon MPS when certified</li><li><CheckCircle2 />Automatic verified downloads</li><li><CheckCircle2 />Safe CPU fallback</li></ul><strong>Choose your computer <Download /></strong></a></div>
  </section>;
}

function Downloads({ publicLink, external }: { publicLink: (path: string) => string; external: boolean }) {
  return <section><PageTitle eyebrow="CLIENTS" title="Downloads" copy="Install the native client or enter immediately through the universal browser worker." />
    <div className="download-grid"><a className="download-card ready" href={publicLink("/downloads/windows")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Laptop />WINDOWS</span><h2>Windows 10/11</h2><p>x64 installer · automatic CPU runtime, CUDA verification and ROCm on certified Windows 11 hardware</p><strong>Download installer <Download /></strong></a><a className="download-card ready" href={publicLink("/downloads/macos-arm64")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Cpu />MACOS · APPLE SILICON</span><h2>Mac M1 or newer</h2><p>macOS 14+ arm64 DMG · bundled PyTorch runtime with automatic Metal / MPS verification</p><strong>Download DMG <Download /></strong></a><a className="download-card ready" href={publicLink("/downloads/macos-x64")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Cpu />MACOS · INTEL</span><h2>Mac Intel</h2><p>x64 DMG · self-contained CPU runtime; WebGPU remains available in the browser</p><strong>Download DMG <Download /></strong></a><a className="download-card ready" href={publicLink("/downloads/linux-deb")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Cpu />LINUX · DEBIAN</span><h2>Ubuntu / Debian</h2><p>x64 DEB · automatic certified CPU runtime; compatible GPUs can contribute through the browser worker</p><strong>Download DEB <Download /></strong></a><a className="download-card ready" href={publicLink("/downloads/linux-rpm")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Cpu />LINUX · RPM</span><h2>Fedora / RHEL</h2><p>x64 RPM · automatic certified CPU runtime; compatible GPUs can contribute through the browser worker</p><strong>Download RPM <Download /></strong></a><a className="download-card ready" href={publicLink("/mobile/?autostart=1")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Globe2 />UNIVERSAL</span><h2>Browser worker</h2><p>Android, iOS and desktops · real WebGPU probe with automatic CPU fallback after confirmation</p><strong>Open and confirm <ExternalLink /></strong></a></div>
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
  return <section className="content-page">
    <PageTitle eyebrow="NATIVE NODE" title="This device" copy="Hardware detected on this computer and the resources offered to mycellios." actions={<button disabled={busy} onClick={() => void toggleContribution()}><CirclePower size={16} />{snapshot.settings.contributionEnabled ? "Pause contribution" : "Start contributing"}</button>} />
    <div className="hardware-hero">
      <div className="device-orb"><Cpu size={42} /></div>
      <div><span className="eyebrow">{snapshot.platform.toUpperCase()}</span><h2>{snapshot.localHardware.hostname}</h2><p>{snapshot.localHardware.gpus.length} accelerator(s) · {formatMemory(snapshot.localHardware.ramMb)} RAM</p></div>
      <span className={`state-badge ${snapshot.contribution.state === "connected" ? "online" : snapshot.contribution.state}`}>{contributionLabel}</span>
    </div>
    <AcceleratorProgressPanel acceleration={snapshot.acceleration} contributionState={snapshot.contribution.state} />
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

function AcceleratorCompactBanner({ acceleration, contributionState, onOpen }: { acceleration: AcceleratorProgressSnapshot; contributionState: DashboardSnapshot["contribution"]["state"] | undefined; onOpen: () => void }) {
  const cpuLabel = accelerationCpuLabel(acceleration, contributionState);
  const cpuUnavailable = acceleration.cpu?.state === "unavailable";
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
      <span><small>{cpuUnavailable ? "NEEDS ATTENTION" : "AVAILABLE NOW"}</small><strong>{cpuLabel}</strong><em>{acceleration.cpu?.deviceName || "Local CPU runtime"}</em></span>
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

function AcceleratorProgressPanel({ acceleration, contributionState }: { acceleration: AcceleratorProgressSnapshot; contributionState: DashboardSnapshot["contribution"]["state"] }) {
  const cpuActive = (acceleration.cpu?.activeStages ?? 0) > 0;
  const cpuUnavailable = acceleration.cpu?.state === "unavailable";
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
  const panelTitle = accelerationPanelTitle(acceleration, contributionState);
  const panelCopy = accelerationPanelCopy(acceleration, contributionState);
  const showProgress = gpuPreparing || progress !== null;
  return <article className={`accelerator-progress-panel gpu-${acceleration.gpu?.state ?? "not-detected"}`}>
    <header className="accelerator-progress-header">
      <div className="accelerator-progress-heading">
        <span className="accelerator-runtime-icon">{gpuPreparing ? <LoaderCircle className="spin" /> : gpuActive ? <Zap /> : issue ? <CircleAlert /> : <Cpu />}</span>
        <div><span>LOCAL COMPUTE TRANSITION</span><h2>{panelTitle}</h2><p>{panelCopy}</p></div>
      </div>
      <span className={`accelerator-overall-badge ${gpuActive ? "gpu-active" : cpuActive ? "cpu-active" : cpuUnavailable ? "unavailable" : connectionError ? "connection-error" : "cpu-ready"}`}><i />{gpuActive ? "GPU ACTIVE" : cpuActive ? "CPU ACTIVE" : cpuUnavailable ? "CPU UNAVAILABLE" : contributionState === "connected" ? "CPU READY · NODE ONLINE" : contributionState === "connecting" ? "CPU READY · CONNECTING" : connectionError ? "CPU READY · CONNECTION ERROR" : "CPU READY · CONTRIBUTION PAUSED"}</span>
    </header>

    <div className="accelerator-engine-grid">
      <section className={`accelerator-engine-card cpu ${cpuActive ? "active" : cpuUnavailable ? "unavailable" : connectionError ? "connection-error" : "ready"}`} aria-label="CPU runtime status">
        <div className="accelerator-engine-title"><span><Cpu /></span><div><small>CPU RUNTIME</small><strong>{accelerationCpuLabel(acceleration, contributionState)}</strong></div><b>{acceleration.cpu?.precision?.toUpperCase() ?? "FP32"}</b></div>
        <h3>{acceleration.cpu?.deviceName || "Local CPU"}</h3>
        <p>{accelerationCpuCopy(acceleration, contributionState)}</p>
        <div className="accelerator-engine-foot"><span><i />{cpuActive ? `${acceleration.cpu.activeStages} active stage${acceleration.cpu.activeStages === 1 ? "" : "s"}` : cpuUnavailable ? "Runtime unavailable" : contributionState === "connected" ? "Ready for network work" : contributionState === "connecting" ? "Connecting to the network" : connectionError ? "Network connection needs attention" : "Ready when contribution resumes"}</span></div>
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
    <div className="settings-section"><div><Gauge size={20} /><div><h3>Contribution</h3><p>Register real hardware only, or expose a model through a verified local runtime.</p></div></div><div className="settings-fields"><label>Runtime<AppSelect ariaLabel="Contribution runtime" value={draft.adapterMode} onChange={(value) => update("adapterMode", value as DesktopSettings["adapterMode"])} options={[{ value: "connectivity-test", label: "Hardware only (no model)" }, { value: "local-model-runtime", label: "Local local model runtime" }]} /></label><label>Offered VRAM (GB)<input type="number" min="1" step="1" value={draft.offeredVramMb / 1_024} onChange={(event) => update("offeredVramMb", Math.round(Number(event.target.value) * 1_024))} /></label>{draft.adapterMode === "local-model-runtime" && <><label>local model runtime model<input value={draft.modelName} onChange={(event) => update("modelName", event.target.value)} /></label><label>local model runtime URL<input value={draft.adapterBaseUrl} onChange={(event) => update("adapterBaseUrl", event.target.value)} /></label><label>Pinned digest<input value={draft.modelDigest} onChange={(event) => update("modelDigest", event.target.value)} placeholder="sha256:…" /></label></>}</div></div>
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
  if (worker.kind === "browser") return "SESIÓN DE NAVEGADOR ACTIVA";
  if (worker.kind === "cell") return "CELDA DE CÓMPUTO";
  return "DISPOSITIVO INSTALADO";
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

function accelerationCpuLabel(acceleration: AcceleratorProgressSnapshot, contributionState?: DashboardSnapshot["contribution"]["state"]): string {
  if ((acceleration.cpu?.activeStages ?? 0) > 0) return "CPU ACTIVE";
  if (acceleration.cpu?.state === "unavailable") return "CPU UNAVAILABLE";
  if (contributionState === "connected") return "CPU READY · NODE ONLINE";
  if (contributionState === "connecting") return "CPU READY · CONNECTING";
  if (contributionState === "error") return "CPU READY · CONNECTION ERROR";
  return "CPU READY · CONTRIBUTION PAUSED";
}

function accelerationCpuCopy(acceleration: AcceleratorProgressSnapshot, contributionState: DashboardSnapshot["contribution"]["state"]): string {
  if (acceleration.cpu?.state === "unavailable") return "The certified CPU runtime could not start. Open the runtime log, then restart or update mycellios.";
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

function accelerationPanelTitle(acceleration: AcceleratorProgressSnapshot, contributionState: DashboardSnapshot["contribution"]["state"]): string {
  const gpuState = acceleration.gpu?.state;
  if (acceleration.cpu?.state === "unavailable") return "The local compute runtime needs attention.";
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

function accelerationPanelCopy(acceleration: AcceleratorProgressSnapshot, contributionState: DashboardSnapshot["contribution"]["state"]): string {
  const gpuState = acceleration.gpu?.state;
  if (acceleration.cpu?.state === "unavailable") return "The certified CPU runtime is not available. Review the runtime log before contributing from this device.";
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
function relativeTimeEs(value: string) { const seconds = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 1_000)); if (seconds < 5) return "ahora"; if (seconds < 60) return `hace ${seconds} s`; const minutes = Math.floor(seconds / 60); if (minutes < 60) return `hace ${minutes} min`; return `hace ${Math.floor(minutes / 60)} h`; }

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
  return execution.fallback ? "MIXED · FALLBACK" : "MIXED CPU + GPU";
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

function ExecutionBadge({ execution, large = false }: { execution: EffectiveExecutionSummary; large?: boolean }) {
  const Icon = execution.deviceType === "gpu" ? Zap : execution.deviceType === "mixed" ? Network : Cpu;
  const title = execution.fallbackReasons.length > 0 ? execution.fallbackReasons.join(" · ") : executionDetail(execution);
  return <span className={`execution-badge ${execution.deviceType}${execution.fallback ? " fallback" : ""}${large ? " large" : ""}`} title={title}><Icon /><span><strong>{executionShortLabel(execution)}</strong>{large && <small>{execution.verified ? execution.backends.map((backend) => backend.toUpperCase()).join(" + ") : "NO DEVICE CLAIM"}</small>}</span></span>;
}

function ExecutionStages({ execution, compact = false }: { execution: EffectiveExecutionSummary; compact?: boolean }) {
  if (execution.stages.length === 0) return null;
  return <div className={`execution-stage-list${compact ? " compact" : ""}`}>
    {execution.stages.map((stage) => {
      const summary = summarizeExecutionStages([stage]);
      return <div className="execution-stage" key={`${stage.nodeId}-${stage.stageIndex}`}><span><small>STAGE {stage.stageIndex + 1} · LAYERS {stage.layerStart}–{stage.layerEnd}</small><strong>{stage.deviceName}</strong><em>{shortId(stage.nodeId)} · {stage.backend.toUpperCase()} · {stage.precision}</em></span><ExecutionBadge execution={summary} /></div>;
    })}
  </div>;
}

function inferenceModelOption(snapshot: PublicSnapshot, model: PublicSnapshot["models"][number]) {
  const deployments = snapshot.workers.flatMap((worker) => worker.deployments).filter((deployment) => deployment.model === model.id);
  const adapters = deployments.map((deployment) => deployment.adapter).filter((adapter): adapter is NonNullable<PublicDeployment["adapter"]> => adapter !== undefined);
  const connectivityOnly = model.id === "mycellios-connectivity-check" || (adapters.length > 0 && adapters.every((adapter) => adapter === "mock"));
  const freeSlots = deployments.reduce((total, deployment) => total + deployment.freeSlots, 0);
  const routeLabel = model.pipelines > 0
    ? `${model.pipelines} pipeline${model.pipelines === 1 ? "" : "s"}`
    : `${model.replicas} réplica${model.replicas === 1 ? "" : "s"}`;
  return { ...model, connectivityOnly, freeSlots, routeLabel, execution: summarizeDeploymentExecution(deployments) };
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
