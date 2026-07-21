import {
  Activity,
  ArrowRight,
  Boxes,
  Check,
  CheckCircle2,
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
import { useCallback, useEffect, useMemo, useState, type CSSProperties } from "react";
import type {
  DashboardSnapshot,
  DesktopBridge,
  DesktopSettings,
  DesktopUpdateStatus,
} from "../../src/desktop/contracts";
import brandIcon from "../../src/renderer/assets/mycellios-icon.png";
import "./panel.css";
import "./panel-downloads.css";
import "./panel-desktop.css";

type PanelView = "overview" | "nodes" | "models" | "jobs" | "inference" | "join" | "downloads" | "machine" | "settings";

interface PanelProps {
  desktopBridge?: DesktopBridge;
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
}

interface PublicDeployment {
  deploymentId: string;
  model: string;
  mode: string;
  freeSlots: number;
  tokensPerSecond: number;
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
  jobs: PublicJob[];
}

const EMPTY: PublicSnapshot = {
  capturedAt: new Date(0).toISOString(),
  version: "—",
  summary: { registered: 0, connected: 0, online: 0, mobile: 0, offeredVramMb: 0, completedJobs: 0 },
  workers: [],
  models: [],
  jobs: [],
};

const sharedNavItems: Array<{ id: PanelView; label: string; icon: typeof Network }> = [
  { id: "overview", label: "Network", icon: Network },
  { id: "nodes", label: "Nodes", icon: Server },
  { id: "models", label: "Models", icon: Boxes },
  { id: "jobs", label: "Jobs", icon: Activity },
  { id: "inference", label: "Inference", icon: MessageSquareText },
  { id: "join", label: "Join network", icon: Zap },
  { id: "downloads", label: "Downloads", icon: Download },
];

function initialView(desktop: boolean): PanelView {
  if (desktop) return "overview";
  if (window.location.pathname === "/join") return "join";
  if (window.location.pathname === "/downloads") return "downloads";
  const requested = new URLSearchParams(window.location.search).get("view");
  return sharedNavItems.some((item) => item.id === requested) ? requested as PanelView : "overview";
}

function Panel({ desktopBridge }: PanelProps = {}) {
  const desktop = desktopBridge !== undefined;
  const navItems = useMemo(() => desktop
    ? [
        ...sharedNavItems.slice(0, 1),
        { id: "machine" as const, label: "This machine", icon: Cpu },
        ...sharedNavItems.slice(1),
        { id: "settings" as const, label: "Settings", icon: Settings },
      ]
    : sharedNavItems, [desktop]);
  const [view, setView] = useState<PanelView>(() => initialView(desktop));
  const [snapshot, setSnapshot] = useState<PublicSnapshot>(EMPTY);
  const [desktopSnapshot, setDesktopSnapshot] = useState<DashboardSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
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
      const path = next === "join" ? "/join" : next === "downloads" ? "/downloads" : "/network";
      const query = next === "overview" || next === "join" || next === "downloads" ? "" : `?view=${next}`;
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

  async function sendPrompt(model: string, prompt: string) {
    if (desktopBridge) return (await desktopBridge.sendChat({ model, prompt })).text;
    const response = await fetch("/v1/chat/completions", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ model, messages: [{ role: "user", content: prompt }], max_tokens: 256 }) });
    const body = await response.json() as { choices?: Array<{ message?: { content?: string } }>; error?: { message?: string } };
    if (!response.ok) throw new Error(body.error?.message ?? `HTTP ${response.status}`);
    return body.choices?.[0]?.message?.content ?? "The network returned an empty response.";
  }

  const publicOrigin = desktopSnapshot?.settings.coordinatorMode === "remote"
    ? desktopSnapshot.coordinatorUrl.replace(/\/$/, "")
    : "https://www.mycellios.com";
  const publicLink = (path: string) => desktop ? `${publicOrigin}${path}` : path;
  const externalProps = desktop ? { target: "_blank", rel: "noreferrer" } as const : {};

  return (
    <div className={`public-panel ${desktop ? `desktop-panel platform-${desktopSnapshot?.platform ?? "loading"}` : ""}`}>
      <aside className={menuOpen ? "panel-sidebar open" : "panel-sidebar"}>
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
          <a href={publicLink("/mobile/")} {...externalProps}><Smartphone size={17} /><span>Mobile worker</span><ExternalLink size={13} /></a>
          <a href={publicLink("/")} {...externalProps}><House size={17} /><span>Landing</span><ExternalLink size={13} /></a>
        </div>
        <div className="panel-sidebar-status"><i /><div><strong>Public test network</strong><span>No login required</span></div></div>
      </aside>

      <div className="panel-main">
        <header className="panel-topbar">
          <button
            className="panel-menu-button"
            aria-label={menuOpen ? "Close menu" : "Open menu"}
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((current) => !current)}
          >{menuOpen ? <X /> : <Menu />}</button>
          <div><span className="panel-live-dot" /><strong>mycellios network</strong><small>{error ? "Coordinator unavailable" : "Live coordinator"}</small></div>
          <div className="panel-top-actions">
            {desktopSnapshot?.update.state === "ready" && <button className="panel-update-ready" onClick={() => void desktopBridge?.installUpdate()} title="Restart and install update"><Download size={15} /></button>}
            {desktopSnapshot && <span>v{desktopSnapshot.appVersion}</span>}
            <span>API {snapshot.version}</span>
            <button onClick={() => void refresh()} aria-label="Refresh"><RefreshCw className={loading ? "spin" : ""} size={17} /></button>
            {!desktop && <a href="/mobile/">Contribute <Zap size={15} /></a>}
            {desktopBridge && <div className="panel-window-controls"><button onClick={() => void desktopBridge.minimizeWindow()} aria-label="Minimize"><Minus size={16} /></button><button onClick={() => void desktopBridge.toggleMaximizeWindow()} aria-label="Maximize"><Maximize2 size={15} /></button><button className="close" onClick={() => void desktopBridge.closeWindow()} aria-label="Close"><X size={16} /></button></div>}
          </div>
        </header>

        {error && <div className="panel-error"><CircleAlert size={17} /> Could not refresh the coordinator: {error}</div>}
        <main className="panel-content">
          {loading && snapshot.capturedAt === EMPTY.capturedAt ? <PanelLoading /> : (
            <>
              {view === "overview" && <Overview snapshot={snapshot} onNavigate={navigate} publicLink={publicLink} external={desktop} />}
              {view === "nodes" && <Nodes snapshot={snapshot} onRemove={removeWorker} onClearOffline={clearOfflineWorkers} />}
              {view === "models" && <Models snapshot={snapshot} />}
              {view === "jobs" && <Jobs snapshot={snapshot} />}
              {view === "inference" && <Inference snapshot={snapshot} onSend={sendPrompt} />}
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
    <section>
      <PageTitle eyebrow="LIVE FABRIC" title="Distributed network" copy="Every active browser and desktop node connected to the public mycellios coordinator." actions={<><button onClick={() => onNavigate("nodes")}>Manage nodes</button><a href={publicLink("/mobile/")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}>Add this device <ArrowRight size={16} /></a></>} />
      <div className="panel-stat-grid">
        <Stat icon={Radio} label="Connected" value={String(snapshot.summary.connected)} detail={`${snapshot.summary.registered} registered`} />
        <Stat icon={HardDrive} label="Offered memory" value={formatMemory(snapshot.summary.offeredVramMb)} detail="Across desktop nodes" />
        <Stat icon={Boxes} label="Models" value={String(snapshot.models.length)} detail="Available now" />
        <Stat icon={CheckCircle2} label="Completed" value={String(snapshot.summary.completedJobs)} detail="Recent jobs" />
      </div>
      <div className="overview-grid">
        <article className="network-canvas">
          <div className="canvas-head"><div><span>PHYSICAL NETWORK</span><strong>{active.length} live nodes</strong></div><span className="canvas-live"><i /> LIVE</span></div>
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
        </article>
        <article className="activity-card">
          <div className="card-heading"><div><span>RECENT ACTIVITY</span><h2>Network jobs</h2></div><button onClick={() => onNavigate("jobs")}>View all</button></div>
          <div className="activity-list">
            {snapshot.jobs.slice(0, 7).map((job) => <div key={job.id}><i className={job.status} /><div><strong>{job.model}</strong><span>{shortId(job.id)} · {relativeTime(job.updatedAt)}</span></div><b>{job.status}</b></div>)}
            {snapshot.jobs.length === 0 && <Empty icon={Activity} title="No jobs yet" copy="Jobs will appear here when you test a connected model." />}
          </div>
        </article>
      </div>
    </section>
  );
}

function Nodes({ snapshot, onRemove, onClearOffline }: { snapshot: PublicSnapshot; onRemove: (workerId: string) => Promise<void>; onClearOffline: () => Promise<void> }) {
  const [busy, setBusy] = useState<string | null>(null);
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
    <section>
      <PageTitle eyebrow="NODE REGISTRY" title="Connected machines" copy="Inspect hardware, contribution status and recent activity for every registered device." actions={<button disabled={busy !== null} onClick={() => void clearOffline()}><Trash2 size={15} /> Clear offline</button>} />
      <div className="node-card-grid">
        {snapshot.workers.map((worker) => <article className="node-card" key={worker.id}>
          <div className="node-card-head"><div className={`node-device ${worker.kind}`}>{worker.kind === "browser" ? <Smartphone /> : <Laptop />}</div><div><span>{worker.kind === "browser" ? "BROWSER WORKER" : "DESKTOP AGENT"}</span><h2>{workerLabel(worker)}</h2><small>{shortId(worker.id)}</small></div><span className={`node-status ${worker.status}`}><i />{worker.connected ? "connected" : worker.status}</span></div>
          <div className="node-card-metrics"><Metric label="Hardware" value={worker.gpus[0]?.model ?? "Unknown"} /><Metric label="Offered" value={formatMemory(worker.offeredVramMb)} /><Metric label="Region" value={worker.region} /><Metric label="Completed" value={String(worker.jobsCompleted)} /></div>
          <div className="node-card-foot"><span>Last seen {relativeTime(worker.lastSeenAt)}</span><button disabled={busy === worker.id} onClick={() => void remove(worker.id)}>{busy === worker.id ? <LoaderCircle className="spin" /> : <Trash2 />} Remove</button></div>
        </article>)}
        {snapshot.workers.length === 0 && <div className="wide-empty"><Empty icon={Server} title="No registered nodes" copy="Open the mobile worker or install mycellios to add the first machine." /><a href="/mobile/">Connect a device <ArrowRight size={15} /></a></div>}
      </div>
    </section>
  );
}

function Models({ snapshot }: { snapshot: PublicSnapshot }) {
  return <section><PageTitle eyebrow="MODEL FABRIC" title="Available models" copy="Replicas and distributed pipelines announced by connected desktop nodes." />
    <div className="panel-table"><div className="panel-table-head"><span>Model</span><span>Replicas</span><span>Pipelines</span><span>Status</span></div>
      {snapshot.models.map((model) => <div className="panel-table-row" key={model.id}><strong><Boxes size={17} />{model.id}</strong><span>{model.replicas}</span><span>{model.pipelines}</span><b><i />Available</b></div>)}
      {snapshot.models.length === 0 && <Empty icon={Boxes} title="No models announced" copy="Connect a desktop node with local model runtime to publish the first model." />}
    </div></section>;
}

function Jobs({ snapshot }: { snapshot: PublicSnapshot }) {
  return <section><PageTitle eyebrow="EXECUTION LOG" title="Network jobs" copy="Recent inference requests and their execution state." />
    <div className="panel-table jobs-table"><div className="panel-table-head"><span>Request</span><span>Model</span><span>Tokens</span><span>Status</span></div>
      {snapshot.jobs.map((job) => <div className="panel-table-row" key={job.id}><strong><Activity size={16} />{shortId(job.id)}<small>{relativeTime(job.createdAt)}</small></strong><span>{job.model}</span><span>{job.inputTokens + job.outputTokens}</span><b className={job.status}><i />{job.status}</b></div>)}
      {snapshot.jobs.length === 0 && <Empty icon={Activity} title="No jobs yet" copy="Use the inference console when a model becomes available." />}
    </div></section>;
}

function Inference({ snapshot, onSend }: { snapshot: PublicSnapshot; onSend: (model: string, prompt: string) => Promise<string> }) {
  const [model, setModel] = useState(snapshot.models[0]?.id ?? "");
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const selectedModel = useMemo(() => snapshot.models.some((item) => item.id === model) ? model : snapshot.models[0]?.id ?? "", [model, snapshot.models]);
  async function send() {
    if (!selectedModel || !prompt.trim()) return;
    setSending(true); setError(null); setResult(null);
    try {
      setResult(await onSend(selectedModel, prompt.trim()));
    } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); }
    finally { setSending(false); }
  }
  return <section><PageTitle eyebrow="INFERENCE CONSOLE" title="Test the network" copy="Send a real request through the public test coordinator." />
    <div className="inference-console"><div className="inference-toolbar"><label>MODEL<select value={selectedModel} onChange={(event) => setModel(event.target.value)}><option value="">No model available</option>{snapshot.models.map((item) => <option key={item.id}>{item.id}</option>)}</select></label><span><i />PUBLIC TEST API</span></div>
      <div className="inference-output">{!result && !error && <Empty icon={MessageSquareText} title="Console ready" copy={selectedModel ? "Write a prompt to test the complete route." : "Connect a model before sending a request."} />}{result && <div className="inference-message"><img src={brandIcon} alt="" /><div><span>mycellios · {selectedModel}</span><p>{result}</p></div></div>}{error && <div className="inference-error"><CircleAlert />{error}</div>}</div>
      <div className="inference-input"><textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Ask the network something…" /><button disabled={!selectedModel || !prompt.trim() || sending} onClick={() => void send()}>{sending ? <LoaderCircle className="spin" /> : <Send />}</button></div>
    </div></section>;
}

function JoinNetwork({ publicLink, external }: { publicLink: (path: string) => string; external: boolean }) {
  return <section><PageTitle eyebrow="ZERO-CONFIG JOIN" title="Add this device" copy="No account, invitation or terminal. Choose the fastest option for this device." />
    <div className="join-grid"><a className="join-card featured" href={publicLink("/mobile/")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><div><Smartphone /><span>INSTANT</span></div><h2>Use this browser</h2><p>Works on Android, iPhone, tablets and desktop browsers. Keep the page visible while contributing.</p><ul><li><CheckCircle2 />WebGPU when available</li><li><CheckCircle2 />CPU fallback everywhere</li><li><CheckCircle2 />Nothing to install</li></ul><strong>Open browser worker <ArrowRight /></strong></a>
      <a className="join-card" href={publicLink("/downloads")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><div><Laptop /><span>WINDOWS · MACOS</span></div><h2>Install the desktop agent</h2><p>Runs in the background and can expose native GPU runtimes such as local model runtime.</p><ul><li><CheckCircle2 />Windows 10/11</li><li><CheckCircle2 />Apple Silicon and Intel Macs</li><li><CheckCircle2 />Background contribution</li></ul><strong>Choose your computer <Download /></strong></a></div>
  </section>;
}

function Downloads({ publicLink, external }: { publicLink: (path: string) => string; external: boolean }) {
  return <section><PageTitle eyebrow="CLIENTS" title="Downloads" copy="Install the native client or enter immediately through the universal browser worker." />
    <div className="download-grid"><a className="download-card ready" href={publicLink("/downloads/windows")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Laptop />WINDOWS</span><h2>Windows 10/11</h2><p>x64 installer · Electron desktop agent with automatic updates</p><strong>Download installer <Download /></strong></a><a className="download-card ready" href={publicLink("/downloads/macos-arm64")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Cpu />MACOS · APPLE SILICON</span><h2>Mac M1–M4</h2><p>Unsigned test DMG · Apple Silicon arm64</p><strong>Download DMG <Download /></strong></a><a className="download-card ready" href={publicLink("/downloads/macos-x64")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Cpu />MACOS · INTEL</span><h2>Mac Intel</h2><p>Unsigned test DMG · Intel x64</p><strong>Download DMG <Download /></strong></a><a className="download-card ready" href={publicLink("/mobile/")} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}><span><Globe2 />UNIVERSAL</span><h2>Browser worker</h2><p>Android, iOS, macOS, Linux and Windows</p><strong>Open now <ExternalLink /></strong></a></div>
  </section>;
}

function Machine({ snapshot, bridge, onSnapshot }: { snapshot: DashboardSnapshot; bridge: DesktopBridge; onSnapshot: (snapshot: DashboardSnapshot) => void }) {
  const [busy, setBusy] = useState(false);
  async function toggleContribution() {
    setBusy(true);
    try { onSnapshot(await bridge.setContribution(!snapshot.settings.contributionEnabled)); }
    finally { setBusy(false); }
  }
  return <section className="content-page">
    <PageTitle eyebrow="LOCAL NODE" title="This machine" copy="Hardware detected on this computer and the resources offered to mycellios." actions={<button disabled={busy} onClick={() => void toggleContribution()}><CirclePower size={16} />{snapshot.settings.contributionEnabled ? "Pause contribution" : "Start contributing"}</button>} />
    <div className="hardware-hero">
      <div className="device-orb"><Cpu size={42} /></div>
      <div><span className="eyebrow">{snapshot.platform.toUpperCase()}</span><h2>{snapshot.localHardware.hostname}</h2><p>{snapshot.localHardware.gpus.length} accelerator(s) · {formatMemory(snapshot.localHardware.ramMb)} RAM</p></div>
      <span className={`state-badge ${snapshot.settings.contributionEnabled ? "online" : "paused"}`}>{snapshot.settings.contributionEnabled ? "Contributing" : "Paused"}</span>
    </div>
    <div className="cards-grid">
      {snapshot.localHardware.gpus.map((gpu) => <article className="hardware-card" key={gpu.id}><div className="hardware-card-icon"><Gauge size={23} /></div><span className="card-kicker">{gpu.vendor}</span><h3>{gpu.model}</h3><div className="capacity-line"><span style={{ width: `${Math.min(100, snapshot.settings.offeredVramMb / Math.max(1, gpu.physicalVramMb + (gpu.sharedMemoryMb ?? 0)) * 100)}%` }} /></div><div className="hardware-metrics"><DesktopMetric label="Physical VRAM" value={formatMemory(gpu.physicalVramMb)} /><DesktopMetric label="Shared" value={formatMemory(gpu.sharedMemoryMb ?? 0)} /><DesktopMetric label="Utilization" value={`${gpu.utilizationPct ?? 0}%`} /><DesktopMetric label="Temperature" value={gpu.temperatureC === undefined ? "—" : `${gpu.temperatureC} °C`} /></div></article>)}
      {snapshot.localHardware.gpus.length === 0 && <div className="wide-empty"><Empty icon={Cpu} title="No GPU detected" copy="This machine can still run the connectivity test and CPU-compatible workloads." /></div>}
    </div>
  </section>;
}

function DesktopSettingsView({ snapshot, bridge, onSnapshot }: { snapshot: DashboardSnapshot; bridge: DesktopBridge; onSnapshot: (snapshot: DashboardSnapshot) => void }) {
  const [draft, setDraft] = useState(snapshot.settings);
  const [saving, setSaving] = useState(false);
  const [checkingUpdate, setCheckingUpdate] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => setDraft(snapshot.settings), [snapshot.settings]);
  function update<K extends keyof DesktopSettings>(key: K, value: DesktopSettings[K]) { setDraft((current) => ({ ...current, [key]: value })); }
  async function save() {
    setSaving(true); setError(null);
    try { onSnapshot(await bridge.saveSettings(draft)); }
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
    <div className="settings-section"><div><Globe2 size={20} /><div><h3>Coordinator</h3><p>Use the public network or host an isolated local coordinator.</p></div></div><div className="settings-fields"><label>Mode<select value={draft.coordinatorMode} onChange={(event) => update("coordinatorMode", event.target.value as DesktopSettings["coordinatorMode"])}><option value="remote">Public mycellios network</option><option value="local">Local network on this machine</option></select></label>{draft.coordinatorMode === "remote" && <label>Coordinator URL<input value={draft.remoteCoordinatorUrl} onChange={(event) => update("remoteCoordinatorUrl", event.target.value)} /></label>}<label>Region<input value={draft.region} onChange={(event) => update("region", event.target.value)} placeholder="auto" /></label></div></div>
    <div className="settings-section"><div><Gauge size={20} /><div><h3>Contribution</h3><p>Select the runtime and memory offered by this node.</p></div></div><div className="settings-fields"><label>Runtime<select value={draft.adapterMode} onChange={(event) => update("adapterMode", event.target.value as DesktopSettings["adapterMode"])}><option value="connectivity-test">Connectivity test</option><option value="local-model-runtime">Local local model runtime</option></select></label><label>Offered VRAM (MB)<input type="number" min="512" step="256" value={draft.offeredVramMb} onChange={(event) => update("offeredVramMb", Number(event.target.value))} /></label>{draft.adapterMode === "local-model-runtime" && <><label>local model runtime model<input value={draft.modelName} onChange={(event) => update("modelName", event.target.value)} /></label><label>local model runtime URL<input value={draft.adapterBaseUrl} onChange={(event) => update("adapterBaseUrl", event.target.value)} /></label><label>Pinned digest<input value={draft.modelDigest} onChange={(event) => update("modelDigest", event.target.value)} placeholder="sha256:…" /></label></>}</div></div>
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
    try { onSnapshot(await bridge.saveSettings({ ...snapshot.settings, coordinatorMode: "remote", remoteCoordinatorUrl: "https://www.mycellios.com", remoteCoordinatorToken: "", contributionEnabled: contribute, onboardingComplete: true })); }
    catch (caught) { setError(errorText(caught)); }
    finally { setSaving(false); }
  }
  return <div className="modal-backdrop"><div className="onboarding-card"><div className="onboarding-brand"><img src={brandIcon} alt="" /><span>GET STARTED</span></div><h1>Connect this machine to mycellios</h1><p>The desktop agent and the web panel will show the same public network.</p><div className="mode-grid single-mode"><button className="selected"><Globe2 size={23} /><strong>Public mycellios network</strong><span>https://www.mycellios.com</span><Check size={16} className="mode-check" /></button></div><label className="contribute-choice"><input type="checkbox" checked={contribute} onChange={(event) => setContribute(event.target.checked)} /><span><strong>Contribute this machine's resources</strong><small>Start with a safe connectivity test; switch to local model runtime later.</small></span></label>{error && <div className="inline-error"><CircleAlert size={17} />{error}</div>}<button className="primary-button onboarding-submit" disabled={saving} onClick={() => void complete()}>{saving ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={18} />}Enter mycellios <ChevronRight size={18} /></button><div className="onboarding-security"><ShieldCheck size={15} />Connected through HTTPS/WSS to the public coordinator.</div></div></div>;
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
function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) { return <label className="toggle-row"><span>{label}</span><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><i /></label>; }
function errorText(error: unknown) { return error instanceof Error ? error.message : String(error); }

function PageTitle({ eyebrow, title, copy, actions }: { eyebrow: string; title: string; copy: string; actions?: React.ReactNode }) {
  return <div className="panel-page-title"><div><span>{eyebrow}</span><h1>{title}</h1><p>{copy}</p></div>{actions && <div className="page-actions">{actions}</div>}</div>;
}

function Stat({ icon: Icon, label, value, detail }: { icon: typeof Radio; label: string; value: string; detail: string }) { return <article className="panel-stat"><div><Icon /></div><span>{label}</span><strong>{value}</strong><small>{detail}</small></article>; }
function Metric({ label, value }: { label: string; value: string }) { return <div><span>{label}</span><strong>{value}</strong></div>; }
function Empty({ icon: Icon, title, copy }: { icon: typeof Activity; title: string; copy: string }) { return <div className="panel-empty"><Icon /><strong>{title}</strong><p>{copy}</p></div>; }
function PanelLoading() { return <div className="panel-loading"><LoaderCircle className="spin" /><strong>Connecting to mycellios</strong><span>Loading the public network snapshot…</span></div>; }
function workerLabel(worker: PublicWorker) { return worker.kind === "browser" ? `Browser ${shortId(worker.id)}` : worker.gpus[0]?.model || `Desktop ${shortId(worker.id)}`; }
function shortId(value: string) { return value.length <= 10 ? value : `${value.slice(0, 6)}…${value.slice(-4)}`; }
function formatMemory(value: number) { return value >= 1_024 ? `${(value / 1_024).toFixed(value >= 10_240 ? 0 : 1)} GB` : `${Math.round(value)} MB`; }
function relativeTime(value: string) { const seconds = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 1_000)); if (seconds < 5) return "now"; if (seconds < 60) return `${seconds}s ago`; const minutes = Math.floor(seconds / 60); if (minutes < 60) return `${minutes}m ago`; return `${Math.floor(minutes / 60)}h ago`; }

export { Panel };
