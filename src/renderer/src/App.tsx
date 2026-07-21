import {
  Activity,
  Boxes,
  Check,
  ChevronRight,
  CircleAlert,
  CirclePower,
  Cpu,
  Download,
  Gauge,
  Globe2,
  HardDrive,
  Info,
  Laptop,
  LoaderCircle,
  Maximize2,
  MemoryStick,
  MessageSquareText,
  Minus,
  Network,
  RefreshCw,
  Send,
  Server,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Thermometer,
  Wifi,
  X,
  Zap,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import type {
  ChatResponse,
  DashboardSnapshot,
  DesktopSettings,
  DesktopUpdateStatus,
} from "../../desktop/contracts";
import brandIcon from "../assets/mycellios-icon.png";
import { NetworkGraph, type GraphSelection } from "./NetworkGraph";

type View = "network" | "hardware" | "models" | "chat" | "settings";

const EMPTY_MESSAGE = "Could not load mycellios status.";

export function App() {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [view, setView] = useState<View>("network");
  const [loading, setLoading] = useState(true);
  const [actionError, setActionError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>("network");
  const [selection, setSelection] = useState<GraphSelection>({
    kind: "network",
    label: "mycellios network",
  });

  const refresh = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true);
    try {
      const next = await window.mycellios.getSnapshot();
      setSnapshot(next);
      setActionError(null);
    } catch (error) {
      setActionError(errorText(error));
    } finally {
      if (!quiet) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(true), 4_000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const toggleContribution = useCallback(async () => {
    if (!snapshot) return;
    setLoading(true);
    try {
      const next = await window.mycellios.setContribution(!snapshot.settings.contributionEnabled);
      setSnapshot(next);
      setActionError(null);
    } catch (error) {
      setActionError(errorText(error));
    } finally {
      setLoading(false);
    }
  }, [snapshot]);

  const handleSelect = useCallback((id: string, next: GraphSelection) => {
    setSelectedId(id);
    setSelection(next);
  }, []);

  if (!snapshot && loading) {
    return <LaunchScreen />;
  }

  if (!snapshot) {
    return (
      <div className="fatal-state">
        <CircleAlert size={30} />
        <h1>{EMPTY_MESSAGE}</h1>
        <p>{actionError}</p>
        <button className="primary-button" onClick={() => void refresh()}>
          Reintentar
        </button>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <TitleBar snapshot={snapshot} loading={loading} onRefresh={() => void refresh()} />
      <Sidebar view={view} setView={setView} />
      <main className="main-stage">
        {(actionError || snapshot.connectionError || snapshot.runtimeError) && (
          <ErrorBanner
            message={actionError ?? snapshot.runtimeError ?? snapshot.connectionError ?? ""}
            onClose={() => setActionError(null)}
          />
        )}
        {view === "network" && (
          <NetworkView
            snapshot={snapshot}
            selectedId={selectedId}
            selection={selection}
            onSelect={handleSelect}
            onToggleContribution={toggleContribution}
          />
        )}
        {view === "hardware" && <HardwareView snapshot={snapshot} />}
        {view === "models" && <ModelsView snapshot={snapshot} />}
        {view === "chat" && <ChatView snapshot={snapshot} />}
        {view === "settings" && (
          <SettingsView
            snapshot={snapshot}
            onSaved={setSnapshot}
            onUpdate={(update) =>
              setSnapshot((current) => (current ? { ...current, update } : current))
            }
          />
        )}
      </main>
      {!snapshot.settings.onboardingComplete && (
        <Onboarding snapshot={snapshot} onComplete={setSnapshot} />
      )}
    </div>
  );
}

function LaunchScreen() {
  return (
    <div className="launch-screen">
      <div className="launch-orbit" />
      <img src={brandIcon} alt="" />
      <h1>mycellios</h1>
      <p>Preparing the distributed network</p>
      <LoaderCircle className="spin" size={22} />
    </div>
  );
}

function TitleBar({
  snapshot,
  loading,
  onRefresh,
}: {
  snapshot: DashboardSnapshot;
  loading: boolean;
  onRefresh: () => void;
}) {
  const connected = snapshot.health?.status === "ok" && !snapshot.connectionError;
  return (
    <header className="titlebar">
      <div className="window-drag-region" />
      <div className="brand-lockup">
        <img src={brandIcon} alt="mycellios logo" />
        <div>
          <strong>mycellios</strong>
          <span>distributed intelligence</span>
        </div>
      </div>
      <div className="network-pill">
        <span className={`status-dot ${connected ? "online" : "offline"}`} />
        <span>{connected ? "Network connected" : "Disconnected"}</span>
        <span className="network-pill-divider" />
        <span>{snapshot.coordinatorUrl.replace(/^https?:\/\//, "")}</span>
      </div>
      <div className="titlebar-actions">
        {snapshot.update.state === "ready" && (
          <button
            className="update-ready-button"
            onClick={() => void window.mycellios.installUpdate()}
            title="Restart and install the downloaded update"
          >
            <Download size={14} /> Restart to update
          </button>
        )}
        <span className="version-label">v{snapshot.appVersion}</span>
        <button className="icon-button" onClick={onRefresh} aria-label="Refresh status">
          <RefreshCw className={loading ? "spin" : ""} size={17} />
        </button>
        <span className="window-controls" aria-label="Window controls">
          <button
            className="window-control"
            onClick={() => void window.mycellios.minimizeWindow()}
            aria-label="Minimize window"
          >
            <Minus size={16} />
          </button>
          <button
            className="window-control"
            onClick={() => void window.mycellios.toggleMaximizeWindow()}
            aria-label="Maximize or restore window"
          >
            <Maximize2 size={14} />
          </button>
          <button
            className="window-control close"
            onClick={() => void window.mycellios.closeWindow()}
            aria-label="Close window"
          >
            <X size={16} />
          </button>
        </span>
      </div>
    </header>
  );
}

function Sidebar({ view, setView }: { view: View; setView: (view: View) => void }) {
  const items: Array<{ id: View; label: string; icon: typeof Network }> = [
    { id: "network", label: "Network", icon: Network },
    { id: "hardware", label: "Machine", icon: Cpu },
    { id: "models", label: "Models", icon: Boxes },
    { id: "chat", label: "Chat", icon: MessageSquareText },
  ];
  return (
    <aside className="sidebar">
      <nav>
        {items.map((item) => {
          const Icon = item.icon;
          return (
            <button
              className={view === item.id ? "active" : ""}
              key={item.id}
              onClick={() => setView(item.id)}
              title={item.label}
            >
              <Icon size={20} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>
      <button
        className={`sidebar-settings ${view === "settings" ? "active" : ""}`}
        onClick={() => setView("settings")}
        title="Settings"
      >
        <Settings size={20} />
        <span>Settings</span>
      </button>
    </aside>
  );
}

function NetworkView({
  snapshot,
  selectedId,
  selection,
  onSelect,
  onToggleContribution,
}: {
  snapshot: DashboardSnapshot;
  selectedId: string | null;
  selection: GraphSelection;
  onSelect: (id: string, selection: GraphSelection) => void;
  onToggleContribution: () => void;
}) {
  const online = snapshot.workers.filter((worker) => worker.status === "online").length;
  const totalVram = snapshot.workers.reduce((sum, worker) => sum + worker.offeredVramMb, 0);
  return (
    <section className="network-layout">
      <div className="graph-panel">
        <div className="view-heading floating-heading">
          <div>
            <span className="eyebrow">LIVE TOPOLOGY</span>
            <h1>Connection map</h1>
          </div>
          <div className="graph-controls">
            <button className="segmented active"><Network size={15} /> Physical network</button>
            <button className="segmented"><Activity size={15} /> Active route</button>
          </div>
        </div>
        <div className="graph-glow graph-glow-one" />
        <div className="graph-glow graph-glow-two" />
        <NetworkGraph
          snapshot={snapshot}
          selectedId={selectedId}
          onSelect={onSelect}
        />
        <div className="graph-legend">
          <span><i className="legend-dot cyan" /> This machine</span>
          <span><i className="legend-dot green" /> Online</span>
          <span><i className="legend-dot purple" /> Model</span>
          <span><i className="legend-dot grey" /> Offline</span>
        </div>
        <div className="network-stats-strip">
          <MiniStat icon={Server} value={String(online)} label="online nodes" />
          <MiniStat icon={MemoryStick} value={formatMemory(totalVram)} label="shared VRAM" />
          <MiniStat icon={Boxes} value={String(snapshot.models.length)} label="models" />
          <MiniStat
            icon={Wifi}
            value={snapshot.health ? `${snapshot.health.workers.connected}` : "0"}
            label="WSS connections"
          />
        </div>
      </div>
      <Inspector
        snapshot={snapshot}
        selection={selection}
        onToggleContribution={onToggleContribution}
      />
    </section>
  );
}

function Inspector({
  snapshot,
  selection,
  onToggleContribution,
}: {
  snapshot: DashboardSnapshot;
  selection: GraphSelection;
  onToggleContribution: () => void;
}) {
  if (selection.kind === "worker") {
    const { worker } = selection;
    const gpu = worker.gpus[0];
    return (
      <aside className="inspector">
        <InspectorHeader icon={Server} eyebrow="NETWORK NODE" title={selection.label} />
        <StatusCard status={worker.status} connected={worker.connected} />
        <div className="metric-grid">
          <Metric label="Offered VRAM" value={formatMemory(worker.offeredVramMb)} />
          <Metric label="Reliability" value={`${Math.round(worker.reliability * 100)}%`} />
          <Metric label="Jobs" value={String(worker.jobsCompleted)} />
          <Metric label="Region" value={worker.region} />
        </div>
        {gpu && (
          <div className="detail-card">
            <span className="card-kicker">ACCELERATOR</span>
            <h3>{gpu.model}</h3>
            <div className="detail-row"><Gauge size={15} /> {gpu.utilizationPct ?? 0}% utilization</div>
            <div className="detail-row"><Thermometer size={15} /> {gpu.temperatureC ?? "—"} °C</div>
            <div className="detail-row"><Zap size={15} /> {gpu.powerW ?? "—"} W</div>
          </div>
        )}
        <div className="detail-card">
          <span className="card-kicker">{worker.mobile ? "MOBILE EXPERTS" : "DEPLOYMENTS"}</span>
          {worker.mobile?.residentExperts.map((expert) => (
            <div className="deployment-row" key={expert.artifactId}>
              <div><span className="status-dot online" /><strong>{expert.modelId}</strong></div>
              <span>L{expert.layer} · E{expert.expert}</span>
            </div>
          ))}
          {worker.mobile && worker.mobile.residentExperts.length === 0 && (
            <div className="deployment-row"><span>Benchmark only · awaiting an MoE expert</span></div>
          )}
          {worker.deployments.map((deployment) => (
            <div className="deployment-row" key={deployment.deploymentId}>
              <div><span className="status-dot online" /><strong>{deployment.model}</strong></div>
              <span>{deployment.tokensPerSecond.toFixed(1)} tok/s</span>
            </div>
          ))}
        </div>
      </aside>
    );
  }

  if (selection.kind === "model") {
    return (
      <aside className="inspector">
        <InspectorHeader icon={Boxes} eyebrow="DISTRIBUTED MODEL" title={selection.model.id} />
        <div className="hero-metric-card purple-card">
          <span>Availability</span>
          <strong>{selection.model.replicas + selection.model.pipelines}</strong>
          <small>usable routes</small>
        </div>
        <div className="metric-grid">
          <Metric label="Replicas" value={String(selection.model.replicas)} />
          <Metric label="Pipelines" value={String(selection.model.pipelines)} />
        </div>
      </aside>
    );
  }

  if (selection.kind === "device") {
    const gpu = snapshot.localHardware.gpus[0];
    return (
      <aside className="inspector">
        <InspectorHeader icon={Laptop} eyebrow="THIS MACHINE" title={snapshot.localHardware.hostname} />
        <div className={`contribution-card ${snapshot.settings.contributionEnabled ? "active" : ""}`}>
          <div>
            <span className="card-kicker">CONTRIBUTION</span>
            <strong>{snapshot.settings.contributionEnabled ? "Active" : "Paused"}</strong>
          </div>
          <button className="power-button" onClick={onToggleContribution}>
            <CirclePower size={21} />
          </button>
        </div>
        <div className="metric-grid">
          <Metric label="RAM" value={formatMemory(snapshot.localHardware.ramMb)} />
          <Metric label="VRAM allocation" value={formatMemory(snapshot.settings.offeredVramMb)} />
        </div>
        <div className="detail-card">
          <span className="card-kicker">DETECTED HARDWARE</span>
          <h3>{gpu?.model ?? "Unidentified GPU"}</h3>
          <div className="detail-row"><Cpu size={15} /> {snapshot.platform}</div>
          <div className="detail-row"><MemoryStick size={15} /> {formatMemory(gpu?.physicalVramMb ?? 0)} VRAM</div>
        </div>
      </aside>
    );
  }

  return (
    <aside className="inspector">
      <InspectorHeader icon={Globe2} eyebrow="CONTROL PLANE" title="mycellios network" />
      <div className="hero-metric-card">
        <span>Global status</span>
        <strong>{snapshot.health?.workers.online ?? 0}</strong>
        <small>ready nodes</small>
      </div>
      <div className="metric-grid">
        <Metric label="Registered" value={String(snapshot.health?.workers.registered ?? 0)} />
        <Metric label="Connected" value={String(snapshot.health?.workers.connected ?? 0)} />
        <Metric label="Models" value={String(snapshot.models.length)} />
        <Metric label="API version" value={snapshot.health?.version ?? "—"} />
      </div>
      <div className="detail-card security-card">
        <ShieldCheck size={22} />
        <div>
          <h3>Protected channel</h3>
          <p>{snapshot.settings.coordinatorMode === "remote" ? "HTTPS/WSS required" : "Isolated local loopback"}</p>
        </div>
      </div>
      <button className="wide-button" onClick={onToggleContribution}>
        <CirclePower size={17} />
        {snapshot.settings.contributionEnabled ? "Pause this node" : "Contribute this machine"}
      </button>
    </aside>
  );
}

function HardwareView({ snapshot }: { snapshot: DashboardSnapshot }) {
  return (
    <section className="content-page">
      <PageHeading eyebrow="LOCAL NODE" title="This machine" description="Detected hardware and the allocation mycellios may use." />
      <div className="hardware-hero">
        <div className="device-orb"><Cpu size={42} /></div>
        <div>
          <span className="eyebrow">{snapshot.platform.toUpperCase()}</span>
          <h2>{snapshot.localHardware.hostname}</h2>
          <p>{snapshot.localHardware.gpus.length} accelerator(s) · {formatMemory(snapshot.localHardware.ramMb)} RAM</p>
        </div>
        <span className={`state-badge ${snapshot.settings.contributionEnabled ? "online" : "paused"}`}>
          {snapshot.settings.contributionEnabled ? "Contributing" : "Paused"}
        </span>
      </div>
      <div className="cards-grid">
        {snapshot.localHardware.gpus.map((gpu) => (
          <article className="hardware-card" key={gpu.id}>
            <div className="hardware-card-icon"><Gauge size={23} /></div>
            <span className="card-kicker">{gpu.vendor}</span>
            <h3>{gpu.model}</h3>
            <div className="capacity-line"><span style={{ width: `${Math.min(100, snapshot.settings.offeredVramMb / Math.max(1, gpu.physicalVramMb + (gpu.sharedMemoryMb ?? 0)) * 100)}%` }} /></div>
            <div className="hardware-metrics">
              <Metric label="Physical VRAM" value={formatMemory(gpu.physicalVramMb)} />
              <Metric label="Shared" value={formatMemory(gpu.sharedMemoryMb ?? 0)} />
              <Metric label="Utilization" value={`${gpu.utilizationPct ?? 0}%`} />
              <Metric label="Temperature" value={gpu.temperatureC === undefined ? "—" : `${gpu.temperatureC} °C`} />
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function ModelsView({ snapshot }: { snapshot: DashboardSnapshot }) {
  return (
    <section className="content-page">
      <PageHeading eyebrow="MODEL FABRIC" title="Available models" description="Replicas and pipelines announced by connected nodes." />
      <div className="data-table">
        <div className="table-head"><span>Model</span><span>Replicas</span><span>Pipelines</span><span>Status</span></div>
        {snapshot.models.length === 0 ? (
          <EmptyState icon={Boxes} title="No models yet" text="Connect a node with local model runtime or a compatible runtime to announce its first model." />
        ) : snapshot.models.map((model) => (
          <div className="table-row" key={model.id}>
            <span className="model-name"><span className="model-glyph"><Sparkles size={16} /></span>{model.id}</span>
            <span>{model.replicas}</span>
            <span>{model.pipelines}</span>
            <span className="state-badge online"><Check size={13} /> Available</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function ChatView({ snapshot }: { snapshot: DashboardSnapshot }) {
  const [model, setModel] = useState(snapshot.models[0]?.id ?? "");
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState<ChatResponse | null>(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    if (!model || !prompt.trim()) return;
    setSending(true);
    setError(null);
    try {
      setResult(await window.mycellios.sendChat({ model, prompt }));
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setSending(false);
    }
  }

  return (
    <section className="content-page chat-page">
      <PageHeading eyebrow="INFERENCE CONSOLE" title="Test the network" description="Send a real request to an available model." />
      <div className="chat-console">
        <div className="chat-toolbar">
          <label>Model</label>
          <select value={model} onChange={(event) => setModel(event.target.value)}>
            <option value="">Select a model</option>
            {snapshot.models.map((item) => <option key={item.id} value={item.id}>{item.id}</option>)}
          </select>
        </div>
        <div className="chat-transcript">
          {!result && !error && <EmptyState icon={MessageSquareText} title="Console ready" text="Choose a model and ask something to verify the complete route." />}
          {error && <div className="chat-error"><CircleAlert size={18} /> {error}</div>}
          {result && (
            <div className="assistant-message">
              <div className="assistant-avatar"><Sparkles size={17} /></div>
              <div><span>mycellios · {result.model}</span><p>{result.text}</p><small>{result.totalTokens} tokens · route {result.routeClass}</small></div>
            </div>
          )}
        </div>
        <div className="chat-composer">
          <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Ask the mycellios network…" />
          <button className="send-button" disabled={sending || !model || !prompt.trim()} onClick={() => void submit()}>
            {sending ? <LoaderCircle className="spin" size={19} /> : <Send size={19} />}
          </button>
        </div>
      </div>
    </section>
  );
}

function SettingsView({
  snapshot,
  onSaved,
  onUpdate,
}: {
  snapshot: DashboardSnapshot;
  onSaved: (snapshot: DashboardSnapshot) => void;
  onUpdate: (update: DesktopUpdateStatus) => void;
}) {
  const [draft, setDraft] = useState(snapshot.settings);
  const [saving, setSaving] = useState(false);
  const [checkingUpdate, setCheckingUpdate] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => setDraft(snapshot.settings), [snapshot.settings]);

  function update<K extends keyof DesktopSettings>(key: K, value: DesktopSettings[K]) {
    setDraft((current) => ({ ...current, [key]: value }));
  }

  async function save() {
    setSaving(true);
    setError(null);
    try {
      onSaved(await window.mycellios.saveSettings(draft));
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setSaving(false);
    }
  }

  async function checkUpdate() {
    setCheckingUpdate(true);
    setError(null);
    try {
      onUpdate(await window.mycellios.checkForUpdates());
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setCheckingUpdate(false);
    }
  }

  return (
    <section className="content-page settings-page">
      <PageHeading eyebrow="PREFERENCES" title="Settings" description="Connection, contribution, and application behavior." />
      {error && <div className="inline-error"><CircleAlert size={17} /> {error}</div>}
      <div className="settings-section">
        <div><Globe2 size={20} /><div><h3>Coordinator</h3><p>Create a local network or connect this machine to a remote network.</p></div></div>
        <div className="settings-fields">
          <label>Mode<select value={draft.coordinatorMode} onChange={(event) => update("coordinatorMode", event.target.value as DesktopSettings["coordinatorMode"])}><option value="local">Local network on this machine</option><option value="remote">Join a remote coordinator</option></select></label>
          {draft.coordinatorMode === "remote" && <label>Public coordinator URL<input value={draft.remoteCoordinatorUrl} onChange={(event) => update("remoteCoordinatorUrl", event.target.value)} placeholder="https://www.mycellios.com" /></label>}
          <label>Region<input value={draft.region} onChange={(event) => update("region", event.target.value)} placeholder="auto" /></label>
        </div>
      </div>
      <div className="settings-section">
        <div><Gauge size={20} /><div><h3>Contribution</h3><p>Choose the runtime and the memory limit being offered.</p></div></div>
        <div className="settings-fields">
          <label>Runtime<select value={draft.adapterMode} onChange={(event) => update("adapterMode", event.target.value as DesktopSettings["adapterMode"])}><option value="connectivity-test">Connectivity test</option><option value="local-model-runtime">Local local model runtime</option></select></label>
          <label>Offered VRAM (MB)<input type="number" min="512" step="256" value={draft.offeredVramMb} onChange={(event) => update("offeredVramMb", Number(event.target.value))} /></label>
          {draft.adapterMode === "local-model-runtime" && <><label>local model runtime model<input value={draft.modelName} onChange={(event) => update("modelName", event.target.value)} placeholder="qwen3:4b" /></label><label>local model runtime URL<input value={draft.adapterBaseUrl} onChange={(event) => update("adapterBaseUrl", event.target.value)} /></label><label>Pinned digest<input value={draft.modelDigest} onChange={(event) => update("modelDigest", event.target.value)} placeholder="sha256:…" /></label></>}
        </div>
      </div>
      <div className="settings-section compact-settings">
        <div><SlidersHorizontal size={20} /><div><h3>Application</h3><p>Automatic startup and background operation.</p></div></div>
        <div className="toggle-list">
          <Toggle label="Start with the system" checked={draft.launchAtLogin} onChange={(value) => update("launchAtLogin", value)} />
          <Toggle label="Keep running in the tray when closed" checked={draft.closeToTray} onChange={(value) => update("closeToTray", value)} />
          <Toggle label="Contribute resources" checked={draft.contributionEnabled} onChange={(value) => update("contributionEnabled", value)} />
        </div>
      </div>
      <div className="settings-section">
        <div><Download size={20} /><div><h3>Automatic updates</h3><p>New Windows versions download in the background and install on restart.</p></div></div>
        <div className="update-settings">
          <div>
            <span className={`update-state ${snapshot.update.state}`}>{updateStateLabel(snapshot.update.state)}</span>
            <strong>Version {snapshot.appVersion}</strong>
            <p>{snapshot.update.message}</p>
          </div>
          <div className="update-actions">
            <button className="secondary-button" disabled={checkingUpdate || snapshot.update.state === "checking" || snapshot.update.state === "downloading"} onClick={() => void checkUpdate()}>
              <RefreshCw className={checkingUpdate ? "spin" : ""} size={15} /> Check now
            </button>
            {snapshot.update.state === "ready" && (
              <button className="primary-button" onClick={() => void window.mycellios.installUpdate()}>
                <Download size={15} /> Restart and update
              </button>
            )}
          </div>
        </div>
      </div>
      <div className="settings-footer"><button className="primary-button" disabled={saving} onClick={() => void save()}>{saving ? <LoaderCircle className="spin" size={17} /> : <Check size={17} />} Save and reconnect</button></div>
    </section>
  );
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

function Onboarding({ snapshot, onComplete }: { snapshot: DashboardSnapshot; onComplete: (snapshot: DashboardSnapshot) => void }) {
  const [mode, setMode] = useState<"local" | "remote">(snapshot.settings.coordinatorMode);
  const [remoteUrl, setRemoteUrl] = useState(snapshot.settings.remoteCoordinatorUrl);
  const [contribute, setContribute] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function complete() {
    setSaving(true);
    setError(null);
    try {
      onComplete(await window.mycellios.saveSettings({
        ...snapshot.settings,
        coordinatorMode: mode,
        remoteCoordinatorUrl: remoteUrl,
        remoteCoordinatorToken: "",
        contributionEnabled: contribute,
        onboardingComplete: true,
      }));
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="modal-backdrop">
      <div className="onboarding-card">
        <div className="onboarding-brand"><img src={brandIcon} alt="" /><span>GET STARTED</span></div>
        <h1>Connect this machine to mycellios</h1>
        <p>The application will configure the coordinator and detect the hardware automatically.</p>
        <div className="mode-grid">
          <button className={mode === "local" ? "selected" : ""} onClick={() => setMode("local")}><Server size={23} /><strong>Create a local network</strong><span>This machine hosts the coordinator.</span>{mode === "local" && <Check size={16} className="mode-check" />}</button>
          <button className={mode === "remote" ? "selected" : ""} onClick={() => setMode("remote")}><Globe2 size={23} /><strong>Join a network</strong><span>Connect over HTTPS/WSS.</span>{mode === "remote" && <Check size={16} className="mode-check" />}</button>
        </div>
        {mode === "remote" && <label className="onboarding-input">Public coordinator URL<input value={remoteUrl} onChange={(event) => setRemoteUrl(event.target.value)} /></label>}
        <label className="contribute-choice"><input type="checkbox" checked={contribute} onChange={(event) => setContribute(event.target.checked)} /><span><strong>Contribute this machine's resources</strong><small>Start with a safe connectivity test; you can switch to local model runtime later.</small></span></label>
        {error && <div className="inline-error"><CircleAlert size={17} /> {error}</div>}
        <button className="primary-button onboarding-submit" disabled={saving} onClick={() => void complete()}>{saving ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={18} />} Enter mycellios <ChevronRight size={18} /></button>
        <div className="onboarding-security"><ShieldCheck size={15} /> Remote connections without TLS are rejected automatically.</div>
      </div>
    </div>
  );
}

function InspectorHeader({ icon: Icon, eyebrow, title }: { icon: typeof Server; eyebrow: string; title: string }) {
  return <div className="inspector-header"><div className="inspector-icon"><Icon size={20} /></div><div><span className="eyebrow">{eyebrow}</span><h2>{title}</h2></div></div>;
}

function StatusCard({ status, connected }: { status: string; connected: boolean }) {
  return <div className={`status-card ${status}`}><div><span className={`status-dot ${connected ? "online" : "offline"}`} /><strong>{status === "online" ? "Node operational" : status}</strong></div><span>{connected ? "WebSocket active" : "No active channel"}</span></div>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function MiniStat({ icon: Icon, value, label }: { icon: typeof Server; value: string; label: string }) {
  return <div className="mini-stat"><Icon size={17} /><div><strong>{value}</strong><span>{label}</span></div></div>;
}

function PageHeading({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return <div className="page-heading"><span className="eyebrow">{eyebrow}</span><h1>{title}</h1><p>{description}</p></div>;
}

function EmptyState({ icon: Icon, title, text }: { icon: typeof Boxes; title: string; text: string }) {
  return <div className="empty-state"><Icon size={28} /><h3>{title}</h3><p>{text}</p></div>;
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return <label className="toggle-row"><span>{label}</span><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><i /></label>;
}

function ErrorBanner({ message, onClose }: { message: string; onClose: () => void }) {
  return <div className="error-banner"><CircleAlert size={17} /><span>{message}</span><button onClick={onClose}><X size={15} /></button></div>;
}

function formatMemory(megabytes: number): string {
  if (megabytes >= 1_024) return `${(megabytes / 1_024).toFixed(megabytes >= 10_240 ? 0 : 1)} GB`;
  return `${Math.round(megabytes)} MB`;
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
