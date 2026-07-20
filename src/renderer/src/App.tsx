import {
  Activity,
  Boxes,
  Check,
  ChevronRight,
  CircleAlert,
  CirclePower,
  Cpu,
  Gauge,
  Globe2,
  HardDrive,
  Info,
  Laptop,
  LoaderCircle,
  MemoryStick,
  MessageSquareText,
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
} from "../../desktop/contracts";
import brandIcon from "../assets/mycellios-icon.png";
import { NetworkGraph, type GraphSelection } from "./NetworkGraph";

type View = "network" | "hardware" | "models" | "chat" | "settings";

const EMPTY_MESSAGE = "No se pudo cargar el estado de mycellios.";

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
          <SettingsView snapshot={snapshot} onSaved={setSnapshot} />
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
      <p>Preparando la red distribuida</p>
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
        <img src={brandIcon} alt="Logotipo de mycellios" />
        <div>
          <strong>mycellios</strong>
          <span>distributed intelligence</span>
        </div>
      </div>
      <div className="network-pill">
        <span className={`status-dot ${connected ? "online" : "offline"}`} />
        <span>{connected ? "Red conectada" : "Sin conexión"}</span>
        <span className="network-pill-divider" />
        <span>{snapshot.coordinatorUrl.replace(/^https?:\/\//, "")}</span>
      </div>
      <div className="titlebar-actions">
        <span className="version-label">v{snapshot.appVersion}</span>
        <button className="icon-button" onClick={onRefresh} aria-label="Actualizar estado">
          <RefreshCw className={loading ? "spin" : ""} size={17} />
        </button>
      </div>
    </header>
  );
}

function Sidebar({ view, setView }: { view: View; setView: (view: View) => void }) {
  const items: Array<{ id: View; label: string; icon: typeof Network }> = [
    { id: "network", label: "Red", icon: Network },
    { id: "hardware", label: "Equipo", icon: Cpu },
    { id: "models", label: "Modelos", icon: Boxes },
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
        title="Configuración"
      >
        <Settings size={20} />
        <span>Ajustes</span>
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
            <h1>Mapa de conexiones</h1>
          </div>
          <div className="graph-controls">
            <button className="segmented active"><Network size={15} /> Red física</button>
            <button className="segmented"><Activity size={15} /> Ruta activa</button>
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
          <span><i className="legend-dot cyan" /> Este equipo</span>
          <span><i className="legend-dot green" /> Online</span>
          <span><i className="legend-dot purple" /> Modelo</span>
          <span><i className="legend-dot grey" /> Offline</span>
        </div>
        <div className="network-stats-strip">
          <MiniStat icon={Server} value={String(online)} label="nodos online" />
          <MiniStat icon={MemoryStick} value={formatMemory(totalVram)} label="VRAM compartida" />
          <MiniStat icon={Boxes} value={String(snapshot.models.length)} label="modelos" />
          <MiniStat
            icon={Wifi}
            value={snapshot.health ? `${snapshot.health.workers.connected}` : "0"}
            label="conexiones WSS"
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
        <InspectorHeader icon={Server} eyebrow="NODO DE RED" title={selection.label} />
        <StatusCard status={worker.status} connected={worker.connected} />
        <div className="metric-grid">
          <Metric label="VRAM ofrecida" value={formatMemory(worker.offeredVramMb)} />
          <Metric label="Fiabilidad" value={`${Math.round(worker.reliability * 100)}%`} />
          <Metric label="Trabajos" value={String(worker.jobsCompleted)} />
          <Metric label="Región" value={worker.region} />
        </div>
        {gpu && (
          <div className="detail-card">
            <span className="card-kicker">ACELERADOR</span>
            <h3>{gpu.model}</h3>
            <div className="detail-row"><Gauge size={15} /> {gpu.utilizationPct ?? 0}% utilización</div>
            <div className="detail-row"><Thermometer size={15} /> {gpu.temperatureC ?? "—"} °C</div>
            <div className="detail-row"><Zap size={15} /> {gpu.powerW ?? "—"} W</div>
          </div>
        )}
        <div className="detail-card">
          <span className="card-kicker">DESPLIEGUES</span>
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
        <InspectorHeader icon={Boxes} eyebrow="MODELO DISTRIBUIDO" title={selection.model.id} />
        <div className="hero-metric-card purple-card">
          <span>Disponibilidad</span>
          <strong>{selection.model.replicas + selection.model.pipelines}</strong>
          <small>rutas utilizables</small>
        </div>
        <div className="metric-grid">
          <Metric label="Réplicas" value={String(selection.model.replicas)} />
          <Metric label="Pipelines" value={String(selection.model.pipelines)} />
        </div>
      </aside>
    );
  }

  if (selection.kind === "device") {
    const gpu = snapshot.localHardware.gpus[0];
    return (
      <aside className="inspector">
        <InspectorHeader icon={Laptop} eyebrow="ESTE EQUIPO" title={snapshot.localHardware.hostname} />
        <div className={`contribution-card ${snapshot.settings.contributionEnabled ? "active" : ""}`}>
          <div>
            <span className="card-kicker">CONTRIBUCIÓN</span>
            <strong>{snapshot.settings.contributionEnabled ? "Activa" : "Pausada"}</strong>
          </div>
          <button className="power-button" onClick={onToggleContribution}>
            <CirclePower size={21} />
          </button>
        </div>
        <div className="metric-grid">
          <Metric label="RAM" value={formatMemory(snapshot.localHardware.ramMb)} />
          <Metric label="Cuota VRAM" value={formatMemory(snapshot.settings.offeredVramMb)} />
        </div>
        <div className="detail-card">
          <span className="card-kicker">HARDWARE DETECTADO</span>
          <h3>{gpu?.model ?? "GPU no identificada"}</h3>
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
        <span>Estado global</span>
        <strong>{snapshot.health?.workers.online ?? 0}</strong>
        <small>nodos preparados</small>
      </div>
      <div className="metric-grid">
        <Metric label="Registrados" value={String(snapshot.health?.workers.registered ?? 0)} />
        <Metric label="Conectados" value={String(snapshot.health?.workers.connected ?? 0)} />
        <Metric label="Modelos" value={String(snapshot.models.length)} />
        <Metric label="Versión API" value={snapshot.health?.version ?? "—"} />
      </div>
      <div className="detail-card security-card">
        <ShieldCheck size={22} />
        <div>
          <h3>Canal protegido</h3>
          <p>{snapshot.settings.coordinatorMode === "remote" ? "HTTPS/WSS obligatorio" : "Loopback local aislado"}</p>
        </div>
      </div>
      <button className="wide-button" onClick={onToggleContribution}>
        <CirclePower size={17} />
        {snapshot.settings.contributionEnabled ? "Pausar este nodo" : "Aportar este equipo"}
      </button>
    </aside>
  );
}

function HardwareView({ snapshot }: { snapshot: DashboardSnapshot }) {
  return (
    <section className="content-page">
      <PageHeading eyebrow="LOCAL NODE" title="Este equipo" description="Hardware detectado y cuota que mycellios puede utilizar." />
      <div className="hardware-hero">
        <div className="device-orb"><Cpu size={42} /></div>
        <div>
          <span className="eyebrow">{snapshot.platform.toUpperCase()}</span>
          <h2>{snapshot.localHardware.hostname}</h2>
          <p>{snapshot.localHardware.gpus.length} acelerador(es) · {formatMemory(snapshot.localHardware.ramMb)} RAM</p>
        </div>
        <span className={`state-badge ${snapshot.settings.contributionEnabled ? "online" : "paused"}`}>
          {snapshot.settings.contributionEnabled ? "Contribuyendo" : "Pausado"}
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
              <Metric label="VRAM física" value={formatMemory(gpu.physicalVramMb)} />
              <Metric label="Compartida" value={formatMemory(gpu.sharedMemoryMb ?? 0)} />
              <Metric label="Utilización" value={`${gpu.utilizationPct ?? 0}%`} />
              <Metric label="Temperatura" value={gpu.temperatureC === undefined ? "—" : `${gpu.temperatureC} °C`} />
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
      <PageHeading eyebrow="MODEL FABRIC" title="Modelos disponibles" description="Réplicas y pipelines anunciados por los nodos conectados." />
      <div className="data-table">
        <div className="table-head"><span>Modelo</span><span>Réplicas</span><span>Pipelines</span><span>Estado</span></div>
        {snapshot.models.length === 0 ? (
          <EmptyState icon={Boxes} title="Todavía no hay modelos" text="Conecta un nodo con local model runtime o un runtime compatible para anunciar su primer modelo." />
        ) : snapshot.models.map((model) => (
          <div className="table-row" key={model.id}>
            <span className="model-name"><span className="model-glyph"><Sparkles size={16} /></span>{model.id}</span>
            <span>{model.replicas}</span>
            <span>{model.pipelines}</span>
            <span className="state-badge online"><Check size={13} /> Disponible</span>
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
      <PageHeading eyebrow="INFERENCE CONSOLE" title="Probar la red" description="Envía una petición real a un modelo disponible." />
      <div className="chat-console">
        <div className="chat-toolbar">
          <label>Modelo</label>
          <select value={model} onChange={(event) => setModel(event.target.value)}>
            <option value="">Selecciona un modelo</option>
            {snapshot.models.map((item) => <option key={item.id} value={item.id}>{item.id}</option>)}
          </select>
        </div>
        <div className="chat-transcript">
          {!result && !error && <EmptyState icon={MessageSquareText} title="Consola preparada" text="Elige un modelo y pregunta algo para verificar la ruta completa." />}
          {error && <div className="chat-error"><CircleAlert size={18} /> {error}</div>}
          {result && (
            <div className="assistant-message">
              <div className="assistant-avatar"><Sparkles size={17} /></div>
              <div><span>mycellios · {result.model}</span><p>{result.text}</p><small>{result.totalTokens} tokens · ruta {result.routeClass}</small></div>
            </div>
          )}
        </div>
        <div className="chat-composer">
          <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Pregunta a la red mycellios…" />
          <button className="send-button" disabled={sending || !model || !prompt.trim()} onClick={() => void submit()}>
            {sending ? <LoaderCircle className="spin" size={19} /> : <Send size={19} />}
          </button>
        </div>
      </div>
    </section>
  );
}

function SettingsView({ snapshot, onSaved }: { snapshot: DashboardSnapshot; onSaved: (snapshot: DashboardSnapshot) => void }) {
  const [draft, setDraft] = useState(snapshot.settings);
  const [saving, setSaving] = useState(false);
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

  return (
    <section className="content-page settings-page">
      <PageHeading eyebrow="PREFERENCES" title="Configuración" description="Conexión, contribución y comportamiento de la aplicación." />
      {error && <div className="inline-error"><CircleAlert size={17} /> {error}</div>}
      <div className="settings-section">
        <div><Globe2 size={20} /><div><h3>Coordinador</h3><p>Crea una red local o conecta este equipo a una red remota.</p></div></div>
        <div className="settings-fields">
          <label>Modo<select value={draft.coordinatorMode} onChange={(event) => update("coordinatorMode", event.target.value as DesktopSettings["coordinatorMode"])}><option value="local">Red local en este equipo</option><option value="remote">Unirse a coordinador remoto</option></select></label>
          {draft.coordinatorMode === "remote" && <label>URL segura<input value={draft.remoteCoordinatorUrl} onChange={(event) => update("remoteCoordinatorUrl", event.target.value)} placeholder="https://network.mycellios.app" /></label>}
          <label>Región<input value={draft.region} onChange={(event) => update("region", event.target.value)} placeholder="auto" /></label>
        </div>
      </div>
      <div className="settings-section">
        <div><Gauge size={20} /><div><h3>Contribución</h3><p>Define el runtime y el límite de memoria que se ofrece.</p></div></div>
        <div className="settings-fields">
          <label>Runtime<select value={draft.adapterMode} onChange={(event) => update("adapterMode", event.target.value as DesktopSettings["adapterMode"])}><option value="connectivity-test">Prueba de conectividad</option><option value="local-model-runtime">local model runtime local</option></select></label>
          <label>VRAM ofrecida (MB)<input type="number" min="512" step="256" value={draft.offeredVramMb} onChange={(event) => update("offeredVramMb", Number(event.target.value))} /></label>
          {draft.adapterMode === "local-model-runtime" && <><label>Modelo local model runtime<input value={draft.modelName} onChange={(event) => update("modelName", event.target.value)} placeholder="qwen3:4b" /></label><label>URL de local model runtime<input value={draft.adapterBaseUrl} onChange={(event) => update("adapterBaseUrl", event.target.value)} /></label><label>Digest fijado<input value={draft.modelDigest} onChange={(event) => update("modelDigest", event.target.value)} placeholder="sha256:…" /></label></>}
        </div>
      </div>
      <div className="settings-section compact-settings">
        <div><SlidersHorizontal size={20} /><div><h3>Aplicación</h3><p>Inicio automático y ejecución en segundo plano.</p></div></div>
        <div className="toggle-list">
          <Toggle label="Iniciar con el sistema" checked={draft.launchAtLogin} onChange={(value) => update("launchAtLogin", value)} />
          <Toggle label="Seguir en la bandeja al cerrar" checked={draft.closeToTray} onChange={(value) => update("closeToTray", value)} />
          <Toggle label="Aportar recursos" checked={draft.contributionEnabled} onChange={(value) => update("contributionEnabled", value)} />
        </div>
      </div>
      <div className="settings-footer"><button className="primary-button" disabled={saving} onClick={() => void save()}>{saving ? <LoaderCircle className="spin" size={17} /> : <Check size={17} />} Guardar y reconectar</button></div>
    </section>
  );
}

function Onboarding({ snapshot, onComplete }: { snapshot: DashboardSnapshot; onComplete: (snapshot: DashboardSnapshot) => void }) {
  const [mode, setMode] = useState<"local" | "remote">("local");
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
        <div className="onboarding-brand"><img src={brandIcon} alt="" /><span>PUESTA EN MARCHA</span></div>
        <h1>Conecta este equipo a mycellios</h1>
        <p>La aplicación configurará el coordinador y detectará el hardware automáticamente.</p>
        <div className="mode-grid">
          <button className={mode === "local" ? "selected" : ""} onClick={() => setMode("local")}><Server size={23} /><strong>Crear red local</strong><span>Este equipo aloja el coordinador.</span>{mode === "local" && <Check size={16} className="mode-check" />}</button>
          <button className={mode === "remote" ? "selected" : ""} onClick={() => setMode("remote")}><Globe2 size={23} /><strong>Unirse a una red</strong><span>Conecta mediante HTTPS/WSS.</span>{mode === "remote" && <Check size={16} className="mode-check" />}</button>
        </div>
        {mode === "remote" && <label className="onboarding-input">URL del coordinador<input value={remoteUrl} onChange={(event) => setRemoteUrl(event.target.value)} /></label>}
        <label className="contribute-choice"><input type="checkbox" checked={contribute} onChange={(event) => setContribute(event.target.checked)} /><span><strong>Aportar recursos de este equipo</strong><small>Empieza con una prueba segura de conectividad; podrás cambiar a local model runtime después.</small></span></label>
        {error && <div className="inline-error"><CircleAlert size={17} /> {error}</div>}
        <button className="primary-button onboarding-submit" disabled={saving} onClick={() => void complete()}>{saving ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={18} />} Entrar en mycellios <ChevronRight size={18} /></button>
        <div className="onboarding-security"><ShieldCheck size={15} /> Las conexiones remotas sin TLS se rechazan automáticamente.</div>
      </div>
    </div>
  );
}

function InspectorHeader({ icon: Icon, eyebrow, title }: { icon: typeof Server; eyebrow: string; title: string }) {
  return <div className="inspector-header"><div className="inspector-icon"><Icon size={20} /></div><div><span className="eyebrow">{eyebrow}</span><h2>{title}</h2></div></div>;
}

function StatusCard({ status, connected }: { status: string; connected: boolean }) {
  return <div className={`status-card ${status}`}><div><span className={`status-dot ${connected ? "online" : "offline"}`} /><strong>{status === "online" ? "Nodo operativo" : status}</strong></div><span>{connected ? "WebSocket activo" : "Sin canal activo"}</span></div>;
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
