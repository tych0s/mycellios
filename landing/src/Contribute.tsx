import {
  Activity,
  CheckCircle2,
  Cpu,
  Gauge,
  MonitorSmartphone,
  Power,
  Radio,
  ShieldCheck,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  getMobileWorkerSnapshot,
  initializeMobileWorker,
  setMobileWorkerLevel,
  subscribeMobileWorker,
  toggleMobileWorker,
  type MobileWorkerLevel,
  type MobileWorkerSnapshot,
} from "../../src/mobile/runtime";

export function Contribute() {
  const [worker, setWorker] = useState<MobileWorkerSnapshot>(() => getMobileWorkerSnapshot());
  const [handoffRequested, setHandoffRequested] = useState(false);
  const startButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    initializeMobileWorker();
    const unsubscribe = subscribeMobileWorker(setWorker);
    let focusFrame: number | null = null;
    const location = new URL(window.location.href);
    if (location.searchParams.get("autostart") === "1") {
      // This legacy landing-page hint may focus the explicit consent control,
      // but it must never start CPU or GPU work without a real user click.
      location.searchParams.delete("autostart");
      window.history.replaceState(
        window.history.state,
        "",
        `${location.pathname}${location.search}${location.hash}`,
      );
      setHandoffRequested(true);
      focusFrame = window.requestAnimationFrame(() => startButtonRef.current?.focus());
    }
    return () => {
      unsubscribe();
      if (focusFrame !== null) window.cancelAnimationFrame(focusFrame);
    };
  }, []);

  const connected = worker.running && worker.connection === "online";
  const connecting = worker.starting || worker.connection === "connecting";
  const needsAttention = !worker.running && worker.connectionLabel === "Unavailable";
  const contributionLabel = connected
    ? "Connected"
    : needsAttention
      ? "Needs attention"
      : connecting || worker.running
        ? worker.connectionLabel
        : "Paused";
  const badgeState = connected ? "online" : needsAttention ? "error" : connecting ? "connecting" : "paused";
  const actionLabel = worker.starting
    ? "Preparing…"
    : worker.running
      ? "Pause contribution"
      : "Start contributing";
  const statusTitle = connected
    ? "Browser contribution active"
    : needsAttention
      ? "Browser worker unavailable"
      : connecting
        ? `${worker.connectionLabel} browser worker`
        : handoffRequested
          ? "Confirm before contributing"
          : "Browser contribution paused";
  const statusText = handoffRequested && !worker.running && !needsAttention
    ? "Nothing is running yet. Select Start contributing to allow CPU or GPU work on this device."
    : worker.statusText;

  return (
    <section className={`contribute-page ${connected ? "is-running" : ""}`}>
      <div className="panel-page-title">
        <div>
          <span>BROWSER NODE</span>
          <h1>This device</h1>
          <p>Hardware available to mycellios from this browser.</p>
        </div>
        <div className="page-actions">
          <button
            ref={startButtonRef}
            disabled={worker.starting}
            onClick={() => {
              setHandoffRequested(false);
              void toggleMobileWorker();
            }}
          >
            {worker.starting ? <Radio className="spin" size={16} /> : <Power size={16} />}
            {actionLabel}
          </button>
        </div>
      </div>

      <div className="hardware-hero browser-hardware-hero">
        <div className="device-orb"><MonitorSmartphone size={36} /></div>
        <div>
          <span className="eyebrow">BROWSER</span>
          <h2>Browser worker</h2>
          <p>{worker.backend === "webgpu" ? "WebGPU" : "CPU"} · {worker.gpuLabel}</p>
        </div>
        <span className={`state-badge ${badgeState}`}>{contributionLabel}</span>
      </div>

      <div className="browser-contribution-status">
        <Activity size={16} />
        <div><strong>{statusTitle}</strong><span>{statusText}</span></div>
        <small>This browser worker only remains available while this tab stays open and visible.</small>
      </div>

      <div className="contribute-metrics" aria-label="Device contribution metrics">
        <WorkerMetric icon={Cpu} label="Engine" value={worker.backend === "webgpu" ? "WebGPU" : "CPU"} detail={worker.gpuLabel} />
        <WorkerMetric icon={Gauge} label="Performance" value={formatGflops(worker.estimatedGflops)} detail="estimated GFLOPS" />
        <WorkerMetric icon={CheckCircle2} label="Checked" value={String(worker.verifiedTasks)} detail="matrix checks or replicated consensus" />
        <WorkerMetric icon={MonitorSmartphone} label="Display" value={worker.wakeLockActive ? "Awake" : "Normal"} detail="Wake Lock" />
      </div>

      <div className="contribute-detail-grid">
        <article className="contribute-settings-card">
          <div className="contribute-card-head">
            <div><span className="card-kicker">POWER LEVEL</span><h2>Resource profile</h2></div>
            <Cpu size={19} />
          </div>
          <p>Higher power completes work sooner, but increases energy use and temperature.</p>
          <div className="contribute-levels" role="group" aria-label="Power level">
            {(["low", "balanced", "maximum"] as MobileWorkerLevel[]).map((level) => (
              <button
                key={level}
                className={worker.level === level ? "active" : ""}
                disabled={worker.running || worker.starting}
                onClick={() => setMobileWorkerLevel(level)}
              >{level[0]?.toUpperCase()}{level.slice(1)}</button>
            ))}
          </div>
          <div className="contribute-note"><ShieldCheck size={17} /><span><strong>Private by design</strong>Only compute tasks and capability measurements leave this device.</span></div>
        </article>

        <article className="contribute-activity-card">
          <div className="contribute-card-head">
            <div><span className="card-kicker">LIVE ACTIVITY</span><h2>Cell events</h2></div>
            <span className={`contribute-activity-dot ${connected ? "active" : ""}`} />
          </div>
          <ol>
            {worker.activity.map((entry, index) => <li key={`${entry}-${index}`}><Activity size={14} /><span>{entry}</span></li>)}
          </ol>
          <footer>Work pauses automatically when this page is hidden or suspended.</footer>
        </article>
      </div>
    </section>
  );
}

function WorkerMetric({ icon: Icon, label, value, detail }: { icon: typeof Cpu; label: string; value: string; detail: string }) {
  return <article><div><Icon size={19} /></div><span>{label}</span><strong>{value}</strong><small>{detail}</small></article>;
}

function formatGflops(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "—";
  if (value < 1) return value.toFixed(2);
  if (value < 100) return value.toFixed(1);
  return Math.round(value).toLocaleString();
}
