import {
  Activity,
  CheckCircle2,
  Cpu,
  Gauge,
  MonitorSmartphone,
  Power,
  Radio,
  ShieldCheck,
  Zap,
} from "lucide-react";
import { useEffect, useState } from "react";
import {
  getMobileWorkerSnapshot,
  initializeMobileWorker,
  setMobileWorkerLevel,
  subscribeMobileWorker,
  toggleMobileWorker,
  type MobileWorkerLevel,
  type MobileWorkerSnapshot,
} from "../../src/mobile/runtime";
import brandIcon from "./assets/mycellios-app-icon-v2.png";

export function Contribute() {
  const [worker, setWorker] = useState<MobileWorkerSnapshot>(() => getMobileWorkerSnapshot());

  useEffect(() => {
    initializeMobileWorker();
    return subscribeMobileWorker(setWorker);
  }, []);

  return (
    <section className={`contribute-page ${worker.running ? "is-running" : ""}`}>
      <div className="panel-page-title">
        <div>
          <span>THIS DEVICE · ONE CELL</span>
          <h1>Contribute power</h1>
          <p>Turn this device into a verified compute cell while you keep mycellios open.</p>
        </div>
        <div className={`contribute-connection ${worker.connection}`}>
          <i />
          <span>{worker.connectionLabel}</span>
        </div>
      </div>

      <article className="contribute-hero">
        <div className="contribute-cell" aria-hidden="true">
          <i className="cell-ring ring-one" />
          <i className="cell-ring ring-two" />
          <img src={brandIcon} alt="" />
          <span className="cell-pulse pulse-one" />
          <span className="cell-pulse pulse-two" />
        </div>
        <div className="contribute-hero-copy">
          <span className="card-kicker">VOLUNTARY COMPUTE</span>
          <h2>{worker.running ? "Your cell is live." : "Add this device to the network."}</h2>
          <p>{worker.statusText}</p>
          <button
            className={`contribute-toggle ${worker.running ? "stop" : "start"}`}
            disabled={worker.starting}
            onClick={() => void toggleMobileWorker()}
          >
            {worker.starting ? <Radio className="spin" size={19} /> : worker.running ? <Power size={19} /> : <Zap size={19} />}
            {worker.starting ? "Preparing device…" : worker.running ? "Stop contributing" : "Contribute power"}
          </button>
          <small><ShieldCheck size={14} /> You stay in control. Contribution can be stopped at any time.</small>
        </div>
      </article>

      <div className="contribute-metrics" aria-label="Device contribution metrics">
        <WorkerMetric icon={Cpu} label="Engine" value={worker.backend === "webgpu" ? "WebGPU" : "CPU"} detail={worker.gpuLabel} />
        <WorkerMetric icon={Gauge} label="Performance" value={formatGflops(worker.estimatedGflops)} detail="estimated GFLOPS" />
        <WorkerMetric icon={CheckCircle2} label="Verified" value={String(worker.verifiedTasks)} detail="real tasks" />
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
            <span className={`contribute-activity-dot ${worker.running ? "active" : ""}`} />
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
