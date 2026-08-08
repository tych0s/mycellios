import {
  ArrowDown,
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  Boxes,
  Check,
  ChevronDown,
  ChevronRight,
  CircleDot,
  CirclePlay,
  Coins,
  Cpu,
  Download,
  Gauge,
  GitBranch,
  HardDriveDownload,
  Layers3,
  MemoryStick,
  Menu,
  MonitorDown,
  Network,
  Pause,
  Play,
  Radio,
  Route,
  ShieldCheck,
  Sparkles,
  Users,
  Workflow,
  X,
  Zap,
} from "lucide-react";
import { lazy, Suspense, useEffect, useRef, useState, type CSSProperties } from "react";
import efficiencyCurve from "../assets/efficiency-curve-ai.webp";
import storyImage from "../assets/mycelium-story.webp";
import { applySeoMetadata } from "../seo";
import { SupportAssistant } from "../SupportAssistant";
import heroImage from "./assets/mycellios-rebrand-hero.webp";
import liveImage from "./assets/mycellios-live-network.webp";
import "./rebrand.css";

const brandLogo = "/assets/logos/logo.png";
const EMAIL = "hello@mycellios.com";
const RELEASES_URL = "https://github.com/tych0s/mycellios";
const ExplainerVideo = lazy(() => import("../ExplainerVideo"));

const steps = [
  { id: "01", title: "Your device", copy: "Choose how much capacity to share. You stay in control." },
  { id: "02", title: "A worker sprouts", copy: "mycellios detects the hardware and turns it into usable capacity." },
  { id: "03", title: "It reaches the network", copy: "The node finds a compatible cell by memory, latency, and availability." },
  { id: "04", title: "It runs one part", copy: "Each worker receives only the model stage it can process." },
  { id: "05", title: "One answer returns", copy: "The network gathers the result and returns it as a single stream." },
] as const;

const workers = [
  { species: "Small", name: "Home computer", hardware: "8–16 GB · CPU / iGPU", model: "Compact models", load: "Light", group: "single" },
  { species: "Robust", name: "Dedicated GPU", hardware: "12–24 GB VRAM", model: "Accelerated stages", load: "Medium", group: "strong" },
  { species: "Colony", name: "Farm or community", hardware: "Multiple machines", model: "Distributed models", load: "Coordinated", group: "cluster" },
  { species: "Dormant", name: "Offline node", hardware: "Reserved capacity", model: "Receives no tasks", load: "Paused", group: "dormant" },
] as const;

const hardwareProfiles = {
  "integrated": { label: "Mycelium worker", type: "Home computer", models: "Compact models and support tasks", compute: "Light contribution", capacity: "System memory" },
  "rtx-3060": { label: "Chanterelle worker", type: "Dedicated GPU", models: "Quantized stages up to 12 GB", compute: "Medium contribution", capacity: "12 GB VRAM" },
  "rtx-4090": { label: "Boletus worker", type: "High-performance GPU", models: "Large stages and accelerated routes", compute: "High contribution", capacity: "24 GB VRAM" },
  "multi": { label: "Distributed colony", type: "Farm or community", models: "Models split across multiple nodes", compute: "Coordinated contribution", capacity: "Combined capacity" },
} as const;

const heroSignals = [
  { worker: "worker_04", status: "Compatible route found" },
  { worker: "worker_12", status: "Stage assigned to cell" },
  { worker: "worker_07", status: "Response gathered at origin" },
] as const;

const explainerSteps = [
  ["01", "Offer capacity", "Choose what each device can contribute."],
  ["02", "Connect nodes", "Useful machines form one coordinated network."],
  ["03", "Split the model", "Every node holds only its assigned part."],
  ["04", "Stream the answer", "One request returns one continuous response."],
] as const;

const accelerationPhases = [
  { step: "NOW · 01", title: "Fit the model", copy: "Pool heterogeneous memory so models can run beyond the limit of any single machine.", signal: "MEMORY" },
  { step: "NEXT · 02", title: "Shorten the route", copy: "Keep experts resident and move each request through fewer, faster network boundaries.", signal: "LOCALITY" },
  { step: "SCALE · 03", title: "Multiply the paths", copy: "Coordinate parallel cells so added capacity can become useful throughput instead of added latency.", signal: "TOKENS / S" },
] as const;

const pipelineStages = [
  { number: "01", name: "Analyze", detail: "memory · topology" },
  { number: "02", name: "Partition", detail: "layers · experts" },
  { number: "03", name: "Assign", detail: "the best machine" },
  { number: "04", name: "Coordinate", detail: "one route" },
] as const;

const pipelineNodes = ["GPU · 12 GB", "GPU · 4 GB", "CPU · 64 GB", "GPU · 8 GB", "STANDBY"] as const;

const architecturePrinciples = [
  { title1: "One model,", title2: "many memories", copy: "Each node stores and processes only one part. The complete model emerges from collaboration.", label: "PARTITIONING" },
  { title1: "Cells first,", title2: "then the network", copy: "Nearby machines form fast cells. The WAN connects a few virtual stages, not hundreds of hops.", label: "HIERARCHICAL MESH" },
  { title1: "The route changes.", title2: "The work continues.", copy: "The scheduler measures capacity, connectivity, and availability to prepare alternatives as the network changes.", label: "ADAPTIVE ROUTING" },
] as const;

const evidenceRows = [
  ["Partitioned-model runtime", "Real pipeline and compatible API", "READY"],
  ["Heterogeneous scheduler", "Memory, topology, and availability", "READY"],
  ["Stage recovery", "Alternate route and greedy continuation", "LAB"],
  ["Model larger than every node", "Test across 2–4 physical machines", "TESTING"],
] as const;

const downloads = {
  windows: { label: "Windows", detail: "Windows 10/11 · x64", filename: "mycellios-windows-x64.exe" },
  "mac-arm64": { label: "macOS", detail: "Apple Silicon", filename: "mycellios-macos-arm64.dmg" },
  "linux-deb": { label: "Linux", detail: "Ubuntu / Debian · x64", filename: "mycellios-linux-x64.deb" },
  "linux-rpm": { label: "Linux", detail: "Fedora / RHEL · x64", filename: "mycellios-linux-x64.rpm" },
} as const;

type DownloadKey = keyof typeof downloads;

function downloadUrl(key: DownloadKey): string {
  if (key === "windows") return "/downloads/windows";
  if (key === "mac-arm64") return "/downloads/macos-arm64";
  if (key === "linux-deb") return "/downloads/linux-deb";
  if (key === "linux-rpm") return "/downloads/linux-rpm";
  throw new Error(`Unsupported download target: ${String(key)}`);
}

const installSteps = [
  { label: "Install", detail: "Desktop agent added", status: "DONE" },
  { label: "Detect", detail: "CPU · RAM · GPU mapped", status: "DONE" },
  { label: "Choose", detail: "Join or create a network", status: "READY" },
] as const;

const futureCards = {
  improvement: {
    index: "01",
    label: "AUTONOMOUS IMPROVEMENT",
    metric: "0.1",
    unit: "%",
    metricDetail: "OF VERIFIED USEFUL COMPUTE",
    title: "The network improves the network.",
    copy: "Reserve a strictly bounded share of useful compute for infrastructure R&D: regression and failure tests, new scheduling algorithms, and candidate routes or topologies evaluated in isolation.",
    steps: ["Test failures and regressions", "Trial algorithms and routes", "Promote only verified gains"],
    footer: "Bounded budget · auditable history · controlled rollout",
  },
  token: {
    index: "02",
    label: "VERIFIABLE INCENTIVES",
    metric: "TOKEN",
    metricDetail: "FOR ACCEPTED NETWORK WORK",
    title: "Useful compute becomes verifiable value.",
    copy: "Build a crypto token around work the network actually accepts — not advertised capacity or passive connection — with quality, reliability and fraud resistance built in.",
    steps: ["Internal credits first", "Public testnet without monetary value", "Audited launch after legal review"],
    footer: "No active token · no financial promise",
  },
  router: {
    index: "03",
    label: "INTELLIGENT MODEL ROUTING",
    metric: "ONE API",
    metricDetail: "EVERY MODEL · BEST AVAILABLE PATH",
    title: "One request. The right intelligence.",
    copy: "A unified AI router inspired by the model gateways behind products such as OpenRouter and Cursor. It will evaluate the task, quality, cost, latency, privacy and live network capacity before selecting the best model and execution path.",
    steps: ["Unify Mycellios, local and external models", "Choose by quality, price, speed and privacy", "Fail over without changing the application"],
    footer: "Provider-neutral · policy-controlled · every routing decision observable",
    input: "UNIFIED REQUEST",
    engine: "ROUTING ENGINE",
    policies: ["QUALITY", "COST", "LATENCY", "PRIVACY"],
    destinations: ["MYCELLIOS", "LOCAL GPU", "OPEN MODELS", "CLOUD APIs"],
  },
} as const;

const participateCards = [
  {
    label: "FOR CONTRIBUTORS",
    title: "Turn idle capacity into useful capacity.",
    copy: "Connect a PC, workstation, or server. Decide how much you contribute and pause whenever you want.",
    bullets: ["Different hardware, one network", "Voluntary contribution", "Verifiable work"],
  },
  {
    label: "FOR TEAMS AND ORGANIZATIONS",
    title: "Run open models on infrastructure you control.",
    copy: "Create private networks for labs, companies, and communities with distributed hardware.",
    bullets: ["Larger models through pooled memory", "A familiar API for your applications", "Topology adapted to your network"],
  },
] as const;

type HardwareKey = keyof typeof hardwareProfiles;

function Brand() {
  return <a className="rb-brand" href="#rb-top" aria-label="mycellios, home"><img src={brandLogo} alt="" /><span>mycellios</span></a>;
}

function NetworkThreads({ compact = false }: { compact?: boolean }) {
  return (
    <svg className={`rb-threads ${compact ? "is-compact" : ""}`} viewBox="0 0 900 520" aria-hidden="true">
      <g className="rb-thread-lines">
        <path d="M40 350C150 295 190 390 285 310S450 210 535 275s148 42 318-90" />
        <path d="M80 170c100 5 116 102 211 94s137-98 226-52 145 123 305 75" />
        <path d="M120 450c42-112 129-106 196-57s109-8 137-94 110-131 195-84 92 121 173 154" />
        <path d="M172 91c40 82 96 114 164 82s120-11 169 62 99 91 178 41 104-43 161-19" />
      </g>
      {["80,170", "172,91", "285,310", "336,173", "453,299", "517,212", "535,275", "648,215", "683,276", "821,369", "853,185"].map((point, index) => {
        const [cx, cy] = point.split(",");
        return <circle key={point} cx={cx} cy={cy} r={index % 3 === 0 ? 7 : 4} style={{ "--i": index } as CSSProperties} />;
      })}
    </svg>
  );
}

function HeroNetwork() {
  const nodes = [
    [110, 258, "worker_01"], [238, 126, "worker_04"], [346, 372, "worker_07"],
    [476, 206, "worker_12"], [592, 426, "worker_15"], [712, 132, "worker_18"],
    [818, 316, "worker_21"], [690, 514, "worker_24"],
  ] as const;
  return (
    <div className="rb-hero-network" aria-hidden="true">
      <svg viewBox="0 0 900 620" role="presentation">
        <ellipse className="rb-world-orbit orbit-one" cx="505" cy="314" rx="386" ry="238" />
        <ellipse className="rb-world-orbit orbit-two" cx="505" cy="314" rx="280" ry="184" />
        <g className="rb-world-lines">
          <path d="M110 258C220 236 227 118 346 372S527 172 712 132 766 249 818 316" />
          <path d="M238 126c42 72 92 173 238 80s132 37 216 220" />
          <path d="M110 258c105 74 122 168 236 114s145-10 246 54 132 75 198 88" />
          <path d="M346 372c95-32 144-20 246 54s148-22 226-110" />
          <path d="M476 206c-8 88 31 152 116 220s122 40 98 88" />
        </g>
        <g className="rb-world-nodes">
          {nodes.map(([cx, cy, label], index) => <g key={label} className={index === 3 ? "is-origin" : ""} style={{ "--i": index } as CSSProperties}><circle cx={cx} cy={cy} r={index === 3 ? 13 : 8} /><circle className="rb-node-ring" cx={cx} cy={cy} r={index === 3 ? 25 : 17} /><title>{label}</title></g>)}
        </g>
        <circle className="rb-world-pulse" cx="476" cy="206" r="5" />
      </svg>
      <span className="rb-network-label label-origin">origin · request</span>
      <span className="rb-network-label label-north">north cell · ready</span>
      <span className="rb-network-label label-south">south cell · response</span>
      <div className="rb-network-readout"><i /> 08 workers connected <b>·</b> 03 routes active</div>
    </div>
  );
}

function ExplainerSection() {
  const section = useRef<HTMLElement>(null);
  const [shouldLoad, setShouldLoad] = useState(false);

  useEffect(() => {
    const target = section.current;
    if (!target) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry?.isIntersecting) return;
        setShouldLoad(true);
        observer.disconnect();
      },
      { rootMargin: "420px 0px" },
    );
    observer.observe(target);
    return () => observer.disconnect();
  }, []);

  return (
    <section className="rb-explainer" id="explainer" ref={section} aria-labelledby="rb-explainer-title">
      <div className="rb-shell">
        <div className="rb-explainer-heading rb-reveal">
          <p className="rb-kicker"><i /><span>00 / Start here — mycellios in 20 seconds</span></p>
          <h2 id="rb-explainer-title">Several devices.<br /><em>One AI model.</em></h2>
          <p>Each computer contributes only the capacity you choose. mycellios connects that capacity, distributes the model, and returns one streamed answer.</p>
        </div>

        <div className="rb-explainer-player rb-reveal">
          {shouldLoad ? (
            <Suspense fallback={<div className="rb-explainer-loading"><CirclePlay /><span>Preparing the explanation…</span></div>}>
              <ExplainerVideo />
            </Suspense>
          ) : (
            <div className="rb-explainer-loading"><CirclePlay /><span>Preparing the explanation…</span></div>
          )}
        </div>

        <ol className="rb-explainer-steps rb-reveal">
          {explainerSteps.map(([number, title, detail]) => (
            <li key={number}>
              <span>{number}</span>
              <strong>{title}</strong>
              <p>{detail}</p>
            </li>
          ))}
        </ol>
        <p className="rb-explainer-note rb-reveal">Conceptual flow · Real capacity and speed are measured in Network and Tests.</p>
      </div>
    </section>
  );
}

function AccelerationSection() {
  return (
    <section className="rb-accel" id="acceleration" aria-labelledby="rb-accel-title">
      <div className="rb-shell">
        <div className="rb-section-title rb-reveal">
          <p className="rb-kicker">02 — The compounding curve</p>
          <h2 id="rb-accel-title">Intelligence gets cheaper.<br /><em>Distribution makes it go further.</em></h2>
          <p>Model efficiency is already moving at extraordinary speed. mycellios is being built to compound that progress: pooling memory first, shortening routes next, and increasing effective tokens per second as the network matures.</p>
        </div>

        <article className="rb-curve rb-reveal" aria-label="Observed benchmark shift on ARC-AGI-1 between December 2024 and December 2025">
          <img src={efficiencyCurve} alt="An abstract luminous mycelium network accelerating into a dense flow of information" loading="lazy" decoding="async" />
          <div className="rb-curve-veil" />
          <div className="rb-curve-topline"><span>Observed benchmark shift</span><span>ARC-AGI-1 · 12 MONTHS</span></div>
          <span className="rb-curve-axis">CAPABILITY PER DOLLAR</span>
          <svg className="rb-curve-plot" viewBox="0 0 1000 470" preserveAspectRatio="none" aria-hidden="true">
            <defs>
              <linearGradient id="rbEfficiencyLine" x1="0" x2="1"><stop offset="0" stopColor="#70806d" /><stop offset=".55" stopColor="#ad7a48" /><stop offset="1" stopColor="#e4b06e" /></linearGradient>
              <filter id="rbCurveGlow" x="-30%" y="-50%" width="160%" height="200%"><feGaussianBlur stdDeviation="7" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
            </defs>
            <g className="rb-curve-grid">
              {[90, 180, 270, 360].map((y) => <line key={`h-${y}`} x1="80" y1={y} x2="930" y2={y} />)}
              {[160, 350, 540, 730, 920].map((x) => <line key={`v-${x}`} x1={x} y1="55" x2={x} y2="405" />)}
            </g>
            <path className="rb-curve-shadow" d="M150 365 C330 360 475 315 590 240 S750 115 845 82" />
            <path className="rb-curve-line" d="M150 365 C330 360 475 315 590 240 S750 115 845 82" />
            <g className="rb-curve-point point-start"><circle cx="150" cy="365" r="20" /><circle cx="150" cy="365" r="5" /></g>
            <g className="rb-curve-point point-end" filter="url(#rbCurveGlow)"><circle cx="845" cy="82" r="24" /><circle cx="845" cy="82" r="6" /></g>
          </svg>
          <div className="rb-benchmark rb-benchmark-from"><span>DEC 2024</span><strong>o3-preview · high compute</strong><b>&gt;$3,000 / task</b><small>87.5% ARC-AGI-1</small></div>
          <div className="rb-benchmark rb-benchmark-to"><span>DEC 2025</span><strong>Gemini 3 Flash · high</strong><b>$0.231 / task</b><small>84.7% ARC-AGI-1</small></div>
          <div className="rb-curve-multiplier"><strong>≈13,000×</strong><span>lower cost per task</span><ArrowUpRight /></div>
        </article>

        <div className="rb-curve-evidence rb-reveal">
          <p>The same benchmark family, but different systems and evaluation configurations. This is a directional efficiency signal—not an apples-to-apples price comparison.</p>
          <div>
            <a href="https://arcprize.org/blog/oai-o3-pub-breakthrough" target="_blank" rel="noreferrer">o3 analysis <ArrowUpRight /></a>
            <a href="https://arcprize.org/leaderboard?hl=en-US" target="_blank" rel="noreferrer">Inspect the ARC Prize data <ArrowUpRight /></a>
          </div>
        </div>

        <div className="rb-accel-roadmap-head rb-reveal">
          <div><span>The mycellios development curve</span><h3>Fit. Route. Multiply.</h3></div>
          <p>Every layer removes a different bottleneck. The target is a compounding throughput curve, not a single benchmark trick.</p>
          <small>DIRECTION · NOT A PERFORMANCE FORECAST</small>
        </div>
        <div className="rb-accel-roadmap">
          {accelerationPhases.map((phase, index) => (
            <article className="rb-accel-phase rb-reveal" key={phase.step}>
              <div className="rb-phase-line"><i /><span>{phase.step}</span><b>0{index + 1}</b></div>
              <h4>{phase.title}</h4><p>{phase.copy}</p><strong>{phase.signal}</strong>
              {index < accelerationPhases.length - 1 && <ArrowRight className="rb-phase-arrow" />}
            </article>
          ))}
        </div>
        <div className="rb-curve-equation rb-reveal"><i /><span>MODEL EFFICIENCY × NETWORK EFFICIENCY = COMPOUND SYSTEM SPEED</span><i /></div>
      </div>
    </section>
  );
}

function ArchitectureSection() {
  return (
    <section className="rb-arch rb-shell" id="architecture" aria-labelledby="rb-arch-title">
      <div className="rb-section-title rb-reveal">
        <p className="rb-kicker">04 — Adaptive architecture</p>
        <h2 id="rb-arch-title">We do not connect more machines.<br /><em>We build the right route.</em></h2>
        <p>Adding a slow node can make the entire system worse. GDLP/2 analyzes each model, groups nearby resources, and selects only the machines that add value to that run.</p>
      </div>

      <div className="rb-pipeline rb-reveal">
        <div className="rb-pipeline-topline"><span>GDLP / 2</span><span>ROUTE COMPILER</span><span className="rb-pipeline-live"><i />LIVE</span></div>
        <div className="rb-pipeline-model">
          <div className="rb-model-cube"><Boxes /><span>70B</span></div>
          <div><small>TARGET MODEL</small><strong>Too large<br />for a single node</strong></div>
          <ArrowDownRight />
        </div>
        <div className="rb-pipeline-flow">
          {pipelineStages.map((stage, index) => (
            <div className="rb-pipeline-stage" key={stage.number}>
              <span>{stage.number}</span>
              <div><strong>{stage.name}</strong><small>{stage.detail}</small></div>
              {index < pipelineStages.length - 1 && <ChevronRight className="rb-pipeline-arrow" />}
            </div>
          ))}
        </div>
        <div className="rb-pipeline-nodes">
          {pipelineNodes.map((node, index) => (
            <div key={node} className={index === 4 ? "standby" : ""}>
              <i /><span>{node}</span>
              <small>{index === 4 ? "reserve" : `shard ${String.fromCharCode(65 + index)}`}</small>
            </div>
          ))}
        </div>
      </div>

      <div className="rb-arch-principles">
        {architecturePrinciples.map((principle, index) => {
          const Icon = [Layers3, Network, Workflow][index] ?? Network;
          return (
            <article className="rb-arch-card rb-reveal" key={principle.label}>
              <div className="rb-arch-card-top"><Icon /><span>0{index + 1}</span></div>
              <h3>{principle.title1}<br />{principle.title2}</h3>
              <p>{principle.copy}</p>
              <span className="rb-arch-label">{principle.label} <ArrowUpRight /></span>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function EvidenceSection() {
  return (
    <section className="rb-evidence" id="evidence" aria-labelledby="rb-evidence-title">
      <div className="rb-evidence-inner rb-shell">
        <div className="rb-evidence-copy rb-reveal">
          <p className="rb-kicker"><i /><span>07 — Evidence before promises</span></p>
          <h2 id="rb-evidence-title">We are building<br /><em>in public.</em></h2>
          <p>We do not call a simulation a global network. Every important claim passes a reproducible test before it becomes a product promise.</p>
          <div className="rb-evidence-legend"><span><i className="done" />Implemented</span><span><i className="next" />Next physical milestone</span></div>
        </div>
        <div className="rb-board rb-reveal" aria-label="Build status board, dated 20 July 2026">
          <div className="rb-board-head"><span>BUILD STATUS</span><span>20 · 07 · 2026</span></div>
          {evidenceRows.map((row, index) => (
            <div className={`rb-board-row ${index === 3 ? "active" : ""}`} key={row[0]}>
              <i className={index === 3 ? "next" : "done"}>{index === 3 ? <CircleDot /> : <Check />}</i>
              <div><strong>{row[0]}</strong><small>{row[1]}</small></div>
              <span>{row[2]}</span>
            </div>
          ))}
          <div className="rb-board-footer"><span>NEXT GATE</span><strong>Multi-node · GPU · LAN</strong><Gauge /></div>
        </div>
      </div>
    </section>
  );
}

function InstallSection() {
  const [recommended, setRecommended] = useState<DownloadKey>("windows");
  const [macVisitor, setMacVisitor] = useState(false);

  useEffect(() => {
    const agent = navigator.userAgent.toLowerCase();
    const isAppleMobile =
      /iphone|ipad|ipod/.test(agent) ||
      (agent.includes("mac") && navigator.maxTouchPoints > 1);
    if (agent.includes("mac") && !isAppleMobile) {
      setMacVisitor(true);
      setRecommended("mac-arm64");
    }
    else if (agent.includes("linux")) setRecommended("linux-deb");
    else setRecommended("windows");
  }, []);

  const selected = downloads[recommended];

  return (
    <section className="rb-install" id="install" aria-labelledby="rb-install-title">
      <div className="rb-install-layout rb-shell">
        <div className="rb-install-copy rb-reveal">
          <p className="rb-kicker rb-kicker-light"><i /><span>09 — From download to network</span></p>
          <h2 id="rb-install-title">One click.<br /><em>Your machine joins the organism.</em></h2>
          <p>Install the desktop app, choose how you want to participate, and mycellios detects your hardware automatically. No terminal, Docker, or manual configuration required.</p>
          <div className="rb-install-detected"><span><MonitorDown />Recommended for this device</span><strong>{selected.label} · {selected.detail}</strong></div>
          <a className="rb-install-primary" href={downloadUrl(recommended)}>
            <span className="rb-install-icon"><Download /></span>
            <span><small>Download for</small><strong>{selected.label}</strong></span>
            <ArrowDownRight />
          </a>
          <div className="rb-install-meta"><ShieldCheck /><span>Early access · automatic hardware detection</span></div>
        </div>

        <div className="rb-console rb-reveal" aria-label="First run console preview">
          <div className="rb-console-top"><div><i /><i /><i /></div><span>MYCELLIOS · FIRST RUN</span><b>01:18</b></div>
          <div className="rb-console-body">
            <div className="rb-console-orb"><span /><Radio /><i /></div>
            <div className="rb-console-ready"><span><i />NETWORK READY</span><strong>this machine</strong><small>NODE · 7FA3_C2E1</small></div>
            <div className="rb-console-steps">
              {installSteps.map((step, index) => (
                <div key={step.label} style={{ "--install-delay": `${index * .45}s` } as CSSProperties}>
                  <span>0{index + 1}</span><i><Check /></i>
                  <p><strong>{step.label}</strong><small>{step.detail}</small></p>
                  <b>{step.status}</b>
                </div>
              ))}
            </div>
            <div className="rb-console-promise"><ShieldCheck />YOU DECIDE WHAT TO SHARE · PAUSE AT ANY TIME</div>
          </div>
        </div>
      </div>

      <div className="rb-shelf rb-shell rb-reveal">
        <div className="rb-shelf-head"><span>Other downloads</span><a href={RELEASES_URL} target="_blank" rel="noreferrer">Release notes and checksums <ArrowUpRight /></a></div>
        <div className="rb-shelf-options">
          {(Object.keys(downloads) as DownloadKey[]).map((key) => {
            const option = downloads[key];
            return (
              <a className={key === recommended ? "recommended" : ""} href={downloadUrl(key)} key={key}>
                <HardDriveDownload />
                <span><strong>{option.label}</strong><small>{option.detail}</small></span>
                {key === recommended ? <b>RECOMMENDED</b> : <ArrowDownRight />}
              </a>
            );
          })}
        </div>
        {macVisitor && (
          <aside className="rb-mac-guide" aria-labelledby="rb-mac-test-title">
            <div className="rb-mac-guide-head">
              <span><ShieldCheck />MACOS TEST BUILD</span>
              <div>
                <h3 id="rb-mac-test-title">Try mycellios without an Apple developer licence</h3>
                <p>The app is ad-hoc signed and package-attested. macOS still asks you to approve this test build once.</p>
              </div>
              <div className="rb-mac-guide-download">
                <a href={downloadUrl("mac-arm64")}><Cpu /><span><strong>Apple Silicon</strong><small>M1 · M2 · M3 · M4</small></span><Download /></a>
              </div>
            </div>
            <ol>
              <li><b>01</b><span><strong>Install</strong><small>Open the DMG and drag mycellios to Applications.</small></span></li>
              <li><b>02</b><span><strong>Try to open it once</strong><small>macOS shows a security warning. Close that message.</small></span></li>
              <li><b>03</b><span><strong>Open Privacy &amp; Security</strong><small>Go to System Settings → Privacy &amp; Security.</small></span></li>
              <li><b>04</b><span><strong>Choose Open Anyway</strong><small>Confirm once, then open mycellios normally.</small></span></li>
            </ol>
            <footer><Check />No Terminal commands required <i /> Manual downloads remain required for test builds</footer>
          </aside>
        )}
        <p className="rb-install-safety"><ShieldCheck />Every installer is built and package-verified in GitHub Actions. Code signing is being rolled out; early builds may still trigger an operating-system publisher warning.</p>
      </div>
    </section>
  );
}

function RoadmapSection() {
  const { improvement, token, router } = futureCards;
  return (
    <section className="rb-roadmap" id="roadmap" aria-labelledby="rb-roadmap-title">
      <div className="rb-shell">
        <div className="rb-roadmap-heading rb-reveal">
          <div>
            <p className="rb-kicker">10 — The next layer</p>
            <h2 id="rb-roadmap-title">A network that<br /><em>improves itself.</em></h2>
          </div>
          <div className="rb-roadmap-intro">
            <span className="rb-roadmap-status"><CircleDot />VISION · NOT LIVE YET</span>
            <p>Three long-term systems designed to make mycellios more capable every day, route every request intelligently, and return value to the machines that make that progress possible.</p>
          </div>
        </div>

        <div className="rb-roadmap-stage">
          <article className="rb-future-card rb-reveal">
            <header><span>{improvement.index} / {improvement.label}</span><Sparkles /></header>
            <div className="rb-future-metric"><strong>{improvement.metric}</strong><span><b>{improvement.unit}</b><small>{improvement.metricDetail}</small></span></div>
            <h3>{improvement.title}</h3>
            <p>{improvement.copy}</p>
            <ol>{improvement.steps.map((step, index) => <li key={step}><b>0{index + 1}</b><span>{step}</span></li>)}</ol>
            <footer><ShieldCheck /><span>{improvement.footer}</span></footer>
          </article>

          <div className="rb-roadmap-bridge" aria-hidden="true"><i /><span><Zap /></span><b>PROVEN WORK → MEASURABLE VALUE</b></div>

          <article className="rb-future-card rb-reveal">
            <header><span>{token.index} / {token.label}</span><Coins /></header>
            <div className="rb-future-metric word"><strong>{token.metric}</strong><span><small>{token.metricDetail}</small></span></div>
            <h3>{token.title}</h3>
            <p>{token.copy}</p>
            <ol>{token.steps.map((step, index) => <li key={step}><b>0{index + 1}</b><span>{step}</span></li>)}</ol>
            <footer><ShieldCheck /><span>{token.footer}</span></footer>
          </article>
        </div>

        <article className="rb-router rb-reveal">
          <header><span>{router.index} / {router.label}</span><Route /></header>
          <div className="rb-router-content">
            <div className="rb-router-copy">
              <div className="rb-future-metric word"><strong>{router.metric}</strong><span><small>{router.metricDetail}</small></span></div>
              <h3>{router.title}</h3>
              <p>{router.copy}</p>
            </div>
            <div className="rb-router-visual" aria-label="A unified request routed to the best available AI model">
              <div className="rb-router-input"><Radio /><span>{router.input}</span></div>
              <i className="rb-router-line incoming" />
              <div className="rb-router-engine"><Route /><strong>{router.engine}</strong><span>{router.policies.map((policy) => <b key={policy}>{policy}</b>)}</span></div>
              <i className="rb-router-line outgoing" />
              <div className="rb-router-destinations">{router.destinations.map((destination, index) => <span key={destination} style={{ "--route-delay": `${index * .7}s` } as CSSProperties}><i />{destination}</span>)}</div>
            </div>
          </div>
          <ol>{router.steps.map((step, index) => <li key={step}><b>0{index + 1}</b><span>{step}</span></li>)}</ol>
          <footer><ShieldCheck /><span>{router.footer}</span></footer>
        </article>

        <div className="rb-roadmap-principle rb-reveal"><ShieldCheck /><span>Direction, not a promise. Each stage must prove security, usefulness and measurable results before it can reach the public network.</span></div>
      </div>
    </section>
  );
}

export function RebrandLanding() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [heroSignal, setHeroSignal] = useState(0);
  const [flowStep, setFlowStep] = useState(0);
  const [metric, setMetric] = useState(0);
  const [activeWorker, setActiveWorker] = useState(1);
  const [hardware, setHardware] = useState<HardwareKey>("rtx-3060");
  const [paused, setPaused] = useState(false);
  const flowRef = useRef<HTMLElement>(null);
  const pageRef = useRef<HTMLElement>(null);

  useEffect(() => {
    applySeoMetadata("/");
  }, []);

  useEffect(() => {
    const page = pageRef.current;
    if (!page) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      page.querySelectorAll(".rb-reveal").forEach((element) => element.classList.add("is-visible"));
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => entries.forEach((entry) => entry.isIntersecting && entry.target.classList.add("is-visible")),
      { threshold: 0.12 },
    );
    page.querySelectorAll(".rb-reveal").forEach((element) => observer.observe(element));
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const timer = window.setInterval(
      () => setHeroSignal((current) => (current + 1) % heroSignals.length),
      3800,
    );
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const onScroll = () => {
      const section = flowRef.current;
      if (!section) return;
      const rect = section.getBoundingClientRect();
      const distance = Math.max(1, section.offsetHeight - window.innerHeight);
      const progress = Math.min(0.999, Math.max(0, -rect.top / distance));
      setFlowStep(Math.min(4, Math.floor(progress * 5)));
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (paused) return;
    const timer = window.setInterval(() => setMetric((current) => (current + 1) % 3), 3200);
    return () => window.clearInterval(timer);
  }, [paused]);

  useEffect(() => {
    if (!mobileOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobileOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [mobileOpen]);

  const liveMetrics = [
    { value: "5", label: "stages in a distributed run" },
    { value: "1", label: "answer gathered for the user" },
    { value: "0", label: "unmeasured performance claims" },
  ];
  const profile = hardwareProfiles[hardware];
  const currentStep = steps[flowStep] ?? steps[0]!;
  const currentWorker = workers[activeWorker] ?? workers[0]!;
  const currentMetric = liveMetrics[metric] ?? liveMetrics[0]!;
  const currentHeroSignal = heroSignals[heroSignal] ?? heroSignals[0]!;

  const scrollToFlowStep = (index: number) => {
    const section = flowRef.current;
    if (!section) return;
    const distance = Math.max(0, section.offsetHeight - window.innerHeight);
    const stepProgress = (index + 0.5) / steps.length;
    window.scrollTo({
      top: section.offsetTop + distance * stepProgress,
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
    });
  };

  return (
    <main className="rb-page" id="rb-top" ref={pageRef}>
      <header className="rb-header">
        <div className="rb-header-inner rb-shell">
          <Brand />
          <nav aria-label="Main navigation">
            <a href="#how-it-works">How it works</a><a href="#architecture">Architecture</a><a href="#evidence">Evidence</a><a href="#install">Install</a><a href="#roadmap">Roadmap</a>
          </nav>
          <a className="rb-pill rb-pill-dark rb-desktop-cta" href="#hardware">Find your worker <ArrowRight /></a>
          <button className="rb-menu" type="button" aria-label={mobileOpen ? "Close menu" : "Open menu"} aria-expanded={mobileOpen} onClick={() => setMobileOpen(!mobileOpen)}>{mobileOpen ? <X /> : <Menu />}</button>
          {mobileOpen && <div className="rb-mobile-nav"><a href="#how-it-works" onClick={() => setMobileOpen(false)}>How it works</a><a href="#live-network" onClick={() => setMobileOpen(false)}>Live network</a><a href="#architecture" onClick={() => setMobileOpen(false)}>Architecture</a><a href="#evidence" onClick={() => setMobileOpen(false)}>Evidence</a><a href="#install" onClick={() => setMobileOpen(false)}>Install</a><a href="#hardware" onClick={() => setMobileOpen(false)}>Find your worker</a></div>}
        </div>
      </header>

      <section className="rb-hero" aria-labelledby="rb-hero-title">
        <img className="rb-hero-image" src={heroImage} alt="A mushroom colony connected by a mycelium network and warm nodes" fetchPriority="high" decoding="async" />
        <HeroNetwork />
        <NetworkThreads />
        <div className="rb-hero-inner rb-shell">
          <div className="rb-hero-copy">
            <p className="rb-status-pill"><i /><strong>EARLY NETWORK</strong><b />Physical multi-node validation underway</p>
            <p className="rb-kicker"><i /><span>Distributed AI · Rooted in nature</span></p>
            <h1 id="rb-hero-title" aria-label="Intelligence grows through the network.">Intelligence<br />grows through the <em>network.</em></h1>
            <p className="rb-lede">Mycellios connects capacity across different machines to run AI models that cannot fit on a single device.</p>
            <div className="rb-actions"><a className="rb-pill rb-pill-dark" href="/join">Join the network <ArrowRight /></a><a className="rb-text-link" href="#how-it-works">See how it works <ArrowDown /></a></div>
            <p className="rb-hero-footnote"><ShieldCheck /><span>No account or invitation required during public testing.</span></p>
          </div>
          <figure className="rb-active-card" aria-label="Conceptual network status">
            <span><i /> Concept sequence</span><div className="rb-active-card-content" key={currentHeroSignal.worker}><strong>{currentHeroSignal.worker}</strong><small>{currentHeroSignal.status}</small></div>
            <div className="rb-mini-route"><b /><b /><b /><b /></div>
          </figure>
          <p className="rb-hero-note">Every mushroom is capacity. The mycelium decides how to connect it.</p>
        </div>
      </section>

      <ExplainerSection />

      <section className="rb-manifesto rb-shell" aria-label="Manifesto">
        <p className="rb-manifesto-side rb-reveal">An organism,<br />not a data center.</p>
        <div className="rb-manifesto-main rb-reveal">
          <p className="rb-kicker"><i /><span>The capacity exists. It is fragmented.</span></p>
          <h2>The most powerful AI lives behind walls of silicon. <em>We are building another door.</em></h2>
          <p className="rb-manifesto-copy">Millions of PCs, workstations, and small servers spend much of the day underused. Alone, they are not enough. Coordinated, they can become a new class of infrastructure.</p>
          <figure className="rb-manifesto-visual">
            <img src={storyImage} alt="Fine mycelium threads connecting colonies beneath a forest floor" loading="lazy" decoding="async" />
            <figcaption>Independent organisms · one living network</figcaption>
          </figure>
          <a className="rb-text-link" href="#architecture">Discover GDLP/2 <ArrowUpRight /></a>
        </div>
      </section>

      <section className="rb-flow" id="how-it-works" ref={flowRef}>
        <div className="rb-flow-sticky rb-shell">
          <div className="rb-flow-heading"><p className="rb-kicker">01 — How it works</p><h2>One task.<br />Many organisms.</h2><p>Scroll to follow a request as it moves through the network.</p></div>
          <div className={`rb-flow-scene step-${flowStep}`}>
            <div className="rb-device"><span /><span /><span /><b>Your request</b></div>
            <div className="rb-sprout"><i /><b /></div>
            <svg className="rb-flow-roots" viewBox="0 0 720 260" aria-hidden="true"><path d="M74 120c75 0 91 44 154 44s77-81 144-81 104 104 176 104 73-63 126-63" /><path d="M226 164c-23 38-47 52-85 64m88-64c35 22 59 46 73 80m69-161c-26 46-23 88 8 126m-8-126c37 34 66 39 104 22m73 82c-2 31 16 48 55 60" /></svg>
            <div className="rb-worker-node node-a"><i /><span>worker_01</span></div><div className="rb-worker-node node-b"><i /><span>worker_02</span></div><div className="rb-worker-node node-c"><i /><span>worker_03</span></div>
            <div className="rb-result"><Check /><span>Answer</span></div>
            <div className="rb-flow-pulse" />
          </div>
          <div className="rb-flow-copy">
            <div className="rb-flow-copy-content" key={currentStep.id}><span>{currentStep.id} / 05</span><h3>{currentStep.title}</h3><p>{currentStep.copy}</p></div>
            <div className="rb-progress">{steps.map((step, index) => <button type="button" key={step.id} className={index <= flowStep ? "is-active" : ""} onClick={() => scrollToFlowStep(index)} aria-label={`Go to step ${step.id}: ${step.title}`} />)}</div>
          </div>
          <ol className="rb-flow-fallback">{steps.map((step) => <li key={step.id}><span>{step.id}</span><div><h3>{step.title}</h3><p>{step.copy}</p></div></li>)}</ol>
        </div>
      </section>

      <AccelerationSection />

      <section className="rb-live" id="live-network">
        <img src={liveImage} alt="Living mycelium carrying warm signals between small mushroom colonies" loading="lazy" decoding="async" />
        <div className="rb-live-shade" />
        <div className="rb-live-inner rb-shell">
          <div className="rb-live-head"><p className="rb-kicker rb-kicker-light"><i /> 03 — Live network</p><h2>Activity emerges.<br />The network responds.</h2><p className="rb-live-copy">Connections light up only when a task finds a useful route.</p><p className="rb-live-disclaimer">Narrative visualization · not production telemetry</p></div>
          <div className="rb-live-map"><NetworkThreads compact /><span className="rb-map-label label-eu">Madrid · new worker</span><span className="rb-map-label label-us">Virginia · stage complete</span><span className="rb-map-label label-sa">São Paulo · route available</span></div>
          <aside className="rb-live-feed"><div className="rb-feed-head"><span>Concept sequence</span><i>sample</i></div><ul><li><b /><span>New worker connected<small>Madrid · RTX 3060</small></span><time>now</time></li><li><b /><span>Model stage executed<small>Route 08 · northern cell</small></span><time>4 s</time></li><li><b /><span>Response gathered<small>Stream returned to origin</small></span><time>11 s</time></li></ul></aside>
          <div className="rb-live-metric" aria-label="Rotating conceptual summary"><div className="rb-metric-value" key={metric} aria-live="polite" aria-atomic="true"><span>{currentMetric.value}</span><p>{currentMetric.label}</p></div><button type="button" onClick={() => setPaused(!paused)} aria-label={paused ? "Resume summary" : "Pause summary"}>{paused ? <Play /> : <Pause />}</button></div>
        </div>
      </section>

      <ArchitectureSection />

      <section className="rb-workers rb-shell" id="workers">
        <div className="rb-section-title"><p className="rb-kicker">05 — Workers</p><h2>One species for<br />every kind of capacity.</h2><p>Mushrooms are the network's living language: they signal power, grouping, and availability.</p></div>
        <div className="rb-worker-tabs" role="tablist" aria-label="Worker types">{workers.map((worker, index) => <button key={worker.species} id={`worker-tab-${index}`} type="button" role="tab" aria-selected={activeWorker === index} aria-controls="worker-panel" className={activeWorker === index ? "is-active" : ""} onClick={() => setActiveWorker(index)} onFocus={() => setActiveWorker(index)} onMouseEnter={() => setActiveWorker(index)}><span>{String(index + 1).padStart(2, "0")}</span>{worker.species}</button>)}</div>
        <div className="rb-worker-stage" id="worker-panel" role="tabpanel" aria-labelledby={`worker-tab-${activeWorker}`}>
          <div key={`mushroom-${currentWorker.species}`} className={`rb-mushroom-family ${currentWorker.group}`} aria-hidden="true"><div className="cap cap-one" /><div className="stem stem-one" /><div className="cap cap-two" /><div className="stem stem-two" /><div className="cap cap-three" /><div className="stem stem-three" /><div className="family-roots" /></div>
          <article key={currentWorker.species}><p>{currentWorker.species}</p><h3>{currentWorker.name}</h3><dl><div><dt><Cpu /> Hardware</dt><dd>{currentWorker.hardware}</dd></div><div><dt><Sparkles /> Role</dt><dd>{currentWorker.model}</dd></div><div><dt><Network /> Load</dt><dd>{currentWorker.load}</dd></div></dl><small>Illustrative representation. The actual role depends on the model, topology, and availability.</small></article>
        </div>
      </section>

      <section className="rb-benefits" id="principles">
        <div className="rb-benefits-inner rb-shell">
          <div className="rb-section-title rb-section-title-wide"><p className="rb-kicker">06 — Principles</p><h2>The network behaves<br />like something alive.</h2></div>
          <article className="rb-benefit rb-benefit-center"><span>01</span><div><p>Decentralized</p><h3>When one node goes dark,<br />the route grows again.</h3></div><NetworkThreads compact /></article>
          <div className="rb-benefit-pair"><article className="rb-benefit rb-benefit-private"><ShieldCheck /><span>02</span><p>Private</p><h3>Your capacity is shared.<br />Your rules remain.</h3><div className="rb-privacy-flow"><b>device</b><i /><b>private network</b><i className="blocked" /><b>outside</b></div></article><article className="rb-benefit rb-benefit-scale"><span>03</span><p>Scalable</p><h3>Every new worker expands<br />the possible routes.</h3><div className="rb-spore-field">{Array.from({ length: 18 }, (_, i) => <i key={i} style={{ "--i": i } as CSSProperties} />)}</div></article></div>
          <div className="rb-benefit-pair rb-benefit-modes">
            <article className="rb-benefit rb-benefit-exact">
              <span>04</span>
              <p>Exact when it matters</p>
              <h3>Distribution changes where the model is computed, <em>not the result it must produce.</em></h3>
              <div className="rb-mode-foot"><ShieldCheck /><span>Verification and rollback</span></div>
            </article>
            <article className="rb-benefit rb-benefit-optin">
              <span>05</span>
              <p>Voluntary approximation</p>
              <h3>Compression and alternative codecs are used only when chosen by the user, <em>and the degradation is measured.</em></h3>
              <div className="rb-mode-foot"><Check /><span>Always opt-in</span></div>
            </article>
          </div>
          <p className="rb-modes-note">The architecture explicitly separates reference-equivalent inference from every approximate optimization. They are never mixed silently.</p>
          <article className="rb-benefit rb-benefit-community"><span>06 — Community-built</span><h3>The answer does not belong to one machine.<br /><em>Many machines build it.</em></h3><div className="rb-contributors"><b>laptop</b><i /><b>workstation</b><i /><b>server</b><i /><strong>one answer</strong></div></article>
        </div>
      </section>

      <EvidenceSection />

      <section className="rb-hardware" id="hardware">
        <div className="rb-hardware-inner rb-shell">
          <div className="rb-hardware-copy"><p className="rb-kicker rb-kicker-light">08 — Your place in the network</p><h2>What kind of<br />worker would you be?</h2><p>Select an illustrative setup and discover the role it could play within the mycelium.</p><label htmlFor="hardware-select">Your hardware</label><div className="rb-select-wrap"><select id="hardware-select" value={hardware} onChange={(event) => setHardware(event.target.value as HardwareKey)}><option value="integrated">CPU / integrated graphics</option><option value="rtx-3060">NVIDIA RTX 3060 · 12 GB</option><option value="rtx-4090">NVIDIA RTX 4090 · 24 GB</option><option value="multi">Multiple machines</option></select><ChevronDown /></div><a className="rb-pill rb-pill-light" href="/downloads">Turn my machine into a worker <ArrowRight /></a></div>
          <div className="rb-profile" key={hardware}><div className="rb-profile-top"><img src={brandLogo} alt="" /><span>Illustrative profile</span><i>available</i></div><div className="rb-profile-mushroom"><div className="rb-profile-cap" /><div className="rb-profile-stem" /><div className="rb-profile-roots" /></div><h3>{profile.label}</h3><p>{profile.type}</p><dl><div><dt><MemoryStick /> Capacity</dt><dd>{profile.capacity}</dd></div><div><dt><Sparkles /> Could run</dt><dd>{profile.models}</dd></div><div><dt><Cpu /> Compute</dt><dd>{profile.compute}</dd></div></dl><small>Reward estimates are not available yet.</small></div>
        </div>
      </section>

      <InstallSection />

      <RoadmapSection />

      <section className="rb-participate rb-shell" aria-labelledby="rb-participate-title">
        <div className="rb-section-title rb-reveal">
          <p className="rb-kicker">11 — Build the network</p>
          <h2 id="rb-participate-title">There is more than one way<br /><em>to take part.</em></h2>
          <p>We are looking for the first machines, organizations, and people ready to turn a difficult idea into real infrastructure.</p>
        </div>
        <div className="rb-participate-grid">
          {participateCards.map((audience, index) => (
            <article className={`rb-participate-card rb-reveal ${index === 1 ? "featured" : ""}`} key={audience.label}>
              <div className="rb-participate-icon">{index === 0 ? <Cpu /> : <Users />}</div>
              <span>{audience.label}</span>
              <h3>{audience.title}</h3>
              <p>{audience.copy}</p>
              <ul>{audience.bullets.map((bullet) => <li key={bullet}><Check />{bullet}</li>)}</ul>
            </article>
          ))}
        </div>
      </section>

      <section className="rb-closing" aria-labelledby="rb-closing-title">
        <div className="rb-closing-inner rb-shell rb-reveal">
          <Brand />
          <p className="rb-kicker rb-kicker-light"><i /><span>Founding network</span></p>
          <h2 id="rb-closing-title">Many machines.<br /><em>One model.</em></h2>
          <p>Join the first mycellios test network and help us prove that the next great machine can be a community.</p>
          <a className="rb-pill rb-pill-light" href="/join">Connect this device <Zap /></a>
          <small>Public test network · no login required</small>
        </div>
      </section>

      <footer className="rb-footer">
        <div className="rb-footer-inner rb-shell">
          <div><Brand /><p>Distributed intelligence,<br />rooted in nature.</p></div>
          <nav aria-label="Explore"><span>Explore</span><a href="#explainer">Start here</a><a href="#how-it-works">How it works</a><a href="#live-network">Live network</a><a href="#roadmap">Roadmap</a></nav>
          <nav aria-label="Project"><span>Project</span><a href="/docs">Documentation</a><a href="/network">Network status</a><a href="/blog">Blog</a><a href="/mobile/">Mobile worker</a><a href="https://github.com/tych0s/mycellios"><GitBranch /> GitHub</a></nav>
          <div className="rb-footer-note">
            <div className="rb-footer-contact"><a href="/downloads">Downloads</a><a href={`mailto:${EMAIL}`}>{EMAIL}</a><span>© 2026 mycellios</span></div>
            <a href="#rb-top">Back to top <ArrowDown /></a>
          </div>
        </div>
      </footer>
      <SupportAssistant surface="landing" />
    </main>
  );
}
