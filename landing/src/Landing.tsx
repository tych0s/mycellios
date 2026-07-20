import {
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  Boxes,
  Check,
  ChevronRight,
  CircleDot,
  Cpu,
  Download,
  Gauge,
  HardDriveDownload,
  Layers3,
  MonitorDown,
  Network,
  Radio,
  ShieldCheck,
  Sparkles,
  Users,
  Workflow,
  Zap,
} from "lucide-react";
import { useEffect, useRef, useState, type CSSProperties, type PointerEvent } from "react";
import brandIcon from "../../src/renderer/assets/mycellios-icon.png";
import efficiencyCurve from "./assets/efficiency-curve-ai.webp";
import myceliumHero from "./assets/mycelium-hero.jpg";
import myceliumNetwork from "./assets/mycelium-network.jpg";

const EMAIL = "hello@mycellios.com";
const RELEASE_BASE = "https://github.com/tych0s/mycellios/download";
const RELEASES_URL = "https://github.com/tych0s/mycellios";

const downloads = {
  windows: { label: "Windows", detail: "Windows 10/11 · x64", filename: "mycellios-windows-x64.exe" },
  "mac-arm64": { label: "macOS", detail: "Apple Silicon", filename: "mycellios-macos-arm64.dmg" },
  "mac-x64": { label: "macOS", detail: "Intel", filename: "mycellios-macos-x64.dmg" },
  "linux-deb": { label: "Linux", detail: "Ubuntu / Debian · x64", filename: "mycellios-linux-x64.deb" },
  "linux-rpm": { label: "Linux", detail: "Fedora / RHEL · x64", filename: "mycellios-linux-x64.rpm" },
} as const;

type DownloadKey = keyof typeof downloads;

function downloadUrl(key: DownloadKey): string {
  return `${RELEASE_BASE}/${downloads[key].filename}`;
}

const copy = {
    meta: { title: "mycellios — One model. Many machines.", description: "mycellios coordinates heterogeneous machines to run AI models that do not fit on any single computer." },
    home: "mycellios, home",
    languageLabel: "Change language",
    nav: { aria: "Main navigation", vision: "Vision", architecture: "Architecture", evidence: "Real progress", join: "Download" },
    hero: {
      status: "Physical multi-node validation underway",
      line1: "Your next", line2: "supercomputer", line3: "is already on.",
      copy: "mycellios coordinates different computers to run AI models that do not fit on any single machine.",
      primary: "Download mycellios", secondary: "See how it works",
      footnote: "Contribution is voluntary, measurable, and can be paused at any time.",
    },
    visual: { aria: "Visualization of a network of machines forming a single model", model: "MODEL", route: "active route", nodes: "9 nodes", heterogeneous: "heterogeneous", capacity: "Capacity", coordinated: "coordinated", caption: "Concept visualization · network forming" },
    signals: ["LARGE MODELS", "HETEROGENEOUS HARDWARE", "DISTRIBUTED MEMORY", "ADAPTIVE ROUTES", "EXACT MODE", "MoE MODELS"],
    manifesto: { index: "01 / THE IDEA", kicker: "The capacity exists. It is fragmented.", title: "The most powerful AI lives behind walls of silicon. We are building another door.", copy: "Millions of PCs, workstations, and small servers spend much of the day underused. Alone, they are not enough. Coordinated, they can become a new class of infrastructure.", link: "Discover GDLP/2" },
    mycelium: {
      eyebrow: "The inspiration", title: "The network learned how to live before we did.",
      copy: "Mycelium has no center controlling everything. It senses locally, transmits signals, redistributes resources, and finds new paths as its environment changes. mycellios turns that living logic into computing infrastructure.",
      imageAlt: "Bioluminescent mycelium network transmitting signals through its branches",
      live: "LIVE SIGNAL", packet: "packet 0x7F", metrics: ["No single center", "Local signals", "Adaptive network"],
    },
    acceleration: {
      eyebrow: "The compounding curve",
      title: "Intelligence gets cheaper. Distribution makes it go further.",
      copy: "Model efficiency is already moving at extraordinary speed. mycellios is being built to compound that progress: pooling memory first, shortening routes next, and increasing effective tokens per second as the network matures.",
      observed: "Observed benchmark shift",
      from: { date: "DEC 2024", model: "o3-preview · high compute", cost: ">$3,000 / task", score: "87.5% ARC-AGI-1" },
      to: { date: "DEC 2025", model: "Gemini 3 Flash · high", cost: "$0.231 / task", score: "84.7% ARC-AGI-1" },
      multiplier: "≈13,000×",
      multiplierLabel: "lower cost per task",
      axis: "CAPABILITY PER DOLLAR",
      note: "The same benchmark family, but different systems and evaluation configurations. This is a directional efficiency signal—not an apples-to-apples price comparison.",
      source: "Inspect the ARC Prize data",
      roadmapEyebrow: "The mycellios development curve",
      roadmapTitle: "Fit. Route. Multiply.",
      roadmapCopy: "Every layer removes a different bottleneck. The target is a compounding throughput curve, not a single benchmark trick.",
      direction: "DIRECTION · NOT A PERFORMANCE FORECAST",
      equation: "MODEL EFFICIENCY × NETWORK EFFICIENCY = COMPOUND SYSTEM SPEED",
      phases: [
        { step: "NOW · 01", title: "Fit the model", copy: "Pool heterogeneous memory so models can run beyond the limit of any single machine.", signal: "MEMORY" },
        { step: "NEXT · 02", title: "Shorten the route", copy: "Keep experts resident and move each request through fewer, faster network boundaries.", signal: "LOCALITY" },
        { step: "SCALE · 03", title: "Multiply the paths", copy: "Coordinate parallel cells so added capacity can become useful throughput instead of added latency.", signal: "TOKENS / S" },
      ],
    },
    architecture: { eyebrow: "Adaptive architecture", title: "We do not connect more machines. We build the right route.", copy: "Adding a slow node can make the entire system worse. GDLP/2 analyzes each model, groups nearby resources, and selects only the machines that add value to that run." },
    pipeline: {
      target: "TARGET MODEL", tooLarge1: "Too large", tooLarge2: "for a single node", reserve: "reserve",
      stages: [
        { number: "01", name: "Analyze", detail: "memory · topology" },
        { number: "02", name: "Partition", detail: "layers · experts" },
        { number: "03", name: "Assign", detail: "the best machine" },
        { number: "04", name: "Coordinate", detail: "one route" },
      ],
    },
    principles: [
      { title1: "One model,", title2: "many memories", copy: "Each node stores and processes only one part. The complete model emerges from collaboration.", label: "PARTITIONING" },
      { title1: "Cells first,", title2: "then the network", copy: "Nearby machines form fast cells. The WAN connects a few virtual stages, not hundreds of hops.", label: "HIERARCHICAL MESH" },
      { title1: "The route changes.", title2: "The work continues.", copy: "The scheduler measures capacity, connectivity, and availability to prepare alternatives as the network changes.", label: "ADAPTIVE ROUTING" },
    ],
    modes: { eyebrow: "One network, two guarantees", title1: "Fast when it helps.", title2: "Exact when it matters.", copy: "The architecture explicitly separates reference-equivalent inference from every approximate optimization. They are never mixed silently.", exactTitle: "Exact mode", exactCopy: "Distribution changes where the model is computed, not the result it must produce.", exactFooter: "Verification and rollback", optTitle: "Voluntary approximation", optCopy: "Compression and alternative codecs are used only when chosen by the user and the degradation is measured.", optFooter: "Always opt-in" },
    evidence: { eyebrow: "Evidence before promises", title: "We are building in public.", copy: "We do not call a simulation a global network. Every important claim passes a reproducible test before it becomes a product promise.", done: "Implemented", next: "Next physical milestone", gate: "NEXT GATE", rows: [["Partitioned-model runtime", "Real pipeline and compatible API", "READY"], ["Heterogeneous scheduler", "Memory, topology, and availability", "READY"], ["Stage recovery", "Alternate route and greedy continuation", "LAB"], ["Model larger than every node", "Test across 2–4 physical machines", "TESTING"]] },
    install: {
      eyebrow: "From download to network",
      title1: "One click.",
      title2: "Your machine joins the organism.",
      copy: "Install the desktop app, choose how you want to participate, and mycellios detects your hardware automatically. No terminal, Docker, or manual configuration required.",
      detected: "Recommended for this device",
      download: "Download for",
      early: "Early access · automatic hardware detection",
      alternatives: "Other downloads",
      allReleases: "Release notes and checksums",
      safety: "Every installer is built and package-verified in GitHub Actions. Code signing is being rolled out; early builds may still trigger an operating-system publisher warning.",
      consoleLabel: "MYCELLIOS · FIRST RUN",
      ready: "NETWORK READY",
      node: "this machine",
      steps: [
        { label: "Install", detail: "Desktop agent added", status: "DONE" },
        { label: "Detect", detail: "CPU · RAM · GPU mapped", status: "DONE" },
        { label: "Choose", detail: "Join or create a network", status: "READY" },
      ],
      promise: "YOU DECIDE WHAT TO SHARE · PAUSE AT ANY TIME",
    },
    participate: { eyebrow: "Build the network", title: "There is more than one way to take part.", copy: "We are looking for the first machines, organizations, and people ready to turn a difficult idea into real infrastructure.", contributor: { label: "FOR CONTRIBUTORS", title: "Turn idle capacity into useful capacity.", copy: "Connect a PC, workstation, or server. Decide how much you contribute and pause whenever you want.", bullets: ["Different hardware, one network", "Voluntary contribution", "Verifiable work"] }, organization: { label: "FOR TEAMS AND ORGANIZATIONS", title: "Run open models on infrastructure you control.", copy: "Create private networks for labs, companies, and communities with distributed hardware.", bullets: ["Larger models through pooled memory", "A familiar API for your applications", "Topology adapted to your network"] } },
    closing: { eyebrow: "Founding network", title1: "Many machines.", title2: "One model.", copy: "Join the first mycellios test network and help us prove that the next great machine can be a community.", button: "Request early access", email: "Write to us at", subject: "I want to join mycellios" },
  footer: { tagline: "Distributed intelligence, built together.", architecture: "Architecture", status: "Status", contact: "Contact" },
} as const;

const nodes = [
  { x: 90, y: 130, label: "RTX 3060", tone: "cyan" }, { x: 185, y: 66, label: "CPU · 64 GB", tone: "blue" },
  { x: 300, y: 116, label: "4 GB VRAM", tone: "violet" }, { x: 535, y: 76, label: "RX 6800", tone: "cyan" },
  { x: 608, y: 188, label: "Mini PC", tone: "violet" }, { x: 564, y: 355, label: "Apple Silicon", tone: "blue" },
  { x: 432, y: 474, label: "CPU · 32 GB", tone: "cyan" }, { x: 210, y: 492, label: "GTX 1650", tone: "violet" },
  { x: 72, y: 392, label: "Workstation", tone: "blue" },
] as const;

const links = [[0, 1], [0, 2], [0, 8], [1, 2], [1, 3], [2, 4], [2, 6], [3, 4], [3, 5], [4, 5], [5, 6], [6, 7], [6, 8], [7, 8], [2, 7]] as const;

function Brand({ compact = false, homeLabel }: { compact?: boolean; homeLabel: string }) {
  return <a className={`brand ${compact ? "brand-compact" : ""}`} href="#top" aria-label={homeLabel}><img src={brandIcon} alt="" /><span>mycellios</span></a>;
}

function LivingBackdrop() {
  const paths = ["M-60 190 C180 40 280 330 520 190 S850 55 1080 220 S1350 340 1540 110", "M-80 480 C190 600 280 310 520 450 S850 650 1090 430 S1370 280 1540 520", "M120 -50 C230 170 440 100 530 320 S690 650 900 480 S1160 180 1490 290", "M20 740 C270 610 430 760 620 560 S880 260 1120 470 S1330 710 1520 590"];
  return (
    <div className="living-background" aria-hidden="true">
      <img src={myceliumHero} alt="" />
      <svg viewBox="0 0 1440 800" preserveAspectRatio="xMidYMid slice">
        {paths.map((path, index) => <g key={path}><path d={path} className="hypha-base" /><path d={path} className="hypha-pulse" style={{ "--path-delay": `${index * -1.7}s` } as CSSProperties} /></g>)}
      </svg>
      <div className="spore-field" />
    </div>
  );
}

function NetworkHero({ text }: { text: typeof copy.visual }) {
  return (
    <div className="network-stage" aria-label={text.aria}>
      <div className="stage-grid" /><div className="stage-glow" />
      <svg viewBox="0 0 680 560" role="img" aria-hidden="true">
        <defs>
          <linearGradient id="linkGradient" x1="0" x2="1"><stop offset="0" stopColor="#66f6d1" /><stop offset=".52" stopColor="#768cff" /><stop offset="1" stopColor="#cf6dff" /></linearGradient>
          <radialGradient id="coreGradient"><stop offset="0" stopColor="#e8fffb" /><stop offset=".28" stopColor="#65f5d0" /><stop offset=".68" stopColor="#6576ef" /><stop offset="1" stopColor="#9d52d5" /></radialGradient>
          <filter id="softGlow" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="8" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>
        <g className="mesh-lines">
          {links.map(([from, to], index) => { const start = nodes[from]; const end = nodes[to]; return <line key={`${from}-${to}`} x1={start.x} y1={start.y} x2={end.x} y2={end.y} className={index % 3 === 0 ? "signal-line" : ""} />; })}
          {nodes.map((node) => <line key={`core-${node.label}`} x1={node.x} y1={node.y} x2="340" y2="286" className="core-link" />)}
        </g>
        <g className="core" filter="url(#softGlow)"><circle cx="340" cy="286" r="93" className="core-orbit orbit-a" /><circle cx="340" cy="286" r="68" className="core-orbit orbit-b" /><circle cx="340" cy="286" r="49" fill="url(#coreGradient)" /><circle cx="340" cy="286" r="26" className="core-center" /></g>
        <g className="node-points">{nodes.map((node, index) => <g key={node.label} className={`mesh-node node-${node.tone}`} style={{ "--delay": `${index * -0.31}s` } as CSSProperties}><circle cx={node.x} cy={node.y} r="18" className="node-halo" /><circle cx={node.x} cy={node.y} r="7" className="node-dot" /></g>)}</g>
      </svg>
      <div className="core-label"><span>01</span><strong>{text.model}</strong><small>{text.route}</small></div>
      <div className="floating-stat stat-one"><Cpu size={14} /><span>{text.nodes}</span><strong>{text.heterogeneous}</strong></div>
      <div className="floating-stat stat-two"><Zap size={14} /><span>{text.capacity}</span><strong>{text.coordinated}</strong></div>
      <div className="stage-caption"><CircleDot size={13} /> {text.caption}</div>
    </div>
  );
}

function SectionHeading({ eyebrow, title, copy }: { eyebrow: string; title: string; copy: string }) {
  return <div className="section-heading reveal"><span className="eyebrow"><i />{eyebrow}</span><h2>{title}</h2><p>{copy}</p></div>;
}

function MyceliumStory({ text }: { text: typeof copy.mycelium }) {
  return (
    <section className="mycelium-story" id="mycelium">
      <div className="mycelium-visual reveal">
        <img src={myceliumNetwork} alt={text.imageAlt} loading="lazy" />
        <div className="mycelium-scan" />
        <svg viewBox="0 0 1000 620" preserveAspectRatio="none" aria-hidden="true"><path d="M50 390 C210 180 330 510 480 290 S730 80 950 240" /><path d="M80 470 C260 540 340 230 540 390 S790 520 940 330" /></svg>
        <div className="signal-chip"><i /><span>{text.live}</span><strong>{text.packet}</strong></div>
      </div>
      <div className="container mycelium-copy">
        <div className="mycelium-number reveal">02 / MYCELIUM</div>
        <div className="mycelium-message reveal"><span className="eyebrow"><i />{text.eyebrow}</span><h2>{text.title}</h2><p>{text.copy}</p></div>
        <div className="mycelium-metrics reveal">{text.metrics.map((metric, index) => <div key={metric}><span>0{index + 1}</span><strong>{metric}</strong></div>)}</div>
      </div>
    </section>
  );
}

function EfficiencyCurve({ text }: { text: typeof copy.acceleration }) {
  return (
    <section className="acceleration-section" id="acceleration">
      <div className="container">
        <div className="acceleration-heading reveal">
          <span className="eyebrow"><i />{text.eyebrow}</span>
          <h2>{text.title}</h2>
          <p>{text.copy}</p>
        </div>

        <article className="efficiency-curve reveal">
          <img src={efficiencyCurve} alt="An abstract luminous mycelium network accelerating into a dense flow of information" loading="lazy" />
          <div className="efficiency-veil" />
          <div className="curve-topline"><span>{text.observed}</span><span>ARC-AGI-1 · 12 MONTHS</span></div>
          <span className="curve-axis">{text.axis}</span>
          <svg className="curve-plot" viewBox="0 0 1000 470" preserveAspectRatio="none" aria-hidden="true">
            <defs>
              <linearGradient id="efficiencyLine" x1="0" x2="1"><stop offset="0" stopColor="#7085ff" /><stop offset=".55" stopColor="#65f6d1" /><stop offset="1" stopColor="#e8fffb" /></linearGradient>
              <filter id="curveGlow" x="-30%" y="-50%" width="160%" height="200%"><feGaussianBlur stdDeviation="7" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
            </defs>
            <g className="curve-grid">
              {[90, 180, 270, 360].map((y) => <line key={`h-${y}`} x1="80" y1={y} x2="930" y2={y} />)}
              {[160, 350, 540, 730, 920].map((x) => <line key={`v-${x}`} x1={x} y1="55" x2={x} y2="405" />)}
            </g>
            <path className="curve-shadow" d="M150 365 C330 360 475 315 590 240 S750 115 845 82" />
            <path className="curve-line" d="M150 365 C330 360 475 315 590 240 S750 115 845 82" />
            <path className="curve-signal" d="M150 365 C330 360 475 315 590 240 S750 115 845 82" />
            <g className="curve-point point-start"><circle cx="150" cy="365" r="20" /><circle cx="150" cy="365" r="5" /></g>
            <g className="curve-point point-end" filter="url(#curveGlow)"><circle cx="845" cy="82" r="24" /><circle cx="845" cy="82" r="6" /></g>
          </svg>
          <div className="benchmark-point benchmark-from"><span>{text.from.date}</span><strong>{text.from.model}</strong><b>{text.from.cost}</b><small>{text.from.score}</small></div>
          <div className="benchmark-point benchmark-to"><span>{text.to.date}</span><strong>{text.to.model}</strong><b>{text.to.cost}</b><small>{text.to.score}</small></div>
          <div className="curve-multiplier"><strong>{text.multiplier}</strong><span>{text.multiplierLabel}</span><ArrowUpRight size={17} /></div>
        </article>

        <div className="curve-evidence reveal">
          <p>{text.note}</p>
          <div>
            <a href="https://arcprize.org/blog/oai-o3-pub-breakthrough" target="_blank" rel="noreferrer">o3 analysis <ArrowUpRight size={13} /></a>
            <a href="https://arcprize.org/leaderboard?hl=en-US" target="_blank" rel="noreferrer">{text.source} <ArrowUpRight size={13} /></a>
          </div>
        </div>

        <div className="roadmap-heading reveal">
          <div><span>{text.roadmapEyebrow}</span><h3>{text.roadmapTitle}</h3></div>
          <p>{text.roadmapCopy}</p>
          <small>{text.direction}</small>
        </div>
        <div className="acceleration-roadmap">
          {text.phases.map((phase, index) => (
            <article className="roadmap-phase reveal" key={phase.step}>
              <div className="phase-line"><i /><span>{phase.step}</span><b>0{index + 1}</b></div>
              <h4>{phase.title}</h4><p>{phase.copy}</p><strong>{phase.signal}</strong>
              {index < text.phases.length - 1 && <ArrowRight className="phase-arrow" size={18} />}
            </article>
          ))}
        </div>
        <div className="curve-equation reveal"><i /><span>{text.equation}</span><i /></div>
      </div>
    </section>
  );
}

function PipelineVisual({ text }: { text: typeof copy.pipeline }) {
  return (
    <div className="pipeline-card reveal">
      <div className="pipeline-topline"><span>GDLP / 2</span><span>ROUTE COMPILER</span><span className="live-dot">LIVE</span></div>
      <div className="pipeline-model"><div className="model-cube"><Boxes size={29} /><span>70B</span></div><div><small>{text.target}</small><strong>{text.tooLarge1}<br />{text.tooLarge2}</strong></div><ArrowDownRight size={24} /></div>
      <div className="pipeline-flow">{text.stages.map((stage, index) => <div className="pipeline-stage" key={stage.number}><span>{stage.number}</span><div><strong>{stage.name}</strong><small>{stage.detail}</small></div>{index < text.stages.length - 1 && <ChevronRight className="pipeline-arrow" size={17} />}</div>)}</div>
      <div className="pipeline-nodes">{["GPU · 12 GB", "GPU · 4 GB", "CPU · 64 GB", "GPU · 8 GB", "STANDBY"].map((node, index) => <div key={node} className={index === 4 ? "standby" : ""}><i /><span>{node}</span><small>{index === 4 ? text.reserve : `shard ${String.fromCharCode(65 + index)}`}</small></div>)}</div>
    </div>
  );
}

function InstallSection({ text }: { text: typeof copy.install }) {
  const [recommended, setRecommended] = useState<DownloadKey>("windows");

  useEffect(() => {
    const agent = navigator.userAgent.toLowerCase();
    if (agent.includes("mac")) setRecommended("mac-arm64");
    else if (agent.includes("linux")) setRecommended("linux-deb");
    else setRecommended("windows");
  }, []);

  const selected = downloads[recommended];

  return (
    <section className="install-section" id="install">
      <div className="install-field" aria-hidden="true"><i /><i /><i /><i /><i /></div>
      <div className="container install-layout">
        <div className="install-copy reveal">
          <span className="eyebrow"><i />{text.eyebrow}</span>
          <h2>{text.title1}<br /><em>{text.title2}</em></h2>
          <p>{text.copy}</p>
          <div className="recommended-platform"><span><MonitorDown size={14} />{text.detected}</span><strong>{selected.label} · {selected.detail}</strong></div>
          <a className="install-primary" href={downloadUrl(recommended)}>
            <span className="install-download-icon"><Download size={22} /></span>
            <span><small>{text.download}</small><strong>{selected.label}</strong></span>
            <ArrowDownRight size={20} />
          </a>
          <div className="install-meta"><ShieldCheck size={14} /><span>{text.early}</span></div>
        </div>

        <div className="install-console reveal">
          <div className="install-console-top"><div><i /><i /><i /></div><span>{text.consoleLabel}</span><b>01:18</b></div>
          <div className="install-console-body">
            <div className="install-node-orb"><span /><Radio size={27} /><i /></div>
            <div className="install-ready"><span><i />{text.ready}</span><strong>{text.node}</strong><small>NODE · 7FA3_C2E1</small></div>
            <div className="install-steps">
              {text.steps.map((step, index) => <div key={step.label} style={{ "--install-delay": `${index * .45}s` } as CSSProperties}><span>0{index + 1}</span><i><Check size={11} /></i><p><strong>{step.label}</strong><small>{step.detail}</small></p><b>{step.status}</b></div>)}
            </div>
            <div className="install-promise"><ShieldCheck size={13} />{text.promise}</div>
          </div>
        </div>
      </div>

      <div className="container download-shelf reveal">
        <div className="download-shelf-head"><span>{text.alternatives}</span><a href={RELEASES_URL} target="_blank" rel="noreferrer">{text.allReleases} <ArrowUpRight size={13} /></a></div>
        <div className="download-options">
          {(Object.keys(downloads) as DownloadKey[]).map((key) => {
            const option = downloads[key];
            return <a className={key === recommended ? "recommended" : ""} href={downloadUrl(key)} key={key}><HardDriveDownload size={18} /><span><strong>{option.label}</strong><small>{option.detail}</small></span>{key === recommended ? <b>RECOMMENDED</b> : <ArrowDownRight size={14} />}</a>;
          })}
        </div>
        <p className="install-safety"><ShieldCheck size={14} />{text.safety}</p>
      </div>
    </section>
  );
}

function Landing() {
  const page = useRef<HTMLDivElement>(null);
  const t = copy;

  useEffect(() => {
    document.documentElement.lang = "en";
    document.title = t.meta.title;
    document.querySelector('meta[name="description"]')?.setAttribute("content", t.meta.description);
    document.querySelector('meta[property="og:title"]')?.setAttribute("content", t.meta.title);
    document.querySelector('meta[property="og:description"]')?.setAttribute("content", t.meta.description);
  }, [t.meta.description, t.meta.title]);

  useEffect(() => {
    const observer = new IntersectionObserver((entries) => entries.forEach((entry) => entry.isIntersecting && entry.target.classList.add("is-visible")), { threshold: 0.14 });
    page.current?.querySelectorAll(".reveal").forEach((element) => observer.observe(element));
    return () => observer.disconnect();
  }, []);

  function trackPointer(event: PointerEvent<HTMLDivElement>) {
    const bounds = event.currentTarget.getBoundingClientRect();
    event.currentTarget.style.setProperty("--pointer-x", `${event.clientX - bounds.left}px`);
    event.currentTarget.style.setProperty("--pointer-y", `${event.clientY - bounds.top}px`);
  }

  return (
    <div className="site" id="top" ref={page} onPointerMove={trackPointer}>
      <LivingBackdrop /><div className="ambient-pointer" />
      <header className="nav-shell"><nav className="nav container" aria-label={t.nav.aria}><Brand homeLabel={t.home} /><div className="nav-links"><a href="#vision">{t.nav.vision}</a><a href="#architecture">{t.nav.architecture}</a><a href="#evidence">{t.nav.evidence}</a></div><div className="nav-actions"><a className="nav-cta" href="#install">{t.nav.join} <Download size={15} /></a></div></nav></header>

      <main>
        <section className="hero container">
          <div className="hero-copy"><div className="availability"><span /><strong>EARLY NETWORK</strong><i /> {t.hero.status}</div><h1>{t.hero.line1}<br />{t.hero.line2}<br /><em>{t.hero.line3}</em></h1><p>{t.hero.copy}</p><div className="hero-actions"><a className="button button-primary" href="#install">{t.hero.primary} <Download size={17} /></a><a className="button button-secondary" href="#architecture">{t.hero.secondary} <ArrowDownRight size={17} /></a></div><div className="hero-footnote"><ShieldCheck size={15} /><span>{t.hero.footnote}</span></div></div>
          <NetworkHero text={t.visual} />
        </section>

        <section className="signal-strip" aria-label={t.nav.vision}><div className="signal-track">{t.signals.map((item) => <span key={item}>{item}<i /></span>)}</div></section>

        <section className="manifesto container" id="vision"><div className="manifesto-index reveal">{t.manifesto.index}</div><div className="manifesto-copy reveal"><span>{t.manifesto.kicker}</span><h2>{t.manifesto.title}</h2></div><div className="manifesto-side reveal"><p>{t.manifesto.copy}</p><a href="#architecture">{t.manifesto.link} <ArrowUpRight size={15} /></a></div></section>

        <MyceliumStory text={t.mycelium} />

        <EfficiencyCurve text={t.acceleration} />

        <section className="architecture-section" id="architecture"><div className="container"><SectionHeading eyebrow={t.architecture.eyebrow} title={t.architecture.title} copy={t.architecture.copy} /><PipelineVisual text={t.pipeline} /><div className="principles-grid">{t.principles.map((principle, index) => { const Icon = [Layers3, Network, Workflow][index] ?? Network; return <article className="principle-card reveal" key={principle.label}><div className="card-number">0{index + 1}</div><Icon size={25} /><h3>{principle.title1}<br />{principle.title2}</h3><p>{principle.copy}</p><span>{principle.label} <ArrowUpRight size={13} /></span></article>; })}</div></div></section>

        <section className="difference container"><div className="difference-copy reveal"><span className="eyebrow"><i />{t.modes.eyebrow}</span><h2>{t.modes.title1}<br /><em>{t.modes.title2}</em></h2><p>{t.modes.copy}</p></div><div className="mode-stack"><article className="mode-card exact reveal"><div><ShieldCheck size={23} /><span>01</span></div><h3>{t.modes.exactTitle}</h3><p>{t.modes.exactCopy}</p><footer><Check size={14} /> {t.modes.exactFooter}</footer></article><article className="mode-card optin reveal"><div><Sparkles size={23} /><span>02</span></div><h3>{t.modes.optTitle}</h3><p>{t.modes.optCopy}</p><footer><Check size={14} /> {t.modes.optFooter}</footer></article></div></section>

        <section className="evidence-section" id="evidence"><div className="container evidence-grid"><div><SectionHeading eyebrow={t.evidence.eyebrow} title={t.evidence.title} copy={t.evidence.copy} /><div className="status-legend reveal"><span><i className="done" />{t.evidence.done}</span><span><i className="next" />{t.evidence.next}</span></div></div><div className="status-board reveal"><div className="board-head"><span>BUILD STATUS</span><span>20 · 07 · 2026</span></div>{t.evidence.rows.map((row, index) => <div className={`status-row ${index === 3 ? "active" : ""}`} key={row[0]}><i className={index === 3 ? "next" : "done"}>{index === 3 ? <CircleDot size={12} /> : <Check size={12} />}</i><div><strong>{row[0]}</strong><small>{row[1]}</small></div><span>{row[2]}</span></div>)}<div className="board-footer"><span>{t.evidence.gate}</span><strong>Multi-node · GPU · LAN</strong><Gauge size={19} /></div></div></div></section>

        <InstallSection text={t.install} />

        <section className="participate container"><SectionHeading eyebrow={t.participate.eyebrow} title={t.participate.title} copy={t.participate.copy} /><div className="participate-grid">{([t.participate.contributor, t.participate.organization] as const).map((audience, index) => <article className={`participate-card ${index === 1 ? "featured" : ""} reveal`} key={audience.label}><div className="participate-icon">{index === 0 ? <Cpu size={28} /> : <Users size={28} />}</div><span>{audience.label}</span><h3>{audience.title}</h3><p>{audience.copy}</p><ul>{audience.bullets.map((bullet) => <li key={bullet}><Check size={14} />{bullet}</li>)}</ul></article>)}</div></section>

        <section className="closing" id="join"><div className="closing-grid" /><div className="container closing-inner reveal"><Brand compact homeLabel={t.home} /><span className="eyebrow"><i />{t.closing.eyebrow}</span><h2>{t.closing.title1}<br /><em>{t.closing.title2}</em></h2><p>{t.closing.copy}</p><a className="button button-primary button-large" href={`mailto:${EMAIL}?subject=${encodeURIComponent(t.closing.subject)}`}>{t.closing.button} <ArrowUpRight size={18} /></a><small>{t.closing.email} <a href={`mailto:${EMAIL}`}>{EMAIL}</a></small></div></section>
      </main>

      <footer className="footer container"><Brand homeLabel={t.home} /><p>{t.footer.tagline}</p><div><a href="#architecture">{t.footer.architecture}</a><a href="#evidence">{t.footer.status}</a><a href={`mailto:${EMAIL}`}>{t.footer.contact}</a></div><span>© 2026 mycellios</span></footer>
    </div>
  );
}

export { Landing };
