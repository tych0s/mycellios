import {
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  Boxes,
  Check,
  ChevronRight,
  CirclePlay,
  CircleDot,
  Coins,
  Cpu,
  Download,
  Gauge,
  HardDriveDownload,
  Layers3,
  MonitorDown,
  Network,
  Radio,
  Route,
  ShieldCheck,
  Sparkles,
  Users,
  Workflow,
  Zap,
} from "lucide-react";
import { lazy, Suspense, useEffect, useRef, useState, type CSSProperties, type PointerEvent } from "react";
import brandIcon from "./assets/mycellios-app-icon-v2.png";
import efficiencyCurve from "./assets/efficiency-curve-ai.webp";
import myceliumNetwork from "./assets/mycelium-network.jpg";
import { MyceliumHero } from "./MyceliumHero";

const EMAIL = "hello@mycellios.com";
const RELEASES_URL = "https://github.com/tych0s/mycellios";
const ExplainerVideo = lazy(() => import("./ExplainerVideo"));

const downloads = {
  windows: { label: "Windows", detail: "Windows 10/11 · x64", filename: "mycellios-windows-x64.exe" },
  "mac-arm64": { label: "macOS", detail: "Apple Silicon", filename: "mycellios-macos-arm64.dmg" },
  "mac-x64": { label: "macOS", detail: "Intel", filename: "mycellios-macos-x64.dmg" },
  "linux-deb": { label: "Linux", detail: "Ubuntu / Debian · x64", filename: "mycellios-linux-x64.deb" },
  "linux-rpm": { label: "Linux", detail: "Fedora / RHEL · x64", filename: "mycellios-linux-x64.rpm" },
} as const;

type DownloadKey = keyof typeof downloads;

function downloadUrl(key: DownloadKey): string {
  if (key === "windows") return "/downloads/windows";
  if (key === "mac-arm64") return "/downloads/macos-arm64";
  if (key === "mac-x64") return "/downloads/macos-x64";
  if (key === "linux-deb") return "/downloads/linux-deb";
  if (key === "linux-rpm") return "/downloads/linux-rpm";
  throw new Error(`Unsupported download target: ${String(key)}`);
}

const copy = {
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
    explainer: {
      index: "00 / START HERE",
      eyebrow: "mycellios in 20 seconds",
      title: "Several devices. One AI model.",
      copy: "Each computer contributes only the capacity you choose. mycellios connects that capacity, distributes the model, and returns one streamed answer.",
      loading: "Preparing the explanation…",
      note: "Conceptual flow · Real capacity and speed are measured in Network and Tests.",
      steps: [
        ["01", "Offer capacity", "Choose what each device can contribute."],
        ["02", "Connect nodes", "Useful machines form one coordinated network."],
        ["03", "Split the model", "Every node holds only its assigned part."],
        ["04", "Stream the answer", "One request returns one continuous response."],
      ],
    },
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
    future: {
      eyebrow: "The next layer",
      title1: "A network that",
      title2: "improves itself.",
      copy: "Three long-term systems designed to make mycellios more capable every day, route every request intelligently, and return value to the machines that make that progress possible.",
      status: "VISION · NOT LIVE YET",
      bridge: "PROVEN WORK → MEASURABLE VALUE",
      improvement: {
        index: "01",
        label: "AUTONOMOUS IMPROVEMENT",
        metric: "0.1",
        unit: "%",
        metricDetail: "OF VERIFIED USEFUL COMPUTE",
        title: "The network improves the network.",
        copy: "Reserve a strictly bounded share of useful compute for daily agents that inspect failures, run isolated experiments and propose measurable improvements.",
        steps: ["Observe real bottlenecks", "Experiment inside a sandbox", "Promote only verified gains"],
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
      principle: "Direction, not a promise. Each stage must prove security, usefulness and measurable results before it can reach the public network.",
    },
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
  footer: { tagline: "Distributed intelligence, built together.", architecture: "Architecture", status: "Status" },
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
  const hyphae = [
    { d: "M1510 180 C1320 185 1195 250 1040 312 C850 388 702 380 552 470 C360 585 205 565 -70 720", depth: 0, delay: 0 },
    { d: "M1245 232 C1210 145 1152 75 1048 -30", depth: 1, delay: .3 },
    { d: "M1158 270 C1140 390 1198 475 1110 612", depth: 1, delay: .55 },
    { d: "M1040 312 C946 211 885 126 748 38", depth: 1, delay: .75 },
    { d: "M944 352 C910 468 843 575 724 690", depth: 1, delay: .95 },
    { d: "M820 382 C742 301 664 224 526 177", depth: 1, delay: 1.1 },
    { d: "M690 404 C635 522 556 630 423 758", depth: 1, delay: 1.25 },
    { d: "M552 470 C438 405 318 342 132 371", depth: 1, delay: 1.45 },
    { d: "M403 548 C300 502 190 478 32 523", depth: 2, delay: 1.75 },
    { d: "M1210 146 C1310 96 1404 91 1510 112", depth: 2, delay: .95 },
    { d: "M1168 444 C1274 428 1373 466 1490 546", depth: 2, delay: 1.2 },
    { d: "M946 211 C1012 112 1017 40 995 -38", depth: 2, delay: 1.35 },
    { d: "M885 126 C823 72 784 24 773 -35", depth: 2, delay: 1.65 },
    { d: "M843 575 C913 652 1008 715 1146 770", depth: 2, delay: 1.65 },
    { d: "M742 301 C668 327 605 322 526 284", depth: 2, delay: 1.85 },
    { d: "M635 522 C701 602 744 698 748 805", depth: 2, delay: 1.95 },
    { d: "M438 405 C420 303 371 225 278 154", depth: 2, delay: 2.05 },
    { d: "M318 342 C244 277 159 242 36 259", depth: 2, delay: 2.2 },
    { d: "M300 502 C252 623 169 712 36 802", depth: 2, delay: 2.35 },
    { d: "M1110 612 C1195 683 1324 717 1494 700", depth: 2, delay: 1.7 },
    { d: "M1318 190 C1370 270 1422 316 1506 340", depth: 3, delay: 1.55 },
    { d: "M1164 444 C1244 506 1290 570 1304 655", depth: 3, delay: 2.05 },
    { d: "M1040 312 C1082 230 1100 167 1082 94", depth: 3, delay: 1.75 },
    { d: "M944 352 C877 280 820 242 748 232", depth: 3, delay: 2.05 },
    { d: "M843 575 C802 510 781 455 792 392", depth: 3, delay: 2.3 },
    { d: "M742 301 C704 229 651 177 578 143", depth: 3, delay: 2.15 },
    { d: "M635 522 C570 556 521 602 488 666", depth: 3, delay: 2.4 },
    { d: "M552 470 C494 500 450 554 430 622", depth: 3, delay: 2.55 },
    { d: "M438 405 C352 390 286 404 224 454", depth: 3, delay: 2.6 },
    { d: "M318 342 C269 204 198 124 82 81", depth: 3, delay: 2.75 },
    { d: "M300 502 C202 540 112 606 22 702", depth: 3, delay: 2.85 },
    { d: "M526 177 C473 106 394 62 294 42", depth: 3, delay: 2.55 },
  ] as const;
  return (
    <div className="living-background" aria-hidden="true">
      <svg viewBox="0 0 1440 900" preserveAspectRatio="xMidYMid slice">
        <defs>
          <linearGradient id="hyphaGradient" x1="1" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#e2edff" /><stop offset=".42" stopColor="#83b5ff" /><stop offset=".75" stopColor="#5c7fda" /><stop offset="1" stopColor="#445caa" /></linearGradient>
          <radialGradient id="colonyGlow"><stop offset="0" stopColor="#5c96ff" stopOpacity=".14" /><stop offset="1" stopColor="#11172f" stopOpacity="0" /></radialGradient>
        </defs>
        <g className="colony-glow">
          <ellipse cx="1040" cy="312" rx="330" ry="250" fill="url(#colonyGlow)" />
          <ellipse cx="552" cy="470" rx="285" ry="230" fill="url(#colonyGlow)" />
        </g>
        <g className="mycelium-web">
          {hyphae.map((hypha) => <path key={hypha.d} d={hypha.d} pathLength="1" className={`hypha-branch depth-${hypha.depth}`} style={{ "--grow-delay": `${hypha.delay}s` } as CSSProperties} />)}
        </g>
        <g className="hypha-crosslinks">
          <path d="M1210 146 C1112 181 1078 228 1040 312" /><path d="M944 352 C848 412 762 456 635 522" /><path d="M820 382 C702 365 628 397 552 470" /><path d="M438 405 C402 466 370 516 300 502" />
          <path d="M1318 190 C1235 314 1190 384 1164 444" /><path d="M1082 94 C958 168 875 238 820 382" /><path d="M748 232 C692 363 634 431 552 470" /><path d="M792 392 C673 408 558 388 438 405" /><path d="M488 666 C350 616 260 554 224 454" />
        </g>
        <g className="fruiting-bodies">
          <g transform="translate(1040 312)"><path d="M-8 5 C-7-10 7-10 8 5Z" /><path d="M0 5 L0 18" /></g>
          <g transform="translate(552 470) scale(.8)"><path d="M-8 5 C-7-10 7-10 8 5Z" /><path d="M0 5 L0 18" /></g>
          <g transform="translate(843 575) scale(.55)"><path d="M-8 5 C-7-10 7-10 8 5Z" /><path d="M0 5 L0 18" /></g>
        </g>
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
          <linearGradient id="networkLinkGradient" x1="0" x2="1"><stop offset="0" stopColor="#66f6d1" /><stop offset=".52" stopColor="#768cff" /><stop offset="1" stopColor="#cf6dff" /></linearGradient>
          <radialGradient id="coreGradient"><stop offset="0" stopColor="#e8fffb" /><stop offset=".28" stopColor="#65f5d0" /><stop offset=".68" stopColor="#6576ef" /><stop offset="1" stopColor="#9d52d5" /></radialGradient>
        </defs>
        <g className="cell-regions">
          <path d="M37 102 C97 25 221 22 297 75 C351 113 348 187 292 220 C216 265 106 237 48 181 C22 155 19 126 37 102Z" />
          <path d="M371 29 C462 -11 596 17 646 97 C688 165 642 248 566 271 C487 294 398 259 365 194 C333 132 325 49 371 29Z" />
          <path d="M326 323 C391 274 504 290 574 351 C643 411 613 508 523 532 C431 557 322 515 293 437 C277 394 289 351 326 323Z" />
          <path d="M31 316 C88 268 190 284 246 346 C298 404 270 501 187 531 C105 561 24 516 11 433 C4 386 8 336 31 316Z" />
        </g>
        <g className="bio-filaments">
          <path d="M30 286 C126 226 180 314 263 261 S421 167 526 250 S632 363 681 306" />
          <path d="M74 545 C143 430 257 467 319 370 S427 185 602 120" />
          <path d="M0 175 C128 205 210 139 321 210 S490 381 680 407" />
        </g>
        <g className="mesh-lines">
          {links.map(([from, to], index) => { const start = nodes[from]; const end = nodes[to]; return <line key={`${from}-${to}`} x1={start.x} y1={start.y} x2={end.x} y2={end.y} className={index % 3 === 0 ? "signal-line" : ""} />; })}
          {nodes.map((node, index) => <path key={`core-${node.label}`} d={`M${node.x} ${node.y} Q${(node.x + 340) / 2 + (index % 2 ? 18 : -18)} ${(node.y + 286) / 2 + (index % 3 - 1) * 16} 340 286`} className="core-link" />)}
        </g>
        <g className="micro-cells">
          <g transform="translate(145 300)"><circle r="11" /><circle r="2" /></g><g transform="translate(505 166)"><circle r="8" /><circle r="1.7" /></g><g transform="translate(525 438)"><circle r="12" /><circle r="2.2" /></g><g transform="translate(273 54)"><circle r="7" /><circle r="1.5" /></g>
        </g>
        <g className="core"><circle cx="340" cy="286" r="93" className="core-orbit orbit-a" /><circle cx="340" cy="286" r="68" className="core-orbit orbit-b" /><circle cx="340" cy="286" r="56" className="core-membrane" /><circle cx="340" cy="286" r="49" fill="url(#coreGradient)" /><circle cx="340" cy="286" r="26" className="core-center" /></g>
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

function ExplainerSection({ text }: { text: typeof copy.explainer }) {
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
    <section className="explainer-section" id="explainer" ref={section}>
      <div className="container">
        <div className="explainer-heading reveal">
          <span className="explainer-index">{text.index}</span>
          <div>
            <span className="eyebrow"><i />{text.eyebrow}</span>
            <h2>{text.title}</h2>
          </div>
          <p>{text.copy}</p>
        </div>

        <div className="explainer-player-shell reveal">
          {shouldLoad ? (
            <Suspense fallback={<div className="explainer-loading"><CirclePlay size={40} /><span>{text.loading}</span></div>}>
              <ExplainerVideo />
            </Suspense>
          ) : (
            <div className="explainer-loading"><CirclePlay size={40} /><span>{text.loading}</span></div>
          )}
        </div>

        <div className="explainer-steps reveal">
          {text.steps.map(([number, title, detail]) => (
            <div key={number}>
              <span>{number}</span>
              <strong>{title}</strong>
              <p>{detail}</p>
            </div>
          ))}
        </div>
        <p className="explainer-note reveal">{text.note}</p>
      </div>
    </section>
  );
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
              <linearGradient id="efficiencyLine" x1="0" x2="1"><stop offset="0" stopColor="#7189ce" /><stop offset=".55" stopColor="#5c96ff" /><stop offset="1" stopColor="#e2edff" /></linearGradient>
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

function FutureRoadmap({ text }: { text: typeof copy.future }) {
  return (
    <section className="future-roadmap-section" id="roadmap">
      <div className="future-roadmap-field" aria-hidden="true"><i /><i /><i /><i /></div>
      <div className="container">
        <div className="future-roadmap-heading reveal">
          <div>
            <span className="eyebrow"><i />{text.eyebrow}</span>
            <h2>{text.title1}<br /><em>{text.title2}</em></h2>
          </div>
          <div className="future-roadmap-intro">
            <span className="future-roadmap-status"><CircleDot size={12} />{text.status}</span>
            <p>{text.copy}</p>
          </div>
        </div>

        <div className="future-roadmap-stage">
          <article className="future-roadmap-card improvement reveal" data-index={text.improvement.index}>
            <header><span>{text.improvement.index} / {text.improvement.label}</span><Sparkles size={23} /></header>
            <div className="future-roadmap-metric"><strong>{text.improvement.metric}</strong><span><b>{text.improvement.unit}</b><small>{text.improvement.metricDetail}</small></span></div>
            <h3>{text.improvement.title}</h3>
            <p>{text.improvement.copy}</p>
            <ol>{text.improvement.steps.map((step, index) => <li key={step}><b>0{index + 1}</b><span>{step}</span></li>)}</ol>
            <footer><ShieldCheck size={15} /><span>{text.improvement.footer}</span></footer>
          </article>

          <div className="future-roadmap-bridge" aria-hidden="true"><i /><span><Zap size={15} /></span><b>{text.bridge}</b></div>

          <article className="future-roadmap-card token reveal" data-index={text.token.index}>
            <header><span>{text.token.index} / {text.token.label}</span><Coins size={23} /></header>
            <div className="future-roadmap-metric word"><strong>{text.token.metric}</strong><span><small>{text.token.metricDetail}</small></span></div>
            <h3>{text.token.title}</h3>
            <p>{text.token.copy}</p>
            <ol>{text.token.steps.map((step, index) => <li key={step}><b>0{index + 1}</b><span>{step}</span></li>)}</ol>
            <footer><ShieldCheck size={15} /><span>{text.token.footer}</span></footer>
          </article>
        </div>

        <article className="future-roadmap-router reveal" data-index={text.router.index}>
          <header><span>{text.router.index} / {text.router.label}</span><Route size={23} /></header>
          <div className="future-router-content">
            <div className="future-router-copy">
              <div className="future-roadmap-metric word"><strong>{text.router.metric}</strong><span><small>{text.router.metricDetail}</small></span></div>
              <h3>{text.router.title}</h3>
              <p>{text.router.copy}</p>
            </div>
            <div className="future-router-visual" aria-label="A unified request routed to the best available AI model">
              <div className="future-router-input"><Radio size={15} /><span>{text.router.input}</span></div>
              <i className="future-router-line incoming" />
              <div className="future-router-engine"><Route size={24} /><strong>{text.router.engine}</strong><span>{text.router.policies.map((policy) => <b key={policy}>{policy}</b>)}</span></div>
              <i className="future-router-line outgoing" />
              <div className="future-router-destinations">{text.router.destinations.map((destination, index) => <span key={destination} style={{ "--route-delay": `${index * .7}s` } as CSSProperties}><i />{destination}</span>)}</div>
            </div>
          </div>
          <ol>{text.router.steps.map((step, index) => <li key={step}><b>0{index + 1}</b><span>{step}</span></li>)}</ol>
          <footer><ShieldCheck size={15} /><span>{text.router.footer}</span></footer>
        </article>

        <div className="future-roadmap-principle reveal"><ShieldCheck size={17} /><span>{text.principle}</span></div>
      </div>
    </section>
  );
}

function InstallSection({ text }: { text: typeof copy.install }) {
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
        {macVisitor && (
          <aside className="mac-test-guide" aria-labelledby="mac-test-title">
            <div className="mac-test-guide-head">
              <span><ShieldCheck size={15} />MACOS TEST BUILD</span>
              <div>
                <h3 id="mac-test-title">Try mycellios without an Apple developer licence</h3>
                <p>The app is ad-hoc signed and package-attested. macOS still asks you to approve this test build once.</p>
              </div>
              <div className="mac-test-downloads">
                <a href={downloadUrl("mac-arm64")}><Cpu size={15} /><span><strong>Apple Silicon</strong><small>M1 · M2 · M3 · M4</small></span><Download size={14} /></a>
                <a href={downloadUrl("mac-x64")}><Cpu size={15} /><span><strong>Intel</strong><small>Intel processor</small></span><Download size={14} /></a>
              </div>
            </div>
            <ol>
              <li><b>01</b><span><strong>Install</strong><small>Open the DMG and drag mycellios to Applications.</small></span></li>
              <li><b>02</b><span><strong>Try to open it once</strong><small>macOS shows a security warning. Close that message.</small></span></li>
              <li><b>03</b><span><strong>Open Privacy &amp; Security</strong><small>Go to System Settings → Privacy &amp; Security.</small></span></li>
              <li><b>04</b><span><strong>Choose Open Anyway</strong><small>Confirm once, then open mycellios normally.</small></span></li>
            </ol>
            <footer><Check size={13} />No Terminal commands required <i /> Manual downloads remain required for test builds</footer>
          </aside>
        )}
        <p className="install-safety"><ShieldCheck size={14} />{text.safety}</p>
      </div>
    </section>
  );
}

function Landing() {
  const page = useRef<HTMLDivElement>(null);
  const t = copy;

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
      <div className="scroll-progress" aria-hidden="true" /><LivingBackdrop /><div className="ambient-pointer" />
      <header className="nav-shell"><nav className="nav container" aria-label={t.nav.aria}><Brand homeLabel={t.home} /><div className="nav-links"><a href="/network">Live network</a><a href="/mobile/">Contribute</a><a href="#architecture">{t.nav.architecture}</a><a href="#roadmap">Roadmap</a></div><div className="nav-actions"><a className="nav-blog-link" href="/blog" aria-label="Read the Mycellios blog"><BookOpen size={15} />Blog</a><a className="nav-cta" href="/join">Join now <Zap size={15} /></a></div></nav></header>

      <main>
        <section className="hero">
          <MyceliumHero />
          <div className="hero-inner container">
            <div className="hero-copy"><div className="availability"><span /><strong>EARLY NETWORK</strong><i /> {t.hero.status}</div><h1>{t.hero.line1}<br />{t.hero.line2}<br /><em>{t.hero.line3}</em></h1><p>{t.hero.copy}</p><div className="hero-actions"><a className="button button-primary" href="/join">Join the network <Zap size={17} /></a><a className="button button-secondary" href="#explainer">{t.hero.secondary} <CirclePlay size={17} /></a></div><div className="hero-readouts" aria-label="Network capabilities"><div><span>01</span><strong>POOL MEMORY</strong><small>BEYOND ONE MACHINE</small></div><div><span>02</span><strong>ROUTE SHARDS</strong><small>THROUGH THE FASTEST CELL</small></div><div><span>03</span><strong>KEEP GROWING</strong><small>AS NEW GPUS APPEAR</small></div></div><div className="hero-footnote"><ShieldCheck size={15} /><span>No account or invitation required during public testing.</span></div></div>
          </div>
        </section>

        <section className="signal-strip" aria-label={t.nav.vision}>
          <div className="signal-track">
            {[0, 1].map((copy) => (
              <div className="signal-sequence" key={copy} aria-hidden={copy === 1}>
                {t.signals.map((item) => <span key={`${copy}-${item}`}>{item}<i /></span>)}
              </div>
            ))}
          </div>
        </section>

        <ExplainerSection text={t.explainer} />

        <section className="manifesto container" id="vision"><div className="manifesto-index reveal">{t.manifesto.index}</div><div className="manifesto-copy reveal"><span>{t.manifesto.kicker}</span><h2>{t.manifesto.title}</h2></div><div className="manifesto-side reveal"><p>{t.manifesto.copy}</p><a href="#architecture">{t.manifesto.link} <ArrowUpRight size={15} /></a></div></section>

        <MyceliumStory text={t.mycelium} />

        <EfficiencyCurve text={t.acceleration} />

        <section className="architecture-section" id="architecture"><div className="container"><SectionHeading eyebrow={t.architecture.eyebrow} title={t.architecture.title} copy={t.architecture.copy} /><PipelineVisual text={t.pipeline} /><div className="principles-grid">{t.principles.map((principle, index) => { const Icon = [Layers3, Network, Workflow][index] ?? Network; return <article className="principle-card reveal" key={principle.label}><div className="card-number">0{index + 1}</div><Icon size={25} /><h3>{principle.title1}<br />{principle.title2}</h3><p>{principle.copy}</p><span>{principle.label} <ArrowUpRight size={13} /></span></article>; })}</div></div></section>

        <section className="difference container"><div className="difference-copy reveal"><span className="eyebrow"><i />{t.modes.eyebrow}</span><h2>{t.modes.title1}<br /><em>{t.modes.title2}</em></h2><p>{t.modes.copy}</p></div><div className="mode-stack"><article className="mode-card exact reveal"><div><ShieldCheck size={23} /><span>01</span></div><h3>{t.modes.exactTitle}</h3><p>{t.modes.exactCopy}</p><footer><Check size={14} /> {t.modes.exactFooter}</footer></article><article className="mode-card optin reveal"><div><Sparkles size={23} /><span>02</span></div><h3>{t.modes.optTitle}</h3><p>{t.modes.optCopy}</p><footer><Check size={14} /> {t.modes.optFooter}</footer></article></div></section>

        <section className="evidence-section" id="evidence"><div className="container evidence-grid"><div><SectionHeading eyebrow={t.evidence.eyebrow} title={t.evidence.title} copy={t.evidence.copy} /><div className="status-legend reveal"><span><i className="done" />{t.evidence.done}</span><span><i className="next" />{t.evidence.next}</span></div></div><div className="status-board reveal"><div className="board-head"><span>BUILD STATUS</span><span>20 · 07 · 2026</span></div>{t.evidence.rows.map((row, index) => <div className={`status-row ${index === 3 ? "active" : ""}`} key={row[0]}><i className={index === 3 ? "next" : "done"}>{index === 3 ? <CircleDot size={12} /> : <Check size={12} />}</i><div><strong>{row[0]}</strong><small>{row[1]}</small></div><span>{row[2]}</span></div>)}<div className="board-footer"><span>{t.evidence.gate}</span><strong>Multi-node · GPU · LAN</strong><Gauge size={19} /></div></div></div></section>

        <FutureRoadmap text={t.future} />

        <InstallSection text={t.install} />

        <section className="participate container"><SectionHeading eyebrow={t.participate.eyebrow} title={t.participate.title} copy={t.participate.copy} /><div className="participate-grid">{([t.participate.contributor, t.participate.organization] as const).map((audience, index) => <article className={`participate-card ${index === 1 ? "featured" : ""} reveal`} key={audience.label}><div className="participate-icon">{index === 0 ? <Cpu size={28} /> : <Users size={28} />}</div><span>{audience.label}</span><h3>{audience.title}</h3><p>{audience.copy}</p><ul>{audience.bullets.map((bullet) => <li key={bullet}><Check size={14} />{bullet}</li>)}</ul></article>)}</div></section>

        <section className="closing" id="join"><div className="closing-grid" /><div className="container closing-inner reveal"><Brand compact homeLabel={t.home} /><span className="eyebrow"><i />{t.closing.eyebrow}</span><h2>{t.closing.title1}<br /><em>{t.closing.title2}</em></h2><p>{t.closing.copy}</p><a className="button button-primary button-large" href="/join">Connect this device <Zap size={18} /></a><small>Public test network · no login required</small></div></section>
      </main>

      <footer className="footer container"><Brand homeLabel={t.home} /><p>{t.footer.tagline}</p><div><a href="/network">Network panel</a><a href="#roadmap">Roadmap</a><a href="/blog">Blog</a><a href="/mobile/">Mobile worker</a><a href="/downloads">Downloads</a><a href={`mailto:${EMAIL}`}>{EMAIL}</a></div><span>© 2026 mycellios</span></footer>
    </div>
  );
}

export { Landing };
