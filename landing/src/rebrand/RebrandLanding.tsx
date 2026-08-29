import {
  ArrowDown,
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  Check,
  CircleDot,
  Coins,
  Cpu,
  Download,
  GitBranch,
  HardDriveDownload,
  Menu,
  Route,
  ShieldCheck,
  Sparkles,
  Users,
  X,
  Zap,
} from "lucide-react";
import { useEffect, useRef, useState, type CSSProperties, type FormEvent } from "react";
import { applySeoMetadata } from "../seo";
import { FruitingMark } from "./FruitingMark";
import { HeroGlobe } from "./HeroGlobe";
import { HeroGroundColony } from "./HeroGroundColony";
import { HeroMushroom } from "./HeroMushroom";
import { MyceliumNetwork } from "./MyceliumNetwork";
import { LocalStrip } from "./LocalStrip";
import { ModelsSection } from "./ModelsSection";
import { PaymentsSection } from "./PaymentsSection";
import { QuestionsSection } from "./QuestionsSection";
import { ScrollStory } from "./ScrollStory";
import { SponsorFooter } from "./SponsorFooter";
import { SporeDrift } from "./SporeDrift";
import { SporeMenu } from "./SporeMenu";
import { useCountUp, useInView, useRevealOnScroll } from "./use-motion";
import "./rebrand.css";

/*
 * The landing is a scroll pitch, not a product manual.
 *
 * After the hero it runs a fixed order of sections, each answering exactly one
 * question a first-time visitor or an investor asks: how does it work, who pays
 * whom, which models can I run, do I have to join anything, where do I fit, is
 * any of this actually built, what about the parts I am sceptical of, how do I
 * start, and who is behind it. Detail that belongs to the docs, the panel or
 * the blog is linked, not restated — the earlier fourteen-section version made
 * the reader work for a story that fits in this.
 *
 * Every visual here is generated in the browser (SVG, canvas, CSS) and driven
 * by the reader's own scroll. There are no still renders standing in for the
 * system: a diagram that moves when you move explains a distributed runtime
 * better than a photograph of mycelium ever did.
 *
 * There is deliberately no per-section illustration. Two passes tried one — lit
 * 3-D bodies floating in the gutters, then flat line plates sitting in the
 * grid — and both failed for the same underlying reason: an extra drawing beside
 * copy that already has a working diagram (the rail, the table, the board) is a
 * second thing to look at, and it makes the section busier without making the
 * claim clearer. The sections illustrate themselves.
 */

const brandLogo = "/assets/logos/logo.png";
const EMAIL = "hello@mycellios.com";
const RELEASES_URL = "https://github.com/tych0s/mycellios";
const GITHUB_URL = "https://github.com/tych0s/mycellios";
const X_URL = "https://x.com/mycellios";
const TELEGRAM_URL = "https://t.me/mycellios";

/* Lucide dropped its brand marks, so the three social glyphs are inline paths.
   They are decorative: each link carries its own accessible name. */
function GithubMark() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" focusable="false">
      <path d="M12 .5C5.37.5 0 5.87 0 12.5c0 5.3 3.44 9.8 8.21 11.39.6.11.82-.26.82-.58 0-.29-.01-1.05-.02-2.06-3.34.73-4.04-1.61-4.04-1.61-.55-1.39-1.34-1.76-1.34-1.76-1.09-.75.08-.73.08-.73 1.21.09 1.84 1.24 1.84 1.24 1.07 1.83 2.81 1.3 3.5.99.11-.78.42-1.31.76-1.61-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.12-.3-.54-1.52.12-3.18 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6.01 0c2.29-1.55 3.3-1.23 3.3-1.23.66 1.66.24 2.88.12 3.18.77.84 1.23 1.91 1.23 3.22 0 4.61-2.8 5.63-5.48 5.92.43.37.81 1.1.81 2.22 0 1.6-.01 2.89-.01 3.29 0 .32.22.7.83.58A12.01 12.01 0 0 0 24 12.5C24 5.87 18.63.5 12 .5Z" />
    </svg>
  );
}

function XMark() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" focusable="false">
      <path d="M18.24 2.25h3.31l-7.23 8.26 8.5 11.24h-6.65l-5.22-6.82-5.96 6.82H1.68l7.73-8.84L1.25 2.25h6.82l4.71 6.23 5.46-6.23Zm-1.16 17.52h1.83L7.01 4.13H5.05l12.03 15.64Z" />
    </svg>
  );
}

function TelegramMark() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" focusable="false">
      <path d="M21.94 3.4a1.2 1.2 0 0 0-1.24-.2L2.72 10.4c-.86.34-.83 1.57.05 1.86l4.4 1.47 1.7 5.24c.2.62 1 .8 1.45.33l2.47-2.56 4.5 3.3c.55.4 1.34.11 1.5-.56l3.36-14.6a1.2 1.2 0 0 0-.21-1.05ZM9.35 14.3l-.53 3.53-1.19-3.66 9.1-5.98-7.38 6.11Z" />
    </svg>
  );
}

export function LandingHeader({ active }: { active?: "network" }) {
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    if (!mobileOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobileOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [mobileOpen]);

  return <header className="rb-header">
    <div className="rb-header-inner rb-shell">
      <Brand />
      <nav aria-label="Main navigation">
        <a href="/network?view=inference">Chat</a><a href="/create">Create</a><a href="/earn">Earn</a><SporeMenu /><a className={active === "network" ? "active" : undefined} href="/network" aria-current={active === "network" ? "page" : undefined}>Live network</a><a href="/blog">Blog</a>
      </nav>
      <div className="rb-header-actions rb-desktop-cta">
        <a className="rb-social" href={GITHUB_URL} target="_blank" rel="noreferrer" aria-label="mycellios on GitHub"><GithubMark /></a>
        <a className="rb-social" href={X_URL} target="_blank" rel="noreferrer" aria-label="mycellios on X"><XMark /></a>
        <a className="rb-social" href={TELEGRAM_URL} target="_blank" rel="noreferrer" aria-label="mycellios on Telegram"><TelegramMark /></a>
        <a className="rb-pill rb-pill-ghost" href="/dashboard">Login</a>
      </div>
      <button className="rb-menu" type="button" aria-label={mobileOpen ? "Close menu" : "Open menu"} aria-expanded={mobileOpen} onClick={() => setMobileOpen(!mobileOpen)}>{mobileOpen ? <X /> : <Menu />}</button>
      {mobileOpen && <>
        <div className="rb-mobile-backdrop" aria-hidden="true" onClick={() => setMobileOpen(false)} />
        <nav className="rb-mobile-nav" aria-label="Mobile navigation" onClick={() => setMobileOpen(false)}>
          <div className="rb-mobile-group"><a href="/network?view=inference">Chat</a><a href="/create">Create</a><a href="/earn">Earn</a></div>
          <div className="rb-mobile-group"><span className="rb-mobile-kicker">$ SPORE</span><div className="rb-mobile-sub"><a href="/spore">Staking</a><a href="/spore/treasury">Treasury</a><a href="/spore/data">Data</a></div></div>
          <div className="rb-mobile-group"><a className={active === "network" ? "active" : undefined} href="/network" aria-current={active === "network" ? "page" : undefined}>Live network</a><a href="/blog">Blog</a></div>
          <a className="rb-mobile-cta" href="/dashboard">Login</a>
          <div className="rb-mobile-social"><a className="rb-social" href={GITHUB_URL} target="_blank" rel="noreferrer" aria-label="mycellios on GitHub"><GithubMark /></a><a className="rb-social" href={X_URL} target="_blank" rel="noreferrer" aria-label="mycellios on X"><XMark /></a><a className="rb-social" href={TELEGRAM_URL} target="_blank" rel="noreferrer" aria-label="mycellios on Telegram"><TelegramMark /></a></div>
        </nav>
      </>}
    </div>
  </header>;
}

const doors = [
  {
    label: "Developers",
    title: "One API. Served by a network.",
    copy: "An OpenAI-compatible endpoint that runs on pooled machines instead of a data center.",
    action: { href: "/network?view=inference", text: "Try it live" },
    icon: Sparkles,
  },
  {
    label: "GPU owners",
    title: "Idle hardware. Useful work.",
    copy: "Contribute what you choose, pause whenever you want, and only accepted work counts.",
    action: { href: "/earn", text: "Start contributing" },
    icon: Cpu,
  },
  {
    label: "Open-model community",
    title: "Open models need open infrastructure.",
    copy: "The runtime, the scheduler and the evidence behind every claim are public.",
    action: { href: GITHUB_URL, text: "Read the source", external: true },
    icon: Users,
  },
] as const;

const evidenceRows = [
  { title: "Partitioned-model runtime", detail: "Real pipeline, compatible API", status: "READY", done: true },
  { title: "Heterogeneous scheduler", detail: "Memory, topology, availability", status: "READY", done: true },
  { title: "Stage recovery", detail: "Alternate route, greedy continuation", status: "LAB", done: true },
  { title: "Model larger than every node", detail: "2–4 physical machines", status: "TESTING", done: false },
] as const;

const downloads = {
  windows: { label: "Windows", detail: "Windows 10/11 · x64" },
  "mac-arm64": { label: "macOS", detail: "Apple Silicon" },
  "linux-deb": { label: "Linux", detail: "Ubuntu / Debian · x64" },
  "linux-rpm": { label: "Linux", detail: "Fedora / RHEL · x64" },
} as const;

type DownloadKey = keyof typeof downloads;

function downloadUrl(key: DownloadKey): string {
  if (key === "windows") return "/downloads/windows";
  if (key === "mac-arm64") return "/downloads/macos-arm64";
  if (key === "linux-deb") return "/downloads/linux-deb";
  if (key === "linux-rpm") return "/downloads/linux-rpm";
  throw new Error(`Unsupported download target: ${String(key)}`);
}

function Brand() {
  return <a className="rb-brand" href="/" aria-label="mycellios, home"><img src={brandLogo} alt="" /><span>mycellios</span></a>;
}

/* Downloads. The visitor's platform is detected so the primary action is a
   single button; every other build stays one click away rather than on screen. */
function GetStartedSection() {
  const [recommended, setRecommended] = useState<DownloadKey>("windows");
  const [macVisitor, setMacVisitor] = useState(false);

  useEffect(() => {
    const agent = navigator.userAgent.toLowerCase();
    const isAppleMobile = /iphone|ipad|ipod/.test(agent) || (agent.includes("mac") && navigator.maxTouchPoints > 1);
    if (agent.includes("mac") && !isAppleMobile) {
      setMacVisitor(true);
      setRecommended("mac-arm64");
    } else if (agent.includes("linux")) setRecommended("linux-deb");
    else setRecommended("windows");
  }, []);

  const selected = downloads[recommended];

  return (
    <section className="rb-get" id="install" aria-labelledby="rb-get-title">
      <div className="rb-shell rb-get-inner">
        <div className="rb-get-copy rb-reveal">
          <p className="rb-kicker rb-kicker-light"><i /><span>Join</span></p>
          <h2 id="rb-get-title">One install.<br /><em>Your machine joins the network.</em></h2>
          <p>Hardware is detected automatically. No terminal, no Docker, no configuration.</p>
          <a className="rb-get-primary" href={downloadUrl(recommended)}>
            <span className="rb-get-icon"><Download /></span>
            <span><small>Download for</small><strong>{selected.label}</strong></span>
            <ArrowDownRight />
          </a>
          <p className="rb-get-meta"><ShieldCheck />You choose what to share · pause at any time</p>
        </div>

        <div className="rb-get-side rb-reveal">
          <div className="rb-get-shelf">
            {(Object.keys(downloads) as DownloadKey[]).map((key, index) => (
              <a className={key === recommended ? "recommended" : ""} href={downloadUrl(key)} key={key} style={{ "--i": index } as CSSProperties}>
                <HardDriveDownload />
                <span><strong>{downloads[key].label}</strong><small>{downloads[key].detail}</small></span>
                {key === recommended ? <b>PICKED</b> : <ArrowDownRight />}
              </a>
            ))}
          </div>
          {macVisitor && (
            <details className="rb-mac-note">
              <summary>Opening the macOS test build</summary>
              <ol>
                <li>Open the DMG and drag mycellios to Applications.</li>
                <li>Open it once — macOS shows a warning. Close it.</li>
                <li>System Settings → Privacy &amp; Security → <strong>Open Anyway</strong>.</li>
              </ol>
            </details>
          )}
          <p className="rb-get-safety">
            Installers are built and package-verified in GitHub Actions. Code signing is rolling out; early
            builds may still trigger a publisher warning.
            <a href={RELEASES_URL} target="_blank" rel="noreferrer">Release notes <ArrowUpRight /></a>
          </p>
        </div>
      </div>
    </section>
  );
}

export function RebrandLanding() {
  const [heroPrompt, setHeroPrompt] = useState("");
  const pageRef = useRef<HTMLElement>(null);

  useEffect(() => {
    applySeoMetadata("/");
  }, []);

  useRevealOnScroll(pageRef);

  // The hero input is an entry point, not an inference client: the prompt is
  // handed to the panel chat, which owns models, auth, and streaming.
  const submitHeroPrompt = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const text = heroPrompt.trim();
    if (!text) return;
    window.sessionStorage.setItem("mycellios.hero-prompt", text);
    window.location.assign("/network?view=inference");
  };

  return (
    <main className="rb-page" id="rb-top" ref={pageRef}>
      {/* The brand is a fungal network, so the page grows one while you read.
          It is anchored to the base of the hero's mushroom and descends the
          whole document from there — roots of the organism above it, not a
          decoration that happens to be nearby. */}
      <MyceliumNetwork />

      <LandingHeader />

      <section className="rb-hero" aria-labelledby="rb-hero-title">
        <HeroGlobe />
        <HeroGroundColony />
        {/* A grounded fruiting body beside the scroll cue. Its foot meets the
            viewport edge, so the page-wide mycelium can grow from a believable
            origin instead of from artwork floating behind the headline.
            Desktop only: `wantsMushroom()` keeps the renderer and the host off
            phones, where every framing of the organism fought the copy. */}
        <HeroMushroom />
        <div className="rb-hero-inner rb-shell">
          <div className="rb-hero-copy">
            {/* Name the category immediately, then state the product's defining
                difference without repeating the story section's own headline.
                Broken across three lines rather than two: the copy column is
                50vw and the hero organism starts at 50% + 110px, so the
                two-line setting ran the headline under the mushroom. */}
            <h1 id="rb-hero-title" aria-label="The open compute layer for distributed AI.">The open compute<br />layer<br /><em>for distributed AI.</em></h1>
            {/* Kept under the lede's 470px measure so it sets as one line: the
                longer version broke with "anyway." orphaned on its own. */}
            <p className="rb-lede">Mycellios splits the model across many machines.</p>
            <form className="rb-hero-prompt" onSubmit={submitHeroPrompt}>
              <label className="rb-visually-hidden" htmlFor="rb-hero-prompt-input">Ask the network</label>
              <input id="rb-hero-prompt-input" name="prompt" type="text" autoComplete="off" placeholder="Ask the impossible…" value={heroPrompt} onChange={(event) => setHeroPrompt(event.target.value)} />
              <button type="submit" aria-label="Send this prompt to the network"><ArrowRight /></button>
            </form>
          </div>
        </div>
        {/* The hero owns the whole viewport, so nothing below it peeks in to hint
            there is more. This cue is that hint, and it jumps to the story. */}
        <a className="rb-scroll-cue" href="#how-it-works" aria-label="Scroll to the next section">
          <span>Scroll</span>
          <ArrowDown />
        </a>
        {/* Mycelium origin on phones. There the organism stands behind the
            globe — its base is mid-hero, which would start the colony in the
            middle of the copy — so the network anchors to the hero's floor
            instead and still begins right after the hero, like the desktop. */}
        <div id="rb-hero-base" aria-hidden="true" />
      </section>

      <ScrollStory />

      <PaymentsSection />

      <ModelsSection />

      {/* Two sections under one drift.

          The local strip and the doors are separate arguments — "you do not
          have to join anything", then "here is where you fit" — and they stay
          separate sections with their own backgrounds and their own rule
          between them. What they share is air: this is the one stretch of the
          page with no diagram, no plate and no body in it, and the spores are
          what fills it. A field mounted on either section alone stops dead at
          that rule, which is what the layer looked like and is the opposite of
          weather.

          So the wrapper, and only the wrapper: it establishes the containing
          block the layer sizes itself to and does nothing else. No background,
          no padding, no z-index of its own — every one of those would take
          something away from the two sections, which still paint their own. The
          one decoration on both, and the only place on the page where the
          organism appears without a body: a fourth fruiting body here would be
          the page repeating itself, and what drifts above a colony is what the
          colony released. See SporeDrift.tsx. */}
      <div className="rb-drift-band">
        {/* Half a section, on purpose: "you do not have to join anything" is one
            line of reassurance between the catalogue and the three doors, and a
            full chapter would give it more weight than it needs. */}
        <LocalStrip />

        <section className="rb-doors" id="doors" aria-labelledby="rb-doors-title">
          <div className="rb-shell">
            <div className="rb-doors-head rb-reveal">
              <p className="rb-kicker"><i /><span>Where you fit</span></p>
              <h2 id="rb-doors-title">Three doors.<br /><em>One network.</em></h2>
            </div>
            <div className="rb-doors-grid">
              {doors.map((door, index) => {
                const Icon = door.icon;
                const external = "external" in door.action && door.action.external;
                return (
                  <article className="rb-door rb-reveal" key={door.label} style={{ "--i": index } as CSSProperties}>
                    <div className="rb-door-top"><Icon /><span>{door.label}</span></div>
                    <h3>{door.title}</h3>
                    <p>{door.copy}</p>
                    <a href={door.action.href} {...(external ? { target: "_blank", rel: "noreferrer" } : {})}>
                      {door.action.text} <ArrowUpRight />
                    </a>
                  </article>
                );
              })}
            </div>
          </div>
        </section>

        {/* Last child, and that is load-bearing rather than a formatting choice.
            The layer and the two sections all paint in the same auto/0 stacking
            level, where tree order decides — mounted first, the field would go
            under both opaque section backgrounds and be invisible, which is the
            same failure as the layer opacity that used to be on it. A z-index
            here instead would put it above the content wrappers too. */}
        <SporeDrift />
      </div>

      <section className="rb-proof" id="evidence" aria-labelledby="rb-proof-title">
        <div className="rb-shell rb-proof-inner">
          <div className="rb-proof-copy rb-reveal">
            <p className="rb-kicker"><i /><span>Evidence</span></p>
            <h2 id="rb-proof-title">We are building<br /><em>in public.</em></h2>
            <p>No simulation is called a global network. Every claim passes a reproducible test before it becomes a promise.</p>
          </div>
          <div className="rb-board rb-reveal" aria-label="Build status">
            {evidenceRows.map((row, index) => (
              // `--i` staggers the row entrance; see .rb-board-row in the CSS.
              <div className={`rb-board-row ${row.done ? "" : "active"}`} key={row.title} style={{ "--i": index } as CSSProperties}>
                <i className={row.done ? "done" : "next"}>{row.done ? <Check /> : <CircleDot />}</i>
                <div><strong>{row.title}</strong><small>{row.detail}</small></div>
                <span>{row.status}</span>
              </div>
            ))}
            <div className="rb-board-foot"><span>Next gate</span><strong>Multi-node · GPU · LAN</strong></div>
          </div>
        </div>

        <div className="rb-shell rb-next-strip rb-reveal">
          <span><Route />Unified model routing</span>
          <span><Coins />Rewards for accepted work</span>
          <span><Zap />Self-improving scheduler</span>
          <b>VISION · NOT LIVE YET · no active token, no financial promise</b>
        </div>
      </section>

      {/* Doubts are answered before the install ask, not after it: a reader who
          has just been told what is still in testing has objections, and the
          page loses them if it asks for the download first. */}
      <QuestionsSection />

      <GetStartedSection />

      <section className="rb-closing" aria-labelledby="rb-closing-title">
        <div className="rb-closing-inner rb-shell rb-reveal">
          {/* Third and last appearance of the fruiting body, and the only still
              one: the globe shows the network fruiting, the backdrop grows one
              as you read, and the page ends on the finished organism. Three
              scales of the same shape from one generator — the payoff of the
              scroll, standing over the final ask. */}
          <FruitingMark />
          <Brand />
          <h2 id="rb-closing-title">Many machines.<br /><em>One model.</em></h2>
          <a className="rb-pill rb-pill-light" href="/join">Connect this device <Zap /></a>
          <small>Public test network · no login required</small>
        </div>
      </section>

      <SponsorFooter />
      <footer className="rb-footer">
        <div className="rb-footer-inner rb-shell">
          <div className="rb-footer-brand"><Brand /><p>Distributed intelligence built from the capacity already around us.</p></div>
          <nav aria-label="Product"><span>Product</span><a href="/network?view=inference">Chat</a><a href="/create">Create</a><a href="/earn">Earn</a><a href="/spore">$ SPORE</a></nav>
          <nav aria-label="Network"><span>Network</span><a href="#how-it-works">How it works</a><a href="/network">Network</a><a href="/downloads">Downloads</a><a href="/mobile/">Browser worker</a></nav>
          <nav aria-label="Resources"><span>Resources</span><a href="/docs">Documentation</a><a href="/blog">Blog</a><a href={GITHUB_URL}><GitBranch /> GitHub</a><a href={`mailto:${EMAIL}`}>Contact</a></nav>
          <div className="rb-footer-bottom"><span>© 2026 mycellios</span><a href="#rb-top">Back to top <ArrowDown /></a></div>
        </div>
      </footer>
    </main>
  );
}
